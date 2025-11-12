import torch
import numpy as np
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    SpectralAngleMapper,
    PeakSignalNoiseRatio
)
from torchmetrics.image.fid import FrechetInceptionDistance
from src.metrics.inception import InceptionV3Wrapper


# ===============================================================
# Helper: dynamically import a class from a string path
# ===============================================================
def load_class(class_path: str):
    """Dynamically import class from src package."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(f"src.{module_name}")
    return getattr(module, class_name)


# ===============================================================
# Main evaluation
# ===============================================================
def evaluate(checkpoint_path, config_model_path, config_dataset_path, subsets):
    """General evaluation for FastGAN (noise→HSI) or CycleGAN (RGB→HSI)."""

    # ---------------------------------------------------------------
    # Environment setup
    # ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # Load dataset config and instantiate DataModule
    # ---------------------------------------------------------------
    cfg_data = OmegaConf.load(config_dataset_path)
    DataModuleClass = load_class(cfg_data.data.class_path)
    dm_args = OmegaConf.to_container(cfg_data.data.init_args, resolve=True)
    datamodule = DataModuleClass(**dm_args)

    datamodule.prepare_data()
    datamodule.setup("fit")  # ensure train/val/test datasets exist

    # sanity check for internal DMs
    if hasattr(datamodule, "rgb_dm") and datamodule.rgb_dm is not None:
        datamodule.rgb_dm.setup("fit")
    if hasattr(datamodule, "hsi_dm") and datamodule.hsi_dm is not None:
        datamodule.hsi_dm.setup("fit")

    print("✅ Data modules prepared and set up successfully.")

    # ---------------------------------------------------------------
    # Load model config and checkpoint
    # ---------------------------------------------------------------
    cfg_model = OmegaConf.load(config_model_path)
    ModelClass = load_class(cfg_model.model.class_path)
    model_args = OmegaConf.to_container(cfg_model.model.init_args, resolve=True)

    model = ModelClass.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,
        **model_args
    )
    model.eval().to(device)
    model_name = ModelClass.__name__.lower()
    print(f"✅ Loaded model: {ModelClass.__name__}")

    # ---------------------------------------------------------------
    # Initialize metrics
    # ---------------------------------------------------------------
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    sam = SpectralAngleMapper().to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    inception_model = InceptionV3Wrapper(normalize_input=False, in_chans=16)
    fid = FrechetInceptionDistance(
        inception_model,
        input_img_size=(16, 256, 256)
    ).to(device)
    fid.eval()

    # ---------------------------------------------------------------
    # Model-specific loader logic
    # ---------------------------------------------------------------
    if "cyclegan" in model_name:
        loaders = {"rgb": datamodule.rgb_dm.val_dataloader()}
        hsi_iter = iter(datamodule.hsi_dm.val_dataloader())
        print("\nEvaluating CycleGAN (RGB → HSI only)\n")

    elif "fastgan" in model_name:
        loaders = {"hsi": datamodule.val_dataloader()}
        print("\nEvaluating FastGAN (noise → HSI)\n")

    else:
        raise ValueError(f"Unsupported model type: {ModelClass.__name__}")

    # ---------------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------------
    for subset in subsets:
        print(f"Evaluating subset: {subset}")
        fid.reset(); sam.reset(); ssim.reset(); psnr.reset()
        sam_sum = ssim_sum = psnr_sum = 0.0
        count = 0

        for domain_name, loader in loaders.items():
            print(f"  Domain: {domain_name}")
            for batch_idx, batch in enumerate(tqdm(loader)):
                if batch_idx >= 8:  # limit iterations for faster eval
                    break

                # -------------------------------------------------------
                # FASTGAN: noise → HSI
                # -------------------------------------------------------
                if "fastgan" in model_name:
                    imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    real_hsi = imgs.to(device)
                    noise = torch.randn(real_hsi.size(0), model.hparams.nz, device=device)
                    fake_hsi = model(noise)[0].float()

                # -------------------------------------------------------
                # CYCLEGAN: RGB → HSI
                # -------------------------------------------------------
                elif "cyclegan" in model_name:
                    # get RGB image batch
                    if isinstance(batch, (list, tuple)):
                        rgb_img = batch[0]
                    elif isinstance(batch, dict):
                        rgb_img = batch.get("rgb") or next(iter(batch.values()))
                    else:
                        rgb_img = batch
                    rgb_img = rgb_img.to(device)

                    with torch.no_grad():
                        fake_hsi = model.G_AB(rgb_img)

                    # get real HSI batch
                    try:
                        real_hsi_batch = next(hsi_iter)
                    except StopIteration:
                        hsi_iter = iter(datamodule.hsi_dm.val_dataloader())
                        real_hsi_batch = next(hsi_iter)

                    real_hsi = (real_hsi_batch[0] if isinstance(real_hsi_batch, (list, tuple))
                                else real_hsi_batch).to(device)

                # -------------------------------------------------------
                # Normalize to [0,1]
                # -------------------------------------------------------
                fake_norm = (fake_hsi + 1) / 2
                real_norm = (real_hsi + 1) / 2
                fake_norm = fake_norm.clamp(0, 1)
                real_norm = real_norm.clamp(0, 1)

                # -------------------------------------------------------
                # Compute metrics
                # -------------------------------------------------------
                fid.update(fake_norm, real=False)
                fid.update(real_norm, real=True)

                sam_val = sam(fake_norm, real_norm)
                ssim_val = ssim(fake_norm, real_norm)
                psnr_val = psnr(fake_norm, real_norm)

                sam_sum += sam_val.item()
                ssim_sum += ssim_val.item()
                psnr_sum += psnr_val.item()
                count += 1

                # -------------------------------------------------------
                # Save visualization once
                # -------------------------------------------------------
                if batch_idx == 1:
                    idx = np.random.randint(0, fake_norm.size(0))
                    mean_reflectance = fake_norm[idx].mean(dim=0).cpu().numpy()
                    plt.imshow(mean_reflectance, cmap='gray')
                    plt.title("Mean Reflectance of Generated HSI")
                    plt.axis("off")
                    plt.colorbar()
                    plt.savefig("fake_example_mean_reflectance.png")
                    plt.close()

        # -----------------------------------------------------------
        # Print mean metrics
        # -----------------------------------------------------------
        mean_sam = sam_sum / max(count, 1)
        mean_ssim = ssim_sum / max(count, 1)
        mean_psnr = psnr_sum / max(count, 1)
        fid_val = fid.compute().item()

        print(f"\n{subset} — SAM: {mean_sam:.4f} | SSIM: {mean_ssim:.4f} | "
              f"PSNR: {mean_psnr:.2f} | FID: {fid_val:.2f}\n")

    print("✅ Evaluation complete.")



# Example usage
if __name__ == "__main__":
    config_dataset_path = "/mnt/datahdd/kris_volume/dgm-2025.2/projects/hyperskin/configs/data/joint_rgb_hsi_dermoscopy.yaml"
    config_model_path = "/mnt/datahdd/kris_volume/dgm-2025.2/projects/hyperskin/configs/model/hsi_cycle_gan.yaml"
    checkpoint_path = "/mnt/datahdd/kris_volume/dgm-2025.2/projects/hyperskin/logs/hypersynth/9p1whuza/checkpoints/last.ckpt"

    subsets = ["val"]
    evaluate(checkpoint_path, config_model_path, config_dataset_path, subsets)
