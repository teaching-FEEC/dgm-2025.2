import argparse
import torch
import numpy as np
import wandb
from tqdm import tqdm
from pytorch_lightning import seed_everything
from src.metrics.inception import InceptionV3Wrapper
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    SpectralAngleMapper,
    PeakSignalNoiseRatio
)
from torchmetrics.image.fid import FrechetInceptionDistance
from src.modules.generative.gan.fastgan.fastgan import FastGANModule  
from src.modules.generative.gan.fastgan.operation import load_params, copy_G_params
from src.modules.vae_module import VAE
from src.modules.generative.gan.wgan import WGANModule
from src.data_modules.hsi_dermoscopy import HSIDermoscopyDataModule
from omegaconf import OmegaConf, ListConfig
import matplotlib.pyplot as plt

def evaluate(checkpoint_path, config_model_path, config_dataset_path, subsets: list[str],
             device: str = "cuda", model_name: str = "FastGAN",
             num_noise_repeats: int = 1):
    """Evaluate a trained generator using multiple noise repetitions per real batch."""

    cfg = OmegaConf.load(config_dataset_path)
    dm_args = OmegaConf.to_container(cfg.data.init_args, resolve=True)

    print(f"Global max: {type(dm_args['global_max'])}")
    print(f"Global min: {type(dm_args['global_min'])}")
    datamodule = HSIDermoscopyDataModule(**dm_args)
    datamodule.prepare_data()
    datamodule.setup(None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model checkpoint ---
    if model_name == "FastGAN":
        cfg_model = OmegaConf.load(config_model_path)
        model_args = OmegaConf.to_container(cfg_model.model.init_args, resolve=True)
        model = FastGANModule.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            strict=False,
            **model_args
        )

    seed_everything(42)
    model.eval()
    model.to(device)

    # --- EMA of generator ---
    backup_params = copy_G_params(model.netG)
    load_params(model.netG, model.avg_param_G)

    # --- Metrics ---
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    sam  = SpectralAngleMapper().to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    inception_model = InceptionV3Wrapper(
        normalize_input=False, in_chans=model.hparams.nc
    )
    inception_model.eval()

    fid = FrechetInceptionDistance(
        inception_model,
        input_img_size=(model.hparams.nc, model.hparams.im_size, model.hparams.im_size),
    ).to(device)
    fid.eval()

    for subset in subsets:
        print(f"\nAvaliando subset: {subset}")

        if subset == "train":
            loader = datamodule.train_dataloader()
        elif subset == "val":
            loader = datamodule.val_dataloader()
        elif subset == "test":
            loader = datamodule.test_dataloader()
        elif subset == "all":
            loader = datamodule.all_dataloader()
        else:
            raise ValueError(f"Subset '{subset}' não reconhecido")

        fid.reset(); sam.reset(); ssim.reset(); psnr.reset()
        sam_sum = ssim_sum = psnr_sum = 0.0
        count = 0

        with torch.no_grad():
            for batch_idx, (real_img, _) in enumerate(tqdm(loader)):
                if batch_idx >= 8:
                    break

                real_img = real_img.to(device)
                batch_size = real_img.size(0)

                # Repeat this real batch with different noise samples
                # for rep in range(num_noise_repeats):
                noise = torch.randn(batch_size, model.hparams.nz, device=device)
                fake_imgs = model(noise)[0].float()

                # Normalize [-1,1] → [0,1]
                fake_norm = (fake_imgs + 1) / 2
                real_norm = (real_img + 1) / 2
                fake_norm = fake_norm.clamp(0, 1)
                real_norm = real_norm.clamp(0, 1)
                if batch_idx == 1 :
                    idx = np.random.randint(0, batch_size)
                    fake_example = real_norm[idx].cpu()
                    mean_reflectance = fake_example.mean(dim=0).numpy() 
                    print("Exemplo de imagem fake:")
                    print("shape:", fake_imgs[0].shape)
                    plt.figure(figsize=(5, 5))
                    plt.imshow(mean_reflectance, cmap='gray')
                    plt.title("Mean Reflectance of Generated HSI Image")
                    plt.colorbar()
                    plt.axis("off")
                    plt.savefig("fake_example_mean_reflectance.png")
                    plt.show()
                    
                # --- Update metrics ---
                fid.update(fake_norm, real=False)
                fid.update(real_norm, real=True)

                sam_val  = sam(fake_norm,  real_norm)
                ssim_val = ssim(fake_norm, real_norm)
                psnr_val = psnr(fake_norm, real_norm)

                sam_sum  += sam_val.item()
                ssim_sum += ssim_val.item()
                psnr_sum += psnr_val.item()
                count += 1

        # --- Compute means and print ---
        print("count:", count)
        mean_sam  = sam_sum  / count
        mean_ssim = ssim_sum / count
        mean_psnr = psnr_sum / count
        fid_val   = fid.compute().item()

        print(f" {subset} — SAM: {mean_sam:.4f} | SSIM: {mean_ssim:.4f} | "
              f"PSNR: {mean_psnr:.2f} | FID: {fid_val:.2f}")

    load_params(model.netG, backup_params)



if __name__ == "__main__":
    print("Iniciando avaliação...")
    config_dataset_path = "/mnt/datahdd/kris_volume/dgm-2025.2/projects/hyperskin/configs/data/hsi_dermoscopy_croppedv2.yaml"
    # config_dataset_path = "/mnt/datahdd/kris_volume/dgm-2025.2/projects/hyperskin/configs/data/hsi_dermoscopy_synth_cropped.yaml"
    config_model_path = "/mnt/datahdd/kris_volume/dgm-2025.2/projects/hyperskin/configs/model/hsi_fastgan.yaml"
    checkpoint_path = "/mnt/datahdd/kris_volume/dgm-2025.2/projects/hyperskin/logs/j7od0bbk_fastgan_melanoma_step=0-val_MIFID=114.7889.ckpt"

    subsets = ["all"]
    evaluate(checkpoint_path, config_model_path, config_dataset_path,  subsets)