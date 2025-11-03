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


def evaluate(model_ckpt: str, datamodule, subsets: list[str], device: str = "cuda", model_name: str = "FastGAN"):
 
    seed_everything(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if model_name == "FastGAN":
        print(f"Loading model: {model_ckpt}")
        model = FastGANModule.load_from_checkpoint(model_ckpt, map_location=device)
    elif model_name == "VAE":
        model = VAE.load_from_checkpoint(model_ckpt, map_location=device)
    elif model_name == "HSIGan":
        model = HSIGanModule.load_from_checkpoint(model_ckpt, map_location=device)


    model.eval()
    model.to(device)
    # EMA do gerador
    backup_params = copy_G_params(model.netG)
    load_params(model.netG, model.avg_param_G)


    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    sam = SpectralAngleMapper().to(device)
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
        print(f"Avaliando subset: {subset}")

        if subset == "train":
            loader = datamodule.train_dataloader()
        elif subset == "val":
            loader = datamodule.val_dataloader()
        elif subset == "test":
            loader = datamodule.test_dataloader()
        elif subset == 'all':
            loader = datamodule.all_dataloader()

        fid.reset()
        sam.reset()
        ssim.reset()
        psnr.reset()

        sam_sum, ssim_sum, psnr_sum, count = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for real_img, _ in tqdm(loader):
                real_img = real_img.to(device)
                batch_size = real_img.size(0)

                noise = torch.randn(batch_size, model.hparams.nz, device=device)
                fake_imgs = model(noise)[0].float()
                fake_norm = (fake_imgs + 1) / 2
                real_norm = (real_img + 1) / 2
                fake_norm = fake_norm.clamp(0, 1)
                real_norm = real_norm.clamp(0, 1)

                fid.update(fake_norm, real=False)
                fid.update(real_norm, real=True)

                sam_val = sam(fake_norm, real_norm)
                ssim_val = ssim(fake_norm, real_norm)
                psnr_val = psnr(fake_norm, real_norm)

                sam_sum += sam_val.item()
                ssim_sum += ssim_val.item()
                psnr_sum += psnr_val.item()
                count += 1

        mean_sam = sam_sum / count
        mean_ssim = ssim_sum / count
        mean_psnr = psnr_sum / count
        fid_val = fid.compute().item()
        wandb.log({
            f"{subset}/SAM": mean_sam,
            f"{subset}/SSIM": mean_ssim,
            f"{subset}/PSNR": mean_psnr,
            f"{subset}/FID": fid_val,
        })

        print(f" {subset} — SAM: {mean_sam:.4f} | SSIM: {mean_ssim:.4f} | PSNR: {mean_psnr:.2f} | FID: {fid_val:.2f}")


    load_params(model.netG, backup_params)
    wandb.finish()



