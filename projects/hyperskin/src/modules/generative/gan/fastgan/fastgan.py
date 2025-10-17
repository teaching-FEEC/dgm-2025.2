from git import Optional
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import random

import pytorch_lightning as pl

import warnings

import torchvision
from tqdm import tqdm

from src.losses.lpips import PerceptualLoss
from src.metrics.inception import InceptionV3Wrapper
from src.models.fastgan.fastgan import weights_init
from src.modules.generative.gan.fastgan.operation import copy_G_params, load_params
from src.models.fastgan.fastgan import Generator, Discriminator
from src.transforms.diffaug import DiffAugment
import os
import torchvision.utils as vutils
import wandb
from torchmetrics.image import (
    SpectralAngleMapper,
    RelativeAverageSpectralError,
    StructuralSimilarityIndexMeasure,
    TotalVariation
)
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from scipy.io import savemat

warnings.filterwarnings('ignore')


policy = 'color,translation'


def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]

# -------------------
# GAN Module
# -------------------
class FastGANModule(pl.LightningModule):
    def __init__(self,
                 im_size: int = 256,
                 nc: int = 3,
                 ndf: int = 64,
                 ngf: int = 64,
                 nz: int = 256,
                 nlr: float = 0.0002,
                 nbeta1: float = "0.5",
                 nbeta2: float = 0.999,
                 log_images_on_step_n: int = 1,
                 log_reconstructions: bool = False,
                 val_check_interval: int = 500,
                 num_val_batches: int = 8,
                 pred_output_dir: str = "generated_samples",
                 pred_num_samples: int = 100,
                 pred_hyperspectral: bool = True,
                 pred_global_min: Optional[float | list[float]] = None,
                 pred_global_max: Optional[float | list[float]] = None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # manual optimization like original

        # --- model setup ---

        self.netG = Generator(ngf=ngf, nz=nz, im_size=im_size, nc=nc)
        self.netG.apply(weights_init)

        self.netD = Discriminator(ndf=ndf, im_size=im_size, nc=nc)
        self.netD.apply(weights_init)

        # EMA parameters
        self.avg_param_G = copy_G_params(self.netG)

        # Perceptual & Inception
        self.percept = PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, in_chans=nc)

        self.register_buffer("fixed_noise", torch.FloatTensor(8, self.hparams.nz).normal_(0, 1))
        self.last_real = None
        self.last_rec = None

        # Validation metrics
        self.sam = SpectralAngleMapper()
        self.rase = RelativeAverageSpectralError()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.tv = TotalVariation()

        self.inception_model = InceptionV3Wrapper(
            normalize_input=False, in_chans=self.hparams.nc
        )
        self.inception_model.eval()
        self.mifid = MemorizationInformedFrechetInceptionDistance(self.inception_model)
        self.mifid.eval()
        self.fid = FrechetInceptionDistance(
            self.inception_model,
            input_img_size=(self.hparams.nc, self.hparams.im_size, self.hparams.im_size),
        )
        self.fid.eval()

    def forward(self, z):
        return self.netG(z)

    def on_train_start(self):
        """Properly initialize EMA params on correct device."""
        # Deep copy all parameters to correct device
        self.avg_param_G = [
            p.data.clone().detach().to(self.device)
            for p in self.netG.parameters()
        ]

    def train_d_step(self, real_image, fake_images, label="real"):
        """Discriminator step identical to original script"""
        if label == "real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = self.netD(real_image, label, part=part)
            rand_weight = torch.rand_like(pred)
            err = (
                F.relu(rand_weight * 0.2 + 0.8 - pred).mean()
                + self.percept(rec_all, F.interpolate(real_image, rec_all.shape[2])).sum()
                + self.percept(rec_small, F.interpolate(real_image, rec_small.shape[2])).sum()
                + self.percept(
                    rec_part, F.interpolate(crop_image_by_part(real_image, part), rec_part.shape[2])
                ).sum()
            )
            return err, pred.mean(), rec_all, rec_small, rec_part
        else:
            pred = self.netD(fake_images, label)
            rand_weight = torch.rand_like(pred)
            err = F.relu(rand_weight * 0.2 + 0.8 + pred).mean()
            return err, pred.mean()

    def training_step(self, batch, batch_idx):
        real_image, label = batch
        batch_size = real_image.size(0)
        noise = torch.randn(
            batch_size, self.hparams.nz,
            device=self.device,
            dtype=torch.float32
        )

        opt_g, opt_d = self.optimizers()

        fake_images = self(noise)

        fake_images = [fi.float() for fi in fake_images]
        real_image = DiffAugment(real_image, policy=policy)
        fake_images_aug = [DiffAugment(fake, policy=policy) for fake in fake_images]

        # --------- Train D ---------
        opt_d.zero_grad(set_to_none=True)
        self.netD.zero_grad(set_to_none=True)

        err_dr, pred_real, rec_all, rec_small, rec_part = self.train_d_step(
            real_image, fake_images_aug, "real"
        )
        self.manual_backward(err_dr)

        err_df, pred_fake = self.train_d_step(
            real_image, [fi.detach() for fi in fake_images_aug], "fake"
        )
        self.manual_backward(err_df)

        opt_d.step()

        # --------- Train G ---------
        opt_g.zero_grad(set_to_none=True)
        self.netG.zero_grad(set_to_none=True)

        pred_g = self.netD(fake_images_aug, "fake")
        err_g = -pred_g.mean()

        self.manual_backward(err_g)
        opt_g.step()

        # --------- EMA update after G step ---------
        for p, avg_p in zip(self.netG.parameters(), self.avg_param_G):
            avg_p.mul_(0.999).add_(p.data, alpha=0.001)

        # --------- Log losses ---------
        d_loss_total = err_dr + err_df
        self.log("train/d_loss_real", err_dr, on_step=True, on_epoch=False)
        self.log("train/d_loss_fake", err_df, on_step=True, on_epoch=False)
        self.log("train/d_loss_total", d_loss_total, on_step=True, on_epoch=False)
        self.log("train/g_loss", err_g, on_step=True, on_epoch=False)

        # Store some visuals
        if ((self.global_step + 1) // batch_size) % self.hparams.log_images_on_step_n == 0:
            num_images = min(8, batch_size)
            self.last_real = real_image[:num_images].detach().cpu()
            self.last_rec = (rec_all[:num_images].detach().cpu(), rec_small[:num_images].detach().cpu(),
                             rec_part[:num_images].detach().cpu())

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = len(batch)

        if ((self.global_step + 1) // batch_size) % self.hparams.val_check_interval == 0:
            self._run_validation()

        if ((self.global_step + 1) // batch_size) % self.hparams.log_images_on_step_n == 0:
            backup_para = copy_G_params(self.netG)
            load_params(self.netG, self.avg_param_G)
            self.netG.eval()

            sample = self(self.fixed_noise)[0].add(1).mul(0.5)
            if self.hparams.nc > 3:
                sample = sample.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            sample = sample.clamp(0, 1)

            # Create and log sample grid
            sample_grid = torchvision.utils.make_grid(sample, nrow=4).detach().cpu()
            self.logger.experiment.log({
                "generated_samples": wandb.Image(sample_grid)
            })

            if self.last_real is not None and self.last_rec is not None and self.hparam.log_reconstructions:
                rec_all, rec_small, rec_part = self.last_rec
                rec = torch.cat(
                    [F.interpolate(self.last_real, 128), rec_all, rec_small, rec_part]
                ).add(1).mul(0.5)
                if self.hparams.nc > 3:
                    rec = rec.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                rec = rec.clamp(0, 1)

                # Create and log reconstruction grid
                rec_grid = torchvision.utils.make_grid(rec, nrow=4).detach().cpu()
                self.logger.experiment.log({
                    "reconstructions": wandb.Image(rec_grid)
                })

                self.last_real = None
                self.last_rec = None

            load_params(self.netG, backup_para)
            self.netG.train()

    def _run_validation(self):
        """Run validation metrics on multiple batches from val_dataloader
        using rolling sums for efficiency."""
        self.mifid.reset()
        self.fid.reset()
        self.sam.reset()
        self.rase.reset()
        self.ssim.reset()
        self.tv.reset()

        val_loader = self.trainer.datamodule.train_dataloader()

        # rolling sums for basic metrics
        sam_sum = 0.0
        rase_sum = 0.0
        ssim_sum = 0.0
        tv_sum = 0.0
        count = 0

        with torch.no_grad():
            for i, (real_image, _) in enumerate(val_loader):
                if i >= self.hparams.num_val_batches:
                    break

                real_image = real_image.to(self.device, non_blocking=True)
                batch_size = real_image.size(0)
                noise = torch.randn(batch_size, self.hparams.nz, device=self.device)

                fake_images = self(noise)
                fake = fake_images[0].float()

                # Normalize [-1,1] → [0,1]
                fake_norm = (fake + 1) / 2
                real_norm = (real_image + 1) / 2
                fake_norm = fake_norm.clamp(0, 1)
                real_norm = real_norm.clamp(0, 1)

                # Update metrics that need all pairs (cumulative, no reduction)
                self.mifid.update(fake, real=False)
                self.mifid.update(real_image, real=True)
                self.fid.update(fake, real=False)
                self.fid.update(real_image, real=True)

                # Rolling sum updates
                eps = 1e-8

                # Clamp inputs to avoid negative or zero spectral values
                fake_norm = fake_norm.clamp(eps, 1.0)
                real_norm = real_norm.clamp(eps, 1.0)

                # Compute SAM safely (avoid NaNs)
                try:
                    sam_val = self.sam(fake_norm, real_norm)
                    if torch.isnan(sam_val):
                        sam_val = torch.tensor(0.0, device=self.device)
                except Exception:
                    sam_val = torch.tensor(0.0, device=self.device)

                # Use nan_to_num fallback for safety in all metrics
                rase_val = torch.nan_to_num(self.rase(fake_norm, real_norm), nan=0.0)
                ssim_val = torch.nan_to_num(self.ssim(fake_norm, real_norm), nan=0.0)
                tv_val = torch.nan_to_num(self.tv(fake_norm), nan=0.0)

                sam_sum += sam_val.item()
                rase_sum += rase_val.item()
                ssim_sum += ssim_val.item()
                tv_sum += tv_val.item()
                count += 1

            # Compute overall metrics
            mean_sam = sam_sum / max(count, 1)
            mean_rase = rase_sum / max(count, 1)
            mean_ssim = ssim_sum / max(count, 1)
            mean_tv = tv_sum / max(count, 1)

            mifid = self.mifid.compute()
            fid = self.fid.compute()

            self.log_dict(
                {
                    "val/SAM": mean_sam,
                    "val/RASE": mean_rase,
                    "val/SSIM": mean_ssim,
                    "val/TV": mean_tv,
                    "val/MIFID": mifid,
                    "val/FID": fid,
                },
                prog_bar=True,
                sync_dist=True,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_step(
        self,
        batch,
        batch_idx
    ):
        """
        Generate and save new samples.
        Uses datamodule.global_min/max if provided,
        else falls back to self.hparams.global_min/max if available.
        """

        os.makedirs(self.hparams.pred_output_dir, exist_ok=True)
        self.netG.eval()

        # Load EMA parameters temporarily
        backup_para = copy_G_params(self.netG)
        load_params(self.netG, self.avg_param_G)

        device = self.device
        nz = self.hparams.nz

        # --- Get normalization limits ---
        gmin = gmax = None

        if hasattr(self.hparams, "pred_global_min"):
            gmin = getattr(self.hparams, "pred_global_min", None)
            gmax = getattr(self.hparams, "pred_global_max", None)

        with torch.no_grad():
            for i in tqdm(range(self.hparams.pred_num_samples), desc="Generating samples"):
                noise = torch.randn(1, nz, device=device)
                fake_imgs = self(noise)
                fake = fake_imgs[0].float().cpu()

                if self.hparams.pred_hyperspectral:
                    # ---------------------
                    # Hyperspectral (denormalized cube)
                    # ---------------------
                    # fake_denorm = (fake + 1) / 2  # Convert [-1, 1] → [0, 1]
                    fake_denorm = fake.clamp(-1, 1).add(1).div(2)  # Robust conversion [-1, 1] → [0, 1]

                    if gmin is not None and gmax is not None:
                        gmin_arr = np.array(gmin)
                        gmax_arr = np.array(gmax)

                        # Scalar or per-band
                        if gmin_arr.size == 1:
                            fake_denorm = fake_denorm * (gmax_arr - gmin_arr) + gmin_arr
                        else:
                            gmin_t = torch.tensor(gmin_arr).view(1, -1, 1, 1)
                            gmax_t = torch.tensor(gmax_arr).view(1, -1, 1, 1)
                            fake_denorm = fake_denorm * (gmax_t - gmin_t) + gmin_t

                    fake_np = fake_denorm.squeeze().numpy()
                    # reshape to (H, W, C)
                    fake_np = np.transpose(fake_np, (1, 2, 0))
                    mat_path = os.path.join(self.hparams.pred_output_dir, f"sample_{i:04d}.mat")
                    savemat(mat_path, {"cube": fake_np})
                else:
                    # ---------------------
                    # False RGB (mean spectra)
                    # ---------------------
                    fake_rgb = (fake + 1) / 2
                    mean_band = fake_rgb.mean(dim=1, keepdim=True)
                    rgb = mean_band.repeat(1, 3, 1, 1).clamp(0, 1)
                    rgb_uint8 = (rgb * 255).byte()

                    save_path = os.path.join(self.hparams.pred_output_dir, f"sample_{i:04d}.png")
                    torchvision.utils.save_image(rgb_uint8 / 255.0, save_path)

        # Restore generator after prediction
        load_params(self.netG, backup_para)
        self.netG.train()

    def configure_optimizers(self):
        opt_g = optim.Adam(self.netG.parameters(),
                           lr=self.hparams.nlr,
                           betas=(self.hparams.nbeta1, self.hparams.nbeta2))
        opt_d = optim.Adam(self.netD.parameters(),
                           lr=self.hparams.nlr,
                           betas=(self.hparams.nbeta1, self.hparams.nbeta2))
        return [opt_g, opt_d]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["avg_param_G"] = [p.clone().cpu() for p in self.avg_param_G]

    def on_load_checkpoint(self, checkpoint):
        if "avg_param_G" in checkpoint:
            self.avg_param_G = [p.to(self.device) for p in checkpoint["avg_param_G"]]
        else:
            print("Warning: avg_param_G not found in checkpoint.")
