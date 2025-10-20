from git import Optional
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import random

import pytorch_lightning as pl

import warnings
from skimage import filters, morphology
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

        # Validation metrics
        self.sam = SpectralAngleMapper()
        self.rase = RelativeAverageSpectralError()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.tv = TotalVariation()

        self.inception_model = InceptionV3Wrapper(
            normalize_input=False, in_chans=self.hparams.nc
        )
        self.inception_model.eval()
        self.fid = FrechetInceptionDistance(
            self.inception_model,
            input_img_size=(self.hparams.nc, self.hparams.im_size, self.hparams.im_size),
        )
        self.fid.eval()

        self.real_spectra = None
        self.fake_spectra = None
        self.lesion_class_name = None

    def setup(self, stage: str) -> None:
        datamodule = self.trainer.datamodule

        if stage == 'fit':
            train_dataset = datamodule.data_train
        elif stage == 'validate':
            train_dataset = datamodule.data_val
        elif stage == 'test' or stage == 'predict':
            train_dataset = datamodule.data_test
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if isinstance(train_dataset, torch.utils.data.ConcatDataset):
            base_dataset = train_dataset.datasets[0]
        else:
            base_dataset = train_dataset

        if isinstance(base_dataset, torch.utils.data.Subset):
            base_dataset = base_dataset.dataset

        train_indices = datamodule.train_indices
        train_labels = np.array([base_dataset.labels[i] for i in train_indices])
        unique_labels = np.unique(train_labels)

        if len(unique_labels) == 1:
            lesion_class_int = unique_labels[0]
            inverse_labels_map = {v: k for k, v in base_dataset.labels_map.items()}
            self.lesion_class_name = inverse_labels_map[lesion_class_int]

        # Initialize spectra storage
        self.real_spectra = {"normal_skin": [], self.lesion_class_name: []} if self.lesion_class_name else None
        self.fake_spectra = {"normal_skin": [], self.lesion_class_name: []} if self.lesion_class_name else None

        # get global min/max from datamodule if available
        if hasattr(datamodule, "global_min") and hasattr(datamodule, "global_max") and \
            (datamodule.global_min is not None) and (datamodule.global_max is not None):
            self.hparams.pred_global_min = datamodule.global_min
            self.hparams.pred_global_max = datamodule.global_max

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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = len(batch)
        backup_para = copy_G_params(self.netG)
        load_params(self.netG, self.avg_param_G)
        self.netG.eval()

        if ((self.global_step + 1) // batch_size) % self.hparams.val_check_interval == 0:
            self._run_validation()

        if ((self.global_step + 1) // batch_size) % self.hparams.log_images_on_step_n == 0:
            sample = self(self.fixed_noise)[0].add(1).mul(0.5)
            if self.hparams.nc > 3:
                sample = sample.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            sample = sample.clamp(0, 1)

            # Create and log sample grid
            sample_grid = torchvision.utils.make_grid(sample, nrow=4).detach().cpu()
            self.logger.experiment.log({
                "generated_samples": wandb.Image(sample_grid)
            })

        load_params(self.netG, backup_para)
        self.netG.train()

    def _run_validation(self):
        """Run validation metrics on multiple batches from val_dataloader
        using rolling sums for efficiency."""
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
                self.fid.update(fake, real=False)
                self.fid.update(real_image, real=True)

                # Rolling sum updates
                eps = 1e-8

                # Clamp inputs to avoid negative or zero spectral values
                fake_norm_clamped = fake_norm.clamp(eps, 1.0)
                real_norm_clamped = real_norm.clamp(eps, 1.0)

                # Compute SAM safely (avoid NaNs)
                try:
                    sam_val = self.sam(fake_norm_clamped, real_norm_clamped)
                    if torch.isnan(sam_val):
                        sam_val = torch.tensor(0.0, device=self.device)
                except Exception:
                    sam_val = torch.tensor(0.0, device=self.device)

                # Use nan_to_num fallback for safety in all metrics
                rase_val = torch.nan_to_num(
                    self.rase(fake_norm_clamped, real_norm_clamped), nan=0.0
                )
                ssim_val = torch.nan_to_num(
                    self.ssim(fake_norm_clamped, real_norm_clamped), nan=0.0
                )
                tv_val = torch.nan_to_num(self.tv(fake_norm_clamped), nan=0.0)

                sam_sum += sam_val.item()
                rase_sum += rase_val.item()
                ssim_sum += ssim_val.item()
                tv_sum += tv_val.item()
                count += 1

                # --- Compute spectra for this batch ---
                if self.lesion_class_name is not None:
                    self.compute_spectra_statistics(real_norm_clamped, fake_norm_clamped)

            # Compute overall metrics
            mean_sam = sam_sum / max(count, 1)
            mean_rase = rase_sum / max(count, 1)
            mean_ssim = ssim_sum / max(count, 1)
            mean_tv = tv_sum / max(count, 1)

            fid = self.fid.compute()

            self.log_dict(
                {
                    "val/SAM": mean_sam,
                    "val/RASE": mean_rase,
                    "val/SSIM": mean_ssim,
                    "val/TV": mean_tv,
                    "val/FID": fid,
                },
                prog_bar=True,
                sync_dist=True,
            )

            # --- Plot mean spectra ---
            if self.lesion_class_name is not None and (self.real_spectra or self.fake_spectra):
                self._plot_mean_spectra()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def compute_spectra_statistics(self, real_norm, fake_norm):
        for img, spectra_dict in [
            (real_norm, self.real_spectra),
            (fake_norm, self.fake_spectra),
        ]:
            for b in range(img.size(0)):
                # Convert to numpy: (C, H, W) -> (H, W, C)
                image_np = img[b].cpu().numpy().transpose(1, 2, 0)

                # Compute mean across spectral bands for thresholding
                mean_image = image_np.mean(axis=-1)

                # Otsu thresholding to separate lesion from background
                try:
                    otsu_thresh = filters.threshold_otsu(mean_image)
                    binary_mask = mean_image < (otsu_thresh * 1)

                    # Extract lesion spectrum (pixels inside mask)
                    if np.any(binary_mask):
                        spectrum = image_np[binary_mask].mean(axis=0)
                        spectra_dict[self.lesion_class_name].append(spectrum)

                    # Extract normal skin spectrum (pixels outside mask)
                    normal_skin_mask = ~binary_mask
                    if np.any(normal_skin_mask):
                        normal_spectrum = image_np[normal_skin_mask].mean(axis=0)
                        spectra_dict["normal_skin"].append(normal_spectrum)
                except Exception:
                    # Skip if Otsu fails (e.g., uniform image)
                    continue


    def _plot_mean_spectra(self):
        """
        Plot mean spectra comparing real vs synthetic data.
        Creates separate plots for normal_skin and lesion class.
        """
        import matplotlib.pyplot as plt

        labels = ["normal_skin", self.lesion_class_name]

        # Compute statistics
        real_stats = {}
        fake_stats = {}

        for label_name in labels:
            if self.real_spectra.get(label_name):
                arr = np.array(self.real_spectra[label_name])
                real_stats[label_name] = {
                    "mean": np.mean(arr, axis=0),
                    "std": np.std(arr, axis=0),
                }

            if self.fake_spectra.get(label_name):
                arr = np.array(self.fake_spectra[label_name])
                fake_stats[label_name] = {
                    "mean": np.mean(arr, axis=0),
                    "std": np.std(arr, axis=0),
                }

        # Get number of bands
        ref_stats = real_stats or fake_stats
        ref_label = next(
            (lbl for lbl in labels if ref_stats.get(lbl) is not None), None
        )
        if ref_label is None:
            print("No stats available for plotting mean spectra")
            return

        n_bands = len(ref_stats[ref_label]["mean"])
        bands = np.arange(1, n_bands + 1)

        # Create subplots
        fig, axes = plt.subplots(1, len(labels), figsize=(15, 5))
        if len(labels) == 1:
            axes = [axes]

        for ax, lbl in zip(axes, labels):
            rs = real_stats.get(lbl)
            fs = fake_stats.get(lbl)

            if rs is None and fs is None:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {lbl}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Plot real data
            if rs is not None:
                ax.plot(
                    bands,
                    rs["mean"],
                    linestyle="-",
                    linewidth=2.5,
                    color="C0",
                    label="Real",
                )
                ax.fill_between(
                    bands,
                    rs["mean"] - rs["std"],
                    rs["mean"] + rs["std"],
                    color="C0",
                    alpha=0.15,
                )

            # Plot synthetic data
            if fs is not None:
                ax.plot(
                    bands,
                    fs["mean"],
                    linestyle="--",
                    linewidth=2.0,
                    color="C3",
                    label="Synthetic",
                )
                ax.fill_between(
                    bands,
                    fs["mean"] - fs["std"],
                    fs["mean"] + fs["std"],
                    color="C3",
                    alpha=0.15,
                )

            ax.set_title(f"{lbl.replace('_', ' ').title()}")
            ax.set_xlabel("Spectral Band")
            ax.set_ylabel("Reflectance")
            ax.legend()
            ax.grid(True, alpha=0.25)

        plt.suptitle(
            "Mean Spectra Comparison: Real vs Synthetic",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()

        # Log to wandb
        self.logger.experiment.log({"val/mean_spectra": wandb.Image(fig)})
        plt.close(fig)

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
            noise = torch.randn(self.hparams.pred_num_samples, nz, device=device)
            fake_imgs = self(noise)
            fake = fake_imgs[0].float().cpu()

            assert torch.min(fake) >= -1.0 and torch.max(fake) <= 1.0, \
                "Generated images are out of expected range [-1, 1]"

            for i in tqdm(range(self.hparams.pred_num_samples), desc="Generating samples"):
                fake_img = fake[i : i + 1, :, :, :].to(device)

                if self.hparams.pred_hyperspectral:
                    # ---------------------
                    # Hyperspectral (denormalized cube)
                    # ---------------------
                    fake_denorm = fake_img.add(1).div(2)

                    if gmin is not None and gmax is not None:
                        gmin_arr = torch.tensor(gmin, device=device) if isinstance(gmin, list) else np.array(gmin)
                        gmax_arr = torch.tensor(gmax, device=device) if isinstance(gmax, list) else np.array(gmax)

                        # Scalar or per-band
                        if gmin_arr.size == 1:
                            fake_denorm = fake_denorm * (gmax_arr - gmin_arr) + gmin_arr
                        else:
                            gmin_t = torch.tensor(gmin_arr).view(1, -1, 1, 1)
                            gmax_t = torch.tensor(gmax_arr).view(1, -1, 1, 1)
                            fake_denorm = fake_denorm * (gmax_t - gmin_t) + gmin_t

                    fake_np = fake_denorm.cpu().squeeze().numpy()
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
