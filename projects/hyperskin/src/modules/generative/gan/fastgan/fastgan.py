from typing import Any, Self
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

from src.data_modules.joint_rgb_hsi_dermoscopy import JointRGBHSIDataModule
from src.losses.lpips import PerceptualLoss
from src.metrics.inception import InceptionV3Wrapper
from src.models.fastgan.fastgan import weights_init
from src.modules.generative.gan.fastgan.operation import copy_G_params, load_params
from src.models.fastgan.fastgan import Generator, Discriminator

# Import SPADE versions
from src.models.fastgan.spade_fastgan import GeneratorSPADE, DiscriminatorSPADE
from src.transforms.diffaug import DiffAugment
import os
import torchvision.utils as vutils
import wandb
from torchmetrics.image import (
    SpectralAngleMapper,
    RelativeAverageSpectralError,
    StructuralSimilarityIndexMeasure,
    TotalVariation,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from scipy.io import savemat

from src.utils.tags_and_run_name import add_tags_and_run_name_to_logger

warnings.filterwarnings("ignore")


policy = "color,translation"


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
    def __init__(
        self,
        im_size: int = 256,
        nc: int = 3,
        ndf: int = 64,
        ngf: int = 64,
        nz: int = 256,
        nlr: float = 0.0002,
        nbeta1: float = 0.5,
        nbeta2: float = 0.999,
        log_images_on_step_n: int = 1,
        val_check_interval: int = 500,
        num_val_batches: int = 8,
        pred_output_dir: str = "generated_samples",
        pred_num_samples: int = 100,
        pred_hyperspectral: bool = True,
        pred_global_min: Optional[float | list[float]] = None,
        pred_global_max: Optional[float | list[float]] = None,
        # SPADE configuration
        use_spade: bool = False,
        spade_conditioning: str = "rgb_mask",  # "rgb_mask" or "rgb_image"
        log_reconstructions: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Validate SPADE configuration
        if self.hparams.use_spade:
            assert self.hparams.spade_conditioning in ["rgb_mask", "rgb_image"], (
                f"Invalid spade_conditioning: {self.hparams.spade_conditioning}"
            )

        # --- model setup ---
        if self.hparams.use_spade:
            # Determine label_nc based on conditioning type
            label_nc = 1 if self.hparams.spade_conditioning == "rgb_mask" else 3

            self.netG = GeneratorSPADE(ngf=ngf, nz=nz, im_size=im_size, nc=nc, label_nc=label_nc)
            self.netD = DiscriminatorSPADE(ndf=ndf, im_size=im_size, nc=nc, label_nc=label_nc)
        else:
            self.netG = Generator(ngf=ngf, nz=nz, im_size=im_size, nc=nc)
            self.netD = Discriminator(ndf=ndf, im_size=im_size, nc=nc)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # EMA parameters
        self.avg_param_G = copy_G_params(self.netG)

        # Perceptual & Inception
        self.percept = PerceptualLoss(model="net-lin", net="vgg", use_gpu=True, in_chans=nc)

        self.register_buffer("fixed_noise", torch.FloatTensor(8, self.hparams.nz).normal_(0, 1))

        # Store fixed conditioning for validation visualization
        if self.hparams.use_spade:
            self.fixed_conditioning = None

        # Validation metrics
        self.sam = SpectralAngleMapper()
        self.rase = RelativeAverageSpectralError()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.tv = TotalVariation()

        self.inception_model = InceptionV3Wrapper(normalize_input=False, in_chans=self.hparams.nc)
        self.inception_model.eval()
        self.fid = FrechetInceptionDistance(
            self.inception_model,
            input_img_size=(self.hparams.nc, self.hparams.im_size, self.hparams.im_size),
        )
        self.fid.eval()

        self.real_spectra = None
        self.fake_spectra = None
        self.lesion_class_name = None

    def _get_tags_and_run_name(self):
        """Automatically derive tags and a run name from FastGANModule hyperparameters."""
        hparams = getattr(self, "hparams", None)
        if hparams is None:
            return

        tags = []
        run_name = "fastgan_"

        if hparams.use_spade:
            tags.append("spade")
            run_name += "spade_"
            tags.append(f"cond_{hparams.spade_conditioning}")
            run_name += f"{hparams.spade_conditioning}_"

        # Core configuration flags
        tags.append(f"imsize_{hparams.im_size}")
        tags.append(f"nz_{hparams.nz}")
        run_name += f"{hparams.im_size}px_"

        # Model architecture features
        tags.append(f"ndf_{hparams.ndf}")
        tags.append(f"ngf_{hparams.ngf}")

        # Clean trailing underscore
        run_name = run_name.rstrip("_")

        return tags, run_name

    def setup(self, stage: str) -> None:
        add_tags_and_run_name_to_logger(self)
        datamodule = self.trainer.datamodule

        # Validate datamodule structure if using SPADE
        if self.hparams.use_spade:
            if not isinstance(datamodule, JointRGBHSIDataModule):
                raise ValueError(
                    "When use_spade=True, datamodule must be JointRGBHSIDataModule to provide RGB conditioning"
                )

            # Verify that datamodule returns dict batches
            test_loader = datamodule.train_dataloader()
            if not isinstance(test_loader, dict):
                raise ValueError("When use_spade=True, datamodule must return dict batches with 'hsi' and 'rgb' keys")
            if "rgb" not in test_loader or "hsi" not in test_loader:
                raise ValueError("Datamodule dict must contain both 'hsi' and 'rgb' keys")

        # Get HSI datamodule for metadata
        if isinstance(datamodule, JointRGBHSIDataModule):
            hsi_dm = datamodule.hsi_dm
        else:
            hsi_dm = datamodule

        if stage == "fit":
            train_dataset = hsi_dm.data_train
        elif stage == "validate":
            train_dataset = hsi_dm.data_val
        elif stage == "test" or stage == "predict":
            train_dataset = hsi_dm.data_test
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if isinstance(train_dataset, torch.utils.data.ConcatDataset):
            base_dataset = train_dataset.datasets[0]
        else:
            base_dataset = train_dataset

        if isinstance(base_dataset, torch.utils.data.Subset):
            base_dataset = base_dataset.dataset

        train_indices = hsi_dm.train_indices
        train_labels = np.array([base_dataset.labels[i] for i in train_indices])
        unique_labels = np.unique(train_labels)

        if len(unique_labels) == 1:
            lesion_class_int = unique_labels[0]
            inverse_labels_map = {v: k for k, v in base_dataset.labels_map.items()}
            self.lesion_class_name = inverse_labels_map[lesion_class_int]

        # Initialize spectra storage
        self.real_spectra = {"normal_skin": [], self.lesion_class_name: []} if self.lesion_class_name else None
        self.fake_spectra = {"normal_skin": [], self.lesion_class_name: []} if self.lesion_class_name else None

        # Get global min/max from datamodule if available
        if (
            hasattr(hsi_dm, "global_min")
            and hasattr(hsi_dm, "global_max")
            and (hsi_dm.global_min is not None)
            and (hsi_dm.global_max is not None)
        ):
            self.hparams.pred_global_min = hsi_dm.global_min
            self.hparams.pred_global_max = hsi_dm.global_max

    def forward(self, z, seg=None):
        """
        Args:
            z: noise vector
            seg: conditioning for SPADE (optional)
        """
        if self.hparams.use_spade:
            return self.netG(z, seg)
        return self.netG(z)

    def on_train_start(self):
        """Properly initialize EMA params on correct device."""
        self.avg_param_G = [p.data.clone().detach().to(self.device) for p in self.netG.parameters()]

    def get_conditioning(self, batch):
        """Extract conditioning from batch based on configuration"""
        if not self.hparams.use_spade:
            return None

        if self.hparams.spade_conditioning == "rgb_mask":
            _, rgb_mask, _ = self.process_batch(batch, "rgb")
            rgb_mask = rgb_mask.to(self.device)
            if rgb_mask.ndim == 3: # (B, H, W)
                rgb_mask = rgb_mask.unsqueeze(1)  # (B, 1, H, W)
            return rgb_mask.float()
        elif self.hparams.spade_conditioning == "rgb_image":
            rgb_image, _, _ = self.process_batch(batch, "rgb")
            rgb_image = rgb_image.to(self.device)
            return rgb_image
        else:
            raise ValueError(f"Invalid conditioning: {self.hparams.spade_conditioning}")

    def train_d_step(self, real_image, fake_images, seg, label="real"):
        """Discriminator step with optional SPADE conditioning"""
        if label == "real":
            part = random.randint(0, 3)

            if self.hparams.use_spade:
                pred, [rec_all, rec_small, rec_part] = self.netD(real_image, label, seg=seg, part=part)
            else:
                pred, [rec_all, rec_small, rec_part] = self.netD(real_image, label, part=part)

            rand_weight = torch.rand_like(pred)
            err = (
                F.relu(rand_weight * 0.2 + 0.8 - pred).mean()
                + self.percept(rec_all, F.interpolate(real_image, rec_all.shape[2])).sum()
                + self.percept(rec_small, F.interpolate(real_image, rec_small.shape[2])).sum()
                + self.percept(rec_part, F.interpolate(crop_image_by_part(real_image, part), rec_part.shape[2])).sum()
            )
            return err, pred.mean(), rec_all, rec_small, rec_part
        else:
            if self.hparams.use_spade:
                pred = self.netD(fake_images, label, seg=seg)
            else:
                pred = self.netD(fake_images, label)

            rand_weight = torch.rand_like(pred)
            err = F.relu(rand_weight * 0.2 + 0.8 + pred).mean()
            return err, pred.mean()

    def process_batch(self, batch, key: Optional[str] = None):
        if not isinstance(batch, dict) and key is not None:
            raise ValueError("Batch is not a dict, but key was provided.")

        if isinstance(batch, dict) and key is not None:
            image, mask, label = batch[key]
            return image, mask, label
        else:
            real_image, real_mask, label = batch
            return real_image, real_mask, label

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            real_image, real_mask, label = self.process_batch(batch, "hsi")
        elif len(batch) == 3:
            real_image, real_mask, label = batch
        elif len(batch) == 2:
            real_image, label = batch
        else:
            raise ValueError("Batch format not recognized.")

        # Get conditioning if using SPADE
        seg = self.get_conditioning(batch) if self.hparams.use_spade else None

        batch_size = real_image.size(0)
        noise = torch.randn(batch_size, self.hparams.nz, device=self.device, dtype=torch.float32)

        opt_g, opt_d = self.optimizers()

        # Generate with conditioning
        fake_images = self(noise, seg)

        fake_images = [fi.float() for fi in fake_images]
        real_image = DiffAugment(real_image, policy=policy)
        fake_images_aug = [DiffAugment(fake, policy=policy) for fake in fake_images]

        # Apply augmentation to conditioning as well
        seg_aug = DiffAugment(seg, policy=policy) if seg is not None else None

        # --------- Train D ---------
        opt_d.zero_grad(set_to_none=True)
        self.netD.zero_grad(set_to_none=True)

        err_dr, pred_real, rec_all, rec_small, rec_part = self.train_d_step(
            real_image, fake_images_aug, seg_aug, "real"
        )
        self.manual_backward(err_dr)

        err_df, pred_fake = self.train_d_step(real_image, [fi.detach() for fi in fake_images_aug], seg_aug, "fake")
        self.manual_backward(err_df)

        opt_d.step()

        # --------- Train G ---------
        opt_g.zero_grad(set_to_none=True)
        self.netG.zero_grad(set_to_none=True)

        if self.hparams.use_spade:
            pred_g = self.netD(fake_images_aug, "fake", seg=seg_aug)
        else:
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
        current_step = self.global_step // 2

        backup_para = copy_G_params(self.netG)
        load_params(self.netG, self.avg_param_G)
        self.netG.eval()

        if current_step % self.hparams.val_check_interval == 0:
            self._run_validation()

        if current_step % self.hparams.log_images_on_step_n == 0:
            # Store fixed conditioning on first log
            if self.hparams.use_spade and self.fixed_conditioning is None:
                seg = self.get_conditioning(batch)
                self.fixed_conditioning = seg[:8].clone().detach()

            # Generate with fixed conditioning
            fixed_seg = self.fixed_conditioning if self.hparams.use_spade else None
            sample = self(self.fixed_noise, fixed_seg)[0].add(1).mul(0.5)

            # Convert to 3 channel view if HSI > 3 channels
            if self.hparams.nc > 3:
                sample = sample.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            sample = sample.clamp(0, 1)

            # --- NEW: also visualize the conditioning input ---
            if fixed_seg is not None:
                # Denormalize SPADE conditioning (assumed same [-1,1] normalization)
                cond_vis = fixed_seg.add(1).mul(0.5).clamp(0, 1)

                # If conditioning is grayscale (e.g., mask), expand to RGB for better viewing
                if cond_vis.size(1) == 1:
                    cond_vis = cond_vis.repeat(1, 3, 1, 1)
                cond_vis = cond_vis.clamp(0, 1)

                # Concatenate conditioning + generated horizontally
                vis_combined = torch.cat([cond_vis, sample], dim=0)
            else:
                vis_combined = sample

            sample_grid = torchvision.utils.make_grid(vis_combined, nrow=4
                                                      ).detach().cpu()

            if hasattr(self.logger, "experiment") and self.logger.experiment is not None:
                self.logger.experiment.log({"generated_samples": wandb.Image(sample_grid)})

        load_params(self.netG, backup_para)
        self.netG.train()

    def _iterate_val_loaders(self, val_loader):
        """Handle both dict of dataloaders or single dataloader cases."""
        if isinstance(val_loader, dict):
            iterators = {k: iter(v) for k, v in val_loader.items()}
            while True:
                batch_dict = {}
                exhausted = False
                for name, it in iterators.items():
                    try:
                        batch_dict[name] = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                if exhausted:
                    break
                yield batch_dict
        else:
            yield from val_loader

    def _run_validation(self):
        """Run validation metrics on multiple batches"""
        self.fid.reset()
        self.sam.reset()
        self.rase.reset()
        self.ssim.reset()
        self.tv.reset()

        val_loader = self.trainer.datamodule.train_dataloader()

        sam_sum = 0.0
        rase_sum = 0.0
        ssim_sum = 0.0
        tv_sum = 0.0
        count = 0

        with torch.no_grad():
            for i, batch in enumerate(self._iterate_val_loaders(val_loader)):
                if i >= self.hparams.num_val_batches:
                    break

                if isinstance(batch, dict):
                    real_image, real_mask, label = self.process_batch(batch, "hsi")
                    seg = self.get_conditioning(batch)
                else:
                    if len(batch) == 3:
                        real_image, real_mask, label = batch
                    elif len(batch) == 2:
                        real_image, label = batch
                    else:
                        raise ValueError("Batch format not recognized.")
                    seg = None

                real_image = real_image.to(self.device, non_blocking=True)
                batch_size = real_image.size(0)
                noise = torch.randn(batch_size, self.hparams.nz, device=self.device)

                fake_images = self(noise, seg)
                fake = fake_images[0].float()

                fake_norm = (fake + 1) / 2
                real_norm = (real_image + 1) / 2
                fake_norm = fake_norm.clamp(0, 1)
                real_norm = real_norm.clamp(0, 1)

                self.fid.update(fake, real=False)
                self.fid.update(real_image, real=True)

                eps = 1e-8
                fake_norm_clamped = fake_norm.clamp(eps, 1.0)
                real_norm_clamped = real_norm.clamp(eps, 1.0)

                try:
                    sam_val = self.sam(fake_norm_clamped, real_norm_clamped)
                    if torch.isnan(sam_val):
                        sam_val = torch.tensor(0.0, device=self.device)
                except Exception:
                    sam_val = torch.tensor(0.0, device=self.device)

                rase_val = torch.nan_to_num(self.rase(fake_norm_clamped, real_norm_clamped), nan=0.0)
                ssim_val = torch.nan_to_num(self.ssim(fake_norm_clamped, real_norm_clamped), nan=0.0)
                tv_val = torch.nan_to_num(self.tv(fake_norm_clamped), nan=0.0)

                sam_sum += sam_val.item()
                rase_sum += rase_val.item()
                ssim_sum += ssim_val.item()
                tv_sum += tv_val.item()
                count += 1

                if self.lesion_class_name is not None:
                    self.compute_spectra_statistics(real_norm_clamped, fake_norm_clamped)

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
                image_np = img[b].cpu().numpy().transpose(1, 2, 0)
                mean_image = image_np.mean(axis=-1)

                try:
                    otsu_thresh = filters.threshold_otsu(mean_image)
                    binary_mask = mean_image < (otsu_thresh * 1)

                    if np.any(binary_mask):
                        spectrum = image_np[binary_mask].mean(axis=0)
                        spectra_dict[self.lesion_class_name].append(spectrum)

                    normal_skin_mask = ~binary_mask
                    if np.any(normal_skin_mask):
                        normal_spectrum = image_np[normal_skin_mask].mean(axis=0)
                        spectra_dict["normal_skin"].append(normal_spectrum)
                except Exception:
                    continue

    def _plot_mean_spectra(self):
        """Plot mean spectra comparing real vs synthetic data."""
        import matplotlib.pyplot as plt

        labels = ["normal_skin", self.lesion_class_name]

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

        ref_stats = real_stats or fake_stats
        ref_label = next((lbl for lbl in labels if ref_stats.get(lbl) is not None), None)
        if ref_label is None:
            print("No stats available for plotting mean spectra")
            return

        n_bands = len(ref_stats[ref_label]["mean"])
        bands = np.arange(1, n_bands + 1)

        fig, axes = plt.subplots(1, len(labels), figsize=(15, 5))
        if len(labels) == 1:
            axes = [axes]

        for ax, lbl in zip(axes, labels):
            rs = real_stats.get(lbl)
            fs = fake_stats.get(lbl)

            if rs is None and fs is None:
                ax.text(0.5, 0.5, f"No data for {lbl}", ha="center", va="center", transform=ax.transAxes)
                continue

            if rs is not None:
                ax.plot(bands, rs["mean"], linestyle="-", linewidth=2.5, color="C0", label="Real")
                ax.fill_between(bands, rs["mean"] - rs["std"], rs["mean"] + rs["std"], color="C0", alpha=0.15)

            if fs is not None:
                ax.plot(bands, fs["mean"], linestyle="--", linewidth=2.0, color="C3", label="Synthetic")
                ax.fill_between(bands, fs["mean"] - fs["std"], fs["mean"] + fs["std"], color="C3", alpha=0.15)

            ax.set_title(f"{lbl.replace('_', ' ').title()}")
            ax.set_xlabel("Spectral Band")
            ax.set_ylabel("Reflectance")
            ax.legend()
            ax.grid(True, alpha=0.25)

        plt.suptitle("Mean Spectra Comparison: Real vs Synthetic", fontsize=14, y=1.02)
        plt.tight_layout()

        self.logger.experiment.log({"val/mean_spectra": wandb.Image(fig)})
        plt.close(fig)

    def predict_step(self, batch, batch_idx):
        """Generate and save new samples."""
        os.makedirs(self.hparams.pred_output_dir, exist_ok=True)
        self.netG.eval()

        backup_para = copy_G_params(self.netG)
        load_params(self.netG, self.avg_param_G)

        device = self.device
        nz = self.hparams.nz

        gmin = gmax = None
        if hasattr(self.hparams, "pred_global_min"):
            gmin = getattr(self.hparams, "pred_global_min", None)
            gmax = getattr(self.hparams, "pred_global_max", None)

        with torch.no_grad():
            noise = torch.randn(self.hparams.pred_num_samples, nz, device=device)

            # For SPADE, use conditioning from batch
            seg = None
            if self.hparams.use_spade:
                seg = self.get_conditioning(batch)
                # Repeat/expand to match number of samples
                if seg.size(0) < self.hparams.pred_num_samples:
                    repeats = (self.hparams.pred_num_samples + seg.size(0) - 1) // seg.size(0)
                    seg = seg.repeat(repeats, 1, 1, 1)[: self.hparams.pred_num_samples]
            fake_imgs = self(noise, seg)
            fake = fake_imgs[0].float().cpu()

            assert torch.min(fake) >= -1.0 and torch.max(fake) <= 1.0, (
                "Generated images are out of expected range [-1, 1]"
            )

            for i in tqdm(range(self.hparams.pred_num_samples), desc="Generating samples"):
                fake_img = fake[i : i + 1, :, :, :].to(device)

                if self.hparams.pred_hyperspectral:
                    fake_denorm = fake_img.add(1).div(2)

                    if gmin is not None and gmax is not None:
                        gmin_arr = torch.tensor(gmin, device=device) if isinstance(gmin, list) else np.array(gmin)
                        gmax_arr = torch.tensor(gmax, device=device) if isinstance(gmax, list) else np.array(gmax)

                        if gmin_arr.size == 1:
                            fake_denorm = fake_denorm * (gmax_arr - gmin_arr) + gmin_arr
                        else:
                            gmin_t = torch.tensor(gmin_arr).view(1, -1, 1, 1)
                            gmax_t = torch.tensor(gmax_arr).view(1, -1, 1, 1)
                            fake_denorm = fake_denorm * (gmax_t - gmin_t) + gmin_t

                    fake_np = fake_denorm.cpu().squeeze().numpy()
                    fake_np = np.transpose(fake_np, (1, 2, 0))
                    mat_path = os.path.join(self.hparams.pred_output_dir, f"sample_{i:04d}.mat")
                    savemat(mat_path, {"cube": fake_np})
                else:
                    fake_rgb = (fake + 1) / 2
                    mean_band = fake_rgb.mean(dim=1, keepdim=True)
                    rgb = mean_band.repeat(1, 3, 1, 1).clamp(0, 1)
                    rgb_uint8 = (rgb * 255).byte()

                    save_path = os.path.join(self.hparams.pred_output_dir, f"sample_{i:04d}.png")
                    torchvision.utils.save_image(rgb_uint8 / 255.0, save_path)

        load_params(self.netG, backup_para)
        self.netG.train()

    def configure_optimizers(self):
        opt_g = optim.Adam(
            self.netG.parameters(), lr=self.hparams.nlr, betas=(self.hparams.nbeta1, self.hparams.nbeta2)
        )
        opt_d = optim.Adam(
            self.netD.parameters(), lr=self.hparams.nlr, betas=(self.hparams.nbeta1, self.hparams.nbeta2)
        )
        return [opt_g, opt_d]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["avg_param_G"] = [p.clone().cpu() for p in self.avg_param_G]

    # ...existing code...
    def on_load_checkpoint(self, checkpoint):
        """
        Robust checkpoint loader:
        - restore avg_param_G if present
        - attempt to load state_dict non-strictly, removing keys that don't match this model
          (e.g. metric/net additions like 'mifid.*' that were removed from the codebase)
        """
        # restore EMA params if present
        if "avg_param_G" in checkpoint:
            self.avg_param_G = [p.to(self.device) for p in checkpoint["avg_param_G"]]
        else:
            print("Warning: avg_param_G not found in checkpoint.")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        **kwargs,
    ):
        try:
            model = super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Attempting to load with strict=False")
            model = super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=False,
                **kwargs,
            )

        return model

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        try:
            super().load_state_dict(state_dict, strict)
        except Exception as e:
            print(f"Error loading state_dict: {e}")
            print("Attempting to load with strict=False")
            super().load_state_dict(state_dict, strict=False)

if __name__ == "__main__":
    # Simple test to instantiate the module
    model = FastGANModule(
        im_size=64,
        nc=10,
        ndf=32,
        ngf=32,
        nz=128,
        use_spade=True,
        spade_conditioning="rgb_image",
    )
    model.load_from_checkpoint
