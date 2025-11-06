from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_with_logits
from torch.optim import Adam
import numpy as np
import torchvision
import wandb
import os
from tqdm import tqdm
from scipy.io import savemat
import warnings

from torchmetrics.image import (
    SpectralAngleMapper,
    RelativeAverageSpectralError,
    StructuralSimilarityIndexMeasure,
    TotalVariation,
)
from torchmetrics.image.fid import FrechetInceptionDistance

from src.losses.lpips import PerceptualLoss
from src.metrics.inception import InceptionV3Wrapper
from skimage import filters

from src.utils.tags_and_run_name import add_tags_and_run_name_to_logger

warnings.filterwarnings("ignore")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels, kernel_size),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels, kernel_size),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 9,
    ):
        super().__init__()
        self.model = nn.Sequential(
            self._initial_block(in_channels=in_channels, out_channels=64),
            *self._downsampling_blocks(in_channels=64, num_blocks=2),
            *self._residual_blocks(in_channels=256, num_blocks=num_res_blocks),
            *self._upsampling_blocks(in_channels=256, num_blocks=2),
            self._output_block(in_channels=64, out_channels=out_channels),
        )

    def _initial_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def _downsampling_blocks(
        self,
        in_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            blocks.append(nn.InstanceNorm2d(in_channels * 2))
            blocks.append(nn.LeakyReLU(0.2))
            in_channels *= 2
        return blocks

    def _residual_blocks(self, in_channels: int, num_blocks: int):
        return [ResidualBlock(in_channels) for _ in range(num_blocks)]

    def _upsampling_blocks(
        self,
        in_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.ConvTranspose2d(
                    in_channels,
                    in_channels // 2,
                    kernel_size,
                    stride,
                    padding,
                    output_padding=1,
                )
            )
            blocks.append(nn.InstanceNorm2d(in_channels // 2))
            blocks.append(nn.LeakyReLU(0.2))
            in_channels //= 2
        return blocks

    def _output_block(
        self,
        in_channels: int,
        out_channels: int,
    ):
        return nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7),
            # nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            self._discriminator_block(in_channels, 64, stride=2),
            self._discriminator_block(64, 128, stride=2),
            self._discriminator_block(128, 256, stride=2),
            self._discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def _discriminator_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 1,
        padding: int = 1,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


def copy_cyclegan_params(G_AB, G_BA):
    """Copy parameters from both generators for EMA"""
    return [
        p.data.clone().detach()
        for p in list(G_AB.parameters()) + list(G_BA.parameters())
    ]


def load_cyclegan_params(G_AB, G_BA, avg_params):
    """Load EMA parameters to both generators"""
    n_params_AB = len(list(G_AB.parameters()))
    avg_params_AB = avg_params[:n_params_AB]
    avg_params_BA = avg_params[n_params_AB:]

    for p, avg_p in zip(G_AB.parameters(), avg_params_AB):
        p.data.copy_(avg_p)
    for p, avg_p in zip(G_BA.parameters(), avg_params_BA):
        p.data.copy_(avg_p)


class CycleGANModule(pl.LightningModule):
    """
    CycleGAN with perceptual identity + perceptual reconstruction (LPIPS) losses
    and hinge-style discriminator losses.
    """

    def __init__(
        self,
        rgb_channels: int = 3,
        hsi_channels: int = 16,
        image_size: int = 256,
        lambda_identity: float = 0.5,
        lambda_cycle: float = 10.0,
        lambda_perceptual: float = 1.0,
        lambda_l1: float = 0.5,
        lr: float = 0.0002,
        log_images_on_step_n: int = 1,
        val_check_interval: int = 500,
        num_val_batches: int = 8,
        pred_output_dir: str = "generated_samples",
        pred_num_samples: int = 100,
        pred_hyperspectral: bool = True,
        pred_global_min: float | list[float] | None = None,
        pred_global_max: float | list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # ------------------ Generator/Discriminator setup ------------------
        self.G_AB = Generator(rgb_channels, hsi_channels)
        self.G_BA = Generator(hsi_channels, rgb_channels)
        self.D_A = Discriminator(rgb_channels)
        self.D_B = Discriminator(hsi_channels)

        self.avg_param_G = None

        # ------------------ Perceptual modules ------------------
        self.percept_rgb = PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=True, in_chans=rgb_channels
        )
        self.percept_hsi = PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=True, in_chans=hsi_channels
        )

        # ------------------ Logging and buffers ------------------
        self.register_buffer("fixed_rgb", torch.zeros(8, rgb_channels,
                                                      self.hparams.image_size,
                                                      self.hparams.image_size))

        # ------------------ Validation metrics ------------------
        self.sam = SpectralAngleMapper()
        self.rase = RelativeAverageSpectralError()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.tv = TotalVariation()

        self.inception_model = InceptionV3Wrapper(
            normalize_input=False, in_chans=hsi_channels
        )
        self.inception_model.eval()
        self.fid = FrechetInceptionDistance(
            self.inception_model,
            input_img_size=(hsi_channels, self.hparams.image_size, self.hparams.image_size),
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
        run_name = "cyclegan_"

        hparams = getattr(self, "hparams", None)
        if hparams is None:
            return

        # add init args to tags and run name
        run_name += f"{hparams.image_size}px_"
        run_name += f"{hparams.rgb_channels}to{hparams.hsi_channels}_"
        tags.append(f"imsize_{hparams.image_size}")
        tags.append(f"rgbch_{hparams.rgb_channels}")
        tags.append(f"hsich_{hparams.hsi_channels}")
        tags.append(f"lambda_id_{hparams.lambda_identity}")
        tags.append(f"lambda_cycle_{hparams.lambda_cycle}")
        tags.append(f"lambda_percept_{hparams.lambda_perceptual}")
        tags.append(f"lambda_l1_{hparams.lambda_l1}")

        # append learning rate using scientific notation
        tags.append(f"lr_{hparams.lr:.0e}")

        return tags, run_name

    def setup(self, stage: str) -> None:
        add_tags_and_run_name_to_logger(self)
        from src.data_modules.joint_rgb_hsi_dermoscopy import (
            JointRGBHSIDataModule,
        )

        datamodule = self.trainer.datamodule

        if not isinstance(datamodule, JointRGBHSIDataModule):
            raise ValueError(
                "CycleGAN requires JointRGBHSIDataModule"
            )

        hsi_dm = datamodule.hsi_dm

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
        train_labels = np.array(
            [base_dataset.labels[i] for i in train_indices]
        )
        unique_labels = np.unique(train_labels)

        if len(unique_labels) == 1:
            lesion_class_int = unique_labels[0]
            inverse_labels_map = {
                v: k for k, v in base_dataset.labels_map.items()
            }
            self.lesion_class_name = inverse_labels_map[lesion_class_int]

        # Initialize spectra storage
        self.real_spectra = (
            {"normal_skin": [], self.lesion_class_name: []}
            if self.lesion_class_name
            else None
        )
        self.fake_spectra = (
            {"normal_skin": [], self.lesion_class_name: []}
            if self.lesion_class_name
            else None
        )

        # Get global min/max from datamodule
        if (
            hasattr(hsi_dm, "global_min")
            and hasattr(hsi_dm, "global_max")
            and (hsi_dm.global_min is not None)
            and (hsi_dm.global_max is not None)
        ):
            self.hparams.pred_global_min = hsi_dm.global_min
            self.hparams.pred_global_max = hsi_dm.global_max

    def on_train_start(self):
        """Initialize EMA params on correct device"""
        self.avg_param_G = [
            p.data.clone().detach().to(self.device)
            for p in list(self.G_AB.parameters())
            + list(self.G_BA.parameters())
        ]

    def forward(self, real_A, real_B):
        fake_A = self.G_BA(real_B)
        fake_B = self.G_AB(real_A)
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        return fake_A, fake_B, rec_A, rec_B

    def process_batch(self, batch):
        """Extract RGB and HSI from batch dict"""
        if not isinstance(batch, dict):
            raise ValueError("Batch must be a dict with 'rgb' and 'hsi' keys")

        rgb_image, rgb_mask, rgb_label = batch["rgb"]
        hsi_image, hsi_mask, hsi_label = batch["hsi"]

        return rgb_image, hsi_image

    def _calculate_g_loss(
        self, real_A, real_B, fake_A, fake_B, rec_A, rec_B
    ) -> dict:
        """
        Compute the generator loss with:
          - Hinge-style adversarial loss
          - Perceptual + L1 identity loss
          - Perceptual + L1 cycle reconstruction loss
        """
        # ------------------ (1) Adversarial hinge loss ------------------
        logits_fake_A = self.D_A(fake_A)
        logits_fake_B = self.D_B(fake_B)

        adv_loss_A = -logits_fake_A.mean()
        adv_loss_B = -logits_fake_B.mean()
        adv_loss = adv_loss_A + adv_loss_B

        # ------------------ (2) Identity perceptual + L1 ------------------
        # identity_A = self.G_BA(real_B)
        # identity_B = self.G_AB(real_A)
        # l1_identity = F.l1_loss(identity_A, real_A) + F.l1_loss(identity_B, real_B)
        # percept_identity_rgb = self.percept_rgb(identity_A, real_A).sum()
        # percept_identity_hsi = self.percept_hsi(identity_B, real_B).sum()
        # identity_loss = l1_identity * self.hparams.lambda_l1 + self.hparams.lambda_perceptual * (
        #     percept_identity_rgb + percept_identity_hsi
        # )

        # ------------------ (3) Cycle perceptual + L1 ------------------
        l1_cycle = F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B)
        percept_cycle_rgb = self.percept_rgb(rec_A, real_A).sum()
        percept_cycle_hsi = self.percept_hsi(rec_B, real_B).sum()
        cycle_loss = l1_cycle * self.hparams.lambda_l1 + self.hparams.lambda_perceptual * (
            percept_cycle_rgb + percept_cycle_hsi
        )

        # ------------------ (4) Total generator loss ------------------
        g_loss = (
            adv_loss
            # + self.hparams.lambda_identity * identity_loss
            + self.hparams.lambda_cycle * cycle_loss
        )

        return {
            "adv_loss": adv_loss,
            # "identity_loss": identity_loss,
            "cycle_loss": cycle_loss,
            "g_loss": g_loss,
        }

    def _calculate_d_loss(self, real_A, real_B, fake_A, fake_B) -> dict:
        """
        Hinge-style discriminator loss:
          D_loss = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
        """
        # --- Domain A ---
        logits_real_A = self.D_A(real_A)
        logits_fake_A = self.D_A(fake_A.detach())
        d_loss_real_A = F.relu(1.0 - logits_real_A).mean()
        d_loss_fake_A = F.relu(1.0 + logits_fake_A).mean()
        d_loss_A = 0.5 * (d_loss_real_A + d_loss_fake_A)

        # --- Domain B ---
        logits_real_B = self.D_B(real_B)
        logits_fake_B = self.D_B(fake_B.detach())
        d_loss_real_B = F.relu(1.0 - logits_real_B).mean()
        d_loss_fake_B = F.relu(1.0 + logits_fake_B).mean()
        d_loss_B = 0.5 * (d_loss_real_B + d_loss_fake_B)

        d_loss = d_loss_A + d_loss_B

        return {
            "d_loss": d_loss,
            "d_loss_A": d_loss_A,
            "d_loss_B": d_loss_B,
        }

    # ----------------------------------------------------------------------
    # Training step
    # ----------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        real_rgb, real_hsi = self.process_batch(batch)
        fake_rgb, fake_hsi, rec_rgb, rec_hsi = self(real_rgb, real_hsi)

        d_optim, g_optim = self.optimizers()

        # --- Train Discriminator ---
        d_loss_dict = self._calculate_d_loss(
            real_A=real_rgb, real_B=real_hsi, fake_A=fake_rgb, fake_B=fake_hsi
        )
        d_optim.zero_grad(set_to_none=True)
        self.manual_backward(d_loss_dict["d_loss"])
        d_optim.step()

        # --- Train Generator ---
        g_loss_dict = self._calculate_g_loss(
            real_A=real_rgb,
            real_B=real_hsi,
            fake_A=fake_rgb,
            fake_B=fake_hsi,
            rec_A=rec_rgb,
            rec_B=rec_hsi,
        )
        g_optim.zero_grad(set_to_none=True)
        self.manual_backward(g_loss_dict["g_loss"])
        g_optim.step()

        # --- EMA Update ---
        for p, avg_p in zip(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            self.avg_param_G,
        ):
            avg_p.mul_(0.999).add_(p.data, alpha=0.001)

        # --- Logging ---
        log_dict = {f"train/{k}": v for k, v in {**d_loss_dict, **g_loss_dict}.items()}
        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        rgb_image, _ = self.process_batch(batch)
        backup_para = copy_cyclegan_params(self.G_AB, self.G_BA)
        load_cyclegan_params(self.G_AB, self.G_BA, self.avg_param_G)
        self.G_AB.eval()
        self.G_BA.eval()


        # Run validation periodically
        if (
            self.global_step // 2
        ) % self.hparams.val_check_interval == 0:
            self._run_validation()

        # Log generated samples periodically
        if (
            self.global_step // 2
        ) % self.hparams.log_images_on_step_n == 0:
            # Store fixed RGB input on first log
            if torch.all(self.fixed_rgb == 0):
                real_rgb, _ = self.process_batch(batch)
                self.fixed_rgb = real_rgb[:8].clone().detach()

            with torch.no_grad():
                # RGB -> HSI -> RGB cycle
                fake_hsi = self.G_AB(self.fixed_rgb)
                rec_rgb = self.G_BA(fake_hsi)

                # Denormalize for visualization
                fixed_rgb_vis = self.fixed_rgb.add(1).mul(0.5).clamp(0, 1)
                rec_rgb_vis = rec_rgb.add(1).mul(0.5).clamp(0, 1)

                # For HSI, convert to 3-channel visualization
                if fake_hsi.size(1) > 3:
                    fake_hsi_vis = (
                        fake_hsi.mean(dim=1, keepdim=True)
                        .repeat(1, 3, 1, 1)
                        .add(1)
                        .mul(0.5)
                        .clamp(0, 1)
                    )
                else:
                    fake_hsi_vis = fake_hsi.add(1).mul(0.5).clamp(0, 1)

                # Concatenate: original RGB | fake HSI | reconstructed RGB
                vis_combined = torch.cat(
                    [fixed_rgb_vis, fake_hsi_vis, rec_rgb_vis], dim=0
                )

                sample_grid = (
                    torchvision.utils.make_grid(vis_combined, nrow=vis_combined.size(0) // 3)
                    .detach()
                    .cpu()
                )
                self.logger.experiment.log(
                    {
                        "generated_samples": wandb.Image(
                            sample_grid,
                            caption="RGB | Fake HSI | Reconstructed RGB",
                        )
                    }
                )

        load_cyclegan_params(self.G_AB, self.G_BA, backup_para)
        self.G_AB.train()
        self.G_BA.train()

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

                real_rgb, real_hsi = self.process_batch(batch)
                real_rgb = real_rgb.to(self.device, non_blocking=True)
                real_hsi = real_hsi.to(self.device, non_blocking=True)

                # Generate RGB -> HSI
                fake_hsi = self.G_AB(real_rgb)

                # Normalize to [0, 1]
                fake_norm = (fake_hsi + 1) / 2
                real_norm = (real_hsi + 1) / 2
                fake_norm = fake_norm.clamp(0, 1)
                real_norm = real_norm.clamp(0, 1)

                # Update FID
                self.fid.update(fake_hsi, real=False)
                self.fid.update(real_hsi, real=True)

                # Compute metrics
                eps = 1e-8
                fake_norm_clamped = fake_norm.clamp(eps, 1.0)
                real_norm_clamped = real_norm.clamp(eps, 1.0)

                try:
                    sam_val = self.sam(fake_norm_clamped, real_norm_clamped)
                    if torch.isnan(sam_val):
                        sam_val = torch.tensor(0.0, device=self.device)
                except Exception:
                    sam_val = torch.tensor(0.0, device=self.device)

                rase_val = torch.nan_to_num(
                    self.rase(fake_norm_clamped, real_norm_clamped), nan=0.0
                )
                ssim_val = torch.nan_to_num(
                    self.ssim(fake_norm_clamped, real_norm_clamped), nan=0.0
                )
                tv_val = torch.nan_to_num(
                    self.tv(fake_norm_clamped), nan=0.0
                )

                sam_sum += sam_val.item()
                rase_sum += rase_val.item()
                ssim_sum += ssim_val.item()
                tv_sum += tv_val.item()
                count += 1

                if self.lesion_class_name is not None:
                    self.compute_spectra_statistics(
                        real_norm_clamped, fake_norm_clamped
                    )

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

            if self.lesion_class_name is not None and (
                self.real_spectra or self.fake_spectra
            ):
                self._plot_mean_spectra()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_spectra_statistics(self, real_norm, fake_norm):
        """Extract spectra from lesion regions using Otsu thresholding"""
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
                        normal_spectrum = image_np[normal_skin_mask].mean(
                            axis=0
                        )
                        spectra_dict["normal_skin"].append(normal_spectrum)
                except Exception:
                    continue

    def _plot_mean_spectra(self):
        """Plot mean spectra comparing real vs synthetic data"""
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
        ref_label = next(
            (lbl for lbl in labels if ref_stats.get(lbl) is not None), None
        )
        if ref_label is None:
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
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {lbl}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

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

        self.logger.experiment.log({"val/mean_spectra": wandb.Image(fig)})
        plt.close(fig)

    def predict_step(self, batch, batch_idx):
        """Generate HSI from RGB samples"""
        os.makedirs(self.hparams.pred_output_dir, exist_ok=True)

        backup_para = copy_cyclegan_params(self.G_AB, self.G_BA)
        load_cyclegan_params(self.G_AB, self.G_BA, self.avg_param_G)
        self.G_AB.eval()

        gmin = getattr(self.hparams, "pred_global_min", None)
        gmax = getattr(self.hparams, "pred_global_max", None)

        with torch.no_grad():
            real_rgb, _ = self.process_batch(batch)
            real_rgb = real_rgb.to(self.device)

            # Generate HSI from RGB
            fake_hsi = self.G_AB(real_rgb).cpu()

            for i in tqdm(
                range(fake_hsi.size(0)), desc="Generating HSI samples"
            ):
                fake_img = fake_hsi[i : i + 1, :, :, :]

                if self.hparams.pred_hyperspectral:
                    fake_denorm = fake_img.add(1).div(2)

                    if gmin is not None and gmax is not None:
                        gmin_arr = (
                            torch.tensor(gmin, device=self.device)
                            if isinstance(gmin, list)
                            else np.array(gmin)
                        )
                        gmax_arr = (
                            torch.tensor(gmax, device=self.device)
                            if isinstance(gmax, list)
                            else np.array(gmax)
                        )

                        if gmin_arr.size == 1:
                            fake_denorm = (
                                fake_denorm * (gmax_arr - gmin_arr) + gmin_arr
                            )
                        else:
                            gmin_t = torch.tensor(gmin_arr).view(1, -1, 1, 1)
                            gmax_t = torch.tensor(gmax_arr).view(1, -1, 1, 1)
                            fake_denorm = (
                                fake_denorm * (gmax_t - gmin_t) + gmin_t
                            )

                    fake_np = fake_denorm.squeeze().numpy()
                    fake_np = np.transpose(fake_np, (1, 2, 0))
                    mat_path = os.path.join(
                        self.hparams.pred_output_dir,
                        f"sample_{batch_idx}_{i:04d}.mat",
                    )
                    savemat(mat_path, {"cube": fake_np})
                else:
                    fake_rgb = (fake_img + 1) / 2
                    mean_band = fake_rgb.mean(dim=1, keepdim=True)
                    rgb = mean_band.repeat(1, 3, 1, 1).clamp(0, 1)

                    save_path = os.path.join(
                        self.hparams.pred_output_dir,
                        f"sample_{batch_idx}_{i:04d}.png",
                    )
                    torchvision.utils.save_image(rgb, save_path)

        load_cyclegan_params(self.G_AB, self.G_BA, backup_para)
        self.G_AB.train()

    def configure_optimizers(self):
        d_optim = Adam(
            list(self.D_A.parameters()) + list(self.D_B.parameters()),
            lr=self.hparams.lr,
            betas=(0.5, 0.999),
        )
        g_optim = Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=self.hparams.lr,
            betas=(0.5, 0.999),
        )
        return [d_optim, g_optim], []

    def on_save_checkpoint(self, checkpoint):
        """Save EMA parameters"""
        checkpoint["avg_param_G"] = [
            p.clone().cpu() for p in self.avg_param_G
        ]

    def on_load_checkpoint(self, checkpoint):
        """Load EMA parameters"""
        if "avg_param_G" in checkpoint:
            self.avg_param_G = [
                p.to(self.device) for p in checkpoint["avg_param_G"]
            ]
        else:
            print("Warning: avg_param_G not found in checkpoint.")
