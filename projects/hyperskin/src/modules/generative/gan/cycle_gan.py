from typing import Optional

from matplotlib.pyplot import isinteractive
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
from pytorch_lightning.trainer.states import TrainerFn

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

from src.models.fastgan.unet_generator import define_D, define_G
from src.models.cycle_gan.cycle_gan import Generator, Discriminator
from src.utils.spectra_plot import MeanSpectraMetric
from src.utils.tags_and_run_name import add_tags_and_run_name_to_logger


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
    CycleGAN with perceptual reconstruction (LPIPS) losses
    and hinge-style discriminator losses.
    """

    def __init__(
        self,
        rgb_channels: int = 3,
        hsi_channels: int = 16,
        image_size: int = 256,
        lambda_cycle: float = 10.0,
        lambda_perceptual: float = 1.0,
        lr: float = 0.0002,
        log_images_on_step_n: int = 1,
        val_check_interval: int = 500,
        num_val_batches: int = 8,
        pred_output_dir: str = "generated_samples",
        pred_num_samples: int = 100,
        pred_hyperspectral: bool = True,
        pred_global_min: float | list[float] | None = None,
        pred_global_max: float | list[float] | None = None,
        label_smoothing: float = 0.0,
        noise_std_start: float = 0.05,
        noise_std_end: float = 0.0,
        noise_std: float | None = None,
        noise_decay_steps: int = 100000,
        model_opt: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # ------------------ Generator/Discriminator setup ------------------
        if model_opt is not None:
            self.G_AB = define_G(rgb_channels, hsi_channels, model_opt["ngf"], model_opt["netG"], model_opt["norm"],
                                            not model_opt["no_dropout"], model_opt["init_type"], model_opt["init_gain"])
            self.G_BA = define_G(hsi_channels, rgb_channels, model_opt["ngf"], model_opt["netG"], model_opt["norm"],
                                            not model_opt["no_dropout"], model_opt["init_type"], model_opt["init_gain"])

            self.D_A = define_D(rgb_channels, model_opt["ndf"], model_opt["netD"],
                                            model_opt["n_layers_D"], model_opt["norm"], model_opt["init_type"],
                                            model_opt["init_gain"])
            self.D_B = define_D(hsi_channels, model_opt["ndf"], model_opt["netD"],
                                            model_opt["n_layers_D"], model_opt["norm"], model_opt["init_type"],
                                            model_opt["init_gain"])
        else:
            self.G_AB = Generator(rgb_channels, hsi_channels)
            self.G_BA = Generator(hsi_channels, rgb_channels)
            self.D_A = Discriminator(rgb_channels)
            self.D_B = Discriminator(hsi_channels)

        self.avg_param_G = None

        # ------------------ Perceptual modules ------------------
        self.percept_A = PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=True, in_chans=rgb_channels
        )
        self.percept_B = PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=True, in_chans=hsi_channels
        )

        # ------------------ Logging and buffers ------------------
        self.register_buffer("fixed_A", torch.zeros(8, rgb_channels,
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

        self.spectra_metric = MeanSpectraMetric()

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

        # append learning rate using scientific notation
        tags.append(f"lr_{hparams.lr:.0e}")

        return tags, run_name.rstrip("_")

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
            if isinstance(batch, list) or isinstance(batch, tuple):
                return batch[0], batch[1]
            if isinstance(batch, torch.Tensor):
                return batch

        rgb_mask = None
        hsi_mask = None
        if isinstance(batch["rgb"], list) or isinstance(batch["rgb"], tuple):
            rgb_image = batch["rgb"][0]
            rgb_mask = batch["rgb"][1] if len(batch["rgb"]) > 1 else None

        if isinstance(batch["hsi"], list) or isinstance(batch["hsi"], tuple):
            hsi_image = batch["hsi"][0]
            hsi_mask = batch["hsi"][1] if len(batch["hsi"]) > 1 else None


        if isinstance(batch["rgb"], torch.Tensor):
            rgb_image = batch["rgb"]
        else:
            rgb_image = batch["rgb"][0]

        if isinstance(batch["hsi"], torch.Tensor):
            hsi_image = batch["hsi"]
        else:
            hsi_image = batch["hsi"][0]

        if rgb_mask is not None and hsi_mask is not None:
            return rgb_image, hsi_image, rgb_mask, hsi_mask
        else:
            return rgb_image, hsi_image

    def _calculate_g_loss(
        self, real_A, real_B, fake_A, fake_B, rec_A, rec_B
    ) -> dict:
        """
        Compute the generator loss with:
          - Hinge-style adversarial loss
          - Perceptual + L1 cycle reconstruction loss
        """
        # ------------------ (1) Adversarial hinge loss ------------------
        logits_fake_A = self.D_A(fake_A)
        logits_fake_B = self.D_B(fake_B)

        adv_loss_A = bce_with_logits(logits_fake_A, torch.ones_like(logits_fake_A))
        adv_loss_B = bce_with_logits(logits_fake_B, torch.ones_like(logits_fake_B))

        adv_loss = adv_loss_A + adv_loss_B

        cycle_loss_A = F.l1_loss(rec_A, real_A)
        cycle_loss_B = F.l1_loss(rec_B, real_B)
        cycle_loss = cycle_loss_A + cycle_loss_B

        if self.hparams.lambda_perceptual > 0:
            cycle_loss_percept_A = self.percept_A(rec_A, real_A).sum()
            cycle_loss_percept_B = self.percept_B(rec_B, real_B).sum()
            cycle_loss_percept = cycle_loss_percept_A + cycle_loss_percept_B
        else:
            cycle_loss_percept_A = torch.tensor(0.0, device=self.device)
            cycle_loss_percept_B = torch.tensor(0.0, device=self.device)
            cycle_loss_percept = torch.tensor(0.0, device=self.device)

        # ------------------ (4) Total generator loss ------------------
        g_loss = (
            adv_loss
            + self.hparams.lambda_cycle * cycle_loss
            + self.hparams.lambda_perceptual * cycle_loss_percept
        )

        return {
            "adv_loss_A": adv_loss_A,
            "adv_loss_B": adv_loss_B,
            "cycle_loss_percept_A": cycle_loss_percept_A,
            "cycle_loss_percept_B": cycle_loss_percept_B,
            "cycle_loss_A": cycle_loss_A,
            "cycle_loss_B": cycle_loss_B,
            "cycle_loss_percept": cycle_loss_percept,
            "adv_loss": adv_loss,
            "cycle_loss": cycle_loss,
            "g_loss": g_loss,
        }

    def get_current_noise_std(self) -> float:
        """Compute linearly decayed noise_std based on training progress."""
        start = getattr(self.hparams, "noise_std_start", 0.0)
        end = getattr(self.hparams, "noise_std_end", 0.0)
        decay_steps = getattr(self.hparams, "noise_decay_steps", 100000)

        if self.hparams.noise_std is not None:
            return self.hparams.noise_std

        if start == end or decay_steps <= 0:
            return start

        # Compute proportion of training completed
        step = float(self.global_step)
        progress = min(step / decay_steps, 1.0)

        # Linear decay schedule: interpolate between start and end
        return start + (end - start) * progress

    def _calculate_d_loss(self, real_A, real_B, fake_A, fake_B) -> dict:
        """
        Binary cross-entropy discriminator loss with optional
        Gaussian input noise and one-sided label smoothing.
        """

        noise_std = self.get_current_noise_std()
        label_smoothing = getattr(self.hparams, "label_smoothing", 0.0)

        # ---- (1) Add Gaussian noise to discriminator inputs ----
        if noise_std > 0:
            real_A = real_A + torch.randn_like(real_A) * noise_std
            fake_A = fake_A + torch.randn_like(fake_A) * noise_std
            real_B = real_B + torch.randn_like(real_B) * noise_std
            fake_B = fake_B + torch.randn_like(fake_B) * noise_std

        # ---- (2) Create soft labels for "real" samples ----
        if label_smoothing > 0:
            real_label_A = torch.empty_like(self.D_A(real_A)).uniform_(
                1.0 - label_smoothing, 1.0
            )
            real_label_B = torch.empty_like(self.D_B(real_B)).uniform_(
                1.0 - label_smoothing, 1.0
            )
        else:
            real_label_A = torch.ones_like(self.D_A(real_A))
            real_label_B = torch.ones_like(self.D_B(real_B))

        fake_label_A = torch.zeros_like(real_label_A)
        fake_label_B = torch.zeros_like(real_label_B)

        # ---- (3) Compute discriminator losses ----
        logits_real_A = self.D_A(real_A)
        logits_fake_A = self.D_A(fake_A.detach())
        d_loss_real_A = bce_with_logits(logits_real_A, real_label_A)
        d_loss_fake_A = bce_with_logits(logits_fake_A, fake_label_A)
        d_loss_A = 0.5 * (d_loss_real_A + d_loss_fake_A)

        logits_real_B = self.D_B(real_B)
        logits_fake_B = self.D_B(fake_B.detach())
        d_loss_real_B = bce_with_logits(logits_real_B, real_label_B)
        d_loss_fake_B = bce_with_logits(logits_fake_B, fake_label_B)
        d_loss_B = 0.5 * (d_loss_real_B + d_loss_fake_B)

        d_loss = d_loss_A + d_loss_B

        return {
            "d_loss_real_A": d_loss_real_A,
            "d_loss_fake_A": d_loss_fake_A,
            "d_loss_A": d_loss_A,
            "d_loss_real_B": d_loss_real_B,
            "d_loss_fake_B": d_loss_fake_B,
            "d_loss_B": d_loss_B,
            "d_loss": d_loss,
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

        # Compute discriminator accuracy
        with torch.no_grad():
            logits_real_A = self.D_A(real_rgb)
            logits_fake_A = self.D_A(fake_rgb.detach())
            logits_real_B = self.D_B(real_hsi)
            logits_fake_B = self.D_B(fake_hsi.detach())

            pred_real_A = (torch.sigmoid(logits_real_A) > 0.5).float()
            pred_fake_A = (torch.sigmoid(logits_fake_A) <= 0.5).float()
            pred_real_B = (torch.sigmoid(logits_real_B) > 0.5).float()
            pred_fake_B = (torch.sigmoid(logits_fake_B) <= 0.5).float()

            acc_real_A = pred_real_A.mean()
            acc_fake_A = pred_fake_A.mean()
            acc_real_B = pred_real_B.mean()
            acc_fake_B = pred_fake_B.mean()

            d_acc_A = 0.5 * (acc_real_A + acc_fake_A)
            d_acc_B = 0.5 * (acc_real_B + acc_fake_B)
            d_acc_mean = 0.5 * (d_acc_A + d_acc_B)

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
        log_dict.update(
            {
                "train/D_acc_A": d_acc_A,
                "train/D_acc_B": d_acc_B,
                "train/D_acc_mean": d_acc_mean,
            }
        )

        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )
        self.log("train/noise_std", self.get_current_noise_std(), prog_bar=False, logger=True)

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
            if torch.all(self.fixed_A == 0):
                real_rgb, _ = self.process_batch(batch)
                self.fixed_A = real_rgb[:8].clone().detach()

            with torch.no_grad():
                # RGB -> HSI -> RGB cycle
                fake_hsi = self.G_AB(self.fixed_A)
                rec_rgb = self.G_BA(fake_hsi)

                # Denormalize for visualization
                fixed_A_vis = self.fixed_A.add(1).mul(0.5).clamp(0, 1)
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
                    [fixed_A_vis, fake_hsi_vis, rec_rgb_vis], dim=0
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

                self.spectra_metric.update(real_hsi, fake_hsi)

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

        fig = self.spectra_metric.plot()
        if fig is not None:
            self.logger.experiment.log({"Mean Spectra": wandb.Image(fig)})
        self.spectra_metric.reset()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_step(self, batch, batch_idx, dataloader_idx):
        """Generate HSI from RGB samples"""
        os.makedirs(self.hparams.pred_output_dir, exist_ok=True)

        gmin = getattr(self.hparams, "pred_global_min", None)
        gmax = getattr(self.hparams, "pred_global_max", None)
        gmin = torch.tensor(gmin).to(self.device) if gmin is not None else None
        gmax = torch.tensor(gmax).to(self.device) if gmax is not None else None

        if dataloader_idx == 1:
            # compute real spectra from HSI dataloader
            masks = None
            if isinstance(batch, tuple) or isinstance(batch, list):
                real_hsi, masks = batch
            else:
                real_hsi = batch
            real_hsi_denorm = real_hsi.add(1).div(2)

            if gmin is not None and gmax is not None:
                real_hsi_denorm = real_hsi_denorm * (gmax - gmin) + gmin
            self.spectra_metric.update(
                real_hsi_denorm,
                is_fake=False,
                masks=masks,
            )
            return

        backup_para = copy_cyclegan_params(self.G_AB, self.G_BA)
        if self.avg_param_G is None:
            load_cyclegan_params(self.G_AB, self.G_BA, self.avg_param_G)
        self.G_AB.eval()


        with torch.no_grad():
            batch = self.process_batch(batch)
            masks = None
            if isinstance(batch, tuple):
                real_rgb = batch[0]
                masks = batch[1] if len(batch) > 2 else None
            else:
                real_rgb = batch

            real_rgb = real_rgb.to(self.device)
            fake_hsi = self.G_AB(real_rgb)

            for i in tqdm(
                range(fake_hsi.size(0)), desc="Generating HSI samples"
            ):
                fake_img = fake_hsi[i : i + 1, :, :, :]

                if self.hparams.pred_hyperspectral:
                    fake_denorm = fake_img.add(1).div(2)

                    if gmin is not None and gmax is not None:
                        fake_denorm = (
                            fake_denorm * (gmax - gmin) + gmin
                        )

                    self.spectra_metric.update(
                        fake_denorm,
                        fake_denorm,
                        masks=masks[i : i + 1, :, :, :] if masks is not None else None
                    )
                    fake_np = fake_denorm.squeeze().cpu().numpy()
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

    def on_predict_end(self) -> None:
        fig = self.spectra_metric.plot()
        if fig is not None:
            fig.savefig("predicted_mean_spectra.png")
        self.spectra_metric.reset()
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
        del checkpoint["hyper_parameters"]
        del checkpoint["datamodule_hyper_parameters"]
