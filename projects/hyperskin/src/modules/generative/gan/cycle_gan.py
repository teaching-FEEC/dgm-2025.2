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
from PIL import Image

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

from src.modules.generative.base_predictor import BasePredictorMixin
from src.models.fastgan.unet_generator import define_D, define_G
from src.models.cycle_gan.cycle_gan import Generator, Discriminator
from src.utils.spectra_plot import MeanSpectraMetric
from src.utils.tags_and_run_name import add_tags_and_run_name_to_logger
from src.utils.utils import _iterate_val_loaders


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


class CycleGANModule(BasePredictorMixin, pl.LightningModule):
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
        pred_hyperspectral: bool = True,
        pred_global_min: float | list[float] | None = None,
        pred_global_max: float | list[float] | None = None,
        label_smoothing: float = 0.0,
        noise_std_start: float = 0.0,
        noise_std_end: float = 0.0,
        noise_std: float | None = None,
        noise_decay_steps: int = 100000,
        model_opt: dict | None = None,

        # auxiliary classifier parameters
        aux_num_classes: Optional[int] = None,
        aux_loss_weight_d: float = 1.0,
        aux_loss_weight_g: float = 1.0,
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
                                            model_opt["init_gain"],
                                            num_classes=aux_num_classes)
            self.D_B = define_D(hsi_channels, model_opt["ndf"], model_opt["netD"],
                                            model_opt["n_layers_D"], model_opt["norm"], model_opt["init_type"],
                                            model_opt["init_gain"],
                                            num_classes=aux_num_classes)
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
        self.batch_size = None

    @staticmethod
    def _split_d_output(d_out):
        """Handle D(x) that returns either logits or (logits, class_logits)."""
        if isinstance(d_out, (tuple, list)) and len(d_out) == 2:
            return d_out[0], d_out[1]
        return d_out, None

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
        # get batch size
        self.fixed_A_batch_size = min(8, hsi_dm.batch_size)

            # ------------------ Logging and buffers ------------------
        self.register_buffer("fixed_A", torch.zeros((self.fixed_A_batch_size,
                                                     self.hparams.rgb_channels,
                                                     self.hparams.image_size,
                                                     self.hparams.image_size)))


    def on_train_start(self):
        """Initialize EMA params on correct device"""
        self.avg_param_G = [
            p.data.clone().detach().to(self.device)
            for p in list(self.G_AB.parameters())
            + list(self.G_BA.parameters())
        ]

        # do a quick sanity check on the train_dataloader to see if both domains are present
        # also if auxiliary classification is used, check labels are present
        train_loader = self.trainer.train_dataloader
        batch = next(_iterate_val_loaders(train_loader))
        if "rgb" not in batch or "hsi" not in batch:
            raise ValueError(
                "Input batch must contain both 'rgb' and 'hsi' domains for CycleGAN."
            )
        if (
            self.hparams.aux_num_classes is not None
            and self.hparams.aux_loss_weight_d > 0.0
        ):
            if "label" not in batch["rgb"] or "label" not in batch["hsi"]:
                raise ValueError(
                    "Auxiliary classification enabled but label key not found in batch."
                )

    def forward(self, real_A, real_B):
        fake_A = self.G_BA(real_B)
        fake_B = self.G_AB(real_A)
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        return fake_A, fake_B, rec_A, rec_B

    def _calculate_g_loss(
        self, real_A, real_B, fake_A, fake_B, rec_A, rec_B,
        label_A: torch.Tensor | None = None,
        label_B: torch.Tensor | None = None,
    ) -> dict:
        """
        Compute the generator loss with:
          - Hinge-style adversarial loss
          - Perceptual + L1 cycle reconstruction loss
        """
        # ------------------ (1) Adversarial hinge loss ------------------
        d_out_fake_A = self.D_A(fake_A)
        d_out_fake_B = self.D_B(fake_B)
        logits_fake_A, cls_fake_A = self._split_d_output(d_out_fake_A)
        logits_fake_B, cls_fake_B = self._split_d_output(d_out_fake_B)

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

        # (4) Auxiliary classification loss (for generator)
        aux_g_loss_A = torch.tensor(0.0, device=self.device)
        aux_g_loss_B = torch.tensor(0.0, device=self.device)

        if (
            self.hparams.aux_num_classes is not None
            and label_A is not None
            and label_B is not None
            and self.hparams.aux_loss_weight_g > 0.0
        ):
            # Re-run D or reuse, but here we assume d_out_fake_* from above:
            if cls_fake_A is not None:
                aux_g_loss_A = F.cross_entropy(cls_fake_A, label_A)
            if cls_fake_B is not None:
                aux_g_loss_B = F.cross_entropy(cls_fake_B, label_B)

        aux_g_loss = aux_g_loss_A + aux_g_loss_B
        aux_g_loss = self.hparams.aux_loss_weight_g * aux_g_loss

        # ------------------ (4) Total generator loss ------------------
        g_loss = (
            adv_loss
            + self.hparams.lambda_cycle * cycle_loss
            + self.hparams.lambda_perceptual * cycle_loss_percept
            + aux_g_loss
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
            "aux_g_loss_A": aux_g_loss_A,
            "aux_g_loss_B": aux_g_loss_B,
            "aux_g_loss": aux_g_loss,
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

    def _calculate_d_loss(
        self,
        real_A,
        real_B,
        fake_A,
        fake_B,
        label_A: Optional[torch.Tensor] = None,
        label_B: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Discriminator loss:
          - BCE real/fake (with noise + label smoothing)
          - Optional auxiliary classification loss on real samples.
        """
        noise_std = self.get_current_noise_std()
        label_smoothing = getattr(self.hparams, "label_smoothing", 0.0)

        # (1) Add Gaussian noise
        if noise_std > 0:
            real_A = real_A + torch.randn_like(real_A) * noise_std
            fake_A = fake_A + torch.randn_like(fake_A) * noise_std
            real_B = real_B + torch.randn_like(real_B) * noise_std
            fake_B = fake_B + torch.randn_like(fake_B) * noise_std

        # (2) Real labels with smoothing
        # Use a forward pass to get label tensors with correct shape
        logits_real_A, cls_real_A = self._split_d_output(self.D_A(real_A))
        logits_real_B, cls_real_B = self._split_d_output(self.D_B(real_B))

        if label_smoothing > 0:
            real_label_A = torch.empty_like(logits_real_A).uniform_(
                1.0 - label_smoothing,
                1.0,
            )
            real_label_B = torch.empty_like(logits_real_B).uniform_(
                1.0 - label_smoothing,
                1.0,
            )
        else:
            real_label_A = torch.ones_like(logits_real_A)
            real_label_B = torch.ones_like(logits_real_B)

        fake_label_A = torch.zeros_like(real_label_A)
        fake_label_B = torch.zeros_like(real_label_B)

        # (3) Real/fake losses for A
        logits_fake_A, cls_fake_A = self._split_d_output(
            self.D_A(fake_A.detach())
        )

        d_loss_real_A = bce_with_logits(logits_real_A, real_label_A)
        d_loss_fake_A = bce_with_logits(logits_fake_A, fake_label_A)
        d_loss_A = 0.5 * (d_loss_real_A + d_loss_fake_A)

        # (4) Real/fake losses for B
        logits_fake_B, cls_fake_B = self._split_d_output(
            self.D_B(fake_B.detach())
        )

        d_loss_real_B = bce_with_logits(logits_real_B, real_label_B)
        d_loss_fake_B = bce_with_logits(logits_fake_B, fake_label_B)
        d_loss_B = 0.5 * (d_loss_real_B + d_loss_fake_B)

        # (5) Auxiliary classification loss (on real samples)
        aux_d_loss_A = torch.tensor(0.0, device=self.device)
        aux_d_loss_B = torch.tensor(0.0, device=self.device)

        if (
            self.hparams.aux_num_classes is not None
            and label_A is not None
            and label_B is not None
            and self.hparams.aux_loss_weight_d > 0.0
        ):
            if cls_real_A is not None:
                aux_d_loss_A = F.cross_entropy(cls_real_A, label_A)
            if cls_real_B is not None:
                aux_d_loss_B = F.cross_entropy(cls_real_B, label_B)

        aux_d_loss = aux_d_loss_A + aux_d_loss_B
        aux_d_loss = self.hparams.aux_loss_weight_d * aux_d_loss

        d_loss = d_loss_A + d_loss_B + aux_d_loss

        return {
            "d_loss_real_A": d_loss_real_A,
            "d_loss_fake_A": d_loss_fake_A,
            "d_loss_A": d_loss_A,
            "d_loss_real_B": d_loss_real_B,
            "d_loss_fake_B": d_loss_fake_B,
            "d_loss_B": d_loss_B,
            "aux_d_loss_A": aux_d_loss_A,
            "aux_d_loss_B": aux_d_loss_B,
            "aux_d_loss": aux_d_loss,
            "d_loss": d_loss,
        }

    # ----------------------------------------------------------------------
    # Training step
    # ----------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        rgb_batch = batch["rgb"]
        hsi_batch = batch["hsi"]

        real_rgb = rgb_batch["image"]
        real_hsi = hsi_batch["image"]
        rgb_labels = rgb_batch.get("label", None)
        hsi_labels = hsi_batch.get("label", None)

        fake_rgb, fake_hsi, rec_rgb, rec_hsi = self(real_rgb, real_hsi)

        d_optim, g_optim = self.optimizers()

        # --- Train Discriminator ---
        d_loss_dict = self._calculate_d_loss(
            real_A=real_rgb,
            real_B=real_hsi,
            fake_A=fake_rgb,
            fake_B=fake_hsi,
            label_A=rgb_labels,
            label_B=hsi_labels,
        )

        # discriminator accuracy (only uses rf logits)
        with torch.no_grad():
            logits_real_A, _ = self._split_d_output(self.D_A(real_rgb))
            logits_fake_A, _ = self._split_d_output(self.D_A(fake_rgb.detach()))
            logits_real_B, _ = self._split_d_output(self.D_B(real_hsi))
            logits_fake_B, _ = self._split_d_output(self.D_B(fake_hsi.detach()))

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
            label_A=rgb_labels,
            label_B=hsi_labels,
        )

        g_optim.zero_grad(set_to_none=True)
        self.manual_backward(g_loss_dict["g_loss"])
        g_optim.step()

        # --- EMA and logging (unchanged, but includes new keys) ---
        for p, avg_p in zip(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            self.avg_param_G,
        ):
            avg_p.mul_(0.999).add_(p.data, alpha=0.001)

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
        self.log(
            "train/noise_std",
            self.get_current_noise_std(),
            prog_bar=False,
            logger=True,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
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
                real_rgb = batch["rgb"]["image"]
                self.fixed_A = real_rgb[:self.fixed_A_batch_size].clone().detach()

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

                if hasattr(self.logger, "experiment"):
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
            for i, batch in enumerate(_iterate_val_loaders(val_loader)):
                if i >= self.hparams.num_val_batches:
                    break

                real_rgb = batch["rgb"]["image"]
                real_hsi = batch["hsi"]["image"]
                real_rgb = real_rgb.to(self.device)
                real_hsi = real_hsi.to(self.device)
                rgb_labels = batch["rgb"].get("label", None)
                hsi_labels = batch["hsi"].get("label", None)

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

                self.spectra_metric.update(real_norm, is_fake=False, labels=hsi_labels)
                self.spectra_metric.update(fake_norm, is_fake=True, labels=rgb_labels)

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
        if fig is not None and hasattr(self.logger, "experiment"):
            self.logger.experiment.log({"Mean Spectra": wandb.Image(fig)})
        self.spectra_metric.reset()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_predict_start(self) -> None:
        datamodule = self.trainer.datamodule
        predict_dataloader = datamodule.predict_dataloader()

        if len(predict_dataloader) > 1:
            print(
                "WARNING: More than one predict dataloader detected. "
                "Probably using JointRGBHSIDataModule, but CycleGAN predict only uses the rgb dataloader."
                "All images with channels different than rgb_channels will be ignored."
                "Use --data.init_args.rgb_only=True to avoid this warning."
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        real_rgb = batch['image']

        if real_rgb.size(1) != self.hparams.rgb_channels:
            return

        labels = batch.get('label', None)
        backup_para = copy_cyclegan_params(self.G_AB, self.G_BA)
        load_cyclegan_params(self.G_AB, self.G_BA, self.avg_param_G)
        self.G_AB.eval()

        with torch.no_grad():
            fake_hsi = self.G_AB(real_rgb)

        self._save_generated_batch(
            fake_batch=fake_hsi,
            batch_idx=batch_idx,
            pred_hyperspectral=self.hparams.pred_hyperspectral,
            labels=labels.tolist() if labels is not None else None,
        )

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
