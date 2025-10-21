# -*- coding: utf-8 -*-
"""
vae_module.py — fixed logging (no W&B step warnings), per-validation-step image logging,
no clamping, encoder/decoder signatures match your original VAE.

Key fixes:
  • Logs ONE generated sample **per validation step** (as requested).
  • Removed explicit `step=...` in all `wandb.log(...)` calls to avoid
    "steps must be monotonically increasing" warnings — Lightning’s WandbLogger
    will handle step ordering automatically.
  • Kept epoch-end spectra plotting and best-metric tracking.
  • Encoder/Decoder constructors use the SAME positional (C,H,W) argument shape
    as your original module (no `input_dim` / `output_shape` keyword drift).
  • Absolutely NO clamping anywhere — decoder must end with sigmoid.

Heavily commented for clarity.
"""

# ---------------------------
# Typing helpers
# ---------------------------
from typing import Dict, List, Optional, Tuple

# ---------------------------
# Third-party imports
# ---------------------------
import numpy as np                                  # spectra accumulation/statistics
import torch                                        # tensors / autograd
import torch.nn.functional as F                     # L1 loss
import pytorch_lightning as pl                      # Lightning base class
import torchvision                                  # image grid helpers
from torchvision.utils import make_grid             # grid-making for logging images
from torch import Tensor                            # tensor alias
from torch.optim import Adam                        # optimizer
from torchmetrics import MeanMetric, MinMetric, MaxMetric  # running/best metrics
import wandb                                        # experiment logging backend (via Lightning)
from skimage import filters                         # Otsu thresholding for masks

# ---------------------------
# Project-local imports
# ---------------------------
from src.models.vae.vae_model import (              # your generic encoder/decoder
    GenericEncoder,
    GenericDecoder,
)
from src.metrics.synthesis_metrics import SynthMetrics  # SSIM/PSNR/SAM bundle


class VAE(pl.LightningModule):
    """
    Variational Autoencoder with:
      • L1 + weighted KLD objective,
      • SynthMetrics on reconstructions,
      • ONE generated sample logged **each validation step**,
      • FastGAN-like spectra statistics & plots at validation epoch end,
      • No clamping anywhere (decoder must output ~[0,1] via sigmoid).
    """

    def __init__(
        self,
        # ---- data/model shape ----
        img_channels: int,                               # number of channels (1, 3, or >3 for HSI)
        img_size: int = 256,                             # spatial size (H=W)
        latent_dim: int = 20,                            # bottleneck size

        # ---- optimization ----
        lr: float = 1e-4,                                # learning rate
        betas: Tuple[float, float] = (0.5, 0.999),       # Adam betas
        weight_decay: float = 1e-5,                      # Adam weight decay

        # ---- loss ----
        kld_weight: float = 1e-2,                        # KLD coefficient

        # ---- metrics ----
        metrics: Tuple[str, ...] = ("ssim", "psnr", "sam", 'fid'),  # which SynthMetrics to compute

        # ---- architecture ----
        block: str = "conv",                              # block type in Generic* modules
        model_type: str = "vae",                          # forwarded to Generic* (keep 'vae')

        # ---- validation sampling / plots ----
        val_num_sample_batches: int = 2,                  # how many val batches to scan for spectra
        # per-step logging controls:
        val_log_generated_every_n_steps: int = 1,         # log a generated sample every N val steps
        val_generated_per_step: int = 1,                  # number of generated images per val step (keep it 1)
        # epoch-end grid controls (optional eye-candy):
        val_sample_grid_nz: int = 16,                     # how many latents in the epoch-end grid
        log_sample_grid_nrow: int = 4,                    # rows in the epoch-end grid
    ):
        super().__init__()                                # init Lightning base

        self.save_hyperparameters()                       # store all hparams in checkpoint

        # ========= MODEL (match original constructor signatures) =========
        self.encoder = GenericEncoder(
            (img_channels, img_size, img_size),           # positional (C,H,W) EXACTLY like the original
            latent_dim=latent_dim,
            block_type=block,
            model_type=model_type
        )
        self.decoder = GenericDecoder(
            (img_channels, img_size, img_size),           # positional (C,H,W) EXACTLY like the original
            latent_dim=latent_dim,
            block_type=block,
            model_type=model_type,
            final_activation="sigmoid"                    # outputs ~[0,1]; we never clamp
        )

        # ========= RUNNING METRICS =========
        self.train_loss = MeanMetric()                    # epoch-mean training loss
        self.val_loss   = MeanMetric()                    # epoch-mean validation loss
        self.test_loss  = MeanMetric()                    # epoch-mean test loss
        self.val_loss_best = MinMetric()                  # best validation loss (lowest)

        # ========= VALIDATION QUALITY METRICS =========
        self.val_metrics = SynthMetrics(                  # SSIM/PSNR/SAM on reconstructions
            metrics=list(metrics),
            data_range=1.0
        )
        self.val_best: Dict[str, MaxMetric] = {          # best-so-far trackers for each metric
            name: MaxMetric() for name in self.val_metrics._order
        }

        # ========= FIXED LATENTS (for optional epoch-end grid) =========
        self.register_buffer(
            "fixed_noise",
            torch.randn(self.hparams.val_sample_grid_nz, latent_dim),
            persistent=False
        )

        # ========= SPECTRA BOOKKEEPING (single-class only) =========
        self.lesion_class_name: Optional[str] = None
        self.real_spectra: Optional[Dict[str, List[np.ndarray]]] = None
        self.fake_spectra: Optional[Dict[str, List[np.ndarray]]] = None

    # ---------------------------------------------------------------------
    # Core VAE utilities
    # ---------------------------------------------------------------------
    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Differentiable sample: z = mu + exp(0.5*log_var) * eps."""
        if not torch.is_tensor(mu):
            mu = torch.as_tensor(mu, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(log_var):
            log_var = torch.as_tensor(log_var, device=self.device, dtype=torch.float32)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Encode → reparameterize → decode (no clamping)."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    # ---------------------------------------------------------------------
    # Shared step
    # ---------------------------------------------------------------------
    def _common_step(self, batch, batch_idx: int, split: str) -> Tensor:
        """Compute loss and (if val) metrics; update running means and log per-step scalars."""
        x, _ = batch
        x_hat, mu, log_var = self.forward(x)

        recon = F.l1_loss(x_hat, x)                                 # L1 recon loss
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())  # standard KLD
        loss = recon + self.hparams.kld_weight * kld                # total loss

        # Per-step scalar logging (Lightning->W&B; no custom step numbers here)
        self.log_dict(
            {
                f"{split}_loss": loss,
                f"{split}_recon_loss": recon,
                f"{split}_kld": kld,
            },
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=torch.cuda.device_count() > 1
        )

        # Update epoch running metrics
        if split == "train":
            self.train_loss.update(loss)
        elif split == "val":
            self.val_loss.update(loss)
        else:
            self.test_loss.update(loss)

        # On validation: compute SynthMetrics over the epoch (log on_epoch=True below)
        if split == "val":
            with torch.no_grad():
                m = self.val_metrics(x_hat, x)
            self.log_dict(
                {f"val/{k}": v for k, v in m.items()},
                prog_bar=True,
                logger=True,
                on_step=False,           # aggregate only
                on_epoch=True,
                sync_dist=torch.cuda.device_count() > 1
            )

        return loss

    # ---------------------------------------------------------------------
    # Training / Validation / Test steps
    # ---------------------------------------------------------------------
    def training_step(self, batch, batch_idx: int) -> Tensor:
        """Standard training step; also logs epoch-mean training loss."""
        loss = self._common_step(batch, batch_idx, split="train")
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        """
        Validation step:
          • Run shared loss/metrics.
          • Log ONE **generated** sample (decoder(z)) every N validation steps.
            - No clamping.
            - If C>3, show mean-channel expanded to 3 channels.
            - DO NOT pass a custom `step=` to W&B to avoid "monotonic step" warnings.
        """
        # ---- do the usual validation compute ----
        loss = self._common_step(batch, batch_idx, split="val")

        # ---- per-step generated sample logging (controlled by hparam) ----
        if (batch_idx % self.hparams.val_log_generated_every_n_steps) == 0:
            try:
                # Number of images to log per step (default 1)
                n = int(self.hparams.val_generated_per_step)

                # Sample random latents and decode to images (no clamp)
                z = torch.randn(n, self.hparams.latent_dim, device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    gen = self.decoder(z).detach()                    # (n, C, H, W)

                # If hyperspectral, make a mean-channel RGB visualization
                if gen.size(1) > 3:
                    gen_vis = gen.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)  # (n,3,H,W)
                else:
                    gen_vis = gen

                # Make a 1-row grid with n images (no normalization/scaling)
                grid = make_grid(gen_vis, nrow=n, normalize=False, value_range=None).cpu()

                # IMPORTANT: DO NOT set `step=` here; let WandbLogger decide the step.
                if hasattr(self.logger, "experiment"):
                    self.logger.experiment.log({"val/generated_sample": wandb.Image(grid)})
            except Exception as e:
                self.print(f"[VAE] Per-step generated logging failed at val step {batch_idx}: {e}")

        # Return the loss (Lightning expects it)
        return loss

    def test_step(self, batch, batch_idx: int) -> Tensor:
        """Standard test step; logs epoch-mean test loss."""
        loss = self._common_step(batch, batch_idx, split="test")
        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # ---------------------------------------------------------------------
    # Setup: detect single-class and prepare spectra containers
    # ---------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        """If the dataset is single-class, record the class name and init spectra buckets."""
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None:
            return

        # Pick dataset to inspect depending on stage
        if stage == "fit":
            dataset = getattr(dm, "data_train", None)
        elif stage == "validate":
            dataset = getattr(dm, "data_val", None)
        elif stage in ("test", "predict"):
            dataset = getattr(dm, "data_test", None)
        else:
            dataset = getattr(dm, "data_train", None)
        if dataset is None:
            return

        # Unwrap common wrappers
        base = dataset
        if isinstance(base, torch.utils.data.ConcatDataset):
            base = base.datasets[0]
        if isinstance(base, torch.utils.data.Subset):
            base = base.dataset

        # Attempt to infer single-class
        try:
            train_indices = dm.train_indices
            train_labels = np.array([base.labels[i] for i in train_indices])
            unique = np.unique(train_labels)
            if len(unique) == 1:
                inv = {v: k for k, v in base.labels_map.items()}
                self.lesion_class_name = inv[unique[0]]
                self.real_spectra = {"normal_skin": [], self.lesion_class_name: []}
                self.fake_spectra = {"normal_skin": [], self.lesion_class_name: []}
        except Exception:
            pass  # dataset may not expose these attributes

    # ---------------------------------------------------------------------
    # Validation epoch end: log val_loss (epoch), bests, optional grid, spectra plots
    # ---------------------------------------------------------------------
    def on_validation_epoch_end(self) -> None:
        """
        • Log epoch-level 'val_loss' so EarlyStopping('val_loss') works.
        • Update best-of metrics.
        • (Optional) log a nice fixed-noise grid once per epoch (eye-candy).
        • Compute spectra stats over a few val batches and plot mean spectra (single-class only).
        """
        # ---- epoch-level val_loss for EarlyStopping ----
        v = self.val_loss.compute()
        self.log("val_loss", v, prog_bar=True, logger=True, sync_dist=torch.cuda.device_count() > 1)
        self.val_loss_best.update(v)
        self.log("val/loss_best", self.val_loss_best.compute(), logger=True, prog_bar=True)
        self.val_loss.reset()

        # ---- update best-of for other metrics ----
        for name, mm in self.val_best.items():
            key = f"val/{name}"
            if key in self.trainer.callback_metrics:
                mm.update(self.trainer.callback_metrics[key])
                self.log(f"val/{name}_best", mm.compute(), logger=True)

        # ---- OPTIONAL: epoch-end grid from fixed_noise (no clamp) ----
        try:
            z = self.fixed_noise.to(self.device, dtype=torch.float32)
            with torch.no_grad():
                samples = self.decoder(z).detach()                      # (N, C, H, W)
            if samples.size(1) > 3:
                vis = samples.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            else:
                vis = samples
            grid = make_grid(
                vis,
                nrow=self.hparams.log_sample_grid_nrow,
                normalize=False,
                value_range=None
            ).cpu()
            if hasattr(self.logger, "experiment"):
                # DO NOT set step=...; let WandbLogger keep steps monotonic.
                self.logger.experiment.log({"val/generated_grid_epoch_end": wandb.Image(grid)})
        except Exception as e:
            self.print(f"[VAE] Epoch-end grid logging failed: {e}")

        # ---- spectra stats over a few val batches (single-class only) ----
        if self.lesion_class_name is not None:
            try:
                dm = getattr(self.trainer, "datamodule", None)
                val_loader = None
                if dm is not None:
                    if hasattr(dm, "val_dataloader"):
                        val_loader = dm.val_dataloader()
                    elif hasattr(dm, "train_dataloader"):
                        val_loader = dm.train_dataloader()

                if val_loader is not None:
                    # reset spectra buckets for this epoch
                    self.real_spectra = {"normal_skin": [], self.lesion_class_name: []}
                    self.fake_spectra = {"normal_skin": [], self.lesion_class_name: []}

                    with torch.no_grad():
                        for i, (real_image, _) in enumerate(val_loader):
                            if i >= self.hparams.val_num_sample_batches:
                                break
                            real_image = real_image.to(self.device, non_blocking=True)
                            bsz = real_image.size(0)
                            z = torch.randn(bsz, self.hparams.latent_dim, device=self.device, dtype=torch.float32)
                            fake_image = self.decoder(z)                  # no clamp
                            self.compute_spectra_statistics(real_image, fake_image)

                    self._plot_mean_spectra()                             # log figure to W&B
            except Exception as e:
                self.print(f"[VAE] Spectra stats/plotting failed: {e}")

    # ---------------------------------------------------------------------
    # Spectra utilities (FastGAN-like), adapted here
    # ---------------------------------------------------------------------
    def compute_spectra_statistics(self, real_tensor: Tensor, fake_tensor: Tensor) -> None:
        """Accumulate mean spectra inside/outside Otsu mask for both real and generated images."""
        def _accumulate(batch: Tensor, bucket: Dict[str, List[np.ndarray]]) -> None:
            for b in range(batch.size(0)):
                im = batch[b].detach().float().cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
                mean_im = im.mean(axis=-1)                                        # grayscale proxy
                try:
                    t = filters.threshold_otsu(mean_im)                           # Otsu threshold
                    lesion = mean_im < (t * 1.0)                                  # lesion region
                    if np.any(lesion):
                        bucket[self.lesion_class_name].append(im[lesion].mean(axis=0))   # (C,)
                    bg = ~lesion                                                 # background region
                    if np.any(bg):
                        bucket["normal_skin"].append(im[bg].mean(axis=0))        # (C,)
                except Exception:
                    continue  # robust to degenerate masks

        if self.real_spectra is not None:
            _accumulate(real_tensor, self.real_spectra)
        if self.fake_spectra is not None:
            _accumulate(fake_tensor, self.fake_spectra)

    def _plot_mean_spectra(self) -> None:
        """Plot mean±std spectra for 'normal_skin' and lesion class; log to W&B."""
        import matplotlib.pyplot as plt

        if self.real_spectra is None or self.fake_spectra is None:
            return

        labels = ["normal_skin", self.lesion_class_name]
        real_stats: Dict[str, Dict[str, np.ndarray]] = {}
        fake_stats: Dict[str, Dict[str, np.ndarray]] = {}

        for lbl in labels:
            if lbl is None:
                continue
            if self.real_spectra.get(lbl):
                arr = np.asarray(self.real_spectra[lbl], dtype=np.float32)
                real_stats[lbl] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}
            if self.fake_spectra.get(lbl):
                arr = np.asarray(self.fake_spectra[lbl], dtype=np.float32)
                fake_stats[lbl] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0)}

        ref = real_stats or fake_stats
        ref_lbl = next((k for k in labels if k and (k in ref)), None)
        if ref_lbl is None:
            return

        n_bands = int(len(ref[ref_lbl]["mean"]))
        x = np.arange(1, n_bands + 1)

        n_panels = sum(int(lbl is not None) for lbl in labels)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        ax_i = 0
        for lbl in labels:
            if lbl is None:
                continue
            ax = axes[ax_i]; ax_i += 1
            rs = real_stats.get(lbl); fs = fake_stats.get(lbl)
            if rs is None and fs is None:
                ax.text(0.5, 0.5, f"No data for {lbl}", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            if rs is not None:
                ax.plot(x, rs["mean"], linewidth=2.5, label="Real")
                ax.fill_between(x, rs["mean"] - rs["std"], rs["mean"] + rs["std"], alpha=0.15)
            if fs is not None:
                ax.plot(x, fs["mean"], linewidth=2.0, linestyle="--", label="Synthetic")
                ax.fill_between(x, fs["mean"] - fs["std"], fs["mean"] + fs["std"], alpha=0.15)
            ax.set_title(lbl.replace("_", " ").title())
            ax.set_xlabel("Spectral Band")
            ax.set_ylabel("Reflectance")
            ax.grid(True, alpha=0.25)
            ax.legend()

        plt.suptitle("Mean Spectra: Real vs Synthetic", y=1.02)
        plt.tight_layout()
        if hasattr(self.logger, "experiment"):
            # No custom step here — avoid out-of-order warnings.
            self.logger.experiment.log({"val/mean_spectra": wandb.Image(fig)})
        plt.close(fig)

    # ---------------------------------------------------------------------
    # Optional utility for logging arbitrary image tensors without clamping
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _log_images(self, images: Tensor, name: str, nrow: int = 4) -> None:
        """Helper: log an image grid; HSI → mean-channel RGB for visualization."""
        if images.size(1) > 3:
            images = images.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        grid = make_grid(images, nrow=nrow, normalize=False, value_range=None).cpu()
        if hasattr(self.logger, "experiment"):
            # Do not pass `step=...`.
            self.logger.experiment.log({name: [wandb.Image(grid)]})

    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------
    def configure_optimizers(self):
        """Adam optimizer with provided LR/betas/weight_decay."""
        return Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=tuple(self.hparams.betas),
            weight_decay=self.hparams.weight_decay
        )
