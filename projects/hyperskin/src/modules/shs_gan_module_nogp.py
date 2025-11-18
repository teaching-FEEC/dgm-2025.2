from pathlib import Path
import pytorch_lightning as pl
import torch
import torchvision
import torch.nn.functional as F

from torchmetrics import MeanMetric, MinMetric, MaxMetric
from pytorch_lightning.loggers import WandbLogger

# ---- project modules (unchanged) ----
from src.models.shs_gan.shs_discriminator import Critic3D          # keep calling it "critic"
from src.models.shs_gan.shs_generator import Generator
from src.data_modules.hsi_dermoscopy import HSIDermoscopyDataModule
from src.metrics.synthesis_metrics import SynthMetrics, _NoOpMetric


class SHSGAN_NoGP(pl.LightningModule):
    """
    SHS-GAN LightningModule **without** gradient penalty.
    Losses follow the "classic" GAN (BCE with logits) as in `gan.py`.

    Key changes vs the original WGAN-GP version:
      1) Removed gradient penalty (no Lipschitz constraint).
      2) Use Binary Cross-Entropy with logits for both generator and critic.
      3) Default to a single critic update per generator step (n_critic=1),
         though we keep the argument for compatibility (you can set >1 if you want).
      4) Kept names, logging, and validation metrics as close as possible.
    """

    def __init__(
        self,
        in_channels: int = 3,                 # noise "image" channels fed to the generator
        out_channels: int = 16,               # hyperspectral channels generated
        img_size: tuple = (256, 256),         # H, W of the noise "image"
        base_filters: int = 64,               # generator base width
        critic_fft_arm: bool = True,          # Critic3D option (unchanged)
        lr: float = 2e-4,                     # Adam lr (same spirit as gan.py defaults)
        betas: tuple = (0.5, 0.999),          # Adam betas (gan.py uses b1=0.5, b2=0.999)
        n_critic: int = 1,                    # keep param for compatibility (can be >1)
        num_log_samples: int = 2,             # samples to log at epoch end
        log_channels: tuple = (0, 1, 2),      # which generated HSI channels to log as grid
        metrics: list = ('ssim', 'psnr', 'sam')  # validation image quality metrics
    ):
        super().__init__()
        # save_hyperparameters keeps most original field names
        self.save_hyperparameters()
        # we will drive the optimization steps manually, like the original shsgan
        self.automatic_optimization = False

        # ---- Networks (unchanged names) ----
        # Generator maps noise "image" (B, in_channels, H, W) -> HSI (B, out_channels, H, W)
        self.generator = Generator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters
        )

        # Critic evaluates realism of the HSI (keep name "critic")
        # IMPORTANT: For BCEWithLogitsLoss we expect **logits** from the critic (no sigmoid inside).
        self.critic = Critic3D(in_channels=out_channels, fft_arm=critic_fft_arm)

        # ---- Running metrics (kept structure; removed gp-related metrics) ----
        # training
        self.train_g_loss = MeanMetric()
        self.train_d_loss = MeanMetric()
        # validation
        self.val_g_loss = MeanMetric()
        self.val_d_loss = MeanMetric()
        # testing
        self.test_g_loss = MeanMetric()
        self.test_d_loss = MeanMetric()

        # Track best generator loss on val set
        self.val_g_loss_best = MinMetric()

        # High-level synthesis metrics for validation (unchanged)
        self.val_metrics = SynthMetrics(metrics=metrics, data_range=1.0)
        self.val_best = {name: MaxMetric() for name in self.val_metrics._order}

        # cache H, W for noise sampling convenience
        self._H, self._W = self.hparams.img_size

    # ------------------------------------------------------------------------
    # Helper functions (clear & isolated; easy to follow and test)
    # ------------------------------------------------------------------------

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        """
        Create a spatial "noise image" shaped like the generator input:
          z ~ N(0, I) with shape (B, in_channels, H, W)

        We keep the same *interface* as the original shsgan (noise is image-shaped),
        but switch the underlying losses to BCE like in gan.py.
        """
        # Allocate z on the current device to avoid device mismatches
        return torch.randn(
            batch_size,
            self.hparams.in_channels,
            self._H,
            self._W,
            device=self.device
        )

    @staticmethod
    def _bce_with_logits(pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Standard Binary Cross-Entropy with logits (numerically stable).
        Mirrors `gan.py`'s F.binary_cross_entropy_with_logits usage.
        """
        return F.binary_cross_entropy_with_logits(pred_logits, targets)

    def _make_targets(self, batch_size: int, is_real: bool, like: torch.Tensor) -> torch.Tensor:
        """
        Create label tensors for BCE training:
          • real targets = 1
          • fake targets = 0
        Shapes follow (B, 1) like in gan.py. We map them to the same device/dtype as 'like'.
        """
        val = 1.0 if is_real else 0.0
        t = torch.full((batch_size, 1), fill_value=val, device=like.device, dtype=like.dtype)
        return t

    def _critic_step(self, real_hsi: torch.Tensor) -> torch.Tensor:
        """
        One critic (discriminator) update using BCE (no gradient penalty).

        Logic matches gan.py:
          D_loss = 0.5 * [ BCE(D(real), 1) + BCE(D(fake.detach()), 0) ]
        """
        # Get optimizers
        opt_g, opt_c = self.optimizers()

        # Put critic optimizer under Lightning's toggle (keeps hooks/scopes tidy)
        self.toggle_optimizer(opt_c)

        # ---- real pass ----
        # The critic should output logits. Targets are 'real' (1s).
        bsz = real_hsi.size(0)
        real_logits = self.critic(real_hsi)
        real_targets = self._make_targets(bsz, is_real=True, like=real_hsi)
        real_loss = self._bce_with_logits(real_logits, real_targets)

        # ---- fake pass ----
        # Sample fresh noise and create *detached* fakes (do not backprop to G here).
        z = self._sample_noise(bsz)
        fake_hsi = self.generator(z).detach()
        fake_logits = self.critic(fake_hsi)
        fake_targets = self._make_targets(bsz, is_real=False, like=real_hsi)
        fake_loss = self._bce_with_logits(fake_logits, fake_targets)

        # ---- combine losses and step ----
        d_loss = 0.5 * (real_loss + fake_loss)

        opt_c.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_c.step()

        self.untoggle_optimizer(opt_c)
        return d_loss

    def _generator_step(self, batch_size: int, ref_tensor: torch.Tensor) -> torch.Tensor:
        """
        One generator update using BCE:

          G_loss = BCE(D(G(z)), 1)

        This pushes the generator to produce samples the critic classifies as real.
        """
        # Get optimizers
        opt_g, opt_c = self.optimizers()

        # Put generator optimizer under toggle scope
        self.toggle_optimizer(opt_g)

        # Sample noise and generate fakes
        z = self._sample_noise(batch_size)
        fake_hsi = self.generator(z)

        # Critic prediction on *non-detached* fakes (so gradients flow into G)
        fake_logits = self.critic(fake_hsi)

        # Targets are "real"
        targets = self._make_targets(batch_size, is_real=True, like=ref_tensor)
        g_loss = self._bce_with_logits(fake_logits, targets)

        # Backprop + step
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        self.untoggle_optimizer(opt_g)
        return g_loss

    # ------------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------------

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward is just the generator; this mirrors gan.py's interface.
        """
        return self.generator(z)

    def on_train_start(self):
        """
        Same behavior as original:
          • reset best val generator loss
          • if the datamodule is HSIDermoscopyDataModule, save the splits locally
          • and upload to W&B if available
        """
        self.val_g_loss_best.reset()

        if self.trainer.logger and hasattr(self.trainer.logger, 'save_dir') and \
           isinstance(self.trainer.datamodule, HSIDermoscopyDataModule):
            logger = self.trainer.logger
            datamodule = self.trainer.datamodule

            # Save splits locally for reproducibility
            split_dir = Path(logger.save_dir) / "data_splits"
            datamodule.save_splits_to_disk(split_dir)

            # Upload splits to W&B and auto-name the run
            if isinstance(logger, WandbLogger):
                run = logger.experiment
                new_nm = f"shsgan-nogp-f{self.hparams.base_filters}_fft-{self.hparams.critic_fft_arm}"
                run.name = new_nm
                run.notes = "Auto-named by on_train_start (no-GP BCE-GAN)"
                run.tags = list(set(run.tags or []).union({"gan", "hsi", "bce", "nogp"}))
                run.save(str(split_dir / "*.txt"), base_path=logger.save_dir)

    def _gan_step(self, batch, stage: str):
        """
        One *full* GAN iteration:
          • n_critic critic updates (optionally >1; default 1)
          • 1 generator update

        We keep this function name similar to the original `gan_step`,
        but it now implements BCE-style training (no gradient penalty).
        """
        imgs, _ = batch                      # imgs = real HSI: (B, out_channels, H, W)
        bsz = imgs.size(0)

        # ---- Critic updates ----
        d_loss = None
        for _ in range(self.hparams.n_critic):
            d_loss = self._critic_step(real_hsi=imgs)  # returns the last step's loss

        # ---- Generator update ----
        g_loss = self._generator_step(batch_size=bsz, ref_tensor=imgs)

        # ---- Update running metrics for the chosen stage ----
        if stage == "train":
            self.train_g_loss.update(g_loss.detach())
            self.train_d_loss.update(d_loss.detach())
        elif stage == "val":
            self.val_g_loss.update(g_loss.detach())
            self.val_d_loss.update(d_loss.detach())
        elif stage == "test":
            self.test_g_loss.update(g_loss.detach())
            self.test_d_loss.update(d_loss.detach())

        return g_loss, d_loss

    def training_step(self, batch, batch_idx):
        """
        Training step mirrors gan.py:
          1) update generator (once)
          2) update critic (we do critic first internally in _gan_step)
          3) log scalar losses
        """
        g_loss, d_loss = self._gan_step(batch, stage="train")
        # log epoch-averaged meters (Lightning will call .compute() as needed)
        self.log("train/g_loss", self.train_g_loss, prog_bar=True, on_epoch=True)
        self.log("train/d_loss", self.train_d_loss, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        """
        Validation mirrors original behavior:
          • run a GAN step (so losses get logged/updated)
          • compute synthesis metrics comparing generated vs real
        """
        # allow grads (similar to original val) but not strictly necessary; okay either way
        with torch.enable_grad():
            g_loss, d_loss = self._gan_step(batch, stage="val")

        imgs, _ = batch
        B, _, H, W = imgs.shape

        # Make fresh fakes for metrics (decouples from training step, clearer logging)
        fake_imgs = self.generator(self._sample_noise(B).to(imgs.device))

        results = self.val_metrics(fake_imgs, imgs)

        # Log per-batch metrics under "val/*"
        log_dict = {
            "val_g_loss": g_loss.detach(),
            "val_d_loss": d_loss.detach(),
            **{f"val/{k}": v for k, v in results.items()}
        }
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """
        Keep same pattern as gan.py:
          • compute epoch-aggregated metrics
          • update best trackers
        """
        g_loss_epoch = self.val_g_loss.compute()
        self.val_g_loss_best.update(g_loss_epoch)
        self.log("val/g_loss_best", self.val_g_loss_best.compute(), prog_bar=True, on_step=False, on_epoch=True)

        # Also mirror gan.py's handling of SynthMetrics epoch compute
        epoch_vals = self.val_metrics.compute()
        for name, value in epoch_vals.items():
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True)
            self.val_best[name](value)
            self.log(f"val/{name}_best", self.val_best[name].compute(), on_step=False, on_epoch=True)

        # IMPORTANT: reset val_metrics internal state for next epoch (if SynthMetrics is stateful)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """
        Optional: a test pass that mirrors train/val logging for completeness.
        """
        g_loss, d_loss = self._gan_step(batch, stage="test")
        self.log("test/g_loss", self.test_g_loss, prog_bar=True, on_epoch=True)
        self.log("test/d_loss", self.test_d_loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        """
        Optimizers mirror gan.py (Adam with (betas) = (0.5, 0.999) by default).
        We keep the same two-optimizer return signature as original shsgan.
        """
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=self.hparams.betas)
        opt_c = torch.optim.Adam(self.critic.parameters(),   lr=self.hparams.lr, betas=self.hparams.betas)
        return opt_g, opt_c

    def on_train_epoch_end(self):
        """
        Same visualization pattern as original:
         • sample B noise images
         • generate HSI
         • select a few channels to form a grid and log it to all supported loggers
        """
        B = self.hparams.num_log_samples
        C = self.hparams.in_channels
        H, W = self.hparams.img_size

        # noise "rgb" here just means 3 channels if in_channels==3; it's arbitrary noise
        z = torch.randn(B, C, H, W, device=self.device).type_as(next(self.generator.parameters()))
        fake_hsi = self.generator(z)

        # select channels that exist
        log_channels = [c for c in self.hparams.log_channels if c < fake_hsi.size(1)]
        img_to_log = fake_hsi[:, log_channels, :, :] if len(log_channels) > 0 else fake_hsi[:, :1, :, :]

        # make and send grid
        grid = torchvision.utils.make_grid(img_to_log, normalize=True)
        for logger in self.loggers:
            if hasattr(logger.experiment, "add_image"):
                logger.experiment.add_image("generated_hsi_rgb", grid, self.current_epoch)
