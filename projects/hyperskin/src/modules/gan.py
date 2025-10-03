from pytorch_lightning import LightningModule
import torch
import torchvision
from zmq import has

from src.models.gan.discriminator import Discriminator
from src.models.gan.generator import Generator
import torch.nn.functional as F
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import SpectralAngleMapper
from torchmetrics import MaxMetric
from metrics.synthesis_metrics import SynthMetrics, _NoOpMetric


class GANModule(LightningModule):
    """
    >>> GAN(img_shape=(1, 8, 8))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GAN(
      (generator): Generator(
        (model): Sequential(...)
      )
      (discriminator): Discriminator(
        (model): Sequential(...)
      )
    )
    """

    def __init__(
        self,
        img_shape: tuple = (1, 28, 28),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
        metrics: list = ['ssim', 'psnr', 'sam']
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
        self.val_metrics = SynthMetrics(metrics=metrics, data_range=1.0)
        self.val_best = {name: MaxMetric() for name in self.val_metrics._order}
        
    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch):
        imgs, _ = batch

        opt_g, opt_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # Train generator
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        self.toggle_optimizer(opt_g)
        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # Train discriminator
        # Measure discriminator's ability to classify real from generated samples
        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        self.toggle_optimizer(opt_d)
        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.log_dict({"d_loss": d_loss, "g_loss": g_loss})

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return opt_g, opt_d

    def on_train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        for logger in self.loggers:
            if hasattr(logger.experiment, "add_image"):
                logger.experiment.add_image("generated_images", grid, self.current_epoch)

    def validation_step(self, batch, batch_idx):
            imgs, _ = batch
            fake_imgs = self(torch.randn(imgs.shape[0], self.hparams.latent_dim, device=imgs.device))
            results = self.val_metrics(fake_imgs, imgs)
            for k, v in results.items():
                self.log(f"val/{k}", v, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """
        End-of-epoch:
          • compute epoch-aggregated metrics from SynthMetrics
          • update 'best_*' MaxMetric trackers
          • log current epoch values and best-so-far values
          • reset SynthMetrics state for the next epoch
        """
        # 1) Compute current epoch metrics as a dict, e.g., {'ssim': t, 'psnr': t, 'sam': t}
        epoch_vals = self.val_metrics.compute()

        # 2) Log current (this-epoch) values and update best trackers
        for name, value in epoch_vals.items():
            # Log the epoch aggregate (e.g., mean across all batches)
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=True)

            # Update best-so-far tracker and log its current best value
            self.val_best[name](value)                                  # push this epoch value into MaxMetric
            self.log(f"val/{name}_best", self.val_best[name].compute(), on_step=False, on_epoch=True)