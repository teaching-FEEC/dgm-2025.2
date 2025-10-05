from pathlib import Path
import pytorch_lightning as pl
import torch
import torchvision
from torch import autograd
from torchmetrics import MeanMetric, MinMetric
from pytorch_lightning.loggers import WandbLogger

from src.models.shs_gan.shs_discriminator import Critic3D
from src.models.shs_gan.shs_generator import Generator
from src.data_modules.hsi_dermoscopy import HSIDermoscopyDataModule


from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import SpectralAngleMapper
from torchmetrics import MaxMetric
from src.metrics.synthesis_metrics import SynthMetrics, _NoOpMetric
'''In WGAN-GP, the critic must satisfy the 1-Lipschitz constraint, 
meaning its output cannot change faster than the input, which ensures the 
Wasserstein distance is valid. Instead of clipping weights, WGAN-GP enforces 
this through a gradient penalty, which softly pushes the gradient norm of the 
critic with respect to interpolated samples toward 1, keeping the critic smooth and stable. 
Additionally, the critic is usually updated multiple times per generator update (n_critics) 
so it can reliably approximate the Wasserstein distance before the generator adapts. 
Together, these techniques stabilize GAN training, prevent gradient explosions, 
and help avoid mode collapse.
'''

class SHSGAN(pl.LightningModule):
    def __init__(self,
                 in_channels=3,
                 out_channels=16,
                 img_size=(256, 256),
                 base_filters=64,
                 critic_fft_arm=True,
                 lr=0.0002,
                 betas=(0.5, 0.999),
                 lambda_gp=10.0,
                 n_critic=5,
                 num_log_samples=2,
                 log_channels=(0, 1, 2),
                 metrics: list = ['ssim', 'psnr', 'sam'],
                 ):
        """
        SHS-GAN LightningModule with WGAN-GP 
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Generator maps input image/noise -> hyperspectral cube
        self.generator = Generator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters
        )

        # Critic evaluates hyperspectral cube realism
        self.critic = Critic3D(in_channels=out_channels, fft_arm=critic_fft_arm)

        # ---- Metrics ----
        #training
        self.train_g_loss = MeanMetric()
        self.train_d_loss = MeanMetric()
        self.train_gp     = MeanMetric()
        #validation
        self.val_g_loss = MeanMetric()
        self.val_d_loss = MeanMetric()
        self.val_gp     = MeanMetric()
        #testing
        self.test_g_loss = MeanMetric()
        self.test_d_loss = MeanMetric()
        self.test_gp     = MeanMetric()

        # track best generator loss
        self.val_g_loss_best = MinMetric()

        self.val_metrics = SynthMetrics(metrics=metrics, data_range=1.0)
        self.val_best = {name: MaxMetric() for name in self.val_metrics._order}

    def forward(self, z):
        return self.generator(z)

    def on_train_start(self):
        # This hook runs once at the beginning of training. 
        # It resets the best validation metric and, if the datamodule is HSIDermoscopyDataModule,
        #  it saves the train/val/test splits to disk for reproducibility. 
        self.val_g_loss_best.reset()

        if self.trainer.logger and hasattr(self.trainer.logger, 'save_dir') and \
           isinstance(self.trainer.datamodule, HSIDermoscopyDataModule):
            logger = self.trainer.logger
            datamodule = self.trainer.datamodule

            # Save splits locally
            split_dir = Path(logger.save_dir) / "data_splits"
            datamodule.save_splits_to_disk(split_dir)

            # Upload splits to W&B if active
            if isinstance(logger, WandbLogger):
                run = logger.experiment
                new_nm = f"shsgan-f{self.hparams.base_filters}_fft-{self.hparams.critic_fft_arm}"
                run.name = new_nm
                run.notes = "Auto-named by on_train_start"
                run.tags = list(set(run.tags or []).union({"gan", "hsi", "auto"}))
                run.save(str(split_dir / "*.txt"), base_path=logger.save_dir)


    def calculate_gradient_penalty(self, real_images, fake_images):
        #This implements the WGAN-GP gradient penalty. To satisfy the Lipschitz constraint, 
        # we interpolate between real and fake samples, pass them through the critic, 
        #compute gradients with respect to the interpolated samples, and penalize deviations of gradient norms from 1.
        B = real_images.size(0)

        eta = torch.rand(B, 1, 1, 1, device=real_images.device).expand_as(real_images)
        interpolated = eta * real_images + (1 - eta) * fake_images
        interpolated.requires_grad_(True)   

        prob_interpolated = self.critic(interpolated)

        gradients = autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,            
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.reshape(B, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean() * self.hparams.lambda_gp
        return gp


    def gan_step(self, batch, stage: str):
        hsi, _ = batch
        B, _, H, W = hsi.shape
        opt_g, opt_c = self.optimizers()

        # ---- Train Critic multiple times ----
        for _ in range(self.hparams.n_critic):
            z = torch.randn(B, self.hparams.in_channels, H, W, device=self.device)

            #Two versions of fake_hsi
            fake_hsi_detached = self.generator(z).detach()  # for critic loss
            fake_hsi = self.generator(z)                    # for GP 

            real_score = self.critic(hsi)
            fake_score = self.critic(fake_hsi_detached)

            # Enable grad in val/test
            with torch.enable_grad():
                gp = self.calculate_gradient_penalty(hsi, fake_hsi)

            d_loss = torch.mean(fake_score) - torch.mean(real_score) + gp #Wasserstein min–max game: the critic approximates Wasserstein distance, and the generator minimizes it.

            opt_c.zero_grad()
            self.manual_backward(d_loss)
            opt_c.step()

        # ---- Train Generator ----
        z = torch.randn(B, self.hparams.in_channels, H, W, device=self.device)
        fake_hsi = self.generator(z)
        g_loss = -torch.mean(self.critic(fake_hsi))

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # ---- Update metrics ----
        if stage == "train":
            self.train_g_loss.update(g_loss)
            self.train_d_loss.update(d_loss)
            self.train_gp.update(gp)
        elif stage == "val":
            self.val_g_loss.update(g_loss)
            self.val_d_loss.update(d_loss)
            self.val_gp.update(gp)
        elif stage == "test":
            self.test_g_loss.update(g_loss)
            self.test_d_loss.update(d_loss)
            self.test_gp.update(gp)

        return g_loss, d_loss, gp



    def training_step(self, batch, batch_idx):
        g_loss, d_loss, gp = self.gan_step(batch, stage="train")
        self.log("train/g_loss", self.train_g_loss, prog_bar=True, on_epoch=True)
        self.log("train/d_loss", self.train_d_loss, prog_bar=True, on_epoch=True)
        self.log("train/gp", self.train_gp, prog_bar=True, on_epoch=True)

    # def validation_step(self, batch, batch_idx):
    #     with torch.enable_grad():   
    #         g_loss, d_loss, gp = self.gan_step(batch, stage="val")
    #     self.log_dict({"val_g_loss": g_loss, "val_d_loss": d_loss, "val_gp": gp}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        #GAN training metrics
        with torch.enable_grad():   
            g_loss, d_loss, gp = self.gan_step(batch, stage="val")
        
        # Image quality metrics
        # imgs, _ = batch
        # fake_imgs = self(torch.randn(imgs.shape[0], self.hparams.latent_dim, device=imgs.device))
        # results = self.val_metrics(fake_imgs, imgs)
        imgs, _ = batch
        B, _, H, W = imgs.shape
        fake_imgs = self.generator(torch.randn(B, self.hparams.in_channels, H, W, device=imgs.device))
        results = self.val_metrics(fake_imgs, imgs)


        self.log_dict({
            "val_g_loss": g_loss, 
            "val_d_loss": d_loss, 
            "val_gp": gp,
            **{f"val/{k}": v for k, v in results.items()}
        }, prog_bar=True)
    def on_validation_epoch_end(self):
        g_loss = self.val_g_loss.compute()
        self.val_g_loss_best.update(g_loss)
        self.log("val/g_loss_best", self.val_g_loss_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        g_loss, d_loss, gp = self.gan_step(batch, stage="test")
        self.log("test/g_loss", self.test_g_loss, prog_bar=True, on_epoch=True)
        self.log("test/d_loss", self.test_d_loss, prog_bar=True, on_epoch=True)
        self.log("test/gp", self.test_gp, prog_bar=True, on_epoch=True)


    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=self.hparams.lr, betas=self.hparams.betas)
        opt_c = torch.optim.Adam(self.critic.parameters(),
                                 lr=self.hparams.lr, betas=self.hparams.betas)
        return opt_g, opt_c


    def on_train_epoch_end(self):
        B = self.hparams.num_log_samples
        C = self.hparams.in_channels
        H, W = self.hparams.img_size

        rgb = torch.randn(B, C, H, W).type_as(next(self.generator.parameters()))
        fake_hsi = self.generator(rgb)

        log_channels = [c for c in self.hparams.log_channels if c < fake_hsi.size(1)]
        img_to_log = fake_hsi[:, log_channels, :, :]
        grid = torchvision.utils.make_grid(img_to_log, normalize=True)

        for logger in self.loggers:
            if hasattr(logger.experiment, "add_image"):
                logger.experiment.add_image("generated_hsi_rgb", grid, self.current_epoch)
