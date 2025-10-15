
from typing import List, Tuple
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torchinfo import summary
from torchvision.utils import make_grid
from torchmetrics import MeanMetric, MinMetric
from pytorch_lightning.loggers import WandbLogger

from src.models.vae.vae_model import Encoder, Decoder


from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import SpectralAngleMapper
from torchmetrics import MaxMetric
from src.metrics.synthesis_metrics import SynthMetrics, _NoOpMetric


class VAE(pl.LightningModule):
    """
    Variational Autoencoder (VAE): Auto-Encoding Variational Bayes.
    https://arxiv.org/abs/1312.6114

    VAE is a generative model that learns a probabilistic mapping between the input data
    space and a latent space. The encoder maps an input to a distribution in the latent space,
    and the decoder maps points in the latent space back to the data space. VAE introduces a
    regularization term to ensure that the learned latent space is continuous, making it suitable
    for generative tasks.
    """

    def __init__(
        self,
        img_channels: int,
        img_size=(256, 256),
        latent_dim: int = 20,
        lr: float = 1e-4,
        betas: list =[0.5, 0.999],
        weight_decay: float = 1e-5,
        kld_weight: float = 1e-2,
        metrics: list = ['ssim', 'psnr', 'sam'],
    ):
        super(VAE, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(
            img_channels=img_channels, img_size=img_size, latent_dim=latent_dim
        )
        self.decoder = Decoder(
            img_channels=img_channels, img_size=img_size, latent_dim=latent_dim
        )

         # ---- Metrics ----
        #training
        self.train_loss = MeanMetric()
      
        #validation
        self.val_loss = MeanMetric()
        
        #testing
        self.test_loss = MeanMetric()
        
        # track best generator loss
        self.val_loss_best = MinMetric()

        self.val_metrics = SynthMetrics(metrics=metrics, data_range=1.0)
        self.val_best = {name: MaxMetric() for name in self.val_metrics._order}

        self.z = torch.randn([16, latent_dim])
        self.summary()

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from the latent Gaussian distribution.

        This is a key component of VAEs, enabling backpropagation through the random sampling step.

        Args:
            mu (Tensor): Mean of the Gaussian distribution.
            log_var (Tensor): Log variance of the Gaussian distribution.

        Returns:
            Tensor: Sampled latent variable.
        """
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def _common_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, split: str
    ) -> Tensor:
        assert split in ["train", "val", "test"]
        x, c = batch
        x_hat, mu, log_var = self(x)

        recon_loss = l1_loss(x_hat, x)
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + self.hparams.kld_weight * kld

        self.log_dict(
            {
                f"{split}_loss": loss,
                f"{split}_recon_loss": recon_loss,
                f"{split}_kld": kld,
            },
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )

        if not self.training:
            """Log and cache latent variables for visualization."""
            if batch_idx == 0:
                self._log_images(self.decoder(self.z.to(self.device)) ,
                    "Random Generation",
                )

                self.latents = []
                self.conds = []

            z = self.reparameterize(mu, log_var).detach().cpu()
            c = c.cpu()

            self.latents.append(z)
            self.conds.append(c)

            fake_imgs = self.decoder(self.z.to(self.device))
            results = self.val_metrics(fake_imgs, x)

            self.log_dict({
                **{f"val/{k}": v for k, v in results.items()}
            }, prog_bar=True)

        # ---- Update metrics ----
        if split == "train":
            self.train_loss.update(loss)
            
        elif split == "val":
            self.val_loss.update(loss)
            
        elif split == "test":
            self.test_loss.update(loss)

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self._common_step(batch, batch_idx, "train")
        self.log("train/loss", self.train_loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._common_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        g_loss = self.val_loss.compute()
        self.val_best.update(g_loss)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

        self._log_latent_embeddings()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.b1, self.hparams.b2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    @torch.no_grad()
    def _log_images(self, images: Tensor, fig_name: str):
        fig = make_grid(
            tensor=images,
            value_range=(-1, 1),
            normalize=True,
        )
        self.logger.experiment.log(
            {fig_name: [wandb.Image(fig)]}, step=self.global_step
        )

    @torch.no_grad()
    def _log_latent_embeddings(self):
        """Log the latent space embeddings to WandB for visualization."""
        z = torch.cat(self.latents)
        c = torch.cat(self.conds)
        data = {f"z_{i}": z[:, i] for i in range(z.size(1))}
        data["c"] = c
        data = pd.DataFrame(data)
        self.logger.experiment.log(
            {"latent space": wandb.Table(data=data)}, step=self.global_step
        )

        del self.latents
        del self.conds

    def summary(
        self,
        col_names: List[str] = [
            "input_size",
            "output_size",
            "num_params",
            "params_percent",
            "kernel_size",
            "mult_adds",
            "trainable",
        ],
    ):
        x = torch.randn(
            [
                1,
                self.hparams.img_channels,
                self.hparams.img_size,
                self.hparams.img_size,
            ]
        )

        summary(
            self,
            input_data=x,
            col_names=col_names,
        )