import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from torch import Tensor
from torch.nn.functional import l1_loss
from torch.optim import Adam
from torchinfo import summary
from torchvision.utils import make_grid
from torchmetrics import MeanMetric, MinMetric, MaxMetric
from src.models.vae.vae_model import Encoder, Decoder, GenericDecoder, GenericEncoder
from src.metrics.synthesis_metrics import SynthMetrics

class VAE(pl.LightningModule):
    def __init__(
        self,
        img_channels: int,
        img_size=256,
        latent_dim: int = 20,
        lr: float = 1e-4,
        betas: list = [0.5, 0.999],
        weight_decay: float = 1e-5,
        kld_weight: float = 1e-2,
        metrics: list = ['ssim', 'psnr', 'sam'],
        block:str ='conv',
    ):
        super().__init__()
        self.save_hyperparameters()  # keep YAML-compatible

        #self.encoder = Encoder(img_channels=img_channels, img_size=img_size, latent_dim=latent_dim)
        #self.decoder = Decoder(img_channels=img_channels, img_size=img_size, latent_dim=latent_dim)
        self.encoder = GenericEncoder((img_channels,img_size,img_size),latent_dim=latent_dim,block_type=block,model_type='vae')
        self.decoder = GenericDecoder((img_channels,img_size,img_size),latent_dim=latent_dim,block_type=block,model_type='vae',final_activation='sigmoid')

        
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        self.test_loss  = MeanMetric()
        self.val_loss_best = MinMetric()

        self.val_metrics = SynthMetrics(metrics=metrics, data_range=1.0)
        self.val_best = {name: MaxMetric() for name in self.val_metrics._order}
        

        # Precompute a latent bank lazily (device-agnostic here; move later)
        self._latent_bank = torch.randn(16, latent_dim)

        # DO **NOT** call summary() or log anything from __init__.
        # Any exception here will bubble up to jsonargparse and cause your Union error.

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def _common_step(self, batch, batch_idx: int, split: str):
        x, c = batch
        x_hat, mu, log_var = self(x)
        recon = l1_loss(x_hat, x)
        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon + self.hparams.kld_weight * kld
        
        self.log_dict({f"{split}_loss": loss,
                       f"{split}_recon_loss": recon,
                       f"{split}_kld": kld},
                      prog_bar=True, logger=True, sync_dist=torch.cuda.device_count() > 1)

        # val/test-only: compute image metrics on reconstructions (not random samples)
        if not self.training:
            results = self.val_metrics(x_hat, x)
            self.log_dict({f"val/{k}": v for k, v in results.items()},
                          prog_bar=True, logger=True, sync_dist=torch.cuda.device_count() > 1)
        if split == "train":
            self.train_loss.update(loss)
        elif split == "val":
            self.val_loss.update(loss)
        else:
            self.test_loss.update(loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "train")
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        v = self.val_loss.compute()
        self.val_loss_best.update(v)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
        # update best of image metrics if present
        for name, mm in self.val_best.items():
            key = f"val/{name}"
            if key in self.trainer.callback_metrics:
                mm.update(self.trainer.callback_metrics[key])
                self.log(f"val/{name}_best", mm.compute())

        self.val_loss.reset()

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "test")
        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(),
                    lr=self.hparams.lr,
                    betas=tuple(self.hparams.betas),
                    weight_decay=self.hparams.weight_decay)

    @torch.no_grad()
    def _log_images(self, images: Tensor, name: str):
        grid = make_grid(images, value_range=(-1, 1), normalize=True)
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log({name: [wandb.Image(grid)]}, step=self.global_step)
