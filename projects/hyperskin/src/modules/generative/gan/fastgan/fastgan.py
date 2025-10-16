import torch
import torch.optim as optim
import torch.nn.functional as F

import random

import pytorch_lightning as pl

import warnings

import torchvision

from src.losses.lpips import PerceptualLoss
from src.metrics.inception import InceptionV3
from src.models.fastgan.fastgan import weights_init
from src.modules.generative.gan.fastgan.operation import copy_G_params, load_params
from src.models.fastgan.fastgan import Generator, Discriminator
from src.transforms.diffaug import DiffAugment
import os
import torchvision.utils as vutils
import wandb
warnings.filterwarnings('ignore')


policy = 'color,translation'


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
    def __init__(self,
                 im_size: int = 256,
                 nc: int = 3,
                 ndf: int = 64,
                 ngf: int = 64,
                 nz: int = 256,
                 nlr: float = 0.0002,
                 nbeta1: float = 0.5,
                 nbeta2: float = 0.999,
                 log_images_on_epoch_n: int = 1,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # manual optimization like original

        # --- model setup ---

        self.netG = Generator(ngf=ngf, nz=nz, im_size=im_size, nc=nc)
        self.netG.apply(weights_init)

        self.netD = Discriminator(ndf=ndf, im_size=im_size, nc=nc)
        self.netD.apply(weights_init)

        # EMA parameters
        self.avg_param_G = copy_G_params(self.netG)

        # Perceptual & Inception
        self.percept = PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, in_chans=nc)

        self.register_buffer("fixed_noise", torch.FloatTensor(8, self.hparams.nz).normal_(0, 1))
        self.last_real = None
        self.last_rec = None

    def forward(self, z):
        return self.netG(z)

    def on_train_start(self):
        """Properly initialize EMA params on correct device."""
        # Deep copy all parameters to correct device
        self.avg_param_G = [
            p.data.clone().detach().to(self.device)
            for p in self.netG.parameters()
        ]

    def train_d_step(self, real_image, fake_images, label="real"):
        """Discriminator step identical to original script"""
        if label == "real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = self.netD(real_image, label, part=part)
            rand_weight = torch.rand_like(pred)
            err = (
                F.relu(rand_weight * 0.2 + 0.8 - pred).mean()
                + self.percept(rec_all, F.interpolate(real_image, rec_all.shape[2])).sum()
                + self.percept(rec_small, F.interpolate(real_image, rec_small.shape[2])).sum()
                + self.percept(
                    rec_part, F.interpolate(crop_image_by_part(real_image, part), rec_part.shape[2])
                ).sum()
            )
            return err, pred.mean(), rec_all, rec_small, rec_part
        else:
            pred = self.netD(fake_images, label)
            rand_weight = torch.rand_like(pred)
            err = F.relu(rand_weight * 0.2 + 0.8 + pred).mean()
            return err, pred.mean()

    def training_step(self, batch, batch_idx):
        real_image, label = batch
        batch_size = real_image.size(0)
        noise = torch.randn(
            batch_size, self.hparams.nz,
            device=self.device,
            dtype=torch.float32
        )

        opt_g, opt_d = self.optimizers()

        fake_images = self(noise)

        fake_images = [fi.float() for fi in fake_images]
        real_image = DiffAugment(real_image, policy=policy)
        fake_images_aug = [DiffAugment(fake, policy=policy) for fake in fake_images]

        # --------- Train D ---------
        opt_d.zero_grad(set_to_none=True)
        self.netD.zero_grad(set_to_none=True)

        err_dr, pred_real, rec_all, rec_small, rec_part = self.train_d_step(
            real_image, fake_images_aug, "real"
        )
        self.manual_backward(err_dr)

        err_df, pred_fake = self.train_d_step(
            real_image, [fi.detach() for fi in fake_images_aug], "fake"
        )
        self.manual_backward(err_df)

        opt_d.step()

        # --------- Train G ---------
        opt_g.zero_grad(set_to_none=True)
        self.netG.zero_grad(set_to_none=True)

        pred_g = self.netD(fake_images_aug, "fake")
        err_g = -pred_g.mean()

        self.manual_backward(err_g)
        opt_g.step()

        # --------- EMA update after G step ---------
        for p, avg_p in zip(self.netG.parameters(), self.avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)


               # Store some visuals
        if batch_idx == 0:
            num_images = min(8, batch_size)
            self.last_real = real_image[:num_images].detach()
            self.last_rec = (rec_all[:num_images].detach(), rec_small[:num_images].detach(),
                             rec_part[:num_images].detach())

    def on_train_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.hparams.log_images_on_epoch_n == 0:
            backup_para = copy_G_params(self.netG)
            load_params(self.netG, self.avg_param_G)
            self.netG.eval()

            sample = self(self.fixed_noise)[0].add(1).mul(0.5)
            if self.hparams.nc > 3:
                sample = sample.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            sample = sample.clamp(0, 1)

            # Create and log sample grid
            sample_grid = torchvision.utils.make_grid(sample, nrow=4)
            self.logger.experiment.log({
                "generated_samples": wandb.Image(sample_grid)
            })

            if self.last_real is not None:
                rec_all, rec_small, rec_part = self.last_rec
                rec = torch.cat(
                    [F.interpolate(self.last_real, 128), rec_all, rec_small, rec_part]
                ).add(1).mul(0.5)
                if self.hparams.nc > 3:
                    rec = rec.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                rec = rec.clamp(0, 1)

                # Create and log reconstruction grid
                rec_grid = torchvision.utils.make_grid(rec, nrow=4)
                self.logger.experiment.log({
                    "reconstructions": wandb.Image(rec_grid)
                })

            load_params(self.netG, backup_para)
            self.netG.train()

    def configure_optimizers(self):
        opt_g = optim.Adam(self.netG.parameters(),
                           lr=self.hparams.nlr,
                           betas=(self.hparams.nbeta1, self.hparams.nbeta2))
        opt_d = optim.Adam(self.netD.parameters(),
                           lr=self.hparams.nlr,
                           betas=(self.hparams.nbeta1, self.hparams.nbeta2))
        return [opt_g, opt_d]

