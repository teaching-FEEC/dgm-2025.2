import os
from pytorch_lightning import Callback, LightningModule, Trainer
from src.modules.generative.gan.fastgan.operation import copy_G_params, load_params
import torchvision.utils as vutils
import torch

class ImageSavingCallback(Callback):
    """Callback for saving generated images during training"""

    def __init__(self, fixed_noise_shape: tuple, saved_image_folder, nc=3):
        super().__init__()
        self.fixed_noise = torch.randn(fixed_noise_shape)
        self.saved_image_folder = saved_image_folder
        self.nc = nc

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.fixed_noise = self.fixed_noise.to(pl_module.device)
        os.makedirs(self.saved_image_folder, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % pl_module.hparams.save_interval == 0:
            self._save_images(trainer, pl_module)

    def _save_images(self, trainer, pl_module):
        pl_module.eval()
        backup_para = copy_G_params(pl_module.netG)
        load_params(pl_module.netG, pl_module.avg_param_G)

        with torch.no_grad():
            sample = pl_module.netG(self.fixed_noise)[0].add(1).mul(0.5)

            if self.nc > 3:
                sample = sample.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

            vutils.save_image(sample, f"{self.saved_image_folder}/{trainer.global_step}.jpg", nrow=10)

            # Save reconstruction images
            if hasattr(pl_module, "last_reconstructions"):
                rec = pl_module.last_reconstructions
                if self.nc > 3:
                    rec = rec.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

                vutils.save_image(rec, f"{self.saved_image_folder}/rec_{trainer.global_step}.jpg")

        load_params(pl_module.netG, backup_para)
        pl_module.train()
