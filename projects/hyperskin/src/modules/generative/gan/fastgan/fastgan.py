#from typing import Any, Self
from git import Optional
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import random

import pytorch_lightning as pl

import warnings
from skimage import filters
import torchvision
from tqdm import tqdm

from src.data_modules.joint_rgb_hsi_dermoscopy import JointRGBHSIDataModule
from src.losses.lpips import PerceptualLoss
from src.metrics.inception import InceptionV3Wrapper
from src.models.fastgan.fastgan import Generator, weights_init
from src.models.timm import TIMMModel
from src.modules.generative.gan.fastgan.barlow_twins import BarlowTwinsProjector, off_diagonal
from src.modules.generative.gan.fastgan.operation import copy_G_params, load_params
from src.models.fastgan.fastgan import Discriminator
from pytorch_lightning.trainer.trainer import TrainerFn
import kornia.augmentation as K

# Import SPADE versions
from src.models.fastgan.spade_fastgan import GeneratorSPADE, DiscriminatorSPADE
from src.transforms.diffaug import DiffAugment
import os
import wandb
from PIL import Image
from torchmetrics.image import (
    SpectralAngleMapper,
    RelativeAverageSpectralError,
    StructuralSimilarityIndexMeasure,
    TotalVariation,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from scipy.io import savemat

from src.modules.generative.base_predictor import BasePredictorMixin
from src.utils.spectra_plot import MeanSpectraMetric
from src.utils.tags_and_run_name import add_tags_and_run_name_to_logger
from src.utils.utils import _iterate_val_loaders
warnings.filterwarnings("ignore")


policy = "color,translation"


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
class FastGANModule(BasePredictorMixin, pl.LightningModule):
    def __init__(
        self,
        im_size: int = 256,
        nc: int = 3,
        ndf: int = 64,
        ngf: int = 64,
        nz: int = 256,
        nlr: float = 0.0002,
        nlr_G: Optional[float] = None,
        nlr_D: Optional[float] = None,
        hinge_loss_rand_weight: bool = True,
        noise_std_start: float = 0.0,
        noise_std_end: float = 0.0,
        noise_std: Optional[float] = None,
        noise_decay_steps: int = 10000,
        use_topk: bool = False,
        topk_start: float = 0.5,
        topk_end: float = 1.0,
        topk_decay_steps: int = 50000,
        freeze_d: int = 0,
        nbeta1: float = 0.5,
        nbeta2: float = 0.999,
        log_images_on_step_n: int = 1,
        val_check_interval: int = 500,
        num_val_batches: int = 8,
        pred_output_dir: str = "generated_samples",
        pred_hyperspectral: bool = True,
        pred_global_min: Optional[float | list[float]] = None,
        pred_global_max: Optional[float | list[float]] = None,
        # SPADE configuration
        use_spade: bool = False,
        spade_conditioning: str = "rgb_mask",
        # Adversarial classifier configuration
        use_adversarial_classifier: bool = False,
        classifier_config: Optional[dict] = None,
        classifier_weights_path: Optional[str] = None,
        adversarial_loss_weight: float = 0.1,
        classifier_loss_type: str = "confidence",
        # Barlow Twins configuration
        use_barlow_twins: bool = False,
        barlow_twins_weight: float = 0.005,
        barlow_twins_lambd: float = 0.0051,
        barlow_twins_projector_dim: int = 2048,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        if self.hparams.nlr is not None and self.hparams.nlr_G is None:
            self.hparams.nlr_G = self.hparams.nlr
        if self.hparams.nlr is not None and self.hparams.nlr_D is None:
            self.hparams.nlr_D = self.hparams.nlr

        # Validate SPADE configuration
        if self.hparams.use_spade:
            assert self.hparams.spade_conditioning in ["rgb_mask", "rgb_image"], (
                f"Invalid spade_conditioning: {self.hparams.spade_conditioning}"
            )

        # --- model setup ---
        if self.hparams.use_spade:
            label_nc = 1 if self.hparams.spade_conditioning == "rgb_mask" else 3
            self.netG = GeneratorSPADE(ngf=ngf, nz=nz, im_size=im_size, nc=nc, label_nc=label_nc)
            self.netD = DiscriminatorSPADE(ndf=ndf, im_size=im_size, nc=nc, label_nc=label_nc)
        else:
            self.netG = Generator(ngf=ngf, nz=nz, im_size=im_size, nc=nc)
            self.netD = Discriminator(ndf=ndf, im_size=im_size, nc=nc)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # Optionally freeze D layers
        if self.hparams.freeze_d > 0:
            self.freeze_d(self.hparams.freeze_d)

        # EMA parameters
        self.avg_param_G = copy_G_params(self.netG)

        # Perceptual & Inception
        self.percept = PerceptualLoss(model="net-lin", net="vgg", use_gpu=True, in_chans=nc)

        self.register_buffer("fixed_noise", torch.FloatTensor(8, self.hparams.nz).normal_(0, 1))

        if self.hparams.use_spade:
            self.fixed_conditioning = None

        # --- Barlow Twins Setup ---
        if self.hparams.use_barlow_twins:
            self.barlow_projector = BarlowTwinsProjector(
                input_dim=512,
                hidden_dim=self.hparams.barlow_twins_projector_dim,
                output_dim=self.hparams.barlow_twins_projector_dim,
            )
            self.barlow_projector.apply(weights_init)
            
            # Albumentations augmentations
            self.augment = torch.nn.Sequential(
                K.RandomResizedCrop(size=(im_size, im_size), scale=(0.8, 1.0)),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomBrightness(0.4),
                K.RandomContrast(0.4),
                K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=1.0),
                K.RandomSolarize(p=0.2),
            )
        else:
            self.barlow_projector = None

        # --- Adversarial Classifier Setup ---
        if self.hparams.use_adversarial_classifier:
            assert self.hparams.classifier_config is not None, (
                "classifier_config must be provided when use_adversarial_classifier=True"
            )

            self.classifier = TIMMModel(**self.hparams.classifier_config)

            if self.hparams.classifier_weights_path is not None:
                checkpoint = torch.load(
                    self.hparams.classifier_weights_path,
                    map_location='cpu',
                    weights_only=False,
                )
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    state_dict = {
                        k.replace('net.', ''): v
                        for k, v in state_dict.items()
                    }
                else:
                    state_dict = checkpoint
                self.classifier.load_state_dict(state_dict)
                print(f"Loaded adversarial classifier weights from {self.hparams.classifier_weights_path}")

            # Freeze classifier and set to eval mode
            for param in self.classifier.parameters():
                param.requires_grad = False
            self.classifier.eval()
        else:
            self.classifier = None

        # Validation metrics
        self.sam = SpectralAngleMapper()
        self.rase = RelativeAverageSpectralError()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.tv = TotalVariation()

        self.inception_model = InceptionV3Wrapper(normalize_input=False, in_chans=self.hparams.nc)
        self.inception_model.eval()
        self.fid = FrechetInceptionDistance(
            self.inception_model,
            input_img_size=(self.hparams.nc, self.hparams.im_size, self.hparams.im_size),
        )
        self.fid.eval()

        self.spectra_metric = MeanSpectraMetric()

    def apply_augmentation(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply GPU-native augmentations (Kornia).
        images: (B, C, H, W) in [-1, 1]
        """
        # Convert to [0, 1]
        images = (images + 1) / 2
        # Apply augmentations directly on GPU
        augmented = self.augment(images)
        # Convert back to [-1, 1]
        return augmented * 2 - 1   

    def compute_barlow_twins_loss(self, features1, features2):
        """
        Compute Barlow Twins loss between two sets of features
        
        Args:
            features1: Features from first augmented view
            features2: Features from second augmented view
        """
        # Project features
        features1 = F.adaptive_avg_pool2d(features1, 1).view(features1.size(0), -1)
        features2 = F.adaptive_avg_pool2d(features2, 1).view(features2.size(0), -1)
        z1 = self.barlow_projector(features1)
        z2 = self.barlow_projector(features2)
        
        # Compute empirical cross-correlation matrix
        batch_size = z1.size(0)
        c = (z1.T @ z2) / batch_size
        
        # Loss: encourage diagonal to be 1, off-diagonal to be 0
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.hparams.barlow_twins_lambd * off_diag
        
        return loss

    def _get_tags_and_run_name(self):
        """Automatically derive tags and a run name from FastGANModule hyperparameters."""
        hparams = getattr(self, "hparams", None)
        if hparams is None:
            return

        tags = []
        if hparams.use_spade:
            tags.append("spade")
            run_name = "spade_fastgan_"
        else:
            run_name = "fastgan_"
            tags.append("fastgan")

        # Core configuration flags
        tags.append(f"imsize_{hparams.im_size}")
        tags.append(f"nz_{hparams.nz}")
        run_name += f"{hparams.im_size}px_"

        # Model architecture features
        tags.append(f"ndf_{hparams.ndf}")
        tags.append(f"ngf_{hparams.ngf}")

        # Clean trailing underscore
        run_name = run_name.rstrip("_")

        return tags, run_name

    def setup(self, stage: str) -> None:
        add_tags_and_run_name_to_logger(self)
        datamodule = self.trainer.datamodule

        # Validate datamodule structure if using SPADE
        if self.hparams.use_spade:
            if not isinstance(datamodule, JointRGBHSIDataModule):
                raise ValueError(
                    "When use_spade=True, datamodule must be JointRGBHSIDataModule to provide RGB conditioning"
                )

        if isinstance(datamodule, JointRGBHSIDataModule):
            datamodule = datamodule.hsi_dm
        # Get global min/max from datamodule
        if (
            hasattr(datamodule, "global_min")
            and hasattr(datamodule, "global_max")
            and (datamodule.global_min is not None)
            and (datamodule.global_max is not None)
        ):
            self.hparams.pred_global_min = datamodule.global_min
            self.hparams.pred_global_max = datamodule.global_max

    def forward(self, z, seg=None):
        """
        Args:
            z: noise vector
            seg: conditioning for SPADE (optional)
        """
        if self.hparams.use_spade:
            return self.netG(z, seg)
        return self.netG(z)

    def on_train_start(self):
        """Properly initialize EMA params on correct device."""
        self.avg_param_G = [p.data.clone().detach().to(self.device) for p in self.netG.parameters()]
        
        self.freeze_d(self.hparams.freeze_d)

    def freeze_d(self, n_layers: int) -> None:
        """
        Freeze the first `n_layers` hierarchical components in the Discriminator.
        Layer order is defined as:
          [down_from_big, down_4, se_2_16, down_8, se_4_32, down_16,
           down_32, se_8_64, down_64]

        Args:
            n_layers (int): Number of discriminator submodules to freeze, in order.
        """
        # Define ordered list of submodules in forward order
        layer_order = [
            ("down_from_big", self.netD.down_from_big),
            ("down_4", self.netD.down_4),
            ("se_2_16", getattr(self.netD, "se_2_16", None)),
            ("down_8", self.netD.down_8),
            ("se_4_32", getattr(self.netD, "se_4_32", None)),
            ("down_16", self.netD.down_16),
            ("down_32", self.netD.down_32),
            ("se_8_64", getattr(self.netD, "se_8_64", None)),
            ("down_64", self.netD.down_64),
        ]

        frozen_layers = []

        for i, (name, layer) in enumerate(layer_order):
            if layer is None:
                continue
            if i < n_layers:
                for p in layer.parameters():
                    p.requires_grad = False
                frozen_layers.append(name)
            else:
                # Ensure later layers are still trainable
                for p in layer.parameters():
                    p.requires_grad = True

        if len(frozen_layers) == 0:
            print("[FastGAN] No discriminator layers frozen.")
        else:
            print(f"[FastGAN] Frozen discriminator layers: {frozen_layers}")

    # Add this as a new method inside the FastGANModule class
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
        # Use global_step // 2 to get the number of training iterations
        step = float(self.global_step // 2)
        progress = min(step / decay_steps, 1.0)

        # Linear decay schedule: interpolate between start and end
        return start + (end - start) * progress

    def get_current_topk_fraction(self) -> float:
        """Compute gradually increasing Top‑k fraction for curriculum GAN training."""
        if not self.hparams.use_topk:
            return 1.0

        start = self.hparams.topk_start
        end = self.hparams.topk_end
        decay_steps = self.hparams.topk_decay_steps

        if decay_steps <= 0:
            return end

        # proportion of training progress (using global_step // 2 like noise std)
        step = float(self.global_step // 2)
        progress = min(step / decay_steps, 1.0)

        # linear interpolation
        return start + (end - start) * progress

    def extract_discriminator_features(self, images, seg=None):
        """Extract intermediate features from discriminator for Barlow Twins"""
        # We'll extract features from the discriminator forward pass
        # For this, we need to modify the discriminator call slightly
        
        # Process original size image through discriminator layers
        feat_2 = self.netD.down_from_big(images)
        feat_4 = self.netD.down_4(feat_2)
        feat_8 = self.netD.down_8(feat_4)
        feat_16 = self.netD.down_16(feat_8)
        feat_16 = self.netD.se_2_16(feat_2, feat_16)
        feat_32 = self.netD.down_32(feat_16)
        feat_32 = self.netD.se_4_32(feat_4, feat_32)
        feat_last = self.netD.down_64(feat_32)
        feat_last = self.netD.se_8_64(feat_8, feat_last)
        
        return feat_last

    def train_d_step(self, real_image, fake_images, seg, label="real"):
        """Discriminator step with optional SPADE conditioning"""
        if label == "real":
            part = random.randint(0, 3)

            if self.hparams.use_spade:
                pred, [rec_all, rec_small, rec_part] = self.netD(real_image, label, seg=seg, part=part)
            else:
                pred, [rec_all, rec_small, rec_part] = self.netD(real_image, label, part=part)
                
            if self.hparams.hinge_loss_rand_weight:
                rand_weight = torch.rand_like(pred) * 0.2 + 0.8 
            else:
                rand_weight = torch.ones_like(pred)

            err = (
                F.relu(rand_weight - pred).mean()
                + self.percept(rec_all, F.interpolate(real_image, rec_all.shape[2])).sum()
                + self.percept(rec_small, F.interpolate(real_image, rec_small.shape[2])).sum()
                + self.percept(rec_part, F.interpolate(crop_image_by_part(real_image, part), rec_part.shape[2])).sum()
            )
            return err, pred.mean(), rec_all, rec_small, rec_part
        else:
            if self.hparams.use_spade:
                pred = self.netD(fake_images, label, seg=seg)
            else:
                pred = self.netD(fake_images, label)

            if self.hparams.hinge_loss_rand_weight:
                rand_weight = torch.rand_like(pred) * 0.2 + 0.8 
            else:
                rand_weight = torch.ones_like(pred)

            err = F.relu(rand_weight + pred).mean()
            return err, pred.mean()

    def get_real_image(self, batch):
        if "hsi" in batch:
            real_image = batch["hsi"]["image"]
        else:
            real_image = batch["image"]
        return real_image

    def get_conditioning(self, batch):
        seg = None
        if self.hparams.use_spade:
            if self.hparams.spade_conditioning == "rgb_mask":
                if "rgb" in batch:
                    seg = batch["rgb"]["mask"]
                else:
                    seg = batch["mask"]

                if seg.ndim == 3:  # (B, H, W)
                    seg = seg.unsqueeze(1)  # (B, 1, H, W)
            elif self.hparams.spade_conditioning == "rgb_image":
                if "rgb" in batch:
                    seg = batch["rgb"]["image"]
                else:
                    seg = batch["image"]
            else:
                raise ValueError(f"Invalid spade_conditioning: {self.hparams.spade_conditioning}")
        return seg

    # Update the training_step method to include adversarial loss:
    def training_step(self, batch, batch_idx):
        real_image = self.get_real_image(batch)
        seg = self.get_conditioning(batch)

        batch_size = real_image.size(0)
        noise = torch.randn(batch_size, self.hparams.nz, device=self.device, dtype=torch.float32)

        opt_g, opt_d = self.optimizers()

        fake_images = self(noise, seg)
        fake_images = [fi.float() for fi in fake_images]
        real_image = DiffAugment(real_image, policy=policy)
        fake_images_aug = [DiffAugment(fake, policy=policy) for fake in fake_images]

        seg_aug = DiffAugment(seg, policy=policy) if seg is not None else None

        # Add noise scheduler
        noise_std = self.get_current_noise_std()
        noisy_real_image = real_image
        noisy_fake_images_aug = fake_images_aug

        if noise_std > 0:
            noisy_real_image = real_image + torch.randn_like(real_image) * noise_std
            noisy_fake_images_aug = [
                f + torch.randn_like(f) * noise_std for f in fake_images_aug
            ]

        # --------- Train D ---------
        opt_d.zero_grad(set_to_none=True)
        self.netD.zero_grad(set_to_none=True)
        if self.hparams.use_barlow_twins:
            self.barlow_projector.zero_grad(set_to_none=True)

        err_dr, pred_real, rec_all, rec_small, rec_part = self.train_d_step(
            noisy_real_image, noisy_fake_images_aug, seg_aug, "real"
        )
        
        # Barlow Twins loss on real data
        barlow_loss = torch.tensor(0.0, device=self.device)
        if self.hparams.use_barlow_twins:
            # Create two augmented views of the real image
            real_view1 = self.apply_augmentation(real_image)
            real_view2 = self.apply_augmentation(real_image)
            
            # Extract features from both views
            features1 = self.extract_discriminator_features(real_view1, seg_aug)
            features2 = self.extract_discriminator_features(real_view2, seg_aug)
            
            # Compute Barlow Twins loss
            barlow_loss = self.compute_barlow_twins_loss(features1, features2)
            barlow_loss = barlow_loss * self.hparams.barlow_twins_weight
        
        # Total discriminator loss
        d_loss = err_dr + barlow_loss
        self.manual_backward(d_loss)

        err_df, pred_fake = self.train_d_step(
            noisy_real_image, [fi.detach() for fi in noisy_fake_images_aug], seg_aug, "fake"
        )
        self.manual_backward(err_df)

        opt_d.step()

        # --------- Train G ---------
        opt_g.zero_grad(set_to_none=True)
        self.netG.zero_grad(set_to_none=True)

        if self.hparams.use_spade:
            pred_g = self.netD(fake_images_aug, "fake", seg=seg_aug)
        else:
            pred_g = self.netD(fake_images_aug, "fake")

        if self.hparams.use_topk:
            topk_frac = self.get_current_topk_fraction()
            k = max(1, int(pred_g.numel() * topk_frac))
            topk_vals, _ = torch.topk(pred_g.view(-1), k, largest=True)
            err_g = -topk_vals.mean()

            self.log("train/topk_fraction", topk_frac, on_step=True, on_epoch=False)
            self.log("train/topk_k", float(k), on_step=True, on_epoch=False)
        else:
            err_g = -pred_g.mean()

        # Adversarial Classifier Loss
        if self.classifier is not None:
            fake_for_classifier = (fake_images[0] + 1) / 2
            fake_for_classifier = fake_for_classifier.clamp(0, 1)
            self.classifier.eval()

            logits = self.classifier(fake_for_classifier)
            probs = torch.softmax(logits, dim=-1)
            melanoma_prob = probs[:, 0]

            boundary_loss = torch.mean(torch.abs(melanoma_prob - 0.5))
            adversarial_loss = boundary_loss * self.hparams.adversarial_loss_weight

            err_g = err_g + adversarial_loss
            self.log("train/adversarial_loss", adversarial_loss, on_step=True, on_epoch=False)

        self.manual_backward(err_g)
        opt_g.step()

        # EMA update after G step
        for p, avg_p in zip(self.netG.parameters(), self.avg_param_G):
            avg_p.mul_(0.999).add_(p.data, alpha=0.001)

        # Log losses
        d_loss_total = err_dr + err_df
        self.log("train/d_loss_real", err_dr, on_step=True, on_epoch=False)
        self.log("train/d_loss_fake", err_df, on_step=True, on_epoch=False)
        self.log("train/d_loss_total", d_loss_total, on_step=True, on_epoch=False)
        self.log("train/g_loss", err_g, on_step=True, on_epoch=False)
        self.log("train/noise_std", noise_std, on_step=True, on_epoch=False)
        
        if self.hparams.use_barlow_twins:
            self.log("train/barlow_twins_loss", barlow_loss, on_step=True, on_epoch=False)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        current_step = self.global_step // 2

        backup_para = copy_G_params(self.netG)
        load_params(self.netG, self.avg_param_G)
        self.netG.eval()

        if current_step % self.hparams.val_check_interval == 0:
            self._run_validation()

        if current_step % self.hparams.log_images_on_step_n == 0:
            # Store fixed conditioning on first log
            if self.hparams.use_spade and self.fixed_conditioning is None:
                seg = self.get_conditioning(batch)
                self.fixed_conditioning = seg[:8].clone().detach()

            # Generate with fixed conditioning
            fixed_seg = self.fixed_conditioning if self.hparams.use_spade else None
            # limit self.fixed_noise to match fixed_seg size
            fixed_noise = self.fixed_noise[: min(8, fixed_seg.size(0))] if self.hparams.use_spade else self.fixed_noise
            sample = self(fixed_noise, fixed_seg)[0].add(1).mul(0.5)

            # Convert to 3 channel view if HSI > 3 channels
            if self.hparams.nc > 3:
                sample = sample.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            sample = sample.clamp(0, 1)

            # --- NEW: also visualize the conditioning input ---
            if fixed_seg is not None:
                # Denormalize SPADE conditioning (assumed same [-1,1] normalization)
                cond_vis = fixed_seg.add(1).mul(0.5).clamp(0, 1)

                # If conditioning is grayscale (e.g., mask), expand to RGB for better viewing
                if cond_vis.size(1) == 1:
                    cond_vis = cond_vis.repeat(1, 3, 1, 1)
                cond_vis = cond_vis.clamp(0, 1)

                # Concatenate conditioning + generated horizontally
                vis_combined = torch.cat([cond_vis, sample], dim=0)
            else:
                vis_combined = sample

            sample_grid = torchvision.utils.make_grid(vis_combined, nrow=4
                                                      ).detach().cpu()

            if hasattr(self.logger, "experiment") and self.logger.experiment is not None:
                self.logger.experiment.log({"generated_samples": wandb.Image(sample_grid)})

        load_params(self.netG, backup_para)
        self.netG.train()

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

                real_image = self.get_real_image(batch)
                seg = self.get_conditioning(batch)

                if self.hparams.use_spade and "rgb" in batch:
                    label = batch["rgb"].get("label", None)
                else:
                    label = batch.get("label", None)
                
                if label is not None:
                    label = label.to(self.device, non_blocking=True)
                    
                seg = seg.to(self.device, non_blocking=True) if seg is not None else None

                real_image = real_image.to(self.device, non_blocking=True)
                batch_size = real_image.size(0)
                noise = torch.randn(batch_size, self.hparams.nz, device=self.device)

                fake_images = self(noise, seg)
                fake = fake_images[0].float()

                fake_norm = (fake + 1) / 2
                real_norm = (real_image + 1) / 2
                fake_norm = fake_norm.clamp(0, 1)
                real_norm = real_norm.clamp(0, 1)

                self.fid.update(fake, real=False)
                self.fid.update(real_image, real=True)

                eps = 1e-8
                fake_norm_clamped = fake_norm.clamp(eps, 1.0)
                real_norm_clamped = real_norm.clamp(eps, 1.0)

                try:
                    sam_val = self.sam(fake_norm_clamped, real_norm_clamped)
                    if torch.isnan(sam_val):
                        sam_val = torch.tensor(0.0, device=self.device)
                except Exception:
                    sam_val = torch.tensor(0.0, device=self.device)

                rase_val = torch.nan_to_num(self.rase(fake_norm_clamped, real_norm_clamped), nan=0.0)
                ssim_val = torch.nan_to_num(self.ssim(fake_norm_clamped, real_norm_clamped), nan=0.0)
                tv_val = torch.nan_to_num(self.tv(fake_norm_clamped), nan=0.0)

                sam_sum += sam_val.item()
                rase_sum += rase_val.item()
                ssim_sum += ssim_val.item()
                tv_sum += tv_val.item()
                count += 1

                self.spectra_metric.update(real_norm, is_fake=False, labels=label)
                self.spectra_metric.update(fake_norm, is_fake=True, labels=label)

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
        if fig is not None and hasattr(self.logger, "experiment") and self.logger.experiment is not None:
            self.logger.experiment.log({"Mean Spectra": wandb.Image(fig)})
        self.spectra_metric.reset()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_predict_start(self) -> None:
        datamodule = self.trainer.datamodule
        predict_dataloader = datamodule.predict_dataloader()

        if len(predict_dataloader) > 1 and self.hparams.use_spade:
            print(
                "WARNING: More than one predict dataloader detected. "
                "Probably using JointRGBHSIDataModule, but SPADE FastGAN predict only uses the rgb dataloader."
                "All images with channels different than rgb_channels will be ignored."
                "Use --data.init_args.rgb_only=True to avoid this warning."
            )


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch_size = batch["image"].size(0)

        if batch["image"].size(1) != 3 and self.hparams.use_spade:
            return  # Skip non-RGB images

        backup_para = copy_G_params(self.netG)
        load_params(self.netG, self.avg_param_G)
        self.netG.eval()

        seg = None
        if self.hparams.use_spade:
            masks = batch.get("mask", None)
            seg = batch.get("image", None) if self.hparams.spade_conditioning == "rgb_image" else masks

        with torch.no_grad():
            noise = torch.randn(batch_size, self.hparams.nz, device=self.device)
            fake_imgs = self(noise, seg)
            fake = fake_imgs[0]

        self._save_generated_batch(
            fake_batch=fake,
            batch_idx=batch_idx,
            pred_hyperspectral=self.hparams.pred_hyperspectral,
        )

        load_params(self.netG, backup_para)
        self.netG.train()

    # def configure_optimizers(self):
    #     # Generator parameters
    #     opt_g = optim.Adam(
    #         self.netG.parameters(), 
    #         lr=self.hparams.nlr_G, 
    #         betas=(self.hparams.nbeta1, self.hparams.nbeta2)
    #     )
        
    #     # Discriminator parameters (include Barlow projector if enabled)
    #     d_params = list(self.netD.parameters())
    #     if self.hparams.use_barlow_twins:
    #         d_params += list(self.barlow_projector.parameters())
        
    #     opt_d = optim.Adam(
    #         d_params,
    #         lr=self.hparams.nlr_D, 
    #         betas=(self.hparams.nbeta1, self.hparams.nbeta2)
    #     )
        
    #     return [opt_g, opt_d]
    
    def configure_optimizers(self):
        # Generator parameters
        opt_g = optim.Adam(
            self.netG.parameters(), 
            lr=self.hparams.nlr_G, 
            betas=(self.hparams.nbeta1, self.hparams.nbeta2)
        )
        
        # Discriminator parameters (include Barlow projector if enabled)
        d_params = list(self.netD.parameters())
        if self.hparams.use_barlow_twins:
            d_params += list(self.barlow_projector.parameters())
        
        opt_d = optim.Adam(
            d_params,
            lr=self.hparams.nlr_D, 
            betas=(self.hparams.nbeta1, self.hparams.nbeta2)
        )
        
        return [opt_g, opt_d]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["avg_param_G"] = [p.clone().cpu() for p in self.avg_param_G]
        if self.hparams.use_barlow_twins:
            checkpoint["barlow_projector"] = self.barlow_projector.state_dict()
        
    def on_load_checkpoint(self, checkpoint):
        # load barlow projector state dict if applicable
        if self.hparams.use_barlow_twins and "barlow_projector" in checkpoint:
            self.barlow_projector.load_state_dict(checkpoint["barlow_projector"])
            
        if "avg_param_G" in checkpoint:
            self.avg_param_G = [p.clone().to(self.device) for p in checkpoint["avg_param_G"]]

if __name__ == "__main__":
    # Simple test to instantiate the module
    model = FastGANModule(
        im_size=64,
        nc=10,
        ndf=32,
        ngf=32,
        nz=128,
        use_spade=True,
        spade_conditioning="rgb_image",
    )
    model.load_from_checkpoint
