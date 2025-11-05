from typing import Optional
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
from src.models.fastgan.fastgan import weights_init
from src.modules.generative.gan.fastgan.operation import (
    copy_G_params,
    load_params,
)
from src.models.fastgan.fastgan import Generator, Discriminator
from src.transforms.diffaug import DiffAugment
import os
import torchvision.utils as vutils
import wandb
from torchmetrics.image import (
    SpectralAngleMapper,
    RelativeAverageSpectralError,
    StructuralSimilarityIndexMeasure,
    TotalVariation,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from scipy.io import savemat

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


class CycleFastGANModule(pl.LightningModule):
    def __init__(
        self,
        im_size: int = 256,
        nc_rgb: int = 3,
        nc_hsi: int = 64,
        ndf: int = 64,
        ngf: int = 64,
        nz: int = 256,
        nlr: float = 0.0002,
        nbeta1: float = 0.5,
        nbeta2: float = 0.999,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        log_images_on_step_n: int = 1,
        val_check_interval: int = 500,
        num_val_batches: int = 8,
        pred_output_dir: str = "generated_samples",
        pred_num_samples: int = 100,
        pred_hyperspectral: bool = True,
        pred_global_min: Optional[float | list[float]] = None,
        pred_global_max: Optional[float | list[float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # RGB → HSI Generator (G_AB) and Discriminator (D_B for HSI)
        self.G_RGB2HSI = Generator(ngf=ngf, nz=nz, im_size=im_size, nc=nc_hsi)
        self.D_HSI = Discriminator(ndf=ndf, im_size=im_size, nc=nc_hsi)

        # HSI → RGB Generator (G_BA) and Discriminator (D_A for RGB)
        self.G_HSI2RGB = Generator(ngf=ngf, nz=nz, im_size=im_size, nc=nc_rgb)
        self.D_RGB = Discriminator(ndf=ndf, im_size=im_size, nc=nc_rgb)

        # Initialize weights
        self.G_RGB2HSI.apply(weights_init)
        self.D_HSI.apply(weights_init)
        self.G_HSI2RGB.apply(weights_init)
        self.D_RGB.apply(weights_init)

        # EMA parameters for both generators
        self.avg_param_G_RGB2HSI = copy_G_params(self.G_RGB2HSI)
        self.avg_param_G_HSI2RGB = copy_G_params(self.G_HSI2RGB)

        # Perceptual loss for both domains
        self.percept_hsi = PerceptualLoss(model="net-lin", net="vgg", use_gpu=True, in_chans=nc_hsi)
        self.percept_rgb = PerceptualLoss(model="net-lin", net="vgg", use_gpu=True, in_chans=nc_rgb)

        # Fixed noise for visualization
        self.register_buffer("fixed_noise", torch.FloatTensor(8, self.hparams.nz).normal_(0, 1))

        # Validation metrics
        self.sam = SpectralAngleMapper()
        self.rase = RelativeAverageSpectralError()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.tv = TotalVariation()

        self.inception_model = InceptionV3Wrapper(normalize_input=False, in_chans=self.hparams.nc_hsi)
        self.inception_model.eval()
        self.fid = FrechetInceptionDistance(
            self.inception_model,
            input_img_size=(
                self.hparams.nc_hsi,
                self.hparams.im_size,
                self.hparams.im_size,
            ),
        )
        self.fid.eval()

        self.real_spectra = None
        self.fake_spectra = None
        self.lesion_class_name = None

    def setup(self, stage: str) -> None:
        datamodule = self.trainer.datamodule

        if not isinstance(datamodule, JointRGBHSIDataModule):
            raise ValueError("CycleFastGAN requires JointRGBHSIDataModule")

        hsi_dm = datamodule.hsi_dm

        if stage == "fit":
            train_dataset = hsi_dm.data_train
        elif stage == "validate":
            train_dataset = hsi_dm.data_val
        elif stage == "test" or stage == "predict":
            train_dataset = hsi_dm.data_test
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if isinstance(train_dataset, torch.utils.data.ConcatDataset):
            base_dataset = train_dataset.datasets[0]
        else:
            base_dataset = train_dataset

        if isinstance(base_dataset, torch.utils.data.Subset):
            base_dataset = base_dataset.dataset

        train_indices = hsi_dm.train_indices
        train_labels = np.array([base_dataset.labels[i] for i in train_indices])
        unique_labels = np.unique(train_labels)

        if len(unique_labels) == 1:
            lesion_class_int = unique_labels[0]
            inverse_labels_map = {v: k for k, v in base_dataset.labels_map.items()}
            self.lesion_class_name = inverse_labels_map[lesion_class_int]

        self.real_spectra = {"normal_skin": [], self.lesion_class_name: []} if self.lesion_class_name else None
        self.fake_spectra = {"normal_skin": [], self.lesion_class_name: []} if self.lesion_class_name else None

        if (
            hasattr(hsi_dm, "global_min")
            and hasattr(hsi_dm, "global_max")
            and (hsi_dm.global_min is not None)
            and (hsi_dm.global_max is not None)
        ):
            self.hparams.pred_global_min = hsi_dm.global_min
            self.hparams.pred_global_max = hsi_dm.global_max

    def forward_rgb2hsi(self, z):
        return self.G_RGB2HSI(z)

    def forward_hsi2rgb(self, z):
        return self.G_HSI2RGB(z)

    def on_train_start(self):
        """Initialize EMA params on correct device."""
        self.avg_param_G_RGB2HSI = [p.data.clone().detach().to(self.device) for p in self.G_RGB2HSI.parameters()]
        self.avg_param_G_HSI2RGB = [p.data.clone().detach().to(self.device) for p in self.G_HSI2RGB.parameters()]

    def train_d_step(self, real_image, fake_images, discriminator, percept, label="real"):
        """Discriminator step using FastGAN multi-scale approach"""
        if label == "real":
            part = random.randint(0, 3)
            pred, [rec_all, rec_small, rec_part] = discriminator(real_image, label, part=part)

            rand_weight = torch.rand_like(pred)
            err = (
                F.relu(rand_weight * 0.2 + 0.8 - pred).mean()
                + percept(rec_all, F.interpolate(real_image, rec_all.shape[2])).sum()
                + percept(rec_small, F.interpolate(real_image, rec_small.shape[2])).sum()
                + percept(
                    rec_part,
                    F.interpolate(crop_image_by_part(real_image, part), rec_part.shape[2]),
                ).sum()
            )
            return err, pred.mean()
        else:
            pred = discriminator(fake_images, label)
            rand_weight = torch.rand_like(pred)
            err = F.relu(rand_weight * 0.2 + 0.8 + pred).mean()
            return err, pred.mean()

    def process_batch(self, batch):
        rgb_image, rgb_mask, rgb_label = batch["rgb"]
        hsi_image, hsi_mask, hsi_label = batch["hsi"]
        return rgb_image, hsi_image

    def training_step(self, batch, batch_idx):
        real_rgb, real_hsi = self.process_batch(batch)
        batch_size = real_rgb.size(0)

        # Generate noise
        noise_rgb = torch.randn(batch_size, self.hparams.nz, device=self.device, dtype=torch.float32)
        noise_hsi = torch.randn(batch_size, self.hparams.nz, device=self.device, dtype=torch.float32)

        opt_g_rgb2hsi, opt_g_hsi2rgb, opt_d_rgb, opt_d_hsi = self.optimizers()

        # Generate fake images from noise
        fake_hsi_from_noise = self.forward_rgb2hsi(noise_rgb)
        fake_rgb_from_noise = self.forward_hsi2rgb(noise_hsi)

        # Cycle translations: RGB → HSI → RGB and HSI → RGB → HSI
        fake_hsi = self.G_RGB2HSI(noise_rgb)  # RGB→HSI
        fake_rgb = self.G_HSI2RGB(noise_hsi)  # HSI→RGB

        # For cycle consistency, we translate real images
        fake_hsi_from_rgb = self.G_RGB2HSI(noise_rgb)  # Use noise-based
        fake_rgb_from_hsi = self.G_HSI2RGB(noise_hsi)  # Use noise-based

        # Reconstruct
        rec_rgb = self.G_HSI2RGB(noise_rgb)  # HSI→RGB→HSI cycle
        rec_hsi = self.G_RGB2HSI(noise_hsi)  # RGB→HSI→RGB cycle

        # Apply DiffAugment
        real_rgb_aug = DiffAugment(real_rgb, policy=policy)
        real_hsi_aug = DiffAugment(real_hsi, policy=policy)
        fake_rgb_aug = [DiffAugment(f, policy=policy) for f in fake_rgb_from_noise]
        fake_hsi_aug = [DiffAugment(f, policy=policy) for f in fake_hsi_from_noise]

        # --------- Train D_RGB ---------
        opt_d_rgb.zero_grad(set_to_none=True)
        self.D_RGB.zero_grad(set_to_none=True)

        err_dr_rgb, pred_real_rgb = self.train_d_step(real_rgb_aug, fake_rgb_aug, self.D_RGB, self.percept_rgb, "real")
        self.manual_backward(err_dr_rgb)

        err_df_rgb, pred_fake_rgb = self.train_d_step(
            real_rgb_aug,
            [f.detach() for f in fake_rgb_aug],
            self.D_RGB,
            self.percept_rgb,
            "fake",
        )
        self.manual_backward(err_df_rgb)
        opt_d_rgb.step()

        # --------- Train D_HSI ---------
        opt_d_hsi.zero_grad(set_to_none=True)
        self.D_HSI.zero_grad(set_to_none=True)

        err_dr_hsi, pred_real_hsi = self.train_d_step(real_hsi_aug, fake_hsi_aug, self.D_HSI, self.percept_hsi, "real")
        self.manual_backward(err_dr_hsi)

        err_df_hsi, pred_fake_hsi = self.train_d_step(
            real_hsi_aug,
            [f.detach() for f in fake_hsi_aug],
            self.D_HSI,
            self.percept_hsi,
            "fake",
        )
        self.manual_backward(err_df_hsi)
        opt_d_hsi.step()

        # --------- Train G_RGB2HSI ---------
        opt_g_rgb2hsi.zero_grad(set_to_none=True)
        self.G_RGB2HSI.zero_grad(set_to_none=True)

        pred_g_hsi = self.D_HSI(fake_hsi_aug, "fake")
        err_g_adv_hsi = -pred_g_hsi.mean()

        # Cycle consistency loss (simplified for noise-based generation)
        # In true CycleGAN, we'd do: rgb → hsi → rgb and compare with original rgb
        # Here we use a simplified version with identity mapping
        identity_hsi = self.G_RGB2HSI(noise_rgb)
        err_cycle = F.l1_loss(rec_hsi[0], fake_hsi_from_noise[0])

        err_g_rgb2hsi = err_g_adv_hsi + err_cycle * self.hparams.lambda_cycle

        self.manual_backward(err_g_rgb2hsi)
        opt_g_rgb2hsi.step()

        # --------- Train G_HSI2RGB ---------
        opt_g_hsi2rgb.zero_grad(set_to_none=True)
        self.G_HSI2RGB.zero_grad(set_to_none=True)

        pred_g_rgb = self.D_RGB(fake_rgb_aug, "fake")
        err_g_adv_rgb = -pred_g_rgb.mean()

        # Cycle loss for RGB
        err_cycle_rgb = F.l1_loss(rec_rgb[0], fake_rgb_from_noise[0])

        err_g_hsi2rgb = err_g_adv_rgb + err_cycle_rgb * self.hparams.lambda_cycle

        self.manual_backward(err_g_hsi2rgb)
        opt_g_hsi2rgb.step()

        # --------- EMA update ---------
        for p, avg_p in zip(self.G_RGB2HSI.parameters(), self.avg_param_G_RGB2HSI):
            avg_p.mul_(0.999).add_(p.data, alpha=0.001)

        for p, avg_p in zip(self.G_HSI2RGB.parameters(), self.avg_param_G_HSI2RGB):
            avg_p.mul_(0.999).add_(p.data, alpha=0.001)

        # --------- Log losses ---------
        self.log("train/d_loss_rgb", err_dr_rgb + err_df_rgb, on_step=True)
        self.log("train/d_loss_hsi", err_dr_hsi + err_df_hsi, on_step=True)
        self.log("train/g_loss_rgb2hsi", err_g_rgb2hsi, on_step=True)
        self.log("train/g_loss_hsi2rgb", err_g_hsi2rgb, on_step=True)
        self.log("train/cycle_loss", err_cycle + err_cycle_rgb, on_step=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = len(batch["hsi"])

        # Backup and load EMA params
        backup_rgb2hsi = copy_G_params(self.G_RGB2HSI)
        backup_hsi2rgb = copy_G_params(self.G_HSI2RGB)

        load_params(self.G_RGB2HSI, self.avg_param_G_RGB2HSI)
        load_params(self.G_HSI2RGB, self.avg_param_G_HSI2RGB)

        self.G_RGB2HSI.eval()
        self.G_HSI2RGB.eval()

        if ((self.global_step + 1) // batch_size) % self.hparams.val_check_interval == 0:
            self._run_validation()

        if ((self.global_step + 1) // batch_size) % self.hparams.log_images_on_step_n == 0:
            # Generate HSI from noise
            sample_hsi = self.forward_rgb2hsi(self.fixed_noise)[0].add(1).mul(0.5)

            # Generate RGB from noise
            sample_rgb = self.forward_hsi2rgb(self.fixed_noise)[0].add(1).mul(0.5)

            # Convert HSI to viewable format
            if self.hparams.nc_hsi > 3:
                sample_hsi = sample_hsi.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            sample_hsi = sample_hsi.clamp(0, 1)
            sample_rgb = sample_rgb.clamp(0, 1)

            # Log both
            grid_hsi = torchvision.utils.make_grid(sample_hsi, nrow=4).detach().cpu()
            grid_rgb = torchvision.utils.make_grid(sample_rgb, nrow=4).detach().cpu()

            self.logger.experiment.log(
                {
                    "generated_hsi": wandb.Image(grid_hsi),
                    "generated_rgb": wandb.Image(grid_rgb),
                }
            )

        # Restore params
        load_params(self.G_RGB2HSI, backup_rgb2hsi)
        load_params(self.G_HSI2RGB, backup_hsi2rgb)
        self.G_RGB2HSI.train()
        self.G_HSI2RGB.train()

    def _iterate_val_loaders(self, val_loader):
        """Handle both dict of dataloaders or single dataloader cases."""
        if isinstance(val_loader, dict):
            iterators = {k: iter(v) for k, v in val_loader.items()}
            while True:
                batch_dict = {}
                exhausted = False
                for name, it in iterators.items():
                    try:
                        batch_dict[name] = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                if exhausted:
                    break
                yield batch_dict
        else:
            yield from val_loader


    def _run_validation(self):
        """Run validation metrics"""
        self.fid.reset()
        self.sam.reset()
        self.rase.reset()
        self.ssim.reset()
        self.tv.reset()

        val_loader = self.trainer.datamodule.train_dataloader()

        sam_sum = rase_sum = ssim_sum = tv_sum = 0.0
        count = 0

        with torch.no_grad():
            for i, batch in enumerate(self._iterate_val_loaders(val_loader)):
                if i >= self.hparams.num_val_batches:
                    break

                real_rgb, real_hsi = self.process_batch(batch)
                real_hsi = real_hsi.to(self.device, non_blocking=True)
                batch_size = real_hsi.size(0)

                noise = torch.randn(batch_size, self.hparams.nz, device=self.device)

                fake_hsi = self.forward_rgb2hsi(noise)[0].float()

                fake_norm = (fake_hsi + 1) / 2
                real_norm = (real_hsi + 1) / 2
                fake_norm = fake_norm.clamp(0, 1)
                real_norm = real_norm.clamp(0, 1)

                self.fid.update(fake_hsi, real=False)
                self.fid.update(real_hsi, real=True)

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

                if self.lesion_class_name is not None:
                    self.compute_spectra_statistics(real_norm_clamped, fake_norm_clamped)

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

            if self.lesion_class_name is not None and (self.real_spectra or self.fake_spectra):
                self._plot_mean_spectra()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_spectra_statistics(self, real_norm, fake_norm):
        for img, spectra_dict in [
            (real_norm, self.real_spectra),
            (fake_norm, self.fake_spectra),
        ]:
            for b in range(img.size(0)):
                image_np = img[b].cpu().numpy().transpose(1, 2, 0)
                mean_image = image_np.mean(axis=-1)

                try:
                    otsu_thresh = filters.threshold_otsu(mean_image)
                    binary_mask = mean_image < (otsu_thresh * 1)

                    if np.any(binary_mask):
                        spectrum = image_np[binary_mask].mean(axis=0)
                        spectra_dict[self.lesion_class_name].append(spectrum)

                    normal_skin_mask = ~binary_mask
                    if np.any(normal_skin_mask):
                        normal_spectrum = image_np[normal_skin_mask].mean(axis=0)
                        spectra_dict["normal_skin"].append(normal_spectrum)
                except Exception:
                    continue

    def _plot_mean_spectra(self):
        """Plot mean spectra comparing real vs synthetic data."""
        import matplotlib.pyplot as plt

        labels = ["normal_skin", self.lesion_class_name]
        real_stats = {}
        fake_stats = {}

        for label_name in labels:
            if self.real_spectra.get(label_name):
                arr = np.array(self.real_spectra[label_name])
                real_stats[label_name] = {
                    "mean": np.mean(arr, axis=0),
                    "std": np.std(arr, axis=0),
                }

            if self.fake_spectra.get(label_name):
                arr = np.array(self.fake_spectra[label_name])
                fake_stats[label_name] = {
                    "mean": np.mean(arr, axis=0),
                    "std": np.std(arr, axis=0),
                }

        ref_stats = real_stats or fake_stats
        ref_label = next((lbl for lbl in labels if ref_stats.get(lbl) is not None), None)
        if ref_label is None:
            return

        n_bands = len(ref_stats[ref_label]["mean"])
        bands = np.arange(1, n_bands + 1)

        fig, axes = plt.subplots(1, len(labels), figsize=(15, 5))
        if len(labels) == 1:
            axes = [axes]

        for ax, lbl in zip(axes, labels):
            rs = real_stats.get(lbl)
            fs = fake_stats.get(lbl)

            if rs is None and fs is None:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {lbl}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            if rs is not None:
                ax.plot(
                    bands,
                    rs["mean"],
                    linestyle="-",
                    linewidth=2.5,
                    color="C0",
                    label="Real",
                )
                ax.fill_between(
                    bands,
                    rs["mean"] - rs["std"],
                    rs["mean"] + rs["std"],
                    color="C0",
                    alpha=0.15,
                )

            if fs is not None:
                ax.plot(
                    bands,
                    fs["mean"],
                    linestyle="--",
                    linewidth=2.0,
                    color="C3",
                    label="Synthetic",
                )
                ax.fill_between(
                    bands,
                    fs["mean"] - fs["std"],
                    fs["mean"] + fs["std"],
                    color="C3",
                    alpha=0.15,
                )

            ax.set_title(f"{lbl.replace('_', ' ').title()}")
            ax.set_xlabel("Spectral Band")
            ax.set_ylabel("Reflectance")
            ax.legend()
            ax.grid(True, alpha=0.25)

        plt.suptitle("Mean Spectra Comparison: Real vs Synthetic", fontsize=14, y=1.02)
        plt.tight_layout()

        self.logger.experiment.log({"val/mean_spectra": wandb.Image(fig)})
        plt.close(fig)

    def predict_step(self, batch, batch_idx):
        """Generate and save new samples."""
        os.makedirs(self.hparams.pred_output_dir, exist_ok=True)
        self.G_RGB2HSI.eval()

        backup_para = copy_G_params(self.G_RGB2HSI)
        load_params(self.G_RGB2HSI, self.avg_param_G_RGB2HSI)

        device = self.device
        nz = self.hparams.nz

        gmin = gmax = None
        if hasattr(self.hparams, "pred_global_min"):
            gmin = getattr(self.hparams, "pred_global_min", None)
            gmax = getattr(self.hparams, "pred_global_max", None)

        with torch.no_grad():
            noise = torch.randn(self.hparams.pred_num_samples, nz, device=device)
            fake_imgs = self.forward_rgb2hsi(noise)
            fake = fake_imgs[0].float().cpu()

            for i in tqdm(range(self.hparams.pred_num_samples), desc="Generating samples"):
                fake_img = fake[i : i + 1, :, :, :].to(device)

                if self.hparams.pred_hyperspectral:
                    fake_denorm = fake_img.add(1).div(2)

                    if gmin is not None and gmax is not None:
                        gmin_arr = torch.tensor(gmin, device=device) if isinstance(gmin, list) else np.array(gmin)
                        gmax_arr = torch.tensor(gmax, device=device) if isinstance(gmax, list) else np.array(gmax)

                        if gmin_arr.size == 1:
                            fake_denorm = fake_denorm * (gmax_arr - gmin_arr) + gmin_arr
                        else:
                            gmin_t = torch.tensor(gmin_arr).view(1, -1, 1, 1)
                            gmax_t = torch.tensor(gmax_arr).view(1, -1, 1, 1)
                            fake_denorm = fake_denorm * (gmax_t - gmin_t) + gmin_t

                    fake_np = fake_denorm.cpu().squeeze().numpy()
                    fake_np = np.transpose(fake_np, (1, 2, 0))
                    mat_path = os.path.join(self.hparams.pred_output_dir, f"sample_{i:04d}.mat")
                    savemat(mat_path, {"cube": fake_np})
                else:
                    fake_rgb = (fake + 1) / 2
                    mean_band = fake_rgb.mean(dim=1, keepdim=True)
                    rgb = mean_band.repeat(1, 3, 1, 1).clamp(0, 1)
                    rgb_uint8 = (rgb * 255).byte()

                    save_path = os.path.join(self.hparams.pred_output_dir, f"sample_{i:04d}.png")
                    torchvision.utils.save_image(rgb_uint8 / 255.0, save_path)

        load_params(self.G_RGB2HSI, backup_para)
        self.G_RGB2HSI.train()

    def configure_optimizers(self):
        opt_g_rgb2hsi = optim.Adam(
            self.G_RGB2HSI.parameters(),
            lr=self.hparams.nlr,
            betas=(self.hparams.nbeta1, self.hparams.nbeta2),
        )
        opt_g_hsi2rgb = optim.Adam(
            self.G_HSI2RGB.parameters(),
            lr=self.hparams.nlr,
            betas=(self.hparams.nbeta1, self.hparams.nbeta2),
        )
        opt_d_rgb = optim.Adam(
            self.D_RGB.parameters(),
            lr=self.hparams.nlr,
            betas=(self.hparams.nbeta1, self.hparams.nbeta2),
        )
        opt_d_hsi = optim.Adam(
            self.D_HSI.parameters(),
            lr=self.hparams.nlr,
            betas=(self.hparams.nbeta1, self.hparams.nbeta2),
        )
        return [opt_g_rgb2hsi, opt_g_hsi2rgb, opt_d_rgb, opt_d_hsi]

    def on_save_checkpoint(self, checkpoint):
        checkpoint["avg_param_G_RGB2HSI"] = [p.clone().cpu() for p in self.avg_param_G_RGB2HSI]
        checkpoint["avg_param_G_HSI2RGB"] = [p.clone().cpu() for p in self.avg_param_G_HSI2RGB]

    def on_load_checkpoint(self, checkpoint):
        if "avg_param_G_RGB2HSI" in checkpoint:
            self.avg_param_G_RGB2HSI = [p.to(self.device) for p in checkpoint["avg_param_G_RGB2HSI"]]
        if "avg_param_G_HSI2RGB" in checkpoint:
            self.avg_param_G_HSI2RGB = [p.to(self.device) for p in checkpoint["avg_param_G_HSI2RGB"]]
