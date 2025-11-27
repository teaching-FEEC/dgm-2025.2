import torch
from torch import Tensor
from torch.optim import Adam, RMSprop

from src.modules.generative.gan.dcgan import DCGAN
from src.models.shs_gan.shs_generator import Generator
from src.models.shs_gan.shs_discriminator import Critic3D
import pytorch_lightning as pl

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchinfo import summary
from torchvision.utils import make_grid
import wandb
import warnings

def extract_x(batch):
    """Extracts the image tensor from any batch structure."""
    if isinstance(batch, dict):
        for key in ["image", "hsi", "img", "data"]:
            if key in batch:
                return batch[key]
        raise ValueError(f"Cannot find image key in batch dict: {batch.keys()}")

    if isinstance(batch, (tuple, list)):
        return batch[0]

    if isinstance(batch, torch.Tensor):
        return batch

    raise TypeError(f"Unsupported batch type: {type(batch)}")


class SHSWGANModule(pl.LightningModule):
    """
    Wasserstein Generative Adversarial Network (WGAN).

    WGAN is an alternative to traditional GAN training. It provides more stable learning,
    avoids mode collapse, and offers meaningful learning curves useful for debugging and hyperparameter tuning.
    This implementation allows for two methods to enforce the 1-Lipschitz constraint:
    gradient penalty (gp) or weight clipping (clip).
    """

    def __init__(
        self,
        img_channels: int = 16,
        input_channels: int = 1,
        img_size: int = 64,
        d_lr: float = 5e-5,
        g_lr: float = 2e-4,
        weight_decay: float = 0,
        b1: float = 0.5,
        b2: float = 0.9,
        n_critic: int = 5,
        clip_value: float = 0.01,
        grad_penalty: float = 10,
        constraint_method: str = "gp",
        calculate_metrics: bool = False,
        metrics: list[str] = [],
        summary: bool = True,
        log_images_after_n_epochs: int = 1,
        log_metrics_after_n_epochs: int = 1,
    ) :
        super().__init__() 

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.calculate_metrics = calculate_metrics
        self.metrics = metrics

        assert constraint_method in [
            "gp",
            "clip",
            "sn",
        ], "Either gradient penalty (gp) or weight clipping (clip) to enforce 1-Lipschitz constraint."
        self.clip_value = clip_value
        self.grad_penalty = grad_penalty
        self.constraint_method = constraint_method


        self.generator = Generator(
            in_channels= input_channels,
            out_channels=img_channels,
            base_filters=64
        )

        self.critic = Critic3D(in_channels=img_channels, fft_arm=False)

        if self.metrics:
            self.fid = FrechetInceptionDistance() if "fid" in self.metrics else None
            self.kid = KernelInceptionDistance(subset_size=100) if "kid" in self.metrics else None
            self.inception_score = InceptionScore() if "is" in self.metrics else None

        self.z = torch.randn([16, self.hparams.input_channels, 
                                   self.hparams.img_size, 
                                   self.hparams.img_size])
        if summary:
            self.summary()


    def training_step(self, batch, batch_idx):
        x = extract_x(batch)

        x_min = float(x.min().cpu())
        x_max = float(x.max().cpu())

        B, _, H, W = x.shape
        z = torch.randn(B, self.hparams.input_channels, H, W, device=self.device)

        d_optim, g_optim = self.optimizers()
        x_hat = self.generator(z)

        # --- Train Discriminator ---
        if (self.global_step + 1) % (self.hparams.n_critic + 1) != 0:
            loss_dict = self._calculate_d_loss(x, x_hat)
            d_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["d_loss"])
            loss_dict["d_grad_norm"] = self._compute_grad_norm(self.critic)
            d_optim.step()

        # --- Train Generator ---
        else:
            loss_dict = self._calculate_g_loss(x_hat)
            g_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["g_loss"])
            loss_dict["g_grad_norm"] = self._compute_grad_norm(self.generator)
            g_optim.step()

        log_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
        self.log_dict(
            log_loss_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )


    def _calculate_d_loss(self, x: Tensor, x_hat: Tensor) -> Tensor:
        d_loss_real = self.critic(x).mean()
        d_loss_fake = self.critic(x_hat.detach()).mean()
        d_loss = d_loss_fake - d_loss_real

        loss_dict = {
            "d_loss": d_loss,
            "d_loss_real": d_loss_real,
            "d_loss_fake": d_loss_fake,
        }

        if self.training:
            if self.hparams.constraint_method == "gp":
                gradient_penalty = self._calculate_gradient_penalty(x, x_hat)
                d_loss += gradient_penalty
                loss_dict["gradient_penalty"] = gradient_penalty

            elif self.hparams.constraint_method == "clip":
                self._weight_clipping()

            elif self.hparams.constraint_method == "sn":
                pass

            else:
                raise ValueError(
                    f"{self.hparams.constraint_method} not expected, "
                    "constraint method must be either 'gp' for gradient penalty "
                    "or 'clip' for weight clipping."
                )
        return loss_dict

    def _calculate_g_loss(self, x_hat: Tensor) -> Tensor:
        g_loss = -self.critic(x_hat).mean()
        loss_dict = {"g_loss": g_loss}
        return loss_dict

    def _calculate_gradient_penalty(self, x: Tensor, x_hat: Tensor) -> Tensor:
        """
        Calculates the gradient penalty for WGAN-GP.

        The gradient penalty ensures the discriminator's gradients have a norm close to 1,
        enforcing the Lipschitz constraint. This results in a more stable training process and
        mitigates the issue of mode collapse.

        Args:
            x (Tensor): A batch of real images from the dataset.
            x_hat (Tensor): A batch of images produced by the generator.

        Returns:
            Tensor: The computed gradient penalty.
        """

        # Generate random tensor for interpolation
        alpha = torch.rand(x.size(0), 1, 1, 1, device=self.device)

        # Create interpolated samples by blending real and generated images
        interpolated_images = alpha * x + (1 - alpha) * x_hat
        interpolated_images.requires_grad_(True)

        # Compute the critic's scores for interpolated samples
        scores_on_interpolated = self.critic(interpolated_images)

        # Calculate gradients of the scores with respect to the interpolated images
        gradients = torch.autograd.grad(
            outputs=scores_on_interpolated,
            inputs=interpolated_images,
            grad_outputs=torch.ones_like(scores_on_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute the gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        grad_penalty = ((gradient_norm - 1) ** 2).mean()

        return grad_penalty * self.hparams.grad_penalty

    def _weight_clipping(self):
        """
        Clip the discriminator's weights for stable training,
        which ensures the discriminator's gradients are bounded.
        A crude way to enforce the 1-Lipschitz constraint on the critic.
        """
        for param in self.critic.parameters():
            param.data.clamp_(
                -self.hparams.clip_value,
                self.hparams.clip_value,
            )

    @staticmethod
    def _compute_grad_norm(model: torch.nn.Module) -> torch.Tensor:
        """
        Computes the total (global) L2 norm of gradients across all parameters in a model.

        Args:
            model (torch.nn.Module): The model for which to compute gradient norms.

        Returns:
            torch.Tensor: Scalar tensor indicating the global gradient norm.
        """
        total_norm = 0.0
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return torch.tensor(0.0)

        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return torch.tensor(total_norm, device=parameters[0].device)

    def configure_optimizers(self):
        if self.hparams.constraint_method == "clip":
            # Empirically the authors recommended RMSProp optimizer on the critic,
            # rather than a momentum based optimizer such as Adam which could cause instability in the model training.
            d_optim = RMSprop(
                self.critic.parameters(),
                lr=self.hparams.d_lr,
            )
            g_optim = RMSprop(
                self.generator.parameters(),
                lr=self.hparams.g_lr,
            )

        elif self.hparams.constraint_method == "gp":
            d_optim = Adam(
                self.critic.parameters(),
                lr=self.hparams.d_lr,
                betas=(self.hparams.b1, self.hparams.b2),
                weight_decay=self.hparams.weight_decay,
            )
            g_optim = Adam(
                self.generator.parameters(),
                lr=self.hparams.g_lr,
                betas=(self.hparams.b1, self.hparams.b2),
                weight_decay=self.hparams.weight_decay,
            )

        elif self.hparams.constraint_method == "sn":
            d_optim = Adam(
                self.critic.parameters(),
                lr=self.hparams.d_lr,
                betas=(self.hparams.b1, self.hparams.b2),
                weight_decay=self.hparams.weight_decay,
            )
            g_optim = Adam(
                self.generator.parameters(),
                lr=self.hparams.g_lr,
                betas=(self.hparams.b1, self.hparams.b2),
                weight_decay=self.hparams.weight_decay,
            )
        return [d_optim, g_optim], []


    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)

    def _common_step(self, batch, mode: str):
        x = extract_x(batch)

        z = torch.randn(
            x.size(0),
            self.hparams.input_channels,
            self.hparams.img_size,
            self.hparams.img_size,
            device=self.device,
        )

        x_hat = self.generator(z)
        d_optim, g_optim = self.optimizers()

        # --- Discriminator ---
        loss_dict = self._calculate_d_loss(x, x_hat)
        if self.training:
            d_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["d_loss"])
            d_optim.step()

        # --- Generator ---
        loss_dict.update(self._calculate_g_loss(x_hat))
        if self.training:
            g_optim.zero_grad(set_to_none=True)
            self.manual_backward(loss_dict["g_loss"])
            g_optim.step()

        loss_dict = {f"{mode}/{k}": v for k, v in loss_dict.items()}
        self.log_dict(
            loss_dict,
            prog_bar=True,
            logger=True,
            sync_dist=torch.cuda.device_count() > 1,
        )

        return x, x_hat, loss_dict

    def validation_step(self, batch, batch_idx):
        x = extract_x(batch)
        x, x_hat, loss_dict = self._common_step(batch, "val")

        if self.calculate_metrics and (self.current_epoch + 1) % self.hparams.log_metrics_after_n_epochs == 0:
            self.update_metrics(x, x_hat)

        return loss_dict


    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.log_images_after_n_epochs == 0:
            self._log_images(fig_name="Random Generation", batch_size=16)

        if self.calculate_metrics and (self.current_epoch + 1) % self.hparams.log_metrics_after_n_epochs == 0:
            metrics = self.compute_metrics()

            metrics = {f"val/{k}": v for k, v in metrics.items()}
            self.log_dict(
                metrics,
                prog_bar=True,
                logger=True,
                sync_dist=torch.cuda.device_count() > 1,
            )

            self.fid.reset() if "fid" in self.metrics else None
            self.kid.reset() if "kid" in self.metrics else None
            self.inception_score = InceptionScore().to(self.device) if "is" in self.metrics else None

    def update_metrics(self, x, x_hat):
        # if x_hat has 1 channel, repeat to make it 3 channels
        if x_hat.size(1) == 1:
            x_hat = x_hat.repeat(1, 3, 1, 1)
        elif x_hat.size(1) > 3:
            x_hat = x_hat.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        # Update metrics with real and generated images
        x = (
            x
            .add_(1.0)
            .mul_(127.5)
            .byte()
        )
        x_hat = (
            x_hat
            .add_(1.0)
            .mul_(127.5)
            .byte()
        )


        if "fid" in self.metrics:
            self.fid.update(x, real=True)
            self.fid.update(x_hat, real=False)

        if "kid" in self.metrics:
            self.kid.update(x, real=True)
            self.kid.update(x_hat, real=False)

        if "is" in self.metrics:
            self.inception_score.update(x_hat)

    def compute_metrics(self) -> dict[str, Tensor]:
        fid_score = self.fid.compute() if "fid" in self.metrics else None
        kid_mean, kid_std = self.kid.compute() if "kid" in self.metrics else None, None
        is_mean, is_std = self.inception_score.compute() if "is" in self.metrics else None, None

        metrics = {}
        if fid_score is not None:
            metrics["fid_score"] = fid_score
        if kid_mean is not None:
            metrics["mean_kid_score"] = kid_mean
        if kid_std is not None:
            metrics["std_kid_score"] = kid_std
        if is_mean is not None:
            metrics["mean_inception_score"] = is_mean
        if is_std is not None:
            metrics["std_inception_score"] = is_std
        return metrics

    @torch.no_grad()
    def _log_images(self, fig_name: str, batch_size: int):
        sample_images = self.generator(self.z.to(self.device))

        if sample_images.size(1) == 1:
            sample_images = sample_images.repeat(1, 3, 1, 1)
        elif sample_images.size(1) > 3:
            sample_images = sample_images.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        fig = make_grid(
            tensor=sample_images,
            value_range=(-1, 1),
            normalize=True,
        )
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log({fig_name: [wandb.Image(fig)]})

    def summary(
        self,
        col_names: list[str] = [
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
                self.hparams.input_channels,
                self.hparams.img_size,
                self.hparams.img_size,
            ]
        )

        summary(
            self.generator,
            input_data=torch.randn(1, self.hparams.input_channels, 
                                   self.hparams.img_size, 
                                   self.hparams.img_size, 
                                   device=self.device),
            col_names=col_names,
        )

        summary(
            self.critic,
            input_data=x,
            col_names=col_names,
        )
