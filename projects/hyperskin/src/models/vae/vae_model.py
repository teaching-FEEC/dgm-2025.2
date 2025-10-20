
from typing import List, Literal, Optional, Tuple, Dict
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


class Encoder(nn.Module):
    """
    Encoder defines the approximate posterior distribution q(z|x),
    which takes an input image as observation and outputs a latent representation defined by
    mean (mu) and log variance (log_var). These parameters parameterize the Gaussian
    distribution from which we can sample latent variables.
    """
    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int,
    ):
        super(Encoder, self).__init__()
        self.img_shape: List[int] = [img_channels, img_size, img_size]
        self.latent_dim: int = latent_dim

        # Neural network layers to process the input image
        self.layers = nn.Sequential(
            nn.Linear(np.prod(self.img_shape), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )

        # Layers to produce mu and log_var
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_flatten = x.view(x.size(0), -1)
        out = self.layers(x_flatten)
        return self.mu(out), self.log_var(out)

class Decoder(nn.Module):
    """
    Decoder defines the conditional distribution of the observation p(x|z),
    which takes a latent sample as input and outputs the parameters for a conditional distribution of the observation.
    """

    def __init__(
        self,
        img_channels: int,
        img_size: int,
        latent_dim: int,
    ):
        super(Decoder, self).__init__()
        self.img_shape: List[int] = [img_channels, img_size, img_size]
        self.latent_dim: int = latent_dim

        # Neural network layers to process the latent variable and produce the reconstructed image
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, np.prod(self.img_shape)),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        out = self.layers(z)
        return out.view(out.size(0), *self.img_shape)

    def random_sample(self, batch_size: int) -> Tensor:
        z = torch.randn([batch_size, self.latent_dim], device=self.device)
        return self(z)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    

# ======================================================================
# Patch: Make GenericEncoder / GenericDecoder VAE-friendly out of the box
# ----------------------------------------------------------------------
# Why this exists:
#   Your VAE module expects:
#     mu, log_var = encoder(x)         # i.e., a 2-tuple of tensors
#     x_hat        = decoder(z)        # i.e., a single tensor
#
#   But the earlier "generic" versions returned dicts, so Python tried to
#   unpack the dict *keys* ("embedding", "log_covariance") into
#   (mu, log_var) and later did: 0.5 * "log_covariance"  -> TypeError.
#
# What this patch changes:
#   - When model_type == "vae" and you DON'T ask for intermediates
#     (i.e., output_layer_levels is None), the encoder returns (mu, log_var).
#   - When model_type == "vae" and no intermediates are requested,
#     the decoder returns just the reconstruction tensor.
#
#   - If you DO request intermediates (output_layer_levels not None),
#     both modules fall back to the rich dict output so you can inspect layers.
#
# This keeps your current VAE module working while preserving advanced usage.
# ======================================================================


# ----------------------------- Helpers --------------------------------

def _auto_base_channels(input_dim: Tuple[int, int, int]) -> int:
    """
    Choose a reasonable starting number of channels as a function of resolution
    and input channel count. This keeps compute stable across datasets.
    """
    # Unpack input shape (C, H, W)
    c, h, w = input_dim
    # Drive capacity by the smallest spatial side
    min_dim = min(h, w)

    # Resolution buckets → base channels
    if min_dim <= 32:
        bc = 128
    elif min_dim <= 64:
        bc = 64
    elif min_dim <= 128:
        bc = 48
    elif min_dim <= 256:
        bc = 32
    else:
        bc = 24

    # Adjust for non-RGB channel counts; never go below 16
    bc = max(16, int(bc * (3 / max(1, c))))
    return bc


def _auto_min_spatial_size(input_dim: Tuple[int, int, int]) -> int:
    """
    Target spatial size after encoder downsampling / decoder starting size.
    """
    # Unpack input shape
    _, h, w = input_dim
    # Base on the smaller spatial side
    min_dim = min(h, w)

    # Reasonable targets per resolution regime
    if min_dim <= 64:
        return 4
    elif min_dim <= 128:
        return 8
    elif min_dim <= 256:
        return 8
    elif min_dim <= 512:
        return 16
    else:
        return 32


class ResBlock(nn.Module):
    """
    Minimal residual bottleneck used when block_type='resnet'.

    Layout:
      x → 1x1 (reduce) → ReLU → 3x3 → ReLU → 1x1 (expand) → +x
    """
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, in_channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection
        return x + self.net(x)


# ============================== ENCODER ================================

class GenericEncoder(nn.Module):
    """
    A compact, auto-scaling encoder with VAE-friendly return types.

    --- IMPORTANT RETURN BEHAVIOR ---
    • If model_type == "vae" and output_layer_levels is None:
        returns (mu, log_var)  # a 2-tuple of tensors  ✅ VAE-friendly
    • Otherwise (different model_type or you request intermediates):
        returns a dict with keys like "embedding", "log_covariance",
        and optionally "embedding_layer_{k}" for intermediates.

    This preserves advanced introspection without breaking your VAE.
    """

    def __init__(
        self,
        input_dim: Tuple[int, int, int],                          # (C, H, W)
        latent_dim: int,                                          # size of z
        block_type: Literal["conv", "resnet"] = "conv",           # trunk family
        model_type: Literal["ae", "vae", "svae", "vqvae", "hrqvae"] = "ae",
        base_channels: Optional[int] = None,                      # first stage width
        channel_multipliers: Optional[List[float]] = None,        # per-stage widen
        num_res_blocks: int = 2,                                  # resnet head depth
        use_batch_norm: bool = True,                              # BN in conv blocks
        min_spatial_size: Optional[int] = None,                   # encoder target size
    ):
        super().__init__()

        # --------- stash basic config ---------
        self.input_dim = input_dim
        self.latent_dim = int(latent_dim)
        self.block_type = block_type
        self.model_type = model_type
        self.in_channels = int(input_dim[0])

        # --------- derive depth / schedule ---------
        _, H, W = input_dim
        base_channels = _auto_base_channels(input_dim) if base_channels is None else base_channels
        min_spatial_size = _auto_min_spatial_size(input_dim) if min_spatial_size is None else min_spatial_size

        # Count stride-2 downsamples to reach target size
        self.num_down_layers = self._calc_layers(min(H, W), min_spatial_size)

        # Default channel schedule like [1,2,4,8,...] if none provided
        if channel_multipliers is None:
            default = [1, 2, 4, 8]
            if self.num_down_layers > len(default):
                extra = [default[-1] * (2 ** i) for i in range(self.num_down_layers - len(default))]
                channel_multipliers = default + extra
            else:
                channel_multipliers = default[: self.num_down_layers]

        # --------- build trunk ---------
        layers = nn.ModuleList()
        in_ch = self.in_channels
        out_ch = in_ch  # will be set in loop

        for mult in channel_multipliers:
            out_ch = int(base_channels * mult)
            layers.append(self._down_block(in_ch, out_ch, use_batch_norm))
            in_ch = out_ch

        # Optional residual bottleneck stack (post-conv)
        if self.block_type == "resnet" and num_res_blocks > 0:
            bottleneck = max(out_ch // 4, 32)
            layers.append(nn.Sequential(*[
                ResBlock(in_channels=out_ch, bottleneck_channels=bottleneck) for _ in range(num_res_blocks)
            ]))

        self.layers = layers
        self.depth = len(layers)

        # --------- build heads ---------
        final_h = H // (2 ** self.num_down_layers)
        final_w = W // (2 ** self.num_down_layers)
        self.final_spatial = (final_h, final_w)
        self.final_channels = out_ch
        self.flat_feat = out_ch * final_h * final_w

        self._build_heads()

    # -- building blocks --

    def _calc_layers(self, min_dim: int, target: int) -> int:
        """
        How many halving steps do we need to go from min_dim to target?
        Always at least 2 layers for a minimal hierarchy.
        """
        n, cur = 0, min_dim
        while cur > target:
            cur //= 2
            n += 1
        return max(n, 2)

    def _down_block(self, in_ch: int, out_ch: int, use_bn: bool) -> nn.Module:
        """
        A simple stride-2 downsampling block:
          Conv2d(k=4, s=2, p=1) → (BN) → ReLU
        """
        ops = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)]
        if use_bn:
            ops.append(nn.BatchNorm2d(out_ch))
        ops.append(nn.ReLU(inplace=True))
        return nn.Sequential(*ops)

    def _build_heads(self) -> None:
        """
        Heads for each model family.
        """
        if self.model_type == "ae":
            self.fc_mu = nn.Linear(self.flat_feat, self.latent_dim)

        elif self.model_type == "vae":
            self.fc_mu = nn.Linear(self.flat_feat, self.latent_dim)
            self.fc_logvar = nn.Linear(self.flat_feat, self.latent_dim)

        elif self.model_type == "svae":
            self.fc_mu = nn.Linear(self.flat_feat, self.latent_dim)
            self.fc_logconc = nn.Linear(self.flat_feat, 1)

        elif self.model_type == "vqvae":
            self.conv_lat = nn.Conv2d(self.final_channels, self.latent_dim, kernel_size=1, stride=1, bias=False)

        elif self.model_type == "hrqvae":
            fh, fw = self.final_spatial
            self.conv_lat = nn.Conv2d(self.final_channels, self.latent_dim, kernel_size=(fh, fw), stride=1, bias=False)

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    # -- forward --

    def forward(
        self,
        x: torch.Tensor,
        output_layer_levels: Optional[List[int]] = None
    ):
        """
        Encode a batch.

        If you don't request intermediates and model_type == "vae", returns:
            (mu, log_var)     # ✅ 2-tuple expected by your VAE

        Otherwise returns a dict with:
            "embedding" and model-specific extras, plus optional
            "embedding_layer_{k}" for selected layers.
        """
        # Decide whether we should emit a simple tuple (for VAE) or a rich dict
        vae_simple_return = (self.model_type == "vae" and output_layer_levels is None)

        # If returning a dict (advanced mode), collect into a plain Python dict
        out_dict: Dict[str, torch.Tensor] = {} if not vae_simple_return else None  # type: ignore

        # Figure out how deep to run (intermediate requests may stop earlier)
        max_depth = self.depth
        if output_layer_levels is not None:
            assert all(self.depth >= l > 0 or l == -1 for l in output_layer_levels), \
                f"Requested layers {output_layer_levels} exceed depth={self.depth}"
            max_depth = self.depth if (-1 in output_layer_levels) else max(output_layer_levels)

        # Run trunk
        h = x
        for i in range(max_depth):
            h = self.layers[i](h)

            # If returning dict and intermediates are requested, stash them
            if (not vae_simple_return) and output_layer_levels is not None and (i + 1) in output_layer_levels:
                out_dict[f"embedding_layer_{i + 1}"] = h  # type: ignore

            # Emit heads after final trunk layer
            if i + 1 == self.depth:
                if self.model_type == "ae":
                    mu = self.fc_mu(h.reshape(x.size(0), -1))
                    if vae_simple_return:
                        # Not possible since model_type != "vae"; just for completeness
                        return mu
                    out_dict["embedding"] = mu  # type: ignore

                elif self.model_type == "vae":
                    flat = h.reshape(x.size(0), -1)
                    mu = self.fc_mu(flat)
                    log_var = self.fc_logvar(flat)
                    if vae_simple_return:
                        # ✅ Return exactly what your VAE module expects
                        return mu, log_var
                    out_dict["embedding"] = mu  # type: ignore
                    out_dict["log_covariance"] = log_var  # type: ignore

                elif self.model_type == "svae":
                    flat = h.reshape(x.size(0), -1)
                    mu = self.fc_mu(flat)
                    log_conc = self.fc_logconc(flat)
                    if vae_simple_return:
                        # Not possible since model_type != "vae"; just for completeness
                        return mu, log_conc
                    out_dict["embedding"] = mu  # type: ignore
                    out_dict["log_concentration"] = log_conc  # type: ignore

                elif self.model_type in ("vqvae", "hrqvae"):
                    emb = self.conv_lat(h)
                    if vae_simple_return:
                        # Not possible since model_type != "vae"; just for completeness
                        return emb
                    out_dict["embedding"] = emb  # type: ignore

        # If we are here, we are returning the rich dict
        return out_dict  # type: ignore


# ============================== DECODER ================================

class GenericDecoder(nn.Module):
    """
    A compact, auto-scaling decoder with VAE-friendly return types.

    --- IMPORTANT RETURN BEHAVIOR ---
    • If model_type == "vae" and output_layer_levels is None:
        returns reconstruction  # a tensor  ✅ VAE-friendly
    • Otherwise (different model_type or you request intermediates):
        returns a dict with at least "reconstruction" and optionally
        "reconstruction_layer_{k}" for intermediates.
    """

    def __init__(
        self,
        input_dim: Tuple[int, int, int],                          # (C, H, W)
        latent_dim: int,                                          # size of z
        block_type: Literal["conv", "resnet"] = "conv",
        model_type: Literal["ae", "vae", "svae", "vqvae", "hrqvae"] = "ae",
        base_channels: Optional[int] = None,
        channel_multipliers: Optional[List[float]] = None,
        num_res_blocks: int = 2,
        use_batch_norm: bool = True,
        min_spatial_size: Optional[int] = None,
        final_activation: Literal["sigmoid", "tanh"] = "sigmoid",
    ):
        super().__init__()

        # --------- stash basic config ---------
        self.input_dim = input_dim
        self.latent_dim = int(latent_dim)
        self.block_type = block_type
        self.model_type = model_type
        self.out_channels = int(input_dim[0])
        self.final_activation = final_activation

        # --------- derive depth / schedule ---------
        _, H, W = input_dim
        base_channels = _auto_base_channels(input_dim) if base_channels is None else base_channels
        min_spatial_size = _auto_min_spatial_size(input_dim) if min_spatial_size is None else min_spatial_size

        # Count stride-2 upsamples required to recover (H, W)
        self.num_up_layers = self._calc_layers(min(H, W), min_spatial_size)

        # Default decreasing schedule like [8,4,2,1,...]
        if channel_multipliers is None:
            default = [8, 4, 2, 1]
            if self.num_up_layers > len(default):
                extra = [max(1, default[-1] // (2 ** i)) for i in range(self.num_up_layers - len(default))]
                channel_multipliers = default + extra
            else:
                channel_multipliers = default[: self.num_up_layers]
        else:
            # If you passed the encoder schedule, reverse it for decoding
            channel_multipliers = list(reversed(channel_multipliers))

        # Starting feature map size & channels
        self.start_h = H // (2 ** self.num_up_layers)
        self.start_w = W // (2 ** self.num_up_layers)
        self.start_channels = int(base_channels * channel_multipliers[0])

        # --------- build trunk ---------
        layers = nn.ModuleList()

        # Initial projection from latent to a feature map
        if self.model_type == "vqvae":
            # Latent already has spatial dims: (B, D, h', w')
            layers.append(nn.ConvTranspose2d(self.latent_dim, self.start_channels, kernel_size=1, bias=False))
        elif self.model_type == "hrqvae":
            # Latent is (B, D, 1, 1): expand to (start_h, start_w)
            layers.append(nn.ConvTranspose2d(self.latent_dim, self.start_channels,
                                             kernel_size=(self.start_h, self.start_w), bias=False))
        else:
            # AE/VAE/SVAE: latent is vector -> Linear then reshape in forward()
            layers.append(nn.Linear(self.latent_dim, self.start_channels * self.start_h * self.start_w))

        in_ch = self.start_channels

        # Optional residual stack before upsampling
        if self.block_type == "resnet" and num_res_blocks > 0:
            bottleneck = max(in_ch // 4, 32)
            layers.append(nn.Sequential(*[
                ResBlock(in_channels=in_ch, bottleneck_channels=bottleneck) for _ in range(num_res_blocks)
            ], nn.ReLU(inplace=True)))

        # Progressive upsampling back to image size
        for i, mult in enumerate(channel_multipliers[1:] + [1]):  # "+[1]" ensures final jump to image space
            out_ch = int(base_channels * mult)
            is_last = i == (len(channel_multipliers) - 1)
            if is_last:
                layers.append(self._final_block(in_ch, self.out_channels))
            else:
                layers.append(self._up_block(in_ch, out_ch, use_batch_norm))
            in_ch = self.out_channels if is_last else out_ch

        self.layers = layers
        self.depth = len(layers)

    # -- building blocks --

    def _calc_layers(self, min_dim: int, target: int) -> int:
        """
        How many halving steps would the encoder do? Mirror that for decoding.
        """
        n, cur = 0, min_dim
        while cur > target:
            cur //= 2
            n += 1
        return max(n, 2)

    def _up_block(self, in_ch: int, out_ch: int, use_bn: bool) -> nn.Module:
        """
        A simple stride-2 upsampling block:
          ConvTranspose2d(k=3, s=2, p=1, op=1) → (BN) → ReLU
        """
        ops = [nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1, bias=False)]
        if use_bn:
            ops.append(nn.BatchNorm2d(out_ch))
        ops.append(nn.ReLU(inplace=True))
        return nn.Sequential(*ops)

    def _final_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """
        Final transposed conv to image channels + activation:
          - 'sigmoid' → [0, 1]
          - 'tanh'    → [-1, 1]
        """
        act = nn.Sigmoid() if self.final_activation == "sigmoid" else nn.Tanh()
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1, bias=False),
            act,
        )

    # -- forward --

    def forward(
        self,
        z: torch.Tensor,
        output_layer_levels: Optional[List[int]] = None
    ):
        """
        Decode a batch.

        If you don't request intermediates and model_type == "vae", returns:
            reconstruction   # ✅ a tensor expected by your VAE

        Otherwise returns a dict with:
            "reconstruction" and optionally "reconstruction_layer_{k}".
        """
        # VAE-friendly simple return?
        vae_simple_return = (self.model_type == "vae" and output_layer_levels is None)

        # Otherwise collect into a dict for advanced inspection
        out_dict: Dict[str, torch.Tensor] = {} if not vae_simple_return else None  # type: ignore

        # Decide how deep to run
        max_depth = self.depth
        if output_layer_levels is not None:
            assert all(self.depth >= l > 0 or l == -1 for l in output_layer_levels), \
                f"Requested layers {output_layer_levels} exceed depth={self.depth}"
            max_depth = self.depth if (-1 in output_layer_levels) else max(output_layer_levels)

        # Forward through layers
        h = z
        for i in range(max_depth):
            h = self.layers[i](h)

            # After first layer for vector latents, reshape Linear output to (B, C, H, W)
            if i == 0 and self.model_type not in ("vqvae", "hrqvae"):
                h = h.reshape(z.size(0), self.start_channels, self.start_h, self.start_w)

            # Save intermediates if returning dict and requested
            if (not vae_simple_return) and output_layer_levels is not None and (i + 1) in output_layer_levels:
                out_dict[f"reconstruction_layer_{i + 1}"] = h  # type: ignore

            # On the very last layer we have the full reconstruction
            if i + 1 == self.depth:
                if vae_simple_return:
                    # ✅ Return exactly what your VAE module expects
                    return h
                out_dict["reconstruction"] = h  # type: ignore

        # If here, return the dict (advanced mode)
        return out_dict  # type: ignore
