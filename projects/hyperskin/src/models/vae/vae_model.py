
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