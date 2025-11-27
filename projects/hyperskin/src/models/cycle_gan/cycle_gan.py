from torch import nn
import torch


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels, kernel_size),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels, kernel_size),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 9,
    ):
        super().__init__()
        self.model = nn.Sequential(
            self._initial_block(in_channels=in_channels, out_channels=64),
            *self._downsampling_blocks(in_channels=64, num_blocks=2),
            *self._residual_blocks(in_channels=256, num_blocks=num_res_blocks),
            *self._upsampling_blocks(in_channels=256, num_blocks=2),
            self._output_block(in_channels=64, out_channels=out_channels),
        )

    def _initial_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def _downsampling_blocks(
        self,
        in_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            blocks.append(nn.InstanceNorm2d(in_channels * 2))
            blocks.append(nn.LeakyReLU(0.2))
            in_channels *= 2
        return blocks

    def _residual_blocks(self, in_channels: int, num_blocks: int):
        return [ResidualBlock(in_channels) for _ in range(num_blocks)]

    def _upsampling_blocks(
        self,
        in_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.ConvTranspose2d(
                    in_channels,
                    in_channels // 2,
                    kernel_size,
                    stride,
                    padding,
                    output_padding=1,
                )
            )
            blocks.append(nn.InstanceNorm2d(in_channels // 2))
            blocks.append(nn.LeakyReLU(0.2))
            in_channels //= 2
        return blocks

    def _output_block(
        self,
        in_channels: int,
        out_channels: int,
    ):
        return nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Conv2d(in_channels, out_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(
            self._discriminator_block(in_channels, 64, stride=2),
            self._discriminator_block(64, 128, stride=2),
            self._discriminator_block(128, 256, stride=2),
            self._discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def _discriminator_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 1,
        padding: int = 1,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)
