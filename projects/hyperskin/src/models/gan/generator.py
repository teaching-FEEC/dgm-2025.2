import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Automatically configured generator that upsamples until reaching
    the desired output shape.
    """

    def __init__(self, latent_dim: int, img_shape: tuple):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        num_channels, target_h, target_w = img_shape

        # Start from 4x4 feature map
        self.init_size = 4
        self.proj_channels = 512

        # Fully connected projection from latent vector → feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.proj_channels * self.init_size * self.init_size),
            nn.ReLU(True),
        )

        channels = [512, 256, 128, 64]

        layers = []
        current_channels = self.proj_channels
        current_size = self.init_size
        i = 1

        # Upsampling loop
        while current_size < min(target_h, target_w):
            out_channels = channels[min(i, len(channels) - 1)]
            layers += [
                nn.ConvTranspose2d(current_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            ]
            current_channels = out_channels
            current_size *= 2
            i += 1

        # Final conv to match channels (e.g. 16 for HSI)
        layers += [
            nn.ConvTranspose2d(current_channels, num_channels, 3, 1, 1),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        # Project and reshape latent vector
        out = self.fc(z)
        out = out.view(z.size(0), self.proj_channels, self.init_size, self.init_size)
        img = self.model(out)

        # Ensure exact resolution
        img = torch.nn.functional.interpolate(
            img, size=self.img_shape[1:], mode="bilinear", align_corners=False
        )
        return img