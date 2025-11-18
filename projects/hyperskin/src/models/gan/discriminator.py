import torch.nn as nn

class Discriminator(nn.Module):
    """
    Automatically builds a CNN-based discriminator that mirrors AutoGenerator.

    Example:
    >>> D = AutoDiscriminator((1, 28, 28))
    >>> x = torch.randn(16, 1, 28, 28)
    >>> out = D(x)
    >>> out.shape
    torch.Size([16, 1])
    """

    def __init__(self, img_shape):
        super().__init__()
        num_channels, H, W = img_shape

        layers = []
        current_channels = num_channels
        hidden_channels = [64, 128, 256, 512]

        current_size = min(H, W)
        i = 0

        # Keep downsampling until we reach <= 4x4
        while current_size > 4 and i < len(hidden_channels):
            out_channels = hidden_channels[i]
            layers += [
                nn.Conv2d(
                    current_channels, out_channels, kernel_size=4,
                    stride=2, padding=1
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            # Add batchnorm after first layer if >64 channels
            if i > 0:
                layers.insert(-1, nn.BatchNorm2d(out_channels))

            current_size = current_size // 2
            current_channels = out_channels
            i += 1

        self.conv = nn.Sequential(*layers)

        # Flatten and classify
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels * current_size * current_size, 1)
        )

    def forward(self, img):
        x = self.conv(img)
        return self.fc(x)
