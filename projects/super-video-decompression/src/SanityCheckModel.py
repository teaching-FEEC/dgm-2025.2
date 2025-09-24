import torch
import torch.nn as nn

class Identity3Channel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: [B, 3, H, W] RGB image
        returns: same tensor as input
        """
        return x