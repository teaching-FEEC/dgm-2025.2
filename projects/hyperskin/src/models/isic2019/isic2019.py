from typing import Optional
import torch
from torch import nn
from src.losses.lpips.pretrained_networks import adapt_input_conv
from src.models.isic2019.classifier import Net2D

class ISIC2019Model(nn.Module):
    def __init__(self, in_chans: int = 16, num_classes: int = 2, adapt_fc: bool = False,
                 freeze_backbone: bool = False,
                 weights_path: str = ''):
        super().__init__()
        # Initialize model architecture here
        self.model = Net2D()

        if weights_path:
            # Load the pretrained weights
            weights = torch.load(weights_path, map_location=lambda storage, loc: storage)
            weights = {k.replace('module.', '') : v for k,v in weights.items()}
            self.model.load_state_dict(weights)

        if in_chans != 3:
            # Modify the first convolutional layer to accept in_chans input channels
            first_conv = self.model.backbone.conv_stem
            with torch.no_grad():
                adapted_weight = adapt_input_conv(in_chans, first_conv.weight.data)
            new_conv = nn.Conv2d(
                in_chans,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            new_conv.weight.data = adapted_weight
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()
            self.model.backbone.conv_stem = new_conv
        if num_classes != 8 and adapt_fc:
            # Adapt the final fully connected layer to have num_classes outputs, while maintaining weights
            old_fc = self.model.fc
            in_features = old_fc.in_features
            new_fc = nn.Linear(in_features, 2)
            with torch.no_grad():
                new_fc.weight.data = old_fc.weight.data[:2, :].clone()
                new_fc.bias.data = old_fc.bias.data[:2].clone()
            self.model.fc = new_fc
        elif num_classes != 8 and not adapt_fc:
            # Replace the final fully connected layer with a new one
            old_fc = self.model.fc
            in_features = old_fc.in_features
            new_fc = nn.Linear(in_features, num_classes)
            self.model.fc = new_fc

        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)
