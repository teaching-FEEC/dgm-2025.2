import torch.nn as nn
import timm
import segmentation_models_pytorch as smp

class SMPModel(nn.Module):
    def __init__(self, arch_name: str, encoder_name: str, num_classes: int, in_chans: int,
                 pretrained: bool, **kwargs):
        super().__init__()
        self.model = smp.create_model(arch=arch_name, encoder_name=encoder_name,
                                     in_channels=in_chans, classes=num_classes,
                                     encoder_weights='imagenet' if pretrained else None,
                                     **kwargs)

    def forward(self, x):
        return self.model(x)
