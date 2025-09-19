import torch.nn as nn
import timm

class TIMMModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, in_chans: int,
                 pretrained: bool, features_only: str,
                 scriptable: bool, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name, num_classes=num_classes,
                                       in_chans=in_chans, pretrained=pretrained,
                                       features_only=features_only,
                                       scriptable=scriptable,
                                       **kwargs)

    def forward(self, x):
        return self.model(x)
