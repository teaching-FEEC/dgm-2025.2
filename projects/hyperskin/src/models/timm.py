import torch.nn as nn
import timm

def freeze_backbone_unfreeze_classifier(model):
    """
    Freezes the backbone/encoder of a timm model and leaves only the classifier trainable.
    Works across different timm architectures.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Try to unfreeze classifier using timm API
    try:
        classifier = model.get_classifier()
    except Exception:
        classifier = None

    if classifier is not None:
        # Unfreeze the classifier
        for param in classifier.parameters():
            param.requires_grad = True
    else:
        # Fallback: look for common classifier attributes
        if hasattr(model, "fc"):  # ResNets
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, "head"):  # EfficientNet, ViT, ConvNeXt, etc.
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Could not find classifier head in this model.")

    return model

class TIMMModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, in_chans: int,
                 pretrained: bool, features_only: str,
                 scriptable: bool, freeze_backbone: bool = False, **kwargs):
        super().__init__()
        self.model = timm.create_model(model_name, num_classes=num_classes,
                                       in_chans=in_chans, pretrained=pretrained,
                                       features_only=features_only,
                                       scriptable=scriptable,
                                       **kwargs)
        if freeze_backbone:
            self.model = freeze_backbone_unfreeze_classifier(self.model)

    def forward(self, x):
        return self.model(x)
