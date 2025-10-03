import importlib
from typing import Optional
import torch.nn as nn
import timm

def unfreeze_norm_layers(model: nn.Module):
    """Unfreeze all normalization layers (BN, LN, GN, etc.)."""
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.SyncBatchNorm,
    )
    for module in model.modules():
        if isinstance(module, norm_types):
            unfreeze_module(module)
    return model

def freeze_module(module: nn.Module):
    """Freeze all params in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module):
    """Unfreeze all params in a module."""
    for param in module.parameters():
        param.requires_grad = True


def get_module_from_path(model: nn.Module, layer_path: str) -> nn.Module:
    """Navigate a dotted path into the model and return the submodule."""
    module = model
    for attr in layer_path.split("."):
        if not hasattr(module, attr):
            raise ValueError(
                f"Invalid layer path: '{layer_path}', missing '{attr}'"
            )
        module = getattr(module, attr)
    return module


def freeze_backbone_only(model: nn.Module):
    """Freeze the entire backbone then unfreeze just the classifier head."""
    for param in model.parameters():
        param.requires_grad = False

    # Try timm classifier accessor
    try:
        classifier = model.get_classifier()
        if classifier is not None:
            unfreeze_module(classifier)
    except Exception:
        # Fallback to common classifier attributes
        for attr in ["fc", "head", "classifier"]:
            if hasattr(model, attr):
                unfreeze_module(getattr(model, attr))
                break
    return model


def freeze_specific_layers(model: nn.Module, layers: list[str]):
    for layer_path in layers:
        module = get_module_from_path(model, layer_path)
        freeze_module(module)
    return model


def unfreeze_specific_layers(model: nn.Module, layers: list[str]):
    for layer_path in layers:
        module = get_module_from_path(model, layer_path)
        unfreeze_module(module)
    return model


class TIMMModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        in_chans: int,
        pretrained: bool = False,
        features_only: bool = False,
        scriptable: bool = False,
        freeze_backbone: bool = False,
        freeze_layers: Optional[list[str]] = None,
        unfreeze_layers: Optional[list[str]] = None,
        custom_head: Optional[list] = None,
        unfreeze_norm: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            num_classes=num_classes,
            in_chans=in_chans,
            pretrained=pretrained,
            features_only=features_only,
            scriptable=scriptable,
            **kwargs,
        )

        if custom_head is not None:
            self._replace_classifier_with_custom_head(custom_head)

        # Apply freezing rules
        if freeze_backbone:
            self.model = freeze_backbone_only(self.model)

        if unfreeze_norm:
            self.model = unfreeze_norm_layers(self.model)

        if freeze_layers is not None:
            self.model = freeze_specific_layers(self.model, freeze_layers)

        if unfreeze_layers is not None:
            self.model = unfreeze_specific_layers(self.model, unfreeze_layers)

    def _get_classifier_info(self):
        """Get classifier module and its attribute name."""
        classifier_name = None
        if hasattr(self.model, "default_cfg"):
            classifier_name = self.model.default_cfg.get("classifier")

        if classifier_name is None:
            for name in ["head", "fc", "classifier"]:
                if hasattr(self.model, name):
                    classifier_name = name
                    break

        if classifier_name is None:
            raise ValueError("Could not determine classifier attribute name")

        classifier = getattr(self.model, classifier_name)
        return classifier, classifier_name

    def _get_in_features(self, classifier):
        """Extract input features from classifier module."""
        if isinstance(classifier, nn.Linear):
            return classifier.in_features
        if hasattr(classifier, "in_features"):
            return classifier.in_features

        for m in reversed(list(classifier.modules())):
            if hasattr(m, "in_features"):
                return m.in_features

        raise ValueError("Could not determine classifier input dimension")

    def _replace_classifier_with_custom_head(self, custom_head):
        """Replace model's classifier with custom head in-place."""
        classifier, classifier_name = self._get_classifier_info()
        in_features = self._get_in_features(classifier)

        head_layers = []
        for i, layer_cfg in enumerate(custom_head):
            class_path = layer_cfg["class_path"]
            init_args = layer_cfg.get("init_args", {}).copy()

            if i == 0 and "Linear" in class_path and "in_features" not in init_args:
                init_args["in_features"] = in_features

            module_name, class_name = class_path.rsplit(".", 1)
            cls = getattr(importlib.import_module(module_name), class_name)
            head_layers.append(cls(**init_args))

        new_head = nn.Sequential(*head_layers)
        setattr(self.model, classifier_name, new_head)

    def forward(self, x):
        return self.model(x)
