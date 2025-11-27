from torch import nn

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
