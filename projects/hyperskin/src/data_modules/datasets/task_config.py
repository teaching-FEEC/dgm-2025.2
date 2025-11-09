from dataclasses import dataclass
from typing import Literal
import torch


@dataclass
class DatasetSample:
    """Structured container for dataset samples."""

    image: torch.Tensor
    label: torch.Tensor | None = None
    mask: torch.Tensor | None = None

    def to_tuple(self):
        """Convert to tuple in fixed order."""
        items = [self.image]
        if self.mask is not None:
            items.append(self.mask)
        if self.label is not None:
            items.append(self.label)
        return tuple(items)


@dataclass
class TaskConfig:
    """Configuration for what data to return."""

    return_image: bool = True
    return_label: bool = False
    return_mask: bool = False
    binary_classification: bool = False  # melanoma vs nevus
    label_type: Literal["multilabel", "binary", "class_index"] = "multilabel"
    label_mapping: dict[str, int] | None = None
    filter_classes: list[str] | None = None
