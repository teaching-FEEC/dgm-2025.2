from enum import IntEnum, auto
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from PIL import Image
import albumentations as A

from src.data_modules.datasets.task_config import DatasetSample, TaskConfig

class HSIDermoscopyTask(IntEnum):
    CLASSIFICATION_ALL_CLASSES = auto()
    CLASSIFICATION_MELANOMA_VS_OTHERS = auto()
    CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_VS_OTHERS = auto()
    CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_NEVI = auto()
    SEGMENTATION = auto()
    GENERATION = auto()

# Base label mappings
FULL_LABELS_MAP = {
    "dysplastic_nevi": 0,
    "melanoma": 1,
    "solar_lentigo": 2,
    "interepidermal_carcinoma": 3,
    "nevi": 4,
    "seborrheic_keratosis": 5,
    "melanocytic_lesion": 6,
}

OTHERS_LABELS_MAP = {
    "IEC": "interepidermal_carcinoma",
    "Lentigo": "solar_lentigo",
    "Nv": "nevi",
    "SK": "seborrheic_keratosis",
    "Melt": "melanocytic_lesion",
}


# Predefined task configurations
HSI_TASK_CONFIGS = {
    # Classification tasks
    "classification_all_classes": TaskConfig(
        return_label=True, label_mapping=FULL_LABELS_MAP
    ),
    "classification_melanoma_vs_others": TaskConfig(
        return_label=True, label_mapping={"melanoma": 0, "others": 1}, binary_classification=True
    ),
    "classification_melanoma_vs_dysplastic_vs_others": TaskConfig(
        return_label=True,
        label_mapping={"melanoma": 0, "dysplastic_nevi": 1, "others": 2},
    ),
    "classification_melanoma_vs_dysplastic_nevi": TaskConfig(
        return_label=True,
        label_mapping={"melanoma": 0, "dysplastic_nevi": 1},
        binary_classification=True,
        filter_classes=["melanoma", "dysplastic_nevi"],
    ),
    # Segmentation task
    "segmentation": TaskConfig(
        return_label=True, return_mask=True, label_mapping=FULL_LABELS_MAP
    ),
    # Generation tasks
    "generation_unconditional": TaskConfig(return_image=True),
    "generation_conditional_label": TaskConfig(
        return_image=True, return_label=True, label_mapping=FULL_LABELS_MAP
    ),
    "generation_conditional_mask": TaskConfig(return_image=True, return_mask=True),
    "generation_conditional_full": TaskConfig(
        return_image=True,
        return_label=True,
        return_mask=True,
        label_mapping=FULL_LABELS_MAP,
    ),
    "generation": TaskConfig(return_image=True,
                             binary_classification=True,
                             label_mapping={"melanoma": 0, "dysplastic_nevi": 1},
                             filter_classes=["melanoma", "dysplastic_nevi"]
                             ),
}


class HSIDermoscopyDataset(Dataset):
    """
    Flexible HSI dermoscopy dataset supporting multiple tasks.

    Args:
        data_dir: Root directory containing the dataset
        transform: Albumentations transform
        task: TaskConfig or string key from HSI_TASK_CONFIGS
        force_create_df: Force recreation of metadata CSV
        save_labels_df: Save metadata CSV to disk
    """

    def __init__(
        self,
        data_dir: str = "data/hsi_dermoscopy",
        transform: A.Compose | None = None,
        task: TaskConfig | str = "classification_all_classes",
        force_create_df: bool = True,
        save_labels_df: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Handle task config
        if isinstance(task, str):
            if task not in HSI_TASK_CONFIGS:
                raise ValueError(f"Unknown task: {task}. Available: {list(HSI_TASK_CONFIGS.keys())}")
            self.task_config = HSI_TASK_CONFIGS[task]
        else:
            self.task_config = task

        # Load or create metadata
        self.metadata_path = self.data_dir / "metadata.csv"
        self._load_or_create_metadata(force_create_df, save_labels_df)

        # Setup labels and filtering based on task
        self._setup_task()

    def _load_or_create_metadata(self, force_create: bool, save: bool):
        """Load existing metadata or create new."""
        if self.metadata_path.exists() and not force_create:
            self.labels_df = pd.read_csv(self.metadata_path)
        else:
            self.labels_df = self._create_metadata()
            if save:
                self.labels_df.to_csv(self.metadata_path, index=False)

    def _create_metadata(self) -> pd.DataFrame:
        """Create metadata DataFrame by scanning directory structure."""
        images_dir = self.data_dir / "images"

        paths = {
            "dysplastic_nevi": images_dir / "DNCube",
            "melanoma": images_dir / "MMCube",
            "others": images_dir / "OtherCube",
        }

        # Validate paths
        for label, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing directory: {path}")

        data = []

        # Process dysplastic nevi and melanoma
        for label in ["dysplastic_nevi", "melanoma"]:
            for mat_file in paths[label].glob("*.mat"):
                masks = self.find_masks(mat_file)
                data.append({"file_path": str(mat_file), "label": label, "masks": masks, "filename": mat_file.stem})

        # Process other lesions
        for mat_file in paths["others"].glob("*.mat"):
            filename = mat_file.stem
            label = self._extract_other_label(filename)
            if label:
                masks = self.find_masks(mat_file)
                data.append({"file_path": str(mat_file), "label": label, "masks": masks, "filename": filename})

        return pd.DataFrame(data)

    def _extract_other_label(self, filename: str) -> str | None:
        """Extract label from filename for 'other' category."""
        for short_label, full_label in OTHERS_LABELS_MAP.items():
            if short_label in filename:
                return full_label
        return None

    def find_masks(self, mat_file: Path) -> str | None:
        """
        Find all masks for a given .mat file.
        Returns semicolon-separated paths or None.
        """
        base_mask_path = str(mat_file).replace("images", "masks").replace(".mat", "")

        masks = []

        # Try finding masks with _crop00, _crop01, etc. suffixes
        idx = 0
        while True:
            mask_path = f"{base_mask_path}_crop{idx:02d}_mask.png"
            if Path(mask_path).exists():
                masks.append(mask_path)
                idx += 1
            else:
                break

        # If no numbered masks found, try the exact name
        if not masks:
            mask_path = f"{base_mask_path}_mask.png"
            if Path(mask_path).exists():
                masks.append(mask_path)

        return ";".join(masks) if masks else None

    def _setup_task(self):
        """Setup labels and filter data based on task configuration."""
        # Apply label mapping transformations
        if self.task_config.label_mapping:
            # Check if we need to collapse to "others"
            if "others" in self.task_config.label_mapping:
                self._apply_others_mapping()

            self.labels_map = self.task_config.label_mapping
        else:
            self.labels_map = FULL_LABELS_MAP

        # Filter classes if specified
        if self.task_config.filter_classes:
            original_len = len(self.labels_df)
            self.labels_df = self.labels_df[self.labels_df["label"].isin(self.task_config.filter_classes)].reset_index(
                drop=True
            )
            filtered_count = original_len - len(self.labels_df)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} samples not in {self.task_config.filter_classes}")

        # Filter samples without masks for segmentation tasks
        if self.task_config.return_mask:
            original_len = len(self.labels_df)
            self.labels_df = self.labels_df.dropna(subset=["masks"]).reset_index(drop=True)
            filtered_count = original_len - len(self.labels_df)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} samples without masks for segmentation/generation task")

    def _apply_others_mapping(self):
        """Apply label mapping that collapses classes to 'others'."""

        def map_to_others(label: str) -> str:
            # Get the label mapping keys (the classes we care about)
            kept_classes = [k for k in self.task_config.label_mapping.keys() if k != "others"]
            return label if label in kept_classes else "others"

        self.labels_df["label"] = self.labels_df["label"].apply(map_to_others)

    def _load_image(self, idx: int) -> np.ndarray:
        """Load HSI image from .mat file."""
        file_path = self.labels_df.iloc[idx]["file_path"]
        mat_data = loadmat(file_path)
        # loadmat returns dict, get the last item's value (the actual data)
        image = mat_data.popitem()[-1]
        return image.astype(np.float32)

    def _load_mask(self, idx: int) -> np.ndarray | None:
        """Load and combine all masks for a sample."""
        masks_str = self.labels_df.iloc[idx]["masks"]
        if pd.isna(masks_str):
            return None

        mask_paths = masks_str.split(";")
        masks = []
        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask, dtype=np.uint8)
            masks.append(mask)

        # Combine multiple masks using maximum (logical OR)
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = np.maximum(combined, mask)

        return (combined > 0).astype(np.uint8)

    def _get_label(self, idx: int) -> int:
        """Get integer label for a sample."""
        label_str = self.labels_df.iloc[idx]["label"]
        return self.labels_map[label_str]

    def get_masks_list(self, idx: int) -> list[str]:
        """Get list of mask paths for a given index."""
        masks_str = self.labels_df.iloc[idx]["masks"]
        if pd.isna(masks_str):
            return []
        return masks_str.split(";")

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a sample from the dataset.

        Returns:
            tuple: (image, [mask,] [label,])
        """
        # Load image
        image = self._load_image(idx)
        mask = None

        # Load mask if needed
        if self.task_config.return_mask:
            mask = self._load_mask(idx)
            if mask is None:
                raise ValueError(f"Mask required but not found for idx {idx}")

        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                transformed = self.transform(image=image)
                image = transformed["image"]
        else:
            # Convert to tensor if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Convert mask to tensor
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.long)

        # Build sample
        sample = DatasetSample(image=image)

        if self.task_config.return_label:
            label = self._get_label(idx)
            sample.label = torch.tensor(label, dtype=torch.long)

        if self.task_config.return_mask:
            sample.mask = mask

        return sample.to_dict()

    @property
    def labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return np.array([self._get_label(i) for i in range(len(self))])

    @property
    def num_classes(self) -> int:
        """Number of classes in current task."""
        return len(self.labels_map)

    @property
    def class_names(self) -> list[str]:
        """Get class names in label order."""
        return sorted(self.labels_map.keys(), key=lambda k: self.labels_map[k])


if __name__ == "__main__":
    # Example 1: Classification (all classes)
    dataset = HSIDermoscopyDataset(
        data_dir="data/hsi_dermoscopy",
        task_config="classification_all_classes",
        transform=A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(),
                A.pytorch.ToTensorV2(),
            ]
        ),
    )
    print(f"All classes: {len(dataset)} samples, {dataset.num_classes} classes")
    sample = dataset[0]
    print(f"Sample: image shape={sample.image.shape}, label={sample.label}")

    # Example 2: Melanoma vs Others
    dataset = HSIDermoscopyDataset(
        data_dir="data/hsi_dermoscopy",
        task_config="classification_melanoma_vs_others",
    )
    print(f"\nMelanoma vs Others: {len(dataset)} samples")
    print(f"Classes: {dataset.class_names}")

    # Example 3: Segmentation
    dataset = HSIDermoscopyDataset(
        data_dir="data/hsi_dermoscopy_croppedv2_256_with_masks",
        task_config="segmentation",
    )
    print(f"\nSegmentation: {len(dataset)} samples")
    sample = dataset[0]
    print(f"Sample: image={sample.image.shape}, mask={sample.mask.shape}, label={sample.label}")

    # Example 4: Unconditional Generation
    dataset = HSIDermoscopyDataset(
        data_dir="data/hsi_dermoscopy",
        task_config="generation_unconditional",
    )
    print(f"\nUnconditional generation: {len(dataset)} samples")
    sample = dataset[0]
    print(f"Sample: image shape={sample.image.shape}")

    # Example 5: Conditional Generation (full)
    dataset = HSIDermoscopyDataset(
        data_dir="data/hsi_dermoscopy_croppedv2_256_with_masks",
        task_config="generation_conditional_full",
    )
    print(f"\nConditional generation (full): {len(dataset)} samples")
    sample = dataset[0]
    print(f"Sample: image={sample.image.shape}, mask={sample.mask.shape}, label={sample.label}")
