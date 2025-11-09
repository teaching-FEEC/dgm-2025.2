from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

from src.data_modules.datasets.task_config import DatasetSample, TaskConfig


# Predefined task configs
MILK10K_TASK_CONFIGS = {
    "multilabel_classification": TaskConfig(return_label=True, label_type="multilabel"),
    "binary_classification": TaskConfig(return_label=True, binary_classification=True, label_type="binary"),
    "segmentation": TaskConfig(return_label=True, return_mask=True, label_type="multilabel"),
    "generation":  TaskConfig(return_image=True),
    "generation_unconditional": TaskConfig(return_image=True),
    "generation_conditional_label": TaskConfig(return_image=True, return_label=True, label_type="multilabel"),
    "generation_conditional_mask": TaskConfig(return_image=True, return_mask=True),
    "generation_conditional_full": TaskConfig(
        return_image=True, return_label=True, return_mask=True, label_type="multilabel"
    ),
    "generation_melanoma_vs_nevus": TaskConfig(
        return_image=True, binary_classification=True, label_type="binary"
    ),
}

class MILK10kDataset(Dataset):
    """
    Flexible MILK10k dataset supporting multiple tasks.

    Args:
        root_dir: Root directory containing the dataset
        transform: Albumentations transform
        task: TaskConfig or string key from TASK_CONFIGS
        dermoscopic_only: Filter to dermoscopic images only
    """

    def __init__(
        self,
        root_dir: str,
        transform: A.Compose | None = None,
        task: TaskConfig | str = "multilabel_classification",
        dermoscopic_only: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.dermoscopic_only = dermoscopic_only

        # Handle task config
        if isinstance(task, str):
            if task not in MILK10K_TASK_CONFIGS:
                raise ValueError(f"Unknown task: {task}. Available: {list(MILK10K_TASK_CONFIGS.keys())}")
            self.task_config = MILK10K_TASK_CONFIGS[task]
        else:
            self.task_config = task

        # Load and prepare data
        self._load_data()
        self._setup_classes()
        self._filter_data()

    def _load_data(self):
        """Load metadata and ground truth CSVs."""
        gt_path = self.root_dir / "MILK10k_Training_GroundTruth.csv"
        meta_path = self.root_dir / "MILK10k_Training_Metadata.csv"

        self.gt_df = pd.read_csv(gt_path)
        self.meta_df = pd.read_csv(meta_path)
        self.data = pd.merge(self.meta_df, self.gt_df, on="lesion_id")

    def _setup_classes(self):
        """Setup class mappings."""
        self.class_map = {
            "AKIEC": "actinic_keratosis_intraepidermal_carcinoma",
            "BCC": "basal_cell_carcinoma",
            "BEN_OTH": "other_benign_proliferations",
            "BKL": "benign_keratinocytic_lesion",
            "DF": "dermatofibroma",
            "INF": "inflammatory_infectious_conditions",
            "MAL_OTH": "other_malignant_proliferations",
            "MEL": "melanoma",
            "NV": "melanocytic_nevus",
            "SCCKA": "squamous_cell_carcinoma_keratoacanthoma",
            "VASC": "vascular_lesions_hemorrhage",
        }
        self.class_codes = list(self.class_map.keys())
        self.class_names = [self.class_map[c] for c in self.class_codes]

    def _filter_data(self):
        """Filter dataset based on task requirements."""
        # Filter dermoscopic images
        if self.dermoscopic_only:
            self.data = self.data[self.data["image_type"] == "dermoscopic"].reset_index(drop=True)

        # Expand for multiple crops per lesion
        self._expand_crops()

        # Add masks column
            # used in DatasetExporter
        self.data["masks"] = self.data["image_path"].apply(self.find_masks)

        # Task-specific filtering
        if self.task_config.binary_classification:
            # Filter to melanoma vs nevus only
            mask = (self.data["MEL"] == 1) | (self.data["NV"] == 1)
            self.data = self.data[mask].reset_index(drop=True)
            self.class_names = ["melanoma", "nevus"]

        if self.task_config.return_mask:
            # Remove samples without masks
            self.data = self.data.dropna(subset=["masks"]).reset_index(drop=True)

    def _expand_crops(self):
        """Expand dataset to include all cropped variants."""
        expanded_rows = []

        for _, row in self.data.iterrows():
            lesion_dir = self.root_dir / "images" / row["lesion_id"]
            crop_matches = sorted(lesion_dir.glob(f"{row['isic_id']}_crop*.jpg"))

            if not crop_matches:
                unsuffixed = lesion_dir / f"{row['isic_id']}.jpg"
                if unsuffixed.exists():
                    row["image_path"] = str(unsuffixed)
                    expanded_rows.append(row.copy())
                continue

            for crop_img in crop_matches:
                new_row = row.copy()
                new_row["image_path"] = str(crop_img)
                expanded_rows.append(new_row)

        self.data = pd.DataFrame(expanded_rows).reset_index(drop=True)

    # used in DatasetExporter
    def find_masks(self, image_path: str) -> str | None:
        """Find associated mask files for an image."""
        masks_dir = self.root_dir / "masks"
        if not masks_dir.exists():
            return None

        image_path = Path(image_path)
        isic_id = image_path.stem.split("_crop")[0]
        base_path = masks_dir / image_path.parent.name

        masks = sorted(base_path.glob(f"{isic_id}_crop*_mask.png"))
        if not masks:
            plain_mask = base_path / f"{isic_id}_mask.png"
            if plain_mask.exists():
                masks = [plain_mask]

        return ";".join(map(str, masks)) if masks else None

    def get_masks_list(self, idx: int) -> list[str]:
        """Get list of mask paths for a given index."""
        mask_str = self.data.iloc[idx]["masks"]
        if pd.isna(mask_str):
            return []
        return mask_str.split(";")

    def _load_image(self, idx: int) -> np.ndarray:
        """Load image as RGB numpy array."""
        path = self.data.iloc[idx]["image_path"]
        return np.array(Image.open(path).convert("RGB"))

    def _load_mask(self, idx: int) -> np.ndarray | None:
        """Load and combine all masks for a sample."""
        mask_str = self.data.iloc[idx]["masks"]
        if pd.isna(mask_str):
            return None

        mask_paths = mask_str.split(";")
        masks = [np.array(Image.open(p).convert("L"), dtype=np.uint8) for p in mask_paths]

        # Combine multiple masks
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = np.maximum(combined, mask)

        return (combined > 0).astype(np.uint8)

    def _get_label(self, idx: int) -> np.ndarray:
        """Extract label based on task configuration."""
        row = self.data.iloc[idx]

        if self.task_config.binary_classification:
            return 0 if row["MEL"] == 1 else 1
        else:
            return row[self.class_codes].astype(float).values

    def __len__(self) -> int:
        return len(self.data)

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
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Convert mask to tensor
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.long)

        # Build sample
        sample = DatasetSample(image=image)

        if self.task_config.return_label:
            label = self._get_label(idx)
            dtype = torch.long if self.task_config.label_type == "binary" else torch.float
            sample.label = torch.tensor(label, dtype=dtype)

        if self.task_config.return_mask:
            sample.mask = mask

        return_sample = sample.to_tuple()
        if len(return_sample) == 1:
            return return_sample[0]
        return return_sample

    @property
    def labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return np.array([self._get_label(i) for i in range(len(self))])
