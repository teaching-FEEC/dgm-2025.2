from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(
        Path(__file__).parent.parent.parent,
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=False,
    )

from src.data_modules.datasets.task_config import DatasetSample, TaskConfig


# Predefined task configs
ISIC2019_TASK_CONFIGS = {
    "multilabel_classification": TaskConfig(
        return_label=True, label_type="multilabel"
    ),
    "binary_classification": TaskConfig(
        return_label=True, binary_classification=True, label_type="binary"
    ),
    "segmentation": TaskConfig(
        return_label=True, return_mask=True, label_type="multilabel"
    ),
    "generation": TaskConfig(return_image=True),
    "generation_unconditional": TaskConfig(return_image=True),
    "generation_conditional_label": TaskConfig(
        return_image=True, return_label=True, label_type="multilabel"
    ),
    "generation_conditional_mask": TaskConfig(
        return_image=True, return_mask=True
    ),
    "generation_conditional_full": TaskConfig(
        return_image=True,
        return_label=True,
        return_mask=True,
        label_type="multilabel",
    ),
    "generation_melanoma_vs_nevus": TaskConfig(
        return_image=True, binary_classification=True, label_type="binary"
    ),
}


class ISIC2019Dataset(Dataset):
    """
    Flexible ISIC2019 dataset supporting multiple tasks.

    Args:
        root_dir: Root directory containing the dataset
        transform: Albumentations transform
        task: TaskConfig or string key from ISIC2019_TASK_CONFIGS
    """

    def __init__(
        self,
        root_dir: str,
        transform: A.Compose | None = None,
        task: TaskConfig | str = "multilabel_classification",
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Handle task config
        if isinstance(task, str):
            if task not in ISIC2019_TASK_CONFIGS:
                raise ValueError(
                    f"Unknown task: {task}. "
                    f"Available: {list(ISIC2019_TASK_CONFIGS.keys())}"
                )
            self.task_config = ISIC2019_TASK_CONFIGS[task]
        else:
            self.task_config = task

        # Load and prepare data
        self._load_data()
        self._setup_classes()
        self._filter_data()

    def _load_data(self):
        """Load metadata and ground truth CSVs."""
        gt_path = self.root_dir / "ISIC_2019_Training_GroundTruth.csv"
        meta_path = self.root_dir / "ISIC_2019_Training_Metadata.csv"

        self.gt_df = pd.read_csv(gt_path)
        self.meta_df = pd.read_csv(meta_path)
        self.data = pd.merge(self.meta_df, self.gt_df, on="image")

    def _setup_classes(self):
        """Setup class mappings."""
        self.class_map = {
            "MEL": "melanoma",
            "NV": "melanocytic_nevus",
            "BCC": "basal_cell_carcinoma",
            "AK": "actinic_keratosis",
            "BKL": "benign_keratinocytic_lesion",
            "DF": "dermatofibroma",
            "VASC": "vascular_lesions",
            "SCC": "squamous_cell_carcinoma",
            "UNK": "unknown",
        }
        self.class_codes = list(self.class_map.keys())
        self.class_names = [self.class_map[c] for c in self.class_codes]

    def _filter_data(self):
        """Filter dataset based on task requirements."""
        # Expand for multiple crops per image
        self._expand_crops()

        # Preload masks efficiently
        self._index_masks()

        # Add mask information if available
        self.data["masks"] = self.data["image"].map(self.mask_index).fillna(np.nan)

        # Task-specific filtering
        if self.task_config.binary_classification:
            # Filter to melanoma vs nevus only
            mask = (self.data["MEL"] == 1) | (self.data["NV"] == 1)
            self.data = self.data[mask].reset_index(drop=True)
            self.class_names = ["melanoma", "melanocytic_nevus"]

        if self.task_config.return_mask:
            # Remove samples without masks
            self.data = self.data.dropna(subset=["masks"]).reset_index(drop=True)


    def _index_masks(self):
        """Build a map of image_id → semicolon-separated mask paths."""
        masks_dir = self.root_dir / "masks"
        self.mask_index = {}

        if not masks_dir.exists():
            return

        # Scan all mask files once
        for mask_file in masks_dir.glob("*_mask.png"):
            stem = mask_file.stem  # e.g., "ISIC_12345_crop2_mask"
            # Remove the '_mask' suffix
            base_id = stem.replace("_mask", "")
            # Some may be cropped variants like ISIC_12345_crop2
            self.mask_index.setdefault(base_id.split("_crop")[0], [])
            # Use the exact base_id to differentiate crop-level masks
            self.mask_index[base_id.split("_crop")[0]].append(str(mask_file))

        # Merge variants into image_id level
        clean_index = {}
        for key, paths in self.mask_index.items():
            clean_index[key] = ";".join(sorted(paths)) if paths else None

        self.mask_index = clean_index

    def _expand_crops(self):
        """Efficiently expand dataset to include all cropped variants."""
        images_dir = self.root_dir / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        # Index all image files once
        all_images = list(images_dir.glob("*.jpg"))
        image_map = {}

        for img_path in all_images:
            stem = img_path.stem  # e.g. ISIC_12345_crop1 or ISIC_12345
            base_id = stem.split("_crop")[0]
            image_map.setdefault(base_id, []).append(str(img_path))

        # Prepare expanded rows
        expanded_rows = []
        for _, row in self.data.iterrows():
            image_id = row["image"]
            if image_id not in image_map:
                continue
            for img_path in image_map[image_id]:
                new_row = row.copy()
                new_row["image_path"] = img_path
                expanded_rows.append(new_row)

        self.data = pd.DataFrame(expanded_rows).reset_index(drop=True)

    def find_masks(self, image_path: str) -> str | None:
        """Find associated mask files for an image."""
        masks_dir = self.root_dir / "masks"
        if not masks_dir.exists():
            return None

        image_path = Path(image_path)
        image_id = image_path.stem.split("_crop")[0]

        # Look for crop-specific masks
        masks = sorted(masks_dir.glob(f"{image_id}_crop*_mask.png"))
        if not masks:
            # Look for plain mask
            plain_mask = masks_dir / f"{image_id}_mask.png"
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
        if len(mask_paths) == 0:
            return None
        masks = [
            np.array(Image.open(p).convert("L"), dtype=np.uint8)
            for p in mask_paths
        ]

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

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.

        Returns:
            dict: {"image": tensor, "mask": tensor (optional), "label":
                   tensor (optional)}
        """
        # Load image
        image = self._load_image(idx)
        mask = None

        # Load mask if needed
        if self.task_config.return_mask:
            mask = self._load_mask(idx)
            if mask is None:
                raise ValueError(
                    f"Mask required but not found for idx {idx}"
                )

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
            dtype = (
                torch.long
                if self.task_config.label_type == "binary"
                else torch.float
            )
            sample.label = torch.tensor(label, dtype=dtype)

        if self.task_config.return_mask:
            sample.mask = mask

        return sample.to_dict()

    @property
    def labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return np.array([self._get_label(i) for i in range(len(self))])

if __name__ == "__main__":
    # Simple test
    dataset = ISIC2019Dataset(
        root_dir="data/ISIC2019",
        task="segmentation",
    )
    print(f"Dataset size: {len(dataset)}")
    random_idx = np.random.randint(len(dataset))
    sample = dataset[random_idx]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    if "label" in sample:
        print(f"Label: {sample['label']}")
    if "mask" in sample:
        print(f"Mask shape: {sample['mask'].shape}")
