import os
import zipfile
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import gdown
import albumentations as A


class BaseDataModule(pl.LightningDataModule):
    """
    Base class for PyTorch Lightning DataModules that provides:

    - Train/val/test splitting (with stratification)
    - Saving and loading of splits
    - Automatic dataset download from Google Drive
    - Flexible Albumentations transform construction
    - Optional min-max normalization transforms
    """

    def __init__(
        self,
        train_val_test_split: tuple[int, int, int] | tuple[float, float, float],
        data_dir: str,
        allowed_labels: Optional[list[int | str]] = None,
        google_drive_id: Optional[str] = None,
        transforms: Optional[dict[str, list[dict[str, Any]]]] = None,
        image_size: Optional[int] = None,
        global_max: Optional[float | list[float]] = None,
        global_min: Optional[float | list[float]] = None,
        range_mode: str = "0_1",
        **kwargs,
    ):
        super().__init__()

        # --- storage ---
        self.train_val_test_split = train_val_test_split
        self.data_dir = Path(data_dir)
        self.allowed_labels = allowed_labels
        self.google_drive_id = google_drive_id

        # --- normalization params ---
        self.global_max = global_max
        self.global_min = global_min
        self.range_mode = range_mode
        self.image_size = image_size

        # --- transforms ---
        self.transforms_cfg = transforms or {}
        self.transforms_train: Optional[A.Compose] = None
        self.transforms_val: Optional[A.Compose] = None
        self.transforms_test: Optional[A.Compose] = None

        # --- split indices ---
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None
        self.test_indices: Optional[np.ndarray] = None

        if transforms:
            self._build_transforms()

    # =========================================================================
    # ABSTRACT METHODS TO BE IMPLEMENTED BY SUBCLASSES
    # =========================================================================
    def get_dataset_indices_and_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, labels) for full dataset."""
        raise NotImplementedError

    def get_labels_map(self) -> dict[str, int]:
        """Map string labels to integers."""
        raise NotImplementedError

    # =========================================================================
    # TRANSFORMS LOGIC
    # =========================================================================
    def _build_transforms(self):
        """Build albumentations transforms for train/val/test splits."""
        def build_transform_list(transforms: list[dict]):
            out = []
            for transform in transforms:
                cls_name = transform["class_path"]
                init_args = transform.get("init_args", {})
                if hasattr(A, cls_name):
                    transform_cls = getattr(A, cls_name)
                    out.append(transform_cls(**init_args))
                else:
                    raise ValueError(f"Albumentations has no transform '{cls_name}'")
            return A.Compose(out)

        if "train" in self.transforms_cfg:
            self.transforms_train = build_transform_list(self.transforms_cfg["train"])
        if "val" in self.transforms_cfg:
            self.transforms_val = build_transform_list(self.transforms_cfg["val"])
        if "test" in self.transforms_cfg:
            self.transforms_test = build_transform_list(self.transforms_cfg["test"])

        # Add normalization if provided
        if self.global_max is not None and self.global_min is not None:
            from src.transforms import NormalizeByMinMax

            if isinstance(self.global_max, list) and isinstance(self.global_min, list):
                norm_transform = NormalizeByMinMax(
                    mins=self.global_min,
                    maxs=self.global_max,
                    range_mode=self.range_mode,
                    clip=True,
                )
            else:
                norm_transform = NormalizeByMinMax(
                    mins=[self.global_min] * 16,
                    maxs=[self.global_max] * 16,
                    range_mode=self.range_mode,
                    clip=True,
                )

            for t in ["train", "val", "test"]:
                comp = getattr(self, f"transforms_{t}")
                if comp is not None:
                    comp.transforms.append(norm_transform)
                else:
                    setattr(self, f"transforms_{t}", A.Compose([norm_transform]))

    # =========================================================================
    # SPLIT MANAGEMENT LOGIC
    # =========================================================================
    def get_split_dir(self) -> Path:
        return self.data_dir.parent / "splits"

    def _filter_and_remap_indices(
        self,
        dataset_indices: np.ndarray,
        dataset_labels: np.ndarray,
        allowed_labels: Optional[list[int | str]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter indices by allowed labels (if specified) and remap them."""
        if allowed_labels is None:
            return dataset_indices, dataset_labels

        if isinstance(allowed_labels[0], str):
            labels_map = self.get_labels_map()
            allowed_labels = [labels_map[label] for label in allowed_labels]

        mask = np.isin(dataset_labels, allowed_labels)
        indices = dataset_indices[mask]
        labels = dataset_labels[mask]

        if len(indices) == 0:
            raise ValueError(f"No samples found for allowed_labels={allowed_labels}")

        remap_dict = {old: new for new, old in enumerate(sorted(allowed_labels))}
        labels = np.array([remap_dict[label] for label in labels])
        return indices, labels

    def setup_splits(self, seed: int = 42) -> None:
        """Generate stratified train/val/test splits."""
        indices, labels = self.get_dataset_indices_and_labels()
        indices, labels = self._filter_and_remap_indices(indices, labels, self.allowed_labels)

        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[counts >= 3]
        if len(valid_labels) < len(unique_labels):
            mask = np.isin(labels, valid_labels)
            indices, labels = indices[mask], labels[mask]

        # --- handle absolute sizes or ratios ---
        tvt = self.train_val_test_split
        if all(isinstance(x, int) for x in tvt):
            train_size, val_size, test_size = tvt
            total = train_size + val_size + test_size
            if total != len(indices):
                raise ValueError(f"Split sum ({total}) != dataset length ({len(indices)})")
            train_idx, temp_idx, y_train, y_temp = train_test_split(
                indices, labels, train_size=train_size, stratify=labels, random_state=seed
            )
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx, y_temp, train_size=val_size, stratify=y_temp, random_state=seed
            )
        elif all(isinstance(x, float) for x in tvt) and abs(sum(tvt) - 1.0) < 1e-6:
            train_ratio, val_ratio, test_ratio = tvt
            train_idx, temp_idx, y_train, y_temp = train_test_split(
                indices, labels, train_size=train_ratio, stratify=labels, random_state=seed
            )
            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx, y_temp, train_size=val_size, stratify=y_temp, random_state=seed
            )
        else:
            raise ValueError("train_val_test_split must be integers summing to N, or floats summing to 1")

        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx

    def save_splits_to_disk(self, split_dir: Optional[Path] = None):
        split_dir = split_dir or self.get_split_dir()
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(split_dir / "train.txt", self.train_indices, fmt="%d")
        np.savetxt(split_dir / "val.txt", self.val_indices, fmt="%d")
        np.savetxt(split_dir / "test.txt", self.test_indices, fmt="%d")
        print(f"Saved splits to {split_dir}")

    def load_splits_from_disk(self, split_dir: Optional[Path] = None) -> bool:
        split_dir = split_dir or self.get_split_dir()
        files = [split_dir / n for n in ["train.txt", "val.txt", "test.txt"]]
        if all(f.exists() for f in files):
            self.train_indices = np.loadtxt(files[0], dtype=int)
            self.val_indices = np.loadtxt(files[1], dtype=int)
            self.test_indices = np.loadtxt(files[2], dtype=int)
            print(f"Loaded splits from {split_dir}")
            return True
        return False

    def ensure_splits_exist(self):
        print("Generating new data splits...")
        self.setup_splits()
        self.save_splits_to_disk()

    # =========================================================================
    # GOOGLE DRIVE DOWNLOAD LOGIC
    # =========================================================================
    def download_and_extract_if_needed(self):
        """Automatically download and extract dataset zip from Google Drive."""
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            print(f"Dataset already found at {self.data_dir}")
            return

        if not self.google_drive_id:
            raise ValueError("google_drive_id must be provided to download dataset")

        zip_path = Path(f"{self.data_dir.name}.zip")
        print(f"Downloading dataset from Google Drive to {zip_path}...")
        gdown.download(id=self.google_drive_id, output=str(zip_path), quiet=False)

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.data_dir.parent)
        zip_path.unlink(missing_ok=True)
        print("Extraction complete.")

    # =========================================================================
    # COMBINED PREPARATION PIPELINE
    # =========================================================================
    def prepare_data(self):
        """Download data if missing and ensure train/val/test splits exist."""
        self.download_and_extract_if_needed()
        self.ensure_splits_exist()
