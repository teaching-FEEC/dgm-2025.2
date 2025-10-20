import os
from pathlib import Path

from git import Optional
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torchvision.transforms import transforms as T
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.io import savemat

import gdown
import zipfile
from PIL import Image

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(
        Path(__file__).parent.parent.parent, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
    )

from src.samplers.infinite import InfiniteSamplerWrapper
from src.utils.transform import smallest_maxsize_and_centercrop
from src.samplers.balanced_batch_sampler import BalancedBatchSampler
from src.data_modules.datasets.hsi_dermoscopy_dataset import HSIDermoscopyDataset, HSIDermoscopyTask

class HSIDermoscopyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task: str | HSIDermoscopyTask,
        train_val_test_split: tuple[int, int, int] | tuple[float, float, float],
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = False,
        data_dir: str = "data/hsi_dermoscopy",
        image_size: int = 224,
        transforms: Optional[dict] = None,
        allowed_labels: Optional[list[int | str]] = None,
        google_drive_id: Optional[str] = None,
        balanced_sampling: bool = False,
        synthetic_data_dir: Optional[str] = None,
        global_max: float | list[float] = None,
        global_min: float | list[float] = None,
        infinite_train: bool = False,
        sample_size: Optional[int] = None,
        range_mode: Optional[str] = None,

    ):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(task, str):
            self.hparams.task = HSIDermoscopyTask[task]
        print(self.hparams)

        self.transforms_train = None
        self.transforms_test = None
        self.transforms_val = None

        if transforms is not None:
            if "train" in transforms:
                self.transforms_train = A.Compose(self.get_transforms(transforms, "train"))
            if "val" in transforms:
                self.transforms_val = A.Compose(self.get_transforms(transforms, "val"))
            if "test" in transforms:
                self.transforms_test = A.Compose(self.get_transforms(transforms, "test"))

        if range_mode in self.hparams: 
            self.range_mode = range_mode
        elif self.hparams.task != HSIDermoscopyTask.GENERATION: 
            self.range_mode = range_mode
        else: 
            self.range_mode = '0_1'

        # if global_max and global_min are provided, add NormalizeByMinMax to transforms
        from src.transforms import NormalizeByMinMax

        if global_max is not None and global_min is not None:
            if isinstance(global_max, list) and isinstance(global_min, list):
                transform = NormalizeByMinMax(
                    mins=global_min,
                    maxs=global_max,
                    range_mode=self.range_mode,
                    clip=True,
                )
            elif isinstance(global_max, (int, float)) and isinstance(global_min, (int, float)):
                transform = NormalizeByMinMax(
                    mins=[global_min] * 16,
                    maxs=[global_max] * 16,
                    range_mode = self.range_mode,
                    clip=True,
                )
            else:
                raise ValueError("global_max and global_min must be both lists or both scalars")
            if self.transforms_train is not None:
                self.transforms_train.transforms.append(transform)
            else:
                self.transforms_train = A.Compose([transform])
            if self.transforms_val is not None:
                self.transforms_val.transforms.append(transform)
            else:
                self.transforms_val = A.Compose([transform])
            if self.transforms_test is not None:
                self.transforms_test.transforms.append(transform)
            else:
                self.transforms_test = A.Compose([transform])

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

        self.train_indices: np.ndarray = None
        self.val_indices: np.ndarray = None
        self.test_indices: np.ndarray = None

        self.global_max = global_max

        self.global_min = global_min

    def get_transforms(self, transforms: dict, stage: str) -> list[A.BasicTransform]:
        transforms_list = []
        for transform in transforms[stage]:
            class_name = transform["class_path"]
            init_args = transform.get("init_args", {})
            if hasattr(A, class_name):
                transform_cls = getattr(A, class_name)
                transforms_list.append(transform_cls(**init_args))
            else:
                raise ValueError(f"Albumentations has no transform named {class_name}")
        return transforms_list

    def prepare_data(self):
        if not os.path.exists(self.hparams.data_dir) or not os.listdir(self.hparams.data_dir):
            # check if a .zip file with the data_dir name exists in the parent directory
            downloaded = False
            filename = f"{Path(self.hparams.data_dir).name}.zip"
            if not os.path.exists(filename):
                print(f"Downloading HSI Dermoscopy dataset to {self.hparams.data_dir}...")

                if self.hparams.google_drive_id is None or self.hparams.google_drive_id == "":
                    raise ValueError("google_drive_id must be provided to download the dataset.")

                filename = gdown.download(id=self.hparams.google_drive_id, quiet=False)
                downloaded = True
            else:
                print(f"Found existing zip file {filename}, skipping download.")

            os.makedirs(os.path.dirname(self.hparams.data_dir), exist_ok=True)

            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(Path(self.hparams.data_dir).parent)
                if downloaded:
                    os.remove(filename)

        self.setup_splits()

    # Add helper method to filter and remap labels
    def _filter_and_remap_indices(self, dataset_indices, dataset_labels, allowed_labels):
        if allowed_labels is not None:
            # Normalize allowed_labels into integer form
            if isinstance(allowed_labels[0], str):
                # Map strings to dataset integers
                string_to_int = {
                    name: idx
                    for name, idx in HSIDermoscopyDataset(
                        task=self.hparams.task, data_dir=self.hparams.data_dir
                    ).labels_map.items()
                }
                allowed_labels = [string_to_int[label] for label in allowed_labels]

            mask = np.isin(dataset_labels, allowed_labels)
            filtered_indices = dataset_indices[mask]
            filtered_labels = dataset_labels[mask]

            if len(filtered_indices) == 0:
                raise ValueError(f"No samples found for allowed_labels={allowed_labels}")

            # Remap labels to contiguous [0..N-1]
            allowed_labels_sorted = sorted(allowed_labels)
            remap_dict = {old: new for new, old in enumerate(allowed_labels_sorted)}

            filtered_labels = np.array([remap_dict[label] for label in filtered_labels])
            return filtered_indices, filtered_labels
        return dataset_indices, dataset_labels

    def setup_splits(self):
        seed = 42
        full_dataset = HSIDermoscopyDataset(task=self.hparams.task, data_dir=self.hparams.data_dir)

        indices = np.arange(len(full_dataset))
        labels = full_dataset.labels_df["label"].map(full_dataset.labels_map).to_numpy()

        # Apply filtering
        # First filter by allowed labels
        indices, labels = self._filter_and_remap_indices(indices, labels, self.hparams.allowed_labels)

        # Remove labels that have less than 3 samples since they can't be stratified
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[label_counts >= 3]

        if len(valid_labels) < len(unique_labels):
            mask = np.isin(labels, valid_labels)
            indices = indices[mask]
            labels = labels[mask]

            # Remap labels to be contiguous again
            label_map = {old: new for new, old in enumerate(sorted(valid_labels))}
            labels = np.array([label_map[label] for label in labels])
            print(f"Warning: Filtered out labels with less than 3 samples. Remaining labels: {valid_labels.tolist()}")

        # Integer-based splits
        if all(isinstance(x, int) for x in self.hparams.train_val_test_split):
            train_size, val_size, test_size = self.hparams.train_val_test_split
            if train_size + val_size + test_size != len(full_dataset):
                raise ValueError(
                    "When using absolute numbers for train/val/test split, the sum must equal the dataset length."
                )

            train_idx, temp_idx, train_y, temp_y = train_test_split(
                indices,
                labels,
                train_size=train_size,
                random_state=seed,
                stratify=labels,
            )

            val_idx, test_idx, _, _ = train_test_split(
                temp_idx,
                temp_y,
                train_size=val_size,
                random_state=seed,
                stratify=temp_y,
            )

        # Ratio-based splits
        elif (
            all(isinstance(x, float) for x in self.hparams.train_val_test_split)
            and abs(sum(self.hparams.train_val_test_split) - 1.0) < 1e-6
        ):
            train_ratio, val_ratio, test_ratio = self.hparams.train_val_test_split

            train_idx, temp_idx, train_y, temp_y = train_test_split(
                indices,
                labels,
                train_size=train_ratio,
                random_state=seed,
                stratify=labels,
            )

            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx,
                temp_y,
                train_size=val_size,
                random_state=seed,
                stratify=temp_y,
            )
        else:
            raise ValueError("train_val_test_split must be either all integers or floats summing to 1.")

        self.train_indices, self.val_indices, self.test_indices = (
            train_idx,
            val_idx,
            test_idx,
        )

    def save_splits_to_disk(self, split_dir: Path):
        """Saves the generated splits to a specified directory."""
        if self.train_indices is None or self.val_indices is None or self.test_indices is None:
            raise RuntimeError("Splits have not been generated. Call setup_splits() first.")

        os.makedirs(split_dir, exist_ok=True)
        np.savetxt(split_dir / "train.txt", self.train_indices, fmt="%d")
        np.savetxt(split_dir / "val.txt", self.val_indices, fmt="%d")
        np.savetxt(split_dir / "test.txt", self.test_indices, fmt="%d")
        print(f"Saved data splits to {split_dir}")

    def setup(self, stage: str = None):
        if self.train_indices is None or self.val_indices is None or self.test_indices is None:
            split_dir = Path(self.hparams.data_dir).parent / "splits"
            if (
                (split_dir / "train.txt").exists()
                and (split_dir / "val.txt").exists()
                and (split_dir / "test.txt").exists()
            ):
                self.train_indices = np.loadtxt(split_dir / "train.txt", dtype=int)
                self.val_indices = np.loadtxt(split_dir / "val.txt", dtype=int)
                self.test_indices = np.loadtxt(split_dir / "test.txt", dtype=int)
            else:
                # last resort, regenerate
                self.setup_splits()

        # Use the indices from setup_splits to create the splits
        if stage in ["fit", "validate"] or stage is None and (self.data_train is None or self.data_val is None):
            self.data_train = HSIDermoscopyDataset(
                task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms_train,
            )
            self.data_train = torch.utils.data.Subset(self.data_train, self.train_indices)

            # Add synthetic data to training set if provided
            if self.hparams.synthetic_data_dir is not None:
                synthetic_dataset = HSIDermoscopyDataset(
                    task=self.hparams.task,
                    data_dir=self.hparams.synthetic_data_dir,
                    transform=self.transforms_train,
                )

                # Use all samples from synthetic dataset
                synthetic_indices = np.arange(len(synthetic_dataset))

                # Apply label filtering if specified
                if self.hparams.allowed_labels is not None:
                    synthetic_labels = synthetic_dataset.labels_df["label"].map(synthetic_dataset.labels_map).to_numpy()
                    synthetic_indices, _ = self._filter_and_remap_indices(
                        synthetic_indices, synthetic_labels, self.hparams.allowed_labels
                    )

                synthetic_subset = torch.utils.data.Subset(synthetic_dataset, synthetic_indices)

                # Concatenate real and synthetic training data
                self.data_train = torch.utils.data.ConcatDataset([self.data_train, synthetic_subset])

                print(f"Added {len(synthetic_subset)} synthetic samples to training set")

            self.data_val = HSIDermoscopyDataset(
                task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms_val,
            )
            self.data_val = torch.utils.data.Subset(self.data_val, self.val_indices)

        # Assign test dataset
        if stage in ["test", "predict"] or stage is None and self.data_test is None:
            self.data_test = HSIDermoscopyDataset(
                task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms_test,
            )
            self.data_test = torch.utils.data.Subset(self.data_test, self.test_indices)

    def train_dataloader(self):
        sampler = None

        if (
            self.hparams.task
            in [
                HSIDermoscopyTask.CLASSIFICATION_MELANOMA_VS_OTHERS,
                HSIDermoscopyTask.CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_NEVI,
            ]
            and self.hparams.balanced_sampling
        ):
            # Handle both Subset and ConcatDataset
            if isinstance(self.data_train, torch.utils.data.ConcatDataset):
                # Extract labels from concatenated datasets
                labels = []
                for dataset in self.data_train.datasets:
                    if isinstance(dataset, torch.utils.data.Subset):
                        labels.extend([dataset.dataset.labels[i] for i in dataset.indices])
                    else:
                        labels.extend(dataset.labels)
                labels = np.array(labels)
            else:
                # Original Subset case
                labels = np.array([self.data_train.dataset.labels[i] for i in self.data_train.indices])

            sampler = BalancedBatchSampler(labels, batch_size=self.hparams.batch_size)
            return DataLoader(
                self.data_train,
                batch_sampler=sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )

        if self.hparams.infinite_train:
            if self.hparams.sample_size and self.hparams.sample_size > 0 and \
                self.hparams.sample_size < len(self.data_train):
                sampler = InfiniteSamplerWrapper(SubsetRandomSampler(torch.arange(self.hparams.sample_size)))
            else:
                sampler = InfiniteSamplerWrapper(self.data_train)
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True if sampler is None else False,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        if self.hparams.task == HSIDermoscopyTask.GENERATION:
            dummy_dataset = torch.utils.data.TensorDataset(torch.zeros(1, 1))
            return DataLoader(
                dummy_dataset,
                batch_size=1,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
        else:
            return self.test_dataloader()

    def all_dataloader(self):
        full_dataset = HSIDermoscopyDataset(
            task=self.hparams.task, data_dir=self.hparams.data_dir, transform=self.transforms_test
        )

        # use _filter_and_remap_indices to filter the full dataset
        indices = np.arange(len(full_dataset))
        labels = full_dataset.labels_df["label"].map(full_dataset.labels_map).to_numpy()
        indices, _ = self._filter_and_remap_indices(indices, labels, self.hparams.allowed_labels)
        filtered_dataset = torch.utils.data.Subset(full_dataset, indices)

        return DataLoader(
            filtered_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage=None):
        # Called on every process after trainer is done
        pass

    def global_normalization(self, cube: np.ndarray, clip_interval: tuple[int, int] = (0, 1)) -> np.ndarray:
        if self.global_max is None or self.global_min is None:
            raise ValueError("Global max and min values must be set for global normalization.")

        if isinstance(self.global_min, int) and isinstance(self.global_max, int):
            # single value provided, use for all bands
            cube = (cube - self.global_min) / (self.global_max - self.global_min)
            if clip_interval == (-1, 1):
                cube = cube * 2 - 1
        elif isinstance(self.global_min, list) and isinstance(self.global_max, list):
            if len(self.global_min) != cube.shape[2] or len(self.global_max) != cube.shape[2]:
                raise ValueError("Length of global_min and global_max lists must match number of bands in cube")
            # per-band normalization
            for b in range(cube.shape[2]):
                cube[:, :, b] = (cube[:, :, b] - self.global_min[b]) / (self.global_max[b] - self.global_min[b])
                if clip_interval == (-1, 1):
                    cube[:, :, b] = cube[:, :, b] * 2 - 1
        cube = np.clip(cube, clip_interval[0], clip_interval[1])
        cube = cube.astype("float32")
        return cube

    def _export_single_crop(
        self,
        cube: np.ndarray,
        mask: np.ndarray,
        output_root: Path,
        structure: str,
        split_name: str,
        label_name: str,
        counter: dict,
        extension: str,
        mode: str,
        bands: Optional[list[int]],
        bbox_scale: float,
        image_size: Optional[int],
        global_normalization: bool,
        original_path: str,  # Added parameter
        crop_idx: int,  # Added parameter
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Export a single cropped image based on one mask."""
        # Convert to export mode
        if mode == "rgb":
            rgb_data = self._convert_to_rgb(cube, bands)
        else:  # hyper
            rgb_data = cube if bands is None else cube[:, :, bands]

        # Crop
        cropped_data = self._crop_with_bbox(rgb_data, mask, bbox_scale)
        if cropped_data is None:
            return None, None

        counter["count"] += 1

        # Determine output paths
        img_path, mask_path = self._get_export_paths(
            output_root,
            structure,
            split_name,
            label_name,
            counter["count"],
            counter,
            extension,
            original_path=original_path,  # Pass original path
            crop_idx=crop_idx,  # Pass crop index
        )

        if image_size is not None:
            cropped_data = smallest_maxsize_and_centercrop(cropped_data, image_size)

        # Save image
        if mode == "rgb":
            cropped_data = cropped_data.astype("uint8")
            Image.fromarray(cropped_data).save(img_path)
        else:
            if global_normalization:
                cropped_data = self.global_normalization(cropped_data, clip_interval=(-1, 1))
            savemat(img_path, {"cube": cropped_data})

        return img_path, mask_path

    def export_dataset(
        self,
        output_dir: str,
        splits: Optional[list[str]] = None,
        mode: str = "rgb",
        extension: Optional[str] = None,
        bands: Optional[list[int]] = None,
        crop_with_mask: bool = False,
        bbox_scale: float = 1.5,
        structure: str = "original",
        allowed_labels: Optional[list[int | str]] = None,
        image_size: Optional[int] = None,
        global_normalization: bool = False,
    ) -> None:
        """
        Export dataset with flexible options.

        Args:
            output_dir: Root directory for exported files
            splits: List of splits to export (["train", "val", "test"]).
                    If None, exports all splits.
            mode: "rgb" or "hyper" for export format
            extension: Custom file extension (default: .png for rgb, .mat for hyper)
            bands: Band indices to export/use for RGB
            crop_with_mask: Whether to crop images using mask bounding boxes
            bbox_scale: Scale factor for bounding box (default 1.5 = 50% padding)
            structure: Directory structure
            allowed_labels: List of labels to export (can be int or str)
            image_size: If specified, resize images to this size
            global_normalization: If True, use global min-max normalization for both RGB and hyperspectral data
        """
        from scipy.io import loadmat, savemat

        if splits is None:
            splits = ["train", "val", "test"]

        # Ensure setup has been called
        if self.train_indices is None:
            self.setup_splits()

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        if extension is None:
            extension = ".png" if mode == "rgb" else ".mat"
        elif not extension.startswith("."):
            extension = f".{extension}"

        # Get full dataset
        full_dataset = HSIDermoscopyDataset(task=self.hparams.task, data_dir=self.hparams.data_dir)

        # Convert allowed_labels to set of integers
        allowed_label_ints = None
        if allowed_labels is not None:
            allowed_label_ints = set()
            for label in allowed_labels:
                if isinstance(label, str):
                    if label in full_dataset.labels_map:
                        allowed_label_ints.add(full_dataset.labels_map[label])
                    else:
                        raise ValueError(
                            f"Label '{label}' not found in dataset. Available: {list(full_dataset.labels_map.keys())}"
                        )
                else:
                    allowed_label_ints.add(label)

        def get_label_name(idx: int) -> str:
            label_int = full_dataset.labels[idx]
            for name, val in full_dataset.labels_map.items():
                if val == label_int:
                    return name
            return "unknown"

        def should_export(idx: int) -> bool:
            if allowed_label_ints is None:
                return True
            return full_dataset.labels[idx] in allowed_label_ints

        # Export each split
        path_mapping = {}
        split_indices_map = {
            "train": self.train_indices,
            "val": self.val_indices,
            "test": self.test_indices,
        }

        total_exported = 0
        for split_name in splits:
            if split_name not in split_indices_map:
                print(f"Warning: Unknown split '{split_name}', skipping")
                continue

            indices = split_indices_map[split_name]
            counter = {"count": 0}

            # Filter indices by allowed labels
            if allowed_label_ints is not None:
                filtered_indices = [idx for idx in indices if should_export(idx)]
                if len(filtered_indices) == 0:
                    print(f"Warning: No samples in '{split_name}' match allowed_labels={allowed_labels}")
                    continue
            else:
                filtered_indices = indices

            for idx in tqdm(
                filtered_indices,
                desc=f"Exporting {split_name} split ({mode} mode)",
            ):
                if not should_export(idx):
                    continue

                row = full_dataset.labels_df.iloc[idx]
                label_name = get_label_name(idx)

                # Load hyperspectral cube
                cube = loadmat(row["file_path"]).popitem()[-1].astype("float32")

                # Get masks for this image
                mask_paths = full_dataset.get_masks_list(idx)

                if crop_with_mask and mask_paths:
                    # Export one cropped image per mask
                    for mask_idx, mask_path in enumerate(mask_paths):
                        mask = np.array(Image.open(mask_path).convert("L"))

                        img_path, _ = self._export_single_crop(
                            cube,
                            mask,
                            output_root,
                            structure,
                            split_name,
                            label_name,
                            counter,
                            extension,
                            mode,
                            bands,
                            bbox_scale,
                            image_size,
                            global_normalization,
                            original_path=str(row["file_path"]),  # Pass original path
                            crop_idx=mask_idx,  # Pass crop index
                        )

                        if img_path:
                            path_mapping[str(img_path)] = str(row["file_path"])
                            total_exported += 1
                else:
                    # Export without cropping (original behavior)
                    counter["count"] += 1

                    # Convert to export mode
                    if mode == "rgb":
                        rgb_data = self._convert_to_rgb(cube, bands)
                    else:
                        rgb_data = cube if bands is None else cube[:, :, bands]

                    if image_size is not None:
                        rgb_data = smallest_maxsize_and_centercrop(rgb_data, image_size)

                    # Determine output paths
                    img_path, mask_path = self._get_export_paths(
                        output_root,
                        structure,
                        split_name,
                        label_name,
                        idx,
                        counter,
                        extension,
                        original_path=str(row["file_path"]),  # Pass original path
                        crop_idx=None,  # No crop index for non-cropped
                    )

                    # Save image
                    if mode == "rgb":
                        Image.fromarray(rgb_data).save(img_path)
                    else:
                        if global_normalization:
                            rgb_data = self.global_normalization(rgb_data, clip_interval=(-1, 1))
                        savemat(img_path, {"cube": rgb_data})

                    path_mapping[str(img_path)] = str(row["file_path"])

                    # Save mask if applicable
                    if structure not in ["images_only"] and mask_paths and mask_path is not None:
                        # For non-cropped export, combine all masks
                        masks = [np.array(Image.open(mp).convert("L")) for mp in mask_paths]
                        combined_mask = masks[0].copy()
                        for mask in masks[1:]:
                            combined_mask = np.maximum(combined_mask, mask)

                        Image.fromarray(combined_mask).save(mask_path)
                        path_mapping[str(mask_path)] = ";".join(mask_paths)

                    total_exported += 1

        # Save path mapping
        mapping_file = output_root / "path_mapping.csv"
        pd.DataFrame.from_dict(path_mapping, orient="index", columns=["original_path"]).to_csv(mapping_file)

        print(f"\nExported {total_exported} samples to {output_root}")
        print(f"Structure: {structure}, Mode: {mode}, Cropped: {crop_with_mask}")
        if allowed_labels:
            print(f"Filtered to labels: {allowed_labels}")
        print(f"Saved path mapping to {mapping_file}")

    def _convert_to_rgb(self, cube: np.ndarray, bands: Optional[list[int]]) -> np.ndarray:
        """Convert hyperspectral cube to RGB image."""
        if bands is None:
            # No bands specified: mean across all bands
            band_data = np.mean(cube, axis=2, keepdims=True)
            rgb = np.repeat(band_data, 3, axis=2)
        elif len(bands) == 1:
            # Single band: replicate on 3 channels
            band_data = cube[:, :, bands[0:1]]
            rgb = np.repeat(band_data, 3, axis=2)
        elif len(bands) == 3:
            # Three bands: use for RGB
            rgb = cube[:, :, bands]
        else:
            # More than 3: take mean and replicate
            band_data = np.mean(cube[:, :, bands], axis=2, keepdims=True)
            rgb = np.repeat(band_data, 3, axis=2)

        # Normalize to 0-255 using global min-max normalization
        rgb = self.global_normalization(rgb, clip_interval=(0, 1))
        rgb = (rgb * 255).astype("uint8")
        return rgb

    def _crop_with_bbox(self, img: np.ndarray, mask: np.ndarray, bbox_scale: float) -> Optional[np.ndarray]:
        """Crop image using mask bounding box with scaling."""
        ys, xs = np.where(mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return None

        h, w = img.shape[:2]
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        bbox_h = y_max - y_min + 1
        bbox_w = x_max - x_min + 1
        cy = (y_min + y_max) / 2
        cx = (x_min + x_max) / 2

        new_h = bbox_h * bbox_scale
        new_w = bbox_w * bbox_scale

        y_min = max(0, int(round(cy - new_h / 2)))
        y_max = min(h - 1, int(round(cy + new_h / 2)))
        x_min = max(0, int(round(cx - new_w / 2)))
        x_max = min(w - 1, int(round(cx + new_w / 2)))

        return img[y_min : y_max + 1, x_min : x_max + 1]

    def _get_export_paths(
        self,
        output_root: Path,
        structure: str,
        split_name: str,
        label_name: str,
        idx: int,
        counter: dict,
        extension: str,
        original_path: Optional[str] = None,  # Added parameter
        crop_idx: Optional[int] = None,  # Added parameter
    ) -> tuple[Path, Optional[Path]]:
        """Determine output paths based on directory structure."""

        if structure == "original":
            if original_path is None:
                raise ValueError("original_path must be provided for structure='original'")

            # Get the relative path from the data directory
            original_path_obj = Path(original_path)
            data_dir = Path(self.hparams.data_dir)

            try:
                # Get relative path from data_dir
                rel_path = original_path_obj.relative_to(data_dir)
            except ValueError:
                # Fallback: preserve at least the last few directory levels
                rel_path = Path(*original_path_obj.parts[-3:])

            # Create filename
            base_name = rel_path.stem
            if crop_idx is not None:
                filename = f"{base_name}_crop{crop_idx:02d}{extension}"
            else:
                filename = f"{base_name}{extension}"

            # Preserve directory structure
            img_path = output_root / rel_path.parent / filename
            img_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path = img_path.parent / filename.replace(extension, "_mask.png")

        elif structure == "imagenet":
            # ImageNet structure: split/class/images
            counter["count"] += 1
            filename = f"{label_name}_{counter['count']:05d}{extension}"
            img_path = output_root / split_name / label_name / filename
            img_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path = img_path.parent / filename.replace(extension, "_mask.png")

        elif structure == "flat":
            # Flat with label in filename
            counter["count"] += 1
            filename = f"{split_name}_{label_name}_{counter['count']:05d}{extension}"
            img_path = output_root / filename
            mask_path = output_root / filename.replace(extension, "_mask.png")

        elif structure == "flat_with_masks":
            # Separate images/ and masks/ directories
            images_dir = output_root / "images"
            masks_dir = output_root / "masks"
            images_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)
            counter["count"] += 1
            filename = f"{split_name}_{label_name}_{counter['count']:05d}{extension}"
            img_path = images_dir / filename
            mask_path = masks_dir / filename.replace(extension, "_mask.png")

        elif structure == "images_only":
            # Only images, no subdirectories
            counter["count"] += 1
            filename = f"{split_name}_{label_name}_{counter['count']:05d}{extension}"
            img_path = output_root / filename
            mask_path = None

        else:
            raise ValueError(f"Unknown structure: {structure}")

        return img_path, mask_path


if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(
        Path(__file__).parent.parent.parent, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
    )

    # Example usage
    image_size = 256
    data_module = HSIDermoscopyDataModule(
        task="CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_NEVI",
        train_val_test_split=(0.7, 0.15, 0.15),
        batch_size=8,
        data_dir="data/hsi_dermoscopy",
        image_size=image_size,
        transforms={
            "train": [
                {"class_path": "HorizontalFlip", "init_args": {"p": 0.5}},
                {"class_path": "VerticalFlip", "init_args": {"p": 0.5}},
                {"class_path": "SmallestMaxSize", "init_args": {"max_size": image_size}},
                {"class_path": "CenterCrop", "init_args": {"height": image_size, "width": image_size}},
                {"class_path": "ToTensorV2", "init_args": {}},
            ],
            "val": [
                {"class_path": "SmallestMaxSize", "init_args": {"max_size": image_size}},
                {"class_path": "CenterCrop", "init_args": {"height": image_size, "width": image_size}},
                {"class_path": "ToTensorV2", "init_args": {}},
            ],
            "test": [
                {"class_path": "SmallestMaxSize", "init_args": {"max_size": image_size}},
                {"class_path": "CenterCrop", "init_args": {"height": image_size, "width": image_size}},
                {"class_path": "ToTensorV2", "init_args": {}},
            ],
        },
        google_drive_id="1557yQpqO3baKVSqstuKLjr31NuC2eqcO",
        # synthetic_data_dir="data/hsi_dermoscopy_cropped_synth",
    )
    data_module.prepare_data()
    data_module.setup()

    print(f"Train samples: {len(data_module.data_train)}")
    print(f"Val samples: {len(data_module.data_val)}")
    print(f"Test samples: {len(data_module.data_test)}")

    # train_dataloader = data_module.train_dataloader()
    # train_dataset = train_dataloader.dataset

    # plot_dataset_mosaic(train_dataset, m=0, n=50, save_path="melanoma_train_mosaic.png", nrow=10)

    # Export dataset example
    data_module.export_dataset(
        output_dir="export/hsi_dermoscopy_croppedv2_256",
        # splits=["train", "val", "test"],
        crop_with_mask=True,
        bbox_scale=2,
        structure="original",
        mode="hyper",
        image_size=image_size,
        # allowed_labels=["melanoma"],
        global_normalization=False,
    )
