import os
from pathlib import Path

from git import Optional
import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.transforms import transforms as T
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from sklearn.model_selection import train_test_split

from src.data_modules.datasets.hsi_dermoscopy_dataset import HSIDermoscopyDataset, HSIDermoscopyTask
import gdown
import zipfile

from src.samplers.balanced_batch_sampler import BalancedBatchSampler

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
    ):
        super().__init__()
        self.save_hyperparameters()

        if isinstance(task, str):
            self.hparams.task = HSIDermoscopyTask[task]

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

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

        self.train_indices: np.ndarray = None
        self.val_indices: np.ndarray = None
        self.test_indices: np.ndarray = None

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

            with zipfile.ZipFile(filename, 'r') as zip_ref:
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
                    name: idx for name, idx in HSIDermoscopyDataset(
                        task=self.hparams.task,
                        data_dir=self.hparams.data_dir
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
        full_dataset = HSIDermoscopyDataset(
            task=self.hparams.task, data_dir=self.hparams.data_dir
        )

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
                    "When using absolute numbers for train/val/test split, "
                    "the sum must equal the dataset length."
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
        elif all(isinstance(x, float) for x in self.hparams.train_val_test_split) and \
                abs(sum(self.hparams.train_val_test_split) - 1.0) < 1e-6:

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
            raise ValueError(
                "train_val_test_split must be either all integers or floats summing to 1."
            )

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
            if (split_dir / "train.txt").exists() and (split_dir / "val.txt").exists() and \
            (split_dir / "test.txt").exists():
                self.train_indices = np.loadtxt(split_dir / "train.txt", dtype=int)
                self.val_indices = np.loadtxt(split_dir / "val.txt", dtype=int)
                self.test_indices = np.loadtxt(split_dir / "test.txt", dtype=int)
            else:
                # last resort, regenerate
                self.setup_splits()

        # Use the indices from setup_splits to create the splits
        if stage in ['fit', 'validate'] or stage is None and (self.data_train is None or self.data_val is None):
            self.data_train = HSIDermoscopyDataset(
                task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms_train,
            )
            self.data_train = torch.utils.data.Subset(self.data_train, self.train_indices)

            self.data_val = HSIDermoscopyDataset(
                task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms_val,
            )
            self.data_val = torch.utils.data.Subset(self.data_val, self.val_indices)

        # Assign test dataset
        if stage == 'test' or stage is None and self.data_test is None:
            self.data_test = HSIDermoscopyDataset(
                task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms_test,
            )
            self.data_test = torch.utils.data.Subset(self.data_test, self.test_indices)

    def train_dataloader(self):
        if self.hparams.task in [
            HSIDermoscopyTask.CLASSIFICATION_MELANOMA_VS_OTHERS,
            HSIDermoscopyTask.CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_NEVI,
        ] and self.hparams.balanced_sampling:
            labels = np.array(
                [self.data_train.dataset.labels[i] for i in self.data_train.indices]
            )
            sampler = BalancedBatchSampler(labels, batch_size=self.hparams.batch_size)
            return DataLoader(
                self.data_train,
                batch_sampler=sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
        else:
            return DataLoader(
                self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
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

    def all_dataloader(self):
        full_dataset = HSIDermoscopyDataset(task=self.hparams.task, data_dir=self.hparams.data_dir)

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
