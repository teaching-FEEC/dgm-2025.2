from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.samplers.infinite import InfiniteSamplerWrapper

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(
        Path(__file__).parent.parent.parent, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
    )


from src.data_modules.base import BaseDataModule
from src.data_modules.datasets.milk10k_dataset import (
    MILK10kDataset,
    MILK10kTask,
)


class MILK10kDataModule(BaseDataModule):
    def __init__(
        self,
        task: str | MILK10kTask,
        train_val_test_split: tuple[int, int, int] | tuple[float, float, float],
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = False,
        data_dir: str = "data/MILK10k",
        transforms: Optional[dict] = None,
        allowed_labels: Optional[list[int | str]] = None,
        image_size: int = 224,
        global_max: Optional[float | list[float]] = None,
        global_min: Optional[float | list[float]] = None,
        google_drive_id: Optional[str] = "183BASdfQ55TgtRFSdfQ6k3qaSeeNOMp1",
        infinite_train: bool = False,
        sample_size: Optional[int] = None,
        range_mode: str = '-1_1',
        normalize_mask_tanh: bool = False,
        **kwargs,
    ):
        super().__init__(
            train_val_test_split=train_val_test_split,
            data_dir=data_dir,
            allowed_labels=allowed_labels,
            google_drive_id=google_drive_id,
            transforms=transforms,
            image_size=image_size,
            global_max=global_max,
            global_min=global_min,
            range_mode = range_mode,
            normalize_mask_tanh=normalize_mask_tanh,
        )
        self.save_hyperparameters(
            {
                "data_dir": data_dir,
                "task": task,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "allowed_labels": allowed_labels,
                "infinite_train": infinite_train,
                "sample_size": sample_size,
            }
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if isinstance(task, str):
            self.hparams.task = MILK10kTask[task.upper()]

        # Initialize dataset attributes
        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.full_dataset = None

    def prepare_data(self):
        # use superclass method first
        self.download_and_extract_if_needed()
        self.full_dataset = MILK10kDataset(
            root_dir=self.hparams.data_dir,
            task=self.hparams.task,
        )
        self.ensure_splits_exist()

    def get_split_dir(self) -> Path:
        return Path(self.hparams.data_dir).parent / "splits" / "milk10k"

    def get_dataset_indices_and_labels(self) -> tuple[np.ndarray, np.ndarray]:

        indices = np.arange(len(self.full_dataset))

        if self.hparams.task == MILK10kTask.MULTILABEL or \
            self.hparams.task == MILK10kTask.SEGMENTATION or \
            self.hparams.task == MILK10kTask.GENERATION:
            # For multilabel, use first positive class index for stratification
            label_matrix = self.full_dataset.data[self.full_dataset.class_codes].values
            labels = np.argmax(label_matrix, axis=1)
        elif self.hparams.task == MILK10kTask.MELANOMA_VS_NEVUS or \
            self.hparams.task == MILK10kTask.GENERATION_MELANOMA_VS_NEVUS:
            # For binary: melanoma=0, nevus=1
            labels = self.full_dataset.data["NV"].to_numpy()
        else:
            raise ValueError(f"Unknown task: {self.hparams.task}")

        return indices, labels

    def get_labels_map(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.full_dataset.class_names)}

    def setup(self, stage: str = None):
        if stage in ["fit", "validate"] or stage is None and (
            self.data_train is None or self.data_val is None
        ):
            self.data_train = MILK10kDataset(
                root_dir=self.hparams.data_dir,
                task=self.hparams.task,
                transform=self.transforms_train,
            )
            self.data_train = torch.utils.data.Subset(
                self.data_train, self.train_indices
            )

            if len(self.val_indices) > 0:
                self.data_val = MILK10kDataset(
                    root_dir=self.hparams.data_dir,
                    task=self.hparams.task,
                    transform=self.transforms_val,
                )
                self.data_val = torch.utils.data.Subset(
                    self.data_val, self.val_indices
                )
        elif stage in ["test", "predict"] or stage is None and self.data_test is None:
            self.data_test = MILK10kDataset(
                root_dir=self.hparams.data_dir,
                task=self.hparams.task,
                transform=self.transforms_test,
            )
            self.data_test = torch.utils.data.Subset(
                self.data_test, self.test_indices
            )
        else:
            # Apply filtering if specified
            self.full_indices = np.arange(len(self.full_dataset))
            if self.hparams.allowed_labels is not None:
                if self.hparams.task == MILK10kTask.MULTILABEL:
                    label_matrix = self.full_dataset.data[self.full_dataset.class_codes].values
                    labels = np.argmax(label_matrix, axis=1)
                else:
                    labels = self.full_dataset.data["NV"].to_numpy()

                self.full_indices, _ = self._filter_and_remap_indices(
                    self.full_indices, labels, self.hparams.allowed_labels
                )


    def train_dataloader(self):
        sampler = None
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
        return self.test_dataloader()

    def all_dataloader(self):
        return DataLoader(
            torch.utils.data.Subset(self.full_dataset, self.full_indices),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    # Add this method to MILK10kDataModule class
    def export_dataset(self, output_dir: str, **kwargs):
        """Export dataset using the RGB exporter."""
        from src.utils.dataset_exporter import RGBDatasetExporter

        exporter = RGBDatasetExporter(self, output_dir)
        exporter.export(**kwargs)

        # also copy MILK10k_Training_GroundTruth.csv and MILK10k_Training_Metadata.csv from the data_dir to output_dir
        import shutil
        data_dir = Path(self.hparams.data_dir)
        shutil.copy(data_dir / "MILK10k_Training_GroundTruth.csv",
                    Path(output_dir) / "MILK10k_Training_GroundTruth.csv")
        shutil.copy(data_dir / "MILK10k_Training_Metadata.csv",
                    Path(output_dir) / "MILK10k_Training_Metadata.csv")

    def _get_tags_and_run_name(self):
        """Attach automatic tags and run name inferred from hparams."""

        hparams = getattr(self, "hparams", None)
        if hparams is None:
            return

        tags = ["milk10k"]
        run_name = "milk10k_"

        if hasattr(hparams, "data_dir") and "crop" in hparams.data_dir.lower():
                tags.append("cropped")
                run_name += "crop_"

        if getattr(hparams, "infinite_train", False):
            tags.append("infinite_train")

        if hasattr(hparams, "allowed_labels") and hparams.allowed_labels:
            labels = hparams.allowed_labels
            if isinstance(labels, list):
                for label in labels:
                    tags.append(label.lower())

        # Core metadata
        if getattr(hparams, 'task', None):
            run_name += f"{str(getattr(hparams, 'task')).lower()}_"
            tags.append(str(getattr(hparams, 'task')).lower())

        if "train" in hparams.transforms:
            transforms = hparams.transforms["train"]
            not_augs = [
                "ToTensorV2",
                "Normalize",
                "PadIfNeeded",
                "CenterCrop",
                "Resize",
                "Equalize",
                "SmallestMaxSize",
                "LongestMaxSize",
            ]
            has_augmentation = any(
                transform.get("class_path") not in not_augs for transform in transforms
            )
            if has_augmentation:
                run_name += "aug_"
                tags.append("augmented")

        return tags, run_name.rstrip("_")

if __name__ == "__main__":
    image_size = 256
    data_module = MILK10kDataModule(
        task="segmentation",
        train_val_test_split=(0.7, 0.15, 0.15),
        batch_size=8,
        data_dir="data/MILK10k",
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
    )
    data_module.prepare_data()
    data_module.setup()

    # Export dataset example
    data_module.export_dataset(
        output_dir="export/milk10k_melanoma_cropped_256",
        crop_with_mask=True,
        bbox_scale=2,
        structure="original",
        image_size=image_size,
        allowed_labels=[
                        "melanoma",
                        # "melanocytic_nevus"
                        ],
        global_normalization=False,
        export_cropped_masks=True,
    )
