from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

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
        )

        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if isinstance(task, str):
            self.hparams.task = MILK10kTask[task.upper()]

        # Initialize dataset attributes
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def get_dataset_indices_and_labels(self) -> tuple[np.ndarray, np.ndarray]:
        full_dataset = MILK10kDataset(
            root_dir=self.hparams.data_dir, task=self.hparams.task
        )

        indices = np.arange(len(full_dataset))

        if self.hparams.task == MILK10kTask.MULTILABEL:
            # For multilabel, use first positive class index for stratification
            label_matrix = full_dataset.data[full_dataset.class_codes].values
            labels = np.argmax(label_matrix, axis=1)
        elif self.hparams.task == MILK10kTask.MELANOMA_VS_NEVUS:
            # For binary: melanoma=0, nevus=1
            labels = full_dataset.data["NV"].to_numpy()
        else:
            raise ValueError(f"Unknown task: {self.hparams.task}")

        return indices, labels

    def get_labels_map(self) -> dict[str, int]:
        full_dataset = MILK10kDataset(
            root_dir=self.hparams.data_dir, task=self.hparams.task
        )
        return {name: i for i, name in enumerate(full_dataset.class_names)}

    def setup(self, stage: str = None):
        self.ensure_splits_exist()

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

            self.data_val = MILK10kDataset(
                root_dir=self.hparams.data_dir,
                task=self.hparams.task,
                transform=self.transforms_val,
            )
            self.data_val = torch.utils.data.Subset(
                self.data_val, self.val_indices
            )

        if stage in ["test", "predict"] or stage is None and self.data_test is None:
            self.data_test = MILK10kDataset(
                root_dir=self.hparams.data_dir,
                task=self.hparams.task,
                transform=self.transforms_test,
            )
            self.data_test = torch.utils.data.Subset(
                self.data_test, self.test_indices
            )

    def train_dataloader(self):
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

    def predict_dataloader(self):
        return self.test_dataloader()

    def all_dataloader(self):
        full_dataset = MILK10kDataset(
            root_dir=self.hparams.data_dir,
            task=self.hparams.task,
            transform=self.transforms_test,
        )

        # Apply filtering if specified
        indices = np.arange(len(full_dataset))
        if self.hparams.allowed_labels is not None:
            if self.hparams.task == MILK10kTask.MULTILABEL:
                label_matrix = full_dataset.data[full_dataset.class_codes].values
                labels = np.argmax(label_matrix, axis=1)
            else:
                labels = full_dataset.data["NV"].to_numpy()

            indices, _ = self._filter_and_remap_indices(
                indices, labels, self.hparams.allowed_labels
            )

        return DataLoader(
            torch.utils.data.Subset(full_dataset, indices),
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

if __name__ == "__main__":
    image_size = 256
    data_module = MILK10kDataModule(
        task="melanoma_vs_nevus",
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
