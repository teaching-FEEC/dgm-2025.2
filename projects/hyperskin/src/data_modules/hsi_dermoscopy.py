from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from collections import Counter

from src.data_modules.datasets.task_config import TaskConfig
from src.samplers.finite import FiniteSampler

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(
        Path(__file__).parent.parent.parent,
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=False,
    ) #mudar o caminho raiz do projeto

from src.data_modules.base import BaseDataModule
from src.samplers.infinite import (
    InfiniteBalancedBatchSampler,
    InfiniteSamplerWrapper,
)
from src.samplers.balanced_batch_sampler import BalancedBatchSampler
from src.data_modules.datasets.hsi_dermoscopy_dataset import (
    HSI_TASK_CONFIGS,
    HSIDermoscopyDataset,
)
import pytorch_lightning as pl


class HSIDermoscopyDataModule(BaseDataModule, pl.LightningDataModule):
    def __init__(
        self,
        task: str | TaskConfig,
        train_val_test_split: tuple[int, int, int] | tuple[float, float, float],
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = False,
        data_dir: str = "data/hsi_dermoscopy",
        transforms: Optional[dict] = None,
        google_drive_id: Optional[str] = None,
        allowed_labels: Optional[list[int | str]] = None,
        image_size: int = 224,
        global_max: Optional[float | list[float]] = None,
        global_min: Optional[float | list[float]] = None,
        balanced_sampling: bool = False,
        infinite_train: bool = False,
        synthetic_data_dir: Optional[str] = None,
        range_mode: str = "-1_1",
        normalize_mask_tanh: bool = False,
        pred_num_samples: Optional[int] = None,
        undersample_strategy: Optional[str] = None,
        oversample_strategy: Optional[str] = None,
        sampling_random_state: int = 42,
        synth_mode: Optional[str] ='mixed_train', #full_train, full_val, mixed_train, None
        synth_ratio: float = 1.0,  # only used if synth_mode is mixed_train
        **kwargs,
    ):
        """
        Args:
            undersample_strategy: Strategy for undersampling. Options:
                - "random": Random undersampling to match minority class
                - "majority": Undersample only majority class
                - None: No undersampling
            oversample_strategy: Strategy for oversampling. Options:
                - "random": Random oversampling to match majority class
                - "minority": Oversample only minority class
                - None: No oversampling
            sampling_random_state: Random seed for sampling operations
        """
        #iniciando a BaseDataModule e o LightningDataModule
        super().__init__(
            train_val_test_split=train_val_test_split,
            data_dir=data_dir,
            allowed_labels=allowed_labels,
            google_drive_id=google_drive_id,
            transforms=transforms,
            image_size=image_size,
            global_max=global_max,
            global_min=global_min,
            range_mode=range_mode,
            normalize_mask_tanh=normalize_mask_tanh,
        )

        # Normalize dict -> TaskConfig
        if isinstance(task, dict):
            task = TaskConfig(**task)

        if isinstance(task, str):
            task = task.lower()
            if task not in HSI_TASK_CONFIGS:
                raise ValueError(
                    f"Unknown task: {task}. "
                    f"Available: {list(HSI_TASK_CONFIGS.keys())}"
                )
            self.task_config = HSI_TASK_CONFIGS[task]
        elif isinstance(task, TaskConfig):
            self.task_config = task
        else:
            raise TypeError("task must be a str, dict, or TaskConfig")

        self.save_hyperparameters(
            {
                "task": task,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "balanced_sampling": balanced_sampling,
                "infinite_train": infinite_train,
                "synthetic_data_dir": synthetic_data_dir,
                "pred_num_samples": pred_num_samples,
                "data_dir": data_dir,
                "allowed_labels": allowed_labels,
                "undersample_strategy": undersample_strategy,
                "oversample_strategy": oversample_strategy,
                "sampling_random_state": sampling_random_state,
                "synth_mode": synth_mode,
                "synth_ratio": synth_ratio,
            }
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def get_dataset_indices_and_labels(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        full_dataset = HSIDermoscopyDataset(
            task=self.hparams.task, data_dir=self.hparams.data_dir
        )
        indices = np.arange(len(full_dataset))
        labels = (
            full_dataset.labels_df["label"]
            .map(full_dataset.labels_map)
            .to_numpy()
        )
        return indices, labels

    def get_labels_map(self) -> dict[str, int]:
        full_dataset = HSIDermoscopyDataset(
            task=self.hparams.task, data_dir=self.hparams.data_dir
        )
        return full_dataset.labels_map

    def _get_labels_from_dataset(
        self, dataset: torch.utils.data.Dataset
    ) -> np.ndarray:
        """Extract labels from dataset (handles Subset and ConcatDataset)."""
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            labels = []
            for ds in dataset.datasets:
                labels.extend(self._get_labels_from_dataset(ds))
            return np.array(labels)
        elif isinstance(dataset, torch.utils.data.Subset):
            base_labels = self._get_labels_from_dataset(dataset.dataset)
            return base_labels[dataset.indices]
        else:
            return dataset.labels

    def _apply_sampling(
        self, dataset: torch.utils.data.Dataset, split_name: str = "train"
    ) -> torch.utils.data.Dataset:
        """
        Apply undersampling and/or oversampling to the dataset.

        Args:
            dataset: Input dataset (can be Subset or ConcatDataset)
            split_name: Name of the split for logging

        Returns:
            New dataset with sampling applied
        """
        if (
            self.hparams.undersample_strategy is None
            and self.hparams.oversample_strategy is None
        ):
            return dataset

        # Extract labels
        labels = self._get_labels_from_dataset(dataset)
        indices = np.arange(len(labels))

        # Count classes
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))

        print(f"\n[{split_name}] Before sampling: {len(labels)} samples")
        for label, count in label_counts.items():
            print(f"  Class {label}: {count} samples")

        # Apply undersampling
        if self.hparams.undersample_strategy:
            indices, labels = self._undersample(
                indices, labels, self.hparams.undersample_strategy
            )

        # Apply oversampling
        if self.hparams.oversample_strategy:
            indices, labels = self._oversample(
                indices, labels, self.hparams.oversample_strategy
            )

        # Count after sampling
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))

        print(f"[{split_name}] After sampling: {len(labels)} samples")
        for label, count in label_counts.items():
            print(f"  Class {label}: {count} samples")

        # Create new subset with sampled indices
        # Create a Subset of the existing dataset
        return torch.utils.data.Subset(dataset, indices.tolist())

    def _undersample(
        self, indices: np.ndarray, labels: np.ndarray, strategy: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply undersampling strategy."""
        np.random.seed(self.hparams.sampling_random_state)

        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = counts.min()

        if strategy == "random":
            # Undersample all classes to match minority
            new_indices = []
            for label in unique_labels:
                label_mask = labels == label
                label_indices = indices[label_mask]
                sampled = np.random.choice(
                    label_indices, size=min_count, replace=False
                )
                new_indices.extend(sampled)

            new_indices = np.array(new_indices)
            new_labels = labels[new_indices]
            return new_indices, new_labels

        elif strategy == "majority":
            # Undersample only majority class(es)
            max_count = counts.max()
            if max_count == min_count:
                return indices, labels

            new_indices = []
            for label in unique_labels:
                label_mask = labels == label
                label_indices = indices[label_mask]
                label_count = len(label_indices)

                if label_count > min_count:
                    # Undersample this class
                    sampled = np.random.choice(
                        label_indices, size=min_count, replace=False
                    )
                    new_indices.extend(sampled)
                else:
                    new_indices.extend(label_indices)

            new_indices = np.array(new_indices)
            new_labels = labels[new_indices]
            return new_indices, new_labels

        else:
            raise ValueError(
                f"Unknown undersample_strategy: {strategy}. "
                f"Use 'random' or 'majority'"
            )

    def _oversample(
        self, indices: np.ndarray, labels: np.ndarray, strategy: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply oversampling strategy."""
        np.random.seed(self.hparams.sampling_random_state)

        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = counts.max()

        if strategy == "random":
            # Oversample all classes to match majority
            new_indices = []
            for label in unique_labels:
                label_mask = labels == label
                label_indices = indices[label_mask]
                label_count = len(label_indices)

                if label_count < max_count:
                    # Oversample this class
                    n_to_add = max_count - label_count
                    sampled = np.random.choice(
                        label_indices, size=n_to_add, replace=True
                    )
                    new_indices.extend(label_indices)
                    new_indices.extend(sampled)
                else:
                    new_indices.extend(label_indices)

            new_indices = np.array(new_indices)
            new_labels = labels[new_indices]
            return new_indices, new_labels

        elif strategy == "minority":
            # Oversample only minority class(es)
            min_count = counts.min()
            if min_count == max_count:
                return indices, labels

            new_indices = []
            for label in unique_labels:
                label_mask = labels == label
                label_indices = indices[label_mask]
                label_count = len(label_indices)

                if label_count < max_count:
                    # Oversample this class
                    n_to_add = max_count - label_count
                    sampled = np.random.choice(
                        label_indices, size=n_to_add, replace=True
                    )
                    new_indices.extend(label_indices)
                    new_indices.extend(sampled)
                else:
                    new_indices.extend(label_indices)

            new_indices = np.array(new_indices)
            new_labels = labels[new_indices]
            return new_indices, new_labels

        else:
            raise ValueError(
                f"Unknown oversample_strategy: {strategy}. "
                f"Use 'random' or 'minority'"
            )

    def _create_concat_subset(
        self,
        concat_dataset: torch.utils.data.ConcatDataset,
        indices: np.ndarray,
    ) -> torch.utils.data.Subset:
        """Create subset from ConcatDataset using global indices."""
        return torch.utils.data.Subset(concat_dataset, indices)

    def setup(self, stage: str = None):
        if stage in ["fit", "validate"] or stage is None and (
            self.data_train is None or self.data_val is None
        ):
            self.data_train = HSIDermoscopyDataset(
                task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms_train,
            )
            self.data_train = torch.utils.data.Subset( #aqui acontece a subdivisão dos dados só de treino 
                self.data_train, self.train_indices
            ) 

            self.data_val = HSIDermoscopyDataset(
                task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms_val,
            )
            self.data_val = torch.utils.data.Subset(
                self.data_val, self.val_indices
            )

            # Add synthetic data to training set if provided
            if self.hparams.synthetic_data_dir is not None:
                synthetic_dataset = HSIDermoscopyDataset( #dataset sintético inteiro
                    task=self.hparams.task,
                    data_dir=self.hparams.synthetic_data_dir,
                    transform=self.transforms_train,
                )
                indices_ratio = int(len(synthetic_dataset)*self.hparams.synth_ratio) #quantos dados sintéticos vamos usar NÃO COLOCAR 0 
                synthetic_indices = np.arange(indices_ratio)

                if self.hparams.allowed_labels is not None:
                    synthetic_labels = (
                        synthetic_dataset.labels_df["label"]
                        .map(synthetic_dataset.labels_map)
                        .to_numpy()
                    )
                    synthetic_indices, _ = self._filter_and_remap_indices( #filtra apenas as labels que queremos usar
                        synthetic_indices,
                        synthetic_labels,
                        self.hparams.allowed_labels,
                    )

                synthetic_subset = torch.utils.data.Subset(
                    synthetic_dataset, synthetic_indices
                )
                if self.hparams.synth_mode == 'full_train':  #usa só sintético no treino
                    self.data_train = synthetic_subset
                elif self.hparams.synth_mode == 'full_val': #usa só sintético na validação
                    self.data_val = synthetic_subset
                elif self.hparams.synth_mode == 'mixed_train':  #usa uma mistura de real e sintético no treino
                    self.data_train = torch.utils.data.ConcatDataset(
                        [self.data_train, synthetic_subset]
                    )

                print(
                    f"Added {len(synthetic_subset)} synthetic samples "
                    f"to training set"
                )

            # Apply sampling after synthetic data is added
            self.data_train = self._apply_sampling(
                self.data_train, split_name="train"
            )

           
        if stage in ["test", "predict"] or stage is None and (
            self.data_test is None
        ):
            self.data_test = HSIDermoscopyDataset(
                task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms_test,
            )
            self.data_test = torch.utils.data.Subset(
                self.data_test, self.test_indices
            )

        # Print split statistics
        self._print_split_statistics()

    def _print_split_statistics(self):
        """Print label distribution table for each split."""
        print("\n" + "=" * 70)
        print("DATASET SPLIT STATISTICS")
        print("=" * 70)

        # Get label map for display
        labels_map = self.get_labels_map()
        inv_labels_map = {v: k for k, v in labels_map.items()}

        splits_info = []
        if self.data_train is not None:
            splits_info.append(("Train", self.data_train))
        if self.data_val is not None:
            splits_info.append(("Val", self.data_val))
        if self.data_test is not None:
            splits_info.append(("Test", self.data_test))

        for split_name, dataset in splits_info:
            labels = self._get_labels_from_dataset(dataset)
            unique_labels, counts = np.unique(labels, return_counts=True)

            print(f"\n{split_name} Split:")
            print(f"{'Class':<30} {'Count':<10} {'Percentage':<10}")
            print("-" * 50)

            total = len(labels)
            for label, count in zip(unique_labels, counts):
                class_name = inv_labels_map.get(label, f"Unknown({label})")
                percentage = (count / total) * 100
                print(
                    f"{class_name:<30} {count:<10} {percentage:>6.2f}%"
                )

            print(f"{'Total':<30} {total:<10} {100.0:>6.2f}%")

        print("=" * 70 + "\n")

    def train_dataloader(self):
        sampler = None
        labels = None
        if self.hparams.balanced_sampling:
            labels = self._get_labels_from_dataset(self.data_train)

        if self.hparams.infinite_train:
            sampler = InfiniteSamplerWrapper(self.data_train)
        elif (
            self.hparams.balanced_sampling
            and self.hparams.infinite_train
        ):
            sampler = InfiniteBalancedBatchSampler(
                labels, batch_size=self.hparams.batch_size
            )
            return DataLoader(
                self.data_train,
                batch_sampler=sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
        elif self.hparams.balanced_sampling and not self.hparams.infinite_train:
            sampler = BalancedBatchSampler(
                labels, batch_size=self.hparams.batch_size
            )

            return DataLoader(
                self.data_train,
                batch_sampler=sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler,
            shuffle=(sampler is None),
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
        if not self.hparams.pred_num_samples:
            return self.all_dataloader()
        else:
            dataloader = self.all_dataloader()
            sampler = FiniteSampler(
                dataloader.dataset, self.hparams.pred_num_samples
            )
            return DataLoader(
                dataloader.dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                sampler=sampler,
            )

    def all_dataloader(self):
        full_dataset = HSIDermoscopyDataset(
            task=self.hparams.task,
            data_dir=self.hparams.data_dir,
            transform=self.transforms_test,
        )

        indices = np.arange(len(full_dataset))
        labels = (
            full_dataset.labels_df["label"]
            .map(full_dataset.labels_map)
            .to_numpy()
        )
        indices, _ = self._filter_and_remap_indices(
            indices, labels, self.hparams.allowed_labels
        )
        filtered_dataset = torch.utils.data.Subset(full_dataset, indices)

        return DataLoader(
            filtered_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage=None):
        pass

    def export_dataset(self, output_dir: str, **kwargs):
        """Export dataset using the HSI exporter."""
        from src.utils.dataset_exporter import HSIDatasetExporter

        exporter = HSIDatasetExporter(
            self,
            output_dir,
            global_min=self.global_min,
            global_max=self.global_max,
        )
        exporter.export(**kwargs)

    def _get_tags_and_run_name(self):
        """Attach automatic tags and run name inferred from hparams."""

        hparams = getattr(self, "hparams", None)
        if hparams is None:
            return

        tags = ["hsi_dermoscopy"]
        run_name = "hsi_"

        if hasattr(hparams, "data_dir") and "crop" in (
            hparams.data_dir.lower()
        ):
            tags.append("cropped")
            run_name += "crop_"

        if getattr(hparams, "balanced_sampling", False):
            tags.append("balanced_sampling")

        if getattr(hparams, "infinite_train", False):
            tags.append("infinite_train")

        if hasattr(hparams, "allowed_labels") and hparams.allowed_labels:
            labels = hparams.allowed_labels
            labels_map = self.get_labels_map()
            inv_labels_map = {v: k for k, v in labels_map.items()}
            labels = [
                inv_labels_map[label] if isinstance(label, int) else label
                for label in labels
            ]
            for label in labels:
                tags.append(label.lower())

        if getattr(hparams, "synthetic_data_dir", None):
            run_name += "synth_"
            tags.append("synthetic_data")
        
        if getattr(hparams, "synth_mode", None):
            run_name += hparams.synth_mode + "_"
            tags.append(hparams.synth_mode)

        if getattr(hparams, "synth_ratio", None):
            if hparams.synth_ratio != 1.0:
                run_name += "r" + str(hparams.synth_ratio) + "_"
                tags.append(hparams.synth_mode)

        if "train" in self.transforms_cfg:
            transforms = self.transforms_cfg["train"]
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
                transform.get("class_path") not in not_augs
                for transform in transforms
            )
            if has_augmentation:
                run_name += "aug_"
                tags.append("augmented")

        if self.hparams.undersample_strategy == "random":
            run_name += "randunder_"
            tags.append("random_undersampling")
        elif self.hparams.undersample_strategy == "majority":
            run_name += "majunder_"
            tags.append("majority_undersampling")

        if self.hparams.oversample_strategy == "random":
            run_name += "randover_"
            tags.append("random_oversampling")
        elif self.hparams.oversample_strategy == "minority":
            run_name += "minover_"
            tags.append("minority_oversampling")

        return tags, run_name.rstrip("_")


if __name__ == "__main__":
    # Example usage with sampling
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
                {
                    "class_path": "SmallestMaxSize",
                    "init_args": {"max_size": image_size},
                },
                {
                    "class_path": "CenterCrop",
                    "init_args": {
                        "height": image_size,
                        "width": image_size,
                    },
                },
                {"class_path": "ToTensorV2", "init_args": {}},
            ],
            "val": [
                {
                    "class_path": "SmallestMaxSize",
                    "init_args": {"max_size": image_size},
                },
                {
                    "class_path": "CenterCrop",
                    "init_args": {
                        "height": image_size,
                        "width": image_size,
                    },
                },
                {"class_path": "ToTensorV2", "init_args": {}},
            ],
            "test": [
                {
                    "class_path": "SmallestMaxSize",
                    "init_args": {"max_size": image_size},
                },
                {
                    "class_path": "CenterCrop",
                    "init_args": {
                        "height": image_size,
                        "width": image_size,
                    },
                },
                {"class_path": "ToTensorV2", "init_args": {}},
            ],
        },
        google_drive_id="1BQWqSq5Q0xfu381VNwyVU8XlXaIy9ds9",
        # synthetic_data_dir="data/hsi_dermoscopy_cropped_synth",
        # Example: Apply random undersampling
        undersample_strategy="random",
        # Example: Apply random oversampling
        # oversample_strategy="random",
    )
    data_module.prepare_data()
    data_module.setup()

    # The split statistics will be printed automatically after setup
