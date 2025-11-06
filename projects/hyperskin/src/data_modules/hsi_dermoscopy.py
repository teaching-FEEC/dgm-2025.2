from pathlib import Path

from git import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(
        Path(__file__).parent.parent.parent, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False
    )

from src.data_modules.base import BaseDataModule
from src.samplers.infinite import InfiniteSamplerWrapper
from src.utils.transform import smallest_maxsize_and_centercrop
from src.samplers.balanced_batch_sampler import BalancedBatchSampler
from src.data_modules.datasets.hsi_dermoscopy_dataset import HSIDermoscopyDataset, HSIDermoscopyTask

class HSIDermoscopyDataModule(BaseDataModule):
    def __init__(
        self,
        task: str | HSIDermoscopyTask,
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
        infinite_train:  bool = False,
        sample_size: Optional[int] = None,
        synthetic_data_dir: Optional[str] = None,
        range_mode: str = '-1_1',
        normalize_mask_tanh: bool = False,
        **kwargs,
    ):
        self.save_hyperparameters()
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

        # save data_dir to hyperparams for access in setup
        self.save_hyperparameters({"data_dir": data_dir,
                                   "synthetic_data_dir": synthetic_data_dir,
                                   "balanced_sampling": balanced_sampling,
                                   "infinite_train": infinite_train,
                                   "batch_size": batch_size,
                                   "num_workers": num_workers,
                                   "pin_memory": pin_memory,
                                   "sample_size": sample_size,
                                   "allowed_labels": allowed_labels,
                                    "task": task
                                   })

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if isinstance(task, str):
            self.hparams.task = HSIDermoscopyTask[task.upper()]

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def get_dataset_indices_and_labels(self) -> tuple[np.ndarray, np.ndarray]:
        full_dataset = HSIDermoscopyDataset(
            task=self.hparams.task, data_dir=self.hparams.data_dir
        )
        indices = np.arange(len(full_dataset))
        labels = full_dataset.labels_df["label"].map(full_dataset.labels_map).to_numpy()
        return indices, labels

    def get_labels_map(self) -> dict[str, int]:
        full_dataset = HSIDermoscopyDataset(
            task=self.hparams.task, data_dir=self.hparams.data_dir
        )
        return full_dataset.labels_map

    def setup(self, stage: str = None):
        self.ensure_splits_exist()

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

    # Add this method to HSIDermoscopyDataModule class
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

        if hasattr(hparams, "data_dir") and "crop" in hparams.data_dir.lower():
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
            labels = [inv_labels_map[label] if isinstance(label, int) else label for label in labels]
            for label in labels:
                tags.append(label.lower())

        # Core metadata
        if getattr(hparams, 'task', None):
            task_name = getattr(hparams, 'task').name.lower()
            if "segmentation" in task_name:
                run_name += "seg_"
            elif "generation" in task_name:
                run_name += "gen_"
            else:
                run_name += "cls_"

            run_name += f"{getattr(hparams, 'task').name.lower()}_"
            tags.append(getattr(hparams, 'task').name.lower())

        if getattr(hparams, "synthetic_data_dir", None):
            run_name += "synth_"
            tags.append("synthetic_data")

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
                transform.get("class_path") not in not_augs for transform in transforms
            )
            if has_augmentation:
                run_name += "aug_"
                tags.append("augmented")

        return tags, run_name.rstrip("_")

if __name__ == "__main__":

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
        google_drive_id="1BQWqSq5Q0xfu381VNwyVU8XlXaIy9ds9",
        # synthetic_data_dir="data/hsi_dermoscopy_cropped_synth",
    )
    data_module.prepare_data()
    data_module.setup()

    # Export dataset example
    data_module.export_dataset(
        output_dir="export/hsi_dermoscopy_croppedv2_256_with_masks",
        # splits=["train", "val", "test"],
        crop_with_mask=True,
        bbox_scale=2,
        structure="original",
        image_size=image_size,
        allowed_labels=["melanoma", "dysplastic_nevi"],
        global_normalization=False,
        export_cropped_masks=True,
    )
