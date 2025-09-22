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

from src.data_modules.datasets.hsi_dermoscopy import HSIDermoscopyDataset, HSIDermoscopyTask

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
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        if isinstance(task, str):
            self.hparams.task = HSIDermoscopyTask[task]

        self.transforms_train = None
        self.transforms_test = None

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
        if not os.listdir(self.hparams.data_dir):
            url = 'https://drive.google.com/drive/folders/13ZXXxtUMjEzLAjoAqC19IJEc5RgpFf2f'
            raise RuntimeError(
                f"Data directory {self.hparams.data_dir} is empty. "
                f"Please download the dataset from {url} and place it there."
            )

        self.setup_splits()

    def setup_splits(self):
        seed = 42
        split_dir = Path(self.hparams.data_dir) / 'splits'

        # append a dir to splits_dir containing the seed used for the split, the task name, and the train/val/test ratio
        split_dir = split_dir / f'seed{seed}_{self.hparams.task.name}_'f'{self.hparams.train_val_test_split[0]}_' \
                    f'{self.hparams.train_val_test_split[1]}_' \
                    f'{self.hparams.train_val_test_split[2]}'

        # check if train.txt, val.txt, test.txt exist in split_dir
        files_exist = all((split_dir / f).exists() for f in ['train.txt', 'val.txt', 'test.txt'])

        if not split_dir.exists() or not files_exist:
            os.makedirs(split_dir, exist_ok=True)
            full_dataset = HSIDermoscopyDataset(task=self.hparams.task,
                data_dir=self.hparams.data_dir)

            indices = full_dataset.label_df_indices

            # if the split ratios are integers, use them as absolute numbers
            if all(isinstance(x, int) for x in self.hparams.train_val_test_split):
                train_size, val_size, test_size = self.hparams.train_val_test_split
                if train_size + val_size + test_size != len(full_dataset):
                    raise ValueError("When using absolute numbers for train/val/test split, "
                                     "the sum must equal the total number of samples in the dataset.")
                self.train_indices, temp_indices = train_test_split(indices, train_size=train_size, random_state=seed)
                self.val_indices, self.test_indices = train_test_split(temp_indices, train_size=val_size,
                                                                       random_state=seed)
            # if the split ratios are floats and sum to 1, use them as proportions
            elif all(isinstance(x, float) for x in self.hparams.train_val_test_split) and \
                    abs(sum(self.hparams.train_val_test_split) - 1.0) < 1e-6:
                train_ratio, val_ratio, test_ratio = self.hparams.train_val_test_split
                self.train_indices, temp_indices = train_test_split(indices, train_size=train_ratio, random_state=seed)
                val_size = val_ratio / (val_ratio + test_ratio)
                self.val_indices, self.test_indices = train_test_split(temp_indices, train_size=val_size,
                                                                       random_state=seed)
            else:
                raise ValueError("train_val_test_split must be either all integers or all floats that sum to 1.")

            # save the indices to txt files
            np.savetxt(split_dir / 'train.txt', self.train_indices, fmt='%d')
            np.savetxt(split_dir / 'val.txt', self.val_indices, fmt='%d')
            np.savetxt(split_dir / 'test.txt', self.test_indices, fmt='%d')
        else:
            self.train_indices = np.loadtxt(split_dir / 'train.txt', dtype=int)
            self.val_indices = np.loadtxt(split_dir / 'val.txt', dtype=int)
            self.test_indices = np.loadtxt(split_dir / 'test.txt', dtype=int)

    def setup(self, stage: str = None):
        # Create the full dataset
        full_dataset = HSIDermoscopyDataset(task=self.hparams.task, data_dir=self.hparams.data_dir)

        # Use the indices from setup_splits to create the splits
        if stage == 'fit' or stage is None and self.data_train is None and self.data_val is None:
            full_dataset.transform = self.transforms_train
            self.data_train = torch.utils.data.Subset(full_dataset, self.train_indices)

            full_dataset.transform = self.transforms_val
            self.data_val = torch.utils.data.Subset(full_dataset, self.val_indices)

        # Assign test dataset
        if stage == 'test' or stage is None and self.data_test is None:
            full_dataset.transform = self.transforms_test
            self.data_test = torch.utils.data.Subset(full_dataset, self.test_indices)

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

    def teardown(self, stage=None):
        # Called on every process after trainer is done
        pass
