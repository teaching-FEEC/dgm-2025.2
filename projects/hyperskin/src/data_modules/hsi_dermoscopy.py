import os
import pytorch_lightning as pl
from torchvision.transforms import transforms as T
from torch.utils.data import Dataset, random_split, DataLoader
import albumentations as A

from src.data_modules.datasets.hsi_dermoscopy import HSIDermoscopyDataset, HSIDermoscopyTask

class HSIDermoscopyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        task: str | HSIDermoscopyTask,
        train_val_test_split: tuple[int, int, int] | tuple[float, float, float],
        input_shape: tuple[int, int, int],
        batch_size: int,
        num_workers: int = 8,
        pin_memory: bool = False,
        data_dir: str = "data/hsi_dermoscopy"
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        if isinstance(task, str):
            self.hparams.task = HSIDermoscopyTask[task]

        self.transforms = A.Compose([
            A.PadIfNeeded(min_height=self.hparams.input_shape[1], min_width=self.hparams.input_shape[2]),
            A.Resize(height=self.hparams.input_shape[1], width=self.hparams.input_shape[2]),
            A.ToTensorV2()
        ])

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def prepare_data(self):
        if not os.listdir(self.hparams.data_dir):
            url = 'https://drive.google.com/drive/folders/13ZXXxtUMjEzLAjoAqC19IJEc5RgpFf2f'
            raise RuntimeError(
                f"Data directory {self.hparams.data_dir} is empty. "
                f"Please download the dataset from {url} and place it there."
            )


    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None and self.data_train is None and self.data_val is None:
            full_dataset = HSIDermoscopyDataset(task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms)
            self.data_train, self.data_val, self.data_test = random_split(full_dataset,
                                                                          self.hparams.train_val_test_split)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None and self.data_test is None:
            self.data_test = HSIDermoscopyDataset(task=self.hparams.task,
                data_dir=self.hparams.data_dir,
                transform=self.transforms)

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
