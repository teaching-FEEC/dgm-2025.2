from pathlib import Path
from typing import Literal, Optional

import torch
import pytorch_lightning as pl

from src.samplers.finite import FiniteSampler

if __name__ == "__main__":
    import pyrootutils

    pyrootutils.setup_root(
        Path(__file__).parent.parent.parent,
        project_root_env_var=True,
        dotenv=True,
        pythonpath=True,
        cwd=False,
    )
from src.data_modules.hsi_dermoscopy import HSIDermoscopyDataModule
from src.data_modules.milk10k import MILK10kDataModule
from src.data_modules.isic2019 import ISIC2019DataModule


class JointRGBHSIDataModule(pl.LightningDataModule):
    """Parent DataModule that wraps both HSI Dermoscopy and RGB DataModules.

    It doesn't merge datasets; instead, it exposes a dictionary of dataloaders
    (HSI first, then RGB) for training, validation, and testing.
    """
    prepare_data_per_node: bool = True

    def __init__(
        self,
        hsi_config: dict,
        rgb_config: Optional[dict] = None,
        milk10k_config: Optional[dict] = None,
        rgb_dataset: Literal["milk10k", "isic2019"] = "milk10k",
        num_workers: int = 8,
        pin_memory: bool = False,
        rgb_only: bool = False,
        pred_num_samples: int | None = 100,
    ):
        super().__init__()  # Ensure Lightning internal hooks exist

        if milk10k_config is not None and rgb_dataset == "milk10k" and rgb_config is None:
            rgb_config = milk10k_config

        if rgb_config is None:
            raise ValueError("rgb_config must be provided if milk10k_config is not used.")

        # Saves configs for LightningCLI and adds `_log_hyperparams`
        self.save_hyperparameters({
            "hsi_config": hsi_config,
            "rgb_config": rgb_config,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "rgb_only": rgb_only,
            "pred_num_samples": pred_num_samples,
        })

        # Now safely build internal datamodules
        self.hsi_dm = HSIDermoscopyDataModule(**hsi_config)
        if rgb_dataset == "milk10k":
            self.rgb_dm = MILK10kDataModule(**rgb_config)
        elif rgb_dataset == "isic2019":
            self.rgb_dm = ISIC2019DataModule(**rgb_config)
        else:
            raise ValueError(f"Unsupported rgb_dataset: {rgb_dataset}")
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        """Prepare both datamodules."""
        if not self.hparams.rgb_only:
            self.hsi_dm.prepare_data()
        self.rgb_dm.prepare_data()

    def setup(self, stage: str | None = None):
        """Setup both datamodules."""
        if not self.hparams.rgb_only:
            self.hsi_dm.setup(stage)
        self.rgb_dm.setup(stage)

    def train_dataloader(self):
        """Return a dict of training dataloaders: {'hsi': ..., 'rgb': ...}."""
        if self.hparams.rgb_only:
            return {
                "rgb": self.rgb_dm.train_dataloader(),
            }
        return {
            "hsi": self.hsi_dm.train_dataloader(),
            "rgb": self.rgb_dm.train_dataloader(),
        }

    def val_dataloader(self):
        """Return validation dataloaders dict."""
        if self.hparams.rgb_only:
            return self.rgb_dm.val_dataloader()
        return [
            self.hsi_dm.val_dataloader(),
            self.rgb_dm.val_dataloader()
        ]

    def test_dataloader(self):
        """Return test dataloaders dict."""
        if self.hparams.rgb_only:
            return self.rgb_dm.test_dataloader()
        return [
            self.hsi_dm.test_dataloader(),
            self.rgb_dm.test_dataloader(),
        ]

    def predict_dataloader(self):
        """Return both predict dataloaders (if needed)."""
        if self.hparams.rgb_only:
            if self.hparams.pred_num_samples:
                sampler = FiniteSampler(self.rgb_dm.predict_dataloader().dataset,
                                        self.hparams.pred_num_samples)
                return torch.utils.data.DataLoader(
                    self.rgb_dm.predict_dataloader().dataset,
                    batch_size=self.hparams.rgb_config.get("batch_size", 1),
                    sampler=sampler,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )
            else:
                return self.rgb_dm.predict_dataloader()
        else:
            dataloaders = []
            for dm in [self.hsi_dm, self.rgb_dm]:
                if self.hparams.pred_num_samples:
                    sampler = FiniteSampler(
                        dm.predict_dataloader().dataset,
                        self.hparams.pred_num_samples,
                    )
                    dl = torch.utils.data.DataLoader(
                        dm.predict_dataloader().dataset,
                        batch_size=dm.hparams.get("batch_size", 1),
                        sampler=sampler,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                    )
                    dataloaders.append(dl)
                else:
                    dataloaders.append(dm.predict_dataloader())
            return dataloaders

    def teardown(self, stage: str | None = None):
        """Teardown both datamodules."""
        if not self.hparams.rgb_only:
            self.hsi_dm.teardown(stage)
        self.rgb_dm.teardown(stage)

    def _get_tags_and_run_name(self):
        """Attach automatic tags and run name inferred from hparams."""

        if not self.hparams.rgb_only:
            tags, run_name = self.hsi_dm._get_tags_and_run_name()
            run_name = run_name.replace("hsi_", "hsi_rgb_")
        else:
            tags, run_name = self.rgb_dm._get_tags_and_run_name()

        return tags, run_name

if __name__ == "__main__":
    image_size = 256

    transforms = {
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
    }

    hsi_cfg = dict(
        task="GENERATION",
        train_val_test_split=(0.7, 0.15, 0.15),
        batch_size=8,
        data_dir="data/hsi_dermoscopy_croppedv2_256_with_masks",
        image_size=image_size,
        transforms=transforms,
        allowed_labels=["melanoma"],
    )

    rgb_cfg = dict(
        task="GENERATION_MELANOMA_VS_NEVUS",
        train_val_test_split=(1, 0, 0),
        batch_size=8,
        data_dir="data/milk10k_melanoma_nevus_cropped_256",
        image_size=image_size,
        transforms=transforms,
        allowed_labels=["melanoma"]
    )

    dm_joint = JointRGBHSIDataModule(hsi_config=hsi_cfg, rgb_config=rgb_cfg)

    dm_joint.prepare_data()
    dm_joint.setup()

    loaders = dm_joint.train_dataloader()
    print("Train dataloaders keys:", loaders.keys())

    for name, loader in loaders.items():
        print(f"{name} dataloader size:", len(loader))

    hsi_loader = loaders["hsi"]
    rgb_loader = loaders["rgb"]

    hsi_sample = next(iter(hsi_loader))
    rgb_sample = next(iter(rgb_loader))

    hsi_image, hsi_mask, hsi_label = hsi_sample
    rgb_image, rgb_mask, rgb_label = rgb_sample

    # assert masks are binary
    assert torch.unique(hsi_mask).tolist() == [0, 1], "HSI mask is not binary"
    assert torch.unique(rgb_mask).tolist() == [0, 1], "RGB mask is not binary"
