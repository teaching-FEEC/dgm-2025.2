# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.data_modules.hsi_dermoscopy import HSIDermoscopyDataModule
from src.models import TIMMModel
from pytorch_lightning.loggers import WandbLogger

class HSIClassifierModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        features_only: bool,
        in_chans: int,
        scriptable: bool,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.net = TIMMModel(model_name=self.hparams.model_name,
                             num_classes=self.hparams.num_classes,
                             pretrained=self.hparams.pretrained,
                             features_only=self.hparams.features_only,
                             in_chans=self.hparams.in_chans,
                             scriptable=self.hparams.scriptable)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        metric = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()

        # for averaging loss across batches
        loss_metric = MeanMetric()
        self.train_loss = loss_metric.clone()
        self.val_loss = loss_metric.clone()
        self.test_loss = loss_metric.clone()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

        # Save data splits if the logger and datamodule are configured correctly
        if self.trainer.logger and hasattr(self.trainer.logger, 'save_dir') and \
            isinstance(self.trainer.datamodule, HSIDermoscopyDataModule):
            logger = self.trainer.logger
            datamodule = self.trainer.datamodule

            # Define local path for saving splits
            split_dir = Path(logger.save_dir) / "data_splits"

            # Save splits to local log directory
            datamodule.save_splits_to_disk(split_dir)

            # If using wandb, upload the split files
            if isinstance(logger, WandbLogger):
                # Use glob to save all .txt files in the directory
                logger.experiment.save(str(split_dir / "*.txt"), base_path=logger.save_dir)

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute())

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass


if __name__ == "__main__":
    m = HSIClassifierModule()
    print(m)
