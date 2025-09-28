# src/models/hsi_segmentation_module.py
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score

from src.models.smp import SMPModel


class HSISegmentationModule(pl.LightningModule):
    def __init__(
        self,
        arch_name: str = "unet",
        encoder_name: str = "resnet18",
        num_classes: int = 2,
        in_chans: int = 16,
        pretrained: bool = True,
        lr: float = 1e-4,
        loss_name: str = "ce",  # could be "ce" or "dice" or "ce+dice"
    ):
        super().__init__()
        self.save_hyperparameters()

        # 🔹 Build Segmentation Model
        self.net = SMPModel(
            arch_name=self.hparams.arch_name,
            encoder_name=self.hparams.encoder_name,
            num_classes=self.hparams.num_classes,
            in_chans=self.hparams.in_chans,
            pretrained=self.hparams.pretrained,
        )

        # 🔹 Loss
        if loss_name == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == "dice":
            import segmentation_models_pytorch as smp

            self.criterion = smp.losses.DiceLoss(mode="multiclass")
        elif loss_name == "ce+dice":
            import segmentation_models_pytorch as smp

            self.criterion = (
                nn.CrossEntropyLoss(),
                smp.losses.DiceLoss(mode="multiclass"),
            )
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        # 🔹 Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_iou = MulticlassJaccardIndex(num_classes=num_classes)
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes)
        self.test_iou = MulticlassJaccardIndex(num_classes=num_classes)

        self.val_f1 = MulticlassF1Score(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes)

        # Track best val IoU
        self.val_iou_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def _shared_step(self, batch: Any):
        x, y = batch  # y shape: (B, H, W)
        logits = self.forward(x)  # shape: (B, num_classes, H, W)

        if isinstance(self.criterion, tuple):  # ce+dice combo
            loss = self.criterion[0](logits, y) + self.criterion[1](logits, y)
        else:
            loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    # ---------------- TRAINING ----------------
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch)
        self.train_loss(loss)
        self.train_iou(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True)
        return loss

    # ---------------- VALIDATION ----------------
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch)
        self.val_loss(loss)
        self.val_iou(preds, targets)
        self.val_f1(preds, targets)

        self.log("val/loss", self.val_loss, on_epoch=True)
        self.log("val/iou", self.val_iou, on_epoch=True)
        self.log("val/f1", self.val_f1, on_epoch=True)

    def on_validation_epoch_end(self):
        current_iou = self.val_iou.compute()
        self.val_iou_best(current_iou)
        self.log("val/iou_best", self.val_iou_best.compute())

    # ---------------- TEST ----------------
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self._shared_step(batch)
        self.test_loss(loss)
        self.test_iou(preds, targets)
        self.test_f1(preds, targets)

        self.log("test/loss", self.test_loss, on_epoch=True)
        self.log("test/iou", self.test_iou, on_epoch=True)
        self.log("test/f1", self.test_f1, on_epoch=True)

    # ---------------- OPTIMIZER ----------------
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
