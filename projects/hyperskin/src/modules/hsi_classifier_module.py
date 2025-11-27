# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
import inspect
from pathlib import Path
from pdb import run
from typing import Any, Optional
import numpy as np

import pytorch_lightning as pl
import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import F1Score, Precision, Recall, Specificity
from torchmetrics import SpecificityAtSensitivity
from torch.nn.functional import one_hot
import importlib

from src.data_modules.hsi_dermoscopy import HSIDermoscopyDataModule
from src.models import TIMMModel
from pytorch_lightning.loggers import WandbLogger

from src.utils.tags_and_run_name import add_tags_and_run_name_to_logger

def load_class(path: str):
    """
    Load a class dynamically given a dotted path like:
      - "losses.FocalLoss"
      - "src.losses.focal_loss.FocalLoss"

    The path can reference either:
    - a class re-exported in __init__.py
    - or a class defined in a submodule
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class HSIClassifierModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        in_chans: int = None,
        pretrained: Optional[bool] = None,
        features_only: Optional[bool] = None,
        scriptable: Optional[bool] = None,
        model_name: Optional[str] = None,
        criterion: Optional[dict] = None,
        min_sensitivity: float = 0.95,
        freeze_backbone: bool = False,
        freeze_layers: Optional[list[str]] = None,
        unfreeze_layers: Optional[list[str]] = None,
        custom_head: Optional[list[dict]] = None,
        unfreeze_norm: bool = False,
        metric_threshold: float = 0.5,

        # ISIC2019 specific arguments
        isic2019_weights_path: Optional[str] = None,
        adapt_fc: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        if self.hparams.num_classes ==2:
            self.class_task = 'binary'
        elif self.hparams.num_classes >2:
            self.class_task = 'multiclass'
        else:
            print('We should have 2 or more classes')
            return AssertionError()

        if self.hparams.isic2019_weights_path is not None:
            from src.models.isic2019.isic2019 import ISIC2019Model
            self.net = ISIC2019Model(weights_path=self.hparams.isic2019_weights_path,
                                     num_classes=self.hparams.num_classes,
                                     adapt_fc=self.hparams.adapt_fc,
                                     in_chans=self.hparams.in_chans, freeze_backbone=self.hparams.freeze_backbone)
        else:
            self.net = TIMMModel(model_name=self.hparams.model_name,
                                num_classes=self.hparams.num_classes,
                                pretrained=self.hparams.pretrained,
                                features_only=self.hparams.features_only,
                                in_chans=self.hparams.in_chans,
                                scriptable=self.hparams.scriptable,
                                freeze_backbone=self.hparams.freeze_backbone,
                                freeze_layers=self.hparams.freeze_layers,
                                unfreeze_layers=self.hparams.unfreeze_layers,
                                custom_head=self.hparams.custom_head,
                                unfreeze_norm=self.hparams.unfreeze_norm)

        if self.hparams.criterion is not None:
            if "class_path" not in self.hparams.criterion:
                raise ValueError("Criterion must have 'class_path' key")
            if "init_args" not in self.hparams.criterion or self.hparams.criterion["init_args"] is None:
                self.hparams.criterion["init_args"] = {}

            try:
                criterion_cls = load_class(self.hparams.criterion["class_path"])
            except AttributeError:
                raise ValueError(f"Criterion {self.hparams.criterion['class_path']} not found ")
            self.criterion = criterion_cls(**self.hparams.criterion["init_args"])
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches

        metric_acc = Accuracy(task=self.class_task, num_classes=self.hparams.num_classes,
        threshold=self.hparams.metric_threshold)
        self.train_acc = metric_acc.clone()
        self.val_acc = metric_acc.clone()
        self.test_acc = metric_acc.clone()

        f1_metric = F1Score(task=self.class_task, num_classes=self.hparams.num_classes,
        threshold=self.hparams.metric_threshold)
        self.train_f1 = f1_metric.clone()
        self.val_f1   = f1_metric.clone()
        self.test_f1  = f1_metric.clone()

        prec_metric = Precision(task=self.class_task, num_classes=self.hparams.num_classes,
        threshold=self.hparams.metric_threshold)
        self.val_prec   = prec_metric.clone()
        self.test_prec  = prec_metric.clone()

        rec_metric = Recall(task=self.class_task, num_classes=self.hparams.num_classes,
        threshold=self.hparams.metric_threshold)
        self.train_rec = rec_metric.clone()
        self.train_real_rec = rec_metric.clone()
        self.train_synth_rec = rec_metric.clone()
        self.val_rec   = rec_metric.clone()
        self.test_rec  = rec_metric.clone()

        spec_metric = Specificity(task=self.class_task, num_classes=self.hparams.num_classes,
                                               threshold=self.hparams.metric_threshold)
        self.train_spec = spec_metric.clone()
        self.train_real_spec = spec_metric.clone()
        self.train_synth_spec = spec_metric.clone()
        self.val_spec = spec_metric.clone()
        self.test_spec = spec_metric.clone()

        # for averaging loss across batches
        loss_metric = MeanMetric()
        self.train_loss = loss_metric.clone()
        self.val_loss = loss_metric.clone()
        self.test_loss = loss_metric.clone()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_spec_at_sens_best = MaxMetric()

        self.val_spec_at_sens = SpecificityAtSensitivity(min_sensitivity=self.hparams.min_sensitivity,
                                                         task=self.class_task,
                                                         num_classes=self.hparams.num_classes)
        self.test_spec_at_sens = SpecificityAtSensitivity(min_sensitivity=self.hparams.min_sensitivity,
                                                          task=self.class_task,
                                                          num_classes=self.hparams.num_classes)
        
        self.val_f1_best = MaxMetric()

    def _get_tags_and_run_name(self):
        """Automatically derive tags and a run name from FastGANModule hyperparameters."""
        hparams = getattr(self, "hparams", None)
        if hparams is None:
            return

        tags = []
        run_name = ""

        if self.hparams.isic2019_weights_path is None:
            run_name += hparams.model_name
            tags.append(hparams.model_name)
            # add pretrained tag
            if hparams.pretrained:
                tags.append("pretrained")
                run_name += "_pt"
            else:
                tags.append("not_pretrained")

            # add freeze_backbone tag
            if hparams.freeze_backbone:
                tags.append("frozen_backbone")
                run_name += "_fb"
        else:
            run_name += "isic2019"

            if self.hparams.freeze_backbone:
                tags.append("frozen_backbone")
                run_name += "_fb"

            if self.hparams.num_classes != 8 and not self.hparams.adapt_fc:
                tags.append("clean_fc")
                run_name += "_clean_fc"

            if self.hparams.num_classes != 8 and self.hparams.adapt_fc:
                tags.append("adapted_fc")
                run_name += "_adapted_fc"

        # add criterion tag
        if hparams.criterion is not None:
            criterion_class = hparams.criterion.get("class_path", "")
            criterion_class = "".join(["_" + c.lower() if c.isupper() else c for c in criterion_class])
            criterion_name = criterion_class.split(".")
            if len(criterion_name) > 0:
                criterion_name = criterion_name[-1].lstrip("_")
                # criterion_name is in CamelCase, convert to snake_case
                tags.append(criterion_name)
            else:
                tags.append(criterion_class.lstrip("_"))

        return tags, run_name

    def setup(self, stage: str) -> None:
        add_tags_and_run_name_to_logger(self)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_spec_at_sens_best.reset()

        # Save data splits if the logger and datamodule are configured correctly
        if self.trainer.logger and hasattr(self.trainer.logger, 'save_dir') and \
            isinstance(self.trainer.datamodule, HSIDermoscopyDataModule):
            logger = self.trainer.logger
            datamodule = self.trainer.datamodule

            # If using wandb, upload the split files
            if isinstance(logger, WandbLogger):
                split_dir = Path(logger.save_dir) / logger.name / logger.version / "splits"

                # Save splits to local log directory
                datamodule.save_splits_to_disk(split_dir)
                run = logger.experiment

                # Upload the split files to wandb
                run.save(str(split_dir / "*.txt"), base_path=str(split_dir.parent))


    def model_step(self, batch: Any):
        if isinstance(batch, dict):
            x = batch["image"]
            y = batch["label"]
        else:
            x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, logits

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.model_step(batch)

        if isinstance(batch, dict) and "synthetic_label" in batch:
            synth_labels = batch["synthetic_label"]
            # synth_labels is a batch of (batch_size,) with a number 0 or 1, indicating if the sample is real or synthetic
            # get which indices are synthetic
            synth_indices = (synth_labels == 1).nonzero(as_tuple=True)[0]
            real_indices = (synth_labels == 0).nonzero(as_tuple=True)[0]

            if len(synth_indices) > 0:
                synth_preds = preds[synth_indices]
                synth_targets = targets[synth_indices]
                real_preds = preds[real_indices]
                real_targets = targets[real_indices]

                if (synth_targets == 0).any():
                    self.train_synth_spec(synth_preds, synth_targets)
                    self.log("train/specificity_synth", self.train_synth_spec, on_step=False, on_epoch=True)

                if (real_targets == 0).any():
                    self.train_real_spec(real_preds, real_targets)
                    self.log("train/specificity_real", self.train_real_spec, on_step=False, on_epoch=True)

                if (synth_targets == 1).any():
                    self.train_synth_rec(synth_preds, synth_targets)
                    self.log("train/recall_synth", self.train_synth_rec, on_step=False, on_epoch=True)

                if (real_targets == 1).any():
                    self.train_real_rec(real_preds, real_targets)
                    self.log("train/recall_real", self.train_real_rec, on_step=False, on_epoch=True)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1 (preds, targets)
        self.train_spec(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train/specificity", self.train_spec, on_step=False, on_epoch=True)

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

    def on_validation_start(self):
        self.val_acc.threshold = self.hparams.metric_threshold
        self.val_f1.threshold = self.hparams.metric_threshold
        self.val_prec.threshold = self.hparams.metric_threshold
        self.val_rec.threshold = self.hparams.metric_threshold
        self.val_spec.threshold = self.hparams.metric_threshold

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds,targets)
        self.val_prec(preds,targets)
        self.val_rec(preds,targets)
        self.val_spec(preds,targets)
        self.val_spec_at_sens(logits, one_hot(targets, num_classes=self.hparams.num_classes))

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/prec", self.val_prec, on_step=False, on_epoch=True)
        self.log("val/rec", self.val_rec, on_step=False, on_epoch=True)
        self.log("val/specificity", self.val_spec, on_step=False, on_epoch=True)

        specificity, threshold = self.val_spec_at_sens.compute()
        self.log(f"val/spec@sens={self.hparams.min_sensitivity}", specificity, on_step=False, on_epoch=True)
        self.log(f"val/spec@sens={self.hparams.min_sensitivity}_threshold", threshold, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute())

        sens_at_spec, threshold = self.val_spec_at_sens.compute()
        self.val_spec_at_sens_best(sens_at_spec)
        self.log(f"val/spec@sens={self.hparams.min_sensitivity}_best", self.val_spec_at_sens_best.compute())
        
        self.val_f1_best(self.val_f1.compute())
        self.log("val/f1_best", self.val_f1_best.compute())

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds,targets)
        self.test_prec(preds,targets)
        self.test_rec(preds,targets)
        self.test_spec(preds,targets)
        self.test_spec_at_sens(logits, one_hot(targets, num_classes=self.hparams.num_classes))

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/prec", self.test_prec, on_step=False, on_epoch=True)
        self.log("test/rec", self.test_rec, on_step=False, on_epoch=True)
        self.log("test/sensitivity", self.test_rec, on_step=False, on_epoch=True)
        self.log("test/specificity", self.test_spec, on_step=False, on_epoch=True)

        specificity, threshold = self.test_spec_at_sens.compute()
        self.log(f"test/spec@sens={self.hparams.min_sensitivity}", specificity, on_step=False, on_epoch=True)
        self.log(f"test/spec@sens={self.hparams.min_sensitivity}_threshold", threshold, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        pass

if __name__ == "__main__":
    m = HSIClassifierModule()
    print(m)
