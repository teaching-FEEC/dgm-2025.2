import os
from pathlib import Path
import sys
import warnings
from typing import Any, Optional
import secrets
from xml.parsers.expat import model
from lightning_fabric.utilities.cloud_io import get_filesystem
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import (
    LightningArgumentParser,
    LightningCLI,
    LRSchedulerTypeUnion,
    ReduceLROnPlateau,
    SaveConfigCallback,
)
from pytorch_lightning.loggers import Logger, WandbLogger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR

import torch
from src.data_modules.datasets.hsi_dermoscopy_dataset import HSIDermoscopyTask
torch.serialization.add_safe_globals([HSIDermoscopyTask])


def safe_parse_ckpt_path(self):
    """Same as original, but silently continue on parse failure."""
    if not self.config.get("subcommand"):
        return
    ckpt_path = self.config[self.config.subcommand].get("ckpt_path")
    if ckpt_path and Path(ckpt_path).is_file():
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        hparams = ckpt.get("hyper_parameters", {})
        hparams.pop("_instantiator", None)
        if not hparams:
            return
        hparams = {self.config.subcommand: {"model": hparams}}
        try:
            self.config = self.parser.parse_object(hparams, self.config)
        except SystemExit:
            sys.stderr.write("Warning: Failed to parse ckpt_path hyperparameters. Continuing as-is.\n")
            # just continue instead of raising

class WandbSaveConfigCallback(SaveConfigCallback):

    def _build_run_name_and_tags(self) -> tuple[str, list[str]]:
        """Constructs a run name and tags from config."""
        run_name = ""
        tags: list[str] = []

        # Handle data-related tags
        if hasattr(self.config.data, "init_args"):
            data_args = self.config.data.init_args

            # Crop
            if hasattr(data_args, "data_dir") and "crop" in data_args.data_dir.lower():
                run_name += "crop_"
                tags.append("cropped")

            if hasattr(data_args, "balanced_sampling") and data_args.balanced_sampling:
                tags.append("balanced_sampling")

            if hasattr(data_args, "infinite_train") and data_args.infinite_train:
                tags.append("infinite_train")

            if hasattr(data_args, "allowed_labels") and data_args.allowed_labels:
                labels = data_args.allowed_labels
                if isinstance(labels, list):
                    for label in labels:
                        tags.append(label.lower())
                elif isinstance(labels, str):
                    tags.append(labels.lower())

            # Task
            if hasattr(data_args, "task"):
                task = data_args.task.lower()
                if "classification" in task:
                    run_name += "cls_"
                    tags.append("classification")
                elif "segmentation" in task:
                    run_name += "seg_"
                    tags.append("segmentation")
                elif "generation" in task:
                    run_name += "gen_"
                    tags.append("generation")

                if run_name:
                    first_underscore_index = task.find("_")
                    if first_underscore_index != -1:
                        tags.append(task[first_underscore_index + 1 :])

            # Transforms (Detect Augmentation)
            if hasattr(data_args, "transforms") and "train" in data_args.transforms:
                transforms = data_args.transforms["train"]
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

            # Synthetic data
            if hasattr(data_args, "synthetic_data_dir") and data_args.synthetic_data_dir is not None:
                run_name += "synth_"
                tags.append("synthetic_data")

        if hasattr(self.config, "model") and self.config.model is not None and \
                "class_path" in self.config.model and "fastgan" in self.config.model["class_path"].lower():
            run_name += "fastgan_"
            tags.append("fastgan")

        if hasattr(self.config, "model") and self.config.model is not None and \
                "class_path" in self.config.model and "vae" in self.config.model["class_path"].lower():
            run_name += "VAE_"
            tags.append("VAE")

        # Handle model-related tags
        if hasattr(self.config.model, "init_args"):
            model_args = self.config.model.init_args

            if hasattr(model_args, "model_name"):
                run_name += f"{model_args.model_name}_"
                tags.append(model_args.model_name.lower())

            if hasattr(model_args, "arch_name"):
                run_name += f"{model_args.arch_name}_"
                tags.append(model_args.arch_name.lower())

            if hasattr(model_args, "encoder_name"):
                run_name += f"{model_args.encoder_name}_"
                tags.append(model_args.encoder_name.lower())

            if hasattr(model_args, "pretrained"):
                if model_args.pretrained:
                    run_name += "pt_"
                    tags.append("pretrained")
                else:
                    tags.append("not_pretrained")

            if hasattr(model_args, "in_chans"):
                in_chans = model_args.in_chans
                img_type = "rgb" if in_chans == 3 else "hsi"
                run_name += f"{img_type}{in_chans}_"
                tags.extend([img_type, f"{in_chans}_channels"])

            if hasattr(model_args, "freeze_backbone"):
                if model_args.freeze_backbone:
                    run_name += "fb_"
                    tags.append("frozen_backbone")

            if hasattr(model_args, "nz"):
                nz = model_args.nz
                tags.append(f"z{nz}")

        # Optimizer
        if hasattr(self.config, "optimizer") and self.config.optimizer is not None and \
                "class_path" in self.config.optimizer:
            optimizer = self.config.optimizer["class_path"]
            # split by . and take last part
            optimizer = "".join(["_" + c.lower() if c.isupper() else c for c in optimizer])
            optimizer_name = optimizer.split(".")
            if len(optimizer_name) > 0:
                optimizer_name = optimizer_name[-1].lstrip("_")
                # optimizer_name is in CamelCase, convert to snake_case
                tags.append(optimizer_name)
            else:
                tags.append(optimizer.lstrip("_"))

            if hasattr(model_args, "criterion") and model_args.criterion is not None and \
                "class_path" in model_args.criterion:
                criterion = model_args.criterion["class_path"]
                # split by . and take last part
                criterion = "".join(["_" + c.lower() if c.isupper() else c for c in criterion])
                criterion_name = criterion.split(".")
                if len(criterion_name) > 0:
                    criterion_name = criterion_name[-1].lstrip("_")
                    # criterion_name is in CamelCase, convert to snake_case
                    tags.append(criterion_name)
                else:
                    tags.append(criterion.lstrip("_"))
        # Add unique ID suffix
        run_name += f"{secrets.randbits(24)}"

        return run_name, tags

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        # run_name, tags = self._build_run_name_and_tags()

        # apply the name and tags to all loggers
        # for _logger in trainer.loggers:
        #     if isinstance(_logger, WandbLogger):
        #         # only set the name if it hasn't been manually by the user
        #         if hasattr(trainer.logger, "_name") and not trainer.logger._name:
        #             _logger.experiment.name = run_name
        #         _logger.experiment.tags = tuple(
        #             set(_logger.experiment.tags).union(set(tags))
        #         )

        log_dir = trainer.log_dir  # this broadcasts the directory
        if trainer.logger is not None and trainer.logger.name is not None and trainer.logger.version is not None:
            log_dir = os.path.join(log_dir, trainer.logger.name, str(trainer.logger.version))
        config_path = os.path.join(log_dir, self.config_filename)
        fs = get_filesystem(log_dir)

        if not self.overwrite:
            # check if the file exists on rank 0
            file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
            # broadcast whether to fail to all ranks
            file_exists = trainer.strategy.broadcast(file_exists)
            if file_exists:
                raise RuntimeError(
                    f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                    " results of a previous run. You can delete the previous config file,"
                    " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                    ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                )

        # save the file on rank 0
        if trainer.is_global_zero:
            # save only on rank zero to avoid race conditions.
            # the `log_dir` needs to be created as we rely on the logger to do it usually
            # but it hasn't logged anything at this point
            fs.makedirs(log_dir, exist_ok=True)
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            self.already_saved = True
            # save optimizer and lr scheduler config
            for _logger in trainer.loggers:
                if isinstance(_logger, Logger):
                    config = {}
                    if "optimizer" in self.config and self.config["optimizer"] is not None \
                        and self.config["optimizer"] != {}:
                        config["optimizer"] = {
                            k.replace("init_args.", ""): v for k, v in dict(self.config["optimizer"]).items()
                        }
                    if "lr_scheduler" in self.config and self.config["lr_scheduler"] is not None and \
                        self.config["lr_scheduler"] != {}:
                        config["lr_scheduler"] = {
                            k.replace("init_args.", ""): v for k, v in dict(self.config["lr_scheduler"]).items()
                        }
                    _logger.log_hyperparams(config)

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

class CustomLightningCLI(LightningCLI):
    def __init__(
        self,
        save_config_callback: Optional[type[SaveConfigCallback]] = None,
        parser_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        new_parser_kwargs = {
            sub_command: dict(default_config_files=[os.path.join("configs", "default.yaml")])
            for sub_command in ["fit", "validate", "test", "predict"]
        }
        new_parser_kwargs.update(parser_kwargs or {})

        if os.environ.get("WANDB_MODE", "online") != "disabled" and save_config_callback is None:
                save_config_callback = WandbSaveConfigCallback

        original = LightningCLI._parse_ckpt_path
        LightningCLI._parse_ckpt_path = safe_parse_ckpt_path
        try:
            super().__init__(save_config_callback=save_config_callback, parser_kwargs=new_parser_kwargs, **kwargs)
        finally:
            LightningCLI._parse_ckpt_path = original

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--ignore_warnings", default=False, type=bool, help="Ignore warnings")
        parser.add_argument("--git_commit_before_fit", default=False, type=bool, help="Git commit before training")
        parser.add_argument(
            "--test_after_fit", default=False, type=bool, help="Run test on the best checkpoint after training"
        )

    def before_instantiate_classes(self) -> None:
        if self.config[self.subcommand].get("ignore_warnings"):
            warnings.filterwarnings("ignore")

    def before_fit(self) -> None:
        if self.config.fit.get("git_commit_before_fit") and not os.environ.get("DEBUG", False):
            logger = self.trainer.logger
            if isinstance(logger, WandbLogger):
                version = getattr(logger, "version")
                name = getattr(logger, "_name")
                message = "Commit Message"
                if name and version:
                    message = f"{name}_{version}"
                elif name:
                    message = name
                elif version:
                    message = version
                os.system(f'git commit -am "{message}"')

    def after_fit(self) -> None:
        if self.config.fit.get("test_after_fit") and not os.environ.get("DEBUG", False):
            self._run_subcommand("test")

    def before_test(self) -> None:
        if self.trainer.checkpoint_callback and self.trainer.checkpoint_callback.best_model_path:
            tested_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        elif self.config_init[self.config_init["subcommand"]]["ckpt_path"]:
            return
        else:
            tested_ckpt_path = None
        self.config_init[self.config_init["subcommand"]]["ckpt_path"] = tested_ckpt_path

    def _prepare_subcommand_kwargs(self, subcommand: str) -> dict[str, Any]:
        """Prepares the keyword arguments to pass to the subcommand to run."""
        fn_kwargs = {
            k: v
            for k, v in self.config_init[self.config_init["subcommand"]].items()
            if k in self._subcommand_method_arguments[subcommand]
        }
        fn_kwargs["model"] = self.model
        if self.datamodule is not None:
            fn_kwargs["datamodule"] = self.datamodule
        return fn_kwargs

    @staticmethod
    def configure_optimizers(
        lightning_module: LightningModule, optimizer: Optimizer, lr_scheduler: Optional[LRSchedulerTypeUnion] = None
    ) -> Any:
        """Override to customize the :meth:`~pytorch_lightning.core.LightningModule.configure_optimizers` method.

        Args:
            lightning_module: A reference to the model.
            optimizer: The optimizer.
            lr_scheduler: The learning rate scheduler (if used).

        """
        if lr_scheduler is None:
            return optimizer
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "monitor": lr_scheduler.monitor},
            }
        if isinstance(lr_scheduler, (OneCycleLR, CyclicLR)):
            # CyclicLR and OneCycleLR are step-based schedulers, where the default interval is "epoch".
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}
        return [optimizer], [lr_scheduler]
