import secrets
from pytorch_lightning.trainer.states import TrainerFn

def add_tags_and_run_name_to_logger(self):
    """Add derived tags and run name to the logger if using WandB."""
    stage = self.trainer.state.stage

    stage_str = stage.value

    if not hasattr(self.trainer, "datamodule"):
        return

    datamodule = self.trainer.datamodule

    data_tags = []
    data_run_name = ""
    if hasattr(datamodule, "_get_tags_and_run_name"):
        data_tags, data_run_name = datamodule._get_tags_and_run_name()

    if hasattr(self.logger, "experiment") and self.logger.experiment is not None:
        tags, run_name = self._get_tags_and_run_name()
        if hasattr(self.trainer.logger, "_name") and not self.trainer.logger._name:
            if stage_str:
                run_name = f"{stage_str}_{run_name}"
            self.trainer.logger.experiment.name = run_name
            if data_run_name:
                self.trainer.logger.experiment.name += f"_{data_run_name}"
                    # Add unique ID suffix
            self.trainer.logger.experiment.name += f"_{secrets.randbits(24)}"
        if data_tags:
            tags = tags + data_tags
        if stage_str:
            tags.append(stage_str)

        self.trainer.logger.experiment.tags = tuple(
            set(self.trainer.logger.experiment.tags).union(set(tags))
        )
