from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import os
from tqdm import tqdm

class FixedModelCheckpoint(ModelCheckpoint):
    def __init__(self, perform_prediction: bool = False, *args, **kwargs):
        """
        Args:
            perform_prediction: If True, runs the prediction loop and saves outputs 
                                whenever a checkpoint is saved.
            *args, **kwargs: Standard arguments for ModelCheckpoint.
        """
        self.perform_prediction = perform_prediction
        # Track the last step we ran prediction for to prevent double runs (Best + Last)
        self._last_prediction_step = -1
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        """Override save to fix '_class_path' bug AND optionally trigger prediction."""
        
        # ------------------------------------------------------------------
        # 1. ORIGINAL FIX: Fix hyper_parameters structure
        # ------------------------------------------------------------------
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only=self.save_weights_only)

        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            if "_class_path" in hparams:
                hparams["class_path"] = hparams.pop("_class_path")

            if len(hparams) > 2:
                init_args = {k: v for k, v in hparams.items() if k not in ["class_path", "_instantiator"]}
                new_hparams = {
                    "class_path": hparams["class_path"],
                    "init_args": init_args,
                    "_instantiator": hparams["_instantiator"],
                }
                checkpoint["hyper_parameters"] = new_hparams

        # ------------------------------------------------------------------
        # 2. SAVE TO DISK
        # ------------------------------------------------------------------
        trainer.save_checkpoint(filepath, weights_only=self.save_weights_only)
        torch.save(checkpoint, filepath)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # Notify loggers as usual
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(self)
                
            # ------------------------------------------------------------------
            # 3. OPTIONAL: RUN PREDICTION ON SAVE
            # ------------------------------------------------------------------
            if self.perform_prediction:
                self._run_prediction_on_save(trainer)

    def _run_prediction_on_save(self, trainer):
        """Manually runs the prediction loop on the current model state."""
        
        # --- PREVENT DUPLICATE RUNS ---
        # If we already ran prediction for this specific global step, skip it.
        # This handles the case where PL saves 'best' and 'last' checkpoints back-to-back.
        if trainer.global_step == self._last_prediction_step:
            return

        self._last_prediction_step = trainer.global_step
        # ------------------------------

        pl_module = trainer.lightning_module
        
        # 1. Get current metric value for folder naming
        metric_name = self.monitor
        metric_value = 0.0
        if metric_name is not None and metric_name in trainer.callback_metrics:
            metric_value = trainer.callback_metrics[metric_name].item()
            
        current_step = trainer.global_step
        
        # 2. Construct new output directory
        base_pred_dir = getattr(pl_module.hparams, "pred_output_dir", "generated_samples")
        clean_metric_name = metric_name.replace("/", "_") if metric_name else "metric"
        sub_dir_name = f"step_{current_step}_{clean_metric_name}_{metric_value:.4f}"
        new_save_dir = os.path.join(base_pred_dir, sub_dir_name)
        
        os.makedirs(new_save_dir, exist_ok=True)
        print(f"\n[FixedModelCheckpoint] Saving predictions to: {new_save_dir}")

        # 3. Temporarily swap the output directory in hparams
        original_pred_dir = pl_module.hparams.pred_output_dir
        pl_module.hparams.pred_output_dir = new_save_dir
        
        # 4. Prepare for inference
        pl_module.eval()
        
        try:
            dm = trainer.datamodule
            if dm is None:
                print("[FixedModelCheckpoint] Warning: No datamodule found, skipping prediction.")
                pl_module.hparams.pred_output_dir = original_pred_dir
                pl_module.train()
                return
                
            predict_loader = dm.predict_dataloader()
            if isinstance(predict_loader, list):
                predict_loader = predict_loader[0]
                
        except Exception as e:
            print(f"[FixedModelCheckpoint] Error getting predict_dataloader: {e}")
            pl_module.hparams.pred_output_dir = original_pred_dir
            pl_module.train()
            return

        # 5. Run manual prediction loop
        with torch.no_grad():
            for i, batch in enumerate(tqdm(predict_loader, desc="Checkpoint Prediction")):
                batch = trainer.strategy.batch_to_device(batch, pl_module.device)
                pl_module.predict_step(batch, i)

        # 6. Cleanup: Restore original state
        pl_module.hparams.pred_output_dir = original_pred_dir
        pl_module.train()
        print(f"[FixedModelCheckpoint] Prediction complete.")