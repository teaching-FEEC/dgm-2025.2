# src/modules/generative/base_predictor.py

import torch
import os
import numpy as np
from tqdm import tqdm
from scipy.io import savemat
from PIL import Image
from typing import Optional


class BasePredictorMixin:
    """
    A pure mixin (no Lightning inheritance!) for shared predict_step logic.
    Safe to combine with LightningModule subclasses.
    """

    def _prepare_global_minmax(self):
        """Prepare min/max tensors for scale restoration."""
        gmin = getattr(self.hparams, "pred_global_min", None)
        gmax = getattr(self.hparams, "pred_global_max", None)

        if gmin is not None and gmax is not None:
            gmin = torch.tensor(gmin, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
            gmax = torch.tensor(gmax, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)

        return gmin, gmax

    def _save_hyperspectral_sample(self, fake_img, batch_idx, i, gmin, gmax, output_dir):
        """Save hyperspectral (multi-channel) outputs as .mat files."""
        fake_denorm = fake_img.add(1).div(2).clamp(0, 1)
        if gmin is not None and gmax is not None:
            fake_denorm = fake_denorm * (gmax - gmin) + gmin

        fake_np = fake_denorm.squeeze().cpu().numpy().transpose(1, 2, 0)
        os.makedirs(output_dir, exist_ok=True)
        savemat(os.path.join(output_dir, f"sample_{batch_idx}_{i:04d}.mat"), {"cube": fake_np})

    def _save_rgb_sample(self, fake_img, batch_idx, i, output_dir):
        """Save RGB version visualization"""
        fake_rgb = fake_img.add(1).div(2).clamp(0, 1)
        mean_band = fake_rgb.mean(dim=1, keepdim=True)
        rgb = mean_band.repeat(1, 3, 1, 1).clamp(0, 1)
        rgb_uint8 = (rgb.cpu().squeeze().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(rgb_uint8).save(os.path.join(output_dir, f"sample_{batch_idx}_{i:04d}.png"))

    def _save_generated_batch(self, fake_batch, batch_idx, pred_hyperspectral=True, output_dir=None):
        """Generic routine to iterate and save generated images."""
        output_dir = output_dir or getattr(self.hparams, "pred_output_dir", "generated_samples")
        gmin, gmax = self._prepare_global_minmax()

        for i in tqdm(range(fake_batch.size(0)), desc="Saving generated samples"):
            fake_img = fake_batch[i : i + 1]
            if pred_hyperspectral:
                self._save_hyperspectral_sample(fake_img, batch_idx, i, gmin, gmax, output_dir)
            else:
                self._save_rgb_sample(fake_img, batch_idx, i, output_dir)

    def on_load_checkpoint(self, checkpoint):
        """
        Robust checkpoint loader:
        - restore avg_param_G if present
        - attempt to load state_dict non-strictly, removing keys that don't match this model
          (e.g. metric/net additions like 'mifid.*' that were removed from the codebase)
        """
        # restore EMA params if present
        if "avg_param_G" in checkpoint:
            self.avg_param_G = [p.to(self.device) for p in checkpoint["avg_param_G"]]
        else:
            print("Warning: avg_param_G not found in checkpoint.")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        **kwargs,
    ):
        try:
            model = super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Attempting to load with strict=False")
            model = super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=False,
                **kwargs,
            )

        return model

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """
        Robust state_dict loader that automatically retries with strict=False
        in case of missing or unexpected keys.

        Can override LightningModule.load_state_dict() safely,
        as long as this mixin is listed before LightningModule in the subclass MRO.
        """
        try:
            # Note: use super() to climb the MRO to LightningModule.load_state_dict
            super().load_state_dict(state_dict, strict)
        except Exception as e:
            # Optionally log the reason
            print(f"[BasePredictorMixin] Error loading state_dict: {e}")
            print("[BasePredictorMixin] Retrying with strict=False")
            super().load_state_dict(state_dict, strict=False)
