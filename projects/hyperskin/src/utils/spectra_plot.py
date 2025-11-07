import torch
from torchmetrics import Metric
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
from typing import Optional


class MeanSpectraMetric(Metric):
    """Compute and store mean spectral signatures for real/fake hyperspectral batches.

    Supports optional lesion mask input for precise lesion/normal separation.
    """

    full_state_update = False  # update() called each step, not aggregated by DDP

    def __init__(self, lesion_class_name: Optional[str] = None):
        """
        Args:
            lesion_class_name (Optional[str]): name of the lesion class.
                If None, "lesion" is used as default label.
        """
        super().__init__()
        self.lesion_class_name = lesion_class_name or "lesion"

        self.add_state("real_spectra", default=[], dist_reduce_fx=None)
        self.add_state("fake_spectra", default=[], dist_reduce_fx=None)

    def update(
        self,
        batch: torch.Tensor,
        is_fake: bool,
        masks: Optional[torch.Tensor] = None,
    ):
        """Update the stored mean spectra from a hyperspectral batch.

        Args:
            batch (torch.Tensor): Hyperspectral batch, shape (B, C, H, W)
            is_fake (bool): If True, updates fake_spectra; otherwise real_spectra
            masks (Optional[torch.Tensor]): Binary lesion masks of shape (B, 1, H, W)
                If provided, used to extract lesion and normal pixels.
        """
        spectra_list = self.fake_spectra if is_fake else self.real_spectra

        img_np = batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
        mask_np = (
            masks.detach().cpu().numpy()[:, 0, :, :]
            if masks is not None
            else None
        )

        for idx, image_np in enumerate(img_np):
            mean_image = image_np.mean(axis=-1)
            try:
                # lesion mask selection
                if mask_np is not None:
                    binary_mask = mask_np[idx].astype(bool)
                else:
                    otsu_thresh = filters.threshold_otsu(mean_image)
                    binary_mask = mean_image < (otsu_thresh * 1.0)

                if np.any(binary_mask):
                    lesion_spec = image_np[binary_mask].mean(axis=0)
                    spectra_list.append((self.lesion_class_name, lesion_spec))

                normal_mask = ~binary_mask
                if np.any(normal_mask):
                    normal_spec = image_np[normal_mask].mean(axis=0)
                    spectra_list.append(("normal_skin", normal_spec))

            except Exception:
                continue

    def compute(self) -> dict[str, dict[str, np.ndarray]]:
        """Compute mean and std spectra for all collected samples."""
        result = {"real": {}, "fake": {}}

        for name, spectra_list in [("real", self.real_spectra),
                                   ("fake", self.fake_spectra)]:
            per_label: dict[str, list[np.ndarray]] = {}
            for lbl, spec in spectra_list:
                per_label.setdefault(lbl, []).append(spec)

            for lbl, arr_list in per_label.items():
                arr = np.stack(arr_list, axis=0)
                result[name][lbl] = {
                    "mean": np.mean(arr, axis=0),
                    "std": np.std(arr, axis=0),
                }
        return result

    def plot(self) -> Optional[plt.Figure]:
        """Return a matplotlib figure comparing real vs synthetic mean spectra."""
        stats = self.compute()
        real_stats = stats["real"]
        fake_stats = stats["fake"]

        labels = ["normal_skin", self.lesion_class_name]
        ref_stats = real_stats or fake_stats
        ref_label = next((lbl for lbl in labels if lbl in ref_stats), None)
        if ref_label is None:
            return None

        n_bands = len(ref_stats[ref_label]["mean"])
        bands = np.arange(1, n_bands + 1)

        fig, axes = plt.subplots(1, len(labels), figsize=(15, 5))
        if len(labels) == 1:
            axes = [axes]

        for ax, lbl in zip(axes, labels):
            rs = real_stats.get(lbl)
            fs = fake_stats.get(lbl)

            if rs is None and fs is None:
                ax.text(
                    0.5, 0.5, f"No data for {lbl}",
                    ha="center", va="center", transform=ax.transAxes
                )
                continue

            if rs is not None:
                ax.plot(bands, rs["mean"], "C0-", lw=2.5, label="Real")
                ax.fill_between(
                    bands, rs["mean"] - rs["std"],
                    rs["mean"] + rs["std"], color="C0", alpha=0.15
                )

            if fs is not None:
                ax.plot(bands, fs["mean"], "C3--", lw=2.0, label="Synthetic")
                ax.fill_between(
                    bands, fs["mean"] - fs["std"],
                    fs["mean"] + fs["std"], color="C3", alpha=0.15
                )

            ax.set_title(lbl.replace("_", " ").title())
            ax.set_xlabel("Spectral Band")
            ax.set_ylabel("Reflectance")
            ax.legend()
            ax.grid(alpha=0.25)

        plt.suptitle(
            "Mean Spectra Comparison: Real vs Synthetic", fontsize=14, y=1.02
        )
        plt.tight_layout()
        return fig

    def reset(self):
        """Reset stored spectra."""
        self.real_spectra = []
        self.fake_spectra = []
