from collections.abc import Sequence
import torch
from torchmetrics import Metric
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt


class MeanSpectraMetric(Metric):
    """Compute and store mean spectral signatures for real/fake hyperspectral batches.

    Supports:
    - optional lesion mask input for precise lesion/normal separation.
    - optional binary labels (0/1) specifying lesion type per image.
      Labels indicate lesion class at image level; masks indicate lesion pixels.
    """

    full_state_update = False  # update() called each step, not aggregated by DDP

    def __init__(
        self,
        lesion_class_name: Sequence[str] | None = None,
    ):
        """
        Args:
            lesion_class_name (Optional[Sequence[str]]):
                - If None:
                    Default lesion class names are ["lesion_0", "lesion_1"].
                - If a sequence with 1 element:
                    Used as name for lesion class 1; lesion class 0 defaults to "lesion_0".
                - If a sequence with 2 elements:
                    lesion_class_name[0] -> name for label 0 lesion class
                    lesion_class_name[1] -> name for label 1 lesion class

            Note:
                If labels are passed to update(), this metric supports only binary
                labels {0, 1}. When labels are provided AND custom lesion_class_name
                is specified, it must have exactly 2 names.
        """
        super().__init__()

        # Normalize lesion class names configuration
        if lesion_class_name is None:
            self.lesion_class_names: list[str] = ["lesion_0", "lesion_1"]
        else:
            lesion_class_name = list(lesion_class_name)
            if len(lesion_class_name) == 1:
                # Single custom name -> treat as label-1, keep default for label-0
                self.lesion_class_names = ["lesion_0", lesion_class_name[0]]
            elif len(lesion_class_name) == 2:
                self.lesion_class_names = [
                    str(lesion_class_name[0]),
                    str(lesion_class_name[1]),
                ]
            else:
                raise ValueError(
                    "lesion_class_name must be None, a single name, "
                    "or a sequence of exactly 2 names."
                )

        self.add_state("real_spectra", default=[], dist_reduce_fx=None)
        self.add_state("fake_spectra", default=[], dist_reduce_fx=None)

    def update(
        self,
        batch: torch.Tensor,
        is_fake: bool,
        masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """Update the stored mean spectra from a hyperspectral batch.

        Args:
            batch (torch.Tensor): Hyperspectral batch, shape (B, C, H, W).
            is_fake (bool): If True, updates fake_spectra; otherwise real_spectra.
            masks (Optional[torch.Tensor]):
                Binary lesion masks of shape (B, 1, H, W).
                If provided, used to extract lesion vs normal pixels.
            labels (Optional[torch.Tensor]):
                Tensor of shape (B,) with values in {0, 1}, indicating lesion type
                for each image. This is an image-level class, not mutually exclusive
                with masks:
                  - masks -> where lesion is
                  - labels -> what lesion class (0 or 1) that lesion is
                Rules:
                  - Only binary labels {0, 1} are supported.
                  - If provided, and custom lesion_class_names are used, there must
                    be exactly 2 class names.
        """
        spectra_list = self.fake_spectra if is_fake else self.real_spectra

        if labels is not None:
            if labels.dim() != 1:
                raise ValueError("labels must have shape (B,) when provided.")
            unique_vals = torch.unique(labels)
            if not torch.all((unique_vals == 0) | (unique_vals == 1)):
                raise ValueError(
                    "Only binary labels {0, 1} are supported for this metric."
                )
            # Enforce exactly 2 names when labels are used
            if len(self.lesion_class_names) != 2:
                raise ValueError(
                    "When labels are provided, lesion_class_name must define "
                    "exactly 2 class names."
                )

        img_np = batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
        mask_np = (
            masks.detach().cpu().numpy()[:, 0, :, :]
            if masks is not None
            else None
        )
        labels_np = labels.detach().cpu().numpy() if labels is not None else None

        for idx, image_np in enumerate(img_np):
            mean_image = image_np.mean(axis=-1)
            try:
                # Determine lesion mask
                if mask_np is not None:
                    lesion_mask = mask_np[idx].astype(bool)
                else:
                    # Fallback: Otsu on mean image
                    otsu_thresh = filters.threshold_otsu(mean_image)
                    lesion_mask = mean_image < (otsu_thresh * 1.0)

                # Normal skin: outside lesion mask
                normal_mask = ~lesion_mask

                # Compute spectra for normal skin if exists
                if np.any(normal_mask):
                    normal_spec = image_np[normal_mask].mean(axis=0)
                    spectra_list.append(("normal_skin", normal_spec))

                # If there is no lesion area, nothing else to log for this image
                if not np.any(lesion_mask):
                    continue

                # If labels are provided: choose lesion class name per image
                if labels_np is not None:
                    lbl_val = int(labels_np[idx])
                    lesion_label_name = self.lesion_class_names[lbl_val]
                    lesion_spec = image_np[lesion_mask].mean(axis=0)
                    spectra_list.append((lesion_label_name, lesion_spec))
                else:
                    # No labels: keep original generic lesion behavior
                    lesion_spec = image_np[lesion_mask].mean(axis=0)
                    # Use first lesion class name as the generic one
                    spectra_list.append((self.lesion_class_names[0], lesion_spec))

            except Exception:
                # Robust to occasional issues; skip problematic samples
                continue

    def compute(self) -> dict[str, dict[str, np.ndarray]]:
        """Compute mean and std spectra for all collected samples.

        Returns:
            Dict with structure:
            {
                "real": {
                    "<label>": {"mean": np.ndarray, "std": np.ndarray},
                    ...
                },
                "fake": {
                    "<label>": {"mean": np.ndarray, "std": np.ndarray},
                    ...
                },
            }
        """
        result = {"real": {}, "fake": {}}

        for name, spectra_list in [
            ("real", self.real_spectra),
            ("fake", self.fake_spectra),
        ]:
            per_label = {}
            for lbl, spec in spectra_list:
                per_label.setdefault(lbl, []).append(spec)

            for lbl, arr_list in per_label.items():
                arr = np.stack(arr_list, axis=0)
                result[name][lbl] = {
                    "mean": np.mean(arr, axis=0),
                    "std": np.std(arr, axis=0),
                }

        return result

    def plot(self) -> plt.Figure | None:
        """Return a matplotlib figure comparing real vs synthetic mean spectra.

        - Always considers:
            - normal_skin
            - lesion_class_names[0]
            - lesion_class_names[1] (if present in data)
        - Adds a third subplot for the second lesion class when available.
        """
        stats = self.compute()
        real_stats = stats["real"]
        fake_stats = stats["fake"]

        # Define the order of labels to visualize
        labels = ["normal_skin"] + self.lesion_class_names

        # Determine if we have any data at all
        ref_stats = real_stats or fake_stats
        if not ref_stats:
            return None

        # Determine number of spectral bands from any available label
        ref_label = None
        for lbl in labels:
            if lbl in ref_stats:
                ref_label = lbl
                break
        if ref_label is None:
            return None

        n_bands = len(ref_stats[ref_label]["mean"])
        bands = np.arange(1, n_bands + 1)

        # Create subplots: one per label (including potentially 3rd lesion)
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
        if len(labels) == 1:
            axes = [axes]

        for ax, lbl in zip(axes, labels):
            rs = real_stats.get(lbl)
            fs = fake_stats.get(lbl)

            if rs is None and fs is None:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {lbl}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            if rs is not None:
                ax.plot(bands, rs["mean"], "C0-", lw=2.5, label="Real")
                ax.fill_between(
                    bands,
                    rs["mean"] - rs["std"],
                    rs["mean"] + rs["std"],
                    color="C0",
                    alpha=0.15,
                )

            if fs is not None:
                ax.plot(bands, fs["mean"], "C3--", lw=2.0, label="Synthetic")
                ax.fill_between(
                    bands,
                    fs["mean"] - fs["std"],
                    fs["mean"] + fs["std"],
                    color="C3",
                    alpha=0.15,
                )

            ax.set_title(lbl.replace("_", " ").title())
            ax.set_xlabel("Spectral Band")
            ax.set_ylabel("Reflectance")
            ax.legend()
            ax.grid(alpha=0.25)

        plt.suptitle(
            "Mean Spectra Comparison: Real vs Synthetic",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        return fig

    def reset(self):
        """Reset stored spectra."""
        self.real_spectra = []
        self.fake_spectra = []
