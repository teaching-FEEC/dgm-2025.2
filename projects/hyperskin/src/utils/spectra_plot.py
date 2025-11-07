import numpy as np
from skimage import filters

def compute_spectra_statistics(real_norm, fake_norm, real_spectra, fake_spectra, lesion_class_name):
    """Extract spectra from lesion regions using Otsu thresholding"""
    for img, spectra_dict in [
        (real_norm, real_spectra),
        (fake_norm, fake_spectra),
    ]:
        for b in range(img.size(0)):
            image_np = img[b].cpu().numpy().transpose(1, 2, 0)
            mean_image = image_np.mean(axis=-1)

            try:
                otsu_thresh = filters.threshold_otsu(mean_image)
                binary_mask = mean_image < (otsu_thresh * 1)

                if np.any(binary_mask):
                    spectrum = image_np[binary_mask].mean(axis=0)
                    spectra_dict[lesion_class_name].append(spectrum)

                normal_skin_mask = ~binary_mask
                if np.any(normal_skin_mask):
                    normal_spectrum = image_np[normal_skin_mask].mean(
                        axis=0
                    )
                    spectra_dict["normal_skin"].append(normal_spectrum)
            except Exception:
                continue

def plot_mean_spectra(lesion_class_name, real_spectra, fake_spectra):
    """Plot mean spectra comparing real vs synthetic data"""
    import matplotlib.pyplot as plt

    labels = ["normal_skin", lesion_class_name]

    real_stats = {}
    fake_stats = {}

    for label_name in labels:
        if real_spectra.get(label_name):
            arr = np.array(real_spectra[label_name])
            real_stats[label_name] = {
                "mean": np.mean(arr, axis=0),
                "std": np.std(arr, axis=0),
            }

        if fake_spectra.get(label_name):
            arr = np.array(fake_spectra[label_name])
            fake_stats[label_name] = {
                "mean": np.mean(arr, axis=0),
                "std": np.std(arr, axis=0),
            }

    ref_stats = real_stats or fake_stats
    ref_label = next(
        (lbl for lbl in labels if ref_stats.get(lbl) is not None), None
    )
    if ref_label is None:
        return

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
                0.5,
                0.5,
                f"No data for {lbl}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        if rs is not None:
            ax.plot(
                bands,
                rs["mean"],
                linestyle="-",
                linewidth=2.5,
                color="C0",
                label="Real",
            )
            ax.fill_between(
                bands,
                rs["mean"] - rs["std"],
                rs["mean"] + rs["std"],
                color="C0",
                alpha=0.15,
            )

        if fs is not None:
            ax.plot(
                bands,
                fs["mean"],
                linestyle="--",
                linewidth=2.0,
                color="C3",
                label="Synthetic",
            )
            ax.fill_between(
                bands,
                fs["mean"] - fs["std"],
                fs["mean"] + fs["std"],
                color="C3",
                alpha=0.15,
            )

        ax.set_title(f"{lbl.replace('_', ' ').title()}")
        ax.set_xlabel("Spectral Band")
        ax.set_ylabel("Reflectance")
        ax.legend()
        ax.grid(True, alpha=0.25)

    plt.suptitle(
        "Mean Spectra Comparison: Real vs Synthetic",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    return fig
