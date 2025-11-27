import os
import argparse
import numpy as np
import torch
import torchvision.utils as vutils
from scipy.io import loadmat
from PIL import Image

# Default normalization bounds
DEFAULT_GLOBAL_MAX = np.array([
    0.6203158, 0.6172642, 0.46794897, 0.4325111, 0.4996644, 0.61997396,
    0.7382196, 0.86097705, 0.88304037, 0.9397393, 1.1892519, 1.5035477,
    1.4947973, 1.4737314, 1.6318618, 1.7226081
])

DEFAULT_GLOBAL_MIN = np.array([
    0.00028473, 0.0043945, 0.00149752, 0.00167517, 0.00190101, 0.0028114,
    0.00394378, 0.00488099, 0.00257091, 0.00215704, 0.00797662, 0.01205248,
    0.01310135, 0.01476806, 0.01932094, 0.02020744
])


def normalize_image(image: np.ndarray, global_min, global_max):
    """Normalize each channel using global min and max, clamping values to [0, 1]."""
    norm = (image - global_min) / (global_max - global_min)
    norm = np.clip(norm, 0.0, 1.0)
    return norm


def hyperspectral_to_rgb(hsi: np.ndarray):
    """
    Convert a hyperspectral (256,256,16) cube to an RGB image
    by averaging all channels.
    """
    rgb = np.mean(hsi, axis=2)
    rgb = np.stack([rgb, rgb, rgb], axis=2)
    return rgb


def find_mat_files(root_dir):
    """Recursively find all .mat files under the given directory."""
    mat_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".mat"):
                mat_files.append(os.path.join(dirpath, f))
    return mat_files


def main(input_dir, output_path, n=50, global_min=None, global_max=None):
    global_min = global_min or DEFAULT_GLOBAL_MIN
    global_max = global_max or DEFAULT_GLOBAL_MAX

    # Recursively gather .mat files
    files = find_mat_files(input_dir)
    if not files:
        print(f"❌ No .mat files found in {input_dir}.")
        return

    files = files[:n]
    images = []

    for path in files:
        fname = os.path.basename(path)
        mat = loadmat(path)

        # Try to find proper key automatically
        key = next((k for k in mat.keys() if not k.startswith("__")), None)
        if key is None:
            print(f"⚠️ Skipping {fname}, no valid data key found.")
            continue

        img = mat[key]
        if img.ndim != 3 or img.shape[-1] != 16:
            print(f"⚠️ Skipping {fname}, invalid shape {img.shape}.")
            continue

        norm_img = normalize_image(img, global_min, global_max)
        rgb_img = hyperspectral_to_rgb(norm_img)

        tensor_img = torch.tensor(rgb_img.transpose(2, 0, 1), dtype=torch.float32)
        images.append(tensor_img)

    if not images:
        print("❌ No valid images found.")
        return

    grid = vutils.make_grid(images, nrow=10, normalize=True)
    vutils.save_image(grid, output_path)
    print(f"✅ Saved image grid to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize and visualize hyperspectral images (recursive search)"
    )
    parser.add_argument("input_dir", type=str, help="Directory with .mat files (searched recursively)")
    parser.add_argument("--output_path", type=str, default="hsi_grid.png", help="Output PNG path")
    parser.add_argument("--n", type=int, default=50, help="Number of images to include in grid")
    args = parser.parse_args()

    main(args.input_dir, args.output_path, args.n)
