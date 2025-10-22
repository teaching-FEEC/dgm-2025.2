#!/usr/bin/env python3
"""
Crop images based on binary masks with an optional margin multiplier.

Usage:
    python crop_with_masks.py \
        --images_dir path/to/images \
        --masks_dir path/to/masks \
        --output_dir path/to/output \
        --margin 1.5
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from PIL import Image
import albumentations as A

def get_bbox_from_mask(mask: np.ndarray):
    """Compute the bounding box (ymin, xmin, ymax, xmax) of a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return ys.min(), xs.min(), ys.max(), xs.max()


def expand_bbox(bbox, shape, margin=1.5):
    """Expand the bounding box by a given multiplier margin within image bounds."""
    ymin, xmin, ymax, xmax = bbox
    h, w = shape[:2]

    cy, cx = (ymin + ymax) / 2, (xmin + xmax) / 2
    box_h, box_w = ymax - ymin, xmax - xmin

    new_h = box_h * margin
    new_w = box_w * margin

    ymin_new = int(max(0, cy - new_h / 2))
    ymax_new = int(min(h, cy + new_h / 2))
    xmin_new = int(max(0, cx - new_w / 2))
    xmax_new = int(min(w, cx + new_w / 2))

    return ymin_new, xmin_new, ymax_new, xmax_new


def crop_image_with_mask(image_path, mask_path, output_path, margin=1.5, transform=None):
    """Crop one image according to its mask and save the output."""
    # Load image (.mat)
    image = loadmat(image_path).popitem()[-1]

    mask = np.array(Image.open(mask_path).convert("L")) > 0

    bbox = get_bbox_from_mask(mask)
    if bbox is None:
        print(f"Warning: No mask found for {mask_path.name}, skipping.")
        return

    bbox_expanded = expand_bbox(bbox, image.shape, margin)

    ymin, xmin, ymax, xmax = bbox_expanded
    cropped = image[ymin:ymax, xmin:xmax]
    if transform:
        augmented = transform(image=cropped)
        cropped = augmented["image"]

    savemat(output_path, {"DataCubeC": cropped})
    # save as png too for quick visualization

    # normalize to 0-255 for png
    # cropped_min = cropped.min()
    # cropped_max = cropped.max()
    # cropped_norm = (cropped - cropped_min) / (cropped_max - cropped_min) * 255
    # cropped_norm = cropped_norm
    # # take the mean and replicate to 3 channels
    # cropped_rgb = np.stack([cropped_norm.mean(axis=2)] * 3, axis=2).astype(np.uint8)
    # Image.fromarray(cropped_rgb).save(output_path.with_suffix(".png"))



def main():
    parser = argparse.ArgumentParser(description="Crop images using binary masks.")
    parser.add_argument("--images_dir", required=True, help="Directory with .mat images.")
    parser.add_argument("--masks_dir", required=True, help="Directory with mask .png files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save cropped images.")
    parser.add_argument("--margin", type=float, default=1.5, help="Margin multiplier (default=1.5).")
    parser.add_argument(
        "--pad_and_resize_shape",
        type=int,
        default=256,
        help="Final size to pad and resize cropped images (default=256).",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = A.Compose(
        [
            A.SmallestMaxSize(max_size=args.pad_and_resize_shape),
            A.CenterCrop(height=args.pad_and_resize_shape, width=args.pad_and_resize_shape),
        ]
    )

    for mat_path in images_dir.glob("*.mat"):
        mask_path = masks_dir / f"{mat_path.stem}.png"
        if not mask_path.exists():
            print(f"Mask not found for {mat_path.name}, skipping.")
            continue

        output_path = output_dir / f"{mat_path.stem}_cropped.mat"
        crop_image_with_mask(mat_path, mask_path, output_path, margin=args.margin, transform=transform)

        print(f"Saved cropped image to {output_path}")


if __name__ == "__main__":
    main()
