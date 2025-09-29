import os
import cv2
import json
import base64
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image
from sympy import im


def mask_to_base64(mask: np.ndarray) -> str:
    """Encode mask as base64 PNG string."""
    pil_img = Image.fromarray(mask.astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_bounding_box(mask: np.ndarray):
    """Compute bounding box (xmin, ymin, xmax, ymax) from mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:  # empty mask
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def process_dataset(input_dir: str):
    input_dir = Path(input_dir)
    images_dir = input_dir / "images"
    masks_dir = input_dir / "masks"

    for mask_path in masks_dir.glob("*.png"):
        # Match original image name
        if not mask_path.name.endswith("_mask.png"):
            continue
        image_name = mask_path.name.replace("_mask.png", ".png")
        image_path = images_dir / image_name

        if not image_path.exists():
            print(f"Warning: image for {mask_path.name} not found.")
            continue

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Skipping {mask_path}, could not read image.")
            continue

        # Bounding box
        bbox = get_bounding_box(mask)
        if bbox is None:
            print(f"Skipping {mask_path}, empty mask.")
            continue

        x_min, y_min, x_max, y_max = map(int, bbox)
        points = [[float(x_min), float(y_min)], [float(x_max), float(y_max)]]

        # Crop the mask to bbox (so Labelme aligns it correctly)
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

        # Encode the cropped mask
        encoded_mask = mask_to_base64(cropped_mask)

        # Read original image for size
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        # Build JSON
        data = {
            "version": "5.8.3",
            "flags": {},
            "shapes": [
                {
                    "label": "skin lesion",
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "mask",
                    "flags": {},
                    "mask": encoded_mask,  # base64 of cropped mask
                }
            ],
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }
        # Save JSON in images folder
        json_path = images_dir / (image_name.replace(".png", ".json"))
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved annotation: {json_path}")


if __name__ == "__main__":
    import sys
    sys.argv += ["export/hsi_dermoscopy_rgb"]  # Example argument for testing
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Labelme JSONs from mask dataset"
    )
    parser.add_argument("input_dir", help="Path to dataset directory")
    args = parser.parse_args()

    process_dataset(args.input_dir)
