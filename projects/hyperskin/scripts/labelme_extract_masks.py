import os
import json
import argparse
import base64
from pathlib import Path
from PIL import Image
import io
import numpy as np


def paste_mask_on_canvas(mask_img, bbox, image_width, image_height):
    """
    Places the decoded mask into the full-size image canvas
    at the bounding box location defined by points.
    """
    # bbox: [[x1, y1], [x2, y2]] with top-left and bottom-right coords
    (x1, y1), (x2, y2) = bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    canvas = Image.new("L", (image_width, image_height), 0)

    # Resize mask to exactly match bbox size (should match already, but safe)
    bbox_w, bbox_h = (x2 - x1), (y2 - y1)
    mask_img = mask_img.resize((bbox_w, bbox_h), Image.NEAREST)

    canvas.paste(mask_img, (x1, y1))
    return np.array(canvas, dtype=np.uint8)


def extract_masks(input_dir: str, output_dir: str = "./masks", combine: bool = False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in input_dir.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        image_height = data.get("imageHeight")
        image_width = data.get("imageWidth")
        image_name = Path(data.get("imagePath", json_file.stem + ".png")).stem

        masks = []
        for shape in data.get("shapes", []):
            if "mask" in shape and "points" in shape and shape["mask"]:
                mask_b64 = shape["mask"]
                mask_bytes = base64.b64decode(mask_b64)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

                # Insert into correct position in full-size canvas
                mask_full = paste_mask_on_canvas(
                    mask_img, shape["points"], image_width, image_height
                )
                masks.append(mask_full)

        if not masks:
            continue

        if combine:
            combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            for m in masks:
                combined_mask = np.maximum(combined_mask, m)

            out_img = Image.fromarray(combined_mask, mode="L")
            output_path = output_dir / f"{image_name}_mask.png"
            out_img.save(output_path, format="PNG")
            print(f"✔ Saved combined mask: {output_path}")

        else:
            for idx, m in enumerate(masks):
                out_img = Image.fromarray(m, mode="L")
                output_path = output_dir / f"{image_name}_crop{idx:02d}_mask.png"
                out_img.save(output_path, format="PNG")
                print(f"✔ Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract masks from LabelMe JSON files and save as PNG."
    )
    parser.add_argument("input_dir", help="Directory containing LabelMe JSON files.")
    parser.add_argument(
        "--output_dir",
        default="./masks",
        help="Directory to save extracted masks (default: ./masks)",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="If set, combine all masks in a JSON file into one image.",
    )
    args = parser.parse_args()

    extract_masks(args.input_dir, args.output_dir, args.combine)
