import os
import argparse
import numpy as np
from PIL import Image

def convert_masks(input_dir, output_dir="masks"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all subdirectories recursively
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(".png"):
                continue

            input_path = os.path.join(root, filename)

            output_path = os.path.join(output_dir, filename)

            # Load as float (0.0–1.0)
            img = Image.open(input_path).convert("F")
            arr = np.array(img)

            # Scale to 0–255 and convert to uint8
            arr_255 = (arr * 255).astype(np.uint8)

            # Save as grayscale PNG
            Image.fromarray(arr_255, mode="L").save(output_path)

            print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively convert binary mask PNGs (0–1) to 0–255 grayscale PNGs"
    )
    parser.add_argument("input_dir", help="Path to directory containing input .png masks")
    parser.add_argument(
        "--output_dir", default="masks", help="Directory to save converted masks"
    )
    args = parser.parse_args()

    convert_masks(args.input_dir, args.output_dir)
