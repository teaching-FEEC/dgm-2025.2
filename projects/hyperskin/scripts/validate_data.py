#!/usr/bin/env python3
import os
import argparse
import numpy as np
from scipy.io import loadmat

def load_image_from_mat(file_path: str) -> np.ndarray:
    """Load a .mat file and extract the contained image array as float32."""
    mat_data = loadmat(file_path)
    image = mat_data.popitem()[-1]
    return image.astype(np.float32)


def check_image_condition(image: np.ndarray) -> str | None:
    """
    Verify that an image satisfies the shape and interval conditions.

    Returns:
        None if valid, or a string describing why it is invalid.
    """
    # if image.shape != (256, 256, 16):
    #     return f"wrong shape {image.shape}"

    # Check normalization ranges
    all_in_0_1 = np.all((image >= 0) & (image <= 1))
    all_in_minus1_1 = np.all((image >= -1) & (image <= 1))

    if all_in_0_1:
        return "values normalized to [0, 1]"
    if all_in_minus1_1:
        return "values normalized to [-1, 1]"

    # If neither range applies, it's valid
    return None


def validate_subdirectories(root_dir: str):
    """
    For each subdirectory, recursively check .mat files.
    Print invalid subdirectories with number of files and error reason.
    """
    for subdir_name in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        mat_files = []
        for dirpath, _, filenames in os.walk(subdir_path):
            for filename in filenames:
                if filename.lower().endswith(".mat"):
                    mat_files.append(os.path.join(dirpath, filename))

        if not mat_files:
            continue  # Skip empty subdirs

        invalid_reasons = []
        for file_path in mat_files:
            try:
                image = load_image_from_mat(file_path)
                reason = check_image_condition(image)
                if reason:
                    invalid_reasons.append(reason)
            except Exception as e:
                invalid_reasons.append(f"error loading file ({e})")

        if invalid_reasons:
            unique_reasons = sorted(set(invalid_reasons))
            print(f"Invalid subdirectory: {subdir_name}")
            print(f"  .mat files: {len(mat_files)}")
            for r in unique_reasons:
                print(f"  - {r}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate subdirectories containing .mat image files."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the root directory containing subdirectories."
    )

    args = parser.parse_args()
    validate_subdirectories(args.root_dir)


if __name__ == "__main__":
    main()