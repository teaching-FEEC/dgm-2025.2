import glob
import os
import shutil
import pandas as pd

def rename_and_move_images(csv_file, base_dir):
    # Load CSV
    if csv_file is None:
        csv_file = os.path.join(base_dir, "path_mapping.csv")
    df = pd.read_csv(csv_file)

    df.iloc[: ,0] = df.iloc[:, 0].str.replace("images", "masks")
    df.iloc[:, 1] = df.iloc[:, 1].str.replace("images", "masks")

    for _, row in df.iterrows():
        old_name = row[0]  # The current PNG name (first column)
        mat_path = row[1]  # The original_path like DNCube/124.mat

        # Create destination relative path by replacing .mat with .png
        relative_dst = os.path.splitext(mat_path)[0] + ".png"
        dst_path = os.path.join(base_dir, relative_dst)

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Source path is inside base_dir
        src_path = os.path.join(old_name)
        # remove _mask from the filename
        src_path = src_path.replace("_mask", "")

        # One source file could have generated multiple masks with different suffixes _00.png, _01.png, etc.
        # We need to find all such files and move them
        src_files = glob.glob(src_path.replace(".png", "_*.png"))

        if not src_files:
            print(f"Warning: No source files found for {src_path}")
            continue

        for src_file in src_files:
            # Create new destination path by adding the suffix before .png
            suffix = os.path.basename(src_file).replace(os.path.basename(src_path).replace(".png", ""), "")
            new_dst_path = dst_path.replace(".png", f"{suffix}")

            # Move the file
            shutil.move(src_file, new_dst_path)
            print(f"Moved {src_file} to {new_dst_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rename and move PNG images into nested directories based on CSV mapping."
    )
    parser.add_argument("--csv_file", type=str, default=None, help="Path to the CSV file with mappings.")
    parser.add_argument(
        "base_dir", type=str, help="Directory containing the PNG images (also used as output base)."
    )
    args = parser.parse_args()

    rename_and_move_images(args.csv_file, args.base_dir)
