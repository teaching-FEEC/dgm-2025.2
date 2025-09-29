import os
import shutil
import pandas as pd

def rename_and_move_images(csv_file, base_dir):
    # Load CSV
    if csv_file is None:
        csv_file = os.path.join(base_dir, "path_mapping.csv")
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        old_name = row[0]  # The current PNG name (first column)
        mat_path = row[1]  # The original_path like DNCube/124.mat

        # Create destination relative path by replacing .mat with .png
        relative_dst = os.path.splitext(mat_path)[0] + ".png"
        dst_path = os.path.join(base_dir, relative_dst)

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Source path is inside base_dir
        src_path = os.path.join(base_dir, old_name)

        # Move and rename file if it exists
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
            print(f"Moved: {src_path} → {dst_path}")
        else:
            print(f"⚠️ File not found: {src_path}")

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
