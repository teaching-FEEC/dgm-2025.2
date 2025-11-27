import re
import shutil
import pandas as pd
from pathlib import Path


def organize_isic_masks(
    input_dir,
    csv_path="data/MILK10k/MILK10k_Training_Metadata.csv",
):
    """
    Moves mask files like ISIC_0184224_00_mask.png from input_dir into
    subfolders named after their lesion_id, based on a CSV file mapping.
    """

    input_dir = Path(input_dir)
    csv_path = Path(csv_path)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a valid directory.")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load the CSV
    df = pd.read_csv(csv_path)

    required_cols = {"lesion_id", "isic_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {', '.join(required_cols)}. Found: {df.columns}"
        )

    # Map ISIC IDs → lesion IDs
    isic_to_lesion = dict(zip(df["isic_id"], df["lesion_id"]))

    # Pattern like ISIC_0184224 inside filenames such as ISIC_0184224_00_mask.png
    pattern = re.compile(r"(ISIC_\d+)")

    moved_count = 0
    skipped_count = 0

    for file_path in input_dir.glob("*.png"):
        if not file_path.is_file():
            continue

        match = pattern.search(file_path.name)
        if not match:
            print(f"⚠️  Skipping (no ISIC ID found): {file_path.name}")
            skipped_count += 1
            continue

        isic_id = match.group(1)
        lesion_id = isic_to_lesion.get(isic_id)

        if lesion_id is None:
            print(f"⚠️  No lesion_id found for {isic_id}, skipping {file_path.name}")
            skipped_count += 1
            continue

        # Create subfolder for this lesion
        lesion_dir = input_dir / lesion_id
        lesion_dir.mkdir(exist_ok=True)

        target_path = lesion_dir / file_path.name
        shutil.move(str(file_path), str(target_path))

        print(f"✅  Moved {file_path.name} → {lesion_id}/")
        moved_count += 1

    print(f"\nDone! Moved {moved_count} files, skipped {skipped_count}.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Organize ISIC mask image files into folders by lesion_id."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing the .png mask files (e.g., ISIC_0184224_00_mask.png).",
    )
    parser.add_argument(
        "--csv",
        default="data/MILK10k/MILK10k_Training_Metadata.csv",
        help="Path to the ground truth CSV (default: data/MILK10k/MILK10k_Training_Metadata.csv).",
    )

    args = parser.parse_args()
    organize_isic_masks(args.input_dir, args.csv)
