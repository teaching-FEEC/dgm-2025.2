#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize MILK10k mask files into IL_xxxxx/ISIC_xxxxx_crop00_mask.png "
            "structure based on a mapping CSV."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the original mask PNG files.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="CSV file with mapping from exported filenames to original_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base directory where reorganized masks will be written.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    return parser.parse_args()


def load_mapping(csv_path: Path) -> dict:
    """
    Returns:
        mapping: dict[str, tuple[str, str]]
            key: zero-padded ID string, e.g. '00001'
            value: (IL_id, ISIC_id)
    """
    mapping: dict[str, tuple[str, str]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        # Handle possible unnamed index column in first position
        reader = csv.DictReader(f)
        # Normalize header names (strip spaces, lowercase)
        fieldnames = {name.strip().lower(): name for name in reader.fieldnames or []}

        # We expect a column named "original_path" (case-insensitive)
        if "original_path" not in fieldnames:
            raise ValueError(
                f"CSV must contain an 'original_path' column. "
                f"Found columns: {reader.fieldnames}"
            )

        original_path_key = fieldnames["original_path"]

        # Find the column that has the exported filename path
        # Often it's the second column or has a name like '' or 'index'
        # We'll treat all columns except 'original_path' as candidates and
        # parse each row robustly.
        candidate_keys = [
            k
            for k in reader.fieldnames
            if k is not None and k != original_path_key
        ]

        if not candidate_keys:
            raise ValueError(
                "CSV must have at least one column with the exported filename "
                "besides 'original_path'."
            )

        # Regex to extract:
        #   train_melanocytic_nevus_00001
        # and get the 00001 part
        id_pattern = re.compile(r"train_melanocytic_nevus_(\d+)", re.IGNORECASE)

        for row in reader:
            original_path = row[original_path_key].strip()

            # Determine which column has the export path for this row
            export_value = None
            for key in candidate_keys:
                val = (row.get(key) or "").strip()
                if val:
                    export_value = val
                    break

            if not export_value:
                continue

            m = id_pattern.search(export_value)
            if not m:
                continue

            sample_id = m.group(1)  # e.g. '00001'

            # Parse original_path: data/MILK10k/images/IL_0008891/ISIC_1498519.jpg
            parts = original_path.split("/")
            if len(parts) < 2:
                continue

            # Expect the last two components to be IL_xxx dir and ISIC_xxx.jpg
            il_id = None
            isic_id = None

            # Search from the end safely
            for i in range(len(parts) - 1):
                if parts[i].startswith("IL_"):
                    il_id = parts[i]
                    # Next component should be ISIC_xxx.ext
                    if i + 1 < len(parts):
                        fname = parts[i + 1]
                        if fname.startswith("ISIC_"):
                            isic_id = os.path.splitext(fname)[0]
                    break

            if not il_id or not isic_id:
                continue

            mapping[sample_id] = (il_id, isic_id)

    return mapping


def reorganize_masks(
    input_dir: Path, output_dir: Path, mapping: dict, copy: bool = False
) -> None:
    """
    For each mask file in input_dir:
        train_melanocytic_nevus_00001_crop00_mask.png
    find '00001', look up (IL_xxx, ISIC_xxx), and move/copy to:
        output_dir/IL_xxx/ISIC_xxx_crop00_mask.png
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pattern to extract ID and rest of suffix from mask filename
    # Example filename:
    #   train_melanocytic_nevus_00001_crop00_mask.png
    # Groups:
    #   (1) -> '00001'
    #   (2) -> '_crop00_mask.png'
    mask_pattern = re.compile(
        r"^train_melanocytic_nevus_(\d+)(.*_mask\.png)$", re.IGNORECASE
    )

    moved = 0
    skipped_no_match = 0
    skipped_no_mapping = 0

    for entry in sorted(input_dir.iterdir()):
        if not entry.is_file():
            continue
        if not entry.name.lower().endswith("_mask.png"):
            continue

        m = mask_pattern.match(entry.name)
        if not m:
            skipped_no_match += 1
            continue

        sample_id = m.group(1)  # '00001'
        suffix = m.group(2)  # '_crop00_mask.png'

        if sample_id not in mapping:
            skipped_no_mapping += 1
            continue

        il_id, isic_id = mapping[sample_id]

        # New name: ISIC_xxxxxxxx_suffix (e.g. ISIC_1498519_crop00_mask.png)
        new_filename = f"{isic_id}{suffix}"
        target_dir = output_dir / il_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / new_filename

        if copy:
            shutil.copy2(entry, target_path)
        else:
            shutil.move(str(entry), target_path)

        moved += 1

    print("Done.")
    print(f"Moved/Copied: {moved}")
    print(f"Skipped (no filename match): {skipped_no_match}")
    print(f"Skipped (no CSV mapping): {skipped_no_mapping}")


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.csv_path)
    if not mapping:
        raise SystemExit("No mappings loaded from CSV; check input file.")
    reorganize_masks(args.input_dir, args.output_dir, mapping, copy=args.copy)


if __name__ == "__main__":
    main()
