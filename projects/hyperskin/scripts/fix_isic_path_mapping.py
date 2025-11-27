#!/usr/bin/env python3
import argparse
import csv
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fix path_mapping.csv first column after images were renamed to "
            "ISIC_xxx.jpg, using original_path as reference."
        )
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing renamed ISIC_xxx.jpg files.",
    )
    parser.add_argument(
        "--mapping-csv",
        "-m",
        required=True,
        help="Existing path_mapping.csv to fix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing the updated CSV.",
    )
    return parser.parse_args()


def fix_mapping(input_dir: Path, mapping_csv: Path, dry_run: bool):
    input_dir = input_dir.resolve()

    with mapping_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "original_path" not in fieldnames:
            raise ValueError(
                "CSV must have an 'original_path' column. "
                f"Found: {fieldnames}"
            )

        # First column (not original_path) is assumed the path column to fix
        path_col = None
        for col in fieldnames:
            if col != "original_path":
                path_col = col
                break

        if path_col is None:
            raise ValueError("Could not determine path column to fix.")

        rows = list(reader)

    # Prepare changes
    changes = []
    for row in rows:
        orig = (row.get("original_path") or "").strip()
        if not orig:
            continue

        new_name = Path(orig).name  # ISIC_xxx.jpg
        new_path = f"{input_dir.name}/{new_name}"
        old_path = (row.get(path_col) or "").strip()

        # Optionally: verify file exists
        abs_img = input_dir / new_name
        if not abs_img.exists():
            print(
                f"Warning: expected image not found for mapping: {abs_img}",
                file=sys.stderr,
            )

        if old_path != new_path:
            changes.append((old_path, new_path))
            row[path_col] = new_path

    # Report / write
    if dry_run:
        print(f"[DRY-RUN] Would update {len(changes)} rows in {mapping_csv}:")
        for old, new in changes[:20]:
            print(f"  {old} -> {new}")
        if len(changes) > 20:
            print(f"  ... and {len(changes) - 20} more")
        return

    # Backup
    backup = mapping_csv.with_suffix(".bak.csv")
    shutil.copy2(mapping_csv, backup)
    print(f"Backup of original mapping written to: {backup}")

    # Write updated CSV in-place
    with mapping_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(
        f"Updated {mapping_csv} with {len(changes)} fixed paths in '{path_col}'."
    )


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    mapping_csv = Path(args.mapping_csv)

    if not input_dir.is_dir():
        print(f"Error: input-dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not mapping_csv.is_file():
        print(f"Error: mapping-csv not found: {mapping_csv}", file=sys.stderr)
        sys.exit(1)

    try:
        fix_mapping(input_dir, mapping_csv, args.dry_run)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
