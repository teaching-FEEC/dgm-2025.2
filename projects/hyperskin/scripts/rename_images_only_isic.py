#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rename images and LabelMe JSONs using path_mapping.csv, update "
            "JSON imagePath, and rewrite mapping CSV to new paths."
        )
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing .jpg and .json files.",
    )
    parser.add_argument(
        "--mapping-csv",
        "-m",
        required=True,
        help=(
            "CSV with columns: <current_path>,original_path. "
            "Example row: "
            "export/milk10k_nevi/train_xxx.jpg,data/.../ISIC_xxx.jpg"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without modifying files.",
    )
    parser.add_argument(
        "--no-overwrite-mapping",
        action="store_true",
        help=(
            "Do not overwrite the original mapping CSV. "
            "Write updated version as <mapping-csv>.updated.csv instead."
        ),
    )
    return parser.parse_args()


def load_mapping(mapping_csv: Path, input_dir: Path):
    """
    Returns:
      mapping: dict {old_image_path_abs: new_image_name}
      rows: original CSV rows list (to later rewrite)
      current_col: name of the first column
    """
    mapping = {}
    rows = []

    with mapping_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or "original_path" not in fieldnames:
            raise ValueError(
                "CSV must have an 'original_path' column. "
                f"Found: {fieldnames}"
            )

        # First column (not original_path) is assumed the current-path column
        current_col = None
        for col in fieldnames:
            if col != "original_path":
                current_col = col
                break

        if current_col is None:
            raise ValueError(
                "Could not infer current-path column in mapping CSV."
            )

        for row in reader:
            rows.append(row)

            current_path_str = (row.get(current_col) or "").strip()
            original_path_str = (row.get("original_path") or "").strip()
            if not current_path_str or not original_path_str:
                continue

            # Old name: use basename from first column
            old_name = Path(current_path_str).name
            old_image_path = (input_dir / old_name).resolve()

            # New name: basename of original_path (e.g. ISIC_1498519.jpg)
            new_name = Path(original_path_str).name
            if not new_name.lower().endswith(".jpg"):
                print(
                    f"Warning: original_path '{original_path_str}' "
                    "does not end with .jpg; skipping.",
                    file=sys.stderr,
                )
                continue

            mapping[old_image_path] = new_name

    return mapping, rows, current_col


def plan_renames(mapping, input_dir: Path):
    image_renames = []
    json_renames = []
    json_updates = []
    updated_mapping_entries = []  # (new_image_rel_path, original_path)

    for old_image_abs, new_image_name in mapping.items():
        if not old_image_abs.exists():
            print(
                f"Warning: image not found, skipping: {old_image_abs}",
                file=sys.stderr,
            )
            continue

        if not old_image_abs.name.lower().endswith(".jpg"):
            print(
                f"Warning: not a .jpg file in mapping, skipping: "
                f"{old_image_abs}",
                file=sys.stderr,
            )
            continue

        new_image_abs = (input_dir / new_image_name).resolve()
        if new_image_abs.exists() and new_image_abs != old_image_abs:
            raise FileExistsError(
                f"Target image already exists: {new_image_abs}"
            )

        image_renames.append((old_image_abs, new_image_abs))

        # JSON handling
        old_json_abs = old_image_abs.with_suffix(".json")
        if old_json_abs.exists():
            new_json_abs = new_image_abs.with_suffix(".json")
            if new_json_abs.exists() and new_json_abs != old_json_abs:
                raise FileExistsError(
                    f"Target JSON already exists: {new_json_abs}"
                )
            json_renames.append((old_json_abs, new_json_abs))
            json_updates.append(new_json_abs)

        # For mapping CSV, we want new path relative to input_dir's parent
        # But your example uses "export/milk10k_nevi/ISIC_xxx.jpg"
        # So we construct: f"{input_dir.name}/{new_image_name}"
        new_rel = f"{input_dir.name}/{new_image_name}"
        updated_mapping_entries.append((old_image_abs, new_rel))

    return image_renames, json_renames, json_updates, updated_mapping_entries


def apply_renames(image_renames, json_renames, dry_run: bool):
    # Images
    for old_path, new_path in image_renames:
        if old_path == new_path:
            continue
        if dry_run:
            print(f"[DRY-RUN] mv {old_path} -> {new_path}")
        else:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)

    # JSONs
    for old_path, new_path in json_renames:
        if old_path == new_path:
            continue
        if dry_run:
            print(f"[DRY-RUN] mv {old_path} -> {new_path}")
        else:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)


def update_json_image_paths(json_paths, dry_run: bool):
    for json_path in json_paths:
        if not json_path.exists():
            print(
                f"Warning: JSON not found for update: {json_path}",
                file=sys.stderr,
            )
            continue

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(
                f"Warning: Failed to read JSON {json_path}: {e}",
                file=sys.stderr,
            )
            continue

        new_image_name = json_path.with_suffix(".jpg").name
        old_image_path_val = data.get("imagePath")

        if old_image_path_val == new_image_name:
            continue

        data["imagePath"] = new_image_name

        if dry_run:
            print(
                f"[DRY-RUN] update {json_path}: "
                f"imagePath '{old_image_path_val}' -> '{new_image_name}'"
            )
        else:
            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(
                    f"Warning: Failed to write JSON {json_path}: {e}",
                    file=sys.stderr,
                )


def rewrite_mapping_csv(
    mapping_csv: Path,
    rows,
    current_col: str,
    updated_entries,
    input_dir: Path,
    dry_run: bool,
    no_overwrite: bool,
):
    # Build a lookup: old_abs -> new_rel
    old_to_new_rel = {old: new_rel for old, new_rel in updated_entries}

    # Output path
    if no_overwrite:
        out_csv = mapping_csv.with_suffix(".updated.csv")
    else:
        out_csv = mapping_csv

    # Backup original if overwriting
    if not no_overwrite and not dry_run:
        backup = mapping_csv.with_suffix(".bak.csv")
        shutil.copy2(mapping_csv, backup)
        print(f"Backup of original mapping written to: {backup}")

    # Re-open original to get fieldnames reliably
    with mapping_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    if dry_run:
        print(f"[DRY-RUN] would write updated mapping to: {out_csv}")
        return

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            current_path_str = (row.get(current_col) or "").strip()
            if current_path_str:
                old_name = Path(current_path_str).name
                old_abs = (input_dir / old_name).resolve()
                if old_abs in old_to_new_rel:
                    row[current_col] = old_to_new_rel[old_abs]
            writer.writerow(row)

    print(f"Updated mapping written to: {out_csv}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    mapping_csv = Path(args.mapping_csv).resolve()

    if not input_dir.is_dir():
        print(f"Error: input-dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if not mapping_csv.is_file():
        print(f"Error: mapping-csv not found: {mapping_csv}", file=sys.stderr)
        sys.exit(1)

    mapping, rows, current_col = load_mapping(mapping_csv, input_dir)
    if not mapping:
        print("No valid mappings found.", file=sys.stderr)
        sys.exit(1)

    try:
        (
            image_renames,
            json_renames,
            json_updates,
            updated_entries,
        ) = plan_renames(mapping, input_dir)
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not image_renames:
        print("No matching images to rename.", file=sys.stderr)
        sys.exit(0)

    apply_renames(image_renames, json_renames, args.dry_run)
    update_json_image_paths(json_updates, args.dry_run)
    rewrite_mapping_csv(
        mapping_csv,
        rows,
        current_col,
        updated_entries,
        input_dir,
        args.dry_run,
        args.no_overwrite_mapping,
    )

    if args.dry_run:
        print("Dry run completed. No files or CSV were modified.")
    else:
        print("Renaming, JSON updates, and mapping CSV update completed.")


if __name__ == "__main__":
    main()
