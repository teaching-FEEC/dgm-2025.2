#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility to build a *symlinked* synthetic dataset split with controlled class
distributions.

You start from a folder like:

    synthetic_dataset/
        MM_Cube/
            img_000.npy
            ...
        DN_Cube/
            img_000.npy
            ...
        Other_Cube/
            ...

We create a new folder with the same subfolder structure, but instead of
copying files, we create symbolic links pointing back to the original
synthetic files.

Selection rules are controlled by:
    - synth_mode   ∈ {"full_train", "mixed_train", "full_val"}
    - equity_mode  ∈ {
                        "real-synth_balanced",
                        "mm-dn_equal",
                        "mm-dn_balanced",
                        "imbalanced",
                        "mm-dn_compensated"
                      }

All numbers / rules are heavily commented below.

NOTE:
- This code assumes images are `.npy` files. Change the GLOB_PATTERN constant
  if your extension is different.
- Symlinks work best on Linux / macOS. On Windows you may need admin rights,
  or enable developer mode for symlinks.
"""

import os                      # basic path operations + symlink
import glob                    # file pattern matching
from pathlib import Path       # nicer path handling
from typing import Tuple       # type hints
import argparse                # optional CLI interface

# -------------------------------------------------------------------------
# Constants describing the *real* dataset (given in your description)
# -------------------------------------------------------------------------

# Number of real MM images
REAL_MM_COUNT = 66

# Number of real DN images
REAL_DN_COUNT = 134

# File extension of your synthetic cubes (change if needed)
GLOB_PATTERN = "*.mat"


def compute_selection_counts(
    n_mm_avail: int,
    n_dn_avail: int,
    synth_mode: str,
    equity_mode: str,
) -> Tuple[int, int]:
    """
    Decide HOW MANY synthetic MM and DN images to select, based on:
        - how many are available in synthetic dataset
        - synth_mode
        - equity_mode

    Parameters
    ----------
    n_mm_avail : int
        Number of synthetic MM files available.
    n_dn_avail : int
        Number of synthetic DN files available.
    synth_mode : str
        One of {"full_train", "mixed_train", "full_val"}.
    equity_mode : str
        One of {"real-synth_balanced", "mm-dn_equal", "mm-dn_balanced",
                "imbalanced", "mm-dn_compensated"}.

    Returns
    -------
    (n_mm, n_dn) : Tuple[int, int]
        Number of MM and DN synthetic images to select.

    Notes
    -----
    Rules are chosen to match your comments in the prompt as closely as
    possible; where the spec was incomplete, defaults are documented.

    - full_train + mm-dn_equal:
        Use ALL synthetic data, limited by what's available, 1:1.
        For your case: 100 MM & 100 DN.

    - full_train + imbalanced:
        Keep original real distribution: 1 MM : 2 DN (ratio only).
        For your case: 50 MM & 100 DN (maximal with 1:2 and ≤100 each).

    - full_train + real-synth_balanced:
        Per class, synthetic count = min(real_count, available_synth_count).
        MM: min(66, 100) = 66
        DN: min(134, 100) = 100

    - full_train + mm-dn_balanced:
        Alias of mm-dn_equal here: balanced MM:DN (1:1), use min(avail).

    - full_train + mm-dn_compensated:
        Not defined in your text for full_train; we raise an error.

    - mixed_train + mm-dn_balanced:
        From your description:
            "There should be inverted distribution of real data,
             2 MM for every 1 DN. And 100 imgs in MM"
        So we:
            n_mm = all available MM (typically 100)
            n_dn = floor(n_mm / 2), limited by available DN.

        For your case: 100 MM & 50 DN.

    - mixed_train + mm-dn_compensated:
        From your description:
            "MM folder should have 66 MM (the same amount as real data)
             and DN folder should have 13 DN"
        So:
            n_mm = min(66, n_mm_avail)
            n_dn = min(13, n_dn_avail)

    - mixed_train + imbalanced:
        Same 1 MM : 2 DN ratio as above. For your 100/100 case: 50/100.

    - mixed_train + real-synth_balanced:
        Same as full_train real-synth_balanced.

    - full_val + *any* equity_mode:
        For validation we simply use ALL synthetic data:
            n_mm = n_mm_avail
            n_dn = n_dn_avail
        Equity mode is ignored, but we keep it as a parameter for API
        consistency.
    """
    # Normalize mode strings to avoid tiny typos in user code
    synth_mode = synth_mode.strip().lower()
    equity_mode = equity_mode.strip().lower()

    # ----------------------------- full_val ------------------------------
    if synth_mode == "full_val":
        # Use ALL available synthetic data for validation, ignoring equity.
        n_mm = n_mm_avail
        n_dn = n_dn_avail
        return n_mm, n_dn

    # ---------------------------- full_train -----------------------------
    if synth_mode == "full_train":
        if equity_mode == "mm-dn_equal":
            # Balanced 1:1, but cannot exceed what's available
            n = min(n_mm_avail, n_dn_avail)
            return n, n

        elif equity_mode == "imbalanced":
            # Keep original distribution of real data: 1 MM : 2 DN
            # We choose the *maximal* counts that respect the ratio and
            # availability.
            # Let mm = k, dn = 2*k, with constraints:
            #   k <= n_mm_avail
            #   2*k <= n_dn_avail  -> k <= n_dn_avail // 2
            k = min(n_mm_avail, n_dn_avail // 2)
            return k, 2 * k

        elif equity_mode == "real-synth_balanced":
            # Per class, synthetic ≤ real count, but also ≤ available
            mm = min(REAL_MM_COUNT, n_mm_avail)
            dn = min(REAL_DN_COUNT, n_dn_avail)
            return mm, dn

        elif equity_mode == "mm-dn_balanced":
            # For full_train, interpret as simple 1:1 balancing.
            n = min(n_mm_avail, n_dn_avail)
            return n, n

        elif equity_mode == "mm-dn_compensated":
            raise ValueError(
                "equity_mode='mm-dn_compensated' is not defined for "
                "synth_mode='full_train' in this implementation."
            )

        else:
            raise ValueError(f"Unknown equity_mode: {equity_mode}")

    # --------------------------- mixed_train -----------------------------
    if synth_mode == "mixed_train":
        if equity_mode == "mm-dn_balanced":
            # Inverted distribution of real data: 2 MM : 1 DN,  100 MM and 50 DN
            # "And 100 imgs in MM" -> generalize to "use all MM" and deduce DN.
            n_mm = n_mm_avail
            n_dn = min(n_dn_avail, n_mm // 2)
            return n_mm, n_dn

        elif equity_mode in ("mm-dn_compensated", "mm_dn_compensated"):
            # Match your explicit numbers:
            #   MM: 66  (same as real)
            #   DN: 13  (smaller than real)
            n_mm = min(REAL_MM_COUNT, n_mm_avail)   # typically 66
            n_dn = min(13, n_dn_avail)              # typically 13
            return n_mm, n_dn

        elif equity_mode == "imbalanced":
            # Same "real-like" 1:2 ratio as before
            k = min(n_mm_avail, n_dn_avail // 2) #100 DN and 50 MM 
            return k, 2 * k

        elif equity_mode == "real-synth_balanced":
            # Same as full_train real-synth_balanced
            mm = min(REAL_MM_COUNT, n_mm_avail)
            dn = min(REAL_DN_COUNT, n_dn_avail)
            return mm, dn

        elif equity_mode == "mm-dn_equal":
            # Reasonable default: 1:1, use min(avail)
            n = min(n_mm_avail, n_dn_avail)
            return n, n

        else:
            raise ValueError(f"Unknown equity_mode: {equity_mode}")

    # -------------------------- unknown synth_mode -----------------------
    raise ValueError(f"Unknown synth_mode: {synth_mode}")


def make_symlink(src: Path, dst: Path) -> None:
    """
    Create a symbolic link `dst` pointing to `src`.

    If the destination already exists, we raise an error, to avoid silently
    overwriting anything.
    """
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    # Create parent directories if necessary
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Create the symlink
    os.symlink(src, dst)


def build_synthetic_split(
    synth_folder: str,
    output_root: str,
    synth_mode: str,
    equity_mode: str,
    verbose: bool = True,
) -> None:
    """
    Main function: create an output folder with the same structure as
    `synth_folder`, but filled with symlinks according to the selected
    synth_mode and equity_mode.

    Parameters
    ----------
    synth_folder : str
        Path to the original synthetic dataset root (contains MM_Cube, DN_Cube,
        Other_Cube).
    output_root : str
        Path to the *parent* directory where we will create the new split
        folder. The split folder will be named:
            f"{synth_mode}_{equity_mode}"
    synth_mode : str
        One of {"full_train", "mixed_train", "full_val"}.
    equity_mode : str
        One of {"real-synth_balanced", "mm-dn_equal", "mm-dn_balanced",
                "imbalanced", "mm-dn_compensated"}.
    verbose : bool
        If True, print human-readable summary of what is done.
    """
    # Convert to Path objects for convenience and safety
    synth_root = Path(synth_folder).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    # Check that the synthetic root exists
    if not synth_root.is_dir():
        raise NotADirectoryError(f"synthetic root not found: {synth_root}")

    # Define the standard subfolders (from your description)
    mm_dir = synth_root / "MMCube"
    dn_dir = synth_root / "DNCube"
    other_dir = synth_root / "OtherCube"

    # Sanity-check that mandatory dirs exist
    if not mm_dir.is_dir():
        raise NotADirectoryError(f"Missing 'MMCube' in: {synth_root}")
    if not dn_dir.is_dir():
        raise NotADirectoryError(f"Missing 'DNCube' in: {synth_root}")
    # Other_Cube may be optional, but we handle it if present

    # Collect all MM files and DN files using the glob pattern
    mm_files = sorted(mm_dir.glob(GLOB_PATTERN))
    dn_files = sorted(dn_dir.glob(GLOB_PATTERN))

    # Count how many are available
    n_mm_avail = len(mm_files)
    n_dn_avail = len(dn_files)

    if verbose:
        print(f"[INFO] Found {n_mm_avail} MM files and {n_dn_avail} DN files.")

    # Ask the selection logic how many we should pick
    n_mm_select, n_dn_select = compute_selection_counts(
        n_mm_avail=n_mm_avail,
        n_dn_avail=n_dn_avail,
        synth_mode=synth_mode,
        equity_mode=equity_mode,
    )

    if verbose:
        print(
            f"[INFO] Selection counts for synth_mode='{synth_mode}', "
            f"equity_mode='{equity_mode}': "
            f"{n_mm_select} MM, {n_dn_select} DN."
        )

    # Define the output split directory name (unique per mode combination)
    split_name = f"{synth_mode}__{equity_mode}"
    out_split_dir = output_root / split_name

    # Safeguard: if the folder already exists and isn't empty, we stop
    if out_split_dir.exists() and any(out_split_dir.iterdir()):
        raise FileExistsError(
            f"Output split directory already exists and is not empty: "
            f"{out_split_dir}"
        )

    # Create the base output directory (if not existing)
    out_split_dir.mkdir(parents=True, exist_ok=True)

    # Create paths for the output subfolders
    out_mm_dir = out_split_dir /"images"/ "MMCube"
    out_dn_dir = out_split_dir /"images"/ "DNCube"
    out_other_dir = out_split_dir /"images"/ "OtherCube"

    # Make sure subfolders exist
    out_mm_dir.mkdir(parents=True, exist_ok=True)
    out_dn_dir.mkdir(parents=True, exist_ok=True)
    # Only create Other_Cube if the original exists
    if other_dir.is_dir():
        out_other_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------- Symlink MM files ---------------------------
    # We use the first N sorted files; if you want randomness, you can
    # shuffle mm_files before slicing.
    for src in mm_files[:n_mm_select]:
        dst = out_mm_dir / src.name
        make_symlink(src, dst)

    # ------------------------- Symlink DN files ---------------------------
    for src in dn_files[:n_dn_select]:
        dst = out_dn_dir / src.name
        make_symlink(src, dst)

    # ---------------------- Symlink Other_Cube files ----------------------
    # Strategy: copy ALL Others, because they're typically not part of the
    # MM vs DN balancing.
    if other_dir.is_dir():
        other_files = sorted(other_dir.glob("*"))
        for src in other_files:
            # Skip subdirectories; only symlink files here
            if src.is_file():
                dst = out_other_dir / src.name
                make_symlink(src, dst)

    if verbose:
        print(f"[DONE] Created symlinked split at: {out_split_dir}")
        print(f"       MMCube: {n_mm_select} files")
        print(f"       DNCube: {n_dn_select} files")
        if other_dir.is_dir():
            print("       OtherCube: all files were symlinked.")


# -------------------------------------------------------------------------
# Optional: command-line interface
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Create an argument parser so you can run from terminal, e.g.:
    #
    # python build_synth_split.py \
    #   --synth_folder path_to_synthetic_data_folder \
    #   --output_root ./synthetic_splits \
    #   --synth_mode mixed_train \
    #   --equity_mode mm-dn_balanced
    #
    parser = argparse.ArgumentParser(
        description=(
            "Create symlinked synthetic dataset split with controlled "
            "MM/DN distributions."
        )
    )

    # Path to the original synthetic dataset
    parser.add_argument(
        "--synth_folder",
        type=str,
        required=True,
        help="Path to synthetic dataset folder (contains MM_Cube, DN_Cube, Other_Cube).",
    )

    # Where to put the created splits
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help=(
            "Directory where the new split folder will be created. "
            "It will be named '<synth_mode>__<equity_mode>'."
        ),
    )

    # synth_mode argument
    parser.add_argument(
        "--synth_mode",
        type=str,
        choices=["full_train", "mixed_train", "full_val"],
        required=True,
        help="Synthetic usage mode: full_train, mixed_train, or full_val.",
    )

    # equity_mode argument
    parser.add_argument(
        "--equity_mode",
        type=str,
        choices=[
            "real-synth_balanced",
            "mm-dn_equal",
            "mm-dn_balanced",
            "imbalanced",
            "mm-dn_compensated",
        ],
        required=True,
        help="Equity strategy for MM vs DN and real vs synthetic.",
    )

    args = parser.parse_args()

    # Call the main function with parsed arguments
    build_synthetic_split(
        synth_folder=args.synth_folder,
        output_root=args.output_root,
        synth_mode=args.synth_mode,
        equity_mode=args.equity_mode,
        verbose=True,
    )
