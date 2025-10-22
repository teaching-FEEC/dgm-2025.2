#!/usr/bin/env python3
"""
fix_checkpoint.py — apply the '_class_path' → 'class_path' fix
and restructure `hyper_parameters` for an existing checkpoint file.

Usage:
    python fix_checkpoint.py path/to/checkpoint.ckpt
"""

import sys
import torch

import pyrootutils
from pathlib import Path
pyrootutils.setup_root(search_from=Path(__file__).parent, indicator=".project-root", pythonpath=True)

def fix_checkpoint(path: str) -> None:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        print(f"❌ No such file: {ckpt_path}")
        sys.exit(1)

    print(f"🧩 Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "hyper_parameters" not in checkpoint:
        print("ℹ️ No 'hyper_parameters' found — nothing to fix.")
        return

    hparams = checkpoint["hyper_parameters"]
    changed = False

    # Rename key
    if "_class_path" in hparams:
        hparams["class_path"] = hparams.pop("_class_path")
        changed = True

    # Restructure if there are multiple keys
    if len(hparams) > 2 and "class_path" in hparams and "_instantiator" in hparams:
        init_args = {
            k: v
            for k, v in hparams.items()
            if k not in ["class_path", "_instantiator"]
        }
        new_hparams = {
            "class_path": hparams["class_path"],
            "init_args": init_args,
            "_instantiator": hparams["_instantiator"],
        }
        checkpoint["hyper_parameters"] = new_hparams
        changed = True

    if changed:
        fixed_path = ckpt_path.with_name(ckpt_path.stem + "_fixed.ckpt")
        torch.save(checkpoint, fixed_path)
        print(f"✅ Fixed checkpoint saved to: {fixed_path}")
    else:
        print("✔️ No changes necessary — checkpoint already correct.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    fix_checkpoint(sys.argv[1])
