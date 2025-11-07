import argparse
import sys
import torch
from typing import Iterable

DEFAULT_IGNORE_PREFIXES = ("mifid.", "metrics.mifid.", "mifid_inception.", "mifid.")


def clean_state_dict(sd: dict, ignore_prefixes: Iterable[str]) -> tuple[dict, list]:
    new_sd = {}
    removed = []
    for k, v in sd.items():
        if any(k.startswith(p) for p in ignore_prefixes):
            removed.append(k)
            continue

        # strip common wrapper prefixes
        if k.startswith("model."):
            stripped = k[len("model.") :]
            if stripped not in new_sd:
                new_sd[stripped] = v
            else:
                # prefer unstripped if both exist, keep first seen
                pass
        elif k.startswith("module."):
            stripped = k[len("module.") :]
            if stripped not in new_sd:
                new_sd[stripped] = v
            else:
                pass
        else:
            # avoid overwriting if same stripped key already added
            if k not in new_sd:
                new_sd[k] = v
    return new_sd, removed


def main():
    p = argparse.ArgumentParser(description="Clean checkpoint state_dict by removing unwanted prefixes (e.g. mifid).")
    p.add_argument("src", help="Source checkpoint path")
    p.add_argument("dst", nargs="?", help="Destination path (default: <src>.clean.ckpt)")
    p.add_argument(
        "--ignore",
        help="Comma-separated prefixes to remove from state_dict keys (default: mifid prefixes)",
        default=",".join(DEFAULT_IGNORE_PREFIXES),
    )
    args = p.parse_args()

    src = args.src
    dst = args.dst or src + ".clean.ckpt"
    ignore_prefixes = tuple(x.strip() for x in args.ignore.split(",") if x.strip())

    try:
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"ERROR: failed to load checkpoint {src}: {e}", file=sys.stderr)
        sys.exit(2)

    # possible keys that may hold state dicts
    sd_keys = ["state_dict", "model_state_dict", "state_dict_best", "state"]

    state_dict = None
    for k in sd_keys:
        if isinstance(ckpt.get(k, None), dict):
            state_dict = ckpt[k]
            sd_container_key = k
            break

    # fallback: if top-level looks like a state_dict (tensors as values)
    if state_dict is None and isinstance(ckpt, dict):
        # Heuristic: detect if values are tensors / numpy / lists -> treat as state_dict
        sample_vals = list(ckpt.values())[:5]
        if sample_vals and all(hasattr(v, "dtype") or isinstance(v, (torch.Tensor,)) for v in sample_vals):
            state_dict = ckpt
            sd_container_key = None

    if not isinstance(state_dict, dict):
        print("No state_dict found in checkpoint; nothing to do. Saving original checkpoint to destination.")
        torch.save(ckpt, dst)
        print(f"Saved to {dst}")
        return

    new_sd, removed = clean_state_dict(state_dict, ignore_prefixes)

    if not new_sd:
        print("WARNING: cleaned state_dict is empty. Aborting save.", file=sys.stderr)
        sys.exit(3)

    # place cleaned state dict back into checkpoint
    if sd_container_key is not None:
        ckpt[sd_container_key] = new_sd
    else:
        ckpt = new_sd

    try:
        torch.save(ckpt, dst)
    except Exception as e:
        print(f"ERROR: failed to save cleaned checkpoint {dst}: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"Saved cleaned checkpoint to: {dst}")
    print(f"Removed {len(removed)} keys (showing up to 20):")
    for key in removed[:20]:
        print("  -", key)


if __name__ == "__main__":
    main()