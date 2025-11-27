#!/usr/bin/env python3
import wandb
import argparse
import sys


def get_runs(entity: str, project: str, contains: str):
    """Fetch all runs from a W&B project that contain `contains` in their name."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    filtered = [
        run.id for run in runs if contains.lower() in (run.name or "").lower()
    ]
    return filtered


def save_run_ids(run_ids, outfile: str):
    """Save run IDs to a file (space-separated)."""
    if not run_ids:
        print("No runs found matching the criteria.")
        return

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(" ".join(run_ids))

    print(f"✅ Saved {len(run_ids)} run IDs to {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch W&B runs containing a substring in their name and save run IDs."
    )
    parser.add_argument("--entity", help="W&B entity (team or user)", default="k298976-unicamp")
    parser.add_argument("--project", help="W&B project name", default="hypersynth")
    parser.add_argument(
        "--contains", required=True, help="Substring to match in the run name"
    )
    parser.add_argument(
        "--out", default="run_ids.csv", help="Output filename (default: run_ids.csv)"
    )

    args = parser.parse_args()

    try:
        run_ids = get_runs(args.entity, args.project, args.contains)
        if run_ids:
            print(" ".join(run_ids))
        else:
            print("No matching run IDs found.")
        save_run_ids(run_ids, args.out)
    except wandb.CommError as e:
        print(f"❌ W&B API connection error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()