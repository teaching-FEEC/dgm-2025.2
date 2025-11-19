#!/usr/bin/env python3
import argparse
import re
import wandb
import pandas as pd


def get_best_metric_step(run, best_key="val/f1", maximize=True):
    """
    Finds the step where best_key (e.g., val/f1 or val/FID) is optimal.
    If maximize=True, chooses the highest value; otherwise, the lowest.
    Returns the step number, metric value, and all val/ metrics at that step.
    """
    history = run.history(samples=10000, pandas=True)

    if best_key not in history.columns:
        print(f"⚠️ Run {run.id} has no '{best_key}' metric.")
        return None

    if maximize:
        best_idx = history[best_key].idxmax()
        direction = "↑ max"
    else:
        best_idx = history[best_key].idxmin()
        direction = "↓ min"

    if pd.isna(best_idx):
        print(f"⚠️ Run {run.id} has NaN or empty values for '{best_key}'.")
        return None

    best_row = history.loc[best_idx]
    val_metrics = {k: v for k, v in best_row.items() if k.startswith("val/")}

    step = int(best_row["_step"]) if "_step" in best_row else best_idx
    best_value = val_metrics.get(best_key, None)

    print(f"Selected by {direction}: {best_key} = {best_value:.6f}")
    return step, best_value, val_metrics


def extract_run_ids_from_file(file_path):
    """Parses a file of W&B run URLs and extracts unique run IDs."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    run_ids = []
    for line in lines:
        match = re.search(r"/runs/([a-z0-9]+)/?", line)
        if match:
            run_ids.append(match.group(1))

    seen = set()
    run_ids = [r for r in run_ids if not (r in seen or seen.add(r))]
    return run_ids


def compute_specificity_from_prec_recall(precision, recall, prevalence):
    """
    Computes specificity when precision, recall, and prevalence are known.
    specificity = 1 - (recall * prevalence * (1 - precision) / precision) / (1 - prevalence)
    """
    if precision <= 0 or precision > 1 or recall < 0 or recall > 1:
        return None
    try:
        return 1 - (
            (recall * prevalence * (1 - precision) / precision)
            / (1 - prevalence)
        )
    except ZeroDivisionError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Finds the step with the best metric for given W&B runs, "
            "prints all val/* metrics, and saves a CSV summary."
        )
    )
    parser.add_argument(
        "--entity_project",
        default="k298976-unicamp/hypersynth",
        help="W&B entity/project string (e.g., 'username/project_name').",
    )
    parser.add_argument("--file", help="Path to file containing W&B run URLs.")
    parser.add_argument(
        "--best_key",
        default="val/f1",
        help="Metric key to maximize, e.g. 'val/f1' or 'val/FID'.",
    )
    parser.add_argument("--csv", default="results.csv", help="Output CSV path.")
    parser.add_argument(
        "--separator",
        default=" ",
        help="CSV field separator (default: space).",
    )
    parser.add_argument(
        "--decimal",
        default=",",
        help="Decimal format (default: ',').",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help=(
            "List of validation metrics to include in the CSV. "
            "If omitted, all discovered 'val/*' metrics will be included."
        ),
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        default=True,
        help="If set, maximize the metric (e.g. F1, accuracy).",
    )
    parser.add_argument("--no-maximize", dest="maximize", action="store_false")
    parser.add_argument(
        "run_ids",
        nargs="*",
        help="Optional: list of W&B run IDs (overrides --file).",
    )
    parser.add_argument(
        "--include_meta",
        action="store_true",
        help="Include 'run_id' and 'best_step' columns in the CSV.",
    )
    parser.add_argument(
        "--prevalence",
        type=float,
        default=0.674418605,
        help="Class prevalence (default: 0.674418605).",
    )
    args = parser.parse_args()

    # Determine run IDs
    run_ids = args.run_ids or (
        extract_run_ids_from_file(args.file) if args.file else None
    )
    if not run_ids:
        print("❌ No run IDs found. Provide --file or run IDs directly.")
        return

    entity, project = args.entity_project.split("/")[:2]
    api = wandb.Api()
    results = []
    discovered_metrics = set()

    for run_id in run_ids:
        print(f"\n=== Processing run: {run_id} ===")
        run = api.run(f"{entity}/{project}/{run_id}")
        result = get_best_metric_step(run, best_key=args.best_key, maximize=args.maximize)
        if result is None:
            continue

        step, best_value, val_metrics = result

        # --- Check or derive specificity ---
        if "val/specificity" not in val_metrics:
            prec = val_metrics.get("val/prec")
            rec = val_metrics.get("val/rec")
            if prec is not None and rec is not None:
                spec = compute_specificity_from_prec_recall(prec, rec, args.prevalence)
                if spec is not None:
                    val_metrics["val/specificity"] = spec
                    print(
                        f"🧮 Derived val/specificity = {spec:.6f} "
                        f"(from val/prec={prec:.4f}, val/rec={rec:.4f}, prevalence={args.prevalence})"
                    )
                else:
                    print("⚠️ Could not compute specificity (invalid values).")
            else:
                print("ℹ️ val/specificity, val/prec, or val/rec missing — cannot derive specificity.")

        print(f"Best step: {step}")
        print(f"{args.best_key}: {best_value:.6f}")

        for k, v in val_metrics.items():
            if k != args.best_key:
                print(f"  {k}: {v}")

        discovered_metrics.update(val_metrics.keys())

        row = {"run_id": run_id, "best_step": step}
        row.update(val_metrics)
        results.append(row)

    if not results:
        print("⚠️ No results to save.")
        return

    df = pd.DataFrame(results)

    # Determine final column order
    if args.columns:
        final_columns = args.columns
    else:
        final_columns = sorted([c for c in discovered_metrics if c.startswith("val/")])
        print(
            f"ℹ️ No --columns provided. Using discovered columns ({len(final_columns)}):"
        )
        for c in final_columns:
            print(f"  {c}")

    for key in final_columns:
        if key not in df.columns:
            df[key] = float("nan")

    if args.include_meta:
        ordered_columns = ["run_id", "best_step"] + final_columns
    else:
        ordered_columns = final_columns

    df = df[ordered_columns]

    print(f"Saving CSV to {args.csv} (sep='{args.separator}', decimal='{args.decimal}')")
    df.to_csv(args.csv, sep=args.separator, decimal=args.decimal, index=False)
    print("✅ CSV saved.")


if __name__ == "__main__":
    main()