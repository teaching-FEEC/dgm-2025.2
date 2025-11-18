import wandb
import pandas as pd
from tqdm import tqdm

# Initialize API
api = wandb.Api()

# Replace with your entity/project
entity = "k298976-unicamp"
project = "hypersynth"

# Fetch runs (consider filtering if you have many runs)
runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
print(f"Found {len(runs)} runs")

results = []

for run in tqdm(runs, desc="Processing runs"):
    try:
        best_sens_spec = -float("inf")
        best_row_data = None

        # Stream history instead of loading all at once
        for row in run.scan_history(
            keys=["val/spec@sens=0.95", "val/f1", "_step"],
            page_size=1000,  # Process in chunks
        ):
            sens_spec = row.get("val/spec@sens=0.95")

            # Skip if metric is missing
            if sens_spec is None:
                continue

            # Track best sensitivity@specificity
            if sens_spec > best_sens_spec:
                best_sens_spec = sens_spec
                best_row_data = {
                    "best_step": row.get("_step"),
                    "best_val/spec@sens=0.95": sens_spec,
                    "val/f1_at_best": row.get("val/f1"),
                }

        # Only append if we found valid data
        if best_row_data:
            results.append({"run_id": run.id, "run_name": run.name, **best_row_data})

    except Exception as e:
        print(f"Error processing {run.name}: {e}")
        continue

# Convert to DataFrame
results_df = pd.DataFrame(results)
# Print top 10
print("\nTop 10 runs by val/f1_at_best:")
print(results_df.sort_values("val/f1_at_best", ascending=False).head(10))

# Save results
results_df.to_csv("wandb_best_metrics_summary.csv", index=False)
print(f"\nSaved {len(results_df)} runs to wandb_best_metrics_summary.csv")
