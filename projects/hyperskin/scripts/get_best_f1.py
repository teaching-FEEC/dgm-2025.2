import wandb
import pandas as pd

# 1️⃣ Initialize API
api = wandb.Api()

# 2️⃣ Replace with your own entity/project
entity = "k298976-unicamp"
project = "hypersynth"

# 3️⃣ Fetch runs
runs = api.runs(f"{entity}/{project}")

# 4️⃣ Iterate through runs and collect results
results = []

for run in runs:
    # download all metric history for this run
    history = run.history(samples=10000)  # adjust if you have many steps

    if 'val/sens@spec=0.95' not in history or 'val/f1' not in history:
        print(f"Skipping {run.name} — missing metric(s)")
        continue

    # find the step with the best val/sens@spec=0.95
    best_idx = history['val/sens@spec=0.95'].idxmax()
    best_row = history.loc[best_idx]

    best_step = best_row['_step']
    best_sens_spec = best_row['val/sens@spec=0.95']
    f1_at_best = best_row['val/f1']

    results.append({
        "run_id": run.id,
        "run_name": run.name,
        "best_step": best_step,
        "best_val/sens@spec=0.95": best_sens_spec,
        "val/f1_at_best": f1_at_best,
    })

# 5️⃣ Convert to a nice DataFrame and/or CSV
results_df = pd.DataFrame(results)

# print the first 10 rows, sorted by f1_at_best
print(results_df.sort_values(by="val/f1_at_best", ascending=False).head(10))
