#!/usr/bin/env python3
import argparse
import wandb


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Get W&B run IDs by synthetic_data_dir values, grouped by model type."
        )
    )
    parser.add_argument(
        "--entity_project",
        default="k298976-unicamp/hypersynth",
        help="W&B entity/project string (e.g., 'username/project_name').",
    )
    parser.add_argument(
        "--synthetic_data_dirs",
        nargs="+",
        required=True,
        help="One or more synthetic_data_dir values to search for.",
    )
    args = parser.parse_args()

    api = wandb.Api()

    for data_dir in args.synthetic_data_dirs:
        # Server-side filter for efficiency
        runs = api.runs(
            args.entity_project,
            filters={"config.synthetic_data_dir": data_dir},
        )

        groups = {"isic2019": [], "densenet201": [], "other": []}

        for run in runs:
            cfg = run.config
            
            # if cfg is a string, try to parse it as a dict
            if isinstance(cfg, str) and "densenet201" in cfg:
                cfg = {"model_name": "densenet201"}
            if isinstance(cfg, str) and "isic2019" in cfg:
                cfg = {"isic2019_weights_path": "some_path"}
            
            if "isic2019_weights_path" in cfg:
                groups["isic2019"].append(run.id)
            elif cfg.get("model_name") == "densenet201":
                groups["densenet201"].append(run.id)
            else:
                groups["other"].append(run.id)

        print(f"\nSynthetic data dir: {data_dir}")
        print(f"  ISIC2019 model runs: {' '.join(groups['isic2019']) or '(none)'}")
        print(f"  DenseNet-201 model runs: {' '.join(groups['densenet201']) or '(none)'}")
        print(f"  Other runs: {' '.join(groups['other']) or '(none)'}")


if __name__ == "__main__":
    main()