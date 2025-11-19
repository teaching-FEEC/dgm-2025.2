import os
import argparse
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
import pandas as pd

from torchmetrics.image import (
    SpectralAngleMapper,
    RelativeAverageSpectralError,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    TotalVariation,
)
from torchmetrics.image.fid import FrechetInceptionDistance

import pyrootutils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    pythonpath=True,
    dotenv=True,
)

from src.utils.spectra_plot import MeanSpectraMetric
from src.data_modules.hsi_dermoscopy import HSIDermoscopyDataModule
from src.metrics.image_precision_recall import ImagePrecisionRecallMetric
from src.metrics.inception import InceptionV3Wrapper


# === Dataset helper ===
def get_hsi_dataset(data_dir: str, allowed_labels=None):
    image_size = 256
    datamodule = HSIDermoscopyDataModule(
        task="classification_melanoma_vs_dysplastic_nevi",
        data_dir=data_dir,
        train_val_test_split=(0.7, 0.15, 0.15),
        allowed_labels=allowed_labels,
        range_mode="-1_1",
        global_max=[
            0.6203158, 0.6172642, 0.46794897, 0.4325111, 0.4996644, 0.61997396,
            0.7382196, 0.86097705, 0.88304037, 0.9397393, 1.1892519, 1.5035477,
            1.4947973, 1.4737314, 1.6318618, 1.7226081,
        ],
        global_min=[
            0.00028473, 0.0043945, 0.00149752, 0.00167517, 0.00190101, 0.0028114,
            0.00394378, 0.00488099, 0.00257091, 0.00215704, 0.00797662, 0.01205248,
            0.01310135, 0.01476806, 0.01932094, 0.02020744,
        ],
        batch_size=1,
        transforms={
            "test": [
                {"class_path": "SmallestMaxSize", "init_args": {"max_size": image_size}},
                {
                    "class_path": "CenterCrop",
                    "init_args": {"height": image_size, "width": image_size},
                },
                {"class_path": "ToTensorV2", "init_args": {}},
            ],
        },
    )
    datamodule.prepare_data()
    datamodule.setup()
    dataset = datamodule.all_dataloader().dataset
    
    return dataset


# === Validation Module ===
class HSIValidationModule:
    def __init__(self, feature_extractor: nn.Module | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sam = SpectralAngleMapper().to(self.device)
        self.rase = RelativeAverageSpectralError().to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.tv = TotalVariation().to(self.device)

        self.inception_model = InceptionV3Wrapper(
            normalize_input=False, in_chans=16
        )
        if feature_extractor is None:
            feature_extractor = self.inception_model

        self.inception_model.eval()

        self.fid = FrechetInceptionDistance(
            self.inception_model,
            input_img_size=(16, 256, 256),
        ).to(self.device)
        self.fid.eval()

        self.precision_recall = ImagePrecisionRecallMetric(feature_extractor).to(
            self.device
        )
        
        self.mean_spectra_metric = MeanSpectraMetric().to(self.device)

    @torch.no_grad()
    def run_validation(self, real_dataset, fake_dataset, output_dir="validation_images",
                       batch_size=8):
        self.fid.reset()
        self.sam.reset()
        self.rase.reset()
        self.psnr.reset()
        self.ssim.reset()
        self.tv.reset()

        sam_vals, rase_vals, psnr_vals, ssim_vals, tv_vals = [], [], [], [], []

        real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
        fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

        n_batches = min(len(real_loader), len(fake_loader))
        print(f"Running validation over {n_batches} batches...")

        for real_batch, fake_batch in tqdm(
            zip(real_loader, fake_loader), total=n_batches, desc="Validation..."
        ):
            real_images = (
                real_batch["image"].to(self.device)
                if isinstance(real_batch, dict)
                else real_batch[0].to(self.device)
            )
            fake_images = (
                fake_batch["image"].to(self.device)
                if isinstance(fake_batch, dict)
                else fake_batch[0].to(self.device)
            )
            
            # assert that both real_images and fake_images are in the range [-1, 1]
            assert torch.min(real_images) >= -1.0 and torch.max(real_images) <= 1.0, f"Real images are not in the range [-1, 1]. Min: {torch.min(real_images)}, Max: {torch.max(real_images)}"
            assert torch.min(fake_images) >= -1.0 and torch.max(fake_images) <= 1.0, f"Fake images are not in the range [-1, 1]. Min: {torch.min(fake_images)}, Max: {torch.max(fake_images)}"
            
            # assert that there are negative values in both real_images and fake_images
            assert torch.min(real_images) < 0.0, "Real images do not contain negative values."
            assert torch.min(fake_images) < 0.0, "Fake images do not contain negative values."
            
            # === Optional: Extract masks and labels if they exist ===
            real_masks = real_batch.get("mask").to(self.device) if isinstance(real_batch, dict) and "mask" in real_batch else None
            fake_masks = fake_batch.get("mask").to(self.device) if isinstance(fake_batch, dict) and "mask" in fake_batch else None
            real_labels = real_batch.get("label").to(self.device) if isinstance(real_batch, dict) and "label" in real_batch else None
            fake_labels = fake_batch.get("label").to(self.device) if isinstance(fake_batch, dict) and "label" in fake_batch else None

            batch_size = min(real_images.shape[0], fake_images.shape[0])
            real_images, fake_images = real_images[:batch_size], fake_images[:batch_size]

            # FID + Precision / Recall require [-1, 1]
            self.fid.update(real_images, real=True)
            self.fid.update(fake_images, real=False)
            self.precision_recall.update(real_images, fake=False)
            self.precision_recall.update(fake_images, fake=True)

            # Normalize to [0, 1]
            real_images = (real_images + 1) / 2
            fake_images = (fake_images + 1) / 2
            real_images = real_images.clamp(0, 1)
            fake_images = fake_images.clamp(0, 1)

            sam_val = torch.nan_to_num(self.sam(fake_images, real_images), nan=0.0)
            rase_val = torch.nan_to_num(self.rase(fake_images, real_images), nan=0.0)
            psnr_val = torch.nan_to_num(self.psnr(fake_images, real_images), nan=0.0)
            ssim_val = torch.nan_to_num(self.ssim(fake_images, real_images), nan=0.0)
            tv_val = torch.nan_to_num(self.tv(fake_images), nan=0.0)

            sam_vals.append(sam_val.item())
            rase_vals.append(rase_val.item())
            psnr_vals.append(psnr_val.item())
            ssim_vals.append(ssim_val.item())
            tv_vals.append(tv_val.item())
            
            self.mean_spectra_metric.update(real_images, is_fake=False, masks=real_masks, labels=real_labels)
            self.mean_spectra_metric.update(fake_images, is_fake=True, masks=fake_masks, labels=fake_labels)

        fid_score = self.fid.compute().item()
        pr_scores = self.precision_recall.compute()
    
        # === Save Mean Spectra Plot ===
        os.makedirs(output_dir, exist_ok=True)
        fig = self.mean_spectra_metric.plot()
        if fig is not None:
            save_path = os.path.join(output_dir, f"{dataset_name}_mean_spectra.png")
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"✅ Mean spectra plot saved to: {save_path}")
        else:
            print("⚠️ No valid mean spectra plot could be generated.")

        return {
            "SAM_mean": np.mean(sam_vals),
            "SAM_std": np.std(sam_vals, ddof=1),
            "RASE_mean": np.mean(rase_vals),
            "RASE_std": np.std(rase_vals, ddof=1),
            "PSNR_mean": np.mean(psnr_vals),
            "PSNR_std": np.std(psnr_vals, ddof=1),
            "SSIM_mean": np.mean(ssim_vals),
            "SSIM_std": np.std(ssim_vals, ddof=1),
            "TV_mean": np.mean(tv_vals),
            "TV_std": np.std(tv_vals, ddof=1),
            "FID": fid_score,
            "Precision": pr_scores["precision"],
            "Recall": pr_scores["recall"],
        }


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate synthetic HSI datasets against real data."
    )
    parser.add_argument(
        "--real_data_dir",
        type=str,
        default="data/hsi_dermoscopy_croppedv2_256_with_masks",
    )
    parser.add_argument(
        "--fake_data_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of paths to synthetic datasets to evaluate.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="validation_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--csv_sep",
        type=str,
        default=" ",
        help="CSV separator (default: space).",
    )
    parser.add_argument(
        "--no_dataset_name",
        action="store_true",
        help="If specified, omit the dataset name column from CSV/output.",
    )
    args = parser.parse_args()


    validator = HSIValidationModule()

    all_results = []
    print("\n=== Starting validation for multiple fake datasets ===\n")
    
    for fake_dir in args.fake_data_dirs:
        dataset_name = os.path.basename(os.path.normpath(fake_dir))
        print(f"\n--- Evaluating synthetic dataset: {dataset_name} ---")

        fake_dataset = get_hsi_dataset(fake_dir)
        
        allowed_labels = None
        # get labels from fake dataset and check if it contains melanoma and dysplastic nevi
        # print the dataset class names
        # iterate over the fake dataset and collect all labels
        fake_labels = set()
        for i in range(len(fake_dataset)):
            item = fake_dataset[i]
            if 0 in fake_labels and 1 in fake_labels:
                break  # both labels found
            if isinstance(item, dict) and "label" in item:
                label = item["label"].item()
            else:
                label = item[1]  # assuming label is the second item
            fake_labels.add(label)
        
        # if fake dataset contains only one of the two labels, set allowed_labels accordingly
        if 0 in fake_labels and 1 not in fake_labels:
            allowed_labels = ["melanoma"]
        elif 1 in fake_labels and 0 not in fake_labels:
            allowed_labels = ["dysplastic_nevi"]
            
        real_dataset = get_hsi_dataset(args.real_data_dir, allowed_labels)
        results = validator.run_validation(real_dataset, fake_dataset)

        combined_metrics = {
            "Dataset": dataset_name,
            "SAM_mean": results["SAM_mean"],
            "SAM_std": results["SAM_std"],
            "RASE_mean": results["RASE_mean"],
            "RASE_std": results["RASE_std"],
            "PSNR_mean": results["PSNR_mean"],
            "PSNR_std": results["PSNR_std"],
            "SSIM_mean": results["SSIM_mean"],
            "SSIM_std": results["SSIM_std"],
            "TV_mean": results["TV_mean"],
            "TV_std": results["TV_std"],
            "FID": results["FID"],
            "Precision": results["Precision"],
            "Recall": results["Recall"],
        }

        if not args.no_dataset_name:
            combined_metrics = {"Dataset": dataset_name, **combined_metrics}

        all_results.append(combined_metrics)

    # Print summary
    table = [
        ["SAM", f"{results['SAM_mean']:.6f}", f"{results['SAM_std']:.6f}"],
        ["RASE", f"{results['RASE_mean']:.6f}", f"{results['RASE_std']:.6f}"],
        ["PSNR", f"{results['PSNR_mean']:.6f}", f"{results['PSNR_std']:.6f}"],
        ["SSIM", f"{results['SSIM_mean']:.6f}", f"{results['SSIM_std']:.6f}"],
        ["TV", f"{results['TV_mean']:.6f}", f"{results['TV_std']:.6f}"],
        ["FID", f"{results['FID']:.6f}", "-"],
        ["Precision", f"{results['Precision']:.6f}", "-"],
        ["Recall", f"{results['Recall']:.6f}", "-"],
    ]

    print(f"\n=== Validation Results for: {dataset_name} ===\n")
    print(tabulate(table, headers=["Metric", "Mean", "Std"], tablefmt="grid"))

    # === Save to CSV ===
    df = pd.DataFrame(all_results)

    # Convert all float columns to strings using comma as decimal separator
    for col in df.select_dtypes(include=["float", "float64", "float32"]).columns:
        df[col] = df[col].map(lambda x: f"{x:.6f}".replace(".", ","))

    df.to_csv(args.csv_path, sep=args.csv_sep, index=False)
    print(f"\n✅ Results saved to {args.csv_path} (decimal=','; separator='{args.csv_sep}')")