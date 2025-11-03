import os
import enum
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path


class MILK10kTask(enum.Enum):
    """Available tasks for MILK10k dataset."""

    MULTILABEL = "multilabel"
    MELANOMA_VS_NEVUS = "melanoma_vs_nevus"
    SEGMENTATION = "segmentation"


class MILK10kDataset(Dataset):
    """
    PyTorch dataset for MILK10k skin lesion tasks:
      - Multi-label classification
      - Binary melanoma vs nevus
      - Segmentation (with per-lesion masks)

    Args:
        root_dir (str): Root directory (e.g. 'data/MILK10k')
        transform (callable, optional): Albumentations transform for image (and mask)
        task (MILK10kTask): One of the defined tasks.
        return_metadata (bool): If True, include metadata in return tuple.
    """

    def __init__(
        self,
        root_dir,
        transform: A.Compose | None = None,
        task=MILK10kTask.MULTILABEL,
        return_metadata=False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.task = task
        self.return_metadata = return_metadata

        # Load metadata and ground truth
        gt_path = self.root_dir / "MILK10k_Training_GroundTruth.csv"
        meta_path = self.root_dir / "MILK10k_Training_Metadata.csv"

        self.gt_df = pd.read_csv(gt_path)
        self.meta_df = pd.read_csv(meta_path)

        # Merge into single dataframe
        self.data = pd.merge(self.meta_df, self.gt_df, on="lesion_id")

        # Construct absolute image paths
        self.data["image_path"] = self.data.apply(
            lambda row: os.path.join(
                self.root_dir,
                "images",
                row["lesion_id"],
                f"{row['isic_id']}.jpg",
            ),
            axis=1,
        )

        # Add masks column (only relevant in segmentation mode)
        self.data["masks"] = self.data["isic_id"].apply(self.find_masks)

        # Class mapping
        self.class_map = {
            "AKIEC": "actinic_keratosis_intraepidermal_carcinoma",
            "BCC": "basal_cell_carcinoma",
            "BEN_OTH": "other_benign_proliferations",
            "BKL": "benign_keratinocytic_lesion",
            "DF": "dermatofibroma",
            "INF": "inflammatory_infectious_conditions",
            "MAL_OTH": "other_malignant_proliferations",
            "MEL": "melanoma",
            "NV": "melanocytic_nevus",
            "SCCKA": "squamous_cell_carcinoma_keratoacanthoma",
            "VASC": "vascular_lesions_hemorrhage",
        }

        self.class_codes = list(self.class_map.keys())
        self.class_names = [self.class_map[c] for c in self.class_codes]

        # Filter binary task
        if self.task == MILK10kTask.MELANOMA_VS_NEVUS:
            mel_mask = self.data["MEL"] == 1
            nev_mask = self.data["NV"] == 1
            self.data = self.data.loc[mel_mask | nev_mask].reset_index(drop=True)

        # Filter segmentation task
        elif self.task == MILK10kTask.SEGMENTATION:
            no_mask_rows = self.data[self.data["masks"].isna()]
            if not no_mask_rows.empty:
                print(
                    f"Warning: {len(no_mask_rows)} samples have no masks and are ignored."
                )
            self.data = self.data.dropna(subset=["masks"]).reset_index(drop=True)

    # -------------------------------------------------------------

    def find_masks(self, isic_id: str) -> str | None:
        """Find all associated masks for an image (with numbered suffixes)."""
        masks_dir = self.root_dir / "masks"
        if not masks_dir.exists():
            return None

        masks = []
        base_path = masks_dir / isic_id

        # Search for numbered masks (_00, _01, etc)
        idx = 0
        while True:
            mask_path = base_path.parent / f"{isic_id}_{idx:02d}.png"
            if mask_path.exists():
                masks.append(str(mask_path))
                idx += 1
            else:
                break

        # If none found, try unsuffixed
        if not masks:
            plain_mask = base_path.parent / f"{isic_id}.png"
            if plain_mask.exists():
                masks.append(str(plain_mask))

        return ";".join(masks) if masks else None

    def get_masks_list(self, index: int) -> list[str]:
        """Get list of mask paths for the sample at index."""
        val = self.data.iloc[index]["masks"]
        if pd.isna(val):
            return []
        return val.split(";")

    # -------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def _get_label(self, row):
        """Helper to extract per-task label."""
        if self.task in [MILK10kTask.MULTILABEL, MILK10kTask.SEGMENTATION]:
            return row[self.class_codes].astype(float).values
        elif self.task == MILK10kTask.MELANOMA_VS_NEVUS:
            if row["MEL"] == 1:
                return 0
            elif row["NV"] == 1:
                return 1
            else:
                raise ValueError("Row not compatible with melanoma_vs_nevus task.")
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        label = self._get_label(row)

        # --- Segmentation Task ---
        if self.task == MILK10kTask.SEGMENTATION:
            mask_paths = self.get_masks_list(idx)
            if not mask_paths:
                raise ValueError(f"No masks found for {row['isic_id']}")

            masks = [np.array(Image.open(p).convert("L"), dtype=np.uint8) for p in mask_paths]
            combined_mask = masks[0].copy()
            for mask in masks[1:]:
                combined_mask = np.maximum(combined_mask, mask)

            if self.transform:
                transformed = self.transform(image=image, mask=combined_mask)
                image = transformed["image"]
                combined_mask = transformed["mask"]
            else:
                image = torch.tensor(image).permute(2, 0, 1)

            return (
                image.float(),
                torch.as_tensor(combined_mask, dtype=torch.long),
                torch.tensor(label, dtype=torch.float),
            )

        # --- Classification Tasks ---
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if self.return_metadata:
            metadata = {
                "lesion_id": row["lesion_id"],
                "isic_id": row["isic_id"],
                "age_approx": row.get("age_approx"),
                "sex": row.get("sex"),
                "site": row.get("site"),
            }
            return image.float(), torch.tensor(label, dtype=torch.long), metadata

        return image.float(), torch.tensor(label, dtype=torch.long)
