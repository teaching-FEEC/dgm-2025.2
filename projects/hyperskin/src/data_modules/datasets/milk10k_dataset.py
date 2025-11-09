from enum import auto
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path

from zmq import IntEnum


class MILK10kTask(IntEnum):
    """Available tasks for MILK10k dataset."""

    MULTILABEL = auto()
    MELANOMA_VS_NEVUS = auto()
    SEGMENTATION = auto()
    GENERATION = auto()
    GENERATION_MELANOMA_VS_NEVUS = auto()


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
        images_only: bool = False,
        dermoscopic_only: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.task = task
        self.images_only = images_only
        self.return_metadata = return_metadata
        self.dermoscopic_only = dermoscopic_only

        # Load metadata and ground truth
        gt_path = self.root_dir / "MILK10k_Training_GroundTruth.csv"
        meta_path = self.root_dir / "MILK10k_Training_Metadata.csv"

        self.gt_df = pd.read_csv(gt_path)
        self.meta_df = pd.read_csv(meta_path)

        # Merge into single dataframe
        self.data = pd.merge(self.meta_df, self.gt_df, on="lesion_id")

        # filter dermoscopic images only if specified image_type=='dermoscopic'
        if self.dermoscopic_only:
            dermo_mask = self.data["image_type"] == "dermoscopic"
            self.data = self.data.loc[dermo_mask].reset_index(drop=True)

        # Construct absolute image paths and expand dataset for multiple crops
        expanded_rows = []

        for _, row in self.data.iterrows():
            lesion_dir = self.root_dir / "images" / row["lesion_id"]

            # Find all cropped variants (may include _crop00, _crop01, ...)
            crop_matches = sorted(lesion_dir.glob(f"{row['isic_id']}_crop*.jpg"))

            # Fallback to unsuffixed image if no cropped ones found
            if not crop_matches:
                unsuffixed = lesion_dir / f"{row['isic_id']}.jpg"
                if unsuffixed.exists():
                    row["image_path"] = str(unsuffixed)
                    expanded_rows.append(row.copy())
                continue

            for extra_img in crop_matches:
                new_row = row.copy()
                new_row["image_path"] = str(extra_img)
                expanded_rows.append(new_row)

        # Convert expanded row list back to DataFrame
        self.data = pd.DataFrame(expanded_rows).reset_index(drop=True)

        # add masks column
        self.data["masks"] = self.data["image_path"].apply(self.find_masks)

        # Drop rows with missing images (if any)
        missing_imgs = self.data[self.data["image_path"].isna()]
        if not missing_imgs.empty:
            print(f"Warning: {len(missing_imgs)} samples have missing image files and were removed.")
            self.data = self.data.dropna(subset=["image_path"]).reset_index(drop=True)


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
        if self.task == MILK10kTask.MELANOMA_VS_NEVUS or self.task == MILK10kTask.GENERATION_MELANOMA_VS_NEVUS:
            mel_mask = self.data["MEL"] == 1
            nev_mask = self.data["NV"] == 1
            self.data = self.data.loc[mel_mask | nev_mask].reset_index(drop=True)
            self.class_names = ["melanoma", "nevus"]

        # Filter segmentation task
        elif self.task == MILK10kTask.SEGMENTATION or self.task == MILK10kTask.GENERATION:
            no_mask_rows = self.data[self.data["masks"].isna()]
            if not no_mask_rows.empty:
                print(
                    f"Warning: {len(no_mask_rows)} samples have no masks and are ignored."
                )
            self.data = self.data.dropna(subset=["masks"]).reset_index(drop=True)

    # -------------------------------------------------------------

    def find_masks(self, image_path: str) -> str | None:
        """Find all associated masks for an image (with numbered suffixes)."""
        masks_dir = self.root_dir / "masks"
        if not masks_dir.exists():
            return None

        image_path = Path(image_path)
        isic_id = image_path.stem.split("_crop")[0]  # strip any _cropXX suffix
        base_path = masks_dir / image_path.parent.name

        # Search cropped masks (_cropXX_mask.png)
        masks = sorted(base_path.glob(f"{isic_id}_crop*_mask.png"))

        if not masks:
            # Fallback to unsuffixed mask
            plain_mask = base_path / f"{isic_id}_mask.png"
            if plain_mask.exists():
                masks = [plain_mask]

        return ";".join(map(str, masks)) if masks else None

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
        if self.task in [MILK10kTask.MULTILABEL, MILK10kTask.SEGMENTATION, MILK10kTask.GENERATION]:
            return row[self.class_codes].astype(float).values
        elif self.task == MILK10kTask.MELANOMA_VS_NEVUS or self.task == MILK10kTask.GENERATION_MELANOMA_VS_NEVUS:
            if row["MEL"] == 1:
                return 0
            elif row["NV"] == 1:
                return 1
            else:
                raise ValueError("Row not compatible with melanoma_vs_nevus task.")
        else:
            raise ValueError(f"Unknown task: {self.task}")

    @property
    def labels(self) -> np.ndarray:
        """Return integer labels for each sample, as a NumPy array."""
        labels_list = []
        for _, row in self.data.iterrows():
            labels_list.append(self._get_label(row))
        return np.array(labels_list)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        label = self._get_label(row)

        # --- Segmentation Task ---
        if self.task == MILK10kTask.SEGMENTATION or self.task == MILK10kTask.GENERATION or \
            self.task == MILK10kTask.GENERATION_MELANOMA_VS_NEVUS:
            mask_paths = self.get_masks_list(idx)
            if not mask_paths:
                raise ValueError(f"No masks found for {row['isic_id']}")

            masks = [np.array(Image.open(p).convert("L"), dtype=np.uint8) for p in mask_paths]
            combined_mask = masks[0].copy()
            for mask in masks[1:]:
                combined_mask = np.maximum(combined_mask, mask)
            combined_mask = (combined_mask > 0).astype(np.uint8)  # Binarize

            if self.transform:
                transformed = self.transform(image=image, mask=combined_mask)
                image = transformed["image"]
                combined_mask = transformed["mask"]
            else:
                image = torch.tensor(image).permute(2, 0, 1)

            if self.images_only:
                return image.float()

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
            if self.images_only:
                return image.float()
            return image.float(), torch.tensor(label, dtype=torch.long), metadata

        if self.images_only:
            return image.float()
        return image.float(), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # Example usage
    dataset = MILK10kDataset(
        root_dir="data/milk10k_melanoma_cropped_256",
        task=MILK10kTask.MULTILABEL,
        transform=A.Compose([
            A.Resize(224, 224),
            # A.Normalize(),
            A.pytorch.ToTensorV2(),
        ]),
    )

    # get global max and min values for normalization
    max_vals = np.ones((3,)) * -1e6
    min_vals = np.ones((3,)) * 1e6

    for i in range(len(dataset)):
        image, label = dataset[i]
        max_vals = np.maximum(max_vals, image.view(3, -1).max(dim=1).values.numpy())
        min_vals = np.minimum(min_vals, image.view(3, -1).min(dim=1).values.numpy())

    print("Global max values per channel:", max_vals)
    print("Global min values per channel:", min_vals)

