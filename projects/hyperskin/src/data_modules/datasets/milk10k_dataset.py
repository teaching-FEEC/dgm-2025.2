import os
import enum
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A

class MILK10kTask(enum.Enum):
    """Available tasks for MILK10k dataset."""

    MULTILABEL = "multilabel"
    MELANOMA_VS_NEVUS = "melanoma_vs_nevus"


class MILK10kDataset(Dataset):
    """
    PyTorch dataset for MILK10k skin lesion classification.

    Supports:
      - Multi-label prediction across 11 classes
      - Binary classification (melanoma=0 vs melanocytic_nevus=1)

    Args:
        root_dir (str): Root directory (e.g. 'data/MILK10k').
        transform (callable, optional): Transform applied to each image.
        target_transform (callable, optional): Transform applied to label.
        allowed_labels (list[str], optional): Filter subset of samples by
            their class names (snake_case, e.g. ['melanoma']).
        task (MILK10kTask): Task type. Defaults to MILK10kTask.MULTILABEL.
        return_metadata (bool): If True, returns (image, label, metadata).
    """

    def __init__(
        self,
        root_dir,
        transform: A.Compose | None = None,
        allowed_labels=None,
        task=MILK10kTask.MULTILABEL,
        return_metadata=False,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.allowed_labels = allowed_labels
        self.task = task
        self.return_metadata = return_metadata

        # CSV paths
        gt_path = os.path.join(root_dir, "MILK10k_Training_GroundTruth.csv")
        meta_path = os.path.join(root_dir, "MILK10k_Training_Metadata.csv")

        # Load dataframes
        self.gt_df = pd.read_csv(gt_path)
        self.meta_df = pd.read_csv(meta_path)

        # Merge on lesion_id
        self.data = pd.merge(self.meta_df, self.gt_df, on="lesion_id")

        # Construct absolute image paths
        self.data["image_path"] = self.data.apply(
            lambda row: os.path.join(
                root_dir, "images", row["lesion_id"], f"{row['isic_id']}.jpg"
            ),
            axis=1,
        )

        # Human-readable, snake_case class names
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

        # Optional filtering by allowed_labels
        if allowed_labels is not None:
            valid_set = set(self.class_names)
            invalid = [label for label in allowed_labels if label not in valid_set]
            if invalid:
                raise ValueError(
                    f"Invalid label(s) in allowed_labels: {invalid}\n"
                    f"Valid labels are: {sorted(valid_set)}"
                )

            allowed_codes = [
                code
                for code, name in self.class_map.items()
                if name in allowed_labels
            ]
            mask = self.data[allowed_codes].sum(axis=1) > 0
            self.data = self.data.loc[mask].reset_index(drop=True)

        # Validate binary task compatibility
        if task == MILK10kTask.MELANOMA_VS_NEVUS:
            # Keep only rows that are either melanoma or nevus
            mel_mask = self.data["MEL"] == 1
            nev_mask = self.data["NV"] == 1
            self.data = self.data.loc[mel_mask | nev_mask].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        # --- Label processing depending on task ---
        if self.task == MILK10kTask.MULTILABEL:
            label = row[self.class_codes].astype(float).values

        elif self.task == MILK10kTask.MELANOMA_VS_NEVUS:
            if row["MEL"] == 1:
                label = 0
            elif row["NV"] == 1:
                label = 1
            else:
                raise ValueError(
                    f"Unexpected class row for binary task in index {idx}"
                )
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        # --- Apply transform ---
        if self.transform:
            image = np.array(image)
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

if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # --- Multi-label (default) ---
    dataset_multilabel = MILK10kDataset(
        root_dir="data/MILK10k",
        transform=transform,
    )

    # --- Filtered + binary (melanoma vs nevus) ---
    dataset_binary = MILK10kDataset(
        root_dir="data/MILK10k",
        transform=transform,
        task=MILK10kTask.MELANOMA_VS_NEVUS,
    )

    print("Multilabel classes:", dataset_multilabel.class_names)
    print("Binary dataset size:", len(dataset_binary))

    # value counts for using the data dataframe
    print("Class distribution in binary dataset:")
    print(dataset_binary.data[["MEL", "NV"]].sum())
