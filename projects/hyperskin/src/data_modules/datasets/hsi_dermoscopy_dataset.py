from enum import IntEnum, auto
from typing import Optional

import numpy as np
import albumentations as A
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import loadmat
import torch
from PIL import Image

labels_map = {
    "dysplastic_nevi": 0,
    "melanoma": 1,
    "solar_lentigo": 2,
    "interepidermal_carcinoma": 3,
    "nevi": 4,
    "seborrheic_keratosis": 5,
    "melanocytic_lesion": 6
}

others_labels_map = {
    "IEC": "interepidermal_carcinoma",
    "Lentigo": "solar_lentigo",
    "Nv": "nevi",
    "SK": "seborrheic_keratosis",
    "Melt": "melanocytic_lesion"
}


class HSIDermoscopyTask(IntEnum):
    CLASSIFICATION_ALL_CLASSES = auto()
    CLASSIFICATION_MELANOMA_VS_OTHERS = auto()
    CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_VS_OTHERS = auto()
    CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_NEVI = auto()
    SEGMENTATION = auto()
    GENERATION = auto()


class HSIDermoscopyDataset(Dataset):
    def __init__(
        self,
        task: HSIDermoscopyTask,
        data_dir: str = "data/hsi_dermoscopy",
        transform: Optional[A.Compose] = None,
        force_create_df: bool = True,
        save_labels_df: bool = True,
        images_only: bool = False,
    ):
        self.transform = transform
        self.data_dir = data_dir
        self.task = task
        self.images_only = images_only

        self.labels_map = labels_map

        self.dir_path = Path(self.data_dir)
        self.labels_df_path = self.dir_path / "metadata.csv"

        if self.labels_df_path.exists() and not force_create_df:
            self.labels_df = pd.read_csv(self.labels_df_path)
        else:
            self.labels_df = self.create_df()

            if save_labels_df:
                self.labels_df.to_csv(self.labels_df_path, index=False)

        self.setup_labels_df()

    def find_masks(self, file_path: Path) -> Optional[str]:
        """
        Find all masks for a given image file.
        Returns semicolon-separated paths or None.
        """
        base_mask_path = str(file_path).replace("images", "masks").replace(
            ".mat", ""
        )
        masks = []

        # Try finding masks with _00, _01, etc. suffixes
        idx = 0
        while True:
            mask_path = f"{base_mask_path}_crop{idx:02d}_mask.png"
            if Path(mask_path).exists():
                masks.append(mask_path)
                idx += 1
            else:
                break

        # If no numbered masks found, try the exact name
        if not masks:
            mask_path = f"{base_mask_path}_mask.png"
            if Path(mask_path).exists():
                masks.append(mask_path)

        return ";".join(masks) if masks else None

    def get_masks_list(self, index: int) -> list[str]:
        """Get list of mask paths for a given index."""
        masks_str = self.labels_df.iloc[index]['masks']
        if pd.isna(masks_str):
            return []
        return masks_str.split(";")

    def setup_labels_df(self):
        if self.task == HSIDermoscopyTask.CLASSIFICATION_MELANOMA_VS_OTHERS:
            self.labels_df['label'] = self.labels_df['label'].apply(
                lambda x: 'melanoma' if x == 'melanoma' else 'others'
            )
            self.labels_map = {"melanoma": 0, "others": 1}
        elif self.task == HSIDermoscopyTask.CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_VS_OTHERS:
            def map_labels(x):
                if x == 'melanoma':
                    return 'melanoma'
                elif x == 'dysplastic_nevi':
                    return 'dysplastic_nevi'
                else:
                    return 'others'
            self.labels_df['label'] = self.labels_df['label'].apply(map_labels)
            self.labels_map = {"melanoma": 0, "dysplastic_nevi": 1, "others": 2}
        elif self.task == HSIDermoscopyTask.CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_NEVI:
            self.labels_df = self.labels_df[
                self.labels_df['label'].isin(['melanoma', 'dysplastic_nevi'])
            ].reset_index(drop=True)
            self.labels_map = {"melanoma": 0, "dysplastic_nevi": 1}
        elif self.task == HSIDermoscopyTask.CLASSIFICATION_ALL_CLASSES or \
             self.task == HSIDermoscopyTask.GENERATION:
            pass
        elif self.task == HSIDermoscopyTask.SEGMENTATION:
            # Filter out rows without masks
            nan_mask_rows = self.labels_df[self.labels_df['masks'].isna()]
            if not nan_mask_rows.empty:
                print(
                    f"Warning: {len(nan_mask_rows)} samples do not have masks "
                    "and will be ignored for segmentation task."
                )
            self.labels_df = self.labels_df.dropna(
                subset=['masks']
            ).reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    @property
    def num_classes(self):
        return len(labels_map)

    @property
    def label_df_indices(self) -> np.ndarray:
        return self.labels_df.index.to_numpy()

    @property
    def labels(self) -> np.ndarray:
        """Return integer labels for each sample, as a NumPy array."""
        return self.labels_df["label"].map(self.labels_map).to_numpy()

    def create_df(self):
        images_path = self.dir_path / "images"
        dysplastic_nevi_path = images_path / "DNCube"
        melanoma_path = images_path / "MMCube"
        other_lesions_path = images_path / "OtherCube"

        if not all([
            dysplastic_nevi_path.exists(),
            melanoma_path.exists(),
            other_lesions_path.exists()
        ]):
            raise FileNotFoundError(
                f"One or more data directories do not exist in {self.data_dir}"
            )

        data = []

        for lesion_type, path in [
            ("dysplastic_nevi", dysplastic_nevi_path),
            ("melanoma", melanoma_path)
        ]:
            for file in path.glob("*.mat"):
                masks = self.find_masks(file)
                data.append({
                    "file_path": str(file),
                    "label": lesion_type,
                    "masks": masks
                })

        for file in other_lesions_path.glob("*.mat"):
            filename = file.stem
            for short_label, full_label in others_labels_map.items():
                if short_label in filename:
                    masks = self.find_masks(file)
                    data.append({
                        "file_path": str(file),
                        "label": full_label,
                        "masks": masks
                    })
                    break

        return pd.DataFrame(data)

    def __getitem__(self, index):
        label = self.labels_map[self.labels_df.iloc[index]['label']]
        image = loadmat(self.labels_df.iloc[index]['file_path']).popitem()[-1]
        image = image.astype('float32')

        if self.task == HSIDermoscopyTask.SEGMENTATION or \
           self.task == HSIDermoscopyTask.GENERATION:
            mask_paths = self.get_masks_list(index)
            if not mask_paths:
                raise ValueError(f"No masks found for index {index}")

            # Load all masks
            masks = []
            for mask_path in mask_paths:
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask, dtype=np.uint8)
                masks.append(mask)

            # Combine masks using maximum (logical OR for binary masks)
            combined_mask = masks[0].copy()
            for mask in masks[1:]:
                combined_mask = np.maximum(combined_mask, mask)
            combined_mask = (combined_mask > 0).astype(np.uint8)

            if self.transform is not None:
                augmented = self.transform(image=image, mask=combined_mask)
                image = augmented['image']
                combined_mask = augmented['mask']

            combined_mask = torch.as_tensor(combined_mask, dtype=torch.long)
            label = torch.tensor(label, dtype=torch.long)
            if self.images_only:
                return image
            return image, combined_mask, label
        else:
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented['image']
            label = torch.tensor(label, dtype=torch.long)
            if self.images_only:
                return image
            return image, label

    def __len__(self):
        return len(self.labels_df)


if __name__ == "__main__":
    dataset = HSIDermoscopyDataset(
        task=HSIDermoscopyTask.GENERATION,
        data_dir="data/hsi_dermoscopy_croppedv2_256_with_masks"
    )

    for i in range(len(dataset)):
        image, mask, label = dataset[i]
        # print(f"Image shape: {image.shape}, Label: {label}")
        # print(f"Mask shape: {mask.shape}")
