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
from tqdm import tqdm

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

# create HSIDermoscopyTask enum
class HSIDermoscopyTask(IntEnum):
    CLASSIFICATION_ALL_CLASSES = auto()
    CLASSIFICATION_MELANOMA_VS_OTHERS = auto()
    CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_VS_OTHERS = auto()
    CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_NEVI = auto()
    SEGMENTATION = auto()
    GENERATION = auto()


class HSIDermoscopyDataset(Dataset):
    def __init__(self, task: HSIDermoscopyTask,
                 data_dir: str = "data/hsi_dermoscopy",
                 transform: Optional[A.Compose] = None):
        self.transform = transform
        self.data_dir = data_dir
        self.task = task

        self.labels_map = labels_map

        self.dir_path = Path(self.data_dir)
        self.labels_df_path = self.dir_path / "metadata.csv"
        if not Path(self.data_dir).exists():
            self.labels_df = pd.read_csv(self.labels_df_path)
        else:
            self.labels_df = self.create_df()
            self.labels_df.to_csv(self.labels_df_path, index=False)

        self.setup_labels_df()

    def setup_labels_df(self):
        if self.task == HSIDermoscopyTask.CLASSIFICATION_MELANOMA_VS_OTHERS:
            self.labels_df['label'] = self.labels_df['label'].apply(
                lambda x: 'melanoma' if x == 'melanoma' else 'others')
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
            self.labels_df = self.labels_df[self.labels_df['label'].isin(
                ['melanoma', 'dysplastic_nevi'])].reset_index(drop=True)
            self.labels_map = {"melanoma": 0, "dysplastic_nevi": 1}
        elif self.task == HSIDermoscopyTask.CLASSIFICATION_ALL_CLASSES or self.task == HSIDermoscopyTask.GENERATION:
            pass
        elif self.task == HSIDermoscopyTask.SEGMENTATION:
            # get rows where mask is nan
            nan_mask_rows = self.labels_df[self.labels_df['mask'].isna()]
            if not nan_mask_rows.empty:
                print(f"Warning: {len(nan_mask_rows)} "
                      "samples do not have masks and will be ignored for segmentation task.")
            self.labels_df = self.labels_df.dropna(subset=['mask']).reset_index(drop=True)
            pass
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

        if not dysplastic_nevi_path.exists() or not melanoma_path.exists() or not other_lesions_path.exists():
            raise FileNotFoundError(f"One or more data directories do not exist in {self.data_dir}")

        data = []

        for lesion_type, path in [("dysplastic_nevi", dysplastic_nevi_path),
                                  ("melanoma", melanoma_path)]:
            for file in path.glob("*.mat"):
                # mask path is the same as file path but in masks directory and with .png extension
                mask_path = str(file).replace("images", "masks").replace(".mat", ".png")
                if not Path(mask_path).exists():
                    mask_path = None
                data.append({"file_path": str(file), "label": lesion_type, "mask": mask_path})

        for file in other_lesions_path.glob("*.mat"):
            filename = file.stem
            for short_label, full_label in others_labels_map.items():
                if short_label in filename:
                    # mask path is the same as file path but in masks directory and with .png extension
                    mask_path = str(file).replace("images", "masks").replace(".mat", ".png")
                    if not Path(mask_path).exists():
                        mask_path = None
                    data.append({"file_path": str(file), "label": full_label, "mask": mask_path})
                    break

        return pd.DataFrame(data)

    def __getitem__(self, index):

        label = self.labels_map[self.labels_df.iloc[index]['label']]

        image = loadmat(self.labels_df.iloc[index]['file_path']).popitem()[-1]
        image = image.astype('float32')


        if self.task == HSIDermoscopyTask.SEGMENTATION:
            mask_path = self.labels_df.iloc[index]['mask']
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask, dtype=np.uint8)
            if self.transform is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            # mask is already a tensor, just convert to long
            mask = torch.as_tensor(mask, dtype=torch.long)
            label = torch.tensor(label, dtype=torch.long)
            return image, mask, label
        else:
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented['image']
            label = torch.tensor(label, dtype=torch.long)
            return image, label

    def __len__(self):
        return len(self.labels_df)

if __name__ == "__main__":
    dataset = HSIDermoscopyDataset(
        task=HSIDermoscopyTask.CLASSIFICATION_MELANOMA_VS_DYSPLASTIC_NEVI,
        data_dir="data/hsi_dermoscopy_cropped_synth"
    )

    for i in range(len(dataset)):
        image, label = dataset[i]
        print(f"Image shape: {image.shape}, Label: {label}")