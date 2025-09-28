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
                ['melanoma', 'dysplastic_nevi'])].reset_index(drop=True)   #importa apenas dados com labels melanoma ou dysplastic_nevi
            self.labels_map = {"melanoma": 0, "dysplastic_nevi": 1}
        elif self.task == HSIDermoscopyTask.CLASSIFICATION_ALL_CLASSES:
            pass
        elif self.task == HSIDermoscopyTask.SEGMENTATION:
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
                mask_path = str(file).replace("Cube", "masks").replace(".mat", ".png")
                data.append({"file_path": str(file), "label": lesion_type, "mask": mask_path})

        for file in other_lesions_path.glob("*.mat"):
            filename = file.stem
            for short_label, full_label in others_labels_map.items():
                if short_label in filename:
                    # mask path is the same as file path but in masks directory and with .png extension
                    mask_path = str(file).replace("OtherCube", "masks").replace(".mat", ".png")
                    data.append({"file_path": str(file), "label": full_label, "mask": mask_path})
                    break

        return pd.DataFrame(data)

    def __getitem__(self, index):

        label = self.labels_map[self.labels_df.iloc[index]['label']]

        image = loadmat(self.labels_df.iloc[index]['file_path']).popitem()[-1]
        image = image.astype('float32')

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.task == HSIDermoscopyTask.SEGMENTATION:
            mask_path = self.labels_df.iloc[index]['mask']
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask, dtype=np.uint8)
            mask = torch.tensor(mask, dtype=torch.long)
            return image, mask
        else:
            label = torch.tensor(label, dtype=torch.long)
            return image, label

    def __len__(self):
        return len(self.labels_df)

    def export_images(self, output_dir: str, bands: Optional[list[int]] = None) -> None:
        """
        Export all images in the dataset as PNGs.
        If bands is None, uses mean of all bands for grayscale.
        If bands has 1 value, repeats that band across RGB channels.
        If bands has 3 values, uses them as RGB channels.

        Args:
            output_dir (str): Directory where the PNGs will be saved.
            bands (list[int], optional): Band indices to use. Defaults to None.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create mapping dictionary
        path_mapping = {}

        for idx, row in tqdm(self.labels_df.iterrows(), total=len(self.labels_df), desc="Exporting images"):
            cube = loadmat(row["file_path"]).popitem()[-1].astype("float32")

            if bands is None:
                # Use mean of all bands
                band_data = np.mean(cube, axis=2, keepdims=True)
                rgb = np.repeat(band_data, 3, axis=2)
            elif len(bands) == 1:
                # Repeat single band across RGB channels
                try:
                    band_data = cube[:, :, bands[0:1]]
                    rgb = np.repeat(band_data, 3, axis=2)
                except IndexError:
                    raise ValueError(f"Band index {bands[0]} is out of range for cube {row['file_path']}")
            elif len(bands) == 3:
                # Use specified RGB bands
                try:
                    rgb = cube[:, :, bands]
                except IndexError:
                    raise ValueError(f"Band indices {bands} are out of range for cube {row['file_path']}")
            else:
                raise ValueError("bands must be None, or a list of 1 or 3 indices")

            # Normalize to 0–255
            rgb_min, rgb_max = rgb.min(), rgb.max()
            if rgb_max > rgb_min:
                rgb_norm = ((rgb - rgb_min) / (rgb_max - rgb_min) * 255).astype("uint8")
            else:
                rgb_norm = np.zeros_like(rgb, dtype="uint8")

            # Convert to PIL image and save
            img = Image.fromarray(rgb_norm)
            label = row["label"]
            filename = f"{Path(row['file_path']).stem}_{label}.png"
            img.save(output_path / filename)

            # Store mapping
            rel_path = Path(row["file_path"]).relative_to(self.dir_path)
            path_mapping[str(output_path / filename)] = str(rel_path)

        # Save mapping to file
        mapping_file = output_path / "path_mapping.csv"
        pd.DataFrame.from_dict(path_mapping, orient='index', columns=['original_path']).to_csv(mapping_file)

        print(f"Exported {len(self.labels_df)} images to {output_path}")
        print(f"Saved path mapping to {mapping_file}")


if __name__ == "__main__":
    dataset = HSIDermoscopyDataset(
        task=HSIDermoscopyTask.CLASSIFICATION_ALL_CLASSES,
        data_dir="data/hsi_dermoscopy"
    )

    dataset.export_images("exported_images_mean")
