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

    def export_dataset(
        self,
        output_dir: str,
        mode: str = "rgb",
        bands: Optional[list[int]] = None,
        crop_with_mask: bool = False,
        bbox_scale: float = 1.0,
        flat_export: bool = False,
        images_only: bool = False,
    ) -> None:
        """
        Export dataset samples (images & optionally masks).

        Modes:
        - "rgb": export as PNG images.
        - "hyperspectral": export as .mat files.

        Args:
            output_dir (str): Root directory where files will be saved.
            mode (str): "rgb" or "hyperspectral".
            bands (list[int], optional): Band indices to use.
            crop_with_mask (bool): If True, cropped around mask, skip mask export.
            bbox_scale (float): Scaling factor for bounding box (if cropping).
            flat_export (bool): If True, export to flat directory structure.
            images_only (bool): If True with flat_export, save only images directly
                                in output_dir, no masks. Ignored if flat_export=False.
        """
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        # Prepare subdirs if flat export and not images_only
        if flat_export and not images_only:
            images_root = output_root / "images"
            masks_root = output_root / "masks"
            images_root.mkdir(exist_ok=True, parents=True)
            masks_root.mkdir(exist_ok=True, parents=True)
        else:
            images_root = output_root
            masks_root = None  # unused if flat_export=False or images_only=True

        path_mapping = {}

        for idx, row in tqdm(
            self.labels_df.iterrows(),
            total=len(self.labels_df),
            desc=f"Exporting {mode}{' cropped' if crop_with_mask else ''}",
        ):
            cube = loadmat(row["file_path"]).popitem()[-1].astype("float32")

            # --- Convert to export mode ---
            if mode == "rgb":
                if bands is None:
                    band_data = np.mean(cube, axis=2, keepdims=True)
                    rgb = np.repeat(band_data, 3, axis=2)
                elif len(bands) == 1:
                    band_data = cube[:, :, bands[0:1]]
                    rgb = np.repeat(band_data, 3, axis=2)
                elif len(bands) == 3:
                    rgb = cube[:, :, bands]
                else:
                    raise ValueError(
                        "In RGB mode, bands must be None, or a list of 1 or 3 indices."
                    )

                rgb_min, rgb_max = rgb.min(), rgb.max()
                if rgb_max > rgb_min:
                    export_img = (
                        (rgb - rgb_min) / (rgb_max - rgb_min) * 255
                    ).astype("uint8")
                else:
                    export_img = np.zeros_like(rgb, dtype="uint8")

            elif mode == "hyperspectral":
                export_img = cube if not bands else cube[:, :, bands]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            use_img = export_img
            h, w = export_img.shape[:2]

            # --- Cropping if enabled ---
            if crop_with_mask:
                mask_path = row.get("mask", None)
                if mask_path and Path(mask_path).exists():
                    mask = np.array(Image.open(mask_path).convert("L"))
                    ys, xs = np.where(mask > 0)
                    if len(ys) > 0 and len(xs) > 0:
                        y_min, y_max = ys.min(), ys.max()
                        x_min, x_max = xs.min(), xs.max()

                        bbox_h = y_max - y_min + 1
                        bbox_w = x_max - x_min + 1
                        cy = (y_min + y_max) / 2
                        cx = (x_min + x_max) / 2

                        new_h, new_w = bbox_h * bbox_scale, bbox_w * bbox_scale
                        y_min = max(0, int(round(cy - new_h / 2)))
                        y_max = min(h - 1, int(round(cy + new_h / 2)))
                        x_min = max(0, int(round(cx - new_w / 2)))
                        x_max = min(w - 1, int(round(cx + new_w / 2)))

                        use_img = export_img[y_min:y_max + 1, x_min:x_max + 1]
                    else:
                        continue
                else:
                    continue

            # --- Build image path ---
            orig_rel_path = Path(row["file_path"]).relative_to(self.dir_path)
            orig_no_ext = orig_rel_path.with_suffix("")

            if flat_export:
                base_filename = f"{orig_no_ext.name}_{row['label']}"
                out_img_path = (
                    images_root / (base_filename + (".png" if mode == "rgb" else ".mat"))
                )
            else:
                out_img_path = (
                    images_root
                    / orig_no_ext.parent
                    / (orig_no_ext.name + (".png" if mode == "rgb" else ".mat"))
                )
                out_img_path.parent.mkdir(parents=True, exist_ok=True)

            # --- Save image ---
            if mode == "rgb":
                Image.fromarray(use_img).save(out_img_path)
            else:
                from scipy.io import savemat

                savemat(out_img_path, {"cube": use_img})

            path_mapping[str(out_img_path)] = str(orig_rel_path)

            # --- Save mask ---
            if (
                not crop_with_mask
                and row.get("mask", None)
                and not (flat_export and images_only)
            ):
                mask_path = Path(row["mask"])
                if mask_path.exists():
                    mask_img = Image.open(mask_path)
                    if flat_export:
                        image_filename = out_img_path.stem
                        mask_out_path = masks_root / (image_filename + "_mask.png")
                    else:
                        mask_rel = mask_path.relative_to(self.dir_path)
                        mask_out_path = output_root / mask_rel
                        mask_out_path.parent.mkdir(parents=True, exist_ok=True)
                    mask_img.save(mask_out_path)
                    path_mapping[str(mask_out_path)] = str(
                        mask_path.relative_to(self.dir_path)
                    )

        # --- Save mapping CSV ---
        mapping_file = output_root / "path_mapping.csv"
        pd.DataFrame.from_dict(
            path_mapping, orient="index", columns=["original_path"]
        ).to_csv(mapping_file)

        print(
            f"Exported {len(path_mapping)} files "
            f"to {output_root} (flat={flat_export}, cropped={crop_with_mask}, images_only={images_only})"
        )
        print(f"Saved path mapping to {mapping_file}")

if __name__ == "__main__":
    dataset = HSIDermoscopyDataset(
        task=HSIDermoscopyTask.CLASSIFICATION_ALL_CLASSES,
        data_dir="data/hsi_dermoscopy"
    )

    dataset.export_dataset(
        output_dir="export/hsi_dermoscopy_rgb",
        mode="rgb",
        crop_with_mask=False,
        bbox_scale=1.5,
        flat_export=True,
        images_only=False,
    )
