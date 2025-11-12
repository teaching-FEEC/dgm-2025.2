from pathlib import Path
from typing import Optional, Any, Protocol
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat, savemat
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from src.utils.transform import smallest_maxsize_and_centercrop


class ExportableDataset(Protocol):
    """Protocol defining what a dataset needs to support export."""

    def get_masks_list(self, idx: int) -> list[str]:
        """Return list of mask paths for sample at index."""
        ...

    def __len__(self) -> int:
        ...


class DatasetExporter:
    """
    Generic dataset exporter that works with both RGB and hyperspectral data.

    Args:
        data_module: DataModule instance with split indices
        output_dir: Root directory for exported files
        data_type: "rgb" or "hyperspectral"
        file_extension: File extension for images (.png, .jpg, .mat)
        global_min: Min values for normalization (hyperspectral only)
        global_max: Max values for normalization (hyperspectral only)
    """

    def __init__(
        self,
        data_module: Any,
        output_dir: str,
        data_type: str = "rgb",
        file_extension: Optional[str] = None,
        global_min: Optional[float | list[float]] = None,
        global_max: Optional[float | list[float]] = None,
    ):
        self.data_module = data_module
        self.output_dir = Path(output_dir)
        self.data_type = data_type
        self.global_min = global_min
        self.global_max = global_max

        # Set default file extension
        if file_extension is None:
            self.file_extension = (
                ".mat" if data_type == "hyperspectral" else ".png"
            )
        else:
            self.file_extension = (
                file_extension if file_extension.startswith(".")
                else f".{file_extension}"
            )

    def export(
        self,
        splits: Optional[list[str]] = None,
        bands: Optional[list[int]] = None,
        crop_with_mask: bool = False,
        bbox_scale: float = 1.5,
        structure: str = "original",
        allowed_labels: Optional[list[int | str]] = None,
        image_size: Optional[int] = None,
        global_normalization: bool = False,
        export_cropped_masks: bool = True,
    ) -> None:
        """
        Export dataset with flexible options.

        Args:
            splits: List of splits to export (["train", "val", "test"])
            bands: Band indices to use (hyperspectral) or for RGB conversion
            crop_with_mask: Whether to crop using mask bounding boxes
            bbox_scale: Scale factor for bounding box
            structure: Directory structure ("original", "imagenet", "flat", etc.)
            allowed_labels: Filter by specific labels
            image_size: Resize images to this size
            global_normalization: Use global min-max normalization
            export_cropped_masks: Export cropped masks when crop_with_mask=True
        """
        if splits is None:
            splits = ["train", "val", "test"]

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get full dataset and mapping
        full_dataset = self._get_full_dataset()
        labels_map = self._get_labels_map()

        # Convert allowed_labels to integers
        allowed_label_ints = self._convert_labels_to_ints(
            allowed_labels, labels_map
        )

        # Export splits
        path_mapping = {}
        split_indices_map = self._get_split_indices()
        total_exported = 0

        for split_name in splits:
            if split_name not in split_indices_map:
                print(f"Warning: Unknown split '{split_name}', skipping")
                continue

            indices = split_indices_map[split_name]
            filtered_indices = self._filter_indices_by_labels(
                indices, full_dataset, allowed_label_ints
            )

            if len(filtered_indices) == 0:
                print(
                    f"Warning: No samples in '{split_name}' match "
                    f"allowed_labels={allowed_labels}"
                )
                continue

            counter = {"count": 0}

            for idx in tqdm(
                filtered_indices,
                desc=f"Exporting {split_name} split ({self.data_type} mode)",
            ):
                # Load image data
                image_data, original_path = self._load_image_data(
                    full_dataset, idx
                )
                label_name = self._get_label_name(full_dataset, idx, labels_map)

                # Get masks
                mask_paths = full_dataset.get_masks_list(idx)

                if crop_with_mask and mask_paths:
                    # Export cropped images
                    for mask_idx, mask_path in enumerate(mask_paths):
                        mask = np.array(Image.open(mask_path).convert("L"))

                        img_path, saved_mask_path = self._export_single_crop(
                            image_data,
                            mask,
                            split_name,
                            label_name,
                            counter,
                            bands,
                            bbox_scale,
                            image_size,
                            global_normalization,
                            original_path,
                            mask_idx,
                            structure,
                            export_cropped_masks,
                        )

                        if img_path:
                            path_mapping[str(img_path)] = original_path
                            if saved_mask_path and export_cropped_masks:
                                path_mapping[str(saved_mask_path)] = mask_path
                            total_exported += 1
                else:
                    # Export without cropping
                    img_path, mask_path = self._export_full_image(
                        image_data,
                        mask_paths,
                        split_name,
                        label_name,
                        counter,
                        bands,
                        image_size,
                        global_normalization,
                        original_path,
                        structure,
                    )

                    if img_path:
                        path_mapping[str(img_path)] = original_path
                        if mask_path and mask_paths:
                            path_mapping[str(mask_path)] = ";".join(mask_paths)
                        total_exported += 1

        # Save path mapping
        mapping_file = self.output_dir / "path_mapping.csv"
        pd.DataFrame.from_dict(
            path_mapping, orient="index", columns=["original_path"]
        ).to_csv(mapping_file)

        print(f"\nExported {total_exported} samples to {self.output_dir}")
        print(
            f"Structure: {structure}, Type: {self.data_type}, "
            f"Cropped: {crop_with_mask}"
        )
        if crop_with_mask and export_cropped_masks:
            print("Cropped masks: Exported")
        if allowed_labels:
            print(f"Filtered to labels: {allowed_labels}")
        print(f"Saved path mapping to {mapping_file}")

    def _get_full_dataset(self) -> Dataset:
        """Get the full dataset from the data module."""
        raise NotImplementedError("Subclass must implement _get_full_dataset")

    def _get_labels_map(self) -> dict[str, int]:
        """Get label name to integer mapping."""
        return self.data_module.get_labels_map()

    def _get_split_indices(self) -> dict[str, np.ndarray]:
        """Get indices for each split."""
        return {
            "train": self.data_module.train_indices,
            "val": self.data_module.val_indices,
            "test": self.data_module.test_indices,
        }

    def _load_image_data(
        self, dataset: Dataset, idx: int
    ) -> tuple[np.ndarray, str]:
        """
        Load image data and return (data_array, original_path).
        Must be implemented by subclass or detected automatically.
        """
        raise NotImplementedError("Subclass must implement _load_image_data")

    def _get_label_name(
        self, dataset: Dataset, idx: int, labels_map: dict[str, int]
    ) -> str:
        """Get string label name for an index."""
        raise NotImplementedError("Subclass must implement _get_label_name")

    def _convert_labels_to_ints(
        self, allowed_labels: Optional[list[int | str]], labels_map: dict[str, int]
    ) -> Optional[set[int]]:
        """Convert string labels to integer labels."""
        if allowed_labels is None:
            return None

        allowed_label_ints = set()
        for label in allowed_labels:
            if isinstance(label, str):
                if label in labels_map:
                    allowed_label_ints.add(labels_map[label])
                else:
                    raise ValueError(
                        f"Label '{label}' not found. "
                        f"Available: {list(labels_map.keys())}"
                    )
            else:
                allowed_label_ints.add(label)
        return allowed_label_ints

    def _filter_indices_by_labels(
        self,
        indices: np.ndarray,
        dataset: Dataset,
        allowed_label_ints: Optional[set[int]],
    ) -> np.ndarray:
        """Filter indices by allowed labels."""
        if allowed_label_ints is None:
            return indices

        filtered = []
        for idx in indices:
            label = self._get_label_int(dataset, idx)
            if label in allowed_label_ints:
                filtered.append(idx)
        return np.array(filtered)

    def _get_label_int(self, dataset: Dataset, idx: int) -> int:
        """Get integer label for an index."""
        raise NotImplementedError("Subclass must implement _get_label_int")

    def _process_image_data(
        self,
        image_data: np.ndarray,
        bands: Optional[list[int]],
        normalize: bool,
    ) -> np.ndarray:
        """Process image data based on type (RGB or hyperspectral)."""
        if self.data_type == "rgb":
            # RGB data: already in correct format
            if image_data.dtype != np.uint8:
                # Normalize to 0-255 if needed
                if image_data.max() <= 1.0:
                    image_data = (image_data * 255).astype(np.uint8)
            return image_data
        else:
            # Hyperspectral data
            if bands is None:
                processed = image_data
            else:
                processed = image_data[:, :, bands]

            if normalize and self.global_min is not None and self.global_max is not None:
                processed = self._global_normalization(processed)

            return processed

    def _convert_to_rgb(
        self, data: np.ndarray, bands: Optional[list[int]]
    ) -> np.ndarray:
        """Convert data to RGB format."""
        if self.data_type == "rgb":
            return data  # Already RGB

        # Hyperspectral to RGB conversion
        if bands is None:
            band_data = np.mean(data, axis=2, keepdims=True)
            rgb = np.repeat(band_data, 3, axis=2)
        elif len(bands) == 1:
            band_data = data[:, :, bands[0:1]]
            rgb = np.repeat(band_data, 3, axis=2)
        elif len(bands) == 3:
            rgb = data[:, :, bands]
        else:
            band_data = np.mean(data[:, :, bands], axis=2, keepdims=True)
            rgb = np.repeat(band_data, 3, axis=2)

        # Normalize to 0-255
        if self.global_min is not None and self.global_max is not None:
            rgb = self._global_normalization(rgb, clip_interval=(0, 1))
        else:
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        rgb = (rgb * 255).astype(np.uint8)
        return rgb

    def _global_normalization(
        self, data: np.ndarray, clip_interval: tuple[int, int] = (-1, 1)
    ) -> np.ndarray:
        """Apply global normalization."""
        if self.global_max is None or self.global_min is None:
            raise ValueError("Global max/min must be set for normalization")

        if isinstance(self.global_min, (int, float)) and isinstance(
            self.global_max, (int, float)
        ):
            data = (data - self.global_min) / (
                self.global_max - self.global_min + 1e-8
            )
            if clip_interval == (-1, 1):
                data = data * 2 - 1
        elif isinstance(self.global_min, list) and isinstance(
            self.global_max, list
        ):
            if len(self.global_min) != data.shape[2]:
                raise ValueError(
                    "Length of global_min/max must match number of bands"
                )
            for b in range(data.shape[2]):
                data[:, :, b] = (data[:, :, b] - self.global_min[b]) / (
                    self.global_max[b] - self.global_min[b] + 1e-8
                )
                if clip_interval == (-1, 1):
                    data[:, :, b] = data[:, :, b] * 2 - 1

        data = np.clip(data, clip_interval[0], clip_interval[1])
        return data.astype("float32")

    def _export_single_crop(
        self,
        image_data: np.ndarray,
        mask: np.ndarray,
        split_name: str,
        label_name: str,
        counter: dict,
        bands: Optional[list[int]],
        bbox_scale: float,
        image_size: Optional[int],
        global_normalization: bool,
        original_path: str,
        crop_idx: int,
        structure: str,
        export_cropped_masks: bool,
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Export a single cropped image."""
        # Get bounding box
        ys, xs = np.where(mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return None, None

        h, w = image_data.shape[:2]
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        bbox_h = y_max - y_min + 1
        bbox_w = x_max - x_min + 1
        cy = (y_min + y_max) / 2
        cx = (x_min + x_max) / 2

        new_h = bbox_h * bbox_scale
        new_w = bbox_w * bbox_scale

        y_min_crop = max(0, int(round(cy - new_h / 2)))
        y_max_crop = min(h - 1, int(round(cy + new_h / 2)))
        x_min_crop = max(0, int(round(cx - new_w / 2)))
        x_max_crop = min(w - 1, int(round(cx + new_w / 2)))

        # Crop
        cropped_data = image_data[
            y_min_crop : y_max_crop + 1, x_min_crop : x_max_crop + 1
        ]
        cropped_mask = mask[
            y_min_crop : y_max_crop + 1, x_min_crop : x_max_crop + 1
        ]

        if cropped_data is None or cropped_data.size == 0:
            return None, None

        counter["count"] += 1

        # Get paths
        img_path, mask_path = self._get_export_paths(
            structure,
            split_name,
            label_name,
            counter,
            original_path,
            crop_idx,
        )
        # Resize if needed
        if image_size is not None:
            cropped_data = smallest_maxsize_and_centercrop(cropped_data, image_size)
            if export_cropped_masks and mask_path is not None:
                cropped_mask_3d = np.expand_dims(cropped_mask, axis=-1)
                cropped_mask_3d = smallest_maxsize_and_centercrop(
                    cropped_mask_3d, image_size
                )
                cropped_mask = cropped_mask_3d.squeeze(-1)

        # Save based on data type
        self._save_image(
            cropped_data, img_path, bands, global_normalization
        )

        if export_cropped_masks and mask_path is not None:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((cropped_mask * 255).astype("uint8")).save(mask_path)

        return img_path, mask_path

    def _export_full_image(
        self,
        image_data: np.ndarray,
        mask_paths: list[str],
        split_name: str,
        label_name: str,
        counter: dict,
        bands: Optional[list[int]],
        image_size: Optional[int],
        global_normalization: bool,
        original_path: str,
        structure: str,
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Export full (non-cropped) image."""
        counter["count"] += 1

        # Resize if needed
        if image_size is not None:
            image_data = smallest_maxsize_and_centercrop(image_data, image_size)

        # Get paths
        img_path, mask_path = self._get_export_paths(
            structure,
            split_name,
            label_name,
            counter,
            original_path,
            None,
        )

        # Save image
        self._save_image(image_data, img_path, bands, global_normalization)

        # Save combined mask if applicable
        if (
            structure not in ["images_only"]
            and mask_paths
            and mask_path is not None
        ):
            masks = [
                np.array(Image.open(mp).convert("L")) for mp in mask_paths
            ]
            combined_mask = masks[0].copy()
            for mask in masks[1:]:
                combined_mask = np.maximum(combined_mask, mask)

            if image_size is not None:
                combined_mask = smallest_maxsize_and_centercrop(
                    combined_mask, image_size
                )

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(combined_mask).save(mask_path)

        return img_path, mask_path

    def _save_image(
        self,
        image_data: np.ndarray,
        img_path: Path,
        bands: Optional[list[int]],
        normalize: bool,
    ):
        """Save image based on data type."""
        img_path.parent.mkdir(parents=True, exist_ok=True)

        if image_data.shape[2] == 3 and self.data_type == "rgb":
            # Already RGB
            rgb_data = image_data
            Image.fromarray(rgb_data.astype("uint8")).save(img_path)
            return

        if self.data_type == "rgb" or self.file_extension in [".png", ".jpg"]:
            # Save as RGB
            rgb_data = self._convert_to_rgb(image_data, bands)
            Image.fromarray(rgb_data).save(img_path)
        else:
            # Save as hyperspectral .mat
            processed = self._process_image_data(image_data, bands, normalize)
            savemat(img_path, {"cube": processed})

    def _get_export_paths(
        self,
        structure: str,
        split_name: str,
        label_name: str,
        counter: dict,
        original_path: str,
        crop_idx: Optional[int],
    ) -> tuple[Path, Optional[Path]]:
        """Determine output paths based on structure."""
        if structure == "original":
            original_path_obj = Path(original_path)
            data_dir = Path(self.data_module.hparams.data_dir)

            try:
                rel_path = original_path_obj.relative_to(data_dir)
            except ValueError:
                rel_path = Path(*original_path_obj.parts[-3:])

            base_name = rel_path.stem
            if crop_idx is not None:
                filename = f"{base_name}_crop{crop_idx:02d}{self.file_extension}"
            else:
                filename = f"{base_name}{self.file_extension}"

            img_path = self.output_dir / rel_path.parent / filename
            img_path.parent.mkdir(parents=True, exist_ok=True)

            # remove images from the rel_path to get mask path
            rel_path_parts = list(rel_path.parent.parts)
            if "images" in rel_path_parts:
                images_index = rel_path_parts.index("images")
                rel_path_parts[images_index] = "masks"
                rel_path = Path(*rel_path_parts) / rel_path.name

            mask_filename = filename.replace(self.file_extension, "_mask.png")
            mask_path = (
            self.output_dir
            / rel_path.parent
            / mask_filename
            )
            mask_path.parent.mkdir(parents=True, exist_ok=True)

        elif structure == "imagenet":
            filename = f"{label_name}_{counter['count']:05d}{self.file_extension}"
            img_path = self.output_dir / split_name / label_name / filename
            img_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path = img_path.parent / filename.replace(
                self.file_extension, "_mask.png"
            )

        elif structure == "flat":
            filename = (
                f"{split_name}_{label_name}_{counter['count']:05d}"
                f"{self.file_extension}"
            )
            img_path = self.output_dir / filename
            mask_path = self.output_dir / filename.replace(
                self.file_extension, "_mask.png"
            )

        elif structure == "flat_with_masks":
            images_dir = self.output_dir / "images"
            masks_dir = self.output_dir / "masks"
            images_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)
            filename = (
                f"{split_name}_{label_name}_{counter['count']:05d}"
                f"{self.file_extension}"
            )
            img_path = images_dir / filename
            mask_path = masks_dir / filename.replace(
                self.file_extension, "_mask.png"
            )

        elif structure == "images_only":
            filename = (
                f"{split_name}_{label_name}_{counter['count']:05d}"
                f"{self.file_extension}"
            )
            img_path = self.output_dir / filename
            mask_path = None

        else:
            raise ValueError(f"Unknown structure: {structure}")

        return img_path, mask_path


class HSIDatasetExporter(DatasetExporter):
    """Exporter for HSI Dermoscopy hyperspectral data."""

    def __init__(self, data_module, output_dir: str, **kwargs):
        super().__init__(
            data_module,
            output_dir,
            data_type="hyperspectral",
            **kwargs,
        )

    def _get_full_dataset(self):
        from src.data_modules.datasets.hsi_dermoscopy_dataset import (
            HSIDermoscopyDataset,
        )

        return HSIDermoscopyDataset(
            task=self.data_module.hparams.task,
            data_dir=self.data_module.hparams.data_dir,
        )

    def _load_image_data(self, dataset, idx):
        row = dataset.labels_df.iloc[idx]
        cube = loadmat(row["file_path"]).popitem()[-1].astype("float32")
        return cube, str(row["file_path"])

    def _get_label_name(self, dataset, idx, labels_map):
        label_int = dataset.labels[idx]
        for name, val in labels_map.items():
            if val == label_int:
                return name
        return "unknown"

    def _get_label_int(self, dataset, idx):
        return dataset.labels[idx]


class RGBDatasetExporter(DatasetExporter):
    """Exporter for RGB datasets like MILK10k."""

    def __init__(self, data_module, output_dir: str, rgb_dataset_cls, **kwargs):
        # Override any hyperspectral-specific settings
        kwargs.pop("global_min", None)
        kwargs.pop("global_max", None)
        self.rgb_dataset_cls = rgb_dataset_cls
        super().__init__(
            data_module,
            output_dir,
            data_type="rgb",
            file_extension=".jpg",
            **kwargs,
        )

    def _get_full_dataset(self):
        return self.rgb_dataset_cls(
            root_dir=self.data_module.hparams.data_dir,
            task=self.data_module.hparams.task,
        )

    def _load_image_data(self, dataset, idx):
        row = dataset.data.iloc[idx]
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        return image, str(row["image_path"])

    def _get_label_name(self, dataset, idx, labels_map):
        row = dataset.data.iloc[idx]
        # For multilabel, use first positive class
        for code in dataset.class_codes:
            if row[code] == 1:
                return dataset.class_map[code]
        return "unknown"

    def _get_label_int(self, dataset, idx):
        row = dataset.data.iloc[idx]
        # Return first positive class index
        for i, code in enumerate(dataset.class_codes):
            if row[code] == 1:
                return i
        return -1
