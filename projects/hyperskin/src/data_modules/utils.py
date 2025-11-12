from typing import Optional
import numpy as np
import torch


def _crop_with_bbox(img: np.ndarray, mask: np.ndarray, bbox_scale: float) -> Optional[np.ndarray]:
    """Crop image using mask bounding box with scaling."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return None

    h, w = img.shape[:2]
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    cy = (y_min + y_max) / 2
    cx = (x_min + x_max) / 2

    new_h = bbox_h * bbox_scale
    new_w = bbox_w * bbox_scale

    y_min = max(0, int(round(cy - new_h / 2)))
    y_max = min(h - 1, int(round(cy + new_h / 2)))
    x_min = max(0, int(round(cx - new_w / 2)))
    x_max = min(w - 1, int(round(cx + new_w / 2)))

    return img[y_min : y_max + 1, x_min : x_max + 1]
