import numpy as np
import torch
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

class NormalizeByMinMax(ImageOnlyTransform):
    def __init__(
        self,
        mins,
        maxs,
        range_mode="0_1",
        clip=True,
        always_apply=False,
        p=1.0,
    ):
        """
        Args:
            mins (list, np.ndarray, or torch.Tensor): Per-channel minimum values.
            maxs (list, np.ndarray, or torch.Tensor): Per-channel maximum values.
            range_mode (str): Either "0_1" for [0, 1] or "-1_1" for [-1, 1].
            clip (bool): If True, clip output to the chosen range.
            always_apply (bool): Whether to always apply this transform.
            p (float): Probability of applying the transform.
        """
        super().__init__(p)
        self.mins = torch.as_tensor(mins, dtype=torch.float32)
        self.maxs = torch.as_tensor(maxs, dtype=torch.float32)
        self.range_mode = range_mode
        self.clip = clip

        if self.mins.shape != self.maxs.shape:
            raise ValueError("mins and maxs must have the same length")

        if range_mode not in ("0_1", "-1_1"):
            raise ValueError("range_mode must be one of ['0_1', '-1_1']")

    def apply(self, img, **params):
        # Convert to tensor if not already
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        img = img.float()


        if img.shape[0] != len(self.mins):
            raise ValueError(
                f"Expected {len(self.mins)} channels, found {img.shape[0]}"
            )

        device = img.device
        mins = self.mins.to(device)
        maxs = self.maxs.to(device)

        denom = maxs - mins
        denom = torch.where(denom == 0, torch.tensor(1e-8, device=device), denom)

        norm_img = (img - mins[:, None, None]) / denom[:, None, None]

        if self.range_mode == "-1_1":
            norm_img = norm_img * 2.0 - 1.0

        if self.clip:
            min_val = -1.0 if self.range_mode == "-1_1" else 0.0
            norm_img = torch.clamp(norm_img, min_val, 1.0)

        # Convert back to (H, W, C)

        return norm_img
