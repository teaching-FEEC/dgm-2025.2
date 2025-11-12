import torch
from torch import nn
from torchmetrics import Metric
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class ImagePrecisionRecallMetric(Metric):
    """
    Precision and Recall metric for evaluating synthetic image quality.

    This metric compares the distribution of features from real and generated
    images using manifold estimation.

    Parameters
    ----------
    feature_extractor : nn.Module
        A model that maps an image batch to a feature tensor of shape
        (batch_size, feature_dim).
    k : int, optional
        Number of nearest neighbors used for manifold estimation (default: 3).
    """

    full_state_update = False  # we manually handle batching

    def __init__(self, feature_extractor: nn.Module, k: int = 3):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.k = k

        # Buffers to store features across multiple updates
        self.add_state("real_features", default=[], dist_reduce_fx=None)
        self.add_state("fake_features", default=[], dist_reduce_fx=None)

    @torch.no_grad()
    def update(self, images: torch.Tensor, fake: bool = False):
        """
        Update the metric with a batch of images.

        Parameters
        ----------
        images: torch.Tensor
            Tensor of images with shape (B, C, H, W).
        fake: bool
            If True, the images are synthetic; otherwise, they are real.
        """
        self.feature_extractor.eval()

        image_feat = self.feature_extractor(images)

        # Detach and move to CPU to avoid increasing GPU memory over many batches
        if fake:
            self.fake_features.extend(image_feat.detach().cpu())
        else:
            self.real_features.extend(image_feat.detach().cpu())

    def compute(self) -> dict:
        """
        Compute the final precision and recall after all batches have been processed.
        """
        if len(self.real_features) == 0 or len(self.fake_features) == 0:
            raise ValueError("No features accumulated. Call update() with data first.")

        real_feats = torch.stack(list(self.real_features))
        fake_feats = torch.stack(list(self.fake_features))

        # Match the number of samples
        num_samples = min(len(real_feats), len(fake_feats))
        real_feats = real_feats[:num_samples]
        fake_feats = fake_feats[:num_samples]

        precision = self._manifold_estimate(real_feats, fake_feats, self.k)
        recall = self._manifold_estimate(fake_feats, real_feats, self.k)

        return {"precision": precision, "recall": recall}

    def _manifold_estimate(self, A_features: torch.Tensor, B_features: torch.Tensor, k: int) -> float:
        """
        Estimate manifold overlap for precision/recall computation.

        Parameters
        ----------
        A_features : torch.Tensor
            Reference features (N, D).
        B_features : torch.Tensor
            Comparison features (N, D).
        k : int
            Number of nearest neighbors.

        Returns
        -------
        float
            Ratio of B samples that fall within the manifold of A.
        """
        # Compute pairwise distances
        dist_matrix = torch.cdist(A_features, A_features, p=2)  # (N, N)
        kth_distances = dist_matrix.kthvalue(k + 1, dim=1).values  # skip self-distance
        thresholds = kth_distances.unsqueeze(0)  # shape (1, N)

        # Distance from each B to each A
        dist_BA = torch.cdist(B_features, A_features, p=2)

        # Check if each B lies within any A's k-NN radius
        within_manifold = (dist_BA <= thresholds).any(dim=1)
        return within_manifold.float().mean().item()
