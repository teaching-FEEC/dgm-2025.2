import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    """
    Balanced batch sampler for binary classification.

    Each batch contains exactly 50/50 samples from each class.
    - If minority_class_size < majority_class_size, we reuse minority samples with replacement.
    - Majority-class samples rotate across batches so that eventually all are seen.

    Example:
        batch_size = 32
        → each batch has 16 + 16.
    """

    def __init__(self, labels: np.ndarray, batch_size: int, seed: int = 42):
        super().__init__(labels)
        assert batch_size % 2 == 0, "Batch size must be even for balanced batches."

        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.batch_half = batch_size // 2
        self.rng = np.random.default_rng(seed)

        # Indices per class
        unique_classes = np.unique(self.labels)
        if len(unique_classes) != 2:
            raise ValueError(
                "BalancedBatchSampler is implemented for binary classification only."
            )

        self.class_indices = {
            cls: np.where(self.labels == cls)[0] for cls in unique_classes
        }

    def __iter__(self):
        # shuffle class indices
        for cls in self.class_indices:
            self.rng.shuffle(self.class_indices[cls])

        # "cursors" to track where we are in each class
        cursors = {cls: 0 for cls in self.class_indices}

        while True:
            batch_indices = []
            for cls, indices in self.class_indices.items():
                start = cursors[cls]
                end = start + self.batch_half

                if end > len(indices):
                    # If we run out, reshuffle and wrap around
                    self.rng.shuffle(indices)
                    start = 0
                    end = self.batch_half
                chosen = indices[start:end]
                cursors[cls] = end
                batch_indices.extend(chosen)

            self.rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        # Roughly size of dataset / batch_size, limited by the minority class
        min_class_size = min(len(idxs) for idxs in self.class_indices.values())
        return (2 * min_class_size) // self.batch_size
