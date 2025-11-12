import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Sampler
from typing import Optional, Any
from collections.abc import Iterable

def InfiniteSampler(n: int):
    """Infinite index sampler over [0, n)."""
    if n <= 0:
        raise ValueError(
            f"Invalid number of samples: {n}.\n"
            "Make sure that images are present in the given path."
        )

    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteBalancedBatchSampler(Sampler[list[int]]):
    """
    Infinite balanced batch sampler for binary classification.

    Yields infinite sequence of balanced batches:
    - Each batch has 50/50 from each class.
    - Internally similar to BalancedBatchSampler, but no stopping.
    """

    def __init__(self, labels: np.ndarray, batch_size: int, seed: int = 42):
        super().__init__(labels)
        assert batch_size % 2 == 0, "Batch size must be even for balanced batches."

        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.batch_half = batch_size // 2
        self.rng = np.random.default_rng(seed)

        unique_classes = np.unique(self.labels)
        if len(unique_classes) != 2:
            raise ValueError(
                "InfiniteBalancedBatchSampler supports binary classification only."
            )

        self.class_indices: dict[Any, np.ndarray] = {
            cls: np.where(self.labels == cls)[0] for cls in unique_classes
        }

        # Initialize shuffled indices and cursors
        self._reset_state()

    def _reset_state(self) -> None:
        self.indices_shuffled: dict[Any, np.ndarray] = {}
        self.cursors: dict[Any, int] = {}
        for cls, idxs in self.class_indices.items():
            shuffled = np.array(idxs, copy=True)
            self.rng.shuffle(shuffled)
            self.indices_shuffled[cls] = shuffled
            self.cursors[cls] = 0

    def __iter__(self) -> Iterable[list[int]]:
        # Local copies so each iterator is independent / re-entrant
        rng = np.random.default_rng(self.rng.integers(0, 2**32 - 1))
        indices_shuffled = {}
        cursors = {}

        for cls, idxs in self.class_indices.items():
            shuffled = np.array(idxs, copy=True)
            rng.shuffle(shuffled)
            indices_shuffled[cls] = shuffled
            cursors[cls] = 0

        while True:
            batch_indices: list[int] = []
            for cls, indices in indices_shuffled.items():
                start = cursors[cls]
                end = start + self.batch_half

                if end > len(indices):
                    # Reshuffle and wrap
                    rng.shuffle(indices)
                    start = 0
                    end = self.batch_half

                chosen = indices[start:end]
                cursors[cls] = end
                batch_indices.extend(chosen)

            rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        # Arbitrary large number; DataLoader won't exhaust this.
        return 2 ** 31


class InfiniteSamplerWrapper(Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
