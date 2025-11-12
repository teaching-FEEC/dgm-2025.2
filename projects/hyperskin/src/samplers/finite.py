import numpy as np
import torch
from torch.utils.data import Sampler


class FiniteSampler(Sampler[int]):
    """Samples a fixed number of elements from the dataset."""

    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = min(num_samples, len(data_source))  # ensure within range

    def __iter__(self):
        # randomly sample without replacement
        indices = np.random.permutation(len(self.data_source))[: self.num_samples]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
