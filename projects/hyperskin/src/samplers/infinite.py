import numpy as np
import torch.utils.data as data

def InfiniteSampler(n):
    """Data sampler"""
    # check if the number of samples is valid
    if n <= 0:
        raise ValueError(f"Invalid number of samples: {n}.\nMake sure that images are present in the given path.")
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
