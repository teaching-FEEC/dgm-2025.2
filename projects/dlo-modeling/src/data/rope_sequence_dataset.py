import torch
import numpy as np
from torch.utils.data import Dataset

class RopeSequenceDataset(Dataset):
    """
    Dataset that yields sequences of states and actions for the Dreamer model.

    Returns:
    - states_seq: (T, L, 3)
    - actions_seq: (T, 4)

    Normalization:
    - Handled identically to the original RopeDataset.
    """

    def __init__(
        self,
        rope_states: "np.ndarray | torch.Tensor",
        actions: "np.ndarray | torch.Tensor",
        sequence_length: int = 50, # T: The chunk size for training
        normalize: bool = True,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        center_of_mass: bool = False,
    ):
        # Convert to tensor
        states = torch.as_tensor(rope_states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)

        assert states.ndim == 3 and states.shape[2] == 3, "rope_states must be (N, L, 3)"
        assert actions.ndim == 2 and actions.shape[1] >= 4, "actions must be (N, 4) or more"
        assert states.shape[0] == actions.shape[0], "Batch dimension must match"

        # We only need the first 4 action dims
        self.actions = actions[:, :4]
        self.states = states

        self.T = sequence_length
        self.N = states.shape[0]
        self.L = states.shape[1]
        self.center_of_mass = center_of_mass

        # --- ✅ Center-of-Mass Normalization (per-sample) ---
        if self.center_of_mass:
            com = self.states.mean(dim=1, keepdim=True)  # (N, 1, 3)
            self.states = self.states - com

        # --- ✅ Mean/Std Normalization ---
        if normalize:
            self.mean = self.states.mean(dim=(0, 1), keepdim=True)  # (1,1,3)
            self.std = self.states.std(dim=(0, 1), keepdim=True) + 1e-8
        else:
            if mean is not None and std is not None:
                self.mean = mean
                self.std = std
            else:
                # Set to no-op if no mean/std provided
                self.mean = torch.zeros(1, 1, 3, dtype=torch.float32)
                self.std = torch.ones(1, 1, 3, dtype=torch.float32)

        self.states = (self.states - self.mean) / self.std

    def __len__(self) -> int:
        # We can start a sequence from any point up to T steps from the end.
        return self.N - self.T + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sequence (chunk) of states and actions.

        idx: The *start* of the sequence.
        """
        end_idx = idx + self.T

        # S_idx ... S_{end_idx-1}
        states_seq = self.states[idx:end_idx]    # (T, L, 3)

        # A_idx ... A_{end_idx-1}
        actions_seq = self.actions[idx:end_idx]  # (T, 4)

        return states_seq, actions_seq