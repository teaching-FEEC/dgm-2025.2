import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple

class RopeDataset(Dataset):
    """Dataset that yields (src_state, action_per_link, tgt_next_state).

    - src_state: (L, 3)
    - action_per_link: (L, 4)  -> (dx,dy,dz,flag)
    - tgt_next_state: (L, 3)

    Normalization:
    - When `normalize=True`, mean/std are computed from this dataset.
    - When `normalize=False`, use externally provided `mean` and `std` (from training set).
    - When `center_of_mass=True`, each rope configuration is centered by its own CoM.
    """

    def __init__(
        self,
        rope_states: "np.ndarray | torch.Tensor",
        actions: "np.ndarray | torch.Tensor",
        normalize: bool = True,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        center_of_mass: bool = False,
        dense: bool = False,
    ):
        # Convert to tensor
        states = torch.as_tensor(rope_states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)

        assert states.ndim == 3 and states.shape[2] == 3, "rope_states must be (N, L, 3)"
        assert actions.ndim == 2 and actions.shape[1] >= 4, "actions must be (N, 4) or more"
        assert states.shape[0] == actions.shape[0], "Batch dimension must match"

        self.center_of_mass = center_of_mass
        self.states = states
        self.actions = actions
        self.dense = dense
        self.N = states.shape[0]
        self.L = states.shape[1]

        if self.center_of_mass:
            # Subtract each sample's center of mass across all links
            com = self.states.mean(dim=1, keepdim=True)  # (N, 1, 3)
            self.states = self.states - com

        # --- Mean/Std Normalization ---
        if normalize:
            # Compute mean and std from this dataset
            self.mean = self.states.mean(dim=(0, 1), keepdim=True)  # (1,1,3)
            self.std = self.states.std(dim=(0, 1), keepdim=True) + 1e-8
        else:
            if mean is not None and std is not None:
              self.mean = mean
              self.std = std

        self.states = (self.states - self.mean) / self.std

    def __len__(self):
        # Predict t+1 from t
        return self.N - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = self.states[idx]          # (L, 3)
        tgt_next = self.states[idx + 1] # (L, 3)
        action = self.actions[idx]      # (4,)
        action_map = self._action_to_per_link(action) if not self.dense else torch.Tensor(action)
        return src, action_map, tgt_next

    def _action_to_per_link(self, action: torch.Tensor) -> torch.Tensor:
        """Convert action vector (4,) into a per-link map (L,4)."""
        amap = torch.zeros(self.L, 3, dtype=torch.float32)
        idx = int(round(float(action[3].item())))
        idx = max(0, min(self.L - 1, idx))
        amap[idx, :3] = action[:3]
        return amap
    
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

        # --- Center-of-Mass Normalization (per-sample) ---
        if self.center_of_mass:
            com = self.states.mean(dim=1, keepdim=True)  # (N, 1, 3)
            self.states = self.states - com

        # ---  Mean/Std Normalization ---
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