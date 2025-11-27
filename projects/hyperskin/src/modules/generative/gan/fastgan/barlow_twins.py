import torch
import torch.nn as nn

def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsProjector(nn.Module):
    """Projector head for Barlow Twins loss"""

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 2048):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )
        self.bn = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x):
        return self.projector(x)

    def compute_loss(self, z1, z2, lambd: float = 0.0051):
        """Compute Barlow Twins loss"""
        # Normalize
        z1 = self.bn(z1)
        z2 = self.bn(z2)

        # Cross-correlation matrix
        batch_size = z1.shape[0]
        c = (z1.T @ z2) / batch_size

        # Loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        return loss