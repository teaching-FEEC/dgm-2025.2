import torch
import math
from torch import nn

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences (per-link positions).
    Input/outputs use shape (batch, seq_len, d_model).
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        return x + self.pos_emb[:, :x.size(1)]