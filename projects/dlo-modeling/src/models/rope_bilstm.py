import torch
import torch.nn as nn
from base_model import BaseRopeModel

# (Assume BaseRopeModel and PositionalEncoding are imported)

class RopeBiLSTM(BaseRopeModel):
    """
    Encoder-decoder model for rope dynamics using only BiLSTMs.
    (state_t, action_t) -> state_{t+1}

    Supports two modes:
    1.  use_dense_action=True:
        -   action_t (B, 4) -> [dx, dy, dz, link_id]
        -   The state and dense action are projected separately.
        -   The action is "scattered" to its link_id and added to the
        -   state embedding, which is then fed to the BiLSTM.

    2.  use_dense_action=False:
        -   action_t (B, L, 4) -> Sparse map [dx, dy, dz, flag]
        -   The state and sparse map are concatenated (B, L, 7)
        -   and projected before being fed to the BiLSTM.
    """

    def __init__(
        self,
        seq_len: int = 70,
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_dense_action: bool = False,
        action_dim: int = 4, # Dim of sparse action_map features, e.g., (L, 4)
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.use_dense_action = use_dense_action

        # --- Model Components ---
        if use_dense_action:
            # State projection
            self.state_fc = nn.Linear(3, d_model)
            # Dense action projections
            self.action_xyz_fc = nn.Linear(3, d_model)
            self.action_link_embedding = nn.Embedding(seq_len, d_model)
            bilstm_input_size = d_model # BiLSTM sees the combined (state + action) embedding
        else:
            # Sparse input projection (state + action_map)
            # Original code had '6', correcting to 3 (state) + action_dim (4) = 7
            self.input_fc = nn.Linear(3 + action_dim, d_model)
            bilstm_input_size = d_model # BiLSTM sees the projected (state + action)

        self.bilstm = nn.LSTM(
            input_size=bilstm_input_size,
            hidden_size=d_model // 2,  # bidirectional ⇒ output = d_model
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # --- Output ---
        self.output_fc = nn.Linear(d_model, 3)

    def forward(
        self,
        src_state: torch.Tensor,    # (B, L, 3)
        action: torch.Tensor,       # (B, L, 4) or (B, 4)
        decoder_inputs: torch.Tensor = None, # (B, L, 3) - Ignored
    ) -> torch.Tensor:
        """
        Args:
            src_state: (batch, L, 3) - current rope state
            action:
                - (batch, 4) if use_dense_action=True
                - (batch, L, 4) if use_dense_action=False
            decoder_inputs: (batch, L, 3) - Ignored
        Returns:
            pred_next_state: (batch, L, 3)
        """
        B, L, _ = src_state.shape

        if self.use_dense_action:
            # --- Dense Action Path ---
            # 1. Project state
            state_emb = self.state_fc(src_state) # (B, L, d_model)

            # 2. Project dense action and scatter
            action_xyz = action[:, :3]
            action_link_id = action[:, 3].round().long().clamp(0, L - 1) # (B,)

            xyz_emb = self.action_xyz_fc(action_xyz) # (B, d_model)
            link_emb = self.action_link_embedding(action_link_id) # (B, d_model)
            action_vec = xyz_emb + link_emb # (B, d_model)

            # Create the (B, L, d_model) sparse map
            action_emb = torch.zeros(B, L, self.d_model,
                                     device=src_state.device,
                                     dtype=state_emb.dtype)
            batch_indices = torch.arange(B, device=src_state.device)
            action_emb[batch_indices, action_link_id, :] = action_vec

            # 3. Combine state and action embeddings
            in_data = state_emb + action_emb # (B, L, d_model)

        else:
            # --- Sparse Action Path ---
            # action is action_map (B, L, 4)
            src_cat = torch.cat([src_state, action], dim=-1)  # (B, L, 7)
            in_data = self.input_fc(src_cat) # (B, L, d_model)

        # --- Run BiLSTM ---
        bilstm_out, _ = self.bilstm(in_data)  # (B, L, d_model)

        # --- Delta prediction ---
        delta = self.output_fc(bilstm_out)  # (B, L, 3)

        # --- Residual prediction ---
        pred_next_state = src_state + delta

        return pred_next_state
