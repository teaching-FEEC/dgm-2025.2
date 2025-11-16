import torch
from torch import nn
from .positional_encoding import PositionalEncoding # FIXED: Relative import
from .base_model import BaseRopeModel # FIXED: Relative import


class RopeBERT(BaseRopeModel):
    """
    Bidirectional Transformer (BERT-style) that maps (state_t, action_t) -> state_{t+1}.
    ...
    """

    def __init__(
        self,
        seq_len: int = 70,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_dense_action: bool = False,
        action_dim: int = 4, # Dim of sparse action_map features, e.g., (L, 4)
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_dense_action = use_dense_action

        # --- Input Projection ---
        # Project state (L, 3) -> (L, d_model)
        self.state_fc = nn.Linear(3, d_model)

        # --- Positional Encoding (using external class) ---
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=seq_len)

        # --- MODIFICATION: LayerNorm removed as requested ---
        # self.layernorm_in = nn.LayerNorm(d_model) 
        self.dropout_in = nn.Dropout(dropout)

        # --- Action Projections ---
        if use_dense_action:
            # Project (dx, dy, dz) vector
            self.action_xyz_fc = nn.Linear(3, d_model)
            # Embed link_id
            self.action_link_embedding = nn.Embedding(seq_len, d_model)
        else:
            # Project sparse (L, 4) map
            self.action_fc = nn.Linear(action_dim, d_model)

        # --- Bidirectional Transformer Encoder (BERT style) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Output Projection (delta prediction) ---
        self.final_norm = nn.LayerNorm(d_model)
        self.output_fc = nn.Linear(d_model, 3)

    def forward(self, src_state: torch.Tensor, action: torch.Tensor, decoder_inputs: torch.Tensor = None) -> torch.Tensor:
        """
        ...
        """
        B, L, _ = src_state.size()

        # 1. Project state
        state_emb = self.state_fc(src_state)  # (B, L, d_model)

        # 2. Project action and create (B, L, d_model) action embedding
        if self.use_dense_action:
            # Dense Path: (B, 4) -> (B, L, d_model)

            # (B, 4) -> (B, 3) and (B,)
            action_xyz = action[:, :3]
            action_link_id = action[:, 3].round().long().clamp(0, L - 1) # (B,)

            # (B, 3) -> (B, d_model)
            xyz_emb = self.action_xyz_fc(action_xyz)
            # (B,) -> (B, d_model)
            link_emb = self.action_link_embedding(action_link_id)

            # Combined action vector (B, d_model)
            action_vec = xyz_emb + link_emb

            # Create the sparse map: (B, L, d_model)
            action_emb = torch.zeros(B, L, self.d_model,
                                     device=src_state.device,
                                     dtype=state_emb.dtype)

            # Get batch indices for scattering
            batch_indices = torch.arange(B, device=src_state.device)

            # Scatter the action_vec into the map at the correct link_id
            action_emb[batch_indices, action_link_id, :] = action_vec

        else:
            # Sparse Path: (B, L, 4) -> (B, L, d_model)
            action_emb = self.action_fc(action)  # (B, L, d_model)

        # 3. Combine state and action
        # This tells the model "at this point, this action happened."
        x = state_emb + action_emb
        
        # --- MODIFICATION: LayerNorm removed as requested ---
        # x = self.layernorm_in(x) # Normalize the combined state+action

        # 4. Add positional embeddings
        x = self.pos_enc(x)
        x = self.dropout_in(x)

        # 5. Bidirectional encoding
        x = self.encoder(x) # (B, L, d_model)

        # 6. Refined Output Head
        x = self.final_norm(x)    # Normalize features (B, L, d_model)
        delta = self.output_fc(x) # Project to delta (B, L, 3)

        # 7. Apply residual connection
        pred_next_state = src_state + delta

        return pred_next_state