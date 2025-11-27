import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from positional_encoding import LearnedPositionalEncoding, PositionalEncoding
from base_model import BaseRopeModel


class RopeTransformer(BaseRopeModel):
    """
    Transformer model that maps (state_t, action_t) -> state_{t+1}.

    Supports two modes:
    1.  use_dense_action=True:
        -   Architecture: TransformerDecoder (Decoder-Only).
        -   state_t (B, L, 3) -> 'tgt' sequence.
        -   action_t (B, 4) -> 'memory' sequence (B, 1, d_model).
        -   This uses cross-attention from state to action.

    2.  use_dense_action=False:
        -   Architecture: Standard Transformer (Encoder-Decoder).
        -   state_t (B, L, 3) -> 'src' sequence (Encoder input).
        -   action_t (B, L, 4) -> 'tgt' sequence (Decoder input).
    """

    def __init__(
        self,
        seq_len: int = 70,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3, # Used for sparse (original) mode
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        num_lstm_layers: int = 0, # Set to 0 to disable LSTM
        dropout: float = 0.1,
        use_dense_action: bool = False,
        action_dim: int = 4, # Dim of sparse action_map features, e.g., (L, 4)
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_dense_action = use_dense_action
        self._use_batch_first = True # Standardize on batch_first

        # --- Shared Components ---

        # --- FIX ---
        # Call the external PositionalEncoding class with the correct
        # (d_model, max_len) signature.
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=seq_len)
        # -----------

        self.output_fc = nn.Linear(d_model, 3)

        if num_lstm_layers > 0:
            self.bi_lstm = nn.LSTM(
                d_model, d_model // 2, num_lstm_layers,
                batch_first=True, bidirectional=True
            )
        else:
            self.bi_lstm = None

        # --- Architecture-Specific Components ---
        if use_dense_action:
            # --- 1. Dense Action (Cross-Attention) Architecture ---
            # Input projection for state (becomes decoder 'tgt')
            self.state_input_fc = nn.Linear(3, d_model)

            # Input projections for dense action (becomes 'memory')
            self.action_xyz_fc = nn.Linear(3, d_model)
            self.action_link_embedding = nn.Embedding(seq_len, d_model)

            # Decoder-only stack
            decoder_layer = nn.TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            )
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        else:
            # --- 2. Sparse Action (Encoder-Decoder) Architecture ---
            # (Based on your original code's logic)

            # Encoder input (sees state)
            self.enc_input_fc = nn.Linear(3, d_model)

            # Decoder input (sees sparse action map)
            self.dec_input_fc = nn.Linear(action_dim, d_model)

            # Standard Encoder-Decoder Transformer
            self.transformer = nn.Transformer(
                d_model, nhead, num_encoder_layers, num_decoder_layers,
                dim_feedforward, dropout, batch_first=True
            )

    def forward(self, src_state: torch.Tensor, action: torch.Tensor, decoder_inputs: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            src_state: (batch, L, 3)
            action:
                - (batch, 4) if use_dense_action=True
                - (batch, L, 4) if use_dense_action=False
            decoder_inputs: (batch, L, 3) - Ignored. src_state is used.

        Returns:
            pred_next_state: (batch, L, 3)
        """
        batch = src_state.size(0)

        if self.use_dense_action:
            # --- Dense Action (Decoder-Only) Path ---

            # 1. Prepare 'tgt' (from state)
            # (B, L, 3) -> (B, L, d_model)
            tgt_emb = self.state_input_fc(src_state) * math.sqrt(self.d_model)
            tgt_emb = self.pos_enc(tgt_emb)

            # 2. Prepare 'memory' (from action)
            # (B, 4) -> (B, 3) and (B,)
            action_xyz = action[:, :3]
            action_link_id = action[:, 3].round().long() # (B,)

            # (B, 3) -> (B, d_model)
            xyz_emb = self.action_xyz_fc(action_xyz)
            # (B,) -> (B, d_model)
            link_emb = self.action_link_embedding(action_link_id)

            # (B, d_model) -> (B, 1, d_model)
            memory = (xyz_emb + link_emb).unsqueeze(1)

            # 3. Run Transformer Decoder
            # out = (B, L, d_model)
            out = self.transformer_decoder(tgt=tgt_emb, memory=memory)

        else:
            # --- Sparse Action (Encoder-Decoder) Path ---

            # 1. Prepare 'src' (Encoder input from state)
            # (B, L, 3) -> (B, L, d_model)
            src_emb = self.enc_input_fc(src_state) * math.sqrt(self.d_model)
            src_emb = self.pos_enc(src_emb)

            # 2. Prepare 'tgt' (Decoder input from action map)
            # (B, L, 4) -> (B, L, d_model)
            tgt_emb = self.dec_input_fc(action) * math.sqrt(self.d_model)
            tgt_emb = self.pos_enc(tgt_emb)

            # 3. Run Transformer
            # out = (B, L, d_model)
            if self._use_batch_first:
                out = self.transformer(src_emb, tgt_emb)
            else:
                out = self.transformer(src_emb.permute(1, 0, 2), tgt_emb.permute(1, 0, 2)).permute(1, 0, 2)

        # --- Shared Output Path ---
        if self.bi_lstm is not None:
            out, _ = self.bi_lstm(out)

        delta = self.output_fc(out)  # (batch, L, 3)
        pred_next_state = src_state + delta

        return pred_next_state