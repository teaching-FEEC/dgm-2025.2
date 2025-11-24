import torch
import torch.nn as nn
import math
from diffusers import DDPMScheduler

class RopeDiffusion(nn.Module):
    """
    Conditional Diffusion Model for Rope Dynamics.
    Predicts the noise added to state t+1, conditioned on state t and action t.
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        action_dim: int,
        dropout: float = 0.1,
        num_train_timesteps: int = 1000
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Scheduler (manages noise and timesteps)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2"
        )

        # --- Input Projections ---
        # Condition = Current_State (3) + Action (4 or more)
        # Input to model = Noisy_Target (3) + Condition
        cond_dim = 3 + action_dim
        self.input_fc = nn.Linear(3 + cond_dim, d_model)

        # --- Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # --- Backbone (Transformer) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Output Projection (Predict Noise) ---
        self.output_fc = nn.Linear(d_model, 3) 

    def forward(self, noisy_tgt, timesteps, src_state, action):
        """
        Forward pass for TRAINING (Noise Prediction).
        """
        B, L, _ = noisy_tgt.shape
        
        # 1. Expand dense action if needed: (B, 4) -> (B, L, 4)
        if action.dim() == 2:
            action = action.unsqueeze(1).expand(-1, L, -1)
            
        # 2. Concatenate inputs: (B, L, 3+3+4)
        # We treat (src_state, action) as the condition
        x = torch.cat([noisy_tgt, src_state, action], dim=-1)
        
        # Project to d_model
        x = self.input_fc(x) # (B, L, d_model)
        
        # 3. Add Time Embedding
        t_emb = self.time_mlp(timesteps) # (B, d_model)
        t_emb = t_emb.unsqueeze(1)       # (B, 1, d_model)
        x = x + t_emb                    # Broadcast add
        
        # 4. Run Transformer
        x = self.encoder(x)
        
        # 5. Predict Noise
        pred_noise = self.output_fc(x)
        return pred_noise

    @torch.no_grad()
    def sample(self, src_state, action, device):
        """
        Full denoising loop for INFERENCE.
        Starts with noise and iteratively removes it to find the next state.
        """
        B, L, _ = src_state.shape
        model_device = next(self.parameters()).device
        
        # Start with pure Gaussian noise
        curr_sample = torch.randn((B, L, 3), device=model_device)
        
        # Loop backwards: T -> ... -> 0
        for t in self.scheduler.timesteps:
            # Batch of current timestep
            timesteps = torch.full((B,), t, device=model_device, dtype=torch.long)
            
            # Predict noise at this step
            pred_noise = self(curr_sample, timesteps, src_state, action)
            
            # Remove a bit of noise (Scheduler Step)
            curr_sample = self.scheduler.step(pred_noise, t, curr_sample).prev_sample
            
        return curr_sample

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings