import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from tqdm import tqdm
from .base_model import BaseRopeModel

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.Mish()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_dim, out_channels),
        )
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.Mish()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        time_emb = self.time_mlp(t)
        h += time_emb.unsqueeze(-1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.residual_conv(x)

class ConditionalUnet1D(nn.Module):
    def __init__(self, state_dim=3, action_dim=4, base_dim=64):
        super().__init__()
        input_channels = state_dim * 2 + action_dim 
        self.time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, self.time_dim),
            nn.Mish(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.init_conv = nn.Conv1d(input_channels, base_dim, 3, padding=1)
        self.down1 = ResidualBlock1D(base_dim, base_dim, self.time_dim)
        self.down2 = ResidualBlock1D(base_dim, base_dim*2, self.time_dim)
        self.down_sample = nn.Conv1d(base_dim*2, base_dim*2, 3, stride=2, padding=1)
        
        self.mid1 = ResidualBlock1D(base_dim*2, base_dim*2, self.time_dim)
        self.mid2 = ResidualBlock1D(base_dim*2, base_dim*2, self.time_dim)
        
        self.up_sample = nn.ConvTranspose1d(base_dim*2, base_dim*2, 4, stride=2, padding=1)
        self.up2 = ResidualBlock1D(base_dim*4, base_dim, self.time_dim)
        self.up1 = ResidualBlock1D(base_dim*2, base_dim, self.time_dim)
        self.final_conv = nn.Conv1d(base_dim, state_dim, 1)

    def forward(self, x, t, cond_state, cond_action):
        B, _, L = x.shape
        t = self.time_mlp(t)
        cond_action = cond_action.unsqueeze(-1).expand(-1, -1, L)
        net_in = torch.cat([x, cond_state, cond_action], dim=1)
        
        x1 = self.init_conv(net_in)
        x1 = self.down1(x1, t)
        x2 = self.down2(x1, t)
        x_mid = self.down_sample(x2)
        
        x_mid = self.mid1(x_mid, t)
        x_mid = self.mid2(x_mid, t)
        
        x_up = self.up_sample(x_mid)
        if x_up.shape[2] != x2.shape[2]:
             x_up = F.interpolate(x_up, size=x2.shape[2], mode='nearest')

        x_up = torch.cat([x_up, x2], dim=1)
        x_up = self.up2(x_up, t)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.up1(x_up, t)
        return self.final_conv(x_up)

class RopeDiffusion(BaseRopeModel):
    def __init__(
        self, 
        seq_len=70, 
        action_dim=4, 
        n_steps=50,       # Defaulting to your 50
        beta_start=0.0001, 
        beta_end=0.2,     # <--- INCREASED from 0.02 to 0.2
        base_dim=64
    ):
        super().__init__()
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.n_steps = n_steps
        
        self.model = ConditionalUnet1D(state_dim=3, action_dim=action_dim, base_dim=base_dim)
        
        # --- FIX: Stronger Noise Schedule for low steps ---
        # If steps < 100, we force beta_end to be high to ensure full signal destruction
        if n_steps <= 100 and beta_end < 0.1:
            print(f"Warning: Increasing beta_end to 0.2 for low step count ({n_steps})")
            beta_end = 0.2
            
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, n_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

    def forward(self, state, action, target_next_state=None, decoder_inputs=None):
        # Permute (B, L, C) -> (B, C, L)
        state_p = state.transpose(1, 2)
        
        if target_next_state is not None:
            # Training
            target_p = target_next_state.transpose(1, 2)
            B = state.shape[0]
            t = torch.randint(0, self.n_steps, (B,), device=state.device).long()
            noise = torch.randn_like(target_p)
            
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t][:, None, None]
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
            
            x_noisy = sqrt_alpha_t * target_p + sqrt_one_minus_alpha_t * noise
            noise_pred = self.model(x_noisy, t, state_p, action)
            return F.mse_loss(noise_pred, noise)
        else:
            # Inference
            return self.sample(state, action)

    @torch.no_grad()
    def sample(self, state, action):
        B = state.shape[0]
        device = state.device
        
        # Start from pure noise
        x = torch.randn(B, 3, self.seq_len, device=device)
        state_p = state.transpose(1, 2)
        
        for i in reversed(range(self.n_steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            predicted_noise = self.model(x, t, state_p, action)
            
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            
            # DDPM Update
            coeff1 = 1 / torch.sqrt(alpha)
            coeff2 = beta / torch.sqrt(1 - alpha_cumprod)
            
            mean = coeff1 * (x - coeff2 * predicted_noise)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta) 
                x = mean + sigma * noise
            else:
                x = mean
                
        return x.transpose(1, 2) # Return (B, L, C)

    def train_model(self, train_dataset, val_dataset, device, batch_size=256, epochs=10, lr=1e-3, checkpoint_path=None, **kwargs):
        from torch.utils.data import DataLoader
        import torch.optim as optim
        
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        best_val_loss = float("inf")

        print(f"Starting Diffusion Training ({self.n_steps} steps)...")

        for epoch in range(epochs):
            self.train()
            train_loss_acc = 0.0
            
            for src, action_map, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                src, action_map, tgt = src.to(device), action_map.to(device), tgt.to(device)
                optimizer.zero_grad()
                loss = self(src, action_map, target_next_state=tgt)
                loss.backward()
                optimizer.step()
                train_loss_acc += loss.item()
            
            avg_train_loss = train_loss_acc / len(train_loader)
            val_loss = self._evaluate_diffusion_loss(val_dataset, device, batch_size)
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f} | Val Loss={val_loss:.6f}")
            
            if checkpoint_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.state_dict(), checkpoint_path)
                print(f"✅ Saved checkpoint.")

    def _evaluate_diffusion_loss(self, dataset, device, batch_size):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for src, action_map, tgt in loader:
                src, action_map, tgt = src.to(device), action_map.to(device), tgt.to(device)
                loss = self(src, action_map, target_next_state=tgt)
                total_loss += loss.item()
        return total_loss / len(loader)