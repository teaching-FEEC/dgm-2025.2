import torch
import torch.nn as nn
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Pfade setup
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('src/models'))

# Imports
from models.rope_diffusion import RopeDiffusion
from utils import (
    set_seed, load_data_from_npz, normalize_data, center_data, 
    plot_model_comparison, weighted_rope_loss
)

# --- Config ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Params 
D_MODEL = 384
NHEAD = 8
NUM_LAYERS = 6
DIM_FF = D_MODEL * 4
# Diffusion Specific
NUM_TIMESTEPS = 1000 

# Training Params
BATCH_SIZE = 32
EPOCHS = 100 
LR = 1e-4
PATIENCE = 10 # Early Stopping

class DiffusionWrapper(nn.Module):
    """
    Wrapper, damit plot_model_comparison 'model(src, action)' aufrufen kann,
    aber intern die sample()-Methode nutzt.
    """
    def __init__(self, diffusion_model, device):
        super().__init__()
        self.model = diffusion_model
        self.device = device
    
    def forward(self, src, action):
        return self.model.sample(src, action, self.device)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for src, action, tgt in tqdm(loader, desc="Training"):
        src, action, tgt = src.to(device), action.to(device), tgt.to(device)
        
        # 1. Sample Noise
        noise = torch.randn_like(tgt)
        
        # 2. Sample Random Timesteps
        B = src.shape[0]
        timesteps = torch.randint(
            0, model.scheduler.config.num_train_timesteps, 
            (B,), device=device
        ).long()
        
        # 3. Add Noise (Forward Process)
        noisy_tgt = model.scheduler.add_noise(tgt, noise, timesteps)
        
        # 4. Predict Noise
        pred_noise = model(noisy_tgt, timesteps, src, action)
        
        # 5. Loss (MSE auf Noise) - Standard für Diffusion Training
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, action, tgt in loader:
            src, action, tgt = src.to(device), action.to(device), tgt.to(device)
            
            # Validation Loss is also Noise MSE (Sampling is too slow here)
            noise = torch.randn_like(tgt)
            timesteps = torch.randint(
                0, model.scheduler.config.num_train_timesteps, 
                (src.shape[0],), device=device
            ).long()
            noisy_tgt = model.scheduler.add_noise(tgt, noise, timesteps)
            pred_noise = model(noisy_tgt, timesteps, src, action)
            
            loss = nn.functional.mse_loss(pred_noise, noise)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def evaluate_rollout(model, test_src, test_act, test_tgt, device, steps, num_rollouts, denorm_stats=None):
    """Autoregressive Rollout using Diffusion Sampling"""
    print(f"Starting Diffusion Rollout ({steps} steps, {num_rollouts} samples)...")
    model.eval()
    total_loss = 0
    
    # Sequential indices
    start_indices = range(min(num_rollouts, len(test_src) - steps))
    
    denorm = denorm_stats is not None
    if denorm:
        train_mean, train_std = denorm_stats
        train_mean, train_std = train_mean.to(device), train_std.to(device)
        eps = 1e-8

    with torch.no_grad():
        for i in tqdm(start_indices, desc="Rollout"):
            step_losses = []
            curr_state = test_src[i].unsqueeze(0).to(device)
            
            for k in range(steps):
                curr_action = test_act[i+k].unsqueeze(0).to(device)
                real_tgt = test_tgt[i+k].unsqueeze(0).to(device)
                
                # --- Sampling Step ---
                pred_next = model.sample(curr_state, curr_action, device)
                # ---------------------
                
                # Loss Calculation (Weighted MSE on State)
                if denorm:
                    pred_dn = pred_next * (train_std + eps) + train_mean
                    tgt_dn = real_tgt * (train_std + eps) + train_mean
                    src_dn = curr_state * (train_std + eps) + train_mean
                    loss = weighted_rope_loss(pred_dn, tgt_dn, src_dn, curr_action)
                else:
                    # Use weighted loss even for normalized data comparison
                    loss = weighted_rope_loss(pred_next, real_tgt, curr_state, curr_action)
                
                step_losses.append(loss.item())
                curr_state = pred_next # Auto-regressive update
            
            total_loss += np.mean(step_losses)
            
    return total_loss / len(start_indices)

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    try:
        (
            src_tr, act_tr, tgt_tr,
            src_val, act_val, tgt_val,
            src_test, act_test, tgt_test,
            USE_DENSE, ACT_DIM, SEQ_LEN
        ) = load_data_from_npz(seed=SEED)
    except Exception as e:
        print(f"Data load error: {e}"); sys.exit(1)

    # Raw Tensors
    src_tr, act_tr, tgt_tr = map(lambda x: torch.tensor(x, dtype=torch.float32), [src_tr, act_tr, tgt_tr])
    src_val, act_val, tgt_val = map(lambda x: torch.tensor(x, dtype=torch.float32), [src_val, act_val, tgt_val])
    src_test, act_test, tgt_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [src_test, act_test, tgt_test])

    # Experiments
    experiment_types = ['standard', 'com_plus_standard']
    
    for norm_type in experiment_types:
        print(f"\n=== DIFFUSION EXPERIMENT: {norm_type.upper()} ===")
        
        # Data Prep
        denorm_stats = None
        if norm_type == 'standard':
            (src_tr_n, tgt_tr_n, src_val_n, tgt_val_n, src_test_n, tgt_test_n, mean, std) = normalize_data(
                src_tr, tgt_tr, src_val, tgt_val, src_test, tgt_test
            )
            denorm_flag = True
            denorm_stats = (mean, std)
            
        elif norm_type == 'com_plus_standard':
            (src_tr_c, tgt_tr_c, src_val_c, tgt_val_c, src_test_c, tgt_test_c) = center_data(
                src_tr, tgt_tr, src_val, tgt_val, src_test, tgt_test
            )
            (src_tr_n, tgt_tr_n, src_val_n, tgt_val_n, src_test_n, tgt_test_n, mean, std) = normalize_data(
                src_tr_c, tgt_tr_c, src_val_c, tgt_val_c, src_test_c, tgt_test_c
            )
            denorm_flag = True
            denorm_stats = (mean, std)

        # Datasets
        train_ds = TensorDataset(src_tr_n, act_tr, tgt_tr_n)
        val_ds = TensorDataset(src_val_n, act_val, tgt_val_n)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Model Setup
        print("Instantiating RopeDiffusion...")
        model = RopeDiffusion(
            seq_len=SEQ_LEN,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FF,
            action_dim=ACT_DIM,
            num_train_timesteps=NUM_TIMESTEPS
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        
        # Paths
        ckpt_dir = os.path.join("checkpoints_diffusion", norm_type)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "diffusion_best.pth")
        
        # Training Loop
        best_val_loss = float('inf')
        no_improve = 0
        
        print("Starting Training...")
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
            val_loss = validate(model, val_loader, DEVICE)
            
            print(f"Epoch {epoch+1}: Train Loss (Noise MSE): {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved best model to {ckpt_path}")
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break
        
        # Evaluation
        print("Loading best model for evaluation...")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        
        # Run Rollout (Reduced samples because diffusion is slow)
        # 1000 steps, 20 rollouts (instead of 100)
        rollout_loss = evaluate_rollout(
            model, src_test_n, act_test, tgt_test_n, DEVICE, 
            steps=1000, num_rollouts=20, denorm_stats=denorm_stats
        )
        print(f"Final Rollout Loss ({norm_type}): {rollout_loss:.6f}")
        
        # Plotting
        plots_dir = os.path.join("comparisons_diffusion", norm_type)
        os.makedirs(plots_dir, exist_ok=True)
        
        # Wrap model for the plotter
        inference_model = DiffusionWrapper(model, DEVICE)
        models_dict = {"Diffusion": inference_model}
        
        # Simple test set for plotting (taking sequential samples)
        # Just reuse the test tensors for plotting index access
        plot_ds = TensorDataset(src_test_n, act_test, tgt_test_n)
        
        print("Plotting samples...")
        for i in range(5): # Plot first 5 samples of test set
            save_file = os.path.join(plots_dir, f"diffusion_sample_{i}.png")
            plot_model_comparison(
                models_dict=models_dict,
                dataset=plot_ds,
                device=DEVICE,
                index=i,
                denormalize=denorm_flag,
                train_mean=mean,
                train_std=std,
                save_path=save_file,
                use_dense_action=USE_DENSE
            )

if __name__ == "__main__":
    main()