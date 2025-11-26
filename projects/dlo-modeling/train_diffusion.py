import torch
from torch import nn
import numpy as np
import sys
import os
import gc 

# --- 1. Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))
sys.path.append(os.path.join(current_dir, 'src', 'models'))

# --- 2. Imports ---
try:
    # Only importing the Diffusion model now
    from src.models.rope_diffusion import RopeDiffusion
    
    # Keep standard utils and data loaders
    from src.data.rope_dataset import RopeDataset 
    from src.utils import (
        set_seed, plot_model_comparison, load_and_split_data, cleanup_memory
    )
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Import error: {e}")
    sys.exit(1)

# --- 3. Constants ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
BATCH_SIZE = 256    
EPOCHS = 20         # Diffusion often needs more epochs, but starting with 20
LR = 1e-4           # Lower learning rate is usually better for Diffusion

# Diffusion specific parameters
DIFFUSION_STEPS = 50  # Lower = Faster training/inference, Higher = Better quality (standard is 1000)
BASE_DIM = 64         # Width of the U-Net

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    all_test_losses_normalized = {}
    all_test_losses_denormalized = {}
    all_test_losses_rollout = {} 
    
    # Data path
    data_path = 'src/data/rope_state_action_next_state_mil.npz'
    
    # 1. Load Data (Raw)
    try:
        (
            src_train_raw_np, act_train_raw_np, tgt_train_raw_np,
            src_val_raw_np,   act_val_raw_np,   tgt_val_raw_np,
            src_test_raw_np,  act_test_raw_np,  tgt_test_raw_np,
            USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN
        ) = load_and_split_data(data_path=data_path, seed=SEED)
        
    except Exception as e:
        print(f"\n---!! An error occurred while loading data: {e}")
        raise
        
    # Convert all raw numpy arrays to raw PyTorch tensors
    src_train_raw = torch.tensor(src_train_raw_np, dtype=torch.float32)
    act_train_raw = torch.tensor(act_train_raw_np, dtype=torch.float32)
    tgt_train_raw = torch.tensor(tgt_train_raw_np, dtype=torch.float32)
    
    src_val_raw = torch.tensor(src_val_raw_np, dtype=torch.float32)
    act_val_raw = torch.tensor(act_val_raw_np, dtype=torch.float32)
    
    src_test_raw = torch.tensor(src_test_raw_np, dtype=torch.float32)
    act_test_raw = torch.tensor(act_test_raw_np, dtype=torch.float32)
    tgt_test_raw = torch.tensor(tgt_test_raw_np, dtype=torch.float32)
    
    # --- Define Experiments ---
    # We keep the normalization experiments to see if Center of Mass helps Diffusion
    experiment_types = ['standard', 'com_plus_standard']

    for norm_type in experiment_types:
        print(f"\n=======================================================")
        print(f"  STARTING DIFFUSION EXPERIMENT: {norm_type.upper()}") 
        print(f"=======================================================\n")
        
        checkpoints_dir = os.path.join("checkpoints_diffusion", norm_type)
        os.makedirs(checkpoints_dir, exist_ok=True)

        # === 1. Prepare Data ===
        use_com = (norm_type == 'com_plus_standard')
        
        train_ds = RopeDataset(
            rope_states=src_train_raw, actions=act_train_raw, 
            normalize=True, center_of_mass=use_com, dense=USE_DENSE_ACTION
        )
        train_mean = train_ds.mean
        train_std = train_ds.std
        
        val_ds = RopeDataset(
            rope_states=src_val_raw, actions=act_val_raw, 
            normalize=False, mean=train_mean, std=train_std,
            center_of_mass=use_com, dense=USE_DENSE_ACTION
        )
        test_ds = RopeDataset(
            rope_states=src_test_raw, actions=act_test_raw, 
            normalize=False, mean=train_mean, std=train_std,
            center_of_mass=use_com, dense=USE_DENSE_ACTION
        )

        denorm_stats_for_rollout = (train_mean, train_std)

        # Initialize result containers
        norm_losses = {} 
        denorm_losses = {} 
        rollout_losses = {} 

        # === 2. Diffusion Training ===
        print(f"\n>>> Initializing RopeDiffusion...")
        
        model = RopeDiffusion(
            seq_len=SEQ_LEN,
            action_dim=ACTION_DIM,
            n_steps=DIFFUSION_STEPS,
            base_dim=BASE_DIM
        ).to(DEVICE)
        
        checkpoint_path = os.path.join(checkpoints_dir, "diffusion_best.pth")
        
        # Train (Using the custom loop inside RopeDiffusion)
        model.train_model(
            train_dataset=train_ds, 
            val_dataset=val_ds, 
            device=DEVICE,
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            lr=LR,
            checkpoint_path=checkpoint_path
        )
        
        # Reload best model for Evaluation
        print("Reloading best checkpoint for evaluation...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        
        # === 3. Evaluation ===
        print(f"Evaluating Diffusion (This involves sampling and is slower than standard models)...")
        
        # 1. Normalized Loss (Sampling vs Normalized GT)
        # Note: We use a smaller batch size here because sampling requires creating 
        # full-sized noise tensors which can eat VRAM.
        n_loss = model.evaluate_model(
            test_dataset=test_ds, device=DEVICE, batch_size=64,
            criterion=nn.MSELoss()
        )
        norm_losses["Diffusion"] = n_loss
        print(f"  -> Normalized Test Loss: {n_loss:.6f}")
        
        # 2. Denormalized Loss (Sampling vs Real World Coords)
        dn_loss = model.evaluate_model_denormalized(
            test_dataset=test_ds, device=DEVICE,
            train_mean=train_mean, train_std=train_std,
            batch_size=64, criterion=nn.MSELoss() 
        )
        denorm_losses["Diffusion"] = dn_loss
        print(f"  -> Denormalized Test Loss: {dn_loss:.6f}")
        
        # 3. Rollout
        # WARNING: Rollout is Step * Diffusion_Steps. 
        # e.g. 50 rollout steps * 50 diffusion steps = 2500 passes per sample.
        # We reduce the steps and num_rollouts here for sanity.
        print("Running Autoregressive Rollout...")
        r_loss = model.evaluate_autoregressive_rollout(
            test_src_tensor=src_test_raw, test_act_tensor=act_test_raw,
            test_tgt_tensor=tgt_test_raw, device=DEVICE, 
            steps=50,       # Reduced from 1000
            num_rollouts=20, # Reduced from 100
            criterion=nn.MSELoss(), denormalize_stats=denorm_stats_for_rollout
        )
        rollout_losses["Diffusion"] = r_loss
        print(f"  -> Rollout Loss: {r_loss:.6f}")

        # Store results
        all_test_losses_normalized[norm_type] = norm_losses
        all_test_losses_denormalized[norm_type] = denorm_losses
        all_test_losses_rollout[norm_type] = rollout_losses
        
        # === 4. Plotting ===
        if len(test_ds) > 0:
            print(f"\n--- Plotting Diffusion Predictions ({norm_type}) ---")
            plots_dir = os.path.join("comparisons_diffusion", norm_type)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Put model in dict for the plotter function
            model.eval()
            plotting_models = {"Diffusion": model}
            
            num_samples = min(5, len(test_ds))
            indices = np.random.choice(len(test_ds), num_samples, replace=False)
            
            with torch.no_grad():
                for i, idx in enumerate(indices, 1): 
                    filename = f"diffusion_sample_{i}.png"
                    plot_model_comparison(
                        models_dict=plotting_models, 
                        dataset=test_ds, 
                        device=DEVICE, index=idx, denormalize=True,        
                        train_mean=train_mean, train_std=train_std,      
                        save_path=os.path.join(plots_dir, filename),
                        use_dense_action=USE_DENSE_ACTION 
                    )
            
            # Cleanup for next loop
            del plotting_models

        # Cleanup Memory
        cleanup_memory(model)
                
    print("\n\nAll diffusion experiments finished.")
    print_loss_table(all_test_losses_normalized, "Final Normalized Test Loss")
    print_loss_table(all_test_losses_denormalized, "Final Denormalized Test Loss")
    print_loss_table(all_test_losses_rollout, "Final Rollout Test Loss")

def print_loss_table(all_test_losses, title):
    print(f"\n\n=======================================================")
    print(f"           {title}")
    print(f"=======================================================")
    
    if not all_test_losses:
        print("No test losses recorded.")
        return

    first_norm_type = list(all_test_losses.keys())[0]
    model_names = list(all_test_losses[first_norm_type].keys())
    
    header = "| Normalization |"
    separator = "|---|"
    for name in model_names:
        header += f" {name} Loss |"
        separator += "---|"
    print(header)
    print(separator)
    
    for norm_type, model_losses in all_test_losses.items():
        row = f"| {norm_type} |"
        for name in model_names:
            loss = model_losses.get(name)
            if isinstance(loss, float):
                row += f" {loss:.6f} |"
            elif isinstance(loss, str): 
                row += f" {loss} |"
            else:
                row += f" N/A |"
        print(row)

if __name__ == "__main__":
    main()