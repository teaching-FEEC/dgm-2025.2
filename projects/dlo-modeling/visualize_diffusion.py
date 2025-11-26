import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

# --- Imports ---
try:
    from src.models.rope_diffusion import RopeDiffusion
    from src.data.rope_dataset import RopeDataset
    from src.utils import (
        load_and_split_data, 
        plot_model_comparison, 
        animate_rope
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure 'plot_model_comparison' and 'animate_rope' are in src/utils.py")
    sys.exit(1)

# --- Constants ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = 'src/data/rope_state_action_next_state_mil.npz'

# Model Hyperparameters (Must match training!)
SEQ_LEN = 70      # Depends on your data
ACTION_DIM = 4    # Depends on your data
DIFFUSION_STEPS = 50 
BASE_DIM = 64

def main():
    print(f"--- Loading Data & Model on {DEVICE} ---")

    # 1. Load Data
    (
        _, _, _, # Train ignored
        _, _, _, # Val ignored
        src_test_raw_np, act_test_raw_np, tgt_test_raw_np,
        USE_DENSE_ACTION, ACTION_DIM_LOADED, SEQ_LEN_LOADED
    ) = load_and_split_data(data_path=DATA_PATH, seed=SEED)

    # Convert to Tensor
    src_test = torch.tensor(src_test_raw_np, dtype=torch.float32)
    act_test = torch.tensor(act_test_raw_np, dtype=torch.float32)
    tgt_test = torch.tensor(tgt_test_raw_np, dtype=torch.float32)

    # Re-create Dataset for Normalization Stats
    # We need to load training stats to denormalize correctly
    # (In a real pipeline, save mean/std to a file. Here we re-calculate for simplicity)
    src_train_dummy = torch.zeros((10, SEQ_LEN_LOADED, 3)) # Dummy just to init class if needed, 
    # actually better to load the real train set to get correct Mean/Std if you didn't save them.
    # For now, let's load the real train split just to get the stats:
    src_train_np, _, _, _, _, _, _, _, _, _, _, _ = load_and_split_data(data_path=DATA_PATH, seed=SEED)
    src_train = torch.tensor(src_train_np, dtype=torch.float32)
    
    # Calculate stats
    train_mean = torch.mean(src_train, dim=(0, 1))
    train_std = torch.std(src_train, dim=(0, 1))
    print(f"Stats loaded: Mean={train_mean}, Std={train_std}")

    # Create Test Dataset
    test_ds = RopeDataset(
        rope_states=src_test, actions=act_test, 
        normalize=False, mean=train_mean, std=train_std,
        center_of_mass=False, dense=USE_DENSE_ACTION # Set center_of_mass=True if you trained with it
    )

    # 2. Load Model
    model = RopeDiffusion(
        seq_len=SEQ_LEN_LOADED,
        action_dim=ACTION_DIM_LOADED,
        n_steps=DIFFUSION_STEPS,
        base_dim=BASE_DIM
    ).to(DEVICE)

    # Path to your saved checkpoint
    # CHECK THIS PATH matches your saved file
    ckpt_path = "checkpoints_diffusion/standard/diffusion_best.pth" 
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Checking 'com_plus_standard' folder...")
        ckpt_path = "checkpoints_diffusion/com_plus_standard/diffusion_best.pth"
        
    print(f"Loading weights from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # Create output directory
    viz_dir = "visualizations_diffusion"
    os.makedirs(viz_dir, exist_ok=True)

    # --- VISUALIZATION 1: Static Comparison (3 Views) ---
    print("\n>>> Generating Static 3D Plots...")
    
    # plot_model_comparison expects a dictionary of models
    models_dict = {"Diffusion_50Step": model}
    
    # Pick 3 random samples
    indices = np.random.choice(len(test_ds), 3, replace=False)
    
    for i, idx in enumerate(indices):
        save_path = os.path.join(viz_dir, f"static_view_{i}_sample_{idx}.png")
        
        plot_model_comparison(
            models_dict=models_dict,
            dataset=test_ds,
            device=DEVICE,
            index=idx,
            denormalize=True,
            train_mean=train_mean,
            train_std=train_std,
            save_path=save_path,
            use_dense_action=USE_DENSE_ACTION
        )
        print(f"Saved static plot: {save_path}")

    # --- VISUALIZATION 2: Animation (Rollout) ---
    print("\n>>> Generating Video Animation (Autoregressive Rollout)...")
    print("Note: This is slow because Diffusion runs 50 loops per frame.")
    
    video_path = os.path.join(viz_dir, "diffusion_rollout.mp4")
    
    # We use animate_rope from your provided snippets
    # It will use the model to predict t+1, then use that prediction for t+2...
    anim = animate_rope(
        model=model,
        dataset=test_ds,
        start_idx=0,        # Start at beginning of test set
        steps=30,           # Number of frames (keep low for diffusion speed)
        interval=200,
        device=DEVICE,
        denormalize=True,
        save=True,
        save_path=video_path,
        train_mean=train_mean,
        train_std=train_std,
        teacher_forcing=False # False = Real Autoregressive Rollout (Harder)
    )
    
    print(f"Done. Visualizations saved to {viz_dir}/")

if __name__ == "__main__":
    main()