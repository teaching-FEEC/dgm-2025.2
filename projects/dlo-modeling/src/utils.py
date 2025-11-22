import numpy as np
import torch
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import random
from models.base_model import BaseRopeModel

def center_data(*tensors):
    """
    Subtracts the center of mass (mean over seq_len) from each sample.
    Input tensors are (N, L, 3).
    """
    centered_tensors = []
    for t in tensors:
        if t is None:
            centered_tensors.append(None)
            continue
        # Calculate CoM per-sample: (N, L, 3) -> (N, 1, 3)
        com = torch.mean(t, dim=1, keepdim=True)
        centered_tensors.append(t - com)
        
    if len(centered_tensors) == 1:
        return centered_tensors[0]
    return tuple(centered_tensors)


def normalize_data(src_train, tgt_train, src_val, tgt_val, src_test, tgt_test):
    """
    Normalizes state data using statistics from the training set.
    """
    print("Normalizing data with mean/std...")
    train_mean = torch.mean(src_train, dim=(0, 1)) # Shape (3,)
    train_std = torch.std(src_train, dim=(0, 1))   # Shape (3,)
    epsilon = 1e-8
    
    src_train_norm = (src_train - train_mean) / (train_std + epsilon)
    tgt_train_norm = (tgt_train - train_mean) / (train_std + epsilon)
    src_val_norm = (src_val - train_mean) / (train_std + epsilon)
    tgt_val_norm = (tgt_val - train_mean) / (train_std + epsilon)
    src_test_norm = (src_test - train_mean) / (train_std + epsilon)
    tgt_test_norm = (tgt_test - train_mean) / (train_std + epsilon)
    
    print(f"Calculated Train Mean: {train_mean.numpy()}")
    print(f"Calculated Train Std: {train_std.numpy()}")
    
    return (
        src_train_norm, tgt_train_norm,
        src_val_norm, tgt_val_norm,
        src_test_norm, tgt_test_norm,
        train_mean, train_std
    )


def load_and_split_data(
    data_path,
    seed=42, 
    create_demo_set=False, 
    demo_size=100,
    train_ratio=0.8,
    val_ratio=0.1,
):
    """
    Loads data from NPZ, detects dimensions, and splits into Train/Val/Test.
    Conditionally creates and returns a sequential Demo set.

    Args:
        seed (int): Random seed for reproducibility.
        create_demo_set (bool): If True, reserves the last N items for a demo set and returns it.
        demo_size (int): Number of items to reserve for the demo set.
        train_ratio (float): Proportion of non-demo data used for training.
        val_ratio (float): Proportion of non-demo data used for validation.

    Returns:
        If create_demo_set is True:
            (train_set, val_set, test_set, demo_set, USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN)
        
        If create_demo_set is False:
            (train_set, val_set, test_set, USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN)
            
        *Each set is a tuple: (src_data, act_data, tgt_data)
    """
    print("Loading data from NPZ...")
    
    # === Path Definition ===
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print(f"Please check that the file exists in the '{''.join(data_path.split('/')[:-1])}' folder.")
        sys.exit(1)
        
    data = np.load(data_path, allow_pickle=True)
    
    # === Key Extraction ===
    states_key = 'states'          
    actions_key = 'actions'        
    next_states_key = 'next_states' 

    try:
        src_data = data[states_key]
        act_data = data[actions_key]
        tgt_data = data[next_states_key]
    except KeyError as e:
        print(f"Error: Key {e} not found in {data_path}.")
        print(f"Available keys are: {list(data.keys())}")
        sys.exit(1)

    if act_data.ndim == 3:
        print("Detected 3D sparse action data (N, L, D)")
        USE_DENSE_ACTION = False
        ACTION_DIM = act_data.shape[2]
    elif act_data.ndim == 2:
        print("Detected 2D dense action data (N, D)")
        USE_DENSE_ACTION = True
        ACTION_DIM = act_data.shape[1]
    
    SEQ_LEN = src_data.shape[1] # (N, L, 3) -> L
    
    print("Splitting data...")
    
    total_size = len(src_data)
    all_indices = np.arange(total_size)
    
    # 1. Handle Demo Set (Sequential from the end)
    if create_demo_set:
        # Ensure we don't take more than we have
        if demo_size >= total_size:
             raise ValueError("Demo size cannot be larger than total dataset.")
             
        pool_indices = all_indices[:-demo_size]
        demo_indices = all_indices[-demo_size:]
        print(f"Reserving last {demo_size} samples for Sequential Demo set.")
    else:
        pool_indices = all_indices
        # demo_indices is unused if create_demo_set is False
        print("No Demo set reserved.")

    np.random.seed(seed)
    np.random.shuffle(pool_indices)
    
    pool_size = len(pool_indices)
    n_train = int(pool_size * train_ratio)
    n_val = int(pool_size * val_ratio)
    
    train_indices = pool_indices[:n_train]
    val_indices = pool_indices[n_train : n_train + n_val]
    test_indices = pool_indices[n_train + n_val:]
    
    def extract_set(indices):
        return src_data[indices], act_data[indices], tgt_data[indices]

    src_train, act_train, tgt_train = extract_set(train_indices)
    src_val,   act_val,   tgt_val   = extract_set(val_indices)
    src_test,  act_test,  tgt_test  = extract_set(test_indices)

    print(f"Data split complete:")
    print(f"  Train: {len(src_train)}")
    print(f"  Val:   {len(src_val)}")
    print(f"  Test:  {len(src_test)} (Randomly selected)")

    if create_demo_set:
        src_demo, act_demo, tgt_demo = extract_set(demo_indices)
        print(f"  Demo:  {len(src_demo)} (Sequential)")
        
        return (
            src_train, act_train, tgt_train,
            src_val,   act_val,   tgt_val,
            src_test,  act_test,  tgt_test,
            src_demo,  act_demo,  tgt_demo,
            USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN
        )
    else:
        return (
            src_train, act_train, tgt_train,
            src_val,   act_val,   tgt_val,
            src_test,  act_test,  tgt_test,
            # Demo tuple omitted
            USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN
        )

def plot_model_comparison(
    models_dict: dict[str, BaseRopeModel],
    dataset,
    device,
    index: int = 0,
    denormalize: bool = True,
    train_mean: torch.Tensor = None,
    train_std: torch.Tensor = None,
    save_path: str = None,
    use_dense_action: bool = False,
    arrow_scale: float = 0.5 
):
    """
    Plots predictions from multiple models, with 3 views (3D, XY, YZ)
    per model in an N_models x 3 grid. Also plots the action arrow.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Get Ground Truth Data ---
    src_norm, action_raw, tgt_norm = dataset[index]
    
    # Move to device for model input
    src_dev = src_norm.unsqueeze(0).to(device)
    action_map_dev = action_raw.unsqueeze(0).to(device)
    
    # Keep on CPU for plotting
    tgt_next_cpu = tgt_norm.unsqueeze(0).cpu()
    src_cpu = src_norm.unsqueeze(0).cpu()

    # Denormalize ground truth for plotting
    if denormalize:
        if train_mean is None or train_std is None:
            raise ValueError("You must provide train_mean and train_std for denormalization.")
        tgt_plot = (tgt_next_cpu * train_std + train_mean).squeeze(0)
        src_plot = (src_cpu * train_std + train_mean).squeeze(0)
    else:
        tgt_plot = tgt_next_cpu.squeeze(0)
        src_plot = src_cpu.squeeze(0)

    # --- 2. Get Action Data for Plotting ---
    
    if use_dense_action:
        action_vec = action_raw[:3]
        # Handle clamping safely
        link_id_raw = action_raw[3].round().long()
        link_id = link_id_raw.clamp(0, src_plot.shape[0] - 1)
    else:
        link_id = torch.argmax(action_raw[:, 3]).long()
        action_vec = action_raw[link_id, :3]
    
    origin_point = src_plot[link_id]
    action_vec_scaled = action_vec * arrow_scale

    # --- 3. Create Subplots ---
    n_models = len(models_dict)
    if n_models == 0:
        print("No models to plot.")
        return
    
    fig = plt.figure(figsize=(18, n_models * 6)) 
    fig.suptitle(f"Model Comparison (Sample {index})", fontsize=18)

    # --- 4. Get predictions and plot on each axis ---
    for row_idx, (name, model) in enumerate(models_dict.items()):
        
        model.eval()
        with torch.no_grad():
            pred_next = model(src_dev, action_map_dev)
            if isinstance(pred_next, tuple):
                pred_next = pred_next[0]
            pred_next_cpu = pred_next.cpu()

            if denormalize:
                pred_plot = (pred_next_cpu * train_std + train_mean).squeeze(0)
            else:
                pred_plot = pred_next_cpu.squeeze(0)
        
        # Plot 1: 3D View
        ax_3d = fig.add_subplot(n_models, 3, (row_idx * 3) + 1, projection='3d')
        ax_3d.plot(src_plot[:, 0], src_plot[:, 1], src_plot[:, 2], 'o-', color='green', label='Initial', markersize=4)
        ax_3d.plot(tgt_plot[:, 0], tgt_plot[:, 1], tgt_plot[:, 2], 'o-', color='blue', label='Real', markersize=4)
        ax_3d.plot(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], 'o--', color='red', label=f'Pred', markersize=4)
        ax_3d.quiver(origin_point[0], origin_point[1], origin_point[2],
                     action_vec_scaled[0], action_vec_scaled[1], action_vec_scaled[2],
                     color='magenta', label='Action')
        ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
        ax_3d.set_title(f"{name} (3D View)")
        ax_3d.legend()

        # Plot 2: XY View
        ax_xy = fig.add_subplot(n_models, 3, (row_idx * 3) + 2)
        ax_xy.plot(src_plot[:, 0], src_plot[:, 1], 'o-', color='green', label='Initial', markersize=4)
        ax_xy.plot(tgt_plot[:, 0], tgt_plot[:, 1], 'o-', color='blue', label='Real', markersize=4)
        ax_xy.plot(pred_plot[:, 0], pred_plot[:, 1], 'o--', color='red', label=f'Pred', markersize=4)
        ax_xy.quiver(origin_point[0], origin_point[1],
                     action_vec_scaled[0], action_vec_scaled[1],
                     color='magenta', label='Action', angles='xy', scale_units='xy', scale=1)
        ax_xy.set_xlabel('X'); ax_xy.set_ylabel('Y')
        ax_xy.set_title(f"{name} (XY View)")
        ax_xy.set_aspect('equal', 'box')
        ax_xy.grid(True)
        
        # Plot 3: YZ View
        ax_yz = fig.add_subplot(n_models, 3, (row_idx * 3) + 3)
        ax_yz.plot(src_plot[:, 1], src_plot[:, 2], 'o-', color='green', label='Initial', markersize=4)
        ax_yz.plot(tgt_plot[:, 1], tgt_plot[:, 2], 'o-', color='blue', label='Real', markersize=4)
        ax_yz.plot(pred_plot[:, 1], pred_plot[:, 2], 'o--', color='red', label=f'Pred', markersize=4)
        ax_yz.quiver(origin_point[1], origin_point[2],
                     action_vec_scaled[1], action_vec_scaled[2],
                     color='magenta', label='Action', angles='xy', scale_units='xy', scale=1)
        ax_yz.set_xlabel('Y'); ax_yz.set_ylabel('Z')
        ax_yz.set_title(f"{name} (YZ View)")
        ax_yz.set_aspect('equal', 'box')
        ax_yz.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 

    if save_path:
        print(f"Saving comparison plot to {save_path}...")
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Plot saved successfully.")
        except Exception as e:
            print(f"Error saving plot: {e}")
    plt.close(fig) # Close to save memory

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)