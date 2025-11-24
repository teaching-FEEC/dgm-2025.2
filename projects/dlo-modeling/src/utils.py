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
from models.dreamer_model import DreamerRopeModel
from src.data.rope_dataset import RopeSequenceDataset
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
def load_and_split_data_interleaved(
    data_path,
    rollout_size=1000,   # Size of a full simulation episode
    sequence_length=50,  # Size of atomic chunk to shuffle
    seed=42, 
    create_demo_set=False, 
    demo_size=100,
    train_ratio=0.8,
    val_ratio=0.1
):
    """
    Loads data from NPZ, creates a sequential Demo set (optional), and splits the 
    remaining data by shuffling CHUNKS within rollouts.
    
    This ensures that even small datasets have representation in Train/Val/Test
    across different phases of the simulation, while maintaining local temporal 
    structure within the 'sequence_length'.

    Args:
        data_path (str): Path to the .npz file.
        rollout_size (int): Total steps in one simulation rollout (e.g., 1000 or 10000).
        sequence_length (int): The size of the chunk (T) used for shuffling.
        seed (int): Random seed.
        create_demo_set (bool): If True, reserves the last N items for a demo set.
        demo_size (int): Number of items to reserve for the demo set.
        train_ratio (float): Ratio of chunks for training.
        val_ratio (float): Ratio of chunks for validation.

    Returns:
        (train_set, val_set, test_set, [demo_set], USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN)
    """
    print(f"Loading data from NPZ (Interleaved Split)...")
    
    # === 1. Path & Load ===
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        try:
             parent = os.path.dirname(data_path)
             print(f"Please check that the file exists in the '{parent}' folder.")
        except:
             pass
        sys.exit(1)
        
    data = np.load(data_path, allow_pickle=True)
    
    # === 2. Key Extraction ===
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

    # === 3. Auto-detect Properties ===
    if act_data.ndim == 3:
        print("Detected 3D sparse action data (N, L, D)")
        USE_DENSE_ACTION = False
        ACTION_DIM = act_data.shape[2]
    elif act_data.ndim == 2:
        print("Detected 2D dense action data (N, D)")
        USE_DENSE_ACTION = True
        ACTION_DIM = act_data.shape[1]
    
    SEQ_LEN = src_data.shape[1] # (N, L, 3) -> L
    
    # === 4. Handle Demo Set ===
    # We strip the demo set BEFORE performing the interleaved logic
    # to ensure the rollout math (divisibility) applies to the training pool.
    
    total_samples = len(src_data)
    
    if create_demo_set:
        if demo_size >= total_samples:
             raise ValueError("Demo size cannot be larger than total dataset.")
        
        # Reserve last N samples
        src_demo = src_data[-demo_size:]
        act_demo = act_data[-demo_size:]
        tgt_demo = tgt_data[-demo_size:]
        
        # Remaining data for split
        src_pool = src_data[:-demo_size]
        act_pool = act_data[:-demo_size]
        tgt_pool = tgt_data[:-demo_size]
        
        print(f"Reserving last {demo_size} samples for Sequential Demo set.")
    else:
        src_pool = src_data
        act_pool = act_data
        tgt_pool = tgt_data
        print("No Demo set reserved.")

    # === 5. Interleaved Splitting Logic ===
    print("Splitting data (Interleaved Chunks)...")
    
    pool_len = len(src_pool)
    num_rollouts = pool_len // rollout_size
    
    if num_rollouts == 0:
        print(f"Warning: Dataset size ({pool_len}) is smaller than rollout_size ({rollout_size}).")
        print("Adjusting rollout_size to dataset size to prevent crash, but interleaving may be ineffective.")
        rollout_size = pool_len
        num_rollouts = 1

    chunks_per_rollout = rollout_size // sequence_length
    
    # Calculate exact cutoff to ensure perfect reshaping
    clean_cutoff = num_rollouts * chunks_per_rollout * sequence_length
    
    # Slice raw data to fit divisible dimensions
    s_clean = src_pool[:clean_cutoff]
    a_clean = act_pool[:clean_cutoff]
    t_clean = tgt_pool[:clean_cutoff]

    # --- Reshape into (Rollouts, Chunks, Time, Features) ---
    # We use *shape[1:] to automatically handle remaining dimensions (e.g. L, 3) or (D,)
    s_reshaped = s_clean.reshape(num_rollouts, chunks_per_rollout, sequence_length, *s_clean.shape[1:])
    a_reshaped = a_clean.reshape(num_rollouts, chunks_per_rollout, sequence_length, *a_clean.shape[1:])
    t_reshaped = t_clean.reshape(num_rollouts, chunks_per_rollout, sequence_length, *t_clean.shape[1:])

    # --- Generate Shuffle Indices ---
    np.random.seed(seed)
    chunk_idxs = np.arange(chunks_per_rollout)
    np.random.shuffle(chunk_idxs)

    # --- Split Indices ---
    n_train = int(train_ratio * chunks_per_rollout)
    n_val = int(val_ratio * chunks_per_rollout)
    # Test gets the rest

    train_idxs = chunk_idxs[:n_train]
    val_idxs = chunk_idxs[n_train : n_train + n_val]
    test_idxs = chunk_idxs[n_train + n_val:]

    # --- Extract & Flatten ---
    # Helper to select chunks and flatten back to (N, ...)
    def get_split(indices, original_shape_suffix):
        if len(indices) == 0:
            return np.empty((0, *original_shape_suffix))
        # Select: (Rollouts, Selected_Chunks, Time, ...)
        subset = s_reshaped[:, indices] # Example usage, need to apply to s, a, t generically
        # We need to flatten the first 3 dims: Rollouts * Selected_Chunks * Time
        return subset.reshape(-1, *original_shape_suffix)

    # We do this manually for s, a, t to avoid closure scope issues with variables
    
    # Train
    src_train = s_reshaped[:, train_idxs].reshape(-1, *s_clean.shape[1:])
    act_train = a_reshaped[:, train_idxs].reshape(-1, *a_clean.shape[1:])
    tgt_train = t_reshaped[:, train_idxs].reshape(-1, *t_clean.shape[1:])
    
    # Val
    src_val = s_reshaped[:, val_idxs].reshape(-1, *s_clean.shape[1:])
    act_val = a_reshaped[:, val_idxs].reshape(-1, *a_clean.shape[1:])
    tgt_val = t_reshaped[:, val_idxs].reshape(-1, *t_clean.shape[1:])
    
    # Test
    src_test = s_reshaped[:, test_idxs].reshape(-1, *s_clean.shape[1:])
    act_test = a_reshaped[:, test_idxs].reshape(-1, *a_clean.shape[1:])
    tgt_test = t_reshaped[:, test_idxs].reshape(-1, *t_clean.shape[1:])

    print(f"Interleaved Split Stats:")
    print(f"  Rollouts used: {num_rollouts}")
    print(f"  Chunks/Rollout: {chunks_per_rollout} (Seq Len: {sequence_length})")
    print(f"  Train: {len(src_train)}")
    print(f"  Val:   {len(src_val)}")
    print(f"  Test:  {len(src_test)}")

    if create_demo_set:
        print(f"  Demo:  {len(src_demo)} (Sequential)")
        return (
            (src_train, act_train, tgt_train),
            (src_val,   act_val,   tgt_val),
            (src_test,  act_test,  tgt_test),
            (src_demo,  act_demo,  tgt_demo),
            USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN
        )
    else:
        return (
            (src_train, act_train, tgt_train),
            (src_val,   act_val,   tgt_val),
            (src_test,  act_test,  tgt_test),
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

def plot_dreamer_prediction(
    model: DreamerRopeModel,
    dataset: RopeSequenceDataset,
    device=None,
    seq_index: int = 0,
    time_step: int = 10,
    denormalize: bool = True,
):
    """
    Plot a single one-step-ahead 'dreamed' prediction vs. ground truth in 3D.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    try:
        states_seq, actions_seq = dataset[seq_index]
    except Exception as e:
        print(f"Error getting data from dataset at index {seq_index}: {e}")
        return

    if time_step >= len(states_seq) - 1:
        print(f"Error: time_step {time_step} is out of bounds for sequence of length {len(states_seq)}. Clamping to max valid.")
        time_step = len(states_seq) - 2
    if time_step < 0:
        time_step = 0

    src = states_seq[time_step]         # S_t
    tgt_next = states_seq[time_step + 1] # S_{t+1} (Ground Truth)
    action = actions_seq[time_step]     # A_t (Action to take at step t)

    # We need A_{t-1} for the observe step
    action_minus_1 = actions_seq[time_step - 1] if time_step > 0 else torch.zeros_like(action)
    with torch.no_grad():
        # Get initial state (B=1)
        h_0, c_0, z_0 = model.get_initial_hidden_state(1, device=device)

        # ---
        # --- FIX IS HERE: ---
        # ---
        # Squeeze the batched (1, D) initial states to unbatched (D,)
        # to match the unbatched S_t and A_t_minus_1.
        # The observe method will correctly re-batch all of them.
        h_0 = h_0.squeeze(0)
        c_0 = c_0.squeeze(0)
        z_0 = z_0.squeeze(0)

        # 5. Get the model's internal state by "observing" S_t
        h_t, c_t, z_t, _, _ = model.observe(
            src.to(device), action_minus_1.to(device), h_0, c_0, z_0
        )

        # 6. "Dream" one step forward from (h_t, z_t) using A_t
        pred_next_b, _, _, _ = model.dream(h_t, c_t, z_t, action.to(device))

    pred_next = pred_next_b.cpu().squeeze(0)
    src = src.cpu()
    tgt_next = tgt_next.cpu()

    if denormalize:
        if hasattr(dataset, 'mean') and hasattr(dataset, 'std'):
            mean = dataset.mean.cpu().squeeze()
            std = dataset.std.cpu().squeeze()
            pred_next = pred_next * std + mean
            tgt_next = tgt_next * std + mean
            src = src * std + mean
        else:
            print("Warning: Could not denormalize. Dataset has no 'mean' or 'std' attribute.")

    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(src[:, 0], src[:, 1], src[:, 2], 'o-', color='green', label='Initial State (t)')
    ax.plot(tgt_next[:, 0], tgt_next[:, 1], tgt_next[:, 2], 'o-', color='blue', label='Real State (t+1)')
    ax.plot(pred_next[:, 0], pred_next[:, 1], pred_next[:, 2], 'x--', color='red', label='Predicted State (t+1)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"Dreamer One-Step Prediction (Seq {seq_index}, t={time_step})")
    plt.tight_layout()
    plt.show()

# --- UPDATED: Animation Utility Function ---

def animate_dreamer_rollout(
    model: "DreamerRopeModel",
    demo_dataset: "RopeSequenceDataset",
    start_idx: int = 0,
    steps: int = 48,
    interval: int = 100,
    teacher_forcing: bool = False,
    train_mean: torch.Tensor = None,
    train_std: torch.Tensor = None
) -> HTML:
    """
    Generates an HTML animation of a Dreamer model rollout in 3D,
    comparing predictions to ground truth.

    Args:
        model: The trained DreamerRopeModel.
        demo_dataset: RopeSequenceDataset to pull data from.
        start_idx: Which sequence in the dataset to use.
        steps: How many steps to animate.
        ...
    """
    model.eval()
    device = next(model.parameters()).device

    # Get the sequence data
    # The dataset __getitem__ returns the sequence
    states_seq, actions_seq = demo_dataset[start_idx]
    states_seq = states_seq.to(device)
    actions_seq = actions_seq.to(device)

    # --- Data Generation ---
    num_frames = min(steps + 1, len(states_seq))

    with torch.no_grad():
        if teacher_forcing:
            # "Observe" every step and reconstruct.
            # This shows how well the model can autoencode.
            # We must ensure we only feed in the number of frames we will animate
            recon_states, _, _ = model(
                states_seq[:num_frames].unsqueeze(0),
                actions_seq[:num_frames].unsqueeze(0)
            )
            predictions = recon_states.squeeze(0).cpu().numpy()

        else:
            # Autoregressive "Dream" rollout
            predictions_list = []

            # 1. Observe S_0 to get initial latent state
            S_0 = states_seq[0]
            A_minus_1 = torch.zeros(4, device=device)

            # Get the initial (B=1) hidden states
            h_0, c_0, z_0 = model.get_initial_hidden_state(1, device)

            # Squeeze to 1D (D,) to match the flawed unbatching logic
            # in the model's observe/dream methods.
            h_t = h_0.squeeze(0)
            c_t = c_0.squeeze(0)
            z_t = z_0.squeeze(0)

            # The observe method will internally re-batch these to (1, D)
            # because S_0 is unbatched.
            h_t, c_t, z_t, _, _ = model.observe(
                S_0, A_minus_1, h_t, c_t, z_t
            )

            predictions_list.append(S_0.cpu().numpy()) # Add the ground truth start


            # 2. Loop and dream
            for i in range(num_frames - 1): # -1 because S_0 is already added
                A_t = actions_seq[i]

                # Dream one step
                S_hat_tp1, h_tp1, c_tp1, z_tp1 = model.dream(h_t, c_t, z_t, A_t)

                predictions_list.append(S_hat_tp1.cpu().numpy())

                # Update latent state for next dream
                h_t, c_t, z_t = h_tp1, c_tp1, z_tp1

            predictions = np.array(predictions_list)

    # Get ground truth for comparison
    ground_truth = states_seq.cpu().numpy()

    # --- Denormalization ---
    # Use mean/std from the dataset object
    if train_mean is None and train_std is None:
        if hasattr(demo_dataset, 'mean') and hasattr(demo_dataset, 'std'):
            mean_np = demo_dataset.mean.cpu().numpy().squeeze()
            std_np = demo_dataset.std.cpu().numpy().squeeze()
            predictions = predictions * std_np + mean_np
            ground_truth = ground_truth * std_np + mean_np

    # --- Animation ---
    plt.close('all') # Close previous plots
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Find global bounds for all data to set axis limits
    all_data = np.concatenate([predictions[:num_frames], ground_truth[:num_frames]], axis=0)
    center = all_data.mean(axis=(0, 1))
    # Handle potential NaNs if data is bad
    if np.isnan(all_data).any():
        print("Warning: NaN detected in animation data. Skipping.")
        return HTML("Error: NaN in animation data.")
    max_range = (np.nanmax(all_data, axis=(0, 1)) - np.nanmin(all_data, axis=(0, 1))).max() / 2.0 + 0.5
    if max_range == 0 or np.isnan(max_range):
        max_range = 1.0 # Default range if data is flat

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Initialize lines
    line_gt, = ax.plot([], [], [], 'o-', lw=2, label='Ground Truth')
    line_pred, = ax.plot([], [], [], 'x-', lw=2, label='Prediction')
    ax.legend()
    title = ax.set_title(f'Frame 0 / {num_frames - 1}')

    def update(frame):
        # Ground Truth
        data_gt = ground_truth[frame]
        line_gt.set_data(data_gt[:, 0], data_gt[:, 1])
        line_gt.set_3d_properties(data_gt[:, 2])

        # Prediction
        data_pred = predictions[frame]
        line_pred.set_data(data_pred[:, 0], data_pred[:, 1])
        line_pred.set_3d_properties(data_pred[:, 2])

        title.set_text(f'Frame {frame} / {num_frames - 1} ({"Dreaming" if not teacher_forcing else "Reconstructing"})')
        return line_gt, line_pred, title

    ani = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=interval,
        blit=False # Blit must be False for 3D plots
    )

    plt.close(fig) # Prevent static plot from showing
    return HTML(ani.to_jshtml())

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)