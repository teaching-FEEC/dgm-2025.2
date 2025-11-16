import numpy as np
import torch
import os  
from torch.utils.data import TensorDataset  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import random
# This import needs to point to where base_model.py is.
# Assuming it is in 'models/' directory.
from models.base_model import BaseRopeModel

# --- vvvv UNCHANGED FUNCTIONS vvvv ---

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

# --- vvvv MODIFIED FUNCTION vvvv ---
def load_data_from_npz(seed=42):
    """
    Loads and splits data from an NPZ file into raw numpy arrays.
    Normalization is no longer done here.
    The test set is NOT shuffled, to allow for rollouts.
    """
    print("Loading data from NPZ...")
    
    # === Path as requested ===
    data_path = 'src/data/rope_state_action_next_state_mil.npz'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please check that the file exists in the 'src/data' folder.")
        sys.exit(1)
        
    data = np.load(data_path)
    
    # Keys as per your description
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

    # --- Auto-detect data properties ---
    if act_data.ndim == 3:
        print("Detected 3D sparse action data (N, L, D)")
        USE_DENSE_ACTION = False
        ACTION_DIM = act_data.shape[2]
    elif act_data.ndim == 2:
        print("Detected 2D dense action data (N, D)")
        USE_DENSE_ACTION = True
        ACTION_DIM = act_data.shape[1]
    
    SEQ_LEN = src_data.shape[1] # (N, L, 3) -> L
    
    # --- Split Data (Numpy) ---
    print("Splitting data...")
    total_size = len(src_data)
    
    # --- MODIFICATION ---
    # We split indices sequentially first
    all_indices = np.arange(total_size)
    
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    
    # Split indices sequentially
    train_val_indices = all_indices[:train_size + val_size]
    test_indices = all_indices[train_size + val_size:] # Kept in order
    
    # Shuffle only the train/val indices
    np.random.seed(seed)
    np.random.shuffle(train_val_indices)
    
    # Get final train and val indices
    train_indices = train_val_indices[:train_size]
    val_indices = train_val_indices[train_size:]
    
    # Split numpy arrays
    src_train, act_train, tgt_train = src_data[train_indices], act_data[train_indices], tgt_data[train_indices]
    src_val,   act_val,   tgt_val   = src_data[val_indices],   act_data[val_indices],   tgt_data[val_indices]
    src_test,  act_test,  tgt_test  = src_data[test_indices],  act_data[test_indices],  tgt_data[test_indices]
    # --- END MODIFICATION ---

    print(f"Data split: {len(src_train)} train, {len(src_val)} val, {len(src_test)} test (Test set is sequential)")
    
    # Return 9 raw numpy arrays and 3 config values
    return (
        src_train, act_train, tgt_train,
        src_val,   act_val,   tgt_val,
        src_test,  act_test,  tgt_test,
        USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN
    )

# --- vvvv UNCHANGED PLOTTING FUNCTION vvvv ---

def plot_model_comparison(
    models_dict: dict[str, BaseRopeModel],
    dataset,
    device,
    index: int = 0,
    denormalize: bool = True,
    train_mean: torch.Tensor = None,
    train_std: torch.Tensor = None,
    save_path: str = None,
    use_dense_action: bool = False, # <-- ADDED
    arrow_scale: float = 0.5       # <-- ADDED (controls arrow size)
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
    # action_raw is the raw tensor from the dataset (on CPU)
    # src_plot is the denormalized ground truth (on CPU)
    
    if use_dense_action:
        # action_raw is (4,)
        action_vec = action_raw[:3]
        link_id = action_raw[3].round().long().clamp(0, src_plot.shape[0] - 1)
    else:
        # action_raw is (L, 4) -> [dx, dy, dz, flag]
        # Find the link with the non-zero flag
        link_id = torch.argmax(action_raw[:, 3]).long()
        action_vec = action_raw[link_id, :3]
    
    # Get the origin point from the denormalized source plot
    origin_point = src_plot[link_id]
    
    # Scale the action vector for plotting
    # We use the raw action_vec magnitude as requested
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
        
        # --- Get Model Prediction ---
        model.eval()
        with torch.no_grad():
            pred_next = model(src_dev, action_map_dev)
            if isinstance(pred_next, tuple):
                pred_next = pred_next[0]
            
            pred_next_cpu = pred_next.cpu()

            # Denormalize prediction
            if denormalize:
                pred_plot = (pred_next_cpu * train_std + train_mean).squeeze(0)
            else:
                pred_plot = pred_next_cpu.squeeze(0)
        
        # --- Create the 3 plots for this model ---
        
        # Plot 1: 3D View
        ax_3d = fig.add_subplot(n_models, 3, (row_idx * 3) + 1, projection='3d')
        ax_3d.plot(src_plot[:, 0], src_plot[:, 1], src_plot[:, 2], 'o-', color='green', label='Initial State (t)', markersize=4)
        ax_3d.plot(tgt_plot[:, 0], tgt_plot[:, 1], tgt_plot[:, 2], 'o-', color='blue', label='Real State (t+1)', markersize=4)
        ax_3d.plot(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], 'o--', color='red', label=f'Pred: {name}')
        # Add the action arrow
        ax_3d.quiver(origin_point[0], origin_point[1], origin_point[2],
                     action_vec_scaled[0], action_vec_scaled[1], action_vec_scaled[2],
                     color='magenta', label='Action')
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f"{name} (3D View)")
        ax_3d.legend()

        # Plot 2: XY View (Top-Down)
        ax_xy = fig.add_subplot(n_models, 3, (row_idx * 3) + 2)
        ax_xy.plot(src_plot[:, 0], src_plot[:, 1], 'o-', color='green', label='Initial State (t)', markersize=4)
        ax_xy.plot(tgt_plot[:, 0], tgt_plot[:, 1], 'o-', color='blue', label='Real State (t+1)', markersize=4)
        ax_xy.plot(pred_plot[:, 0], pred_plot[:, 1], 'o--', color='red', label=f'Pred: {name}')
        # Add the action arrow
        ax_xy.quiver(origin_point[0], origin_point[1],
                     action_vec_scaled[0], action_vec_scaled[1],
                     color='magenta', label='Action',
                     angles='xy', scale_units='xy', scale=1) # scale=1 means data units
        ax_xy.set_xlabel('X')
        ax_xy.set_ylabel('Y')
        ax_xy.set_title(f"{name} (XY View)")
        ax_xy.set_aspect('equal', 'box')
        ax_xy.legend()
        ax_xy.grid(True)
        
        # Plot 3: YZ View (Side)
        ax_yz = fig.add_subplot(n_models, 3, (row_idx * 3) + 3)
        ax_yz.plot(src_plot[:, 1], src_plot[:, 2], 'o-', color='green', label='Initial State (t)', markersize=4)
        ax_yz.plot(tgt_plot[:, 1], tgt_plot[:, 2], 'o-', color='blue', label='Real State (t+1)', markersize=4)
        ax_yz.plot(pred_plot[:, 1], pred_plot[:, 2], 'o--', color='red', label=f'Pred: {name}')
        # Add the action arrow
        ax_yz.quiver(origin_point[1], origin_point[2],
                     action_vec_scaled[1], action_vec_scaled[2],
                     color='magenta', label='Action',
                     angles='xy', scale_units='xy', scale=1)
        ax_yz.set_xlabel('Y')
        ax_yz.set_ylabel('Z')
        ax_yz.set_title(f"{name} (YZ View)")
        ax_yz.set_aspect('equal', 'box')
        ax_yz.legend()
        ax_yz.grid(True)

    # Adjust layout to prevent title overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 

    # --- 5. Save and Show ---
    if save_path:
        print(f"Saving comparison plot to {save_path}...")
        try:
            # Use bbox_inches='tight' to include titles/labels
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Plot saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    plt.show()
def plot_rope_predictions(
    model,
    dataset,
    device=None,
    index: int = 0,
    denormalize: bool = True,
    train_mean: torch.Tensor = None,
    train_std: torch.Tensor = None,
):
    """
    Plot real vs predicted rope states in 3D for a given sample index.
    (This is the original single-model plotting function)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    src, action_map, tgt_next = dataset[index]

    src = src.unsqueeze(0).to(device)
    action_map = action_map.unsqueeze(0).to(device)
    tgt_next = tgt_next.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_next = model(src, action_map)
        if isinstance(pred_next, tuple):
            pred_next = pred_next[0]

    # Move to CPU
    pred_next, tgt_next, src = pred_next.cpu(), tgt_next.cpu(), src.cpu()

    if denormalize:
        if train_mean is None or train_std is None:
            raise ValueError("You must provide train_mean and train_std for denormalization.")
        pred_next = pred_next * train_std + train_mean
        tgt_next = tgt_next * train_std + train_mean
        src = src * train_std + train_mean

    pred_next = pred_next.squeeze(0)
    tgt_next = tgt_next.squeeze(0)
    src = src.squeeze(0)

    # Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(src[:, 0], src[:, 1], src[:, 2], 'o-', color='green', label='Initial State (t)')
    ax.plot(tgt_next[:, 0], tgt_next[:, 1], tgt_next[:, 2], 'o-', color='blue', label='Real State (t+1)')
    ax.plot(pred_next[:, 0], pred_next[:, 1], pred_next[:, 2], 'o--', color='red', label='Predicted State (t+1)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"Rope State Prediction (Sample {index})")
    plt.tight_layout()
    plt.show()


def animate_rope(
    model,
    dataset,
    start_idx: int = 0,
    steps: int = 50,
    interval: int = 200,
    device: torch.device = None,
    denormalize: bool = True,
    save: bool = False,
    train_mean: torch.Tensor = None,
    train_std: torch.Tensor = None,
    dynamic_lim: bool = True,
    teacher_forcing: bool = True,
    center_of_mass: bool = False,
):
    """
    Animate predicted vs real rope states in 3D over multiple timesteps.
    ... (rest of docstring) ...
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    real_line, = ax.plot([], [], [], "o-", color="blue", label="Real")
    pred_line, = ax.plot([], [], [], "o--", color="red", label="Predicted")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")

    # Compute visualization limits
    pos = np.array([dataset[i][2] for i in range(min(steps, len(dataset)))])
    pos = pos.reshape(-1, 3)
    lim = float(pos.std(axis=0).max()) * 1.5
    if dynamic_lim:
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
    else:
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])

    # Safe broadcasting
    if train_mean is not None:
        train_mean = train_mean.view(1, -1)
    if train_std is not None:
        train_std = train_std.view(1, -1)

    src, action_map, tgt_next = dataset[start_idx]
    src = src.unsqueeze(0).to(device)
    action_map = action_map.unsqueeze(0).to(device)
    pred_state = src.clone()  # Initial predicted state (for autoregressive mode)

    def init():
        real_line.set_data([], [])
        real_line.set_3d_properties([])
        pred_line.set_data([], [])
        pred_line.set_3d_properties([])
        return real_line, pred_line


    # Frame Update Function
    def update(frame):
        nonlocal pred_state
        idx = start_idx + frame
        if idx >= len(dataset):
            return real_line, pred_line

        src, action_map, tgt_next = dataset[idx]
        src = src.unsqueeze(0).to(device)
        action_map = action_map.unsqueeze(0).to(device)
        tgt_next = tgt_next.cpu()

        # Prediction step
        with torch.no_grad():
            if teacher_forcing:
                model_input = src
            else:
                model_input = pred_state
            
            # model may return extra outputs
            pred_next_tuple = model(model_input, action_map)
            if isinstance(pred_next_tuple, tuple):
                pred_next = pred_next_tuple[0]
            else:
                pred_next = pred_next_tuple

            if not teacher_forcing:
                pred_state = pred_next.detach()  # use model prediction as next input

        pred_next = pred_next.cpu().squeeze(0)

        # Optional denormalization
        if denormalize:
            if train_mean is None or train_std is None:
                raise ValueError("Provide train_mean and train_std for de-normalization.")
            pred_next = pred_next * train_std + train_mean
            tgt_next = tgt_next * train_std + train_mean

        # Optional CoM centering for visualization
        if center_of_mass:
            pred_next -= pred_next.mean(dim=0, keepdim=True)
            tgt_next -= tgt_next.mean(dim=0, keepdim=True)

        real_line.set_data(tgt_next[:, 0], tgt_next[:, 1])
        real_line.set_3d_properties(tgt_next[:, 2])

        pred_line.set_data(pred_next[:, 0], pred_next[:, 1])
        pred_line.set_3d_properties(pred_next[:, 2])

        ax.set_title(f"Frame {idx} | {'Teacher Forcing' if teacher_forcing else 'Autoregressive'}")
    return real_line, pred_line

    ani = FuncAnimation(
        fig,
        update,
        frames=min(steps, len(dataset) - start_idx),
        init_func=init,
        blit=False,
        interval=interval,
        repeat=False,
    )

    if save:
        ani.save("rope_animation.mp4", writer="ffmpeg", fps=6)
        print("Saved animation as rope_animation.mp4") # <-- Fixed: only prints if save=True

    plt.close(fig)
    return HTML(ani.to_jshtml())

def split_data(rope_states, actions, train_ratio=0.8, val_ratio=0.1, shuffle = True ,seed=42):
    """
    Split the dataset into train, validation, and test sets.
    (This function is not used by this main.py but is kept for utility)
    """
    np.random.seed(seed)
    n = len(rope_states) - 100
    if shuffle:
      indices = np.random.permutation(n)
    else:
      indices = range(len(rope_states))
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    rope_train, rope_val, rope_test = rope_states[train_idx], rope_states[val_idx], rope_states[test_idx]
    actions_train, actions_val, actions_test = actions[train_idx], actions[val_idx], actions[test_idx]

    rope_demo, actions_demo = rope_states[-100:], actions[-100:]
    return (rope_train, rope_val, rope_test), (actions_train, actions_val, actions_test), (rope_demo, actions_demo)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rope_loss(pred, tgt, src):
    mse_pos = torch.nn.functional.mse_loss(pred, tgt)
    mse_delta = torch.nn.functional.mse_loss(pred - src, tgt - src)
    return mse_pos + mse_delta