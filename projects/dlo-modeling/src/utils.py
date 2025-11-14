import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import random
# This import needs to point to where base_model.py is.
# Assuming it is in 'models/' directory.
from models.base_model import BaseRopeModel

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

    Args:
        model: Trained model (must implement a forward method compatible with dataset).
        dataset: Dataset split (train/val/test).
        device: torch.device.
        index: Index of the sample to visualize.
        denormalize: Whether to denormalize using training statistics.
        train_mean: Mean tensor from the training set.
        train_std: Std tensor from the training set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    src, action_map, tgt_next = dataset[index]

    src = src.unsqueeze(0).to(device)
    action_map = action_map.unsqueeze(0).to(device)
    tgt_next = tgt_next.unsqueeze(0).to(device)

    with torch.no_grad():
        # model may return extra outputs
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
def plot_model_comparison(
    models_dict: dict[str, BaseRopeModel],
    dataset,
    device,
    index: int = 0,
    denormalize: bool = True,
    train_mean: torch.Tensor = None,
    train_std: torch.Tensor = None,
    save_path: str = None,  # <-- ADDED THIS ARGUMENT
):
    """
    Plots predictions from multiple models on the same 3D plot.

    Args:
        models_dict: A dictionary of { "model_name": model_instance }
        dataset: Dataset split (train/val/test).
        device: torch.device.
        index: Index of the sample to visualize.
        denormalize: Whether to denormalize.
        train_mean: Mean tensor from the training set.
        train_std: Std tensor from the training set.
        save_path: (str) Optional. Path to save the plot image.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get ground truth data
    src, action_map, tgt_next = dataset[index]
    src_dev = src.unsqueeze(0).to(device)
    action_map_dev = action_map.unsqueeze(0).to(device)
    tgt_next_cpu = tgt_next.unsqueeze(0).cpu()
    src_cpu = src.unsqueeze(0).cpu()

    # Denormalize ground truth for plotting
    if denormalize:
        if train_mean is None or train_std is None:
            raise ValueError("You must provide train_mean and train_std for denormalization.")
        tgt_plot = (tgt_next_cpu * train_std + train_mean).squeeze(0)
        src_plot = (src_cpu * train_std + train_mean).squeeze(0)
    else:
        tgt_plot = tgt_next_cpu.squeeze(0)
        src_plot = src_cpu.squeeze(0)


    # --- Get predictions from all models ---
    predictions = {}
    for name, model in models_dict.items():
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
            
            predictions[name] = pred_plot

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot ground truth
    ax.plot(src_plot[:, 0], src_plot[:, 1], src_plot[:, 2], 'o-', color='green', label='Initial State (t)', markersize=8)
    ax.plot(tgt_plot[:, 0], tgt_plot[:, 1], tgt_plot[:, 2], 'o-', color='blue', label='Real State (t+1)', markersize=8)

    # Plot predictions
    # Use a color cycle
    colors = plt.cm.get_cmap('hsv', len(predictions) + 1)
    
    for i, (name, pred) in enumerate(predictions.items()):
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], 'o--', color=colors(i), label=f'Pred: {name}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title(f"Model Comparison (Sample {index})")
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

    # --- vvvv THIS IS THE NEW SECTION vvvv ---
    if save_path:
        print(f"Saving comparison plot to {save_path}...")
        # We use fig.savefig and bbox_inches='tight' to ensure the legend is saved
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Plot saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    # --- ^^^^ THIS IS THE NEW SECTION ^^^^ ---

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
    Ensures rope_states and actions remain aligned.
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

    # --- THIS IS THE FIX ---
    return mse_pos + mse_delta
    # ------------------------

    # Rope length consistency
    def length_loss(pred, tgt):
        # Ensure pred and tgt are at least 3D (B, L, 3)
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(0)