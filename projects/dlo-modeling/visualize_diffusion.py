import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# --- 1. Robust Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))
sys.path.append(os.path.join(current_dir, 'src', 'models'))

# --- 2. Imports ---
try:
    from src.models.rope_diffusion import RopeDiffusion
    from src.data.rope_dataset import RopeDataset
    from src.utils import load_and_split_data
except ImportError as e:
    print(f"\nCRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# --- 3. Constants ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = 'src/data/rope_state_action_next_state_mil.npz'

# Model Hyperparameters
SEQ_LEN = 70      
ACTION_DIM = 4    
DIFFUSION_STEPS = 100 
BASE_DIM = 64

# --- 4. Local Fixed Visualization Functions ---

def plot_model_comparison_fixed(
    models_dict, dataset, device, index=0, 
    denormalize=True, train_mean=None, train_std=None, 
    save_path=None, use_dense_action=False, arrow_scale=0.5
):
    """Local fixed version of plot_model_comparison."""
    src_norm, action_raw, tgt_norm = dataset[index]
    
    src_dev = src_norm.unsqueeze(0).to(device)
    action_map_dev = action_raw.unsqueeze(0).to(device)
    
    # Denormalize Ground Truth
    mean_cpu = train_mean.cpu().view(1, 3) if train_mean is not None else 0
    std_cpu = train_std.cpu().view(1, 3) + 1e-8 if train_std is not None else 1

    if denormalize:
        src_plot = src_norm.cpu() * std_cpu + mean_cpu
        tgt_plot = tgt_norm.cpu() * std_cpu + mean_cpu
    else:
        src_plot = src_norm.cpu()
        tgt_plot = tgt_norm.cpu()

    # Action Vector
    if use_dense_action:
        action_vec = action_raw[:3]
        link_id = action_raw[3].round().long().clamp(0, src_plot.shape[0] - 1)
    else:
        link_id = torch.argmax(action_raw[:, 3]).long()
        action_vec = action_raw[link_id, :3]
    
    origin_point = src_plot[link_id]
    action_vec_scaled = action_vec * arrow_scale

    n_models = len(models_dict)
    fig = plt.figure(figsize=(18, n_models * 6)) 
    
    for row_idx, (name, model) in enumerate(models_dict.items()):
        model.eval()
        with torch.no_grad():
            pred = model(src_dev, action_map_dev)
            if isinstance(pred, tuple): pred = pred[0]
            pred = pred.cpu().squeeze(0)

            if denormalize:
                pred_plot = pred * std_cpu + mean_cpu
            else:
                pred_plot = pred

        # 3D View
        ax = fig.add_subplot(n_models, 3, (row_idx * 3) + 1, projection='3d')
        ax.plot(src_plot[:,0], src_plot[:,1], src_plot[:,2], 'o-', c='g', label='Init', alpha=0.3)
        ax.plot(tgt_plot[:,0], tgt_plot[:,1], tgt_plot[:,2], 'o-', c='b', label='Real')
        ax.plot(pred_plot[:,0], pred_plot[:,1], pred_plot[:,2], 'o--', c='r', label='Pred')
        ax.quiver(origin_point[0], origin_point[1], origin_point[2],
                  action_vec_scaled[0], action_vec_scaled[1], action_vec_scaled[2], color='m')
        ax.set_title(f"{name} (3D)")
        ax.legend()

        # XY
        ax = fig.add_subplot(n_models, 3, (row_idx * 3) + 2)
        ax.plot(src_plot[:,0], src_plot[:,1], 'o-', c='g', alpha=0.3)
        ax.plot(tgt_plot[:,0], tgt_plot[:,1], 'o-', c='b')
        ax.plot(pred_plot[:,0], pred_plot[:,1], 'o--', c='r')
        ax.set_title("XY View")
        ax.set_aspect('equal', 'box')
        
        # YZ
        ax = fig.add_subplot(n_models, 3, (row_idx * 3) + 3)
        ax.plot(src_plot[:,1], src_plot[:,2], 'o-', c='g', alpha=0.3)
        ax.plot(tgt_plot[:,1], tgt_plot[:,2], 'o-', c='b')
        ax.plot(pred_plot[:,1], pred_plot[:,2], 'o--', c='r')
        ax.set_title("YZ View")
        ax.set_aspect('equal', 'box')

    if save_path: fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def animate_rope_fixed(
    model, dataset, start_idx=0, steps=50, interval=200, device=None,
    denormalize=True, save=False, save_path='anim.mp4',
    train_mean=None, train_std=None
):
    """Local fixed version of animate_rope with proper auto-scaling."""
    model.eval()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    real_line, = ax.plot([], [], [], "o-", c="blue", label="Real")
    pred_line, = ax.plot([], [], [], "o--", c="red", label="Pred")
    ax.legend()

    # --- 1. Pre-calculate Limits based on Real Data ---
    # We scan the sequence to find the min/max coordinates
    print(f"Scanning {steps} frames to determine plot limits...")
    
    mean_cpu = train_mean.cpu().view(1, 1, 3) if train_mean is not None else 0
    std_cpu = train_std.cpu().view(1, 1, 3) + 1e-8 if train_std is not None else 1
    
    all_points = []
    for i in range(min(steps, len(dataset)-start_idx)):
        _, _, tgt = dataset[start_idx+i]
        if denormalize:
            tgt = tgt.unsqueeze(0) * std_cpu + mean_cpu
        all_points.append(tgt.view(-1, 3).numpy())
    
    all_points = np.concatenate(all_points, axis=0)
    
    # Set Limits with padding
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    pad = (maxs - mins).max() * 0.1
    
    ax.set_xlim(mins[0]-pad, maxs[0]+pad)
    ax.set_ylim(mins[1]-pad, maxs[1]+pad)
    ax.set_zlim(mins[2]-pad, maxs[2]+pad)
    
    print(f"DEBUG Plot Limits: X{mins[0]:.2f}:{maxs[0]:.2f}, Y{mins[1]:.2f}:{maxs[1]:.2f}, Z{mins[2]:.2f}:{maxs[2]:.2f}")

    # --- 2. Animation Loop ---
    src, action_map, _ = dataset[start_idx]
    src = src.unsqueeze(0).to(device)
    action_map = action_map.unsqueeze(0).to(device)
    curr_state = src.clone()

    def update(frame):
        nonlocal curr_state
        idx = start_idx + frame
        if idx >= len(dataset): return real_line, pred_line

        # Ground Truth
        _, action_map, tgt_next = dataset[idx]
        action_map = action_map.unsqueeze(0).to(device)
        
        # Prediction (Autoregressive)
        with torch.no_grad():
            pred_next = model(curr_state, action_map)
            curr_state = pred_next.detach()

        # Denormalize for plotting
        p_plot = pred_next.cpu().squeeze(0)
        t_plot = tgt_next.cpu()
        
        if denormalize:
            p_plot = p_plot * std_cpu.squeeze(0) + mean_cpu.squeeze(0)
            t_plot = t_plot * std_cpu.squeeze(0) + mean_cpu.squeeze(0)

        real_line.set_data(t_plot[:,0], t_plot[:,1])
        real_line.set_3d_properties(t_plot[:,2])
        
        pred_line.set_data(p_plot[:,0], p_plot[:,1])
        pred_line.set_3d_properties(p_plot[:,2])
        
        ax.set_title(f"Frame {idx}")
        return real_line, pred_line

    ani = FuncAnimation(
        fig, update, frames=min(steps, len(dataset)-start_idx),
        blit=False, interval=interval
    )
    
    if save:
        ani.save(save_path, writer='ffmpeg', fps=5)
        print(f"Saved: {save_path}")

# --- 5. Main Script ---

def main():
    print(f"--- Loading Data & Model on {DEVICE} ---")

    # 1. Load Data
    data_tuple = load_and_split_data(
        data_path=DATA_PATH, seed=SEED, create_demo_set=True, demo_size=200
    )
    
    (
        src_train_np, _, _, 
        _, _, _, 
        src_test_raw_np, act_test_raw_np, _, 
        src_demo_np, act_demo_np, _, 
        USE_DENSE_ACTION, ACTION_DIM_LOADED, SEQ_LEN_LOADED
    ) = data_tuple

    # 2. Stats
    src_train = torch.tensor(src_train_np, dtype=torch.float32)
    train_mean = torch.mean(src_train, dim=(0, 1))
    train_std = torch.std(src_train, dim=(0, 1))
    print(f"Stats: Mean={train_mean}, Std={train_std}")

    # 3. Datasets (Normalize=True is crucial for Diffusion)
    test_ds = RopeDataset(
        rope_states=torch.tensor(src_test_raw_np, dtype=torch.float32), 
        actions=torch.tensor(act_test_raw_np, dtype=torch.float32), 
        normalize=True, mean=train_mean, std=train_std, center_of_mass=False, dense=USE_DENSE_ACTION 
    )

    demo_ds = RopeDataset(
        rope_states=torch.tensor(src_demo_np, dtype=torch.float32), 
        actions=torch.tensor(act_demo_np, dtype=torch.float32), 
        normalize=True, mean=train_mean, std=train_std, center_of_mass=False, dense=USE_DENSE_ACTION 
    )

    # 4. Model
    model = RopeDiffusion(
        seq_len=SEQ_LEN_LOADED, action_dim=ACTION_DIM_LOADED,
        n_steps=DIFFUSION_STEPS, base_dim=BASE_DIM
    ).to(DEVICE)

    ckpt_path = "checkpoints_diffusion/standard/diffusion_best.pth" 
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints_diffusion/com_plus_standard/diffusion_best.pth"
        
    print(f"Loading: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # 5. Run Visualizations
    viz_dir = "visualizations_diffusion"
    os.makedirs(viz_dir, exist_ok=True)

    print("\n>>> Static Plots (Test Set)...")
    idx = np.random.choice(len(test_ds))
    plot_model_comparison_fixed(
        {"Diffusion": model}, test_ds, DEVICE, index=idx,
        denormalize=True, train_mean=train_mean, train_std=train_std,
        save_path=os.path.join(viz_dir, "static_plot.png"),
        use_dense_action=USE_DENSE_ACTION
    )

    print("\n>>> Video Animation (Demo Set)...")
    animate_rope_fixed(
        model, demo_ds, start_idx=0, steps=30, device=DEVICE,
        denormalize=True, save=True, save_path=os.path.join(viz_dir, "rollout.mp4"),
        train_mean=train_mean, train_std=train_std
    )

if __name__ == "__main__":
    main()