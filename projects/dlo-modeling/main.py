import torch
import numpy as np
import sys
import os
from torch.utils.data import TensorDataset

# --- 1. Path Setup ---
# Add 'src' and 'src/models' to the system path
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('src/models'))

# --- 2. Imports from your files ---
try:
    from models.rope_bert import RopeBERT
    from models.rope_bilstm import RopeBiLSTM
    from models.rope_transformer import RopeTransformer
    # Import all the functions we need from utils
    from utils import (
        set_seed, plot_model_comparison, rope_loss, load_data_from_npz,
        normalize_data, center_data
    )
    from models.base_model import BaseRopeModel
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print("Please ensure 'main.py' is in the root directory (above 'src')")
    print(f"Import error: {e}")
    sys.exit(1)


# --- 3. Constants ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters (Medium size)
D_MODEL = 384
NHEAD = 8
NUM_LAYERS = 6     
DIM_FF = D_MODEL * 4 # Standard Transformer FFN size

# Training parameters
BATCH_SIZE = 32    
EPOCHS = 10        
LR = 1e-4          

# This helper function is now inside main.py
def instantiate_models(SEQ_LEN, USE_DENSE_ACTION, ACTION_DIM):
    """
    Creates a new dictionary of models.
    Called for each experiment to ensure models are fresh.
    """
    models_to_compare = {}

    models_to_compare["BiLSTM"] = RopeBiLSTM(
        seq_len=SEQ_LEN, 
        d_model=D_MODEL, 
        num_layers=NUM_LAYERS,
        dropout=0.1,
        use_dense_action=USE_DENSE_ACTION, 
        action_dim=ACTION_DIM
    )

    models_to_compare["BERT"] = RopeBERT(
        seq_len=SEQ_LEN, 
        d_model=D_MODEL, 
        nhead=NHEAD, 
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF, 
        dropout=0.1,
        use_dense_action=USE_DENSE_ACTION, 
        action_dim=ACTION_DIM
    )

    models_to_compare["Transformer"] = RopeTransformer(
        seq_len=SEQ_LEN, 
        d_model=D_MODEL, 
        nhead=NHEAD, 
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS, 
        dim_feedforward=DIM_FF,
        dropout=0.1,
        use_dense_action=USE_DENSE_ACTION, 
        action_dim=ACTION_DIM
    )
    return models_to_compare

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # Dictionaries to store all experiment results
    all_test_losses_normalized = {}
    all_test_losses_denormalized = {}
    all_test_losses_rollout = {} 

    # 1. Load Data (Raw)
    try:
        (
            src_train_raw_np, act_train_raw_np, tgt_train_raw_np,
            src_val_raw_np,   act_val_raw_np,   tgt_val_raw_np,
            src_test_raw_np,  act_test_raw_np,  tgt_test_raw_np,
            USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN
        ) = load_data_from_npz(seed=SEED)
        
    except Exception as e:
        print(f"\n---!! An error occurred while loading data: {e}")
        raise
        
    print(f"\n--- Global Config ---")
    print(f"Using dense action: {USE_DENSE_ACTION}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Action dimension: {ACTION_DIM}")
    print(f"----------------------\n")
    
    # Convert all raw numpy arrays to raw PyTorch tensors
    src_train_raw = torch.tensor(src_train_raw_np, dtype=torch.float32)
    act_train_raw = torch.tensor(act_train_raw_np, dtype=torch.float32)
    tgt_train_raw = torch.tensor(tgt_train_raw_np, dtype=torch.float32)
    
    src_val_raw = torch.tensor(src_val_raw_np, dtype=torch.float32)
    act_val_raw = torch.tensor(act_val_raw_np, dtype=torch.float32)
    tgt_val_raw = torch.tensor(tgt_val_raw_np, dtype=torch.float32)
    
    src_test_raw = torch.tensor(src_test_raw_np, dtype=torch.float32)
    act_test_raw = torch.tensor(act_test_raw_np, dtype=torch.float32)
    tgt_test_raw = torch.tensor(tgt_test_raw_np, dtype=torch.float32)
    
    # --- Define Experiments ---
    # --- vvvv MODIFIED SECTION vvvv ---
    experiment_types = ['standard', 'com_plus_standard']
    # --- ^^^^ END MODIFIED SECTION ^^^^ ---

    for norm_type in experiment_types:
        print(f"\n=======================================================")
        print(f"  STARTING EXPERIMENT: {norm_type.upper()}")
        print(f"=======================================================\n")
        
        # --- 1. Prepare Data for this experiment ---
        denorm_stats_for_rollout = None
        
        if norm_type == 'standard':
            print("Applying 'Standard' normalization (mean/std)")
            (
                src_train, tgt_train,
                src_val, tgt_val,
                src_test, tgt_test,
                plot_mean, plot_std
            ) = normalize_data(
                src_train_raw, tgt_train_raw,
                src_val_raw, tgt_val_raw,
                src_test_raw, tgt_test_raw
            )
            denorm_flag = True
            denorm_stats_for_rollout = (plot_mean, plot_std)
            
        elif norm_type == 'com_only':
            # This block will now be skipped, but is safe to leave
            print("Applying 'Center of Mass (CoM)' normalization")
            (
                src_train, tgt_train,
                src_val, tgt_val,
                src_test, tgt_test
            ) = center_data(
                src_train_raw, tgt_train_raw,
                src_val_raw, tgt_val_raw,
                src_test_raw, tgt_test_raw
            )
            plot_mean, plot_std = None, None
            denorm_flag = False
            denorm_stats_for_rollout = None 

        elif norm_type == 'com_plus_standard':
            print("Applying 'CoM + Standard' normalization")
            # First, center all data
            (
                src_train_com, tgt_train_com,
                src_val_com, tgt_val_com,
                src_test_com, tgt_test_com
            ) = center_data(
                src_train_raw, tgt_train_raw,
                src_val_raw, tgt_val_raw,
                src_test_raw, tgt_test_raw
            )
            # Second, normalize the CoM data
            (
                src_train, tgt_train,
                src_val, tgt_val,
                src_test, tgt_test,
                plot_mean, plot_std
            ) = normalize_data(
                src_train_com, tgt_train_com,
                src_val_com, tgt_val_com,
                src_test_com, tgt_test_com
            )
            # This will denormalize plots back to the CoM-centered space
            denorm_flag = True
            denorm_stats_for_rollout = (plot_mean, plot_std)

        # Create datasets (actions are always the raw actions)
        # For train/val, the target is just t+1 (which is `tgt_train`)
        train_ds = TensorDataset(src_train, act_train_raw, tgt_train)
        val_ds = TensorDataset(src_val, act_val_raw, tgt_val)
        
        # The test dataset is NOT a DataLoader, but the raw tensors
        # This is required for the sequential rollout evaluation
        
        # --- 2. Instantiate Fresh Models ---
        print("Instantiating fresh models...")
        models_to_compare = instantiate_models(SEQ_LEN, USE_DENSE_ACTION, ACTION_DIM)

        # --- 3. Train models ---
        print(f"\n--- Starting Model Training ({norm_type}) ---")
        
        checkpoints_dir = os.path.join("checkpoints", norm_type)
        os.makedirs(checkpoints_dir, exist_ok=True)
        print(f"Checkpoints will be saved to the '{checkpoints_dir}' directory.")
        
        for name, model in models_to_compare.items():
            print(f"\nTraining {name}...")
            filename = f"{name}_best.pth"
            checkpoint_path = os.path.join(checkpoints_dir, filename)
            
            model.train_model(
                train_dataset=train_ds,
                val_dataset=val_ds,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr=LR,
                checkpoint_path=checkpoint_path, 
                criterion=rope_loss 
            )
            
            print(f"Loading best weights for {name} from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

        # --- 4. Evaluate models ---
        print(f"\n--- Starting Model Evaluation ({norm_type}) ---")
        
        norm_losses = {} 
        denorm_losses = {} 
        rollout_losses = {} 

        # Create a simple test dataset for single-step eval
        test_ds_single_step = TensorDataset(src_test, act_test_raw, tgt_test)
        
        for name, model in models_to_compare.items():
            # 1. Calculate NORMALIZED loss (t+1)
            norm_loss = model.evaluate_model(
                test_dataset=test_ds_single_step,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                criterion=rope_loss
            )
            norm_losses[name] = norm_loss
            print(f"  Normalized Test Loss (t+1) ({name} @ {norm_type}): {norm_loss:.6f}")

            # 2. Calculate DENORMALIZED loss (t+1)
            if denorm_flag: # i.e., 'standard' or 'com_plus_standard'
                denorm_loss = model.evaluate_model_denormalized(
                    test_dataset=test_ds_single_step,
                    device=DEVICE,
                    train_mean=plot_mean, 
                    train_std=plot_std,
                    batch_size=BATCH_SIZE,
                    criterion=rope_loss
                )
                denorm_losses[name] = denorm_loss
                print(f"Denormalized Test Loss (t+1) ({name} @ {norm_type}): {denorm_loss:.6f}")
            else:
                denorm_losses[name] = None 

            # 3. Calculate AUTOREGRESSIVE ROLLOUT loss (t+1 to t+10)
            print(f"Starting 10-step rollout for {name}...")
            rollout_loss = model.evaluate_autoregressive_rollout(
                test_src_tensor=src_test,
                test_act_tensor=act_test_raw,
                test_tgt_tensor=tgt_test,
                device=DEVICE,
                steps=10,
                criterion=rope_loss,
                # For rollout, we calculate loss in the "real" space
                # For 'com_only', this is the centered space (denorm_stats=None)
                # For others, it's the denormalized space
                denormalize_stats=denorm_stats_for_rollout
            )
            rollout_losses[name] = rollout_loss
            print(f" Rollout Loss (Avg t+1...t+10) ({name} @ {norm_type}): {rollout_loss:.6f}")

        
        # Store this experiment's results in the main dictionaries
        all_test_losses_normalized[norm_type] = norm_losses
        all_test_losses_denormalized[norm_type] = denorm_losses
        all_test_losses_rollout[norm_type] = rollout_losses
        
        # --- 5. Plot Comparison (on single-step test_ds) ---
        print(f"\n--- Plotting Model Comparison ({norm_type}) ---")
        
        # Create a subdirectory for this experiment's plots
        plots_dir = os.path.join("comparisons", norm_type)
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Plots will be saved to the '{plots_dir}' directory.")

        num_plots = 5
        if len(test_ds_single_step) == 0:
            print("Test dataset is empty, skipping plotting.")
        else:
            num_samples_to_plot = min(num_plots, len(test_ds_single_step))
            random_indices = np.random.choice(
                len(test_ds_single_step), 
                num_samples_to_plot, 
                replace=False
            )
            print(f"Plotting for {num_samples_to_plot} random samples (Indices: {random_indices})")
            
            for i, idx in enumerate(random_indices, 1): 
                filename = f"model_comparison_{i}.png"
                plot_save_file = os.path.join(plots_dir, filename)
                
                print(f"\n--- Plotting Sample {i}/{num_samples_to_plot} (Data Index: {idx}) ---")
                print(f"Displaying plot and saving to {plot_save_file}...")
                
                plot_model_comparison(
                    models_dict=models_to_compare,
                    dataset=test_ds_single_step,
                    device=DEVICE,
                    index=idx, 
                    denormalize=denorm_flag,        
                    train_mean=plot_mean,   
                    train_std=plot_std,     
                    save_path=plot_save_file,
                    use_dense_action=USE_DENSE_ACTION 
                )
    
    print("\n\nAll experiments finished.")
    
    # Print the final summary tables
    print_loss_table(
        all_test_losses_normalized, 
        "Final Normalized Test Loss (t+1 only)"
    )
    print_loss_table(
        all_test_losses_denormalized, 
        "Final Denormalized Test Loss (t+1 only, real-world units)"
    )
    print_loss_table(
        all_test_losses_rollout, 
        "Final Rollout Test Loss (Avg. t+1 to t+10, real-world units)"
    )


def print_loss_table(all_test_losses, title):
    """
    Prints a formatted Markdown table of all test losses.
    """
    print(f"\n\n=======================================================")
    print(f"           {title}")
    print(f"=======================================================")
    
    if not all_test_losses:
        print("No test losses recorded.")
        return

    # Get model names dynamically from the first entry
    first_norm_type = list(all_test_losses.keys())[0]
    model_names = list(all_test_losses[first_norm_type].keys())
    
    # Header
    header = "| Normalization |"
    separator = "|---|"
    for name in model_names:
        header += f" {name} Loss |"
        separator += "---|"
    print(header)
    print(separator)
    
    # Rows
    for norm_type, model_losses in all_test_losses.items():
        row = f"| {norm_type} |"
        for name in model_names:
            loss = model_losses.get(name)
            if isinstance(loss, float):
                row += f" {loss:.6f} |"
            else:
                row += f" N/A |"
        print(row)


if __name__ == "__main__":
    main()