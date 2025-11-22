import torch
import numpy as np
import sys
import os

# --- 1. Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))
sys.path.append(os.path.join(current_dir, 'src', 'models'))

# --- 2. Imports ---
try:
    from src.models.rope_bert import RopeBERT
    from src.models.rope_bilstm import RopeBiLSTM
    from src.models.rope_transformer import RopeTransformer
    from src.losses import RopeLoss, WeightedRopeLoss, PhysicsInformedRopeLoss
    from src.data.rope_dataset import RopeDataset
    from src.utils import (
        set_seed, plot_model_comparison, load_and_split_data, 
    )
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Import error: {e}")
    sys.exit(1)

# --- 3. Constants ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
D_MODEL = 384
NHEAD = 8
NUM_LAYERS = 6      
DIM_FF = D_MODEL * 4 

# Training parameters
BATCH_SIZE = 32     
EPOCHS = 100        
LR = 1e-3           

def instantiate_models(SEQ_LEN, USE_DENSE_ACTION, ACTION_DIM):
    models_to_compare = {}
    models_to_compare["BiLSTM"] = RopeBiLSTM(
        seq_len=SEQ_LEN, d_model=D_MODEL, num_layers=NUM_LAYERS, dropout=0.1,
        use_dense_action=USE_DENSE_ACTION, action_dim=ACTION_DIM
    )
    models_to_compare["BERT"] = RopeBERT(
        seq_len=SEQ_LEN, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF, dropout=0.1, use_dense_action=USE_DENSE_ACTION,
        action_dim=ACTION_DIM
    )
    models_to_compare["Transformer"] = RopeTransformer(
        seq_len=SEQ_LEN, d_model=D_MODEL, nhead=NHEAD, num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS, dim_feedforward=DIM_FF, dropout=0.1,
        use_dense_action=USE_DENSE_ACTION, action_dim=ACTION_DIM
    )
    return models_to_compare

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    all_test_losses_normalized = {}
    all_test_losses_denormalized = {}
    all_test_losses_rollout = {} 
    
    data_path = '<INSERT_THE_DATA_PATH_HERE>' # Updated default path
    data_path= '/home/lucasvd/lucasvd/IA_generativa_multimodal/rope_prediction/datasets/rope_state_action_next_state.npz'
    # 1. Load Data (Raw)
    try:
        (
            src_train_raw_np, act_train_raw_np, tgt_train_raw_np,
            src_val_raw_np,   act_val_raw_np,   tgt_val_raw_np,
            src_test_raw_np,  act_test_raw_np,  tgt_test_raw_np,
            USE_DENSE_ACTION, ACTION_DIM, SEQ_LEN
        ) = load_and_split_data(data_path=data_path,seed=SEED) # removed explicit path arg if relying on default
        
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
    
    src_test_raw = torch.tensor(src_test_raw_np, dtype=torch.float32)
    act_test_raw = torch.tensor(act_test_raw_np, dtype=torch.float32)
    tgt_test_raw = torch.tensor(tgt_test_raw_np, dtype=torch.float32) # Kept for rollout ground truth
    
    # --- Define Experiments ---
    experiment_types = ['standard', 'com_plus_standard']

    for norm_type in experiment_types:
        print(f"\n=======================================================")
        print(f"  STARTING EXPERIMENT: {norm_type.upper()}") 
        print(f"=======================================================\n")
        
        # --- 1. Prepare Data using RopeDataset ---
        
        # Determine CoM flag based on experiment type
        use_com = (norm_type == 'com_plus_standard')
        print(f"Configuration -> Center of Mass: {use_com}, Dense Action: {USE_DENSE_ACTION}")

        # Instantiate Train Dataset (Calculates Mean/Std)
        train_ds = RopeDataset(
            rope_states=src_train_raw, 
            actions=act_train_raw, 
            normalize=True, 
            center_of_mass=use_com, 
            dense=USE_DENSE_ACTION
        )

        # Extract stats from training set to use in Val/Test
        train_mean = train_ds.mean
        train_std = train_ds.std
        
        # Instantiate Val/Test Datasets (Use Train Stats)
        val_ds = RopeDataset(
            rope_states=src_val_raw, 
            actions=act_val_raw, 
            normalize=False, 
            mean=train_mean, 
            std=train_std,
            center_of_mass=use_com, 
            dense=USE_DENSE_ACTION
        )

        test_ds = RopeDataset(
            rope_states=src_test_raw, 
            actions=act_test_raw, 
            normalize=False, 
            mean=train_mean, 
            std=train_std,
            center_of_mass=use_com, 
            dense=USE_DENSE_ACTION
        )

        # Set stats for rollout/denormalization
        denorm_flag = True
        denorm_stats_for_rollout = (train_mean, train_std)
        
        # --- 2. Instantiate Fresh Models ---
        print("Instantiating fresh models...")
        models_to_compare = instantiate_models(SEQ_LEN, USE_DENSE_ACTION, ACTION_DIM)

        # --- 3. Train models ---
        print(f"\n--- Starting Model Training ({norm_type}) ---")
        
        checkpoints_dir = os.path.join("checkpoints2", norm_type)
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        for name, model in models_to_compare.items():
            print(f"\nTraining {name}...")
            filename = f"{name}_best.pth"
            checkpoint_path = os.path.join(checkpoints_dir, filename)
            
            model.train_model(
                train_dataset=train_ds, # Passing RopeDataset directly
                val_dataset=val_ds,     # Passing RopeDataset directly
                device=DEVICE,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr=LR,
                checkpoint_path=checkpoint_path, 
                criterion=WeightedRopeLoss() 
            )
            
            print(f"Loading best weights for {name} from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

        # --- 4. Evaluate models ---
        print(f"\n--- Starting Model Evaluation ({norm_type}) ---")
        
        norm_losses = {} 
        denorm_losses = {} 
        rollout_losses = {} 

        for name, model in models_to_compare.items():
            # 1. Normalized loss
            norm_loss = model.evaluate_model(
                test_dataset=test_ds, # Passing RopeDataset directly
                device=DEVICE,
                batch_size=BATCH_SIZE,
                criterion=WeightedRopeLoss()
            )
            norm_losses[name] = norm_loss
            print(f"  Normalized Test Loss (t+1) ({name} @ {norm_type}): {norm_loss:.6f}")

            # 2. Denormalized loss
            if denorm_flag:
                denorm_loss = model.evaluate_model_denormalized(
                    test_dataset=test_ds, # Passing RopeDataset directly
                    device=DEVICE,
                    train_mean=train_mean, 
                    train_std=train_std,
                    batch_size=BATCH_SIZE,
                    criterion=WeightedRopeLoss() 
                )
                denorm_losses[name] = denorm_loss
                print(f"Denormalized Test Loss (t+1) ({name} @ {norm_type}): {denorm_loss:.6f}")
            else:
                denorm_losses[name] = None 

            # 3. Autoregressive Rollout
            # Note: evaluate_autoregressive_rollout usually expects raw tensors for initiation
            print(f"Starting 1000-step rollout for {name} (100 samples)...")
            rollout_loss = model.evaluate_autoregressive_rollout(
                test_src_tensor=src_test_raw, # Keep using raw tensors for rollout start points
                test_act_tensor=act_test_raw,
                test_tgt_tensor=tgt_test_raw,
                device=DEVICE,
                steps=1000,
                criterion=WeightedRopeLoss(), 
                denormalize_stats=denorm_stats_for_rollout,
                num_rollouts=100
            )
            rollout_losses[name] = rollout_loss
            print(f" Rollout Loss (Avg t+1...t+1000) ({name} @ {norm_type}): {rollout_loss:.6f}")

        
        all_test_losses_normalized[norm_type] = norm_losses
        all_test_losses_denormalized[norm_type] = denorm_losses
        all_test_losses_rollout[norm_type] = rollout_losses
        
        # --- 5. Plot Comparison ---
        print(f"\n--- Plotting Model Comparison ({norm_type}) ---")
        plots_dir = os.path.join("comparisons2", norm_type)
        os.makedirs(plots_dir, exist_ok=True)

        num_plots = 5
        # Use length of test_ds for plotting availability
        ds_len = len(test_ds)
        
        if ds_len == 0:
            print("Test dataset is empty, skipping plotting.")
        else:
            num_samples_to_plot = min(num_plots, ds_len)
            random_indices = np.random.choice(ds_len, num_samples_to_plot, replace=False)
            
            for i, idx in enumerate(random_indices, 1): 
                filename = f"model_comparison_{i}.png"
                plot_save_file = os.path.join(plots_dir, filename)
                
                plot_model_comparison(
                    models_dict=models_to_compare,
                    dataset=test_ds, # Pass RopeDataset
                    device=DEVICE,
                    index=idx, 
                    denormalize=denorm_flag,        
                    train_mean=train_mean,    
                    train_std=train_std,      
                    save_path=plot_save_file,
                    use_dense_action=USE_DENSE_ACTION 
                )
    
    print("\n\nAll experiments finished.")
    print_loss_table(all_test_losses_normalized, "Final Normalized Test Loss (t+1 only)")
    print_loss_table(all_test_losses_denormalized, "Final Denormalized Test Loss (t+1 only, real-world units)")
    print_loss_table(all_test_losses_rollout, "Final Rollout Test Loss (Avg. t+1 to t+1000, 100 rollouts, real-world units)")

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
            else:
                row += f" N/A |"
        print(row)

if __name__ == "__main__":
    main()