import torch
from torch import nn
import numpy as np
import sys
import os
import gc # Garbage collector interface

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
    from src.models.dreamer_model import DreamerRopeModel 
    
    from src.losses import RopeLoss, WeightedRopeLoss, PhysicsInformedRopeLoss
    from src.data.rope_dataset import RopeDataset, RopeSequenceDataset 
    
    from src.utils import (
        set_seed, plot_model_comparison, load_and_split_data, cleanup_memory, load_and_split_data_interleaved
    )
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Import error: {e}")
    sys.exit(1)

# --- 3. Constants ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 4
DIM_FF = D_MODEL * 2

# Training parameters
BATCH_SIZE = 256    
EPOCHS = 10 
LR = 1e-3           

def get_model_class_and_params(model_name, seq_len, use_dense, act_dim):
    """
    Helper to return the Class and parameters dict for dynamic instantiation.
    """
    common_params = {
        "seq_len": seq_len,
        "use_dense_action": use_dense,
        "action_dim": act_dim,
        "dropout": 0.1
    }
    
    if model_name == "BiLSTM":
        params = {**common_params, "d_model": D_MODEL, "num_layers": NUM_LAYERS}
        return RopeBiLSTM, params
        
    elif model_name == "BERT":
        params = {**common_params, "d_model": D_MODEL, "nhead": NHEAD, 
                  "num_layers": NUM_LAYERS, "dim_feedforward": DIM_FF}
        return RopeBERT, params
        
    elif model_name == "Transformer":
        params = {**common_params, "d_model": D_MODEL, "nhead": NHEAD, 
                  "num_encoder_layers": NUM_LAYERS, "num_decoder_layers": NUM_LAYERS, 
                  "dim_feedforward": DIM_FF}
        return RopeTransformer, params
    
    return None, None

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    all_test_losses_normalized = {}
    all_test_losses_denormalized = {}
    all_test_losses_rollout = {} 
    
    # Updated default path
    data_path = '<INSERT_PATH_HERE>'
    
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
    experiment_types = ['com_plus_standard']
    standard_model_names = ["BiLSTM", "BERT", "Transformer"]

    for norm_type in experiment_types:
        print(f"\n=======================================================")
        print(f"  STARTING EXPERIMENT: {norm_type.upper()}") 
        print(f"=======================================================\n")
        
        checkpoints_dir = os.path.join("checkpoints2", norm_type)
        os.makedirs(checkpoints_dir, exist_ok=True)

        # === 1. Prepare Data ===
        use_com = (norm_type == 'com_plus_standard')
        
        # --- A. Standard Dataset ---
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

        # Stats for rollout/denormalization
        denorm_flag = True
        denorm_stats_for_rollout = (train_mean, train_std)

        # Initialize result containers for this experiment
        norm_losses = {} 
        denorm_losses = {} 
        rollout_losses = {} 

        # === 2. Sequential Training (Standard Models) ===
        for name in standard_model_names:
            print(f"\n>>> Processing {name} ({norm_type})...")
            
            # A. Instantiate ONE model
            ModelClass, params = get_model_class_and_params(name, SEQ_LEN, USE_DENSE_ACTION, ACTION_DIM)
            model = ModelClass(**params).to(DEVICE)
            
            # B. Train
            filename = f"{name}_best.pth"
            checkpoint_path = os.path.join(checkpoints_dir, filename)
            
            model.train_model(
                train_dataset=train_ds, val_dataset=val_ds, device=DEVICE,
                batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
                checkpoint_path=checkpoint_path, criterion=PhysicsInformedRopeLoss(w_pos=1.0, w_vel=1.0, w_stretch=1.5, w_bend=1.0, w_overlap=0.00)
            )
            
            # Reload best for Eval
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            
            # C. Evaluate
            print(f"Evaluating {name}...")
            # 1. Normalized
            n_loss = model.evaluate_model(
                test_dataset=test_ds, device=DEVICE, batch_size=BATCH_SIZE,
                criterion=nn.MSELoss()
            )
            norm_losses[name] = n_loss
            
            # 2. Denormalized
            dn_loss = model.evaluate_model_denormalized(
                test_dataset=test_ds, device=DEVICE,
                train_mean=train_mean, train_std=train_std,
                batch_size=BATCH_SIZE, criterion=nn.MSELoss() 
            )
            denorm_losses[name] = dn_loss
            
            # 3. Rollout
            r_loss = model.evaluate_autoregressive_rollout(
                test_src_tensor=src_test_raw, test_act_tensor=act_test_raw,
                test_tgt_tensor=tgt_test_raw, device=DEVICE, steps=1000,
                criterion=nn.MSELoss(), denormalize_stats=denorm_stats_for_rollout,
                num_rollouts=100
            )
            rollout_losses[name] = r_loss
            
            # D. MEMORY CLEANUP (Crucial Step)
            # We delete the model instance so VRAM is freed for the next model
            cleanup_memory(model)

        # === 3. Dreamer Training (Isolated) ===
        print(f"\n>>> Processing Dreamer ({norm_type})...")

        (
            src_train_raw, act_train_raw, tgt_train_raw,
            src_val_raw,   act_val_raw,   tgt_val_raw,
            src_test_raw,  act_test_raw,  tgt_test_raw,
            _, _, _,
            USE_DENSE_ACTION, ACTION_DIM, _,
        ) = load_and_split_data_interleaved(data_path=data_path, seed=SEED,train_ratio=0.8,
            val_ratio=0.1,create_demo_set=False, sequence_length = 200)
        #Manually defining to 10% of the raw files to avoid issues with rollout changes
        SEQ_LEN = 20
        # Prepare Dreamer Data
        dreamer_train_ds = RopeSequenceDataset(
            src_train_raw, act_train_raw, sequence_length=SEQ_LEN, normalize=True,  center_of_mass = use_com
        )
        dreamer_val_ds = RopeSequenceDataset(
            src_val_raw, act_val_raw, sequence_length=SEQ_LEN, normalize=False,
            mean=dreamer_train_ds.mean, std=dreamer_train_ds.std, center_of_mass = use_com
        )
        dreamer_test_ds = RopeSequenceDataset(
            src_test_raw, act_test_raw, sequence_length=SEQ_LEN, normalize=False,
            mean=dreamer_train_ds.mean, std=dreamer_train_ds.std, center_of_mass = use_com
        )
        
        dreamer_config = {
            "L": SEQ_LEN, "d_embed": 256, "d_action": 64, "d_rnn": 512, "d_z": 32, "beta_kl": 1.0, 
            "recon_loss_fn": PhysicsInformedRopeLoss(w_pos=1.0, w_vel=1.0, w_stretch=1.5, w_bend=1.0, w_overlap=0.00)
        }
        
        dreamer_model = DreamerRopeModel(**dreamer_config).to(DEVICE)
        dreamer_ckpt_path = os.path.join(checkpoints_dir, "dreamer_best.pth")
        
        dreamer_model.train_model(
            dreamer_train_ds, dreamer_val_ds, device=DEVICE,
            batch_size=350, epochs=EPOCHS, lr=1e-4,
            checkpoint_path=dreamer_ckpt_path, recon_loss_fn=dreamer_config["recon_loss_fn"]
        )
        
        # Eval Dreamer
        d_loss, d_recon, d_kl = dreamer_model.evaluate_model(
            dreamer_test_ds, DEVICE, batch_size=350,
            checkpoint_path=dreamer_ckpt_path, criterion=nn.MSELoss()
        )
        norm_losses["Dreamer"] = d_loss
        denorm_losses["Dreamer"] = "N/A" 
        rollout_losses["Dreamer"] = "N/A"
        
        print(f"  Dreamer Eval Done.")
        
        # Dreamer Cleanup
        cleanup_memory(dreamer_model)

        # Store results for this experiment type
        all_test_losses_normalized[norm_type] = norm_losses
        all_test_losses_denormalized[norm_type] = denorm_losses
        all_test_losses_rollout[norm_type] = rollout_losses
        
        # === 4. Plotting Phase (Reloading Models) ===
        # We only reload models if we actually have data to plot.
        # This is separated to ensure minimal memory overlap (Inference mode takes less VRAM).
        if len(test_ds) > 0:
            print(f"\n--- Plotting Model Comparison ({norm_type}) ---")
            plots_dir = os.path.join("comparisons2", norm_type)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Re-instantiate Standard Models for Plotting
            print("Reloading models for plotting (Inference Mode)...")
            plotting_models = {}
            
            try:
                for name in standard_model_names:
                    ModelClass, params = get_model_class_and_params(name, SEQ_LEN, USE_DENSE_ACTION, ACTION_DIM)
                    m = ModelClass(**params).to(DEVICE)
                    ckpt = os.path.join(checkpoints_dir, f"{name}_best.pth")
                    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                    m.eval() # Crucial for memory saving
                    plotting_models[name] = m
                
                # Generate Plots
                num_samples = min(5, len(test_ds))
                indices = np.random.choice(len(test_ds), num_samples, replace=False)
                
                with torch.no_grad(): # Crucial for memory saving
                    for i, idx in enumerate(indices, 1): 
                        filename = f"model_comparison_{i}.png"
                        plot_model_comparison(
                            models_dict=plotting_models, 
                            dataset=test_ds, 
                            device=DEVICE, index=idx, denormalize=denorm_flag,        
                            train_mean=train_mean, train_std=train_std,      
                            save_path=os.path.join(plots_dir, filename),
                            use_dense_action=USE_DENSE_ACTION 
                        )
            finally:
                # Cleanup plotting models immediately
                print("Cleaning up plotting models...")
                for m in plotting_models.values():
                    del m
                del plotting_models
                cleanup_memory(None)
                
    
    print("\n\nAll experiments finished.")
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