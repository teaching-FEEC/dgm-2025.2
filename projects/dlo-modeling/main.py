import torch
import numpy as np
import sys
import os
from torch.utils.data import TensorDataset, random_split

# --- 1. Path Setup ---
# Add 'src' and 'src/models' to the system path
sys.path.append(os.path.abspath('src'))
sys.path.append(os.path.abspath('src/models'))

# --- 2. Imports from your files ---
try:
    from models.rope_bert import RopeBERT
    from models.rope_bilstm import RopeBiLSTM
    from models.rope_transformer import RopeTransformer
    # split_data wird nicht mehr benötigt, da wir random_split verwenden
    from utils import set_seed, plot_model_comparison, rope_loss
    from models.base_model import BaseRopeModel
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print("Please ensure 'main.py' is in the root directory (above 'src')")
    print(f"Import error: {e}")
    sys.exit(1)


# --- 3. Constants ---
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
# Sie müssen diese möglicherweise an Ihre geladenen Daten anpassen
SEQ_LEN = 70       # Sequence length of the rope
STATE_DIM = 3      # (x, y, z)
ACTION_DIM = 4     # Sparse action (dx, dy, dz, flag)

# Model parameters
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2     # Klein halten für schnelles Training
DIM_FF = D_MODEL * 2

# Training parameters
BATCH_SIZE = 32
EPOCHS = 5         # Niedrig halten für einen schnellen Testlauf
LR = 1e-3
USE_DENSE_ACTION = True # Use dense actions (B, 4) based on data format


def load_data_from_npz():
    """
    Lädt Daten aus einer NPZ-Datei, die 'states', 'actions' 
    und 'next_states' enthält, und teilt sie auf.
    
    !!!! SIE MÜSSEN DIESE FUNKTION BEARBEITEN !!!!
    """
    print("Loading data from NPZ...")
    
    # === BEARBEITEN SIE DIESEN PFAD ===
    data_path = 'src/data/rope_state_action_next_state_mil.npz'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please edit 'main.py' to point to your .npz file.")
        sys.exit(1)
        
    data = np.load(data_path)
    
    # === BEARBEITEN SIE DIESE KEYS ===
    # Dies sollten die Namen der Arrays in Ihrer .npz-Datei sein
    states_key = 'states'         # Ihr 'src'-Tensor (Zustand t)
    actions_key = 'actions'       # Ihr 'action'-Tensor (Aktion t)
    next_states_key = 'next_states' # Ihr 'tgt'-Tensor (Zustand t+1)

    try:
        src_data = data[states_key]
        act_data = data[actions_key]
        tgt_data = data[next_states_key]
    except KeyError as e:
        print(f"Error: Key {e} not found in {data_path}.")
        print(f"Available keys are: {list(data.keys())}")
        print("Please edit 'main.py' to use the correct keys for your data.")
        sys.exit(1)

    # Sicherstellen, dass alle Arrays die gleiche Anzahl von Samples haben
    if not (len(src_data) == len(act_data) == len(tgt_data)):
        print(f"Error: Data arrays have different lengths (number of samples)!")
        print(f"  {states_key}: {len(src_data)}")
        print(f"  {actions_key}: {len(act_data)}")
        print(f"  {next_states_key}: {len(tgt_data)}")
        print("All arrays must have the same first dimension (N_samples).")
        sys.exit(1)

    # In Tensoren umwandeln
    src_tensor = torch.tensor(src_data, dtype=torch.float32)
    act_tensor = torch.tensor(act_data, dtype=torch.float32)
    tgt_tensor = torch.tensor(tgt_data, dtype=torch.float32)

    # Ein vollständiges Dataset erstellen
    full_dataset = TensorDataset(src_tensor, act_tensor, tgt_tensor)
    
    # Dataset aufteilen (z.B. 80% Train, 10% Val, 10% Test)
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size
    
    print(f"Splitting data ({total_size} total samples):")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    print(f"  Test:  {test_size} samples")

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED) # Unseren Seed verwenden
    )
    
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    return train_dataset, val_dataset, test_dataset


def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    # 1. Daten aus NPZ laden
    try:
        train_ds, val_ds, test_ds = load_data_from_npz()
    except Exception as e:
        print(f"\n---!! An error occurred while loading data: {e}")
        print("Please check the 'load_data_from_npz' function in 'main.py' and ensure")
        print("your file paths and NPZ keys are correct. ---\n")
        # Exception erneut auslösen, um den vollen Traceback anzuzeigen
        raise

    # 2. Modelle instanziieren
    print("Instantiating models...")
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
    
    # 3. Modelle trainieren
    print("\n--- Starting Model Training ---")
    for name, model in models_to_compare.items():
        print(f"\nTraining {name}...")
        checkpoint_path = f"{name}_best_model.pth"
        
        model.train_model(
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LR,
            checkpoint_path=checkpoint_path,
            criterion=rope_loss  # Ihre benutzerdefinierte Loss-Funktion
        )
        
        # Die besten Gewichte für Evaluierung und Plotten laden
        print(f"Loading best weights for {name} from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    # 4. Modelle auf dem Test-Set evaluieren
    print("\n--- Starting Model Evaluation ---")
    test_losses = {}
    for name, model in models_to_compare.items():
        test_loss = model.evaluate_model(
            test_dataset=test_ds,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            criterion=rope_loss
        )
        test_losses[name] = test_loss
        print(f"Final Test Loss ({name}): {test_loss:.6f}")
# 5. Plot Comparison
    print("\n--- Plotting Model Comparison ---")
    
    # --- vvvv DEFINE YOUR FILENAME HERE vvvv ---
    plot_save_file = "model_comparison.png"
    # --- ^^^^ DEFINE YOUR FILENAME HERE ^^^^ ---
    
    print(f"Displaying plot and saving to {plot_save_file}...")
    
    # 'plot_model_comparison'-Funktion aus utils.py verwenden
    plot_model_comparison(
        models_dict=models_to_compare,
        dataset=test_ds,
        device=DEVICE,
        index=0,  # Das erste Sample aus dem Test-Set plotten
        denormalize=False, # auf True setzen, falls Ihre Daten normalisiert sind
        train_mean=None,   # Mittelwert angeben, falls denormalize=True
        train_std=None,    # Standardabweichung angeben, falls denormalize=True
        save_path=plot_save_file # <-- ADDED THIS LINE
    )
    
    print("Script finished.")

if __name__ == "__main__":
    main()