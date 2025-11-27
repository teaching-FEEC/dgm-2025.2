import numpy as np 
import os 
import torch 
from scipy.io import savemat, loadmat
import yaml

# read .yaml files
def read_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# given the config yaml file, get the data_dir and synthetic_data_dir from init_args and return 2 dicts, one for real data and one for synthetic data
# inside the dicts, have keys as 'MM', 'DN', and 'masks' (if masks exist)
def get_data_dirs(yaml_path):
    config = read_yaml(yaml_path)
    init_args = config['data']['init_args']
    data_dir = init_args['data_dir']
    synth_data_dir = init_args['synth_data_dir']
    
    real_data_dirs = {
        'MM': os.path.join(data_dir, 'MMCube'),
        'DN': os.path.join(data_dir, 'DNCube')
    }
    
    synth_data_dirs = {
        'MM': os.path.join(synth_data_dir, 'MMCube'),
        'DN': os.path.join(synth_data_dir, 'DNCube')
    }
    
    # check if masks exist
    mask_dir = os.path.join(data_dir, 'masks')
    if os.path.exists(mask_dir):
        real_data_dirs['masks'] = mask_dir
    
    synth_mask_dir = os.path.join(synth_data_dir, 'masks')
    if os.path.exists(synth_mask_dir):
        synth_data_dirs['masks'] = synth_mask_dir
    
    return real_data_dirs, synth_data_dirs



#create mixed data splits
def create_mixed_splits(real_indices, synth_indices, split_ratio, seed=42):
    np.random.seed(seed)
    np.random.shuffle(real_indices)
    np.random.shuffle(synth_indices)
    
    num_real = len(real_indices)
    num_synth = len(synth_indices)
    
    num_real_train = int(split_ratio[0] * num_real)
    num_real_val = int(split_ratio[1] * num_real)
    
    num_synth_train = int(split_ratio[0] * num_synth)
    num_synth_val = int(split_ratio[1] * num_synth)
    
    train_indices = np.concatenate((real_indices[:num_real_train], synth_indices[:num_synth_train]))
    val_indices = np.concatenate((real_indices[num_real_train:num_real_train + num_real_val], 
                                  synth_indices[num_synth_train:num_synth_train + num_synth_val]))
    test_indices = np.concatenate((real_indices[num_real_train + num_real_val:], 
                                   synth_indices[num_synth_train + num_synth_val:]))
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return train_indices, val_indices, test_indices


# create data fold directories
def create_datafold_dirs(base_path, fold_names):
    for fold in fold_names:
        fold_path = os.path.join(base_path, fold)
        os.makedirs(fold_path, exist_ok=True)
        print(f"Created directory: {fold_path}")        


