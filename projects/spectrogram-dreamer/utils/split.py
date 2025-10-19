import argparse
import json
import shutil
from pathlib import Path
import random

def create_splits(
    source_dir: Path,
    output_base_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    pt_files = sorted(list(source_dir.glob("*.pt")))
    print(f"Encontrados {len(pt_files)} arquivos")
    
    random.seed(seed)
    random.shuffle(pt_files)
    
    n_total = len(pt_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = pt_files[:n_train]
    val_files = pt_files[n_train:n_train + n_val]
    test_files = pt_files[n_train + n_val:]
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_files)} ({len(train_files)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_files)} ({len(val_files)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_files)} ({len(test_files)/n_total*100:.1f}%)")
    
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_dir = output_base_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopiando {split_name}...")
        for file in files:
            shutil.copy2(file, split_dir / file.name)
    
    metadata = {
        'seed': seed,
        'total_files': n_total,
        'train': {'n_files': len(train_files), 'ratio': train_ratio},
        'val': {'n_files': len(val_files), 'ratio': val_ratio},
        'test': {'n_files': len(test_files), 'ratio': test_ratio}
    }
    
    with open(output_base_dir / 'splits_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)    


parser = argparse.ArgumentParser()
parser.add_argument('--source-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

create_splits(args.source_dir, args.output_dir, seed=args.seed)