import random
from pathlib import Path
import shutil

token_dir = Path("data/tokenized_spectrograms")
train_dir = token_dir / "train"
val_dir = token_dir / "val"

train_dir.mkdir(exist_ok=True)
val_dir.mkdir(exist_ok=True)

token_files = list(token_dir.glob("*.pt"))

random.shuffle(token_files)
split_idx = int(0.9 * len(token_files))
train_files = token_files[:split_idx]
val_files = token_files[split_idx:]

for f in train_files:
    shutil.copy(f, train_dir / f.name)
for f in val_files:
    shutil.copy(f, val_dir / f.name)

print(f"Treino: {len(train_files)} arquivos")
print(f"Validação: {len(val_files)} arquivos")