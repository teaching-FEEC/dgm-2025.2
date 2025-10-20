import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def load_training_data(model_dir: Path):
    history_path = model_dir / 'training_history.json'
    info_path = model_dir / 'training_info.json'
    
    if not history_path.exists():
        raise FileNotFoundError(f"Histórico não encontrado: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    info = None
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
    
    return history, info


def plot_losses(history: dict, save_path: Path = None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VQ Tokenizer Training Metrics', fontsize=16)
    
    epochs = history['epochs']
    
    # Loss total
    ax = axes[0, 0]
    ax.plot(epochs, history['train_total_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_total_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss (Recon + VQ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reconstruction loss
    ax = axes[0, 1]
    ax.plot(epochs, history['train_recon_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_recon_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss (MSE)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # VQ loss
    ax = axes[1, 0]
    ax.plot(epochs, history['train_vq_loss'], label='Train', linewidth=2)
    ax.plot(epochs, history['val_vq_loss'], label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('VQ Loss')
    ax.set_title('Vector Quantization Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Train vs Val comparison
    ax = axes[1, 1]
    train_val_ratio = [t/v if v > 0 else 1.0 for t, v in 
                       zip(history['train_total_loss'], history['val_total_loss'])]
    ax.plot(epochs, train_val_ratio, linewidth=2, color='purple')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect match')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train/Val Ratio')
    ax.set_title('Overfitting Check (Train/Val Ratio)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico salvo em: {save_path}")
    
    plt.close()


parser = argparse.ArgumentParser(description='Analisar treino VQ Tokenizer')
parser.add_argument('model_dir', type=str, help='Diretório do modelo treinado')
args = parser.parse_args()

model_dir = Path(args.model_dir)
history, info = load_training_data(model_dir)
save_path = model_dir / 'training_plots.png'
plot_losses(history, save_path=save_path)
