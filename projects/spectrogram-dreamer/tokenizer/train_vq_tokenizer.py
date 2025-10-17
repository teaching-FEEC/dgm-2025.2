import os
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from tokenizer import SpectrogramVQTokenizer, SpectrogramPatchTokenizer


class SpectrogramPatchDataset(Dataset):
       
    def __init__(self, pt_files: List[str], patch_tokenizer: SpectrogramPatchTokenizer):
       
        self.pt_files = pt_files
        self.patch_tokenizer = patch_tokenizer
        
        # Carrega todos os patches na memória (para datasets pequenos)
        # Para datasets grandes, considere lazy loading
        print("Carregando e tokenizando espectrogramas...")
        self.all_patches = []
        
        for pt_file in tqdm(pt_files):
            try:
                data = torch.load(pt_file)
                spec = data['spec']
                patches = patch_tokenizer.tokenize(spec)
                self.all_patches.append(patches)
            except Exception as e:
                print(f"Erro ao carregar {pt_file}: {e}")
        
        # Concatena todos os patches
        if len(self.all_patches) > 0:
            self.all_patches = torch.cat(self.all_patches, dim=0)
            print(f"Total de patches carregados: {len(self.all_patches)}")
        else:
            raise ValueError("Nenhum patch foi carregado!")
    
    def __len__(self):
        return len(self.all_patches)
    
    def __getitem__(self, idx):
        return self.all_patches[idx]


def train_vq_tokenizer(
    model: SpectrogramVQTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_dir: str
) -> SpectrogramVQTokenizer:   
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Treinamento
        model.train()
        train_recon_loss = 0.0
        train_vq_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Época {epoch+1}/{num_epochs} [Train]"):
            batch = batch.unsqueeze(0).to(device)  # (1, patch_dim) -> (batch=1, 1, patch_dim)
            
            optimizer.zero_grad()
            
            patches_recon, vq_loss, _ = model(batch)
            
            # Perda de reconstrução
            recon_loss = nn.functional.mse_loss(patches_recon, batch)
            
            # Perda total
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            train_recon_loss += recon_loss.item()
            train_vq_loss += vq_loss.item()
        
        train_recon_loss /= len(train_loader)
        train_vq_loss /= len(train_loader)
        
        # Validação
        model.eval()
        val_recon_loss = 0.0
        val_vq_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Época {epoch+1}/{num_epochs} [Val]"):
                batch = batch.unsqueeze(0).to(device)
                
                patches_recon, vq_loss, _ = model(batch)
                recon_loss = nn.functional.mse_loss(patches_recon, batch)
                
                val_recon_loss += recon_loss.item()
                val_vq_loss += vq_loss.item()
        
        val_recon_loss /= len(val_loader)
        val_vq_loss /= len(val_loader)
        
        print(f"\nÉpoca {epoch+1}/{num_epochs}")
        print(f"  Train - Recon Loss: {train_recon_loss:.4f}, VQ Loss: {train_vq_loss:.4f}")
        print(f"  Val   - Recon Loss: {val_recon_loss:.4f}, VQ Loss: {val_vq_loss:.4f}")
        
        # Salva melhor modelo
        if val_recon_loss + val_vq_loss < best_val_loss:
            best_val_loss = val_recon_loss + val_vq_loss
            checkpoint_path = os.path.join(save_dir, 'best_vq_tokenizer.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_recon_loss': train_recon_loss,
                'train_vq_loss': train_vq_loss,
                'val_recon_loss': val_recon_loss,
                'val_vq_loss': val_vq_loss,
                'model_config': {
                    'n_mels': model.n_mels,
                    'patch_size': model.patch_size,
                    'embedding_dim': model.embedding_dim,
                    'num_embeddings': model.num_embeddings
                }
            }, checkpoint_path)
            print(f"  ✓ Modelo salvo em {checkpoint_path}")
    
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinar VQ Tokenizer para espectrogramas")
    parser.add_argument("--data-dir", "-d", type=Path, required=True, 
                        help="Diretório com arquivos .pt de espectrograma")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, 
                        help="Diretório para salvar o modelo treinado")
    parser.add_argument("--patch-size", type=int, default=16, 
                        help="Tamanho do patch temporal")
    parser.add_argument("--embedding-dim", type=int, default=64, 
                        help="Dimensão dos embeddings")
    parser.add_argument("--num-embeddings", type=int, default=512, 
                        help="Tamanho do codebook (vocabulário)")
    parser.add_argument("--hidden-dim", type=int, default=256, 
                        help="Dimensão das camadas ocultas")
    parser.add_argument("--n-mels", type=int, default=80, 
                        help="Número de bandas mel")
    parser.add_argument("--batch-size", type=int, default=256, 
                        help="Tamanho do batch")
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Número de épocas")
    parser.add_argument("--lr", type=float, default=1e-3, 
                        help="Taxa de aprendizado")
    parser.add_argument("--val-split", type=float, default=0.1, 
                        help="Proporção dos dados para validação")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help="Dispositivo (cuda/cpu)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    
    # Prepara diretórios
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Lista arquivos .pt
    pt_files = [
        os.path.join(args.data_dir, f) 
        for f in os.listdir(args.data_dir) 
        if f.endswith('.pt')
    ]
    
    if len(pt_files) == 0:
        raise ValueError(f"Nenhum arquivo .pt encontrado em {args.data_dir}")
    
    print(f"Encontrados {len(pt_files)} arquivos de espectrograma")
    
    # Split train/val
    np.random.shuffle(pt_files)
    split_idx = int(len(pt_files) * (1 - args.val_split))
    train_files = pt_files[:split_idx]
    val_files = pt_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Tokenizador de patches
    patch_tokenizer = SpectrogramPatchTokenizer(
        patch_size=args.patch_size,
        stride=args.patch_size,  # Sem overlap
        normalize=True
    )
    
    # Datasets
    train_dataset = SpectrogramPatchDataset(train_files, patch_tokenizer)
    val_dataset = SpectrogramPatchDataset(val_files, patch_tokenizer)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Windows pode ter problemas com multiprocessing
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    # Modelo
    model = SpectrogramVQTokenizer(
        n_mels=args.n_mels,
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        hidden_dim=args.hidden_dim
    )
    
    print(f"\nModelo VQ Tokenizer:")
    print(f"  - Input: ({args.n_mels} mels) x ({args.patch_size} frames) = {args.n_mels * args.patch_size} dim")
    print(f"  - Embedding dim: {args.embedding_dim}")
    print(f"  - Codebook size: {args.num_embeddings}")
    print(f"  - Device: {args.device}")
    
    # Treina
    device = torch.device(args.device)
    trained_model = train_vq_tokenizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=str(args.output_dir)
    )
    
    print(f"\n Treinamento concluído! Modelo salvo em {args.output_dir}")
