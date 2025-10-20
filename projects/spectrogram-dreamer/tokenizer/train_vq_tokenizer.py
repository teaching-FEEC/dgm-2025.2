import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import time
from datetime import datetime

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
    
    # Estruturas para armazenar histórico de métricas
    history = {
        'train_recon_loss': [],
        'train_vq_loss': [],
        'train_total_loss': [],
        'val_recon_loss': [],
        'val_vq_loss': [],
        'val_total_loss': [],
        'learning_rates': [],
        'epochs': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }
    
    # Informações de treino
    training_info = {
        'start_time': datetime.now().isoformat(),
        'model_config': {
            'n_mels': model.n_mels,
            'patch_size': model.patch_size,
            'embedding_dim': model.embedding_dim,
            'num_embeddings': model.num_embeddings,
            'hidden_dim': model.hidden_dim
        },
        'training_config': {
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': train_loader.batch_size,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'device': str(device)
        }
    }
    
    start_time = time.time()
    
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
        
        # Armazenar métricas de treino
        train_total_loss = train_recon_loss + train_vq_loss
        history['train_recon_loss'].append(train_recon_loss)
        history['train_vq_loss'].append(train_vq_loss)
        history['train_total_loss'].append(train_total_loss)
        history['learning_rates'].append(learning_rate)
        history['epochs'].append(epoch + 1)
        
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
        
        # Armazenar métricas de validação
        val_total_loss = val_recon_loss + val_vq_loss
        history['val_recon_loss'].append(val_recon_loss)
        history['val_vq_loss'].append(val_vq_loss)
        history['val_total_loss'].append(val_total_loss)
        
        print(f"\nÉpoca {epoch+1}/{num_epochs}")
        print(f"  Train - Recon Loss: {train_recon_loss:.4f}, VQ Loss: {train_vq_loss:.4f}, Total: {train_total_loss:.4f}")
        print(f"  Val   - Recon Loss: {val_recon_loss:.4f}, VQ Loss: {val_vq_loss:.4f}, Total: {val_total_loss:.4f}")
        
        # Salva melhor modelo
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            history['best_epoch'] = epoch + 1
            history['best_val_loss'] = best_val_loss
            checkpoint_path = os.path.join(save_dir, 'best_vq_tokenizer.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_recon_loss': train_recon_loss,
                'train_vq_loss': train_vq_loss,
                'val_recon_loss': val_recon_loss,
                'val_vq_loss': val_vq_loss,
                'best_val_loss': best_val_loss,
                'model_config': {
                    'n_mels': model.n_mels,
                    'patch_size': model.patch_size,
                    'embedding_dim': model.embedding_dim,
                    'num_embeddings': model.num_embeddings,
                    'hidden_dim': model.hidden_dim
                }
            }, checkpoint_path)
            print(f"  ✓ Modelo salvo em {checkpoint_path}")
        
        # Salvar histórico periodicamente (a cada 10 épocas)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            history_path = os.path.join(save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
    
    # Finalizar informações de treino
    end_time = time.time()
    training_time = end_time - start_time
    
    training_info['end_time'] = datetime.now().isoformat()
    training_info['total_time_seconds'] = training_time
    training_info['total_time_formatted'] = f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}min {training_time%60:.0f}s"
    training_info['final_train_loss'] = history['train_total_loss'][-1]
    training_info['final_val_loss'] = history['val_total_loss'][-1]
    training_info['best_val_loss'] = history['best_val_loss']
    training_info['best_epoch'] = history['best_epoch']
    
    # Salvar informações
    info_path = os.path.join(save_dir, 'training_info.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    # Salvar histórico final
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    csv_path = os.path.join(save_dir, 'training_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_recon_loss,train_vq_loss,train_total_loss,val_recon_loss,val_vq_loss,val_total_loss,learning_rate\n')
        for i in range(len(history['epochs'])):
            f.write(f"{history['epochs'][i]},{history['train_recon_loss'][i]:.6f},{history['train_vq_loss'][i]:.6f},"
                   f"{history['train_total_loss'][i]:.6f},{history['val_recon_loss'][i]:.6f},{history['val_vq_loss'][i]:.6f},"
                   f"{history['val_total_loss'][i]:.6f},{history['learning_rates'][i]:.6f}\n")
    
    print()
    print(f"Tempo total: {training_info['total_time_formatted']}")
    print(f"Melhor época: {history['best_epoch']}")
    print(f"Melhor val loss: {history['best_val_loss']:.4f}")
    
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinar VQ Tokenizer para espectrogramas")
    parser.add_argument("--data-dir", "-d", type=Path, required=True, 
                        help="Diretório com arquivos .pt de espectrograma (TRAIN)")
    parser.add_argument("--val-dir", type=Path, default=None,
                        help="Diretório com arquivos .pt de validação (opcional)")
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
                        help="Proporção dos dados para validação (usado se --val-dir não fornecido)")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help="Dispositivo (cuda/cpu)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    
    # Prepara diretórios
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Lista arquivos .pt de treino
    train_files = [
        os.path.join(args.data_dir, f) 
        for f in os.listdir(args.data_dir) 
        if f.endswith('.pt')
    ]
    
    if len(train_files) == 0:
        raise ValueError(f"Nenhum arquivo .pt encontrado em {args.data_dir}")
    
    # Validação
    if args.val_dir is not None:
        # Usa diretório de validação separado
        val_files = [
            os.path.join(args.val_dir, f) 
            for f in os.listdir(args.val_dir) 
            if f.endswith('.pt')
        ]
        if len(val_files) == 0:
            raise ValueError(f"Nenhum arquivo .pt encontrado em {args.val_dir}")
        print(f"Encontrados {len(train_files)} arquivos de treino e {len(val_files)} de validação (diretórios separados)")
    else:
        # Faz split interno aleatório
        print(f"Encontrados {len(train_files)} arquivos de espectrograma")
        np.random.shuffle(train_files)
        split_idx = int(len(train_files) * (1 - args.val_split))
        val_files = train_files[split_idx:]
        train_files = train_files[:split_idx]
        print(f"Split interno: Train: {len(train_files)}, Val: {len(val_files)}")
    
    print(f"Total - Train: {len(train_files)}, Val: {len(val_files)}")
    
    patch_tokenizer = SpectrogramPatchTokenizer(
        patch_size=args.patch_size,
        stride=args.patch_size,  # Sem overlap
        normalize=True
    )
    
    train_dataset = SpectrogramPatchDataset(train_files, patch_tokenizer)
    val_dataset = SpectrogramPatchDataset(val_files, patch_tokenizer)
    
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
