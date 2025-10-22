import os
import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectrogramPatchTokenizer: 
    
    def __init__(
        self, 
        patch_size: int = 16,
        stride: Optional[int] = None,
        normalize: bool = True
    ):       
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.normalize = normalize
    
    def tokenize(self, spectrogram: torch.Tensor) -> torch.Tensor:       
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)  # Adiciona dimensão de canal
        
        C, n_mels, time_steps = spectrogram.shape
        
        # Calcula número de patches
        num_patches = (time_steps - self.patch_size) // self.stride + 1
        
        patches = []
        for i in range(num_patches):
            start = i * self.stride
            end = start + self.patch_size
            patch = spectrogram[:, :, start:end]  # (C, n_mels, patch_size)
            
            # Flatten: (C, n_mels, patch_size) -> (C * n_mels * patch_size)
            patch_flat = patch.reshape(-1)
            
            if self.normalize:
                # Normaliza cada patch individualmente
                patch_mean = patch_flat.mean()
                patch_std = patch_flat.std()
                patch_flat = (patch_flat - patch_mean) / (patch_std + 1e-8)
            
            patches.append(patch_flat)
        
        return torch.stack(patches)  # (num_patches, C * n_mels * patch_size)
    
    def get_token_dim(self, n_mels: int, n_channels: int = 1) -> int:
        """Retorna a dimensão de cada token/patch."""
        return n_channels * n_mels * self.patch_size


class VectorQuantizer(nn.Module):
   
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25
    ):        
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: dicionário de embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      
        # Flatten para facilitar o cálculo de distâncias
        z_flat = z.reshape(-1, self.embedding_dim)  # (batch * seq_len, embedding_dim)
        
        # Calcula distâncias entre z e todos os embeddings do codebook
        # d(z, e) = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        
        # Encontra o embedding mais próximo
        encoding_indices = torch.argmin(distances, dim=1)  # (batch * seq_len)
        
        # Lookup dos embeddings quantizados
        z_q_flat = self.embedding(encoding_indices)
        
        # Reshape de volta
        z_q = z_q_flat.reshape(z.shape)
        
        # Calcula perdas
        # VQ loss: força os embeddings do codebook a se moverem em direção aos encodings
        vq_loss = F.mse_loss(z_q.detach(), z)
        # Commitment loss: força os encodings a se comprometerem com um embedding
        commitment_loss = F.mse_loss(z_q, z.detach())
        
        loss = vq_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: gradient flui através de z_q como se fosse z
        z_q = z + (z_q - z).detach()
        
        # Reshape indices
        indices = encoding_indices.reshape(z.shape[0], z.shape[1])
        
        return z_q, loss, indices
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:       
        return self.embedding(indices)


class SpectrogramVQTokenizer(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        patch_size: int = 16,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        hidden_dim: int = 256,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.commitment_cost = commitment_cost
        
        input_dim = n_mels * patch_size
        
        # Encoder: patch -> embedding contínuo
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder: embedding -> patch reconstruído
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      
        z = self.encoder(patches)  # (batch, num_patches, embedding_dim)
        z_q, _, indices = self.vq(z)
        return z_q, indices
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:      
        return self.decoder(z_q)
    
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(patches)
        z_q, vq_loss, indices = self.vq(z)
        patches_recon = self.decoder(z_q)
        
        return patches_recon, vq_loss, indices


def load_and_tokenize_spectrogram(
    pt_path: str,
    patch_tokenizer: SpectrogramPatchTokenizer
) -> Tuple[torch.Tensor, Dict]:
   
    data = torch.load(pt_path)
    
    spec = data['spec']
    metadata = {
        'mean': data['mean'],
        'std': data['std'],
        'sr': data['sr'],
        'n_fft': data['n_fft'],
        'hop_length': data['hop_length'],
        'n_mels': data.get('n_mels'),
        'is_log_mel': data.get('is_log_mel', True)
    }
    
    patches = patch_tokenizer.tokenize(spec)
    
    return patches, metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenizar espectrogramas em patches discretos")
    parser.add_argument("--input-dir", "-i", type=Path, required=True, 
                        help="Diretório com arquivos .pt de espectrograma")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, 
                        help="Diretório para salvar os tokens")
    parser.add_argument("--patch-size", type=int, default=16, 
                        help="Tamanho do patch temporal (frames)")
    parser.add_argument("--stride", type=int, default=None, 
                        help="Stride entre patches (default: sem overlap)")
    parser.add_argument("--normalize-patches", action="store_true", 
                        help="Normalizar cada patch individualmente")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    
    input_dir = str(args.input_dir)
    output_dir = str(args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializa o tokenizador
    tokenizer = SpectrogramPatchTokenizer(
        patch_size=args.patch_size,
        stride=args.stride,
        normalize=args.normalize_patches
    )
    
    print(f"Tokenizando espectrogramas de {input_dir}")
    print(f"Patch size: {args.patch_size}, Stride: {args.stride or args.patch_size}")
    
    # Processa cada arquivo .pt
    for fname in os.listdir(input_dir):
        if not fname.endswith('.pt'):
            continue
        
        input_path = os.path.join(input_dir, fname)
        
        try:
            patches, metadata = load_and_tokenize_spectrogram(input_path, tokenizer)
            
            # Salva os patches tokenizados
            output_data = {
                'patches': patches,
                'num_patches': patches.shape[0],
                'patch_dim': patches.shape[1],
                'patch_size': args.patch_size,
                'stride': args.stride or args.patch_size,
                'metadata': metadata
            }
            
            output_path = os.path.join(output_dir, fname.replace('.pt', '_tokens.pt'))
            torch.save(output_data, output_path)
            
            print(f"✓ {fname}: {patches.shape[0]} patches extraídos -> {output_path}")
            
        except Exception as e:
            print(f" Erro ao processar {fname}: {e}")
    
    print(f"\nTokenização concluída! Arquivos salvos em {output_dir}")
