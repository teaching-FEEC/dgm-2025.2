import os
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

from tokenizer import SpectrogramPatchTokenizer, SpectrogramVQTokenizer


def apply_vq_tokenizer(model_path: str, spectrogram_dir: str, output_dir: str, patch_size: int = 16, device: str = 'cpu'):
    
    # Carrega checkpoint do modelo
    print(f"Carregando modelo de {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Inicializa modelo
    model = SpectrogramVQTokenizer(
        n_mels=model_config['n_mels'],
        patch_size=model_config['patch_size'],
        embedding_dim=model_config['embedding_dim'],
        num_embeddings=model_config['num_embeddings'],
        hidden_dim=model_config.get('hidden_dim', 256)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Modelo carregado!")
    print(f"  - Config: {model_config}")
    print(f"  - Train loss: {checkpoint['train_recon_loss']:.4f} (recon) + {checkpoint['train_vq_loss']:.4f} (vq)")
    print(f"  - Val loss: {checkpoint['val_recon_loss']:.4f} (recon) + {checkpoint['val_vq_loss']:.4f} (vq)")
    
    # Tokenizador de patches
    patch_tokenizer = SpectrogramPatchTokenizer(
        patch_size=patch_size,
        stride=patch_size,
        normalize=True
    )
    
    # Cria diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Lista arquivos
    spec_files = [f for f in os.listdir(spectrogram_dir) if f.endswith('.pt')]
    
    print(f"\nTokenizando {len(spec_files)} espectrogramas...")
    
    for fname in tqdm(spec_files):
        input_path = os.path.join(spectrogram_dir, fname)
        output_path = os.path.join(output_dir, fname.replace('.pt', '_vq_tokens.pt'))
        
        try:
            # Carrega espectrograma
            data = torch.load(input_path)
            spec = data['spec']
            
            # Extrai patches
            patches = patch_tokenizer.tokenize(spec)
            patches_batch = patches.unsqueeze(0).to(device)  # (1, num_patches, patch_dim)
            
            # Encode -> Quantize
            with torch.no_grad():
                z_q, indices = model.encode(patches_batch)
                patches_recon = model.decode(z_q)
            
            # Calcula erro de reconstrução
            recon_error = torch.nn.functional.mse_loss(
                patches_recon.squeeze(0).cpu(), 
                patches
            ).item()
            
            # Salva tokens discretos
            output_data = {
                'tokens': indices.squeeze(0).cpu(),  # (num_patches,)
                'embeddings': z_q.squeeze(0).cpu(),  # (num_patches, embedding_dim)
                'patches_original': patches,
                'patches_reconstructed': patches_recon.squeeze(0).cpu(),
                'reconstruction_error': recon_error,
                'num_patches': patches.shape[0],
                'patch_size': patch_size,
                'metadata': {
                    'sr': data.get('sr'),
                    'n_mels': data.get('n_mels'),
                    'n_fft': data.get('n_fft'),
                    'hop_length': data.get('hop_length')
                },
                'model_config': model_config
            }
            
            torch.save(output_data, output_path)
            
        except Exception as e:
            print(f"\nErro ao processar {fname}: {e}")
    
    print(f"\nTokenização VQ concluída! Arquivos salvos em {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aplicar VQ Tokenizer treinado em espectrogramas")
    parser.add_argument("--model", "-m", type=Path, required=True,
                        help="Caminho para o modelo treinado (.pt)")
    parser.add_argument("--input-dir", "-i", type=Path, required=True,
                        help="Diretório com espectrogramas")
    parser.add_argument("--output-dir", "-o", type=Path, required=True,
                        help="Diretório para salvar tokens VQ")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Tamanho dos patches")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Dispositivo (cuda/cpu)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    
    apply_vq_tokenizer(
        model_path=str(args.model),
        spectrogram_dir=str(args.input_dir),
        output_dir=str(args.output_dir),
        patch_size=args.patch_size,
        device=args.device
    )
