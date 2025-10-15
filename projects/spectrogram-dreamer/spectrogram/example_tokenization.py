import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from tokenizer import (
    SpectrogramPatchTokenizer, 
    SpectrogramVQTokenizer,
    load_and_tokenize_spectrogram
)


def visualize_patches(patches: torch.Tensor, n_mels: int, patch_size: int, max_patches: int = 10):
    """Visualiza alguns patches do espectrograma."""
    num_patches = min(max_patches, patches.shape[0])
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_patches):
        patch = patches[i].reshape(n_mels, patch_size).numpy()
        axes[i].imshow(patch, aspect='auto', origin='lower', cmap='magma')
        axes[i].set_title(f'Patch {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('patches_visualization.png')
    print("Visualização dos patches salva em 'patches_visualization.png'")
    plt.close()


def visualize_reconstruction(
    original_patches: torch.Tensor,
    reconstructed_patches: torch.Tensor,
    n_mels: int,
    patch_size: int,
    num_samples: int = 5
):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Original
        orig_patch = original_patches[i].reshape(n_mels, patch_size).numpy()
        axes[0, i].imshow(orig_patch, aspect='auto', origin='lower', cmap='magma')
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')
        
        # Reconstruído
        recon_patch = reconstructed_patches[i].reshape(n_mels, patch_size).numpy()
        axes[1, i].imshow(recon_patch, aspect='auto', origin='lower', cmap='magma')
        axes[1, i].set_title(f'Reconstruído {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png')
    print("Comparação de reconstrução salva em 'reconstruction_comparison.png'")
    plt.close()


def analyze_codebook_usage(indices: torch.Tensor, num_embeddings: int):
    indices_flat = indices.flatten().numpy()
    unique_indices = np.unique(indices_flat)
    
    print(f"\nAnálise do Codebook:")
    print(f"  - Tamanho do codebook: {num_embeddings}")
    print(f"  - Embeddings únicos usados: {len(unique_indices)}")
    print(f"  - Utilização: {len(unique_indices)/num_embeddings*100:.1f}%")
    
    # Histograma de uso
    plt.figure(figsize=(12, 4))
    plt.hist(indices_flat, bins=num_embeddings, edgecolor='black')
    plt.xlabel('Índice do Embedding')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Uso do Codebook')
    plt.tight_layout()
    plt.savefig('codebook_usage.png')
    print("Histograma de uso do codebook salvo em 'codebook_usage.png'")
    plt.close()


def main():
    print("=" * 60)
    print("Demonstração do Sistema de Tokenização de Espectrogramas")
    print("=" * 60)
    
    # Configurações
    SPEC_FILE = Path("data/espectrogramas/exemplo.pt")  # Ajuste para seu arquivo
    PATCH_SIZE = 16
    N_MELS = 80
    EMBEDDING_DIM = 64
    NUM_EMBEDDINGS = 512
    
    print(f"\nCarregando espectrograma de: {SPEC_FILE}")
    
    # ============================================
    # 1. TOKENIZAÇÃO SIMPLES (PATCHES)
    # ============================================
    print("\n" + "="*60)
    print("ETAPA 1: Tokenização Simples (Patches)")
    print("="*60)
    
    patch_tokenizer = SpectrogramPatchTokenizer(
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,  # Sem overlap
        normalize=True
    )
    
    patches, metadata = load_and_tokenize_spectrogram(str(SPEC_FILE), patch_tokenizer)
    
    print(f"\nEspectrograma tokenizado!")
    print(f"  - Shape dos patches: {patches.shape}")
    print(f"  - Número de patches: {patches.shape[0]}")
    print(f"  - Dimensão de cada patch: {patches.shape[1]}")
    print(f"  - Metadados: sr={metadata['sr']}, n_mels={metadata['n_mels']}")
    
    # Visualiza alguns patches
    visualize_patches(patches, N_MELS, PATCH_SIZE)
    
    # ============================================
    # 2. VQ TOKENIZAÇÃO (DISCRETIZAÇÃO)
    # ============================================
    print("\n" + "="*60)
    print("ETAPA 2: VQ Tokenização (Tokens Discretos)")
    print("="*60)
    
    # Inicializa o modelo VQ
    vq_model = SpectrogramVQTokenizer(
        n_mels=N_MELS,
        patch_size=PATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_embeddings=NUM_EMBEDDINGS,
        hidden_dim=256
    )
    
    # Para demonstração, vamos apenas fazer um forward pass
    # (em produção, você carregaria um modelo treinado)
    print("\nUsando modelo NÃO treinado (apenas para demonstração)")
    print("   Para resultados reais, treine o modelo primeiro com:")
    print("   python train_vq_tokenizer.py --data-dir <path> --output-dir ./models")
    
    vq_model.eval()
    with torch.no_grad():
        patches_batch = patches.unsqueeze(0)  # (1, num_patches, patch_dim)
        
        # Encode -> Quantize -> Get indices
        z_q, indices = vq_model.encode(patches_batch)
        
        # Reconstruct
        patches_recon = vq_model.decode(z_q)
    
    print(f"\nVQ Tokenização concluída!")
    print(f"  - Shape dos tokens: {indices.shape}")
    print(f"  - Tokens únicos: {len(torch.unique(indices))}")
    print(f"  - Range de tokens: [{indices.min().item()}, {indices.max().item()}]")
    
    # Analisa uso do codebook
    analyze_codebook_usage(indices, NUM_EMBEDDINGS)
    
    # Visualiza reconstrução
    visualize_reconstruction(
        patches[:5], 
        patches_recon.squeeze(0)[:5].detach(),
        N_MELS,
        PATCH_SIZE
    )
    
    # ============================================
    # 3. PREPARAÇÃO PARA O MODELO DE MUNDO
    # ============================================
    print("\n" + "="*60)
    print("ETAPA 3: Preparação para o Modelo de Mundo (RSSM)")
    print("="*60)
    
    # Os tokens discretos estão prontos para serem usados
    # como observações no modelo de mundo
    print(f"\n Sequência de tokens discretos pronta:")
    print(f"  - Formato: {indices.shape}")
    print(f"  - Cada token representa {PATCH_SIZE} frames de áudio")
    print(f"  - Vocabulário: {NUM_EMBEDDINGS} tokens possíveis")
    
    print(f"\n Próximos passos:")
    print(f"  1. Treinar o VQ Tokenizer em todo o dataset")
    print(f"  2. Tokenizar todos os espectrogramas")
    print(f"  3. Criar dataset de sequências de tokens")
    print(f"  4. Implementar o RSSM (Recurrent State Space Model)")
    print(f"  5. Treinar o modelo de mundo com as sequências")
    
    # Salva exemplo de saída
    output_data = {
        'patches': patches,
        'tokens': indices,
        'embeddings': z_q,
        'metadata': metadata,
        'config': {
            'patch_size': PATCH_SIZE,
            'n_mels': N_MELS,
            'embedding_dim': EMBEDDING_DIM,
            'num_embeddings': NUM_EMBEDDINGS
        }
    }
    
    torch.save(output_data, 'tokenization_example_output.pt')
    print(f"\n✓ Saída de exemplo salva em 'tokenization_example_output.pt'")
    
    print("\n" + "="*60)
    print("Demonstração concluída!")
    print("="*60)


if __name__ == "__main__":
    # Verifica se o arquivo de exemplo existe
    example_file = Path("data/espectrogramas/exemplo.pt")
    
    if not example_file.exists():
        print(f"Arquivo de exemplo não encontrado: {example_file}")
        print("\nPara rodar este exemplo:")
        print("1. Primeiro gere espectrogramas com:")
        print("   python spectogram.py --input-dir <audio_dir> --output-dir ./data/espectrogramas")
        print("2. Copie um arquivo .pt para ./data/espectrogramas/exemplo.pt")
        print("3. Execute este script novamente")
    else:
        main()
