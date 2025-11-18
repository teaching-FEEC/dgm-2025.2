import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d


def load_tokens(token_file: str) -> Dict:
    return torch.load(token_file)


def analyze_token_statistics(token_file: str) -> Dict:  
    data = load_tokens(token_file)
    
    if 'patches' in data:
        patches = data['patches']
        stats = {
            'num_patches': patches.shape[0],
            'patch_dim': patches.shape[1],
            'patch_mean': patches.mean().item(),
            'patch_std': patches.std().item(),
            'patch_min': patches.min().item(),
            'patch_max': patches.max().item(),
        }
    else:
        stats = {}
    
    return stats


def visualize_token_sequence(
    indices: torch.Tensor,
    output_path: str = 'token_sequence.png',
    title: str = 'Sequência de Tokens Discretos'
):
   
    if indices.dim() == 2:
        indices = indices[0]  # Pega primeira sequência se batch
    
    indices_np = indices.cpu().numpy()
    
    plt.figure(figsize=(15, 3))
    plt.plot(indices_np, marker='o', markersize=3, linestyle='-', linewidth=0.5)
    plt.xlabel('Posição Temporal (patch index)')
    plt.ylabel('Token ID')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Sequência de tokens salva em {output_path}")


def visualize_token_matrix(
    indices: torch.Tensor,
    output_path: str = 'token_matrix.png',
    title: str = 'Matriz de Tokens'
):
    
    if indices.dim() == 2:
        indices = indices[0]
    
    indices_np = indices.cpu().numpy()
    
    # Reshape para matriz (aproximadamente quadrada)
    seq_len = len(indices_np)
    rows = int(np.sqrt(seq_len))
    cols = int(np.ceil(seq_len / rows))
    
    # Pad se necessário
    padded_len = rows * cols
    if len(indices_np) < padded_len:
        indices_np = np.pad(indices_np, (0, padded_len - len(indices_np)), constant_values=-1)
    
    matrix = indices_np[:rows*cols].reshape(rows, cols)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(matrix, aspect='auto', cmap='tab20', interpolation='nearest')
    plt.colorbar(label='Token ID')
    plt.xlabel('Tempo (cols)')
    plt.ylabel('Tempo (rows)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Matriz de tokens salva em {output_path}")


def analyze_token_distribution(
    indices_list: List[torch.Tensor],
    num_embeddings: int,
    output_path: str = 'token_distribution.png'
):
   
    # Concatena todos os índices
    all_indices = torch.cat([idx.flatten() for idx in indices_list]).cpu().numpy()
    
    # Conta frequências
    counter = Counter(all_indices)
    
    # Prepara dados
    token_ids = np.arange(num_embeddings)
    frequencies = np.array([counter.get(i, 0) for i in token_ids])
    
    # Estatísticas
    total_tokens = len(all_indices)
    unique_tokens = len(counter)
    usage_percent = (unique_tokens / num_embeddings) * 100
    
    print(f"\n📊 Estatísticas de Distribuição de Tokens:")
    print(f"  - Total de tokens: {total_tokens}")
    print(f"  - Tokens únicos usados: {unique_tokens}/{num_embeddings}")
    print(f"  - Utilização do vocabulário: {usage_percent:.1f}%")
    print(f"  - Token mais frequente: {counter.most_common(1)[0] if counter else 'N/A'}")
    
    # Visualiza
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Histograma completo
    axes[0].bar(token_ids, frequencies, width=1.0, edgecolor='none')
    axes[0].set_xlabel('Token ID')
    axes[0].set_ylabel('Frequência')
    axes[0].set_title('Distribuição de Tokens (Completa)')
    axes[0].grid(True, alpha=0.3)
    
    # Top 50 tokens mais frequentes
    top_n = 50
    top_tokens = counter.most_common(top_n)
    if top_tokens:
        top_ids, top_freqs = zip(*top_tokens)
        axes[1].bar(range(len(top_ids)), top_freqs)
        axes[1].set_xlabel(f'Rank (Top {top_n})')
        axes[1].set_ylabel('Frequência')
        axes[1].set_title(f'Top {top_n} Tokens Mais Frequentes')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Distribuição de tokens salva em {output_path}")


def compare_token_sequences(
    indices1: torch.Tensor,
    indices2: torch.Tensor,
    labels: Tuple[str, str] = ('Sequência 1', 'Sequência 2'),
    output_path: str = 'token_comparison.png'
):
    """
    Compara duas sequências de tokens lado a lado.
    """
    if indices1.dim() == 2:
        indices1 = indices1[0]
    if indices2.dim() == 2:
        indices2 = indices2[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 6))
    
    # Sequência 1
    axes[0].plot(indices1.cpu().numpy(), marker='o', markersize=2, linestyle='-', linewidth=0.5)
    axes[0].set_ylabel('Token ID')
    axes[0].set_title(labels[0])
    axes[0].grid(True, alpha=0.3)
    
    # Sequência 2
    axes[1].plot(indices2.cpu().numpy(), marker='o', markersize=2, linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Posição Temporal')
    axes[1].set_ylabel('Token ID')
    axes[1].set_title(labels[1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Comparação salva em {output_path}")


def analyze_temporal_patterns(
    indices: torch.Tensor,
    window_size: int = 10,
    output_path: str = 'temporal_patterns.png'
):
   
    if indices.dim() == 2:
        indices = indices[0]
    
    indices_np = indices.cpu().numpy()
    
    # Calcula transições
    transitions = np.diff(indices_np)
    
    # Calcula repetições (tokens consecutivos iguais)
    repetitions = (transitions == 0).astype(int)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 9))
    
    # Tokens originais
    axes[0].plot(indices_np, marker='o', markersize=2, linestyle='-', linewidth=0.5)
    axes[0].set_ylabel('Token ID')
    axes[0].set_title('Sequência Original')
    axes[0].grid(True, alpha=0.3)
    
    # Magnitude das transições
    axes[1].plot(np.abs(transitions), color='orange')
    axes[1].set_ylabel('|Transição|')
    axes[1].set_title('Magnitude das Transições entre Tokens')
    axes[1].grid(True, alpha=0.3)
    
    # Repetições (smoothed)
    repetitions_smooth = uniform_filter1d(repetitions.astype(float), size=window_size)
    axes[2].plot(repetitions_smooth, color='green')
    axes[2].set_xlabel('Posição Temporal')
    axes[2].set_ylabel('Taxa de Repetição')
    axes[2].set_title(f'Taxa de Repetição (janela={window_size})')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # Estatísticas
    print(f"\nAnálise de Padrões Temporais:")
    print(f"  - Transição média: {np.abs(transitions).mean():.2f}")
    print(f"  - Transição máxima: {np.abs(transitions).max():.0f}")
    print(f"  - Taxa de repetição: {repetitions.mean()*100:.1f}%")
    print(f"✓ Análise temporal salva em {output_path}")


def reconstruct_spectrogram_from_patches(
    patches: torch.Tensor,
    n_mels: int = 80,
    patch_size: int = 16
) -> torch.Tensor:  
    num_patches = patches.shape[0]
    
    # Reshape patches: (num_patches, n_mels * patch_size) -> (num_patches, n_mels, patch_size)
    patches_reshaped = patches.view(num_patches, n_mels, patch_size)
    
    # Concatena temporalmente: (n_mels, num_patches * patch_size)
    spectrogram = patches_reshaped.permute(1, 0, 2).contiguous().view(n_mels, -1)
    
    return spectrogram


def compare_spectrograms(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    output_path: str = 'spectrogram_comparison.png',
    title_prefix: str = 'Comparação'
):   
    # Converte para numpy
    original_np = original.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    
    # Calcula diferença
    diff = np.abs(original_np - reconstructed_np)
    
    # Métricas
    mse = np.mean((original_np - reconstructed_np) ** 2)
    mae = np.mean(diff)
    max_error = np.max(diff)
    
    # Correlação de Pearson
    corr, _ = pearsonr(original_np.flatten(), reconstructed_np.flatten())
    
    # Visualiza
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # Original
    im0 = axes[0].imshow(original_np, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'{title_prefix} - Original')
    axes[0].set_ylabel('Mel Band')
    plt.colorbar(im0, ax=axes[0])
    
    # Reconstruído
    im1 = axes[1].imshow(reconstructed_np, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'{title_prefix} - Reconstruído pelo VQ Tokenizer')
    axes[1].set_ylabel('Mel Band')
    plt.colorbar(im1, ax=axes[1])
    
    # Diferença absoluta
    im2 = axes[2].imshow(diff, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    axes[2].set_title(f'Diferença Absoluta (MSE={mse:.4f}, MAE={mae:.4f}, Max={max_error:.4f})')
    axes[2].set_ylabel('Mel Band')
    plt.colorbar(im2, ax=axes[2])
    
    # Distribuição de erros
    axes[3].hist(diff.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[3].set_xlabel('Erro Absoluto')
    axes[3].set_ylabel('Frequência')
    axes[3].set_title(f'Distribuição de Erros (Correlação: {corr:.4f})')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nMétricas de Reconstrução:")
    print(f"  - MSE (Mean Squared Error): {mse:.6f}")
    print(f"  - MAE (Mean Absolute Error): {mae:.6f}")
    print(f"  - Erro Máximo: {max_error:.6f}")
    print(f"  - Correlação de Pearson: {corr:.6f}")
    print(f"Comparação salva em {output_path}")
    
    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'correlation': corr
    }


def validate_vq_reconstruction(
    token_file: str,
    original_spectrogram_file: str,
    output_path: str = 'vq_validation.png',
    n_mels: int = 80,
    patch_size: int = 16
):
    print(f"\n{'='*60}")
    print(f"Validando Reconstrução VQ")
    print(f"{'='*60}")
    
    # Carrega dados
    print(f"Carregando tokens VQ: {os.path.basename(token_file)}")
    vq_data = torch.load(token_file)
    
    print(f"Carregando espectrograma original: {os.path.basename(original_spectrogram_file)}")
    spec_data = torch.load(original_spectrogram_file)
    
    # Extrai espectrograma (pode ser dict ou tensor)
    if isinstance(spec_data, dict):
        spec_original = spec_data['spec']
    else:
        spec_original = spec_data
    
    # Remove batch dimension se existir
    if spec_original.dim() == 3:
        spec_original = spec_original.squeeze(0)  # (1, 80, T) -> (80, T)
    
    # Verifica se tem patches reconstruídos
    if 'patches_reconstructed' in vq_data:
        print("Usando patches reconstruídos do VQ")
        patches_reconstructed = vq_data['patches_reconstructed']
    else:
        print("Patches reconstruídos não encontrados no arquivo VQ")
        return None
    
    # Reconstrói espectrograma
    print("Reconstruindo espectrograma a partir dos patches...")
    spec_reconstructed = reconstruct_spectrogram_from_patches(
        patches_reconstructed,
        n_mels=n_mels,
        patch_size=patch_size
    )
    
    # Ajusta tamanho se necessário (pode ter padding)
    min_time = min(spec_original.shape[1], spec_reconstructed.shape[1])
    spec_original = spec_original[:, :min_time]
    spec_reconstructed = spec_reconstructed[:, :min_time]
    
    print(f"Shape original: {spec_original.shape}")
    print(f"Shape reconstruído: {spec_reconstructed.shape}")
    
    # Compara
    base_name = os.path.splitext(os.path.basename(token_file))[0]
    metrics = compare_spectrograms(
        spec_original,
        spec_reconstructed,
        output_path=output_path,
        title_prefix=base_name
    )
    
    # Salva também as reconstruções separadas
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    im0 = axes[0].imshow(spec_original.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Espectrograma Original')
    axes[0].set_xlabel('Tempo (frames)')
    axes[0].set_ylabel('Mel Band')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(spec_reconstructed.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title('Espectrograma Reconstruído (VQ)')
    axes[1].set_xlabel('Tempo (frames)')
    axes[1].set_ylabel('Mel Band')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    side_by_side_path = output_path.replace('.png', '_side_by_side.png')
    plt.savefig(side_by_side_path, dpi=150)
    plt.close()
    
    print(f"✓ Comparação lado a lado salva em {side_by_side_path}")
    
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analisar e visualizar tokens de espectrogramas")
    parser.add_argument("--token-dir", "-t", type=Path, required=True,
                        help="Diretório com arquivos de tokens (*_tokens.pt)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("token_analysis"),
                        help="Diretório para salvar visualizações")
    parser.add_argument("--num-embeddings", type=int, default=512,
                        help="Tamanho do vocabulário do modelo")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Número máximo de arquivos a analisar")
    parser.add_argument("--validate-reconstruction", action="store_true",
                        help="Validar reconstrução comparando com espectrogramas originais")
    parser.add_argument("--spectrogram-dir", type=Path, default=None,
                        help="Diretório com espectrogramas originais (.pt)")
    parser.add_argument("--n-mels", type=int, default=80,
                        help="Número de bandas mel")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="Tamanho do patch temporal")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    
    output_dir = str(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Análise e Visualização de Tokens")
    print("=" * 60)
    
    # Lista arquivos de tokens
    token_files = [
        os.path.join(args.token_dir, f)
        for f in os.listdir(args.token_dir)
        if f.endswith('_tokens.pt')
    ]
    
    if len(token_files) == 0:
        print(f"Nenhum arquivo *_tokens.pt encontrado em {args.token_dir}")
        exit(1)
    
    if args.max_files:
        token_files = token_files[:args.max_files]
    
    print(f"\nEncontrados {len(token_files)} arquivos de tokens")
    
    # Carrega todos os tokens
    all_indices = []
    
    for i, token_file in enumerate(token_files):
        print(f"\nProcessando {i+1}/{len(token_files)}: {os.path.basename(token_file)}")
        
        try:
            data = load_tokens(token_file)
            
            # Estatísticas de patches
            if 'patches' in data:
                stats = analyze_token_statistics(token_file)
                print(f"  - Patches: {stats['num_patches']}")
                print(f"  - Dimensão: {stats['patch_dim']}")
            
            # Se tem tokens discretos, analisa
            if 'tokens' in data or 'indices' in data:
                indices = data.get('tokens', data.get('indices'))
                all_indices.append(indices)
                
                # Visualiza primeira sequência em detalhe
                if i == 0:
                    base_name = os.path.splitext(os.path.basename(token_file))[0]
                    
                    visualize_token_sequence(
                        indices,
                        output_path=os.path.join(output_dir, f'{base_name}_sequence.png'),
                        title=f'Sequência de Tokens: {base_name}'
                    )
                    
                    visualize_token_matrix(
                        indices,
                        output_path=os.path.join(output_dir, f'{base_name}_matrix.png'),
                        title=f'Matriz de Tokens: {base_name}'
                    )
                    
                    analyze_temporal_patterns(
                        indices,
                        output_path=os.path.join(output_dir, f'{base_name}_temporal.png')
                    )
        
        except Exception as e:
            print(f"  ✗ Erro: {e}")
    
    # Análise agregada
    if len(all_indices) > 0:
        print("\n" + "=" * 60)
        print("Análise Agregada de Todos os Tokens")
        print("=" * 60)
        
        analyze_token_distribution(
            all_indices,
            args.num_embeddings,
            output_path=os.path.join(output_dir, 'token_distribution_all.png')
        )
        
        # Compara primeiras duas sequências se houver
        if len(all_indices) >= 2:
            compare_token_sequences(
                all_indices[0],
                all_indices[1],
                labels=(
                    os.path.basename(token_files[0]),
                    os.path.basename(token_files[1])
                ),
                output_path=os.path.join(output_dir, 'token_comparison.png')
            )
    
    # Validação de reconstrução (se solicitado)
    if args.validate_reconstruction:
        if args.spectrogram_dir is None:
            print("\n --spectrogram-dir não especificado. Tentando usar data/espectrogramas/")
            args.spectrogram_dir = Path("data/espectrogramas")
        
        if not args.spectrogram_dir.exists():
            print(f"\n✗ Diretório de espectrogramas não encontrado: {args.spectrogram_dir}")
        else:
            print("\n" + "=" * 60)
            print("Validação de Reconstrução VQ")
            print("=" * 60)
            
            all_metrics = []
            
            # Valida alguns arquivos
            validation_files = token_files[:min(5, len(token_files))]
            
            for i, token_file in enumerate(validation_files):
                # Encontra espectrograma original correspondente
                base_name = os.path.basename(token_file).replace('_vq_tokens.pt', '.pt')
                spec_file = os.path.join(args.spectrogram_dir, base_name)
                
                if not os.path.exists(spec_file):
                    print(f"\n Espectrograma original não encontrado: {base_name}")
                    continue
                
                print(f"\n--- Validação {i+1}/{len(validation_files)} ---")
                
                output_file = os.path.join(
                    output_dir,
                    f'{os.path.splitext(base_name)[0]}_reconstruction_validation.png'
                )
                
                metrics = validate_vq_reconstruction(
                    token_file=token_file,
                    original_spectrogram_file=spec_file,
                    output_path=output_file,
                    n_mels=args.n_mels,
                    patch_size=args.patch_size
                )
                
                if metrics:
                    all_metrics.append(metrics)
            
            # Resumo de métricas
            if all_metrics:
                print("\n" + "=" * 60)
                print("Resumo das Métricas de Reconstrução")
                print("=" * 60)
                
                avg_mse = np.mean([m['mse'] for m in all_metrics])
                avg_mae = np.mean([m['mae'] for m in all_metrics])
                avg_corr = np.mean([m['correlation'] for m in all_metrics])
                
                print(f"  - MSE médio: {avg_mse:.6f}")
                print(f"  - MAE médio: {avg_mae:.6f}")
                print(f"  - Correlação média: {avg_corr:.6f}")
                
                # Salva resumo
                summary_path = os.path.join(output_dir, 'reconstruction_summary.txt')
                with open(summary_path, 'w') as f:
                    f.write("Resumo das Métricas de Reconstrução VQ\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Arquivos validados: {len(all_metrics)}\n\n")
                    f.write(f"MSE médio: {avg_mse:.6f}\n")
                    f.write(f"MAE médio: {avg_mae:.6f}\n")
                    f.write(f"Correlação média: {avg_corr:.6f}\n\n")
                    f.write("Métricas individuais:\n")
                    f.write("-" * 60 + "\n")
                    for i, m in enumerate(all_metrics):
                        f.write(f"\nArquivo {i+1}:\n")
                        f.write(f"  MSE: {m['mse']:.6f}\n")
                        f.write(f"  MAE: {m['mae']:.6f}\n")
                        f.write(f"  Correlação: {m['correlation']:.6f}\n")
                
                print(f"\n✓ Resumo salvo em {summary_path}")
    
    print("\n" + "=" * 60)
    print(f"Análise concluída! Visualizações salvas em {output_dir}")
    print("=" * 60)
