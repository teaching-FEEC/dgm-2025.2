import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr


def reconstruct_spectrogram_from_patches(
    patches: torch.Tensor,
    n_mels: int = 80,
    patch_size: int = 16
) -> torch.Tensor:
    num_patches = patches.shape[0]
    patches_reshaped = patches.view(num_patches, n_mels, patch_size)
    spectrogram = patches_reshaped.permute(1, 0, 2).contiguous().view(n_mels, -1)
    return spectrogram


def calculate_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    original_np = original.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    
    diff = original_np - reconstructed_np
    
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(diff))
    
    # Correlação de Pearson
    corr, _ = pearsonr(original_np.flatten(), reconstructed_np.flatten())
    
    # PSNR (Peak Signal-to-Noise Ratio)
    max_val = original_np.max()
    psnr = 20 * np.log10(max_val / rmse) if rmse > 0 else float('inf')
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'correlation': corr,
        'psnr': psnr
    }


def visualize_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    metrics: dict,
    output_path: str,
    title: str = 'Validação de Reconstrução VQ'
):
    original_np = original.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    diff = np.abs(original_np - reconstructed_np)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Original
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(original_np, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax0.set_title('Espectrograma Original', fontsize=12, fontweight='bold')
    ax0.set_ylabel('Mel Band')
    plt.colorbar(im0, ax=ax0)
    
    # Reconstruído
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(reconstructed_np, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    ax1.set_title('Espectrograma Reconstruído (VQ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mel Band')
    plt.colorbar(im1, ax=ax1)
    
    # Diferença absoluta
    ax2 = fig.add_subplot(gs[1, :])
    im2 = ax2.imshow(diff, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
    ax2.set_title('Diferença Absoluta (Erro de Reconstrução)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mel Band')
    ax2.set_xlabel('Tempo (frames)')
    plt.colorbar(im2, ax=ax2)
    
    # Histograma de erros
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.hist(diff.flatten(), bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax3.set_xlabel('Erro Absoluto')
    ax3.set_ylabel('Frequência')
    ax3.set_title('Distribuição de Erros', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Métricas
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    metrics_text = f"""
    MÉTRICAS DE QUALIDADE
    
    MSE:  {metrics['mse']:.6f}
    RMSE: {metrics['rmse']:.6f}
    MAE:  {metrics['mae']:.6f}
    
    Erro Máximo:  {metrics['max_error']:.6f}
    
    Correlação:   {metrics['correlation']:.6f}
    PSNR:         {metrics['psnr']:.2f} dB
    
    Shape:        {original_np.shape}
    """
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def validate_single_file(
    token_file: str,
    spectrogram_file: str,
    output_dir: str,
    n_mels: int = 80,
    patch_size: int = 16
) -> dict:
    
    # Carrega dados
    vq_data = torch.load(token_file)
    spec_data = torch.load(spectrogram_file)
    
    # Extrai espectrograma original
    if isinstance(spec_data, dict):
        spec_original = spec_data['spec']
    else:
        spec_original = spec_data
    
    # Remove batch dimension
    if spec_original.dim() == 3:
        spec_original = spec_original.squeeze(0)
    
    # Reconstrói
    patches_reconstructed = vq_data['patches_reconstructed']
    spec_reconstructed = reconstruct_spectrogram_from_patches(
        patches_reconstructed, n_mels, patch_size
    )
    
    # Ajusta tamanho
    min_time = min(spec_original.shape[1], spec_reconstructed.shape[1])
    spec_original = spec_original[:, :min_time]
    spec_reconstructed = spec_reconstructed[:, :min_time]
    
    # Calcula métricas
    metrics = calculate_metrics(spec_original, spec_reconstructed)
    
    # Visualiza
    base_name = os.path.splitext(os.path.basename(token_file))[0]
    output_path = os.path.join(output_dir, f'{base_name}_validation.png')
    
    visualize_comparison(
        spec_original,
        spec_reconstructed,
        metrics,
        output_path,
        title=f'Validação: {base_name}'
    )
    
    print(f"✓ {base_name}")
    print(f"  Correlação: {metrics['correlation']:.4f} | PSNR: {metrics['psnr']:.2f} dB")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validar reconstrução do VQ Tokenizer")
    parser.add_argument("--token-dir", type=Path, required=True,
                        help="Diretório com tokens VQ")
    parser.add_argument("--spec-dir", type=Path, required=True,
                        help="Diretório com espectrogramas originais")
    parser.add_argument("--output-dir", type=Path, default=Path("validation_results"),
                        help="Diretório para salvar resultados")
    parser.add_argument("--max-files", type=int, default=10,
                        help="Número máximo de arquivos a validar")
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--patch-size", type=int, default=16)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print(" Validação de Reconstrução do VQ Tokenizer")
    print("=" * 70)
    
    # Lista arquivos
    token_files = sorted([
        os.path.join(args.token_dir, f)
        for f in os.listdir(args.token_dir)
        if f.endswith('_vq_tokens.pt')
    ])[:args.max_files]
    
    if not token_files:
        print(f"Nenhum arquivo *_vq_tokens.pt encontrado em {args.token_dir}")
        return
    
    print(f"\nValidando {len(token_files)} arquivos...\n")
    
    all_metrics = []
    
    for i, token_file in enumerate(token_files, 1):
        # Encontra espectrograma correspondente
        base_name = os.path.basename(token_file).replace('_vq_tokens.pt', '.pt')
        spec_file = os.path.join(args.spec_dir, base_name)
        
        if not os.path.exists(spec_file):
            print(f"⚠️  [{i}/{len(token_files)}] Espectrograma não encontrado: {base_name}")
            continue
        
        print(f"[{i}/{len(token_files)}] ", end="")
        
        try:
            metrics = validate_single_file(
                token_file,
                spec_file,
                str(args.output_dir),
                args.n_mels,
                args.patch_size
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"✗ Erro: {e}")
    
    # Resumo
    if all_metrics:
        print("\n" + "=" * 70)
        print(" Resumo Estatístico")
        print("=" * 70)
        
        avg_corr = np.mean([m['correlation'] for m in all_metrics])
        std_corr = np.std([m['correlation'] for m in all_metrics])
        avg_psnr = np.mean([m['psnr'] for m in all_metrics])
        std_psnr = np.std([m['psnr'] for m in all_metrics])
        avg_mse = np.mean([m['mse'] for m in all_metrics])
        
        print(f"\n  Arquivos validados:  {len(all_metrics)}")
        print(f"\n  Correlação:  {avg_corr:.4f} ± {std_corr:.4f}")
        print(f"  PSNR:        {avg_psnr:.2f} ± {std_psnr:.2f} dB")
        print(f"  MSE:         {avg_mse:.6f}")
        
        # Avaliação qualitativa
        print(f"\n  Qualidade da Reconstrução:")
        if avg_corr > 0.9:
            quality = "EXCELENTE"
        elif avg_corr > 0.8:
            quality = "MUITO BOA"
        elif avg_corr > 0.7:
            quality = "BOA"
        elif avg_corr > 0.6:
            quality = "REGULAR"
        else:
            quality = "PRECISA MELHORAR "
        
        print(f"    → {quality}")
        
        print(f"\n  Resultados salvos em: {args.output_dir}")
        print("=" * 70)


if __name__ == "__main__":
    main()
