
import matplotlib.pyplot as plt
import torch
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Converte arquivos .pt de espectrograma (salvos como dict) em imagens PNG.")
parser.add_argument('--input-dir', '-i', type=str, required=True, help='Diretório com arquivos .pt')
parser.add_argument('--output-dir', '-o', type=str, required=True, help='Diretório para salvar PNGs')
parser.add_argument('--cmap', type=str, default='magma', help='Mapa de cores para o PNG')
parser.add_argument('--vmin', type=float, default=None, help='Valor mínimo para colormap (opcional)')
parser.add_argument('--vmax', type=float, default=None, help='Valor máximo para colormap (opcional)')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

def load_spec(path: str):
    """Load a .pt file and return a 2D numpy array for plotting and a title dict."""
    data = torch.load(path)

    if isinstance(data, dict):
        if 'spec' not in data:
            raise ValueError(f"Arquivo {path} parece ser um dict mas não contém a chave 'spec'.")
        spec = data['spec']

        if isinstance(spec, torch.Tensor):
            spec = spec.detach().cpu().numpy()
        spec = np.array(spec)

        if spec.ndim == 3 and spec.shape[0] == 1:
            spec = spec[0]

        mean = data.get('mean', None)
        std = data.get('std', None)
        if mean is not None and std is not None:
            if isinstance(mean, torch.Tensor):
                mean = float(mean.detach().cpu().item())
            if isinstance(std, torch.Tensor):
                std = float(std.detach().cpu().item())
            spec = spec * (std + 1e-8) + mean

        is_log_mel = data.get('is_log_mel', None)
        title_extra = f"sr={data.get('sr', '?')}, n_mels={data.get('n_mels', '?')}, log_mel={is_log_mel}"

        return spec, title_extra

    if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
        spec = data
        if isinstance(spec, torch.Tensor):
            spec = spec.detach().cpu().numpy()
        spec = np.array(spec)
        if spec.ndim == 3 and spec.shape[0] == 1:
            spec = spec[0]
        return spec, ""

    raise ValueError(f"Formato de arquivo .pt não suportado: {path}")


for fname in os.listdir(input_dir):
    if not fname.endswith('.pt'):
        continue

    inpath = os.path.join(input_dir, fname)
    try:
        spec, title_extra = load_spec(inpath)
    except Exception as e:
        print(f"Falha ao carregar {fname}: {e}")
        continue

    if spec.ndim != 2:
        print(f"Espectrograma em {fname} não é 2D (shape={spec.shape}), pulando")
        continue

    plt.figure(figsize=(10, 4))
    img = plt.imshow(spec, aspect='auto', origin='lower', cmap=args.cmap, vmin=args.vmin, vmax=args.vmax)
    cbar = plt.colorbar(img, format='%+2.0f dB')
    plt.title(f"{fname}  {title_extra}")
    plt.tight_layout()
    outpath = os.path.join(output_dir, fname.replace('.pt', '.png'))
    plt.savefig(outpath)
    plt.close()
    print(f"Salvo: {outpath}")

