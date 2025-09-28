
import matplotlib.pyplot as plt
import torch
import os
import argparse

parser = argparse.ArgumentParser(description="Converte tensores .pt de espectrograma em imagens PNG.")
parser.add_argument('--input-dir', '-i', type=str, required=True, help='Diretório com arquivos .pt')
parser.add_argument('--output-dir', '-o', type=str, required=True, help='Diretório para salvar PNGs')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".pt"):
        continue

    spec = torch.load(os.path.join(input_dir, fname))

    if spec.ndim == 3:
        spec = spec[0]

    spec = spec.numpy()
    spec = 10 * torch.log10(torch.tensor(spec) + 1e-10).numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect="auto", origin="lower", cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(fname)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname.replace(".pt", ".png")))
    plt.close()

