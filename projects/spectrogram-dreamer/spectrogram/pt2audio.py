import os
import sys
import argparse
from pathlib import Path
import torch
import torchaudio

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruir áudio a partir de um espectrograma Log-Mel usando Griffin-Lim.")
    parser.add_argument("--input-dir", "-i", type=Path, required=True, help="Diretório com arquivos de espectrograma (.pt)")
    parser.add_argument("--filename", "-f", type=str, required=True, help="Nome do arquivo .pt a reconstruir (ex: audio.pt)")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Diretório para salvar o áudio reconstruído")
    parser.add_argument("--gl-iters", type=int, default=32, help="Número de iterações do algoritmo Griffin-Lim")
    return parser.parse_args(argv)

args = parse_args()
input_path = os.path.join(args.input_dir, args.filename)
output_dir = str(args.output_dir)

if not os.path.exists(input_path):
	print(f"Erro: Arquivo de espectrograma não encontrado em {input_path}")
	sys.exit(1)

os.makedirs(output_dir, exist_ok=True)
base_name = os.path.splitext(args.filename)[0]

print(f"Carregando dados de: {args.filename}")
loaded_data = torch.load(input_path)

spec_norm = loaded_data['spec'].cpu()
mean = loaded_data['mean'].cpu()
std = loaded_data['std'].cpu()
sr = loaded_data['sr']
n_fft = loaded_data['n_fft']
hop_length = loaded_data['hop_length']
n_mels = loaded_data['n_mels']
is_log_mel = loaded_data.get('is_log_mel', True)

if not is_log_mel or n_mels is None:
	print("Erro: O espectrograma não é log-mel")
	sys.exit(1)

spec_denorm = (spec_norm * std.item()) + mean.item()
spec_power_mel = torchaudio.functional.DB_to_amplitude(spec_denorm, ref=1.0, power=0.5).pow(2)
n_stft = n_fft // 2 + 1

inverse_mel_scale = torchaudio.transforms.InverseMelScale(
	n_stft=n_stft,
	n_mels=n_mels,
	sample_rate=sr
)

spec_power_linear = inverse_mel_scale(spec_power_mel)

griffin_lim = torchaudio.transforms.GriffinLim(
	n_fft=n_fft,
	hop_length=hop_length,
	n_iter=args.gl_iters,
	power=2.0 # mesmo usado na STFT original
)

waveform_reconstructed = griffin_lim(spec_power_linear)

if waveform_reconstructed.dim() == 1:
	waveform_reconstructed = waveform_reconstructed.unsqueeze(0)

output_audio_path = os.path.join(output_dir, base_name + "_reconstruido.wav")
torchaudio.save(output_audio_path, waveform_reconstructed, sr)

print(f"\nSucesso! Áudio reconstruído salvo em: {output_audio_path}")

# python reconstruir_audio.py -i /caminho/specs -f seu_audio.pt -o /caminho/reconstruidos --gl-iters 64