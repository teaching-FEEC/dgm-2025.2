import os
import time
import argparse
from pathlib import Path

import torch
import torchaudio
import librosa
import numpy as np

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerar espectrogramas a partir de arquivos de áudio usando librosa.stft")
    parser.add_argument("--input-dir", "-i", type=Path, required=True, help="Diretório com arquivos de áudio")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Diretório para salvar os espectrogramas")
    parser.add_argument("--ext", type=str, default='mp3', help="Extensão dos arquivos de áudio a processar (ex: mp3, wav, flac)")
    parser.add_argument("--sr", type=int, default=None, help="Taxa de amostragem para reamostragem (padrão: manter original)")
    parser.add_argument("--n-fft", type=int, default=2048, help="Tamanho da janela FFT")
    parser.add_argument("--hop-length", type=int, default=512, help="Passo entre janelas sucessivas")
    parser.add_argument("--n-mels", type=int, default=80, help="Número de faixas Mel")
    parser.add_argument("--cmap", type=str, default="magma", help="Mapa de cores do espectrograma")
    parser.add_argument("--no-recursive", action="store_true", help="Não buscar em subdiretórios")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescrever espectrogramas existentes")
    return parser.parse_args(argv)

args = parse_args()

input_dir = str(args.input_dir)
output_dir = str(args.output_dir)
ext = args.ext.lower().lstrip('.')
n_fft = args.n_fft
hop_length = args.hop_length
sr_target = args.sr
n_mels = args.n_mels

os.makedirs(output_dir, exist_ok=True)
start_time = time.time()

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(f".{ext}"):
        continue

    path = os.path.join(input_dir, fname)
    base = os.path.splitext(fname)[0]

    output_path = os.path.join(output_dir, base + ".pt")
    if not args.overwrite and os.path.exists(output_path):
        print(f"Pulado: {fname} (já existe)")
        continue

    # Usa librosa para carregar MP3 (mais robusto)
    try:
        # Carrega com librosa
        waveform_np, sr = librosa.load(path, sr=sr_target, mono=False)
        
        # Converte para tensor torch
        if waveform_np.ndim == 1:
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # Mono: (samples,) -> (1, samples)
        else:
            waveform = torch.from_numpy(waveform_np)  # Stereo: (2, samples)
            
    except Exception as e:
        print(f"Erro ao carregar {fname}: {e}")
        continue

    # Se necessário, converte para mono
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0 # Potência (magnitude^2) para o cálculo padrão de dB
    )
    spec = mel_spectrogram_transform(waveform)

    amplitude_to_db_transform = torchaudio.transforms.AmplitudeToDB()
    spec = amplitude_to_db_transform(spec)
    
    mean = spec.mean()
    std = spec.std()
    spec = (spec - mean) / (std + 1e-8)

    torch.save({
        'spec': spec,
        'mean': mean,
        'std': std,
        'sr': sr,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_mels': n_mels,
        'is_log_mel': True
    }, os.path.join(output_dir, base + ".pt"))

    print(f"Processado: {fname}")

end_time = time.time()
print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")