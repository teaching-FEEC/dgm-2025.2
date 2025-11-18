# Spectrogram Tools (resumo)

Códigos para gerar, visualizar e reconstruir áudio a partir de espectrogramas.

Principais scripts:
- [projects/spectrogram-dreamer/spectrogram/spectogram.py](spectogram.py) — gera espectrogramas a partir de arquivos de áudio.
- [projects/spectrogram-dreamer/spectrogram/pt2png.py](pt2png.py) — converte arquivos `.pt` de espectrograma em imagens PNG.
- [projects/spectrogram-dreamer/spectrogram/pt2audio.py](pt2audio.py) — reconstrói áudio a partir de espectrogramas log-mel usando Griffin-Lim.


### Exemplos de uso

```bash
python3 spectrogram/spectogram.py --ext mp3 --input-dir data/cv-corpus-21.0-delta-2025-03-14/pt/clips --output-dir data/raw_spectrograms/

python3 spectrogram/pt2audio.py -i data/spectrograms -f common_voice_pt_41964522.pt -o spectrogram/reconstruido/ --gl-iters 64

python3 spectrogram/pt2png.py --input-dir data/spectrograms --output-dir spectrogram/imgs/
```
