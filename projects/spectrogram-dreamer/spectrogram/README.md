# Spectrogram Tools (resumo)

Códigos para gerar, visualizar e reconstruir áudio a partir de espectrogramas.

Principais scripts:
- [projects/spectrogram-dreamer/spectrogram/spectogram.py](projects/spectrogram-dreamer/spectrogram/spectogram.py) — gera espectrogramas a partir de arquivos de áudio.
- [projects/spectrogram-dreamer/spectrogram/pt2png.py](projects/spectrogram-dreamer/spectrogram/pt2png.py) — converte arquivos `.pt` de espectrograma em imagens PNG.
- [projects/spectrogram-dreamer/spectrogram/pt2audio.py](projects/spectrogram-dreamer/spectrogram/pt2audio.py) — reconstrói áudio a partir de espectrogramas log-mel usando Griffin-Lim.

### Instalação
```bash
# criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# atualizar pip e instalar dependências
python -m pip install --upgrade pip
pip install -r projects/spectrogram-dreamer/spectrogram/requirements.txt
```

### Exemplos de uso

```bash
python3 spectogram.py --ext mp3 --input-dir data/cv-corpus-21.0-delta-2025-03-14/pt/clips --output-dir out/

python3 pt2audio.py -i out/ -f common_voice_pt_41964522.pt -o reconstruido/ --gl-iters 64

python3 pt2png.py --input-dir out/ --output-dir img/
```
