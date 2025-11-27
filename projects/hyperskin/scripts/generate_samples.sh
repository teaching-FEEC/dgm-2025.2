# FastGAN Melanoma
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/j7od0bbk/config.yaml --ckpt_path="logs/hypersynth/j7od0bbk/checkpoints/step=0-val_MIFID=114.7889.ckpt" --trainer.logger=false

# FastGAN Nevi
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/3113cnnm/config.yaml --ckpt_path="logs/hypersynth/3113cnnm/checkpoints/step=0-val_FID=99.1142.ckpt" --trainer.logger=false

# SPADE FastGAN Melanoma
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/16ztzy8j/config.yaml --ckpt_path="logs/hypersynth/16ztzy8j/checkpoints/step=0-val_FID=86.3025.ckpt" --trainer.logger=false

# CycleGAN Melanoma
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/i3nz2pqi/config.yaml --ckpt_path="logs/hyp
ersynth/i3nz2pqi/checkpoints/step=0-val_FID=104.0855.ckpt" --trainer.logger=false

# AC CycleGAN Melanoma + Nevi
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/ju3w3sw1/config.yaml --ckpt_path="logs/hypersynth/ju3w3sw1/checkpoints/step=0-val_FID=117.5433.ckpt" --trainer.logger=false

# FastGAN Trained with Melanoma + Nevi
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/98xb02br/config.yaml --ckpt_path="logs/hypersynth/98xb02br/checkpoints/step=0-val_FID=80.2161.ckpt" --trainer.logger=false

# FastGAN Pretrained Melanoma
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/0ta2r1jy/config.yaml --ckpt_path="logs/hypersynth/0ta2r1jy/checkpoints/step=0-val_FID=90.5197.ckpt" --trainer.logger=false

# FastGAN Pretrained Nevi
WANDB_MODE=disabled python src/main.py predict -c logs/hypersynth/1r91ijoj/config.yaml --ckpt_path="logs/hypersynth/1r91ijoj/checkpoints/step=0-val_FID=97.4476.ckpt" --trainer.logger=false