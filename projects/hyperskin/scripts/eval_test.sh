

# no transforms data augmentation, no pretraining, no synth data, no balanced sampling, no focal loss id qq6h7y0j
#python src/main.py test -c configs/data/hsi_dermoscopy_croppedv2_notaug.yaml -c configs/model/densenet201_fromscratch.yaml --ckpt_path logs/hypersynth/qq6h7y0j/checkpoints/epoch=79-val/f1=0.8852.ckpt

# no transforms data augmentation, no pretraining, no synth data, YES balanced sampling, no focal loss:  mg63a7bo
#python src/main.py test -c configs/data/hsi_dermoscopy_croppedv2_notaug_batchr.yaml -c configs/model/densenet201_fromscratch.yaml --ckpt_path logs/hypersynth/mg63a7bo/checkpoints/epoch=196-val/f1=0.8571.ckpt # já foi

# no transforms data augmentation, no pretraining, no synth data, no balanced sampling, YES focal loss qu4aduws
#python src/main.py test -c configs/data/hsi_dermoscopy_croppedv2_notaug.yaml -c configs/model/densenet201_fromscratch_focalloss.yaml --ckpt_path logs/hypersynth/qu4aduws/checkpoints/epoch=83-val/f1=0.8889.ckpt

# no transforms data augmentation, no pretraining, no synth data, YES balanced sampling, YES focal loss a80q0rf3
#python src/main.py test -c configs/data/hsi_dermoscopy_croppedv2_notaug_batchr.yaml -c configs/model/densenet201_fromscratch_focalloss.yaml --ckpt_path  logs/hypersynth/a80q0rf3/checkpoints/epoch=72-val/f1=0.8750.ckpt

# no transforms data augmentation, no pretraining, YES synth data, no balanced sampling, no focal loss 9wehtuuh
python src/main.py test -c configs/data/hsi_dermoscopy_croppedv2_notaugsynth.yaml -c configs/model/densenet201_fromscratch.yaml --ckpt_path  logs/hypersynth/9wehtuuh/checkpoints/epoch=76-val/f1=0.9000.ckpt


# transforms data augmentation, no pretraining, YES synth data, no balanced sampling, no focal loss
python src/main.py test -c configs/data/hsi_dermoscopy_croppedv2_augsynth.yaml -c configs/model/densenet201_fromscratch.yaml --ckpt_path logs/hypersynth/4dwtb0nc/checkpoints/epoch=308-val/f1=0.8814.ckpt

# transforms data augmentation, no pretraining, no synth data, no balanced sampling, no focal loss 6kyd9wcb
#python src/main.py test -c configs/data/hsi_dermoscopy_croppedv2_aug.yaml -c configs/model/densenet201_fromscratch.yaml --ckpt_path logs/hypersynth/6kyd9wcb/checkpoints/epoch=87-val/f1=0.8667.ckpt
