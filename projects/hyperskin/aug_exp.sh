

# no transforms data augmentation, no pretraining, no synth data, no balanced sampling, no focal loss
python src/main.py fit -c configs/data/hsi_dermoscopy_croppedv2_notaug.yaml -c configs/model/densenet201_fromscratch.yaml

# no transforms data augmentation, no pretraining, no synth data, YES balanced sampling, no focal loss
python src/main.py fit -c configs/data/hsi_dermoscopy_croppedv2_notaug_batchr.yaml -c configs/model/densenet201_fromscratch.yaml

# no transforms data augmentation, no pretraining, no synth data, no balanced sampling, YES focal loss
python src/main.py fit -c configs/data/hsi_dermoscopy_croppedv2_notaug.yaml -c configs/model/densenet201_fromscratch_focalloss.yaml 

# no transforms data augmentation, no pretraining, no synth data, YES balanced sampling, YES focal loss
python src/main.py fit -c configs/data/hsi_dermoscopy_croppedv2_notaug_batchr.yaml -c configs/model/densenet201_fromscratch_focalloss.yaml

# no transforms data augmentation, no pretraining, YES synth data, no balanced sampling, no focal loss
python src/main.py fit -c configs/data/hsi_dermoscopy_croppedv2_augsynth.yaml -c configs/model/densenet201_fromscratch.yaml


# transforms data augmentation, no pretraining, YES synth data, no balanced sampling, no focal loss
python src/main.py fit -c configs/data/hsi_dermoscopy_croppedv2_augsynth.yaml -c configs/model/densenet201_fromscratch.yaml

# transforms data augmentation, no pretraining, no synth data, no balanced sampling, no focal loss
python src/main.py fit -c configs/data/hsi_dermoscopy_croppedv2_aug.yaml -c configs/model/densenet201_fromscratch.yaml 
