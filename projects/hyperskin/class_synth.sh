#train on synthetic data variations
python src/main.py fit -c configs/data/hsi_dermoscopy_train-synth.yaml -c configs/model/hsi_classifier_densenet201_best.yaml

python src/main.py fit -c configs/data/hsi_dermoscopy_val-synth.yaml -c configs/model/hsi_classifier_densenet201_best.yaml

python src/main.py fit -c configs/data/hsi_dermoscopy_mixed-train03.yaml -c configs/model/hsi_classifier_densenet201_best.yaml
python src/main.py fit -c configs/data/hsi_dermoscopy_mixed-train05.yaml -c configs/model/hsi_classifier_densenet201_best.yaml
python src/main.py fit -c configs/data/hsi_dermoscopy_mixed-train07.yaml -c configs/model/hsi_classifier_densenet201_best.yaml