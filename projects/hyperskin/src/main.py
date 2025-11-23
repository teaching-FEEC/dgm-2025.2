import os

import pyrootutils

pyrootutils.setup_root(__file__, project_root_env_var=True, dotenv=True, pythonpath=True, cwd=False)
from src.utils import CustomLightningCLI  # noqa: E402

if __name__ == "__main__":
    if os.environ.get("DEBUG", False):
        import debugpy

        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    cli = CustomLightningCLI(parser_kwargs={"parser_mode": "omegaconf"}, save_config_kwargs={"overwrite": True})

#python main.py fit -c /mnt/datahdd/kris_volume/dgm-2025.2/projects/hyperskin/configs/data/hsi_dermoscopy_synth_cropped.yaml -c /mnt/datahdd/kris_volume/dgm-2025.2/projects/hyperskin/configs/model/shs_gan.yaml

#python /mnt/datassd/kris_volume/dgm-2025.2/projects/hyperskin/src/main.py fit -c /mnt/datassd/kris_volume/dgm-2025.2/projects/hyperskin/configs/data/joint_rgb_hsi_dermoscopy.yaml -c /mnt/datassd/kris_volume/dgm-2025.2/projects/hyperskin/configs/model/hsi_cycle_gan.yaml
#python /mnt/datassd/kris_volume/dgm-2025.2/projects/hyperskin/src/main.py fit -c /mnt/datassd/kris_volume/dgm-2025.2/projects/hyperskin/configs/data/hsi_dermoscopy_croppedv2.yaml -c /mnt/datassd/kris_volume/dgm-2025.2/projects/hyperskin/configs/model/shs_wgan.yaml