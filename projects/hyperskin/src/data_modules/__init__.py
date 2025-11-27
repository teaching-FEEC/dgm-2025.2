from src.data_modules.mnist import MNISTDataModule
from src.data_modules.hsi_dermoscopy import HSIDermoscopyDataModule
from src.data_modules.milk10k import MILK10kDataModule
from src.data_modules.joint_rgb_hsi_dermoscopy import JointRGBHSIDataModule
__all__ = [
    "MNISTDataModule",
    "HSIDermoscopyDataModule",
    "MILK10kDataModule",
    "JointRGBHSIDataModule",
]
