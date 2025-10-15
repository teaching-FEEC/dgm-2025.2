from src.modules.mnist_module import SimpleNetModule
from src.modules.gan import GANModule
from src.modules.hsi_classifier_module import HSIClassifierModule
from src.modules.hsi_segmentation_module import HSISegmentationModule
from src.modules.generative.gan.wgan import WGANModule
__all__ = [
    "SimpleNetModule",
    "GANModule",
    "HSIClassifierModule",
    "HSISegmentationModule",
    "WGANModule",
]
