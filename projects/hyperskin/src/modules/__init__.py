from src.modules.mnist_module import SimpleNetModule
from src.modules.gan import GANModule
from src.modules.hsi_classifier_module import HSIClassifierModule
from src.modules.hsi_segmentation_module import HSISegmentationModule
from src.modules.generative.gan.wgan import WGANModule
from src.modules.vae_module import VAE
__all__ = [
    "SimpleNetModule",
    "GANModule",
    "HSIClassifierModule",
    "HSISegmentationModule",
    "WGANModule",
    'VAE'
]
