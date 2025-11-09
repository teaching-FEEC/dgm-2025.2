from src.modules.mnist_module import SimpleNetModule
from src.modules.gan import GANModule
from src.modules.hsi_classifier_module import HSIClassifierModule
from src.modules.hsi_segmentation_module import HSISegmentationModule
from src.modules.generative.gan.wgan import WGANModule
from src.modules.vae_module import VAE
from src.modules.generative.gan.fastgan.fastgan import FastGANModule
from src.modules.generative.gan.cycle_gan import CycleGANModule
__all__ = [
    "SimpleNetModule",
    "GANModule",
    "HSIClassifierModule",
    "HSISegmentationModule",
    "WGANModule",
    'VAE'
    "FastGANModule"
]
