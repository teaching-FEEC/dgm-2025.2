"""
Super Resolution Validation Package

This package provides comprehensive validation tools for super resolution models
using three key metrics: PSNR, LPIPS, and SSIM.

Classes:
    PSNRValidator: Peak Signal-to-Noise Ratio validation
    LPIPSValidator: Learned Perceptual Image Patch Similarity validation  
    SSIMValidator: Structural Similarity Index Measure validation
    SuperResolutionValidator: Main validation orchestrator

Example usage:
    from validation import SuperResolutionValidator
    
    validator = SuperResolutionValidator(model_name="MyModel")
    results = validator.validate_model("./original", "./predicted")
"""

from .psnr_validator import PSNRValidator
from .lpips_validator import LPIPSValidator
from .ssim_validator import SSIMValidator
from .main_validation import SuperResolutionValidator
from .image_downsampler import ImageDownsampler

__version__ = "1.0.0"
__author__ = "Super Resolution Team"

__all__ = [
    'PSNRValidator',
    'LPIPSValidator', 
    'SSIMValidator',
    'SuperResolutionValidator',
    'ImageDownsampler'
]
