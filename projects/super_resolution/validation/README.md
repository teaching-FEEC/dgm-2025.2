# Super Resolution Validation Framework

A comprehensive validation framework for super resolution models using three key metrics: **PSNR**, **LPIPS**, and **SSIM**.

## Overview

This framework provides robust evaluation tools for super resolution models by implementing three complementary metrics:

- **📐 PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level reconstruction accuracy
- **👁️ LPIPS (Learned Perceptual Image Patch Similarity)**: Evaluates perceptual similarity using deep features
- **📊 SSIM (Structural Similarity Index Measure)**: Assesses structural information preservation

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. For LPIPS functionality, ensure PyTorch is properly installed:
```bash
# CPU version
pip install torch torchvision

# GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

```python
from validation import SuperResolutionValidator

# Initialize validator
validator = SuperResolutionValidator(model_name="MyModel")

# Run comprehensive validation
results = validator.validate_model(
    original_images_dir="./original_images",
    predicted_images_dir="./predicted_images",
    output_dir="./validation_results"
)
```

### Command Line Usage

```bash
python main_validation.py \
    --original ./original_images \
    --predicted ./predicted_images \
    --model-name "MyModel" \
    --output ./results
```

### Individual Metric Usage

```python
from validation import PSNRValidator, LPIPSValidator, SSIMValidator

# PSNR validation
psnr_validator = PSNRValidator()
psnr_stats = psnr_validator.validate_directory("./original", "./predicted")
print(f"Mean PSNR: {psnr_stats['mean_psnr']:.2f} dB")

# LPIPS validation
lpips_validator = LPIPSValidator(net='alex')
lpips_stats = lpips_validator.validate_directory("./original", "./predicted")
print(f"Mean LPIPS: {lpips_stats['mean_lpips']:.4f}")

# SSIM validation
ssim_validator = SSIMValidator()
ssim_stats = ssim_validator.validate_directory("./original", "./predicted")
print(f"Mean SSIM: {ssim_stats['mean_ssim']:.4f}")
```

## Directory Structure

Your image directories should be organized as follows:

```
project/
├── original_images/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── predicted_images/
│   ├── image1.jpg  # Same filename as original
│   ├── image2.png  # Same filename as original
│   └── ...
└── validation/
    ├── main_validation.py
    ├── psnr_validator.py
    ├── lpips_validator.py
    └── ssim_validator.py
```

## Output

The validation framework generates:

1. **Console Output**: Real-time progress and summary statistics
2. **JSON Summary**: Complete validation results with metadata
3. **CSV Files**: Detailed per-image results for each metric
4. **Combined CSV**: All metrics for each image in one file

### Example Output

```
============================================================
SUPER RESOLUTION MODEL VALIDATION
============================================================
Model: MyModel
Original Images: ./original_images
Predicted Images: ./predicted_images
============================================================

🔍 Running PSNR validation...
Processing 100 images for PSNR validation...
✓ PSNR validation completed: Mean = 28.45 dB

👁️ Running LPIPS validation...
Processing 100 images for LPIPS validation...
✓ LPIPS validation completed: Mean = 0.1234

📊 Running SSIM validation...
Processing 100 images for SSIM validation...
✓ SSIM validation completed: Mean = 0.8567

============================================================
VALIDATION SUMMARY - MyModel
============================================================

📐 PSNR (Peak Signal-to-Noise Ratio):
   Mean:   28.45 dB
   Std:    2.34 dB
   Min:    24.12 dB
   Max:    35.67 dB
   Median: 28.23 dB
   Images: 100

👁️ LPIPS (Learned Perceptual Image Patch Similarity):
   Mean:   0.1234
   Std:    0.0456
   Min:    0.0678
   Max:    0.2345
   Median: 0.1198
   Images: 100

📊 SSIM (Structural Similarity Index Measure):
   Mean:   0.8567
   Std:    0.0234
   Min:    0.7890
   Max:    0.9123
   Median: 0.8598
   Images: 100

🎯 Overall Assessment:
   PSNR: Good (25-30 dB)
   LPIPS: Good (0.1-0.2)
   SSIM: Good (0.8-0.9)
```

## Metric Interpretation

### PSNR (Higher is Better)
- **Excellent**: ≥30 dB
- **Good**: 25-30 dB  
- **Fair**: 20-25 dB
- **Poor**: <20 dB

### LPIPS (Lower is Better)
- **Excellent**: ≤0.1
- **Good**: 0.1-0.2
- **Fair**: 0.2-0.3
- **Poor**: >0.3

### SSIM (Higher is Better)
- **Excellent**: ≥0.9
- **Good**: 0.8-0.9
- **Fair**: 0.7-0.8
- **Poor**: <0.7

## Advanced Usage

### Custom Configuration

```python
# Initialize with custom settings
validator = SuperResolutionValidator(
    model_name="MyModel",
    use_gpu=True,           # Use GPU for LPIPS
    lpips_net='vgg'         # Use VGG network for LPIPS
)

# PSNR with custom max pixel value
psnr_validator = PSNRValidator(max_pixel_value=1.0)  # For normalized images

# SSIM with custom window size
ssim_validator = SSIMValidator(win_size=7, multichannel=True)
```

### Single Image Validation

```python
# Validate single image pair
psnr_value, filename = psnr_validator.validate_single_pair(
    "original.jpg", "predicted.jpg"
)
print(f"PSNR for {filename}: {psnr_value:.2f} dB")
```

## Dependencies

- **opencv-python**: Image loading and processing
- **scikit-image**: SSIM calculation
- **Pillow**: Additional image format support
- **numpy**: Numerical computations
- **pandas**: Data handling and CSV export
- **torch & torchvision**: Deep learning backend for LPIPS
- **lpips**: Perceptual similarity calculation

## Troubleshooting

### Common Issues

1. **LPIPS Import Error**: 
   ```bash
   pip install lpips torch torchvision
   ```

2. **GPU Memory Issues**:
   ```python
   validator = SuperResolutionValidator(model_name="MyModel", use_gpu=False)
   ```

3. **Image Size Mismatch**: The framework automatically handles size mismatches by resizing predicted images to match originals.

4. **Missing Images**: The framework skips missing image pairs and reports warnings.

## License

This validation framework is part of the Super Resolution project for the IA376N course at Unicamp.
