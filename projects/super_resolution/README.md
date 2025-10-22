# Super Resolution Validation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive validation framework for super resolution models using **PSNR**, **LPIPS**, and **SSIM** metrics. This package provides tools for model evaluation, training data generation, and performance analysis with beautiful visualizations.

## Features

### **Comprehensive Metrics**
- **PSNR** (Peak Signal-to-Noise Ratio) - Pixel-level reconstruction accuracy
- **LPIPS** (Learned Perceptual Image Patch Similarity) - Perceptual quality assessment
- **SSIM** (Structural Similarity Index Measure) - Structural information preservation

### **Validation Tools**
- **Multi-metric validation** with statistical analysis
- **Batch processing** for entire image directories
- **Quality categorization** (Excellent/Good/Fair/Poor)
- **Performance ranking** and comparison
- **Correlation analysis** between metrics

### **Data Generation**
- **Image downsampling** for training data creation
- **Quality degradation** simulation (blur, noise, compression)
- **Multiple downsampling methods** (bicubic, bilinear, lanczos, etc.)
- **Parallel processing** for efficiency

### **Analysis & Visualization**
- **Interactive analysis** with Jupyter notebooks
- **Multiple visualization types** (distributions, correlations, dashboards)
- **Performance insights** and recommendations
- **Export capabilities** (CSV, JSON, plots)

## Installation

### Using pip (Recommended)

```bash
# Install from PyPI (when published)
pip install super-resolution-validation

# Or install from source
pip install git+https://github.com/superres-team/super-resolution-validation.git
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/superres-team/super-resolution-validation.git
cd super-resolution-validation

# Install in development mode
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### Optional Dependencies

```bash
# GPU acceleration (CUDA)
pip install "super-resolution-validation[gpu]"

# Jupyter notebook support
pip install "super-resolution-validation[notebook]"

# Development tools
pip install "super-resolution-validation[dev]"

# Everything
pip install "super-resolution-validation[all]"
```

## Quick Start

### Basic Validation

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

# Print summary
print(f"Mean PSNR: {results['metrics']['psnr']['mean_psnr']:.2f} dB")
print(f"Mean LPIPS: {results['metrics']['lpips']['mean_lpips']:.4f}")
print(f"Mean SSIM: {results['metrics']['ssim']['mean_ssim']:.4f}")
```

### Individual Metrics

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

### Data Generation

```python
from validation import ImageDownsampler

# Create low-resolution training data
downsampler = ImageDownsampler(scale_factor=4, quality_reduction=True)

# Process entire directory
stats = downsampler.process_directory(
    input_dir="./high_res_images",
    output_dir="./low_res_images",
    downsample_method='bicubic',
    blur_kernel=3,
    noise_level=0.5,
    jpeg_quality=75
)

print(f"Processed {stats['successful']} images successfully")
```

## Command Line Interface

The package provides convenient command-line tools:

### Model Validation
```bash
# Run comprehensive validation
sr-validate --original ./original --predicted ./predicted --model-name MyModel

# With custom settings
sr-validate --original ./original --predicted ./predicted --model-name MyModel --no-gpu --lpips-net vgg
```

### Data Generation
```bash
# Create low-resolution training data
sr-downsample --input ./high_res --output ./low_res --scale 4 --method bicubic

# With quality degradation
sr-downsample --input ./2k_images --output ./512px_images --blur 3 --noise 0.5 --jpeg-quality 75
```

### Analysis
```bash
# Analyze validation results
sr-analyze --results ./validation_results
```

## Example Output

### Validation Summary
```
VALIDATION SUMMARY
==================================================

PSNR (Peak Signal-to-Noise Ratio):
   Mean:   28.45 dB
   Std:    2.34 dB
   Min:    24.12 dB
   Max:    35.67 dB
   Median: 28.23 dB
   Images: 100

LPIPS (Learned Perceptual Image Patch Similarity):
   Mean:   0.1234
   Std:    0.0456
   Min:    0.0678
   Max:    0.2345
   Median: 0.1198
   Images: 100

SSIM (Structural Similarity Index Measure):
   Mean:   0.8567
   Std:    0.0234
   Min:    0.7890
   Max:    0.9123
   Median: 0.8598
   Images: 100

Overall Assessment:
   PSNR: Good (25-30 dB)
   LPIPS: Good (0.1-0.2)
   SSIM: Good (0.8-0.9)
```

### Performance Insights
```
PERFORMANCE INSIGHTS & RECOMMENDATIONS
============================================================
PSNR: Good performance (Mean: 28.45 dB, Std: 2.34)
LPIPS: Good perceptual quality (Mean: 0.1234, Std: 0.0456)
SSIM: Good structural similarity (Mean: 0.8567, Std: 0.0234)

Recommendations:
   - Model shows consistent performance across all metrics
   - Consider perceptual loss functions to improve LPIPS scores
   - Strong correlation between PSNR and SSIM indicates good structural preservation
```

## Visualizations

The framework generates comprehensive visualizations:

1. **Metric Distributions** - Histograms and box plots
2. **Correlation Analysis** - Heatmaps and scatter plots  
3. **Performance Ranking** - Best/worst image comparisons
4. **Quality Dashboard** - Pie charts and trend analysis
5. **Statistical Summaries** - Detailed metric tables

## Quality Thresholds

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

## Project Structure

```
super-resolution-validation/
├── validation/                 # Main package
│   ├── __init__.py            # Package initialization
│   ├── main_validation.py     # Main validation orchestrator
│   ├── psnr_validator.py      # PSNR metric implementation
│   ├── lpips_validator.py     # LPIPS metric implementation
│   ├── ssim_validator.py      # SSIM metric implementation
│   ├── image_downsampler.py   # Data generation tools
│   └── notebooks/             # Analysis notebooks
├── tests/                     # Test suite
├── docs/                      # Documentation
├── examples/                  # Usage examples
├── pyproject.toml            # Modern Python packaging
├── setup.py                  # Backward compatibility
├── README.md                 # This file
└── LICENSE                   # MIT license
```

## Configuration

### Environment Variables
```bash
# GPU settings
export CUDA_VISIBLE_DEVICES=0
export SR_USE_GPU=true

# Default paths
export SR_DATA_DIR=./data
export SR_RESULTS_DIR=./results
```

### Configuration Files
```python
# Custom configuration
config = {
    'psnr': {'max_pixel_value': 255.0},
    'lpips': {'net': 'alex', 'use_gpu': True},
    'ssim': {'win_size': 11, 'multichannel': True},
    'output': {'save_plots': True, 'plot_format': 'png'}
}

validator = SuperResolutionValidator(model_name="MyModel", config=config)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=validation --cov-report=html

# Run specific test categories
pytest tests/test_validators.py
pytest tests/test_downsampler.py
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/superres-team/super-resolution-validation.git
cd super-resolution-validation
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black validation/
isort validation/

# Run type checking
mypy validation/
```

## Documentation

- **API Documentation**: https://super-resolution-validation.readthedocs.io
- **Examples**: [examples/](examples/)
- **Notebooks**: [validation/notebooks/](validation/notebooks/)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## Issues & Support

- **Bug Reports**: [GitHub Issues](https://github.com/superres-team/super-resolution-validation/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/superres-team/super-resolution-validation/discussions)
- **Questions**: [Stack Overflow](https://stackoverflow.com/questions/tagged/super-resolution-validation)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LPIPS**: [Learned Perceptual Image Patch Similarity](https://github.com/richzhang/PerceptualSimilarity)
- **PyTorch**: Deep learning framework
- **scikit-image**: Image processing library
- **OpenCV**: Computer vision library

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{super_resolution_validation,
  title={Super Resolution Validation Framework},
  author={Super Resolution Team},
  year={2025},
  url={https://github.com/superres-team/super-resolution-validation}
}
```

---

**Made with care for the computer vision community**
