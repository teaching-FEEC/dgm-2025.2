"""
Setup script for Super Resolution Validation Framework

This setup.py file provides backward compatibility and additional configuration
for the super-resolution-validation package. The main configuration is in pyproject.toml.
"""

import os
import sys
from pathlib import Path

from setuptools import setup, find_packages

# Ensure we're using Python 3.8+
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required. You are using Python {}.{}.".format(
        sys.version_info.major, sys.version_info.minor))

# Get the long description from the README file
here = Path(__file__).parent.resolve()
long_description = ""

readme_path = here / "README.md"
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
else:
    # Fallback description if README.md doesn't exist
    long_description = """
    Comprehensive validation framework for super resolution models using PSNR, LPIPS, and SSIM metrics.
    
    This package provides tools for:
    - Validating super resolution model performance
    - Generating low-resolution training data
    - Comprehensive analysis and visualization
    - Multiple evaluation metrics (PSNR, LPIPS, SSIM)
    """

# Read version from __init__.py
def get_version():
    """Get version from validation/__init__.py"""
    version_file = here / "validation" / "__init__.py"
    if version_file.exists():
        with open(version_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Define package data
package_data = {
    "validation": [
        "*.json",
        "*.yaml", 
        "*.yml",
        "data/**/*",
        "notebooks/**/*"
    ]
}

# Define entry points for command-line scripts
entry_points = {
    "console_scripts": [
        "sr-validate=validation.main_validation:main",
        "sr-downsample=validation.image_downsampler:main",
        "sr-analyze=validation.notebooks.validation_analysis_simple:main",
    ]
}

# Core dependencies
install_requires = [
    # Core image processing and computer vision
    "opencv-python>=4.5.0",
    "scikit-image>=0.18.0", 
    "Pillow>=8.0.0",
    
    # Numerical computing and data handling
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    
    # Deep learning framework for LPIPS
    "torch>=1.9.0",
    "torchvision>=0.10.0",
    "lpips>=0.1.4",
    
    # Progress bars and utilities
    "tqdm>=4.60.0",
    
    # Visualization (for analysis notebooks)
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0"
]

# Optional dependencies
extras_require = {
    # Development dependencies
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.0.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "isort>=5.0.0",
        "mypy>=0.910"
    ],
    
    # GPU acceleration dependencies
    "gpu": [
        "torch>=1.9.0+cu118",
        "torchvision>=0.10.0+cu118"
    ],
    
    # Jupyter notebook dependencies
    "notebook": [
        "jupyter>=1.0.0",
        "jupyterlab>=3.0.0",
        "ipywidgets>=7.6.0"
    ]
}

# Add 'all' extra that includes everything
extras_require["all"] = list(set(
    extras_require["dev"] + 
    extras_require["notebook"]
    # Note: GPU extras are excluded from 'all' to avoid conflicts
))

# Setup configuration
setup(
    name="super-resolution-validation",
    version=get_version(),
    description="Comprehensive validation framework for super resolution models using PSNR, LPIPS, and SSIM metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Super Resolution Team",
    author_email="team@superres.ai",
    maintainer="Super Resolution Team",
    maintainer_email="team@superres.ai",
    
    # URLs
    url="https://github.com/superres-team/super-resolution-validation",
    project_urls={
        "Documentation": "https://super-resolution-validation.readthedocs.io",
        "Source": "https://github.com/superres-team/super-resolution-validation",
        "Tracker": "https://github.com/superres-team/super-resolution-validation/issues",
    },
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    package_data=package_data,
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points
    entry_points=entry_points,
    
    # Metadata
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "super-resolution",
        "image-processing",
        "computer-vision", 
        "deep-learning",
        "validation",
        "metrics",
        "psnr",
        "lpips",
        "ssim"
    ],
    
    # Additional metadata
    zip_safe=False,
    platforms=["any"],
)

# Post-installation message
def print_installation_message():
    """Print helpful message after installation."""
    print("\n" + "="*60)
    print("Super Resolution Validation Framework installed successfully!")
    print("="*60)
    print("\nAvailable commands:")
    print("   sr-validate    - Run model validation")
    print("   sr-downsample  - Create low-resolution training data")  
    print("   sr-analyze     - Analyze validation results")
    print("\nQuick start:")
    print("   1. Import: from validation import SuperResolutionValidator")
    print("   2. Create: validator = SuperResolutionValidator('MyModel')")
    print("   3. Run: validator.validate_model('./original', './predicted')")
    print("\nDocumentation: https://super-resolution-validation.readthedocs.io")
    print("Issues: https://github.com/superres-team/super-resolution-validation/issues")
    print("="*60)

if __name__ == "__main__":
    # Run setup
    setup()
    
    # Print installation message if this was called directly
    if "install" in sys.argv:
        print_installation_message()
