"""
Image Downsampler for Super Resolution Training Data Generation

This module creates low-resolution versions of high-quality images by downsampling them
4x smaller with reduced quality. Useful for generating training pairs for super resolution models.
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
from typing import Union, Tuple, Optional, List
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from datetime import datetime
import io



class ImageDownsampler:
    """
    A class for downsampling high-quality images to create low-resolution versions
    for super resolution model training and evaluation.
    """
    
    def __init__(self, scale_factor: int = 4, quality_reduction: bool = True):
        """
        Initialize the image downsampler.
        
        Args:
            scale_factor (int): Factor by which to reduce image size (default: 4)
            quality_reduction (bool): Whether to apply additional quality reduction (default: True)
        """
        self.scale_factor = scale_factor
        self.quality_reduction = quality_reduction
        self.processed_files = []
        self.failed_files = []
        
    def downsample_image(self, image: np.ndarray, method: str = 'bicubic') -> np.ndarray:
        """
        Downsample a single image by the specified scale factor.
        
        Args:
            image (np.ndarray): Input high-resolution image
            method (str): Downsampling method ('bicubic', 'bilinear', 'nearest', 'lanczos')
            
        Returns:
            np.ndarray: Downsampled low-resolution image
        """
        height, width = image.shape[:2]
        new_height = height // self.scale_factor
        new_width = width // self.scale_factor
        
        # Choose interpolation method
        interpolation_methods = {
            'bicubic': cv2.INTER_CUBIC,
            'bilinear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'lanczos': cv2.INTER_LANCZOS4,
            'area': cv2.INTER_AREA
        }
        
        interpolation = interpolation_methods.get(method, cv2.INTER_CUBIC)
        
        # Downsample the image
        downsampled = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        return downsampled
    
    def apply_quality_degradation(self, image: np.ndarray, 
                                 blur_kernel: int = 3,
                                 noise_level: float = 0.5,
                                 jpeg_quality: int = 75) -> np.ndarray:
        """
        Apply quality degradation to simulate real-world low-resolution images.
        
        Args:
            image (np.ndarray): Input image
            blur_kernel (int): Size of Gaussian blur kernel (default: 3)
            noise_level (float): Level of Gaussian noise to add (default: 0.5)
            jpeg_quality (int): JPEG compression quality (default: 75)
            
        Returns:
            np.ndarray: Quality-degraded image
        """
        degraded = image.copy()
        
        # Apply Gaussian blur
        if blur_kernel > 1:
            degraded = cv2.GaussianBlur(degraded, (blur_kernel, blur_kernel), 0)
        
        # Add Gaussian noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, degraded.shape).astype(np.float32)
            degraded = degraded.astype(np.float32) + noise
            degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        
        # Simulate JPEG compression
        if jpeg_quality < 100:
            # Convert to PIL for JPEG compression simulation
            pil_image = Image.fromarray(degraded)
            
            # Save to bytes with JPEG compression and reload
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=jpeg_quality)
            buffer.seek(0)
            compressed_image = Image.open(buffer)
            degraded = np.array(compressed_image)
        
        return degraded
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path (Union[str, Path]): Path to the image file
            
        Returns:
            np.ndarray: Loaded image as numpy array (RGB format)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = str(image_path)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Try loading with PIL first (better format support)
        try:
            pil_image = Image.open(image_path).convert('RGB')
            image = np.array(pil_image)
            return image
        except Exception:
            # Fallback to OpenCV
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path], 
                   format: str = 'PNG') -> bool:
        """
        Save an image to file.
        
        Args:
            image (np.ndarray): Image to save
            output_path (Union[str, Path]): Output file path
            format (str): Image format ('PNG', 'JPEG', 'BMP', etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to PIL and save
            pil_image = Image.fromarray(image)
            
            # Set quality for JPEG
            save_kwargs = {}
            if format.upper() == 'JPEG':
                save_kwargs['quality'] = 95
                save_kwargs['optimize'] = True
            
            pil_image.save(output_path, format=format, **save_kwargs)
            return True
            
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")
            return False
    
    def process_single_image(self, input_path: Union[str, Path], 
                           output_path: Union[str, Path],
                           downsample_method: str = 'bicubic',
                           blur_kernel: int = 3,
                           noise_level: float = 0.5,
                           jpeg_quality: int = 75) -> Tuple[bool, str]:
        """
        Process a single image: downsample and apply quality degradation.
        
        Args:
            input_path (Union[str, Path]): Path to input high-resolution image
            output_path (Union[str, Path]): Path to save low-resolution image
            downsample_method (str): Downsampling method
            blur_kernel (int): Gaussian blur kernel size
            noise_level (float): Noise level
            jpeg_quality (int): JPEG compression quality
            
        Returns:
            Tuple[bool, str]: (Success status, filename)
        """
        try:
            # Load image
            image = self.load_image(input_path)
            original_size = image.shape[:2]
            
            # Downsample
            downsampled = self.downsample_image(image, method=downsample_method)
            new_size = downsampled.shape[:2]
            
            # Apply quality degradation if enabled
            if self.quality_reduction:
                downsampled = self.apply_quality_degradation(
                    downsampled, blur_kernel, noise_level, jpeg_quality
                )
            
            # Save result
            success = self.save_image(downsampled, output_path)
            
            if success:
                filename = os.path.basename(input_path)
                result = {
                    'filename': filename,
                    'input_path': str(input_path),
                    'output_path': str(output_path),
                    'original_size': original_size,
                    'downsampled_size': new_size,
                    'scale_factor': self.scale_factor,
                    'method': downsample_method,
                    'quality_degradation': self.quality_reduction
                }
                self.processed_files.append(result)
                return True, filename
            else:
                self.failed_files.append(str(input_path))
                return False, os.path.basename(str(input_path))
                
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            self.failed_files.append(str(input_path))
            return False, os.path.basename(str(input_path))
    
    def process_directory(self, input_dir: Union[str, Path], 
                         output_dir: Union[str, Path],
                         downsample_method: str = 'bicubic',
                         blur_kernel: int = 3,
                         noise_level: float = 0.5,
                         jpeg_quality: int = 75,
                         max_workers: int = 4,
                         preserve_structure: bool = True) -> dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir (Union[str, Path]): Input directory with high-resolution images
            output_dir (Union[str, Path]): Output directory for low-resolution images
            downsample_method (str): Downsampling method
            blur_kernel (int): Gaussian blur kernel size
            noise_level (float): Noise level
            jpeg_quality (int): JPEG compression quality
            max_workers (int): Number of parallel workers
            preserve_structure (bool): Whether to preserve subdirectory structure
            
        Returns:
            dict: Processing statistics and results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        if preserve_structure:
            # Find all images recursively
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_dir.rglob(f"*{ext}"))
                image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
        else:
            # Find images only in root directory
            image_files = [f for f in input_dir.iterdir() 
                          if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to process...")
        print(f"Downsampling by {self.scale_factor}x using {downsample_method} method")
        if self.quality_reduction:
            print(f"Quality degradation: blur={blur_kernel}, noise={noise_level}, jpeg={jpeg_quality}")
        
        # Process images in parallel
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {}
            
            for image_file in image_files:
                if preserve_structure:
                    # Preserve directory structure
                    relative_path = image_file.relative_to(input_dir)
                    output_path = output_dir / relative_path
                else:
                    # Flat structure
                    output_path = output_dir / image_file.name
                
                future = executor.submit(
                    self.process_single_image,
                    image_file, output_path, downsample_method,
                    blur_kernel, noise_level, jpeg_quality
                )
                future_to_file[future] = image_file
            
            # Process results with progress bar
            with tqdm(total=len(image_files), desc="Processing images") as pbar:
                for future in as_completed(future_to_file):
                    success, filename = future.result()
                    if success:
                        successful += 1
                        pbar.set_postfix({"✓": successful, "✗": failed})
                    else:
                        failed += 1
                        pbar.set_postfix({"✓": successful, "✗": failed})
                    pbar.update(1)
        
        # Calculate statistics
        stats = {
            'total_images': len(image_files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(image_files) * 100 if image_files else 0,
            'scale_factor': self.scale_factor,
            'method': downsample_method,
            'quality_degradation': self.quality_reduction,
            'processing_time': datetime.now().isoformat(),
            'input_directory': str(input_dir),
            'output_directory': str(output_dir)
        }
        
        print(f"\nProcessing completed:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        
        return stats
    
    def get_processing_results(self) -> dict:
        """
        Get detailed processing results.
        
        Returns:
            dict: Processing results with file details
        """
        return {
            'processed_files': self.processed_files,
            'failed_files': self.failed_files,
            'total_processed': len(self.processed_files),
            'total_failed': len(self.failed_files)
        }
    
    def save_processing_log(self, output_path: Union[str, Path]):
        """
        Save processing log to JSON file.
        
        Args:
            output_path (Union[str, Path]): Path to save the log file
        """
        results = self.get_processing_results()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Processing log saved to: {output_path}")


def main():
    """Main function with configurable variables."""
    # Configuration variables - modify these as needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    high_quality_images_dir = os.path.join(data_dir, "high_quality_images")
    low_quality_images_dir = os.path.join(data_dir, "low_quality_images")

    input_dir = high_quality_images_dir         # Input directory containing high-resolution images
    output_dir = low_quality_images_dir         # Output directory for low-resolution images
    scale_factor = 4                        # Scale factor for downsampling
    downsample_method = 'bicubic'          # Downsampling method ('bicubic', 'bilinear', 'nearest', 'lanczos', 'area')
    quality_reduction = True               # Whether to apply quality degradation
    blur_kernel = 3                        # Gaussian blur kernel size
    noise_level = 0.5                      # Gaussian noise level
    jpeg_quality = 75                      # JPEG compression quality
    max_workers = 4                        # Number of parallel workers
    preserve_structure = True              # Whether to preserve directory structure
    save_log = True                        # Whether to save processing log
    
    print("Image Downsampler for Super Resolution Training")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Scale factor: {scale_factor}x")
    print(f"Method: {downsample_method}")
    print(f"Quality reduction: {quality_reduction}")
    if quality_reduction:
        print(f"  - Blur kernel: {blur_kernel}")
        print(f"  - Noise level: {noise_level}")
        print(f"  - JPEG quality: {jpeg_quality}")
    print(f"Workers: {max_workers}")
    print(f"Preserve structure: {preserve_structure}")
    print("=" * 50)
    
    # Create downsampler
    downsampler = ImageDownsampler(
        scale_factor=scale_factor,
        quality_reduction=quality_reduction
    )
    
    # Process directory
    try:
        stats = downsampler.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            downsample_method=downsample_method,
            blur_kernel=blur_kernel,
            noise_level=noise_level,
            jpeg_quality=jpeg_quality,
            max_workers=max_workers,
            preserve_structure=preserve_structure
        )
        
        # Save log if requested
        if save_log:
            log_path = Path(output_dir) / "processing_log.json"
            downsampler.save_processing_log(log_path)
        
        print(f"\nDownsampling completed successfully!")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"\nDownsampling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function with configured variables
    main()
