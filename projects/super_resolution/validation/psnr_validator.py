"""
PSNR (Peak Signal-to-Noise Ratio) Validator for Super Resolution Models

This module implements PSNR calculation for evaluating super resolution model performance.
PSNR measures the ratio between the maximum possible power of a signal and the power 
of corrupting noise that affects the fidelity of its representation.
"""

import numpy as np
import cv2
from typing import Union, Tuple
import os
from pathlib import Path
import pandas as pd


class PSNRValidator:
    """
    A class for calculating PSNR (Peak Signal-to-Noise Ratio) between original and predicted images.
    
    PSNR is a widely used metric in image processing that measures the quality of reconstruction
    by comparing the peak signal power to the mean squared error between images.
    """
    
    def __init__(self, max_pixel_value: float = 255.0):
        """
        Initialize the PSNR validator.
        
        Args:
            max_pixel_value (float): Maximum possible pixel value (default: 255.0 for 8-bit images)
        """
        self.max_pixel_value = max_pixel_value
        self.results = []
    
    def calculate_psnr(self, original: np.ndarray, predicted: np.ndarray) -> float:
        """
        Calculate PSNR between two images.
        
        Args:
            original (np.ndarray): Original high-resolution image
            predicted (np.ndarray): Predicted/reconstructed image
            
        Returns:
            float: PSNR value in dB
            
        Raises:
            ValueError: If images have different shapes or invalid dimensions
        """
        if original.shape != predicted.shape:
            raise ValueError(f"Image shapes don't match: {original.shape} vs {predicted.shape}")
        
        # Convert to float to avoid overflow
        original = original.astype(np.float64)
        predicted = predicted.astype(np.float64)
        
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((original - predicted) ** 2)
        
        # Handle perfect reconstruction case
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        psnr = 20 * np.log10(self.max_pixel_value / np.sqrt(mse))
        return psnr
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path (Union[str, Path]): Path to the image file
            
        Returns:
            np.ndarray: Loaded image as numpy array
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = str(image_path)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image in color (BGR format)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def validate_single_pair(self, original_path: Union[str, Path], 
                           predicted_path: Union[str, Path]) -> Tuple[float, str]:
        """
        Validate a single pair of original and predicted images.
        
        Args:
            original_path (Union[str, Path]): Path to original image
            predicted_path (Union[str, Path]): Path to predicted image
            
        Returns:
            Tuple[float, str]: PSNR value and image filename
        """
        try:
            original = self.load_image(original_path)
            predicted = self.load_image(predicted_path)
            
            psnr_value = self.calculate_psnr(original, predicted)
            filename = os.path.basename(original_path)
            
            result = {
                'filename': filename,
                'psnr': psnr_value,
                'original_path': str(original_path),
                'predicted_path': str(predicted_path)
            }
            self.results.append(result)
            
            return psnr_value, filename
            
        except Exception as e:
            print(f"Error processing {original_path}: {str(e)}")
            return 0.0, os.path.basename(str(original_path))
    
    def validate_directory(self, original_dir: Union[str, Path], 
                          predicted_dir: Union[str, Path]) -> dict:
        """
        Validate all image pairs in two directories.
        
        Args:
            original_dir (Union[str, Path]): Directory containing original images
            predicted_dir (Union[str, Path]): Directory containing predicted images
            
        Returns:
            dict: Dictionary containing validation results and statistics
        """
        original_dir = Path(original_dir)
        predicted_dir = Path(predicted_dir)
        
        if not original_dir.exists():
            raise FileNotFoundError(f"Original directory not found: {original_dir}")
        
        if not predicted_dir.exists():
            raise FileNotFoundError(f"Predicted directory not found: {predicted_dir}")
        
        # Get all image files from original directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        original_files = [f for f in original_dir.iterdir() 
                         if f.suffix.lower() in image_extensions]
        
        psnr_values = []
        processed_files = []
        
        print(f"Processing {len(original_files)} images for PSNR validation...")
        
        for original_file in original_files:
            # Look for corresponding predicted file
            predicted_file = predicted_dir / original_file.name
            
            if not predicted_file.exists():
                print(f"Warning: No predicted image found for {original_file.name}")
                continue
            
            psnr_value, filename = self.validate_single_pair(original_file, predicted_file)
            
            if psnr_value > 0:  # Valid PSNR calculated
                psnr_values.append(psnr_value)
                processed_files.append(filename)
                print(f"  {filename}: PSNR = {psnr_value:.2f} dB")
        
        # Calculate statistics
        if psnr_values:
            stats = {
                'mean_psnr': np.mean(psnr_values),
                'std_psnr': np.std(psnr_values),
                'min_psnr': np.min(psnr_values),
                'max_psnr': np.max(psnr_values),
                'median_psnr': np.median(psnr_values),
                'num_images': len(psnr_values),
                'processed_files': processed_files
            }
        else:
            stats = {
                'mean_psnr': 0.0,
                'std_psnr': 0.0,
                'min_psnr': 0.0,
                'max_psnr': 0.0,
                'median_psnr': 0.0,
                'num_images': 0,
                'processed_files': []
            }
        
        return stats
    
    def get_results(self) -> list:
        """
        Get all validation results.
        
        Returns:
            list: List of dictionaries containing validation results
        """
        return self.results
    
    def clear_results(self):
        """Clear all stored results."""
        self.results = []
    
    def save_results(self, output_path: Union[str, Path]):
        """
        Save results to a CSV file.
        
        Args:
            output_path (Union[str, Path]): Path to save the results CSV file
        """      
        if not self.results:
            print("No results to save.")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"PSNR results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    validator = PSNRValidator()

    
    # Example: validate single pair
    original_image_path = r"C:\Users\sucho\Downloads\image_test.png"
    predicted_image_path = r"C:\Users\sucho\Downloads\image_test.png"
    psnr_value, filename = validator.validate_single_pair(original_image_path, predicted_image_path)
    print(f"PSNR for {filename}: {psnr_value:.2f} dB")

    # # Example: validate directory
    # original_images_dir = "original_images/"
    # predicted_images_dir = "predicted_images/"
    # stats = validator.validate_directory(original_images_dir, predicted_images_dir)
    # print(f"Mean PSNR: {stats['mean_psnr']:.2f} dB")
    
    print("PSNR Validator initialized. Use validate_single_pair() or validate_directory() methods.")
