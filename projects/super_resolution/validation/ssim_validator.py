"""
SSIM (Structural Similarity Index Measure) Validator for Super Resolution Models

This module implements SSIM calculation for evaluating super resolution model performance.
SSIM measures the structural similarity between images by comparing luminance, contrast,
and structure, providing a metric that correlates well with human visual perception.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from typing import Union, Tuple, Optional
import os
from pathlib import Path
import pandas as pd

class SSIMValidator:
    """
    A class for calculating SSIM (Structural Similarity Index Measure) between 
    original and predicted images.
    
    SSIM measures structural similarity by comparing luminance, contrast, and structure
    between images, providing a perceptually meaningful metric for image quality assessment.
    """
    
    def __init__(self, win_size: Optional[int] = None, multichannel: bool = True, 
                 data_range: Optional[float] = None):
        """
        Initialize the SSIM validator.
        
        Args:
            win_size (Optional[int]): Size of the sliding window (default: None, uses optimal size)
            multichannel (bool): Whether to compute SSIM for multichannel images
            data_range (Optional[float]): Data range of the input image (default: None, auto-detect)
        """
        self.win_size = win_size
        self.multichannel = multichannel
        self.data_range = data_range
        self.results = []
    
    def calculate_ssim(self, original: np.ndarray, predicted: np.ndarray, 
                      full: bool = False) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Calculate SSIM between two images.
        
        Args:
            original (np.ndarray): Original high-resolution image
            predicted (np.ndarray): Predicted/reconstructed image
            full (bool): Whether to return the full SSIM map
            
        Returns:
            Union[float, Tuple[float, np.ndarray]]: SSIM value or (SSIM value, SSIM map)
            
        Raises:
            ValueError: If images have different shapes or invalid dimensions
        """
        if original.shape != predicted.shape:
            raise ValueError(f"Image shapes don't match: {original.shape} vs {predicted.shape}")
        
        # Convert to appropriate data type
        original = original.astype(np.float64)
        predicted = predicted.astype(np.float64)
        
        # Determine data range if not specified
        data_range = self.data_range
        if data_range is None:
            data_range = max(original.max(), predicted.max()) - min(original.min(), predicted.min())
        
        # Handle different image dimensions
        if len(original.shape) == 2:  # Grayscale
            multichannel = False
        elif len(original.shape) == 3 and original.shape[2] == 1:  # Single channel
            original = original.squeeze()
            predicted = predicted.squeeze()
            multichannel = False
        else:  # Multi-channel (RGB)
            multichannel = self.multichannel
        
        try:
            # Calculate SSIM using scikit-image
            ssim_value = ssim(
                original, 
                predicted,
                win_size=self.win_size,
                multichannel=multichannel,
                data_range=data_range,
                full=full,
                channel_axis=-1 if multichannel else None
            )
            
            return ssim_value
            
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            # Fallback to manual SSIM calculation for grayscale
            if not multichannel:
                return self._manual_ssim(original, predicted)
            else:
                # Convert to grayscale and calculate
                original_gray = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                predicted_gray = cv2.cvtColor(predicted.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                return self._manual_ssim(original_gray.astype(np.float64), 
                                       predicted_gray.astype(np.float64))
    
    def _manual_ssim(self, original: np.ndarray, predicted: np.ndarray, 
                    k1: float = 0.01, k2: float = 0.03, win_size: int = 11) -> float:
        """
        Manual SSIM calculation as fallback.
        
        Args:
            original (np.ndarray): Original grayscale image
            predicted (np.ndarray): Predicted grayscale image
            k1 (float): Algorithm parameter (default: 0.01)
            k2 (float): Algorithm parameter (default: 0.03)
            win_size (int): Window size for local calculations (default: 11)
            
        Returns:
            float: SSIM value
        """
        # Constants
        data_range = 255.0  # Assume 8-bit images
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
        
        # Create Gaussian kernel
        kernel = cv2.getGaussianKernel(win_size, 1.5)
        kernel = np.outer(kernel, kernel)
        
        # Calculate local means
        mu1 = cv2.filter2D(original, -1, kernel)
        mu2 = cv2.filter2D(predicted, -1, kernel)
        
        # Calculate local variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(original ** 2, -1, kernel) - mu1_sq
        sigma2_sq = cv2.filter2D(predicted ** 2, -1, kernel) - mu2_sq
        sigma12 = cv2.filter2D(original * predicted, -1, kernel) - mu1_mu2
        
        # Calculate SSIM
        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim_map = numerator / denominator
        return np.mean(ssim_map)
    
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
            Tuple[float, str]: SSIM value and image filename
        """
        try:
            original = self.load_image(original_path)
            predicted = self.load_image(predicted_path)
            
            ssim_value = self.calculate_ssim(original, predicted)
            filename = os.path.basename(original_path)
            
            result = {
                'filename': filename,
                'ssim': ssim_value,
                'original_path': str(original_path),
                'predicted_path': str(predicted_path)
            }
            self.results.append(result)
            
            return ssim_value, filename
            
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
        
        ssim_values = []
        processed_files = []
        
        print(f"Processing {len(original_files)} images for SSIM validation...")
        
        for original_file in original_files:
            # Look for corresponding predicted file
            predicted_file = predicted_dir / original_file.name
            
            if not predicted_file.exists():
                print(f"Warning: No predicted image found for {original_file.name}")
                continue
            
            ssim_value, filename = self.validate_single_pair(original_file, predicted_file)
            
            if ssim_value > 0:  # Valid SSIM calculated
                ssim_values.append(ssim_value)
                processed_files.append(filename)
                print(f"  {filename}: SSIM = {ssim_value:.4f}")
        
        # Calculate statistics
        if ssim_values:
            stats = {
                'mean_ssim': np.mean(ssim_values),
                'std_ssim': np.std(ssim_values),
                'min_ssim': np.min(ssim_values),
                'max_ssim': np.max(ssim_values),
                'median_ssim': np.median(ssim_values),
                'num_images': len(ssim_values),
                'processed_files': processed_files
            }
        else:
            stats = {
                'mean_ssim': 0.0,
                'std_ssim': 0.0,
                'min_ssim': 0.0,
                'max_ssim': 0.0,
                'median_ssim': 0.0,
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
        print(f"SSIM results saved to: {output_path}")
    
    def calculate_ssim_map(self, original_path: Union[str, Path], 
                          predicted_path: Union[str, Path]) -> Tuple[float, np.ndarray]:
        """
        Calculate SSIM and return the full SSIM map for visualization.
        
        Args:
            original_path (Union[str, Path]): Path to original image
            predicted_path (Union[str, Path]): Path to predicted image
            
        Returns:
            Tuple[float, np.ndarray]: SSIM value and SSIM map
        """
        original = self.load_image(original_path)
        predicted = self.load_image(predicted_path)
        
        return self.calculate_ssim(original, predicted, full=True)


if __name__ == "__main__":
    # Example usage
    validator = SSIMValidator()
    
    # Example: validate single pair
    # ssim_value, filename = validator.validate_single_pair("original.jpg", "predicted.jpg")
    # print(f"SSIM for {filename}: {ssim_value:.4f}")
    
    # Example: validate directory
    # stats = validator.validate_directory("original_images/", "predicted_images/")
    # print(f"Mean SSIM: {stats['mean_ssim']:.4f}")
    
    # Example: get SSIM map for visualization
    # ssim_value, ssim_map = validator.calculate_ssim_map("original.jpg", "predicted.jpg")
    # print(f"SSIM: {ssim_value:.4f}, Map shape: {ssim_map.shape}")
    
    print("SSIM Validator initialized. Use validate_single_pair() or validate_directory() methods.")
