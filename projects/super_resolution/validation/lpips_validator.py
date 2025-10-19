"""
LPIPS (Learned Perceptual Image Patch Similarity) Validator for Super Resolution Models

This module implements LPIPS calculation for evaluating super resolution model performance.
LPIPS uses deep neural networks to measure perceptual similarity between images,
providing a metric that correlates better with human perception than traditional metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import lpips
from typing import Union, Tuple, Optional
import os
from pathlib import Path
import pandas as pd

class LPIPSValidator:
    """
    A class for calculating LPIPS (Learned Perceptual Image Patch Similarity) between 
    original and predicted images.
    
    LPIPS measures perceptual similarity using deep features from pre-trained networks,
    providing a metric that better correlates with human perception compared to 
    traditional pixel-based metrics.
    """
    
    def __init__(self, net: str = 'alex', use_gpu: bool = True):
        """
        Initialize the LPIPS validator.
        
        Args:
            net (str): Network to use for feature extraction ('alex', 'vgg', 'squeeze')
            use_gpu (bool): Whether to use GPU if available
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Initialize LPIPS model
        try:
            self.lpips_model = lpips.LPIPS(net=net).to(self.device)
            print(f"LPIPS model initialized with {net} network on {self.device}")
        except Exception as e:
            print(f"Error initializing LPIPS model: {e}")
            print("Make sure to install lpips: pip install lpips")
            raise
        
        self.net = net
        self.results = []
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def calculate_lpips(self, original: Union[np.ndarray, torch.Tensor], 
                       predicted: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Calculate LPIPS between two images.
        
        Args:
            original (Union[np.ndarray, torch.Tensor]): Original high-resolution image
            predicted (Union[np.ndarray, torch.Tensor]): Predicted/reconstructed image
            
        Returns:
            float: LPIPS distance (lower is better, 0 means identical)
            
        Raises:
            ValueError: If images have different shapes or invalid dimensions
        """
        # Convert numpy arrays to PIL Images if needed
        if isinstance(original, np.ndarray):
            if original.shape[2] == 3:  # RGB
                original_pil = Image.fromarray(original.astype(np.uint8))
            else:
                raise ValueError("Original image must have 3 channels (RGB)")
        else:
            original_pil = transforms.ToPILImage()(original)
        
        if isinstance(predicted, np.ndarray):
            if predicted.shape[2] == 3:  # RGB
                predicted_pil = Image.fromarray(predicted.astype(np.uint8))
            else:
                raise ValueError("Predicted image must have 3 channels (RGB)")
        else:
            predicted_pil = transforms.ToPILImage()(predicted)
        
        # Check if images have the same size
        if original_pil.size != predicted_pil.size:
            print(f"Warning: Resizing images to match. Original: {original_pil.size}, Predicted: {predicted_pil.size}")
            # Resize predicted to match original
            predicted_pil = predicted_pil.resize(original_pil.size, Image.LANCZOS)
        
        # Transform images to tensors
        original_tensor = self.transform(original_pil).unsqueeze(0).to(self.device)
        predicted_tensor = self.transform(predicted_pil).unsqueeze(0).to(self.device)
        
        # Calculate LPIPS
        with torch.no_grad():
            lpips_distance = self.lpips_model(original_tensor, predicted_tensor)
        
        return lpips_distance.item()
    
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
        
        try:
            # Load image using PIL (handles various formats better)
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            return image_array
        except Exception as e:
            raise ValueError(f"Could not load image {image_path}: {str(e)}")
    
    def validate_single_pair(self, original_path: Union[str, Path], 
                           predicted_path: Union[str, Path]) -> Tuple[float, str]:
        """
        Validate a single pair of original and predicted images.
        
        Args:
            original_path (Union[str, Path]): Path to original image
            predicted_path (Union[str, Path]): Path to predicted image
            
        Returns:
            Tuple[float, str]: LPIPS distance and image filename
        """
        try:
            original = self.load_image(original_path)
            predicted = self.load_image(predicted_path)
            
            lpips_distance = self.calculate_lpips(original, predicted)
            filename = os.path.basename(original_path)
            
            result = {
                'filename': filename,
                'lpips': lpips_distance,
                'original_path': str(original_path),
                'predicted_path': str(predicted_path)
            }
            self.results.append(result)
            
            return lpips_distance, filename
            
        except Exception as e:
            print(f"Error processing {original_path}: {str(e)}")
            return float('inf'), os.path.basename(str(original_path))
    
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
        
        lpips_values = []
        processed_files = []
        
        print(f"Processing {len(original_files)} images for LPIPS validation...")
        
        for original_file in original_files:
            # Look for corresponding predicted file
            predicted_file = predicted_dir / original_file.name
            
            if not predicted_file.exists():
                print(f"Warning: No predicted image found for {original_file.name}")
                continue
            
            lpips_distance, filename = self.validate_single_pair(original_file, predicted_file)
            
            if lpips_distance != float('inf'):  # Valid LPIPS calculated
                lpips_values.append(lpips_distance)
                processed_files.append(filename)
                print(f"  {filename}: LPIPS = {lpips_distance:.4f}")
        
        # Calculate statistics
        if lpips_values:
            stats = {
                'mean_lpips': np.mean(lpips_values),
                'std_lpips': np.std(lpips_values),
                'min_lpips': np.min(lpips_values),
                'max_lpips': np.max(lpips_values),
                'median_lpips': np.median(lpips_values),
                'num_images': len(lpips_values),
                'processed_files': processed_files
            }
        else:
            stats = {
                'mean_lpips': float('inf'),
                'std_lpips': 0.0,
                'min_lpips': float('inf'),
                'max_lpips': float('inf'),
                'median_lpips': float('inf'),
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
        print(f"LPIPS results saved to: {output_path}")
    
    def set_network(self, net: str):
        """
        Change the network used for LPIPS calculation.
        
        Args:
            net (str): Network to use ('alex', 'vgg', 'squeeze')
        """
        if net != self.net:
            try:
                self.lpips_model = lpips.LPIPS(net=net).to(self.device)
                self.net = net
                print(f"LPIPS network changed to: {net}")
            except Exception as e:
                print(f"Error changing network to {net}: {e}")


if __name__ == "__main__":
    # Example usage
    try:
        validator = LPIPSValidator(net='alex')
        
        # Example: validate single pair
        # lpips_distance, filename = validator.validate_single_pair("original.jpg", "predicted.jpg")
        # print(f"LPIPS for {filename}: {lpips_distance:.4f}")
        
        # Example: validate directory
        # stats = validator.validate_directory("original_images/", "predicted_images/")
        # print(f"Mean LPIPS: {stats['mean_lpips']:.4f}")
        
        print("LPIPS Validator initialized. Use validate_single_pair() or validate_directory() methods.")
        
    except Exception as e:
        print(f"Failed to initialize LPIPS validator: {e}")
        print("Make sure to install required dependencies:")
        print("  pip install lpips torch torchvision pillow")
