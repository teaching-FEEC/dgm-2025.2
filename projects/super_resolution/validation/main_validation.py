"""
Main Validation Script for Super Resolution Models

This script provides a comprehensive validation framework for super resolution models
using PSNR, LPIPS, and SSIM metrics. It processes directories of original and predicted
images and generates detailed reports with statistical analysis.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import validation classes
from psnr_validator import PSNRValidator
from lpips_validator import LPIPSValidator
from ssim_validator import SSIMValidator


class SuperResolutionValidator:
    """
    Main validation class that orchestrates all three metrics (PSNR, LPIPS, SSIM)
    for comprehensive super resolution model evaluation.
    """
    
    def __init__(self, model_name: str, use_gpu: bool = True, lpips_net: str = 'alex'):
        """
        Initialize the comprehensive validator.
        
        Args:
            model_name (str): Name of the model being validated
            use_gpu (bool): Whether to use GPU for LPIPS calculation
            lpips_net (str): Network to use for LPIPS ('alex', 'vgg', 'squeeze')
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        
        # Initialize validators
        print("Initializing validation metrics...")
        
        try:
            self.psnr_validator = PSNRValidator()
            print("PSNR validator initialized")
        except Exception as e:
            print(f"Failed to initialize PSNR validator: {e}")
            self.psnr_validator = None
        
        try:
            self.lpips_validator = LPIPSValidator(net=lpips_net, use_gpu=use_gpu)
            print("LPIPS validator initialized")
        except Exception as e:
            print(f"Failed to initialize LPIPS validator: {e}")
            print("  Make sure to install: pip install lpips torch torchvision")
            self.lpips_validator = None
        
        try:
            self.ssim_validator = SSIMValidator()
            print("SSIM validator initialized")
        except Exception as e:
            print(f"Failed to initialize SSIM validator: {e}")
            self.ssim_validator = None
        
        self.results = {}
        self.validation_time = None
    
    def validate_model(self, original_images_dir: str, predicted_images_dir: str, 
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation using all available metrics.
        
        Args:
            original_images_dir (str): Directory containing original high-resolution images
            predicted_images_dir (str): Directory containing model predictions
            output_dir (Optional[str]): Directory to save results (default: current directory)
            
        Returns:
            Dict[str, Any]: Comprehensive validation results
        """
        print(f"\n{'='*60}")
        print(f"SUPER RESOLUTION MODEL VALIDATION")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Original Images: {original_images_dir}")
        print(f"Predicted Images: {predicted_images_dir}")
        print(f"{'='*60}\n")
        
        self.validation_time = datetime.now()
        
        # Validate directories exist
        if not os.path.exists(original_images_dir):
            raise FileNotFoundError(f"Original images directory not found: {original_images_dir}")
        
        if not os.path.exists(predicted_images_dir):
            raise FileNotFoundError(f"Predicted images directory not found: {predicted_images_dir}")
        
        # Initialize results dictionary
        validation_results = {
            'model_name': self.model_name,
            'validation_time': self.validation_time.isoformat(),
            'original_dir': original_images_dir,
            'predicted_dir': predicted_images_dir,
            'metrics': {}
        }
        
        # Run PSNR validation
        if self.psnr_validator:
            print("Running PSNR validation...")
            try:
                psnr_stats = self.psnr_validator.validate_directory(
                    original_images_dir, predicted_images_dir
                )
                validation_results['metrics']['psnr'] = psnr_stats
                print(f"✓ PSNR validation completed: Mean = {psnr_stats['mean_psnr']:.2f} dB")
            except Exception as e:
                print(f"✗ PSNR validation failed: {e}")
                validation_results['metrics']['psnr'] = {'error': str(e)}
        
        # Run LPIPS validation
        if self.lpips_validator:
            print("\nRunning LPIPS validation...")
            try:
                lpips_stats = self.lpips_validator.validate_directory(
                    original_images_dir, predicted_images_dir
                )
                validation_results['metrics']['lpips'] = lpips_stats
                print(f"✓ LPIPS validation completed: Mean = {lpips_stats['mean_lpips']:.4f}")
            except Exception as e:
                print(f"✗ LPIPS validation failed: {e}")
                validation_results['metrics']['lpips'] = {'error': str(e)}
        
        # Run SSIM validation
        if self.ssim_validator:
            print("\nRunning SSIM validation...")
            try:
                ssim_stats = self.ssim_validator.validate_directory(
                    original_images_dir, predicted_images_dir
                )
                validation_results['metrics']['ssim'] = ssim_stats
                print(f"✓ SSIM validation completed: Mean = {ssim_stats['mean_ssim']:.4f}")
            except Exception as e:
                print(f"✗ SSIM validation failed: {e}")
                validation_results['metrics']['ssim'] = {'error': str(e)}
        
        # Store results
        self.results = validation_results
        
        # Save results if output directory is specified
        if output_dir:
            self.save_results(output_dir)
        
        # Print summary
        self.print_summary()
        
        return validation_results
    
    def print_summary(self):
        """Print a comprehensive summary of validation results."""
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY - {self.model_name}")
        print(f"{'='*60}")
        
        if 'metrics' not in self.results:
            print("No validation results available.")
            return
        
        metrics = self.results['metrics']
        
        # PSNR Summary
        if 'psnr' in metrics and 'error' not in metrics['psnr']:
            psnr = metrics['psnr']
            print(f"\nPSNR (Peak Signal-to-Noise Ratio):")
            print(f"   Mean:   {psnr['mean_psnr']:.2f} dB")
            print(f"   Std:    {psnr['std_psnr']:.2f} dB")
            print(f"   Min:    {psnr['min_psnr']:.2f} dB")
            print(f"   Max:    {psnr['max_psnr']:.2f} dB")
            print(f"   Median: {psnr['median_psnr']:.2f} dB")
            print(f"   Images: {psnr['num_images']}")
        
        # LPIPS Summary
        if 'lpips' in metrics and 'error' not in metrics['lpips']:
            lpips = metrics['lpips']
            print(f"\nLPIPS (Learned Perceptual Image Patch Similarity):")
            print(f"   Mean:   {lpips['mean_lpips']:.4f}")
            print(f"   Std:    {lpips['std_lpips']:.4f}")
            print(f"   Min:    {lpips['min_lpips']:.4f}")
            print(f"   Max:    {lpips['max_lpips']:.4f}")
            print(f"   Median: {lpips['median_lpips']:.4f}")
            print(f"   Images: {lpips['num_images']}")
        
        # SSIM Summary
        if 'ssim' in metrics and 'error' not in metrics['ssim']:
            ssim = metrics['ssim']
            print(f"\nSSIM (Structural Similarity Index Measure):")
            print(f"   Mean:   {ssim['mean_ssim']:.4f}")
            print(f"   Std:    {ssim['std_ssim']:.4f}")
            print(f"   Min:    {ssim['min_ssim']:.4f}")
            print(f"   Max:    {ssim['max_ssim']:.4f}")
            print(f"   Median: {ssim['median_ssim']:.4f}")
            print(f"   Images: {ssim['num_images']}")
        
        # Overall Assessment
        print(f"\nOverall Assessment:")
        self._print_quality_assessment()
        
        print(f"\n{'='*60}")
    
    def _print_quality_assessment(self):
        """Print quality assessment based on metric values."""
        if 'metrics' not in self.results:
            return
        
        metrics = self.results['metrics']
        assessments = []
        
        # PSNR Assessment
        if 'psnr' in metrics and 'error' not in metrics['psnr']:
            psnr_mean = metrics['psnr']['mean_psnr']
            if psnr_mean >= 30:
                assessments.append("PSNR: Excellent (≥30 dB)")
            elif psnr_mean >= 25:
                assessments.append("PSNR: Good (25-30 dB)")
            elif psnr_mean >= 20:
                assessments.append("PSNR: Fair (20-25 dB)")
            else:
                assessments.append("PSNR: Poor (<20 dB)")
        
        # LPIPS Assessment (lower is better)
        if 'lpips' in metrics and 'error' not in metrics['lpips']:
            lpips_mean = metrics['lpips']['mean_lpips']
            if lpips_mean <= 0.1:
                assessments.append("LPIPS: Excellent (≤0.1)")
            elif lpips_mean <= 0.2:
                assessments.append("LPIPS: Good (0.1-0.2)")
            elif lpips_mean <= 0.3:
                assessments.append("LPIPS: Fair (0.2-0.3)")
            else:
                assessments.append("LPIPS: Poor (>0.3)")
        
        # SSIM Assessment
        if 'ssim' in metrics and 'error' not in metrics['ssim']:
            ssim_mean = metrics['ssim']['mean_ssim']
            if ssim_mean >= 0.9:
                assessments.append("SSIM: Excellent (≥0.9)")
            elif ssim_mean >= 0.8:
                assessments.append("SSIM: Good (0.8-0.9)")
            elif ssim_mean >= 0.7:
                assessments.append("SSIM: Fair (0.7-0.8)")
            else:
                assessments.append("SSIM: Poor (<0.7)")
        
        for assessment in assessments:
            print(f"   {assessment}")
    
    def save_results(self, output_dir: str):
        """
        Save validation results to files.
        
        Args:
            output_dir (str): Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = self.validation_time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.model_name}_validation_{timestamp}"
        
        # Save JSON summary
        json_path = output_path / f"{base_filename}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Summary saved to: {json_path}")
        
        # Save detailed CSV results
        self._save_detailed_csv(output_path, base_filename)
        
        # Save individual metric results
        if self.psnr_validator and self.psnr_validator.get_results():
            psnr_path = output_path / f"{base_filename}_psnr.csv"
            self.psnr_validator.save_results(psnr_path)
        
        if self.lpips_validator and self.lpips_validator.get_results():
            lpips_path = output_path / f"{base_filename}_lpips.csv"
            self.lpips_validator.save_results(lpips_path)
        
        if self.ssim_validator and self.ssim_validator.get_results():
            ssim_path = output_path / f"{base_filename}_ssim.csv"
            self.ssim_validator.save_results(ssim_path)
    
    def _save_detailed_csv(self, output_path: Path, base_filename: str):
        """Save combined detailed results to CSV."""
        try:
            # Combine all results into a single DataFrame
            combined_data = []
            
            # Get all results
            psnr_results = self.psnr_validator.get_results() if self.psnr_validator else []
            lpips_results = self.lpips_validator.get_results() if self.lpips_validator else []
            ssim_results = self.ssim_validator.get_results() if self.ssim_validator else []
            
            # Create a mapping by filename
            results_by_file = {}
            
            for result in psnr_results:
                filename = result['filename']
                if filename not in results_by_file:
                    results_by_file[filename] = {'filename': filename}
                results_by_file[filename]['psnr'] = result['psnr']
            
            for result in lpips_results:
                filename = result['filename']
                if filename not in results_by_file:
                    results_by_file[filename] = {'filename': filename}
                results_by_file[filename]['lpips'] = result['lpips']
            
            for result in ssim_results:
                filename = result['filename']
                if filename not in results_by_file:
                    results_by_file[filename] = {'filename': filename}
                results_by_file[filename]['ssim'] = result['ssim']
            
            # Convert to list
            combined_data = list(results_by_file.values())
            
            if combined_data:
                df = pd.DataFrame(combined_data)
                csv_path = output_path / f"{base_filename}_detailed.csv"
                df.to_csv(csv_path, index=False)
                print(f"Detailed results saved to: {csv_path}")
            
        except Exception as e:
            print(f"Warning: Could not save detailed CSV: {e}")


def main():
    """Main function with configurable variables."""
    # Configuration variables - modify these as needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    original_images_dir = os.path.join(data_dir, "high_quality_images")    # Directory containing original images
    predicted_images_dir = os.path.join(data_dir, "predicted_images")   # Directory containing predicted images
    model_name = "TestInvSR"                                               # Name of the model being validated
    output_dir = os.path.join(data_dir, "validation_results")          # Output directory for results
    use_gpu = True                                                         # Whether to use GPU for LPIPS
    lpips_net = 'alex'                                                     # Network for LPIPS calculation ('alex', 'vgg', 'squeeze')
    
    print("Super Resolution Model Validation")
    print("=" * 50)
    print(f"Model name: {model_name}")
    print(f"Original images: {original_images_dir}")
    print(f"Predicted images: {predicted_images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Use GPU: {use_gpu}")
    print(f"LPIPS network: {lpips_net}")
    print("=" * 50)
    
    # Create validator
    validator = SuperResolutionValidator(
        model_name=model_name,
        use_gpu=use_gpu,
        lpips_net=lpips_net
    )
    
    # Run validation
    try:
        results = validator.validate_model(
            original_images_dir=original_images_dir,
            predicted_images_dir=predicted_images_dir,
            output_dir=output_dir
        )
        
        print(f"\nValidation completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nValidation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
