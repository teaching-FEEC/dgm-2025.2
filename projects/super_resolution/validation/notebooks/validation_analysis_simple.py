"""
Super Resolution Model Performance Analysis Script

This script analyzes validation results from super resolution models using PSNR, LPIPS, and SSIM metrics.
It creates comprehensive visualizations to understand model performance patterns and characteristics.

Usage:
    python validation_analysis_simple.py

Configuration:
    Modify the validation_results_path variable to point to your validation results directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")
print("Plotting style configured")


class ValidationAnalyzer:
    """Class to analyze super resolution validation results."""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.summary_data = None
        self.detailed_data = None
        self.psnr_data = None
        self.lpips_data = None
        self.ssim_data = None
        
    def load_results(self, model_prefix: str = None) -> bool:
        """Load validation results from files."""
        try:
            # Find files
            if model_prefix:
                pattern = f"{model_prefix}_validation_*"
            else:
                # Find the most recent validation files
                json_files = list(self.results_path.glob("*_summary.json"))
                if not json_files:
                    print("No summary JSON files found!")
                    return False
                
                # Get the most recent file
                latest_file = max(json_files, key=os.path.getctime)
                model_prefix = latest_file.stem.replace("_summary", "")
                print(f"Using latest results: {model_prefix}")
            
            # Load summary JSON
            summary_file = self.results_path / f"{model_prefix}_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.summary_data = json.load(f)
                print(f"Loaded summary: {summary_file.name}")
            
            # Load detailed CSV
            detailed_file = self.results_path / f"{model_prefix}_detailed.csv"
            if detailed_file.exists():
                self.detailed_data = pd.read_csv(detailed_file)
                print(f"Loaded detailed data: {detailed_file.name}")
            
            # Load individual metric CSVs
            psnr_file = self.results_path / f"{model_prefix}_psnr.csv"
            if psnr_file.exists():
                self.psnr_data = pd.read_csv(psnr_file)
                print(f"Loaded PSNR data: {psnr_file.name}")
            
            lpips_file = self.results_path / f"{model_prefix}_lpips.csv"
            if lpips_file.exists():
                self.lpips_data = pd.read_csv(lpips_file)
                print(f"Loaded LPIPS data: {lpips_file.name}")
            
            ssim_file = self.results_path / f"{model_prefix}_ssim.csv"
            if ssim_file.exists():
                self.ssim_data = pd.read_csv(ssim_file)
                print(f"Loaded SSIM data: {ssim_file.name}")
            
            return True
            
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from loaded data."""
        if not self.summary_data:
            return {}
        
        return self.summary_data.get('metrics', {})
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if not self.summary_data:
            return {}
        
        return {
            'model_name': self.summary_data.get('model_name', 'Unknown'),
            'validation_time': self.summary_data.get('validation_time', 'Unknown'),
            'original_dir': self.summary_data.get('original_dir', 'Unknown'),
            'predicted_dir': self.summary_data.get('predicted_dir', 'Unknown')
        }
    
    def plot_metric_distributions(self):
        """Create distribution plots for all metrics."""
        if self.detailed_data is None:
            print("No detailed data available for plotting!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Super Resolution Model Performance - Metric Distributions', fontsize=16, fontweight='bold')
        
        # PSNR Distribution
        if 'psnr' in self.detailed_data.columns:
            axes[0, 0].hist(self.detailed_data['psnr'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(self.detailed_data['psnr'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {self.detailed_data["psnr"].mean():.2f} dB')
            axes[0, 0].set_title('PSNR Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('PSNR (dB)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # LPIPS Distribution
        if 'lpips' in self.detailed_data.columns:
            axes[0, 1].hist(self.detailed_data['lpips'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[0, 1].axvline(self.detailed_data['lpips'].mean(), color='red', linestyle='--',
                              label=f'Mean: {self.detailed_data["lpips"].mean():.4f}')
            axes[0, 1].set_title('LPIPS Distribution', fontweight='bold')
            axes[0, 1].set_xlabel('LPIPS (lower is better)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # SSIM Distribution
        if 'ssim' in self.detailed_data.columns:
            axes[1, 0].hist(self.detailed_data['ssim'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].axvline(self.detailed_data['ssim'].mean(), color='red', linestyle='--',
                              label=f'Mean: {self.detailed_data["ssim"].mean():.4f}')
            axes[1, 0].set_title('SSIM Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('SSIM')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Combined Box Plot
        metrics_data = []
        if 'psnr' in self.detailed_data.columns:
            # Normalize PSNR to 0-1 scale for comparison
            psnr_norm = (self.detailed_data['psnr'] - self.detailed_data['psnr'].min()) / \
                       (self.detailed_data['psnr'].max() - self.detailed_data['psnr'].min())
            metrics_data.append(('PSNR (norm)', psnr_norm))
        
        if 'lpips' in self.detailed_data.columns:
            # Invert LPIPS (1 - lpips) so higher is better like other metrics
            lpips_inv = 1 - self.detailed_data['lpips']
            metrics_data.append(('LPIPS (inv)', lpips_inv))
        
        if 'ssim' in self.detailed_data.columns:
            metrics_data.append(('SSIM', self.detailed_data['ssim']))
        
        if metrics_data:
            box_data = [data for _, data in metrics_data]
            box_labels = [label for label, _ in metrics_data]
            
            bp = axes[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
            colors = ['skyblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
            
            axes[1, 1].set_title('Metrics Comparison (Normalized)', fontweight='bold')
            axes[1, 1].set_ylabel('Normalized Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print("Distribution plots created successfully!")
    
    def plot_correlations(self):
        """Create correlation plots between metrics."""
        if self.detailed_data is None:
            print("No detailed data available for correlation analysis!")
            return
        
        # Check which metrics are available
        available_metrics = [col for col in ['psnr', 'lpips', 'ssim'] if col in self.detailed_data.columns]
        
        if len(available_metrics) < 2:
            print("Need at least 2 metrics for correlation analysis!")
            return
        
        # Calculate correlation matrix
        corr_data = self.detailed_data[available_metrics].corr()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Metric Correlations Analysis', fontsize=16, fontweight='bold')
        
        # Correlation heatmap
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0], cbar_kws={'label': 'Correlation Coefficient'})
        axes[0].set_title('Correlation Matrix', fontweight='bold')
        
        # Scatter plot for strongest correlation
        if len(available_metrics) >= 2:
            # Find the pair with highest absolute correlation (excluding diagonal)
            corr_abs = corr_data.abs()
            np.fill_diagonal(corr_abs.values, 0)
            max_corr_idx = np.unravel_index(corr_abs.values.argmax(), corr_abs.shape)
            metric1 = corr_abs.index[max_corr_idx[0]]
            metric2 = corr_abs.columns[max_corr_idx[1]]
            
            axes[1].scatter(self.detailed_data[metric1], self.detailed_data[metric2], 
                           alpha=0.6, s=50)
            
            # Add trend line
            z = np.polyfit(self.detailed_data[metric1], self.detailed_data[metric2], 1)
            p = np.poly1d(z)
            axes[1].plot(self.detailed_data[metric1], p(self.detailed_data[metric1]), 
                        "r--", alpha=0.8, linewidth=2)
            
            corr_val = corr_data.loc[metric1, metric2]
            axes[1].set_title(f'{metric1.upper()} vs {metric2.upper()}\\n(r = {corr_val:.3f})', fontweight='bold')
            axes[1].set_xlabel(metric1.upper())
            axes[1].set_ylabel(metric2.upper())
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print correlation insights
        print("Correlation Insights:")
        for i, metric1 in enumerate(available_metrics):
            for j, metric2 in enumerate(available_metrics):
                if i < j:
                    corr_val = corr_data.loc[metric1, metric2]
                    if abs(corr_val) > 0.7:
                        strength = "Strong"
                    elif abs(corr_val) > 0.4:
                        strength = "Moderate"
                    else:
                        strength = "Weak"
                    
                    direction = "positive" if corr_val > 0 else "negative"
                    print(f"   {metric1.upper()} vs {metric2.upper()}: {strength} {direction} correlation ({corr_val:.3f})")
    
    def plot_quality_dashboard(self):
        """Create a comprehensive quality assessment dashboard."""
        if self.detailed_data is None:
            print("No detailed data available for dashboard!")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Super Resolution Model - Quality Assessment Dashboard', fontsize=18, fontweight='bold')
        
        # Quality thresholds
        psnr_thresholds = {'Excellent': 30, 'Good': 25, 'Fair': 20}
        lpips_thresholds = {'Excellent': 0.1, 'Good': 0.2, 'Fair': 0.3}
        ssim_thresholds = {'Excellent': 0.9, 'Good': 0.8, 'Fair': 0.7}
        
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        
        # PSNR Quality Distribution
        if 'psnr' in self.detailed_data.columns:
            ax1 = fig.add_subplot(gs[0, 0])
            psnr_data = self.detailed_data['psnr']
            
            excellent = (psnr_data >= psnr_thresholds['Excellent']).sum()
            good = ((psnr_data >= psnr_thresholds['Good']) & (psnr_data < psnr_thresholds['Excellent'])).sum()
            fair = ((psnr_data >= psnr_thresholds['Fair']) & (psnr_data < psnr_thresholds['Good'])).sum()
            poor = (psnr_data < psnr_thresholds['Fair']).sum()
            
            labels = ['Excellent\\n(≥30dB)', 'Good\\n(25-30dB)', 'Fair\\n(20-25dB)', 'Poor\\n(<20dB)']
            sizes = [excellent, good, fair, poor]
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('PSNR Quality Distribution', fontweight='bold')
        
        # Combined metrics over images (line plot)
        ax4 = fig.add_subplot(gs[1, :])
        
        x = range(len(self.detailed_data))
        
        if 'psnr' in self.detailed_data.columns:
            # Normalize PSNR to 0-1 scale
            psnr_norm = (self.detailed_data['psnr'] - self.detailed_data['psnr'].min()) / \
                       (self.detailed_data['psnr'].max() - self.detailed_data['psnr'].min())
            ax4.plot(x, psnr_norm, 'o-', label='PSNR (normalized)', alpha=0.7, linewidth=2, markersize=4)
        
        if 'lpips' in self.detailed_data.columns:
            # Invert LPIPS so higher is better
            lpips_inv = 1 - self.detailed_data['lpips']
            ax4.plot(x, lpips_inv, 's-', label='LPIPS (inverted)', alpha=0.7, linewidth=2, markersize=4)
        
        if 'ssim' in self.detailed_data.columns:
            ax4.plot(x, self.detailed_data['ssim'], '^-', label='SSIM', alpha=0.7, linewidth=2, markersize=4)
        
        ax4.set_title('Metric Performance Across All Images', fontweight='bold')
        ax4.set_xlabel('Image Index')
        ax4.set_ylabel('Normalized Score (Higher is Better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.show()
        print("Quality Assessment Dashboard created successfully!")
    
    def print_summary_stats(self):
        """Display summary statistics."""
        if not self.summary_data:
            print("No summary data available!")
            return
        
        stats = self.get_summary_stats()
        
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        # PSNR Statistics
        if 'psnr' in stats and 'error' not in stats['psnr']:
            psnr = stats['psnr']
            print(f"\\nPSNR (Peak Signal-to-Noise Ratio):")
            print(f"   Mean:   {psnr['mean_psnr']:.2f} dB")
            print(f"   Std:    {psnr['std_psnr']:.2f} dB")
            print(f"   Min:    {psnr['min_psnr']:.2f} dB")
            print(f"   Max:    {psnr['max_psnr']:.2f} dB")
            print(f"   Median: {psnr['median_psnr']:.2f} dB")
            print(f"   Images: {psnr['num_images']}")
        
        # LPIPS Statistics
        if 'lpips' in stats and 'error' not in stats['lpips']:
            lpips = stats['lpips']
            print(f"\\nLPIPS (Learned Perceptual Image Patch Similarity):")
            print(f"   Mean:   {lpips['mean_lpips']:.4f}")
            print(f"   Std:    {lpips['std_lpips']:.4f}")
            print(f"   Min:    {lpips['min_lpips']:.4f}")
            print(f"   Max:    {lpips['max_lpips']:.4f}")
            print(f"   Median: {lpips['median_lpips']:.4f}")
            print(f"   Images: {lpips['num_images']}")
        
        # SSIM Statistics
        if 'ssim' in stats and 'error' not in stats['ssim']:
            ssim = stats['ssim']
            print(f"\\nSSIM (Structural Similarity Index Measure):")
            print(f"   Mean:   {ssim['mean_ssim']:.4f}")
            print(f"   Std:    {ssim['std_ssim']:.4f}")
            print(f"   Min:    {ssim['min_ssim']:.4f}")
            print(f"   Max:    {ssim['max_ssim']:.4f}")
            print(f"   Median: {ssim['median_ssim']:.4f}")
            print(f"   Images: {ssim['num_images']}")
        
        print("\\n" + "=" * 50)


def main():
    """Main function to run the analysis."""
    # Configuration - Modify this path to point to your validation results
    validation_results_path = "../data/validation_results"
    
    # Convert to absolute path
    validation_results_path = os.path.abspath(validation_results_path)
    print(f"Validation results path: {validation_results_path}")
    
    # Check if path exists
    if not os.path.exists(validation_results_path):
        print("Path does not exist! Please check the path.")
        return
    
    print("Path exists!")
    
    # List available files
    files = [f for f in os.listdir(validation_results_path) if f.endswith(('.json', '.csv'))]
    print(f"Found {len(files)} result files:")
    for file in sorted(files):
        print(f"   - {file}")
    
    # Initialize analyzer
    analyzer = ValidationAnalyzer(validation_results_path)
    print("\\nValidationAnalyzer initialized!")
    
    # Load the validation results
    success = analyzer.load_results()
    
    if not success:
        print("Failed to load validation results!")
        return
    
    print("\\nAll data loaded successfully!")
    
    # Display model information
    model_info = analyzer.get_model_info()
    print("\\nModel Information:")
    print(f"   Model Name: {model_info['model_name']}")
    print(f"   Validation Time: {model_info['validation_time']}")
    print(f"   Original Images: {model_info['original_dir']}")
    print(f"   Predicted Images: {model_info['predicted_dir']}")
    
    # Display data shapes
    if analyzer.detailed_data is not None:
        print(f"\\nData Shape: {analyzer.detailed_data.shape}")
        print(f"   Columns: {list(analyzer.detailed_data.columns)}")
        print(f"\\nFirst few rows:")
        print(analyzer.detailed_data.head())
    
    # Print summary statistics
    analyzer.print_summary_stats()
    
    # Create visualizations
    print("\\nCreating visualizations...")
    
    # 1. Metric distributions
    analyzer.plot_metric_distributions()
    
    # 2. Correlation analysis
    analyzer.plot_correlations()
    
    # 3. Quality dashboard
    analyzer.plot_quality_dashboard()
    
    print("\\nAnalysis completed successfully!")
    print("All visualizations have been generated.")


if __name__ == "__main__":
    main()
