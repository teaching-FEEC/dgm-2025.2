#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Enhancement utilities for InvSR:
- Adaptive Guidance Scale
- Edge-Preserving Enhancement
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .util_adaptive import compute_image_complexity


def compute_adaptive_guidance_scale(base_guidance: float, complexity: float,
                                    min_guidance: float = 1.0, max_guidance: float = 10.0) -> float:
    """
    Compute adaptive guidance scale based on image complexity.
    
    Args:
        base_guidance: Base guidance scale (e.g., 7.5)
        complexity: Complexity score (0-1)
        min_guidance: Minimum guidance scale
        max_guidance: Maximum guidance scale
    
    Returns:
        adaptive_guidance: Adjusted guidance scale
    """
    # Higher complexity = higher guidance (more control needed)
    # Lower complexity = lower guidance (less control needed, more natural)
    
    # Map complexity to guidance adjustment
    # Simple images: reduce guidance slightly (more natural)
    # Complex images: increase guidance (more control)
    
    if complexity < 0.3:
        # Simple image: reduce guidance by 10-20%
        guidance = base_guidance * (0.8 + 0.1 * (complexity / 0.3))
    elif complexity < 0.6:
        # Medium complexity: keep base guidance
        guidance = base_guidance
    else:
        # High complexity: increase guidance by 10-30%
        excess_complexity = (complexity - 0.6) / 0.4  # Normalize to [0, 1]
        guidance = base_guidance * (1.0 + 0.3 * excess_complexity)
    
    # Clamp to valid range
    guidance = max(min_guidance, min(max_guidance, guidance))
    
    return float(guidance)


def edge_preserving_enhancement(image: Tensor, strength: float = 0.3) -> Tensor:
    """
    Apply edge-preserving enhancement to improve edge sharpness and detail.
    
    Args:
        image: B x C x H x W tensor, normalized to [0, 1]
        strength: Enhancement strength (0.0-1.0)
    
    Returns:
        enhanced: B x C x H x W tensor with enhanced edges
    """
    if strength <= 0.0:
        return image
    
    # Convert to grayscale for edge detection
    if image.shape[1] == 3:
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    else:
        gray = image
    
    # Sobel operators for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    
    # Compute gradients
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize edge magnitude to [0, 1]
    edge_max = edge_magnitude.view(image.shape[0], -1).max(dim=1, keepdim=True)[0]
    edge_max = edge_max.unsqueeze(-1).unsqueeze(-1) + 1e-8
    edge_normalized = edge_magnitude / edge_max
    
    # Create edge mask (stronger edges = higher weight)
    edge_mask = edge_normalized.clamp(0.0, 1.0)
    
    # Apply unsharp masking for edge enhancement
    # Unsharp mask = original + (original - blurred) * strength
    kernel_size = 5
    sigma = 1.5
    kernel = _create_gaussian_kernel(kernel_size, sigma, image.dtype, image.device)
    
    # Expand kernel to match number of channels for depthwise convolution
    num_channels = image.shape[1]
    kernel = kernel.repeat(num_channels, 1, 1, 1)
    
    blurred = F.conv2d(image, kernel, padding=kernel_size//2, groups=num_channels)
    unsharp = image + (image - blurred) * strength
    
    # Blend: use unsharp in edge regions, original in smooth regions
    # Edge regions get more enhancement
    edge_strength = edge_mask * strength * 2.0  # Double strength in edge regions
    enhanced = image * (1.0 - edge_strength) + unsharp * edge_strength
    
    return enhanced.clamp(0.0, 1.0)


def _create_gaussian_kernel(size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Create a Gaussian kernel for blurring."""
    if size % 2 == 0:
        size += 1
    
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    
    kernel = g[:, None] * g[None, :]
    kernel = kernel.view(1, 1, size, size)
    
    return kernel


def adaptive_sharpening(image: Tensor, complexity: float, max_strength: float = 0.4) -> Tensor:
    """
    Apply adaptive sharpening based on image complexity.
    
    Args:
        image: B x C x H x W tensor
        complexity: Complexity score (0-1)
        max_strength: Maximum sharpening strength
    
    Returns:
        sharpened: B x C x H x W tensor
    """
    # Higher complexity = more sharpening needed
    strength = max_strength * complexity
    
    if strength <= 0.0:
        return image
    
    return edge_preserving_enhancement(image, strength=strength)


