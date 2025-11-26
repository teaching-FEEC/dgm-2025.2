#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Adaptive utilities for InvSR:
- Image complexity analysis
- Adaptive scheduler (timestep selection)
- Attention-guided fusion
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional


def compute_image_complexity(image: Tensor) -> float:
    """
    Compute a complexity score (0-1) for an image.
    Higher score = more complex image (needs more refinement).
    
    Args:
        image: B x C x H x W tensor, normalized to [0, 1]
    
    Returns:
        complexity_score: float in [0, 1]
    """
    if image.dim() == 4:
        # Use first image in batch if batch size > 1
        img = image[0] if image.shape[0] > 0 else image
    else:
        img = image
    
    # Convert to grayscale if RGB
    if img.shape[0] == 3:
        # Use luminance: 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    else:
        gray = img[0]
    
    # 1. Variance (texture complexity)
    variance = torch.var(gray).item()
    
    # 2. Gradient magnitude (edge density)
    # Compute gradients
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    
    gray_padded = gray.unsqueeze(0).unsqueeze(0)
    grad_x = F.conv2d(gray_padded, sobel_x, padding=1)
    grad_y = F.conv2d(gray_padded, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    gradient_mean = torch.mean(gradient_magnitude).item()
    
    # 3. Local variance (spatial complexity)
    # Compute local variance using a sliding window
    kernel_size = 5
    kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                       dtype=gray.dtype, device=gray.device) / (kernel_size ** 2)
    local_mean = F.conv2d(gray_padded, kernel, padding=kernel_size//2)
    local_var = F.conv2d((gray_padded - local_mean)**2, kernel, padding=kernel_size//2)
    local_variance_mean = torch.mean(local_var).item()
    
    # 4. Entropy (information content)
    # Discretize to 256 bins for entropy calculation
    gray_np = gray.detach().cpu().numpy()
    hist, _ = np.histogram(gray_np.flatten(), bins=256, range=(0, 1))
    hist = hist + 1e-10  # Avoid log(0)
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist))
    entropy_normalized = entropy / 8.0  # Normalize to [0, 1] (max entropy for 256 bins is ~8)
    
    # Combine metrics (weighted average)
    # Normalize each metric to [0, 1] range
    variance_norm = min(variance * 10, 1.0)  # Scale variance
    gradient_norm = min(gradient_mean * 5, 1.0)  # Scale gradient
    local_var_norm = min(local_variance_mean * 20, 1.0)  # Scale local variance
    
    # Weighted combination
    complexity = (
        0.25 * variance_norm +
        0.30 * gradient_norm +
        0.25 * local_var_norm +
        0.20 * entropy_normalized
    )
    
    return float(np.clip(complexity, 0.0, 1.0))


def adaptive_timesteps(base_num_steps: int, complexity: float, 
                       min_timestep: int = 0, max_timestep: int = 250,
                       original_timesteps: Optional[List[int]] = None) -> List[int]:
    """
    Generate adaptive timesteps based on image complexity.
    Uses original timesteps as reference to maintain valid ranges.
    
    Args:
        base_num_steps: Base number of steps (e.g., 3)
        complexity: Complexity score (0-1)
        min_timestep: Minimum timestep value (default: 0)
        max_timestep: Maximum timestep value (default: 250)
        original_timesteps: Original timesteps to use as reference (optional)
    
    Returns:
        timesteps: List of timestep values in descending order
    """
    # If original timesteps provided, use them as base and adjust
    if original_timesteps is not None and len(original_timesteps) > 0:
        original_timesteps = sorted(original_timesteps, reverse=True)
        start_t = original_timesteps[0]
        end_t = original_timesteps[-1] if len(original_timesteps) > 1 else min_timestep
        
        # Adjust number of steps based on complexity
        if complexity < 0.3:
            # Simple image: use base steps, more spacing
            num_steps = base_num_steps
            # Keep original spacing but make it slightly more spaced
            if num_steps == 1:
                timesteps = [start_t]
            else:
                # Use original timesteps but with more spacing
                timesteps = np.linspace(start_t, max(end_t, 20), num_steps, dtype=np.int64).tolist()
                timesteps = sorted(timesteps, reverse=True)
        elif complexity < 0.6:
            # Medium complexity: add 1 step
            num_steps = base_num_steps + 1
            # Add one intermediate step
            timesteps = np.linspace(start_t, max(end_t, 20), num_steps, dtype=np.int64).tolist()
            timesteps = sorted(timesteps, reverse=True)
        else:
            # High complexity: add 2 steps, tighter spacing
            num_steps = base_num_steps + 2
            # More steps with tighter spacing
            timesteps = np.linspace(start_t, max(end_t, 20), num_steps, dtype=np.int64).tolist()
            timesteps = sorted(timesteps, reverse=True)
        
        # Ensure timesteps are within valid range
        timesteps = [max(min_timestep, min(max_timestep, int(t))) for t in timesteps]
        timesteps = sorted(list(set(timesteps)), reverse=True)  # Remove duplicates and sort
        
        # Ensure we have at least the base number of steps
        if len(timesteps) < base_num_steps:
            # Fallback to original timesteps
            return original_timesteps
        
        return timesteps
    
    # Fallback: generate from scratch (conservative approach)
    # Adjust number of steps based on complexity
    if complexity < 0.3:
        # Simple image: use base steps, more spacing
        num_steps = base_num_steps
    elif complexity < 0.6:
        # Medium complexity: add 1 step
        num_steps = base_num_steps + 1
    else:
        # High complexity: add 2 steps
        num_steps = base_num_steps + 2
    
    # Generate timesteps conservatively
    if num_steps == 1:
        timesteps = [max_timestep]
    else:
        # Use conservative range: start from max, end at reasonable minimum (not 0)
        safe_min = max(20, min_timestep)  # Don't go too low
        timesteps = np.linspace(max_timestep, safe_min, num_steps, dtype=np.int64).tolist()
        timesteps = sorted(timesteps, reverse=True)
    
    return timesteps


def compute_attention_map(image: Tensor, method: str = 'gradient') -> Tensor:
    """
    Compute attention map highlighting important regions.
    
    Args:
        image: B x C x H x W tensor
        method: 'gradient' (edge-based) or 'variance' (texture-based)
    
    Returns:
        attention_map: B x 1 x H x W tensor, values in [0, 1]
    """
    if image.dim() == 4:
        b, c, h, w = image.shape
    else:
        image = image.unsqueeze(0)
        b, c, h, w = image.shape
    
    # Convert to grayscale if needed
    if c == 3:
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        gray = gray.unsqueeze(1)  # B x 1 x H x W
    else:
        gray = image
    
    if method == 'gradient':
        # Sobel operators for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        attention = torch.sqrt(grad_x**2 + grad_y**2)
        
    elif method == 'variance':
        # Local variance as attention
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                           dtype=gray.dtype, device=gray.device) / (kernel_size ** 2)
        local_mean = F.conv2d(gray, kernel, padding=kernel_size//2)
        attention = F.conv2d((gray - local_mean)**2, kernel, padding=kernel_size//2)
    
    else:
        raise ValueError(f"Unknown attention method: {method}")
    
    # Normalize to [0, 1]
    attention_min = attention.view(b, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    attention_max = attention.view(b, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
    attention = (attention - attention_min) / (attention_max - attention_min + 1e-8)
    
    # Apply Gaussian smoothing for smoother attention
    sigma = 2.0
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    x = torch.arange(kernel_size, dtype=gray.dtype, device=gray.device) - kernel_size // 2
    gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    gaussian_2d = gaussian_2d.view(1, 1, kernel_size, kernel_size)
    
    attention = F.conv2d(attention, gaussian_2d, padding=kernel_size//2)
    
    return attention.clamp(0.0, 1.0)


def attention_guided_fusion(images: List[Tensor], attention_maps: Optional[List[Tensor]] = None,
                           method: str = 'weighted') -> Tensor:
    """
    Fuse multiple images using attention-guided weighting.
    
    Args:
        images: List of B x C x H x W tensors to fuse
        attention_maps: Optional list of B x 1 x H x W attention maps.
                       If None, will be computed automatically.
        method: 'weighted' (attention-weighted average) or 'max' (attention-weighted max)
    
    Returns:
        fused: B x C x H x W tensor
    """
    if len(images) == 0:
        raise ValueError("At least one image required for fusion")
    
    if len(images) == 1:
        return images[0]
    
    # Ensure all images have same shape
    b, c, h, w = images[0].shape
    for img in images[1:]:
        if img.shape != (b, c, h, w):
            raise ValueError(f"All images must have same shape. Got {img.shape} vs {(b, c, h, w)}")
    
    # Compute attention maps if not provided
    if attention_maps is None:
        attention_maps = [compute_attention_map(img) for img in images]
    
    # Normalize attention maps to sum to 1 at each pixel
    attention_stack = torch.stack(attention_maps, dim=0)  # N x B x 1 x H x W
    attention_sum = attention_stack.sum(dim=0, keepdim=True)  # 1 x B x 1 x H x W
    attention_normalized = attention_stack / (attention_sum + 1e-8)  # N x B x 1 x H x W
    
    if method == 'weighted':
        # Weighted average based on attention
        images_stack = torch.stack(images, dim=0)  # N x B x C x H x W
        weights = attention_normalized  # N x B x 1 x H x W
        fused = (images_stack * weights).sum(dim=0)  # B x C x H x W
        
    elif method == 'max':
        # Attention-weighted max (take pixel from image with highest attention)
        images_stack = torch.stack(images, dim=0)  # N x B x C x H x W
        max_indices = attention_normalized.argmax(dim=0)  # B x 1 x H x W
        # Select pixels based on max attention
        b_idx = torch.arange(b, device=images[0].device).view(b, 1, 1, 1)
        c_idx = torch.arange(c, device=images[0].device).view(1, c, 1, 1)
        h_idx = torch.arange(h, device=images[0].device).view(1, 1, h, 1)
        w_idx = torch.arange(w, device=images[0].device).view(1, 1, 1, w)
        
        fused = images_stack[max_indices, b_idx, c_idx, h_idx, w_idx].squeeze(1)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    return fused.clamp(0.0, 1.0)

