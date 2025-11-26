#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Smart Chopping with Adaptive Overlap:
- Analyzes local complexity of image regions
- Adjusts overlap dynamically (25-50% based on complexity)
- Uses attention-guided blending for seamless results
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional
import numpy as np

from .util_image import ImageSpliterTh
from .util_adaptive import compute_image_complexity, compute_attention_map


def compute_local_complexity(image: Tensor, h_start: int, h_end: int, 
                             w_start: int, w_end: int) -> float:
    """
    Compute local complexity for a specific region of the image.
    
    Args:
        image: B x C x H x W tensor
        h_start, h_end, w_start, w_end: Region coordinates
    
    Returns:
        complexity: float in [0, 1]
    """
    # Extract region
    region = image[:, :, h_start:h_end, w_start:w_end]
    
    # Ensure region is not empty
    if region.numel() == 0:
        return 0.5  # Default medium complexity
    
    # Compute complexity for this region
    # Use first image in batch if batch size > 1
    if region.shape[0] > 1:
        region = region[0:1]
    
    complexity = compute_image_complexity(region)
    
    return complexity


def compute_adaptive_stride(pch_size: int, local_complexity: float, 
                           min_overlap: float = 0.25, max_overlap: float = 0.75) -> int:
    """
    Compute adaptive stride based on local complexity.
    
    Args:
        pch_size: Patch size
        local_complexity: Local complexity score (0-1)
        min_overlap: Minimum overlap ratio (default: 0.25 = 25%)
        max_overlap: Maximum overlap ratio (default: 0.50 = 50%)
    
    Returns:
        stride: Stride value (overlap = 1 - stride/pch_size)
    """
    # Map complexity to overlap: high complexity = high overlap
    overlap_ratio = min_overlap + (max_overlap - min_overlap) * local_complexity
    
    # Convert overlap to stride
    # overlap = 1 - stride/pch_size  =>  stride = pch_size * (1 - overlap)
    stride = int(pch_size * (1.0 - overlap_ratio))
    
    # Ensure stride is at least 1 and at most pch_size
    stride = max(1, min(pch_size, stride))
    
    return stride


class ImageSpliterAdaptive(ImageSpliterTh):
    """
    Adaptive Image Splitter with smart overlap based on local complexity.
    Extends ImageSpliterTh with dynamic overlap adjustment.
    """
    
    def __init__(self, im, pch_size, base_stride=None, sf=1, extra_bs=1, 
                 weight_type='Gaussian', adaptive_overlap=True, 
                 min_overlap=0.25, max_overlap=0.50, complexity_map=None):
        '''
        Input:
            im: n x c x h x w, torch tensor, float, low-resolution image in SR
            pch_size: patch size
            base_stride: base stride (if None, uses pch_size * 0.5 for 50% overlap)
            sf: scale factor in image super-resolution
            extra_bs: extra batch size
            weight_type: 'Gaussian' or 'ones'
            adaptive_overlap: enable adaptive overlap
            min_overlap: minimum overlap ratio (0.25 = 25%)
            max_overlap: maximum overlap ratio (0.50 = 50%)
            complexity_map: pre-computed complexity map (optional, B x H x W)
        '''
        # Set base stride if not provided
        if base_stride is None:
            base_stride = int(pch_size * 0.5)  # Default 50% overlap
        
        # Initialize parent with base stride (will be adjusted per patch)
        super().__init__(im, pch_size, base_stride, sf, extra_bs, weight_type)
        
        self.adaptive_overlap = adaptive_overlap
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.complexity_map = complexity_map
        
        # Pre-compute complexity map if not provided and adaptive is enabled
        if adaptive_overlap and complexity_map is None:
            self._precompute_complexity_map()
        
        # Recompute starts with adaptive stride if enabled
        if adaptive_overlap:
            self._compute_adaptive_starts()
    
    def _precompute_complexity_map(self):
        """Pre-compute complexity map for the entire image."""
        bs, chn, height, width = self.im_ori.shape
        
        # Compute complexity in a sliding window fashion
        window_size = min(self.pch_size, min(height, width))
        step = max(1, window_size // 4)  # Sample every quarter of patch size
        
        complexity_map = torch.zeros(bs, height, width, 
                                     dtype=torch.float32, device=self.im_ori.device)
        
        # Sample complexity at regular intervals
        for h in range(0, height, step):
            for w in range(0, width, step):
                h_end = min(h + window_size, height)
                w_end = min(w + window_size, width)
                h_start = max(0, h_end - window_size)
                w_start = max(0, w_end - window_size)
                
                # Ensure valid region
                if h_end > h_start and w_end > w_start:
                    local_comp = compute_local_complexity(
                        self.im_ori, h_start, h_end, w_start, w_end
                    )
                    
                    # Fill the region with this complexity value
                    complexity_map[:, h_start:h_end, w_start:w_end] = local_comp
        
        self.complexity_map = complexity_map
    
    def _get_local_complexity(self, h_start: int, h_end: int, 
                             w_start: int, w_end: int) -> float:
        """Get local complexity for a region."""
        if self.complexity_map is not None:
            # Average complexity in the region
            region_comp = self.complexity_map[:, h_start:h_end, w_start:w_end]
            return float(region_comp.mean().item())
        else:
            # Compute on-the-fly
            return compute_local_complexity(self.im_ori, h_start, h_end, w_start, w_end)
    
    def _compute_adaptive_starts(self):
        """Recompute starts list with adaptive stride."""
        bs, chn, height, width = self.im_ori.shape
        
        # Use a grid-based approach with adaptive stride
        # Start with base stride, then adjust based on complexity
        self.height_starts_list = []
        self.width_starts_list = []
        
        # Height starts - use adaptive stride based on local complexity
        h = 0
        while h < height:
            h_end = min(h + self.pch_size, height)
            h_start = max(0, h_end - self.pch_size)
            
            self.height_starts_list.append(h_start)
            
            # Compute stride for next position based on current region complexity
            if self.adaptive_overlap and h_end < height:
                # Sample complexity in the current region
                local_comp = self._get_local_complexity(h_start, h_end, 0, width)
                stride = compute_adaptive_stride(
                    self.pch_size, local_comp, self.min_overlap, self.max_overlap
                )
            else:
                stride = self.stride
            
            # Next start position
            if h_end >= height:
                break
            h = h + stride
        
        # Width starts
        w = 0
        while w < width:
            w_end = min(w + self.pch_size, width)
            w_start = max(0, w_end - self.pch_size)
            
            self.width_starts_list.append(w_start)
            
            # Compute stride for next position based on current region complexity
            if self.adaptive_overlap and w_end < width:
                # Sample complexity in the current region
                local_comp = self._get_local_complexity(0, height, w_start, w_end)
                stride = compute_adaptive_stride(
                    self.pch_size, local_comp, self.min_overlap, self.max_overlap
                )
            else:
                stride = self.stride
            
            # Next start position
            if w_end >= width:
                break
            w = w + stride
        
        # Generate all combinations
        self.starts_list = []
        for h_start in self.height_starts_list:
            for w_start in self.width_starts_list:
                self.starts_list.append([h_start, w_start])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_starts = []
        for start in self.starts_list:
            start_tuple = tuple(start)
            if start_tuple not in seen:
                seen.add(start_tuple)
                unique_starts.append(start)
        self.starts_list = unique_starts
        
        # Update length
        self.length = len(self.starts_list)
    
    def get_weight(self, height, width, attention_map=None):
        """
        Get blending weight with optional attention guidance.
        
        Args:
            height, width: Dimensions of the patch
            attention_map: Optional attention map for the patch (1 x 1 x H x W)
        """
        if self.weight_type == 'ones':
            kernel = torch.ones(1, 1, height, width, 
                              dtype=self.dtype, device=self.im_ori.device)
        elif self.weight_type == 'Gaussian':
            kernel_h = self.generate_kernel_1d(height).reshape(-1, 1)
            kernel_w = self.generate_kernel_1d(width).reshape(1, -1)
            kernel = np.matmul(kernel_h, kernel_w)
            kernel = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
            kernel = kernel.to(dtype=self.dtype, device=self.im_ori.device)
        else:
            raise ValueError(f"Unsupported weight type: {self.weight_type}")
        
        # Apply attention guidance if provided
        if attention_map is not None:
            # Resize attention map to match kernel size
            if attention_map.shape[-2:] != (height, width):
                attention_map = F.interpolate(
                    attention_map, size=(height, width), 
                    mode='bilinear', align_corners=False
                )
            # Combine Gaussian weight with attention
            kernel = kernel * (0.7 + 0.3 * attention_map)  # 70% Gaussian, 30% attention
        
        return kernel
    
    def update(self, pch_res, index_infos, attention_maps=None):
        '''
        Update result with attention-guided blending.
        
        Input:
            pch_res: (n*extra_bs) x c x pch_size x pch_size, float
            index_infos: [(h_start, h_end, w_start, w_end),]
            attention_maps: Optional list of attention maps for each patch
        '''
        assert pch_res.shape[0] % self.true_bs == 0
        pch_list = torch.split(pch_res, self.true_bs, dim=0)
        assert len(pch_list) == len(index_infos)
        
        for ii, (h_start, h_end, w_start, w_end) in enumerate(index_infos):
            current_pch = pch_list[ii].type(self.dtype)
            
            # Get attention map for this patch if available
            attn_map = None
            if attention_maps is not None and ii < len(attention_maps):
                attn_map = attention_maps[ii]
            
            # Get adaptive weight
            current_weight = self.get_weight(
                current_pch.shape[-2], 
                current_pch.shape[-1],
                attention_map=attn_map
            )
            
            self.im_res[:, :, h_start:h_end, w_start:w_end] += current_pch * current_weight
            self.pixel_count[:, :, h_start:h_end, w_start:w_end] += current_weight

