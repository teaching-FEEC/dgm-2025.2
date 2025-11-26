'''
# --------------------------------------------------------------------------------
#   Color fixed script from Li Yi (https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
# --------------------------------------------------------------------------------
'''

import torch
import numpy as np
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from torchvision.transforms import ToTensor, ToPILImage

from .util_image import rgb2ycbcrTorch, ycbcr2rgbTorch

def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq

def ycbcr_color_replace(content_feat:Tensor, style_feat:Tensor):
    """
    Apply ycbcr decomposition, so that the content will have the same color as the style.
    """
    content_y = rgb2ycbcrTorch(content_feat, only_y=True)
    style_ycbcr = rgb2ycbcrTorch(style_feat, only_y=False)

    target_ycbcr = torch.cat([content_y, style_ycbcr[:, 1:,]], dim=1)

    target_rgb = ycbcr2rgbTorch(target_ycbcr)

    return target_rgb

def histogram_matching_torch(content_feat: Tensor, style_feat: Tensor, n_bins: int = 256):
    """
    Apply histogram matching to match the color distribution of content_feat to style_feat.
    This is a memory-efficient implementation that works directly with PyTorch tensors.
    
    Args:
        content_feat: B x C x H x W tensor, the content image to be transformed
        style_feat: B x C x H x W tensor, the reference style image
        n_bins: Number of bins for histogram computation (default: 256)
    
    Returns:
        matched: B x C x H x W tensor with matched histogram
    """
    device = content_feat.device
    dtype = content_feat.dtype
    b, c, h, w = content_feat.shape
    
    # Normalize to [0, 1] if needed
    content_norm = content_feat.clamp(0.0, 1.0)
    style_norm = style_feat.clamp(0.0, 1.0)
    
    matched = torch.zeros_like(content_norm)
    
    for channel in range(c):
        content_channel = content_norm[:, channel, :, :].reshape(b, -1)  # B x (H*W)
        style_channel = style_norm[:, channel, :, :].reshape(b, -1)  # B x (H*W)
        
        for batch_idx in range(b):
            content_flat = content_channel[batch_idx].cpu().numpy()  # H*W
            style_flat = style_channel[batch_idx].cpu().numpy()  # H*W
            
            # Compute histograms
            content_hist, content_bins = np.histogram(content_flat, bins=n_bins, range=(0.0, 1.0))
            style_hist, style_bins = np.histogram(style_flat, bins=n_bins, range=(0.0, 1.0))
            
            # Avoid division by zero
            content_hist = content_hist.astype(np.float32) + 1e-7
            style_hist = style_hist.astype(np.float32) + 1e-7
            
            # Compute cumulative distribution functions
            content_cdf = np.cumsum(content_hist)
            content_cdf = content_cdf / content_cdf[-1]  # Normalize
            
            style_cdf = np.cumsum(style_hist)
            style_cdf = style_cdf / style_cdf[-1]  # Normalize
            
            # Compute bin centers
            content_bin_centers = (content_bins[:-1] + content_bins[1:]) / 2.0
            style_bin_centers = (style_bins[:-1] + style_bins[1:]) / 2.0
            
            # Map each pixel value using vectorized interpolation
            # For each pixel in content, find its CDF value, then find matching value in style
            # Use interpolation to map from content CDF to style values
            # First, compute CDF values for each content pixel
            content_cdf_values = np.interp(content_flat, content_bin_centers, content_cdf)
            # Then, map from style CDF to style bin centers (inverse CDF mapping)
            matched_flat = np.interp(content_cdf_values, style_cdf, style_bin_centers)
            
            matched[batch_idx, channel, :, :] = torch.from_numpy(matched_flat.reshape(h, w)).to(device, dtype)
    
    return matched.clamp(0.0, 1.0)

def adaptive_histogram_matching(content_feat: Tensor, style_feat: Tensor, blend_ratio: float = 0.7):
    """
    Adaptive histogram matching with blending for better color preservation.
    Combines histogram matching with original image to preserve details.
    
    Args:
        content_feat: B x C x H x W tensor, the content image
        style_feat: B x C x H x W tensor, the reference style image
        blend_ratio: Blending ratio between matched and original (0.0-1.0)
    
    Returns:
        matched: B x C x H x W tensor with adaptively matched histogram
    """
    matched = histogram_matching_torch(content_feat, style_feat)
    # Blend with original to preserve details
    result = blend_ratio * matched + (1.0 - blend_ratio) * content_feat
    return result.clamp(0.0, 1.0)

def hybrid_color_fix(content_feat: Tensor, style_feat: Tensor, method: str = 'adaptive', 
                     blend_ratio: float = 0.7, ycbcr_weight: float = 0.3, 
                     wavelet_weight: float = 0.3, hist_weight: float = 0.4):
    """
    Hybrid color fixing method that combines multiple techniques for optimal results.
    This method adaptively combines YCbCr, Wavelet, and Histogram Matching.
    
    Args:
        content_feat: B x C x H x W tensor, the content image (SR result)
        style_feat: B x C x H x W tensor, the reference image (LQ upsampled)
        method: 'adaptive' (weighted combination) or 'best' (selects best method)
        blend_ratio: Blending ratio for histogram matching (0.0-1.0)
        ycbcr_weight: Weight for YCbCr method in adaptive mode
        wavelet_weight: Weight for Wavelet method in adaptive mode
        hist_weight: Weight for Histogram Matching in adaptive mode
    
    Returns:
        result: B x C x H x W tensor with improved color matching
    """
    if method == 'adaptive':
        # Weighted combination of all methods
        ycbcr_result = ycbcr_color_replace(content_feat, style_feat)
        wavelet_result = wavelet_reconstruction(content_feat, style_feat)
        hist_result = adaptive_histogram_matching(content_feat, style_feat, blend_ratio)
        
        # Normalize weights
        total_weight = ycbcr_weight + wavelet_weight + hist_weight
        ycbcr_weight = ycbcr_weight / total_weight
        wavelet_weight = wavelet_weight / total_weight
        hist_weight = hist_weight / total_weight
        
        # Combine results
        result = (ycbcr_weight * ycbcr_result + 
                 wavelet_weight * wavelet_result + 
                 hist_weight * hist_result)
        
        return result.clamp(0.0, 1.0)
    
    elif method == 'best':
        # Select best method based on color difference
        ycbcr_result = ycbcr_color_replace(content_feat, style_feat)
        wavelet_result = wavelet_reconstruction(content_feat, style_feat)
        hist_result = adaptive_histogram_matching(content_feat, style_feat, blend_ratio)
        
        # Compute color difference (mean squared error in YCbCr space)
        def compute_color_diff(result, reference):
            result_ycbcr = rgb2ycbcrTorch(result, only_y=False)
            ref_ycbcr = rgb2ycbcrTorch(reference, only_y=False)
            diff = torch.mean((result_ycbcr - ref_ycbcr) ** 2, dim=(2, 3))  # B x C
            return torch.mean(diff, dim=1)  # B
        
        ycbcr_diff = compute_color_diff(ycbcr_result, style_feat)
        wavelet_diff = compute_color_diff(wavelet_result, style_feat)
        hist_diff = compute_color_diff(hist_result, style_feat)
        
        # Select method with minimum color difference for each batch
        results = torch.stack([ycbcr_result, wavelet_result, hist_result], dim=1)  # B x 3 x C x H x W
        diffs = torch.stack([ycbcr_diff, wavelet_diff, hist_diff], dim=1)  # B x 3
        best_indices = torch.argmin(diffs, dim=1)  # B
        
        # Select best result for each batch
        b, _, c, h, w = results.shape
        result = torch.zeros(b, c, h, w, device=content_feat.device, dtype=content_feat.dtype)
        for i in range(b):
            result[i] = results[i, best_indices[i]]
        
        return result.clamp(0.0, 1.0)
    
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'adaptive' or 'best'.")


