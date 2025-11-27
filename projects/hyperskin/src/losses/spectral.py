import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConsistencyLoss(nn.Module):
    """
    Computes spectral consistency using:
    1. Stable SAM (1 - Cosine Similarity)
    2. Spectral Gradient loss (Log-Cosh or L1)
    
    Expects inputs to be roughly in [-1, 1] (tanh output), 
    which it normalizes to [0, 1] internally.
    """
    def __init__(
        self, 
        sam_weight: float = 20.0, 
        grad_weight: float = 20.0, 
        grad_loss_type: str = "logcosh"  # Options: 'l1', 'mse', 'logcosh'
    ):
        super().__init__()
        self.sam_weight = sam_weight
        self.grad_weight = grad_weight
        self.grad_loss_type = grad_loss_type

    def _log_cosh(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Log-Cosh loss: approximating |x - y| + log(2) for large diffs, 
        and (x - y)^2 / 2 for small diffs.
        """
        diff = x - y
        # torch.cosh can overflow for large values, but spectral values [0,1] are safe.
        # We add a tiny epsilon to log to ensure numerical stability.
        return torch.mean(torch.log(torch.cosh(diff) + 1e-12))

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        # 1. Normalize inputs from [-1, 1] to [0, 1] for physical validity
        preds_norm = (preds + 1) * 0.5
        target_norm = (target + 1) * 0.5
        
        total_loss = 0.0
        
        # 2. Stable SAM (Cosine Similarity) Loss
        if self.sam_weight > 0:
            # Dot product along channel dimension (dim 1)
            dot_product = (preds_norm * target_norm).sum(dim=1)
            preds_len = preds_norm.norm(dim=1)
            target_len = target_norm.norm(dim=1)
            
            # Avoid division by zero
            denominator = preds_len * target_len + 1e-8
            cos_sim = dot_product / denominator
            
            # instead of acos(cos_sim), we minimize (1 - cos_sim).
            # Range: 0 (perfect alignment) to 2 (opposite direction)
            sam_proxy_loss = (1.0 - cos_sim).mean()
            total_loss += self.sam_weight * sam_proxy_loss

        # 3. Spectral Gradient (Derivative) Loss
        if self.grad_weight > 0:
            # Compute differences between adjacent spectral bands
            # Represents the "slope" or shape of the curve
            diff_preds = preds_norm[:, 1:, :, :] - preds_norm[:, :-1, :, :]
            diff_target = target_norm[:, 1:, :, :] - target_norm[:, :-1, :, :]
            
            if self.grad_loss_type == 'l1':
                grad_loss = F.l1_loss(diff_preds, diff_target)
            elif self.grad_loss_type == 'mse':
                grad_loss = F.mse_loss(diff_preds, diff_target)
            elif self.grad_loss_type == 'logcosh':
                grad_loss = self._log_cosh(diff_preds, diff_target)
            else:
                raise ValueError(f"Unknown grad_loss_type: {self.grad_loss_type}")
                
            total_loss += self.grad_weight * grad_loss
            
        return total_loss