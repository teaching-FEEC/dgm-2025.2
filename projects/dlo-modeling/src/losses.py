import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedRopeLoss(nn.Module):
    """
    Composite loss for DLOs combining Position, Velocity, Stretching, Bending,
    Gaussian Overlap, AND Collision Repulsion.
    """
    def __init__(self, w_pos=1.0, w_vel=1.0, w_stretch=0.5, w_bend=0.1, w_overlap=0.0, overlap_sigma=0.1, w_collision=0.0, collision_dist=0.05):
        super().__init__()
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_stretch = w_stretch
        self.w_bend = w_bend
        self.w_overlap = w_overlap
        self.overlap_sigma = overlap_sigma
        self.w_collision = w_collision
        self.collision_dist = collision_dist

    def forward(self, pred, tgt, src):
        """
        Args:
            pred: Predicted states S_hat_t (B, T, L, 3)
            tgt: Target states S_t (B, T, L, 3)
            src: Previous states S_{t-1} (B, T, L, 3)
        """
        # 1. Position Loss (MSE)
        l_pos = F.mse_loss(pred, tgt)

        # 2. Velocity Loss (MSE of deltas)
        l_vel = F.mse_loss(pred - src, tgt - src)

        # 3. Stretch Loss (Inextensibility)
        link_vec_pred = pred[:, :, 1:] - pred[:, :, :-1]
        link_vec_tgt = tgt[:, :, 1:] - tgt[:, :, :-1]

        link_len_pred = torch.norm(link_vec_pred, dim=-1)
        link_len_tgt = torch.norm(link_vec_tgt, dim=-1)

        l_stretch = F.mse_loss(link_len_pred, link_len_tgt)

        # 4. Bending Loss (Curvature Consistency)
        curv_pred = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]
        curv_tgt = tgt[:, :, 2:] - 2 * tgt[:, :, 1:-1] + tgt[:, :, :-2]

        l_bend = F.mse_loss(curv_pred, curv_tgt)

        # 5. Gaussian Overlap Loss (Soft Dice)
        # dist_sq: squared Euclidean distance between corresponding points
        dist_sq = torch.sum((pred - tgt)**2, dim=-1) # (B, T, L)
        overlap = torch.exp(-dist_sq / (2 * self.overlap_sigma**2))
        l_overlap = 1.0 - overlap.mean()

        # 6. Collision Repulsion Loss (Self-Intersection)
        l_coll = torch.tensor(0.0, device=pred.device)

        if self.w_collision > 0:
            # Flatten B and T to process all frames: (N, L, 3)
            B, T, L, C = pred.shape
            flat_pred = pred.reshape(-1, L, 3)

            # Compute pairwise distance matrix: (N, L, L)
            dists = torch.cdist(flat_pred, flat_pred, p=2) # (N, L, L)

            # Create mask for non-adjacent nodes (ignore self and immediate neighbors)
            # We want indices where abs(i - j) > 1
            indices = torch.arange(L, device=pred.device)
            mask = (torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)) > 1).float() # (L, L)

            # Apply mask
            masked_dists = dists * mask.unsqueeze(0) + (1 - mask.unsqueeze(0)) * 1e6 # Push masked to infinity

            # Hinge loss: penalize if dist < collision_dist
            # We want dist > collision_dist -> collision_dist - dist < 0 -> ReLU gives 0
            # If dist < collision_dist -> collision_dist - dist > 0 -> Penalty
            loss_matrix = F.relu(self.collision_dist - masked_dists)

            # Mean over valid pairs (mask sum approx L*L)
            l_coll = loss_matrix.sum() / (B * T * L * L)

        # Weighted Sum
        total_loss = (self.w_pos * l_pos +
                      self.w_vel * l_vel +
                      self.w_stretch * l_stretch +
                      self.w_bend * l_bend +
                      self.w_overlap * l_overlap +
                      self.w_collision * l_coll)

        return total_loss

class RopeLoss(nn.Module):
    """
    Standard MSE on position + delta (velocity).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, tgt, src, *args, **kwargs):
        """
        Args:
            pred: (B, L, 3) Predicted state
            tgt:  (B, L, 3) Target state
            src:  (B, L, 3) Source state
            *args, **kwargs: Ignored (allows compatibility with loops passing action)
        """
        # Standard MSE on position
        mse_pos = F.mse_loss(pred, tgt)
        
        # Standard MSE on velocity (delta)
        mse_delta = F.mse_loss(pred - src, tgt - src)
        
        return mse_pos + mse_delta

class WeightedRopeLoss(nn.Module):
    """
    Loss function that applies higher penalty to the link that was acted upon,
    and decaying penalty for neighboring links.
    """
    def __init__(self, sigma=5.0, base_weight=1.0, boost_weight=5.0):
        super().__init__()
        self.sigma = sigma
        self.base_weight = base_weight
        self.boost_weight = boost_weight

    def forward(self, pred, tgt, src, action):
        """
        Args:
            pred:   (B, L, 3) Predicted state
            tgt:    (B, L, 3) Target state
            src:    (B, L, 3) Source state
            action: (B, 4) or (B, L, 4) Action tensor containing link index info
        """
        B, L, _ = pred.shape
        device = pred.device
        
        # 1. Identify the index of the acted link for each batch item
        if action.dim() == 2: 
            # Dense (B, 4) -> [dx, dy, dz, link_id]
            link_indices = action[:, 3].long().clamp(0, L - 1)
        else: 
            # Sparse (B, L, 4) - Assumes index 3 is the 'active' flag or feature
            # We find the index L where this feature is maximized
            link_indices = torch.argmax(action[:, :, 3], dim=1)
            
        # 2. Create a spatial weight mask (B, L)
        # Create a range [0, 1, ..., L-1]
        seq_indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L) # (B, L)
        link_indices_exp = link_indices.unsqueeze(1).expand(B, L) # (B, L)
        
        # Calculate distance from the acted link |i - center|
        dist = torch.abs(seq_indices - link_indices_exp).float()
        
        # Create Gaussian-like weights
        gaussian_weights = torch.exp(- (dist**2) / (2 * self.sigma**2))
        
        # Apply Base Weight + Gaussian Boost
        final_weights = self.base_weight + (self.boost_weight * gaussian_weights) # (B, L)
        
        # Expand weights to (B, L, 1) for broadcasting against (x,y,z)
        final_weights = final_weights.unsqueeze(-1)
        
        # 3. Calculate Squared Errors (Unreduced)
        pos_error = (pred - tgt) ** 2
        delta_error = ((pred - src) - (tgt - src)) ** 2
        
        # 4. Apply Weights
        weighted_pos_error = pos_error * final_weights
        weighted_delta_error = delta_error * final_weights
        
        # 5. Mean reduction
        return weighted_pos_error.mean() + weighted_delta_error.mean()