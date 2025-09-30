
from CannyEdgeDetectorModel import DifferentiableCanny
import torch
import torch.nn as nn
import torch.nn.functional as F

class CannyEdgeLoss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, low_threshold=0.09, high_threshold=0.26, mask_threshold=0.5, loss_type="l1"):
        super().__init__()
        self.canny = DifferentiableCanny(kernel_size, sigma, low_threshold, high_threshold)
        self.mask_threshold = mask_threshold
        if loss_type == "l1":
            self.criterion = F.l1_loss
        elif loss_type == "mse":
            self.criterion = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss_type {loss_type}, use 'l1' or 'mse'.")

    def forward(self, pred, target):
        """
        pred:   [B,3,H,W] generated RGB image in [0,1]
        target: [B,3,H,W] target RGB image in [0,1]
        """
        # Compute differentiable canny edges
        edges_pred = self.canny(pred)    # [B,3,H,W]
        edges_gt   = self.canny(target)  # [B,3,H,W]

        # Create relevance mask (OR between edges)
        mask = ((edges_pred > self.mask_threshold) | (edges_gt > self.mask_threshold)).float()  # [B,3,H,W]

        # Difference
        edges_pred = edges_pred * mask
        edges_gt = edges_gt * mask
        
        diff = (edges_pred - edges_gt) * mask

        # Avoid division by zero (e.g. completely flat image)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Masked reduction
        if self.criterion == F.l1_loss:
            loss = diff.abs().sum() / mask.sum()
        else:
            loss = (diff ** 2).sum() / mask.sum()

        return loss, mask
    
    
    import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt

# Assume DifferentiableCanny and CannyEdgeLoss are already defined above
from PIL import Image
import torchvision.transforms as T

def load_image(path, size):
    # Open image, resize, convert to tensor in [0,1]
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),  # Converts to [C,H,W] float32 in [0,1]
    ])
    return transform(img).unsqueeze(0) 

def test_canny_loss():
    # Create two toy RGB images [B,3,H,W]
    '''
    B, C, H, W = 1, 3, 64, 64
    img1 = torch.zeros((B, C, H, W))
    img2 = torch.zeros((B, C, H, W))

    # Draw a white square in img1
    img1[:, :, 16:48, 16:48] = 1.0
    # Draw a slightly shifted square in img2
    img2[:, :, 20:52, 20:52] = 1.0
    '''   
    img1 = load_image('./frame_00256l.webp', size=(202,480))
    img2 = load_image('./frame_00256h.webp', size=(202,480))
    
    #plt.imshow(img1.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    #plt.imshow(img2.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    #plt.show()
    canny = DifferentiableCanny(5, 1, 0.1, 0.3)
    pred = canny(img1)
    plt.imshow(pred.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    plt.show()
    pred = canny(img2)
    plt.imshow(pred.permute(0, 2, 3, 1)[0,:,:,:], cmap="gray")
    plt.show()
    
    # Instantiate loss
    loss_fn = CannyEdgeLoss(loss_type="l1")

    # Forward pass
    loss, mask = loss_fn(img1, img2)

    # Recompute edges + mask just for visualization
    edges_pred = loss_fn.canny(img1)
    edges_gt   = loss_fn.canny(img2)
    #mask = ((edges_pred > 0.1) | (edges_gt > 0.1)).float()

    print("Loss value:", loss.item())
    print("Mask shape:", mask.shape)

    # Convert mask to numpy for plotting (take first batch, first channel)
    mask_np = mask[0,0].detach().cpu().numpy()

    plt.imshow(mask_np, cmap="gray")
    plt.title("Canny OR Mask")
    plt.axis("off")
    plt.show()
    
    
test_canny_loss()