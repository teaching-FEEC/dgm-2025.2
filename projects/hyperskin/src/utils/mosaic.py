import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image


def plot_dataset_mosaic(dataset, m: int, n: int, save_path: str, nrow: int = 4):
    """
    Plots a mosaic of samples from a PyTorch dataset and saves it as an image.

    Args:
        dataset (torch.utils.data.Dataset): Dataset returning (image, label)
        m (int): Starting index (inclusive)
        n (int): Ending index (exclusive)
        save_path (str): File path where to save the generated image
        nrow (int): Number of images per row in the mosaic grid (default=4)
    """
    # Get samples in the range [m, n)
    images, labels = [], []
    for i in range(m, n):
        x, y = dataset[i]

        # if x has more than 3 channels, get the mean across channels and repeat to make it 3 channels
        if x.shape[0] > 3:
            x = x.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        
        images.append(x)
        labels.append(y)

    # Stack all sample images into a grid
    grid = make_grid(images, nrow=nrow, padding=2)

    # Convert to PIL image for saving
    pil_img = to_pil_image(grid)

    # Plot (optional)
    plt.figure(figsize=(12, 8))
    plt.imshow(pil_img)
    plt.axis("off")

    # Display labels above the figure
    plt.title(f"Samples {m} to {n-1}")

    # Save mosaic image
    pil_img.save(save_path)
    print(f"✅ Mosaic saved to {save_path}")