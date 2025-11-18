from skimage.transform import resize
import numpy as np

def smallest_maxsize_and_centercrop(
    img: np.ndarray, image_size: int
) -> np.ndarray:
    """
    Mimics:
        A.SmallestMaxSize(max_size=image_size),
        A.CenterCrop(height=image_size, width=image_size)


    Args:
        img (np.ndarray): Input image, shape (H, W, C).
        image_size (int): Target output dimension (square).

    Returns:
        np.ndarray: Transformed image of shape (image_size, image_size, C).
    """
    h, w, _ = img.shape

    scale = image_size / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # skimage can resize N-channel images directly
    resized = resize(
        img,
        (new_h, new_w, img.shape[-1]),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )

    # Center crop
    start_y = (new_h - image_size) // 2
    start_x = (new_w - image_size) // 2
    cropped = resized[start_y:start_y + image_size, start_x:start_x + image_size]
    return cropped
