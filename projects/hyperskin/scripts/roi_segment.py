import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage import segmentation
from skimage.segmentation import flood_fill
from scipy.ndimage import binary_dilation
import cv2
from skimage import morphology, measure

import gradio as gr

import pyrootutils
from pathlib import Path

def get_border_mask(image, iterations=20, threshold=0.1):
    """
    Compute a border mask by flood-filling the corners, then dilating.
    """
    mean_img = image.mean(axis=-1)
    image_mask = (mean_img < threshold).astype(np.uint8)  # threshold for dark/empty borders
    filled_mask = image_mask.copy()

    h, w = image_mask.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]

    # Flood fill from corners
    for c in corners:
        # if filled_mask[c] == 1:
            filled_mask = flood_fill(filled_mask, c, 2)

    border_mask = filled_mask == 2
    dilated_mask = binary_dilation(border_mask, iterations=iterations)

    return dilated_mask


# Point to your repo root manually
root = pyrootutils.setup_root(
    Path(__file__).parent,
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=False,
)

# print current working directory
print(f"Current working directory: {Path.cwd()}")

from src.data_modules.hsi_dermoscopy import HSIDermoscopyDataModule  # noqa: E402
from src.data_modules.datasets.hsi_dermoscopy import HSIDermoscopyTask  # noqa: E402

# ------------------- Data Loading -------------------
data_module = HSIDermoscopyDataModule(
    data_dir="data/hsi_dermoscopy",
    task=HSIDermoscopyTask.CLASSIFICATION_ALL_CLASSES,
    allowed_labels=["melanoma"],
    train_val_test_split=[.70, .15, .15],
    batch_size=16
)
data_module.prepare_data()
data_module.setup()
data_loader = data_module.all_dataloader()
dataset = data_loader.dataset


# ------------------- Segmentation Functions -------------------
def preprocess_image(L_GRAY):
    return ndi.median_filter(L_GRAY, size=(3, 3, 3))


def ske_segmentation(L_C, L_PAT, p_r=0.1, binarization_threshold=0.9):
    L_PAT_reshaped = L_PAT.reshape(1, 1, -1)
    L_D = (L_C - L_PAT_reshaped) / (L_PAT_reshaped + 1e-8)
    L_BR = np.where(np.abs(L_D) < p_r, 1, 0)
    L_R = np.sum(L_BR, axis=2)
    L_V = L_R / (np.max(L_R) + 1e-8)
    return L_V > binarization_threshold


def s3d_segmentation(data_cube, roi_mask, thresholds=(0.3, 0.2, 0.1)):
    """
    Implements the 3D segmentation method (S3D) from Koprowski & Olczyk (2016).

    Args:
        data_cube: numpy array of shape (M, N, I), the hyperspectral cube
        roi_mask: numpy array of shape (M, N), binary mask selecting ROI for reference spectrum
        thresholds: tuple of thresholds (pr1, pr2, pr3) as percentages

    Returns:
        segmented_image: numpy array (M, N, 3) segmentation mask with RGB coding
    """
    M, N, I = data_cube.shape

    # 1. Compute reference spectrum L_PAT(i)
    roi_pixels = data_cube[roi_mask > 0]  # shape: num_pixels x I
    L_PAT = roi_pixels.mean(axis=0)

    # 2. Difference cube: LV(m,n,i)
    LV = data_cube - L_PAT

    # 3. Apply 3D median filtering (spatial + spectral)
    LV_filtered = ndi.median_filter(LV, size=(3, 3, 3))

    # 4. Compute normalized similarity measure per pixel
    # Euclidean distance in spectral space
    distances = np.linalg.norm(LV_filtered, axis=2)
    distances = distances / distances.max()

    # 5. Thresholding to create segmentation layers
    segmented_image = np.zeros((M, N, 3))  # RGB encoding

    for channel, pr in enumerate(thresholds):
        mask = distances < pr
        segmented_image[..., channel] = mask.astype(float)

    return segmented_image, distances

def sh_segmentation(L_C, L_PAT, n_clusters=2):
    m_dim, n_dim, i_dim = L_C.shape
    small_dims = (m_dim // 8, n_dim // 8)
    L_C_small = resize(L_C, (small_dims[0], small_dims[1], i_dim), anti_aliasing=True)

    peaks, _ = find_peaks(L_PAT, prominence=np.std(L_PAT)/5)
    if len(peaks) < 3:
        peaks = np.argsort(L_PAT)[-3:]
    else:
        peaks = peaks[:3]

    num_pixels_small = small_dims[0] * small_dims[1]
    feature_matrix = np.zeros((num_pixels_small, len(peaks)))
    flat_L_C_small = L_C_small.reshape(num_pixels_small, i_dim)
    for i, peak_idx in enumerate(peaks):
        feature_matrix[:, i] = np.abs(flat_L_C_small[:, peak_idx] - L_PAT[peak_idx])

    feature_matrix_scaled = StandardScaler().fit_transform(feature_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

    # if feature matrix contains NaN or Inf, replace them with 0
    feature_matrix_scaled = np.nan_to_num(feature_matrix_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    labels = kmeans.fit_predict(feature_matrix_scaled)

    segmentation_map = resize(
        labels.reshape(small_dims),
        (m_dim, n_dim),
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )
    return segmentation_map.astype(int)


# ------------------- Visualization Helper -------------------
def visualize_results(image, roi_coords, ske_mask, s3d_mask, sh_mask, L_PAT, alpha=0.4):
    rgb_bands = [image.shape[2]-1, image.shape[2]//2, 0]
    rgb_image = image[:, :, rgb_bands]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    plt.close('all')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Segmentation Results with Overlays", fontsize=16)

    # Original with ROI
    rect = plt.Rectangle(
        (roi_coords["n_start"], roi_coords["m_start"]),
        roi_coords["n_end"] - roi_coords["n_start"],
        roi_coords["m_end"] - roi_coords["m_start"],
        edgecolor='red', facecolor='none', lw=2
    )
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title("Original (RGB) with ROI")
    axes[0, 0].axis('off')

    # ROI Spectrum
    axes[0, 1].plot(L_PAT)
    axes[0, 1].set_title("Reference Spectrum (L_PAT)")
    axes[0, 1].grid(True)

    # --- Overlay segmentation masks ---
    # SKE mask overlay
    axes[0, 2].imshow(rgb_image)
    axes[0, 2].imshow(ske_mask, cmap="jet", alpha=alpha)
    axes[0, 2].set_title("SKE Segmentation Overlay")
    axes[0, 2].axis("off")

    # S3D mask overlay
    axes[1, 0].imshow(rgb_image)
    axes[1, 0].imshow(s3d_mask, cmap="jet", alpha=alpha)
    axes[1, 0].set_title("S3D Segmentation Overlay")
    axes[1, 0].axis("off")

    # SH mask overlay
    axes[1, 1].imshow(rgb_image)
    axes[1, 1].imshow(sh_mask, cmap="jet", alpha=alpha)
    axes[1, 1].set_title("SH Segmentation Overlay")
    axes[1, 1].axis("off")

    # Empty plot
    axes[1, 2].axis('off')

    plt.tight_layout()
    return fig

def roi_from_edits(edits):
    if not edits or "layers" not in edits or len(edits["layers"]) == 0:
        return None

    layer = edits["layers"][0]   # RGBA mask of ROI drawn
    alpha = layer[..., 3]        # alpha channel
    ys, xs = np.where(alpha > 0) # pixels marked

    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return dict(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

def get_ellipse_mask(
    image,
    ellipse_shrink_factor=0.8
):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 1: threshold
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Step 2: all contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    # Step 3: largest contour
    largest = max(contours, key=cv2.contourArea)
    largest_img = contour_img.copy()
    cv2.drawContours(largest_img, [largest], -1, (0, 0, 255), 2)

    # Step 4: fitted ellipse
    ellipse = cv2.fitEllipse(largest)
    mask_orig = np.zeros_like(gray)
    cv2.ellipse(mask_orig, ellipse, 255, -1)

    ellipse_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(ellipse_img, ellipse, (255, 0, 0), 2)

    # Step 5: shrink ellipse
    (x, y), (MA, ma), angle = ellipse
    MA_shrunk, ma_shrunk = MA * ellipse_shrink_factor, ma * ellipse_shrink_factor
    ellipse_shrunk = ((x, y), (MA_shrunk, ma_shrunk), angle)

    return ellipse_shrunk

def crop_inside_ellipse(
    image,
    ellipse_shrink_factor=0.8,
    square_shrink_factor=1.0,
    method="square",  # "square" or "bbox"
):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Get ellipse
    ellipse_shrunk = get_ellipse_mask(image)
    (x, y), (MA_shrunk, ma_shrunk), angle = ellipse_shrunk

    # Draw ellipse
    ellipse_bbox_img = image.copy()
    cv2.ellipse(ellipse_bbox_img, ellipse_shrunk, (255, 0, 0), 2)

    # Step 6: crop depending on method
    if method == "square":
        a = MA_shrunk / 2.0
        b = ma_shrunk / 2.0
        s = int(min(np.sqrt(2) * a, np.sqrt(2) * b) * square_shrink_factor)

        x_min = int(x - s / 2)
        x_max = int(x + s / 2)
        y_min = int(y - s / 2)
        y_max = int(y + s / 2)

        cv2.rectangle(ellipse_bbox_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cropped = image[y_min:y_max, x_min:x_max]

    elif method == "bbox":
        mask_shrunk = np.zeros_like(gray)
        cv2.ellipse(mask_shrunk, ellipse_shrunk, 255, -1)

        ys, xs = np.where(mask_shrunk == 255)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min) * square_shrink_factor
        h = (y_max - y_min) * square_shrink_factor

        x_min = int(cx - w / 2)
        x_max = int(cx + w / 2)
        y_min = int(cy - h / 2)
        y_max = int(cy + h / 2)

        cv2.rectangle(ellipse_bbox_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cropped = image[y_min:y_max, x_min:x_max]

    else:
        raise ValueError("method must be 'square' or 'bbox'")

    return cropped


def rasterize_ellipse_mask(image, ellipse, shrink_factor=0.8):
    """Turn fitted ellipse into a boolean mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = np.zeros_like(gray, dtype=np.uint8)

    (x, y), (MA, ma), angle = ellipse
    MA_shrunk, ma_shrunk = MA * shrink_factor, ma * shrink_factor
    ellipse_shrunk = ((x, y), (MA_shrunk, ma_shrunk), angle)

    cv2.ellipse(mask, ellipse_shrunk, 255, -1)

    return mask.astype(bool)


def apply_mask_to_hsi(hsi_image, ellipse_mask):
    """
    Apply ellipse mask to hyperspectral cube.
    - hsi_image: [H, W, B]
    - ellipse_mask: [H, W] boolean
    """
    masked = hsi_image.copy()
    masked[~ellipse_mask] = 0.0  # could also use np.nan
    return masked

def preprocess_sample(image, channels=(0, 10, 12)):
    """Select channels, apply median filter, and rescale to 8-bit."""
    image_sel = image[..., list(channels)]
    image_med = ndi.median_filter(image_sel, size=(3, 3, 3))
    image_rescaled = (
        (image_med - image_med.min()) / (image_med.max() - image_med.min()) * 255
    ).astype(np.uint8)
    return image_rescaled

def s3d_get_ellipse_mask(
    image,
    ellipse_shrink_factor=0.8
):
    gray = cv2.cvtColor(preprocess_sample(image), cv2.COLOR_RGB2GRAY)
    # Step 1: threshold
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Step 2: all contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    # Step 3: largest contour
    largest = max(contours, key=cv2.contourArea)
    largest_img = contour_img.copy()
    cv2.drawContours(largest_img, [largest], -1, (0, 0, 255), 2)

    # Step 4: fitted ellipse
    ellipse = cv2.fitEllipse(largest)
    mask_orig = np.zeros_like(gray)
    cv2.ellipse(mask_orig, ellipse, 255, -1)

    ellipse_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(ellipse_img, ellipse, (255, 0, 0), 2)

    # Step 5: shrink ellipse
    (x, y), (MA, ma), angle = ellipse
    MA_shrunk, ma_shrunk = MA * ellipse_shrink_factor, ma * ellipse_shrink_factor
    ellipse_shrunk = ((x, y), (MA_shrunk, ma_shrunk), angle)

    mask_shrunk = np.zeros_like(gray)
    cv2.ellipse(mask_shrunk, ellipse_shrunk, 255, -1)
    return mask_shrunk


with gr.Blocks() as demo:
    gr.Markdown("## HSI Dermoscopy Segmentation Tool")

    with gr.Row():
        idx_input = gr.Number(
            label="Image Index",
            value=-1,
            interactive=True
        )
        random_checkbox = gr.Checkbox(label="Use Random Image", value=True)

    # New row for load button (so it appears below the checkbox)
    with gr.Row():
        load_button = gr.Button("🔀 Load Random Sample")

    brush = gr.Brush(default_size=5)
    image_editor = gr.ImageEditor(
        type="numpy",
        label="Draw ROI on Image",
        canvas_size=(512, 272),
        fixed_canvas=True,
        brush=brush,          # Enable brush
        eraser=True,         # Hide eraser
        layers=False,         # Hide layer tool
        transforms=None,       # Disable crop/resize
        sources=None
    )
    run_button = gr.Button("Run Segmentation")

    output_plot = gr.Plot()
    selected_idx = gr.State(-1)

    def process_with_state(use_random, edits, idx_sel):
        """Run segmentation using the already stored dataset index."""
        image, label = dataset[idx_sel]

        # if image.ndim == 2:
        #     image = image[..., None]

        # --- Generate ellipse mask on RGB composite ---
        rgb_bands = [image.shape[2] - 1, image.shape[2] // 2, 0]
        rgb_image = image[:, :, rgb_bands]
        rgb_norm = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        rgb_uint8 = (rgb_norm * 255).astype(np.uint8)

        ellipse = get_ellipse_mask(rgb_uint8)
        ellipse_mask = rasterize_ellipse_mask(rgb_uint8, ellipse)

        roi = roi_from_edits(edits)
        if roi is None:
            raise ValueError("No ROI drawn. Please draw a ROI on the image.")

        x, y, w, h = roi.get("x"), roi.get("y"), roi.get("width"), roi.get("height")
        roi_slice = (slice(y, y+h), slice(x, x+w))

        roi_mask = np.zeros(image.shape[:2], dtype=bool)
        roi_mask[roi_slice] = True

        s3d_image, dist = s3d_segmentation(image, roi_mask)
        ellipse_mask = s3d_get_ellipse_mask(image, ellipse_shrink_factor=0.9)
        s3d_mask = ~(dist > 0.3)
        s3d_mask[ellipse_mask == 0] = 0

        s3d_mask = morphology.remove_small_holes(s3d_mask, area_threshold=500)

        # get largest connected component of seg_img channel 0 as final mask
        labeled = measure.label(s3d_mask)
        props = measure.regionprops(labeled)
        if props:
            largest = max(props, key=lambda r: r.area)
            s3d_mask = (labeled == largest.label).astype(np.uint8)

        # --- Apply mask to HSI cube ---
        image = apply_mask_to_hsi(image, ellipse_mask)

        # Now do preprocessing & segmentation
        image_p = preprocess_image(image)

        L_PAT = np.mean(image_p[roi_slice], axis=(0, 1))

        ske_mask = ske_segmentation(image_p, L_PAT, p_r=0.2, binarization_threshold=0.85)
        sh_map = sh_segmentation(image_p, L_PAT, n_clusters=2)

        roi_cluster_label = np.median(sh_map[roi_slice])
        sh_mask = (sh_map == roi_cluster_label)

        ske_mask[ellipse_mask == 0] = 0
        sh_mask[ellipse_mask == 0] = 0

        fig = visualize_results(
            image,
            {"m_start": y, "m_end": y+h, "n_start": x, "n_end": x+w},
            ske_mask, s3d_mask, sh_mask, L_PAT
        )
        return fig

    # --- Functions ---
    def pick_image(idx, use_random):
        """Pick image based on idx or random, return RGB & index."""
        if use_random or idx < 0 or idx >= len(dataset):
            idx_sel = np.random.randint(0, len(dataset))
        else:
            idx_sel = int(idx)
        # idx_sel = 13
        print(f"Selected image index: {idx_sel}")
        image, _ = dataset[idx_sel]
        rgb_bands = [image.shape[2]-1, image.shape[2]//2, 0]
        rgb_image = image[:, :, rgb_bands]
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
        return rgb_image, idx_sel

    # --- Bindings ---
    idx_input.change(
        fn=pick_image,
        inputs=[idx_input, random_checkbox],
        outputs=[image_editor, selected_idx],
    )
    random_checkbox.change(
        fn=pick_image,
        inputs=[idx_input, random_checkbox],
        outputs=[image_editor, selected_idx],
    )

    def toggle_idx_field(use_random):
        return gr.update(interactive=not use_random)
    random_checkbox.change(
        toggle_idx_field,
        inputs=random_checkbox,
        outputs=idx_input,
    )

    # 🔀 NEW button always forces random sample
    def pick_random_only():
        return pick_image(-1, True)

    load_button.click(
        fn=pick_random_only,
        inputs=[],
        outputs=[image_editor, selected_idx],
    )

    run_button.click(
        fn=process_with_state,
        inputs=[random_checkbox, image_editor, selected_idx],
        outputs=output_plot,
    )

    # --- Load first image on app start ---
    demo.load(
        fn=pick_image,
        inputs=[idx_input, random_checkbox],
        outputs=[image_editor, selected_idx],
    )

demo.launch()
