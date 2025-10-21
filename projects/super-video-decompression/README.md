# `Super Descompressão de Vídeo`
# `Super Video Decompression`

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

> |Name  | RA | Specialization|
> |--|--|--|
> | Fernando Barbosa Gomes  | 211503  | Computer Engineering|
> | Pedro Luis Rebollo  | 217460  | Computer Engineering|
> | Victor Manoel Rodrigues Guilherme  | 213307  | Computer Engineering|

## Importants Links

[Link to E1 presentation video](https://files.realmsapp.top/PresentationE1.mp4)

[Link to slideshow E1](https://docs.google.com/presentation/d/1TISrxtNkQHBbZlzTeAIGRFAGPHKGeUeE0Cg3LDZzKp4/edit?usp=sharing)

## Abstract
This project addresses Super Video Decompression, aiming to restore perceptual quality in lossy compressed video frames and enhance resolution. The work defined three core tasks: 1x decompression, 2x super-resolution, and 1x-to-2x decompression adequacy. We trained three model sizes: Super, Mega, and Ultra, all utilizing the compact architecture chosen for real-time feasibility and simplification for shaders. The methodology employed a custom dataset of 78:37 minutes of animation, significantly reduced via deduplication using average hashing and Hamming distance. Training incorporated specialized loss functions, including Canny Edge Loss, Patch Variance Loss, and Face Aware Loss, to optimize detail recovery.

## Problem Description / Motivation

The motivation for this project stems from the fact that nearly 80% of global internet bandwidth is consumed by video streaming. To facilitate the large-scale distribution required by major platforms (e.g., YouTube, Netflix, Hulu), most videos are compressed using lossy algorithms such as H.264/AVC. While effective for file size reduction, this approach inevitably leads to a decline in image and audio quality, a process that cannot be fully reversed using traditional methods.

The task of Super Video Decompression is related to Super-Resolution (SR) techniques, which generally aim to improve image quality, often applied to noisy photographs or medical imaging. However, typical SR models often upscale from high-data formats like Full HD or address noise/compression added synthetically (e.g., JPEG compression). The proposed project differs by addressing the specific challenges of lossy video compression, aiming to reduce or remove compression artifacts while simultaneously increasing the video's resolution. The goal is to develop a model capable of decompressing a sequence of images that has been compressed in both spatial (area) and temporal dimensions.

To achieve this goal, the project was structured to solve three specific technical problems: 

1. Decompression (1x): Input consists of compressed frames, and the output is the decompressed frame at 1x resolution.
2. Super Resolution (2x): Input consists of frames reduced in size (without compression artifacts), and the output is the frame at 2x resolution.
3. Decompression Adequacy for 2x: Input consists of "decompressed" frames at 1x resolution, and the output is the frame upscaled to 2x resolution.

## Objective


## Methodology

The methodology proposed for the Super Video Decompression project encompasses architectural selection, rigorous dataset preparation, specialized loss function utilization, and a comprehensive evaluation plan designed to ensure model robustness and real-time feasibility.

### Model Tasks and Architecture Justification
The project necessitated the training of specific models tailored to three distinct problem tasks:

1. Decompression (1x): Input consists of compressed frames (H.264/AVC), output is the restored image at 1x resolution.
2. Super Resolution (2x): Input consists of frames reduced in size (downscaled but without compression artifacts), output is the frame at 2x resolution.
3. Decompression Adequacy for 2X (ad2x): Input consists of frames that have been "decompressed" at 1x resolution, output is the frame upscaled to 2x resolution.
### Architectural Design

The compact architecture was selected as the foundational network design. This choice was justified by its ability to achieve real-time feasibility during operation and its structural simplicity, which facilitates the crucial step of converting the final model into shaders for optimized deployment.

For each of the three tasks, three distinct model sizes were trained: Super, Mega, and Ultra. This expanded division was necessary because the default smallest size, Super, was found to be too rigid and possess limited learning capacity. Consequently, the Mega model was introduced, defined as having precisely double the parameter count of the Super model while maintaining approximately one-third the parameter count of the Ultra model.

### Training Regimen and Loss Functions
The training methodology involved exploring advanced architectures. The network utilized convolutional sequential layers to maintain compatibility with varying input sizes.

Specialized loss functions were critical for optimizing detail recovery and artifact suppression:

1. Canny Edge Loss: This function measures the overlap of edges using Dice Loss (0.25%) and applies L1 (Charbonier Loss) exclusively in the regions identified as edges by a Differentiable Canny Edge Detector.
2. Patch Variance Loss: This function applies Mean Squared Error (MSE), scaled (1 for compressed image, 0 for original image), weighted by the formula 1−variance over 20-pixel patches. This mechanism effectively directs the loss function's attention to areas with high variation.
3. Face Aware Loss (FAL): This loss combines SSIM, Charbonier, and MSE metrics, calculated strictly within the regions identified as faces by an initial face detector. This attention mechanism requires the creation of specialized batches containing only facial patches.
4. Combined Patch Variance: A combination of SSIM, Charbonier, and MSE is measured, employing the same attention mechanism leveraged in the Patch Variance Loss.

### Evaluation Methodology
Evaluation will utilize both quantitative and perceptual metrics to determine if the objectives were met.

 Quantitative Metrics:
  
  PSNR (Peak Signal-to-Noise Ratio): Used to measure pixel-level differences between the generated output and the ground truth.
  
  SSIM (Structural Similarity Index): Used to evaluate the structural similarity and quality preservation.

Perceptual Metric:
    
  DISTS (Deep Image Structure and Texture Similarity): Used to provide a measure of perceptual quality based on deep feature representations.

The models will be benchmarked to confirm their real-time feasibility, a key project requirement. While preliminary performance estimations can be derived from existing compact model benchmarks (e.g.,), project-specific benchmarking remains an eventual necessity [4, 13:55]. A specific test will be conducted to determine if the ad2x model exhibits generalized superiority, allowing it to be used in place of the dedicated 2x model in production environments.


## Bibliographic References
BARAKA MAISELI; ABDALLA, A. T. Seven decades of image super-resolution: achievements, challenges, and opportunities. EURASIP Journal on Advances in Signal Processing, v. 2024, n. 1, 18 jul. 2024.

CHEN, Z. et al. NTIRE 2025 Challenge on Image Super-Resolution (x4): Methods and Results. Disponível em: <https://arxiv.org/abs/2504.14582>. Acesso em: 16 set. 2025.
