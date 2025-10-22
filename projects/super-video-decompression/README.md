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

[Link to slideshow E2](https://docs.google.com/presentation/d/1ml3qdwYOR1TSd1Hl9HbC9HwfaJqkPAOex2bFLj-FGyg/edit?usp=sharing)


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

The methodology proposed for the Super Video Decompression project encompasses, training framework selection, architectural selection, dataset preparation, novel loss function creation and experimentation and a evaluation plan designed validate models according to fidelity and perceptual quality, finally we will also benchmark model combinations for real time feasibility.

### Training Framework

We used NeoSR (https://github.com/neosr-project/neosr) as the training framework for our Super Video Decompression problem. It provides implementations of state of the art models from competitions and the literature, as well as losses, metrics and tools to help train a model for image related tasks. 

It is has some rigidity to it and we had to rewrite portions of it to introduce our own losses, specially the face aware loss which required a big rewrite. It also seems to be calculating SSIM wrong as a performance metric( output 1.8 instead of the literature range of -1 to 1)

NeoSr was choosen to improve our iteration time and because there is a lot of pretrain models prepared for it.

### Model Tasks and Architecture Justification

The project required the training of specific models tailored to three distinct problem tasks:

1. Decompression (1x): Input consists of compressed frames (H.264/AVC), output is the restored image at 1x resolution.
2. Super Resolution (2x): Input consists of frames reduced in size (downscaled but without compression artifacts), output is the frame at 2x resolution.
3. Decompression Adequacy to 2X (ad2x): Input consists of frames that have been time "decompressed" at 1x resolution, output is the frame upscaled to 2x resolution.

### Architectural Design

The compact architecture (a VGG style rectangular convolutional network) was selected as the foundational network architecture. This choice was justified by the combination of its variations  achieving real-time feasibility during operation and its structural simplicity, which facilitates the crucial step of converting the final model into shaders for optimized use in production.

For each of the three tasks, three distinct model sizes were trained: Super(47k parameters), Mega(93k parameters), and Ultra(280k parameters). 

The default smallest size, Super, was found to be too rigid and possess limited learning capacity. For that reason we created the custom sized Mega model, defined as having around double the parameter count of the Super model while maintaining approximately one-third the parameter count of the Ultra model.
Still, all model variations may be useful, when taking account all possible devices that could take advantage of this project.
For production deployment, typical improvement pipeline would have the following formats:
- Ultra (1x) ->  Ultra  (ad2x) ->   Super  (2x)   (for powerful machines)
- Ultra (1x) ->  Mega  (ad2x)  -> Mega(2x) (for powerful machines)
- Ultra (1x) ->  Super (ad2x)  -> Super (2x)  (for intermediary machines)
- Mega (1x) ->  Mega (ad2x)  ->  Mega (2x) (for intermediary machines)
- Super (1x) ->  Super (ad2x) ->  Super (2x) (for edge devices, mobile or weak notebooks)


### Training Regimen and Loss Functions

Starting from a pretrain, at first model training was not improving the validation metrics. Compressed images are mathematically very similar to the Original ones, so we needed a way to create a Loss that was more sensitive to the issues created by compression. 


We created four novel Loss functions to guide the model into learning, they were critical for optimizing detail recovery and artifact suppression:

1. Canny Edge Loss: This function measures the overlap of edges using Dice Loss (0.25%) and applies L1 (Charbonier Loss, 0.75%) exclusively in the regions identified as edges by a Differentiable Canny Edge Detector.
2. Patch Variance Loss: This function applies Mean Squared Error (MSE), scaled (1 for compressed image, 0 for original image), weighted by color variance over 20-pixel patches. This mechanism effectively directs the loss function's attention to areas with high detail and object movement.
3. Face Aware Loss: This loss combines SSIM, Charbonier, and MSE losses, calculated strictly within the regions identified as faces by an initial face detector. This attention mechanism requires the creation of specialized batches containing only facial patches and pre computing face bounding boxes for the whole dataset, it required a huge rewrite of the training framework.
4. Combined Patch Variance: A combination of SSIM, Charbonier, and MSE is measured, employing the same attention mechanism leveraged in the Patch Variance Loss.

### The Loss Balancing Act

We balanced the weights of each loss used during training, so that they would value 1 for the compressed image(lq source) and 0 for the original image, when compared to the original image. This way, no loss would overpower the others, numerically. The risk by doing this balancing is that the gradients may have a too shallow slope to move, when each loss is pointing in a different, antagonizing direction, it may however lead into directions that improve all losses at times, so the actual benefit or disavantage of the Loss Balancing Act is very hard to measure.


### Evaluation Methodology
Evaluation will utilize both quantitative and perceptual metrics to determine if the objectives were met.

 Quantitative Metrics:
  
  PSNR (Peak Signal-to-Noise Ratio): Used to measure pixel-level differences between the generated output and the ground truth.
  
  SSIM (Structural Similarity Index): Used to evaluate the structural similarity and quality preservation.

Perceptual Metric:
    
  DISTS (Deep Image Structure and Texture Similarity): Used to provide a measure of perceptual quality based on deep feature representations.

The model combinations will be benchmarked to confirm their real-time feasibility.
While preliminary performance estimations can be derived from existing compact model benchmarks (here: https://github.com/the-database/mpv-upscale-2x_animejanai/wiki/Benchmarks#running-benchmarks), project-specific benchmarking remains an eventual necessity. 

A specific test will be conducted to determine if the ad2x model exhibits generalized superiority, allowing it to be used in place of the dedicated 2x model in production environments.

### Datasets and Evolution
We created our own dataset for this task, using the blender foundation open movies. A varied animation dataset, with 3d and 2d styles. A complete introduction to the database is here:
https://github.com/VictorManoelRG/dgm-2025.2/tree/main/projects/super-video-decompression/data


|Dataset | Web Address | Descriptive Summary|
|----- | ----- | -----|
|Blender Foundation Open Movies Compression Decompression | https://huggingface.co/datasets/Fransferdy/blender_foundation_open_movies_compression_decompression | Contains 10 animations videos, 78 minutes, some 3d some 2d|

Initially we deduplicated images that were less than 20% different, ending up with 6k~ images, but we lost too many in between frames, so in a second attempt we remade the dataset with 10% different images, at 15k images per compression quality.
Videos were compressed with AV 264 codec, at compression levels 10 through 40 with a stride of 2, then converted into webp images at 90/100 quality, we then split the dataset into train, test and val. With test being 30% of the total, and val being 3% of test.

For our model training we used the following combinations:

34 compression to 10 (at same resolution) (decompressModel generated here)

34 compression to 10 (upscale 2X resolution) <- experiment failed, model produced terrible results

10 compression to 10 (upscale 2X resolution)

28 compression decompressed with decompressModel to 10  (upscale 2X resolution)


face aware dataset filtering(only images with faces left) ~3k images:

10 compression to 10 (upscale 2X resolution with face aware loss)

28 compression decompressed with decompressModel to 10  (upscale 2X resolution with face aware loss)

## Bibliographic References

BARAKA MAISELI; ABDALLA, A. T. Seven decades of image super-resolution: achievements, challenges, and opportunities. EURASIP Journal on Advances in Signal Processing, v. 2024, n. 1, 18 jul. 2024.

CHEN, Z. et al. NTIRE 2025 Challenge on Image Super-Resolution (x4): Methods and Results. Disponível em: <https://arxiv.org/abs/2504.14582>. Acesso em: 16 set. 2025.
