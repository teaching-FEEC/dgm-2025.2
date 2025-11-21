# `<Super Descompressão de Vídeo>`
# `<Super Video Decompression>`

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


## Project Summary Description
Today, nearly 80% of global internet bandwidth is consumed by video streaming. Of this total, around 90% of the content is delivered in 1080p resolution or lower, through platforms such as Twitch.tv, YouTube, Netflix, Disney+, and Hulu. To make large-scale distribution feasible, most videos are compressed using lossy algorithms like H.264/AVC. While this approach is effective in reducing file size, it inevitably leads to lower image and audio quality, with no straightforward way to fully reverse the process and recover the original material.

Super-resolution techniques are methods developed to improve the quality of images. They are usually applied to cases such as noisy photographs, medical imaging, or video conferencing, where the goal is to recover details that are not clearly visible in the original material. A related approach is video decompression, which addresses the specific problem of lossy video compression. In this case, the objective is to reduce or remove compression artifacts while at the same time increasing the resolution of the video, aiming to approximate the original quality before compression.

The main goal of the project to develop a model capable of improving compressed video frames towards their original quality would work by reducing compression artifacts and enhancing resolution in a way that restores perceptual details. Such a system could be applied in different contexts. For streaming platforms, it would enable the transmission of videos at lower resolutions and bitrates, with the model decompressing and upscaling them on the client side to recover near-original quality. This approach would also improve accessibility for users with poor internet connectivity, as they could experience higher perceptual quality without requiring more bandwidth. In the context of recovering lost or degraded media, the model could be used to remaster antique or low-quality footage, improving clarity and preserving cultural content that lacks high-fidelity sources. Finally, for general video enhancement, the method could upscale low-resolution recordings, such as older videos or digital content captured with outdated devices, providing a perceptual quality closer to modern standards.

## Proposed Methodology

The project will use a dataset constructed from Creative Commons licensed animations provided by the Blender Foundation, totaling 78:37 minutes of video. These were selected due to their open licensing and high-quality source material. High-resolution frames (1920×1080) were paired with compressed versions at 15 different compression levels down to 480×202 resolution, generated using HandbrakeCLI with H.264/AVC encoding. To manage data volume, WebP image sequences were extracted, and deduplication was performed using average hashing and Hamming distance, reducing the dataset from 110,911 HQ and 1,663,665 LQ images to 6,056 HQ and 90,840 LQ images.

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

## Workflow Diagrams

![Workflow Diagrams](./docs/Super%20Video%20Decompression.drawio.svg "Workflow Diagrams")

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

Models were trained for around 150k Iterations, the best models according to PSNR, DISTS and SSIM metrics were chosen from the 150 models generated during training. Most best models were within 90k to 150k range of training. Some were finetuned for more 150k iterations.

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

## Experiments, Results, and Discussion of Results

| x1 (34 to 10)         | PSNR    | DISTS <Better | SSIM \* |
| --------------------- | ------- | ------------- | ------- |
| baseline (identity)   | 32,9054 | 1,1195        | 1,8710  |
| pretrain super        | 32,2302 | 1,1113        | 1,8652  |
| pretrain mega         | 32,1229 | 1,0999        | 1,8574  |
| pretrain ultra        | 32,2294 | 1,1129        | 1,8599  |
| super                 | 33,4368 | 1,0960        | 1,8802  |
| mega                  | 33,5054 | 1,0939        | 1,8820  |
| ultra                 | 33,5668 | 1,0908        | 1,8830  |

First, note how the compressed frame has a PSNR and SSIM better than the pretrain models(which have already been pretrained for at least 500k iterations!), this shows how the problem we are trying to solve is hard, and how "close to the original image" the compressed one is, according to quantitative metrics.
PSNR is measured in logarithimic scale, so small variations mean big differences.
Throughout the literature, is is said that a PSNR from 30 to 34 is good, and 35 to 38 is excellent. However our compressed, perceptually terrible image already starts at 32 PSNR in these metrics, our best model PSNR is only 0.6 PSNR from the baseline, but image quality is profoundly improved. The culprit here is that most of the image is not relevant, since most of the image is a background, with low colour variation, and our metrics take that global value into account. In reallity, perceptual quality lies in the few parts of the image that have moving objects, the part that captures human attention, these moving objects are where most of the compression artifacts arise, which is why we see the compressed images as ugly/bad and these global metrics pay little attention to these.

Although it seems the quantitative values don't differ much, each 0.0001 in each metric makes a huge difference in perceptual quality in the generated image, because these improvements happen where it matters in the image. Here the ultra model is generating overall smoother (with less artifacts) frames. Image quality is easier measured with humans eyes, the mega variation here for example, seems to better reconstruct fine details in frames, than the ultra.

## Conclusion

During the development of this project we created 5 novel loss functions with attention mechanisms, which 4 were beneficial for training models for the decompression and upscaling tasks. Our dataset made with creative commons blender foundation open movies, proved diverse enough for the models to learn without overfitting, after training our models improved in all 3 metrics, showing both good fidelity and perceptual quality, at the end of this project we have 3 variations of compact models for each of the 3 tasks, making it possible to run combinations in the most diverse hardware, from mobile phones to strong desktop computers. At E2 delivery we still have to convert our models to shaders, in order to make the usage of them easier and more performant and we still have to complete the quantitative evaluation of the 2X models.


For the feasibility of the shader version of our model, we used the compact network for all three models, however we discussed using Sebica and Spanplus models for the first model of our improving pipeline (the 1X time decompression model), these models have attention mechanisms that could help improve the 1X model even more, since the 2X models receive as input the output of the 1X model it is of utter importance that the 1X is the best model (and probably the heaviest computationally), these models with attention may prove to be a better 1X alternative, at the cost of not being easily converted into a shader, this may be experimented in a future work.

We also dabbled around the idea of using the previous frame as a guide to the following frame, during training and inference, the previous frame would be converted into grayscale and fed as the fourth channel of the current frame image, the model would then have to learn to use that extra channel to make better images in the end. Since we were using a somewhat rigid training framework, we dropped this idea, but in a more flexible setting it could be experimented with, it is also harder to implent in inference mode(in a video player for example).

For evaluation, both quantitative and perceptual metrics will be applied. PSNR will be used to measure pixel-level differences, SSIM will be used to evaluate structural similarity, and DISTS will provide a perceptual similarity measure based on deep feature representations. The expected outcome is a model capable of restoring compressed frames to higher perceptual quality while ensuring feasibility for streaming use cases.

## Schedule
| Phase                          | Duration (weeks) | Final Date |
|--------------------------------|------------------|----------------|
| Literature Review              | 3                |  09/26|
| Dataset Generation             | 2                |  09/19|
| Model Implementation & Testing | 5                | 10/31 |
| Final Evaluation               | 2                | 11/12|
| Documentation                  | 2                | 11/21 | 

## Bibliographic References
BARAKA MAISELI; ABDALLA, A. T. Seven decades of image super-resolution: achievements, challenges, and opportunities. EURASIP Journal on Advances in Signal Processing, v. 2024, n. 1, 18 jul. 2024.

CHEN, Z. et al. NTIRE 2025 Challenge on Image Super-Resolution (x4): Methods and Results. Disponível em: <https://arxiv.org/abs/2504.14582>. Acesso em: 16 set. 2025.
