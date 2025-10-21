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


## Project Summary Description
Today, nearly 80% of global internet bandwidth is consumed by video streaming. Of this total, around 90% of the content is delivered in 1080p resolution or lower, through platforms such as Twitch.tv, YouTube, Netflix, Disney+, and Hulu. To make large-scale distribution feasible, most videos are compressed using lossy algorithms like H.264/AVC. While this approach is effective in reducing file size, it inevitably leads to lower image and audio quality, with no straightforward way to fully reverse the process and recover the original material.

Super-resolution techniques are methods developed to improve the quality of images. They are usually applied to cases such as noisy photographs, medical imaging, or video conferencing, where the goal is to recover details that are not clearly visible in the original material. A related approach is video decompression, which addresses the specific problem of lossy video compression. In this case, the objective is to reduce or remove compression artifacts while at the same time increasing the resolution of the video, aiming to approximate the original quality before compression.

The main goal of the project to develop a model capable of improving compressed video frames towards their original quality would work by reducing compression artifacts and enhancing resolution in a way that restores perceptual details. Such a system could be applied in different contexts. For streaming platforms, it would enable the transmission of videos at lower resolutions and bitrates, with the model decompressing and upscaling them on the client side to recover near-original quality. This approach would also improve accessibility for users with poor internet connectivity, as they could experience higher perceptual quality without requiring more bandwidth. In the context of recovering lost or degraded media, the model could be used to remaster antique or low-quality footage, improving clarity and preserving cultural content that lacks high-fidelity sources. Finally, for general video enhancement, the method could upscale low-resolution recordings, such as older videos or digital content captured with outdated devices, providing a perceptual quality closer to modern standards.

## Proposed Methodology

The project will use a dataset constructed from Creative Commons licensed animations provided by the Blender Foundation, totaling 78:37 minutes of video. These were selected due to their open licensing and high-quality source material. High-resolution frames (1920×1080) were paired with compressed versions at 15 different compression levels down to 480×202 resolution, generated using HandbrakeCLI with H.264/AVC encoding. To manage data volume, WebP image sequences were extracted, and deduplication was performed using average hashing and Hamming distance, reducing the dataset from 110,911 HQ and 1,663,665 LQ images to 6,056 HQ and 90,840 LQ images.

The architecture is still under definition, but the training process will likely involve the use of generative adversarial networks (GANs) with discriminators such as MetaGAN or PatchGAN. Different loss functions, including perceptual loss, DISTS loss, and wavelet loss, will be investigated. Experiments may also include attention-based models, while convolutional sequential layers will be maintained to allow any size input. Models will be benchmarked for real-time feasibility.

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
