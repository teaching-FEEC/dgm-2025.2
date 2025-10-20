# `<Project Title in Portuguese>`
# `<Super-resolução de imagens com modelos generativos>`

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

### Deliverables
The presentation for the E2 delivery can be found here * ADD LINK

> Include name, RA, and specialization focus of each group member.  
> |Name  | RA | Specialization|
> |--|--|--|
> | Brendon Erick Euzébio Rus Peres  | 256130  | Computer Engineering|
> | Miguel Angelo Romanichen Suchodolak | 178808 | Computer Engineering|
> | Núbia Sidene Almeida das Virgens  | 299001  | Computer Engineering|

## Abstract

The project addresses the problem of **image super-resolution**, a fundamental task in computer vision that aims to reconstruct high-quality images from their low-resolution counterparts. This problem has strong practical relevance in areas such as:
- **Low-budget devices**: Get one poor picture and enhance it
- **Medical Imaging**: Enhancing diagnostic image quality
- **Satellite Analysis**: Improving remote sensing data
- **Image Restoration**: Recovering historical or degraded photographs
- **Personal Use**: Upscaling cherished memories and favorite images 
  
## Objective

The main objective of the project is to design and evaluate a **generative model** capable of improving the resolution and perceptual quality of input images. The model will receive as input a low-resolution image and will output a high-resolution reconstruction with enhanced details and sharper definition, in general its goals can be set by:

- Improving resolution and perceptual quality of input images
- Preserving fine details and texture information
- Generating visually convincing high-resolution reconstructions
- Outperforming traditional interpolation methods, like bicubic

As inspiration, this project will draw on the methodology and architecture proposed in the repository [InvSR: Invertible Super-Resolution](https://github.com/zsyOAOA/InvSR).  

## Methodology

For the development of this stage of the project, two implementation approaches for the image super-resolution model were evaluated: one executed in the Google Colab environment and the other in a local environment using VS Code.

Initially, both platforms were configured and tested with the goal of analyzing the model’s performance under different execution conditions. After the comparative tests, the VS Code environment was selected as the most suitable due to its better processing time, greater stability, and enhanced control over experiments.

From this definition onward, all subsequent stages of training, validation, and performance analysis were carried out in VS Code. The main evaluation metrics — such as PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and LPIPS (Learned Perceptual Image Patch Similarity) — were applied in this environment to assess the quality of the generated images and to compare the results with the high-resolution reference images.

### Datasets and Evolution
 List the datasets used in the project.  

| Dataset | Type | Description | Usage | LINK |
|---------|------|-------------|-------|-------|
| **DIV2K** | Training/Validation | 2K resolution diverse images | Primary training data | Not yet here
| **Flickr2K** | Training | High-quality photos from Flickr | Additional training data | [Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k?resource=download)
| **Set5/Set14** | Evaluation | Standard benchmark sets | Performance evaluation | Not yet here

  - The project intends to use standard super-resolution datasets such as **DIV2K**, **Flickr2K**, and possibly **Set5/Set14** for evaluation.  
  - These datasets were chosen because they are widely adopted benchmarks in super-resolution tasks, providing a reliable basis for comparison with the literature.  

### Workflow

<img width="1284" height="2404" alt="image" src="https://github.com/user-attachments/assets/b1b9dfae-4fc9-475e-b238-a3268ce3109e" />


## Experiments, Results, and Discussion of Results

In this partial submission (E2), this section presents the exploratory phase of the project, in which different implementation environments were tested and preliminary results were analyzed.

Two approaches were initially evaluated for implementing the image super-resolution model: one using Google Colab and another using a local environment (VS Code). After comparative tests, the VS Code environment was selected as the main platform due to its superior performance, stability, and control during model execution.

All subsequent experiments — including training, validation, and performance evaluation — were conducted in the VS Code environment. The validation process focused on analyzing key performance metrics, such as PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and LPIPS (Learned Perceptual Image Patch Similarity), to assess the visual quality and fidelity of the generated images. These metrics and validation outputs can be viewed in detail at the following link:
https://github.com/nubiasidene/dgm-2025.2-g4/tree/main/projects/super_resolution/validation

The discussion of results highlights that both environments enabled correct model execution, but the local implementation in VS Code provided a significant reduction in simulation time (from minutes in Colab to milliseconds locally) and allowed greater control over experiments and reproducibility. These findings guided the group’s decision to continue project development exclusively in VS Code.

At this stage, the obtained results represent partial progress, focused on validating the methodological choices and ensuring the consistency of the implementation pipeline. The next steps will include refining the model’s architecture, performing additional experiments, and broadening the evaluation metrics to better capture perceptual and quantitative aspects of super-resolution performance.


## Schedule  

Delivery | Week | Activity |  
|--------|------|----------|  
E1       | 1–2  | Literature review, dataset preparation, and baseline setup (bicubic, ESRGAN). |  
E2       | 3–4  | Initial implementation of the InvSR model. |  
E2       | 5–6  | Model training and hyperparameter tuning. |  
E2       | 7    | Intermediate evaluation and analysis of quantitative/qualitative results. |  
E3       | 8    | Refinements (integration of additional techniques, e.g., perceptual loss). |  
E3       | 9    | Final experiments, result consolidation, and comparison with benchmarks. |  
E3       | 10   | Report writing and final presentation preparation. |  


## Bibliographic References  
- **Bjorn, M., et al.** - *"A Lightweight Image Super-Resolution Transformer Trained on Low-Resolution Images Only"* ([arXiv 2025](https://arxiv.org/))
- **Miao, Y., et al.** - *"A general survey on medical image super-resolution via deep learning"* ([ScienceDirect 2025](https://www.sciencedirect.com/))
- **Chen, Z., et al.** - *"NTIRE2025 Challenge on Image Super-Resolution (×4): Methods and Results"* ([arXiv 2025](https://arxiv.org/))
- **Wang, W., et al.** - *"A lightweight large receptive field network LrfSR for image super resolution"* ([Nature 2025](https://www.nature.com/))
- **Guo, Z., et al.** - *"Invertible Image Rescaling"* ([NeurIPS 2022](https://proceedings.neurips.cc/))
- **Wang, X., et al.** - *"ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"* ([ECCV 2018](https://arxiv.org/))
- **Saharia, C., et al.** - *"Image Super-Resolution via Iterative Refinement"* ([IEEE TPAMI 2022](https://ieeexplore.ieee.org/))
