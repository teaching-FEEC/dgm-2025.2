# `<Super-resolução de imagens com modelos generativos>`
# `<Image Super-Resolution with Generative Models>`

## Presentation

This project originated in the context of the graduate course IA376N - Generative AI: from models to multimodal applications, offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).


 |Name  | RA | Specialization|
 |--|--|--|
 | Brendon Erick Euzébio Rus Peres  | 256130  | Computer Engineering|
 | Miguel Angelo Romanichen Suchodolak | 178808 | Computer Engineering|
 | Núbia Sidene Almeida das Virgens  | 299001  | Computer Engineering|

## Abstract

The project addresses the problem of **image super-resolution**, a fundamental task in computer vision that aims to reconstruct high-quality images from their low-resolution counterparts. This problem has strong practical relevance in areas such as:
- **Low-budget devices**: Get one poor picture and enhance it
- **Medical Imaging**: Enhancing diagnostic image quality
- **Satellite Analysis**: Improving remote sensing data
- **Image Restoration**: Recovering historical or degraded photographs
- **Personal Use**: Upscaling cherished memories and favorite images 

## Problem Description / Motivation



## Objective

The main objective of the project is to design and evaluate a **generative model** capable of improving the resolution and perceptual quality of input images. The model will receive as input a low-resolution image and will output a high-resolution reconstruction with enhanced details and sharper definition, in general its goals can be set by:

- Improving resolution and perceptual quality of input images
- Preserving fine details and texture information
- Generating visually convincing high-resolution reconstructions
- Outperforming traditional interpolation methods, like bicubic

As inspiration, this project will draw on the methodology and architecture proposed in the repository [InvSR: Invertible Super-Resolution](https://github.com/zsyOAOA/InvSR).  
  

## Methodology

The methodology adopted in this project was organized into two main implementation approaches, complemented by a punctual use of high-performance hardware (A100 GPU) in the final stage. The objective was to evaluate the performance of the image super-resolution model across different computational environments while ensuring reproducibility, consistency, and rigorous assessment of reconstruction quality.

### 1. Approach 1 — Initial Development in Google Colab

Google Colab was used during the initial development phase to build and validate the super-resolution pipeline. At this stage:
- dependencies, datasets, and preprocessing functions were configured;
- the workflow was validated to ensure correctness;
- preliminary tests were executed to confirm model stability;
- fast prototype runs were performed to validate the end-to-end process.

Colab was chosen because it enables quick experimentation without requiring local computational resources and provides GPU access (typically T4 or P100) suitable for early testing.

### 2. Approach 2 — Local Development and Full Training in VS Code

After validation in Colab, the main development process was conducted locally in VS Code, including:
- full organization of the codebase;
- complete model training;
- hyperparameter tuning;
- reproducibility experiments;
- metric computation and analysis;
- documentation of the results.

The local environment allowed precise control over versions, dependencies, repository structure, and integration with GitHub.

### 3. Punctual Use of the A100 GPU in the Final Stages

The NVIDIA A100 GPU was used only in the final phase of the project with the purpose of:
- accelerating the training of heavier models such as deep architectures and VAEs;
- validating model performance under high-end computational hardware;
- comparing training time against the local environment;
- generating more robust and consistent final results.

The A100 was not part of the primary pipeline; instead, it was employed exclusively for final optimization and verification of the training procedure.

### Datasets and Evolution
List the datasets used in the project.  

| Dataset | Type | Description | Usage | LINK |
|---------|------|-------------|-------|-------|
| **DIV2K** | Training/Validation | 2K resolution diverse images | Primary training data | Not yet here
| **Flickr2K** | Training | High-quality photos from Flickr | Additional training data | [Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k?resource=download)
| **Set5/Set14** | Evaluation | Standard benchmark sets | Performance evaluation | Not yet here

- The project intends to use standard super-resolution datasets such as **DIV2K**, **Flickr2K**, and possibly **Set5/Set14** for evaluation.  
- These datasets were chosen because they are widely adopted benchmarks in super-resolution tasks, providing a reliable basis for comparison with the literature.

### Evaluation Methodology

The model’s performance was assessed using both quantitative and qualitative criteria:

### Quantitative Evaluation

- PSNR — measures signal fidelity
- SSIM — measures structural similarity
- LPIPS — measures perceptual similarity based on deep features

### Qualitative Evaluation

- visual inspection of HR vs. SR reconstructions;
- analysis of textures, edges, colors, and artifacts;
- observation of sharpness and noise behavior.

### Success Criteria

The model is considered successful if:

- PSNR and SSIM surpass the bicubic baseline;
- LPIPS is reduced (lower = more perceptually similar);
- the reconstructed images present visually improved details;
- the model behaves consistently across environments (Colab, local, A100).

### Workflow
> Use a tool that allows you to design the workflow and save it as an image (e.g., Draw.io). Insert the image in this section.  
> You may choose to use a workflow manager (Sacred, Pachyderm, etc.), in which case use the manager to generate a diagram for you.  
> Remember that the goal of drawing the workflow is to help anyone who wishes to reproduce your experiments.  

## Experiments, Results, and Discussion of Results

> In the partial project submission (E2), this section may contain partial results, explorations of implemented solutions, and  
> discussions about such experiments, including decisions to change the project trajectory or the description of new experiments as a result of these explorations.  

> In the final project submission (E3), this section should list the **main** results obtained (not necessarily all), which best represent the fulfillment of the project objectives.  

> The discussion of results may be carried out in a separate section or integrated into the results section. This is a matter of style.  
> It is considered fundamental that the presentation of results should not serve as a treatise whose only purpose is to show that "a lot of work was done."  
> What is expected from this section is that it **presents and discusses** only the most **relevant results**, highlighting the **strengths and/or limitations** of the methodology, emphasizing aspects of **performance**, and containing content that can be classified as **organized, didactic, and reproducible sharing of knowledge relevant to the community**.  

## Conclusion

> The Conclusion section should recover the main information already presented in the report and point to future work.  
> In the partial project submission (E2), it may contain information about which steps or how the project will be conducted until its completion.  
> In the final project submission (E3), the conclusion is expected to outline, among other aspects, possibilities for the project’s continuation.  

## Bibliographic References
- **Bjorn, M., et al.** - *"A Lightweight Image Super-Resolution Transformer Trained on Low-Resolution Images Only"* ([arXiv 2025](https://arxiv.org/))
- **Miao, Y., et al.** - *"A general survey on medical image super-resolution via deep learning"* ([ScienceDirect 2025](https://www.sciencedirect.com/))
- **Chen, Z., et al.** - *"NTIRE2025 Challenge on Image Super-Resolution (×4): Methods and Results"* ([arXiv 2025](https://arxiv.org/))
- **Wang, W., et al.** - *"A lightweight large receptive field network LrfSR for image super resolution"* ([Nature 2025](https://www.nature.com/))
- **Guo, Z., et al.** - *"Invertible Image Rescaling"* ([NeurIPS 2022](https://proceedings.neurips.cc/))
- **Wang, X., et al.** - *"ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"* ([ECCV 2018](https://arxiv.org/))
- **Saharia, C., et al.** - *"Image Super-Resolution via Iterative Refinement"* ([IEEE TPAMI 2022](https://ieeexplore.ieee.org/))
