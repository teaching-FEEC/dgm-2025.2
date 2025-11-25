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

Many real-world images inherently exhibit low resolution due to limitations in acquisition devices, environmental conditions, or historical constraints. This issue is particularly relevant in several domains where image clarity is essential for analysis and decision-making.

## Objective

The main objective of the project is to design and evaluate a **generative model** capable of improving the resolution and perceptual quality of input images. The model will receive as input a low-resolution image and will output a high-resolution reconstruction with enhanced details and sharper definition, in general its goals can be set by:

- Improving resolution and perceptual quality of input images
- Preserving fine details and texture information
- Generating visually convincing high-resolution reconstructions
- Outperforming traditional interpolation methods, like bicubic

As inspiration, this project will draw on the methodology and architecture proposed in the repository [InvSR: Invertible Super-Resolution](https://github.com/zsyOAOA/InvSR).  
  

## Methodology

Investigated Approaches and Rationale for Invertible Neural Networks (INNs)

During the preliminary phase of the project, different families of super-resolution architectures were explored to establish comparative baselines and justify the final modeling direction. Among the techniques investigated were:

- GAN-based models, such as ESRGAN, widely adopted for perceptual image enhancement due to their ability to produce sharp and visually appealing textures;
- Diffusion models, considered for comparison due to their state-of-the-art performance in generative image reconstruction and high-fidelity detail synthesis.

However, traditional deep learning architectures rely on standard components — such as convolutions, pooling operations, or nonlinear activations like ReLU — that are not inherently invertible. For instance, pooling discards spatial information, preventing exact reconstruction of an input from its output.

To address this limitation, the project focused on Invertible Neural Networks (INNs). INNs employ specialized building blocks, including:

- Coupling layers, which split and transform the input in a way that allows exact inversion;
- Invertible 1×1 convolutions, which preserve dimensionality and enable reversible feature transformations.

These properties allow the network to support both forward (low-resolution → high-resolution) and inverse (high-resolution → low-resolution) mappings, making INNs particularly suitable for super-resolution tasks where information preservation is critical.

### Integration Into the Methodology

The methodology adopted in this project was structured around two core implementation approaches and a supplementary high-performance execution phase using the A100 GPU. These methodological steps were designed to evaluate model performance in different environments while ensuring reproducibility and rigorous quality assessment.

### Approach 1 — Initial Development in Google Colab

Google Colab was used for early prototyping and validation of the super-resolution pipeline. In this stage:
- dependencies, datasets, and preprocessing routines were configured;
- the pipeline was validated end-to-end;
- preliminary experiments were conducted to ensure model stability;
- lightweight tests were run to refine workflow consistency.

Colab was especially suitable for rapid iteration due to its integrated GPU availability (T4/P100) and ease of setup.

### Approach 2 — Local Development and Full Training in VS Code

Once validated, the full development and training process was carried out locally in VS Code, including:
- full organization and structuring of the codebase;
- complete model training using INNs and baseline architectures;
- hyperparameter tuning;
- reproducibility experiments;
- computation and analysis of metrics (PSNR, SSIM, LPIPS);
- documentation and integration with GitHub.

The local environment provided fine-grained control over versions, dependencies, and source-code management.

### Approach 3 - Punctual Use of the A100 GPU in the Final Stages

The NVIDIA A100 GPU was used only in the concluding stage, with the goals of:
- accelerating the training of deeper or more computationally demanding architectures (e.g., VAE-based or diffusion-based baselines);
- validating the model under high-performance conditions;
- comparing execution times with local hardware;
- producing robust final results.

The A100 was not part of the main pipeline; its use was strictly targeted for optimization and final verification.



## Datasets and Evolution
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
