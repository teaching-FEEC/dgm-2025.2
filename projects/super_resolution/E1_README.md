


# Super-Resolução de Imagens com Modelos Generativos  
# Image Super-Resolution with Generative Models  

## Presentation  

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).  

The presentation for the E1 delivery can be found here [Super presentation of Super Resolution](https://docs.google.com/presentation/d/1vwSB4IJ2oS3T2qzf-c0n5qcp6ztlI2302o7GSDTtjw0/edit?usp=sharing)

|Name  | RA | Specialization|
|--|--|--|
| Brendon Erick Euzébio Rus Peres  | 256130  | Computing Engineering |
| Miguel Angelo Romanichen Suchodolak  | 178808  | Computing Engineering |
| Núbia Sidene Almeida das Virgens  | 299001  | Computing Engineering |  

## Project Summary Description  

The project addresses the problem of **image super-resolution**, a fundamental task in computer vision that aims to reconstruct high-quality images from their low-resolution counterparts. This problem has strong practical relevance in areas such as medical imaging, satellite image analysis, restoration of historical or degraded images, and even the simple fact of improving the quality of an image that you simply like, or one from a cherished memory of your life.  

The main objective of the project is to design and evaluate a **generative model** capable of improving the resolution and perceptual quality of input images. The model will receive as input a low-resolution image and will output a high-resolution reconstruction with enhanced details and sharper definition.  

As inspiration, this project will draw on the methodology and architecture proposed in the repository [InvSR: Invertible Super-Resolution](https://github.com/zsyOAOA/InvSR).  

## Proposed Methodology  

For this first stage, the proposed methodology is as follows:  

- **Datasets**:  
  - The project intends to use standard super-resolution datasets such as **DIV2K**, **Flickr2K**, and possibly **Set5/Set14** for evaluation.  
  - These datasets were chosen because they are widely adopted benchmarks in super-resolution tasks, providing a reliable basis for comparison with the literature.  

- **Generative Modeling Approaches**:  
  - Exploration of **invertible neural networks** as proposed in *InvSR*.  
  - Consideration of **GAN-based models** (e.g., ESRGAN) for perceptual quality improvement.  
  - Study of diffusion-based approaches for possible integration or comparison.  

- **Reference Articles**:  
  - Bjorn, M., et al. *"A Lightweight Image Super-Resolution Transformer Trained on Low-Resolution Images Only"*, arXiv 2025.
  - Miao, Y., et al. *"A general survey on medical image super-resolution via deep learning"*, ScienceDirect 2025.
  - Chen, Z., et al. *"NTIRE2025 Challenge on Image Super-Resolution (×4): MethodsandResults"*, arXiv 2025.
  - Wang, W., et al. *"A lightweight large receptive field network LrfSR for image super resolution"*, Nature 2025.
  - Guo, Z., et al. *"Invertible Image Rescaling."* NeurIPS 2022.  
  - Wang, X., et al. *"ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks."* ECCV 2018.  
  - Saharia, C., et al. *"Image Super-Resolution via Iterative Refinement."* IEEE TPAMI 2022.

- **Tools**:  
  - **PyTorch** for model development and training.  
  - **Weights & Biases** or TensorBoard for experiment tracking.  
  - Google Colab or local GPU cluster for computational resources.  

- **Expected Results**:  
  - Improved reconstruction quality compared to bicubic upscaling baselines.  
  - Generation of visually convincing high-resolution images, preserving textures and fine details.  

- **Evaluation**:  
  - Quantitative metrics: **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)**.  
  - Perceptual metrics: **LPIPS (Learned Perceptual Image Patch Similarity)**.  
  - Qualitative evaluation through visual inspection and comparison with ground truth images.  

## Schedule  

| Week | Activity |  
|------|----------|  
| 1–2  | Literature review, dataset preparation, and baseline setup (bicubic, ESRGAN). |  
| 3–4  | Initial implementation of the InvSR model. |  
| 5–6  | Model training and hyperparameter tuning. |  
| 7    | Intermediate evaluation and analysis of quantitative/qualitative results. |  
| 8    | Refinements (integration of additional techniques, e.g., perceptual loss). |  
| 9    | Final experiments, result consolidation, and comparison with benchmarks. |  
| 10   | Report writing and final presentation preparation. |  

## Bibliographic References  

- Guo, Z., et al. *Invertible Image Rescaling.* NeurIPS, 2022.  
- Wang, X., et al. *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.* ECCV Workshops, 2018.  
- Saharia, C., et al. *Image Super-Resolution via Iterative Refinement.* IEEE TPAMI, 2022.  
- Ledig, C., et al. *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN).* CVPR, 2017.  

