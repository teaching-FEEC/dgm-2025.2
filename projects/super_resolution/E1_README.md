


# Image Super-Resolution with Generative Models  
  

## Presentation  

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).  

### Deliverables
The presentation for the E1 delivery can be found here [Presentation Image Super-Resolution with Generative Models](https://docs.google.com/presentation/d/1LrxHj0p9UAfdooWXWIvNTW1KwR4ejZvf/edit?slide=id.p1#slide=id.p1)

|Name  | RA | Specialization|
|--|--|--|
| Brendon Erick Euzébio Rus Peres  | 256130  | Computing Engineering |
| Miguel Angelo Romanichen Suchodolak  | 178808  | Computing Engineering |
| Núbia Sidene Almeida das Virgens  | 299001  | Computing Engineering |  

## Project Summary Description  

The project addresses the problem of **image super-resolution**, a fundamental task in computer vision that aims to reconstruct high-quality images from their low-resolution counterparts. This problem has strong practical relevance in areas such as:
- **Low-budget devices**: Get one poor picture and enhance it
- **Medical Imaging**: Enhancing diagnostic image quality
- **Satellite Analysis**: Improving remote sensing data
- **Image Restoration**: Recovering historical or degraded photographs
- **Personal Use**: Upscaling cherished memories and favorite images 

The main objective of the project is to design and evaluate a **generative model** capable of improving the resolution and perceptual quality of input images. The model will receive as input a low-resolution image and will output a high-resolution reconstruction with enhanced details and sharper definition, in general its goals can be set by:

- Improving resolution and perceptual quality of input images
- Preserving fine details and texture information
- Generating visually convincing high-resolution reconstructions
- Outperforming traditional interpolation methods, like bicubic

As inspiration, this project will draw on the methodology and architecture proposed in the repository [InvSR: Invertible Super-Resolution](https://github.com/zsyOAOA/InvSR).  

## Proposed Methodology  

For this first stage, the proposed methodology is as follows:  

### **Datasets**:

| Dataset | Type | Description | Usage |
|---------|------|-------------|-------|
| **DIV2K** | Training/Validation | 2K resolution diverse images | Primary training data |
| **Flickr2K** | Training | High-quality photos from Flickr | Additional training data |
| **Set5/Set14** | Evaluation | Standard benchmark sets | Performance evaluation |

  - The project intends to use standard super-resolution datasets such as **DIV2K**, **Flickr2K**, and possibly **Set5/Set14** for evaluation.  
  - These datasets were chosen because they are widely adopted benchmarks in super-resolution tasks, providing a reliable basis for comparison with the literature.  

### **Generative Modeling Approaches**:
  #### Primary Approach: Invertible Neural Networks
  - **InvSR Architecture**: Exploring invertible transformations for bidirectional mapping
  - **Coupling Layers**: Implementing affine coupling for information preservation
  - **Multi-scale Processing**: Handling different resolution scales efficiently

  #### Alternative Approaches
  - **GAN-based Models**: ESRGAN for enhanced perceptual quality
  - **Diffusion Models**: Iterative refinement for high-quality generation
  - **Hybrid Architectures**: Combining multiple generative paradigms

  - Exploration of **invertible neural networks** as proposed in *InvSR*.  
  - Consideration of **GAN-based models** (e.g., ESRGAN) for perceptual quality improvement.  
  - Study of diffusion-based approaches for possible integration or comparison.  

### **Tools**:  
  - **PyTorch** for model development and training.  
  - **Weights & Biases** or TensorBoard for experiment tracking.  
  - Google Colab or local GPU cluster for computational resources.  

### **Expected Results**:  
    - **Quantitative Improvements**: Significant gains over bicubic interpolation baselines
    - **Visual Quality**: Perceptually convincing high-resolution reconstructions
    - **Detail Preservation**: Enhanced texture and fine-structure recovery
    - **Computational Efficiency**: Balanced trade-off between quality and inference speed

### **Evaluation**:  
  #### Quantitative Metrics
  - **📐 PSNR** (Peak Signal-to-Noise Ratio): Pixel-level reconstruction accuracy
  - **🔍 SSIM** (Structural Similarity Index): Structural information preservation
  - **👁️ LPIPS** (Learned Perceptual Image Patch Similarity): Perceptual similarity assessment

    #### Qualitative Assessment
    - **Visual Inspection**: Side-by-side comparison with ground truth
    - **User Studies**: Perceptual quality evaluation
    - **Ablation Studies**: Component-wise contribution analysis

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
- **Bjorn, M., et al.** - *"A Lightweight Image Super-Resolution Transformer Trained on Low-Resolution Images Only"* ([arXiv 2025](https://arxiv.org/))
- **Miao, Y., et al.** - *"A general survey on medical image super-resolution via deep learning"* ([ScienceDirect 2025](https://www.sciencedirect.com/))
- **Chen, Z., et al.** - *"NTIRE2025 Challenge on Image Super-Resolution (×4): Methods and Results"* ([arXiv 2025](https://arxiv.org/))
- **Wang, W., et al.** - *"A lightweight large receptive field network LrfSR for image super resolution"* ([Nature 2025](https://www.nature.com/))
- **Guo, Z., et al.** - *"Invertible Image Rescaling"* ([NeurIPS 2022](https://proceedings.neurips.cc/))
- **Wang, X., et al.** - *"ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"* ([ECCV 2018](https://arxiv.org/))
- **Saharia, C., et al.** - *"Image Super-Resolution via Iterative Refinement"* ([IEEE TPAMI 2022](https://ieeexplore.ieee.org/))
