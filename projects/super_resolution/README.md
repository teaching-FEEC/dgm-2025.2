


# Image Super-Resolution with Generative Models  
  

## Presentation  

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).  

### Deliverables
- The presentation for the E1 delivery can be found here [Presentation Image Super-Resolution with Generative Models](https://docs.google.com/presentation/d/1LrxHj0p9UAfdooWXWIvNTW1KwR4ejZvf/edit?slide=id.p1#slide=id.p1)
- The presentation for the E2 delivery can be found here [Presentation E2 Google Slides](https://docs.google.com/presentation/d/1zLd4ip43czBegF7HHgJYoS5qRbJ5Y2kQ/edit?slide=id.p1)
- The presentation for the E3 delivery can be found here [Presentation E3 Google Slides](https://docs.google.com/presentation/d/1Fs8ZEbfoHMGtiiKB4u0T6H9FeGYvkwkk/edit?slide=id.p1#slide=id.p1)

|Name  | RA | Specialization|
|--|--|--|
| Brendon Erick Euzébio Rus Peres  | 256130  | Computing Engineering |
| Miguel Angelo Romanichen Suchodolak  | 178808  | Computing Engineering |
| Núbia Sidene Almeida das Virgens  | 299001  | Computing Engineering |  

## Project Summary Description  
Our project does the sam
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

| Dataset | Type | Description | Usage | LINK |
|---------|------|-------------|-------|-------|
| **DIV2K** | Training/Validation | 2K resolution diverse images | Primary training data | Not yet here
| **Flickr2K** | Training | High-quality photos from Flickr | Additional training data | [Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k?resource=download)
| **Set5/Set14** | Evaluation | Standard benchmark sets | Performance evaluation | Not yet here

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
  - **Custom Validation Framework** for comprehensive model evaluation (PSNR, LPIPS, SSIM).  

### **Expected Results**:  
  - **Quantitative Improvements**: Significant gains over bicubic interpolation baselines
  - **Visual Quality**: Perceptually convincing high-resolution reconstructions
  - **Detail Preservation**: Enhanced texture and fine-structure recovery
  - **Computational Efficiency**: Balanced trade-off between quality and inference speed

### **Evaluation**:  
  #### Quantitative Metrics
  - **PSNR** (Peak Signal-to-Noise Ratio): Pixel-level reconstruction accuracy
  - **SSIM** (Structural Similarity Index): Structural information preservation
  - **LPIPS** (Learned Perceptual Image Patch Similarity): Perceptual similarity assessment

  #### Qualitative Assessment
  - **Visual Inspection**: Side-by-side comparison with ground truth
  - **User Studies**: Perceptual quality evaluation
  - **Ablation Studies**: Component-wise contribution analysis

#### Workflow
![WhatsApp Image 2025-11-24 at 16 56 15_3ab7d84d](https://github.com/user-attachments/assets/283f969e-bbb1-45f4-ad3f-b25f7a2171cf)

## Installation and Setup

This section provides step-by-step instructions to install dependencies and run the Gradio interface (`app.py`) for image super-resolution.

### Prerequisites

- **Python 3.8+** (Python 3.10 or 3.11 recommended)
- **CUDA-capable GPU** (recommended for faster inference, but CPU is also supported)
- **Git** (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd projects/super_resolution
```

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python packages:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Navigate to the `src` directory and install the required packages:

```bash
cd src
pip install -r requirements.txt
```

**Note:** The installation may take several minutes as it includes PyTorch, diffusers, and other deep learning libraries. If you have CUDA installed, PyTorch will automatically use GPU acceleration.

### Step 4: Download Model Weights

The model weights will be automatically downloaded on first run. However, you can also download them manually:

- **Noise Predictor**: `noise_predictor_sd_turbo_v5.pth` (automatically downloaded from HuggingFace)
- **Stable Diffusion Turbo**: The base model will be downloaded automatically when first used

The weights will be saved in the `src/weights/` directory.

### Step 5: Run the Application

From the `src` directory, run:

```bash
python app.py
```

The Gradio interface will start and display a local URL (typically `http://127.0.0.1:7860`). Open this URL in your web browser to access the interface.

### Using the Interface

The application provides two main tabs:

1. **Single Image**: Process a single low-resolution image
   - Upload an image
   - Configure parameters (number of steps, chopping size, color fixing method, etc.)
   - Click "Process" to generate the super-resolved image

2. **Batch Processing**: Process multiple images from a directory
   - Enter the path to a directory containing images
   - Configure parameters
   - Click "Process Folder" to process all images

### Configuration Options

- **Number of steps**: Controls the number of denoising steps (1-5 recommended for speed vs quality trade-off)
- **Chopping size**: Patch size for processing large images (128 or 256)
- **Color Fixing Method**: Choose between None, YCbCr, Wavelet, Histogram, or Hybrid
- **Adaptive Scheduler**: Automatically adjusts timesteps based on image complexity
- **Smart Chopping**: Adaptive overlap based on local complexity (faster + better quality)
- **Edge-Preserving Enhancement**: Enhances edges while preserving smooth regions
- **Adaptive Guidance Scale**: Adjusts guidance scale based on image complexity

### Troubleshooting

**Issue: CUDA out of memory**
- Reduce the `chopping_size` (e.g., from 256 to 128)
- Process smaller images or use CPU mode

**Issue: Module not found errors**
- Ensure you're in the `src` directory when running `app.py`
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Issue: Model download fails**
- Check your internet connection
- The models are downloaded from HuggingFace - ensure you have access

**Issue: Gradio interface doesn't open**
- Check if the port 7860 is already in use
- The interface will show the URL in the terminal output

### System Requirements

- **Minimum RAM**: 8GB (16GB recommended)
- **GPU Memory**: 4GB VRAM minimum (8GB+ recommended for best performance)
- **Disk Space**: ~10GB for models and dependencies

### Additional Notes

- The first run may take longer as models are downloaded
- Processing time depends on image size, number of steps, and hardware
- Results are saved in the `invsr_output/` directory (for single images) or in a subdirectory of the input folder (for batch processing)



## InvSR vs Our Model
<img width="742" height="706" alt="architecture" src="https://github.com/user-attachments/assets/19a24cf6-33b6-4c03-b383-8b6689b45f1e" />



## Key features implemented on the model

### 1. Adaptive Scheduler
- Automatically adjusts timesteps based on image complexity
- Location: `utils/util_adaptive.py`
- Usage: Enable "Adaptive Scheduler" checkbox in app.py

### 2. Attention-Guided Fusion
- Combines multiple results using attention maps
- Methods: 'weighted' (smooth) or 'max' (aggressive)
- Location: `utils/util_adaptive.py`
- Usage: Enable "Attention-Guided Fusion" checkbox

### 3. Smart Chopping
- Adaptive overlap based on local complexity
- Overlap range: 25-75% (configurable)
- Location: `utils/util_smart_chopping.py`
- Usage: Enable "Smart Chopping" checkbox

### 4. Hybrid Color Fixing
- Combines YCbCr, Wavelet, and Histogram Matching
- Modes: 'adaptive' (weighted) or 'best' (selection)
- Location: `utils/util_color_fix.py`
- Default: 'hybrid' method

### 5. Edge-Preserving Enhancement
- Enhances edges while preserving smooth regions
- Uses Sobel operators for edge detection
- Location: `utils/util_enhancement.py`
- Usage: Enable "Edge-Preserving Enhancement" checkbox

### 6. Adaptive Guidance Scale
- Adjusts cfg_scale based on image complexity
- Location: `utils/util_enhancement.py`
- Usage: Enable "Adaptive Guidance Scale" checkbox

## File Structure

```
src/
├── app.py                    # Gradio interface
├── sampler_invsr.py          # Main sampler
├── utils/                    # Utility modules
│   ├── util_adaptive.py      # Adaptive scheduler & attention fusion
│   ├── util_color_fix.py     # Color fixing methods
│   ├── util_enhancement.py   # Edge enhancement & adaptive guidance
│   ├── util_smart_chopping.py # Smart chopping
│   └── ...
├── basicsr/                  # BasicSR library
├── datapipe/                 # Dataset utilities
├── src/diffusers/            # Modified diffusers library
├── configs/                  # Configuration files
│   └── sample-sd-turbo.yaml  # Main config
└── weights/                  # Model weights
    ├── noise_predictor_sd_turbo_v5.pth
    ├── vgg16_sdturbo_lpips.pth
    └── models--stabilityai--sd-turbo/
```

## Results by Our Model
Observe how the aliasing (or 'jagged edges') present on the stairs in the following image is reduced.
<br>
<img width="499" height="331" alt="sample1" src="https://github.com/user-attachments/assets/7faa3285-c23d-4259-a11c-b1c19d0bb212" />
<br>
<br>
<img width="504" height="202" alt="sample2" src="https://github.com/user-attachments/assets/51351252-cfda-49c3-81dd-6c28cdf28452" />

## Evaluation Metrics
<img width="2196" height="2753" alt="metrics" src="https://github.com/user-attachments/assets/c5d1b903-8511-4dd1-ac8d-59a28f053540" />


### Important Notes

#### Differences Between Methods

**Hybrid Color Fixing vs Attention-Guided Fusion:**
- **Hybrid Color Fixing**: Global color correction (fixed weights or best method selection)
- **Attention-Guided Fusion**: Local pixel-level fusion based on attention maps (adaptive per region)

**When to Use:**
- Hybrid Color Fixing: During color fixing step (replaces single method)
- Attention Fusion: After color fixing (combines multiple results)

#### Pipeline Order

1. Image preprocessing
2. Adaptive scheduler (if enabled) - adjusts timesteps
3. Denoising loop
4. Color fixing (YCbCr/Wavelet/Histogram/Hybrid)
5. Attention-guided fusion (if enabled)
6. Edge-preserving enhancement (if enabled)
7. Output






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
