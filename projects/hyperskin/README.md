# `Synthesis of Hyperspectral Skin Lesion Images for Data Augmentation using Generative Models`
# `Síntese de um Dataset Hiperespectral de Lesões de Pele utilizando IA Generativa `

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

|Name  | RA | Specialization|
|--|--|--|
| Kristhian André Oliveira Aguilar  | 298976  | Computer Engineering|
| Ana Clara Caznok Silveira  | 231745  | Computer Engineering|
| Aline Yoshida Machado | 265732 | Physics Engineering|

# Abstract / Project Description
Hyperspectral Imaging (HSI) combines imaging and spectroscopy, giving each pixel a continuous spectrum across wavelengths. HSI captures how light interacts with molecules, as their composition, vibrations, and structure affect photon behavior. These light–matter interactions create distinct spectral patterns that act like unique “fingerprints” for each material. Thanks to its ability to distinguish different materials, tissues, and substances, Hyperspectral Imaging (HSI) has become a valuable tool in Remote Sensing, Agriculture, and Medicine. In medicine, its capacity to go beyond standard RGB imaging is mainly used to detect tumors. However, publicly available hyperspectral tumor datasets are scarce, which often leads melanoma classification models to overfit or perform poorly in subsampled classes. Therefore, the main goal of this project is to construct a generative ai model that creates a synthetic hyperspectral dermoscopy dataset. More specifically, we investigated how synthetic hyperspectral images affect the performance of a melanoma classifier, hoping that a classifier trained with both synthetic and real hyperspectral images, would outperform a classifier trained with only real images. 

To test this hypothesis, we trained generative models, including SHSGAN, DCGAN, FastGAN and VAE, to produce realistic hyperspectral melanoma images and evaluated their quality using spectral and perceptual metrics. Among them, FastGAN achieved the best balance between spectral accuracy and structural realism, generating synthetic lesions that closely resembled real samples. These synthetic images were then integrated into the training of melanoma classifiers based on DenseNet and ResNet architectures. The classifiers trained with both real and synthetic data outperformed those trained solely on real data, achieving higher validation accuracy (0.84 vs. 0.79) and F1-score (0.89 vs. 0.85).


# Main Goal
Therefore, the main goal of this project is to construct a generative ai model that learns the distribution of real hyperspectral images and through them is able to create a synthetic hyperspectral melanoma dataset. Desired output: a synthetic hyperspectral dataset of skin lesions and melanoma. 
## Main Hypothesis 
A classifier trained with synthetic AND real data will have better results than if only trained in real data

## Secondary Questions
This work also explores the general use of hyperspectral synthetic data to improve classification. In order to support the use of synthetic images we also ask the following questions: 
- Is synthetic data truly necessary to improve classification performance, or can conventional data augmentation methods achieve similar gains?
- Does the inclusion of synthetic samples enhance the performance of a classifier trained from scratch when only a small amount of real data is available?
- Does synthetic data improve the performance of a generalist classifier pretrained on large-scale datasets such as ImageNet?
- Does it also benefit a specialist classifier pretrained on RGB melanoma images?
- What is the optimal proportion of synthetic data to mix with real data during training?
- How does the quality of synthetic images influence downstream classifier performance?

## Presentation
### Slides
https://docs.google.com/presentation/d/16PthBsxWUrnjjb5saw3rDCTorvTyX7CcnbJKjngEmF8/edit?usp=sharing

## Dataset
- Github link: https://github.com/heugyy/HSIDermoscopy
- Dataset download link: https://drive.google.com/drive/folders/13ZXXxtUMjEzLAjoAqC19IJEc5RgpFf2f?usp=sharing

# Literature Overview on Hyperspectral Generative Models

#### SHS GAN [5]
The model receives as input a standard RGB image and its task is to generate a synthetic hyperspectral cube. The objective of the Generator is to learn a mapping from the RGB domain to the HS domain, so that the distribution of the synthetic HS cubes becomes similar to the distribution of real HS cubes. The RGB image is used as input to the Generator so that the synthetic HS cube preserves the spatial details and textures of the input image and also keeps the color properties coherent with what appears in the RGB. The Critic is trained to evaluate whether the generated HS cubes are realistic. It does so by analyzing spatial patterns and also the smoothness and shape of spectral curves, which are emphasized by looking at the data in both the image and Fourier-transformed spectral domains. In addition, the synthetic HS cube can be converted back into RGB using a deterministic transformation. This reconstructed RGB image is compared to the original input RGB, and differences are penalized during training. This step enforces consistency between the generated HS cube and the original RGB image. It is used a WGAN training pipeline

![SHS Architecture Diagram](images/SHS-GAN.png)

#### FastGAN
- **Minimalist Architecture:** The model uses a lightweight GAN structure with a single convolution layer per resolution and very few channels (e.g., three) at high resolutions ($\ge 512^2$) to ensure low computational cost and fast training.
- **Skip-Layer Excitation (SLE):** A core module designed for faster training. It strengthens the gradient flow by using feature maps from low-resolution layers (e.g., $8^2$) to perform channel-wise re-calibration (multiplication) on feature maps at high-resolution layers (e.g., $128^2$).
- **Self-Supervised Discriminator:** To prevent the discriminator (D) from overfitting on small datasets, it is regularized using a self-supervised task.
- **Auto-Encoding Regularization:** The discriminator is trained as an encoder, and small, auxiliary decoders are added to reconstruct intermediate feature maps back into images. This forces D to learn comprehensive and descriptive features from the real images.
- **Perceptual Loss ($\mathcal{L}_{recons}$):** The discriminator's auto-encoding regularization is achieved using a perceptual loss (LPIPS). This loss compares the reconstructed images generated by the discriminator's decoders against the original, forcing the discriminator to learn meaningful and comprehensive features rather than just memorizing.
- **Hinge Loss:** The model employs the hinge version of the adversarial loss for iteratively training the generator and discriminator, as it was found to compute the fastest with little performance difference from other losses.
- **Unsupervised Disentanglement:** A direct benefit of the SLE module is that the generator automatically learns to disentangle style and content, enabling style-mixing applications similar to StyleGAN without the added complexity.

#### Autoencoder
The autoencoder is composed by an encoder and a decoder. The encoder compresses the input HSI image into a lower-dimensional latent representation, while the decoder reconstructs the original image from this representation. A variational autoencoder (VAE) is a type of autoencoder that learns a probabilistic mapping from the input data to a latent space, allowing for the generation of new samples by sampling from this latent space. VAEs are especially adept at modeling complex, high-dimensional data and continuous latent spaces, making them extremely useful for tasks like generating diverse sets of similar images. Palsson et al. [2] used a VAE paired with a GAN framework to generate high-resolution synthetic hyperspectral images. Liu et al. [3] proposed a model inspired autoencoder (MIAE) to fuse low-resolution HSI with high-resolution RGB images to generate high-resolution HSI. We used an AutoEncoder with a Convolutional Encoder and a UNet decoder

# Workflow and Methodology
![workflow](images/fluxograma_completo.png)
Our methodology was designed to test whether the inclusion of synthetic hyperspectral images can improve tumor classification compared to training with only real data. And if so, what are the experimental conditions necessary for that improvement. As shown in the workflow, the process begins with the preprocessing of the hyperspectral dataset, where images are segmented and cropped in the region containing the lesion. The preprocessed data are then used to generate synthetic data and also train a classification model that classifies between malignant melanoma and dysplasic nevi. 

The generation stage is divided into two strategies. Unconditional models (SHSGAN, FastGAN, and VAE) learn directly from the hyperspectral lesions. Conditional models (RGB CycleGAN and RGB SpadeFastGAN) use the RGB image of the lesion as additional guidance to produce the hyperspectral output. Both strategies aim to create synthetic images that faithfully represent real tissue. We assess the realism of the generated hyperspectral cubes through SAM, SSIM, PSNR, and FID metrics. These measurements ensure the synthetic samples capture both the spectral signature and overall visual quality of real lesions.

To understand how synthetic images influence malignant tumor classification, we evaluate two groups of classifiers: a generalist DenseNet, which benefits from ImageNet pretraining, and a specialist EfficientNet-b6, which incorporates prior knowledge from an RGB melanoma model trained on MILK10K. Both architectures are also tested when trained from scratch.
Each classifier is trained twice. We begin by training the model only on real hyperspectral data to set a performance baseline. We then retrain the same architecture using an expanded dataset that includes both real and synthetic hyperspectral images. Comparing these two training conditions using metrics such as F1-score, Accuracy, and Specificity, reveals when and for which model types synthetic data provides a meaningful advantage.

## Dataset Description
| Dataset | Web Address | Descriptive Summary |
| :--- | :--- | :--- |
| A Hyperspectral Dermoscopy Dataset | https://github.com/heugyy/HSIDermoscopy | A dataset of 330 hyperspectral dermoscopy images for melanoma detection. Each image has 16 spectral bands (465 nm to 630 nm) and includes histopathologically validated cases of melanoma, dysplastic nevus, and other lesion types. |
* **Size:** The dataset contains a total of **330 hyperspectral images**.
* **Format:** Each raw image is converted into a **$256 \times 512 \times 16$ data cube**. Each image contains **16 spectral bands** covering the visible wavelength spectrum from **465 nm to 630 nm**. The paper also states that "Each band image is constituted of approximately $512 \times 272$ pixels".
* **Annotation:** All 330 images are **histopathologically validated**, meaning the diagnosis for each image was confirmed by pathologists after a biopsy.
* **Class Distribution:** 
  * Melanoma: 85 images.
  * Dysplastic Nevus: 175 images.
  * Other Lesions: 70 images.
* The "Other" category includes solar lentigo, IEC, nevi, and Seborrheic Keratosis.
  
## Data Preprocessing
#### Semi-Automatic Segmentation of Skin Lesions
* The primary goal of this step is to isolate skin lesions from the surrounding healthy skin in the hyperspectral images. This isolation is crucial for training the generative models, as it allows them to focus on learning the characteristics of the lesions without being influenced by irrelevant background information.
* Initially, to facilitate the creation of segmentation masks, each 16-channel hyperspectral image is converted into a false RGB PNG image. This is achieved by calculating the average value across all 16 channels and repeating this average three times to create a grayscale-like RGB image. These false RGB images are exclusively used for the segmentation step.
* We utilize Label Studio in conjunction with the Segment Anything Model 2 (SAM2) for semi-automatic annotation.
* For every false RGB image, a segmentation mask is created for each skin lesion present. SAM2 assists in efficiently outlining the lesions, and human annotators refine these masks to ensure accuracy. Each distinct lesion within an image receives its own individual mask. This segmentation process is applied to all 330 images in the dataset.

#### Cropped Hyperspectral Image Generation
* After the segmentation masks are generated, we return to the original 16-channel hyperspectral images (not the false RGB versions).
* For every segmented skin lesion mask within an image, we generate a corresponding cropped hyperspectral image, obtained from the bounding box of the segmentation mask.
* To capture relevant neighboring skin information crucial for classification, the bounding box is scaled by a 50% margin. This ensures that the cropped image includes not just the lesion but also a surrounding area of skin.

## Evaluating synthesis results
We would like for the generated images to be: clear, realistic and useful. 
- Image Clarity/Quality : Peak Signal-to-Noise Ratio (PSNR)
- Image realism : Spectral Angle Mapper (SAM) for average melanoma spectral signature , SSIM with real images, adapted FID 
- Usability: Given a baseline classifier that classifies images into melanoma and not melanoma, first train the classifier with only real data then with real + synthetic data and see if F1 score improves. Then, train only on synthetic data and test on real data to see if classifier performs similarly 
---
Here are the following explanations for the most used metrics
####  Structural Similarity Index Measure (SSIM)
Measures the structural similarity between two images, focusing on luminance, contrast, and structural patterns.

**Equation:**
$`SSIM(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}`$
where:
- $`\mu_x, \mu_y`$ are the means of images $x$ and $y$
- $`\sigma_x^2, \sigma_y^2`$ are their variances
- $`\sigma_{xy}`$ is their covariance
- $`C_1, C_2`$ are small constants to avoid division by zero
- Range: **[0, 1]**
- **SSIM ≈ 1** → high structural similarity  
- **SSIM ≈ 0** → weak similarity  

#### Peak Signal-to-Noise Ratio (PSNR)
Quantifies image reconstruction quality in terms of pixel-wise fidelity, how much noise or distortion is present compared to a reference image.

**Equation:**
$` PSNR(x, y) = 10 \log_{10}\left( \frac{L^2}{MSE} \right) `$
with
$`MSE = \frac{1}{N}\sum_{i=1}^{N}(x_i - y_i)^2`$
where $`L`$ is the maximum possible pixel value (e.g., 1.0 or 255).
- Higher PSNR → better image quality
- Typical values:
  - \> 40 dB → excellent
  - 30–40 dB → good
  - < 30 dB → degraded or noisy

#### Fréchet Inception Distance (FID)
Measures the distributional distance between real and generated image features extracted from a deep network (Inception-v3).  
It evaluates how close the overall statistics of generated images are to the real ones.
In our context, we must use an adapted FID, once the pre trained weights are fit for a 3-channel RGB input. Since we have a 16 channel image, it is not possible to perform the inference of the model. Therefore, we used the Inception V3 model with the excpetion of the first layer. This layer, we adapted to a 16-channel input by replicating the kernel weights untill it reached the desired channel.

**Equation:**
$`FID = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)`$
where:
- $`\mu_r, \Sigma_r`$: mean and covariance of features from **real images**
- $`\mu_g, \Sigma_g`$: mean and covariance of features from **generated images**
- Lower FID → better quality and diversity

#### Spectral Angle Mapper (SAM)
**Purpose:**  
Used for hyperspectral images, SAM measures the spectral similarity between two spectra (one per pixel) by computing the angle between their spectral vectors.

**Equation:**
$`SAM(x, y) = \arccos\left(\frac{x \cdot y}{\|x\| \, \|y\|}\right)`$
where $x$ and $y$ are spectral vectors of a pixel in the reference and generated images.
- **Units**: radians or degrees
- **Lower SAM** → higher spectral similarity

#### Summary Table

| Metric | Domain | Range | Better When | Evaluates |
|:-------|:--------|:-------|:--------------|:-------------|
| **SSIM** | Spatial | [-1, 1] | ↑ | Structural similarity |
| **PSNR** | Spatial | [0, ∞) dB | ↑ | Pixel-wise fidelity |
| **FID** | Feature / Perceptual | [0, ∞) | ↓ | Realism & diversity |
| **SAM** | Spectral | [0°, ∞°) | ↓ | Spectral shape similarity |
---

# Experiments, Results, and Discussion of Results
Our aim in this work is to offer a comprehensive understanding of how synthetic images influence classification. For that reason, we examine both our main hypothesis and the related secondary questions. This session is divided into two main components. The first describes the experimental process behind generating synthetic hyperspectral images, and the second examines the main and secondary research questions that this work aims to answer.

### Generative Model Training

#### Data Augmentations During Generative Model Training
During the model training phase, the cropped hyperspectral images undergo a series of augmentations using the `albumentations` library. These augmentations are vital for addressing data scarcity and improving the model's generalization capabilities.
The applied augmentations include:
* **ShiftScaleRotate:** Randomly shifts, scales, and rotates the image.
* **RandomCrop:** Takes a random crop from the image.
* **VerticalFlip:** Flips the image vertically.
* **HorizontalFlip:** Flips the image horizontally.
After the augmentations are applied, the images undergo a final transformation to standardize their size and shape for model input:
* **SmallestMaxSize:** The image is resized such that its smallest dimension matches a predefined target size (e.g., 224 pixels).
* **CenterCrop:** A center crop is then applied to make the image perfectly square (e.g., 224x224 pixels) and of the appropriate dimensions for the input layer of the neural network model.
Additionally, each channel of the hyperspectral images is normalized using min-max normalization based on pre-calculated global minimum and maximum values for each channel across the entire dataset. This normalization ensures that the pixel values for each channel are scaled to a consistent range, typically between 0 and 1, which is crucial for stable and efficient training of the neural network models.

#### SHS GAN
The SHS-GAN experiment was initially used to synthesize 64×64 hyperspectral images of melanoma. The main goal was to reproduce and understand the implementation described in the reference paper, which proposed several techniques for handling hyperspectral data.
We began with a baseline DCGAN.  
The Generator consists of a sequence of transposed convolutional blocks that progressively upsample a latent vector into an image. Each block includes a `ConvTranspose2d`, followed by `BatchNorm2d` and `ReLU` activations, except for the final layer, which uses a `Tanh` activation. The architecture adapts to different image sizes (28×28, 64×64, 256×256) by adjusting the depth and number of filters. All weights are initialized with a normal distribution to ensure stable training.  
The Discriminator mirrors this structure using standard convolutional layers for downsampling. Each block includes `Conv2d`, optional `BatchNorm2d`, and `LeakyReLU` activations, with the last layer outputting a single scalar (without activation).
Next, we progressively introduced the main components of SHS-GAN and evaluated their impact on image synthesis, adapting each modification to our context. In the first experiment, we replaced the 2D convolutional filters in the discriminator with 3D convolutions. Since hyperspectral data includes an additional spectral dimension, convolving in three dimensions allows the network to capture spectral correlations that 2D convolutions cannot. In the second experiment, we replaced **Batch Normalization** with **Spectral Normalization**, a regularization method that stabilizes GAN training. It constrains the spectral norm (the largest singular value of each layer’s weight matrix) to 1, thereby controlling the Lipschitz constant of the network and improving stability.In the third experiment, we added a spectral-frequency arm, which processes the same hyperspectral cube after applying a Fast Fourier Transform (FFT) along the spectral dimension. The resulting spectral-frequency representation is then combined with the spatial arm (which contains standard convolutions), as proposed in the reference model.During training, we observed that the model was highly sensitive to hyperparameters, and only a narrow set of configurations produced realistic results. Some combinations generated pure noise, while others yielded better synthetic images.  
The batch size had a strong influence: given our dataset of approximately 70 melanoma images, small batch sizes (1, 2, or 4) produced plausible results, while larger batch sizes (≥16) led to unstable outputs and noise.

The training followed the WGAN formulation, using two key hyperparameters: `gradient_penalty` and `n_critic`.  The gradient penalty enforces the Lipschitz continuity constraint on the discriminator, preventing it from developing excessively steep gradients. This results in smoother and more realistic training dynamics, reducing mode collapse and improving convergence. The `n_critic` parameter defines how many times the critic is updated per generator update—commonly greater than one—to ensure the critic accurately estimates the Wasserstein distance before each generator step.  
In our setup, we used a gradient penalty of 10 and n_critic = 2. Another crucial hyperparameter was the learning rate, set to approximately **1×10⁻⁵**.  
Higher learning rates destabilized training, causing the generated images to fluctuate drastically across epochs, while extremely low rates resulted in persistent noise even after many epochs. Overall, our experiments indicate that incorporating 3D convolutional layers in the discriminator improves the synthesis of spectral characteristics, as shown in the comparison below. The spectral profiles generated using 3D convolutions are more consistent with the real data than those obtained with 2D convolutions.

![Comparison Spectral Axis between 3D and 2D conv](images/3d_2d_spectral_comparison.png)

However, spectral normalization and the addition of the FFT arm did not lead to noticeable improvements.  
While further optimization might help, we believe this outcome is related to the limited spectral depth of our dataset (only a few channels), in contrast to the reference [6], which used data with 29 spectral bands.  
In our case, additional mechanisms designed to enhance spectral relationships may not have a significant impact given the lower spectral resolution.

![Metrics for SHS-GAN](images/SHS-GAN-metric.png)

We expect the quality of synthetic hyperspectral images to improve with future generator optimizations.  
Our next step is to use RGB images as inputs to the generator instead of random noise and include a reconstruction loss between the generated and real RGB representations.  
This approach encourages the generator to learn meaningful spatial and color relationships, using them to reconstruct high-quality hyperspectral data.

![Exp1](images/2d-conv-plot.png)  
![Exp2](images/3d-convolution-plot.png)  
![Exp3](images/3d-conv-sn.png)  
![Exp4](images/3D-conv-sn-fft.png)
#### FastGAN
The FastGAN experiment was conducted to generate synthetic hyperspectral images of melanoma lesions using a 16-channel input configuration and an image size of 256×256 pixels. The model was trained with a learning rate of 0.0002 and a latent dimension of 256, following the original FastGAN training procedure that includes manual optimization, exponential moving average updates, and perceptual consistency losses. The goal was to evaluate how well the generator could reproduce realistic skin lesion patterns and retain spectral properties similar to real melanoma samples.  

Quantitatively, the model reached a Frechet Inception Distance (FID) of about 114.7, indicating a moderate difference between the distributions of real and generated images. The Spectral Angle Mapper (SAM) of 0.17 shows that the spectral shapes of synthetic images were reasonably aligned with those of real samples, meaning that the generator could preserve band relationships across the hyperspectral channels. The Relative Average Spectral Error (RASE) value, however, was high at around 3634, reflecting differences in reflectance magnitude between synthetic and real images. The Structural Similarity Index Measure (SSIM) of 0.67 suggests that the generated lesions shared similar overall spatial structures with real ones, although fine details and boundary sharpness were less well captured. The total variation metric indicated that the generated results were relatively smooth, with less noise but also less textural variation than real images. 

![Spectral Comparison](images/fastgan_sample_comparison.png)

Visually, the comparison between real and synthetic melanoma images highlights several encouraging patterns. The synthetic lesions reproduced the typical round or irregular shapes of melanoma and maintained appropriate global contrast between lesion cores and surrounding skin. The overall appearance of the generated images was realistic enough to resemble natural skin textures, though a lack of diversity between generated samples and some intensity discrepancies were noted. Despite that, the model clearly learned the underlying structure and general appearance of melanoma lesions.

![Mean Spectra](images/fastgan_mean_spectra_507_faf8cbbe2c81f99f74d4.png)

The spectral analysis supported these visual findings. The plotted mean spectra showed that the synthetic and real data followed nearly the same trends across most wavelengths. Normal skin spectra overlapped closely, demonstrating that the model learned background reflectance behavior, while melanoma lesion spectra exhibited similar shapes but slightly shifted magnitudes. This agrees with the spectral metrics and suggests that FastGAN was able to reproduce physically plausible spectral patterns.  

Overall, the experiment demonstrates that FastGAN is a viable architecture for hyperspectral skin lesion synthesis. The generated images capture the main structural and spectral traits of melanoma lesions, providing a realistic extension of the training data space. While further optimization is needed to improve texture detail, reduce intensity discrepancies and increase sample diversity, these results show promising potential for using generative adversarial methods to augment hyperspectral datasets and support skin cancer research.

#### VAE Autoencoder 

Similarly as the FastGAN, VAE autoencoder was trained with a 16-channel input configuration and an image size of 256×256 pixels. The model was trained with a learning rate of 0.0002 and a latent dimension of 64. Loss function was set to have a term with a KL-divergence regularizer weighted by kld_weight = 1×10⁻², encouraging smooth, semantically meaningful latents while preserving spectral fidelity. Overall the results look like melanoma images but lack the details present in a realistic hyperspectral image. Spectral similarity was also achieved. 
![vaeimages](images/vae-results.png)
![vae_spectra](images/vae_spectra.png)

## Classifier Training with Synthetic Data 

### Is synthetic data truly necessary to improve classification performance, or can conventional data augmentation methods achieve similar gains?

The first series of classification experiments aimed to determine whether traditional data balancing strategies could effectively improve melanoma classification, or if a synthetic hyperspectral dataset would be necessary to achieve better performance—especially for underrepresented classes. Experiments were performed in total using the **DenseNet201** architecture trained from scratch. Among them, representative runs are detailed below. These tests compared different balancing strategies—Focal Loss, Batch Regularization, and their combination—against a baseline trained with no balancing method.

The baseline model performed well, achieving 0.8852 F1 and 0.6429 specificity, and it remained the best result among all experiments that used only real data.
Real-data augmentations and balancing strategies did not outperform the baseline. Batch Regularization and Focal Loss reduced either F1-score or specificity, and their combination produced the weakest results. Even extensive augmentations such as rotation, equalization, normalization, and flipping reached only 0.8667 F1, slightly below the baseline.

The best performance came from using synthetic images. Training with synthetic hyperspectral data alone produced the highest F1-score (0.9000) and the highest specificity (0.7143). When synthetic samples were combined with traditional augmentations, results remained strong and continued to outperform the real-only augmented model.

# Experiment Table

| ID | Augmentation                       | Synthetic data | Val F1  | Specificity | 
|:--:|:-----------------------------------|:--------------:|:-------:|:-----------:|
| 1  | NONE                               | No             | 0.8852  | 0.6429      |
| 2  | Batch Reg                          | No             | 0.8571  | 0.5000      |
| 3  | Focal Loss                         | No             | 0.8889  | 0.5714      |
| 4  | Batch Reg + Focal Loss             | No             | 0.8750  | 0.5000      |
| 5  | NONE                               | Yes            | 0.9000  | 0.7143      |
| 6  | Rotate90 + Equalize + Norm + Flip  | No             | 0.8667  | 0.6429      |
| 7  | Rotate90 + Equalize + Norm + Flip  | Yes            | 0.8814  | 0.7143      |


The experiments showed that traditional data balancing methods did not enhance the model’s performance. The highest F1-score and balanced sensitivity-specificity tradeoff were obtained when no balancing strategy was used. DenseNet201 performed best with the natural data distribution, suggesting that the limited dataset size and class imbalance hindered the effectiveness of focal loss and regularization techniques.

### Does the inclusion of synthetic samples enhance the performance of a classifier trained from scratch when only a small amount of real data is available?

The objective of this experiment was to verify whether training a hyperspectral dermoscopy skin lesion classifier with additional synthetic melanoma data, generated using the FastGAN architecture, could improve performance in distinguishing melanoma from dysplastic nevi. The classifier was trained on the cropped hyperspectral images, the only difference between the two experiments being that the second training included synthetic melanoma samples to balance the dataset, ensuring that the number of melanoma and dysplastic nevi instances was equal.  

| Metric (On Validation Set) | Without Synthetic Data | With Synthetic Data |
|:--|:--:|:--:|
| Accuracy | 0.79 | **0.84** |
| Best Accuracy | 0.81 | **0.86** |
| F1-score | 0.85 | **0.89** |
| Precision | **0.83** | 0.81 |
| Recall | 0.86 | **1.00** |
| Specificity (at Sensitivity 0.95) | **0.37** | 0.32 |
| Loss | 0.53 | **0.51** |

The results showed that the model trained with real data only reached a validation accuracy of approximately 0.79, with a best recorded validation accuracy of 0.81. Its validation F1-score was 0.85, precision 0.83, and recall 0.86. On the other hand, when trained with synthetic data, the model achieved higher overall performance, reaching a validation accuracy of 0.84 and a best validation accuracy of 0.86. The F1-score increased to 0.89, and although validation precision slightly decreased from 0.83 to 0.81, recall improved dramatically from 0.86 to 1.00. The validation loss also decreased slightly, indicating more stable learning, while the specificity at a sensitivity of 0.95 dropped moderately from 0.37 to 0.32. These results demonstrate that the model trained with synthetic data achieved stronger classification performance, particularly in detecting melanoma cases, without overfitting or losing generalization.  

The inclusion of synthetic melanoma samples effectively mitigated the class imbalance that typically biases the network toward the majority class. By generating a balanced training distribution, the model became substantially more sensitive to melanoma patterns, increasing its ability to detect malignant lesions. This improvement, however, came with a modest reduction in specificity. While this means the classifier produced more false-positive melanoma predictions, it also ensured that all melanoma cases were correctly detected. In medical imaging scenarios, particularly in melanoma screening, higher sensitivity is often prioritized since missing a malignant case is far more critical than a false alarm. Therefore, the trade-off observed here is generally acceptable from a clinical standpoint.  

From a training perspective, balancing the data appeared to stabilize convergence and reduce loss variability between epochs. The model with synthetic data reached higher accuracy and lower loss after more epochs, suggesting that the GAN-generated samples helped the network generalize better across the minority class. The improved F1-score reflects a more balanced classification behavior, confirming that synthetic hyperspectral melanoma data played a beneficial role in improving both learning stability and predictive fairness.  

In summary, integrating FastGAN-generated hyperspectral melanoma samples into the training set led to a measurable improvement in classification performance. The balanced model achieved perfect recall and higher overall validation accuracy compared to the model trained on real data alone. Although the slight drop in specificity indicates a small increase in false positives, the results strongly suggest that GAN-based data augmentation is an effective strategy for addressing class imbalance in hyperspectral dermoscopy classification tasks.  

### Does synthetic data improve the performance of a generalist classifier pretrained on large-scale datasets such as ImageNet?

### Does it also benefit a specialist classifier pretrained on RGB melanoma images?

### What is the optimal proportion of synthetic data to mix with real data during training?

### How does the quality of synthetic images influence downstream classifier performance?


# Conclusion
Our results show that classifiers trained with synthetic hyperspectral data perform better than those relying only on standard data augmentation or regularization. Adding GAN-generated melanoma images helped the models generalize better and improved sensitivity, especially for underrepresented classes. We also found that architectures built for high-resolution image generation, like FastGAN, work better than those designed specifically for hyperspectral synthesis—likely because our dataset has higher spatial resolution (256×256) and fewer spectral channels than typical HSI data. Overall, GANs outperformed VAEs on small, high-resolution datasets, producing sharper and more realistic lesion details.
Next, we plan to test classification models trained purely on synthetic data against real samples, use large RGB skin lesion datasets to generate more diverse images, and explore CycleGAN and pretraining strategies to improve realism. Conditioning FastGAN on RGB images or binary masks may also help control lesion structure and further boost the quality of generated hyperspectral data.


## Bibliographic References
1. D. A. Abuhani, I. Zualkernan, R. Aldamani and M. Alshafai, "Generative Artificial Intelligence for Hyperspectral Sensor Data: A Review," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 6422-6439, 2025, doi: https://doi.org/10.1109/JSTARS.2025.3538759.
2. Palsson, Burkni, Magnus O. Ulfarsson, and Johannes R. Sveinsson. 2023. "Synthesis of Synthetic Hyperspectral Images with Controllable Spectral Variability Using a Generative Adversarial Network" Remote Sensing 15, no. 16: 3919. https://doi.org/10.3390/rs15163919
3. Liu, J., Wu, Z., Xiao, L., & Wu, X. J. (2022). Model Inspired Autoencoder for Unsupervised Hyperspectral Image Super-Resolution. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-12. https://doi.org/10.1109/tgrs.2022.3143156
4. L. Zhu, Y. Chen, P. Ghamisi and J. A. Benediktsson, "Generative Adversarial Networks for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 9, pp. 5046-5063, Sept. 2018, doi: 10.1109/TGRS.2018.2805286.
5. J. Hauser, G. Shtendel, A. Zeligman, A. Averbuch and M. Nathan, "SHS-GAN: Synthetic Enhancement of a Natural Hyperspectral Database," in IEEE Transactions on Computational Imaging, vol. 7, pp. 505-517, 2021, doi: 10.1109/TCI.2021.3079818.
6. B. Liu, Y. Zhu, K. Song, and A. Elgammal, "Towards Faster and Stabilized GAN Training for High-Fidelity Few-Shot Image Synthesis," arXiv preprint arXiv:2101.04775, 2021. Available: https://arxiv.org/abs/2101.04775.
