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

## Abstract / Project Description
Hyperspectral Imaging (HSI) combines imaging and spectroscopy, giving each pixel a continuous spectrum across wavelengths. HSI captures how light interacts with molecules, as their composition, vibrations, and structure affect photon behavior. These light–matter interactions create distinct spectral patterns that act like unique “fingerprints” for each material. Thanks to its ability to distinguish different materials, tissues, and substances, Hyperspectral Imaging (HSI) has become a valuable tool in Remote Sensing, Agriculture, and Medicine. In medicine, its capacity to go beyond standard RGB imaging is mainly used to detect tumors. However, publicly available hyperspectral tumor datasets are scarce, which often leads melanoma classification models to overfit or perform poorly in subsampled classes. Therefore, the main goal of this project is to construct a generative ai model that creates a synthetic hyperspectral dermoscopy dataset. More specifically, we hope that a classifier trained with both synthetic and real hyperspectral images, outperform a classifier trained with only real images. 

To test this hypothesis, we trained generative models—including SHSGAN, DCGAN, FastGAN, and VAE—to produce realistic hyperspectral melanoma images and evaluated their quality using spectral and perceptual metrics. Among them, FastGAN achieved the best balance between spectral accuracy and structural realism, generating synthetic lesions that closely resembled real samples. These synthetic images were then integrated into the training of melanoma classifiers based on DenseNet and ResNet architectures. The classifiers trained with both real and synthetic data outperformed those trained solely on real data, achieving higher validation accuracy (0.84 vs. 0.79) and F1-score (0.89 vs. 0.85), along with perfect recall for melanoma detection.

### Main goal
Therefore, the main goal of this project is to construct a generative ai model that learns the distribution of real hyperspectral images and through them is able to create a synthetic hyperspectral melanoma dataset.
Desired output: a synthetic hyperspectral dataset of skin lesions and melanoma. 

#### Main Hypothesis 
A classifier trained with synthetic AND real data will have better results than if only trained in real data

### Presentation
#### Slides
https://docs.google.com/presentation/d/1lx3-yT1-Smwg8uxjNo1hsLG3ESJFPZldXnlBdIy1wgo/edit?usp=sharing

### Dataset
- Github link: https://github.com/heugyy/HSIDermoscopy
- Dataset download link: https://drive.google.com/drive/folders/13ZXXxtUMjEzLAjoAqC19IJEc5RgpFf2f?usp=sharing

## Proposed Methodology
> For the first submission, the proposed methodology must clarify:  
> * Which dataset(s) the project intends to use, justifying the choice(s).
> * Which generative modeling approaches the group already sees as interesting to be studied.  
> * Reference articles already identified and that will be studied or used as part of the project planning.  
> * Tools to be used (based on the group’s current vision of the project).  
> * Expected results.  
> * Proposal for evaluating the synthesis results.
### Tools to be used
The project will be developed using Python, with the following libraries:
- TensorFlow/Keras or PyTorch for building and training the generative models.
- NumPy, Pandas and matplotlib for data exploration, manipulation and analysis.
- Scikit-learn for implementing classic machine learning algorithms and evaluation metrics.
- OpenCV and PIL for image processing tasks.
- Jupyter Notebooks for experimentation.
- [pytorch-lightning-template](https://github.com/DavidZhang73/pytorch-lightning-template/tree/main) to structure the code in a modular and organized way. This template includes:
  - `wandb` for experiment tracking and visualization.
  - `hydra` for configuration management.
  - `pytorch-lightning` for simplifying the training loop and model management.
  - `torchmetrics` for easy access to common metrics.

### Generative Models

#### SHS GAN [5]

- The model receives as input a standard RGB image and its task is to generate a synthetic hyperspectral cube. The objective of the Generator is to learn a mapping from the RGB domain to the HS domain, so that the distribution of the synthetic HS cubes becomes similar to the distribution of real HS cubes.

- the RGB image is used as input to the Generator so that the synthetic HS cube preserves the spatial details and textures of the input image and also keeps the color properties coherent with what appears in the RGB.

- The Critic is trained to evaluate whether the generated HS cubes are realistic. It does so by analyzing spatial patterns and also the smoothness and shape of spectral curves, which are emphasized by looking at the data in both the image and Fourier-transformed spectral domains.

- In addition, the synthetic HS cube can be converted back into RGB using a deterministic transformation. This reconstructed RGB image is compared to the original input RGB, and differences are penalized during training. This step enforces consistency between the generated HS cube and the original RGB image.

- It is used a WGAN training pipeline

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
- The autoencoder is composed by an encoder and a decoder. The encoder compresses the input HSI image into a lower-dimensional latent representation, while the decoder reconstructs the original image from this representation. A variational autoencoder (VAE) is a type of autoencoder that learns a probabilistic mapping from the input data to a latent space, allowing for the generation of new samples by sampling from this latent space.
- VAEs are especially adept at modeling complex, high-dimensional data and continuous latent spaces, making them extremely useful for tasks like generating diverse sets of similar images.
- Palsson et al. [2] used a VAE paired with a GAN framework to generate high-resolution synthetic hyperspectral images.
- Liu et al. [3] proposed a model inspired autoencoder (MIAE) to fuse low-resolution HSI with high-resolution RGB images to generate high-resolution HSI.
- We used an AutoEncoder with a Convolutional Encoder and a UNet decoder


### Evaluating synthesis results



We would like for the generated images to be: clear, realistic and useful. 
- Image Clarity/Quality : Peak Signal-to-Noise Ratio (PSNR)
- Image realism : Spectral Angle Mapper (SAM) for average melanoma spectral signature , SSIM with real images, adapted FID 
- Usability: Given a baseline classifier that classifies images into melanoma and not melanoma, first train the classifier with only real data then with real + synthetic data and see if F1 score improves. Then, train only on synthetic data and test on real data to see if classifier performs similarly 

---
Here are the following explanations for the most used metrics
####  Structural Similarity Index Measure (SSIM)

Measures the structural similarity between two images, focusing on luminance, contrast, and structural patterns.

**Equation:**

$$
SSIM(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

where:
- $\mu_x, \mu_y$ are the means of images $x$ and $y$
- $\sigma_x^2, \sigma_y^2$ are their variances
- $\sigma_{xy}$ is their covariance
- $C_1, C_2$ are small constants to avoid division by zero
- Range: **[0, 1]**
- **SSIM ≈ 1** → high structural similarity  
- **SSIM ≈ 0** → weak similarity  


#### Peak Signal-to-Noise Ratio (PSNR)

Quantifies image reconstruction quality in terms of pixel-wise fidelity, how much noise or distortion is present compared to a reference image.

**Equation:**

$$
PSNR(x, y) = 10 \log_{10}\left( \frac{L^2}{MSE} \right)
$$

with

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(x_i - y_i)^2
$$

where $L$ is the maximum possible pixel value (e.g., 1.0 or 255).


- Higher PSNR → better image quality
- Typical values:
  - > 40 dB → excellent
  - 30–40 dB → good
  - < 30 dB → degraded or noisy

---

#### Fréchet Inception Distance (FID)
Measures the distributional distance between real and generated image features extracted from a deep network (Inception-v3).  
It evaluates how close the overall statistics of generated images are to the real ones.

In our context, we must use an adapted FID, once the pre trained weights are fit for a 3-channel RGB input. Since we have a 16 channel image, it is not possible to perform the inference of the model. Therefore, we used the Inception V3 model with the excpetion of the first layer. This layer, we adapted to a 16-channel input by replicating the kernel weights untill it reached the desired channel.

**Equation:**
$$
FID = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

where:
- $\mu_r, \Sigma_r$: mean and covariance of features from **real images**
- $\mu_g, \Sigma_g$: mean and covariance of features from **generated images**

- Lower FID → better quality and diversity



#### Spectral Angle Mapper (SAM)

**Purpose:**  
Used for hyperspectral images, SAM measures the spectral similarity between two spectra (one per pixel) by computing the angle between their spectral vectors.

**Equation:**
$$
SAM(x, y) = \arccos\left(\frac{x \cdot y}{\|x\| \, \|y\|}\right)
$$

where $x$ and $y$ are spectral vectors of a pixel in the reference and generated images.

- Units: radians or degrees**
- **Lower SAM → higher spectral similarity


#### Summary Table

| Metric | Domain | Range | Better When | Evaluates |
|:-------|:--------|:-------|:--------------|:-------------|
| **SSIM** | Spatial | [-1, 1] | ↑ | Structural similarity |
| **PSNR** | Spatial | [0, ∞) dB | ↑ | Pixel-wise fidelity |
| **FID** | Feature / Perceptual | [0, ∞) | ↓ | Realism & diversity |
| **SAM** | Spectral | [0°, ∞°) | ↓ | Spectral shape similarity |

---


### Dataset Description
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

### Workflow
![workflow](images/hypersynth_flux.png)
Our methodology was designed to test whether the inclusion of synthetic hyperspectral images can improve tumor classification compared to training with only real data. As shown in the workflow, the process begins with the preprocessing of the hyperspectral dataset, where images are segmented and cropped in the region containing the lesion. The preprocessed data are then used in a generation stage, where four generative models—SHSGAN, DCGAN, FastGAN, and VAE—are trained to produce synthetic hyperspectral tumor images. These models learn the complex spectral and spatial characteristics of malignant tumors, generating synthetic melanoma samples as close to real as possible. The quality of these synthetic images is evaluated using standard generation metrics, including the Spectral Angle Mapper (SAM), Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Fréchet Inception Distance (FID), which assess both spectral and perceptual similarity to the real data.

Following image generation, two classification models are trained to distinguish malignant from benign tumors. The first classifier is trained exclusively with real hyperspectral images, while the second combines real and synthetic images in its training set. For both cases, two deep convolutional architectures, DenseNet and ResNet, are employed, each trained under two conditions: using pre-trained RGB weights or from scratch directly on hyperspectral data. The performance of each classifier is assessed using classification metrics such as F1-score, Accuracy, and SpecAtSens (Specificity at Sensitivity).


## Schedule
![Project Schedule](images/schedule.png)


## Experiments, Results, and Discussion of Results
> In each topic describe the experiments carried out, the results obtained, and a brief discussion of the results.
### Data Preprocessing
#### Semi-Automatic Segmentation of Skin Lesions
* The primary goal of this step is to isolate skin lesions from the surrounding healthy skin in the hyperspectral images. This isolation is crucial for training the generative models, as it allows them to focus on learning the characteristics of the lesions without being influenced by irrelevant background information.
* Initially, to facilitate the creation of segmentation masks, each 16-channel hyperspectral image is converted into a false RGB PNG image. This is achieved by calculating the average value across all 16 channels and repeating this average three times to create a grayscale-like RGB image. These false RGB images are exclusively used for the segmentation step.
* We utilize Label Studio in conjunction with the Segment Anything Model 2 (SAM2) for semi-automatic annotation.
* For every false RGB image, a segmentation mask is created for each skin lesion present. SAM2 assists in efficiently outlining the lesions, and human annotators refine these masks to ensure accuracy. Each distinct lesion within an image receives its own individual mask. This segmentation process is applied to all 330 images in the dataset.

#### Cropped Hyperspectral Image Generation
* After the segmentation masks are generated, we return to the original 16-channel hyperspectral images (not the false RGB versions).
* For every segmented skin lesion mask within an image, we generate a corresponding cropped hyperspectral image, obtained from the bounding box of the segmentation mask.
* To capture relevant neighboring skin information crucial for classification, the bounding box is scaled by a 50% margin. This ensures that the cropped image includes not just the lesion but also a surrounding area of skin.

#### Data Augmentations During Training
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

### Classifier Training

### Generative Model Training

#### SHS GAN
The SHS GAN experiment was used to synthesize 64x64 melanoma hyperspectral images at first. The main idea was to understand what was implemented in the reference article, which had lots of methods to deal with hyperspectral data. 

At first we used a simple DCGAN. The Generator is composed of a sequence of transposed convolutional blocks that progressively upsample a latent vector into an image. Each block includes a ConvTranspose2d, followed by BatchNorm2d and ReLU activations, except for the last layer, which uses a Tanh activation. The architecture adapts to different image sizes (28×28, 64×64, 256×256) by varying depth and number of filters, and all weights are initialized with a normal distribution for stable training.
The Discriminator mirrors this structure with standard convolutional layers that downsample the image. Each block uses Conv2d, optional BatchNorm2d, and LeakyReLU activations, with the last layer outputting a single scalar without activation.

We add progressively characteristics of SHS GAN and evaluate the main differences it caused on the image, adapting to our context. For the first experiment we evaluated the implementations done on the Discriminator. The first one is replacing the 2D Convolution filter with a 3D convolution. Since we are dealing with an additional spectral channel, convolving in a third dimension might capture spectral relations that a 2D convolution does not.

The second experiment was replacing the Batch Normalization with the Spectral Normalization, which is a regularization technique introduced to stabilize the training of GANs. It works by constraining the spectral norm, considered the largest singular value of each layer’s weight matrix to 1, effectively controlling the Lipschitz constant of the network.

The third experiments was adding the the spectral-frequency arm, which receives the same HS cube after applying a Fast Fourier Transform along the spectral dimension and combined with the spatial arm, containing simple convolutions.

The first observation was that the training was very sensitive to several hyperparameters, and there were a few selections that resulted in pure noise, while the others generated more accurate representations. The first hyperparameter that changed the course of the generation was the batch size. Considering our dataset consisted of approximately 70 images of melanoma, batches of 1,2 and 4 were able to create synthetic data. On the contrary to batch size 16 and higher, which resulted in pure noise.

The training was performed using a WGAN, which meant that we used gradient_penalty and n_critic as hyperparemeters. The gradient penalty term is added to the discriminator loss to enforce the Lipschitz continuity constraint, which stabilizes training by preventing the critic from having excessively steep gradients. This helps the model produce smoother and more realistic distributions, avoiding mode collapse and improving convergence. The n_critic parameter defines how many times the critic is updated for each generator update—usually greater than one—so that the critic can better approximate the Wasserstein distance before the generator is adjusted. For our experiments we used gradient penalty equal to 10, and n_critic was 2.

A very important hyperparameter was the learning rate. We used a learning rate in the order of 1e-5. Higher learning rates distabilized the trainig so for each epoch the images changed drastically, and extremely low learning rates generated noise even after several epochs.

Regarding our experiments, we concluded that using 3D convolutional layers in the discriminator of gan is helpful to synthesize spectral characteristics, given that the spectral pattern of this version is more similar than using 2d conv layers.

![Comparison Spectral Axis between 3D and 2D conv](images/3d_2d_spectral_comparison.png)

 However spectral normalization and the introduction of an FFT component on the discriminator did not improve the results. It might require some optimization process considering the hyperparemeters are very sensitive to training, however what we believe is that our reference[6] uses data with 29 channels, which is far above what we have and perhaps these additional characteristics to enphazise spectral relations might not cause significant change in our context.

 ![Metrics for SHS-GAN](images/SHS-GAN-metric.png)


However, we believe that with the introduction of the optimizations made in the generator in the next steps of the project, the quality of the synthetich HSI data will improve. The main idea is to use RGB images as input to the generator instead of noise, and during the training we calculate a reconstruction Loss of HSI->RGB and add in the cost function. This way the generator learns spatial and color information and use to reconstruct HSI images.



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

### Classifier Training with Synthetic Data
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

> In the partial project submission (E2), this section may contain partial results, explorations of implemented solutions, and  
> discussions about such experiments, including decisions to change the project trajectory or the description of new experiments as a result of these explorations.  

> In the final project submission (E3), this section should list the **main** results obtained (not necessarily all), which best represent the fulfillment of the project objectives.  

> The discussion of results may be carried out in a separate section or integrated into the results section. This is a matter of style.  
> It is considered fundamental that the presentation of results should not serve as a treatise whose only purpose is to show that "a lot of work was done."  
> What is expected from this section is that it **presents and discusses** only the most **relevant results**, highlighting the **strengths and/or limitations** of the methodology, emphasizing aspects of **performance**, and containing content that can be classified as **organized, didactic, and reproducible sharing of knowledge relevant to the community**.  

## Conclusion
**Kris TODO*
> The Conclusion section should recover the main information already presented in the report and point to future work.  
> In the partial project submission (E2), it may contain information about which steps or how the project will be conducted until its completion.  
> In the final project submission (E3), the conclusion is expected to outline, among other aspects, possibilities for the project’s continuation.  

## Bibliographic References
1. D. A. Abuhani, I. Zualkernan, R. Aldamani and M. Alshafai, "Generative Artificial Intelligence for Hyperspectral Sensor Data: A Review," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 6422-6439, 2025, doi: https://doi.org/10.1109/JSTARS.2025.3538759.
2. Palsson, Burkni, Magnus O. Ulfarsson, and Johannes R. Sveinsson. 2023. "Synthesis of Synthetic Hyperspectral Images with Controllable Spectral Variability Using a Generative Adversarial Network" Remote Sensing 15, no. 16: 3919. https://doi.org/10.3390/rs15163919
3. Liu, J., Wu, Z., Xiao, L., & Wu, X. J. (2022). Model Inspired Autoencoder for Unsupervised Hyperspectral Image Super-Resolution. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-12. https://doi.org/10.1109/tgrs.2022.3143156
4. L. Zhu, Y. Chen, P. Ghamisi and J. A. Benediktsson, "Generative Adversarial Networks for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 9, pp. 5046-5063, Sept. 2018, doi: 10.1109/TGRS.2018.2805286.
5. J. Hauser, G. Shtendel, A. Zeligman, A. Averbuch and M. Nathan, "SHS-GAN: Synthetic Enhancement of a Natural Hyperspectral Database," in IEEE Transactions on Computational Imaging, vol. 7, pp. 505-517, 2021, doi: 10.1109/TCI.2021.3079818.
6. B. Liu, Y. Zhu, K. Song, and A. Elgammal, "Towards Faster and Stabilized GAN Training for High-Fidelity Few-Shot Image Synthesis," arXiv preprint arXiv:2101.04775, 2021. Available: https://arxiv.org/abs/2101.04775.
