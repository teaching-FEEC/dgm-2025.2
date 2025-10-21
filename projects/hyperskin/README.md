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

## Abstract

> Summary of the objective, methodology **and results** obtained (in submission E2 it is possible to report partial results). Suggested maximum of 100 words. 

## Project Summary Description
Hyperspectral Imaging (HSI) combines imaging and spectroscopy, giving each pixel a continuous spectrum across wavelengths. HSI captures how light interacts with molecules, as their composition, vibrations, and structure affect photon behavior. These light–matter interactions create distinct spectral patterns that act like unique “fingerprints” for each material. Thanks to its ability to distinguish different materials, tissues, and substances, Hyperspectral Imaging (HSI) has become a valuable tool in Remote Sensing, Agriculture, and Medicine. In medicine, its capacity to go beyond standard RGB imaging is mainly used to detect tumors. However, publicly available hyperspectral tumor datasets are scarce, which often leads melanoma classification models to overfit or perform poorly in subsampled classes. Therefore, the main goal of this project is to construct a generative ai model that creates a synthetic hyperspectral dermoscopy dataset. More specifically, we hope that a classifier trained with both synthetic and real hyperspectral images, outperform a classifier trained with only real images. 

### Main goal
Therefore, the main goal of this project is to construct a generative ai model that learns the distribution of real hyperspectral images and through them is able to create a synthetic hyperspectral melanoma dataset.
Desired output: a synthetic hyperspectral dataset of skin lesions and melanoma. 

#### Main Hypothesis 
A classifier trained with synthetic AND real data will have better results than if only trained in real data. Specially looking at subsampled classes 

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

#### SHS GAN [6]

- The model receives as input a standard RGB image and its task is to generate a synthetic hyperspectral cube. The objective of the Generator is to learn a mapping from the RGB domain to the HS domain, so that the distribution of the synthetic HS cubes becomes similar to the distribution of real HS cubes.

- the RGB image is used as input to the Generator so that the synthetic HS cube preserves the spatial details and textures of the input image and also keeps the color properties coherent with what appears in the RGB.

- The Critic is trained to evaluate whether the generated HS cubes are realistic. It does so by analyzing spatial patterns and also the smoothness and shape of spectral curves, which are emphasized by looking at the data in both the image and Fourier-transformed spectral domains.

- In addition, the synthetic HS cube can be converted back into RGB using a deterministic transformation. This reconstructed RGB image is compared to the original input RGB, and differences are penalized during training. This step enforces consistency between the generated HS cube and the original RGB image.

- It is used a WGAN training pipeline

![SHS Architecture Diagram](images/SHS-GAN.png)

#### FastGAN
*Kris TODO*

#### Autoencoder
- The autoencoder is composed by an encoder and a decoder. The encoder compresses the input HSI image into a lower-dimensional latent representation, while the decoder reconstructs the original image from this representation.
- A variational autoencoder (VAE) is a type of autoencoder that learns a probabilistic mapping from the input data to a latent space, allowing for the generation of new samples by sampling from this latent space.
- VAEs are especially adept at modeling complex, high-dimensional data and continuous latent spaces, making them extremely useful for tasks like generating diverse sets of similar images.
- Palsson et al. [2] used a VAE paired with a GAN framework to generate high-resolution synthetic hyperspectral images.
- Liu et al. [3] proposed a model inspired autoencoder (MIAE) to fuse low-resolution HSI with high-resolution RGB images to generate high-resolution HSI.
- We used an AutoEncoder with a Convolutional Encoder and a UNet decoder


### Evaluating synthesis results
We would like for the generated images to be: clear, realistic and useful. 
- Image Clarity/Quality : Variance, Spatial and Spectral Entropy, SNR
- Image realism : Spectral Angle Mapper for average melanoma spectral signature , SSIM with real images
- Usability: Given a baseline classifier that classifies images into melanoma and not melanoma, first train the classifier with only real data then with real + synthetic data and see if F1 score improves. Then, train only on synthetic data and test on real data to see if classifier performs similarly 

### Datasets and Evolution
*Kris TODO*
> List the datasets used in the project.  
> For each dataset, include a mini-table in the model below and then provide details on how it was analyzed/used, as in the example below.  

|Dataset | Web Address | Descriptive Summary|
|----- | ----- | -----|
|Dataset Title | http://base1.org/ | Brief summary (two or three lines) about the dataset.|

> Provide a description of what you concluded about this dataset. Suggested guiding questions or information to include:  
> * What is the dataset format, size, type of annotation?  
> * What transformations and preprocessing were done? Cleaning, re-annotation, etc.  
> * Include a summary with descriptive statistics of the dataset(s).  
> * Use tables and/or charts to describe the main aspects of the dataset that are relevant to the project.  

### Workflow


Our methodology was designed to test whether the inclusion of synthetic hyperspectral images can improve tumor classification compared to training with only real data. As shown in the workflow, the process begins with the preprocessing of the hyperspectral dataset, where images are segmented and cropped in the region containing the lesion. The preprocessed data are then used in a generation stage, where four generative models—SHSGAN, DCGAN, FastGAN, and VAE—are trained to produce synthetic hyperspectral tumor images. These models learn the complex spectral and spatial characteristics of malignant tumors, generating synthetic melanoma samples as close to real as possible. The quality of these synthetic images is evaluated using standard generation metrics, including the Spectral Angle Mapper (SAM), Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Fréchet Inception Distance (FID), which assess both spectral and perceptual similarity to the real data.

Following image generation, two classification models are trained to distinguish malignant from benign tumors. The first classifier is trained exclusively with real hyperspectral images, while the second combines real and synthetic images in its training set. For both cases, two deep convolutional architectures, DenseNet and ResNet, are employed, each trained under two conditions: using pre-trained RGB weights or from scratch directly on hyperspectral data. The performance of each classifier is assessed using classification metrics such as F1-score, Accuracy, and SpecAtSens (Specificity at Sensitivity).

## Schedule
![Project Schedule](images/schedule.png)


## Experiments, Results, and Discussion of Results
> In each topic describe the experiments carried out, the results obtained, and a brief discussion of the results.
### Data Preprocessing
**Kris TODO*

### Classifier Training

### Generative Model Training

#### FastGAN
**Kris TODO*

### Classifier Training with Synthetic Data
**Kris TODO*

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
