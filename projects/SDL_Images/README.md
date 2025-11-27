# Exploração Guiada do _Manifold_ de Expressões Faciais Emocionais
# Guided Exploration of Emotional Facial Expressions Manifold 

## Presentation 
link: https://www.canva.com/design/DAG5yiYbwfw/iEA-CWcEBDfvbX2VxDYApw/edit

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

|Name  | RA | Specialization|
|--|--|--|
| Alan Gonçalves            | 122507 | System Analysis|
| João Pedro Meira Gonçalo  | 218767 | Electrical Engineering |

## Abstract

This project explores the generation and control of facial expressions using Generative Models. Using the AffectNet dataset, we trained a Variational Autoencoder (VAE) to learn a latent representation of human faces. We mapped this latent space using the Russell's Circumplex Model of Affect semantics (Valence and Arousal). The methodology involved finding orthogonal direction vectors in the latent space that correspond to changes in emotion intensity. The results demonstrate the ability to traverse the emotional manifold, generating faces that smoothly transition between states such as "sad" (low arousal, negative valence) and "happy" (high arousal, positive valence), validated through disentanglement, alignment and linearity metrics (MIG, cosine similarity, $R^2$).

## Problem Description / Motivation

Dimensional Emotion Models in psychology look into mapping different emotions felt through a continuous space. One of the most popular approaches are based on the Russell Circumplex Model of Affect, mapping each emotion as a pair of a metric describing how pleasant an emotion (valence) and how intense it is (arousal). However, continuous visualization of those diagramms is still lacking, especially regarding facial expressions aligned with the circumplex modeling.

This way, we look to decouple facial identity from emotional expression, allowing for the precise exploration of emotions (e.g., making a face look "happier" or "calmer") without altering the person's identity, using the continuous dimensions of valence and Arousal rather than discrete categorical labels (happy, sad, etc.).

The motivation for this project is to bridge the gap between Generative Deep Learning and Psychological Affect Theory. Instead of treating emotions as discrete buckets (Happy vs. Sad), we treat them as a continuous navigation problem. If we can map the latent space to Russell's Circumplex Model, we can explore the *transitions* and *nuances* of human expression that lie between the standard categories.

## Objective

The primary objective is to map the latent space of a generative model to the Russell Circumplex Model of Affect to enable structured exploration of the emotional manifold.

**Specific Objectives:**
1.  Train a VAE on the AffectNet dataset to learn good latent representation of human faces.
2.  Extract latent vectors ($z$) and regress them against continuous valence and arousal annotations.
3.  Compute and orthogonalize the direction vectors ($d_{val}$, $d_{aro}$) that best represent these emotional axes in the high-dimensional space, validating the geometric properties of the projection (linearity and orthogonality).
4.  Visualize the emotion manifold by generating a grid of faces sampled systematically along these axes.

## Methodology

We adopted a "Latent Space Mapping" approach: first learning the latent emotions (training VAE), then making the map (finding the axes), and finally scroll through it (generating the facial expressions with a high resolution GAN).

This way we could combine the high-fidelity generative capabilities of StyleGAN3 with the psychological framework of Russell’s Circumplex Model encoded through a latent space by the disentanglement capacity of the β-VAE architecture. The core proposal is to treat emotions not as discrete classes, but as continuous vectors. The emotions directions were learned at the VAE space latent space ($Z$ space), and then mapped to the GAN latent space ($W$ space).

**VAE / Disentangling model**
*  We use a convolutional VAE (an encoder/decoder backbone is defined in the script) to learn a compact latent representation Z of faces. The VAE latent is used to discover directions associated with valence and arousal. The VAE class and training/visualization helpers are implemented in the notebook/script.

**Latent-to-emotion regression**
*  After training the VAE, we extract latents Z for the test set and fit simple linear regressors (Ridge regression) to predict valence and arousal from Z. The normalized regression coefficients are stored as direction vectors (val_direction, aro_direction) and saved to disk for later use in traversal and GAN bridging. This provides a straightforward and interpretable mapping from latent coordinates → affective dimension.

**Metrics & disentanglement diagnostics**
*  **Informativeness (R²):** measure how well Z linearly predicts valence/arousal via Ridge; targets in our evaluation are R²_valence > 0.80 and R²_arousal > 0.60 (reported in code). 
*  MIG (Mutual Information Gap): implemented to quantify whether a single latent dimension captures a given factor (valence or arousal); the code computes a mean MIG and prints the top neuron indices for each factor. 
*  **Utility / linear separability:** create valence/arousal quadrants and train a logistic classifier on Z to estimate quadrant accuracy (proxy for meaningful clustering of affective states).

**StyleGAN bridging & fine-tuning**
*  To obtain high-fidelity images, we fine-tune a StyleGAN (StyleGAN3 by NVLabs in the script) on a curated subset of “extreme” emotional images and bridge the VAE and GAN latent spaces: generate samples from the GAN, encode or regress their latent representations into the VAE directions, and learn a mapping to apply valence/arousal edits in GAN w-space. The code defines an EmotionExplorer class that loads a pretrained StyleGAN generator and direction maps d_val / d_aro for per-layer editing. Fine-tuning and mapping steps are part of the pipeline so generation can be controlled along affective axes.

**Tools & environment**
*  Python, PyTorch (torch & torchvision), facenet-pytorch (MTCNN), OpenCV, Pillow, NumPy. A requirements.txt is generated and wheel download / install steps are included for reproducible Colab runs.
*  Sklearn (Ridge, LogisticRegression, scoring utilities), tqdm for progress bars, matplotlib for visualizations.
*  StyleGAN3 repo is cloned and prepared for fine-tuning; the project includes code to export the expressive subset and run GAN training scripts.

### Datasets and Evolution

We utilized a reduced version of the AffectNet Dataset (around 280k images), focusing specifically on its continuous dimensional annotations rather than just categorical labels. 

|Dataset | Web Address | Descriptive Summary|
|----- | ----- | -----|
|AffectNet | [Link](https://www.mohammadmahoor.com/pages/databases/affectnet/) | The largest database of facial expressions in the wild, containing ~420k images manually annotated with Valence and Arousal scores (float values from -1 to 1).|

**Dataset Analysis & Preprocessing:**
* **Format:** Images were processed as $128 \times 128$ tensors.
* **Dimensional Annotation:** Unlike many projects that use the 'Expression' column (Int), we utilized the 'Valence' and 'Arousal' columns (Float).
* **Preprocessing:**
    * **MTCNN:** Used for strict face detection to remove background noise.
      *  Faces are cropped and resized to 128×128 for the VAE pipeline.
      *  Images for which MTCNN returns None (no face or severe occlusion) are discarded during preprocessing.
    * **Histogram Equalization (CLAHE):** Applied to the luminance channel to reduce lighting variation while preserving chromaticity, trying to prevent the model from learning lighting conditions (e.g., "dark image = sad") instead of facial features.
    * **Filtering:** Images with low face detection confidence or extreme poses were discarded to ensure the manifold represents frontal facial geometry.

When analyzing the labels distribution through the samples, we realized a severe underbalancing problem, with some classes having more than 8 times more samples than others. So, two different were followed:
  * **Undersampling (balanced subset):** construct a balanced training set by sampling a fixed number of images per expression (the code uses samples_per_category = 3500 as default). Balanced images and annotations are copied into /content/dataset/undersampling/... for faster experiments and perfectly balanced class distributions. Use this when compute resources or training time are limited.
  * **Weighted-loss (full dataset):** keep the entire dataset (retain all images) and compute class weights weight = total_samples / (n_classes * count) to apply in the loss function during training. The code saves the weights to class_weights.npy for later loading in training. This can preserve variability and is recommended if compute permits.


### Workflow

1.  Preprocessing and preparing the samples (As described above)
2.  Train and analyzing a disentangling β-VAE to obtain emotion directions in latent space
3.  Fine-tune StyleGAN3 expressions layers with AffectNet images to get more visible expressions
4.  Bridge between β-VAE emotional latent code and StyleGAN3 image generation latent code
5.  Explore emotions synthesis with a ortogonal facial expressions grid

## Experiments, Results, and Discussion of Results

### 1. Manifold Linearity Analysis
To verify if the latent space actually encodes emotions linearly as posited by the circumplex model, we measured the $R^2$ score of our linear probes.
* **Result:** We observed significant positive correlation ($R^2 > 0.6$ typically for VAEs on aligned faces) for both axes. This confirms that "Emotion" is not a localized cluster but a directional vector in the VAE's latent space.

### 2. Orthogonality and Disentanglement
A critical requirement for the Circumplex model is that Valence (Positivity) and Arousal (Intensity) are independent.
* **Experiment:** We calculated the cosine similarity between our computed Valence vector and Arousal vector.
* **Result:** After applying Gram-Schmidt orthogonalization, the cosine similarity was reduced to near zero ($\approx 0.0$), ensuring that increasing "Excitement" does not inadvertently make the face look "Happier" or "Sadder". The axes are geometrically perpendicular in the latent space.

### 3. The "Emotion Grid" Visualization
The most significant result is the **Learned Manifold Grid**, a $7 \times 7$ visualization where:
* **X-Axis (Valence):** Traverses from negative (left) to positive (right).
* **Y-Axis (Arousal):** Traverses from low (bottom) to high (top).

The center of the grid represents the "Neutral" face. As we move to the top-left (High Arousal, Low Valence), the face naturally morphs into a distressed/fearful expression. As we move to the bottom-right (Low Arousal, High Valence), the face relaxes into a calm satisfaction. This visual proof confirms the model learned the topology of human affect without explicit rule-based programming.

## Conclusion

This project successfully demonstrated that emotion dimensional models from psychology are not just a psychological construct but a realizable geometric structure within the latent space of deep generative models. By aligning the mathematical axes of a VAE with the psychological axes of valence and arousal, we created a tool for emotion manifold exploration. The resulting system allows for the synthesis of nuanced emotional states that lie between standard categories, offering a more granular approach to facial expression generation. Future work could investigate, for instance, 3D mappings to capture the details of the emotional manifold more accurately.

## Bibliographic References
1.  **Russell, J. A. (1980).** "A circumplex model of affect". *Journal of Personality and Social Psychology*.
2.  **Mollahosseini, A., et al. (2017).** "AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild".
3.  **Kingma, D. P., & Welling, M. (2013).** "Auto-Encoding Variational Bayes".
4.  **Radford, A., et al. (2015).** "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks".
5.  **Higgins I., et al. (2017).** "Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework."
6.  **Karras T., et al. (2019).** "A Style-Based Generator Architecture for Generative Adversarial Networks."
