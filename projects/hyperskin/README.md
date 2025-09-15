# `<Project Title in Portuguese>`
# `<Project Title in English>`

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

> Include name, RA, and specialization focus of each group member. Groups must have at most three members.
|Name  | RA | Specialization|
|--|--|--|
| Kristhian Aguilar  | 298976  | Computer Engineering|
| Ana Clara Caznok Silveira  | 231745  | Computer Engineering|
| Name3  | 123456  | XXX|

## Project Summary Description
Hyperspectral Imaging (HSI) combines imaging and spectroscopy, giving each pixel a continuous spectrum across wavelengths. HSI captures how light interacts with molecules, as their composition, vibrations, and structure affect photon behavior. These light–matter interactions create distinct spectral patterns that act like unique “fingerprints” for each material. Thanks to its ability to distinguish different materials, tissues, and substances, Hyperspectral Imaging (HSI) has become a valuable tool in Remote Sensing, Agriculture, and Medicine. In medicine, its capacity to go beyond standard RGB imaging is mainly used to detect tumors. However, publicly available hyperspectral tumor datasets are scarce, which often leads classification models to overfit or perform poorly.

### Main goal
Therefore, the main goal of this project is to construct a generative ai model that learns the distribution of real hyperspectral images and through them is able to create a synthetic HyperTumor dataset. 
Desired output: a synthetic hyperspectral dataset of skin lesions and melanoma. 

### Presentation
> Include in this section a link to the presentation video of the project proposal (maximum 5 minutes).

## Proposed Methodology
> For the first submission, the proposed methodology must clarify:  
> * Which dataset(s) the project intends to use, justifying the choice(s).
> * Which generative modeling approaches the group already sees as interesting to be studied.  
> * Reference articles already identified and that will be studied or used as part of the project planning.  
> * Tools to be used (based on the group’s current vision of the project).  
> * Expected results.  
> * Proposal for evaluating the synthesis results.

### Evaluating synthesis results
We would like for the generated images to be: clear, realistic and useful. 
(-) Image Clarity/Quality : Variance, Spatial and Spectral Entropy, PSNR and SSIM
(-) Image realism : (FID) Frechet Inception Distance adapted for HSI embeddings, Spectral Angle Mapper for average melanoma spectral signature 
(-) Usability: Given a baseline classifier that classifies images into melanoma and not melanoma, first train the classifier with only real data then with real + synthetic data and see if F1 score improves

## Schedule
> Proposed schedule. Try to estimate how many weeks will be spent on each stage of the project.  

## Bibliographic References
> Point out in this section the bibliographic references adopted in the project.
