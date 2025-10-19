# `<Project Title in Portuguese>`
# `<Image Super-Resolution with Generative Models>`

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

### Deliverables
The presentation for the E2 delivery can be found here * ADD LINK

> Include name, RA, and specialization focus of each group member.  
> |Name  | RA | Specialization|
> |--|--|--|
> | Brendon Erick Euzébio Rus Peres  | 256130  | Computer Engineering|
> | Miguel Angelo Romanichen Suchodolak | 178808 | Computer Engineering|
> | Núbia Sidene Almeida das Virgens  | 299001  | Computer Engineering|

## Abstract

The project addresses the problem of **image super-resolution**, a fundamental task in computer vision that aims to reconstruct high-quality images from their low-resolution counterparts. This problem has strong practical relevance in areas such as:
- **Low-budget devices**: Get one poor picture and enhance it
- **Medical Imaging**: Enhancing diagnostic image quality
- **Satellite Analysis**: Improving remote sensing data
- **Image Restoration**: Recovering historical or degraded photographs
- **Personal Use**: Upscaling cherished memories and favorite images 
  
## Objective

The main objective of the project is to design and evaluate a **generative model** capable of improving the resolution and perceptual quality of input images. The model will receive as input a low-resolution image and will output a high-resolution reconstruction with enhanced details and sharper definition, in general its goals can be set by:

- Improving resolution and perceptual quality of input images
- Preserving fine details and texture information
- Generating visually convincing high-resolution reconstructions
- Outperforming traditional interpolation methods, like bicubic

As inspiration, this project will draw on the methodology and architecture proposed in the repository [InvSR: Invertible Super-Resolution](https://github.com/zsyOAOA/InvSR).  

## Methodology

> Clearly and objectively describe, citing references, the methodology proposed to achieve the project objectives.  
> Describe datasets used.  
> Cite reference algorithms.  
> Justify the reasons for the chosen methods.  
> Point out relevant tools.  
> Describe the evaluation methodology (how will it be assessed whether the objectives were met or not?).  

### Datasets and Evolution
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

## Schedule  

Delivery | Week | Activity |  
|--------|------|----------|  
E1       | 1–2  | Literature review, dataset preparation, and baseline setup (bicubic, ESRGAN). |  
E2       | 3–4  | Initial implementation of the InvSR model. |  
E2       | 5–6  | Model training and hyperparameter tuning. |  
E2       | 7    | Intermediate evaluation and analysis of quantitative/qualitative results. |  
E3       | 8    | Refinements (integration of additional techniques, e.g., perceptual loss). |  
E3       | 9    | Final experiments, result consolidation, and comparison with benchmarks. |  
E3       | 10   | Report writing and final presentation preparation. |  


## Bibliographic References  
- **Bjorn, M., et al.** - *"A Lightweight Image Super-Resolution Transformer Trained on Low-Resolution Images Only"* ([arXiv 2025](https://arxiv.org/))
- **Miao, Y., et al.** - *"A general survey on medical image super-resolution via deep learning"* ([ScienceDirect 2025](https://www.sciencedirect.com/))
- **Chen, Z., et al.** - *"NTIRE2025 Challenge on Image Super-Resolution (×4): Methods and Results"* ([arXiv 2025](https://arxiv.org/))
- **Wang, W., et al.** - *"A lightweight large receptive field network LrfSR for image super resolution"* ([Nature 2025](https://www.nature.com/))
- **Guo, Z., et al.** - *"Invertible Image Rescaling"* ([NeurIPS 2022](https://proceedings.neurips.cc/))
- **Wang, X., et al.** - *"ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"* ([ECCV 2018](https://arxiv.org/))
- **Saharia, C., et al.** - *"Image Super-Resolution via Iterative Refinement"* ([IEEE TPAMI 2022](https://ieeexplore.ieee.org/))
