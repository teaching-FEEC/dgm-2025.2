# `<Abordagem Generativa para o Conjunto de Dados First-Impressions via Modelo OCEAN>`
# `<Generative Approach for First-Impressions Dataset by OCEAN Model>`

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

> |Name  | RA | Specialization|
> |--|--|--|
> | Alan Gonçalves            | 122507 | System Analysis|
> | João Pedro Meira Gonçalo  | 218767 | Electrical Engineering |

## Project Summary Description
> **Description of the project theme, including generating context and motivation:**
>   * This project addresses the challenge of personality trait classification based on the OCEAN (Big Five) model, using the First Impressions dataset from ChaLearn. The OCEAN model is a theoretical framework widely adopted in psychology to describe personality across five major dimensions: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. The dataset provides short video clips annotated with OCEAN-based personality traits, enabling computational models to infer psychological characteristics from visual and behavioral cues.
>   * A key motivation of this project lies in the class imbalance present in the dataset, which impacts the performance and generalization of machine learning models. To mitigate this issue, we propose the use of generative models to create synthetic images, aiming to balance the distribution of classes and improve the robustness of predictive models.
> 
> **Description of the main goal of the project:**
>   * The main goal of the project is to develop and evaluate a generative model capable of producing synthetic face images aligned with specific OCEAN traits. By generating additional data for underrepresented categories, the project seeks to reduce dataset imbalance and enhance the accuracy and fairness of personality trait classification tasks.
>  
> **Clarify what the output of the generative model will be:**
>   * The output of the generative model will be a set of synthetic facial images conditioned on OCEAN trait labels, which will complement the original First Impressions dataset. These synthetic samples will be used to balance the dataset, supporting downstream tasks of training and validating classification models for personality prediction.
>   
> **Include in this section a link to the presentation video of the project proposal (maximum 5 minutes).**

## Proposed Methodology
> For the first submission, the proposed methodology must clarify:  
> * **Which dataset(s) the project intends to use, justifying the choice(s)**
>   * The project will use the ChaLearn First Impressions V2 dataset, which contains ten thousand of short video clips annotated with the OCEAN (Big Five) personality traits. This dataset was chosen because it is one of the most widely recognized and publicly available benchmarks for automatic personality recognition. Its annotations are directly aligned with the OCEAN model, ensuring consistency with the theoretical framework adopted in this project.
> * **Which generative modeling approaches the group already sees as interesting to be studied**
>   * The group considers Generative Adversarial Networks (GANs) in large-scale implementations as an interesting approach to be studied. GANs have demonstrated state-of-the-art performance in producing high-quality and realistic facial images, which aligns with the project’s goal of generating synthetic samples for balancing the dataset. Large-scale GAN architectures, such as StyleGAN or BigGAN, offer improved stability during training and enhanced capacity to model fine-grained details, making them suitable for the complexity of facial data. Additionally, conditional versions of GANs (cGANs or AC-GANs) can be explored to generate synthetic images aligned with specific OCEAN personality traits, ensuring the relevance of the generated data for the downstream classification task.
> * **Reference articles already identified and that will be studied or used as part of the project planning**
>   * Brock, A., Donahue, J., & Simonyan, K. (2019). Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096.
>   * Ziegler, M., Horstmann, K. T., & Ziegler, J. (2019). Personality in situations: Going beyond the OCEAN and introducing the Situation Five. Psychological Assessment, 31(4), 567. American Psychological Association.
>   * Ilmini, W. M. K. S., & Fernando, T. G. I. (2017). Computational personality traits assessment: A review. In 2017 IEEE International Conference on Industrial and Information Systems (ICIIS) (pp. 1–6). IEEE.
>   * Yesu, K., Shandilya, S., Rekharaj, N., Kumar, A., & Sairam, P. S. (2021). Big Five Personality Traits Inference from Five Facial Shapes Using CNN. In 2021 IEEE 4th International Conference on Computing, Power and Communication Technologies (GUCON) (pp. 1–6). IEEE.
>   * Helm, D., & Kampel, M. (2020). Single-modal video analysis of personality traits using low-level visual features. In 2020 Tenth International Conference on Image Processing Theory, Tools and Applications (IPTA) (pp. 1–6). IEEE.
>   * Figueira, A., & Vaz, B. (2022). Survey on synthetic data generation, evaluation methods and GANs. Mathematics, 10(15), 2733. MDPI.
> * **Tools to be used (based on the group’s current vision of the project)**
>   * Compute & Environment: Google Collaboratory with GPU support (CUDA/cuDNN), GitHub for code versioning, Deep Learning Frameworks, PyTorch (principal), TensorFlow/Keras.
>   * Data & Preprocessing: OpenCV and FFmpeg for frames extraction, NumPy and Pandas for data handling.
>   * Evaluation: Scikit-learn for discrimination metrics, FID/KID for synthetic image quality evaluation.
> * **Expected results**
>   * Data Generation: Creation of synthetic facial images conditioned on OCEAN traits using GAN-based models. Balanced extension of the First Impressions dataset.
>   * Model Performance: Improved classification accuracy and fairness across OCEAN traits. More robust models due to reduced data imbalance.
>   * Evaluation: Quantitative improvements measured by FID/KID for image quality. Higher accuracy, F1-score, and ROC-AUC in downstream classification tasks.
> * **Proposal for evaluating the synthesis results**
>   * To evaluate the synthesis results, the project will combine quantitative, qualitative, and downstream-task-based approaches. Quantitative evaluation will rely on established metrics such as FID, KID, and LPIPS, which measure the quality, diversity, and perceptual similarity of the generated images. Qualitative evaluation will include visual inspection to verify coherence between synthetic samples and their intended OCEAN traits.
>   * Finally, the most important evaluation will be the impact of synthetic data on the downstream classification task: models trained with the balanced dataset (real + synthetic) will be compared to baseline models, using accuracy, F1-score, and ROC-AUC as key performance indicators. This multi-faceted evaluation ensures both the realism of the generated data and its practical utility for personality trait prediction.

## Schedule
> **Proposed schedule. Try to estimate how many weeks will be spent on each stage of the project**
> * Weeks 1–2 — Dataset Preparation
>   * Familiarization with the First Impressions dataset.
>   * Data cleaning, preprocessing, and frame extraction.
>   * Exploratory analysis to confirm class imbalance.
> * Weeks 3–5 — Model Design & Implementation
>   * Selection of generative approach (Large-Scale GAN, cGAN).
>   * Initial prototyping in PyTorch/TensorFlow.
>   * Setting up training environment on Google Colab with GPUs.
> * Weeks 6–7 — Synthetic Data Generation
>   * Train GAN models to generate facial images conditioned on OCEAN traits.
>   * Generate first batch of synthetic samples.
>   * Qualitative inspection of image realism and trait alignment.
> * Weeks 8–9 — Evaluation of Synthetic Data
>   * Compute quantitative metrics (FID, KID, LPIPS).
>   * Conduct visual analysis of generated samples.
>   * Select best-performing generative model.
> * Week 10 — Final Report & Presentation
>   * Retrain classification models with balanced dataset (real + synthetic).
>   * Measure improvements in accuracy, F1-score, ROC-AUC.
>   * Consolidate results, discussions, and insights.
>   * Prepare final paper and presentation video.

## Bibliographic References
> **Point out in this section the bibliographic references adopted in the project.**
>   * Brock, A., Donahue, J., & Simonyan, K. (2019). Large scale GAN training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096.
>   * Ziegler, M., Horstmann, K. T., & Ziegler, J. (2019). Personality in situations: Going beyond the OCEAN and introducing the Situation Five. Psychological Assessment, 31(4), 567. American Psychological Association.
>   * Ilmini, W. M. K. S., & Fernando, T. G. I. (2017). Computational personality traits assessment: A review. In 2017 IEEE International Conference on Industrial and Information Systems (ICIIS) (pp. 1–6). IEEE.
>   * Yesu, K., Shandilya, S., Rekharaj, N., Kumar, A., & Sairam, P. S. (2021). Big Five Personality Traits Inference from Five Facial Shapes Using CNN. In 2021 IEEE 4th International Conference on Computing, Power and Communication Technologies (GUCON) (pp. 1–6). IEEE.
>   * Helm, D., & Kampel, M. (2020). Single-modal video analysis of personality traits using low-level visual features. In 2020 Tenth International Conference on Image Processing Theory, Tools and Applications (IPTA) (pp. 1–6). IEEE.
>   * Figueira, A., & Vaz, B. (2022). Survey on synthetic data generation, evaluation methods and GANs. Mathematics, 10(15), 2733. MDPI.

