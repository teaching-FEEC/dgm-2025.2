# Modelagem da Manipulação de Objetos Lineares Deformáveis
# Modeling the Manipulation of Deformable Linear Objects

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

> | Name | RA | Specialization |
> |--|--|--|
> | Tim Missal | 298836 | Computer Science |
> | Lucas Vinícius Domingues | 291414 | Computer Science |
> | Natália da Silva Guimarães | 298997 | Engineering |

## Abstract

This project aims to develop a **probabilistic dynamics model** for **Deformable Linear Objects (DLOs)**, which is fundamental for autonomous manipulation with risk awareness. We generated a synthetic dataset of transitions (**state, action, next state**) using the **MuJoCo** simulator. Architectures such as **BiLSTM**, **Transformer + BiLSTM**, and **VAE** were evaluated to predict the rope's future configuration. Initial results demonstrate the feasibility of learning, with **BiLSTM showing the lowest error**. The next phase will involve massive dataset expansion and the exploration of **Latent World Models**[1] to better quantify the uncertainty of action outcomes.

---

## 1.Problem Description / Motivation

The autonomous manipulation of Deformable Linear Objects (**DLOs**), such as cables and ropes, is a significant challenge due to ambiguity in perception (like overlaps) and **unpredictable action outcomes** caused by high friction in entangled states.

Classical methods fail to **quantify this uncertainty**, opting for actions that maximize immediate gain but may lead to high-risk states in the long term. The central motivation of this project is to develop a **model** that predicts a **probability distribution** over future states, allowing the agent to make decisions that **minimize risk** over an extended time horizon, rather than simply pursuing instantaneous reward.

## 2.Objectives

Develop a **dynamics model** capable of predicting the future state and **quantifying the uncertainty** of a Deformable Linear Object (**DLO**), given its current configuration and an applied robotic action.

## 3. Methodology

### 3.1 Overview

The project aims to develop a model capable of predicting the future state of a **Deformable Linear Object (DLO)** given its current configuration and an applied action. This predictive model serves as the foundation for future development of **risk-aware agents** for autonomous DLO manipulation.

### 3.2 Dataset Generation

To train and evaluate the proposed models, a synthetic dataset of **(state, action, next_state)** tuples was generated using the **MuJoCo physics engine** (Multi-Joint dynamics with Contact). MuJoCo is a widely used simulation framework for robotics and deformable object modeling due to its accurate contact and joint dynamics.

#### Simulation Setup

- The rope was modeled as a chain of **70 connected cylinders** linked through ball joints, simulating a flexible, deformable body.  
- **State representation ($\mathbf{s}_{t}$):** The 3D coordinates $(\mathbf{x}, \mathbf{y}, \mathbf{z})$ of the rope’s 70 cylinders.  
- **Action representation ($\mathbf{a}_{t}$):** A 3D force vector applied to a single cylinder at each step.  
- **Next state ($\mathbf{s}_{t+1}$):** Rope configuration after the force is applied.  
- **Data rate:** The current implementation generates approximately **7 datapoints/second** (~600,000 per day).

### 3.3 Modeling Approaches

Three model architectures were evaluated to learn the rope dynamics:

#### (a) BiLSTM

- Based on the model proposed by Yan et al. (2019) [3], which demonstrated strong results for state estimation in DLO manipulation.  
- The **Bidirectional LSTM** captures both forward and backward dependencies between rope links, essential since each link’s motion depends on its neighbors on both sides.  
- **Residual connections** were added to allow the model to predict position deltas, improving stability and convergence.

#### (b) Transformer + BiLSTM Hybrid

- Combines a **Transformer encoder-decoder** (Vaswani et al., 2017) [3] to model global dependencies between rope segments with a **BiLSTM** to capture local directional dynamics.  
- This hybrid model draws inspiration from Viswanath et al. (2023) [4], which used learned representations of 1D objects for manipulation and inspection tasks.

#### (c) Variational Autoencoder (VAE)

- **VAEs** were explored as a foundation for probabilistic world modeling.  
- The encoder maps state-action pairs to a latent distribution, allowing the decoder to predict a **probabilistic next state**, aligning with the goal of capturing uncertainty in rope dynamics.

### 3.4 Implementation and Tools

| Component | Tool/Language | Version/Details |
| :--- | :--- | :--- |
| **Simulation** | MuJoCo | v3.1 |
| **Programming Language** | Python | 3.10 |
| **Deep Learning Framework** | PyTorch, Transformers | - |
| **Data Processing** | NumPy, Pandas | - |
| **Visualization** | Matplotlib | - |

#### Training Setup

- **3,000 samples** used in current phase  
- Models trained for **100 epochs**  
- Checkpointing based on **validation loss (MSE)**  
- Data **normalized** per coordinate (mean and std)

## 4. Evaluation Methodology

### 4.1 Metrics

Model performance was evaluated using:

- **Mean Squared Error (MSE):** Quantifies the average Euclidean distance between predicted and ground-truth rope coordinates.

  $\text{MSE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1}(y_i - \hat{y}_i)^2$

  where *N* is the number of rope links, $Y_i$ is the actual position of the rope link, and $\hat{Y}_i$ is the predicted position. A lower MSE indicates a better fit of the model to the data.

- **Qualitative Visualization:** Predicted and true rope shapes were visualized to assess spatial similarity and clustering tendencies.

**Planned Extensions:** Future work will incorporate **Average Displacement Error (ADE)** and **Dice Coefficient** to better measure deformation overlap.

### 4.2 Assessment Criteria

The objectives are considered met when:

1. The model accurately predicts next-state configurations (**low MSE/ADE**).  
2. The model generalizes to unseen actions and configurations.  
3. Probabilistic models (e.g., VAE) demonstrate the ability to represent **uncertainty** in outcomes.

## 5. Experiments, Results, and Discussion

### 5.1 Experimental Setup

A dataset of approximately 3,000 samples was generated in **MuJoCo**, simulating a rope composed of 70 interconnected cylindrical segments with ball joints. Each sample consisted of a state $s_{t}$ (the 3D positions of all rope segments), an action $a_{t}$ (a force vector applied to a specific segment), and the resulting next state $s_{t+1}$.  
The dataset was normalized and divided into training, validation, and test sets. All models were trained using the **Mean Squared Error (MSE)** loss.

Three model families were tested:

- **BiLSTM** — captures bidirectional dependencies along the rope.  
- **Transformer + BiLSTM** — combines global attention and local sequential modeling.  
- **VAE** — tests a generative probabilistic baseline.

### 5.2 Results

| Model | MSE |
|--------|-----|
| BiLSTM (2 layers) | **1.04** |
| Transformer + BiLSTM (1 encoder/decoder + 1 BiLSTM) | 1.06 |
| VAE | 1.06 |

Visual inspection of predicted centerlines revealed that all models could reproduce the general rope configuration but tended to produce slightly **clustered** predictions concentrated near the original region.  
The **BiLSTM** achieved marginally lower error and visually smoother deformations, suggesting that local sequential dependencies dominate rope dynamics in this limited dataset.

### 5.3 Discussion of Results

Although the differences between models are small, several tendencies emerged:

- **Dataset size is the primary limitation.** With only 3,000 samples, models quickly overfit, and larger, more diverse data are needed.  
- **Clustering and low regional diversity.** All models produced predictions biased toward more clustered rope configurations.  
- **BiLSTM achieved best results.** For small datasets, recurrent architectures outperform more modern alternatives.  
- **Transformer and BiLSTM hybrids** capture deformations better but are prone to concentrated predictions.  
- **VAE predictions** show higher variability, representing uncertainty but at reduced positional accuracy.

The experiments highlight that **performance is currently constrained by data diversity**, not architectural sophistication. Future iterations will include over one million samples with varied physical parameters (friction, stiffness, action complexity, and bimanual manipulation).  
Additionally, **latent world models (e.g., Dreamer [3])** will be explored to capture multimodal dynamics and uncertainty.

## 6. Conclusion

This partial submission presented the initial progress toward building a **probabilistic model for the manipulation of Deformable Linear Objects (DLOs)**. The project addressed the challenges of representing and predicting rope dynamics under applied actions using data generated in **MuJoCo** simulations. A dataset of approximately 3,000 samples was created, encoding rope states as sequences of 3D coordinates and actions as localized forces.

Three neural architectures — **BiLSTM**, **Transformer + BiLSTM**, and **VAE** — were tested to predict the next rope configuration. The **BiLSTM** achieved slightly better results, indicating that local sequential dependencies dominate at this stage. The small dataset size limited generalization capacity, reinforcing the need for larger and more varied data.

The experiments demonstrated the feasibility of learning rope dynamics from simulation but also exposed the limitations of simple deterministic models in capturing uncertainty. This motivates the next phase, which will focus on:

- **Dataset Expansion:** generate over one million samples with varied physical parameters.  
- **Action Complexity:** include multi-step and bimanual manipulations.  
- **Model Enhancement:** explore **latent world models (e.g., Dreamer[3])** and **spatial transformers**.  
- **Evaluation and Planning:** introduce probabilistic metrics such as **Average Displacement Error (ADE)** and **Dice Coefficient**.

Ultimately, the goal is to obtain a dynamics model that not only predicts the next state of the rope accurately but also **quantifies uncertainty**, enabling **risk-aware planning** for autonomous DLO manipulation tasks.

## 7. Bibliographic References

[1] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to Control: Learning Behaviors by Latent Imagination,” *Proc. ICLR 2020*, 2020. [Online]. Available: https://arxiv.org/abs/1912.01603

[2] Yan, Mengyuan, et al. “Self-Supervised Learning of State Estimation for Manipulating Deformable Linear Objects.” ArXiv.org, 2019. [Online]  Available: arxiv.org/abs/1911.06283.

[3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. Advances in neural information processing systems, 30. NIPS papers. You can also cite it via its arXiv number: Vaswani, A., et al. (2017). Attention Is All You Need. arXiv preprint arXiv:1706.03762.  [Online] Available https://arxiv.org/abs/1706.03762. 

[4] D. Ha and J. Schmidhuber, “World Models,” *CoRR*, vol. abs/1803.10122, 2018. [Online]. Available: https://arxiv.org/abs/1803.10122  

# Presentation Link

[Google Slides Presentation](https://docs.google.com/presentation/d/1fw3_m6minAr5l9Ks6CPWKIoQIB-UsBzjtunPPyIErtY/edit?usp=sharing)
