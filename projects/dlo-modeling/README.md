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

## Project Summary Description

The autonomous manipulation of Deformable Linear Objects (DLOs) using robotic arms has proven to be a difficult task for mainly two reasons: the perceived state of the DLO might not reflect reality perfectly, as different textures or overlaps can lead to ambiguity; and actions performed on the DLO might lead to unexpected results due to high friction in entangled states. 

Classical methods used in some of the main applications of the manipulation of DLOs, such as Knot Tying and Knot Untangling, fail to capture the uncertainty in performed actions and instead calculate the best possible action to, e.g., simplify a DLO state analytically. Since this doesn't account for the uncertainty that can result from the reasons mentioned above, the best action might lead to a state that is much worse than before. In other words, the best action might be a potentially risky one. 

Instead of choosing actions with the highest immediate reward, we believe an agent should pursue actions which minimize the risk over the long term. To do this, a Dynamics Model is needed that, given a perceived state and an action, can predict a probability distribution over next states. Using this (world) model, an agent could be developed which can make better decisions, as famously shown in the Dreamer Papers [1, 2].

To summarize, we want to create a model with the following characteristics:

- **Input:**
  - **State ($s_t$):** DLO state given as a rope centerline. We decide not to work on images of DLOs such as ropes and cables, since the perception of a DLO centerline out of an image is a different topic that is studied in separate channels. Instead, we assume to have a model that generates the state information for us.
  - **Action ($a_t$):** An action performed on some segment of the DLO. For example, an action might be: pull some segment in the middle of the rope in direction xyz with force x N.
- **Output ($s_{t+1}$):** The state of the DLO at the next timestep, after performing $a_t$ on $s_t$.

The goal of having such a model is to be able to quantify whether or not executing action $a$ in state $s$ is a good or bad decision by calculating the action's risk. Work that builds upon this model could then use it to train a high-level planner for tasks that involve the manipulation of DLOs.

# 1. Methodology

## 1.1 Overview

The project aims to develop a model capable of predicting the future state of a **Deformable Linear Object (DLO)** given its current configuration and an applied action. This predictive model serves as the foundation for future development of **risk-aware agents** for autonomous DLO manipulation.

## 1.2 Dataset Generation

To train and evaluate the proposed models, a synthetic dataset of **(state, action, next_state)** tuples was generated using the **MuJoCo physics engine** (Multi-Joint dynamics with Contact). MuJoCo is a widely used simulation framework for robotics and deformable object modeling due to its accurate contact and joint dynamics.

### Simulation Setup

- The rope was modeled as a chain of **70 connected cylinders** linked through ball joints, simulating a flexible, deformable body.  
- **State representation ($\mathbf{s}_{t}$):** The 3D coordinates $(\mathbf{x}, \mathbf{y}, \mathbf{z})$ of the rope’s 70 cylinders.  
- **Action representation ($\mathbf{a}_{t}$):** A 3D force vector applied to a single cylinder at each step.  
- **Next state ($\mathbf{s}_{t+1}$):** Rope configuration after the force is applied.  
- **Data rate:** The current implementation generates approximately **7 datapoints/second** (~600,000 per day).

## 1.3 Modeling Approaches

Three model architectures were evaluated to learn the rope dynamics:

### (a) BiLSTM

- Based on the model proposed by Yan et al. (2019) [3], which demonstrated strong results for state estimation in DLO manipulation.  
- The **Bidirectional LSTM** captures both forward and backward dependencies between rope links, essential since each link’s motion depends on its neighbors on both sides.  
- **Residual connections** were added to allow the model to predict position deltas, improving stability and convergence.

### (b) Transformer + BiLSTM Hybrid

- Combines a **Transformer encoder-decoder** (Vaswani et al., 2017) to model global dependencies between rope segments with a **BiLSTM** to capture local directional dynamics.  
- This hybrid model draws inspiration from Viswanath et al. (2023) [2], which used learned representations of 1D objects for manipulation and inspection tasks.

### (c) Variational Autoencoder (VAE)

- Inspired by Ha and Schmidhuber (2018) [3] and Hafner et al. (2019) [4], **VAEs** were explored as a foundation for probabilistic world modeling.  
- The encoder maps state-action pairs to a latent distribution, allowing the decoder to predict a **probabilistic next state**, aligning with the goal of capturing uncertainty in rope dynamics.

## 1.4 Implementation and Tools

| Component | Tool/Language | Version/Details |
| :--- | :--- | :--- |
| **Simulation** | MuJoCo | v3.1 |
| **Programming Language** | Python | 3.10 |
| **Deep Learning Framework** | PyTorch, Transformers | - |
| **Data Processing** | NumPy, Pandas | - |
| **Visualization** | Matplotlib | - |

### Training Setup

- **3,000 samples** used in current phase  
- Models trained for **100 epochs**  
- Checkpointing based on **validation loss (MSE)**  
- Data **normalized** per coordinate (mean and std)

# 2. Evaluation Methodology

## 2.1 Metrics

Model performance was evaluated using:

- **Mean Squared Error (MSE):** Quantifies the average Euclidean distance between predicted and ground-truth rope coordinates.

  $\text{MSE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1}(y_i - \hat{y}_i)^2$

  where *N* is the number of rope links, $Y_i$ is the actual position of the rope link, and $\hat{Y}_i$ is the predicted position. A lower MSE indicates a better fit of the model to the data.

- **Qualitative Visualization:** Predicted and true rope shapes were visualized to assess spatial similarity and clustering tendencies.

**Planned Extensions:** Future work will incorporate **Average Displacement Error (ADE)** and **Dice Coefficient** to better measure deformation overlap.

## 2.2 Assessment Criteria

The objectives are considered met when:

1. The model accurately predicts next-state configurations (**low MSE/ADE**).  
2. The model generalizes to unseen actions and configurations.  
3. Probabilistic models (e.g., VAE) demonstrate the ability to represent **uncertainty** in outcomes.

# 3. Experiments, Results, and Discussion

## 3.1 Experimental Setup

A dataset of approximately 3,000 samples was generated in **MuJoCo**, simulating a rope composed of 70 interconnected cylindrical segments with ball joints. Each sample consisted of a state $s_{t}$ (the 3D positions of all rope segments), an action $a_{t}$ (a force vector applied to a specific segment), and the resulting next state $s_{t+1}$.  
The dataset was normalized and divided into training, validation, and test sets. All models were trained using the **Mean Squared Error (MSE)** loss.

Three model families were tested:

- **BiLSTM** — captures bidirectional dependencies along the rope.  
- **Transformer + BiLSTM** — combines global attention and local sequential modeling.  
- **VAE** — tests a generative probabilistic baseline.

## 3.2 Results

| Model | MSE |
|--------|-----|
| BiLSTM (2 layers) | **1.04** |
| Transformer + BiLSTM (1 encoder/decoder + 1 BiLSTM) | 1.06 |
| VAE | 1.06 |

Visual inspection of predicted centerlines revealed that all models could reproduce the general rope configuration but tended to produce slightly **clustered** predictions concentrated near the original region.  
The **BiLSTM** achieved marginally lower error and visually smoother deformations, suggesting that local sequential dependencies dominate rope dynamics in this limited dataset.

## 3.3 Discussion of Results

Although the differences between models are small, several tendencies emerged:

- **Dataset size is the primary limitation.** With only 3,000 samples, models quickly overfit, and larger, more diverse data are needed.  
- **Clustering and low regional diversity.** All models produced predictions biased toward more clustered rope configurations.  
- **BiLSTM achieved best results.** For small datasets, recurrent architectures outperform more modern alternatives.  
- **Transformer and BiLSTM hybrids** capture deformations better but are prone to concentrated predictions.  
- **VAE predictions** show higher variability, representing uncertainty but at reduced positional accuracy.

The experiments highlight that **performance is currently constrained by data diversity**, not architectural sophistication. Future iterations will include over one million samples with varied physical parameters (friction, stiffness, action complexity, and bimanual manipulation).  
Additionally, **latent world models (e.g., Dreamer)** will be explored to capture multimodal dynamics and uncertainty.

# 4. Conclusion

This partial submission presented the initial progress toward building a **probabilistic model for the manipulation of Deformable Linear Objects (DLOs)**. The project addressed the challenges of representing and predicting rope dynamics under applied actions using data generated in **MuJoCo** simulations. A dataset of approximately 3,000 samples was created, encoding rope states as sequences of 3D coordinates and actions as localized forces.

Three neural architectures — **BiLSTM**, **Transformer + BiLSTM**, and **VAE** — were tested to predict the next rope configuration. The **BiLSTM** achieved slightly better results, indicating that local sequential dependencies dominate at this stage. The small dataset size limited generalization capacity, reinforcing the need for larger and more varied data.

The experiments demonstrated the feasibility of learning rope dynamics from simulation but also exposed the limitations of simple deterministic models in capturing uncertainty. This motivates the next phase, which will focus on:

- **Dataset Expansion:** generate over one million samples with varied physical parameters.  
- **Action Complexity:** include multi-step and bimanual manipulations.  
- **Model Enhancement:** explore **latent world models (e.g., Dreamer)** and **spatial transformers**.  
- **Evaluation and Planning:** introduce probabilistic metrics such as **Average Displacement Error (ADE)** and **Dice Coefficient**.

Ultimately, the goal is to obtain a dynamics model that not only predicts the next state of the rope accurately but also **quantifies uncertainty**, enabling **risk-aware planning** for autonomous DLO manipulation tasks.

# 5. Schedule

| Phase | Duration | Description |
|:------|:----------|:-------------|
| Dataset Generation | 2 Weeks | Generate synthetic dataset with diverse parameters |
| Model Development | 4 Weeks | Implement and train model architectures |
| Evaluation | 2 Weeks | Analyze results and model generalization |
| Final Integration | 2 Weeks | Implement risk-aware planning module |

# 6. Bibliographic References

[1] D. Ha and J. Schmidhuber, “World Models,” *CoRR*, vol. abs/1803.10122, 2018. [Online]. Available: https://arxiv.org/abs/1803.10122  

[2] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to Control: Learning Behaviors by Latent Imagination,” *Proc. ICLR 2020*, 2020. [Online]. Available: https://arxiv.org/abs/1912.01603

[3] Yan, Mengyuan, et al. “Self-Supervised Learning of State Estimation for Manipulating Deformable Linear Objects.” ArXiv.org, 2019. [Online]  Available: arxiv.org/abs/1911.06283.
# Presentation Link

[Google Slides Presentation](https://docs.google.com/presentation/d/1fw3_m6minAr5l9Ks6CPWKIoQIB-UsBzjtunPPyIErtY/edit?usp=sharing)
