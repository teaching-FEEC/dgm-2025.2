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

This project aims to develop a **probabilistic dynamics model** for **Deformable Linear Objects (DLOs)**, which is fundamental for autonomous manipulation with risk awareness. We generated a synthetic dataset of 1 million transitions (**state, action, next state**) using the **MuJoCo** simulator. Architectures such as **BiLSTM**, **BERT**, **Transformer**, **Diffusion-Models** and **Dreamer** were evaluated to predict the rope's future configuration. The best performance was shown by the **Dreamer** model.

---

## 1.Problem Description / Motivation

The autonomous manipulation of Deformable Linear Objects (**DLOs**), such as cables and ropes, is a significant challenge due to ambiguity in perception (like overlaps) and **unpredictable action outcomes** caused by high friction in entangled states.

Classical methods fail to **quantify this uncertainty**, opting for actions that maximize immediate gain but may lead to high-risk states in the long term. The central motivation of this project is to develop a **model** that predicts a **probability distribution** over future states, allowing the agent to make decisions that **minimize risk** over an extended time horizon, rather than simply pursuing instantaneous reward.

## 2.Objectives

Develop a **dynamics model** capable of predicting the future state and **quantifying the uncertainty** of a Deformable Linear Object (**DLO**), given its current configuration and an applied action.

## 3. Methodology

### 3.1 Overview

The project aims to develop a model capable of predicting the future state of a **Deformable Linear Object (DLO)** given its current configuration and an applied action. This predictive model serves as the foundation for future development of **risk-aware agents** for autonomous DLO manipulation. Our workflow for our most successful model, the **Dreamer**, is described in the following picture. Workflows for other models can be found in the attached presentation. 

![Dreamer-Architecture](path/to/image.jpg)

### 3.2 Dataset Generation

To train and evaluate the proposed models, a synthetic dataset of **(state, action, next_state)** tuples was generated using the **MuJoCo physics engine** (Multi-Joint dynamics with Contact). MuJoCo is a widely used simulation framework for robotics and deformable object modeling due to its accurate contact and joint dynamics.

#### Simulation Setup

- The rope was modeled as a chain of **70 connected cylinders** linked through ball joints, simulating a flexible, deformable body.  
- **State representation ($\mathbf{s}_{t}$):** The 3D coordinates $(\mathbf{x}, \mathbf{y}, \mathbf{z})$ of the rope’s 70 cylinders.  
- **Action representation ($\mathbf{a}_{t}$):** A 3D force vector applied to a single cylinder at each step.  
- **Next state ($\mathbf{s}_{t+1}$):** Rope configuration after the force is applied.  

### 3.3 Modeling Approaches

Three model architectures were evaluated to learn the rope dynamics:

#### (a) BiLSTM

- Based on the model proposed by Yan et al. (2019) [2], which demonstrated strong results for state estimation in DLO manipulation.  
- The **Bidirectional LSTM** captures both forward and backward dependencies between rope links, essential since each link’s motion depends on its neighbors on both sides.  
- **Residual connections** were added to allow the model to predict position deltas, improving stability and convergence.

#### (b) BERT

#### (c) Transformer

#### (d) Diffusion

#### (e) Dreamer

We employ a "World Model" architecture to solve a supervised problem: predicting the next state of a deformable object (rope) given its current state and an applied action. Rather than optimizing on the high-dimensional 3D coordinates directly, we learn the transition dynamics in a compact latent space.

#### Training Setup

- **1.000.000 samples** used 
- Models trained for **50 epochs**  
- Checkpointing based on **validation loss (MSE)**  
- Data **normalized** using the individual center of mass (CoM) of datapoints

## 4. Evaluation Methodology

### 4.1 Metrics

Model performance was evaluated using:

- **Mean Squared Error (MSE):** Quantifies the average Euclidean distance between predicted and ground-truth rope coordinates.

  $\text{MSE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1}(y_i - \hat{y}_i)^2$

  where *N* is the number of rope links, $Y_i$ is the actual position of the rope link, and $\hat{Y}_i$ is the predicted position. A lower MSE indicates a better fit of the model to the data.

- **MSE on Autoregression:** Quantifies how much the predicted values differ from the ground-truth over 1000 steps

- **Qualitative Visualization:** Predicted and true rope shapes were visualized to assess spatial similarity and clustering tendencies.


### 4.2 Assessment Criteria

The objectives are considered met when:

1. The model accurately predicts next-state configurations (**low MSE/ADE**).  
2. The model generalizes to unseen actions and configurations.  

## 5. Experiments, Results, and Discussion

### 5.1 Experimental Setup

A dataset of approximately 1.000.000 samples was generated in **MuJoCo**, simulating a rope composed of 70 interconnected cylindrical segments with ball joints. Each sample consisted of a state $s_{t}$ (the 3D positions of all rope segments), an action $a_{t}$ (a force vector applied to a specific segment), and the resulting next state $s_{t+1}$.  
The dataset was normalized using a Center-of-Mass approach and divided into training, validation, and test sets. All models were trained using the **Physics Informed Loss** (Source?) loss and evaluated using **Mean Squared Error** (MSE)


### 5.2 Results

The tested model families and the corresponding results were the following: 

| Model | Params | MSE |
|--------|-----|
| BiLSTM  | 1.12M |  1.43 |
| BERT  | 1.9M |  1.43 |
| Transformer  | 407K |  1.45 |
| Diffusion  | 4 |  3 |
| Dreamer  | 11M |  1.16 |



Visual inspection of predicted centerlines revealed that the BiLSTM, BERT and Transformer failed to predict the rope states accurately. Instead, these models produced rope states which were clustered near the rope positions. In contrast, Dreamer seemed to follow the general structure of the rope rather accurately.

### 5.3 Discussion of Results

Several tendencies emerged:

- **Clustering and low regional diversity.** Some models produced predictions biased toward more clustered rope configurations.  
- **Dreamer achieved best results.** ...  
- **BiLSTM, BERT and Transformers perform way worse than Dreamer.** This suggests that models which do not consider temporal steps tend to perform worse, especially on pure autoregressive scenarios
- **Something about Diffusion maybe** 

The experiments highlight that...

## 6. Conclusion

This final submission presents the progress towards building a **probabilistic model for the manipulation of Deformable Linear Objects (DLOs)**. The project addressed the challenges of representing and predicting rope dynamics under applied actions using data generated in **MuJoCo** simulations. A dataset of 1 million samples was created, encoding rope states as sequences of 3D coordinates and actions as localized forces.

Five neural architectures — **BiLSTM**, **BERT**, **Transformer**, **Diffusion Models** and **Dreamer** — were tested to predict the next rope configuration. While the BiLSTM, BERT and the Transformer failed to accurately predict rope states, Dreamer showed promising results. We believe that Dreamer achieved better results due to it's approach of splitting the learning into a deterministic and a stochastic part.

Ultimately, the goal to obtain a dynamics model which accurately predicts rope dynamics has been reached, with room for improvement. Future work could focus on repeating our experiments using bigger and more optimized models, especially using Dreamer, which seemed to be the most promising of our tested models. 



## 7. Bibliographic References

[1] D. Ha and J. Schmidhuber, “World Models,” *CoRR*, vol. abs/1803.10122, 2018. [Online]. Available: https://arxiv.org/abs/1803.10122  

[2] Yan, Mengyuan, et al. “Self-Supervised Learning of State Estimation for Manipulating Deformable Linear Objects.” ArXiv.org, 2019. [Online]  Available: arxiv.org/abs/1911.06283.

[3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. Advances in neural information processing systems, 30. NIPS papers. You can also cite it via its arXiv number: Vaswani, A., et al. (2017). Attention Is All You Need. arXiv preprint arXiv:1706.03762.  [Online] Available https://arxiv.org/abs/1706.03762. 

[4] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to Control: Learning Behaviors by Latent Imagination,” *Proc. ICLR 2020*, 2020. [Online]. Available: https://arxiv.org/abs/1912.01603



# Presentation Link

[Google Slides Presentation](https://docs.google.com/presentation/d/14xAsvx7EaFsWOUKr3ieodYBcjxsf4goc3SWKETlq8cc/edit?usp=sharing)
