# `<Project Title in Portuguese>`
# `<Modeling the Manipulation of Deformable Linear Objects>`

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

> |Name  | RA | Specialization|
> |--|--|--|
> | Tim Missal  | 298836  | Computer Science |
> | Lucas Vinícius Domingues  | 291414  | Computer Science|
> | Natália da Silva Guimarães  | 298997 | XXX|

## Project Summary Description

The autonomous manipulation of Deformable Linear Objects (DLOs) using robotic arms has proven to be a difficult task for mainly two reasons: the perceived state of the DLO might not reflect reality perfectly, as different textures or overlaps can lead to ambiguity; and actions performed on the DLO might lead to unexpected results due to high friction in entangled states. 

Classical methods used in some of the main applications of the manipulation of DLOs, such as Knot Tying and Knot Untangling, fail to capture the uncertainty in performed actions and instead calculate the best possible action to, e.g. simplify a DLO state, analytically. Since this doesn't account for the uncertainty that can result from the reasons mentioned above, the best action might lead to a state that is much worse than before. In other words, the best action might be a potentially risky one. So, instead of choosing actions with the highest immediate reward, we believe an agent should pursue actions which minimize the risk over the long term. To do this, a Dynamics Model is needed that, given a perceived state and an action, can predict a probability distribution over next states. Using this (world) model, an agent could be developed which can make better decisions, as famously shown in the Dreamer Paper [1, 2].

To summarize, we want to create a model with the following characteristics:

- Input:
  - State: $s_t$ DLO state given as a rope centerline. We decide not to work on images of DLOs such as ropes and cables, since the perception of a DLO centerline out of an image is a different topic that is studied in seperate channels. Instead, we assume to have a model that generates the state information for us.
  - Action: $a_t$ An action that is performed on some segment of the DLO. For example, an action might be: pull some segment in the middle of the rope in direction xyz with force x N.
- Output: $s_{t+1}$ The state of the DLO at the next timestep, after performing $a_t$ on $s_t$.

The goal of having such a model is to be able to quantify whether or not executing action a in state s is a good or bad decision by calculating the actions risk. Work that builds upon this model could then use it to train a high-level planner for tasks that involve the manipulation of DLOs.

## Proposed Methodology

1. Generate own Dataset of ($s_t$, $a_t$, $s_{t+1}$) using MuJoCo by performing random actions on a piece of rope. This gives us the freedom to generate more / different data as needed.
2. Train Model using Dataset
    - For this, the preferred approach is to build a World Model that can correctly capture the dynamics of the DLO. Word models learn to predict the future behavior of phenomena conditioned on actions and based on latent representations of past episodes, making them great candidates for the creation of generalizable models.
    - For this task, different architechures can be tested, such as Transformers, VAEs, or even whole frameworks like Dreamer [1]

4. Evaluate the model. Some candidate metrrics can be:
    - Average Displacement Error (ADE): Given predicted centerline $s_{t_1}$ and ground-truth $y_t$, where each is a sequence of L ordered 3D points, calculate average euclidean distance over the L points. (overlap between predicted next state and real next state)
    - Dice Coefficient: Measures the similarity between two sets or spatial data

## Schedule

Total: 10 Weeks

1. Dataset Generation: 2 Weeks
2. Model: 4 Weeks
3. Evaluation: 2 Weeks

## Bibliographic References

[1] D. Ha and J. Schmidhuber, “World Models,” *CoRR*, vol. abs/1803.10122, 2018. [Online]. Available: https://arxiv.org/abs/1803.10122  

[2] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to Control: Learning Behaviors by Latent Imagination,” in *Proc. ICLR 2020*, 2020. [Online]. Available: https://arxiv.org/abs/1912.01603
