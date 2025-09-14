# `<Project Title in Portuguese>`
# `<Modeling the Manipulation of Deformable Linear Objects>`

## Presentation

This project originated in the context of the graduate course *IA376N - Generative AI: from models to multimodal applications*, 
offered in the second semester of 2025, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

> Include name, RA, and specialization focus of each group member. Groups must have at most three members.
> |Name  | RA | Specialization|
> |--|--|--|
> | Name1  | 123456  | Computer Engineering|
> | Name2  | 123456  | Electrical Engineering|
> | Name3  | 123456  | XXX|

## Project Summary Description

The autonomous manipulation of Deformable Linear Objects (DLOs) using robotic arms has proven to be a difficult task for mainly two reasons: the perceived state of the DLO might not reflect reality perfectly, as different textures or overlaps can lead to ambiguity; and actions performed on the DLO might lead to unexpected results due to high friction in entangled states. 

Classical methods used in some of the main applications of the manipulation of DLOs, such as Knot Tying and Knot Untangling, fail to capcure the uncertainty in performed actions and instead calculate the best possible action to, e.g. simplify a DLO state, analytically. Since this doesn't account for the uncertainty that can result from the reasons mentioned above, the best action might lead to a state that is much worse than before. In other worse, the best action might be a potentially risky one. So, instead of choosing actions with the highest immediate reward, we believe an agent should pursue actions which minimize the risk over the long term. To do this, a Dynamics Model is needed that, given a perceived state and an action, can predict a probability distribution over next states. Using this model, an agent could be developed which can make better decisions (link Dreamer?).

To summarize, we want to create a model with the following characteristics:

- Input:
  - State: s_t DLO state given as a rope centerline. We decide not to work on images of DLOs such as ropes and cables, since the perception of a DLO centerline out of an image is a different topic that is studied in seperate channels. Instead, we assume to have a model that generates the state information for us.
  - Action: a_t An action that is performed on some segment of the DLO. For example, an action might be: pull some segment in the middle of the rope in direction xyz with force x N.
- Output: s_t+1 The state of the DLO at the next timestep, after performing a_t on s_t.

The goal of that model is to be able to quantify whether or not executing action a in state s is a good or bad decision. Work that builds upon this model could then  
> Description of the project theme, including generating context and motivation.  
> Description of the main goal of the project.  
> Clarify what the output of the generative model will be.  
>   
> Include in this section a link to the presentation video of the project proposal (maximum 5 minutes).

## Proposed Methodology

1. Generate own Dataset of (s_t, a_t, s_t+1) using MuJoCo by performing random actions on a piece of rope. This gives us the freedom to generate more / different data as needed.
2. Train Model using Dataset (Transformer? Diffusion?)
3. Evaluate using dice coefficient (overlap between predicted next state and real next state)


> For the first submission, the proposed methodology must clarify:  
> * Which dataset(s) the project intends to use, justifying the choice(s).  
> * Which generative modeling approaches the group already sees as interesting to be studied.  
> * Reference articles already identified and that will be studied or used as part of the project planning.  
> * Tools to be used (based on the group’s current vision of the project).  
> * Expected results.  
> * Proposal for evaluating the synthesis results.  

## Schedule
> Proposed schedule. Try to estimate how many weeks will be spent on each stage of the project.  

Total: 10 Weeks

1. Dataset Generation: 2 Weeks
2. Model: 4 Weeks
3. Evaluation: 2 Weeks

## Bibliographic References

https://arxiv.org/abs/1803.10122

https://arxiv.org/pdf/1912.01603

> Point out in this section the bibliographic references adopted in the project.
