# Optimizing Microbial Behavior: Integrating Reinforcement Learning and Genetic Algorithms with dFBA

## Overview
Microbial communities exhibit emergent behaviors that are difficult to predict from individual cell characteristics. Current modeling approaches often struggle to capture the interplay between metabolism, gene regulation, and collective decision-making within these communities. Several works consider this interplay between metabolic processes and gene regulation to capture the intricate mechanisms governing metabolic reactions. Few of the approaches incorporate a set of flexible logic rules for gene regulation into a metabolic model. While integrating the gene regulation with metabolic modeling offers a more comprehensive understanding of microbial communities, it presents significant challenges for capturing long-term adaptation strategies. Existing approaches often prioritize immediate metabolic optimization. Focus on short-term efficiency may not adequately represent the dynamics and trade-offs needed to thrive in fluctuating environments. Additionally, selecting the most relevant genes for accurate dFBA modeling remains a challenge.

## Table of Contents
* [Brief of or Work](#brief-work)
* [Pipeline Architecture](#pipeline-architecture)
* [Core components in the framework](#core-components)
    * [Reinforcement Learning](#reinforcement-learning)
    * [ Proximal Policy Optimization](#proximal-policy-optimization)
    * [Reinforcement Learning (RL) - Dynamic Flux Balance Analysis (dFBA)](#rl-dfba)
    * [Gene Regulatory Boolean Networks](#gene-regulatory-boolean-networks)
    * [Genetic Algorithms](#genetic-algorithms)
* [Contributing](optional)
* [License](#license)
* [Getting Help](#getting-help)
- Data
- Models
- Possible future work

## Brief Work
This study proposes a novel framework that combines three methods, dynamic flux balance
analysis for metabolic modeling, boolean networks for capturing gene regulatory interactions,
and reinforcement learning to simulate microbial decision-making. A key innovation is the
incorporation of a genetic algorithm to optimize the gene selection for the dFBA model. This
optimization process removes redundancy and ensures the model focuses on the most relevant
genes, leading to a more accurate representation of the community’s metabolic capabilities.
The RL agent then interacts with a combined dFBA-boolean network model, representing the
microbial community, and learns strategies to optimize its state based on a defined reward.
We apply this framework to a simulated microbial community, Escherichia coli with well-defined
metabolic pathways and gene regulatory networks. The GA will be employed to identify the
optimal subset of genes for the dFBA model. The interaction of the RL agent with the model will
generate data on emergent properties exhibited by the simulated community. We observe how the
interplay between metabolic constraints, gene regulation, and RL-driven decision-making leads
to the emergence of complex behaviors in the community. This will provide valuable insights
into how microbial communities function and potentially unveil new strategies for manipulating
them for desired functionalities.

## Pipeline Architecture
Here is our proposed pipeline or the same:

![The flowchart of the described module](https://github.com/anshul-2010/Computational-Systems-Biology/blob/main/images/display/Flowchart.jpg)

## Core Components
### Reinforcement Learning
Reinforcement Learning (RL) offers a powerful framework for training agents to learn optimal decision-making strategies through interaction with an environment. An agent, within this paradigm, is an active learner that explores its environment and selects the set of actions that maximize the desired rewards. The environment encompasses the external world surrounding the agent, proving the state information and responding to agent’s actions. 
$$\pi(a|s, \theta) = P(A_{t} = a | S_{t} = s, \theta_{t} = \theta)$$
$$v_{\pi} = E_{\pi}[G_{t}|S_{t} = s]$$

The policy function (π), as shown in the equation 1, plays a central role in RL, representing the agent’s decisionmaking strategy. It maps the given state (s) of agent to a probability distribution over possible actions (a). This function dictates the agent’s behavior and is continuously refined through the learning process to favor actions that yield higher expected rewards. The value function (v_π), as seen in equation 2, quantifies the desirability of being in that state within the context of the chosen policy. The ultimate objective of an RL agent is to maximize its return (Gt).
$$G_{t} = \sum_{k=0}^{T} \gamma^{k}.R_{t+k+1}$$

This metric in equation 3 represents the discounted sum of all future rewards the agent anticipates receiving from a specific state (t). The discount factor (γ) determines the importance of future rewards. Here, the inclusion of this factor ensures that long-term consequences are duly considered.

### Proximal Policy Optimization
A significant challenge arises when the underlying mathematical operations within the environment involve linear programming (LP) problems, as is the case with dFBA in our framework. The actions recommended by the policy function can fall outside the feasible region of the LP, hindering the training process.
$$L_{CLIP}(\theta) =  \hat{E_{t}}[min(r_{t}(\theta)\hat{A_{t}}, clip(r_{t}(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A_{t}})]$$

PPO addresses this challenge by introducing the concept of policy proximity. This approach strives to mitigate abrupt changes in the policy space during updates, ensuring the suggested actions remain within the feasible region of the LP problem. The core principle lies in utilizing a surrogate objective function, as seen in equation 4. This function essentially compels the policy to prioritize actions that yield higher returns while maintaining a degree of similarity to the previous policy.

### RL-dFBA
### Gene Regulatory Boolean Networks
### Genetic Algorithms

