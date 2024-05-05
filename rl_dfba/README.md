## RL-dFBA
Reinforcement Learning (RL) offers a powerful framework for training agents to learn optimal decision-making strategies through interaction with an environment. An agent, within this paradigm, is an active learner that explores its environment and selects the set of actions that maximize the desired rewards. The environment encompasses the external world surrounding the agent, proving the state information and responding to agent’s actions. The reward function assigns a numerical value to the agent, reflecting the desirability of the state-action pair (s, a). This signal serves as the guiding principle for the agent’s learning process. The agent then utilizes the received reward to update its policy function (π), aiming to favor actions that consistently lead to higher cumulative rewards.

Here, in our approach, RL agent represents a microbial system’s metabolic network within the community. The state of the environment, as perceived by the agent, is characterized by the continuous measurement of specific extracellular metabolic concentrations. These concentrations offer valuable insights into the community’s overall metabolic state. To influence the behavior, the agent takes actions that corresponds to adjustments in reaction fluxes within the metabolic model. Both the action space and state space are observed to be continuous. The policy gradient family of algorithms have been observed to work well in such continuous scenarios. 

Within the domain of RL, policy gradient algorithms offer a compelling approach. These algorithms achieve this by directly manipulating the agent’s policy function, a parameterized function (often a neural network) that maps a state to an action. The parameters of this policy function are fine-tuned to incentivize the actions that lead to higher returns. However, a significant challenge arises when the underlying mathematical operations within the environment involve linear programming (LP) problems, as is the case with dFBA in our framework. The actions recommended by the policy function can fall outside the feasible region of the LP, hindering the training process. PPO, addresses this challenge by introducing the concept of policy proximity.

Incorporating these elements into our framework led to the stabilization of the training process. In our current framework, we have neural networks, each composed of 10 linear layers and a tanh activation function for both the policy function (actor network) and the value function (critic network). The interaction between the actor and critic models within the environment is visually depicted below. We have chosen the Adam optimizer for the training. This combination of PPO leads to a robust learning in microbial optimization framework.

![Actor Critic Model](https://github.com/anshul-2010/Computational-Systems-Biology/blob/main/images/display/Actor_Critic.png)