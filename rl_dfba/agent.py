from nn import NN
from mapping_matrix import general_uptake
from distutils.log import warn
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class Agent:
    """ 
    This defines an agent in the environment. The core of an agent is essentially a COBRA model
    Observed environment states need to be known for an agent
    Also, the set of reactions that the agent can manipulate

    Args:
        name (str): A name given to an agent.
        model (cobra.Model): A cobra model describing the metabolism of the agent. In our case iJO1366
        actor_network (NN): The neural network class, policy function module, to be used for the actor network
        critic_network (NN): The neural network class, value function module, to be used for the critic network
        optimizer_critic (torch.optim.Adam): The Adam optimizer for tuning the critic network
        optimizer_actor (torch.optim.Adam): The Adam optimizer for tuning the actor network
        actions (list): list of reaction names that the agent has control over
        observables (list): list of the names of metabolites surrounding the agent
        clip (float): gradient clipping threshhold in PPO algorithm
        actor_var (float): Amount of exploration in the actor network
        grad_updates (int): Number of steps of gradient decent in each training step
        lr_actor (float) : The learning rate for the actor network
        lr_critic (float) : The learning rate for the critic network

    """
    def __init__(self, name, model, actor_network, critic_network, optimizer_critic, optimizer_actor,
                actions, observables, gamma, clip=0.01, actor_var=0.1, grad_updates=1, lr_actor=0.001,
                lr_critic=0.001):

        self.name = name
        self.model = model
        self.optimizer_critic = optimizer_critic
        self.optimizer_actor = optimizer_actor
        self.gamma = gamma
        self.observables = observables
        self.actions = [self.model.reactions.index(item) for item in actions]
        self.observables = observables
        self.general_uptake_kinetics=general_uptake
        self.clip = clip
        self.actor_var = actor_var
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.grad_updates = grad_updates
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.cov_var = torch.full(size=(len(self.actions),), fill_value=0.1)
        self.cov_mat = torch.diag(self.cov_var)

    def get_actions(self,observation:np.ndarray):
        """
        This method will draw the actions from a normal distribution around the actor network prediction
        """
        mean = self.actor_network_(torch.tensor(observation, dtype=torch.float32)).detach()
        dist = Normal(mean, self.actor_var)
        action = dist.sample()
        log_prob =torch.sum(dist.log_prob(action))
        return action.detach().numpy(), log_prob

    def evaluate(self, batch_obs:np.ndarray ,batch_acts:np.ndarray):
        """
        Calculates the value of the states, as well as the log probability of the actions taken
        """
        V = self.critic_network_(batch_obs).squeeze()
        mean = self.actor_network_(batch_obs)
        dist = Normal(mean, self.actor_var)
        log_probs = torch.sum(dist.log_prob(batch_acts),dim=1)

        return V, log_probs

    def compute_rtgs(self, batch_rews:list):
        """
        Given a batch of rewards , it calculates the discouted return for each state for that batch
        """

        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs    
    
"""
The code is completely inspired from
https://github.com/chan-csu/SPAM-DFBA.git
"""