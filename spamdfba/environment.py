from mapping_matrix import Build_Mapping_Matrix
from distutils.log import warn
import numpy as np
import time

DEFAULT_PLOTLY_COLORS=['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']*10
DEFAULT_PLOTLY_COLORS_BACK=['rgba(31, 119, 180,0.2)', 'rgba(255, 127, 14,0.2)',
                       'rgba(44, 160, 44,0.2)', 'rgba(214, 39, 40,0.2)',
                       'rgba(148, 103, 189,0.2)', 'rgba(140, 86, 75,0.2)',
                       'rgba(227, 119, 194,0.2)', 'rgba(127, 127, 127,0.2)',
                       'rgba(188, 189, 34,0.2)', 'rgba(23, 190, 207,0.2)']*10

class Environment:
    """ 
    An environment is a collection of agents and extracellular reactions

    Args:
        name (str): A name for the environment
        agents (Iterable): An object including the collection of the agents used in the environment
        extracellular_reactions (Iterable): An object consisting of a collection of extracellular reactions
        initial_condition (dict): A dictionary describing the initial concentration of all species in the environment of each state
        inlet_conditions (dict): A dictionary describing the inlet concentration of all species in the environment of each state
        number_of_batches (int): Number of batches performed in a simulation
        dt (float): Specifies the time step for DFBA calculations
        dilution_rate (float): The dilution rate of the bioreactor
        episodes_per_batch (int): Number of episodes executed with same actor function in parallel
        episode_length (int): Number of time points existing within a given episode
        training (bool): Whether to run in training mode. If false, no training happens
        constant (list): A list of components that we want to hold their concentration constant during the simulations

    """
    def __init__(self, name, agents, extracellular_reactions, initial_condition, inlet_conditions, number_of_batches=100,
                 dt=0.1, dilution_rate=0.05, episodes_per_batch=10, episode_length=1000, training=True, constant=[]):
        self.name=name
        self.agents = agents
        self.num_agents = len(agents)
        self.extracellular_reactions = extracellular_reactions
        self.dt = dt
        self.constant=constant
        self.episodes_per_batch=episodes_per_batch
        self.number_of_batches=number_of_batches
        self.dilution_rate = dilution_rate
        self.training=training
        self.mapping_matrix=self.resolve_exchanges()
        self.species=self.extract_species()
        self.resolve_extracellular_reactions(extracellular_reactions)
        self.initial_condition =np.zeros((len(self.species),))
        for key,value in initial_condition.items():
            self.initial_condition[self.species.index(key)]=value
        self.inlet_conditions = np.zeros((len(self.species),))
        for key,value in inlet_conditions.items():
            self.inlet_conditions[self.species.index(key)]=value
        self.set_observables()
        self.set_networks()
        self.reset()
        self.time_dict={
            "optimization":[], "step":[], "episode":[]}
        self.episode_length=episode_length
        self.rewards={agent.name:[] for agent in self.agents}

    def resolve_exchanges(self):
        """
        Determines the exchange reaction mapping for the community
        This mapping is to keep track of metabolite pool change by relating exctacellular concentrations 
        with production or consumption by the agents
        """
        models=[agent.model for agent in self.agents]
        return Build_Mapping_Matrix(models)

    def extract_species(self):
        """
        Determines the extracellular species in the community before extracellula reactions
        """
        species=[ag.name for ag in self.agents]
        species.extend(self.mapping_matrix["Ex_sp"])
        return species

    def resolve_extracellular_reactions(self,extracellular_reactions):
        """ 
        Determines the extracellular reactions for the community
        This method adds any new compounds required to run DFBA to the system
        Args:
            extracellular_reactions list[dict]: list of extracellular reactions
        """
        species=[]
        [species.extend(list(item["reaction"].keys())) for item in extracellular_reactions]
        new_species=[item for item in species if item not in self.species]
        if len(new_species)>0:
            warn("The following species are not in the community: {}".format(new_species))
            self.species.extend(list(set(new_species)))



    def reset(self):
        """ 
        Resets the environment to its initial state
        """
        self.state = self.initial_condition.copy()
        self.rewards={agent.name:[] for agent in self.agents}
        self.time_dict={
            "optimization":[], "step":[], "episode":[]}

    def step(self):
        """
        Performs a single DFBA step in the environment
        This method provides similar interface as other RL libraries
        It returns current state, rewards given to each agent from FBA calculations, actions each agent took,
        and next state calculated similar to DFBA
        """
        self.temp_actions=[]
        self.state[self.state<0]=0
        dCdt = np.zeros(self.state.shape)
        Sols = list([0 for i in range(len(self.agents))])
        for i,M in enumerate(self.agents):
            for index,item in enumerate(self.mapping_matrix["Ex_sp"]):
                if self.mapping_matrix['Mapping_Matrix'][index,i]!=-1:
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].upper_bound=100
                    M.model.reactions[self.mapping_matrix['Mapping_Matrix'][index,i]].lower_bound=-M.general_uptake_kinetics(self.state[index+len(self.agents)])


            for index,flux in enumerate(M.actions):
                if M.a[index]<0:
                    M.model.reactions[M.actions[index]].lower_bound=max(M.a[index],M.model.reactions[M.actions[index]].lower_bound)
                else:
                    M.model.reactions[M.actions[index]].lower_bound=min(M.a[index],10)
            t_0=time.time()
            Sols[i] = self.agents[i].model.optimize()
            self.time_dict["optimization"].append(time.time()-t_0)
            if Sols[i].status == 'infeasible':
                self.agents[i].reward=-1
                dCdt[i] = 0
            else:
                dCdt[i] += Sols[i].objective_value*self.state[i]
                self.agents[i].reward =Sols[i].objective_value*self.state[i]

        for i in range(self.mapping_matrix["Mapping_Matrix"].shape[0]):

            for j in range(len(self.agents)):

                if self.mapping_matrix["Mapping_Matrix"][i, j] != -1:
                    if Sols[j].status == 'infeasible':
                        dCdt[i+len(self.agents)] += 0
                    else:
                        dCdt[i+len(self.agents)] += Sols[j].fluxes.iloc[self.mapping_matrix["Mapping_Matrix"]
                                                    [i, j]]*self.state[j]

        for ex_reaction in self.extracellular_reactions:
            rate=ex_reaction["kinetics"][0](*[self.state[self.species.index(item)] for item in ex_reaction["kinetics"][1]])
            for metabolite in ex_reaction["reaction"].keys():
                dCdt[self.species.index(metabolite)]+=ex_reaction["reaction"][metabolite]*rate
        dCdt+=self.dilution_rate*(self.inlet_conditions-self.state)
        C=self.state.copy()
        for item in self.constant:
            dCdt[self.species.index(item)]=0
        self.state += dCdt*self.dt

        Cp=self.state.copy()
        return C,list(i.reward for i in self.agents),list(i.a for i in self.agents),Cp


    def set_observables(self):
        """ 
        Sets the observables for the agents in the environment
        """
        for agent in self.agents:
            agent.observables=[self.species.index(item) for item in agent.observables]

    def set_networks(self):
        """
        Sets up the networks and optimizers for the agents in the environment
        """
        if self.training==True:
            for agent in self.agents:
                agent.actor_network_=agent.actor_network(len(agent.observables)+1,len(agent.actions))
                agent.critic_network_=agent.critic_network(len(agent.observables)+1,1)
                agent.optimizer_value_ = agent.optimizer_critic(agent.critic_network_.parameters(), lr=agent.lr_critic)
                agent.optimizer_policy_ = agent.optimizer_actor(agent.actor_network_.parameters(), lr=agent.lr_actor)

"""
The code is completely inspired from
https://github.com/chan-csu/SPAM-DFBA.git
"""