from rl_dfba import nn as NN
from rl_dfba import agent as Agent
from rl_dfba import simulation as Simulation
from rl_dfba import environment as Environment
from rl_dfba.mapping_matrix import Build_Mapping_Matrix, general_kinetic, general_uptake, mass_transfer
from rl_dfba.mapping_matrix import run_episode, run_episode_single, rollout
import torch
import os
import cobra
import random

model_base = cobra.io.read_sbml_model("https://github.com/anshul-2010/Computational-Systems-Biology/blob/main/data_model/iJO1366.xml")
medium = model_base.medium.copy()
test_model = model_base.copy()

model_base

## Finding the suitable genes and reactions for our analysis
rxns_itp = ["NTP10", "ATPHs", "ADK4", "NTPP9"]
genes_itp = []
genes_names_itp = []
metabolites_itp = []

# Obtain the gene reaction rules and their corresponding gene ids
for rxn in model_base.reactions:
  if rxn.id in rxns_itp:
    gene_i = rxn.gene_reaction_rule.split(" or ")
    genes_itp += gene_i

# Obtain the gene names corresponding to the gene ids
for gene in model_base.genes:
  if str(gene) in genes_itp:
    genes_names_itp.append(gene.name)

# Obtain the rxn metabolites ids
for rxn in model_base.reactions:
  if rxn.id in rxns_itp:
    metabolites = [meta.id for meta in rxn.metabolites]
    metabolites_itp += metabolites

metabolites_itp = list(set(metabolites_itp))

print(rxns_itp)
print(genes_itp)
print(genes_names_itp)
print(metabolites_itp)

aaa = 0
for rxn in model_base.reactions:
  aaa += 1
  if rxn.id == "BIOMASS_Ec_iJO1366_core_53p95M":
    print(rxn.id)
    print(aaa)

model_base.medium

ic={
    key.lstrip("EX_"):3 for key,val in model_base.medium.items()
}

ic['glc__D_e']=500
ic['agent1']=0.1


# Fitness function
def fitness(individual):
  model1 = model_base.copy()
  gene_list = []
  for i in range(len(individual)):
    if individual[i] == 0:
      gene_list.append(genes_itp[i])

  cobra.manipulation.delete_model_genes(model1, gene_list)
  model1.exchange_reactions = tuple([model1.reactions.index(i) for i in model1.exchanges])
  model1.biomass_ind=model1.reactions.index("BIOMASS_Ec_iJO1366_core_53p95M")
  model1.solver = "glpk"

  agent1 = Agent(
        "agent1",
        model=model1,
        actor_network=NN,
        critic_network=NN,
        clip=0.1,
        lr_actor=0.0001,
        lr_critic=0.001,
        actor_var=0.05,
        grad_updates=10,
        optimizer_actor=torch.optim.Adam,
        optimizer_critic=torch.optim.Adam,
        observables=[
            "agent1",
            "glc__D_e"
        ],
        actions=["BIOMASS_Ec_iJO1366_core_53p95M"],
        gamma=1,
    )
  constants=list(ic.keys())
  constants.remove("agent1")
  constants.remove("glc__D_e")

  env = Environment(
        "Gene_Reg_Ecoli_iJO1366" ,
        agents=[agent1],
        dilution_rate=0,
        extracellular_reactions=[],
        initial_condition=ic,
        inlet_conditions={},
        dt=0.1,
        episode_length=100,
        number_of_batches=10,
        episodes_per_batch=4,
        constant=constants,
    )
  individual_copy = [str(i) for i in individual]
  path = ''.join(individual_copy)
  os.mkdir("https://github.com/anshul-2010/Computational-Systems-Biology/blob/main/results/{}".format(path))
  sim=Simulation(name=env.name,
              env=env,
              save_dir="https://github.com/anshul-2010/Computational-Systems-Biology/blob/main/results/{}/".format(path),
              store_return=[],
              )
  sim.run()
  fitness_score = max(sim.store_return)
  return fitness_score

# Single point crossover
def crossover(a, b):
  length = len(a)
  p = random.randint(1, length - 1)
  a_new = a[0:p] + b[p:]
  b_new = b[0:p] + a[p:]
  return a_new, b_new

# Single point mutation
def mutation(a, p = 0.5):
  length = len(a)
  prob = random.random()
  if prob > p:
    pos = random.randint(1,length-1)
    a[pos] = int(not a[pos])
  return a

# Selection via Tournament Method
def selection(population, fitnesses, k = 3):
  new_population = []
  for i in range(len(population)):
    index_list = [i for i in range(len(population))]
    indices_k = random.sample(index_list, k)
    fitness_compare = [fitnesses[j] for j in indices_k]
    zipped_list = list(zip(fitness_compare, indices_k))
    sorted_zipped_list = sorted(zipped_list)
    sorted_fitness, sorted_indices = zip(*sorted_zipped_list)
    new_population.append(population[sorted_indices[-1]])
  return new_population

fitness_combo = {}
# Initialization
population_size = 4
num_generations = 10
# Initial population
population = [[random.randint(0, 1) for _ in range(5)] for _ in range(population_size)]
# Genetic algorithm loop
for _ in range(num_generations):
  fitnesses = []
  for i in range(len(population)):
    ind = [str(k) for k in population[i]]
    ind = ''.join(ind)
    if ind not in fitness_combo:
      fitness_combo[ind] = fitness(population[i])
    fitnesses.append(fitness_combo[ind])

  selected_individuals = selection(population, fitnesses)
  offspring = []
  for i in range(population_size//2):
      parent1, parent2 = random.sample(selected_individuals, 2)
      child1, child2 = crossover(parent1, parent2)
      offspring.append(child1)
      offspring.append(child2)
  offspring = [mutation(x) for x in offspring]
  population = offspring

# Final population
print(population)

