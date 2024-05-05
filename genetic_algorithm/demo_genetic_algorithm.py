import numpy as np
import random

# Fitness function
def fitness(individual):
  fitness_score = np.sum(individual)
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

# Initialization
population_size = 4
num_generations = 10

# Initial population
population1 = [[random.randint(0, 1) for _ in range(5)] for _ in range(population_size)]
# Genetic algorithm loop
for _ in range(num_generations):
  print(population, _)
  # fitnesses = [fitness(individual) for individual in population]
  fitnesses = []
  for i in range(len(population)):
    fitnesses.append(fitness(population[i]))

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