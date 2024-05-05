## Genetic Algorithm
Genetic algorithms (GAs) are a powerful class of optimization algorithms inspired by the principles of natural selection. They operate on a population of candidate solutions and iteratively improve them through a process that mimics biological evolution. It begins with the representation, where each candidate solution within the population is encoded as a chromosome. In the context of our discussion, the chromosome might represent a specific gene expression profile, where each gene is encoded as either on or off. The genetic algorithm is shown below:

<img src="https://github.com/anshul-2010/Computational-Systems-Biology/blob/main/images/display/genetic_algorithm.png" alt="Genetic algorithm overflow" width="700"/>

The GA starts with an initial population of chromosomes, which can be randomly generated or seeded with specific solutions if prior knowledge exists. A fitness function is defined to evaluate the performance of each individual within the population. This function assigns a higher score to solutions that are closer to the desired outcome. In our case, the fitness function might evaluate the performance of a particular gene expression profile within the RL-dFBA framework, potentially using the accumulated return as the fitness score. Individuals with higher fitness scores are more likely to be selected for reproduction. This selection process ensures that successful traits are propagated to the next generation. Various selection techniques exist, such as the roulette wheel selection or tournament selection. Selected individuals undergo a process call crossover, where genetic material is exchanged to create offspring. Different crossover operations exist, such as single-point crossover or two-point crossover.

With a low probability, random mutations are introduced into the genetic makeup. It helps maintain diversity in the population and prevents premature convergence towards a local optimum. Different mutation operators can be employed, such as bit-flip mutation for binary encoded chromosomes. A new generation of individuals is formed by combining the offspring with a portion of the parent population. The GA continues iterating through these steps until a termination criterion is met. This criterion could be reaching a defined number of generations, achieving a desired fitness level, or observing stagnation in the improvement of the population.

'''
Algorithm 1 Genetic Algorithm
Require: Initial population and max number of generations to run
the algorithm. Also, tournament selection method is the selection
method incorporated. Single-point crossover and flip mutations
are used.
1: INITIALIZATION
population L99 Initialize-Population(population size)
2: while individual len(population) do
Randomly Initialize population or Take the attractors’ from
the boolean network.
3: end while
4: LOOP (GENERATION LOOP)
5: while generation max gen+1 do
6: Choose parents based on the fitness function (RL-DFBA)
7: for child=0, child population size / 2, child ++ do
8: Select the pair of parents and perform single point crossover
between the parents
9: Mutate(child1) with low probability
10: Mutate(child2) with low probability
11: end for
12: end while
13: Replace the old population with the new generated off-springs
14: OUTPUT
15: Calcuate the best individual in the population and return it
'''

GAs excel at exploring the search space for promising solutions. By leveraging attractor states from the gene regulatory network as the initial population, this exploration can be further directed towards biologically relevant regions of the search space. GAs are also adept at handling complex search spaces and avoid getting trapped in local optima. This is particularly beneficial for RL-dFBA with gene regulation, where the interaction between gene expression and metabolic behavior can lead to intricate fitness landscapes.