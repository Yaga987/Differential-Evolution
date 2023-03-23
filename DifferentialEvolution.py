import numpy as np

# Define the problem to optimize
def objective_function(x):
    return x[0]**2 + x[1]**2

# Define the DE algorithm function
def differential_evolution(objective_function, bounds, population_size=10, crossover_probability=0.5, mutation_factor=0.8, max_iterations=1000):
    # Initialize the population randomly within the bounds
    population = np.random.rand(population_size, len(bounds))
    population = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * population
    
    # Evaluate the objective function for each individual in the population
    fitness = np.array([objective_function(individual) for individual in population])
    
    # Iterate until the maximum number of iterations is reached
    for i in range(max_iterations):
        # Select three distinct individuals randomly from the population
        indexes = np.random.choice(population_size, 3, replace=False)
        x1, x2, x3 = population[indexes]
        
        # Generate a new trial individual by combining the three selected individuals using the DE strategy
        trial_individual = x1 + mutation_factor * (x2 - x3)
        
        # Crossover the trial individual with the original individual using the crossover probability
        crossover_mask = np.random.rand(len(bounds)) < crossover_probability
        trial_individual[crossover_mask] = population[indexes[0]][crossover_mask]
        
        # Evaluate the fitness of the trial individual
        trial_fitness = objective_function(trial_individual)
        
        # Replace the original individual with the trial individual if it has better fitness
        if trial_fitness < fitness[indexes[0]]:
            population[indexes[0]] = trial_individual
            fitness[indexes[0]] = trial_fitness
    
    # Return the best individual found
    best_index = np.argmin(fitness)
    return population[best_index], fitness[best_index]

# Define the bounds of the problem
bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])

# Run the DE algorithm to optimize the problem
best_solution, best_fitness = differential_evolution(objective_function, bounds)

# Print the best solution and fitness found
print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
