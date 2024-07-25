import numpy as np

# Example 7x3 population array
population = np.random.rand(7, 3)

# Assuming the last column represents the fitness of the individuals
# If the fitness needs to be computed, replace this with the appropriate fitness function
fitness = population[:, -1]

# 1. Roulette Wheel Selection
def roulette_wheel_selection(population, fitness, num_select):
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    selected_indices = np.random.choice(len(fitness), size=num_select, p=probabilities)
    return population[selected_indices]

# 2. Tournament Selection
def tournament_selection(population, fitness, num_select, tournament_size=3):
    selected_indices = []
    for _ in range(num_select):
        participants = np.random.choice(len(fitness), size=tournament_size, replace=False)
        best = participants[np.argmax(fitness[participants])]
        selected_indices.append(best)
    return population[selected_indices]

# 3. Rank-Based Selection
def rank_based_selection(population, fitness, num_select):
    sorted_indices = np.argsort(fitness)
    ranks = np.arange(1, len(fitness) + 1)
    total_ranks = np.sum(ranks)
    probabilities = ranks / total_ranks
    selected_indices = np.random.choice(sorted_indices, size=num_select, p=probabilities)
    return population[selected_indices]

# Number of individuals to select
num_select = 5

# Perform selections
selected_by_roulette = roulette_wheel_selection(population, fitness, num_select)
selected_by_tournament = tournament_selection(population, fitness, num_select, tournament_size=3)
selected_by_rank = rank_based_selection(population, fitness, num_select)

print("Selected by Roulette Wheel Selection:\n", selected_by_roulette)
print("Selected by Tournament Selection:\n", selected_by_tournament)
print("Selected by Rank-Based Selection:\n", selected_by_rank)
