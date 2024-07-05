import random

# Define the G2 function
def calculate_G2(P, a, B, E, alpha):
    try:
        G2 = (6 * P**2 * a) / (B * E * (1 + alpha))
    except ZeroDivisionError:
        print("Error: Division by zero in G2 calculation.")
        return None
    except OverflowError:
        print("Overflow error in G2 calculation.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in G2 calculation: {e}")
        return None
    return G2

# Parameters for the genetic algorithm
POPULATION_SIZE = 100
GENERATIONS = 20
MUTATION_RATE = 0.01

# Create an individual
def create_individual():
    return {
        'P': random.uniform(0.1, 10),
        'a': random.uniform(0.1, 10),
        'B': random.uniform(0.1, 10),
        'E': random.uniform(0.1, 10),
        'alpha': random.uniform(0.1, 10)
    }

# Initialize population
def initialize_population(size):
    return [create_individual() for _ in range(size)]

# Evaluate fitness of an individual
def evaluate_fitness(individual):
    G2 = calculate_G2(individual['P'], individual['a'], individual['B'], individual['E'], individual['alpha'])
    if G2 is None:
        return float('inf')  # Handle invalid G2 values
    return abs(G2 - target_G2)  # Assume target_G2 is the goal value

# Mutation: mutate individual genes
def mutate(individual):
    for key in individual.keys():
        if random.random() < MUTATION_RATE:
            individual[key] = random.uniform(0.1, 10)
    return individual

# Genetic Algorithm with only mutation
def genetic_algorithm():
    population = initialize_population(POPULATION_SIZE)
    best_individuals = []

    for generation in range(GENERATIONS):
        fitnesses = [evaluate_fitness(values) for values in population]
        ranked_population = sorted(zip(population, fitnesses), key=lambda x: x[1])

        best_population= ranked_population[-1]
        # Record the best individual of this generation
        best_individuals.append(best_population)

        # Generate new population through mutation
        new_population = []


        # Elitism: keep the best individual
        #new_population.append(ranked_population[0][0])

        for i in range (len(population)):
            #individual = random.choice(population)
            new_individual = mutate(population[i])
            new_population.append(new_individual)

        population = new_population

    # Rank the best individuals across all generations
    return best_individuals

# Target G2 value for the fitness function
target_G2 = 50.0

# Run the genetic algorithm
best_individuals = genetic_algorithm()

# Print the best individuals
for ind in best_individuals:  # Print top 10 best individuals
    print(ind)