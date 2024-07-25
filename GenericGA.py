import numpy as np
import csv
import matplotlib.pyplot as plt

###################################################################################################################
###################################################################################################################
#parsing the data from the CSV file
def read_csv(file_path):
    #Read data from CSV file, convert "," to "." and extract displacement and force values.
    time = []
    displacement = []
    force = []
    deflection = []
    bending_stress = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            next(reader)  # Skip header
            for row in reader:
                # Replace commas with periods
                row = [entry.replace(',', '.') for entry in row]
                if float(row[2]) >= 0:
                    bending_stress.append(float(row[4]))
                    deflection.append(float(row[3]))
                    displacement.append(float(row[2]))  # Column 3 for Displacement
                    force.append(float(row[1]))
                    time.append(float(row[0])) # Column 2 for Force
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print("An error occurred:", e)
    return time, force, displacement, deflection, bending_stress
###################################################################################################################
#Defining a fitness function with regards to multi-objective optimization, maximizing the force, while minimizing time and displacement
def fitness(myarray):
    time, displacement, force = myarray
    # Adjust weights as needed
    weight_time = 0.5
    weight_displacement = 0.5
    weight_force = 1.0
    fitness = weight_force * force - (weight_time * time + weight_displacement * displacement)
    if fitness < 0:
        fitness = -fitness
    return fitness

####################################################################################################################
#################################### SELECTION METHODOLOGIES #######################################################
####################################################################################################################

# 1. Roulette Wheel Selection
def roulette_wheel_selection(population, fitness, num_select):
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    selected_indices = np.random.choice(len(fitness), size=num_select, p=probabilities)
    return population[selected_indices], selected_indices

# 2. Tournament Selection
def tournament_selection(population, fitness, num_select, tournament_size=3):
    selected_indices = []
    for _ in range(num_select):
        participants = np.random.choice(len(fitness), size=tournament_size, replace=False)
        best = participants[np.argmax(fitness[participants])]
        selected_indices.append(best)
    return population[selected_indices], selected_indices

# 3. Rank-Based Selection
def rank_based_selection(population, fitness, num_select):
    sorted_indices = np.argsort(fitness)
    ranks = np.arange(1, len(fitness) + 1)
    total_ranks = np.sum(ranks)
    probabilities = ranks / total_ranks
    selected_indices = np.random.choice(sorted_indices, size=num_select, p=probabilities)
    return population[selected_indices], selected_indices

###################################################################################################################
########################################## CROSSOVER METHODOLOGIES ################################################
###################################################################################################################

# 1. One-Point Crossover
def one_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

# 2. Two-Point Crossover
def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(np.random.choice(range(1, len(parent1)), size=2, replace=False))
    child1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
    child2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
    return child1, child2

# 3. Uniform Crossover
def uniform_crossover(parent1, parent2):
    mask = np.random.rand(len(parent1)) > 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2

# 4. Arithmetic Crossover
def arithmetic_crossover(parent1, parent2, alpha=0.5):
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2
###################################################################################################################

def mutation(child1, child2, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        child1[0] = child1[0] + 0.01
        child2[0] = child2[0] + 0.01

    return [child1, child2]

###################################################################################################################
###################################################################################################################
#path to the csv file
mypath = 'Daten_Saeed/CA_MMB_M1_0,77.csv'
#read csv and extract all the parameters
time, force, displacement, deflection, bending_stress = read_csv(mypath)
mid = int(len(force)/2)

#Defining the Genetic Algorithm parameters
population_size = len(force)
num_generations = 50
mutation_rate = 0.1
crossover_rate = 0.8

#creating the population
population = np.column_stack((time, displacement, force))
pop1 = population

for generation in range(num_generations):
    #calculating the fitness
    fitness_arr = []
    for myarray in population:
        fitness_arr.append(fitness(myarray))
    #number of individuals to select
    num_select = 2
    #num_select = 0.5*len(population)
    selected_parents, selected_indices = roulette_wheel_selection(population, fitness_arr, num_select)
    child1, child2 = one_point_crossover(selected_parents[0], selected_parents[1])
    mutated_pop = mutation(child1, child2)
    for index in selected_indices:
        i = 0
        population[index] = mutated_pop[i]
        i = i+1
#print("pop1", pop1)
#print("population", population)
time_fin, displacement_fin, force_fin = np.column_stack(population)
plt.figure()
plt.scatter(displacement_fin, force_fin, color='blue')
plt.xlabel('Displacement (mm)')
plt.ylabel('Force (N)')
plt.title('Displacement-Time Graph')
plt.grid(True)
plt.show()