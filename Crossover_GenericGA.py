import numpy as np

# Example 2x3 parent arrays
parent1 = np.array([0.1, 0.2, 0.3])
parent2 = np.array([0.4, 0.5, 0.6])

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

# Perform crossovers
child1_one_point, child2_one_point = one_point_crossover(parent1, parent2)
child1_two_point, child2_two_point = two_point_crossover(parent1, parent2)
child1_uniform, child2_uniform = uniform_crossover(parent1, parent2)
child1_arithmetic, child2_arithmetic = arithmetic_crossover(parent1, parent2)

print("Parent 1:", parent1)
print("Parent 2:", parent2)
print("One-Point Crossover:\nChild 1:", child1_one_point, "\nChild 2:", child2_one_point)
print("Two-Point Crossover:\nChild 1:", child1_two_point, "\nChild 2:", child2_two_point)
print("Uniform Crossover:\nChild 1:", child1_uniform, "\nChild 2:", child2_uniform)
print("Arithmetic Crossover:\nChild 1:", child1_arithmetic, "\nChild 2:", child2_arithmetic)
