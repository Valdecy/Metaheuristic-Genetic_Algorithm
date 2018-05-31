############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Genetic Algorithm

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Genetic_Algorithm, File: Python-MH-Genetic Algorithm.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Genetic_Algorithm>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import random
import os

# Function: Initialize Variables
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5]):
    population = pd.DataFrame(np.zeros((population_size, len(min_values))))
    population['Fitness'] = 0.0
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population.iloc[i,j] = random.uniform(min_values[j], max_values[j])
        population.iloc[i,-1] = target_function(population.iloc[i,0:population.shape[1]-1])
    return population

# Function: Fitness
def fitness_function(population): 
    fitness = pd.DataFrame(np.zeros((population.shape[0], 1)))
    fitness['Probability'] = 0.0
    for i in range(0, fitness.shape[0]):
        fitness.iloc[i,0] = 1/(1+ population.iloc[i,-1] + abs(population.iloc[:,-1].min()))
    fit_sum = fitness.iloc[:,0].sum()
    fitness.iloc[0,1] = fitness.iloc[0,0]
    for i in range(1, fitness.shape[0]):
        fitness.iloc[i,1] = (fitness.iloc[i,0] + fitness.iloc[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness.iloc[i,1] = fitness.iloc[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness.iloc[i, 1]):
          ix = i
          break
    return ix

# Function: Offspring
def breeding(population, fitness):
    offspring = population.copy(deep = True)

    for i in range (0, offspring.shape[0]):
        i1 = roulette_wheel(fitness)
        i2 = roulette_wheel(fitness)
        while i1 == i2:
            i2 = roulette_wheel(fitness)
        if (offspring.shape[1] - 1 > 1):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability >= 0.5):
                choose_mean = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                for j in range(0, offspring.shape[1] - 1):
                    if (choose_mean < 0.5):
                        offspring.iloc[i,j] = (population.iloc[i1, j] + population.iloc[i2, j])/2
                    else:
                        rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                        offspring.iloc[i,j] = rand*population.iloc[i1, j] + (1 - rand)*population.iloc[i2, j]
            else:
                choose_initial  = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                point_crossover = np.random.randint(offspring.shape[1] - 1, size = 1)[0]
                if (choose_initial < 0.5):
                    initial = i1
                    final   = i2
                else:
                    initial = i2
                    final   = i1
                if (point_crossover == 0):
                    offspring.iloc[i,0] = population.iloc[initial, 0]
                    for j in range(1, offspring.shape[1] - 1):
                        offspring.iloc[i,j] = population.iloc[final, j] 
                elif (point_crossover == population.shape[1] - 2):
                    offspring.iloc[i,-2] = population.iloc[initial, -2]
                    for j in range(offspring.shape[1] - 3, 0, -1):
                        offspring.iloc[i,j] = population.iloc[final, j]
                else:
                    for j in range(0, offspring.shape[1] - 1):
                        if(j < point_crossover):
                            offspring.iloc[i,j] = population.iloc[initial, j]
                        else:
                            offspring.iloc[i,j] = population.iloc[final, j]
        else:
            offspring.iloc[i,0] = (population.iloc[i1, 0] + population.iloc[i2, 0])/2
        offspring.iloc[i,-1] = target_function(offspring.iloc[i,0:offspring.shape[1]-1])
    return offspring

# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5]):
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - 1):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                choose_mutation = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                if (choose_mutation < 0.5):
                    choose_operation = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                    if(choose_operation < 0.5):
                        offspring.iloc[i,j] = offspring.iloc[i,j] + random.uniform(0, 1)*(max_values[j]- offspring.iloc[i,j])
                        if (offspring.iloc[i,j] > max_values[j]):
                            offspring.iloc[i,j] = max_values[j]
                        elif (offspring.iloc[i,j] < min_values[j]):
                            offspring.iloc[i,j] = min_values[j] 
                    else:
                        offspring.iloc[i,j] = offspring.iloc[i,j] - random.uniform(0, 1)*(offspring.iloc[i,j] - min_values[j])
                        if (offspring.iloc[i,j] > max_values[j]):
                            offspring.iloc[i,j] = max_values[j]
                        elif (offspring.iloc[i,j] < min_values[j]):
                            offspring.iloc[i,j] = min_values[j]
                else:
                    offspring.iloc[i,j] = random.uniform(min_values[j], max_values[j])
        offspring.iloc[i,-1] = target_function(offspring.iloc[i,0:offspring.shape[1]-1])
    return offspring

# GA Function
def genetic_algorithm(population_size = 5, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], generations = 50):    
    count = 0
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values)
    fitness = fitness_function(population)    
    elite = population.iloc[population['Fitness'].idxmin(),:].copy(deep = True)
    
    while (count <= generations):
        
        print("Iteration = ", count, " f(x) = ", elite[-1])
        
        offspring = breeding(population, fitness)
        population = mutation(offspring, mutation_rate = mutation_rate, min_values = min_values, max_values = max_values)
        fitness = fitness_function(population)
        if(elite[-1] > population.iloc[population['Fitness'].idxmin(),:][-1]):
            elite = population.iloc[population['Fitness'].idxmin(),:].copy(deep = True) 
        
        count = count + 1 
        
    print(elite)    
    return elite

######################## Part 1 - Usage ####################################

# Function to be Minimized. Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def target_function (variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

ga = genetic_algorithm(population_size = 500, mutation_rate = 0.1, min_values = [-5,-5], max_values = [5,5], generations = 100)
