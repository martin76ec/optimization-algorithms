import random
import matplotlib.pyplot as plt
import numpy as np

def init_population(population_size, chromosome_len):
    '''Initialize population as a list of random list of bits'''
    population = []
    for _ in range(population_size):
        cromosome = ''
        for _ in range(chromosome_len):
            bit = random.choice([0,1])
            cromosome += str(bit)
        population.append(cromosome)
    return population

def bin2float(num):
    '''convert a binary number to float.
    num: str'''
    if num[0] == '1':
        #negative number
        return -1 * int(num[1:],base=2)
    elif num[0] == '0':
        #positive number
        return int(num[1:], base=2)

def fitness(fun, population, chromosome_len):
    population_fitness = []
    for individual in population:
        x = individual[:chromosome_len//2]
        x = bin2float(x)
        y = individual[chromosome_len//2:]
        y = bin2float(y)

        individual_fitness = fun(x/100, y/100) #Considering 3 decimals
        population_fitness.append(individual_fitness)
    return population_fitness


def pick_elites(population, population_fitness, n_elites):
    s_pop = []
    for _, individual in sorted(zip(population_fitness, population)):
        s_pop.append(individual)
    return s_pop[:n_elites]

def mutation(individual, chromosome_len, mutation_prob):
    do_mutation = random.choices([1,0], weights=[mutation_prob, 1-mutation_prob])[0]
    if do_mutation:
        mutation_index = random.randint(0, chromosome_len-1)
        ls_individual = list(individual)
        if ls_individual[mutation_index] == '0':
            ls_individual[mutation_index] = '1'
            individual = ''.join(ls_individual)
        elif ls_individual[mutation_index] == '1':
            ls_individual[mutation_index] = '0'
            individual = ''.join(ls_individual)
    return individual

def crossover(elites, chromosome_len):
    father, mother = random.choices(elites, k=2)
    crossover_index = random.randint(1, chromosome_len-1)

    child0 = father[:crossover_index] + mother[crossover_index:]
    child1 = mother[:crossover_index] + father[crossover_index:]

    return [child0, child1]

def genetic_algorithm(fun, population_size, chromosome_len, max_epochs, n_elites, mut_prob):
    population = init_population(population_size, chromosome_len)
    best_ind, best_fit = None, float('inf')
    scores = []

    for epoch in range(max_epochs):
        population_fitness = fitness(fun, population, chromosome_len)

        if min(population_fitness) < best_fit:
            best_fit = min(population_fitness)
            best_ind = population[population_fitness.index(best_fit)]

        elites = pick_elites(population, population_fitness, n_elites)
        new_population = elites[:]
        while len(new_population) < population_size:
            childs = crossover(elites, chromosome_len)
            childs[0], childs[1] = mutation(childs[0], chromosome_len, mut_prob), mutation(childs[1],chromosome_len, mut_prob)
            new_population.extend(childs)
        population = new_population
        print('-----BEGIN-----',epoch, '\n', population, '\n', '-----END-----')

        print(f"Epoch {epoch}: best_fitness = {best_fit}")
        scores.append(best_fit)

    x = bin2float(best_ind[:chromosome_len//2])/100
    y = bin2float(best_ind[chromosome_len//2:])/100
    final_eval = fun(x,y)
    return x,y,final_eval,scores

    

# if __name__ == "__main__":
#     population_size = 100
#     chromosome_len = 20
#     max_epochs = 40
#     n_elites = 10
#     mut_prob = 0.2
#
#     f_sphere = lambda x,y: (x)**2 + (y)**2
#
#     x, y, final_eval,scores = genetic_algorithm(f_sphere, population_size, chromosome_len, max_epochs, n_elites, mut_prob)
#     print("\nBest aprox found:")
#     print(f"x = {x}, y = {y}")
#     print(f"Final evaluation: {final_eval}")
#
#     plt.plot(range(len(scores)), scores, label="Global bests through epochs", color='r')
#     plt.ylabel("Best Global Fitness")
#     plt.xlabel("Epochs")
#     plt.show()
