import numpy as np


def tournament_selection(population, costs, n_selections):
    selected_indices = []
    pop_size = len(population)
    for _ in range(n_selections):
        i1, i2 = np.random.randint(0, pop_size, 2)
        selected_indices.append(i1 if costs[i1] < costs[i2] else i2)
    return population[selected_indices]


def ordered_crossover(parents, rate=0.8):
    offspring = parents.copy()
    chrom_len = parents.shape[1]
    for i in range(0, len(offspring) - 1, 2):
        if np.random.rand() < rate:
            p1, p2 = offspring[i].copy(), offspring[i + 1].copy()

            def fill_offspring(parent_main, parent_donor):
                start, end = sorted(
                    np.random.choice(range(chrom_len), 2, replace=False)
                )
                child = np.full(chrom_len, -1)
                child[start:end] = parent_main[start:end]

                # Fill remaining slots with genes from donor in order they appear
                remaining = [gene for gene in parent_donor if gene not in child]
                child[child == -1] = remaining
                return child

            offspring[i] = fill_offspring(p1, p2)
            offspring[i + 1] = fill_offspring(p2, p1)
    return offspring


def swap_mutation(population, mut_prob):
    for i in range(len(population)):
        if np.random.rand() < mut_prob:
            idx1, idx2 = np.random.randint(0, population.shape[1], 2)
            population[i, [idx1, idx2]] = population[i, [idx2, idx1]]
    return population


def genetic_algorithm_tsp(
    nodes,
    population_size,
    max_epochs,
    n_elites,
    mut_prob,
    crossover_func=ordered_crossover,
):
    num_cities = len(nodes)
    population = np.array(
        [np.random.permutation(num_cities) for _ in range(population_size)]
    )

    best_eval = float("inf")
    best_tour = None

    for epoch in range(max_epochs):
        costs = np.array(
            [
                np.sum(np.linalg.norm(nodes[p] - nodes[np.roll(p, -1)], axis=1))
                for p in population
            ]
        )

        min_idx = np.argmin(costs)
        if costs[min_idx] < best_eval:
            best_eval, best_tour = costs[min_idx], population[min_idx].copy()

        n_to_select = population_size - n_elites
        mating_pool = tournament_selection(population, costs, n_to_select)
        new_gen = crossover_func(mating_pool)
        new_gen = swap_mutation(new_gen, mut_prob)

        elites = population[np.argsort(costs)[:n_elites]]
        population = np.vstack((new_gen, elites))

    return best_tour, best_eval
