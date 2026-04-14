import numpy as np


def genetic_algorithm(
    fitness_func,
    population_size,
    chromosome_len,
    max_epochs,
    n_elites,
    mut_prob,
    bounds,
):
    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(population_size, chromosome_len),
    )

    best_eval = float("inf")
    best_sol_coords = None

    for _ in range(max_epochs):
        costs = []

        for individual in population:
            cost = fitness_func(individual)
            costs.append(cost)

        costs = np.array(costs)
        min_cost_idx = np.argmin(costs)
        if costs[min_cost_idx] < best_eval:
            best_eval = costs[min_cost_idx]
            best_sol_coords = population[min_cost_idx].copy()

        selected_indices = []
        for _ in range(population_size - n_elites):
            i1, i2 = np.random.randint(0, population_size, 2)
            selected_indices.append(i1 if costs[i1] < costs[i2] else i2)

        new_population = population[selected_indices].copy()

        for i in range(0, len(new_population) - 1, 2):
            if np.random.rand() < 0.8:
                cp = np.random.randint(1, chromosome_len)
                p1, p2 = new_population[i].copy(), new_population[i + 1].copy()
                new_population[i] = np.concatenate((p1[:cp], p2[cp:]))
                new_population[i + 1] = np.concatenate((p2[:cp], p1[cp:]))

        mask = np.random.rand(*new_population.shape) < mut_prob
        noise = np.random.uniform(-1, 1, size=new_population.shape) * 0.1
        new_population[mask] += noise[mask]

        for i in range(chromosome_len):
            new_population[:, i] = np.clip(
                new_population[:, i], bounds[i][0], bounds[i][1]
            )

        elites_indices = np.argsort(costs)[:n_elites]
        elites = population[elites_indices]
        population = np.vstack((new_population, elites))

    return best_sol_coords, best_eval, None
