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
    population = np.random.randint(2, size=(population_size, chromosome_len))

    n_bits_per_var = chromosome_len // 2

    best_eval = float("inf")
    best_sol_coords = None

    for epoch in range(max_epochs):
        decoded_coords = []
        costs = []

        for individual in population:
            bit_x = individual[:n_bits_per_var]
            bit_y = individual[n_bits_per_var:]

            coords = []
            for i, bits in enumerate([bit_x, bit_y]):
                low, high = bounds[i]
                int_val = int("".join(map(str, bits)), 2)
                precision = (2**n_bits_per_var) - 1
                real_val = low + (int_val / precision) * (high - low)
                coords.append(real_val)

            cost = fitness_func(coords)
            decoded_coords.append(coords)
            costs.append(cost)

        min_cost_idx = np.argmin(costs)
        if costs[min_cost_idx] < best_eval:
            best_eval = costs[min_cost_idx]
            best_sol_coords = decoded_coords[min_cost_idx]

        selected_indices = []
        for _ in range(population_size - n_elites):
            i1, i2 = np.random.randint(0, population_size, 2)
            selected_indices.append(i1 if costs[i1] < costs[i2] else i2)

        new_population = population[selected_indices]

        for i in range(0, len(new_population) - 1, 2):
            if np.random.rand() < 0.8:
                cp = np.random.randint(1, chromosome_len - 1)
                parent1, parent2 = (
                    new_population[i].copy(),
                    new_population[i + 1].copy(),
                )
                new_population[i] = np.concatenate((parent1[:cp], parent2[cp:]))
                new_population[i + 1] = np.concatenate((parent2[:cp], parent1[cp:]))

        mask = np.random.rand(*new_population.shape) < mut_prob
        new_population ^= mask

        elites_indices = np.argsort(costs)[:n_elites]
        elites = population[elites_indices]
        population = np.vstack((new_population, elites))

    return best_sol_coords, best_eval, None
