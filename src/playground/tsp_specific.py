import numpy as np
from algorithms.ant import aco_tsp


def traveling_problem():
    distances = np.array(
        [
            [0, 29, 20, 21, 16, 31, 100, 12, 4, 31],
            [29, 0, 15, 29, 28, 40, 72, 21, 29, 41],
            [20, 15, 0, 15, 14, 25, 81, 9, 23, 27],
            [21, 29, 15, 0, 4, 12, 92, 12, 25, 13],
            [16, 28, 14, 4, 0, 16, 94, 9, 20, 16],
            [31, 40, 25, 12, 16, 0, 98, 24, 36, 3],
            [100, 72, 81, 92, 94, 98, 0, 90, 101, 99],
            [12, 21, 9, 12, 9, 24, 90, 0, 15, 25],
            [4, 29, 23, 25, 20, 36, 101, 15, 0, 35],
            [31, 41, 27, 13, 16, 3, 99, 25, 35, 0],
        ]
    )

    num_cities = len(distances)

    def ga_tsp_dist_matrix(
        dist_matrix, pop_size=100, max_epochs=500, n_elites=2, mut_prob=0.1
    ):
        import numpy as np

        def get_cost(tour):
            return sum(
                dist_matrix[tour[i]][tour[(i + 1) % num_cities]]
                for i in range(num_cities)
            )

        from algorithms.ga_tsp import (
            tournament_selection,
            ordered_crossover,
            swap_mutation,
        )

        population = np.array(
            [np.random.permutation(num_cities) for _ in range(pop_size)]
        )
        best_eval = float("inf")
        best_tour = None

        for _ in range(max_epochs):
            costs = np.array([get_cost(p) for p in population])

            min_idx = np.argmin(costs)
            if costs[min_idx] < best_eval:
                best_eval, best_tour = costs[min_idx], population[min_idx].copy()

            n_to_select = pop_size - n_elites
            mating_pool = tournament_selection(population, costs, n_to_select)
            new_gen = ordered_crossover(mating_pool)
            new_gen = swap_mutation(new_gen, mut_prob)

            elites = population[np.argsort(costs)[:n_elites]]
            population = np.vstack((new_gen, elites))

        return best_tour, best_eval

    ga_tour, ga_dist = ga_tsp_dist_matrix(distances)

    aco_params = {
        "n_ants": 20,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.5,
        "q": 100,
        "iterations": 100,
    }
    aco_tour, aco_dist = aco_tsp(distances, **aco_params)

    print("--- Traveling Problem (custom) Results ---")
    print(f"Genetic Algorithm: Best Distance = {ga_dist}, Path = {ga_tour}")
    print(f"Ant Colony Optimization: Best Distance = {aco_dist}, Path = {aco_tour}")
    print(f"Overall Minimum Distance: {min(ga_dist, aco_dist)}")
    print(f"Overall Best Path: {ga_tour if ga_dist < aco_dist else aco_tour}")
