import numpy as np
import matplotlib.pyplot as plt
from algorithms.ant import aco_tsp
from algorithms.ga_tsp import tournament_selection, ordered_crossover, swap_mutation


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

    city_coords = np.array(
        [
            [0, 0],
            [29, 0],
            [20, 15],
            [21, 29],
            [16, 4],
            [31, 12],
            [100, 50],
            [12, 9],
            [4, 23],
            [31, 3],
        ]
    )

    def ga_tsp_dist_matrix(
        dist_matrix, pop_size=100, max_epochs=500, n_elites=2, mut_prob=0.1
    ):
        population = np.array(
            [np.random.permutation(num_cities) for _ in range(pop_size)]
        )
        best_eval = float("inf")
        best_tour = None

        for _ in range(max_epochs):
            costs = np.array(
                [
                    sum(
                        dist_matrix[p[i]][p[(i + 1) % num_cities]]
                        for i in range(num_cities)
                    )
                    for p in population
                ]
            )

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

    print("##### TSP 10 cities ####")
    print(f"GA: Best Distance = {ga_dist}, Path = {list(ga_tour)}")
    print(f"ACO: Best Distance = {aco_dist}, Path = {aco_tour}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, tour, _, title in [
        (ax1, ga_tour, ga_dist, f"GA\nDist: {ga_dist:.2f}"),
        (ax2, aco_tour, aco_dist, f"ACO\nDist: {aco_dist:.2f}"),
    ]:
        path_coords = city_coords[tour]
        path_coords = np.vstack([path_coords, path_coords[0]])
        ax.plot(
            path_coords[:, 0],
            path_coords[:, 1],
            "o-",
            markersize=6,
            linewidth=1.5,
            color="#e74c3c",
        )
        for idx, (x, y) in enumerate(city_coords):
            ax.annotate(
                str(idx), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8
            )
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("TSP — 10-City Instance", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("figures/fig5_tsp_specific.png")
