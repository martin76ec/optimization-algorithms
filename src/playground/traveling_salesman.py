from algorithms.ga_tsp import genetic_algorithm_tsp, ordered_crossover, pmx_crossover
from algorithms.ant import aco_tsp
from utils.graphs import graph_build, solution_plot

import matplotlib.pyplot as plt
import numpy as np


def calculate_distance_matrix(coords):
    return np.array(
        [
            [np.linalg.norm(np.array(c1) - np.array(c2)) for c2 in coords]
            for c1 in coords
        ]
    )


def traveling_salesman():
    graph_rand_1 = graph_build(n=100, mode="random")
    graph_rand_2 = graph_build(n=100, mode="random")
    graph_rand_3 = graph_build(n=100, mode="random")

    graph_1 = graph_build(n=25, mode="grid")
    graph_2 = graph_build(n=100, mode="grid")
    graph_3 = graph_build(n=225, mode="grid")

    pop_size = 100
    epochs = 300
    elites = 2
    mutation = 0.05

    graphs = [
        ("Random 1", graph_rand_1),
        ("Random 2", graph_rand_2),
        ("Random 3", graph_rand_3),
        ("Grid 25", graph_1),
        ("Grid 100", graph_2),
        ("Grid 225", graph_3),
    ]

    aco_params = {
        "n_ants": 20,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.5,
        "q": 100,
        "iterations": 100,
    }

    fig_ox, axes_ox = plt.subplots(2, 3, figsize=(15, 10))
    fig_pmx, axes_pmx = plt.subplots(2, 3, figsize=(15, 10))
    fig_aco_basic, axes_aco_basic = plt.subplots(2, 3, figsize=(15, 10))
    fig_aco_enhanced, axes_aco_enhanced = plt.subplots(2, 3, figsize=(15, 10))

    axes_ox = axes_ox.flatten()
    axes_pmx = axes_pmx.flatten()
    axes_aco_basic = axes_aco_basic.flatten()
    axes_aco_enhanced = axes_aco_enhanced.flatten()

    for i, (name, g) in enumerate(graphs):
        coords = g["positions"]
        dist_matrix = calculate_distance_matrix(coords)

        tour_ox, dist_ox = genetic_algorithm_tsp(
            nodes=coords,
            population_size=pop_size,
            max_epochs=epochs,
            n_elites=elites,
            mut_prob=mutation,
            crossover_func=ordered_crossover,
        )

        tour_pmx, dist_pmx = genetic_algorithm_tsp(
            nodes=coords,
            population_size=pop_size,
            max_epochs=epochs,
            n_elites=elites,
            mut_prob=mutation,
            crossover_func=pmx_crossover,
        )

        tour_aco_basic, dist_aco_basic = aco_tsp(dist_matrix=dist_matrix, **aco_params)

        tour_aco_enhanced, dist_aco_enhanced = aco_tsp(
            dist_matrix=dist_matrix,
            **aco_params,
            tau_min=0.01,
            tau_max=10.0,
            elitism=True,
        )

        solution_plot(
            axes_ox[i], coords, tour_ox, f"GA-OX: {name}\nDist: {dist_ox:.2f}"
        )
        solution_plot(
            axes_pmx[i], coords, tour_pmx, f"GA-PMX: {name}\nDist: {dist_pmx:.2f}"
        )
        solution_plot(
            axes_aco_basic[i],
            coords,
            tour_aco_basic,
            f"ACO Basic: {name}\nDist: {dist_aco_basic:.2f}",
        )
        solution_plot(
            axes_aco_enhanced[i],
            coords,
            tour_aco_enhanced,
            f"ACO Enhanced: {name}\nDist: {dist_aco_enhanced:.2f}",
        )

        print(f"{name}:")
        print(f"  GA-OX:          {dist_ox:.2f}")
        print(f"  GA-PMX:         {dist_pmx:.2f}")
        print(f"  ACO Basic:      {dist_aco_basic:.2f}")
        print(f"  ACO Enhanced:   {dist_aco_enhanced:.2f}")
        print()

    for fig, fig_obj in [
        ("fig1", fig_ox),
        ("fig2", fig_pmx),
        ("fig3", fig_aco_basic),
        ("fig4", fig_aco_enhanced),
    ]:
        fig_obj.tight_layout()

    fig_ox.savefig("figures/fig1_ga_ox.png")
    fig_pmx.savefig("figures/fig2_ga_pmx.png")
    fig_aco_basic.savefig("figures/fig3_aco_basic.png")
    fig_aco_enhanced.savefig("figures/fig4_aco_enhanced.png")
