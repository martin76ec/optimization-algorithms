from algorithms.ga_tsp import genetic_algorithm_tsp
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


def fitness_build(coords, graph):
    node_positions = graph["positions"]
    distances = [np.linalg.norm(np.array(coords) - pos) for pos in node_positions]
    return sum(distances)


def traveling_salesman():
    graph_rand_1 = graph_build(n=100, mode="random")
    graph_rand_2 = graph_build(n=100, mode="random")
    graph_rand_3 = graph_build(n=100, mode="random")

    graph_1 = graph_build(n=25, mode="grid")
    graph_2 = graph_build(n=100, mode="grid")
    graph_3 = graph_build(n=225, mode="grid")

    # inspect_node_separation(graph_rand_1)
    # inspect_node_separation(graph_1)

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

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))

    axes = axes.flatten()
    axes2 = axes2.flatten()

    aco_params = {
        "n_ants": 20,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.5,
        "q": 100,
        "iterations": 100,
    }

    for i, (name, g) in enumerate(graphs):
        coords = g["positions"]

        best_tour, best_score = genetic_algorithm_tsp(
            nodes=coords,
            population_size=pop_size,
            max_epochs=epochs,
            n_elites=elites,
            mut_prob=mutation,
        )

        dist_matrix = calculate_distance_matrix(coords)
        best_tour_aco, score_aco = aco_tsp(dist_matrix=dist_matrix, **aco_params)

        solution_plot(axes[i], coords, best_tour, f"{name}\nDist: {best_score:.2f}")
        solution_plot(
            axes2[i], coords, best_tour_aco, f"ACO: {name}\nDist: {score_aco:.2f}"
        )

    plt.tight_layout()
    fig.savefig("./figures/fig1.png")
    fig2.savefig("figures/fig2.png")
