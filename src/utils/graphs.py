import matplotlib.pyplot as plt
import numpy as np


def generate_nodes(n, mode="random", separation=10):
    if mode == "random":
        return np.random.rand(n, 2) * 100
    else:
        side = int(np.sqrt(n))
        coords = np.array(
            [[i * separation, j * separation] for i in range(side) for j in range(side)]
        )
        return coords


def matrix_build(coords):
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return matrix


def graph_build(n, mode="random"):
    positions = generate_nodes(n, mode)
    graph = matrix_build(positions)

    return {"positions": positions, "graph": graph}


def solution_plot(ax, coords, tour, title):
    path_coords = coords[tour]
    path_coords = np.vstack([path_coords, path_coords[0]])
    ax.plot(
        path_coords[:, 0],
        path_coords[:, 1],
        "o-",
        markersize=4,
        linewidth=1,
        color="#e74c3c",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def inspect_node_separation(graph, title="Grid Separation Check"):
    coords = np.array(graph["positions"])

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c="blue", s=30, label="Nodes")

    for i in range(min(len(coords), 5)):
        plt.text(
            coords[i, 0] + 1,
            coords[i, 1] + 1,
            f"({coords[i,0]:.0f},{coords[i,1]:.0f})",
            fontsize=9,
        )

    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 101, 10))

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.title(title)
    plt.xlabel("X Units")
    plt.ylabel("Y Units")
    plt.axis("equal")
    plt.show()
