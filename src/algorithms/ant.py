import numpy as np


def calculate_probabilities(i, pheromone, dist_matrix, visited, alpha, beta):
    n_cities = len(dist_matrix)
    mask = np.ones(n_cities, dtype=bool)

    # Ensure indices are mapped correctly
    visited_indices = list(visited)
    mask[visited_indices] = False

    # Heuristic information (1/distance)
    eta = 1.0 / (dist_matrix[i] + 1e-10)

    # Calculate components
    phi = (pheromone[i] ** alpha) * (eta**beta)
    phi[~mask] = 0

    sum_phi = np.sum(phi)
    if sum_phi == 0:
        probs = np.where(mask, 1.0, 0)
    else:
        probs = phi / sum_phi

    # Normalization fix for np.random.choice precision requirements
    probs = probs / np.sum(probs)
    return probs


def construct_path(n_cities, pheromone, dist_matrix, alpha, beta):
    path = [np.random.randint(n_cities)]
    visited = {path[0]}

    while len(path) < n_cities:
        curr = path[-1]
        probs = calculate_probabilities(
            curr, pheromone, dist_matrix, visited, alpha, beta
        )
        next_city = np.random.choice(n_cities, p=probs)
        path.append(next_city)
        visited.add(next_city)

    dist = sum(dist_matrix[path[k]][path[(k + 1) % n_cities]] for k in range(n_cities))
    return path, dist


def aco_tsp(dist_matrix, n_ants, alpha, beta, rho, q, iterations):
    n_cities = len(dist_matrix)
    pheromone = np.ones((n_cities, n_cities)) * 0.1
    best_path, best_dist = None, float("inf")

    for _ in range(iterations):
        paths, dists = [], []

        for _ in range(n_ants):
            p, d = construct_path(n_cities, pheromone, dist_matrix, alpha, beta)
            paths.append(p)
            dists.append(d)
            if d < best_dist:
                best_dist, best_path = d, p

        pheromone *= 1 - rho
        for path, dist in zip(paths, dists):
            deposit = q / dist
            for k in range(n_cities):
                u, v = path[k], path[(k + 1) % n_cities]
                pheromone[u][v] += deposit
                pheromone[v][u] += deposit

    return best_path, best_dist
