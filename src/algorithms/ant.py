import numpy as np


def calculate_probabilities(i, pheromone, dist_matrix, visited, alpha, beta):
    n_cities = len(dist_matrix)
    probs = []
    for j in range(n_cities):
        if j not in visited:
            p = (pheromone[i][j] ** alpha) * ((1.0 / dist_matrix[i][j]) ** beta)
            probs.append(p)
        else:
            probs.append(0)
    return np.array(probs) / sum(probs)


def construct_path(n_cities, pheromone, dist_matrix, alpha, beta):
    path = [np.random.randint(n_cities)]
    visited = {path[0]}

    while len(path) < n_cities:
        i = path[-1]
        probs = calculate_probabilities(i, pheromone, dist_matrix, visited, alpha, beta)
        next_city = np.random.choice(range(n_cities), p=probs)
        path.append(next_city)
        visited.add(next_city)

    dist = sum(dist_matrix[path[k]][path[k + 1]] for k in range(n_cities - 1))
    dist += dist_matrix[path[-1]][path[0]]
    return path, dist


def aco_tsp(dist_matrix, n_ants, alpha, beta, rho, q, iterations):
    n_cities = len(dist_matrix)
    pheromone = np.ones((n_cities, n_cities)) * 0.1
    best_path, best_dist = None, float("inf")

    for _ in range(iterations):
        all_paths, all_dists = [], []

        for _ in range(n_ants):
            path, dist = construct_path(n_cities, pheromone, dist_matrix, alpha, beta)
            all_paths.append(path)
            all_dists.append(dist)

            if dist < best_dist:
                best_dist, best_path = dist, path

        pheromone *= 1 - rho
        for path, dist in zip(all_paths, all_dists):
            for k in range(n_cities - 1):
                pheromone[path[k]][path[k + 1]] += q / dist
            pheromone[path[-1]][path[0]] += q / dist

    return best_path, best_dist
