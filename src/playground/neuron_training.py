import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_z(X, weights):
    X0, X1, X2 = X
    w = weights
    return (
        w[0] * X0
        + w[1] * X1
        + w[2] * X2
        + w[3] * (X1**2)
        + w[4] * (X2**2)
        + w[5] * (X1 * X2)
    )


def loss_function(y_true, y_pred):
    return 0.5 * (y_pred - y_true) ** 2


def total_loss(X_data, y_data, weights):
    loss = 0
    for X, y in zip(X_data, y_data):
        z = compute_z(X, weights)
        y_hat = sigmoid(z)
        loss += loss_function(y, y_hat)
    return loss / len(X_data)


def gradient_descent(X_data, y_data, learning_rate=0.01, epochs=1000):
    weights = np.random.randn(6)

    for _ in range(epochs):
        grad = np.zeros(6)
        for X, y in zip(X_data, y_data):
            z = compute_z(X, weights)
            y_hat = sigmoid(z)

            common = (y_hat - y) * y_hat * (1 - y_hat)

            dz_dw = np.array([X[0], X[1], X[2], X[1] ** 2, X[2] ** 2, X[1] * X[2]])
            grad += common * dz_dw

        weights -= (learning_rate / len(X_data)) * grad

    return weights


def train_with_ga(X_data, y_data, pop_size=50, epochs=100):
    def ga_objective(weights):
        return total_loss(X_data, y_data, weights)

    from algorithms.ga import genetic_algorithm

    def loss_wrapper(chromosome):
        return total_loss(X_data, y_data, chromosome)

    bounds = [(-10, 10)] * 6

    best_weights, best_loss, _ = genetic_algorithm(
        loss_wrapper,
        population_size=pop_size,
        chromosome_len=6,
        max_epochs=epochs,
        n_elites=2,
        mut_prob=0.1,
        bounds=bounds,
    )
    return best_weights, best_loss


def generate_synthetic_data(n_samples=100):
    np.random.seed(42)
    X1 = np.random.uniform(-1, 1, n_samples)
    X2 = np.random.uniform(-1, 1, n_samples)

    true_w = np.array([0.5, 1.2, -0.8, 0.4, -0.3, 0.7])

    X_data = []
    y_data = []

    for x1, x2 in zip(X1, X2):
        X = [1, x1, x2]
        z = compute_z(X, true_w)
        y = 1 if sigmoid(z) > 0.5 else 0
        X_data.append(X)
        y_data.append(y)

    return np.array(X_data), np.array(y_data)


def neuron_party():
    X_data, y_data = generate_synthetic_data()

    print("Training with Gradient Descent...")
    gd_weights = gradient_descent(X_data, y_data)
    gd_loss = total_loss(X_data, y_data, gd_weights)
    print(f"GD Weights: {gd_weights}")
    print(f"GD Loss: {gd_loss:.4f}")

    print("\nTraining with Genetic Algorithm...")
    ga_weights, ga_loss = train_with_ga(X_data, y_data)
    print(f"GA Weights: {ga_weights}")
    print(f"GA Loss: {ga_loss:.4f}")
