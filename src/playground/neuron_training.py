import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithms.ga import genetic_algorithm


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_z(X, weights):
    w = weights
    X0, X1, X2 = X
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


def train_with_ga(X_data, y_data, pop_size=100, epochs=300):
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


def load_data():
    X_df = pd.read_csv("data/Xnonlinear.csv", index_col=0)
    y_df = pd.read_csv("data/ynonlinear.csv", index_col=0)

    X1 = X_df.iloc[:, 0].values
    X2 = X_df.iloc[:, 1].values
    y_raw = y_df.iloc[:, 0].values

    y = np.where(y_raw == -1, 0.0, 1.0)

    X_data = np.column_stack([np.ones(len(X1)), X1, X2])
    return X_data, y, X1, X2, y_raw


def plot_decision_boundary(ax, weights, X1, X2, y_raw, title):
    h = 0.05
    x_min, x_max = X1.min() - 0.5, X1.max() + 0.5
    y_min, y_max = X2.min() - 0.5, X2.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            X = [1, xx[i, j], yy[i, j]]
            z = compute_z(X, weights)
            Z[i, j] = sigmoid(z)

    ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 25), cmap="RdBu", alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=1.5)

    mask_pos = y_raw == 1
    mask_neg = y_raw == -1
    ax.scatter(
        X1[mask_pos],
        X2[mask_pos],
        c="blue",
        s=15,
        label="Class 1",
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        X1[mask_neg],
        X2[mask_neg],
        c="red",
        s=15,
        label="Class -1",
        edgecolors="k",
        linewidths=0.3,
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend(fontsize=8)


def neuron_party():
    X_data, y_data, X1, X2, y_raw = load_data()

    print("Training with Gradient Descent...")
    gd_weights = gradient_descent(X_data, y_data, learning_rate=0.1, epochs=2000)
    gd_loss = total_loss(X_data, y_data, gd_weights)
    print(f"GD Weights: {gd_weights}")
    print(f"GD Loss: {gd_loss:.4f}")

    print("\nTraining with Genetic Algorithm...")
    ga_weights, ga_loss = train_with_ga(X_data, y_data)
    print(f"GA Weights: {ga_weights}")
    print(f"GA Loss: {ga_loss:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_decision_boundary(
        ax1, gd_weights, X1, X2, y_raw, f"Gradient Descent\nLoss: {gd_loss:.4f}"
    )
    plot_decision_boundary(
        ax2, ga_weights, X1, X2, y_raw, f"Genetic Algorithm\nLoss: {ga_loss:.4f}"
    )

    fig.suptitle("Neuron Training — Decision Boundary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("figures/fig6_neuron.png")
