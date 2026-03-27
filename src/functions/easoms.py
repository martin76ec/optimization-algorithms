import numpy as np

def easom(x):
    """
    Global minimum: f(pi, pi) = -1
    Typically evaluated on x_i in [-100, 100]
    """
    x1, x2 = x[0], x[1]
    cost = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi)**2 - (x2 - np.pi)**2)
    return cost
