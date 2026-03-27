import numpy as np

def ackley(x, a=20, b=0.2, c=2*np.pi):
    """
    Global minimum: f(0, ..., 0) = 0
    Typically evaluated on the hypercube x_i in [-32.768, 32.768]
    """
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum_sq_term + cos_term + a + np.exp(1)
