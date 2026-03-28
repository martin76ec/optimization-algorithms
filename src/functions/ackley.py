import math


def ackley(x, a=20, b=0.2, c=2 * math.pi):
    """
    Returns a standard Python float.
    x: list of numbers (coordinates)
    """
    d = len(x)

    # Standard Python sum and math.sqrt replace np.sum and np.sqrt
    sum_sq = sum(xi**2 for xi in x)
    sum_sq_term = -a * math.exp(-b * math.sqrt(sum_sq / d))

    # Standard math.cos and math.exp replace np.cos and np.exp
    sum_cos = sum(math.cos(c * xi) for xi in x)
    cos_term = -math.exp(sum_cos / d)

    # Result is a native float
    return sum_sq_term + cos_term + a + math.exp(1)
