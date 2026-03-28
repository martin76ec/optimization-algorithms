import math


def round_sig(x, sig=4):
    """Rounds a number to a specific number of significant digits."""
    if x == 0:
        return 0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)
