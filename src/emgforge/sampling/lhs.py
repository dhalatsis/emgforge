import numpy as np


def latin_hypercube(n: int, d: int, seed: int = 0) -> np.ndarray:
    """
    Returns an (n, d) array in [0,1] using a simple Latin Hypercube scheme.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        x[:, j] = (perm + rng.random(n)) / n
    return x


__all__ = ["latin_hypercube"]

