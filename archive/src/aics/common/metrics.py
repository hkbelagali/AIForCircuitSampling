"""Generic information-theoretic and concentration diagnostics on probability
distributions over computational basis states.
"""

import numpy as np


def shannon_entropy(probs, tol=1e-14):
    """H = -sum_x p_x log p_x in nats. Zeros (within tol) drop out."""
    p = np.asarray(probs, dtype=float)
    p = p[p > tol]
    return float(-np.sum(p * np.log(p)))


def participation_ratio(probs):
    """PR = 1 / sum_x p_x^2. Effective number of supported states."""
    p = np.asarray(probs, dtype=float)
    return float(1.0 / np.sum(p ** 2))


def effective_dim(probs, tol=1e-14):
    """exp(H). Smooth proxy for support size; equals support count when uniform."""
    return float(np.exp(shannon_entropy(probs, tol=tol)))
