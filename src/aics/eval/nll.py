"""Held-out NLL and derived metrics.

  held_nll       := - mean_{z ~ held} log q(z)             # raw NLL
  per_bit        := held_nll / n
  per_bit_normed := per_bit / log(2)                       # uniform = 1, lower = better
  excess         := held_nll - H_estimate(held_pC)         # >= 0; 0 = ideal
  uniform_nll    := n · log(2)                             # the trivial baseline
"""
import numpy as np


def held_nll(log_q_held):
    """- mean log q(z) for held-out z. `log_q_held` is (k,) numpy/torch."""
    arr = np.asarray(log_q_held)
    return float(-arr.mean())


def per_bit_nll(nll, n):
    """NLL per bit; uniform = log(2)."""
    return float(nll / n)


def normalised_nll(nll, n):
    """NLL per bit normalised so uniform = 1.0. Lower = better."""
    return float(nll / (n * np.log(2)))


def nll_excess(nll, held_pC):
    """NLL excess over the entropy of p_C (estimated from held p_C samples).

    >= 0 always; 0 iff the model exactly matches p_C on the held set.
    """
    H_est = float(-np.log(np.maximum(held_pC, 1e-30)).mean())
    return float(nll - H_est)


def uniform_nll(n):
    """The trivial baseline: NLL of the uniform distribution."""
    return float(n * np.log(2))
