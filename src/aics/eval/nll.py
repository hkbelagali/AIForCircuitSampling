"""Held-out NLL metrics. Uniform baseline = n · log(2)."""
import numpy as np


def held_nll(log_q_held):
    return float(-np.asarray(log_q_held).mean())


def per_bit_nll(nll, n):
    return float(nll / n)


def normalised_nll(nll, n):
    """NLL per bit / log(2). Uniform = 1, lower = better."""
    return float(nll / (n * np.log(2)))


def nll_excess(nll, held_pC):
    """nll − H_estimate(held_pC). >= 0; 0 iff model matches p_C on held set."""
    H_est = float(-np.log(np.maximum(held_pC, 1e-30)).mean())
    return float(nll - H_est)


def uniform_nll(n):
    return float(n * np.log(2))
