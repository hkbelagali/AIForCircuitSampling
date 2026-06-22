"""Held-out NLL metrics. Uniform baseline = H_uniform(n_qubits) = n · log(2)."""
import numpy as np

from .entropy import H_uniform  # uniform_nll was a duplicate of this


def held_nll(log_q_held):
    return float(-np.asarray(log_q_held).mean())


def per_bit_nll(nll, n_qubits):
    return float(nll / n_qubits)


def normalized_nll(nll, n_qubits):
    """NLL per bit / log(2). Uniform = 1, lower = better."""
    return float(nll / (n_qubits * np.log(2)))


def nll_excess(nll, held_pC):
    """nll − H_estimate(held_pC). >= 0; 0 iff model matches p_C on held set."""
    H_est = float(-np.log(np.maximum(held_pC, 1e-30)).mean())
    return float(nll - H_est)
