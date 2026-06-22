"""Entropy estimators.

H_pC_estimate(pC_on_held_samples) is a Monte Carlo estimator of H(p_C)
from samples z ~ p_C: H ≈ - mean log p_C(z). Unbiased for samples drawn
from p_C, biased otherwise.

H_uniform(n) is the trivial baseline n · log 2.

KL_to_uniform(pC) and TV_to_uniform(pC) take the full distribution; only
feasible at small n.
"""
import numpy as np


def H_pC_estimate(pC_on_samples):
    """Monte Carlo estimate of H(p_C) from samples z ~ p_C: -<log p_C(z)>.

    Returns nats.
    """
    arr = np.asarray(pC_on_samples)
    return float(-np.log(np.maximum(arr, 1e-30)).mean())


def H_uniform(n):
    return float(n * np.log(2))


def KL_to_uniform(pC):
    """KL(p_C || uniform). Requires the full (2^n,) distribution."""
    n = int(np.log2(len(pC)))
    uniform = 1.0 / (1 << n)
    nz = pC > 0
    return float((pC[nz] * np.log(pC[nz] / uniform)).sum())


def TV_to_uniform(pC):
    """Total variation distance to uniform. Requires the full (2^n,) distribution."""
    n = int(np.log2(len(pC)))
    uniform = 1.0 / (1 << n)
    return float(0.5 * np.abs(pC - uniform).sum())
