"""Entropy estimators. KL / TV require the full (2^n,) distribution."""
import numpy as np


def H_pC_estimate(pC_on_samples):
    """Monte Carlo H(p_C) from z ~ p_C: -<log p_C(z)>. Returns nats."""
    return float(-np.log(np.maximum(np.asarray(pC_on_samples), 1e-30)).mean())


def H_uniform(n):
    return float(n * np.log(2))


def KL_to_uniform(pC):
    n = int(np.log2(len(pC)))
    uniform = 1.0 / (1 << n)
    nz = pC > 0
    return float((pC[nz] * np.log(pC[nz] / uniform)).sum())


def TV_to_uniform(pC):
    n = int(np.log2(len(pC)))
    return float(0.5 * np.abs(pC - 1.0 / (1 << n)).sum())
