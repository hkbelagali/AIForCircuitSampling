"""Linear cross-entropy benchmarking.

XEB(samples; p_C) = D · mean(p_C(z_i)) − 1
   uniform z   → 0
   z ~ p_C, p_C = Porter-Thomas → 1
"""
import numpy as np

from ..io.conventions import bits_to_int


def linear_xeb(samples, pC):
    """`samples` is either (k,) integer indices or (k, n) MSB-first bits."""
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = bits_to_int(samples)
    return float(len(pC) * pC[samples].mean() - 1)


def normalized_xeb(xeb_gen, xeb_uniform, xeb_held):
    """(xeb_gen − xeb_uniform) / (xeb_held − xeb_uniform). 0 = uniform, 1 = ceiling."""
    denom = xeb_held - xeb_uniform
    if abs(denom) < 1e-12:
        return float("nan")
    return float((xeb_gen - xeb_uniform) / denom)
