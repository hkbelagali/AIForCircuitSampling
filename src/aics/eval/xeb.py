"""Linear cross-entropy benchmarking.

XEB(samples; p_C) = D · mean(p_C(z_i)) − 1
   uniform z   → 0
   z ~ p_C, p_C = Porter-Thomas → 1
"""
import numpy as np

from ..io.conventions import bits_to_int


def linear_xeb(samples_idx, pC):
    return float(len(pC) * pC[samples_idx].mean() - 1)


def linear_xeb_from_bits(samples_bits, pC):
    return linear_xeb(bits_to_int(samples_bits), pC)


def normalized_xeb(xeb_gen, xeb_uniform, xeb_held):
    """(xeb_gen − xeb_uniform) / (xeb_held − xeb_uniform). 0 = uniform, 1 = ceiling."""
    denom = xeb_held - xeb_uniform
    if abs(denom) < 1e-12:
        return float("nan")
    return float((xeb_gen - xeb_uniform) / denom)
