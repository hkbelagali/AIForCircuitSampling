"""Linear cross-entropy benchmarking (XEB).

XEB(samples; p_C) := D · E_{z ~ samples}[p_C(z)] − 1
                   = D · mean(p_C(z_i)) − 1

  Uniform z:         XEB → 0
  Porter-Thomas z:   XEB → 1 (the typical RCS target)
  Peaked / under-scrambled:  XEB can exceed 1

Use the bit-array variants for sample arrays in MSB-first qubit order.
"""
import numpy as np

from ..io.conventions import bits_to_int


def linear_xeb(samples_idx, pC):
    """XEB from integer sample indices. samples_idx: (k,) ints; pC: (2^n,)."""
    return float(len(pC) * pC[samples_idx].mean() - 1)


def linear_xeb_from_bits(samples_bits, pC):
    """XEB from a (k, n) MSB-first bit array."""
    return linear_xeb(bits_to_int(samples_bits), pC)


def normalized_xeb(xeb_gen, xeb_uniform, xeb_held):
    """Normalised XEB:  (xeb_gen − xeb_uniform) / (xeb_held − xeb_uniform).

      0  = uniform-noise baseline
      1  = data ceiling (samples from p_C scored against p_C)

    The interesting axis for downstream plots.
    """
    denom = xeb_held - xeb_uniform
    if abs(denom) < 1e-12:
        return float("nan")
    return float((xeb_gen - xeb_uniform) / denom)
