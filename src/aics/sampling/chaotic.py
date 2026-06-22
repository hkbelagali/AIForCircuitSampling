"""sample_chaotic — BIASED on non-PT distributions. Use sample_exact_tn instead.

Wraps quimb.Circuit.sample_chaotic, which puts a uniform prior on every
qubit outside `marginal_qubits` (literally rng.choice("0","1")). Kept as
the v1 baseline for chaotic-vs-unbiased comparisons.
"""
import numpy as np

from ._quimb_circuit import _resolve_qcirc


def sample_chaotic(circ_or_qcirc, qubits=None, *, k_samples,
                    marginal_qubits=None, seed=None,
                    optimize="auto-hq", dtype="complex64", to_backend=None):
    """Returns (k_samples, n) uint8, MSB-first."""
    qcirc = _resolve_qcirc(circ_or_qcirc, qubits,
                            dtype=dtype, to_backend=to_backend)
    n = qcirc.N
    if marginal_qubits is None:
        marginal_qubits = min(20, n)
    out = np.empty((k_samples, n), dtype=np.uint8)
    gen = qcirc.sample_chaotic(
        C=k_samples, marginal_qubits=marginal_qubits,
        seed=seed, optimize=optimize, dtype=dtype,
    )
    for i, bitstring in enumerate(gen):
        out[i] = np.frombuffer(bitstring.encode("ascii"),
                                dtype=np.uint8) - ord("0")
    return out
