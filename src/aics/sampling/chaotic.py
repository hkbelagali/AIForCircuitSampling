"""sample_chaotic — BIASED on non-Porter-Thomas distributions. Kept as the
v1 baseline for chaotic-vs-unbiased comparisons.

Wraps quimb's `Circuit.sample_chaotic`, which puts a UNIFORM PRIOR on
every qubit not in `marginal_qubits` (literally `rng.choice(("0", "1"))`).
For non-fully-PT circuits this introduces measurable bias in XEB / NLL
when `marginal_qubits < n`. Use `aics.sampling.exact_tn.sample_exact_tn`
instead for unbiased draws.
"""
import numpy as np

from ._quimb_circuit import _resolve_qcirc


def sample_chaotic(circ_or_qcirc, qubits=None, *, k_samples,
                    marginal_qubits=None, seed=None,
                    optimize="auto-hq", dtype="complex64", to_backend=None):
    """Draw k_samples bitstrings via quimb.Circuit.sample_chaotic.

    BIASED on non-PT distributions when marginal_qubits < n. Use
    `sample_exact_tn` for unbiased.

    `circ_or_qcirc` may be either a cirq.Circuit (in which case `qubits`
    must be passed) or a prebuilt quimb Circuit. `marginal_qubits` defaults
    to min(20, n).

    Returns (k_samples, n) uint8 bit array, MSB-first (qubits[0] = bit 0).
    """
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
