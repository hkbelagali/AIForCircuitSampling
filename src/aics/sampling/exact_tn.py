"""sample_exact_tn — unbiased TN sampler (default).

Wraps quimb.Circuit.sample: sequential marginal-conditional contraction
with lightcone trimming and marginal caching. Exact (no truncation).
"""
import numpy as np

from ._quimb_circuit import _resolve_qcirc


def sample_exact_tn(circ, qubits=None, *, k_samples,
                     seed=None, group_size=10, optimize="auto-hq",
                     dtype="complex64", to_backend=None,
                     max_marginal_storage=2 ** 20):
    """`circ` is a cirq.Circuit (+ qubits) or a prebuilt quimb Circuit.
    Returns (k_samples, n_qubits) uint8, MSB-first."""
    qcirc = _resolve_qcirc(circ, qubits, dtype=dtype, to_backend=to_backend)
    n_qubits = qcirc.N
    out = np.empty((k_samples, n_qubits), dtype=np.uint8)
    gen = qcirc.sample(
        C=k_samples, group_size=group_size, seed=seed,
        optimize=optimize, dtype=dtype,
        max_marginal_storage=max_marginal_storage,
    )
    for i, bitstring in enumerate(gen):
        out[i] = np.frombuffer(bitstring.encode("ascii"),
                                dtype=np.uint8) - ord("0")
    return out
