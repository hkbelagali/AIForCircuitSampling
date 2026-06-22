"""sample_exact_tn — unbiased TN-based sampler. Default sampling path.

Wraps quimb's `Circuit.sample`, which does sequential marginal-conditional
sampling: at each qubit q in order, contracts the lightcone-trimmed TN to
get `P(b_q | b_<q)` exactly, draws `b_q` from it, caches the marginal,
moves on. No coin-flip fallback like `sample_chaotic` has.

Exact (no truncation) — equivalent to TEBD-style sampling on a fully
contracted TN. For our 2-row Boixo ladders at depth 10 the lightcone
treewidth is bounded ~4, so each marginal is cheap and total cost is
~ k * n / group_size contractions with marginal caching reducing the
constant further.
"""
import numpy as np

from ._quimb_circuit import _resolve_qcirc


def sample_exact_tn(circ_or_qcirc, qubits=None, *, k_samples,
                     seed=None, group_size=10, optimize="auto-hq",
                     dtype="complex64", to_backend=None,
                     max_marginal_storage=2 ** 20):
    """Draw k_samples bitstrings via sequential marginal-conditional
    contraction (quimb.Circuit.sample).

    `group_size` sets how many qubits share a single marginal contraction
    (larger = fewer contractions, each more expensive).

    Returns (k_samples, n) uint8 bit array, MSB-first (qubits[0] = bit 0).
    """
    qcirc = _resolve_qcirc(circ_or_qcirc, qubits,
                            dtype=dtype, to_backend=to_backend)
    n = qcirc.N
    out = np.empty((k_samples, n), dtype=np.uint8)
    gen = qcirc.sample(
        C=k_samples, group_size=group_size, seed=seed,
        optimize=optimize, dtype=dtype,
        max_marginal_storage=max_marginal_storage,
    )
    for i, bitstring in enumerate(gen):
        out[i] = np.frombuffer(bitstring.encode("ascii"),
                                dtype=np.uint8) - ord("0")
    return out
