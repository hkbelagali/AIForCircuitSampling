"""TN-based amplitude evaluation: p_C(z) = |<z|U|0>|^2 for arbitrary
bitstrings, computed via quimb's contraction (sharing the same TN as the
samplers).
"""
import numpy as np

from ._quimb_circuit import _resolve_qcirc


def amplitudes_tn(circ_or_qcirc, qubits=None, bitstrings=None, *,
                    optimize="auto-hq", dtype=None, tree=None):
    """Return |<z|U|0>|^2 for each bitstring z.

    bitstrings: (k, n) uint8 array, MSB-first (qubits[0] = bit 0).
    tree: cached cotengra path (optional, speeds repeated calls).

    Returns (k,) float64.
    """
    if bitstrings is None:
        raise ValueError("bitstrings is required")
    qcirc = _resolve_qcirc(circ_or_qcirc, qubits)
    bs = np.asarray(bitstrings, dtype=np.uint8)
    k = len(bs)
    probs = np.empty(k, dtype=np.float64)
    opt = tree if tree is not None else optimize
    for i in range(k):
        s = "".join(str(int(b)) for b in bs[i])
        amp = qcirc.amplitude(s, optimize=opt, dtype=dtype)
        amp_val = complex(amp.item()) if hasattr(amp, "item") else complex(amp)
        probs[i] = float(abs(amp_val) ** 2)
    return probs


def prepare_amplitude_tree(circ_or_qcirc, qubits=None, *,
                             optimize="auto-hq"):
    """Precompute a cotengra contraction tree for repeated `amplitudes_tn`
    calls. Returns an object you can pass back as `tree=...`.
    """
    qcirc = _resolve_qcirc(circ_or_qcirc, qubits)
    n = qcirc.N
    return qcirc.amplitude_rehearse("0" * n, optimize=optimize)["tree"]
