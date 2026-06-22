"""p_C(z) = |<z|U|0>|^2 via quimb amplitude contraction."""
import numpy as np

from ._quimb_circuit import _resolve_qcirc


def amplitudes_tn(circ, qubits=None, bitstrings=None, *,
                    optimize="auto-hq", dtype=None):
    """(k, n_qubits) uint8 bits (MSB-first) → (k,) float64 probabilities."""
    if bitstrings is None:
        raise ValueError("bitstrings is required")
    qcirc = _resolve_qcirc(circ, qubits)
    bs = np.asarray(bitstrings, dtype=np.uint8)
    probs = np.empty(len(bs), dtype=np.float64)
    for i in range(len(bs)):
        s = "".join(str(int(b)) for b in bs[i])
        amp = qcirc.amplitude(s, optimize=optimize, dtype=dtype)
        amp_val = complex(amp.item()) if hasattr(amp, "item") else complex(amp)
        probs[i] = float(abs(amp_val) ** 2)
    return probs
