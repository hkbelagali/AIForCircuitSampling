"""Canonical bit/qubit ordering for aics.

`bits` is a (..., n_qubits) array; axis q is qubit q. qubits[0] is the MSB:

    int_value = sum_q bits[..., q] * 2^(n-1-q)

Matches cirq's `final_state_vector` when `qubit_order=qubits`.
"""
import numpy as np


def bits_to_int(bits):
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.shape[-1]
    return (bits @ (1 << np.arange(n))[::-1]).astype(np.int64)


def int_to_bits(values, n_qubits):
    values = np.asarray(values, dtype=np.int64)
    out = np.zeros(values.shape + (n_qubits,), dtype=np.uint8)
    for q in range(n_qubits):
        out[..., n_qubits - 1 - q] = (values >> q) & 1
    return out
