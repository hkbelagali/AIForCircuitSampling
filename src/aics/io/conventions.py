"""Canonical bit/qubit ordering for aics.

`bits` is a (..., n) array; axis q is qubit q. qubits[0] is the MSB:

    int_value = sum_{q} bits[..., q] * 2^(n-1-q)

This matches cirq's `final_state_vector` when `qubit_order=qubits`.
"""
import numpy as np


def bits_to_int(bits):
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.shape[-1]
    return (bits @ (1 << np.arange(n))[::-1]).astype(np.int64)


def int_to_bits(values, n):
    values = np.asarray(values, dtype=np.int64)
    out = np.zeros(values.shape + (n,), dtype=np.uint8)
    for q in range(n):
        out[..., n - 1 - q] = (values >> q) & 1
    return out


def bits_to_strings(bits):
    return ["".join(str(int(b)) for b in row)
            for row in np.asarray(bits, dtype=np.uint8)]


def strings_to_bits(strings):
    return np.array([[int(c) for c in s] for s in strings], dtype=np.uint8)
