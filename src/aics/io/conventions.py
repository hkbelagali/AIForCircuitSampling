"""Canonical bit/qubit ordering for the aics codebase.

ALL aics modules use these helpers when converting between bit-arrays and
integer indices. If you find yourself writing `(bits @ powers)` or
`(idx >> k) & 1` in any other module, replace the call with one of these
and add a one-line docstring claim of compliance.

Convention (matches cirq's statevector ordering):

  - `bits` is a (..., n) array where axis `q` corresponds to qubit `q`
    in the qubit list `[q_0, q_1, ..., q_{n-1}]`.
  - `qubits[0]` is the MSB. That is, integer index
        i  =  sum_{q=0}^{n-1}  bits[..., q] * 2^(n-1-q)
  - This is the same convention cirq uses for `final_state_vector` when
    you pass `qubit_order=qubits`. So `int_to_bits(np.arange(2**n), n)`
    matches the row indexing of cirq's statevector.

DON'T mix conventions silently. If a function returns bits in some other
order (e.g. quimb's `sample` strings, which match the qubit-tag order),
adapt them at the boundary with `aics.io.conventions` helpers and document
the conversion.
"""
import numpy as np


def bits_to_int(bits):
    """(..., n) bit array -> (...,) ints. `bits[..., 0]` is MSB."""
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.shape[-1]
    powers = (1 << np.arange(n))[::-1]
    return (bits @ powers).astype(np.int64)


def int_to_bits(values, n):
    """(...,) ints -> (..., n) bit array. Output[..., 0] is MSB."""
    values = np.asarray(values, dtype=np.int64)
    out = np.zeros(values.shape + (n,), dtype=np.uint8)
    for q in range(n):
        out[..., n - 1 - q] = (values >> q) & 1
    return out


def bits_to_strings(bits):
    """(k, n) bits -> length-n '0'/'1' strings, list of length k. MSB-first."""
    arr = np.asarray(bits, dtype=np.uint8)
    return ["".join(str(int(b)) for b in row) for row in arr]


def strings_to_bits(strings):
    """List of '0'/'1' strings (MSB-first) -> (k, n) uint8 bit array."""
    return np.array([[int(c) for c in s] for s in strings], dtype=np.uint8)
