"""Exact reference: full statevector and cirq-based Born sampling. Use for
small n (<= ~26) and for parity-check tests against the TN samplers.

Bit ordering: matches cirq's statevector convention. For state index `i`
and qubit order [q_0, ..., q_{n-1}]:
    bit 0 (MSB)  = value of q_0
    bit n-1 (LSB) = value of q_{n-1}
i.e. `int_value = sum_k bit_k * 2^(n-1-k)`. The canonical helpers live in
`aics.io.conventions`; the local `bits_to_int` / `int_to_bits` here are
retained for backward-compat and follow the same convention.
"""

import cirq
import numpy as np


def exact_probabilities(circuit, qubits):
    """Exact p_C(x) = |<x|U|0>|^2 for all 2^n bitstrings, in cirq ordering."""
    sv = cirq.Simulator().simulate(
        circuit, qubit_order=qubits).final_state_vector
    p = np.abs(sv) ** 2
    return p.astype(np.float64)


def sample_from_circuit(circuit, qubits, k, seed=None):
    """Draw k samples from p_C via cirq. Returns (k,) int array, cirq ordering."""
    measured = circuit + cirq.measure(*qubits, key="m")
    sim = cirq.Simulator(seed=seed)
    result = sim.run(measured, repetitions=k)
    bits = result.measurements["m"]
    return bits_to_int(bits)


def bits_to_int(bits):
    """(k, n) bit array -> (k,) ints, bits[:, 0] as MSB."""
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.shape[1]
    powers = (1 << np.arange(n))[::-1]
    return (bits @ powers).astype(np.int64)


def int_to_bits(values, n):
    """(k,) ints -> (k, n) bit array, MSB-first."""
    values = np.asarray(values, dtype=np.int64)
    out = np.zeros((values.size, n), dtype=np.int64)
    for i in range(n):
        out[:, n - 1 - i] = (values >> i) & 1
    return out
