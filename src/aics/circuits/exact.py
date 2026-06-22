"""Exact reference via cirq full statevector. Tractable up to n ≈ 26."""
import cirq
import numpy as np


def exact_probabilities(circuit, qubits):
    """p_C(x) = |<x|U|0>|^2 for all 2^n bitstrings, in cirq's qubit_order."""
    sv = cirq.Simulator().simulate(
        circuit, qubit_order=qubits).final_state_vector
    return (np.abs(sv) ** 2).astype(np.float64)


def sample_from_circuit(circuit, qubits, k, seed=None):
    """Draw k Born samples via cirq. Returns (k,) int array, MSB-first."""
    measured = circuit + cirq.measure(*qubits, key="m")
    bits = cirq.Simulator(seed=seed).run(measured, repetitions=k).measurements["m"]
    return bits_to_int(bits)


def bits_to_int(bits):
    """(k, n) bits → (k,) ints, bits[:, 0] is MSB."""
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.shape[1]
    return (bits @ (1 << np.arange(n))[::-1]).astype(np.int64)


def int_to_bits(values, n):
    """(k,) ints → (k, n) bits, MSB-first."""
    values = np.asarray(values, dtype=np.int64)
    out = np.zeros((values.size, n), dtype=np.int64)
    for i in range(n):
        out[:, n - 1 - i] = (values >> i) & 1
    return out
