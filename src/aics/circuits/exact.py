"""Exact reference via cirq full statevector. Tractable up to n ≈ 26."""
import cirq
import numpy as np

from ..io.conventions import bits_to_int, int_to_bits  # re-exported for convenience

__all__ = [
    "exact_probabilities", "sample_from_circuit",
    "bits_to_int", "int_to_bits",
]


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
