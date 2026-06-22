"""Sycamore brickwork RCS (Arute et al. 2019): cirq_google.SYC fSim with
GRID_STAGGERED_PATTERN. Alternative to the boixo_v2 default.
"""
import cirq
import cirq_google
from cirq.experiments import random_quantum_circuit_generation as rqcg


_SUGGESTED_GRIDS = {
    8: (2, 4), 10: (2, 5), 12: (3, 4), 16: (4, 4), 20: (4, 5),
}


def grid_for(n_qubits):
    return _SUGGESTED_GRIDS.get(n_qubits, (1, n_qubits))


def make_sycamore_rcs_circuit(n_qubits=None, depth=10, seed=42,
                                n_rows=None, n_cols=None,
                                two_qubit_gate=None, pattern=None,
                                add_final_single_qubit_layer=True):
    """Returns (qubits, circuit). Grid taken from grid_for(n_qubits)
    unless n_rows/n_cols are passed explicitly.
    """
    if n_rows is None or n_cols is None:
        if n_qubits is None:
            raise ValueError("pass either n_qubits or both n_rows and n_cols")
        n_rows, n_cols = grid_for(n_qubits)
    if two_qubit_gate is None:
        two_qubit_gate = cirq_google.SYC
    if pattern is None:
        pattern = rqcg.GRID_STAGGERED_PATTERN
    qubits = cirq.GridQubit.rect(n_rows, n_cols)
    circuit = rqcg.random_rotations_between_grid_interaction_layers_circuit(
        qubits=qubits, depth=depth, seed=seed,
        two_qubit_op_factory=lambda a, b, _: two_qubit_gate.on(a, b),
        pattern=pattern,
        add_final_single_qubit_layer=add_final_single_qubit_layer,
    )
    return qubits, circuit
