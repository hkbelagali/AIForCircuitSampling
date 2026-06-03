"""Sycamore-style brickwork random circuits.

Uses cirq.experiments.random_quantum_circuit_generation under the hood — the
same function that recirq.random_circuit_sampling.make_rcs_circuit calls
internally. Default two-qubit gate is cirq_google.SYC (Sycamore fSim), and
default pattern is GRID_STAGGERED_PATTERN (the ABCDCDAB 8-cycle from
arXiv:1910.11333 Sec. VII C).

We deliberately bypass `recirq` directly because (a) recirq requires Python
>=3.10 and we're on 3.9, and (b) recirq's setup.py force-installs heavy extras
(openfermion, pyscf, ...) we don't need. The methodology here is identical.
"""

import cirq
import cirq_google
from cirq.experiments import random_quantum_circuit_generation as rqcg


_SUGGESTED_GRIDS = {
    8: (2, 4),
    10: (2, 5),
    12: (3, 4),
    16: (4, 4),
    20: (4, 5),
}


def grid_for(n):
    """A sensible (rows, cols) for a given qubit count."""
    if n in _SUGGESTED_GRIDS:
        return _SUGGESTED_GRIDS[n]
    return (1, n)  # 1D chain fallback


def make_rcs_circuit(n_rows, n_cols, depth, seed,
                     two_qubit_gate=None, pattern=None,
                     add_final_single_qubit_layer=True):
    """Sycamore-style brickwork RCS circuit on an n_rows x n_cols grid.

    Mirrors recirq.random_circuit_sampling.make_rcs_circuit. Single-qubit
    layer factory defaults to cirq's Sycamore set {sqrt(X), sqrt(Y), sqrt(W)}.

    Returns (qubits, circuit).
    """
    if two_qubit_gate is None:
        two_qubit_gate = cirq_google.SYC
    if pattern is None:
        pattern = rqcg.GRID_STAGGERED_PATTERN

    qubits = cirq.GridQubit.rect(n_rows, n_cols)
    circuit = rqcg.random_rotations_between_grid_interaction_layers_circuit(
        qubits=qubits,
        depth=depth,
        seed=seed,
        two_qubit_op_factory=lambda a, b, _: two_qubit_gate.on(a, b),
        pattern=pattern,
        add_final_single_qubit_layer=add_final_single_qubit_layer,
    )
    return qubits, circuit
