"""Sycamore-style brickwork RCS circuits — alternative to the Boixo v2
default. Use this when matching Arute 2019 hardware connectivity / gates.

Uses cirq.experiments.random_quantum_circuit_generation under the hood —
the same function recirq.random_circuit_sampling.make_rcs_circuit calls.
Default two-qubit gate is cirq_google.SYC (Sycamore fSim) with the
GRID_STAGGERED_PATTERN ABCDCDAB 8-cycle from arXiv:1910.11333 Sec. VII C.

We bypass recirq directly because (a) recirq requires Python >= 3.10 and
we're on 3.9, and (b) recirq's setup.py force-installs heavy extras
(openfermion, pyscf, …) we don't need. Methodology is identical.
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
    return (1, n)


def make_sycamore_rcs_circuit(n_qubits=None, depth=10, seed=42,
                                n_rows=None, n_cols=None,
                                two_qubit_gate=None, pattern=None,
                                add_final_single_qubit_layer=True):
    """Sycamore-style brickwork RCS circuit. Same calling shape as
    `aics.circuits.boixo_v2.make_boixo_v2_rcs_circuit`: pass `n_qubits`
    and the grid is chosen via `grid_for(n_qubits)`. Override with
    explicit `n_rows`/`n_cols` if needed.

    Returns (qubits, circuit).
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
        qubits=qubits,
        depth=depth,
        seed=seed,
        two_qubit_op_factory=lambda a, b, _: two_qubit_gate.on(a, b),
        pattern=pattern,
        add_final_single_qubit_layer=add_final_single_qubit_layer,
    )
    return qubits, circuit


# Back-compat alias for the function name used by archive/src/aics
make_rcs_circuit = make_sycamore_rcs_circuit
