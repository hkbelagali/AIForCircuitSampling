"""Boixo et al. 2018 ("Google v2") random circuit: CZ + {T, sqrt-X, sqrt-Y}
sandwiched between H layers. Ported from cell 9 of rcs_ml_experiment.ipynb.

`depth` counts CZ layers; each carries a layer of single-qubit gates.
"""
import random
from typing import Callable, Iterable, Sequence, TypeVar, cast

import cirq

T = TypeVar("T")


def _choice(rand_gen: Callable[[], float], sequence: Sequence[T]) -> T:
    return sequence[int(rand_gen() * len(sequence))]


def _make_cz_layer(qubits: Iterable[cirq.GridQubit], layer_index: int):
    layer_index_map = [0, 3, 2, 1, 4, 7, 6, 5]
    internal_layer_index = layer_index_map[layer_index % 8]
    dir_row = internal_layer_index % 2
    dir_col = 1 - dir_row
    shift = (internal_layer_index >> 1) % 4
    for q in qubits:
        q2 = cirq.GridQubit(q.row + dir_row, q.col + dir_col)
        if q2 not in qubits:
            continue
        if (q.row * (2 - dir_row) + q.col * (2 - dir_col)) % 4 != shift:
            continue
        yield cirq.CZ(q, q2)


def _add_cz_layer(layer_index: int, circuit: cirq.Circuit) -> int:
    cz_layer = None
    while not cz_layer:
        qubits = cast(Iterable[cirq.GridQubit], circuit.all_qubits())
        cz_layer = list(_make_cz_layer(qubits, layer_index))
        layer_index += 1
    circuit.append(cz_layer, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return layer_index


def _generate_rcs_circuit(qubits, depth, seed):
    non_diagonal_gates = [cirq.X ** (1 / 2), cirq.Y ** (1 / 2)]
    rand_gen = random.Random(seed).random
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q) for q in qubits)

    layer_index = 0
    if depth:
        layer_index = _add_cz_layer(layer_index, circuit)
        for q in qubits:
            if not circuit.operation_at(q, 1):
                circuit.append(cirq.T(q), strategy=cirq.InsertStrategy.EARLIEST)
        for moment_index in range(2, depth + 1):
            layer_index = _add_cz_layer(layer_index, circuit)
            for q in qubits:
                if not circuit.operation_at(q, moment_index):
                    last_op = circuit.operation_at(q, moment_index - 1)
                    if last_op:
                        gate = cast(cirq.GateOperation, last_op).gate
                        if gate == cirq.CZ:
                            circuit.append(
                                _choice(rand_gen, non_diagonal_gates).on(q),
                                strategy=cirq.InsertStrategy.EARLIEST,
                            )
                        elif gate != cirq.T:
                            circuit.append(cirq.T(q),
                                            strategy=cirq.InsertStrategy.EARLIEST)

    circuit.append([cirq.H(q) for q in qubits],
                    strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return circuit


def grid_dimensions(n_qubits):
    """Smallest n_rows >= 2 dividing n_qubits, else (1, n_qubits).

    Even n_qubits → 2-row ladder; n=12 → (2, 6), n=24 → (2, 12), etc.
    """
    for n_rows in range(2, n_qubits + 1):
        if n_qubits % n_rows == 0:
            return n_rows, n_qubits // n_rows
    return 1, n_qubits


def make_boixo_v2_rcs_circuit(n_qubits, depth=10, seed=42):
    """Returns (qubits, circuit). `depth` = number of CZ layers."""
    n_rows, n_cols = grid_dimensions(n_qubits)
    qubits = [cirq.GridQubit(i, j)
              for i in range(n_rows) for j in range(n_cols)]
    return qubits, _generate_rcs_circuit(qubits, depth, seed)
