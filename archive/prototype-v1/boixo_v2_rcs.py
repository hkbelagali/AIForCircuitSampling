"""Google-v2 / Boixo-2018 RCS circuit construction.

Direct port of cell 9 of Ryan's notebook (rcs_ml_experiment.ipynb), which
itself sources from
  https://github.com/quantumlib/ReCirq/blob/main/recirq/beyond_classical/google_v2_beyond_classical.py
i.e. the "Google v2" supremacy benchmark from Boixo et al., Nature Physics
14, 595 (2018) — arXiv:1608.00263.

NOTE: this is NOT the same as the Sycamore-brickwork RCS in
`aics.circuits.brickwork` (which uses the cirq_google.SYC fSim gate and
the GRID_STAGGERED_PATTERN from the 2019 supremacy paper). The two are
distinct circuit families:

  - Boixo v2 (this file): CZ + {T, sqrt(X), sqrt(Y)} sandwiched between
    H-layers. Originally evaluated at depth >= 30 to reach Porter-Thomas;
    at depth 10 the distribution is chaotic but not maximally PT.

  - Sycamore brickwork (brickwork.py): cirq_google.SYC + {sqrt(X),
    sqrt(Y), sqrt(W)}, matches Arute 2019 hardware. Reaches PT faster
    due to stronger entangling gate.

We use Boixo v2 here to match Ryan's existing n=12, 16 data exactly
(same circuit, depth, and seed -> bit-identical amplitudes), enabling
direct head-to-head scaling comparison.
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


def generate_rcs_circuit(qubits: Iterable[cirq.GridQubit], cz_depth: int,
                          seed: int) -> cirq.Circuit:
    """Google v2 RCS circuit (Boixo et al. 2018)."""
    non_diagonal_gates = [cirq.X ** (1 / 2), cirq.Y ** (1 / 2)]
    rand_gen = random.Random(seed).random
    circuit = cirq.Circuit()

    circuit.append(cirq.H(q) for q in qubits)

    layer_index = 0
    if cz_depth:
        layer_index = _add_cz_layer(layer_index, circuit)
        for q in qubits:
            if not circuit.operation_at(q, 1):
                circuit.append(cirq.T(q), strategy=cirq.InsertStrategy.EARLIEST)
        for moment_index in range(2, cz_depth + 1):
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
                            circuit.append(
                                cirq.T(q),
                                strategy=cirq.InsertStrategy.EARLIEST,
                            )

    circuit.append([cirq.H(q) for q in qubits],
                    strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return circuit


def grid_dimensions(n_qubits: int):
    """Ryan's exact formula: smallest n_rows >= 2 such that n_rows divides
    n_qubits, else (1, n_qubits). Yields 2-row ladders for even n.

    n=12 -> (2, 6)    n=24 -> (2, 12)    n=40 -> (2, 20)
    n=16 -> (2, 8)    n=32 -> (2, 16)    n=48 -> (2, 24)
    n=20 -> (2, 10)
    """
    for n_rows in range(2, n_qubits + 1):
        if n_qubits % n_rows == 0:
            return n_rows, n_qubits // n_rows
    return 1, n_qubits


def make_boixo_v2_rcs_circuit(n_qubits, cz_depth=10, seed=42):
    """Build a Boixo-v2 RCS circuit at the requested size.

    Returns (qubits, circuit). Mirrors the calling convention of
    `aics.circuits.brickwork.make_rcs_circuit` so downstream code can
    swap between circuit families without other changes.
    """
    n_rows, n_cols = grid_dimensions(n_qubits)
    qubits = [cirq.GridQubit(i, j)
              for i in range(n_rows) for j in range(n_cols)]
    circuit = generate_rcs_circuit(qubits, cz_depth, seed)
    return qubits, circuit
