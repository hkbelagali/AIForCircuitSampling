"""Circuit construction sanity. Importable from both prototype/ (pre-migration)
and src/aics/circuits/ (post-migration) via conftest.py path setup.
"""
import numpy as np
import pytest

from aics.circuits.boixo_v2 import make_boixo_v2_rcs_circuit, grid_dimensions
from aics.circuits.exact import exact_probabilities


def test_grid_dimensions_returns_2_row_ladders_for_even_n():
    for n in (4, 8, 12, 16, 20, 24):
        rows, cols = grid_dimensions(n)
        assert rows * cols == n
        assert rows <= cols, f"non-canonical: n={n} -> {rows}x{cols}"
        assert rows == 2 or n < 4, f"expected 2-row ladder, got {rows}x{cols}"


def test_boixo_circuit_is_deterministic_in_seed():
    qubits1, c1 = make_boixo_v2_rcs_circuit(8, cz_depth=4, seed=42)
    qubits2, c2 = make_boixo_v2_rcs_circuit(8, cz_depth=4, seed=42)
    qubits3, c3 = make_boixo_v2_rcs_circuit(8, cz_depth=4, seed=43)
    assert c1 == c2, "same seed should produce identical circuit"
    assert c1 != c3, "different seed should produce different circuit"


def test_boixo_circuit_has_correct_qubit_count():
    for n in (4, 8, 12):
        qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=2, seed=0)
        assert len(qubits) == n
        assert len(circ.all_qubits()) == n


def test_exact_probabilities_sum_to_one_small_n():
    qubits, circ = make_boixo_v2_rcs_circuit(6, cz_depth=4, seed=0)
    pC = exact_probabilities(circ, qubits)
    assert pC.shape == (1 << 6,)
    assert pC.dtype in (np.float32, np.float64)
    assert pC.min() >= 0.0
    assert abs(pC.sum() - 1.0) < 1e-6, f"sum(pC) = {pC.sum()}, expected 1.0"


def test_exact_probabilities_deterministic_in_seed():
    qubits, circ = make_boixo_v2_rcs_circuit(4, cz_depth=2, seed=7)
    p1 = exact_probabilities(circ, qubits)
    p2 = exact_probabilities(circ, qubits)
    assert np.allclose(p1, p2), "exact_probabilities must be deterministic"
