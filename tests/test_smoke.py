"""Smoke tests: run small cells end-to-end and check they converge."""

import pytest

from m9 import run_cell


@pytest.mark.parametrize("L,k,max_steps", [
    (4, 200, 800),
    (6, 100, 1500),
])
def test_cell_converges(L, k, max_steps):
    rec = run_cell(L, k, seed=0, max_steps=max_steps)
    st = rec["steps_to_threshold"]
    assert st is not None, f"L={L} k={k} did not converge within {max_steps} steps"
    assert st <= max_steps
