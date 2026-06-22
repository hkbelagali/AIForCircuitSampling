"""report() and train_cell() — public API smoke tests."""
import tempfile
from pathlib import Path

import numpy as np
import torch

from aics import train_cell, report, AutoregressiveRNN
from aics.io import save_samples


def _toy_npz(path, n_qubits=4, k=200, k_held=40, k_uni=20, seed=0):
    """Make a small fake sample bundle for testing — no real circuit needed."""
    rng = np.random.default_rng(seed)
    D = 1 << n_qubits
    pC = rng.dirichlet(np.ones(D) * 0.5)
    idx_train = rng.choice(D, size=k, p=pC)
    idx_held = rng.choice(D, size=k_held, p=pC)
    idx_uni = rng.integers(0, D, size=k_uni)
    to_bits = lambda idxs: np.array(
        [[(int(i) >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
         for i in idxs], dtype=np.uint8)
    save_samples(
        path,
        train_bits=to_bits(idx_train), train_pC=pC[idx_train],
        held_bits=to_bits(idx_held), held_pC=pC[idx_held],
        uniform_bits=to_bits(idx_uni), uniform_pC=pC[idx_uni],
        meta={"n": n_qubits, "depth": 2, "circuit_seed": 0,
               "sample_seed": 0, "depth_label": "toy"},
    )


def test_report_returns_canonical_keys():
    n = 4
    torch.manual_seed(0)
    model = AutoregressiveRNN(n_bits=n, hidden=8, n_layers=1)
    rng = np.random.default_rng(0)
    held_bits = rng.integers(0, 2, size=(50, n), dtype=np.uint8)
    held_pC = rng.dirichlet(np.ones(50))
    uniform_pC = rng.dirichlet(np.ones(50))
    out = report(model, held_bits=held_bits, held_pC=held_pC,
                  uniform_pC=uniform_pC, n_qubits=n, device="cpu")
    for key in ("held_nll", "normalized_nll", "nll_excess",
                 "xeb_gen", "xeb_held_cache", "xeb_uniform_cache", "xeb_norm"):
        assert key in out, f"missing {key}"
    assert np.isfinite(out["held_nll"])
    assert np.isfinite(out["xeb_gen"])


def test_report_empty_when_no_held():
    n = 4
    torch.manual_seed(0)
    model = AutoregressiveRNN(n_bits=n, hidden=8, n_layers=1)
    assert report(model, n_qubits=n) == {}


def test_train_cell_end_to_end_nll():
    n = 4
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "toy.npz"
        _toy_npz(npz, n_qubits=n, k=100, k_held=20, k_uni=20)
        result, model = train_cell(
            npz, k_train=80, hidden=8, n_layers=1, loss="nll",
            total_steps=20, min_epochs=2, max_epochs=2, batch_size=16,
            device="cpu",
        )
    assert result["loss"] == "nll"
    assert result["k_train"] == 80
    assert result["pt_regularizer"] is True
    assert "xeb_norm" in result
    assert isinstance(model, AutoregressiveRNN)


def test_train_cell_rejects_curriculum_with_nll():
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "toy.npz"
        _toy_npz(npz, n_qubits=4, k=40)
        try:
            train_cell(npz, k_train=20, hidden=8, n_layers=1, loss="nll",
                        curriculum="weight_ascending",
                        total_steps=2, min_epochs=1, max_epochs=1,
                        device="cpu")
        except ValueError as e:
            assert "curriculum" in str(e)
        else:
            raise AssertionError("expected ValueError for nll + curriculum")


def test_train_cell_rejects_pt_with_z_pauli():
    with tempfile.TemporaryDirectory() as tmp:
        npz = Path(tmp) / "toy.npz"
        _toy_npz(npz, n_qubits=4, k=40)
        try:
            train_cell(npz, k_train=20, hidden=8, n_layers=1,
                        loss="z_pauli", pt_regularizer=True,
                        epochs_per_stage=2, device="cpu")
        except ValueError as e:
            assert "pt_regularizer" in str(e)
        else:
            raise AssertionError("expected ValueError for z_pauli + pt_regularizer")
