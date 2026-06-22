"""Z-observable training: AutoregressiveRNN fits empirical <Z_S> targets."""
import numpy as np
import pytest
import torch

from aics.training.z_pauli import train_z_pauli
from aics.eval.z_observables import (
    enumerate_z_supports, empirical_z_expectations, model_z_expectations,
)
from aics.models.autoregressive_rnn import AutoregressiveRNN


def _toy_samples(n_qubits, k, seed):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(k, n_qubits)).astype(np.uint8)
    powers = 2 ** np.arange(n_qubits - 1, -1, -1, dtype=np.int64)
    return (bits @ powers).astype(np.int64), bits


def test_enumerate_z_supports_count():
    """n=4, w≤2: 1 + C(4,1) + C(4,2) = 11."""
    supports, weights = enumerate_z_supports(4, max_weight=2)
    assert len(supports) == 11
    assert weights.tolist() == [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]


def test_empirical_z_uniform_samples_near_zero():
    n_qubits = 6
    samples_int, _ = _toy_samples(n_qubits, k=20_000, seed=0)
    supports, _ = enumerate_z_supports(n_qubits, max_weight=3)
    emps = empirical_z_expectations(samples_int, supports, n_qubits)
    assert abs(emps[0] - 1.0) < 1e-12
    assert np.abs(emps[1:]).max() < 0.05


def test_z_pauli_fits_weight1():
    n_qubits = 4
    samples_int, _ = _toy_samples(n_qubits, k=200, seed=1)
    supports, weights = enumerate_z_supports(n_qubits, max_weight=1)
    torch.manual_seed(0)
    model = AutoregressiveRNN(n_bits=n_qubits, hidden=16, n_layers=1)
    final_loss = train_z_pauli(
        model, samples_int, supports, weights, n_qubits,
        epochs=80, lr=3e-3, device="cpu", verbose=False,
    )
    assert final_loss < 0.1, f"final loss {final_loss:.4f}"


def test_model_z_expectations_shape():
    n_qubits = 4
    supports, _ = enumerate_z_supports(n_qubits, max_weight=2)
    torch.manual_seed(0)
    model = AutoregressiveRNN(n_bits=n_qubits, hidden=8, n_layers=1)
    exps = np.asarray(model_z_expectations(model, supports, n_qubits))
    assert exps.shape == (len(supports),)
    assert abs(exps[0] - 1.0) < 1e-4
    assert np.abs(exps).max() <= 1.0 + 1e-6
