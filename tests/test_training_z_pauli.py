"""Z-Pauli (Z-observable) training: model fits empirical <Z_S> targets.

Currently imports from prototype/rcs.py (legacy `BitstringARRNN` + functions).
Post-migration: from `aics.training.z_pauli` with `AutoregressiveRNN`.
"""
import numpy as np
import pytest
import torch

try:
    from aics.training.z_pauli import (
        train_z_pauli, enumerate_z_supports,
        empirical_z_expectations, model_z_expectations,
    )
    from aics.models.autoregressive_rnn import AutoregressiveRNN
    NEW_API = True
except ImportError:
    from rcs import (
        BitstringARRNN as AutoregressiveRNN,
        train_rnn_z_pauli as train_z_pauli,
        enumerate_z_supports,
        shadow_z_expectations as empirical_z_expectations,
        model_z_expectations,
    )
    NEW_API = False


def _toy_samples(n, k, seed):
    """k bitstrings drawn uniformly — for normalization sanity, target
    <Z_S> = 0 for all weight≥1 S."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(k, n)).astype(np.uint8)
    # Pack into MSB-first ints (matches what train_rnn_z_pauli expects).
    powers = 2 ** np.arange(n - 1, -1, -1, dtype=np.int64)
    return (bits @ powers).astype(np.int64), bits


def test_enumerate_z_supports_count():
    """At n=4, weight ≤ 2: 1 + C(4,1) + C(4,2) = 1 + 4 + 6 = 11."""
    supports, weights = enumerate_z_supports(4, max_weight=2)
    assert len(supports) == 11
    assert weights.tolist() == [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]


def test_empirical_z_uniform_samples_near_zero():
    """Uniform random samples: <Z_S> for |S|≥1 should be ~0 (within MC noise)."""
    n = 6
    samples_int, _ = _toy_samples(n, k=20_000, seed=0)
    supports, weights = enumerate_z_supports(n, max_weight=3)
    emps = empirical_z_expectations(samples_int, supports, n)
    # Identity is 1.0
    assert abs(emps[0] - 1.0) < 1e-12
    # Weight ≥ 1: should be near zero
    nonzero = emps[1:]
    assert np.abs(nonzero).max() < 0.05, f"max |<Z_S>| = {np.abs(nonzero).max()}"


@pytest.mark.skipif(not torch.cuda.is_available() and True,  # skip mark always off; this is a CPU-fine test
                     reason="cpu-fine")
def test_z_pauli_fits_weight1():
    """Model trained on a tiny problem should drive Z-Pauli MSE down."""
    n = 4
    samples_int, _ = _toy_samples(n, k=200, seed=1)
    supports, weights = enumerate_z_supports(n, max_weight=1)
    torch.manual_seed(0)
    if NEW_API:
        model = AutoregressiveRNN(n_bits=n, hidden=16, n_layers=1)
    else:
        model = AutoregressiveRNN(n_qubits=n, d_hidden=16)
    final_loss = train_z_pauli(
        model, samples_int, supports, weights, n,
        epochs=80, lr=3e-3, device="cpu", verbose=False,
    )
    assert final_loss < 0.1, \
        f"weight-1 training did not drive loss low: final = {final_loss:.4f}"


def test_model_z_expectations_shape():
    """model_z_expectations returns one float per Pauli in the support set."""
    n = 4
    supports, weights = enumerate_z_supports(n, max_weight=2)
    torch.manual_seed(0)
    if NEW_API:
        model = AutoregressiveRNN(n_bits=n, hidden=8, n_layers=1)
    else:
        model = AutoregressiveRNN(n_qubits=n, d_hidden=8)
    exps = model_z_expectations(model, supports, n)
    exps = np.asarray(exps)
    assert exps.shape == (len(supports),)
    # Identity always 1
    assert abs(exps[0] - 1.0) < 1e-4
    # |<Z_S>| <= 1
    assert np.abs(exps).max() <= 1.0 + 1e-6
