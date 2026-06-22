"""Eval metrics: XEB, parity, per-weight RMS error — known-input sanity."""
import numpy as np
import pytest

try:
    from aics.eval.xeb import linear_xeb
    from aics.eval.z_observables import parity_per_support, per_weight_rms_err
except ImportError:
    from m_rcs_nll_eval_cell import (
        linear_xeb, parity_per_support, per_weight_rms_err,
    )


def test_linear_xeb_uniform_samples_near_zero():
    """XEB = D · <p_C(z)>_z~uniform − 1.  For uniform z this is D·(1/D) − 1 = 0."""
    n = 6
    D = 1 << n
    pC = np.full(D, 1.0 / D)  # uniform "p_C" — XEB always 0 regardless of z
    rng = np.random.default_rng(0)
    idx = rng.integers(0, D, size=5000)
    xeb = linear_xeb(idx, pC)
    assert abs(xeb) < 1e-9, f"uniform-on-uniform XEB = {xeb}"


def test_linear_xeb_peaked_distribution():
    """If p_C is peaked on a few states and we sample those states, XEB is high."""
    D = 16
    pC = np.zeros(D)
    pC[0] = pC[3] = pC[7] = 1.0 / 3
    idx = np.array([0, 3, 7, 0, 3, 7])
    xeb = linear_xeb(idx, pC)
    expected = D * (1.0 / 3) - 1  # D * mean(pC[idx])
    assert abs(xeb - expected) < 1e-9


def test_parity_per_support_identity():
    """The empty support (identity Pauli) always has <I> = 1."""
    samples = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    out = parity_per_support(samples, [()])
    assert out[0] == 1.0


def test_parity_per_support_single_qubit():
    """<Z_0> on samples = mean(1 - 2*bit_0). For [0,1,0,1,1] -> (3 zeros, 2 ones) → 0.2."""
    samples = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
                        dtype=np.uint8)
    out = parity_per_support(samples, [(0,)])
    # 2 zeros, 3 ones at qubit 0 → mean = (1+1-1-1-1)/5 = -0.2
    assert abs(out[0] - (-0.2)) < 1e-9


def test_parity_per_support_pair():
    """<Z_0 Z_1> = mean((-1)^{b0+b1}). For [(00),(11),(01),(10)] → +1,+1,-1,-1 → 0."""
    samples = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.uint8)
    out = parity_per_support(samples, [(0, 1)])
    assert abs(out[0] - 0.0) < 1e-9


def test_per_weight_rms_err():
    supports = [(), (0,), (1,), (0, 1)]
    pred = np.array([1.0, 0.5, -0.3, 0.0])
    true = np.array([1.0, 0.6, -0.4, 0.2])
    out = per_weight_rms_err(supports, pred, true)
    # weight 0: pred=true → 0
    assert abs(out[0]) < 1e-12
    # weight 1: errors {0.1, 0.1} → RMS = 0.1
    assert abs(out[1] - 0.1) < 1e-12
    # weight 2: errors {0.2} → RMS = 0.2
    assert abs(out[2] - 0.2) < 1e-12
