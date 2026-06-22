"""Eval metrics: XEB, parity, per-weight RMS error — known-input sanity."""
import numpy as np

from aics.eval.xeb import linear_xeb
from aics.eval.z_observables import empirical_z_expectations, per_weight_rms_err


def test_linear_xeb_uniform_samples_near_zero():
    """For uniform p_C, XEB = D · (1/D) − 1 = 0 regardless of z."""
    n = 6
    D = 1 << n
    pC = np.full(D, 1.0 / D)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, D, size=5000)
    assert abs(linear_xeb(idx, pC)) < 1e-9


def test_linear_xeb_peaked_distribution():
    D = 16
    pC = np.zeros(D)
    pC[0] = pC[3] = pC[7] = 1.0 / 3
    idx = np.array([0, 3, 7, 0, 3, 7])
    expected = D * (1.0 / 3) - 1
    assert abs(linear_xeb(idx, pC) - expected) < 1e-9


def test_linear_xeb_accepts_bit_array():
    """linear_xeb now dispatches on ndim: (k,) ints or (k, n) bits."""
    n = 3
    D = 1 << n
    pC = np.zeros(D)
    pC[5] = 1.0  # bitstring "101"
    bits = np.array([[1, 0, 1]], dtype=np.uint8)  # int 5
    assert abs(linear_xeb(bits, pC) - (D * 1.0 - 1)) < 1e-9


def test_empirical_z_identity():
    """Empty support (Z_∅ = I) always has <I> = 1."""
    samples = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    assert empirical_z_expectations(samples, [()], 3)[0] == 1.0


def test_empirical_z_single_qubit():
    """For [0,1,0,1,1] at qubit 0: <Z_0> = mean((-1)^bit) = -0.2."""
    samples = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
                        dtype=np.uint8)
    out = empirical_z_expectations(samples, [(0,)], 3)
    assert abs(out[0] - (-0.2)) < 1e-9


def test_empirical_z_pair():
    """For [(00),(11),(01),(10)]: <Z_0 Z_1> averages to 0."""
    samples = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.uint8)
    out = empirical_z_expectations(samples, [(0, 1)], 2)
    assert abs(out[0]) < 1e-9


def test_per_weight_rms_err():
    supports = [(), (0,), (1,), (0, 1)]
    pred = np.array([1.0, 0.5, -0.3, 0.0])
    true = np.array([1.0, 0.6, -0.4, 0.2])
    out = per_weight_rms_err(supports, pred, true)
    assert abs(out[0]) < 1e-12        # w=0: errors all zero
    assert abs(out[1] - 0.1) < 1e-12  # w=1: RMS of {0.1, 0.1} = 0.1
    assert abs(out[2] - 0.2) < 1e-12  # w=2: RMS of {0.2} = 0.2
