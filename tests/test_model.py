"""Ryan's AutoregressiveRNN: shape, normalization, determinism, sampling."""
import numpy as np
import pytest
import torch

try:
    from aics.models.autoregressive_rnn import AutoregressiveRNN
except ImportError:
    from m_rcs_nll_eval_cell import AutoregressiveRNN


@pytest.fixture
def model_n6():
    torch.manual_seed(0)
    return AutoregressiveRNN(n_bits=6, hidden=32, n_layers=2)


def test_log_prob_shape(model_n6):
    x = torch.zeros(5, 6)
    lp = model_n6.log_prob(x)
    assert lp.shape == (5,), f"log_prob shape {lp.shape}, expected (5,)"


def test_log_prob_normalizes_to_one(model_n6):
    """Sum over all 2^n bitstrings of exp(log_prob(x)) must equal 1.

    Strong test that the autoregressive parameterization is properly normalized.
    """
    n = 6
    D = 1 << n
    all_bits = torch.tensor(
        [[(i >> (n - 1 - k)) & 1 for k in range(n)] for i in range(D)],
        dtype=torch.float32,
    )
    with torch.no_grad():
        lp = model_n6.log_prob(all_bits)
    total = float(torch.exp(lp).sum())
    assert abs(total - 1.0) < 1e-4, f"sum_x p(x) = {total}, expected 1.0"


def test_sample_bits_shape_and_validity(model_n6):
    samples = model_n6.sample_bits(20)
    assert samples.shape == (20, 6)
    assert set(np.unique(samples).tolist()) <= {0.0, 1.0}


def test_sample_bits_deterministic_under_seed():
    torch.manual_seed(42)
    m = AutoregressiveRNN(n_bits=4, hidden=16, n_layers=1)
    torch.manual_seed(7)
    s1 = m.sample_bits(50)
    torch.manual_seed(7)
    s2 = m.sample_bits(50)
    assert np.array_equal(s1, s2), "samples should be deterministic given torch seed"


def test_log_prob_grad_flows(model_n6):
    """Gradient of log_prob w.r.t. parameters must be non-trivial — catches
    detached/no-grad bugs."""
    x = torch.zeros(2, 6)
    lp = model_n6.log_prob(x).sum()
    lp.backward()
    nonzero_grads = sum(
        1 for p in model_n6.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert nonzero_grads > 0, "no parameter received non-zero gradient"
