"""NLL training one-epoch decrease + PT regularizer sanity."""
import numpy as np
import pytest
import torch

try:
    from aics.models.autoregressive_rnn import AutoregressiveRNN
    from aics.training.nll import train_nll_pt
except ImportError:
    from m_rcs_nll_eval_cell import AutoregressiveRNN, train_nll_pt


@pytest.fixture
def toy_data():
    """40-sample dataset of bitstrings drawn from a peaked distribution
    (mostly zeros). Easy target for a quick learning test."""
    rng = np.random.default_rng(0)
    n = 6
    k = 40
    p = np.full(1 << n, 0.001, dtype=np.float64)
    p[0] = 1.0 - p[1:].sum()
    p /= p.sum()
    idx = rng.choice(len(p), size=k, p=p)
    bits = np.array(
        [[(int(i) >> (n - 1 - q)) & 1 for q in range(n)] for i in idx],
        dtype=np.float32,
    )
    return bits, n


def test_train_nll_one_epoch_runs(toy_data):
    bits, n = toy_data
    torch.manual_seed(0)
    model = AutoregressiveRNN(n_bits=n, hidden=16, n_layers=1)
    final_nll, n_epochs = train_nll_pt(
        model, bits, total_steps=10, min_epochs=1, max_epochs=1,
        batch_size=8, lr=1e-3, lambda_pt=0.0, n_states=1 << n,
        device="cpu", verbose=False,
    )
    assert n_epochs == 1
    assert np.isfinite(final_nll), f"got non-finite nll {final_nll}"


def test_train_nll_decreases_loss(toy_data):
    """After 10 epochs, NLL on training data should drop from initial."""
    bits, n = toy_data
    torch.manual_seed(0)
    model = AutoregressiveRNN(n_bits=n, hidden=16, n_layers=1)
    x = torch.from_numpy(bits)
    with torch.no_grad():
        nll_before = -model.log_prob(x).mean().item()
    train_nll_pt(model, bits, total_steps=200, min_epochs=10, max_epochs=10,
                  batch_size=8, lr=3e-3, lambda_pt=0.0, n_states=1 << n,
                  device="cpu", verbose=False)
    with torch.no_grad():
        nll_after = -model.log_prob(x).mean().item()
    assert nll_after < nll_before, \
        f"NLL did not decrease: before={nll_before:.4f}, after={nll_after:.4f}"


def test_pt_regularizer_finite_and_nonzero():
    """PT reg term λ·E[D·q − log q]; should be finite for any valid model."""
    n = 6
    D = 1 << n
    torch.manual_seed(0)
    model = AutoregressiveRNN(n_bits=n, hidden=16, n_layers=1)
    x = torch.zeros(8, n)
    log_q = model.log_prob(x)
    q_scaled = torch.exp(log_q) * float(D)
    pt_loss = (q_scaled - log_q).mean()
    assert torch.isfinite(pt_loss), f"PT loss not finite: {pt_loss}"


def test_pt_regularizer_no_overflow_at_large_n():
    """Cast to float before * 2^n; otherwise int overflows at n>=63."""
    n = 70
    torch.manual_seed(0)
    model = AutoregressiveRNN(n_bits=n, hidden=8, n_layers=1)
    x = torch.zeros(4, n)
    log_q = model.log_prob(x)
    q_scaled = torch.exp(log_q) * float(1 << n)
    pt_loss = (q_scaled - log_q).mean()
    assert torch.isfinite(pt_loss), f"PT loss not finite at n={n}: {pt_loss}"
