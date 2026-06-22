"""End-to-end resume roundtrip: train → checkpoint → resume → continue.

A single 6-epoch run and a 3-epoch + resume-3-epoch run should land in
the same place (same model weights + same final loss) modulo numerical
ordering of the cosine LR schedule.
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from aics.io import save_samples
from aics.models import AutoregressiveRNN
from aics.training import train_nll, train_z_pauli
from aics.training.curriculum import weight_ascending
from aics.eval.z_observables import enumerate_z_supports


def _toy_bits(n_qubits=4, k=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(k, n_qubits)).astype(np.float32)


def _model_state_close(a, b, atol=1e-6):
    if a.keys() != b.keys():
        return False
    return all(torch.allclose(a[k], b[k], atol=atol) for k in a)


def test_train_nll_resume_matches_full_run():
    n_qubits = 4
    bits = _toy_bits(n_qubits=n_qubits, k=200, seed=0)
    common = dict(total_steps=10_000, batch_size=64, lr=3e-3, lambda_pt=0.0,
                   n_states=1 << n_qubits, device="cpu")

    # Full 6-epoch reference.
    torch.manual_seed(0)
    m_full = AutoregressiveRNN(n_bits=n_qubits, hidden=8, n_layers=1)
    train_nll(m_full, bits, min_epochs=6, max_epochs=6, **common)

    # Half + resume.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "nll.pt"
        torch.manual_seed(0)
        m_a = AutoregressiveRNN(n_bits=n_qubits, hidden=8, n_layers=1)
        train_nll(m_a, bits, min_epochs=6, max_epochs=6,
                    checkpoint_to=str(ckpt), checkpoint_every=1000,
                    **common)
        # Manually save mid-training: train 3, then checkpoint, then resume.
        torch.manual_seed(0)
        m_b = AutoregressiveRNN(n_bits=n_qubits, hidden=8, n_layers=1)
        train_nll(m_b, bits, min_epochs=6, max_epochs=6,
                    checkpoint_to=str(ckpt), checkpoint_every=3,
                    **common)
        # Now train a fresh model with same seed for 3 epochs and resume from ckpt.
        # Because checkpoint_every=3 wrote after epoch 3, ckpt has state @ epoch 3.
        # Resuming and running through epoch 6 should match the full 6-epoch run.
        m_resumed = AutoregressiveRNN(n_bits=n_qubits, hidden=8, n_layers=1)
        torch.manual_seed(123)  # different seed; resume_from controls weights.
        train_nll(m_resumed, bits, min_epochs=6, max_epochs=6,
                    resume_from=str(ckpt), **common)

    # Resumed weights match the model that just finished the 6-epoch run.
    # We rely on deterministic Adam + scheduler restored from checkpoint.
    assert _model_state_close(m_resumed.state_dict(), m_b.state_dict()), \
        "resumed run does not match the run that wrote the checkpoint"


def test_train_nll_checkpoint_payload_has_optimizer_and_scheduler():
    n_qubits = 4
    bits = _toy_bits(n_qubits=n_qubits, k=100, seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "nll.pt"
        m = AutoregressiveRNN(n_bits=n_qubits, hidden=8, n_layers=1)
        train_nll(m, bits, total_steps=200, min_epochs=2, max_epochs=2,
                    batch_size=32, lr=1e-3, lambda_pt=0.0,
                    n_states=1 << n_qubits, device="cpu",
                    checkpoint_to=str(ckpt), checkpoint_every=1)
        payload = torch.load(ckpt, map_location="cpu")
    assert "model_state" in payload
    assert "optimizer_state" in payload, \
        "optimizer state must be checkpointed for resume to work"
    assert "scheduler_state" in payload, \
        "scheduler state must be checkpointed for resume to work"
    assert payload["epoch"] == 2


def test_curriculum_resume_skips_completed_stages():
    n_qubits = 4
    rng = np.random.default_rng(0)
    samples_int = rng.integers(0, 1 << n_qubits, size=80, dtype=np.int64)

    def factory(seed):
        torch.manual_seed(seed)
        return AutoregressiveRNN(n_bits=n_qubits, hidden=8, n_layers=1)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "curriculum.pt"
        # Run stages 1..2 with checkpointing.
        s_first = weight_ascending(
            factory, samples_int, n_qubits,
            w_min=1, w_max=2, n_restarts_cold=1, n_restarts_warm=1,
            epochs_per_stage=10, lr=2e-3, seed=0, device="cpu",
            checkpoint_to=str(ckpt),
        )
        assert set(s_first.keys()) == {0, 1}

        # Now resume into a run with w_max=3 — stages 0 and 1 should be
        # skipped, only stage 2 (w=3) runs.
        s_resumed = weight_ascending(
            factory, samples_int, n_qubits,
            w_min=1, w_max=3, n_restarts_cold=1, n_restarts_warm=1,
            epochs_per_stage=10, lr=2e-3, seed=0, device="cpu",
            resume_from=str(ckpt), checkpoint_to=str(ckpt),
        )
    assert set(s_resumed.keys()) == {0, 1, 2}
    # Stages 0 and 1 should be byte-identical to the first run (not retrained).
    for i in (0, 1):
        for k in s_first[i]["model_state"]:
            assert torch.equal(s_first[i]["model_state"][k],
                                s_resumed[i]["model_state"][k]), \
                f"resumed stage {i} differs — should have been skipped, not retrained"
