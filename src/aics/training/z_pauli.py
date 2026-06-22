"""Z-observable training: fit the model so <Z_S>_θ matches empirical
<Z_S> targets on a chosen support set.

  L(θ) = Σ_S  α_S · ( <Z_S>_θ − <Z_S>_empirical )^2

with <Z_S>_θ computed via a full-distribution forward over 2^n bitstrings
(so this scales to n ≲ 20 only). Uses the same `AutoregressiveRNN` as
NLL training — the legacy `BitstringARRNN` is retired.

The PT regulariser is intentionally NOT exposed here: it's a constraint
on the model distribution motivated by NLL training; under Z-observable
loss the data only constrains low-weight moments and the PT pressure
would fight the Z-loss for capacity. See the discussion in the cleanup
design notes if you need to argue otherwise.
"""
import numpy as np
import torch

from ..eval.z_observables import (
    empirical_z_expectations, parity_matrix, enumerate_z_supports,
)
from ..io.conventions import int_to_bits


def train_z_pauli(model, samples_int, supports, weights, n, *,
                    alpha=None, epochs=400, lr=2e-3, device=None,
                    verbose=False, log_every=80, logger=None,
                    stage_label="z_pauli"):
    """Fit `model` so <Z_S>_θ matches the empirical targets on `supports`.

    Parameters
    ----------
    model : AutoregressiveRNN
    samples_int : (k,) ints (MSB-first) — bitstrings from p_C
    supports, weights : output of enumerate_z_supports(n, max_weight)
    alpha : optional (n_obs,) per-Pauli weight; None ⇒ uniform.
    epochs : training epochs.
    lr : Adam learning rate (cosine-annealed).
    device : torch device; defaults to model's device.

    Returns final loss (float). Use `logger=JsonLogger(...)` to write
    per-epoch JSON for learning-curve plots.
    """
    device = device or next(model.parameters()).device
    targets_np = empirical_z_expectations(samples_int, supports, n)
    W_np = parity_matrix(supports, n)
    W = torch.from_numpy(W_np).to(torch.float64).to(device)
    targets = torch.from_numpy(targets_np).to(torch.float64).to(device)
    if alpha is None:
        alpha = np.ones(len(supports), dtype=np.float64)
    alpha_t = torch.from_numpy(np.asarray(alpha, dtype=np.float64)).to(device)

    dim = 1 << n
    all_int = np.arange(dim, dtype=np.int64)
    all_bits_t = torch.from_numpy(int_to_bits(all_int, n)).float().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr / 100)
    final_loss = float("nan")
    for ep in range(epochs):
        logp = model.log_prob(all_bits_t).to(torch.float64)
        p = torch.softmax(logp, dim=0)
        exps = W @ p
        loss = (alpha_t * (exps - targets).pow(2)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        final_loss = float(loss)
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  ep {ep:>4}: loss={final_loss:.4e}", flush=True)
        if logger is not None:
            logger.log(stage=stage_label, epoch=ep, n_epochs=epochs,
                        z_pauli_loss=final_loss,
                        max_weight=int(weights.max() if len(weights) > 0 else 0))
    return final_loss


__all__ = ["train_z_pauli", "enumerate_z_supports"]
