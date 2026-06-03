"""Stochastic Reconfiguration (SR) for an autoregressive bitstring wavefunction
on a sector-restricted Hamiltonian.

Algorithm (one SR step):
  1. Draw a batch x_1..x_B ~ |psi_theta|^2 autoregressively (no MCMC -- AR + mask
     gives iid samples).
  2. E_loc(x_i) via `local_energy_hubbard` (detached).
  3. Per-sample score O_i = grad_theta log|psi(x_i)| via a backward loop.
  4. Center O and DeltaE.
  5. Form S and F. Use DUAL form (T = O_c O_c^T) when P > 2B (the common
     regime: ~10k params vs B=256).
  6. Solve (S + lambda I) dtheta = F (or dual variant), clip, theta -= lr * dtheta.
  7. lambda_{t+1} = max(lambda_min, lambda_0 * lambda_decay**t).

References: Zhang & Di Ventra `transformer_quantum_state/SR.py` (algorithm),
Sorella PRL 80, 4558 (1998) (the original SR).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class SRConfig:
    lr: float = 0.03
    lambda_0: float = 1e-2
    lambda_min: float = 1e-4
    lambda_decay: float = 0.99
    batch_size: int = 256
    clip_norm: float = 1.0


@dataclass
class SRState:
    step: int = 0
    history: list = field(default_factory=list)


def _flatten_grads(params):
    return torch.cat([p.grad.detach().flatten() for p in params])


def sr_step(model, ctx, sr_state, cfg, local_energy_fn, device="cpu"):
    """One SR update, mutating model parameters in-place. Returns metrics dict."""
    params = [p for p in model.parameters() if p.requires_grad]
    P_total = sum(p.numel() for p in params)
    B = cfg.batch_size

    # 1. Sample.
    x_bits = model.sample(B)                                              # (B, 2L) long

    # 2. Local energies (detached).
    E_vec = local_energy_fn(model, x_bits, ctx, device=device)
    E_mean = float(E_vec.mean().item())
    E_var = float(E_vec.var().item())

    # 3. Per-sample score O = grad log|psi|.
    log_psi = model.log_psi_mag(x_bits)                                   # (B,) with grad
    O = torch.zeros(B, P_total, device=device)
    for i in range(B):
        model.zero_grad(set_to_none=False)
        log_psi[i].backward(retain_graph=(i < B - 1))
        O[i] = _flatten_grads(params)
    model.zero_grad(set_to_none=False)

    # 4. Center.
    O_mean = O.mean(dim=0, keepdim=True)
    O_c = O - O_mean
    dE = E_vec.to(device).float() - E_mean

    lam = max(cfg.lambda_min, cfg.lambda_0 * (cfg.lambda_decay ** sr_state.step))

    # 5/6. Dense vs dual SR solve.
    if P_total <= 2 * B:
        S = (O_c.T @ O_c) / B + lam * torch.eye(P_total, device=device)
        F = (O_c.T @ dE) / B
        dtheta = torch.linalg.solve(S, F)
    else:
        T = (O_c @ O_c.T) / B + lam * torch.eye(B, device=device)
        y = torch.linalg.solve(T, dE / B)
        dtheta = O_c.T @ y

    # 7. Clip + update.
    norm_dtheta = float(dtheta.norm().item())
    if norm_dtheta > cfg.clip_norm:
        dtheta = dtheta * (cfg.clip_norm / norm_dtheta)

    with torch.no_grad():
        offset = 0
        for p in params:
            n = p.numel()
            p.add_(dtheta[offset:offset + n].view_as(p), alpha=-cfg.lr)
            offset += n

    sr_state.step += 1
    metrics = {
        "step": sr_state.step,
        "E_mean": E_mean,
        "E_var": E_var,
        "lambda": lam,
        "norm_dtheta": norm_dtheta,
    }
    sr_state.history.append(metrics)
    return metrics
