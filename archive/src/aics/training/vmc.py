"""VMC optimization with the Adam-style energy-gradient estimator.

For a sector-AR wavefunction sampled iid from |psi|^2, the gradient of the
energy <E> is

    d<E>/dtheta = 2 * E_{x ~ |psi|^2} [ (O(x) - <O>) (E_loc(x) - <E>) ]

where O(x) = d log|psi(x)| / dtheta. This expectation is the gradient of the
scalar loss

    L(theta) = mean_i [ log|psi(x_i)| * (E_loc(x_i) - <E>)_detached ]

so we can compute it with a SINGLE backward pass per VMC step. No per-sample
gradients. ~100-500x faster than the per-sample-backward SR step.

This is the pragmatic Moss et al. setup: Adam on the energy gradient, with the
sample batch redrawn from the autoregressive model each step.

We keep `src/aics/training/sr.py` available for reference but the drivers
default to this estimator.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from aics.models.ar_transformer import _HandAdam


@dataclass
class VMCConfig:
    lr: float = 5e-3
    batch_size: int = 256
    clip_norm: float = 1.0
    betas: tuple = (0.9, 0.999)


@dataclass
class VMCState:
    step: int = 0
    history: list = field(default_factory=list)
    optimizer: Any = None        # _HandAdam, lazily built on first sr/vmc_step


def vmc_step(model, ctx, vmc_state, cfg, local_energy_fn, device="cpu",
             signed=False):
    """One Adam-VMC update, mutating model parameters in-place. Returns metrics dict.

    If signed=True, the surrogate loss uses log|Psi(x)| = log|tanh(sign_logit(x))|
    + log_psi_mag(x), so gradients flow through the model's learned sign head.
    Requires model.learn_signs=True and a local_energy_fn that consumes
    model-predicted signs (e.g., local_energy_hubbard_signed).
    """
    if vmc_state.optimizer is None:
        params = [p for p in model.parameters() if p.requires_grad]
        vmc_state.optimizer = _HandAdam(params, lr=cfg.lr, betas=cfg.betas)
    opt = vmc_state.optimizer
    params = opt.params

    # 1. iid sample batch from |psi|^2 (no MCMC).
    B = cfg.batch_size
    x_bits = model.sample(B)                                  # (B, 2L) long

    # 2. Local energies (detached coefficients).
    with torch.no_grad():
        E_vec = local_energy_fn(model, x_bits, ctx, device=device).float()
    E_mean = float(E_vec.mean().item())
    E_var = float(E_vec.var().item())
    centered_E = (E_vec - E_mean).detach()

    # 3. log|psi| with grad on the same samples; one backward pass.
    if signed:
        # log|Psi| = log|tanh(sign_logit)| + log_psi_mag
        log_abs_s = torch.log(torch.abs(model.soft_sign(x_bits)) + 1e-12)
        log_psi = log_abs_s + model.log_psi_mag(x_bits)
    else:
        log_psi = model.log_psi_mag(x_bits)                   # (B,) with grad
    loss = (log_psi * centered_E).mean()                      # scalar; grads only on log_psi

    opt.zero_grad()
    loss.backward()

    # 4. Gradient clip (over the flat param vector for a stable norm).
    grad_norm = float(
        torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None))
    )
    if grad_norm > cfg.clip_norm:
        scale = cfg.clip_norm / grad_norm
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)

    opt.step()

    vmc_state.step += 1
    metrics = {
        "step": vmc_state.step,
        "E_mean": E_mean,
        "E_var": E_var,
        "grad_norm": grad_norm,
    }
    vmc_state.history.append(metrics)
    return metrics
