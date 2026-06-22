"""Curriculum schedules for Z-observable training.

Currently one schedule:  weight_ascending — train at w=w_min cold (n_restarts_cold
random inits, keep best); then for each w in (w_min+1, ..., w_max) do
n_restarts_warm runs of which the first warm-starts from the previous
stage's best state and the rest are cold inits at that w.

Curriculum is loss-specific (only valid with `--loss z_pauli`); the CLI
in `scripts/train.py` errors if you pass it with `--loss nll`.
"""
from copy import deepcopy

import torch

from ..eval.z_observables import enumerate_z_supports
from .z_pauli import train_z_pauli


def weight_ascending(
    model_factory,
    samples_int,
    n,
    *,
    w_min=1,
    w_max=4,
    n_restarts_cold=4,
    n_restarts_warm=2,
    epochs_per_stage=400,
    lr=2e-3,
    seed=0,
    device=None,
    logger=None,
    verbose=False,
):
    """Warm-start curriculum across max-weight = w_min..w_max.

    `model_factory(seed)` produces a fresh model on the requested device.
    Returns a dict of {stage_index: {w, best_loss, model_state}}.

    The final stage's `model_state` is what you'd load into your model
    for downstream evaluation.
    """
    stages = {}
    warm_state = None
    for stage_idx, w in enumerate(range(w_min, w_max + 1)):
        supports, weights = enumerate_z_supports(n, max_weight=w)
        n_restarts = n_restarts_cold if warm_state is None else n_restarts_warm
        best_loss = float("inf")
        best_state = None
        for r_i in range(n_restarts):
            torch.manual_seed(seed * 17 + r_i * 7919 + w * 1_000_003)
            model = model_factory(seed=seed * 17 + r_i * 7919 + w * 1_000_003)
            if r_i == 0 and warm_state is not None:
                model.load_state_dict(warm_state)
            if device is not None:
                model = model.to(device)
            final = train_z_pauli(
                model, samples_int, supports, weights, n,
                epochs=epochs_per_stage, lr=lr, device=device,
                verbose=verbose, logger=logger,
                stage_label=f"z_pauli/w{w}/r{r_i}",
            )
            if final < best_loss:
                best_loss = final
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
        warm_state = best_state
        stages[stage_idx] = {
            "w": w, "best_loss": best_loss,
            "n_restarts": n_restarts, "model_state": best_state,
        }
        if verbose:
            print(f"[curriculum] stage {stage_idx}: w={w}  best_loss={best_loss:.4e}",
                  flush=True)
    return stages


# Registry — for CLI dispatch.
SCHEDULES = {
    "weight_ascending": weight_ascending,
}
