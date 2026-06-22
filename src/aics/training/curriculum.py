"""Curriculum schedules for Z-observable training."""
import torch

from ..eval.z_observables import enumerate_z_supports
from .z_pauli import train_z_pauli


def weight_ascending(model_factory, samples_int, n_qubits, *,
                       w_min=1, w_max=4,
                       n_restarts_cold=4, n_restarts_warm=2,
                       epochs_per_stage=400, lr=2e-3, seed=0,
                       device=None, logger=None, verbose=False):
    """Warm-start across max_weight = w_min..w_max.

    Returns {stage_idx: {w, best_loss, n_restarts, model_state}}.
    The last stage's model_state is the one to evaluate downstream.
    """
    stages = {}
    warm_state = None
    for stage_idx, w in enumerate(range(w_min, w_max + 1)):
        supports, weights = enumerate_z_supports(n_qubits, max_weight=w)
        n_restarts = n_restarts_cold if warm_state is None else n_restarts_warm
        best_loss, best_state = float("inf"), None
        for r in range(n_restarts):
            init_seed = seed * 17 + r * 7919 + w * 1_000_003
            torch.manual_seed(init_seed)
            model = model_factory(seed=init_seed)
            if r == 0 and warm_state is not None:
                model.load_state_dict(warm_state)
            if device is not None:
                model = model.to(device)
            final = train_z_pauli(
                model, samples_int, supports, weights, n_qubits,
                epochs=epochs_per_stage, lr=lr, device=device,
                verbose=verbose, logger=logger,
                stage_label=f"z_pauli/w{w}/r{r}",
            )
            if final < best_loss:
                best_loss = final
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
        warm_state = best_state
        stages[stage_idx] = {"w": w, "best_loss": best_loss,
                              "n_restarts": n_restarts,
                              "model_state": best_state}
        if verbose:
            print(f"[curriculum] stage {stage_idx}: w={w}  best={best_loss:.4e}",
                  flush=True)
    return stages


SCHEDULES = {"weight_ascending": weight_ascending}
