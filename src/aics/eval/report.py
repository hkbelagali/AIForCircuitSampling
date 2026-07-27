"""eval.report — canonical eval dict for a trained model + sample bundle.

Used by scripts/train.py and by aics.train_cell. Keeps the result-dict
schema in one place so plots and JSON I/O always see the same fields.
"""
import numpy as np
import torch

from .xeb import normalized_xeb
from .nll import held_nll, normalized_nll, nll_excess
from .diagnostics import compute_bundle, classify


@torch.no_grad()
def report(model, n_qubits, *, held_bits=None, held_pC=None, uniform_pC=None,
            device=None, nll_trajectory=None, final_nll=None):
    """Score `model` against a sample bundle. Returns a dict of metrics.

    Missing inputs → missing keys (no error). Always at least returns {}.
    """
    device = device or next(model.parameters()).device
    D = 1 << n_qubits
    out = {}
    if held_bits is None:
        return out
    held_t = torch.from_numpy(np.asarray(held_bits, dtype=np.float32)).to(device)
    log_q_held = model.log_prob(held_t).cpu().numpy()
    q_held = np.exp(log_q_held)
    out["held_nll"] = held_nll(log_q_held)
    out["normalized_nll"] = normalized_nll(out["held_nll"], n_qubits)
    if held_pC is not None:
        out["nll_excess"] = nll_excess(out["held_nll"], held_pC)
        # KL(pC || q) estimated on held: E_{z~pC}[log pC(z)/q(z)]. Same
        # number as diagnostic.held_gap but named for citability. Negative
        # KL is possible in principle for finite-sample estimates but rare.
        H_pC = float(-np.log(np.maximum(np.asarray(held_pC), 1e-30)).mean())
        out["kl_divergence"] = float(out["held_nll"]) - H_pC
        out["xeb_gen"] = float(D * q_held.mean() - 1)
        out["xeb_held_cache"] = float(D * held_pC.mean() - 1)
        if uniform_pC is not None:
            out["xeb_uniform_cache"] = float(D * uniform_pC.mean() - 1)
            out["xeb_norm"] = normalized_xeb(
                out["xeb_gen"], out["xeb_uniform_cache"], out["xeb_held_cache"])
    # Diagnostic bundle + classifier tag
    bundle = compute_bundle(
        nll_trajectory=nll_trajectory, final_nll=final_nll,
        held_nll=out.get("held_nll"), held_pC=held_pC, n_qubits=n_qubits,
    )
    out["diagnostic"] = bundle
    out["diagnostic"]["tag"] = classify(out)
    return out
