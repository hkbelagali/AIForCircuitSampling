"""Per-cell diagnostic bundle + classifier.

The bundle summarises "why the model landed where it did" in three
metrics that downstream tooling (escalator, plots) can act on:

  nll_slope_last_20pct   train NLL slope over the final 20% of epochs.
                           <= 0 means "converged"; > threshold means
                           "training still improving when we stopped."
  held_gap               held_nll - H_pC_est. Distance from the ideal
                           held score achievable by p_C itself. Zero =
                           at ceiling.
  train_held_delta       train_nll - held_nll. Large *negative* value
                           (train much lower) signals overfitting.

`classify(result)` folds these three numbers plus a couple of the
built-in fields into one of six diagnostic tags used by the escalator:

  at_ceiling         model is essentially matching p_C on held.
  sample_limited     converged, gap large, no overfit — need more data.
  capacity_limited   converged, gap large, overfitting — need bigger model.
  epochs_limited     still improving when we stopped — need more epochs.
  borderline         partial success, no single axis dominates.
  broken             NaN, negative XEB, etc.

Thresholds are tunable; the shape of the tree is the point.
"""
from __future__ import annotations
from typing import Iterable

import numpy as np


def _finite(x, default=float("nan")):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def nll_slope(trajectory: Iterable[float], final_frac: float = 0.2) -> float:
    """Slope (per-epoch, in nats) over the last `final_frac` of the trajectory.

    Robust to short trajectories: returns 0 if fewer than 3 tail points.
    """
    arr = np.asarray(list(trajectory), dtype=np.float64)
    n = len(arr)
    if n < 3:
        return 0.0
    tail_n = max(3, int(np.ceil(n * final_frac)))
    tail = arr[-tail_n:]
    x = np.arange(len(tail), dtype=np.float64)
    # simple least-squares slope
    denom = ((x - x.mean()) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x.mean()) * (tail - tail.mean())).sum() / denom)


def compute_bundle(*, nll_trajectory=None, final_nll=None, held_nll=None,
                    held_pC=None, n_qubits=None) -> dict:
    """Assemble the diagnostic bundle from what's available.

    Missing inputs → NaN in the corresponding fields (not an error).
    """
    out = {}
    if nll_trajectory is not None:
        out["nll_slope_last_20pct"] = nll_slope(nll_trajectory, 0.2)
    else:
        out["nll_slope_last_20pct"] = float("nan")
    if held_nll is not None and held_pC is not None:
        H_pC_est = float(-np.log(np.maximum(np.asarray(held_pC), 1e-30)).mean())
        out["held_gap"] = float(held_nll) - H_pC_est
        out["H_pC_est"] = H_pC_est
    else:
        out["held_gap"] = float("nan"); out["H_pC_est"] = float("nan")
    if final_nll is not None and held_nll is not None:
        out["train_held_delta"] = float(final_nll) - float(held_nll)
    else:
        out["train_held_delta"] = float("nan")
    # train_gap: how far the model's training NLL is from H(p_C). If the
    # model is stuck near uniform (n·log 2), train_gap ~ n·log 2 − H(p_C),
    # which is large. This distinguishes "can't fit training data at all"
    # (capacity/epochs) from "fits training but doesn't generalize" (sample).
    if final_nll is not None and np.isfinite(out.get("H_pC_est", float("nan"))):
        out["train_gap"] = float(final_nll) - out["H_pC_est"]
    else:
        out["train_gap"] = float("nan")
    return out


def classify(result: dict, *, ceiling_gap=0.05, borderline_gap=0.5,
              slope_still_learning=-1e-3,
              overfit_delta=-0.3,
              underfit_train_gap=0.3) -> str:
    """Return one of six tags based on the diagnostic bundle in `result`.

    Reads either result["diagnostic"] (if present) or the result-level
    xeb_norm / final_nll / held_nll fields as fallback.
    """
    xeb_gen = _finite(result.get("xeb_gen"))
    xeb_norm = _finite(result.get("xeb_norm"))
    if not np.isfinite(xeb_gen) and not np.isfinite(xeb_norm):
        return "broken"
    if xeb_gen < -0.1 or (np.isfinite(xeb_norm) and xeb_norm < -0.5):
        return "broken"

    diag = result.get("diagnostic") or {}
    slope = _finite(diag.get("nll_slope_last_20pct"))
    held_gap = _finite(diag.get("held_gap"))
    train_gap = _finite(diag.get("train_gap"))
    delta = _finite(diag.get("train_held_delta"))

    # 1. At ceiling — held NLL matches H(p_C) within tolerance.
    if np.isfinite(held_gap) and held_gap <= ceiling_gap:
        return "at_ceiling"

    # 2. Still learning — training NLL trajectory hasn't leveled off.
    if np.isfinite(slope) and slope < slope_still_learning:
        return "epochs_limited"

    # 3. Model can't even fit the training data — capacity (or epochs, but
    #    the slope check above already caught that). Ordered before the
    #    sample-vs-overfit split because sample_limited only makes sense
    #    if the model is at least fitting the training set it has.
    if np.isfinite(train_gap) and train_gap > underfit_train_gap:
        return "capacity_limited"

    # 4. Fits training but not held — overfitting = capacity_limited.
    if np.isfinite(delta) and delta < overfit_delta:
        return "capacity_limited"

    # 5. Fits training AND doesn't overfit, but held is off — sample_limited.
    if np.isfinite(held_gap) and held_gap > borderline_gap:
        return "sample_limited"

    return "borderline"


ALL_TAGS = (
    "at_ceiling", "epochs_limited", "sample_limited",
    "capacity_limited", "borderline", "broken",
)
