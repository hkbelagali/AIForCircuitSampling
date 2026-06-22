"""Two-panel side-by-side, sharey:
  (left)  single weight-3 Z observable IIZIZZII error vs k (shadow + AR-RNN)
  (right) with-sign-head energy error vs k, lines per w_train
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def median_iqr(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")

    # Left panel data: single observable
    single = json.loads((base / "m_hubbard_single_obs.json").read_text())
    true_O = single["true_O"]
    ks_single = [int(k) for k in single["ks"]]
    shadow_med, shadow_lo, shadow_hi = [], [], []
    model_med, model_lo, model_hi = [], [], []
    for k in ks_single:
        s_errs = [abs(c["shadow_O"] - true_O) for c in single["cells"][str(k)]]
        m_errs = [abs(c["model_O"] - true_O) for c in single["cells"][str(k)]]
        m, q1, q3 = median_iqr(s_errs)
        shadow_med.append(m); shadow_lo.append(q1); shadow_hi.append(q3)
        m, q1, q3 = median_iqr(m_errs)
        model_med.append(m); model_lo.append(q1); model_hi.append(q3)

    # Right panel data: with-sign-head energy error
    hub_dir = base / "m_hubbard_signhead"
    by_wk = defaultdict(lambda: defaultdict(list))
    for p in sorted(hub_dir.glob("*_sign1.json")):
        c = json.loads(p.read_text())
        by_wk[int(c["w_max"])][int(c["k_train"])].append(c)
    ws = sorted(by_wk)
    colors = [cm.viridis(i / max(1, len(ws) - 1)) for i in range(len(ws))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Left
    ax = axes[0]
    ax.plot(ks_single, shadow_med, "o-", color="#1f77b4", lw=1.8, ms=5,
            label="Sampled estimator")
    ax.fill_between(ks_single, shadow_lo, shadow_hi, color="#1f77b4", alpha=0.18)
    ax.plot(ks_single, model_med, "s-", color="#d62728", lw=1.8, ms=5,
            label="AR-RNN")
    ax.fill_between(ks_single, model_lo, model_hi, color="#d62728", alpha=0.18)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"err $\langle IIZIZZII\rangle$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    # Right
    ax = axes[1]
    for w, col in zip(ws, colors):
        ks = sorted(by_wk[w])
        med, lo, hi = [], [], []
        for k in ks:
            vals = [abs(c["rel_E_err"]) for c in by_wk[w][k]]
            m, q1, q3 = median_iqr(vals)
            med.append(m); lo.append(q1); hi.append(q3)
        ax.plot(ks, med, "o-", color=col, lw=1.8, ms=4,
                label=fr"$w_{{\rm train}}\leq{w}$")
        ax.fill_between(ks, lo, hi, color=col, alpha=0.18)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"$|E_\theta - E_0|\,/\,|E_0|$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    axes[0].set_ylim(1e-3, 3)
    fig.tight_layout()
    out = base / "single_obs_vs_signhead.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
