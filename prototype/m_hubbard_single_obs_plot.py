"""Plot: error in <IIZIZZII> vs k for shadow estimator and trained model.
True <O> = 0 by half-filling P-H symmetry."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def median_iqr(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")
    data = json.loads((base / "m_hubbard_single_obs.json").read_text())
    true_O = data["true_O"]
    ks = [int(k) for k in data["ks"]]

    shadow_med, shadow_lo, shadow_hi = [], [], []
    model_med, model_lo, model_hi = [], [], []
    for k in ks:
        s_errs = [abs(c["shadow_O"] - true_O) for c in data["cells"][str(k)]]
        m_errs = [abs(c["model_O"] - true_O) for c in data["cells"][str(k)]]
        m, q1, q3 = median_iqr(s_errs)
        shadow_med.append(m); shadow_lo.append(q1); shadow_hi.append(q3)
        m, q1, q3 = median_iqr(m_errs)
        model_med.append(m); model_lo.append(q1); model_hi.append(q3)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ks_arr = np.asarray(ks, dtype=float)

    ax.plot(ks, shadow_med, "o-", color="#1f77b4", lw=1.8, ms=5,
            label="Sampled estimator")
    ax.fill_between(ks, shadow_lo, shadow_hi, color="#1f77b4", alpha=0.18)

    ax.plot(ks, model_med, "s-", color="#d62728", lw=1.8, ms=5,
            label="AR-RNN")
    ax.fill_between(ks, model_lo, model_hi, color="#d62728", alpha=0.18)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"err $\langle IIZIZZII\rangle$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out = base / "m_hubbard_single_obs.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
