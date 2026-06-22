"""Overlay F_cl vs k for the 7 peaked+RCS interpolation curves.
Each curve is one d_append value."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def median_iqr(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")
    in_dir = base / f"m_peaked_plus_rcs_n{args.n}"
    out = base / f"peaked_plus_rcs_overlay_n{args.n}.png"

    by_da_k = defaultdict(lambda: defaultdict(list))
    peak_prob_by_da = {}
    for p in sorted(in_dir.glob("*.json")):
        c = json.loads(p.read_text())
        da = int(c["d_append"]); k = int(c["k_train"])
        by_da_k[da][k].append(c)
        peak_prob_by_da[da] = c.get("peak_prob", float("nan"))

    das = sorted(by_da_k)
    colors = [cm.viridis(i / max(1, len(das) - 1)) for i in range(len(das))]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for da, col in zip(das, colors):
        ks = sorted(by_da_k[da])
        med, lo, hi = [], [], []
        for k in ks:
            vals = [c["F_cl"] for c in by_da_k[da][k]]
            m, q1, q3 = median_iqr(vals)
            med.append(m); lo.append(q1); hi.append(q3)
        peak = peak_prob_by_da[da]
        label = fr"$d_\lambda={da}$" + (f"  (peak={peak:.2f})" if peak == peak else "")
        ax.plot(ks, med, "o-", color=col, lw=1.8, ms=4, label=label)
        ax.fill_between(ks, lo, hi, color=col, alpha=0.18)

    ax.set_xscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"$F_{\rm cl}(p_\theta, p_C)$")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Classical Fidelity — peaked + appended RCS layers (n={args.n})")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8, title=r"appended layers")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
