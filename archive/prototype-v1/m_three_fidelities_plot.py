"""Three-panel side-by-side: fidelity vs k for three 8-qubit systems.
  (a) Hubbard L=4 with sign head — quantum fidelity, lines per w_train
  (b) Hubbard L=4 without sign head — classical fidelity, lines per w_train
  (c) RCS n=8 — classical fidelity, single line (NLL training, no w_train axis)
Shared y-axis across all three.
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


def load_hubbard(in_dir, sign_flag):
    cells = []
    suffix = f"_sign{sign_flag}.json"
    for p in sorted(in_dir.glob(f"*{suffix}")):
        cells.append(json.loads(p.read_text()))
    by_wk = defaultdict(lambda: defaultdict(list))
    for c in cells:
        by_wk[int(c["w_max"])][int(c["k_train"])].append(c)
    return by_wk


def load_rcs_z_pauli(in_dir):
    cells = []
    for p in sorted(in_dir.glob("*.json")):
        cells.append(json.loads(p.read_text()))
    by_wk = defaultdict(lambda: defaultdict(list))
    for c in cells:
        by_wk[int(c["w_train"])][int(c["k_train"])].append(c)
    return by_wk


def plot_hubbard(ax, by_wk, ws, colors, key="fidelity"):
    for w, col in zip(ws, colors):
        ks = sorted(by_wk[w])
        if not ks: continue
        med, lo, hi = [], [], []
        for k in ks:
            vals = [c[key] for c in by_wk[w][k]]
            m, q1, q3 = median_iqr(vals)
            med.append(m); lo.append(q1); hi.append(q3)
        ax.plot(ks, med, "o-", color=col, lw=1.8, ms=4,
                label=fr"$w_{{\rm train}}\leq{w}$")
        ax.fill_between(ks, lo, hi, color=col, alpha=0.18)


def plot_rcs_zpauli(ax, by_wk, ws, colors):
    for w, col in zip(ws, colors):
        ks = sorted(by_wk[w])
        if not ks: continue
        med, lo, hi = [], [], []
        for k in ks:
            vals = [c["F_cl"] for c in by_wk[w][k]]
            m, q1, q3 = median_iqr(vals)
            med.append(m); lo.append(q1); hi.append(q3)
        ax.plot(ks, med, "o-", color=col, lw=1.8, ms=4,
                label=fr"$w_{{\rm train}}\leq{w}$")
        ax.fill_between(ks, lo, hi, color=col, alpha=0.18)


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")
    hub_dir = base / "m_hubbard_signhead"
    rcs_dir = base / "m_rcs_z_pauli_k_sweep"
    out = base / "three_fidelities_8q.png"

    signed = load_hubbard(hub_dir, 1)
    unsigned = load_hubbard(hub_dir, 0)
    rcs = load_rcs_z_pauli(rcs_dir)

    ws = sorted(set(signed) | set(unsigned) | set(rcs))
    colors = [cm.viridis(i / max(1, len(ws) - 1)) for i in range(len(ws))]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), sharey=True)

    plot_hubbard(axes[0], signed, ws, colors, key="fidelity")
    axes[0].set_title(r"(a) N=8 Hubbard, with sign head — quantum fidelity")
    axes[0].set_ylabel("fidelity")

    plot_hubbard(axes[1], unsigned, ws, colors, key="fidelity")
    axes[1].set_title(r"(b) N=8 Hubbard, without sign head — classical fidelity")

    plot_rcs_zpauli(axes[2], rcs, ws, colors)
    axes[2].set_title("(c) N=8 RCS — classical fidelity")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$k_{\rm train}$")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="best", fontsize=8, title=r"$w_{\rm train}$")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
