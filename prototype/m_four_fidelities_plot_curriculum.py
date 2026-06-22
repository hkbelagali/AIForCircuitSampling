"""Curriculum version of the 4-panel fidelity figure at n=8.
Reads from the new curriculum result dirs and writes to a separate file
(four_fidelities_8q_curriculum.png) so the original plot is untouched.
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


def load_grouped(in_dir, w_field, k_field, glob_pat="*.json"):
    by_wk = defaultdict(lambda: defaultdict(list))
    if not in_dir.exists():
        return by_wk
    for p in sorted(in_dir.glob(glob_pat)):
        c = json.loads(p.read_text())
        by_wk[int(c[w_field])][int(c[k_field])].append(c)
    return by_wk


def plot_panel(ax, by_wk, ws, colors, key, title):
    has_any = False
    for w, col in zip(ws, colors):
        ks = sorted(by_wk[w])
        if not ks:
            continue
        has_any = True
        med, lo, hi = [], [], []
        for k in ks:
            vals = [c[key] for c in by_wk[w][k]]
            m, q1, q3 = median_iqr(vals)
            med.append(m); lo.append(q1); hi.append(q3)
        ax.plot(ks, med, "o-", color=col, lw=1.8, ms=4,
                label=fr"$w_{{\rm train}}\leq{w}$")
        ax.fill_between(ks, lo, hi, color=col, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which="both")
    if not has_any:
        ax.text(0.5, 0.5, "(pending)", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")

    hub_signed = load_grouped(base / "m_hubbard_curriculum", "w_max", "k_train",
                               "*_sign1.json")
    hub_unsigned = load_grouped(base / "m_hubbard_curriculum", "w_max", "k_train",
                                 "*_sign0.json")
    rcs_signed = load_grouped(base / "m_rcs_curriculum", "w_max", "k_train")
    rcs_unsigned = load_grouped(base / "m_rcs_z_pauli_curriculum", "w_train", "k_train")

    ws = sorted(set(hub_signed) | set(hub_unsigned) | set(rcs_signed) | set(rcs_unsigned))
    if not ws:
        ws = [1, 2, 3, 4]
    colors = [cm.viridis(i / max(1, len(ws) - 1)) for i in range(len(ws))]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharey=True)

    plot_panel(axes[0, 0], hub_signed, ws, colors, "fidelity",
                "(a) N=8 Hubbard, quantum fidelity")
    plot_panel(axes[0, 1], rcs_signed, ws, colors, "fidelity",
                "(b) N=8 RCS, quantum fidelity")
    plot_panel(axes[1, 0], hub_unsigned, ws, colors, "fidelity",
                "(c) N=8 Hubbard, classical fidelity")
    plot_panel(axes[1, 1], rcs_unsigned, ws, colors, "F_cl",
                "(d) N=8 RCS, classical fidelity")

    axes[0, 0].set_ylabel("fidelity")
    axes[1, 0].set_ylabel("fidelity")
    axes[0, 0].legend(loc="best", fontsize=8, title=r"$w_{\rm train}$")
    axes[0, 1].legend(loc="best", fontsize=8, title=r"$w_{\rm train}$")
    fig.suptitle("Curriculum protocol (cold w=1 + warm w=2..)", y=0.995, fontsize=11)
    fig.tight_layout()

    out = base / "four_fidelities_8q_curriculum.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
