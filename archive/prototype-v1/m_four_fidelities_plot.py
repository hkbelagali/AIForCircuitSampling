"""4-panel fidelity figure at n=8.
  Top row    — Quantum fidelity |<ψ_θ|ψ_target>|²:  (a) Hubbard  (b) RCS
  Bottom row — Classical fidelity F_cl(p_θ, p_C):    (c) Hubbard  (d) RCS
Lines per w_train; sharey across all four."""

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
    for p in sorted(in_dir.glob(glob_pat)):
        c = json.loads(p.read_text())
        by_wk[int(c[w_field])][int(c[k_field])].append(c)
    return by_wk


def plot_panel(ax, by_wk, ws, colors, key, title):
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
    ax.set_xscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which="both")


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")

    # Hubbard signed (16 restarts) — prefer this dir, fallback to original
    hub_signed_dir = base / "m_hubbard_signhead_16r"
    if not hub_signed_dir.exists() or not list(hub_signed_dir.glob("*_sign1.json")):
        hub_signed_dir = base / "m_hubbard_signhead"
    hub_signed = load_grouped(hub_signed_dir, "w_max", "k_train", "*_sign1.json")

    # Hubbard unsigned (existing)
    hub_unsigned = load_grouped(base / "m_hubbard_signhead", "w_max", "k_train",
                                  "*_sign0.json")

    # RCS signed: curriculum (w-warm-start) wherever available, fall back
    # to original m_rcs_signhead_v2 for cells curriculum hasn't filled in.
    rcs_signed = load_grouped(base / "m_rcs_signhead_v2", "w_max", "k_train")
    rcs_curr = load_grouped(base / "m_rcs_curriculum", "w_max", "k_train")
    for w, ks in rcs_curr.items():
        for k, cells in ks.items():
            rcs_signed[w][k] = cells  # override

    # RCS unsigned (existing Z-Pauli)
    rcs_unsigned = load_grouped(base / "m_rcs_z_pauli_k_sweep", "w_train", "k_train")

    ws = sorted(set(hub_signed) | set(hub_unsigned) | set(rcs_signed) | set(rcs_unsigned))
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
    fig.tight_layout()

    out = base / "four_fidelities_8q.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
