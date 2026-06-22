"""Overlay F_cl vs k for the d_P sweep — one curve per PQC depth."""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args()
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")
    in_dir = base / f"m_peaked_dPsweep_n{args.n}"
    out = base / f"peaked_dPsweep_overlay_n{args.n}.png"

    by_dP_k = defaultdict(lambda: defaultdict(list))
    peak_by_dP = {}
    d_R_seen = None
    for p in sorted(in_dir.glob("*.json")):
        c = json.loads(p.read_text())
        dP = int(c["d_P"]); k = int(c["k_train"])
        by_dP_k[dP][k].append(c)
        peak_by_dP[dP] = c.get("peak_prob", float("nan"))
        d_R_seen = c.get("d_R", d_R_seen)

    dPs = sorted(by_dP_k)
    colors = [cm.viridis(i / max(1, len(dPs) - 1)) for i in range(len(dPs))]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for dP, col in zip(dPs, colors):
        ks = sorted(by_dP_k[dP])
        med, lo, hi = [], [], []
        for k in ks:
            vals = [c["F_cl"] for c in by_dP_k[dP][k]]
            m, q1, q3 = median_iqr(vals)
            med.append(m); lo.append(q1); hi.append(q3)
        peak = peak_by_dP[dP]
        label = fr"$d_P={dP}$" + (f"  (peak={peak:.3f})" if peak == peak else "")
        ax.plot(ks, med, "o-", color=col, lw=1.8, ms=4, label=label)
        ax.fill_between(ks, lo, hi, color=col, alpha=0.18)

    # Overlay Hubbard NLL k-sweep (only at n=8 for now)
    hub_dir = base / "m_hubbard_nll_k_sweep"
    hub_files = sorted(hub_dir.glob("k*_s*.json"))
    if args.n == 8 and hub_files:
        by_k = defaultdict(list)
        for p in hub_files:
            c = json.loads(p.read_text())
            by_k[int(c["k_train"])].append(c["classical_fidelity"])
        ks = sorted(by_k)
        med, lo, hi = [], [], []
        for k in ks:
            m, q1, q3 = median_iqr(by_k[k])
            med.append(m); lo.append(q1); hi.append(q3)
        ax.plot(ks, med, "*-", color="black", lw=1.8, ms=12,
                label="Hubbard L=4", mfc="gold", mew=1.2)
        ax.fill_between(ks, lo, hi, color="black", alpha=0.10)

    ax.set_xscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"$F_{\rm cl}(p_\theta, p_C)$")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Classical Fidelity — vary PQC depth at fixed RQC depth "
                 fr"($n={args.n}$, $d_R={d_R_seen}$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8, title="PQC depth")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
