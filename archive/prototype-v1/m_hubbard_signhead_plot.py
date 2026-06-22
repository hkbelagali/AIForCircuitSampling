"""2x3 plot: rows = sign-head on / off, cols = (rel_E_err, fidelity, TV).
x-axis = k_train (log), one line per w_max."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def load_all(in_dir):
    cells = [json.loads(p.read_text()) for p in sorted(in_dir.glob("*.json"))]
    by_signw = defaultdict(lambda: defaultdict(list))  # (sign, w) -> k -> [cells]
    for c in cells:
        by_signw[(bool(c["use_sign_head"]), int(c["w_max"]))][int(c["k_train"])].append(c)
    return by_signw


def median_iqr(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def plot_metric(ax, by_signw, sign, key, ws, colors, transform=None, ylabel=""):
    for w, col in zip(ws, colors):
        sub = by_signw[(sign, w)]
        ks = sorted(sub)
        if not ks: continue
        med, lo, hi = [], [], []
        for k in ks:
            vals = [c[key] for c in sub[k]]
            if transform is not None:
                vals = [transform(v) for v in vals]
            m, q1, q3 = median_iqr(vals)
            med.append(m); lo.append(q1); hi.append(q3)
        ax.plot(ks, med, "o-", color=col, lw=1.8, ms=4,
                label=fr"$w\leq{w}$")
        ax.fill_between(ks, lo, hi, color=col, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")


def main():
    in_dir = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results/m_hubbard_signhead")
    out = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results/m_hubbard_signhead_compare.png")
    by_signw = load_all(in_dir)
    if not by_signw:
        raise SystemExit(f"no cells in {in_dir}")

    ws = sorted({w for (_, w) in by_signw})
    colors = [cm.viridis(i / max(1, len(ws) - 1)) for i in range(len(ws))]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for row, sign in enumerate([True, False]):
        title_prefix = "with sign head" if sign else "without sign head"
        fid_label = ("quantum fidelity $|\\langle\\psi_\\theta|\\psi_0\\rangle|^2$"
                      if sign else "classical fidelity $F_{\\rm cl}$")
        plot_metric(axes[row, 0], by_signw, sign, "rel_E_err", ws, colors,
                     transform=abs, ylabel=r"$|E_\theta - E_0|\,/\,|E_0|$")
        plot_metric(axes[row, 1], by_signw, sign, "fidelity", ws, colors,
                     ylabel=fid_label)
        plot_metric(axes[row, 2], by_signw, sign, "TV", ws, colors,
                     ylabel="TV$(p_\\theta, p_0)$")
        axes[row, 0].set_yscale("log")
        axes[row, 0].set_title(f"({chr(ord('a') + 3*row)}) Energy error · {title_prefix}")
        axes[row, 1].set_title(f"({chr(ord('a') + 3*row + 1)}) Fidelity · {title_prefix}")
        axes[row, 2].set_title(f"({chr(ord('a') + 3*row + 2)}) TV distance · {title_prefix}")
        axes[row, 1].set_ylim(0, 1.02)
    axes[0, 0].legend(loc="best", fontsize=9, title=r"$w_{\rm train}$")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
