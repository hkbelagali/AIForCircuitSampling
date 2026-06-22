"""Side-by-side Z-Pauli extrapolation: Hubbard L=4 vs RCS n=8.

Both systems have n=8 qubits, k_train=2000. The shadow noise floor is
the same in both. The 'predict-zero' baseline differs hugely: Hubbard
ground-state Z correlators carry real structure (RMS ~0.5 at low weight),
RCS Porter-Thomas expectations are tiny (RMS ~0.06).

Story: in BOTH systems the AR-RNN fits trained weights down to shadow
noise and gives essentially zero information at unseen weights (model
error ≈ predict-zero). The CONSEQUENCE of that failure differs: in
Hubbard the lost signal is large, in RCS it is small.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


import matplotlib.cm as cm
COLORS = {w: cm.viridis((w - 1) / 7) for w in range(1, 9)}


def median_iqr(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def collect(in_dir):
    cells = [json.loads(p.read_text()) for p in sorted(in_dir.glob("*.json"))]
    err_model, err_shadow, rms_true = (
        defaultdict(lambda: defaultdict(list)),
        defaultdict(list),
        defaultdict(list),
    )
    n = cells[0]["n"]
    k = cells[0]["k_train"]
    for c in cells:
        w_train = c["w_train"]
        for w in range(1, n + 1):
            wk = str(w) if str(w) in c["err_by_weight_model"] else w
            if wk not in c["err_by_weight_model"]:
                continue
            err_model[w_train][w].append(c["err_by_weight_model"][wk])
            err_shadow[w].append(c["err_by_weight_shadow"][wk])
            rms_true[w].append(c["true_rms_by_weight"][wk])
    return n, k, err_model, err_shadow, rms_true


def plot_one(ax, in_dir, title, weights_to_plot=None,
             include_w1=True, show_predict_zero=True):
    n, k, err_model, err_shadow, rms_true = collect(in_dir)
    weights = weights_to_plot or list(range(1, n + 1))

    if show_predict_zero:
        zero_med = [median_iqr(rms_true[w])[0] for w in weights]
        ax.plot(weights, zero_med, color="grey", ls=":", lw=1.8, alpha=0.9,
                label=r"predict $0$")

    for w_train in sorted(err_model):
        if not include_w1 and w_train == 1:
            continue
        ys, lo, hi = [], [], []
        for w in weights:
            m, q1, q3 = median_iqr(err_model[w_train][w])
            ys.append(m); lo.append(q1); hi.append(q3)
        ax.plot(weights, ys, "o-", color=COLORS[w_train], lw=1.8,
                label=fr"model, $w\leq{w_train}$")
        ax.fill_between(weights, lo, hi, color=COLORS[w_train], alpha=0.18)
        if w_train < max(weights):
            ax.axvline(w_train + 0.5, color=COLORS[w_train],
                       lw=0.6, ls=":", alpha=0.35)

    ax.set_xlabel(r"evaluation weight $w$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xticks(weights)
    return n, k


def main():
    rcs_dir = Path(__file__).resolve().parents[1] / "results" / "m_rcs_z_pauli"
    hub_dir = Path(__file__).resolve().parents[1] / "results" / "m_hubbard_z_pauli"
    out_path = Path(__file__).resolve().parents[1] / "results" / "z_pauli_hubbard_vs_rcs.png"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # Top row: full view — include w=1 and predict-zero, shared y-axis
    plot_one(axes[0, 0], hub_dir, "N=8 Fermi-Hubbard",
             include_w1=True, show_predict_zero=True)
    plot_one(axes[0, 1], rcs_dir, "N=8 RCS",
             include_w1=True, show_predict_zero=True)
    for ax in axes[0, :]:
        ax.set_ylim(bottom=0)

    # Bottom row: zoom — drop w=1 and predict-zero, shared y-axis
    plot_one(axes[1, 0], hub_dir, r"N=8 Fermi-Hubbard (zoom, $w\geq 2$)",
             include_w1=False, show_predict_zero=False)
    plot_one(axes[1, 1], rcs_dir, r"N=8 RCS (zoom, $w\geq 2$)",
             include_w1=False, show_predict_zero=False)
    bot_ymax = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    for ax in axes[1, :]:
        ax.set_ylim(0, bot_ymax)

    for ax in axes[:, 0]:
        ax.set_ylabel("mean Z error")
    for ax in axes.flat:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
