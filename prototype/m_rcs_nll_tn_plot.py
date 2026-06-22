"""Plot Ryan's headline figure from our TN-backed NLL sweep.

Replicates cell 17 of rcs_ml_experiment.ipynb (without titles): a 2-panel
figure showing normalized XEB vs n_qubits, with lines per N_train label
and a heatmap.

Normalized XEB = (xeb_gen - xeb_uniform) / (xeb_held - xeb_uniform)
                  0 = uniform, 1 = data ceiling

Aggregates over model_seed (median). One plot per hidden width.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def poly_label(n_train, n):
    opts = {
        n**2: 'n²',
        int(n**2 * np.log2(n)): 'n²log n',
        n**3: 'n³',
        int(n**3 * np.log2(n)): 'n³log n',
        10_000: '10k',
        100_000: '100k',
    }
    return opts.get(n_train, str(n_train))


def load_runs(in_dir):
    runs = []
    for f in sorted(in_dir.glob("*.json")):
        try:
            c = json.load(open(f))
        except Exception:
            continue
        runs.append(c)
    return runs


def normalize_xeb(runs):
    out = []
    for r in runs:
        denom = r["xeb_held_cache"] - r["xeb_uniform_cache"]
        xeb_norm = ((r["xeb_gen"] - r["xeb_uniform_cache"]) / denom
                    if abs(denom) > 1e-12 else np.nan)
        out.append({
            "n": r["n"], "k_train": r["k_train"], "hidden": r["hidden"],
            "model_seed": r["model_seed"],
            "xeb_norm": float(xeb_norm),
            "xeb_gen": r["xeb_gen"],
            "xeb_held": r["xeb_held_cache"],
            "xeb_unif": r["xeb_uniform_cache"],
        })
    return out


def aggregate_seed(rows):
    """Median over model_seed per (n, k_train, hidden)."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["n"], r["k_train"], r["hidden"])].append(r["xeb_norm"])
    return {k: float(np.nanmedian(v)) for k, v in grouped.items()}


def plot_one(agg, hidden, n_values, train_labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _plot_row(axes, agg, hidden, n_values, train_labels)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot_row(axes, agg, hidden, n_values, train_labels):
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(train_labels)))

    for label, col in zip(train_labels, colors):
        xs, ys = [], []
        for n in n_values:
            ks = sorted({k for (nn, k, h) in agg if nn == n and h == hidden})
            for k in ks:
                if poly_label(k, n) == label:
                    v = agg.get((n, k, hidden))
                    if v is not None and not np.isnan(v):
                        xs.append(n); ys.append(v)
                    break
        if xs:
            axes[0].plot(xs, ys, "o-", color=col, lw=2, ms=7, label=label)

    axes[0].axhline(1.0, color="green", ls="--", lw=1.2)
    axes[0].axhline(0.0, color="grey", ls="--", lw=1.2)
    axes[0].set_xlabel("Number of qubits (n)")
    axes[0].set_ylabel("Normalised XEB")
    axes[0].legend(fontsize=9, title=r"$N_{\rm train}$", loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(n_values)

    heatmap = np.full((len(train_labels), len(n_values)), np.nan)
    for j, n in enumerate(n_values):
        for i, label in enumerate(train_labels):
            ks = sorted({k for (nn, k, h) in agg if nn == n and h == hidden})
            for k in ks:
                if poly_label(k, n) == label:
                    v = agg.get((n, k, hidden))
                    if v is not None and not np.isnan(v):
                        heatmap[i, j] = v
                    break

    im = axes[1].imshow(heatmap, aspect="auto", cmap="RdYlGn",
                        vmin=0, vmax=1, origin="upper")
    axes[1].set_xticks(range(len(n_values)))
    axes[1].set_xticklabels(n_values)
    axes[1].set_yticks(range(len(train_labels)))
    axes[1].set_yticklabels(train_labels)
    axes[1].set_xlabel("Number of qubits (n)")
    axes[1].set_ylabel(r"$N_{\rm train}$")
    plt.colorbar(im, ax=axes[1], label="Normalised XEB")
    for i in range(len(train_labels)):
        for j in range(len(n_values)):
            v = heatmap[i, j]
            if not np.isnan(v):
                color = "black" if 0.2 < v < 0.8 else "white"
                axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                              fontsize=7.5, color=color)


def plot_stack(agg, hiddens, n_values, train_labels, out_path):
    n_rows = len(hiddens)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    for row, h in enumerate(hiddens):
        _plot_row(axes[row], agg, h, n_values, train_labels)
        # Row label on the far left
        axes[row, 0].text(-0.12, 0.5, f"hidden = {h}",
                          transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center",
                          fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")
    runs = load_runs(base / "m_rcs_nll_tn_eval")
    print(f"loaded {len(runs)} valid cells")
    rows = normalize_xeb(runs)
    agg = aggregate_seed(rows)

    n_values = sorted({n for (n, _, _) in agg})
    train_labels = ['n²', 'n²log n', 'n³', 'n³log n', '10k', '100k']
    hiddens = sorted({h for (_, _, h) in agg})
    print(f"n values: {n_values}")
    print(f"hidden widths: {hiddens}")

    for h in hiddens:
        out = base / f"rcs_nll_tn_xeb_norm_h{h}.png"
        plot_one(agg, h, n_values, train_labels, out)

    out_stacked = base / "rcs_nll_tn_xeb_norm_stack.png"
    plot_stack(agg, hiddens, n_values, train_labels, out_stacked)


if __name__ == "__main__":
    main()
