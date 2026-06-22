"""Normalised XEB heatmap from Stage B JSON results. One figure per hidden width."""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aics.io import load_result


def _poly_label(k, n_qubits):
    opts = {
        n_qubits * n_qubits: "n²",
        int(n_qubits * n_qubits * np.log2(n_qubits)): "n²log n",
        n_qubits ** 3: "n³",
        int(n_qubits ** 3 * np.log2(n_qubits)): "n³log n",
        10_000: "10k",
        100_000: "100k",
    }
    return opts.get(k, str(k))


def _gather(in_dir):
    rows = []
    for f in sorted(Path(in_dir).glob("*.json")):
        try:
            r = load_result(f)
        except Exception:
            continue
        if "xeb_norm" in r:
            rows.append(r)
    return rows


def _aggregate(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["n"], r["k_train"], r["hidden"])].append(r["xeb_norm"])
    return {k: float(np.nanmedian(v)) for k, v in grouped.items()}


def _plot_one(ax_line, ax_heat, agg, hidden, n_qubits_values, train_labels, title):
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(train_labels)))
    for label, col in zip(train_labels, colors):
        xs, ys = [], []
        for n_qubits in n_qubits_values:
            for k in sorted({k for (nn, k, h) in agg
                              if nn == n_qubits and h == hidden}):
                if _poly_label(k, n_qubits) == label:
                    v = agg.get((n_qubits, k, hidden))
                    if v is not None and not np.isnan(v):
                        xs.append(n_qubits); ys.append(v)
                    break
        if xs:
            ax_line.plot(xs, ys, "o-", color=col, lw=2, ms=7, label=label)
    ax_line.axhline(1.0, color="green", ls="--", lw=1.0)
    ax_line.axhline(0.0, color="grey", ls="--", lw=1.0)
    ax_line.set_xlabel("Number of qubits")
    ax_line.set_ylabel("Normalized XEB")
    ax_line.set_title(title)
    ax_line.legend(fontsize=8, title=r"$N_{\rm train}$")
    ax_line.grid(alpha=0.3)
    ax_line.set_xticks(n_qubits_values)

    heat = np.full((len(train_labels), len(n_qubits_values)), np.nan)
    for j, n_qubits in enumerate(n_qubits_values):
        for i, label in enumerate(train_labels):
            for k in sorted({k for (nn, k, h) in agg
                              if nn == n_qubits and h == hidden}):
                if _poly_label(k, n_qubits) == label:
                    v = agg.get((n_qubits, k, hidden))
                    if v is not None and not np.isnan(v):
                        heat[i, j] = v
                    break
    im = ax_heat.imshow(heat, aspect="auto", cmap="RdYlGn",
                          vmin=0, vmax=1, origin="upper")
    ax_heat.set_xticks(range(len(n_qubits_values))); ax_heat.set_xticklabels(n_qubits_values)
    ax_heat.set_yticks(range(len(train_labels))); ax_heat.set_yticklabels(train_labels)
    ax_heat.set_xlabel("n_qubits"); ax_heat.set_ylabel(r"$N_{\rm train}$")
    plt.colorbar(im, ax=ax_heat, label="Normalized XEB")
    for i in range(len(train_labels)):
        for j in range(len(n_qubits_values)):
            v = heat[i, j]
            if not np.isnan(v):
                color = "black" if 0.2 < v < 0.8 else "white"
                ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center",
                              fontsize=7.5, color=color)


def plot_xeb_heatmaps(in_dir, out_dir,
                       train_labels=("n²", "n²log n", "n³", "n³log n", "10k", "100k")):
    rows = _gather(in_dir)
    if not rows:
        print(f"[plot] no result files under {in_dir}")
        return
    agg = _aggregate(rows)
    n_qubits_values = sorted({nq for (nq, _, _) in agg})
    hiddens = sorted({h for (_, _, h) in agg})
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for h in hiddens:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        _plot_one(axes[0], axes[1], agg, h, n_qubits_values, list(train_labels),
                    title=f"hidden = {h}")
        out_path = out_dir / f"rcs_nll_xeb_norm_h{h}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", type=str, default="results/m_rcs_nll_tn_eval")
    p.add_argument("--out_dir", type=str, default="plots")
    args = p.parse_args()
    plot_xeb_heatmaps(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
