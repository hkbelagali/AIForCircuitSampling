"""Overlay biased (sample_chaotic) and unbiased (sample_exact_tn) Stage B
results. Pairs cells by (n_qubits, k_train, hidden, model_seed); plots
normalized XEB side-by-side and a delta heatmap.

Defaults match the current layout:
  --biased_dir   results/m_rcs_nll_tn_eval
  --unbiased_dir results/m_rcs_nll_tn_eval_pcz_aics
"""
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
    """Median normalized XEB over model_seed per (n_qubits, k_train, hidden)."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["n"], r["k_train"], r["hidden"])].append(r["xeb_norm"])
    return {k: float(np.nanmedian(v)) for k, v in grouped.items()}


def plot_compare(biased_dir, unbiased_dir, out_path,
                  train_labels=("n²", "n²log n", "n³", "n³log n", "10k", "100k")):
    rows_b = _gather(biased_dir)
    rows_u = _gather(unbiased_dir)
    if not rows_u:
        print(f"[plot] no unbiased results under {unbiased_dir} — nothing to compare")
        return
    agg_b = _aggregate(rows_b)
    agg_u = _aggregate(rows_u)

    n_qubits_values = sorted({nq for (nq, _, _) in agg_u})
    hiddens = sorted({h for (_, _, h) in agg_u})
    train_labels = list(train_labels)

    for hidden in hiddens:
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(train_labels)))

        # Left + middle: lineplots, biased then unbiased.
        for ax, agg, title in ((axes[0], agg_b, "chaotic (biased)"),
                                 (axes[1], agg_u, "exact_tn (unbiased)")):
            for label, col in zip(train_labels, colors):
                xs, ys = [], []
                for nq in n_qubits_values:
                    for k in sorted({k for (n, k, h) in agg if n == nq and h == hidden}):
                        if _poly_label(k, nq) == label:
                            v = agg.get((nq, k, hidden))
                            if v is not None and not np.isnan(v):
                                xs.append(nq); ys.append(v)
                            break
                if xs:
                    ax.plot(xs, ys, "o-", color=col, lw=2, ms=7, label=label)
            ax.axhline(1.0, color="green", ls="--", lw=1.0)
            ax.axhline(0.0, color="grey", ls="--", lw=1.0)
            ax.set_xlabel("Number of qubits")
            ax.set_ylabel("Normalized XEB")
            ax.set_title(f"{title} (hidden = {hidden})")
            ax.legend(fontsize=8, title=r"$N_{\rm train}$")
            ax.grid(alpha=0.3)
            ax.set_xticks(n_qubits_values)

        # Right: delta heatmap (unbiased − biased).
        delta = np.full((len(train_labels), len(n_qubits_values)), np.nan)
        for j, nq in enumerate(n_qubits_values):
            for i, label in enumerate(train_labels):
                # find the k matching this label
                for k in sorted({k for (n, k, h) in agg_u if n == nq and h == hidden}):
                    if _poly_label(k, nq) == label:
                        v_u = agg_u.get((nq, k, hidden))
                        v_b = agg_b.get((nq, k, hidden))
                        if v_u is not None and v_b is not None:
                            delta[i, j] = v_u - v_b
                        break
        vmax = max(0.1, np.nanmax(np.abs(delta))) if not np.all(np.isnan(delta)) else 0.1
        im = axes[2].imshow(delta, aspect="auto", cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax, origin="upper")
        axes[2].set_xticks(range(len(n_qubits_values)))
        axes[2].set_xticklabels(n_qubits_values)
        axes[2].set_yticks(range(len(train_labels)))
        axes[2].set_yticklabels(train_labels)
        axes[2].set_xlabel("n_qubits"); axes[2].set_ylabel(r"$N_{\rm train}$")
        axes[2].set_title("Δ(xeb_norm)  =  unbiased − chaotic")
        plt.colorbar(im, ax=axes[2])
        for i in range(len(train_labels)):
            for j in range(len(n_qubits_values)):
                v = delta[i, j]
                if not np.isnan(v):
                    axes[2].text(j, i, f"{v:+.2f}", ha="center", va="center",
                                  fontsize=7.5, color="black")

        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        full = out_path.parent / f"{out_path.stem}_h{hidden}{out_path.suffix}"
        plt.tight_layout()
        plt.savefig(full, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] wrote {full}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--biased_dir", default="results/m_rcs_nll_tn_eval")
    p.add_argument("--unbiased_dir", default="results/m_rcs_nll_tn_eval_pcz_aics")
    p.add_argument("--out", default="plots/chaotic_vs_pcz.png")
    args = p.parse_args()
    plot_compare(args.biased_dir, args.unbiased_dir, args.out)


if __name__ == "__main__":
    main()
