"""Overlay biased (sample_chaotic) vs unbiased (PCZ) Stage B NLL results.

Reads two parallel result trees:
  results/m_rcs_nll_tn_eval/         (sample_chaotic baseline)
  results/m_rcs_nll_tn_eval_pcz/     (pcz_sampler.sample_pcz_marginal)

For each (n, k_train, hidden) it plots two lines (chaotic vs pcz) so the
gap visually shows the impact of the chaotic-marginal bias on downstream
sample-complexity claims.

If the conclusion (e.g. "models fail to learn for k < 2^n") is invariant
to the bias, the two curves overlap. If they diverge, the bias is doing
real work and the writeup needs the unbiased numbers.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
            "model_seed": r["model_seed"], "xeb_norm": float(xeb_norm),
        })
    return out


def aggregate_seed(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["n"], r["k_train"], r["hidden"])].append(r["xeb_norm"])
    return {k: float(np.nanmedian(v)) for k, v in grouped.items()}


def poly_label(n_train, n):
    opts = {
        n**2: "n²",
        int(n**2 * np.log2(n)): "n²log n",
        n**3: "n³",
        int(n**3 * np.log2(n)): "n³log n",
        10_000: "10k",
        100_000: "100k",
    }
    return opts.get(n_train, str(n_train))


def plot_compare(agg_chaotic, agg_pcz, hidden, n_values, train_labels, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(train_labels)))

    for label, col in zip(train_labels, colors):
        xs_c, ys_c = [], []
        xs_p, ys_p = [], []
        for n in n_values:
            for k in sorted({k for (nn, k, h) in agg_chaotic if nn == n and h == hidden}):
                if poly_label(k, n) == label:
                    v = agg_chaotic.get((n, k, hidden))
                    if v is not None and not np.isnan(v):
                        xs_c.append(n); ys_c.append(v)
                    break
            for k in sorted({k for (nn, k, h) in agg_pcz if nn == n and h == hidden}):
                if poly_label(k, n) == label:
                    v = agg_pcz.get((n, k, hidden))
                    if v is not None and not np.isnan(v):
                        xs_p.append(n); ys_p.append(v)
                    break
        if xs_c:
            axes[0].plot(xs_c, ys_c, "o--", color=col, lw=1.5, ms=6,
                          alpha=0.6, label=f"{label} (chaotic)")
        if xs_p:
            axes[0].plot(xs_p, ys_p, "s-", color=col, lw=2.0, ms=7,
                          label=f"{label} (pcz)")

    axes[0].axhline(1.0, color="green", ls=":", lw=1.0)
    axes[0].axhline(0.0, color="grey", ls=":", lw=1.0)
    axes[0].set_xlabel("Number of qubits (n)")
    axes[0].set_ylabel("Normalised XEB")
    axes[0].set_title(f"hidden = {hidden}; dashed = chaotic, solid = pcz (unbiased)")
    axes[0].legend(fontsize=7, ncol=2, loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(n_values)

    # Right panel: delta = pcz - chaotic, heatmap
    delta = np.full((len(train_labels), len(n_values)), np.nan)
    for j, n in enumerate(n_values):
        for i, label in enumerate(train_labels):
            for k in sorted({k for (nn, k, h) in agg_chaotic if nn == n and h == hidden}):
                if poly_label(k, n) == label:
                    vc = agg_chaotic.get((n, k, hidden))
                    vp = agg_pcz.get((n, k, hidden))
                    if vc is not None and vp is not None \
                            and not (np.isnan(vc) or np.isnan(vp)):
                        delta[i, j] = vp - vc
                    break

    vmax = max(0.2, np.nanmax(np.abs(delta)) if not np.all(np.isnan(delta)) else 0.2)
    im = axes[1].imshow(delta, aspect="auto", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, origin="upper")
    axes[1].set_xticks(range(len(n_values)))
    axes[1].set_xticklabels(n_values)
    axes[1].set_yticks(range(len(train_labels)))
    axes[1].set_yticklabels(train_labels)
    axes[1].set_xlabel("Number of qubits (n)")
    axes[1].set_ylabel(r"$N_{\rm train}$")
    axes[1].set_title("Δ(xeb_norm) = pcz - chaotic  (red = pcz higher → chaotic was pessimistic)")
    plt.colorbar(im, ax=axes[1])
    for i in range(len(train_labels)):
        for j in range(len(n_values)):
            v = delta[i, j]
            if not np.isnan(v):
                axes[1].text(j, i, f"{v:+.2f}", ha="center", va="center",
                              fontsize=7, color="black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")
    runs_c = load_runs(base / "m_rcs_nll_tn_eval")
    runs_p = load_runs(base / "m_rcs_nll_tn_eval_pcz")
    print(f"chaotic runs: {len(runs_c)}; pcz runs: {len(runs_p)}")
    if not runs_p:
        print("no pcz runs yet — skipping plot")
        return
    agg_c = aggregate_seed(normalize_xeb(runs_c))
    agg_p = aggregate_seed(normalize_xeb(runs_p))

    n_values = sorted({n for (n, _, _) in agg_p})
    train_labels = ["n²", "n²log n", "n³", "n³log n", "10k", "100k"]
    hiddens = sorted({h for (_, _, h) in agg_p})
    print(f"n values: {n_values}; hiddens: {hiddens}")

    for h in hiddens:
        plot_compare(agg_c, agg_p, h, n_values, train_labels,
                      base / f"rcs_nll_pcz_vs_chaotic_h{h}.png")


if __name__ == "__main__":
    main()
