"""Companion to plot_sycamore_bars.py: compare clean_xeb on ALL 5k held vs
on the subset NOT present in the training pool. Shows how much of the reported
denoising / clean fit comes from train/held bitstring overlap.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json",
                     default="results/maxn_run/train_sycamore/overlap_eval.json")
    ap.add_argument("--out", default="plots/sycamore_overlap.png")
    args = ap.parse_args()

    rows = json.loads(Path(args.in_json).read_text())

    def series(kind, field):
        pts = [(r["k_train"], r[field]) for r in rows if r["kind"] == kind]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for ax, kind, title in zip(axes, ("tn", "exp"),
                                 ("Clean training (ideal TN samples)",
                                  "Noisy training (Sycamore device samples)")):
        xs_all, ys_all = series(kind, "clean_xeb_all")
        xs_f,   ys_f   = series(kind, "clean_xeb_filtered")

        ax.plot(xs_all, ys_all, "o-", lw=2.2, ms=8, color="#1f77b4",
                 label="all 5k held")
        ax.plot(xs_f, ys_f, "s--", lw=2.2, ms=8, color="#d62728",
                 label=r"held $\setminus$ training")

        ax.axhline(1.0, color="green", ls="--", lw=1.0, alpha=0.5)
        ax.axhline(0.0, color="grey",  ls="-",  lw=0.8, alpha=0.4)
        ax.set_ylim(-1.1, 1.15)
        ax.set_xscale("log")
        ax.set_xlabel("k_train")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left", fontsize=10, frameon=False)

    axes[0].set_ylabel("clean_xeb")
    fig.suptitle("Sycamore n=20 depth-14 — clean_xeb with train/held overlap removed",
                  y=1.02, fontsize=13)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
