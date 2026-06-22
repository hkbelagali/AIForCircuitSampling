"""Plot the k_train sweep: XEB / novel-mass / classical fidelity / TV
vs k_train, one curve per depth, error bars from seeds."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {4: "#4c72b0", 10: "#dd8452", 20: "#55a467"}


def collect(in_dir):
    cells = []
    for p in sorted(in_dir.glob("n*_d*_k*_s*.json")):
        cells.append(json.loads(p.read_text()))
    by_dk = defaultdict(list)  # (depth, k) -> list of cell dicts
    for c in cells:
        by_dk[(c["depth"], c["k_train"])].append(c)
    return by_dk


def median_iqr(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def plot_metric(ax, by_dk, depths, ks, key, transform=None, label_fn=None):
    for d in depths:
        ys, lo, hi = [], [], []
        ks_present = []
        for k in ks:
            vals = [c[key] for c in by_dk.get((d, k), [])]
            if not vals:
                continue
            if transform is not None:
                vals = [transform(v, c) for v, c in zip(vals, by_dk[(d, k)])]
            m, q1, q3 = median_iqr(vals)
            ys.append(m); lo.append(q1); hi.append(q3); ks_present.append(k)
        if ks_present:
            ax.plot(ks_present, ys, "o-", color=COLORS[d], lw=1.6,
                    label=label_fn(d) if label_fn else f"depth {d}")
            ax.fill_between(ks_present, lo, hi, color=COLORS[d], alpha=0.18)


def main():
    in_dir = Path(__file__).resolve().parents[1] / "results" / "m_rcs_sweep"
    out_path = Path(__file__).resolve().parents[1] / "results" / "m_rcs_sweep_summary.png"
    by_dk = collect(in_dir)
    if not by_dk:
        raise SystemExit(f"no cells found under {in_dir}")
    depths = sorted({d for (d, _) in by_dk})
    ks = sorted({k for (_, k) in by_dk})
    dim = next(iter(by_dk.values()))[0]["dim"]
    n = next(iter(by_dk.values()))[0]["n"]
    print(f"loaded {sum(len(v) for v in by_dk.values())} cells, "
          f"depths={depths}, ks={ks}, dim={dim}")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # (a) Linear XEB on model samples vs k
    ax = axes[0, 0]
    plot_metric(ax, by_dk, depths, ks, "candidate_xeb")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5,
               label=r"ideal $p_C$ ($F_{\rm XEB}=1$)")
    ax.axhline(0.0, color="grey", lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"linear XEB on model samples")
    ax.set_title("(a) Candidate XEB")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    # (b) Classical (Bhattacharyya) fidelity vs k
    ax = axes[0, 1]
    plot_metric(ax, by_dk, depths, ks, "classical_fidelity")
    ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5,
               label=r"$F_{\rm cl}=1$")
    ax.set_xscale("log"); ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"$F_{\rm cl}(p_\theta, p_C) = (\sum \sqrt{p_\theta p_C})^2$")
    ax.set_title("(b) Magnitude (Bhattacharyya) fidelity")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    # (c) Novel mass: prob model puts on unseen-in-training bitstrings
    ax = axes[1, 0]
    plot_metric(ax, by_dk, depths, ks, "novel_mass")
    ax.set_xscale("log"); ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"$P_{\rm model}(x \notin \mathrm{train})$")
    ax.set_title("(c) Probability mass outside training support")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)

    # (d) TV distance to p_C
    ax = axes[1, 1]
    plot_metric(ax, by_dk, depths, ks, "tv_distance")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"TV$(p_\theta, p_C)$")
    ax.set_title("(d) Total-variation distance")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"RCS+XEB sweep: n={n} ({dim} bitstrings), brickwork "
                 f"Sycamore-style; 3 seeds, median + IQR shading",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
