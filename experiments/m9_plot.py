"""Aggregate the M9 per-cell JSONs and produce the headline figures.

Reads every results/m9_cells/L{L}_k{k}_s{seed}.json, groups by (L, c) where
c = k / L^2, and reports median + IQR of "VMC steps to threshold" across the 8
seeds per cell.

Cells whose JSON marks `reached: false` are right-censored at MAX_VMC_STEPS;
the median + IQR are robust to this for as long as < 50% of seeds are
censored. Cells with >= 50% censoring are reported as "censored" markers
(open circles) rather than filled circles.

Two figures:
  m9_steps_vs_samples.png  -- x = k (or c), y = steps_to_threshold, log y,
                              one curve per L. Headline.
  m9_heatmap.png           -- (L, c) -> steps_to_threshold heatmap.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m9_cells"
OUT_DIR = CELLS_DIR.parent


EXCLUDE_L = {8}   # capacity-limited at d_hidden=32; reported separately, not in headline


def load_records():
    out = []
    for p in sorted(CELLS_DIR.glob("L*_k*_s*.json")):
        r = json.loads(p.read_text())
        if r["L"] in EXCLUDE_L:
            continue
        out.append(r)
    return out


def aggregate(records):
    """Group records by (L, k). Return dict (L, k) -> list of steps_to_threshold."""
    by_cell = defaultdict(list)
    max_steps = 0
    for r in records:
        L, k, s = r["L"], r["k"], r["seed"]
        max_steps = max(max_steps, r["max_steps"])
        if r["steps_to_threshold"] is None:
            by_cell[(L, k)].append(("censored", r["max_steps"]))
        else:
            by_cell[(L, k)].append(("reached", r["steps_to_threshold"]))
    return by_cell, max_steps


def cell_summary(values, max_steps):
    """Compute median, IQR, and censoring fraction for a (L, k) cell."""
    raw = np.array([v[1] for v in values], dtype=float)
    censored = sum(1 for v in values if v[0] == "censored")
    n = len(values)
    cens_frac = censored / max(n, 1)
    # Use right-censored sentinel = max_steps (conservative)
    raw_censored = raw.copy()
    return {
        "median": float(np.median(raw_censored)),
        "q25": float(np.quantile(raw_censored, 0.25)),
        "q75": float(np.quantile(raw_censored, 0.75)),
        "n": n,
        "cens_frac": cens_frac,
        "max_steps": max_steps,
    }


def main():
    records = load_records()
    if not records:
        print("No M9 cells found. Run experiments/m9_run_sweep.py first.")
        return

    by_cell, max_steps = aggregate(records)

    # Organize by L
    L_to_curves = defaultdict(list)
    for (L, k), vals in by_cell.items():
        c = k / (L * L) if L else 0
        summ = cell_summary(vals, max_steps)
        L_to_curves[L].append((k, c, summ))
    for L in L_to_curves:
        L_to_curves[L].sort(key=lambda x: x[0])

    # ---- Figure 1: steps vs samples (headline) -----------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    cmap = plt.get_cmap("viridis")
    L_list = sorted(L_to_curves.keys())
    for i, L in enumerate(L_list):
        ks = np.array([c[0] for c in L_to_curves[L]])
        meds = np.array([c[2]["median"] for c in L_to_curves[L]])
        q25 = np.array([c[2]["q25"] for c in L_to_curves[L]])
        q75 = np.array([c[2]["q75"] for c in L_to_curves[L]])
        cens = np.array([c[2]["cens_frac"] for c in L_to_curves[L]])
        color = cmap(i / max(1, len(L_list) - 1))
        # Plot solid line where < 50% censored, dashed where >= 50%
        ax.fill_between(ks, q25, q75, color=color, alpha=0.18)
        # Filled markers for trustworthy cells, open for heavily censored
        ok = cens < 0.5
        if ok.any():
            ax.plot(ks[ok], meds[ok], "o-", color=color, lw=1.8, ms=5,
                    label=f"$L={L}$  (n={2*L})")
        if (~ok).any():
            ax.plot(ks[~ok], meds[~ok], "o", color=color, ms=6,
                    markerfacecolor="white", markeredgewidth=1.5,
                    label=None)
    ax.set_xlabel("training samples $k$")
    ax.set_ylabel(r"VMC steps to $|\Delta E|/|E_0| \leq 0.01$")
    ax.set_yscale("log")
    ax.set_title("M9: data-pretrain + Adam-VMC -- steps to chemical accuracy")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out1 = OUT_DIR / "m9_steps_vs_samples.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Wrote {out1}", flush=True)

    # ---- Figure 2: same but vs c = k / L^2 (n-normalized) ------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, L in enumerate(L_list):
        cs = np.array([c[1] for c in L_to_curves[L]])
        meds = np.array([c[2]["median"] for c in L_to_curves[L]])
        q25 = np.array([c[2]["q25"] for c in L_to_curves[L]])
        q75 = np.array([c[2]["q75"] for c in L_to_curves[L]])
        cens = np.array([c[2]["cens_frac"] for c in L_to_curves[L]])
        color = cmap(i / max(1, len(L_list) - 1))
        ax.fill_between(cs, q25, q75, color=color, alpha=0.18)
        ok = cens < 0.5
        if ok.any():
            ax.plot(cs[ok], meds[ok], "o-", color=color, lw=1.8, ms=5,
                    label=f"$L={L}$  (n={2*L})")
        if (~ok).any():
            ax.plot(cs[~ok], meds[~ok], "o", color=color, ms=6,
                    markerfacecolor="white", markeredgewidth=1.5)
    ax.set_xlabel(r"$c = k / L^2$  (poly-in-$n$ sample budget)")
    ax.set_ylabel(r"VMC steps to $|\Delta E|/|E_0| \leq 0.01$")
    ax.set_yscale("log")
    ax.set_title(r"M9: same data normalized to $c = k/L^2$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out2 = OUT_DIR / "m9_steps_vs_c.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Wrote {out2}", flush=True)

    # ---- Figure 3: heatmap (L vs c) ----------------------------------------
    # build a (n_L, n_c) matrix
    all_c = sorted({c for L in L_list for (_, c, _) in L_to_curves[L]})
    M = np.full((len(L_list), len(all_c)), np.nan)
    C = np.zeros_like(M)
    for li, L in enumerate(L_list):
        cdict = {round(c, 4): summ for (_, c, summ) in L_to_curves[L]}
        for ci, c in enumerate(all_c):
            if round(c, 4) in cdict:
                M[li, ci] = cdict[round(c, 4)]["median"]
                C[li, ci] = cdict[round(c, 4)]["cens_frac"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(np.log10(M), aspect="auto", origin="lower",
                   extent=[all_c[0], all_c[-1], min(L_list) - 0.5, max(L_list) + 0.5],
                   cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}$(median steps to threshold)")
    ax.set_xlabel(r"$c = k / L^2$")
    ax.set_ylabel("$L$")
    ax.set_yticks(L_list)
    ax.set_title("M9: $\\log_{10}$(VMC steps to chemical accuracy) over $(L, c)$")
    fig.tight_layout()
    out3 = OUT_DIR / "m9_heatmap.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Wrote {out3}", flush=True)

    # ---- Summary table -----------------------------------------------------
    print("\n=== M9 summary: median (q25-q75) steps to threshold, by (L, k) ===")
    print(f"  {'L':>2} {'k':>4} {'c=k/L^2':>8} {'med':>6} {'q25':>6} {'q75':>6} "
          f"{'cens':>5} {'n':>3}")
    for L in L_list:
        for (k, c, summ) in L_to_curves[L]:
            print(f"  {L:>2} {k:>4} {c:>8.3f} {summ['median']:>6.0f} "
                  f"{summ['q25']:>6.0f} {summ['q75']:>6.0f} "
                  f"{summ['cens_frac']:>5.0%} {summ['n']:>3}")


if __name__ == "__main__":
    main()
