"""Aggregate the M13 per-cell JSONs and produce the headline figure --
mirrors m9_plot.py but for the Heisenberg sweep.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m13_cells"
OUT_DIR = CELLS_DIR.parent

# Cap displayed k per L so each curve ends at its first-reached floor.
# Populated after each new sweep extension; same pattern as m9_plot's PLOT_KMAX.
PLOT_KMAX = {}


def load_records():
    out = []
    for p in sorted(CELLS_DIR.glob("L*_k*_s*.json")):
        out.append(json.loads(p.read_text()))
    return out


def aggregate(records):
    by_cell = defaultdict(list)
    max_steps = 0
    for r in records:
        L, k = r["L"], r["k"]
        max_steps = max(max_steps, r["max_steps"])
        if r["steps_to_threshold"] is None:
            by_cell[(L, k)].append(("censored", r["max_steps"]))
        else:
            by_cell[(L, k)].append(("reached", r["steps_to_threshold"]))
    return by_cell, max_steps


def cell_summary(values, max_steps):
    raw = np.array([v[1] for v in values], dtype=float)
    censored = sum(1 for v in values if v[0] == "censored")
    n = len(values)
    return {
        "median": float(np.median(raw)),
        "q25": float(np.quantile(raw, 0.25)),
        "q75": float(np.quantile(raw, 0.75)),
        "n": n,
        "cens_frac": censored / max(n, 1),
        "max_steps": max_steps,
    }


def main():
    records = load_records()
    if not records:
        print("No M13 cells found. Run experiments/m13_run_sweep.py first.")
        return

    by_cell, max_steps = aggregate(records)
    L_to_curves = defaultdict(list)
    for (L, k), vals in by_cell.items():
        if L in PLOT_KMAX and k > PLOT_KMAX[L]:
            continue
        c = k / (L * L) if L else 0
        summ = cell_summary(vals, max_steps)
        L_to_curves[L].append((k, c, summ))
    for L in L_to_curves:
        L_to_curves[L].sort(key=lambda x: x[0])

    # Headline figure
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    cmap = plt.get_cmap("viridis")
    L_list = sorted(L_to_curves.keys())
    for i, L in enumerate(L_list):
        ks_all = np.array([c[0] for c in L_to_curves[L]])
        meds_all = np.array([c[2]["median"] for c in L_to_curves[L]])
        q25_all = np.array([c[2]["q25"] for c in L_to_curves[L]])
        q75_all = np.array([c[2]["q75"] for c in L_to_curves[L]])
        cens_all = np.array([c[2]["cens_frac"] for c in L_to_curves[L]])
        keep = cens_all < 0.5
        ks, meds, q25, q75 = ks_all[keep], meds_all[keep], q25_all[keep], q75_all[keep]
        color = cmap(i / max(1, len(L_list) - 1))
        ax.fill_between(ks, q25, q75, color=color, alpha=0.18)
        ax.plot(ks, meds, "o-", color=color, lw=1.8, ms=5,
                label=f"$L={L}$  (n={L})")
    ax.set_xlabel("training samples")
    ax.set_ylabel(r"VMC steps to $|\Delta E|/|E_0| \leq 0.01$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out1 = OUT_DIR / "m13_steps_vs_samples.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Wrote {out1}", flush=True)

    # ---- Figure 2: vs c = k / L^2 ----
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, L in enumerate(L_list):
        cs_all = np.array([c[1] for c in L_to_curves[L]])
        meds_all = np.array([c[2]["median"] for c in L_to_curves[L]])
        q25_all = np.array([c[2]["q25"] for c in L_to_curves[L]])
        q75_all = np.array([c[2]["q75"] for c in L_to_curves[L]])
        cens_all = np.array([c[2]["cens_frac"] for c in L_to_curves[L]])
        keep = cens_all < 0.5
        cs, meds, q25, q75 = cs_all[keep], meds_all[keep], q25_all[keep], q75_all[keep]
        color = cmap(i / max(1, len(L_list) - 1))
        ax.fill_between(cs, q25, q75, color=color, alpha=0.18)
        ax.plot(cs, meds, "o-", color=color, lw=1.8, ms=5,
                label=f"$L={L}$  (n={L})")
    ax.set_xlabel(r"$c = k / L^2$")
    ax.set_ylabel(r"VMC steps to $|\Delta E|/|E_0| \leq 0.01$")
    ax.set_yscale("log")
    ax.set_title(r"M13 (Heisenberg): normalized to $c = k/L^2$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out2 = OUT_DIR / "m13_steps_vs_c.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Wrote {out2}", flush=True)

    # ---- Figure 3: heatmap over (L, c) ----
    all_c = sorted({c for L in L_list for (_, c, _) in L_to_curves[L]})
    M = np.full((len(L_list), len(all_c)), np.nan)
    for li, L in enumerate(L_list):
        cdict = {round(c, 4): summ for (_, c, summ) in L_to_curves[L]}
        for ci, c in enumerate(all_c):
            if round(c, 4) in cdict:
                M[li, ci] = cdict[round(c, 4)]["median"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(np.log10(np.clip(M, 1, None)), aspect="auto", origin="lower",
                   extent=[all_c[0], all_c[-1], min(L_list) - 1.0, max(L_list) + 1.0],
                   cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}$(median steps to threshold)")
    ax.set_xlabel(r"$c = k / L^2$")
    ax.set_ylabel("$L$")
    ax.set_yticks(L_list)
    ax.set_title(r"M13 (Heisenberg): $\log_{10}$(VMC steps) over $(L, c)$")
    fig.tight_layout()
    out3 = OUT_DIR / "m13_heatmap.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Wrote {out3}", flush=True)

    # Summary table
    print(f"\n=== M13 (Heisenberg) summary: median (q25-q75), by (L, k) ===")
    print(f"  {'L':>2} {'k':>4} {'c=k/L^2':>8} {'med':>6} {'q25':>6} {'q75':>6} "
          f"{'cens':>5} {'n':>3}")
    for L in L_list:
        for (k, c, summ) in L_to_curves[L]:
            print(f"  {L:>2} {k:>4} {c:>8.3f} {summ['median']:>6.0f} "
                  f"{summ['q25']:>6.0f} {summ['q75']:>6.0f} "
                  f"{summ['cens_frac']:>5.0%} {summ['n']:>3}")


if __name__ == "__main__":
    main()
