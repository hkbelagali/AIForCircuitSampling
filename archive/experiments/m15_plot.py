"""Aggregate M15 cells and plot:
  - One M9-style figure per weight w: steps to Pauli-threshold vs k, per L.
  - Family plot: k_floor(w) vs w for each L (testing k_floor ~ 4^w hypothesis).
"""

import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m15_cells"
OUT_DIR = CELLS_DIR.parent


def load_records():
    out = []
    for p in sorted(CELLS_DIR.glob("L*_k*_s*.json")):
        out.append(json.loads(p.read_text()))
    return out


def aggregate(records, max_weight):
    # by_cell[(L, k)] -> dict {w: list of step counts (using max_steps when None)}
    by_cell = defaultdict(lambda: {w: [] for w in range(max_weight + 1)})
    max_steps_global = 0
    for r in records:
        L, k = r["L"], r["k"]
        max_steps = r["max_steps"]
        max_steps_global = max(max_steps_global, max_steps)
        psteps = r.get("steps_to_P_threshold", {})
        for w in range(max_weight + 1):
            v = psteps.get(str(w))      # JSON serializes int keys as strings
            if v is None:
                v = psteps.get(w)
            v = v if v is not None else max_steps
            by_cell[(L, k)][w].append(v)
    return by_cell, max_steps_global


def main():
    records = load_records()
    if not records:
        print("No M15 cells found yet.")
        return
    max_weight = max(r.get("max_weight", 3) for r in records)
    by_cell, max_steps = aggregate(records, max_weight)

    # ---- per-weight figures: steps vs k, curves per L ----
    L_set = sorted({L for (L, k) in by_cell.keys()})
    for w in range(max_weight + 1):
        if w == 0:
            continue  # identity, trivial threshold
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        cmap = plt.get_cmap("viridis")
        for li, L in enumerate(L_set):
            rows = sorted([(k, by_cell[(L, k)][w]) for (LL, k) in by_cell if LL == L])
            ks = []; meds = []; q25 = []; q75 = []
            for k, v in rows:
                if not v: continue
                v = np.asarray(v, dtype=float)
                cens = float(np.mean(v >= max_steps))
                if cens >= 0.5: continue
                ks.append(k); meds.append(float(np.median(v)))
                q25.append(float(np.quantile(v, 0.25)))
                q75.append(float(np.quantile(v, 0.75)))
            ks = np.asarray(ks); meds = np.asarray(meds)
            if len(ks) == 0: continue
            color = cmap(li / max(1, len(L_set) - 1))
            ax.fill_between(ks, q25, q75, color=color, alpha=0.18)
            ax.plot(ks, meds, "o-", color=color, lw=1.8, ms=5,
                    label=f"$L={L}$  (n={2*L})")
        ax.set_xlabel("training samples")
        ax.set_ylabel(rf"VMC steps to $\max_{{|P|\leq {w}}}|\langle P\rangle$-err$|\leq 0.01$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"M15: steps to Pauli-threshold for weight $\\leq {w}$")
        fig.tight_layout()
        out = OUT_DIR / f"m15_steps_w{w}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Wrote {out}", flush=True)

    # ---- family plot: at fixed L, steps-to-threshold vs w for several k ----
    # Pick the largest k present for each L, show all w on one plot.
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    cmap = plt.get_cmap("viridis")
    for li, L in enumerate(L_set):
        ks_for_L = sorted({k for (LL, k) in by_cell if LL == L})
        if not ks_for_L: continue
        k_max = ks_for_L[-1]
        meds_by_w = []
        for w in range(1, max_weight + 1):
            v = np.asarray(by_cell[(L, k_max)][w], dtype=float)
            cens = float(np.mean(v >= max_steps))
            if cens >= 0.5:
                meds_by_w.append(np.nan); continue
            meds_by_w.append(float(np.median(v)))
        ws = np.arange(1, max_weight + 1)
        color = cmap(li / max(1, len(L_set) - 1))
        ax.plot(ws, meds_by_w, "o-", color=color, lw=1.8, ms=6,
                label=f"$L={L}$  ($k={k_max}$)")
    ax.set_xlabel("Pauli weight $w$")
    ax.set_ylabel(r"VMC steps to threshold (at largest $k$)")
    ax.set_xticks(np.arange(1, max_weight + 1))
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("M15: VMC steps vs Pauli weight (per L, at saturated $k$)")
    fig.tight_layout()
    out = OUT_DIR / "m15_steps_vs_weight.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {out}", flush=True)

    # ---- compact summary table ----
    print(f"\n=== M15 summary: median steps to Pauli threshold per (L, k, w) ===")
    for L in L_set:
        ks_for_L = sorted({k for (LL, k) in by_cell if LL == L})
        for k in ks_for_L:
            cell = by_cell[(L, k)]
            sline = f"  L={L} k={k:>4d}: "
            for w in range(0, max_weight + 1):
                v = np.asarray(cell[w], dtype=float)
                cens = float(np.mean(v >= max_steps)) if len(v) else 1.0
                med = float(np.median(v)) if len(v) else float("nan")
                sline += f"w{w}={med:>5.0f}({cens:.0%}) "
            print(sline)


if __name__ == "__main__":
    main()
