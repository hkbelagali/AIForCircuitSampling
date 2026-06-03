"""Inverse plot for the Heisenberg sweep: for each L, find k such that median
steps-to-threshold first crosses below a target value, plot k_target(L).
Mirrors m9_plot_k_for_steps.py.
"""

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m13_cells"
OUT_DIR = CELLS_DIR.parent


def load_curves():
    by_cell = defaultdict(list)
    for p in sorted(CELLS_DIR.glob("L*_k*_s*.json")):
        r = json.loads(p.read_text())
        steps = r["steps_to_threshold"]
        if steps is None:
            steps = r["max_steps"]
        by_cell[(r["L"], r["k"])].append(steps)
    L_data = defaultdict(list)
    for (L, k), vals in by_cell.items():
        v = np.asarray(vals, dtype=float)
        L_data[L].append((k, float(np.median(v)),
                          float(np.quantile(v, 0.25)),
                          float(np.quantile(v, 0.75))))
    for L in L_data:
        L_data[L].sort()
    return L_data


def k_at_target(curve, target, idx):
    """Smallest k along `curve` where curve[i][idx] first crosses below `target`.
    Log-linear interpolation in k. Returns None if no crossing."""
    for i in range(len(curve) - 1):
        v1, v2 = curve[i][idx], curve[i + 1][idx]
        k1, k2 = curve[i][0], curve[i + 1][0]
        if v1 > target >= v2:
            if v1 == v2:
                return float(k2)
            t = (v1 - target) / (v1 - v2)
            if k1 <= 0 or k2 <= 0:
                return float(k1 + t * (k2 - k1))
            return float(np.exp(np.log(k1) + t * (np.log(k2) - np.log(k1))))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=30)
    args = p.parse_args()

    L_data = load_curves()
    results = []
    print(f"\n=== samples to reach VMC steps = {args.target} (Heisenberg) ===")
    print(f"  {'L':>2}  {'k_med':>8}  {'k(q75)':>8}  {'k(q25)':>8}  "
          f"{'D':>5}  {'k/D':>6}")
    for L in sorted(L_data.keys()):
        curve = L_data[L]
        k_med = k_at_target(curve, args.target, 1)
        k_q25 = k_at_target(curve, args.target, 2)
        k_q75 = k_at_target(curve, args.target, 3)
        D = comb(L, L // 2)
        ratio = (k_med / D) if k_med else float("nan")
        results.append((L, k_med, k_q25, k_q75, D, ratio))
        def _f(x): return f"{x:.0f}" if x is not None else "--"
        print(f"  {L:>2}  {_f(k_med):>8}  {_f(k_q75):>8}  {_f(k_q25):>8}  "
              f"{D:>5}  {ratio:>6.2f}")

    Ls = [r[0] for r in results]
    k_meds = [r[1] for r in results]
    k_q25s = [r[2] if r[2] is not None else r[1] for r in results]
    k_q75s = [r[3] if r[3] is not None else r[1] for r in results]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    valid = [i for i, k in enumerate(k_meds) if k is not None]
    Ls_v = [Ls[i] for i in valid]
    meds_v = [k_meds[i] for i in valid]
    q25_v = [k_q25s[i] for i in valid]
    q75_v = [k_q75s[i] for i in valid]
    ax.fill_between(Ls_v, q25_v, q75_v, alpha=0.18, color="C0")
    ax.plot(Ls_v, meds_v, "o-", color="C0", lw=2, ms=7)
    ax.set_xlabel("$L$")
    ax.set_ylabel(f"training samples to reach {args.target} VMC steps")
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = OUT_DIR / f"m13_k_for_steps{args.target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
