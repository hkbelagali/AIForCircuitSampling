"""Invert the M9 curves: for each L, find k such that median steps-to-threshold
equals a target value, and plot k_target(L). Defaults to steps=30, but
configurable via --target.

This is the dual of the steps-vs-k headline: instead of "how many VMC steps
does k samples buy?", we ask "how many samples do you need to limit VMC to
30 polishing steps?".
"""

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m9_cells"
OUT_DIR = CELLS_DIR.parent
EXCLUDE_L = {8}


def load_curves():
    """Per-L sorted list of (k, median, q25, q75) over all cells on disk."""
    by_cell = defaultdict(list)
    for p in sorted(CELLS_DIR.glob("L*_k*_s*.json")):
        r = json.loads(p.read_text())
        if r["L"] in EXCLUDE_L:
            continue
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
    """Find smallest k along `curve` where curve[i][idx] crosses below `target`.
    idx selects which column (1=median, 2=q25, 3=q75). Log-linear interp in k.
    Returns float k, or None if no crossing."""
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
    p.add_argument("--target", type=int, default=30, help="target VMC steps")
    args = p.parse_args()

    L_data = load_curves()
    results = []
    print(f"\n=== samples needed to reach VMC steps = {args.target} ===")
    print(f"  {'L':>2}  {'k_med':>8}  {'k(q75)':>8}  {'k(q25)':>8}  "
          f"{'D':>5}  {'k/D':>6}")
    for L in sorted(L_data.keys()):
        curve = L_data[L]
        k_med = k_at_target(curve, args.target, 1)
        k_q25 = k_at_target(curve, args.target, 2)  # easier cells; smaller k
        k_q75 = k_at_target(curve, args.target, 3)  # harder cells; larger k
        D = comb(L, L // 2) ** 2
        ratio = k_med / D if k_med else float("nan")
        results.append((L, k_med, k_q25, k_q75, D, ratio))
        def _f(x): return f"{x:.0f}" if x is not None else "--"
        print(f"  {L:>2}  {_f(k_med):>8}  {_f(k_q75):>8}  {_f(k_q25):>8}  "
              f"{D:>5}  {ratio:>6.2f}")

    # Plot
    Ls = [r[0] for r in results]
    k_meds = [r[1] for r in results]
    k_q25s = [r[2] if r[2] is not None else r[1] for r in results]
    k_q75s = [r[3] if r[3] is not None else r[1] for r in results]
    Ds = [r[4] for r in results]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.fill_between(Ls, k_q25s, k_q75s, alpha=0.18, color="C0",
                    label="IQR band (q25-q75)")
    ax.plot(Ls, k_meds, "o-", color="C0", lw=2, ms=7,
            label=f"median $k$ to reach {args.target} VMC steps")
    # Overlay sector dim D and 40*D for reference
    ax.plot(Ls, Ds, "--", color="gray", lw=1.0, alpha=0.6,
            label=r"$D_{\rm sector} = \binom{L}{L/2}^2$")
    ax.plot(Ls, [40 * D for D in Ds], ":", color="C3", lw=1.0, alpha=0.7,
            label=r"$40 \cdot D_{\rm sector}$  (empirical $k_{\rm floor}$)")
    ax.set_xlabel("$L$")
    ax.set_ylabel(f"training samples $k$ to reach {args.target} VMC steps")
    ax.set_yscale("log")
    ax.set_xticks(Ls)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / f"m9_k_for_steps{args.target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
