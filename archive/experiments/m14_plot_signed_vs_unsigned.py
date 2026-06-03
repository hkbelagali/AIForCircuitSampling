"""Compare signed (m14) vs unsigned (m9) sample complexity at a fixed L:
overlay median + IQR curves of steps-to-threshold vs k.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M9_DIR = ROOT / "results" / "m9_cells"
M14_DIR = ROOT / "results" / "m14_cells"


def load(cells_dir, L):
    by_k = defaultdict(list)
    for p in sorted(cells_dir.glob(f"L{L}_k*_s*.json")):
        r = json.loads(p.read_text())
        if r["L"] != L:
            continue
        st = r["steps_to_threshold"]
        if st is None:
            st = r["max_steps"]
        by_k[r["k"]].append(st)
    rows = []
    for k in sorted(by_k.keys()):
        v = np.asarray(by_k[k], dtype=float)
        cens = int(np.sum(v >= r["max_steps"]))
        rows.append((k, float(np.median(v)),
                     float(np.quantile(v, 0.25)),
                     float(np.quantile(v, 0.75)),
                     cens, len(v)))
    return rows


def plot_curve(ax, rows, label, color, marker):
    ks = np.array([r[0] for r in rows])
    meds = np.array([r[1] for r in rows])
    q25 = np.array([r[2] for r in rows])
    q75 = np.array([r[3] for r in rows])
    cens = np.array([r[4] / r[5] for r in rows])
    keep = cens < 0.5
    if keep.any():
        ax.fill_between(ks[keep], q25[keep], q75[keep], color=color, alpha=0.15)
        ax.plot(ks[keep], meds[keep], "-", color=color, lw=1.8, marker=marker, ms=6,
                label=label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    args = p.parse_args()
    L = args.L

    m9 = load(M9_DIR, L)
    m14 = load(M14_DIR, L)
    print(f"L={L}: m9 cells {len(m9)} k-values, m14 cells {len(m14)} k-values")

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    plot_curve(ax, m9, "ED-given signs (M9)", "C0", "o")
    plot_curve(ax, m14, "Learned signs (M14, 2-layer head)", "C3", "s")
    ax.set_xlabel("training samples")
    ax.set_ylabel(r"VMC steps to $|\Delta E|/|E_0| \leq 0.01$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Hubbard L={L}: sign-learning cost vs ED-signs baseline")
    fig.tight_layout()
    out = ROOT / "results" / f"m14_vs_m9_L{L}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")

    print(f"\n=== L={L} median steps comparison ===")
    print(f"  {'k':>6}  {'M9 (unsigned)':>14}  {'M14 (signed)':>13}  {'ratio':>6}")
    m9_dict = {r[0]: r[1] for r in m9}
    m14_dict = {r[0]: r[1] for r in m14}
    all_ks = sorted(set(m9_dict.keys()) | set(m14_dict.keys()))
    for k in all_ks:
        u = m9_dict.get(k)
        s = m14_dict.get(k)
        u_s = f"{u:.0f}" if u else "--"
        s_s = f"{s:.0f}" if s else "--"
        ratio_s = f"{s/u:.1f}x" if (u and s) else "--"
        print(f"  {k:>6d}  {u_s:>14}  {s_s:>13}  {ratio_s:>6}")


if __name__ == "__main__":
    main()
