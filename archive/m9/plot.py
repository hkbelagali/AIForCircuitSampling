"""The two headline M9 plots."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(cells_dir):
    by_cell = defaultdict(list)
    max_steps = 0
    for p in sorted(Path(cells_dir).glob("L*_k*_s*.json")):
        r = json.loads(p.read_text())
        max_steps = max(max_steps, r["max_steps"])
        st = r["steps_to_threshold"]
        by_cell[(r["L"], r["k"])].append(st if st is not None else r["max_steps"])
    return by_cell, max_steps


def plot_steps_vs_samples(cells_dir, *, plot_kmax=None, ax=None):
    by_cell, max_steps = _load(cells_dir)
    plot_kmax = plot_kmax or {}
    if ax is None:
        _, ax = plt.subplots(figsize=(8.2, 5.4))
    L_set = sorted({L for (L, _) in by_cell})
    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(L_set):
        items = sorted((k, v) for (LL, k), v in by_cell.items()
                       if LL == L and (L not in plot_kmax or k <= plot_kmax[L]))
        ks, meds, q25, q75 = [], [], [], []
        for k, vals in items:
            a = np.asarray(vals, dtype=float)
            if (a >= max_steps).mean() >= 0.5:
                continue
            ks.append(k); meds.append(np.median(a))
            q25.append(np.quantile(a, 0.25)); q75.append(np.quantile(a, 0.75))
        if not ks:
            continue
        c = cmap(i / max(1, len(L_set) - 1))
        ax.fill_between(ks, q25, q75, color=c, alpha=0.18)
        ax.plot(ks, meds, "o-", color=c, lw=1.8, ms=5, label=f"$L={L}$  (n={2*L})")
    ax.set_xlabel("training samples")
    ax.set_ylabel(r"VMC steps to $|\Delta E|/|E_0| \leq 0.01$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    return ax


def _k_at_target(curve, target, col):
    for i in range(len(curve) - 1):
        v1, v2 = curve[i][col], curve[i + 1][col]
        k1, k2 = curve[i][0], curve[i + 1][0]
        if v1 > target >= v2 and v1 != v2:
            t = (v1 - target) / (v1 - v2)
            if k1 <= 0 or k2 <= 0:
                return float(k1 + t * (k2 - k1))
            return float(np.exp(np.log(k1) + t * (np.log(k2) - np.log(k1))))
    return None


def plot_k_for_steps(cells_dir, target=30, *, ax=None):
    by_cell, _ = _load(cells_dir)
    by_L = defaultdict(list)
    for (L, k), vals in by_cell.items():
        a = np.asarray(vals, dtype=float)
        by_L[L].append((k, float(np.median(a)),
                        float(np.quantile(a, 0.25)),
                        float(np.quantile(a, 0.75))))
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))
    Ls, k_med, k_q25, k_q75 = [], [], [], []
    for L in sorted(by_L):
        curve = sorted(by_L[L])
        km = _k_at_target(curve, target, 1)
        if km is None:
            continue
        Ls.append(L); k_med.append(km)
        k_q25.append(_k_at_target(curve, target, 2) or km)
        k_q75.append(_k_at_target(curve, target, 3) or km)
    if Ls:
        ax.fill_between(Ls, k_q25, k_q75, alpha=0.18, color="C0")
        ax.plot(Ls, k_med, "o-", color="C0", lw=2, ms=7)
    ax.set_xlabel("$L$")
    ax.set_ylabel(f"training samples to reach {target} VMC steps")
    ax.set_yscale("log")
    ax.set_xticks(Ls or [3, 4, 5, 6, 7])
    ax.grid(alpha=0.3, which="both")
    return ax
