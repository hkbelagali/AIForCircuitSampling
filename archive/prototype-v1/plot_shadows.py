"""Aggregate shadow-sweep cells and produce four figures:

  shadow_infidelity_vs_k.png   — 1 - F vs k_total, curve per L (mean across
                                  seeds, IQR shaded).
  shadow_energy_err_vs_k.png   — |E - E_0| / |E_0| vs k_total.
  shadow_pauli_mean_vs_k.png   — one panel per L; mean absolute Pauli error
                                  vs k_total, one curve per Pauli weight.
  shadow_pauli_mean_vs_weight.png — one panel per L; same data plotted with
                                     weight on the x-axis, one curve per k.

Cell-level "Pauli error" is the mean of |<P>_model - <P>_true| over all
even-Y, sector-preserving Paulis of a given exact weight; seed-level
aggregation is the mean across seeds (so the curve label is just "mean
absolute error").
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(cells_dir):
    return [json.loads(p.read_text())
            for p in sorted(Path(cells_dir).glob("L*_kz*_kr*_s*.json"))]


def aggregate(recs):
    by_cell = defaultdict(list)
    for r in recs:
        by_cell[(r["L"], r["k_total"])].append(r)
    return by_cell


def _get_w_value(rec, w, kind):
    """kind in {'cumulative', 'exact'}. Returns float or None.
    Cell-level quantity is MEAN Pauli error within weight-w Paulis
    (when available) — falls back to the older max-based field for
    cumulative if mean is missing."""
    if kind == "cumulative":
        d = (rec.get("pauli_max_err_cumulative")
             or rec.get("pauli_max_err_by_weight") or {})
    else:
        d = (rec.get("pauli_mean_err_exact")
             or rec.get("pauli_max_err_exact") or {})
    v = d.get(str(w))
    if v is None: v = d.get(w)
    return float(v) if v is not None else None


def _weights_present(recs, kind="cumulative"):
    if kind == "exact":
        keys = ("pauli_mean_err_exact", "pauli_max_err_exact")
    else:
        keys = ("pauli_max_err_cumulative", "pauli_max_err_by_weight")
    ws = set()
    for r in recs:
        d = {}
        for k in keys:
            if r.get(k): d = r[k]; break
        for k in d.keys():
            try: ws.add(int(k))
            except (TypeError, ValueError): pass
    return sorted(w for w in ws if w >= 1)


def _curve(by_L_curves, metric_fn, ax, ylabel, title=None, hline=None,
           log_x=True, log_y=True):
    cmap = plt.get_cmap("viridis")
    Ls = sorted(by_L_curves.keys())
    for i, L in enumerate(Ls):
        kts, means, lo, hi = [], [], [], []
        for k_t in sorted(by_L_curves[L].keys()):
            vals = [metric_fn(r) for r in by_L_curves[L][k_t]]
            vals = [v for v in vals if v is not None]
            if not vals: continue
            kts.append(k_t)
            v = np.asarray(vals, dtype=float)
            means.append(float(np.mean(v)))
            lo.append(float(np.quantile(v, 0.25)))
            hi.append(float(np.quantile(v, 0.75)))
        if not kts: continue
        c = cmap(i / max(1, len(Ls) - 1))
        ax.fill_between(kts, lo, hi, color=c, alpha=0.18)
        ax.plot(kts, means, "o-", color=c, lw=1.8, ms=5,
                label=f"$L={L}$ (n={2*L})")
    if hline is not None:
        ax.axhline(hline, color="r", ls="--", lw=1.0, alpha=0.7)
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)


def _plot_all_weights_vs_k(by_L, weights, out_path, kind="exact"):
    Ls = sorted(by_L.keys())
    fig, axes = plt.subplots(1, len(Ls), figsize=(4.8 * len(Ls), 5.2),
                             sharey=True)
    if len(Ls) == 1: axes = [axes]
    cmap = plt.get_cmap("turbo")
    for ax, L in zip(axes, Ls):
        ws_here = [w for w in weights if w <= 2 * L]
        k_ts = sorted(by_L[L].keys())
        for j, w in enumerate(ws_here):
            kts, means = [], []
            for k_t in k_ts:
                vals = [_get_w_value(r, w, kind) for r in by_L[L][k_t]]
                vals = [v for v in vals if v is not None]
                if not vals: continue
                kts.append(k_t)
                means.append(float(np.mean(vals)))
            if not kts: continue
            c = cmap(j / max(1, len(ws_here) - 1))
            ax.plot(kts, means, "o-", color=c, lw=1.2, ms=4,
                    label=f"$w={w}$")
        ax.axhline(0.01, color="r", ls="--", lw=0.8, alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("training samples")
        ax.grid(alpha=0.3, which="both")
        if ax is axes[0]:
            ax.set_ylabel("mean absolute error")
        ax.legend(fontsize=7, loc="best", ncol=2,
                  title=f"$L={L}$  (n={2*L})")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_err_vs_weight(by_L, weights, out_path, kind="exact"):
    Ls = sorted(by_L.keys())
    fig, axes = plt.subplots(1, len(Ls), figsize=(4.8 * len(Ls), 5.2),
                             sharey=True)
    if len(Ls) == 1: axes = [axes]
    kmap = plt.get_cmap("plasma")
    all_kts = sorted({k for L in Ls for k in by_L[L].keys()})
    for ax, L in zip(axes, Ls):
        ws_here = [w for w in weights if w <= 2 * L]
        k_ts = sorted(by_L[L].keys())
        for j, k_t in enumerate(k_ts):
            ws_x, means = [], []
            for w in ws_here:
                vals = [_get_w_value(r, w, kind) for r in by_L[L][k_t]]
                vals = [v for v in vals if v is not None]
                if not vals: continue
                ws_x.append(w)
                means.append(float(np.mean(vals)))
            if not ws_x: continue
            c = kmap(all_kts.index(k_t) / max(1, len(all_kts) - 1))
            ax.plot(ws_x, means, "o-", color=c, lw=1.4, ms=5)
        ax.set_xlabel("weight")
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xticks(ws_here)
        if ax is axes[0]:
            ax.set_ylabel("mean absolute error")
        # Per-panel L label inside the axes box.
        ax.text(0.02, 0.98, f"$L={L}$ (n={2*L})", transform=ax.transAxes,
                ha="left", va="top", fontsize=10)
    handles = [plt.Line2D([0], [0],
                          color=kmap(j / max(1, len(all_kts) - 1)),
                          lw=1.4, marker="o", label=f"$k={k:.0e}$")
               for j, k in enumerate(all_kts)]
    axes[-1].legend(handles=handles, loc="upper left", fontsize=8,
                    bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=str, default="results/shadow_cells_v3")
    ap.add_argument("--out-dir", type=str, default="results/v3")
    args = ap.parse_args()

    recs = load(args.cells)
    if not recs:
        print(f"No cells found in {args.cells}"); return
    print(f"loaded {len(recs)} cells")

    by_cell = aggregate(recs)
    by_L = defaultdict(dict)
    for (L, k_t), rs in by_cell.items():
        by_L[L][k_t] = rs

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _curve(by_L, lambda r: max(1.0 - r["fidelity"], 1e-5), ax,
           ylabel=r"$1 - F(\psi_\theta, \psi_0)$",
           title="Shadow MLE: infidelity vs sample budget",
           hline=0.01)
    out = out_dir / "shadow_infidelity_vs_k.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    _curve(by_L, lambda r: max(abs(r["rel_err"]), 1e-5), ax,
           ylabel=r"$|E_\theta - E_0| / |E_0|$",
           title="Shadow MLE: energy relative error vs sample budget",
           hline=0.01)
    out = out_dir / "shadow_energy_err_vs_k.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}"); plt.close(fig)

    exact_weights = _weights_present(recs, "exact")
    if exact_weights:
        out = out_dir / "shadow_pauli_mean_vs_k.png"
        _plot_all_weights_vs_k(by_L, exact_weights, out, kind="exact")
        print(f"wrote {out}")
        out = out_dir / "shadow_pauli_mean_vs_weight.png"
        _plot_err_vs_weight(by_L, exact_weights, out, kind="exact")
        print(f"wrote {out}")

    print(f"\n=== shadow sweep summary (mean across seeds) ===")
    print(f"  {'L':>2} {'k_tot':>7} {'fid_mean':>9} {'rel_mean':>9}")
    for L in sorted(by_L.keys()):
        for k_t in sorted(by_L[L].keys()):
            rs = by_L[L][k_t]
            fids = np.array([r["fidelity"] for r in rs])
            errs = np.array([abs(r["rel_err"]) for r in rs])
            print(f"  {L:>2} {k_t:>7} {np.mean(fids):>9.4f} "
                  f"{np.mean(errs):>9.4f}")

    if exact_weights:
        print(f"\n=== mean Pauli error per exact weight (mean across seeds) ===")
        print(f"  {'L':>2} {'k_tot':>7}", end="")
        for w in exact_weights: print(f" {'w'+str(w):>8}", end="")
        print()
        for L in sorted(by_L.keys()):
            for k_t in sorted(by_L[L].keys()):
                rs = by_L[L][k_t]
                print(f"  {L:>2} {k_t:>7}", end="")
                for w in exact_weights:
                    if w > 2 * L:
                        print(f" {'-':>8}", end=""); continue
                    vals = [_get_w_value(r, w, "exact") for r in rs]
                    vals = [v for v in vals if v is not None]
                    if vals: print(f" {float(np.mean(vals)):>8.4f}", end="")
                    else:    print(f" {'-':>8}", end="")
                print()


if __name__ == "__main__":
    main()
