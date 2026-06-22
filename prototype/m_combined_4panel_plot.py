"""Four-panel figure combining the d_append (RCS-append) and d_P (PQC-depth)
sweeps at n=8 and n=12. Row 1: d_append sweeps. Row 2: d_P sweeps."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def median_iqr(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def load_dir(in_dir, key_field):
    by_key_k = defaultdict(lambda: defaultdict(list))
    peak_by_key = {}
    for p in sorted(in_dir.glob("*.json")):
        c = json.loads(p.read_text())
        key = int(c[key_field]); k = int(c["k_train"])
        by_key_k[key][k].append(c)
        peak_by_key[key] = c.get("peak_prob", float("nan"))
    return by_key_k, peak_by_key


def plot_panel(ax, by_key_k, peak_by_key, key_label, title):
    keys = sorted(by_key_k)
    colors = [cm.viridis(i / max(1, len(keys) - 1)) for i in range(len(keys))]
    for key, col in zip(keys, colors):
        ks = sorted(by_key_k[key])
        med, lo, hi = [], [], []
        for k in ks:
            vals = [c["F_cl"] for c in by_key_k[key][k]]
            m, q1, q3 = median_iqr(vals)
            med.append(m); lo.append(q1); hi.append(q3)
        peak = peak_by_key[key]
        label = fr"{key_label}={key}" + (f"  (pk={peak:.3f})" if peak == peak else "")
        ax.plot(ks, med, "o-", color=col, lw=1.6, ms=4, label=label)
        ax.fill_between(ks, lo, hi, color=col, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    ax.set_ylabel(r"$F_{\rm cl}$")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=7)


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Row 1: d_append sweeps
    da_n8, peak_da_n8 = load_dir(base / "m_peaked_plus_rcs_n8", "d_append")
    da_n12, peak_da_n12 = load_dir(base / "m_peaked_plus_rcs_n12", "d_append")
    plot_panel(axes[0, 0], da_n8, peak_da_n8, r"$d_\lambda$",
                r"(a) N=8, peaked + appended RCS")
    plot_panel(axes[0, 1], da_n12, peak_da_n12, r"$d_\lambda$",
                r"(b) N=12, peaked + appended RCS")

    # Row 2: d_P sweeps
    dp_n8, peak_dp_n8 = load_dir(base / "m_peaked_dPsweep_n8", "d_P")
    dp_n12, peak_dp_n12 = load_dir(base / "m_peaked_dPsweep_n12", "d_P")
    plot_panel(axes[1, 0], dp_n8, peak_dp_n8, r"$d_P$",
                r"(c) N=8, PQC depth sweep")
    plot_panel(axes[1, 1], dp_n12, peak_dp_n12, r"$d_P$",
                r"(d) N=12, PQC depth sweep")

    fig.tight_layout()
    out = base / "peaked_family_4panel.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
