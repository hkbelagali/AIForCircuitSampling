"""Plot F_cl(k=16) / peak_weight vs peak_weight, for all 4 sweep datasets.
The ratio captures how much the model's F_cl exceeds the 'just memorize
the peak' baseline. Ratio = 1 means F_cl matches peak exactly; > 1 means
model extracts additional information from non-peak strings."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def median(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def load_first_point(in_dir, key_field, k_target=16):
    cells_by_key = defaultdict(list)
    for p in sorted(in_dir.glob("*.json")):
        c = json.loads(p.read_text())
        if int(c["k_train"]) != k_target:
            continue
        cells_by_key[int(c[key_field])].append(c)
    out = []
    for key, cells in cells_by_key.items():
        peak = cells[0].get("peak_prob", float("nan"))
        fcl = median([c["F_cl"] for c in cells])
        out.append((peak, fcl, key))
    return sorted(out)


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")

    datasets = [
        ("n=8 d_random",  base / "m_peaked_plus_rcs_n8",  "d_append", "o", "#1f77b4"),
        ("n=12 d_random", base / "m_peaked_plus_rcs_n12", "d_append", "s", "#d62728"),
        ("n=8 d_peak",    base / "m_peaked_dPsweep_n8",   "d_P",      "^", "#2ca02c"),
        ("n=12 d_peak",   base / "m_peaked_dPsweep_n12",  "d_P",      "v", "#9467bd"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, in_dir, key_field, marker, color in datasets:
        pts = load_first_point(in_dir, key_field, k_target=16)
        if not pts:
            continue
        peaks = np.array([p[0] for p in pts])
        fcls = np.array([p[1] for p in pts])
        ratio = fcls / np.maximum(peaks, 1e-12)
        ax.plot(peaks, ratio, marker + "-", color=color, lw=1.5, ms=7,
                label=label, mfc="none", mew=1.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("peak weight")
    ax.set_ylabel(r"$F_{\rm cl}\,/\,$peak")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = base / "fcl_over_peak_k16.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
