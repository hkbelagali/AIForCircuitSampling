"""Plot per-weight mean abs error: model vs truth, one curve per w_train,
plus the shadow-noise reference and a 'predict zero' reference (RMS of
true expectations at each weight)."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {1: "#4c72b0", 2: "#dd8452", 3: "#55a467", 4: "#c44e52", 5: "#8172b3"}


def median_iqr(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def main():
    in_dir = Path(__file__).resolve().parents[1] / "results" / "m_rcs_z_pauli"
    out_path = Path(__file__).resolve().parents[1] / "results" / "m_rcs_z_pauli_extrapolation.png"
    cells = [json.loads(p.read_text()) for p in sorted(in_dir.glob("*.json"))]
    if not cells:
        raise SystemExit(f"no cells found under {in_dir}")

    n = cells[0]["n"]
    depth = cells[0]["depth"]
    k_train = cells[0]["k_train"]

    # Group by (w_train, weight) and (weight) for shadow + truth
    err_model = defaultdict(lambda: defaultdict(list))   # w_train -> w -> [errs]
    err_shadow = defaultdict(list)                        # w -> [errs]
    rms_true = defaultdict(list)                          # w -> [rms]
    for c in cells:
        w_train = c["w_train"]
        for w in range(1, n + 1):
            wk = str(w) if str(w) in c["err_by_weight_model"] else w
            err_model[w_train][w].append(c["err_by_weight_model"][wk])
            err_shadow[w].append(c["err_by_weight_shadow"][wk])
            rms_true[w].append(c["true_rms_by_weight"][wk])

    weights = list(range(1, n + 1))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Reference: shadow noise at this k
    shadow_med = [median_iqr(err_shadow[w])[0] for w in weights]
    ax.plot(weights, shadow_med, "k--", lw=1.5, alpha=0.7,
            label=fr"direct shadow ($k={k_train}$)")

    # Reference: predict-zero (RMS of true expectations)
    zero_med = [median_iqr(rms_true[w])[0] for w in weights]
    ax.plot(weights, zero_med, "k:", lw=1.5, alpha=0.5,
            label=r"predict $0$  (RMS$\langle Z_S\rangle$)")

    # One curve per w_train
    for w_train in sorted(err_model):
        ys, lo, hi = [], [], []
        for w in weights:
            m, q1, q3 = median_iqr(err_model[w_train][w])
            ys.append(m); lo.append(q1); hi.append(q3)
        ax.plot(weights, ys, "o-", color=COLORS[w_train], lw=1.8,
                label=fr"model, trained on $|S|\leq{w_train}$")
        ax.fill_between(weights, lo, hi, color=COLORS[w_train], alpha=0.18)
        ax.axvline(w_train + 0.0, color=COLORS[w_train], lw=0.5, ls=":", alpha=0.4)

    ax.set_yscale("log")
    ax.set_xlabel(r"evaluation weight $w$")
    ax.set_ylabel(r"$\langle|\langle Z_S\rangle_\theta - \langle Z_S\rangle_{p_C}|\rangle_{|S|=w}$")
    ax.set_title(f"Z-Pauli extrapolation: n={n}, depth={depth}, k_train={k_train}, "
                 f"3 seeds (median + IQR)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
