"""M4 (Stage 2): XEB estimator compression sweep.

For each depth d in DEPTH_VALUES at n = N_QUBITS, generate N_CIRCUITS random
Sycamore-style instances. For each instance, compute the exact p_C, then:

  for k in K_VALUES:
    for rep in 1..N_REPS:
      draw k i.i.d. samples from p_C  (numpy, exact)
      compute each estimator F-hat(samples)

Per-circuit per-k aggregations:
  - mean F-hat (estimator bias diagnostic; truth F = 1 for noiseless samples)
  - per-circuit sample std (over N_REPS reps)

Over-circuit aggregations (the headline):
  - mean over circuits of per-circuit mean (bias)
  - std over circuits of per-circuit means (over-circuit variance)
  - mean over circuits of per-circuit sample std (sample variance)
  - RMSE_total = sqrt(mean over (circuits, reps) of (F-hat - 1)^2)

Headline plot:
  - one panel per depth
  - x = k (log), y = RMSE_total (log)
  - one curve per estimator
  - shows compression ratio = k_empirical / k_eff at fixed RMSE

Sanity / bridge-to-Stage-3 (H2.3): each F-hat is a scalar; it cannot be
inverted to recover individual high-p_C strings. Improving the estimator does
not improve generation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aics.circuits.brickwork import grid_for, make_rcs_circuit
from aics.circuits.exact import exact_probabilities
from aics.circuits.xeb_estimators import (
    linear_xeb,
    log_xeb,
    mle_fidelity_xeb,
    truncated_linear_xeb,
)


N_QUBITS = 10
DEPTH_VALUES = [4, 8, 14]
N_CIRCUITS = 20
K_VALUES = [10, 30, 100, 300, 1000, 3000]
N_REPS = 50

ESTIMATORS = {
    "linear (empirical)": linear_xeb,
    "log": log_xeb,
    "truncated linear (q=0.99)":
        lambda s, p: truncated_linear_xeb(s, p, q_low=0.0, q_high=0.99),
    "MLE fidelity":      mle_fidelity_xeb,
}
ESTIMATOR_COLORS = {
    "linear (empirical)":        "#888",
    "log":                        "#e76f51",
    "truncated linear (q=0.99)": "#f4a261",
    "MLE fidelity":              "#2a9d8f",
}


def sweep_one_depth(d, n_rows, n_cols, rng, verbose=True):
    """Per-circuit, per-k aggregates for one depth value."""
    # results[est][k] is a list of length N_CIRCUITS of arrays (N_REPS,)
    results = {est: {k: [] for k in K_VALUES} for est in ESTIMATORS}
    if verbose:
        print(f"\n--- depth = {d} ({N_CIRCUITS} circuits, n={N_QUBITS}) ---",
              flush=True)
    for c in range(N_CIRCUITS):
        qubits, circuit = make_rcs_circuit(n_rows, n_cols, d, seed=c)
        p_C = exact_probabilities(circuit, qubits)
        # numpy.random.choice requires exact sum-to-1; correct float roundoff.
        p_C = p_C / p_C.sum()
        N = len(p_C)
        # Repeated sampling -> for each k, N_REPS replications
        for k in K_VALUES:
            f_per_rep = {est: np.empty(N_REPS) for est in ESTIMATORS}
            for rep in range(N_REPS):
                samples = rng.choice(N, size=k, p=p_C)
                for est_name, est_fn in ESTIMATORS.items():
                    f_per_rep[est_name][rep] = est_fn(samples, p_C)
            for est in ESTIMATORS:
                results[est][k].append(f_per_rep[est])
        if verbose and (c + 1) % 5 == 0:
            print(f"  circuits done: {c + 1}/{N_CIRCUITS}", flush=True)
    return results


def aggregate(results):
    """For each (est, k): compute bias, over-circuit std, sample-std, RMSE."""
    summary = {est: {} for est in results}
    for est, by_k in results.items():
        for k, per_circuit in by_k.items():
            # per_circuit: list of (N_REPS,) arrays
            arr = np.array(per_circuit)  # (N_CIRCUITS, N_REPS)
            # per-circuit mean
            circ_means = arr.mean(axis=1)
            circ_stds = arr.std(axis=1)
            bias = float(circ_means.mean() - 1.0)
            over_circuit_std = float(circ_means.std())
            sample_std = float(circ_stds.mean())
            rmse_total = float(np.sqrt(((arr - 1.0) ** 2).mean()))
            summary[est][k] = {
                "bias": bias,
                "over_circuit_std": over_circuit_std,
                "sample_std": sample_std,
                "rmse_total": rmse_total,
            }
    return summary


def main():
    n = N_QUBITS
    n_rows, n_cols = grid_for(n)
    rng = np.random.default_rng(0)
    print(f"=== M4 Stage 2: n={n}, depths={DEPTH_VALUES}, "
          f"{N_CIRCUITS} circuits, k in {K_VALUES}, {N_REPS} reps ===")

    all_summaries = {}
    for d in DEPTH_VALUES:
        results = sweep_one_depth(d, n_rows, n_cols, rng)
        all_summaries[d] = aggregate(results)

    # --- Tables -------------------------------------------------------------
    for d in DEPTH_VALUES:
        print(f"\n--- depth = {d} ---")
        print(f"  {'estimator':<28} {'k':>5} {'bias':>9} "
              f"{'over_circ_std':>14} {'sample_std':>12} {'RMSE':>9}")
        for est in ESTIMATORS:
            for k in K_VALUES:
                s = all_summaries[d][est][k]
                print(f"  {est:<28} {k:>5} {s['bias']:+9.4f} "
                      f"{s['over_circuit_std']:>14.4f} "
                      f"{s['sample_std']:>12.4f} {s['rmse_total']:>9.4f}")

    # --- Headline plot: RMSE vs k, one panel per depth ----------------------
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, len(DEPTH_VALUES), figsize=(5.5 * len(DEPTH_VALUES), 4.5),
                            sharey=True)
    if len(DEPTH_VALUES) == 1:
        axs = [axs]
    for ax, d in zip(axs, DEPTH_VALUES):
        for est in ESTIMATORS:
            rmse = [all_summaries[d][est][k]["rmse_total"] for k in K_VALUES]
            ax.plot(K_VALUES, rmse, "o-", color=ESTIMATOR_COLORS[est],
                    label=est, linewidth=1.7, markersize=5)
        # Theoretical 1/sqrt(k) reference for Porter-Thomas linear
        k_arr = np.array(K_VALUES, dtype=float)
        ax.plot(k_arr, 1.0 / np.sqrt(k_arr), ":", color="black", alpha=0.4,
                label=r"$1/\sqrt{k}$ guide")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"sample budget $k$")
        ax.set_title(f"depth = {d}")
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel(r"$\sqrt{\mathbb{E}[(\widehat{F} - 1)^2]}$  (RMSE)")
    axs[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"M4 Stage 2: XEB estimator RMSE vs sample budget   "
        f"(n={n}, {N_CIRCUITS} circuits, {N_REPS} reps)",
        y=1.02,
    )
    fig.tight_layout()
    out = out_dir / "m4_stage2_estimator_rmse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out}")

    # --- Compression ratio summary ------------------------------------------
    # k_eff(est, d, target_rmse): smallest k where rmse_est(k) <= target
    targets = [0.5, 0.2, 0.1, 0.05]
    print("\n=== compression ratios vs empirical linear ===")
    for d in DEPTH_VALUES:
        print(f"\n  depth {d}:")
        print(f"    {'target RMSE':>13}  {'lin k_eff':>10}  {'log k_eff':>10}  "
              f"{'trunc k_eff':>12}  {'MLE k_eff':>10}")
        for tgt in targets:
            row = [f"{tgt:>13.3f}"]
            for est in ESTIMATORS:
                k_eff = None
                for k in K_VALUES:
                    if all_summaries[d][est][k]["rmse_total"] <= tgt:
                        k_eff = k
                        break
                row.append(f"{str(k_eff) if k_eff is not None else '>max':>10}")
            print("    " + "  ".join(row))


if __name__ == "__main__":
    main()
