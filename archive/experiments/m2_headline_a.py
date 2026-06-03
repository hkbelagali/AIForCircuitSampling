"""M2 Headline A: novel-XEB vs Hamming radius, with proper k/epochs/T sweep.

For n = N_QUBITS qubits, depth D, and N_SEEDS independent Sycamore-style
circuit instances, train AR transformers on training-sample budgets k, with
training-epoch budgets E, sample m candidates at temperatures T, and compute:
  - linear XEB on the candidate set,
  - unique fraction = #unique(candidates) / m,
  - radius-filtered novel-XEB(r) = mean over (unique novel) of dim * p_C - 1.

Aggregate across seeds (mean + std). Three-panel headline figure:
  (left)  novel-XEB(r) for k in K, fixed (epochs, T)
  (middle) novel-XEB(r) for T in TEMPS, fixed (epochs, k)
  (right) unique fraction at r = 0 vs T, one curve per k

Key statement to be supported or refuted:
  "Any positive XEB disappears once exact memorization and near-neighbor
  leakage are removed; increasing sampling temperature restores diversity but
  not XEB."
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(8)

from aics.circuits.brickwork import grid_for, make_rcs_circuit
from aics.circuits.exact import (
    bits_to_int,
    exact_probabilities,
    int_to_bits,
    sample_from_circuit,
)
from aics.circuits.xeb import linear_xeb, novel_xeb_vs_radius
from aics.models.ar_transformer import ARTransformer, train_ar


# ---- config -----------------------------------------------------------------
N_QUBITS = 10
DEPTH = 14
N_SEEDS = 5
K_VALUES = [50, 200, 1000]
EPOCH_VALUES = [50, 100]
TEMPERATURE_VALUES = [1.0, 1.5, 2.0]
M_CANDIDATES = 10000

# Reference slices for the 3-panel headline (chosen after run if needed):
HEADLINE_FIXED_EPOCHS = 100
HEADLINE_FIXED_T_FOR_K_PANEL = 1.0
HEADLINE_FIXED_K_FOR_T_PANEL = 200
HEADLINE_RADII = list(range(0, N_QUBITS + 1))
# ----------------------------------------------------------------------------


def nn_noise_baseline(training_int, m, n, h, rng):
    if len(training_int) == 0:
        return np.zeros(0, dtype=np.int64)
    seeds = rng.choice(training_int, size=m, replace=True)
    out = np.empty(m, dtype=np.int64)
    for i, s in enumerate(seeds):
        new = int(s)
        positions = rng.choice(n, size=h, replace=False)
        for b in positions:
            new ^= (1 << int(b))
        out[i] = new
    return out


def compute_metrics(candidate_int, training_int, p_C, n, radii):
    out = {
        "linear_xeb": linear_xeb(candidate_int, p_C),
        "n_unique": int(np.unique(candidate_int).size),
        "n_samples": int(candidate_int.size),
    }
    out["unique_fraction"] = out["n_unique"] / max(out["n_samples"], 1)
    nxeb = novel_xeb_vs_radius(candidate_int, training_int, p_C, n, radii=radii)
    for r in radii:
        v, c = nxeb[r]
        out[f"nxeb_r{r}"] = v
        out[f"count_r{r}"] = c
    return out


def aggregate(records, keys, fields, radii):
    """Group records by `keys` and average `fields` (with std) across seeds."""
    groups = defaultdict(list)
    for r in records:
        gkey = tuple(r[k] for k in keys)
        groups[gkey].append(r)
    out = []
    for gkey, recs in groups.items():
        agg = dict(zip(keys, gkey))
        agg["n_seeds"] = len(recs)
        for f in fields:
            vals = np.array([r[f] for r in recs if not (isinstance(r[f], float)
                                                        and np.isnan(r[f]))])
            if len(vals) == 0:
                agg[f"{f}_mean"] = float("nan")
                agg[f"{f}_std"] = float("nan")
            else:
                agg[f"{f}_mean"] = float(np.mean(vals))
                agg[f"{f}_std"] = float(np.std(vals))
        for r_ in radii:
            f = f"nxeb_r{r_}"
            vals = np.array([rec[f] for rec in recs if not (isinstance(rec[f], float)
                                                            and np.isnan(rec[f]))])
            if len(vals) == 0:
                agg[f"{f}_mean"] = float("nan")
                agg[f"{f}_std"] = float("nan")
            else:
                agg[f"{f}_mean"] = float(np.mean(vals))
                agg[f"{f}_std"] = float(np.std(vals))
            counts = np.array([rec[f"count_r{r_}"] for rec in recs])
            agg[f"count_r{r_}_mean"] = float(np.mean(counts))
        out.append(agg)
    return out


def main():
    n = N_QUBITS
    n_rows, n_cols = grid_for(n)
    radii = HEADLINE_RADII
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"=== M2 Headline A: n={n} ({n_rows}x{n_cols}), depth={DEPTH}, "
          f"{N_SEEDS} seeds, m={M_CANDIDATES} ===")
    print(f"  K  = {K_VALUES}")
    print(f"  E  = {EPOCH_VALUES}")
    print(f"  T  = {TEMPERATURE_VALUES}")

    rng = np.random.default_rng(0)
    records = []
    baseline_records = []

    for s_idx, c_seed in enumerate(range(N_SEEDS)):
        print(f"\n--- seed {c_seed} ({s_idx+1}/{N_SEEDS}) ---")
        qubits, circuit = make_rcs_circuit(n_rows, n_cols, DEPTH, seed=c_seed)
        p_C = exact_probabilities(circuit, qubits)
        dim = len(p_C)
        print(f"  dim={dim}, p_C.max={p_C.max():.4f}, "
              f"H={float(-(p_C[p_C>0]*np.log(p_C[p_C>0])).sum()):.3f} nats "
              f"(uniform={np.log(dim):.3f})")

        # Baselines per seed: NN-noise (h=1) and uniform-random
        # Use the largest k for training set as the "training" the baselines see
        ref_k = max(K_VALUES)
        ref_train_int = sample_from_circuit(circuit, qubits, ref_k, seed=c_seed)
        nn_int = nn_noise_baseline(ref_train_int, M_CANDIDATES, n, h=1, rng=rng)
        unif_int = rng.integers(0, dim, size=M_CANDIDATES, dtype=np.int64)
        for label, cand in (("nn_noise", nn_int), ("uniform", unif_int)):
            m = compute_metrics(cand, ref_train_int, p_C, n, radii)
            m.update({"seed": c_seed, "model": label})
            baseline_records.append(m)

        for k in K_VALUES:
            train_int = sample_from_circuit(circuit, qubits, k, seed=c_seed)
            X_train = int_to_bits(train_int, n)
            for E in EPOCH_VALUES:
                print(f"  [training k={k} E={E}]", flush=True)
                torch.manual_seed(c_seed * 1000 + E)
                model = ARTransformer(n_qubits=n, d_model=64, n_layers=2, n_heads=4)
                nll = train_ar(model, X_train, n_epochs=E, lr=2e-3,
                               batch_size=32, verbose=False)
                model.eval()
                for T in TEMPERATURE_VALUES:
                    cand_bits = model.sample(M_CANDIDATES,
                                             temperature=T).cpu().numpy()
                    cand_int = bits_to_int(cand_bits)
                    m = compute_metrics(cand_int, train_int, p_C, n, radii)
                    m.update({"seed": c_seed, "k": k, "epochs": E, "T": T,
                              "final_nll": nll})
                    records.append(m)
                    print(f"  k={k:>4} E={E:>3} T={T:.1f}  "
                          f"XEB={m['linear_xeb']:+.3f}  "
                          f"uniq={m['unique_fraction']:.3f}  "
                          f"nxeb(r=1)={m['nxeb_r1']:+.3f} "
                          f"({m['count_r1']} cands)", flush=True)

    # ---- Aggregate ----------------------------------------------------------
    fields = ["linear_xeb", "unique_fraction"] + [f"count_r{r}" for r in radii]
    agg = aggregate(records, keys=("k", "epochs", "T"), fields=fields, radii=radii)
    agg_baselines = aggregate(baseline_records, keys=("model",),
                              fields=["linear_xeb", "unique_fraction"], radii=radii)

    # ---- Headline plot ------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: novel-XEB(r) for varying k, fixed (epochs, T)
    ax = axs[0]
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(K_VALUES):
        rec = next((a for a in agg if a["k"] == k and a["epochs"] == HEADLINE_FIXED_EPOCHS
                    and a["T"] == HEADLINE_FIXED_T_FOR_K_PANEL), None)
        if rec is None:
            continue
        y = np.array([rec[f"nxeb_r{r}_mean"] for r in radii])
        ye = np.array([rec[f"nxeb_r{r}_std"] for r in radii])
        color = cmap(i / max(1, len(K_VALUES) - 1))
        ax.errorbar(radii, y, yerr=ye, fmt="o-", color=color,
                    label=f"k = {k}", capsize=2)
    for label, color in (("nn_noise", "#f4a261"), ("uniform", "#888")):
        rec = next((a for a in agg_baselines if a["model"] == label), None)
        if rec is None:
            continue
        y = np.array([rec[f"nxeb_r{r}_mean"] for r in radii])
        ax.plot(radii, y, "--", color=color, alpha=0.7, label=label)
    ax.axhline(0, color="black", lw=0.5, alpha=0.6)
    ax.axhline(1, color="#2a9d8f", lw=0.8, linestyle=":", alpha=0.6,
               label=r"ideal ($p_C$ samples)")
    ax.set_xlabel("Hamming radius $r$ from training set")
    ax.set_ylabel(r"$\widehat{F}^{\rm novel}_{\rm XEB}(r)$")
    ax.set_title(f"k sweep at epochs={HEADLINE_FIXED_EPOCHS}, T={HEADLINE_FIXED_T_FOR_K_PANEL}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: novel-XEB(r) for varying T, fixed (epochs, k)
    ax = axs[1]
    cmap = plt.get_cmap("plasma")
    for i, T in enumerate(TEMPERATURE_VALUES):
        rec = next((a for a in agg if a["k"] == HEADLINE_FIXED_K_FOR_T_PANEL
                    and a["epochs"] == HEADLINE_FIXED_EPOCHS and a["T"] == T), None)
        if rec is None:
            continue
        y = np.array([rec[f"nxeb_r{r}_mean"] for r in radii])
        ye = np.array([rec[f"nxeb_r{r}_std"] for r in radii])
        color = cmap(i / max(1, len(TEMPERATURE_VALUES) - 1))
        ax.errorbar(radii, y, yerr=ye, fmt="s-", color=color,
                    label=f"T = {T:.1f}", capsize=2)
    for label, color in (("nn_noise", "#f4a261"), ("uniform", "#888")):
        rec = next((a for a in agg_baselines if a["model"] == label), None)
        if rec is None:
            continue
        y = np.array([rec[f"nxeb_r{r}_mean"] for r in radii])
        ax.plot(radii, y, "--", color=color, alpha=0.7, label=label)
    ax.axhline(0, color="black", lw=0.5, alpha=0.6)
    ax.axhline(1, color="#2a9d8f", lw=0.8, linestyle=":", alpha=0.6,
               label=r"ideal ($p_C$ samples)")
    ax.set_xlabel("Hamming radius $r$ from training set")
    ax.set_title(f"T sweep at epochs={HEADLINE_FIXED_EPOCHS}, k={HEADLINE_FIXED_K_FOR_T_PANEL}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: unique fraction vs T, one curve per k
    ax = axs[2]
    for i, k in enumerate(K_VALUES):
        ys, yes = [], []
        for T in TEMPERATURE_VALUES:
            rec = next((a for a in agg if a["k"] == k
                        and a["epochs"] == HEADLINE_FIXED_EPOCHS and a["T"] == T), None)
            if rec is None:
                ys.append(np.nan); yes.append(0.0)
            else:
                ys.append(rec["unique_fraction_mean"])
                yes.append(rec["unique_fraction_std"])
        color = plt.get_cmap("viridis")(i / max(1, len(K_VALUES) - 1))
        ax.errorbar(TEMPERATURE_VALUES, ys, yerr=yes, fmt="o-",
                    color=color, label=f"k = {k}", capsize=3)
    ax.set_xlabel("sampling temperature $T$")
    ax.set_ylabel(f"unique fraction  (m={M_CANDIDATES})")
    ax.set_title(f"diversity vs T at epochs={HEADLINE_FIXED_EPOCHS}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"M2 Headline A: AR transformer on RCS  (n={n}, depth={DEPTH}, "
        f"{N_SEEDS} circuit seeds, m={M_CANDIDATES} candidates)",
        y=1.02,
    )
    fig.tight_layout()
    out = out_dir / "m2_headline_a.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out}")

    # ---- Summary table ------------------------------------------------------
    print("\n=== aggregated table (mean over seeds) ===")
    print(f"  {'k':>4} {'E':>3} {'T':>4} | {'XEB':>7} {'uniq':>6} "
          f"{'nxeb_r0':>8} {'nxeb_r1':>8} {'nxeb_r2':>8} {'nxeb_r3':>8}")
    for rec in sorted(agg, key=lambda r: (r["k"], r["epochs"], r["T"])):
        print(f"  {rec['k']:>4} {rec['epochs']:>3} {rec['T']:>4.1f} | "
              f"{rec['linear_xeb_mean']:+7.3f} {rec['unique_fraction_mean']:6.3f} "
              f"{rec['nxeb_r0_mean']:+8.3f} {rec['nxeb_r1_mean']:+8.3f} "
              f"{rec['nxeb_r2_mean']:+8.3f} {rec['nxeb_r3_mean']:+8.3f}")
    print("\n  baselines (mean over seeds):")
    for rec in agg_baselines:
        print(f"    {rec['model']:>9}: "
              f"XEB={rec['linear_xeb_mean']:+.3f}  "
              f"nxeb_r1={rec['nxeb_r1_mean']:+.3f}  "
              f"nxeb_r2={rec['nxeb_r2_mean']:+.3f}")


if __name__ == "__main__":
    main()
