"""M3 Stage 1: Conditional AR transformer on Hubbard PBC half-filling.

Trains one AR transformer with scalar conditioning c = U/t on a combined
dataset of |a_x|^2 samples at TRAIN_US. Evaluates by sampling at each U in
EVAL_US (mix of in-distribution and out-of-distribution), ranking generated
candidates by their own model log-likelihood (no leakage of the true GS), and
comparing Delta E_var(S) vs |S| against M1 baselines (random, single-double
excitation closure, greedy CIPSI PT2).

Tests:
  H1.1 (in-dist):  does AR beat closure / greedy at matched |S|?
  H1.2 (OOD):      does AR transfer to held-out U/t?  Train on {2, 4, 8},
                   evaluate at 6 (interpolation) and 12 (extrapolation).
"""

import gc
import resource
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

from aics.baselines.excitation import excitation_closure_with_sizes
from aics.baselines.greedy import greedy_sci_expansion
from aics.baselines.random_uniform import random_subset
from aics.chemistry.amplitude_sampling import (
    bits_to_state_int,
    sample_from_amplitudes,
    state_int_to_bits,
)
from aics.chemistry.hubbard_setup import hubbard_gs_setup
from aics.chemistry.variational import var_eigenvalue
from aics.common.symmetry import sector_states
from aics.models.ar_transformer import (
    ARTransformerConditional,
    train_ar_conditional,
)


# ---- config ----------------------------------------------------------------
L = 6
T_HOP = 1.0
TRAIN_US = [2.0, 4.0, 8.0]
EVAL_US = [4.0, 6.0, 8.0, 12.0]
K_PER_U = 200
M_CANDIDATES = 3000
N_SEEDS = 2
N_EPOCHS = 100
LR = 2e-3
BATCH_SIZE = 32
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
SAMPLE_TEMPERATURE = 1.0
N_RANDOM_TRIALS = 15
# ----------------------------------------------------------------------------

torch.set_num_threads(8)


def pick_seed_index_from_psi(psi, allowed_indices):
    a_sq = np.abs(psi) ** 2
    return int(allowed_indices[int(np.argmax(a_sq[allowed_indices]))])


def occ_string(state_int, L):
    L_mask = (1 << L) - 1
    up = state_int & L_mask
    dn = state_int >> L
    chars = []
    for i in range(L):
        u, d = (up >> i) & 1, (dn >> i) & 1
        chars.append({(0, 0): ".", (1, 0): "u", (0, 1): "d", (1, 1): "2"}[(u, d)])
    return "".join(chars)


def sweep_sizes_for(n_allowed, max_pts=20):
    if n_allowed <= max_pts:
        return list(range(1, n_allowed + 1))
    step = max(1, n_allowed // max_pts)
    pts = sorted(set([1, 2, 4, 8, 16] + list(range(20, n_allowed + 1, step))))
    if pts[-1] != n_allowed:
        pts.append(n_allowed)
    return [p for p in pts if p <= n_allowed]


def evaluate_at_u(setup, sizes, seed_idx, seed_state, L, rng, ar_state_ints_sorted,
                  n_trials_random=N_RANDOM_TRIALS):
    """For a given setup (H, E_0, allowed, ...) and a pre-sorted AR candidate list,
    compute per-baseline Delta E_var curves and the AR curve."""
    H = setup["H"]
    E_0 = setup["E_0"]
    allowed = setup["allowed"]
    states = sector_states(L, L // 2, L // 2)
    state_to_idx = {int(s): i for i, s in enumerate(states)}
    allowed_set = set(int(states[i]) for i in allowed)

    out = {"sizes": list(sizes), "E_0": E_0, "allowed": setup["n_allowed"]}

    # random (mean over trials)
    rand_means = np.zeros(len(sizes))
    rand_stds = np.zeros(len(sizes))
    for k_, sz in enumerate(sizes):
        vals = np.empty(n_trials_random)
        for t in range(n_trials_random):
            S = random_subset(allowed, sz, rng)
            vals[t] = var_eigenvalue(H, S.tolist()) - E_0
        rand_means[k_] = vals.mean()
        rand_stds[k_] = vals.std()
    out["random"] = (rand_means, rand_stds)

    # excitation closure from dominant seed (deterministic)
    closures = excitation_closure_with_sizes(
        [seed_state], L, max_radius=setup["n_allowed"], allowed_set=allowed_set,
    )
    cl_items = sorted(closures.items())
    out["closure_sizes"] = np.array([len(S) for _, S in cl_items])
    out["closure_delta"] = np.array(
        [var_eigenvalue(H, [state_to_idx[int(s)] for s in S]) - E_0
         for _, S in cl_items]
    )

    # greedy SCI from dominant seed
    gr_sizes_l, gr_e, _ = greedy_sci_expansion(H, allowed, seed=seed_idx,
                                               max_size=setup["n_allowed"])
    out["greedy_sizes"] = np.array(gr_sizes_l)
    out["greedy_delta"] = np.array(gr_e) - E_0

    # AR candidates (sorted by model log-prob already)
    ar_indices = np.array([state_to_idx[int(s)] for s in ar_state_ints_sorted
                           if int(s) in state_to_idx])
    # Walk through sizes, picking top |S| AR candidates each time
    ar_delta = np.empty(len(sizes))
    for k_, sz in enumerate(sizes):
        if ar_indices.size >= sz:
            S_idx = ar_indices[:sz].tolist()
        else:
            S_idx = ar_indices.tolist()
        ar_delta[k_] = var_eigenvalue(H, S_idx) - E_0
    out["ar_delta"] = ar_delta

    return out


def run_seed(seed, train_us, eval_us):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    print(f"\n=== seed {seed} ===", flush=True)

    # ---- build training data: jointly sample from |a_x|^2 at each train_U ----
    print(f"  sampling training data: {len(train_us)} U values, "
          f"{K_PER_U} per U...", flush=True)
    all_bits = []
    all_c = []
    train_setups = {}
    states = sector_states(L, L // 2, L // 2)
    for U in train_us:
        setup = hubbard_gs_setup(L, T_HOP, U, pbc=True, verbose=False)
        train_setups[U] = setup
        bits, _, _ = sample_from_amplitudes(setup["psi_0"], states, L,
                                            K_PER_U, rng)
        all_bits.append(bits)
        all_c.append(np.full(K_PER_U, U, dtype=np.float32))
    X_train = np.concatenate(all_bits, axis=0)
    c_train = np.concatenate(all_c, axis=0)

    # ---- train conditional AR ----
    print(f"  training conditional AR (epochs={N_EPOCHS}, d={D_MODEL}, "
          f"layers={N_LAYERS})...", flush=True)
    model = ARTransformerConditional(L=L, n_up=L // 2, n_dn=L // 2,
                                     d_model=D_MODEL, n_layers=N_LAYERS,
                                     n_heads=N_HEADS)
    final_nll = train_ar_conditional(model, X_train, c_train,
                                     n_epochs=N_EPOCHS, lr=LR,
                                     batch_size=BATCH_SIZE,
                                     verbose=True, log_every=30)
    print(f"  final NLL: {final_nll:.4f}", flush=True)

    # ---- evaluate at each U in eval_us ----
    model.eval()
    seed_results = {}
    for U in eval_us:
        # Either reuse training setup or build new (for OOD U)
        if U in train_setups:
            setup = train_setups[U]
        else:
            setup = hubbard_gs_setup(L, T_HOP, U, pbc=True, verbose=False)

        seed_idx = pick_seed_index_from_psi(setup["psi_0"], setup["allowed"])
        seed_state = int(states[seed_idx])

        # Sample M_CANDIDATES candidates from AR conditional on c = U
        c_tensor = torch.full((M_CANDIDATES,), float(U))
        cand_bits = model.sample(c_tensor, temperature=SAMPLE_TEMPERATURE).cpu().numpy()
        cand_int = bits_to_state_int(cand_bits, L)
        # Compute per-candidate log_prob
        with torch.no_grad():
            lp = model.log_prob(
                torch.from_numpy(cand_bits).long(), c_tensor
            ).cpu().numpy()
        # Deduplicate, retain best log-prob per state
        unique_int, first_idx = np.unique(cand_int, return_index=True)
        unique_lp = lp[first_idx]
        # Sort by model's log-prob descending: AR's own best-first ranking
        order = np.argsort(-unique_lp)
        sorted_state_ints = unique_int[order]

        sizes = sweep_sizes_for(setup["n_allowed"], max_pts=25)
        res = evaluate_at_u(setup, sizes, seed_idx, seed_state, L, rng,
                            sorted_state_ints)
        res["U"] = U
        res["seed_state_occ"] = occ_string(seed_state, L)
        res["n_unique_ar"] = len(unique_int)
        in_dist = U in train_us
        res["in_distribution"] = in_dist
        print(f"  eval U={U:>4.1f} ({'in-dist' if in_dist else 'OOD'}):  "
              f"n_unique_AR={len(unique_int)}, "
              f"seed={res['seed_state_occ']}, "
              f"AR_min_dE={res['ar_delta'].min():.4f} at |S|={sizes[int(res['ar_delta'].argmin())]}",
              flush=True)
        seed_results[U] = res

    return seed_results


def aggregate_curves(per_seed_results, eval_us):
    """For each U, mean/std across seeds of each curve."""
    agg = {}
    for U in eval_us:
        per_seed = [r[U] for r in per_seed_results]
        ref = per_seed[0]
        sizes = ref["sizes"]
        ar = np.array([r["ar_delta"] for r in per_seed])
        rand = np.array([r["random"][0] for r in per_seed])
        # closure_sizes can differ across seeds in principle; here baselines use
        # the same seed_state across seeds for given U, so they coincide. Take
        # the first seed's sizes/values.
        cl_sizes = ref["closure_sizes"]
        cl_d = np.array([r["closure_delta"] for r in per_seed])
        gr_sizes = ref["greedy_sizes"]
        gr_d = np.array([r["greedy_delta"] for r in per_seed])
        agg[U] = {
            "sizes": sizes,
            "ar_mean": ar.mean(axis=0), "ar_std": ar.std(axis=0),
            "rand_mean": rand.mean(axis=0), "rand_std": rand.std(axis=0),
            "cl_sizes": cl_sizes,
            "cl_mean": cl_d.mean(axis=0), "cl_std": cl_d.std(axis=0),
            "gr_sizes": gr_sizes,
            "gr_mean": gr_d.mean(axis=0), "gr_std": gr_d.std(axis=0),
            "in_distribution": ref["in_distribution"],
            "allowed": ref["allowed"],
        }
    return agg


def main():
    print(f"=== M3 Stage 1: conditional AR on Hubbard L={L} PBC half-filling ===")
    print(f"  TRAIN_US = {TRAIN_US}   EVAL_US = {EVAL_US}")
    print(f"  K_PER_U = {K_PER_U}   N_EPOCHS = {N_EPOCHS}   "
          f"M_CANDIDATES = {M_CANDIDATES}   N_SEEDS = {N_SEEDS}")

    per_seed = []
    for s in range(N_SEEDS):
        print(f"  [RSS pre-seed {s}: {_rss_mb():.0f} MB]", flush=True)
        per_seed.append(run_seed(s, TRAIN_US, EVAL_US))
        gc.collect()
        print(f"  [RSS post-seed {s}: {_rss_mb():.0f} MB]", flush=True)
    agg = aggregate_curves(per_seed, EVAL_US)

    # ---- plot: 1 panel per eval U ------------------------------------------
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, len(EVAL_US), figsize=(5.0 * len(EVAL_US), 4.4),
                            sharey=False)
    if len(EVAL_US) == 1:
        axs = [axs]
    for ax, U in zip(axs, EVAL_US):
        a = agg[U]
        sizes = np.array(a["sizes"])
        ax.errorbar(sizes, a["rand_mean"], yerr=a["rand_std"],
                    fmt="o-", color="#888", markersize=3, alpha=0.7, label="random",
                    capsize=2)
        ax.errorbar(a["cl_sizes"], a["cl_mean"], yerr=a["cl_std"],
                    fmt="d-", color="#2a9d8f", markersize=5,
                    label="excitation closure", capsize=2)
        ax.errorbar(a["gr_sizes"], a["gr_mean"], yerr=a["gr_std"],
                    fmt="-", color="#e76f51", linewidth=2,
                    label="greedy SCI (PT2)")
        ax.errorbar(sizes, a["ar_mean"], yerr=a["ar_std"],
                    fmt="s-", color="#264653", markersize=4, linewidth=1.6,
                    label="AR (conditional)", capsize=2)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel(r"$|\mathcal{S}|$")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.grid(alpha=0.3)
        tag = "in-dist" if a["in_distribution"] else "OOD"
        ax.set_title(f"U/t = {U}  ({tag}, allowed = {a['allowed']})")
    axs[0].set_ylabel(r"$\Delta E_{\rm var}(\mathcal{S}) - E_0$  [t]")
    axs[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"M3 Stage 1: conditional AR vs baselines, Hubbard L={L} PBC half-filling   "
        f"(train U/t in {TRAIN_US})",
        y=1.02,
    )
    fig.tight_layout()
    out = out_dir / "m3_stage1_ar_conditional.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out}")

    # ---- compact summary table --------------------------------------------
    print("\n=== compact summary: Delta E_var at three reference |S| values ===")
    print(f"  {'U':>5} {'kind':>9} {'|S|':>5} {'random':>10} "
          f"{'closure':>10} {'greedy':>10} {'AR':>10}")
    for U in EVAL_US:
        a = agg[U]
        n_allowed = a["allowed"]
        ref_sizes = [10, max(20, n_allowed // 4), n_allowed]
        for rs in ref_sizes:
            sizes = np.array(a["sizes"])
            k_rand = int(np.argmin(np.abs(sizes - rs)))
            k_cl = int(np.argmin(np.abs(np.array(a["cl_sizes"]) - rs)))
            k_gr = int(np.argmin(np.abs(np.array(a["gr_sizes"]) - rs)))
            tag = "in-dist" if a["in_distribution"] else "OOD"
            print(f"  {U:>5.1f} {tag:>9} {rs:>5} "
                  f"{a['rand_mean'][k_rand]:>+10.4f} "
                  f"{a['cl_mean'][k_cl]:>+10.4f} "
                  f"{a['gr_mean'][k_gr]:>+10.4f} "
                  f"{a['ar_mean'][k_rand]:>+10.4f}")


if __name__ == "__main__":
    main()
