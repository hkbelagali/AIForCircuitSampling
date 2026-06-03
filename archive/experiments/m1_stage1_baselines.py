"""M1: Stage 1 Task A baselines (extended).

(A) Main figure: 3-panel ΔE_var vs |S| for L in {4, 6, 8}, U/t = 4, PBC half-filling.
(B) U-sweep figure: greedy SCI ΔE_var vs |S| at L=6 for U/t in U_SWEEP.

Baseline lineup (per DESIGN.md M1, simplified): random / excitation closure /
neighborhood random-walk / greedy CIPSI-PT2. All restricted to the GS irrep
allowed support (M0.5 machinery).

Seed: the dominant GS configuration (argmax |a_x|^2 within allowed). This is
a strong prior for the baselines — the ML model in M3 must beat *this*, not
the weaker min-H_diag seed.

L=8 cap: |S| sweep capped at S_CAP_L8 since the sector dim is 4900 and full
greedy convergence would take hours. We're after the early-|S| ordering, not
exhaustive convergence.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from aics.baselines.excitation import excitation_closure_with_sizes
from aics.baselines.greedy import greedy_sci_expansion
from aics.baselines.neighborhood import random_walk_neighborhood
from aics.baselines.random_uniform import random_subset
from aics.chemistry.hubbard_setup import hubbard_gs_setup
from aics.chemistry.variational import var_eigenvalue
from aics.common.symmetry import sector_states


L_VALS = [4, 6, 8]
U_VAL = 4.0
T_HOP = 1.0
N_RANDOM_TRIALS = 20
N_NEIGHBORHOOD_TRIALS = 10
U_SWEEP = [0.5, 1.0, 2.0, 4.0, 8.0]
S_CAP_L8 = 300


def pick_seed_index_from_psi(psi_0, allowed_indices):
    """Dominant GS configuration restricted to allowed support."""
    a_sq = np.abs(psi_0) ** 2
    a_sq_allowed = a_sq[allowed_indices]
    return int(allowed_indices[int(np.argmax(a_sq_allowed))])


def sweep_sizes_for(n_allowed, cap=None):
    if cap is None:
        cap = n_allowed
    n = min(n_allowed, cap)
    if n <= 30:
        return list(range(1, n + 1))
    pts = sorted(set(
        list(range(1, 11)) + [12, 16, 20, 25, 30]
        + list(range(40, n + 1, max(1, n // 15)))
    ))
    if pts[-1] != n:
        pts.append(n)
    return [p for p in pts if p <= n]


def occ_string(state_int, L):
    up = state_int & ((1 << L) - 1)
    dn = state_int >> L
    chars = []
    for i in range(L):
        u = (up >> i) & 1
        d = (dn >> i) & 1
        chars.append({(0, 0): ".", (1, 0): "u", (0, 1): "d", (1, 1): "2"}[(u, d)])
    return "".join(chars)


def run_baselines(L, U, rng, cap=None):
    print(f"\n--- L={L}, U/t={U}, PBC half-filling ---")
    setup = hubbard_gs_setup(L, T_HOP, U, pbc=True, verbose=True)
    H, E_0, allowed = setup["H"], setup["E_0"], setup["allowed"]
    n_allowed = setup["n_allowed"]
    print(f"  sector dim  = {H.shape[0]}   allowed = {n_allowed}   E_0 = {E_0:.6f}")
    print(f"  irrep eigs  = {setup['irrep_eigs']}    used_T = {setup['used_T']}")

    states = sector_states(L, L // 2, L // 2)
    state_to_idx = {int(s): i for i, s in enumerate(states)}
    allowed_state_set = set(int(s) for s in states[allowed])

    seed_idx = pick_seed_index_from_psi(setup["psi_0"], allowed)
    seed_state = int(states[seed_idx])
    seed_amp_sq = float(np.abs(setup["psi_0"][seed_idx]) ** 2)
    print(f"  dominant seed = idx {seed_idx}, state {occ_string(seed_state, L)}, "
          f"|a_x|^2 = {seed_amp_sq:.4f}")

    sizes = sweep_sizes_for(n_allowed, cap=cap)
    max_size = sizes[-1]
    print(f"  |S| sweep: {sizes[0]} .. {sizes[-1]} ({len(sizes)} points)")

    print(f"  random ({N_RANDOM_TRIALS} trials/size)...", flush=True)
    rand_m = np.zeros(len(sizes)); rand_s = np.zeros(len(sizes))
    for k, sz in enumerate(sizes):
        vals = np.empty(N_RANDOM_TRIALS)
        for t in range(N_RANDOM_TRIALS):
            S = random_subset(allowed, sz, rng)
            vals[t] = var_eigenvalue(H, S.tolist()) - E_0
        rand_m[k], rand_s[k] = vals.mean(), vals.std()

    print(f"  neighborhood ({N_NEIGHBORHOOD_TRIALS} trials/size)...", flush=True)
    nbhd_m = np.zeros(len(sizes)); nbhd_s = np.zeros(len(sizes))
    for k, sz in enumerate(sizes):
        vals = np.empty(N_NEIGHBORHOOD_TRIALS)
        for t in range(N_NEIGHBORHOOD_TRIALS):
            S_states = random_walk_neighborhood(seed_state, sz, allowed_state_set,
                                                 L, rng)
            S_idx = [state_to_idx[int(s)] for s in S_states]
            vals[t] = var_eigenvalue(H, S_idx) - E_0
        nbhd_m[k], nbhd_s[k] = vals.mean(), vals.std()

    print(f"  excitation closure...", flush=True)
    closures = excitation_closure_with_sizes(
        [seed_state], L, max_radius=max_size, allowed_set=allowed_state_set
    )
    cl_items = sorted(closures.items())
    cl_sizes = np.array([len(S) for _, S in cl_items])
    cl_d = np.array([var_eigenvalue(H, [state_to_idx[int(s)] for s in S]) - E_0
                     for _, S in cl_items])
    print(f"    closure sizes: {cl_sizes.tolist()}")

    print(f"  greedy SCI (max_size={max_size})...", flush=True)
    gr_sizes_l, gr_e, _ = greedy_sci_expansion(H, allowed, seed=seed_idx,
                                               max_size=max_size)
    gr_sizes = np.array(gr_sizes_l)
    gr_d = np.array(gr_e) - E_0
    if (gr_d < 1e-4).any():
        first = int(gr_sizes[int(np.argmax(gr_d < 1e-4))])
        print(f"    greedy reaches dE_var < 1e-4 at |S| = {first}")
    else:
        print(f"    greedy min dE_var = {gr_d.min():.6f} at |S| = "
              f"{gr_sizes[gr_d.argmin()]}")

    return dict(L=L, U=U, n_allowed=n_allowed, E_0=E_0,
                sizes=np.array(sizes), rand_m=rand_m, rand_s=rand_s,
                nbhd_m=nbhd_m, nbhd_s=nbhd_s,
                cl_sizes=cl_sizes, cl_d=cl_d, gr_sizes=gr_sizes, gr_d=gr_d)


def main():
    rng = np.random.default_rng(0)
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)

    # --- Main figure: L sweep at U/t = 4 ----------------------------------
    print("\n========== MAIN FIGURE: L sweep at U/t = 4 ==========")
    main_results = []
    for L in L_VALS:
        cap = S_CAP_L8 if L == 8 else None
        main_results.append(run_baselines(L, U_VAL, rng, cap=cap))

    fig, axs = plt.subplots(1, len(L_VALS), figsize=(5.5 * len(L_VALS), 4.3),
                            sharey=False)
    if len(L_VALS) == 1:
        axs = [axs]
    for ax, r in zip(axs, main_results):
        ax.errorbar(r["sizes"], r["rand_m"], yerr=r["rand_s"], label="random",
                    fmt="o-", color="#888", markersize=3, alpha=0.7, capsize=2)
        ax.errorbar(r["sizes"], r["nbhd_m"], yerr=r["nbhd_s"],
                    label="neighborhood (random walk)",
                    fmt="s-", color="#f4a261", markersize=3, alpha=0.85, capsize=2)
        ax.plot(r["cl_sizes"], r["cl_d"], "d-", color="#2a9d8f",
                label="excitation closure", markersize=6)
        ax.plot(r["gr_sizes"], r["gr_d"], "-", color="#e76f51",
                label="greedy SCI (PT2)", linewidth=2)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel(r"$|\mathcal{S}|$")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.grid(alpha=0.3)
        cap_note = " (capped)" if r["L"] == 8 else ""
        ax.set_title(f"L = {r['L']}, allowed = {r['n_allowed']}{cap_note}")
    axs[0].set_ylabel(r"$\Delta E_{\rm var}(\mathcal{S}) = E_{\rm var} - E_0$  [t]")
    axs[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"M1: Stage 1 Task A baselines, 1D Hubbard PBC half-filling, "
                 f"U/t = {U_VAL}", y=1.02)
    fig.tight_layout()
    out_main = out_dir / "m1_stage1_baselines.png"
    fig.savefig(out_main, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out_main}")

    # --- U-sweep figure: greedy curves at L = 6 ---------------------------
    print("\n========== U-SWEEP FIGURE: L = 6 ==========")
    u_results = []
    for U in U_SWEEP:
        u_results.append(run_baselines(L=6, U=U, rng=rng))

    fig2, ax = plt.subplots(figsize=(7.5, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(u_results):
        color = cmap(i / max(1, len(u_results) - 1))
        ax.plot(r["gr_sizes"], r["gr_d"], "-", color=color, linewidth=1.6,
                label=f"U/t = {r['U']:.1f}  (allowed={r['n_allowed']})")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel(r"$|\mathcal{S}|$")
    ax.set_ylabel(r"$\Delta E_{\rm var}(\mathcal{S})$  [t]")
    ax.set_title(r"M1 U-sweep: greedy SCI at L=6 PBC half-filling")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig2.tight_layout()
    out_u = out_dir / "m1_u_sweep_l6.png"
    fig2.savefig(out_u, dpi=150)
    print(f"\nWrote {out_u}")


if __name__ == "__main__":
    main()
