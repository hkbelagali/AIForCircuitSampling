"""M8: three-curve replication of the Moss et al. "data + VMC" story at L=6.

For a single (L, U/t, k), train an RNN wavefunction in three modes:
  (1) data-only        : MLE pretrain on k samples from |psi_0|^2.
  (2) vmc-only         : random init, then Adam-VMC for N_VMC steps.
  (3) data + vmc       : MLE pretrain, then Adam-VMC.

Plot <E>/L (exact, via the full-sector evaluator) vs VMC step, one curve per
mode, shaded across seeds. Horizontal reference at E_0/L.

The VMC step is Adam on the energy-gradient surrogate
    L(theta) = mean_i log|psi(x_i)| * (E_loc(x_i) - <E>).detach()
which gives the true grad_<E> in one backward pass per step (vs the SR
per-sample-backward loop). ~60-250x faster per step.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from aics.chemistry.amplitude_sampling import sample_from_amplitudes
from aics.chemistry.local_energy import local_energy_hubbard, make_hubbard_context
from aics.eval.energy import model_energy_exact
from aics.models.ar_rnn import ARRNN
from aics.training.mle_pretrain import train_rnn_mle
from aics.training.vmc import VMCConfig, VMCState, vmc_step

torch.set_num_threads(4)

# ---- config ----------------------------------------------------------------
L = 6
U = 4.0
T_HOP = 1.0
K_DATA = 200
N_PRETRAIN_EPOCHS = 100
PRETRAIN_LR = 2e-3
PRETRAIN_BATCH = 32
N_VMC = 3000
EVAL_EVERY = 50
SEEDS = [0, 1, 2]
D_HIDDEN = 32
N_LAYERS = 1
VMC_LR = 3e-3
VMC_BATCH = 256
# ----------------------------------------------------------------------------

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "m8_results.json"


def run_one(L, U, mode, seed):
    """Returns dict with steps (list of int, including 0 for the post-pretrain checkpoint),
    energies (list of float, <E>/L at each checkpoint), and final dE/|E_0|."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ctx = make_hubbard_context(L, T_HOP, U, pbc=True)

    model = ARRNN(L=L, n_up=L // 2, n_dn=L // 2, d_hidden=D_HIDDEN, n_layers=N_LAYERS)

    steps, energies = [], []

    if "data" in mode:
        bits, _, _ = sample_from_amplitudes(ctx.psi_0, ctx.states, L, K_DATA, rng)
        train_rnn_mle(model, bits, n_epochs=N_PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
                      batch_size=PRETRAIN_BATCH)

    # Always record the t=0 checkpoint (post-pretrain or random init).
    steps.append(0)
    energies.append(model_energy_exact(model, ctx) / L)

    if "vmc" in mode:
        cfg = VMCConfig(lr=VMC_LR, batch_size=VMC_BATCH, clip_norm=1.0)
        state = VMCState()
        for step in range(1, N_VMC + 1):
            vmc_step(model, ctx, state, cfg, local_energy_hubbard)
            if step % EVAL_EVERY == 0:
                steps.append(step)
                energies.append(model_energy_exact(model, ctx) / L)

    E_final = energies[-1] * L
    dE = (E_final - ctx.E_0) / abs(ctx.E_0)
    return {"steps": steps, "energies": energies, "E0_per_L": ctx.E_0 / L,
            "dE_rel_final": dE, "mode": mode, "seed": seed,
            "E_0": ctx.E_0, "L": L}


def main():
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results = {}
    for mode in ("data", "vmc", "data_vmc"):
        for s in SEEDS:
            key = f"{mode}_s{s}"
            print(f"  running {key}...", flush=True)
            r = run_one(L, U, mode, s)
            results[key] = r
            print(f"    final dE/|E_0| = {r['dE_rel_final']:+.4e}", flush=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}", flush=True)

    # ---- plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    mode_style = {
        "data":      ("data only",        "#888",    "o", "-"),
        "vmc":       ("VMC only",         "#e76f51", "s", "-"),
        "data_vmc":  ("data + VMC",       "#2a9d8f", "D", "-"),
    }
    E0_per_L = results[f"data_s{SEEDS[0]}"]["E0_per_L"]
    ax.axhline(E0_per_L, color="black", ls="--", lw=0.8, alpha=0.6,
               label=r"$E_0/L$")
    for mode, (label, color, marker, ls) in mode_style.items():
        # Collect per-seed traces; for data-only mode each is just a single horizontal point.
        traces = [results[f"{mode}_s{s}"] for s in SEEDS]
        if mode == "data":
            ys = np.array([r["energies"][0] for r in traces])
            ax.axhline(float(ys.mean()), color=color, ls=":", lw=1.4, alpha=0.8,
                       label=f"{label} ($k={K_DATA}$, MLE only)")
        else:
            # All traces share the same step grid (deterministic) modulo seeds.
            steps = np.array(traces[0]["steps"])
            E = np.array([r["energies"] for r in traces])  # (n_seeds, n_pts)
            med = np.median(E, axis=0); lo = np.min(E, axis=0); hi = np.max(E, axis=0)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.18)
            ax.plot(steps, med, ls=ls, marker=marker, color=color, ms=4, lw=1.7,
                    label=label)
    ax.set_xlabel("VMC step")
    ax.set_ylabel(r"$\langle E\rangle / L$")
    ax.set_title(f"M8: data / VMC / data+VMC on Hubbard $L={L}$, $U/t={U}$")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_PATH.parent / "m8_rnn_vmc_replication.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
