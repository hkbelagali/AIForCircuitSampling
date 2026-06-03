"""M9: per-cell driver for the sample-complexity sweep.

One invocation = one cell:

    python experiments/m9_sample_complexity_sweep.py --L 6 --k 36 --seed 3

Pipeline per cell (Protocol B, independent end-to-end seed):
  1. Hubbard setup via make_hubbard_context.
  2. Draw k bitstrings from |psi_0|^2 (k > 0); skip if k == 0 (pure VMC baseline).
  3. MLE pretrain via train_rnn_mle.
  4. Adam-VMC loop until |<E> - E_0|/|E_0| <= THRESHOLD or MAX_VMC_STEPS reached.
  5. Write per-cell JSON to results/m9_cells/L{L}_k{k}_s{seed}.json with the
     step at which threshold was first met (or null if never), plus the full
     energy trace and metadata.

Cells with an existing JSON are skipped (resumable).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aics.chemistry.amplitude_sampling import sample_from_amplitudes
from aics.chemistry.local_energy import local_energy_hubbard, make_hubbard_context
from aics.eval.energy import model_energy_exact
from aics.models.ar_rnn import ARRNN
from aics.training.mle_pretrain import train_rnn_mle
from aics.training.vmc import VMCConfig, VMCState, vmc_step


# ---- defaults (mirror the M8 setup) ----------------------------------------
U_DEFAULT = 4.0
T_HOP = 1.0
THRESHOLD = 0.01
MAX_VMC_STEPS = 5000
EVAL_EVERY = 10
N_PRETRAIN_EPOCHS = 100
PRETRAIN_LR = 2e-3
PRETRAIN_BATCH = 32
VMC_LR = 3e-3
VMC_BATCH = 256
D_HIDDEN = 32
N_LAYERS = 1
# ----------------------------------------------------------------------------

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m9_cells"


def run_cell(L, k, seed, U=U_DEFAULT, max_steps=MAX_VMC_STEPS,
             threshold=THRESHOLD, eval_every=EVAL_EVERY,
             verbose=True):
    cell_path = CELLS_DIR / f"L{L}_k{k}_s{seed}.json"
    if cell_path.exists():
        if verbose:
            print(f"  skip (exists): {cell_path}", flush=True)
        return json.loads(cell_path.read_text())

    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    ctx = make_hubbard_context(L, T_HOP, U, pbc=True)
    model = ARRNN(L=L, n_up=L // 2, n_dn=L // 2, d_hidden=D_HIDDEN, n_layers=N_LAYERS)

    if k > 0:
        bits, _, _ = sample_from_amplitudes(ctx.psi_0, ctx.states, L, k, rng)
        train_rnn_mle(model, bits, n_epochs=N_PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
                      batch_size=PRETRAIN_BATCH)

    # Initial checkpoint (post-pretrain or random init).
    steps = [0]
    energies = [model_energy_exact(model, ctx)]
    dE0 = (energies[0] - ctx.E_0) / abs(ctx.E_0)
    steps_to_threshold = 0 if abs(dE0) <= threshold else None

    cfg = VMCConfig(lr=VMC_LR, batch_size=VMC_BATCH, clip_norm=1.0)
    state = VMCState()

    for step in range(1, max_steps + 1):
        vmc_step(model, ctx, state, cfg, local_energy_hubbard)
        if step % eval_every == 0 or step == max_steps:
            E = model_energy_exact(model, ctx)
            steps.append(step)
            energies.append(E)
            rel = (E - ctx.E_0) / abs(ctx.E_0)
            if steps_to_threshold is None and abs(rel) <= threshold:
                steps_to_threshold = step
                # Don't break; continue to record full trace, but a more
                # CPU-efficient run could early-stop here. We early-stop:
                break

    elapsed = time.time() - t0
    record = {
        "L": L, "k": k, "seed": seed, "U": U, "threshold": threshold,
        "n_up": L // 2, "n_dn": L // 2,
        "max_steps": max_steps,
        "eval_every": eval_every,
        "E_0": ctx.E_0,
        "steps": steps,
        "energies": energies,
        "steps_to_threshold": steps_to_threshold,
        "reached": steps_to_threshold is not None,
        "elapsed_sec": elapsed,
        "d_hidden": D_HIDDEN, "n_layers": N_LAYERS,
        "vmc_lr": VMC_LR, "vmc_batch": VMC_BATCH,
        "pretrain_epochs": N_PRETRAIN_EPOCHS, "pretrain_lr": PRETRAIN_LR,
        "pretrain_batch": PRETRAIN_BATCH,
    }
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    cell_path.write_text(json.dumps(record))
    if verbose:
        st = record["steps_to_threshold"]
        print(f"  L={L} k={k} s={seed}: steps_to_threshold = "
              f"{st if st is not None else 'NOT REACHED'}  "
              f"(elapsed {elapsed:.1f}s)", flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--U", type=float, default=U_DEFAULT)
    p.add_argument("--max-steps", type=int, default=MAX_VMC_STEPS)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    args = p.parse_args()
    run_cell(args.L, args.k, args.seed, U=args.U,
             max_steps=args.max_steps, threshold=args.threshold)


if __name__ == "__main__":
    main()
