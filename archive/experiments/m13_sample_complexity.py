"""M13: per-cell sample-complexity driver for the 1D Heisenberg AFM, mirroring
m9_sample_complexity_sweep.py but in the spin basis with closed-form Marshall
signs (no ED dependency for sign machinery).

One invocation = one cell:
    python experiments/m13_sample_complexity.py --L 6 --k 100 --seed 3

Outputs results/m13_cells/L{L}_k{k}_s{seed}.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aics.spin.heisenberg import (
    make_heisenberg_context, sample_from_amplitudes_spin, state_int_to_bits_spin,
)
from aics.spin.local_energy import local_energy_heisenberg
from aics.eval.energy import model_energy_exact_spin
from aics.models.ar_rnn_spin import ARRNNSpin
from aics.training.mle_pretrain import train_rnn_mle
from aics.training.vmc import VMCConfig, VMCState, vmc_step


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

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m13_cells"


def run_cell(L, k, seed, J=1.0, max_steps=MAX_VMC_STEPS,
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
    ctx = make_heisenberg_context(L, J=J, pbc=True)
    model = ARRNNSpin(L=L, n_up=L // 2, d_hidden=D_HIDDEN, n_layers=N_LAYERS)

    if k > 0:
        bits, _, _ = sample_from_amplitudes_spin(ctx.psi_0, ctx.states, L, k, rng)
        train_rnn_mle(model, bits, n_epochs=N_PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
                      batch_size=PRETRAIN_BATCH)

    steps = [0]
    energies = [model_energy_exact_spin(model, ctx)]
    dE0 = (energies[0] - ctx.E_0) / abs(ctx.E_0)
    steps_to_threshold = 0 if abs(dE0) <= threshold else None

    cfg = VMCConfig(lr=VMC_LR, batch_size=VMC_BATCH, clip_norm=1.0)
    state = VMCState()

    for step in range(1, max_steps + 1):
        vmc_step(model, ctx, state, cfg, local_energy_heisenberg)
        if step % eval_every == 0 or step == max_steps:
            E = model_energy_exact_spin(model, ctx)
            steps.append(step)
            energies.append(E)
            rel = (E - ctx.E_0) / abs(ctx.E_0)
            if steps_to_threshold is None and abs(rel) <= threshold:
                steps_to_threshold = step
                break

    elapsed = time.time() - t0
    record = {
        "L": L, "k": k, "seed": seed, "J": J, "threshold": threshold,
        "n_up": L // 2,
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
        "model_class": "ARRNNSpin",
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
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=MAX_VMC_STEPS)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    args = p.parse_args()
    run_cell(args.L, args.k, args.seed, J=args.J,
             max_steps=args.max_steps, threshold=args.threshold)


if __name__ == "__main__":
    main()
