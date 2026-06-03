"""M15: M9 pipeline with in-line tracking of weight-<= W Pauli expectation
errors. ED signs (ctx.signs) as the oracle; only magnitudes are learned.
For each weight w in 0..W, log:
  - max |<P>_model - <P>_true| over Paulis with weight <= w, at every
    EVAL_EVERY checkpoint
  - steps_to_Pauli_threshold(w) = first step where max-err <= 0.01

Per cell:  python experiments/m15_pauli_tracking.py --L 6 --k 100 --seed 3
Outputs:   results/m15_cells/L{L}_k{k}_s{seed}.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aics.chemistry.amplitude_sampling import (
    sample_from_amplitudes, state_int_to_bits,
)
from aics.chemistry.local_energy import local_energy_hubbard, make_hubbard_context
from aics.chemistry.pauli_observables import (
    build_pauli_ops, pauli_expectations, max_abs_error_cumulative,
)
from aics.eval.energy import model_energy_exact
from aics.models.ar_rnn import ARRNN
from aics.training.mle_pretrain import train_rnn_mle
from aics.training.vmc import VMCConfig, VMCState, vmc_step


U_DEFAULT = 4.0
T_HOP = 1.0
THRESHOLD_E = 0.01      # energy relative threshold (M9 metric)
THRESHOLD_P = 0.01      # Pauli absolute threshold
MAX_VMC_STEPS = 5000
EVAL_EVERY = 10
N_PRETRAIN_EPOCHS = 100
PRETRAIN_LR = 2e-3
PRETRAIN_BATCH = 32
VMC_LR = 3e-3
VMC_BATCH = 256
D_HIDDEN = 32
N_LAYERS = 1
DEFAULT_MAX_WEIGHT = 3

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m15_cells"


def _psi_from_model(model, ctx, states_bits):
    """Compute the model's full sector wavefunction Psi = signs * |Psi_theta|,
    normalized. Returns (D,) float64."""
    model.eval()
    with torch.no_grad():
        log_mag = model.log_psi_mag(
            torch.from_numpy(states_bits.astype(np.int64)).long()
        ).cpu().numpy()
    psi = ctx.signs.astype(np.float64) * np.exp(log_mag)
    norm = float(np.sqrt(np.sum(psi * psi)))
    if norm == 0.0:
        return psi  # degenerate; let callers handle
    return psi / norm


def run_cell(L, k, seed, U=U_DEFAULT, max_steps=MAX_VMC_STEPS,
             threshold=THRESHOLD_E, threshold_pauli=THRESHOLD_P,
             eval_every=EVAL_EVERY, max_weight=DEFAULT_MAX_WEIGHT,
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
    states_bits = state_int_to_bits(ctx.states, L)

    # Precompute Pauli ops + true expectations (oracle baseline).
    ops, ops_meta = build_pauli_ops(ctx, max_weight=max_weight)
    psi_true = ctx.signs.astype(np.float64) * np.abs(ctx.psi_0)
    psi_true = psi_true / np.sqrt(np.sum(psi_true ** 2))
    vals_true = pauli_expectations(ops, psi_true)

    model = ARRNN(L=L, n_up=L // 2, n_dn=L // 2,
                  d_hidden=D_HIDDEN, n_layers=N_LAYERS)

    if k > 0:
        bits, _, _ = sample_from_amplitudes(ctx.psi_0, ctx.states, L, k, rng)
        train_rnn_mle(model, bits, n_epochs=N_PRETRAIN_EPOCHS, lr=PRETRAIN_LR,
                      batch_size=PRETRAIN_BATCH)

    # ---- per-checkpoint logging ----
    steps_log = [0]
    energies = []
    pauli_max_err = {w: [] for w in range(0, max_weight + 1)}

    def checkpoint(step_count):
        E = model_energy_exact(model, ctx)
        energies.append(E)
        psi_model = _psi_from_model(model, ctx, states_bits)
        vals_model = pauli_expectations(ops, psi_model)
        err = max_abs_error_cumulative(ops_meta, vals_model, vals_true,
                                       max_weight=max_weight)
        for w in range(0, max_weight + 1):
            pauli_max_err[w].append(err.get(w, float("nan")))

    checkpoint(0)
    rel0 = (energies[0] - ctx.E_0) / abs(ctx.E_0)
    steps_to_E_threshold = 0 if abs(rel0) <= threshold else None
    steps_to_P_threshold = {
        w: (0 if pauli_max_err[w][0] <= threshold_pauli else None)
        for w in range(0, max_weight + 1)
    }

    cfg = VMCConfig(lr=VMC_LR, batch_size=VMC_BATCH, clip_norm=1.0)
    state = VMCState()

    last_step = 0
    for step in range(1, max_steps + 1):
        vmc_step(model, ctx, state, cfg, local_energy_hubbard)
        if step % eval_every == 0 or step == max_steps:
            checkpoint(step)
            steps_log.append(step)
            last_step = step
            rel = (energies[-1] - ctx.E_0) / abs(ctx.E_0)
            if steps_to_E_threshold is None and abs(rel) <= threshold:
                steps_to_E_threshold = step
            for w in range(0, max_weight + 1):
                if steps_to_P_threshold[w] is None and \
                        pauli_max_err[w][-1] <= threshold_pauli:
                    steps_to_P_threshold[w] = step
            # Early-stop only when ALL targets reached
            all_done = (steps_to_E_threshold is not None) and \
                all(v is not None for v in steps_to_P_threshold.values())
            if all_done:
                break

    elapsed = time.time() - t0
    record = {
        "L": L, "k": k, "seed": seed, "U": U,
        "threshold_E": threshold, "threshold_pauli": threshold_pauli,
        "n_up": L // 2, "n_dn": L // 2,
        "max_steps": max_steps, "eval_every": eval_every,
        "max_weight": max_weight,
        "n_pauli_ops": len(ops),
        "pauli_count_by_weight": {w: len(idxs)
                                   for w, idxs in ops_meta["by_weight"].items()},
        "E_0": ctx.E_0,
        "steps": steps_log,
        "energies": energies,
        "pauli_max_err_by_weight": pauli_max_err,
        "steps_to_E_threshold": steps_to_E_threshold,
        "steps_to_P_threshold": steps_to_P_threshold,
        "reached_E": steps_to_E_threshold is not None,
        "elapsed_sec": elapsed,
        "d_hidden": D_HIDDEN, "n_layers": N_LAYERS,
        "vmc_lr": VMC_LR, "vmc_batch": VMC_BATCH,
        "pretrain_epochs": N_PRETRAIN_EPOCHS, "pretrain_lr": PRETRAIN_LR,
    }
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    cell_path.write_text(json.dumps(record))
    if verbose:
        print(f"  L={L} k={k} s={seed}: E_step={steps_to_E_threshold}  "
              f"P_steps={steps_to_P_threshold}  ({elapsed:.1f}s)", flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-weight", type=int, default=DEFAULT_MAX_WEIGHT)
    p.add_argument("--max-steps", type=int, default=MAX_VMC_STEPS)
    args = p.parse_args()
    run_cell(args.L, args.k, args.seed, max_weight=args.max_weight,
             max_steps=args.max_steps)


if __name__ == "__main__":
    main()
