"""At fixed L, run multiple seeds of signed-learning Hubbard and report median
steps-to-threshold across k. Compare directly to the unsigned-M9 cells.
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
from aics.chemistry.local_energy import make_hubbard_context
from aics.chemistry.local_energy_signed import local_energy_hubbard_signed
from aics.eval.energy import model_energy_exact_signed
from aics.models.ar_rnn import ARRNN
from aics.training.mle_pretrain import train_rnn_mle
from aics.training.vmc import VMCConfig, VMCState, vmc_step


D_HIDDEN = 32
THRESHOLD = 0.01
MAX_VMC_STEPS = 5000
EVAL_EVERY = 10

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m14_cells"


def run_cell(L, k, seed, verbose=False):
    out = CELLS_DIR / f"L{L}_k{k}_s{seed}.json"
    if out.exists():
        return json.loads(out.read_text())

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    ctx = make_hubbard_context(L, 1.0, 4.0, pbc=True)
    model = ARRNN(L=L, n_up=L // 2, n_dn=L // 2,
                  d_hidden=D_HIDDEN, learn_signs=True)
    if k > 0:
        bits, _, _ = sample_from_amplitudes(ctx.psi_0, ctx.states, L, k, rng)
        train_rnn_mle(model, bits, n_epochs=100, lr=2e-3, batch_size=32)

    steps = [0]
    E0 = model_energy_exact_signed(model, ctx)
    energies = [E0]
    rel0 = (E0 - ctx.E_0) / abs(ctx.E_0)
    steps_to_threshold = 0 if abs(rel0) <= THRESHOLD else None

    cfg = VMCConfig(lr=3e-3, batch_size=256, clip_norm=1.0)
    state = VMCState()
    for step in range(1, MAX_VMC_STEPS + 1):
        vmc_step(model, ctx, state, cfg, local_energy_hubbard_signed, signed=True)
        if step % EVAL_EVERY == 0 or step == MAX_VMC_STEPS:
            E = model_energy_exact_signed(model, ctx)
            steps.append(step); energies.append(E)
            rel = (E - ctx.E_0) / abs(ctx.E_0)
            if steps_to_threshold is None and abs(rel) <= THRESHOLD:
                steps_to_threshold = step
                break

    record = {
        "L": L, "k": k, "seed": seed,
        "steps_to_threshold": steps_to_threshold,
        "reached": steps_to_threshold is not None,
        "max_steps": MAX_VMC_STEPS,
        "elapsed_sec": time.time() - t0,
        "model_class": "ARRNN(learn_signs=True, 2-layer sign head)",
    }
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    if verbose:
        print(f"  L={L} k={k} s={seed}: step={steps_to_threshold}  "
              f"({record['elapsed_sec']:.0f}s)", flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--k-list", type=str,
                   default="68,128,200,320,500,800,1250,2000",
                   help="comma-separated list of k values")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7")
    args = p.parse_args()

    ks = [int(x) for x in args.k_list.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    L = args.L

    print(f"L={L}  k_list={ks}  seeds={seeds}", flush=True)
    by_k = {}
    for k in ks:
        results = []
        for s in seeds:
            r = run_cell(L, k, s, verbose=True)
            results.append(r["steps_to_threshold"] if r["steps_to_threshold"] is not None
                           else MAX_VMC_STEPS)
        results = np.asarray(results, dtype=float)
        cens = int(np.sum(results >= MAX_VMC_STEPS))
        by_k[k] = (float(np.median(results)),
                   float(np.quantile(results, 0.25)),
                   float(np.quantile(results, 0.75)),
                   cens, len(results))

    print(f"\n=== L={L} signed sample-complexity ===")
    print(f"  {'k':>6}  {'med':>6}  {'q25':>6}  {'q75':>6}  {'cens':>5}/n")
    for k in ks:
        med, q25, q75, cens, n = by_k[k]
        print(f"  {k:>6d}  {med:>6.0f}  {q25:>6.0f}  {q75:>6.0f}  {cens:>3d}/{n}")


if __name__ == "__main__":
    main()
