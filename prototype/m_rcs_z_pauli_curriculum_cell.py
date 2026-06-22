"""RCS n=8 Z-Pauli (unsigned) curriculum cell: train BitstringARRNN at
w_min cold, then warm-start each successive w from the previous w's best
state. Mirrors m_rcs_curriculum_cell.py for direct comparison.

CLI: one (seed, k_train) per call. Loops w=1..4 internally.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import numpy as np
import torch

from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import exact_probabilities, sample_from_circuit
from rcs import (
    BitstringARRNN, classical_fidelity, kl_divergence,
    model_full_distribution, train_rnn_z_pauli, tv_distance,
    enumerate_z_supports,
)


def train_stage(n, d_hidden, samples_int, supports, weights, epochs, lr, *,
                  n_restarts, warm_state, seed, w_stage, device):
    best_loss = float("inf")
    best_state = None
    for r_i in range(n_restarts):
        if warm_state is None:
            torch.manual_seed(seed * 17 + r_i * 7919)
        else:
            torch.manual_seed(seed * 17 + r_i * 7919 + w_stage * 1_000_003)
        model = BitstringARRNN(n_qubits=n, d_hidden=d_hidden).to(device)
        if r_i == 0 and warm_state is not None:
            model.load_state_dict(warm_state)
        final = train_rnn_z_pauli(model, samples_int, supports, weights, n,
                                    epochs=epochs, lr=lr, device=device,
                                    verbose=False)
        if final < best_loss:
            best_loss = final
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return best_loss, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k_train", type=int, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--circuit_seed", type=int, default=0)
    parser.add_argument("--d_hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--w_min", type=int, default=1)
    parser.add_argument("--w_max", type=int, default=4)
    parser.add_argument("--n_restarts_cold", type=int, default=4)
    parser.add_argument("--n_restarts_warm", type=int, default=2)
    parser.add_argument("--out_subdir", type=str, default="m_rcs_z_pauli_curriculum")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(__file__).resolve().parents[1] / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"RCS n={args.n} d={args.depth} cs={args.circuit_seed}  "
          f"seed={args.seed} k={args.k_train}  w∈[{args.w_min},{args.w_max}]  "
          f"device={device}", flush=True)

    rows, cols = grid_for(args.n)
    qubits, circuit = make_rcs_circuit(rows, cols, args.depth, seed=args.circuit_seed)
    p_C = exact_probabilities(circuit, qubits)
    samples_int = sample_from_circuit(circuit, qubits, args.k_train, seed=args.seed)

    warm_state = None
    for w in range(args.w_min, args.w_max + 1):
        tag = (f"n{args.n}_d{args.depth}_k{args.k_train}_w{w}"
               f"_cs{args.circuit_seed}_s{args.seed}")
        json_p = out_dir / f"{tag}.json"
        ckpt_p = out_dir / f"{tag}.pt"
        if json_p.exists() and ckpt_p.exists():
            warm_state = torch.load(ckpt_p, map_location=device, weights_only=True)
            cached = json.loads(json_p.read_text())
            print(f"  w={w}: cached  F_cl={cached['F_cl']:.4f}", flush=True)
            continue

        supports, weights = enumerate_z_supports(args.n, w)
        is_cold = (w == args.w_min) or (warm_state is None)
        n_r = args.n_restarts_cold if is_cold else args.n_restarts_warm

        t0 = time.time()
        best_loss, best_state = train_stage(
            args.n, args.d_hidden, samples_int, supports, weights,
            args.epochs, args.lr,
            n_restarts=n_r,
            warm_state=None if is_cold else warm_state,
            seed=args.seed, w_stage=w, device=device,
        )
        elapsed = time.time() - t0

        model = BitstringARRNN(n_qubits=args.n, d_hidden=args.d_hidden).to(device)
        model.load_state_dict(best_state)
        p_model = model_full_distribution(model, args.n, device)
        F_cl = classical_fidelity(p_model, p_C)
        tv = tv_distance(p_model, p_C)
        kl = kl_divergence(p_model, p_C)

        rec = {
            "n": args.n, "depth": args.depth, "k_train": args.k_train,
            "w_train": w, "seed": args.seed,
            "d_hidden": args.d_hidden, "epochs": args.epochs,
            "n_restarts": n_r, "warm_started": (not is_cold),
            "final_loss": float(best_loss),
            "F_cl": float(F_cl), "TV": float(tv), "kl": float(kl),
            "elapsed_sec": elapsed,
        }
        json_p.write_text(json.dumps(rec))
        torch.save(best_state, ckpt_p)
        warm_state = best_state
        print(f"  w={w}: F_cl={F_cl:.4f}  loss={best_loss:.3e}  "
              f"(restarts={n_r}, warm={not is_cold}, {elapsed:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
