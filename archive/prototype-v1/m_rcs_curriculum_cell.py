"""RCS-with-signs curriculum cell: train w=4 cold, then warm-start each
successive w (5..8) from the previous w's best state.

CLI mirrors m_rcs_signhead_cell.py but loops over w_max internally.
Writes one JSON per (w_max, seed) to results/<out_subdir>/ and one .pt
checkpoint per (w_max, seed) alongside.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import numpy as np
import torch

import cirq
from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import int_to_bits

from random_shadows import build_loss_paulis_full, torch_ops_full
from fast_shadow_targets import shadow_targets_full_fast
from complex_arrnn import (
    ARPhaseComplexARRNN as ComplexARRNN,
    sample_shadows_random_pauli_complex,
    train_complex_pauli_loss,
)


def rcs_state(n, depth, seed):
    qubits, circ = make_rcs_circuit(*grid_for(n), depth, seed=seed)
    sv = cirq.Simulator().simulate(circ, qubit_order=qubits).final_state_vector
    psi = np.asarray(sv, dtype=np.complex128)
    psi /= np.linalg.norm(psi) or 1.0
    return psi


def quantum_fidelity(psi_model, psi_target):
    return float(np.abs(np.vdot(psi_model, psi_target)) ** 2)


def train_stage(n, d_hidden, ops, targets_t, alpha_t, all_bits_t, *,
                  n_restarts, epochs, lr, warm_state, seed, w_stage, device):
    """Run n_restarts on the given Pauli set. Restart 0 uses warm_state if
    provided; remaining restarts are cold random inits. Cold-stage uses the
    original protocol's torch seeds (seed*17 + r_i*7919) so w=w_min hit rate
    matches prior runs; warm stages shift by w_stage to vary cold hedges."""
    best_loss = float("inf")
    best_state = None
    for r_i in range(n_restarts):
        if warm_state is None:
            torch.manual_seed(seed * 17 + r_i * 7919)
        else:
            torch.manual_seed(seed * 17 + r_i * 7919 + w_stage * 1_000_003)
        model = ComplexARRNN(n_qubits=n, d_hidden=d_hidden).to(device)
        if r_i == 0 and warm_state is not None:
            model.load_state_dict(warm_state)
        final = train_complex_pauli_loss(model, ops, targets_t, alpha_t,
                                          all_bits_t, epochs=epochs, lr=lr,
                                          verbose=False)
        if final < best_loss:
            best_loss = final
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return best_loss, best_state


def evaluate(model, all_bits_t, psi_RCS):
    with torch.no_grad():
        psi_np = model.psi(all_bits_t, use_phase=True).cpu().numpy().astype(np.complex128)
    psi_np /= np.linalg.norm(psi_np) or 1.0
    p_model = np.abs(psi_np) ** 2
    p_RCS = np.abs(psi_RCS) ** 2
    p_RCS = p_RCS / p_RCS.sum()
    p_model = p_model / p_model.sum()
    QF = quantum_fidelity(psi_np, psi_RCS)
    F_cl = float(np.square(np.sqrt(np.maximum(p_model * p_RCS, 0)).sum()))
    TV = 0.5 * float(np.abs(p_model - p_RCS).sum())
    return QF, F_cl, TV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k_train", type=int, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--circuit_seed", type=int, default=0)
    parser.add_argument("--d_hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w_min", type=int, default=4)
    parser.add_argument("--w_max", type=int, default=8)
    parser.add_argument("--n_restarts_cold", type=int, default=16,
                         help="restarts at the cold-start stage (w=w_min)")
    parser.add_argument("--n_restarts_warm", type=int, default=4,
                         help="restarts at each warm stage (w>w_min); restart 0 warm-init")
    parser.add_argument("--batch_size", type=int, default=8192,
                         help="shadow_targets batch size; reduce to fit on smaller GPUs")
    parser.add_argument("--out_subdir", type=str, default="m_rcs_curriculum")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(__file__).resolve().parents[1] / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"n={args.n} d={args.depth} seed={args.seed} k={args.k_train}  "
          f"curriculum w∈[{args.w_min},{args.w_max}]  device={device}", flush=True)

    psi_RCS = rcs_state(args.n, args.depth, args.circuit_seed)
    rng = np.random.default_rng(args.seed)
    U_r, b_r = sample_shadows_random_pauli_complex(psi_RCS, args.n, args.k_train, rng)

    dim = 1 << args.n
    all_int = np.arange(dim, dtype=np.int64)
    all_bits_t = torch.from_numpy(int_to_bits(all_int, args.n)).long().to(device)

    warm_state = None
    for w in range(args.w_min, args.w_max + 1):
        tag = f"n{args.n}_d{args.depth}_k{args.k_train}_w{w}_s{args.seed}"
        json_p = out_dir / f"{tag}.json"
        ckpt_p = out_dir / f"{tag}.pt"

        # Resume: if both files exist, just load checkpoint as warm state and skip
        if json_p.exists() and ckpt_p.exists():
            warm_state = torch.load(ckpt_p, map_location=device, weights_only=True)
            cached = json.loads(json_p.read_text())
            print(f"  w={w}: cached  QF={cached['fidelity']:.4f}", flush=True)
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loss_paulis = build_loss_paulis_full(args.n, w)
        bs = args.batch_size if torch.cuda.is_available() else 1024
        targets_np = shadow_targets_full_fast(loss_paulis, U_r, b_r,
                                                device=device, batch_size=bs)
        ops = torch_ops_full(loss_paulis, device)
        targets_t = torch.from_numpy(targets_np).double().to(device)
        alpha_t = torch.ones(loss_paulis["n_paulis"], dtype=torch.float64, device=device)

        is_cold = (w == args.w_min) or (warm_state is None)
        n_r = args.n_restarts_cold if is_cold else args.n_restarts_warm

        t0 = time.time()
        best_loss, best_state = train_stage(
            args.n, args.d_hidden, ops, targets_t, alpha_t, all_bits_t,
            n_restarts=n_r, epochs=args.epochs, lr=args.lr,
            warm_state=None if is_cold else warm_state,
            seed=args.seed, w_stage=w, device=device,
        )
        elapsed = time.time() - t0

        model = ComplexARRNN(n_qubits=args.n, d_hidden=args.d_hidden).to(device)
        model.load_state_dict(best_state)
        QF, F_cl, TV = evaluate(model, all_bits_t, psi_RCS)

        rec = {
            "n": args.n, "depth": args.depth, "k_train": args.k_train,
            "w_max": w, "seed": args.seed, "d_hidden": args.d_hidden,
            "epochs": args.epochs, "n_restarts": n_r,
            "warm_started": (not is_cold),
            "final_loss": best_loss,
            "fidelity": QF, "F_cl": F_cl, "TV": TV,
            "elapsed_sec": elapsed,
        }
        json_p.write_text(json.dumps(rec))
        torch.save(best_state, ckpt_p)
        warm_state = best_state
        print(f"  w={w}: QF={QF:.4f}  loss={best_loss:.4e}  "
              f"(restarts={n_r}, warm={not is_cold}, {elapsed/60:.1f} min)",
              flush=True)


if __name__ == "__main__":
    main()
