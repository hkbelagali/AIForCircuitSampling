"""RCS-with-signs cell driver: train ComplexARRNN via Pauli loss on random
Pauli shadow data from the RCS state. CLI takes --w_max for SLURM array.

Computes quantum fidelity |<ψ_model | ψ_RCS>|² against the complex RCS state.
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

import cirq
from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import int_to_bits

from random_shadows import build_loss_paulis_full, shadow_targets_full, torch_ops_full
from fast_shadow_targets import shadow_targets_full_fast
from complex_arrnn import (
    ARPhaseComplexARRNN as ComplexARRNN,  # AR-factorized phase (fixed architecture)
    model_expectations_complex,
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


def run_cell(n, depth, psi_RCS, k_train, w_max, seed, *, n_restarts,
              d_hidden=64, epochs=1500, lr=1e-3, device="cpu"):
    rng = np.random.default_rng(seed)
    dim = 1 << n

    # Sample random Pauli shadow shots
    U_r, b_r = sample_shadows_random_pauli_complex(psi_RCS, n, k_train, rng)

    # Build Pauli operator set + shadow targets
    loss_paulis = build_loss_paulis_full(n, w_max)
    # Fixed batch size — assumes GPU has enough memory (H200 at high w_max)
    bs = 8192 if torch.cuda.is_available() else 1024
    targets_np = shadow_targets_full_fast(loss_paulis, U_r, b_r,
                                            device=device, batch_size=bs)
    ops = torch_ops_full(loss_paulis, device)
    targets_t = torch.from_numpy(targets_np).double().to(device)
    alpha_t = torch.ones(loss_paulis["n_paulis"], dtype=torch.float64, device=device)

    # Full-Hilbert input bits for the AR-RNN (MSB-first via int_to_bits)
    all_int = np.arange(dim, dtype=np.int64)
    all_bits_t = torch.from_numpy(int_to_bits(all_int, n)).long().to(device)

    t0 = time.time()
    best_loss = float("inf")
    best_state = None
    for r_i in range(n_restarts):
        torch.manual_seed(seed * 17 + r_i * 7919)
        model = ComplexARRNN(n_qubits=n, d_hidden=d_hidden).to(device)
        final = train_complex_pauli_loss(model, ops, targets_t, alpha_t,
                                          all_bits_t, epochs=epochs, lr=lr,
                                          verbose=False)
        if final < best_loss:
            best_loss = final
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    elapsed = time.time() - t0
    model.load_state_dict(best_state)

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

    return {
        "n": n, "depth": depth, "k_train": k_train, "w_max": w_max,
        "seed": seed, "d_hidden": d_hidden, "epochs": epochs,
        "n_restarts": n_restarts, "final_loss": best_loss,
        "fidelity": QF, "F_cl": F_cl, "TV": TV,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_max", type=int, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--circuit_seed", type=int, default=0)
    parser.add_argument("--n_restarts", type=int, default=16)
    parser.add_argument("--out_subdir", type=str, default="m_rcs_signhead_v2")
    parser.add_argument("--seed", type=int, default=None,
                         help="only run this seed (for SLURM array). default: all 0..7")
    parser.add_argument("--k_train", type=int, default=None,
                         help="only run this k_train (for SLURM array). default: all in grid")
    args = parser.parse_args()

    ks_default = [100, 300, 1000, 3000, 10000]
    ks = [args.k_train] if args.k_train is not None else ks_default
    seeds = [args.seed] if args.seed is not None else list(range(8))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(__file__).resolve().parents[1] / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"n={args.n} depth={args.depth} circuit_seed={args.circuit_seed}  "
          f"w_max={args.w_max} restarts={args.n_restarts}  device={device}",
          flush=True)

    psi_RCS = rcs_state(args.n, args.depth, args.circuit_seed)
    print(f"  RCS state norm = {np.linalg.norm(psi_RCS):.6f}, "
          f"max |ψ|^2 = {(np.abs(psi_RCS)**2).max():.4f}", flush=True)

    t0 = time.time()
    for k in ks:
        for seed in seeds:
            tag = f"n{args.n}_d{args.depth}_k{k}_w{args.w_max}_s{seed}"
            p = out_dir / f"{tag}.json"
            if p.exists():
                continue
            out = run_cell(args.n, args.depth, psi_RCS,
                            k_train=k, w_max=args.w_max, seed=seed,
                            n_restarts=args.n_restarts, device=device)
            p.write_text(json.dumps(out))
        files = list(out_dir.glob(f"n{args.n}_d{args.depth}_k{k}_w{args.w_max}_s*.json"))
        QFs = [json.loads(f.read_text())["fidelity"] for f in files]
        print(f"  w={args.w_max} k={k:>5}: QF med={np.median(QFs):.4f} "
              f"({len(QFs)} seeds)", flush=True)
    print(f"total: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
