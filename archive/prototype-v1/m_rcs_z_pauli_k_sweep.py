"""RCS n=8 Z-Pauli loss training, sweep over BOTH k_train and w_train,
fixed circuit. Compute classical fidelity for each cell to be the RCS
analog of the Hubbard with/without-sign-head fidelity plots."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import numpy as np
import torch

from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import (
    bits_to_int, exact_probabilities, int_to_bits, sample_from_circuit,
)
from rcs import (
    BitstringARRNN, classical_fidelity, kl_divergence, model_full_distribution,
    train_rnn_z_pauli, tv_distance, enumerate_z_supports,
)


def run_cell(n, depth, circuit_cache, k_train, w_train, seed,
             d_hidden=64, epochs=400, lr=2e-3, device="cpu"):
    circuit, qubits, p_C = circuit_cache
    train_int = sample_from_circuit(circuit, qubits, k_train, seed=seed)
    supports, weights = enumerate_z_supports(n, w_train)
    torch.manual_seed(seed)
    model = BitstringARRNN(n_qubits=n, d_hidden=d_hidden).to(device)
    t0 = time.time()
    final_loss = train_rnn_z_pauli(
        model, train_int, supports, weights, n,
        epochs=epochs, lr=lr, device=device, verbose=False)
    elapsed = time.time() - t0
    p_model = model_full_distribution(model, n, device)
    F_cl = classical_fidelity(p_model, p_C)
    tv = tv_distance(p_model, p_C)
    kl = kl_divergence(p_model, p_C)
    return {
        "n": n, "depth": depth, "k_train": k_train, "w_train": w_train,
        "seed": seed, "d_hidden": d_hidden, "epochs": epochs,
        "final_loss": float(final_loss),
        "F_cl": float(F_cl), "TV": float(tv), "kl": float(kl),
        "elapsed_sec": elapsed,
    }


def main():
    n = 8
    depth = 10
    circuit_seed = 0
    ks = [100, 300, 1000, 3000, 10000]
    w_trains = [1, 2, 3, 4]
    seeds = list(range(8))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows, cols = grid_for(n)
    qubits, circuit = make_rcs_circuit(rows, cols, depth, seed=circuit_seed)
    p_C = exact_probabilities(circuit, qubits)
    cache = (circuit, qubits, p_C)

    out_dir = Path(__file__).resolve().parents[1] / "results" / "m_rcs_z_pauli_k_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"n={n} depth={depth} circuit_seed={circuit_seed}")
    print(f"ks={ks}  w_trains={w_trains}  seeds={seeds}")
    print(f"device={device}, output={out_dir}\n", flush=True)

    t0 = time.time()
    for w in w_trains:
        for k in ks:
            row = []
            for seed in seeds:
                p = out_dir / f"n{n}_d{depth}_k{k}_w{w}_cs{circuit_seed}_s{seed}.json"
                if p.exists():
                    out = json.loads(p.read_text())
                else:
                    out = run_cell(n=n, depth=depth, circuit_cache=cache,
                                    k_train=k, w_train=w, seed=seed,
                                    d_hidden=64, epochs=400, device=device)
                    p.write_text(json.dumps(out))
                row.append(out)
            fcls = [r["F_cl"] for r in row]
            tvs = [r["TV"] for r in row]
            print(f"  w={w} k={k:>5}: F_cl={np.median(fcls):.4f}  TV={np.median(tvs):.4f}",
                  flush=True)
        print()
    print(f"total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
