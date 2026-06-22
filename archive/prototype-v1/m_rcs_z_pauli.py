"""Z-Pauli extrapolation test: train RNN on weight ≤ w_train Z observables
(shadow targets from k bitstring samples), evaluate per-weight error of
model expectations vs true p_C expectations at every weight 0..n.

Hypothesis: the AR factorization buys some 'for-free' fit at weights above
w_train, but error climbs sharply once you cross out of the trained set.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import torch

from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import exact_probabilities
from rcs import run_z_pauli_cell


def main():
    n = 8
    depth = 10
    k_train = 2000
    w_trains = [1, 2, 3, 4, 5, 6, 7, 8]
    seeds = list(range(16))
    circuit_seed = 0  # fixed circuit — seeds only vary samples + model init

    out_dir = Path(__file__).resolve().parents[1] / "results" / "m_rcs_z_pauli"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"n={n} depth={depth} k_train={k_train}  circuit_seed={circuit_seed}")
    print(f"w_train sweep {w_trains} x seeds {seeds}")
    print(f"writing to {out_dir}\n", flush=True)
    rows, cols = grid_for(n)
    qubits, circuit = make_rcs_circuit(rows, cols, depth, seed=circuit_seed)
    p_C = exact_probabilities(circuit, qubits)
    cache = (circuit, qubits, p_C)

    t0 = time.time()
    for seed in seeds:
        for w_train in w_trains:
            cell_path = out_dir / f"n{n}_d{depth}_k{k_train}_w{w_train}_cs{circuit_seed}_s{seed}.json"
            if cell_path.exists():
                print(f"  skip s={seed} w={w_train} (exists)", flush=True)
                continue
            t_cell = time.time()
            out = run_z_pauli_cell(n=n, depth=depth, k_train=k_train,
                                     w_train=w_train, seed=seed,
                                     d_hidden=64, epochs=400, lr=2e-3,
                                     device=device, circuit_cache=cache,
                                     verbose=False)
            cell_path.write_text(json.dumps(out))
            err = out["err_by_weight_model"]
            shadow = out["err_by_weight_shadow"]
            er_str = "  ".join(f"w{w}={err[str(w) if isinstance(next(iter(err)), str) else w]:.3f}"
                                for w in range(1, n + 1))
            print(f"  s={seed} w_train={w_train}: trained_loss={out['final_loss']:.3e}  "
                  f"({time.time() - t_cell:.1f}s)", flush=True)
            print(f"     model err per w: " +
                  "  ".join(f"w{w}={err[w]:.3f}" for w in range(1, n + 1)),
                  flush=True)
            print(f"     shadow err per w: " +
                  "  ".join(f"w{w}={shadow[w]:.3f}" for w in range(1, n + 1)),
                  flush=True)
    print(f"\ntotal: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
