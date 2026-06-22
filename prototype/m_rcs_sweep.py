"""k_train sweep on n=8 RCS at three depths.

For each (depth, seed) we build the circuit once and reuse it across all
k_train values. Per-cell JSON dropped into results/m_rcs_sweep/.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import torch

from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import exact_probabilities
from rcs import run_rcs_xeb_cell


def main():
    n = 8
    depths = [4, 10, 20]
    ks = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    seeds = [0, 1, 2]
    m_candidates = 5000

    out_dir = Path(__file__).resolve().parents[1] / "results" / "m_rcs_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, cols = grid_for(n)
    print(f"n={n} ({rows}x{cols} grid), depths={depths}, ks={ks}, seeds={seeds}")
    print(f"device={device}, writing to {out_dir}\n", flush=True)

    t_global = time.time()
    for depth in depths:
        for seed in seeds:
            t_dseed = time.time()
            qubits, circuit = make_rcs_circuit(rows, cols, depth, seed=seed)
            p_C = exact_probabilities(circuit, qubits)
            cache = (circuit, qubits, p_C)
            for k in ks:
                cell_path = out_dir / f"n{n}_d{depth}_k{k}_s{seed}.json"
                if cell_path.exists():
                    print(f"  skip d={depth} s={seed} k={k} (exists)", flush=True)
                    continue
                t0 = time.time()
                out = run_rcs_xeb_cell(
                    n=n, depth=depth, k_train=k, m_candidates=m_candidates,
                    seed=seed, d_hidden=64, epochs=400, lr=2e-3, batch_size=64,
                    k_held=1000, device=device, verbose=False,
                    circuit_cache=cache,
                )
                cell_path.write_text(json.dumps(out))
                print(f"  d={depth} s={seed} k={k:>5}: "
                      f"xeb={out['candidate_xeb']:>6.3f}  "
                      f"F_cl={out['classical_fidelity']:>6.3f}  "
                      f"novel_mass={out['novel_mass']:>6.3f}  "
                      f"TV={out['tv_distance']:>5.3f}  "
                      f"({time.time() - t0:.1f}s)",
                      flush=True)
            print(f"  -> depth={depth} seed={seed} done "
                  f"({time.time() - t_dseed:.1f}s)\n", flush=True)
    print(f"total elapsed: {(time.time() - t_global)/60:.1f} min")


if __name__ == "__main__":
    main()
