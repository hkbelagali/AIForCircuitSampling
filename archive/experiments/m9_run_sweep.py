"""Local multiprocess driver for the M9 sample-complexity sweep.

Spawns N_WORKERS independent Python processes (default: ~half the machine's
logical CPUs), each running a single torch thread, and dispatches per-cell
(L, k, seed) jobs. Each cell writes its own results/m9_cells/L{L}_k{k}_s{seed}.json
so the sweep is fully resumable (already-done cells are skipped).

Sweet spot: 1 torch thread per process (intra-cell threading gave no benefit
in the L=6/L=8 micro-benchmark). With 192 cores available we can run 100+
cells truly in parallel.

CLI:
    python experiments/m9_run_sweep.py [--workers N] [--c-max C] [--n-c N]
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

# Pin BLAS / OMP threads to 1 BEFORE numpy is imported, anywhere in the
# parent or in a spawned worker. With 96 workers, default 64-thread OpenBLAS
# would launch 96*64 = 6144 threads and blow past RLIMIT_NPROC (4096).
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np


# ---- grid spec -------------------------------------------------------------
# L=8 dropped from the headline -- it was capacity-limited at d_hidden=32
# (100% censored). L=3, 5, 7 added for smoother coverage in n. Note: odd L
# uses N_up = N_dn = L // 2 (near-half-filling), since exact half-filling
# requires even L for an Sz=0 sector. Odd-L PBC chains aren't bipartite
# (odd cycle), but our signs_from_psi is ED-based and works for any L.
L_VALUES = [3, 4, 5, 6, 7]
SEEDS = list(range(8))
C_MIN = 0.0
C_MAX_DEFAULT = 4.25
N_C_DEFAULT = 18
# k(L, c) = round(c * L^2) -- expresses k as a polynomial-in-L knob; c=0 is
# the pure-VMC baseline, c=4.25 covers > sector dim at L=4 (sampling with
# replacement is fine).
# ----------------------------------------------------------------------------

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m9_cells"


def _k_grid(c_max=C_MAX_DEFAULT, n_c=N_C_DEFAULT):
    cs = np.linspace(C_MIN, c_max, n_c)
    return [(L, int(round(c * L * L)), s, float(c)) for L in L_VALUES for c in cs for s in SEEDS]


def _worker_init():
    """Pool initializer: enforce 1 thread per worker (BLAS env was already set
    in the parent, but be explicit on import order). Spawn-mode workers re-execute
    the module top-level, which sets the env vars before numpy import."""
    import torch
    torch.set_num_threads(1)


def _worker(job):
    L, k, seed, c = job
    import torch
    torch.set_num_threads(1)
    from m9_sample_complexity_sweep import run_cell
    t0 = time.time()
    rec = run_cell(L, k, seed, verbose=False)
    return {
        "L": L, "k": k, "seed": seed, "c": c,
        "steps_to_threshold": rec.get("steps_to_threshold"),
        "reached": rec.get("reached"),
        "elapsed": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                        help="number of parallel cell processes (default: cpu_count/2)")
    parser.add_argument("--c-max", type=float, default=C_MAX_DEFAULT)
    parser.add_argument("--n-c", type=int, default=N_C_DEFAULT)
    parser.add_argument("--dry-run", action="store_true",
                        help="just print the grid and exit")
    args = parser.parse_args()

    grid = _k_grid(c_max=args.c_max, n_c=args.n_c)
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    # Skip cells whose JSON already exists.
    todo = [job for job in grid
            if not (CELLS_DIR / f"L{job[0]}_k{job[1]}_s{job[2]}.json").exists()]

    by_L = {L: sum(1 for j in grid if j[0] == L) for L in L_VALUES}
    print(f"grid: {len(grid)} cells (L breakdown: {by_L})  "
          f"todo: {len(todo)}  workers: {args.workers}", flush=True)
    if args.dry_run:
        return

    if not todo:
        print("all cells already done; nothing to do.", flush=True)
        return

    t0 = time.time()
    n_done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init) as pool:
        for res in pool.imap_unordered(_worker, todo, chunksize=1):
            n_done += 1
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - n_done) / rate if rate > 0 else float("inf")
            st = res["steps_to_threshold"]
            tag = f"step={st}" if st is not None else "MAX"
            print(f"  [{n_done:>3}/{len(todo)}] L={res['L']} k={res['k']:>3} "
                  f"s={res['seed']}: {tag:>10}  ({res['elapsed']:5.1f}s)  "
                  f"rate={rate:.2f} c/s  ETA={eta/60:.1f}m", flush=True)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min. {len(todo)} cells written under {CELLS_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
