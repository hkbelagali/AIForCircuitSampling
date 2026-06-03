"""Multiprocess driver for the M15 Pauli-tracking sweep. Mirrors m9_run_sweep.py."""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np


L_VALUES = [4, 5, 6, 7]
SEEDS = list(range(8))
C_MIN = 0.0
C_MAX_DEFAULT = 4.25
N_C_DEFAULT = 18
MAX_WEIGHT_DEFAULT = 3

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m15_cells"


def _k_grid(c_max, n_c):
    cs = np.linspace(C_MIN, c_max, n_c)
    return [(L, int(round(c * L * L)), s, float(c))
            for L in L_VALUES for c in cs for s in SEEDS]


def _worker_init():
    import torch
    torch.set_num_threads(1)


def _worker(job):
    L, k, seed, c, max_weight = job
    import torch
    torch.set_num_threads(1)
    from m15_pauli_tracking import run_cell
    t0 = time.time()
    rec = run_cell(L, k, seed, max_weight=max_weight, verbose=False)
    return {
        "L": L, "k": k, "seed": seed, "c": c,
        "E_step": rec.get("steps_to_E_threshold"),
        "P_steps": rec.get("steps_to_P_threshold"),
        "elapsed": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--c-max", type=float, default=C_MAX_DEFAULT)
    parser.add_argument("--n-c", type=int, default=N_C_DEFAULT)
    parser.add_argument("--max-weight", type=int, default=MAX_WEIGHT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grid = [(L, k, s, c, args.max_weight)
            for (L, k, s, c) in _k_grid(args.c_max, args.n_c)]
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [job for job in grid
            if not (CELLS_DIR / f"L{job[0]}_k{job[1]}_s{job[2]}.json").exists()]

    by_L = {L: sum(1 for j in grid if j[0] == L) for L in L_VALUES}
    print(f"grid: {len(grid)} cells (L breakdown: {by_L})  "
          f"todo: {len(todo)}  workers: {args.workers}  "
          f"max_weight: {args.max_weight}", flush=True)
    if args.dry_run:
        return
    if not todo:
        print("all cells already done.", flush=True)
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
            estep = res["E_step"]
            psteps = res["P_steps"]
            tag = f"E={estep}, P={psteps}"
            print(f"  [{n_done:>4}/{len(todo)}] L={res['L']} k={res['k']:>4} "
                  f"s={res['seed']}: {tag}  ({res['elapsed']:5.1f}s)  "
                  f"ETA={eta/60:.1f}m", flush=True)

    print(f"\nDONE in {(time.time()-t0)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
