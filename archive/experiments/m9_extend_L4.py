"""Extend the L=4 k-axis beyond k_max=68 to test whether VMC steps drop
toward ~0 as k saturates the sector (the pattern observed at L=3 where
k = 4.2 x D_sector gave steps ~10).

L=4 sector dim is 36, so k=68 was only 1.9 x sector. This script adds k
values up to ~7 x sector (=252) to check if the VMC-step curve continues
to drop, mirroring the L=3 saturation behavior.

Per-cell JSONs land in results/m9_cells/, fully resumable.
"""

import os
# Pin BLAS threads before numpy import (see m9_run_sweep.py rationale).
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import time
from pathlib import Path

L = 4
SEEDS = list(range(8))
# k values beyond the original 0..68 grid; targeting up to ~7x sector (D=36)
EXTRA_KS = [76, 84, 92, 104, 116, 128, 144, 160, 180, 200, 220, 252]
N_WORKERS = 64

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m9_cells"


def _worker_init():
    import torch
    torch.set_num_threads(1)


def _worker(job):
    L_, k, s = job
    import torch
    torch.set_num_threads(1)
    from m9_sample_complexity_sweep import run_cell
    return run_cell(L_, k, s, verbose=False)


def main():
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(L, k, s) for k in EXTRA_KS for s in SEEDS
            if not (CELLS_DIR / f"L{L}_k{k}_s{s}.json").exists()]
    print(f"L={L} extension: {len(EXTRA_KS)} new k values ({EXTRA_KS[0]}..{EXTRA_KS[-1]}), "
          f"{len(jobs)} cells todo (workers={N_WORKERS})", flush=True)
    if not jobs:
        print("nothing to do.", flush=True)
        return
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS, initializer=_worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            ttr = r.get("steps_to_threshold")
            tag = f"step={ttr}" if ttr is not None else "MAX"
            print(f"  [{i:>3}/{len(jobs)}] L={r['L']} k={r['k']:>3} s={r['seed']}: "
                  f"{tag:>10}  ({r['elapsed_sec']:5.1f}s)", flush=True)
    print(f"DONE in {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
