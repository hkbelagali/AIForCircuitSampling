"""Push L=7 until the VMC-steps curve hits the eval-every=10 floor (10 steps),
mirroring L=3..6. Floor crossing scaled as ~40-50 x D_sector for L=4,5,6.
L=7 sector dim is 1225, so predicted floor crossing is around k = 50000.

Per-cell JSONs land in results/m9_cells/, fully resumable.
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import time
from pathlib import Path

L = 7
EXTRA_KS = [500, 1000, 2000, 4000, 8000, 16000, 32000, 50000, 65000]
SEEDS = list(range(8))
N_WORKERS = 96

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
    print(f"L={L} push to floor: {len(EXTRA_KS)} new k values "
          f"({EXTRA_KS[0]}..{EXTRA_KS[-1]}), {len(jobs)} cells todo "
          f"(workers={N_WORKERS})", flush=True)
    if not jobs:
        print("nothing to do.", flush=True)
        return
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS, initializer=_worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1), 1):
            ttr = r.get("steps_to_threshold")
            tag = f"step={ttr}" if ttr is not None else "MAX"
            print(f"  [{i:>3}/{len(jobs)}] L={r['L']} k={r['k']:>5} s={r['seed']}: "
                  f"{tag:>10}  ({r['elapsed_sec']:7.1f}s)", flush=True)
    print(f"DONE in {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
