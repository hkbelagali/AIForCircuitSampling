"""Push L=5 beyond k=2500 until the VMC-steps curve hits the eval-every=10
floor (10 steps), mirroring L=3 and L=4. Current L=5 trend:
  k=1500 -> 60, k=2000 -> 40, k=2500 -> 30  (roughly halving per ~1.5k samples)
so we expect k in the 5k-10k range to be needed.

Per-cell JSONs land in results/m9_cells/, fully resumable.
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import time
from pathlib import Path

L = 5
EXTRA_KS = [3000, 4000, 5000, 6500, 8000, 10000, 12500, 15000]
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
                  f"{tag:>10}  ({r['elapsed_sec']:6.1f}s)", flush=True)
    print(f"DONE in {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
