"""Push L=4 and L=5 out to k=2500 to map the deep-saturation behavior of
VMC-steps-to-threshold (the L=3 story showed steps -> ~10 as k >> sector dim;
this asks whether L=4 (sector 36) and L=5 (sector 100) hit a similar floor).

k targets:
  L=4: 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500  (9x..69x sector)
  L=5: 150, 250, 400, 600, 800, 1100, 1500, 2000, 2500        (1.5x..25x sector)

Per-cell JSONs land in results/m9_cells/, fully resumable.
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import time
from pathlib import Path

JOBS_PER_L = {
    4: [320, 400, 500, 640, 800, 1000, 1250, 1600, 2000, 2500],
    5: [150, 250, 400, 600, 800, 1100, 1500, 2000, 2500],
}
SEEDS = list(range(8))
N_WORKERS = 96

CELLS_DIR = Path(__file__).resolve().parents[1] / "results" / "m9_cells"


def _worker_init():
    import torch
    torch.set_num_threads(1)


def _worker(job):
    L, k, s = job
    import torch
    torch.set_num_threads(1)
    from m9_sample_complexity_sweep import run_cell
    return run_cell(L, k, s, verbose=False)


def main():
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for L, ks in JOBS_PER_L.items():
        for k in ks:
            for s in SEEDS:
                if not (CELLS_DIR / f"L{L}_k{k}_s{s}.json").exists():
                    jobs.append((L, k, s))
    total = sum(len(ks) for ks in JOBS_PER_L.values()) * len(SEEDS)
    print(f"L=4,5 push to k=2500: {total} total cells; {len(jobs)} todo "
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
            print(f"  [{i:>3}/{len(jobs)}] L={r['L']} k={r['k']:>4} s={r['seed']}: "
                  f"{tag:>10}  ({r['elapsed_sec']:6.1f}s)", flush=True)
    print(f"DONE in {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
