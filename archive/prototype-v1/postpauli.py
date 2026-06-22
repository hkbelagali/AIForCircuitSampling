"""Post-hoc Pauli scan: for each cell JSON in --cells, read psi_model
and stream every sector-preserving, even-Y Pauli of weight 0..2L,
recording per-exact-weight max + mean |<P>_model - <P>_true| and the
running cumulative max. Writes the results back into the cell JSON
under:
  pauli_max_err_exact, pauli_mean_err_exact, pauli_count_exact,
  pauli_max_err_cumulative.

Parallelizes over cells (one Python worker per cell). Skips cells that
already have the full range populated unless --force.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _worker(job):
    path, force = job
    import numpy as np
    from m9 import Hubbard
    from pauli import err_streaming, cumulative_from_exact
    rec = json.loads(Path(path).read_text())
    L = rec["L"]
    max_w = 2 * L
    if (not force and "pauli_mean_err_exact" in rec
        and len(rec["pauli_mean_err_exact"]) >= max_w + 1):
        return path, 0.0, "skipped"
    if "psi_model" not in rec:
        return path, 0.0, "no psi_model"
    psi_m = np.asarray(rec["psi_model"], dtype=np.float64)
    ctx = Hubbard(L=L, U=4.0)
    t0 = time.time()
    max_exact, mean_exact, cnt_exact = err_streaming(
        ctx, psi_m, ctx.psi_0, max_w)
    cum_max = cumulative_from_exact(max_exact, max_w)
    dt = time.time() - t0
    rec["pauli_max_err_exact"] = {int(w): float(max_exact.get(w, 0.0))
                                  for w in range(max_w + 1)}
    rec["pauli_mean_err_exact"] = {int(w): float(mean_exact.get(w, 0.0))
                                   for w in range(max_w + 1)}
    rec["pauli_count_exact"] = {int(w): int(cnt_exact.get(w, 0))
                                for w in range(max_w + 1)}
    rec["pauli_max_err_cumulative"] = {int(w): float(cum_max.get(w, 0.0))
                                       for w in range(max_w + 1)}
    Path(path).write_text(json.dumps(rec))
    return path, dt, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=str, default="results/shadow_cells_v3")
    ap.add_argument("--workers", type=int,
                    default=max(1, os.cpu_count() - 2))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = sorted(str(p) for p in Path(args.cells).glob("L*_kz*_kr*_s*.json"))
    if not paths:
        print(f"no cells in {args.cells}"); return
    print(f"processing {len(paths)} cells with {args.workers} workers")
    jobs = [(p, args.force) for p in paths]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for i, (path, dt, msg) in enumerate(
                pool.imap_unordered(_worker, jobs), 1):
            print(f"  [{i:>3}/{len(paths)}] {Path(path).name}: {msg} "
                  f"({dt:.1f}s)", flush=True)
    print(f"total wallclock: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
