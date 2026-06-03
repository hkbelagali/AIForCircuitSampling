"""Parallel sweep over (L, k, seed). One JSON per cell under --out."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _worker(job):
    L, k, seed, out_dir = job
    p = Path(out_dir) / f"L{L}_k{k}_s{seed}.json"
    if p.exists():
        return L, k, seed, json.loads(p.read_text())["steps_to_threshold"]
    from m9 import run_cell
    rec = run_cell(L, k, seed)
    p.write_text(json.dumps(rec))
    return L, k, seed, rec["steps_to_threshold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--Ls", type=str, default="3,4,5,6,7")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--c-grid", type=str, default="0,0.25,0.5,1,1.5,2,2.5,3,3.5,4,4.25",
                    help="k = round(c * L^2) for each c")
    ap.add_argument("--out", type=str, default="cells")
    args = ap.parse_args()
    Ls = [int(x) for x in args.Ls.split(",")]
    cs = [float(x) for x in args.c_grid.split(",")]
    out = Path(args.out); out.mkdir(exist_ok=True)
    jobs = [(L, int(round(c * L * L)), s, str(out))
            for L in Ls for c in cs for s in range(args.seeds)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for L, k, s, steps in pool.imap_unordered(_worker, jobs):
            tag = f"step={steps}" if steps is not None else "MAX"
            print(f"L={L} k={k} s={s}: {tag}", flush=True)


if __name__ == "__main__":
    main()
