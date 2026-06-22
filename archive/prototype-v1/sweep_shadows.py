"""Multiprocess sweep over (L, k_z, k_random, seed) shadow-MLE cells.
One JSON per cell, fully resumable.

Pin BLAS to a small thread count before any numpy/torch import; otherwise N
workers x default_threads_per_worker = thread explosion.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")


def _cell_path(out_dir, L, k_z, k_random, seed):
    return Path(out_dir) / f"L{L}_kz{k_z}_kr{k_random}_s{seed}.json"


def _worker(job):
    L, k_z, k_random, seed, out_dir, threads, device, kwargs = job
    p = _cell_path(out_dir, L, k_z, k_random, seed)
    if p.exists():
        return json.loads(p.read_text())
    import torch
    torch.set_num_threads(threads)
    from shadows import run_shadow_cell
    rec = run_shadow_cell(L=L, k_z=k_z, k_random=k_random, seed=seed,
                          device=device, **kwargs)
    p.write_text(json.dumps(rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/shadow_cells_v3")
    ap.add_argument("--Ls", type=str, default="4,5,6")
    ap.add_argument("--seeds", type=int, default=4,
                    help="Number of seeds 0..seeds-1 (ignored if --seed-list set)")
    ap.add_argument("--seed-list", type=str, default=None,
                    help='comma-separated explicit seeds, e.g. "0,2"; overrides --seeds')
    ap.add_argument("--k-totals", type=str, default="1000,3000,10000,30000,100000",
                    help="comma-separated total shot budgets per cell")
    ap.add_argument("--kz-frac", type=float, default=0.25,
                    help="fraction of k_total allocated to Phase A (Z-only)")
    ap.add_argument("--pauli-max-weight", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=16,
                    help="N sign-head restarts per cell (use ≥8 to avoid "
                         "Phase-B sign-basin failures)")
    ap.add_argument("--selector", type=str, default="val",
                    choices=["val", "train", "oracle_fid"],
                    help='restart selection criterion: held-out NLL (val), '
                         'training NLL (train), or oracle fid (cheating)')
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs-A", type=int, default=1500)
    ap.add_argument("--epochs-B", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sub-k", type=int, default=0,
                    help="0/None = full batch (recommended for stability)")
    ap.add_argument("--d-hidden", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", type=str, default="cpu",
                    help='"cpu" or "cuda". On cuda, runs sequentially '
                         '(workers arg ignored).')
    args = ap.parse_args()

    Ls = [int(x) for x in args.Ls.split(",")]
    k_totals = [int(x) for x in args.k_totals.split(",")]
    if args.seed_list:
        seeds = [int(x) for x in args.seed_list.split(",")]
    else:
        seeds = list(range(args.seeds))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        epochs_A=args.epochs_A, epochs_B=args.epochs_B, lr=args.lr,
        sub_k=(args.sub_k if args.sub_k > 0 else None),
        d_hidden=args.d_hidden,
        freeze_mag_B=True, lr_decay=True,
        pauli_max_weight=args.pauli_max_weight,
        n_restarts=args.restarts,
        selector=args.selector,
        val_frac=args.val_frac,
    )

    grid = []
    for L in Ls:
        for k_t in k_totals:
            k_z = max(1, int(round(args.kz_frac * k_t)))
            k_r = max(1, k_t - k_z)
            for s in seeds:
                if _cell_path(out, L, k_z, k_r, s).exists():
                    continue
                grid.append((L, k_z, k_r, s, str(out),
                             args.threads_per_worker, args.device, kwargs))

    total_planned = (len(Ls) * len(k_totals) * len(seeds))
    print(f"grid: {total_planned} cells planned, {len(grid)} todo, "
          f"workers={args.workers}, threads/worker={args.threads_per_worker}",
          flush=True)
    if args.dry_run:
        return
    if not grid:
        print("nothing to do.", flush=True); return

    import time
    t0 = time.time()
    if args.device.startswith("cuda"):
        # Sequential on GPU (one process, one device).
        for n, job in enumerate(grid, 1):
            rec = _worker(job)
            elapsed = time.time() - t0
            rate = n / elapsed
            eta = (len(grid) - n) / rate if rate > 0 else float("inf")
            print(f"  [{n:>4}/{len(grid)}] L={rec['L']} "
                  f"kz={rec['k_z']:>5} kr={rec['k_random']:>5} "
                  f"s={rec['seed']}: fid={rec['fidelity']:.4f} "
                  f"rel={rec['rel_err']:+.3f}  ({rec['elapsed_sec']:5.1f}s)  "
                  f"ETA={eta/60:.1f}m", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for n, rec in enumerate(pool.imap_unordered(_worker, grid), 1):
                elapsed = time.time() - t0
                rate = n / elapsed
                eta = (len(grid) - n) / rate if rate > 0 else float("inf")
                print(f"  [{n:>4}/{len(grid)}] L={rec['L']} "
                      f"kz={rec['k_z']:>5} kr={rec['k_random']:>5} "
                      f"s={rec['seed']}: fid={rec['fidelity']:.4f} "
                      f"rel={rec['rel_err']:+.3f}  ({rec['elapsed_sec']:5.1f}s)  "
                      f"ETA={eta/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
