"""Peaked-circuit version of m_xeb_vs_kl_at_nsq: n=8, fixed peaked
circuit, 16 sample seeds, full metric panel including held-out XEB
and memorization baselines."""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from peaked import build_peaked_pC
from rcs import run_rcs_xeb_cell


def main():
    n = 8
    depth_rqc = n
    depth_pqc = n // 2
    circuit_seed = 0
    seeds = list(range(16))
    ks = [16, 32, 64, 128, 256, 1024, 10000]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    peaked = build_peaked_pC(n=n, depth_rqc=depth_rqc, depth_pqc=depth_pqc,
                              seed=circuit_seed, n_iters=2000, lr=0.05,
                              device=device, verbose=True)
    p_C = peaked["p_C"]
    dim = len(p_C)
    ideal = dim * float((p_C ** 2).sum()) - 1.0
    H_pC = -float((p_C[p_C > 0] * np.log(p_C[p_C > 0])).sum())
    print(f"\nPEAKED n={n} depth_rqc={depth_rqc} depth_pqc={depth_pqc} "
          f"circuit_seed={circuit_seed}")
    print(f"  D={dim}, ideal XEB = {ideal:.4f}, H(p_C) = {H_pC:.4f}")
    print(f"  peak at idx={peaked['peak_idx']:>3d}, prob={peaked['peak_prob']:.4f}")
    print()

    results = defaultdict(list)
    for k in ks:
        for seed in seeds:
            out = run_rcs_xeb_cell(
                n=n, depth=0, k_train=k, m_candidates=5000,
                seed=seed + 200000,
                d_hidden=64, epochs=400, lr=2e-3, batch_size=min(64, k),
                k_held=500, device=device, verbose=False,
                p_C_only=p_C,
            )
            results[k].append(out)
        xebs = [r["candidate_xeb"] for r in results[k]]
        kls = [r["kl_model_vs_truth"] for r in results[k]]
        fcls = [r["classical_fidelity"] for r in results[k]]
        tvs = [r["tv_distance"] for r in results[k]]
        print(f"  k={k:>5}: XEB={np.median(xebs):+8.4f}±{np.std(xebs):.3f}  "
              f"KL={np.median(kls):>7.3f}  F_cl={np.median(fcls):.3f}  "
              f"TV={np.median(tvs):.3f}", flush=True)

    out_path = Path(__file__).resolve().parents[1] / "results" / "m_peaked_xeb_vs_kl.json"
    out_path.write_text(json.dumps({
        "ideal_xeb": ideal, "H_pC": H_pC, "n": n,
        "depth_rqc": depth_rqc, "depth_pqc": depth_pqc,
        "circuit_seed": circuit_seed, "seeds": seeds, "ks": ks,
        "peak_idx": peaked["peak_idx"], "peak_prob": peaked["peak_prob"],
        "cells": {str(k): results[k] for k in ks},
    }, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
