"""Directly reproduce the advisor's claim with our pipeline at the
matching sample budget. n=8, depth=10, fixed circuit, k ranging from
n=8 (sub-n^2) through n^2=64 up to ~D=256, 16 seeds.

Report XEB, KL, F_cl, TV alongside the population 'ideal XEB' that XEB
asymptotes to under PT (regardless of distributional learning)."""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import numpy as np
import torch

from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import exact_probabilities
from rcs import run_rcs_xeb_cell


def main():
    n = 8
    depth = 10
    circuit_seed = 0
    seeds = list(range(16))
    ks = [16, 32, 64, 128, 256, 1024, 10000]   # bracket n^2=64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    qubits, circuit = make_rcs_circuit(*grid_for(n), depth, seed=circuit_seed)
    p_C = exact_probabilities(circuit, qubits)
    dim = len(p_C)
    ideal = dim * float((p_C ** 2).sum()) - 1.0
    H_pC = -float((p_C[p_C > 0] * np.log(p_C[p_C > 0])).sum())
    print(f"n={n} depth={depth} circuit_seed={circuit_seed}")
    print(f"  D={dim}, ideal XEB = {ideal:.4f}, H(p_C) = {H_pC:.4f}")
    print()

    results = defaultdict(list)
    for k in ks:
        for seed in seeds:
            t0 = time.time()
            out = run_rcs_xeb_cell(
                n=n, depth=depth, k_train=k, m_candidates=5000,
                seed=seed + 100000,  # not the circuit seed
                d_hidden=64, epochs=400, lr=2e-3, batch_size=min(64, k),
                k_held=500, device=device, verbose=False,
                circuit_cache=(circuit, qubits, p_C),
            )
            results[k].append(out)
        elapsed = time.time() - t0
        # Median over seeds
        xebs = [r["candidate_xeb"] for r in results[k]]
        kls = [r["kl_model_vs_truth"] for r in results[k]]
        fcls = [r["classical_fidelity"] for r in results[k]]
        tvs = [r["tv_distance"] for r in results[k]]
        print(f"  k={k:>5}: XEB={np.median(xebs):+.4f}±{np.std(xebs):.3f}  "
              f"KL={np.median(kls):>7.3f}  F_cl={np.median(fcls):.3f}  "
              f"TV={np.median(tvs):.3f}  ({np.std(xebs):.3f} XEB sd over 16 seeds)",
              flush=True)

    out_path = Path(__file__).resolve().parents[1] / "results" / "m_xeb_vs_kl_at_nsq.json"
    out_path.write_text(json.dumps({
        "ideal_xeb": ideal, "H_pC": H_pC, "n": n, "depth": depth,
        "circuit_seed": circuit_seed, "seeds": seeds, "ks": ks,
        "cells": {str(k): results[k] for k in ks},
    }, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
