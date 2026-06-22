"""Reproduce the RCS k-sweep with a Porter–Thomas entropy penalty.

Same circuit (n=8, depth=10, circuit_seed=0), same 16 sample/init seeds,
same k grid. New axis: pt_penalty_lambda ∈ {0, 0.1, 1.0, 10.0}.

Hypothesis under test: a PT entropy regularizer can keep the model from
collapsing to the q_emp baseline at small k, by forcing high-entropy
distributions. Measure: classical fidelity, held-out XEB, comparison
to memorization baseline.
"""

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
    ks = [16, 32, 64, 128, 256, 1024, 10000]
    lambdas = [0.0, 0.1, 1.0, 10.0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    qubits, circuit = make_rcs_circuit(*grid_for(n), depth, seed=circuit_seed)
    p_C = exact_probabilities(circuit, qubits)
    dim = len(p_C)
    ideal = dim * float((p_C ** 2).sum()) - 1.0
    H_pC = -float((p_C[p_C > 0] * np.log(p_C[p_C > 0])).sum())
    H_PT = float(np.log(dim) - (1 - 0.5772156649015329))
    print(f"n={n} depth={depth} circuit_seed={circuit_seed}")
    print(f"  D={dim}, ideal XEB = {ideal:.4f}, H(p_C) = {H_pC:.4f}, "
          f"H_PT target = {H_PT:.4f}")
    print(f"  lambdas = {lambdas}, ks = {ks}, seeds = {seeds}\n", flush=True)

    results = {}
    for lam in lambdas:
        print(f"=== lambda = {lam} ===", flush=True)
        results[lam] = defaultdict(list)
        for k in ks:
            t0 = time.time()
            for seed in seeds:
                out = run_rcs_xeb_cell(
                    n=n, depth=depth, k_train=k, m_candidates=5000,
                    seed=seed + 100000,
                    d_hidden=64, epochs=400, lr=2e-3, batch_size=min(64, k),
                    k_held=500, device=device, verbose=False,
                    circuit_cache=(circuit, qubits, p_C),
                    pt_penalty_lambda=lam, pt_penalty_target=H_PT,
                )
                results[lam][k].append(out)
            xebs = [r["candidate_xeb"] for r in results[lam][k]]
            kls = [r["kl_model_vs_truth"] for r in results[lam][k]]
            fcls = [r["classical_fidelity"] for r in results[lam][k]]
            mhs = [r["model_entropy"] for r in results[lam][k]]
            nmass = [r["novel_mass"] for r in results[lam][k]]
            elapsed = time.time() - t0
            print(f"  k={k:>5}: XEB={np.median(xebs):+.3f}  KL={np.median(kls):.3f}  "
                  f"F_cl={np.median(fcls):.3f}  H(mod)={np.median(mhs):.3f}  "
                  f"novel_mass={np.median(nmass):.3f}  ({elapsed:.1f}s)",
                  flush=True)
        print()

    out_path = Path(__file__).resolve().parents[1] / "results" / "m_rcs_pt_penalty.json"
    out_path.write_text(json.dumps({
        "ideal_xeb": ideal, "H_pC": H_pC, "H_PT": H_PT, "n": n, "depth": depth,
        "circuit_seed": circuit_seed, "seeds": seeds, "ks": ks,
        "lambdas": lambdas,
        "cells": {str(lam): {str(k): results[lam][k] for k in ks}
                   for lam in lambdas},
    }, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
