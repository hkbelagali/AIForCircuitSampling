"""Compute p_ideal for the first N TN training samples in tn_pool.npz.

Writes an npz with the p_C values so we can compute per-k clean-samples XEB.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from aics.io import load_samples
from aics.sampling import amplitudes_tn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True,
                    help="tn_pool.npz (has train_bits from TN sampling)")
    p.add_argument("--circuit_py", required=True)
    p.add_argument("--n", type=int, required=True, help="how many bits to compute")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Import circuit
    import importlib.util
    spec = importlib.util.spec_from_file_location("circ", args.circuit_py)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    circ, qubits = mod.CIRCUIT, list(mod.QUBIT_ORDER)

    data = load_samples(args.pool)
    train_bits = np.asarray(data["train_bits"], dtype=np.uint8)[: args.n]
    print(f"[compute_pC] computing p_ideal for first {len(train_bits)} of "
          f"{len(data['train_bits'])} TN samples")

    t0 = time.time()
    pC = amplitudes_tn(circ, qubits, train_bits)
    print(f"[compute_pC] done in {time.time()-t0:.1f}s  "
          f"XEB={(1 << len(qubits)) * pC.mean() - 1:+.4f}")
    np.savez(args.out, train_bits=train_bits, train_pC=pC)
    print(f"[compute_pC] wrote {args.out}")


if __name__ == "__main__":
    main()
