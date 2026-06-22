"""Smoke test: n=8 Sycamore-style RCS, depth=10, k_train=1000.

Train the BitstringARRNN (battle-tested GRU body, no sign head, no sector
mask) on bitstring samples from the circuit, then evaluate linear XEB and
novel-XEB vs Hamming radius on candidates sampled from the model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rcs import run_rcs_xeb_cell


def main():
    n = 8
    depth = 10
    k_train = 1000
    m_candidates = 5000
    seed = 0

    print(f"=== RCS+XEB smoke test (n={n}, depth={depth}, k_train={k_train}) ===")
    out = run_rcs_xeb_cell(n=n, depth=depth, k_train=k_train,
                            m_candidates=m_candidates, seed=seed,
                            d_hidden=64, epochs=400, lr=2e-3, batch_size=64)

    radii = out["radii"]
    print(f"\n  novel-XEB vs Hamming radius from training set:")
    print(f"  {'r':>3} | {'model':>9} {'n_cand':>7} | {'uniform':>9}")
    for r in radii:
        v_m, c_m = out["nxeb_model"][r]
        v_u, _ = out["nxeb_uniform"][r]
        print(f"  {r:>3} | {v_m:>9.4f} {c_m:>7d} | {v_u:>9.4f}")


if __name__ == "__main__":
    main()
