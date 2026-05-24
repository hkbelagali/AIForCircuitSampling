"""M2: Stage 3 RCS minimum viable.

Setup: n=8 (2x4 grid), depth=10, Sycamore-style brickwork via cirq's
random_quantum_circuit_generation with cirq_google.SYC entangler — the same
methodology as recirq.random_circuit_sampling.

Pipeline:
  1. Build circuit, compute exact p_C over all 2^n bitstrings.
  2. Draw k training samples from p_C; verify linear XEB ~ 1 (Porter-Thomas).
  3. Train a small AR transformer on the training samples.
  4. Sample m candidates from the trained model.
  5. Compute novel-XEB vs Hamming radius r from training set (Headline A
     from DESIGN.md §4.3), compared against:
       - nearest-neighbor-noise baseline (random 1-bit flips from training)
       - uniform-random over {0,1}^n
  6. Plot.

Prediction (H3.1): novel-XEB decays rapidly as r grows.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import (
    bits_to_int,
    exact_probabilities,
    int_to_bits,
    sample_from_circuit,
)
from aics.circuits.xeb import linear_xeb, novel_xeb_vs_radius
from aics.models.ar_transformer import ARTransformer, train_ar


def nn_noise_baseline(training_int, m, n, h, rng):
    """m candidates by flipping h random bits in random training strings."""
    if len(training_int) == 0:
        return np.zeros(0, dtype=np.int64)
    seeds = rng.choice(training_int, size=m, replace=True)
    out = np.empty(m, dtype=np.int64)
    for i, s in enumerate(seeds):
        new = int(s)
        positions = rng.choice(n, size=h, replace=False)
        for b in positions:
            new ^= (1 << int(b))
        out[i] = new
    return out


def main():
    # ---- config ----
    n = 8
    depth = 10
    k_train = 100
    m_candidates = 2000
    circuit_seed = 0
    torch_seed = 0
    rng = np.random.default_rng(0)

    n_rows, n_cols = grid_for(n)
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"=== M2 minimum viable: Sycamore-style RCS ===")
    print(f"  n = {n} ({n_rows}x{n_cols} grid)   depth = {depth}   "
          f"k_train = {k_train}   m_candidates = {m_candidates}")

    # ---- circuit + exact p_C ----
    qubits, circuit = make_rcs_circuit(n_rows, n_cols, depth, seed=circuit_seed)
    print(f"  circuit moments: {len(circuit)}")
    p_C = exact_probabilities(circuit, qubits)
    assert np.isclose(p_C.sum(), 1.0, atol=1e-9), f"p_C sum = {p_C.sum()}"
    dim = len(p_C)
    print(f"  dim = {dim}, p_C.max = {p_C.max():.4f}, "
          f"effective entropy = {-(p_C[p_C>0] * np.log(p_C[p_C>0])).sum():.3f} nats")

    # ---- sample training data ----
    training_int = sample_from_circuit(circuit, qubits, k_train, seed=circuit_seed)
    training_unique = np.unique(training_int)
    F_train = linear_xeb(training_int, p_C)
    print(f"\n  unique training strings: {len(training_unique)} / {k_train}")
    print(f"  linear XEB on training samples: {F_train:.4f}  (Porter-Thomas ~ 1)")

    # ---- train AR transformer ----
    print(f"\n  training AR transformer (d_model=64, layers=2, n_heads=4)...")
    torch.manual_seed(torch_seed)
    model = ARTransformer(n_qubits=n, d_model=64, n_layers=2, n_heads=4)
    X_train = int_to_bits(training_int, n)
    final_nll = train_ar(model, X_train, n_epochs=300, lr=2e-3, batch_size=32,
                         verbose=True, log_every=50)
    print(f"  final mean NLL: {final_nll:.4f}   "
          f"(uniform-random would be {n * np.log(2):.4f})")

    # ---- sample candidates from model ----
    model.eval()
    cand_bits = model.sample(m_candidates).cpu().numpy()
    cand_int = bits_to_int(cand_bits)
    cand_unique = np.unique(cand_int)
    print(f"\n  AR samples drawn: {m_candidates}  (unique: {len(cand_unique)})")
    F_cand = linear_xeb(cand_int, p_C)
    print(f"  linear XEB on AR samples (with duplicates): {F_cand:.4f}")

    # ---- baselines for novel-XEB comparison ----
    nn_int = nn_noise_baseline(training_int, m_candidates, n, h=1, rng=rng)
    unif_int = rng.integers(0, dim, size=m_candidates, dtype=np.int64)

    radii = list(range(0, n + 1))
    nxeb_ar = novel_xeb_vs_radius(cand_int, training_int, p_C, n, radii=radii)
    nxeb_nn = novel_xeb_vs_radius(nn_int, training_int, p_C, n, radii=radii)
    nxeb_un = novel_xeb_vs_radius(unif_int, training_int, p_C, n, radii=radii)

    print(f"\n  novel-XEB vs Hamming radius from training set:")
    print(f"  {'r':>3} | {'AR':>9} {'AR n_cand':>10} | {'NN-noise':>9} | "
          f"{'uniform':>9}")
    for r in radii:
        v_ar, c_ar = nxeb_ar[r]
        v_nn, _ = nxeb_nn[r]
        v_un, _ = nxeb_un[r]
        print(f"  {r:>3} | {v_ar:>9.4f} {c_ar:>10d} | "
              f"{v_nn:>9.4f} | {v_un:>9.4f}")

    # ---- plot ----
    def y(d): return [d[r][0] for r in radii]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(radii, y(nxeb_ar), "o-", color="#e76f51", linewidth=2,
            label=f"AR transformer  (k={k_train}, m={m_candidates})")
    ax.plot(radii, y(nxeb_nn), "s--", color="#f4a261",
            label="NN-noise  (1-bit flips from training)")
    ax.plot(radii, y(nxeb_un), "d:", color="#888",
            label=r"uniform random over $\{0,1\}^n$")
    ax.axhline(1.0, color="#2a9d8f", linewidth=0.8, linestyle="--", alpha=0.6,
               label=r"ideal $p_C$ samples  ($\mathbb{E}[F_{XEB}] = 1$)")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6,
               label=r"uniform-random expectation  ($\mathbb{E}[F_{XEB}] = 0$)")
    ax.set_xlabel("Hamming radius $r$ from training set")
    ax.set_ylabel(r"$\widehat{F}^{\rm novel}_{\rm XEB}(r)$")
    ax.set_title(f"M2 Headline A: novel-XEB vs Hamming radius   "
                 f"(RCS n={n}, depth={depth}, k={k_train})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "m2_rcs_novel_xeb_vs_radius.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
