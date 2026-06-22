"""Diagnose the overfitting picture from the smoke test:
 - held-out NLL on a fresh 1000 samples from p_C (vs the train-set NLL)
 - fraction of model probability mass on the unique training strings
 - 'uniform-over-unique-training' candidate-XEB baseline
 - rank-1 baseline: most-frequent training string repeatedly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import numpy as np
import torch

from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import (
    bits_to_int, exact_probabilities, int_to_bits, sample_from_circuit,
)
from aics.circuits.xeb import linear_xeb
from rcs import BitstringARRNN, train_nll


def main():
    n, depth = 8, 10
    k_train, k_held = 1000, 1000
    m = 5000
    seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows, cols = grid_for(n)
    qubits, circuit = make_rcs_circuit(rows, cols, depth, seed=seed)
    p_C = exact_probabilities(circuit, qubits)
    dim = len(p_C)
    H_true = float(-(p_C[p_C > 0] * np.log(p_C[p_C > 0])).sum())
    H_uniform = n * float(np.log(2))

    train_int = sample_from_circuit(circuit, qubits, k_train, seed=seed)
    held_int = sample_from_circuit(circuit, qubits, k_held, seed=seed + 12345)
    train_uniq = np.unique(train_int)

    print(f"dim={dim}  k_train={k_train}  unique_train={len(train_uniq)}  k_held={k_held}")
    print(f"H(p_C) = {H_true:.4f}   H(uniform) = {H_uniform:.4f}")
    print(f"train XEB = {linear_xeb(train_int, p_C):.4f}   held XEB = {linear_xeb(held_int, p_C):.4f}")

    X_tr = torch.from_numpy(int_to_bits(train_int, n)).long().to(device)
    X_held = torch.from_numpy(int_to_bits(held_int, n)).long().to(device)
    torch.manual_seed(seed)
    model = BitstringARRNN(n_qubits=n, d_hidden=64).to(device)
    final_train_nll = train_nll(model, X_tr, epochs=400, lr=2e-3,
                                  batch_size=64, verbose=False)
    model.eval()
    with torch.no_grad():
        held_nll = float(-model.log_prob(X_held).mean())
        # full distribution over all 2^n bitstrings
        all_int = np.arange(dim, dtype=np.int64)
        all_bits = torch.from_numpy(int_to_bits(all_int, n)).long().to(device)
        full_logp = model.log_prob(all_bits).cpu().numpy()
        full_p = np.exp(full_logp - full_logp.max())
        full_p /= full_p.sum()
        # candidate samples
        cand_bits = model.sample(m).cpu().numpy()
        cand_int = bits_to_int(cand_bits)

    cand_unique = np.unique(cand_int)
    cand_in_train = np.isin(cand_int, train_uniq).mean()
    mass_on_train = float(full_p[train_uniq].sum())
    H_model = float(-(full_p[full_p > 0] * np.log(full_p[full_p > 0])).sum())
    kl_model_vs_truth = float((full_p[full_p > 0] *
                               (np.log(full_p[full_p > 0]) -
                                np.log(np.maximum(p_C[full_p > 0], 1e-300)))).sum())

    print()
    print(f"train NLL    = {final_train_nll:.4f}   (below H(p_C) ⇒ overfitting)")
    print(f"held NLL     = {held_nll:.4f}")
    print(f"H(model)     = {H_model:.4f}   KL(model || p_C) = {kl_model_vs_truth:.4f}")
    print(f"P_model(x in train_unique) = {mass_on_train:.4f}  "
          f"(uniform would be {len(train_uniq)/dim:.4f})")
    print()
    print(f"candidate unique = {len(cand_unique)} / {m}   "
          f"in training = {cand_in_train:.1%}")
    print(f"candidate XEB         = {linear_xeb(cand_int, p_C):.4f}")

    # baselines on the candidate set with the same m
    rng = np.random.default_rng(seed + 99)
    unif_train = rng.choice(train_uniq, size=m, replace=True)
    cnts = np.bincount(train_int, minlength=dim)
    train_prob = cnts / cnts.sum()
    weighted_train = rng.choice(dim, size=m, p=train_prob)
    most_freq = np.full(m, np.argmax(cnts), dtype=np.int64)

    print(f"uniform-over-unique-train XEB = {linear_xeb(unif_train, p_C):.4f}")
    print(f"reweight-by-train-counts XEB  = {linear_xeb(weighted_train, p_C):.4f}")
    print(f"most-frequent-train XEB       = {linear_xeb(most_freq, p_C):.4f}")


if __name__ == "__main__":
    main()
