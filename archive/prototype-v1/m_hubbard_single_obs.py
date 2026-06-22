"""Single-observable Hubbard test: train SignedARRNN (no sign head) via
NLL on bitstring samples drawn from |psi_0|^2. Track error in the
single weight-3 Z observable IIZIZZII (Z on qubits 2, 4, 5; 0-indexed
qubit 0 is leftmost).

The shadow-noise floor on this single observable is
  std[hat<O>] = sqrt((1 - <O>^2)/k) ~ 1/sqrt(k)
The model's <O>_theta equals q_emp's <O> after NLL training collapses
to the empirical distribution, so we expect model error ≈ shadow error.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from m9.hubbard import Hubbard, state_int_to_bits
from shadows import SignedARRNN


def parity_chi(bits, S):
    """chi_S(x) = (-1)^(sum_{i in S} x_i) for each row in bits."""
    return np.where(bits[:, S].sum(axis=1) & 1 == 0, 1.0, -1.0).astype(np.float64)


def run_cell(ctx, k, seed, S, chi_sector, x_bits_all_t, d_hidden=32,
             epochs=400, lr=2e-3, device="cpu"):
    L = ctx.L
    p_true = ctx.psi_0 ** 2
    p_true = p_true / p_true.sum()
    rng = np.random.default_rng(seed)

    # Draw k Z-basis bitstring samples
    sample_idx = rng.choice(len(ctx.states), size=k, p=p_true)
    train_states = ctx.states[sample_idx]
    train_bits = state_int_to_bits(train_states, L)

    # Shadow estimate of <O>
    train_chi = parity_chi(train_bits, S)
    shadow_O = float(train_chi.mean())

    # NLL training via unique-strings collapsing
    unique_states, counts = np.unique(train_states, return_counts=True)
    unique_bits = state_int_to_bits(unique_states, L)
    u_bits_t = torch.from_numpy(unique_bits.astype(np.int64)).long().to(device)
    w_t = torch.from_numpy(counts.astype(np.float64) / counts.sum()).to(device)

    torch.manual_seed(seed)
    model = SignedARRNN(L=L, n_up=L // 2, n_dn=L // 2,
                         d_hidden=d_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr / 100)
    for ep in range(epochs):
        _, log_p = model._features_and_logq(u_bits_t)
        nll = -(w_t * log_p.double()).sum()
        opt.zero_grad(); nll.backward(); opt.step(); sched.step()

    # Evaluate <O>_model from full model distribution over the sector
    with torch.no_grad():
        _, log_p_all = model._features_and_logq(x_bits_all_t)
        p_model = torch.softmax(log_p_all, dim=0).cpu().numpy()
    model_O = float((p_model * chi_sector).sum())

    return {
        "k": k, "seed": seed, "shadow_O": shadow_O, "model_O": model_O,
        "n_unique_train": int(len(unique_states)),
    }


def main():
    L = 4
    U = 4.0
    S = [2, 4, 5]   # IIZIZZII (qubit 0 leftmost; Z on qubits 2,4,5; weight 3)
    ks = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    seeds = list(range(32))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ctx = Hubbard(L=L, U=U)
    n = 2 * L
    p_true = ctx.psi_0 ** 2
    p_true = p_true / p_true.sum()
    sector_bits = state_int_to_bits(ctx.states, L)
    chi_sector = parity_chi(sector_bits, S)
    true_O = float((p_true * chi_sector).sum())
    print(f"Hubbard L={L}, sector dim={len(ctx.states)}")
    print(f"Observable IIZIZZII (support {S}, weight {len(S)})")
    print(f"  TRUE <O> = {true_O:.6f}")
    print(f"  ks={ks}, seeds={seeds}\n", flush=True)

    x_bits_all_t = torch.from_numpy(
        sector_bits.astype(np.int64)).long().to(device)

    results = defaultdict(list)
    t0 = time.time()
    for k in ks:
        t_k = time.time()
        for seed in seeds:
            out = run_cell(ctx, k=k, seed=seed, S=S, chi_sector=chi_sector,
                            x_bits_all_t=x_bits_all_t, device=device)
            results[k].append(out)
        shadow_errs = [abs(r["shadow_O"] - true_O) for r in results[k]]
        model_errs = [abs(r["model_O"] - true_O) for r in results[k]]
        print(f"  k={k:>6}: shadow_err med={np.median(shadow_errs):.4f}  "
              f"model_err med={np.median(model_errs):.4f}  "
              f"({time.time() - t_k:.1f}s)", flush=True)

    out_path = Path(__file__).resolve().parents[1] / "results" / "m_hubbard_single_obs.json"
    out_path.write_text(json.dumps({
        "L": L, "U": U, "S": S, "true_O": true_O, "ks": ks, "seeds": seeds,
        "cells": {str(k): results[k] for k in ks},
    }))
    print(f"\ntotal: {(time.time() - t0):.1f}s")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
