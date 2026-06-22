"""Z-Pauli extrapolation test on Hubbard half-filled ground state.

Mirror of m_rcs_z_pauli.py — same architecture (AR-RNN), same training
loss, same per-weight evaluation — only difference is target distribution
(Hubbard GS p = |psi_0|^2 over the sector instead of RCS p_C over 2^N)
and the sector mask on the RNN.

Z observables only — no sign head needed since <Z_S> depends only on
|psi(x)|^2 regardless of sign.
"""

import json
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from m9.hubbard import Hubbard, state_int_to_bits
from shadows import SignedARRNN


def enumerate_z_supports(n, max_weight):
    supports = [()]
    weights = [0]
    for w in range(1, max_weight + 1):
        for S in combinations(range(n), w):
            supports.append(S); weights.append(w)
    return supports, np.asarray(weights, dtype=np.int64)


def parity_matrix_on_states(supports, states, L):
    """W of shape (n_obs, len(states)): W[i, k] = chi_{S_i}(states[k])."""
    bits = state_int_to_bits(states, L)  # (D, 2L), LSB-first qubit i at bit i
    W = np.ones((len(supports), len(states)), dtype=np.float64)
    for i, S in enumerate(supports):
        if not S:
            continue
        parity = bits[:, list(S)].sum(axis=1) & 1
        W[i] = np.where(parity == 0, 1.0, -1.0)
    return W


def shadow_z_expectations_from_sample_states(sample_states, supports, L):
    """Each sample is a sector state int; compute mean of chi_S over samples."""
    if len(sample_states) == 0:
        return np.zeros(len(supports), dtype=np.float64)
    bits = state_int_to_bits(sample_states, L)
    out = np.empty(len(supports), dtype=np.float64)
    for i, S in enumerate(supports):
        if not S:
            out[i] = 1.0; continue
        parity = bits[:, list(S)].sum(axis=1) & 1
        out[i] = float(np.where(parity == 0, 1.0, -1.0).mean())
    return out


def model_z_expectations_hubbard(model, ctx, supports, device, W_cache=None):
    """Forward all sector states, softmax-normalize, contract with parity matrix."""
    x_bits = torch.from_numpy(
        state_int_to_bits(ctx.states, ctx.L).astype(np.int64)).long().to(device)
    with torch.no_grad():
        _, log_p = model._features_and_logq(x_bits)
    p = torch.softmax(log_p, dim=0).double().cpu().numpy()
    W = W_cache if W_cache is not None else parity_matrix_on_states(
        supports, ctx.states, ctx.L)
    return W @ p


def train_signed_arrnn_z_pauli(model, ctx, supports, weights, sample_states, *,
                                 epochs=400, lr=2e-3, device=None, verbose=False):
    device = device or next(model.parameters()).device
    L = ctx.L
    targets_np = shadow_z_expectations_from_sample_states(sample_states, supports, L)
    W_np = parity_matrix_on_states(supports, ctx.states, L)
    W = torch.from_numpy(W_np).double().to(device)
    targets = torch.from_numpy(targets_np).double().to(device)
    x_bits = torch.from_numpy(
        state_int_to_bits(ctx.states, L).astype(np.int64)).long().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr / 100)
    for ep in range(epochs):
        _, log_p = model._features_and_logq(x_bits)
        p = torch.softmax(log_p, dim=0).double()
        exps = W @ p
        loss = (exps - targets).pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (ep % 80 == 0 or ep == epochs - 1):
            print(f"  ep {ep:>4}: loss={float(loss):.4e}", flush=True)
    return float(loss)


def run_z_pauli_cell_hubbard(L, k_train, w_train, seed, *,
                              d_hidden=64, epochs=400, lr=2e-3,
                              U=4.0, device=None, verbose=False, ctx=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if ctx is None:
        ctx = Hubbard(L=L, U=U)
    n = 2 * L
    rng = np.random.default_rng(seed)
    p_true = ctx.psi_0 ** 2
    p_true /= p_true.sum()
    sample_idx = rng.choice(len(ctx.states), size=k_train, p=p_true)
    sample_states = ctx.states[sample_idx]

    train_supports, train_w = enumerate_z_supports(n, w_train)
    torch.manual_seed(seed)
    model = SignedARRNN(L=L, n_up=L // 2, n_dn=L // 2,
                         d_hidden=d_hidden).to(device)
    t0 = time.time()
    final_loss = train_signed_arrnn_z_pauli(
        model, ctx, train_supports, train_w, sample_states,
        epochs=epochs, lr=lr, device=device, verbose=verbose)
    elapsed = time.time() - t0

    full_supports, full_w = enumerate_z_supports(n, n)
    W_full = parity_matrix_on_states(full_supports, ctx.states, L)
    true_exp = W_full @ p_true
    model_exp = model_z_expectations_hubbard(model, ctx, full_supports, device, W_cache=W_full)
    shadow_exp = shadow_z_expectations_from_sample_states(sample_states, full_supports, L)

    err_model = np.abs(model_exp - true_exp)
    err_shadow = np.abs(shadow_exp - true_exp)
    by_w_model, by_w_shadow, rms_true = {}, {}, {}
    for w in range(n + 1):
        mask = (full_w == w)
        if mask.sum() == 0:
            continue
        by_w_model[int(w)] = float(err_model[mask].mean())
        by_w_shadow[int(w)] = float(err_shadow[mask].mean())
        rms_true[int(w)] = float(np.sqrt(np.square(true_exp[mask]).mean()))

    return {
        "system": "hubbard", "L": L, "n": n, "U": U,
        "k_train": k_train, "w_train": w_train, "seed": seed,
        "d_hidden": d_hidden, "epochs": epochs,
        "final_loss": final_loss, "elapsed_sec": elapsed,
        "err_by_weight_model": by_w_model,
        "err_by_weight_shadow": by_w_shadow,
        "true_rms_by_weight": rms_true,
    }


def main():
    L = 4
    k_train = 2000
    w_trains = [1, 2, 3, 4, 5, 6, 7, 8]
    seeds = list(range(16))
    out_dir = Path(__file__).resolve().parents[1] / "results" / "m_hubbard_z_pauli"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ctx = Hubbard(L=L, U=4.0)
    n = 2 * L
    print(f"Hubbard L={L} (n={n}, sector dim={len(ctx.states)})  "
          f"k_train={k_train}  w_trains={w_trains}  seeds={seeds}\n", flush=True)
    t0 = time.time()
    for seed in seeds:
        for w_train in w_trains:
            cell_path = out_dir / f"L{L}_k{k_train}_w{w_train}_s{seed}.json"
            if cell_path.exists():
                print(f"  skip s={seed} w={w_train} (exists)", flush=True)
                continue
            t_cell = time.time()
            out = run_z_pauli_cell_hubbard(L=L, k_train=k_train, w_train=w_train,
                                            seed=seed, d_hidden=64, epochs=1500,
                                            lr=2e-3, device=device, ctx=ctx)
            cell_path.write_text(json.dumps(out))
            err = out["err_by_weight_model"]
            shadow = out["err_by_weight_shadow"]
            print(f"  s={seed} w_train={w_train}: loss={out['final_loss']:.3e}  ({time.time() - t_cell:.1f}s)",
                  flush=True)
            print(f"     model err per w: " +
                  "  ".join(f"w{w}={err[w]:.3f}" for w in range(1, n + 1)),
                  flush=True)
            print(f"     shadow err per w: " +
                  "  ".join(f"w{w}={shadow[w]:.3f}" for w in range(1, n + 1)),
                  flush=True)
    print(f"\ntotal: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
