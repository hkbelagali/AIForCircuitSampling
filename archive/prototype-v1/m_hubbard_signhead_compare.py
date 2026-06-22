"""Hubbard L=4: train SignedARRNN with vs without sign head on the same
random-Pauli loss. Sweep (k_train, w_max). Collect energy, fidelity, TV.

With sign head:  fidelity = quantum fidelity |<psi_model|psi_0>|^2
Without sign head: fidelity = classical fidelity (sum sqrt(p p_C))^2

Both models train on full random-Pauli targets (X, Y, Z) of weight ≤ w_max.
The unsigned model is constrained to non-negative amplitudes via
model.psi(..., use_sign=False) and therefore can't fit off-diagonal targets
that require sign structure — this is the difference we're measuring.
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
from shadows import SignedARRNN, sample_shadows_random_pauli
import pauli_loss
from hubbard_z_pauli import (
    enumerate_z_supports, parity_matrix_on_states,
    shadow_z_expectations_from_sample_states,
)


def run_cell(ctx, k_train, w_max, seed, use_sign_head, n_restarts=1,
             d_hidden=32, epochs=2000, lr=1e-3, device="cpu"):
    """Train SignedARRNN with sign head on/off. Optional n_restarts: do
    multiple independent inits sharing the same shadow data, keep the
    lowest-training-loss model. Used to fix sign-head bistability at
    low w_max."""
    L = ctx.L
    rng = np.random.default_rng(seed)
    x_bits_t = torch.from_numpy(
        state_int_to_bits(ctx.states, L).astype(np.int64)).long().to(device)

    if use_sign_head:
        # Random-Pauli shadows + full Pauli loss (X, Y, Z up to w_max)
        U_r, b_r = sample_shadows_random_pauli(ctx.psi_0, ctx.states, L,
                                                k_train, rng)
        loss_paulis = pauli_loss.build_loss_paulis(ctx, w_max)
        targets_np = pauli_loss.shadow_targets(loss_paulis, U_r, b_r)
        ops = pauli_loss.torch_ops(loss_paulis, device)
        targets_t = torch.from_numpy(targets_np).double().to(device)
        alpha_t = torch.ones(loss_paulis["n_paulis"], dtype=torch.float64,
                              device=device)

        def model_exps(psi):
            return pauli_loss.model_expectations(psi, ops)
    else:
        # Z-only protocol: bitstring shots from |psi_0|^2 + Z-only Pauli loss
        n = 2 * L
        p_true = ctx.psi_0 ** 2
        p_true = p_true / p_true.sum()
        sample_idx = rng.choice(len(ctx.states), size=k_train, p=p_true)
        sample_states = ctx.states[sample_idx]
        supports, _ = enumerate_z_supports(n, w_max)
        targets_np = shadow_z_expectations_from_sample_states(
            sample_states, supports, L)
        W_np = parity_matrix_on_states(supports, ctx.states, L)
        W_t = torch.from_numpy(W_np).double().to(device)
        targets_t = torch.from_numpy(targets_np).double().to(device)
        alpha_t = torch.ones(len(supports), dtype=torch.float64, device=device)

        def model_exps(psi):
            p = psi.pow(2)
            return W_t @ p

    t0 = time.time()
    best_loss = float("inf")
    best_state = None
    for r_i in range(n_restarts):
        torch.manual_seed(seed * 17 + r_i * 7919 + (1 if use_sign_head else 0))
        model = SignedARRNN(L=L, n_up=L // 2, n_dn=L // 2,
                             d_hidden=d_hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr / 100)
        for ep in range(epochs):
            psi = model.psi(x_bits_t, use_sign=use_sign_head).to(torch.float64)
            psi = psi / (psi.pow(2).sum() + 1e-30).sqrt()
            exps = model_exps(psi)
            diff = exps - targets_t
            loss = (alpha_t * diff.pow(2)).sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        final = float(loss.detach())
        if final < best_loss:
            best_loss = final
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    with torch.no_grad():
        psi_np = model.psi(x_bits_t, use_sign=use_sign_head).cpu().numpy().astype(np.float64)
    nrm = np.linalg.norm(psi_np) or 1.0
    psi_np = psi_np / nrm

    # Probabilities & p_true
    p_model = psi_np ** 2
    p_true = ctx.psi_0 ** 2
    p_true = p_true / p_true.sum()
    p_model = p_model / p_model.sum()

    # Energy (always = <psi|H|psi> with the model's chosen amplitudes)
    E = float(psi_np @ (ctx.H @ psi_np))

    # Fidelity
    if use_sign_head:
        QF = float((ctx.psi_0 @ psi_np) ** 2)
    else:
        # classical (Bhattacharyya) fidelity — magnitude-only
        QF = float(np.square(np.sqrt(np.maximum(p_model * p_true, 0)).sum()))

    TV = 0.5 * float(np.abs(p_model - p_true).sum())

    return {
        "L": L, "k_train": k_train, "w_max": w_max, "seed": seed,
        "use_sign_head": use_sign_head, "d_hidden": d_hidden, "epochs": epochs,
        "E_model": E, "E_0": float(ctx.E_0),
        "rel_E_err": (E - ctx.E_0) / abs(ctx.E_0) if ctx.E_0 != 0 else 0.0,
        "fidelity": QF, "TV": TV, "elapsed_sec": elapsed,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_max", type=int, default=None,
                         help="only run this w_max (for SLURM array). default: run all")
    parser.add_argument("--use_sign_head", type=int, default=None,
                         help="only run this sign-head setting (0 or 1). default: both")
    parser.add_argument("--n_restarts_signed", type=int, default=4,
                         help="restarts for signed cells")
    parser.add_argument("--out_subdir", type=str, default="m_hubbard_signhead",
                         help="output dir name under results/")
    parser.add_argument("--seed", type=int, default=None,
                         help="only run this seed (for SLURM array). default: all 0..7")
    args = parser.parse_args()

    L = 4
    ctx = Hubbard(L=L, U=4.0)
    print(f"Hubbard L={L} (sector dim={len(ctx.states)}), E_0={ctx.E_0:.6f}")

    ks = [100, 300, 1000, 3000, 10000]
    w_maxes = [args.w_max] if args.w_max is not None else [1, 2, 3, 4]
    seeds = [args.seed] if args.seed is not None else list(range(8))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sign_options = ([bool(args.use_sign_head)] if args.use_sign_head is not None
                     else [True, False])

    out_dir = Path(__file__).resolve().parents[1] / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, output={out_dir}")
    print(f"ks={ks}, w_maxes={w_maxes}, seeds={seeds}, "
          f"n_restarts_signed={args.n_restarts_signed}\n", flush=True)

    t0 = time.time()
    for use_sign_head in sign_options:
        print(f"=== use_sign_head = {use_sign_head} ===", flush=True)
        for w_max in w_maxes:
            for k in ks:
                row = []
                for seed in seeds:
                    tag = f"L{L}_k{k}_w{w_max}_s{seed}_sign{int(use_sign_head)}"
                    p = out_dir / f"{tag}.json"
                    if p.exists():
                        out = json.loads(p.read_text())
                    else:
                        n_r = args.n_restarts_signed if use_sign_head else 1
                        out = run_cell(ctx, k_train=k, w_max=w_max, seed=seed,
                                        use_sign_head=use_sign_head,
                                        n_restarts=n_r, device=device)
                        p.write_text(json.dumps(out))
                    row.append(out)
                fids = [r["fidelity"] for r in row]
                Es = [r["E_model"] for r in row]
                tvs = [r["TV"] for r in row]
                print(f"  w={w_max} k={k:>5}: "
                      f"E={np.median(Es):+.4f} (E_0={ctx.E_0:+.4f})  "
                      f"Fid={np.median(fids):.4f}  TV={np.median(tvs):.4f}",
                      flush=True)
        print()
    print(f"total: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
