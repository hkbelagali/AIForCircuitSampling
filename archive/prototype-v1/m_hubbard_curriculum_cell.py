"""Hubbard L=4 curriculum cell: train SignedARRNN at w_min cold, then
warm-start each successive w from the previous w's best state. Mirrors
m_rcs_curriculum_cell.py for direct comparison.

CLI: one (seed, k_train, use_sign_head) per call. Loops w=1..4 internally.
"""

import argparse
import json
import sys
import time
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


def setup_targets(ctx, k_train, w_max, rng_state, use_sign_head, device):
    """Build (targets, model_exps_fn) for this w_max, reusing the shadow
    data sampled at the top of the cell (rng_state)."""
    rng = np.random.default_rng(rng_state)
    L = ctx.L
    if use_sign_head:
        U_r, b_r = sample_shadows_random_pauli(ctx.psi_0, ctx.states, L,
                                                k_train, rng)
        loss_paulis = pauli_loss.build_loss_paulis(ctx, w_max)
        targets_np = pauli_loss.shadow_targets(loss_paulis, U_r, b_r)
        ops = pauli_loss.torch_ops(loss_paulis, device)
        targets_t = torch.from_numpy(targets_np).double().to(device)
        alpha_t = torch.ones(loss_paulis["n_paulis"], dtype=torch.float64, device=device)

        def model_exps(psi):
            return pauli_loss.model_expectations(psi, ops)
    else:
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
            return W_t @ psi.pow(2)
    return targets_t, alpha_t, model_exps


def train_one(model, model_exps, targets_t, alpha_t, x_bits_t, use_sign_head,
              epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
    for _ in range(epochs):
        psi = model.psi(x_bits_t, use_sign=use_sign_head).to(torch.float64)
        psi = psi / (psi.pow(2).sum() + 1e-30).sqrt()
        exps = model_exps(psi)
        loss = (alpha_t * (exps - targets_t).pow(2)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return float(loss.detach())


def train_stage(L, d_hidden, x_bits_t, targets_t, alpha_t, model_exps, *,
                use_sign_head, n_restarts, epochs, lr, warm_state, seed,
                w_stage, device):
    best_loss = float("inf")
    best_state = None
    for r_i in range(n_restarts):
        if warm_state is None:
            torch.manual_seed(seed * 17 + r_i * 7919)
        else:
            torch.manual_seed(seed * 17 + r_i * 7919 + w_stage * 1_000_003)
        model = SignedARRNN(L=L, n_up=L // 2, n_dn=L // 2,
                             d_hidden=d_hidden).to(device)
        if r_i == 0 and warm_state is not None:
            model.load_state_dict(warm_state)
        final = train_one(model, model_exps, targets_t, alpha_t, x_bits_t,
                          use_sign_head, epochs, lr)
        if final < best_loss:
            best_loss = final
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return best_loss, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k_train", type=int, required=True)
    parser.add_argument("--use_sign_head", type=int, required=True,
                         help="0 or 1")
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--U", type=float, default=4.0)
    parser.add_argument("--d_hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w_min", type=int, default=1)
    parser.add_argument("--w_max", type=int, default=4)
    parser.add_argument("--n_restarts_cold", type=int, default=8)
    parser.add_argument("--n_restarts_warm", type=int, default=2)
    parser.add_argument("--out_subdir", type=str, default="m_hubbard_curriculum")
    args = parser.parse_args()

    use_sign_head = bool(args.use_sign_head)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = Hubbard(L=args.L, U=args.U)
    out_dir = Path(__file__).resolve().parents[1] / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Hubbard L={args.L} U={args.U}  seed={args.seed} k={args.k_train} "
          f"sign={int(use_sign_head)}  w∈[{args.w_min},{args.w_max}]  device={device}",
          flush=True)

    x_bits_t = torch.from_numpy(
        state_int_to_bits(ctx.states, args.L).astype(np.int64)).long().to(device)

    warm_state = None
    for w in range(args.w_min, args.w_max + 1):
        tag = f"L{args.L}_k{args.k_train}_w{w}_s{args.seed}_sign{int(use_sign_head)}"
        json_p = out_dir / f"{tag}.json"
        ckpt_p = out_dir / f"{tag}.pt"
        if json_p.exists() and ckpt_p.exists():
            warm_state = torch.load(ckpt_p, map_location=device, weights_only=True)
            cached = json.loads(json_p.read_text())
            print(f"  w={w}: cached  fid={cached['fidelity']:.4f}", flush=True)
            continue

        targets_t, alpha_t, model_exps = setup_targets(
            ctx, args.k_train, w, args.seed, use_sign_head, device)

        is_cold = (w == args.w_min) or (warm_state is None)
        n_r = args.n_restarts_cold if is_cold else args.n_restarts_warm

        t0 = time.time()
        best_loss, best_state = train_stage(
            args.L, args.d_hidden, x_bits_t, targets_t, alpha_t, model_exps,
            use_sign_head=use_sign_head, n_restarts=n_r,
            epochs=args.epochs, lr=args.lr,
            warm_state=None if is_cold else warm_state,
            seed=args.seed, w_stage=w, device=device,
        )
        elapsed = time.time() - t0

        model = SignedARRNN(L=args.L, n_up=args.L // 2, n_dn=args.L // 2,
                             d_hidden=args.d_hidden).to(device)
        model.load_state_dict(best_state)
        with torch.no_grad():
            psi_np = model.psi(x_bits_t, use_sign=use_sign_head).cpu().numpy().astype(np.float64)
        nrm = np.linalg.norm(psi_np) or 1.0
        psi_np = psi_np / nrm

        p_model = psi_np ** 2
        p_true = ctx.psi_0 ** 2
        p_true = p_true / p_true.sum()
        p_model = p_model / p_model.sum()
        E = float(psi_np @ (ctx.H @ psi_np))
        if use_sign_head:
            QF = float((ctx.psi_0 @ psi_np) ** 2)
        else:
            QF = float(np.square(np.sqrt(np.maximum(p_model * p_true, 0)).sum()))
        TV = 0.5 * float(np.abs(p_model - p_true).sum())

        rec = {
            "L": args.L, "k_train": args.k_train, "w_max": w, "seed": args.seed,
            "use_sign_head": use_sign_head, "d_hidden": args.d_hidden,
            "epochs": args.epochs, "n_restarts": n_r,
            "warm_started": (not is_cold),
            "E_model": E, "E_0": float(ctx.E_0),
            "rel_E_err": (E - ctx.E_0) / abs(ctx.E_0) if ctx.E_0 != 0 else 0.0,
            "final_loss": best_loss,
            "fidelity": QF, "TV": TV, "elapsed_sec": elapsed,
        }
        json_p.write_text(json.dumps(rec))
        torch.save(best_state, ckpt_p)
        warm_state = best_state
        print(f"  w={w}: fid={QF:.4f}  loss={best_loss:.3e}  "
              f"(restarts={n_r}, warm={not is_cold}, {elapsed:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
