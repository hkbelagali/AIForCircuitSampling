"""Variant of m12_sign_learning_smoketest.py that uses sample-based MLE
(k Born-rule samples) for magnitude pretrain, matching the actual M9 protocol.
Tests whether joint magnitude + sign learning works when we don't hand the
model the exact distribution.
"""

import argparse
import numpy as np
import torch

from aics.chemistry.amplitude_sampling import sample_from_amplitudes, state_int_to_bits
from aics.chemistry.local_energy import make_hubbard_context
from aics.chemistry.local_energy_signed import local_energy_hubbard_signed
from aics.eval.energy import model_energy_exact_signed
from aics.models.ar_rnn import ARRNN
from aics.training.mle_pretrain import train_rnn_mle
from aics.training.vmc import VMCConfig, VMCState, vmc_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--U", type=float, default=4.0)
    p.add_argument("--k", type=int, default=200,
                   help="number of training samples (M9 protocol)")
    p.add_argument("--d-hidden", type=int, default=32)
    p.add_argument("--mle-epochs", type=int, default=100)
    p.add_argument("--vmc-steps", type=int, default=5000)
    p.add_argument("--vmc-lr", type=float, default=3e-3)
    p.add_argument("--vmc-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    L = args.L
    ctx = make_hubbard_context(L, t=1.0, U=args.U, pbc=True)
    print(f"L={L}, U={args.U}, k={args.k}: sector dim = {len(ctx.states)}, "
          f"E_0 = {ctx.E_0:.6f}", flush=True)

    model = ARRNN(L=L, n_up=L // 2, n_dn=L // 2,
                  d_hidden=args.d_hidden, learn_signs=True)

    states_bits = state_int_to_bits(ctx.states, L)

    # --- 1. Sample k bitstrings from |psi_0|^2 and MLE pretrain. ------------
    print(f"\n[1/2] MLE pretrain on k={args.k} samples ({args.mle_epochs} epochs)...",
          flush=True)
    bits, _, _ = sample_from_amplitudes(ctx.psi_0, ctx.states, L, args.k, rng)
    train_rnn_mle(model, bits, n_epochs=args.mle_epochs, lr=2e-3, batch_size=32)

    with torch.no_grad():
        log_mag_all = model.log_psi_mag(
            torch.from_numpy(states_bits.astype(np.int64)).long()
        ).cpu().numpy()
        s_init = model.soft_sign(
            torch.from_numpy(states_bits.astype(np.int64)).long()
        ).cpu().numpy()
    q_model = np.exp(log_mag_all) ** 2
    q_model /= q_model.sum()
    p_target = (np.abs(ctx.psi_0) ** 2)
    p_target /= p_target.sum()
    tv = 0.5 * np.sum(np.abs(q_model - p_target))
    init_match = max(
        (np.sign(s_init) == ctx.signs).mean(),
        (np.sign(s_init) == -ctx.signs).mean(),
    )
    print(f"     post-pretrain TV = {tv:.4f},  sign-match (random) = {init_match:.2%}",
          flush=True)

    # --- 2. VMC with signed local energy. ------------------------------------
    print(f"\n[2/2] VMC ({args.vmc_steps} steps, lr={args.vmc_lr})...", flush=True)
    cfg = VMCConfig(lr=args.vmc_lr, batch_size=args.vmc_batch, clip_norm=1.0)
    state = VMCState()

    E0 = ctx.E_0
    steps_to_threshold = None
    for step in range(1, args.vmc_steps + 1):
        vmc_step(model, ctx, state, cfg, local_energy_hubbard_signed, signed=True)
        if step % args.eval_every == 0 or step == 1 or step == args.vmc_steps:
            E = model_energy_exact_signed(model, ctx)
            rel = (E - E0) / abs(E0)
            with torch.no_grad():
                s = model.soft_sign(
                    torch.from_numpy(states_bits.astype(np.int64)).long()
                ).cpu().numpy()
                lm = model.log_psi_mag(
                    torch.from_numpy(states_bits.astype(np.int64)).long()
                ).cpu().numpy()
            mr = max((np.sign(s) == ctx.signs).mean(),
                     (np.sign(s) == -ctx.signs).mean())
            sat = np.mean(np.abs(s) > 0.9)
            q = np.exp(lm) ** 2; q /= q.sum()
            tv_now = 0.5 * np.sum(np.abs(q - p_target))
            print(f"     step {step:>4d}: <H>={E:+.5f}  rel={rel:+.4f}  "
                  f"sign_match={mr:.2%}  saturated={sat:.2%}  TV={tv_now:.3f}",
                  flush=True)
            if steps_to_threshold is None and abs(rel) <= 0.01:
                steps_to_threshold = step

    print(f"\n[summary] steps to |rel error| <= 0.01: "
          f"{steps_to_threshold if steps_to_threshold else 'NOT REACHED'}")


if __name__ == "__main__":
    main()
