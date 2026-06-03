"""Smoke test: can the ARRNN's learned sign head recover the ED sign structure
of the half-filled bipartite Hubbard GS via VMC alone?

Protocol (isolates the sign-learning question):
  1. Build Hubbard context at L=4.
  2. Pretrain the magnitude head via soft-target MLE on |psi_0|^2 (gives it
     the exact GS magnitudes, removing the magnitude-learning confound).
  3. Initialize the sign head randomly.
  4. Run VMC with signed local-energy + signed model-energy. The only gradient
     signal for the sign head is the variational energy.
  5. Report:
       - convergence of <H>_model toward E_0 over VMC steps,
       - fraction of states whose hard-sign(soft_sign) matches ED sign,
       - magnitude TV between |psi_model|^2 and |psi_0|^2 (should stay tiny
         since magnitude head was pretrained).

If this works at L=4, sign learning is feasible and we can integrate into M9.
"""

import argparse
import numpy as np
import torch

from aics.chemistry.local_energy import make_hubbard_context
from aics.chemistry.local_energy_signed import local_energy_hubbard_signed
from aics.chemistry.amplitude_sampling import state_int_to_bits
from aics.eval.energy import model_energy_exact_signed
from aics.models.ar_rnn import ARRNN
from aics.training.mle_pretrain import train_rnn_softtarget
from aics.training.vmc import VMCConfig, VMCState, vmc_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--U", type=float, default=4.0)
    p.add_argument("--d-hidden", type=int, default=32)
    p.add_argument("--mle-epochs", type=int, default=400)
    p.add_argument("--vmc-steps", type=int, default=2000)
    p.add_argument("--vmc-lr", type=float, default=3e-3)
    p.add_argument("--vmc-batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    L = args.L
    ctx = make_hubbard_context(L, t=1.0, U=args.U, pbc=True)
    print(f"L={L}, U={args.U}: sector dim = {len(ctx.states)}, E_0 = {ctx.E_0:.6f}",
          flush=True)

    model = ARRNN(L=L, n_up=L // 2, n_dn=L // 2,
                  d_hidden=args.d_hidden, learn_signs=True)

    # --- 1. Pretrain magnitudes on the exact target distribution. -----------
    print(f"\n[1/3] Magnitude pretrain via soft-target MLE ({args.mle_epochs} epochs)...",
          flush=True)
    states_bits = state_int_to_bits(ctx.states, L)
    p_target = (np.abs(ctx.psi_0) ** 2).astype(np.float64)
    train_rnn_softtarget(model, states_bits, p_target,
                         n_epochs=args.mle_epochs, batch_size=512, lr=2e-3,
                         seed=args.seed)

    # Magnitude quality check.
    with torch.no_grad():
        log_mag_all = model.log_psi_mag(
            torch.from_numpy(states_bits.astype(np.int64)).long()
        ).cpu().numpy()
    q_model = np.exp(log_mag_all) ** 2
    q_model /= q_model.sum()
    tv = 0.5 * np.sum(np.abs(q_model - p_target / p_target.sum()))
    print(f"     post-pretrain TV(|psi_model|^2, |psi_0|^2) = {tv:.4f}", flush=True)

    # Hard-sign check at random init.
    with torch.no_grad():
        s_init = model.soft_sign(
            torch.from_numpy(states_bits.astype(np.int64)).long()
        ).cpu().numpy()
    init_sign_match = (np.sign(s_init) == ctx.signs).mean()
    init_sign_match_alt = (np.sign(s_init) == -ctx.signs).mean()
    print(f"     pre-VMC sign-match rate (vs ED, both global conventions): "
          f"{init_sign_match:.2%} / {init_sign_match_alt:.2%}", flush=True)

    # --- 2. VMC to learn signs. ---------------------------------------------
    print(f"\n[2/3] VMC with signed local energy ({args.vmc_steps} steps)...",
          flush=True)
    cfg = VMCConfig(lr=args.vmc_lr, batch_size=args.vmc_batch, clip_norm=1.0)
    state = VMCState()

    E0 = ctx.E_0
    for step in range(1, args.vmc_steps + 1):
        vmc_step(model, ctx, state, cfg, local_energy_hubbard_signed, signed=True)
        if step % args.eval_every == 0 or step == 1 or step == args.vmc_steps:
            E = model_energy_exact_signed(model, ctx)
            rel = (E - E0) / abs(E0)
            with torch.no_grad():
                s = model.soft_sign(
                    torch.from_numpy(states_bits.astype(np.int64)).long()
                ).cpu().numpy()
            mr = max((np.sign(s) == ctx.signs).mean(),
                     (np.sign(s) == -ctx.signs).mean())
            sat = np.mean(np.abs(s) > 0.9)  # what fraction of states are saturated
            print(f"     step {step:>4d}: <H>={E:+.5f}  rel={rel:+.4f}  "
                  f"sign_match={mr:.2%}  saturated={sat:.2%}", flush=True)

    # --- 3. Final report. ---------------------------------------------------
    print(f"\n[3/3] Final state:")
    with torch.no_grad():
        s_final = model.soft_sign(
            torch.from_numpy(states_bits.astype(np.int64)).long()
        ).cpu().numpy()
        log_mag_final = model.log_psi_mag(
            torch.from_numpy(states_bits.astype(np.int64)).long()
        ).cpu().numpy()
    hard_signs = np.sign(s_final)
    match_pos = (hard_signs == ctx.signs).mean()
    match_neg = (hard_signs == -ctx.signs).mean()
    best_match = max(match_pos, match_neg)
    print(f"   Sign match rate: {best_match:.2%}  "
          f"(+convention {match_pos:.2%} / -convention {match_neg:.2%})")
    q_final = np.exp(log_mag_final) ** 2
    q_final /= q_final.sum()
    tv_final = 0.5 * np.sum(np.abs(q_final - p_target / p_target.sum()))
    print(f"   Magnitude TV: {tv_final:.4f} (was {tv:.4f} before VMC)")
    E_final = model_energy_exact_signed(model, ctx)
    print(f"   Final <H>={E_final:+.5f}  vs E_0={E0:+.5f}  "
          f"(rel error {(E_final-E0)/abs(E0):+.4f})")


if __name__ == "__main__":
    main()
