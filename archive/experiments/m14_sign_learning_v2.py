"""Hubbard sign-learning v2: try several improvements over m12 and report
side-by-side at L=4.

Improvements tested:
  1. Deeper sign head (2-layer MLP instead of single linear).
  2. Marshall-rule supervised warm-start of the sign head.
  3. Freeze magnitude head during sign-learning VMC.

Each run:
  - Samples k bitstrings, MLE-pretrain magnitude head.
  - (optional) Marshall warm-start of sign head.
  - VMC with signed local energy.
  - Report steps-to-threshold, sign match, magnitude TV.
"""

import argparse
import time

import numpy as np
import torch

from aics.chemistry.amplitude_sampling import (
    sample_from_amplitudes, state_int_to_bits,
)
from aics.chemistry.local_energy import make_hubbard_context
from aics.chemistry.local_energy_signed import local_energy_hubbard_signed
from aics.chemistry.marshall import marshall_signs_batch
from aics.eval.energy import model_energy_exact_signed
from aics.models.ar_rnn import ARRNN, _HandAdam
from aics.training.mle_pretrain import train_rnn_mle
from aics.training.vmc import VMCConfig, VMCState, vmc_step


def marshall_warmstart_sign_head(model, ctx, n_epochs=200, lr=1e-2,
                                 batch_size=256, device="cpu"):
    """Supervise the sign head to match the analytic Marshall formula
    sign(x) = (-1)^{N_dn^A} via MSE on tanh(logit). Closed-form target, no ED."""
    states = ctx.states
    L = ctx.L
    bits_all = state_int_to_bits(states, L)
    target = marshall_signs_batch(states, L).astype(np.float64)
    target_t = torch.from_numpy(target).float().to(device)
    bits_t = torch.from_numpy(bits_all.astype(np.int64)).long().to(device)

    sign_params = list(model.sign_params())
    opt = _HandAdam(sign_params, lr=lr)
    model.train()
    N = len(states)
    rng = np.random.default_rng(0)
    for _ in range(n_epochs):
        perm = rng.permutation(N)
        for s in range(0, N, batch_size):
            idx = torch.from_numpy(perm[s:s + batch_size]).long()
            pred = torch.tanh(model.sign_logit(bits_t[idx]))
            loss = ((pred - target_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


def freeze_params(params):
    for p in params:
        p.requires_grad_(False)


def unfreeze_params(params):
    for p in params:
        p.requires_grad_(True)


def evaluate(model, ctx, states_bits, p_target):
    with torch.no_grad():
        bits_t = torch.from_numpy(states_bits.astype(np.int64)).long()
        log_mag = model.log_psi_mag(bits_t).cpu().numpy()
        s = model.soft_sign(bits_t).cpu().numpy()
    q = np.exp(log_mag) ** 2; q /= q.sum()
    tv = 0.5 * np.sum(np.abs(q - p_target))
    sm = max((np.sign(s) == ctx.signs).mean(),
             (np.sign(s) == -ctx.signs).mean())
    sat = np.mean(np.abs(s) > 0.9)
    return tv, sm, sat


def run_config(name, L, k, seed, mle_epochs, vmc_steps, eval_every,
               marshall_warmstart=False, freeze_mag=False, vmc_lr=3e-3):
    print(f"\n=== {name} (L={L}, k={k}, seed={seed}, MW={marshall_warmstart}, "
          f"FZ={freeze_mag}) ===", flush=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    ctx = make_hubbard_context(L, 1.0, 4.0, pbc=True)
    model = ARRNN(L=L, n_up=L // 2, n_dn=L // 2, d_hidden=32, learn_signs=True)
    states_bits = state_int_to_bits(ctx.states, L)
    p_target = (np.abs(ctx.psi_0) ** 2); p_target /= p_target.sum()

    # 1. Sample-based MLE on magnitudes.
    t0 = time.time()
    bits, _, _ = sample_from_amplitudes(ctx.psi_0, ctx.states, L, k, rng)
    train_rnn_mle(model, bits, n_epochs=mle_epochs, lr=2e-3, batch_size=32)

    # 2. Optional Marshall warm-start of the sign head.
    if marshall_warmstart:
        marshall_warmstart_sign_head(model, ctx, n_epochs=200, lr=1e-2)

    tv0, sm0, sat0 = evaluate(model, ctx, states_bits, p_target)
    print(f"  [post-pretrain] TV={tv0:.3f}  sign_match={sm0:.2%}  sat={sat0:.2%}",
          flush=True)

    # 3. Optional freeze of magnitude params.
    mag_params = list(model.magnitude_params())
    if freeze_mag:
        freeze_params(mag_params)

    # 4. VMC.
    cfg = VMCConfig(lr=vmc_lr, batch_size=256, clip_norm=1.0)
    state = VMCState()
    steps_to_threshold = None
    for step in range(1, vmc_steps + 1):
        vmc_step(model, ctx, state, cfg, local_energy_hubbard_signed, signed=True)
        if step % eval_every == 0 or step == vmc_steps:
            E = model_energy_exact_signed(model, ctx)
            rel = (E - ctx.E_0) / abs(ctx.E_0)
            tv, sm, sat = evaluate(model, ctx, states_bits, p_target)
            if step % (eval_every * 5) == 0 or step == vmc_steps:
                print(f"  step {step:>4d}: rel={rel:+.4f}  "
                      f"sign_match={sm:.2%}  sat={sat:.2%}  TV={tv:.3f}",
                      flush=True)
            if steps_to_threshold is None and abs(rel) <= 0.01:
                steps_to_threshold = step
                # Don't break -- want to also report final state.

    if freeze_mag:
        unfreeze_params(mag_params)
    elapsed = time.time() - t0
    tv_f, sm_f, sat_f = evaluate(model, ctx, states_bits, p_target)
    E_f = model_energy_exact_signed(model, ctx)
    print(f"  [final] steps={steps_to_threshold}  rel={(E_f-ctx.E_0)/abs(ctx.E_0):+.4f}  "
          f"sign_match={sm_f:.2%}  sat={sat_f:.2%}  TV={tv_f:.3f}  "
          f"({elapsed:.0f}s)", flush=True)
    return {
        "name": name, "L": L, "k": k, "seed": seed,
        "marshall_warmstart": marshall_warmstart,
        "freeze_mag": freeze_mag,
        "steps_to_threshold": steps_to_threshold,
        "final_rel_err": (E_f - ctx.E_0) / abs(ctx.E_0),
        "sign_match": sm_f, "magnitude_TV": tv_f,
        "elapsed_sec": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mle-epochs", type=int, default=100)
    p.add_argument("--vmc-steps", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=20)
    args = p.parse_args()

    results = []
    configs = [
        ("Baseline (no warmstart, joint)", False, False),
        ("Marshall warmstart, joint",       True,  False),
        ("Marshall warmstart, freeze mag",  True,  True),
        ("No warmstart, freeze mag",        False, True),
    ]
    for name, mw, fm in configs:
        r = run_config(name, args.L, args.k, args.seed,
                       args.mle_epochs, args.vmc_steps, args.eval_every,
                       marshall_warmstart=mw, freeze_mag=fm)
        results.append(r)

    print("\n\n=== summary ===")
    print(f"  {'config':<36s}  {'steps':>6s}  {'rel_err':>8s}  "
          f"{'sm':>6s}  {'TV':>5s}")
    for r in results:
        st = r["steps_to_threshold"]
        st_s = f"{st}" if st is not None else "--"
        print(f"  {r['name']:<36s}  {st_s:>6s}  {r['final_rel_err']:>+8.4f}  "
              f"{r['sign_match']:>6.1%}  {r['magnitude_TV']:>5.3f}")


if __name__ == "__main__":
    main()
