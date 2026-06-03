"""Training: MLE pretrain, local energy, VMC step."""

import numpy as np
import torch

from .hubbard import bits_to_state_int, state_int_to_bits


def train_mle(model, X_bits, epochs=100, lr=2e-3, batch=32):
    X = torch.as_tensor(X_bits, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), batch):
            loss = -model.log_prob(X[perm[s:s + batch]]).mean()
            opt.zero_grad(); loss.backward(); opt.step()


def local_energy(model, x_bits, ctx):
    L, U, t = ctx.L, ctx.U, ctx.t
    Lm = (1 << L) - 1
    xn = x_bits.detach().cpu().numpy() if torch.is_tensor(x_bits) else x_bits
    xn = np.asarray(xn, dtype=np.int64)
    x_int = bits_to_state_int(xn, L)
    up = x_int & Lm
    dn = (x_int >> L) & Lm

    E_diag = U * np.array([bin(int(u & d)).count("1") for u, d in zip(up, dn)],
                          dtype=np.float64)
    chunks_i, chunks_xp, chunks_co = [], [], []
    for i, j in ctx.bonds:
        bi, bj = 1 << i, 1 << j
        lo, hi = min(i, j), max(i, j)
        jw_mask = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
        for is_up, occ, other in ((True, up, dn), (False, dn, up)):
            jw = np.array([bin(int(o & jw_mask)).count("1") & 1 for o in occ])
            jw_sign = np.where(jw, -1.0, 1.0)
            for src, tgt in ((bi, bj), (bj, bi)):
                m = ((occ & src) != 0) & ((occ & tgt) == 0)
                if not m.any():
                    continue
                idx = np.where(m)[0]
                new_occ = (occ[idx] ^ src) | tgt
                new_x = (new_occ | (other[idx] << L)) if is_up else (
                    (new_occ << L) | other[idx])
                chunks_i.append(idx)
                chunks_xp.append(new_x)
                chunks_co.append(-t * jw_sign[idx])

    if not chunks_i:
        return torch.from_numpy(E_diag).double()

    si = np.concatenate(chunks_i).astype(np.int64)
    xp = np.concatenate(chunks_xp).astype(np.int64)
    co = np.concatenate(chunks_co)
    union = np.concatenate([x_int, xp])
    uniq, inv = np.unique(union, return_inverse=True)
    with torch.no_grad():
        log_mag = model.log_psi(
            torch.from_numpy(state_int_to_bits(uniq, L).astype(np.int64)).long()
        ).cpu().numpy()
    signs = np.array([ctx.signs[ctx.idx[int(s)]] for s in uniq], dtype=np.float64)
    x_inv, xp_inv = inv[:len(x_int)], inv[len(x_int):]
    ratio = (signs[xp_inv] / signs[x_inv[si]]) * np.exp(
        log_mag[xp_inv] - log_mag[x_inv[si]])
    E_off = np.zeros(len(x_int), dtype=np.float64)
    np.add.at(E_off, si, co * ratio)
    return torch.from_numpy(E_diag + E_off).double()


def vmc_step(model, ctx, opt, batch=256):
    x = model.sample(batch)
    E_vec = local_energy(model, x, ctx).float()
    E_mean = float(E_vec.mean())
    log_psi = model.log_psi(x)
    loss = (log_psi * (E_vec - E_mean).detach()).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return E_mean


def model_energy_exact(model, ctx):
    bits = state_int_to_bits(ctx.states, ctx.L)
    with torch.no_grad():
        log_mag = model.log_psi(torch.from_numpy(bits.astype(np.int64)).long()).cpu().numpy()
    psi = ctx.signs * np.exp(log_mag)
    psi = psi / np.linalg.norm(psi)
    return float(psi @ (ctx.H @ psi))
