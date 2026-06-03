"""Local energy for 1D Heisenberg AFM at S^z = 0.

E_loc(x) = sum_{x'} <x|H|x'> psi(x') / psi(x)
        = E_diag(x) + sum over NN bonds with antiparallel spins of
            (J/2) * (sign(x_flipped)/sign(x)) * exp(log|psi(x_flipped)| - log|psi(x)|)

Matrix elements have NO Jordan-Wigner string factors (we work directly in
the spin basis, not fermionic). Signs come from the Marshall sign rule, which
is exact for the Heisenberg singlet GS on a bipartite lattice.
"""

import numpy as np
import torch

from aics.spin.heisenberg import bits_to_state_int_spin, state_int_to_bits_spin


def _popcount_array(arr):
    v = np.asarray(arr, dtype=np.uint64).copy()
    out = np.zeros(v.shape, dtype=np.int64)
    while v.any():
        out += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return out


def local_energy_heisenberg(model, x_bits, ctx, device="cpu"):
    """Returns (B,) detached float64 torch tensor."""
    L, J, bonds = ctx.L, ctx.J, ctx.bonds

    if torch.is_tensor(x_bits):
        x_bits_np = x_bits.detach().cpu().numpy().astype(np.int64)
    else:
        x_bits_np = np.asarray(x_bits, dtype=np.int64)
    B = x_bits_np.shape[0]
    x_ints = bits_to_state_int_spin(x_bits_np, L)

    # Diagonal: J/4 * sum over bonds, +1 if both same, -1 if different.
    E_diag = np.zeros(B, dtype=np.float64)
    chunks_idx, chunks_xp, chunks_coeff = [], [], []
    for (a, b) in bonds:
        xa = (x_ints >> a) & 1
        xb = (x_ints >> b) & 1
        same = (xa == xb)
        E_diag += (J / 4) * np.where(same, 1.0, -1.0)
        # Off-diagonal: flip both bits on differing pairs.
        diff_idx = np.where(~same)[0]
        if len(diff_idx) == 0:
            continue
        new_x = x_ints[diff_idx] ^ ((1 << a) | (1 << b))
        chunks_idx.append(diff_idx)
        chunks_xp.append(new_x)
        chunks_coeff.append(np.full(len(diff_idx), J / 2, dtype=np.float64))

    if not chunks_idx:
        return torch.from_numpy(E_diag).to(device)

    sample_idx_arr = np.concatenate(chunks_idx).astype(np.int64)
    xp_int_arr = np.concatenate(chunks_xp).astype(np.int64)
    coeff_arr = np.concatenate(chunks_coeff).astype(np.float64)

    # Union of x and x', batched forward.
    all_ints = np.concatenate([x_ints, xp_int_arr])
    unique_ints, inverse = np.unique(all_ints, return_inverse=True)
    unique_bits = state_int_to_bits_spin(unique_ints, L)
    bits_tensor = torch.from_numpy(unique_bits.astype(np.int64)).long().to(device)

    with torch.no_grad():
        log_mag = model.log_psi_mag(bits_tensor).cpu().numpy()

    unique_signs = np.array(
        [ctx.signs[ctx.state_to_idx[int(s)]] for s in unique_ints],
        dtype=np.float64,
    )

    x_inv = inverse[:B]
    xp_inv = inverse[B:]
    log_x_per_hop = log_mag[x_inv[sample_idx_arr]]
    log_xp_per_hop = log_mag[xp_inv]
    sx_per_hop = unique_signs[x_inv[sample_idx_arr]]
    sxp_per_hop = unique_signs[xp_inv]
    ratio = (sxp_per_hop / sx_per_hop) * np.exp(log_xp_per_hop - log_x_per_hop)

    E_off = np.zeros(B, dtype=np.float64)
    np.add.at(E_off, sample_idx_arr, coeff_arr * ratio)

    return torch.from_numpy(E_diag + E_off).to(device)
