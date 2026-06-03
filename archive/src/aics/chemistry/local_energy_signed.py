"""Variant of local_energy_hubbard that uses the model's predicted sign head
(via tanh of a learned logit) in place of the ED-derived ctx.signs lookup.

Same enumeration of NN hops; only the per-hop sign factor differs. Use this
when the AR-RNN was constructed with learn_signs=True.
"""

import numpy as np
import torch

from aics.chemistry.amplitude_sampling import bits_to_state_int, state_int_to_bits
from aics.chemistry.local_energy import _popcount_array


def local_energy_hubbard_signed(model, x_bits, ctx, device="cpu", sign_eps=1e-6):
    """E_loc(x) for Hubbard, with signs read from the model's sign head.

    Returns (B,) float64 detached torch tensor.

    sign_eps: stability floor; if |soft_sign(x)| < sign_eps the denominator
    gets clamped to avoid division by zero. The sign head should saturate away
    from 0 once training has begun, so this guard is rarely active in practice.
    """
    L, t, U, pbc, bonds = ctx.L, ctx.t, ctx.U, ctx.pbc, ctx.bonds
    L_mask = (1 << L) - 1

    if torch.is_tensor(x_bits):
        x_bits_np = x_bits.detach().cpu().numpy().astype(np.int64)
    else:
        x_bits_np = np.asarray(x_bits, dtype=np.int64)
    B = x_bits_np.shape[0]
    x_ints = bits_to_state_int(x_bits_np, L)
    up = x_ints & L_mask
    dn = (x_ints >> L) & L_mask

    double_occ = _popcount_array(up & dn)
    E_diag = U * double_occ.astype(np.float64)

    chunks_idx, chunks_xp, chunks_coeff = [], [], []
    for (i, j) in bonds:
        bi, bj = 1 << i, 1 << j
        i_lo, j_hi = (i, j) if i < j else (j, i)
        jw_mask = ((1 << j_hi) - 1) ^ ((1 << (i_lo + 1)) - 1)
        for spin_is_up in (True, False):
            occ = up if spin_is_up else dn
            other = dn if spin_is_up else up
            jw_parity = _popcount_array(occ & jw_mask) & 1
            jw_sign = np.where(jw_parity, -1, 1).astype(np.float64)

            for source_b, target_b in ((bi, bj), (bj, bi)):
                mask_hop = ((occ & source_b) != 0) & ((occ & target_b) == 0)
                if not mask_hop.any():
                    continue
                idx = np.where(mask_hop)[0]
                occ_h = occ[idx]
                new_occ = (occ_h ^ source_b) | target_b
                if spin_is_up:
                    new_x = (other[idx] << L) | new_occ
                else:
                    new_x = (new_occ << L) | other[idx]
                chunks_idx.append(idx)
                chunks_xp.append(new_x)
                chunks_coeff.append((-t) * jw_sign[idx])

    if not chunks_idx:
        return torch.from_numpy(E_diag).to(device)

    sample_idx_arr = np.concatenate(chunks_idx).astype(np.int64)
    xp_int_arr = np.concatenate(chunks_xp).astype(np.int64)
    coeff_arr = np.concatenate(chunks_coeff).astype(np.float64)

    all_ints = np.concatenate([x_ints, xp_int_arr])
    unique_ints, inverse = np.unique(all_ints, return_inverse=True)
    unique_bits = state_int_to_bits(unique_ints, L)
    bits_tensor = torch.from_numpy(unique_bits.astype(np.int64)).long().to(device)

    with torch.no_grad():
        log_mag = model.log_psi_mag(bits_tensor).cpu().numpy()
        soft_signs = model.soft_sign(bits_tensor).cpu().numpy()

    x_inv = inverse[:B]
    xp_inv = inverse[B:]
    log_x_per_hop = log_mag[x_inv[sample_idx_arr]]
    log_xp_per_hop = log_mag[xp_inv]
    s_x_per_hop = soft_signs[x_inv[sample_idx_arr]]
    s_xp_per_hop = soft_signs[xp_inv]

    # Clamp |s_x| away from zero to avoid blow-up.
    s_x_safe = np.where(np.abs(s_x_per_hop) < sign_eps,
                        np.sign(s_x_per_hop + 1e-30) * sign_eps,
                        s_x_per_hop)
    ratio = (s_xp_per_hop / s_x_safe) * np.exp(log_xp_per_hop - log_x_per_hop)

    E_off = np.zeros(B, dtype=np.float64)
    np.add.at(E_off, sample_idx_arr, coeff_arr * ratio)

    return torch.from_numpy(E_diag + E_off).to(device)
