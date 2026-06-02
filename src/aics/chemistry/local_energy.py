"""Local energy for the 1D Hubbard model in a fixed (N_up, N_dn) sector.

    E_loc(x) = sum_{x'} <x|H|x'> psi(x') / psi(x)

For each sample x in a batch, the connected states x' (and their <x|H|x'>
matrix elements with JW sign) are enumerated analytically by iterating over
NN bonds and both spin channels. The model is queried with a single batched
forward pass over the UNION of all unique x and x' across the batch, then the
amplitude ratios psi(x')/psi(x) = (sign(x')/sign(x)) * exp(log|psi(x')| - log|psi(x)|)
are assembled.

Sign structure is supplied externally via a `HubbardContext` (built once via
`make_hubbard_context`, which uses ED to get the exact GS sign lookup).
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from aics.chemistry.amplitude_sampling import bits_to_state_int, state_int_to_bits
from aics.chemistry.hubbard_setup import hubbard_gs_setup
from aics.chemistry.marshall import signs_from_psi
from aics.common.symmetry import sector_states


@dataclass
class HubbardContext:
    L: int
    t: float
    U: float
    pbc: bool
    n_up: int
    n_dn: int
    bonds: list           # list of (i, j) NN pairs
    states: np.ndarray    # (D,) sorted sector state ints
    state_to_idx: Dict[int, int]
    signs: np.ndarray     # (D,) int64 of +-1 (ED-based exact)
    H: Any                # scipy.sparse.csr_matrix (D, D) in the sector
    E_0: float
    psi_0: np.ndarray     # exact GS (for reference / verification only)


def make_hubbard_context(L, t, U, pbc=True):
    """One-time setup: ED + signs + sector index. Use for L<=8."""
    setup = hubbard_gs_setup(L, t, U, pbc=pbc)
    states = sector_states(L, L // 2, L // 2)
    state_to_idx = {int(s): i for i, s in enumerate(states)}
    signs = signs_from_psi(setup["psi_0"])
    bonds = [(i, i + 1) for i in range(L - 1)]
    if pbc and L > 2:
        bonds.append((0, L - 1))
    return HubbardContext(
        L=L, t=t, U=U, pbc=pbc, n_up=L // 2, n_dn=L // 2,
        bonds=bonds, states=states, state_to_idx=state_to_idx, signs=signs,
        H=setup["H"], E_0=setup["E_0"], psi_0=setup["psi_0"],
    )


def _popcount_array(arr):
    v = np.asarray(arr, dtype=np.uint64).copy()
    out = np.zeros(v.shape, dtype=np.int64)
    while v.any():
        out += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return out


def local_energy_hubbard(model, x_bits, ctx, device="cpu", log_psi_mag_override=None):
    """E_loc(x_i) over a sample batch.

    Returns a (B,) float64 torch tensor (detached).

    `log_psi_mag_override`, if given, must be a callable mapping (numpy state
    ints, L) -> numpy log|psi| array, used in place of the model. This is for
    verification (substitute the exact psi_0 lookup).
    """
    L, t, U, pbc, bonds = ctx.L, ctx.t, ctx.U, ctx.pbc, ctx.bonds
    L_mask = (1 << L) - 1

    if torch.is_tensor(x_bits):
        x_bits_np = x_bits.detach().cpu().numpy().astype(np.int64)
    else:
        x_bits_np = np.asarray(x_bits, dtype=np.int64)
    B = x_bits_np.shape[0]
    x_ints = bits_to_state_int(x_bits_np, L)            # (B,)
    up = x_ints & L_mask
    dn = (x_ints >> L) & L_mask

    # Diagonal: U * (double occupations)
    double_occ = _popcount_array(up & dn)
    E_diag = U * double_occ.astype(np.float64)

    # Off-diagonal enumeration -- vectorized over the batch for each
    # (bond, spin, direction) triple. For B samples * n_bonds bonds *
    # 2 spins * 2 directions this is now ~4*n_bonds numpy passes on (B,)
    # int arrays instead of a Python triple-nested loop.
    chunks_idx, chunks_xp, chunks_coeff = [], [], []
    for (i, j) in bonds:
        bi, bj = 1 << i, 1 << j
        i_lo, j_hi = (i, j) if i < j else (j, i)
        # bits strictly between i and j (exclusive endpoints) -> JW string
        jw_mask = ((1 << j_hi) - 1) ^ ((1 << (i_lo + 1)) - 1)
        for spin_is_up in (True, False):
            occ = up if spin_is_up else dn
            other = dn if spin_is_up else up
            jw_parity = _popcount_array(occ & jw_mask) & 1
            jw_sign = np.where(jw_parity, -1, 1).astype(np.float64)

            for source_b, target_b in ((bi, bj), (bj, bi)):
                # hop from source bit to target bit: n_source=1, n_target=0
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

    # Batched log|psi| over union of x and x'.
    all_ints = np.concatenate([x_ints, xp_int_arr])
    unique_ints, inverse = np.unique(all_ints, return_inverse=True)
    if log_psi_mag_override is not None:
        log_mag = log_psi_mag_override(unique_ints, L)
    else:
        unique_bits = state_int_to_bits(unique_ints, L)
        with torch.no_grad():
            log_mag = model.log_psi_mag(
                torch.from_numpy(unique_bits.astype(np.int64)).long().to(device)
            ).cpu().numpy()

    # Signs at each unique state (via state_to_idx lookup into ctx.signs).
    unique_signs = np.array(
        [ctx.signs[ctx.state_to_idx[int(s)]] for s in unique_ints],
        dtype=np.float64,
    )

    # Per-hop ratios psi(x')/psi(x).
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
