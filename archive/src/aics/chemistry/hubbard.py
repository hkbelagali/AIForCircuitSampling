"""1D Hubbard model in a fixed (N_up, N_dn) sector, sparse JW representation.

    H = -t sum_{<ij>, sigma} (c^dag_{i,sigma} c_{j,sigma} + h.c.)
        + U sum_i n_{i,up} n_{i,dn}

Jordan-Wigner ordering: (up_0, ..., up_{L-1}, dn_0, ..., dn_{L-1}). With the
state-integer convention in aics.common.symmetry, the up bits sit below the dn
bits, so within-spin hoppings touch only same-spin modes between the endpoints
when computing JW signs (cross-spin Z strings square to identity).
"""

import numpy as np
import scipy.sparse as sp

from aics.common.symmetry import sector_states


def _popcount(x):
    return bin(x).count("1")


def _jw_sign(spin_occ, i, j):
    """Sign of an intra-spin hop between sites i and j given that spin's
    occupation pattern. The sign is (-1)^(occupied modes strictly between i and j).
    """
    if i > j:
        i, j = j, i
    mask = ((1 << j) - 1) ^ ((1 << (i + 1)) - 1)
    return -1 if _popcount(spin_occ & mask) % 2 else 1


def build_hubbard_1d(L, t, U, n_up, n_dn, pbc=True):
    """Sparse Hubbard Hamiltonian in the (n_up, n_dn) sector of a 1D chain.

    Returns a CSR matrix of shape (D, D), D = C(L, n_up) * C(L, n_dn), indexed
    by the canonical sorted-state ordering from sector_states.
    """
    states = sector_states(L, n_up, n_dn)
    D = states.size
    state_to_idx = {int(s): i for i, s in enumerate(states)}

    bonds = [(i, i + 1) for i in range(L - 1)]
    if pbc and L > 2:
        bonds.append((0, L - 1))

    rows, cols, vals = [], [], []
    L_mask = (1 << L) - 1

    for idx in range(D):
        x = int(states[idx])
        up = x & L_mask
        dn = x >> L

        d_occ = _popcount(up & dn)
        if d_occ:
            rows.append(idx); cols.append(idx); vals.append(U * d_occ)

        for spin_is_up, spin_occ in ((True, up), (False, dn)):
            for i, j in bonds:
                bi, bj = 1 << i, 1 << j
                # c^dag_j c_i: hop i -> j (need n_i = 1, n_j = 0)
                if (spin_occ & bi) and not (spin_occ & bj):
                    new_occ = (spin_occ ^ bi) | bj
                    sign = _jw_sign(spin_occ, i, j)
                    new_x = (
                        ((dn << L) | new_occ) if spin_is_up
                        else ((new_occ << L) | up)
                    )
                    rows.append(state_to_idx[new_x]); cols.append(idx); vals.append(-t * sign)
                # c^dag_i c_j: hop j -> i (need n_j = 1, n_i = 0)
                if (spin_occ & bj) and not (spin_occ & bi):
                    new_occ = (spin_occ ^ bj) | bi
                    sign = _jw_sign(spin_occ, i, j)
                    new_x = (
                        ((dn << L) | new_occ) if spin_is_up
                        else ((new_occ << L) | up)
                    )
                    rows.append(state_to_idx[new_x]); cols.append(idx); vals.append(-t * sign)

    return sp.csr_matrix((vals, (rows, cols)), shape=(D, D))
