"""Half-filled 1D Fermi-Hubbard at fixed (N_up=L/2, N_dn=L/2).

State encoding: 2L-bit int. Bit i = up_i, bit L+i = dn_i (LSB-first).
"""

from itertools import combinations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def state_int_to_bits(ints, L):
    ints = np.asarray(ints, dtype=np.int64)
    out = np.zeros((ints.size, 2 * L), dtype=np.int64)
    for i in range(2 * L):
        out[:, i] = (ints >> i) & 1
    return out


def bits_to_state_int(bits, L):
    return (np.asarray(bits, dtype=np.int64) * (1 << np.arange(2 * L, dtype=np.int64))).sum(axis=1)


def _sector_states(L):
    L2 = L // 2
    out = []
    for ups in combinations(range(L), L2):
        u = sum(1 << i for i in ups)
        for dns in combinations(range(L), L2):
            d = sum(1 << i for i in dns)
            out.append(u | (d << L))
    return np.asarray(sorted(out), dtype=np.int64)


def _bonds(L, pbc):
    bs = [(i, i + 1) for i in range(L - 1)]
    if pbc and L > 2:
        bs.append((L - 1, 0))
    return bs


def _hops(state, L, bonds):
    Lm = (1 << L) - 1
    up = state & Lm
    dn = (state >> L) & Lm
    for i, j in bonds:
        lo, hi = min(i, j), max(i, j)
        mid = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
        bi, bj = 1 << i, 1 << j
        for is_up, occ, other in ((True, up, dn), (False, dn, up)):
            jw_sign = 1 if bin(occ & mid).count("1") % 2 == 0 else -1
            for src, tgt in ((bi, bj), (bj, bi)):
                if (occ & src) and not (occ & tgt):
                    new_occ = (occ ^ src) | tgt
                    new_state = (new_occ | (other << L)) if is_up else ((new_occ << L) | other)
                    yield new_state, jw_sign


class Hubbard:
    """Half-filled 1D Hubbard context: H, ED ground state, signs, samples."""

    def __init__(self, L, U=4.0, t=1.0, pbc=True):
        assert L % 2 == 0 or L >= 3, "L must be at least 3"
        self.L, self.U, self.t = L, U, t
        self.states = _sector_states(L)
        self.idx = {int(s): k for k, s in enumerate(self.states)}
        self.bonds = _bonds(L, pbc)
        D = len(self.states)
        Lm = (1 << L) - 1
        rows, cols, data = [], [], []
        for k, s in enumerate(self.states):
            si = int(s)
            up = si & Lm
            dn = (si >> L) & Lm
            rows.append(k); cols.append(k)
            data.append(U * bin(up & dn).count("1"))
            for new_s, jw in _hops(si, L, self.bonds):
                rows.append(self.idx[int(new_s)]); cols.append(k); data.append(-t * jw)
        self.H = sp.csr_matrix((data, (rows, cols)), shape=(D, D))
        eigvals, eigvecs = spla.eigsh(self.H, k=1, which="SA")
        psi = eigvecs[:, 0].astype(np.float64)
        if psi[np.argmax(np.abs(psi))] < 0:
            psi = -psi
        self.psi_0 = psi
        self.E_0 = float(eigvals[0])
        self.signs = np.where(np.abs(psi) < 1e-12, 1, np.sign(psi)).astype(np.int64)

    def sample(self, n, rng):
        p = (self.psi_0 ** 2)
        idx = rng.choice(len(p), size=n, p=p / p.sum())
        return state_int_to_bits(self.states[idx], self.L)
