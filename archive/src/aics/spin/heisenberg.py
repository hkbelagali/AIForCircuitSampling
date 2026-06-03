"""1D Heisenberg AFM at S^z=0 sector: ED setup, sampling, Marshall sign rule.

Basis convention: L-bit integers with exactly L/2 ones, where bit i = 1 means
spin-up at site i, bit i = 0 means spin-down. Sublattice A = even sites
{0, 2, ..., L-2}; B = odd sites {1, 3, ..., L-1}.

H = J sum_{<ij>} S_i . S_j  with J > 0 (antiferromagnetic), PBC for L > 2.

Sign rule: Marshall's 1955 theorem -- exact for the Heisenberg AFM singlet
GS on bipartite lattices --
        sign(|x>) = (-1)^{N_down on A(x)}.
This is closed-form and does NOT require ED to compute.
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class HeisContext:
    L: int
    J: float
    pbc: bool
    n_up: int
    bonds: list
    states: np.ndarray           # (D,) sorted state ints (L-bit, popcount = n_up)
    state_to_idx: Dict[int, int]
    signs: np.ndarray            # (D,) +-1 from Marshall (closed-form, exact)
    H: Any                       # scipy.sparse.csr_matrix (D, D)
    E_0: float
    psi_0: np.ndarray            # (D,) exact GS (used for sampling and threshold check)


def _popcount(v):
    v = int(v); c = 0
    while v:
        c += v & 1
        v >>= 1
    return c


def sector_states_spin(L, n_up):
    """All L-bit integers with popcount == n_up, sorted ascending."""
    out = [x for x in range(1 << L) if _popcount(x) == n_up]
    return np.asarray(out, dtype=np.int64)


def _A_mask(L):
    """Bitmask of even sites 0, 2, ..., L-2."""
    m = 0
    for i in range(0, L, 2):
        m |= 1 << i
    return m


def marshall_signs_spin(state_ints, L):
    """Marshall sign rule (-1)^{N_down on A} for L-bit spin states.
    Down spin at site i means bit i = 0; A = even sites."""
    state_ints = np.asarray(state_ints, dtype=np.int64)
    L_mask = (1 << L) - 1
    A = _A_mask(L)
    # downs on A: bits in A that are 0 in state.
    downs_on_A = (~state_ints) & L_mask & A
    v = downs_on_A.astype(np.uint64).copy()
    pc = np.zeros_like(v, dtype=np.int64)
    while v.any():
        pc += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return np.where(pc & 1, -1, 1).astype(np.int64)


def make_heisenberg_context(L, J=1.0, pbc=True):
    """Build a HeisContext: ED at the S^z = 0 sector."""
    assert L % 2 == 0, "Heisenberg AFM bipartite half-filled needs even L"
    n_up = L // 2
    states = sector_states_spin(L, n_up)
    D = len(states)
    state_to_idx = {int(s): i for i, s in enumerate(states)}

    bonds = [(i, i + 1) for i in range(L - 1)]
    if pbc and L > 2:
        bonds.append((L - 1, 0))

    rows, cols, data = [], [], []
    for i, s in enumerate(states):
        s = int(s)
        diag = 0.0
        for (a, b) in bonds:
            xa = (s >> a) & 1
            xb = (s >> b) & 1
            if xa == xb:
                diag += J / 4
            else:
                diag -= J / 4
                # S^+_a S^-_b + h.c. flips both spins on the bond
                s_new = s ^ ((1 << a) | (1 << b))
                j = state_to_idx[int(s_new)]
                rows.append(i); cols.append(j); data.append(J / 2)
        rows.append(i); cols.append(i); data.append(diag)

    H = sp.csr_matrix((data, (rows, cols)), shape=(D, D)).tocsr()

    if D <= 4:
        # Dense fallback for tiny systems
        eigvals, eigvecs = np.linalg.eigh(H.toarray())
        E_0 = float(eigvals[0])
        psi_0 = eigvecs[:, 0].astype(np.float64)
    else:
        eigvals, eigvecs = spla.eigsh(H, k=1, which="SA")
        E_0 = float(eigvals[0])
        psi_0 = eigvecs[:, 0].astype(np.float64)

    signs = marshall_signs_spin(states, L)

    return HeisContext(
        L=L, J=J, pbc=pbc, n_up=n_up, bonds=bonds,
        states=states, state_to_idx=state_to_idx, signs=signs,
        H=H, E_0=E_0, psi_0=psi_0,
    )


def sample_from_amplitudes_spin(psi, states, L, k, rng):
    """Draw k L-bit spin samples from |psi|^2."""
    p = np.abs(psi) ** 2
    p = p / p.sum()
    indices = rng.choice(len(p), size=k, p=p)
    state_ints = np.asarray(states[indices], dtype=np.int64)
    bits = np.zeros((k, L), dtype=np.int64)
    for i in range(L):
        bits[:, i] = (state_ints >> i) & 1
    return bits, state_ints, indices


def bits_to_state_int_spin(bits, L):
    """(k, L) bit array -> (k,) ints, LSB-first."""
    bits = np.asarray(bits, dtype=np.int64)
    powers = (1 << np.arange(L, dtype=np.int64))
    return (bits * powers).sum(axis=1)


def state_int_to_bits_spin(state_ints, L):
    """(k,) ints -> (k, L) bit array, LSB-first."""
    state_ints = np.asarray(state_ints, dtype=np.int64)
    out = np.zeros((state_ints.size, L), dtype=np.int64)
    for i in range(L):
        out[:, i] = (state_ints >> i) & 1
    return out
