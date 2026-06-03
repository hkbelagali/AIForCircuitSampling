"""Sign-structure utilities for the ground state of the half-filled Hubbard model.

Two routes:

1. `signs_from_psi(psi_0)` -- EXACT, ED-based lookup. Given the exact GS vector
   (e.g. from `hubbard_gs_setup`), returns an int array of signs in {-1, +1}
   indexed by sector position. Use this for any system we can diagonalize.

2. `marshall_signs_batch(state_ints, L)` -- the analytic Marshall sign rule
   `sign(x) = (-1)^(N_down on sublattice A)`. This is exact for the Heisenberg
   AFM (large-U limit) but does NOT match the exact Hubbard GS signs in the
   LSB-first computational basis at finite U / arbitrary basis rotations. Kept
   for diagnostic / large-L fallback use; flag and verify before using it.

State convention (from aics.common.symmetry): bit i (i < L) = up_i, bit L+i =
dn_i, LSB-first.
"""

import numpy as np


def signs_from_psi(psi_0, tol=1e-12):
    """Exact sign lookup from an ED ground state. Returns int64 array of signs.

    Components with |psi| < tol are assigned +1 (their contribution is zero
    regardless, so the sign choice is immaterial). A global sign of psi_0 is
    absorbed: we fix psi_0(argmax|psi_0|) > 0 by convention.
    """
    psi = np.asarray(psi_0)
    ref = int(np.argmax(np.abs(psi)))
    flip = -1 if psi[ref] < 0 else 1
    psi = psi * flip
    signs = np.where(np.abs(psi) < tol, 1, np.sign(psi)).astype(np.int64)
    return signs


def _A_mask(L):
    """Bitmask selecting even sites 0, 2, ..., L-2."""
    m = 0
    for i in range(0, L, 2):
        m |= 1 << i
    return m


def marshall_signs_batch(state_ints, L):
    """Analytic Marshall sign sign(x) = (-1)^(N_down on sublattice A).

    NOTE: this rule is *exact for Heisenberg AFM* on bipartite lattices but
    does not match the exact Hubbard GS signs at finite U in the standard
    computational basis. Kept for diagnostics and as a fallback if ED is too
    expensive at large L. Use `signs_from_psi` whenever ED is available.
    """
    state_ints = np.asarray(state_ints, dtype=np.int64)
    A = _A_mask(L)
    dn = (state_ints >> L) & ((1 << L) - 1)
    masked = dn & A
    v = masked.astype(np.uint64).copy()
    pc = np.zeros_like(v, dtype=np.int64)
    while v.any():
        pc += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return np.where(pc & 1, -1, 1).astype(np.int64)
