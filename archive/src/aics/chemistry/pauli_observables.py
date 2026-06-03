"""Pauli observable enumeration and exact expectation evaluation on a fixed
(N_up, N_dn) Hubbard sector.

Conventions:
  - Hubbard basis: 2L bits, with bit i = up_i for i in [0, L), bit L+i = dn_i.
  - Pauli operator at bit b: X, Y, or Z (or identity).
  - "Weight" of a Pauli string = number of non-identity factors.

For real Hubbard GS wavefunction Psi (signs * sqrt(prob)):
  <Psi|P|Psi> is real for any Hermitian Pauli P.
  - Paulis with an odd number of Y factors have <P> = 0 identically
    (matrix elements purely imaginary; real-Psi expectation vanishes).
  - Paulis whose X|Y mask changes (N_up, N_dn) on the sampled state contribute
    zero by sector orthogonality.
  We exploit both pruning rules.

The module precomputes, for each Pauli, a sparse representation of its
restriction to the sector as a list of (i, j, c) triples. <P> for any
wavefunction Psi is then sum_{(i,j,c)} Psi[i] * c * Psi[j].
"""

from dataclasses import dataclass
from itertools import combinations, product
from typing import List, Tuple

import numpy as np


_PAULI_X, _PAULI_Y, _PAULI_Z = 1, 2, 3  # tag values for type-of-Pauli-at-bit


@dataclass
class PauliOp:
    name: str             # e.g. "X0 Y3 Z5"
    weight: int
    nY: int
    x_mask: int           # bits with X
    y_mask: int           # bits with Y
    z_mask: int           # bits with Z
    sign_factor: int      # (-1)^(nY/2); +-1 (only defined when nY is even)
    triples: np.ndarray   # (n_nonzero, 3) int64 array of (i_row, j_col, c) -- c in {+1,-1}


def _popcount(v):
    v = int(v); c = 0
    while v:
        c += v & 1
        v >>= 1
    return c


def _enumerate_paulis(n_qubits, max_weight):
    """Yield (x_mask, y_mask, z_mask, weight, nY, name) for all Paulis up to
    a given weight. Skips Paulis with odd nY (they have zero real expectation
    against any real wavefunction)."""
    qubit_indices = list(range(n_qubits))
    for w in range(0, max_weight + 1):
        if w == 0:
            yield (0, 0, 0, 0, 0, "I")
            continue
        for support in combinations(qubit_indices, w):
            for types in product((_PAULI_X, _PAULI_Y, _PAULI_Z), repeat=w):
                xm = ym = zm = 0
                nY = 0
                parts = []
                for qbit, tp in zip(support, types):
                    if tp == _PAULI_X:
                        xm |= 1 << qbit
                        parts.append(f"X{qbit}")
                    elif tp == _PAULI_Y:
                        ym |= 1 << qbit
                        nY += 1
                        parts.append(f"Y{qbit}")
                    else:
                        zm |= 1 << qbit
                        parts.append(f"Z{qbit}")
                if nY % 2 != 0:
                    continue  # zero expectation on real Psi
                yield (xm, ym, zm, w, nY, " ".join(parts))


def build_pauli_ops(ctx, max_weight):
    """For a HubbardContext, enumerate weight-<= max_weight Paulis with even
    number of Y's, and precompute sector-restricted triples for each.

    Returns a list of PauliOp records (one per non-zero Pauli) plus a meta
    dict {"by_weight": {w: [indices into list]}}.
    """
    L = ctx.L
    n_qubits = 2 * L
    states = ctx.states.astype(np.int64)
    state_to_idx = ctx.state_to_idx
    L_mask = (1 << L) - 1

    up_arr = states & L_mask
    dn_arr = (states >> L) & L_mask
    Nup = _popcount(int(up_arr[0]))   # sector is constant; pick any
    Ndn = _popcount(int(dn_arr[0]))

    ops: List[PauliOp] = []
    by_weight = {}
    for (xm, ym, zm, w, nY, name) in _enumerate_paulis(n_qubits, max_weight):
        xy_mask = xm | ym
        yz_mask = ym | zm
        sign_factor = 1 if (nY // 2) % 2 == 0 else -1

        # Sector check: applying X|Y mask flips those bits. For the result to
        # remain in the (Nup, Ndn) sector for at least some states, we need to
        # iterate per-state.
        new_states = states ^ xy_mask
        new_up = new_states & L_mask
        new_dn = (new_states >> L) & L_mask
        # Bitwise popcount over int64 columns
        def pc(arr):
            v = np.asarray(arr, dtype=np.uint64).copy()
            out = np.zeros(v.shape, dtype=np.int64)
            while v.any():
                out += (v & np.uint64(1)).astype(np.int64)
                v >>= np.uint64(1)
            return out
        n_up_new = pc(new_up)
        n_dn_new = pc(new_dn)
        in_sector = (n_up_new == Nup) & (n_dn_new == Ndn)
        if not in_sector.any():
            continue

        # For surviving states, build triples.
        idx_rows = np.where(in_sector)[0].astype(np.int64)
        new_states_kept = new_states[idx_rows]
        # Look up j-indices
        j_indices = np.array([state_to_idx[int(s)] for s in new_states_kept],
                             dtype=np.int64)
        # Phase factor: (-1)^(popcount(x & yz_mask)) per row, times sign_factor.
        # x here is the ORIGINAL state.
        phase_count = pc(states[idx_rows] & yz_mask) & 1
        phase = np.where(phase_count == 0, 1, -1).astype(np.int64) * sign_factor
        triples = np.stack([idx_rows, j_indices, phase], axis=-1).astype(np.int64)

        ops.append(PauliOp(
            name=name, weight=w, nY=nY,
            x_mask=xm, y_mask=ym, z_mask=zm,
            sign_factor=sign_factor, triples=triples,
        ))
        by_weight.setdefault(w, []).append(len(ops) - 1)

    return ops, {"by_weight": by_weight, "n_qubits": n_qubits, "Nup": Nup, "Ndn": Ndn}


def pauli_expectations(ops: List[PauliOp], psi: np.ndarray):
    """Compute <P>_psi = sum_{(i,j,c) in triples} psi[i] * c * psi[j] for each P.

    Returns a (len(ops),) numpy array.
    """
    out = np.empty(len(ops), dtype=np.float64)
    for k, op in enumerate(ops):
        i, j, c = op.triples[:, 0], op.triples[:, 1], op.triples[:, 2]
        out[k] = float(np.sum(psi[i] * psi[j] * c.astype(np.float64)))
    return out


def max_abs_error_by_weight(ops_meta, vals_model, vals_true, max_weight=None):
    """Max |<P>_model - <P>_true| restricted to each weight bucket.

    Returns a dict {w: max_err}. Includes only weights present in ops_meta.
    """
    err = np.abs(vals_model - vals_true)
    by_weight = ops_meta["by_weight"]
    out = {}
    if max_weight is None:
        max_weight = max(by_weight.keys())
    for w in range(0, max_weight + 1):
        if w not in by_weight:
            continue
        idx = np.asarray(by_weight[w], dtype=np.int64)
        out[w] = float(err[idx].max())
    return out


def max_abs_error_cumulative(ops_meta, vals_model, vals_true, max_weight=None):
    """Max |error| over Paulis with weight <= w, for each w."""
    err = np.abs(vals_model - vals_true)
    by_weight = ops_meta["by_weight"]
    out = {}
    if max_weight is None:
        max_weight = max(by_weight.keys())
    cum_idx = []
    for w in range(0, max_weight + 1):
        if w in by_weight:
            cum_idx.extend(by_weight[w])
        if not cum_idx:
            continue
        idx = np.asarray(cum_idx, dtype=np.int64)
        out[w] = float(err[idx].max())
    return out
