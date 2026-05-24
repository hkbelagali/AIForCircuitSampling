"""Lattice-symmetry diagnostics for chemistry/Hubbard ground states.

Implements the exact symmetry generators of the 1D Hubbard model in fixed
(n_up, n_dn) sector and provides projectors onto 1D irreducible representations
plus diagnostics for GS symmetry signatures.

Symmetries implemented (with state convention from aics.common.symmetry):
  - Translation T (1 site shift), PBC only. Per-spin sign (-1)^(N_sigma - 1)
    iff wrap bit set. Order L on (n_up, n_dn) when total N is appropriate.
  - Reflection R (i -> L-1-i for both spins). Per-spin sign
    (-1)^(N_sigma*(N_sigma-1)/2), state-independent. Order 2.
  - Spin-flip S (up <-> dn at each site). Sign (-1)^(N_up * N_dn),
    state-independent. Order 2. Preserves sector only when N_up == N_dn.

Particle-hole at half-filling on a bipartite lattice is also exact but
deferred — translation + reflection + spin-flip already account for the
dominant computational-basis suppression on L = 4.
"""

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from aics.common.symmetry import sector_states


# ---- helpers ---------------------------------------------------------------


def _bit_reverse(x, n_bits):
    out = 0
    for i in range(n_bits):
        if (x >> i) & 1:
            out |= 1 << (n_bits - 1 - i)
    return out


# ---- symmetry operators ----------------------------------------------------


def translation_op_1d(L, n_up, n_dn):
    """One-site cyclic translation T in the (n_up, n_dn) sector of a 1D chain."""
    states = sector_states(L, n_up, n_dn)
    D = states.size
    state_to_idx = {int(s): i for i, s in enumerate(states)}
    L_mask = (1 << L) - 1
    high_bit = 1 << (L - 1)

    rows, cols, vals = [], [], []
    for idx in range(D):
        x = int(states[idx])
        up = x & L_mask
        dn = x >> L
        new_up = ((up << 1) & L_mask) | ((up >> (L - 1)) & 1)
        new_dn = ((dn << 1) & L_mask) | ((dn >> (L - 1)) & 1)
        sign_up = -1 if (up & high_bit) and (n_up % 2 == 0) else 1
        sign_dn = -1 if (dn & high_bit) and (n_dn % 2 == 0) else 1
        sign = sign_up * sign_dn
        new_x = (new_dn << L) | new_up
        rows.append(state_to_idx[new_x]); cols.append(idx); vals.append(complex(sign))

    return sp.csr_matrix((vals, (rows, cols)), shape=(D, D), dtype=np.complex128)


def reflection_op_1d(L, n_up, n_dn):
    """Spatial reflection R: site i -> L-1-i for both spins.

    Per-spin sign is (-1)^(N_sigma * (N_sigma - 1) / 2), state-independent
    (comes from reversing N operator factors -> N(N-1)/2 anticommutations).
    """
    states = sector_states(L, n_up, n_dn)
    D = states.size
    state_to_idx = {int(s): i for i, s in enumerate(states)}
    L_mask = (1 << L) - 1

    sign_up = (-1) ** (n_up * (n_up - 1) // 2)
    sign_dn = (-1) ** (n_dn * (n_dn - 1) // 2)
    sign = float(sign_up * sign_dn)

    rows, cols, vals = [], [], []
    for idx in range(D):
        x = int(states[idx])
        up = x & L_mask
        dn = x >> L
        new_up = _bit_reverse(up, L)
        new_dn = _bit_reverse(dn, L)
        new_x = (new_dn << L) | new_up
        rows.append(state_to_idx[new_x]); cols.append(idx); vals.append(complex(sign))

    return sp.csr_matrix((vals, (rows, cols)), shape=(D, D), dtype=np.complex128)


def spin_flip_op(L, n_up, n_dn):
    """Spin-flip S: c_{i,up} <-> c_{i,dn} at each site.

    Sign (-1)^(N_up * N_dn) from interchanging the up-block past the dn-block.
    Requires n_up == n_dn to preserve the (n_up, n_dn) sector.
    """
    if n_up != n_dn:
        raise ValueError("spin_flip preserves the (n_up, n_dn) sector only when n_up == n_dn")
    states = sector_states(L, n_up, n_dn)
    D = states.size
    state_to_idx = {int(s): i for i, s in enumerate(states)}
    L_mask = (1 << L) - 1

    sign = float((-1) ** (n_up * n_dn))

    rows, cols, vals = [], [], []
    for idx in range(D):
        x = int(states[idx])
        up = x & L_mask
        dn = x >> L
        new_x = (up << L) | dn
        rows.append(state_to_idx[new_x]); cols.append(idx); vals.append(complex(sign))

    return sp.csr_matrix((vals, (rows, cols)), shape=(D, D), dtype=np.complex128)


# ---- decompositions / projectors -------------------------------------------


def momentum_decomposition(psi, T, bin_decimals=4):
    """Decompose |psi|^2 weight by translation eigenvalue."""
    T_dense = T.toarray() if sp.issparse(T) else np.asarray(T)
    T_S, U = scipy.linalg.schur(T_dense, output="complex")
    evals = np.diag(T_S)
    if not np.allclose(np.abs(evals), 1.0, atol=1e-8):
        raise ValueError("T is not unitary; max |1 - |lambda|| = "
                         f"{np.max(np.abs(1 - np.abs(evals)))}")
    psi_in_T = U.conj().T @ psi.astype(np.complex128)
    weights = np.abs(psi_in_T) ** 2
    phases = np.angle(evals)
    binned = {}
    for ph, w in zip(phases, weights):
        key = round(float(ph), bin_decimals)
        binned[key] = binned.get(key, 0.0) + float(w)
    return sorted(binned.items())


def sector_dim_by_phase(T, bin_decimals=4):
    """Dimension of each translation-eigenvalue subspace."""
    T_dense = T.toarray() if sp.issparse(T) else np.asarray(T)
    T_S, _ = scipy.linalg.schur(T_dense, output="complex")
    evals = np.diag(T_S)
    phases = np.angle(evals)
    counts = {}
    for ph in phases:
        key = round(float(ph), bin_decimals)
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items())


def gs_irrep_signature(psi, ops, tol=1e-8):
    """Return {name: eigenvalue} for psi under each op in ops dict.

    Warns if psi is not an eigenvector (residual ||O psi - lambda psi|| > tol).
    """
    psi_c = psi.astype(np.complex128)
    out = {}
    warnings = []
    for name, op in ops.items():
        op_psi = op @ psi_c
        lam = complex(np.vdot(psi_c, op_psi))
        residual = float(np.linalg.norm(op_psi - lam * psi_c))
        if residual > tol:
            warnings.append((name, residual))
        out[name] = lam
    return out, warnings


def project_to_irrep(ops_orders_eigs):
    """Build the projector onto a 1D irrep specified by (op, order, eigenvalue) tuples.

    For each op O with order m (O^m = I) and target eigenvalue lambda:
        P_O = (1/m) sum_{a=0}^{m-1} lambda^{-a} O^a
    The combined projector is the product Pi P_O. For abelian generators this
    is exact; for D_n generated by T and R with real eigenvalues lambda_T = ±1
    and real lambda_R, the product P_T P_R also coincides with the proper
    character projector onto the corresponding 1D irrep (the +/-1 condition
    eliminates the non-commutativity in the sum).
    """
    P = None
    for op, order, eig in ops_orders_eigs:
        D = op.shape[0]
        P_op = sp.csr_matrix((D, D), dtype=np.complex128)
        op_a = sp.eye(D, format="csr", dtype=np.complex128)
        for a in range(order):
            P_op = P_op + (complex(eig) ** (-a) / order) * op_a
            op_a = op_a @ op
        P = P_op if P is None else P @ P_op
    return P


def symmetry_allowed_support(P, tol=1e-10):
    """Computational basis indices |x> with ||P|x>|| > tol. Sparse-friendly:
    avoids ever materializing the (D, D) dense projector.
    """
    if sp.issparse(P):
        col_norm_sq = np.asarray(P.multiply(P.conj()).sum(axis=0)).ravel().real
        col_norm = np.sqrt(np.maximum(col_norm_sq, 0.0))
    else:
        col_norm = np.linalg.norm(P, axis=0)
    support = np.where(col_norm > tol)[0]
    return int(support.size), support


def irrep_subspace_dim(P, tol=1e-10):
    """Dimension of the irrep subspace = trace(P) for an idempotent P.

    Uses the projector identity P^2 = P => rank(P) = trace(P). Falls back to
    rounding to the nearest integer to absorb small numerical drift.
    """
    if sp.issparse(P):
        tr = complex(P.diagonal().sum())
    else:
        tr = complex(np.trace(P))
    return int(round(tr.real))
