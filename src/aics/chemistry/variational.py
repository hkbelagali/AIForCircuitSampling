"""Variational subspace diagonalization for Stage 1 Task A.

Given the full sector Hamiltonian H and a subset S = {x_1, ..., x_N} of basis
state indices, computes the smallest eigenvalue of H restricted to span(S),
i.e. E_var(S) = min_{|psi> in span(S)} <psi|H|psi> / <psi|psi>.

The headline Stage 1 metric is Delta E_var(S) = E_var(S) - E_0.
"""

import numpy as np
import scipy.linalg
import scipy.sparse as sp


def restrict_hamiltonian(H, indices):
    """Return H[indices,:][:,indices] as a Hermitized dense ndarray."""
    indices = np.asarray(list(indices), dtype=int)
    H_sub = H[indices, :][:, indices]
    H_dense = H_sub.toarray() if sp.issparse(H_sub) else np.asarray(H_sub)
    return 0.5 * (H_dense + H_dense.conj().T)


def var_eigenvalue(H, indices):
    """Smallest eigenvalue of H on span of the canonical basis vectors at `indices`."""
    indices = list(indices)
    if not indices:
        return float("inf")
    if len(indices) == 1:
        return float(H[indices[0], indices[0]].real)
    H_sub = restrict_hamiltonian(H, indices)
    evals = scipy.linalg.eigh(H_sub, eigvals_only=True)
    return float(evals[0])


def var_eigenvalue_and_vec(H, indices):
    """Ground eigenpair on span(indices). Vector is returned in the subspace basis."""
    indices = list(indices)
    if not indices:
        return float("inf"), np.zeros(0)
    if len(indices) == 1:
        return float(H[indices[0], indices[0]].real), np.array([1.0])
    H_sub = restrict_hamiltonian(H, indices)
    evals, evecs = scipy.linalg.eigh(H_sub)
    return float(evals[0]), evecs[:, 0]
