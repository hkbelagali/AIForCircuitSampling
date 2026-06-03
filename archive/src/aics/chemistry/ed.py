"""Ground-state exact diagonalization for sparse Hermitian operators."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla


def ground_state(H, dense_threshold=400):
    """Return (E_0, psi_0) for Hermitian H.

    Dense numpy.linalg.eigh below dense_threshold dimension; sparse ARPACK
    eigsh otherwise. Hermitizes H defensively before solving.
    """
    D = H.shape[0]
    if D <= dense_threshold:
        H_dense = H.toarray() if sp.issparse(H) else np.asarray(H)
        H_dense = 0.5 * (H_dense + H_dense.conj().T)
        evals, evecs = np.linalg.eigh(H_dense)
        return float(evals[0]), evecs[:, 0]
    if not sp.issparse(H):
        H = sp.csr_matrix(H)
    H = 0.5 * (H + H.conj().T)
    evals, evecs = sla.eigsh(H, k=1, which="SA")
    return float(evals[0]), evecs[:, 0]
