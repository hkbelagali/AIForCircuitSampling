"""Greedy selected-CI expansion with the standard CIPSI / PT2 criterion.

At each step:
  1. Variationally diagonalize H restricted to current S, obtaining (E_S, psi_S).
  2. For each candidate x in allowed \\ S, compute the coupling c_x = <x|H|psi_S>
     and the PT2 score |c_x|^2 / (H[x,x] - E_S) (the 2nd-order energy correction
     in magnitude). When the denominator vanishes (degeneracy), promote x with
     infinite priority.
  3. Add the argmax candidate to S; repeat.

Restricted strictly to a caller-supplied allowed-support set. Returns the
sequence (sizes[], energies[]) so the M1 driver can plot Delta E_var vs |S|.
"""

import numpy as np
import scipy.linalg
import scipy.sparse as sp


def _gs_of_restricted(H, S):
    H_S = H[S, :][:, S]
    H_S = H_S.toarray() if sp.issparse(H_S) else np.asarray(H_S)
    H_S = 0.5 * (H_S + H_S.conj().T)
    evals, evecs = scipy.linalg.eigh(H_S)
    return float(evals[0]), evecs[:, 0]


def greedy_sci_expansion(H, allowed, seed=None, max_size=None):
    """Run greedy SCI from `seed` (or auto-pick min-H_diag). Returns sizes, energies, final S."""
    allowed = np.asarray(allowed, dtype=int)
    if max_size is None:
        max_size = len(allowed)
    max_size = min(int(max_size), len(allowed))
    allowed_diag = np.array([float(H[int(i), int(i)].real) for i in allowed])

    if seed is None:
        seed = int(allowed[int(np.argmin(allowed_diag))])

    S = [int(seed)]
    S_set = {S[0]}
    energies = [float(H[S[0], S[0]].real)]
    sizes = [1]

    while len(S) < max_size:
        if len(S) == 1:
            E_S = float(H[S[0], S[0]].real)
            psi_S = np.array([1.0])
        else:
            E_S, psi_S = _gs_of_restricted(H, S)

        cand_mask = np.array([int(a) not in S_set for a in allowed])
        cand_indices = allowed[cand_mask]
        if cand_indices.size == 0:
            break

        H_cand_S = H[cand_indices, :][:, S]
        H_cand_S = H_cand_S.toarray() if sp.issparse(H_cand_S) else np.asarray(H_cand_S)
        coupling = H_cand_S @ psi_S
        coupling_sq = np.abs(coupling) ** 2

        cand_diag = allowed_diag[cand_mask]
        denom = cand_diag - E_S
        eps = 1e-10
        score = np.where(denom > eps,
                         coupling_sq / np.maximum(denom, eps),
                         np.full_like(coupling_sq, np.inf))
        best = int(cand_indices[int(np.argmax(score))])
        S.append(best)
        S_set.add(best)

        E_new, _ = _gs_of_restricted(H, S)
        energies.append(E_new)
        sizes.append(len(S))

    return sizes, energies, S
