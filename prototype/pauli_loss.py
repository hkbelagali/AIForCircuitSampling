"""Shadow-Pauli loss: train psi_theta to match shadow estimates
<P>_shadow for every even-Y, sector-preserving Pauli of weight <= w_max.

Loss form:
    L(theta) = sum_P  alpha_P * ( <P>_theta - <P>_shadow )^2

with two weighting choices:
    weighting='uniform'  -> alpha_P = 1
    weighting='variance' -> alpha_P = 3^{-|P|}   (~ 1 / Var[<P>_shadow])

Computation:
  - shadow_targets:  per-Pauli unbiased shadow estimator from (U_t, b_t) shots
    via the standard random-Pauli inverse channel formula
       <P>_t = 3^|P| * [U_t matches P on supp(P)] * (-1)^{sum_q in S b_t,q}.
  - <P>_theta:  for each Pauli we precompute the sector-restricted triples
    (i, j, c) such that <P>_theta = sum_{(i,j,c)} psi[i] * c * psi[j].
    All Paulis' triples are stacked into single I/J/C/P_idx arrays so that
    one torch.index_add_ computes every <P>_theta in a single step.
"""

from itertools import combinations, product

import numpy as np
import torch

from pauli import _popcount, _popcount_arr

_X, _Y, _Z = 1, 2, 3
# Map between U_pattern codes (0=Z, 1=X, 2=Y -- the convention used in
# shadows.py for sample_shadows_random_pauli) and the (xm, ym, zm) bitmask
# convention used here.
_U_TO_XYZ = {0: _Z, 1: _X, 2: _Y}


def _enumerate_sector_paulis(ctx, max_weight):
    """Yield (xm, ym, zm, w, nY, supp_indices, supp_codes) for every
    even-Y, sector-preserving Pauli of weight 0..max_weight.

    Sector preservation requires even popcount in EACH HALF of the flip
    mask xy = xm | ym (the per-qubit X/Y bits that swap basis states).
    Z-only Paulis always preserve the sector regardless of support.

    supp_codes[i] is the U_pattern code (0/1/2) for the i-th support qubit.
    """
    L = ctx.L
    n_qubits = 2 * L
    Lm = (1 << L) - 1
    yield 0, 0, 0, 0, 0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    for w in range(1, max_weight + 1):
        for support in combinations(range(n_qubits), w):
            sup_arr = np.array(support, dtype=np.int64)
            for types in product((_X, _Y, _Z), repeat=w):
                nY = sum(1 for t in types if t == _Y)
                if nY % 2 != 0:
                    continue
                xm = ym = zm = 0
                codes = np.empty(w, dtype=np.int64)
                for i, (q, t) in enumerate(zip(support, types)):
                    if t == _X:
                        xm |= 1 << q; codes[i] = 1
                    elif t == _Y:
                        ym |= 1 << q; codes[i] = 2
                    else:
                        zm |= 1 << q; codes[i] = 0
                xy = xm | ym
                up_flip = _popcount(xy & Lm)
                dn_flip = _popcount((xy >> L) & Lm)
                if (up_flip & 1) or (dn_flip & 1):
                    continue
                yield xm, ym, zm, w, nY, sup_arr, codes


def _triples_for_pauli(xm, ym, zm, w, nY, ctx):
    """Sector-restricted matrix elements for a single Pauli: returns
    (i_arr, j_arr, c_arr) so that <P>_psi = sum_t psi[i_t] * c_t * psi[j_t]."""
    L = ctx.L
    Lm = (1 << L) - 1
    states = ctx.states.astype(np.int64)
    Nup = _popcount(int(states[0]) & Lm)
    Ndn = _popcount((int(states[0]) >> L) & Lm)
    xy = xm | ym
    yz = ym | zm
    sign_fac = 1 if (nY // 2) % 2 == 0 else -1
    new_states = states ^ xy
    new_up = new_states & Lm
    new_dn = (new_states >> L) & Lm
    in_sector = (_popcount_arr(new_up) == Nup) & (_popcount_arr(new_dn) == Ndn)
    if not in_sector.any():
        return None
    idx_rows = np.where(in_sector)[0]
    # State-to-index lookup via searchsorted on sorted states.
    sort_perm = np.argsort(states, kind="stable").astype(np.int64)
    sorted_states = states[sort_perm]
    inv_perm = np.empty_like(sort_perm)
    inv_perm[sort_perm] = np.arange(len(states), dtype=np.int64)
    pos = np.searchsorted(sorted_states, new_states[idx_rows])
    j_idx = inv_perm[pos]
    phase_par = _popcount_arr(states[idx_rows] & yz) & 1
    c = sign_fac * np.where(phase_par == 0, 1, -1).astype(np.int64)
    return idx_rows.astype(np.int64), j_idx, c


def build_loss_paulis(ctx, max_weight):
    """Enumerate Paulis once. Returns dict containing:
      'xyz':           (n_P, 3) int — xm, ym, zm per Pauli
      'weight':        (n_P,) int   — |P|
      'supports':      list[np.int64 array] of qubit indices per Pauli
      'supp_codes':    list[np.int64 array] of U-codes per Pauli
      'I','J','C','P': flat arrays for stacked triples (for fast <P>_theta)
      'n_paulis':      int
    """
    xyz = []; weights = []; supports = []; supp_codes = []
    all_I = []; all_J = []; all_C = []; all_P = []
    p_idx = 0
    # Always include identity at index 0.
    for xm, ym, zm, w, nY, sup_arr, codes in _enumerate_sector_paulis(
            ctx, max_weight):
        tr = _triples_for_pauli(xm, ym, zm, w, nY, ctx)
        if tr is None:
            continue
        i, j, c = tr
        xyz.append((xm, ym, zm)); weights.append(w)
        supports.append(sup_arr); supp_codes.append(codes)
        all_I.append(i); all_J.append(j); all_C.append(c)
        all_P.append(np.full(len(i), p_idx, dtype=np.int64))
        p_idx += 1
    return {
        "xyz": np.asarray(xyz, dtype=np.int64),
        "weight": np.asarray(weights, dtype=np.int64),
        "supports": supports,
        "supp_codes": supp_codes,
        "I": np.concatenate(all_I),
        "J": np.concatenate(all_J),
        "C": np.concatenate(all_C).astype(np.float64),
        "P": np.concatenate(all_P),
        "n_paulis": p_idx,
    }


def shadow_targets(loss_paulis, U_pattern, b_out):
    """Compute the shadow estimate <P>_shadow for each Pauli in loss_paulis.
    Returns (n_paulis,) float64."""
    n_p = loss_paulis["n_paulis"]
    weights = loss_paulis["weight"]
    supports = loss_paulis["supports"]
    supp_codes = loss_paulis["supp_codes"]
    out = np.empty(n_p, dtype=np.float64)
    out[0] = 1.0  # identity Pauli
    for p in range(1, n_p):
        S = supports[p]; codes = supp_codes[p]
        w = weights[p]
        U_at_S = U_pattern[:, S]               # (k, w)
        matches = (U_at_S == codes[None, :]).all(axis=1)  # (k,) bool
        if not matches.any():
            out[p] = 0.0
            continue
        b_at_S = b_out[:, S]                   # (k, w)
        parity = b_at_S.sum(axis=1) & 1        # (k,)
        sign = np.where(parity == 0, 1.0, -1.0)
        out[p] = float((matches.astype(np.float64) * sign).mean() * (3.0 ** w))
    return out


def torch_ops(loss_paulis, device):
    """Move stacked-triple arrays onto torch for fast <P>_theta computation."""
    return {
        "I": torch.from_numpy(loss_paulis["I"]).long().to(device),
        "J": torch.from_numpy(loss_paulis["J"]).long().to(device),
        "C": torch.from_numpy(loss_paulis["C"]).double().to(device),
        "P": torch.from_numpy(loss_paulis["P"]).long().to(device),
        "n_paulis": loss_paulis["n_paulis"],
    }


def model_expectations(psi, ops):
    """psi: (D,) real torch tensor.  Returns (n_paulis,) torch tensor of
    <P>_theta = sum_t psi[I_t] * C_t * psi[J_t] grouped by Pauli."""
    psi = psi.to(torch.float64)
    products = psi[ops["I"]] * ops["C"] * psi[ops["J"]]
    out = torch.zeros(ops["n_paulis"], dtype=torch.float64, device=psi.device)
    out.index_add_(0, ops["P"], products)
    return out


def alpha_array(loss_paulis, weighting):
    """Per-Pauli loss weight."""
    w = loss_paulis["weight"]
    if weighting == "uniform":
        return np.ones(len(w), dtype=np.float64)
    if weighting == "variance":
        return 3.0 ** (-w.astype(np.float64))
    raise ValueError(f"unknown weighting {weighting!r}")
