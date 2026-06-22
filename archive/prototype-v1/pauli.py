"""Weight-bounded Pauli observables on the Hubbard (N_up, N_dn) sector.

For each Pauli P with even number of Y factors that has a non-empty
sector-preserving matrix-element set, precomputes triples (i, j, c) so
that <Psi|P|Psi> = sum_{i,j,c in P.triples} Psi[i] * c * Psi[j] for
any real Psi over the sector.

Paulis with an odd number of Y factors are pruned (zero expectation
against real wavefunctions). Paulis whose X|Y mask leaves the sector
on every basis state are also dropped.
"""

from itertools import combinations, product

import numpy as np

_X, _Y, _Z = 1, 2, 3


def _popcount(v):
    v = int(v); c = 0
    while v:
        c += v & 1
        v >>= 1
    return c


def _popcount_arr(a):
    v = np.asarray(a, dtype=np.uint64).copy()
    out = np.zeros(v.shape, dtype=np.int64)
    while v.any():
        out += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return out


def _enumerate(n_qubits, max_weight):
    yield 0, 0, 0, 0, 0
    for w in range(1, max_weight + 1):
        for support in combinations(range(n_qubits), w):
            for types in product((_X, _Y, _Z), repeat=w):
                xm = ym = zm = 0
                nY = 0
                for q, t in zip(support, types):
                    if t == _X: xm |= 1 << q
                    elif t == _Y: ym |= 1 << q; nY += 1
                    else: zm |= 1 << q
                if nY % 2 == 0:
                    yield xm, ym, zm, w, nY


def build_pauli_triples(ctx, max_weight):
    """Returns (ops, weights) where ops is a list of (np.int64 (n_triples, 3))
    triples arrays and weights is a parallel list of weights."""
    L = ctx.L
    n_qubits = 2 * L
    states = ctx.states.astype(np.int64)
    state_to_idx = ctx.idx
    Lm = (1 << L) - 1
    Nup = _popcount(int(states[0]) & Lm)
    Ndn = _popcount((int(states[0]) >> L) & Lm)

    ops, weights = [], []
    for xm, ym, zm, w, nY in _enumerate(n_qubits, max_weight):
        xy = xm | ym
        yz = ym | zm
        sign_fac = 1 if (nY // 2) % 2 == 0 else -1
        new_states = states ^ xy
        new_up = new_states & Lm
        new_dn = (new_states >> L) & Lm
        in_sector = (_popcount_arr(new_up) == Nup) & (_popcount_arr(new_dn) == Ndn)
        if not in_sector.any():
            continue
        idx_rows = np.where(in_sector)[0]
        j_idx = np.array([state_to_idx[int(s)] for s in new_states[idx_rows]],
                         dtype=np.int64)
        phase_par = _popcount_arr(states[idx_rows] & yz) & 1
        c = sign_fac * np.where(phase_par == 0, 1, -1).astype(np.int64)
        triples = np.stack([idx_rows.astype(np.int64), j_idx, c], axis=-1)
        ops.append(triples)
        weights.append(w)
    return ops, np.asarray(weights, dtype=np.int64)


def expectations(ops, psi):
    out = np.empty(len(ops), dtype=np.float64)
    for k, tr in enumerate(ops):
        i, j, c = tr[:, 0], tr[:, 1], tr[:, 2]
        out[k] = float(np.sum(psi[i] * psi[j] * c.astype(np.float64)))
    return out


def max_err_by_weight(weights, vals_model, vals_true, max_w):
    """Returns {w: max |<P>_model - <P>_true| over weight-<=w Paulis}."""
    err = np.abs(vals_model - vals_true)
    out = {}
    mask = np.zeros(len(weights), dtype=bool)
    for w in range(0, max_w + 1):
        mask |= (weights == w)
        if mask.any():
            out[w] = float(err[mask].max())
    return out


def err_streaming(ctx, psi_model, psi_true, max_weight):
    """Stream all even-Y, sector-preserving Paulis up to weight max_weight;
    return ({w: max |<P>_model - <P>_true| at exact weight w},
            {w: mean |<P>_model - <P>_true| at exact weight w},
            {w: count of Paulis evaluated at exact weight w}).

    Memory-efficient: never materializes a triples list. Suitable for
    max_weight up to N. Cheap parity pre-filter rejects ~75% of Paulis
    before the expensive sector check.
    """
    L = ctx.L
    n_qubits = 2 * L
    states = ctx.states.astype(np.int64)
    Lm = (1 << L) - 1
    Nup = _popcount(int(states[0]) & Lm)
    Ndn = _popcount((int(states[0]) >> L) & Lm)
    sort_perm = np.argsort(states, kind="stable").astype(np.int64)
    sorted_states = states[sort_perm]
    inv_perm = np.empty_like(sort_perm)
    inv_perm[sort_perm] = np.arange(len(states), dtype=np.int64)

    psi_m = np.asarray(psi_model, dtype=np.float64)
    psi_t = np.asarray(psi_true, dtype=np.float64)

    max_at_w = {0: abs(float(psi_m @ psi_m) - float(psi_t @ psi_t))}
    sum_at_w = {0: max_at_w[0]}
    cnt_at_w = {0: 1}

    for xm, ym, zm, w, nY in _enumerate(n_qubits, max_weight):
        if w == 0: continue
        xy = xm | ym
        flip_up = xy & Lm
        flip_dn = (xy >> L) & Lm
        if ((_popcount(flip_up) & 1) != 0) or ((_popcount(flip_dn) & 1) != 0):
            continue
        yz = ym | zm
        sign_fac = 1.0 if (nY // 2) % 2 == 0 else -1.0
        new_states = states ^ xy
        new_up = new_states & Lm
        new_dn = (new_states >> L) & Lm
        in_sector = (_popcount_arr(new_up) == Nup) & (_popcount_arr(new_dn) == Ndn)
        if not in_sector.any():
            continue
        idx_rows = np.where(in_sector)[0]
        new_in = new_states[idx_rows]
        pos = np.searchsorted(sorted_states, new_in)
        j_idx = inv_perm[pos]
        phase_par = _popcount_arr(states[idx_rows] & yz) & 1
        c = sign_fac * np.where(phase_par == 0, 1.0, -1.0)
        v_m = float(np.sum(psi_m[idx_rows] * psi_m[j_idx] * c))
        v_t = float(np.sum(psi_t[idx_rows] * psi_t[j_idx] * c))
        e = abs(v_m - v_t)
        prev = max_at_w.get(w)
        if prev is None or e > prev: max_at_w[w] = e
        sum_at_w[w] = sum_at_w.get(w, 0.0) + e
        cnt_at_w[w] = cnt_at_w.get(w, 0) + 1
    mean_at_w = {w: sum_at_w[w] / cnt_at_w[w] for w in sum_at_w}
    return max_at_w, mean_at_w, cnt_at_w


def cumulative_from_exact(err_exact, max_w):
    """Convert {w: max-err-at-exact-weight-w} to {w: max-err-over-weight-<=w}."""
    out = {}
    running = 0.0
    for w in range(max_w + 1):
        if w in err_exact:
            running = max(running, err_exact[w])
        out[w] = running
    return out
