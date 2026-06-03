"""Linear cross-entropy benchmarking (XEB) and radius-filtered novel-XEB.

  F_XEB^{(k)} = dim * mean_i [p_C(x_i)] - 1

For random Porter-Thomas samples x_i ~ p_C, this estimator has mean ~1 (after
the Porter-Thomas correction). Uniform-random bitstrings give mean 0.

  novel-XEB(r) = same, restricted to candidates whose Hamming distance to the
                 training set is at least r.

The radius-filtered version is the headline-A metric for Stage 3.
"""

import numpy as np


def linear_xeb(samples_int, p_C):
    """Linear XEB on a sample set indexed into the exact p_C array."""
    if len(samples_int) == 0:
        return float("nan")
    dim = int(len(p_C))
    return dim * float(p_C[samples_int].mean()) - 1.0


def _popcount_uint64(arr):
    """Per-element popcount on a numpy array of integers (works for any int)."""
    v = np.asarray(arr, dtype=np.uint64).copy()
    out = np.zeros(v.shape, dtype=np.int64)
    while v.any():
        out += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return out


def min_hamming_to_set(candidates_int, training_int):
    """Min Hamming distance from each candidate to the set of training bitstrings.

    For m candidates and k training entries, returns an array of length m.
    Memory ~ O(m * k) ints, so OK up to ~1e7 pairs at small n.
    """
    candidates = np.asarray(candidates_int, dtype=np.int64)
    training = np.unique(np.asarray(training_int, dtype=np.int64))
    if training.size == 0:
        # No training -> "novel everywhere"; report a Hamming-distance sentinel.
        return np.full(candidates.shape, np.iinfo(np.int64).max, dtype=np.int64)
    xor = candidates[:, None] ^ training[None, :]   # (m, k)
    pc = _popcount_uint64(xor)
    return pc.min(axis=1)


def novel_xeb_vs_radius(candidates_int, training_int, p_C, n, radii=None,
                         dedup=True):
    """Radius-filtered novel-XEB.

    Returns dict {r: (xeb, count)} where count is the number of candidates
    satisfying min_dH(candidate, training) >= r.
    """
    if radii is None:
        radii = list(range(0, n + 1))
    cand = np.asarray(candidates_int, dtype=np.int64)
    if dedup:
        cand = np.unique(cand)
    if cand.size == 0:
        return {r: (float("nan"), 0) for r in radii}
    min_dH = min_hamming_to_set(cand, training_int)
    dim = int(len(p_C))
    out = {}
    for r in radii:
        mask = min_dH >= r
        if mask.sum() == 0:
            out[r] = (float("nan"), 0)
        else:
            xeb = dim * float(p_C[cand[mask]].mean()) - 1.0
            out[r] = (xeb, int(mask.sum()))
    return out
