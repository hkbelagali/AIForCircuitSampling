"""Z-observable expectations and per-weight RMS error.

For a support set S ⊆ {0..n-1}, Z_S = ∏_{q∈S} Z_q and its expectation in
state ψ is
    <Z_S>_ψ = Σ_z (-1)^{|z ∩ S|} |ψ(z)|^2.

Empirical estimator from k bitstring samples:
    <Z_S>_empirical = mean_t (1 - 2 b_t,S)   where b_t,S = sum_{q∈S} bit_t,q mod 2.

We previously called this `shadow_z_expectations`; the "shadow" naming
was leftover from earlier Hubbard-tomography work and didn't apply — these
are direct sample-mean estimators, no shadow inverse-channel involved.
The legacy name remains as an alias for back-compat with existing prototypes.
"""
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch

from ..io.conventions import int_to_bits


def enumerate_z_supports(n, max_weight):
    """Returns (supports, weights) for every Z-Pauli subset of size ≤ max_weight.

    Identity (empty support, weight 0) is index 0.
    """
    supports = [()]
    weights = [0]
    for w in range(1, max_weight + 1):
        for S in combinations(range(n), w):
            supports.append(S)
            weights.append(w)
    return supports, np.asarray(weights, dtype=np.int64)


def parity_matrix(supports, n):
    """W of shape (n_obs, 2^n): W[i, x] = (-1)^{|x ∩ S_i|}.

    Bits are MSB-first to match `aics.io.conventions`.
    """
    dim = 1 << n
    all_int = np.arange(dim, dtype=np.int64)
    all_bits = int_to_bits(all_int, n)
    W = np.ones((len(supports), dim), dtype=np.float64)
    for i, S in enumerate(supports):
        if not S:
            continue
        parity = all_bits[:, list(S)].sum(axis=1) & 1
        W[i] = np.where(parity == 0, 1.0, -1.0)
    return W


def empirical_z_expectations(samples_int, supports, n):
    """Empirical <Z_S> = mean (-1)^{|x ∩ S|} over k integer-indexed samples.
    `samples_int` is MSB-first. Returns (n_obs,) float64.
    """
    if len(samples_int) == 0:
        return np.zeros(len(supports), dtype=np.float64)
    sample_bits = int_to_bits(samples_int, n)
    return _z_expectations_from_bits(sample_bits, supports)


def empirical_z_expectations_from_bits(samples_bits, supports):
    """Same as `empirical_z_expectations` but takes a (k, n) bit array."""
    return _z_expectations_from_bits(np.asarray(samples_bits, dtype=np.uint8),
                                       supports)


def _z_expectations_from_bits(sample_bits, supports):
    out = np.empty(len(supports), dtype=np.float64)
    for i, S in enumerate(supports):
        if not S:
            out[i] = 1.0
            continue
        parity = sample_bits[:, list(S)].sum(axis=1) & 1
        out[i] = float(np.where(parity == 0, 1.0, -1.0).mean())
    return out


# Back-compat alias for callers that still say "shadow_z_expectations".
shadow_z_expectations = empirical_z_expectations


def parity_per_support(samples_bits, supports):
    """Equivalent to `empirical_z_expectations_from_bits` — historical
    name used by the NLL eval cell."""
    return empirical_z_expectations_from_bits(samples_bits, supports)


@torch.no_grad()
def model_z_expectations(model, supports, n, device=None):
    """<Z_S>_θ via full-distribution forward over 2^n bitstrings. Only
    tractable for small n (≲ 20). Returns (n_obs,) numpy float64.
    """
    device = device or next(model.parameters()).device
    dim = 1 << n
    all_int = np.arange(dim, dtype=np.int64)
    all_bits_t = torch.from_numpy(int_to_bits(all_int, n)).float().to(device)
    logp = model.log_prob(all_bits_t).to(torch.float64)
    p = torch.softmax(logp, dim=0).cpu().numpy()
    W = parity_matrix(supports, n)
    return W @ p


def per_weight_rms_err(supports, pred, true):
    """Group |pred[i] - true[i]|^2 by |S_i| = w, return RMS per weight as dict."""
    by_w = defaultdict(list)
    for j, S in enumerate(supports):
        by_w[len(S)].append((pred[j] - true[j]) ** 2)
    return {w: float(np.sqrt(np.mean(arr))) for w, arr in by_w.items()}


def per_weight_rms_true(supports, true):
    """RMS magnitude of the true expectations per weight — useful as a
    normalising scale for per-weight error."""
    by_w = defaultdict(list)
    for j, S in enumerate(supports):
        by_w[len(S)].append(true[j] ** 2)
    return {w: float(np.sqrt(np.mean(arr))) for w, arr in by_w.items()}
