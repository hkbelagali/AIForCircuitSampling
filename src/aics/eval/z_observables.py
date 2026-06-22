"""Z-observable expectations and per-weight RMS error.

  <Z_S>_psi = sum_z (-1)^{|z ∩ S|} |psi(z)|^2
  <Z_S>_empirical = mean_t (1 - 2 b_{t,S})  where b_{t,S} = sum_{q in S} bit_{t,q} mod 2.
"""
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch

from ..io.conventions import int_to_bits


def enumerate_z_supports(n, max_weight):
    """(supports, weights). Identity (empty support) is index 0."""
    supports = [()]
    weights = [0]
    for w in range(1, max_weight + 1):
        for S in combinations(range(n), w):
            supports.append(S)
            weights.append(w)
    return supports, np.asarray(weights, dtype=np.int64)


def parity_matrix(supports, n):
    """W[i, x] = (-1)^{|x ∩ S_i|} for x ∈ [0, 2^n), MSB-first."""
    dim = 1 << n
    all_bits = int_to_bits(np.arange(dim, dtype=np.int64), n)
    W = np.ones((len(supports), dim), dtype=np.float64)
    for i, S in enumerate(supports):
        if S:
            parity = all_bits[:, list(S)].sum(axis=1) & 1
            W[i] = np.where(parity == 0, 1.0, -1.0)
    return W


def empirical_z_expectations(samples_int, supports, n):
    if len(samples_int) == 0:
        return np.zeros(len(supports), dtype=np.float64)
    return _z_expectations_from_bits(int_to_bits(samples_int, n), supports)


def empirical_z_expectations_from_bits(samples_bits, supports):
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


# Historical name used by m_rcs_nll_eval_cell.py.
parity_per_support = empirical_z_expectations_from_bits


@torch.no_grad()
def model_z_expectations(model, supports, n, device=None):
    """<Z_S>_θ via full-distribution forward over 2^n bitstrings. Tractable n ≲ 20."""
    device = device or next(model.parameters()).device
    dim = 1 << n
    all_int = np.arange(dim, dtype=np.int64)
    all_bits_t = torch.from_numpy(int_to_bits(all_int, n)).float().to(device)
    logp = model.log_prob(all_bits_t).to(torch.float64)
    p = torch.softmax(logp, dim=0).cpu().numpy()
    return parity_matrix(supports, n) @ p


def per_weight_rms_err(supports, pred, true):
    """Group squared errors by |S|, return RMS per weight as {w: rms}."""
    by_w = defaultdict(list)
    for j, S in enumerate(supports):
        by_w[len(S)].append((pred[j] - true[j]) ** 2)
    return {w: float(np.sqrt(np.mean(arr))) for w, arr in by_w.items()}


def per_weight_rms_true(supports, true):
    by_w = defaultdict(list)
    for j, S in enumerate(supports):
        by_w[len(S)].append(true[j] ** 2)
    return {w: float(np.sqrt(np.mean(arr))) for w, arr in by_w.items()}
