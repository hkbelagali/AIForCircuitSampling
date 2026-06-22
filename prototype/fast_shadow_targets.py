"""Vectorized GPU implementation of shadow_targets_full.

Replaces a Python-for-loop over n_paulis (~3811 iterations of per-Pauli
numpy ops) with a single torch tensor op vectorized across Paulis,
batched across shots to control memory.

Padding scheme: every Pauli's support is padded to max_w (=w_max) with
-1 markers. Invalid positions are masked out of the match check and
zero'd out of the parity sum.
"""

import numpy as np
import torch


def build_padded_supports(loss_paulis):
    """Convert variable-length support/code lists to fixed-shape padded arrays."""
    n_p = loss_paulis["n_paulis"]
    weights = loss_paulis["weight"]
    max_w = int(weights.max()) if weights.size > 0 else 0
    sup = np.full((n_p, max(max_w, 1)), -1, dtype=np.int64)
    cod = np.full((n_p, max(max_w, 1)), -1, dtype=np.int64)
    for p in range(n_p):
        w = len(loss_paulis["supports"][p])
        if w > 0:
            sup[p, :w] = loss_paulis["supports"][p]
            cod[p, :w] = loss_paulis["supp_codes"][p]
    return sup, cod, weights, max_w


def shadow_targets_full_fast(loss_paulis, U_pattern, b_out, device="cuda",
                              batch_size=8192):
    """Vectorized equivalent of shadow_targets_full.
    Returns shape-(n_paulis,) float64 numpy array."""
    sup, cod, weights, max_w = build_padded_supports(loss_paulis)
    n_p = loss_paulis["n_paulis"]
    k = U_pattern.shape[0]

    sup_t = torch.from_numpy(sup).to(device)              # (n_p, max_w)
    cod_t = torch.from_numpy(cod).to(device)              # (n_p, max_w)
    valid_mask = (sup_t >= 0)                              # (n_p, max_w)
    safe_sup = sup_t.clamp(min=0)                          # (n_p, max_w)
    flat_idx = safe_sup.flatten()                          # (n_p * max_w,)

    U_t = torch.from_numpy(U_pattern).to(device)           # (k, n)
    b_t = torch.from_numpy(b_out.astype(np.int64)).to(device)

    sum_per_p = torch.zeros(n_p, dtype=torch.float64, device=device)
    for s in range(0, k, batch_size):
        e = min(s + batch_size, k)
        U_chunk = U_t[s:e]                                 # (bs, n)
        b_chunk = b_t[s:e]                                 # (bs, n)

        # Gather: (bs, n_p, max_w)
        U_at_S = U_chunk[:, flat_idx].reshape(-1, n_p, max_w)
        b_at_S = b_chunk[:, flat_idx].reshape(-1, n_p, max_w)

        # Match check: all valid positions must equal code
        # (invalid positions automatically "pass" via | ~valid_mask)
        pos_match = (U_at_S == cod_t[None]) | (~valid_mask[None])
        matches = pos_match.all(dim=-1)                    # (bs, n_p)

        # Parity over valid positions only (mask invalid to 0)
        masked_b = b_at_S * valid_mask[None].long()
        parity = masked_b.sum(dim=-1) & 1                  # (bs, n_p)
        sign = torch.where(parity == 0, 1.0, -1.0).double()

        contrib = matches.double() * sign
        sum_per_p += contrib.sum(dim=0)

    scaling = (3.0 ** torch.from_numpy(weights.astype(np.float64)).to(device)) / k
    out = (sum_per_p * scaling).cpu().numpy()
    out[0] = 1.0
    return out


# Sanity check: compare to the existing slow implementation.
def _verify():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))
    from random_shadows import build_loss_paulis_full, shadow_targets_full
    import time
    n, w_max, k = 8, 4, 5000
    loss_paulis = build_loss_paulis_full(n, w_max)
    rng = np.random.default_rng(0)
    U = rng.integers(1, 4, size=(k, n), dtype=np.int64)
    b = rng.integers(0, 2, size=(k, n), dtype=np.int64)

    t0 = time.time(); slow = shadow_targets_full(loss_paulis, U, b); t_slow = time.time() - t0
    t0 = time.time(); fast = shadow_targets_full_fast(loss_paulis, U, b); t_fast = time.time() - t0
    diff = np.abs(slow - fast).max()
    print(f"n={n}, w_max={w_max}, k={k}, n_paulis={loss_paulis['n_paulis']}")
    print(f"  slow: {t_slow:.2f}s")
    print(f"  fast: {t_fast:.2f}s")
    print(f"  max |slow - fast| = {diff:.3e}")
    print(f"  speedup = {t_slow / t_fast:.1f}x")
    assert diff < 1e-10, "Mismatch!"


if __name__ == "__main__":
    _verify()
