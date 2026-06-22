"""Peaked-circuit construction (Zhang et al., arXiv:2404.14493) — PyTorch
re-implementation that matches the paper's qmps-canonical / left-canonical
brickwall pipeline exactly.

Construction:
 - RQC: depth_rqc layers, each a sweep of nearest-neighbor 4x4 Haar gates
   on alternating even/odd pairs (brickwall). All gates fixed (random Haar).
 - PQC: depth_pqc layers, same brickwall structure, but each 4x4 gate is
   trainable. Parameterized as a complex 4x4 matrix passed through a polar
   projection to U(4) at every forward pass.
 - Objective: maximize |<U_pqc 0^n | U_rqc 0^n>|^2 = |<0^n | U_pqc^† U_rqc | 0^n>|^2.

The peaked physical circuit U_pqc^† U_rqc has a peak on |0^n> with weight
= the optimized objective. We materialize the dense statevector at small
n and return p_C.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "peaked_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(n, depth_rqc, depth_pqc, seed):
    return f"n{n}_dR{depth_rqc}_dP{depth_pqc}_s{seed}_pt"


def _haar_unitary(rng, dim=4):
    """Haar-random unitary via QR of complex Gaussian."""
    A = (rng.standard_normal((dim, dim)) +
         1j * rng.standard_normal((dim, dim))) / np.sqrt(2)
    Q, R = np.linalg.qr(A)
    # Fix phases: ensure diag(R) is real positive (gives uniform Haar)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph[None, :]


def _brickwall_pairs(n, depth):
    """List of (layer_idx, qubit_a, qubit_b) for a brickwall of `depth` layers.
    Even layers: pairs (0,1), (2,3), ...; odd layers: (1,2), (3,4), ..."""
    pairs = []
    for d in range(depth):
        start = 0 if d % 2 == 0 else 1
        for i in range(start, n - 1, 2):
            pairs.append((d, i, i + 1))
    return pairs


def _apply_2q(psi_t, G, qa, qb):
    """Apply 4x4 gate G to a torch state of shape (2,)*n. Qubit-0-first axes."""
    G4 = G.reshape(2, 2, 2, 2)  # (qa_out, qb_out, qa_in, qb_in)
    psi_perm = torch.movedim(psi_t, (qa, qb), (0, 1))
    out = torch.einsum("ijab,ab...->ij...", G4, psi_perm)
    return torch.movedim(out, (0, 1), (qa, qb))


def _apply_brickwall(psi_t, gates_list, pairs):
    for g_idx, (d, qa, qb) in enumerate(pairs):
        psi_t = _apply_2q(psi_t, gates_list[g_idx], qa, qb)
    return psi_t


def _polar_unitary(A):
    """Closest-unitary projection of 4x4 complex A via SVD (polar decomp)."""
    U, _, Vh = torch.linalg.svd(A, full_matrices=False)
    return U @ Vh


def build_peaked_pC(n, depth_rqc=None, depth_pqc=None, seed=0,
                    n_iters=2000, lr=0.05, device="cpu",
                    verbose=True, recompute=False):
    """Build the peaked circuit and return p_C as a dense numpy array."""
    depth_rqc = n if depth_rqc is None else depth_rqc
    depth_pqc = n // 2 if depth_pqc is None else depth_pqc
    key = _cache_key(n, depth_rqc, depth_pqc, seed)
    cache_path = CACHE_DIR / f"{key}.pkl"
    if cache_path.exists() and not recompute:
        if verbose:
            print(f"  loaded peaked p_C from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if verbose:
        print(f"  building peaked circuit: n={n}, depth_rqc={depth_rqc}, "
              f"depth_pqc={depth_pqc}, seed={seed}", flush=True)
    rng = np.random.default_rng(seed)

    rqc_pairs = _brickwall_pairs(n, depth_rqc)
    pqc_pairs = _brickwall_pairs(n, depth_pqc)

    # RQC: fixed Haar gates
    rqc_gates = [torch.from_numpy(_haar_unitary(rng).astype(np.complex128))
                 .to(device) for _ in rqc_pairs]

    # PQC: trainable complex 4x4 parameters, projected to U(4) each step.
    # Initialize as small perturbation of identity (paper does identity init).
    pqc_params = []
    for _ in pqc_pairs:
        init = (np.eye(4) +
                0.01 * (rng.standard_normal((4, 4)) +
                        1j * rng.standard_normal((4, 4))))
        p = torch.tensor(init, dtype=torch.complex128, device=device,
                          requires_grad=True)
        pqc_params.append(p)

    # |0^n> as a (2,)*n state
    psi0_flat = torch.zeros(1 << n, dtype=torch.complex128, device=device)
    psi0_flat[0] = 1.0
    psi0 = psi0_flat.reshape((2,) * n)

    # Precompute U_rqc|0> (fixed during optimization)
    with torch.no_grad():
        rqc_state = psi0.clone()
        rqc_state = _apply_brickwall(rqc_state, rqc_gates, rqc_pairs)
    rqc_state = rqc_state.detach()

    opt = torch.optim.Adam(pqc_params, lr=lr)
    t0 = time.time()
    best_loss = float("inf")
    for it in range(n_iters):
        pqc_gates = [_polar_unitary(p) for p in pqc_params]
        pqc_state = psi0.clone()
        pqc_state = _apply_brickwall(pqc_state, pqc_gates, pqc_pairs)
        overlap = torch.vdot(pqc_state.reshape(-1), rqc_state.reshape(-1))
        loss = -(overlap.real ** 2 + overlap.imag ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (it % 50 == 0 or it == n_iters - 1):
            print(f"  it {it:>4}: |<pqc|rqc>|^2 = {-float(loss):.4f}", flush=True)
        if float(loss) < best_loss:
            best_loss = float(loss)
    elapsed = time.time() - t0
    overlap2 = -best_loss

    # Build peaked state |Psi> = U_pqc^dagger U_rqc |0^n>
    with torch.no_grad():
        pqc_gates = [_polar_unitary(p) for p in pqc_params]
        psi = rqc_state.clone()
        for g_idx, (d, qa, qb) in reversed(list(enumerate(pqc_pairs))):
            psi = _apply_2q(psi, pqc_gates[g_idx].conj().T, qa, qb)
        psi_peaked = psi.reshape(-1).cpu().numpy()
    psi_peaked = psi_peaked.astype(np.complex128)
    psi_peaked /= np.linalg.norm(psi_peaked) or 1.0
    p_C = np.abs(psi_peaked) ** 2
    p_C /= p_C.sum() or 1.0
    peak_idx = int(np.argmax(p_C))
    peak_prob = float(p_C[peak_idx])

    if verbose:
        print(f"  done in {elapsed:.1f}s, |<rqc|pqc>|^2 = {overlap2:.4f}, "
              f"peak prob at |0^n> = {p_C[0]:.4f}, global peak = {peak_prob:.4f}"
              f" (idx={peak_idx})", flush=True)

    out = {
        "n": n, "depth_rqc": depth_rqc, "depth_pqc": depth_pqc, "seed": seed,
        "p_C": p_C, "peak_idx": peak_idx, "peak_prob": peak_prob,
        "p_at_zero": float(p_C[0]),
        "overlap_sq": overlap2, "elapsed_sec": elapsed,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


def sample_from_pC(p_C, k, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(len(p_C), size=k, p=p_C).astype(np.int64)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = build_peaked_pC(n=8, depth_rqc=8, depth_pqc=4, seed=0,
                           n_iters=2000, lr=0.05, device=device,
                           recompute=True, verbose=True)
    p_C = out["p_C"]
    D = len(p_C)
    print(f"\np_C sum = {p_C.sum():.6f}")
    print(f"p_C[0] (peak target) = {p_C[0]:.4f}")
    print(f"global peak: idx={out['peak_idx']} ({out['peak_idx']:08b}), prob={out['peak_prob']:.4f}")
    print(f"uniform = 1/D = {1/D:.4f}, so peak is {out['peak_prob'] * D:.1f}x uniform")
    print(f"top-5 probs: {sorted(p_C, reverse=True)[:5]}")
    print(f"ideal XEB on samples from p_C: {D * (p_C**2).sum() - 1:.4f}")
    print(f"H(p_C) = {-float((p_C[p_C>0] * np.log(p_C[p_C>0])).sum()):.4f}  "
          f"(uniform: {np.log(D):.4f})")
