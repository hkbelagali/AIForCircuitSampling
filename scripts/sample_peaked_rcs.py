"""Stage A (peaked+RCS variant): build a Zhang-et-al. peaked circuit followed
by ``d_append`` random-Haar brickwall layers, then draw a sample bundle in
the standard aics format so downstream trainers (e.g. n_dynamics_snapshot)
can consume it just like an RCS bundle.

Circuit construction
--------------------
    U = R_{d_append}  ...  R_1  U_pqc^dagger  U_rqc

where U_rqc is a fixed random brickwall (depth ``depth_rqc``), U_pqc is a
brickwall (depth ``depth_pqc``) of 4x4 gates optimized to maximize
|<pqc|rqc>|^2 (peaked-circuit objective), and R_i are ``d_append`` extra
Haar brickwall layers.  ``d_append = 0`` is a fully peaked circuit;
``d_append -> infty`` is essentially RCS. Follows
archive/prototype-v1/m_peaked_plus_rcs_cell.build_combined_pC exactly.

Output
------
``results/tn_samples_peaked_rcs/peaked_n{n}_dR{dR}_dP{dP}_ps{peak_seed}_da{d_append}_as{append_seed}_k{k_max}.npz``

Same keys as standard RCS bundles (train_bits/pC, held_bits/pC,
uniform_bits/pC, meta), PLUS ``p_C_full`` (2^n float64) since the
peaked circuit can't be cheaply re-derived from meta alone. Downstream
scripts that recompute p_C from a boixo/sycamore builder should be
patched to short-circuit on ``p_C_full``.

Usage
-----
    python scripts/sample_peaked_rcs.py --n 12 --d_append 4
    python scripts/sample_peaked_rcs.py --n 16 --d_append 0 --k_max 100000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Import helpers from archive peaked.py (safe: only numpy + torch deps).
_ARCHIVE = Path(__file__).resolve().parents[1] / "archive" / "prototype-v1"
if str(_ARCHIVE) not in sys.path:
    sys.path.insert(0, str(_ARCHIVE))

from peaked import (  # noqa: E402
    _apply_2q, _brickwall_pairs, _haar_unitary, _polar_unitary,
)

from aics.io._repro import provenance  # noqa: E402
from aics.io.conventions import int_to_bits  # noqa: E402


def build_combined_pC(n, depth_rqc, depth_pqc, peak_seed, d_append,
                       append_seed, device="cpu", n_iters=2000, lr=0.05,
                       verbose=True):
    """Return (p_C: (2^n,) float64, peak_idx: int, peak_prob: float, aux).

    Reimplementation of archive/prototype-v1/m_peaked_plus_rcs_cell.build_combined_pC
    so the current script has no dependency on archive rcs.py / archive aics.
    """
    if verbose:
        print(f"  building peaked: n={n} d_R={depth_rqc} d_P={depth_pqc} "
              f"peak_seed={peak_seed}", flush=True)

    rqc_pairs = _brickwall_pairs(n, depth_rqc)
    pqc_pairs = _brickwall_pairs(n, depth_pqc)

    rng = np.random.default_rng(peak_seed)
    rqc_gates = [torch.from_numpy(_haar_unitary(rng).astype(np.complex128)).to(device)
                  for _ in rqc_pairs]

    # PQC: complex 4x4 params projected to U(4) each forward pass.
    pqc_params = []
    for _ in pqc_pairs:
        init = (np.eye(4) + 0.01 *
                 (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))))
        p = torch.tensor(init, dtype=torch.complex128, device=device,
                          requires_grad=True)
        pqc_params.append(p)

    psi0 = torch.zeros((2,) * n, dtype=torch.complex128, device=device)
    psi0.reshape(-1)[0] = 1.0

    with torch.no_grad():
        rqc_state = psi0.clone()
        for g_idx, (_, qa, qb) in enumerate(rqc_pairs):
            rqc_state = _apply_2q(rqc_state, rqc_gates[g_idx], qa, qb)
        rqc_state = rqc_state.detach()

    opt = torch.optim.Adam(pqc_params, lr=lr)
    t0 = time.time()
    best_loss = float("inf")
    for it in range(n_iters):
        pqc_gates = [_polar_unitary(p) for p in pqc_params]
        pqc_state = psi0.clone()
        for g_idx, (_, qa, qb) in enumerate(pqc_pairs):
            pqc_state = _apply_2q(pqc_state, pqc_gates[g_idx], qa, qb)
        overlap = torch.vdot(pqc_state.reshape(-1), rqc_state.reshape(-1))
        loss = -(overlap.real ** 2 + overlap.imag ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_val = loss.detach().item()
        best_loss = min(best_loss, loss_val)
        if verbose and (it % 200 == 0 or it == n_iters - 1):
            print(f"    peaked it {it:>4}: |<pqc|rqc>|^2 = {-loss_val:.4f}",
                  flush=True)
    overlap_sq = -best_loss
    if verbose:
        print(f"  peaked done in {time.time() - t0:.1f}s "
              f"|overlap|^2 = {overlap_sq:.4f}", flush=True)

    # Appended RCS layers with an independent seed.
    append_rng = np.random.default_rng(append_seed + 10 ** 7)
    append_pairs = _brickwall_pairs(n, d_append) if d_append > 0 else []
    append_gates = [torch.from_numpy(_haar_unitary(append_rng).astype(np.complex128)).to(device)
                     for _ in append_pairs]

    # |Psi> = R_{d_append} ... R_1 U_pqc^dag U_rqc |0>.
    with torch.no_grad():
        pqc_gates = [_polar_unitary(p) for p in pqc_params]
        psi = rqc_state.clone()
        for g_idx, (_, qa, qb) in reversed(list(enumerate(pqc_pairs))):
            psi = _apply_2q(psi, pqc_gates[g_idx].conj().T, qa, qb)
        for g_idx, (_, qa, qb) in enumerate(append_pairs):
            psi = _apply_2q(psi, append_gates[g_idx], qa, qb)
    psi_vec = psi.reshape(-1).cpu().numpy().astype(np.complex128)
    psi_vec /= np.linalg.norm(psi_vec) or 1.0
    p_C = (np.abs(psi_vec) ** 2).astype(np.float64)
    p_C /= p_C.sum() or 1.0
    peak_idx = int(np.argmax(p_C))
    peak_prob = float(p_C[peak_idx])
    aux = {
        "overlap_sq": overlap_sq,
        "peak_prob": peak_prob,
        "peak_idx": peak_idx,
        "p_at_zero": float(p_C[0]),
    }
    return p_C, peak_idx, peak_prob, aux


def draw_bits(p_C, k, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p_C), size=k, p=p_C).astype(np.int64)
    bits = int_to_bits(idx, n).astype(np.uint8)
    pC = p_C[idx].astype(np.float64)
    return bits, pC


def draw_uniform(k, n, seed):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
    return bits


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--depth_rqc", type=int, default=8,
                    help="d_R: number of RQC (fixed Haar) brickwall layers")
    p.add_argument("--depth_pqc", type=int, default=4,
                    help="d_P: number of PQC (trainable) brickwall layers")
    p.add_argument("--peak_seed", type=int, default=0)
    p.add_argument("--d_append", type=int, required=True,
                    help="number of Haar brickwall layers appended AFTER "
                         "the peaked block (0 = fully peaked)")
    p.add_argument("--append_seed", type=int, default=0)
    p.add_argument("--k_max", type=int, default=100_000)
    p.add_argument("--k_held", type=int, default=10_000)
    p.add_argument("--k_uniform", type=int, default=2_000)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--n_iters", type=int, default=2000,
                    help="peaked-optimization iterations")
    p.add_argument("--lr", type=float, default=0.05,
                    help="peaked-optimization learning rate")
    p.add_argument("--device", type=str, default=None,
                    help="'cuda' | 'cpu' | 'cuda:N'; default auto")
    p.add_argument("--out_dir", type=str,
                    default="results/tn_samples_peaked_rcs")
    args = p.parse_args()

    if args.n > 20:
        raise SystemExit(
            f"--n={args.n}: exact-state peaked+RCS is memory-limited to n<=20 "
            "(2^n complex128 statevector).")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[sample_peaked_rcs] n={args.n} d_R={args.depth_rqc} "
          f"d_P={args.depth_pqc} d_append={args.d_append} "
          f"peak_seed={args.peak_seed} append_seed={args.append_seed} "
          f"device={device}", flush=True)

    t0 = time.time()
    p_C, peak_idx, peak_prob, aux = build_combined_pC(
        n=args.n, depth_rqc=args.depth_rqc, depth_pqc=args.depth_pqc,
        peak_seed=args.peak_seed, d_append=args.d_append,
        append_seed=args.append_seed, device=device,
        n_iters=args.n_iters, lr=args.lr, verbose=True,
    )
    D = len(p_C)
    ideal_xeb = float(D * (p_C ** 2).sum() - 1.0)
    H = float(-(p_C[p_C > 0] * np.log(p_C[p_C > 0])).sum())
    print(f"  p_C ready: D={D} peak_idx={peak_idx} peak_prob={peak_prob:.4f} "
          f"ideal_xeb={ideal_xeb:.4f} H={H:.4f} (uniform log D = {np.log(D):.4f})",
          flush=True)

    # Draw bundles.
    t0d = time.time()
    train_bits, train_pC = draw_bits(
        p_C, args.k_max, args.n, seed=args.sample_seed + 100_000)
    print(f"  drew {args.k_max} train in {time.time() - t0d:.1f}s  "
          f"XEB(train)={D * train_pC.mean() - 1:+.4f}", flush=True)

    held_bits = held_pC = uniform_bits = uniform_pC = None
    if args.k_held > 0:
        held_bits, held_pC = draw_bits(
            p_C, args.k_held, args.n, seed=args.sample_seed + 999_991)
        print(f"  held {args.k_held}  XEB(held)={D * held_pC.mean() - 1:+.4f}",
              flush=True)
    if args.k_uniform > 0:
        uniform_bits = draw_uniform(args.k_uniform, args.n,
                                      seed=args.sample_seed + 777)
        # For a uniform draw, look up the true p_C via bit->int.
        from aics.io.conventions import bits_to_int
        u_int = bits_to_int(uniform_bits)
        uniform_pC = p_C[u_int].astype(np.float64)
        print(f"  uniform {args.k_uniform}  "
              f"XEB(uniform)={D * uniform_pC.mean() - 1:+.4f}", flush=True)

    meta = {
        "n": args.n,
        "family": "peaked_plus_rcs",
        "depth_rqc": args.depth_rqc,
        "depth_pqc": args.depth_pqc,
        "peak_seed": args.peak_seed,
        "d_append": args.d_append,
        "append_seed": args.append_seed,
        "sample_seed": args.sample_seed,
        "k_max": args.k_max,
        "k_held": args.k_held,
        "k_uniform": args.k_uniform,
        "n_iters": args.n_iters,
        "lr": args.lr,
        "ideal_xeb": ideal_xeb,
        "peak_prob": peak_prob,
        "peak_idx": peak_idx,
        "overlap_sq": aux["overlap_sq"],
        "p_at_zero": aux["p_at_zero"],
        # Compatibility keys that downstream code may look up.
        "depth": args.depth_rqc + args.depth_pqc + args.d_append,
        "circuit": "peaked_plus_rcs",
        "circuit_seed": args.peak_seed,
        "provenance": provenance(),
    }

    tag = (f"peaked_n{args.n}_dR{args.depth_rqc}_dP{args.depth_pqc}"
           f"_ps{args.peak_seed}_da{args.d_append}_as{args.append_seed}"
           f"_k{args.k_max}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.npz"

    save = {
        "train_bits": np.ascontiguousarray(train_bits, dtype=np.uint8),
        "train_pC": np.ascontiguousarray(train_pC, dtype=np.float64),
        "p_C_full": np.ascontiguousarray(p_C, dtype=np.float64),
        "meta": json.dumps(meta),
    }
    if held_bits is not None:
        save["held_bits"] = np.ascontiguousarray(held_bits, dtype=np.uint8)
        save["held_pC"] = np.ascontiguousarray(held_pC, dtype=np.float64)
    if uniform_bits is not None:
        save["uniform_bits"] = np.ascontiguousarray(uniform_bits, dtype=np.uint8)
        save["uniform_pC"] = np.ascontiguousarray(uniform_pC, dtype=np.float64)
    np.savez(out_path, **save)
    print(f"  wrote {out_path}  (total {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
