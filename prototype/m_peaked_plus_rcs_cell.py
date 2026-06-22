"""CLI driver: runs (k, seed) sweep for ONE d_append value.

Usage: python m_peaked_plus_rcs_cell.py --d_append <int>

Each invocation:
  1. Builds the base peaked circuit (n=8, d_R=8, d_P=4, peak_seed=0).
  2. Appends d_append RCS layers (brickwall, Haar 4x4 gates, fixed append_seed).
  3. For each (k_train, seed) cell: trains AR-RNN via NLL, computes F_cl.
  4. Saves per-cell JSON.

Designed for SLURM array submission — one array task per d_append value.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import numpy as np
import torch

from peaked import (
    _apply_2q, _brickwall_pairs, _haar_unitary, _polar_unitary,
    build_peaked_pC,
)
from rcs import (
    BitstringARRNN, classical_fidelity, kl_divergence, model_full_distribution,
    train_nll, tv_distance,
)
from aics.circuits.exact import int_to_bits


def _build_rqc_gates(n, depth, seed):
    rng = np.random.default_rng(seed)
    pairs = _brickwall_pairs(n, depth)
    return [torch.from_numpy(_haar_unitary(rng).astype(np.complex128))
            for _ in pairs], pairs


def _apply_layers(psi_t, gates, pairs):
    for g_idx, (_, qa, qb) in enumerate(pairs):
        psi_t = _apply_2q(psi_t, gates[g_idx], qa, qb)
    return psi_t


def build_peaked_plus_rcs_state(n, d_R, d_P, peak_seed, d_append, append_seed,
                                  device="cpu"):
    """Compute peaked + appended-RCS dense state. Uses cached peaked
    optimization where possible."""
    # Get base peaked p_C (cached) — but we need the full state vector,
    # not just p_C. Easiest: rerun (cheap, ~25s on cuda) and recover.
    out = build_peaked_pC(n=n, depth_rqc=d_R, depth_pqc=d_P, seed=peak_seed,
                           device=device, verbose=False)
    p_C_peaked = out["p_C"]
    # Reconstruct dense statevector — but signs are arbitrary in peaked
    # (real, since we used real polar projection). For F_cl we only need
    # |psi|^2 = p_C, so use sqrt(p_C) as the state going into RCS.
    # NOTE: phases of psi_peaked matter for the appended RCS to produce
    # the right output distribution! Reconstruct from full forward.
    return p_C_peaked  # placeholder — see below


def build_combined_pC(n, d_R, d_P, peak_seed, d_append, append_seed,
                      device="cpu", verbose=False):
    """Compute p_C of U_rcs^{d_append} U_pqc^dagger U_rqc |0^n>.
    Reconstructs the state from scratch (peaked optimization + apply gates).
    """
    # Step 1: build & optimize peaked
    if verbose: print(f"  building peaked: n={n}, d_R={d_R}, d_P={d_P}, seed={peak_seed}",
                       flush=True)
    rqc_pairs = _brickwall_pairs(n, d_R)
    pqc_pairs = _brickwall_pairs(n, d_P)
    rng = np.random.default_rng(peak_seed)
    rqc_gates = [torch.from_numpy(_haar_unitary(rng).astype(np.complex128)).to(device)
                  for _ in rqc_pairs]
    # PQC: optimize
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
        rqc_state = _apply_layers(psi0.clone(), rqc_gates, rqc_pairs).detach()
    opt = torch.optim.Adam(pqc_params, lr=0.05)
    for it in range(2000):
        pqc_g = [_polar_unitary(p) for p in pqc_params]
        pqc_state = _apply_layers(psi0.clone(), pqc_g, pqc_pairs)
        overlap = torch.vdot(pqc_state.reshape(-1), rqc_state.reshape(-1))
        loss = -(overlap.real ** 2 + overlap.imag ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    if verbose: print(f"  peaked done, |overlap|^2 = {-float(loss):.4f}", flush=True)

    # Step 2: build appended RCS gates (independent seed)
    append_rng = np.random.default_rng(append_seed + 10**7)
    append_pairs = _brickwall_pairs(n, d_append) if d_append > 0 else []
    append_gates = [torch.from_numpy(_haar_unitary(append_rng).astype(np.complex128)).to(device)
                     for _ in append_pairs]

    # Step 3: construct the combined output state
    #   |Psi> = U_rcs U_pqc^dagger U_rqc |0>
    with torch.no_grad():
        pqc_g_opt = [_polar_unitary(p) for p in pqc_params]
        # Start from U_rqc|0>
        psi = rqc_state.clone()
        # Apply U_pqc^dagger (gates in reverse, each daggered)
        for g_idx, (_, qa, qb) in reversed(list(enumerate(pqc_pairs))):
            psi = _apply_2q(psi, pqc_g_opt[g_idx].conj().T, qa, qb)
        # Apply appended RCS forward
        psi = _apply_layers(psi, append_gates, append_pairs)
    psi_vec = psi.reshape(-1).cpu().numpy().astype(np.complex128)
    psi_vec /= np.linalg.norm(psi_vec) or 1.0
    p_C = (np.abs(psi_vec) ** 2).astype(np.float64)
    p_C /= p_C.sum() or 1.0
    return p_C, int(np.argmax(p_C)), float(p_C.max())


def run_cell(n, p_C, k_train, seed, d_hidden=64, epochs=400, lr=2e-3,
             device="cpu"):
    """Single (k, seed) cell — sample, train, compute F_cl."""
    dim = len(p_C)
    rng = np.random.default_rng(seed + 100000)
    train_int = rng.choice(dim, size=k_train, p=p_C).astype(np.int64)
    X_train = torch.from_numpy(int_to_bits(train_int, n)).long().to(device)
    torch.manual_seed(seed)
    model = BitstringARRNN(n_qubits=n, d_hidden=d_hidden).to(device)
    t0 = time.time()
    train_nll(model, X_train, epochs=epochs, lr=lr,
              batch_size=k_train, verbose=False)
    elapsed = time.time() - t0
    p_model = model_full_distribution(model, n, device)
    return {
        "k_train": k_train, "seed": seed, "elapsed_sec": elapsed,
        "F_cl": classical_fidelity(p_model, p_C),
        "TV": tv_distance(p_model, p_C),
        "kl": kl_divergence(p_model, p_C),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_append", type=int, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--d_R", type=int, default=8)
    parser.add_argument("--d_P", type=int, default=4)
    parser.add_argument("--peak_seed", type=int, default=0)
    parser.add_argument("--append_seed", type=int, default=0)
    args = parser.parse_args()

    ks = [16, 32, 64, 128, 256, 1024, 10000]
    seeds = list(range(16))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = (Path(__file__).resolve().parents[1] / "results"
               / f"m_peaked_plus_rcs_n{args.n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"d_append={args.d_append}  n={args.n}  device={device}", flush=True)

    t0 = time.time()
    p_C, peak_idx, peak_prob = build_combined_pC(
        n=args.n, d_R=args.d_R, d_P=args.d_P,
        peak_seed=args.peak_seed, d_append=args.d_append,
        append_seed=args.append_seed, device=device, verbose=True)
    D = len(p_C)
    ideal_xeb = D * float((p_C ** 2).sum()) - 1.0
    print(f"  d_append={args.d_append}: peak_idx={peak_idx}, peak_prob={peak_prob:.4f}, "
          f"ideal XEB={ideal_xeb:.4f}", flush=True)

    meta = {
        "n": args.n, "d_R": args.d_R, "d_P": args.d_P,
        "peak_seed": args.peak_seed, "append_seed": args.append_seed,
        "d_append": args.d_append, "ideal_xeb": ideal_xeb,
        "peak_prob": peak_prob, "peak_idx": peak_idx,
    }

    for k in ks:
        for seed in seeds:
            tag = f"da{args.d_append}_k{k}_s{seed}"
            cell_path = out_dir / f"{tag}.json"
            if cell_path.exists():
                continue
            out = run_cell(args.n, p_C, k_train=k, seed=seed, device=device)
            out.update(meta)
            cell_path.write_text(json.dumps(out))
        # row summary
        row_files = list(out_dir.glob(f"da{args.d_append}_k{k}_s*.json"))
        fcls = [json.loads(f.read_text())["F_cl"] for f in row_files]
        print(f"  k={k:>5}: F_cl med={np.median(fcls):.4f} "
              f"({len(fcls)} seeds)", flush=True)
    print(f"total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
