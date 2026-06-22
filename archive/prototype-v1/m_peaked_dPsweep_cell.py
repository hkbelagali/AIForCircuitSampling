"""Sweep over PQC depth at fixed RQC depth (no appended RCS). One d_P
per SLURM array task. Output files: dP{d_P}_k{k}_s{s}.json."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import numpy as np
import torch

from peaked import _apply_2q, _brickwall_pairs, _haar_unitary, _polar_unitary
from rcs import (
    BitstringARRNN, classical_fidelity, kl_divergence, model_full_distribution,
    train_nll, tv_distance,
)
from aics.circuits.exact import int_to_bits


def _apply_layers(psi_t, gates, pairs):
    for g_idx, (_, qa, qb) in enumerate(pairs):
        psi_t = _apply_2q(psi_t, gates[g_idx], qa, qb)
    return psi_t


def build_peaked_dP(n, d_R, d_P, peak_seed, device="cpu", n_iters=2000, lr=0.05):
    """Build peaked state U_pqc^dagger U_rqc |0^n>; allows d_P=0 (= just U_rqc)."""
    rqc_pairs = _brickwall_pairs(n, d_R)
    pqc_pairs = _brickwall_pairs(n, d_P)
    rng = np.random.default_rng(peak_seed)
    rqc_gates = [torch.from_numpy(_haar_unitary(rng).astype(np.complex128)).to(device)
                  for _ in rqc_pairs]
    psi0 = torch.zeros((2,) * n, dtype=torch.complex128, device=device)
    psi0.reshape(-1)[0] = 1.0
    with torch.no_grad():
        rqc_state = _apply_layers(psi0.clone(), rqc_gates, rqc_pairs).detach()

    overlap_sq = float("nan")
    if d_P == 0:
        # No PQC — peaked output IS U_rqc|0> = rqc_state
        psi = rqc_state.clone()
    else:
        # Optimize PQC params
        pqc_params = []
        for _ in pqc_pairs:
            init = (np.eye(4) + 0.01 *
                    (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))))
            p = torch.tensor(init, dtype=torch.complex128, device=device,
                              requires_grad=True)
            pqc_params.append(p)
        opt = torch.optim.Adam(pqc_params, lr=lr)
        for it in range(n_iters):
            pqc_g = [_polar_unitary(p) for p in pqc_params]
            pqc_state = _apply_layers(psi0.clone(), pqc_g, pqc_pairs)
            overlap = torch.vdot(pqc_state.reshape(-1), rqc_state.reshape(-1))
            loss = -(overlap.real ** 2 + overlap.imag ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        overlap_sq = -float(loss)
        # Apply U_pqc^dagger to rqc_state
        with torch.no_grad():
            pqc_g_opt = [_polar_unitary(p) for p in pqc_params]
            psi = rqc_state.clone()
            for g_idx, (_, qa, qb) in reversed(list(enumerate(pqc_pairs))):
                psi = _apply_2q(psi, pqc_g_opt[g_idx].conj().T, qa, qb)

    psi_vec = psi.reshape(-1).cpu().numpy().astype(np.complex128)
    psi_vec /= np.linalg.norm(psi_vec) or 1.0
    p_C = (np.abs(psi_vec) ** 2).astype(np.float64)
    p_C /= p_C.sum() or 1.0
    return p_C, int(np.argmax(p_C)), float(p_C.max()), overlap_sq


def run_cell(n, p_C, k_train, seed, d_hidden=64, epochs=400, lr=2e-3,
             device="cpu"):
    dim = len(p_C)
    rng = np.random.default_rng(seed + 100000)
    train_int = rng.choice(dim, size=k_train, p=p_C).astype(np.int64)
    X_train = torch.from_numpy(int_to_bits(train_int, n)).long().to(device)
    torch.manual_seed(seed)
    model = BitstringARRNN(n_qubits=n, d_hidden=d_hidden).to(device)
    t0 = time.time()
    train_nll(model, X_train, epochs=epochs, lr=lr, batch_size=k_train,
              verbose=False)
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
    parser.add_argument("--d_P", type=int, required=True)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--d_R", type=int, default=12)
    parser.add_argument("--peak_seed", type=int, default=0)
    args = parser.parse_args()

    ks = [16, 32, 64, 128, 256, 1024, 10000]
    seeds = list(range(16))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = (Path(__file__).resolve().parents[1] / "results"
                / f"m_peaked_dPsweep_n{args.n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"d_P={args.d_P}  n={args.n}  d_R={args.d_R}  device={device}",
          flush=True)

    t0 = time.time()
    p_C, peak_idx, peak_prob, overlap_sq = build_peaked_dP(
        n=args.n, d_R=args.d_R, d_P=args.d_P,
        peak_seed=args.peak_seed, device=device)
    D = len(p_C)
    ideal_xeb = D * float((p_C ** 2).sum()) - 1.0
    print(f"  d_P={args.d_P}: peak_idx={peak_idx}, peak_prob={peak_prob:.4f}, "
          f"overlap_sq={overlap_sq:.4f}, ideal XEB={ideal_xeb:.4f}", flush=True)

    meta = {
        "n": args.n, "d_R": args.d_R, "d_P": args.d_P,
        "peak_seed": args.peak_seed, "ideal_xeb": ideal_xeb,
        "peak_prob": peak_prob, "peak_idx": peak_idx, "overlap_sq": overlap_sq,
    }

    for k in ks:
        for seed in seeds:
            tag = f"dP{args.d_P}_k{k}_s{seed}"
            cell_path = out_dir / f"{tag}.json"
            if cell_path.exists():
                continue
            out = run_cell(args.n, p_C, k_train=k, seed=seed, device=device)
            out.update(meta)
            cell_path.write_text(json.dumps(out))
        row_files = list(out_dir.glob(f"dP{args.d_P}_k{k}_s*.json"))
        fcls = [json.loads(f.read_text())["F_cl"] for f in row_files]
        print(f"  k={k:>5}: F_cl med={np.median(fcls):.4f} "
              f"({len(fcls)} seeds)", flush=True)
    print(f"total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
