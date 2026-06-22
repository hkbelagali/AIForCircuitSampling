"""NLL-trained RCS + per-weight Z-Pauli error evaluation.

Ports Ryan's notebook pipeline (rcs_ml_experiment.ipynb):
  - 2-layer LSTM AutoregressiveRNN, hidden=128
  - NLL training on Born samples (z-basis only)
  - Porter-Thomas entropy regularizer (lambda=0.01)
  - Fixed gradient-step budget (50k steps) with min/max-epoch caps
  - Sycamore-style brickwork circuit (CZ_DEPTH=10, fixed circuit seed)

After training, evaluates <Z_S>_theta vs <Z_S>_p_C for all Z-Pauli supports
of weight |S| <= w_eval_max, grouping errors by weight. Matches the
schema of m_rcs_z_pauli (err_by_weight_model/_shadow/true_rms_by_weight)
so existing comparison plot machinery works.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import exact_probabilities, sample_from_circuit
from rcs import classical_fidelity, tv_distance, kl_divergence, enumerate_z_supports


# Ryan's defaults from rcs_ml_experiment.ipynb cell 2
BATCH_SIZE = 512
TOTAL_STEPS = 50_000
MIN_EPOCHS = 50
MAX_EPOCHS = 5_000
N_TEST_OVERLAP = 10_000
LAMBDA_PT = 0.01


class AutoregressiveRNN(nn.Module):
    """LSTM-based autoregressive model — exact port of Ryan's cell 14."""

    def __init__(self, n_bits, hidden=128, n_layers=2):
        super().__init__()
        self.n_bits = n_bits
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        bsz = x.shape[0]
        sos = torch.zeros(bsz, 1, 1, device=x.device, dtype=x.dtype)
        inp = torch.cat([sos, x[:, :-1].unsqueeze(-1)], dim=1)
        out, _ = self.lstm(inp)
        return self.head(out).squeeze(-1)

    def log_prob(self, x):
        logits = self.forward(x)
        return -nn.functional.binary_cross_entropy_with_logits(
            logits, x, reduction="none").sum(dim=1)

    @torch.no_grad()
    def sample_bits(self, n):
        device = next(self.parameters()).device
        samples = torch.zeros(n, self.n_bits, device=device)
        h = None
        inp = torch.zeros(n, 1, 1, device=device)
        for i in range(self.n_bits):
            out, h = self.lstm(inp, h)
            prob = torch.sigmoid(self.head(out.squeeze(1)))
            bit = torch.bernoulli(prob)
            samples[:, i] = bit.squeeze(1)
            inp = bit.unsqueeze(1)
        return samples.cpu().numpy()


def index_to_bitstring_msb(idx, n_bits):
    """Integer idx -> binary array MSB first (matches Ryan's convention)."""
    return np.array([(idx >> (n_bits - 1 - i)) & 1 for i in range(n_bits)],
                     dtype=np.float32)


def all_bits_msb(n_bits):
    """(2^n, n) float32 array, MSB-first ordering."""
    D = 1 << n_bits
    return np.array([index_to_bitstring_msb(i, n_bits) for i in range(D)],
                     dtype=np.float32)


def bitstrings_to_indices_msb(bits):
    """(N, n_bits) binary -> int indices, MSB first."""
    n_bits = bits.shape[1]
    powers = 2 ** np.arange(n_bits - 1, -1, -1)
    return (bits @ powers).astype(np.int64)


def linear_xeb(samples_idx, p_C):
    return float(len(p_C) * p_C[samples_idx].mean() - 1)


def parity_per_support(samples_bits, supports):
    """Empirical <Z_S> from samples for each support S.
    samples_bits: (k, n) uint8. supports: list of (tuple of qubit indices).
    Returns (n_supports,) float array."""
    out = np.empty(len(supports), dtype=np.float64)
    for j, S in enumerate(supports):
        if len(S) == 0:
            out[j] = 1.0
        else:
            parity = samples_bits[:, list(S)].sum(axis=1) & 1
            out[j] = 1.0 - 2.0 * parity.mean()
    return out


def per_weight_rms_err(supports, pred, true):
    """Group |pred - true|^2 by |S|=w, return RMS error per weight as dict."""
    by_w = defaultdict(list)
    for j, S in enumerate(supports):
        by_w[len(S)].append((pred[j] - true[j]) ** 2)
    return {w: float(np.sqrt(np.mean(arr))) for w, arr in by_w.items()}


def per_weight_rms_true(supports, true):
    by_w = defaultdict(list)
    for j, S in enumerate(supports):
        by_w[len(S)].append(true[j] ** 2)
    return {w: float(np.sqrt(np.mean(arr))) for w, arr in by_w.items()}


@torch.no_grad()
def model_full_distribution(model, n_bits, device, batch=4096):
    """p_theta(x) over all 2^n bitstrings, MSB-first ordering (matches sampling)."""
    D = 1 << n_bits
    all_bits = all_bits_msb(n_bits)
    out = np.empty(D, dtype=np.float64)
    for s in range(0, D, batch):
        x = torch.from_numpy(all_bits[s:s + batch]).to(device)
        lp = model.log_prob(x).cpu().numpy()
        out[s:s + batch] = lp
    out -= out.max()
    out = np.exp(out)
    out /= out.sum()
    return out


def train_nll_pt(model, train_bits, total_steps, min_epochs, max_epochs,
                  batch_size, lr, lambda_pt, n_states, device, verbose=False):
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_bits.astype(np.float32))),
        batch_size=min(batch_size, len(train_bits)), shuffle=True)
    steps_per_epoch = max(1, len(train_bits) // batch_size)
    n_epochs = min(max_epochs, max(min_epochs, total_steps // steps_per_epoch))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    last_nll = float("nan")
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            nll_loss = -model.log_prob(batch).mean()
            if lambda_pt > 0:
                log_q = model.log_prob(batch)
                # Cast to float — Python int(2^n) overflows at large n (e.g. n=70).
                q_scaled = torch.exp(log_q) * float(n_states)
                pt_loss = (q_scaled - log_q).mean()
                loss = nll_loss + lambda_pt * pt_loss
            else:
                loss = nll_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += float(nll_loss.detach())
        scheduler.step()
        last_nll = ep_loss / len(loader)
        if verbose and (ep % 50 == 0 or ep == n_epochs - 1):
            print(f"  ep {ep:>4}/{n_epochs}: NLL = {last_nll:.4f}", flush=True)
    return last_nll, n_epochs


def run_cell(n, depth, circuit_seed, k_train, w_eval_max, seed, *,
              hidden=128, n_layers=2, lambda_pt=LAMBDA_PT,
              total_steps=TOTAL_STEPS, min_epochs=MIN_EPOCHS,
              max_epochs=MAX_EPOCHS, lr=1e-3, device="cpu"):
    rows, cols = grid_for(n)
    qubits, circuit = make_rcs_circuit(rows, cols, depth, seed=circuit_seed)
    p_C = exact_probabilities(circuit, qubits)
    D = len(p_C)
    H_true = float(-(p_C[p_C > 0] * np.log(p_C[p_C > 0])).sum())

    # Sample k_train Born samples (cirq, MSB-first integer indices).
    train_int = sample_from_circuit(circuit, qubits, k_train, seed=seed)
    # MSB-first convention (matches Ryan's index_to_bitstring)
    train_bits = np.unpackbits(
        train_int.astype(">i8").view(np.uint8)).reshape(-1, 64)[:, -n:]

    held_int = sample_from_circuit(circuit, qubits, N_TEST_OVERLAP,
                                     seed=seed + 999983)
    held_bits = np.unpackbits(
        held_int.astype(">i8").view(np.uint8)).reshape(-1, 64)[:, -n:]

    # Ideal / uniform XEB baselines on training samples
    xeb_train = linear_xeb(train_int, p_C)
    uniform_idx = np.random.default_rng(seed + 7).integers(0, D, size=N_TEST_OVERLAP)
    xeb_unif = linear_xeb(uniform_idx, p_C)
    xeb_ideal = linear_xeb(held_int, p_C)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AutoregressiveRNN(n_bits=n, hidden=hidden, n_layers=n_layers).to(device)

    t0 = time.time()
    final_nll, n_epochs = train_nll_pt(
        model, train_bits, total_steps, min_epochs, max_epochs,
        BATCH_SIZE, lr, lambda_pt, D, device, verbose=False)
    train_time = time.time() - t0

    model.eval()
    # Generalisation XEB on held-out (samples from p_C, score with q_theta)
    held_t = torch.from_numpy(held_bits.astype(np.float32)).to(device)
    with torch.no_grad():
        log_q_held = model.log_prob(held_t).cpu().numpy()
    xeb_gen = float(D * np.exp(log_q_held).mean() - 1)

    # Model samples XEB
    model_samples = model.sample_bits(N_TEST_OVERLAP)
    model_idx = bitstrings_to_indices_msb(model_samples)
    xeb_model = linear_xeb(model_idx, p_C)

    # Full distribution metrics (only at small n)
    p_model = model_full_distribution(model, n, device)
    F_cl = classical_fidelity(p_model, p_C)
    tv = tv_distance(p_model, p_C)
    kl = kl_divergence(p_model, p_C)

    # Per-weight Z-Pauli evaluation
    supports, _ = enumerate_z_supports(n, w_eval_max)
    # Truth via full distribution
    all_b = all_bits_msb(n).astype(np.int64)
    true_Z = np.empty(len(supports), dtype=np.float64)
    for j, S in enumerate(supports):
        if len(S) == 0:
            true_Z[j] = 1.0
        else:
            parity = all_b[:, list(S)].sum(axis=1) & 1
            true_Z[j] = float(((1 - 2 * parity) * p_C).sum())
    # Model via full distribution
    model_Z = np.empty(len(supports), dtype=np.float64)
    for j, S in enumerate(supports):
        if len(S) == 0:
            model_Z[j] = 1.0
        else:
            parity = all_b[:, list(S)].sum(axis=1) & 1
            model_Z[j] = float(((1 - 2 * parity) * p_model).sum())
    # Shadow baseline: empirical <Z_S> from train samples
    shadow_Z = parity_per_support(train_bits.astype(np.int64), supports)

    err_by_weight_model = per_weight_rms_err(supports, model_Z, true_Z)
    err_by_weight_shadow = per_weight_rms_err(supports, shadow_Z, true_Z)
    true_rms_by_weight = per_weight_rms_true(supports, true_Z)

    return {
        "n": n, "depth": depth, "circuit_seed": circuit_seed,
        "k_train": k_train, "w_eval_max": w_eval_max, "seed": seed,
        "n_supports": len(supports),
        "hidden": hidden, "n_layers": n_layers,
        "lambda_pt": lambda_pt, "lr": lr,
        "n_epochs": n_epochs, "total_steps": total_steps,
        "final_nll": final_nll,
        "H_true": H_true,
        "xeb_train": xeb_train, "xeb_gen": xeb_gen,
        "xeb_model": xeb_model,
        "xeb_ideal": xeb_ideal, "xeb_unif": xeb_unif,
        "F_cl": F_cl, "TV": tv, "kl": kl,
        "train_time_sec": train_time,
        "err_by_weight_model": err_by_weight_model,
        "err_by_weight_shadow": err_by_weight_shadow,
        "true_rms_by_weight": true_rms_by_weight,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k_train", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--circuit_seed", type=int, default=42,
                         help="Ryan's default. Use 0 to compare with our n=8 Z-Pauli data.")
    parser.add_argument("--w_eval_max", type=int, default=None,
                         help="default: n (full enumeration)")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lambda_pt", type=float, default=LAMBDA_PT)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_subdir", type=str, default="m_rcs_nll_eval")
    args = parser.parse_args()

    if args.w_eval_max is None:
        args.w_eval_max = args.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(__file__).resolve().parents[1] / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = (f"n{args.n}_d{args.depth}_k{args.k_train}_w{args.w_eval_max}"
           f"_cs{args.circuit_seed}_s{args.seed}")
    p = out_dir / f"{tag}.json"
    if p.exists():
        print(f"  cached: {tag}", flush=True)
        return

    print(f"RCS NLL eval: n={args.n} d={args.depth} cs={args.circuit_seed} "
          f"k={args.k_train} seed={args.seed} w_eval_max={args.w_eval_max} "
          f"device={device}", flush=True)
    rec = run_cell(args.n, args.depth, args.circuit_seed,
                    args.k_train, args.w_eval_max, args.seed,
                    hidden=args.hidden, n_layers=args.n_layers,
                    lambda_pt=args.lambda_pt, lr=args.lr, device=device)
    p.write_text(json.dumps(rec))
    print(f"  XEB_gen={rec['xeb_gen']:.4f}  F_cl={rec['F_cl']:.4f}  "
          f"NLL={rec['final_nll']:.3f}  "
          f"err_by_w(model)={ {k: round(v, 4) for k, v in rec['err_by_weight_model'].items()} }  "
          f"({rec['train_time_sec']:.1f}s)",
          flush=True)


if __name__ == "__main__":
    main()
