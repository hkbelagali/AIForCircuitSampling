"""RCS + XEB pipeline using the battle-tested GRU AR-RNN.

The model is the same body we used for Hubbard (single GRU + linear head)
but with the half-filling sector mask removed and no sign head: XEB only
cares about the bitstring distribution |psi(x)|^2.

Bit-order convention: cirq MSB-first throughout — training samples come
from cirq's measurement output, candidate samples are decoded with
`bits_to_int` (cirq order) before indexing into p_C.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import (
    bits_to_int, exact_probabilities, int_to_bits, sample_from_circuit,
)
from aics.circuits.xeb import linear_xeb, novel_xeb_vs_radius


_BOS = 2


class BitstringARRNN(nn.Module):
    """AR-RNN over {0,1}^N, no sector mask, no sign head."""

    def __init__(self, n_qubits, d_hidden=64):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_hidden = d_hidden
        self.emb = nn.Embedding(3, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, 2)

    def _shifted(self, x):
        bos = torch.full((x.shape[0], 1), _BOS, dtype=torch.long, device=x.device)
        return torch.cat([bos, x[:, :-1]], dim=1)

    def log_prob(self, x):
        logits = self.head(self.gru(self.emb(self._shifted(x)))[0])
        return -F.cross_entropy(
            logits.transpose(1, 2), x, reduction="none"
        ).sum(dim=1)

    @torch.no_grad()
    def sample(self, n_samples, temperature=1.0):
        dev = next(self.parameters()).device
        x = torch.zeros(n_samples, self.n_qubits, dtype=torch.long, device=dev)
        h = None
        cur = torch.full((n_samples, 1), _BOS, dtype=torch.long, device=dev)
        for i in range(self.n_qubits):
            out, h = self.gru(self.emb(cur), h)
            logits = self.head(out[:, 0, :]) / temperature
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            x[:, i] = sampled
            cur = sampled.unsqueeze(-1)
        return x


def train_nll(model, X_train, epochs=300, lr=2e-3, batch_size=64,
              log_every=50, verbose=True,
              pt_penalty_lambda=0.0, pt_penalty_target=None, n_qubits=None):
    """Minimize -mean log_prob(x) on training bitstrings.

    X_train: (k, N) int64 torch tensor on the model's device.

    At small N we collapse the per-sample NLL to a weighted sum over
    *unique* training strings — mathematically identical but 100-200×
    faster than mini-batching because kernel-launch overhead dominates
    actual compute at D=2^N=256.

    The `batch_size` argument is now ignored; one full step per epoch.

    If pt_penalty_lambda > 0, add a Porter–Thomas entropy regularizer:
        loss = NLL + lambda * (H(p_theta) - H_target)^2
    H_target defaults to the PT value log(D) - (1 - gamma_E).
    """
    EULER_GAMMA = 0.5772156649015329
    if n_qubits is None:
        n_qubits = X_train.shape[1]
    D = 1 << n_qubits
    use_pt = pt_penalty_lambda > 0.0

    # Collapse training samples → unique strings + weights.
    train_int = X_train.cpu().numpy()
    # X_train is (k, N) bits; convert to integers in cirq MSB-first order.
    powers = (1 << np.arange(n_qubits, dtype=np.int64))[::-1]
    train_idx = (train_int.astype(np.int64) @ powers)
    unique_ints, counts = np.unique(train_idx, return_counts=True)
    weights = counts.astype(np.float32) / counts.sum()
    u_bits = torch.from_numpy(int_to_bits(unique_ints, n_qubits)).long().to(
        X_train.device)
    w_t = torch.from_numpy(weights).to(torch.float32).to(X_train.device)

    if use_pt:
        if pt_penalty_target is None:
            pt_penalty_target = float(np.log(D) - (1.0 - EULER_GAMMA))
        all_int_t = torch.from_numpy(
            int_to_bits(np.arange(D, dtype=np.int64), n_qubits)).long().to(
            X_train.device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr / 100)

    last_nll = float("nan")
    last_h = float("nan")
    for ep in range(epochs):
        log_p = model.log_prob(u_bits)
        nll = -(w_t.double() * log_p).sum()
        loss = nll
        if use_pt:
            logp_all = model.log_prob(all_int_t)
            p_all = torch.softmax(logp_all, dim=0)
            H_model = -(p_all * torch.log(p_all + 1e-30)).sum()
            penalty = (H_model - pt_penalty_target) ** 2
            loss = nll + pt_penalty_lambda * penalty
            last_h = float(H_model.detach())
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        last_nll = float(nll.detach())
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            if use_pt:
                print(f"  ep {ep:>4}: NLL = {last_nll:.4f}  H(p_theta) = {last_h:.4f} "
                      f"(target {pt_penalty_target:.4f})", flush=True)
            else:
                print(f"  ep {ep:>4}: NLL = {last_nll:.4f}", flush=True)
    return last_nll


@torch.no_grad()
def model_full_distribution(model, n, device, batch=4096):
    """p_theta(x) over all 2^n bitstrings, returned as float64 numpy."""
    dim = 1 << n
    all_int = np.arange(dim, dtype=np.int64)
    all_bits = int_to_bits(all_int, n)
    all_t = torch.from_numpy(all_bits).long().to(device)
    out = np.empty(dim, dtype=np.float64)
    for s in range(0, dim, batch):
        logp = model.log_prob(all_t[s:s + batch]).cpu().numpy().astype(np.float64)
        out[s:s + batch] = np.exp(logp)
    out /= out.sum() or 1.0
    return out


def classical_fidelity(p, q):
    """Bhattacharyya overlap squared: (sum_x sqrt(p(x) q(x)))^2."""
    return float(np.square(np.sqrt(np.clip(p, 0, None) *
                                     np.clip(q, 0, None)).sum()))


def tv_distance(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def kl_divergence(p, q, eps=1e-300):
    """KL(p || q) in nats."""
    mask = p > 0
    return float((p[mask] * (np.log(p[mask]) - np.log(np.maximum(q[mask], eps)))).sum())


def _popcount_arr(a):
    v = np.asarray(a, dtype=np.uint64).copy()
    out = np.zeros(v.shape, dtype=np.int64)
    while v.any():
        out += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return out


def min_hamming_to_set(all_int, set_int):
    """For each integer in all_int, return its min Hamming distance to set_int."""
    if len(set_int) == 0:
        return np.full(len(all_int), 1 << 30, dtype=np.int64)
    set_int = np.unique(set_int)
    xor = np.asarray(all_int, dtype=np.int64)[:, None] ^ set_int[None, :]
    return _popcount_arr(xor).min(axis=1)


def _renorm_xeb(q, p_C, mask):
    """D * sum_{x in mask} q(x) p_C(x) / sum_{x in mask} q(x) - 1.
    Returns NaN if mass on mask is zero."""
    D = len(p_C)
    Z = float(q[mask].sum())
    if Z <= 1e-15:
        return float("nan")
    return D * float((q[mask] * p_C[mask]).sum()) / Z - 1.0


def held_out_xeb(p_model, p_C, train_int, n, radii):
    """Hamming-radius hold-out, computed exactly from the model's full
    distribution. Returns one dict per radius:
      r=0: full Hilbert (no hold-out, sanity check)
      r=1: hold out exact training-set support
      r≥2: hold out the Hamming-r-ball around training set
    Each dict has model XEB, the corrected ideal XEB (replace q by p_C),
    the uniform-on-region XEB (uniform reference), and the model's mass
    on the held-out region."""
    D = len(p_C)
    all_int = np.arange(D, dtype=np.int64)
    min_dH = min_hamming_to_set(all_int, train_int)
    out = {}
    for r in radii:
        mask = min_dH >= r
        n_strings = int(mask.sum())
        if n_strings == 0:
            out[int(r)] = {"model": float("nan"), "ideal": float("nan"),
                            "uniform": float("nan"), "mass": 0.0,
                            "n_strings": 0}
            continue
        model_xeb = _renorm_xeb(p_model, p_C, mask)
        ideal_xeb = _renorm_xeb(p_C, p_C, mask)
        unif = np.full(D, 1.0 / D, dtype=np.float64)
        unif_xeb = _renorm_xeb(unif, p_C, mask)
        out[int(r)] = {
            "model": model_xeb, "ideal": ideal_xeb, "uniform": unif_xeb,
            "mass": float(p_model[mask].sum()),
            "n_strings": n_strings,
        }
    return out


def memorization_baselines(p_C, train_int):
    """Two pure-memorization baselines:
      q_emp(x)  ∝ count_train(x)            (frequency-weighted echo)
      q_unif(x) ∝ 1_{x ∈ unique(train)}     (flat over unique train strings)
    Both supported entirely on the training set, so held-out XEB is NaN.
    KL is finite (p_C > 0 everywhere)."""
    D = len(p_C)
    counts = np.bincount(np.asarray(train_int, dtype=np.int64), minlength=D)
    q_emp = counts.astype(np.float64) / max(counts.sum(), 1)
    train_uniq = np.unique(train_int)
    q_unif = np.zeros(D, dtype=np.float64)
    if len(train_uniq):
        q_unif[train_uniq] = 1.0 / len(train_uniq)
    def metrics(q):
        xeb = D * float((q * p_C).sum()) - 1.0
        kl = kl_divergence(q, p_C)
        return {"xeb": float(xeb), "kl": float(kl)}
    return {"q_emp": metrics(q_emp), "q_unif_train": metrics(q_unif)}


def evaluate_cell(model, p_C, train_int, m_candidates, n, device, rng):
    """All metrics for one trained model. Returns dict."""
    dim = len(p_C)
    train_uniq = np.unique(train_int)

    p_model = model_full_distribution(model, n, device)
    novel_mass = float(p_model.sum() - p_model[train_uniq].sum())

    cand_bits = model.sample(m_candidates).cpu().numpy()
    cand_int = bits_to_int(cand_bits)
    cand_uniq = np.unique(cand_int)
    cand_in_train_frac = float(np.isin(cand_int, train_uniq).mean())
    cand_xeb = linear_xeb(cand_int, p_C)

    F_cl = classical_fidelity(p_model, p_C)
    tv = tv_distance(p_model, p_C)
    kl = kl_divergence(p_model, p_C)
    H_model = float(-(p_model[p_model > 0] *
                       np.log(p_model[p_model > 0])).sum())

    radii = list(range(0, n + 1))
    nxeb = novel_xeb_vs_radius(cand_int, train_int, p_C, n, radii=radii)

    heldout = held_out_xeb(p_model, p_C, train_int, n, radii)
    memo = memorization_baselines(p_C, train_int)

    # xeb_gen = D * E_{z~p_C}[p_model(z)] - 1 (Ryan-style generalisation XEB)
    xeb_gen = float(dim * np.dot(p_C, p_model) - 1.0)

    return {
        "candidate_xeb": float(cand_xeb),
        "xeb_gen": xeb_gen,
        "candidate_unique": int(len(cand_uniq)),
        "candidate_in_train_frac": cand_in_train_frac,
        "novel_mass": novel_mass,
        "classical_fidelity": F_cl,
        "tv_distance": tv,
        "kl_model_vs_truth": kl,
        "model_entropy": H_model,
        "nxeb_model": {int(r): (float(v), int(c)) for r, (v, c) in nxeb.items()},
        "heldout_xeb": heldout,
        "memorization": memo,
    }


def enumerate_z_supports(n, max_weight):
    """Returns (supports_list, weights_array) for all Z-Pauli subsets of
    weight 0..max_weight on n qubits. Identity (weight 0) is first."""
    from itertools import combinations
    supports = [()]
    weights = [0]
    for w in range(1, max_weight + 1):
        for S in combinations(range(n), w):
            supports.append(S)
            weights.append(w)
    return supports, np.asarray(weights, dtype=np.int64)


def parity_matrix(supports, n):
    """W of shape (n_obs, 2^n): W[i, x] = chi_{S_i}(x) = (-1)^|x ∩ S_i|.

    Uses cirq MSB-first convention so it lines up with bits_to_int."""
    dim = 1 << n
    all_int = np.arange(dim, dtype=np.int64)
    all_bits = int_to_bits(all_int, n)  # (dim, n), MSB-first
    W = np.ones((len(supports), dim), dtype=np.float64)
    for i, S in enumerate(supports):
        if not S:
            continue
        parity = all_bits[:, list(S)].sum(axis=1) & 1
        W[i] = np.where(parity == 0, 1.0, -1.0)
    return W


def shadow_z_expectations(samples_int, supports, n):
    """Empirical <Z_S> = mean over samples of chi_S(x). Returns (n_obs,)."""
    if len(samples_int) == 0:
        return np.zeros(len(supports), dtype=np.float64)
    sample_bits = int_to_bits(samples_int, n)  # (k, n), MSB-first
    out = np.empty(len(supports), dtype=np.float64)
    for i, S in enumerate(supports):
        if not S:
            out[i] = 1.0
            continue
        parity = sample_bits[:, list(S)].sum(axis=1) & 1
        out[i] = float(np.where(parity == 0, 1.0, -1.0).mean())
    return out


def train_rnn_z_pauli(model, samples_int, supports, weights, n, *,
                       alpha=None, epochs=400, lr=2e-3, device=None,
                       verbose=False, log_every=80):
    """Fit model so that model expectations <Z_S>_theta match shadow targets
    on the trained Pauli set. Uses a full-distribution differentiable forward
    over 2^n bitstrings (fine for small n)."""
    device = device or next(model.parameters()).device
    targets_np = shadow_z_expectations(samples_int, supports, n)
    W_np = parity_matrix(supports, n)
    W = torch.from_numpy(W_np).to(torch.float64).to(device)
    targets = torch.from_numpy(targets_np).to(torch.float64).to(device)
    if alpha is None:
        alpha = np.ones(len(supports), dtype=np.float64)
    alpha_t = torch.from_numpy(np.asarray(alpha, dtype=np.float64)).to(device)

    dim = 1 << n
    all_int = np.arange(dim, dtype=np.int64)
    all_bits_t = torch.from_numpy(int_to_bits(all_int, n)).long().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr / 100)
    for ep in range(epochs):
        logp = model.log_prob(all_bits_t).to(torch.float64)
        p = torch.softmax(logp, dim=0)
        exps = W @ p
        loss = (alpha_t * (exps - targets).pow(2)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  ep {ep:>4}: loss={float(loss):.4e}", flush=True)
    return float(loss)


@torch.no_grad()
def model_z_expectations(model, supports, n, device=None):
    device = device or next(model.parameters()).device
    dim = 1 << n
    all_int = np.arange(dim, dtype=np.int64)
    all_bits_t = torch.from_numpy(int_to_bits(all_int, n)).long().to(device)
    logp = model.log_prob(all_bits_t).to(torch.float64)
    p = torch.softmax(logp, dim=0).cpu().numpy()
    W = parity_matrix(supports, n)
    return W @ p


def run_z_pauli_cell(n, depth, k_train, w_train, seed,
                     d_hidden=64, epochs=400, lr=2e-3,
                     device=None, verbose=False, circuit_cache=None):
    """Train RNN on Z-Pauli targets of weight ≤ w_train (shadow from k_train
    bitstring samples), evaluate per-weight error of model vs truth on ALL
    Z observables of weight ≤ n."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if circuit_cache is None:
        rows, cols = grid_for(n)
        qubits, circuit = make_rcs_circuit(rows, cols, depth, seed=seed)
        p_C = exact_probabilities(circuit, qubits)
    else:
        circuit, qubits, p_C = circuit_cache

    rng = np.random.default_rng(seed * 1009 + 7)
    train_int = sample_from_circuit(circuit, qubits, k_train, seed=seed)

    # Train on weight ≤ w_train
    train_supports, train_w = enumerate_z_supports(n, w_train)
    torch.manual_seed(seed)
    model = BitstringARRNN(n_qubits=n, d_hidden=d_hidden).to(device)
    t0 = time.time()
    final_loss = train_rnn_z_pauli(model, train_int, train_supports, train_w,
                                     n, epochs=epochs, lr=lr, device=device,
                                     verbose=verbose)
    elapsed = time.time() - t0

    # Evaluate on ALL weights ≤ n
    full_supports, full_w = enumerate_z_supports(n, n)
    W_full = parity_matrix(full_supports, n)
    true_exp = W_full @ p_C
    model_exp = model_z_expectations(model, full_supports, n, device=device)
    shadow_exp = shadow_z_expectations(train_int, full_supports, n)

    err_model = np.abs(model_exp - true_exp)
    err_shadow = np.abs(shadow_exp - true_exp)
    by_w_model, by_w_shadow, by_w_true_rms = {}, {}, {}
    for w in range(n + 1):
        mask = (full_w == w)
        if mask.sum() == 0:
            continue
        by_w_model[int(w)] = float(err_model[mask].mean())
        by_w_shadow[int(w)] = float(err_shadow[mask].mean())
        by_w_true_rms[int(w)] = float(np.sqrt(np.square(true_exp[mask]).mean()))

    return {
        "n": n, "depth": depth, "k_train": k_train, "w_train": w_train,
        "seed": seed, "d_hidden": d_hidden, "epochs": epochs,
        "final_loss": final_loss, "elapsed_sec": elapsed,
        "err_by_weight_model": by_w_model,
        "err_by_weight_shadow": by_w_shadow,
        "true_rms_by_weight": by_w_true_rms,
    }


def run_rcs_xeb_cell(n, depth, k_train, m_candidates, seed,
                     d_hidden=64, epochs=300, lr=2e-3, batch_size=64,
                     k_held=1000, device=None, verbose=True,
                     circuit_cache=None, p_C_only=None,
                     pt_penalty_lambda=0.0, pt_penalty_target=None):
    """End-to-end: build circuit, sample, train RNN, full evaluation.

    If p_C_only is passed, we skip cirq entirely and sample from p_C
    multinomially — for peaked circuits or any context where p_C is
    pre-computed but no cirq circuit exists."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    if p_C_only is not None:
        p_C = np.asarray(p_C_only, dtype=np.float64)
        circuit = qubits = None
    elif circuit_cache is None:
        rows, cols = grid_for(n)
        qubits, circuit = make_rcs_circuit(rows, cols, depth, seed=seed)
        p_C = exact_probabilities(circuit, qubits)
    else:
        circuit, qubits, p_C = circuit_cache
    dim = len(p_C)
    H_true = float(-(p_C[p_C > 0] * np.log(p_C[p_C > 0])).sum())

    if p_C_only is not None:
        rng_tr = np.random.default_rng(seed)
        rng_ho = np.random.default_rng(seed + 999983)
        train_int = rng_tr.choice(dim, size=k_train, p=p_C).astype(np.int64)
        held_int = rng_ho.choice(dim, size=k_held, p=p_C).astype(np.int64)
    else:
        train_int = sample_from_circuit(circuit, qubits, k_train, seed=seed)
        held_int = sample_from_circuit(circuit, qubits, k_held, seed=seed + 999983)
    f_train = linear_xeb(train_int, p_C)
    f_held = linear_xeb(held_int, p_C)

    X_train = torch.from_numpy(int_to_bits(train_int, n)).long().to(device)
    X_held = torch.from_numpy(int_to_bits(held_int, n)).long().to(device)
    torch.manual_seed(seed)
    model = BitstringARRNN(n_qubits=n, d_hidden=d_hidden).to(device)
    t0 = time.time()
    final_nll = train_nll(model, X_train, epochs=epochs, lr=lr,
                          batch_size=batch_size, verbose=verbose,
                          log_every=max(1, epochs // 5),
                          pt_penalty_lambda=pt_penalty_lambda,
                          pt_penalty_target=pt_penalty_target,
                          n_qubits=n)
    train_time = time.time() - t0
    with torch.no_grad():
        held_nll = float(-model.log_prob(X_held).mean())
    uniform_nll = n * float(np.log(2))

    model.eval()
    metrics = evaluate_cell(model, p_C, train_int, m_candidates, n, device, rng)

    return {
        "n": n, "depth": depth, "k_train": k_train,
        "m_candidates": m_candidates, "seed": seed,
        "dim": dim, "H_true": H_true,
        "train_xeb": float(f_train), "held_xeb": float(f_held),
        "train_nll": float(final_nll), "held_nll": held_nll,
        "uniform_nll": uniform_nll, "train_time_sec": train_time,
        "d_hidden": d_hidden, "epochs": epochs,
        "pt_penalty_lambda": float(pt_penalty_lambda),
        **metrics,
    }
