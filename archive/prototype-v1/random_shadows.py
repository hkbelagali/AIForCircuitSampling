"""Shadow training on a random real unit vector on the *full* 2^N Hilbert
space (no half-filling sector). Parallel to shadows.py but with:
  - UnconstrainedARRNN: AR-RNN with no sector mask, generates over all
    2^N bitstrings.
  - Pauli enumeration over every even-Y Pauli on N qubits (no sector
    filter — every Pauli has nontrivial matrix elements on the full
    Hilbert space).

Reuses sample_shadows_random_pauli / sample_shadows_z_only / transition_matrix
from shadows.py — those already work for any state on any subspace.
"""

import json
import time
from itertools import combinations, product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import shadows
from shadows import (
    sample_shadows_random_pauli, sample_shadows_z_only,
    transition_matrix, _model_device, _BOS,
)
from m9.hubbard import state_int_to_bits

_X, _Y, _Z = 1, 2, 3


class UnconstrainedARRNN(nn.Module):
    """Sign-augmented AR-RNN over all 2^N bitstrings — no sector mask."""

    def __init__(self, n_qubits, d_hidden=32, d_sign_hidden=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.L = n_qubits // 2  # only for downstream compat
        self.emb = nn.Embedding(3, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, 2)
        d_sh = d_sign_hidden or 4 * d_hidden
        self.sign_head = nn.Sequential(
            nn.Linear(2 * d_hidden, d_sh), nn.ReLU(),
            nn.Linear(d_sh, d_sh), nn.ReLU(),
            nn.Linear(d_sh, 1),
        )

    def _shifted(self, x):
        bos = torch.full((x.shape[0], 1), _BOS, dtype=torch.long, device=x.device)
        return torch.cat([bos, x[:, :-1]], dim=1)

    def _features_and_logq(self, x):
        feat = self.gru(self.emb(self._shifted(x)))[0]
        logits = self.head(feat)  # no mask
        log_p = -F.cross_entropy(logits.transpose(1, 2), x, reduction="none").sum(dim=1)
        return feat, log_p

    def psi(self, x, use_sign=True):
        feat, log_p = self._features_and_logq(x)
        mag = torch.exp(0.5 * log_p)
        if not use_sign:
            return mag
        pooled = torch.cat([feat[:, -1, :], feat.mean(dim=1)], dim=-1)
        sign = torch.tanh(self.sign_head(pooled).squeeze(-1))
        return sign * mag


# ---- Full-Hilbert Pauli enumeration (no sector filter) ----

def _enumerate_full_paulis(n_qubits, max_weight):
    """Yield every even-Y Pauli on n_qubits of weight 0..max_weight.
    Returns (xm, ym, zm, w, nY, supp_indices, supp_codes)."""
    yield 0, 0, 0, 0, 0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    for w in range(1, max_weight + 1):
        for support in combinations(range(n_qubits), w):
            sup_arr = np.array(support, dtype=np.int64)
            for types in product((_X, _Y, _Z), repeat=w):
                nY = sum(1 for t in types if t == _Y)
                if nY % 2 != 0:
                    continue
                xm = ym = zm = 0
                codes = np.empty(w, dtype=np.int64)
                for i, (q, t) in enumerate(zip(support, types)):
                    if t == _X:
                        xm |= 1 << q; codes[i] = 1
                    elif t == _Y:
                        ym |= 1 << q; codes[i] = 2
                    else:
                        zm |= 1 << q; codes[i] = 0
                yield xm, ym, zm, w, nY, sup_arr, codes


def _triples_for_full_pauli(xm, ym, zm, w, nY, n_qubits):
    """For a Pauli on the full 2^N Hilbert: every basis state maps to
    another via the XOR with xy. Returns (i, j, c)."""
    D = 1 << n_qubits
    states = np.arange(D, dtype=np.int64)
    xy = xm | ym
    yz = ym | zm
    sign_fac = 1 if (nY // 2) % 2 == 0 else -1
    new_states = states ^ xy
    # Phase: count bits of states & yz (and Y contribution already in sign_fac)
    phase_par = _popcount_int_arr(states & yz) & 1
    c = sign_fac * np.where(phase_par == 0, 1, -1).astype(np.int64)
    return states.copy(), new_states, c


def _popcount_int_arr(a):
    v = np.asarray(a, dtype=np.uint64).copy()
    out = np.zeros(v.shape, dtype=np.int64)
    while v.any():
        out += (v & np.uint64(1)).astype(np.int64)
        v >>= np.uint64(1)
    return out


def build_loss_paulis_full(n_qubits, max_weight):
    """Stacked-triples representation for fast model_expectations."""
    xyz = []; weights = []; supports = []; supp_codes = []
    all_I = []; all_J = []; all_C = []; all_P = []
    p_idx = 0
    for xm, ym, zm, w, nY, sup_arr, codes in _enumerate_full_paulis(
            n_qubits, max_weight):
        i, j, c = _triples_for_full_pauli(xm, ym, zm, w, nY, n_qubits)
        xyz.append((xm, ym, zm)); weights.append(w)
        supports.append(sup_arr); supp_codes.append(codes)
        all_I.append(i); all_J.append(j); all_C.append(c)
        all_P.append(np.full(len(i), p_idx, dtype=np.int64))
        p_idx += 1
    return {
        "xyz": np.asarray(xyz, dtype=np.int64),
        "weight": np.asarray(weights, dtype=np.int64),
        "supports": supports,
        "supp_codes": supp_codes,
        "I": np.concatenate(all_I),
        "J": np.concatenate(all_J),
        "C": np.concatenate(all_C).astype(np.float64),
        "P": np.concatenate(all_P),
        "n_paulis": p_idx,
    }


def shadow_targets_full(loss_paulis, U_pattern, b_out):
    """Same formula as pauli_loss.shadow_targets but for the full Hilbert
    Pauli set."""
    n_p = loss_paulis["n_paulis"]
    weights = loss_paulis["weight"]
    supports = loss_paulis["supports"]
    supp_codes = loss_paulis["supp_codes"]
    out = np.empty(n_p, dtype=np.float64)
    out[0] = 1.0  # identity
    for p in range(1, n_p):
        S = supports[p]; codes = supp_codes[p]; w = weights[p]
        U_at_S = U_pattern[:, S]
        matches = (U_at_S == codes[None, :]).all(axis=1)
        if not matches.any():
            out[p] = 0.0; continue
        b_at_S = b_out[:, S]
        parity = b_at_S.sum(axis=1) & 1
        sign = np.where(parity == 0, 1.0, -1.0)
        out[p] = float((matches.astype(np.float64) * sign).mean() * (3.0 ** w))
    return out


def model_expectations_full(psi, ops):
    psi = psi.to(torch.float64)
    products = psi[ops["I"]] * ops["C"] * psi[ops["J"]]
    out = torch.zeros(ops["n_paulis"], dtype=torch.float64, device=psi.device)
    out.index_add_(0, ops["P"], products)
    return out


def true_expectations_full(psi_true, loss_paulis):
    """Exact <P>_truth via the stacked triples."""
    psi = np.asarray(psi_true, dtype=np.float64)
    products = psi[loss_paulis["I"]] * loss_paulis["C"] * psi[loss_paulis["J"]]
    out = np.zeros(loss_paulis["n_paulis"], dtype=np.float64)
    np.add.at(out, loss_paulis["P"], products)
    return out


def alpha_array(loss_paulis, weighting):
    w = loss_paulis["weight"]
    if weighting == "uniform":
        return np.ones(len(w), dtype=np.float64)
    if weighting == "variance":
        return 3.0 ** (-w.astype(np.float64))
    raise ValueError(f"unknown weighting {weighting!r}")


def torch_ops_full(loss_paulis, device):
    return {
        "I": torch.from_numpy(loss_paulis["I"]).long().to(device),
        "J": torch.from_numpy(loss_paulis["J"]).long().to(device),
        "C": torch.from_numpy(loss_paulis["C"]).double().to(device),
        "P": torch.from_numpy(loss_paulis["P"]).long().to(device),
        "n_paulis": loss_paulis["n_paulis"],
    }


# ---- Training cells ----

def _eval_full_random(model, ctx, x_bits_t):
    """Returns (fid, mag_TV, E) against ctx.psi_0 (the random target)."""
    dev = _model_device(model)
    x = x_bits_t.to(dev) if torch.is_tensor(x_bits_t) else x_bits_t
    with torch.no_grad():
        psi_theta = model.psi(x).cpu().numpy().astype(np.float64)
        mag = model.psi(x, use_sign=False).cpu().numpy().astype(np.float64)
    nrm = np.linalg.norm(psi_theta) or 1.0
    psi = psi_theta / nrm
    fid = float((ctx.psi_0 @ psi) ** 2)
    E = float(psi @ (ctx.H @ psi)) if ctx.H is not None else 0.0
    q_model = mag ** 2; q_model /= q_model.sum() or 1.0
    p_true = ctx.psi_0 ** 2; p_true /= p_true.sum()
    mag_tv = 0.5 * float(np.abs(q_model - p_true).sum())
    return fid, mag_tv, E


def run_random_cell_mle(n_qubits, k_total, seed, *, d_hidden=32,
                         epochs_A=1500, epochs_B=5000, lr=1e-3,
                         n_restarts=16, selector="val", val_frac=0.1,
                         device="cuda", verbose=False, pauli_max_weight=3):
    """Shadow-MLE training on a random Gaussian unit vector target."""
    from random_state import RandomHilbertState
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ctx = RandomHilbertState(n_qubits=n_qubits, seed=seed)
    model = UnconstrainedARRNN(n_qubits=n_qubits, d_hidden=d_hidden).to(device)
    x_bits_t = torch.from_numpy(
        state_int_to_bits(ctx.states, ctx.L).astype(np.int64)).long().to(device)
    t0 = time.time()
    # Two-stage protocol: kz Z-only shots for magnitudes, kr random shots for signs.
    k_z = max(1, int(round(0.25 * k_total)))
    k_r = max(1, k_total - k_z)
    U_z, b_z = sample_shadows_z_only(ctx.psi_0, ctx.states, ctx.L, k_z, rng)
    U_r, b_r = sample_shadows_random_pauli(ctx.psi_0, ctx.states, ctx.L, k_r, rng)
    k_val = max(50, int(val_frac * k_r))
    k_val = min(k_val, max(0, k_r // 4))
    k_tr = k_r - k_val
    U_tr, b_tr = U_r[:k_tr], b_r[:k_tr]
    if k_val > 0: U_val, b_val = U_r[k_tr:], b_r[k_tr:]
    mag_params = [p for n, p in model.named_parameters() if not n.startswith("sign_head")]
    sign_params = [p for n, p in model.named_parameters() if n.startswith("sign_head")]
    # Phase A: Z-only magnitudes
    shadows.shadow_mle_train(model, U_z, b_z, ctx, epochs=epochs_A, lr=lr,
                             sub_k=None, report_every=epochs_A + 1, use_sign=False,
                             params=mag_params, label="A", lr_decay=True,
                             silent=not verbose)
    mag_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                 if not k.startswith("sign_head")}
    # Phase B: random-Pauli signs with restarts + val selection
    best_score = float("inf"); best_state = None
    gen = torch.Generator(device=device).manual_seed(seed * 9973)
    for r_i in range(n_restarts):
        sd = model.state_dict()
        for k, v in mag_state.items(): sd[k] = v.clone()
        model.load_state_dict(sd)
        shadows._reinit_sign_head(model, gen)
        shadows.shadow_mle_train(model, U_tr, b_tr, ctx, epochs=epochs_B, lr=lr,
                                 sub_k=None, report_every=epochs_B + 1,
                                 use_sign=True, params=sign_params,
                                 label=f"B-{r_i}", lr_decay=True,
                                 cache_features=True, silent=not verbose)
        train_loss = shadows._final_loss(model, U_tr, b_tr, ctx, device)
        val_loss = (shadows._final_loss(model, U_val, b_val, ctx, device)
                    if k_val > 0 else None)
        score = val_loss if (selector == "val" and val_loss is not None) else train_loss
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    fid, mag_tv, E = _eval_full_random(model, ctx, x_bits_t)
    with torch.no_grad():
        psi_model = model.psi(x_bits_t).cpu().numpy().astype(np.float64)
    psi_model = psi_model / (np.linalg.norm(psi_model) or 1.0)
    return {
        "n_qubits": n_qubits, "L": ctx.L, "k_total": k_total, "seed": seed,
        "k_z": k_z, "k_random": k_r,
        "fidelity": fid, "mag_TV": mag_tv,
        "E_model": E, "E_0": ctx.E_0,
        "rel_err": (E - ctx.E_0) / abs(ctx.E_0) if ctx.E_0 != 0 else 0.0,
        "psi_model": psi_model.tolist(),
        "loss": "shadow_mle", "target": "random_hilbert",
        "epochs_A": epochs_A, "epochs_B": epochs_B, "lr": lr,
        "d_hidden": d_hidden, "n_restarts": n_restarts,
        "elapsed_sec": time.time() - t0,
    }


def run_random_cell_pauli(n_qubits, k_total, seed, *, w_max, weighting,
                           d_hidden=32, epochs=2000, lr=1e-3,
                           device="cuda", verbose=False):
    """Pauli-loss training on a random Gaussian unit vector target."""
    from random_state import RandomHilbertState
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ctx = RandomHilbertState(n_qubits=n_qubits, seed=seed)
    model = UnconstrainedARRNN(n_qubits=n_qubits, d_hidden=d_hidden).to(device)
    x_bits_t = torch.from_numpy(
        state_int_to_bits(ctx.states, ctx.L).astype(np.int64)).long().to(device)
    t0 = time.time()
    U_r, b_r = sample_shadows_random_pauli(ctx.psi_0, ctx.states, ctx.L,
                                           k_total, rng)
    loss_paulis = build_loss_paulis_full(n_qubits, w_max)
    targets = shadow_targets_full(loss_paulis, U_r, b_r)
    alphas = alpha_array(loss_paulis, weighting)
    ops = torch_ops_full(loss_paulis, device)
    targets_t = torch.from_numpy(targets).double().to(device)
    alpha_t = torch.from_numpy(alphas).double().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=lr / 100)
    for ep in range(epochs):
        psi = model.psi(x_bits_t).to(torch.float64)
        psi = psi / (psi.pow(2).sum() + 1e-30).sqrt()
        exps = model_expectations_full(psi, ops)
        diff = exps - targets_t
        loss = (alpha_t * diff.pow(2)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and ep % 200 == 0:
            print(f"  ep {ep}: loss={loss.item():.4g}", flush=True)
    fid, mag_tv, E = _eval_full_random(model, ctx, x_bits_t)
    with torch.no_grad():
        psi_model = model.psi(x_bits_t).cpu().numpy().astype(np.float64)
    psi_model = psi_model / (np.linalg.norm(psi_model) or 1.0)
    return {
        "n_qubits": n_qubits, "L": ctx.L, "k_total": k_total, "seed": seed,
        "fidelity": fid, "mag_TV": mag_tv,
        "E_model": E, "E_0": ctx.E_0,
        "rel_err": (E - ctx.E_0) / abs(ctx.E_0) if ctx.E_0 != 0 else 0.0,
        "psi_model": psi_model.tolist(),
        "loss": "pauli", "target": "random_hilbert",
        "w_max": w_max, "weighting": weighting,
        "n_paulis_trained": int(loss_paulis["n_paulis"]),
        "epochs": epochs, "lr": lr, "d_hidden": d_hidden,
        "elapsed_sec": time.time() - t0,
    }
