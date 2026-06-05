"""Train sign-augmented AR-RNN by MLE on classical shadows
(random 1-local Pauli-basis measurements) of the half-filled Hubbard
ground state, then evaluate fidelity, energy, and per-weight Pauli
expectation errors.

Two-stage protocol (run_shadow_cell):
  Phase A: k_z Z-basis Born samples constrain the magnitudes
           (signs forced to +1 in the loss).
  Phase B: k_random random-Pauli shadow shots constrain the signs
           with magnitudes frozen.

The sign head is bistable: about a third of random initializations land
in a "wrong-sign" basin where fidelity is ~0. Mitigated by n_restarts
(default 16 in our sweeps) plus held-out random-Pauli NLL as the
selector. See run_shadow_cell."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from m9 import Hubbard
from m9.hubbard import state_int_to_bits


# Per-qubit transition amplitudes T[basis, b, x] = <b|U|x>.
#   basis 0 -> Z (apply I, then Z-measure)
#   basis 1 -> X (apply H, then Z-measure)
#   basis 2 -> Y (apply HS^dagger, then Z-measure)
_T = np.array([
    [[1, 0], [0, 1]],
    [[1, 1], [1, -1]],
    [[1, -1j], [1, 1j]],
], dtype=np.complex128)
_T[1] /= np.sqrt(2)
_T[2] /= np.sqrt(2)


def _apply_1q_gate(psi, gate, qubit, N):
    """Apply a 2x2 unitary at the given qubit on a 2^N statevector."""
    psi = psi.reshape([2] * N)
    axis = N - 1 - qubit
    psi = np.moveaxis(psi, axis, 0)
    psi = np.einsum("ij,j...->i...", gate, psi)
    psi = np.moveaxis(psi, 0, axis)
    return psi.reshape(-1)


def _sample_with_pattern(psi_full, U_pattern, rng):
    N = U_pattern.size
    D_full = 1 << N
    psi_rot = psi_full
    for q in range(N):
        basis = int(U_pattern[q])
        if basis == 0:
            continue
        psi_rot = _apply_1q_gate(psi_rot, _T[basis], q, N)
    probs = np.abs(psi_rot) ** 2
    probs = probs / probs.sum()
    b_int = int(rng.choice(D_full, p=probs))
    b_bits = np.array([(b_int >> q) & 1 for q in range(N)], dtype=np.int64)
    return b_bits


def sample_shadows_random_pauli(psi_sector, sector_states, L, k, rng):
    """Each shot: pick a random per-qubit Pauli basis, then Born-sample
    the rotated state. Returns (U_pattern[k, 2L], b_out[k, 2L])."""
    N = 2 * L
    D_full = 1 << N
    psi_full = np.zeros(D_full, dtype=np.complex128)
    psi_full[sector_states] = psi_sector
    U_pattern = rng.integers(0, 3, size=(k, N))
    b_out = np.empty((k, N), dtype=np.int64)
    for t in range(k):
        b_out[t] = _sample_with_pattern(psi_full, U_pattern[t], rng)
    return U_pattern, b_out


def sample_shadows_z_only(psi_sector, sector_states, L, k, rng):
    """All-Z baseline: each shot is a plain Born-rule bitstring."""
    N = 2 * L
    p = (psi_sector ** 2)
    idx = rng.choice(len(p), size=k, p=p / p.sum())
    states = sector_states[idx]
    b_out = np.zeros((k, N), dtype=np.int64)
    for i in range(N):
        b_out[:, i] = (states >> i) & 1
    U_pattern = np.zeros((k, N), dtype=np.int64)
    return U_pattern, b_out


def transition_matrix(U_pattern, b_out, sector_states, L):
    """Complex (k, D) matrix where M[t, j] = <b^(t)|U^(t)|x_j>.
    Vectorized via the precomputed per-qubit T lookup."""
    N = 2 * L
    x_bits = state_int_to_bits(sector_states, L)         # (D, 2L)
    Up = U_pattern[:, None, :]                           # (k, 1, N)
    Bp = b_out[:, None, :]                               # (k, 1, N)
    Xp = x_bits[None, :, :]                              # (1, D, N)
    T_per = _T[Up, Bp, Xp]                               # (k, D, N) complex
    return T_per.prod(axis=-1)                           # (k, D)


# ---- Sign-augmented AR-RNN ----

_BOS = 2


class SignedARRNN(nn.Module):
    def __init__(self, L, n_up, n_dn, d_hidden=32, d_sign_hidden=None):
        super().__init__()
        self.L = L
        self.n_up = n_up
        self.n_dn = n_dn
        self.emb = nn.Embedding(3, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, 2)
        d_sh = d_sign_hidden or 4 * d_hidden
        # Sign head reads pooled GRU features (mean + last across positions).
        self.sign_head = nn.Sequential(
            nn.Linear(2 * d_hidden, d_sh), nn.ReLU(),
            nn.Linear(d_sh, d_sh), nn.ReLU(),
            nn.Linear(d_sh, 1),
        )

    def _mask(self, x):
        B, L = x.shape[0], self.L
        dev = x.device
        z = torch.zeros(B, 1, dtype=torch.long, device=dev)
        up_so_far = torch.cat([z, x[:, :L].long().cumsum(1)[:, :-1]], dim=1)
        dn_so_far = torch.cat([z, x[:, L:].long().cumsum(1)[:, :-1]], dim=1)
        remaining = (L - torch.arange(L, device=dev)).unsqueeze(0)
        bu, bd = self.n_up - up_so_far, self.n_dn - dn_so_far
        m = torch.zeros(B, 2 * L, 2, dtype=torch.bool, device=dev)
        m[:, :L, 0] = (bu >= 0) & (bu <= remaining - 1)
        m[:, :L, 1] = (bu >= 1) & (bu - 1 <= remaining - 1)
        m[:, L:, 0] = (bd >= 0) & (bd <= remaining - 1)
        m[:, L:, 1] = (bd >= 1) & (bd - 1 <= remaining - 1)
        return m

    def _shifted(self, x):
        bos = torch.full((x.shape[0], 1), _BOS, dtype=torch.long, device=x.device)
        return torch.cat([bos, x[:, :-1]], dim=1)

    def _features_and_logq(self, x):
        feat = self.gru(self.emb(self._shifted(x)))[0]                  # (B, 2L, d)
        logits = self.head(feat).masked_fill(~self._mask(x), float("-inf"))
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


def _model_device(model):
    return next(model.parameters()).device


def shadow_mle_train(model, U_pattern, b_out, ctx, epochs=200, lr=1e-3,
                     sub_k=None, report_every=20, use_sign=True, params=None,
                     label="", lr_decay=False, cache_features=False, silent=False):
    """Stochastic full-batch MLE on shadow shots.

    cache_features=True (requires use_sign=True and frozen GRU+mag-head):
    pre-compute GRU features + magnitudes once and only forward through the
    sign head per step. Much faster at large L. Math identical to the
    full-recompute path when those params are actually frozen.

    lr_decay=True: cosine LR schedule from lr to lr/100 over `epochs`."""
    dev = _model_device(model)
    M_full = torch.from_numpy(transition_matrix(U_pattern, b_out, ctx.states, ctx.L)
                              ).to(torch.complex128).to(dev)
    x_bits_t = torch.from_numpy(
        state_int_to_bits(ctx.states, ctx.L).astype(np.int64)).long().to(dev)
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                       eta_min=lr / 100)
             if lr_decay else None)
    k = M_full.shape[0]

    if cache_features:
        with torch.no_grad():
            feat, log_p = model._features_and_logq(x_bits_t)
        feat = feat.detach()
        mag_cached = torch.exp(0.5 * log_p).detach()
        pooled = torch.cat([feat[:, -1, :], feat.mean(dim=1)], dim=-1)

    def _compute_psi():
        if cache_features:
            sign = torch.tanh(model.sign_head(pooled).squeeze(-1))
            return sign * mag_cached
        return model.psi(x_bits_t, use_sign=use_sign)

    trace = []
    for ep in range(epochs):
        if sub_k is not None and sub_k < k:
            idx = torch.randperm(k, device=dev)[:sub_k]
            M = M_full[idx]
        else:
            M = M_full
        psi_theta = _compute_psi().to(torch.complex128)
        psi_theta = psi_theta / torch.sqrt(
            (psi_theta.real ** 2 + psi_theta.imag ** 2).sum() + 1e-30)
        amp = M @ psi_theta
        prob = amp.real ** 2 + amp.imag ** 2
        loss = -torch.log(prob + 1e-30).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if sched is not None: sched.step()
        if not silent and (ep % report_every == 0 or ep == epochs - 1):
            fid, fid_oracle, mag_tv, E = _eval_full(model, ctx, x_bits_t)
            trace.append((ep, loss.item(), fid, E))
            cur_lr = opt.param_groups[0]["lr"]
            print(f"  [{label}] ep {ep:>4d}: loss={loss.item():.3f}  "
                  f"fid={fid:.4f}  fid|oracle={fid_oracle:.4f}  "
                  f"mag_TV={mag_tv:.3f}  rel={(E-ctx.E_0)/abs(ctx.E_0):+.3f}  "
                  f"lr={cur_lr:.1e}", flush=True)
    return trace


def two_stage_split_train(model, ctx, k_z, k_random, epochs_A=100, epochs_B=200,
                          lr=1e-3, sub_k=None, report_every=50, rng=None,
                          lr_decay=False, freeze_mag_in_B=False, verbose=True):
    """Allocate the shot budget: k_z Z-only shots for magnitude pretrain,
    k_random Pauli shadows for joint sign+magnitude fit.

    Total shots = k_z + k_random. This is the honest sample-complexity protocol.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    U_z, b_z = sample_shadows_z_only(ctx.psi_0, ctx.states, ctx.L, k_z, rng)
    U_r, b_r = sample_shadows_random_pauli(ctx.psi_0, ctx.states, ctx.L, k_random, rng)

    mag_params = [p for n, p in model.named_parameters() if not n.startswith("sign_head")]
    sign_params = [p for n, p in model.named_parameters() if n.startswith("sign_head")]
    if verbose:
        print(f"=== Phase A: magnitude MLE on {k_z} Z-only shots ===")
    shadow_mle_train(model, U_z, b_z, ctx, epochs=epochs_A, lr=lr, sub_k=sub_k,
                     report_every=report_every, use_sign=False,
                     params=mag_params, label="A", lr_decay=lr_decay,
                     silent=not verbose)
    B_params = sign_params if freeze_mag_in_B else model.parameters()
    label_B = "B (signs only)" if freeze_mag_in_B else "B (joint)"
    if verbose:
        print(f"=== Phase {label_B}: MLE on {k_random} random-Pauli shots ===")
    shadow_mle_train(model, U_r, b_r, ctx, epochs=epochs_B, lr=lr, sub_k=sub_k,
                     report_every=report_every, use_sign=True,
                     params=B_params, label="B", lr_decay=lr_decay,
                     cache_features=freeze_mag_in_B, silent=not verbose)


def _eval_full(model, ctx, x_bits_t):
    """Returns (fid_learned, fid_oracle_signs, mag_TV, E)."""
    x = x_bits_t.to(_model_device(model)) if torch.is_tensor(x_bits_t) else x_bits_t
    with torch.no_grad():
        psi_theta = model.psi(x).cpu().numpy().astype(np.float64)
        mag = model.psi(x, use_sign=False).cpu().numpy().astype(np.float64)
    nrm = np.linalg.norm(psi_theta) or 1.0
    psi = psi_theta / nrm
    fid_learned = float((ctx.psi_0 @ psi) ** 2)
    E = float(psi @ (ctx.H @ psi))
    psi_oracle = ctx.signs.astype(np.float64) * mag
    psi_oracle /= np.linalg.norm(psi_oracle) or 1.0
    fid_oracle = float((ctx.psi_0 @ psi_oracle) ** 2)
    q_model = mag ** 2; q_model /= q_model.sum()
    p_true = ctx.psi_0 ** 2; p_true /= p_true.sum()
    mag_tv = 0.5 * float(np.abs(q_model - p_true).sum())
    return fid_learned, fid_oracle, mag_tv, E


def _reinit_sign_head(model, gen):
    for m in model.sign_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, generator=gen)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _final_loss(model, U, b, ctx, device):
    """One forward pass on all shots to get the final training loss."""
    dev = _model_device(model)
    M = torch.from_numpy(transition_matrix(U, b, ctx.states, ctx.L)
                         ).to(torch.complex128).to(dev)
    x = torch.from_numpy(
        state_int_to_bits(ctx.states, ctx.L).astype(np.int64)).long().to(dev)
    with torch.no_grad():
        psi = model.psi(x).to(torch.complex128)
        psi = psi / torch.sqrt((psi.real ** 2 + psi.imag ** 2).sum() + 1e-30)
        amp = M @ psi
        prob = amp.real ** 2 + amp.imag ** 2
        return float(-torch.log(prob + 1e-30).mean().item())


def _pauli_err_for_state(psi_model, ctx, max_weight):
    """Returns {w: max|<P>_model - <P>_true|, weight<=w} for w in 1..max_weight."""
    from pauli import build_pauli_triples, expectations, max_err_by_weight
    psi = psi_model / (np.linalg.norm(psi_model) or 1.0)
    ops, weights = build_pauli_triples(ctx, max_weight)
    v_true = expectations(ops, ctx.psi_0)
    v_model = expectations(ops, psi)
    return max_err_by_weight(weights, v_model, v_true, max_weight)


def run_shadow_cell(L, k_z, k_random, seed, *,
                    epochs_A=1000, epochs_B=5000, lr=1e-3, sub_k=2000,
                    d_hidden=32, freeze_mag_B=True, lr_decay=True,
                    fidelity_threshold=0.99, device="cpu", verbose=False,
                    pauli_max_weight=3, n_restarts=1,
                    selector="val", val_frac=0.1, min_val_shots=50,
                    record_per_restart=True):
    """End-to-end training of a sign-augmented AR-RNN on shadow shots.
    Returns a record dict suitable for JSON serialization.

    selector: how to pick best restart of Phase B. "val" uses held-out
    random-Pauli NLL; "train" uses training NLL (old behavior); "oracle_fid"
    uses true fidelity (cheating baseline — for diagnostic plots only).

    val_frac/min_val_shots: how many random-Pauli shots to hold out for
    selection. The held-out shots are NOT used in Phase B training.

    record_per_restart: if True, save per-restart diagnostics in the output
    JSON under "restart_records" — so plots can re-select after the fact.

    If pauli_max_weight > 0, also computes max-absolute-error of model Pauli
    expectations vs ED truth for each weight 0..pauli_max_weight.
    """
    import time
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ctx = Hubbard(L=L, U=4.0)
    model = SignedARRNN(L=L, n_up=L // 2, n_dn=L // 2,
                        d_hidden=d_hidden).to(device)
    t0 = time.time()
    x_bits_t = torch.from_numpy(
        state_int_to_bits(ctx.states, ctx.L).astype(np.int64)).long().to(device)
    restart_records = []
    if n_restarts <= 1 or not freeze_mag_B:
        two_stage_split_train(model, ctx, k_z=k_z, k_random=k_random,
                              epochs_A=epochs_A, epochs_B=epochs_B,
                              lr=lr, sub_k=sub_k, report_every=epochs_B + 1,
                              rng=rng, lr_decay=lr_decay,
                              freeze_mag_in_B=freeze_mag_B, verbose=verbose)
    else:
        # Phase A once, then n_restarts trials of Phase B.
        U_z, b_z = sample_shadows_z_only(ctx.psi_0, ctx.states, ctx.L, k_z, rng)
        U_r, b_r = sample_shadows_random_pauli(ctx.psi_0, ctx.states, ctx.L,
                                               k_random, rng)
        # Train/val split on random-Pauli shots.
        k_val = max(min_val_shots, int(val_frac * k_random))
        k_val = min(k_val, max(0, k_random // 4))
        k_tr = k_random - k_val
        U_tr, b_tr = U_r[:k_tr], b_r[:k_tr]
        if k_val > 0:
            U_val, b_val = U_r[k_tr:], b_r[k_tr:]
        mag_params = [p for n, p in model.named_parameters()
                      if not n.startswith("sign_head")]
        sign_params = [p for n, p in model.named_parameters()
                       if n.startswith("sign_head")]
        if verbose: print(f"=== Phase A ({k_z} Z shots) ===")
        shadow_mle_train(model, U_z, b_z, ctx, epochs=epochs_A, lr=lr,
                         sub_k=sub_k, report_every=epochs_A + 1, use_sign=False,
                         params=mag_params, label="A", lr_decay=lr_decay,
                         silent=not verbose)
        mag_state = {k: v.detach().clone()
                     for k, v in model.state_dict().items()
                     if not k.startswith("sign_head")}
        best_score = float("inf")
        best_state = None
        gen = torch.Generator(device=device).manual_seed(seed * 9973)
        for r_i in range(n_restarts):
            sd = model.state_dict()
            for k, v in mag_state.items(): sd[k] = v.clone()
            model.load_state_dict(sd)
            _reinit_sign_head(model, gen)
            shadow_mle_train(model, U_tr, b_tr, ctx, epochs=epochs_B, lr=lr,
                             sub_k=sub_k, report_every=epochs_B + 1,
                             use_sign=True, params=sign_params,
                             label=f"B-{r_i}", lr_decay=lr_decay,
                             cache_features=True, silent=not verbose)
            train_loss = _final_loss(model, U_tr, b_tr, ctx, device)
            val_loss = (_final_loss(model, U_val, b_val, ctx, device)
                        if k_val > 0 else None)
            r_fid, r_fid_o, r_magtv, r_E = _eval_full(model, ctx, x_bits_t)
            rec = {"restart": r_i, "train_loss": train_loss,
                   "val_loss": val_loss, "fid": r_fid,
                   "fid_oracle": r_fid_o, "mag_TV": r_magtv, "E": r_E}
            if record_per_restart:
                restart_records.append(rec)
            if selector == "train":
                score = train_loss
            elif selector == "val":
                score = val_loss if val_loss is not None else train_loss
            elif selector == "oracle_fid":
                score = -r_fid
            else:
                raise ValueError(f"unknown selector: {selector}")
            if verbose:
                vl = f" val={val_loss:.4f}" if val_loss is not None else ""
                print(f"  restart {r_i}: train={train_loss:.4f}{vl} "
                      f"fid={r_fid:.4f} fid_o={r_fid_o:.4f}", flush=True)
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
    fid, fid_oracle, mag_tv, E = _eval_full(model, ctx, x_bits_t)

    pauli_err = {}
    with torch.no_grad():
        psi_model = model.psi(x_bits_t).cpu().numpy().astype(np.float64)
    psi_model = psi_model / (np.linalg.norm(psi_model) or 1.0)
    if pauli_max_weight > 0:
        pauli_err = _pauli_err_for_state(psi_model, ctx, pauli_max_weight)

    elapsed = time.time() - t0
    return {
        "L": L, "k_z": k_z, "k_random": k_random, "seed": seed,
        "k_total": k_z + k_random,
        "E_0": ctx.E_0, "E_model": E,
        "rel_err": (E - ctx.E_0) / abs(ctx.E_0),
        "fidelity": fid, "fid_oracle_signs": fid_oracle,
        "mag_TV": mag_tv,
        "pauli_max_err_by_weight": pauli_err,
        "psi_model": psi_model.tolist(),
        "epochs_A": epochs_A, "epochs_B": epochs_B,
        "lr": lr, "sub_k": sub_k, "d_hidden": d_hidden,
        "freeze_mag_B": freeze_mag_B, "lr_decay": lr_decay,
        "n_restarts": n_restarts,
        "selector": selector,
        "restart_records": restart_records,
        "reached_fidelity_threshold": fid >= fidelity_threshold,
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    # Smoke test on a tiny cell; the full sweep lives in sweep_shadows.py.
    import json
    rec = run_shadow_cell(L=4, k_z=250, k_random=750, seed=0,
                          epochs_A=300, epochs_B=600, lr=1e-3, sub_k=None,
                          n_restarts=4, device="cpu", verbose=True,
                          pauli_max_weight=2)
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("psi_model", "restart_records")},
                     indent=2))
