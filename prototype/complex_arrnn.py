"""Complex-amplitude AR-RNN for learning RCS with phases.

ψ(x) = sqrt(p(x)) · exp(i φ(x))
  - p(x) factorized via AR + softmax (same as BitstringARRNN)
  - φ(x) from a pooled-feature MLP (analogous to the sign head, but
    outputting a continuous phase in radians)

Training: random-Pauli shadow loss on even-Y Hermitian Paulis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_BOS = 2


class ComplexARRNN(nn.Module):
    def __init__(self, n_qubits, d_hidden=64, d_phase_hidden=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_hidden = d_hidden
        self.emb = nn.Embedding(3, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, 2)
        d_ph = d_phase_hidden or 4 * d_hidden
        self.phase_head = nn.Sequential(
            nn.Linear(2 * d_hidden, d_ph), nn.ReLU(),
            nn.Linear(d_ph, d_ph), nn.ReLU(),
            nn.Linear(d_ph, 1),
        )

    def _shifted(self, x):
        bos = torch.full((x.shape[0], 1), _BOS, dtype=torch.long, device=x.device)
        return torch.cat([bos, x[:, :-1]], dim=1)

    def _features_and_logq(self, x):
        feat = self.gru(self.emb(self._shifted(x)))[0]
        logits = self.head(feat)
        log_p = -F.cross_entropy(logits.transpose(1, 2), x, reduction="none").sum(dim=1)
        return feat, log_p

    def psi(self, x, use_phase=True):
        feat, log_p = self._features_and_logq(x)
        mag = torch.exp(0.5 * log_p)
        if not use_phase:
            return mag.to(torch.complex128)
        pooled = torch.cat([feat[:, -1, :], feat.mean(dim=1)], dim=-1)
        phi = self.phase_head(pooled).squeeze(-1)
        mag_c = mag.to(torch.complex128)
        phase = torch.exp(1j * phi.to(torch.complex128))
        return mag_c * phase


class ARPhaseComplexARRNN(nn.Module):
    """ComplexARRNN with AR-factorized phase: phi(x) = sum_i phi_i(x_i | x_{<i})
    via per-position phase heads consuming the same GRU hidden states as the
    magnitude AR factorization.

    Compared to ComplexARRNN, this:
      - replaces the pooled phase MLP with a per-position nn.Linear(d_hidden, 2)
      - sums per-position phase contributions to form the total phase
      - provides gradient flow on phase at every AR step (n grads per bitstring
        instead of 1)
    """

    def __init__(self, n_qubits, d_hidden=64):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_hidden = d_hidden
        self.emb = nn.Embedding(3, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, 2)
        self.phase_head = nn.Linear(d_hidden, 2)  # per-position phase, one per bit value

    def _shifted(self, x):
        bos = torch.full((x.shape[0], 1), _BOS, dtype=torch.long, device=x.device)
        return torch.cat([bos, x[:, :-1]], dim=1)

    def _features_and_logq(self, x):
        feat = self.gru(self.emb(self._shifted(x)))[0]
        logits = self.head(feat)
        log_p = -F.cross_entropy(logits.transpose(1, 2), x, reduction="none").sum(dim=1)
        return feat, log_p

    def psi(self, x, use_phase=True):
        feat, log_p = self._features_and_logq(x)
        mag = torch.exp(0.5 * log_p)
        if not use_phase:
            return mag.to(torch.complex128)
        # Per-position phase logits (B, N, 2), pick the one matching actual x_i
        phase_logits = self.phase_head(feat)
        chosen = phase_logits.gather(-1, x.unsqueeze(-1)).squeeze(-1)  # (B, N)
        phi = chosen.sum(dim=1)  # (B,)
        mag_c = mag.to(torch.complex128)
        return mag_c * torch.exp(1j * phi.to(torch.complex128))


def model_expectations_complex(psi_c, ops):
    """⟨P⟩_θ = Σ_t ψ_{I_t}^* · C_t · ψ_{J_t}, grouped by Pauli index.
    Returns real Hermitian expectations (.real of complex sum)."""
    psi_c = psi_c.to(torch.complex128)
    C = ops["C"].to(torch.complex128) if torch.is_complex(ops["C"]) else ops["C"]
    products = psi_c[ops["I"]].conj() * C * psi_c[ops["J"]]
    out = torch.zeros(ops["n_paulis"], dtype=torch.complex128, device=psi_c.device)
    out.index_add_(0, ops["P"], products)
    return out.real


def _h_gate():
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def _sh_gate():
    H = _h_gate()
    Sdag = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    return H @ Sdag  # H Sdag rotates Y eigenbasis to Z eigenbasis


def _apply_1q_msb(psi, U, q, n):
    """Apply 1-qubit gate U to qubit q. Uses LSB convention (qubit q ↔
    bit position q of state index), matching shadows.py's analytical
    Pauli formula and sample_shadows_random_pauli — NOT cirq's MSB
    convention. Name kept for backward compatibility, but the convention
    is LSB."""
    psi_t = psi.reshape((2,) * n)
    axis = n - 1 - q
    psi_t = np.moveaxis(psi_t, axis, 0)
    psi_t = np.einsum("ij,j...->i...", U, psi_t)
    psi_t = np.moveaxis(psi_t, 0, axis)
    return psi_t.reshape(-1)


def sample_shadows_random_pauli_complex(psi_full, n, k, rng):
    """Random-Pauli classical-shadow sampling from a complex state.

    Returns (U_pattern, b_out):
      U_pattern[t, q] ∈ {0, 1, 2} = Z, X, Y (Pauli basis measured on qubit q in shot t)
      b_out[t, q] ∈ {0, 1} (measurement outcome). MSB-first convention.

    NOTE: code values match build_loss_paulis_full's convention
    (Z=0, X=1, Y=2) — DO NOT swap.
    """
    D = 1 << n
    psi_full = np.asarray(psi_full, dtype=np.complex128)
    H, SH = _h_gate(), _sh_gate()
    U_pattern = rng.integers(0, 3, size=(k, n), dtype=np.int64)
    b_out = np.zeros((k, n), dtype=np.int64)
    for t in range(k):
        psi_rot = psi_full.copy()
        for q in range(n):
            u_code = int(U_pattern[t, q])
            if u_code == 1:    # X basis → apply H
                psi_rot = _apply_1q_msb(psi_rot, H, q, n)
            elif u_code == 2:  # Y basis → apply HS†
                psi_rot = _apply_1q_msb(psi_rot, SH, q, n)
            # u_code == 0: Z basis, no rotation
        p = np.abs(psi_rot) ** 2
        p /= p.sum() or 1.0
        idx = int(rng.choice(D, p=p))
        # LSB-first bit extraction: bit at position q = (idx >> q) & 1
        for q in range(n):
            b_out[t, q] = (idx >> q) & 1
    return U_pattern, b_out


def train_complex_pauli_loss(model, ops, targets_t, alpha_t, x_bits_t, *,
                              epochs=2000, lr=1e-3, verbose=False, log_every=200):
    """Adam on squared-error loss between model Pauli expectations (complex AR-RNN)
    and shadow targets."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr / 100)
    for ep in range(epochs):
        psi = model.psi(x_bits_t, use_phase=True)
        nrm = (psi.real ** 2 + psi.imag ** 2).sum().sqrt()
        psi = psi / (nrm + 1e-30)
        exps = model_expectations_complex(psi, ops)
        diff = exps - targets_t
        loss = (alpha_t * diff.pow(2)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  ep {ep}: loss={float(loss):.4e}", flush=True)
    return float(loss)
