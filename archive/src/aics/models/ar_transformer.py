"""Small autoregressive transformer over {0, 1}^n bitstrings.

Factorization: q(x) = prod_i q(x_i | x_<i). At position i the model sees
[BOS, x_0, ..., x_{i-1}] (left-shifted by one) and outputs logits over {0, 1}.
Causal mask on the transformer encoder enforces the prefix dependence.

Used in M2 for Stage 3 RCS: unconstrained binary generation (RCS has no
symmetry sector to project to). Kept deliberately small (~2 layers, ~64 dim)
to test the strict "novel high-p_C" question, not to bake in capacity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# `torch.optim.Adam` triggers torch._dynamo, which imports a symbol from sympy
# 1.13+ that this env's sympy 1.4 doesn't have. Rather than touch the shared
# env, we use a minimal hand-rolled Adam (pure tensor ops, no dynamo).
class _HandAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd:
                g = g.add(p, alpha=self.wd)
            m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            denom = (v.sqrt() / (bc2 ** 0.5)).add_(self.eps)
            p.addcdiv_(m, denom, value=-self.lr / bc1)


BOS_TOKEN = 2


class ARTransformer(nn.Module):
    def __init__(self, n_qubits, d_model=64, n_layers=2, n_heads=4, dim_ff=None,
                 dropout=0.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_model = d_model
        if dim_ff is None:
            dim_ff = 4 * d_model
        self.tok_embed = nn.Embedding(3, d_model)
        self.pos_embed = nn.Embedding(n_qubits, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)

    def _shifted_input(self, x):
        bos = torch.full((x.shape[0], 1), BOS_TOKEN, dtype=torch.long,
                         device=x.device)
        return torch.cat([bos, x[:, :-1]], dim=1)

    def forward(self, x_shifted):
        """x_shifted: (B, n) with BOS at position 0. Returns (B, n, 2) logits."""
        B, n = x_shifted.shape
        positions = torch.arange(n, device=x_shifted.device).expand(B, -1)
        h = self.tok_embed(x_shifted) + self.pos_embed(positions)
        mask = torch.triu(
            torch.full((n, n), float("-inf"), device=x_shifted.device),
            diagonal=1,
        )
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(h)

    def log_prob(self, x):
        """Joint log-likelihood log q(x) for x in (B, n) with values in {0, 1}."""
        logits = self.forward(self._shifted_input(x))
        per_pos = -F.cross_entropy(logits.transpose(1, 2), x, reduction="none")
        return per_pos.sum(dim=1)

    @torch.no_grad()
    def sample(self, n_samples, temperature=1.0, device=None):
        """Autoregressive samples of shape (n_samples, n) with values in {0, 1}.

        `temperature` rescales logits before softmax: T > 1 broadens the
        per-position distribution and increases diversity; T < 1 sharpens it.
        T = 1 is true model sampling.
        """
        if device is None:
            device = next(self.parameters()).device
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        x = torch.zeros(n_samples, self.n_qubits, dtype=torch.long, device=device)
        for i in range(self.n_qubits):
            logits = self.forward(self._shifted_input(x))[:, i, :] / temperature
            probs = F.softmax(logits, dim=-1)
            x[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        return x


def train_ar(model, X_train, n_epochs=200, lr=1e-3, batch_size=64,
             device="cpu", verbose=False, log_every=25):
    """Adam-train the AR transformer to maximize log-likelihood of X_train.

    X_train: (k, n) numpy array or tensor of {0, 1} values. Returns final mean NLL.
    """
    if isinstance(X_train, np.ndarray):
        X_train = torch.from_numpy(X_train.astype(np.int64))
    X_train = X_train.long().to(device)
    model.to(device).train()
    opt = _HandAdam(list(model.parameters()), lr=lr)
    k = X_train.shape[0]
    final_nll = float("nan")
    for ep in range(n_epochs):
        perm = torch.randperm(k, device=device)
        total = 0.0
        for s in range(0, k, batch_size):
            batch = X_train[perm[s:s + batch_size]]
            log_p = model.log_prob(batch)
            loss = -log_p.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += -float(log_p.sum().item())
        final_nll = total / k
        if verbose and (ep == 0 or (ep + 1) % log_every == 0 or ep == n_epochs - 1):
            print(f"    epoch {ep+1:3d}/{n_epochs}: mean NLL = {final_nll:.4f}")
    return final_nll


# ----------------------------------------------------------------------------
# Conditional AR transformer with (N_up, N_dn) sector mask (used in M3).
# Conditioning is a continuous scalar c (e.g., Hubbard U/t) embedded via an
# MLP and added to all positional embeddings. The per-position sector mask
# enforces exact particle number per spin during sampling.
# State convention: positions 0..L-1 are up bits (LSB-first), positions
# L..2L-1 are dn bits.
# ----------------------------------------------------------------------------


class ARTransformerConditional(nn.Module):
    def __init__(self, L, n_up, n_dn, d_model=64, n_layers=2, n_heads=4,
                 dim_ff=None, dropout=0.0):
        super().__init__()
        self.L = L
        self.n_positions = 2 * L
        self.n_up = n_up
        self.n_dn = n_dn
        self.d_model = d_model
        if dim_ff is None:
            dim_ff = 4 * d_model
        self.tok_embed = nn.Embedding(3, d_model)
        self.pos_embed = nn.Embedding(self.n_positions, d_model)
        self.cond_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 2)

    def _shifted_input(self, x):
        bos = torch.full((x.shape[0], 1), BOS_TOKEN, dtype=torch.long,
                         device=x.device)
        return torch.cat([bos, x[:, :-1]], dim=1)

    def forward(self, x_shifted, c):
        """x_shifted: (B, 2L) with BOS at position 0. c: (B,) scalar conditioning."""
        B, n = x_shifted.shape
        if c.dim() == 1:
            c_in = c.unsqueeze(-1).float()
        else:
            c_in = c.float()
        c_emb = self.cond_mlp(c_in)                      # (B, d_model)
        positions = torch.arange(n, device=x_shifted.device).expand(B, -1)
        h = (self.tok_embed(x_shifted) + self.pos_embed(positions)
             + c_emb.unsqueeze(1))                       # broadcast c over positions
        mask = torch.triu(
            torch.full((n, n), float("-inf"), device=x_shifted.device),
            diagonal=1,
        )
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(h)

    def _sector_mask(self, x):
        """Per-position (B, 2L, 2) validity mask given prefix x.

        At position i with cumulative count up_so_far in {0, ..., N_up} and
        remaining = L - i positions to fill in the spin sector, token t is
        valid iff:
          (N_sigma - up_so_far - t) is in [0, remaining - 1]
        i.e., the remaining positions can still satisfy the particle count.
        """
        B = x.shape[0]
        L = self.L
        device = x.device

        up_part = x[:, :L].long()
        dn_part = x[:, L:].long()
        zero_col = torch.zeros(B, 1, dtype=torch.long, device=device)
        up_so_far = torch.cat([zero_col, up_part.cumsum(dim=1)[:, :-1]], dim=1)
        dn_so_far = torch.cat([zero_col, dn_part.cumsum(dim=1)[:, :-1]], dim=1)

        pos = torch.arange(L, device=device)
        remaining = (L - pos).unsqueeze(0)               # (1, L)

        budget_up = self.n_up - up_so_far                # (B, L)
        can_zero_up = (budget_up <= remaining - 1) & (budget_up >= 0)
        can_one_up = (budget_up >= 1) & (budget_up - 1 <= remaining - 1)

        budget_dn = self.n_dn - dn_so_far
        can_zero_dn = (budget_dn <= remaining - 1) & (budget_dn >= 0)
        can_one_dn = (budget_dn >= 1) & (budget_dn - 1 <= remaining - 1)

        mask = torch.zeros(B, 2 * L, 2, dtype=torch.bool, device=device)
        mask[:, :L, 0] = can_zero_up
        mask[:, :L, 1] = can_one_up
        mask[:, L:, 0] = can_zero_dn
        mask[:, L:, 1] = can_one_dn
        return mask

    def log_prob(self, x, c, apply_mask=True):
        logits = self.forward(self._shifted_input(x), c)
        if apply_mask:
            mask = self._sector_mask(x)
            logits = logits.masked_fill(~mask, float("-inf"))
        per_pos = -F.cross_entropy(logits.transpose(1, 2), x, reduction="none")
        return per_pos.sum(dim=1)

    @torch.no_grad()
    def sample(self, c, temperature=1.0):
        """c: (B,) tensor of conditioning values, one per sample.
        Returns (B, 2L) long tensor in {0, 1}.
        """
        device = c.device if torch.is_tensor(c) else next(self.parameters()).device
        if not torch.is_tensor(c):
            c = torch.tensor([float(c)], device=device)
        B = c.shape[0]
        x = torch.zeros(B, self.n_positions, dtype=torch.long, device=device)
        for i in range(self.n_positions):
            mask = self._sector_mask(x)
            logits = self.forward(self._shifted_input(x), c)[:, i, :] / temperature
            logits = logits.masked_fill(~mask[:, i, :], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            x[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        return x


def train_ar_conditional(model, X_train, c_train, n_epochs=100, lr=2e-3,
                         batch_size=32, device="cpu", verbose=False, log_every=25):
    """Joint training of a conditional AR. X_train: (k, 2L) bits; c_train: (k,)."""
    if isinstance(X_train, np.ndarray):
        X_train = torch.from_numpy(X_train.astype(np.int64))
    if isinstance(c_train, np.ndarray):
        c_train = torch.from_numpy(c_train.astype(np.float32))
    X_train = X_train.long().to(device)
    c_train = c_train.float().to(device)
    model.to(device).train()
    opt = _HandAdam(list(model.parameters()), lr=lr)
    k = X_train.shape[0]
    final_nll = float("nan")
    for ep in range(n_epochs):
        perm = torch.randperm(k, device=device)
        total = 0.0
        for s in range(0, k, batch_size):
            idx = perm[s:s + batch_size]
            log_p = model.log_prob(X_train[idx], c_train[idx])
            loss = -log_p.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += -float(log_p.sum().item())
        final_nll = total / k
        if verbose and (ep == 0 or (ep + 1) % log_every == 0 or ep == n_epochs - 1):
            print(f"    epoch {ep+1:3d}/{n_epochs}: mean NLL = {final_nll:.4f}",
                  flush=True)
    return final_nll
