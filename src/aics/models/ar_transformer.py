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
