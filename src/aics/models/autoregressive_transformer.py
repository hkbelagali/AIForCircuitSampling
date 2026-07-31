"""Causal self-attention autoregressive model over bitstrings.
"""
import numpy as np
import torch
import torch.nn as nn


class AutoregressiveTransformer(nn.Module):
    def __init__(self, n_bits, d_model=128, n_layers=2, n_heads=4, dim_ff=None,
                 dropout=0.0):
        super().__init__()
        self.n_bits = n_bits
        self.d_model = d_model
        if dim_ff is None:
            dim_ff = 4 * d_model
        self.in_proj = nn.Linear(1, d_model)
        self.pos_embed = nn.Embedding(n_bits, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.full((n_bits, n_bits), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        """(B, n_bits) → (B, n_bits) logits."""
        bsz = x.shape[0]
        sos = torch.zeros(bsz, 1, 1, device=x.device, dtype=x.dtype)
        inp = torch.cat([sos, x[:, :-1].unsqueeze(-1)], dim=1)
        positions = torch.arange(self.n_bits, device=x.device)
        h = self.in_proj(inp) + self.pos_embed(positions)
        h = self.encoder(h, mask=self._causal_mask, is_causal=True)
        return self.head(h).squeeze(-1)

    def log_prob(self, x):
        logits = self.forward(x)
        return -nn.functional.binary_cross_entropy_with_logits(
            logits, x, reduction="none").sum(dim=1)

    @torch.no_grad()
    def sample_bits(self, n_samples):
        """n_samples bitstrings drawn autoregressively.

        Returns (n_samples, n_bits) numpy float in {0., 1.}.
        """
        device = next(self.parameters()).device
        samples = torch.zeros(n_samples, self.n_bits, device=device)
        for i in range(self.n_bits):
            logits = self.forward(samples)[:, i]
            samples[:, i] = torch.bernoulli(torch.sigmoid(logits))
        return samples.cpu().numpy()

    @torch.no_grad()
    def full_distribution(self, n_states=None, n_bits=None):
        """p(x) over all 2^n bitstrings, MSB-first. Tractable for small n."""
        if n_bits is None:
            n_bits = self.n_bits
        if n_states is None:
            n_states = 1 << n_bits
        device = next(self.parameters()).device
        all_bits = np.array(
            [[(i >> (n_bits - 1 - q)) & 1 for q in range(n_bits)]
             for i in range(n_states)], dtype=np.float32)
        lp = self.log_prob(torch.from_numpy(all_bits).to(device)).cpu().numpy()
        p = np.exp(lp - lp.max())
        return p / p.sum()
