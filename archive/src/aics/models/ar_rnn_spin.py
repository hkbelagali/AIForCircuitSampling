"""AR-RNN over L-bit spin states with a single S^z = 0 sector mask.

Mirrors aics.models.ar_rnn.ARRNN but with n_positions = L (not 2L) and a
single particle-count constraint (N_up = L/2) instead of independent up/down
sector budgets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aics.models.ar_transformer import _HandAdam


BOS_TOKEN = 2


class ARRNNSpin(nn.Module):
    def __init__(self, L, n_up, d_hidden=None, n_layers=1, dropout=0.0):
        super().__init__()
        self.L = L
        self.n_positions = L
        self.n_up = n_up
        if d_hidden is None:
            d_hidden = max(2 * L, 32)
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.tok_embed = nn.Embedding(3, d_hidden)
        self.gru = nn.GRU(
            d_hidden, d_hidden, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(d_hidden, 2)

    def _shifted_input(self, x):
        bos = torch.full((x.shape[0], 1), BOS_TOKEN, dtype=torch.long, device=x.device)
        return torch.cat([bos, x[:, :-1]], dim=1)

    def _sector_mask(self, x):
        """(B, L, 2) mask: True = token allowed given remaining up-spin budget."""
        B = x.shape[0]
        L = self.L
        device = x.device
        zero_col = torch.zeros(B, 1, dtype=torch.long, device=device)
        up_so_far = torch.cat([zero_col, x.long().cumsum(dim=1)[:, :-1]], dim=1)
        pos = torch.arange(L, device=device)
        remaining = (L - pos).unsqueeze(0)
        budget = self.n_up - up_so_far
        can_zero = (budget <= remaining - 1) & (budget >= 0)
        can_one = (budget >= 1) & (budget - 1 <= remaining - 1)
        mask = torch.zeros(B, L, 2, dtype=torch.bool, device=device)
        mask[:, :, 0] = can_zero
        mask[:, :, 1] = can_one
        return mask

    def forward(self, x_shifted):
        h_seq = self.tok_embed(x_shifted)
        out, _ = self.gru(h_seq)
        return self.head(out)

    def log_prob(self, x, apply_mask=True):
        logits = self.forward(self._shifted_input(x))
        if apply_mask:
            mask = self._sector_mask(x)
            logits = logits.masked_fill(~mask, float("-inf"))
        per_pos = -F.cross_entropy(logits.transpose(1, 2), x, reduction="none")
        return per_pos.sum(dim=1)

    def log_psi_mag(self, x):
        return 0.5 * self.log_prob(x)

    @torch.no_grad()
    def sample(self, n_samples, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        device = next(self.parameters()).device
        B = n_samples
        x = torch.zeros(B, self.n_positions, dtype=torch.long, device=device)
        h = None
        cur_tok = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
        for i in range(self.n_positions):
            inp_emb = self.tok_embed(cur_tok)
            out, h = self.gru(inp_emb, h)
            logits = self.head(out[:, 0, :]) / temperature
            mask = self._sector_mask(x)
            logits = logits.masked_fill(~mask[:, i, :], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            x[:, i] = sampled
            cur_tok = sampled.unsqueeze(-1)
        return x


__all__ = ["ARRNNSpin", "_HandAdam"]
