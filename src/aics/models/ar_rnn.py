"""Autoregressive RNN wavefunction over {0, 1}^(2L) with (N_up, N_dn) sector mask.

Architecture (Moss et al. arXiv:2308.02647 style): single-layer GRU, hidden
dim = max(2 * 2L, 32). Embeddings for {0, 1, BOS}, GRU rolls left-to-right over
2L positions (up then dn, LSB-first per the state convention in
aics.common.symmetry). Output head produces logits for each of the 2 possible
tokens; the sector mask zeros out tokens that would violate the per-spin
particle-count budget at the current position.

Born rule: q(x) = exp(log_prob(x)) is the joint probability over the sector.
|psi(x)| = sqrt(q(x)); the sign is applied externally via the Marshall sign
rule (exact for half-filled bipartite Hubbard; see aics.chemistry.marshall).

Models only the magnitude, not the sign -- the MSR sign has no gradient
contribution to log|psi|, so the AR factorization gives access to per-sample
score functions cleanly through log_prob alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aics.models.ar_transformer import _HandAdam  # reuse our Adam (sympy/torch workaround)


BOS_TOKEN = 2


class ARRNN(nn.Module):
    def __init__(self, L, n_up, n_dn, d_hidden=None, n_layers=1, dropout=0.0):
        super().__init__()
        self.L = L
        self.n_positions = 2 * L
        self.n_up = n_up
        self.n_dn = n_dn
        if d_hidden is None:
            d_hidden = max(2 * self.n_positions, 32)
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.tok_embed = nn.Embedding(3, d_hidden)
        self.gru = nn.GRU(
            d_hidden, d_hidden, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(d_hidden, 2)

    # ---- shifting / masking (sector-mask logic mirrors ARTransformerConditional) --

    def _shifted_input(self, x):
        bos = torch.full((x.shape[0], 1), BOS_TOKEN, dtype=torch.long, device=x.device)
        return torch.cat([bos, x[:, :-1]], dim=1)

    def _sector_mask(self, x):
        """Per-position (B, 2L, 2) bool mask: True = token allowed under the
        remaining (N_up, N_dn) budget given the prefix in x.

        At position i (in the up block, i < L), budget_up = N_up - up_so_far,
        remaining = L - i. A token t is valid iff
            (budget_up - t) in [0, remaining - 1]
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
        remaining = (L - pos).unsqueeze(0)

        budget_up = self.n_up - up_so_far
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

    # ---- forward / log-likelihood -----------------------------------------------

    def forward(self, x_shifted):
        """x_shifted: (B, 2L) long tokens (BOS at position 0). Returns (B, 2L, 2) logits."""
        h_seq = self.tok_embed(x_shifted)
        out, _ = self.gru(h_seq)
        return self.head(out)

    def log_prob(self, x, apply_mask=True):
        """log q(x) = log |psi(x)|^2 with (N_up, N_dn) sector enforcement."""
        logits = self.forward(self._shifted_input(x))
        if apply_mask:
            mask = self._sector_mask(x)
            logits = logits.masked_fill(~mask, float("-inf"))
        per_pos = -F.cross_entropy(logits.transpose(1, 2), x, reduction="none")
        return per_pos.sum(dim=1)

    def log_psi_mag(self, x):
        """log |psi(x)| -- half the log probability (sign handled externally via MSR)."""
        return 0.5 * self.log_prob(x)

    # ---- sampling ---------------------------------------------------------------

    @torch.no_grad()
    def sample(self, n_samples, temperature=1.0):
        """Autoregressive samples of shape (n_samples, 2L) from the masked distribution."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        device = next(self.parameters()).device
        B = n_samples
        x = torch.zeros(B, self.n_positions, dtype=torch.long, device=device)
        h = None
        cur_tok = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
        for i in range(self.n_positions):
            inp_emb = self.tok_embed(cur_tok)            # (B, 1, d_hidden)
            out, h = self.gru(inp_emb, h)
            logits = self.head(out[:, 0, :]) / temperature  # (B, 2)
            mask = self._sector_mask(x)
            logits = logits.masked_fill(~mask[:, i, :], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            x[:, i] = sampled
            cur_tok = sampled.unsqueeze(-1)
        return x


# ---- re-export the Adam shim so callers only need one import -------------------

__all__ = ["ARRNN", "_HandAdam"]
