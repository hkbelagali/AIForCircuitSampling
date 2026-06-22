"""Autoregressive GRU wavefunction over 2L bits with (N_up, N_dn) sector mask.
Models only |psi(x)|; signs are applied externally via Hubbard.signs.
This is an open issue, still a work in progress.  Initially it appeared that signs
did not need to be learned due to Marshall sign rule, but does not apply to these systems.
Intended fix is to not just gather diagonal (Z-basis) bitstrings, but general bitstrings
a la shadows.  We still demonstrate that we learn probability densities and show a decrease
in complexity, but not enough on its own!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_BOS = 2


class ARRNN(nn.Module):
    def __init__(self, L, n_up, n_dn, d_hidden=32):
        super().__init__()
        self.L = L
        self.n_up = n_up
        self.n_dn = n_dn
        self.emb = nn.Embedding(3, d_hidden)
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, 2)

    def _sector_mask(self, x):
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

    def forward(self, x):
        return self.head(self.gru(self.emb(self._shifted(x)))[0])

    def log_prob(self, x):
        logits = self.forward(x).masked_fill(~self._sector_mask(x), float("-inf"))
        return -F.cross_entropy(logits.transpose(1, 2), x, reduction="none").sum(dim=1)

    def log_psi(self, x):
        return 0.5 * self.log_prob(x)

    @torch.no_grad()
    def sample(self, n):
        dev = next(self.parameters()).device
        x = torch.zeros(n, 2 * self.L, dtype=torch.long, device=dev)
        cur = torch.full((n, 1), _BOS, dtype=torch.long, device=dev)
        h = None
        for i in range(2 * self.L):
            out, h = self.gru(self.emb(cur), h)
            logits = self.head(out[:, 0, :])
            logits = logits.masked_fill(~self._sector_mask(x)[:, i, :], float("-inf"))
            tok = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
            x[:, i] = tok
            cur = tok.unsqueeze(-1)
        return x
