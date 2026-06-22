"""LSTM autoregressive model over bitstrings. Cell 14 of Ryan's notebook."""
import numpy as np
import torch
import torch.nn as nn


class AutoregressiveRNN(nn.Module):
    def __init__(self, n_bits, hidden=128, n_layers=2):
        super().__init__()
        self.n_bits = n_bits
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        """(B, n_bits) → (B, n_bits) logits."""
        bsz = x.shape[0]
        sos = torch.zeros(bsz, 1, 1, device=x.device, dtype=x.dtype)
        inp = torch.cat([sos, x[:, :-1].unsqueeze(-1)], dim=1)
        out, _ = self.lstm(inp)
        return self.head(out).squeeze(-1)

    def log_prob(self, x):
        logits = self.forward(x)
        return -nn.functional.binary_cross_entropy_with_logits(
            logits, x, reduction="none").sum(dim=1)

    @torch.no_grad()
    def sample_bits(self, n):
        """n bitstrings drawn autoregressively. Returns (n, n_bits) numpy {0,1}."""
        device = next(self.parameters()).device
        samples = torch.zeros(n, self.n_bits, device=device)
        h = None
        inp = torch.zeros(n, 1, 1, device=device)
        for i in range(self.n_bits):
            out, h = self.lstm(inp, h)
            prob = torch.sigmoid(self.head(out.squeeze(1)))
            bit = torch.bernoulli(prob)
            samples[:, i] = bit.squeeze(1)
            inp = bit.unsqueeze(1)
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
