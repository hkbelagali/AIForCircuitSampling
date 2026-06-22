"""Ryan's AutoregressiveRNN — LSTM-based AR model over binary strings.

Direct port of cell 14 of `notebooks/rcs_ml_experiment.ipynb`. The model
factorises p(x) = p(x_1) · p(x_2|x_1) ⋯ p(x_n|x_1,…,x_{n-1}) via a
two-layer LSTM with a single-output linear head producing per-position
logits.

This is the model used for both NLL training (`aics.training.nll`) and
Z-observable training (`aics.training.z_pauli`). The older `BitstringARRNN`
from `prototype/rcs.py` is retired.

Bit convention: model emits / consumes (B, n) float arrays in MSB-first
qubit order (qubits[0] = bit 0 = first emitted bit), matching
`aics.io.conventions`.
"""
import numpy as np
import torch
import torch.nn as nn


class AutoregressiveRNN(nn.Module):
    """LSTM autoregressive model.

    Parameters
    ----------
    n_bits : int
        Number of qubits / bitstring length.
    hidden : int, optional
        LSTM hidden width. Ryan's default = 128.
    n_layers : int, optional
        Number of LSTM layers. Ryan's default = 2.
    """

    def __init__(self, n_bits: int, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.n_bits = n_bits
        self.lstm = nn.LSTM(1, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_bits) → logits: (B, n_bits)."""
        bsz = x.shape[0]
        sos = torch.zeros(bsz, 1, 1, device=x.device, dtype=x.dtype)
        inp = torch.cat([sos, x[:, :-1].unsqueeze(-1)], dim=1)
        out, _ = self.lstm(inp)
        return self.head(out).squeeze(-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of per-bit log-probs → log p(x). Shape: (B,)."""
        logits = self.forward(x)
        return -nn.functional.binary_cross_entropy_with_logits(
            logits, x, reduction="none").sum(dim=1)

    @torch.no_grad()
    def sample_bits(self, n: int) -> np.ndarray:
        """Autoregressive sampling — one bit at a time. Returns (n, n_bits)
        numpy float array in {0.0, 1.0}."""
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
    def full_distribution(self, n_states: int = None,
                            n_bits: int = None) -> np.ndarray:
        """p(x) for all 2^n_bits bitstrings (MSB-first). Only feasible
        for small n. n_states/n_bits args kept for back-compat — both
        default to the model's n_bits.
        """
        if n_bits is None:
            n_bits = self.n_bits
        if n_states is None:
            n_states = 1 << n_bits
        device = next(self.parameters()).device
        all_bits = np.array(
            [[(i >> (n_bits - 1 - q)) & 1 for q in range(n_bits)]
             for i in range(n_states)],
            dtype=np.float32,
        )
        x = torch.from_numpy(all_bits).to(device)
        lp = self.log_prob(x).cpu().numpy()
        p = np.exp(lp - lp.max())
        return p / p.sum()
