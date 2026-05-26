"""Exact sample-complexity diagnostics.

Accuracy of a trained generative model q_theta is measured by the forward KL
from the true distribution p:

    D_KL(p || q) = sum_x p(x) (log p(x) - log q(x)) = CE(p, q) - H(p).

For small systems we evaluate q_theta(x) for EVERY basis state, so D_KL is
exact -- no test-set Monte Carlo noise. The capacity floor (best achievable KL
at infinite data, set by model expressiveness) is estimated by training the
same architecture against p directly via weighted (soft-target) MLE.

Bit conventions match the training pipelines:
  - RCS: MSB-first (cirq), via aics.circuits.exact.int_to_bits.
  - Hubbard: LSB-first sector convention, via state_int_to_bits.
"""

import numpy as np
import torch

from aics.chemistry.amplitude_sampling import state_int_to_bits
from aics.circuits.exact import int_to_bits as _int_to_bits_msb
from aics.models.ar_transformer import _HandAdam


def model_log_probs_rcs(model, n, batch=4096, device="cpu"):
    """log q_theta(x) for all 2^n strings (unconstrained AR model)."""
    N = 1 << n
    log_q = np.empty(N, dtype=np.float64)
    model.eval()
    for s in range(0, N, batch):
        ints = np.arange(s, min(s + batch, N))
        bits = _int_to_bits_msb(ints, n)
        with torch.no_grad():
            lp = model.log_prob(torch.from_numpy(bits).long().to(device)).cpu().numpy()
        log_q[s:s + len(ints)] = lp
    return log_q


def model_log_probs_hubbard(model, sector_state_ints, L, c_value, batch=2048,
                            device="cpu"):
    """log q_theta(x) for all sector states (conditional masked AR at c=c_value)."""
    sector_state_ints = np.asarray(sector_state_ints)
    N = len(sector_state_ints)
    log_q = np.empty(N, dtype=np.float64)
    model.eval()
    for s in range(0, N, batch):
        chunk = sector_state_ints[s:s + batch]
        bits = state_int_to_bits(chunk, L)
        bt = torch.from_numpy(bits).long().to(device)
        ct = torch.full((len(chunk),), float(c_value), device=device)
        with torch.no_grad():
            lp = model.log_prob(bt, ct).cpu().numpy()
        log_q[s:s + len(chunk)] = lp
    return log_q


def forward_kl(p, log_q):
    """Exact forward KL and total q-mass (normalization check, should be ~1)."""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    mask = p > 0
    kl = float(np.sum(p[mask] * (np.log(p[mask]) - log_q[mask])))
    q_mass = float(np.exp(log_q).sum())
    return kl, q_mass


def shannon_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def train_softtarget(model, support_bits, p, c_value=None, n_epochs=200,
                     batch=512, lr=2e-3, device="cpu", seed=0):
    """Capacity-floor training: minibatch weighted MLE on the full support.

    Minimizes CE(p, q) = -sum_x p(x) log q(x), i.e. the infinite-data MLE
    objective. The resulting KL is the best the architecture can achieve.
    """
    rng = np.random.default_rng(seed)
    N = len(p)
    bits_t = torch.from_numpy(np.asarray(support_bits, dtype=np.int64)).long().to(device)
    p_t = torch.from_numpy((p / p.sum()).astype(np.float32)).to(device)
    opt = _HandAdam(list(model.parameters()), lr=lr)
    model.train()
    for _ in range(n_epochs):
        perm = rng.permutation(N)
        for s in range(0, N, batch):
            idx = torch.from_numpy(perm[s:s + batch]).long()
            if c_value is None:
                log_q = model.log_prob(bits_t[idx])
            else:
                ct = torch.full((idx.numel(),), float(c_value), device=device)
                log_q = model.log_prob(bits_t[idx], ct)
            w = p_t[idx]
            loss = -(w * log_q).sum() / w.sum().clamp_min(1e-12)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
