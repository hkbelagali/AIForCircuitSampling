"""MLE pretraining for an autoregressive bitstring wavefunction.

Given k samples from |psi_0|^2 (or any target distribution), minimize the
empirical negative log-likelihood:

    L(theta) = - (1/k) sum_i log q_theta(x_i)

This is a simplified mirror of `aics.models.ar_transformer.train_ar_conditional`
without the conditioning argument.
"""

import numpy as np
import torch

from aics.models.ar_transformer import _HandAdam


def train_rnn_mle(model, X_bits, n_epochs=100, lr=2e-3, batch_size=32,
                  device="cpu", verbose=False, log_every=25):
    """Train an ARRNN by MLE on X_bits (k, 2L) bit array.

    Returns the final mean training NLL.
    """
    if isinstance(X_bits, np.ndarray):
        X_bits = torch.from_numpy(X_bits.astype(np.int64))
    X_bits = X_bits.long().to(device)
    model.to(device).train()
    opt = _HandAdam(list(model.parameters()), lr=lr)
    k = X_bits.shape[0]
    final_nll = float("nan")
    for ep in range(n_epochs):
        perm = torch.randperm(k, device=device)
        total = 0.0
        for s in range(0, k, batch_size):
            batch = X_bits[perm[s:s + batch_size]]
            log_p = model.log_prob(batch)
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


def train_rnn_softtarget(model, support_bits, p, n_epochs=200, batch_size=512,
                         lr=2e-3, device="cpu", seed=0):
    """Capacity-floor training: minibatch weighted MLE on the full support
    (= infinite-data MLE objective).

    Minimizes CE(p, q) = -sum_x p(x) log q_theta(x). The resulting KL/TV is the
    best the architecture can achieve, used for the M7-style capacity ceiling.
    """
    rng = np.random.default_rng(seed)
    N = len(p)
    bits_t = torch.from_numpy(np.asarray(support_bits, dtype=np.int64)).long().to(device)
    p_t = torch.from_numpy((p / p.sum()).astype(np.float32)).to(device)
    opt = _HandAdam(list(model.parameters()), lr=lr)
    model.train()
    for _ in range(n_epochs):
        perm = rng.permutation(N)
        for s in range(0, N, batch_size):
            idx = torch.from_numpy(perm[s:s + batch_size]).long()
            log_q = model.log_prob(bits_t[idx])
            w = p_t[idx]
            loss = -(w * log_q).sum() / w.sum().clamp_min(1e-12)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
