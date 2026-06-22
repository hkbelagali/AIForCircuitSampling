"""NLL training of AutoregressiveRNN on bitstring samples.

  loss = - mean log q(z)  +  λ · pt_term(log q)    (if λ > 0)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .pt_regularizer import pt_term

# Ryan's defaults (rcs_ml_experiment.ipynb cell 2).
BATCH_SIZE = 512
TOTAL_STEPS = 50_000
MIN_EPOCHS = 50
MAX_EPOCHS = 5_000
LAMBDA_PT = 0.01


def train_nll_pt(model, train_bits, total_steps=TOTAL_STEPS,
                  min_epochs=MIN_EPOCHS, max_epochs=MAX_EPOCHS,
                  batch_size=BATCH_SIZE, lr=1e-3, lambda_pt=LAMBDA_PT,
                  n_states=None, device="cpu", verbose=False,
                  logger=None, clip_grad=1.0):
    """train_bits (k, n) float, MSB-first. Returns (final_nll, n_epochs)."""
    n_bits = train_bits.shape[1]
    if lambda_pt > 0 and n_states is None:
        n_states = 1 << n_bits
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_bits.astype(np.float32))),
        batch_size=min(batch_size, len(train_bits)), shuffle=True)
    steps_per_epoch = max(1, len(train_bits) // batch_size)
    n_epochs = min(max_epochs, max(min_epochs, total_steps // steps_per_epoch))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    last_nll = float("nan")
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            log_q = model.log_prob(batch)
            nll = -log_q.mean()
            loss = nll + lambda_pt * pt_term(log_q, n_states) if lambda_pt > 0 else nll
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            ep_loss += float(nll.detach())
        scheduler.step()
        last_nll = ep_loss / len(loader)
        if verbose and (ep % 50 == 0 or ep == n_epochs - 1):
            print(f"  ep {ep:>4}/{n_epochs}: NLL = {last_nll:.4f}", flush=True)
        if logger is not None:
            logger.log(stage="nll", epoch=ep, n_epochs=n_epochs,
                        avg_nll=last_nll, lambda_pt=lambda_pt)
    return last_nll, n_epochs
