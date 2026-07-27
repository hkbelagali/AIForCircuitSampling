"""NLL training of AutoregressiveRNN on bitstring samples.

  loss = - mean log q(z)  +  λ · pt_term(log q)    (if λ > 0)

Resume / checkpoint: pass `resume_from` to pick up model + optimizer +
scheduler + epoch from a prior checkpoint; pass `checkpoint_to` (and
`checkpoint_every`) to write the same state periodically so a SLURM
cancel can be recovered cleanly.
"""
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..runtime import load_checkpoint, save_checkpoint
from .pt_regularizer import pt_term


def _log_prob_via_forward(model, batch):
    """log q(z) routed through forward + BCE so torch.compile(model.forward)
    actually accelerates training. Calling model.log_prob directly bypasses
    the compiled __call__ path via __getattr__ and produces no speedup.
    """
    logits = model(batch)
    return -F.binary_cross_entropy_with_logits(
        logits, batch, reduction="none").sum(dim=1)

# Ryan's defaults (rcs_ml_experiment.ipynb cell 2).
BATCH_SIZE = 512
TOTAL_STEPS = 50_000
MIN_EPOCHS = 50
MAX_EPOCHS = 5_000
LAMBDA_PT = 0.01


def train_nll(model, train_bits, total_steps=TOTAL_STEPS,
               min_epochs=MIN_EPOCHS, max_epochs=MAX_EPOCHS,
               batch_size=BATCH_SIZE, lr=1e-3, lambda_pt=LAMBDA_PT,
               n_states=None, device="cpu", verbose=False,
               logger=None, clip_grad=1.0,
               resume_from=None, checkpoint_to=None, checkpoint_every=50):
    """train_bits (k, n_qubits) float, MSB-first. Returns (final_nll, n_epochs).

    lambda_pt = 0 disables the Porter-Thomas regulariser.
    resume_from: path to a checkpoint produced by save_checkpoint; loads
                 model, optimizer, scheduler, and resumes at saved epoch.
    checkpoint_to: path to write the same state every `checkpoint_every`
                   epochs and at the end.
    """
    n_bits = train_bits.shape[1]
    if lambda_pt > 0 and n_states is None:
        n_states = 1 << n_bits

    on_cuda = device.startswith("cuda")
    if on_cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # drop_last on CUDA gives torch.compile / CUDA graphs a static batch shape.
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_bits.astype(np.float32))),
        batch_size=min(batch_size, len(train_bits)),
        shuffle=True, drop_last=on_cuda)
    steps_per_epoch = max(1, len(train_bits) // batch_size)
    n_epochs = min(max_epochs, max(min_epochs, total_steps // steps_per_epoch))

    # Compile the forward path (model.log_prob bypasses __call__ via __getattr__,
    # so wrapping the module was a no-op — swap in a compiled forward instead
    # and route the loop through _log_prob_via_forward).
    if on_cuda:
        try:
            model.forward = torch.compile(model.forward, mode="reduce-overhead")
            if verbose:
                print("using torch.compile(reduce-overhead)", flush=True)
        except Exception as e:
            if verbose:
                print(f"torch.compile unavailable: {e}", flush=True)

    try:
        optimizer = optim.Adam(model.parameters(), lr=lr, fused=on_cuda)
    except TypeError:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    start_epoch = 0
    last_nll = float("nan")

    use_bf16 = on_cuda and torch.cuda.is_bf16_supported()
    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                     if use_bf16 else contextlib.nullcontext())

    if resume_from:
        payload = load_checkpoint(resume_from, model, optimizer, scheduler,
                                    map_location=device)
        start_epoch = int(payload.get("epoch") or 0)
        last_nll = float(payload.get("best_loss") or float("nan"))

    for ep in range(start_epoch, n_epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            with autocast_ctx:
                log_q = _log_prob_via_forward(model, batch)
                nll = -log_q.mean()
                loss = nll + lambda_pt * pt_term(log_q, n_states) if lambda_pt > 0 else nll
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            ep_loss += float(nll.detach())
            n_batches += 1
        scheduler.step()
        last_nll = ep_loss / max(1, n_batches)
        if verbose and (ep % 50 == 0 or ep == n_epochs - 1):
            print(f"  ep {ep:>4}/{n_epochs}: NLL = {last_nll:.4f}", flush=True)
        if logger is not None:
            logger.log(stage="nll", epoch=ep, n_epochs=n_epochs,
                        avg_nll=last_nll, lambda_pt=lambda_pt)
        if checkpoint_to and (ep % checkpoint_every == 0 or ep == n_epochs - 1):
            save_checkpoint(checkpoint_to, model, optimizer, scheduler,
                              epoch=ep + 1, best_loss=last_nll)
    return last_nll, n_epochs
