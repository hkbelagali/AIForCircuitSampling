"""Train a model at n=16, snapshot the full q_model distribution at densely
log-spaced steps, plus per-step training metrics. Runs once per pt_on/off.

Output: results/n_dynamics/pt{on|off}.npz containing:
  p_C            (D,)          exact ideal distribution
  train_mask     (D,) bool     bitstring in training set
  filt_mask      (D,) bool     bitstring in held \\ training
  neither_mask   (D,) bool     everything else
  steps          (S,)          gradient step indices
  epochs         (S,)          epoch index at each step
  q_dist         (S, D)        model distribution over the full state space
  train_nll      (S,)          on-the-batch training NLL at each step
  held_nll       (S,)          on held_bits at each step (small, cheap)

D = 2^n = 65536 at n=16. Each snapshot = 262 KB float32. ~500-1000 snapshots
per run keeps memory under ~200 MB.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aics.circuits import make_boixo_v2_rcs_circuit
from aics.circuits.exact import exact_probabilities
from aics.io import load_samples
from aics.io.conventions import bits_to_int
from aics.models import AutoregressiveRNN


LAMBDA_PT = 0.01
BATCH = 512
LR = 1e-3


def build_schedule(total_steps):
    """Log-spaced early + coarser after — avoids paying dense snapshot cost
    every step during the (boring) uniform-init transient."""
    s = set()
    # Log-space the first 100 steps (~15 snapshots instead of 100)
    s.update([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
    # Every 10 for 100-500
    s.update(range(100, 501, 10))
    # Every 25 steps for 500-3000
    s.update(range(500, 3001, 25))
    # Every 100 steps thereafter
    s.update(range(3000, total_steps + 1, 100))
    # Also Fibonacci-like anchors so plots have gentle log-x visualization
    s.update([144, 233, 377, 610, 987, 1597, 2584, 4181, 6765,
              10946, 17711, 28657])
    return sorted(x for x in s if x <= total_steps)


_bits_cache = {}

def snapshot_q_full(model, D, n_bits, device, batch=65536):
    """Return q_model over all 2^n_bits states as a (D,) float32 vector.

    Pre-computes the enumerate-all-bits tensor on GPU once per (D, n_bits, device).
    Keeps computation on GPU, does a single .cpu() at the end.
    """
    key = (D, n_bits, str(device))
    if key not in _bits_cache:
        all_int = np.arange(D, dtype=np.int64)
        bits_np = np.zeros((D, n_bits), dtype=np.float32)
        for i in range(n_bits):
            bits_np[:, i] = ((all_int >> (n_bits - 1 - i)) & 1).astype(np.float32)
        _bits_cache[key] = torch.from_numpy(bits_np).to(device)
    bits_gpu = _bits_cache[key]
    p_gpu = torch.empty(D, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, D, batch):
            end = min(start + batch, D)
            lp = model.log_prob(bits_gpu[start:end])
            p_gpu[start:end] = torch.exp(lp).to(torch.float32)
    return p_gpu.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_npz", required=True)
    ap.add_argument("--k_train", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--model_seed", type=int, default=1)
    ap.add_argument("--use_pt", type=int, choices=[0, 1], required=True,
                     help="1 = pt_regularizer on, 0 = off")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    print(f"device={device}  pt={'on' if args.use_pt else 'off'}", flush=True)

    # Load samples
    data = load_samples(args.samples_npz)
    n = int(data["meta"]["n"]); D = 1 << n
    depth = int(data["meta"]["depth"])
    circuit_seed = int(data["meta"].get("circuit_seed", 42))
    rows = data["meta"].get("rows"); rows = int(rows) if rows else None
    cols = data["meta"].get("cols"); cols = int(cols) if cols else None
    train_bits = np.asarray(data["train_bits"])[:args.k_train]
    held_bits  = np.asarray(data["held_bits"])
    print(f"n={n}  D={D}  k_train={len(train_bits)}  k_held={len(held_bits)}",
           flush=True)

    # Exact p_C for all states. Prefer a bundle-shipped p_C_full (needed for
    # non-RCS circuits like peaked+RCS that can't be reproduced from meta alone).
    # load_samples returns a dict without p_C_full, so peek at the raw npz.
    t0 = time.time()
    with np.load(args.samples_npz, allow_pickle=True) as _raw:
        has_p_C_full = "p_C_full" in _raw.files
        if has_p_C_full:
            p_C_full = np.asarray(_raw["p_C_full"], dtype=np.float64)
    if has_p_C_full:
        print(f"Using bundle p_C_full: sum={p_C_full.sum():.6f} (skipped rebuild)",
               flush=True)
    else:
        print("Computing exact p_C ...", flush=True)
        qubits, circ = make_boixo_v2_rcs_circuit(
            n, depth=depth, seed=circuit_seed, rows=rows, cols=cols
        )
        p_C_full = exact_probabilities(circ, qubits).astype(np.float64)
        print(f"  p_C computed in {time.time() - t0:.1f}s  sum={p_C_full.sum():.6f}",
               flush=True)

    # Region masks
    train_int = bits_to_int(train_bits)
    held_int  = bits_to_int(held_bits)
    train_set = set(int(z) for z in train_int)
    held_set  = set(int(z) for z in held_int)
    train_mask = np.zeros(D, dtype=bool)
    for z in train_set: train_mask[z] = True
    held_mask = np.zeros(D, dtype=bool)
    for z in held_set: held_mask[z] = True
    filt_mask = held_mask & (~train_mask)
    neither_mask = (~held_mask) & (~train_mask)
    print(f"regions:  train={train_mask.sum()}  filt(held\\train)={filt_mask.sum()}  "
           f"neither={neither_mask.sum()}", flush=True)

    # Global perf toggles
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Model + optimizer
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    model = AutoregressiveRNN(n_bits=n, hidden=args.hidden, n_layers=2).to(device)

    # Compile the FORWARD path (not the whole module — log_prob bypasses __call__
    # via __getattr__ so wrapping the module was a no-op). We route the training
    # loss through model.forward directly and compute BCE outside.
    forward_uncompiled = model.forward
    if device.startswith("cuda"):
        try:
            compiled_forward = torch.compile(forward_uncompiled, mode="reduce-overhead")
            print("using torch.compile(model.forward, reduce-overhead)", flush=True)
        except Exception as e:
            compiled_forward = forward_uncompiled
            print(f"torch.compile failed: {e} — running eager", flush=True)
    else:
        compiled_forward = forward_uncompiled

    try:
        optim = torch.optim.Adam(model.parameters(), lr=LR, fused=True)
        print("using fused Adam", flush=True)
    except TypeError:
        optim = torch.optim.Adam(model.parameters(), lr=LR)

    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    print(f"bf16 autocast: {use_bf16}", flush=True)

    # Prepare training tensors — pre-move ENTIRE train set to GPU once
    train_t = torch.from_numpy(train_bits.astype(np.float32)).to(device)
    held_t  = torch.from_numpy(held_bits.astype(np.float32)).to(device)

    steps_per_epoch = max(1, len(train_bits) // BATCH)
    total_steps = args.epochs * steps_per_epoch
    schedule = build_schedule(total_steps)
    schedule_set = set(schedule)
    print(f"total_steps={total_steps}  n_snapshots={len(schedule)}", flush=True)

    # Snapshot storage — allocate lazily as list, stack at end
    snap_step, snap_epoch, snap_q = [], [], []
    snap_train_nll, snap_held_nll = [], []

    def take_snapshot(step, epoch, batch_nll):
        q = snapshot_q_full(model, D, n, device)
        with torch.no_grad():
            log_q_held = model.log_prob(held_t).cpu().numpy()
            held_nll = float(-log_q_held.mean())
        snap_step.append(step)
        snap_epoch.append(epoch)
        snap_q.append(q)
        snap_train_nll.append(float(batch_nll))
        snap_held_nll.append(held_nll)

    # Initial snapshot (step 0)
    take_snapshot(0, 0, float("nan"))

    # Training loop — simple SGD with shuffled batches, bf16 autocast if GPU.
    # Uses compiled_forward + BCE-outside so torch.compile actually applies.
    # Only take .item() calls on snapshot steps (avoids per-step GPU sync).
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else torch.autocast(device_type="cpu", enabled=False)
    )
    step = 0
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(train_t), device=device)
        for i in range(steps_per_epoch):
            batch = train_t[perm[i * BATCH:(i + 1) * BATCH]]
            with autocast_ctx:
                logits = compiled_forward(batch)
                log_q_batch = -F.binary_cross_entropy_with_logits(
                    logits, batch, reduction="none").sum(dim=1)
                nll = -log_q_batch.mean()
                if args.use_pt:
                    q_scaled = torch.exp(log_q_batch) * D
                    pt_loss  = (q_scaled - log_q_batch).mean()
                    loss = nll + LAMBDA_PT * pt_loss
                else:
                    loss = nll
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step in schedule_set:
                take_snapshot(step, epoch, nll.item())
        if epoch % 25 == 0:
            # nll here is a GPU tensor from the last training step; force sync
            print(f"  ep{epoch:>4}  step={step}  batch_nll={float(nll):.4f}  "
                   f"elapsed={time.time() - t_train:.1f}s", flush=True)

    # Final snapshot if not already
    if step not in schedule_set:
        take_snapshot(step, args.epochs, nll.item())

    print(f"training done in {time.time() - t_train:.1f}s", flush=True)

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        p_C=p_C_full.astype(np.float32),
        train_mask=train_mask, filt_mask=filt_mask, neither_mask=neither_mask,
        steps=np.asarray(snap_step),
        epochs=np.asarray(snap_epoch),
        q_dist=np.stack(snap_q, axis=0),
        train_nll=np.asarray(snap_train_nll),
        held_nll=np.asarray(snap_held_nll),
        # Meta
        n_qubits=n, k_train=args.k_train, hidden=args.hidden,
        epochs_planned=args.epochs, use_pt=args.use_pt, model_seed=args.model_seed,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
