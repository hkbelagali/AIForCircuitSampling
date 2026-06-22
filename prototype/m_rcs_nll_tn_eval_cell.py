"""Stage B: NLL training + XEB evaluation, using pre-sampled TN data.

Mirrors m_rcs_nll_eval_cell.py but replaces cirq.Simulator's exact
statevector with TN-cached amplitudes. Enables n > ~20 where the full
distribution is intractable.

Loads from results/tn_samples/<tag>.npz produced by tn_rcs_sample.py:
  - train_bits, train_pC      (Born samples + their cached p_C)
  - held_bits, held_pC        (held-out test set + their p_C)
  - uniform_bits, uniform_pC  (uniform-random + p_C, for sanity)

Slices train_bits[:k_train] for the requested k_train. Trains the same
AutoregressiveRNN (LSTM, hidden=128, NLL + PT regularizer) Ryan uses.
Evaluates:
  - XEB_train: D * mean(p_C(train)) - 1   (constant per cell — invariant of model)
  - XEB_held:  D * mean(p_C(held))  - 1   (constant — the achievable ceiling)
  - XEB_gen:   D * mean(p_model(held)) - 1  (model scored on held-out Born samples)
  - XEB_model: D * mean(p_C(model_samples)) - 1  (model samples scored vs truth)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Reuse the AutoregressiveRNN and training params from the cirq-based driver
from m_rcs_nll_eval_cell import (
    AutoregressiveRNN, train_nll_pt,
    BATCH_SIZE, TOTAL_STEPS, MIN_EPOCHS, MAX_EPOCHS, LAMBDA_PT,
)


def load_tn_samples(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return {
        "train_bits": z["train_bits"],
        "train_pC": z["train_pC"],
        "held_bits": z["held_bits"],
        "held_pC": z["held_pC"],
        "uniform_bits": z["uniform_bits"],
        "uniform_pC": z["uniform_pC"],
        "meta": meta,
    }


def maybe_compute_model_sample_xeb(model, n, n_samples, qcirc, optimize,
                                     device):
    """Draw n_samples from the trained model, contract their TN amplitudes,
    return XEB_model = D * mean(p_C(z_model)) - 1. Skipped if qcirc is None."""
    if qcirc is None or n_samples <= 0:
        return None
    import tn_rcs
    model.eval()
    samples = model.sample_bits(n_samples).astype(np.uint8)
    pC = tn_rcs.amplitudes_tn(qcirc, samples, optimize=optimize)
    D = 1 << n
    return float(D * pC.mean() - 1)


def run_cell(npz_path, *, k_train, model_seed,
              hidden=128, n_layers=2, lambda_pt=LAMBDA_PT,
              total_steps=TOTAL_STEPS, min_epochs=MIN_EPOCHS,
              max_epochs=MAX_EPOCHS, lr=1e-3,
              compute_model_xeb=True, k_model_samples=10_000,
              device="cpu", optimize="greedy"):
    data = load_tn_samples(npz_path)
    meta = data["meta"]
    n = meta["n"]
    depth = meta["depth"]
    circuit_seed = meta["circuit_seed"]
    if k_train > len(data["train_bits"]):
        raise ValueError(
            f"requested k_train={k_train} exceeds cached k_max={len(data['train_bits'])}")
    train_bits = data["train_bits"][:k_train]
    train_pC = data["train_pC"][:k_train]
    held_bits = data["held_bits"]
    held_pC = data["held_pC"]
    uniform_pC = data["uniform_pC"]

    D = 1 << n
    H_true_est = float(-np.log(np.maximum(held_pC, 1e-30)).mean())

    # Reference XEB values from the cached p_C (model-independent)
    xeb_train_cache = float(D * train_pC.mean() - 1)
    xeb_held_cache = float(D * held_pC.mean() - 1)
    xeb_uniform_cache = float(D * uniform_pC.mean() - 1)

    # Train
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    model = AutoregressiveRNN(n_bits=n, hidden=hidden, n_layers=n_layers).to(device)

    t0 = time.time()
    final_nll, n_epochs = train_nll_pt(
        model, train_bits.astype(np.float32), total_steps, min_epochs, max_epochs,
        BATCH_SIZE, lr, lambda_pt, D, device, verbose=False,
    )
    train_time = time.time() - t0

    # XEB_gen: held bitstrings scored by model probability
    model.eval()
    held_t = torch.from_numpy(held_bits.astype(np.float32)).to(device)
    with torch.no_grad():
        log_q_held = model.log_prob(held_t).cpu().numpy()
    q_held = np.exp(log_q_held)
    xeb_gen = float(D * q_held.mean() - 1)

    # NLL on held-out (Ryan's metric)
    held_nll = float(-log_q_held.mean())
    uniform_nll = n * float(np.log(2))

    # XEB_model + diversity / collapse metrics: optional, needs fresh TN
    xeb_model = None
    model_xeb_time = None
    model_unique_frac = None
    model_top1_frac = None
    if compute_model_xeb:
        import tn_rcs
        qcirc, _, _ = tn_rcs.build_for_n(
            n, depth, circuit_seed, circuit_kind="boixo_v2",
            use_mps=False,
            dtype="complex64" if device.startswith("cuda") else "complex128",
            to_backend="torch-cuda" if device.startswith("cuda") else None,
        )
        model.eval()
        t0 = time.time()
        ms = model.sample_bits(k_model_samples).astype(np.uint8)
        model_pC = tn_rcs.amplitudes_tn(qcirc, ms, optimize=optimize)
        xeb_model = float(D * model_pC.mean() - 1)
        # Diversity / collapse from model samples (MSB-first ints)
        ms_int = np.packbits(
            np.pad(ms, ((0, 0), (0, (8 - n % 8) % 8)), constant_values=0),
            axis=1,
        ).view(np.uint8 if n <= 8 else np.dtype("S" + str((n + 7) // 8)))
        # Cheap: hash bit rows for uniqueness
        ms_hash = [tuple(row) for row in ms]
        from collections import Counter
        cnt = Counter(ms_hash)
        model_unique_frac = float(len(cnt) / len(ms_hash))
        model_top1_frac = float(cnt.most_common(1)[0][1] / len(ms_hash))
        model_xeb_time = time.time() - t0

    # Capacity / quality metrics derived from the held set
    log2 = float(np.log(2))
    nll_held_per_bit = held_nll / n  # normalized; uniform = log(2)
    nll_held_normed = nll_held_per_bit / log2  # 1 = uniform, lower = better
    # Estimate H(p_C) from held samples: -<log p_C>_z~p_C
    H_pC_est = float(-np.log(np.maximum(held_pC, 1e-30)).mean())
    nll_excess = held_nll - H_pC_est  # >=0; 0 = ideal

    return {
        "n": n, "depth": depth, "circuit_seed": circuit_seed,
        "sample_seed": meta["sample_seed"],
        "k_train": k_train, "model_seed": model_seed,
        "hidden": hidden, "n_layers": n_layers,
        "lambda_pt": lambda_pt, "lr": lr,
        "n_epochs": n_epochs, "total_steps": total_steps,
        "final_nll": final_nll, "held_nll": held_nll, "uniform_nll": uniform_nll,
        "xeb_train_cache": xeb_train_cache,
        "xeb_held_cache": xeb_held_cache,
        "xeb_uniform_cache": xeb_uniform_cache,
        "xeb_gen": xeb_gen,
        "xeb_model": xeb_model,
        "gap_held_minus_gen": float(xeb_held_cache - xeb_gen),
        "H_true_est": H_true_est,
        "H_pC_est": H_pC_est,
        "nll_held_per_bit": nll_held_per_bit,
        "nll_held_normed": nll_held_normed,
        "nll_excess": nll_excess,
        "model_unique_frac": model_unique_frac,
        "model_top1_frac": model_top1_frac,
        "train_time_sec": train_time,
        "model_xeb_time_sec": model_xeb_time,
        "npz_path": str(npz_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_npz", type=str, required=True)
    p.add_argument("--k_train", type=int, required=True)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--lambda_pt", type=float, default=LAMBDA_PT)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no_model_xeb", action="store_true",
                    help="skip the model-sample XEB (saves TN time)")
    p.add_argument("--k_model_samples", type=int, default=10_000)
    p.add_argument("--optimize", type=str, default="greedy")
    p.add_argument("--out_subdir", type=str, default="m_rcs_nll_tn_eval")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(__file__).resolve().parents[1] / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = Path(args.samples_npz)
    if not npz.is_absolute():
        npz = Path(__file__).resolve().parents[1] / npz
    base = npz.stem
    tag = f"{base}_k{args.k_train}_h{args.hidden}_ms{args.model_seed}"
    out_path = out_dir / f"{tag}.json"
    if out_path.exists():
        print(f"cached: {tag}", flush=True)
        return
    print(f"npz={npz}  k_train={args.k_train}  model_seed={args.model_seed} "
          f"device={device}", flush=True)

    rec = run_cell(
        npz, k_train=args.k_train, model_seed=args.model_seed,
        hidden=args.hidden, n_layers=args.n_layers,
        lambda_pt=args.lambda_pt, lr=args.lr,
        compute_model_xeb=not args.no_model_xeb,
        k_model_samples=args.k_model_samples,
        device=device, optimize=args.optimize,
    )
    out_path.write_text(json.dumps(rec))
    xm = (f"  XEB_model={rec['xeb_model']:.3f}"
          if rec['xeb_model'] is not None else "")
    print(f"  XEB_gen={rec['xeb_gen']:.3f}  XEB_held(ceil)={rec['xeb_held_cache']:.3f}"
          f"  NLL={rec['final_nll']:.3f}{xm}  ({rec['train_time_sec']:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
