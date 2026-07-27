"""Filtered-overlap clean_xeb: recompute clean_xeb splitting the held set into
strings that appear in the training subsample vs strings that don't.

Trained-model bar (clean_xeb) = model's XEB on ALL clean held bitstrings.
Filtered bar                  = model's XEB on clean held bitstrings NOT in
                                 the first k_train rows of the training pool
                                 (which is exactly what train_cell used).

The gap between the two isolates train/held overlap ("leakage") from genuine
generalization to unseen strings.
"""
import json
from pathlib import Path

import numpy as np
import torch

from aics.io import load_samples
from aics.io.conventions import bits_to_int
from aics.models import AutoregressiveRNN

CKPT_DIR = Path("results/maxn_run/train_sycamore")
CLEAN_NPZ = Path("results/tn_samples_pcz/sycamore_n20_tn_pool.npz")
EXP_NPZ = Path("results/tn_samples_pcz/sycamore_n20_exp_pool.npz")

KS = [1000, 10000, 50000, 100000, 250000]
KINDS = ("tn", "exp")


def name_for(kind, k):
    if k == 100000:
        return f"n20_s0_e0_{kind}_h512_s50000_ddp"
    return f"sweep_{kind}_k{k}_h512_s50000"


def eval_xeb(model, bits, pC, device, D):
    if len(bits) == 0:
        return float("nan")
    t = torch.from_numpy(np.asarray(bits, dtype=np.float32)).to(device)
    log_q = model.log_prob(t).cpu().numpy()
    q = np.exp(log_q)
    return float(D * q.mean() - 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    clean = load_samples(CLEAN_NPZ)
    exp   = load_samples(EXP_NPZ)
    n = clean["meta"]["n"]
    D = 1 << n

    held_bits = np.asarray(clean["held_bits"])         # (5000, n)
    held_pC   = np.asarray(clean["held_pC"])
    held_int  = bits_to_int(held_bits)                 # (5000,)

    train_pool_int = {
        "tn":  bits_to_int(np.asarray(clean["train_bits"])),
        "exp": bits_to_int(np.asarray(exp["train_bits"])),
    }

    rows = []
    for kind in KINDS:
        for k in KS:
            ckpt_path = CKPT_DIR / f"{name_for(kind, k)}.ckpt"
            if not ckpt_path.exists():
                print(f"skip missing {ckpt_path}")
                continue
            payload = torch.load(ckpt_path, map_location=device)
            model = AutoregressiveRNN(n_bits=n, hidden=512, n_layers=2).to(device)
            model.load_state_dict(payload["model_state"])
            model.eval()

            train_int = train_pool_int[kind][:k]
            train_set = set(train_int.tolist())
            in_train = np.array([z in train_set for z in held_int], dtype=bool)
            n_overlap = int(in_train.sum())

            with torch.no_grad():
                clean_xeb_all       = eval_xeb(model, held_bits,       held_pC,       device, D)
                clean_xeb_filtered  = eval_xeb(model, held_bits[~in_train], held_pC[~in_train], device, D)
                clean_xeb_overlap   = eval_xeb(model, held_bits[in_train],  held_pC[in_train],  device, D)

            row = {
                "kind": kind, "k_train": k,
                "n_held_overlap": n_overlap,
                "n_held_nonoverlap": int((~in_train).sum()),
                "clean_xeb_all": clean_xeb_all,
                "clean_xeb_filtered": clean_xeb_filtered,
                "clean_xeb_overlap": clean_xeb_overlap,
            }
            rows.append(row)
            print(f"{kind:>3} k={k:>7}  overlap={n_overlap:>4}/5000  "
                  f"all={clean_xeb_all:>7.3f}  filt={clean_xeb_filtered:>7.3f}  "
                  f"olap={clean_xeb_overlap:>7.3f}")

    out = CKPT_DIR / "overlap_eval.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
