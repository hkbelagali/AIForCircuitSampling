"""Downsample a dynamics-snapshot .npz to ~target snapshots by keeping every
Nth in the stored schedule. Small metadata / q_dist arrays are indexed;
p_C, masks, and scalar meta pass through unchanged. Writes to <path>.downsampled
by default.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("--target", type=int, default=200,
                     help="approximate number of snapshots to keep")
    ap.add_argument("--out", default=None, help="output path")
    args = ap.parse_args()

    src = Path(args.in_path)
    dst = Path(args.out) if args.out else src.with_suffix(".downsampled.npz")

    print(f"loading {src}...", flush=True)
    d = np.load(src, allow_pickle=True)
    steps = d["steps"]
    S = len(steps)
    stride = max(1, S // args.target)
    # Keep first + last always, plus evenly-strided in between
    keep = np.arange(0, S, stride)
    if keep[-1] != S - 1:
        keep = np.concatenate([keep, [S - 1]])
    keep = np.unique(keep)
    print(f"keeping {len(keep)} of {S} snapshots (stride={stride})", flush=True)

    # Pass-through vs indexed keys
    out_dict = {}
    for key in d.files:
        arr = d[key]
        if key in ("steps", "epochs", "q_dist", "train_nll", "held_nll"):
            out_dict[key] = arr[keep]
        else:
            out_dict[key] = arr

    print(f"writing {dst} (compressed)...", flush=True)
    np.savez_compressed(dst, **out_dict)
    print(f"done. src size={src.stat().st_size / 1e9:.2f}GB  "
           f"dst size={dst.stat().st_size / 1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
