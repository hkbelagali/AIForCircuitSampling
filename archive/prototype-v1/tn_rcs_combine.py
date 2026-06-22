"""Combine chunked Stage A npz files into one canonical npz.

After running tn_rcs_sample.py with --n_chunks=K, you'll have K files
named like
  n{n}_d{d}_cs{cs}_ss{ss}_c{0..K-1}of{K}_k{k_per_chunk}.npz

This script merges them into the canonical single-shot filename
  n{n}_d{d}_cs{cs}_ss{ss}_k{k_total}.npz
that Stage B expects. Held + uniform come from chunk 0.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def combine(n, depth, circuit_seed, sample_seed, n_chunks, in_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Single-shot case (n_chunks=1) writes directly to the canonical name —
    # nothing to combine, just verify it exists.
    if n_chunks == 1:
        canonical = (in_dir
                     / f"n{n}_d{depth}_cs{circuit_seed}_ss{sample_seed}_k100000.npz")
        if not canonical.exists():
            # Try the chunked form too in case it was generated that way
            chunk_glob = f"n{n}_d{depth}_cs{circuit_seed}_ss{sample_seed}_c0of1_k*.npz"
            matches = list(in_dir.glob(chunk_glob))
            if matches:
                # Copy/rename to canonical
                import shutil
                shutil.copy(matches[0], canonical)
                print(f"  copied {matches[0].name} → {canonical.name}")
            else:
                raise FileNotFoundError(
                    f"single-shot expected at {canonical} (no chunked fallback)")
        else:
            print(f"  already canonical: {canonical.name}")
        return
    chunks = []
    for c in range(n_chunks):
        glob = f"n{n}_d{depth}_cs{circuit_seed}_ss{sample_seed}_c{c}of{n_chunks}_k*.npz"
        matches = list(in_dir.glob(glob))
        if not matches:
            raise FileNotFoundError(f"chunk {c}/{n_chunks} missing: {glob}")
        if len(matches) > 1:
            raise ValueError(f"multiple matches for chunk {c}: {matches}")
        chunks.append(matches[0])
    print(f"combining {n_chunks} chunks for n={n}, ss={sample_seed}")
    for c, p in enumerate(chunks):
        print(f"  chunk {c}: {p.name}")
    train_bits_list, train_pC_list = [], []
    held_bits = held_pC = uniform_bits = uniform_pC = None
    meta0 = None
    for c, p in enumerate(chunks):
        z = np.load(p, allow_pickle=True)
        train_bits_list.append(z["train_bits"])
        train_pC_list.append(z["train_pC"])
        if c == 0:
            meta0 = json.loads(str(z["meta"]))
            if "held_bits" in z.files:
                held_bits = z["held_bits"]
                held_pC = z["held_pC"]
            if "uniform_bits" in z.files:
                uniform_bits = z["uniform_bits"]
                uniform_pC = z["uniform_pC"]
    train_bits = np.concatenate(train_bits_list, axis=0)
    train_pC = np.concatenate(train_pC_list, axis=0)
    k_total = len(train_bits)

    # Sanity-check XEB on the combined set
    D = 1 << n
    xeb_train = float(D * train_pC.mean() - 1)
    xeb_held = float(D * held_pC.mean() - 1) if held_pC is not None else None
    xeb_uniform = float(D * uniform_pC.mean() - 1) if uniform_pC is not None else None
    print(f"  combined k_train={k_total}  XEB(train)={xeb_train:+.4f}"
          + (f"  XEB(held)={xeb_held:+.4f}" if xeb_held is not None else "")
          + (f"  XEB(unif)={xeb_uniform:+.4f}" if xeb_uniform is not None else ""))

    meta = dict(meta0) if meta0 else {}
    meta.update(dict(
        n=n, depth=depth, circuit_seed=circuit_seed, sample_seed=sample_seed,
        k_max=k_total, n_chunks=n_chunks,
        xeb_train=xeb_train, xeb_held=xeb_held, xeb_uniform=xeb_uniform,
    ))
    tag = f"n{n}_d{depth}_cs{circuit_seed}_ss{sample_seed}_k{k_total}"
    out_path = out_dir / f"{tag}.npz"
    save_kwargs = dict(
        train_bits=train_bits, train_pC=train_pC, meta=json.dumps(meta),
    )
    if held_bits is not None:
        save_kwargs["held_bits"] = held_bits
        save_kwargs["held_pC"] = held_pC
    if uniform_bits is not None:
        save_kwargs["uniform_bits"] = uniform_bits
        save_kwargs["uniform_pC"] = uniform_pC
    np.savez(out_path, **save_kwargs)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--circuit_seed", type=int, default=42)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--n_chunks", type=int, required=True)
    p.add_argument("--in_dir", type=str,
                    default=str(Path(__file__).resolve().parents[1] /
                                "results" / "tn_samples"))
    p.add_argument("--out_dir", type=str,
                    default=str(Path(__file__).resolve().parents[1] /
                                "results" / "tn_samples"))
    args = p.parse_args()
    combine(args.n, args.depth, args.circuit_seed, args.sample_seed,
             args.n_chunks, args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
