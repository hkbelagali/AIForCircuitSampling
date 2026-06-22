"""Stage A: TN-based persistent sampler for Boixo v2 RCS bitstrings.

Per (n, depth, circuit_seed, sample_seed) tuple, we draw:
  - k_max training bitstrings (the maximum k_train we care about)
  - k_held held-out bitstrings (the test set used for XEB_gen / NLL_held)

We then pre-compute p_C(z) for BOTH sets via TN amplitude evaluation
(the heavy work) and persist everything to a single .npz so that
downstream Stage B cells just np.load() and slice.

Pre-computing amplitudes once is the key cost saving: at n=24 each
amplitude is ~30 ms, so 100k + 10k = 110k amplitudes ≈ 1 hour per
(n, circuit_seed) combo. Stage B can then run many (k_train, model_seed)
cells in seconds against this cached pile.

Schema (.npz):
  train_bits     (k_max, n)     uint8, MSB-first
  train_pC       (k_max,)       float64, p_C(train_bits[i])
  held_bits      (k_held, n)    uint8
  held_pC        (k_held,)      float64
  uniform_bits   (k_uni, n)     uint8  # for normalization sanity
  uniform_pC     (k_uni,)       float64
  meta           dict           {n, depth, circuit_seed, sample_seed, ...}
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import tn_rcs
from boixo_v2_rcs import grid_dimensions


def sample_with_tn(n, depth, circuit_seed, sample_seed, k_max, k_held,
                    k_uniform, use_gpu, dtype, marginal_qubits, optimize,
                    sampler="chaotic", pcz_group_size=10):
    """Sample k_max training bitstrings + (if k_held>0) k_held held +
    (if k_uniform>0) k_uniform uniform-random, then compute p_C(z) for all.

    sampler:
      "chaotic" — quimb.sample_chaotic with `marginal_qubits` (biased when
                  marginal_qubits < n on non-PT distributions)
      "pcz"     — pcz_sampler.sample_pcz_marginal (unbiased: sequential
                  marginal-conditional via quimb.Circuit.sample)
    """
    backend = "torch-cuda" if use_gpu else None
    print(f"Building qcirc: n={n} depth={depth} cs={circuit_seed}", flush=True)
    t0 = time.time()
    qcirc, qubits, _ = tn_rcs.build_for_n(
        n, depth, circuit_seed, circuit_kind="boixo_v2",
        use_mps=False, dtype=dtype, to_backend=backend,
    )
    print(f"  built in {time.time() - t0:.1f}s", flush=True)

    def _draw(k, seed, label):
        print(f"Drawing {k} {label} bitstrings (sampler={sampler})...", flush=True)
        ts = time.time()
        if sampler == "chaotic":
            bits = tn_rcs.sample_tn(
                qcirc, k_samples=k, seed=seed,
                marginal_qubits=min(marginal_qubits, n),
                optimize=optimize,
                dtype="complex64" if use_gpu else dtype,
            )
        elif sampler == "pcz":
            import pcz_sampler
            from boixo_v2_rcs import make_boixo_v2_rcs_circuit
            qubits_c, circ = make_boixo_v2_rcs_circuit(
                n, cz_depth=depth, seed=circuit_seed)
            tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits_c)
            bits = pcz_sampler.sample_pcz_marginal(
                tn, k_samples=k, seed=seed,
                group_size=pcz_group_size,
                optimize=optimize,
                dtype="complex64" if use_gpu else "complex128",
            )
        else:
            raise ValueError(f"unknown sampler {sampler!r}")
        dt = time.time() - ts
        print(f"  {dt:.1f}s ({dt/k*1000:.2f} ms/sample)", flush=True)
        return bits

    train_bits = _draw(k_max, sample_seed, "training")

    train_pC = tn_rcs.amplitudes_tn(qcirc, train_bits, optimize=optimize)
    D = 1 << n
    xeb_train = float(D * train_pC.mean() - 1)
    print(f"  train_pC done, XEB(train)={xeb_train:+.4f}", flush=True)

    held_bits = held_pC = uniform_bits = uniform_pC = None
    xeb_held = xeb_uniform = None
    if k_held > 0:
        held_bits = _draw(k_held, sample_seed + 99991, "held-out")
        held_pC = tn_rcs.amplitudes_tn(qcirc, held_bits, optimize=optimize)
        xeb_held = float(D * held_pC.mean() - 1)
        print(f"  XEB(held)={xeb_held:+.4f}", flush=True)
    if k_uniform > 0:
        rng = np.random.default_rng(sample_seed + 777)
        uniform_bits = rng.integers(0, 2, size=(k_uniform, n), dtype=np.uint8)
        uniform_pC = tn_rcs.amplitudes_tn(qcirc, uniform_bits, optimize=optimize)
        xeb_uniform = float(D * uniform_pC.mean() - 1)
        print(f"  XEB(uniform)={xeb_uniform:+.4f}  (expect ~0)", flush=True)

    return dict(
        train_bits=train_bits, train_pC=train_pC,
        held_bits=held_bits, held_pC=held_pC,
        uniform_bits=uniform_bits, uniform_pC=uniform_pC,
        xeb_train=xeb_train, xeb_held=xeb_held, xeb_uniform=xeb_uniform,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--circuit_seed", type=int, default=42)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--k_max", type=int, default=100_000,
                    help="total bitstrings across all chunks for this n")
    p.add_argument("--k_held", type=int, default=10_000)
    p.add_argument("--k_uniform", type=int, default=2_000)
    p.add_argument("--chunk_idx", type=int, default=0,
                    help="this chunk's index in [0, n_chunks)")
    p.add_argument("--n_chunks", type=int, default=1,
                    help="total chunks; each does k_max/n_chunks samples")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--dtype", type=str, default="complex128",
                    help="complex128 default for CPU; pass --gpu for complex64")
    p.add_argument("--optimize", type=str, default="greedy")
    p.add_argument("--marginal_qubits", type=int, default=20)
    p.add_argument("--sampler", type=str, default="chaotic",
                    choices=["chaotic", "pcz"],
                    help="chaotic: quimb.sample_chaotic — BIASED on non-PT "
                         "distributions when marginal<n (kept as v1 baseline only). "
                         "pcz: pcz_sampler.sample_pcz_marginal — unbiased "
                         "sequential marginal-conditional sampler. USE pcz FOR "
                         "NEW RUNS unless you know why you want the biased baseline.")
    p.add_argument("--pcz_group_size", type=int, default=10)
    p.add_argument("--out_subdir", type=str, default="tn_samples")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.gpu and args.dtype == "complex128":
        args.dtype = "complex64"
    if args.n_chunks < 1 or not (0 <= args.chunk_idx < args.n_chunks):
        raise ValueError("chunk_idx must be in [0, n_chunks)")
    # k_max passed in is the TOTAL we want across all chunks; each chunk
    # contributes k_max / n_chunks. Held + uniform only done by chunk 0.
    k_per_chunk = args.k_max // args.n_chunks
    args.sample_seed_effective = args.sample_seed * 100003 + args.chunk_idx
    do_held = (args.chunk_idx == 0)
    args._k_per_chunk = k_per_chunk
    args._do_held = do_held

    out_dir = Path(__file__).resolve().parents[1] / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.n_chunks == 1:
        tag = (f"n{args.n}_d{args.depth}_cs{args.circuit_seed}"
               f"_ss{args.sample_seed}_k{args.k_max}")
    else:
        tag = (f"n{args.n}_d{args.depth}_cs{args.circuit_seed}"
               f"_ss{args.sample_seed}_c{args.chunk_idx}of{args.n_chunks}"
               f"_k{args._k_per_chunk}")
    out_path = out_dir / f"{tag}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"already exists: {out_path}")
        return

    grid = grid_dimensions(args.n)
    print(f"n={args.n} grid={grid} depth={args.depth} cs={args.circuit_seed}")
    print(f"chunk {args.chunk_idx}/{args.n_chunks}  "
          f"effective sample_seed={args.sample_seed_effective}")
    print(f"k_per_chunk={args._k_per_chunk} k_held={args.k_held if args._do_held else 0}  "
          f"GPU={args.gpu} dtype={args.dtype}\n")

    # If this isn't chunk 0, skip held + uniform
    k_held = args.k_held if args._do_held else 0
    k_uniform = args.k_uniform if args._do_held else 0

    t0 = time.time()
    out = sample_with_tn(
        args.n, args.depth, args.circuit_seed, args.sample_seed_effective,
        args._k_per_chunk, k_held, k_uniform,
        args.gpu, args.dtype, args.marginal_qubits, args.optimize,
        sampler=args.sampler, pcz_group_size=args.pcz_group_size,
    )
    total = time.time() - t0

    meta = {
        "n": args.n, "depth": args.depth, "circuit_seed": args.circuit_seed,
        "sample_seed": args.sample_seed, "grid": list(grid),
        "k_max": args.k_max, "k_held": args.k_held, "k_uniform": args.k_uniform,
        "dtype": args.dtype, "use_gpu": args.gpu,
        "optimize": args.optimize, "marginal_qubits": args.marginal_qubits,
        "sampler": args.sampler, "pcz_group_size": args.pcz_group_size,
        "xeb_train": out["xeb_train"], "xeb_held": out["xeb_held"],
        "xeb_uniform": out["xeb_uniform"],
        "total_time_sec": total,
    }
    save_kwargs = dict(
        train_bits=out["train_bits"], train_pC=out["train_pC"],
        meta=json.dumps(meta),
    )
    if out["held_bits"] is not None:
        save_kwargs["held_bits"] = out["held_bits"]
        save_kwargs["held_pC"] = out["held_pC"]
    if out["uniform_bits"] is not None:
        save_kwargs["uniform_bits"] = out["uniform_bits"]
        save_kwargs["uniform_pC"] = out["uniform_pC"]
    np.savez(out_path, **save_kwargs)
    print(f"\nwrote {out_path}  ({total:.0f}s total)")


if __name__ == "__main__":
    main()
