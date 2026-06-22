"""Stage A: build an RCS circuit, draw bitstring samples, compute p_C(z)
for each, and write an .npz sample bundle for downstream training.

Usage examples
--------------

  # Single-shot, default everything (Boixo v2, depth 10, exact_tn sampler):
  python scripts/sample.py --n 12

  # Multi-chunk on SLURM array; the LAST chunk auto-combines into the
  # canonical nN_d10_cs42_ss0_kK.npz file:
  python scripts/sample.py --n 32 --n_chunks 8 --chunk_idx $SLURM_ARRAY_TASK_ID

  # Force a re-combine after the fact:
  python scripts/sample.py --n 32 --n_chunks 8 --combine

  # Biased v1 baseline for chaotic-vs-pcz comparison plots:
  python scripts/sample.py --n 16 --sampler chaotic --chaotic_marginal_qubits 12
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from aics.circuits import (
    make_boixo_v2_rcs_circuit, make_sycamore_rcs_circuit,
    grid_dimensions,
)
from aics.sampling import sample_exact_tn, sample_chaotic, amplitudes_tn
from aics.io import save_samples, combine_chunks
from aics.training import print_hardware, assert_device_available
from aics.eval import linear_xeb_from_bits


CIRCUIT_BUILDERS = {
    "boixo_v2": make_boixo_v2_rcs_circuit,
    "sycamore": make_sycamore_rcs_circuit,
}


def _build_circuit(name, n, depth, seed):
    if name == "boixo_v2":
        return make_boixo_v2_rcs_circuit(n, cz_depth=depth, seed=seed)
    elif name == "sycamore":
        return make_sycamore_rcs_circuit(n_qubits=n, depth=depth, seed=seed)
    else:
        raise ValueError(f"unknown circuit family: {name!r}")


def _chunk_tag(n, depth, cs, ss, chunk_idx, n_chunks, k_per_chunk):
    return (f"n{n}_d{depth}_cs{cs}_ss{ss}"
            f"_c{chunk_idx}of{n_chunks}_k{k_per_chunk}")


def _canonical_tag(n, depth, cs, ss, k_total):
    return f"n{n}_d{depth}_cs{cs}_ss{ss}_k{k_total}"


def _draw(sampler_name, circ, qubits, k, *, seed, dtype, marginal_qubits=None):
    if sampler_name == "exact_tn":
        return sample_exact_tn(circ, qubits, k_samples=k, seed=seed, dtype=dtype)
    elif sampler_name == "chaotic":
        return sample_chaotic(circ, qubits, k_samples=k, seed=seed, dtype=dtype,
                                marginal_qubits=marginal_qubits)
    else:
        raise ValueError(f"unknown sampler: {sampler_name!r}")


def run_chunk(args, k_per_chunk, do_held_uniform):
    """Sample one chunk + p_C(z); held + uniform attached only on chunk 0."""
    sample_seed_eff = args.sample_seed * 100003 + (args.chunk_idx or 0)
    print(f"[sample] n={args.n} depth={args.depth} family={args.circuit} "
          f"chunk={args.chunk_idx or 0}/{args.n_chunks} "
          f"effective_seed={sample_seed_eff}",
          flush=True)

    qubits, circ = _build_circuit(args.circuit, args.n, args.depth, args.circuit_seed)
    t0 = time.time()
    train_bits = _draw(args.sampler, circ, qubits, k_per_chunk,
                        seed=sample_seed_eff, dtype=args.dtype,
                        marginal_qubits=args.chaotic_marginal_qubits)
    print(f"  drew {k_per_chunk} train in {time.time() - t0:.1f}s", flush=True)
    train_pC = amplitudes_tn(circ, qubits, train_bits)
    D = 1 << args.n
    print(f"  XEB(train)={D * train_pC.mean() - 1:+.4f}", flush=True)

    held_bits = held_pC = uniform_bits = uniform_pC = None
    if do_held_uniform and args.k_held > 0:
        t0 = time.time()
        held_bits = _draw(args.sampler, circ, qubits, args.k_held,
                            seed=sample_seed_eff + 99991, dtype=args.dtype,
                            marginal_qubits=args.chaotic_marginal_qubits)
        held_pC = amplitudes_tn(circ, qubits, held_bits)
        print(f"  held {args.k_held} in {time.time() - t0:.1f}s "
              f"XEB(held)={D * held_pC.mean() - 1:+.4f}", flush=True)
    if do_held_uniform and args.k_uniform > 0:
        rng = np.random.default_rng(sample_seed_eff + 777)
        uniform_bits = rng.integers(0, 2,
                                      size=(args.k_uniform, args.n),
                                      dtype=np.uint8)
        uniform_pC = amplitudes_tn(circ, qubits, uniform_bits)
        print(f"  uniform {args.k_uniform} XEB(uniform)={D * uniform_pC.mean() - 1:+.4f}",
              flush=True)

    grid = grid_dimensions(args.n) if args.circuit == "boixo_v2" else (args.n, 1)
    meta = {
        "n": args.n, "depth": args.depth, "circuit": args.circuit,
        "circuit_seed": args.circuit_seed,
        "sample_seed": args.sample_seed,
        "sample_seed_effective": sample_seed_eff,
        "grid": list(grid),
        "sampler": args.sampler,
        "chaotic_marginal_qubits": args.chaotic_marginal_qubits if args.sampler == "chaotic" else None,
        "k_max": args.k_max, "k_held": args.k_held, "k_uniform": args.k_uniform,
        "n_chunks": args.n_chunks, "chunk_idx": args.chunk_idx,
        "dtype": args.dtype,
    }

    out_dir = Path(args.out_dir)
    if args.n_chunks > 1:
        tag = _chunk_tag(args.n, args.depth, args.circuit_seed,
                          args.sample_seed, args.chunk_idx,
                          args.n_chunks, k_per_chunk)
    else:
        tag = _canonical_tag(args.n, args.depth, args.circuit_seed,
                              args.sample_seed, args.k_max)
    out_path = out_dir / f"{tag}.npz"
    save_samples(
        out_path,
        train_bits=train_bits, train_pC=train_pC,
        held_bits=held_bits, held_pC=held_pC,
        uniform_bits=uniform_bits, uniform_pC=uniform_pC,
        meta=meta,
    )
    print(f"  wrote {out_path}", flush=True)
    return out_path


def maybe_combine(args):
    """Combine chunks if all K are present (auto-trigger at end of last chunk)."""
    if args.n_chunks <= 1:
        return None
    k_per_chunk = args.k_max // args.n_chunks
    out_dir = Path(args.out_dir)
    chunk_paths = [
        out_dir / f"{_chunk_tag(args.n, args.depth, args.circuit_seed, args.sample_seed, i, args.n_chunks, k_per_chunk)}.npz"
        for i in range(args.n_chunks)
    ]
    missing = [p for p in chunk_paths if not p.exists()]
    if missing:
        print(f"[combine] skipping — {len(missing)}/{args.n_chunks} chunks missing",
              flush=True)
        return None
    canonical = out_dir / f"{_canonical_tag(args.n, args.depth, args.circuit_seed, args.sample_seed, args.k_max)}.npz"
    combine_chunks(chunk_paths, canonical)
    print(f"[combine] wrote {canonical}  ({args.n_chunks} chunks merged)", flush=True)
    return canonical


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--circuit", choices=list(CIRCUIT_BUILDERS), default="boixo_v2")
    p.add_argument("--circuit_seed", type=int, default=42)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--k_max", type=int, default=100_000,
                    help="total training samples across all chunks")
    p.add_argument("--k_held", type=int, default=10_000)
    p.add_argument("--k_uniform", type=int, default=2_000)
    p.add_argument("--sampler", choices=["exact_tn", "chaotic"],
                    default="exact_tn",
                    help="exact_tn (unbiased, default) | chaotic (BIASED, v1 baseline)")
    p.add_argument("--chaotic_marginal_qubits", type=int, default=20)
    p.add_argument("--chunk_idx", type=int, default=None,
                    help="0..n_chunks-1; omit to run all chunks serially")
    p.add_argument("--n_chunks", type=int, default=1)
    p.add_argument("--combine", action="store_true",
                    help="just merge existing chunks into canonical .npz and exit")
    p.add_argument("--gpu", action="store_true",
                    help="require CUDA; hard error if unavailable")
    p.add_argument("--dtype", type=str, default="complex128",
                    help="quimb contraction dtype; complex64 recommended on GPU")
    p.add_argument("--out_dir", type=str, default="results/tn_samples")
    args = p.parse_args()

    device = assert_device_available(args.gpu)
    if args.gpu and args.dtype == "complex128":
        # Don't silently override — but warn.
        print("[sample] note: --gpu with complex128 is supported but slow; "
              "consider --dtype complex64", flush=True)
    print_hardware(device, dtype=args.dtype,
                    extra=f"sampler={args.sampler}")

    if args.combine:
        maybe_combine(args)
        return

    # (a) explicit chunk on SLURM array
    if args.chunk_idx is not None:
        k_per_chunk = args.k_max // args.n_chunks
        do_held_uniform = (args.chunk_idx == 0)
        run_chunk(args, k_per_chunk, do_held_uniform)
        # auto-trigger combine if this was the last chunk and all others exist
        maybe_combine(args)
        return

    # (b) no chunk_idx → run all chunks serially in this process
    if args.n_chunks == 1:
        run_chunk(args, args.k_max, do_held_uniform=True)
    else:
        k_per_chunk = args.k_max // args.n_chunks
        for i in range(args.n_chunks):
            args.chunk_idx = i
            run_chunk(args, k_per_chunk, do_held_uniform=(i == 0))
        maybe_combine(args)


if __name__ == "__main__":
    main()
