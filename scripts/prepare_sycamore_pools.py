"""Prepare three .npz sample pools for the Sycamore comparison sweep.

For circuit (n20, m14, s0, e0, pEFGH):
  exp_pool.npz     — 250k experimental measurements as train_bits (shuffled deterministically).
                     Held = 5k TN-ideal samples (with p_ideal from TN). Used for both:
                       * training runs (clean XEB is auto-computed on held)
                       * varying k_train downstream (first k of shuffled 250k)

  tn_pool.npz      — 250k TN-ideal-sampled bits as train_bits.
                     Held = 5k TN-ideal samples (different draws from exp_pool's held).

  device_held.npz  — 5000 experimental measurements with p_ideal from Google's
                     amplitudes file as held_bits/held_pC. train_bits = zeros
                     placeholder (won't be used). This is used to evaluate a
                     TRAINED model's "device XEB trained" =
                       D * E_{z~experimental}[q_model(z)] - 1

All three files share:
  meta.n = 20, meta.circuit_py = <path>, meta.circuit_family = "sycamore"

The device XEB baseline (D * E_{z~experimental}[p_ideal(z)] - 1) is written
into device_held.npz's meta under "device_xeb_baseline" so downstream can
report it consistently.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from aics.sampling import sample_exact_tn, amplitudes_tn
from aics.io import save_samples


def load_circuit(path):
    spec = importlib.util.spec_from_file_location("circ", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CIRCUIT, list(mod.QUBIT_ORDER)


def read_measurements(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append([int(c) for c in line])
    return np.asarray(out, dtype=np.uint8)


def read_amplitudes(path):
    """Return (bits (k, n), pC (k,))"""
    bits, pc = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            b = [int(c) for c in parts[0]]
            re, im = float(parts[1]), float(parts[2])
            bits.append(b)
            pc.append(re * re + im * im)
    return np.asarray(bits, dtype=np.uint8), np.asarray(pc, dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--circuit_py", required=True)
    p.add_argument("--measurements", required=True)
    p.add_argument("--amplitudes", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k_train_max", type=int, default=250_000)
    p.add_argument("--k_held", type=int, default=5_000)
    p.add_argument("--k_uniform", type=int, default=2_000)
    p.add_argument("--k_device_held", type=int, default=5_000)
    p.add_argument("--shuffle_seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    circ, qubits = load_circuit(args.circuit_py)
    n = len(qubits)
    D = 1 << n
    print(f"[prep] circuit: {n} qubits")

    # ---- Common meta ----
    common_meta = {
        "n": n, "family": "sycamore", "rows": None, "cols": None,
        "depth": 14, "circuit": "sycamore",
        "circuit_seed": None, "sample_seed": args.shuffle_seed,
        "grid": None, "dtype": "complex64",
        "circuit_py": str(args.circuit_py),
        "measurements_file": str(args.measurements),
        "amplitudes_file": str(args.amplitudes),
    }

    # ---- Ideal held/uniform (TN) — shared across all pools ----
    t0 = time.time()
    held_ideal = sample_exact_tn(circ, qubits, k_samples=args.k_held,
                                   seed=args.shuffle_seed + 99991, dtype="complex64")
    print(f"[prep] TN-sampled held_ideal in {time.time()-t0:.1f}s")
    rng = np.random.default_rng(args.shuffle_seed + 777)
    uniform_bits = rng.integers(0, 2, size=(args.k_uniform, n), dtype=np.uint8)

    t0 = time.time()
    held_ideal_pC = amplitudes_tn(circ, qubits, held_ideal)
    print(f"[prep] amp held_ideal in {time.time()-t0:.1f}s  "
          f"XEB={D * held_ideal_pC.mean() - 1:+.4f}")
    t0 = time.time()
    uniform_pC = amplitudes_tn(circ, qubits, uniform_bits)
    print(f"[prep] amp uniform in {time.time()-t0:.1f}s  "
          f"XEB={D * uniform_pC.mean() - 1:+.4f}")

    zeros_pC = lambda k: np.zeros(k, dtype=np.float64)

    # ---- exp_pool.npz ----
    t0 = time.time()
    exp_bits = read_measurements(args.measurements)
    print(f"[prep] read {len(exp_bits)} measurements in {time.time()-t0:.1f}s")
    rng2 = np.random.default_rng(args.shuffle_seed)
    perm = rng2.permutation(len(exp_bits))
    exp_train_shuffled = exp_bits[perm][:args.k_train_max]
    # Reserve unused experimental for device_held
    exp_reserved_for_device = exp_bits[perm][args.k_train_max:]

    exp_out = out_dir / f"sycamore_n{n}_exp_pool.npz"
    save_samples(exp_out,
                   train_bits=exp_train_shuffled,
                   train_pC=zeros_pC(len(exp_train_shuffled)),
                   held_bits=held_ideal, held_pC=held_ideal_pC,
                   uniform_bits=uniform_bits, uniform_pC=uniform_pC,
                   meta={**common_meta, "sampler": "experimental",
                          "k_max": args.k_train_max})
    print(f"[prep] wrote {exp_out}  ({exp_out.stat().st_size/1e6:.1f} MB)")

    # ---- tn_pool.npz ----
    t0 = time.time()
    tn_train = sample_exact_tn(circ, qubits, k_samples=args.k_train_max,
                                  seed=args.shuffle_seed, dtype="complex64")
    print(f"[prep] TN-sampled {len(tn_train)} train in {time.time()-t0:.1f}s")
    tn_out = out_dir / f"sycamore_n{n}_tn_pool.npz"
    save_samples(tn_out,
                   train_bits=tn_train, train_pC=zeros_pC(len(tn_train)),
                   held_bits=held_ideal, held_pC=held_ideal_pC,
                   uniform_bits=uniform_bits, uniform_pC=uniform_pC,
                   meta={**common_meta, "sampler": "tn",
                          "k_max": args.k_train_max})
    print(f"[prep] wrote {tn_out}  ({tn_out.stat().st_size/1e6:.1f} MB)")

    # ---- device_held.npz ----
    # For "device XEB trained" eval: held = experimental samples with p_ideal
    # from Google's amplitudes file. Draw them from the RESERVED portion
    # (never appeared in training) to avoid the "eval on training data" issue.
    t0 = time.time()
    all_amp_bits, all_amp_pC = read_amplitudes(args.amplitudes)
    print(f"[prep] read {len(all_amp_bits)} amp rows in {time.time()-t0:.1f}s")

    # Match reserved bits to amplitude bits by index (both same permutation of measurements).
    # Simpler: since amp file has same order as measurements, apply same perm.
    amp_bits_perm = all_amp_bits[perm]
    amp_pC_perm = all_amp_pC[perm]
    device_held_bits = amp_bits_perm[args.k_train_max:args.k_train_max + args.k_device_held]
    device_held_pC = amp_pC_perm[args.k_train_max:args.k_train_max + args.k_device_held]

    device_baseline = float(D * device_held_pC.mean() - 1)
    print(f"[prep] device XEB baseline (reserved held) = {device_baseline:+.4f}")

    device_out = out_dir / f"sycamore_n{n}_device_held.npz"
    save_samples(device_out,
                   train_bits=np.zeros((1, n), dtype=np.uint8),
                   train_pC=zeros_pC(1),
                   held_bits=device_held_bits, held_pC=device_held_pC,
                   uniform_bits=uniform_bits, uniform_pC=uniform_pC,
                   meta={**common_meta, "sampler": "experimental_held",
                          "k_max": 1, "device_xeb_baseline": device_baseline})
    print(f"[prep] wrote {device_out}  ({device_out.stat().st_size/1e6:.1f} MB)")

    print("\n[prep] SUMMARY:")
    print(f"  exp_pool: 250k experimental training bits + 5k TN ideal held")
    print(f"  tn_pool:  250k TN-sampled training bits + 5k TN ideal held")
    print(f"  device_held: {args.k_device_held} experimental held bits w/ p_ideal from Google")
    print(f"  device_xeb_baseline = {device_baseline:+.4f}")


if __name__ == "__main__":
    main()
