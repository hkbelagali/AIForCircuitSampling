"""Convert one Google Sycamore circuit's data into our .npz sample bundle.

Two output modes:
  --mode=experimental — training bits come from measurements_*.txt
                        (Sycamore's noisy samples); held/uniform + all
                        pC values come from our TN sampler on the same
                        cirq circuit.
  --mode=tn           — same circuit, but training bits also come from
                        our TN sampler; useful as an apples-to-apples
                        control against experimental mode.

Held pC comes from the ideal TN pipeline in BOTH modes so the eval is
comparable — you're always measuring how well the trained model ranks
the true pC peaks, only the training-sample source differs.

  python scripts/sycamore_ingest.py \\
      --circuit_py sycamore_data/n20/n20_m14/circuit_n20_m14_s0_e0_pEFGH.py \\
      --measurements sycamore_data/n20/n20_m14/measurements_n20_m14_s0_e0_pEFGH.txt \\
      --k_train 400000 --k_held 5000 --k_uniform 2000 \\
      --mode experimental \\
      --out results/tn_samples_pcz/sycamore_n20_m14_s0_e0_pEFGH_exp.npz
"""
import argparse
import importlib.util
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


def load_measurements(path, k, seed):
    """Load first k bitstrings from a measurements file."""
    rng = np.random.default_rng(seed)
    bits_list = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            bits_list.append([int(c) for c in line])
            if len(bits_list) >= k:
                break
    arr = np.asarray(bits_list, dtype=np.uint8)
    # shuffle so training subset isn't just the first N in file order
    idx = rng.permutation(len(arr))
    return arr[idx]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--circuit_py", required=True,
                    help="Google-format circuit_*.py file")
    p.add_argument("--measurements", default=None,
                    help="measurements_*.txt (required for mode=experimental)")
    p.add_argument("--mode", choices=["experimental", "tn"], default="experimental")
    p.add_argument("--k_train", type=int, default=400_000)
    p.add_argument("--k_held", type=int, default=5_000)
    p.add_argument("--k_uniform", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="complex64")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    circ, qubits = load_circuit(args.circuit_py)
    n = len(qubits)
    D = 1 << n
    print(f"[ingest] loaded circuit: {n} qubits, {len(circ)} moments")

    # Training bits
    if args.mode == "experimental":
        if not args.measurements:
            raise ValueError("mode=experimental requires --measurements")
        t0 = time.time()
        train_bits = load_measurements(args.measurements, args.k_train,
                                          args.seed)
        print(f"[ingest] read {len(train_bits)} experimental samples in "
              f"{time.time() - t0:.1f}s")
    else:  # tn
        t0 = time.time()
        train_bits = sample_exact_tn(circ, qubits, k_samples=args.k_train,
                                        seed=args.seed, dtype=args.dtype)
        print(f"[ingest] TN-sampled {len(train_bits)} in "
              f"{time.time() - t0:.1f}s")

    # Held / uniform ALWAYS from ideal TN so eval is comparable.
    t0 = time.time()
    held_bits = sample_exact_tn(circ, qubits, k_samples=args.k_held,
                                  seed=args.seed + 99991, dtype=args.dtype)
    print(f"[ingest] TN-sampled {len(held_bits)} held in "
          f"{time.time() - t0:.1f}s")

    rng = np.random.default_rng(args.seed + 777)
    uniform_bits = rng.integers(0, 2, size=(args.k_uniform, n), dtype=np.uint8)

    # Amplitudes only for held + uniform (used in xeb_norm eval). Training
    # doesn't need pC — the LSTM loss only uses bitstrings. Zero it out.
    train_pC = np.zeros(len(train_bits), dtype=np.float64)
    print(f"[ingest] train_pC set to zeros (LSTM only needs bitstrings for NLL)")
    t0 = time.time()
    held_pC = amplitudes_tn(circ, qubits, held_bits)
    print(f"[ingest] amplitudes_tn held ({len(held_bits)}) in "
          f"{time.time() - t0:.1f}s  XEB={D * held_pC.mean() - 1:+.4f}")
    t0 = time.time()
    uniform_pC = amplitudes_tn(circ, qubits, uniform_bits)
    print(f"[ingest] amplitudes_tn uniform ({len(uniform_bits)}) in "
          f"{time.time() - t0:.1f}s  XEB={D * uniform_pC.mean() - 1:+.4f}")

    meta = {
        "n": n, "family": "sycamore", "rows": None, "cols": None,
        "depth": None,  # circuit-specific; not our simple depth
        "circuit": "sycamore",
        "circuit_seed": None,
        "sample_seed": args.seed,
        "grid": None,
        "sampler": args.mode,  # "experimental" or "tn"
        "k_max": args.k_train, "k_held": args.k_held, "k_uniform": args.k_uniform,
        "dtype": args.dtype,
        "circuit_py": str(args.circuit_py),
        "measurements": str(args.measurements) if args.measurements else None,
    }
    out_path = Path(args.out)
    save_samples(out_path,
                   train_bits=train_bits, train_pC=train_pC,
                   held_bits=held_bits, held_pC=held_pC,
                   uniform_bits=uniform_bits, uniform_pC=uniform_pC,
                   meta=meta)
    print(f"[ingest] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
