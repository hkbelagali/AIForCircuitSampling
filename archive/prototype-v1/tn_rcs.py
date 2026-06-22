"""TN-based RCS sampling and amplitude evaluation.

Replaces cirq.Simulator's 2^n statevector with quimb's tensor-network
contractions, enabling RCS work at n > 24 where exact statevector blows up.

Two backends, unified API:
  - Exact: `qtn.Circuit`, cotengra full contraction. No truncation.
  - MPS:   `qtn.CircuitMPS`, bond dim capped by `max_bond`. Approximate.

Sampling uses `sample_chaotic` (frugal Boixo-style); amplitude evaluation
uses `amplitude()` with the same cotengra tree. Contraction trees can be
cached on disk and re-used across sampling + amplitude calls.

GPU: pass `to_backend='torch'` and `device='cuda'`. Contractions run on
PyTorch CUDA via quimb's backend dispatch.

Circuit-family agnostic: `cirq_to_quimb` takes any cirq.Circuit + qubit
list. The `build_for_n` convenience defaults to Boixo v2 (matches Ryan's
notebook) but accepts `circuit_kind='sycamore'` to switch to our
brickwork.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cirq
import quimb.tensor as qtn

from boixo_v2_rcs import make_boixo_v2_rcs_circuit, grid_dimensions


def _resolve_to_backend(to_backend, dtype):
    """quimb 1.11's `to_backend` arg expects a callable, not a string.
    Convert common shortcuts ('torch', 'torch-cuda', None) into the right
    callable.
    """
    if to_backend is None or callable(to_backend):
        return to_backend
    if to_backend in ("torch", "torch-cpu"):
        import torch
        td = getattr(torch, "complex128" if "128" in dtype else "complex64")
        return lambda x: torch.as_tensor(x, dtype=td)
    if to_backend in ("torch-cuda", "torch-gpu", "cuda"):
        import torch
        td = getattr(torch, "complex128" if "128" in dtype else "complex64")
        return lambda x: torch.as_tensor(x, dtype=td, device="cuda")
    raise ValueError(f"unknown to_backend shortcut: {to_backend!r}")


def cirq_to_quimb(cirq_circuit, qubits, *, use_mps=False, max_bond=None,
                    cutoff=1e-10, dtype="complex128", to_backend=None):
    """Translate any cirq.Circuit + qubit list into a quimb Circuit
    (exact) or CircuitMPS (truncated).

    `to_backend` accepts a callable, or the shortcuts {'torch',
    'torch-cuda'}. None = numpy (CPU).

    Returns (qcirc, qubits_to_idx).
    """
    n = len(qubits)
    qubits_to_idx = {q: i for i, q in enumerate(qubits)}
    backend_fn = _resolve_to_backend(to_backend, dtype)

    if use_mps:
        qcirc = qtn.CircuitMPS(
            N=n, max_bond=max_bond, cutoff=cutoff,
            dtype=dtype, to_backend=backend_fn,
        )
    else:
        qcirc = qtn.Circuit(
            N=n, psi0_dtype=dtype, dtype=dtype, to_backend=backend_fn,
        )

    for op in cirq_circuit.all_operations():
        U = np.asarray(cirq.unitary(op), dtype=dtype)
        where = tuple(qubits_to_idx[q] for q in op.qubits)
        qcirc.apply_gate_raw(U, where=where)

    return qcirc, qubits_to_idx


def prepare_amplitude_tree(qcirc, *, optimize="auto-hq", cache_path=None,
                            example_b=None):
    """Pre-compute and (optionally) persist the cotengra contraction tree
    used by `amplitude(...)`. Subsequent amplitude calls reuse this tree.

    `example_b` is any concrete bitstring; the tree is independent of it.
    """
    if isinstance(qcirc, qtn.CircuitMPS):
        return None  # MPS contractions are local — no global tree needed
    n = qcirc.N
    b = example_b if example_b is not None else "0" * n
    info = qcirc.amplitude_rehearse(b=b, optimize=optimize)
    tree = info.get("tree", info)
    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(tree, f)
    return tree


def load_amplitude_tree(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def sample_tn(qcirc, k_samples, *, marginal_qubits=None, seed=None,
                optimize="auto-hq", dtype="complex64"):
    """BIASED on non-Porter-Thomas distributions. Use pcz_sampler.sample_pcz_marginal instead.

    Wraps quimb's `sample_chaotic`, which puts a UNIFORM PRIOR on every qubit
    not in `marginal_qubits` (literally `rng.choice(("0", "1"))`). For our
    depth-10 2-row Boixo-v2 circuits the state is not fully PT, so the
    chaotic-marginal assumption introduces measurable bias in XEB / NLL when
    marginal_qubits < n. Kept here only as the v1 baseline; new runs should
    use the unbiased sequential marginal-conditional sampler.

    Returns (k_samples, n) uint8 array, MSB-first to match cirq convention.
    """
    n = qcirc.N
    if marginal_qubits is None:
        marginal_qubits = min(20, n)
    out = np.empty((k_samples, n), dtype=np.uint8)
    gen = qcirc.sample_chaotic(
        C=k_samples, marginal_qubits=marginal_qubits,
        seed=seed, optimize=optimize, dtype=dtype,
    )
    for i, bitstring in enumerate(gen):
        out[i] = np.frombuffer(bitstring.encode("ascii"),
                                dtype=np.uint8) - ord("0")
    return out


def sample_mps(qcirc, k_samples, *, seed=None):
    """Sample from a CircuitMPS via its native MPS sampler.

    Returns (k_samples, n) uint8 array, MSB-first.
    """
    n = qcirc.N
    out = np.empty((k_samples, n), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for i, bitstring in enumerate(qcirc.sample(C=k_samples, seed=rng)):
        out[i] = np.frombuffer(bitstring.encode("ascii"),
                                dtype=np.uint8) - ord("0")
    return out


def amplitudes_tn(qcirc, bitstrings, *, tree=None, optimize=None,
                    dtype=None):
    """|<z|psi>|^2 for each bitstring z.

    bitstrings: (k, n) uint8, MSB-first.
    tree: cached contraction tree from prepare_amplitude_tree (optional).
    Returns (k,) float64.
    """
    k = len(bitstrings)
    probs = np.empty(k, dtype=np.float64)
    opt = tree if tree is not None else (optimize or "auto-hq")
    for i in range(k):
        b = "".join(str(int(x)) for x in bitstrings[i])
        amp = qcirc.amplitude(b, optimize=opt, dtype=dtype)
        # quimb amplitude may return a torch tensor on GPU; extract scalar
        if hasattr(amp, "item"):
            amp_val = complex(amp.item())
        else:
            amp_val = complex(amp)
        probs[i] = float(abs(amp_val) ** 2)
    return probs


def build_for_n(n, depth=10, circuit_seed=42, circuit_kind="boixo_v2",
                  **kwargs):
    """Convenience: build a quimb circuit at the requested size.

    circuit_kind:
      - "boixo_v2": Google v2 / Boixo 2018 CZ-based circuit
                    (matches Ryan's notebook bit-identically)
      - "sycamore": Sycamore brickwork via cirq_google.SYC
                    (matches Arute 2019 hardware)

    Returns (qcirc, qubits, qubits_to_idx).
    """
    if circuit_kind == "boixo_v2":
        qubits, cirq_circuit = make_boixo_v2_rcs_circuit(
            n, cz_depth=depth, seed=circuit_seed)
    elif circuit_kind == "sycamore":
        from aics.circuits.brickwork import make_rcs_circuit, grid_for
        rows, cols = grid_for(n)
        qubits, cirq_circuit = make_rcs_circuit(rows, cols, depth, circuit_seed)
    else:
        raise ValueError(f"unknown circuit_kind: {circuit_kind!r}")

    qcirc, qubits_to_idx = cirq_to_quimb(cirq_circuit, qubits, **kwargs)
    return qcirc, qubits, qubits_to_idx


# ----- standalone smoke test: compare TN amplitudes vs cirq at small n -----

def _smoke_test(n=12, depth=10, circuit_seed=42, circuit_kind="boixo_v2",
                  use_gpu=False, use_mps=False, max_bond=None,
                  k_sample=200, validate_vs_cirq=True):
    """Validate the TN pipeline at the requested size.

    At small n (n <= ~20) we can compare against cirq's exact_probabilities.
    At larger n we skip the cirq comparison and only do self-consistency.
    """
    import time
    if validate_vs_cirq:
        from aics.circuits.exact import exact_probabilities

    backend = "torch-cuda" if use_gpu else None
    print(f"Building n={n} circuit_kind={circuit_kind} depth={depth} "
          f"cs={circuit_seed}  "
          f"mode={'MPS' if use_mps else 'exact'}"
          f"{f' max_bond={max_bond}' if (use_mps and max_bond) else ''}  "
          f"backend={backend or 'numpy'}")

    t0 = time.time()
    qcirc, qubits, _ = build_for_n(
        n, depth, circuit_seed, circuit_kind=circuit_kind,
        use_mps=use_mps, max_bond=max_bond,
        dtype="complex128" if not use_gpu else "complex64",
        to_backend=backend,
    )
    print(f"  build time: {time.time() - t0:.2f}s  grid={getattr(qubits[-1], 'row', None) is not None and (qubits[-1].row + 1, qubits[-1].col + 1)}")

    if validate_vs_cirq:
        # Rebuild the same circuit (numpy) for the cirq exact comparison
        if circuit_kind == "boixo_v2":
            _, cirq_circuit = make_boixo_v2_rcs_circuit(
                n, cz_depth=depth, seed=circuit_seed)
        else:
            from aics.circuits.brickwork import make_rcs_circuit, grid_for
            _, cirq_circuit = make_rcs_circuit(
                *grid_for(n), depth, circuit_seed)
        p_C_cirq = exact_probabilities(cirq_circuit, qubits)

        rng = np.random.default_rng(0)
        D = 1 << n
        test_idx = rng.choice(D, size=8, replace=False)
        test_bits = np.array(
            [[(int(i) >> (n - 1 - q)) & 1 for q in range(n)] for i in test_idx],
            dtype=np.uint8,
        )

        print("\nComputing TN amplitudes vs cirq exact_probabilities...")
        t0 = time.time()
        probs_tn = amplitudes_tn(qcirc, test_bits)
        elapsed = time.time() - t0
        print(f"  TN time: {elapsed:.2f}s for {len(test_bits)} amplitudes "
              f"({elapsed/len(test_bits)*1000:.1f} ms/amp)")
        print(f"  {'idx':>8} {'p_cirq':>14} {'p_TN':>14}  rel_diff")
        for i, idx in enumerate(test_idx):
            rel = abs(probs_tn[i] - p_C_cirq[idx]) / max(p_C_cirq[idx], 1e-30)
            print(f"  {idx:>8} {p_C_cirq[idx]:>14.6e} {probs_tn[i]:>14.6e}  {rel:.2e}")

        max_rel = max(abs(probs_tn[i] - p_C_cirq[idx]) / max(p_C_cirq[idx], 1e-30)
                       for i, idx in enumerate(test_idx))
        print(f"\n  max relative diff: {max_rel:.2e}")
        # 1e-4 tolerance: quimb drops to single precision internally
        tol = 1e-3 if use_gpu else 1e-4  # GPU defaults complex64 → more drift
        assert max_rel < tol, f"TN amplitudes do not match cirq (tol={tol})!"
        print(f"  OK: matches cirq within {tol:.0e}")

    sampler = "sample" if use_mps else "sample_chaotic"
    print(f"\nSampling {k_sample} bitstrings via {sampler}...")
    t0 = time.time()
    if use_mps:
        samples = sample_mps(qcirc, k_samples=k_sample, seed=0)
    else:
        samples = sample_tn(qcirc, k_samples=k_sample, seed=0,
                             dtype="complex64" if use_gpu else "complex128")
    elapsed = time.time() - t0
    print(f"  sample time: {elapsed:.2f}s "
          f"({elapsed/k_sample*1000:.1f} ms/sample)")

    if validate_vs_cirq:
        idx_tn = np.array(
            [int("".join(str(b) for b in row), 2) for row in samples],
            dtype=np.int64,
        )
        xeb_tn = 2**n * p_C_cirq[idx_tn].mean() - 1
        idx_uniform = np.random.default_rng(2).integers(0, 1 << n, size=k_sample)
        xeb_uniform = 2**n * p_C_cirq[idx_uniform].mean() - 1
        print(f"  XEB(TN samples vs cirq p_C) = {xeb_tn:.4f}  (expect ~1.0)")
        print(f"  XEB(uniform vs cirq p_C)    = {xeb_uniform:.4f}  (expect ~0.0)")
    else:
        # No truth — self-consistency: score TN samples against TN amplitudes
        print("  no cirq truth available; computing TN-vs-TN consistency XEB...")
        probs_self = amplitudes_tn(qcirc, samples)
        xeb_self = 2**n * probs_self.mean() - 1
        # uniform baseline via TN amplitudes
        D = 1 << n
        uniform_int = np.random.default_rng(2).integers(0, D, size=min(k_sample, 1000))
        uniform_bits = np.array(
            [[(int(i) >> (n - 1 - q)) & 1 for q in range(n)] for i in uniform_int],
            dtype=np.uint8,
        )
        probs_uniform = amplitudes_tn(qcirc, uniform_bits)
        xeb_uniform = 2**n * probs_uniform.mean() - 1
        print(f"  XEB(TN samples scored vs TN amps) = {xeb_self:.4f}  (expect ~1.0)")
        print(f"  XEB(uniform scored vs TN amps)    = {xeb_uniform:.4f}  (expect ~0.0)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--circuit_seed", type=int, default=42)
    p.add_argument("--circuit_kind", choices=["boixo_v2", "sycamore"],
                    default="boixo_v2")
    p.add_argument("--gpu", action="store_true", help="use torch+cuda backend")
    p.add_argument("--mps", action="store_true", help="use CircuitMPS (truncated)")
    p.add_argument("--max_bond", type=int, default=None)
    p.add_argument("--k_sample", type=int, default=200)
    p.add_argument("--no_cirq", action="store_true",
                    help="skip cirq comparison (required for n>20)")
    args = p.parse_args()
    _smoke_test(n=args.n, depth=args.depth, circuit_seed=args.circuit_seed,
                  circuit_kind=args.circuit_kind,
                  use_gpu=args.gpu, use_mps=args.mps, max_bond=args.max_bond,
                  k_sample=args.k_sample,
                  validate_vs_cirq=not args.no_cirq)
