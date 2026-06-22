"""Unbiased TN sampler for our 2-row Boixo-v2 RCS circuits.

Replaces quimb's `sample_chaotic` (which puts a uniform prior on
non-marginal qubits — the bias our advisor flagged) with two paths whose
marginals are computed exactly from the TN:

  1. `sample_pcz_exact` — contract the full statevector once, then draw
     from |psi|^2 via multinomial. Tractable while psi fits in memory
     (n <= ~26 for complex128, more for complex64). Bit-exact correct.

  2. `sample_pcz_marginal` — PCZ-style frugal sampling: at qubit q,
     contract the marginal P(b_q | b_<q) by projecting earlier bits on
     both bra and ket and tracing out later bits. Cost grows roughly
     n * (cost of doubled-TN contraction). Targets n > 26 on
     bounded-treewidth circuits.

Public API:
    cirq_to_pcz_tn(circ, qubits, dtype, device, leave_outputs_open=True)
        -> dict with keys
           tensors, tensor_bonds, edges, bond_dims, final_qubits,
           open_indices, n, qcirc (the quimb Circuit), cirq_circuit, qubits
    compute_amplitude(tn, bitstring)
        -> complex amplitude <z|psi>
    compute_amplitudes_batched(tn, bitstrings)
        -> 1d complex array of amplitudes
    sample_pcz(tn, *, k_samples, seed)
        -> (k, n) uint8 sample matrix, exact Born sampling

The artensor-shaped TN (tensors/bonds/etc.) is retained because (a) L1 tests
validate the adapter is structurally correct and (b) it gives us a clean way
to hand off to artensor's slicing if we ever need that GPU path.

Index/bitstring convention: bitstring[q] is qubit q. To match cirq's MSB-first
state vector ordering used by `exact_probabilities`, qubits[0] is the MSB.
"""
from __future__ import annotations

import numpy as np
import torch
import cirq


# ---------------------------------------------------------------------------
# Adapter: cirq.Circuit -> artensor-style structures (kept for tests + future
# integration with artensor's GPU slicing path).
# ---------------------------------------------------------------------------

def _cirq_op_to_tensor(op, dtype, device):
    u = cirq.unitary(op)
    k = len(op.qubits)
    assert u.shape == (2 ** k, 2 ** k)
    arr = u.reshape((2,) * (2 * k))
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=dtype, device=device)


def _build_artensor_dict(cirq_circuit, qubits, dtype, device):
    n = len(qubits)
    qubit_to_idx = {q: i for i, q in enumerate(qubits)}
    tensors, tensor_bonds = {}, {}
    for i in range(n):
        tensors[i] = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        tensor_bonds[i] = [f"0-{i}"]
    wire_loc = [0] * n
    next_tid = n
    for moment in cirq_circuit:
        for op in moment.operations:
            qs = [qubit_to_idx[q] for q in op.qubits]
            interleaved = []
            for q in qs:
                interleaved.append(f"{wire_loc[q] + 1}-{q}")  # out
                interleaved.append(f"{wire_loc[q]}-{q}")       # in
            inds = interleaved[0::2] + interleaved[1::2]
            tensors[next_tid] = _cirq_op_to_tensor(op, dtype, device)
            tensor_bonds[next_tid] = inds
            for q in qs:
                wire_loc[q] += 1
            next_tid += 1
    final_bond_for_q = {q: f"{wire_loc[q]}-{q}" for q in range(n)}
    qubit_to_final_tid = {}
    for tid, inds in tensor_bonds.items():
        for q in range(n):
            if final_bond_for_q[q] in inds:
                qubit_to_final_tid[q] = tid
    assert len(qubit_to_final_tid) == n
    final_qubits = set(qubit_to_final_tid.values())
    bond_dims = {b: 2.0 for b in set().union(*tensor_bonds.values())}
    return {
        "tensors": tensors,
        "tensor_bonds": tensor_bonds,
        "edges": set(bond_dims.keys()),
        "bond_dims": bond_dims,
        "final_qubits": final_qubits,
        "open_indices": [final_bond_for_q[q] for q in range(n)],
        "qubit_to_final_tid": qubit_to_final_tid,
        "wire_loc": wire_loc,
        "n": n,
        "dtype": dtype,
        "device": device,
    }


def cirq_to_pcz_tn(cirq_circuit, qubits, *, dtype=torch.complex64, device="cpu",
                    leave_outputs_open=True, use_mps=False):
    """Build the artensor-shaped TN dict + a quimb Circuit. Returns a dict.

    `leave_outputs_open` is accepted for API compatibility (we always do).
    `use_mps` controls whether the quimb side uses CircuitMPS (truncated) or
    qtn.Circuit (exact). Default: exact.
    """
    import tn_rcs
    art = _build_artensor_dict(cirq_circuit, qubits, dtype, device)
    qcirc, _ = tn_rcs.cirq_to_quimb(cirq_circuit, qubits, use_mps=use_mps)
    art["qcirc"] = qcirc
    art["cirq_circuit"] = cirq_circuit
    art["qubits"] = list(qubits)
    return art


# ---------------------------------------------------------------------------
# Amplitudes via quimb
# ---------------------------------------------------------------------------

def _bits_str_to_array(bs, n):
    if isinstance(bs, str):
        assert len(bs) == n, f"bitstring length {len(bs)} != n={n}"
        return np.array([int(c) for c in bs], dtype=np.uint8)
    arr = np.asarray(bs, dtype=np.uint8)
    assert arr.shape == (n,)
    return arr


def compute_amplitude(tn, bitstring, *, optimize="auto-hq"):
    """Single amplitude <z|psi> via quimb. bitstring[i] = qubit i."""
    n = tn["n"]
    bits = _bits_str_to_array(bitstring, n)
    s = "".join(str(int(b)) for b in bits)
    amp = tn["qcirc"].amplitude(s, optimize=optimize)
    if hasattr(amp, "item"):
        return complex(amp.item())
    return complex(amp)


def compute_amplitudes_batched(tn, bitstrings, *, optimize="auto-hq"):
    """Batched amplitudes. `bitstrings` is a list of '0/1' strings or (k,n) array."""
    n = tn["n"]
    if isinstance(bitstrings, (list, tuple)) and len(bitstrings) > 0 \
            and isinstance(bitstrings[0], str):
        rows = [b for b in bitstrings]
    else:
        bs = np.asarray(bitstrings, dtype=np.uint8)
        rows = ["".join(str(int(c)) for c in row) for row in bs]
    k = len(rows)
    out = np.empty(k, dtype=np.complex128)
    qcirc = tn["qcirc"]
    for i, s in enumerate(rows):
        amp = qcirc.amplitude(s, optimize=optimize)
        out[i] = complex(amp.item()) if hasattr(amp, "item") else complex(amp)
    return out


# ---------------------------------------------------------------------------
# Sampling: exact (full statevector) and marginal-conditional (PCZ-style)
# ---------------------------------------------------------------------------

def _full_statevector_quimb(qcirc, n, *, optimize="auto-hq"):
    """Contract the quimb TN to a full (2^n,) statevector. Tractable for
    n <= ~26. Returns numpy complex128 array of length 2^n in MSB-first
    qubit order (so index k -> bits = format(k, f'0{n}b'), bit 0 = qubit 0).
    """
    # quimb's `to_dense` returns a vector with multi-index (q0, q1, ..., q_{n-1})
    # flattened. Reshape to confirm axis ordering then flatten back.
    psi_tn = qcirc.psi
    # `to_dense` collects open inds in their order in qcirc.outputs; pin order
    # explicitly to qubit index 0..n-1.
    inds = [f"k{i}" for i in range(n)]
    psi = psi_tn.to_dense(inds, optimize=optimize)
    arr = np.asarray(psi).reshape(-1)
    return arr


def sample_pcz_exact(tn, k_samples, *, seed=0, optimize="auto-hq"):
    """Build the full statevector and multinomial-sample from |psi|^2.

    Returns (k_samples, n) uint8, where row[q] = sampled bit for qubit q.
    Indexing convention: MSB-first (matches `exact_probabilities`).
    """
    n = tn["n"]
    psi = _full_statevector_quimb(tn["qcirc"], n, optimize=optimize)
    probs = (psi.conj() * psi).real
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if s <= 0:
        raise RuntimeError("statevector has nonpositive total mass")
    probs = probs / s
    rng = np.random.default_rng(seed)
    idxs = rng.choice(probs.size, size=k_samples, replace=True, p=probs)
    out = np.empty((k_samples, n), dtype=np.uint8)
    # MSB-first: bit 0 = qubit 0 = highest-order bit
    for i, idx in enumerate(idxs):
        for q in range(n):
            out[i, q] = (idx >> (n - 1 - q)) & 1
    return out


def sample_pcz_marginal(tn, k_samples, *, seed=0, group_size=10,
                          optimize="auto-hq", dtype="complex64",
                          max_marginal_storage=2**20):
    """Frugal PCZ-style unbiased sampler via `quimb.Circuit.sample`.

    Sequential marginal-conditional: P(b_q | b_<q) is contracted from the
    lightcone-trimmed TN at each step, then b_q is drawn from it. `group_size`
    sets how many qubits share a single marginal contraction (larger = fewer
    contractions but each is more expensive).

    This is the path that scales beyond the exact-statevector regime (n > ~26).
    For our 2-row Boixo-v2 ladders at depth 10 the lightcone treewidth is
    bounded ~4, so each marginal is cheap and total cost ~ k_samples * n /
    group_size contractions.

    Returns (k_samples, n) uint8, row[q] = bit for qubit q. MSB-first compatible.
    """
    qcirc = tn["qcirc"]
    n = tn["n"]
    out = np.empty((k_samples, n), dtype=np.uint8)
    gen = qcirc.sample(
        C=k_samples, group_size=group_size, seed=seed,
        optimize=optimize, dtype=dtype,
        max_marginal_storage=max_marginal_storage,
    )
    for i, bitstring in enumerate(gen):
        out[i] = np.frombuffer(bitstring.encode("ascii"), dtype=np.uint8) - ord("0")
    return out


def sample_pcz(tn, *, k_samples, seed=0, mode="auto",
                group_size=10, optimize="auto-hq", dtype="complex64"):
    """Public sampling API.

    mode="auto"     — exact statevector if n <= 24, else marginal
    mode="exact"    — `sample_pcz_exact` (full statevector)
    mode="marginal" — `sample_pcz_marginal` (quimb.Circuit.sample)
    """
    n = tn["n"]
    if mode == "auto":
        mode = "exact" if n <= 24 else "marginal"
    if mode == "exact":
        return sample_pcz_exact(tn, k_samples, seed=seed, optimize=optimize)
    elif mode == "marginal":
        return sample_pcz_marginal(tn, k_samples, seed=seed,
                                     group_size=group_size,
                                     optimize=optimize, dtype=dtype)
    else:
        raise ValueError(f"unknown mode {mode!r}")
