"""cirq.Circuit → quimb.Circuit adapter. Internal — `aics.sampling.*` modules
build the quimb circuit on demand and operate on it.
"""
import cirq
import numpy as np
import quimb.tensor as qtn


def _resolve_to_backend(to_backend, dtype):
    """quimb 1.11 expects `to_backend` as a callable. Convert string
    shortcuts ('torch', 'torch-cuda', None) into the right callable.
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
    """Translate any cirq.Circuit + qubit list into a quimb `Circuit`
    (exact) or `CircuitMPS` (truncated).

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


def _resolve_qcirc(circ_or_qcirc, qubits, **build_kwargs):
    """Accept either (cirq.Circuit, qubits) or a prebuilt qcirc. Returns qcirc."""
    if isinstance(circ_or_qcirc, (qtn.Circuit, qtn.CircuitMPS)):
        return circ_or_qcirc
    if qubits is None:
        raise ValueError("must pass qubits when circ is a cirq.Circuit")
    qcirc, _ = cirq_to_quimb(circ_or_qcirc, qubits, **build_kwargs)
    return qcirc
