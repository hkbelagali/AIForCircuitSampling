"""Z-observable training: fit <Z_S>_θ to empirical <Z_S>.

  L(θ) = Σ_S  α_S · ( <Z_S>_θ − <Z_S>_empirical )^2

<Z_S>_θ uses a full-distribution forward over 2^n_qubits bitstrings —
tractable n_qubits ≲ 20.
"""
import numpy as np
import torch

from ..eval.z_observables import empirical_z_expectations, parity_matrix
from ..io.conventions import int_to_bits


def train_z_pauli(model, samples_int, supports, weights, n_qubits, *,
                    alpha=None, epochs=400, lr=2e-3, device=None,
                    verbose=False, log_every=80, logger=None,
                    stage_label="z_pauli"):
    """samples_int (k,) MSB-first ints; supports/weights from enumerate_z_supports.

    Returns final loss (float).
    """
    device = device or next(model.parameters()).device
    targets = torch.from_numpy(
        empirical_z_expectations(samples_int, supports, n_qubits)
    ).to(torch.float64).to(device)
    W = torch.from_numpy(parity_matrix(supports, n_qubits)).to(torch.float64).to(device)
    if alpha is None:
        alpha = np.ones(len(supports), dtype=np.float64)
    alpha_t = torch.from_numpy(np.asarray(alpha, dtype=np.float64)).to(device)
    all_bits_t = torch.from_numpy(
        int_to_bits(np.arange(1 << n_qubits, dtype=np.int64), n_qubits)
    ).float().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr / 100)
    final_loss = float("nan")
    for ep in range(epochs):
        logp = model.log_prob(all_bits_t).to(torch.float64)
        p = torch.softmax(logp, dim=0)
        loss = (alpha_t * (W @ p - targets).pow(2)).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        final_loss = float(loss)
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"  ep {ep:>4}: loss={final_loss:.4e}", flush=True)
        if logger is not None:
            logger.log(stage=stage_label, epoch=ep, n_epochs=epochs,
                        z_pauli_loss=final_loss,
                        max_weight=int(weights.max() if len(weights) > 0 else 0))
    return final_loss
