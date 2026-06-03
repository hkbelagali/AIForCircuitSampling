"""Exact <E> = <psi|H|psi> / <psi|psi> at a model checkpoint.

For L <= 8 the sector dim is at most ~5000, so iterating |psi(x)| over the
whole sector is cheap. This is the gold-standard energy used for plotting
M8 convergence curves and deciding "threshold reached" in M9.
"""

import numpy as np
import torch

from aics.chemistry.amplitude_sampling import state_int_to_bits


def model_energy_exact(model, ctx, device="cpu", batch=4096):
    """Returns exact <E> of the model's wavefunction restricted to the sector."""
    states = ctx.states
    bits_all = state_int_to_bits(states, ctx.L)
    N = len(states)
    model.eval()
    log_mag = np.empty(N, dtype=np.float64)
    for s in range(0, N, batch):
        chunk = bits_all[s:s + batch]
        with torch.no_grad():
            lm = model.log_psi_mag(
                torch.from_numpy(chunk.astype(np.int64)).long().to(device)
            ).cpu().numpy()
        log_mag[s:s + len(chunk)] = lm
    psi = ctx.signs.astype(np.float64) * np.exp(log_mag)
    norm = float(np.sqrt(np.sum(psi * psi)))
    if norm == 0.0:
        return float("nan")
    psi = psi / norm
    return float(np.dot(psi, ctx.H @ psi))


def model_energy_exact_signed(model, ctx, device="cpu", batch=4096):
    """Like model_energy_exact, but uses the model's learned sign head instead
    of ctx.signs. Requires model.learn_signs=True."""
    states = ctx.states
    bits_all = state_int_to_bits(states, ctx.L)
    N = len(states)
    model.eval()
    log_mag = np.empty(N, dtype=np.float64)
    soft_signs = np.empty(N, dtype=np.float64)
    for s in range(0, N, batch):
        chunk = bits_all[s:s + batch]
        bits_t = torch.from_numpy(chunk.astype(np.int64)).long().to(device)
        with torch.no_grad():
            log_mag[s:s + len(chunk)] = model.log_psi_mag(bits_t).cpu().numpy()
            soft_signs[s:s + len(chunk)] = model.soft_sign(bits_t).cpu().numpy()
    psi = soft_signs * np.exp(log_mag)
    norm = float(np.sqrt(np.sum(psi * psi)))
    if norm == 0.0:
        return float("nan")
    psi = psi / norm
    return float(np.dot(psi, ctx.H @ psi))


def model_energy_exact_spin(model, ctx, device="cpu", batch=4096):
    """Exact <E> for an ARRNNSpin model + HeisContext.

    Mirrors model_energy_exact but uses L-bit spin basis and Marshall signs.
    """
    from aics.spin.heisenberg import state_int_to_bits_spin
    states = ctx.states
    L = ctx.L
    bits_all = state_int_to_bits_spin(states, L)
    N = len(states)
    model.eval()
    log_mag = np.empty(N, dtype=np.float64)
    for s in range(0, N, batch):
        chunk = bits_all[s:s + batch]
        with torch.no_grad():
            lm = model.log_psi_mag(
                torch.from_numpy(chunk.astype(np.int64)).long().to(device)
            ).cpu().numpy()
        log_mag[s:s + len(chunk)] = lm
    psi = ctx.signs.astype(np.float64) * np.exp(log_mag)
    norm = float(np.sqrt(np.sum(psi * psi)))
    if norm == 0.0:
        return float("nan")
    psi = psi / norm
    return float(np.dot(psi, ctx.H @ psi))
