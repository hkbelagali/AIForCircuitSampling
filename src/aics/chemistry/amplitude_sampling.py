"""Sample bitstrings from |psi|^2 of a Hubbard ground state in a fixed sector.

State convention (matches aics.common.symmetry): a state integer has bit i
(0 <= i < L) for spin-up site i and bit (L + i) for spin-down site i, LSB
first. The bit array layout returned here is consistent: bits[:, 0] is up_0,
bits[:, L] is dn_0, etc. -- which is what ARTransformerConditional expects.
"""

import numpy as np


def sample_from_amplitudes(psi, sector_states_arr, L, k, rng):
    """Draw k bitstrings from |psi|^2.

    Args
    ----
    psi : (D,) ground state in canonical sector ordering
    sector_states_arr : (D,) state ints for the canonical sector basis
    L : number of sites (so n_positions = 2L)
    k : number of samples
    rng : numpy.random.Generator

    Returns
    -------
    bits : (k, 2L) int array of {0, 1}, LSB-first per Hubbard state convention
    state_ints : (k,) sampled state integers
    indices : (k,) indices into sector_states_arr
    """
    p = np.abs(psi) ** 2
    p = p / p.sum()
    indices = rng.choice(len(p), size=k, p=p)
    state_ints = np.asarray(sector_states_arr[indices], dtype=np.int64)
    n = 2 * L
    bits = np.zeros((k, n), dtype=np.int64)
    for i in range(n):
        bits[:, i] = (state_ints >> i) & 1
    return bits, state_ints, indices


def bits_to_state_int(bits, L):
    """(k, 2L) bit array -> (k,) state ints, LSB-first."""
    bits = np.asarray(bits, dtype=np.int64)
    n = bits.shape[1]
    powers = (1 << np.arange(n, dtype=np.int64))
    return (bits * powers).sum(axis=1)


def state_int_to_bits(state_ints, L):
    """(k,) state ints -> (k, 2L) bits, LSB-first."""
    state_ints = np.asarray(state_ints, dtype=np.int64)
    n = 2 * L
    out = np.zeros((state_ints.size, n), dtype=np.int64)
    for i in range(n):
        out[:, i] = (state_ints >> i) & 1
    return out
