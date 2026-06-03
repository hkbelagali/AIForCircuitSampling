"""Particle-number / spin-sector enumeration for fermionic systems.

State-integer convention for L sites and 2L Jordan-Wigner qubits:
    bit i           (0 <= i < L)   = occupation of (site i, spin up)
    bit L + i                      = occupation of (site i, spin dn)

So a state x = (dn << L) | up where `up`, `dn` are each L-bit integers.
"""

from itertools import combinations

import numpy as np


def _set_bits(positions):
    x = 0
    for p in positions:
        x |= 1 << p
    return x


def sector_states(L, n_up, n_dn):
    """Sorted int64 array of states in the (n_up, n_dn) sector for L sites.

    Length is C(L, n_up) * C(L, n_dn). Sorted ascending so binary search via
    np.searchsorted gives the canonical index.
    """
    up_ints = [_set_bits(c) for c in combinations(range(L), n_up)]
    dn_ints = [_set_bits(c) for c in combinations(range(L), n_dn)]
    states = np.fromiter(
        ((d << L) | u for d in dn_ints for u in up_ints),
        dtype=np.int64,
        count=len(up_ints) * len(dn_ints),
    )
    states.sort()
    return states


def state_index(sorted_states, x):
    """Index of integer state x in the sorted_states array."""
    return int(np.searchsorted(sorted_states, x))
