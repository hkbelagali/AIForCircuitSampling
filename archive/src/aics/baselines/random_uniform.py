"""Uniform-random baseline: sample basis-state subsets uniformly from a
caller-supplied allowed-support set (typically the GS irrep allowed support).
"""

import numpy as np


def random_subset(allowed, size, rng):
    """Uniform-without-replacement subset of `allowed` of given `size`."""
    allowed = np.asarray(allowed, dtype=int)
    if size >= allowed.size:
        return np.sort(allowed.copy())
    return np.sort(rng.choice(allowed, size=int(size), replace=False))
