"""Random-walk neighborhood baseline.

Starting from a seed state, repeatedly takes a random single-excitation step,
accumulating distinct visited states (filtered to allowed_set) until |S| = size.
A stall (no valid excitations) triggers a restart from a random visited state.

Stochastic: trials must be averaged over many RNG seeds to compare against the
deterministic baselines (excitation closure, greedy SCI).
"""

import numpy as np

from aics.baselines.excitation import single_excitations


def random_walk_neighborhood(seed, size, allowed_set, L, rng, max_total_steps=None):
    """Construct a size-`size` random-walk subset rooted at `seed`, restricted to
    `allowed_set`. Returns a sorted list of state ints.
    """
    seed = int(seed)
    S = {seed}
    if size <= 1:
        return [seed]
    current = seed
    if max_total_steps is None:
        max_total_steps = 200 * size
    stalls = 0
    stall_limit = max(50, 5 * size)
    for _ in range(max_total_steps):
        if len(S) >= size:
            break
        excs = [x for x in single_excitations(current, L) if x in allowed_set]
        if not excs:
            stalls += 1
            if stalls > stall_limit:
                break
            current = next(iter(S)) if len(S) == 1 else list(S)[int(rng.integers(len(S)))]
            continue
        new = int(excs[int(rng.integers(len(excs)))])
        S.add(new)
        current = new
    return sorted(S)
