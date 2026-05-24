"""Single-excitation generators and excitation-closure baseline for Hubbard CI.

A single excitation maps |...n_{i,sigma} ... n_{j,sigma}...> to its image under
c^dag_{j,sigma} c_{i,sigma} for any sites i (occupied) and j (empty) of the
same spin. The standard CI-singles closure of a reference determinant.

excitation_closure_with_sizes returns the chain of sets visited at each radius,
which the M1 driver uses to plot Delta E_var vs |S| as r grows.
"""


def single_excitations(x_int, L):
    """Yield all states reachable by one same-spin single excitation from x_int.
    Preserves (N_up, N_dn) by construction.
    """
    out = []
    L_mask = (1 << L) - 1
    up = x_int & L_mask
    dn = x_int >> L
    for spin_is_up, occ in ((True, up), (False, dn)):
        empties = (~occ) & L_mask
        for i in range(L):
            if not ((occ >> i) & 1):
                continue
            for j in range(L):
                if not ((empties >> j) & 1):
                    continue
                new_occ = (occ ^ (1 << i)) | (1 << j)
                new_x = ((dn << L) | new_occ) if spin_is_up else ((new_occ << L) | up)
                out.append(new_x)
    return out


def excitation_closure_with_sizes(seeds, L, max_radius, allowed_set=None):
    """Return dict {r: visited_set} for r = 0, 1, ..., until saturation or max_radius.

    visited_set at radius r is the set of states reachable from `seeds` in at
    most r single excitations, optionally restricted to `allowed_set`.
    """
    visited = set(int(s) for s in seeds)
    out = {0: set(visited)}
    frontier = set(visited)
    for r in range(1, max_radius + 1):
        new_frontier = set()
        for x in frontier:
            for y in single_excitations(x, L):
                if allowed_set is not None and y not in allowed_set:
                    continue
                if y not in visited:
                    new_frontier.add(y)
        if not new_frontier:
            break
        visited.update(new_frontier)
        out[r] = set(visited)
        frontier = new_frontier
    return out
