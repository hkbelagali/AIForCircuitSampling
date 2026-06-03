"""End-to-end setup helper for Stage 1 Task A on the Hubbard model.

Builds the sector Hamiltonian, computes the ground state, identifies the GS
symmetry signature (M0.5 machinery), constructs the joint 1D-irrep projector,
and returns the algebraically-allowed computational basis support.

If translation T's eigenvalue on the GS is complex (i.e. the GS lives in a
2D irrep of D_L like k = +/- pi/2), translation is dropped from the projector
and a warning is printed; the resulting allowed support is then a stricter
condition that may over-restrict — flag and inspect.
"""

import numpy as np

from aics.chemistry.ed import ground_state
from aics.chemistry.hubbard import build_hubbard_1d
from aics.chemistry.symmetry_chem import (
    gs_irrep_signature,
    project_to_irrep,
    reflection_op_1d,
    spin_flip_op,
    symmetry_allowed_support,
    translation_op_1d,
)


def _round_eig(eig, order):
    phase = np.angle(eig)
    m = int(round(phase * order / (2 * np.pi))) % order
    return np.exp(2j * np.pi * m / order)


def hubbard_gs_setup(L, t, U, n_up=None, n_dn=None, pbc=True, verbose=False):
    """Returns a dict with keys H, E_0, psi_0, allowed, irrep_eigs, n_allowed,
    P_irrep, used_T (whether translation was used in the projector)."""
    if n_up is None:
        n_up = L // 2
    if n_dn is None:
        n_dn = L // 2

    H = build_hubbard_1d(L, t, U, n_up, n_dn, pbc=pbc)
    E_0, psi_0 = ground_state(H)

    ops = {}
    if pbc:
        ops["T"] = translation_op_1d(L, n_up, n_dn)
    ops["R"] = reflection_op_1d(L, n_up, n_dn)
    if n_up == n_dn:
        ops["S"] = spin_flip_op(L, n_up, n_dn)

    eigs_raw, _ = gs_irrep_signature(psi_0, ops)
    irrep_eigs = {}
    used_T = False

    ops_orders_eigs = []
    if pbc:
        T_eig = eigs_raw["T"]
        if abs(T_eig.imag) < 1e-3 and abs(abs(T_eig.real) - 1.0) < 1e-3:
            T_clean = complex(round(T_eig.real))
            irrep_eigs["T"] = T_clean
            ops_orders_eigs.append((ops["T"], L, T_clean))
            used_T = True
        else:
            if verbose:
                print(f"  warning: GS T eigenvalue {T_eig} suggests a 2D irrep; "
                      f"dropping T projection (allowed support may over-restrict)")
            irrep_eigs["T_raw"] = T_eig
    for name in ("R", "S"):
        if name not in ops:
            continue
        lam = eigs_raw[name]
        clean = complex(round(lam.real)) if abs(lam.imag) < 1e-3 else _round_eig(lam, 2)
        irrep_eigs[name] = clean
        ops_orders_eigs.append((ops[name], 2, clean))

    P_irrep = project_to_irrep(ops_orders_eigs)
    n_allowed, allowed = symmetry_allowed_support(P_irrep)

    return {
        "H": H, "E_0": E_0, "psi_0": psi_0,
        "allowed": allowed, "n_allowed": n_allowed,
        "irrep_eigs": irrep_eigs, "used_T": used_T,
        "P_irrep": P_irrep,
    }
