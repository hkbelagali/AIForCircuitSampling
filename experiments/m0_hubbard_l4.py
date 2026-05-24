"""M0: Hubbard ED on a 1D L=4 chain at half-filling.

Hand-checks:

  U=0 PBC: non-interacting; single-particle dispersion eps_k = -2t cos(2 pi k / L).
           For L=4, levels are {-2, 0, 0, 2}. Two electrons per spin in the lowest
           two distinct levels give E_0 = -2 per spin, total -4.

  U=0 OBC: dispersion eps_k = -2t cos(k pi / (L+1)), k = 1,...,L. Lowest two per
           spin sum to -2(cos(pi/5) + cos(2pi/5)). Total E_0 = -4(cos(pi/5) + cos(2pi/5)).

Both must match the ED output to numerical precision.
Then runs U=4 and emits the |a_x|^2 histogram (sorted, log y) to results/.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from aics.chemistry.ed import ground_state
from aics.chemistry.hubbard import build_hubbard_1d


def main():
    L = 4
    t = 1.0

    # ---- Hand-checks --------------------------------------------------------
    H0_pbc = build_hubbard_1d(L, t, U=0.0, n_up=L // 2, n_dn=L // 2, pbc=True)
    E0_pbc, _ = ground_state(H0_pbc)
    expected_pbc = -4.0
    assert np.isclose(E0_pbc, expected_pbc, atol=1e-10), (
        f"U=0 PBC L={L} half-filling: expected {expected_pbc}, got {E0_pbc}"
    )
    print(f"[hand-check] U=0 L={L} PBC half-filling: E_0 = {E0_pbc:.10f}  "
          f"(expected {expected_pbc})  PASS")

    H0_obc = build_hubbard_1d(L, t, U=0.0, n_up=L // 2, n_dn=L // 2, pbc=False)
    E0_obc, _ = ground_state(H0_obc)
    expected_obc = -4.0 * (np.cos(np.pi / 5) + np.cos(2 * np.pi / 5))
    assert np.isclose(E0_obc, expected_obc, atol=1e-10), (
        f"U=0 OBC L={L} half-filling: expected {expected_obc}, got {E0_obc}"
    )
    print(f"[hand-check] U=0 L={L} OBC half-filling: E_0 = {E0_obc:.10f}  "
          f"(expected {expected_obc:.10f})  PASS")

    # ---- Main run -----------------------------------------------------------
    U = 4.0
    H = build_hubbard_1d(L, t, U=U, n_up=L // 2, n_dn=L // 2, pbc=True)
    Eg, psi = ground_state(H)
    a_sq = np.abs(psi) ** 2
    a_sq_sorted = np.sort(a_sq)[::-1]

    print(f"\nU/t={U} L={L} PBC half-filling:")
    print(f"  sector dim    = {len(psi)}")
    print(f"  E_0           = {Eg:.6f}")
    print(f"  top-5 |a_x|^2 = {a_sq_sorted[:5]}")
    print(f"  IPR           = {1.0 / float(np.sum(a_sq ** 2)):.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "m0_hubbard_l4_u4_amplitudes.png"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(len(a_sq_sorted)), a_sq_sorted, width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("basis state (ranked by amplitude)")
    ax.set_ylabel(r"$|a_x|^2$")
    ax.set_title(f"1D Hubbard $L={L}$ PBC half-filling, $U/t={U}$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
