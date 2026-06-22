"""Bar plots of p(x) for Hubbard L=4 GS, RCS n=8 depth-10, peaked n=8.
1x3 panels, log y-axis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import matplotlib.pyplot as plt
import numpy as np

from m9.hubbard import Hubbard
from aics.circuits.brickwork import make_rcs_circuit, grid_for
from aics.circuits.exact import exact_probabilities
from peaked import build_peaked_pC


def main():
    n = 8
    D = 1 << n

    # Hubbard L=4 GS p(x) embedded in full 2^n space
    ctx = Hubbard(L=4, U=4.0)
    p_hub = np.zeros(D, dtype=np.float64)
    p_hub[ctx.states] = ctx.psi_0 ** 2
    p_hub /= p_hub.sum()

    # RCS n=8, depth=10
    qubits, circ = make_rcs_circuit(*grid_for(n), 10, seed=0)
    p_rcs = exact_probabilities(circ, qubits)

    # Peaked n=8, d_R=8, d_P=4
    pk = build_peaked_pC(n=n, depth_rqc=8, depth_pqc=4, seed=0,
                          device="cuda", verbose=False)
    p_pk = pk["p_C"]

    xs = np.arange(D)
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    titles = ["Hubbard L=4 GS", "RCS n=8 depth=10", "Peaked n=8 (d_R=8, d_P=4)"]
    ps_sorted = [np.sort(p)[::-1] for p in [p_hub, p_rcs, p_pk]]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for row, (yscale, ylim) in enumerate([("linear", (0, None)),
                                            ("log", (1e-8, 1))]):
        for col, (p, color, title) in enumerate(zip(ps_sorted, colors, titles)):
            ax = axes[row, col]
            ax.bar(xs, p, color=color, width=1.0)
            ax.set_yscale(yscale)
            if row == 0:
                ax.set_title(title)
            if row == 1:
                ax.set_xlabel("rank (sorted by $p$)")
            ax.set_xlim(-1, D)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.3, axis="y", which="both")
        axes[row, 0].set_ylabel(r"$p(x)$")

    fig.tight_layout()
    out = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results/three_distributions_n8.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
