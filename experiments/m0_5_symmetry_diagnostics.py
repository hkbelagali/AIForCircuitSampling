"""M0.5: symmetry sanity layer + entropy decomposition.

Three goals:

1. Identify the ground state's algebraic symmetry signature under translation
   (PBC only), reflection, and spin-flip.
2. Build the joint 1D-irrep projector, count the algebraically-allowed
   computational support, and check it matches the empirical (|a_x|^2 > tol)
   support count. If they match, the suppression observed in M0 is *fully*
   explained by exact symmetry.
3. Report an entropy decomposition:
       log D_sector  - log D_allowed   = "exact-symmetry compression"
       log D_allowed - H_actual        = "residual compressible structure"
       log D_sector  - H_actual        = "total compression"
   The second piece is the thing a generative model could conceivably learn
   beyond what symmetry already gives for free.

Writes:
  results/m0_5_pbc_vs_obc_amplitudes.png   (sorted |a_x|^2 histograms)
  results/m0_5_entropy_decomposition.png   (stacked-bar entropy breakdown)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from aics.chemistry.ed import ground_state
from aics.chemistry.hubbard import build_hubbard_1d
from aics.chemistry.symmetry_chem import (
    gs_irrep_signature,
    irrep_subspace_dim,
    project_to_irrep,
    reflection_op_1d,
    spin_flip_op,
    symmetry_allowed_support,
    translation_op_1d,
)
from aics.common.metrics import participation_ratio, shannon_entropy


SUPPORT_TOL = 1e-10


def diagnose(label, L, t, U, pbc):
    print(f"\n--- {label}: 1D Hubbard L={L}, U/t={U}, half-filling, pbc={pbc} ---")
    H = build_hubbard_1d(L, t, U=U, n_up=L // 2, n_dn=L // 2, pbc=pbc)
    Eg, psi = ground_state(H)
    a_sq = np.abs(psi) ** 2
    D = len(psi)
    supp = int(np.sum(a_sq > SUPPORT_TOL))
    Hp = shannon_entropy(a_sq)
    PR = participation_ratio(a_sq)
    print(f"  E_0 = {Eg:.6f}")
    print(f"  sector dim D                       = {D}")
    print(f"  empirical support (|a_x|^2 > tol)  = {supp}")
    print(f"  Shannon entropy H(p) [nats]        = {Hp:.4f}")
    print(f"  participation ratio PR             = {PR:.3f}")
    print(f"  log D                              = {np.log(D):.4f}")
    print(f"  log(support)                       = {np.log(supp):.4f}")

    ops = {}
    if pbc:
        ops["T"] = translation_op_1d(L, L // 2, L // 2)
    ops["R"] = reflection_op_1d(L, L // 2, L // 2)
    ops["S"] = spin_flip_op(L, L // 2, L // 2)

    eigs, warnings = gs_irrep_signature(psi, ops)
    print("  GS eigenvalues:")
    for name, lam in eigs.items():
        re, im = lam.real, lam.imag
        print(f"    {name} : {re:+.6f} {'+' if im >= 0 else '-'} {abs(im):.6f}i")
    for name, residual in warnings:
        print(f"    WARNING: psi not an eigenvector of {name} (residual {residual:.2e})")

    ops_orders_eigs = []
    if pbc:
        ops_orders_eigs.append((ops["T"], L, eigs["T"]))
    ops_orders_eigs.append((ops["R"], 2, eigs["R"]))
    ops_orders_eigs.append((ops["S"], 2, eigs["S"]))
    P = project_to_irrep(ops_orders_eigs)

    psi_proj = P @ psi.astype(np.complex128)
    norm_proj = float(np.linalg.norm(psi_proj))
    overlap = float(np.abs(np.vdot(psi, psi_proj)))
    n_allowed, _ = symmetry_allowed_support(P)
    d_irrep = irrep_subspace_dim(P)
    print(f"  ||P psi||                          = {norm_proj:.6f}  (expect 1.0)")
    print(f"  |<psi | P | psi>|                  = {overlap:.6f}  (expect 1.0)")
    print(f"  algebraically-allowed support      = {n_allowed}  "
          f"(empirical: {supp}; match? {n_allowed == supp})")
    print(f"  irrep subspace dim                 = {d_irrep}")
    print(f"  log(allowed)                       = {np.log(n_allowed):.4f}")

    return {
        "label": label,
        "psi": psi,
        "Eg": Eg,
        "D": D,
        "support": supp,
        "n_allowed": n_allowed,
        "d_irrep": d_irrep,
        "entropy": Hp,
        "PR": PR,
        "log_D": float(np.log(D)),
        "log_allowed": float(np.log(n_allowed)) if n_allowed > 0 else 0.0,
    }


def main():
    L = 4
    t = 1.0
    U = 4.0

    print("=== M0.5: symmetry sanity + entropy decomposition ===")

    pbc = diagnose("PBC", L, t, U, pbc=True)
    obc = diagnose("OBC", L, t, U, pbc=False)

    print("\n=== summary: 'how much apparent compression is exact symmetry?' ===")
    for d in (pbc, obc):
        sym_compr = d["log_D"] - d["log_allowed"]
        resid_compr = d["log_allowed"] - d["entropy"]
        total_compr = d["log_D"] - d["entropy"]
        print(f"\n  {d['label']}:")
        print(f"    log D                = {d['log_D']:.4f} nats   (sector max entropy)")
        print(f"    log allowed          = {d['log_allowed']:.4f} nats   (after exact symmetry)")
        print(f"    H(actual)            = {d['entropy']:.4f} nats   (the GS itself)")
        print(f"    exact-sym compression  = log D - log allowed = {sym_compr:.4f} nats")
        print(f"    residual structure     = log allowed - H     = {resid_compr:.4f} nats")
        print(f"    total compression      = log D - H           = {total_compr:.4f} nats")
        if total_compr > 1e-4:
            print(f"    -> fraction of total from exact symmetry = "
                  f"{100 * sym_compr / total_compr:.1f}%")

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)

    # --- Plot 1: amplitude histograms ---------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, d in ((axs[0], pbc), (axs[1], obc)):
        a_sq_sorted = np.sort(np.abs(d["psi"]) ** 2)[::-1]
        ax.bar(np.arange(len(a_sq_sorted)), a_sq_sorted, width=1.0)
        ax.set_yscale("log")
        ax.set_xlabel("basis state (ranked)")
        ax.set_title(f"{d['label']}   L={L}, U/t={U}, half-filling")
    axs[0].set_ylabel(r"$|a_x|^2$")
    fig.tight_layout()
    p1 = out_dir / "m0_5_pbc_vs_obc_amplitudes.png"
    fig.savefig(p1, dpi=150)

    # --- Plot 2: stacked-bar entropy decomposition --------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    labels = [d["label"] for d in (pbc, obc)]
    H_vals = np.array([d["entropy"] for d in (pbc, obc)])
    log_allowed = np.array([d["log_allowed"] for d in (pbc, obc)])
    log_D = np.array([d["log_D"] for d in (pbc, obc)])
    residual = log_allowed - H_vals
    sym = log_D - log_allowed

    x = np.arange(len(labels))
    ax2.bar(x, H_vals, width=0.5, label=r"$H(|a_x|^2)$ (residual to learn)",
            color="#3a7ca5")
    ax2.bar(x, residual, width=0.5, bottom=H_vals,
            label=r"$\log d_{\rm allowed} - H$ (compressible structure)",
            color="#f4a261")
    ax2.bar(x, sym, width=0.5, bottom=H_vals + residual,
            label=r"$\log D - \log d_{\rm allowed}$ (exact symmetry)",
            color="#e76f51")
    for i, d in enumerate((pbc, obc)):
        ax2.text(i, d["log_D"] + 0.04, f"$\\log D = {d['log_D']:.2f}$",
                 ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("entropy [nats]")
    ax2.set_title(f"Entropy decomposition: 1D Hubbard L={L}, U/t={U}")
    ax2.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    p2 = out_dir / "m0_5_entropy_decomposition.png"
    fig2.savefig(p2, dpi=150)

    print(f"\nWrote {p1}")
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
