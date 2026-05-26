"""M7: sample-complexity learning curves.

Central question: how many training samples k are needed to reach a target
accuracy (exact forward KL <= eps) on the full distribution, and how does that
scale with system size n?

  chemistry (Hubbard, structured)   -- predict k_eps ~ poly(n)
  RCS (Porter-Thomas, chaotic)      -- predict k_eps ~ exp(n)

For each (family, n): a capacity-floor KL (train the architecture against exact
p) and a k-sweep of sample-limited KLs (averaged over seeds). Exact KL over the
full support; no test-set Monte Carlo.

Both families get the exact symmetries of their problem quotiented out:
  - Hubbard: (N_up, N_dn) sector mask (particle number) -> KL over the sector.
  - RCS: none -> KL over all 2^n.
so we measure the *residual* learnability, consistent with the M0.5 philosophy.

Resumable: every config is checkpointed to results/m7_results.json.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from aics.chemistry.amplitude_sampling import state_int_to_bits
from aics.chemistry.hubbard_setup import hubbard_gs_setup
from aics.circuits.brickwork import grid_for, make_rcs_circuit
from aics.circuits.exact import exact_probabilities, int_to_bits
from aics.common.symmetry import sector_states
from aics.eval.sample_complexity import (
    forward_kl,
    model_log_probs_hubbard,
    model_log_probs_rcs,
    shannon_entropy,
    train_softtarget,
)
from aics.models.ar_transformer import (
    ARTransformer,
    ARTransformerConditional,
    train_ar,
    train_ar_conditional,
)

torch.set_num_threads(4)

# ---- config ----------------------------------------------------------------
HUBBARD_LS = [4, 6, 8]          # n = 8, 12, 16
HUBBARD_U = 4.0
RCS_NS = [8, 10, 12]
RCS_DEPTH = 12
K_VALUES = [10, 30, 100, 300, 1000]
N_SEEDS = 2
EPOCHS = 80
FLOOR_EPOCHS = 200
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
KL_TARGETS = [2.0, 1.0, 0.5]    # for k_eps extraction
# ----------------------------------------------------------------------------

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "m7_results.json"


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def save_results(res):
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=2))


# ---- per-family setup ------------------------------------------------------

def rcs_p(n):
    n_rows, n_cols = grid_for(n)
    qubits, circuit = make_rcs_circuit(n_rows, n_cols, RCS_DEPTH, seed=1234)
    p = exact_probabilities(circuit, qubits)
    return p / p.sum()


def hubbard_p(L):
    setup = hubbard_gs_setup(L, 1.0, HUBBARD_U, pbc=True)
    p = np.abs(setup["psi_0"]) ** 2
    p = p / p.sum()
    states = sector_states(L, L // 2, L // 2)
    return p, states


# ---- train + exact-KL eval -------------------------------------------------

def eval_rcs(n, p, k, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    samples = rng.choice(len(p), size=k, p=p)
    X = int_to_bits(samples, n)
    model = ARTransformer(n, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS)
    train_ar(model, X, n_epochs=EPOCHS, lr=2e-3, batch_size=32)
    return forward_kl(p, model_log_probs_rcs(model, n))


def floor_rcs(n, p):
    torch.manual_seed(0)
    model = ARTransformer(n, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS)
    all_bits = int_to_bits(np.arange(len(p)), n)
    train_softtarget(model, all_bits, p, c_value=None, n_epochs=FLOOR_EPOCHS)
    return forward_kl(p, model_log_probs_rcs(model, n))


def eval_hubbard(L, p, states, k, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.choice(len(p), size=k, p=p)
    X = state_int_to_bits(states[idx], L)
    c = np.full(k, HUBBARD_U, dtype=np.float32)
    model = ARTransformerConditional(L, L // 2, L // 2, d_model=D_MODEL,
                                     n_layers=N_LAYERS, n_heads=N_HEADS)
    train_ar_conditional(model, X, c, n_epochs=EPOCHS, lr=2e-3, batch_size=32)
    return forward_kl(p, model_log_probs_hubbard(model, states, L, HUBBARD_U))


def floor_hubbard(L, p, states):
    torch.manual_seed(0)
    model = ARTransformerConditional(L, L // 2, L // 2, d_model=D_MODEL,
                                     n_layers=N_LAYERS, n_heads=N_HEADS)
    all_bits = state_int_to_bits(states, L)
    train_softtarget(model, all_bits, p, c_value=HUBBARD_U, n_epochs=FLOOR_EPOCHS)
    return forward_kl(p, model_log_probs_hubbard(model, states, L, HUBBARD_U))


# ---- main sweep ------------------------------------------------------------

def run_sweep():
    res = load_results()

    for L in HUBBARD_LS:
        n = 2 * L
        p, states = hubbard_p(L)
        H_p = shannon_entropy(p)
        print(f"\n[hubbard L={L} n={n}] support={len(p)} H={H_p:.3f} "
              f"log|supp|={np.log(len(p)):.3f}", flush=True)
        fkey = f"hubbard_L{L}_floor"
        if fkey not in res:
            kl, qm = floor_hubbard(L, p, states)
            res[fkey] = dict(kl=kl, q_mass=qm, H=H_p, n=n, support=len(p))
            save_results(res)
            print(f"  floor: KL={kl:.4f} (qmass={qm:.3f})", flush=True)
        for k in K_VALUES:
            for s in range(N_SEEDS):
                key = f"hubbard_L{L}_k{k}_s{s}"
                if key in res:
                    continue
                kl, qm = eval_hubbard(L, p, states, k, s)
                res[key] = dict(kl=kl, q_mass=qm, n=n, k=k, seed=s,
                                support=len(p), H=H_p)
                save_results(res)
                print(f"  k={k:>4} s={s}: KL={kl:.4f} (qmass={qm:.3f})", flush=True)

    for n in RCS_NS:
        p = rcs_p(n)
        H_p = shannon_entropy(p)
        print(f"\n[rcs n={n}] support={len(p)} H={H_p:.3f} "
              f"log|supp|={np.log(len(p)):.3f}", flush=True)
        fkey = f"rcs_n{n}_floor"
        if fkey not in res:
            kl, qm = floor_rcs(n, p)
            res[fkey] = dict(kl=kl, q_mass=qm, H=H_p, n=n, support=len(p))
            save_results(res)
            print(f"  floor: KL={kl:.4f} (qmass={qm:.3f})", flush=True)
        for k in K_VALUES:
            for s in range(N_SEEDS):
                key = f"rcs_n{n}_k{k}_s{s}"
                if key in res:
                    continue
                kl, qm = eval_rcs(n, p, k, s)
                res[key] = dict(kl=kl, q_mass=qm, n=n, k=k, seed=s,
                                support=len(p), H=H_p)
                save_results(res)
                print(f"  k={k:>4} s={s}: KL={kl:.4f} (qmass={qm:.3f})", flush=True)

    return res


# ---- aggregation + plots ---------------------------------------------------

def curve(res, family, size):
    prefix = f"{family}_L{size}" if family == "hubbard" else f"{family}_n{size}"
    ks, kl_mean, kl_std = [], [], []
    for k in K_VALUES:
        vals = [res[f"{prefix}_k{k}_s{s}"]["kl"] for s in range(N_SEEDS)
                if f"{prefix}_k{k}_s{s}" in res]
        if vals:
            ks.append(k)
            kl_mean.append(float(np.mean(vals)))
            kl_std.append(float(np.std(vals)))
    floor = res.get(f"{prefix}_floor", {}).get("kl", float("nan"))
    n = 2 * size if family == "hubbard" else size
    return np.array(ks), np.array(kl_mean), np.array(kl_std), floor, n


def k_eps(ks, kl_mean, target):
    """Smallest k with mean KL <= target (linear interp in log k). None if never."""
    below = np.where(kl_mean <= target)[0]
    if below.size == 0:
        return None
    j = below[0]
    if j == 0:
        return float(ks[0])
    k1, k2 = ks[j - 1], ks[j]
    y1, y2 = kl_mean[j - 1], kl_mean[j]
    if y1 == y2:
        return float(k2)
    t = (y1 - target) / (y1 - y2)
    return float(np.exp(np.log(k1) + t * (np.log(k2) - np.log(k1))))


def make_plots(res):
    out_dir = RESULTS_PATH.parent
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel A: Hubbard learning curves
    ax = axs[0]
    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(HUBBARD_LS):
        ks, m, sd, fl, n = curve(res, "hubbard", L)
        if ks.size == 0:
            continue
        c = cmap(i / max(1, len(HUBBARD_LS) - 1))
        ax.errorbar(ks, m, yerr=sd, fmt="o-", color=c, capsize=2,
                    label=f"L={L} (n={n})")
        if np.isfinite(fl):
            ax.axhline(fl, color=c, ls=":", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training samples $k$")
    ax.set_ylabel(r"$D_{\rm KL}(p\,\|\,q_\theta)$  [nats]")
    ax.set_title("Hubbard (structured)\ndotted = capacity floor")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # Panel B: RCS learning curves
    ax = axs[1]
    cmap = plt.get_cmap("plasma")
    for i, n in enumerate(RCS_NS):
        ks, m, sd, fl, _ = curve(res, "rcs", n)
        if ks.size == 0:
            continue
        c = cmap(i / max(1, len(RCS_NS) - 1))
        ax.errorbar(ks, m, yerr=sd, fmt="s-", color=c, capsize=2, label=f"n={n}")
        if np.isfinite(fl):
            ax.axhline(fl, color=c, ls=":", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training samples $k$")
    ax.set_title("RCS (Porter-Thomas)\ndotted = capacity floor")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # Panel C: k_eps(n) scaling
    ax = axs[2]
    target = KL_TARGETS[1]  # 1.0 nat
    for family, sizes, marker, color in (("hubbard", HUBBARD_LS, "o-", "#2a9d8f"),
                                         ("rcs", RCS_NS, "s-", "#e76f51")):
        ns, keps = [], []
        for size in sizes:
            ks, m, sd, fl, n = curve(res, family, size)
            if ks.size == 0:
                continue
            ke = k_eps(ks, m, target)
            ns.append(n)
            keps.append(ke if ke is not None else np.nan)
        ax.plot(ns, keps, marker, color=color,
                label=f"{family}", markersize=6)
    ax.set_yscale("log")
    ax.set_xlabel("system size $n$")
    ax.set_ylabel(rf"$k_\epsilon$ to reach $D_{{\rm KL}}\leq{target}$ nats")
    ax.set_title("sample complexity vs n\n(NaN = target not reached at max k)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    fig.suptitle("M7: exact-KL sample complexity -- chemistry vs RCS", y=1.02)
    fig.tight_layout()
    out = out_dir / "m7_sample_complexity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out}", flush=True)


def print_table(res):
    print("\n=== summary (mean KL over seeds) ===", flush=True)
    print(f"  {'family':>8} {'n':>3} {'support':>8} {'floor':>7} "
          + " ".join(f"k={k}" for k in K_VALUES), flush=True)
    for family, sizes in (("hubbard", HUBBARD_LS), ("rcs", RCS_NS)):
        for size in sizes:
            ks, m, sd, fl, n = curve(res, family, size)
            prefix = f"{family}_L{size}" if family == "hubbard" else f"{family}_n{size}"
            supp = res.get(f"{prefix}_floor", {}).get("support", "?")
            mdict = {int(kk): mm for kk, mm in zip(ks, m)}
            row = "  ".join(f"{mdict.get(k, float('nan')):6.3f}" for k in K_VALUES)
            print(f"  {family:>8} {n:>3} {str(supp):>8} {fl:>7.3f}  {row}", flush=True)


def main():
    res = run_sweep()
    make_plots(res)
    print_table(res)


if __name__ == "__main__":
    main()
