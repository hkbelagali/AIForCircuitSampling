"""M7: accuracy vs training-set size -- the sample-complexity experiment.

Headline question: how does accuracy on the full distribution improve with the
number of training samples k, for distributions of decreasing structure?

  Neel Hubbard (U/t=8)   -- strong-coupling, Neel-ordered, most structured
  regular Hubbard (U/t=4) -- intermediate coupling
  RCS (Porter-Thomas)     -- chaotic, least structured

Accuracy = 1 - TV(p, q_theta) = sum_x min(p, q) in [0, 1] (distribution
overlap), computed EXACTLY over the full support (no test-set Monte Carlo).
We also record the forward KL and the capacity-ceiling (best accuracy at
infinite data via soft-target training) so the curves are confound-controlled.

Both families get their exact symmetries quotiented out (Hubbard: (N_up, N_dn)
sector mask; RCS: none), so we measure residual learnability.

Resumable: every config checkpointed to results/m7_results.json.
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
    total_variation,
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
HUBBARD_VARIANTS = [("neel", 8.0), ("regular", 4.0)]   # label, U/t
HUBBARD_LS = [6]                # matched n = 12 only (headline is fixed-n)
RCS_NS = [12]
RCS_DEPTH = 12
K_VALUES = [10, 30, 100, 300, 1000]
N_SEEDS = 2
EPOCHS = 80
FLOOR_EPOCHS = 200
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
N_MATCH = 12                    # matched system size for the headline panel
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


def hubbard_p(L, U):
    setup = hubbard_gs_setup(L, 1.0, U, pbc=True)
    p = np.abs(setup["psi_0"]) ** 2
    p = p / p.sum()
    return p, sector_states(L, L // 2, L // 2)


# ---- train + exact metrics -------------------------------------------------

def _metrics(p, log_q):
    kl, qm = forward_kl(p, log_q)
    return dict(kl=kl, tv=total_variation(p, log_q), q_mass=qm)


def eval_rcs(n, p, k, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X = int_to_bits(rng.choice(len(p), size=k, p=p), n)
    model = ARTransformer(n, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS)
    train_ar(model, X, n_epochs=EPOCHS, lr=2e-3, batch_size=32)
    return _metrics(p, model_log_probs_rcs(model, n))


def floor_rcs(n, p):
    torch.manual_seed(0)
    model = ARTransformer(n, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS)
    train_softtarget(model, int_to_bits(np.arange(len(p)), n), p,
                     c_value=None, n_epochs=FLOOR_EPOCHS)
    return _metrics(p, model_log_probs_rcs(model, n))


def eval_hubbard(L, U, p, states, k, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X = state_int_to_bits(states[rng.choice(len(p), size=k, p=p)], L)
    c = np.full(k, U, dtype=np.float32)
    model = ARTransformerConditional(L, L // 2, L // 2, d_model=D_MODEL,
                                     n_layers=N_LAYERS, n_heads=N_HEADS)
    train_ar_conditional(model, X, c, n_epochs=EPOCHS, lr=2e-3, batch_size=32)
    return _metrics(p, model_log_probs_hubbard(model, states, L, U))


def floor_hubbard(L, U, p, states):
    torch.manual_seed(0)
    model = ARTransformerConditional(L, L // 2, L // 2, d_model=D_MODEL,
                                     n_layers=N_LAYERS, n_heads=N_HEADS)
    train_softtarget(model, state_int_to_bits(states, L), p,
                     c_value=U, n_epochs=FLOOR_EPOCHS)
    return _metrics(p, model_log_probs_hubbard(model, states, L, U))


def _done(res, key):
    return key in res and "tv" in res[key]


def run_sweep():
    res = load_results()

    for label, U in HUBBARD_VARIANTS:
        for L in HUBBARD_LS:
            n = 2 * L
            p, states = hubbard_p(L, U)
            H_p = shannon_entropy(p)
            tag = f"hubbard_{label}_L{L}"
            print(f"\n[{tag} n={n} U/t={U}] support={len(p)} H={H_p:.3f}", flush=True)
            fkey = f"{tag}_floor"
            if not _done(res, fkey):
                m = floor_hubbard(L, U, p, states)
                res[fkey] = {**m, "H": H_p, "n": n, "support": len(p), "U": U}
                save_results(res)
                print(f"  floor: acc={1-m['tv']:.4f} KL={m['kl']:.4f} "
                      f"(qmass={m['q_mass']:.3f})", flush=True)
            for k in K_VALUES:
                for s in range(N_SEEDS):
                    key = f"{tag}_k{k}_s{s}"
                    if _done(res, key):
                        continue
                    m = eval_hubbard(L, U, p, states, k, s)
                    res[key] = {**m, "n": n, "k": k, "seed": s,
                                "support": len(p), "H": H_p, "U": U}
                    save_results(res)
                    print(f"  k={k:>4} s={s}: acc={1-m['tv']:.4f} "
                          f"KL={m['kl']:.4f}", flush=True)

    for n in RCS_NS:
        p = rcs_p(n)
        H_p = shannon_entropy(p)
        tag = f"rcs_n{n}"
        print(f"\n[{tag}] support={len(p)} H={H_p:.3f}", flush=True)
        fkey = f"{tag}_floor"
        if not _done(res, fkey):
            m = floor_rcs(n, p)
            res[fkey] = {**m, "H": H_p, "n": n, "support": len(p)}
            save_results(res)
            print(f"  floor: acc={1-m['tv']:.4f} KL={m['kl']:.4f}", flush=True)
        for k in K_VALUES:
            for s in range(N_SEEDS):
                key = f"{tag}_k{k}_s{s}"
                if _done(res, key):
                    continue
                m = eval_rcs(n, p, k, s)
                res[key] = {**m, "n": n, "k": k, "seed": s,
                            "support": len(p), "H": H_p}
                save_results(res)
                print(f"  k={k:>4} s={s}: acc={1-m['tv']:.4f} KL={m['kl']:.4f}",
                      flush=True)

    return res


# ---- plots -----------------------------------------------------------------

def _series(res, prefix, metric):
    ks, mean, std = [], [], []
    for k in K_VALUES:
        vals = [res[f"{prefix}_k{k}_s{s}"][metric]
                for s in range(N_SEEDS) if f"{prefix}_k{k}_s{s}" in res]
        if vals:
            ks.append(k); mean.append(float(np.mean(vals))); std.append(float(np.std(vals)))
    floor = res.get(f"{prefix}_floor", {}).get(metric)
    return np.array(ks), np.array(mean), np.array(std), floor


HEADLINE_SERIES = [
    ("hubbard_neel", r"Hubbard $U/t=8$", "#2a9d8f", "o-"),
    ("hubbard_regular", r"Hubbard $U/t=4$", "#457b9d", "s-"),
    ("rcs", "RCS", "#e76f51", "D-"),
]


def _p_for_family(fam):
    L_match = N_MATCH // 2
    if fam == "hubbard_neel":
        return hubbard_p(L_match, 8.0)[0]
    if fam == "hubbard_regular":
        return hubbard_p(L_match, 4.0)[0]
    return rcs_p(N_MATCH)


def _uniform_overlap(p):
    return float(np.sum(np.minimum(p, 1.0 / len(p))))


def make_accuracy_plot(res):
    L_match = N_MATCH // 2
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    xr = K_VALUES[-1]
    label_x = xr * 1.25
    placed = []

    def label_line(y, text, color):
        if any(abs(y - yp) < 0.03 for yp in placed):
            return
        ax.text(label_x, y, text, color=color, fontsize=8,
                va="center", ha="left", clip_on=False)
        placed.append(y)

    for fam, label, color, fmt in HEADLINE_SERIES:
        prefix = f"{fam}_L{L_match}" if fam.startswith("hubbard") else f"{fam}_n{N_MATCH}"
        ks, tv_m, tv_s, tv_floor = _series(res, prefix, "tv")
        if ks.size == 0:
            continue
        ax.errorbar(ks, 1 - tv_m, yerr=tv_s, fmt=fmt, color=color, label=label,
                    capsize=3, markersize=6, linewidth=1.9)
        unif = _uniform_overlap(_p_for_family(fam))
        ax.axhline(unif, color=color, ls="--", alpha=0.5)
        label_line(unif, "uniform", color)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(K_VALUES[0] * 0.8, xr * 1.7)
    ax.set_xlabel("training samples")
    ax.set_ylabel("accuracy (1 − TV)")
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = RESULTS_PATH.parent / "m7_accuracy_vs_samples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out}", flush=True)


def make_kl_plot(res):
    """Supplementary: same three series, forward KL (log) vs k at matched n."""
    L_match = N_MATCH // 2
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    for fam, label, color, fmt in HEADLINE_SERIES:
        prefix = f"{fam}_L{L_match}" if fam.startswith("hubbard") else f"{fam}_n{N_MATCH}"
        ks, kl_m, kl_s, kl_floor = _series(res, prefix, "kl")
        if ks.size == 0:
            continue
        ax.errorbar(ks, kl_m, yerr=kl_s, fmt=fmt, color=color, label=label,
                    capsize=3, markersize=6, linewidth=1.9)
        if kl_floor is not None:
            ax.axhline(kl_floor, color=color, ls=":", alpha=0.55)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training samples  $k$")
    ax.set_ylabel(r"$D_{\rm KL}(p\,\|\,q_\theta)$  [nats]")
    ax.set_title(f"Forward KL vs training samples (exact, matched $n={N_MATCH}$)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = RESULTS_PATH.parent / "m7_kl_vs_samples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}", flush=True)


def print_table(res):
    print(f"\n=== accuracy (1 - TV), mean over seeds, matched n={N_MATCH} ===", flush=True)
    L_match = N_MATCH // 2
    print(f"  {'series':>26} {'ceil':>6} " + " ".join(f"k={k}" for k in K_VALUES), flush=True)
    for fam, label, _, _ in HEADLINE_SERIES:
        prefix = f"{fam}_L{L_match}" if fam.startswith("hubbard") else f"{fam}_n{N_MATCH}"
        ks, tv_m, tv_s, tv_floor = _series(res, prefix, "tv")
        accs = {int(k): 1 - t for k, t in zip(ks, tv_m)}
        row = "  ".join(f"{accs.get(k, float('nan')):5.3f}" for k in K_VALUES)
        ceil = (1 - tv_floor) if tv_floor is not None else float("nan")
        print(f"  {label:>26} {ceil:6.3f}  {row}", flush=True)


def main():
    res = run_sweep()
    make_accuracy_plot(res)
    make_kl_plot(res)
    print_table(res)


if __name__ == "__main__":
    main()
