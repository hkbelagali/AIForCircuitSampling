"""NLL-trained RCS vs shadow-baseline vs Z-Pauli-trained RCS:
per-weight Z-observable RMS error as a function of k_train.

Two panels (n=8, n=12) showing the NLL pipeline's local-observable
accuracy alongside the trivial shadow baseline and (at n=8 only) our
Z-Pauli-trained data for direct comparison.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def load_grouped(dir_, k_field="k_train"):
    by_nk = defaultdict(lambda: defaultdict(list))
    for f in sorted(dir_.glob("*.json")):
        c = json.loads(f.read_text())
        by_nk[c["n"]][c[k_field]].append(c)
    return by_nk


def agg_err_by_weight(cells, key, w_list):
    """Return {w: (median, q25, q75)} aggregated across seeds."""
    out = {}
    for w in w_list:
        vals = [c[key].get(str(w), c[key].get(w)) for c in cells]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        a = np.array(vals, dtype=np.float64)
        out[w] = (float(np.median(a)),
                  float(np.quantile(a, 0.25)),
                  float(np.quantile(a, 0.75)))
    return out


def cumulative_rms(cells, key, wmax, n, normalize=True):
    """Cumulative error over all observables of weight 1..wmax.
    Both versions accumulate the TOTAL squared error
      S = sum_{w=1}^{wmax} |O_w| * (per-weight RMS)^2
        = sum_{|S|<=wmax} (<O>_theta - <O>)^2.
    Normalized: sqrt(S / N_tot)  -- RMS per observable
    Unnormalized: sqrt(S)        -- raw L2 of the error vector
    where N_tot = sum_{w=1}^{wmax} C(n, w).
    """
    import math
    vals = []
    for c in cells:
        d = c[key]
        S = 0.0
        N_tot = 0
        ok = True
        for w in range(1, wmax + 1):
            v = d.get(str(w), d.get(w))
            if v is None:
                ok = False
                break
            nw = math.comb(n, w)
            S += nw * (float(v) ** 2)
            N_tot += nw
        if ok:
            vals.append(np.sqrt(S / N_tot) if normalize else np.sqrt(S))
    if not vals:
        return None
    a = np.array(vals, dtype=np.float64)
    return (float(np.median(a)),
            float(np.quantile(a, 0.25)),
            float(np.quantile(a, 0.75)))


def plot_panel(ax, nll_data_n, wmax_list, n, normalize, title=""):
    colors = {wm: cm.viridis(i / max(1, len(wmax_list) - 1))
              for i, wm in enumerate(wmax_list)}

    ks = sorted(nll_data_n)
    sampled_in_legend = False
    for wm in wmax_list:
        med_m, lo_m, hi_m = [], [], []
        med_s, lo_s, hi_s = [], [], []
        ks_used = []
        for k in ks:
            cells = nll_data_n[k]
            em = cumulative_rms(cells, "err_by_weight_model", wm, n, normalize)
            es = cumulative_rms(cells, "err_by_weight_shadow", wm, n, normalize)
            if em is None:
                continue
            ks_used.append(k)
            med_m.append(em[0]); lo_m.append(em[1]); hi_m.append(em[2])
            med_s.append(es[0]); lo_s.append(es[1]); hi_s.append(es[2])
        if not med_m:
            continue
        ax.plot(ks_used, med_m, "o-", color=colors[wm], lw=1.8, ms=4,
                label=fr"NLL model, $w_{{\max}}={wm}$")
        ax.fill_between(ks_used, lo_m, hi_m, color=colors[wm], alpha=0.15)
        lbl = "sampled estimator" if not sampled_in_legend else None
        ax.plot(ks_used, med_s, "x--", color=colors[wm], lw=1.0, ms=5,
                alpha=0.7, label=lbl)
        sampled_in_legend = True

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k_{\rm train}$")
    if normalize:
        ax.set_ylabel(
            r"$\sqrt{\,\frac{1}{\#\mathcal{O}_{w \leq w_{\max}}}"
            r"\sum_{w \leq w_{\max}}(\langle O\rangle_\theta - \langle O\rangle)^2\,}$")
    else:
        ax.set_ylabel(
            r"$\sqrt{\,\sum_{w \leq w_{\max}}"
            r"(\langle O\rangle_\theta - \langle O\rangle)^2\,}$")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")

    nll = load_grouped(base / "m_rcs_nll_eval", k_field="k_train")

    wmax_list = [1, 2, 4, 8]

    for normalize, suffix in [(True, "norm"), (False, "unnorm")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        plot_panel(axes[0], nll[8], wmax_list, n=8,
                    normalize=normalize, title="n=8")
        plot_panel(axes[1], nll[12], wmax_list, n=12,
                    normalize=normalize, title="n=12")
        axes[0].legend(loc="best", fontsize=8, ncol=1)
        axes[1].legend(loc="best", fontsize=8, ncol=1)
        fig.tight_layout()
        out = base / f"rcs_nll_local_obs_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
