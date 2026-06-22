"""Plot the RCS-vs-Peaked story: across both systems, the AR-RNN matches
the q_emp memorization baseline at every k. Peaked looks 'easier' by F_cl
because the peak dominates, but the held-out (corrected-ideal) test shows
neither system actually generalizes outside the training support."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def median_iqr(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def metric_curve(data, key_path, agg=median_iqr):
    """Extract a metric across k's. key_path is a list to traverse cell dict."""
    ks = sorted(int(k) for k in data["ks"])
    med, lo, hi = [], [], []
    for k in ks:
        vals = []
        for c in data["cells"][str(k)]:
            v = c
            for k_ in key_path:
                v = v[k_]
            if isinstance(v, dict):
                continue
            vals.append(v)
        m, q1, q3 = agg(vals)
        med.append(m); lo.append(q1); hi.append(q3)
    return ks, med, lo, hi


def line_with_band(ax, x, med, lo, hi, **kwargs):
    label = kwargs.pop("label", None)
    color = kwargs.pop("color", None)
    ls = kwargs.pop("ls", "-")
    marker = kwargs.pop("marker", "o")
    ax.plot(x, med, ls, color=color, marker=marker, label=label,
            lw=1.6, ms=4, **kwargs)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18)


def plot_xeb(ax, data, title, show_legend=True):
    ks, m_med, m_lo, m_hi = metric_curve(data, ["candidate_xeb"])
    _, emp_med, emp_lo, emp_hi = metric_curve(data, ["memorization", "q_emp", "xeb"])
    _, un_med, un_lo, un_hi = metric_curve(data, ["memorization", "q_unif_train", "xeb"])
    line_with_band(ax, ks, m_med, m_lo, m_hi, color="#1f77b4", label="model")
    line_with_band(ax, ks, emp_med, emp_lo, emp_hi, color="#ff7f0e",
                   label=r"$q_{\rm emp}$ (freq-weighted)", marker="s")
    line_with_band(ax, ks, un_med, un_lo, un_hi, color="#2ca02c",
                   label=r"$q_{\rm unif\,train}$ (flat over uniques)", marker="^")
    ax.axhline(data["ideal_xeb"], color="k", ls="--", lw=1, alpha=0.6,
               label=fr"ideal ({data['ideal_xeb']:.2f})")
    ax.set_xscale("log"); ax.set_xlabel("$k$")
    ax.set_ylabel("XEB")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    if show_legend:
        ax.legend(loc="best", fontsize=8)


def plot_kl(ax, data, title, show_legend=True):
    ks, m_med, m_lo, m_hi = metric_curve(data, ["kl_model_vs_truth"])
    _, emp_med, emp_lo, emp_hi = metric_curve(data, ["memorization", "q_emp", "kl"])
    line_with_band(ax, ks, m_med, m_lo, m_hi, color="#1f77b4", label="model")
    line_with_band(ax, ks, emp_med, emp_lo, emp_hi, color="#ff7f0e",
                   label=r"$q_{\rm emp}$", marker="s")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$k$"); ax.set_ylabel(r"KL$(\,\cdot\,\|\,p_C)$  (nats)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    if show_legend:
        ax.legend(loc="best", fontsize=8)


def plot_heldout(ax, data, title, show_legend=True):
    ks = sorted(int(k) for k in data["ks"])
    def fetch(key):
        med, lo, hi = [], [], []
        for k in ks:
            vals = [c["heldout_xeb"]["1"][key] for c in data["cells"][str(k)]]
            m, q1, q3 = median_iqr(vals)
            med.append(m); lo.append(q1); hi.append(q3)
        return med, lo, hi
    m_med, m_lo, m_hi = fetch("model")
    i_med, _, _ = fetch("ideal")
    u_med, _, _ = fetch("uniform")
    line_with_band(ax, ks, m_med, m_lo, m_hi, color="#1f77b4", label="model on $S^c$")
    ax.plot(ks, i_med, "k--", lw=1.4, label="corrected ideal", marker="x")
    ax.plot(ks, u_med, color="grey", ls=":", lw=1.4, label="uniform on $S^c$", marker=".")
    ax.set_xscale("log"); ax.set_xlabel("$k$")
    ax.set_ylabel("XEB on $S^c$ (held-out)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    if show_legend:
        ax.legend(loc="best", fontsize=8)


def plot_fcl_overlay(ax, data_rcs, data_peaked):
    """One axis, both systems, F_cl vs k."""
    for data, color, label in [(data_rcs, "#1f77b4", "RCS"),
                                 (data_peaked, "#d62728", "Peaked")]:
        ks, med, lo, hi = metric_curve(data, ["classical_fidelity"])
        line_with_band(ax, ks, med, lo, hi, color=color, label=label)
    ax.set_xscale("log"); ax.set_xlabel("$k$")
    ax.set_ylabel(r"$F_{\rm cl}(p_\theta, p_C)$")
    ax.set_title("Classical Fidelity Comparison - N=8 RCS vs N=8 Peaked")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8)


def main():
    base = Path("/mnt/ffs24/home/rowlan91/AIForCircuitSampling/results")
    rcs = load(base / "m_xeb_vs_kl_at_nsq.json")
    peaked = load(base / "m_peaked_xeb_vs_kl.json")

    # ---- 2x2 grid: rows = systems, cols = (XEB, KL) ----
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    plot_xeb(axes[0, 0], rcs, "N=8 RCS  —  XEB", show_legend=True)
    plot_kl(axes[0, 1], rcs, "N=8 RCS  —  KL", show_legend=True)
    plot_xeb(axes[1, 0], peaked, "N=8 Peaked  —  XEB", show_legend=False)
    plot_kl(axes[1, 1], peaked, "N=8 Peaked  —  KL", show_legend=False)
    fig.tight_layout()
    out1 = base / "rcs_vs_peaked_panels.png"
    fig.savefig(out1, dpi=150)
    print(f"wrote {out1}")

    # ---- F_cl overlay (compact, one panel) ----
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    plot_fcl_overlay(ax2, rcs, peaked)
    fig2.tight_layout()
    out2 = base / "rcs_vs_peaked_fcl_overlay.png"
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
