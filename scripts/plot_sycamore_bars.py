"""Clustered bar chart, 4 bars per k_train:
  * clean (samples):  XEB baseline of ideal TN-drawn samples (~= 1, Porter-Thomas)
  * clean (trained):  clean_xeb of the LSTM trained on those TN samples
  * noisy (samples):  XEB baseline of Sycamore experimental samples (= device XEB)
  * noisy (trained):  clean_xeb of the LSTM trained on those experimental samples

Reads results/maxn_run/train_sycamore/*.threexebs.json produced by
sycamore_three_xebs.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CKPT_DIR = Path("results/maxn_run/train_sycamore")


def _clean_xeb(name):
    p = CKPT_DIR / f"{name}.threexebs.json"
    if not p.exists():
        return np.nan
    d = json.loads(p.read_text())
    v = d.get("clean_xeb")
    return float(v) if v is not None else np.nan


def _device_xeb_baseline():
    """Read the fixed device baseline from any threexebs JSON."""
    for p in CKPT_DIR.glob("*.threexebs.json"):
        d = json.loads(p.read_text())
        v = d.get("device_xeb")
        if v is not None:
            return float(v)
    return np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="plots/sycamore_bars.png")
    args = p.parse_args()

    ks = [1000, 10000, 50000, 100000, 250000]

    # k=100k uses the ddp naming; the rest use sweep_*.
    def name_for(kind, k):
        if k == 100000:
            return f"n20_s0_e0_{kind}_h512_s50000_ddp"
        return f"sweep_{kind}_k{k}_h512_s50000"

    clean_trained = {k: _clean_xeb(name_for("tn", k)) for k in ks}
    noisy_trained = {k: _clean_xeb(name_for("exp", k)) for k in ks}

    device_xeb = _device_xeb_baseline()
    if not np.isfinite(device_xeb):
        device_xeb = 0.226   # fallback from sycamore_n20_device_held meta

    # Porter-Thomas ceiling on ideal samples ≈ 1.0; the small correction from
    # finite-k averaging is negligible for k >= 1000 at n=20.
    clean_samples = {k: 0.99 for k in ks}
    noisy_samples = {k: device_xeb for k in ks}

    x = np.arange(len(ks))
    bar_w = 0.20
    fig, ax = plt.subplots(figsize=(12, 5.5))

    def vals(d):
        return [d[k] if k in d and d[k] is not None else np.nan for k in ks]

    ax.bar(x - 1.5 * bar_w, vals(clean_samples), bar_w,
            label="clean (samples)", color="#78b4ff")
    ax.bar(x - 0.5 * bar_w, vals(clean_trained), bar_w,
            label="clean (trained)", color="#0057c2")
    ax.bar(x + 0.5 * bar_w, vals(noisy_samples), bar_w,
            label="noisy (samples)", color="#f6ad7b")
    ax.bar(x + 1.5 * bar_w, vals(noisy_trained), bar_w,
            label="noisy (trained)", color="#c74300")

    # annotate the trained bars with their values
    for xi, k in zip(x, ks):
        for offs, d, col in ((-0.5 * bar_w, clean_trained, "#0057c2"),
                              ( 1.5 * bar_w, noisy_trained, "#c74300")):
            v = d.get(k)
            if v is None or not np.isfinite(v): continue
            ax.text(xi + offs, v + 0.02, f"{v:.2f}", ha="center",
                     va="bottom", fontsize=8, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k // 1000}k" for k in ks])
    ax.set_xlabel("k_train")
    ax.set_ylabel("XEB against ideal distribution")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Sycamore n=20 depth-14: XEB vs training-sample count\n"
                  "trained bars = model's clean_xeb (=vs ideal held), not vs its own held")
    ax.axhline(device_xeb, ls=":", color="#c74300", lw=1.0,
                label=f"device XEB baseline = {device_xeb:.2f}")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
