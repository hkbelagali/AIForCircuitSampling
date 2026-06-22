"""Diagnostic plots for sign-head restart bistability.

  1. Per-restart fidelity histogram across all (seed, cell) records — shows the
     bimodal "good basin / bad basin" structure.
  2. P(selected restart is bad) vs n_restarts used, for each selector.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=str, default="results/shadow_cells_v3")
    ap.add_argument("--out-dir", type=str, default="results/v3")
    ap.add_argument("--bad-fid-thresh", type=float, default=0.9)
    args = ap.parse_args()

    recs = [json.loads(p.read_text())
            for p in sorted(Path(args.cells).glob("L*.json"))]
    if not recs:
        print(f"no cells in {args.cells}"); return
    print(f"loaded {len(recs)} cells, "
          f"{sum(len(r['restart_records']) for r in recs)} restarts total")

    all_fids = np.concatenate([np.array([x["fid"] for x in r["restart_records"]])
                               for r in recs])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(all_fids, bins=np.linspace(0, 1, 41), color="steelblue",
            edgecolor="black", lw=0.6)
    ax.axvline(args.bad_fid_thresh, color="r", ls="--", lw=1.2,
               label=f"fail threshold (fid < {args.bad_fid_thresh})")
    ax.set_xlabel("per-restart fidelity")
    ax.set_ylabel("count")
    ax.set_title(f"Sign-head restart fidelity distribution "
                 f"({len(recs)} cells × {len(recs[0]['restart_records'])} restarts)")
    ax.legend()
    out = out_dir / "restart_fid_histogram.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}"); plt.close(fig)

    # Selector comparison: for each cell, simulate N restarts (subset of the
    # first N in the record) and report P(selected fid < threshold).
    n_total = len(recs[0]["restart_records"])
    Ns = [n for n in [1, 2, 4, 6, 8, 12, 16] if n <= n_total]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    selectors = ["train", "val", "oracle"]
    colors = {"train": "C0", "val": "C1", "oracle": "C2"}
    for sel in selectors:
        ps_fail = []
        ps_mean_fid = []
        for N in Ns:
            sel_fids = []
            for r in recs:
                rs = r["restart_records"][:N]
                fids = np.array([x["fid"] for x in rs])
                if sel == "train":
                    i = int(np.argmin([x["train_loss"] for x in rs]))
                elif sel == "val":
                    i = int(np.argmin([x["val_loss"] if x["val_loss"] is not None
                                        else x["train_loss"] for x in rs]))
                else:
                    i = int(np.argmax(fids))
                sel_fids.append(fids[i])
            sel_fids = np.array(sel_fids)
            ps_fail.append(float((sel_fids < args.bad_fid_thresh).mean()))
            ps_mean_fid.append(float(sel_fids.mean()))
        ax.plot(Ns, ps_fail, "o-", color=colors[sel], lw=1.8, ms=6,
                label=f"selector={sel}")
    ax.set_xlabel("n_restarts used")
    ax.set_ylabel(f"P(selected fid < {args.bad_fid_thresh})")
    ax.set_xscale("log")
    ax.set_title(f"Failure rate vs restart count "
                 f"({len(recs)} cells, L={recs[0]['L']}, "
                 f"k_total={recs[0]['k_total']})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    out = out_dir / "restart_failure_vs_n.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}"); plt.close(fig)

    # Per-cell numerical summary.
    print(f"\n=== per-cell summary (threshold={args.bad_fid_thresh}) ===")
    print(f"{'seed':>4} {'good/N':>7}", end="")
    for sel in selectors:
        for N in [1, 4, 8, 16]:
            if N <= n_total: print(f" {sel[:3]}_N{N}".rjust(9), end="")
    print()
    for r in recs:
        rs = r["restart_records"]
        n_good = sum(1 for x in rs if x["fid"] > args.bad_fid_thresh)
        print(f"{r['seed']:>4} {n_good:>3}/{len(rs):<3}", end="")
        for sel in selectors:
            for N in [1, 4, 8, 16]:
                if N > n_total: continue
                sub = rs[:N]
                if sel == "train":
                    i = int(np.argmin([x["train_loss"] for x in sub]))
                elif sel == "val":
                    i = int(np.argmin([x["val_loss"] for x in sub]))
                else:
                    i = int(np.argmax([x["fid"] for x in sub]))
                print(f" {sub[i]['fid']:>8.3f}", end="")
        print()


if __name__ == "__main__":
    main()
