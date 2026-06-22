"""Hubbard L=4 NLL k-sweep — same protocol as the peaked/RCS sweeps but
target distribution is the half-filled Hubbard ground state embedded in
the full 2^N space (zero outside the sector). Trained with BitstringARRNN
(no sector mask, so the model has to learn the sector constraint from data).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from m9.hubbard import Hubbard
from rcs import run_rcs_xeb_cell


def main():
    n = 8
    L = 4
    ctx = Hubbard(L=L, U=4.0)
    D = 1 << n
    p_C = np.zeros(D, dtype=np.float64)
    p_C[ctx.states] = ctx.psi_0 ** 2
    p_C /= p_C.sum()
    print(f"Hubbard L={L}: nonzero strings = {(p_C > 0).sum()} / {D}")

    ks = [16, 32, 64, 128, 256, 1024, 10000]
    seeds = list(range(16))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(__file__).resolve().parents[1] / "results" / "m_hubbard_nll_k_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for k in ks:
        for seed in seeds:
            tag = f"k{k}_s{seed}"
            cell_path = out_dir / f"{tag}.json"
            if cell_path.exists():
                continue
            out = run_rcs_xeb_cell(
                n=n, depth=0, k_train=k, m_candidates=5000,
                seed=seed + 100000, d_hidden=64, epochs=400, lr=2e-3,
                batch_size=k, k_held=500, device=device, verbose=False,
                p_C_only=p_C,
            )
            cell_path.write_text(json.dumps(out))
        files = list(out_dir.glob(f"k{k}_s*.json"))
        fcls = [json.loads(f.read_text())["classical_fidelity"] for f in files]
        print(f"  k={k:>5}: F_cl med={np.median(fcls):.4f}  ({len(fcls)} seeds)",
              flush=True)
    print(f"total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
