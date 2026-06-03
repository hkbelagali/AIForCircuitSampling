"""Sanity-check the closed-form Marshall sign rule against ED for the
half-filled Heisenberg AFM on a bipartite 1D chain. Should match perfectly.
"""

import argparse
import numpy as np

from aics.spin.heisenberg import make_heisenberg_context, marshall_signs_spin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    args = p.parse_args()
    L = args.L

    ctx = make_heisenberg_context(L, J=1.0, pbc=True)
    print(f"L={L}  sector dim = {len(ctx.states)}  E_0 = {ctx.E_0:.6f}")

    # ED-derived signs, normalized so the argmax-|psi| component is positive.
    psi = ctx.psi_0
    ref = int(np.argmax(np.abs(psi)))
    psi = psi * (1 if psi[ref] > 0 else -1)
    sign_ed = np.where(np.abs(psi) < 1e-12, 1, np.sign(psi)).astype(np.int64)

    sign_mar = marshall_signs_spin(ctx.states, L)
    # Match up to a global sign.
    a = int((sign_ed == sign_mar).sum())
    b = int((sign_ed == -sign_mar).sum())
    best = max(a, b)
    print(f"Marshall matches ED on {best} / {len(ctx.states)} states "
          f"(positive convention {a}, negative {b}).")
    if best == len(ctx.states):
        print(f"PERFECT  -- Marshall is exact for Heisenberg.")
    else:
        # Show mismatches
        if a < b:
            sign_mar = -sign_mar
        mismatches = np.where(sign_ed != sign_mar)[0]
        print(f"Mismatches on {len(mismatches)} states:")
        for i in mismatches[:10]:
            s = int(ctx.states[i])
            print(f"  s={s:>4d}={s:0{L}b}  |psi|={abs(psi[i]):.4e}  "
                  f"ED={sign_ed[i]:+d}  MSR={sign_mar[i]:+d}")


if __name__ == "__main__":
    main()
