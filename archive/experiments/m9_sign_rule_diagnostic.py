"""Diagnose the discrepancy between the analytic Heisenberg-Marshall sign rule
and the exact ED ground-state signs for the half-filled 1D Hubbard model.

Dumps, for each sector state at small L, a row containing:
  - state_int
  - bit decomposition by sublattice / spin
  - ED-derived sign (signs_from_psi)
  - Heisenberg MSR sign ((-1)^{N_dn on sublattice A})
  - ratio g(x) = ED_sign / MSR_sign  in {-1, +1}
  - candidate corrective parities (N_doubly_occ, N_up on B, ...)

The hope: g(x) is a clean (-1)^{f(x)} for some simple f(x), so we can bolt
that onto the analytic formula and avoid `signs_from_psi`.
"""

import argparse
import numpy as np

from aics.chemistry.hubbard_setup import hubbard_gs_setup
from aics.chemistry.marshall import marshall_signs_batch, signs_from_psi, _A_mask
from aics.common.symmetry import sector_states


def popcount(v):
    v = int(v)
    c = 0
    while v:
        c += v & 1
        v >>= 1
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--U", type=float, default=4.0)
    p.add_argument("--show-zero", action="store_true",
                   help="show rows with |psi_0| ~ 0 (sign is irrelevant there)")
    args = p.parse_args()

    L = args.L
    setup = hubbard_gs_setup(L, t=1.0, U=args.U, pbc=True)
    psi_0 = setup["psi_0"]
    states = sector_states(L, L // 2, L // 2)

    sign_ed = signs_from_psi(psi_0)
    sign_mar = marshall_signs_batch(states, L)
    ratio = (sign_ed * sign_mar).astype(np.int64)  # in {-1, +1}

    L_mask = (1 << L) - 1
    A_mask = _A_mask(L)
    B_mask = L_mask ^ A_mask

    print(f"L={L} U={args.U}  sector dim = {len(states)}")
    print(f"  ED-signs / Heisenberg-MSR ratio breakdown:")
    print(f"    +1 (match): {int((ratio > 0).sum())} states")
    print(f"    -1 (flip):  {int((ratio < 0).sum())} states")
    print()
    print(f"  header: state_int  up_bits   dn_bits   |psi|     "
          f"NupA NupB NdnA NdnB Ndo NdA NdB | ED MSR ratio")

    rows = []
    for i, s in enumerate(states):
        up = int(s) & L_mask
        dn = (int(s) >> L) & L_mask
        psi = float(psi_0[i])
        abs_psi = abs(psi)
        if not args.show_zero and abs_psi < 1e-12:
            continue
        n_up_A = popcount(up & A_mask)
        n_up_B = popcount(up & B_mask)
        n_dn_A = popcount(dn & A_mask)
        n_dn_B = popcount(dn & B_mask)
        n_doub = popcount(up & dn)
        n_doub_A = popcount(up & dn & A_mask)
        n_doub_B = popcount(up & dn & B_mask)
        ed = int(sign_ed[i])
        mr = int(sign_mar[i])
        rt = int(ratio[i])
        rows.append((int(s), up, dn, abs_psi, n_up_A, n_up_B, n_dn_A, n_dn_B,
                     n_doub, n_doub_A, n_doub_B, ed, mr, rt))
        print(f"  s={int(s):>5}  up={up:0{L}b}  dn={dn:0{L}b}  "
              f"|psi|={abs_psi:.4f}  "
              f"{n_up_A} {n_up_B} {n_dn_A} {n_dn_B} {n_doub} {n_doub_A} {n_doub_B} | "
              f"{ed:+d} {mr:+d} {rt:+d}")

    print()
    print("=== correlation of ratio (-1=flip, +1=match) against candidate parities ===")
    # Pull integer columns only so bitwise ops work.
    int_cols = np.array([(r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[13])
                         for r in rows], dtype=np.int64)
    NupA, NupB, NdnA, NdnB, Ndoub, NdA, NdB, rat = (int_cols[:, i] for i in range(8))
    parities = {
        "N_up_A":           NupA & 1,
        "N_up_B":           NupB & 1,
        "N_dn_A":           NdnA & 1,
        "N_dn_B":           NdnB & 1,
        "N_doub":           Ndoub & 1,
        "N_doub_A":         NdA & 1,
        "N_doub_B":         NdB & 1,
        "N_up_A+N_dn_A":    (NupA + NdnA) & 1,
        "N_up_B+N_dn_B":    (NupB + NdnB) & 1,
        "N_up_A+N_up_B":    (NupA + NupB) & 1,
        "N_up_A+N_doub":    (NupA + Ndoub) & 1,
        "N_up_B+N_doub":    (NupB + Ndoub) & 1,
        "N_dn_A+N_doub":    (NdnA + Ndoub) & 1,
        "N_dn_B+N_doub":    (NdnB + Ndoub) & 1,
        "N_up_A+N_dn_B":    (NupA + NdnB) & 1,
        "N_up_B+N_dn_A":    (NupB + NdnA) & 1,
        "N_up_A+N_doub_A":  (NupA + NdA) & 1,
        "N_up_B+N_doub_B":  (NupB + NdB) & 1,
    }
    ratio_neg = (rat < 0).astype(np.int64)  # 1 where ED disagrees with MSR
    print(f"  candidate parity        --> matches `ratio<0` on ? / {len(rows)} states")
    for name, par in parities.items():
        agree = int((par == ratio_neg).sum())
        print(f"    {name:<18s}  -->  {agree:>3d} / {len(rows)}  "
              f"({'PERFECT' if agree == len(rows) else ('PERFECT(inverse)' if agree == 0 else '')})")

    # ---- systematic GF(2) linear-algebra search ----
    print()
    print(f"=== brute-force GF(2) search for formula f(x) = sum_i alpha_i feat_i (mod 2) ===")
    # Features: each individual bit, each pair-product of bits.
    nbits = 2 * L
    state_ints = np.array([r[0] for r in rows], dtype=np.int64)
    feat_mat = []
    feat_names = []
    # Single-bit features
    for i in range(nbits):
        feat_mat.append(((state_ints >> i) & 1).astype(np.int64))
        feat_names.append(f"x_{i}")
    # Pair-product features
    for i in range(nbits):
        bi = (state_ints >> i) & 1
        for j in range(i + 1, nbits):
            bj = (state_ints >> j) & 1
            feat_mat.append((bi & bj).astype(np.int64))
            feat_names.append(f"x_{i}*x_{j}")
    # Triple-product features
    for i in range(nbits):
        bi = (state_ints >> i) & 1
        for j in range(i + 1, nbits):
            bj = (state_ints >> j) & 1
            for k in range(j + 1, nbits):
                bk = (state_ints >> k) & 1
                feat_mat.append((bi & bj & bk).astype(np.int64))
                feat_names.append(f"x_{i}*x_{j}*x_{k}")
    F = np.array(feat_mat).T % 2     # (n_rows, n_features) over GF(2)
    b = ratio_neg.copy() % 2

    # Solve F @ alpha = b over GF(2) via Gaussian elimination.
    aug = np.concatenate([F, b.reshape(-1, 1)], axis=1).astype(np.int64) % 2
    n_eq, n_var = F.shape
    pivot_col_of_row = []
    r = 0
    for c in range(n_var):
        if r >= n_eq:
            break
        # Find pivot
        piv = None
        for rr in range(r, n_eq):
            if aug[rr, c] == 1:
                piv = rr
                break
        if piv is None:
            continue
        aug[[r, piv]] = aug[[piv, r]]
        for rr in range(n_eq):
            if rr != r and aug[rr, c] == 1:
                aug[rr] = (aug[rr] + aug[r]) % 2
        pivot_col_of_row.append(c)
        r += 1
    # Check consistency
    consistent = True
    for rr in range(r, n_eq):
        if aug[rr, n_var] == 1:
            consistent = False
            break
    if not consistent:
        print("  NO solution in the (1-bit ∪ 2-bit-AND) feature space — "
              "the corrective factor is NOT a degree-≤2 polynomial over GF(2).")
        print("  Need higher-degree features or a different parameterization.")
    else:
        # Read off the solution; non-pivot vars are free, set to 0 for minimal soln.
        alpha = np.zeros(n_var, dtype=np.int64)
        for row_idx, pcol in enumerate(pivot_col_of_row):
            alpha[pcol] = aug[row_idx, n_var]
        chosen = [feat_names[i] for i, a in enumerate(alpha) if a == 1]
        print(f"  FOUND solution: ratio_flip(x) = "
              f"({'  XOR  '.join(chosen) if chosen else '0'})  (mod 2)")
        # Verify on the same L
        recompute = (F @ alpha) % 2
        ok = bool(np.all(recompute == b))
        print(f"  In-sample (L={L}): matches on {int((recompute == b).sum())} / {len(b)} states  "
              f"({'OK' if ok else 'MISMATCH'})")
        # Rank report (over-determined or under-determined?)
        rank_F = r
        print(f"  Feature matrix rank: {rank_F} (constraints={len(b)}, features={n_var}).  "
              f"Solution dof = {n_var - rank_F} -- if >0, formula is one of many fits.")

        # Test a UNIVERSAL conjecture: ratio_flip(x) = parity of N_dn on A.
        # This is x_{L+i} for even i (i.e., down bits on A-sites).
        conj = np.zeros(len(rows), dtype=np.int64)
        for ii in range(0, L, 2):
            conj ^= ((state_ints >> (L + ii)) & 1).astype(np.int64)
        match = int((conj == b).sum())
        print(f"  Universal conjecture: ratio_flip = parity(N_dn on A)")
        print(f"    Matches on {match}/{len(rows)} states  "
              f"({'GENERALIZES' if match == len(rows) else 'fails'})")

    # ---- Direct test: is the GS all-positive in our basis? ----
    n_pos = int((sign_ed[np.abs(psi_0) >= 1e-12] > 0).sum())
    n_neg = int((sign_ed[np.abs(psi_0) >= 1e-12] < 0).sum())
    print()
    print(f"=== Direct test: ED sign breakdown over nonzero amplitudes ===")
    print(f"  +1: {n_pos}    -1: {n_neg}   (N_up = N_dn = L/2 = {L//2}; L mod 4 = {L % 4})")
    if n_neg == 0:
        print(f"  ALL POSITIVE  ->  analytic rule for L={L} is just sign(x) = +1.")

    # ---- Permutation-parity-derived analytic formula ----
    # sign_BL(x) = (-1)^{P(x) + N_dn^A(x)} where P = #{(i,j): i > j, up_i=1, dn_j=1}
    print()
    print(f"=== Derived analytic formula: sign_BL(x) = (-1)^{{P(x) + N_dn^A(x)}} ===")
    print(f"  where P(x) = #{{(i,j) : i > j, up_i=1, dn_j=1}}")
    L_mask_loc = (1 << L) - 1
    up_arr = np.array(states, dtype=np.int64) & L_mask_loc
    dn_arr = (np.array(states, dtype=np.int64) >> L) & L_mask_loc

    # Compute P(x) for each state.
    def compute_P(up, dn, L):
        # P = sum_{i > j} up_i * dn_j
        # = sum_i up_i * (sum_{j < i} dn_j)
        total = 0
        for i in range(L):
            if (up >> i) & 1:
                # count dn bits at positions j < i
                lower_mask = (1 << i) - 1
                total += bin(dn & lower_mask).count("1")
        return total

    P_arr = np.array([compute_P(int(u), int(d), L) for u, d in zip(up_arr, dn_arr)])

    # N_dn^A
    A_mask_loc = _A_mask(L)
    N_dn_A = np.array([bin(int(d) & A_mask_loc).count("1") for d in dn_arr])

    predicted_sign = np.where(((P_arr + N_dn_A) & 1) == 0, 1, -1)
    # Compare to ED, accounting for global sign ambiguity (try both signs).
    match_a = int(((predicted_sign == sign_ed) | (np.abs(psi_0) < 1e-12)).sum())
    match_b = int(((-predicted_sign == sign_ed) | (np.abs(psi_0) < 1e-12)).sum())
    nonzero = int((np.abs(psi_0) >= 1e-12).sum())
    print(f"  +sign convention: matches ED on {match_a} / {len(states)}  ({nonzero} nonzero)")
    print(f"  -sign convention: matches ED on {match_b} / {len(states)}")
    best = max(match_a, match_b)
    if best == len(states):
        print(f"  PERFECT MATCH (up to global sign)  -->  derivation is correct!")
    else:
        # Show some example mismatches
        mismatches = predicted_sign != sign_ed if match_a >= match_b else -predicted_sign != sign_ed
        mismatches &= np.abs(psi_0) >= 1e-12
        idxs = np.where(mismatches)[0][:10]
        print(f"  Mismatch on {len(np.where(mismatches)[0])} / {nonzero} nonzero states.")
        print(f"  Example mismatches (first 10):")
        for idx in idxs:
            s_ = int(states[idx])
            u_, d_ = s_ & L_mask_loc, (s_ >> L) & L_mask_loc
            print(f"    s={s_:>4d}  up={u_:0{L}b} dn={d_:0{L}b}  "
                  f"P={P_arr[idx]}  N_dn^A={N_dn_A[idx]}  "
                  f"predicted={(1 if best == match_a else -1)*predicted_sign[idx]:+d}  ED={sign_ed[idx]:+d}")


if __name__ == "__main__":
    main()
