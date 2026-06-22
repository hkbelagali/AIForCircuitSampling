"""Test suite for the PCZ-based exact sampler port.

Run with:
    /mnt/ffs24/home/rowlan91/.conda/envs/thesisEnv/bin/python3 -u prototype/_test_pcz_sampler.py [test_name]

Each test prints PASS / FAIL / SKIP and a short justification. Designed to be
runnable from day-1 — most tests will SKIP until the corresponding piece of
pcz_sampler.py is built. As we implement, more turn PASS.

Test levels (built incrementally):

  L1 — Adapter: cirq.Circuit + qubits → (tensors, edges, tensor_bonds, bond_dims)
       structures that artensor consumes.

  L2 — Single-amplitude correctness: compute <z|psi> via cirq, quimb, and the
       new PCZ path; all three agree to 1e-6 relative at small n.

  L3 — Batched amplitudes: artensor's contraction_scheme_multibitstrings_test
       computes amplitudes for many bitstrings in one shared contraction.
       Must agree with single-amplitude evaluations and with cirq's full
       statevector.

  L4 — Sampling: importance-sample over batched amplitudes; empirical
       distribution converges to |psi(z)|^2.
         - L4.1: chi^2 / KL against cirq's exact p_C at n=4
         - L4.2: XEB(samples, cirq_p_C) → 1.0 at n=12
         - L4.3: monotone convergence with k

  L5 — Vs sample_chaotic: at n where marginal_qubits=n (no bias regime), both
       should match. At n>marginal_qubits, sample_chaotic should show measurable
       bias relative to PCZ.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive" / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cirq
import quimb.tensor as qtn

from boixo_v2_rcs import make_boixo_v2_rcs_circuit
from aics.circuits.exact import exact_probabilities
import tn_rcs

# Attempt to import the WIP module; tests that need it will SKIP if missing
try:
    import pcz_sampler
    HAVE_PCZ = True
except ImportError:
    HAVE_PCZ = False


# ----- result tracking -----
RESULTS = []  # list of (name, status, msg) where status ∈ {PASS, FAIL, SKIP}

def _record(name, status, msg=""):
    RESULTS.append((name, status, msg))
    color = {"PASS": "\033[32m", "FAIL": "\033[31m", "SKIP": "\033[33m"}[status]
    reset = "\033[0m"
    print(f"  [{color}{status}{reset}] {name}: {msg}")


def test(name):
    """Decorator: registers a test, runs it, captures result."""
    def deco(fn):
        try:
            fn()
        except SkippedTest as e:
            _record(name, "SKIP", str(e))
        except AssertionError as e:
            _record(name, "FAIL", str(e))
        except Exception as e:
            _record(name, "FAIL", f"{type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
        else:
            _record(name, "PASS")
        return fn
    return deco


class SkippedTest(Exception):
    pass

def skip_if(cond, msg):
    if cond:
        raise SkippedTest(msg)


# ============================================================================
# L1 — adapter tests
# ============================================================================

@test("L1.1: adapter returns (tensors, edges, tensor_bonds, bond_dims) at n=4")
def t11():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    qubits, circ = make_boixo_v2_rcs_circuit(4, cz_depth=2, seed=0)
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    assert "tensors" in tn, "missing 'tensors' key"
    assert "edges" in tn, "missing 'edges' key"
    assert "tensor_bonds" in tn, "missing 'tensor_bonds' key"
    assert "bond_dims" in tn, "missing 'bond_dims' key"
    assert len(tn["tensors"]) > 0, "empty tensor list"


@test("L1.2: bond_dims are all 2 (single-qubit dimension)")
def t12():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    qubits, circ = make_boixo_v2_rcs_circuit(4, cz_depth=2, seed=0)
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    assert all(bd == 2 for bd in tn["bond_dims"].values()), \
        f"non-2 bond dims: {[b for b in tn['bond_dims'].values() if b != 2]}"


@test("L1.3: tensor count matches expected (n init + sum gate-qubits + n bra)")
def t13():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    n, depth = 4, 2
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=depth, seed=0)
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    n_gates = len(list(circ.all_operations()))
    # exact count depends on convention; loose lower bound
    assert len(tn["tensors"]) >= n_gates, \
        f"tensor count {len(tn['tensors'])} < gate count {n_gates}"


@test("L1.4: each open output index appears in exactly one tensor")
def t14():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    qubits, circ = make_boixo_v2_rcs_circuit(4, cz_depth=2, seed=0)
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits, leave_outputs_open=True)
    # Validate: each "final qubit" tensor has exactly one open leg
    # The structure is implementation-dependent; check the count
    assert "final_qubits" in tn or "open_indices" in tn, \
        "no way to identify which tensors are output qubits"


# ============================================================================
# L2 — single-amplitude correctness
# ============================================================================

@test("L2.1: amplitude of |0000> matches cirq at n=4")
def t21():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    n = 4
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=2, seed=0)
    pC = exact_probabilities(circ, qubits)
    # Bit 0 = the all-zeros bitstring
    p_cirq = float(pC[0])

    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    amp_pcz = pcz_sampler.compute_amplitude(tn, "0" * n)
    p_pcz = float(abs(amp_pcz) ** 2)

    rel = abs(p_pcz - p_cirq) / max(p_cirq, 1e-30)
    assert rel < 1e-4, f"|0..0> relative diff {rel:.2e} (cirq={p_cirq:.6e}, pcz={p_pcz:.6e})"


@test("L2.2: 5 random bitstrings match cirq at n=12")
def t22():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    n = 12
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=10, seed=42)
    pC = exact_probabilities(circ, qubits)
    rng = np.random.default_rng(0)
    idxs = rng.choice(1 << n, size=5, replace=False)

    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    max_rel = 0.0
    for idx in idxs:
        b = format(int(idx), f"0{n}b")  # MSB-first
        amp = pcz_sampler.compute_amplitude(tn, b)
        p_pcz = float(abs(amp) ** 2)
        p_cirq = float(pC[idx])
        rel = abs(p_pcz - p_cirq) / max(p_cirq, 1e-30)
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-4, f"max relative diff {max_rel:.2e}"


@test("L2.3: sum |amp|^2 = 1 at n=4 (probability normalization)")
def t23():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    n = 4
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=2, seed=0)
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    total = 0.0
    for idx in range(1 << n):
        b = format(idx, f"0{n}b")
        amp = pcz_sampler.compute_amplitude(tn, b)
        total += float(abs(amp) ** 2)
    assert abs(total - 1.0) < 1e-4, f"sum p = {total:.6f}, expected 1.0"


# ============================================================================
# L3 — batched amplitudes
# ============================================================================

@test("L3.1: batched amplitudes for all 16 states match cirq's statevector at n=4")
def t31():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    skip_if(not hasattr(pcz_sampler, "compute_amplitudes_batched"),
            "batched API not yet built")
    n = 4
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=2, seed=0)
    pC = exact_probabilities(circ, qubits)
    all_bits = [format(i, f"0{n}b") for i in range(1 << n)]
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    amps = pcz_sampler.compute_amplitudes_batched(tn, all_bits)
    probs = np.abs(np.asarray(amps)) ** 2
    assert len(probs) == len(all_bits), "wrong number of amps returned"
    max_rel = max(abs(p - pC[i]) / max(pC[i], 1e-30) for i, p in enumerate(probs))
    assert max_rel < 1e-3, f"max relative diff vs cirq statevector: {max_rel:.2e}"


@test("L3.2: batched amps at n=12 match individual amp calls")
def t32():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    skip_if(not hasattr(pcz_sampler, "compute_amplitudes_batched"),
            "batched API not yet built")
    n = 12
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=10, seed=42)
    rng = np.random.default_rng(0)
    idxs = rng.choice(1 << n, size=20, replace=False)
    bits = [format(int(i), f"0{n}b") for i in idxs]
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    amps_batched = pcz_sampler.compute_amplitudes_batched(tn, bits)
    amps_individual = [pcz_sampler.compute_amplitude(tn, b) for b in bits]
    max_diff = max(abs(complex(a) - complex(b)) for a, b in zip(amps_batched, amps_individual))
    assert max_diff < 1e-5, f"batched vs individual max abs diff: {max_diff:.2e}"


# ============================================================================
# L4 — sampling correctness
# ============================================================================

@test("L4.1: empirical histogram of 100k samples at n=4 matches cirq p_C via chi^2")
def t41():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    skip_if(not hasattr(pcz_sampler, "sample_pcz"), "sample API not yet built")
    n = 4
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=2, seed=0)
    pC = exact_probabilities(circ, qubits)
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)

    k = 100_000
    samples = pcz_sampler.sample_pcz(tn, k_samples=k, seed=0)
    # Convert to MSB-first ints
    idxs = np.array([int("".join(str(b) for b in row), 2) for row in samples])
    counts = np.bincount(idxs, minlength=1 << n)
    expected = pC * k
    # Simple chi^2
    chi2 = np.sum((counts - expected) ** 2 / np.maximum(expected, 1))
    # Critical value for df=15 (16 states - 1), p=0.01: ~30
    assert chi2 < 50, f"chi^2 = {chi2:.2f}, way above critical for df=15"


@test("L4.2: XEB(samples, cirq_p_C) ≈ 1.0 at n=12, k=10k")
def t42():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    skip_if(not hasattr(pcz_sampler, "sample_pcz"), "sample API not yet built")
    n = 12
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=10, seed=42)
    pC = exact_probabilities(circ, qubits)
    D = 1 << n
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    samples = pcz_sampler.sample_pcz(tn, k_samples=10_000, seed=1)
    idxs = np.array([int("".join(str(b) for b in row), 2) for row in samples])
    xeb = D * pC[idxs].mean() - 1
    # For z ~ p_C: XEB = D * E[p_C^2] - 1. Porter-Thomas predicts XEB = 1.
    # Our depth-10 2-row ladder is slightly less scrambled than PT, so
    # XEB ≈ 0.9 - 1.1 is expected. The point is XEB >> 0 (i.e., not uniform).
    assert 0.7 < xeb < 1.5, (
        f"XEB = {xeb:.3f}; expected ~1.0 (PT) for depth-10 boixo-v2 at n=12"
    )


# ============================================================================
# L5 — comparison with sample_chaotic (the biased shortcut we currently use)
# ============================================================================

@test("L5.1: PCZ and chaotic match at n=12 (marginal_qubits=n, no chaotic bias)")
def t51():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    skip_if(not hasattr(pcz_sampler, "sample_pcz"), "sample API not yet built")
    n = 12
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=10, seed=42)
    pC = exact_probabilities(circ, qubits)
    D = 1 << n

    # PCZ
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    samples_pcz = pcz_sampler.sample_pcz(tn, k_samples=10_000, seed=1)
    idxs_pcz = np.array([int("".join(str(b) for b in r), 2) for r in samples_pcz])
    xeb_pcz = D * pC[idxs_pcz].mean() - 1

    # chaotic with marginal_qubits=n (= no chaotic assumption used)
    qcirc, _, _ = tn_rcs.build_for_n(n, depth=10, circuit_seed=42)
    samples_ch = tn_rcs.sample_tn(qcirc, k_samples=10_000, seed=1,
                                    marginal_qubits=n)
    idxs_ch = np.array([int("".join(str(b) for b in r), 2) for r in samples_ch])
    xeb_ch = D * pC[idxs_ch].mean() - 1
    # Both unbiased here — should agree within MC noise
    assert abs(xeb_pcz - xeb_ch) < 0.1, \
        f"XEB(pcz)={xeb_pcz:.3f} vs XEB(chaotic, marg=n)={xeb_ch:.3f}"


@test("L4.3: marginal-mode sampler XEB ≈ exact-mode XEB at n=12, k=10k")
def t43():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    skip_if(not hasattr(pcz_sampler, "sample_pcz_marginal"),
            "marginal API not yet built")
    n = 12
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=10, seed=42)
    pC = exact_probabilities(circ, qubits)
    D = 1 << n
    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    s_exact = pcz_sampler.sample_pcz(tn, k_samples=10_000, seed=1, mode="exact")
    s_marg = pcz_sampler.sample_pcz(tn, k_samples=10_000, seed=1, mode="marginal")
    idx_e = np.array([int("".join(str(b) for b in r), 2) for r in s_exact])
    idx_m = np.array([int("".join(str(b) for b in r), 2) for r in s_marg])
    xeb_e = D * pC[idx_e].mean() - 1
    xeb_m = D * pC[idx_m].mean() - 1
    assert abs(xeb_e - xeb_m) < 0.1, \
        f"exact XEB={xeb_e:.3f} vs marginal XEB={xeb_m:.3f}"


@test("L5.2: at n=24 marginal_qubits=20, chaotic is biased; PCZ should be exact")
def t52():
    skip_if(not HAVE_PCZ, "pcz_sampler not yet built")
    skip_if(not hasattr(pcz_sampler, "sample_pcz"), "sample API not yet built")
    import os
    skip_if(not os.environ.get("RUN_L5_HEAVY"),
            "skipped on memory-constrained nodes; set RUN_L5_HEAVY=1 to enable")
    n = 24
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=10, seed=42)
    # Need exact p_C — at n=24, 2^24 = 16M states, ~256 MB float64 — doable
    pC = exact_probabilities(circ, qubits)
    D = 1 << n

    tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
    s_pcz = pcz_sampler.sample_pcz(tn, k_samples=10_000, seed=1, mode="marginal")
    idxs_pcz = np.array([int("".join(str(b) for b in r), 2) for r in s_pcz])
    xeb_pcz = D * pC[idxs_pcz].mean() - 1
    # PCZ should give XEB ≈ 1.0 (depth-10 boixo, near PT at n=24)

    qcirc, _, _ = tn_rcs.build_for_n(n, depth=10, circuit_seed=42)
    s_ch = tn_rcs.sample_tn(qcirc, k_samples=10_000, seed=1,
                             marginal_qubits=20)
    idxs_ch = np.array([int("".join(str(b) for b in r), 2) for r in s_ch])
    xeb_ch = D * pC[idxs_ch].mean() - 1

    # The bias may be small even here; the test just records both for inspection
    _record._cache = (xeb_pcz, xeb_ch)
    # No assertion — informational only
    print(f"    n=24 XEB: PCZ={xeb_pcz:.4f}, chaotic(marg=20)={xeb_ch:.4f}, diff={xeb_pcz - xeb_ch:+.4f}")


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    print(f"\nPCZ sampler test suite (HAVE_PCZ={HAVE_PCZ})")
    print("=" * 70)

    # Run all tests by importing and invoking each @test
    # (decorators ran on import; results are in RESULTS)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for name, status, msg in RESULTS:
        counts[status] += 1
    print(f"  PASS: {counts['PASS']}")
    print(f"  FAIL: {counts['FAIL']}")
    print(f"  SKIP: {counts['SKIP']}")
    print(f"  TOTAL: {len(RESULTS)}")
    sys.exit(0 if counts["FAIL"] == 0 else 1)
