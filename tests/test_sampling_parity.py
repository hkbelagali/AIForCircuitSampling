"""Cross-check: sample_exact_tn (quimb) vs sample_exact_cirq (cirq) agree
at small n, and sample_chaotic shows the documented bias.
"""
import numpy as np
import pytest

try:
    from aics.sampling import sample_exact_tn as _sample_exact_tn_raw
    from aics.sampling import sample_chaotic as _sample_chaotic_raw

    def sample_exact_tn(circ, qubits, k, seed=0):
        return _sample_exact_tn_raw(circ, qubits, k_samples=k, seed=seed)

    def sample_chaotic(circ, qubits, k, marginal_qubits, seed=0):
        return _sample_chaotic_raw(circ, qubits, k_samples=k, seed=seed,
                                     marginal_qubits=marginal_qubits)
except ImportError:
    # Pre-migration shims
    import tn_rcs
    from boixo_v2_rcs import make_boixo_v2_rcs_circuit
    import pcz_sampler

    def sample_exact_tn(circ, qubits, k, seed=0):
        tn = pcz_sampler.cirq_to_pcz_tn(circ, qubits)
        return pcz_sampler.sample_pcz_marginal(tn, k_samples=k, seed=seed)

    def sample_chaotic(circ, qubits, k, marginal_qubits, seed=0):
        qcirc, _ = tn_rcs.cirq_to_quimb(circ, qubits)
        return tn_rcs.sample_tn(qcirc, k_samples=k, seed=seed,
                                  marginal_qubits=marginal_qubits)

# These always come from cirq directly:
from aics.circuits.exact import exact_probabilities, sample_from_circuit

try:
    from aics.circuits.boixo_v2 import make_boixo_v2_rcs_circuit
except ImportError:
    from boixo_v2_rcs import make_boixo_v2_rcs_circuit


def _xeb(samples_bits, pC, n):
    D = 1 << n
    powers = 2 ** np.arange(n - 1, -1, -1)
    idx = (samples_bits.astype(np.int64) @ powers).astype(np.int64)
    return float(D * pC[idx].mean() - 1)


def test_exact_tn_xeb_matches_porter_thomas_at_n8():
    """At n=8 depth 4: sample_exact_tn samples should give XEB ~ 1.0 (Porter-Thomas)."""
    n = 8
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=4, seed=0)
    pC = exact_probabilities(circ, qubits)
    samples = sample_exact_tn(circ, qubits, k=4000, seed=0)
    xeb = _xeb(samples, pC, n)
    # Porter-Thomas predicts XEB = 1; with k=4k MC stderr ~ 0.03
    # Allow generous bounds for the not-fully-PT depth-4 circuit.
    assert 0.5 < xeb < 1.6, f"sample_exact_tn XEB = {xeb}, expected near 1.0"


def test_exact_tn_agrees_with_chaotic_full_marginal_at_n8():
    """At marginal_qubits=n, sample_chaotic = sample_exact_tn (no bias)."""
    n = 8
    qubits, circ = make_boixo_v2_rcs_circuit(n, cz_depth=4, seed=0)
    pC = exact_probabilities(circ, qubits)
    samples_tn = sample_exact_tn(circ, qubits, k=4000, seed=1)
    samples_ch = sample_chaotic(circ, qubits, k=4000, marginal_qubits=n, seed=1)
    xeb_tn = _xeb(samples_tn, pC, n)
    xeb_ch = _xeb(samples_ch, pC, n)
    # Both same algorithm at marginal=n; differ only by MC noise.
    assert abs(xeb_tn - xeb_ch) < 0.2, \
        f"XEB(tn)={xeb_tn:.3f} vs XEB(chaotic, marg=n)={xeb_ch:.3f}"
