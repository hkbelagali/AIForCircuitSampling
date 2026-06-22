"""Entropy estimators. KL / TV require the full (2^n,) distribution."""
import numpy as np


def entropy_mc_estimate(pC_at_samples):
    """Monte Carlo estimator of H(p_C) from Born-sampled bitstrings:

        H ≈ -mean(log p_C(z_i))   for   z_i ~ p_C

    `pC_at_samples` is p_C evaluated at each sample. Unbiased IFF the
    samples were drawn from p_C; biased otherwise. Returns nats.
    """
    return float(-np.log(np.maximum(np.asarray(pC_at_samples), 1e-30)).mean())


def H_uniform(n_qubits):
    """Entropy of the uniform distribution on n_qubits qubits, in nats."""
    return float(n_qubits * np.log(2))


def KL_to_uniform(pC):
    n_qubits = int(np.log2(len(pC)))
    uniform = 1.0 / (1 << n_qubits)
    nz = pC > 0
    return float((pC[nz] * np.log(pC[nz] / uniform)).sum())


def TV_to_uniform(pC):
    n_qubits = int(np.log2(len(pC)))
    return float(0.5 * np.abs(pC - 1.0 / (1 << n_qubits)).sum())
