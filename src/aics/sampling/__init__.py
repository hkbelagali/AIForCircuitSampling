"""Sampling and amplitude evaluation.

  sample_exact_tn   — default, unbiased (quimb sequential marginal-conditional)
  sample_chaotic    — biased baseline (chaotic-marginal assumption)
  amplitudes_tn     — p_C(z) for arbitrary bitstrings

For exact reference via cirq full statevector (n <= 26, used in tests),
import `aics.circuits.exact.sample_from_circuit` directly.
"""
from .chaotic import sample_chaotic
from .exact_tn import sample_exact_tn
from .amplitudes import amplitudes_tn, prepare_amplitude_tree
from ._quimb_circuit import cirq_to_quimb

__all__ = [
    "sample_chaotic", "sample_exact_tn",
    "amplitudes_tn", "prepare_amplitude_tree",
    "cirq_to_quimb",
]
