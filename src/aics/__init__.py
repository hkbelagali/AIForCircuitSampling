"""aics — AI for Circuit Sampling: NLL / Z-observable learning of RCS
output distributions, organised around Ryan LaRose's RCS notebook.

Public sub-packages:

  aics.circuits   — circuit construction (boixo_v2 default, sycamore alt.) +
                     exact (cirq-based) reference
  aics.sampling   — chaotic (biased baseline, warned), exact_tn (default,
                     unbiased via quimb sequential marginal-conditional),
                     amplitudes
  aics.models     — AutoregressiveRNN (Ryan's LSTM)
  aics.training   — nll (with optional PT regularizer), z_pauli (with
                     optional curriculum), shared Trainer + checkpoint hooks
  aics.eval       — xeb, nll-metrics, z_observables, diversity, entropy
  aics.io         — bit/qubit conventions, npz I/O for samples, JSON I/O for
                     training results
"""

__version__ = "0.2.0"
