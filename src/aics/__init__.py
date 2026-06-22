"""aics — learning RCS output distributions from samples.

Sub-packages: circuits, sampling, models, training, eval, io.
Convenience top-level re-exports below; submodule paths still work.
"""
from .cell import train_cell
from .models import AutoregressiveRNN
from .sampling import sample_exact_tn, sample_chaotic, amplitudes_tn
from .eval import report
from .circuits import make_boixo_v2_rcs_circuit, exact_probabilities

__version__ = "0.2.0"
__all__ = [
    "train_cell",
    "AutoregressiveRNN",
    "sample_exact_tn", "sample_chaotic", "amplitudes_tn",
    "report",
    "make_boixo_v2_rcs_circuit", "exact_probabilities",
]
