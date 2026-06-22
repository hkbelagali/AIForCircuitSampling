"""aics — learning RCS output distributions from samples.

Sub-packages: circuits, sampling, models, training, eval, io.
Top-level helpers: train_cell.
"""
from .cell import train_cell

__version__ = "0.2.0"
__all__ = ["train_cell"]
