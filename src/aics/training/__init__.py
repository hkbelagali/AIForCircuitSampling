"""Training: NLL (with PT regularizer) + Z-Pauli (with curriculum)."""
from .nll import train_nll_pt, BATCH_SIZE, TOTAL_STEPS, MIN_EPOCHS, MAX_EPOCHS, LAMBDA_PT
from .z_pauli import train_z_pauli
from .pt_regularizer import pt_term
from .curriculum import weight_ascending, SCHEDULES
from ._trainer import (
    print_hardware, assert_device_available,
    save_checkpoint, load_checkpoint, JsonLogger,
)

__all__ = [
    "train_nll_pt", "train_z_pauli", "pt_term",
    "weight_ascending", "SCHEDULES",
    "print_hardware", "assert_device_available",
    "save_checkpoint", "load_checkpoint", "JsonLogger",
    "BATCH_SIZE", "TOTAL_STEPS", "MIN_EPOCHS", "MAX_EPOCHS", "LAMBDA_PT",
]
