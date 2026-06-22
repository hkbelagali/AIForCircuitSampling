"""Shared scaffolding for the two training loops: checkpoint save/load,
per-epoch JSON logging, hardware/device print at start.

Both `aics.training.nll.train_nll` and `aics.training.z_pauli.train_z_pauli`
use this — they share gradient clipping, optimizer/scheduler setup, and
checkpoint format, only the per-epoch loss computation differs.
"""
import json
import time
from pathlib import Path

import torch


def print_hardware(device, dtype=None, extra=None):
    """Loud single-line hardware banner. Call once at the start of any
    training-script entry point so SLURM logs make the configuration obvious.
    """
    line = [f"device={device}"]
    if device.startswith("cuda"):
        i = torch.cuda.current_device()
        line.append(f"gpu={torch.cuda.get_device_name(i)}")
        free, total = torch.cuda.mem_get_info(i)
        line.append(f"gpu_mem={free / 1e9:.1f}/{total / 1e9:.1f} GB free")
    line.append(f"torch_threads={torch.get_num_threads()}")
    if dtype is not None:
        line.append(f"dtype={dtype}")
    if extra:
        line.append(extra)
    print("[hardware] " + "  ".join(line), flush=True)


def assert_device_available(want_gpu: bool, requested_label: str = "--gpu"):
    """Hard error if `--gpu` was requested but CUDA isn't available.
    No silent fallback to CPU — the user said determinism over convenience.
    Returns the device string to use.
    """
    if want_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{requested_label} was requested but torch.cuda.is_available() is False. "
                "Either drop the flag (run on CPU) or move to a GPU node."
            )
        return "cuda"
    return "cpu"


def save_checkpoint(path, model, optimizer=None, scheduler=None,
                      epoch=None, best_loss=None, **extras):
    """Write a single .pt checkpoint. Always overwrites."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        **extras,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                      map_location="cpu"):
    """Restore from `save_checkpoint`. Returns the full payload dict so
    callers can read epoch / best_loss / any custom extras.
    """
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


class JsonLogger:
    """One JSON object per line — `plot.py` reads these for learning curves.

    Each call to `.log(**fields)` appends one row with an auto-added
    `epoch` and wall-time delta. Open the file in append mode so resume
    is sane.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a")
        self._t0 = time.time()

    def log(self, **fields):
        fields = {"t_sec": round(time.time() - self._t0, 3), **fields}
        self._fp.write(json.dumps(fields, default=str) + "\n")
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
