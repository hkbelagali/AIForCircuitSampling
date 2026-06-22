"""Stage A sample-bundle .npz I/O.

Schema:
  train_bits     (k, n) uint8    MSB-first
  train_pC       (k,)   float64  p_C(train_bits[i])
  held_bits      (k_h, n) uint8    } chunk 0 only
  held_pC        (k_h,)   float64  }
  uniform_bits   (k_u, n) uint8    }
  uniform_pC     (k_u,)   float64  }
  meta           json string  + {provenance}
"""
import json
from pathlib import Path

import numpy as np

from ._repro import provenance


def save_samples(path, *, train_bits, train_pC,
                  held_bits=None, held_pC=None,
                  uniform_bits=None, uniform_pC=None, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta["provenance"] = provenance()
    save = {
        "train_bits": np.ascontiguousarray(train_bits, dtype=np.uint8),
        "train_pC": np.ascontiguousarray(train_pC, dtype=np.float64),
        "meta": json.dumps(meta),
    }
    if held_bits is not None:
        save["held_bits"] = np.ascontiguousarray(held_bits, dtype=np.uint8)
        save["held_pC"] = np.ascontiguousarray(held_pC, dtype=np.float64)
    if uniform_bits is not None:
        save["uniform_bits"] = np.ascontiguousarray(uniform_bits, dtype=np.uint8)
        save["uniform_pC"] = np.ascontiguousarray(uniform_pC, dtype=np.float64)
    np.savez(path, **save)


def load_samples(path):
    z = np.load(path, allow_pickle=True)
    out = {
        "train_bits": z["train_bits"],
        "train_pC": z["train_pC"],
        "meta": json.loads(str(z["meta"])),
    }
    for key in ("held_bits", "held_pC", "uniform_bits", "uniform_pC"):
        if key in z.files:
            out[key] = z[key]
    return out


def combine_chunks(chunk_paths, out_path):
    """Concatenate train_*; copy held/uniform from chunk 0."""
    chunk_paths = list(chunk_paths)
    if not chunk_paths:
        raise ValueError("no chunks to combine")
    chunks = [load_samples(p) for p in chunk_paths]
    meta = dict(chunks[0]["meta"])
    meta["chunk_provenance"] = [c["meta"].get("provenance") for c in chunks]
    meta["n_chunks_combined"] = len(chunks)
    save_samples(
        out_path,
        train_bits=np.concatenate([c["train_bits"] for c in chunks], axis=0),
        train_pC=np.concatenate([c["train_pC"] for c in chunks], axis=0),
        held_bits=chunks[0].get("held_bits"),
        held_pC=chunks[0].get("held_pC"),
        uniform_bits=chunks[0].get("uniform_bits"),
        uniform_pC=chunks[0].get("uniform_pC"),
        meta=meta,
    )
