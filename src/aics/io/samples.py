"""npz I/O for Stage A sample bundles.

Schema (.npz):
  train_bits      (k_max, n)         uint8, MSB-first (qubits[0] = MSB)
  train_pC        (k_max,)           float64, p_C(train_bits[i])
  held_bits       (k_held, n)        uint8
  held_pC         (k_held,)          float64
  uniform_bits    (k_uni, n)         uint8
  uniform_pC      (k_uni,)           float64
  meta            json-string        {n, depth, circuit_seed, sample_seed,
                                       sampler, ..., provenance}
"""
import json
from pathlib import Path

import numpy as np

from ._repro import provenance


def save_samples(path, *, train_bits, train_pC,
                  held_bits=None, held_pC=None,
                  uniform_bits=None, uniform_pC=None,
                  meta=None):
    """Write a Stage A sample bundle to `path` (.npz). `meta` is a dict;
    a provenance stamp (git commit, timestamp, hostname) is auto-added.
    """
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
    """Read a Stage A sample bundle. Returns dict with same keys as save_samples
    (held_*/uniform_* present iff they were saved). `meta` is parsed back to dict.
    """
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
    """Merge chunked Stage A samples into one canonical .npz.

    chunk_paths is a list of files produced with `chunk_idx` 0..K-1.
    train_bits/train_pC are concatenated; held_* and uniform_* come from
    chunk 0 only (matching tn_rcs_sample.py's behavior).

    The combined file's meta carries each chunk's provenance under
    'chunk_provenance' so we don't lose the per-chunk traceback.
    """
    chunk_paths = list(chunk_paths)
    if not chunk_paths:
        raise ValueError("no chunks to combine")
    chunks = [load_samples(p) for p in chunk_paths]
    train_bits = np.concatenate([c["train_bits"] for c in chunks], axis=0)
    train_pC = np.concatenate([c["train_pC"] for c in chunks], axis=0)
    meta = dict(chunks[0]["meta"])
    meta["chunk_provenance"] = [c["meta"].get("provenance") for c in chunks]
    meta["n_chunks_combined"] = len(chunks)
    save_samples(
        out_path,
        train_bits=train_bits, train_pC=train_pC,
        held_bits=chunks[0].get("held_bits"),
        held_pC=chunks[0].get("held_pC"),
        uniform_bits=chunks[0].get("uniform_bits"),
        uniform_pC=chunks[0].get("uniform_pC"),
        meta=meta,
    )
