"""JSON I/O for Stage B training-cell results.

`save_result(path, fields)` stamps `provenance` (git commit + timestamp +
hostname + pid) into the written JSON. `load_result(path)` returns the
full dict including the stamp.

This keeps every result file traceable to (a) the exact code that
produced it and (b) the dataset bundle that was its input (which itself
carries its own provenance via `samples.save_samples`).
"""
import json
from pathlib import Path

from ._repro import provenance


def save_result(path, fields):
    """Write a result dict to `path` (.json). Adds provenance under
    `provenance` key. Overwrites any existing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(fields)
    payload["provenance"] = provenance()
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_result(path):
    """Read a result JSON file. Returns a dict."""
    return json.loads(Path(path).read_text())
