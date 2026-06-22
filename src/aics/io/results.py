"""Stage B result JSON I/O. Provenance stamp added on write."""
import json
from pathlib import Path

from ._repro import provenance


def save_result(path, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(fields)
    payload["provenance"] = provenance()
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_result(path):
    return json.loads(Path(path).read_text())
