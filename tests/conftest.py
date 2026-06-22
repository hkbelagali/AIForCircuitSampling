"""Pytest config: make prototype/ and archive/src/ importable during the
src/aics/ migration window.

After migration completes and prototype/ is moved to archive/prototype-v1/,
this file gets trimmed to only point at src/ (or removed entirely if the
package is installed editable).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Order matters: src/ first so post-migration imports prefer it; prototype/
# and archive/src/ stay reachable so tests can still cover legacy code paths
# during the transition.
for p in (ROOT / "src", ROOT / "prototype", ROOT / "archive" / "src"):
    s = str(p)
    if p.exists() and s not in sys.path:
        sys.path.insert(0, s)
