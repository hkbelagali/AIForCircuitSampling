"""Provenance stamp: git commit + timestamp + host, embedded in every artifact."""
import datetime as _dt
import os
import socket
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHED_COMMIT = None


def git_commit():
    global _CACHED_COMMIT
    if _CACHED_COMMIT is not None:
        return _CACHED_COMMIT
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        diff = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _CACHED_COMMIT = sha if diff == 0 else f"{sha}-dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        _CACHED_COMMIT = "<no-git>"
    return _CACHED_COMMIT


def provenance(config=None):
    out = {
        "git_commit": git_commit(),
        "timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    if config is not None:
        out["config"] = dict(config)
    return out
