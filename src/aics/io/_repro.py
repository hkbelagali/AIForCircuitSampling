"""Reproducibility helpers: embed git commit + timestamp + config snapshot
into every artifact (npz, json, png side-car) so we can trace back from
any plot to the exact code + flags that produced it.
"""
import datetime as _dt
import os
import socket
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHED_COMMIT = None


def git_commit():
    """Current HEAD short SHA, or '<dirty>'/'<no-git>' on failure. Cached
    per process — call after any new commit if you want a refreshed value
    (via `git_commit.cache_clear()`)."""
    global _CACHED_COMMIT
    if _CACHED_COMMIT is not None:
        return _CACHED_COMMIT
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        # Mark dirty if working tree differs
        diff = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _CACHED_COMMIT = sha if diff == 0 else f"{sha}-dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        _CACHED_COMMIT = "<no-git>"
    return _CACHED_COMMIT


def provenance(config=None):
    """Provenance stamp for a freshly-produced artifact: git commit, UTC
    timestamp, hostname, optional config dict.
    """
    out = {
        "git_commit": git_commit(),
        "timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    if config is not None:
        out["config"] = dict(config)
    return out
