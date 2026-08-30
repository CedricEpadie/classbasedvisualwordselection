"""Atomic, crash-safe I/O helpers used by every pipeline step.

The core guarantee: a reader never sees a partially-written file. We always
write to a temporary file in the same directory, flush + fsync, then
`os.replace` (atomic on POSIX and Windows) onto the final path.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_bytes(path: Path, data_writer) -> None:
    """`data_writer(fh)` writes bytes/text to the given open file handle."""
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            data_writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)

    def _write(fh):
        fh.write(json.dumps(obj, indent=2, default=str).encode("utf-8"))

    _atomic_write_bytes(path, _write)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def atomic_write_pickle(path: str | Path, obj: Any) -> None:
    path = Path(path)

    def _write(fh):
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    _atomic_write_bytes(path, _write)


def read_pickle(path: str | Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def atomic_write_npy(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)

    def _write(fh):
        np.save(fh, array)

    _atomic_write_bytes(path, _write)


def read_npy(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def path_exists_and_valid(path: str | Path) -> bool:
    """A file counts as valid if it exists and is non-empty (a crash mid
    atomic-write can never leave a non-empty final file thanks to `os.replace`,
    but an empty file could still exist from an older, pre-atomic bug)."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 0
