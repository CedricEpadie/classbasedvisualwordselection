"""Structured, per-run logging setup.

Every run gets its own log file `logs/run_<run_id>.log` in addition to
console output, so a long, multi-day run can always be audited after the
fact even if the terminal is gone.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def setup_logger(name: str, logs_dir: str | Path, run_id: str) -> logging.Logger:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_{run_id}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        # Avoid duplicate handlers if setup_logger is called more than once
        # (e.g. interactive re-runs in the same Python process).
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


@contextmanager
def log_step(logger: logging.Logger, step_name: str):
    """Context manager that logs start/end/duration/errors (with full
    stack trace) for a pipeline step."""
    logger.info("STEP START | %s", step_name)
    start = time.time()
    try:
        yield
    except Exception:
        duration = time.time() - start
        logger.exception("STEP FAILED | %s | duration=%.2fs", step_name, duration)
        raise
    else:
        duration = time.time() - start
        logger.info("STEP DONE  | %s | duration=%.2fs", step_name, duration)
