"""Execution-state manager: the core of the checkpointing / resume-on-crash
system.

Every pipeline step is registered here before it runs and marked
`done`/`failed` after. On the next launch, `PipelineState.should_run` tells
the orchestrator whether a step's cached output is still valid (same
config hash) so it can be skipped instead of recomputed.

The state file itself is written atomically (see `utils.io_utils`), so a
crash mid-write can never corrupt it.

The state is namespaced per *dataset* (not per run_id): a run_id changes
whenever any part of the config changes (see `config.ensure_run_id`), but
most steps only care about a small, relevant slice of the config (e.g.
preprocessing only cares about `preprocessing`, not about which
classifiers are enabled). Namespacing by dataset instead means that slice
comparison (`should_run`'s `config_hash` check, computed from a
`cfg.section_hash(...)` scoped to what that specific step actually
depends on) is what decides whether to skip a step, so an unrelated config
change (e.g. adding a classifier) does not spuriously invalidate
preprocessing/feature-extraction/vocabulary/histogram steps that were
already computed for that dataset.
"""
from __future__ import annotations

import datetime
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from src.utils.io_utils import atomic_write_json, path_exists_and_valid, read_json


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class StepRecord:
    name: str
    status: StepStatus = StepStatus.PENDING
    config_hash: Optional[str] = None
    timestamp: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["status"] = self.status.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "StepRecord":
        d = dict(d)
        d["status"] = StepStatus(d["status"])
        return StepRecord(**d)


class PipelineState:
    """Loads/saves `state/pipeline_state_<namespace>.json` (namespace is
    normally the dataset name — see module docstring) and answers the
    single question every step needs answered: "has this exact step, with
    this exact (step-scoped) config, already completed successfully?"
    """

    def __init__(self, state_dir: str | Path, namespace: str):
        self.state_dir = Path(state_dir)
        self.namespace = namespace
        self.state_path = self.state_dir / f"pipeline_state_{namespace}.json"
        self.steps: Dict[str, StepRecord] = {}
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if path_exists_and_valid(self.state_path):
            raw = read_json(self.state_path)
            self.steps = {k: StepRecord.from_dict(v) for k, v in raw.get("steps", {}).items()}

    def _save(self) -> None:
        payload = {
            "namespace": self.namespace,
            "updated_at": datetime.datetime.now().isoformat(),
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
        }
        atomic_write_json(self.state_path, payload)

    # ------------------------------------------------------------------ #
    def should_run(self, step_name: str, config_hash: str) -> bool:
        """True if the step needs to (re)run: never ran, failed previously,
        or ran with a different (step-scoped) configuration."""
        rec = self.steps.get(step_name)
        if rec is None:
            return True
        if rec.status != StepStatus.DONE:
            return True
        if rec.config_hash != config_hash:
            return True
        return False

    def mark_running(self, step_name: str, config_hash: str) -> None:
        self.steps[step_name] = StepRecord(
            name=step_name,
            status=StepStatus.RUNNING,
            config_hash=config_hash,
            timestamp=datetime.datetime.now().isoformat(),
        )
        self._save()

    def invalidate(self, step_name: str) -> None:
        """Force a step back to "needs to run": drop its record entirely so
        the next `should_run` call returns True regardless of config hash.

        Used when a step's cache hit turns out to be untrustworthy -- e.g.
        its config hash still matches, but one or more of its expected
        output files are missing on disk (typically because the dataset
        gained/lost files after the step was marked done). A no-op if the
        step has no recorded state.
        """
        if step_name in self.steps:
            del self.steps[step_name]
            self._save()

    def mark_done(self, step_name: str, duration_seconds: float) -> None:
        rec = self.steps[step_name]
        rec.status = StepStatus.DONE
        rec.duration_seconds = duration_seconds
        rec.timestamp = datetime.datetime.now().isoformat()
        self._save()

    def mark_failed(self, step_name: str, exc: BaseException) -> None:
        rec = self.steps.setdefault(step_name, StepRecord(name=step_name))
        rec.status = StepStatus.FAILED
        rec.error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        rec.timestamp = datetime.datetime.now().isoformat()
        self._save()

    def summary(self) -> Dict[str, str]:
        return {name: rec.status.value for name, rec in self.steps.items()}
