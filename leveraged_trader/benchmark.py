from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Literal

WorkflowPhase = Literal[
    "download",
    "db_sync",
    "grid_compute",
    "report_generation",
    "alpaca",
]


@dataclass(frozen=True)
class WorkflowPhaseSnapshot:
    download_seconds: float
    db_sync_seconds: float
    grid_compute_seconds: float
    report_generation_seconds: float
    alpaca_seconds: float


@dataclass
class WorkflowPhaseTimings:
    download_seconds: float = 0.0
    db_sync_seconds: float = 0.0
    grid_compute_seconds: float = 0.0
    report_generation_seconds: float = 0.0
    alpaca_seconds: float = 0.0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _active_counts: dict[WorkflowPhase, int] = field(default_factory=dict, init=False, repr=False)
    _active_started_at: dict[WorkflowPhase, float] = field(default_factory=dict, init=False, repr=False)

    def add(self, phase: WorkflowPhase, elapsed_seconds: float) -> None:
        field_name = f"{phase}_seconds"
        with self._lock:
            setattr(
                self,
                field_name,
                getattr(self, field_name) + max(0.0, elapsed_seconds),
            )

    def begin(self, phase: WorkflowPhase, started_at: float) -> None:
        with self._lock:
            active_count = self._active_counts.get(phase, 0)
            if active_count == 0:
                self._active_started_at[phase] = started_at
            self._active_counts[phase] = active_count + 1

    def end(self, phase: WorkflowPhase, finished_at: float) -> None:
        field_name = f"{phase}_seconds"
        with self._lock:
            active_count = self._active_counts.get(phase, 0)
            if active_count <= 0:
                raise RuntimeError(f"Workflow phase {phase!r} ended without starting.")
            if active_count > 1:
                self._active_counts[phase] = active_count - 1
                return

            started_at = self._active_started_at.pop(phase)
            self._active_counts.pop(phase)
            setattr(
                self,
                field_name,
                getattr(self, field_name) + max(0.0, finished_at - started_at),
            )

    def snapshot(self) -> WorkflowPhaseSnapshot:
        with self._lock:
            return WorkflowPhaseSnapshot(
                download_seconds=self.download_seconds,
                db_sync_seconds=self.db_sync_seconds,
                grid_compute_seconds=self.grid_compute_seconds,
                report_generation_seconds=self.report_generation_seconds,
                alpaca_seconds=self.alpaca_seconds,
            )


@dataclass(frozen=True)
class WorkflowTimer:
    started_at_utc: datetime
    perf_counter_started: float

    @classmethod
    def start(cls) -> WorkflowTimer:
        return cls(
            started_at_utc=datetime.now(UTC),
            perf_counter_started=time.perf_counter(),
        )

    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.perf_counter_started)
