from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime


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
