from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on some platforms.
    resource = None  # type: ignore[assignment]


@dataclass(frozen=True)
class WorkflowBenchmark:
    status: str
    started_at_utc: datetime
    finished_at_utc: datetime
    wall_seconds: float
    cpu_seconds: float
    cpu_utilization_percent: float
    peak_rss_mb: float | None
    current_rss_mb: float | None
    asset_count: int
    completed_asset_count: int
    skipped_asset_count: int
    rows_processed: int
    workflow_concurrency: int

    def as_csv_row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "started_at_utc": self.started_at_utc.isoformat(),
            "finished_at_utc": self.finished_at_utc.isoformat(),
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "cpu_utilization_percent": self.cpu_utilization_percent,
            "peak_rss_mb": self.peak_rss_mb,
            "current_rss_mb": self.current_rss_mb,
            "asset_count": self.asset_count,
            "completed_asset_count": self.completed_asset_count,
            "skipped_asset_count": self.skipped_asset_count,
            "rows_processed": self.rows_processed,
            "workflow_concurrency": self.workflow_concurrency,
        }


@dataclass(frozen=True)
class BenchmarkTracker:
    started_at_utc: datetime
    perf_counter_started: float
    process_time_started: float

    @classmethod
    def start(cls) -> BenchmarkTracker:
        return cls(
            started_at_utc=datetime.now(UTC),
            perf_counter_started=time.perf_counter(),
            process_time_started=time.process_time(),
        )

    def finish(
        self,
        *,
        asset_run_results: Iterable[Any],
        workflow_concurrency: int,
        status: str = "completed",
    ) -> WorkflowBenchmark:
        finished_at_utc = datetime.now(UTC)
        wall_seconds = max(0.0, time.perf_counter() - self.perf_counter_started)
        cpu_seconds = max(0.0, time.process_time() - self.process_time_started)
        cpu_utilization_percent = (cpu_seconds / wall_seconds * 100.0) if wall_seconds else 0.0
        results = list(asset_run_results)
        return WorkflowBenchmark(
            status=status,
            started_at_utc=self.started_at_utc,
            finished_at_utc=finished_at_utc,
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            cpu_utilization_percent=cpu_utilization_percent,
            peak_rss_mb=_peak_rss_mb(),
            current_rss_mb=_current_rss_mb(),
            asset_count=len(results),
            completed_asset_count=sum(1 for result in results if getattr(result, "status", None) == "done"),
            skipped_asset_count=sum(1 for result in results if getattr(result, "status", None) != "done"),
            rows_processed=sum(_rows_processed(result) for result in results),
            workflow_concurrency=workflow_concurrency,
        )


def _rows_processed(result: Any) -> int:
    value = getattr(result, "rows_processed", None)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _peak_rss_mb() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = float(usage.ru_maxrss)
    if sys.platform == "darwin":
        return value / 1024 / 1024
    return value / 1024


def _current_rss_mb() -> float | None:
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return None
    try:
        with open(status_path, encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024
    except OSError:
        return None
    return None
