from __future__ import annotations

import unittest

from leveraged_trader.benchmark import WorkflowPhaseTimings


class WorkflowPhaseTimingsTests(unittest.TestCase):
    def test_overlapping_phase_intervals_count_elapsed_time_once(self) -> None:
        timings = WorkflowPhaseTimings()

        timings.begin("download", 10.0)
        timings.begin("download", 12.0)
        timings.end("download", 15.0)
        timings.end("download", 18.0)

        self.assertEqual(timings.snapshot().download_seconds, 8.0)

    def test_separate_phase_intervals_exclude_idle_time(self) -> None:
        timings = WorkflowPhaseTimings()

        timings.begin("download", 10.0)
        timings.end("download", 12.0)
        timings.begin("download", 15.0)
        timings.end("download", 18.0)

        self.assertEqual(timings.snapshot().download_seconds, 5.0)

    def test_phase_cannot_end_without_starting(self) -> None:
        timings = WorkflowPhaseTimings()

        with self.assertRaisesRegex(RuntimeError, "ended without starting"):
            timings.end("download", 10.0)


if __name__ == "__main__":
    unittest.main()
