from __future__ import annotations

import json
import runpy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from leveraged_trader.indicators import compute_rsi

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research-resolution.py"
_RUNNER = runpy.run_path(str(_SCRIPT))
grid_definitions = _RUNNER["grid_definitions"]
prepare = _RUNNER["prepare"]
load = _RUNNER["load"]
freeze = _RUNNER["freeze"]
run_phase = _RUNNER["run_phase"]
analyze = _RUNNER["analyze"]
digest = _RUNNER["digest"]
write_json = _RUNNER["write_json"]


class GridResolutionDefinitionTests(unittest.TestCase):
    def test_fine_grid_matches_requested_inclusive_axes_and_4059_pairs_per_direction(self) -> None:
        grids, metadata = grid_definitions()
        fine = grids[0]
        self.assertEqual(fine.name, "r1_t1")
        self.assertEqual(fine.long_rsi, tuple(range(10, 51)))
        self.assertEqual(fine.short_rsi, tuple(range(50, 91)))
        self.assertEqual(fine.targets, tuple(value / 100 for value in range(102, 201)))
        self.assertEqual(metadata[0]["candidate_count"], 4059)
        for direction in ("long", "inverse"):
            self.assertEqual(fine.pairs(direction).shape, (4059, 2))

    def test_all_research_families_preserve_endpoints_without_duplicate_pairs(self) -> None:
        grids, metadata = grid_definitions()
        self.assertEqual(len({grid.name for grid in grids}), len(grids))
        self.assertEqual({row["family"] for row in metadata}, {
            "joint", "rsi_only", "target_only", "phase_sensitivity", "incumbent",
        })
        for grid, row in zip(grids, metadata, strict=True):
            with self.subTest(grid=grid.name):
                for direction in ("long", "inverse"):
                    pairs = grid.pairs(direction)
                    self.assertEqual(len(pairs), len(np.unique(pairs, axis=0)))
                    self.assertEqual(len(pairs), row["candidate_count"])
                if row["family"] != "incumbent":
                    self.assertEqual((min(grid.long_rsi), max(grid.long_rsi)), (10, 50))
                    self.assertEqual((min(grid.short_rsi), max(grid.short_rsi)), (50, 90))
                    self.assertEqual((min(grid.targets), max(grid.targets)), (1.02, 2.0))

    def test_joint_ladder_reaches_four_pairs_and_axis_sweeps_keep_other_axis_fine(self) -> None:
        grids, metadata = grid_definitions()
        fine = grids[0]
        joint_counts = []
        for grid, row in zip(grids, metadata, strict=True):
            if row["family"] == "joint":
                joint_counts.append(row["candidate_count"])
            elif row["family"] == "rsi_only":
                self.assertEqual(grid.targets, fine.targets)
            elif row["family"] == "target_only":
                self.assertEqual(grid.long_rsi, fine.long_rsi)
                self.assertEqual(grid.short_rsi, fine.short_rsi)
        self.assertEqual(joint_counts, sorted(joint_counts, reverse=True))
        self.assertEqual(joint_counts[-1], 4)
        terminal = next(grid for grid in grids if grid.name == "r40_t98")
        np.testing.assert_array_equal(terminal.pairs("long"), [[10, 1.02], [10, 2], [50, 1.02], [50, 2]])

    def test_shifted_grid_checks_interior_phase_while_retaining_both_endpoints(self) -> None:
        grids, _ = grid_definitions()
        by_name = {grid.name: grid for grid in grids}
        shifted = by_name["r4_t4_shift"]
        self.assertEqual(shifted.long_rsi, (10, *range(12, 51, 4), 50))
        self.assertEqual(shifted.short_rsi, (50, *range(52, 91, 4), 90))
        self.assertEqual(shifted.targets, tuple(value / 100 for value in (102, *range(104, 201, 4))))
        self.assertNotEqual(shifted.long_rsi, by_name["r4_t4"].long_rsi)
        self.assertNotEqual(shifted.targets, by_name["r4_t4"].targets)


class GridResolutionRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db = self.root / "cached # study?.sqlite"
        self.output = self.root / "study"
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE workflow_assets (symbol TEXT, rsi_symbol TEXT, workflow TEXT, source TEXT)")
            conn.execute(
                "CREATE TABLE market_data (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, "
                "close REAL, volume REAL, PRIMARY KEY(symbol, date))"
            )
        self.signal_dates = pd.bdate_range("2023-07-03", "2026-10-16")
        self.asset_dates = self.signal_dates[35:]
        self.signal_close = 100 + np.sin(np.arange(len(self.signal_dates)) / 3) * 10
        self.add_history("PROXY", self.signal_dates, self.signal_close)
        self.add_history("^IRX", self.signal_dates, np.full(len(self.signal_dates), 4.0))
        self.add_asset("AAA", "PROXY")

    def add_history(self, symbol: str, dates: pd.DatetimeIndex, closes: np.ndarray) -> None:
        with sqlite3.connect(self.db) as conn:
            conn.executemany(
                "INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (symbol, day.date().isoformat(), float(close), float(close + 1),
                     float(close - 1), float(close), 1000.0)
                    for day, close in zip(dates, closes, strict=True)
                ],
            )

    def add_asset(self, symbol: str, signal: str, workflow: str = "Long") -> None:
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO workflow_assets VALUES (?, ?, ?, ?)", (symbol, signal, workflow, "fixture")
            )
        self.add_history(symbol, self.asset_dates, np.full(len(self.asset_dates), 100.0))

    def prepared(self) -> tuple[dict, list, list, list]:
        prepare(self.db, self.output)
        return load(self.output)

    def validation_rows(self, utilities: dict[str, float] | None = None) -> None:
        utilities = utilities or {"r1_t1": 0.1, "r4_t4": 0.2, "current49": 0.1}
        pd.DataFrame([
            {"grid": grid, "signal": "PROXY", "utility": value} for grid, value in utilities.items()
        ]).to_csv(self.output / "validation_rows.csv", index=False)

    def freeze_with_result(self, result: dict | None = None) -> dict:
        with patch.dict(freeze.__globals__, {"analyze": lambda *_args: result or {"recommendation": {"grid": None}}}):
            freeze(self.output)
        return json.loads((self.output / "frozen_choice.json").read_text())

    def test_prepare_uses_read_only_snapshot_and_roundtrips_arrays_without_source_writes(self) -> None:
        original = self.db.read_bytes()
        with patch("sqlite3.connect", wraps=sqlite3.connect) as connect:
            protocol, grids, prepared, metadata = self.prepared()
        connect.assert_called_once_with(self.db.resolve().as_uri() + "?mode=ro", uri=True)
        self.assertEqual(self.db.read_bytes(), original)
        self.assertEqual([item[0] for item in prepared], ["AAA"])
        self.assertEqual(grids, grid_definitions()[0])
        self.assertEqual(metadata, protocol["metadata"])
        self.assertEqual(protocol["validation_years"], list(range(2013, 2023)))
        self.assertEqual(protocol["confirmation_years"], list(range(2023, 2027)))
        self.assertTrue(all(array.flags.c_contiguous for array in prepared[0][4]))
        with self.assertRaises(FileExistsError):
            prepare(self.db, self.output)

    def test_prepare_rsi_uses_earlier_proxy_observations_and_ignores_post_cutoff_data(self) -> None:
        _, _, prepared, _ = self.prepared()
        _, _, _, dates, arrays = prepared[0]
        self.assertEqual(pd.Timestamp(dates[0]), self.asset_dates[0])
        self.assertEqual(pd.Timestamp(dates[-1]), pd.Timestamp("2026-10-08"))
        expected_rsi = compute_rsi(pd.Series(self.signal_close, index=self.signal_dates), 14)
        np.testing.assert_allclose(arrays[3], expected_rsi.reindex(dates).to_numpy())
        np.testing.assert_allclose(arrays[4], 1.04 ** (1 / 252) - 1)
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE market_data SET open=1e9, high=2e9, close=1e9 WHERE date>'2026-10-08'")
        later = self.root / "later"
        prepare(self.db, later)
        repeated = load(later)[2][0]
        np.testing.assert_array_equal(repeated[3], dates)
        for before, after in zip(arrays, repeated[4], strict=True):
            np.testing.assert_array_equal(before, after)

    def test_prepare_preserves_pair_scoped_signal_mapping_and_inverse_direction(self) -> None:
        self.add_asset("BBB", "PROXY", "Short")
        private_symbol = "@strategy-signal-v1/5b22424242222c2250524f5859225d"
        self.add_history(private_symbol, self.signal_dates, 1000 - np.arange(len(self.signal_dates), dtype=float))
        _, _, prepared, _ = self.prepared()
        self.assertEqual([row[:3] for row in prepared], [("AAA", "PROXY", "long"), ("BBB", "PROXY", "inverse")])
        self.assertTrue(np.any(prepared[0][4][3] > 0))
        np.testing.assert_array_equal(prepared[1][4][3], 0.0)
        metadata = json.loads((self.output / "inputs.json").read_text())
        self.assertEqual(metadata[1]["signal_storage"], private_symbol)

    def test_rsi_warmup_is_trimmed_before_minimum_aligned_session_check(self) -> None:
        valid_dates = self.asset_dates[self.asset_dates <= "2026-10-08"][-760:]
        with sqlite3.connect(self.db) as conn:
            conn.execute("DELETE FROM market_data WHERE symbol='AAA'")
            conn.execute("UPDATE workflow_assets SET rsi_symbol='LATE'")
        self.add_history("AAA", valid_dates, np.full(len(valid_dates), 100.0))
        self.add_history("LATE", valid_dates, np.arange(len(valid_dates), dtype=float) + 100)
        _, _, prepared, _ = self.prepared()
        self.assertEqual(prepared, [])
        coverage = pd.read_csv(self.output / "coverage.csv")
        self.assertEqual(coverage.iloc[0]["reason"], "fewer_than_756_aligned_sessions")

    def test_load_rejects_changed_protocol_market_archive_or_input_mapping(self) -> None:
        self.prepared()
        for name, message in (
            ("protocol.json", "Registered protocol changed"),
            ("inputs.npz", "Frozen market inputs changed"),
            ("inputs.json", "Frozen input metadata changed"),
        ):
            with self.subTest(artifact=name):
                path = self.output / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, message):
                    load(self.output)
                path.write_bytes(original)
        metadata = json.loads((self.output / "inputs.json").read_text())
        metadata[0]["direction"] = "inverse"
        write_json(self.output / "inputs.json", metadata)
        with self.assertRaisesRegex(ValueError, "Frozen input metadata changed"):
            load(self.output)

    def test_load_rejects_revised_registered_source(self) -> None:
        self.prepared()
        source = self.root / "registered_source.py"
        source.write_text("original source\n")
        path = self.output / "provenance.json"
        provenance = json.loads(path.read_text())
        provenance["source_sha256"][str(source)] = digest(source)
        write_json(path, provenance)
        load(self.output)
        source.write_text("revised source\n")
        with self.assertRaisesRegex(ValueError, "Registered source changed"):
            load(self.output)

    def test_prepare_registers_transitive_simulation_sources_and_load_rejects_changes(self) -> None:
        self.prepared()
        provenance = json.loads((self.output / "provenance.json").read_text())
        dependencies = ("leveraged_trader/pricing.py", "leveraged_trader/backtest.py")
        for name in dependencies:
            self.assertIn(name, provenance["source_sha256"])
        source_root = self.root / "source"
        for name in provenance["source_sha256"]:
            destination = source_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((_SCRIPT.parent.parent / name).read_bytes())
        with patch.dict(load.__globals__, {"ROOT": source_root}):
            load(self.output)
            for name in dependencies:
                with self.subTest(source=name):
                    source = source_root / name
                    original = source.read_bytes()
                    source.write_bytes(original + b"\n# Revised simulation dependency.\n")
                    with self.assertRaisesRegex(ValueError, f"Registered source changed: {name}"):
                        load(self.output)
                    source.write_bytes(original)

    def test_freeze_prefers_qualified_candidate_and_confirms_neighbors_and_phase_checks(self) -> None:
        self.prepared()
        self.validation_rows({"r1_t1": 0.1, "r4_t4": 0.5, "r8_t8": 0.2})
        frozen = self.freeze_with_result({"recommendation": {"grid": "r8_t8"}})
        self.assertEqual(frozen["evidence_qualified_candidate"], "r8_t8")
        self.assertEqual(frozen["exploratory_leader"], "r4_t4")
        self.assertTrue({
            "r1_t1", "current49", "r5_t5", "r8_t8", "r10_t10", "r40_t98",
            "r5_t5_shift", "r8_t8_shift", "r10_t10_shift",
        } <= set(frozen["confirmation_grids"]))
        self.assertEqual(frozen["validation_rows_sha256"], digest(self.output / "validation_rows.csv"))
        self.assertEqual(frozen["protocol_sha256"], digest(self.output / "protocol.json"))
        self.assertEqual((self.output / "frozen_choice.sha256").read_text().strip(),
                         digest(self.output / "frozen_choice.json"))
        with self.assertRaisesRegex(FileExistsError, "already frozen"):
            freeze(self.output)

    def test_equal_validation_utility_chooses_smallest_exploratory_grid(self) -> None:
        self.prepared()
        self.validation_rows({"r1_t1": 0.2, "r4_t4": 0.2, "r8_t8": 0.2})
        frozen = self.freeze_with_result()
        self.assertIsNone(frozen["evidence_qualified_candidate"])
        self.assertEqual(frozen["exploratory_leader"], "r8_t8")

    def test_freeze_is_rejected_after_confirmation(self) -> None:
        self.output.mkdir()
        for artifact in ("confirmation_rows.csv", "confirmation_run.json"):
            with self.subTest(artifact=artifact):
                path = self.output / artifact
                path.touch()
                with self.assertRaisesRegex(FileExistsError, "after confirmation"):
                    freeze(self.output)
                path.unlink()

    def test_confirmation_uses_frozen_grids_and_future_windows_only(self) -> None:
        self.prepared()
        self.validation_rows()
        frozen = self.freeze_with_result()
        with patch(
            "leveraged_trader.grid_resolution.evaluate_windows", return_value=[{"result": "fixture"}],
        ) as evaluate:
            run_phase(self.output, "confirmation")
        evaluate.assert_called_once()
        self.assertEqual({grid.name for grid in evaluate.call_args.kwargs["grids"]}, set(frozen["confirmation_grids"]))
        self.assertEqual(evaluate.call_args.kwargs["windows"], [
            (2023, "2023-01-01", "2024-01-01"), (2024, "2024-01-01", "2025-01-01"),
            (2025, "2025-01-01", "2026-01-01"), (2026, "2026-01-01", "2026-10-09"),
        ])
        with self.assertRaises(FileExistsError):
            run_phase(self.output, "confirmation")

    def test_confirmation_rejects_changed_frozen_choice_or_validation_evidence(self) -> None:
        self.prepared()
        self.validation_rows()
        self.freeze_with_result()
        for name, message in (
            ("frozen_choice.json", "Frozen confirmation choice changed"),
            ("validation_rows.csv", "Validation evidence changed"),
        ):
            with self.subTest(artifact=name):
                path = self.output / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with (
                    patch("leveraged_trader.grid_resolution.evaluate_windows") as evaluate,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    run_phase(self.output, "confirmation")
                evaluate.assert_not_called()
                path.write_bytes(original)

    def test_confirmation_analysis_uses_frozen_subset_with_original_multiple_comparison_count(self) -> None:
        protocol, _, _, _ = self.prepared()
        pd.DataFrame([
            {"grid": "r1_t1", "signal": "PROXY", "utility": 0.1},
            {"grid": "r4_t4", "signal": "PROXY", "utility": 0.2},
        ]).to_csv(self.output / "confirmation_rows.csv", index=False)
        with patch(
            "leveraged_trader.grid_resolution_analysis.summarize_resolution", return_value={"fixture": True},
        ) as summarize:
            result = analyze(self.output, "confirmation")
        self.assertEqual(result, {"fixture": True})
        self.assertEqual({row["name"] for row in summarize.call_args.args[1]}, {"r1_t1", "r4_t4"})
        self.assertEqual(summarize.call_args.kwargs["comparison_count"], len(protocol["grids"]) - 1)
        self.assertEqual(summarize.call_args.kwargs["bootstrap_samples"], 250000)


if __name__ == "__main__":
    unittest.main()
