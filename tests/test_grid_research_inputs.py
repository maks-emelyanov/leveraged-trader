from __future__ import annotations

import copy
import runpy
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from leveraged_trader.indicators import compute_rsi

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research-grid.py"
_SCRIPT_FUNCTIONS = runpy.run_path(str(_SCRIPT))
load_inputs = _SCRIPT_FUNCTIONS["load_inputs"]
verify_confirmation_inputs = _SCRIPT_FUNCTIONS["verify_confirmation_inputs"]
freeze_choice = _SCRIPT_FUNCTIONS["freeze_choice"]
finalize_decision = _SCRIPT_FUNCTIONS["finalize_decision"]


class GridResearchInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        # Reserved URI characters must remain part of the filename.
        self.db = Path(self.directory.name) / "cached # study?.sqlite"
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "CREATE TABLE workflow_assets (symbol TEXT, rsi_symbol TEXT, workflow TEXT, source TEXT)"
            )
            conn.execute(
                "CREATE TABLE market_data (symbol TEXT, date TEXT, open REAL, high REAL, "
                "low REAL, close REAL, volume REAL, PRIMARY KEY(symbol, date))"
            )
        self.dates = pd.bdate_range("2021-10-01", "2022-01-14")
        self.asset_dates = self.dates[self.dates >= "2021-12-01"]
        self.signal_close = 100 + np.sin(np.arange(len(self.dates)) / 3) * 10
        self.add_history("^IRX", self.dates, np.full(len(self.dates), 4.0))

    def add_history(self, symbol: str, dates: pd.DatetimeIndex, closes: np.ndarray) -> None:
        with sqlite3.connect(self.db) as conn:
            conn.executemany(
                "INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (symbol, date.date().isoformat(), float(close), float(close + 1),
                     float(close - 1), float(close), 1000.0)
                    for date, close in zip(dates, closes, strict=True)
                ],
            )

    def add_asset(self, symbol: str, signal: str, workflow: str = "Long") -> None:
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO workflow_assets VALUES (?, ?, ?, ?)", (symbol, signal, workflow, "fixture")
            )
        self.add_history(symbol, self.asset_dates, np.full(len(self.asset_dates), 100.0))

    def test_source_is_opened_read_only_without_modifying_database(self) -> None:
        self.add_asset("AAA", "PROXY")
        self.add_history("PROXY", self.dates, self.signal_close)
        original_bytes = self.db.read_bytes()
        original_files = set(self.db.parent.iterdir())
        with patch("sqlite3.connect", wraps=sqlite3.connect) as connect:
            prepared, _coverage = load_inputs(self.db, "2022-01-05", 20)
        connect.assert_called_once_with(self.db.resolve().as_uri() + "?mode=ro", uri=True)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(self.db.read_bytes(), original_bytes)
        self.assertEqual(set(self.db.parent.iterdir()), original_files)

    def test_rsi_uses_prior_signal_history_and_ignores_every_post_cutoff_price(self) -> None:
        self.add_asset("AAA", "PROXY")
        self.add_history("PROXY", self.dates, self.signal_close)
        cutoff = "2022-01-05"
        prepared, coverage = load_inputs(self.db, cutoff, 20)
        self.assertEqual(len(prepared), 1)
        _asset, _signal, _direction, dates, arrays = prepared[0]
        self.assertEqual(pd.Timestamp(dates[0]), self.asset_dates[0])
        self.assertEqual(pd.Timestamp(dates[-1]), pd.Timestamp(cutoff))
        full_rsi = compute_rsi(pd.Series(self.signal_close, index=self.dates), 14)
        np.testing.assert_allclose(arrays[3], full_rsi.reindex(dates).to_numpy())
        np.testing.assert_allclose(arrays[4], (1.04 ** (1 / 252)) - 1)
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "UPDATE market_data SET open=1e10, high=2e10, low=1e9, close=1e10 WHERE date > ?", (cutoff,)
            )
        repeated, repeated_coverage = load_inputs(self.db, cutoff, 20)
        self.assertEqual(coverage, repeated_coverage)
        np.testing.assert_array_equal(dates, repeated[0][3])
        for before, after in zip(arrays, repeated[0][4], strict=True):
            np.testing.assert_array_equal(before, after)

    def test_private_signal_is_pair_scoped_and_cohort_preserves_only_requested_assets(self) -> None:
        self.add_asset("AAA", "PROXY", "Short")
        self.add_asset("BBB", "PROXY")
        private_symbol = "@strategy-signal-v1/5b22414141222c2250524f5859225d"
        self.add_history("PROXY", self.dates, np.arange(len(self.dates), dtype=float) + 100)
        self.add_history(private_symbol, self.dates, 200 - np.arange(len(self.dates), dtype=float))
        prepared, coverage = load_inputs(self.db, "2022-01-05", 20)
        self.assertEqual([record[:3] for record in prepared], [
            ("AAA", "PROXY", "inverse"), ("BBB", "PROXY", "long"),
        ])
        np.testing.assert_array_equal(prepared[0][4][3], 0.0)
        np.testing.assert_array_equal(prepared[1][4][3], 100.0)
        self.assertEqual(coverage[0]["signal_storage"], private_symbol)
        self.assertEqual(coverage[1]["signal_storage"], "PROXY")
        selected, selected_coverage = load_inputs(self.db, "2022-01-05", 20, {"BBB"})
        self.assertEqual([record[0] for record in selected], ["BBB"])
        self.assertEqual([record["asset"] for record in selected_coverage], ["BBB"])

    def test_warmup_prefix_is_trimmed_before_enforcing_training_requirement(self) -> None:
        self.add_asset("AAA", "PROXY")
        self.add_history("PROXY", self.asset_dates, self.signal_close[-len(self.asset_dates):])
        prepared, coverage = load_inputs(self.db, "2022-01-05", 5)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(pd.Timestamp(prepared[0][3][0]), self.asset_dates[14])
        self.assertEqual(coverage[0]["first_date"], self.asset_dates[14].date().isoformat())
        self.assertTrue(np.isfinite(prepared[0][4][3]).all())
        excluded, excluded_coverage = load_inputs(self.db, "2022-01-05", 20)
        self.assertEqual(excluded, [])
        self.assertEqual(excluded_coverage[0]["reason"], "insufficient_pre_2022_aligned_history")

    def test_new_assets_cannot_use_post_2021_history_to_qualify(self) -> None:
        self.asset_dates = self.dates[self.dates >= "2022-01-03"]
        self.add_asset("NEW", "PROXY")
        self.add_history("PROXY", self.dates, self.signal_close)
        prepared, coverage = load_inputs(self.db, "2022-01-14", 5)
        self.assertEqual(prepared, [])
        self.assertEqual(coverage[0]["reason"], "insufficient_pre_2022_price_history")

    def confirmation_fixture(self) -> tuple[list, list, Path]:
        self.add_asset("AAA", "PROXY")
        self.add_history("PROXY", self.dates, self.signal_close)
        _validation, validation_coverage = load_inputs(self.db, "2022-01-05", 20)
        output = self.db.parent
        pd.DataFrame(validation_coverage).to_csv(output / "validation_coverage.csv", index=False)
        prepared, coverage = load_inputs(self.db, "2022-01-14", 20)
        return prepared, coverage, output

    def test_confirmation_preserves_frozen_prefix_while_allowing_new_observations(self) -> None:
        prepared, coverage, output = self.confirmation_fixture()
        verify_confirmation_inputs(prepared, coverage, output, {"AAA"}, "2022-01-05")
        prepared = copy.deepcopy(prepared)
        prepared[0][4][0][-1] += 5
        verify_confirmation_inputs(prepared, coverage, output, {"AAA"}, "2022-01-05")

    def test_confirmation_rejects_revised_prices_rsi_dates_or_signal_mapping(self) -> None:
        prepared, coverage, output = self.confirmation_fixture()
        for change in ("price", "rsi", "date", "signal", "direction", "signal_storage"):
            with self.subTest(change=change):
                changed_prepared, changed_coverage = copy.deepcopy((prepared, coverage))
                record = list(changed_prepared[0])
                if change == "price":
                    record[4][0][0] += 1
                elif change == "rsi":
                    record[4][3][0] += 1
                elif change == "date":
                    record[3][0] -= np.timedelta64(1, "D")
                elif change == "signal":
                    record[1] = "OTHER"
                elif change == "direction":
                    record[2] = "inverse"
                else:
                    changed_coverage[0]["signal_storage"] = "PRIVATE"
                changed_prepared[0] = tuple(record)
                with self.assertRaisesRegex(ValueError, "inputs or signal mapping changed"):
                    verify_confirmation_inputs(
                        changed_prepared, changed_coverage, output, {"AAA"}, "2022-01-05",
                    )

    def test_confirmation_rejects_missing_or_added_cohort_members(self) -> None:
        prepared, coverage, output = self.confirmation_fixture()
        for cohort in ({"AAA", "MISSING"}, set()):
            with (
                self.subTest(cohort=cohort),
                self.assertRaisesRegex(ValueError, "entire frozen validation cohort"),
            ):
                verify_confirmation_inputs(prepared, coverage, output, cohort, "2022-01-05")


class GridResearchFreezeTests(unittest.TestCase):
    @staticmethod
    def summary() -> dict:
        return {
            "grids": [{"grid": "current49", "utility": 0.1}, {"grid": "fast_targets56", "utility": 0.2}],
            "paired": [
                {
                    "grid": "fast_targets56", "direction": "all", "familywise_lower_95_utility": 0.01,
                    "discordant_groups_utility": 12, "improvement_share_utility": 0.75,
                    "delta_stress_utility": 0.01, "matched_exposed_signal_groups": 10,
                    "matched_exposed_delta_mean_holding_sessions": -1.0,
                    "matched_exposed_delta_max_holding_sessions": -2.0,
                },
                {"grid": "fast_targets56", "direction": "long", "delta_utility": 0.1},
                {"grid": "fast_targets56", "direction": "inverse", "delta_utility": 0.01},
            ],
        }

    def test_freeze_identifies_leader_and_requires_every_adoption_gate(self) -> None:
        summary = self.summary()
        choice = freeze_choice(summary, "fixture-protocol-digest")
        self.assertEqual(choice["leader"], "fast_targets56")
        self.assertEqual(choice["protocol_sha256"], "fixture-protocol-digest")
        self.assertTrue(choice["validation_adoption_eligible"])
        self.assertTrue(all(choice["validation_gates"].values()))
        failures = (
            ("familywise_lower_95_utility", 0.0, "adjusted_paired_lower_bound_positive"),
            ("discordant_groups_utility", 9, "broad_improvement"),
            ("improvement_share_utility", 0.59, "broad_improvement"),
            ("delta_stress_utility", -0.01, "stress_improvement_nonnegative"),
            ("matched_exposed_signal_groups", 0, "matched_holding_no_worse"),
            ("matched_exposed_delta_mean_holding_sessions", 0.1, "matched_holding_no_worse"),
            ("matched_exposed_delta_max_holding_sessions", 0.1, "matched_holding_no_worse"),
        )
        for field, value, gate in failures:
            with self.subTest(field=field):
                changed = copy.deepcopy(summary)
                changed["paired"][0][field] = value
                rejected = freeze_choice(changed, "fixture-protocol-digest")
                self.assertEqual(rejected["leader"], "fast_targets56")
                self.assertFalse(rejected["validation_gates"][gate])
                self.assertFalse(rejected["validation_adoption_eligible"])
        summary["paired"][2]["delta_utility"] = -0.01
        rejected = freeze_choice(summary, "fixture-protocol-digest")
        self.assertFalse(rejected["validation_gates"]["no_direction_deterioration"])
        self.assertFalse(rejected["validation_adoption_eligible"])

    def test_exact_utility_tie_prefers_smaller_grid_and_incumbent_needs_no_adoption(self) -> None:
        summary = self.summary()
        summary["grids"][1]["utility"] = 0.1
        choice = freeze_choice(summary, "fixture-protocol-digest")
        self.assertEqual(choice["ranking"], ["current49", "fast_targets56"])
        self.assertEqual(choice["validation_gates"], {})
        self.assertFalse(choice["validation_adoption_eligible"])
        summary["grids"][1]["grid"] = "coarse_rsi28"
        for row in summary["paired"]:
            row["grid"] = "coarse_rsi28"
        choice = freeze_choice(summary, "fixture-protocol-digest")
        self.assertEqual(choice["ranking"], ["coarse_rsi28", "current49"])

    def test_freeze_rejects_empty_validation_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "No eligible validation outcomes"):
            freeze_choice({"grids": [], "paired": []}, "fixture-protocol-digest")


class GridResearchDecisionTests(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[dict, dict]:
        frozen = freeze_choice(GridResearchFreezeTests.summary(), "fixture-protocol-digest")
        confirmation = {
            "protocol_sha256": "fixture-protocol-digest",
            "grids": [
                {"grid": "current49", "active_asset_years": 4},
                {"grid": "fast_targets56", "active_asset_years": 5},
            ],
            "paired": [{
                "grid": "fast_targets56", "direction": "all", "delta_utility": 0.0, "delta_stress_utility": 0.0,
            }],
        }
        return frozen, confirmation

    def test_adoption_requires_passed_validation_and_nonnegative_confirmation(self) -> None:
        frozen, confirmation = self.fixture()
        original_frozen = copy.deepcopy(frozen)
        decision = finalize_decision(frozen, confirmation)
        self.assertTrue(decision["adopt_replacement"])
        self.assertTrue(decision["confirmation_noninferiority_gate"])
        self.assertEqual(decision["production_grid"], "fast_targets56")
        self.assertEqual(decision["confirmation_activated_asset_periods"], {"current49": 4, "fast_targets56": 5})
        self.assertEqual(frozen, original_frozen)

    def test_strong_confirmation_cannot_rescue_failed_validation(self) -> None:
        frozen, confirmation = self.fixture()
        frozen["validation_adoption_eligible"] = False
        frozen["validation_gates"]["broad_improvement"] = False
        confirmation["paired"][0].update(delta_utility=10.0, delta_stress_utility=10.0)
        decision = finalize_decision(frozen, confirmation)
        self.assertTrue(decision["confirmation_noninferiority_gate"])
        self.assertFalse(decision["adopt_replacement"])
        self.assertEqual(decision["production_grid"], "current49")

    def test_negative_base_or_stress_confirmation_retains_incumbent(self) -> None:
        for field in ("delta_utility", "delta_stress_utility"):
            with self.subTest(field=field):
                frozen, confirmation = self.fixture()
                confirmation["paired"][0][field] = -0.001
                decision = finalize_decision(frozen, confirmation)
                self.assertFalse(decision["confirmation_noninferiority_gate"])
                self.assertFalse(decision["adopt_replacement"])
                self.assertEqual(decision["production_grid"], "current49")

    def test_confirmation_rejects_wrong_protocol_or_changed_grid_set(self) -> None:
        frozen, confirmation = self.fixture()
        confirmation["protocol_sha256"] = "different-protocol"
        with self.assertRaisesRegex(ValueError, "different protocol"):
            finalize_decision(frozen, confirmation)
        for grids in (["current49"], ["current49", "coarse_rsi28"], ["current49", "fast_targets56", "coarse_rsi28"]):
            with self.subTest(grids=grids):
                frozen, confirmation = self.fixture()
                confirmation["grids"] = [{"grid": name, "active_asset_years": 1} for name in grids]
                with self.assertRaisesRegex(ValueError, "only the frozen leader and incumbent"):
                    finalize_decision(frozen, confirmation)


if __name__ == "__main__":
    unittest.main()
