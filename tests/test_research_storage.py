from __future__ import annotations

import json
import sqlite3
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from leveraged_trader.config import BacktestConfig
from leveraged_trader.research_selection import SELECTION_VERSION, development_trade_activity, select_strategy
from leveraged_trader.storage import (
    _strategy_selection_digest,
    clear_all_strategy_state,
    clear_asset_state,
    init_state_db,
    load_best_strategy_summary,
    load_complete_strategy_equity_curve,
    process_asset_grid,
    strategy_config_fingerprint,
    strategy_state_matches_config,
)


def research_data(periods: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    data: dict[str, object] = {}
    for symbol, close in (
        ("TQQQ", 100 + np.arange(periods) * 0.2 + np.sin(np.arange(periods)) * 2),
        ("QQQ", 100 + np.sin(np.arange(periods) / 3) * 12),
        ("^IRX", np.full(periods, 4.0)),
    ):
        for field in ("Open", "High", "Low", "Close"):
            data[f"{symbol}_{field}"] = close + (1.0 if field == "High" else -1.0 if field == "Low" else 0)
        data[f"{symbol}_Volume"] = 1_000_000
    return pd.DataFrame(data, index=dates)


class ResearchSelectionStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        init_state_db(self.conn)
        self.cfg = BacktestConfig(rsi_period=3)
        self.data = research_data()
        self.buys = [20.0, 40.0]
        self.targets = [1.02, 1.1]
        self.addCleanup(self.conn.close)

    def process(self, data: pd.DataFrame | None = None, *, rebuild: bool = False, rule: str = "lower") -> None:
        process_asset_grid(
            self.conn, self.data if data is None else data, self.cfg,
            "TQQQ", "QQQ", self.buys, self.targets, rebuild, rsi_entry_rule=rule,
        )

    def matches(self) -> bool:
        return strategy_state_matches_config(self.conn, "TQQQ", "QQQ", self.cfg, self.buys, self.targets)

    def selection_result(self, status: str = "passed") -> dict[str, object]:
        block_sessions = np.diff(np.linspace(0, len(self.data) * 4 // 5, 4, dtype=int))
        minimum_rate, bonus = development_trade_activity(np.asarray([[3], [3], [3]]), block_sessions)
        return {
            "buy_rsi": 40.0,
            "profit_target_multiple": 1.1,
            "status": status,
            "selection_score": 1.5,
            "candidate_count": 4,
            "development_sessions": len(self.data) * 4 // 5,
            "holdout_sessions": len(self.data) - len(self.data) * 4 // 5,
            "development_completed_trades": 9,
            "development_min_annual_trades": float(minimum_rate[0]),
            "trade_activity_bonus": float(bonus[0]),
            "development_block_completed_trades": [3, 3, 3],
            "development_block_sharpes": [2.0, 2.0, 2.0],
            "development_block_returns": [0.1, 0.1, 0.1],
            "development_sharpe_median": 2.0,
            "development_sharpe_std": 0.0,
            "mean_holding_sessions": 3.0,
            "max_holding_sessions": 6,
            "holdout_sharpe": 0.5 if status == "passed" else -0.5,
            "holdout_total_return": 0.1 if status == "passed" else -0.1,
            "holdout_completed_trades": 2,
            "holdout_mean_holding_sessions": 3.0,
            "holdout_max_holding_sessions": 6,
        }

    def test_short_history_uses_selective_fallback_for_each_workflow(self) -> None:
        for rule, expected_buy in (("lower", 20.0), ("upper", 40.0)):
            with self.subTest(rule=rule):
                self.process(rebuild=True, rule=rule)
                best = load_best_strategy_summary(self.conn, "TQQQ", "QQQ", rule)
                self.assertEqual((best["buy_rsi"], best["profit_target_multiple"]), (expected_buy, 1.02))
                self.assertEqual(best["selection"]["status"], "insufficient_history")
                self.assertIsNone(best["selection"]["selection_score"])
                self.assertEqual(best["selection"]["method"], "robust")

    def test_selected_pair_owns_retained_curve_even_when_holdout_fails(self) -> None:
        self.data = research_data(700)
        for status in ("passed", "failed_holdout"):
            with self.subTest(status=status), patch(
                "leveraged_trader.research_selection.select_strategy", return_value=self.selection_result(status)
            ):
                self.process(rebuild=True)
                best = load_best_strategy_summary(self.conn, "TQQQ", "QQQ", strategy_state_preverified=True)
                self.assertEqual((best["buy_rsi"], best["profit_target_multiple"]), (40.0, 1.1))
                self.assertEqual(best["selection"]["status"], status)
                curves = self.conn.execute(
                    "SELECT DISTINCT buy_rsi, profit_target_multiple FROM strategy_equity"
                ).fetchall()
                self.assertEqual(curves, [(40.0, 1.1)])
                curve = load_complete_strategy_equity_curve(
                    self.conn, "TQQQ", "QQQ", 40.0, 1.1, rsi_period=3,
                    base_cfg=self.cfg, expected_buy_rsi_values=self.buys,
                    expected_profit_target_values=self.targets,
                )
                self.assertEqual(len(curve), len(self.data))
                self.assertTrue(self.matches())

    def test_history_immediately_below_minimum_remains_authenticated(self) -> None:
        self.data = research_data(629)
        self.process(rebuild=True)
        best = load_best_strategy_summary(self.conn, "TQQQ", "QQQ")
        self.assertEqual(best["selection"]["status"], "insufficient_history")
        self.assertTrue(self.matches())

    def test_real_selection_is_identical_after_rebuild_or_incremental_update(self) -> None:
        self.data = research_data(700)
        self.process(self.data.iloc[:350], rebuild=True)
        self.process(self.data.iloc[350:])
        incremental = load_best_strategy_summary(self.conn, "TQQQ", "QQQ")
        self.assertEqual(incremental["selection"]["status"], "passed")
        self.process(rebuild=True)
        rebuilt = load_best_strategy_summary(self.conn, "TQQQ", "QQQ")
        self.assertEqual(incremental["selection"], rebuilt["selection"])

    def test_tail_update_selects_using_complete_canonical_history(self) -> None:
        consumed: list[tuple[np.ndarray, ...]] = []

        def select(*args: object, **kwargs: object) -> dict[str, object]:
            consumed.append(tuple(np.asarray(values).copy() for values in args[:5]))
            return select_strategy(*args, **kwargs)

        with patch("leveraged_trader.research_selection.select_strategy", side_effect=select):
            self.process(self.data.iloc[:40], rebuild=True)
            self.process(self.data.iloc[40:])
        self.assertEqual([len(values[0]) for values in consumed], [40, 80])
        np.testing.assert_array_equal(consumed[-1][0], self.data["TQQQ_Open"].to_numpy())
        incremental = load_best_strategy_summary(self.conn, "TQQQ", "QQQ")
        self.process(rebuild=True)
        rebuilt = load_best_strategy_summary(self.conn, "TQQQ", "QQQ")
        self.assertEqual(incremental["selection"]["input_digest"], rebuilt["selection"]["input_digest"])

    def test_identical_update_reuses_selection_and_can_repair_its_curve(self) -> None:
        self.process(rebuild=True)
        self.conn.execute("DELETE FROM strategy_equity")
        with patch(
            "leveraged_trader.research_selection.select_strategy", side_effect=AssertionError("unnecessary replay")
        ):
            self.process(self.data.iloc[-1:])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0], len(self.data))
        self.assertTrue(self.matches())

    def test_tampered_and_missing_selection_cannot_authenticate_robust_state(self) -> None:
        self.process(rebuild=True)
        self.conn.execute("UPDATE strategy_selection SET payload_json = '{}' ")
        self.assertIsNone(load_best_strategy_summary(self.conn, "TQQQ", "QQQ"))
        self.assertFalse(self.matches())
        self.process()
        self.assertTrue(self.matches())
        self.conn.execute("DELETE FROM strategy_selection")
        self.assertFalse(self.matches())

    def test_selection_binds_nonwinning_candidates_and_configuration(self) -> None:
        self.process(rebuild=True)
        self.conn.execute(
            "UPDATE strategy_summary SET total_return = total_return + 1 "
            "WHERE buy_rsi = 40 AND profit_target_multiple = 1.1"
        )
        self.assertIsNone(load_best_strategy_summary(self.conn, "TQQQ", "QQQ", strategy_state_preverified=True))
        self.process(rebuild=True)
        self.conn.execute("UPDATE strategy_config SET fingerprint = 'wrong' ")
        self.assertIsNone(load_best_strategy_summary(self.conn, "TQQQ", "QQQ"))

    def test_rehashed_acceptance_edits_cannot_bypass_evidence_checks(self) -> None:
        self.data = research_data(700)
        with patch("leveraged_trader.research_selection.select_strategy", return_value=self.selection_result()):
            self.process(rebuild=True)
        original = self.conn.execute("SELECT * FROM strategy_selection").fetchone()
        for field, invalid in (
            ("status", "failed_holdout"),
            ("holdout_sharpe", -1.0),
            ("holdout_completed_trades", 1),
            ("development_block_completed_trades", [2, 3, 4]),
            ("selection_score", 100.0),
            ("development_sharpe_median", 3.0),
            ("development_min_annual_trades", 100.0),
            ("trade_activity_bonus", 0.5),
        ):
            with self.subTest(field=field):
                payload = json.loads(original[4])
                payload[field] = invalid
                encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
                digest = _strategy_selection_digest((*original[:4], encoded))
                self.conn.execute(
                    "UPDATE strategy_selection SET payload_json = ?, integrity_digest = ?", (encoded, digest)
                )
                self.assertIsNone(load_best_strategy_summary(self.conn, "TQQQ", "QQQ"))
                self.assertFalse(self.matches())

    def test_selection_invalidates_with_asset_or_benchmark_state(self) -> None:
        for clear in (lambda: clear_asset_state(self.conn, "TQQQ", "QQQ"), lambda: clear_all_strategy_state(self.conn)):
            self.process(rebuild=True)
            clear()
            self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM strategy_selection").fetchone()[0], 0)

    def test_selector_failure_rolls_back_history_and_selection_together(self) -> None:
        self.process(self.data.iloc[:40], rebuild=True)
        previous = self.conn.execute("SELECT * FROM strategy_selection").fetchall()
        with (
            patch("leveraged_trader.research_selection.select_strategy", side_effect=RuntimeError("selection failed")),
            self.assertRaisesRegex(RuntimeError, "selection failed"),
        ):
            self.process(self.data.iloc[40:])
        self.assertEqual(self.conn.execute("SELECT * FROM strategy_selection").fetchall(), previous)
        self.assertEqual(self.conn.execute("SELECT MAX(date) FROM market_data WHERE symbol='TQQQ'").fetchone()[0],
                         self.data.index[39].date().isoformat())
        self.assertTrue(self.matches())

    def test_selection_method_changes_fingerprint_and_legacy_has_no_selection(self) -> None:
        legacy = BacktestConfig(rsi_period=3, selection_method="legacy_sharpe")
        self.assertNotEqual(
            strategy_config_fingerprint(self.cfg, self.buys, self.targets),
            strategy_config_fingerprint(legacy, self.buys, self.targets),
        )
        self.process(rebuild=True)
        self.cfg = legacy
        self.process()
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM strategy_selection").fetchone()[0], 0)
        self.assertNotIn("selection", load_best_strategy_summary(self.conn, "TQQQ", "QQQ"))

    def test_schema_reinitialization_preserves_authenticated_selection(self) -> None:
        self.process(rebuild=True)
        previous = self.conn.execute("SELECT * FROM strategy_selection").fetchall()
        init_state_db(self.conn)
        self.assertEqual(self.conn.execute("SELECT * FROM strategy_selection").fetchall(), previous)
        payload = json.loads(previous[0][4])
        self.assertEqual(payload["candidate_count"], 4)
        self.assertTrue(self.matches())

    def test_selection_version_change_invalidates_cached_selection_and_rebuilds(self) -> None:
        self.process(rebuild=True)
        current_fingerprint = strategy_config_fingerprint(self.cfg, self.buys, self.targets)
        with patch("leveraged_trader.storage.SELECTION_VERSION", SELECTION_VERSION - 1):
            old_fingerprint = strategy_config_fingerprint(self.cfg, self.buys, self.targets)
        self.assertNotEqual(current_fingerprint, old_fingerprint)

        original = self.conn.execute("SELECT * FROM strategy_selection").fetchone()
        payload = json.loads(original[4])
        payload["version"] = SELECTION_VERSION - 1
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = _strategy_selection_digest((*original[:4], encoded))
        self.conn.execute("UPDATE strategy_selection SET payload_json = ?, integrity_digest = ?", (encoded, digest))
        self.assertIsNone(load_best_strategy_summary(self.conn, "TQQQ", "QQQ"))
        self.assertFalse(self.matches())

        self.process()
        self.assertTrue(self.matches())
        best = load_best_strategy_summary(self.conn, "TQQQ", "QQQ")
        self.assertEqual(best["selection"]["version"], SELECTION_VERSION)


if __name__ == "__main__":
    unittest.main()
