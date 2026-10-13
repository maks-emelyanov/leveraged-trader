from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd

from leveraged_trader.config import BacktestConfig
from leveraged_trader.reports import build_buy_signal_report, build_sell_signal_report, summarize_saved_results
from leveraged_trader.storage import init_state_db, load_best_strategy_summary, process_asset_grid


class SelectionReportTests(unittest.TestCase):
    """Selection eligibility comes from authenticated storage, never a DataFrame."""

    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.cfg = BacktestConfig()
        self.summary = pd.DataFrame([{
            "Asset": "TQQQ",
            "RSI Symbol": "QQQ",
            "Buy RSI": 30.0,
            "Sell Return Multiple": 1.1,
            "Trades Executed": 999,
            "Sharpe": 99.0,
            "Validation Status": "passed",
            "Selection Score": 999.0,
            "Development Min Annual Trades": 999.0,
            "Trade Activity Bonus": 999.0,
        }])
        self.persisted = {
            "buy_rsi": 30.0,
            "profit_target_multiple": 1.1,
            "trades_executed": 12,
            "sharpe": 1.5,
            "start_date": "2020-01-02",
            "trading_days": 800,
            "selection": {
                "method": "robust",
                "status": "passed",
                "selection_score": 0.8,
                "development_min_annual_trades": 12.0,
                "trade_activity_bonus": 0.25,
            },
        }
        self.state = {"last_date": "2026-01-05", "pending_action": "buy", "in_position": False}

    def report(
        self, *, action: str = "buy", complete_curve: bool = True, structural: bool = False,
    ) -> pd.DataFrame:
        curve = pd.DataFrame({"date": [pd.Timestamp("2026-01-05")], "equity": [110_000.0]})
        with ExitStack() as stack:
            for name, value in (
                ("load_best_strategy_summary", self.persisted),
                ("load_complete_strategy_equity_curve", curve if complete_curve else None),
                ("load_strategy_state", self.state),
                ("load_aligned_rsi_for_asset_session", ("2026-01-05", 25.0)),
            ):
                stack.enter_context(patch(f"leveraged_trader.reports.{name}", return_value=value))
            builder = build_buy_signal_report if action == "buy" else build_sell_signal_report
            provenance = {"allow_unbound_backtest_config": True} if structural else {
                "base_cfg": self.cfg,
                "expected_buy_rsi_values": [30.0],
                "expected_profit_target_values": [1.1],
            }
            return builder(
                self.conn,
                self.summary,
                self.cfg.rsi_period,
                **provenance,
            )

    def test_forged_passed_status_cannot_bypass_failed_or_missing_selection(self) -> None:
        invalid_selections = (
            None,
            {},
            {"method": "robust", "status": "failed_holdout"},
            {"method": "robust", "status": "insufficient_history"},
            {"method": "robust", "status": "insufficient_evidence"},
            {"method": "legacy_sharpe", "status": "passed"},
        )
        for selection in invalid_selections:
            with self.subTest(selection=selection):
                self.persisted["selection"] = selection
                self.assertTrue(self.report().empty)

    def test_passed_selection_displays_persisted_metrics(self) -> None:
        result = self.report()

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "Validation Status"], "passed")
        self.assertEqual(result.loc[0, "Selection Score"], 0.8)
        self.assertEqual(result.loc[0, "Development Min Annual Trades"], 12.0)
        self.assertEqual(result.loc[0, "Trade Activity Bonus"], 0.25)
        self.assertEqual(result.loc[0, "Trades Executed"], 12)
        self.assertEqual(result.loc[0, "Sharpe"], 1.5)

    def test_passed_metadata_still_requires_authenticated_curve(self) -> None:
        self.assertTrue(self.report(complete_curve=False).empty)

    def test_identity_only_inputs_preserve_authenticated_selection_fields(self) -> None:
        self.summary = self.summary[["Asset", "RSI Symbol", "Buy RSI", "Sell Return Multiple"]]
        for action in ("buy", "sell"):
            self.state.update(pending_action=action, in_position=action == "sell")
            for structural in (False, True):
                with self.subTest(action=action, structural=structural):
                    result = self.report(action=action, structural=structural)

                    self.assertEqual(len(result), 1)
                    self.assertEqual(result.loc[0, "Validation Status"], "passed")
                    self.assertEqual(result.loc[0, "Selection Score"], 0.8)
                    self.assertEqual(result.loc[0, "Development Min Annual Trades"], 12.0)
                    self.assertEqual(result.loc[0, "Trade Activity Bonus"], 0.25)

    def test_empty_robust_reports_retain_selection_schema_without_input_metadata(self) -> None:
        self.summary = self.summary[["Asset", "RSI Symbol", "Buy RSI", "Sell Return Multiple"]]
        self.persisted["selection"]["status"] = "failed_holdout"
        rejected = self.report()
        self.summary = self.summary.iloc[:0]
        empty_input = self.report()

        for result in (rejected, empty_input):
            self.assertTrue(result.empty)
            self.assertIn("Validation Status", result.columns)
            self.assertIn("Trade Activity Bonus", result.columns)

    def test_legacy_report_schema_ignores_caller_supplied_selection_fields(self) -> None:
        self.cfg = BacktestConfig(selection_method="legacy_sharpe")
        self.persisted.pop("selection")
        result = self.report()

        self.assertEqual(len(result), 1)
        self.assertNotIn("Validation Status", result.columns)
        self.assertNotIn("Selection Score", result.columns)
        self.assertNotIn("Trade Activity Bonus", result.columns)

    def test_omitting_validation_column_cannot_disable_gate(self) -> None:
        self.summary = self.summary.drop(columns=["Validation Status", "Selection Score"])
        self.persisted.pop("selection")
        self.assertTrue(self.report().empty)

    def test_forged_metrics_cannot_make_unqualified_strategy_eligible(self) -> None:
        self.persisted["sharpe"] = 0.1
        self.assertTrue(self.report().empty)
        self.persisted["sharpe"] = 1.5
        self.persisted["trades_executed"] = 1
        self.assertTrue(self.report().empty)

    def test_failed_validation_does_not_block_existing_position_exit(self) -> None:
        self.persisted["selection"] = {"method": "robust", "status": "failed_holdout"}
        self.state.update(pending_action="sell", in_position=True)

        result = self.report(action="sell")

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "Pending Action"], "sell")
        self.assertEqual(result.loc[0, "Validation Status"], "failed_holdout")


class AuthenticatedSelectionReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        init_state_db(self.conn)
        self.cfg = BacktestConfig(rsi_period=2)
        columns = {}
        for symbol, close in (
            ("TQQQ", 100.0 + np.arange(40)),
            ("QQQ", 200.0 - np.arange(40)),
            ("^IRX", np.zeros(40)),
        ):
            for suffix, multiplier in (("Open", 1), ("High", 1.05), ("Low", 0.95), ("Close", 1), ("Volume", 0)):
                columns[f"{symbol}_{suffix}"] = close * multiplier
        data = pd.DataFrame(columns, index=pd.date_range("2020-01-02", periods=40, freq="B"))
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0],
            profit_target_values=[1.02],
            rebuild=True,
        )
        self.summary = pd.DataFrame([{
            "Asset": "TQQQ",
            "RSI Symbol": "QQQ",
            "Buy RSI": 30.0,
            "Sell Return Multiple": 1.02,
            "Validation Status": "passed",
        }])

    def report(self) -> pd.DataFrame:
        return build_buy_signal_report(
            self.conn,
            self.summary,
            self.cfg.rsi_period,
            base_cfg=self.cfg,
            expected_buy_rsi_values=[30.0],
            expected_profit_target_values=[1.02],
        )

    def test_forged_dataframe_pass_cannot_bypass_authenticated_insufficient_history(self) -> None:
        persisted = load_best_strategy_summary(self.conn, "TQQQ", "QQQ")
        self.assertIsNotNone(persisted)
        self.assertGreater(persisted["trades_executed"], 1)
        self.assertGreater(persisted["sharpe"], 1.0)
        self.assertEqual(persisted["selection"]["status"], "insufficient_history")

        self.assertTrue(self.report().empty)

    def test_missing_selection_blocks_an_otherwise_actionable_authenticated_strategy(self) -> None:
        self.conn.execute("DELETE FROM strategy_selection")
        persisted = load_best_strategy_summary(self.conn, "TQQQ", "QQQ")
        self.assertIsNotNone(persisted)
        self.assertNotIn("selection", persisted)
        # The old structural inspection path proves the normal pending-buy,
        # trade-count, Sharpe, curve, and RSI gates would admit this fixture.
        structural = build_buy_signal_report(
            self.conn, self.summary, self.cfg.rsi_period, allow_unbound_backtest_config=True,
        )
        self.assertEqual(len(structural), 1)

        self.assertTrue(self.report().empty)

    def test_forged_persisted_pass_without_matching_integrity_is_rejected(self) -> None:
        raw = self.conn.execute("SELECT payload_json FROM strategy_selection").fetchone()[0]
        selection = json.loads(raw)
        selection["status"] = "passed"
        self.conn.execute(
            "UPDATE strategy_selection SET payload_json = ?",
            (json.dumps(selection, sort_keys=True, separators=(",", ":")),),
        )

        self.assertIsNone(load_best_strategy_summary(self.conn, "TQQQ", "QQQ"))
        self.assertTrue(self.report().empty)


class PositiveSelectionIntegrationTests(unittest.TestCase):
    def test_real_chronological_selection_authenticates_and_generates_a_buy_report(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        init_state_db(conn)
        cfg = BacktestConfig()
        sessions = 800
        columns = {}
        for symbol, close in (
            ("TQQQ", np.linspace(100.0, 400.0, sessions)),
            ("QQQ", np.linspace(1000.0, 300.0, sessions)),
            ("^IRX", np.zeros(sessions)),
        ):
            for suffix, multiplier in (("Open", 1), ("High", 1.002), ("Low", 0.998), ("Close", 1)):
                columns[f"{symbol}_{suffix}"] = close * multiplier
            columns[f"{symbol}_Volume"] = np.full(sessions, 1_000_000.0)
        # Make the final session hit either candidate target, leaving the
        # continuously oversold signal ready for the next pending entry.
        columns["TQQQ_High"][-1] = columns["TQQQ_Close"][-1] * 1.06
        data = pd.DataFrame(columns, index=pd.date_range("2020-01-02", periods=sessions, freq="B"))
        buys = [20.0, 30.0]
        targets = [1.02, 1.05]
        process_asset_grid(
            conn, data, cfg, "TQQQ", "QQQ", buy_rsi_values=buys, profit_target_values=targets, rebuild=True,
        )

        persisted = load_best_strategy_summary(conn, "TQQQ", "QQQ")
        self.assertIsNotNone(persisted)
        selection = persisted["selection"]
        self.assertEqual(selection["status"], "passed")
        self.assertEqual(selection["candidate_count"], 4)
        self.assertEqual(selection["development_sessions"], 640)
        self.assertEqual(selection["holdout_sessions"], 160)
        self.assertTrue(all(count >= 3 for count in selection["development_block_completed_trades"]))
        block_sessions = np.diff(np.linspace(0, selection["development_sessions"], 4, dtype=int))
        min_annual_trades = float(np.min(
            252.0 * np.array(selection["development_block_completed_trades"]) / block_sessions,
        ))
        self.assertAlmostEqual(selection["development_min_annual_trades"], min_annual_trades)
        self.assertAlmostEqual(selection["trade_activity_bonus"], 0.5 * min_annual_trades / (min_annual_trades + 12.0))
        self.assertGreaterEqual(selection["holdout_completed_trades"], 2)
        self.assertGreater(selection["holdout_sharpe"], 0.0)
        self.assertGreater(selection["mean_holding_sessions"], 1.0)

        summary, curves = summarize_saved_results(
            conn,
            pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}]),
            rsi_period=cfg.rsi_period,
            base_cfg=cfg,
            expected_buy_rsi_values=buys,
            expected_profit_target_values=targets,
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(curves), sessions)
        self.assertEqual(summary.loc[0, "Validation Status"], "passed")
        self.assertEqual(summary.loc[0, "Buy RSI"], selection["buy_rsi"])
        self.assertEqual(summary.loc[0, "Sell Return Multiple"], selection["profit_target_multiple"])
        self.assertEqual(summary.loc[0, "Development Mean Hold Sessions"], selection["mean_holding_sessions"])
        self.assertEqual(summary.loc[0, "Development Min Annual Trades"], min_annual_trades)
        self.assertEqual(summary.loc[0, "Trade Activity Bonus"], selection["trade_activity_bonus"])

        buys_report = build_buy_signal_report(
            conn,
            summary,
            cfg.rsi_period,
            base_cfg=cfg,
            expected_buy_rsi_values=buys,
            expected_profit_target_values=targets,
        )
        self.assertEqual(len(buys_report), 1)
        self.assertEqual(buys_report.loc[0, "Validation Status"], "passed")
        self.assertEqual(buys_report.loc[0, "Pending Action"], "buy")
        self.assertEqual(buys_report.loc[0, "Holdout Completed Trades"], selection["holdout_completed_trades"])
        self.assertEqual(buys_report.loc[0, "Development Min Annual Trades"], min_annual_trades)
        self.assertEqual(buys_report.loc[0, "Trade Activity Bonus"], selection["trade_activity_bonus"])


if __name__ == "__main__":
    unittest.main()
