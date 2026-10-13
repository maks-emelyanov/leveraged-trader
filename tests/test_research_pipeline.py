from __future__ import annotations

import sqlite3
import unittest

import numpy as np
import pandas as pd

from leveraged_trader.config import BacktestConfig
from leveraged_trader.reports import build_buy_signal_report, summarize_saved_results
from leveraged_trader.storage import (
    init_state_db,
    load_best_strategy_summary,
    load_complete_strategy_equity_curve,
    process_asset_grid,
    strategy_state_matches_config,
)


class ResearchPipelineTests(unittest.TestCase):
    def test_real_selector_storage_and_reports_require_passing_authenticated_selection(self) -> None:
        """Exercise real RSI, research blocks, storage, and report gates without mocks."""
        row_count = 800
        signal_close = 100.0 + 10.0 * np.sin(np.arange(row_count) * 2.0 * np.pi / 20.0)
        signal_close[-1] = signal_close[-2] - 10.0
        columns = {}
        for symbol, close in (
            ("TQQQ", np.full(row_count, 100.0)),
            ("QQQ", signal_close),
            ("^IRX", np.full(row_count, 4.0)),
        ):
            for field, delta in (
                ("Open", 0.0),
                ("High", 3.0 if symbol == "TQQQ" else 0.5),
                ("Low", -0.5),
                ("Close", 0.0),
            ):
                columns[f"{symbol}_{field}"] = close + delta
            columns[f"{symbol}_Volume"] = np.full(row_count, 1_000_000.0)
        data = pd.DataFrame(columns, index=pd.bdate_range("2020-01-02", periods=row_count))
        cfg = BacktestConfig()
        buys = list(range(20, 51, 5))
        targets = [1.02, 1.05, 1.10, 1.20, 1.30, 1.50, 2.00]
        report_kwargs = {
            "rsi_period": cfg.rsi_period,
            "base_cfg": cfg,
            "expected_buy_rsi_values": buys,
            "expected_profit_target_values": targets,
        }
        assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])

        with sqlite3.connect(":memory:") as conn:
            init_state_db(conn)
            process_asset_grid(conn, data, cfg, "TQQQ", "QQQ", buys, targets, True)
            best = load_best_strategy_summary(conn, "TQQQ", "QQQ")
            self.assertIsNotNone(best)
            self.assertEqual(best["selection"]["status"], "passed")
            self.assertEqual(best["selection"]["candidate_count"], 49)
            self.assertEqual(best["selection"]["mean_holding_sessions"], 1.0)
            self.assertEqual(best["selection"]["holdout_mean_holding_sessions"], 1.0)
            self.assertTrue(strategy_state_matches_config(conn, "TQQQ", "QQQ", cfg, buys, targets))

            curve = load_complete_strategy_equity_curve(
                conn,
                "TQQQ",
                "QQQ",
                best["buy_rsi"],
                best["profit_target_multiple"],
                **report_kwargs,
            )
            self.assertIsNotNone(curve)
            self.assertEqual(len(curve), row_count)
            summary, curves = summarize_saved_results(conn, assets, rsi_entry_rule="lower", **report_kwargs)
            self.assertEqual(len(summary), 1)
            self.assertEqual(len(curves), row_count)
            report = build_buy_signal_report(conn, summary, **report_kwargs)
            self.assertEqual(len(report), 1)
            self.assertEqual(report.iloc[0]["Validation Status"], "passed")
            self.assertEqual(report.iloc[0]["Pending Action"], "buy")
            self.assertEqual(report.iloc[0]["Buy RSI"], best["buy_rsi"])
            self.assertEqual(report.iloc[0]["Sell Return Multiple"], best["profit_target_multiple"])

            # Even a previously authentic passing DataFrame cannot replace
            # missing persisted validation for a subsequent report request.
            conn.execute("DELETE FROM strategy_selection")
            self.assertFalse(strategy_state_matches_config(conn, "TQQQ", "QQQ", cfg, buys, targets))
            self.assertTrue(build_buy_signal_report(conn, summary, **report_kwargs).empty)


if __name__ == "__main__":
    unittest.main()
