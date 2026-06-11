from __future__ import annotations

import sqlite3
import unittest

import numpy as np
import pandas as pd

from leveraged_trader.backtest import performance_summary
from leveraged_trader.config import BacktestConfig, RISK_FREE_SYMBOL
from leveraged_trader.storage import init_state_db, process_asset_grid


def sample_strategy_data(periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    asset_close = np.linspace(100.0, 140.0, periods)
    signal_close = pd.Series(
        [100.0 - i for i in range(periods // 2)]
        + [80.0 + i for i in range(periods - periods // 2)],
        index=dates,
    )
    return pd.DataFrame(
        {
            "TQQQ_Open": asset_close,
            "TQQQ_High": asset_close + 1.0,
            "TQQQ_Low": asset_close - 1.0,
            "TQQQ_Close": asset_close,
            "TQQQ_Volume": 1_000_000,
            "QQQ_Open": signal_close,
            "QQQ_High": signal_close + 1.0,
            "QQQ_Low": signal_close - 1.0,
            "QQQ_Close": signal_close,
            "QQQ_Volume": 2_000_000,
            f"{RISK_FREE_SYMBOL}_Open": 5.0,
            f"{RISK_FREE_SYMBOL}_High": 5.0,
            f"{RISK_FREE_SYMBOL}_Low": 5.0,
            f"{RISK_FREE_SYMBOL}_Close": 5.0,
            f"{RISK_FREE_SYMBOL}_Volume": 0,
        },
        index=dates,
    )


class StorageOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        init_state_db(self.conn)
        self.cfg = BacktestConfig(rsi_period=3)

    def tearDown(self) -> None:
        self.conn.close()

    def process_grid(self, data: pd.DataFrame, *, rebuild: bool) -> None:
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0, 70.0],
            profit_target_values=[1.05, 1.50],
            rebuild=rebuild,
        )

    def test_process_asset_grid_stores_summaries_for_all_configs_and_only_best_equity(self) -> None:
        data = sample_strategy_data()

        self.process_grid(data, rebuild=True)

        summary_count = self.conn.execute("SELECT COUNT(*) FROM strategy_summary").fetchone()[0]
        state_count = self.conn.execute("SELECT COUNT(*) FROM strategy_state").fetchone()[0]
        equity_configs = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT buy_rsi, profit_target_multiple
                FROM strategy_equity
            )
            """
        ).fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertEqual(summary_count, 4)
        self.assertEqual(state_count, 4)
        self.assertEqual(equity_configs, 1)
        self.assertEqual(equity_rows, len(data))

    def test_best_summary_matches_stored_best_equity_curve(self) -> None:
        data = sample_strategy_data()

        self.process_grid(data, rebuild=True)

        best_summary = pd.read_sql_query(
            """
            SELECT *
            FROM strategy_summary
            ORDER BY sharpe DESC
            LIMIT 1
            """,
            self.conn,
        ).iloc[0]
        equity_df = pd.read_sql_query(
            """
            SELECT date, equity, risk_free_return
            FROM strategy_equity
            ORDER BY date
            """,
            self.conn,
            parse_dates=["date"],
        )
        curve_summary = performance_summary(
            equity_df.set_index("date")["equity"],
            equity_df.set_index("date")["risk_free_return"],
        )

        self.assertAlmostEqual(best_summary["sharpe"], curve_summary["Sharpe"])
        self.assertAlmostEqual(best_summary["total_return"], curve_summary["Total Return"])
        self.assertAlmostEqual(best_summary["cagr"], curve_summary["CAGR"])

    def test_process_asset_grid_updates_summaries_and_rewrites_single_best_equity_curve(self) -> None:
        data = sample_strategy_data()
        first_window = data.iloc[:25]
        update_window = data.iloc[24:]

        self.process_grid(first_window, rebuild=True)
        self.process_grid(update_window, rebuild=False)

        end_dates = self.conn.execute("SELECT DISTINCT end_date FROM strategy_summary").fetchall()
        equity_configs = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT buy_rsi, profit_target_multiple
                FROM strategy_equity
            )
            """
        ).fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertEqual(end_dates, [(data.index[-1].date().isoformat(),)])
        self.assertEqual(equity_configs, 1)
        self.assertEqual(equity_rows, len(data))


if __name__ == "__main__":
    unittest.main()
