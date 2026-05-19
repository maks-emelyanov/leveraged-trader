from __future__ import annotations

import sqlite3
import unittest

import pandas as pd

from leveraged_trader.reports import build_buy_signal_report, build_sell_signal_report
from leveraged_trader.storage import init_state_db, save_rsi_values, save_strategy_state


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        init_state_db(self.conn)
        save_rsi_values(
            self.conn,
            "QQQ",
            14,
            pd.DataFrame(
                {
                    "close": [100.0],
                    "avg_gain": [1.0],
                    "avg_loss": [1.0],
                    "rsi": [25.0],
                },
                index=pd.to_datetime(["2026-01-02"]),
            ),
        )

    def tearDown(self) -> None:
        self.conn.close()

    def test_buy_report_requires_multiple_trades_and_min_sharpe(self) -> None:
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-02",
                "cash": 100000.0,
                "shares": 0.0,
                "in_position": False,
                "entry_price": float("nan"),
                "pending_action": "buy",
                "prev_equity": 100000.0,
                "trades_executed": 2,
            },
        )
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 1.1,
                }
            ]
        )

        report = build_buy_signal_report(self.conn, summary, 14)

        self.assertEqual(report["Asset"].tolist(), ["TQQQ"])

    def test_sell_report_does_not_require_multiple_trades_or_min_sharpe(self) -> None:
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-02",
                "cash": 0.0,
                "shares": 10.0,
                "in_position": True,
                "entry_price": 100.0,
                "pending_action": "sell",
                "prev_equity": 100000.0,
                "trades_executed": 1,
            },
        )
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 1,
                    "Sharpe": float("nan"),
                }
            ]
        )

        report = build_sell_signal_report(self.conn, summary, 14)

        self.assertEqual(report["Asset"].tolist(), ["TQQQ"])


if __name__ == "__main__":
    unittest.main()
