from __future__ import annotations

import unittest

import numpy as np

from leveraged_trader.optimized_backtest import ACTION_BUY, ACTION_NONE, run_grid_summary, run_single_equity_curve


class OptimizedBacktestTests(unittest.TestCase):
    def test_buy_cost_is_included_in_position_sizing(self) -> None:
        result = run_single_equity_curve(
            open_prices=np.array([100.0, 100.0]),
            close_prices=np.array([100.0, 100.0]),
            rsi_values=np.array([20.0, 50.0]),
            risk_free_returns=np.array([0.0, 0.0]),
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0003,
        )

        self.assertGreaterEqual(result[7], 0.0)
        self.assertAlmostEqual(result[8], 100_000.0 / (100.0 * 1.0003))
        self.assertAlmostEqual(result[0][-1], 100_000.0 / 1.0003)

    def test_invalid_open_defers_pending_buy_without_counting_a_trade(self) -> None:
        result = run_single_equity_curve(
            open_prices=np.array([100.0, 0.0]),
            close_prices=np.array([100.0, 100.0]),
            rsi_values=np.array([20.0, 50.0]),
            risk_free_returns=np.array([0.0, 0.0]),
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0003,
        )

        self.assertEqual(result[5].tolist(), [ACTION_BUY, ACTION_BUY])
        self.assertEqual(result[6].tolist(), [0, 0])
        self.assertEqual(result[11], ACTION_BUY)
        self.assertEqual(result[-1], 0)

    def test_grid_summary_matches_single_curve_accounting(self) -> None:
        open_prices = np.array([100.0, 100.0, 105.0])
        close_prices = np.array([100.0, 100.0, 105.0])
        rsi_values = np.array([20.0, 50.0, 50.0])
        risk_free_returns = np.zeros(3)
        single = run_single_equity_curve(
            open_prices,
            close_prices,
            rsi_values,
            risk_free_returns,
            30.0,
            2.0,
            100_000.0,
            0.0003,
        )
        grid = run_grid_summary(
            open_prices,
            close_prices,
            rsi_values,
            risk_free_returns,
            np.array([30.0]),
            np.array([2.0]),
            np.array([0], dtype=np.int64),
            np.array([100_000.0]),
            np.array([0.0]),
            np.array([False]),
            np.array([np.nan]),
            np.array([ACTION_NONE], dtype=np.int64),
            np.array([100_000.0]),
            np.array([0], dtype=np.int64),
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([0], dtype=np.int64),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0], dtype=np.int64),
            np.array([0.0]),
            np.array([0.0]),
            np.array([0], dtype=np.int64),
            np.array([np.nan]),
            0.0003,
        )

        self.assertTrue(grid[0][0])
        self.assertAlmostEqual(grid[1][0], single[7])
        self.assertAlmostEqual(grid[2][0], single[8])
        self.assertEqual(int(grid[3][0]), single[9])
        self.assertEqual(int(grid[5][0]), single[11])
        self.assertEqual(int(grid[7][0]), single[-1])


if __name__ == "__main__":
    unittest.main()
