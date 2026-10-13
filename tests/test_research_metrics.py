from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from leveraged_trader.backtest import performance_summary
from leveraged_trader.optimized_backtest import RSI_ENTRY_LOWER, RSI_ENTRY_UPPER, run_single_equity_curve
from leveraged_trader.research_metrics import grid_research_metrics


class ResearchMetricsTests(unittest.TestCase):
    def _metrics(self, **overrides: object) -> dict[str, np.ndarray]:
        inputs = {
            "open_prices": np.full(8, 100.0),
            "high_prices": np.array([100.0, 105.0, 110.0, 100.0, 110.0, 100.0, 105.0, 106.0]),
            "close_prices": np.full(8, 100.0),
            "rsi_values": np.array([20.0, 50.0, 50.0, 20.0, 50.0, 20.0, 50.0, 50.0]),
            "risk_free_returns": np.zeros(8),
            "buy_rsi_values": np.array([30.0]),
            "profit_target_values": np.array([1.1]),
            "initial_capital": 100_000.0,
            "trading_cost_rate": 0.0,
        }
        return grid_research_metrics(**(inputs | overrides))

    def test_durations_include_same_day_exits_and_final_open_position(self) -> None:
        # Entries at 1, 4, and 6; exits at 2 and 4. The final age is 2.
        for entry_rule in (RSI_ENTRY_LOWER, RSI_ENTRY_UPPER):
            rsi = np.array([20.0, 50.0, 50.0, 20.0, 50.0, 20.0, 50.0, 50.0])
            if entry_rule == RSI_ENTRY_UPPER:
                rsi = 100.0 - rsi
            with self.subTest(entry_rule=entry_rule):
                result = self._metrics(
                    rsi_values=rsi,
                    buy_rsi_values=np.array([30.0 if entry_rule == RSI_ENTRY_LOWER else 70.0]),
                    rsi_entry_rule=entry_rule,
                )
                self.assertEqual(result["completed_trades"][0], 2)
                self.assertAlmostEqual(result["mean_holding_sessions"][0], 5.0 / 3.0)
                self.assertEqual(result["max_holding_sessions"][0], 2)
                self.assertEqual(result["open_holding_sessions"][0], 2)
                self.assertEqual(result["exposure_fraction"][0], 5.0 / 8.0)

    def test_same_day_roundtrips_preserve_entry_exit_costs_and_tick_rounding(self) -> None:
        for price in (100.0, 0.1):
            with self.subTest(price=price):
                result = self._metrics(
                    open_prices=np.full(3, price),
                    high_prices=np.array([price, 110.0 if price == 100.0 else 0.11, 110.0 if price == 100.0 else 0.11]),
                    close_prices=np.full(3, price),
                    rsi_values=np.array([20.0, 20.0, 50.0]),
                    risk_free_returns=np.zeros(3),
                    trading_cost_rate=0.001,
                )
                self.assertEqual(result["completed_trades"][0], 2)
                self.assertEqual(result["mean_holding_sessions"][0], 1.0)
                self.assertEqual(result["max_holding_sessions"][0], 1)
                self.assertEqual(result["open_holding_sessions"][0], 0)
                self.assertEqual(result["exposure_fraction"][0], 2.0 / 3.0)
                self.assertAlmostEqual(result["total_return"][0], (1.1 * 0.999 / 1.001) ** 2 - 1.0)

    def test_open_only_trade_has_censored_age_without_realized_exit(self) -> None:
        result = self._metrics(profit_target_values=np.array([2.0]))
        self.assertEqual(result["completed_trades"][0], 0)
        self.assertEqual(result["mean_holding_sessions"][0], 7.0)
        self.assertEqual(result["max_holding_sessions"][0], 7)
        self.assertEqual(result["open_holding_sessions"][0], 7)
        self.assertEqual(result["exposure_fraction"][0], 7.0 / 8.0)

    def test_metrics_match_single_curve_with_paired_parameters_and_costs(self) -> None:
        open_prices = np.array([100.0, 101.0, 104.01, 103.0, 99.99, 102.0, 106.0, 107.0])
        high_prices = np.array([100.0, 104.01, 104.01, 105.0, 102.0, 106.0, 109.0, 108.0])
        close_prices = np.array([100.0, 102.0, 103.0, 104.0, 101.0, 105.0, 107.0, 106.0])
        rsi_values = np.array([np.nan, 20.0, 80.0, 30.0, 70.0, 50.0, 80.0, 20.0])
        risk_free_returns = np.array([np.nan, 0.0001, 0.0001, 0.0002, 0.0, 0.0001, 0.0001, 0.0002])
        thresholds = np.array([30.0, 55.0, 70.0])
        targets = np.array([1.03, 1.08, 1.04])
        for entry_rule in (RSI_ENTRY_LOWER, RSI_ENTRY_UPPER):
            metrics = grid_research_metrics(
                open_prices,
                high_prices,
                close_prices,
                rsi_values,
                risk_free_returns,
                thresholds,
                targets,
                100_000.0,
                0.0003,
                entry_rule,
            )
            for idx, (threshold, target) in enumerate(zip(thresholds, targets, strict=True)):
                with self.subTest(entry_rule=entry_rule, threshold=threshold, target=target):
                    curve = run_single_equity_curve(
                        open_prices,
                        high_prices,
                        close_prices,
                        rsi_values,
                        risk_free_returns,
                        threshold,
                        target,
                        100_000.0,
                        0.0003,
                        entry_rule,
                    )
                    expected = performance_summary(pd.Series(curve[0]), pd.Series(risk_free_returns))
                    self.assertEqual(metrics["total_return"][idx], expected["Total Return"])
                    self.assertAlmostEqual(metrics["sharpe"][idx], expected["Sharpe"], places=12)
                    self.assertEqual(metrics["completed_trades"][idx], curve[13] // 2)

    def test_missing_benchmark_rejects_partial_sharpe_but_not_missing_first_mark(self) -> None:
        complete = np.zeros(8)
        complete[0] = np.nan
        self.assertTrue(np.isfinite(self._metrics(risk_free_returns=complete)["sharpe"][0]))
        complete[3] = np.nan
        result = self._metrics(risk_free_returns=complete)
        self.assertTrue(np.isnan(result["sharpe"][0]))
        self.assertEqual(result["completed_trades"][0], 2)

    def test_no_trades_and_empty_history_have_explicit_empty_metrics(self) -> None:
        no_trades = self._metrics(rsi_values=np.full(8, np.nan))
        empty = self._metrics(
            **{
                name: np.empty(0)
                for name in ("open_prices", "high_prices", "close_prices", "rsi_values", "risk_free_returns")
            }
        )
        for result in (no_trades, empty):
            self.assertTrue(np.isnan(result["mean_holding_sessions"][0]))
            self.assertTrue(np.isnan(result["sharpe"][0]))
            for name in (
                "completed_trades",
                "max_holding_sessions",
                "open_holding_sessions",
                "exposure_fraction",
                "total_return",
            ):
                self.assertEqual(result[name][0], 0)
        empty_grid = self._metrics(buy_rsi_values=np.empty(0), profit_target_values=np.empty(0))
        self.assertTrue(all(value.shape == (0,) for value in empty_grid.values()))

    def test_invalid_market_and_parameter_arrays_fail_before_simulation(self) -> None:
        invalid_cases = (
            ({"profit_target_values": np.array([1.1, 1.2])}, "must contain 1 values"),
            ({"buy_rsi_values": np.array([True])}, "booleans"),
            ({"high_prices": np.full(8, 99.0)}, "at least"),
            ({"risk_free_returns": np.full(8, -1.0)}, "greater than -1.0"),
            ({"trading_cost_rate": 1.0}, "less than 1.0"),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                self._metrics(**overrides)

    def test_unreachable_extreme_price_is_allowed_but_executed_invalid_target_is_rejected(self) -> None:
        for prices, signals, should_raise in (
            (np.array([1e26, 1.0, 1.0]), np.array([20.0, 50.0, 50.0]), False),
            (np.array([1.0, 1e26, 1.0]), np.array([20.0, 50.0, 50.0]), True),
        ):
            inputs = {
                "open_prices": prices,
                "high_prices": prices,
                "close_prices": prices,
                "rsi_values": signals,
                "risk_free_returns": np.zeros(3),
            }
            if should_raise:
                with self.assertRaisesRegex(ValueError, "tick precision"):
                    self._metrics(**inputs)
            else:
                self.assertEqual(self._metrics(**inputs)["open_holding_sessions"][0], 2)

    def test_nonfinite_account_paths_and_rollups_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self._metrics(
                initial_capital=1e308,
                profit_target_values=np.array([2.0]),
                high_prices=np.array([100.0, 200.0, 110.0, 100.0, 110.0, 100.0, 105.0, 106.0]),
            )
        with self.assertRaisesRegex(ValueError, "returns"):
            self._metrics(risk_free_returns=np.full(8, 1e308))


if __name__ == "__main__":
    unittest.main()
