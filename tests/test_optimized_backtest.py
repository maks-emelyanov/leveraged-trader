from __future__ import annotations

import math
import unittest
import warnings
from datetime import date, datetime, timedelta
from decimal import ROUND_DOWN, Inexact, localcontext

import numpy as np
import pandas as pd

from leveraged_trader.alpaca import _target_sell_price
from leveraged_trader.optimized_backtest import (
    ACTION_BUY,
    ACTION_NONE,
    ACTION_SELL,
    RSI_ENTRY_UPPER,
    _adjacent_float64_values,
    _fully_determined_return_rollup_is_consistent,
    _repeated_float64_sum,
    _return_rollup_equity_endpoints_are_consistent,
    _target_limit_price,
    _target_price_overrides,
    _target_price_with_override,
    run_grid_summary,
    run_single_equity_curve,
)
from leveraged_trader.pricing import buffered_buy_limit_price


class OptimizedBacktestTests(unittest.TestCase):
    @staticmethod
    def _run_minimal_grid(
        buy_rsi_values: np.ndarray | list[float | bool],
        *,
        rsi_entry_rule: object = 0,
        open_prices: object | None = None,
        high_prices: object | None = None,
        close_prices: object | None = None,
        rsi_values: object | None = None,
        risk_free_returns: object | None = None,
        profit_target_values: object | None = None,
        initial_capital: float = 100_000.0,
        trading_cost_rate: float = 0.0,
        state_overrides: dict[str, object] | None = None,
    ) -> tuple:
        config_count = len(buy_rsi_values)
        state = {
            "start_indices": np.zeros(config_count, dtype=np.int64),
            "cash_values": np.full(config_count, initial_capital),
            "share_values": np.zeros(config_count),
            "in_position_values": np.zeros(config_count, dtype=np.bool_),
            "entry_price_values": np.full(config_count, np.nan),
            "pending_action_values": np.full(config_count, ACTION_NONE, dtype=np.int64),
            "prev_equity_values": np.full(config_count, initial_capital),
            "trades_executed_values": np.zeros(config_count, dtype=np.int64),
            "first_equity_values": np.full(config_count, np.nan),
            "last_equity_values": np.full(config_count, np.nan),
            "running_max_equity_values": np.full(config_count, np.nan),
            "return_count_values": np.zeros(config_count, dtype=np.int64),
            "return_sum_values": np.zeros(config_count),
            "return_sum_squares_values": np.zeros(config_count),
            "excess_return_count_values": np.zeros(config_count, dtype=np.int64),
            "excess_return_sum_values": np.zeros(config_count),
            "excess_return_sum_squares_values": np.zeros(config_count),
            "positive_return_count_values": np.zeros(config_count, dtype=np.int64),
            "max_drawdown_values": np.full(config_count, np.nan),
        }
        if state_overrides is not None:
            state.update(state_overrides)
        return run_grid_summary(
            np.array([100.0]) if open_prices is None else open_prices,
            np.array([100.0]) if high_prices is None else high_prices,
            np.array([100.0]) if close_prices is None else close_prices,
            np.array([20.0]) if rsi_values is None else rsi_values,
            np.array([0.0]) if risk_free_returns is None else risk_free_returns,
            buy_rsi_values,
            np.full(config_count, 2.0) if profit_target_values is None else profit_target_values,
            trading_cost_rate=trading_cost_rate,
            rsi_entry_rule=rsi_entry_rule,
            **state,
        )

    @staticmethod
    def _flat_history_state(equity: float = 100_000.0) -> dict[str, np.ndarray]:
        return {
            "cash_values": np.array([equity]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([equity]),
            "trades_executed_values": np.array([0], dtype=np.int64),
            "first_equity_values": np.array([equity]),
            "last_equity_values": np.array([equity]),
            "running_max_equity_values": np.array([equity]),
            "return_count_values": np.array([0], dtype=np.int64),
            "return_sum_values": np.array([0.0]),
            "return_sum_squares_values": np.array([0.0]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "positive_return_count_values": np.array([0], dtype=np.int64),
            "max_drawdown_values": np.array([0.0]),
        }

    @staticmethod
    def _held_history_state(
        *,
        cash: float = 0.0,
        shares: float = 1_000.0,
        entry_price: float = 100.0,
        equity: float = 100_000.0,
    ) -> dict[str, np.ndarray]:
        return {
            "cash_values": np.array([cash]),
            "share_values": np.array([shares]),
            "in_position_values": np.array([True]),
            "entry_price_values": np.array([entry_price]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([equity]),
            "trades_executed_values": np.array([1], dtype=np.int64),
            "first_equity_values": np.array([equity]),
            "last_equity_values": np.array([equity]),
            "running_max_equity_values": np.array([equity]),
            "return_count_values": np.array([1], dtype=np.int64),
            "return_sum_values": np.array([0.0]),
            "return_sum_squares_values": np.array([0.0]),
            "excess_return_count_values": np.array([1], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "positive_return_count_values": np.array([0], dtype=np.int64),
            "max_drawdown_values": np.array([0.0]),
            "resume_close_values": np.array([(equity - cash) / shares if shares > 0.0 else np.nan]),
        }

    def test_single_curve_rejects_boolean_and_out_of_range_buy_rsi(self) -> None:
        for buy_rsi, expected_message in (
            (True, "must not be a boolean"),
            (np.bool_(False), "must not be a boolean"),
            (-0.01, "between 0.0 and 100.0"),
            (100.01, "between 0.0 and 100.0"),
        ):
            with self.subTest(buy_rsi=buy_rsi), self.assertRaisesRegex(ValueError, expected_message):
                run_single_equity_curve(
                    open_prices=np.array([100.0]),
                    high_prices=np.array([100.0]),
                    close_prices=np.array([100.0]),
                    rsi_values=np.array([20.0]),
                    risk_free_returns=np.array([0.0]),
                    buy_rsi=buy_rsi,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=0.0,
                )

    def test_grid_summary_rejects_boolean_and_out_of_range_buy_rsi_values(self) -> None:
        for buy_rsi_values, expected_message in (
            (np.array([True]), "must not contain booleans"),
            ([30.0, True], "must not contain booleans"),
            (np.array([30.0 + 0.0j]), "must not contain complex values"),
            (np.array([-0.01]), "between 0.0 and 100.0"),
            (np.array([100.01]), "between 0.0 and 100.0"),
        ):
            with (
                self.subTest(buy_rsi_values=buy_rsi_values),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                self._run_minimal_grid(buy_rsi_values)

    def test_public_optimized_apis_reject_boolean_and_complex_numeric_values(self) -> None:
        base_inputs: dict[str, object] = {
            "open_prices": np.array([100.0, 100.0]),
            "high_prices": np.array([100.0, 100.0]),
            "close_prices": np.array([100.0, 100.0]),
            "rsi_values": np.array([20.0, 50.0]),
            "risk_free_returns": np.array([0.0, 0.0]),
            "buy_rsi": 30.0,
            "profit_target_multiple": 2.0,
            "initial_capital": 100_000.0,
            "trading_cost_rate": 0.0,
        }
        cases = (
            ("mixed boolean array", {"open_prices": [100.0, True]}, "open_prices must not contain booleans"),
            (
                "zero-dimensional wrapped boolean array",
                {"open_prices": np.array([100.0, np.array(True)], dtype=object)},
                "open_prices must not contain booleans",
            ),
            (
                "complex array",
                {"open_prices": np.array([100.0 + 1.0j, 100.0 - 1.0j])},
                "open_prices must not contain complex values",
            ),
            (
                "zero-dimensional wrapped complex array",
                {"open_prices": np.array([100.0, np.array(100.0 + 1.0j)], dtype=object)},
                "open_prices must not contain complex values",
            ),
            (
                "boolean risk-free return",
                {"risk_free_returns": np.array([0.0, True], dtype=object)},
                "risk_free_returns must not contain booleans",
            ),
            ("boolean scalar", {"initial_capital": True}, "initial_capital must be a finite number"),
            (
                "complex scalar",
                {"profit_target_multiple": np.complex64(2.0 + 0.0j)},
                "profit_target_multiple must be a finite number",
            ),
        )

        for label, override, expected_message in cases:
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                run_single_equity_curve(**{**base_inputs, **override})

    def test_scalar_thresholds_reject_temporal_values_through_zero_dimensional_wrappers(self) -> None:
        nested_temporal = np.empty((), dtype=object)
        nested_temporal[()] = np.array(np.datetime64(30, "ns"))
        temporal_values = (
            np.datetime64(30, "ns"),
            np.array(np.datetime64(30, "ns")),
            nested_temporal,
        )

        for value in temporal_values:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "buy_rsi must be a finite number"):
                run_single_equity_curve(
                    open_prices=np.array([100.0]),
                    high_prices=np.array([100.0]),
                    close_prices=np.array([100.0]),
                    rsi_values=np.array([20.0]),
                    risk_free_returns=np.array([0.0]),
                    buy_rsi=value,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=0.0,
                )

    def test_public_optimized_apis_reject_datetime_and_timedelta_price_dtypes(self) -> None:
        temporal_arrays = {
            "numpy datetime64": np.array(["2026-01-02"], dtype="datetime64[D]"),
            "numpy timedelta64": np.array([1], dtype="timedelta64[D]"),
        }
        for label, values in temporal_arrays.items():
            with (
                self.subTest(label=label, api="single"),
                self.assertRaisesRegex(ValueError, "open_prices must not contain date, datetime, or timedelta"),
            ):
                run_single_equity_curve(
                    open_prices=values,
                    high_prices=np.array([100.0]),
                    close_prices=np.array([100.0]),
                    rsi_values=np.array([20.0]),
                    risk_free_returns=np.array([0.0]),
                    buy_rsi=30.0,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=0.0,
                )
            with (
                self.subTest(label=label, api="grid"),
                self.assertRaisesRegex(ValueError, "open_prices must not contain date, datetime, or timedelta"),
            ):
                self._run_minimal_grid(np.array([30.0]), open_prices=values)

    def test_public_optimized_api_rejects_mixed_temporal_scalar_values(self) -> None:
        nested_temporal = np.empty((), dtype=object)
        nested_temporal[()] = np.array(np.datetime64(30, "ns"))
        temporal_values = (
            date(2026, 1, 2),
            datetime(2026, 1, 2, 12),
            timedelta(days=1),
            np.datetime64("2026-01-02"),
            np.timedelta64(1, "D"),
            pd.Timestamp("2026-01-02"),
            pd.Timedelta(days=1),
            np.array(np.datetime64(30, "ns")),
            nested_temporal,
        )
        for value in temporal_values:
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "open_prices must not contain date, datetime, or timedelta"),
            ):
                run_single_equity_curve(
                    open_prices=np.array([100.0, value], dtype=object),
                    high_prices=np.array([100.0, 100.0]),
                    close_prices=np.array([100.0, 100.0]),
                    rsi_values=np.array([20.0, 50.0]),
                    risk_free_returns=np.array([0.0, 0.0]),
                    buy_rsi=30.0,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=0.0,
                )

    def test_grid_rejects_semantic_values_inside_zero_dimensional_open_wrappers(self) -> None:
        nested_temporal = np.empty((), dtype=object)
        nested_temporal[()] = np.array(np.datetime64(30, "ns"))
        cases = (
            (np.array(True), "open_prices must not contain booleans"),
            (np.array(100.0 + 1.0j), "open_prices must not contain complex values"),
            (np.array(np.datetime64(30, "ns")), "open_prices must not contain date, datetime, or timedelta"),
            (nested_temporal, "open_prices must not contain date, datetime, or timedelta"),
        )
        for value, message in cases:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=np.array([value], dtype=object),
                )

    def test_optimized_numeric_conversion_overflow_is_normalized(self) -> None:
        class OverflowingFloat:
            def __float__(self) -> float:
                raise OverflowError("outside float range")

        with self.assertRaisesRegex(ValueError, "open_prices must be a one-dimensional numeric array"):
            run_single_equity_curve(
                open_prices=np.array([100.0, OverflowingFloat()], dtype=object),
                high_prices=np.array([100.0, 100.0]),
                close_prices=np.array([100.0, 100.0]),
                rsi_values=np.array([20.0, 50.0]),
                risk_free_returns=np.array([0.0, 0.0]),
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=100_000.0,
                trading_cost_rate=0.0,
            )

        with self.assertRaisesRegex(ValueError, "initial_capital must be a finite number"):
            run_single_equity_curve(
                open_prices=np.array([100.0]),
                high_prices=np.array([100.0]),
                close_prices=np.array([100.0]),
                rsi_values=np.array([20.0]),
                risk_free_returns=np.array([0.0]),
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=OverflowingFloat(),
                trading_cost_rate=0.0,
            )

    def test_rsi_entry_rule_requires_a_non_boolean_integer_code(self) -> None:
        invalid_values = (
            True,
            np.bool_(False),
            0.0,
            1.0,
            0.5,
            1.0 + 0.0j,
            *(np.timedelta64(value, unit) for value, unit in ((0, "ns"), (1, "D"), (1, "M"), (1, "Y"))),
            *(np.array(np.timedelta64(value, unit)) for value, unit in ((0, "ns"), (1, "D"), (1, "M"), (1, "Y"))),
        )
        for value in invalid_values:
            with (
                self.subTest(value=value, api="single"),
                self.assertRaisesRegex(ValueError, "rsi_entry_rule must be RSI_ENTRY_LOWER"),
            ):
                run_single_equity_curve(
                    open_prices=np.array([100.0]),
                    high_prices=np.array([100.0]),
                    close_prices=np.array([100.0]),
                    rsi_values=np.array([20.0]),
                    risk_free_returns=np.array([0.0]),
                    buy_rsi=30.0,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=0.0,
                    rsi_entry_rule=value,
                )
            with (
                self.subTest(value=value, api="grid"),
                self.assertRaisesRegex(ValueError, "rsi_entry_rule must be RSI_ENTRY_LOWER"),
            ):
                self._run_minimal_grid(np.array([30.0]), rsi_entry_rule=value)

        result = run_single_equity_curve(
            open_prices=np.array([100.0]),
            high_prices=np.array([100.0]),
            close_prices=np.array([100.0]),
            rsi_values=np.array([80.0]),
            risk_free_returns=np.array([0.0]),
            buy_rsi=70.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0,
            rsi_entry_rule=np.int64(RSI_ENTRY_UPPER),
        )
        self.assertEqual(result[5].tolist(), [ACTION_BUY])

    def test_public_optimized_apis_accept_inclusive_rsi_threshold_boundaries(self) -> None:
        for buy_rsi in (0.0, 100.0):
            with self.subTest(buy_rsi=buy_rsi):
                run_single_equity_curve(
                    open_prices=np.array([100.0]),
                    high_prices=np.array([100.0]),
                    close_prices=np.array([100.0]),
                    rsi_values=np.array([20.0]),
                    risk_free_returns=np.array([0.0]),
                    buy_rsi=buy_rsi,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=0.0,
                )
                self._run_minimal_grid(np.array([buy_rsi]))

    def test_rsi_observations_must_be_in_domain_but_may_be_nan(self) -> None:
        for value in (-1.0, 101.0):
            with (
                self.subTest(value=value, api="single"),
                self.assertRaisesRegex(ValueError, "rsi_values must contain values between 0.0 and 100.0"),
            ):
                run_single_equity_curve(
                    open_prices=np.array([100.0]),
                    high_prices=np.array([100.0]),
                    close_prices=np.array([100.0]),
                    rsi_values=np.array([value]),
                    risk_free_returns=np.array([0.0]),
                    buy_rsi=30.0,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=0.0,
                )
            with (
                self.subTest(value=value, api="grid"),
                self.assertRaisesRegex(ValueError, "rsi_values must contain values between 0.0 and 100.0"),
            ):
                self._run_minimal_grid(np.array([30.0]), rsi_values=np.array([value]))

        run_single_equity_curve(
            open_prices=np.array([100.0]),
            high_prices=np.array([100.0]),
            close_prices=np.array([100.0]),
            rsi_values=np.array([np.nan]),
            risk_free_returns=np.array([0.0]),
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0,
        )
        self._run_minimal_grid(np.array([30.0]), rsi_values=np.array([np.nan]))

    def test_risk_free_returns_must_exceed_total_loss_but_may_be_nan(self) -> None:
        prices = np.array([100.0, 100.0])
        rsi_values = np.array([np.nan, np.nan])
        for value in (-1.0, -1.1):
            with (
                self.subTest(value=value, api="single"),
                self.assertRaisesRegex(ValueError, "risk_free_returns must contain values greater than -1.0"),
            ):
                run_single_equity_curve(
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=rsi_values,
                    risk_free_returns=np.array([np.nan, value]),
                    buy_rsi=30.0,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=0.0,
                )
            with (
                self.subTest(value=value, api="grid"),
                self.assertRaisesRegex(ValueError, "risk_free_returns must contain values greater than -1.0"),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=rsi_values,
                    risk_free_returns=np.array([np.nan, value]),
                )

        valid_returns = np.array([np.nan, np.nextafter(-1.0, 0.0)])
        run_single_equity_curve(
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=rsi_values,
            risk_free_returns=valid_returns,
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0,
        )
        self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=rsi_values,
            risk_free_returns=valid_returns,
        )

    def test_single_curve_rejects_trading_cost_rates_at_or_above_one(self) -> None:
        for trading_cost_rate in [1.0, 1.5]:
            with (
                self.subTest(trading_cost_rate=trading_cost_rate),
                self.assertRaisesRegex(ValueError, "less than 1.0"),
            ):
                run_single_equity_curve(
                    open_prices=np.array([100.0]),
                    high_prices=np.array([100.0]),
                    close_prices=np.array([100.0]),
                    rsi_values=np.array([20.0]),
                    risk_free_returns=np.array([0.0]),
                    buy_rsi=30.0,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=trading_cost_rate,
                )

    def test_grid_summary_rejects_trading_cost_rates_at_or_above_one(self) -> None:
        for trading_cost_rate in [1.0, 1.5]:
            with (
                self.subTest(trading_cost_rate=trading_cost_rate),
                self.assertRaisesRegex(ValueError, "less than 1.0"),
            ):
                run_grid_summary(
                    np.array([100.0]),
                    np.array([100.0]),
                    np.array([100.0]),
                    np.array([20.0]),
                    np.array([0.0]),
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
                    trading_cost_rate,
                )

    def test_grid_rejects_incoherent_position_and_trade_state_before_kernel(self) -> None:
        flat = self._flat_history_state()
        held = self._held_history_state()
        cases = (
            (
                "flat phantom shares",
                {**flat, "share_values": np.array([10.0]), "entry_price_values": np.array([100.0])},
                "flat resumable state requires zero shares",
            ),
            (
                "flat entry price",
                {**flat, "entry_price_values": np.array([100.0])},
                "flat resumable state requires zero shares",
            ),
            (
                "held zero shares",
                {**held, "share_values": np.array([0.0])},
                "in-position resumable state requires positive shares",
            ),
            (
                "held missing entry",
                {**held, "entry_price_values": np.array([np.nan])},
                "in-position resumable state requires positive shares",
            ),
            (
                "held underflowed entry notional",
                self._held_history_state(
                    shares=1e-200,
                    entry_price=1e-200,
                    equity=1.0,
                ),
                "positive finite entry notional",
            ),
            (
                "held even trades",
                {**held, "trades_executed_values": np.array([2], dtype=np.int64)},
                "in-position resumable state requires an odd trade count",
            ),
            (
                "flat odd trades",
                {**flat, "trades_executed_values": np.array([1], dtype=np.int64)},
                "flat resumable state requires an even trade count",
            ),
            (
                "held pending buy",
                {**held, "pending_action_values": np.array([ACTION_BUY], dtype=np.int64)},
                "in-position resumable state cannot have a pending action",
            ),
            (
                "flat pending sell",
                {**flat, "pending_action_values": np.array([ACTION_SELL], dtype=np.int64)},
                "flat resumable state may only have no pending action",
            ),
        )

        for label, state, expected_message in cases:
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                self._run_minimal_grid(np.array([30.0]), state_overrides=state)

    def test_grid_rejects_trade_counts_unreachable_from_rollup_observations(self) -> None:
        cases = (
            (
                "flat one-observation rollup",
                {
                    **self._flat_history_state(),
                    "trades_executed_values": np.array([2], dtype=np.int64),
                },
            ),
            (
                "held two-observation rollup",
                {
                    **self._held_history_state(),
                    "trades_executed_values": np.array([3], dtype=np.int64),
                },
            ),
        )

        for label, state in cases:
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, "cannot exceed twice return_count_values"),
            ):
                self._run_minimal_grid(np.array([30.0]), state_overrides=state)

    def test_grid_rejects_tiny_incoherent_no_work_resume_account_state(self) -> None:
        state = {
            **self._flat_history_state(equity=5e-14),
            "start_indices": np.array([1], dtype=np.int64),
            "prev_equity_values": np.array([8e-14]),
            "last_equity_values": np.array([8e-14]),
        }

        with self.assertRaisesRegex(ValueError, "cash must equal its previous equity"):
            self._run_minimal_grid(
                np.array([30.0]),
                initial_capital=5e-14,
                state_overrides=state,
            )

        maximum_float = np.finfo(np.float64).max
        maximum_state = {
            **self._flat_history_state(equity=maximum_float),
            "start_indices": np.array([1], dtype=np.int64),
            "cash_values": np.array([1.0]),
        }
        with self.assertRaisesRegex(ValueError, "cash must equal its previous equity"):
            self._run_minimal_grid(
                np.array([30.0]),
                initial_capital=maximum_float,
                state_overrides=maximum_state,
            )

        tiny_valid_state = {
            **self._flat_history_state(equity=1e-300),
            "start_indices": np.array([1], dtype=np.int64),
        }
        result = self._run_minimal_grid(
            np.array([30.0]),
            initial_capital=1e-300,
            state_overrides=tiny_valid_state,
        )

        self.assertFalse(result[0][0])
        self.assertEqual(result[1][0], 1e-300)
        self.assertEqual(result[6][0], 1e-300)

    def test_grid_accepts_maximum_reachable_trade_count_for_rollup(self) -> None:
        state = {
            **self._flat_history_state(),
            "start_indices": np.array([1], dtype=np.int64),
            "history_prefix_observation_counts": np.array([1], dtype=np.int64),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "return_count_values": np.array([1], dtype=np.int64),
        }

        result = self._run_minimal_grid(np.array([30.0]), state_overrides=state)

        self.assertFalse(result[0][0])
        self.assertEqual(result[7][0], 2)

    def test_grid_rejects_material_cash_but_allows_ulp_residue_while_held(self) -> None:
        held = self._held_history_state(cash=1.0)
        with self.assertRaisesRegex(ValueError, "only retain float-roundoff cash"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=held)

        capital = 1e15
        trading_cost_rate = 0.1
        turnover = capital / (1.0 + trading_cost_rate)
        shares = turnover / 100.0
        residual_cash = max(capital - turnover * (1.0 + trading_cost_rate), 0.0)
        equity = residual_cash + shares * 100.0
        valid_roundoff = self._held_history_state(
            cash=residual_cash,
            shares=shares,
            equity=equity,
        )
        valid_roundoff["start_indices"] = np.array([2], dtype=np.int64)
        prices = np.array([100.0, 100.0])

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.array([50.0, 50.0]),
            risk_free_returns=np.array([0.0, 0.0]),
            initial_capital=capital,
            trading_cost_rate=trading_cost_rate,
            state_overrides=valid_roundoff,
        )

        self.assertTrue(result[3][0])
        self.assertLessEqual(result[1][0], residual_cash)

    def test_grid_rejects_material_high_capital_held_mark_discontinuity(self) -> None:
        capital = 1e15
        held = self._held_history_state(
            shares=(capital - 500.0) / 100.0,
            equity=capital,
        )
        held["start_indices"] = np.array([1], dtype=np.int64)
        held["history_prefix_observation_counts"] = np.array([1], dtype=np.int64)
        held["resume_close_values"] = np.array([100.0])
        prices = np.array([100.0, 100.0])

        with self.assertRaisesRegex(
            ValueError,
            "shares valued at its last closing price",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.array([50.0, 50.0]),
                risk_free_returns=np.array([0.0, 0.0]),
                initial_capital=capital,
                state_overrides=held,
            )

    def test_grid_subnormal_held_mark_requires_relative_continuity(self) -> None:
        minimum_subnormal = float(np.nextafter(0.0, np.inf))
        prices = np.ones(3)
        exact = self._held_history_state(
            shares=minimum_subnormal,
            entry_price=1.0,
            equity=minimum_subnormal,
        )
        exact["start_indices"] = np.array([2], dtype=np.int64)
        exact["resume_close_values"] = np.array([1.0])

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.full(3, 50.0),
            risk_free_returns=np.zeros(3),
            initial_capital=minimum_subnormal,
            state_overrides=exact,
        )

        self.assertEqual(result[6][0], minimum_subnormal)
        self.assertEqual(result[12][0], 0.0)

        inconsistent = {
            **exact,
            "prev_equity_values": np.array([33.0 * minimum_subnormal]),
            "first_equity_values": np.array([33.0 * minimum_subnormal]),
            "last_equity_values": np.array([33.0 * minimum_subnormal]),
            "running_max_equity_values": np.array([33.0 * minimum_subnormal]),
        }
        with self.assertRaisesRegex(
            ValueError,
            "shares valued at its last closing price",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.full(3, 50.0),
                risk_free_returns=np.zeros(3),
                initial_capital=minimum_subnormal,
                state_overrides=inconsistent,
            )

    def test_adjacent_dbl_max_search_does_not_emit_overflow_warning(self) -> None:
        maximum_float = float(np.finfo(np.float64).max)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            adjacent = _adjacent_float64_values(maximum_float, radius=8)

        self.assertIn(maximum_float, adjacent)
        self.assertTrue(all(np.isfinite(value) for value in adjacent))

    def test_grid_requires_complete_or_pristine_empty_rollup_state(self) -> None:
        partial_rollup = {
            "first_equity_values": np.array([100_000.0]),
        }
        with self.assertRaisesRegex(ValueError, "either all present or all missing"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=partial_rollup)

        non_pristine_empty = {
            "pending_action_values": np.array([ACTION_BUY], dtype=np.int64),
        }
        with self.assertRaisesRegex(ValueError, "empty resumable rollup requires pristine"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=non_pristine_empty)

    def test_grid_rejects_discontinuous_or_impossible_equity_rollup(self) -> None:
        continuity = {
            **self._flat_history_state(),
            "cash_values": np.array([200_000.0]),
            "prev_equity_values": np.array([200_000.0]),
        }
        running_max = {
            **self._flat_history_state(90_000.0),
            "first_equity_values": np.array([100_000.0]),
            "running_max_equity_values": np.array([80_000.0]),
            "return_count_values": np.array([1], dtype=np.int64),
            "return_sum_values": np.array([-0.1]),
            "return_sum_squares_values": np.array([0.01]),
            "max_drawdown_values": np.array([-0.1]),
        }
        positive_drawdown = {
            **self._flat_history_state(),
            "max_drawdown_values": np.array([1.0]),
        }
        below_total_loss = {
            **self._flat_history_state(),
            "max_drawdown_values": np.array([-1.1]),
        }
        impossible_drawdown = {
            **self._flat_history_state(90_000.0),
            "first_equity_values": np.array([100_000.0]),
            "running_max_equity_values": np.array([100_000.0]),
            "return_count_values": np.array([1], dtype=np.int64),
            "return_sum_values": np.array([-0.1]),
            "return_sum_squares_values": np.array([0.01]),
            "max_drawdown_values": np.array([0.0]),
        }
        cases = (
            (continuity, "must match last_equity_values"),
            (running_max, "must be at least the first and last equity"),
            (positive_drawdown, "must be between -1.0 and 0.0"),
            (below_total_loss, "must be between -1.0 and 0.0"),
            (impossible_drawdown, "cannot exceed the drawdown implied"),
        )
        for state, expected_message in cases:
            with self.subTest(message=expected_message), self.assertRaisesRegex(ValueError, expected_message):
                self._run_minimal_grid(np.array([30.0]), state_overrides=state)

    def test_grid_rejects_incoherent_rollup_counts_and_moments(self) -> None:
        base = self._flat_history_state()
        cases = (
            (
                {**base, "return_count_values": np.array([1]), "positive_return_count_values": np.array([2])},
                "positive_return_count_values cannot exceed",
            ),
            (
                {**base, "return_count_values": np.array([1]), "excess_return_count_values": np.array([2])},
                "excess_return_count_values cannot exceed",
            ),
            (
                {**base, "return_sum_squares_values": np.array([-1.0])},
                "return_sum_squares_values must be non-negative",
            ),
            (
                {
                    **base,
                    "return_sum_values": np.array([0.1]),
                    "return_sum_squares_values": np.array([0.01]),
                },
                "Zero return count requires zero sum",
            ),
            (
                {
                    **base,
                    "excess_return_sum_values": np.array([0.1]),
                    "excess_return_sum_squares_values": np.array([0.01]),
                },
                "Zero excess_return count requires zero sum",
            ),
            (
                {
                    **base,
                    "return_count_values": np.array([2]),
                    "return_sum_values": np.array([2.0]),
                    "return_sum_squares_values": np.array([1.0]),
                },
                "return sum and square sum are mathematically inconsistent",
            ),
            (
                {
                    **base,
                    "return_count_values": np.array([2]),
                    "return_sum_values": np.array([-4.0]),
                    "return_sum_squares_values": np.array([8.0]),
                },
                "daily returns bounded below by -1.0",
            ),
            (
                {
                    **base,
                    "return_count_values": np.array([2]),
                    "return_sum_values": np.array([0.0]),
                    "return_sum_squares_values": np.array([8.0]),
                    "positive_return_count_values": np.array([1]),
                },
                "daily returns bounded below by -1.0",
            ),
            (
                {
                    **base,
                    "return_count_values": np.array([2]),
                    "excess_return_count_values": np.array([2]),
                    "excess_return_sum_values": np.array([2.0]),
                    "excess_return_sum_squares_values": np.array([1.0]),
                },
                "excess_return sum and square sum are mathematically inconsistent",
            ),
            (
                {
                    **base,
                    "return_count_values": np.array([2]),
                    "return_sum_values": np.array([0.0]),
                    "return_sum_squares_values": np.array([0.0002]),
                    "return_mean_values": np.array([0.0]),
                    "return_m2_values": np.array([1.0]),
                },
                "return centered moments are inconsistent",
            ),
        )
        for state, expected_message in cases:
            with self.subTest(message=expected_message), self.assertRaisesRegex(ValueError, expected_message):
                self._run_minimal_grid(np.array([30.0]), state_overrides=state)

    def test_grid_requires_a_one_return_rollup_to_match_its_equity_endpoints(self) -> None:
        prices = np.array([100.0, 100.0])
        consistent_state = {
            "start_indices": np.array([2], dtype=np.int64),
            "cash_values": np.array([200.0]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([200.0]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100.0]),
            "last_equity_values": np.array([200.0]),
            "running_max_equity_values": np.array([200.0]),
            "return_count_values": np.array([1], dtype=np.int64),
            "return_sum_values": np.array([1.0]),
            "return_sum_squares_values": np.array([1.0]),
            "return_mean_values": np.array([1.0]),
            "return_m2_values": np.array([0.0]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "excess_return_mean_values": np.array([0.0]),
            "excess_return_m2_values": np.array([0.0]),
            "positive_return_count_values": np.array([1], dtype=np.int64),
            "max_drawdown_values": np.array([0.0]),
        }
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.array([np.nan, np.nan]),
            risk_free_returns=np.array([np.nan, np.nan]),
            state_overrides=consistent_state,
        )
        self.assertFalse(result[0][0])

        contradictions = {
            "positive count": (
                {"positive_return_count_values": np.array([0], dtype=np.int64)},
                "one-return resumable rollup requires moments",
            ),
            "endpoint return": (
                {
                    "return_sum_values": np.array([0.5]),
                    "return_sum_squares_values": np.array([0.25]),
                    "return_mean_values": np.array([0.5]),
                },
                "one-return resumable rollup requires moments",
            ),
            "one-observation M2": (
                {
                    "return_sum_squares_values": np.array([1.25]),
                    "return_m2_values": np.array([0.25]),
                },
                "daily returns bounded below by -1.0",
            ),
        }
        for label, (overrides, expected_message) in contradictions.items():
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=np.array([np.nan, np.nan]),
                    risk_free_returns=np.array([np.nan, np.nan]),
                    state_overrides={**consistent_state, **overrides},
                )

    def test_grid_rejects_multi_return_moments_that_contradict_equity_endpoints(self) -> None:
        prices = np.array([100.0, 100.0, 100.0])
        daily_return = 1.1 - 1.0
        return_sum = 0.0
        return_sum_squares = 0.0
        for _ in range(2):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return
        two_return_state = {
            "start_indices": np.array([3], dtype=np.int64),
            "cash_values": np.array([121.0]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([121.0]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100.0]),
            "last_equity_values": np.array([121.0]),
            "running_max_equity_values": np.array([121.0]),
            "return_count_values": np.array([2], dtype=np.int64),
            "return_sum_values": np.array([return_sum]),
            "return_sum_squares_values": np.array([return_sum_squares]),
            "return_mean_values": np.array([daily_return]),
            "return_m2_values": np.array([0.0]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "excess_return_mean_values": np.array([0.0]),
            "excess_return_m2_values": np.array([0.0]),
            "positive_return_count_values": np.array([2], dtype=np.int64),
            "max_drawdown_values": np.array([0.0]),
        }
        self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.full(3, 50.0),
            risk_free_returns=np.zeros(3),
            state_overrides=two_return_state,
        )

        inconsistent = {
            **two_return_state,
            "cash_values": np.array([120.0]),
            "prev_equity_values": np.array([120.0]),
            "last_equity_values": np.array([120.0]),
            "running_max_equity_values": np.array([120.0]),
        }
        with self.assertRaisesRegex(
            ValueError,
            "endpoint growth bound",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.full(3, 50.0),
                risk_free_returns=np.zeros(3),
                state_overrides=inconsistent,
            )

        zero_moment_prices = np.array([100.0, 100.0, 100.0, 100.0])
        zero_moment_state = {
            **inconsistent,
            "start_indices": np.array([4], dtype=np.int64),
            "return_count_values": np.array([3], dtype=np.int64),
            "return_sum_values": np.array([0.0]),
            "return_sum_squares_values": np.array([0.0]),
            "return_mean_values": np.array([0.0]),
            "return_m2_values": np.array([0.0]),
            "positive_return_count_values": np.array([0], dtype=np.int64),
        }
        with self.assertRaisesRegex(
            ValueError,
            "inconsistent with fully determined strategy returns",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=zero_moment_prices,
                high_prices=zero_moment_prices,
                close_prices=zero_moment_prices,
                rsi_values=np.full(4, 50.0),
                risk_free_returns=np.zeros(4),
                state_overrides=zero_moment_state,
            )

    def test_grid_rejects_impossible_nonconstant_multi_return_endpoint_growth(self) -> None:
        returns = np.array([0.1, -0.1, 0.0])
        state = {
            **self._flat_history_state(equity=200.0),
            "start_indices": np.array([1], dtype=np.int64),
            "history_prefix_observation_counts": np.array([4], dtype=np.int64),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100.0]),
            "running_max_equity_values": np.array([200.0]),
            "return_count_values": np.array([3], dtype=np.int64),
            "return_sum_values": np.array([float(returns.sum())]),
            "return_sum_squares_values": np.array([float(np.sum(returns * returns))]),
            "return_mean_values": np.array([float(returns.mean())]),
            "return_m2_values": np.array([float(np.sum((returns - returns.mean()) ** 2))]),
            "positive_return_count_values": np.array([1], dtype=np.int64),
            "max_drawdown_values": np.array([-0.1]),
        }

        # Positive gross returns with a sum of three have a product of at most
        # one, so these moments cannot grow the account from 100 to 200.
        with self.assertRaisesRegex(ValueError, "endpoint growth bound"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=state)

        positive_returns = np.array([2.0, 1.0, 0.0])
        positive_sum = float(positive_returns.sum())
        positive_squares = float(np.sum(positive_returns * positive_returns))
        positive_mean = float(positive_returns.mean())
        positive_m2 = float(np.sum((positive_returns - positive_mean) ** 2))
        impossible_nonzero_sum_state = {
            **state,
            "cash_values": np.array([1_000.0]),
            "prev_equity_values": np.array([1_000.0]),
            "last_equity_values": np.array([1_000.0]),
            "running_max_equity_values": np.array([1_000.0]),
            "return_sum_values": np.array([positive_sum]),
            "return_sum_squares_values": np.array([positive_squares]),
            "return_mean_values": np.array([positive_mean]),
            "return_m2_values": np.array([positive_m2]),
            "positive_return_count_values": np.array([2], dtype=np.int64),
            "max_drawdown_values": np.array([0.0]),
        }

        # A sum of three permits a gross-product maximum of 2**3, not ten.
        with self.assertRaisesRegex(ValueError, "endpoint growth bound"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=impossible_nonzero_sum_state)
        self.assertFalse(
            _return_rollup_equity_endpoints_are_consistent(
                count=3,
                first_equity=100.0,
                last_equity=1_000.0,
                return_sum=positive_sum,
                return_sum_squares=positive_squares,
                return_mean=positive_mean,
                return_m2=positive_m2,
            )
        )
        self.assertFalse(
            _fully_determined_return_rollup_is_consistent(
                count=3,
                first_equity=100.0,
                last_equity=1_000.0,
                running_max_equity=1_000.0,
                return_sum=positive_sum,
                return_sum_squares=positive_squares,
                return_mean=positive_mean,
                return_m2=positive_m2,
                positive_return_count=2,
                max_drawdown=0.0,
            )
        )

        impossible_amgm_equality_state = {
            **state,
            "cash_values": np.array([100.0]),
            "prev_equity_values": np.array([100.0]),
            "last_equity_values": np.array([100.0]),
            "running_max_equity_values": np.array([100.0]),
        }

        # A geometric mean can equal the arithmetic mean only when variance
        # is zero. These nonzero moments therefore cannot leave equity at 100.
        with self.assertRaisesRegex(ValueError, "endpoint growth bound"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=impossible_amgm_equality_state)
        self.assertFalse(
            _return_rollup_equity_endpoints_are_consistent(
                count=3,
                first_equity=100.0,
                last_equity=100.0,
                return_sum=float(returns.sum()),
                return_sum_squares=float(np.sum(returns * returns)),
                return_mean=float(returns.mean()),
                return_m2=float(np.sum((returns - returns.mean()) ** 2)),
            )
        )
        self.assertFalse(
            _fully_determined_return_rollup_is_consistent(
                count=3,
                first_equity=100.0,
                last_equity=100.0,
                running_max_equity=100.0,
                return_sum=float(returns.sum()),
                return_sum_squares=float(np.sum(returns * returns)),
                return_mean=float(returns.mean()),
                return_m2=float(np.sum((returns - returns.mean()) ** 2)),
                positive_return_count=1,
                max_drawdown=-0.1,
            )
        )

        impossible_loss_state = {
            **state,
            "cash_values": np.array([50.0]),
            "prev_equity_values": np.array([50.0]),
            "last_equity_values": np.array([50.0]),
            "running_max_equity_values": np.array([100.0]),
            "max_drawdown_values": np.array([-0.5]),
        }

        # These small centered moments put every return above about -11.6%,
        # so three such returns cannot halve the account.
        with self.assertRaisesRegex(ValueError, "endpoint growth bound"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=impossible_loss_state)
        self.assertFalse(
            _return_rollup_equity_endpoints_are_consistent(
                count=3,
                first_equity=100.0,
                last_equity=50.0,
                return_sum=float(returns.sum()),
                return_sum_squares=float(np.sum(returns * returns)),
                return_mean=float(returns.mean()),
                return_m2=float(np.sum((returns - returns.mean()) ** 2)),
            )
        )
        self.assertFalse(
            _fully_determined_return_rollup_is_consistent(
                count=3,
                first_equity=100.0,
                last_equity=50.0,
                running_max_equity=100.0,
                return_sum=float(returns.sum()),
                return_sum_squares=float(np.sum(returns * returns)),
                return_mean=float(returns.mean()),
                return_m2=float(np.sum((returns - returns.mean()) ** 2)),
                positive_return_count=1,
                max_drawdown=-0.5,
            )
        )

        impossible_near_product_state = {
            **state,
            "cash_values": np.array([98.0]),
            "prev_equity_values": np.array([98.0]),
            "last_equity_values": np.array([98.0]),
            "running_max_equity_values": np.array([100.0]),
        }

        # The tight fixed-moment lower bound is about 98.96, so the cruder
        # per-return minimum is not enough to make this endpoint feasible.
        with self.assertRaisesRegex(ValueError, "endpoint growth bound"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=impossible_near_product_state)
        self.assertFalse(
            _return_rollup_equity_endpoints_are_consistent(
                count=3,
                first_equity=100.0,
                last_equity=98.0,
                return_sum=float(returns.sum()),
                return_sum_squares=float(np.sum(returns * returns)),
                return_mean=float(returns.mean()),
                return_m2=float(np.sum((returns - returns.mean()) ** 2)),
            )
        )
        self.assertFalse(
            _fully_determined_return_rollup_is_consistent(
                count=3,
                first_equity=100.0,
                last_equity=98.0,
                running_max_equity=100.0,
                return_sum=float(returns.sum()),
                return_sum_squares=float(np.sum(returns * returns)),
                return_mean=float(returns.mean()),
                return_m2=float(np.sum((returns - returns.mean()) ** 2)),
                positive_return_count=1,
                max_drawdown=-0.1,
            )
        )

    def test_grid_rejects_excess_returns_that_require_invalid_risk_free_returns(self) -> None:
        excess_returns = np.array([2.0, 3.0])
        state = {
            **self._flat_history_state(equity=100.0),
            "start_indices": np.array([1], dtype=np.int64),
            "history_prefix_observation_counts": np.array([3], dtype=np.int64),
            "return_count_values": np.array([2], dtype=np.int64),
            "excess_return_count_values": np.array([2], dtype=np.int64),
            "excess_return_sum_values": np.array([float(excess_returns.sum())]),
            "excess_return_sum_squares_values": np.array([float(np.sum(excess_returns * excess_returns))]),
            "excess_return_mean_values": np.array([float(excess_returns.mean())]),
            "excess_return_m2_values": np.array([float(np.sum((excess_returns - excess_returns.mean()) ** 2))]),
        }

        # A zero-return strategy can have excess return at most one because
        # every accepted risk-free return is strictly greater than -1.
        with self.assertRaisesRegex(ValueError, "risk-free returns greater than -1.0"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=state)

        centered_excess_returns = np.array([3.0, -3.0])
        centered_contradiction = {
            **state,
            "excess_return_sum_values": np.array([float(centered_excess_returns.sum())]),
            "excess_return_sum_squares_values": np.array(
                [float(np.sum(centered_excess_returns * centered_excess_returns))]
            ),
            "excess_return_mean_values": np.array([float(centered_excess_returns.mean())]),
            "excess_return_m2_values": np.array(
                [float(np.sum((centered_excess_returns - centered_excess_returns.mean()) ** 2))]
            ),
        }

        # The aligned means pass the sum bound, but the centered spread would
        # require one implied risk-free return to equal -3.
        with self.assertRaisesRegex(ValueError, "risk-free returns greater than -1.0"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=centered_contradiction)

        partial_centered_contradiction = {
            **centered_contradiction,
            "history_prefix_observation_counts": np.array([4], dtype=np.int64),
            "return_count_values": np.array([3], dtype=np.int64),
            "excess_return_count_values": np.array([2], dtype=np.int64),
        }

        # Missing benchmark observations do not make the excess spread
        # feasible: every strategy return is still zero, so each observed
        # excess return must remain below one.
        with self.assertRaisesRegex(ValueError, "risk-free returns greater than -1.0"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=partial_centered_contradiction)

        strategy_returns = np.array([2.0, 0.0, 0.0])
        pairing_state = {
            **self._flat_history_state(equity=300.0),
            "start_indices": np.array([1], dtype=np.int64),
            "history_prefix_observation_counts": np.array([4], dtype=np.int64),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100.0]),
            "return_count_values": np.array([3], dtype=np.int64),
            "return_sum_values": np.array([float(strategy_returns.sum())]),
            "return_sum_squares_values": np.array([float(np.sum(strategy_returns * strategy_returns))]),
            "return_mean_values": np.array([float(strategy_returns.mean())]),
            "return_m2_values": np.array([float(np.sum((strategy_returns - strategy_returns.mean()) ** 2))]),
            "positive_return_count_values": np.array([1], dtype=np.int64),
        }
        repeated_large_excess = np.array([1.5, 1.5])
        impossible_partial_pairing = {
            **pairing_state,
            "excess_return_count_values": np.array([2], dtype=np.int64),
            "excess_return_sum_values": np.array([float(repeated_large_excess.sum())]),
            "excess_return_sum_squares_values": np.array(
                [float(np.sum(repeated_large_excess * repeated_large_excess))]
            ),
            "excess_return_mean_values": np.array([float(repeated_large_excess.mean())]),
            "excess_return_m2_values": np.array(
                [float(np.sum((repeated_large_excess - repeated_large_excess.mean()) ** 2))]
            ),
        }

        # Both observed excess values exceed one and therefore require two
        # positive strategy returns, while the strategy moments admit only one.
        with self.assertRaisesRegex(ValueError, "risk-free returns greater than -1.0"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=impossible_partial_pairing)

        feasible_full_excess = np.array([2.0, 0.5, 0.5])
        feasible_full_pairing = {
            **pairing_state,
            "excess_return_count_values": np.array([3], dtype=np.int64),
            "excess_return_sum_values": np.array([float(feasible_full_excess.sum())]),
            "excess_return_sum_squares_values": np.array([float(np.sum(feasible_full_excess * feasible_full_excess))]),
            "excess_return_mean_values": np.array([float(feasible_full_excess.mean())]),
            "excess_return_m2_values": np.array(
                [float(np.sum((feasible_full_excess - feasible_full_excess.mean()) ** 2))]
            ),
        }

        # These are also the aggregate moments of [1.5, 1.5, 0.0], but the
        # witness above pairs with strategy [2, 0, 0] and valid risk-free
        # returns [0, -0.5, -0.5]. Aggregate validation must preserve it.
        result = self._run_minimal_grid(np.array([30.0]), state_overrides=feasible_full_pairing)
        self.assertFalse(result[0][0])

    def test_grid_validates_constant_multi_return_rollup_path(self) -> None:
        prices = np.full(4, 100.0)
        daily_return = 0.25
        return_sum = 0.0
        return_sum_squares = 0.0
        for _ in range(3):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return
        consistent_state = {
            "start_indices": np.array([4], dtype=np.int64),
            "cash_values": np.array([195.3125]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([195.3125]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100.0]),
            "last_equity_values": np.array([195.3125]),
            "running_max_equity_values": np.array([195.3125]),
            "return_count_values": np.array([3], dtype=np.int64),
            "return_sum_values": np.array([return_sum]),
            "return_sum_squares_values": np.array([return_sum_squares]),
            "return_mean_values": np.array([daily_return]),
            "return_m2_values": np.array([0.0]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "excess_return_mean_values": np.array([0.0]),
            "excess_return_m2_values": np.array([0.0]),
            "positive_return_count_values": np.array([3], dtype=np.int64),
            "max_drawdown_values": np.array([0.0]),
        }
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.full(4, 50.0),
            risk_free_returns=np.zeros(4),
            state_overrides=consistent_state,
        )
        self.assertFalse(result[0][0])

        contradictions = (
            (
                {
                    "cash_values": np.array([156.25]),
                    "prev_equity_values": np.array([156.25]),
                    "last_equity_values": np.array([156.25]),
                    "running_max_equity_values": np.array([156.25]),
                },
                "inconsistent with fully determined strategy returns",
            ),
            (
                {"positive_return_count_values": np.array([2], dtype=np.int64)},
                "inconsistent with fully determined strategy returns",
            ),
            (
                {"max_drawdown_values": np.array([-0.5])},
                "inconsistent with fully determined strategy returns",
            ),
            (
                {"return_sum_values": np.array([np.nextafter(return_sum, np.inf)])},
                "inconsistent with fully determined strategy returns",
            ),
            (
                {"return_sum_values": np.array([return_sum - 1e-10])},
                "endpoint growth bound",
            ),
        )
        for overrides, expected_message in contradictions:
            with (
                self.subTest(expected_message=expected_message),
                self.assertRaisesRegex(
                    ValueError,
                    expected_message,
                ),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=np.full(4, 50.0),
                    risk_free_returns=np.zeros(4),
                    state_overrides={**consistent_state, **overrides},
                )

    def test_native_constant_rollup_requires_reachable_float64_chain(self) -> None:
        prices = np.ones(4)

        def state(
            *,
            first_equity: float,
            last_equity: float,
            running_max_equity: float,
            return_sum: float,
            return_sum_squares: float,
            return_mean: float,
            positive_return_count: int,
            max_drawdown: float,
        ) -> dict[str, np.ndarray]:
            return {
                "start_indices": np.array([4], dtype=np.int64),
                "cash_values": np.array([last_equity]),
                "share_values": np.array([0.0]),
                "in_position_values": np.array([False]),
                "entry_price_values": np.array([np.nan]),
                "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
                "prev_equity_values": np.array([last_equity]),
                "trades_executed_values": np.array([2], dtype=np.int64),
                "first_equity_values": np.array([first_equity]),
                "last_equity_values": np.array([last_equity]),
                "running_max_equity_values": np.array([running_max_equity]),
                "return_count_values": np.array([3], dtype=np.int64),
                "return_sum_values": np.array([return_sum]),
                "return_sum_squares_values": np.array([return_sum_squares]),
                "return_mean_values": np.array([return_mean]),
                "return_m2_values": np.array([0.0]),
                "excess_return_count_values": np.array([0], dtype=np.int64),
                "excess_return_sum_values": np.array([0.0]),
                "excess_return_sum_squares_values": np.array([0.0]),
                "excess_return_mean_values": np.array([0.0]),
                "excess_return_m2_values": np.array([0.0]),
                "positive_return_count_values": np.array(
                    [positive_return_count],
                    dtype=np.int64,
                ),
                "max_drawdown_values": np.array([max_drawdown]),
            }

        impossible_states = (
            state(
                first_equity=1.0,
                last_equity=0.9999999999999992,
                running_max_equity=1.0,
                return_sum=0.0,
                return_sum_squares=0.0,
                return_mean=0.0,
                positive_return_count=0,
                max_drawdown=-7.771561172376096e-16,
            ),
            state(
                first_equity=1.8162543504383857e-40,
                last_equity=6.624686961029942e-40,
                running_max_equity=6.624686961029942e-40,
                return_sum=1.6179669377204529,
                return_sum_squares=0.8726056705188333,
                return_mean=0.5393223125734843,
                positive_return_count=3,
                max_drawdown=0.0,
            ),
        )
        for impossible_state in impossible_states:
            with (
                self.subTest(last_equity=impossible_state["last_equity_values"][0]),
                self.assertRaisesRegex(
                    ValueError,
                    "inconsistent with fully determined strategy returns",
                ),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=np.full(4, 50.0),
                    risk_free_returns=np.zeros(4),
                    state_overrides=impossible_state,
                )

        collision_paths = (
            (
                2.6697493204461804e44,
                2.31295618627179e40,
                2.003845933641212e36,
                1.736046082327176e32,
            ),
            (
                1.189694973410968e-98,
                2.35839118108161e-99,
                4.675155470361205e-100,
                9.267791894482987e-101,
            ),
            (
                8.570548066446746e-205,
                6.273839546575886e-207,
                4.592595753622304e-209,
                3.3618863854592997e-211,
            ),
        )
        for collision_equities in collision_paths:
            return_sum = 0.0
            return_sum_squares = 0.0
            return_mean = 0.0
            return_m2 = 0.0
            running_max_equity = collision_equities[0]
            max_drawdown = 0.0
            for count, (previous_equity, equity) in enumerate(
                zip(
                    collision_equities[:-1],
                    collision_equities[1:],
                    strict=True,
                ),
                start=1,
            ):
                daily_return = equity / previous_equity - 1.0
                return_sum += daily_return
                return_sum_squares += daily_return * daily_return
                delta = daily_return - return_mean
                return_mean += delta / count
                return_m2 += delta * (daily_return - return_mean)
                running_max_equity = max(running_max_equity, equity)
                max_drawdown = min(
                    max_drawdown,
                    equity / running_max_equity - 1.0,
                )
            self.assertEqual(return_m2, 0.0)

            result = self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.full(4, 50.0),
                risk_free_returns=np.zeros(4),
                state_overrides=state(
                    first_equity=collision_equities[0],
                    last_equity=collision_equities[-1],
                    running_max_equity=running_max_equity,
                    return_sum=return_sum,
                    return_sum_squares=return_sum_squares,
                    return_mean=return_mean,
                    positive_return_count=0,
                    max_drawdown=max_drawdown,
                ),
            )
            self.assertFalse(result[0][0])

    def test_native_constant_rollup_handles_chain_search_cap_soundly(self) -> None:
        return_count = 8_200
        daily_return = np.nextafter(1.0, np.inf) - 1.0
        return_sum = 0.0
        return_sum_squares = 0.0
        for _ in range(return_count):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return

        def endpoint_after_steps(first_equity: float, steps: int) -> float:
            equity = first_equity
            for _ in range(steps):
                equity = np.nextafter(equity, np.inf)
            return equity

        def state(first_equity: float, last_equity: float) -> dict[str, np.ndarray]:
            return {
                "start_indices": np.array([1], dtype=np.int64),
                "cash_values": np.array([last_equity]),
                "share_values": np.array([0.0]),
                "in_position_values": np.array([False]),
                "entry_price_values": np.array([np.nan]),
                "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
                "prev_equity_values": np.array([last_equity]),
                "trades_executed_values": np.array([2], dtype=np.int64),
                "first_equity_values": np.array([first_equity]),
                "last_equity_values": np.array([last_equity]),
                "running_max_equity_values": np.array([last_equity]),
                "return_count_values": np.array([return_count], dtype=np.int64),
                "return_sum_values": np.array([return_sum]),
                "return_sum_squares_values": np.array([return_sum_squares]),
                "return_mean_values": np.array([daily_return]),
                "return_m2_values": np.array([0.0]),
                "excess_return_count_values": np.array([0], dtype=np.int64),
                "excess_return_sum_values": np.array([0.0]),
                "excess_return_sum_squares_values": np.array([0.0]),
                "excess_return_mean_values": np.array([0.0]),
                "excess_return_m2_values": np.array([0.0]),
                "positive_return_count_values": np.array(
                    [return_count],
                    dtype=np.int64,
                ),
                "max_drawdown_values": np.array([0.0]),
                "history_prefix_observation_counts": np.array(
                    [return_count],
                    dtype=np.int64,
                ),
            }

        grid_arguments = {
            "open_prices": np.ones(1),
            "high_prices": np.ones(1),
            "close_prices": np.ones(1),
            "rsi_values": np.full(1, 50.0),
            "risk_free_returns": np.zeros(1),
        }
        # Each relevant quotient plateau has exactly the two successors one
        # and two float keys above its predecessor.  Both extremal paths are
        # reachable even though the lower one differs from nominal
        # multiplication and exhausts the general lattice-search budget.
        first_equity = 0.8261517637764438
        lower_last_equity = endpoint_after_steps(first_equity, return_count)
        upper_last_equity = endpoint_after_steps(
            first_equity,
            2 * return_count,
        )
        for last_equity in (lower_last_equity, upper_last_equity):
            result = self._run_minimal_grid(
                np.array([30.0]),
                **grid_arguments,
                state_overrides=state(first_equity, last_equity),
            )
            self.assertFalse(result[0][0])

        # One key beyond the propagated upper envelope is provably
        # unreachable even though the approximate geometric check accepts it.
        invalid_last_equity = endpoint_after_steps(
            first_equity,
            2 * return_count + 1,
        )
        with self.assertRaisesRegex(
            ValueError,
            "inconsistent with fully determined strategy returns",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                **grid_arguments,
                state_overrides=state(first_equity, invalid_last_equity),
            )

    def test_native_zero_return_rollup_bypasses_pattern_work_cap_exactly(
        self,
    ) -> None:
        return_count = 16_385
        arguments = {
            "count": return_count,
            "first_equity": 1.0,
            "return_sum": 0.0,
            "return_sum_squares": 0.0,
            "return_mean": 0.0,
            "return_m2": 0.0,
            "positive_return_count": 0,
            "max_drawdown": 0.0,
            "native_centered_moments_available": True,
        }

        self.assertTrue(
            _fully_determined_return_rollup_is_consistent(
                **arguments,
                last_equity=1.0,
                running_max_equity=1.0,
            )
        )

        impossible_last_equity = np.nextafter(1.0, np.inf)
        self.assertFalse(
            _fully_determined_return_rollup_is_consistent(
                **arguments,
                last_equity=impossible_last_equity,
                running_max_equity=impossible_last_equity,
            )
        )

    def test_native_nonzero_constant_history_above_pattern_work_cap_resumes(
        self,
    ) -> None:
        return_count = 16_385
        daily_return = np.spacing(np.float64(1.0))
        equity = 1.0 + np.arange(return_count + 1, dtype=np.float64) * daily_return
        daily_returns = equity[1:] / equity[:-1] - 1.0
        self.assertTrue(np.all(daily_returns == daily_return))

        return_sum = 0.0
        return_sum_squares = 0.0
        return_mean = 0.0
        return_m2 = 0.0
        for observation_count, observed_return in enumerate(daily_returns, start=1):
            value = float(observed_return)
            return_sum += value
            return_sum_squares += value * value
            delta = value - return_mean
            return_mean += delta / observation_count
            return_m2 += delta * (value - return_mean)

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=equity,
            high_prices=equity,
            close_prices=equity,
            rsi_values=np.full(return_count + 1, 50.0),
            risk_free_returns=np.zeros(return_count + 1),
            state_overrides={
                "start_indices": np.array([return_count + 1], dtype=np.int64),
                "cash_values": np.array([0.0]),
                "share_values": np.array([1.0]),
                "in_position_values": np.array([True]),
                "entry_price_values": np.array([1.0]),
                "prev_equity_values": np.array([equity[-1]]),
                "trades_executed_values": np.array([1], dtype=np.int64),
                "first_equity_values": np.array([equity[0]]),
                "last_equity_values": np.array([equity[-1]]),
                "running_max_equity_values": np.array([equity[-1]]),
                "return_count_values": np.array([return_count], dtype=np.int64),
                "return_sum_values": np.array([return_sum]),
                "return_sum_squares_values": np.array([return_sum_squares]),
                "return_mean_values": np.array([return_mean]),
                "return_m2_values": np.array([return_m2]),
                "excess_return_count_values": np.array([return_count], dtype=np.int64),
                "excess_return_sum_values": np.array([return_sum]),
                "excess_return_sum_squares_values": np.array([return_sum_squares]),
                "excess_return_mean_values": np.array([return_mean]),
                "excess_return_m2_values": np.array([return_m2]),
                "positive_return_count_values": np.array(
                    [return_count],
                    dtype=np.int64,
                ),
                "max_drawdown_values": np.array([0.0]),
                "resume_close_values": np.array([equity[-1]]),
            },
        )

        self.assertFalse(result[0][0])

    def test_native_arbitrary_constant_history_above_pattern_work_cap_resumes(
        self,
    ) -> None:
        return_count = 16_385
        gross_return = 1.000001
        equity = np.empty(return_count + 1, dtype=np.float64)
        equity[0] = 1.0
        for idx in range(return_count):
            equity[idx + 1] = equity[idx] * gross_return
        daily_returns = equity[1:] / equity[:-1] - 1.0
        self.assertEqual(len(np.unique(daily_returns)), 1)

        return_sum = 0.0
        return_sum_squares = 0.0
        return_mean = 0.0
        return_m2 = 0.0
        for observation_count, observed_return in enumerate(daily_returns, start=1):
            value = float(observed_return)
            return_sum += value
            return_sum_squares += value * value
            delta = value - return_mean
            return_mean += delta / observation_count
            return_m2 += delta * (value - return_mean)

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=equity,
            high_prices=equity,
            close_prices=equity,
            rsi_values=np.full(return_count + 1, 50.0),
            risk_free_returns=np.zeros(return_count + 1),
            state_overrides={
                "start_indices": np.array([return_count + 1], dtype=np.int64),
                "cash_values": np.array([0.0]),
                "share_values": np.array([1.0]),
                "in_position_values": np.array([True]),
                "entry_price_values": np.array([1.0]),
                "prev_equity_values": np.array([equity[-1]]),
                "trades_executed_values": np.array([1], dtype=np.int64),
                "first_equity_values": np.array([equity[0]]),
                "last_equity_values": np.array([equity[-1]]),
                "running_max_equity_values": np.array([equity[-1]]),
                "return_count_values": np.array([return_count], dtype=np.int64),
                "return_sum_values": np.array([return_sum]),
                "return_sum_squares_values": np.array([return_sum_squares]),
                "return_mean_values": np.array([return_mean]),
                "return_m2_values": np.array([return_m2]),
                "excess_return_count_values": np.array([return_count], dtype=np.int64),
                "excess_return_sum_values": np.array([return_sum]),
                "excess_return_sum_squares_values": np.array([return_sum_squares]),
                "excess_return_mean_values": np.array([return_mean]),
                "excess_return_m2_values": np.array([return_m2]),
                "positive_return_count_values": np.array(
                    [return_count],
                    dtype=np.int64,
                ),
                "max_drawdown_values": np.array([0.0]),
                "resume_close_values": np.array([equity[-1]]),
            },
        )

        self.assertFalse(result[0][0])

    def test_repeated_float64_sum_matches_literal_addition_across_binades(
        self,
    ) -> None:
        regression_term = float.fromhex("0x1.f295b79c31ed3p-52")
        positive_terms = [
            regression_term,
            *(math.ldexp(regression_term, exponent) for exponent in (-1_000, -500, 500, 900)),
            float.fromhex("0x0.0000000000001p-1022"),
            float.fromhex("0x0.fffffffffffffp-1022"),
            float.fromhex("0x1.0000000000001p+0"),
        ]
        for term in (*positive_terms, *(-value for value in positive_terms)):
            for count in (10, 53, 16_385):
                with self.subTest(term=term.hex(), count=count):
                    literal_sum = 0.0
                    for _ in range(count):
                        literal_sum += term
                    self.assertEqual(
                        _repeated_float64_sum(term, count),
                        literal_sum,
                    )

    def test_repeated_float64_sum_scales_through_ties_and_subnormals(self) -> None:
        huge_count = 1 << 60
        self.assertEqual(
            _repeated_float64_sum(1.0, huge_count),
            float(1 << 53),
        )
        self.assertEqual(
            _repeated_float64_sum(-1.0, huge_count),
            -float(1 << 53),
        )
        smallest_subnormal = float.fromhex("0x0.0000000000001p-1022")
        self.assertEqual(
            _repeated_float64_sum(smallest_subnormal, huge_count),
            math.ldexp(1.0, -1021),
        )

    def test_long_constant_return_rejects_unreachable_endpoint(
        self,
    ) -> None:
        return_count = 16_385
        gross_return = 1.000001
        daily_return = gross_return - 1.0
        nominal_last_equity = 1.0
        return_sum = 0.0
        return_sum_squares = 0.0
        for _ in range(return_count):
            nominal_last_equity *= gross_return
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return

        forged_last_equity = nominal_last_equity
        for _ in range(100):
            forged_last_equity = float(np.nextafter(forged_last_equity, np.inf))
        self.assertFalse(
            _fully_determined_return_rollup_is_consistent(
                count=return_count,
                first_equity=1.0,
                last_equity=forged_last_equity,
                running_max_equity=forged_last_equity,
                return_sum=return_sum,
                return_sum_squares=return_sum_squares,
                return_mean=daily_return,
                return_m2=0.0,
                positive_return_count=return_count,
                max_drawdown=0.0,
                native_centered_moments_available=True,
            )
        )

    def test_constant_return_rejects_unreachable_hole_between_endpoints(
        self,
    ) -> None:
        return_count = 6
        daily_return = 0.75
        return_sum = 0.0
        return_sum_squares = 0.0
        for _ in range(return_count):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return

        # Exact successor replay has two final branches, ending one key below
        # and one key above this value.  A min/max envelope would incorrectly
        # accept the unreachable key in the middle.
        first_equity = float.fromhex("0x1.1c356eb0df72ep+759")
        forged_last_equity = float.fromhex("0x1.fe34c7bad2d18p+763")
        self.assertFalse(
            _fully_determined_return_rollup_is_consistent(
                count=return_count,
                first_equity=first_equity,
                last_equity=forged_last_equity,
                running_max_equity=forged_last_equity,
                return_sum=return_sum,
                return_sum_squares=return_sum_squares,
                return_mean=daily_return,
                return_m2=0.0,
                positive_return_count=return_count,
                max_drawdown=0.0,
                native_centered_moments_available=True,
            )
        )

    def test_constant_return_accepts_reachable_branch_with_dead_envelope_edge(
        self,
    ) -> None:
        equities = [
            float.fromhex(value)
            for value in (
                "0x1.4c6b5a42dc645p-636",
                "0x1.f2a107644a967p-636",
                "0x1.75f8c58b37f0dp-635",
                "0x1.187a942869f4ap-634",
                "0x1.a4b7de3c9eeefp-634",
            )
        ]
        daily_return = 0.5
        return_sum = 0.0
        return_sum_squares = 0.0
        for _ in range(len(equities) - 1):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return

        # The upper edge of an intermediate successor set has no successor,
        # while this interior branch remains an exact native-return chain.
        self.assertTrue(
            _fully_determined_return_rollup_is_consistent(
                count=len(equities) - 1,
                first_equity=equities[0],
                last_equity=equities[-1],
                running_max_equity=equities[-1],
                return_sum=return_sum,
                return_sum_squares=return_sum_squares,
                return_mean=daily_return,
                return_m2=0.0,
                positive_return_count=len(equities) - 1,
                max_drawdown=0.0,
                native_centered_moments_available=True,
            )
        )

    def test_constant_return_history_above_supported_bound_fails_closed(
        self,
    ) -> None:
        return_count = 50_001
        daily_return = 1e-6
        return_sum = _repeated_float64_sum(daily_return, return_count)
        return_sum_squares = _repeated_float64_sum(
            daily_return * daily_return,
            return_count,
        )
        last_equity = float(np.float_power(1.0 + daily_return, return_count))

        self.assertFalse(
            _fully_determined_return_rollup_is_consistent(
                count=return_count,
                first_equity=1.0,
                last_equity=last_equity,
                running_max_equity=last_equity,
                return_sum=return_sum,
                return_sum_squares=return_sum_squares,
                return_mean=daily_return,
                return_m2=0.0,
                positive_return_count=return_count,
                max_drawdown=0.0,
                native_centered_moments_available=True,
            )
        )

    def test_native_nonzero_zero_m2_rollup_fails_closed_above_pattern_work_cap(
        self,
    ) -> None:
        return_count = 16_385
        daily_return = np.nextafter(1.0, np.inf) - 1.0
        return_sum = 0.0
        return_sum_squares = 0.0
        for _ in range(return_count):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return

        self.assertFalse(
            _fully_determined_return_rollup_is_consistent(
                count=return_count,
                first_equity=1.0,
                last_equity=42.0,
                running_max_equity=42.0,
                return_sum=return_sum + 1e-15,
                return_sum_squares=return_sum_squares,
                return_mean=daily_return,
                return_m2=0.0,
                positive_return_count=return_count,
                max_drawdown=0.0,
                native_centered_moments_available=True,
            )
        )

    def test_grid_does_not_treat_tiny_nonzero_m2_as_constant_returns(self) -> None:
        return_count = 1_000
        average_return = 0.01
        return_offset = np.sqrt(4.5e-12 / return_count)
        returns = np.where(
            np.arange(return_count) % 2 == 0,
            average_return - return_offset,
            average_return + return_offset,
        )
        return_sum = 0.0
        return_sum_squares = 0.0
        return_mean = 0.0
        return_m2 = 0.0
        equity = np.longdouble(100.0)
        for observation_count, daily_return in enumerate(returns, start=1):
            return_sum += float(daily_return)
            return_sum_squares += float(daily_return * daily_return)
            delta = float(daily_return) - return_mean
            return_mean += delta / observation_count
            return_m2 += delta * (float(daily_return) - return_mean)
            equity *= np.longdouble(1.0) + np.longdouble(daily_return)
        last_equity = float(equity)
        former_constant_tolerance = (
            256.0
            * np.finfo(np.float64).eps
            * return_count
            * max(return_sum_squares, return_count * return_mean * return_mean)
        )
        self.assertGreater(return_m2, 0.0)
        self.assertLess(return_m2, former_constant_tolerance)

        prices = np.full(return_count + 1, 100.0)
        state = {
            "start_indices": np.array([return_count + 1], dtype=np.int64),
            "cash_values": np.array([last_equity]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([last_equity]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100.0]),
            "last_equity_values": np.array([last_equity]),
            "running_max_equity_values": np.array([last_equity]),
            "return_count_values": np.array([return_count], dtype=np.int64),
            "return_sum_values": np.array([return_sum]),
            "return_sum_squares_values": np.array([return_sum_squares]),
            "return_mean_values": np.array([return_mean]),
            "return_m2_values": np.array([return_m2]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "excess_return_mean_values": np.array([0.0]),
            "excess_return_m2_values": np.array([0.0]),
            "positive_return_count_values": np.array([return_count], dtype=np.int64),
            "max_drawdown_values": np.array([0.0]),
        }
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.full(return_count + 1, 50.0),
            risk_free_returns=np.zeros(return_count + 1),
            state_overrides=state,
        )
        self.assertFalse(result[0][0])

    def test_grid_validates_two_return_multiset_path_metrics(self) -> None:
        prices = np.full(3, 100.0)
        consistent_state = {
            "start_indices": np.array([3], dtype=np.int64),
            "cash_values": np.array([108.0]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([108.0]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100.0]),
            "last_equity_values": np.array([108.0]),
            # The -10%, +20% ordering reaches a maximum of 108. The reverse
            # ordering is also feasible and reaches 120.
            "running_max_equity_values": np.array([108.0]),
            "return_count_values": np.array([2], dtype=np.int64),
            "return_sum_values": np.array([0.1]),
            "return_sum_squares_values": np.array([0.05]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "positive_return_count_values": np.array([1], dtype=np.int64),
            "max_drawdown_values": np.array([-0.1]),
        }
        for running_max in (108.0, 120.0):
            with self.subTest(running_max=running_max):
                result = self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=np.full(3, 50.0),
                    risk_free_returns=np.zeros(3),
                    state_overrides={
                        **consistent_state,
                        "running_max_equity_values": np.array([running_max]),
                    },
                )
                self.assertFalse(result[0][0])

        for overrides in (
            {"positive_return_count_values": np.array([2], dtype=np.int64)},
            {"max_drawdown_values": np.array([-0.9])},
            {"running_max_equity_values": np.array([110.0])},
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(
                    ValueError,
                    "inconsistent with fully determined strategy returns",
                ),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=np.full(3, 50.0),
                    risk_free_returns=np.zeros(3),
                    state_overrides={**consistent_state, **overrides},
                )

    def test_grid_allows_roundoff_ambiguous_zero_root_in_two_return_multiset(self) -> None:
        prices = np.full(3, 100.0)
        zero_and_negative_state = {
            "start_indices": np.array([3], dtype=np.int64),
            "cash_values": np.array([90.0]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([90.0]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100.0]),
            "last_equity_values": np.array([90.0]),
            "running_max_equity_values": np.array([100.0]),
            "return_count_values": np.array([2], dtype=np.int64),
            "return_sum_values": np.array([-0.1]),
            "return_sum_squares_values": np.array([0.01]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "positive_return_count_values": np.array([0], dtype=np.int64),
            "max_drawdown_values": np.array([-0.1]),
        }
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.full(3, 50.0),
            risk_free_returns=np.zeros(3),
            state_overrides=zero_and_negative_state,
        )
        self.assertFalse(result[0][0])

        clearly_positive_state = {
            **zero_and_negative_state,
            "cash_values": np.array([90.9]),
            "prev_equity_values": np.array([90.9]),
            "last_equity_values": np.array([90.9]),
            "return_sum_values": np.array([-0.09]),
            "return_sum_squares_values": np.array([0.0101]),
            "return_mean_values": np.array([-0.045]),
            "return_m2_values": np.array([0.00605]),
        }
        with self.assertRaisesRegex(
            ValueError,
            "inconsistent with fully determined strategy returns",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.full(3, 50.0),
                risk_free_returns=np.zeros(3),
                state_overrides=clearly_positive_state,
            )

    def test_grid_allows_ill_conditioned_two_return_endpoint_reconstruction(self) -> None:
        returns = (np.nextafter(-1.0, 0.0), 0.5)
        return_sum = 0.0
        return_sum_squares = 0.0
        return_mean = 0.0
        return_m2 = 0.0
        equity = 1e12
        running_max = equity
        max_drawdown = 0.0
        for observation_count, daily_return in enumerate(returns, start=1):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return
            delta = daily_return - return_mean
            return_mean += delta / observation_count
            return_m2 += delta * (daily_return - return_mean)
            equity *= 1.0 + daily_return
            running_max = max(running_max, equity)
            max_drawdown = min(max_drawdown, equity / running_max - 1.0)
        self.assertGreater(equity, 0.0)

        prices = np.full(3, 100.0)
        state = {
            "start_indices": np.array([3], dtype=np.int64),
            "cash_values": np.array([equity]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([equity]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([1e12]),
            "last_equity_values": np.array([equity]),
            "running_max_equity_values": np.array([running_max]),
            "return_count_values": np.array([2], dtype=np.int64),
            "return_sum_values": np.array([return_sum]),
            "return_sum_squares_values": np.array([return_sum_squares]),
            "return_mean_values": np.array([return_mean]),
            "return_m2_values": np.array([return_m2]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "excess_return_mean_values": np.array([0.0]),
            "excess_return_m2_values": np.array([0.0]),
            "positive_return_count_values": np.array([1], dtype=np.int64),
            "max_drawdown_values": np.array([max_drawdown]),
        }
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.full(3, 50.0),
            risk_free_returns=np.zeros(3),
            state_overrides=state,
        )
        self.assertFalse(result[0][0])

        with self.assertRaisesRegex(
            ValueError,
            "inconsistent with fully determined strategy returns",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.full(3, 50.0),
                risk_free_returns=np.zeros(3),
                state_overrides={
                    **state,
                    "positive_return_count_values": np.array([2], dtype=np.int64),
                },
            )

        # The unstable endpoint product does not obscure the path peak. The
        # two possible orderings peak at 1e12 or 1.5e12, never at 2e12.
        with self.assertRaisesRegex(
            ValueError,
            "inconsistent with fully determined strategy returns",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.full(3, 50.0),
                risk_free_returns=np.zeros(3),
                state_overrides={
                    **state,
                    "running_max_equity_values": np.array([2e12]),
                    "max_drawdown_values": np.array([equity / 2e12 - 1.0]),
                },
            )

        # Aggregate cancellation has a finite error budget; it cannot explain
        # an endpoint many orders of magnitude above the tiny valid result.
        fake_last_equity = 1.0
        with self.assertRaisesRegex(
            ValueError,
            "inconsistent with first and last equity",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.full(3, 50.0),
                risk_free_returns=np.zeros(3),
                state_overrides={
                    **state,
                    "cash_values": np.array([fake_last_equity]),
                    "prev_equity_values": np.array([fake_last_equity]),
                    "last_equity_values": np.array([fake_last_equity]),
                    "max_drawdown_values": np.array([fake_last_equity / running_max - 1.0]),
                },
            )

    def test_fresh_grid_recovers_ill_conditioned_drawdown_rebound_path(self) -> None:
        open_prices = np.array([1.0, 1.0, 1.01])
        high_prices = np.array([1.0, 1.0, 1.01])
        close_prices = np.array([1.0, 1e-16, 1.01])
        rsi_values = np.zeros(3)
        risk_free_returns = np.zeros(3)

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=open_prices,
            high_prices=high_prices,
            close_prices=close_prices,
            rsi_values=rsi_values,
            risk_free_returns=risk_free_returns,
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[11][0], 2)
        self.assertEqual(result[17][0], 1)
        self.assertAlmostEqual(result[9][0], 101_000.0)
        self.assertAlmostEqual(result[10][0], 101_000.0)
        self.assertLess(result[18][0], -0.999999999999)

        resumable_state = {
            "start_indices": np.array([3], dtype=np.int64),
            "cash_values": result[1],
            "share_values": result[2],
            "in_position_values": result[3],
            "entry_price_values": result[4],
            "pending_action_values": result[5],
            "prev_equity_values": result[6],
            "trades_executed_values": result[7],
            "first_equity_values": result[8],
            "last_equity_values": result[9],
            "running_max_equity_values": result[10],
            "return_count_values": result[11],
            "return_sum_values": result[12],
            "return_sum_squares_values": result[13],
            "excess_return_count_values": result[14],
            "excess_return_sum_values": result[15],
            "excess_return_sum_squares_values": result[16],
            "positive_return_count_values": result[17],
            "max_drawdown_values": result[18],
            "return_mean_values": result[19],
            "return_m2_values": result[20],
            "excess_return_mean_values": result[21],
            "excess_return_m2_values": result[22],
            "resume_close_values": np.array([close_prices[-1]]),
        }
        path_corruptions = (
            (
                {"positive_return_count_values": np.array([0], dtype=np.int64)},
                "risk-free returns greater than -1.0",
            ),
            ({"max_drawdown_values": np.array([0.0])}, "inconsistent with fully determined strategy returns"),
            (
                {
                    "running_max_equity_values": np.array([110_000.0]),
                    "max_drawdown_values": np.array([101_000.0 / 110_000.0 - 1.0]),
                },
                "inconsistent with fully determined strategy returns",
            ),
        )
        for corruption, expected_message in path_corruptions:
            with (
                self.subTest(corruption=corruption),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=open_prices,
                    high_prices=high_prices,
                    close_prices=close_prices,
                    rsi_values=rsi_values,
                    risk_free_returns=risk_free_returns,
                    state_overrides={**resumable_state, **corruption},
                )

        corrupt_last_equity = 1e30
        with self.assertRaisesRegex(
            ValueError,
            "endpoint growth bound",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=open_prices,
                high_prices=high_prices,
                close_prices=close_prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
                state_overrides={
                    **resumable_state,
                    "share_values": np.array([corrupt_last_equity / close_prices[-1]]),
                    "prev_equity_values": np.array([corrupt_last_equity]),
                    "last_equity_values": np.array([corrupt_last_equity]),
                    "running_max_equity_values": np.array([corrupt_last_equity]),
                },
            )

        # Raw float64 sums alone cannot distinguish this coordinated positive
        # path from the true loss/rebound. Native centered moments retain the
        # missing return's sign and must prevent the endpoint from validating
        # its own fabricated path fields.
        coordinated_corrupt_last = 2e21
        with self.assertRaisesRegex(
            ValueError,
            "inconsistent with fully determined strategy returns",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=open_prices,
                high_prices=high_prices,
                close_prices=close_prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
                state_overrides={
                    **resumable_state,
                    "share_values": np.array([coordinated_corrupt_last / close_prices[-1]]),
                    "prev_equity_values": np.array([coordinated_corrupt_last]),
                    "last_equity_values": np.array([coordinated_corrupt_last]),
                    "running_max_equity_values": np.array([coordinated_corrupt_last]),
                    "positive_return_count_values": np.array([2], dtype=np.int64),
                    "max_drawdown_values": np.array([0.0]),
                },
            )

    def test_fresh_grid_accepts_underflowed_gross_before_finite_rebound(self) -> None:
        first_equity = 1e20
        minimum_subnormal = np.nextafter(0.0, 1.0)
        last_equity = 5e-170
        open_prices = np.full(3, first_equity)
        high_prices = np.full(3, first_equity)
        close_prices = np.array([first_equity, minimum_subnormal, last_equity])
        rsi_values = np.zeros(3)
        risk_free_returns = np.zeros(3)

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=open_prices,
            high_prices=high_prices,
            close_prices=close_prices,
            rsi_values=rsi_values,
            risk_free_returns=risk_free_returns,
            initial_capital=first_equity,
        )

        rebound_return = last_equity / minimum_subnormal - 1.0
        self.assertTrue(result[0][0])
        self.assertEqual(result[9][0], last_equity)
        self.assertEqual(result[10][0], first_equity)
        self.assertEqual(result[11][0], 2)
        self.assertEqual(result[12][0], rebound_return)
        self.assertEqual(result[13][0], rebound_return * rebound_return)
        self.assertEqual(result[17][0], 1)
        self.assertEqual(result[18][0], -1.0)

        resumable_state = {
            "start_indices": np.array([3], dtype=np.int64),
            "cash_values": result[1],
            "share_values": result[2],
            "in_position_values": result[3],
            "entry_price_values": result[4],
            "pending_action_values": result[5],
            "prev_equity_values": result[6],
            "trades_executed_values": result[7],
            "first_equity_values": result[8],
            "last_equity_values": result[9],
            "running_max_equity_values": result[10],
            "return_count_values": result[11],
            "return_sum_values": result[12],
            "return_sum_squares_values": result[13],
            "excess_return_count_values": result[14],
            "excess_return_sum_values": result[15],
            "excess_return_sum_squares_values": result[16],
            "positive_return_count_values": result[17],
            "max_drawdown_values": result[18],
            "return_mean_values": result[19],
            "return_m2_values": result[20],
            "excess_return_mean_values": result[21],
            "excess_return_m2_values": result[22],
            "resume_close_values": np.array([close_prices[-1]]),
        }
        self._run_minimal_grid(
            np.array([30.0]),
            open_prices=open_prices,
            high_prices=high_prices,
            close_prices=close_prices,
            rsi_values=rsi_values,
            risk_free_returns=risk_free_returns,
            initial_capital=first_equity,
            state_overrides=resumable_state,
        )

        path_corruptions = (
            (
                {"positive_return_count_values": np.array([0], dtype=np.int64)},
                "risk-free returns greater than -1.0",
            ),
            (
                {
                    "running_max_equity_values": np.array([2e20]),
                    "max_drawdown_values": np.array([-1.0]),
                },
                "inconsistent with fully determined strategy returns",
            ),
        )
        for corruption, expected_message in path_corruptions:
            with (
                self.subTest(corruption=corruption),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=open_prices,
                    high_prices=high_prices,
                    close_prices=close_prices,
                    rsi_values=rsi_values,
                    risk_free_returns=risk_free_returns,
                    initial_capital=first_equity,
                    state_overrides={**resumable_state, **corruption},
                )

        fake_last_equities = (
            1e-180,
            np.nextafter(last_equity, 0.0),
            np.nextafter(last_equity, np.inf),
        )
        for fake_last_equity in fake_last_equities:
            with (
                self.subTest(fake_last_equity=fake_last_equity),
                self.assertRaisesRegex(
                    ValueError,
                    "inconsistent with fully determined strategy returns",
                ),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=open_prices,
                    high_prices=high_prices,
                    close_prices=close_prices,
                    rsi_values=rsi_values,
                    risk_free_returns=risk_free_returns,
                    initial_capital=first_equity,
                    state_overrides={
                        **resumable_state,
                        "cash_values": np.array([fake_last_equity]),
                        "share_values": np.array([0.0]),
                        "in_position_values": np.array([False]),
                        "entry_price_values": np.array([np.nan]),
                        "prev_equity_values": np.array([fake_last_equity]),
                        "trades_executed_values": np.array([2], dtype=np.int64),
                        "last_equity_values": np.array([fake_last_equity]),
                        "resume_close_values": None,
                    },
                )

    def test_grid_accepts_exact_loss_boundary_paths(self) -> None:
        minimum_subnormal = np.nextafter(0.0, 1.0)
        equity_paths = (
            (1e20, minimum_subnormal, 5e-170),
            (1e20, 1.2e20, minimum_subnormal),
            (2.4442863080222966e70, 2.748020975149157e78, 3.41e-321),
            (4.4341431625104903e220, 1.03e-321, 1.10487200807e-313),
            (0.00010340693691205036, 11116.111827433026, 4.26e-321),
            (1e15, 0.9894333625645174e15, minimum_subnormal),
        )

        for first_equity, middle_equity, last_equity in equity_paths:
            with self.subTest(equities=(first_equity, middle_equity, last_equity)):
                close_prices = np.array([first_equity, middle_equity, last_equity])
                open_prices = np.ones(3)
                return_sum = 0.0
                return_sum_squares = 0.0
                return_mean = 0.0
                return_m2 = 0.0
                running_max_equity = first_equity
                max_drawdown = 0.0
                positive_return_count = 0
                expected_returns = (
                    middle_equity / first_equity - 1.0,
                    last_equity / middle_equity - 1.0,
                )
                for count, (equity, daily_return) in enumerate(
                    zip(
                        (middle_equity, last_equity),
                        expected_returns,
                        strict=True,
                    ),
                    start=1,
                ):
                    return_sum += daily_return
                    return_sum_squares += daily_return * daily_return
                    delta = daily_return - return_mean
                    return_mean += delta / count
                    return_m2 += delta * (daily_return - return_mean)
                    positive_return_count += daily_return > 0.0
                    running_max_equity = max(running_max_equity, equity)
                    max_drawdown = min(
                        max_drawdown,
                        equity / running_max_equity - 1.0,
                    )
                resumable_state = {
                    "start_indices": np.array([3], dtype=np.int64),
                    "cash_values": np.array([0.0]),
                    "share_values": np.array([1.0]),
                    "in_position_values": np.array([True]),
                    "entry_price_values": np.array([1.0]),
                    "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
                    "prev_equity_values": np.array([last_equity]),
                    "trades_executed_values": np.array([1], dtype=np.int64),
                    "first_equity_values": np.array([first_equity]),
                    "last_equity_values": np.array([last_equity]),
                    "running_max_equity_values": np.array([running_max_equity]),
                    "return_count_values": np.array([2], dtype=np.int64),
                    "return_sum_values": np.array([return_sum]),
                    "return_sum_squares_values": np.array([return_sum_squares]),
                    "excess_return_count_values": np.array([2], dtype=np.int64),
                    "excess_return_sum_values": np.array([return_sum]),
                    "excess_return_sum_squares_values": np.array([return_sum_squares]),
                    "positive_return_count_values": np.array(
                        [positive_return_count],
                        dtype=np.int64,
                    ),
                    "max_drawdown_values": np.array([max_drawdown]),
                    "return_mean_values": np.array([return_mean]),
                    "return_m2_values": np.array([return_m2]),
                    "excess_return_mean_values": np.array([return_mean]),
                    "excess_return_m2_values": np.array([return_m2]),
                    "resume_close_values": np.array([last_equity]),
                }
                result = self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=open_prices,
                    high_prices=np.maximum(open_prices, close_prices),
                    close_prices=close_prices,
                    rsi_values=np.zeros(3),
                    risk_free_returns=np.zeros(3),
                    profit_target_values=np.array([100.0]),
                    initial_capital=first_equity,
                    state_overrides=resumable_state,
                )

                self.assertEqual(result[9][0], last_equity)
                self.assertEqual(result[10][0], running_max_equity)
                self.assertEqual(result[11][0], 2)
                self.assertEqual(result[17][0], positive_return_count)

    def test_fresh_grid_preserves_positive_then_exact_loss_order(self) -> None:
        first_equity = 1e20
        middle_equity = 1.2e20
        last_equity = np.nextafter(0.0, 1.0)
        open_prices = np.full(3, first_equity)
        close_prices = np.array([first_equity, middle_equity, last_equity])
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=open_prices,
            high_prices=np.maximum(open_prices, close_prices),
            close_prices=close_prices,
            rsi_values=np.zeros(3),
            risk_free_returns=np.zeros(3),
            initial_capital=first_equity,
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[9][0], last_equity)
        self.assertEqual(result[10][0], middle_equity)
        self.assertEqual(result[17][0], 1)
        self.assertEqual(result[18][0], -1.0)

        resumable_state = {
            "start_indices": np.array([3], dtype=np.int64),
            "cash_values": result[1],
            "share_values": result[2],
            "in_position_values": result[3],
            "entry_price_values": result[4],
            "pending_action_values": result[5],
            "prev_equity_values": result[6],
            "trades_executed_values": result[7],
            "first_equity_values": result[8],
            "last_equity_values": result[9],
            "running_max_equity_values": np.array([first_equity]),
            "return_count_values": result[11],
            "return_sum_values": result[12],
            "return_sum_squares_values": result[13],
            "excess_return_count_values": result[14],
            "excess_return_sum_values": result[15],
            "excess_return_sum_squares_values": result[16],
            "positive_return_count_values": result[17],
            "max_drawdown_values": result[18],
            "return_mean_values": result[19],
            "return_m2_values": result[20],
            "excess_return_mean_values": result[21],
            "excess_return_m2_values": result[22],
            "resume_close_values": np.array([last_equity]),
        }
        with self.assertRaisesRegex(
            ValueError,
            "inconsistent with fully determined strategy returns",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=open_prices,
                high_prices=np.maximum(open_prices, close_prices),
                close_prices=close_prices,
                rsi_values=np.zeros(3),
                risk_free_returns=np.zeros(3),
                initial_capital=first_equity,
                state_overrides=resumable_state,
            )

    def test_fresh_grid_recovers_positive_penny_drawdown_rebound_path(self) -> None:
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=np.array([1.0, 1.0, 1.01]),
            high_prices=np.array([1.0, 1.0, 1.01]),
            close_prices=np.array([1.0, 0.001, 1.01]),
            rsi_values=np.zeros(3),
            risk_free_returns=np.zeros(3),
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[11][0], 2)
        self.assertEqual(result[17][0], 1)
        self.assertAlmostEqual(result[9][0], 101_000.0)
        self.assertAlmostEqual(result[10][0], 101_000.0)
        self.assertAlmostEqual(result[18][0], -0.999)

    def test_fresh_grid_searches_bounded_root_neighbors_for_native_moments(self) -> None:
        final_price = 1.6010234485844694
        drawdown_price = 0.03336696523864961
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=np.array([1.0, 1.0, final_price]),
            high_prices=np.array([1.0, 1.0, final_price]),
            close_prices=np.array([1.0, drawdown_price, final_price]),
            rsi_values=np.zeros(3),
            risk_free_returns=np.zeros(3),
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[11][0], 2)
        self.assertEqual(result[17][0], 1)
        self.assertAlmostEqual(result[9][0], 100_000.0 * final_price)
        self.assertAlmostEqual(result[10][0], 100_000.0 * final_price)
        self.assertAlmostEqual(result[18][0], drawdown_price - 1.0)

    def test_fresh_grid_uses_native_moments_for_nearly_equal_large_returns(self) -> None:
        final_price = 99.99999999999997
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=np.array([1.0, 1.0, 10.0]),
            high_prices=np.array([1.0, 10.0, final_price]),
            close_prices=np.array([1.0, 10.0, final_price]),
            rsi_values=np.zeros(3),
            risk_free_returns=np.zeros(3),
            profit_target_values=np.array([100.0]),
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[11][0], 2)
        self.assertEqual(result[17][0], 2)
        self.assertAlmostEqual(result[9][0], 100_000.0 * final_price)
        self.assertAlmostEqual(result[10][0], 100_000.0 * final_price)
        self.assertEqual(result[18][0], 0.0)
        self.assertAlmostEqual(result[19][0], 9.0)
        self.assertGreater(result[20][0], 0.0)

    def test_centered_roots_do_not_require_exact_endpoint_division(self) -> None:
        first_return = 8.966196663777822
        second_return = 8.966196663777845
        for ordered_returns in (
            (first_return, second_return),
            (second_return, first_return),
        ):
            with self.subTest(ordered_returns=ordered_returns):
                first_gross = 1.0 + ordered_returns[0]
                second_gross = 1.0 + ordered_returns[1]
                final_price = first_gross * second_gross
                result = self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=np.array([1.0, 1.0, first_gross]),
                    high_prices=np.array([1.0, first_gross, final_price]),
                    close_prices=np.array([1.0, first_gross, final_price]),
                    rsi_values=np.zeros(3),
                    risk_free_returns=np.zeros(3),
                    profit_target_values=np.array([100.0]),
                    initial_capital=1.0,
                )

                expected_mean = 0.0
                expected_m2 = 0.0
                for count, daily_return in enumerate(ordered_returns, start=1):
                    delta = daily_return - expected_mean
                    expected_mean += delta / count
                    expected_m2 += delta * (daily_return - expected_mean)
                self.assertTrue(result[0][0])
                self.assertEqual(result[11][0], 2)
                self.assertEqual(result[17][0], 2)
                self.assertEqual(result[9][0], final_price)
                self.assertEqual(result[18][0], 0.0)
                self.assertEqual(result[19][0], expected_mean)
                self.assertEqual(result[20][0], expected_m2)

    def test_tiny_nearly_equal_nonconstant_returns_have_bounded_reconstruction(self) -> None:
        unit = np.spacing(1.0)
        for ordered_returns in ((unit, 2.0 * unit), (2.0 * unit, unit)):
            with self.subTest(ordered_returns=ordered_returns):
                first_gross = 1.0 + ordered_returns[0]
                second_gross = 1.0 + ordered_returns[1]
                final_price = first_gross * second_gross
                result = self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=np.array([1.0, 1.0, first_gross]),
                    high_prices=np.array([1.0, first_gross, final_price]),
                    close_prices=np.array([1.0, first_gross, final_price]),
                    rsi_values=np.zeros(3),
                    risk_free_returns=np.zeros(3),
                    profit_target_values=np.array([100.0]),
                    initial_capital=1.0,
                )

                self.assertTrue(result[0][0])
                self.assertEqual(result[11][0], 2)
                self.assertEqual(result[17][0], 2)
                self.assertGreater(result[20][0], 0.0)
                self.assertEqual(result[18][0], 0.0)

    def test_subnormal_equities_reconstruct_two_float64_returns(self) -> None:
        smallest_equity = np.nextafter(0.0, 1.0)
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=np.ones(3),
            high_prices=np.array([1.0, 10.0, 11.0]),
            close_prices=np.array([1.0, 10.0, 11.0]),
            rsi_values=np.zeros(3),
            risk_free_returns=np.zeros(3),
            profit_target_values=np.array([100.0]),
            initial_capital=smallest_equity,
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[11][0], 2)
        self.assertEqual(result[17][0], 2)
        self.assertEqual(result[9][0], 11.0 * smallest_equity)
        self.assertEqual(result[18][0], 0.0)

    def test_native_welford_order_rejects_reversed_running_maximum(self) -> None:
        equities = (
            899_119_309_614.1146,
            104_567_814_545.92247,
            199_487_988_855.8207,
        )
        daily_returns = (
            equities[1] / equities[0] - 1.0,
            equities[2] / equities[1] - 1.0,
        )
        return_sum = 0.0
        return_sum_squares = 0.0
        return_mean = 0.0
        return_m2 = 0.0
        for count, daily_return in enumerate(daily_returns, start=1):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return
            delta = daily_return - return_mean
            return_mean += delta / count
            return_m2 += delta * (daily_return - return_mean)
        prices = np.full(3, 100.0)
        resumable_state = {
            "start_indices": np.array([3], dtype=np.int64),
            "cash_values": np.array([equities[2]]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([equities[2]]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([equities[0]]),
            "last_equity_values": np.array([equities[2]]),
            "running_max_equity_values": np.array([equities[0]]),
            "return_count_values": np.array([2], dtype=np.int64),
            "return_sum_values": np.array([return_sum]),
            "return_sum_squares_values": np.array([return_sum_squares]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "positive_return_count_values": np.array([1], dtype=np.int64),
            "max_drawdown_values": np.array([daily_returns[0]]),
            "return_mean_values": np.array([return_mean]),
            "return_m2_values": np.array([return_m2]),
            "excess_return_mean_values": np.array([0.0]),
            "excess_return_m2_values": np.array([0.0]),
            "resume_close_values": np.array([100.0]),
        }

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.full(3, 50.0),
            risk_free_returns=np.zeros(3),
            state_overrides=resumable_state,
        )
        self.assertFalse(result[0][0])

        with self.assertRaisesRegex(
            ValueError,
            "inconsistent with fully determined strategy returns",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.full(3, 50.0),
                risk_free_returns=np.zeros(3),
                state_overrides={
                    **resumable_state,
                    "running_max_equity_values": np.array([1_715_284_034_530.3728]),
                },
            )

    def test_constant_near_total_loss_uses_float64_gross_rounding_bucket(self) -> None:
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=np.array([1.0, 1.0, 1e-16]),
            high_prices=np.array([1.0, 1.0, 1e-16]),
            close_prices=np.array([1.0, 1e-16, 1e-32]),
            rsi_values=np.zeros(3),
            risk_free_returns=np.zeros(3),
            initial_capital=1.0,
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[11][0], 2)
        self.assertEqual(result[12][0], -1.9999999999999998)
        self.assertEqual(result[19][0], -0.9999999999999999)
        self.assertEqual(result[20][0], 0.0)
        self.assertEqual(result[18][0], -1.0)

    def test_zero_welford_m2_can_represent_adjacent_near_loss_returns(self) -> None:
        equities = (
            2.6697493204461804e44,
            2.31295618627179e40,
            2.003845933641212e36,
        )
        daily_returns = (
            equities[1] / equities[0] - 1.0,
            equities[2] / equities[1] - 1.0,
        )
        return_sum = 0.0
        return_sum_squares = 0.0
        return_mean = 0.0
        return_m2 = 0.0
        for count, daily_return in enumerate(daily_returns, start=1):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return
            delta = daily_return - return_mean
            return_mean += delta / count
            return_m2 += delta * (daily_return - return_mean)
        self.assertNotEqual(daily_returns[0], daily_returns[1])
        self.assertEqual(return_m2, 0.0)
        prices = np.full(3, 100.0)

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.full(3, 50.0),
            risk_free_returns=np.zeros(3),
            state_overrides={
                "start_indices": np.array([3], dtype=np.int64),
                "cash_values": np.array([equities[2]]),
                "share_values": np.array([0.0]),
                "in_position_values": np.array([False]),
                "entry_price_values": np.array([np.nan]),
                "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
                "prev_equity_values": np.array([equities[2]]),
                "trades_executed_values": np.array([2], dtype=np.int64),
                "first_equity_values": np.array([equities[0]]),
                "last_equity_values": np.array([equities[2]]),
                "running_max_equity_values": np.array([equities[0]]),
                "return_count_values": np.array([2], dtype=np.int64),
                "return_sum_values": np.array([return_sum]),
                "return_sum_squares_values": np.array([return_sum_squares]),
                "excess_return_count_values": np.array([0], dtype=np.int64),
                "excess_return_sum_values": np.array([0.0]),
                "excess_return_sum_squares_values": np.array([0.0]),
                "positive_return_count_values": np.array([0], dtype=np.int64),
                "max_drawdown_values": np.array([equities[2] / equities[0] - 1.0]),
                "return_mean_values": np.array([return_mean]),
                "return_m2_values": np.array([return_m2]),
                "excess_return_mean_values": np.array([0.0]),
                "excess_return_m2_values": np.array([0.0]),
                "resume_close_values": np.array([100.0]),
            },
        )

        self.assertFalse(result[0][0])

    def test_grid_rejects_rollups_larger_than_the_resume_prefix(self) -> None:
        prices = np.array([100.0, 100.0])
        for start_idx, return_count in ((0, 1), (1, 1), (2, 10)):
            state = {
                **self._flat_history_state(),
                "start_indices": np.array([start_idx], dtype=np.int64),
                "return_count_values": np.array([return_count], dtype=np.int64),
            }
            with (
                self.subTest(start_idx=start_idx, return_count=return_count),
                self.assertRaisesRegex(ValueError, "cannot exceed the history prefix"),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=np.array([50.0, 50.0]),
                    risk_free_returns=np.array([0.0, 0.0]),
                    state_overrides=state,
                )

    def test_grid_validates_history_prefix_observation_counts(self) -> None:
        for values, expected_message in (
            (np.array([-1], dtype=np.int64), "must be non-negative"),
            (np.array([0, 0], dtype=np.int64), "must contain 1 values"),
        ):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    state_overrides={"history_prefix_observation_counts": values},
                )

    def test_grid_accepts_rollup_within_the_resume_prefix(self) -> None:
        prices = np.array([100.0, 100.0, 100.0])
        state = {
            **self._flat_history_state(),
            "start_indices": np.array([2], dtype=np.int64),
            "return_count_values": np.array([1], dtype=np.int64),
        }

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.array([50.0, 50.0, 50.0]),
            risk_free_returns=np.array([0.0, 0.0, 0.0]),
            state_overrides=state,
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[11][0], 2)

    def test_grid_rejects_zero_trade_rollups_with_strategy_equity_changes(self) -> None:
        prices = np.array([100.0, 100.0, 100.0])
        base = {
            **self._flat_history_state(equity=100.0),
            "start_indices": np.array([3], dtype=np.int64),
            "return_count_values": np.array([2], dtype=np.int64),
        }
        cases = {
            "changed equity": {
                "first_equity_values": np.array([200.0]),
                "running_max_equity_values": np.array([200.0]),
                "return_sum_values": np.array([-0.5]),
                "return_sum_squares_values": np.array([0.25]),
                "max_drawdown_values": np.array([-0.5]),
            },
            "nonzero round-trip returns": {
                "return_sum_squares_values": np.array([0.02]),
                "positive_return_count_values": np.array([1], dtype=np.int64),
            },
        }

        for label, overrides in cases.items():
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(
                    ValueError,
                    "zero-trade resumable rollup requires constant strategy equity and zero strategy returns",
                ),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=np.array([50.0, 50.0, 50.0]),
                    risk_free_returns=np.array([0.0, 0.0, 0.0]),
                    state_overrides={**base, **overrides},
                )

    def test_grid_rejects_nonzero_empty_resume_moments_exactly(self) -> None:
        base = {
            **self._flat_history_state(equity=100.0),
            "start_indices": np.array([1], dtype=np.int64),
        }
        cases = (
            {"return_sum_values": np.array([1e-10])},
            {"return_sum_squares_values": np.array([1e-10])},
            {"excess_return_sum_values": np.array([1e-10])},
            {"excess_return_sum_squares_values": np.array([1e-10])},
            {
                "return_mean_values": np.array([1e-10]),
                "return_m2_values": np.array([0.0]),
            },
            {
                "return_mean_values": np.array([0.0]),
                "return_m2_values": np.array([1e-10]),
            },
            {
                "excess_return_mean_values": np.array([1e-10]),
                "excess_return_m2_values": np.array([0.0]),
            },
            {
                "excess_return_mean_values": np.array([0.0]),
                "excess_return_m2_values": np.array([1e-10]),
            },
            {"max_drawdown_values": np.array([-1e-10])},
        )

        for overrides in cases:
            with self.subTest(fields=tuple(overrides)), self.assertRaises(ValueError):
                self._run_minimal_grid(
                    np.array([30.0]),
                    rsi_values=np.array([50.0]),
                    state_overrides={
                        **base,
                        **overrides,
                    },
                )

    def test_grid_rejects_impossible_positive_return_count_for_positive_sum(self) -> None:
        returns = np.array([0.1, -0.1, 0.1])
        mean = float(np.mean(returns))
        state = {
            "start_indices": np.array([4], dtype=np.int64),
            "cash_values": np.array([108.9]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([108.9]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100.0]),
            "last_equity_values": np.array([108.9]),
            "running_max_equity_values": np.array([110.0]),
            "return_count_values": np.array([3], dtype=np.int64),
            "return_sum_values": np.array([float(np.sum(returns))]),
            "return_sum_squares_values": np.array([float(np.sum(returns * returns))]),
            "return_mean_values": np.array([mean]),
            "return_m2_values": np.array([float(np.sum((returns - mean) ** 2))]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "excess_return_mean_values": np.array([0.0]),
            "excess_return_m2_values": np.array([0.0]),
            "positive_return_count_values": np.array([0], dtype=np.int64),
            "max_drawdown_values": np.array([-0.1]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "positive_return_count_values is inconsistent",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=np.full(4, 100.0),
                high_prices=np.full(4, 100.0),
                close_prices=np.full(4, 100.0),
                rsi_values=np.full(4, 50.0),
                risk_free_returns=np.zeros(4),
                state_overrides=state,
            )

    def test_grid_rejects_sign_count_sum_below_nonpositive_return_bound(self) -> None:
        returns = np.array([-0.9, -0.8, -0.8])
        mean = float(np.mean(returns))
        state = {
            "start_indices": np.array([4], dtype=np.int64),
            "cash_values": np.array([400.0]),
            "share_values": np.array([0.0]),
            "in_position_values": np.array([False]),
            "entry_price_values": np.array([np.nan]),
            "pending_action_values": np.array([ACTION_NONE], dtype=np.int64),
            "prev_equity_values": np.array([400.0]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([100_000.0]),
            "last_equity_values": np.array([400.0]),
            "running_max_equity_values": np.array([100_000.0]),
            "return_count_values": np.array([3], dtype=np.int64),
            "return_sum_values": np.array([float(np.sum(returns))]),
            "return_sum_squares_values": np.array([float(np.sum(returns * returns))]),
            "return_mean_values": np.array([mean]),
            "return_m2_values": np.array([float(np.sum((returns - mean) ** 2))]),
            "excess_return_count_values": np.array([0], dtype=np.int64),
            "excess_return_sum_values": np.array([0.0]),
            "excess_return_sum_squares_values": np.array([0.0]),
            "excess_return_mean_values": np.array([0.0]),
            "excess_return_m2_values": np.array([0.0]),
            "positive_return_count_values": np.array([2], dtype=np.int64),
            "max_drawdown_values": np.array([-0.996]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "positive_return_count_values is inconsistent",
        ):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=np.full(4, 100.0),
                high_prices=np.full(4, 100.0),
                close_prices=np.full(4, 100.0),
                rsi_values=np.full(4, 50.0),
                risk_free_returns=np.zeros(4),
                state_overrides=state,
            )

    def test_grid_validates_held_equity_against_resume_close(self) -> None:
        state = {
            **self._held_history_state(),
            "resume_close_values": None,
        }

        with self.assertRaisesRegex(ValueError, "requires its last closing price"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=state)

        state["resume_close_values"] = np.array([90.0])
        with self.assertRaisesRegex(ValueError, "shares valued at its last closing price"):
            self._run_minimal_grid(np.array([30.0]), state_overrides=state)

        state["start_indices"] = np.array([2], dtype=np.int64)
        state["resume_close_values"] = None
        prices = np.array([100.0, 100.0])
        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.array([50.0, 50.0]),
            risk_free_returns=np.array([0.0, 0.0]),
            state_overrides=state,
        )
        self.assertFalse(result[0][0])

    def test_grid_resumes_centered_moments_without_raw_variance_subtraction(self) -> None:
        first_equity = 100_000.0
        middle_equity = first_equity * 1.001000000001
        last_equity = middle_equity * 1.000999999999
        prior_returns = np.array(
            [
                middle_equity / first_equity - 1.0,
                last_equity / middle_equity - 1.0,
            ]
        )
        prior_sum = 0.0
        prior_sum_squares = 0.0
        prior_mean = 0.0
        prior_m2 = 0.0
        for count, daily_return in enumerate(prior_returns, start=1):
            prior_sum += daily_return
            prior_sum_squares += daily_return * daily_return
            delta = daily_return - prior_mean
            prior_mean += delta / count
            prior_m2 += delta * (daily_return - prior_mean)

        state = {
            **self._flat_history_state(),
            "cash_values": np.array([last_equity]),
            "prev_equity_values": np.array([last_equity]),
            "trades_executed_values": np.array([2], dtype=np.int64),
            "first_equity_values": np.array([first_equity]),
            "last_equity_values": np.array([last_equity]),
            "running_max_equity_values": np.array([last_equity]),
            "return_count_values": np.array([2], dtype=np.int64),
            "return_sum_values": np.array([prior_sum]),
            "return_sum_squares_values": np.array([prior_sum_squares]),
            "return_mean_values": np.array([prior_mean]),
            "return_m2_values": np.array([prior_m2]),
            "excess_return_count_values": np.array([2], dtype=np.int64),
            "excess_return_sum_values": np.array([prior_sum]),
            "excess_return_sum_squares_values": np.array([prior_sum_squares]),
            "excess_return_mean_values": np.array([prior_mean]),
            "excess_return_m2_values": np.array([prior_m2]),
            "positive_return_count_values": np.array([2], dtype=np.int64),
            "history_prefix_observation_counts": np.array([3], dtype=np.int64),
        }

        result = self._run_minimal_grid(np.array([30.0]), state_overrides=state)

        delta = -prior_mean
        expected_mean = prior_mean + delta / 3.0
        expected_m2 = prior_m2 + delta * -expected_mean
        np.testing.assert_allclose(result[19][0], expected_mean, rtol=0.0, atol=1e-18)
        np.testing.assert_allclose(result[20][0], expected_m2, rtol=1e-15)
        np.testing.assert_allclose(result[21][0], expected_mean, rtol=0.0, atol=1e-18)
        np.testing.assert_allclose(result[22][0], expected_m2, rtol=1e-15)

    def test_buy_cost_is_included_in_position_sizing(self) -> None:
        result = run_single_equity_curve(
            open_prices=np.array([100.0, 100.0]),
            high_prices=np.array([100.0, 100.0]),
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

    def test_invalid_open_is_rejected_before_simulation(self) -> None:
        with self.assertRaisesRegex(ValueError, "open_prices must contain positive values"):
            run_single_equity_curve(
                open_prices=np.array([100.0, 0.0]),
                high_prices=np.array([100.0, 100.0]),
                close_prices=np.array([100.0, 100.0]),
                rsi_values=np.array([20.0, 50.0]),
                risk_free_returns=np.array([0.0, 0.0]),
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=100_000.0,
                trading_cost_rate=0.0003,
            )

    def test_upper_rsi_entry_rule_buys_only_on_high_rsi(self) -> None:
        lower_result = run_single_equity_curve(
            open_prices=np.array([100.0, 100.0]),
            high_prices=np.array([100.0, 100.0]),
            close_prices=np.array([100.0, 100.0]),
            rsi_values=np.array([80.0, 80.0]),
            risk_free_returns=np.array([0.0, 0.0]),
            buy_rsi=70.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0003,
        )
        upper_result = run_single_equity_curve(
            open_prices=np.array([100.0, 100.0]),
            high_prices=np.array([100.0, 100.0]),
            close_prices=np.array([100.0, 100.0]),
            rsi_values=np.array([80.0, 80.0]),
            risk_free_returns=np.array([0.0, 0.0]),
            buy_rsi=70.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0003,
            rsi_entry_rule=RSI_ENTRY_UPPER,
        )

        self.assertEqual(lower_result[5].tolist(), [ACTION_NONE, ACTION_NONE])
        self.assertEqual(upper_result[5].tolist(), [ACTION_BUY, ACTION_NONE])
        self.assertEqual(upper_result[-1], 1)

    def test_grid_summary_matches_single_curve_accounting(self) -> None:
        open_prices = np.array([100.0, 100.0, 105.0])
        high_prices = np.array([100.0, 100.0, 210.0])
        close_prices = np.array([100.0, 100.0, 105.0])
        rsi_values = np.array([20.0, 50.0, 50.0])
        risk_free_returns = np.zeros(3)
        single = run_single_equity_curve(
            open_prices,
            high_prices,
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
            high_prices,
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

    def test_resting_target_fills_on_intraday_high_at_limit(self) -> None:
        result = run_single_equity_curve(
            open_prices=np.array([100.0, 100.0, 90.0]),
            high_prices=np.array([100.0, 110.0, 90.0]),
            close_prices=np.array([100.0, 100.0, 90.0]),
            rsi_values=np.array([20.0, 50.0, 50.0]),
            risk_free_returns=np.zeros(3),
            buy_rsi=30.0,
            profit_target_multiple=1.10,
            initial_capital=100_000.0,
            trading_cost_rate=0.0,
        )

        self.assertAlmostEqual(result[7], 110_000.0)
        self.assertEqual(result[8], 0.0)
        self.assertEqual(result[9], 0)
        self.assertEqual(result[-1], 2)

    def test_resting_target_gets_favorable_gap_open_price(self) -> None:
        result = run_single_equity_curve(
            open_prices=np.array([100.0, 100.0, 120.0]),
            high_prices=np.array([100.0, 100.0, 120.0]),
            close_prices=np.array([100.0, 100.0, 120.0]),
            rsi_values=np.array([20.0, 50.0, 50.0]),
            risk_free_returns=np.zeros(3),
            buy_rsi=30.0,
            profit_target_multiple=1.10,
            initial_capital=100_000.0,
            trading_cost_rate=0.0,
        )

        self.assertAlmostEqual(result[7], 120_000.0)
        self.assertEqual(result[-1], 2)

    def test_backtest_target_rounding_matches_live_broker_pricing_at_float_edges(self) -> None:
        cases = [
            (0.56, 9.0),
            (18.6, 1.1),
            (0.14, 5.7),
            (573.7, 70.9),
            (0.1234, 1.001),
            (100.0, 1.1),
            (2.5200000000000005, 2.0),
            (0.3990000000000001, 2.0),
        ]

        for entry_price, profit_target_multiple in cases:
            with self.subTest(entry_price=entry_price, profit_target_multiple=profit_target_multiple):
                simulated = _target_limit_price(entry_price, profit_target_multiple)
                live = _target_sell_price(entry_price, profit_target_multiple)
                self.assertEqual(simulated, live)

        target_indices, offsets, override_rows, override_values = _target_price_overrides(
            np.array([entry for entry, _multiple in cases]),
            np.array([multiple for _entry, multiple in cases]),
        )
        for row_idx, (entry_price, profit_target_multiple) in enumerate(cases):
            with self.subTest(lookup_entry=entry_price, lookup_multiple=profit_target_multiple):
                self.assertEqual(
                    _target_price_with_override.py_func(
                        entry_price,
                        profit_target_multiple,
                        target_indices[row_idx],
                        row_idx,
                        offsets,
                        override_rows,
                        override_values,
                    ),
                    _target_sell_price(entry_price, profit_target_multiple),
                )

    def test_broker_prices_ignore_ambient_decimal_precision_rounding_and_traps(self) -> None:
        expected_sell = _target_sell_price(0.9999999, 1.000001)
        expected_buy = buffered_buy_limit_price(19.9998, 0.1)

        with localcontext() as decimal_context:
            decimal_context.prec = 6
            decimal_context.rounding = ROUND_DOWN
            decimal_context.traps[Inexact] = True
            observed_sell = _target_sell_price(0.9999999, 1.000001)
            observed_buy = buffered_buy_limit_price(19.9998, 0.1)

        self.assertEqual(expected_sell, 1.01)
        self.assertEqual(expected_buy, 19.99)
        self.assertEqual(observed_sell, expected_sell)
        self.assertEqual(observed_buy, expected_buy)

    def test_large_finite_target_prices_raise_clear_value_errors(self) -> None:
        for entry_price in (1e26, 1e27, 1e308):
            for implementation in (_target_limit_price, _target_sell_price):
                with (
                    self.subTest(entry_price=entry_price, implementation=implementation.__name__),
                    self.assertRaisesRegex(ValueError, "cannot be represented at Alpaca tick precision"),
                ):
                    implementation(entry_price, 2.0)

    def test_target_price_rejects_lossy_float_round_trip_below_decimal_tick(self) -> None:
        entry_price = 9e24
        profit_target_multiple = np.nextafter(1.0, 2.0)

        for implementation in (_target_limit_price, _target_sell_price):
            with (
                self.subTest(implementation=implementation.__name__),
                self.assertRaisesRegex(ValueError, "cannot be represented at Alpaca tick precision"),
            ):
                implementation(entry_price, profit_target_multiple)

    def test_optimized_apis_normalize_large_finite_price_rounding_failures(self) -> None:
        for entry_price in (1e26, 1e27, 1e308):
            prices = np.array([entry_price, entry_price])
            with (
                self.subTest(entry_price=entry_price, api="single"),
                self.assertRaisesRegex(ValueError, "cannot be represented at Alpaca tick precision"),
            ):
                run_single_equity_curve(
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=np.array([20.0, 50.0]),
                    risk_free_returns=np.array([0.0, 0.0]),
                    buy_rsi=30.0,
                    profit_target_multiple=2.0,
                    initial_capital=100_000.0,
                    trading_cost_rate=0.0,
                )

            with (
                self.subTest(entry_price=entry_price, api="grid"),
                self.assertRaisesRegex(ValueError, "cannot be represented at Alpaca tick precision"),
            ):
                self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=np.array([20.0, 50.0]),
                    risk_free_returns=np.array([0.0, 0.0]),
                )

    def test_optimized_apis_reject_prices_that_overflow_position_sizing(self) -> None:
        prices = np.array([5e-324, 5e-324])
        with self.assertRaisesRegex(ValueError, "non-finite shares"):
            run_single_equity_curve(
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.array([20.0, 50.0]),
                risk_free_returns=np.array([0.0, 0.0]),
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=100_000.0,
                trading_cost_rate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "non-finite shares"):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=np.array([20.0, 50.0]),
                risk_free_returns=np.array([0.0, 0.0]),
            )

    def test_optimized_apis_reject_executed_share_underflow_before_spending_cash(self) -> None:
        prices = np.array([1.0, 1e100])
        rsi_values = np.array([0.0, np.nan])
        risk_free_returns = np.zeros(2)

        with self.assertRaisesRegex(ValueError, "non-finite shares"):
            run_single_equity_curve(
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=1e-320,
                trading_cost_rate=0.9,
            )
        with self.assertRaisesRegex(ValueError, "non-finite shares"):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
                initial_capital=1e-320,
                trading_cost_rate=0.9,
            )

    def test_no_signal_path_ignores_unreachable_tiny_open_but_reachable_path_rejects_it(
        self,
    ) -> None:
        smallest_positive = np.nextafter(0.0, np.inf)
        prices = np.array([1.0, smallest_positive, 1.0])
        no_signal_rsi = np.full(3, 100.0)
        reachable_rsi = np.array([20.0, 100.0, 100.0])
        risk_free_returns = np.zeros(3)

        single = run_single_equity_curve(
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=no_signal_rsi,
            risk_free_returns=risk_free_returns,
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=1e308,
            trading_cost_rate=0.0,
        )
        grid = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=no_signal_rsi,
            risk_free_returns=risk_free_returns,
            initial_capital=1e308,
        )
        self.assertEqual(single[13], 0)
        self.assertEqual(single[7], 1e308)
        self.assertEqual(grid[7][0], 0)
        self.assertEqual(grid[1][0], 1e308)

        with self.assertRaisesRegex(ValueError, "non-finite shares"):
            run_single_equity_curve(
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=reachable_rsi,
                risk_free_returns=risk_free_returns,
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=1e308,
                trading_cost_rate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "non-finite shares"):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=reachable_rsi,
                risk_free_returns=risk_free_returns,
                initial_capital=1e308,
            )

    def test_consecutive_signals_do_not_make_a_tiny_open_reachable_while_held(self) -> None:
        prices = np.array([1.0, 1.0, 1e-300])
        rsi_values = np.full(3, 20.0)
        risk_free_returns = np.zeros(3)

        single = run_single_equity_curve(
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=rsi_values,
            risk_free_returns=risk_free_returns,
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=1e15,
            trading_cost_rate=0.0,
        )
        grid = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=rsi_values,
            risk_free_returns=risk_free_returns,
            initial_capital=1e15,
        )

        self.assertEqual(single[13], 1)
        self.assertEqual(grid[7][0], 1)
        self.assertEqual(single[8], 1e15)
        self.assertEqual(grid[2][0], single[8])
        self.assertEqual(grid[6][0], single[12])

    def test_resumed_held_position_ignores_consecutive_signals_at_extreme_opens(self) -> None:
        rsi_values = np.full(4, 20.0)
        risk_free_returns = np.zeros(4)
        state = {
            **self._held_history_state(
                shares=100_000.0,
                entry_price=1.0,
                equity=100_000.0,
            ),
            "start_indices": np.array([2], dtype=np.int64),
        }

        for final_open, expected_trades in ((1e-300, 1), (1e26, 2)):
            prices = np.array([1.0, 1.0, 1.0, final_open])
            with self.subTest(final_open=final_open):
                result = self._run_minimal_grid(
                    np.array([30.0]),
                    open_prices=prices,
                    high_prices=prices,
                    close_prices=prices,
                    rsi_values=rsi_values,
                    risk_free_returns=risk_free_returns,
                    state_overrides=state,
                )

                self.assertTrue(result[0][0])
                self.assertEqual(result[7][0], expected_trades)
                self.assertTrue(np.isfinite(result[1][0]))
                self.assertTrue(np.isfinite(result[2][0]))

    def test_no_signal_path_ignores_unreachable_target_overflow_but_reachable_path_rejects_it(
        self,
    ) -> None:
        prices = np.array([1.0, 1e26, 1.0])
        no_signal_rsi = np.full(3, 100.0)
        reachable_rsi = np.array([20.0, 100.0, 100.0])
        risk_free_returns = np.zeros(3)

        single = run_single_equity_curve(
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=no_signal_rsi,
            risk_free_returns=risk_free_returns,
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0,
        )
        grid = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=no_signal_rsi,
            risk_free_returns=risk_free_returns,
        )
        self.assertEqual(single[13], 0)
        self.assertEqual(single[7], 100_000.0)
        self.assertEqual(grid[7][0], 0)
        self.assertEqual(grid[1][0], 100_000.0)

        with self.assertRaisesRegex(ValueError, "cannot be represented at Alpaca tick precision"):
            run_single_equity_curve(
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=reachable_rsi,
                risk_free_returns=risk_free_returns,
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=100_000.0,
                trading_cost_rate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "cannot be represented at Alpaca tick precision"):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=reachable_rsi,
                risk_free_returns=risk_free_returns,
            )

    def test_target_overflow_candidate_is_ignored_when_an_existing_position_makes_it_unreachable(
        self,
    ) -> None:
        prices = np.array([1.0, 1.0, 1e26])
        rsi_values = np.array([20.0, 20.0, 100.0])
        risk_free_returns = np.zeros(3)

        single = run_single_equity_curve(
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=rsi_values,
            risk_free_returns=risk_free_returns,
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0,
        )
        grid = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=rsi_values,
            risk_free_returns=risk_free_returns,
        )

        self.assertEqual(single[13], 2)
        self.assertEqual(grid[7][0], 2)
        self.assertEqual(grid[1][0], single[7])
        self.assertGreater(single[7], 1e30)

    def test_grid_and_single_fail_closed_when_marked_equity_underflows_to_zero(self) -> None:
        open_prices = np.ones(4)
        close_prices = np.array([1.0, 1e-100, 1.0, 1.5])
        high_prices = np.maximum(open_prices, close_prices)
        rsi_values = np.array([0.0, np.nan, np.nan, np.nan])
        risk_free_returns = np.zeros(4)

        with self.assertRaisesRegex(ValueError, "Backtest produced non-positive equity"):
            run_single_equity_curve(
                open_prices=open_prices,
                high_prices=high_prices,
                close_prices=close_prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=1e-300,
                trading_cost_rate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "Backtest produced non-positive equity"):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=open_prices,
                high_prices=high_prices,
                close_prices=close_prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
                initial_capital=1e-300,
            )

    def test_grid_position_sizing_ignores_unreachable_tiny_open_prefix(self) -> None:
        smallest_positive = np.nextafter(0.0, np.inf)
        prices = np.array([smallest_positive, 1.0, 1.0])

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.array([20.0, 50.0, 50.0]),
            risk_free_returns=np.zeros(3),
            initial_capital=1e308,
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[2][0], 1e308)

        single = run_single_equity_curve(
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.array([20.0, 50.0, 50.0]),
            risk_free_returns=np.zeros(3),
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=1e308,
            trading_cost_rate=0.0,
        )
        self.assertEqual(single[8], 1e308)

    def test_grid_target_validation_ignores_unreachable_max_float_prefix(self) -> None:
        maximum_float = np.finfo(np.float64).max
        prices = np.array([maximum_float, 1.0, 1.0])

        result = self._run_minimal_grid(
            np.array([30.0]),
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.array([20.0, 50.0, 50.0]),
            risk_free_returns=np.zeros(3),
        )

        self.assertTrue(result[0][0])
        self.assertEqual(result[2][0], 100_000.0)

        single = run_single_equity_curve(
            open_prices=prices,
            high_prices=prices,
            close_prices=prices,
            rsi_values=np.array([20.0, 50.0, 50.0]),
            risk_free_returns=np.zeros(3),
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0,
        )
        self.assertEqual(single[8], 100_000.0)

    def test_grid_validates_pending_buy_on_current_resume_row(self) -> None:
        smallest_positive = np.nextafter(0.0, np.inf)
        state = {
            **self._flat_history_state(),
            "start_indices": np.array([1], dtype=np.int64),
            "pending_action_values": np.array([ACTION_BUY], dtype=np.int64),
        }
        with self.assertRaisesRegex(ValueError, "non-finite shares"):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=np.array([1.0, smallest_positive]),
                high_prices=np.array([1.0, smallest_positive]),
                close_prices=np.array([1.0, smallest_positive]),
                rsi_values=np.array([50.0, 50.0]),
                risk_free_returns=np.zeros(2),
                initial_capital=1e308,
                state_overrides=state,
            )

    def test_optimized_apis_reject_non_finite_kernel_account_state(self) -> None:
        open_prices = np.array([1.0, 1.0])
        high_prices = np.array([1.0, 2.0])
        close_prices = np.array([1.0, 1.0])
        rsi_values = np.array([20.0, 50.0])
        risk_free_returns = np.zeros(2)

        with self.assertRaisesRegex(ValueError, "Backtest produced non-finite"):
            run_single_equity_curve(
                open_prices=open_prices,
                high_prices=high_prices,
                close_prices=close_prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=1e308,
                trading_cost_rate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "Backtest produced non-finite"):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=open_prices,
                high_prices=high_prices,
                close_prices=close_prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
                initial_capital=1e308,
            )

    def test_optimized_apis_reject_unrepresentable_return_rollups(self) -> None:
        prices = np.array([100.0, 100.0])
        rsi_values = np.array([np.nan, np.nan])
        risk_free_returns = np.array([0.0, 1e308])

        with self.assertRaisesRegex(ValueError, "returns"):
            run_single_equity_curve(
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
                buy_rsi=30.0,
                profit_target_multiple=2.0,
                initial_capital=100_000.0,
                trading_cost_rate=0.0,
            )
        with self.assertRaisesRegex(ValueError, "return square sums"):
            self._run_minimal_grid(
                np.array([30.0]),
                open_prices=prices,
                high_prices=prices,
                close_prices=prices,
                rsi_values=rsi_values,
                risk_free_returns=risk_free_returns,
            )

    def test_single_curve_rejects_mismatched_arrays_before_numba(self) -> None:
        with self.assertRaisesRegex(ValueError, "high_prices must contain 2 values"):
            run_single_equity_curve(
                open_prices=np.array([100.0, 100.0]),
                high_prices=np.array([100.0]),
                close_prices=np.array([100.0, 100.0]),
                rsi_values=np.array([20.0, 50.0]),
                risk_free_returns=np.zeros(2),
                buy_rsi=30.0,
                profit_target_multiple=1.10,
                initial_capital=100_000.0,
                trading_cost_rate=0.0,
            )

    def test_single_curve_rejects_infinite_market_values_before_numba(self) -> None:
        with self.assertRaisesRegex(ValueError, "open_prices must contain only finite values"):
            run_single_equity_curve(
                open_prices=np.array([100.0, np.inf]),
                high_prices=np.array([100.0, 100.0]),
                close_prices=np.array([100.0, 100.0]),
                rsi_values=np.array([20.0, 50.0]),
                risk_free_returns=np.zeros(2),
                buy_rsi=30.0,
                profit_target_multiple=1.10,
                initial_capital=100_000.0,
                trading_cost_rate=0.0,
            )

    def test_single_curve_rejects_impossible_high_before_numba(self) -> None:
        with self.assertRaisesRegex(ValueError, "high_prices must be at least"):
            run_single_equity_curve(
                open_prices=np.array([100.0]),
                high_prices=np.array([99.0]),
                close_prices=np.array([100.0]),
                rsi_values=np.array([20.0]),
                risk_free_returns=np.zeros(1),
                buy_rsi=30.0,
                profit_target_multiple=1.10,
                initial_capital=100_000.0,
                trading_cost_rate=0.0,
            )


if __name__ == "__main__":
    unittest.main()
