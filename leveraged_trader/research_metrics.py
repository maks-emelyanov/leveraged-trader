"""Fresh-grid diagnostics for research, including censored holding periods."""

from __future__ import annotations

import numpy as np

from .backtest import _validate_risk_free_return_domain
from .optimized_backtest import (
    ACTION_BUY,
    ACTION_NONE,
    RSI_ENTRY_LOWER,
    RSI_ENTRY_UPPER,
    _finite_scalar,
    _float_array,
    _grid_entry_candidate_indices,
    _rsi_observation_array,
    _rsi_threshold_array,
    _run_single_equity_curve_kernel,
    _target_price_overrides,
    _target_price_with_override,
    _validate_rsi_entry_rule,
    _validate_simulation_prices,
    njit,
    prange,
    run_single_equity_curve,
)

_METRIC_NAMES = (
    "sharpe",
    "total_return",
    "completed_trades",
    "mean_holding_sessions",
    "max_holding_sessions",
    "exposure_fraction",
    "open_holding_sessions",
)


@njit(cache=True, parallel=True)
def _grid_research_metrics_kernel(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    risk_free_returns: np.ndarray,
    buy_rsi_values: np.ndarray,
    profit_target_values: np.ndarray,
    initial_capital: float,
    trading_cost_rate: float,
    rsi_entry_rule: int,
    target_indices: np.ndarray,
    override_offsets: np.ndarray,
    override_rows: np.ndarray,
    override_values: np.ndarray,
) -> tuple:
    row_count = len(open_prices)
    config_count = len(buy_rsi_values)
    metrics = np.zeros((config_count, 7), dtype=np.float64)
    invalid_results = np.zeros(config_count, dtype=np.bool_)
    for config_idx in prange(config_count):
        entry_targets = np.full(row_count, np.nan, dtype=np.float64)
        for row_idx in range(1, row_count):
            previous_rsi = rsi_values[row_idx - 1]
            threshold = buy_rsi_values[config_idx]
            entry_signal = previous_rsi >= threshold if rsi_entry_rule == RSI_ENTRY_UPPER else previous_rsi <= threshold
            if entry_signal:
                entry_targets[row_idx] = _target_price_with_override(
                    open_prices[row_idx],
                    profit_target_values[config_idx],
                    target_indices[config_idx],
                    row_idx,
                    override_offsets,
                    override_rows,
                    override_values,
                )

        result = _run_single_equity_curve_kernel(
            open_prices,
            high_prices,
            close_prices,
            rsi_values,
            risk_free_returns,
            entry_targets,
            buy_rsi_values[config_idx],
            initial_capital,
            trading_cost_rate,
            rsi_entry_rule,
        )
        invalid = (
            not np.isfinite(result[7])
            or not np.isfinite(result[8])
            or result[7] < 0.0
            or result[8] < 0.0
            or not np.isfinite(result[12])
            or result[12] <= 0.0
        )
        completed_trades = 0
        holding_sum = 0
        holding_max = 0
        exposed_sessions = 0
        entry_row = -1
        prior_trade_count = 0
        prior_position = False
        excess_count = 0
        excess_mean = 0.0
        excess_m2 = 0.0
        return_sum = 0.0
        return_square_sum = 0.0
        excess_sum = 0.0
        excess_square_sum = 0.0
        benchmark_complete = True

        for row_idx in range(row_count):
            equity = result[0][row_idx]
            daily_return = result[1][row_idx]
            invalid = invalid or not np.isfinite(equity) or equity <= 0.0 or not np.isfinite(daily_return)
            invalid = invalid or (result[4][row_idx] == ACTION_BUY and np.isnan(entry_targets[row_idx]))
            current_position = result[3][row_idx] != 0
            trade_delta = result[6][row_idx] - prior_trade_count
            if trade_delta > 0 and not prior_position:
                entry_row = row_idx
            if prior_position or current_position or trade_delta > 0:
                exposed_sessions += 1
            if trade_delta > 0 and (prior_position or trade_delta == 2):
                duration = row_idx - entry_row + 1
                holding_sum += duration
                holding_max = max(holding_max, duration)
                completed_trades += 1
                entry_row = -1
            prior_position = current_position
            prior_trade_count = result[6][row_idx]

            # The first mark has no previous observed session, matching the
            # public performance summary and persisted return rollups.
            if row_idx > 0:
                return_sum += daily_return
                return_square_sum += daily_return * daily_return
                if np.isnan(risk_free_returns[row_idx]):
                    benchmark_complete = False
                else:
                    excess = daily_return - risk_free_returns[row_idx]
                    excess_sum += excess
                    excess_square_sum += excess * excess
                    excess_count += 1
                    delta = excess - excess_mean
                    excess_mean += delta / excess_count
                    excess_m2 += delta * (excess - excess_mean)

        invalid = invalid or not (
            np.isfinite(return_sum)
            and np.isfinite(return_square_sum)
            and np.isfinite(excess_sum)
            and np.isfinite(excess_square_sum)
            and np.isfinite(excess_mean)
            and np.isfinite(excess_m2)
        )
        open_holding_sessions = row_count - entry_row if entry_row >= 0 else 0
        episode_count = completed_trades
        if open_holding_sessions > 0:
            holding_sum += open_holding_sessions
            holding_max = max(holding_max, open_holding_sessions)
            episode_count += 1
        sharpe = np.nan
        if benchmark_complete and excess_count >= 2 and excess_m2 > 0.0:
            sharpe = np.sqrt(252.0) * excess_mean / np.sqrt(excess_m2 / (excess_count - 1))
            invalid = invalid or not np.isfinite(sharpe)
        total_return = result[12] / initial_capital - 1.0
        invalid = invalid or not np.isfinite(total_return)
        metrics[config_idx, 0] = sharpe
        metrics[config_idx, 1] = total_return
        metrics[config_idx, 2] = completed_trades
        metrics[config_idx, 3] = holding_sum / episode_count if episode_count > 0 else np.nan
        metrics[config_idx, 4] = holding_max
        metrics[config_idx, 5] = exposed_sessions / row_count if row_count > 0 else 0.0
        metrics[config_idx, 6] = open_holding_sessions
        invalid_results[config_idx] = invalid
    return metrics, invalid_results


def grid_research_metrics(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    risk_free_returns: np.ndarray,
    buy_rsi_values: np.ndarray,
    profit_target_values: np.ndarray,
    initial_capital: float,
    trading_cost_rate: float,
    rsi_entry_rule: int = RSI_ENTRY_LOWER,
) -> dict[str, np.ndarray]:
    """Evaluate paired parameter arrays from cash using the production simulator.

    Holding periods count observed trading sessions inclusively: entry and
    exit on the same session count as one. A position still open at the final
    mark contributes its observed age to mean/max duration and exposure, but
    never to completed roundtrips. This is a censored age, not its eventual
    lifetime. With no entries, mean duration is NaN and max/open age are zero.
    Exposure is the fraction of all supplied sessions with any asset holding,
    including intraday roundtrips; an empty history has zero exposure/return.

    Sharpe uses sample standard deviation and 252 sessions per year, excludes
    the initial mark, and is NaN unless every subsequent session has a risk-free
    return. Targets retain the production Decimal tick corrections, calculated
    once per unique target and reachable row, rather than per configuration.
    """
    open_prices = _float_array("open_prices", open_prices)
    row_count = len(open_prices)
    high_prices = _float_array("high_prices", high_prices, length=row_count)
    close_prices = _float_array("close_prices", close_prices, length=row_count)
    _validate_simulation_prices(open_prices, high_prices, close_prices)
    rsi_values = _rsi_observation_array(rsi_values, length=row_count)
    risk_free_returns = _float_array("risk_free_returns", risk_free_returns, length=row_count, allow_nan=True)
    _validate_risk_free_return_domain(risk_free_returns)
    buy_rsi_values = _rsi_threshold_array("buy_rsi_values", buy_rsi_values)
    config_count = len(buy_rsi_values)
    profit_target_values = _float_array("profit_target_values", profit_target_values, length=config_count)
    if np.any((profit_target_values <= 1.0) | (profit_target_values > 100.0)):
        raise ValueError("profit_target_values must be greater than 1.0 and at most 100.0.")
    initial_capital = _finite_scalar("initial_capital", initial_capital)
    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive.")
    trading_cost_rate = _finite_scalar("trading_cost_rate", trading_cost_rate)
    if not 0.0 <= trading_cost_rate < 1.0:
        raise ValueError("trading_cost_rate must be non-negative and less than 1.0.")
    rsi_entry_rule = _validate_rsi_entry_rule(rsi_entry_rule)
    entry_candidates = _grid_entry_candidate_indices(
        rsi_values,
        buy_rsi_values,
        np.zeros(config_count, dtype=np.int64),
        np.zeros(config_count, dtype=np.bool_),
        np.full(config_count, ACTION_NONE, dtype=np.int64),
        rsi_entry_rule,
    )
    overrides = _target_price_overrides(
        open_prices,
        profit_target_values,
        entry_candidate_indices=entry_candidates,
    )
    metrics, invalid_results = _grid_research_metrics_kernel(
        open_prices,
        high_prices,
        close_prices,
        rsi_values,
        risk_free_returns,
        buy_rsi_values,
        profit_target_values,
        initial_capital,
        trading_cost_rate,
        rsi_entry_rule,
        *overrides,
    )
    if invalid_results.any():
        # Reuse the detailed production exceptions only for a failing path;
        # ordinary grids stay entirely in compiled code after shared preflight.
        config_idx = int(np.flatnonzero(invalid_results)[0])
        run_single_equity_curve(
            open_prices,
            high_prices,
            close_prices,
            rsi_values,
            risk_free_returns,
            float(buy_rsi_values[config_idx]),
            float(profit_target_values[config_idx]),
            initial_capital,
            trading_cost_rate,
            rsi_entry_rule,
        )
        raise ValueError(f"Backtest produced non-finite research metrics at config index {config_idx}.")
    result = {name: metrics[:, idx].copy() for idx, name in enumerate(_METRIC_NAMES)}
    for name in ("completed_trades", "max_holding_sessions", "open_holding_sessions"):
        result[name] = result[name].astype(np.int64)
    return result
