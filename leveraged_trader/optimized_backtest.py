from __future__ import annotations

from collections.abc import Callable

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without optional runtime dependency
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args: object, **kwargs: object) -> Callable:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func: Callable) -> Callable:
            return func

        return decorator


ACTION_NONE = 0
ACTION_BUY = 1
ACTION_SELL = 2

RSI_ENTRY_LOWER = 0
RSI_ENTRY_UPPER = 1


@njit(cache=True, parallel=True)
def run_grid_summary(
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    risk_free_returns: np.ndarray,
    buy_rsi_values: np.ndarray,
    profit_target_values: np.ndarray,
    start_indices: np.ndarray,
    cash_values: np.ndarray,
    share_values: np.ndarray,
    in_position_values: np.ndarray,
    entry_price_values: np.ndarray,
    pending_action_values: np.ndarray,
    prev_equity_values: np.ndarray,
    trades_executed_values: np.ndarray,
    first_equity_values: np.ndarray,
    last_equity_values: np.ndarray,
    running_max_equity_values: np.ndarray,
    return_count_values: np.ndarray,
    return_sum_values: np.ndarray,
    return_sum_squares_values: np.ndarray,
    excess_return_count_values: np.ndarray,
    excess_return_sum_values: np.ndarray,
    excess_return_sum_squares_values: np.ndarray,
    positive_return_count_values: np.ndarray,
    max_drawdown_values: np.ndarray,
    trading_cost_rate: float,
    rsi_entry_rule: int = RSI_ENTRY_LOWER,
) -> tuple:
    config_count = buy_rsi_values.shape[0]
    row_count = open_prices.shape[0]

    out_cash = cash_values.copy()
    out_shares = share_values.copy()
    out_in_position = in_position_values.copy()
    out_entry_price = entry_price_values.copy()
    out_pending_action = pending_action_values.copy()
    out_prev_equity = prev_equity_values.copy()
    out_trades_executed = trades_executed_values.copy()

    out_first_equity = first_equity_values.copy()
    out_last_equity = last_equity_values.copy()
    out_running_max_equity = running_max_equity_values.copy()
    out_return_count = return_count_values.copy()
    out_return_sum = return_sum_values.copy()
    out_return_sum_squares = return_sum_squares_values.copy()
    out_excess_return_count = excess_return_count_values.copy()
    out_excess_return_sum = excess_return_sum_values.copy()
    out_excess_return_sum_squares = excess_return_sum_squares_values.copy()
    out_positive_return_count = positive_return_count_values.copy()
    out_max_drawdown = max_drawdown_values.copy()
    updated = np.zeros(config_count, dtype=np.bool_)

    for config_idx in prange(config_count):
        start_idx = start_indices[config_idx]
        if start_idx >= row_count:
            continue

        updated[config_idx] = True
        cash = out_cash[config_idx]
        shares = out_shares[config_idx]
        in_position = out_in_position[config_idx]
        entry_price = out_entry_price[config_idx]
        pending_action = out_pending_action[config_idx]
        prev_equity = out_prev_equity[config_idx]
        trades_executed = out_trades_executed[config_idx]

        first_equity = out_first_equity[config_idx]
        last_equity = out_last_equity[config_idx]
        running_max_equity = out_running_max_equity[config_idx]
        return_count = out_return_count[config_idx]
        return_sum = out_return_sum[config_idx]
        return_sum_squares = out_return_sum_squares[config_idx]
        excess_return_count = out_excess_return_count[config_idx]
        excess_return_sum = out_excess_return_sum[config_idx]
        excess_return_sum_squares = out_excess_return_sum_squares[config_idx]
        positive_return_count = out_positive_return_count[config_idx]
        max_drawdown = out_max_drawdown[config_idx]

        buy_rsi = buy_rsi_values[config_idx]
        profit_target_multiple = profit_target_values[config_idx]

        for row_idx in range(start_idx, row_count):
            open_price = open_prices[row_idx]
            close_price = close_prices[row_idx]
            rsi = rsi_values[row_idx]

            deferred_action = ACTION_NONE
            if pending_action == ACTION_BUY and not in_position:
                cost_multiplier = 1.0 + trading_cost_rate
                if open_price > 0.0 and cash > 0.0 and cost_multiplier > 0.0:
                    turnover_notional = cash / cost_multiplier
                    shares = turnover_notional / open_price
                    cash -= turnover_notional * cost_multiplier
                    # Avoid a tiny negative balance caused by floating-point rounding.
                    cash = max(cash, 0.0)
                    in_position = shares > 0.0
                    entry_price = open_price if in_position else np.nan
                    if in_position:
                        trades_executed += 1
                else:
                    deferred_action = ACTION_BUY
            elif pending_action == ACTION_SELL and in_position:
                if open_price > 0.0:
                    turnover_notional = shares * open_price
                    cash += turnover_notional
                    cash -= turnover_notional * trading_cost_rate
                    shares = 0.0
                    in_position = False
                    entry_price = np.nan
                    trades_executed += 1
                else:
                    deferred_action = ACTION_SELL

            equity = cash + shares * close_price
            daily_return = equity / prev_equity - 1.0 if prev_equity > 0.0 else 0.0

            next_action = deferred_action
            if deferred_action == ACTION_NONE and not np.isnan(rsi):
                entry_signal = (
                    rsi >= buy_rsi
                    if rsi_entry_rule == RSI_ENTRY_UPPER
                    else rsi <= buy_rsi
                )
                if (not in_position) and entry_signal:
                    next_action = ACTION_BUY
                elif (
                    in_position
                    and not np.isnan(entry_price)
                    and entry_price > 0.0
                    and close_price / entry_price >= profit_target_multiple
                ):
                    next_action = ACTION_SELL

            if np.isnan(first_equity):
                first_equity = equity
                last_equity = equity
                running_max_equity = equity
                max_drawdown = 0.0
            else:
                return_count += 1
                return_sum += daily_return
                return_sum_squares += daily_return * daily_return
                if daily_return > 0.0:
                    positive_return_count += 1

                risk_free_return = risk_free_returns[row_idx]
                if not np.isnan(risk_free_return):
                    excess_return = daily_return - risk_free_return
                    excess_return_count += 1
                    excess_return_sum += excess_return
                    excess_return_sum_squares += excess_return * excess_return

                last_equity = equity
                running_max_equity = max(running_max_equity, equity)
                if running_max_equity > 0.0:
                    drawdown = equity / running_max_equity - 1.0
                    max_drawdown = min(max_drawdown, drawdown)

            prev_equity = equity
            pending_action = next_action

        out_cash[config_idx] = cash
        out_shares[config_idx] = shares
        out_in_position[config_idx] = in_position
        out_entry_price[config_idx] = entry_price
        out_pending_action[config_idx] = pending_action
        out_prev_equity[config_idx] = prev_equity
        out_trades_executed[config_idx] = trades_executed

        out_first_equity[config_idx] = first_equity
        out_last_equity[config_idx] = last_equity
        out_running_max_equity[config_idx] = running_max_equity
        out_return_count[config_idx] = return_count
        out_return_sum[config_idx] = return_sum
        out_return_sum_squares[config_idx] = return_sum_squares
        out_excess_return_count[config_idx] = excess_return_count
        out_excess_return_sum[config_idx] = excess_return_sum
        out_excess_return_sum_squares[config_idx] = excess_return_sum_squares
        out_positive_return_count[config_idx] = positive_return_count
        out_max_drawdown[config_idx] = max_drawdown

    return (
        updated,
        out_cash,
        out_shares,
        out_in_position,
        out_entry_price,
        out_pending_action,
        out_prev_equity,
        out_trades_executed,
        out_first_equity,
        out_last_equity,
        out_running_max_equity,
        out_return_count,
        out_return_sum,
        out_return_sum_squares,
        out_excess_return_count,
        out_excess_return_sum,
        out_excess_return_sum_squares,
        out_positive_return_count,
        out_max_drawdown,
    )


@njit(cache=True)
def run_single_equity_curve(
    open_prices: np.ndarray,
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    risk_free_returns: np.ndarray,
    buy_rsi: float,
    profit_target_multiple: float,
    initial_capital: float,
    trading_cost_rate: float,
    rsi_entry_rule: int = RSI_ENTRY_LOWER,
) -> tuple:
    row_count = open_prices.shape[0]
    equity_values = np.empty(row_count, dtype=np.float64)
    daily_returns = np.empty(row_count, dtype=np.float64)
    in_position_values = np.empty(row_count, dtype=np.int64)
    action_executed_values = np.empty(row_count, dtype=np.int64)
    pending_action_values = np.empty(row_count, dtype=np.int64)
    trades_executed_values = np.empty(row_count, dtype=np.int64)

    cash = initial_capital
    shares = 0.0
    in_position = False
    entry_price = np.nan
    pending_action = ACTION_NONE
    prev_equity = initial_capital
    trades_executed = 0

    for row_idx in range(row_count):
        open_price = open_prices[row_idx]
        close_price = close_prices[row_idx]
        rsi = rsi_values[row_idx]
        action_executed = ACTION_NONE

        deferred_action = ACTION_NONE
        if pending_action == ACTION_BUY and not in_position:
            cost_multiplier = 1.0 + trading_cost_rate
            if open_price > 0.0 and cash > 0.0 and cost_multiplier > 0.0:
                turnover_notional = cash / cost_multiplier
                shares = turnover_notional / open_price
                cash -= turnover_notional * cost_multiplier
                cash = max(cash, 0.0)
                in_position = shares > 0.0
                entry_price = open_price if in_position else np.nan
                if in_position:
                    trades_executed += 1
                    action_executed = ACTION_BUY
            else:
                deferred_action = ACTION_BUY
        elif pending_action == ACTION_SELL and in_position:
            if open_price > 0.0:
                turnover_notional = shares * open_price
                cash += turnover_notional
                cash -= turnover_notional * trading_cost_rate
                shares = 0.0
                in_position = False
                entry_price = np.nan
                trades_executed += 1
                action_executed = ACTION_SELL
            else:
                deferred_action = ACTION_SELL

        equity = cash + shares * close_price
        daily_return = equity / prev_equity - 1.0 if prev_equity > 0.0 else 0.0

        next_action = deferred_action
        if deferred_action == ACTION_NONE and not np.isnan(rsi):
            entry_signal = (
                rsi >= buy_rsi
                if rsi_entry_rule == RSI_ENTRY_UPPER
                else rsi <= buy_rsi
            )
            if (not in_position) and entry_signal:
                next_action = ACTION_BUY
            elif (
                in_position
                and not np.isnan(entry_price)
                and entry_price > 0.0
                and close_price / entry_price >= profit_target_multiple
            ):
                next_action = ACTION_SELL

        equity_values[row_idx] = equity
        daily_returns[row_idx] = daily_return
        in_position_values[row_idx] = 1 if in_position else 0
        action_executed_values[row_idx] = action_executed
        pending_action_values[row_idx] = next_action
        trades_executed_values[row_idx] = trades_executed

        prev_equity = equity
        pending_action = next_action

    return (
        equity_values,
        daily_returns,
        risk_free_returns.copy(),
        in_position_values,
        action_executed_values,
        pending_action_values,
        trades_executed_values,
        cash,
        shares,
        1 if in_position else 0,
        entry_price,
        pending_action,
        prev_equity,
        trades_executed,
    )
