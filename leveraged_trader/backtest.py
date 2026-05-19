from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig, RISK_FREE_SYMBOL
from .indicators import compute_rsi


def _date_str(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def performance_summary(
    equity_curve: pd.Series,
    risk_free_returns: Optional[pd.Series] = None,
) -> pd.Series:
    rets = equity_curve.pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float)

    if risk_free_returns is not None:
        aligned_returns = pd.concat(
            [rets.rename("Strategy"), risk_free_returns.rename("RiskFree")],
            axis=1,
            join="inner",
        ).dropna()
        sharpe_rets = aligned_returns["Strategy"] - aligned_returns["RiskFree"]
    else:
        sharpe_rets = rets

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(rets)) - 1.0
    vol = rets.std() * np.sqrt(252)
    sharpe = np.sqrt(252) * sharpe_rets.mean() / sharpe_rets.std() if sharpe_rets.std() > 0 else np.nan
    kelly_fraction = rets.mean() / rets.var() if rets.var() > 0 else np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = drawdown.min()

    hit_rate = (rets > 0).mean()

    return pd.Series(
        {
            "Total Return": total_return,
            "CAGR": cagr,
            "Annualized Vol": vol,
            "Sharpe": sharpe,
            "Kelly Fraction": kelly_fraction,
            "Max Drawdown": max_drawdown,
            "Hit Rate": hit_rate,
        }
    )


def backtest_rsi_asset_strategy(
    data: pd.DataFrame,
    cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
) -> pd.DataFrame:
    """
    Strategy:
      - Compute RSI on the signal symbol close.
      - If flat and RSI <= buy_rsi at today's close, buy the selected asset at next open.
      - If long and the selected asset reaches profit_target_multiple times its entry price
        at today's close, sell it at next open.
      - Long-only, 100% allocation when in position, otherwise cash.
    """
    df = data.copy()
    signal_close_col = f"{signal_symbol}_Close"
    signal_rsi_col = f"{signal_symbol}_RSI"
    df[signal_rsi_col] = compute_rsi(df[signal_close_col], cfg.rsi_period)
    open_col = f"{asset_symbol}_Open"
    close_col = f"{asset_symbol}_Close"

    trading_cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0

    cash = cfg.initial_capital
    shares = 0.0
    in_position = False
    entry_price = np.nan
    pending_action: Optional[str] = None

    records = []
    prev_equity = cfg.initial_capital

    dates = list(df.index)

    for i, date in enumerate(dates):
        row = df.loc[date]
        asset_open = float(row[open_col])
        asset_close = float(row[close_col])
        rsi = float(row[signal_rsi_col]) if pd.notna(row[signal_rsi_col]) else np.nan

        turnover_notional = 0.0
        trading_cost = 0.0
        action_executed = pending_action or "none"

        if pending_action == "buy" and not in_position:
            equity_at_open = cash
            shares = equity_at_open / asset_open if asset_open > 0 else 0.0
            turnover_notional = shares * asset_open
            trading_cost = turnover_notional * trading_cost_rate

            cash -= turnover_notional
            cash -= trading_cost
            in_position = shares > 0
            entry_price = asset_open if in_position else np.nan

        elif pending_action == "sell" and in_position:
            turnover_notional = shares * asset_open
            trading_cost = turnover_notional * trading_cost_rate

            cash += turnover_notional
            cash -= trading_cost
            shares = 0.0
            in_position = False
            entry_price = np.nan

        pending_action = None

        equity = cash + shares * asset_close
        daily_return = equity / prev_equity - 1.0 if i > 0 else 0.0
        position_return_multiple = (
            asset_close / entry_price if in_position and pd.notna(entry_price) and entry_price > 0 else np.nan
        )

        next_action: Optional[str] = None
        if pd.notna(rsi):
            if (not in_position) and (rsi <= cfg.buy_rsi):
                next_action = "buy"
            elif in_position and pd.notna(position_return_multiple) and (
                position_return_multiple >= cfg.profit_target_multiple
            ):
                next_action = "sell"

        pending_action = next_action

        records.append(
            {
                "Date": date,
                "SignalSymbol": signal_symbol,
                f"{signal_symbol}_Close": row[signal_close_col],
                f"{signal_symbol}_RSI": rsi,
                f"{asset_symbol}_Open": asset_open,
                f"{asset_symbol}_Close": asset_close,
                "Equity": equity,
                "DailyReturn": daily_return,
                "Cash": cash,
                "Shares": shares,
                "EntryPrice": entry_price,
                "PositionReturnMultiple": position_return_multiple,
                "InPosition": in_position,
                "ActionExecutedToday": action_executed,
                "PendingActionForNextOpen": pending_action or "none",
                "TurnoverNotional": turnover_notional,
                "TradingCost": trading_cost,
            }
        )

        prev_equity = equity

    return pd.DataFrame(records).set_index("Date")


def backtest_rsi_asset_equity(
    data: pd.DataFrame,
    cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
) -> tuple[pd.Series, int]:
    signal_rsi = compute_rsi(data[f"{signal_symbol}_Close"], cfg.rsi_period).to_numpy()
    asset_open = data[f"{asset_symbol}_Open"].to_numpy(dtype=float)
    asset_close = data[f"{asset_symbol}_Close"].to_numpy(dtype=float)

    trading_cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    cash = cfg.initial_capital
    shares = 0.0
    in_position = False
    entry_price = np.nan
    pending_action: Optional[str] = None
    trades_executed = 0
    equity_values: list[float] = []

    for i in range(len(data)):
        open_price = asset_open[i]
        close_price = asset_close[i]
        rsi = signal_rsi[i]

        if pending_action == "buy" and not in_position:
            shares = cash / open_price if open_price > 0 else 0.0
            turnover_notional = shares * open_price
            cash -= turnover_notional
            cash -= turnover_notional * trading_cost_rate
            in_position = shares > 0
            entry_price = open_price if in_position else np.nan
            trades_executed += 1
        elif pending_action == "sell" and in_position:
            turnover_notional = shares * open_price
            cash += turnover_notional
            cash -= turnover_notional * trading_cost_rate
            shares = 0.0
            in_position = False
            entry_price = np.nan
            trades_executed += 1

        pending_action = None
        equity = cash + shares * close_price
        equity_values.append(equity)

        if pd.notna(rsi):
            if (not in_position) and (rsi <= cfg.buy_rsi):
                pending_action = "buy"
            elif (
                in_position
                and pd.notna(entry_price)
                and entry_price > 0
                and close_price / entry_price >= cfg.profit_target_multiple
            ):
                pending_action = "sell"

    return pd.Series(equity_values, index=data.index), trades_executed


def run_parameter_sweep(
    data: pd.DataFrame,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    risk_free_returns: pd.Series,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> tuple[pd.DataFrame, BacktestConfig, pd.DataFrame]:
    rows = []
    best_cfg: Optional[BacktestConfig] = None
    best_strategy: Optional[pd.DataFrame] = None
    best_sharpe = -np.inf

    for buy_rsi in buy_rsi_values:
        for profit_target_multiple in profit_target_values:
            cfg = replace(
                base_cfg,
                buy_rsi=buy_rsi,
                profit_target_multiple=profit_target_multiple,
            )
            equity, trades_executed = backtest_rsi_asset_equity(
                data,
                cfg,
                asset_symbol=asset_symbol,
                signal_symbol=signal_symbol,
            )
            summary = performance_summary(equity, risk_free_returns=risk_free_returns)
            if summary.empty or "Sharpe" not in summary:
                continue

            row = {
                "Buy RSI": buy_rsi,
                "Sell Return Multiple": profit_target_multiple,
                "Trades Executed": trades_executed,
                **summary.to_dict(),
            }
            rows.append(row)

            sharpe = float(summary["Sharpe"])
            if pd.notna(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_cfg = cfg
                best_strategy = pd.DataFrame({"Equity": equity}, index=data.index)

    results = pd.DataFrame(rows).sort_values(
        by=["Sharpe", "Total Return", "CAGR"],
        ascending=False,
    ).reset_index(drop=True)

    if best_cfg is None or best_strategy is None:
        raise ValueError("Parameter sweep did not produce any results.")

    return results, best_cfg, best_strategy


def initial_strategy_state(base_cfg: BacktestConfig) -> dict:
    return {
        "start_date": None,
        "last_date": None,
        "cash": base_cfg.initial_capital,
        "shares": 0.0,
        "in_position": False,
        "entry_price": np.nan,
        "pending_action": "none",
        "prev_equity": base_cfg.initial_capital,
        "trades_executed": 0,
    }


def step_strategy_state(
    state: dict,
    date: pd.Timestamp,
    row: pd.Series,
    rsi: float,
    cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
) -> tuple[dict, dict]:
    open_price = float(row[f"{asset_symbol}_Open"])
    close_price = float(row[f"{asset_symbol}_Close"])
    pending_action = state["pending_action"]
    if pending_action not in {"buy", "sell"}:
        pending_action = "none"

    cash = float(state["cash"])
    shares = float(state["shares"])
    in_position = bool(state["in_position"])
    entry_price = float(state["entry_price"]) if pd.notna(state["entry_price"]) else np.nan
    trades_executed = int(state["trades_executed"])
    action_executed = pending_action

    trading_cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    if pending_action == "buy" and not in_position:
        shares = cash / open_price if open_price > 0 else 0.0
        turnover_notional = shares * open_price
        cash -= turnover_notional
        cash -= turnover_notional * trading_cost_rate
        in_position = shares > 0
        entry_price = open_price if in_position else np.nan
        trades_executed += 1
    elif pending_action == "sell" and in_position:
        turnover_notional = shares * open_price
        cash += turnover_notional
        cash -= turnover_notional * trading_cost_rate
        shares = 0.0
        in_position = False
        entry_price = np.nan
        trades_executed += 1

    equity = cash + shares * close_price
    previous_equity = float(state["prev_equity"])
    daily_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0

    position_return_multiple = (
        close_price / entry_price if in_position and pd.notna(entry_price) and entry_price > 0 else np.nan
    )
    next_action = "none"
    if pd.notna(rsi):
        if (not in_position) and (rsi <= cfg.buy_rsi):
            next_action = "buy"
        elif (
            in_position
            and pd.notna(position_return_multiple)
            and position_return_multiple >= cfg.profit_target_multiple
        ):
            next_action = "sell"

    risk_free_return = np.nan
    risk_free_col = f"{RISK_FREE_SYMBOL}_Close"
    if risk_free_col in row and pd.notna(row[risk_free_col]):
        annual_yield = float(row[risk_free_col]) / 100.0
        risk_free_return = (1.0 + annual_yield) ** (1 / 252) - 1.0

    date_str = _date_str(date)
    new_state = {
        "start_date": state["start_date"] or date_str,
        "last_date": date_str,
        "cash": cash,
        "shares": shares,
        "in_position": in_position,
        "entry_price": entry_price,
        "pending_action": next_action,
        "prev_equity": equity,
        "trades_executed": trades_executed,
    }
    equity_record = {
        "asset_symbol": asset_symbol,
        "signal_symbol": signal_symbol,
        "buy_rsi": cfg.buy_rsi,
        "profit_target_multiple": cfg.profit_target_multiple,
        "date": date_str,
        "equity": equity,
        "daily_return": daily_return,
        "risk_free_return": risk_free_return,
        "in_position": int(in_position),
        "action_executed": action_executed,
        "pending_action": next_action,
        "trades_executed": trades_executed,
    }
    return new_state, equity_record
