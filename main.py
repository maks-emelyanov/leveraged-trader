from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    rsi_period: int = 14
    buy_rsi: float = 30.0
    profit_target_multiple: float = 10.0
    fee_bps: float = 1.0        # commission-like cost per trade notional
    slippage_bps: float = 2.0   # slippage per trade notional
    auto_adjust: bool = True


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder-style RSI using exponentially smoothed average gains/losses.
    """
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases more gracefully
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)

    return rsi


def load_market_data(
    start: str = "1900-01-01",
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Downloads QQQ, TQQQ, SMH, SPY, and SOXL daily data and returns a merged DataFrame.

    Output columns:
        QQQ_Open, QQQ_High, QQQ_Low, QQQ_Close, QQQ_Volume,
        TQQQ_Open, TQQQ_High, TQQQ_Low, TQQQ_Close, TQQQ_Volume,
        SMH_Open, SMH_High, SMH_Low, SMH_Close, SMH_Volume,
        SPY_Open, SPY_High, SPY_Low, SPY_Close, SPY_Volume,
        SOXL_Open, SOXL_High, SOXL_Low, SOXL_Close, SOXL_Volume
    """
    raw = yf.download(
        tickers=["QQQ", "TQQQ", "SMH", "SPY", "SOXL"],
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("No data downloaded.")

    frames = []
    for symbol in ["QQQ", "TQQQ", "SMH", "SPY", "SOXL"]:
        if symbol not in raw:
            raise ValueError(f"Missing downloaded data for {symbol}")

        df = raw[symbol].copy()
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df.columns = [f"{symbol}_{c}" for c in df.columns]
        frames.append(df)

    out = pd.concat(frames, axis=1, join="inner").dropna().sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)

    return out


def performance_summary(equity_curve: pd.Series) -> pd.Series:
    rets = equity_curve.pct_change().dropna()
    if rets.empty:
        return pd.Series(dtype=float)

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(rets)) - 1.0
    vol = rets.std() * np.sqrt(252)
    sharpe = np.sqrt(252) * rets.mean() / rets.std() if rets.std() > 0 else np.nan
    # Kelly fraction for a single risky asset under simple-return assumptions.
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
) -> pd.DataFrame:
    """
    Strategy:
      - Compute RSI on QQQ close.
      - If flat and QQQ RSI <= buy_rsi at today's close, buy the selected asset at next open.
      - If long and the selected asset reaches profit_target_multiple times its entry price
        at today's close, sell it at next open.
      - Long-only, 100% allocation when in position, otherwise cash.
    """
    df = data.copy()
    df["QQQ_RSI"] = compute_rsi(df["QQQ_Close"], cfg.rsi_period)
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
        rsi = float(row["QQQ_RSI"]) if pd.notna(row["QQQ_RSI"]) else np.nan

        turnover_notional = 0.0
        trading_cost = 0.0
        action_executed = pending_action or "none"

        # Execute yesterday's signal at today's open
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

        # Mark to close
        equity = cash + shares * asset_close
        daily_return = equity / prev_equity - 1.0 if i > 0 else 0.0
        position_return_multiple = (
            asset_close / entry_price if in_position and pd.notna(entry_price) and entry_price > 0 else np.nan
        )

        # Generate signal at today's close for next open
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
                "QQQ_Close": row["QQQ_Close"],
                "QQQ_RSI": rsi,
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


def backtest_buy_and_hold(
    data: pd.DataFrame,
    asset_symbol: str,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Buy the selected asset at the first open and hold.
    """
    open_col = f"{asset_symbol}_Open"
    close_col = f"{asset_symbol}_Close"
    df = data[[open_col, close_col]].dropna().copy()
    if df.empty:
        raise ValueError(f"No {asset_symbol} data available for benchmark.")

    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    first_open = float(df[open_col].iloc[0])

    initial_trade_cost = initial_capital * cost_rate
    investable_capital = initial_capital - initial_trade_cost
    shares = investable_capital / first_open

    out = pd.DataFrame(index=df.index)
    out["Equity"] = shares * df[close_col]
    out["DailyReturn"] = out["Equity"].pct_change().fillna(0.0)

    return out


def run_parameter_sweep(
    data: pd.DataFrame,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> tuple[pd.DataFrame, BacktestConfig, pd.DataFrame]:
    rows = []
    best_cfg: Optional[BacktestConfig] = None
    best_strategy: Optional[pd.DataFrame] = None
    best_total_return = -np.inf

    for buy_rsi in buy_rsi_values:
        for profit_target_multiple in profit_target_values:
            cfg = replace(
                base_cfg,
                buy_rsi=buy_rsi,
                profit_target_multiple=profit_target_multiple,
            )
            strategy = backtest_rsi_asset_strategy(data, cfg, asset_symbol=asset_symbol)
            summary = performance_summary(strategy["Equity"])
            trades_executed = int((strategy["ActionExecutedToday"] != "none").sum())

            row = {
                "Buy RSI": buy_rsi,
                "Sell Return Multiple": profit_target_multiple,
                "Trades Executed": trades_executed,
                **summary.to_dict(),
            }
            rows.append(row)

            total_return = float(summary["Total Return"])
            if total_return > best_total_return:
                best_total_return = total_return
                best_cfg = cfg
                best_strategy = strategy

    results = pd.DataFrame(rows).sort_values(
        by=["Total Return", "Sharpe", "CAGR"],
        ascending=False,
    ).reset_index(drop=True)

    if best_cfg is None or best_strategy is None:
        raise ValueError("Parameter sweep did not produce any results.")

    return results, best_cfg, best_strategy


if __name__ == "__main__":
    base_cfg = BacktestConfig(
        initial_capital=100_000,
        rsi_period=14,
        buy_rsi=30,
        profit_target_multiple=10.0,
        fee_bps=1.0,
        slippage_bps=2.0,
        auto_adjust=True,
    )

    data = load_market_data(
        start="1900-01-01",
        end=None,
        auto_adjust=base_cfg.auto_adjust,
    )

    # Refined around the latest standout near buy_rsi=40 and sell targets around 1.8.
    buy_rsi_values = [39, 40, 41, 42]
    profit_target_values = [1.72, 1.75, 1.78, 1.8, 1.82, 1.85, 1.88, 1.9, 1.95, 2.0]

    sweep_results, best_cfg, strategy = run_parameter_sweep(
        data,
        base_cfg,
        asset_symbol="TQQQ",
        buy_rsi_values=buy_rsi_values,
        profit_target_values=profit_target_values,
    )

    tqqq_benchmark = backtest_buy_and_hold(
        data,
        asset_symbol="TQQQ",
        initial_capital=best_cfg.initial_capital,
        fee_bps=best_cfg.fee_bps,
        slippage_bps=best_cfg.slippage_bps,
    )
    smh_benchmark = backtest_buy_and_hold(
        data,
        asset_symbol="SMH",
        initial_capital=best_cfg.initial_capital,
        fee_bps=best_cfg.fee_bps,
        slippage_bps=best_cfg.slippage_bps,
    )
    spy_benchmark = backtest_buy_and_hold(
        data,
        asset_symbol="SPY",
        initial_capital=best_cfg.initial_capital,
        fee_bps=best_cfg.fee_bps,
        slippage_bps=best_cfg.slippage_bps,
    )
    soxl_benchmark = backtest_buy_and_hold(
        data,
        asset_symbol="SOXL",
        initial_capital=best_cfg.initial_capital,
        fee_bps=best_cfg.fee_bps,
        slippage_bps=best_cfg.slippage_bps,
    )
    qqq_benchmark = backtest_buy_and_hold(
        data,
        asset_symbol="QQQ",
        initial_capital=best_cfg.initial_capital,
        fee_bps=best_cfg.fee_bps,
        slippage_bps=best_cfg.slippage_bps,
    )

    curves = pd.concat(
        [
            strategy["Equity"].rename("TQQQ_RSI_Strategy"),
            tqqq_benchmark["Equity"].rename("TQQQ_BuyHold"),
            smh_benchmark["Equity"].rename("SMH_BuyHold"),
            spy_benchmark["Equity"].rename("SPY_BuyHold"),
            soxl_benchmark["Equity"].rename("SOXL_BuyHold"),
            qqq_benchmark["Equity"].rename("QQQ_BuyHold"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    normalized = curves / curves.iloc[0]

    best_summary = pd.DataFrame(
        {
            "TQQQ_RSI_Strategy": performance_summary(curves["TQQQ_RSI_Strategy"]),
            "TQQQ_BuyHold": performance_summary(curves["TQQQ_BuyHold"]),
            "SMH_BuyHold": performance_summary(curves["SMH_BuyHold"]),
            "SPY_BuyHold": performance_summary(curves["SPY_BuyHold"]),
            "SOXL_BuyHold": performance_summary(curves["SOXL_BuyHold"]),
            "QQQ_BuyHold": performance_summary(curves["QQQ_BuyHold"]),
        }
    ).T

    pd.set_option("display.float_format", "{:.4f}".format)

    print("\nParameter sweep results (top 10 by Total Return):")
    print(sweep_results.head(10))

    print(
        "\nBest TQQQ strategy parameters:"
        f" buy_rsi={best_cfg.buy_rsi},"
        f" sell_return_multiple={best_cfg.profit_target_multiple}"
    )

    print("\nNormalized equity curves (tail):")
    print(normalized.tail())

    print("\nPerformance summary for best strategy and comparison portfolios:")
    print(best_summary)

    print("\nRecent TQQQ strategy rows for best parameter set:")
    print(
        strategy[
            [
                "QQQ_RSI",
                "Equity",
                "PositionReturnMultiple",
                "InPosition",
                "ActionExecutedToday",
                "PendingActionForNextOpen",
            ]
        ].tail(20)
    )

    # Optional plot:
    # import matplotlib.pyplot as plt
    # normalized.plot(figsize=(12, 7), title="QQQ RSI -> TQQQ Strategy vs TQQQ/SMH/SPY Buy & Hold")
    # plt.ylabel("Growth of $1")
    # plt.savefig("qqq_rsi_tqqq_strategy.png", dpi=150, bbox_inches="tight")
