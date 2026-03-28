from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# Configuration
# =========================

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    vol_target: float = 0.12          # 12% annualized target vol
    max_weight: float = 0.35          # cap TQQQ allocation at 35%
    rv_window: int = 20               # realized vol lookback
    sma_fast: int = 50
    sma_slow: int = 200
    ret_window: int = 20
    rv_exit_threshold: float = 0.35   # exit if QQQ rv20 > 35%
    daily_drop_exit: float = -0.03    # exit if QQQ daily return <= -3%
    fee_bps: float = 1.0              # per trade notional fee
    slippage_bps: float = 2.0         # per trade notional slippage
    cash_rate_annual: float = 0.0     # optional interest on cash


# =========================
# Data loading
# =========================

def load_from_yfinance(
    symbols: list[str] = ["QQQ", "TQQQ"],
    start: str = "2011-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV data from yfinance.

    Returns:
        {
            "QQQ":  DataFrame[Open, High, Low, Close, Volume],
            "TQQQ": DataFrame[Open, High, Low, Close, Volume],
        }
    """
    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        if symbol not in raw:
            raise ValueError(f"No data returned for {symbol}")

        df = raw[symbol].copy()

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna().sort_index()

        if df.empty:
            raise ValueError(f"Empty dataframe for {symbol}")

        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[symbol] = df

    return out


# =========================
# Strategy prep
# =========================

def prepare_data(qqq: pd.DataFrame, tqqq: pd.DataFrame) -> pd.DataFrame:
    """
    Requires:
      qqq columns:  Open, Close
      tqqq columns: Open, Close

    Returns merged frame indexed by date.
    """
    q = qqq.copy()
    t = tqqq.copy()

    q.columns = [f"QQQ_{c}" for c in q.columns]
    t.columns = [f"TQQQ_{c}" for c in t.columns]

    df = q.join(t, how="inner").sort_index()

    required = ["QQQ_Open", "QQQ_Close", "TQQQ_Open", "TQQQ_Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def build_signals(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    out = df.copy()

    q_close = out["QQQ_Close"]
    q_ret = q_close.pct_change()

    out["qqq_ret_1d"] = q_ret
    out["sma_fast"] = q_close.rolling(cfg.sma_fast).mean()
    out["sma_slow"] = q_close.rolling(cfg.sma_slow).mean()
    out["ret_n"] = q_close / q_close.shift(cfg.ret_window) - 1.0
    out["rv20"] = q_ret.rolling(cfg.rv_window).std() * np.sqrt(252)

    long_filter = (
        (q_close > out["sma_slow"]) &
        (out["sma_fast"] > out["sma_slow"]) &
        (out["ret_n"] > 0)
    )

    exit_trigger = (
        (q_close < out["sma_fast"]) |
        (out["rv20"] > cfg.rv_exit_threshold) |
        (out["qqq_ret_1d"] <= cfg.daily_drop_exit)
    )

    # Re-entry only after the long filter is true for 2 consecutive closes.
    long_filter_2day = long_filter & long_filter.shift(1).fillna(False)

    raw_weight = cfg.vol_target / (3.0 * out["rv20"])
    target_weight = raw_weight.clip(upper=cfg.max_weight)

    # No position unless the 2-day confirmation is satisfied.
    target_weight = target_weight.where(long_filter_2day, 0.0)

    # Exit overrides everything.
    target_weight = target_weight.where(~exit_trigger, 0.0)

    out["long_filter"] = long_filter
    out["long_filter_2day"] = long_filter_2day
    out["exit_trigger"] = exit_trigger
    out["target_weight"] = target_weight.fillna(0.0)

    return out


# =========================
# Core backtests
# =========================

def backtest_tqqq_strategy(
    qqq: pd.DataFrame,
    tqqq: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    """
    Backtest logic:
    - Signal computed on day t close using QQQ
    - Trade executed at day t+1 open in TQQQ
    - Position held until next rebalance/open
    - Portfolio marked daily using TQQQ close
    """
    if cfg is None:
        cfg = BacktestConfig()

    df = prepare_data(qqq, tqqq)
    df = build_signals(df, cfg)

    cash_rate_daily = (1.0 + cfg.cash_rate_annual) ** (1 / 252) - 1.0
    trading_cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0

    dates = df.index.to_list()
    n = len(df)

    cash = cfg.initial_capital
    shares = 0.0

    records = []
    prev_close_equity = cfg.initial_capital

    for i in range(n):
        date = dates[i]
        row = df.loc[date]

        open_px = float(row["TQQQ_Open"])
        close_px = float(row["TQQQ_Close"])

        equity_at_open = cash + shares * open_px

        # Rebalance at today's open using yesterday's target weight
        if i > 0:
            prev_signal_weight = float(df.iloc[i - 1]["target_weight"])
            target_dollar = equity_at_open * prev_signal_weight
            target_shares = 0.0 if open_px <= 0 else target_dollar / open_px

            share_change = target_shares - shares
            turnover_notional = abs(share_change) * open_px
            trading_cost = turnover_notional * trading_cost_rate

            cash -= share_change * open_px
            cash -= trading_cost
            shares = target_shares
        else:
            trading_cost = 0.0
            turnover_notional = 0.0

        cash *= (1.0 + cash_rate_daily)

        equity = cash + shares * close_px
        daily_return = equity / prev_close_equity - 1.0 if i > 0 else 0.0

        records.append(
            {
                "Date": date,
                "QQQ_Close": row["QQQ_Close"],
                "TQQQ_Open": open_px,
                "TQQQ_Close": close_px,
                "signal_weight_for_next_open": row["target_weight"],
                "shares_held_eod": shares,
                "cash_eod": cash,
                "turnover_notional": turnover_notional,
                "trading_cost": trading_cost,
                "equity": equity,
                "daily_return": daily_return,
                "sma_fast": row["sma_fast"],
                "sma_slow": row["sma_slow"],
                "ret_n": row["ret_n"],
                "rv20": row["rv20"],
                "long_filter": row["long_filter"],
                "long_filter_2day": row["long_filter_2day"],
                "exit_trigger": row["exit_trigger"],
            }
        )

        prev_close_equity = equity

    return pd.DataFrame(records).set_index("Date")


def backtest_buy_and_hold(
    asset: pd.DataFrame,
    initial_capital: float = 100_000.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 2.0,
    label: str = "asset",
) -> pd.DataFrame:
    """
    Buy at the first available open and hold forever.
    Mark daily equity to the close.

    asset must have:
      Open, Close
    """
    df = asset[["Open", "Close"]].dropna().sort_index().copy()
    if df.empty:
        raise ValueError(f"{label}: empty dataframe")

    trading_cost_rate = (fee_bps + slippage_bps) / 10_000.0

    first_open = float(df["Open"].iloc[0])
    initial_trade_cost = initial_capital * trading_cost_rate
    investable_capital = initial_capital - initial_trade_cost
    shares = investable_capital / first_open

    out = pd.DataFrame(index=df.index)
    out["equity"] = shares * df["Close"]
    out["daily_return"] = out["equity"].pct_change().fillna(0.0)
    out["shares"] = shares
    out["initial_trade_cost"] = initial_trade_cost

    return out


# =========================
# Performance
# =========================

def performance_summary(equity_curve: pd.Series) -> pd.Series:
    rets = equity_curve.pct_change().dropna()
    if len(rets) == 0:
        return pd.Series(dtype=float)

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(rets)) - 1.0
    vol = rets.std() * np.sqrt(252)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan

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
            "Max Drawdown": max_drawdown,
            "Hit Rate": hit_rate,
        }
    )


# =========================
# Benchmarks runner
# =========================

def run_backtest_with_benchmarks(
    start: str = "2011-01-01",
    end: str | None = None,
    cfg: Optional[BacktestConfig] = None,
    auto_adjust: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      equity_curves: DataFrame with equity curves for strategy and benchmarks
      summary:       DataFrame with performance metrics
    """
    if cfg is None:
        cfg = BacktestConfig()

    data = load_from_yfinance(
        symbols=["QQQ", "TQQQ"],
        start=start,
        end=end,
        auto_adjust=auto_adjust,
    )

    qqq = data["QQQ"]
    tqqq = data["TQQQ"]

    strategy_bt = backtest_tqqq_strategy(qqq, tqqq, cfg)
    qqq_bh = backtest_buy_and_hold(
        qqq,
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        label="QQQ",
    )
    tqqq_bh = backtest_buy_and_hold(
        tqqq,
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        label="TQQQ",
    )

    equity_curves = pd.concat(
        [
            strategy_bt["equity"].rename("Strategy"),
            qqq_bh["equity"].rename("QQQ_BuyHold"),
            tqqq_bh["equity"].rename("TQQQ_BuyHold"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    # Normalize to 1.0 for easier comparison
    normalized = equity_curves.div(equity_curves.iloc[0])

    summary = pd.DataFrame(
        {
            "Strategy": performance_summary(equity_curves["Strategy"]),
            "QQQ_BuyHold": performance_summary(equity_curves["QQQ_BuyHold"]),
            "TQQQ_BuyHold": performance_summary(equity_curves["TQQQ_BuyHold"]),
        }
    ).T

    return normalized, summary


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    cfg = BacktestConfig(
        initial_capital=100_000,
        vol_target=0.12,
        max_weight=0.35,
        fee_bps=1.0,
        slippage_bps=2.0,
        cash_rate_annual=0.0,
    )

    curves, summary = run_backtest_with_benchmarks(
        start="2011-01-01",
        end=None,
        cfg=cfg,
        auto_adjust=True,
    )

    pd.set_option("display.float_format", "{:.4f}".format)

    print("\nNormalized equity curves (tail):")
    print(curves.tail())

    print("\nPerformance summary:")
    print(summary)

    # Visualization:
    import matplotlib.pyplot as plt
    curves.plot(figsize=(12, 7), title="Strategy vs Buy-and-Hold Benchmarks")
    plt.ylabel("Growth of $1")
    plt.show()