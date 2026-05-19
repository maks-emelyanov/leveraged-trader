from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from .config import RISK_FREE_SYMBOL
from .storage import _date_str


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Extract one symbol's OHLCV frame from a multi-symbol yfinance download,
    handling both ticker-first and ticker-second MultiIndex layouts.
    """
    if raw.empty:
        raise ValueError("No data downloaded.")

    if not isinstance(raw.columns, pd.MultiIndex):
        df = raw.copy()
        df.columns = [str(c) for c in df.columns]
        return df

    lvl0 = list(raw.columns.get_level_values(0))
    lvl1 = list(raw.columns.get_level_values(1))

    if symbol in lvl0:
        df = raw[symbol].copy()
        df.columns = [str(c) for c in df.columns]
        return df

    if symbol in lvl1:
        df = raw.xs(symbol, axis=1, level=1).copy()
        df.columns = [str(c) for c in df.columns]
        return df

    unique_lvl0 = sorted(set(map(str, lvl0)))
    unique_lvl1 = sorted(set(map(str, lvl1)))
    raise ValueError(
        f"Missing downloaded data for {symbol}. "
        f"level0={unique_lvl0}, level1={unique_lvl1}"
    )


def load_market_data(
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = True,
    symbols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Downloads daily data from Yahoo Finance and returns a merged DataFrame
    with SYMBOL_Field columns.
    """
    if symbols is None:
        raise ValueError("symbols must be provided")

    symbols = list(dict.fromkeys(symbols))
    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        period="max" if start is None and end is None else None,
        interval="1d",
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("No data downloaded.")

    frames = []
    for symbol in symbols:
        df = _extract_symbol_frame(raw, symbol)

        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if not keep:
            raise ValueError(f"No OHLCV columns found for {symbol}. Columns: {list(df.columns)}")

        df = df[keep].copy()
        df.columns = [f"{symbol}_{c}" for c in df.columns]
        frames.append(df)

    out = pd.concat(frames, axis=1, join="inner").dropna().sort_index()
    if out.empty:
        raise ValueError(f"No overlapping market data available for symbols: {symbols}")

    out.index = pd.to_datetime(out.index).tz_localize(None)

    return out


def load_strategy_data(
    asset_symbol: str,
    signal_symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    core_symbols = [asset_symbol, signal_symbol]
    data = load_market_data(
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        symbols=core_symbols,
    )
    risk_free = load_market_data(
        start=_date_str(data.index.min()),
        end=end,
        auto_adjust=auto_adjust,
        symbols=[RISK_FREE_SYMBOL],
    )
    return data.join(risk_free, how="left").ffill()


def risk_free_daily_returns(data: pd.DataFrame, risk_free_symbol: str = RISK_FREE_SYMBOL) -> pd.Series:
    annual_yield = data[f"{risk_free_symbol}_Close"] / 100.0
    return ((1.0 + annual_yield) ** (1 / 252) - 1.0).rename("RiskFree")
