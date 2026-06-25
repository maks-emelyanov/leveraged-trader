from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi_from_average_gain_loss(avg_gain: pd.Series, avg_loss: pd.Series) -> pd.Series:
    both_flat = avg_gain.eq(0) & avg_loss.eq(0)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi.where(~both_flat, 50.0)


def rsi_value_from_average_gain_loss(avg_gain: float, avg_loss: float) -> float:
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder-style RSI using exponentially smoothed average gains/losses.
    """
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return _rsi_from_average_gain_loss(avg_gain, avg_loss)


def compute_rsi_details(close: pd.Series, period: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rsi = _rsi_from_average_gain_loss(avg_gain, avg_loss)
    return pd.DataFrame(
        {
            "close": close,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "rsi": rsi,
        },
        index=close.index,
    )
