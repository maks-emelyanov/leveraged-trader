from __future__ import annotations

from datetime import date, timedelta
from numbers import Real

import numpy as np
import pandas as pd


def _rsi_from_average_gain_loss(avg_gain: np.ndarray, avg_loss: np.ndarray) -> np.ndarray:
    both_flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rs = avg_gain / np.where(avg_loss == 0.0, np.nan, avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where(avg_loss != 0.0, rsi, 100.0)
    rsi = np.where(avg_gain != 0.0, rsi, 0.0)
    return np.ascontiguousarray(np.where(~both_flat, rsi, 50.0), dtype=np.float64)


def rsi_value_from_average_gain_loss(avg_gain: float, avg_loss: float) -> float:
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in (avg_gain, avg_loss)):
        raise ValueError("RSI average gain and loss must be numeric scalars.")
    try:
        numeric_gain = float(avg_gain)
        numeric_loss = float(avg_loss)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("RSI average gain and loss must be numeric scalars.") from exc
    if not np.isfinite(numeric_gain) or not np.isfinite(numeric_loss):
        raise ValueError("RSI average gain and loss must remain finite.")
    if numeric_gain < 0 or numeric_loss < 0:
        raise ValueError("RSI average gain and loss must be non-negative.")
    if numeric_gain == 0 and numeric_loss == 0:
        return 50.0
    if numeric_loss == 0:
        return 100.0
    if numeric_gain == 0:
        return 0.0
    rs = numeric_gain / numeric_loss
    return 100 - (100 / (1 + rs))


def _validate_rsi_inputs(close: pd.Series, period: int) -> tuple[pd.Series, int]:
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series.")
    invalid_period_types = (bool, np.bool_, np.datetime64, np.timedelta64)
    if isinstance(period, invalid_period_types) or not isinstance(period, (int, np.integer)) or period <= 0:
        raise ValueError("period must be a positive integer.")
    if isinstance(close.index, pd.MultiIndex):
        index_has_missing_labels = any(bool((codes == -1).any()) for codes in close.index.codes)
    else:
        index_has_missing_labels = bool(pd.isna(close.index).any())
    if index_has_missing_labels:
        raise ValueError("close index must not contain missing labels.")
    if close.index.has_duplicates:
        raise ValueError("close index must not contain duplicate observations.")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be sorted in increasing order.")
    invalid_scalar_types = (
        bool,
        np.bool_,
        complex,
        np.complexfloating,
        date,
        timedelta,
        np.datetime64,
        np.timedelta64,
        np.ndarray,
    )
    if (
        pd.api.types.is_datetime64_any_dtype(close.dtype)
        or pd.api.types.is_timedelta64_dtype(close.dtype)
        or any(isinstance(value, invalid_scalar_types) for value in close.array)
    ):
        raise ValueError("close must contain numeric values.")
    try:
        numeric_close = close.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("close must contain numeric values.") from exc
    if numeric_close.isna().any():
        raise ValueError("close must not contain missing values.")
    if np.isinf(numeric_close.to_numpy()).any():
        raise ValueError("close must not contain infinite values.")
    return numeric_close, int(period)


def _overflow_resistant_mean(values: np.ndarray) -> float:
    average = 0.0
    for count, value in enumerate(values, start=1):
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError("RSI average gain and loss must remain finite.")
        average += (numeric_value - average) / count
        if not np.isfinite(average):
            raise ValueError("RSI average gain and loss must remain finite.")
    return average


def _compute_rsi_arrays(close_values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Wilder state in contiguous arrays without pandas scalar access."""
    with np.errstate(over="ignore", invalid="ignore"):
        delta = close_values[1:] - close_values[:-1]
    if not np.isfinite(delta).all():
        raise ValueError("RSI average gain and loss must remain finite.")

    gain = np.clip(delta, 0.0, None)
    loss = -np.clip(delta, None, 0.0)
    avg_gain = np.full(len(close_values), np.nan, dtype=np.float64)
    avg_loss = np.full(len(close_values), np.nan, dtype=np.float64)
    if len(close_values) > period:
        avg_gain[period] = _overflow_resistant_mean(gain[:period])
        avg_loss[period] = _overflow_resistant_mean(loss[:period])
        alpha = 1 / period
        previous_weight = 1 - alpha
        for position in range(period + 1, len(close_values)):
            avg_gain[position] = previous_weight * avg_gain[position - 1] + alpha * gain[position - 1]
            avg_loss[position] = previous_weight * avg_loss[position - 1] + alpha * loss[position - 1]
            if not np.isfinite(avg_gain[position]) or not np.isfinite(avg_loss[position]):
                raise ValueError("RSI average gain and loss must remain finite.")

    return avg_gain, avg_loss, _rsi_from_average_gain_loss(avg_gain, avg_loss)


def _validated_rsi_arrays(close: pd.Series, period: int) -> tuple[pd.Series, int, np.ndarray]:
    numeric_close, period = _validate_rsi_inputs(close, period)
    close_values = np.ascontiguousarray(
        numeric_close.to_numpy(dtype=np.float64, copy=False),
        dtype=np.float64,
    )
    return numeric_close, period, close_values


def _rsi_details_frame(
    close: pd.Series,
    close_values: np.ndarray,
    avg_gain: np.ndarray,
    avg_loss: np.ndarray,
    rsi: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": close_values,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "rsi": rsi,
        },
        index=close.index,
    )


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Return Wilder RSI, seeded with the first period's simple averages."""
    close, period, close_values = _validated_rsi_arrays(close, period)
    _avg_gain, _avg_loss, rsi = _compute_rsi_arrays(close_values, period)
    return pd.Series(rsi, index=close.index, name="rsi", dtype=float, copy=False)


def compute_rsi_details(close: pd.Series, period: int = 14) -> pd.DataFrame:
    close, period, close_values = _validated_rsi_arrays(close, period)
    avg_gain, avg_loss, rsi = _compute_rsi_arrays(close_values, period)
    return _rsi_details_frame(close, close_values, avg_gain, avg_loss, rsi)
