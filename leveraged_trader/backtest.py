from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import BacktestConfig


def _validated_series(
    values: pd.Series,
    *,
    name: str,
    allow_missing: bool,
) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if values.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicate observations.")
    if not values.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing order.")
    if isinstance(values.index, pd.MultiIndex):
        index_has_missing_labels = any(bool((codes == -1).any()) for codes in values.index.codes)
    else:
        index_has_missing_labels = bool(pd.isna(values.index).any())
    if index_has_missing_labels:
        raise ValueError(f"{name} index must not contain missing labels.")
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
        pd.api.types.is_datetime64_any_dtype(values.dtype)
        or pd.api.types.is_timedelta64_dtype(values.dtype)
        or any(isinstance(value, invalid_scalar_types) for value in values.array)
    ):
        raise ValueError(f"{name} must contain numeric values.")
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    array = numeric.to_numpy()
    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinite values.")
    if not allow_missing and np.isnan(array).any():
        raise ValueError(f"{name} must not contain missing values.")
    return numeric


def _require_representable_metric(name: str, value: float, *, allow_nan: bool = False) -> float:
    result = float(value)
    if np.isinf(result) or (np.isnan(result) and not allow_nan):
        raise ValueError(f"performance_summary produced a non-finite {name}.")
    return result


def _validate_risk_free_return_domain(values: object) -> None:
    """Require every observed daily benchmark return to preserve non-negative wealth."""
    array = np.asarray(values, dtype=np.float64)
    invalid = (~np.isnan(array)) & (array <= -1.0)
    if invalid.any():
        raise ValueError("risk_free_returns must contain values greater than -1.0 or missing values.")


def performance_summary(
    equity_curve: pd.Series,
    risk_free_returns: pd.Series | None = None,
) -> pd.Series:
    equity_curve = _validated_series(
        equity_curve,
        name="equity_curve",
        allow_missing=False,
    )
    if (equity_curve <= 0.0).any():
        raise ValueError("equity_curve must contain only positive values.")
    if risk_free_returns is not None:
        risk_free_returns = _validated_series(
            risk_free_returns,
            name="risk_free_returns",
            allow_missing=True,
        )
        _validate_risk_free_return_domain(risk_free_returns)

    # Each observation is one trading session. Calendar gaps (weekends,
    # holidays, or an irregular but explicitly supplied trading calendar) do
    # not create synthetic zero-return sessions.
    equity_values = equity_curve.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        return_values = equity_values[1:] / equity_values[:-1] - 1.0
    if not np.isfinite(return_values).all():
        raise ValueError("equity_curve produces non-finite daily returns.")
    rets = pd.Series(return_values, index=equity_curve.index[1:], name=equity_curve.name)
    if rets.empty:
        return pd.Series(dtype=float)

    if risk_free_returns is not None:
        aligned_returns = pd.concat(
            [rets.rename("Strategy"), risk_free_returns.rename("RiskFree")],
            axis=1,
            join="inner",
        ).dropna()
        with np.errstate(invalid="ignore", over="ignore"):
            sharpe_rets = aligned_returns["Strategy"] - aligned_returns["RiskFree"]
        if not np.isfinite(sharpe_rets.to_numpy(dtype=np.float64)).all():
            raise ValueError("strategy and risk-free returns produce non-finite excess returns.")
    else:
        sharpe_rets = rets

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        equity_ratio = equity_values[-1] / equity_values[0]
        total_return = equity_ratio - 1.0
        cagr = equity_ratio ** (252 / len(rets)) - 1.0
        return_mean = rets.mean()
        return_variance = rets.var()
        return_std = rets.std()
        sharpe_mean = sharpe_rets.mean()
        sharpe_std = sharpe_rets.std()
        vol = return_std * np.sqrt(252)
        sharpe = np.sqrt(252) * sharpe_mean / sharpe_std if sharpe_std > 0 else np.nan
        kelly_fraction = return_mean / return_variance if return_variance > 0 else np.nan

    for name, value, allow_nan in (
        ("total return", total_return, False),
        ("CAGR", cagr, False),
        ("mean return", return_mean, False),
        ("return variance", return_variance, len(rets) < 2),
        ("return standard deviation", return_std, len(rets) < 2),
        ("mean excess return", sharpe_mean, sharpe_rets.empty),
        ("excess-return standard deviation", sharpe_std, len(sharpe_rets) < 2),
        ("annualized volatility", vol, len(rets) < 2),
        ("Sharpe ratio", sharpe, sharpe_std <= 0.0 or np.isnan(sharpe_std)),
        ("Kelly fraction", kelly_fraction, return_variance <= 0.0 or np.isnan(return_variance)),
    ):
        _require_representable_metric(name, value, allow_nan=allow_nan)

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = drawdown.min()

    hit_rate = (rets > 0).mean()

    _require_representable_metric("maximum drawdown", max_drawdown)
    _require_representable_metric("hit rate", hit_rate)

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
