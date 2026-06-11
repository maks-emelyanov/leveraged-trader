from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig


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
