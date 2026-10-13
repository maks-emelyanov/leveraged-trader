"""Chronological Sharpe, holding-duration, and trade-activity selection.

These fixed research rules are deliberately not another tunable parameter grid.
The holdout can reject the development winner, but cannot change its parameters.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .optimized_backtest import RSI_ENTRY_UPPER
from .research_metrics import grid_research_metrics

SELECTION_VERSION = 2
MIN_BLOCK_SESSIONS = 126
DEVELOPMENT_BLOCKS = 3
MIN_DEVELOPMENT_TRADES = 3
MIN_HOLDOUT_TRADES = 2
TRADE_ACTIVITY_WEIGHT = 0.5
TRADE_ACTIVITY_ANNUAL_SCALE = 12.0


def development_trade_activity(
    completed_trades: np.ndarray, block_sessions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the slowest block's annual round-trip rate and bounded reward.

    Counts are shaped (blocks, candidates). The fixed scale and weight express
    a preference, not parameters fitted to returns. Only completed round trips
    earn credit; a long final open position still incurs the duration penalty.
    """
    rates = 252.0 * np.asarray(completed_trades, dtype=float) / np.asarray(block_sessions)[:, None]
    minimum_rate = np.min(rates, axis=0)
    bonus = TRADE_ACTIVITY_WEIGHT * minimum_rate / (minimum_rate + TRADE_ACTIVITY_ANNUAL_SCALE)
    return minimum_rate, bonus


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _neighbor_scores(scores: np.ndarray, buys: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Discount isolated peaks using adjacent values along each parameter axis."""
    buy_axis = sorted(set(buys))
    target_axis = sorted(set(targets))
    lookup = {(float(buy), float(target)): i for i, (buy, target) in enumerate(zip(buys, targets, strict=True))}
    smoothed = scores.copy()
    for i, (buy, target) in enumerate(zip(buys, targets, strict=True)):
        if not np.isfinite(scores[i]):
            continue
        neighbors = []
        for axis, value, other, is_buy in ((buy_axis, buy, target, True), (target_axis, target, buy, False)):
            position = axis.index(value)
            for adjacent in (position - 1, position + 1):
                if 0 <= adjacent < len(axis):
                    key = (axis[adjacent], other) if is_buy else (other, axis[adjacent])
                    if key in lookup:
                        neighbors.append(scores[lookup[key]])
        if neighbors:
            # Invalid neighbors supply no evidence of a stable parameter region.
            supported = [float(x) if np.isfinite(x) else 0.0 for x in neighbors]
            smoothed[i] = min(scores[i], float(np.median(supported)))
    return smoothed


def select_strategy(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    risk_free_returns: np.ndarray,
    buy_rsi_values: np.ndarray,
    profit_target_values: np.ndarray,
    initial_capital: float,
    trading_cost_rate: float,
    rsi_entry_rule: int = 0,
    deadline_check: Callable[[], None] | None = None,
    *,
    metric_evaluator: Callable[..., dict[str, np.ndarray]] | None = None,
    selection_version: int = SELECTION_VERSION,
) -> dict[str, object]:
    """Choose on the first 80%, then accept/reject on the final 20%.

    Each block starts flat with no pending order. RSI may use earlier warmup
    observations, but must be causal. Open positions are marked to market at
    block end and counted in holding duration, without inventing a sell fill.
    Version 1 is retained only to reproduce the frozen historical grid study.
    """
    if type(selection_version) is not int or selection_version not in {1, SELECTION_VERSION}:
        raise ValueError("Unsupported selection version.")
    buys = np.asarray(buy_rsi_values, dtype=float)
    targets = np.asarray(profit_target_values, dtype=float)
    if buys.ndim != 1 or targets.shape != buys.shape or not len(buys):
        raise ValueError("Selection requires nonempty paired one-dimensional parameter arrays.")
    if len(set(zip(buys, targets, strict=True))) != len(buys):
        raise ValueError("Selection parameter pairs must be unique.")
    row_count = len(open_prices)
    holdout_start = row_count * 4 // 5
    boundaries = np.linspace(0, holdout_start, DEVELOPMENT_BLOCKS + 1, dtype=int)
    selective = -buys if rsi_entry_rule == RSI_ENTRY_UPPER else buys
    fallback = min(range(len(buys)), key=lambda i: (selective[i], targets[i]))
    result: dict[str, object] = {
        "version": selection_version,
        "buy_rsi": float(buys[fallback]),
        "profit_target_multiple": float(targets[fallback]),
        "status": "insufficient_history",
        "selection_score": None,
        "candidate_count": len(buys),
        "development_sessions": holdout_start,
        "holdout_sessions": row_count - holdout_start,
        "development_completed_trades": 0,
        "development_min_annual_trades": None,
        "trade_activity_bonus": None,
        "development_block_completed_trades": [],
        "development_block_sharpes": [],
        "development_block_returns": [],
        "development_sharpe_median": None,
        "development_sharpe_std": None,
        "mean_holding_sessions": None,
        "max_holding_sessions": None,
        "holdout_sharpe": None,
        "holdout_total_return": None,
        "holdout_completed_trades": 0,
        "holdout_mean_holding_sessions": None,
        "holdout_max_holding_sessions": None,
    }
    if (
        row_count < 5 * MIN_BLOCK_SESSIONS
        or min(np.diff(boundaries)) < MIN_BLOCK_SESSIONS
        or row_count - holdout_start < MIN_BLOCK_SESSIONS
    ):
        return result

    arrays = (open_prices, high_prices, close_prices, rsi_values, risk_free_returns)

    def evaluate(start: int, stop: int, selected: int | None = None) -> dict[str, np.ndarray]:
        if deadline_check is not None:
            deadline_check()
        indices = slice(None) if selected is None else slice(selected, selected + 1)
        evaluator = grid_research_metrics if metric_evaluator is None else metric_evaluator
        metrics = evaluator(
            *(np.asarray(values)[start:stop] for values in arrays),
            buys[indices], targets[indices], initial_capital, trading_cost_rate, rsi_entry_rule,
        )
        if deadline_check is not None:
            deadline_check()
        return metrics

    blocks = [evaluate(int(start), int(stop)) for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)]
    sharpes = np.stack([block["sharpe"] for block in blocks])
    trades = np.stack([block["completed_trades"] for block in blocks])
    mean_holds = np.stack([block["mean_holding_sessions"] for block in blocks])
    max_holds = np.stack([block["max_holding_sessions"] for block in blocks])
    returns = np.stack([block["total_return"] for block in blocks])
    eligible = (
        np.all(np.isfinite(sharpes), axis=0)
        & np.all(trades >= MIN_DEVELOPMENT_TRADES, axis=0)
        & np.all(returns > 0.0, axis=0)
        & np.all(sharpes > 0.0, axis=0)
        & np.all(np.isfinite(mean_holds), axis=0)
    )
    median_sharpe = np.median(sharpes, axis=0)
    sharpe_std = np.std(sharpes, axis=0)
    # Equal block weights prevent a busy market regime from hiding a slow one.
    mean_hold = np.mean(mean_holds, axis=0)
    max_hold = np.max(max_holds, axis=0)
    scores = median_sharpe - sharpe_std - 0.5 * np.log1p(mean_hold / 21.0) - 0.25 * np.log1p(max_hold / 63.0)
    minimum_trade_rate, activity_bonus = development_trade_activity(trades, np.diff(boundaries))
    if selection_version >= 2:
        # More activity cannot turn weak Sharpe/duration evidence into a pass.
        eligible &= np.isfinite(scores) & (scores > 0.0) & np.isfinite(activity_bonus)
        scores += activity_bonus
    scores[~eligible] = -np.inf
    scores = _neighbor_scores(scores, buys, targets)
    candidates = np.flatnonzero(np.isfinite(scores) & (scores > 0.0))
    if not len(candidates):
        result["status"] = "insufficient_evidence"
        return result
    winner = min(candidates, key=lambda i: (
        -scores[i], -minimum_trade_rate[i] if selection_version >= 2 else 0.0,
        mean_hold[i], max_hold[i], selective[i], targets[i],
    ))
    result.update(
        buy_rsi=float(buys[winner]),
        profit_target_multiple=float(targets[winner]),
        selection_score=float(scores[winner]),
        development_completed_trades=int(trades[:, winner].sum()),
        development_min_annual_trades=float(minimum_trade_rate[winner]) if selection_version >= 2 else None,
        trade_activity_bonus=float(activity_bonus[winner]) if selection_version >= 2 else None,
        development_block_completed_trades=trades[:, winner].astype(int).tolist(),
        development_block_sharpes=sharpes[:, winner].tolist(),
        development_block_returns=returns[:, winner].tolist(),
        development_sharpe_median=float(median_sharpe[winner]),
        development_sharpe_std=float(sharpe_std[winner]),
        mean_holding_sessions=float(mean_hold[winner]),
        max_holding_sessions=int(max_hold[winner]),
    )
    # Evaluate only the frozen winner: a failed holdout never triggers reranking.
    holdout = evaluate(holdout_start, row_count, int(winner))
    holdout_sharpe = float(holdout["sharpe"][0])
    holdout_return = float(holdout["total_return"][0])
    holdout_trades = int(holdout["completed_trades"][0])
    passed = (
        np.isfinite(holdout_sharpe) and holdout_sharpe > 0.0
        and np.isfinite(holdout_return) and holdout_return > 0.0
        and holdout_trades >= MIN_HOLDOUT_TRADES
    )
    result.update(
        status="passed" if passed else "failed_holdout",
        holdout_sharpe=_finite_or_none(holdout_sharpe),
        holdout_total_return=_finite_or_none(holdout_return),
        holdout_completed_trades=holdout_trades,
        holdout_mean_holding_sessions=_finite_or_none(holdout["mean_holding_sessions"][0]),
        holdout_max_holding_sessions=int(holdout["max_holding_sessions"][0]),
    )
    return result
