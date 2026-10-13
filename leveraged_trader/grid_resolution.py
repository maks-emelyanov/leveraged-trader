"""Read-only chronological grid-resolution research using production selection v2.

Each outer window freezes its parameters using only the preceding observations.
Raw development simulations are shared; neighborhoods remain specific to each
grid. Inner holdouts simulate only the already selected pair. Cash outcomes and
unsuccessful selections remain in the returned comparison sample.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np

from .grid_research import (
    ARRAY_NAMES,
    BASE_COST,
    INITIAL_CAPITAL,
    METRIC_NAMES,
    STRESS_COST,
    GridSpec,
    _scalar_metrics,
    _utility,
)
from .optimized_backtest import RSI_ENTRY_LOWER, RSI_ENTRY_UPPER
from .research_metrics import grid_research_metrics
from .research_selection import (
    DEVELOPMENT_BLOCKS,
    MIN_DEVELOPMENT_TRADES,
    TRADE_ACTIVITY_ANNUAL_SCALE,
    TRADE_ACTIVITY_WEIGHT,
    _neighbor_scores,
    development_trade_activity,
    select_strategy,
)

SELECTION_VERSION = 2


class _DevelopmentUnionMetrics:
    """Cache only shared development blocks, never an entire holdout grid."""

    def __init__(self, arrays: tuple[np.ndarray, ...], pairs: np.ndarray):
        self.arrays = arrays
        self.pairs = pairs
        self.lookup = {tuple(pair): i for i, pair in enumerate(pairs)}
        self.boundaries = np.linspace(0, len(arrays[0]) * 4 // 5, DEVELOPMENT_BLOCKS + 1, dtype=int)
        self.blocks = list(zip(self.boundaries[:-1], self.boundaries[1:], strict=True))
        self.cache: dict[tuple, dict[str, np.ndarray]] = {}

    def __call__(self, *args: object) -> dict[str, np.ndarray]:
        first, *_, buys, targets, capital, cost, rule = args
        first = np.asarray(first)
        base = self.arrays[0]
        if not np.shares_memory(first, base):
            raise ValueError("Cached evaluator requires slices of its original market arrays.")
        start = (first.ctypes.data - base.ctypes.data) // base.itemsize
        stop = start + len(first)
        if (start, stop) not in self.blocks:
            return grid_research_metrics(*args)
        key = (start, stop, float(capital), float(cost), int(rule))
        if key not in self.cache:
            self.cache[key] = grid_research_metrics(
                *(values[start:stop] for values in self.arrays), self.pairs[:, 0], self.pairs[:, 1],
                capital, cost, rule,
            )
        indices = [self.lookup[(buy, target)] for buy, target in zip(buys, targets, strict=True)]
        return {name: values[indices].copy() for name, values in self.cache[key].items()}

    def development_scores(self, rule: int) -> np.ndarray | None:
        """Repeat the fixed v2 formula for explanatory diagnostics only."""
        keys = [(start, stop, INITIAL_CAPITAL, BASE_COST, rule) for start, stop in self.blocks]
        if any(key not in self.cache for key in keys):
            return None
        blocks = [self.cache[key] for key in keys]
        sharpes = np.stack([block["sharpe"] for block in blocks])
        trades = np.stack([block["completed_trades"] for block in blocks])
        returns = np.stack([block["total_return"] for block in blocks])
        mean_holds = np.stack([block["mean_holding_sessions"] for block in blocks])
        maximum = np.max(np.stack([block["max_holding_sessions"] for block in blocks]), axis=0)
        scores = (
            np.median(sharpes, axis=0) - np.std(sharpes, axis=0)
            - 0.5 * np.log1p(np.mean(mean_holds, axis=0) / 21.0)
            - 0.25 * np.log1p(maximum / 63.0)
        )
        _, bonus = development_trade_activity(trades, np.diff(self.boundaries))
        eligible = (
            np.all(np.isfinite(sharpes), axis=0) & np.all(sharpes > 0.0, axis=0)
            & np.all(trades >= MIN_DEVELOPMENT_TRADES, axis=0) & np.all(returns > 0.0, axis=0)
            & np.all(np.isfinite(mean_holds), axis=0)
            & np.isfinite(scores) & (scores > 0.0) & np.isfinite(bonus)
        )
        scores += bonus
        scores[~eligible] = -np.inf
        return scores


def _activity_metrics(metrics: Mapping, sessions: int, prefix: str = "") -> dict:
    rate = 252.0 * metrics["completed_trades"] / sessions
    bonus = TRADE_ACTIVITY_WEIGHT * rate / (rate + TRADE_ACTIVITY_ANNUAL_SCALE)
    return {
        f"{prefix}annual_completed_trades": rate,
        f"{prefix}activity_bonus": bonus,
        f"{prefix}activity_utility": _utility(metrics) + bonus,
    }


def evaluate_windows(
    asset: str,
    signal: str,
    direction: str,
    dates: Sequence,
    arrays: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    *,
    grids: Sequence[GridSpec],
    windows: Sequence[tuple[object, str, str]],
    min_training_sessions: int = 630,
    min_test_sessions: int = 126,
) -> list[dict]:
    """Evaluate fixed grids on future windows with exclusive end boundaries.

    Every selection uses v2 and the production full-prefix Sharpe >= 1 gate.
    Original holding-adjusted ``utility`` remains the primary outer diagnostic;
    ``activity_utility`` additionally reports v2's bounded trade-frequency reward.
    Union-neighbor diagnostics are development-only and never alter selection.
    They describe a fine grid only when the supplied union is the fine grid.
    Selection timings include shared-cache work in grid order and are therefore
    not standalone production benchmarks.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    if dates.ndim != 1 or np.isnat(dates).any() or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("dates must be unique, increasing, and non-missing.")
    if isinstance(arrays, Mapping):
        arrays = tuple(arrays[name] for name in ARRAY_NAMES)
    arrays = tuple(np.ascontiguousarray(values, dtype=float) for values in arrays)
    if len(arrays) != 5 or any(values.ndim != 1 or len(values) != len(dates) for values in arrays):
        raise ValueError("Supply five one-dimensional market arrays aligned to dates.")
    if not grids or len({grid.name for grid in grids}) != len(grids):
        raise ValueError("grids must have nonempty, unique names.")
    if min_training_sessions < 1 or min_test_sessions < 1:
        raise ValueError("Require positive training and test session minimums.")
    if len({window[0] for window in windows}) != len(windows):
        raise ValueError("Require unique window labels.")
    parsed_windows = []
    for label, first, last in windows:
        first, last = np.datetime64(first, "D"), np.datetime64(last, "D")
        if np.isnat(first) or np.isnat(last) or first >= last:
            raise ValueError("Window start must precede its non-missing exclusive end.")
        parsed_windows.append((label, first, last))
    pairs_by_grid = {grid.name: grid.pairs(direction) for grid in grids}
    if any(pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) for pairs in pairs_by_grid.values()):
        raise ValueError("Each grid must contain parameter pairs.")
    union = np.unique(np.concatenate(list(pairs_by_grid.values())), axis=0)
    rule = RSI_ENTRY_LOWER if direction == "long" else RSI_ENTRY_UPPER
    rows = []
    for label, first, last in parsed_windows:
        start, stop = (int(np.searchsorted(dates, boundary)) for boundary in (first, last))
        if start < min_training_sessions or stop - start < min_test_sessions:
            continue
        training = tuple(values[:start] for values in arrays)
        evaluator = _DevelopmentUnionMetrics(training, union)
        decisions, timings = {}, {}
        for grid in grids:
            pairs = pairs_by_grid[grid.name]
            timer = perf_counter()
            decisions[grid.name] = select_strategy(
                *training, pairs[:, 0], pairs[:, 1], INITIAL_CAPITAL, BASE_COST, rule,
                metric_evaluator=evaluator, selection_version=SELECTION_VERSION,
            )
            timings[grid.name] = perf_counter() - timer
        raw_scores = evaluator.development_scores(rule)
        union_scores = None if raw_scores is None else _neighbor_scores(raw_scores, union[:, 0], union[:, 1])
        passed = sorted({
            (decision["buy_rsi"], decision["profit_target_multiple"])
            for decision in decisions.values() if decision["status"] == "passed"
        })
        full_metrics = {}
        if passed:
            paired = np.asarray(passed)
            metrics = grid_research_metrics(*training, paired[:, 0], paired[:, 1], INITIAL_CAPITAL, BASE_COST, rule)
            full_metrics = {pair: _scalar_metrics(metrics, i) for i, pair in enumerate(passed)}
        active_pairs = {
            pair for pair, metrics in full_metrics.items()
            if metrics["sharpe"] is not None and metrics["sharpe"] >= 1.0 and metrics["completed_trades"] >= 1
        }
        test_metrics = {}
        if active_pairs:
            paired = np.asarray(sorted(active_pairs))
            for cost in (BASE_COST, STRESS_COST):
                metrics = grid_research_metrics(
                    *(values[start:stop] for values in arrays), paired[:, 0], paired[:, 1],
                    INITIAL_CAPITAL, cost, rule,
                )
                for index, pair in enumerate(paired):
                    test_metrics[(tuple(pair), cost)] = _scalar_metrics(metrics, index)
        cash = dict.fromkeys(METRIC_NAMES, 0.0)
        cash["sharpe"] = None
        for grid in grids:
            decision = decisions[grid.name]
            pair = (decision["buy_rsi"], decision["profit_target_multiple"])
            active = decision["status"] == "passed" and pair in active_pairs
            base = test_metrics[(pair, BASE_COST)] if active else cash
            stress = test_metrics[(pair, STRESS_COST)] if active else cash
            diagnostics = {
                "training_eligible_candidates": None,
                "training_supported_candidates": None,
                "training_isolated_peak_rejections": None,
                "training_union_neighbor_score": None,
                "training_union_neighbor_supported": None,
            }
            if raw_scores is not None:
                pairs = pairs_by_grid[grid.name]
                indices = [evaluator.lookup[tuple(candidate)] for candidate in pairs]
                before = raw_scores[indices]
                after = _neighbor_scores(before, pairs[:, 0], pairs[:, 1])
                eligible = np.isfinite(before) & (before > 0)
                supported = np.isfinite(after) & (after > 0)
                diagnostics.update(
                    training_eligible_candidates=int(eligible.sum()),
                    training_supported_candidates=int(supported.sum()),
                    training_isolated_peak_rejections=int((eligible & ~supported).sum()),
                )
                if decision["selection_score"] is not None:
                    score = union_scores[evaluator.lookup[pair]]
                    diagnostics.update(
                        training_union_neighbor_score=float(score) if np.isfinite(score) else None,
                        training_union_neighbor_supported=bool(np.isfinite(score) and score > 0),
                    )
            rows.append({
                "asset": asset, "signal": signal, "direction": direction, "year": label, "grid": grid.name,
                "candidate_count": len(pairs_by_grid[grid.name]), "training_sessions": start,
                "training_end": str(dates[start - 1]), "test_start": str(dates[start]),
                "test_end": str(dates[stop - 1]), "test_sessions": stop - start,
                **{f"training_{key}": value for key, value in decision.items()},
                "training_full_sharpe": full_metrics.get(pair, {}).get("sharpe"),
                "activation_status": "active" if active else (
                    "failed_full_history_gate" if decision["status"] == "passed" else decision["status"]
                ),
                "buy_rsi": pair[0], "profit_target_multiple": pair[1], "active": active,
                "selection_seconds": timings[grid.name],
                "selection_timing_mode": "shared_development_cache_order_dependent",
                **diagnostics, **base, **{f"stress_{name}": value for name, value in stress.items()},
                "utility": _utility(base), "stress_utility": _utility(stress),
                **_activity_metrics(base, stop - start), **_activity_metrics(stress, stop - start, "stress_"),
            })
    return rows
