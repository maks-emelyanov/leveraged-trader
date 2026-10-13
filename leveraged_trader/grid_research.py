"""Offline, nested chronological comparison of a fixed set of parameter grids.

Supply adjusted market data and causal RSI; this module never fetches data or
writes trading state. Annual test periods start flat and mark open positions to
market. Failed training decisions remain in the comparison as cash outcomes.
The archived study keeps selection version 1 and its original outer utility,
independently of later changes to the production selection objective.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from .optimized_backtest import RSI_ENTRY_LOWER, RSI_ENTRY_UPPER
from .research_metrics import grid_research_metrics
from .research_selection import select_strategy

FROZEN_SELECTION_VERSION = 1
BASE_COST = 0.0003
STRESS_COST = 0.001
INITIAL_CAPITAL = 100_000.0
ARRAY_NAMES = ("open_prices", "high_prices", "close_prices", "rsi_values", "risk_free_returns")
METRIC_NAMES = (
    "sharpe", "total_return", "completed_trades", "mean_holding_sessions",
    "max_holding_sessions", "exposure_fraction", "open_holding_sessions",
)


@dataclass(frozen=True)
class GridSpec:
    name: str
    long_rsi: tuple[float, ...]
    short_rsi: tuple[float, ...]
    targets: tuple[float, ...]

    def pairs(self, direction: str) -> np.ndarray:
        """Return paired RSI/target rows, in a deterministic Cartesian order."""
        if direction not in {"long", "short", "inverse"}:
            raise ValueError("direction must be long, short, or inverse.")
        return np.asarray(list(product(self.long_rsi if direction == "long" else self.short_rsi, self.targets)))


_TARGETS = (1.02, 1.05, 1.10, 1.20, 1.30, 1.50, 2.00)
_LONG = tuple(range(20, 51, 5))
_SHORT = tuple(range(50, 81, 5))
DEFAULT_GRIDS = (
    GridSpec("current49", _LONG, _SHORT, _TARGETS),
    GridSpec("legacy1240", tuple(range(20, 51)), tuple(range(50, 81)), tuple(i / 10 for i in range(11, 51))),
    GridSpec("dense_rsi217", tuple(range(20, 51)), tuple(range(50, 81)), _TARGETS),
    GridSpec("coarse_rsi28", tuple(range(20, 51, 10)), tuple(range(50, 81, 10)), _TARGETS),
    GridSpec("fast_targets56", _LONG, _SHORT, (1.01, 1.02, 1.03, 1.05, 1.075, 1.10, 1.15, 1.20)),
    GridSpec("medium_targets42", _LONG, _SHORT, (1.05, 1.10, 1.20, 1.30, 1.50, 2.00)),
    GridSpec("expanded_rsi63", tuple(range(15, 56, 5)), tuple(range(45, 86, 5)), _TARGETS),
    GridSpec("narrow_rsi35", tuple(range(25, 46, 5)), tuple(range(55, 76, 5)), _TARGETS),
)


class _UnionMetrics:
    """Reuse raw simulations across grids, never their neighborhood scores."""

    def __init__(self, arrays: tuple[np.ndarray, ...], pairs: np.ndarray):
        self.arrays = arrays
        self.pairs = pairs
        self.lookup = {tuple(pair): i for i, pair in enumerate(pairs)}
        self.cache: dict[tuple[int, int, float, float, int], dict[str, np.ndarray]] = {}

    def __call__(self, *args: object) -> dict[str, np.ndarray]:
        first, *_, buys, targets, capital, cost, rule = args
        first = np.asarray(first)
        base = self.arrays[0]
        if not np.shares_memory(first, base):
            raise ValueError("Cached evaluator requires slices of its original market arrays.")
        start = (first.ctypes.data - base.ctypes.data) // base.itemsize
        stop = start + len(first)
        key = (start, stop, float(capital), float(cost), int(rule))
        if key not in self.cache:
            self.cache[key] = grid_research_metrics(
                *(values[start:stop] for values in self.arrays),
                self.pairs[:, 0], self.pairs[:, 1], capital, cost, rule,
            )
        indices = [self.lookup[(buy, target)] for buy, target in zip(buys, targets, strict=True)]
        return {name: values[indices].copy() for name, values in self.cache[key].items()}


def _scalar_metrics(metrics: dict[str, np.ndarray], index: int) -> dict[str, float | int | None]:
    result = {}
    for name in METRIC_NAMES:
        value = float(metrics[name][index])
        if not np.isfinite(value):
            if name not in {"sharpe", "mean_holding_sessions"}:
                raise ValueError(f"Non-finite test metric: {name}.")
            result[name] = None
        else:
            result[name] = int(value) if name.endswith("trades") or name.startswith(("max_", "open_")) else value
    return result


def _utility(metrics: dict[str, float | int | None]) -> float:
    if metrics["exposure_fraction"] == 0.0:
        if metrics["total_return"] != 0.0 or metrics["completed_trades"] != 0:
            raise ValueError("Zero-exposure test metrics must represent a cash outcome.")
        return 0.0
    sharpe, mean, maximum = (metrics[name] for name in ("sharpe", "mean_holding_sessions", "max_holding_sessions"))
    if any(value is None or not np.isfinite(value) for value in (sharpe, mean, maximum)):
        raise ValueError("An exposed test outcome requires finite Sharpe and holding metrics.")
    if mean < 1.0 or maximum < mean or not 0 < metrics["exposure_fraction"] <= 1:
        raise ValueError("Invalid test holding duration or exposure.")
    return float(sharpe - 0.5 * np.log1p(mean / 21.0) - 0.25 * np.log1p(maximum / 63.0))


def evaluate_asset(
    asset: str,
    signal: str,
    direction: str,
    dates: Sequence,
    arrays: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    *,
    grids: Sequence[GridSpec] = DEFAULT_GRIDS,
    validation_years: Sequence[int] = tuple(range(2017, 2023)),
    min_training_sessions: int = 1260,
) -> list[dict]:
    """Freeze each grid's training winner, then test the next calendar year.

    Only years with the required training prefix and at least 126 test sessions
    are eligible. Every grid gets every eligible asset-year, including failed
    training decisions. Base-cost selection is reused for the cost stress test.
    RSI must already be calculated causally, with its preceding warmup history.
    Selection version 1 preserves the archived study's registered objective.
    """
    if len(set(validation_years)) != len(validation_years):
        raise ValueError("Require unique validation years.")
    windows = [(year, f"{year}-01-01", f"{year + 1}-01-01") for year in validation_years]
    return _evaluate_windows(asset, signal, direction, dates, arrays, windows, grids, min_training_sessions)


def evaluate_period(
    asset: str,
    signal: str,
    direction: str,
    dates: Sequence,
    arrays: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    *,
    test_start: str,
    test_end: str,
    grids: Sequence[GridSpec] = DEFAULT_GRIDS,
    min_training_sessions: int = 1260,
) -> list[dict]:
    """Evaluate a version-1 frozen winner over a period with inclusive end."""
    first, last = np.datetime64(test_start, "D"), np.datetime64(test_end, "D")
    if np.isnat(first) or np.isnat(last) or first > last:
        raise ValueError("Require non-missing test_start <= test_end.")
    windows = [(int(str(first)[:4]), first, last + np.timedelta64(1, "D"))]
    return _evaluate_windows(asset, signal, direction, dates, arrays, windows, grids, min_training_sessions)


def _evaluate_windows(asset, signal, direction, dates, arrays, windows, grids, min_training_sessions):
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
    if min_training_sessions < 1:
        raise ValueError("Require positive training sessions.")
    pairs_by_grid = {grid.name: grid.pairs(direction) for grid in grids}
    union = np.unique(np.concatenate(list(pairs_by_grid.values())), axis=0)
    rule = RSI_ENTRY_LOWER if direction == "long" else RSI_ENTRY_UPPER
    rows = []
    for year, first, last_exclusive in windows:
        start = int(np.searchsorted(dates, np.datetime64(first, "D")))
        stop = int(np.searchsorted(dates, np.datetime64(last_exclusive, "D")))
        if start < min_training_sessions or stop - start < 126:
            continue
        training = tuple(values[:start] for values in arrays)
        evaluator = _UnionMetrics(training, union)
        decisions = {}
        for grid in grids:
            pairs = pairs_by_grid[grid.name]
            decisions[grid.name] = select_strategy(
                *training, pairs[:, 0], pairs[:, 1], INITIAL_CAPITAL, BASE_COST, rule,
                metric_evaluator=evaluator, selection_version=FROZEN_SELECTION_VERSION,
            )
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
                for i, pair in enumerate(paired):
                    test_metrics[(tuple(pair), cost)] = _scalar_metrics(metrics, i)
        cash = dict.fromkeys(METRIC_NAMES, 0.0)
        cash["sharpe"] = None
        for grid in grids:
            decision = decisions[grid.name]
            pair = (decision["buy_rsi"], decision["profit_target_multiple"])
            active = decision["status"] == "passed" and pair in active_pairs
            base = test_metrics[(pair, BASE_COST)] if active else cash
            stress = test_metrics[(pair, STRESS_COST)] if active else cash
            rows.append({
                "asset": asset, "signal": signal, "direction": direction, "year": int(year), "grid": grid.name,
                "candidate_count": len(pairs_by_grid[grid.name]), "training_sessions": start,
                "training_end": str(dates[start - 1]), "test_start": str(dates[start]),
                "test_end": str(dates[stop - 1]), "test_sessions": stop - start,
                "training_status": decision["status"], "training_selection_score": decision["selection_score"],
                "training_full_sharpe": full_metrics.get(pair, {}).get("sharpe"),
                "activation_status": "active" if active else (
                    "failed_full_history_gate" if decision["status"] == "passed" else decision["status"]
                ),
                "buy_rsi": pair[0], "profit_target_multiple": pair[1], "active": active,
                **base, **{f"stress_{name}": value for name, value in stress.items()},
                "utility": _utility(base), "stress_utility": _utility(stress),
            })
    return rows


def summarize(
    rows: Sequence[dict],
    *,
    baseline: str = "current49",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20261012,
    comparison_count: int = 7,
) -> dict:
    """Equal-weight signal proxies; bootstrap whole proxies, keeping years paired.

    Two-sided familywise 95% bounds use the 0.05 / (2 * comparison_count)
    bootstrap tail quantiles. The original family count remains fixed when
    evaluating a confirmation subset. Bounds describe evidence, not future gains.
    Cash decisions have utility zero and remain in every mean and paired test.
    """
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    if comparison_count < 1:
        raise ValueError("comparison_count must be positive.")
    if not rows:
        return {"grids": [], "directions": [], "paired": []}
    grid_names = sorted({row["grid"] for row in rows})
    if baseline not in grid_names:
        raise ValueError("The baseline grid must be present.")
    if comparison_count < len(grid_names) - 1:
        raise ValueError("comparison_count must cover all non-baseline grids.")
    keyed = {}
    for row in rows:
        key = (row["grid"], row["asset"], row["signal"], row["direction"], row["year"])
        if key in keyed:
            raise ValueError("Duplicate grid/asset/signal/direction/year outcome.")
        if any(not np.isfinite(row[name]) for name in ("utility", "stress_utility")):
            raise ValueError("Every eligible outcome must have finite utility.")
        keyed[key] = row
    baseline_keys = {key[1:] for key in keyed if key[0] == baseline}
    if any({key[1:] for key in keyed if key[0] == name} != baseline_keys for name in grid_names):
        raise ValueError("All grids must include exactly the same eligible asset-years, including cash.")

    fields = ("utility", "stress_utility", "total_return", "stress_total_return", "exposure_fraction")
    summaries, direction_summaries, comparisons = [], [], []
    for direction in ("all", *sorted({row["direction"] for row in rows})):
        subset = [row for row in rows if direction == "all" or row["direction"] == direction]
        for name in grid_names:
            selected = [row for row in subset if row["grid"] == name]
            groups = defaultdict(list)
            for row in selected:
                groups[row["signal"]].append(row)
            summary = {
                "grid": name, "direction": direction, "asset_years": len(selected), "signal_groups": len(groups),
                "active_asset_years": sum(row["active"] for row in selected),
                **{field: float(np.mean([
                    np.mean([row[field] for row in group]) for group in groups.values()
                ])) for field in fields},
                "activation_fraction": float(np.mean([
                    np.mean([row["active"] for row in group]) for group in groups.values()
                ])),
            }
            for field in ("mean_holding_sessions", "max_holding_sessions", "open_holding_sessions"):
                exposed_groups = [
                    [row[field] for row in group if row["exposure_fraction"] > 0]
                    for group in groups.values()
                ]
                means = [float(np.mean(group)) for group in exposed_groups if group]
                summary[f"active_{field}"] = float(np.mean(means)) if means else None
            sharpe_groups = [
                [row["sharpe"] for row in group if row["exposure_fraction"] > 0]
                for group in groups.values()
            ]
            summary["active_sharpe"] = (
                float(np.mean([np.mean(group) for group in sharpe_groups if group]))
                if any(sharpe_groups) else None
            )
            (summaries if direction == "all" else direction_summaries).append(summary)
            if name == baseline:
                continue
            deltas, holding_deltas = [], []
            matched_count = 0
            for signal in sorted(groups):
                group = groups[signal]
                deltas.append([
                    np.mean([
                        row[field] - keyed[(baseline, row["asset"], signal, row["direction"], row["year"])][field]
                        for row in group
                    ]) for field in ("utility", "stress_utility")
                ])
                matched = [
                    (row, keyed[(baseline, row["asset"], signal, row["direction"], row["year"])])
                    for row in group
                    if row["exposure_fraction"] > 0
                    and keyed[(baseline, row["asset"], signal, row["direction"], row["year"])]["exposure_fraction"] > 0
                ]
                if matched:
                    matched_count += len(matched)
                    holding_deltas.append([
                        np.mean([row[field] - reference[field] for row, reference in matched])
                        for field in ("mean_holding_sessions", "max_holding_sessions")
                    ])
            deltas = np.asarray(deltas)
            rng = np.random.default_rng(bootstrap_seed)
            indices = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
            bootstrap = deltas[indices].mean(axis=1)
            comparison = {
                "grid": name, "baseline": baseline, "direction": direction, "signal_groups": len(deltas),
                "bootstrap_seed": bootstrap_seed, "comparison_count": comparison_count,
                "matched_exposed_asset_years": matched_count, "matched_exposed_signal_groups": len(holding_deltas),
            }
            for i, field in enumerate(("mean_holding_sessions", "max_holding_sessions")):
                comparison[f"matched_exposed_delta_{field}"] = (
                    float(np.asarray(holding_deltas)[:, i].mean()) if holding_deltas else None
                )
            tail = 0.05 / (2 * comparison_count)
            for i, field in enumerate(("utility", "stress_utility")):
                discordant = deltas[:, i][np.abs(deltas[:, i]) > 1e-12]
                comparison.update({
                    f"delta_{field}": float(deltas[:, i].mean()),
                    f"lower_95_{field}": float(np.quantile(bootstrap[:, i], 0.025)),
                    f"upper_95_{field}": float(np.quantile(bootstrap[:, i], 0.975)),
                    f"familywise_lower_95_{field}": float(np.quantile(bootstrap[:, i], tail)),
                    f"familywise_upper_95_{field}": float(np.quantile(bootstrap[:, i], 1 - tail)),
                    f"discordant_groups_{field}": len(discordant),
                    f"improvement_share_{field}": float(np.mean(discordant > 0)) if len(discordant) else None,
                })
            comparisons.append(comparison)
    return {"grids": summaries, "directions": direction_summaries, "paired": comparisons}
