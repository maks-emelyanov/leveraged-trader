from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from functools import lru_cache

import numpy as np

from .backtest import _validate_risk_free_return_domain
from .pricing import target_sell_price

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without optional runtime dependency
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args: object, **kwargs: object) -> Callable:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func: Callable) -> Callable:
            return func

        return decorator


ACTION_NONE = 0
ACTION_BUY = 1
ACTION_SELL = 2

RSI_ENTRY_LOWER = 0
RSI_ENTRY_UPPER = 1
_TEMPORAL_SCALAR_TYPES = (date, datetime, timedelta, np.datetime64, np.timedelta64)
_STATE_RELATIVE_TOLERANCE = 1e-12
_STATE_ABSOLUTE_TOLERANCE = 1e-9
_MOMENT_ROUNDOFF_FACTOR = 64.0 * np.finfo(np.float64).eps


def _unwrap_zero_dimensional_scalar(value: object) -> object:
    """Expose the semantic type hidden inside nested 0-D ndarray wrappers."""
    while isinstance(value, np.ndarray) and value.ndim == 0:
        # ``item()`` converts nanosecond datetime64/timedelta64 values to
        # integers. Indexing preserves the NumPy scalar's semantic dtype.
        value = value[()]
    return value


def _float_array(
    name: str,
    values: np.ndarray,
    *,
    length: int | None = None,
    allow_nan: bool = False,
) -> np.ndarray:
    try:
        source_values = np.asarray(values)
        raw_values = np.asarray(values, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric array.") from exc
    semantic_values = tuple(_unwrap_zero_dimensional_scalar(value) for value in raw_values.flat)
    if any(isinstance(value, (bool, np.bool_)) for value in semantic_values):
        raise ValueError(f"{name} must not contain booleans.")
    if any(isinstance(value, (complex, np.complexfloating)) for value in semantic_values):
        raise ValueError(f"{name} must not contain complex values.")
    if source_values.dtype.kind in {"M", "m"} or any(
        isinstance(value, _TEMPORAL_SCALAR_TYPES) for value in semantic_values
    ):
        raise ValueError(f"{name} must not contain date, datetime, or timedelta values.")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a one-dimensional numeric array.") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must contain {length} values; got {len(array)}.")
    if allow_nan:
        if np.isinf(array).any():
            raise ValueError(f"{name} must not contain infinite values.")
    elif not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array)


def _integer_array(name: str, values: np.ndarray, *, length: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must contain {length} values; got {len(array)}.")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain integers.")
    return np.ascontiguousarray(array, dtype=np.int64)


def _boolean_array(name: str, values: np.ndarray, *, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if len(array) != length:
        raise ValueError(f"{name} must contain {length} values; got {len(array)}.")
    if array.dtype.kind != "b":
        raise ValueError(f"{name} must contain booleans.")
    return np.ascontiguousarray(array, dtype=np.bool_)


def _finite_scalar(name: str, value: float) -> float:
    value = _unwrap_zero_dimensional_scalar(value)
    if isinstance(
        value,
        (*_TEMPORAL_SCALAR_TYPES, bool, np.bool_, complex, np.complexfloating, np.ndarray),
    ):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _rsi_threshold_array(name: str, values: np.ndarray) -> np.ndarray:
    raw_values = np.asarray(values)
    contains_boolean = raw_values.dtype.kind == "b"
    if raw_values.dtype.kind == "O":
        contains_boolean = any(isinstance(value, (bool, np.bool_)) for value in raw_values.flat)
    elif not isinstance(values, np.ndarray):
        try:
            contains_boolean = any(isinstance(value, (bool, np.bool_)) for value in values)
        except TypeError:
            contains_boolean = isinstance(values, (bool, np.bool_))
    if contains_boolean:
        raise ValueError(f"{name} must not contain booleans.")

    array = _float_array(name, values)
    if np.any((array < 0.0) | (array > 100.0)):
        raise ValueError(f"{name} must contain values between 0.0 and 100.0, inclusive.")
    return array


def _rsi_threshold_scalar(name: str, value: float) -> float:
    value = _unwrap_zero_dimensional_scalar(value)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not be a boolean.")
    result = _finite_scalar(name, value)
    if not 0.0 <= result <= 100.0:
        raise ValueError(f"{name} must be between 0.0 and 100.0, inclusive.")
    return result


def _rsi_observation_array(values: np.ndarray, *, length: int) -> np.ndarray:
    array = _float_array("rsi_values", values, length=length, allow_nan=True)
    invalid = (~np.isnan(array)) & ((array < 0.0) | (array > 100.0))
    if invalid.any():
        row_idx = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            "rsi_values must contain values between 0.0 and 100.0, inclusive, or NaN; "
            f"got {array[row_idx]!r} at index {row_idx}."
        )
    return array


def _validate_rsi_entry_rule(value: int) -> int:
    value = _unwrap_zero_dimensional_scalar(value)
    if (
        isinstance(value, (*_TEMPORAL_SCALAR_TYPES, bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value not in {RSI_ENTRY_LOWER, RSI_ENTRY_UPPER}
    ):
        raise ValueError(
            f"rsi_entry_rule must be RSI_ENTRY_LOWER ({RSI_ENTRY_LOWER}) or "
            f"RSI_ENTRY_UPPER ({RSI_ENTRY_UPPER}); got {value!r}."
        )
    return int(value)


def _validate_simulation_prices(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    close_prices: np.ndarray,
) -> None:
    """Enforce the OHLC invariants available to the optimized kernels."""
    for name, values in (
        ("open_prices", open_prices),
        ("high_prices", high_prices),
        ("close_prices", close_prices),
    ):
        invalid = values <= 0.0
        if invalid.any():
            row_idx = int(np.flatnonzero(invalid)[0])
            raise ValueError(f"{name} must contain positive values; got {values[row_idx]!r} at index {row_idx}.")
    invalid_high = high_prices < np.maximum(open_prices, close_prices)
    if invalid_high.any():
        row_idx = int(np.flatnonzero(invalid_high)[0])
        raise ValueError(
            "high_prices must be at least the corresponding open_prices and close_prices; "
            f"invariant failed at index {row_idx}."
        )


def _entry_candidate_indices(
    rsi_values: np.ndarray,
    buy_rsi: float,
    start_idx: int,
    in_position: bool,
    pending_action: int,
    rsi_entry_rule: int,
) -> np.ndarray:
    """Return rows on which the supplied signals could execute an entry."""
    row_count = len(rsi_values)
    if start_idx >= row_count:
        return np.empty(0, dtype=np.int64)

    pending_entry = bool(not in_position and pending_action == ACTION_BUY)
    signal_values = rsi_values[start_idx : row_count - 1]
    if rsi_entry_rule == RSI_ENTRY_UPPER:
        signal_entries = np.flatnonzero(signal_values >= buy_rsi) + start_idx + 1
    else:
        signal_entries = np.flatnonzero(signal_values <= buy_rsi) + start_idx + 1
    if not pending_entry:
        return np.ascontiguousarray(signal_entries, dtype=np.int64)
    return np.concatenate((np.array([start_idx], dtype=np.int64), signal_entries.astype(np.int64, copy=False)))


def _grid_entry_candidate_indices(
    rsi_values: np.ndarray,
    buy_rsi_values: np.ndarray,
    start_indices: np.ndarray,
    in_position_values: np.ndarray,
    pending_action_values: np.ndarray,
    rsi_entry_rule: int,
) -> tuple[np.ndarray, ...]:
    """Build reusable signal-aware entry candidates for grid preflight checks."""
    cache: dict[tuple[float, int, bool, int], np.ndarray] = {}
    candidates: list[np.ndarray] = []
    for buy_rsi, start_idx, in_position, pending_action in zip(
        buy_rsi_values,
        start_indices,
        in_position_values,
        pending_action_values,
        strict=True,
    ):
        key = (float(buy_rsi), int(start_idx), bool(in_position), int(pending_action))
        if key not in cache:
            cache[key] = _entry_candidate_indices(
                rsi_values,
                key[0],
                key[1],
                key[2],
                key[3],
                rsi_entry_rule,
            )
        candidates.append(cache[key])
    return tuple(candidates)


def _require_finite_result(name: str, values: object, *, allow_nan: bool = False) -> None:
    array = np.asarray(values, dtype=np.float64)
    invalid = np.isinf(array) if allow_nan else ~np.isfinite(array)
    if np.any(invalid):
        raise ValueError(f"Backtest produced non-finite {name}.")


def _validate_grid_results(results: tuple) -> None:
    updated = results[0]
    for name, values, allow_nan in (
        ("cash", results[1], False),
        ("shares", results[2], False),
        ("entry prices", results[4], True),
        ("previous equity", results[6], False),
        ("first equity", results[8], True),
        ("last equity", results[9], True),
        ("running maximum equity", results[10], True),
        ("return sums", results[12], False),
        ("return square sums", results[13], False),
        ("excess-return sums", results[15], False),
        ("excess-return square sums", results[16], False),
        ("maximum drawdown", results[18], True),
        ("return means", results[19], False),
        ("return centered square sums", results[20], False),
        ("excess-return means", results[21], False),
        ("excess-return centered square sums", results[22], False),
    ):
        _require_finite_result(name, values, allow_nan=allow_nan)

    if np.any(results[1] < 0.0) or np.any(results[2] < 0.0):
        raise ValueError("Backtest produced negative cash or shares.")
    for name, values in (
        ("trade count", results[7]),
        ("return count", results[11]),
        ("excess-return count", results[14]),
        ("positive-return count", results[17]),
    ):
        if np.any(values < 0):
            raise ValueError(f"Backtest produced an invalid {name}.")
    if np.any(results[6] <= 0.0):
        raise ValueError("Backtest produced non-positive equity.")
    for name, values in (
        ("first equity", results[8]),
        ("last equity", results[9]),
        ("running maximum equity", results[10]),
    ):
        if np.any(~np.isfinite(values[updated])) or np.any(values[updated] <= 0.0):
            raise ValueError(f"Backtest produced invalid {name}.")
    if np.any(~np.isfinite(results[18][updated])):
        raise ValueError("Backtest produced invalid maximum drawdown.")
    if np.any(results[20] < 0.0) or np.any(results[22] < 0.0):
        raise ValueError("Backtest produced negative centered square sums.")


def _validate_return_rollup_inputs(
    daily_returns: np.ndarray,
    risk_free_returns: np.ndarray,
) -> None:
    """Match the persisted rollup's arithmetic before it can overflow."""
    strategy_returns = daily_returns[1:]
    finite_risk_free = ~np.isnan(risk_free_returns[1:])
    with np.errstate(over="ignore", invalid="ignore"):
        excess_returns = strategy_returns[finite_risk_free] - risk_free_returns[1:][finite_risk_free]
        aggregates = (
            np.sum(strategy_returns),
            np.sum(strategy_returns * strategy_returns),
            np.sum(excess_returns),
            np.sum(excess_returns * excess_returns),
        )
    if not np.isfinite(aggregates).all():
        raise ValueError("Backtest returns cannot be summarized with finite float64 rollups.")


def _state_values_are_close(first: float, second: float) -> bool:
    """Compare values derived from the same float64 observation path."""
    if not np.isfinite(first) or not np.isfinite(second):
        return False
    tolerance = _float64_roundoff_tolerance(first, second)
    return bool(abs(np.longdouble(first) - np.longdouble(second)) <= tolerance)


def _float64_inward_ulp(value: float | np.longdouble) -> float:
    """Return one finite float64 step toward zero, including at DBL_MAX."""
    with np.errstate(over="ignore", invalid="ignore"):
        numeric = np.float64(value)
    if not np.isfinite(numeric):
        return 0.0
    return abs(float(numeric - np.nextafter(numeric, np.float64(0.0))))


def _float64_roundoff_tolerance(
    first: float | np.longdouble,
    second: float | np.longdouble,
) -> np.longdouble:
    """Bound ULP slack by the relative error of finite float64 arithmetic."""
    first_extended = np.longdouble(first)
    second_extended = np.longdouble(second)
    ulp_tolerance = np.longdouble(32.0) * max(
        np.longdouble(_float64_inward_ulp(first_extended)),
        np.longdouble(_float64_inward_ulp(second_extended)),
    )
    relative_tolerance = (
        np.longdouble(32.0) * np.longdouble(np.finfo(np.float64).eps) * max(abs(first_extended), abs(second_extended))
    )
    return min(ulp_tolerance, relative_tolerance)


def _account_state_values_are_close(first: float, second: float) -> bool:
    """Compare account values using only their finite float64 roundoff budget."""
    if not np.isfinite(first) or not np.isfinite(second):
        return False
    tolerance = _float64_roundoff_tolerance(first, second)
    return bool(abs(np.longdouble(first) - np.longdouble(second)) <= tolerance)


def _extended_account_values_are_close(
    first: np.longdouble,
    second: np.longdouble,
) -> bool:
    """Compare reconstructed account values without a unit-scale floor."""
    if not np.isfinite(first) or not np.isfinite(second):
        return False
    tolerance = _float64_roundoff_tolerance(first, second)
    if not np.isfinite(tolerance):
        return False
    return bool(abs(first - second) <= tolerance)


def _validate_moment_accumulator(
    *,
    config_idx: int,
    label: str,
    count: int,
    total: float,
    total_squares: float,
) -> None:
    if total_squares < 0.0:
        raise ValueError(f"{label}_sum_squares_values must be non-negative at config index {config_idx}.")
    if count == 0:
        # These accumulators are initialized to literal zero and cannot have
        # acquired rounding error before their first observation.  A fixed
        # absolute tolerance would let a no-work resume preserve injected
        # nonzero moments indefinitely.
        if total != 0.0 or total_squares != 0.0:
            raise ValueError(f"Zero {label} count requires zero sum and square sum at config index {config_idx}.")
        return

    # Cauchy-Schwarz requires |sum(x)| <= sqrt(n) * sqrt(sum(x**2)).
    # Sequential float64 accumulation can miss that boundary by a few ulps;
    # scale the allowance with the number of accumulated observations while
    # capping it well below a material contradiction.
    bound = float(np.sqrt(float(count)) * np.sqrt(total_squares))
    relative_tolerance = min(1e-8, _MOMENT_ROUNDOFF_FACTOR * count)
    tolerance = _MOMENT_ROUNDOFF_FACTOR + relative_tolerance * max(
        abs(total),
        bound,
    )
    if abs(total) > bound + tolerance:
        raise ValueError(f"{label} sum and square sum are mathematically inconsistent at config index {config_idx}.")


def _strategy_return_moments_respect_lower_bound(
    count: int,
    total: float,
    total_squares: float,
) -> bool:
    """Return whether aggregate moments can describe returns no lower than -100%."""
    if count <= 0:
        return True

    count_extended = np.longdouble(count)
    total_extended = np.longdouble(total)
    squares_extended = np.longdouble(total_squares)
    shifted_total = total_extended + count_extended
    relative_tolerance = min(1e-8, _MOMENT_ROUNDOFF_FACTOR * count)
    shifted_tolerance = np.longdouble(_STATE_ABSOLUTE_TOLERANCE) + np.longdouble(relative_tolerance) * max(
        abs(total_extended), count_extended, np.longdouble(1.0)
    )
    if shifted_total < -shifted_tolerance:
        return False

    # For y_i = return_i + 1 >= 0 with fixed sum Y, sum(y_i**2) is
    # at most Y**2. Translating that bound back to raw return squares
    # catches cases such as [-2, 2], whose mean alone looks admissible.
    nonnegative_shifted_total = max(shifted_total, np.longdouble(0.0))
    maximum_squares = (
        nonnegative_shifted_total * nonnegative_shifted_total
        - np.longdouble(2.0) * nonnegative_shifted_total
        + count_extended
    )
    squares_tolerance = np.longdouble(_MOMENT_ROUNDOFF_FACTOR) + np.longdouble(relative_tolerance) * max(
        abs(squares_extended),
        abs(maximum_squares),
        np.longdouble(1.0),
    )
    return bool(squares_extended <= maximum_squares + squares_tolerance)


def _aggregate_roundoff_allowance(
    count: int,
    *moment_pairs: tuple[float, float],
) -> np.longdouble:
    """Conservatively bound sequential float64 sum and observation roundoff."""
    count_extended = np.longdouble(max(count, 1))
    magnitude = np.longdouble(1.0) + count_extended
    for total, total_squares in moment_pairs:
        total_extended = np.longdouble(total)
        squares_extended = np.longdouble(total_squares)
        with np.errstate(over="ignore", invalid="ignore"):
            absolute_sum_bound = np.sqrt(count_extended) * np.sqrt(max(squares_extended, np.longdouble(0.0)))
        magnitude += abs(total_extended) + absolute_sum_bound
    with np.errstate(over="ignore", invalid="ignore"):
        return np.longdouble(4096.0) * np.longdouble(np.finfo(np.float64).eps) * count_extended * magnitude


def _centered_moment_roundoff_allowance(
    count: int,
    *values: float | np.longdouble,
) -> np.longdouble:
    """Conservatively cover float64 error in independently accumulated moments."""
    scale = max(
        (abs(np.longdouble(value)) for value in values),
        default=np.longdouble(1.0),
    )
    scale = max(scale, np.longdouble(1.0))
    with np.errstate(over="ignore", invalid="ignore"):
        return np.longdouble(8192.0) * np.longdouble(np.finfo(np.float64).eps) * np.longdouble(max(count, 1)) * scale


def _return_rollup_endpoint_growth_is_feasible(
    *,
    count: int,
    first_equity: float,
    last_equity: float,
    return_sum: float,
    return_sum_squares: float,
    return_mean: float | None = None,
    return_m2: float | None = None,
) -> bool:
    """Apply moment-implied upper and lower bounds on endpoint growth."""
    if count <= 0:
        return True
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        observed_log_growth = np.log(np.longdouble(last_equity)) - np.log(np.longdouble(first_equity))
    if not np.isfinite(observed_log_growth):
        return False

    # AM-GM bounds the product of the exact positive equity ratios by their
    # arithmetic mean. The kernel persists float64-rounded versions of q_i - 1
    # and then accumulates those returns sequentially. The moment-derived
    # allowance covers both operations without reconstructing a path.
    return_sum_upper_bound = np.longdouble(return_sum) + _aggregate_roundoff_allowance(
        count,
        (return_sum, return_sum_squares),
    )
    mean_return_upper_bound = return_sum_upper_bound / np.longdouble(count)
    if mean_return_upper_bound <= -1.0:
        return False
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        maximum_log_growth = np.longdouble(count) * np.log1p(mean_return_upper_bound)
    if observed_log_growth > maximum_log_growth:
        return False

    # Persisted sums, Welford moments, and division-derived returns are
    # independently rounded. Build one conservative centered-moment envelope
    # for both refinements below rather than reconstructing an ordered path.
    count_extended = np.longdouble(count)
    persisted_mean = np.longdouble(return_sum) / count_extended if return_mean is None else np.longdouble(return_mean)
    if return_m2 is None:
        persisted_m2 = max(
            np.longdouble(return_sum_squares)
            - (np.longdouble(return_sum) * np.longdouble(return_sum) / count_extended),
            np.longdouble(0.0),
        )
    else:
        persisted_m2 = np.longdouble(return_m2)
    centered_roundoff = _centered_moment_roundoff_allowance(
        count,
        return_sum,
        return_sum_squares,
        persisted_mean,
        persisted_m2,
    )
    per_return_roundoff = (
        _aggregate_roundoff_allowance(
            count,
            (return_sum, return_sum_squares),
        )
        / count_extended
    )
    m2_lower_bound = max(persisted_m2 - centered_roundoff, np.longdouble(0.0))
    m2_upper_bound = max(persisted_m2, np.longdouble(0.0)) + centered_roundoff

    # Cartwright-Field sharpens AM-GM when the return variance is nonzero:
    # G <= A - Var / (2 * max(gross_return)). Bound the unknown maximum from
    # the same centered moments. Shifting every rounded gross return upward by
    # the per-observation allowance preserves M2 and also bounds the exact
    # equity ratios componentwise, so the resulting geometric-mean bound
    # remains conservative for the telescoping endpoint ratio.
    gross_mean_upper_bound = np.longdouble(1.0) + mean_return_upper_bound + per_return_roundoff
    with np.errstate(invalid="ignore", over="ignore"):
        maximum_gross_return = gross_mean_upper_bound + np.sqrt(
            m2_upper_bound * np.longdouble(count - 1) / count_extended
        )
    if maximum_gross_return <= 0.0:
        return False
    refined_geometric_mean_upper_bound = gross_mean_upper_bound - (
        (m2_lower_bound / count_extended) / (np.longdouble(2.0) * maximum_gross_return)
    )
    if refined_geometric_mean_upper_bound <= 0.0:
        return False
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        refined_maximum_log_growth = count_extended * np.log1p(refined_geometric_mean_upper_bound - np.longdouble(1.0))
    if observed_log_growth > refined_maximum_log_growth:
        return False

    # With fixed positive-vector mean and M2, the largest product has one
    # high observation and n - 1 equal low observations. Mean increases and
    # variance decreases can only raise that extremum, so the conservative
    # interval endpoints above give a still tighter necessary upper bound.
    if count > 1:
        maximum_product_low_gross = gross_mean_upper_bound - np.sqrt(
            m2_lower_bound / (count_extended * np.longdouble(count - 1))
        )
        maximum_product_high_gross = gross_mean_upper_bound + np.sqrt(
            m2_lower_bound * np.longdouble(count - 1) / count_extended
        )
        if maximum_product_low_gross > 0.0:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                tight_maximum_log_growth = np.log1p(maximum_product_high_gross - np.longdouble(1.0)) + np.longdouble(
                    count - 1
                ) * np.log1p(maximum_product_low_gross - np.longdouble(1.0))
            if observed_log_growth > tight_maximum_log_growth:
                return False

    # A centered n-vector bounds its minimum by
    #   mean(r) - sqrt(M2 * (n - 1) / n).
    # Shift every rounded gross return downward by the per-observation
    # allowance. The exact equity ratios then dominate this positive vector
    # componentwise, so both its minimum and the lower Cartwright-Field bound
    #   G >= A - Var / (2 * min(gross_return))
    # are conservative lower bounds for endpoint growth.
    mean_lower_bound = (
        min(
            persisted_mean,
            np.longdouble(return_sum) / count_extended,
        )
        - per_return_roundoff
    )
    with np.errstate(invalid="ignore", over="ignore"):
        maximum_downward_deviation = np.sqrt(m2_upper_bound * np.longdouble(count - 1) / count_extended)
    gross_mean_lower_bound = np.longdouble(1.0) + mean_lower_bound - per_return_roundoff
    minimum_gross_return = gross_mean_lower_bound - maximum_downward_deviation
    if minimum_gross_return <= 0.0:
        return True
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        minimum_log_growth = count_extended * np.log1p(minimum_gross_return - np.longdouble(1.0))
    refined_geometric_mean_lower_bound = gross_mean_lower_bound - (
        (m2_upper_bound / count_extended) / (np.longdouble(2.0) * minimum_gross_return)
    )
    if refined_geometric_mean_lower_bound > 0.0:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            refined_minimum_log_growth = count_extended * np.log1p(
                refined_geometric_mean_lower_bound - np.longdouble(1.0)
            )
        minimum_log_growth = max(minimum_log_growth, refined_minimum_log_growth)
    # The companion fixed-moment minimum has one low observation and n - 1
    # equal high observations. Evaluating it at the lower mean and upper M2
    # endpoints safely tightens the lower product envelope.
    if count > 1:
        minimum_product_high_gross = gross_mean_lower_bound + np.sqrt(
            m2_upper_bound / (count_extended * np.longdouble(count - 1))
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            tight_minimum_log_growth = np.log1p(minimum_gross_return - np.longdouble(1.0)) + np.longdouble(
                count - 1
            ) * np.log1p(minimum_product_high_gross - np.longdouble(1.0))
        minimum_log_growth = max(minimum_log_growth, tight_minimum_log_growth)
    return bool(observed_log_growth >= minimum_log_growth)


def _excess_return_moments_respect_risk_free_domain(
    *,
    return_count: int,
    return_sum: float,
    return_sum_squares: float,
    return_mean: float,
    return_m2: float,
    excess_return_count: int,
    excess_return_sum: float,
    excess_return_sum_squares: float,
    excess_return_mean: float,
    excess_return_m2: float,
    positive_return_count: int,
) -> bool:
    """Check moment envelopes implied by valid risk-free returns."""
    if excess_return_count <= 0:
        return True
    if return_count <= 0 or excess_return_count > return_count:
        return False

    strategy_count = np.longdouble(return_count)
    benchmark_count = np.longdouble(excess_return_count)
    strategy_aggregate_roundoff = _aggregate_roundoff_allowance(
        return_count,
        (return_sum, return_sum_squares),
    )
    excess_aggregate_roundoff = _aggregate_roundoff_allowance(
        excess_return_count,
        (excess_return_sum, excess_return_sum_squares),
    )
    strategy_centered_roundoff = _centered_moment_roundoff_allowance(
        return_count,
        return_sum,
        return_sum_squares,
        return_mean,
        return_m2,
    )
    excess_centered_roundoff = _centered_moment_roundoff_allowance(
        excess_return_count,
        excess_return_sum,
        excess_return_sum_squares,
        excess_return_mean,
        excess_return_m2,
    )

    # Full strategy moments bound every strategy return above and below its
    # mean. Thus each benchmarked excess return is below one plus the largest
    # feasible strategy return, even when only some strategy observations have
    # a finite benchmark. Apply the corresponding one-sided M2 envelope to the
    # excess-return subset.
    strategy_mean_upper_bound = max(
        np.longdouble(return_mean),
        np.longdouble(return_sum) / strategy_count,
    ) + (strategy_aggregate_roundoff / strategy_count)
    strategy_m2_upper_bound = max(
        np.longdouble(return_m2) + strategy_centered_roundoff,
        np.longdouble(0.0),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        strategy_maximum_upper_bound = strategy_mean_upper_bound + np.sqrt(
            strategy_m2_upper_bound * np.longdouble(return_count - 1) / strategy_count
        )
    observation_roundoff = max(
        strategy_aggregate_roundoff / strategy_count,
        excess_aggregate_roundoff / benchmark_count,
    )
    excess_upper_bound = np.longdouble(1.0) + strategy_maximum_upper_bound + observation_roundoff
    excess_mean_lower_bound = min(
        np.longdouble(excess_return_mean),
        np.longdouble(excess_return_sum) / benchmark_count,
    ) - (excess_aggregate_roundoff / benchmark_count)
    if excess_mean_lower_bound > excess_upper_bound:
        return False
    maximum_excess_mean_gap = max(
        excess_upper_bound - excess_mean_lower_bound,
        np.longdouble(0.0),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        maximum_excess_m2 = (
            benchmark_count * np.longdouble(excess_return_count - 1) * maximum_excess_mean_gap * maximum_excess_mean_gap
        )
    excess_m2_lower_bound = max(
        np.longdouble(excess_return_m2) - excess_centered_roundoff,
        np.longdouble(0.0),
    )
    if excess_m2_lower_bound > maximum_excess_m2:
        return False

    # Any excess return above one requires a positive strategy return because
    # excess = strategy - risk_free and risk_free > -1. For y = excess - 1,
    # Cauchy requires Q_y >= T_y**2 / k when T_y > 0 and at most k entries are
    # positive. If no positive strategy return exists, nonpositive y also
    # requires T_y <= 0 and Q_y <= T_y**2. Work with enclosing intervals for
    # both transformed moments so boundary-valid native states remain valid.
    if positive_return_count < 0:
        return False
    shifted_total_lower_bound = np.longdouble(excess_return_sum) - benchmark_count - excess_aggregate_roundoff
    shifted_total_upper_bound = np.longdouble(excess_return_sum) - benchmark_count + excess_aggregate_roundoff
    excess_m2_upper_bound = max(
        np.longdouble(excess_return_m2) + excess_centered_roundoff,
        np.longdouble(0.0),
    )
    minimum_shifted_total_magnitude = (
        np.longdouble(0.0)
        if shifted_total_lower_bound <= 0.0 <= shifted_total_upper_bound
        else min(abs(shifted_total_lower_bound), abs(shifted_total_upper_bound))
    )
    maximum_shifted_total_magnitude = max(
        abs(shifted_total_lower_bound),
        abs(shifted_total_upper_bound),
    )
    minimum_shifted_square_sum = excess_m2_lower_bound + (
        minimum_shifted_total_magnitude * minimum_shifted_total_magnitude / benchmark_count
    )
    maximum_shifted_square_sum = excess_m2_upper_bound + (
        maximum_shifted_total_magnitude * maximum_shifted_total_magnitude / benchmark_count
    )
    if positive_return_count == 0:
        if shifted_total_lower_bound > 0.0:
            return False
        maximum_nonpositive_square_sum = min(shifted_total_lower_bound, np.longdouble(0.0)) ** 2
        if minimum_shifted_square_sum > maximum_nonpositive_square_sum:
            return False
    elif (
        shifted_total_lower_bound > 0.0
        and maximum_shifted_square_sum * np.longdouble(positive_return_count)
        < shifted_total_lower_bound * shifted_total_lower_bound
    ):
        return False

    if excess_return_count != return_count:
        return True

    # excess_i = strategy_i - risk_free_i and every accepted risk-free return
    # is greater than -1, hence excess_i < strategy_i + 1.  Permit equality
    # within a conservative allowance because both sums are accumulated
    # independently and a benchmark may be the float immediately above -1.
    maximum_excess_sum = (
        np.longdouble(return_sum)
        + np.longdouble(return_count)
        + _aggregate_roundoff_allowance(
            return_count,
            (return_sum, return_sum_squares),
            (excess_return_sum, excess_return_sum_squares),
        )
    )
    if np.longdouble(excess_return_sum) > maximum_excess_sum:
        return False

    # Let d_i = excess_i - strategy_i = -risk_free_i. Every accepted
    # risk-free return is greater than -1, so d_i < 1. For an n-vector with
    # fixed mean mu_d and entries bounded above by one,
    #   M2_d <= n * (n - 1) * (1 - mu_d)**2.
    # The reverse triangle inequality supplies the independently necessary
    # lower bound ||d_c|| >= |sqrt(M2_e) - sqrt(M2_s)|.
    count_extended = strategy_count
    aggregate_roundoff = _aggregate_roundoff_allowance(
        return_count,
        (return_sum, return_sum_squares),
        (excess_return_sum, excess_return_sum_squares),
    )
    difference_sum_lower_bound = np.longdouble(excess_return_sum) - np.longdouble(return_sum) - aggregate_roundoff
    difference_mean_lower_bound = difference_sum_lower_bound / count_extended
    maximum_mean_gap = max(
        np.longdouble(1.0) - difference_mean_lower_bound,
        np.longdouble(0.0),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        maximum_difference_m2 = count_extended * np.longdouble(return_count - 1) * maximum_mean_gap * maximum_mean_gap

    centered_roundoff = _centered_moment_roundoff_allowance(
        return_count,
        return_sum,
        return_sum_squares,
        excess_return_sum,
        excess_return_sum_squares,
        return_m2,
        excess_return_m2,
    )
    strategy_m2_lower = max(np.longdouble(return_m2) - centered_roundoff, np.longdouble(0.0))
    strategy_m2_upper = max(np.longdouble(return_m2) + centered_roundoff, np.longdouble(0.0))
    excess_m2_lower = max(np.longdouble(excess_return_m2) - centered_roundoff, np.longdouble(0.0))
    excess_m2_upper = max(np.longdouble(excess_return_m2) + centered_roundoff, np.longdouble(0.0))
    with np.errstate(invalid="ignore", over="ignore"):
        minimum_centered_norm_difference = max(
            np.sqrt(excess_m2_lower) - np.sqrt(strategy_m2_upper),
            np.sqrt(strategy_m2_lower) - np.sqrt(excess_m2_upper),
            np.longdouble(0.0),
        )
        minimum_difference_m2 = minimum_centered_norm_difference * minimum_centered_norm_difference
    return bool(minimum_difference_m2 <= maximum_difference_m2)


def _positive_return_count_is_feasible(
    count: int,
    total: float,
    total_squares: float,
    positive_return_count: int,
) -> bool:
    """Check whether aggregate moments admit the persisted return signs."""
    if positive_return_count < 0 or positive_return_count > count:
        return False
    if count == 0:
        return positive_return_count == 0
    total_extended = np.longdouble(total)
    squares_extended = np.longdouble(total_squares)
    count_extended = np.longdouble(count)
    positive_extended = np.longdouble(positive_return_count)
    nonpositive_count = count - positive_return_count
    nonpositive_extended = np.longdouble(nonpositive_count)
    scale = max(
        abs(total_extended) * abs(total_extended),
        abs(squares_extended),
        count_extended,
        np.longdouble(1.0),
    )
    tolerance = np.longdouble(256.0 * np.finfo(np.float64).eps) * scale
    unit_return = np.longdouble(np.spacing(np.float64(1.0)))
    if not np.isfinite(total_extended) or not np.isfinite(squares_extended) or squares_extended < 0:
        return False
    if positive_return_count == 0 and (
        (total_extended == 0 and squares_extended != 0) or (total_extended < 0 and squares_extended <= 0)
    ):
        return False
    if positive_return_count == 0 and total_extended < 0:
        minimum_negative_square = (unit_return / np.longdouble(2.0)) ** 2
        if squares_extended < minimum_negative_square:
            return False
    if positive_return_count > 0:
        minimum_positive_squares = positive_extended * unit_return * unit_return
        if squares_extended < minimum_positive_squares:
            return False
    if positive_return_count == 0:
        negative_total = -total_extended
        if total_extended > 0 or negative_total > count_extended:
            return False
        clamped_negative = min(max(negative_total, np.longdouble(0.0)), count_extended)
        integral_negative = np.floor(clamped_negative)
        maximum_squares = integral_negative + (clamped_negative - integral_negative) ** 2
        minimum_squares = (total_extended * total_extended) / count_extended
    elif positive_return_count == count:
        if total_extended < positive_extended * unit_return:
            return False
        minimum_squares = (total_extended * total_extended) / count_extended
        maximum_squares = total_extended * total_extended
    else:
        if total_extended < -nonpositive_extended:
            return False
        denominator = positive_extended if total_extended >= 0 else nonpositive_extended
        minimum_squares = (total_extended * total_extended) / denominator
        maximum_squares = (total_extended + nonpositive_extended) ** 2 + nonpositive_extended
    return bool(squares_extended >= minimum_squares - tolerance and squares_extended <= maximum_squares + tolerance)


def _positive_return_path_is_feasible(
    *,
    count: int,
    positive_return_count: int,
    first_equity: float,
    last_equity: float,
    running_max_equity: float,
    max_drawdown: float,
    total: float,
) -> bool:
    """Check endpoint implications that follow from every return's sign."""
    if count <= 0:
        return True
    if positive_return_count == count:
        with np.errstate(over="ignore", invalid="ignore"):
            endpoint_return = np.longdouble(last_equity) / np.longdouble(first_equity) - 1
        endpoint_tolerance = (
            np.longdouble(256.0 * np.finfo(np.float64).eps)
            * max(abs(endpoint_return), abs(np.longdouble(total)), np.longdouble(1.0))
            * np.longdouble(count)
        )
        return bool(
            _float64_order_key(last_equity) - _float64_order_key(first_equity) >= count
            and running_max_equity == last_equity
            and max_drawdown == 0.0
            and endpoint_return + endpoint_tolerance >= np.longdouble(total)
        )
    if positive_return_count == 0:
        expected_last_drawdown = last_equity / first_equity - 1.0
        return bool(
            running_max_equity == first_equity
            and _state_values_are_close(max_drawdown, expected_last_drawdown)
            and (last_equity == first_equity if total == 0.0 else last_equity < first_equity)
        )
    return True


def _single_return_rollup_is_consistent(
    *,
    first_equity: float,
    last_equity: float,
    return_sum: float,
    return_sum_squares: float,
    return_mean: float,
    return_m2: float,
    positive_return_count: int,
) -> bool:
    """Validate the fully determined rollup for exactly one daily return."""

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        implied_return = last_equity / first_equity - 1.0
        implied_square = implied_return * implied_return
    if not np.isfinite(implied_return) or not np.isfinite(implied_square):
        return False

    def moment_is_close(observed: float, expected: float) -> bool:
        scale = max(abs(observed), abs(expected), 1.0)
        tolerance = 256.0 * np.finfo(np.float64).eps * scale
        return bool(abs(observed - expected) <= tolerance)

    return bool(
        positive_return_count == int(implied_return > 0.0)
        and moment_is_close(return_sum, implied_return)
        and moment_is_close(return_sum_squares, implied_square)
        and moment_is_close(return_mean, implied_return)
        and moment_is_close(return_m2, 0.0)
    )


def _return_rollup_equity_endpoints_are_consistent(
    *,
    count: int,
    first_equity: float,
    last_equity: float,
    return_sum: float,
    return_sum_squares: float,
    return_mean: float | None = None,
    return_m2: float | None = None,
) -> bool:
    """Validate endpoint identities fully determined by aggregate returns."""
    if not _return_rollup_endpoint_growth_is_feasible(
        count=count,
        first_equity=first_equity,
        last_equity=last_equity,
        return_sum=return_sum,
        return_sum_squares=return_sum_squares,
        return_mean=return_mean,
        return_m2=return_m2,
    ):
        return False
    constant_return = _constant_return_implied_by_moments(
        count=count,
        total=return_sum,
        total_squares=return_sum_squares,
        mean=return_mean,
        m2=return_m2,
    )
    if constant_return is not None:
        if _constant_return_endpoint_is_feasible(
            count=count,
            first_equity=first_equity,
            last_equity=last_equity,
            constant_return=constant_return,
        ):
            return True
        if count != 2:
            # Native zero-M2 histories can start with one adjacent float return
            # before repeating their rounded mean. Defer count>2 endpoint
            # decisions to the exact ordered-pattern validator.
            return True
    if count != 2:
        return True
    endpoint_reconstruction = _two_return_endpoint_reconstruction_bounds(
        first_equity=first_equity,
        last_equity=last_equity,
        return_sum=return_sum,
        return_sum_squares=return_sum_squares,
    )
    if endpoint_reconstruction is None:
        return False
    implied_last_equity, endpoint_tolerance = endpoint_reconstruction
    return bool(abs(implied_last_equity - np.longdouble(last_equity)) <= endpoint_tolerance)


def _extended_values_are_close(first: np.longdouble, second: np.longdouble) -> bool:
    """Compare extended-precision state values with the persisted-state tolerance."""
    if not np.isfinite(first) or not np.isfinite(second):
        return False
    scale = max(abs(first), abs(second), np.longdouble(1.0))
    tolerance = np.longdouble(_STATE_ABSOLUTE_TOLERANCE) + (np.longdouble(_STATE_RELATIVE_TOLERANCE) * scale)
    return bool(abs(first - second) <= tolerance)


def _constant_return_endpoint_is_feasible(
    *,
    count: int,
    first_equity: float,
    last_equity: float,
    constant_return: float,
) -> bool:
    """Check whether the endpoint's geometric gross replays one float return."""
    if count <= 0 or first_equity <= 0.0 or last_equity <= 0.0:
        return False
    endpoint_ratio = np.longdouble(last_equity) / np.longdouble(first_equity)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        geometric_gross = np.float_power(
            endpoint_ratio,
            np.longdouble(1.0) / np.longdouble(count),
        )
    if not np.isfinite(geometric_gross) or geometric_gross <= 0.0:
        return False
    # The geometric mean of gross returns stays inside the log-convex rounding
    # bucket that maps every observation to the same float64 daily return.
    # Probe adjacent results only to absorb final power-rounding at a boundary.
    return any(
        gross_return > 0.0 and gross_return - 1.0 == constant_return
        for gross_return in _adjacent_float64_values(float(geometric_gross), radius=2)
    )


def _constant_return_implied_by_moments(
    *,
    count: int,
    total: float,
    total_squares: float,
    mean: float | None,
    m2: float | None,
) -> float | None:
    """Return the sole daily return when the accumulated variance is roundoff-zero."""
    if count <= 0:
        return None

    count_extended = np.longdouble(count)
    implied_mean = np.longdouble(total) / count_extended
    observed_mean = implied_mean if mean is None else np.longdouble(mean)
    centered_moments_provided = mean is not None and m2 is not None
    observed_m2 = (
        np.longdouble(m2)
        if centered_moments_provided
        else np.longdouble(total_squares) - count_extended * implied_mean * implied_mean
    )
    if centered_moments_provided and (not np.isfinite(observed_m2) or observed_m2 != 0.0):
        # Welford's accumulator retains genuine variance far below the scale at
        # which independently accumulated raw sums can distinguish it from
        # cancellation. Exact repeated float returns produce an exact zero M2,
        # so never round a positive persisted M2 down to a deterministic path.
        return None
    expected_squares = count_extended * observed_mean * observed_mean
    observed_squares = np.longdouble(total_squares)
    moment_scale = max(
        abs(observed_squares),
        abs(expected_squares),
        np.longdouble(np.finfo(np.float64).tiny),
    )
    roundoff_factor = 256.0 if centered_moments_provided else 32.0
    relative_tolerance = np.longdouble(min(1e-10, roundoff_factor * np.finfo(np.float64).eps * count))
    tolerance = relative_tolerance * moment_scale
    if (
        abs(observed_mean - implied_mean) * count_extended > tolerance
        or abs(observed_squares - expected_squares) > tolerance
        or (not centered_moments_provided and abs(observed_m2) > tolerance)
    ):
        return None
    return float(observed_mean)


def _two_return_sequences(
    return_sum: float,
    return_sum_squares: float,
) -> tuple[tuple[float, float], ...] | None:
    """Recover both possible orderings of a two-return aggregate."""
    total = np.longdouble(return_sum)
    squares = np.longdouble(return_sum_squares)
    mean = total / np.longdouble(2.0)
    squared_offset = squares / np.longdouble(2.0) - mean * mean
    scale = max(abs(squares), abs(mean * mean), np.longdouble(np.finfo(np.float64).tiny))
    tolerance = np.longdouble(256.0 * np.finfo(np.float64).eps) * scale
    if squared_offset < -tolerance:
        return None
    offset = np.sqrt(max(squared_offset, np.longdouble(0.0)))
    lower = float(mean - offset)
    upper = float(mean + offset)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if lower == upper:
        return ((lower, upper),)
    return ((lower, upper), (upper, lower))


def _two_return_discriminant_is_ill_conditioned(
    return_sum: float,
    return_sum_squares: float,
) -> bool:
    """Whether raw moments cannot reliably resolve the roots' separation."""
    total = np.longdouble(return_sum)
    squares = np.longdouble(return_sum_squares)
    mean = total / np.longdouble(2.0)
    squared_offset = squares / np.longdouble(2.0) - mean * mean
    scale = max(abs(squares), abs(mean * mean), np.longdouble(np.finfo(np.float64).tiny))
    uncertainty = np.longdouble(256.0 * np.finfo(np.float64).eps) * scale
    return bool(abs(squared_offset) <= uncertainty)


def _two_return_gross_sequences(
    *,
    first_equity: float,
    last_equity: float,
    return_sum: float,
    return_sum_squares: float,
) -> (
    tuple[
        tuple[tuple[np.longdouble, np.longdouble], ...],
        np.longdouble,
        str,
        np.longdouble,
    ]
    | None
):
    """Recover both gross-return orderings without unstable near-loss subtraction.

    Raw sum and square-sum accumulators can determine a large return while
    losing every useful bit of the other root.  The independently persisted
    first/last equity ratio still determines the product of the two gross
    returns.  Use that product only when the moment roots are too
    ill-conditioned for the state's strict path tolerance, retaining the raw
    moment reconstruction for ordinary paths.
    """
    return_sequences = _two_return_sequences(return_sum, return_sum_squares)
    if return_sequences is None:
        return None

    root_scale = max(
        np.longdouble(1.0),
        *(abs(np.longdouble(value)) for sequence in return_sequences for value in sequence),
    )
    root_uncertainty = np.longdouble(512.0 * np.finfo(np.float64).eps) * root_scale
    discriminant_is_ill_conditioned = _two_return_discriminant_is_ill_conditioned(
        return_sum,
        return_sum_squares,
    )
    gross_sequences = tuple(
        tuple(np.longdouble(1.0) + np.longdouble(value) for value in sequence) for sequence in return_sequences
    )
    if discriminant_is_ill_conditioned:
        return gross_sequences, root_uncertainty, "centered", root_uncertainty

    # What matters to the compounded path is root error relative to each gross
    # multiplier, not absolute return magnitude. Large positive returns can
    # therefore remain perfectly well conditioned, while a root close to -1
    # needs endpoint conditioning even when its absolute error is tiny.
    minimum_gross = min(gross_return for sequence in gross_sequences for gross_return in sequence)
    if minimum_gross > 0.0 and root_uncertainty <= np.longdouble(_STATE_RELATIVE_TOLERANCE) * minimum_gross:
        return gross_sequences, root_uncertainty, "raw", root_uncertainty

    first_sequence = return_sequences[0]
    stable_return = max(first_sequence, key=abs)
    stable_gross = np.longdouble(1.0) + np.longdouble(stable_return)
    if not np.isfinite(stable_gross) or stable_gross <= 0.0:
        return None
    endpoint_ratio = np.longdouble(last_equity) / np.longdouble(first_equity)
    complementary_gross = endpoint_ratio / stable_gross
    if not np.isfinite(complementary_gross) or complementary_gross <= 0.0:
        return None

    if complementary_gross == stable_gross:
        gross_sequences = ((complementary_gross, stable_gross),)
    else:
        gross_sequences = (
            (complementary_gross, stable_gross),
            (stable_gross, complementary_gross),
        )
    # The endpoint-conditioned smaller multiplier no longer inherits the
    # absolute error of the very large moment root.  Its scale is the useful
    # bound for deciding whether either return's sign is ambiguous at one.
    unit_scale = max(
        np.longdouble(1.0),
        min(abs(stable_gross), abs(complementary_gross)),
    )
    unit_uncertainty = np.longdouble(512.0 * np.finfo(np.float64).eps) * unit_scale
    return gross_sequences, unit_uncertainty, "endpoint", root_uncertainty


def _gross_sequence_matches_native_centered_moments(
    gross_returns: tuple[np.longdouble, np.longdouble],
    *,
    return_sum: float,
    return_sum_squares: float,
    return_mean: float,
    return_m2: float,
) -> bool:
    """Replay every kernel return accumulator for one ordered gross path."""
    daily_returns = tuple(float(gross_return) - 1.0 for gross_return in gross_returns)
    return _daily_return_sequence_matches_native_centered_moments(
        daily_returns,
        return_sum=return_sum,
        return_sum_squares=return_sum_squares,
        return_mean=return_mean,
        return_m2=return_m2,
    )


def _daily_return_sequence_matches_native_centered_moments(
    daily_returns: tuple[float, float],
    *,
    return_sum: float,
    return_sum_squares: float,
    return_mean: float,
    return_m2: float,
) -> bool:
    """Replay every float64 accumulator for one ordered return pair."""
    total = 0.0
    total_squares = 0.0
    mean = 0.0
    m2 = 0.0
    for count, daily_return in enumerate(daily_returns, start=1):
        total += daily_return
        total_squares += daily_return * daily_return
        delta = daily_return - mean
        mean += delta / count
        m2 += delta * (daily_return - mean)
    return total == return_sum and total_squares == return_sum_squares and mean == return_mean and m2 == return_m2


def _gross_sequence_matches_raw_return_moments(
    gross_returns: tuple[np.longdouble, np.longdouble],
    *,
    return_sum: float,
    return_sum_squares: float,
) -> bool:
    """Replay the two sequential raw accumulators without inferring ordering."""
    daily_returns = tuple(float(gross_return) - 1.0 for gross_return in gross_returns)
    total = 0.0
    total_squares = 0.0
    for daily_return in daily_returns:
        total += daily_return
        total_squares += daily_return * daily_return
    return total == return_sum and total_squares == return_sum_squares


def _native_centered_two_return_roots(
    *,
    return_mean: float,
    return_m2: float,
) -> tuple[float, ...]:
    """Recover the two-return multiset from Welford fields.

    For two observations, M2 is half the squared separation. Unlike the raw
    square-sum discriminant, this reconstruction remains well-conditioned
    when the two returns are nearly equal.
    """
    mean = np.longdouble(return_mean)
    offset = np.sqrt(np.longdouble(return_m2) / np.longdouble(2.0))
    lower = float(mean - offset)
    upper = float(mean + offset)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return ()
    if lower == upper:
        return (lower,)
    return lower, upper


def _adjacent_float64_values(center: float, *, radius: int = 4) -> tuple[float, ...]:
    """Return a fixed-width finite ULP neighborhood around one float."""
    values = [center]
    lower = center
    upper = center
    for _ in range(radius):
        # Stepping outward from DBL_MAX legitimately reaches infinity. NumPy
        # reports that boundary as an overflow warning even though infinity is
        # only a search sentinel here, so contain it within this helper.
        with np.errstate(over="ignore", invalid="ignore"):
            lower = float(np.nextafter(lower, -np.inf))
            upper = float(np.nextafter(upper, np.inf))
        if np.isfinite(lower):
            values.append(lower)
        if np.isfinite(upper):
            values.append(upper)
    return tuple(values)


def _positive_float64_from_bits(bits: int) -> float:
    """Convert a positive finite IEEE-754 bit pattern to a Python float."""
    return float(np.array(bits, dtype=np.uint64).view(np.float64))


_FLOAT64_SIGN_BIT = 1 << 63
_FLOAT64_BIT_MASK = (1 << 64) - 1


def _float64_order_key(value: float) -> int:
    """Map one float64 to an integer key ordered by numeric value."""
    bits = int(np.array(value, dtype=np.float64).view(np.uint64))
    if bits & _FLOAT64_SIGN_BIT:
        return (~bits) & _FLOAT64_BIT_MASK
    return bits | _FLOAT64_SIGN_BIT


def _float64_from_order_key(key: int) -> float:
    """Invert ``_float64_order_key``."""
    bits = key & ~_FLOAT64_SIGN_BIT if key & _FLOAT64_SIGN_BIT else (~key) & _FLOAT64_BIT_MASK
    return float(np.array(bits, dtype=np.uint64).view(np.float64))


def _monotone_float64_exact_interval(
    function: Callable[[float], float],
    *,
    target: float,
    lower_key: int,
    upper_key: int,
    ascending: bool,
) -> tuple[int, int] | None:
    """Find the ordered-float key plateau on which a monotone function equals target."""
    lower = lower_key
    upper = upper_key
    while lower < upper:
        midpoint = (lower + upper) // 2
        value = function(_float64_from_order_key(midpoint))
        if (value < target) if ascending else (value > target):
            lower = midpoint + 1
        else:
            upper = midpoint
    first_key = lower
    if function(_float64_from_order_key(first_key)) != target:
        return None

    lower = first_key
    upper = upper_key
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        value = function(_float64_from_order_key(midpoint))
        if (value > target) if ascending else (value < target):
            upper = midpoint - 1
        else:
            lower = midpoint
    return first_key, lower


def _monotone_float64_threshold_key(
    function: Callable[[float], float],
    *,
    target: float,
    lower_key: int,
    upper_key: int,
    ascending: bool,
) -> int | None:
    """Find the first ordered-float key crossing one monotone threshold."""
    lower = lower_key
    upper = upper_key
    while lower < upper:
        midpoint = (lower + upper) // 2
        value = function(_float64_from_order_key(midpoint))
        if (value < target) if ascending else (value > target):
            lower = midpoint + 1
        else:
            upper = midpoint
    value = function(_float64_from_order_key(lower))
    if (value >= target) if ascending else (value <= target):
        return lower
    return None


def _exact_loss_companion_candidates(
    *,
    return_sum: float,
    return_sum_squares: float,
    anchors: tuple[float, ...],
) -> tuple[float, ...]:
    """Recover roots paired with -1 from exact raw-accumulator plateaus."""

    def accumulated_sum(companion: float) -> float:
        return companion + -1.0

    def accumulated_squares(companion: float) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.float64(companion) * np.float64(companion) + np.float64(1.0))

    domains = (
        (-1.0, float(np.nextafter(0.0, -np.inf)), False),
        (0.0, float(np.finfo(np.float64).max), True),
    )
    matches: list[float] = []
    for lower_value, upper_value, squares_ascending in domains:
        lower_key = _float64_order_key(lower_value)
        upper_key = _float64_order_key(upper_value)
        sum_interval = _monotone_float64_exact_interval(
            accumulated_sum,
            target=return_sum,
            lower_key=lower_key,
            upper_key=upper_key,
            ascending=True,
        )
        square_interval = _monotone_float64_exact_interval(
            accumulated_squares,
            target=return_sum_squares,
            lower_key=lower_key,
            upper_key=upper_key,
            ascending=squares_ascending,
        )
        if sum_interval is None or square_interval is None:
            continue
        intersection_lower = max(sum_interval[0], square_interval[0])
        intersection_upper = min(sum_interval[1], square_interval[1])
        if intersection_lower > intersection_upper:
            continue

        candidate_keys = {
            intersection_lower,
            intersection_upper,
            (intersection_lower + intersection_upper) // 2,
        }
        for anchor in anchors:
            if not np.isfinite(anchor):
                continue
            anchor_key = _float64_order_key(min(max(anchor, lower_value), upper_value))
            candidate_keys.add(min(max(anchor_key, intersection_lower), intersection_upper))
        for candidate_key in sorted(candidate_keys):
            candidate = _float64_from_order_key(candidate_key)
            if candidate not in matches:
                matches.append(candidate)
    return tuple(matches)


def _gross_rounding_bucket(daily_return: float) -> tuple[float, float] | None:
    """Find positive float gross bounds whose subtraction maps to one return."""
    minimum_bits = 1  # Smallest positive subnormal.
    maximum_bits = 0x7FEFFFFFFFFFFFFF  # Largest finite float64.

    lower_bits = minimum_bits
    upper_bits = maximum_bits
    while lower_bits < upper_bits:
        midpoint = (lower_bits + upper_bits) // 2
        if _positive_float64_from_bits(midpoint) - 1.0 < daily_return:
            lower_bits = midpoint + 1
        else:
            upper_bits = midpoint
    if _positive_float64_from_bits(lower_bits) - 1.0 != daily_return:
        return None

    first_bits = lower_bits
    upper_bits = maximum_bits
    while lower_bits < upper_bits:
        midpoint = (lower_bits + upper_bits + 1) // 2
        if _positive_float64_from_bits(midpoint) - 1.0 > daily_return:
            upper_bits = midpoint - 1
        else:
            lower_bits = midpoint
    return (
        _positive_float64_from_bits(first_bits),
        _positive_float64_from_bits(lower_bits),
    )


def _endpoint_conditioned_gross_sequences_for_returns(
    daily_returns: tuple[float, float],
    *,
    first_equity: float,
    last_equity: float,
) -> tuple[tuple[np.longdouble, np.longdouble], ...]:
    """Recover gross witnesses from endpoint product and return-rounding buckets."""
    first_bucket = _gross_rounding_bucket(daily_returns[0])
    second_bucket = _gross_rounding_bucket(daily_returns[1])
    if first_bucket is None or second_bucket is None:
        return ()

    endpoint_ratio = np.longdouble(last_equity) / np.longdouble(first_equity)
    lower = max(
        np.longdouble(first_bucket[0]),
        endpoint_ratio / np.longdouble(second_bucket[1]),
    )
    upper = min(
        np.longdouble(first_bucket[1]),
        endpoint_ratio / np.longdouble(second_bucket[0]),
    )
    if lower > upper:
        return ()

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        geometric_center = np.sqrt(lower) * np.sqrt(upper)
    seed_values = (
        float(lower),
        float(upper),
        float(geometric_center),
        float(1.0 + daily_returns[0]),
    )
    matches: list[tuple[np.longdouble, np.longdouble]] = []
    for seed in dict.fromkeys(seed_values):
        if not np.isfinite(seed) or seed <= 0.0:
            continue
        for first_gross in _adjacent_float64_values(seed, radius=2):
            if first_gross <= 0.0 or first_gross - 1.0 != daily_returns[0]:
                continue
            second_center = float(endpoint_ratio / np.longdouble(first_gross))
            if not np.isfinite(second_center) or second_center <= 0.0:
                continue
            for second_gross in _adjacent_float64_values(second_center, radius=2):
                if second_gross <= 0.0 or second_gross - 1.0 != daily_returns[1]:
                    continue
                candidate = (
                    np.longdouble(first_gross),
                    np.longdouble(second_gross),
                )
                if candidate not in matches:
                    matches.append(candidate)
    return tuple(matches)


def _endpoint_conditioned_sequences_matching_native_moments(
    gross_sequences: tuple[tuple[np.longdouble, np.longdouble], ...],
    *,
    first_equity: float,
    last_equity: float,
    native_roots: tuple[float, ...],
    return_sum: float,
    return_sum_squares: float,
    return_mean: float,
    return_m2: float,
    require_native_centered_moments: bool = True,
) -> Iterator[tuple[np.longdouble, np.longdouble]]:
    """Yield endpoint-conditioned gross pairs supported by every accumulator."""
    endpoint_ratio = np.longdouble(last_equity) / np.longdouble(first_equity)
    raw_root_centers = tuple(
        float(gross_return - np.longdouble(1.0)) for sequence in gross_sequences for gross_return in sequence
    )
    seen: set[tuple[np.longdouble, np.longdouble]] = set()
    root_centers = tuple(dict.fromkeys((*native_roots, *raw_root_centers)))
    for root_center in sorted(root_centers, key=abs, reverse=True):
        for candidate_return in _adjacent_float64_values(root_center, radius=8):
            gross_center = float(1.0 + candidate_return)
            if not np.isfinite(gross_center) or gross_center <= 0.0:
                continue
            for candidate_gross in _adjacent_float64_values(gross_center, radius=1):
                if candidate_gross <= 0.0:
                    continue
                complementary_center = float(endpoint_ratio / np.longdouble(candidate_gross))
                if not np.isfinite(complementary_center) or complementary_center <= 0.0:
                    continue
                for complementary_gross in _adjacent_float64_values(
                    complementary_center,
                    radius=2,
                ):
                    if complementary_gross <= 0.0:
                        continue
                    candidate_sequences = (
                        (
                            np.longdouble(complementary_gross),
                            np.longdouble(candidate_gross),
                        ),
                        (
                            np.longdouble(candidate_gross),
                            np.longdouble(complementary_gross),
                        ),
                    )
                    for candidate_sequence in candidate_sequences:
                        if candidate_sequence in seen:
                            continue
                        seen.add(candidate_sequence)
                        if (
                            _gross_sequence_matches_native_centered_moments(
                                candidate_sequence,
                                return_sum=return_sum,
                                return_sum_squares=return_sum_squares,
                                return_mean=return_mean,
                                return_m2=return_m2,
                            )
                            if require_native_centered_moments
                            else _gross_sequence_matches_raw_return_moments(
                                candidate_sequence,
                                return_sum=return_sum,
                                return_sum_squares=return_sum_squares,
                            )
                        ):
                            yield candidate_sequence


def _centered_sequences_matching_native_moments(
    *,
    return_sum: float,
    return_sum_squares: float,
    return_mean: float,
    return_m2: float,
) -> tuple[tuple[np.longdouble, np.longdouble], ...]:
    """Recover ordered gross-return pairs from well-conditioned Welford fields."""
    native_roots = _native_centered_two_return_roots(
        return_mean=return_mean,
        return_m2=return_m2,
    )
    if not native_roots:
        return ()
    if len(native_roots) == 1:
        ordered_root_centers = ((native_roots[0], native_roots[0]),)
    else:
        ordered_root_centers = (
            (native_roots[0], native_roots[1]),
            (native_roots[1], native_roots[0]),
        )

    # For two reachable float64 equity-ratio returns, reversing the two Welford
    # updates gives roots within a handful of ULPs of mean +/- sqrt(M2 / 2).
    # A fixed four-ULP Cartesian neighborhood covers those roundings, preserves
    # order-sensitive M2, and stays constant-time even at zero/subnormal scales.
    matches: list[tuple[np.longdouble, np.longdouble]] = []
    for first_center, second_center in ordered_root_centers:
        first_candidates = _adjacent_float64_values(first_center)
        second_candidates = _adjacent_float64_values(second_center)
        ordered_match: tuple[np.longdouble, np.longdouble] | None = None
        for first_return in first_candidates:
            for second_return in second_candidates:
                gross_sequence = (
                    np.longdouble(float(1.0 + first_return)),
                    np.longdouble(float(1.0 + second_return)),
                )
                if min(gross_sequence) < 0.0 or not _gross_sequence_matches_native_centered_moments(
                    gross_sequence,
                    return_sum=return_sum,
                    return_sum_squares=return_sum_squares,
                    return_mean=return_mean,
                    return_m2=return_m2,
                ):
                    continue
                ordered_match = gross_sequence
                break
            if ordered_match is not None:
                break
        if ordered_match is not None and ordered_match not in matches:
            matches.append(ordered_match)
    return tuple(matches)


def _two_return_endpoint_reconstruction_bounds(
    *,
    first_equity: float,
    last_equity: float,
    return_sum: float,
    return_sum_squares: float,
) -> tuple[np.longdouble, np.longdouble] | None:
    """Return the aggregate-implied endpoint and its bounded float64 error budget."""
    total = np.longdouble(return_sum)
    squares = np.longdouble(return_sum_squares)
    # For two returns r1 and r2, their sum and square sum determine their
    # product: r1*r2 = ((r1+r2)^2 - (r1^2+r2^2)) / 2. Consequently they also
    # determine the compounded endpoint ratio without knowing their order.
    compounded_ratio = np.longdouble(1.0) + total + (total * total - squares) / np.longdouble(2.0)
    aggregate_scale = max(
        np.longdouble(1.0),
        abs(total),
        abs(squares),
        abs(total * total),
    )
    # The sum and square sum were accumulated independently in float64. Near
    # total loss their O(eps) uncertainty can dominate the tiny residual of the
    # subtractive product identity, even though each original return was valid.
    ratio_uncertainty = np.longdouble(64.0 * np.finfo(np.float64).eps) * aggregate_scale
    endpoint_uncertainty = abs(np.longdouble(first_equity)) * ratio_uncertainty
    implied_last_equity = np.longdouble(first_equity) * compounded_ratio
    endpoint_ulp_tolerance = _float64_roundoff_tolerance(
        implied_last_equity,
        last_equity,
    )
    reconstruction_tolerance = endpoint_ulp_tolerance + endpoint_uncertainty
    if not np.isfinite(implied_last_equity) or not np.isfinite(reconstruction_tolerance):
        return None
    return implied_last_equity, reconstruction_tolerance


def _return_sequence_metrics(
    first_equity: float,
    gross_returns: tuple[np.longdouble, ...],
    *,
    reconstructed_unit_tolerance: float | np.longdouble = 0.0,
) -> tuple[np.longdouble, np.longdouble, np.longdouble, int, int] | None:
    """Compute the path metrics fixed by one candidate ordering."""
    equity = np.longdouble(first_equity)
    running_max = equity
    max_drawdown = np.longdouble(0.0)
    minimum_positive_count = 0
    maximum_positive_count = 0
    for gross_return in gross_returns:
        daily_return = float(gross_return) - 1.0
        if daily_return > reconstructed_unit_tolerance:
            minimum_positive_count += 1
            maximum_positive_count += 1
        elif daily_return >= -reconstructed_unit_tolerance:
            # Solving two roots from independently accumulated sum and square
            # sum can move an exact zero a few ulps across either side of zero.
            # Preserve both feasible sign counts only inside that reconstruction
            # error band; material positive and negative roots remain exact.
            maximum_positive_count += 1
        with np.errstate(over="ignore", invalid="ignore"):
            equity *= gross_return
        if not np.isfinite(equity):
            return None
        running_max = max(running_max, equity)
        if running_max <= 0.0:
            return None
        max_drawdown = min(max_drawdown, equity / running_max - np.longdouble(1.0))
    return (
        equity,
        running_max,
        max_drawdown,
        minimum_positive_count,
        maximum_positive_count,
    )


def _kernel_daily_return(current_equity: float, previous_equity: float) -> float:
    """Replay the kernel's float64 equity-division return."""
    if previous_equity <= 0.0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return float(np.float64(current_equity) / np.float64(previous_equity) - np.float64(1.0))


# More weekday sessions than a complete daily history from the provider's
# 1900-01-01 history floor through 2091 can contain (and comfortably above
# the roughly 33,000 weekdays through 2026).  Exact replay is deliberately
# bounded here: a state outside that supported daily-history horizon must fail
# closed instead of falling back to an approximate aggregate proof.
_RETURN_PATTERN_WORK_LIMIT = 50_000
_RETURN_CHAIN_UPDATE_LIMIT = 8_192
_RETURN_PLATEAU_LOCAL_SCAN_LIMIT = 8


def _exact_return_successor_interval(
    previous_equity: float,
    required_return: float,
    *,
    minimum_key: int,
    maximum_key: int,
) -> tuple[int, int] | None:
    """Return the exact positive-float successor plateau for one equity."""
    gross_return = float(1.0 + required_return)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        nominal_equity = float(np.float64(previous_equity) * np.float64(gross_return))
    if (
        np.isfinite(nominal_equity)
        and nominal_equity > 0.0
        and _kernel_daily_return(nominal_equity, previous_equity) == required_return
    ):
        nominal_key = _float64_order_key(nominal_equity)

        def local_boundary(direction: int) -> int | None:
            boundary_key = nominal_key
            for _ in range(_RETURN_PLATEAU_LOCAL_SCAN_LIMIT):
                adjacent_key = boundary_key + direction
                if adjacent_key < minimum_key or adjacent_key > maximum_key:
                    return boundary_key
                if (
                    _kernel_daily_return(
                        _float64_from_order_key(adjacent_key),
                        previous_equity,
                    )
                    != required_return
                ):
                    return boundary_key
                boundary_key = adjacent_key
            return None

        lower_boundary = local_boundary(-1)
        upper_boundary = local_boundary(1)
        if lower_boundary is not None and upper_boundary is not None:
            return lower_boundary, upper_boundary

    return _monotone_float64_exact_interval(
        lambda candidate: _kernel_daily_return(candidate, previous_equity),
        target=required_return,
        lower_key=minimum_key,
        upper_key=maximum_key,
        ascending=True,
    )


def _return_pattern_matches_native_moments(
    *,
    count: int,
    first_daily_return: float,
    repeated_daily_return: float,
    return_sum: float,
    return_sum_squares: float,
    return_mean: float,
    return_m2: float,
) -> bool | None:
    """Replay ``(first, repeated, ..., repeated)`` within the chain work cap."""
    target_moments_are_zero = (
        return_sum == 0.0 and return_sum_squares == 0.0 and return_mean == 0.0 and return_m2 == 0.0
    )
    if repeated_daily_return == 0.0 and (first_daily_return == 0.0 or target_moments_are_zero):
        return bool(first_daily_return == 0.0 and target_moments_are_zero)
    if count > _RETURN_PATTERN_WORK_LIMIT:
        return None
    total = 0.0
    total_squares = 0.0
    mean = 0.0
    m2 = 0.0
    for observation_count in range(1, count + 1):
        daily_return = first_daily_return if observation_count == 1 else repeated_daily_return
        total += daily_return
        total_squares += daily_return * daily_return
        delta = daily_return - mean
        mean += delta / observation_count
        m2 += delta * (daily_return - mean)
    return bool(total == return_sum and total_squares == return_sum_squares and mean == return_mean and m2 == return_m2)


def _repeated_float64_sum(term: float, count: int) -> float:
    """Return the exact sequential float64 sum of one repeated finite term.

    Same-sign repeated addition advances by a constant number of float-grid
    units within each binade.  Jumping directly to the next binade preserves
    round-to-nearest-even behavior while bounding work by the float64 exponent
    range instead of by ``count``.
    """
    if count <= 0 or term == 0.0:
        return 0.0

    sign = -1.0 if term < 0.0 else 1.0
    numerator, denominator = abs(term).as_integer_ratio()
    unit_exponent = -(denominator.bit_length() - 1)
    accumulated_units = 0
    remaining = count

    def rounded_units(exact_units: int) -> int:
        shift = max(0, exact_units.bit_length() - 53)
        if shift == 0:
            return exact_units
        quantum = 1 << shift
        quotient, remainder = divmod(exact_units, quantum)
        halfway = quantum >> 1
        if remainder > halfway or (remainder == halfway and quotient & 1):
            quotient += 1
        return quotient * quantum

    while remaining:
        next_units = rounded_units(accumulated_units + numerator)
        increment = next_units - accumulated_units
        accumulated_units = next_units
        remaining -= 1
        if remaining == 0 or increment == 0:
            break

        next_exact_units = accumulated_units + numerator
        next_shift = max(0, next_exact_units.bit_length() - 53)
        next_quantum = 1 << next_shift
        if accumulated_units % next_quantum:
            # The first result after entering a coarser binade can still be on
            # the old half-grid.  Execute it literally before attempting a
            # jump; an equal-looking increment here need not be the steady
            # stride used after the result aligns to the new grid.
            continue

        following_units = rounded_units(next_exact_units)
        following_increment = following_units - accumulated_units
        if following_increment == 0:
            # Once an aligned add rounds back to the same value, every later
            # identical add does too.
            break
        after_following_units = rounded_units(following_units + numerator)
        if after_following_units - following_units != following_increment:
            # Round-to-nearest-even can use a one-off stride at an exact tie.
            # Two equal future strides prove that the aligned phase is stable
            # for the rest of this binade.
            continue

        binade_boundary = 1 << next_exact_units.bit_length()
        additions_before_boundary = (
            binade_boundary - next_exact_units + following_increment - 1
        ) // following_increment
        additions_to_jump = min(remaining, additions_before_boundary)
        accumulated_units += additions_to_jump * following_increment
        remaining -= additions_to_jump

    trailing_zero_count = (accumulated_units & -accumulated_units).bit_length() - 1
    significand = accumulated_units >> trailing_zero_count
    try:
        magnitude = math.ldexp(
            float(significand),
            unit_exponent + trailing_zero_count,
        )
    except OverflowError:
        magnitude = math.inf
    return sign * magnitude


def _native_return_pattern_chain_is_consistent(
    *,
    count: int,
    first_equity: float,
    last_equity: float,
    running_max_equity: float,
    first_daily_return: float,
    repeated_daily_return: float,
    positive_return_count: int,
    max_drawdown: float,
) -> bool | None:
    """Find the least exact chain for ``(first, repeated, ..., repeated)``."""
    if _gross_rounding_bucket(first_daily_return) is None or _gross_rounding_bucket(repeated_daily_return) is None:
        return False
    expected_positive_count = int(first_daily_return > 0.0) + ((count - 1) if repeated_daily_return > 0.0 else 0)
    if positive_return_count != expected_positive_count:
        return False
    if first_daily_return == 0.0 and repeated_daily_return == 0.0:
        return bool(
            first_equity == last_equity
            and _extended_account_values_are_close(
                np.longdouble(first_equity),
                np.longdouble(running_max_equity),
            )
            and _extended_values_are_close(
                np.longdouble(0.0),
                np.longdouble(max_drawdown),
            )
        )
    if count > _RETURN_PATTERN_WORK_LIMIT:
        return None

    def target_return(edge_index: int) -> float:
        return first_daily_return if edge_index == 1 else repeated_daily_return

    def exact_chain_path_is_consistent(equities: list[float]) -> bool:
        running_max = first_equity
        path_max_drawdown = 0.0
        for edge_index in range(1, count + 1):
            required_return = target_return(edge_index)
            if (
                _kernel_daily_return(
                    equities[edge_index],
                    equities[edge_index - 1],
                )
                != required_return
            ):
                return False
            running_max = max(running_max, equities[edge_index])
            path_max_drawdown = min(
                path_max_drawdown,
                _kernel_daily_return(equities[edge_index], running_max),
            )
        return bool(
            _extended_account_values_are_close(
                np.longdouble(running_max),
                np.longdouble(running_max_equity),
            )
            and _extended_values_are_close(
                np.longdouble(path_max_drawdown),
                np.longdouble(max_drawdown),
            )
        )

    nominal_equities = [first_equity]
    nominal_prefix_is_unique = True
    for edge_index in range(1, count + 1):
        required_return = target_return(edge_index)
        gross_return = float(1.0 + required_return)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            equity = float(np.float64(nominal_equities[-1]) * np.float64(gross_return))
        if (
            not np.isfinite(equity)
            or equity <= 0.0
            or _kernel_daily_return(equity, nominal_equities[-1]) != required_return
        ):
            break
        nominal_equities.append(equity)
        if nominal_prefix_is_unique and any(
            adjacent_equity > 0.0
            and np.isfinite(adjacent_equity)
            and _kernel_daily_return(
                adjacent_equity,
                nominal_equities[-2],
            )
            == required_return
            for adjacent_equity in (
                float(np.nextafter(equity, -np.inf)),
                float(np.nextafter(equity, np.inf)),
            )
        ):
            nominal_prefix_is_unique = False
    if len(nominal_equities) == count + 1:
        if nominal_equities[-1] == last_equity and exact_chain_path_is_consistent(nominal_equities):
            return True
        if nominal_prefix_is_unique:
            return False

    minimum_key = _float64_order_key(float(np.nextafter(0.0, 1.0)))
    maximum_key = _float64_order_key(float(np.finfo(np.float64).max))

    # If multiplication stopped after a singleton prefix (for example because
    # a gross of zero cannot directly produce a positive -100% successor),
    # finish the uniqueness proof with exact quotient-plateau searches.
    unique_equities = nominal_equities
    for edge_index in range(len(unique_equities), count + 1):
        if not nominal_prefix_is_unique:
            break
        previous_equity = unique_equities[-1]
        required_return = target_return(edge_index)
        successor_interval = _monotone_float64_exact_interval(
            lambda candidate, previous=previous_equity: _kernel_daily_return(
                candidate,
                previous,
            ),
            target=required_return,
            lower_key=minimum_key,
            upper_key=maximum_key,
            ascending=True,
        )
        if successor_interval is None:
            return False
        if successor_interval[0] != successor_interval[1]:
            nominal_prefix_is_unique = False
            break
        unique_equities.append(_float64_from_order_key(successor_interval[0]))
    if nominal_prefix_is_unique and len(unique_equities) == count + 1:
        return bool(unique_equities[-1] == last_equity and exact_chain_path_is_consistent(unique_equities))

    # Successor-plateau bounds are monotone in the previous equity.  Replaying
    # their lower and upper edges therefore gives exact necessary endpoint
    # bounds even when the reachable set between them may contain holes.
    lower_equities = [first_equity]
    upper_equities = [first_equity]
    endpoint_envelope_is_complete = True
    for edge_index in range(1, count + 1):
        required_return = target_return(edge_index)
        lower_previous = lower_equities[-1]
        upper_previous = upper_equities[-1]
        lower_interval = _exact_return_successor_interval(
            lower_previous,
            required_return,
            minimum_key=minimum_key,
            maximum_key=maximum_key,
        )
        upper_interval = (
            lower_interval
            if upper_previous == lower_previous
            else _exact_return_successor_interval(
                upper_previous,
                required_return,
                minimum_key=minimum_key,
                maximum_key=maximum_key,
            )
        )
        if lower_interval is None or upper_interval is None:
            endpoint_envelope_is_complete = False
            break
        lower_equities.append(_float64_from_order_key(lower_interval[0]))
        upper_equities.append(_float64_from_order_key(upper_interval[1]))
    if endpoint_envelope_is_complete:
        if last_equity < lower_equities[-1] or last_equity > upper_equities[-1]:
            return False
        if (last_equity == lower_equities[-1] and exact_chain_path_is_consistent(lower_equities)) or (
            last_equity == upper_equities[-1] and exact_chain_path_is_consistent(upper_equities)
        ):
            return True

    minimum_equity = _float64_from_order_key(minimum_key)
    equities = [minimum_equity] * (count + 1)
    equities[0] = first_equity
    equities[count] = last_equity
    pending_edges = list(range(count, 0, -1))
    edge_is_pending = [False] + [True] * count
    update_count = 0

    def enqueue(edge_index: int) -> None:
        if 1 <= edge_index <= count and not edge_is_pending[edge_index]:
            pending_edges.append(edge_index)
            edge_is_pending[edge_index] = True

    while pending_edges:
        edge_index = pending_edges.pop()
        edge_is_pending[edge_index] = False
        previous_equity = equities[edge_index - 1]
        equity = equities[edge_index]
        required_return = target_return(edge_index)
        observed_return = _kernel_daily_return(equity, previous_equity)
        if observed_return == required_return:
            continue

        if observed_return < required_return:
            if edge_index == count:
                return False
            replacement_key = _monotone_float64_threshold_key(
                lambda candidate, previous=previous_equity: _kernel_daily_return(
                    candidate,
                    previous,
                ),
                target=required_return,
                lower_key=_float64_order_key(equity),
                upper_key=maximum_key,
                ascending=True,
            )
            if replacement_key is None:
                return False
            equities[edge_index] = _float64_from_order_key(replacement_key)
            enqueue(edge_index)
            enqueue(edge_index + 1)
        else:
            if edge_index == 1:
                return False
            replacement_key = _monotone_float64_threshold_key(
                lambda candidate, current=equity: _kernel_daily_return(
                    current,
                    candidate,
                ),
                target=required_return,
                lower_key=_float64_order_key(previous_equity),
                upper_key=maximum_key,
                ascending=False,
            )
            if replacement_key is None:
                return False
            equities[edge_index - 1] = _float64_from_order_key(replacement_key)
            enqueue(edge_index - 1)
            enqueue(edge_index)

        update_count += 1
        if update_count > _RETURN_CHAIN_UPDATE_LIMIT:
            return None

    return exact_chain_path_is_consistent(equities)


def _float64_intermediate_equity_candidates(
    daily_returns: tuple[float, float],
    *,
    first_equity: float,
    last_equity: float,
    anchors: tuple[float, ...],
) -> tuple[float, ...]:
    """Intersect the exact float64 quotient-return plateaus for the middle equity."""
    lower_key = _float64_order_key(float(np.nextafter(0.0, 1.0)))
    upper_key = _float64_order_key(float(np.finfo(np.float64).max))
    first_interval = _monotone_float64_exact_interval(
        lambda middle: _kernel_daily_return(middle, first_equity),
        target=daily_returns[0],
        lower_key=lower_key,
        upper_key=upper_key,
        ascending=True,
    )
    second_interval = _monotone_float64_exact_interval(
        lambda middle: _kernel_daily_return(last_equity, middle),
        target=daily_returns[1],
        lower_key=lower_key,
        upper_key=upper_key,
        ascending=False,
    )
    if first_interval is None or second_interval is None:
        return ()
    intersection_lower = max(first_interval[0], second_interval[0])
    intersection_upper = min(first_interval[1], second_interval[1])
    if intersection_lower > intersection_upper:
        return ()

    candidate_keys = {
        intersection_lower,
        intersection_upper,
        (intersection_lower + intersection_upper) // 2,
    }
    for anchor in anchors:
        if not np.isfinite(anchor) or anchor <= 0.0:
            continue
        anchor_key = _float64_order_key(anchor)
        clamped_key = min(max(anchor_key, intersection_lower), intersection_upper)
        candidate_keys.add(clamped_key)
        if clamped_key > intersection_lower:
            candidate_keys.add(clamped_key - 1)
        if clamped_key < intersection_upper:
            candidate_keys.add(clamped_key + 1)
    return tuple(_float64_from_order_key(candidate_key) for candidate_key in sorted(candidate_keys))


def _native_two_return_float64_path_is_consistent(
    daily_returns: tuple[float, float],
    *,
    first_equity: float,
    last_equity: float,
    running_max_equity: float,
    positive_return_count: int,
    max_drawdown: float,
) -> bool:
    """Require an actual positive float64 intermediate equity for a native pair."""
    first_grosses = (
        np.longdouble(1.0) + np.longdouble(daily_returns[0]),
        np.longdouble(float(1.0 + daily_returns[0])),
    )
    second_grosses = (
        np.longdouble(1.0) + np.longdouble(daily_returns[1]),
        np.longdouble(float(1.0 + daily_returns[1])),
    )
    extended_seeds: list[np.longdouble] = []
    for first_gross in first_grosses:
        if np.isfinite(first_gross) and first_gross >= 0.0:
            extended_seeds.append(np.longdouble(first_equity) * first_gross)
    for second_gross in second_grosses:
        if np.isfinite(second_gross) and second_gross > 0.0:
            extended_seeds.append(np.longdouble(last_equity) / second_gross)

    seed_values: list[float] = []
    for extended_seed in extended_seeds:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            seed = float(extended_seed)
        if np.isfinite(seed) and seed not in seed_values:
            seed_values.append(seed)

    def middle_path_is_consistent(middle_equity: float) -> bool:
        if not np.isfinite(middle_equity) or middle_equity <= 0.0:
            return False
        observed_returns = (
            _kernel_daily_return(middle_equity, first_equity),
            _kernel_daily_return(last_equity, middle_equity),
        )
        if observed_returns != daily_returns:
            return False

        running_max = max(first_equity, middle_equity)
        first_drawdown = _kernel_daily_return(middle_equity, running_max)
        running_max = max(running_max, last_equity)
        second_drawdown = _kernel_daily_return(last_equity, running_max)
        implied_max_drawdown = min(0.0, first_drawdown, second_drawdown)
        return bool(
            positive_return_count == int(daily_returns[0] > 0.0) + int(daily_returns[1] > 0.0)
            and _extended_account_values_are_close(
                np.longdouble(running_max),
                np.longdouble(running_max_equity),
            )
            and _extended_values_are_close(
                np.longdouble(implied_max_drawdown),
                np.longdouble(max_drawdown),
            )
        )

    if any(
        middle_path_is_consistent(middle_equity)
        for seed in seed_values
        for middle_equity in _adjacent_float64_values(seed, radius=8)
    ):
        return True

    return any(
        middle_path_is_consistent(middle_equity)
        for middle_equity in _float64_intermediate_equity_candidates(
            daily_returns,
            first_equity=first_equity,
            last_equity=last_equity,
            anchors=(*seed_values, running_max_equity),
        )
    )


def _return_sequence_path_is_consistent(
    gross_returns: tuple[np.longdouble, ...],
    *,
    first_equity: float,
    last_equity: float,
    running_max_equity: float,
    positive_return_count: int,
    max_drawdown: float,
    reconstructed_unit_tolerance: float | np.longdouble,
) -> bool:
    """Check path-dependent fields for one ordered gross-return witness."""
    metrics = _return_sequence_metrics(
        first_equity,
        gross_returns,
        reconstructed_unit_tolerance=reconstructed_unit_tolerance,
    )
    return bool(
        metrics is not None
        and metrics[3] <= positive_return_count <= metrics[4]
        and _extended_account_values_are_close(metrics[0], np.longdouble(last_equity))
        and _extended_account_values_are_close(
            metrics[1],
            np.longdouble(running_max_equity),
        )
        and _extended_values_are_close(
            metrics[2],
            np.longdouble(max_drawdown),
        )
    )


@lru_cache(maxsize=512, typed=True)
def _fully_determined_return_rollup_is_consistent(
    *,
    count: int,
    first_equity: float,
    last_equity: float,
    running_max_equity: float,
    return_sum: float,
    return_sum_squares: float,
    return_mean: float,
    return_m2: float,
    positive_return_count: int,
    max_drawdown: float,
    native_centered_moments_available: bool = True,
) -> bool:
    """Validate path fields when moments determine every return or its two-item multiset.

    Grid points commonly share an unchanged rollup.  Cache the pure proof so
    exact long-history replay is paid once for identical persisted state.
    """
    if not _return_rollup_endpoint_growth_is_feasible(
        count=count,
        first_equity=first_equity,
        last_equity=last_equity,
        return_sum=return_sum,
        return_sum_squares=return_sum_squares,
        return_mean=return_mean,
        return_m2=return_m2,
    ):
        return False
    if native_centered_moments_available and count > 2 and return_m2 == 0.0:
        # Native Welford updates can round M2 to zero only when every return
        # equals the final mean, except that the first return may be one
        # adjacent float away.  Check those exhaustive patterns directly
        # before the raw-moment heuristic: a slightly corrupted sum can make
        # that heuristic decline to classify an otherwise zero-M2 rollup.
        repeated_daily_return = return_mean
        first_return_candidates = tuple(
            dict.fromkeys(
                (
                    float(np.nextafter(repeated_daily_return, -np.inf)),
                    repeated_daily_return,
                    float(np.nextafter(repeated_daily_return, np.inf)),
                )
            )
        )
        for first_daily_return in first_return_candidates:
            if not np.isfinite(first_daily_return) or first_daily_return < -1.0:
                continue
            moments_match = _return_pattern_matches_native_moments(
                count=count,
                first_daily_return=first_daily_return,
                repeated_daily_return=repeated_daily_return,
                return_sum=return_sum,
                return_sum_squares=return_sum_squares,
                return_mean=return_mean,
                return_m2=return_m2,
            )
            if moments_match is None:
                # Native zero-M2 state is fully determined by one of these
                # patterns.  Letting an unreplayed nonzero pattern fall
                # through to tolerant aggregate checks would accept arbitrary
                # endpoint and path fields.  Exact all-zero patterns are
                # handled in constant time before the replay work cap.
                return False
            if not moments_match:
                continue
            chain_is_consistent = _native_return_pattern_chain_is_consistent(
                count=count,
                first_equity=first_equity,
                last_equity=last_equity,
                running_max_equity=running_max_equity,
                first_daily_return=first_daily_return,
                repeated_daily_return=repeated_daily_return,
                positive_return_count=positive_return_count,
                max_drawdown=max_drawdown,
            )
            if chain_is_consistent:
                return True
            if chain_is_consistent is None:
                # Exact native replay is the authority for zero-M2 histories.
                # Never fall through from a bounded/inconclusive chain search
                # to tolerant aggregate reconstruction: that can admit an
                # unreachable endpoint inside a return-rounding envelope.
                return False
        return False

    constant_return = _constant_return_implied_by_moments(
        count=count,
        total=return_sum,
        total_squares=return_sum_squares,
        mean=return_mean,
        m2=return_m2,
    )
    if constant_return is not None:
        constant_endpoint_is_feasible = _constant_return_endpoint_is_feasible(
            count=count,
            first_equity=first_equity,
            last_equity=last_equity,
            constant_return=constant_return,
        )
        if constant_endpoint_is_feasible:
            endpoint_ratio = np.longdouble(last_equity) / np.longdouble(first_equity)
            if constant_return > 0.0:
                implied_running_max = np.longdouble(last_equity)
                implied_max_drawdown = np.longdouble(0.0)
            elif constant_return < 0.0:
                implied_running_max = np.longdouble(first_equity)
                implied_max_drawdown = endpoint_ratio - np.longdouble(1.0)
            else:
                implied_running_max = np.longdouble(first_equity)
                implied_max_drawdown = np.longdouble(0.0)
            constant_path_is_consistent = bool(
                positive_return_count == (count if constant_return > 0.0 else 0)
                and _extended_account_values_are_close(
                    implied_running_max,
                    np.longdouble(running_max_equity),
                )
                and _extended_values_are_close(
                    implied_max_drawdown,
                    np.longdouble(max_drawdown),
                )
            )
            if constant_path_is_consistent and (count != 2 or not native_centered_moments_available):
                return True
        if count != 2:
            return False

    if count != 2:
        return True
    endpoint_reconstruction = _two_return_endpoint_reconstruction_bounds(
        first_equity=first_equity,
        last_equity=last_equity,
        return_sum=return_sum,
        return_sum_squares=return_sum_squares,
    )
    if endpoint_reconstruction is None:
        return False
    implied_last_equity, endpoint_tolerance = endpoint_reconstruction
    if abs(implied_last_equity - np.longdouble(last_equity)) > endpoint_tolerance:
        return False
    gross_reconstruction = _two_return_gross_sequences(
        first_equity=first_equity,
        last_equity=last_equity,
        return_sum=return_sum,
        return_sum_squares=return_sum_squares,
    )
    if gross_reconstruction is None:
        gross_sequences: tuple[tuple[np.longdouble, np.longdouble], ...] = ()
        reconstructed_unit_tolerance = np.longdouble(_STATE_RELATIVE_TOLERANCE)
    else:
        (
            gross_sequences,
            reconstructed_unit_tolerance,
            _reconstruction_mode,
            _stable_root_uncertainty,
        ) = gross_reconstruction
    native_roots = _native_centered_two_return_roots(
        return_mean=return_mean,
        return_m2=return_m2,
    )
    if native_centered_moments_available:
        checked_daily_sequences: dict[tuple[float, float], bool] = {}

        def native_daily_sequence_is_consistent(
            daily_returns: tuple[float, float],
        ) -> bool:
            if daily_returns in checked_daily_sequences:
                return checked_daily_sequences[daily_returns]
            is_consistent = bool(
                _daily_return_sequence_matches_native_centered_moments(
                    daily_returns,
                    return_sum=return_sum,
                    return_sum_squares=return_sum_squares,
                    return_mean=return_mean,
                    return_m2=return_m2,
                )
                and _native_two_return_float64_path_is_consistent(
                    daily_returns,
                    first_equity=first_equity,
                    last_equity=last_equity,
                    running_max_equity=running_max_equity,
                    positive_return_count=positive_return_count,
                    max_drawdown=max_drawdown,
                )
            )
            checked_daily_sequences[daily_returns] = is_consistent
            return is_consistent

        if any(
            native_daily_sequence_is_consistent(tuple(float(gross_return) - 1.0 for gross_return in candidate))
            for candidate in gross_sequences
        ):
            return True

        centered_gross_sequences = _centered_sequences_matching_native_moments(
            return_sum=return_sum,
            return_sum_squares=return_sum_squares,
            return_mean=return_mean,
            return_m2=return_m2,
        )
        if any(
            native_daily_sequence_is_consistent(tuple(float(gross_return) - 1.0 for gross_return in candidate))
            for candidate in centered_gross_sequences
        ):
            return True

        for centered_sequence in centered_gross_sequences:
            daily_returns = tuple(float(gross_return) - 1.0 for gross_return in centered_sequence)
            if any(
                native_daily_sequence_is_consistent(
                    tuple(float(gross_return) - 1.0 for gross_return in endpoint_candidate)
                )
                for endpoint_candidate in (
                    _endpoint_conditioned_gross_sequences_for_returns(
                        daily_returns,
                        first_equity=first_equity,
                        last_equity=last_equity,
                    )
                )
            ):
                return True

        loss_companion_center = float(return_sum + 1.0)
        if np.isfinite(loss_companion_center):
            loss_companion_candidates = _adjacent_float64_values(
                loss_companion_center,
                radius=8,
            )
            if any(
                native_daily_sequence_is_consistent(candidate)
                for companion in loss_companion_candidates
                for candidate in ((-1.0, companion), (companion, -1.0))
            ):
                return True

        loss_candidate_anchors = (
            loss_companion_center,
            *native_roots,
            *(float(gross_return) - 1.0 for gross_sequence in gross_sequences for gross_return in gross_sequence),
        )
        if any(
            native_daily_sequence_is_consistent(candidate)
            for companion in _exact_loss_companion_candidates(
                return_sum=return_sum,
                return_sum_squares=return_sum_squares,
                anchors=loss_candidate_anchors,
            )
            for candidate in ((-1.0, companion), (companion, -1.0))
        ):
            return True

        if not native_roots:
            return False
        endpoint_gross_sequences = _endpoint_conditioned_sequences_matching_native_moments(
            gross_sequences,
            first_equity=first_equity,
            last_equity=last_equity,
            native_roots=native_roots,
            return_sum=return_sum,
            return_sum_squares=return_sum_squares,
            return_mean=return_mean,
            return_m2=return_m2,
        )
        return any(
            native_daily_sequence_is_consistent(tuple(float(gross_return) - 1.0 for gross_return in candidate))
            for candidate in endpoint_gross_sequences
        )

    # Compatibility centered fields were derived from raw moments and carry no
    # independent ordering information, so both raw orderings remain feasible.
    raw_witnesses = tuple(
        gross_returns
        for gross_returns in gross_sequences
        if _gross_sequence_matches_raw_return_moments(
            gross_returns,
            return_sum=return_sum,
            return_sum_squares=return_sum_squares,
        )
    )
    exact_raw_witness = bool(raw_witnesses)
    if exact_raw_witness:
        gross_sequences = raw_witnesses
    if any(
        _return_sequence_path_is_consistent(
            candidate,
            first_equity=first_equity,
            last_equity=last_equity,
            running_max_equity=running_max_equity,
            positive_return_count=positive_return_count,
            max_drawdown=max_drawdown,
            reconstructed_unit_tolerance=(0.0 if exact_raw_witness else reconstructed_unit_tolerance),
        )
        for candidate in gross_sequences
    ):
        return True
    for raw_witness in raw_witnesses:
        daily_returns = tuple(float(gross_return) - 1.0 for gross_return in raw_witness)
        if any(
            _return_sequence_path_is_consistent(
                candidate,
                first_equity=first_equity,
                last_equity=last_equity,
                running_max_equity=running_max_equity,
                positive_return_count=positive_return_count,
                max_drawdown=max_drawdown,
                reconstructed_unit_tolerance=0.0,
            )
            for candidate in _endpoint_conditioned_gross_sequences_for_returns(
                daily_returns,
                first_equity=first_equity,
                last_equity=last_equity,
            )
        ):
            return True
    if not native_roots:
        return False
    endpoint_gross_sequences = _endpoint_conditioned_sequences_matching_native_moments(
        gross_sequences,
        first_equity=first_equity,
        last_equity=last_equity,
        native_roots=native_roots,
        return_sum=return_sum,
        return_sum_squares=return_sum_squares,
        return_mean=return_mean,
        return_m2=return_m2,
        require_native_centered_moments=False,
    )
    return any(
        _return_sequence_path_is_consistent(
            candidate,
            first_equity=first_equity,
            last_equity=last_equity,
            running_max_equity=running_max_equity,
            positive_return_count=positive_return_count,
            max_drawdown=max_drawdown,
            reconstructed_unit_tolerance=0.0,
        )
        for candidate in endpoint_gross_sequences
    )


def _legacy_centered_moment_arrays(
    count_values: np.ndarray,
    total_values: np.ndarray,
    total_squares_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive compatibility inputs; persisted state uses native centered fields."""
    means = np.zeros(len(count_values), dtype=np.float64)
    m2_values = np.zeros(len(count_values), dtype=np.float64)
    for config_idx, count_value in enumerate(count_values):
        count = int(count_value)
        if count <= 0:
            continue
        total = float(total_values[config_idx])
        total_squares = float(total_squares_values[config_idx])
        means[config_idx] = total / count
        m2 = float(np.longdouble(total_squares) - np.longdouble(total) * np.longdouble(total) / np.longdouble(count))
        m2_values[config_idx] = max(m2, 0.0)
    return means, m2_values


def _validate_centered_moment_accumulator(
    *,
    config_idx: int,
    label: str,
    count: int,
    total: float,
    total_squares: float,
    mean: float,
    m2: float,
) -> None:
    if m2 < 0.0:
        raise ValueError(f"{label}_m2_values must be non-negative at config index {config_idx}.")
    if count == 0:
        # Like the raw accumulators, an empty Welford accumulator has exact
        # kernel-origin zeros; there is no arithmetic here that needs a
        # unit-scale comparison floor.
        if mean != 0.0 or m2 != 0.0:
            raise ValueError(f"Zero {label} count requires zero centered moments at config index {config_idx}.")
        return

    expected_total = mean * count
    tolerance = max(
        _STATE_ABSOLUTE_TOLERANCE,
        128.0 * np.finfo(np.float64).eps * count * max(abs(total), abs(expected_total), 1.0),
    )
    if abs(total - expected_total) > tolerance:
        raise ValueError(f"{label} mean is inconsistent with its count and sum at config index {config_idx}.")

    # The raw and centered accumulators are persisted independently.  Checking
    # the mean against the sum is not enough: an arbitrary non-negative M2
    # would otherwise be accepted and poison volatility-based metrics on the
    # next incremental run.  Evaluate the forward identity in extended
    # precision so the validation itself does not introduce the catastrophic
    # cancellation that centered moments are intended to avoid.
    observed_squares = np.longdouble(total_squares)
    expected_squares = np.longdouble(m2) + (np.longdouble(count) * np.longdouble(mean) * np.longdouble(mean))
    scale = max(
        abs(observed_squares),
        abs(expected_squares),
        np.longdouble(np.finfo(np.float64).tiny),
    )
    relative_tolerance = min(1e-8, _MOMENT_ROUNDOFF_FACTOR * count)
    if abs(observed_squares - expected_squares) > relative_tolerance * scale:
        raise ValueError(
            f"{label} centered moments are inconsistent with its count, mean, and square sum "
            f"at config index {config_idx}."
        )


def _validate_resumable_state(
    *,
    start_indices: np.ndarray,
    history_prefix_observation_counts: np.ndarray,
    cash_values: np.ndarray,
    share_values: np.ndarray,
    in_position_values: np.ndarray,
    entry_price_values: np.ndarray,
    pending_action_values: np.ndarray,
    prev_equity_values: np.ndarray,
    trades_executed_values: np.ndarray,
    first_equity_values: np.ndarray,
    last_equity_values: np.ndarray,
    running_max_equity_values: np.ndarray,
    return_count_values: np.ndarray,
    return_sum_values: np.ndarray,
    return_sum_squares_values: np.ndarray,
    excess_return_count_values: np.ndarray,
    excess_return_sum_values: np.ndarray,
    excess_return_sum_squares_values: np.ndarray,
    positive_return_count_values: np.ndarray,
    max_drawdown_values: np.ndarray,
    return_mean_values: np.ndarray,
    return_m2_values: np.ndarray,
    excess_return_mean_values: np.ndarray,
    excess_return_m2_values: np.ndarray,
    resume_close_values: np.ndarray,
    native_centered_moments_available: np.ndarray,
) -> None:
    """Reject resumable account and metric state that the kernel cannot produce."""
    for config_idx in range(len(cash_values)):
        cash = float(cash_values[config_idx])
        shares = float(share_values[config_idx])
        in_position = bool(in_position_values[config_idx])
        entry_price = float(entry_price_values[config_idx])
        pending_action = int(pending_action_values[config_idx])
        prev_equity = float(prev_equity_values[config_idx])
        trades_executed = int(trades_executed_values[config_idx])

        if in_position:
            if shares <= 0.0 or np.isnan(entry_price):
                raise ValueError(
                    "An in-position resumable state requires positive shares and a positive "
                    f"entry price at config index {config_idx}."
                )
            with np.errstate(over="ignore", invalid="ignore"):
                entry_notional = shares * entry_price
            if not np.isfinite(entry_notional) or entry_notional <= 0.0:
                raise ValueError(
                    "An in-position resumable state must have positive finite entry "
                    f"notional at config index {config_idx}."
                )
            account_scale = max(prev_equity, entry_notional, cash)
            cash_tolerance = _float64_roundoff_tolerance(0.0, account_scale)
            if cash > cash_tolerance:
                raise ValueError(
                    f"An in-position resumable state may only retain float-roundoff cash at config index {config_idx}."
                )
            if trades_executed % 2 != 1:
                raise ValueError(
                    f"An in-position resumable state requires an odd trade count at config index {config_idx}."
                )
            if pending_action != ACTION_NONE:
                raise ValueError(
                    f"An in-position resumable state cannot have a pending action at config index {config_idx}."
                )
            resume_close = float(resume_close_values[config_idx])
            if np.isnan(resume_close):
                raise ValueError(
                    f"An in-position resumable state requires its last closing price at config index {config_idx}."
                )
            with np.errstate(over="ignore", invalid="ignore"):
                marked_equity = cash + shares * resume_close
            if not np.isfinite(marked_equity) or not _account_state_values_are_close(
                prev_equity,
                marked_equity,
            ):
                raise ValueError(
                    "An in-position resumable state's previous equity must equal cash plus "
                    "shares valued at its last closing price at config index "
                    f"{config_idx}."
                )
        else:
            if shares != 0.0 or not np.isnan(entry_price):
                raise ValueError(
                    f"A flat resumable state requires zero shares and no entry price at config index {config_idx}."
                )
            if trades_executed % 2 != 0:
                raise ValueError(f"A flat resumable state requires an even trade count at config index {config_idx}.")
            if pending_action not in {ACTION_NONE, ACTION_BUY}:
                raise ValueError(
                    "A flat resumable state may only have no pending action or a pending buy "
                    f"at config index {config_idx}."
                )
            if not _account_state_values_are_close(cash, prev_equity):
                raise ValueError(
                    f"A flat resumable state's cash must equal its previous equity at config index {config_idx}."
                )

        first_equity = float(first_equity_values[config_idx])
        last_equity = float(last_equity_values[config_idx])
        running_max_equity = float(running_max_equity_values[config_idx])
        max_drawdown = float(max_drawdown_values[config_idx])
        rollup_missing = tuple(
            np.isnan(value)
            for value in (
                first_equity,
                last_equity,
                running_max_equity,
                max_drawdown,
            )
        )
        if any(rollup_missing) and not all(rollup_missing):
            raise ValueError(
                "Resumable rollup equity and drawdown fields must be either all present or all "
                f"missing at config index {config_idx}."
            )

        return_count = int(return_count_values[config_idx])
        excess_return_count = int(excess_return_count_values[config_idx])
        positive_return_count = int(positive_return_count_values[config_idx])
        return_sum = float(return_sum_values[config_idx])
        return_sum_squares = float(return_sum_squares_values[config_idx])
        excess_return_sum = float(excess_return_sum_values[config_idx])
        excess_return_sum_squares = float(excess_return_sum_squares_values[config_idx])
        return_mean = float(return_mean_values[config_idx])
        return_m2 = float(return_m2_values[config_idx])
        excess_return_mean = float(excess_return_mean_values[config_idx])
        excess_return_m2 = float(excess_return_m2_values[config_idx])

        if positive_return_count > return_count:
            raise ValueError(
                f"positive_return_count_values cannot exceed return_count_values at config index {config_idx}."
            )
        if excess_return_count > return_count:
            raise ValueError(
                f"excess_return_count_values cannot exceed return_count_values at config index {config_idx}."
            )
        _validate_moment_accumulator(
            config_idx=config_idx,
            label="return",
            count=return_count,
            total=return_sum,
            total_squares=return_sum_squares,
        )
        if not _strategy_return_moments_respect_lower_bound(
            return_count,
            return_sum,
            return_sum_squares,
        ):
            raise ValueError(
                "Resumable return moments cannot describe strategy daily returns bounded below "
                f"by -1.0 at config index {config_idx}."
            )
        _validate_moment_accumulator(
            config_idx=config_idx,
            label="excess_return",
            count=excess_return_count,
            total=excess_return_sum,
            total_squares=excess_return_sum_squares,
        )
        _validate_centered_moment_accumulator(
            config_idx=config_idx,
            label="return",
            count=return_count,
            total=return_sum,
            total_squares=return_sum_squares,
            mean=return_mean,
            m2=return_m2,
        )
        _validate_centered_moment_accumulator(
            config_idx=config_idx,
            label="excess_return",
            count=excess_return_count,
            total=excess_return_sum,
            total_squares=excess_return_sum_squares,
            mean=excess_return_mean,
            m2=excess_return_m2,
        )
        if not _excess_return_moments_respect_risk_free_domain(
            return_count=return_count,
            return_sum=return_sum,
            return_sum_squares=return_sum_squares,
            return_mean=return_mean,
            return_m2=return_m2,
            excess_return_count=excess_return_count,
            excess_return_sum=excess_return_sum,
            excess_return_sum_squares=excess_return_sum_squares,
            excess_return_mean=excess_return_mean,
            excess_return_m2=excess_return_m2,
            positive_return_count=positive_return_count,
        ):
            raise ValueError(
                "Resumable excess-return moments require risk-free returns greater than "
                f"-1.0 at config index {config_idx}."
            )
        if all(rollup_missing):
            if (
                return_count != 0
                or excess_return_count != 0
                or positive_return_count != 0
                or trades_executed != 0
                or in_position
                or pending_action != ACTION_NONE
            ):
                raise ValueError(
                    "An empty resumable rollup requires pristine strategy state and zero counts "
                    f"at config index {config_idx}."
                )
            continue

        if return_count == 1 and not _single_return_rollup_is_consistent(
            first_equity=first_equity,
            last_equity=last_equity,
            return_sum=return_sum,
            return_sum_squares=return_sum_squares,
            return_mean=return_mean,
            return_m2=return_m2,
            positive_return_count=positive_return_count,
        ):
            raise ValueError(
                "A one-return resumable rollup requires moments and positive_return_count_values "
                "to match the return implied by first and last equity at config index "
                f"{config_idx}."
            )
        if not _positive_return_count_is_feasible(
            return_count,
            return_sum,
            return_sum_squares,
            positive_return_count,
        ):
            raise ValueError(
                "positive_return_count_values is inconsistent with fully determined strategy "
                "returns and aggregate return sums at config "
                f"index {config_idx}."
            )
        # The first equity observation begins from pristine state and can only
        # schedule a buy.  Each subsequent observation contributes one return
        # and can execute at most that buy plus one same-day target sell.
        if trades_executed > 2 * return_count:
            raise ValueError(
                f"trades_executed_values cannot exceed twice return_count_values at config index {config_idx}."
            )

        if not _account_state_values_are_close(prev_equity, last_equity):
            raise ValueError(
                f"prev_equity_values must match last_equity_values for resumed history at config index {config_idx}."
            )
        if (
            running_max_equity < first_equity and not _account_state_values_are_close(running_max_equity, first_equity)
        ) or (
            running_max_equity < last_equity and not _account_state_values_are_close(running_max_equity, last_equity)
        ):
            raise ValueError(
                f"running_max_equity_values must be at least the first and last equity at config index {config_idx}."
            )
        if max_drawdown < -1.0 - _STATE_RELATIVE_TOLERANCE or max_drawdown > _STATE_RELATIVE_TOLERANCE:
            raise ValueError(f"max_drawdown_values must be between -1.0 and 0.0 at config index {config_idx}.")
        current_drawdown = last_equity / running_max_equity - 1.0
        if max_drawdown > current_drawdown + _STATE_RELATIVE_TOLERANCE:
            raise ValueError(
                "max_drawdown_values cannot exceed the drawdown implied by last and running "
                f"maximum equity at config index {config_idx}."
            )
        if not _positive_return_path_is_feasible(
            count=return_count,
            positive_return_count=positive_return_count,
            first_equity=first_equity,
            last_equity=last_equity,
            running_max_equity=running_max_equity,
            max_drawdown=max_drawdown,
            total=return_sum,
        ):
            raise ValueError(
                "positive_return_count_values is inconsistent with fully determined strategy "
                f"returns, equity endpoints, running maximum, or drawdown at config index {config_idx}."
            )
        if return_count == 0 and max_drawdown != 0.0:
            raise ValueError(
                f"A one-observation resumable rollup requires zero maximum drawdown at config index {config_idx}."
            )
        if trades_executed == 0 and (
            not _account_state_values_are_close(first_equity, prev_equity)
            or not _account_state_values_are_close(running_max_equity, prev_equity)
            or positive_return_count != 0
            or return_sum != 0.0
            or return_sum_squares != 0.0
            or return_mean != 0.0
            or return_m2 != 0.0
            or max_drawdown != 0.0
        ):
            raise ValueError(
                "A zero-trade resumable rollup requires constant strategy equity and zero "
                f"strategy returns at config index {config_idx}."
            )
        if not _return_rollup_endpoint_growth_is_feasible(
            count=return_count,
            first_equity=first_equity,
            last_equity=last_equity,
            return_sum=return_sum,
            return_sum_squares=return_sum_squares,
            return_mean=return_mean,
            return_m2=return_m2,
        ):
            raise ValueError(
                "Resumable return moments imply an infeasible endpoint growth bound for last_equity_values "
                f"at config index {config_idx}."
            )
        if not _return_rollup_equity_endpoints_are_consistent(
            count=return_count,
            first_equity=first_equity,
            last_equity=last_equity,
            return_sum=return_sum,
            return_sum_squares=return_sum_squares,
            return_mean=return_mean,
            return_m2=return_m2,
        ):
            raise ValueError(
                f"Resumable return moments are inconsistent with first and last equity at config index {config_idx}."
            )
        if not _fully_determined_return_rollup_is_consistent(
            count=return_count,
            first_equity=first_equity,
            last_equity=last_equity,
            running_max_equity=running_max_equity,
            return_sum=return_sum,
            return_sum_squares=return_sum_squares,
            return_mean=return_mean,
            return_m2=return_m2,
            positive_return_count=positive_return_count,
            max_drawdown=max_drawdown,
            native_centered_moments_available=bool(native_centered_moments_available[config_idx]),
        ):
            raise ValueError(
                "Resumable positive-return count, running maximum, or maximum drawdown is "
                "inconsistent with fully determined strategy returns at config index "
                f"{config_idx}."
            )
        # A rollup with N returns necessarily contains N + 1 equity
        # observations.  Those observations must all belong either to history
        # before the supplied market-data window or to the prefix that the
        # kernel is about to skip.  The inequality is intentionally not an
        # equality because a strategy may legitimately begin after older rows.
        available_observations = int(history_prefix_observation_counts[config_idx]) + int(start_indices[config_idx])
        if return_count + 1 > available_observations:
            raise ValueError(
                "Resumable rollup observations cannot exceed the history prefix described by "
                "history_prefix_observation_counts and start_indices at config index "
                f"{config_idx}."
            )


def _validate_single_results(results: tuple) -> None:
    for name, values, allow_nan in (
        ("equity", results[0], False),
        ("daily returns", results[1], False),
        ("risk-free returns", results[2], True),
        ("cash", results[7], False),
        ("shares", results[8], False),
        ("entry price", results[10], True),
        ("previous equity", results[12], False),
    ):
        _require_finite_result(name, values, allow_nan=allow_nan)
    if np.any(results[0] <= 0.0) or results[12] <= 0.0:
        raise ValueError("Backtest produced non-positive equity.")
    if results[7] < 0.0 or results[8] < 0.0:
        raise ValueError("Backtest produced negative cash or shares.")
    _validate_return_rollup_inputs(results[1], results[2])


def _target_limit_price(entry_price: float, profit_target_multiple: float) -> float:
    """Mirror the broker's decimal multiplication and upward tick rounding."""
    return target_sell_price(entry_price, profit_target_multiple)


@njit(cache=True)
def _approximate_target_limit_price(entry_price: float, profit_target_multiple: float) -> float:
    raw_target = entry_price * profit_target_multiple
    scale = 10_000.0 if raw_target < 1.0 else 100.0
    return np.ceil(raw_target * scale) / scale


def _target_price_overrides(
    open_prices: np.ndarray,
    profit_target_values: np.ndarray,
    start_indices: np.ndarray | None = None,
    entry_candidate_indices: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare sparse Decimal corrections for tick-boundary float products."""
    unique_targets, target_indices = np.unique(profit_target_values, return_inverse=True)
    if start_indices is None:
        start_indices = np.zeros(len(profit_target_values), dtype=np.int64)
    offsets = [0]
    override_rows: list[int] = []
    override_values: list[float] = []
    for target_idx, multiple in enumerate(unique_targets):
        matching_configs = np.flatnonzero(target_indices == target_idx)
        if entry_candidate_indices is None:
            matching_starts = start_indices[matching_configs]
            reachable_rows = np.arange(int(np.min(matching_starts)), len(open_prices))
        else:
            reachable_mask = np.zeros(len(open_prices), dtype=np.bool_)
            for config_idx in matching_configs:
                reachable_mask[entry_candidate_indices[int(config_idx)]] = True
            reachable_rows = np.flatnonzero(reachable_mask)
        reachable_opens = open_prices[reachable_rows]
        if not len(reachable_opens):
            offsets.append(len(override_rows))
            continue
        try:
            target_sell_price(float(np.max(reachable_opens)), float(multiple))
            require_exact_validation = False
        except ValueError:
            # An unrepresentable candidate is only an error if the strategy
            # actually enters on that row. Store a sentinel and let the wrapper
            # inspect the kernel's executed entry instead of rejecting a safe
            # path merely because unrelated market data is extreme.
            require_exact_validation = True
        with np.errstate(over="ignore", invalid="ignore"):
            raw_targets = reachable_opens * multiple
            scales = np.where(raw_targets < 1.0, 10_000.0, 100.0)
            scaled_targets = raw_targets * scales

        # Binary and Decimal ceilings can differ only next to a tick boundary
        # (or the $1 tick transition). Resolve those sparse candidates with the
        # same Decimal implementation used by live order pricing.
        with np.errstate(over="ignore", invalid="ignore"):
            nearest_ticks = np.rint(scaled_targets)
            tick_tolerance = 32.0 * np.abs(np.spacing(scaled_targets))
            dollar_tolerance = 32.0 * np.abs(np.spacing(raw_targets))
            candidates = (
                require_exact_validation
                | (~np.isfinite(scaled_targets))
                | (
                    (np.abs(scaled_targets - nearest_ticks) <= tick_tolerance)
                    | (np.abs(raw_targets - 1.0) <= dollar_tolerance)
                )
                & (reachable_opens > 0.0)
            )
        for relative_row_idx in np.flatnonzero(candidates):
            row_idx = int(reachable_rows[int(relative_row_idx)])
            override_rows.append(row_idx)
            try:
                exact_target = target_sell_price(
                    float(open_prices[row_idx]),
                    float(multiple),
                )
            except ValueError:
                exact_target = np.nan
            override_values.append(exact_target)
        offsets.append(len(override_rows))
    return (
        np.ascontiguousarray(target_indices, dtype=np.int64),
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(override_rows, dtype=np.int64),
        np.ascontiguousarray(override_values, dtype=np.float64),
    )


@njit(cache=True)
def _target_price_with_override(
    entry_price: float,
    profit_target_multiple: float,
    target_idx: int,
    row_idx: int,
    override_offsets: np.ndarray,
    override_rows: np.ndarray,
    override_values: np.ndarray,
) -> float:
    start = override_offsets[target_idx]
    end = override_offsets[target_idx + 1]
    while start < end:
        midpoint = start + (end - start) // 2
        candidate_row = override_rows[midpoint]
        if candidate_row < row_idx:
            start = midpoint + 1
        else:
            end = midpoint
    if start < override_offsets[target_idx + 1] and override_rows[start] == row_idx:
        return override_values[start]
    return _approximate_target_limit_price(entry_price, profit_target_multiple)


def _initial_target_prices(
    entry_prices: np.ndarray,
    profit_target_values: np.ndarray,
    in_position_values: np.ndarray,
) -> np.ndarray:
    targets = np.full(len(entry_prices), np.nan, dtype=np.float64)
    for idx in np.flatnonzero(in_position_values):
        targets[idx] = target_sell_price(float(entry_prices[idx]), float(profit_target_values[idx]))
    return targets


@njit(cache=True)
def _resting_sell_limit_fill_price(
    open_price: float,
    high_price: float,
    target_price: float,
) -> float:
    if open_price >= target_price:
        # A resting sell limit participates at a favorable opening price.
        return open_price
    if high_price >= target_price:
        return target_price
    return np.nan


@njit(cache=True, parallel=True)
def _run_grid_summary_kernel(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    risk_free_returns: np.ndarray,
    target_lookup_indices: np.ndarray,
    target_override_offsets: np.ndarray,
    target_override_rows: np.ndarray,
    target_override_values: np.ndarray,
    initial_target_price_values: np.ndarray,
    buy_rsi_values: np.ndarray,
    profit_target_values: np.ndarray,
    start_indices: np.ndarray,
    cash_values: np.ndarray,
    share_values: np.ndarray,
    in_position_values: np.ndarray,
    entry_price_values: np.ndarray,
    pending_action_values: np.ndarray,
    prev_equity_values: np.ndarray,
    trades_executed_values: np.ndarray,
    first_equity_values: np.ndarray,
    last_equity_values: np.ndarray,
    running_max_equity_values: np.ndarray,
    return_count_values: np.ndarray,
    return_sum_values: np.ndarray,
    return_sum_squares_values: np.ndarray,
    excess_return_count_values: np.ndarray,
    excess_return_sum_values: np.ndarray,
    excess_return_sum_squares_values: np.ndarray,
    positive_return_count_values: np.ndarray,
    max_drawdown_values: np.ndarray,
    return_mean_values: np.ndarray,
    return_m2_values: np.ndarray,
    excess_return_mean_values: np.ndarray,
    excess_return_m2_values: np.ndarray,
    trading_cost_rate: float,
    rsi_entry_rule: int = RSI_ENTRY_LOWER,
) -> tuple:
    config_count = buy_rsi_values.shape[0]
    row_count = open_prices.shape[0]

    out_cash = cash_values.copy()
    out_shares = share_values.copy()
    out_in_position = in_position_values.copy()
    out_entry_price = entry_price_values.copy()
    out_pending_action = pending_action_values.copy()
    out_prev_equity = prev_equity_values.copy()
    out_trades_executed = trades_executed_values.copy()

    out_first_equity = first_equity_values.copy()
    out_last_equity = last_equity_values.copy()
    out_running_max_equity = running_max_equity_values.copy()
    out_return_count = return_count_values.copy()
    out_return_sum = return_sum_values.copy()
    out_return_sum_squares = return_sum_squares_values.copy()
    out_excess_return_count = excess_return_count_values.copy()
    out_excess_return_sum = excess_return_sum_values.copy()
    out_excess_return_sum_squares = excess_return_sum_squares_values.copy()
    out_positive_return_count = positive_return_count_values.copy()
    out_max_drawdown = max_drawdown_values.copy()
    out_return_mean = return_mean_values.copy()
    out_return_m2 = return_m2_values.copy()
    out_excess_return_mean = excess_return_mean_values.copy()
    out_excess_return_m2 = excess_return_m2_values.copy()
    out_entry_row_index = np.full(config_count, -1, dtype=np.int64)
    updated = np.zeros(config_count, dtype=np.bool_)

    for config_idx in prange(config_count):
        start_idx = start_indices[config_idx]
        if start_idx >= row_count:
            continue

        updated[config_idx] = True
        cash = out_cash[config_idx]
        shares = out_shares[config_idx]
        in_position = out_in_position[config_idx]
        entry_price = out_entry_price[config_idx]
        target_price = initial_target_price_values[config_idx]
        pending_action = out_pending_action[config_idx]
        prev_equity = out_prev_equity[config_idx]
        trades_executed = out_trades_executed[config_idx]

        first_equity = out_first_equity[config_idx]
        last_equity = out_last_equity[config_idx]
        running_max_equity = out_running_max_equity[config_idx]
        return_count = out_return_count[config_idx]
        return_sum = out_return_sum[config_idx]
        return_sum_squares = out_return_sum_squares[config_idx]
        excess_return_count = out_excess_return_count[config_idx]
        excess_return_sum = out_excess_return_sum[config_idx]
        excess_return_sum_squares = out_excess_return_sum_squares[config_idx]
        positive_return_count = out_positive_return_count[config_idx]
        max_drawdown = out_max_drawdown[config_idx]
        return_mean = out_return_mean[config_idx]
        return_m2 = out_return_m2[config_idx]
        excess_return_mean = out_excess_return_mean[config_idx]
        excess_return_m2 = out_excess_return_m2[config_idx]

        buy_rsi = buy_rsi_values[config_idx]
        profit_target_multiple = profit_target_values[config_idx]
        for row_idx in range(start_idx, row_count):
            open_price = open_prices[row_idx]
            high_price = high_prices[row_idx]
            close_price = close_prices[row_idx]
            rsi = rsi_values[row_idx]

            deferred_action = ACTION_NONE
            if pending_action == ACTION_BUY and not in_position:
                cost_multiplier = 1.0 + trading_cost_rate
                if open_price > 0.0 and cash > 0.0 and cost_multiplier > 0.0:
                    candidate_turnover = cash / cost_multiplier
                    candidate_shares = candidate_turnover / open_price
                    if not np.isfinite(candidate_shares) or candidate_shares <= 0.0:
                        # Preserve an invalid executed sizing result for the
                        # Python validator without burning cash first.  A zero
                        # share underflow must not look like a valid no-trade
                        # path merely because no position was opened.
                        shares = np.nan
                        target_price = np.nan
                    else:
                        shares = candidate_shares
                        cash -= candidate_turnover * cost_multiplier
                        # Avoid a tiny negative balance caused by floating-point rounding.
                        cash = max(cash, 0.0)
                        in_position = True
                        entry_price = open_price
                        out_entry_row_index[config_idx] = row_idx
                        target_price = _target_price_with_override(
                            open_price,
                            profit_target_multiple,
                            target_lookup_indices[config_idx],
                            row_idx,
                            target_override_offsets,
                            target_override_rows,
                            target_override_values,
                        )
                        trades_executed += 1
                else:
                    deferred_action = ACTION_BUY

            if in_position and not np.isnan(entry_price) and entry_price > 0.0 and not np.isnan(target_price):
                sell_price = _resting_sell_limit_fill_price(
                    open_price,
                    high_price,
                    target_price,
                )
                if not np.isnan(sell_price):
                    turnover_notional = shares * sell_price
                    cash += turnover_notional
                    cash -= turnover_notional * trading_cost_rate
                    shares = 0.0
                    in_position = False
                    out_entry_row_index[config_idx] = -1
                    entry_price = np.nan
                    target_price = np.nan
                    trades_executed += 1

            equity = cash + shares * close_price
            if not np.isfinite(equity) or equity <= 0.0:
                # Preserve the invalid mark in a validated output field and stop
                # this configuration. In particular, never turn a recovery from
                # an underflowed zero balance into a synthetic zero return.
                prev_equity = equity
                break
            daily_return = equity / prev_equity - 1.0 if prev_equity > 0.0 else 0.0

            next_action = deferred_action
            if deferred_action == ACTION_NONE and not np.isnan(rsi):
                entry_signal = rsi >= buy_rsi if rsi_entry_rule == RSI_ENTRY_UPPER else rsi <= buy_rsi
                if (not in_position) and entry_signal:
                    next_action = ACTION_BUY

            if np.isnan(first_equity):
                first_equity = equity
                last_equity = equity
                running_max_equity = equity
                max_drawdown = 0.0
            else:
                return_count += 1
                return_delta = daily_return - return_mean
                return_mean += return_delta / return_count
                return_m2 += return_delta * (daily_return - return_mean)
                return_sum += daily_return
                return_sum_squares += daily_return * daily_return
                if daily_return > 0.0:
                    positive_return_count += 1

                risk_free_return = risk_free_returns[row_idx]
                if not np.isnan(risk_free_return):
                    excess_return = daily_return - risk_free_return
                    excess_return_count += 1
                    excess_return_delta = excess_return - excess_return_mean
                    excess_return_mean += excess_return_delta / excess_return_count
                    excess_return_m2 += excess_return_delta * (excess_return - excess_return_mean)
                    excess_return_sum += excess_return
                    excess_return_sum_squares += excess_return * excess_return

                last_equity = equity
                running_max_equity = max(running_max_equity, equity)
                if running_max_equity > 0.0:
                    drawdown = equity / running_max_equity - 1.0
                    max_drawdown = min(max_drawdown, drawdown)

            prev_equity = equity
            pending_action = next_action

        out_cash[config_idx] = cash
        out_shares[config_idx] = shares
        out_in_position[config_idx] = in_position
        out_entry_price[config_idx] = entry_price
        out_pending_action[config_idx] = pending_action
        out_prev_equity[config_idx] = prev_equity
        out_trades_executed[config_idx] = trades_executed

        out_first_equity[config_idx] = first_equity
        out_last_equity[config_idx] = last_equity
        out_running_max_equity[config_idx] = running_max_equity
        out_return_count[config_idx] = return_count
        out_return_sum[config_idx] = return_sum
        out_return_sum_squares[config_idx] = return_sum_squares
        out_excess_return_count[config_idx] = excess_return_count
        out_excess_return_sum[config_idx] = excess_return_sum
        out_excess_return_sum_squares[config_idx] = excess_return_sum_squares
        out_positive_return_count[config_idx] = positive_return_count
        out_max_drawdown[config_idx] = max_drawdown
        out_return_mean[config_idx] = return_mean
        out_return_m2[config_idx] = return_m2
        out_excess_return_mean[config_idx] = excess_return_mean
        out_excess_return_m2[config_idx] = excess_return_m2

    return (
        updated,
        out_cash,
        out_shares,
        out_in_position,
        out_entry_price,
        out_pending_action,
        out_prev_equity,
        out_trades_executed,
        out_first_equity,
        out_last_equity,
        out_running_max_equity,
        out_return_count,
        out_return_sum,
        out_return_sum_squares,
        out_excess_return_count,
        out_excess_return_sum,
        out_excess_return_sum_squares,
        out_positive_return_count,
        out_max_drawdown,
        out_return_mean,
        out_return_m2,
        out_excess_return_mean,
        out_excess_return_m2,
        out_entry_row_index,
    )


@njit(cache=True)
def _run_single_equity_curve_kernel(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    risk_free_returns: np.ndarray,
    entry_target_prices: np.ndarray,
    buy_rsi: float,
    initial_capital: float,
    trading_cost_rate: float,
    rsi_entry_rule: int = RSI_ENTRY_LOWER,
) -> tuple:
    row_count = open_prices.shape[0]
    equity_values = np.empty(row_count, dtype=np.float64)
    daily_returns = np.empty(row_count, dtype=np.float64)
    in_position_values = np.empty(row_count, dtype=np.int64)
    action_executed_values = np.empty(row_count, dtype=np.int64)
    pending_action_values = np.empty(row_count, dtype=np.int64)
    trades_executed_values = np.empty(row_count, dtype=np.int64)

    cash = initial_capital
    shares = 0.0
    in_position = False
    entry_price = np.nan
    target_price = np.nan
    pending_action = ACTION_NONE
    prev_equity = initial_capital
    trades_executed = 0

    for row_idx in range(row_count):
        open_price = open_prices[row_idx]
        high_price = high_prices[row_idx]
        close_price = close_prices[row_idx]
        rsi = rsi_values[row_idx]
        action_executed = ACTION_NONE

        deferred_action = ACTION_NONE
        if pending_action == ACTION_BUY and not in_position:
            cost_multiplier = 1.0 + trading_cost_rate
            if open_price > 0.0 and cash > 0.0 and cost_multiplier > 0.0:
                candidate_turnover = cash / cost_multiplier
                candidate_shares = candidate_turnover / open_price
                if not np.isfinite(candidate_shares) or candidate_shares <= 0.0:
                    # Keep an explicit invalid sizing sentinel.  The public
                    # wrapper rejects it before returning any partially
                    # mutated account state.
                    shares = np.nan
                    target_price = np.nan
                else:
                    shares = candidate_shares
                    cash -= candidate_turnover * cost_multiplier
                    cash = max(cash, 0.0)
                    in_position = True
                    entry_price = open_price
                    target_price = entry_target_prices[row_idx]
                    trades_executed += 1
                    action_executed = ACTION_BUY
            else:
                deferred_action = ACTION_BUY

        if in_position and not np.isnan(entry_price) and entry_price > 0.0 and not np.isnan(target_price):
            sell_price = _resting_sell_limit_fill_price(
                open_price,
                high_price,
                target_price,
            )
            if not np.isnan(sell_price):
                turnover_notional = shares * sell_price
                cash += turnover_notional
                cash -= turnover_notional * trading_cost_rate
                shares = 0.0
                in_position = False
                entry_price = np.nan
                target_price = np.nan
                trades_executed += 1
                action_executed = ACTION_SELL

        equity = cash + shares * close_price
        daily_return = equity / prev_equity - 1.0 if prev_equity > 0.0 else 0.0

        next_action = deferred_action
        if deferred_action == ACTION_NONE and not np.isnan(rsi):
            entry_signal = rsi >= buy_rsi if rsi_entry_rule == RSI_ENTRY_UPPER else rsi <= buy_rsi
            if (not in_position) and entry_signal:
                next_action = ACTION_BUY

        equity_values[row_idx] = equity
        daily_returns[row_idx] = daily_return
        in_position_values[row_idx] = 1 if in_position else 0
        action_executed_values[row_idx] = action_executed
        pending_action_values[row_idx] = next_action
        trades_executed_values[row_idx] = trades_executed

        prev_equity = equity
        pending_action = next_action

    return (
        equity_values,
        daily_returns,
        risk_free_returns.copy(),
        in_position_values,
        action_executed_values,
        pending_action_values,
        trades_executed_values,
        cash,
        shares,
        1 if in_position else 0,
        entry_price,
        pending_action,
        prev_equity,
        trades_executed,
    )


def run_grid_summary(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    risk_free_returns: np.ndarray,
    buy_rsi_values: np.ndarray,
    profit_target_values: np.ndarray,
    start_indices: np.ndarray,
    cash_values: np.ndarray,
    share_values: np.ndarray,
    in_position_values: np.ndarray,
    entry_price_values: np.ndarray,
    pending_action_values: np.ndarray,
    prev_equity_values: np.ndarray,
    trades_executed_values: np.ndarray,
    first_equity_values: np.ndarray,
    last_equity_values: np.ndarray,
    running_max_equity_values: np.ndarray,
    return_count_values: np.ndarray,
    return_sum_values: np.ndarray,
    return_sum_squares_values: np.ndarray,
    excess_return_count_values: np.ndarray,
    excess_return_sum_values: np.ndarray,
    excess_return_sum_squares_values: np.ndarray,
    positive_return_count_values: np.ndarray,
    max_drawdown_values: np.ndarray,
    trading_cost_rate: float,
    rsi_entry_rule: int = RSI_ENTRY_LOWER,
    *,
    return_mean_values: np.ndarray | None = None,
    return_m2_values: np.ndarray | None = None,
    excess_return_mean_values: np.ndarray | None = None,
    excess_return_m2_values: np.ndarray | None = None,
    resume_close_values: np.ndarray | None = None,
    history_prefix_observation_counts: np.ndarray | None = None,
) -> tuple:
    """Run grid backtests after validating every array passed to Numba."""
    open_prices = _float_array("open_prices", open_prices)
    row_count = len(open_prices)
    high_prices = _float_array("high_prices", high_prices, length=row_count)
    close_prices = _float_array("close_prices", close_prices, length=row_count)
    _validate_simulation_prices(open_prices, high_prices, close_prices)
    rsi_values = _rsi_observation_array(rsi_values, length=row_count)
    risk_free_returns = _float_array(
        "risk_free_returns",
        risk_free_returns,
        length=row_count,
        allow_nan=True,
    )
    _validate_risk_free_return_domain(risk_free_returns)

    buy_rsi_values = _rsi_threshold_array("buy_rsi_values", buy_rsi_values)
    config_count = len(buy_rsi_values)
    profit_target_values = _float_array("profit_target_values", profit_target_values, length=config_count)
    if np.any((profit_target_values <= 1.0) | (profit_target_values > 100.0)):
        raise ValueError("profit_target_values must be greater than 1.0 and at most 100.0.")
    rsi_entry_rule = _validate_rsi_entry_rule(rsi_entry_rule)

    start_indices = _integer_array("start_indices", start_indices, length=config_count)
    if np.any(start_indices < 0) or np.any(start_indices > row_count):
        raise ValueError(f"start_indices must be between 0 and {row_count}, inclusive.")
    if history_prefix_observation_counts is None:
        history_prefix_observation_counts = np.zeros(config_count, dtype=np.int64)
    else:
        history_prefix_observation_counts = _integer_array(
            "history_prefix_observation_counts",
            history_prefix_observation_counts,
            length=config_count,
        )
        if np.any(history_prefix_observation_counts < 0):
            raise ValueError("history_prefix_observation_counts must be non-negative.")
    cash_values = _float_array("cash_values", cash_values, length=config_count)
    share_values = _float_array("share_values", share_values, length=config_count)
    if np.any(cash_values < 0.0):
        raise ValueError("cash_values must be non-negative.")
    if np.any(share_values < 0.0):
        raise ValueError("share_values must be non-negative.")
    in_position_values = _boolean_array("in_position_values", in_position_values, length=config_count)
    entry_price_values = _float_array("entry_price_values", entry_price_values, length=config_count, allow_nan=True)
    invalid_entry_prices = (~np.isnan(entry_price_values)) & (entry_price_values <= 0.0)
    if invalid_entry_prices.any():
        raise ValueError("entry_price_values must contain positive values or NaN.")
    pending_action_values = _integer_array("pending_action_values", pending_action_values, length=config_count)
    if not np.isin(pending_action_values, [ACTION_NONE, ACTION_BUY, ACTION_SELL]).all():
        raise ValueError("pending_action_values contains an unknown action code.")
    prev_equity_values = _float_array("prev_equity_values", prev_equity_values, length=config_count)
    if np.any(prev_equity_values <= 0.0):
        raise ValueError("prev_equity_values must be positive.")
    trades_executed_values = _integer_array("trades_executed_values", trades_executed_values, length=config_count)

    first_equity_values = _float_array("first_equity_values", first_equity_values, length=config_count, allow_nan=True)
    last_equity_values = _float_array("last_equity_values", last_equity_values, length=config_count, allow_nan=True)
    running_max_equity_values = _float_array(
        "running_max_equity_values",
        running_max_equity_values,
        length=config_count,
        allow_nan=True,
    )
    return_count_values = _integer_array("return_count_values", return_count_values, length=config_count)
    return_sum_values = _float_array("return_sum_values", return_sum_values, length=config_count)
    return_sum_squares_values = _float_array(
        "return_sum_squares_values", return_sum_squares_values, length=config_count
    )
    excess_return_count_values = _integer_array(
        "excess_return_count_values", excess_return_count_values, length=config_count
    )
    excess_return_sum_values = _float_array("excess_return_sum_values", excess_return_sum_values, length=config_count)
    excess_return_sum_squares_values = _float_array(
        "excess_return_sum_squares_values",
        excess_return_sum_squares_values,
        length=config_count,
    )
    positive_return_count_values = _integer_array(
        "positive_return_count_values", positive_return_count_values, length=config_count
    )
    max_drawdown_values = _float_array("max_drawdown_values", max_drawdown_values, length=config_count, allow_nan=True)

    native_centered_moments_available = np.full(
        config_count,
        return_mean_values is not None,
        dtype=np.bool_,
    )
    if (return_mean_values is None) != (return_m2_values is None):
        raise ValueError("return_mean_values and return_m2_values must be provided together.")
    if return_mean_values is None:
        return_mean_values, return_m2_values = _legacy_centered_moment_arrays(
            return_count_values,
            return_sum_values,
            return_sum_squares_values,
        )
    else:
        return_mean_values = _float_array(
            "return_mean_values",
            return_mean_values,
            length=config_count,
        )
        return_m2_values = _float_array(
            "return_m2_values",
            return_m2_values,
            length=config_count,
        )

    if (excess_return_mean_values is None) != (excess_return_m2_values is None):
        raise ValueError("excess_return_mean_values and excess_return_m2_values must be provided together.")
    if excess_return_mean_values is None:
        excess_return_mean_values, excess_return_m2_values = _legacy_centered_moment_arrays(
            excess_return_count_values,
            excess_return_sum_values,
            excess_return_sum_squares_values,
        )
    else:
        excess_return_mean_values = _float_array(
            "excess_return_mean_values",
            excess_return_mean_values,
            length=config_count,
        )
        excess_return_m2_values = _float_array(
            "excess_return_m2_values",
            excess_return_m2_values,
            length=config_count,
        )

    if resume_close_values is None:
        resume_close_values = np.full(config_count, np.nan, dtype=np.float64)
    else:
        resume_close_values = _float_array(
            "resume_close_values",
            resume_close_values,
            length=config_count,
            allow_nan=True,
        ).copy()
        invalid_resume_closes = (~np.isnan(resume_close_values)) & (resume_close_values <= 0.0)
        if invalid_resume_closes.any():
            raise ValueError("resume_close_values must contain positive values or NaN.")
    for config_idx, start_idx in enumerate(start_indices):
        if start_idx <= 0:
            continue
        derived_resume_close = float(close_prices[start_idx - 1])
        supplied_resume_close = float(resume_close_values[config_idx])
        if not np.isnan(supplied_resume_close) and not _state_values_are_close(
            supplied_resume_close,
            derived_resume_close,
        ):
            raise ValueError(
                "resume_close_values must match the closing price immediately before each "
                f"resume index; mismatch at config index {config_idx}."
            )
        resume_close_values[config_idx] = derived_resume_close

    for name, values in (
        ("first_equity_values", first_equity_values),
        ("last_equity_values", last_equity_values),
        ("running_max_equity_values", running_max_equity_values),
    ):
        invalid = (~np.isnan(values)) & (values <= 0.0)
        if invalid.any():
            raise ValueError(f"{name} must contain positive values or NaN.")
    for name, values in (
        ("trades_executed_values", trades_executed_values),
        ("return_count_values", return_count_values),
        ("excess_return_count_values", excess_return_count_values),
        ("positive_return_count_values", positive_return_count_values),
    ):
        if np.any(values < 0):
            raise ValueError(f"{name} must be non-negative.")

    _validate_resumable_state(
        start_indices=start_indices,
        history_prefix_observation_counts=history_prefix_observation_counts,
        cash_values=cash_values,
        share_values=share_values,
        in_position_values=in_position_values,
        entry_price_values=entry_price_values,
        pending_action_values=pending_action_values,
        prev_equity_values=prev_equity_values,
        trades_executed_values=trades_executed_values,
        first_equity_values=first_equity_values,
        last_equity_values=last_equity_values,
        running_max_equity_values=running_max_equity_values,
        return_count_values=return_count_values,
        return_sum_values=return_sum_values,
        return_sum_squares_values=return_sum_squares_values,
        excess_return_count_values=excess_return_count_values,
        excess_return_sum_values=excess_return_sum_values,
        excess_return_sum_squares_values=excess_return_sum_squares_values,
        positive_return_count_values=positive_return_count_values,
        max_drawdown_values=max_drawdown_values,
        return_mean_values=return_mean_values,
        return_m2_values=return_m2_values,
        excess_return_mean_values=excess_return_mean_values,
        excess_return_m2_values=excess_return_m2_values,
        resume_close_values=resume_close_values,
        native_centered_moments_available=native_centered_moments_available,
    )

    trading_cost_rate = _finite_scalar("trading_cost_rate", trading_cost_rate)
    if not 0.0 <= trading_cost_rate < 1.0:
        raise ValueError("trading_cost_rate must be non-negative and less than 1.0.")
    entry_candidate_indices = _grid_entry_candidate_indices(
        rsi_values,
        buy_rsi_values,
        start_indices,
        in_position_values,
        pending_action_values,
        rsi_entry_rule,
    )
    (
        target_lookup_indices,
        target_override_offsets,
        target_override_rows,
        target_override_values,
    ) = _target_price_overrides(
        open_prices,
        profit_target_values,
        start_indices,
        entry_candidate_indices,
    )
    initial_target_price_values = _initial_target_prices(
        entry_price_values,
        profit_target_values,
        in_position_values & (start_indices < row_count),
    )

    results = _run_grid_summary_kernel(
        open_prices,
        high_prices,
        close_prices,
        rsi_values,
        risk_free_returns,
        target_lookup_indices,
        target_override_offsets,
        target_override_rows,
        target_override_values,
        initial_target_price_values,
        buy_rsi_values,
        profit_target_values,
        start_indices,
        cash_values,
        share_values,
        in_position_values,
        entry_price_values,
        pending_action_values,
        prev_equity_values,
        trades_executed_values,
        first_equity_values,
        last_equity_values,
        running_max_equity_values,
        return_count_values,
        return_sum_values,
        return_sum_squares_values,
        excess_return_count_values,
        excess_return_sum_values,
        excess_return_sum_squares_values,
        positive_return_count_values,
        max_drawdown_values,
        return_mean_values,
        return_m2_values,
        excess_return_mean_values,
        excess_return_m2_values,
        trading_cost_rate,
        rsi_entry_rule,
    )
    invalid_share_configs = np.flatnonzero(~np.isfinite(results[2]))
    if len(invalid_share_configs):
        config_idx = int(invalid_share_configs[0])
        raise ValueError(
            "Position sizing produced non-finite shares at config index "
            f"{config_idx}; the executed entry open price is too small for the available capital."
        )
    if np.any(results[23] < -1) or np.any(results[23] >= row_count):
        raise ValueError("Backtest produced an invalid entry row index.")
    target_price_function = getattr(
        _target_price_with_override,
        "py_func",
        _target_price_with_override,
    )
    for config_idx, entry_row_idx in enumerate(results[23]):
        if entry_row_idx < 0:
            continue
        prepared_target = target_price_function(
            float(open_prices[entry_row_idx]),
            float(profit_target_values[config_idx]),
            int(target_lookup_indices[config_idx]),
            int(entry_row_idx),
            target_override_offsets,
            target_override_rows,
            target_override_values,
        )
        if not np.isfinite(prepared_target) or prepared_target <= 0.0:
            # Re-raise the public Decimal pricing error only for an entry that
            # the simulation actually executed.
            target_sell_price(
                float(open_prices[entry_row_idx]),
                float(profit_target_values[config_idx]),
            )
            raise ValueError("Backtest executed an entry without a valid target price.")
    _validate_grid_results(results)
    result_resume_close_values = resume_close_values.copy()
    if row_count:
        result_resume_close_values[results[0]] = close_prices[-1]
    _validate_resumable_state(
        start_indices=np.full(config_count, row_count, dtype=np.int64),
        history_prefix_observation_counts=history_prefix_observation_counts,
        cash_values=results[1],
        share_values=results[2],
        in_position_values=results[3],
        entry_price_values=results[4],
        pending_action_values=results[5],
        prev_equity_values=results[6],
        trades_executed_values=results[7],
        first_equity_values=results[8],
        last_equity_values=results[9],
        running_max_equity_values=results[10],
        return_count_values=results[11],
        return_sum_values=results[12],
        return_sum_squares_values=results[13],
        excess_return_count_values=results[14],
        excess_return_sum_values=results[15],
        excess_return_sum_squares_values=results[16],
        positive_return_count_values=results[17],
        max_drawdown_values=results[18],
        return_mean_values=results[19],
        return_m2_values=results[20],
        excess_return_mean_values=results[21],
        excess_return_m2_values=results[22],
        resume_close_values=result_resume_close_values,
        native_centered_moments_available=(native_centered_moments_available | results[0]),
    )
    return results


def run_single_equity_curve(
    open_prices: np.ndarray,
    high_prices: np.ndarray,
    close_prices: np.ndarray,
    rsi_values: np.ndarray,
    risk_free_returns: np.ndarray,
    buy_rsi: float,
    profit_target_multiple: float,
    initial_capital: float,
    trading_cost_rate: float,
    rsi_entry_rule: int = RSI_ENTRY_LOWER,
) -> tuple:
    """Run one backtest after validating every array passed to Numba."""
    open_prices = _float_array("open_prices", open_prices)
    row_count = len(open_prices)
    high_prices = _float_array("high_prices", high_prices, length=row_count)
    close_prices = _float_array("close_prices", close_prices, length=row_count)
    _validate_simulation_prices(open_prices, high_prices, close_prices)
    rsi_values = _rsi_observation_array(rsi_values, length=row_count)
    risk_free_returns = _float_array(
        "risk_free_returns",
        risk_free_returns,
        length=row_count,
        allow_nan=True,
    )
    _validate_risk_free_return_domain(risk_free_returns)
    buy_rsi = _rsi_threshold_scalar("buy_rsi", buy_rsi)
    profit_target_multiple = _finite_scalar("profit_target_multiple", profit_target_multiple)
    if not 1.0 < profit_target_multiple <= 100.0:
        raise ValueError("profit_target_multiple must be greater than 1.0 and at most 100.0.")
    initial_capital = _finite_scalar("initial_capital", initial_capital)
    if initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive.")
    trading_cost_rate = _finite_scalar("trading_cost_rate", trading_cost_rate)
    if not 0.0 <= trading_cost_rate < 1.0:
        raise ValueError("trading_cost_rate must be non-negative and less than 1.0.")
    rsi_entry_rule = _validate_rsi_entry_rule(rsi_entry_rule)
    entry_candidate_indices = _entry_candidate_indices(
        rsi_values,
        buy_rsi,
        0,
        False,
        ACTION_NONE,
        rsi_entry_rule,
    )
    entry_target_prices = np.full(row_count, np.nan, dtype=np.float64)
    for row_idx in entry_candidate_indices:
        try:
            entry_target_prices[row_idx] = target_sell_price(
                float(open_prices[row_idx]),
                profit_target_multiple,
            )
        except ValueError:
            # Keep the sentinel through the compiled simulation. It becomes an
            # error below only when this candidate entry was really executed.
            entry_target_prices[row_idx] = np.nan

    results = _run_single_equity_curve_kernel(
        open_prices,
        high_prices,
        close_prices,
        rsi_values,
        risk_free_returns,
        entry_target_prices,
        buy_rsi,
        initial_capital,
        trading_cost_rate,
        rsi_entry_rule,
    )
    if not np.isfinite(results[8]):
        raise ValueError(
            "Position sizing produced non-finite shares; the available capital and "
            "executed entry open price cannot be represented safely."
        )
    invalid_target_entries = np.flatnonzero((results[4] == ACTION_BUY) & np.isnan(entry_target_prices))
    if len(invalid_target_entries):
        entry_row_idx = int(invalid_target_entries[0])
        target_sell_price(
            float(open_prices[entry_row_idx]),
            profit_target_multiple,
        )
        raise ValueError("Backtest executed an entry without a valid target price.")
    _validate_single_results(results)
    return results
