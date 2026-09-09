from __future__ import annotations

import datetime as datetime_module
import hashlib
import json
import math
import re
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, DecimalException
from functools import cache
from numbers import Number

import numpy as np
import pandas as pd

from .accounting import (
    MANAGED_NOTIONAL_MIN_ULP_ALLOWANCE,
    MANAGED_NOTIONAL_RELATIVE_TOLERANCE,
    MANAGED_QUANTITY_ABSOLUTE_TOLERANCE,
    MANAGED_QUANTITY_MAX_TOLERANCE,
    MANAGED_QUANTITY_RELATIVE_TOLERANCE,
    MANAGED_VALUE_MAX_RECONCILIATION_TOLERANCE,
    managed_quantity_tolerance,
    managed_residual_notional_tolerance,
    managed_residual_quantity_is_negligible,
    managed_value_reconciliation_tolerance,
)
from .backtest import _require_representable_metric, initial_strategy_state
from .config import (
    RISK_FREE_SYMBOL,
    BacktestConfig,
    validate_strategy_simulation_configuration,
)
from .indicators import compute_rsi_details, rsi_value_from_average_gain_loss
from .optimized_backtest import (
    ACTION_BUY,
    ACTION_NONE,
    ACTION_SELL,
    RSI_ENTRY_LOWER,
    RSI_ENTRY_UPPER,
    _fully_determined_return_rollup_is_consistent,
    _positive_return_count_is_feasible,
    _positive_return_path_is_feasible,
    _return_rollup_equity_endpoints_are_consistent,
    _single_return_rollup_is_consistent,
    _strategy_return_moments_respect_lower_bound,
    run_grid_summary,
    run_single_equity_curve,
)
from .pricing import (
    canonical_alpaca_limit_price,
    canonical_alpaca_order_quantity,
)
from .pricing import (
    target_sell_price as _target_sell_price,
)
from .runtime_files import (
    prepare_private_runtime_file,
    revalidate_active_sqlite_runtime_file,
    revalidate_private_runtime_file,
)


class SellFillQuantityRegressionError(ValueError):
    """Raised when a broker reports less cumulative fill than was already recorded."""


@dataclass(frozen=True)
class AlpacaManagedBuyIntentClaim:
    """A buy-submission claim with a generation fence.

    Iteration intentionally yields the historical ``(position_id, claimed)``
    pair so existing callers can keep unpacking the result while submission
    owners use ``attempt_count`` and ``state_revision`` to fence their broker
    request and response.
    """

    position_id: int | None
    claimed: bool
    attempt_count: int
    state_revision: int
    symbol_conflict: bool = False
    intent_conflict: bool = False

    def __iter__(self) -> Iterator[int | bool | None]:
        yield self.position_id
        yield self.claimed


@dataclass(frozen=True)
class AlpacaManagedSellRenewalClaim:
    """Exact managed-state generation owned by a sell-renewal worker."""

    state_revision: int
    sell_filled_qty: float | None
    sell_renewal_count: int
    requested_at: str


@dataclass(frozen=True)
class AlpacaManagedSellRenewalSnapshot:
    """Complete local state still owned by an exact renewal lease."""

    state_revision: int
    buy_status: str | None
    filled_qty: float | None
    filled_avg_price: float | None
    target_sell_price: float | None
    remaining_qty: float | None
    sell_status: str | None
    sell_alpaca_order_id: str | None
    sell_filled_qty: float | None
    sell_renewal_count: int
    requested_at: str | None


@dataclass(frozen=True)
class AlpacaManagedSellReplacementSubmissionClaim:
    """Replacement state atomically leased for final broker validation."""

    remaining_qty: float
    state_revision: int
    claimed_at: str


@dataclass(frozen=True)
class AlpacaManagedSellSubmissionClaim:
    """Current protection atomically leased for an initial or retry submission."""

    remaining_qty: float
    target_sell_price: float
    state_revision: int
    claimed_at: str
    buy_status: str | None


SUMMARY_ROLLUP_COLUMNS = {
    "first_equity": "REAL",
    "last_equity": "REAL",
    "running_max_equity": "REAL",
    "return_count": "INTEGER",
    "return_sum": "REAL",
    "return_sum_squares": "REAL",
    "excess_return_count": "INTEGER",
    "excess_return_sum": "REAL",
    "excess_return_sum_squares": "REAL",
    "positive_return_count": "INTEGER",
    "return_mean": "REAL",
    "return_m2": "REAL",
    "excess_return_mean": "REAL",
    "excess_return_m2": "REAL",
}

STRATEGY_SUMMARY_COLUMNS = {
    **SUMMARY_ROLLUP_COLUMNS,
    "integrity_digest": "TEXT",
}

STRATEGY_EQUITY_COLUMNS = {
    "integrity_digest": "TEXT",
}

_STRATEGY_SUMMARY_INTEGRITY_COLUMNS = (
    "asset_symbol",
    "signal_symbol",
    "buy_rsi",
    "profit_target_multiple",
    "start_date",
    "end_date",
    "trading_days",
    "trades_executed",
    "total_return",
    "cagr",
    "annualized_vol",
    "sharpe",
    "kelly_fraction",
    "max_drawdown",
    "hit_rate",
    "first_equity",
    "last_equity",
    "running_max_equity",
    "return_count",
    "return_sum",
    "return_sum_squares",
    "excess_return_count",
    "excess_return_sum",
    "excess_return_sum_squares",
    "positive_return_count",
    "return_mean",
    "return_m2",
    "excess_return_mean",
    "excess_return_m2",
)
_STRATEGY_SUMMARY_INTEGRITY_COLUMN_SQL = ", ".join(_STRATEGY_SUMMARY_INTEGRITY_COLUMNS)
_STRATEGY_SUMMARY_INTEGER_COLUMNS = frozenset(
    {
        "trading_days",
        "trades_executed",
        "return_count",
        "excess_return_count",
        "positive_return_count",
    }
)
_STRATEGY_SUMMARY_TEXT_COLUMNS = frozenset({"asset_symbol", "signal_symbol", "start_date", "end_date"})

STRATEGY_STATE_COLUMNS = {
    "integrity_digest": "TEXT",
    "entry_date": "TEXT",
}

ALPACA_MANAGED_POSITION_COLUMNS = {
    "alpaca_asset_id": "TEXT",
    "state_revision": "INTEGER NOT NULL DEFAULT 0",
    "buy_observation_broker_updated_at": "TEXT",
    "buy_fill_broker_updated_at": "TEXT",
    "buy_fill_component_revisions": "TEXT",
    "buy_fill_pending_observation": "TEXT",
    "buy_causality_quarantine": "TEXT",
    "buy_cancellation_alpaca_order_ids": "TEXT",
    "closed_correction_audited_at": "TEXT",
    "closed_sell_shortfall_reopen_pending": "INTEGER NOT NULL DEFAULT 0",
    "workflow": "TEXT",
    "sell_order_namespace": "TEXT",
    "sell_client_order_id": "TEXT",
    "buy_order_qty": "REAL",
    "buy_order_limit_price": "REAL",
    "buy_submission_claimed_at": "TEXT",
    "buy_submission_attempt_count": "INTEGER NOT NULL DEFAULT 1",
    "sell_expires_at": "TEXT",
    "sell_order_qty": "REAL",
    "sell_order_limit_price": "REAL",
    "sell_observation_broker_updated_at": "TEXT",
    "sell_observation_filled_qty": "REAL",
    "sell_submission_retry_claimed_at": "TEXT",
    "sell_renewal_count": "INTEGER NOT NULL DEFAULT 0",
    "sell_renewal_requested_at": "TEXT",
    "sell_filled_qty": "REAL",
    "sell_filled_avg_price": "REAL",
    "sell_filled_at": "TEXT",
    "realized_pl": "REAL",
    "realized_pl_pct": "REAL",
    "sold_qty": "REAL NOT NULL DEFAULT 0",
    "sold_value": "REAL NOT NULL DEFAULT 0",
    "remaining_qty": "REAL",
}

ALPACA_MANAGED_SELL_FILL_COLUMNS = {
    "broker_updated_at": "TEXT",
    "submitted_qty": "REAL",
    "submitted_limit_price": "REAL",
}

_ALPACA_STATE_REVISION_TRIGGER_NAME = "alpaca_managed_positions_increment_state_revision"
_ALPACA_STATE_REVISION_GUARD_TRIGGER_NAME = "alpaca_managed_positions_validate_revision_update"
_ALPACA_STATE_REVISION_GUARD_TRIGGER_SQL = f"""
    CREATE TRIGGER {_ALPACA_STATE_REVISION_GUARD_TRIGGER_NAME}
    BEFORE UPDATE ON alpaca_managed_positions
    FOR EACH ROW
    WHEN TYPEOF(NEW.id) != 'integer'
      OR NEW.id != OLD.id
      OR TYPEOF(OLD.state_revision) != 'integer'
      OR OLD.state_revision < 0
      OR TYPEOF(NEW.state_revision) != 'integer'
      OR NEW.state_revision < 0
      OR NEW.state_revision NOT IN (OLD.state_revision, OLD.state_revision + 1)
    BEGIN
        SELECT RAISE(ABORT, 'invalid managed-position revision transition');
    END
"""
_ALPACA_STATE_REVISION_TRIGGER_SQL = f"""
    CREATE TRIGGER {_ALPACA_STATE_REVISION_TRIGGER_NAME}
    AFTER UPDATE ON alpaca_managed_positions
    FOR EACH ROW
    WHEN NEW.state_revision = OLD.state_revision
    BEGIN
        UPDATE alpaca_managed_positions
        SET state_revision = OLD.state_revision + 1
        WHERE id = OLD.id;
    END
"""

STRATEGY_STATE_SCHEMA_VERSION = 13
RSI_ENTRY_RULE_LABELS = {
    "lower": RSI_ENTRY_LOWER,
    "upper": RSI_ENTRY_UPPER,
}
_MARKET_DATA_FIELDS = ("Open", "High", "Low", "Close", "Volume")
_MARKET_DATA_PROVIDERS_ATTR = "market_data_providers"
_MARKET_DATA_TEMPORAL_TYPES = (datetime_module.date, datetime, timedelta, np.datetime64, np.timedelta64)


class AssetMarketDataError(ValueError):
    """Raised when one asset's market inputs cannot support a strategy run."""


def _rollback_and_release_savepoint(
    conn: sqlite3.Connection,
    savepoint_name: str,
    failure: BaseException,
) -> None:
    """Best-effort savepoint cleanup without replacing the original failure."""
    cleanup_failures: list[tuple[str, BaseException]] = []
    rollback_succeeded = False
    try:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
    except BaseException as exc:
        cleanup_failures.append(("roll back", exc))
    else:
        rollback_succeeded = True
    # Releasing an outermost savepoint after ROLLBACK TO failed can commit the
    # very partial transaction that cleanup is meant to discard. Only release
    # a savepoint whose rollback is known to have succeeded; otherwise fall
    # through to a full connection rollback below.
    if rollback_succeeded:
        try:
            conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        except BaseException as exc:
            cleanup_failures.append(("release", exc))
    if cleanup_failures:
        try:
            conn.rollback()
        except BaseException as exc:
            cleanup_failures.append(("roll back the connection", exc))
            # A connection whose transaction could not be rolled back must not
            # escape back to its caller: a later commit could otherwise make
            # the partial savepoint work durable. Closing a sqlite3 connection
            # discards its pending transaction and makes accidental reuse fail
            # closed. Preserve any close failure as another note on the
            # original exception rather than replacing it.
            try:
                conn.close()
            except BaseException as close_exc:
                cleanup_failures.append(("close the connection after rollback failure", close_exc))
    for action, cleanup_failure in cleanup_failures:
        failure.add_note(f"Failed to {action} while cleaning SQLite savepoint {savepoint_name}: {cleanup_failure}")


def _rollback_after_commit_failure(
    conn: sqlite3.Connection,
    failure: BaseException,
) -> None:
    """Discard a failed commit or make its still-pending transaction unusable."""
    try:
        conn.rollback()
    except BaseException as rollback_failure:
        failure.add_note(f"Failed to roll back after commit failure: {rollback_failure}")
        # SQLite can leave a transaction active after a failed commit. If its
        # rollback also fails, closing is the only way to prevent a later,
        # unrelated commit from making the failed operation durable.
        try:
            conn.close()
        except BaseException as close_failure:
            failure.add_note(f"Failed to close the connection after commit rollback failure: {close_failure}")


def _commit_owned_transaction(conn: sqlite3.Connection) -> None:
    """Commit an owned transaction without leaving failed work pending."""
    try:
        conn.commit()
    except BaseException as exc:
        _rollback_after_commit_failure(conn, exc)
        raise


def _rollback_owned_operation_after_failure(
    conn: sqlite3.Connection,
    failure: BaseException,
    *,
    operation: str,
) -> None:
    """End an operation-owned transaction or make the connection unusable."""
    try:
        conn.rollback()
    except BaseException as rollback_failure:
        failure.add_note(f"Failed to roll back {operation}: {rollback_failure}")
        try:
            conn.close()
        except BaseException as close_failure:
            failure.add_note(f"Failed to close the connection after {operation} rollback failure: {close_failure}")


def _rollback_owned_operation_if_active(
    conn: sqlite3.Connection,
    failure: BaseException,
    *,
    operation: str,
) -> None:
    """Roll back an operation-owned transaction unless cleanup already closed it."""
    try:
        transaction_active = conn.in_transaction
    except sqlite3.ProgrammingError:
        # Savepoint cleanup closes the connection when it cannot make the
        # pending transaction safe. That is already fail-closed.
        return
    except BaseException as state_failure:
        failure.add_note(f"Failed to inspect the SQLite transaction after {operation} failed: {state_failure}")
        transaction_active = True
    if transaction_active:
        _rollback_owned_operation_after_failure(
            conn,
            failure,
            operation=operation,
        )


@contextmanager
def _consistent_storage_read_snapshot(conn: sqlite3.Connection) -> Iterator[None]:
    """Keep a multi-query authenticated read on one SQLite snapshot."""
    owns_snapshot = not conn.in_transaction
    if owns_snapshot:
        try:
            conn.execute("BEGIN")
        except BaseException as exc:
            _rollback_owned_operation_if_active(
                conn,
                exc,
                operation="storage read-snapshot setup",
            )
            raise
    try:
        yield
    except BaseException as exc:
        if owns_snapshot:
            _rollback_owned_operation_if_active(
                conn,
                exc,
                operation="storage read snapshot",
            )
        raise
    else:
        if owns_snapshot:
            try:
                conn.rollback()
            except BaseException as exc:
                _rollback_owned_operation_after_failure(
                    conn,
                    exc,
                    operation="storage read snapshot",
                )
                raise


def _execute_owned_operation_step(
    conn: sqlite3.Connection,
    sql: str,
    parameters: tuple[object, ...] = (),
    *,
    owns_transaction: bool,
    operation: str,
) -> sqlite3.Cursor:
    """Execute one step and clean up a transaction started by this operation."""
    try:
        return conn.execute(sql, parameters)
    except BaseException as exc:
        if owns_transaction:
            _rollback_owned_operation_if_active(
                conn,
                exc,
                operation=operation,
            )
        raise


def _execute_returning_owned_operation_step[ReturningValue](
    conn: sqlite3.Connection,
    sql: str,
    parameters: tuple[object, ...] = (),
    *,
    owns_transaction: bool,
    operation: str,
    decode: Callable[[object], ReturningValue],
) -> ReturningValue | None:
    """Execute, fetch, and decode one mutation without exposing partial work.

    SQLite applies a DML statement before a ``RETURNING`` row is decoded. A
    converter or row factory can therefore fail from ``fetchone()`` after the
    write has acquired a lock. Keep the statement behind its own savepoint so
    caller-owned transactions retain earlier work, while operation-owned
    transactions are fully ended on any execute, fetch, or decode failure.
    """
    savepoint_name = "managed_returning_operation"
    savepoint_active = False
    try:
        if owns_transaction and not conn.in_transaction:
            conn.execute("BEGIN")
        conn.execute(f"SAVEPOINT {savepoint_name}")
        savepoint_active = True
        cursor = conn.execute(sql, parameters)
        row = cursor.fetchone()
        decoded = None if row is None else decode(row)
        conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        savepoint_active = False
        return decoded
    except BaseException as exc:
        if savepoint_active:
            _rollback_and_release_savepoint(conn, savepoint_name, exc)
        if owns_transaction:
            _rollback_owned_operation_if_active(
                conn,
                exc,
                operation=operation,
            )
        raise


@contextmanager
def _managed_accounting_composite_savepoint(
    conn: sqlite3.Connection,
    savepoint_name: str,
) -> Iterator[None]:
    """Keep a parent/ledger mutation atomic without committing its caller early."""
    owns_transaction = not conn.in_transaction
    savepoint_active = False
    try:
        if owns_transaction:
            # An outermost SQLite SAVEPOINT is committed by RELEASE. Start an
            # explicit transaction first so the operation's final owned commit
            # remains the sole durability boundary and can still be rolled back
            # when commit itself fails.
            conn.execute("BEGIN")
        conn.execute(f"SAVEPOINT {savepoint_name}")
        savepoint_active = True
        yield
        conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        savepoint_active = False
    except BaseException as exc:
        if savepoint_active:
            _rollback_and_release_savepoint(conn, savepoint_name, exc)
        if owns_transaction:
            _rollback_owned_operation_if_active(
                conn,
                exc,
                operation=f"managed-accounting composite {savepoint_name}",
            )
        raise


def _normalize_optional_managed_buy_intent(
    buy_order_qty: float | None,
    buy_order_limit_price: float | None,
) -> tuple[float | None, float | None]:
    """Normalize a complete buy intent while retaining all-NULL legacy rows."""
    if buy_order_qty is None and buy_order_limit_price is None:
        return None, None
    if buy_order_qty is None or buy_order_limit_price is None:
        raise ValueError("A managed Alpaca buy intent requires both quantity and limit price.")
    if (
        isinstance(buy_order_qty, bool)
        or not isinstance(buy_order_qty, Number)
        or isinstance(buy_order_limit_price, bool)
        or not isinstance(buy_order_limit_price, Number)
    ):
        raise ValueError("A managed Alpaca buy intent quantity and limit price must be numeric.")
    normalized_qty = canonical_alpaca_order_quantity(
        buy_order_qty,
        field_name="A managed Alpaca buy intent quantity",
    )
    exact_qty = Decimal(str(normalized_qty))
    if exact_qty != exact_qty.to_integral_value() or not normalized_qty.is_integer():
        raise ValueError("A managed Alpaca buy intent quantity must be a whole-share value.")
    normalized_limit_price = canonical_alpaca_limit_price(
        buy_order_limit_price,
        field_name="A managed Alpaca buy intent limit price",
    )
    return normalized_qty, normalized_limit_price


def _normalize_managed_strategy_economics(
    buy_rsi: object,
    profit_target_multiple: object,
) -> tuple[float, float]:
    """Validate immutable managed-position strategy inputs before persistence."""
    invalid_types = (
        bool,
        np.bool_,
        complex,
        np.complexfloating,
        datetime_module.date,
        timedelta,
        np.datetime64,
        np.timedelta64,
        np.ndarray,
        str,
        bytes,
        bytearray,
    )
    if isinstance(buy_rsi, invalid_types) or isinstance(profit_target_multiple, invalid_types):
        raise ValueError("Managed strategy economics must be finite numeric scalars in the supported ranges.")
    try:
        normalized_buy_rsi = float(buy_rsi)
        normalized_profit_target = float(profit_target_multiple)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Managed strategy economics must be finite numeric scalars in the supported ranges.") from exc
    if not math.isfinite(normalized_buy_rsi) or not 0.0 <= normalized_buy_rsi <= 100.0:
        raise ValueError("A managed buy RSI must be finite and between 0 and 100, inclusive.")
    if not math.isfinite(normalized_profit_target) or not 1.0 < normalized_profit_target <= 100.0:
        raise ValueError("A managed profit-target multiple must be finite, greater than 1, and at most 100.")
    return normalized_buy_rsi, normalized_profit_target


def _normalize_managed_target_sell_price(value: object) -> float:
    return canonical_alpaca_limit_price(
        value,
        field_name="A managed Alpaca target sell price",
    )


def _managed_realized_pl_values(
    *,
    sold_value: float,
    matched_qty: float,
    buy_price: float,
) -> tuple[float, float]:
    """Derive finite authoritative realized P/L or reject the observation."""
    cost_basis = float(matched_qty) * float(buy_price)
    if not math.isfinite(cost_basis) or cost_basis <= 0:
        raise ValueError("Managed realized P/L requires a finite positive cost basis.")
    realized_pl = float(sold_value) - cost_basis
    realized_pl_pct = realized_pl / cost_basis * 100.0
    if not math.isfinite(realized_pl) or not math.isfinite(realized_pl_pct):
        raise ValueError("Managed realized P/L must remain finite.")
    return realized_pl, realized_pl_pct


def _decode_managed_realized_pl_revision(row: object) -> int:
    state_revision = int(row[0])  # type: ignore[index]
    for value in (row[1], row[2]):  # type: ignore[index]
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("Managed realized P/L must remain finite.")
    return state_revision


def _normalize_optional_managed_sell_intent(
    submitted_qty: float | None,
    submitted_limit_price: float | None,
) -> tuple[float | None, float | None]:
    """Normalize independently optional sell-generation economics before writes."""
    if (
        isinstance(submitted_qty, bool)
        or (submitted_qty is not None and not isinstance(submitted_qty, Number))
        or isinstance(submitted_limit_price, bool)
        or (submitted_limit_price is not None and not isinstance(submitted_limit_price, Number))
    ):
        raise ValueError("Managed sell generation economics must be numeric.")
    try:
        exact_qty = None if submitted_qty is None else Decimal(str(submitted_qty))
        normalized_qty = None if exact_qty is None else float(exact_qty)
    except (DecimalException, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("A managed sell generation quantity must be positive and finite.") from exc
    if normalized_qty is not None:
        assert exact_qty is not None
        if not exact_qty.is_finite() or not math.isfinite(normalized_qty) or normalized_qty <= 0:
            raise ValueError("A managed sell generation quantity must be positive and finite.")
        if Decimal(str(normalized_qty)) != exact_qty:
            raise ValueError("A managed sell generation quantity must be exactly representable for persistence.")
        if normalized_qty.is_integer():
            # Whole-share generations can reach the GTC broker boundary, so
            # their durable value must survive Alpaca's nine-decimal wire
            # serialization. Fractional generations are retained for legacy
            # accounting and are deliberately fenced before any GTC POST.
            normalized_qty = canonical_alpaca_order_quantity(
                submitted_qty,
                field_name="A managed sell generation quantity",
            )
    normalized_limit_price = (
        None
        if submitted_limit_price is None
        else canonical_alpaca_limit_price(
            submitted_limit_price,
            field_name="A managed sell generation limit price",
        )
    )
    return normalized_qty, normalized_limit_price


def _normalize_required_managed_sell_intent(
    submitted_qty: object,
    submitted_limit_price: object,
) -> tuple[float, float]:
    """Return a complete durable sell intent or reject legacy/invalid values."""
    normalized_qty, normalized_limit_price = _normalize_optional_managed_sell_intent(
        submitted_qty,  # type: ignore[arg-type]
        submitted_limit_price,  # type: ignore[arg-type]
    )
    if normalized_qty is None or normalized_limit_price is None:
        raise ValueError("A managed sell intent requires both quantity and limit price.")
    return normalized_qty, normalized_limit_price


def _decode_required_managed_sell_intent(row: object) -> tuple[float, float]:
    return _normalize_required_managed_sell_intent(
        row[0],  # type: ignore[index]
        row[1],  # type: ignore[index]
    )


def _decode_managed_buy_submission_confirmation(row: object) -> int:
    buy_order_qty, buy_order_limit_price = _normalize_optional_managed_buy_intent(
        row[1],  # type: ignore[index]
        row[2],  # type: ignore[index]
    )
    if buy_order_qty is None or buy_order_limit_price is None:
        raise ValueError("A managed Alpaca buy-submission confirmation requires a complete immutable intent.")
    return int(row[0])  # type: ignore[index]


def _decode_managed_sell_submission_claim(
    row: object,
    *,
    claimed_at: str,
) -> AlpacaManagedSellSubmissionClaim:
    remaining_qty, target_sell_price = _decode_required_managed_sell_intent(row)
    return AlpacaManagedSellSubmissionClaim(
        remaining_qty=remaining_qty,
        target_sell_price=target_sell_price,
        state_revision=int(row[2]),  # type: ignore[index]
        claimed_at=claimed_at,
        buy_status=None if row[3] is None else str(row[3]),  # type: ignore[index]
    )


def _normalize_managed_sell_fill_economics(
    filled_qty: object,
    filled_value: object,
) -> tuple[float, float]:
    """Return one finite, internally consistent cumulative sell observation."""
    try:
        normalized_qty = float(filled_qty)
        normalized_value = float(filled_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Managed sell-fill economics must be finite and non-negative.") from exc
    if (
        not math.isfinite(normalized_qty)
        or not math.isfinite(normalized_value)
        or normalized_qty < 0
        or normalized_value < 0
        or ((normalized_qty == 0) != (normalized_value == 0))
    ):
        raise ValueError("Managed sell-fill economics must be finite, non-negative, and consistently zero.")
    if normalized_qty > 0:
        unit_price = normalized_value / normalized_qty
        if not math.isfinite(unit_price) or unit_price <= 0:
            raise ValueError("Managed sell-fill economics require a finite positive unit price.")
    return normalized_qty, normalized_value


def _normalize_optional_managed_buy_fill_economics(
    filled_qty: object | None,
    filled_avg_price: object | None,
) -> tuple[float | None, float | None]:
    """Return one finite, internally consistent cumulative buy observation."""
    if filled_qty is None and filled_avg_price is None:
        return None, None
    if isinstance(filled_qty, bool) or (filled_qty is not None and not isinstance(filled_qty, Number)):
        raise ValueError("Managed Alpaca buy fill economics must be finite and internally consistent.")
    if isinstance(filled_avg_price, bool) or (
        filled_avg_price is not None and not isinstance(filled_avg_price, Number)
    ):
        raise ValueError("Managed Alpaca buy fill economics must be finite and internally consistent.")
    try:
        normalized_qty = None if filled_qty is None else float(filled_qty)
        normalized_avg_price = None if filled_avg_price is None else float(filled_avg_price)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Managed Alpaca buy fill economics must be finite and internally consistent.") from exc
    if normalized_qty is None or not math.isfinite(normalized_qty) or normalized_qty < 0:
        raise ValueError("Managed Alpaca buy fill economics must be finite and internally consistent.")
    if normalized_qty == 0:
        if normalized_avg_price is not None:
            raise ValueError("A zero managed Alpaca buy fill cannot have an average fill price.")
        return normalized_qty, None
    if normalized_avg_price is None or not math.isfinite(normalized_avg_price) or normalized_avg_price <= 0:
        raise ValueError("A positive managed Alpaca buy fill requires a finite positive average price.")
    return normalized_qty, normalized_avg_price


def _managed_buy_limit_price_tolerance(limit_price: float) -> float:
    """Return half of Alpaca's applicable limit-price tick."""
    return 0.00005 if limit_price < 1.0 else 0.005


def _managed_buy_fill_causality_issue(
    *,
    buy_order_qty: float | None,
    buy_order_limit_price: float | None,
    buy_status: str,
    filled_qty: float,
    filled_avg_price: float | None,
) -> str | None:
    """Describe a fill that cannot arise from the immutable buy intent."""
    normalized_qty, normalized_limit_price = _normalize_optional_managed_buy_intent(
        buy_order_qty,
        buy_order_limit_price,
    )
    if normalized_qty is None or normalized_limit_price is None:
        return None
    quantity_scale = max(abs(normalized_qty), abs(filled_qty))
    quantity_residual = filled_qty - normalized_qty
    quantity_residual_is_negligible = _managed_accounting_residual_is_negligible(
        quantity_residual,
        quantity_scale=quantity_scale,
        mark_prices=(filled_avg_price, normalized_limit_price),
        value_scale=normalized_qty * normalized_limit_price,
    )
    quantity_exceeds_intent = bool(quantity_residual > 0.0 and not quantity_residual_is_negligible)
    filled_status_underfills_intent = bool(
        str(buy_status).strip().lower() == "filled" and quantity_residual < 0.0 and not quantity_residual_is_negligible
    )
    price_exceeds_limit = filled_avg_price is not None and (
        filled_avg_price > normalized_limit_price + _managed_buy_limit_price_tolerance(normalized_limit_price)
    )
    if not quantity_exceeds_intent and not filled_status_underfills_intent and not price_exceeds_limit:
        return None
    if quantity_exceeds_intent:
        return _MANAGED_BUY_FILL_QUANTITY_EXCEEDS_ISSUE
    if filled_status_underfills_intent:
        return _MANAGED_BUY_FILL_UNDERFILLED_ISSUE
    if price_exceeds_limit:
        return _MANAGED_BUY_FILL_PRICE_EXCEEDS_ISSUE
    return None


_MANAGED_BUY_FILL_QUANTITY_EXCEEDS_ISSUE = "Alpaca buy fill quantity exceeds the immutable managed buy intent"
_MANAGED_BUY_FILL_UNDERFILLED_ISSUE = "Alpaca buy filled status reports less than the immutable managed buy intent"
_MANAGED_BUY_FILL_PRICE_EXCEEDS_ISSUE = "Alpaca buy fill average price exceeds the immutable managed buy limit"

# Exact historical diagnostic used only to migrate the former note-based
# closed-shortfall control into its structured column. Keep this frozen even if
# the current human-readable wording changes.
_LEGACY_CLOSED_SELL_SHORTFALL_REOPEN_NOTE = (
    "authoritative Alpaca closed correction reopened shares with incomplete managed-sell coverage"
)


def _managed_buy_causality_quarantine_note(issue: str) -> str:
    return f"{issue}; protective accounting is retained but automated realized-P/L publication is quarantined"


def _managed_buy_causality_quarantine_assignment_sql(
    *,
    buy_order_qty_expression: str,
    buy_order_limit_price_expression: str,
    buy_status_expression: str,
    filled_qty_expression: str,
    filled_avg_price_expression: str,
    existing_marker_expression: str,
    buy_status_parameters: tuple[object, ...] = (),
) -> tuple[str, tuple[object, ...]]:
    """Return an atomic SQL assignment for an impossible-fill marker."""
    quantity_residual = f"(({filled_qty_expression}) - ({buy_order_qty_expression}))"
    quantity_scale = f"MAX(ABS({filled_qty_expression}), ABS({buy_order_qty_expression}))"
    mark_price = f"MAX(COALESCE({filled_avg_price_expression}, 0), COALESCE({buy_order_limit_price_expression}, 0))"
    value_scale = f"(({buy_order_qty_expression}) * ({buy_order_limit_price_expression}))"
    residual_is_negligible = _managed_quantity_is_negligible_sql(
        quantity_residual,
        quantity_scale_expression=quantity_scale,
        mark_price_expression=mark_price,
        value_scale_expression=value_scale,
    )
    return (
        f"""
        CASE
            WHEN {buy_order_qty_expression} IS NOT NULL
                 AND {buy_order_limit_price_expression} IS NOT NULL
                 AND {filled_qty_expression} IS NOT NULL
                 AND {quantity_residual} > 0
                 AND NOT ({residual_is_negligible})
            THEN ?
            WHEN LOWER(TRIM({buy_status_expression})) = 'filled'
                 AND {buy_order_qty_expression} IS NOT NULL
                 AND {buy_order_limit_price_expression} IS NOT NULL
                 AND {filled_qty_expression} IS NOT NULL
                 AND {quantity_residual} < 0
                 AND NOT ({residual_is_negligible})
            THEN ?
            WHEN {buy_order_limit_price_expression} IS NOT NULL
                 AND {filled_avg_price_expression} IS NOT NULL
                 AND {filled_avg_price_expression} > {buy_order_limit_price_expression}
                     + CASE WHEN {buy_order_limit_price_expression} < 1.0 THEN 0.00005 ELSE 0.005 END
            THEN ?
            ELSE {existing_marker_expression}
        END
        """,
        (
            _managed_buy_causality_quarantine_note(_MANAGED_BUY_FILL_QUANTITY_EXCEEDS_ISSUE),
            *buy_status_parameters,
            _managed_buy_causality_quarantine_note(_MANAGED_BUY_FILL_UNDERFILLED_ISSUE),
            _managed_buy_causality_quarantine_note(_MANAGED_BUY_FILL_PRICE_EXCEEDS_ISSUE),
        ),
    )


_MANAGED_BUY_CAUSALITY_QUARANTINE_STATUSES = frozenset(
    {
        "fill_quantity_regression",
        "identity_mismatch",
        "incomplete_fill_metadata",
        "pending_cancel",
    }
)


def _validate_managed_buy_causality_quarantine(
    *,
    buy_order_qty: float | None,
    buy_order_limit_price: float | None,
    buy_status: str,
    filled_qty: object | None,
    filled_avg_price: object | None,
    notes: object | None,
    quarantine_marker: object | None = None,
) -> None:
    """Require supported writes' sticky marker on impossible persisted fills."""
    normalized_filled_qty, normalized_filled_avg_price = _normalize_optional_managed_buy_fill_economics(
        filled_qty,
        filled_avg_price,
    )
    if normalized_filled_qty is None:
        # Alpaca may report a terminal lifecycle status before its fill
        # economics are available. The broker reconciler owns that transient
        # state and must still be able to load it.
        return
    issue = _managed_buy_fill_causality_issue(
        buy_order_qty=buy_order_qty,
        buy_order_limit_price=buy_order_limit_price,
        buy_status=buy_status,
        filled_qty=normalized_filled_qty,
        filled_avg_price=normalized_filled_avg_price,
    )
    if issue is None:
        return
    if str(buy_status).strip().lower() in _MANAGED_BUY_CAUSALITY_QUARANTINE_STATUSES:
        # These lifecycle states are themselves a durable automation fence.
        # They remain valid even if a later sell-side diagnostic replaced the
        # explanatory free-form notes on the same parent row.
        return
    required_note = _managed_buy_causality_quarantine_note(issue)
    if quarantine_marker == required_note:
        return
    if not isinstance(notes, str) or required_note not in notes:
        raise ValueError(
            "Managed Alpaca buy fill violates its immutable intent without the required quarantine marker."
        )


def _quarantine_managed_buy_causality_issue(
    *,
    buy_status: str,
    notes: str | None,
    issue: str | None,
) -> tuple[str, str | None, str | None]:
    """Keep real exposure accounted while making impossible economics fail closed."""
    if issue is None:
        return buy_status, notes, None
    quarantine_note = _managed_buy_causality_quarantine_note(issue)
    # Preserve the broker lifecycle status so protective sell reconciliation
    # can still close or repair real exposure. Broker callers that discover an
    # identity problem already pass a sticky diagnostic status; direct/import
    # callers retain both their prior context and the causality note, while
    # reporting independently rejects the impossible economics.
    return buy_status, notes, quarantine_note


def _merge_managed_buy_causality_notes(
    existing_notes: str | None,
    incoming_notes: str | None,
    causality_note: str | None,
) -> str | None:
    """Append unique diagnostic fragments while preserving persisted context."""
    if causality_note is None:
        return incoming_notes if incoming_notes is not None else existing_notes
    merged = existing_notes
    for fragment in (incoming_notes, causality_note):
        if fragment is None or fragment == "":
            continue
        if merged is None or merged == "":
            merged = fragment
        elif fragment not in merged:
            merged = f"{merged}; {fragment}"
    return merged


def _managed_buy_notes_assignment(
    incoming_notes: str | None,
    causality_note: str | None,
) -> tuple[str, tuple[object, ...]]:
    """Return an atomic SQL assignment that preserves causality diagnostics."""
    if causality_note is None:
        return "COALESCE(?, notes)", (incoming_notes,)
    if incoming_notes is None or incoming_notes == "":
        return (
            """
            CASE
                WHEN notes IS NULL OR notes = '' THEN ?
                WHEN INSTR(notes, ?) > 0 THEN notes
                ELSE notes || '; ' || ?
            END
            """,
            (causality_note, causality_note, causality_note),
        )
    if causality_note in incoming_notes:
        return (
            """
            CASE
                WHEN notes IS NULL OR notes = '' THEN ?
                WHEN INSTR(notes, ?) > 0 THEN notes
                ELSE notes || '; ' || ?
            END
            """,
            (incoming_notes, incoming_notes, incoming_notes),
        )
    return (
        """
        CASE
            WHEN notes IS NULL OR notes = '' THEN ? || '; ' || ?
            WHEN INSTR(notes, ?) > 0 AND INSTR(notes, ?) > 0 THEN notes
            WHEN INSTR(notes, ?) > 0 THEN notes || '; ' || ?
            WHEN INSTR(notes, ?) > 0 THEN notes || '; ' || ?
            ELSE notes || '; ' || ? || '; ' || ?
        END
        """,
        (
            incoming_notes,
            causality_note,
            incoming_notes,
            causality_note,
            incoming_notes,
            causality_note,
            causality_note,
            incoming_notes,
            incoming_notes,
            causality_note,
        ),
    )


SQLITE_BUSY_TIMEOUT_MS = 60_000
_UNSET = object()
_ALPACA_MANAGED_SELL_DIAGNOSTIC_STATUSES = frozenset(
    {
        "fill_quantity_regression",
        "fractional_qty",
        "incomplete_fill_metadata",
        "incomplete_order_metadata",
        "late_fill_after_close",
        "position_quantity_mismatch",
        "quantity_mismatch",
    }
)
_ALPACA_MANAGED_SELL_STICKY_STATUSES = frozenset(
    {
        "canceled",
        "expired",
        "filled",
        "pending_cancel",
        "rejected",
        "replaced",
        "submission_failed",
        *_ALPACA_MANAGED_SELL_DIAGNOSTIC_STATUSES,
    }
)
_ALPACA_MANAGED_SELL_CLOSEABLE_STATUSES = frozenset({"canceled", "expired", "filled", "rejected"})
_ALPACA_ORDER_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,127}")
_ALPACA_ASSET_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,127}")
_MANAGED_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9./_-]{0,31}")
_NORMALIZABLE_MANAGED_SYMBOL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9./_-]{0,31}")
_ALPACA_BUY_COMPONENT_JSON_MAX_NESTING_DEPTH = 128


def alpaca_order_id_is_canonical(value: object) -> bool:
    """Return whether *value* is an unambiguous broker-order identity."""
    return type(value) is str and _ALPACA_ORDER_ID_PATTERN.fullmatch(value) is not None


def _canonical_alpaca_order_id(value: object, *, field_name: str) -> str:
    if not alpaca_order_id_is_canonical(value):
        raise ValueError(f"{field_name} must be a canonical broker order ID of 1-128 characters")
    return value


def _canonical_alpaca_asset_id(value: object, *, field_name: str) -> str:
    if type(value) is not str or _ALPACA_ASSET_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical Alpaca asset ID")
    return value


def _canonical_optional_alpaca_asset_id(
    value: object | None,
    *,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    return _canonical_alpaca_asset_id(value, field_name=field_name)


def _canonical_managed_symbol(value: object, *, field_name: str) -> str:
    if type(value) is not str or _MANAGED_SYMBOL_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical uppercase symbol")
    return value


def _normalized_managed_symbol(value: object, *, field_name: str) -> str:
    """Normalize safe ASCII public input without Unicode identity folding."""
    if type(value) is not str:
        return _canonical_managed_symbol(value, field_name=field_name)
    normalized = value.strip(" ")
    if _NORMALIZABLE_MANAGED_SYMBOL_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a canonical ASCII symbol")
    return _canonical_managed_symbol(normalized.upper(), field_name=field_name)


def _normalized_optional_alpaca_asset_id(
    value: object | None,
    *,
    field_name: str,
) -> str | None:
    """Trim ordinary ASCII padding without accepting Unicode whitespace."""
    if value is None:
        return None
    normalized = value.strip(" ") if type(value) is str else value
    return _canonical_alpaca_asset_id(normalized, field_name=field_name)


def _managed_lifecycle_status(value: object, *, field_name: str) -> str:
    """Require a status value that SQLite will persist with TEXT storage."""
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a string lifecycle status.")
    return value


def _optional_managed_lifecycle_status(value: object | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _managed_lifecycle_status(value, field_name=field_name)


def _managed_quantity_tolerance_sql(scale_expression: str) -> str:
    """Render the shared managed-quantity policy for an internal SQL expression."""
    return (
        f"MIN({MANAGED_QUANTITY_MAX_TOLERANCE!r}, "
        f"{MANAGED_QUANTITY_ABSOLUTE_TOLERANCE!r} + "
        f"({MANAGED_QUANTITY_RELATIVE_TOLERANCE!r} * ABS({scale_expression})))"
    )


def _managed_quantity_matches_sql(left_expression: str, right_expression: str) -> str:
    scale = f"MAX(ABS({left_expression}), ABS({right_expression}))"
    return f"ABS(({left_expression}) - ({right_expression})) <= {_managed_quantity_tolerance_sql(scale)}"


def _managed_residual_notional_tolerance_sql(scale_expression: str) -> str:
    return (
        f"MIN({MANAGED_VALUE_MAX_RECONCILIATION_TOLERANCE!r}, "
        f"MAX({MANAGED_NOTIONAL_MIN_ULP_ALLOWANCE!r}, "
        f"{MANAGED_NOTIONAL_RELATIVE_TOLERANCE!r} * ABS({scale_expression})))"
    )


def _managed_quantity_is_negligible_sql(
    expression: str,
    *,
    quantity_scale_expression: str,
    mark_price_expression: str,
    value_scale_expression: str,
) -> str:
    quantity = f"ABS({expression})"
    residual_notional = f"({quantity} * ({mark_price_expression}))"
    value_scale = f"MAX(ABS({value_scale_expression}), {residual_notional})"
    return f"""
        COALESCE(
            {quantity} <= {_managed_quantity_tolerance_sql(quantity_scale_expression)}
            AND (
                {quantity} = 0
                OR (
                    ({mark_price_expression}) > 0
                    AND ({mark_price_expression}) - ({mark_price_expression}) IS NOT NULL
                    AND {residual_notional} - {residual_notional} IS NOT NULL
                    AND ({value_scale}) - ({value_scale}) IS NOT NULL
                    AND {residual_notional}
                        <= {_managed_residual_notional_tolerance_sql(value_scale)}
                )
            ),
            0
        )
    """


def _managed_accounting_quantities_match_sql(
    left_expression: str,
    right_expression: str,
    *,
    mark_price_expression: str,
    value_scale_expression: str,
) -> str:
    """Render the shared share-and-notional equality policy for SQL CAS paths."""
    quantity_scale = f"MAX(ABS({left_expression}), ABS({right_expression}))"
    return _managed_quantity_is_negligible_sql(
        f"(({left_expression}) - ({right_expression}))",
        quantity_scale_expression=quantity_scale,
        mark_price_expression=mark_price_expression,
        value_scale_expression=value_scale_expression,
    )


def _managed_position_mark_price_sql() -> str:
    sold_unit_price = """
        CASE
            WHEN COALESCE(sold_qty, 0) > 0 AND COALESCE(sold_value, 0) > 0
            THEN sold_value / sold_qty
            ELSE 0
        END
    """
    return (
        "MAX(COALESCE(target_sell_price, 0), COALESCE(filled_avg_price, 0), "
        "COALESCE(buy_order_limit_price, 0), COALESCE(sell_order_limit_price, 0), "
        f"({sold_unit_price}))"
    )


def _managed_position_quantity_is_negligible_sql(expression: str) -> str:
    mark_price = _managed_position_mark_price_sql()
    quantity_scale = f"MAX(ABS({expression}), ABS(COALESCE(filled_qty, 0)), ABS(COALESCE(sold_qty, 0)))"
    value_scale = (
        f"MAX(ABS(COALESCE(filled_qty, 0) * ({mark_price})), "
        f"ABS(COALESCE(sold_value, 0)), ABS(({expression}) * ({mark_price})))"
    )
    return _managed_quantity_is_negligible_sql(
        expression,
        quantity_scale_expression=quantity_scale,
        mark_price_expression=mark_price,
        value_scale_expression=value_scale,
    )


def _managed_position_quantity_is_positive_sql(expression: str) -> str:
    return f"({expression}) > 0 AND NOT ({_managed_position_quantity_is_negligible_sql(expression)})"


_MANAGED_REMAINING_QUANTITY_SQL = "COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0))"


def _managed_accounting_residual_is_negligible(
    residual_quantity: float,
    *,
    quantity_scale: float,
    mark_prices: tuple[float | None, ...],
    value_scale: float = 0.0,
) -> bool:
    """Apply the shared share-and-notional residual policy in Python paths."""
    normalized_prices: list[float] = []
    for price in mark_prices:
        if price is None:
            continue
        numeric_price = float(price)
        if not math.isfinite(numeric_price):
            return float(residual_quantity) == 0.0
        if numeric_price > 0.0:
            normalized_prices.append(numeric_price)
    mark_price = max(normalized_prices, default=0.0)
    marked_quantity_scale = abs(float(quantity_scale)) * mark_price
    return managed_residual_quantity_is_negligible(
        residual_quantity,
        quantity_scale=quantity_scale,
        mark_price=mark_price,
        value_scale=max(abs(float(value_scale)), marked_quantity_scale),
    )


def _managed_accounting_quantity_is_positive(
    quantity: float,
    *,
    quantity_scale: float,
    mark_prices: tuple[float | None, ...],
    value_scale: float = 0.0,
) -> bool:
    return bool(
        quantity > 0.0
        and not _managed_accounting_residual_is_negligible(
            quantity,
            quantity_scale=quantity_scale,
            mark_prices=mark_prices,
            value_scale=value_scale,
        )
    )


def _managed_accounting_quantities_match(
    left: float,
    right: float,
    *,
    mark_prices: tuple[float | None, ...],
    value_scale: float = 0.0,
) -> bool:
    return _managed_accounting_residual_is_negligible(
        float(left) - float(right),
        quantity_scale=max(abs(float(left)), abs(float(right))),
        mark_prices=mark_prices,
        value_scale=value_scale,
    )


@dataclass
class SummaryRollup:
    first_equity: float | None = None
    last_equity: float | None = None
    running_max_equity: float | None = None
    return_count: int = 0
    return_sum: float = 0.0
    return_sum_squares: float = 0.0
    return_mean: float = 0.0
    return_m2: float = 0.0
    excess_return_count: int = 0
    excess_return_sum: float = 0.0
    excess_return_sum_squares: float = 0.0
    excess_return_mean: float = 0.0
    excess_return_m2: float = 0.0
    positive_return_count: int = 0
    max_drawdown: float | None = None

    @property
    def trading_days(self) -> int:
        return self.return_count + 1 if self.first_equity is not None else 0


class _DeferredCommitSqliteConnection(sqlite3.Connection):
    """Let an enclosing runtime guard retain ownership of the transaction."""

    def commit(self) -> None:
        """Defer helper-controlled commits until the runtime guards pass."""


def save_table_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    expected_guard = revalidate_active_sqlite_runtime_file(db_path)
    runtime_guard = (
        prepare_private_runtime_file(db_path, expected_guard=expected_guard)
        if expected_guard is not None
        else prepare_private_runtime_file(db_path)
    )
    revalidate_active_sqlite_runtime_file(db_path, runtime_guard)
    runtime_db_path = runtime_guard.database_path if runtime_guard is not None else db_path
    conn = sqlite3.connect(
        runtime_db_path,
        timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
        factory=_DeferredCommitSqliteConnection,
    )
    failure: BaseException | None = None
    try:
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        runtime_guard = revalidate_private_runtime_file(runtime_guard)
        revalidate_active_sqlite_runtime_file(db_path, runtime_guard)
        conn.execute("BEGIN IMMEDIATE")
        # pandas' sqlite3 fallback normally commits DROP/CREATE and INSERT
        # separately. This connection defers those commits so runtime-file
        # validation can still roll the complete replacement back.
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        runtime_guard = revalidate_private_runtime_file(runtime_guard)
        revalidate_active_sqlite_runtime_file(db_path, runtime_guard)
        try:
            sqlite3.Connection.commit(conn)
        except BaseException as exc:
            _rollback_after_commit_failure(conn, exc)
            raise
        runtime_guard = revalidate_private_runtime_file(runtime_guard)
        revalidate_active_sqlite_runtime_file(db_path, runtime_guard)
    except BaseException as exc:
        failure = exc
        _rollback_owned_operation_if_active(
            conn,
            exc,
            operation="SQLite table replacement",
        )
        raise
    finally:
        try:
            conn.close()
        except BaseException as close_failure:
            if failure is None:
                raise
            failure.add_note(f"Failed to close the connection after SQLite table replacement: {close_failure}")


def save_workflow_assets(conn: sqlite3.Connection, workflow_assets: pd.DataFrame) -> None:
    workflow_assets.to_sql("workflow_assets", conn, if_exists="replace", index=False)


_STATE_DB_TABLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS strategy_state (
        asset_symbol TEXT NOT NULL,
        signal_symbol TEXT NOT NULL,
        buy_rsi REAL NOT NULL,
        profit_target_multiple REAL NOT NULL,
        start_date TEXT,
        last_date TEXT NOT NULL,
        cash REAL NOT NULL,
        shares REAL NOT NULL,
        in_position INTEGER NOT NULL,
        entry_price REAL,
        pending_action TEXT NOT NULL,
        prev_equity REAL NOT NULL,
        trades_executed INTEGER NOT NULL,
        integrity_digest TEXT,
        entry_date TEXT,
        PRIMARY KEY (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)
    );

    CREATE TABLE IF NOT EXISTS strategy_equity (
        asset_symbol TEXT NOT NULL,
        signal_symbol TEXT NOT NULL,
        buy_rsi REAL NOT NULL,
        profit_target_multiple REAL NOT NULL,
        date TEXT NOT NULL,
        equity REAL NOT NULL,
        daily_return REAL NOT NULL,
        risk_free_return REAL,
        in_position INTEGER NOT NULL,
        action_executed TEXT NOT NULL,
        pending_action TEXT NOT NULL,
        trades_executed INTEGER NOT NULL,
        integrity_digest TEXT,
        PRIMARY KEY (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, date)
    );

    CREATE TABLE IF NOT EXISTS strategy_summary (
        asset_symbol TEXT NOT NULL,
        signal_symbol TEXT NOT NULL,
        buy_rsi REAL NOT NULL,
        profit_target_multiple REAL NOT NULL,
        start_date TEXT,
        end_date TEXT,
        trading_days INTEGER NOT NULL,
        trades_executed INTEGER NOT NULL,
        total_return REAL,
        cagr REAL,
        annualized_vol REAL,
        sharpe REAL,
        kelly_fraction REAL,
        max_drawdown REAL,
        hit_rate REAL,
        first_equity REAL,
        last_equity REAL,
        running_max_equity REAL,
        return_count INTEGER,
        return_sum REAL,
        return_sum_squares REAL,
        excess_return_count INTEGER,
        excess_return_sum REAL,
        excess_return_sum_squares REAL,
        positive_return_count INTEGER,
        return_mean REAL,
        return_m2 REAL,
        excess_return_mean REAL,
        excess_return_m2 REAL,
        integrity_digest TEXT,
        PRIMARY KEY (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)
    );

    CREATE TABLE IF NOT EXISTS rsi_values (
        signal_symbol TEXT NOT NULL,
        rsi_period INTEGER NOT NULL,
        date TEXT NOT NULL,
        close REAL NOT NULL,
        avg_gain REAL,
        avg_loss REAL,
        rsi REAL,
        PRIMARY KEY (signal_symbol, rsi_period, date)
    );

    CREATE TABLE IF NOT EXISTS market_data (
        symbol TEXT NOT NULL,
        date TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        PRIMARY KEY (symbol, date)
    );

    CREATE TABLE IF NOT EXISTS market_history_removal_candidates (
        symbol TEXT PRIMARY KEY,
        missing_dates_fingerprint TEXT NOT NULL,
        missing_date_count INTEGER NOT NULL,
        consecutive_observations INTEGER NOT NULL,
        last_observed_run_id TEXT,
        first_observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS strategy_config (
        asset_symbol TEXT NOT NULL,
        signal_symbol TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        PRIMARY KEY (asset_symbol, signal_symbol)
    );

    CREATE TABLE IF NOT EXISTS strategy_state_generation (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        generation INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS alpaca_managed_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT CHECK (id > 0),
        state_revision INTEGER NOT NULL DEFAULT 0,
        workflow TEXT,
        symbol TEXT NOT NULL,
        alpaca_asset_id TEXT,
        signal_symbol TEXT NOT NULL,
        buy_rsi REAL NOT NULL,
        profit_target_multiple REAL NOT NULL,
        buy_signal_date TEXT NOT NULL,
        buy_client_order_id TEXT NOT NULL UNIQUE,
        buy_alpaca_order_id TEXT,
        buy_submitted_at TEXT,
        buy_order_qty REAL,
        buy_order_limit_price REAL,
        buy_submission_claimed_at TEXT,
        buy_submission_attempt_count INTEGER NOT NULL DEFAULT 1,
        buy_status TEXT NOT NULL,
        filled_qty REAL,
        filled_avg_price REAL,
        filled_at TEXT,
        buy_observation_broker_updated_at TEXT,
        buy_fill_broker_updated_at TEXT,
        buy_fill_component_revisions TEXT,
        buy_fill_pending_observation TEXT,
        buy_causality_quarantine TEXT,
        buy_cancellation_alpaca_order_ids TEXT,
        target_sell_price REAL,
        sell_order_namespace TEXT,
        sell_client_order_id TEXT UNIQUE,
        sell_alpaca_order_id TEXT,
        sell_submitted_at TEXT,
        sell_status TEXT,
        sell_expires_at TEXT,
        sell_order_qty REAL,
        sell_order_limit_price REAL,
        sell_observation_broker_updated_at TEXT,
        sell_observation_filled_qty REAL,
        sell_submission_retry_claimed_at TEXT,
        sell_renewal_count INTEGER NOT NULL DEFAULT 0,
        sell_renewal_requested_at TEXT,
        sell_filled_qty REAL,
        sell_filled_avg_price REAL,
        sell_filled_at TEXT,
        realized_pl REAL,
        realized_pl_pct REAL,
        sold_qty REAL NOT NULL DEFAULT 0,
        sold_value REAL NOT NULL DEFAULT 0,
        remaining_qty REAL,
        closed_at TEXT,
        closed_correction_audited_at TEXT,
        closed_sell_shortfall_reopen_pending INTEGER NOT NULL DEFAULT 0,
        notes TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS alpaca_managed_sell_fills (
        managed_position_id INTEGER NOT NULL,
        alpaca_order_id TEXT NOT NULL,
        filled_qty REAL NOT NULL,
        filled_value REAL NOT NULL,
        broker_updated_at TEXT,
        submitted_qty REAL,
        submitted_limit_price REAL,
        PRIMARY KEY (managed_position_id, alpaca_order_id)
    );

    CREATE TABLE IF NOT EXISTS alpaca_symbol_aliases (
        alpaca_asset_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        first_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (alpaca_asset_id, symbol)
    );
"""


def init_state_db(conn: sqlite3.Connection, *, commit: bool = True) -> None:
    if type(commit) is not bool:
        raise ValueError("commit must be a boolean")
    # ``CREATE TABLE IF NOT EXISTS`` cannot repair imported tables.  Audit any
    # identities that already exist before the schema script can create or
    # alter unrelated objects, so an ambiguous database fails without partial
    # migration side effects.
    _preflight_existing_storage_identities(conn)
    _preflight_existing_storage_table_contracts(conn)
    _preflight_existing_storage_schema_objects(conn)
    savepoint_name = "init_state_db_atomic"
    owns_transaction = not conn.in_transaction
    savepoint_active = False
    try:
        if owns_transaction:
            # Releasing an outermost savepoint commits it immediately. Keep an
            # explicit transaction underneath so commit remains the sole
            # durability boundary and a failed commit can still be rolled back.
            conn.execute("BEGIN")
        conn.execute(f"SAVEPOINT {savepoint_name}")
        savepoint_active = True
        _migrate_state_db(conn)
        conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        savepoint_active = False
        if commit:
            # Initializing the schema is normally an explicit durability
            # boundary. Keep its established default contract of committing an
            # existing caller transaction as well as the transaction opened
            # above. Runtime-guarded workflow initialization defers this one
            # commit until its database-identity checks have passed.
            _commit_owned_transaction(conn)
    except BaseException as exc:
        if savepoint_active:
            _rollback_and_release_savepoint(conn, savepoint_name, exc)
        if owns_transaction:
            _rollback_owned_operation_if_active(
                conn,
                exc,
                operation="state database initialization",
            )
        raise


def _execute_schema_statements(conn: sqlite3.Connection, script: str) -> None:
    """Execute the table-only schema script without executescript's implicit commit."""
    for statement in script.split(";"):
        if statement.strip():
            conn.execute(statement)


_EXPLICIT_ADDITIVE_LEGACY_COLUMNS = {
    "strategy_state": frozenset(STRATEGY_STATE_COLUMNS),
    "strategy_equity": frozenset(STRATEGY_EQUITY_COLUMNS),
    "strategy_summary": frozenset(STRATEGY_SUMMARY_COLUMNS),
    "alpaca_managed_positions": frozenset(ALPACA_MANAGED_POSITION_COLUMNS),
    "alpaca_managed_sell_fills": frozenset(ALPACA_MANAGED_SELL_FILL_COLUMNS),
    "market_history_removal_candidates": frozenset({"last_observed_run_id"}),
}

_STORAGE_IDENTITY_TYPE_CONTRACTS: dict[str, tuple[tuple[str, str, bool], ...]] = {
    "strategy_config": (
        ("asset_symbol", "text", False),
        ("signal_symbol", "text", False),
    ),
    "strategy_state": (
        ("asset_symbol", "text", False),
        ("signal_symbol", "text", False),
        ("buy_rsi", "real", False),
        ("profit_target_multiple", "real", False),
    ),
    "strategy_summary": (
        ("asset_symbol", "text", False),
        ("signal_symbol", "text", False),
        ("buy_rsi", "real", False),
        ("profit_target_multiple", "real", False),
    ),
    "strategy_equity": (
        ("asset_symbol", "text", False),
        ("signal_symbol", "text", False),
        ("buy_rsi", "real", False),
        ("profit_target_multiple", "real", False),
        ("date", "text", False),
    ),
    "market_data": (("symbol", "text", False), ("date", "text", False)),
    "rsi_values": (
        ("signal_symbol", "text", False),
        ("rsi_period", "integer", False),
        ("date", "text", False),
    ),
    "market_history_removal_candidates": (("symbol", "text", False),),
    "alpaca_managed_positions": (
        ("id", "integer", False),
        ("state_revision", "integer", False),
        ("symbol", "text", False),
        ("alpaca_asset_id", "text", True),
        ("signal_symbol", "text", False),
        ("buy_rsi", "real", False),
        ("profit_target_multiple", "real", False),
        ("buy_signal_date", "text", False),
        ("buy_client_order_id", "text", False),
        ("sell_client_order_id", "text", True),
    ),
    "alpaca_managed_sell_fills": (
        ("managed_position_id", "integer", False),
        ("alpaca_order_id", "text", False),
    ),
    "alpaca_symbol_aliases": (
        ("alpaca_asset_id", "text", False),
        ("symbol", "text", False),
    ),
}

_OWNED_STORAGE_TRIGGER_TABLES = {
    _ALPACA_STATE_REVISION_TRIGGER_NAME: "alpaca_managed_positions",
    _ALPACA_STATE_REVISION_GUARD_TRIGGER_NAME: "alpaca_managed_positions",
    "strategy_state_generation_validate_insert": "strategy_state_generation",
    "strategy_state_generation_validate_update": "strategy_state_generation",
    "strategy_state_generation_validate_delete": "strategy_state_generation",
    "alpaca_managed_sell_fills_validate_parent_insert": "alpaca_managed_sell_fills",
    "alpaca_managed_sell_fills_validate_parent_update": "alpaca_managed_sell_fills",
    "alpaca_managed_positions_restrict_sell_fill_parent_delete": "alpaca_managed_positions",
    "alpaca_managed_positions_restrict_sell_fill_parent_id_update": "alpaca_managed_positions",
    **{
        f"leveraged_trader_{table_name}_identity_{operation}_guard": table_name
        for table_name in _STORAGE_IDENTITY_TYPE_CONTRACTS
        for operation in ("insert", "update")
    },
}


def _canonical_storage_table_contracts() -> dict[str, tuple[str, tuple[tuple, ...]]]:
    """Build authoritative table SQL and PRAGMA contracts from the owned schema."""
    with closing(sqlite3.connect(":memory:")) as reference:
        _execute_schema_statements(reference, _STATE_DB_TABLE_SCHEMA)
        contracts: dict[str, tuple[str, tuple[tuple, ...]]] = {}
        for table_name, create_sql in reference.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall():
            contracts[str(table_name)] = (
                str(create_sql),
                tuple(reference.execute(f"PRAGMA table_info({table_name})").fetchall()),
            )
    return contracts


def _preflight_existing_storage_table_contracts(conn: sqlite3.Connection) -> None:
    """Reject partial or incompatible imported tables before schema mutation."""
    for table_name, (_create_sql, canonical_rows) in _canonical_storage_table_contracts().items():
        existing_object = conn.execute(
            "SELECT type, name FROM sqlite_master WHERE name = ? COLLATE NOCASE",
            (table_name,),
        ).fetchone()
        if existing_object is None:
            continue
        if str(existing_object[1]) != table_name:
            raise ValueError(
                f"Managed table {existing_object[1]} must use canonical casing "
                f"{table_name}; repair the imported state database."
            )
        if existing_object[0] != "table":
            raise ValueError(
                f"Schema object {table_name} must be a managed table; rebuild or repair the imported state database."
            )
        actual_rows = tuple(conn.execute(f"PRAGMA table_info({table_name})").fetchall())
        actual_by_name = {str(row[1]): row for row in actual_rows}
        canonical_by_name = {str(row[1]): row for row in canonical_rows}
        additive_columns = _EXPLICIT_ADDITIVE_LEGACY_COLUMNS.get(table_name, frozenset())
        missing_required = sorted(set(canonical_by_name).difference(actual_by_name).difference(additive_columns))
        if missing_required:
            missing = ", ".join(missing_required)
            raise ValueError(
                f"{table_name} is missing required managed columns ({missing}); rebuild or "
                "repair the imported state database."
            )
        unexpected_columns = sorted(set(actual_by_name).difference(canonical_by_name))
        if unexpected_columns:
            unexpected = ", ".join(unexpected_columns)
            raise ValueError(
                f"{table_name} contains unsupported managed columns ({unexpected}); rebuild "
                "or repair the imported state database."
            )
        for column_name, actual_row in actual_by_name.items():
            expected_affinity = _sqlite_declared_type_affinity(canonical_by_name[column_name][2])
            actual_affinity = _sqlite_declared_type_affinity(actual_row[2])
            if actual_affinity != expected_affinity:
                raise ValueError(
                    f"{table_name}.{column_name} has incompatible declared affinity "
                    f"{actual_affinity!r}; expected {expected_affinity!r}. Rebuild or repair "
                    "the imported state database."
                )


def _storage_column_contract_signature(row: tuple) -> tuple[str, str, int, str | None, int]:
    default = None if row[4] is None else _normalized_schema_sql(str(row[4]))
    return (
        str(row[1]),
        str(row[2]).strip().upper(),
        int(row[3]),
        default,
        int(row[5]),
    )


def _storage_table_has_canonical_create_sql(
    actual_create_sql: str | None,
    canonical_create_sql: str,
) -> bool:
    """Prove the exact owned table DDL, including every managed constraint."""
    return _normalized_schema_sql(actual_create_sql) == _normalized_schema_sql(canonical_create_sql)


def _managed_position_sequence_high_water(
    conn: sqlite3.Connection,
) -> int | None:
    """Return a trustworthy managed-position AUTOINCREMENT high-water mark."""
    if (
        conn.execute(
            "SELECT 1 FROM alpaca_managed_positions WHERE TYPEOF(id) != 'integer' OR id <= 0 LIMIT 1"
        ).fetchone()
        is not None
    ):
        raise ValueError(
            "alpaca_managed_positions has a non-positive or invalid managed identity; "
            "rebuild or repair the imported state database."
        )
    max_id = conn.execute("SELECT MAX(id), TYPEOF(MAX(id)) FROM alpaca_managed_positions").fetchone()
    if max_id is None or max_id[0] is None:
        live_high_water = 0
    elif max_id[1] != "integer" or isinstance(max_id[0], bool):
        raise ValueError(
            "alpaca_managed_positions has an invalid managed identity high-water mark; "
            "rebuild or repair the imported state database."
        )
    else:
        live_high_water = int(max_id[0])
        if live_high_water <= 0:
            raise ValueError(
                "alpaca_managed_positions has a non-positive managed identity; rebuild or "
                "repair the imported state database."
            )
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'").fetchone() is None:
        return live_high_water or None
    sequence_rows = conn.execute(
        "SELECT seq, TYPEOF(seq) FROM sqlite_sequence WHERE name = ?",
        ("alpaca_managed_positions",),
    ).fetchall()
    if len(sequence_rows) > 1:
        raise ValueError(
            "alpaca_managed_positions has ambiguous AUTOINCREMENT sequence state; "
            "rebuild or repair the imported state database."
        )
    if not sequence_rows:
        return live_high_water or None
    sequence_value, storage_type = sequence_rows[0]
    if (
        storage_type != "integer"
        or isinstance(sequence_value, bool)
        or int(sequence_value) < live_high_water
        or int(sequence_value) < 0
    ):
        raise ValueError(
            "alpaca_managed_positions has an invalid AUTOINCREMENT sequence high-water "
            "mark; rebuild or repair the imported state database."
        )
    return int(sequence_value)


def _restore_managed_position_sequence_high_water(
    conn: sqlite3.Connection,
    high_water: int | None,
) -> None:
    """Restore history that copying only live AUTOINCREMENT rows cannot retain."""
    conn.execute(
        "DELETE FROM sqlite_sequence WHERE name = ?",
        ("alpaca_managed_positions",),
    )
    if high_water is not None:
        conn.execute(
            "INSERT INTO sqlite_sequence (name, seq) VALUES (?, ?)",
            ("alpaca_managed_positions", high_water),
        )


def _repair_storage_table_contracts(conn: sqlite3.Connection) -> None:
    """Canonically rebuild complete imports whose SQLite constraints were stripped."""
    contracts = _canonical_storage_table_contracts()
    repair_targets: set[str] = set()
    for table_name in contracts:
        create_sql, canonical_rows = contracts[table_name]
        actual_rows = tuple(conn.execute(f"PRAGMA table_info({table_name})").fetchall())
        actual_by_name = {str(row[1]): row for row in actual_rows}
        canonical_by_name = {str(row[1]): row for row in canonical_rows}
        if set(actual_by_name) != set(canonical_by_name):
            raise ValueError(
                f"{table_name} does not match the complete managed column contract; rebuild "
                "or repair the imported state database."
            )
        actual_create_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        requires_sql_repair = not _storage_table_has_canonical_create_sql(
            None if actual_create_row is None else actual_create_row[0],
            create_sql,
        )
        if not requires_sql_repair and all(
            _storage_column_contract_signature(actual_by_name[column_name])
            == _storage_column_contract_signature(expected)
            for column_name, expected in canonical_by_name.items()
        ):
            continue
        repair_targets.add(table_name)

    # Imported managed tables may carry removable foreign keys that are absent
    # from the canonical schema. Rebuild every referencing managed child before
    # its parent so a parent DROP cannot cascade-delete child rows. An unmanaged
    # child cannot be rewritten under this schema contract, so refuse the parent
    # repair atomically instead of mutating external data.
    dependencies: dict[str, set[str]] = {table_name: set() for table_name in repair_targets}
    table_names = [
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
        if not str(row[0]).startswith("sqlite_")
    ]
    for child_table in table_names:
        escaped_child = child_table.replace("'", "''")
        for foreign_key in conn.execute(f"PRAGMA foreign_key_list('{escaped_child}')").fetchall():
            referenced_parent = str(foreign_key[2]).lower()
            parent = next(
                (target for target in repair_targets if target.lower() == referenced_parent),
                None,
            )
            if parent is None:
                continue
            child = next(
                (target for target in repair_targets if target.lower() == child_table.lower()),
                None,
            )
            if child is None:
                raise ValueError(
                    f"Cannot canonically rebuild managed table {parent} while external "
                    f"table {child_table} references it; remove or migrate that foreign "
                    "key before initializing the state database."
                )
            dependencies[parent].add(child)

    repair_order: list[str] = []
    remaining = {table_name: set(children) for table_name, children in dependencies.items()}
    while remaining:
        ready = sorted(table_name for table_name, children in remaining.items() if not children)
        if not ready:
            cycle_tables = ", ".join(sorted(remaining))
            raise ValueError(
                "Managed table foreign keys form a rebuild cycle "
                f"({cycle_tables}); remove the imported foreign keys before initializing "
                "the state database."
            )
        repair_order.extend(ready)
        for table_name in ready:
            remaining.pop(table_name)
        for children in remaining.values():
            children.difference_update(ready)

    for table_name in repair_order:
        create_sql, canonical_rows = contracts[table_name]

        rebuilt_name = f"leveraged_trader_rebuilt_{table_name}"
        if (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = ?",
                (rebuilt_name,),
            ).fetchone()
            is not None
        ):
            raise ValueError(
                f"Schema object {rebuilt_name} conflicts with repair of {table_name}; rebuild "
                "or repair the imported state database."
            )
        quoted_columns = ", ".join(f'"{row[1]}"' for row in canonical_rows)
        managed_position_high_water = (
            _managed_position_sequence_high_water(conn) if table_name == "alpaca_managed_positions" else None
        )
        try:
            rebuilt_create_sql = re.sub(
                rf"(?i)(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?){re.escape(table_name)}",
                rf'\1"{rebuilt_name}"',
                create_sql,
                count=1,
            )
            conn.execute(rebuilt_create_sql)
            conn.execute(f'INSERT INTO "{rebuilt_name}" ({quoted_columns}) SELECT {quoted_columns} FROM "{table_name}"')
            conn.execute(f'DROP TABLE "{table_name}"')
            conn.execute(f'ALTER TABLE "{rebuilt_name}" RENAME TO "{table_name}"')
            if table_name == "alpaca_managed_positions":
                _restore_managed_position_sequence_high_water(
                    conn,
                    managed_position_high_water,
                )
        except sqlite3.Error as exc:
            raise ValueError(
                f"{table_name} cannot be rebuilt with its required nullability, defaults, "
                "primary key, and checks; repair the imported state database."
            ) from exc


def _validate_storage_table_contracts(conn: sqlite3.Connection) -> None:
    """Verify the final managed schema after every additive migration and repair."""
    for table_name, (_create_sql, canonical_rows) in _canonical_storage_table_contracts().items():
        actual_rows = tuple(conn.execute(f"PRAGMA table_info({table_name})").fetchall())
        actual_by_name = {str(row[1]): row for row in actual_rows}
        canonical_by_name = {str(row[1]): row for row in canonical_rows}
        if set(actual_by_name) != set(canonical_by_name) or any(
            _storage_column_contract_signature(actual_by_name[column_name])
            != _storage_column_contract_signature(expected)
            for column_name, expected in canonical_by_name.items()
        ):
            raise ValueError(
                f"{table_name} does not satisfy its managed schema contract; rebuild or "
                "repair the imported state database."
            )
        actual_create_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if not _storage_table_has_canonical_create_sql(
            None if actual_create_row is None else actual_create_row[0],
            _create_sql,
        ):
            raise ValueError(
                f"{table_name} does not satisfy its canonical managed CREATE TABLE "
                "contract; rebuild or repair the imported state database."
            )


def _storage_identity_type_guard_sql(
    table_name: str,
    columns: tuple[tuple[str, str, bool], ...],
    operation: str,
) -> str:
    """Return the exact owned trigger for the identity columns currently present."""
    type_predicates = []
    for column_name, expected_type, nullable in columns:
        allowed_types = "('integer', 'real')" if expected_type == "real" else f"('{expected_type}')"
        if nullable:
            type_predicates.append(
                f"NEW.{column_name} IS NOT NULL AND TYPEOF(NEW.{column_name}) NOT IN {allowed_types}"
            )
        else:
            type_predicates.append(f"NEW.{column_name} IS NULL OR TYPEOF(NEW.{column_name}) NOT IN {allowed_types}")
    if table_name == "alpaca_managed_positions" and any(
        column_name == "state_revision" for column_name, _expected_type, _nullable in columns
    ):
        type_predicates.append("NEW.state_revision < 0")
    if (
        operation == "update"
        and table_name == "alpaca_managed_positions"
        and any(column_name == "id" for column_name, _expected_type, _nullable in columns)
    ):
        type_predicates.append("NEW.id <= 0")
    predicate = " OR ".join(type_predicates)
    trigger_name = f"leveraged_trader_{table_name}_identity_{operation}_guard"
    return f"""
        CREATE TRIGGER {trigger_name}
        BEFORE {operation.upper()} ON {table_name}
        FOR EACH ROW
        WHEN {predicate}
        BEGIN
            SELECT RAISE(ABORT, 'invalid managed storage identity');
        END
    """


def _install_storage_identity_type_guards(conn: sqlite3.Connection) -> None:
    """Enforce canonical identity storage classes after repairing loose imports."""
    for table_name, columns in _STORAGE_IDENTITY_TYPE_CONTRACTS.items():
        for operation in ("insert", "update"):
            trigger_name = f"leveraged_trader_{table_name}_identity_{operation}_guard"
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
            conn.execute(
                _storage_identity_type_guard_sql(
                    table_name,
                    columns,
                    operation,
                )
            )


def _drop_owned_storage_migration_triggers(conn: sqlite3.Connection) -> None:
    """Remove owned triggers that a legacy table rename may have retargeted."""
    for trigger_name, table_name in conn.execute(
        "SELECT name, tbl_name FROM sqlite_master WHERE type = 'trigger'"
    ).fetchall():
        normalized_name = str(trigger_name)
        if _OWNED_STORAGE_TRIGGER_TABLES.get(normalized_name) != str(table_name):
            continue
        conn.execute(f'DROP TRIGGER IF EXISTS "{normalized_name}"')


def _migrate_state_db(conn: sqlite3.Connection) -> None:
    _execute_schema_statements(conn, _STATE_DB_TABLE_SCHEMA)
    # Preflight has already proven that every surviving owned trigger has an
    # exact canonical or mechanically rename-retargeted body. Remove them
    # before additive migrations perform any backfill DML, then reinstall the
    # canonical contracts after all tables have their final names and shapes.
    _drop_owned_storage_migration_triggers(conn)
    _ensure_strategy_state_columns(conn)
    _ensure_strategy_summary_rollup_columns(conn)
    _ensure_strategy_equity_columns(conn)
    _ensure_alpaca_managed_position_columns(conn)
    _ensure_alpaca_managed_sell_fill_columns(conn)
    _ensure_market_history_removal_candidate_columns(conn)
    _repair_storage_table_contracts(conn)
    _ensure_strategy_identity_uniqueness(conn)
    _ensure_strategy_state_generation_singleton(conn)
    _install_alpaca_state_revision_trigger(conn)
    _ensure_alpaca_managed_sell_fill_referential_integrity(conn)
    # Legacy columns must be added and conservatively backfilled before their
    # identities are audited.  Otherwise a valid older schema would be rejected
    # merely because an identity column was introduced in a later release.
    _ensure_core_storage_identity_constraints(conn)
    _validate_storage_table_contracts(conn)
    _install_storage_identity_type_guards(conn)
    _ensure_alpaca_active_symbol_uniqueness(conn)


def _ensure_strategy_state_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(strategy_state)").fetchall()}
    for column_name, column_type in STRATEGY_STATE_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE strategy_state ADD COLUMN {column_name} {column_type}")


def _ensure_strategy_summary_rollup_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(strategy_summary)").fetchall()}
    for column_name, column_type in STRATEGY_SUMMARY_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE strategy_summary ADD COLUMN {column_name} {column_type}")


def _ensure_strategy_equity_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(strategy_equity)").fetchall()}
    for column_name, column_type in STRATEGY_EQUITY_COLUMNS.items():
        if column_name not in existing_columns:
            # Deliberately do not backfill: a legacy curve was not authenticated
            # when it was written and must be rebuilt from canonical market data.
            conn.execute(f"ALTER TABLE strategy_equity ADD COLUMN {column_name} {column_type}")


_STRATEGY_IDENTITY_COLUMNS = {
    "strategy_config": ("asset_symbol", "signal_symbol"),
    "strategy_state": ("asset_symbol", "signal_symbol", "buy_rsi", "profit_target_multiple"),
    "strategy_summary": ("asset_symbol", "signal_symbol", "buy_rsi", "profit_target_multiple"),
    "strategy_equity": (
        "asset_symbol",
        "signal_symbol",
        "buy_rsi",
        "profit_target_multiple",
        "date",
    ),
}

_STRATEGY_IDENTITY_TYPES = {
    "strategy_config": ("text", "text"),
    "strategy_state": ("text", "text", "real", "real"),
    "strategy_summary": ("text", "text", "real", "real"),
    "strategy_equity": ("text", "text", "real", "real", "text"),
}

_STRATEGY_IDENTITY_AFFINITIES = _STRATEGY_IDENTITY_TYPES


def _table_has_unique_index_for_columns(
    conn: sqlite3.Connection,
    table_name: str,
    columns: tuple[str, ...],
) -> bool:
    for index_row in conn.execute(f"PRAGMA index_list({table_name})").fetchall():
        if not bool(index_row[2]) or bool(index_row[4]):
            continue
        index_name = str(index_row[1]).replace("'", "''")
        key_rows = tuple(row for row in conn.execute(f"PRAGMA index_xinfo('{index_name}')").fetchall() if bool(row[5]))
        if (
            len(key_rows) == len(columns)
            and set(row[2] for row in key_rows) == set(columns)
            and all(int(row[1]) >= 0 for row in key_rows)
            and all(str(row[4]).upper() == "BINARY" for row in key_rows)
        ):
            return True
    return False


def _table_has_incompatible_unique_index_for_columns(
    conn: sqlite3.Connection,
    table_name: str,
    columns: tuple[str, ...],
) -> bool:
    """Detect unique constraints that can reject otherwise-distinct identities."""
    expected_column_set = set(columns)
    for index_row in conn.execute(f"PRAGMA index_list({table_name})").fetchall():
        if not bool(index_row[2]):
            continue
        raw_index_name = str(index_row[1])
        index_name = raw_index_name.replace("'", "''")
        key_rows = tuple(row for row in conn.execute(f"PRAGMA index_xinfo('{index_name}')").fetchall() if bool(row[5]))
        simple_columns = {str(row[2]) for row in key_rows if int(row[1]) >= 0 and row[2] is not None}
        identity_rows = [row for row in key_rows if row[2] in expected_column_set]
        if any(str(row[4]).upper() != "BINARY" for row in identity_rows):
            return True
        if any(int(row[1]) < 0 for row in key_rows):
            if expected_column_set.issubset(simple_columns):
                continue
            sql_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
                (raw_index_name,),
            ).fetchone()
            normalized_sql = _normalized_index_key_clause(
                None if sql_row is None else sql_row[0],
                table_name,
            )
            if any(re.search(rf"(?<![a-z0-9_]){re.escape(column)}(?![a-z0-9_])", normalized_sql) for column in columns):
                return True
        intersection = simple_columns.intersection(expected_column_set)
        if intersection and not expected_column_set.issubset(simple_columns):
            return True
    return False


def _ensure_strategy_identity_uniqueness(conn: sqlite3.Connection) -> None:
    """Restore mandatory identity constraints on legacy/custom strategy tables.

    SQLite's ``CREATE TABLE IF NOT EXISTS`` does not repair a table that was
    imported without its primary key.  Silently choosing one duplicate row
    would make resume and reporting nondeterministic, so ambiguous imports fail
    clearly; unambiguous tables receive an equivalent unique index in place.
    """
    for table_name, columns in _STRATEGY_IDENTITY_COLUMNS.items():
        _validate_identity_rows(
            conn,
            table_name,
            columns,
            storage_types=_STRATEGY_IDENTITY_TYPES[table_name],
            declared_affinities=_STRATEGY_IDENTITY_AFFINITIES[table_name],
            identity_label="strategy identities",
        )
        _validate_table_unique_indexes(
            conn,
            table_name,
            _TABLE_ALLOWED_UNIQUE_IDENTITIES[table_name],
        )
        existing_columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        missing_columns = set(columns).difference(existing_columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{table_name} is missing required strategy identity columns ({missing}); "
                "rebuild or repair the imported state database."
            )
        null_predicate = " OR ".join(f"{column} IS NULL" for column in columns)
        if conn.execute(f"SELECT 1 FROM {table_name} WHERE {null_predicate} LIMIT 1").fetchone() is not None:
            raise ValueError(
                f"{table_name} contains a NULL strategy identity; rebuild or repair the imported state database."
            )
        identity_columns = ", ".join(columns)
        duplicate = conn.execute(
            f"""
            SELECT {identity_columns}
            FROM {table_name}
            GROUP BY {identity_columns}
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise ValueError(
                f"{table_name} contains duplicate strategy identities; "
                "rebuild or resolve the duplicate rows before continuing."
            )
        if _table_has_incompatible_unique_index_for_columns(conn, table_name, columns):
            raise ValueError(
                f"{table_name} has an incompatible unique strategy identity collation, "
                "expression, or order; repair the imported schema before continuing."
            )
        if _table_has_unique_index_for_columns(conn, table_name, columns):
            continue
        index_name = f"leveraged_trader_{table_name}_identity_unique"
        existing_named_object = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?",
            (index_name,),
        ).fetchone()
        if existing_named_object is not None:
            raise ValueError(
                f"Could not enforce the required unique strategy identity on {table_name}; "
                f"schema object {index_name} conflicts with the required index."
            )
        conn.execute(f"CREATE UNIQUE INDEX {index_name} ON {table_name} ({identity_columns})")
        if not _table_has_unique_index_for_columns(conn, table_name, columns):
            raise ValueError(
                f"Could not enforce the required unique strategy identity on {table_name}; "
                "repair the conflicting schema object before continuing."
            )


_CORE_STORAGE_IDENTITIES = {
    "market_data": (("symbol", "date"), frozenset()),
    "rsi_values": (("signal_symbol", "rsi_period", "date"), frozenset()),
    "market_history_removal_candidates": (("symbol",), frozenset()),
    "alpaca_managed_sell_fills_alpaca_order": (
        ("alpaca_order_id",),
        frozenset(),
    ),
    "alpaca_symbol_aliases": (("alpaca_asset_id", "symbol"), frozenset()),
    "alpaca_managed_positions_buy_client_order": (
        ("buy_client_order_id",),
        frozenset(),
    ),
    "alpaca_managed_positions_sell_client_order": (
        ("sell_client_order_id",),
        frozenset({"sell_client_order_id"}),
    ),
    "alpaca_managed_positions_sell_alpaca_order": (
        ("sell_alpaca_order_id",),
        frozenset({"sell_alpaca_order_id"}),
    ),
}

_CORE_STORAGE_IDENTITY_TYPES = {
    "market_data": ("text", "text"),
    "rsi_values": ("text", "integer", "text"),
    "market_history_removal_candidates": ("text",),
    "alpaca_managed_sell_fills_alpaca_order": ("text",),
    "alpaca_symbol_aliases": ("text", "text"),
    "alpaca_managed_positions_buy_client_order": ("text",),
    "alpaca_managed_positions_sell_client_order": ("text",),
    "alpaca_managed_positions_sell_alpaca_order": ("text",),
}

_CORE_STORAGE_IDENTITY_AFFINITIES = {
    "market_data": ("text", "text"),
    "rsi_values": ("text", "integer", "text"),
    "market_history_removal_candidates": ("text",),
    "alpaca_managed_sell_fills_alpaca_order": ("text",),
    "alpaca_symbol_aliases": ("text", "text"),
    "alpaca_managed_positions_buy_client_order": ("text",),
    "alpaca_managed_positions_sell_client_order": ("text",),
    "alpaca_managed_positions_sell_alpaca_order": ("text",),
}

_TABLE_ALLOWED_UNIQUE_IDENTITIES = {
    **{table_name: (columns,) for table_name, columns in _STRATEGY_IDENTITY_COLUMNS.items()},
    "market_data": (("symbol", "date"),),
    "rsi_values": (("signal_symbol", "rsi_period", "date"),),
    "market_history_removal_candidates": (("symbol",),),
    "alpaca_managed_sell_fills": (
        ("managed_position_id", "alpaca_order_id"),
        ("alpaca_order_id",),
    ),
    "alpaca_symbol_aliases": (("alpaca_asset_id", "symbol"),),
    "alpaca_managed_positions": (
        ("buy_client_order_id",),
        ("sell_client_order_id",),
        ("sell_alpaca_order_id",),
    ),
}


def _core_storage_identity_table_name(identity_name: str) -> str:
    if identity_name.startswith("alpaca_managed_positions_"):
        return "alpaca_managed_positions"
    if identity_name.startswith("alpaca_managed_sell_fills_"):
        return "alpaca_managed_sell_fills"
    return identity_name


def _storage_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


_ALPACA_ACTIVE_IDENTITY_INDEX_SQL = {
    "alpaca_managed_positions_one_active_symbol": """
        CREATE UNIQUE INDEX alpaca_managed_positions_one_active_symbol
        ON alpaca_managed_positions(UPPER(symbol))
        WHERE closed_at IS NULL
    """,
    "alpaca_managed_positions_one_active_asset": """
        CREATE UNIQUE INDEX alpaca_managed_positions_one_active_asset
        ON alpaca_managed_positions(alpaca_asset_id)
        WHERE closed_at IS NULL AND alpaca_asset_id IS NOT NULL
    """,
}


def _normalized_schema_sql(sql: str | None) -> str:
    """Normalize owned DDL without conflating distinct quoted identifiers."""
    if sql is None:
        return ""
    normalized: list[str] = []
    position = 0
    while position < len(sql):
        character = sql[position]
        if character.isspace():
            position += 1
            continue
        if character == "'":
            value, position = _read_schema_sql_delimited_token(
                sql,
                position,
                closing="'",
                doubled_escape=True,
            )
            normalized.append(f"@string[{value.encode('utf-8').hex()}]")
            continue
        if character in {'"', "`", "["}:
            closing = "]" if character == "[" else character
            value, position = _read_schema_sql_delimited_token(
                sql,
                position,
                closing=closing,
                doubled_escape=True,
            )
            lowered_value = value.lower()
            if re.fullmatch(r"[a-z_][a-z0-9_]*", lowered_value):
                # SQLite treats ordinary quoted and unquoted identifiers
                # equivalently. Keep unusual quoted contents opaque so
                # ``"sym bol"`` can never collapse into ``symbol``.
                normalized.append(lowered_value)
            else:
                normalized.append(f"@identifier[{lowered_value.encode('utf-8').hex()}]")
            continue
        normalized.append(character.lower())
        position += 1
    return "".join(normalized)


def _read_schema_sql_delimited_token(
    sql: str,
    position: int,
    *,
    closing: str,
    doubled_escape: bool,
) -> tuple[str, int]:
    """Decode one SQLite quoted token and return its value and next offset."""
    value: list[str] = []
    position += 1
    while position < len(sql):
        character = sql[position]
        if character != closing:
            value.append(character)
            position += 1
            continue
        if doubled_escape and position + 1 < len(sql) and sql[position + 1] == closing:
            value.append(closing)
            position += 2
            continue
        return "".join(value), position + 1
    # sqlite_master contains only parsed SQL, but retain a distinct marker if
    # a mocked or externally supplied value is nevertheless unterminated.
    value.append("<unterminated>")
    return "".join(value), position


@cache
def _canonical_storage_schema_object_contracts() -> dict[str, tuple[str, str, str]]:
    """Return exact contracts for every explicit index and trigger we own."""
    with closing(sqlite3.connect(":memory:")) as reference:
        _migrate_state_db(reference)
        contracts = {
            str(name).lower(): (
                str(object_type),
                str(table_name),
                _normalized_schema_sql(str(sql)),
            )
            for object_type, name, table_name, sql in reference.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE type IN ('index', 'trigger') AND sql IS NOT NULL
                """
            ).fetchall()
        }

    # Canonical tables satisfy most identities through PRIMARY KEY
    # autoindexes. Constraint-stripped legacy tables instead receive these
    # exact owned indexes during repair, and a later init must recognize them.
    fallback_indexes = {
        f"leveraged_trader_{table_name}_identity_unique": (
            table_name,
            columns,
        )
        for table_name, columns in _STRATEGY_IDENTITY_COLUMNS.items()
    }
    fallback_indexes.update(
        {
            f"leveraged_trader_{identity_name}_identity_unique": (
                _core_storage_identity_table_name(identity_name),
                columns,
            )
            for identity_name, (columns, _nullable_columns) in _CORE_STORAGE_IDENTITIES.items()
        }
    )
    for index_name, (table_name, columns) in fallback_indexes.items():
        index_sql = f"CREATE UNIQUE INDEX {index_name} ON {table_name} ({', '.join(columns)})"
        contracts.setdefault(
            index_name.lower(),
            ("index", table_name, _normalized_schema_sql(index_sql)),
        )
    return contracts


def _existing_identity_guard_sql(
    conn: sqlite3.Connection,
    trigger_name: str,
    table_name: str,
) -> str | None:
    """Build the one legacy guard form valid for the table's present columns."""
    for operation in ("insert", "update"):
        expected_name = f"leveraged_trader_{table_name}_identity_{operation}_guard"
        if trigger_name.lower() != expected_name.lower():
            continue
        existing_columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        present_contracts = tuple(
            contract for contract in _STORAGE_IDENTITY_TYPE_CONTRACTS[table_name] if contract[0] in existing_columns
        )
        if not present_contracts:
            return None
        return _normalized_schema_sql(
            _storage_identity_type_guard_sql(
                table_name,
                present_contracts,
                operation,
            )
        )
    return None


def _preflight_existing_storage_schema_objects(conn: sqlite3.Connection) -> None:
    """Reject unowned or altered main/temporary objects on managed tables."""
    owned_tables = {table_name.lower() for table_name in _canonical_storage_table_contracts()}
    contracts = _canonical_storage_schema_object_contracts()
    schema_objects = [
        (schema_name, *row)
        for schema_name, schema_table in (
            ("main", "sqlite_master"),
            ("temp", "sqlite_temp_master"),
        )
        for row in conn.execute(f"SELECT type, name, tbl_name, sql FROM {schema_table}").fetchall()
    ]
    for schema_name, object_type, name, table_name, sql in schema_objects:
        normalized_name = str(name).lower()
        normalized_table = str(table_name).lower()
        contract = contracts.get(normalized_name)
        attached_to_owned_table = normalized_table in owned_tables

        if schema_name == "temp" and (attached_to_owned_table or contract is not None):
            raise ValueError(
                f"Temporary schema object {name} conflicts with managed table "
                f"{table_name}; remove it before initializing the state database."
            )

        if contract is not None and str(name) != normalized_name:
            raise ValueError(
                f"Managed schema object {name} must use canonical casing "
                f"{normalized_name}; repair the imported state database."
            )

        if contract is None:
            if object_type not in {"index", "trigger"} or not attached_to_owned_table:
                continue
            if object_type == "index" and sql is None and normalized_name.startswith("sqlite_autoindex_"):
                continue
            raise ValueError(
                f"Managed table {table_name} has an unexpected explicit {object_type} "
                f"({name}); remove it before initializing the state database."
            )

        expected_type, expected_table, expected_sql = contract
        actual_sql = _normalized_schema_sql(None if sql is None else str(sql))
        acceptable_sql = {expected_sql}
        if expected_type == "trigger" and normalized_table == expected_table.lower():
            # SQLite rewrites cross-table trigger references when a legacy
            # table is renamed for a constraint-stripped rebuild. If that
            # imported table is then dropped, the surviving owned trigger has
            # one of these exact, inert retargetings. It is safe to recognize
            # because migration drops all owned triggers before any backfill.
            acceptable_sql.update(
                {
                    expected_sql.replace(
                        "fromalpaca_managed_sell_fills",
                        "fromimported_alpaca_managed_sell_fills",
                    ),
                    expected_sql.replace(
                        "fromalpaca_managed_positions",
                        "fromimported_positions",
                    ),
                    expected_sql.replace(
                        "fromalpaca_managed_positions",
                        "fromimported_alpaca_managed_positions",
                    ),
                }
            )
            legacy_guard_sql = _existing_identity_guard_sql(
                conn,
                str(name),
                expected_table,
            )
            if legacy_guard_sql is not None:
                acceptable_sql.add(legacy_guard_sql)
        active_index_is_compatible = bool(
            expected_type == "index"
            and normalized_name in _ALPACA_ACTIVE_IDENTITY_INDEX_SQL
            and object_type == "index"
            and normalized_table == expected_table.lower()
            and _active_identity_index_is_semantically_valid(conn, str(name))
        )
        if (
            object_type != expected_type
            or normalized_table != expected_table.lower()
            or actual_sql not in acceptable_sql
            and not active_index_is_compatible
        ):
            raise ValueError(
                f"Schema object {name} conflicts with its required managed "
                f"{expected_type} contract; rebuild or repair the imported state database."
            )


def _normalized_index_key_clause(sql: str | None, table_name: str) -> str:
    normalized = _normalized_schema_sql(sql)
    table_marker = f"on{_normalized_schema_sql(table_name)}"
    if table_marker not in normalized:
        return ""
    return _strip_redundant_parentheses(normalized.rsplit(table_marker, 1)[1].split("where", 1)[0])


def _outer_parentheses_are_balanced(value: str) -> bool:
    depth = 0
    for index, character in enumerate(value):
        depth += character == "("
        depth -= character == ")"
        if depth == 0 and index != len(value) - 1:
            return False
    return depth == 0


def _strip_redundant_parentheses(value: str) -> str:
    while value.startswith("(") and value.endswith(")") and _outer_parentheses_are_balanced(value):
        value = value[1:-1]
    return value


def _normalized_index_where_clause(sql: str | None) -> str:
    normalized = _normalized_schema_sql(sql)
    if "where" not in normalized:
        return ""
    predicate = _strip_redundant_parentheses(normalized.split("where", 1)[1])
    terms = [_strip_redundant_parentheses(term) for term in predicate.split("and")]
    return "and".join(sorted(terms))


def _active_identity_index_kind(
    conn: sqlite3.Connection,
    index_name: str,
) -> str | None:
    escaped_name = index_name.replace("'", "''")
    index_row = conn.execute(
        "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?",
        (index_name,),
    ).fetchone()
    if index_row is None or index_row[0:2] != ("index", "alpaca_managed_positions"):
        return None
    key_rows = tuple(row for row in conn.execute(f"PRAGMA index_xinfo('{escaped_name}')").fetchall() if bool(row[5]))
    if len(key_rows) != 1 or int(key_rows[0][3]) != 0 or str(key_rows[0][4]).upper() != "BINARY":
        return None
    predicate = _normalized_index_where_clause(index_row[2])
    key_expression = _normalized_index_key_clause(
        index_row[2],
        "alpaca_managed_positions",
    )
    if key_expression.endswith("collatebinary"):
        key_expression = key_expression.removesuffix("collatebinary")
    if key_rows[0][2] is None and key_expression == "upper(symbol)" and predicate == "closed_atisnull":
        return "symbol"
    if key_rows[0][2] == "alpaca_asset_id" and predicate == "alpaca_asset_idisnotnullandclosed_atisnull":
        return "asset"
    return None


def _active_identity_index_is_semantically_valid(
    conn: sqlite3.Connection,
    index_name: str,
) -> bool:
    expected_kind = "symbol" if index_name == "alpaca_managed_positions_one_active_symbol" else "asset"
    return _active_identity_index_kind(conn, index_name) == expected_kind


def _validate_table_unique_indexes(
    conn: sqlite3.Connection,
    table_name: str,
    allowed_identities: tuple[tuple[str, ...], ...],
) -> None:
    """Reject unique constraints that collapse distinct canonical identities."""
    for index_row in conn.execute(f"PRAGMA index_list({table_name})").fetchall():
        if not bool(index_row[2]):
            continue
        raw_index_name = str(index_row[1])
        escaped_name = raw_index_name.replace("'", "''")
        key_rows = tuple(
            row for row in conn.execute(f"PRAGMA index_xinfo('{escaped_name}')").fetchall() if bool(row[5])
        )
        simple_binary_columns = {
            str(row[2])
            for row in key_rows
            if int(row[1]) >= 0 and row[2] is not None and str(row[4]).upper() == "BINARY"
        }
        if any(set(identity).issubset(simple_binary_columns) for identity in allowed_identities):
            continue
        if table_name == "alpaca_managed_positions" and _active_identity_index_kind(conn, raw_index_name) is not None:
            continue
        raise ValueError(
            f"{table_name} has an unrecognized unique constraint that can collapse distinct "
            "canonical identities; repair the imported schema before continuing."
        )


def _validate_no_extra_active_identity_indexes(conn: sqlite3.Connection) -> None:
    expected_names = set(_ALPACA_ACTIVE_IDENTITY_INDEX_SQL)
    for index_row in conn.execute("PRAGMA index_list(alpaca_managed_positions)").fetchall():
        if not bool(index_row[2]) or str(index_row[1]) in expected_names:
            continue
        if _active_identity_index_kind(conn, str(index_row[1])) is not None:
            continue
        index_name = str(index_row[1]).replace("'", "''")
        key_rows = tuple(row for row in conn.execute(f"PRAGMA index_xinfo('{index_name}')").fetchall() if bool(row[5]))
        key_column = key_rows[0][2] if len(key_rows) == 1 else None
        index_sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (str(index_row[1]),),
        ).fetchone()
        normalized_sql = _normalized_index_key_clause(
            None if index_sql_row is None else index_sql_row[0],
            "alpaca_managed_positions",
        )
        owner_columns = {"symbol", "alpaca_asset_id"}
        references_owner_expression = key_column is None and any(
            re.search(rf"(?<![a-z0-9_]){column}(?![a-z0-9_])", normalized_sql) for column in owner_columns
        )
        if (
            key_column in owner_columns
            or references_owner_expression
            or any(row[2] in owner_columns for row in key_rows)
        ):
            raise ValueError(
                "alpaca_managed_positions has an extra unique index that conflicts with "
                "closed-position symbol or asset reuse; repair the imported database before "
                "continuing."
            )


def _validate_existing_active_identity_indexes(conn: sqlite3.Connection) -> None:
    for index_name in _ALPACA_ACTIVE_IDENTITY_INDEX_SQL:
        row = conn.execute(
            "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?",
            (index_name,),
        ).fetchone()
        if row is not None and not _active_identity_index_is_semantically_valid(
            conn,
            index_name,
        ):
            raise ValueError(
                f"Schema object {index_name} conflicts with the required active-position "
                "identity index; repair the imported database before continuing."
            )
    _validate_no_extra_active_identity_indexes(conn)


def _sqlite_declared_type_affinity(declared_type: object) -> str:
    normalized = str(declared_type).upper()
    if "INT" in normalized:
        return "integer"
    if any(token in normalized for token in ("CHAR", "CLOB", "TEXT")):
        return "text"
    if not normalized or "BLOB" in normalized:
        return "blob"
    if any(token in normalized for token in ("REAL", "FLOA", "DOUB")):
        return "real"
    return "numeric"


def _validate_identity_rows(
    conn: sqlite3.Connection,
    table_name: str,
    columns: tuple[str, ...],
    *,
    nullable_columns: frozenset[str] = frozenset(),
    storage_types: tuple[str, ...] | None = None,
    declared_affinities: tuple[str, ...] | None = None,
    identity_label: str = "required identities",
) -> None:
    column_rows = {str(row[1]): row for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    existing_columns = set(column_rows)
    missing_columns = set(columns).difference(existing_columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"{table_name} is missing required identity columns ({missing}); "
            "rebuild or repair the imported state database."
        )

    if declared_affinities is not None:
        invalid_affinities = [
            column
            for column, expected_affinity in zip(
                columns,
                declared_affinities,
                strict=True,
            )
            if _sqlite_declared_type_affinity(column_rows[column][2]) != expected_affinity
        ]
        if invalid_affinities:
            invalid = ", ".join(invalid_affinities)
            raise ValueError(
                f"{table_name} has incompatible declared affinity for identity columns "
                f"({invalid}); rebuild or repair the imported state database."
            )

    required_columns = tuple(column for column in columns if column not in nullable_columns)
    if required_columns:
        null_predicate = " OR ".join(f"{column} IS NULL" for column in required_columns)
        if conn.execute(f"SELECT 1 FROM {table_name} WHERE {null_predicate} LIMIT 1").fetchone() is not None:
            raise ValueError(
                f"{table_name} contains a NULL required identity; rebuild or repair the imported state database."
            )

    if storage_types is not None:
        invalid_type_predicate = " OR ".join(
            f"({column} IS NOT NULL AND TYPEOF({column}) != '{storage_type}')"
            for column, storage_type in zip(columns, storage_types, strict=True)
        )
        if conn.execute(f"SELECT 1 FROM {table_name} WHERE {invalid_type_predicate} LIMIT 1").fetchone() is not None:
            raise ValueError(
                f"{table_name} contains an invalid identity storage type; "
                "rebuild or repair the imported state database."
            )

    identity_columns = ", ".join(columns)
    non_null_predicate = " AND ".join(f"{column} IS NOT NULL" for column in nullable_columns)
    duplicate_where = f"WHERE {non_null_predicate}" if non_null_predicate else ""
    duplicate = conn.execute(
        f"""
        SELECT {identity_columns}
        FROM {table_name}
        {duplicate_where}
        GROUP BY {identity_columns}
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        raise ValueError(
            f"{table_name} contains duplicate {identity_label}; "
            "rebuild or resolve the duplicate rows before continuing."
        )

    if _table_has_incompatible_unique_index_for_columns(conn, table_name, columns):
        raise ValueError(
            f"{table_name} has an incompatible unique identity collation, expression, or order; "
            "rebuild or repair the imported state database."
        )


def _validate_managed_position_primary_key(conn: sqlite3.Connection) -> None:
    position_columns = {
        str(row[1]): row for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)").fetchall()
    }
    id_column = position_columns.get("id")
    primary_key_columns = [row for row in position_columns.values() if int(row[5]) > 0]
    if id_column is None:
        raise ValueError(
            "alpaca_managed_positions is missing required identity column id; "
            "rebuild or repair the imported state database."
        )
    if (
        str(id_column[2]).strip().upper() != "INTEGER"
        or int(id_column[5]) != 1
        or len(primary_key_columns) != 1
        or any(
            bool(row[2])
            for row in conn.execute("PRAGMA index_list(alpaca_managed_positions)").fetchall()
            if str(row[3]).lower() == "pk"
        )
        or not any(
            str(row[1]) == "alpaca_managed_positions" and int(row[4]) == 0
            for row in conn.execute("PRAGMA table_list").fetchall()
        )
    ):
        # A unique index cannot restore SQLite's automatic rowid allocation.
        raise ValueError(
            "alpaca_managed_positions.id must be a rowid-backed INTEGER PRIMARY KEY; "
            "rebuild or repair the imported state database."
        )
    if conn.execute("SELECT 1 FROM alpaca_managed_positions WHERE id <= 0 LIMIT 1").fetchone() is not None:
        raise ValueError(
            "alpaca_managed_positions.id must contain only positive managed identities; "
            "rebuild or repair the imported state database."
        )


def _validate_managed_position_owner_identities(conn: sqlite3.Connection) -> None:
    column_rows = {str(row[1]): row for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)").fetchall()}
    symbol_column = column_rows.get("symbol")
    if (
        symbol_column is None
        or _sqlite_declared_type_affinity(symbol_column[2]) != "text"
        or not bool(symbol_column[3])
        or symbol_column[4] is not None
    ):
        raise ValueError(
            "alpaca_managed_positions.symbol must be a non-null TEXT identity; "
            "rebuild or repair the imported state database."
        )
    asset_column = column_rows.get("alpaca_asset_id")
    if asset_column is not None and (
        _sqlite_declared_type_affinity(asset_column[2]) != "text"
        or bool(asset_column[3])
        or asset_column[4] is not None
    ):
        raise ValueError(
            "alpaca_managed_positions.alpaca_asset_id must be a nullable TEXT identity; "
            "rebuild or repair the imported state database."
        )
    signal_symbol_column = column_rows.get("signal_symbol")
    if (
        signal_symbol_column is None
        or _sqlite_declared_type_affinity(signal_symbol_column[2]) != "text"
        or not bool(signal_symbol_column[3])
        or signal_symbol_column[4] is not None
    ):
        raise ValueError(
            "alpaca_managed_positions.signal_symbol must be a non-null TEXT identity; "
            "rebuild or repair the imported state database."
        )
    closed_column = column_rows.get("closed_at")
    if (
        closed_column is None
        or _sqlite_declared_type_affinity(closed_column[2]) != "text"
        or bool(closed_column[3])
        or closed_column[4] is not None
    ):
        raise ValueError(
            "alpaca_managed_positions.closed_at must be a nullable TEXT lifecycle marker; "
            "rebuild or repair the imported state database."
        )
    selected_columns = "symbol" + (", alpaca_asset_id" if asset_column is not None else "")
    for row in conn.execute(
        f"SELECT {selected_columns}, TYPEOF(symbol)"
        + (", TYPEOF(alpaca_asset_id)" if asset_column is not None else "")
        + " FROM alpaca_managed_positions"
    ).fetchall():
        symbol = row[0]
        if (
            row[2 if asset_column is not None else 1] != "text"
            or not str(symbol).strip()
            or str(symbol) != str(symbol).strip()
        ):
            raise ValueError(
                "alpaca_managed_positions contains an invalid symbol identity; "
                "repair the imported state database before continuing."
            )
        try:
            _canonical_managed_symbol(
                symbol,
                field_name="Imported managed Alpaca position symbol",
            )
        except ValueError as exc:
            raise ValueError(
                "alpaca_managed_positions contains a noncanonical symbol identity; "
                "repair the imported state database before continuing."
            ) from exc
        if asset_column is not None:
            asset_id, asset_storage_type = row[1], row[3]
            if asset_id is not None and (
                asset_storage_type != "text" or not str(asset_id).strip() or str(asset_id) != str(asset_id).strip()
            ):
                raise ValueError(
                    "alpaca_managed_positions contains an invalid Alpaca asset identity; "
                    "repair the imported state database before continuing."
                )
            if asset_id is not None:
                try:
                    _canonical_alpaca_asset_id(
                        asset_id,
                        field_name="Imported managed Alpaca position asset ID",
                    )
                except ValueError as exc:
                    raise ValueError(
                        "alpaca_managed_positions contains a noncanonical Alpaca asset identity; "
                        "repair the imported state database before continuing."
                    ) from exc
    for signal_symbol, storage_type in conn.execute(
        "SELECT signal_symbol, TYPEOF(signal_symbol) FROM alpaca_managed_positions"
    ).fetchall():
        if storage_type != "text":
            raise ValueError(
                "alpaca_managed_positions contains an invalid managed identity storage value; "
                "repair the imported state database before continuing."
            )
        try:
            _canonical_managed_symbol(
                signal_symbol,
                field_name="Imported managed Alpaca position signal symbol",
            )
        except ValueError as exc:
            raise ValueError(
                "alpaca_managed_positions contains a noncanonical signal symbol identity; "
                "repair the imported state database before continuing."
            ) from exc


def _validate_managed_position_economics(conn: sqlite3.Connection) -> None:
    """Reject imported strategy economics that cannot safely drive protection."""
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)").fetchall()}
    if "closed_sell_shortfall_reopen_pending" in columns:
        invalid_shortfall_control = conn.execute(
            """
            SELECT 1
            FROM alpaca_managed_positions
            WHERE TYPEOF(closed_sell_shortfall_reopen_pending) != 'integer'
               OR closed_sell_shortfall_reopen_pending NOT IN (0, 1)
            LIMIT 1
            """
        ).fetchone()
        if invalid_shortfall_control is not None:
            raise ValueError(
                "alpaca_managed_positions contains an invalid closed-sell shortfall control value; "
                "repair the imported state database before continuing."
            )
    if "buy_status" in columns:
        invalid_buy_status = conn.execute(
            "SELECT 1 FROM alpaca_managed_positions WHERE TYPEOF(buy_status) != 'text' LIMIT 1"
        ).fetchone()
        if invalid_buy_status is not None:
            raise ValueError(
                "alpaca_managed_positions contains an invalid buy lifecycle status storage value; "
                "repair the imported state database before continuing."
            )
    if "sell_status" in columns:
        invalid_sell_status = conn.execute(
            """
            SELECT 1
            FROM alpaca_managed_positions
            WHERE sell_status IS NOT NULL AND TYPEOF(sell_status) != 'text'
            LIMIT 1
            """
        ).fetchone()
        if invalid_sell_status is not None:
            raise ValueError(
                "alpaca_managed_positions contains an invalid sell lifecycle status storage value; "
                "repair the imported state database before continuing."
            )
    economics_columns = {"buy_rsi", "profit_target_multiple"}
    if economics_columns.issubset(columns):
        rows = conn.execute(
            """
            SELECT id, buy_rsi, profit_target_multiple,
                   TYPEOF(buy_rsi), TYPEOF(profit_target_multiple)
            FROM alpaca_managed_positions
            """
        ).fetchall()
        for _position_id, buy_rsi, profit_target_multiple, buy_rsi_type, profit_target_type in rows:
            if buy_rsi_type not in ("integer", "real") or profit_target_type not in ("integer", "real"):
                raise ValueError(
                    "alpaca_managed_positions contains an invalid managed identity storage value; "
                    "repair the imported state database before continuing."
                )
            try:
                _normalize_managed_strategy_economics(
                    buy_rsi,
                    profit_target_multiple,
                )
            except ValueError as exc:
                raise ValueError(
                    "alpaca_managed_positions contains invalid managed strategy economics; "
                    "repair the imported state database before continuing."
                ) from exc

    if "target_sell_price" in columns:
        for target_sell_price, storage_type in conn.execute(
            """
            SELECT target_sell_price, TYPEOF(target_sell_price)
            FROM alpaca_managed_positions
            WHERE target_sell_price IS NOT NULL
            """
        ).fetchall():
            if storage_type not in ("integer", "real"):
                raise ValueError(
                    "alpaca_managed_positions contains an invalid managed target sell price; "
                    "repair the imported state database before continuing."
                )
            try:
                _normalize_managed_target_sell_price(target_sell_price)
            except ValueError as exc:
                raise ValueError(
                    "alpaca_managed_positions contains an invalid managed target sell price; "
                    "repair the imported state database before continuing."
                ) from exc

    buy_intent_columns = {"buy_order_qty", "buy_order_limit_price"}
    if buy_intent_columns.issubset(columns):
        rows = conn.execute(
            """
            SELECT buy_order_qty, buy_order_limit_price,
                   TYPEOF(buy_order_qty), TYPEOF(buy_order_limit_price)
            FROM alpaca_managed_positions
            """
        ).fetchall()
        for buy_order_qty, buy_order_limit_price, qty_type, price_type in rows:
            if (buy_order_qty is not None and qty_type not in ("integer", "real")) or (
                buy_order_limit_price is not None and price_type not in ("integer", "real")
            ):
                raise ValueError(
                    "alpaca_managed_positions contains an invalid managed buy intent; "
                    "repair the imported state database before continuing."
                )
            try:
                _normalize_optional_managed_buy_intent(
                    buy_order_qty,
                    buy_order_limit_price,
                )
            except ValueError as exc:
                raise ValueError(
                    "alpaca_managed_positions contains an invalid managed buy intent; "
                    "repair the imported state database before continuing."
                ) from exc

    buy_causality_columns = {
        "buy_order_qty",
        "buy_order_limit_price",
        "buy_status",
        "sell_status",
        "filled_qty",
        "filled_avg_price",
        "notes",
    }
    if buy_causality_columns.issubset(columns):
        quarantine_marker_expression = "buy_causality_quarantine" if "buy_causality_quarantine" in columns else "NULL"
        rows = conn.execute(
            f"""
            SELECT buy_order_qty, buy_order_limit_price, buy_status,
                   filled_qty, filled_avg_price, notes,
                   {quarantine_marker_expression}
            FROM alpaca_managed_positions
            """
        ).fetchall()
        for (
            buy_order_qty,
            buy_order_limit_price,
            buy_status,
            filled_qty,
            filled_avg_price,
            notes,
            quarantine_marker,
        ) in rows:
            try:
                _validate_managed_buy_causality_quarantine(
                    buy_order_qty=buy_order_qty,
                    buy_order_limit_price=buy_order_limit_price,
                    buy_status=buy_status,
                    filled_qty=filled_qty,
                    filled_avg_price=filled_avg_price,
                    notes=notes,
                    quarantine_marker=quarantine_marker,
                )
            except ValueError as exc:
                raise ValueError(
                    "alpaca_managed_positions contains an unquarantined managed buy fill that conflicts "
                    "with its immutable intent; repair the imported state database before continuing."
                ) from exc

    sell_intent_columns = {"sell_order_qty", "sell_order_limit_price"}
    if sell_intent_columns.issubset(columns):
        rows = conn.execute(
            """
            SELECT sell_order_qty, sell_order_limit_price,
                   TYPEOF(sell_order_qty), TYPEOF(sell_order_limit_price)
            FROM alpaca_managed_positions
            """
        ).fetchall()
        for sell_order_qty, sell_order_limit_price, qty_type, price_type in rows:
            if (sell_order_qty is not None and qty_type not in ("integer", "real")) or (
                sell_order_limit_price is not None and price_type not in ("integer", "real")
            ):
                raise ValueError(
                    "alpaca_managed_positions contains an invalid managed sell intent; "
                    "repair the imported state database before continuing."
                )
            try:
                _normalize_optional_managed_sell_intent(
                    sell_order_qty,
                    sell_order_limit_price,
                )
            except ValueError as exc:
                raise ValueError(
                    "alpaca_managed_positions contains an invalid managed sell intent; "
                    "repair the imported state database before continuing."
                ) from exc


def _validate_managed_sell_fill_economics(conn: sqlite3.Connection) -> None:
    """Reject imported sell-ledger values that runtime reconciliation cannot use."""
    if not _storage_table_exists(conn, "alpaca_managed_sell_fills"):
        return
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_sell_fills)").fetchall()}
    if not {"managed_position_id", "filled_qty", "filled_value"}.issubset(columns):
        # The table-contract preflight reports missing foundational columns.
        return
    submitted_qty_expression = "submitted_qty" if "submitted_qty" in columns else "NULL"
    submitted_limit_expression = "submitted_limit_price" if "submitted_limit_price" in columns else "NULL"
    rows = conn.execute(
        f"""
        SELECT managed_position_id, filled_qty, filled_value,
               {submitted_qty_expression}, {submitted_limit_expression},
               TYPEOF(filled_qty), TYPEOF(filled_value),
               TYPEOF({submitted_qty_expression}), TYPEOF({submitted_limit_expression})
        FROM alpaca_managed_sell_fills
        """
    ).fetchall()
    totals: dict[object, tuple[float, float]] = {}
    for (
        position_id,
        filled_qty,
        filled_value,
        submitted_qty,
        submitted_limit_price,
        filled_qty_type,
        filled_value_type,
        submitted_qty_type,
        submitted_limit_type,
    ) in rows:
        invalid_fill_storage_type = bool(
            filled_qty_type not in ("integer", "real") or filled_value_type not in ("integer", "real")
        )
        invalid_submitted_storage_type = bool(
            (submitted_qty is not None and submitted_qty_type not in ("integer", "real"))
            or (submitted_limit_price is not None and submitted_limit_type not in ("integer", "real"))
        )
        if invalid_fill_storage_type:
            raise ValueError(
                "alpaca_managed_sell_fills contains invalid managed sell-fill economics; "
                "repair the imported state database before continuing."
            )
        try:
            normalized_filled_qty, normalized_filled_value = _normalize_managed_sell_fill_economics(
                filled_qty,
                filled_value,
            )
        except ValueError as exc:
            raise ValueError(
                "alpaca_managed_sell_fills contains invalid managed sell-fill economics; "
                "repair the imported state database before continuing."
            ) from exc
        if invalid_submitted_storage_type:
            raise ValueError(
                "alpaca_managed_sell_fills contains invalid managed sell-generation economics; "
                "repair the imported state database before continuing."
            )
        try:
            _normalize_optional_managed_sell_intent(
                submitted_qty,
                submitted_limit_price,
            )
        except ValueError as exc:
            raise ValueError(
                "alpaca_managed_sell_fills contains invalid managed sell-generation economics; "
                "repair the imported state database before continuing."
            ) from exc

        prior_qty, prior_value = totals.get(position_id, (0.0, 0.0))
        total_qty = prior_qty + normalized_filled_qty
        total_value = prior_value + normalized_filled_value
        if not math.isfinite(total_qty) or not math.isfinite(total_value):
            raise ValueError(
                "alpaca_managed_sell_fills contains overflowing cumulative sell economics; "
                "repair the imported state database before continuing."
            )
        totals[position_id] = total_qty, total_value

    _validate_managed_sell_parent_ledger_accounting(conn, totals)
    _validate_current_managed_sell_intent_ledger_consistency(conn)


def _validate_managed_sell_parent_ledger_accounting(
    conn: sqlite3.Connection,
    ledger_totals: Mapping[object, tuple[float, float]],
) -> None:
    """Reject materialized parent accounting that disagrees with its ledger."""
    if not _storage_table_exists(conn, "alpaca_managed_positions"):
        return
    position_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)").fetchall()}
    accounting_columns = (
        "filled_qty",
        "filled_avg_price",
        "sell_filled_qty",
        "sell_filled_avg_price",
        "realized_pl",
        "realized_pl_pct",
        "sold_qty",
        "sold_value",
        "remaining_qty",
    )
    if not {"id", *accounting_columns}.issubset(position_columns):
        # Additive legacy columns are audited again after migration.  Deferring
        # this comparison keeps the preflight compatible with those schemas.
        return

    target_sell_price_expression = "target_sell_price" if "target_sell_price" in position_columns else "NULL"
    buy_limit_price_expression = "buy_order_limit_price" if "buy_order_limit_price" in position_columns else "NULL"
    sell_limit_price_expression = "sell_order_limit_price" if "sell_order_limit_price" in position_columns else "NULL"
    type_expressions = ", ".join(f"TYPEOF({column})" for column in accounting_columns)
    rows = conn.execute(
        f"""
        SELECT id, {", ".join(accounting_columns)},
               {target_sell_price_expression},
               {buy_limit_price_expression},
               {sell_limit_price_expression},
               {type_expressions}
        FROM alpaca_managed_positions
        """
    ).fetchall()
    error_message = (
        "alpaca_managed_positions contains managed sell accounting that conflicts with its "
        "alpaca_managed_sell_fills ledger; repair the imported state database before continuing."
    )

    for row in rows:
        position_id = row[0]
        accounting_values = row[1:10]
        target_sell_price, buy_limit_price, sell_limit_price = row[10:13]
        accounting_types = row[13:22]
        try:
            for column, value, storage_type in zip(
                accounting_columns,
                accounting_values,
                accounting_types,
                strict=True,
            ):
                if value is None:
                    if column in {"sold_qty", "sold_value"}:
                        raise ValueError("Managed cumulative sell accounting cannot be null.")
                elif storage_type not in ("integer", "real"):
                    raise ValueError("Managed sell accounting must use numeric SQLite storage.")

            (
                filled_qty,
                filled_avg_price,
                sell_filled_qty,
                sell_filled_avg_price,
                realized_pl,
                realized_pl_pct,
                sold_qty,
                sold_value,
                remaining_qty,
            ) = (None if value is None else float(value) for value in accounting_values)
            numeric_values = (
                filled_qty,
                filled_avg_price,
                sell_filled_qty,
                sell_filled_avg_price,
                realized_pl,
                realized_pl_pct,
                sold_qty,
                sold_value,
                remaining_qty,
            )
            if any(value is not None and not math.isfinite(value) for value in numeric_values):
                raise ValueError("Managed sell accounting must remain finite.")
            assert sold_qty is not None
            assert sold_value is not None
            if sold_qty < 0.0 or sold_value < 0.0:
                raise ValueError("Managed cumulative sell accounting cannot be negative.")
            if sell_filled_qty is not None and sell_filled_qty < 0.0:
                raise ValueError("Managed cumulative sell quantity cannot be negative.")
            if sell_filled_avg_price is not None and sell_filled_avg_price <= 0.0:
                raise ValueError("Managed cumulative sell average price must be positive.")

            parent_accounting_is_materialized = bool(
                filled_qty is not None
                or filled_avg_price is not None
                or sell_filled_qty is not None
                or sell_filled_avg_price is not None
                or realized_pl is not None
                or realized_pl_pct is not None
                or remaining_qty is not None
                or sold_qty != 0.0
                or sold_value != 0.0
            )
            if not parent_accounting_is_materialized:
                # An early schema stored sell-ledger rows before it had enough
                # authoritative parent state to derive these aggregates.  Leave
                # that explicitly incomplete legacy shape migratable.
                continue

            ledger_qty, ledger_value = ledger_totals.get(position_id, (0.0, 0.0))
            cumulative_sell_avg_price = ledger_value / ledger_qty if ledger_qty > 0.0 else None
            mark_prices = tuple(
                None if value is None else float(value)
                for value in (
                    filled_avg_price,
                    target_sell_price,
                    buy_limit_price,
                    sell_limit_price,
                    cumulative_sell_avg_price,
                )
            )
            if any(value is not None and not math.isfinite(value) for value in mark_prices):
                raise ValueError("Managed accounting mark prices must remain finite.")
            value_scale = max(abs(ledger_value), abs(sold_value))

            def values_match(left: float, right: float) -> bool:
                tolerance = managed_value_reconciliation_tolerance(max(abs(left), abs(right)))
                return abs(left - right) <= tolerance

            if not _managed_accounting_quantities_match(
                sold_qty,
                ledger_qty,
                mark_prices=mark_prices,
                value_scale=value_scale,
            ) or not values_match(sold_value, ledger_value):
                raise ValueError("Managed cumulative sell accounting disagrees with its ledger.")
            if sell_filled_qty is not None and not _managed_accounting_quantities_match(
                sell_filled_qty,
                ledger_qty,
                mark_prices=mark_prices,
                value_scale=value_scale,
            ):
                raise ValueError("Managed aggregate sell-fill quantity disagrees with its ledger.")
            if sell_filled_avg_price is not None and (
                sell_filled_qty is None
                or cumulative_sell_avg_price is None
                or not values_match(sell_filled_avg_price, cumulative_sell_avg_price)
            ):
                raise ValueError("Managed aggregate sell-fill price disagrees with its ledger.")

            normalized_buy_qty, normalized_buy_avg_price = _normalize_optional_managed_buy_fill_economics(
                filled_qty,
                filled_avg_price,
            )
            if normalized_buy_qty is None:
                if (
                    ledger_qty > 0.0
                    or remaining_qty is not None
                    or realized_pl is not None
                    or realized_pl_pct is not None
                ):
                    raise ValueError("Managed sell accounting is missing its filled buy basis.")
                continue
            expected_remaining_qty = normalized_buy_qty - ledger_qty
            quantity_scale = max(abs(expected_remaining_qty), normalized_buy_qty, ledger_qty)
            if remaining_qty is not None and not _managed_accounting_quantities_match(
                remaining_qty,
                expected_remaining_qty,
                mark_prices=mark_prices,
                value_scale=value_scale,
            ):
                raise ValueError("Managed remaining quantity disagrees with its ledger.")

            realized_accounting_expected = False
            expected_realized_pl: float | None = None
            expected_realized_pl_pct: float | None = None
            if normalized_buy_qty > 0.0:
                if normalized_buy_avg_price is None:
                    raise ValueError("Managed sell accounting is missing its filled buy price.")
                accounting_overfill = bool(
                    expected_remaining_qty < 0.0
                    and not _managed_accounting_residual_is_negligible(
                        expected_remaining_qty,
                        quantity_scale=quantity_scale,
                        mark_prices=mark_prices,
                        value_scale=ledger_value,
                    )
                )
                matched_qty = min(ledger_qty, normalized_buy_qty)
                realized_accounting_expected = bool(
                    not accounting_overfill
                    and _managed_accounting_quantity_is_positive(
                        matched_qty,
                        quantity_scale=quantity_scale,
                        mark_prices=(
                            normalized_buy_avg_price,
                            None if target_sell_price is None else float(target_sell_price),
                            cumulative_sell_avg_price,
                        ),
                        value_scale=ledger_value,
                    )
                )
                if realized_accounting_expected:
                    expected_realized_pl, expected_realized_pl_pct = _managed_realized_pl_values(
                        sold_value=ledger_value,
                        matched_qty=matched_qty,
                        buy_price=normalized_buy_avg_price,
                    )
            if realized_accounting_expected:
                if (
                    realized_pl is None
                    or realized_pl_pct is None
                    or expected_realized_pl is None
                    or expected_realized_pl_pct is None
                    or not values_match(realized_pl, expected_realized_pl)
                    or not values_match(realized_pl_pct, expected_realized_pl_pct)
                ):
                    raise ValueError("Managed realized P/L disagrees with its ledger.")
            elif realized_pl is not None or realized_pl_pct is not None:
                raise ValueError("Managed realized P/L cannot exist without a matched sale.")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(error_message) from exc


def _validate_current_managed_sell_intent_ledger_consistency(
    conn: sqlite3.Connection,
) -> None:
    """Reject a current parent/ledger pair with conflicting immutable intent."""
    if not _storage_table_exists(conn, "alpaca_managed_positions"):
        return
    position_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)").fetchall()}
    fill_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_sell_fills)").fetchall()}
    required_position_columns = {
        "id",
        "sell_alpaca_order_id",
        "sell_order_qty",
        "sell_order_limit_price",
    }
    required_fill_columns = {
        "managed_position_id",
        "alpaca_order_id",
        "submitted_qty",
        "submitted_limit_price",
    }
    if not required_position_columns.issubset(position_columns) or not required_fill_columns.issubset(fill_columns):
        return
    rows = conn.execute(
        """
        SELECT positions.sell_order_qty, positions.sell_order_limit_price,
               fills.submitted_qty, fills.submitted_limit_price
        FROM alpaca_managed_positions AS positions
        JOIN alpaca_managed_sell_fills AS fills
          ON fills.managed_position_id = positions.id
         AND fills.alpaca_order_id = positions.sell_alpaca_order_id
        """
    ).fetchall()
    for parent_qty, parent_limit_price, ledger_qty, ledger_limit_price in rows:
        try:
            normalized_parent_qty, normalized_parent_limit_price = _normalize_optional_managed_sell_intent(
                parent_qty,
                parent_limit_price,
            )
            _raise_if_managed_sell_intent_replay_conflicts(
                persisted_qty=ledger_qty,
                persisted_limit_price=ledger_limit_price,
                observed_qty=normalized_parent_qty,
                observed_limit_price=normalized_parent_limit_price,
            )
        except ValueError as exc:
            raise ValueError(
                "A current alpaca_managed_positions sell intent conflicts with its immutable "
                "alpaca_managed_sell_fills generation; repair the imported state database "
                "before continuing."
            ) from exc


def _validate_managed_sell_order_ownership_consistency(conn: sqlite3.Connection) -> None:
    """Reject one broker sell identity claimed by different parent and ledger rows."""
    if not (
        _storage_table_exists(conn, "alpaca_managed_positions")
        and _storage_table_exists(conn, "alpaca_managed_sell_fills")
    ):
        return
    position_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)").fetchall()}
    fill_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_sell_fills)").fetchall()}
    if not {"id", "sell_alpaca_order_id"}.issubset(position_columns) or not {
        "managed_position_id",
        "alpaca_order_id",
    }.issubset(fill_columns):
        return
    conflict = conn.execute(
        """
        SELECT positions.sell_alpaca_order_id
        FROM alpaca_managed_positions AS positions
        JOIN alpaca_managed_sell_fills AS fills
          ON fills.alpaca_order_id = positions.sell_alpaca_order_id
        WHERE positions.id != fills.managed_position_id
        LIMIT 1
        """
    ).fetchone()
    if conflict is not None:
        raise ValueError(
            "An Alpaca sell order ID belongs to different managed positions in parent "
            "and sell-fill storage; repair the imported state database before continuing."
        )


def _validate_managed_symbol_alias_identities(conn: sqlite3.Connection) -> None:
    if not _storage_table_exists(conn, "alpaca_symbol_aliases"):
        return
    columns = {str(row[1]): row for row in conn.execute("PRAGMA table_info(alpaca_symbol_aliases)").fetchall()}
    if not {"alpaca_asset_id", "symbol"}.issubset(columns):
        raise ValueError(
            "alpaca_symbol_aliases is missing a managed identity column; "
            "repair the imported state database before continuing."
        )
    rows = conn.execute(
        """
        SELECT alpaca_asset_id, symbol, TYPEOF(alpaca_asset_id), TYPEOF(symbol)
        FROM alpaca_symbol_aliases
        """
    ).fetchall()
    for asset_id, symbol, asset_type, symbol_type in rows:
        if asset_type != "text" or symbol_type != "text":
            raise ValueError(
                "alpaca_symbol_aliases contains invalid identity storage types; "
                "repair the imported state database before continuing."
            )
        try:
            _canonical_alpaca_asset_id(
                asset_id,
                field_name="Imported managed Alpaca alias asset ID",
            )
            _canonical_managed_symbol(
                symbol,
                field_name="Imported managed Alpaca alias symbol",
            )
        except ValueError as exc:
            raise ValueError(
                "alpaca_symbol_aliases contains a noncanonical managed identity; "
                "repair the imported state database before continuing."
            ) from exc


def _strategy_state_generation_rows_are_valid(conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        "SELECT id, generation, TYPEOF(id), TYPEOF(generation) FROM strategy_state_generation"
    ).fetchall()
    return bool(
        not rows or (len(rows) == 1 and rows[0][2:] == ("integer", "integer") and rows[0][0] == 1 and rows[0][1] >= 0)
    )


def _validate_strategy_state_generation_schema(conn: sqlite3.Connection) -> None:
    column_rows = {str(row[1]): row for row in conn.execute("PRAGMA table_info(strategy_state_generation)").fetchall()}
    missing_columns = {"id", "generation"}.difference(column_rows)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "strategy_state_generation is missing required columns "
            f"({missing}); rebuild or repair the imported state database."
        )
    if any(_sqlite_declared_type_affinity(column_rows[column][2]) != "integer" for column in ("id", "generation")):
        raise ValueError(
            "strategy_state_generation id and generation must have INTEGER affinity; "
            "rebuild or repair the imported state database."
        )


def _preflight_existing_storage_identities(conn: sqlite3.Connection) -> None:
    """Reject ambiguous imported identities before any schema mutation."""
    for table_name, columns in _STRATEGY_IDENTITY_COLUMNS.items():
        if _storage_table_exists(conn, table_name):
            _validate_identity_rows(
                conn,
                table_name,
                columns,
                storage_types=_STRATEGY_IDENTITY_TYPES[table_name],
                declared_affinities=_STRATEGY_IDENTITY_AFFINITIES[table_name],
                identity_label="strategy identities",
            )
            _validate_table_unique_indexes(
                conn,
                table_name,
                _TABLE_ALLOWED_UNIQUE_IDENTITIES[table_name],
            )

    if _storage_table_exists(conn, "alpaca_managed_positions"):
        _validate_managed_position_primary_key(conn)
        _validate_managed_position_owner_identities(conn)
        _validate_managed_position_economics(conn)
        _validate_existing_active_identity_indexes(conn)
        _validate_table_unique_indexes(
            conn,
            "alpaca_managed_positions",
            _TABLE_ALLOWED_UNIQUE_IDENTITIES["alpaca_managed_positions"],
        )
        position_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)").fetchall()
        }
        if "closed_at" in position_columns:
            duplicate_active_symbols = conn.execute(
                """
                SELECT UPPER(symbol)
                FROM alpaca_managed_positions
                WHERE closed_at IS NULL
                GROUP BY UPPER(symbol)
                HAVING COUNT(*) > 1
                ORDER BY UPPER(symbol)
                """
            ).fetchall()
            if duplicate_active_symbols:
                symbols = ", ".join(str(row[0]) for row in duplicate_active_symbols)
                raise ValueError(
                    "Managed Alpaca state contains multiple active positions for the same symbol "
                    f"({symbols}); "
                    "resolve the duplicate rows before running broker automation."
                )
        if "closed_at" in position_columns and "alpaca_asset_id" in position_columns:
            duplicate_active_asset = conn.execute(
                """
                SELECT 1
                FROM alpaca_managed_positions
                WHERE closed_at IS NULL AND alpaca_asset_id IS NOT NULL
                GROUP BY alpaca_asset_id
                HAVING COUNT(*) > 1
                LIMIT 1
                """
            ).fetchone()
            if duplicate_active_asset is not None:
                raise ValueError(
                    "Managed Alpaca state contains multiple active positions for the same "
                    "stable asset; resolve the duplicate rows before broker automation."
                )

    _validate_managed_symbol_alias_identities(conn)

    for identity_name, (columns, nullable_columns) in _CORE_STORAGE_IDENTITIES.items():
        table_name = _core_storage_identity_table_name(identity_name)
        if not _storage_table_exists(conn, table_name):
            continue
        existing_columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        # ``sell_client_order_id`` was added after managed-position storage was
        # introduced and is safely nullable/backfillable.  Every other identity
        # is foundational and cannot be inferred.
        if set(columns).difference(existing_columns) == {"sell_client_order_id"}:
            continue
        _validate_identity_rows(
            conn,
            table_name,
            columns,
            nullable_columns=nullable_columns,
            storage_types=_CORE_STORAGE_IDENTITY_TYPES[identity_name],
            declared_affinities=_CORE_STORAGE_IDENTITY_AFFINITIES[identity_name],
            identity_label=(
                "Alpaca sell-order identities"
                if identity_name
                in {
                    "alpaca_managed_sell_fills_alpaca_order",
                    "alpaca_managed_positions_sell_alpaca_order",
                }
                else "required identities"
            ),
        )
        if table_name != "alpaca_managed_positions":
            _validate_table_unique_indexes(
                conn,
                table_name,
                _TABLE_ALLOWED_UNIQUE_IDENTITIES[table_name],
            )

    _validate_existing_storage_identity_types(conn)
    _validate_managed_sell_fill_economics(conn)
    _validate_managed_sell_order_ownership_consistency(conn)

    if _storage_table_exists(conn, "strategy_state_generation"):
        _validate_strategy_state_generation_schema(conn)
        if not _strategy_state_generation_rows_are_valid(conn):
            raise ValueError(
                "strategy_state_generation must contain at most one non-negative integer "
                "generation at id 1; rebuild or repair the imported state database."
            )

    if (
        _storage_table_exists(conn, "alpaca_managed_positions")
        and _storage_table_exists(conn, "alpaca_managed_sell_fills")
        and conn.execute(
            """
            SELECT 1
            FROM alpaca_managed_sell_fills AS fills
            LEFT JOIN alpaca_managed_positions AS positions
              ON positions.id = fills.managed_position_id
            WHERE positions.id IS NULL
            LIMIT 1
            """
        ).fetchone()
        is not None
    ):
        raise ValueError(
            "alpaca_managed_sell_fills contains an orphan managed position identity; "
            "repair the imported state database before continuing."
        )


def _validate_existing_storage_identity_types(conn: sqlite3.Connection) -> None:
    """Reject rows that could bypass identity guards before those guards exist."""
    for table_name, contracts in _STORAGE_IDENTITY_TYPE_CONTRACTS.items():
        if not _storage_table_exists(conn, table_name):
            continue
        existing_columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
        predicates: list[str] = []
        for column_name, expected_type, nullable in contracts:
            if column_name not in existing_columns:
                continue
            allowed_types = "('integer', 'real')" if expected_type == "real" else f"('{expected_type}')"
            if nullable:
                predicates.append(f"({column_name} IS NOT NULL AND TYPEOF({column_name}) NOT IN {allowed_types})")
            else:
                predicates.append(f"({column_name} IS NULL OR TYPEOF({column_name}) NOT IN {allowed_types})")
        if table_name == "alpaca_managed_positions" and "state_revision" in existing_columns:
            predicates.append("state_revision < 0")
        if table_name == "alpaca_managed_positions" and "id" in existing_columns:
            predicates.append("id <= 0")
        if (
            predicates
            and conn.execute(f"SELECT 1 FROM {table_name} WHERE {' OR '.join(predicates)} LIMIT 1").fetchone()
            is not None
        ):
            raise ValueError(
                f"{table_name} contains an invalid managed identity storage value; "
                "rebuild or repair the imported state database."
            )


def _ensure_core_storage_identity_constraints(conn: sqlite3.Connection) -> None:
    """Restore identities that SQLite upserts and broker ownership depend on."""
    _validate_managed_position_primary_key(conn)
    _validate_managed_position_owner_identities(conn)
    _validate_managed_position_economics(conn)
    _validate_managed_sell_fill_economics(conn)
    _validate_table_unique_indexes(
        conn,
        "alpaca_managed_positions",
        _TABLE_ALLOWED_UNIQUE_IDENTITIES["alpaca_managed_positions"],
    )

    for identity_name, (columns, nullable_columns) in _CORE_STORAGE_IDENTITIES.items():
        table_name = _core_storage_identity_table_name(identity_name)
        _validate_identity_rows(
            conn,
            table_name,
            columns,
            nullable_columns=nullable_columns,
            storage_types=_CORE_STORAGE_IDENTITY_TYPES[identity_name],
            declared_affinities=_CORE_STORAGE_IDENTITY_AFFINITIES[identity_name],
            identity_label=(
                "Alpaca sell-order identities"
                if identity_name
                in {
                    "alpaca_managed_sell_fills_alpaca_order",
                    "alpaca_managed_positions_sell_alpaca_order",
                }
                else "required identities"
            ),
        )
        if table_name != "alpaca_managed_positions":
            _validate_table_unique_indexes(
                conn,
                table_name,
                _TABLE_ALLOWED_UNIQUE_IDENTITIES[table_name],
            )
        identity_columns = ", ".join(columns)
        if _table_has_unique_index_for_columns(conn, table_name, columns):
            continue
        index_name = f"leveraged_trader_{identity_name}_identity_unique"
        existing_named_object = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?",
            (index_name,),
        ).fetchone()
        if existing_named_object is not None:
            raise ValueError(
                f"Could not enforce the required unique identity on {table_name}; "
                f"schema object {index_name} conflicts with the required index."
            )
        conn.execute(f"CREATE UNIQUE INDEX {index_name} ON {table_name} ({identity_columns})")
        if not _table_has_unique_index_for_columns(conn, table_name, columns):
            raise ValueError(
                f"Could not enforce the required unique identity on {table_name}; "
                "repair the conflicting schema object before continuing."
            )
    _validate_managed_sell_order_ownership_consistency(conn)


def _ensure_strategy_state_generation_singleton(conn: sqlite3.Connection) -> None:
    """Validate the global invalidation counter before creating its seed row."""
    _validate_strategy_state_generation_schema(conn)
    if not _strategy_state_generation_rows_are_valid(conn):
        raise ValueError(
            "strategy_state_generation must contain exactly one non-negative integer "
            "generation at id 1; rebuild or repair the imported state database."
        )
    for trigger_name in (
        "strategy_state_generation_validate_insert",
        "strategy_state_generation_validate_update",
        "strategy_state_generation_validate_delete",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
    row_count = conn.execute("SELECT COUNT(*) FROM strategy_state_generation").fetchone()[0]
    if row_count == 0:
        conn.execute("INSERT INTO strategy_state_generation (id, generation) VALUES (1, 0)")
    if not _table_has_unique_index_for_columns(
        conn,
        "strategy_state_generation",
        ("id",),
    ):
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "leveraged_trader_strategy_state_generation_identity_unique "
            "ON strategy_state_generation (id)"
        )
    if not _table_has_unique_index_for_columns(
        conn,
        "strategy_state_generation",
        ("id",),
    ):
        raise ValueError(
            "Could not enforce the strategy_state_generation singleton identity; "
            "repair the conflicting schema object before continuing."
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS "
        "leveraged_trader_strategy_state_generation_singleton_unique "
        "ON strategy_state_generation ((1))"
    )
    # These triggers carry constraints that cannot be added to a legacy table.
    conn.execute(
        """
        CREATE TRIGGER strategy_state_generation_validate_insert
        BEFORE INSERT ON strategy_state_generation
        FOR EACH ROW
        WHEN TYPEOF(NEW.id) != 'integer' OR NEW.id != 1
          OR TYPEOF(NEW.generation) != 'integer' OR NEW.generation < 0
          OR EXISTS (SELECT 1 FROM strategy_state_generation)
        BEGIN
            SELECT RAISE(ABORT, 'invalid strategy state generation');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER strategy_state_generation_validate_update
        BEFORE UPDATE ON strategy_state_generation
        FOR EACH ROW
        WHEN TYPEOF(NEW.id) != 'integer' OR NEW.id != 1
          OR TYPEOF(NEW.generation) != 'integer' OR NEW.generation < 0
          OR NEW.generation != OLD.generation + 1
        BEGIN
            SELECT RAISE(ABORT, 'invalid strategy state generation');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER strategy_state_generation_validate_delete
        BEFORE DELETE ON strategy_state_generation
        FOR EACH ROW
        BEGIN
            SELECT RAISE(ABORT, 'cannot delete strategy state generation');
        END
        """
    )


def _managed_closed_at_is_canonical(value: object, storage_type: object) -> bool:
    """Return whether a persisted close marker matches a runtime-written form."""
    if storage_type != "text" or not isinstance(value, str) or not value:
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value):
        try:
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return False
        return True
    return _normalize_alpaca_closure_timestamp(value) == value


def _managed_closed_at_precedes_causal_boundary(
    closed_at: object,
    *causal_boundaries: object,
) -> bool:
    """Return whether a usable close marker predates any usable lifecycle fact."""

    def parsed_timestamp(value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            normalized = _normalize_alpaca_broker_timestamp(value)
        except ValueError:
            return None
        if normalized is None:
            return None
        return datetime.fromisoformat(normalized.replace("Z", "+00:00"))

    parsed_closed_at = parsed_timestamp(closed_at)
    if parsed_closed_at is None:
        return False
    return any(
        parsed_boundary is not None and parsed_closed_at < parsed_boundary
        for parsed_boundary in (parsed_timestamp(boundary) for boundary in causal_boundaries)
    )


def _ensure_alpaca_managed_position_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)").fetchall()}
    shortfall_control_was_missing = "closed_sell_shortfall_reopen_pending" not in existing_columns
    for column_name, column_type in ALPACA_MANAGED_POSITION_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE alpaca_managed_positions ADD COLUMN {column_name} {column_type}")
    # Older reconcilers used an exact fragment in the shared free-form notes as
    # executable state. Promote interrupted closed-shortfall repairs before any
    # later diagnostic can replace or sanitize that note.
    if shortfall_control_was_missing:
        conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET closed_sell_shortfall_reopen_pending = 1
            WHERE INSTR(COALESCE(notes, ''), ?) > 0
            """,
            (_LEGACY_CLOSED_SELL_SHORTFALL_REOPEN_NOTE,),
        )
    # Earlier schemas retained impossible buy-fill diagnostics only in the
    # shared free-form notes column. Promote every exact recognized diagnostic
    # while it is still present so a later sell-side note cannot erase the
    # durable realized-P/L quarantine introduced by the structured column.
    legacy_causality_rows = conn.execute(
        """
        SELECT id, buy_order_qty, buy_order_limit_price, buy_status,
               filled_qty, filled_avg_price, notes, buy_causality_quarantine
        FROM alpaca_managed_positions
        """
    ).fetchall()
    causality_backfills: list[tuple[str, int, object | None]] = []
    for (
        position_id,
        buy_order_qty,
        buy_order_limit_price,
        buy_status,
        filled_qty,
        filled_avg_price,
        notes,
        quarantine_marker,
    ) in legacy_causality_rows:
        normalized_filled_qty, normalized_filled_avg_price = _normalize_optional_managed_buy_fill_economics(
            filled_qty,
            filled_avg_price,
        )
        if normalized_filled_qty is None:
            continue
        issue = _managed_buy_fill_causality_issue(
            buy_order_qty=buy_order_qty,
            buy_order_limit_price=buy_order_limit_price,
            buy_status=str(buy_status),
            filled_qty=normalized_filled_qty,
            filled_avg_price=normalized_filled_avg_price,
        )
        if issue is None:
            continue
        required_marker = _managed_buy_causality_quarantine_note(issue)
        if isinstance(notes, str) and required_marker in notes and quarantine_marker != required_marker:
            causality_backfills.append((required_marker, int(position_id), quarantine_marker))
    conn.executemany(
        """
        UPDATE alpaca_managed_positions
        SET buy_causality_quarantine = ?
        WHERE id = ? AND buy_causality_quarantine IS ?
        """,
        causality_backfills,
    )
    legacy_rows = conn.execute(
        """
        SELECT id, buy_client_order_id
        FROM alpaca_managed_positions
        WHERE sell_order_namespace IS NULL
        """
    ).fetchall()
    conn.executemany(
        """
        UPDATE alpaca_managed_positions
        SET sell_order_namespace = ?
        WHERE id = ? AND sell_order_namespace IS NULL
        """,
        [
            (alpaca_exit_order_namespace(str(buy_client_order_id)), int(position_id))
            for position_id, buy_client_order_id in legacy_rows
        ],
    )
    # Older writers could persist arbitrary broker values directly in
    # ``closed_at``. SQLite accepts integers and numeric-looking strings as
    # Julian dates, so datetime(closed_at) alone cannot distinguish them from a
    # runtime-written timestamp. A canonical-looking marker that predates a
    # causal lifecycle timestamp is likewise impossible. The original close
    # instant cannot be trusted; replace every malformed or impossible marker
    # with migration time and clear any stale audit marker so the row is
    # examined again now.
    noncanonical_closed_position_ids = [
        (int(position_id),)
        for (
            position_id,
            closed_at,
            storage_type,
            buy_signal_date,
            buy_submitted_at,
            filled_at,
            sell_submitted_at,
            sell_filled_at,
        ) in conn.execute(
            """
            SELECT id, closed_at, TYPEOF(closed_at), buy_signal_date,
                   buy_submitted_at, filled_at, sell_submitted_at, sell_filled_at
            FROM alpaca_managed_positions
            WHERE closed_at IS NOT NULL
            """
        ).fetchall()
        if not _managed_closed_at_is_canonical(closed_at, storage_type)
        or _managed_closed_at_precedes_causal_boundary(
            closed_at,
            buy_signal_date,
            buy_submitted_at,
            filled_at,
            sell_submitted_at,
            sell_filled_at,
        )
    ]
    conn.executemany(
        """
        UPDATE alpaca_managed_positions
        SET closed_at = CURRENT_TIMESTAMP,
            closed_correction_audited_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND closed_at IS NOT NULL
        """,
        noncanonical_closed_position_ids,
    )
    _install_alpaca_state_revision_trigger(conn)


def _install_alpaca_state_revision_trigger(conn: sqlite3.Connection) -> None:
    """Replace the revision fence triggers and verify their exact owned schema."""
    trigger_contracts = (
        (
            _ALPACA_STATE_REVISION_GUARD_TRIGGER_NAME,
            _ALPACA_STATE_REVISION_GUARD_TRIGGER_SQL,
        ),
        (_ALPACA_STATE_REVISION_TRIGGER_NAME, _ALPACA_STATE_REVISION_TRIGGER_SQL),
    )
    for trigger_name, _trigger_sql in trigger_contracts:
        existing = conn.execute(
            "SELECT type FROM sqlite_master WHERE name = ?",
            (trigger_name,),
        ).fetchone()
        if existing is not None and existing[0] != "trigger":
            raise ValueError(
                f"Schema object {trigger_name} conflicts with the required "
                "managed-position revision trigger; rebuild or repair the imported "
                "state database."
            )
    for trigger_name, trigger_sql in trigger_contracts:
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
        conn.execute(trigger_sql)
        installed = conn.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = ?",
            (trigger_name,),
        ).fetchone()
        if (
            installed is None
            or installed[0] != "trigger"
            or _normalized_schema_sql(installed[1]) != _normalized_schema_sql(trigger_sql)
        ):
            raise ValueError(
                "Could not install the required managed-position revision trigger; "
                "rebuild or repair the imported state database."
            )


def _ensure_alpaca_managed_sell_fill_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_sell_fills)").fetchall()}
    for column_name, column_type in ALPACA_MANAGED_SELL_FILL_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE alpaca_managed_sell_fills ADD COLUMN {column_name} {column_type}")
    # A legacy ledger only has immutable economics available for the exact
    # order still attached to its parent.  Backfill that provable generation;
    # older generations remain NULL so the broker audit fails closed instead
    # of blessing mutable values reported after the fact.
    conn.execute(
        """
        UPDATE alpaca_managed_sell_fills AS fills
        SET submitted_qty = COALESCE(
                submitted_qty,
                (
                    SELECT managed.sell_order_qty
                    FROM alpaca_managed_positions AS managed
                    WHERE managed.id = fills.managed_position_id
                      AND managed.sell_alpaca_order_id = fills.alpaca_order_id
                )
            ),
            submitted_limit_price = COALESCE(
                submitted_limit_price,
                (
                    SELECT managed.sell_order_limit_price
                    FROM alpaca_managed_positions AS managed
                    WHERE managed.id = fills.managed_position_id
                      AND managed.sell_alpaca_order_id = fills.alpaca_order_id
                )
            )
        WHERE submitted_qty IS NULL OR submitted_limit_price IS NULL
        """
    )


def _ensure_alpaca_managed_sell_fill_referential_integrity(
    conn: sqlite3.Connection,
) -> None:
    """Enforce the parent relation even when SQLite foreign keys are disabled."""
    orphan = conn.execute(
        """
        SELECT 1
        FROM alpaca_managed_sell_fills AS fills
        LEFT JOIN alpaca_managed_positions AS positions
          ON positions.id = fills.managed_position_id
        WHERE positions.id IS NULL
        LIMIT 1
        """
    ).fetchone()
    if orphan is not None:
        raise ValueError(
            "alpaca_managed_sell_fills contains an orphan managed position identity; "
            "repair the imported state database before continuing."
        )

    trigger_names = (
        "alpaca_managed_sell_fills_validate_parent_insert",
        "alpaca_managed_sell_fills_validate_parent_update",
        "alpaca_managed_positions_restrict_sell_fill_parent_delete",
        "alpaca_managed_positions_restrict_sell_fill_parent_id_update",
    )
    for trigger_name in trigger_names:
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
    conn.execute(
        """
        CREATE TRIGGER alpaca_managed_sell_fills_validate_parent_insert
        BEFORE INSERT ON alpaca_managed_sell_fills
        FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1 FROM alpaca_managed_positions WHERE id = NEW.managed_position_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'managed sell fill has no parent position');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER alpaca_managed_sell_fills_validate_parent_update
        BEFORE UPDATE OF managed_position_id ON alpaca_managed_sell_fills
        FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1 FROM alpaca_managed_positions WHERE id = NEW.managed_position_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'managed sell fill has no parent position');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER alpaca_managed_positions_restrict_sell_fill_parent_delete
        BEFORE DELETE ON alpaca_managed_positions
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1 FROM alpaca_managed_sell_fills WHERE managed_position_id = OLD.id
        )
        BEGIN
            SELECT RAISE(ABORT, 'managed position still owns sell fills');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER alpaca_managed_positions_restrict_sell_fill_parent_id_update
        BEFORE UPDATE OF id ON alpaca_managed_positions
        FOR EACH ROW
        WHEN NEW.id != OLD.id AND EXISTS (
            SELECT 1 FROM alpaca_managed_sell_fills WHERE managed_position_id = OLD.id
        )
        BEGIN
            SELECT RAISE(ABORT, 'managed position still owns sell fills');
        END
        """
    )


def _ensure_alpaca_active_symbol_uniqueness(conn: sqlite3.Connection) -> None:
    duplicate_symbols = conn.execute(
        """
        SELECT UPPER(symbol)
        FROM alpaca_managed_positions
        WHERE closed_at IS NULL
        GROUP BY UPPER(symbol)
        HAVING COUNT(*) > 1
        ORDER BY UPPER(symbol)
        """
    ).fetchall()
    if duplicate_symbols:
        symbols = ", ".join(str(row[0]) for row in duplicate_symbols)
        raise ValueError(
            "Managed Alpaca state contains multiple active positions for the same symbol "
            f"({symbols}); resolve the duplicate rows before running broker automation."
        )
    _validate_existing_active_identity_indexes(conn)
    if (
        conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'alpaca_managed_positions_one_active_symbol'").fetchone()
        is None
    ):
        conn.execute(_ALPACA_ACTIVE_IDENTITY_INDEX_SQL["alpaca_managed_positions_one_active_symbol"])
    duplicate_assets = conn.execute(
        """
        SELECT alpaca_asset_id
        FROM alpaca_managed_positions
        WHERE closed_at IS NULL AND alpaca_asset_id IS NOT NULL
        GROUP BY alpaca_asset_id
        HAVING COUNT(*) > 1
        ORDER BY alpaca_asset_id
        """
    ).fetchall()
    if duplicate_assets:
        asset_ids = ", ".join(str(row[0]) for row in duplicate_assets)
        raise ValueError(
            "Managed Alpaca state contains multiple active positions for the same stable asset "
            f"({asset_ids}); resolve the duplicate rows before running broker automation."
        )
    if (
        conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'alpaca_managed_positions_one_active_asset'").fetchone()
        is None
    ):
        conn.execute(_ALPACA_ACTIVE_IDENTITY_INDEX_SQL["alpaca_managed_positions_one_active_asset"])
    _validate_existing_active_identity_indexes(conn)


def alpaca_exit_order_namespace(buy_client_order_id: str) -> str:
    """Return the durable account-wide identity component for managed exits."""
    return hashlib.sha256(buy_client_order_id.encode("utf-8")).hexdigest()[:20]


def _ensure_market_history_removal_candidate_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(market_history_removal_candidates)").fetchall()
    }
    if "last_observed_run_id" not in existing_columns:
        conn.execute("ALTER TABLE market_history_removal_candidates ADD COLUMN last_observed_run_id TEXT")


def _date_str(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def rsi_entry_rule_code(entry_rule: str) -> int:
    try:
        return RSI_ENTRY_RULE_LABELS[entry_rule]
    except (KeyError, TypeError) as exc:
        supported = ", ".join(sorted(RSI_ENTRY_RULE_LABELS))
        raise ValueError(f"Unsupported RSI entry rule {entry_rule!r}; expected one of: {supported}.") from exc


def best_strategy_order_by_clause(rsi_entry_rule: str = "lower") -> str:
    """Return the shared deterministic ordering for selecting one best grid row.

    Performance ties are common when several targets have not been reached yet.
    Prefer the more selective RSI threshold for the workflow, then the smaller
    profit target, so storage pruning and user-facing reports always choose the
    same configuration even when tied rows have different pending actions.
    """
    entry_rule_code = rsi_entry_rule_code(rsi_entry_rule)
    buy_rsi_direction = "DESC" if entry_rule_code == RSI_ENTRY_UPPER else "ASC"
    return f"""
        sharpe IS NULL,
        sharpe DESC,
        total_return IS NULL,
        total_return DESC,
        cagr IS NULL,
        cagr DESC,
        buy_rsi {buy_rsi_direction},
        profit_target_multiple ASC
    """.strip()


def _validated_strategy_grid_values(
    name: str,
    values: list[float],
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool,
) -> tuple[float, ...]:
    """Validate raw public grid values before bool/NaN coercion changes identity."""
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a collection of numeric scalars.")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a collection of numeric scalars.") from exc
    normalized: list[float] = []
    invalid_types = (
        bool,
        np.bool_,
        complex,
        np.complexfloating,
        datetime_module.date,
        timedelta,
        np.datetime64,
        np.timedelta64,
        np.ndarray,
        str,
        bytes,
        bytearray,
    )
    for value in raw_values:
        if isinstance(value, invalid_types):
            raise ValueError(f"{name} must contain finite numeric scalars in the supported range.")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain finite numeric scalars in the supported range.") from exc
        in_range = minimum <= numeric <= maximum if minimum_inclusive else minimum < numeric <= maximum
        if not math.isfinite(numeric) or not in_range:
            comparison = "between" if minimum_inclusive else "greater than"
            range_description = (
                f"{comparison} {minimum:g} and {maximum:g}, inclusive"
                if minimum_inclusive
                else f"greater than {minimum:g} and at most {maximum:g}"
            )
            raise ValueError(f"{name} must contain values {range_description}.")
        normalized.append(numeric)
    return tuple(normalized)


def _validated_strategy_grid_inputs(
    base_cfg: BacktestConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    validate_strategy_simulation_configuration(base_cfg)
    normalized_buy_rsi = _validated_strategy_grid_values(
        "buy_rsi_values",
        buy_rsi_values,
        minimum=0.0,
        maximum=100.0,
        minimum_inclusive=True,
    )
    normalized_profit_targets = _validated_strategy_grid_values(
        "profit_target_values",
        profit_target_values,
        minimum=1.0,
        maximum=100.0,
        minimum_inclusive=False,
    )
    if not normalized_buy_rsi:
        raise ValueError("buy_rsi_values must contain at least one value.")
    if not normalized_profit_targets:
        raise ValueError("profit_target_values must contain at least one value.")
    return normalized_buy_rsi, normalized_profit_targets


def _canonical_fingerprint_float(value: object) -> float:
    """Return one JSON representation for semantically identical signed zero."""
    normalized = float(value)
    return 0.0 if normalized == 0.0 else normalized


def _normalized_backtest_fingerprint_payload(base_cfg: BacktestConfig) -> dict[str, object]:
    """Canonicalize validated Real subclasses for stable JSON serialization."""
    return {
        "initial_capital": _canonical_fingerprint_float(base_cfg.initial_capital),
        "rsi_period": int(base_cfg.rsi_period),
        "buy_rsi": _canonical_fingerprint_float(base_cfg.buy_rsi),
        "profit_target_multiple": _canonical_fingerprint_float(base_cfg.profit_target_multiple),
        "fee_bps": _canonical_fingerprint_float(base_cfg.fee_bps),
        "slippage_bps": _canonical_fingerprint_float(base_cfg.slippage_bps),
        "auto_adjust": base_cfg.auto_adjust,
    }


def strategy_config_fingerprint(
    base_cfg: BacktestConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rsi_entry_rule: str = "lower",
) -> str:
    """Return a stable identity for every setting that changes a simulation."""
    rsi_entry_rule_code(rsi_entry_rule)
    normalized_buy_rsi, normalized_profit_targets = _validated_strategy_grid_inputs(
        base_cfg,
        buy_rsi_values,
        profit_target_values,
    )
    payload = {
        "schema_version": STRATEGY_STATE_SCHEMA_VERSION,
        "rsi_entry_rule": rsi_entry_rule,
        "backtest": _normalized_backtest_fingerprint_payload(base_cfg),
        "buy_rsi_values": sorted({_canonical_fingerprint_float(value) for value in normalized_buy_rsi}),
        "profit_target_values": sorted({_canonical_fingerprint_float(value) for value in normalized_profit_targets}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _strategy_config_pairs(
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> set[tuple[float, float]]:
    return {
        (float(buy_rsi), float(profit_target_multiple))
        for buy_rsi in buy_rsi_values
        for profit_target_multiple in profit_target_values
    }


def _strategy_state_integrity_digest(
    *,
    asset_symbol: object,
    signal_symbol: object,
    buy_rsi: object,
    profit_target_multiple: object,
    start_date: object,
    last_date: object,
    cash: object,
    shares: object,
    in_position: object,
    entry_price: object,
    pending_action: object,
    prev_equity: object,
    trades_executed: object,
    entry_date: object = None,
) -> str:
    """Authenticate every compact account field needed for a safe resume."""
    normalized_entry_price = (
        None if entry_price is None or pd.isna(entry_price) else _canonical_integrity_float(entry_price)
    )
    payload = {
        "schema": "strategy-state-integrity-v2",
        "asset_symbol": str(asset_symbol),
        "signal_symbol": str(signal_symbol),
        "buy_rsi": _canonical_integrity_float(buy_rsi),
        "profit_target_multiple": _canonical_integrity_float(profit_target_multiple),
        "start_date": None if start_date is None else str(start_date),
        "last_date": None if last_date is None else str(last_date),
        "cash": _canonical_integrity_float(cash),
        "shares": _canonical_integrity_float(shares),
        "in_position": int(in_position),
        "entry_price": normalized_entry_price,
        "entry_date": None if entry_date is None else str(entry_date),
        "pending_action": str(pending_action),
        "prev_equity": _canonical_integrity_float(prev_equity),
        "trades_executed": int(trades_executed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_integrity_float(value: object) -> float:
    """Match the representation SQLite returns for every persisted REAL zero."""
    normalized_value = float(value)
    return 0.0 if normalized_value == 0.0 else normalized_value


def _strategy_summary_integrity_digest(row: tuple) -> str:
    """Authenticate every identity, raw rollup, and derived summary field."""
    if len(row) != len(_STRATEGY_SUMMARY_INTEGRITY_COLUMNS):
        raise ValueError("strategy summary integrity row has an unexpected shape")

    payload: dict[str, object] = {"schema": "strategy-summary-integrity-v1"}
    for column_name, value in zip(_STRATEGY_SUMMARY_INTEGRITY_COLUMNS, row, strict=True):
        if column_name in _STRATEGY_SUMMARY_TEXT_COLUMNS:
            payload[column_name] = None if value is None else str(value)
        elif column_name in _STRATEGY_SUMMARY_INTEGER_COLUMNS:
            payload[column_name] = int(value)
        elif value is None:
            payload[column_name] = None
        else:
            payload[column_name] = _canonical_integrity_float(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _strategy_equity_integrity_digest(
    *,
    asset_symbol: object,
    signal_symbol: object,
    buy_rsi: object,
    profit_target_multiple: object,
    date: object,
    equity: object,
    daily_return: object,
    risk_free_return: object,
    in_position: object,
    action_executed: object,
    pending_action: object,
    trades_executed: object,
) -> str:
    """Authenticate one retained curve observation and its sequence identity."""
    normalized_risk_free_return = (
        None if risk_free_return is None or pd.isna(risk_free_return) else _canonical_integrity_float(risk_free_return)
    )
    payload = {
        "schema": "strategy-equity-integrity-v1",
        "asset_symbol": str(asset_symbol),
        "signal_symbol": str(signal_symbol),
        "buy_rsi": _canonical_integrity_float(buy_rsi),
        "profit_target_multiple": _canonical_integrity_float(profit_target_multiple),
        "date": str(date),
        "equity": _canonical_integrity_float(equity),
        "daily_return": _canonical_integrity_float(daily_return),
        "risk_free_return": normalized_risk_free_return,
        "in_position": int(in_position),
        "action_executed": str(action_executed),
        "pending_action": str(pending_action),
        "trades_executed": int(trades_executed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _strategy_equity_integrity_row_is_valid(
    row: tuple,
    *,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> bool:
    """Verify one natural-order curve row without coercing corrupt values."""
    if not (
        len(row) == 9
        and type(row[0]) is str
        and type(row[1]) is float
        and type(row[2]) is float
        and (row[3] is None or type(row[3]) is float)
        and type(row[4]) is int
        and type(row[5]) is str
        and type(row[6]) is str
        and type(row[7]) is int
        and type(row[8]) is str
    ):
        return False
    try:
        expected_digest = _strategy_equity_integrity_digest(
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            buy_rsi=buy_rsi,
            profit_target_multiple=profit_target_multiple,
            date=row[0],
            equity=row[1],
            daily_return=row[2],
            risk_free_return=row[3],
            in_position=row[4],
            action_executed=row[5],
            pending_action=row[6],
            trades_executed=row[7],
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return row[8] == expected_digest


def _strategy_state_row_has_valid_storage_types(row: tuple) -> bool:
    """Reject SQLite storage classes that digest coercion could normalize."""
    numeric_indexes = (0, 1, 5, 6, 7)
    if any(type(row[index]) not in {int, float} for index in numeric_indexes):
        return False
    if row[9] is not None and type(row[9]) not in {int, float}:
        return False
    return bool(
        (row[2] is None or type(row[2]) is str)
        and type(row[3]) is str
        and type(row[4]) is int
        and type(row[8]) is int
        and row[8] in {0, 1}
        and (row[10] is None or type(row[10]) is str)
        and type(row[11]) is str
        and type(row[12]) is str
    )


def _strategy_summary_row_has_valid_storage_types(row: tuple) -> bool:
    """Reject natural-order summary values that can hide corruption."""
    real_indexes = (2, 3, 13, 15, 16, 17, 19, 20, 22, 23, 25, 26, 27, 28)
    integer_indexes = (6, 7, 18, 21, 24)
    optional_real_indexes = (8, 9, 10, 11, 12, 14)
    return bool(
        len(row) == len(_STRATEGY_SUMMARY_INTEGRITY_COLUMNS) + 1
        and all(type(row[index]) is float for index in real_indexes)
        and all(type(row[index]) is int for index in integer_indexes)
        and type(row[0]) is str
        and type(row[1]) is str
        and type(row[4]) is str
        and type(row[5]) is str
        and all(row[index] is None or type(row[index]) is float for index in optional_real_indexes)
        and type(row[29]) is str
    )


def _strategy_summary_integrity_row_is_valid(row: tuple) -> bool:
    """Verify one persisted summary before coercing or ranking its values."""
    if not _strategy_summary_row_has_valid_storage_types(row):
        return False
    try:
        expected_digest = _strategy_summary_integrity_digest(row[:-1])
    except (TypeError, ValueError, OverflowError):
        return False
    return row[-1] == expected_digest


def _strategy_summary_validation_row(row: tuple) -> tuple:
    """Reorder a natural schema row for the compact rollup validators."""
    return (
        row[2],
        row[3],
        row[4],
        row[5],
        row[6],
        row[7],
        row[15],
        row[16],
        row[17],
        row[18],
        row[19],
        row[20],
        row[21],
        row[22],
        row[23],
        row[24],
        row[13],
        row[25],
        row[26],
        row[27],
        row[28],
        row[8],
        row[9],
        row[10],
        row[11],
        row[12],
        row[14],
    )


def _persisted_derived_metric_matches(stored: object, expected: float | None) -> bool:
    """Compare ranking metrics exactly after recomputing their persisted rollup."""
    expected_missing = expected is None or pd.isna(expected)
    if stored is None or expected_missing:
        return stored is None and expected_missing
    return bool(type(stored) is float and math.isfinite(stored) and math.isfinite(expected) and stored == expected)


def _strategy_state_integrity_row_is_valid(
    row: tuple,
    *,
    asset_symbol: str,
    signal_symbol: str,
) -> bool:
    """Verify one persisted state row before exposing or resuming it."""
    if not _strategy_state_row_has_valid_storage_types(row):
        return False
    try:
        expected_digest = _strategy_state_integrity_digest(
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            buy_rsi=row[0],
            profit_target_multiple=row[1],
            start_date=row[2],
            last_date=row[3],
            cash=row[5],
            shares=row[7],
            in_position=row[8],
            entry_price=row[9],
            entry_date=row[10],
            pending_action=row[11],
            prev_equity=row[6],
            trades_executed=row[4],
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return row[12] == expected_digest


def _persisted_centered_moments_are_consistent(
    count: object,
    total: object,
    total_squares: object,
    mean: object,
    m2: object,
) -> bool:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return False
    try:
        total_value = float(total)
        squares_value = float(total_squares)
        mean_value = float(mean)
        m2_value = float(m2)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        not all(math.isfinite(value) for value in (total_value, squares_value, mean_value, m2_value))
        or squares_value < 0.0
        or m2_value < 0.0
    ):
        return False
    if count == 0:
        return all(abs(value) <= 1e-12 for value in (total_value, squares_value, mean_value, m2_value))

    expected_total = mean_value * count
    total_tolerance = max(
        1e-9,
        128.0 * np.finfo(np.float64).eps * count * max(abs(total_value), abs(expected_total), 1.0),
    )
    if abs(total_value - expected_total) > total_tolerance:
        return False

    observed_squares = np.longdouble(squares_value)
    expected_squares = np.longdouble(m2_value) + (
        np.longdouble(count) * np.longdouble(mean_value) * np.longdouble(mean_value)
    )
    scale = max(
        abs(observed_squares),
        abs(expected_squares),
        np.longdouble(np.finfo(np.float64).tiny),
    )
    relative_tolerance = min(
        1e-8,
        64.0 * np.finfo(np.float64).eps * count,
    )
    return bool(abs(observed_squares - expected_squares) <= relative_tolerance * scale)


def _strategy_summary_validation_row_is_semantically_valid(summary_row: tuple) -> bool:
    """Reject authenticated summaries whose rollup cannot describe real results."""
    if len(summary_row) != 27 or any(value is None for value in summary_row[6:21]):
        return False
    start_date, end_date = summary_row[2:4]
    trading_days, trades_executed = summary_row[4:6]
    return_count = summary_row[9]
    excess_return_count = summary_row[12]
    positive_return_count = summary_row[15]
    counts = (
        trading_days,
        trades_executed,
        return_count,
        excess_return_count,
        positive_return_count,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        return False
    try:
        start_timestamp = pd.Timestamp(start_date)
        end_timestamp = pd.Timestamp(end_date)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        pd.isna(start_timestamp)
        or pd.isna(end_timestamp)
        or str(start_date) != start_timestamp.date().isoformat()
        or str(end_date) != end_timestamp.date().isoformat()
        or start_timestamp.date() > end_timestamp.date()
        or trading_days < 1
        or return_count + 1 != trading_days
        or trades_executed > 2 * return_count
        or excess_return_count > return_count
        or positive_return_count > return_count
        or not _positive_return_count_is_feasible(
            return_count,
            float(summary_row[10]),
            float(summary_row[11]),
            positive_return_count,
        )
        or not _positive_return_path_is_feasible(
            count=return_count,
            positive_return_count=positive_return_count,
            first_equity=float(summary_row[6]),
            last_equity=float(summary_row[7]),
            running_max_equity=float(summary_row[8]),
            max_drawdown=float(summary_row[16]),
            total=float(summary_row[10]),
        )
    ):
        return False

    return_moments_match = _persisted_centered_moments_are_consistent(
        return_count,
        summary_row[10],
        summary_row[11],
        summary_row[17],
        summary_row[18],
    )
    excess_moments_match = _persisted_centered_moments_are_consistent(
        excess_return_count,
        summary_row[13],
        summary_row[14],
        summary_row[19],
        summary_row[20],
    )
    if not return_moments_match or not excess_moments_match:
        return False

    try:
        persisted_rollup = _summary_rollup_from_row(summary_row[6:21])
        expected_metrics = None if persisted_rollup is None else _rollup_metrics(persisted_rollup)
    except (TypeError, ValueError, OverflowError):
        return False
    if expected_metrics is None or any(
        not _persisted_derived_metric_matches(stored_value, expected_metrics[metric_name])
        for stored_value, metric_name in zip(
            summary_row[21:],
            (
                "total_return",
                "cagr",
                "annualized_vol",
                "sharpe",
                "kelly_fraction",
                "hit_rate",
            ),
            strict=True,
        )
    ):
        return False
    return bool(
        _strategy_return_moments_respect_lower_bound(
            return_count,
            float(summary_row[10]),
            float(summary_row[11]),
        )
        and _zero_trade_rollup_is_semantically_valid(summary_row)
    )


def _zero_trade_rollup_is_semantically_valid(
    summary_row: tuple,
    *,
    equity_records: list[dict] | None = None,
) -> bool:
    """A strategy with no fills cannot change its own account equity."""
    if summary_row[5] != 0:
        return True
    try:
        first_equity = float(summary_row[6])
        last_equity = float(summary_row[7])
        running_max_equity = float(summary_row[8])
        max_drawdown = float(summary_row[16])
        strategy_moments = tuple(float(value) for value in summary_row[10:12]) + tuple(
            float(value) for value in summary_row[17:19]
        )
    except (TypeError, ValueError, OverflowError):
        return False
    rollup_is_valid = bool(
        math.isfinite(first_equity)
        and last_equity == first_equity
        and running_max_equity == first_equity
        and summary_row[15] == 0
        and all(value == 0.0 for value in strategy_moments)
        and max_drawdown == 0.0
    )
    if not rollup_is_valid or equity_records is None:
        return rollup_is_valid
    return all(
        float(record["equity"]) == first_equity and float(record["daily_return"]) == 0.0 for record in equity_records
    )


def _strategy_rows_match_config(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    expected_pairs: set[tuple[float, float]],
    *,
    base_cfg: BacktestConfig,
    rsi_entry_rule: str,
) -> bool:
    try:
        expected_initial_capital = float(base_cfg.initial_capital)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(expected_initial_capital) or expected_initial_capital <= 0.0:
        return False
    state_rows = conn.execute(
        """
        SELECT buy_rsi, profit_target_multiple, start_date, last_date,
               trades_executed, cash, prev_equity, shares, in_position,
               entry_price, entry_date, pending_action, integrity_digest
        FROM strategy_state
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchall()
    if any(not _strategy_state_row_has_valid_storage_types(row) for row in state_rows):
        return False
    states_by_pair = {(row[0], row[1]): row for row in state_rows}
    if len(states_by_pair) != len(state_rows):
        return False
    state_pairs = set(states_by_pair)
    if state_pairs != expected_pairs:
        return False

    summary_rows = conn.execute(
        f"""
        SELECT {_STRATEGY_SUMMARY_INTEGRITY_COLUMN_SQL}, integrity_digest
        FROM strategy_summary
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchall()
    # Authentication must precede tuple-key coercion, rollup construction, and
    # ranking-metric validation.  A legacy NULL digest or any coordinated
    # mutation therefore fails closed into the caller's full rebuild path.
    if any(
        not _strategy_summary_integrity_row_is_valid(row)
        or not _strategy_summary_validation_row_is_semantically_valid(_strategy_summary_validation_row(row))
        for row in summary_rows
    ):
        return False
    summaries_by_pair = {(row[2], row[3]): _strategy_summary_validation_row(row) for row in summary_rows}
    if len(summaries_by_pair) != len(summary_rows):
        return False
    summary_pairs = set(summaries_by_pair)
    if summary_pairs != expected_pairs:
        return False

    session_windows: dict[tuple[str, str], tuple[str | None, str | None, int]] = {}
    shared_chronology: tuple[str, str, int, int] | None = None
    canonical_replays: dict[
        tuple[str, str],
        dict[tuple[float, float], tuple[dict, SummaryRollup]] | None,
    ] = {}
    for config_pair in expected_pairs:
        state_row = states_by_pair[config_pair]
        summary_row = summaries_by_pair[config_pair]
        if not _strategy_state_integrity_row_is_valid(
            state_row,
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
        ):
            return False
        state = _strategy_state_from_row(
            (
                state_row[2],
                state_row[3],
                state_row[5],
                state_row[7],
                state_row[8],
                state_row[9],
                state_row[10],
                state_row[11],
                state_row[6],
                state_row[4],
            )
        )
        if not _strategy_state_is_semantically_valid(
            conn,
            asset_symbol,
            state,
        ):
            return False
        state_start_date, state_last_date = state_row[2:4]
        state_trades_executed = state_row[4]
        summary_start_date, summary_end_date, trading_days, summary_trades_executed = summary_row[2:6]
        return_count = summary_row[9]
        excess_return_count = summary_row[12]
        positive_return_count = summary_row[15]

        # Compact equity history is retained only for the current best strategy,
        # so an absent rollup for any other grid point cannot be reconstructed
        # safely.  A persisted state row must also describe the same chronology
        # as its rollup: otherwise start_idx would replay or skip sessions while
        # carrying metrics from a different history window.
        if any(value is None for value in summary_row[6:21]):
            return False
        try:
            summary_first_equity = float(summary_row[6])
            summary_last_equity = float(summary_row[7])
            state_prev_equity = float(state_row[6])
        except (TypeError, ValueError, OverflowError):
            return False
        # These values originate from the same float64 kernel values and are
        # persisted without intervening arithmetic. Exact comparison prevents
        # a fixed absolute tolerance from authenticating radically different
        # account values when initial capital or equity is very small.
        if (
            not all(
                math.isfinite(value)
                for value in (
                    summary_first_equity,
                    expected_initial_capital,
                    state_prev_equity,
                    summary_last_equity,
                )
            )
            or summary_first_equity != expected_initial_capital
            or state_prev_equity != summary_last_equity
        ):
            return False
        chronology_counts = (
            state_trades_executed,
            summary_trades_executed,
            trading_days,
            return_count,
            excess_return_count,
            positive_return_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in chronology_counts):
            return False
        if (
            state_start_date is None
            or state_last_date is None
            or summary_start_date is None
            or summary_end_date is None
            or str(state_start_date) != str(summary_start_date)
            or str(state_last_date) != str(summary_end_date)
            or state_trades_executed != summary_trades_executed
        ):
            return False
        try:
            start_timestamp = pd.Timestamp(state_start_date)
            last_timestamp = pd.Timestamp(state_last_date)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            pd.isna(start_timestamp)
            or pd.isna(last_timestamp)
            # Persisted strategy dates are emitted as canonical date-only ISO
            # strings.  Reject timestamps with times or zones before comparing
            # them so mixed aware/naive corruption cannot escape as TypeError.
            or str(state_start_date) != start_timestamp.date().isoformat()
            or str(state_last_date) != last_timestamp.date().isoformat()
            or start_timestamp.date() > last_timestamp.date()
            or trading_days < 1
            or return_count + 1 != trading_days
            or state_trades_executed > 2 * return_count
            or excess_return_count > return_count
            or positive_return_count > return_count
        ):
            return False

        return_moments_match = _persisted_centered_moments_are_consistent(
            return_count,
            summary_row[10],
            summary_row[11],
            summary_row[17],
            summary_row[18],
        )
        excess_moments_match = _persisted_centered_moments_are_consistent(
            excess_return_count,
            summary_row[13],
            summary_row[14],
            summary_row[19],
            summary_row[20],
        )
        if return_moments_match is not True or excess_moments_match is not True:
            return False

        try:
            persisted_rollup = _summary_rollup_from_row(summary_row[6:21])
            expected_metrics = None if persisted_rollup is None else _rollup_metrics(persisted_rollup)
        except (TypeError, ValueError, OverflowError):
            return False
        replay_window = (str(state_start_date), str(state_last_date))
        if replay_window not in canonical_replays:
            canonical_replays[replay_window] = _canonical_strategy_grid_replay(
                conn,
                asset_symbol=asset_symbol,
                signal_symbol=signal_symbol,
                config_pairs=sorted(expected_pairs),
                base_cfg=base_cfg,
                start_date=replay_window[0],
                end_date=replay_window[1],
                rsi_entry_rule=rsi_entry_rule,
            )
        canonical_replay = canonical_replays[replay_window]
        expected_replay = None if canonical_replay is None else canonical_replay.get(config_pair)
        if (
            expected_replay is None
            or not _strategy_states_exactly_match(state, expected_replay[0])
            or persisted_rollup != expected_replay[1]
        ):
            return False
        if expected_metrics is None or any(
            not _persisted_derived_metric_matches(stored_value, expected_metrics[metric_name])
            for stored_value, metric_name in zip(
                summary_row[21:],
                (
                    "total_return",
                    "cagr",
                    "annualized_vol",
                    "sharpe",
                    "kelly_fraction",
                    "hit_rate",
                ),
                strict=True,
            )
        ):
            return False
        if return_moments_match is True and not _strategy_return_moments_respect_lower_bound(
            return_count,
            float(summary_row[10]),
            float(summary_row[11]),
        ):
            return False
        if (
            return_count == 1
            and return_moments_match is True
            and not _single_return_rollup_is_consistent(
                first_equity=float(summary_row[6]),
                last_equity=float(summary_row[7]),
                return_sum=float(summary_row[10]),
                return_sum_squares=float(summary_row[11]),
                return_mean=float(summary_row[17]),
                return_m2=float(summary_row[18]),
                positive_return_count=positive_return_count,
            )
        ):
            return False
        if return_moments_match is True and not _return_rollup_equity_endpoints_are_consistent(
            count=return_count,
            first_equity=float(summary_row[6]),
            last_equity=float(summary_row[7]),
            return_sum=float(summary_row[10]),
            return_sum_squares=float(summary_row[11]),
            return_mean=float(summary_row[17]),
            return_m2=float(summary_row[18]),
        ):
            return False
        if return_moments_match is True and not _fully_determined_return_rollup_is_consistent(
            count=return_count,
            first_equity=float(summary_row[6]),
            last_equity=float(summary_row[7]),
            running_max_equity=float(summary_row[8]),
            return_sum=float(summary_row[10]),
            return_sum_squares=float(summary_row[11]),
            return_mean=float(summary_row[17]),
            return_m2=float(summary_row[18]),
            positive_return_count=positive_return_count,
            max_drawdown=float(summary_row[16]),
        ):
            return False

        if state_trades_executed == 0:
            try:
                state_cash = float(state_row[5])
                state_prev_equity = float(state_row[6])
                zero_trade_equities = (
                    state_cash,
                    float(summary_row[6]),
                    float(summary_row[7]),
                    float(summary_row[8]),
                )
                zero_trade_strategy_values = (
                    float(summary_row[10]),
                    float(summary_row[11]),
                    float(summary_row[16]),
                    float(summary_row[17]),
                    float(summary_row[18]),
                )
            except (TypeError, ValueError, OverflowError):
                # Let the optimized-state validator retain its specific error
                # for malformed scalar values.  This preflight rejects the
                # otherwise coherent zero-trade contradiction below.
                pass
            else:
                finite_zero_trade_values = (
                    state_prev_equity,
                    *zero_trade_equities,
                    *zero_trade_strategy_values,
                )
                if all(math.isfinite(value) for value in finite_zero_trade_values) and (
                    not all(value == state_prev_equity for value in zero_trade_equities)
                    or positive_return_count != 0
                    or not all(value == 0.0 for value in zero_trade_strategy_values)
                ):
                    return False

        # State and summary dates can be corrupted together and still agree with
        # each other.  Verify that their claimed observation count describes the
        # actual persisted asset sessions in that inclusive window before a
        # compact rollup is allowed to resume.  Rows outside the window are
        # deliberately ignored: a strategy may legitimately begin after older
        # saved history or have an unprocessed market-data tail after a crash.
        session_window = (str(state_start_date), str(state_last_date))
        chronology = (*session_window, int(trading_days), int(return_count))
        # One grid invocation advances every configuration over the same asset
        # rows.  A lone config with a shorter, internally consistent window is
        # therefore corruption rather than a valid partial resume.
        if shared_chronology is None:
            shared_chronology = chronology
        elif chronology != shared_chronology:
            return False
        if session_window not in session_windows:
            persisted_window = conn.execute(
                """
                SELECT MIN(date), MAX(date), COUNT(*)
                FROM market_data
                WHERE symbol = ? AND date BETWEEN ? AND ?
                """,
                (asset_symbol, *session_window),
            ).fetchone()
            session_windows[session_window] = (
                None if persisted_window[0] is None else str(persisted_window[0]),
                None if persisted_window[1] is None else str(persisted_window[1]),
                int(persisted_window[2]),
            )
        persisted_start, persisted_end, persisted_count = session_windows[session_window]
        if (
            persisted_start != session_window[0]
            or persisted_end != session_window[1]
            or persisted_count != trading_days
        ):
            return False
    return True


def strategy_state_matches_config(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    base_cfg: BacktestConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rsi_entry_rule: str = "lower",
) -> bool:
    """Whether persisted state exactly matches the requested simulation setup."""
    rsi_entry_rule_code(rsi_entry_rule)
    normalized_buy_rsi, normalized_profit_targets = _validated_strategy_grid_inputs(
        base_cfg,
        buy_rsi_values,
        profit_target_values,
    )
    expected_pairs = _strategy_config_pairs(
        list(normalized_buy_rsi),
        list(normalized_profit_targets),
    )
    if not expected_pairs:
        return False

    fingerprint = strategy_config_fingerprint(
        base_cfg,
        list(normalized_buy_rsi),
        list(normalized_profit_targets),
        rsi_entry_rule,
    )
    if not strategy_config_matches_fingerprint(conn, asset_symbol, signal_symbol, fingerprint):
        return False

    return _strategy_rows_match_config(
        conn,
        asset_symbol,
        signal_symbol,
        expected_pairs,
        base_cfg=base_cfg,
        rsi_entry_rule=rsi_entry_rule,
    )


def strategy_config_matches_fingerprint(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    fingerprint: str,
) -> bool:
    config_rows = conn.execute(
        """
        SELECT fingerprint
        FROM strategy_config
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchall()
    return bool(len(config_rows) == 1 and type(config_rows[0][0]) is str and config_rows[0][0] == fingerprint)


def save_strategy_config(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    fingerprint: str,
) -> bool:
    conn.execute(
        """
        INSERT INTO strategy_config (asset_symbol, signal_symbol, fingerprint)
        VALUES (?, ?, ?)
        ON CONFLICT(asset_symbol, signal_symbol) DO UPDATE SET fingerprint = excluded.fingerprint
        """,
        (asset_symbol, signal_symbol, fingerprint),
    )


def _market_values_differ(
    existing: object,
    incoming: object,
    field: str,
) -> bool:
    """Return whether one persisted OHLCV field has materially changed.

    Every price field participates in exact downstream calculations or strategy
    branches. Even an adjacent representation can move a high/open across a
    resting limit, alter an RSI threshold through close history, or change
    equity. Preserve exact correction semantics regardless of provider.
    Volume is not consumed by the simulator and retains its historical small
    representation tolerance.
    """
    if existing is None or pd.isna(existing):
        return incoming is not None and not pd.isna(incoming)
    if incoming is None or pd.isna(incoming):
        return True
    existing_value = float(existing)
    incoming_value = float(incoming)
    if existing_value == incoming_value:
        return False
    if not math.isfinite(existing_value) or not math.isfinite(incoming_value):
        return True
    if field != "Volume":
        return True
    return abs(existing_value - incoming_value) > 1e-12 + 1e-12 * abs(incoming_value)


def _revised_market_symbols(
    conn: sqlite3.Connection,
    data: pd.DataFrame,
    symbols: list[str],
) -> set[str]:
    """Detect corrections to an already persisted session before overwriting it."""
    if data.empty:
        return set()

    revised_symbols: set[str] = set()
    for symbol in symbols:
        columns = [f"{symbol}_{field}" for field in _MARKET_DATA_FIELDS]
        if any(column not in data.columns for column in columns):
            continue
        # An outer-joined multi-calendar frame uses an all-null row to mean
        # that this symbol had no session.  Such a row is not a correction to
        # an existing candle and is omitted by ``save_market_data`` as well.
        symbol_data = data.loc[~data.loc[:, columns].isna().all(axis=1)]
        for date, row in symbol_data.iterrows():
            date_str = _date_str(date)
            existing = conn.execute(
                """
                SELECT open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND date = ?
                """,
                (symbol, date_str),
            ).fetchone()
            if existing is None:
                continue
            incoming = [row.get(f"{symbol}_{field}") for field in _MARKET_DATA_FIELDS]
            if any(
                _market_values_differ(
                    old,
                    new,
                    field,
                )
                for field, old, new in zip(
                    _MARKET_DATA_FIELDS,
                    existing,
                    incoming,
                    strict=True,
                )
            ):
                revised_symbols.add(symbol)
                break
    return revised_symbols


def validate_market_data_frame(
    data: pd.DataFrame,
    symbol: str,
    *,
    source: str = "Market data",
) -> None:
    """Reject malformed OHLCV rows before they can reach persisted state."""
    columns = [f"{symbol}_{field}" for field in _MARKET_DATA_FIELDS]
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise AssetMarketDataError(f"{source} for {symbol} is missing required columns: {missing}.")
    if data.empty:
        return

    if pd.api.types.is_numeric_dtype(data.index.dtype):
        raise AssetMarketDataError(f"{source} for {symbol} must use calendar dates as its daily index.")
    for session_label in data.index:
        semantic_label = session_label
        while isinstance(semantic_label, np.ndarray) and semantic_label.ndim == 0:
            semantic_label = semantic_label[()]
        if isinstance(semantic_label, (Number, np.bool_)):
            raise AssetMarketDataError(f"{source} for {symbol} must use calendar dates as its daily index.")
    try:
        session_index = pd.DatetimeIndex(pd.to_datetime(data.index, errors="raise")).tz_localize(None)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssetMarketDataError(f"{source} for {symbol} has an invalid daily session index.") from exc
    if session_index.hasnans:
        raise AssetMarketDataError(f"{source} for {symbol} daily session index must not contain missing dates.")
    if not session_index.equals(session_index.normalize()):
        raise AssetMarketDataError(f"{source} for {symbol} daily session index must contain date-only labels.")
    if session_index.has_duplicates:
        duplicate = session_index[session_index.duplicated()][0].date().isoformat()
        raise AssetMarketDataError(f"{source} for {symbol} daily session index contains duplicate date {duplicate}.")
    if not session_index.is_monotonic_increasing:
        raise AssetMarketDataError(f"{source} for {symbol} daily session index must be sorted in increasing order.")

    numeric = pd.DataFrame(index=data.index)
    for field, column in zip(_MARKET_DATA_FIELDS, columns, strict=True):
        source_values = data[column]
        if getattr(source_values.dtype, "kind", None) in {"M", "m"}:
            invalid_temporal: tuple[int, object] | None = (0, source_values.iloc[0])
        else:
            invalid_temporal = next(
                (
                    (row_idx, value)
                    for row_idx, value in enumerate(source_values.array)
                    if isinstance(value, _MARKET_DATA_TEMPORAL_TYPES)
                ),
                None,
            )
        if invalid_temporal is not None:
            row_idx, value = invalid_temporal
            date_str = _date_str(data.index[row_idx])
            raise AssetMarketDataError(
                f"{source} for {symbol} has invalid OHLCV at {date_str}: "
                f"{field} must be numeric, not a date, datetime, or timedelta (got {value!r})."
            )
        invalid_numeric = next(
            (
                (row_idx, value)
                for row_idx, value in enumerate(source_values.array)
                if isinstance(value, (bool, np.bool_, complex, np.complexfloating))
            ),
            None,
        )
        if invalid_numeric is not None:
            row_idx, value = invalid_numeric
            date = _date_str(data.index[row_idx])
            raise AssetMarketDataError(
                f"{source} for {symbol} has invalid OHLCV at {date}: "
                f"{field} must be a real numeric value, not boolean or complex (got {value!r})."
            )
        numeric[field] = pd.to_numeric(source_values, errors="coerce")

    values = numeric.to_numpy(dtype=np.float64)
    ohlc = values[:, :4]
    volume = values[:, 4]
    minimum_ohlc = -100.0 if symbol == RISK_FREE_SYMBOL else 0.0
    invalid_ohlc = (~np.isfinite(ohlc)) | (ohlc <= minimum_ohlc)
    if invalid_ohlc.any():
        row_idx, field_idx = np.argwhere(invalid_ohlc)[0]
        field = _MARKET_DATA_FIELDS[int(field_idx)]
        value = data.iloc[int(row_idx)][f"{symbol}_{field}"]
        date = _date_str(data.index[int(row_idx)])
        requirement = "greater than -100 and finite" if symbol == RISK_FREE_SYMBOL else "positive and finite"
        raise AssetMarketDataError(
            f"{source} for {symbol} has invalid OHLCV at {date}: {field} must be {requirement} (got {value!r})."
        )

    invalid_volume = (~np.isfinite(volume)) | (volume < 0.0)
    if invalid_volume.any():
        row_idx = int(np.flatnonzero(invalid_volume)[0])
        value = data.iloc[row_idx][f"{symbol}_Volume"]
        date = _date_str(data.index[row_idx])
        raise AssetMarketDataError(
            f"{source} for {symbol} has invalid OHLCV at {date}: "
            f"Volume must be non-negative and finite (got {value!r})."
        )

    open_values, high_values, low_values, close_values = (ohlc[:, idx] for idx in range(4))
    invalid_high = high_values < np.maximum.reduce([open_values, close_values, low_values])
    if invalid_high.any():
        row_idx = int(np.flatnonzero(invalid_high)[0])
        date = _date_str(data.index[row_idx])
        raise AssetMarketDataError(
            f"{source} for {symbol} has invalid OHLCV at {date}: High must be at least Open, Close, and Low."
        )

    invalid_low = low_values > np.minimum.reduce([open_values, close_values, high_values])
    if invalid_low.any():
        row_idx = int(np.flatnonzero(invalid_low)[0])
        date = _date_str(data.index[row_idx])
        raise AssetMarketDataError(
            f"{source} for {symbol} has invalid OHLCV at {date}: Low must be at most Open, Close, and High."
        )


def save_market_data(conn: sqlite3.Connection, data: pd.DataFrame, symbols: list[str]) -> None:
    rows = []
    for symbol in symbols:
        columns = [f"{symbol}_{field}" for field in _MARKET_DATA_FIELDS]
        missing_columns = [column for column in columns if column not in data.columns]
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise AssetMarketDataError(f"Persisted market data for {symbol} is missing required columns: {missing}.")
        # A merged multi-calendar frame can represent an absent session with a
        # wholly empty symbol row. Omit that row; partially populated or
        # malformed candles still fail validation.
        symbol_data = data.loc[~data.loc[:, columns].isna().all(axis=1)]
        validate_market_data_frame(symbol_data, symbol, source="Persisted market data")
        for date, row in symbol_data.iterrows():
            rows.append(
                (
                    symbol,
                    _date_str(date),
                    float(row.get(f"{symbol}_Open")),
                    float(row.get(f"{symbol}_High")),
                    float(row.get(f"{symbol}_Low")),
                    float(row.get(f"{symbol}_Close")),
                    float(row.get(f"{symbol}_Volume")),
                )
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO market_data
        (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _market_history_values(
    data: pd.DataFrame,
    symbol: str,
) -> dict[str, tuple[object, ...]]:
    columns = [f"{symbol}_{field}" for field in _MARKET_DATA_FIELDS]
    values_by_date: dict[str, tuple[object, ...]] = {}
    for date, values in zip(
        data.index,
        data.loc[:, columns].itertuples(index=False, name=None),
        strict=True,
    ):
        values_by_date[_date_str(date)] = tuple(float(value) for value in values)
    return values_by_date


def _synchronize_market_data_history(
    conn: sqlite3.Connection,
    data: pd.DataFrame,
    symbol: str,
    *,
    observation_run_id: str | None = None,
    confirm_boundary_removals: bool = True,
) -> bool:
    """Persist a complete single-symbol history and remove vanished sessions.

    This helper is deliberately used only with provider histories that are
    known to be complete.  Tail updates are not authoritative: treating those
    as complete would erase otherwise valid older bars.
    """
    if data.empty:
        return False

    validate_market_data_frame(data, symbol, source="Authoritative history")

    existing_by_date = {
        str(row[0]): row[1:]
        for row in conn.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE symbol = ?
            """,
            (symbol,),
        ).fetchall()
    }
    incoming_by_date = _market_history_values(data, symbol)
    existing_dates = set(existing_by_date)
    incoming_dates = set(incoming_by_date)
    changed_dates = {
        date
        for date in existing_dates.intersection(incoming_dates)
        if any(
            _market_values_differ(
                existing,
                incoming,
                field,
            )
            for field, existing, incoming in zip(
                _MARKET_DATA_FIELDS,
                existing_by_date[date],
                incoming_by_date[date],
                strict=True,
            )
        )
    }
    new_dates = incoming_dates.difference(existing_dates)
    removed_dates = existing_dates.difference(incoming_dates)
    # Interior gaps are authoritative because the response brackets them. A
    # provider can transiently return a shortened complete-history range,
    # though, so boundary removals require the same missing set twice across
    # committed runs before deletion.
    if incoming_dates:
        incoming_start = min(incoming_dates)
        incoming_end = max(incoming_dates)
        interior_removed_dates = {date for date in removed_dates if incoming_start < date < incoming_end}
        boundary_removed_dates = removed_dates.difference(interior_removed_dates)
    else:
        interior_removed_dates = set()
        boundary_removed_dates = set()

    confirmed_boundary_dates: set[str] = set()
    if boundary_removed_dates and not confirm_boundary_removals:
        # Pair-private signal snapshots are the exact input selected for one
        # strategy. Retaining dates from a prior provider/calendar would merge
        # incompatible RSI histories, so replace their boundary immediately.
        confirmed_boundary_dates = boundary_removed_dates
        conn.execute("DELETE FROM market_history_removal_candidates WHERE symbol = ?", (symbol,))
    elif boundary_removed_dates:
        fingerprint = hashlib.sha256("\0".join(sorted(boundary_removed_dates)).encode("utf-8")).hexdigest()
        candidate = conn.execute(
            """
            SELECT missing_dates_fingerprint, missing_date_count, consecutive_observations,
                   last_observed_run_id
            FROM market_history_removal_candidates
            WHERE symbol = ?
            """,
            (symbol,),
        ).fetchone()
        repeated = bool(
            candidate is not None
            and str(candidate[0]) == fingerprint
            and int(candidate[1]) == len(boundary_removed_dates)
        )
        repeated_in_same_run = bool(
            repeated
            and observation_run_id is not None
            and candidate[3] is not None
            and str(candidate[3]) == observation_run_id
        )
        observations = int(candidate[2]) if repeated_in_same_run else int(candidate[2]) + 1 if repeated else 1
        if observations >= 2:
            confirmed_boundary_dates = boundary_removed_dates
            conn.execute("DELETE FROM market_history_removal_candidates WHERE symbol = ?", (symbol,))
        elif not repeated_in_same_run:
            conn.execute(
                """
                INSERT INTO market_history_removal_candidates
                    (symbol, missing_dates_fingerprint, missing_date_count,
                     consecutive_observations, last_observed_run_id,
                     first_observed_at, last_observed_at)
                VALUES (?, ?, ?, 1, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(symbol) DO UPDATE SET
                    missing_dates_fingerprint = excluded.missing_dates_fingerprint,
                    missing_date_count = excluded.missing_date_count,
                    consecutive_observations = 1,
                    last_observed_run_id = excluded.last_observed_run_id,
                    first_observed_at = CURRENT_TIMESTAMP,
                    last_observed_at = CURRENT_TIMESTAMP
                """,
                (symbol, fingerprint, len(boundary_removed_dates), observation_run_id),
            )
    else:
        conn.execute("DELETE FROM market_history_removal_candidates WHERE symbol = ?", (symbol,))

    confirmed_removed_dates = interior_removed_dates.union(confirmed_boundary_dates)
    # A new tail date is a normal incremental update.  A newly discovered date
    # at or before the prior tail changes historical inputs and needs replay.
    prior_tail = max(existing_dates) if existing_dates else None
    historical_additions = {date for date in new_dates if prior_tail is not None and date <= prior_tail}
    if symbol == RISK_FREE_SYMBOL and new_dates:
        # Asset strategies can run beyond the latest available ^IRX session by
        # forward-filling its last yield.  A subsequently published benchmark
        # bar is therefore historical from the strategy's perspective even
        # when it is a normal tail append to the benchmark table itself.
        latest_processed_row = conn.execute("SELECT MAX(last_date) FROM strategy_state").fetchone()
        latest_processed_date = latest_processed_row[0] if latest_processed_row else None
        if latest_processed_date is not None:
            historical_additions.update(date for date in new_dates if date <= str(latest_processed_date))
    if confirmed_removed_dates:
        conn.executemany(
            "DELETE FROM market_data WHERE symbol = ? AND date = ?",
            [(symbol, date) for date in sorted(confirmed_removed_dates)],
        )

    dates_to_write = new_dates.union(changed_dates)
    if dates_to_write:
        conn.executemany(
            """
            INSERT INTO market_data
            (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
            """,
            [(symbol, date, *values) for date, values in incoming_by_date.items() if date in dates_to_write],
        )

    return bool(changed_dates or confirmed_removed_dates or historical_additions)


def save_rsi_values(
    conn: sqlite3.Connection,
    signal_symbol: str,
    rsi_period: int,
    details: pd.DataFrame,
) -> None:
    rows = [
        (
            signal_symbol,
            rsi_period,
            _date_str(date),
            float(row["close"]),
            None if pd.isna(row["avg_gain"]) else float(row["avg_gain"]),
            None if pd.isna(row["avg_loss"]) else float(row["avg_loss"]),
            None if pd.isna(row["rsi"]) else float(row["rsi"]),
        )
        for date, row in details.iterrows()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO rsi_values
        (signal_symbol, rsi_period, date, close, avg_gain, avg_loss, rsi)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def load_rsi_series_for_dates(
    conn: sqlite3.Connection,
    signal_symbol: str,
    rsi_period: int,
    dates: pd.Index,
) -> pd.Series:
    if len(dates) == 0:
        return pd.Series(dtype=float)

    placeholders = ",".join("?" for _ in dates)
    params = [signal_symbol, rsi_period, *[_date_str(date) for date in dates]]
    df = pd.read_sql_query(
        f"""
        SELECT date, rsi
        FROM rsi_values
        WHERE signal_symbol = ?
          AND rsi_period = ?
          AND date IN ({placeholders})
        """,
        conn,
        params=params,
        parse_dates=["date"],
    )
    if df.empty:
        return pd.Series(index=dates, dtype=float)
    out = df.set_index("date")["rsi"].sort_index()
    out.index = pd.to_datetime(out.index)
    return out.reindex(pd.to_datetime(dates))


def _rsi_cache_rows_are_valid(
    rows: list[tuple],
    close: pd.Series,
    rsi_period: int,
    *,
    cache_last_date: str,
) -> bool:
    """Validate cached Wilder state through the requested observation window."""
    if not rows:
        return False

    close_dates = [_date_str(date) for date in close.index]
    close_by_date = dict(zip(close_dates, close.to_numpy(dtype=np.float64), strict=True))
    cached_dates = [str(row[0]) for row in rows]

    # Within the supplied canonical window, cached observations must form a
    # prefix.  Rows before a tail-only request are useful recurrence context;
    # rows after a prefix request are deliberately excluded by the caller.
    first_requested_date = close_dates[0]
    last_cached_date = cached_dates[-1]
    cached_overlap = [date for date in cached_dates if date >= first_requested_date]
    expected_overlap = (
        close_dates
        if cache_last_date >= close_dates[-1]
        else [date for date in close_dates if date <= last_cached_date]
    )
    if cached_overlap != expected_overlap:
        return False

    expected_avg_gain: float | None = None
    expected_avg_loss: float | None = None
    prior_close: float | None = None
    seed_gain = 0.0
    seed_loss = 0.0

    def finite_float(value: object) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return numeric if np.isfinite(numeric) else None

    def recurrence_values_match(left: float, right: float) -> bool:
        """Allow only scale-aware float roundoff in deterministic Wilder state."""
        if left == right:
            return True
        if left == 0.0 or right == 0.0:
            return False
        left_extended = np.longdouble(left)
        right_extended = np.longdouble(right)
        difference = abs(left_extended - right_extended)
        scale = max(abs(left_extended), abs(right_extended))
        left_inward = np.nextafter(np.float64(left), np.float64(0.0))
        right_inward = np.nextafter(np.float64(right), np.float64(0.0))
        ulp = max(
            abs(left_extended - np.longdouble(left_inward)),
            abs(right_extended - np.longdouble(right_inward)),
        )
        # Both bounds must hold. The ULP cap prevents accepting a materially
        # different ordinary-scale recurrence, while the relative cap prevents
        # a generous fixed number of subnormal ULPs from changing the RSI path.
        tolerance = min(
            np.longdouble(64.0 * np.finfo(np.float64).eps) * scale,
            np.longdouble(32.0) * ulp,
        )
        return bool(difference <= tolerance)

    for position, row in enumerate(rows):
        date, cached_close, cached_avg_gain, cached_avg_loss, cached_rsi = row
        numeric_close = finite_float(cached_close)
        if numeric_close is None:
            return False
        canonical_close = close_by_date.get(str(date))
        # Cached closes are persisted directly from this canonical float64
        # input. No arithmetic separates them, so even a small mismatch means
        # the recurrence belongs to a different price history.
        if canonical_close is not None and numeric_close != float(canonical_close):
            return False

        if position == 0:
            prior_close = numeric_close
        else:
            assert prior_close is not None
            delta = numeric_close - prior_close
            if not np.isfinite(delta):
                return False
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            if position <= rsi_period:
                seed_gain += (gain - seed_gain) / position
                seed_loss += (loss - seed_loss) / position
            elif expected_avg_gain is not None and expected_avg_loss is not None:
                alpha = 1.0 / rsi_period
                expected_avg_gain = (1.0 - alpha) * expected_avg_gain + alpha * gain
                expected_avg_loss = (1.0 - alpha) * expected_avg_loss + alpha * loss
                if not np.isfinite(expected_avg_gain) or not np.isfinite(expected_avg_loss):
                    return False
            prior_close = numeric_close

        if position < rsi_period:
            if any(value is not None for value in (cached_avg_gain, cached_avg_loss, cached_rsi)):
                return False
            continue
        if position == rsi_period:
            expected_avg_gain = seed_gain
            expected_avg_loss = seed_loss

        actual_avg_gain = finite_float(cached_avg_gain)
        actual_avg_loss = finite_float(cached_avg_loss)
        actual_rsi = finite_float(cached_rsi)
        if actual_avg_gain is None or actual_avg_loss is None or actual_rsi is None:
            return False
        assert expected_avg_gain is not None and expected_avg_loss is not None
        if (
            actual_avg_gain < 0.0
            or actual_avg_loss < 0.0
            or not recurrence_values_match(actual_avg_gain, expected_avg_gain)
            or not recurrence_values_match(actual_avg_loss, expected_avg_loss)
        ):
            return False
        try:
            expected_rsi = rsi_value_from_average_gain_loss(expected_avg_gain, expected_avg_loss)
        except ValueError:
            return False
        if not 0.0 <= actual_rsi <= 100.0 or not recurrence_values_match(
            actual_rsi,
            expected_rsi,
        ):
            return False

    return True


def ensure_rsi_values(
    conn: sqlite3.Connection,
    signal_symbol: str,
    rsi_period: int,
    close: pd.Series,
    rebuild: bool,
) -> pd.Series:
    close = close.dropna().sort_index()
    if close.empty:
        return pd.Series(dtype=float)

    def rebuild_values() -> pd.Series:
        conn.execute(
            "DELETE FROM rsi_values WHERE signal_symbol = ? AND rsi_period = ?",
            (signal_symbol, rsi_period),
        )
        details = compute_rsi_details(close, rsi_period)
        save_rsi_values(conn, signal_symbol, rsi_period, details)
        return details["rsi"]

    if rebuild:
        return rebuild_values()

    requested_last_date = _date_str(close.index[-1])
    all_cached_rows = conn.execute(
        """
        SELECT date, close, avg_gain, avg_loss, rsi
        FROM rsi_values
        WHERE signal_symbol = ?
          AND rsi_period = ?
        ORDER BY date
        """,
        (signal_symbol, rsi_period),
    ).fetchall()
    cached_rows = [row for row in all_cached_rows if str(row[0]) <= requested_last_date]
    if not all_cached_rows or not _rsi_cache_rows_are_valid(
        cached_rows,
        close,
        rsi_period,
        cache_last_date=str(all_cached_rows[-1][0]),
    ):
        return rebuild_values()

    last = pd.read_sql_query(
        """
        SELECT date, close, avg_gain, avg_loss
        FROM rsi_values
        WHERE signal_symbol = ? AND rsi_period = ?
        ORDER BY date DESC
        LIMIT 1
        """,
        conn,
        params=(signal_symbol, rsi_period),
    )
    if last.empty:
        return rebuild_values()

    last_date = pd.Timestamp(last.loc[0, "date"])
    new_close = close[close.index > last_date]
    if new_close.empty:
        return load_rsi_series_for_dates(conn, signal_symbol, rsi_period, close.index)

    avg_gain = last.loc[0, "avg_gain"]
    avg_loss = last.loc[0, "avg_loss"]
    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return rebuild_values()

    try:
        avg_gain = float(avg_gain)
        avg_loss = float(avg_loss)
    except (TypeError, ValueError, OverflowError):
        return rebuild_values()
    if not np.isfinite(avg_gain) or not np.isfinite(avg_loss) or avg_gain < 0.0 or avg_loss < 0.0:
        return rebuild_values()

    prev_close = float(last.loc[0, "close"])
    rows = []
    alpha = 1 / rsi_period
    for date, current_close in new_close.items():
        delta = float(current_close) - prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (1 - alpha) * avg_gain + alpha * gain
        avg_loss = (1 - alpha) * avg_loss + alpha * loss
        rsi = rsi_value_from_average_gain_loss(avg_gain, avg_loss)
        rows.append(
            (
                signal_symbol,
                rsi_period,
                _date_str(date),
                float(current_close),
                avg_gain,
                avg_loss,
                rsi,
            )
        )
        prev_close = float(current_close)

    conn.executemany(
        """
        INSERT OR REPLACE INTO rsi_values
        (signal_symbol, rsi_period, date, close, avg_gain, avg_loss, rsi)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return load_rsi_series_for_dates(conn, signal_symbol, rsi_period, close.index)


def load_strategy_state(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    *,
    rsi_period: int | None = None,
    rsi_entry_rule: str = "lower",
) -> dict | None:
    rows = conn.execute(
        """
        SELECT buy_rsi, profit_target_multiple, start_date, last_date,
               trades_executed, cash, prev_equity, shares, in_position,
               entry_price, entry_date, pending_action, integrity_digest
        FROM strategy_state
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND buy_rsi = ?
          AND profit_target_multiple = ?
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    ).fetchall()
    if len(rows) != 1:
        return None
    row = rows[0]
    if not _strategy_state_integrity_row_is_valid(
        row,
        asset_symbol=asset_symbol,
        signal_symbol=signal_symbol,
    ):
        return None
    state = _strategy_state_from_row(
        (
            row[2],
            row[3],
            row[5],
            row[7],
            row[8],
            row[9],
            row[10],
            row[11],
            row[6],
            row[4],
        )
    )
    if not _strategy_state_is_semantically_valid(conn, asset_symbol, state):
        return None
    persisted_signal_symbol = _persisted_strategy_signal_symbol(
        conn,
        asset_symbol,
        signal_symbol,
    )
    has_canonical_signal_history = (
        conn.execute(
            "SELECT 1 FROM market_data WHERE symbol = ? LIMIT 1",
            (persisted_signal_symbol,),
        ).fetchone()
        is not None
    )
    if not has_canonical_signal_history and state["pending_action"] != "none":
        return None
    if has_canonical_signal_history:
        resolved_rsi_period = _resolve_strategy_rsi_period(
            conn,
            persisted_signal_symbol,
            rsi_period,
        )
        if resolved_rsi_period is None or not _compact_state_matches_canonical_chronology(
            conn,
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            buy_rsi=buy_rsi,
            profit_target_multiple=profit_target_multiple,
            state=state,
            rsi_period=resolved_rsi_period,
            rsi_entry_rule=rsi_entry_rule,
        ):
            return None
    return state


def _resolve_strategy_rsi_period(
    conn: sqlite3.Connection,
    signal_symbol: str,
    rsi_period: int | None,
) -> int | None:
    """Resolve a direct-read period without guessing across retained caches."""
    if rsi_period is not None:
        if isinstance(rsi_period, bool) or not isinstance(rsi_period, int) or rsi_period <= 0:
            return None
        return rsi_period
    rsi_period_rows = conn.execute(
        """
        SELECT DISTINCT rsi_period
        FROM rsi_values
        WHERE signal_symbol = ?
        """,
        (signal_symbol,),
    ).fetchall()
    if len(rsi_period_rows) != 1 or type(rsi_period_rows[0][0]) is not int:
        return None
    resolved = int(rsi_period_rows[0][0])
    return resolved if resolved > 0 else None


def _strategy_state_from_row(row: tuple) -> dict:
    return {
        "start_date": row[0],
        "last_date": row[1],
        "cash": row[2],
        "shares": row[3],
        "in_position": bool(row[4]),
        "entry_price": row[5],
        "entry_date": row[6],
        "pending_action": row[7],
        "prev_equity": row[8],
        "trades_executed": row[9],
    }


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


def _strategy_state_is_semantically_valid(
    conn: sqlite3.Connection,
    asset_symbol: str,
    state: dict,
) -> bool:
    """Validate the complete account shape before resume or signal reporting."""
    try:
        start_date = pd.Timestamp(state["start_date"])
        last_date = pd.Timestamp(state["last_date"])
        cash = float(state["cash"])
        shares = float(state["shares"])
        prev_equity = float(state["prev_equity"])
        trades_executed = state["trades_executed"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if (
        pd.isna(start_date)
        or pd.isna(last_date)
        or str(state["start_date"]) != start_date.date().isoformat()
        or str(state["last_date"]) != last_date.date().isoformat()
        or start_date.date() > last_date.date()
        or not all(math.isfinite(value) for value in (cash, shares, prev_equity))
        or cash < 0.0
        or shares < 0.0
        or prev_equity <= 0.0
        or isinstance(trades_executed, bool)
        or not isinstance(trades_executed, int)
        or trades_executed < 0
    ):
        return False

    in_position = bool(state["in_position"])
    pending_action = state["pending_action"]
    entry_price = state["entry_price"]
    entry_date = state.get("entry_date")
    entry_missing = entry_price is None or pd.isna(entry_price)
    if in_position:
        try:
            numeric_entry_price = float(entry_price)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            shares <= 0.0
            or not math.isfinite(numeric_entry_price)
            or numeric_entry_price <= 0.0
            or trades_executed % 2 != 1
            or pending_action not in {"none", "sell"}
        ):
            return False
        entry_notional = shares * numeric_entry_price
        if not math.isfinite(entry_notional) or entry_notional <= 0.0:
            return False
        account_scale = max(prev_equity, entry_notional, cash)
        cash_tolerance = _float64_roundoff_tolerance(0.0, account_scale)
        if cash > cash_tolerance:
            return False
        try:
            entry_timestamp = pd.Timestamp(entry_date)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            entry_date is None
            or pd.isna(entry_timestamp)
            or str(entry_date) != entry_timestamp.date().isoformat()
            or not start_date.date() < entry_timestamp.date() <= last_date.date()
        ):
            return False
        entry_open_rows_exist = conn.execute(
            """
            SELECT 1 FROM market_data
            WHERE symbol = ? AND date BETWEEN ? AND ?
            LIMIT 1
            """,
            (asset_symbol, str(state["start_date"]), str(state["last_date"])),
        ).fetchone()
        entry_open_row = conn.execute(
            """
            SELECT open, TYPEOF(open)
            FROM market_data
            WHERE symbol = ? AND date = ?
            LIMIT 1
            """,
            (asset_symbol, str(entry_date)),
        ).fetchone()
        if entry_open_rows_exist is not None:
            if entry_open_row is None or entry_open_row[1] not in {"integer", "real"}:
                return False
            if not _complete_curve_values_match(
                numeric_entry_price,
                entry_open_row[0],
            ):
                return False
    elif (
        shares != 0.0
        or not entry_missing
        or entry_date is not None
        or trades_executed % 2 != 0
        or pending_action not in {"none", "buy"}
        or cash != prev_equity
    ):
        return False

    close_row = conn.execute(
        "SELECT close, TYPEOF(close) FROM market_data WHERE symbol = ? AND date = ?",
        (asset_symbol, str(state["last_date"])),
    ).fetchone()
    if in_position and close_row is not None:
        if close_row[1] not in {"integer", "real"}:
            return False
        try:
            close_price = float(close_row[0])
            marked_equity = cash + shares * close_price
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not math.isfinite(close_price)
            or close_price <= 0.0
            or not math.isfinite(marked_equity)
            or not _complete_curve_values_match(prev_equity, marked_equity)
        ):
            return False
    return True


def _compact_state_matches_canonical_chronology(
    conn: sqlite3.Connection,
    *,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
    rsi_period: int,
    rsi_entry_rule: str,
    canonical_signal_rsi: pd.Series | None = None,
) -> bool:
    """Replay canonical inputs and authenticate the compact strategy chronology."""
    try:
        rsi_entry_rule_value = rsi_entry_rule_code(rsi_entry_rule)
    except (TypeError, ValueError, OverflowError):
        return False

    market_rows = conn.execute(
        """
        SELECT date, open, high, close
        FROM market_data
        WHERE symbol = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        (asset_symbol, state["start_date"], state["last_date"]),
    ).fetchall()
    if not market_rows:
        return False
    try:
        dates = pd.DatetimeIndex(pd.to_datetime([row[0] for row in market_rows], format="%Y-%m-%d"))
        open_prices = np.asarray([row[1] for row in market_rows], dtype=np.float64)
        high_prices = np.asarray([row[2] for row in market_rows], dtype=np.float64)
        close_prices = np.asarray([row[3] for row in market_rows], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return False
    signal_series = canonical_signal_rsi
    if signal_series is None:
        persisted_signal_symbol = _persisted_strategy_signal_symbol(
            conn,
            asset_symbol,
            signal_symbol,
        )
        signal_series = _canonical_saved_rsi_series(
            conn,
            persisted_signal_symbol,
            rsi_period,
        )
    if (
        signal_series is None
        or len(signal_series) == 0
        or np.isinf(signal_series.to_numpy(dtype=np.float64)).any()
        or not np.isfinite(open_prices).all()
        or not np.isfinite(high_prices).all()
        or not np.isfinite(close_prices).all()
        or np.any(open_prices <= 0.0)
        or np.any(close_prices <= 0.0)
        or np.any(high_prices < np.maximum(open_prices, close_prices))
    ):
        return False
    aligned_rsi, _observation_dates = align_signal_values_to_asset_sessions(
        signal_series,
        dates,
    )
    rsi_values = aligned_rsi.to_numpy(dtype=np.float64)
    if not np.isfinite(rsi_values).any():
        return False

    pending_action = ACTION_NONE
    in_position = False
    entry_row_idx = -1
    target_price = math.nan
    trades_executed = 0
    for row_idx in range(len(dates)):
        if pending_action == ACTION_BUY and not in_position:
            in_position = True
            entry_row_idx = row_idx
            trades_executed += 1
            try:
                target_price = _target_sell_price(
                    float(open_prices[row_idx]),
                    profit_target_multiple,
                )
            except (TypeError, ValueError, OverflowError):
                return False
        if in_position and (open_prices[row_idx] >= target_price or high_prices[row_idx] >= target_price):
            in_position = False
            entry_row_idx = -1
            target_price = math.nan
            trades_executed += 1

        next_action = ACTION_NONE
        rsi = rsi_values[row_idx]
        if np.isfinite(rsi) and not in_position:
            entry_signal = rsi >= buy_rsi if rsi_entry_rule_value == RSI_ENTRY_UPPER else rsi <= buy_rsi
            if entry_signal:
                next_action = ACTION_BUY
        pending_action = next_action

    expected_pending_action = _action_label(int(pending_action))
    if (
        bool(in_position) != bool(state.get("in_position"))
        or trades_executed != state.get("trades_executed")
        or expected_pending_action != state.get("pending_action")
    ):
        return False
    if not in_position:
        entry_price = state.get("entry_price")
        return bool(
            entry_row_idx < 0 and state.get("entry_date") is None and (entry_price is None or pd.isna(entry_price))
        )
    try:
        stored_entry_price = float(state.get("entry_price"))
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        entry_row_idx >= 0
        and str(state.get("entry_date")) == dates[entry_row_idx].date().isoformat()
        and stored_entry_price == float(open_prices[entry_row_idx])
    )


def _compact_position_entry_matches_canonical_chronology(
    conn: sqlite3.Connection,
    *,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
    rsi_period: int,
    rsi_entry_rule: str,
    canonical_signal_rsi: pd.Series | None = None,
) -> bool:
    """Backward-compatible private wrapper for the complete chronology check."""
    return bool(
        state.get("in_position")
        and _compact_state_matches_canonical_chronology(
            conn,
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            buy_rsi=buy_rsi,
            profit_target_multiple=profit_target_multiple,
            state=state,
            rsi_period=rsi_period,
            rsi_entry_rule=rsi_entry_rule,
            canonical_signal_rsi=canonical_signal_rsi,
        )
    )


def _canonical_saved_rsi_series(
    conn: sqlite3.Connection,
    signal_symbol: str,
    rsi_period: int,
) -> pd.Series | None:
    """Derive RSI from persisted market closes without trusting the RSI cache."""
    rows = conn.execute(
        """
        SELECT date, close, TYPEOF(date), TYPEOF(close)
        FROM market_data
        WHERE symbol = ?
        ORDER BY date
        """,
        (signal_symbol,),
    ).fetchall()
    if not rows:
        return None

    date_strings: list[str] = []
    close_values: list[float] = []
    for date, close, date_type, close_type in rows:
        if date_type != "text" or close_type not in {"integer", "real"}:
            return None
        try:
            timestamp = pd.Timestamp(date)
            numeric_close = float(close)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            pd.isna(timestamp)
            or str(date) != timestamp.date().isoformat()
            or not math.isfinite(numeric_close)
            or numeric_close <= 0.0
        ):
            return None
        date_strings.append(str(date))
        close_values.append(numeric_close)

    try:
        dates = pd.DatetimeIndex(pd.to_datetime(date_strings, format="%Y-%m-%d", errors="raise"))
        close_series = pd.Series(close_values, index=dates, dtype=np.float64)
        canonical_rsi = compute_rsi_details(close_series, rsi_period)["rsi"]
    except (TypeError, ValueError, OverflowError):
        return None
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        return None
    return canonical_rsi


def _load_strategy_states_for_asset(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> dict[tuple[float, float], dict]:
    rows = conn.execute(
        """
        SELECT buy_rsi, profit_target_multiple, start_date, last_date, cash,
               shares, in_position, entry_price, entry_date, pending_action, prev_equity,
               trades_executed
        FROM strategy_state
        WHERE asset_symbol = ?
          AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchall()
    identities = [(row[0], row[1]) for row in rows]
    if len(set(identities)) != len(identities):
        return {}
    return {(float(row[0]), float(row[1])): _strategy_state_from_row(row[2:]) for row in rows}


_STRATEGY_STATE_UPSERT_SQL = """
INSERT OR REPLACE INTO strategy_state
(asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, start_date,
 last_date, cash, shares, in_position, entry_price, entry_date, pending_action,
 prev_equity, trades_executed, integrity_digest)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _strategy_state_row(
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
) -> tuple:
    row = (
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
        state["start_date"],
        state["last_date"],
        state["cash"],
        state["shares"],
        int(state["in_position"]),
        None if pd.isna(state["entry_price"]) else state["entry_price"],
        state.get("entry_date"),
        state["pending_action"],
        state["prev_equity"],
        state["trades_executed"],
    )
    return (
        *row,
        _strategy_state_integrity_digest(
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            buy_rsi=buy_rsi,
            profit_target_multiple=profit_target_multiple,
            start_date=state["start_date"],
            last_date=state["last_date"],
            cash=state["cash"],
            shares=state["shares"],
            in_position=state["in_position"],
            entry_price=state["entry_price"],
            entry_date=state.get("entry_date"),
            pending_action=state["pending_action"],
            prev_equity=state["prev_equity"],
            trades_executed=state["trades_executed"],
        ),
    )


def save_strategy_state(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
) -> None:
    conn.execute(
        _STRATEGY_STATE_UPSERT_SQL,
        _strategy_state_row(
            asset_symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
            state,
        ),
    )


def save_strategy_states(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if rows:
        conn.executemany(_STRATEGY_STATE_UPSERT_SQL, rows)


def save_alpaca_managed_buy_order(
    conn: sqlite3.Connection,
    *,
    workflow: str | None = None,
    symbol: str,
    alpaca_asset_id: str | None = None,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    buy_signal_date: str,
    buy_client_order_id: str,
    buy_alpaca_order_id: str | None,
    buy_submitted_at: str | None,
    buy_status: str,
    notes: str | None = None,
    buy_order_qty: float | None = None,
    buy_order_limit_price: float | None = None,
    sell_order_namespace: str | None = None,
    require_applied: bool = False,
    require_existing: bool = False,
    expected_buy_submission_attempt_count: int | None = None,
    expected_state_revision: int | None = None,
    buy_broker_updated_at: str | None | object = _UNSET,
) -> int:
    """Persist an observed broker state for a managed buy intent.

    New submissions must use ``claim_alpaca_managed_buy_intent`` first.  This
    helper intentionally remains an upsert because it is used after Alpaca has
    authoritatively identified an existing order by client order ID. Broker
    recovery paths can require an existing durable intent, preventing an
    observed order from creating managed ownership by itself.
    """
    buy_status = _managed_lifecycle_status(
        buy_status,
        field_name="Managed Alpaca buy status",
    )
    symbol = _normalized_managed_symbol(
        symbol,
        field_name="Managed Alpaca buy symbol",
    )
    signal_symbol = _normalized_managed_symbol(
        signal_symbol,
        field_name="Managed Alpaca buy signal symbol",
    )
    alpaca_asset_id = _normalized_optional_alpaca_asset_id(
        alpaca_asset_id,
        field_name="Managed Alpaca buy asset ID",
    )
    buy_rsi, profit_target_multiple = _normalize_managed_strategy_economics(
        buy_rsi,
        profit_target_multiple,
    )
    buy_order_qty, buy_order_limit_price = _normalize_optional_managed_buy_intent(
        buy_order_qty,
        buy_order_limit_price,
    )
    broker_timestamp_provided = buy_broker_updated_at is not _UNSET
    observed_broker_updated_at = (
        _normalize_alpaca_broker_timestamp(None if buy_broker_updated_at is _UNSET else buy_broker_updated_at)
        if broker_timestamp_provided
        else None
    )
    causality_assignment, causality_assignment_params = _managed_buy_causality_quarantine_assignment_sql(
        buy_order_qty_expression=("COALESCE(alpaca_managed_positions.buy_order_qty, excluded.buy_order_qty)"),
        buy_order_limit_price_expression=(
            "COALESCE(alpaca_managed_positions.buy_order_limit_price, excluded.buy_order_limit_price)"
        ),
        buy_status_expression="excluded.buy_status",
        filled_qty_expression="alpaca_managed_positions.filled_qty",
        filled_avg_price_expression="alpaca_managed_positions.filled_avg_price",
        existing_marker_expression="alpaca_managed_positions.buy_causality_quarantine",
    )
    owns_transaction = not conn.in_transaction
    with _managed_accounting_composite_savepoint(conn, "save_alpaca_managed_buy_order"):
        cursor = conn.execute(
            f"""
        INSERT INTO alpaca_managed_positions
        (workflow, symbol, alpaca_asset_id, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
         buy_client_order_id, buy_alpaca_order_id, buy_submitted_at, buy_order_qty,
         buy_order_limit_price, sell_order_namespace,
         buy_submission_claimed_at, buy_status, buy_observation_broker_updated_at, notes)
        SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?
        WHERE ? = 0 OR EXISTS (
            SELECT 1
            FROM alpaca_managed_positions
            WHERE buy_client_order_id = ?
        )
        ON CONFLICT(buy_client_order_id) DO UPDATE SET
            alpaca_asset_id = COALESCE(alpaca_asset_id, excluded.alpaca_asset_id),
            buy_alpaca_order_id = COALESCE(excluded.buy_alpaca_order_id, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(excluded.buy_submitted_at, buy_submitted_at),
            buy_order_qty = COALESCE(buy_order_qty, excluded.buy_order_qty),
            buy_order_limit_price = COALESCE(buy_order_limit_price, excluded.buy_order_limit_price),
            buy_causality_quarantine = {causality_assignment},
            sell_order_namespace = COALESCE(sell_order_namespace, excluded.sell_order_namespace),
            buy_status = excluded.buy_status,
            buy_observation_broker_updated_at = CASE
                WHEN ? THEN excluded.buy_observation_broker_updated_at
                ELSE buy_observation_broker_updated_at
            END,
            notes = COALESCE(excluded.notes, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE alpaca_managed_positions.closed_at IS NULL
          AND (
                LOWER(alpaca_managed_positions.buy_status) != 'pending_cancel'
                OR LOWER(excluded.buy_status) IN ('filled', 'canceled', 'expired', 'rejected')
              )
          AND alpaca_managed_positions.workflow IS excluded.workflow
          AND UPPER(alpaca_managed_positions.symbol) = UPPER(excluded.symbol)
          AND UPPER(alpaca_managed_positions.signal_symbol) = UPPER(excluded.signal_symbol)
          AND alpaca_managed_positions.buy_rsi IS excluded.buy_rsi
          AND alpaca_managed_positions.profit_target_multiple IS excluded.profit_target_multiple
          AND alpaca_managed_positions.buy_signal_date = excluded.buy_signal_date
          AND (
                alpaca_managed_positions.alpaca_asset_id IS NULL
                OR excluded.alpaca_asset_id IS NULL
                OR alpaca_managed_positions.alpaca_asset_id = excluded.alpaca_asset_id
              )
          AND (
                ? IS NULL
                OR alpaca_managed_positions.buy_submission_attempt_count = ?
              )
          AND (
                ? IS NULL
                OR alpaca_managed_positions.state_revision = ?
              )
          AND (
                ? = 0
                OR alpaca_managed_positions.buy_observation_broker_updated_at IS NULL
                OR (
                    excluded.buy_observation_broker_updated_at IS NOT NULL
                    AND excluded.buy_observation_broker_updated_at
                        > alpaca_managed_positions.buy_observation_broker_updated_at
                )
              )
          AND (
                ? = 0
                OR (
                    alpaca_managed_positions.filled_qty IS NULL
                    AND alpaca_managed_positions.sell_client_order_id IS NULL
                    AND alpaca_managed_positions.buy_order_qty IS NOT NULL
                    AND excluded.buy_order_qty IS NOT NULL
                    AND alpaca_managed_positions.buy_order_qty = excluded.buy_order_qty
                    AND alpaca_managed_positions.buy_order_limit_price IS NOT NULL
                    AND excluded.buy_order_limit_price IS NOT NULL
                    AND alpaca_managed_positions.buy_order_limit_price
                        = excluded.buy_order_limit_price
                    AND (
                        alpaca_managed_positions.buy_alpaca_order_id IS NULL
                        OR excluded.buy_alpaca_order_id IS NULL
                        OR alpaca_managed_positions.buy_alpaca_order_id
                           = excluded.buy_alpaca_order_id
                    )
                )
              )
            """,
            (
                workflow,
                symbol,
                alpaca_asset_id,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                buy_signal_date,
                buy_client_order_id,
                buy_alpaca_order_id,
                buy_submitted_at,
                buy_order_qty,
                buy_order_limit_price,
                sell_order_namespace or (alpaca_exit_order_namespace(buy_client_order_id) if require_applied else None),
                buy_status,
                observed_broker_updated_at,
                notes,
                int(require_existing),
                buy_client_order_id,
                *causality_assignment_params,
                int(broker_timestamp_provided),
                expected_buy_submission_attempt_count,
                expected_buy_submission_attempt_count,
                expected_state_revision,
                expected_state_revision,
                int(broker_timestamp_provided),
                int(require_applied),
            ),
        )
        if require_applied and cursor.rowcount != 1:
            raise RuntimeError("Managed Alpaca buy observation lost its active-state persistence race.")
        row = conn.execute(
            """
            SELECT id, symbol, signal_symbol, alpaca_asset_id
            FROM alpaca_managed_positions
            WHERE buy_client_order_id = ?
            """,
            (buy_client_order_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Managed Alpaca buy position was not persisted.")
        position_id = int(row[0])
        if position_id <= 0:
            raise RuntimeError("Managed Alpaca buy position has an invalid identity.")
        _canonical_managed_symbol(
            row[1],
            field_name="Persisted managed Alpaca buy symbol",
        )
        _canonical_managed_symbol(
            row[2],
            field_name="Persisted managed Alpaca buy signal symbol",
        )
        _canonical_optional_alpaca_asset_id(
            row[3],
            field_name="Persisted managed Alpaca buy asset ID",
        )
    if owns_transaction:
        _commit_owned_transaction(conn)
    return position_id


def claim_alpaca_managed_buy_intent(
    conn: sqlite3.Connection,
    *,
    workflow: str | None = None,
    symbol: str,
    alpaca_asset_id: str | None = None,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    buy_signal_date: str,
    buy_client_order_id: str,
    buy_order_qty: float,
    buy_order_limit_price: float,
    sell_order_namespace: str | None = None,
    allow_retry_after_not_found: bool = False,
) -> AlpacaManagedBuyIntentClaim:
    """Atomically claim a managed-buy client order ID.

    Returns ``(position_id, True)`` only for the process that inserted the
    intent or reactivated a verified missing submission.  A retry is allowed
    only when the caller has already confirmed that Alpaca cannot find the
    deterministic client order ID; that no-side-effect generation adopts the
    retry's complete strategy economics atomically. Competing processes receive
    the existing ID without changing its broker state. A different active
    client ID for the same case-insensitive symbol returns a symbol-conflict
    result without borrowing that unrelated position's ID.
    """
    symbol = _normalized_managed_symbol(
        symbol,
        field_name="Managed Alpaca buy-intent symbol",
    )
    signal_symbol = _normalized_managed_symbol(
        signal_symbol,
        field_name="Managed Alpaca buy-intent signal symbol",
    )
    alpaca_asset_id = _normalized_optional_alpaca_asset_id(
        alpaca_asset_id,
        field_name="Managed Alpaca buy-intent asset ID",
    )
    buy_rsi, profit_target_multiple = _normalize_managed_strategy_economics(
        buy_rsi,
        profit_target_multiple,
    )
    normalized_buy_order_qty, normalized_buy_order_limit_price = _normalize_optional_managed_buy_intent(
        buy_order_qty,
        buy_order_limit_price,
    )
    if normalized_buy_order_qty is None or normalized_buy_order_limit_price is None:
        raise ValueError("A managed Alpaca buy-intent claim requires a complete immutable order intent.")
    buy_order_qty = normalized_buy_order_qty
    buy_order_limit_price = normalized_buy_order_limit_price
    owns_transaction = not conn.in_transaction
    operation = "managed Alpaca buy-intent claim"
    claimed_row = _execute_returning_owned_operation_step(
        conn,
        """
        INSERT INTO alpaca_managed_positions
        (workflow, symbol, alpaca_asset_id, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
         buy_client_order_id, buy_order_qty, buy_order_limit_price, sell_order_namespace,
         buy_submission_claimed_at, buy_status, notes)
        SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'submission_pending', ?
        WHERE NOT EXISTS (
            SELECT 1
            FROM alpaca_managed_positions
            WHERE closed_at IS NULL
              AND (
                    UPPER(symbol) = UPPER(?)
                    OR (? IS NOT NULL AND alpaca_asset_id = ?)
                  )
        )
        ON CONFLICT DO NOTHING
        RETURNING id, buy_submission_attempt_count, state_revision
        """,
        (
            workflow,
            symbol,
            alpaca_asset_id,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
            buy_signal_date,
            buy_client_order_id,
            buy_order_qty,
            buy_order_limit_price,
            sell_order_namespace or alpaca_exit_order_namespace(buy_client_order_id),
            "managed buy submission intent persisted before broker request",
            symbol,
            alpaca_asset_id,
            alpaca_asset_id,
        ),
        owns_transaction=owns_transaction,
        operation=operation,
        decode=lambda row: (int(row[0]), int(row[1]), int(row[2])),  # type: ignore[index]
    )
    claimed = claimed_row is not None
    if not claimed and allow_retry_after_not_found:
        claimed_row = _execute_returning_owned_operation_step(
            conn,
            """
            UPDATE alpaca_managed_positions
            SET state_revision = state_revision + 1,
                workflow = COALESCE(?, workflow),
                alpaca_asset_id = COALESCE(?, alpaca_asset_id),
                signal_symbol = ?,
                buy_rsi = ?,
                profit_target_multiple = ?,
                buy_order_qty = ?,
                buy_order_limit_price = ?,
                sell_order_namespace = ?,
                buy_status = 'submission_pending',
                buy_observation_broker_updated_at = NULL,
                buy_submission_claimed_at = CURRENT_TIMESTAMP,
                buy_submission_attempt_count = buy_submission_attempt_count + 1,
                closed_at = NULL,
                notes = 'managed buy submission retry claimed after Alpaca did not find the client order ID',
                updated_at = CURRENT_TIMESTAMP
            WHERE buy_client_order_id = ?
              AND buy_status = 'submission_not_found'
              AND buy_alpaca_order_id IS NULL
              AND filled_qty IS NULL
              AND sell_client_order_id IS NULL
              AND closed_at IS NOT NULL
              AND UPPER(symbol) = UPPER(?)
              AND buy_signal_date = ?
              AND (alpaca_asset_id IS NULL OR alpaca_asset_id = ?)
              AND NOT EXISTS (
                    SELECT 1
                    FROM alpaca_managed_positions AS other
                    WHERE other.id != alpaca_managed_positions.id
                      AND other.closed_at IS NULL
                      AND (
                            UPPER(other.symbol) = UPPER(?)
                            OR (? IS NOT NULL AND other.alpaca_asset_id = ?)
                          )
                  )
            RETURNING id, buy_submission_attempt_count, state_revision
            """,
            (
                workflow,
                alpaca_asset_id,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                buy_order_qty,
                buy_order_limit_price,
                sell_order_namespace or alpaca_exit_order_namespace(buy_client_order_id),
                buy_client_order_id,
                symbol,
                buy_signal_date,
                alpaca_asset_id,
                symbol,
                alpaca_asset_id,
                alpaca_asset_id,
            ),
            owns_transaction=owns_transaction,
            operation=operation,
            decode=lambda row: (int(row[0]), int(row[1]), int(row[2])),  # type: ignore[index]
        )
        claimed = claimed_row is not None
    _commit_owned_transaction(conn)
    # Submission owners must retain the generation returned by their own
    # write.  Selecting it after commit could observe a later retry claim and
    # accidentally give this stale owner the newer owner's fence token.
    row = claimed_row
    if row is None:
        row = conn.execute(
            """
            SELECT id, buy_submission_attempt_count, state_revision
            FROM alpaca_managed_positions
            WHERE buy_client_order_id = ?
            """,
            (buy_client_order_id,),
        ).fetchone()
    intent_conflict = False
    if not claimed:
        intent_identity_conflict = conn.execute(
            """
            SELECT 1
            FROM alpaca_managed_positions
            WHERE buy_client_order_id = ?
              AND alpaca_asset_id IS NOT NULL
              AND ? IS NOT NULL
              AND alpaca_asset_id != ?
            LIMIT 1
            """,
            (buy_client_order_id, alpaca_asset_id, alpaca_asset_id),
        ).fetchone()
        if intent_identity_conflict is not None:
            return AlpacaManagedBuyIntentClaim(
                position_id=None,
                claimed=False,
                attempt_count=0,
                state_revision=0,
                symbol_conflict=True,
            )
        conflict = conn.execute(
            """
            SELECT 1
            FROM alpaca_managed_positions
            WHERE closed_at IS NULL
              AND (
                    UPPER(symbol) = UPPER(?)
                    OR (? IS NOT NULL AND alpaca_asset_id = ?)
                  )
              AND buy_client_order_id != ?
            LIMIT 1
            """,
            (symbol, alpaca_asset_id, alpaca_asset_id, buy_client_order_id),
        ).fetchone()
        if conflict is not None:
            return AlpacaManagedBuyIntentClaim(
                position_id=None,
                claimed=False,
                attempt_count=0,
                state_revision=0,
                symbol_conflict=True,
            )
        persisted_intent = conn.execute(
            """
            SELECT workflow, symbol, alpaca_asset_id, signal_symbol, buy_rsi,
                   profit_target_multiple, buy_signal_date, buy_order_qty,
                   buy_order_limit_price
            FROM alpaca_managed_positions
            WHERE buy_client_order_id = ?
            """,
            (buy_client_order_id,),
        ).fetchone()
        if persisted_intent is not None:
            (
                persisted_workflow,
                persisted_symbol,
                persisted_asset_id,
                persisted_signal_symbol,
                persisted_buy_rsi,
                persisted_profit_target_multiple,
                persisted_buy_signal_date,
                persisted_buy_order_qty,
                persisted_buy_order_limit_price,
            ) = persisted_intent
            persisted_symbol = _canonical_managed_symbol(
                persisted_symbol,
                field_name="Persisted managed Alpaca buy-intent symbol",
            )
            persisted_signal_symbol = _canonical_managed_symbol(
                persisted_signal_symbol,
                field_name="Persisted managed Alpaca buy-intent signal symbol",
            )
            persisted_asset_id = _canonical_optional_alpaca_asset_id(
                persisted_asset_id,
                field_name="Persisted managed Alpaca buy-intent asset ID",
            )

            def optional_number_matches(persisted: object, requested: object) -> bool:
                if persisted is None or requested is None:
                    return persisted is None and requested is None
                return abs(float(persisted) - float(requested)) <= 0.00000001

            def optional_quantity_matches(persisted: object, requested: object) -> bool:
                if persisted is None or requested is None:
                    return persisted is None and requested is None
                persisted_value = float(persisted)
                requested_value = float(requested)
                persisted_price = (
                    None if persisted_buy_order_limit_price is None else float(persisted_buy_order_limit_price)
                )
                requested_price = None if buy_order_limit_price is None else float(buy_order_limit_price)
                return _managed_accounting_quantities_match(
                    persisted_value,
                    requested_value,
                    mark_prices=(persisted_price, requested_price),
                    value_scale=max(
                        abs(persisted_value * (persisted_price or 0.0)),
                        abs(requested_value * (requested_price or 0.0)),
                    ),
                )

            intent_conflict = bool(
                persisted_workflow != workflow
                or persisted_symbol != symbol
                or persisted_signal_symbol != signal_symbol
                or float(persisted_buy_rsi) != float(buy_rsi)
                or float(persisted_profit_target_multiple) != float(profit_target_multiple)
                or str(persisted_buy_signal_date) != buy_signal_date
                or not optional_quantity_matches(persisted_buy_order_qty, buy_order_qty)
                or not optional_number_matches(persisted_buy_order_limit_price, buy_order_limit_price)
                or (
                    persisted_asset_id is not None
                    and alpaca_asset_id is not None
                    and persisted_asset_id != alpaca_asset_id
                )
            )
    if row is None:
        raise RuntimeError("Managed Alpaca buy position was not persisted.")
    return AlpacaManagedBuyIntentClaim(
        position_id=int(row[0]),
        claimed=claimed,
        attempt_count=int(row[1]),
        state_revision=int(row[2]),
        intent_conflict=intent_conflict,
    )


def confirm_alpaca_managed_buy_submission(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_buy_submission_attempt_count: int,
    expected_state_revision: int,
    expected_buy_order_qty: float,
    expected_buy_order_limit_price: float,
    claimed_at: str,
) -> int | None:
    """Renew and fence an exact managed-buy generation immediately before POST.

    A missing-order reconciler may have observed an expired visibility lease
    while the owner was still performing broker safety checks. Advancing the
    state revision here creates a final ordering point: either an older closer
    wins first and this owner must not submit, or the renewed lease wins and the
    closer's observation can no longer close the row.
    """
    normalized_buy_order_qty, normalized_buy_order_limit_price = _normalize_optional_managed_buy_intent(
        expected_buy_order_qty,
        expected_buy_order_limit_price,
    )
    if normalized_buy_order_qty is None or normalized_buy_order_limit_price is None:
        raise ValueError("A managed Alpaca buy-submission confirmation requires a complete immutable intent.")
    owns_transaction = not conn.in_transaction
    row = _execute_returning_owned_operation_step(
        conn,
        """
        UPDATE alpaca_managed_positions
        SET state_revision = state_revision + 1,
            buy_submission_claimed_at = ?,
            notes = 'managed buy submission generation re-fenced immediately before broker request',
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND state_revision = ?
          AND closed_at IS NULL
          AND buy_status = 'submission_pending'
          AND buy_submission_attempt_count = ?
          AND buy_alpaca_order_id IS NULL
          AND filled_qty IS NULL
          AND sell_client_order_id IS NULL
          AND buy_order_qty = ?
          AND buy_order_limit_price = ?
        RETURNING state_revision, buy_order_qty, buy_order_limit_price
        """,
        (
            claimed_at,
            position_id,
            int(expected_state_revision),
            expected_buy_submission_attempt_count,
            normalized_buy_order_qty,
            normalized_buy_order_limit_price,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca buy-submission confirmation",
        decode=_decode_managed_buy_submission_confirmation,
    )
    _commit_owned_transaction(conn)
    return row


def adopt_alpaca_managed_buy_order_if_submission_not_found(
    conn: sqlite3.Connection,
    *,
    workflow: str | None,
    symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    buy_signal_date: str,
    buy_client_order_id: str,
    buy_alpaca_order_id: str,
    buy_submitted_at: str | None,
    buy_status: str,
    buy_order_qty: float,
    buy_order_limit_price: float,
    alpaca_asset_id: str | None = None,
) -> bool:
    """Reopen only a verified missing-submission intent that is now visible."""
    buy_status = _managed_lifecycle_status(
        buy_status,
        field_name="Recovered managed Alpaca buy status",
    )
    symbol = _normalized_managed_symbol(
        symbol,
        field_name="Recovered managed Alpaca buy symbol",
    )
    signal_symbol = _normalized_managed_symbol(
        signal_symbol,
        field_name="Recovered managed Alpaca buy signal symbol",
    )
    alpaca_asset_id = _normalized_optional_alpaca_asset_id(
        alpaca_asset_id,
        field_name="Recovered managed Alpaca buy asset ID",
    )
    buy_rsi, profit_target_multiple = _normalize_managed_strategy_economics(
        buy_rsi,
        profit_target_multiple,
    )
    normalized_buy_order_qty, normalized_buy_order_limit_price = _normalize_optional_managed_buy_intent(
        buy_order_qty,
        buy_order_limit_price,
    )
    if normalized_buy_order_qty is None or normalized_buy_order_limit_price is None:
        raise ValueError("A recovered managed Alpaca buy intent cannot be legacy NULL.")
    buy_order_qty = normalized_buy_order_qty
    buy_order_limit_price = normalized_buy_order_limit_price
    owns_transaction = not conn.in_transaction
    cursor = _execute_owned_operation_step(
        conn,
        """
        UPDATE alpaca_managed_positions
        SET buy_alpaca_order_id = ?,
            alpaca_asset_id = COALESCE(alpaca_asset_id, ?),
            buy_submitted_at = COALESCE(?, buy_submitted_at),
            buy_status = ?,
            buy_order_qty = COALESCE(buy_order_qty, ?),
            buy_order_limit_price = COALESCE(buy_order_limit_price, ?),
            closed_at = NULL,
            notes = 'strictly validated managed buy became visible after submission-not-found recovery',
            updated_at = CURRENT_TIMESTAMP
        WHERE buy_client_order_id = ?
          AND buy_status = 'submission_not_found'
          AND buy_alpaca_order_id IS NULL
          AND filled_qty IS NULL
          AND sell_client_order_id IS NULL
          AND closed_at IS NOT NULL
          AND workflow IS ?
          AND UPPER(symbol) = UPPER(?)
          AND UPPER(signal_symbol) = UPPER(?)
          AND buy_rsi IS ?
          AND profit_target_multiple IS ?
          AND buy_signal_date = ?
          AND buy_order_qty IS NOT NULL
          AND buy_order_qty = ?
          AND buy_order_limit_price IS NOT NULL
          AND buy_order_limit_price = ?
          AND (alpaca_asset_id IS NULL OR alpaca_asset_id = ?)
        """,
        (
            buy_alpaca_order_id,
            alpaca_asset_id,
            buy_submitted_at,
            buy_status,
            buy_order_qty,
            buy_order_limit_price,
            buy_client_order_id,
            workflow,
            symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
            buy_signal_date,
            buy_order_qty,
            buy_order_limit_price,
            alpaca_asset_id,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca missing-submission buy adoption",
    )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def alpaca_managed_buy_order_observation_issue(
    conn: sqlite3.Connection,
    *,
    workflow: str | None,
    symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    buy_signal_date: str,
    buy_client_order_id: str,
    buy_alpaca_order_id: str,
    buy_order_qty: float,
    buy_order_limit_price: float,
    alpaca_asset_id: str | None = None,
) -> str | None:
    """Return why a broker observation cannot safely update local intent state."""
    symbol = _normalized_managed_symbol(
        symbol,
        field_name="Observed managed Alpaca buy symbol",
    )
    signal_symbol = _normalized_managed_symbol(
        signal_symbol,
        field_name="Observed managed Alpaca buy signal symbol",
    )
    alpaca_asset_id = _normalized_optional_alpaca_asset_id(
        alpaca_asset_id,
        field_name="Observed managed Alpaca buy asset ID",
    )
    buy_rsi, profit_target_multiple = _normalize_managed_strategy_economics(
        buy_rsi,
        profit_target_multiple,
    )
    row = conn.execute(
        """
        SELECT buy_alpaca_order_id, buy_order_qty, buy_order_limit_price, closed_at,
               alpaca_asset_id, workflow, symbol, signal_symbol, buy_rsi,
               profit_target_multiple, buy_signal_date
        FROM alpaca_managed_positions
        WHERE buy_client_order_id = ?
        """,
        (buy_client_order_id,),
    ).fetchone()
    if row is None:
        return None
    (
        persisted_order_id,
        persisted_qty,
        persisted_limit_price,
        closed_at,
        persisted_asset_id,
        persisted_workflow,
        persisted_symbol,
        persisted_signal_symbol,
        persisted_buy_rsi,
        persisted_profit_target_multiple,
        persisted_buy_signal_date,
    ) = row
    persisted_symbol = _canonical_managed_symbol(
        persisted_symbol,
        field_name="Persisted observed managed Alpaca buy symbol",
    )
    persisted_signal_symbol = _canonical_managed_symbol(
        persisted_signal_symbol,
        field_name="Persisted observed managed Alpaca buy signal symbol",
    )
    persisted_asset_id = _canonical_optional_alpaca_asset_id(
        persisted_asset_id,
        field_name="Persisted observed managed Alpaca buy asset ID",
    )
    if persisted_workflow != workflow:
        return "the signal workflow conflicts with the immutable managed buy intent"
    if persisted_symbol != symbol:
        return "the signal asset symbol conflicts with the immutable managed buy intent"
    if persisted_signal_symbol != signal_symbol:
        return "the signal indicator symbol conflicts with the immutable managed buy intent"
    if float(persisted_buy_rsi) != float(buy_rsi):
        return "the signal buy RSI conflicts with the immutable managed buy intent"
    if float(persisted_profit_target_multiple) != float(profit_target_multiple):
        return "the signal profit target conflicts with the immutable managed buy intent"
    if str(persisted_buy_signal_date) != buy_signal_date:
        return "the signal date conflicts with the immutable managed buy intent"
    if closed_at is not None:
        return "the matching managed buy intent is closed and is not eligible for broker-order adoption"
    if persisted_qty is None or persisted_limit_price is None:
        return "the managed buy intent is missing its immutable original quantity or limit price"
    persisted_qty_value = float(persisted_qty)
    if not _managed_accounting_quantities_match(
        persisted_qty_value,
        buy_order_qty,
        mark_prices=(float(persisted_limit_price), buy_order_limit_price),
        value_scale=max(
            abs(persisted_qty_value * float(persisted_limit_price)),
            abs(buy_order_qty * buy_order_limit_price),
        ),
    ):
        return "the broker buy quantity conflicts with the immutable managed buy intent"
    if abs(float(persisted_limit_price) - buy_order_limit_price) > 0.00000001:
        return "the broker buy limit price conflicts with the immutable managed buy intent"
    if persisted_order_id is not None and str(persisted_order_id) != buy_alpaca_order_id:
        return "the broker buy order ID conflicts with the managed buy intent"
    if persisted_asset_id is not None and alpaca_asset_id is not None and persisted_asset_id != alpaca_asset_id:
        return "the broker asset ID conflicts with the immutable managed buy intent"
    return None


def fail_alpaca_managed_buy_submission_if_pending(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_buy_submission_attempt_count: int,
    notes: str,
) -> bool:
    """Close only the still-unsubmitted intent owned by this submit attempt."""
    owns_transaction = not conn.in_transaction
    cursor = _execute_owned_operation_step(
        conn,
        """
        UPDATE alpaca_managed_positions
        SET buy_status = 'submission_failed',
            closed_at = CURRENT_TIMESTAMP,
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND buy_status = 'submission_pending'
          AND buy_submission_attempt_count = ?
          AND buy_alpaca_order_id IS NULL
          AND closed_at IS NULL
        """,
        (notes, position_id, expected_buy_submission_attempt_count),
        owns_transaction=owns_transaction,
        operation="managed Alpaca buy-submission failure",
    )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def _optional_loaded_managed_value(value: object) -> object | None:
    return None if bool(pd.isna(value)) else value


def _validate_loaded_managed_position_intents(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject post-initialization corruption before broker reconciliation sees it."""
    intent_columns = [
        "id",
        "buy_order_qty",
        "buy_order_limit_price",
        "buy_status",
        "sell_status",
        "filled_qty",
        "filled_avg_price",
        "target_sell_price",
        "sell_order_qty",
        "sell_order_limit_price",
        "notes",
        "buy_causality_quarantine",
        "closed_sell_shortfall_reopen_pending",
    ]
    for (
        position_id,
        buy_order_qty,
        buy_order_limit_price,
        buy_status,
        sell_status,
        filled_qty,
        filled_avg_price,
        target_sell_price,
        sell_order_qty,
        sell_order_limit_price,
        notes,
        buy_causality_quarantine,
        closed_sell_shortfall_reopen_pending,
    ) in frame[intent_columns].itertuples(index=False, name=None):
        normalized_sell_status = _optional_loaded_managed_value(sell_status)
        if type(buy_status) is not str or (
            normalized_sell_status is not None and type(normalized_sell_status) is not str
        ):
            raise ValueError(
                f"Managed Alpaca position {int(position_id)} contains an invalid lifecycle status storage value."
            )
        if (
            isinstance(closed_sell_shortfall_reopen_pending, (bool, np.bool_))
            or not isinstance(closed_sell_shortfall_reopen_pending, (int, np.integer))
            or int(closed_sell_shortfall_reopen_pending) not in (0, 1)
        ):
            raise ValueError(
                f"Managed Alpaca position {int(position_id)} contains an invalid closed-sell shortfall control."
            )
        try:
            _normalize_optional_managed_buy_intent(
                _optional_loaded_managed_value(buy_order_qty),  # type: ignore[arg-type]
                _optional_loaded_managed_value(buy_order_limit_price),  # type: ignore[arg-type]
            )
            normalized_target = _optional_loaded_managed_value(target_sell_price)
            if normalized_target is not None:
                _normalize_managed_target_sell_price(normalized_target)
            _normalize_optional_managed_sell_intent(
                _optional_loaded_managed_value(sell_order_qty),  # type: ignore[arg-type]
                _optional_loaded_managed_value(sell_order_limit_price),  # type: ignore[arg-type]
            )
            _validate_managed_buy_causality_quarantine(
                buy_order_qty=_optional_loaded_managed_value(buy_order_qty),  # type: ignore[arg-type]
                buy_order_limit_price=_optional_loaded_managed_value(buy_order_limit_price),  # type: ignore[arg-type]
                buy_status=buy_status,
                filled_qty=_optional_loaded_managed_value(filled_qty),
                filled_avg_price=_optional_loaded_managed_value(filled_avg_price),
                notes=_optional_loaded_managed_value(notes),
                quarantine_marker=_optional_loaded_managed_value(buy_causality_quarantine),
            )
        except ValueError as exc:
            raise ValueError(
                f"Managed Alpaca position {int(position_id)} contains invalid durable broker intent economics."
            ) from exc
    return frame


def load_alpaca_managed_positions(conn: sqlite3.Connection, *, active_only: bool = False) -> pd.DataFrame:
    where = "WHERE closed_at IS NULL" if active_only else ""
    frame = pd.read_sql_query(
        f"""
        SELECT id, state_revision, workflow, symbol, alpaca_asset_id, signal_symbol,
               buy_rsi, profit_target_multiple,
               buy_signal_date, buy_client_order_id, buy_alpaca_order_id,
               buy_submitted_at, buy_order_qty, buy_order_limit_price,
               buy_submission_claimed_at, buy_submission_attempt_count,
               buy_status, filled_qty, filled_avg_price,
               filled_at, buy_observation_broker_updated_at,
               buy_fill_broker_updated_at, buy_fill_component_revisions,
               buy_fill_pending_observation, buy_causality_quarantine,
               buy_cancellation_alpaca_order_ids,
               target_sell_price,
               sell_order_namespace, sell_client_order_id,
               sell_alpaca_order_id, sell_submitted_at, sell_status,
               sell_expires_at, sell_renewal_count, sell_renewal_requested_at,
               sell_order_qty, sell_order_limit_price,
               sell_observation_broker_updated_at, sell_observation_filled_qty,
               sell_submission_retry_claimed_at,
               sell_filled_qty, sell_filled_avg_price, sell_filled_at,
               realized_pl, realized_pl_pct, sold_qty, sold_value, remaining_qty,
               closed_at, closed_correction_audited_at,
               closed_sell_shortfall_reopen_pending,
               notes, created_at, updated_at
        FROM alpaca_managed_positions
        {where}
        ORDER BY id
        """,
        conn,
    )
    return _validate_loaded_managed_position_intents(frame)


def load_recently_closed_alpaca_managed_positions(
    conn: sqlite3.Connection,
    *,
    audit_days: int = 30,
    audit_limit: int = 100,
) -> pd.DataFrame:
    """Load a rotating bounded set of closed positions eligible for broker re-audit."""
    if audit_days < 1:
        raise ValueError("Managed Alpaca closed-position audit window must be at least one day.")
    if audit_limit < 1:
        raise ValueError("Managed Alpaca closed-position audit limit must be at least one.")
    frame = pd.read_sql_query(
        f"""
        SELECT id, state_revision, workflow, symbol, alpaca_asset_id, signal_symbol,
               buy_rsi, profit_target_multiple,
               buy_signal_date, buy_client_order_id, buy_alpaca_order_id,
               buy_submitted_at, buy_order_qty, buy_order_limit_price,
               buy_submission_claimed_at, buy_submission_attempt_count,
               buy_status, filled_qty, filled_avg_price,
               filled_at, buy_observation_broker_updated_at,
               buy_fill_broker_updated_at, buy_fill_component_revisions,
               buy_fill_pending_observation, buy_causality_quarantine,
               buy_cancellation_alpaca_order_ids,
               target_sell_price,
               sell_order_namespace, sell_client_order_id,
               sell_alpaca_order_id, sell_submitted_at, sell_status,
               sell_expires_at, sell_renewal_count, sell_renewal_requested_at,
               sell_order_qty, sell_order_limit_price,
               sell_observation_broker_updated_at, sell_observation_filled_qty,
               sell_submission_retry_claimed_at,
               sell_filled_qty, sell_filled_avg_price, sell_filled_at,
               realized_pl, realized_pl_pct, sold_qty, sold_value, remaining_qty,
               closed_at, closed_correction_audited_at,
               closed_sell_shortfall_reopen_pending,
               notes, created_at, updated_at
        FROM alpaca_managed_positions
        WHERE closed_at IS NOT NULL
          AND (
                sell_submission_retry_claimed_at IS NOT NULL
                OR LOWER(sell_status) = 'pending_cancel'
                OR LOWER(buy_status) IN
                    ('submission_unknown', 'pending_cancel', 'incomplete_fill_metadata', 'identity_mismatch')
                OR (
                    (
                        filled_qty IS NOT NULL
                        OR {_managed_position_quantity_is_positive_sql("COALESCE(sold_qty, 0)")}
                        OR closed_correction_audited_at IS NOT NULL
                        OR (
                            buy_alpaca_order_id IS NOT NULL
                            AND LOWER(buy_status) IN ('canceled', 'done_for_day', 'expired', 'rejected')
                        )
                    )
                    AND datetime(closed_at) >= datetime('now', ?)
                )
              )
        ORDER BY (
                    sell_submission_retry_claimed_at IS NULL
                    AND LOWER(COALESCE(sell_status, '')) != 'pending_cancel'
                    AND LOWER(buy_status) NOT IN
                        ('submission_unknown', 'pending_cancel', 'incomplete_fill_metadata', 'identity_mismatch')
                 ),
                 closed_correction_audited_at IS NOT NULL,
                 closed_correction_audited_at,
                 id
        LIMIT ? + (
            SELECT COUNT(*)
            FROM alpaca_managed_positions AS retained
            WHERE retained.closed_at IS NOT NULL
              AND (
                    retained.sell_submission_retry_claimed_at IS NOT NULL
                    OR LOWER(retained.sell_status) = 'pending_cancel'
                    OR LOWER(retained.buy_status) IN
                        ('submission_unknown', 'pending_cancel', 'incomplete_fill_metadata', 'identity_mismatch')
                  )
        )
        """,
        conn,
        params=(f"-{audit_days} days", audit_limit),
    )
    return _validate_loaded_managed_position_intents(frame)


def mark_alpaca_closed_correction_audited(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_state_revision: int,
    audited_at: str | None = None,
) -> bool:
    """Advance only the rotating audit marker while the audited snapshot is current."""
    marker = audited_at or datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    owns_transaction = not conn.in_transaction
    cursor = _execute_owned_operation_step(
        conn,
        """
        UPDATE alpaca_managed_positions
        SET closed_correction_audited_at = CASE
                WHEN closed_correction_audited_at IS NULL
                     OR closed_correction_audited_at < ?
                THEN ?
                ELSE closed_correction_audited_at
            END
        WHERE id = ? AND state_revision = ? AND closed_at IS NOT NULL
          AND sell_submission_retry_claimed_at IS NULL
        """,
        (marker, marker, position_id, int(expected_state_revision)),
        owns_transaction=owns_transaction,
        operation="managed Alpaca closed-correction audit marker",
    )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def alpaca_managed_sell_fill_order_ids(conn: sqlite3.Connection, position_id: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT alpaca_order_id
        FROM alpaca_managed_sell_fills
        WHERE managed_position_id = ?
        ORDER BY alpaca_order_id
        """,
        (position_id,),
    ).fetchall()
    return [
        _canonical_alpaca_order_id(
            row[0],
            field_name="A persisted Alpaca sell fill order ID",
        )
        for row in rows
    ]


def alpaca_managed_sell_fill_observations(
    conn: sqlite3.Connection,
    position_id: int,
) -> dict[str, tuple[float, float]]:
    """Return the per-order sell ledger used to fence authoritative closure."""
    rows = conn.execute(
        """
        SELECT alpaca_order_id, filled_qty, filled_value
        FROM alpaca_managed_sell_fills
        WHERE managed_position_id = ?
        ORDER BY alpaca_order_id
        """,
        (position_id,),
    ).fetchall()
    observations: dict[str, tuple[float, float]] = {}
    for order_id, filled_qty, filled_value in rows:
        normalized_order_id = _canonical_alpaca_order_id(
            order_id,
            field_name="A persisted Alpaca sell fill order ID",
        )
        observations[normalized_order_id] = _normalize_managed_sell_fill_economics(
            filled_qty,
            filled_value,
        )
    return observations


def alpaca_managed_sell_generation_intents(
    conn: sqlite3.Connection,
    position_id: int,
) -> dict[str, tuple[float | None, float | None]]:
    """Return immutable submitted economics for every retained sell order."""
    rows = conn.execute(
        """
        SELECT alpaca_order_id, submitted_qty, submitted_limit_price
        FROM alpaca_managed_sell_fills
        WHERE managed_position_id = ?
        ORDER BY alpaca_order_id
        """,
        (position_id,),
    ).fetchall()
    intents: dict[str, tuple[float | None, float | None]] = {}
    for order_id, submitted_qty, submitted_limit_price in rows:
        normalized_order_id = _canonical_alpaca_order_id(
            order_id,
            field_name="A persisted Alpaca sell-generation order ID",
        )
        intents[normalized_order_id] = _normalize_optional_managed_sell_intent(
            submitted_qty,
            submitted_limit_price,
        )
    return intents


def _normalize_alpaca_broker_timestamp(value: str | None) -> str | None:
    """Return one lexically sortable UTC representation of broker time."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An Alpaca broker update timestamp must be a non-empty ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        normalized = parsed.astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise ValueError("An Alpaca broker update timestamp must be valid ISO-8601.") from exc
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_alpaca_closure_timestamp(value: str | None) -> str | None:
    """Normalize usable broker closure time, falling back to local transaction time."""
    try:
        normalized = _normalize_alpaca_broker_timestamp(value)
    except ValueError:
        # Closure must remain discoverable by SQLite's datetime() audit
        # window even when a malformed broker payload supplies its timestamp.
        return None
    if normalized is None:
        return None
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class _AlpacaBuyComponentObservation:
    """One order's broker revision and cumulative fill accounting."""

    broker_updated_at: str
    filled_qty: float | None = None
    filled_value: float | None = None

    @property
    def has_accounting(self) -> bool:
        return self.filled_qty is not None and self.filled_value is not None


def _normalize_alpaca_buy_component_revisions(
    revisions: dict[str, object],
) -> dict[str, _AlpacaBuyComponentObservation]:
    """Normalize one complete buy-lineage revision/accounting snapshot.

    Historical snapshots contain only ``order_id -> timestamp``. New snapshots
    additionally retain each order's cumulative quantity and value, which lets
    reconciliation prove that an unchanged ancestor did not silently change
    accounting merely because a zero-fill replacement advanced.
    """
    normalized: dict[str, _AlpacaBuyComponentObservation] = {}
    for raw_order_id, raw_observation in revisions.items():
        order_id = _canonical_alpaca_order_id(
            raw_order_id,
            field_name="An Alpaca buy component revision order ID",
        )
        if order_id in normalized:
            raise ValueError("An Alpaca buy component revision contains duplicate broker order IDs.")
        if isinstance(raw_observation, dict):
            unexpected_keys = set(raw_observation) - {
                "broker_updated_at",
                "filled_qty",
                "filled_value",
            }
            if unexpected_keys:
                raise ValueError("An Alpaca buy component observation contains unknown fields.")
            raw_updated_at = raw_observation.get("broker_updated_at")
            raw_filled_qty = raw_observation.get("filled_qty")
            raw_filled_value = raw_observation.get("filled_value")
            if (raw_filled_qty is None) != (raw_filled_value is None):
                raise ValueError("An Alpaca buy component observation requires both quantity and value.")
            filled_qty = None if raw_filled_qty is None else float(raw_filled_qty)
            filled_value = None if raw_filled_value is None else float(raw_filled_value)
            if (filled_qty is not None and (not math.isfinite(filled_qty) or filled_qty < 0)) or (
                filled_value is not None and (not math.isfinite(filled_value) or filled_value < 0)
            ):
                raise ValueError("Alpaca buy component accounting must be finite and non-negative.")
            if filled_qty is not None and filled_value is not None:
                if (filled_qty == 0) != (filled_value == 0):
                    raise ValueError("Alpaca buy component quantity and value must both be zero or both positive.")
                if filled_qty > 0:
                    unit_price = filled_value / filled_qty
                    if not math.isfinite(unit_price) or unit_price <= 0:
                        raise ValueError("A positive Alpaca buy component fill requires a finite positive unit price.")
        else:
            # Backward compatibility for persisted timestamp-only snapshots.
            raw_updated_at = raw_observation
            filled_qty = None
            filled_value = None
        updated_at = _normalize_alpaca_broker_timestamp(raw_updated_at)  # type: ignore[arg-type]
        if updated_at is None:
            raise ValueError("An Alpaca buy component revision requires a broker timestamp.")
        normalized[order_id] = _AlpacaBuyComponentObservation(
            broker_updated_at=updated_at,
            filled_qty=filled_qty,
            filled_value=filled_value,
        )
    if not normalized:
        raise ValueError("An Alpaca buy component revision snapshot cannot be empty.")
    return dict(sorted(normalized.items()))


def _reject_excessive_alpaca_buy_component_json_nesting(encoded: str) -> None:
    """Bound persisted JSON nesting independently of the interpreter parser."""
    depth = 0
    in_string = False
    escaped = False
    for character in encoded:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _ALPACA_BUY_COMPONENT_JSON_MAX_NESTING_DEPTH:
                raise ValueError(
                    "Persisted Alpaca buy component revisions exceed the supported "
                    f"JSON nesting depth of {_ALPACA_BUY_COMPONENT_JSON_MAX_NESTING_DEPTH}."
                )
        elif character in "]}":
            # JSON validity remains json.loads' responsibility. Do not allow
            # malformed leading closers to hide a later deeply nested value.
            depth = max(0, depth - 1)


def _decode_alpaca_buy_component_revisions(
    encoded: str | None,
) -> dict[str, _AlpacaBuyComponentObservation] | None:
    if encoded is None:
        return None
    if not isinstance(encoded, str):
        raise ValueError("Persisted Alpaca buy component revisions are invalid JSON.")
    _reject_excessive_alpaca_buy_component_json_nesting(encoded)

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError("Persisted Alpaca buy component revisions contain duplicate JSON keys.")
            payload[key] = value
        return payload

    try:
        payload = json.loads(encoded, object_pairs_hook=reject_duplicate_keys)
    except (TypeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Persisted Alpaca buy component revisions are invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Persisted Alpaca buy component revisions must be an object.")
    return _normalize_alpaca_buy_component_revisions(payload)


def _alpaca_buy_component_revisions_payload(
    revisions: dict[str, _AlpacaBuyComponentObservation],
) -> dict[str, object]:
    """Return a stable JSON-compatible representation of component state."""
    payload: dict[str, object] = {}
    for order_id, observation in revisions.items():
        if observation.has_accounting:
            payload[order_id] = {
                "broker_updated_at": observation.broker_updated_at,
                "filled_qty": observation.filled_qty,
                "filled_value": observation.filled_value,
            }
        else:
            payload[order_id] = observation.broker_updated_at
    return payload


def _encode_alpaca_buy_component_revisions(
    revisions: dict[str, _AlpacaBuyComponentObservation] | None,
) -> str | None:
    if revisions is None:
        return None
    return json.dumps(
        _alpaca_buy_component_revisions_payload(revisions),
        sort_keys=True,
        separators=(",", ":"),
    )


def _alpaca_buy_component_latest_revision(
    revisions: dict[str, _AlpacaBuyComponentObservation],
) -> str:
    return max(observation.broker_updated_at for observation in revisions.values())


def _alpaca_buy_component_oldest_revision(
    revisions: dict[str, _AlpacaBuyComponentObservation],
) -> str:
    return min(observation.broker_updated_at for observation in revisions.values())


def _alpaca_buy_component_accounting_is_ordered(
    *,
    persisted: dict[str, _AlpacaBuyComponentObservation] | None,
    observed: dict[str, _AlpacaBuyComponentObservation] | None,
) -> bool:
    """Require every accounting change to belong to an advanced component."""
    if observed is None:
        return False
    if persisted is None:
        return True
    for order_id, persisted_observation in persisted.items():
        observed_observation = observed.get(order_id)
        if observed_observation is None:
            return False
        if persisted_observation.has_accounting:
            if not observed_observation.has_accounting:
                return False
            if observed_observation.broker_updated_at == persisted_observation.broker_updated_at and (
                float(observed_observation.filled_qty) != float(persisted_observation.filled_qty)
                or float(observed_observation.filled_value) != float(persisted_observation.filled_value)
            ):
                return False
        elif observed_observation.has_accounting and (
            observed_observation.broker_updated_at <= persisted_observation.broker_updated_at
        ):
            # A timestamp-only legacy snapshot cannot establish the ancestor's
            # prior accounting. Seed it only from an aggregate-identical replay
            # or wait until that component itself advances.
            return False
    return True


def _validate_alpaca_buy_component_accounting_summary(
    revisions: dict[str, _AlpacaBuyComponentObservation] | None,
    *,
    filled_qty: float,
    filled_avg_price: float | None,
) -> None:
    if revisions is None or not all(observation.has_accounting for observation in revisions.values()):
        return
    component_qty = sum(float(observation.filled_qty) for observation in revisions.values())
    component_value = sum(float(observation.filled_value) for observation in revisions.values())
    aggregate_value = filled_qty * (0.0 if filled_avg_price is None else filled_avg_price)
    values_are_finite = all(
        math.isfinite(value) for value in (component_qty, component_value, filled_qty, aggregate_value)
    )
    value_tolerance = managed_value_reconciliation_tolerance(max(abs(component_value), abs(aggregate_value)))
    quantity_matches = _managed_accounting_quantities_match(
        component_qty,
        filled_qty,
        mark_prices=(filled_avg_price,),
        value_scale=max(abs(component_value), abs(aggregate_value)),
    )
    if (
        not values_are_finite
        or aggregate_value <= 0
        or not quantity_matches
        or abs(component_value - aggregate_value) > value_tolerance
    ):
        raise ValueError("Alpaca buy component accounting conflicts with the aggregate fill.")


def _alpaca_buy_component_revision_order(
    *,
    persisted: dict[str, _AlpacaBuyComponentObservation] | None,
    observed: dict[str, _AlpacaBuyComponentObservation] | None,
) -> tuple[bool, bool]:
    """Return ``(not_stale, newer)`` for two lineage snapshots.

    Every previously observed component must still be present at an equal or
    later revision. New replacement generations establish forward progress
    even while terminal ancestors retain their earlier revisions.
    """
    if observed is None:
        return False, False
    if persisted is None:
        return True, True
    for order_id, persisted_observation in persisted.items():
        observed_observation = observed.get(order_id)
        if (
            observed_observation is None
            or observed_observation.broker_updated_at < persisted_observation.broker_updated_at
        ):
            return False, False
    newer = any(
        order_id not in persisted or observation.broker_updated_at > persisted[order_id].broker_updated_at
        for order_id, observation in observed.items()
    )
    return True, newer


def alpaca_managed_buy_fill_observation_authorizes_mutation(
    *,
    persisted_broker_updated_at: str | None,
    persisted_component_revisions: str | None,
    observed_broker_updated_at: str | None,
    observed_oldest_broker_updated_at: str | None,
    observed_component_revisions: dict[str, object] | None,
) -> bool:
    """Return whether one buy-lineage snapshot can mutate persisted state.

    A complete component map supersedes the aggregate scalar watermark only
    after every observed component is ordered beyond that legacy watermark.
    Once a map is installed, every persisted component must still be present
    and at least one component must have advanced.
    """
    persisted_latest = _normalize_alpaca_broker_timestamp(persisted_broker_updated_at)
    observed_latest = _normalize_alpaca_broker_timestamp(observed_broker_updated_at)
    observed_oldest = _normalize_alpaca_broker_timestamp(observed_oldest_broker_updated_at)
    observed_revisions = (
        None
        if observed_component_revisions is None
        else _normalize_alpaca_buy_component_revisions(observed_component_revisions)
    )
    _validate_alpaca_buy_component_revision_summary(
        observed_revisions,
        latest=observed_latest,
        oldest=observed_oldest,
    )
    persisted_revisions = _decode_alpaca_buy_component_revisions(persisted_component_revisions)
    persisted_revisions_are_scalar_compatible = bool(
        persisted_revisions is None
        or persisted_latest is None
        or _alpaca_buy_component_latest_revision(persisted_revisions) >= persisted_latest
    )
    if persisted_revisions is not None and persisted_revisions_are_scalar_compatible:
        not_stale, newer = _alpaca_buy_component_revision_order(
            persisted=persisted_revisions,
            observed=observed_revisions,
        )
        return bool(
            not_stale
            and newer
            and _alpaca_buy_component_accounting_is_ordered(
                persisted=persisted_revisions,
                observed=observed_revisions,
            )
        )
    if persisted_latest is None:
        # Legacy rows without any broker revision retain the historical
        # fail-closed behavior because no older observation can be proven.
        return True
    if observed_revisions is not None:
        return observed_oldest is not None and observed_oldest > persisted_latest
    return observed_oldest is not None and observed_oldest > persisted_latest


def _validate_alpaca_buy_component_revision_summary(
    revisions: dict[str, _AlpacaBuyComponentObservation] | None,
    *,
    latest: str | None,
    oldest: str | None,
) -> None:
    if revisions is None:
        return
    if latest != _alpaca_buy_component_latest_revision(revisions) or oldest != _alpaca_buy_component_oldest_revision(
        revisions
    ):
        raise ValueError("Alpaca buy component revisions conflict with the lineage revision summary.")


def _encode_alpaca_buy_pending_observation(
    *,
    buy_status: str,
    filled_qty: float,
    filled_avg_price: float | None,
    filled_at: str | None,
    target_sell_price: float | None,
    component_revisions: dict[str, _AlpacaBuyComponentObservation],
) -> str:
    payload = {
        "buy_status": str(buy_status).lower(),
        "component_revisions": _alpaca_buy_component_revisions_payload(component_revisions),
        "filled_at": filled_at,
        "filled_avg_price": filled_avg_price,
        "filled_qty": filled_qty,
        "target_sell_price": target_sell_price,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _normalize_alpaca_order_id_list(order_ids: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_order_id in order_ids:
        order_id = _canonical_alpaca_order_id(
            raw_order_id,
            field_name="An Alpaca cancellation order ID",
        )
        if order_id in seen:
            raise ValueError("Alpaca cancellation order IDs must be unique.")
        seen.add(order_id)
        normalized.append(order_id)
    return sorted(normalized)


def _encode_alpaca_order_id_list(order_ids: list[str]) -> str | None:
    normalized = _normalize_alpaca_order_id_list(order_ids)
    return json.dumps(normalized, separators=(",", ":")) if normalized else None


def _raise_if_managed_sell_intent_replay_conflicts(
    *,
    persisted_qty: object,
    persisted_limit_price: object,
    observed_qty: float | None,
    observed_limit_price: float | None,
) -> None:
    """Reject a replay that would change one broker generation's economics."""
    normalized_persisted_qty, normalized_persisted_limit_price = _normalize_optional_managed_sell_intent(
        persisted_qty,  # type: ignore[arg-type]
        persisted_limit_price,  # type: ignore[arg-type]
    )
    if (
        observed_qty is not None and normalized_persisted_qty is not None and observed_qty != normalized_persisted_qty
    ) or (
        observed_limit_price is not None
        and normalized_persisted_limit_price is not None
        and observed_limit_price != normalized_persisted_limit_price
    ):
        raise ValueError("A managed Alpaca sell order cannot change its immutable generation economics.")


def _validate_managed_sell_generation_assignment(
    conn: sqlite3.Connection,
    position_id: int,
    alpaca_order_id: str,
    *,
    submitted_qty: float | None,
    submitted_limit_price: float | None,
    parent_intent: tuple[object, object, object] | None | object = _UNSET,
    validate_unattached_parent_intent: bool = False,
) -> None:
    """Prove global broker-order ownership and immutable generation economics."""
    parent_owner = conn.execute(
        """
        SELECT id
        FROM alpaca_managed_positions
        WHERE sell_alpaca_order_id = ? AND id != ?
        LIMIT 1
        """,
        (alpaca_order_id, position_id),
    ).fetchone()
    if parent_owner is not None:
        raise ValueError("An Alpaca sell order ID cannot belong to multiple managed positions.")
    ledger_rows = conn.execute(
        """
        SELECT managed_position_id, submitted_qty, submitted_limit_price
        FROM alpaca_managed_sell_fills
        WHERE alpaca_order_id = ?
        """,
        (alpaca_order_id,),
    ).fetchall()
    if len(ledger_rows) > 1 or any(int(row[0]) != position_id for row in ledger_rows):
        raise ValueError("An Alpaca sell order ID cannot belong to multiple managed positions.")
    if ledger_rows:
        _raise_if_managed_sell_intent_replay_conflicts(
            persisted_qty=ledger_rows[0][1],
            persisted_limit_price=ledger_rows[0][2],
            observed_qty=submitted_qty,
            observed_limit_price=submitted_limit_price,
        )

    if parent_intent is _UNSET:
        parent_intent = conn.execute(
            """
            SELECT sell_alpaca_order_id, sell_order_qty, sell_order_limit_price
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
    if parent_intent is not None and (
        parent_intent[0] == alpaca_order_id  # type: ignore[index]
        or (validate_unattached_parent_intent and parent_intent[0] is None)  # type: ignore[index]
    ):
        _raise_if_managed_sell_intent_replay_conflicts(
            persisted_qty=parent_intent[1],  # type: ignore[index]
            persisted_limit_price=parent_intent[2],  # type: ignore[index]
            observed_qty=submitted_qty,
            observed_limit_price=submitted_limit_price,
        )


def _record_alpaca_managed_sell_generation_uncommitted(
    conn: sqlite3.Connection,
    position_id: int,
    alpaca_order_id: str,
    *,
    broker_updated_at: str | None | object = _UNSET,
    submitted_qty: float | None = None,
    submitted_limit_price: float | None = None,
    commit: bool = True,
) -> None:
    """Keep a broker sell generation discoverable before it has any fills.

    Alpaca can correct a canceled order from zero filled shares to a positive
    fill after a later renewal has replaced the managed order identity.  A
    zero-valued ledger row retains every observed broker generation so the
    rotating closed-position audit can continue to fetch those old orders.
    """
    normalized_submitted_qty, normalized_submitted_limit_price = _normalize_optional_managed_sell_intent(
        submitted_qty,
        submitted_limit_price,
    )
    normalized_alpaca_order_id = _canonical_alpaca_order_id(
        alpaca_order_id,
        field_name="An Alpaca sell-generation order ID",
    )
    _validate_managed_sell_generation_assignment(
        conn,
        position_id,
        normalized_alpaca_order_id,
        submitted_qty=normalized_submitted_qty,
        submitted_limit_price=normalized_submitted_limit_price,
    )

    broker_timestamp_provided = broker_updated_at is not _UNSET
    normalized_broker_updated_at = (
        _normalize_alpaca_broker_timestamp(None if broker_updated_at is _UNSET else broker_updated_at)
        if broker_timestamp_provided
        else None
    )
    owns_transaction = not conn.in_transaction
    operation = "managed Alpaca sell-generation persistence"
    cursor = _execute_owned_operation_step(
        conn,
        """
        INSERT INTO alpaca_managed_sell_fills
        (managed_position_id, alpaca_order_id, filled_qty, filled_value,
         broker_updated_at, submitted_qty, submitted_limit_price)
        SELECT id, ?, 0, 0, ?, ?, ?
        FROM alpaca_managed_positions
        WHERE id = ?
        ON CONFLICT(managed_position_id, alpaca_order_id) DO UPDATE SET
            broker_updated_at = CASE
                WHEN excluded.broker_updated_at IS NOT NULL
                     AND alpaca_managed_sell_fills.filled_qty = 0
                     AND alpaca_managed_sell_fills.filled_value = 0
                     AND (
                          alpaca_managed_sell_fills.broker_updated_at IS NULL
                          OR excluded.broker_updated_at > alpaca_managed_sell_fills.broker_updated_at
                     )
                THEN excluded.broker_updated_at
                ELSE alpaca_managed_sell_fills.broker_updated_at
            END,
            submitted_qty = COALESCE(
                alpaca_managed_sell_fills.submitted_qty,
                excluded.submitted_qty
            ),
            submitted_limit_price = COALESCE(
                alpaca_managed_sell_fills.submitted_limit_price,
                excluded.submitted_limit_price
            )
        WHERE (
                excluded.broker_updated_at IS NOT NULL
                AND alpaca_managed_sell_fills.filled_qty = 0
                AND alpaca_managed_sell_fills.filled_value = 0
                AND (
                     alpaca_managed_sell_fills.broker_updated_at IS NULL
                     OR excluded.broker_updated_at > alpaca_managed_sell_fills.broker_updated_at
                )
              )
           OR (
                (excluded.submitted_qty IS NOT NULL OR excluded.submitted_limit_price IS NOT NULL)
                AND (
                     alpaca_managed_sell_fills.submitted_qty IS NULL
                     OR alpaca_managed_sell_fills.submitted_limit_price IS NULL
                )
              )
        """,
        (
            normalized_alpaca_order_id,
            normalized_broker_updated_at,
            normalized_submitted_qty,
            normalized_submitted_limit_price,
            position_id,
        ),
        owns_transaction=owns_transaction,
        operation=operation,
    )
    if cursor.rowcount == 1:
        # Child-table exposure is part of the managed state.  Advance the
        # parent fence for a newly discovered generation or a newer child
        # revision so a final submission CAS cannot miss ledger-only discovery.
        _execute_owned_operation_step(
            conn,
            "UPDATE alpaca_managed_positions SET state_revision = state_revision + 1 WHERE id = ?",
            (position_id,),
            owns_transaction=owns_transaction,
            operation=operation,
        )
    if cursor.rowcount == 0:
        position_exists = _execute_owned_operation_step(
            conn,
            "SELECT 1 FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
            owns_transaction=owns_transaction,
            operation=operation,
        ).fetchone()
        if position_exists is None:
            failure = ValueError(f"Managed Alpaca position {position_id} does not exist.")
            if owns_transaction:
                _rollback_owned_operation_if_active(
                    conn,
                    failure,
                    operation=operation,
                )
            raise failure
    if commit:
        _commit_owned_transaction(conn)


def record_alpaca_managed_sell_generation(
    conn: sqlite3.Connection,
    position_id: int,
    alpaca_order_id: str,
    *,
    broker_updated_at: str | None | object = _UNSET,
    submitted_qty: float | None = None,
    submitted_limit_price: float | None = None,
    commit: bool = True,
) -> None:
    """Keep one managed sell generation atomic with its parent revision."""
    with _managed_accounting_composite_savepoint(
        conn,
        "record_alpaca_managed_sell_generation",
    ):
        _record_alpaca_managed_sell_generation_uncommitted(
            conn,
            position_id,
            alpaca_order_id,
            broker_updated_at=broker_updated_at,
            submitted_qty=submitted_qty,
            submitted_limit_price=submitted_limit_price,
            commit=False,
        )
    if commit:
        _commit_owned_transaction(conn)


def quarantine_alpaca_managed_sell_stale_submission(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    expected_sell_submission_retry_claimed_at: str,
    stale_alpaca_order_ids: list[str],
    notes: str,
    observed_alpaca_asset_id: str | None = None,
    expected_state_revision: int | None = None,
) -> bool:
    """Quarantine possible duplicate exposure without replacing broker identity.

    A retry claim may still belong to a row that a concurrent authoritative
    fill just closed. Reopen that row so the symbol remains in active
    reconciliation until cancellation/exposure is proven terminal. If another
    active row already owns the symbol, SQLite's uniqueness constraint blocks
    reopening and that active row itself conservatively blocks automation.
    """
    observed_alpaca_asset_id = _normalized_optional_alpaca_asset_id(
        observed_alpaca_asset_id,
        field_name="Quarantined managed Alpaca sell asset ID",
    )
    stale_identity_predicate = ""
    stale_identity_params: tuple[object, ...] = ()
    if stale_alpaca_order_ids:
        placeholders = ", ".join("?" for _ in stale_alpaca_order_ids)
        stale_identity_predicate = f"""
              OR sell_alpaca_order_id IN ({placeholders})
        """
        stale_identity_params = tuple(stale_alpaca_order_ids)
    try:
        with _managed_accounting_composite_savepoint(
            conn,
            "quarantine_alpaca_managed_sell_stale_submission_reopen",
        ):
            cursor = conn.execute(
                f"""
            UPDATE alpaca_managed_positions
            SET sell_status = 'quantity_mismatch',
                alpaca_asset_id = COALESCE(alpaca_asset_id, ?),
                closed_at = CASE WHEN ? IS NOT NULL OR alpaca_asset_id IS NOT NULL THEN NULL ELSE closed_at END,
                sell_submission_retry_claimed_at = COALESCE(
                    sell_submission_retry_claimed_at, ?
                ),
                notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND sell_client_order_id = ?
              AND (? IS NULL OR state_revision = ?)
              AND (alpaca_asset_id IS NULL OR ? IS NULL OR alpaca_asset_id = ?)
              AND (
                    sell_submission_retry_claimed_at IS ?
                    {stale_identity_predicate}
                  )
                """,
                (
                    observed_alpaca_asset_id,
                    observed_alpaca_asset_id,
                    expected_sell_submission_retry_claimed_at,
                    notes,
                    position_id,
                    sell_client_order_id,
                    expected_state_revision,
                    expected_state_revision,
                    observed_alpaca_asset_id,
                    observed_alpaca_asset_id,
                    expected_sell_submission_retry_claimed_at,
                    *stale_identity_params,
                ),
            )
    except sqlite3.IntegrityError as exc:
        if getattr(exc, "__notes__", None):
            # Savepoint cleanup could not preserve the caller's transaction.
            # Do not reinterpret that fail-closed condition as the ordinary
            # active-owner collision handled by the fallback below.
            raise
        # A newer active owner for the same symbol/asset can prevent reopening
        # the closed row. Persist the retained-token diagnostic on the closed
        # row anyway so the closed-position pass keeps probing this exposure.
        owns_transaction = not conn.in_transaction
        cursor = _execute_owned_operation_step(
            conn,
            f"""
            UPDATE alpaca_managed_positions
            SET sell_status = 'quantity_mismatch',
                alpaca_asset_id = COALESCE(alpaca_asset_id, ?),
                sell_submission_retry_claimed_at = COALESCE(
                    sell_submission_retry_claimed_at, ?
                ),
                notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND sell_client_order_id = ?
              AND (? IS NULL OR state_revision = ?)
              AND (alpaca_asset_id IS NULL OR ? IS NULL OR alpaca_asset_id = ?)
              AND (
                    sell_submission_retry_claimed_at IS ?
                    {stale_identity_predicate}
                  )
            """,
            (
                observed_alpaca_asset_id,
                expected_sell_submission_retry_claimed_at,
                notes,
                position_id,
                sell_client_order_id,
                expected_state_revision,
                expected_state_revision,
                observed_alpaca_asset_id,
                observed_alpaca_asset_id,
                expected_sell_submission_retry_claimed_at,
                *stale_identity_params,
            ),
            owns_transaction=owns_transaction,
            operation="managed Alpaca stale-sell quarantine fallback",
        )
        _commit_owned_transaction(conn)
        return False
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def resolve_alpaca_managed_sell_stale_submission(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    expected_sell_submission_retry_claimed_at: str,
    stale_alpaca_order_ids: list[str],
    notes: str,
) -> bool:
    """Release a retained retry token after its attributed lineage is terminal.

    A concurrently corrected broker identity is preserved and returned to an
    observable submission state. If the stale lineage was the only identity,
    rotate the client-order generation so a later retry cannot rediscover the
    same terminal Alpaca order. Negative overfill accounting remains in the
    sticky quantity-mismatch quarantine; a balanced zero is released to an
    auditable nonsticky state.
    """
    row = conn.execute(
        """
        SELECT state_revision, sell_alpaca_order_id,
               COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)),
               sell_renewal_count, symbol, sell_order_namespace, closed_at,
               filled_qty, sold_qty, filled_avg_price, target_sell_price,
               buy_order_limit_price, sell_order_limit_price, sold_value
        FROM alpaca_managed_positions
        WHERE id = ?
          AND sell_client_order_id = ?
          AND sell_submission_retry_claimed_at IS ?
        """,
        (position_id, sell_client_order_id, expected_sell_submission_retry_claimed_at),
    ).fetchone()
    if row is None:
        return False
    state_revision = int(row[0])
    current_order_id = None if row[1] is None else str(row[1])
    remaining_qty = float(row[2])
    renewal_count = int(row[3])
    symbol = _canonical_managed_symbol(
        row[4],
        field_name="Persisted managed Alpaca sell-resolution symbol",
    )
    order_namespace = str(row[5] or position_id)
    position_is_closed = row[6] is not None
    filled_qty = float(row[7] or 0.0)
    sold_qty = float(row[8] or 0.0)
    sold_value = float(row[13] or 0.0)
    sold_unit_price = sold_value / sold_qty if sold_qty > 0.0 else None
    residual_is_negligible = _managed_accounting_residual_is_negligible(
        remaining_qty,
        quantity_scale=max(abs(remaining_qty), abs(filled_qty), abs(sold_qty)),
        mark_prices=(row[9], row[10], row[11], row[12], sold_unit_price),
        value_scale=sold_value,
    )
    safe_symbol = "".join(character if character.isalnum() or character == "-" else "-" for character in symbol)
    replacement_sell_client_order_id = f"rsi-exit-{safe_symbol}-{order_namespace}-r{renewal_count + 1}"
    stale_ids = set(stale_alpaca_order_ids)
    rotate_generation = bool(
        remaining_qty > 0.0
        and not residual_is_negligible
        and not position_is_closed
        and (current_order_id is None or current_order_id in stale_ids)
    )
    resolved_status = (
        "quantity_mismatch"
        if remaining_qty < 0.0 and not residual_is_negligible
        else "submission_unknown"
        if residual_is_negligible
        else "quantity_mismatch"
        if position_is_closed
        else "submission_not_found"
        if rotate_generation
        else "submission_unknown"
    )
    owns_transaction = not conn.in_transaction
    cursor = _execute_owned_operation_step(
        conn,
        """
        UPDATE alpaca_managed_positions
        SET sell_client_order_id = CASE WHEN ? THEN ? ELSE sell_client_order_id END,
            sell_alpaca_order_id = CASE WHEN ? THEN NULL ELSE sell_alpaca_order_id END,
            sell_submitted_at = CASE WHEN ? THEN NULL ELSE sell_submitted_at END,
            sell_expires_at = CASE WHEN ? THEN NULL ELSE sell_expires_at END,
            sell_observation_broker_updated_at = CASE
                WHEN ? THEN NULL ELSE sell_observation_broker_updated_at END,
            sell_observation_filled_qty = CASE
                WHEN ? THEN NULL ELSE sell_observation_filled_qty END,
            sell_status = ?,
            sell_renewal_count = CASE WHEN ? THEN ? ELSE sell_renewal_count END,
            sell_renewal_requested_at = NULL,
            sell_order_qty = CASE WHEN ? THEN ? ELSE sell_order_qty END,
            sell_submission_retry_claimed_at = NULL,
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND state_revision = ?
          AND sell_client_order_id = ?
          AND sell_submission_retry_claimed_at IS ?
        """,
        (
            1 if rotate_generation else 0,
            replacement_sell_client_order_id,
            1 if rotate_generation else 0,
            1 if rotate_generation else 0,
            1 if rotate_generation else 0,
            1 if rotate_generation else 0,
            1 if rotate_generation else 0,
            resolved_status,
            1 if rotate_generation else 0,
            renewal_count + 1,
            1 if rotate_generation else 0,
            remaining_qty,
            notes,
            position_id,
            state_revision,
            sell_client_order_id,
            expected_sell_submission_retry_claimed_at,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca stale-sell resolution",
    )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def apply_alpaca_closed_position_broker_correction(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_closed_at: str,
    expected_state_revision: int,
    alpaca_asset_id: str,
    buy_order_qty: float,
    buy_order_limit_price: float,
    buy_status: str,
    filled_qty: float,
    filled_avg_price: float | None,
    filled_at: str | None,
    target_sell_price: float | None,
    sell_status: str | None,
    sell_fills: list[tuple[str, float, float]],
    sell_filled_at: str | None,
    notes: str,
    sell_order_intents: dict[str, tuple[float, float]] | None = None,
    sell_renewal_requested_at: str | None = None,
    buy_fill_broker_updated_at: str | None | object = _UNSET,
    buy_fill_broker_oldest_updated_at: str | None | object = _UNSET,
    buy_fill_component_revisions: dict[str, object] | None | object = _UNSET,
    sell_fill_broker_updated_at: dict[str, str | None] | None = None,
    current_sell_leaf_alpaca_order_id: str | None | object = _UNSET,
    current_sell_leaf_status: str | None | object = _UNSET,
    current_sell_lineage_alpaca_order_ids: list[str] | None = None,
    sell_cancellation_alpaca_order_ids: list[str] | None = None,
    closed_sell_shortfall_reopen_pending: bool = False,
    force_reopen_buy: bool = False,
    force_reopen_sell: bool = False,
    force_reopen: bool = False,
) -> tuple[bool, bool, bool, float]:
    """Apply authoritative broker corrections to a recently closed position.

    Returns ``(applied, reopened, active_conflict, remaining_qty)``.  Quantity
    mismatches reopen the row so normal protective reconciliation sees it on
    the next pass. Side-specific force flags retain cancellation ownership for
    balanced rows without allowing a buy or unrelated historical sell revision
    to authorize current-sell mutations. ``force_reopen`` remains a legacy
    sell-side alias for callers that do not provide broker revision maps. If a
    newer active row owns the same asset, accounting is still corrected but the
    old row remains closed and the caller reports a required manual conflict.
    """
    if type(closed_sell_shortfall_reopen_pending) is not bool:
        raise ValueError("A closed-sell shortfall reopen control must be a boolean.")
    buy_status = _managed_lifecycle_status(
        buy_status,
        field_name="Corrected managed Alpaca buy status",
    )
    sell_status = _optional_managed_lifecycle_status(
        sell_status,
        field_name="Corrected managed Alpaca sell status",
    )
    observed_asset_id = _normalized_optional_alpaca_asset_id(
        alpaca_asset_id,
        field_name="A corrected Alpaca position asset ID",
    )
    if observed_asset_id is None:
        raise ValueError("A corrected Alpaca position requires an Alpaca asset ID.")
    if target_sell_price is not None:
        target_sell_price = _normalize_managed_target_sell_price(target_sell_price)
    immutable_buy_qty, immutable_buy_limit_price = _normalize_optional_managed_buy_intent(
        buy_order_qty,
        buy_order_limit_price,
    )
    if immutable_buy_qty is None or immutable_buy_limit_price is None:
        raise ValueError("A corrected managed Alpaca buy intent cannot be legacy NULL.")
    expected_revision = int(expected_state_revision)
    if expected_revision < 0:
        raise ValueError("Managed Alpaca state revision must be non-negative.")

    observed_buy_qty, buy_price = _normalize_optional_managed_buy_fill_economics(
        filled_qty,
        filled_avg_price,
    )
    if observed_buy_qty is None:
        raise ValueError("A corrected Alpaca buy fill requires a quantity observation.")
    buy_causality_issue = _managed_buy_fill_causality_issue(
        buy_order_qty=immutable_buy_qty,
        buy_order_limit_price=immutable_buy_limit_price,
        buy_status=buy_status,
        filled_qty=observed_buy_qty,
        filled_avg_price=buy_price,
    )
    buy_status, notes, buy_causality_note = _quarantine_managed_buy_causality_issue(
        buy_status=buy_status,
        notes=notes,
        issue=buy_causality_issue,
    )
    buy_timestamp_provided = buy_fill_broker_updated_at is not _UNSET
    observed_buy_broker_updated_at = (
        _normalize_alpaca_broker_timestamp(None if buy_fill_broker_updated_at is _UNSET else buy_fill_broker_updated_at)
        if buy_timestamp_provided
        else None
    )
    if buy_fill_broker_oldest_updated_at is not _UNSET and not buy_timestamp_provided:
        raise ValueError("A closed buy's oldest broker revision requires its latest broker revision.")
    observed_buy_oldest_broker_updated_at = (
        observed_buy_broker_updated_at
        if buy_fill_broker_oldest_updated_at is _UNSET
        else _normalize_alpaca_broker_timestamp(buy_fill_broker_oldest_updated_at)
    )
    if (
        observed_buy_oldest_broker_updated_at is not None
        and observed_buy_broker_updated_at is not None
        and observed_buy_oldest_broker_updated_at > observed_buy_broker_updated_at
    ):
        raise ValueError("A closed buy's oldest broker revision cannot follow its latest revision.")
    buy_component_revisions_provided = buy_fill_component_revisions is not _UNSET
    observed_buy_component_revisions = (
        None
        if buy_fill_component_revisions is _UNSET or buy_fill_component_revisions is None
        else _normalize_alpaca_buy_component_revisions(buy_fill_component_revisions)
    )
    _validate_alpaca_buy_component_revision_summary(
        observed_buy_component_revisions,
        latest=observed_buy_broker_updated_at,
        oldest=observed_buy_oldest_broker_updated_at,
    )
    _validate_alpaca_buy_component_accounting_summary(
        observed_buy_component_revisions,
        filled_qty=observed_buy_qty,
        filled_avg_price=buy_price,
    )
    observed_buy_component_revisions_json = _encode_alpaca_buy_component_revisions(observed_buy_component_revisions)
    observed_buy_pending_observation_json = (
        None
        if observed_buy_component_revisions is None
        else _encode_alpaca_buy_pending_observation(
            buy_status=buy_status,
            filled_qty=observed_buy_qty,
            filled_avg_price=buy_price,
            filled_at=filled_at,
            target_sell_price=(None if target_sell_price is None else float(target_sell_price)),
            component_revisions=observed_buy_component_revisions,
        )
    )

    sell_parent_identity_provided = current_sell_leaf_alpaca_order_id is not _UNSET
    observed_current_sell_leaf_id = (
        None
        if current_sell_leaf_alpaca_order_id is _UNSET or current_sell_leaf_alpaca_order_id is None
        else _canonical_alpaca_order_id(
            current_sell_leaf_alpaca_order_id,
            field_name="A closed correction's current sell leaf ID",
        )
    )
    sell_parent_status_provided = current_sell_leaf_status is not _UNSET
    observed_current_sell_leaf_status = (
        None
        if current_sell_leaf_status is _UNSET or current_sell_leaf_status is None
        else str(current_sell_leaf_status).strip().lower()
    )
    if sell_parent_status_provided and current_sell_leaf_status is not None and not (observed_current_sell_leaf_status):
        raise ValueError("A closed correction's current sell leaf status must be non-empty.")
    cancellation_order_ids = set(_normalize_alpaca_order_id_list(sell_cancellation_alpaca_order_ids or []))
    current_sell_lineage_order_ids = set(_normalize_alpaca_order_id_list(current_sell_lineage_alpaca_order_ids or []))
    if observed_current_sell_leaf_id is not None:
        current_sell_lineage_order_ids.add(observed_current_sell_leaf_id)
    effective_force_reopen_sell = bool(force_reopen_sell or force_reopen)

    normalized_sell_fill_timestamps: dict[str, str | None] | None = None
    if sell_fill_broker_updated_at is not None:
        normalized_sell_fill_timestamps = {}
        for raw_order_id, timestamp in sell_fill_broker_updated_at.items():
            order_id = _canonical_alpaca_order_id(
                raw_order_id,
                field_name="A corrected Alpaca sell fill timestamp order ID",
            )
            normalized_sell_fill_timestamps[order_id] = timestamp

    normalized_fills: list[tuple[str, float, float, str | None]] = []
    seen_fill_order_ids: set[str] = set()
    for raw_order_id, qty, value in sell_fills:
        order_id = _canonical_alpaca_order_id(
            raw_order_id,
            field_name="A corrected Alpaca sell fill order ID",
        )
        if order_id in seen_fill_order_ids:
            raise ValueError("Corrected Alpaca sell fill order IDs must be unique.")
        seen_fill_order_ids.add(order_id)
        observed_qty = float(qty)
        observed_value = float(value)
        if (
            observed_qty < 0
            or not math.isfinite(observed_qty)
            or observed_value < 0
            or not math.isfinite(observed_value)
        ):
            raise ValueError("Corrected Alpaca sell fills must have valid identities, quantities, and values.")
        if (observed_qty == 0) != (observed_value == 0):
            raise ValueError("Corrected Alpaca sell fill quantity and value must both be zero or both positive.")
        if observed_qty > 0:
            unit_price = observed_value / observed_qty
            if not math.isfinite(unit_price) or unit_price <= 0:
                raise ValueError("Corrected Alpaca sell fills require a finite positive unit price.")
        # Keep every broker order identity, including fills corrected to zero.
        # Closed-position audits use this ledger to rediscover sell generations
        # that are no longer reachable from the current replacement chain.
        broker_updated_at = _normalize_alpaca_broker_timestamp(
            None if normalized_sell_fill_timestamps is None else normalized_sell_fill_timestamps.get(order_id)
        )
        normalized_fills.append((order_id, observed_qty, observed_value, broker_updated_at))

    normalized_sell_order_intents: dict[str, tuple[float, float]] = {}
    for raw_order_id, intent in (sell_order_intents or {}).items():
        normalized_order_id = _canonical_alpaca_order_id(
            raw_order_id,
            field_name="A corrected Alpaca sell intent order ID",
        )
        if not isinstance(intent, tuple) or len(intent) != 2:
            raise ValueError("Corrected Alpaca sell intents require an order ID, quantity, and limit price.")
        submitted_qty, submitted_limit_price = _normalize_optional_managed_sell_intent(
            intent[0],
            intent[1],
        )
        if submitted_qty is None or submitted_limit_price is None:
            raise ValueError("Corrected Alpaca sell intents require a quantity and limit price.")
        normalized_sell_order_intents[normalized_order_id] = (
            submitted_qty,
            submitted_limit_price,
        )
    normalized_sell_renewal_requested_at = _normalize_alpaca_broker_timestamp(sell_renewal_requested_at)

    audited_at = datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

    savepoint = "apply_alpaca_closed_position_broker_correction"
    owns_transaction = not conn.in_transaction
    savepoint_active = False
    try:
        if owns_transaction:
            conn.execute("BEGIN")
        conn.execute(f"SAVEPOINT {savepoint}")
        savepoint_active = True
        fence = conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET state_revision = state_revision + 1
            WHERE id = ? AND closed_at = ? AND state_revision = ?
            """,
            (position_id, expected_closed_at, expected_revision),
        )
        if fence.rowcount != 1:
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            savepoint_active = False
            if owns_transaction:
                _commit_owned_transaction(conn)
            return False, False, False, 0.0

        identity = conn.execute(
            """
            SELECT symbol, alpaca_asset_id, buy_order_qty, buy_order_limit_price,
                   buy_alpaca_order_id, buy_status, filled_qty, filled_avg_price, filled_at,
                   target_sell_price, buy_observation_broker_updated_at,
                   buy_fill_broker_updated_at,
                   buy_fill_component_revisions,
                   buy_fill_pending_observation,
                   buy_causality_quarantine,
                   sell_status, sell_filled_at, sell_alpaca_order_id, notes
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
        if identity is None:
            raise ValueError(f"Managed Alpaca position {position_id} does not exist.")
        (
            symbol,
            persisted_asset_id,
            persisted_buy_qty,
            persisted_buy_limit_price,
            persisted_buy_alpaca_order_id,
            persisted_buy_status,
            persisted_filled_qty,
            persisted_filled_avg_price,
            persisted_filled_at,
            persisted_target_sell_price,
            persisted_buy_observation_broker_updated_at,
            persisted_buy_broker_updated_at,
            persisted_buy_component_revisions_json,
            persisted_buy_pending_observation_json,
            persisted_buy_causality_quarantine,
            persisted_sell_status,
            persisted_sell_filled_at,
            persisted_sell_alpaca_order_id,
            persisted_notes,
        ) = identity
        symbol = _canonical_managed_symbol(
            symbol,
            field_name="Persisted corrected managed Alpaca position symbol",
        )
        persisted_asset_id = _canonical_optional_alpaca_asset_id(
            persisted_asset_id,
            field_name="Persisted corrected managed Alpaca position asset ID",
        )
        persisted_buy_observation_broker_updated_at = _normalize_alpaca_broker_timestamp(
            None
            if persisted_buy_observation_broker_updated_at is None
            else str(persisted_buy_observation_broker_updated_at)
        )
        if persisted_asset_id is not None and str(persisted_asset_id) != observed_asset_id:
            raise ValueError("Corrected Alpaca asset ID conflicts with immutable managed state.")
        persisted_buy_causality_issue = _managed_buy_fill_causality_issue(
            buy_order_qty=(immutable_buy_qty if persisted_buy_qty is None else persisted_buy_qty),
            buy_order_limit_price=(
                immutable_buy_limit_price if persisted_buy_limit_price is None else persisted_buy_limit_price
            ),
            buy_status=buy_status,
            filled_qty=observed_buy_qty,
            filled_avg_price=buy_price,
        )
        buy_status, notes, buy_causality_note = _quarantine_managed_buy_causality_issue(
            buy_status=buy_status,
            notes=notes,
            issue=persisted_buy_causality_issue,
        )
        if persisted_buy_qty is not None:
            persisted_buy_qty_value = float(persisted_buy_qty)
            if not _managed_accounting_quantities_match(
                persisted_buy_qty_value,
                immutable_buy_qty,
                mark_prices=(
                    None if persisted_buy_limit_price is None else float(persisted_buy_limit_price),
                    immutable_buy_limit_price,
                ),
                value_scale=max(
                    abs(persisted_buy_qty_value * float(persisted_buy_limit_price or 0.0)),
                    abs(immutable_buy_qty * immutable_buy_limit_price),
                ),
            ):
                raise ValueError("Corrected Alpaca buy quantity conflicts with immutable managed state.")
        if (
            persisted_buy_limit_price is not None
            and abs(float(persisted_buy_limit_price) - immutable_buy_limit_price) > 0.00000001
        ):
            raise ValueError("Corrected Alpaca buy limit price conflicts with immutable managed state.")
        effective_notes = _merge_managed_buy_causality_notes(
            None if persisted_notes is None else str(persisted_notes),
            notes,
            buy_causality_note,
        )

        # Broker-facing callers explicitly provide an aggregate replacement-
        # chain timestamp. A legacy NULL high-water mark can only be seeded by
        # an accounting-identical replay; every mutation must be ordered after
        # an already-persisted broker revision.
        effective_buy_status = buy_status
        effective_buy_qty = observed_buy_qty
        effective_buy_price = buy_price
        effective_buy_filled_at = persisted_filled_at if filled_at is None and observed_buy_qty > 0 else filled_at
        effective_target_sell_price = target_sell_price
        effective_buy_broker_updated_at = persisted_buy_broker_updated_at
        persisted_buy_component_revisions = _decode_alpaca_buy_component_revisions(
            persisted_buy_component_revisions_json
        )
        persisted_buy_component_revisions_are_scalar_compatible = bool(
            persisted_buy_component_revisions is None
            or persisted_buy_broker_updated_at is None
            or _alpaca_buy_component_latest_revision(persisted_buy_component_revisions)
            >= persisted_buy_broker_updated_at
        )
        effective_buy_component_revisions_json = persisted_buy_component_revisions_json
        effective_buy_pending_observation_json = persisted_buy_pending_observation_json
        buy_observation_is_newer = False
        buy_observation_is_stale = False
        buy_observation_matches = False
        accept_buy_observation = True
        if buy_timestamp_provided:
            persisted_observed_qty = 0.0 if persisted_filled_qty is None else float(persisted_filled_qty)
            persisted_observed_price = None if persisted_filled_avg_price is None else float(persisted_filled_avg_price)
            buy_accounting_matches = bool(
                _managed_accounting_quantities_match(
                    persisted_observed_qty,
                    observed_buy_qty,
                    mark_prices=(persisted_observed_price, buy_price),
                    value_scale=max(
                        abs(persisted_observed_qty * (persisted_observed_price or 0.0)),
                        abs(observed_buy_qty * (buy_price or 0.0)),
                    ),
                )
                and (
                    (persisted_observed_price is None and buy_price is None)
                    or (
                        persisted_observed_price is not None
                        and buy_price is not None
                        and abs(persisted_observed_price - buy_price) <= 0.00000001
                    )
                )
            )
            persisted_observed_target = (
                None if persisted_target_sell_price is None else float(persisted_target_sell_price)
            )
            buy_target_matches = bool(
                (persisted_observed_target is None and target_sell_price is None)
                or (
                    persisted_observed_target is not None
                    and target_sell_price is not None
                    and abs(persisted_observed_target - float(target_sell_price)) <= 0.00000001
                )
            )
            buy_observation_matches = bool(
                buy_accounting_matches
                and buy_target_matches
                and str(persisted_buy_status) == str(buy_status)
                and persisted_filled_at == effective_buy_filled_at
            )
            buy_has_no_persisted_observation = bool(
                persisted_filled_qty is None and persisted_buy_broker_updated_at is None
            )
            # A terminal zero-fill observation is recorded on the buy lifecycle
            # watermark even though it has no fill watermark.  Do not let a
            # delayed pre-terminal positive fill exploit that NULL and reopen a
            # closed position. Aggregate replacement-chain revisions may match
            # the lifecycle observation only when the persisted current leaf is
            # present at that revision and another order can carry the fill. A
            # single-order equal-revision payload is contradictory accounting.
            equal_revision_positive_is_replacement_aggregate = bool(
                observed_buy_qty > 0
                and persisted_buy_alpaca_order_id is not None
                and observed_buy_component_revisions is not None
                and observed_buy_broker_updated_at == persisted_buy_observation_broker_updated_at
                and (
                    persisted_leaf_observation := observed_buy_component_revisions.get(
                        str(persisted_buy_alpaca_order_id)
                    )
                )
                is not None
                and persisted_leaf_observation.broker_updated_at == persisted_buy_observation_broker_updated_at
                and any(order_id != str(persisted_buy_alpaca_order_id) for order_id in observed_buy_component_revisions)
            )
            first_positive_fill_is_lifecycle_current = bool(
                observed_buy_qty == 0
                or (persisted_filled_qty is not None and float(persisted_filled_qty) > 0)
                or persisted_buy_observation_broker_updated_at is None
                or (
                    observed_buy_broker_updated_at is not None
                    and observed_buy_broker_updated_at > persisted_buy_observation_broker_updated_at
                )
                or equal_revision_positive_is_replacement_aggregate
            )
            component_revision_not_stale = True
            component_revision_is_newer = False
            component_revision_authorizes_mutation = False
            component_revision_can_advance_summary = True
            pending_observation_matches = False
            clear_pending_observation = False
            component_restage_candidate = False
            legacy_component_seed_candidate = False
            component_revision_install_authorized = False
            if (
                buy_component_revisions_provided
                and persisted_buy_component_revisions is not None
                and persisted_buy_component_revisions_are_scalar_compatible
            ):
                component_revision_not_stale, component_revision_is_newer = _alpaca_buy_component_revision_order(
                    persisted=persisted_buy_component_revisions,
                    observed=observed_buy_component_revisions,
                )
                pending_observation_matches = bool(
                    persisted_buy_pending_observation_json is not None
                    and observed_buy_pending_observation_json == persisted_buy_pending_observation_json
                )
                pending_snapshot_ahead = bool(
                    persisted_buy_pending_observation_json is not None
                    and persisted_buy_broker_updated_at is not None
                    and _alpaca_buy_component_latest_revision(persisted_buy_component_revisions)
                    > persisted_buy_broker_updated_at
                )
                component_accounting_is_ordered = _alpaca_buy_component_accounting_is_ordered(
                    persisted=persisted_buy_component_revisions,
                    observed=observed_buy_component_revisions,
                )
                component_revision_authorizes_mutation = bool(
                    component_revision_not_stale
                    and component_accounting_is_ordered
                    and (
                        pending_snapshot_ahead and pending_observation_matches
                        if persisted_buy_pending_observation_json is not None
                        else component_revision_is_newer
                    )
                )
                component_revision_can_advance_summary = bool(
                    component_accounting_is_ordered
                    and (persisted_buy_pending_observation_json is None or pending_observation_matches)
                )
                clear_pending_observation = bool(
                    component_revision_authorizes_mutation and persisted_buy_pending_observation_json is not None
                )
                component_restage_candidate = bool(
                    persisted_buy_pending_observation_json is not None
                    and observed_buy_pending_observation_json is not None
                    and not pending_observation_matches
                    and component_revision_not_stale
                    and component_revision_is_newer
                    and component_accounting_is_ordered
                )
            elif buy_component_revisions_provided and persisted_buy_broker_updated_at is None:
                component_revision_not_stale, component_revision_is_newer = _alpaca_buy_component_revision_order(
                    persisted=None,
                    observed=observed_buy_component_revisions,
                )
                component_revision_authorizes_mutation = bool(
                    component_revision_not_stale and component_revision_is_newer
                )
            else:
                # Rows created before component snapshots retain the legacy
                # all-components-newer fence until an accounting-identical
                # observation safely seeds their per-order revision map.
                component_revision_not_stale = bool(
                    persisted_buy_broker_updated_at is None
                    or (
                        observed_buy_oldest_broker_updated_at is not None
                        and observed_buy_oldest_broker_updated_at >= persisted_buy_broker_updated_at
                    )
                )
                component_revision_is_newer = bool(
                    observed_buy_broker_updated_at is not None
                    and (
                        buy_has_no_persisted_observation
                        or (
                            persisted_buy_broker_updated_at is not None
                            and observed_buy_oldest_broker_updated_at is not None
                            and observed_buy_oldest_broker_updated_at > persisted_buy_broker_updated_at
                        )
                    )
                )
                component_revision_authorizes_mutation = bool(
                    component_revision_not_stale and component_revision_is_newer
                )
                legacy_component_seed_candidate = bool(
                    buy_component_revisions_provided
                    and (
                        persisted_buy_component_revisions is None
                        or not persisted_buy_component_revisions_are_scalar_compatible
                    )
                    and persisted_buy_broker_updated_at is not None
                    and observed_buy_component_revisions_json is not None
                    and component_revision_not_stale
                    and observed_buy_broker_updated_at is not None
                    and observed_buy_broker_updated_at > persisted_buy_broker_updated_at
                )
            component_revision_install_authorized = bool(
                buy_component_revisions_provided
                and observed_buy_component_revisions_json is not None
                and (
                    persisted_buy_component_revisions is None
                    or not persisted_buy_component_revisions_are_scalar_compatible
                )
                and component_revision_not_stale
            )
            clear_pending_observation = bool(clear_pending_observation or component_revision_install_authorized)
            buy_observation_is_newer = bool(
                observed_buy_broker_updated_at is not None and component_revision_authorizes_mutation
            )
            buy_observation_is_stale = bool(
                (not component_revision_not_stale or not first_positive_fill_is_lifecycle_current)
                and not buy_observation_matches
            )
            accept_buy_observation = bool(
                first_positive_fill_is_lifecycle_current
                and (buy_has_no_persisted_observation or buy_observation_matches or buy_observation_is_newer)
            )
            if component_restage_candidate:
                restaged = conn.execute(
                    """
                    UPDATE alpaca_managed_positions
                    SET buy_fill_component_revisions = ?,
                        buy_fill_pending_observation = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                      AND closed_at = ?
                      AND state_revision = ?
                      AND buy_fill_broker_updated_at IS ?
                      AND buy_fill_component_revisions IS ?
                      AND buy_fill_pending_observation IS ?
                    """,
                    (
                        observed_buy_component_revisions_json,
                        observed_buy_pending_observation_json,
                        position_id,
                        expected_closed_at,
                        expected_revision + 1,
                        persisted_buy_broker_updated_at,
                        persisted_buy_component_revisions_json,
                        persisted_buy_pending_observation_json,
                    ),
                )
                if restaged.rowcount == 1:
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    savepoint_active = False
                    if owns_transaction:
                        _commit_owned_transaction(conn)
                    return False, False, False, 0.0
            if not accept_buy_observation:
                effective_buy_status = str(persisted_buy_status)
                effective_buy_qty = persisted_observed_qty
                effective_buy_price = persisted_observed_price
                effective_buy_filled_at = persisted_filled_at
                effective_target_sell_price = persisted_target_sell_price
            if (
                observed_buy_broker_updated_at is not None
                and component_revision_not_stale
                and component_revision_can_advance_summary
                and (
                    effective_buy_broker_updated_at is None
                    or observed_buy_broker_updated_at > effective_buy_broker_updated_at
                )
                and accept_buy_observation
            ):
                effective_buy_broker_updated_at = observed_buy_broker_updated_at
            if (
                buy_component_revisions_provided
                and observed_buy_component_revisions_json is not None
                and component_revision_can_advance_summary
                and component_revision_not_stale
                and accept_buy_observation
            ):
                effective_buy_component_revisions_json = observed_buy_component_revisions_json
            if clear_pending_observation:
                effective_buy_pending_observation_json = None

            if not accept_buy_observation and legacy_component_seed_candidate:
                seed = conn.execute(
                    """
                    UPDATE alpaca_managed_positions
                    SET buy_fill_component_revisions = ?,
                        buy_fill_pending_observation = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                      AND closed_at = ?
                      AND state_revision = ?
                      AND buy_fill_broker_updated_at IS ?
                      AND buy_fill_component_revisions IS ?
                      AND buy_fill_pending_observation IS ?
                    """,
                    (
                        observed_buy_component_revisions_json,
                        observed_buy_pending_observation_json,
                        position_id,
                        expected_closed_at,
                        expected_revision + 1,
                        persisted_buy_broker_updated_at,
                        persisted_buy_component_revisions_json,
                        persisted_buy_pending_observation_json,
                    ),
                )
                if seed.rowcount == 1:
                    conn.execute(f"RELEASE SAVEPOINT {savepoint}")
                    savepoint_active = False
                    if owns_transaction:
                        _commit_owned_transaction(conn)
                    return False, False, False, 0.0

        effective_buy_causality_quarantine = (
            buy_causality_note if accept_buy_observation else persisted_buy_causality_quarantine
        )

        persisted_sell_rows = conn.execute(
            """
            SELECT alpaca_order_id, filled_qty, filled_value, broker_updated_at,
                   submitted_qty, submitted_limit_price
            FROM alpaca_managed_sell_fills
            WHERE managed_position_id = ?
            """,
            (position_id,),
        ).fetchall()
        for order_id, qty, value, *_rest in persisted_sell_rows:
            _canonical_alpaca_order_id(
                order_id,
                field_name="A persisted Alpaca sell fill order ID",
            )
            persisted_qty_value = float(qty)
            persisted_value_value = float(value)
            if (persisted_qty_value == 0) != (persisted_value_value == 0):
                raise ValueError("Persisted Alpaca sell fill economics are inconsistent.")
            if persisted_qty_value > 0:
                unit_price = persisted_value_value / persisted_qty_value
                if not math.isfinite(unit_price) or unit_price <= 0:
                    raise ValueError("Persisted Alpaca sell fill unit price is invalid.")
        merged_sell_fills = {
            str(order_id): (str(order_id), float(qty), float(value), broker_updated_at)
            for order_id, qty, value, broker_updated_at, _submitted_qty, _submitted_limit_price in persisted_sell_rows
        }
        merged_sell_intents: dict[str, tuple[float | None, float | None]] = {
            str(order_id): (
                None if submitted_qty is None else float(submitted_qty),
                None if submitted_limit_price is None else float(submitted_limit_price),
            )
            for order_id, _qty, _value, _broker_updated_at, submitted_qty, submitted_limit_price in persisted_sell_rows
        }
        for order_id, (submitted_qty, submitted_limit_price) in normalized_sell_order_intents.items():
            persisted_intent = merged_sell_intents.get(order_id)
            if persisted_intent is not None:
                persisted_qty, persisted_limit_price = persisted_intent
                if (
                    persisted_qty is not None
                    and not _managed_accounting_quantities_match(
                        persisted_qty,
                        submitted_qty,
                        mark_prices=(persisted_limit_price, submitted_limit_price),
                        value_scale=max(
                            abs(persisted_qty * (persisted_limit_price or 0.0)),
                            abs(submitted_qty * submitted_limit_price),
                        ),
                    )
                ) or (
                    persisted_limit_price is not None
                    and abs(persisted_limit_price - submitted_limit_price) > 0.00000001
                ):
                    raise ValueError("Corrected Alpaca sell intent conflicts with immutable managed state.")
            merged_sell_intents[order_id] = (
                submitted_qty if persisted_intent is None or persisted_intent[0] is None else persisted_intent[0],
                (
                    submitted_limit_price
                    if persisted_intent is None or persisted_intent[1] is None
                    else persisted_intent[1]
                ),
            )
        sell_timestamps_provided = sell_fill_broker_updated_at is not None
        sell_observation_is_stale = False
        sell_observation_is_newer = False
        sell_order_observation_is_newer: dict[str, bool] = {}
        sell_order_accounting_matches: dict[str, bool] = {}
        sell_order_observation_is_accepted: dict[str, bool] = {}
        for order_id, qty, value, broker_updated_at in normalized_fills:
            persisted_fill = merged_sell_fills.get(order_id)
            if persisted_fill is None or not sell_timestamps_provided:
                merged_sell_fills[order_id] = (order_id, qty, value, broker_updated_at)
                sell_order_observation_is_accepted[order_id] = True
                sell_order_accounting_matches[order_id] = False
                sell_order_observation_is_newer[order_id] = bool(
                    sell_timestamps_provided and broker_updated_at is not None
                )
                if sell_timestamps_provided and broker_updated_at is not None:
                    sell_observation_is_newer = True
                continue
            _, persisted_qty, persisted_value, persisted_broker_updated_at = persisted_fill
            value_tolerance = managed_value_reconciliation_tolerance(max(abs(persisted_value), abs(value)))
            persisted_unit_price = persisted_value / persisted_qty if persisted_qty > 0.0 else None
            observed_unit_price = value / qty if qty > 0.0 else None
            accounting_matches = bool(
                _managed_accounting_quantities_match(
                    persisted_qty,
                    qty,
                    mark_prices=(persisted_unit_price, observed_unit_price),
                    value_scale=max(abs(persisted_value), abs(value)),
                )
                and abs(persisted_value - value) <= value_tolerance
            )
            if persisted_broker_updated_at is not None and (
                broker_updated_at is None or broker_updated_at < persisted_broker_updated_at
            ):
                sell_observation_is_stale = True
                sell_order_observation_is_accepted[order_id] = False
                sell_order_accounting_matches[order_id] = accounting_matches
                sell_order_observation_is_newer[order_id] = False
                continue
            if accounting_matches:
                sell_order_observation_is_accepted[order_id] = True
                sell_order_accounting_matches[order_id] = True
                sell_order_observation_is_newer[order_id] = False
                if broker_updated_at is not None and (
                    persisted_broker_updated_at is None or broker_updated_at > persisted_broker_updated_at
                ):
                    merged_sell_fills[order_id] = (order_id, qty, value, broker_updated_at)
                    # An accounting-identical replay is the safe migration
                    # point for a legacy NULL high-water mark. Once observed,
                    # it can also authorize cancellation of that exact order
                    # in this same atomic correction.
                    sell_observation_is_newer = True
                    sell_order_observation_is_newer[order_id] = True
                continue
            if (
                broker_updated_at is not None
                and persisted_broker_updated_at is not None
                and broker_updated_at > persisted_broker_updated_at
            ):
                merged_sell_fills[order_id] = (order_id, qty, value, broker_updated_at)
                sell_observation_is_newer = True
                sell_order_observation_is_accepted[order_id] = True
                sell_order_accounting_matches[order_id] = False
                sell_order_observation_is_newer[order_id] = True
            else:
                # Equal revisions cannot carry conflicting accounting, and a
                # migrated NULL revision needs an identical replay before any
                # later correction can be ordered safely.
                sell_observation_is_stale = True
                sell_order_observation_is_accepted[order_id] = False
                sell_order_accounting_matches[order_id] = False
                sell_order_observation_is_newer[order_id] = False

        prospective_sold_qty = sum(qty for _, qty, _, _ in merged_sell_fills.values())
        if not math.isfinite(prospective_sold_qty) or prospective_sold_qty < 0:
            raise ValueError("Corrected cumulative Alpaca sell quantity must remain finite and non-negative.")
        effective_sell_filled_at = (
            persisted_sell_filled_at if sell_filled_at is None and prospective_sold_qty > 0 else sell_filled_at
        )
        current_sell_observation_is_newer = bool(
            observed_current_sell_leaf_id is not None
            and sell_order_observation_is_newer.get(observed_current_sell_leaf_id, False)
        )
        current_sell_status_matches = bool(
            sell_parent_status_provided
            and (
                (persisted_sell_alpaca_order_id is None and observed_current_sell_leaf_id is None)
                or str(persisted_sell_alpaca_order_id or "") == str(observed_current_sell_leaf_id or "")
            )
            and str(persisted_sell_status or "").lower() == str(observed_current_sell_leaf_status or "").lower()
        )
        cancellation_observations_authorized = bool(
            cancellation_order_ids
            and all(
                sell_order_observation_is_accepted.get(order_id, False)
                and (
                    sell_order_observation_is_newer.get(order_id, False)
                    or (
                        str(persisted_sell_status or "").lower() == "pending_cancel"
                        and sell_order_accounting_matches.get(order_id, False)
                    )
                )
                for order_id in cancellation_order_ids
            )
        )
        if observed_current_sell_leaf_id in cancellation_order_ids:
            cancellation_observations_authorized = bool(
                cancellation_observations_authorized
                and (
                    current_sell_status_matches
                    or current_sell_observation_is_newer
                    or (
                        str(persisted_sell_status or "").lower() == "pending_cancel"
                        and sell_order_accounting_matches.get(
                            observed_current_sell_leaf_id,
                            False,
                        )
                    )
                )
            )

        sell_status_change_requested = persisted_sell_status != sell_status
        sell_filled_at_change_requested = persisted_sell_filled_at != effective_sell_filled_at
        target_sell_status_is_current_observation = bool(
            sell_parent_status_provided
            and str(sell_status or "").lower() == str(observed_current_sell_leaf_status or "").lower()
        )
        diagnostic_sell_statuses = {
            "pending_cancel",
            "position_quantity_mismatch",
            "quantity_mismatch",
        }
        sell_status_change_authorized = bool(
            not sell_status_change_requested
            or (
                target_sell_status_is_current_observation
                and (current_sell_status_matches or current_sell_observation_is_newer)
            )
            or (str(sell_status or "").lower() == "pending_cancel" and cancellation_observations_authorized)
            or (
                str(sell_status or "").lower() in diagnostic_sell_statuses
                and str(sell_status or "").lower() != "pending_cancel"
                and (buy_observation_is_newer or sell_observation_is_newer or current_sell_observation_is_newer)
            )
        )
        sell_filled_at_change_authorized = bool(
            not sell_filled_at_change_requested
            or any(sell_order_observation_is_newer.get(order_id, False) for order_id in current_sell_lineage_order_ids)
        )
        buy_force_reopen_authorized = bool(
            not force_reopen_buy
            or not buy_timestamp_provided
            or (
                not buy_observation_is_stale
                and accept_buy_observation
                and (buy_has_no_persisted_observation or buy_observation_matches or buy_observation_is_newer)
            )
        )
        sell_force_reopen_authorized = bool(
            not effective_force_reopen_sell or not sell_timestamps_provided or cancellation_observations_authorized
        )
        sell_parent_authority_required = bool(
            sell_status_change_requested or sell_filled_at_change_requested or effective_force_reopen_sell
        )
        if (
            (buy_timestamp_provided and (buy_observation_is_stale or not accept_buy_observation))
            or sell_observation_is_stale
            or (
                sell_timestamps_provided
                and sell_parent_authority_required
                and (
                    not sell_parent_identity_provided
                    or not sell_parent_status_provided
                    or not sell_status_change_authorized
                    or not sell_filled_at_change_authorized
                    or not sell_force_reopen_authorized
                )
            )
            or not buy_force_reopen_authorized
        ):
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            savepoint_active = False
            if owns_transaction:
                _commit_owned_transaction(conn)
            return False, False, False, 0.0

        normalized_fills = sorted(merged_sell_fills.values(), key=lambda fill: fill[0])
        sold_qty = sum(qty for _, qty, _, _ in normalized_fills)
        sold_value = sum(value for _, _, value, _ in normalized_fills)
        if not math.isfinite(sold_qty) or not math.isfinite(sold_value) or sold_qty < 0 or sold_value < 0:
            raise ValueError("Corrected cumulative Alpaca sell accounting must remain finite and non-negative.")
        remaining_qty = effective_buy_qty - sold_qty
        sell_avg_price = sold_value / sold_qty if sold_qty > 0 else None
        residual_is_negligible = _managed_accounting_residual_is_negligible(
            remaining_qty,
            quantity_scale=max(abs(remaining_qty), effective_buy_qty, sold_qty),
            mark_prices=(
                effective_buy_price,
                effective_target_sell_price,
                immutable_buy_limit_price,
                sell_avg_price,
            ),
            value_scale=sold_value,
        )
        accounting_overfill = remaining_qty < 0.0 and not residual_is_negligible
        matched_qty = min(sold_qty, effective_buy_qty)
        realized_pl: float | None = None
        realized_pl_pct: float | None = None
        if not accounting_overfill and matched_qty > 0 and effective_buy_price is not None:
            realized_pl, realized_pl_pct = _managed_realized_pl_values(
                sold_value=sold_value,
                matched_qty=matched_qty,
                buy_price=effective_buy_price,
            )
        derived_accounting_values = (
            remaining_qty,
            matched_qty,
            realized_pl,
            realized_pl_pct,
            sell_avg_price,
        )
        if any(value is not None and not math.isfinite(value) for value in derived_accounting_values):
            raise ValueError("Corrected derived Alpaca sell accounting must remain finite.")

        reopen_required = bool(force_reopen_buy or effective_force_reopen_sell or not residual_is_negligible)
        active_conflict = False
        if reopen_required:
            active_conflict = (
                conn.execute(
                    """
                    SELECT 1
                    FROM alpaca_managed_positions
                    WHERE id != ? AND closed_at IS NULL
                      AND (
                            alpaca_asset_id = ?
                            OR UPPER(symbol) = UPPER(?)
                          )
                    LIMIT 1
                    """,
                    (position_id, observed_asset_id, symbol),
                ).fetchone()
                is not None
            )
        reopen = reopen_required and not active_conflict

        conn.execute(
            """
            INSERT INTO alpaca_symbol_aliases (alpaca_asset_id, symbol)
            VALUES (?, ?)
            ON CONFLICT(alpaca_asset_id, symbol) DO UPDATE SET
                last_seen_at = CURRENT_TIMESTAMP
            """,
            (observed_asset_id, symbol),
        )

        for order_id, _qty, _value, _broker_updated_at in normalized_fills:
            _validate_managed_sell_generation_assignment(
                conn,
                position_id,
                order_id,
                submitted_qty=None,
                submitted_limit_price=None,
            )

        conn.execute(
            "DELETE FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
            (position_id,),
        )
        if normalized_fills:
            conn.executemany(
                """
                INSERT INTO alpaca_managed_sell_fills
                (managed_position_id, alpaca_order_id, filled_qty, filled_value,
                 broker_updated_at, submitted_qty, submitted_limit_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        position_id,
                        order_id,
                        qty,
                        value,
                        broker_updated_at,
                        merged_sell_intents.get(order_id, (None, None))[0],
                        merged_sell_intents.get(order_id, (None, None))[1],
                    )
                    for order_id, qty, value, broker_updated_at in normalized_fills
                ],
            )

        cursor = conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET state_revision = state_revision + 1,
                alpaca_asset_id = COALESCE(alpaca_asset_id, ?),
                buy_order_qty = COALESCE(buy_order_qty, ?),
                buy_order_limit_price = COALESCE(buy_order_limit_price, ?),
                buy_status = ?,
                filled_qty = ?,
                filled_avg_price = ?,
                filled_at = ?,
                buy_fill_broker_updated_at = ?,
                buy_fill_component_revisions = ?,
                buy_fill_pending_observation = ?,
                buy_causality_quarantine = ?,
                target_sell_price = ?,
                sell_status = ?,
                sell_filled_qty = ?,
                sell_filled_avg_price = ?,
                sell_filled_at = ?,
                sell_renewal_requested_at = CASE WHEN ? THEN ? ELSE sell_renewal_requested_at END,
                sold_qty = ?,
                sold_value = ?,
                remaining_qty = ?,
                realized_pl = ?,
                realized_pl_pct = ?,
                closed_at = CASE WHEN ? THEN NULL ELSE closed_at END,
                closed_sell_shortfall_reopen_pending = ?,
                closed_correction_audited_at = CASE
                    WHEN ? THEN closed_correction_audited_at
                    ELSE ?
                END,
                notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND closed_at = ? AND state_revision = ?
            """,
            (
                observed_asset_id,
                immutable_buy_qty,
                immutable_buy_limit_price,
                effective_buy_status,
                effective_buy_qty if effective_buy_qty > 0 else None,
                effective_buy_price,
                effective_buy_filled_at,
                effective_buy_broker_updated_at,
                effective_buy_component_revisions_json,
                effective_buy_pending_observation_json,
                effective_buy_causality_quarantine,
                effective_target_sell_price,
                sell_status,
                sold_qty,
                sell_avg_price,
                effective_sell_filled_at,
                int(reopen and normalized_sell_renewal_requested_at is not None),
                normalized_sell_renewal_requested_at,
                sold_qty,
                sold_value,
                remaining_qty,
                realized_pl,
                realized_pl_pct,
                int(reopen),
                int(reopen and closed_sell_shortfall_reopen_pending),
                int(reopen),
                audited_at,
                effective_notes,
                position_id,
                expected_closed_at,
                expected_revision + 1,
            ),
        )
        if cursor.rowcount != 1:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            savepoint_active = False
            if owns_transaction:
                _commit_owned_transaction(conn)
            return False, False, False, 0.0
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        savepoint_active = False
    except BaseException as exc:
        if savepoint_active:
            _rollback_and_release_savepoint(conn, savepoint, exc)
        if owns_transaction:
            _rollback_owned_operation_if_active(
                conn,
                exc,
                operation="closed-position broker correction",
            )
        raise
    if owns_transaction:
        _commit_owned_transaction(conn)
    return True, reopen, active_conflict, remaining_qty


def active_alpaca_managed_symbols(conn: sqlite3.Connection) -> set[str]:
    if not conn.in_transaction:
        with _consistent_storage_read_snapshot(conn):
            return active_alpaca_managed_symbols(conn)
    position_rows = conn.execute(
        """
        SELECT symbol, alpaca_asset_id
        FROM alpaca_managed_positions
        WHERE closed_at IS NULL
        """
    ).fetchall()
    symbols: set[str] = set()
    for symbol, asset_id in position_rows:
        symbols.add(
            _canonical_managed_symbol(
                symbol,
                field_name="Persisted active managed Alpaca symbol",
            )
        )
        _canonical_optional_alpaca_asset_id(
            asset_id,
            field_name="Persisted active managed Alpaca asset ID",
        )
    alias_rows = conn.execute(
        """
        SELECT a.alpaca_asset_id, a.symbol
        FROM alpaca_symbol_aliases AS a
        JOIN alpaca_managed_positions AS p
          ON p.alpaca_asset_id = a.alpaca_asset_id
        WHERE p.closed_at IS NULL
        """
    ).fetchall()
    for asset_id, symbol in alias_rows:
        _canonical_alpaca_asset_id(
            asset_id,
            field_name="Persisted active managed Alpaca alias asset ID",
        )
        symbols.add(
            _canonical_managed_symbol(
                symbol,
                field_name="Persisted active managed Alpaca alias symbol",
            )
        )
    return symbols


def alpaca_managed_position_aliases(conn: sqlite3.Connection, position_id: int) -> set[str]:
    """Return every ticker known to identify a managed position's stable asset."""
    if not conn.in_transaction:
        with _consistent_storage_read_snapshot(conn):
            return alpaca_managed_position_aliases(conn, position_id)
    position_row = conn.execute(
        """
        SELECT symbol, alpaca_asset_id
        FROM alpaca_managed_positions
        WHERE id = ?
        """,
        (position_id,),
    ).fetchone()
    if position_row is None:
        return set()
    symbols = {
        _canonical_managed_symbol(
            position_row[0],
            field_name="Persisted managed Alpaca alias-owner symbol",
        )
    }
    _canonical_optional_alpaca_asset_id(
        position_row[1],
        field_name="Persisted managed Alpaca alias-owner asset ID",
    )
    alias_rows = conn.execute(
        """
        SELECT a.alpaca_asset_id, a.symbol
        FROM alpaca_symbol_aliases AS a
        JOIN alpaca_managed_positions AS p
          ON p.alpaca_asset_id = a.alpaca_asset_id
        WHERE p.id = ?
        """,
        (position_id,),
    ).fetchall()
    for asset_id, symbol in alias_rows:
        _canonical_alpaca_asset_id(
            asset_id,
            field_name="Persisted managed Alpaca alias asset ID",
        )
        symbols.add(
            _canonical_managed_symbol(
                symbol,
                field_name="Persisted managed Alpaca alias symbol",
            )
        )
    return symbols


def adopt_alpaca_managed_position_asset_if_current(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_state_revision: int,
    alpaca_asset_id: str,
) -> int | None:
    """Bind a legacy active row to one observed stable asset under an exact fence."""
    normalized_asset_id = _normalized_optional_alpaca_asset_id(
        alpaca_asset_id,
        field_name="Managed Alpaca asset adoption stable asset ID",
    )
    if normalized_asset_id is None:
        raise ValueError("Managed Alpaca asset adoption requires a stable asset ID.")
    if expected_state_revision < 0:
        raise ValueError("Managed Alpaca asset adoption requires a non-negative state revision.")
    try:
        with _managed_accounting_composite_savepoint(
            conn,
            "adopt_alpaca_managed_position_asset_if_current",
        ):
            cursor = conn.execute(
                """
            UPDATE alpaca_managed_positions
            SET state_revision = state_revision + 1,
                alpaca_asset_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND state_revision = ?
              AND closed_at IS NULL
              AND alpaca_asset_id IS NULL
              AND NOT EXISTS (
                    SELECT 1
                    FROM alpaca_managed_positions AS other
                    WHERE other.id != alpaca_managed_positions.id
                      AND other.closed_at IS NULL
                      AND other.alpaca_asset_id = ?
                  )
            RETURNING state_revision, symbol, signal_symbol
                """,
                (
                    normalized_asset_id,
                    position_id,
                    expected_state_revision,
                    normalized_asset_id,
                ),
            )
            returned_row = cursor.fetchone()
            if returned_row is None:
                row = None
            else:
                row = int(returned_row[0])
                _canonical_managed_symbol(
                    returned_row[1],
                    field_name="Persisted managed Alpaca asset-adoption symbol",
                )
                _canonical_managed_symbol(
                    returned_row[2],
                    field_name="Persisted managed Alpaca asset-adoption signal symbol",
                )
            if row is not None:
                conn.execute(
                    """
                INSERT INTO alpaca_symbol_aliases (alpaca_asset_id, symbol)
                SELECT alpaca_asset_id, symbol
                FROM alpaca_managed_positions
                WHERE id = ?
                ON CONFLICT(alpaca_asset_id, symbol) DO UPDATE SET
                    last_seen_at = CURRENT_TIMESTAMP
                    """,
                    (position_id,),
                )
    except sqlite3.IntegrityError as exc:
        if getattr(exc, "__notes__", None):
            # A failed savepoint rollback is not an ordinary identity race and
            # must remain observable, especially when cleanup closed ``conn``.
            raise
        return None
    # A deferred constraint can fail only at this boundary. Keep commit-time
    # integrity failures observable; they are not the ordinary execute-time
    # identity conflict represented by the ``None`` return value above.
    _commit_owned_transaction(conn)
    return row


def migrate_alpaca_managed_position_symbol(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    alpaca_asset_id: str,
    current_symbol: str,
    expected_closed_at: str | None = None,
    commit: bool = True,
) -> bool:
    """Persist a broker ticker rename while retaining every known alias.

    By default the update is restricted to an active row. Supplying
    ``expected_closed_at`` allows the symbol-migration pass to update the exact
    recently closed snapshot that will immediately undergo correction audit.
    """
    normalized_asset_id = _canonical_alpaca_asset_id(
        alpaca_asset_id,
        field_name="Managed Alpaca symbol migration asset ID",
    )
    normalized_current_symbol = _canonical_managed_symbol(
        current_symbol,
        field_name="Managed Alpaca symbol migration current symbol",
    )
    row = conn.execute(
        "SELECT symbol, alpaca_asset_id, closed_at FROM alpaca_managed_positions WHERE id = ?",
        (position_id,),
    ).fetchone()
    if row is None:
        return False
    prior_symbol = _canonical_managed_symbol(
        row[0],
        field_name="Managed Alpaca symbol migration prior symbol",
    )
    prior_asset_id = (
        None
        if row[1] is None
        else _canonical_alpaca_asset_id(
            row[1],
            field_name="Managed Alpaca symbol migration prior asset ID",
        )
    )
    observed_closed_at = None if row[2] is None else str(row[2])
    if observed_closed_at != expected_closed_at:
        return False
    if prior_asset_id is not None and prior_asset_id != normalized_asset_id:
        return False
    if expected_closed_at is None:
        collision = conn.execute(
            """
            SELECT 1 FROM alpaca_managed_positions
            WHERE id != ? AND closed_at IS NULL
              AND (UPPER(symbol) = UPPER(?) OR alpaca_asset_id = ?)
            LIMIT 1
            """,
            (position_id, normalized_current_symbol, normalized_asset_id),
        ).fetchone()
        if collision is not None:
            return False
    with _managed_accounting_composite_savepoint(
        conn,
        "migrate_alpaca_managed_position_symbol",
    ):
        if expected_closed_at is None:
            cursor = conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET symbol = ?, alpaca_asset_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND closed_at IS NULL
                  AND (alpaca_asset_id IS NULL OR alpaca_asset_id = ?)
                  AND NOT EXISTS (
                        SELECT 1 FROM alpaca_managed_positions AS other
                        WHERE other.id != alpaca_managed_positions.id
                          AND other.closed_at IS NULL
                          AND (
                                UPPER(other.symbol) = UPPER(?)
                                OR other.alpaca_asset_id = ?
                              )
                      )
                """,
                (
                    normalized_current_symbol,
                    normalized_asset_id,
                    position_id,
                    normalized_asset_id,
                    normalized_current_symbol,
                    normalized_asset_id,
                ),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE alpaca_managed_positions
                SET symbol = ?, alpaca_asset_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND closed_at = ?
                  AND (alpaca_asset_id IS NULL OR alpaca_asset_id = ?)
                """,
                (
                    normalized_current_symbol,
                    normalized_asset_id,
                    position_id,
                    expected_closed_at,
                    normalized_asset_id,
                ),
            )
        if cursor.rowcount == 1:
            conn.executemany(
                """
                INSERT INTO alpaca_symbol_aliases (alpaca_asset_id, symbol)
                VALUES (?, ?)
                ON CONFLICT(alpaca_asset_id, symbol) DO UPDATE SET last_seen_at = CURRENT_TIMESTAMP
                """,
                [
                    (normalized_asset_id, prior_symbol),
                    (normalized_asset_id, normalized_current_symbol),
                ],
            )
    if commit:
        _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def update_alpaca_managed_buy_status(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_buy_status: str,
    expected_buy_alpaca_order_id: str | None,
    expected_filled_qty: float | None,
    expected_sell_client_order_id: str | None,
    buy_status: str,
    buy_alpaca_order_id: str | None = None,
    buy_submitted_at: str | None = None,
    notes: str | None = None,
) -> bool:
    buy_status = _managed_lifecycle_status(
        buy_status,
        field_name="Managed Alpaca buy status",
    )
    causality_assignment, causality_assignment_params = _managed_buy_causality_quarantine_assignment_sql(
        buy_order_qty_expression="buy_order_qty",
        buy_order_limit_price_expression="buy_order_limit_price",
        buy_status_expression="?",
        filled_qty_expression="filled_qty",
        filled_avg_price_expression="filled_avg_price",
        existing_marker_expression="buy_causality_quarantine",
        buy_status_parameters=(buy_status,),
    )
    owns_transaction = not conn.in_transaction
    cursor = _execute_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET buy_status = ?,
            buy_alpaca_order_id = COALESCE(?, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(?, buy_submitted_at),
            buy_causality_quarantine = {causality_assignment},
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND closed_at IS NULL
          AND buy_status = ?
          AND (
                (? IS NULL AND buy_alpaca_order_id IS NULL)
                OR buy_alpaca_order_id = ?
              )
          AND (
                (? IS NULL AND filled_qty IS NULL)
                OR (
                    ? IS NOT NULL
                    AND filled_qty IS NOT NULL
                    AND {_managed_quantity_matches_sql("filled_qty", "?")}
                )
              )
          AND (
                (? IS NULL AND sell_client_order_id IS NULL)
                OR sell_client_order_id = ?
              )
        """,
        (
            buy_status,
            buy_alpaca_order_id,
            buy_submitted_at,
            *causality_assignment_params,
            notes,
            position_id,
            expected_buy_status,
            expected_buy_alpaca_order_id,
            expected_buy_alpaca_order_id,
            expected_filled_qty,
            expected_filled_qty,
            expected_filled_qty,
            expected_filled_qty,
            expected_sell_client_order_id,
            expected_sell_client_order_id,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca buy-status update",
    )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def update_alpaca_managed_buy_status_if_current(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_buy_status: str,
    expected_buy_alpaca_order_id: str | None,
    expected_filled_qty: float | None,
    expected_sell_client_order_id: str | None,
    buy_status: str,
    buy_alpaca_order_id: str | None = None,
    buy_submitted_at: str | None = None,
    notes: str | None = None,
    expected_buy_submission_attempt_count: int | None = None,
    expected_state_revision: int | None = None,
    buy_broker_updated_at: str | None | object = _UNSET,
) -> bool:
    """Persist a buy observation only while its complete local generation is unchanged."""
    buy_status = _managed_lifecycle_status(
        buy_status,
        field_name="Managed Alpaca buy status",
    )
    broker_timestamp_provided = buy_broker_updated_at is not _UNSET
    observed_broker_updated_at = (
        _normalize_alpaca_broker_timestamp(None if buy_broker_updated_at is _UNSET else buy_broker_updated_at)
        if broker_timestamp_provided
        else None
    )
    broker_revision_assignment = "buy_observation_broker_updated_at"
    broker_revision_assignment_params: tuple[object, ...] = ()
    broker_revision_predicate = ""
    broker_revision_params: tuple[object, ...] = ()
    if broker_timestamp_provided:
        broker_revision_assignment = "?"
        broker_revision_assignment_params = (observed_broker_updated_at,)
        broker_revision_predicate = """
          AND (
                buy_observation_broker_updated_at IS NULL
                OR (
                    ? IS NOT NULL
                    AND ? > buy_observation_broker_updated_at
                )
              )
        """
        broker_revision_params = (
            observed_broker_updated_at,
            observed_broker_updated_at,
        )
    causality_assignment, causality_assignment_params = _managed_buy_causality_quarantine_assignment_sql(
        buy_order_qty_expression="buy_order_qty",
        buy_order_limit_price_expression="buy_order_limit_price",
        buy_status_expression="?",
        filled_qty_expression="filled_qty",
        filled_avg_price_expression="filled_avg_price",
        existing_marker_expression="buy_causality_quarantine",
        buy_status_parameters=(buy_status,),
    )
    owns_transaction = not conn.in_transaction
    operation = "managed Alpaca fenced buy-status update"
    cursor = _execute_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET buy_status = ?,
            buy_alpaca_order_id = COALESCE(?, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(?, buy_submitted_at),
            buy_observation_broker_updated_at = {broker_revision_assignment},
            buy_causality_quarantine = {causality_assignment},
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND closed_at IS NULL
          AND buy_status = ?
          AND (? IS NULL OR buy_submission_attempt_count = ?)
          AND (? IS NULL OR state_revision = ?)
          AND buy_alpaca_order_id IS ?
          AND filled_qty IS ?
          AND sell_client_order_id IS ?
          {broker_revision_predicate}
        """,
        (
            buy_status,
            buy_alpaca_order_id,
            buy_submitted_at,
            *broker_revision_assignment_params,
            *causality_assignment_params,
            notes,
            position_id,
            expected_buy_status,
            expected_buy_submission_attempt_count,
            expected_buy_submission_attempt_count,
            expected_state_revision,
            expected_state_revision,
            expected_buy_alpaca_order_id,
            expected_filled_qty,
            expected_sell_client_order_id,
            *broker_revision_params,
        ),
        owns_transaction=owns_transaction,
        operation=operation,
    )
    observation_applied = cursor.rowcount == 1
    observation_is_idempotent = False
    if not observation_applied and broker_timestamp_provided:
        observation_is_idempotent = bool(
            _execute_returning_owned_operation_step(
                conn,
                """
                SELECT 1
                FROM alpaca_managed_positions
                WHERE id = ?
                  AND closed_at IS NULL
                  AND buy_status = ?
                  AND LOWER(buy_status) = LOWER(?)
                  AND (? IS NULL OR buy_submission_attempt_count = ?)
                  AND (? IS NULL OR state_revision = ?)
                  AND buy_alpaca_order_id IS ?
                  AND filled_qty IS ?
                  AND sell_client_order_id IS ?
                  AND buy_observation_broker_updated_at IS ?
                  AND (? IS NULL OR buy_alpaca_order_id IS ?)
                  AND (? IS NULL OR buy_submitted_at IS ?)
                  AND (? IS NULL OR notes IS ?)
                """,
                (
                    position_id,
                    expected_buy_status,
                    buy_status,
                    expected_buy_submission_attempt_count,
                    expected_buy_submission_attempt_count,
                    expected_state_revision,
                    expected_state_revision,
                    expected_buy_alpaca_order_id,
                    expected_filled_qty,
                    expected_sell_client_order_id,
                    observed_broker_updated_at,
                    buy_alpaca_order_id,
                    buy_alpaca_order_id,
                    buy_submitted_at,
                    buy_submitted_at,
                    notes,
                    notes,
                ),
                owns_transaction=owns_transaction,
                operation=operation,
                decode=lambda _row: True,
            )
        )
    if owns_transaction:
        _commit_owned_transaction(conn)
    return observation_applied or observation_is_idempotent


def quarantine_alpaca_managed_buy_side_effect_if_current(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_state_revision: int,
    expected_buy_submission_attempt_count: int,
    expected_buy_status: str,
    expected_buy_alpaca_order_id: str | None,
    expected_filled_qty: float | None,
    expected_filled_avg_price: float | None,
    expected_target_sell_price: float | None,
    expected_sell_client_order_id: str | None,
    expected_closed_at: str | None,
    buy_status: str,
    observed_buy_alpaca_order_id: str | None,
    observed_buy_submitted_at: str | None,
    notes: str,
    expected_buy_cancellation_alpaca_order_ids: str | None | object = _UNSET,
    buy_cancellation_alpaca_order_ids: list[str] | None = None,
) -> bool:
    """Durably quarantine a known broker side effect against an exact row snapshot.

    This deliberately also covers a concurrently closed row. A successful
    broker response cannot be erased merely because local lifecycle state
    advanced while the request was in flight; closed-position auditing still
    needs a durable diagnostic and any exact broker ID that was safely
    available.
    """
    buy_status = _managed_lifecycle_status(
        buy_status,
        field_name="Quarantined managed Alpaca buy status",
    )
    cancellation_ids_assignment = "buy_cancellation_alpaca_order_ids"
    cancellation_ids_params: tuple[object, ...] = ()
    if buy_cancellation_alpaca_order_ids is not None:
        cancellation_ids_assignment = "?"
        cancellation_ids_params = (_encode_alpaca_order_id_list(buy_cancellation_alpaca_order_ids),)
    cancellation_ids_predicate = ""
    cancellation_ids_snapshot_params: tuple[object, ...] = ()
    if expected_buy_cancellation_alpaca_order_ids is not _UNSET:
        cancellation_ids_predicate = "AND buy_cancellation_alpaca_order_ids IS ?"
        cancellation_ids_snapshot_params = (expected_buy_cancellation_alpaca_order_ids,)
    owns_transaction = not conn.in_transaction
    cursor = _execute_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET buy_status = ?,
            buy_alpaca_order_id = COALESCE(buy_alpaca_order_id, ?),
            buy_submitted_at = COALESCE(buy_submitted_at, ?),
            buy_cancellation_alpaca_order_ids = {cancellation_ids_assignment},
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND state_revision = ?
          AND buy_submission_attempt_count = ?
          AND buy_status IS ?
          AND buy_alpaca_order_id IS ?
          AND filled_qty IS ?
          AND filled_avg_price IS ?
          AND target_sell_price IS ?
          AND sell_client_order_id IS ?
          AND closed_at IS ?
          {cancellation_ids_predicate}
        """,
        (
            buy_status,
            observed_buy_alpaca_order_id,
            observed_buy_submitted_at,
            *cancellation_ids_params,
            notes,
            position_id,
            expected_state_revision,
            expected_buy_submission_attempt_count,
            expected_buy_status,
            expected_buy_alpaca_order_id,
            expected_filled_qty,
            expected_filled_avg_price,
            expected_target_sell_price,
            expected_sell_client_order_id,
            expected_closed_at,
            *cancellation_ids_snapshot_params,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca buy-side-effect quarantine",
    )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def close_alpaca_managed_buy_if_current_and_unfilled(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_buy_status: str,
    expected_buy_alpaca_order_id: str | None,
    buy_status: str,
    buy_alpaca_order_id: str | None = None,
    buy_submitted_at: str | None = None,
    closed_at: str | None = None,
    notes: str | None = None,
    expected_buy_submission_attempt_count: int | None = None,
    expected_state_revision: int | None = None,
    buy_broker_updated_at: str | None | object = _UNSET,
) -> bool:
    """Atomically close only the exact observed, never-filled buy generation."""
    buy_status = _managed_lifecycle_status(
        buy_status,
        field_name="Closed managed Alpaca buy status",
    )
    if buy_status.lower() == "submission_not_found" and expected_state_revision is None:
        raise ValueError("Missing managed-buy closure requires the exact observed state revision.")
    normalized_closed_at = _normalize_alpaca_closure_timestamp(closed_at)
    broker_timestamp_provided = buy_broker_updated_at is not _UNSET
    observed_broker_updated_at = (
        _normalize_alpaca_broker_timestamp(None if buy_broker_updated_at is _UNSET else buy_broker_updated_at)
        if broker_timestamp_provided
        else None
    )
    broker_revision_assignment = "buy_observation_broker_updated_at"
    broker_revision_assignment_params: tuple[object, ...] = ()
    broker_revision_predicate = ""
    broker_revision_params: tuple[object, ...] = ()
    if broker_timestamp_provided:
        broker_revision_assignment = "?"
        broker_revision_assignment_params = (observed_broker_updated_at,)
        broker_revision_predicate = """
          AND (
                buy_observation_broker_updated_at IS NULL
                OR (
                    ? IS NOT NULL
                    AND ? > buy_observation_broker_updated_at
                )
                OR (
                    ? IS NOT NULL
                    AND ? = buy_observation_broker_updated_at
                    AND LOWER(buy_status) = LOWER(?)
                )
              )
        """
        broker_revision_params = (
            observed_broker_updated_at,
            observed_broker_updated_at,
            observed_broker_updated_at,
            observed_broker_updated_at,
            buy_status,
        )
    owns_transaction = not conn.in_transaction
    cursor = _execute_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET buy_status = ?,
            buy_alpaca_order_id = COALESCE(?, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(?, buy_submitted_at),
            buy_observation_broker_updated_at = {broker_revision_assignment},
            closed_at = COALESCE(?, CURRENT_TIMESTAMP),
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND closed_at IS NULL
          AND buy_status = ?
          AND (? IS NULL OR buy_submission_attempt_count = ?)
          AND (? IS NULL OR state_revision = ?)
          AND filled_qty IS NULL
          AND sell_client_order_id IS NULL
          AND (
                (? IS NULL AND buy_alpaca_order_id IS NULL)
                OR buy_alpaca_order_id = ?
              )
          {broker_revision_predicate}
        """,
        (
            buy_status,
            buy_alpaca_order_id,
            buy_submitted_at,
            *broker_revision_assignment_params,
            normalized_closed_at,
            notes,
            position_id,
            expected_buy_status,
            expected_buy_submission_attempt_count,
            expected_buy_submission_attempt_count,
            expected_state_revision,
            expected_state_revision,
            expected_buy_alpaca_order_id,
            expected_buy_alpaca_order_id,
            *broker_revision_params,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca unfilled-buy closure",
    )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def mark_alpaca_managed_buy_filled(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    buy_status: str,
    filled_qty: float,
    filled_avg_price: float,
    filled_at: str | None,
    target_sell_price: float,
    buy_fill_broker_updated_at: str | None | object = _UNSET,
    buy_fill_broker_oldest_updated_at: str | None | object = _UNSET,
    buy_fill_component_revisions: dict[str, object] | None | object = _UNSET,
    buy_alpaca_order_id: str | None = None,
    buy_submitted_at: str | None = None,
    notes: str | None = None,
    expected_buy_status: str | None | object = _UNSET,
    expected_buy_alpaca_order_id: str | None | object = _UNSET,
    expected_filled_qty: float | None | object = _UNSET,
    expected_filled_avg_price: float | None | object = _UNSET,
    expected_target_sell_price: float | None | object = _UNSET,
    expected_sell_client_order_id: str | None | object = _UNSET,
    expected_buy_submission_attempt_count: int | None = None,
    return_state_revision: bool = False,
) -> bool | tuple[bool, int | None]:
    """Record the latest cumulative buy fill using aggregate-average cost basis.

    If the parent buy receives additional fills after an exit, the broker's
    cumulative average price becomes the cost basis for every matched share.
    Recompute any already-realized P/L in the same guarded update so the stored
    accounting cannot retain an average from an older cumulative observation.
    """
    buy_status = _managed_lifecycle_status(
        buy_status,
        field_name="Filled managed Alpaca buy status",
    )

    def observation_result(is_current: bool, state_revision: int | None = None) -> bool | tuple[bool, int | None]:
        if return_state_revision:
            return is_current, state_revision
        return is_current

    if expected_target_sell_price is not _UNSET and expected_target_sell_price is not None:
        expected_target_sell_price = _normalize_managed_target_sell_price(
            expected_target_sell_price,
        )
    expected_values = (
        expected_buy_status,
        expected_buy_alpaca_order_id,
        expected_filled_qty,
        expected_filled_avg_price,
        expected_target_sell_price,
        expected_sell_client_order_id,
    )
    has_expected_snapshot = all(value is not _UNSET for value in expected_values)
    if any(value is not _UNSET for value in expected_values) and not has_expected_snapshot:
        raise ValueError("A managed buy fill CAS requires the complete expected buy state.")

    filled_qty = float(filled_qty)
    filled_avg_price = float(filled_avg_price)
    target_sell_price = _normalize_managed_target_sell_price(target_sell_price)
    if not math.isfinite(filled_qty) or filled_qty <= 0:
        raise ValueError("An Alpaca buy fill quantity must be finite and positive.")
    if not math.isfinite(filled_avg_price) or filled_avg_price <= 0:
        raise ValueError("An Alpaca buy fill average price must be finite and positive.")
    buy_causality_note: str | None = None
    buy_intent_qty_snapshot: object | None = None
    buy_intent_limit_price_snapshot: object | None = None
    buy_intent = conn.execute(
        "SELECT buy_order_qty, buy_order_limit_price FROM alpaca_managed_positions WHERE id = ?",
        (position_id,),
    ).fetchone()
    if buy_intent is not None:
        buy_intent_qty_snapshot, buy_intent_limit_price_snapshot = buy_intent
        buy_causality_issue = _managed_buy_fill_causality_issue(
            buy_order_qty=buy_intent_qty_snapshot,  # type: ignore[arg-type]
            buy_order_limit_price=buy_intent_limit_price_snapshot,  # type: ignore[arg-type]
            buy_status=buy_status,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
        )
        buy_status, notes, buy_causality_note = _quarantine_managed_buy_causality_issue(
            buy_status=buy_status,
            notes=notes,
            issue=buy_causality_issue,
        )
    expected_numeric_values = (
        ("filled quantity", expected_filled_qty, True),
        ("filled average price", expected_filled_avg_price, False),
        ("target sell price", expected_target_sell_price, False),
    )
    for label, raw_value, allow_zero in expected_numeric_values:
        if raw_value is _UNSET or raw_value is None:
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"An expected Alpaca buy {label} must be finite and valid.")

    broker_timestamp_provided = buy_fill_broker_updated_at is not _UNSET
    observed_broker_updated_at = (
        _normalize_alpaca_broker_timestamp(None if buy_fill_broker_updated_at is _UNSET else buy_fill_broker_updated_at)
        if broker_timestamp_provided
        else None
    )
    if buy_fill_broker_oldest_updated_at is not _UNSET and not broker_timestamp_provided:
        raise ValueError("A managed buy's oldest broker revision requires its latest broker revision.")
    observed_oldest_broker_updated_at = (
        observed_broker_updated_at
        if buy_fill_broker_oldest_updated_at is _UNSET
        else _normalize_alpaca_broker_timestamp(buy_fill_broker_oldest_updated_at)
    )
    if (
        observed_oldest_broker_updated_at is not None
        and observed_broker_updated_at is not None
        and observed_oldest_broker_updated_at > observed_broker_updated_at
    ):
        raise ValueError("A managed buy's oldest broker revision cannot follow its latest revision.")
    buy_component_revisions_provided = buy_fill_component_revisions is not _UNSET
    if buy_component_revisions_provided and not broker_timestamp_provided:
        raise ValueError("A managed buy's component revisions require its latest broker revision.")
    observed_component_revisions = (
        None
        if buy_fill_component_revisions is _UNSET or buy_fill_component_revisions is None
        else _normalize_alpaca_buy_component_revisions(buy_fill_component_revisions)
    )
    _validate_alpaca_buy_component_revision_summary(
        observed_component_revisions,
        latest=observed_broker_updated_at,
        oldest=observed_oldest_broker_updated_at,
    )
    _validate_alpaca_buy_component_accounting_summary(
        observed_component_revisions,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
    )
    observed_component_revisions_json = _encode_alpaca_buy_component_revisions(observed_component_revisions)
    observed_pending_observation_json = (
        None
        if observed_component_revisions is None
        else _encode_alpaca_buy_pending_observation(
            buy_status=buy_status,
            filled_qty=float(filled_qty),
            filled_avg_price=float(filled_avg_price),
            filled_at=filled_at,
            target_sell_price=float(target_sell_price),
            component_revisions=observed_component_revisions,
        )
    )

    revision_snapshot_predicate = ""
    revision_snapshot_params: tuple[object, ...] = ()
    component_revision_not_stale = True
    component_revision_is_newer = False
    component_revision_authorizes_mutation = False
    component_revision_can_advance_summary = True
    pending_observation_matches = False
    clear_pending_observation = False
    component_restage_candidate = False
    legacy_component_seed_candidate = False
    component_revision_install_authorized = False
    persisted_broker_updated_at: str | None = None
    persisted_component_revisions_json: str | None = None
    if buy_component_revisions_provided:
        revision_row = conn.execute(
            "SELECT buy_fill_broker_updated_at, buy_fill_component_revisions, "
            "buy_fill_pending_observation "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        if revision_row is None:
            return observation_result(False)
        (
            persisted_broker_updated_at,
            persisted_component_revisions_json,
            persisted_pending_observation_json,
        ) = revision_row
        persisted_component_revisions = _decode_alpaca_buy_component_revisions(persisted_component_revisions_json)
        persisted_component_revisions_are_scalar_compatible = bool(
            persisted_component_revisions is None
            or persisted_broker_updated_at is None
            or _alpaca_buy_component_latest_revision(persisted_component_revisions) >= persisted_broker_updated_at
        )
        if persisted_component_revisions is not None and persisted_component_revisions_are_scalar_compatible:
            component_revision_not_stale, component_revision_is_newer = _alpaca_buy_component_revision_order(
                persisted=persisted_component_revisions,
                observed=observed_component_revisions,
            )
            pending_observation_matches = bool(
                persisted_pending_observation_json is not None
                and observed_pending_observation_json == persisted_pending_observation_json
            )
            pending_snapshot_ahead = bool(
                persisted_pending_observation_json is not None
                and persisted_broker_updated_at is not None
                and _alpaca_buy_component_latest_revision(persisted_component_revisions) > persisted_broker_updated_at
            )
            component_accounting_is_ordered = _alpaca_buy_component_accounting_is_ordered(
                persisted=persisted_component_revisions,
                observed=observed_component_revisions,
            )
            component_revision_authorizes_mutation = bool(
                component_revision_not_stale
                and component_accounting_is_ordered
                and (
                    pending_snapshot_ahead and pending_observation_matches
                    if persisted_pending_observation_json is not None
                    else component_revision_is_newer
                )
            )
            component_revision_can_advance_summary = bool(
                component_accounting_is_ordered
                and (persisted_pending_observation_json is None or pending_observation_matches)
            )
            clear_pending_observation = bool(
                component_revision_authorizes_mutation and persisted_pending_observation_json is not None
            )
            component_restage_candidate = bool(
                has_expected_snapshot
                and persisted_pending_observation_json is not None
                and observed_pending_observation_json is not None
                and not pending_observation_matches
                and component_revision_not_stale
                and component_revision_is_newer
                and component_accounting_is_ordered
            )
        elif persisted_broker_updated_at is None:
            component_revision_not_stale, component_revision_is_newer = _alpaca_buy_component_revision_order(
                persisted=None,
                observed=observed_component_revisions,
            )
            component_revision_authorizes_mutation = bool(component_revision_not_stale and component_revision_is_newer)
        else:
            # A migrated scalar high-water mark cannot prove which component
            # advanced. Require the old conservative fence for mutations; an
            # exact accounting replay below may seed the component snapshot.
            component_revision_not_stale = bool(
                observed_oldest_broker_updated_at is not None
                and observed_oldest_broker_updated_at >= persisted_broker_updated_at
            )
            component_revision_is_newer = bool(
                observed_oldest_broker_updated_at is not None
                and observed_oldest_broker_updated_at > persisted_broker_updated_at
            )
            component_revision_authorizes_mutation = bool(component_revision_not_stale and component_revision_is_newer)
            legacy_component_seed_candidate = bool(
                has_expected_snapshot
                and observed_component_revisions_json is not None
                and observed_broker_updated_at is not None
                and component_revision_not_stale
                and observed_broker_updated_at > persisted_broker_updated_at
            )
        component_revision_install_authorized = bool(
            observed_component_revisions_json is not None
            and (persisted_component_revisions is None or not persisted_component_revisions_are_scalar_compatible)
            and component_revision_not_stale
        )
        clear_pending_observation = bool(clear_pending_observation or component_revision_install_authorized)
        revision_snapshot_predicate = """
          AND buy_fill_broker_updated_at IS ?
          AND buy_fill_component_revisions IS ?
          AND buy_fill_pending_observation IS ?
        """
        revision_snapshot_params = (
            persisted_broker_updated_at,
            persisted_component_revisions_json,
            persisted_pending_observation_json,
        )

    snapshot_predicate = ""
    snapshot_params: tuple[object, ...] = ()
    if has_expected_snapshot:
        snapshot_predicate = """
          AND buy_status IS ?
          AND buy_alpaca_order_id IS ?
          AND filled_qty IS ?
          AND filled_avg_price IS ?
          AND target_sell_price IS ?
          AND sell_client_order_id IS ?
        """
        snapshot_params = expected_values

    initial_fill_lifecycle_predicate = ""
    initial_fill_lifecycle_params: tuple[object, ...] = ()
    if broker_timestamp_provided:
        # Fence fill accounting directly against the independent lifecycle
        # high-water mark. Equality remains valid for an ordinary
        # save-status-then-fill observation, but not for a legacy row that
        # persisted zero as a non-NULL fill: its prior positive-fill premise is
        # ambiguous and must advance beyond the lifecycle observation.
        initial_fill_lifecycle_predicate = f"""
          AND (
                buy_observation_broker_updated_at IS NULL
                OR (
                    ? IS NOT NULL
                    AND (
                        ? > buy_observation_broker_updated_at
                        OR (
                            (filled_qty IS NULL OR {_managed_position_quantity_is_positive_sql("filled_qty")})
                            AND ? = buy_observation_broker_updated_at
                        )
                    )
                )
              )
        """
        initial_fill_lifecycle_params = (
            observed_broker_updated_at,
            observed_broker_updated_at,
            observed_broker_updated_at,
        )

    owns_transaction = not conn.in_transaction
    operation = "managed Alpaca buy-fill accounting"
    if component_restage_candidate:
        _execute_owned_operation_step(
            conn,
            f"""
            UPDATE alpaca_managed_positions
            SET buy_fill_component_revisions = ?,
                buy_fill_pending_observation = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND closed_at IS NULL
              AND (? IS NULL OR buy_submission_attempt_count = ?)
              {snapshot_predicate}
              {revision_snapshot_predicate}
            """,
            (
                observed_component_revisions_json,
                observed_pending_observation_json,
                position_id,
                expected_buy_submission_attempt_count,
                expected_buy_submission_attempt_count,
                *snapshot_params,
                *revision_snapshot_params,
            ),
            owns_transaction=owns_transaction,
            operation=operation,
        )
        if owns_transaction:
            _commit_owned_transaction(conn)
        return observation_result(False)

    # Callers without an explicit snapshot are retained for fixture/setup use,
    # but an equal-quantity observation may then only be an idempotent replay.
    # Any status, price, or broker-order mutation at the same cumulative fill
    # requires the complete caller snapshot above.
    observed_filled_qty_sql = repr(filled_qty)
    observed_filled_avg_price_sql = repr(filled_avg_price)
    observed_target_sell_price_sql = repr(target_sell_price)
    persisted_filled_qty_sql = "COALESCE(filled_qty, 0)"
    buy_fill_quantity_mark_price_sql = f"""
        MAX(
            {observed_filled_avg_price_sql},
            {observed_target_sell_price_sql},
            COALESCE(filled_avg_price, 0),
            COALESCE(target_sell_price, 0)
        )
    """
    buy_fill_quantity_scale_sql = f"MAX(ABS({observed_filled_qty_sql}), ABS({persisted_filled_qty_sql}))"
    buy_fill_quantity_value_scale_sql = f"""
        MAX(
            ABS(COALESCE(sold_value, 0)),
            ABS(({observed_filled_qty_sql}) * ({observed_filled_avg_price_sql})),
            ABS(({buy_fill_quantity_scale_sql}) * ({buy_fill_quantity_mark_price_sql}))
        )
    """
    buy_fill_quantities_match_sql = _managed_accounting_quantities_match_sql(
        observed_filled_qty_sql,
        persisted_filled_qty_sql,
        mark_price_expression=buy_fill_quantity_mark_price_sql,
        value_scale_expression=buy_fill_quantity_value_scale_sql,
    )
    buy_fill_quantity_is_not_materially_lower_sql = (
        f"(({observed_filled_qty_sql}) >= ({persisted_filled_qty_sql}) OR ({buy_fill_quantities_match_sql}))"
    )
    buy_fill_quantity_is_within_share_regression_tolerance_sql = f"""
        (
            ({observed_filled_qty_sql})
                + ({_managed_quantity_tolerance_sql(buy_fill_quantity_scale_sql)})
            >= ({persisted_filled_qty_sql})
        )
    """
    buy_fill_quantity_materially_advanced_sql = (
        f"(({observed_filled_qty_sql}) > ({persisted_filled_qty_sql}) AND NOT ({buy_fill_quantities_match_sql}))"
    )
    if has_expected_snapshot and not broker_timestamp_provided:
        # Backward compatibility for local fixture/setup callers. Broker-facing
        # paths explicitly provide a timestamp (including explicit ``None``)
        # and use the ordered branch below.
        equal_quantity_predicate = buy_fill_quantity_is_not_materially_lower_sql
        quantity_params: tuple[object, ...] = ()
    elif has_expected_snapshot and buy_component_revisions_provided:
        equal_quantity_predicate = f"""
            (
                (
                    {buy_fill_quantities_match_sql}
                    AND ABS(filled_avg_price - ?) <= 0.00000001
                    AND ABS(target_sell_price - ?) <= 0.00000001
                    AND buy_status IS ?
                    AND (? IS NULL OR buy_alpaca_order_id IS ?)
                    AND (? IS NULL OR buy_submitted_at IS ?)
                    AND (? IS NULL OR filled_at IS ?)
                )
                OR (
                    {buy_fill_quantity_is_within_share_regression_tolerance_sql}
                    AND ?
                )
            )
        """
        quantity_params = (
            filled_avg_price,
            target_sell_price,
            buy_status,
            buy_alpaca_order_id,
            buy_alpaca_order_id,
            buy_submitted_at,
            buy_submitted_at,
            filled_at,
            filled_at,
            int(component_revision_authorizes_mutation),
        )
    elif has_expected_snapshot:
        equal_quantity_predicate = f"""
            (
                (
                    {buy_fill_quantities_match_sql}
                    AND ABS(filled_avg_price - ?) <= 0.00000001
                    AND ABS(target_sell_price - ?) <= 0.00000001
                    AND buy_status IS ?
                    AND (? IS NULL OR buy_alpaca_order_id IS ?)
                    AND (? IS NULL OR buy_submitted_at IS ?)
                    AND (? IS NULL OR filled_at IS ?)
                )
                OR (
                    {buy_fill_quantity_is_within_share_regression_tolerance_sql}
                    AND ? IS NOT NULL
                    AND buy_fill_broker_updated_at IS NOT NULL
                    AND ? > buy_fill_broker_updated_at
                )
            )
        """
        quantity_params = (
            filled_avg_price,
            target_sell_price,
            buy_status,
            buy_alpaca_order_id,
            buy_alpaca_order_id,
            buy_submitted_at,
            buy_submitted_at,
            filled_at,
            filled_at,
            observed_oldest_broker_updated_at,
            observed_oldest_broker_updated_at,
        )
    else:
        equal_quantity_predicate = f"""
            (
                {buy_fill_quantity_materially_advanced_sql}
                OR (
                    {buy_fill_quantities_match_sql}
                    AND buy_status IS ?
                    AND ABS(filled_avg_price - ?) <= 0.00000001
                    AND ABS(target_sell_price - ?) <= 0.00000001
                    AND (? IS NULL OR buy_alpaca_order_id IS NULL OR buy_alpaca_order_id = ?)
                )
            )
        """
        quantity_params = (
            buy_status,
            filled_avg_price,
            target_sell_price,
            buy_alpaca_order_id,
            buy_alpaca_order_id,
        )
    timestamp_assignment = "buy_fill_broker_updated_at"
    timestamp_params: tuple[object, ...] = ()
    component_revisions_assignment = "buy_fill_component_revisions"
    component_revisions_params: tuple[object, ...] = ()
    notes_assignment, notes_params = _managed_buy_notes_assignment(
        notes,
        buy_causality_note,
    )
    if broker_timestamp_provided:
        assignable_broker_updated_at = (
            observed_broker_updated_at
            if not buy_component_revisions_provided
            or (component_revision_not_stale and component_revision_can_advance_summary)
            else None
        )
        timestamp_assignment = """
            CASE
                WHEN ? IS NULL THEN buy_fill_broker_updated_at
                WHEN buy_fill_broker_updated_at IS NULL THEN ?
                WHEN ? > buy_fill_broker_updated_at THEN ?
                ELSE buy_fill_broker_updated_at
            END
        """
        timestamp_params = (
            assignable_broker_updated_at,
            assignable_broker_updated_at,
            assignable_broker_updated_at,
            assignable_broker_updated_at,
        )
        # An exact replay may safely advance the aggregate high-water mark,
        # but an older/equal replay must not replace diagnostics produced by a
        # later reconciliation.  Initial fills and component-wise newer
        # snapshots may still attach their observation notes.
        if buy_component_revisions_provided:
            component_revisions_assignment = """
                CASE
                    WHEN ? THEN ?
                    WHEN ? THEN ?
                    ELSE buy_fill_component_revisions
                END
            """
            component_revisions_params = (
                int(component_revision_install_authorized),
                observed_component_revisions_json,
                int(component_revision_authorizes_mutation),
                observed_component_revisions_json,
            )
            notes_assignment = f"""
                CASE
                    WHEN filled_qty IS NULL
                         AND buy_fill_broker_updated_at IS NULL
                    THEN {notes_assignment}
                    WHEN ? THEN {notes_assignment}
                    ELSE notes
                END
            """
            notes_params = (
                *notes_params,
                int(component_revision_authorizes_mutation),
                *notes_params,
            )
        else:
            notes_assignment = f"""
                CASE
                    WHEN filled_qty IS NULL
                         AND buy_fill_broker_updated_at IS NULL
                    THEN {notes_assignment}
                    WHEN ? IS NOT NULL
                         AND buy_fill_broker_updated_at IS NOT NULL
                         AND ? > buy_fill_broker_updated_at
                    THEN {notes_assignment}
                    ELSE notes
                END
            """
            notes_params = (
                *notes_params,
                observed_oldest_broker_updated_at,
                observed_oldest_broker_updated_at,
                *notes_params,
            )

    # Avoid issuing an UPDATE for an accounting-identical replay. The table's
    # revision trigger deliberately advances every ordinary UPDATE, so even a
    # byte-for-byte broker retry would otherwise invalidate an in-flight sell
    # submission fence. Metadata migrations and genuinely newer observations
    # still fall through to the guarded UPDATE below.
    current = conn.execute(
        """
        SELECT buy_status, buy_alpaca_order_id, buy_submitted_at,
               buy_submission_attempt_count, filled_qty, filled_avg_price,
               filled_at, buy_fill_broker_updated_at,
               buy_fill_component_revisions, buy_fill_pending_observation,
               target_sell_price, notes, buy_causality_quarantine,
               sold_qty, sold_value, remaining_qty,
               realized_pl, realized_pl_pct, sell_client_order_id, closed_at,
               state_revision, buy_order_qty, buy_order_limit_price
        FROM alpaca_managed_positions
        WHERE id = ?
        """,
        (position_id,),
    ).fetchone()

    def same_number(left: object, right: object) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return float(left) == float(right)

    def same_quantity(
        left: object,
        right: object,
        *,
        mark_prices: tuple[float | None, ...],
        value_scale: float,
    ) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return _managed_accounting_quantities_match(
            float(left),
            float(right),
            mark_prices=mark_prices,
            value_scale=value_scale,
        )

    def same_value(left: object, right: object) -> bool:
        if left is None or right is None:
            return left is None and right is None
        left_value = float(left)
        right_value = float(right)
        tolerance = managed_value_reconciliation_tolerance(max(abs(left_value), abs(right_value)))
        return abs(left_value - right_value) <= tolerance

    if current is not None:
        (
            current_buy_status,
            current_buy_order_id,
            current_buy_submitted_at,
            current_attempt_count,
            current_filled_qty,
            current_filled_avg_price,
            current_filled_at,
            current_broker_updated_at,
            current_component_revisions,
            current_pending_observation,
            current_target_sell_price,
            current_notes,
            current_buy_causality_quarantine,
            current_sold_qty,
            current_sold_value,
            current_remaining_qty,
            current_realized_pl,
            current_realized_pl_pct,
            current_sell_client_order_id,
            current_closed_at,
            current_state_revision,
            current_buy_order_qty,
            current_buy_order_limit_price,
        ) = current
        effective_buy_order_id = buy_alpaca_order_id if buy_alpaca_order_id is not None else current_buy_order_id
        effective_buy_submitted_at = buy_submitted_at if buy_submitted_at is not None else current_buy_submitted_at
        effective_filled_at = filled_at if filled_at is not None else current_filled_at
        effective_broker_updated_at = current_broker_updated_at
        if (
            broker_timestamp_provided
            and assignable_broker_updated_at is not None
            and (effective_broker_updated_at is None or assignable_broker_updated_at > effective_broker_updated_at)
        ):
            effective_broker_updated_at = assignable_broker_updated_at
        effective_component_revisions = current_component_revisions
        if (
            broker_timestamp_provided
            and buy_component_revisions_provided
            and (component_revision_install_authorized or component_revision_authorizes_mutation)
        ):
            effective_component_revisions = observed_component_revisions_json
        effective_pending_observation = None if clear_pending_observation else current_pending_observation
        effective_notes = current_notes
        if not broker_timestamp_provided:
            effective_notes = _merge_managed_buy_causality_notes(
                current_notes,
                notes,
                buy_causality_note,
            )
        elif buy_component_revisions_provided:
            if (
                current_filled_qty is None and current_broker_updated_at is None
            ) or component_revision_authorizes_mutation:
                effective_notes = _merge_managed_buy_causality_notes(
                    current_notes,
                    notes,
                    buy_causality_note,
                )
        elif (current_filled_qty is None and current_broker_updated_at is None) or (
            observed_oldest_broker_updated_at is not None
            and current_broker_updated_at is not None
            and observed_oldest_broker_updated_at > current_broker_updated_at
        ):
            effective_notes = _merge_managed_buy_causality_notes(
                current_notes,
                notes,
                buy_causality_note,
            )

        sold_qty = float(current_sold_qty or 0.0)
        sold_value = float(current_sold_value or 0.0)
        effective_remaining_qty = float(filled_qty) - sold_qty
        quantity_scale = max(
            abs(effective_remaining_qty),
            abs(float(filled_qty)),
            abs(sold_qty),
        )
        sell_avg_price = sold_value / sold_qty if sold_qty > 0.0 else None
        residual_is_negligible = _managed_accounting_residual_is_negligible(
            effective_remaining_qty,
            quantity_scale=quantity_scale,
            mark_prices=(float(filled_avg_price), target_sell_price, sell_avg_price),
            value_scale=sold_value,
        )
        sold_quantity_is_positive = _managed_accounting_quantity_is_positive(
            sold_qty,
            quantity_scale=quantity_scale,
            mark_prices=(float(filled_avg_price), target_sell_price, sell_avg_price),
            value_scale=sold_value,
        )
        if effective_remaining_qty < 0.0 and not residual_is_negligible:
            effective_realized_pl = None
            effective_realized_pl_pct = None
        elif sold_quantity_is_positive:
            matched_qty = min(sold_qty, float(filled_qty))
            effective_realized_pl, effective_realized_pl_pct = _managed_realized_pl_values(
                sold_value=sold_value,
                matched_qty=matched_qty,
                buy_price=float(filled_avg_price),
            )
        else:
            effective_realized_pl = None
            effective_realized_pl_pct = None

        expected_snapshot_matches = bool(
            not has_expected_snapshot
            or (
                current_buy_status == expected_buy_status
                and current_buy_order_id == expected_buy_alpaca_order_id
                and same_number(current_filled_qty, expected_filled_qty)
                and same_number(current_filled_avg_price, expected_filled_avg_price)
                and same_number(current_target_sell_price, expected_target_sell_price)
                and current_sell_client_order_id == expected_sell_client_order_id
            )
        )
        revision_snapshot_matches = bool(
            not buy_component_revisions_provided
            or (
                current_broker_updated_at == persisted_broker_updated_at
                and current_component_revisions == persisted_component_revisions_json
                and current_pending_observation == persisted_pending_observation_json
            )
        )
        exact_replay = bool(
            current_closed_at is None
            and (
                expected_buy_submission_attempt_count is None
                or int(current_attempt_count) == expected_buy_submission_attempt_count
            )
            and expected_snapshot_matches
            and revision_snapshot_matches
            and same_number(current_buy_order_qty, buy_intent_qty_snapshot)
            and same_number(current_buy_order_limit_price, buy_intent_limit_price_snapshot)
            and current_buy_status == buy_status
            and current_buy_order_id == effective_buy_order_id
            and current_buy_submitted_at == effective_buy_submitted_at
            and same_quantity(
                current_filled_qty,
                filled_qty,
                mark_prices=(
                    None if current_filled_avg_price is None else float(current_filled_avg_price),
                    filled_avg_price,
                    None if current_target_sell_price is None else float(current_target_sell_price),
                    target_sell_price,
                ),
                value_scale=max(
                    abs(float(current_sold_value or 0.0)),
                    abs(float(filled_qty) * float(filled_avg_price)),
                ),
            )
            and same_number(current_filled_avg_price, filled_avg_price)
            and current_filled_at == effective_filled_at
            and current_broker_updated_at == effective_broker_updated_at
            and current_component_revisions == effective_component_revisions
            and current_pending_observation == effective_pending_observation
            and same_number(current_target_sell_price, target_sell_price)
            and current_notes == effective_notes
            and current_buy_causality_quarantine == buy_causality_note
            and same_quantity(
                current_remaining_qty,
                effective_remaining_qty,
                mark_prices=(
                    None if current_filled_avg_price is None else float(current_filled_avg_price),
                    filled_avg_price,
                    None if current_target_sell_price is None else float(current_target_sell_price),
                    target_sell_price,
                    sell_avg_price,
                ),
                value_scale=max(
                    abs(float(current_sold_value or 0.0)),
                    abs(sold_value),
                    abs(float(filled_qty) * float(filled_avg_price)),
                ),
            )
            and same_value(current_realized_pl, effective_realized_pl)
            and same_number(current_realized_pl_pct, effective_realized_pl_pct)
        )
        if exact_replay:
            if owns_transaction:
                _commit_owned_transaction(conn)
            return observation_result(True, int(current_state_revision))
    accounting_sold_qty_sql = "COALESCE(sold_qty, 0)"
    accounting_sold_value_sql = "COALESCE(sold_value, 0)"
    accounting_mark_price_sql = f"""
        MAX(
            {filled_avg_price!r},
            {target_sell_price!r},
            CASE
                WHEN {accounting_sold_qty_sql} > 0
                THEN {accounting_sold_value_sql} / {accounting_sold_qty_sql}
                ELSE 0
            END
        )
    """
    accounting_remaining_qty_sql = f"({filled_qty!r} - {accounting_sold_qty_sql})"
    accounting_quantity_scale_sql = f"""
        MAX(
            ABS({accounting_remaining_qty_sql}),
            ABS({filled_qty!r}),
            ABS({accounting_sold_qty_sql})
        )
    """
    accounting_value_scale_sql = f"""
        MAX(
            ABS({accounting_sold_value_sql}),
            ABS(({accounting_quantity_scale_sql}) * ({accounting_mark_price_sql}))
        )
    """
    oversold_residual_is_negligible_sql = _managed_quantity_is_negligible_sql(
        accounting_remaining_qty_sql,
        quantity_scale_expression=accounting_quantity_scale_sql,
        mark_price_expression=accounting_mark_price_sql,
        value_scale_expression=accounting_value_scale_sql,
    )
    sold_quantity_is_negligible_sql = _managed_quantity_is_negligible_sql(
        accounting_sold_qty_sql,
        quantity_scale_expression=accounting_quantity_scale_sql,
        mark_price_expression=accounting_mark_price_sql,
        value_scale_expression=accounting_value_scale_sql,
    )
    updated_state_revision = _execute_returning_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET state_revision = state_revision + 1,
            buy_status = ?,
            buy_alpaca_order_id = COALESCE(?, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(?, buy_submitted_at),
            filled_qty = ?,
            filled_avg_price = ?,
            remaining_qty = ? - COALESCE(sold_qty, 0),
            realized_pl = CASE
                WHEN COALESCE(sold_qty, 0) > ?
                     AND NOT ({oversold_residual_is_negligible_sql}) THEN NULL
                WHEN COALESCE(sold_qty, 0) > 0
                     AND NOT ({sold_quantity_is_negligible_sql}) THEN
                    COALESCE(sold_value, 0) - MIN(COALESCE(sold_qty, 0), ?) * ?
                ELSE NULL
            END,
            realized_pl_pct = CASE
                WHEN COALESCE(sold_qty, 0) > ?
                     AND NOT ({oversold_residual_is_negligible_sql}) THEN NULL
                WHEN COALESCE(sold_qty, 0) > 0
                     AND NOT ({sold_quantity_is_negligible_sql}) THEN
                    (
                        COALESCE(sold_value, 0) - MIN(COALESCE(sold_qty, 0), ?) * ?
                    ) / (MIN(COALESCE(sold_qty, 0), ?) * ?) * 100.0
                ELSE NULL
            END,
            filled_at = COALESCE(?, filled_at),
            buy_fill_broker_updated_at = {timestamp_assignment},
            buy_fill_component_revisions = {component_revisions_assignment},
            buy_fill_pending_observation = CASE WHEN ? THEN NULL
                                                ELSE buy_fill_pending_observation END,
            target_sell_price = ?,
            notes = {notes_assignment},
            buy_causality_quarantine = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND closed_at IS NULL
          AND buy_order_qty IS ?
          AND buy_order_limit_price IS ?
          AND (? IS NULL OR buy_submission_attempt_count = ?)
          AND (
                (
                    filled_qty IS NULL
                    AND buy_fill_broker_updated_at IS NULL
                )
                OR {equal_quantity_predicate}
              )
          {initial_fill_lifecycle_predicate}
          {snapshot_predicate}
          {revision_snapshot_predicate}
          AND (
                LOWER(COALESCE(sell_status, '')) NOT IN ('submission_pending', 'submission_retrying')
                OR sell_alpaca_order_id IS NULL
                OR target_sell_price IS NULL
                OR ABS(target_sell_price - ?) <= 0.00000001
              )
        RETURNING state_revision, realized_pl, realized_pl_pct
        """,
        (
            buy_status,
            buy_alpaca_order_id,
            buy_submitted_at,
            filled_qty,
            filled_avg_price,
            filled_qty,
            filled_qty,
            filled_qty,
            filled_avg_price,
            filled_qty,
            filled_qty,
            filled_avg_price,
            filled_qty,
            filled_avg_price,
            filled_at,
            *timestamp_params,
            *component_revisions_params,
            int(clear_pending_observation),
            target_sell_price,
            *notes_params,
            buy_causality_note,
            position_id,
            buy_intent_qty_snapshot,
            buy_intent_limit_price_snapshot,
            expected_buy_submission_attempt_count,
            expected_buy_submission_attempt_count,
            *quantity_params,
            *initial_fill_lifecycle_params,
            *snapshot_params,
            *revision_snapshot_params,
            target_sell_price,
        ),
        owns_transaction=owns_transaction,
        operation=operation,
        decode=_decode_managed_realized_pl_revision,
    )
    update_applied = updated_state_revision is not None
    if not update_applied and legacy_component_seed_candidate:
        _execute_owned_operation_step(
            conn,
            """
            UPDATE alpaca_managed_positions
            SET buy_fill_component_revisions = ?,
                buy_fill_pending_observation = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND closed_at IS NULL
              AND buy_fill_broker_updated_at IS ?
              AND buy_fill_component_revisions IS ?
              AND buy_fill_pending_observation IS ?
              AND (? IS NULL OR buy_submission_attempt_count = ?)
              AND buy_status IS ?
              AND buy_alpaca_order_id IS ?
              AND filled_qty IS ?
              AND filled_avg_price IS ?
              AND target_sell_price IS ?
              AND sell_client_order_id IS ?
            """,
            (
                observed_component_revisions_json,
                observed_pending_observation_json,
                position_id,
                persisted_broker_updated_at,
                persisted_component_revisions_json,
                persisted_pending_observation_json,
                expected_buy_submission_attempt_count,
                expected_buy_submission_attempt_count,
                *expected_values,
            ),
            owns_transaction=owns_transaction,
            operation=operation,
        )
    if owns_transaction:
        _commit_owned_transaction(conn)
    return observation_result(update_applied, updated_state_revision)


def record_alpaca_managed_sell_order(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    sell_alpaca_order_id: str | None,
    sell_submitted_at: str | None,
    sell_status: str,
    sell_expires_at: str | None = None,
    sell_order_qty: float | None = None,
    sell_order_limit_price: float | None = None,
    increment_renewal_count: bool = False,
    notes: str | None = None,
) -> None:
    sell_status = _managed_lifecycle_status(
        sell_status,
        field_name="Managed Alpaca sell status",
    )
    sell_order_qty, sell_order_limit_price = _normalize_optional_managed_sell_intent(
        sell_order_qty,
        sell_order_limit_price,
    )
    normalized_sell_alpaca_order_id = (
        None
        if sell_alpaca_order_id is None
        else _canonical_alpaca_order_id(
            sell_alpaca_order_id,
            field_name="A managed Alpaca sell order ID",
        )
    )
    with _managed_accounting_composite_savepoint(
        conn,
        "record_alpaca_managed_sell_order",
    ):
        prior_parent_intent = conn.execute(
            """
            SELECT sell_client_order_id, sell_alpaca_order_id,
                   sell_order_qty, sell_order_limit_price
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
        if (
            prior_parent_intent is not None
            and prior_parent_intent[1] is None
            and any(
                value is not None for value in (prior_parent_intent[0], prior_parent_intent[2], prior_parent_intent[3])
            )
        ):
            if prior_parent_intent[0] is not None and prior_parent_intent[0] != sell_client_order_id:
                raise ValueError(
                    "A managed Alpaca sell order cannot replace an unattached generation's client order identity."
                )
            _raise_if_managed_sell_intent_replay_conflicts(
                persisted_qty=prior_parent_intent[2],
                persisted_limit_price=prior_parent_intent[3],
                observed_qty=sell_order_qty,
                observed_limit_price=sell_order_limit_price,
            )
        if normalized_sell_alpaca_order_id is not None:
            _validate_managed_sell_generation_assignment(
                conn,
                position_id,
                normalized_sell_alpaca_order_id,
                submitted_qty=sell_order_qty,
                submitted_limit_price=sell_order_limit_price,
                validate_unattached_parent_intent=True,
            )
        cursor = conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET sell_client_order_id = ?,
                sell_alpaca_order_id = ?,
                sell_submitted_at = ?,
                sell_status = ?,
                sell_expires_at = ?,
                sell_observation_broker_updated_at = NULL,
                sell_observation_filled_qty = NULL,
                sell_order_qty = CASE
                    WHEN ? IS NOT NULL
                         AND (sell_alpaca_order_id IS NULL OR sell_alpaca_order_id = ?)
                    THEN COALESCE(sell_order_qty, ?)
                    ELSE COALESCE(?, sell_order_qty)
                END,
                sell_order_limit_price = CASE
                    WHEN ? IS NOT NULL
                         AND (sell_alpaca_order_id IS NULL OR sell_alpaca_order_id = ?)
                    THEN COALESCE(sell_order_limit_price, ?)
                    ELSE COALESCE(?, sell_order_limit_price)
                END,
                sell_renewal_count = sell_renewal_count + ?,
                sell_renewal_requested_at = NULL,
                sell_submission_retry_claimed_at = NULL,
                notes = COALESCE(?, notes),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                sell_client_order_id,
                normalized_sell_alpaca_order_id,
                sell_submitted_at,
                sell_status,
                sell_expires_at,
                sell_alpaca_order_id,
                sell_alpaca_order_id,
                sell_order_qty,
                sell_order_qty,
                sell_alpaca_order_id,
                sell_alpaca_order_id,
                sell_order_limit_price,
                sell_order_limit_price,
                1 if increment_renewal_count else 0,
                notes,
                position_id,
            ),
        )
        if cursor.rowcount == 1 and normalized_sell_alpaca_order_id is not None:
            persisted_sell_intent = conn.execute(
                """
                SELECT sell_order_qty, sell_order_limit_price
                FROM alpaca_managed_positions
                WHERE id = ?
                """,
                (position_id,),
            ).fetchone()
            if persisted_sell_intent is None:
                raise RuntimeError("Managed sell-order persistence lost its parent position.")
            persisted_sell_order_qty, persisted_sell_order_limit_price = _normalize_optional_managed_sell_intent(
                persisted_sell_intent[0],
                persisted_sell_intent[1],
            )
            record_alpaca_managed_sell_generation(
                conn,
                position_id,
                normalized_sell_alpaca_order_id,
                submitted_qty=persisted_sell_order_qty,
                submitted_limit_price=persisted_sell_order_limit_price,
                commit=False,
            )
    _commit_owned_transaction(conn)


def claim_alpaca_managed_initial_sell_intent(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_order_namespace: str | None,
    sell_client_order_id: str,
    expected_remaining_qty: float,
    expected_target_sell_price: float,
    notes: str,
) -> tuple[float, float] | None:
    """Atomically claim the first protective sell from an unchanged active row."""
    expected_remaining_qty, expected_target_sell_price = _normalize_required_managed_sell_intent(
        expected_remaining_qty,
        expected_target_sell_price,
    )
    owns_transaction = not conn.in_transaction
    row = _execute_returning_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET sell_order_namespace = COALESCE(?, sell_order_namespace),
            sell_client_order_id = ?,
            sell_alpaca_order_id = NULL,
            sell_submitted_at = NULL,
            sell_status = 'submission_pending',
            sell_expires_at = NULL,
            sell_observation_broker_updated_at = NULL,
            sell_observation_filled_qty = NULL,
            sell_order_qty = COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)),
            sell_order_limit_price = target_sell_price,
            sell_submission_retry_claimed_at = NULL,
            sell_renewal_requested_at = NULL,
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND closed_at IS NULL
          AND sell_client_order_id IS NULL
          AND sell_alpaca_order_id IS NULL
          AND (sell_status IS NULL OR LOWER(sell_status) = 'fractional_qty')
          AND {_managed_position_quantity_is_positive_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
          AND {_managed_quantity_matches_sql("COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0))", "?")}
          AND target_sell_price IS NOT NULL
          AND ABS(target_sell_price - ?) <= 0.00000001
        RETURNING sell_order_qty, target_sell_price
        """,
        (
            sell_order_namespace,
            sell_client_order_id,
            notes,
            position_id,
            expected_remaining_qty,
            expected_remaining_qty,
            expected_target_sell_price,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca initial-sell intent claim",
        decode=_decode_required_managed_sell_intent,
    )
    if owns_transaction:
        _commit_owned_transaction(conn)
    return row


def claim_alpaca_managed_initial_sell_submission(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_order_namespace: str | None,
    sell_client_order_id: str,
    expected_remaining_qty: float,
    expected_target_sell_price: float,
    claimed_at: str,
    expected_state_snapshot: AlpacaManagedSellRenewalSnapshot,
    notes: str,
) -> AlpacaManagedSellSubmissionClaim | None:
    """Atomically persist the first sell intent and its broker-request lease."""
    expected_remaining_qty, expected_target_sell_price = _normalize_required_managed_sell_intent(
        expected_remaining_qty,
        expected_target_sell_price,
    )
    owns_transaction = not conn.in_transaction
    claim = _execute_returning_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET state_revision = state_revision + 1,
            sell_order_namespace = COALESCE(?, sell_order_namespace),
            sell_client_order_id = ?,
            sell_alpaca_order_id = NULL,
            sell_submitted_at = NULL,
            sell_status = 'submission_pending',
            sell_expires_at = NULL,
            sell_observation_broker_updated_at = NULL,
            sell_observation_filled_qty = NULL,
            sell_order_qty = COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)),
            sell_order_limit_price = target_sell_price,
            sell_submission_retry_claimed_at = ?,
            sell_renewal_requested_at = NULL,
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND state_revision = ?
          AND closed_at IS NULL
          AND sell_client_order_id IS NULL
          AND sell_alpaca_order_id IS NULL
          AND (sell_status IS NULL OR LOWER(sell_status) = 'fractional_qty')
          AND buy_status IS ?
          AND filled_qty IS ?
          AND filled_avg_price IS ?
          AND target_sell_price IS ?
          AND COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)) IS ?
          AND sell_status IS ?
          AND sell_filled_qty IS ?
          AND sell_renewal_count = ?
          AND sell_renewal_requested_at IS ?
          AND {_managed_position_quantity_is_positive_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
          AND {_managed_quantity_matches_sql("COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0))", "?")}
          AND target_sell_price IS NOT NULL
          AND ABS(target_sell_price - ?) <= 0.00000001
        RETURNING sell_order_qty, sell_order_limit_price, state_revision, buy_status
        """,
        (
            sell_order_namespace,
            sell_client_order_id,
            claimed_at,
            notes,
            position_id,
            expected_state_snapshot.state_revision,
            expected_state_snapshot.buy_status,
            expected_state_snapshot.filled_qty,
            expected_state_snapshot.filled_avg_price,
            expected_state_snapshot.target_sell_price,
            expected_state_snapshot.remaining_qty,
            expected_state_snapshot.sell_status,
            expected_state_snapshot.sell_filled_qty,
            expected_state_snapshot.sell_renewal_count,
            expected_state_snapshot.requested_at,
            expected_remaining_qty,
            expected_remaining_qty,
            expected_target_sell_price,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca initial-sell submission claim",
        decode=lambda returned: _decode_managed_sell_submission_claim(
            returned,
            claimed_at=claimed_at,
        ),
    )
    if owns_transaction:
        _commit_owned_transaction(conn)
    return claim


def update_alpaca_managed_sell_status(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_status: str,
    sell_alpaca_order_id: str | None = None,
    sell_submitted_at: str | None = None,
    sell_expires_at: str | None = None,
    sell_renewal_requested_at: str | None = None,
    notes: str | None = None,
) -> None:
    sell_status = _managed_lifecycle_status(
        sell_status,
        field_name="Managed Alpaca sell status",
    )
    normalized_sell_alpaca_order_id = (
        None
        if sell_alpaca_order_id is None
        else _canonical_alpaca_order_id(
            sell_alpaca_order_id,
            field_name="A managed Alpaca sell status order ID",
        )
    )
    with _managed_accounting_composite_savepoint(
        conn,
        "update_alpaca_managed_sell_status",
    ):
        prior_parent_intent = conn.execute(
            """
            SELECT sell_alpaca_order_id, sell_order_qty, sell_order_limit_price
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
        if (
            prior_parent_intent is not None
            and normalized_sell_alpaca_order_id is not None
            and prior_parent_intent[0] is not None
            and prior_parent_intent[0] != normalized_sell_alpaca_order_id
        ):
            raise ValueError("A managed Alpaca sell status update cannot replace its broker order identity.")
        persisted_sell_order_qty: float | None = None
        persisted_sell_order_limit_price: float | None = None
        if prior_parent_intent is not None and normalized_sell_alpaca_order_id is not None:
            persisted_sell_order_qty, persisted_sell_order_limit_price = _normalize_optional_managed_sell_intent(
                prior_parent_intent[1],
                prior_parent_intent[2],
            )
            _validate_managed_sell_generation_assignment(
                conn,
                position_id,
                normalized_sell_alpaca_order_id,
                submitted_qty=persisted_sell_order_qty,
                submitted_limit_price=persisted_sell_order_limit_price,
                parent_intent=prior_parent_intent,
                validate_unattached_parent_intent=True,
            )
        cursor = conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET sell_status = ?,
                sell_alpaca_order_id = COALESCE(sell_alpaca_order_id, ?),
                sell_submitted_at = COALESCE(?, sell_submitted_at),
                sell_expires_at = COALESCE(?, sell_expires_at),
                sell_renewal_requested_at = COALESCE(?, sell_renewal_requested_at),
                notes = COALESCE(?, notes),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND (
                    ? IS NULL
                    OR sell_alpaca_order_id IS NULL
                    OR sell_alpaca_order_id = ?
                  )
            """,
            (
                sell_status,
                normalized_sell_alpaca_order_id,
                sell_submitted_at,
                sell_expires_at,
                sell_renewal_requested_at,
                notes,
                position_id,
                normalized_sell_alpaca_order_id,
                normalized_sell_alpaca_order_id,
            ),
        )
        if cursor.rowcount == 1 and normalized_sell_alpaca_order_id is not None:
            record_alpaca_managed_sell_generation(
                conn,
                position_id,
                normalized_sell_alpaca_order_id,
                submitted_qty=persisted_sell_order_qty,
                submitted_limit_price=persisted_sell_order_limit_price,
                commit=False,
            )
    _commit_owned_transaction(conn)


def update_alpaca_managed_sell_status_if_current(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_sell_client_order_id: str | None,
    sell_status: str,
    sell_alpaca_order_id: str | None = None,
    sell_submitted_at: str | None = None,
    sell_expires_at: str | None = None,
    sell_order_qty: float | None = None,
    sell_order_limit_price: float | None = None,
    observed_sell_filled_qty: float | None | object = _UNSET,
    sell_broker_updated_at: str | None | object = _UNSET,
    sell_renewal_requested_at: str | None = None,
    clear_sell_submission_retry_claim: bool = False,
    notes: str | None = None,
    expected_sell_status: str | None | object = _UNSET,
    expected_sell_alpaca_order_id: str | None | object = _UNSET,
    expected_sell_filled_qty: float | None | object = _UNSET,
    expected_sell_renewal_count: int | object = _UNSET,
    expected_sell_submission_retry_claimed_at: str | None | object = _UNSET,
    expected_sell_renewal_requested_at: str | None | object = _UNSET,
    expected_buy_status: str | None | object = _UNSET,
    expected_buy_filled_qty: float | None | object = _UNSET,
    expected_buy_filled_avg_price: float | None | object = _UNSET,
    expected_target_sell_price: float | None | object = _UNSET,
    expected_remaining_qty: float | None | object = _UNSET,
    expected_state_revision: int | object = _UNSET,
    allow_verified_replaced_successor: bool = False,
    allow_diagnostic_sell_intent_mismatch: bool = False,
) -> bool:
    """Record a broker observation only if its complete active state is current."""
    sell_status = _managed_lifecycle_status(
        sell_status,
        field_name="Managed Alpaca sell status",
    )
    if allow_diagnostic_sell_intent_mismatch and sell_status.lower() not in {
        *_ALPACA_MANAGED_SELL_DIAGNOSTIC_STATUSES,
        "pending_cancel",
    }:
        raise ValueError("A managed sell intent mismatch may only be authorized for an explicit diagnostic quarantine.")
    expected_values = (
        expected_sell_status,
        expected_sell_alpaca_order_id,
        expected_sell_filled_qty,
        expected_sell_renewal_count,
    )
    has_expected_snapshot = all(value is not _UNSET for value in expected_values)
    if any(value is not _UNSET for value in expected_values) and not has_expected_snapshot:
        raise ValueError("A managed sell status CAS requires the complete expected sell state.")
    observed_fill_qty_provided = observed_sell_filled_qty is not _UNSET
    normalized_observed_fill_qty: float | None = None
    if observed_fill_qty_provided:
        if observed_sell_filled_qty is None:
            raise ValueError("An observed managed sell fill quantity cannot be null.")
        normalized_observed_fill_qty = float(observed_sell_filled_qty)
        if not math.isfinite(normalized_observed_fill_qty) or normalized_observed_fill_qty < 0:
            raise ValueError("An observed managed sell fill quantity must be finite and non-negative.")
    sell_order_qty, sell_order_limit_price = _normalize_optional_managed_sell_intent(
        sell_order_qty,
        sell_order_limit_price,
    )

    snapshot_predicate = ""
    snapshot_params: tuple[object, ...] = ()
    if has_expected_snapshot:
        snapshot_predicate = """
          AND sell_status IS ?
          AND sell_alpaca_order_id IS ?
          AND sell_filled_qty IS ?
          AND sell_renewal_count = ?
        """
        snapshot_params = expected_values
    retry_claim_predicate = ""
    retry_claim_params: tuple[object, ...] = ()
    if expected_sell_submission_retry_claimed_at is not _UNSET:
        retry_claim_predicate = "AND sell_submission_retry_claimed_at IS ?"
        retry_claim_params = (expected_sell_submission_retry_claimed_at,)
    renewal_claim_predicate = ""
    renewal_claim_params: tuple[object, ...] = ()
    if expected_sell_renewal_requested_at is not _UNSET:
        renewal_claim_predicate = "AND sell_renewal_requested_at IS ?"
        renewal_claim_params = (expected_sell_renewal_requested_at,)
    expected_buy_values = (
        expected_buy_status,
        expected_buy_filled_qty,
        expected_buy_filled_avg_price,
        expected_target_sell_price,
        expected_remaining_qty,
    )
    has_expected_buy_snapshot = all(value is not _UNSET for value in expected_buy_values)
    if any(value is not _UNSET for value in expected_buy_values) and not has_expected_buy_snapshot:
        raise ValueError("A managed sell status CAS requires the complete expected parent-buy state.")
    buy_snapshot_predicate = ""
    buy_snapshot_params: tuple[object, ...] = ()
    if has_expected_buy_snapshot:
        buy_snapshot_predicate = """
          AND buy_status IS ?
          AND filled_qty IS ?
          AND filled_avg_price IS ?
          AND target_sell_price IS ?
          AND COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)) IS ?
        """
        buy_snapshot_params = expected_buy_values
    state_revision_predicate = ""
    state_revision_params: tuple[object, ...] = ()
    if expected_state_revision is not _UNSET:
        state_revision_predicate = "AND state_revision = ?"
        state_revision_params = (int(expected_state_revision),)
    broker_revision_predicate = ""
    broker_revision_params: tuple[object, ...] = ()
    broker_revision_assignment = "sell_observation_broker_updated_at"
    broker_revision_assignment_params: tuple[object, ...] = ()
    broker_fill_qty_assignment = "sell_observation_filled_qty"
    broker_fill_qty_assignment_params: tuple[object, ...] = ()
    broker_timestamp_provided = sell_broker_updated_at is not _UNSET
    observed_broker_updated_at = (
        _normalize_alpaca_broker_timestamp(None if sell_broker_updated_at is _UNSET else sell_broker_updated_at)
        if broker_timestamp_provided
        else None
    )
    if broker_timestamp_provided:
        # Compare and store the broker revision in the same guarded UPDATE. A
        # replacement successor owns a distinct revision stream; otherwise an
        # observation may mutate lifecycle state only at a strictly newer
        # broker revision (or while migrating a legacy NULL high-water mark).
        broker_closeable_statuses = tuple(sorted(_ALPACA_MANAGED_SELL_CLOSEABLE_STATUSES))
        broker_closeable_placeholders = ", ".join("?" for _ in broker_closeable_statuses)
        broker_revision_predicate = f"""
          AND (
                sell_observation_broker_updated_at IS NULL
                OR (? IS NOT NULL AND sell_alpaca_order_id IS NOT ?)
                OR (
                    ? IS NOT NULL
                    AND ? > sell_observation_broker_updated_at
                )
                OR (
                    ? IS NOT NULL
                    AND ? = sell_observation_broker_updated_at
                    AND LOWER(COALESCE(sell_status, '')) = 'pending_cancel'
                    AND LOWER(?) IN ({broker_closeable_placeholders})
                )
              )
        """
        broker_revision_params = (
            sell_alpaca_order_id,
            sell_alpaca_order_id,
            observed_broker_updated_at,
            observed_broker_updated_at,
            observed_broker_updated_at,
            observed_broker_updated_at,
            sell_status,
            *broker_closeable_statuses,
        )
        broker_revision_assignment = "?"
        broker_revision_assignment_params = (observed_broker_updated_at,)
        broker_fill_qty_assignment = "CASE WHEN ? THEN ? ELSE NULL END"
        broker_fill_qty_assignment_params = (
            int(observed_fill_qty_provided),
            normalized_observed_fill_qty,
        )
    # Terminal broker states and locally diagnosed accounting/identity states
    # are sticky.  A delayed observation may confirm the same state, advance
    # to a fill, or make the state more conservative, but it must not revive a
    # completed/canceled order or erase a manual-review quarantine.
    sticky_statuses = tuple(sorted(_ALPACA_MANAGED_SELL_STICKY_STATUSES))
    diagnostic_statuses = tuple(sorted(_ALPACA_MANAGED_SELL_DIAGNOSTIC_STATUSES))
    closeable_statuses = tuple(sorted(_ALPACA_MANAGED_SELL_CLOSEABLE_STATUSES))
    sticky_placeholders = ", ".join("?" for _ in sticky_statuses)
    diagnostic_placeholders = ", ".join("?" for _ in diagnostic_statuses)
    closeable_placeholders = ", ".join("?" for _ in closeable_statuses)
    transition_predicate = f"""
          AND (
                LOWER(COALESCE(sell_status, '')) NOT IN ({sticky_placeholders})
                OR LOWER(COALESCE(sell_status, '')) = LOWER(?)
                OR LOWER(?) = 'filled'
                OR LOWER(?) = 'pending_cancel'
                OR LOWER(?) IN ({diagnostic_placeholders})
                OR (
                    LOWER(COALESCE(sell_status, '')) = 'pending_cancel'
                    AND LOWER(?) IN ({closeable_placeholders})
                )
                OR (
                    ?
                    AND LOWER(COALESCE(sell_status, '')) IN ('pending_cancel', 'replaced')
                )
              )
    """
    transition_params: tuple[object, ...] = (
        *sticky_statuses,
        sell_status,
        sell_status,
        sell_status,
        sell_status,
        *diagnostic_statuses,
        sell_status,
        *closeable_statuses,
        1 if allow_verified_replaced_successor else 0,
    )
    owns_transaction = not conn.in_transaction
    with _managed_accounting_composite_savepoint(
        conn,
        "update_alpaca_managed_sell_status_if_current",
    ):
        prior_parent_intent = conn.execute(
            """
            SELECT sell_alpaca_order_id, sell_order_qty, sell_order_limit_price
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
        cursor = conn.execute(
            f"""
            UPDATE alpaca_managed_positions
            SET sell_status = ?,
                sell_alpaca_order_id = COALESCE(?, sell_alpaca_order_id),
                sell_submitted_at = COALESCE(?, sell_submitted_at),
                sell_expires_at = COALESCE(?, sell_expires_at),
                sell_order_qty = CASE
                    WHEN ? IS NOT NULL
                         AND (sell_alpaca_order_id IS NULL OR sell_alpaca_order_id = ?)
                    THEN COALESCE(sell_order_qty, ?)
                    ELSE COALESCE(?, sell_order_qty)
                END,
                sell_order_limit_price = CASE
                    WHEN ? IS NOT NULL
                         AND (sell_alpaca_order_id IS NULL OR sell_alpaca_order_id = ?)
                    THEN COALESCE(sell_order_limit_price, ?)
                    ELSE COALESCE(?, sell_order_limit_price)
                END,
                sell_observation_broker_updated_at = {broker_revision_assignment},
                sell_observation_filled_qty = {broker_fill_qty_assignment},
                sell_renewal_requested_at = COALESCE(?, sell_renewal_requested_at),
                sell_submission_retry_claimed_at = CASE WHEN ? THEN NULL ELSE sell_submission_retry_claimed_at END,
                notes = COALESCE(?, notes),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND closed_at IS NULL
              AND sell_client_order_id IS ?
              {snapshot_predicate}
              {retry_claim_predicate}
              {renewal_claim_predicate}
              {buy_snapshot_predicate}
              {state_revision_predicate}
              {broker_revision_predicate}
              {transition_predicate}
            """,
            (
                sell_status,
                sell_alpaca_order_id,
                sell_submitted_at,
                sell_expires_at,
                sell_alpaca_order_id,
                sell_alpaca_order_id,
                sell_order_qty,
                sell_order_qty,
                sell_alpaca_order_id,
                sell_alpaca_order_id,
                sell_order_limit_price,
                sell_order_limit_price,
                *broker_revision_assignment_params,
                *broker_fill_qty_assignment_params,
                sell_renewal_requested_at,
                1 if clear_sell_submission_retry_claim else 0,
                notes,
                position_id,
                expected_sell_client_order_id,
                *snapshot_params,
                *retry_claim_params,
                *renewal_claim_params,
                *buy_snapshot_params,
                *state_revision_params,
                *broker_revision_params,
                *transition_params,
            ),
        )
        if cursor.rowcount == 1 and sell_alpaca_order_id is not None:
            if not allow_diagnostic_sell_intent_mismatch:
                _validate_managed_sell_generation_assignment(
                    conn,
                    position_id,
                    sell_alpaca_order_id,
                    submitted_qty=sell_order_qty,
                    submitted_limit_price=sell_order_limit_price,
                    parent_intent=prior_parent_intent,
                    validate_unattached_parent_intent=True,
                )
            persisted_sell_intent = conn.execute(
                """
                SELECT sell_order_qty, sell_order_limit_price
                FROM alpaca_managed_positions
                WHERE id = ?
                """,
                (position_id,),
            ).fetchone()
            if persisted_sell_intent is None:
                raise RuntimeError("Managed sell observation lost its parent position.")
            persisted_sell_order_qty, persisted_sell_order_limit_price = _normalize_optional_managed_sell_intent(
                persisted_sell_intent[0],
                persisted_sell_intent[1],
            )
            record_alpaca_managed_sell_generation(
                conn,
                position_id,
                sell_alpaca_order_id,
                broker_updated_at=(
                    observed_broker_updated_at
                    if broker_timestamp_provided
                    and normalized_observed_fill_qty is not None
                    and normalized_observed_fill_qty == 0
                    else _UNSET
                ),
                submitted_qty=persisted_sell_order_qty,
                submitted_limit_price=persisted_sell_order_limit_price,
                commit=False,
            )
    observation_applied = cursor.rowcount == 1
    observation_is_idempotent = False
    if not observation_applied and broker_timestamp_provided:
        # A retry of an already persisted revision is current without being a
        # write. This is important after a process interruption between status
        # persistence and downstream fill reconciliation, and avoids advancing
        # state_revision for harmless replays. Older conflicting observations
        # fail these exact-result predicates and remain rejected.
        observation_is_idempotent = bool(
            _execute_returning_owned_operation_step(
                conn,
                f"""
                SELECT 1
                FROM alpaca_managed_positions
                WHERE id = ?
                  AND closed_at IS NULL
                  AND sell_client_order_id IS ?
                  {snapshot_predicate}
                  {retry_claim_predicate}
                  {renewal_claim_predicate}
                  {buy_snapshot_predicate}
                  {state_revision_predicate}
                  {transition_predicate}
                  AND sell_observation_broker_updated_at IS ?
                  AND (? = 0 OR sell_observation_filled_qty IS ?)
                  AND LOWER(COALESCE(sell_status, '')) = LOWER(?)
                  AND (? IS NULL OR sell_alpaca_order_id IS ?)
                  AND (? IS NULL OR sell_submitted_at IS ?)
                  AND (? IS NULL OR sell_expires_at IS ?)
                  AND (? IS NULL OR sell_order_qty IS ?)
                  AND (? IS NULL OR sell_order_limit_price IS ?)
                  AND (? IS NULL OR sell_renewal_requested_at IS ?)
                  AND (? = 0 OR sell_submission_retry_claimed_at IS NULL)
                """,
                (
                    position_id,
                    expected_sell_client_order_id,
                    *snapshot_params,
                    *retry_claim_params,
                    *renewal_claim_params,
                    *buy_snapshot_params,
                    *state_revision_params,
                    *transition_params,
                    observed_broker_updated_at,
                    int(observed_fill_qty_provided),
                    normalized_observed_fill_qty,
                    sell_status,
                    sell_alpaca_order_id,
                    sell_alpaca_order_id,
                    sell_submitted_at,
                    sell_submitted_at,
                    sell_expires_at,
                    sell_expires_at,
                    sell_order_qty,
                    sell_order_qty,
                    sell_order_limit_price,
                    sell_order_limit_price,
                    sell_renewal_requested_at,
                    sell_renewal_requested_at,
                    1 if clear_sell_submission_retry_claim else 0,
                ),
                owns_transaction=owns_transaction,
                operation="managed Alpaca fenced sell-status replay check",
                decode=lambda _row: True,
            )
        )
    if owns_transaction:
        _commit_owned_transaction(conn)
    return observation_applied or observation_is_idempotent


def alpaca_managed_sell_sticky_status_if_current(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_sell_client_order_id: str | None,
) -> str | None:
    """Return the active generation's sticky status, if any.

    ``update_alpaca_managed_sell_status_if_current`` deliberately returns the
    same false value for a stale compare-and-swap and a forbidden transition
    out of a terminal or manual-review state. Reconciliation callers need to
    distinguish those outcomes so they do not report an unresolved live state
    as a harmless superseded observation.
    """
    row = conn.execute(
        """
        SELECT sell_status
        FROM alpaca_managed_positions
        WHERE id = ?
          AND closed_at IS NULL
          AND sell_client_order_id IS ?
        """,
        (position_id, expected_sell_client_order_id),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    status = str(row[0]).lower()
    return status if status in _ALPACA_MANAGED_SELL_STICKY_STATUSES else None


def claim_alpaca_managed_sell_renewal(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    sell_alpaca_order_id: str,
    expected_target_sell_price: float,
    expected_remaining_qty: float,
    requested_at: str,
    reclaim_before: str,
    notes: str,
) -> bool:
    """Atomically claim cancellation of a managed sell for renewal.

    An abandoned claim may be reclaimed after the caller-provided lease cutoff.
    The generation and active-position predicates prevent a stale reconciler
    from canceling an order whose managed state has already advanced.
    """
    return (
        claim_alpaca_managed_sell_renewal_with_revision(
            conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=sell_alpaca_order_id,
            expected_target_sell_price=expected_target_sell_price,
            expected_remaining_qty=expected_remaining_qty,
            requested_at=requested_at,
            reclaim_before=reclaim_before,
            notes=notes,
        )
        is not None
    )


def claim_alpaca_managed_sell_renewal_with_revision(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    sell_alpaca_order_id: str,
    expected_target_sell_price: float,
    expected_remaining_qty: float,
    requested_at: str,
    reclaim_before: str,
    notes: str,
    expected_renewal_snapshot: AlpacaManagedSellRenewalSnapshot | None = None,
) -> AlpacaManagedSellRenewalClaim | None:
    """Claim renewal and return the exact lease and state-revision fence."""
    if (
        not isinstance(expected_remaining_qty, bool)
        and isinstance(expected_remaining_qty, Number)
        and not isinstance(expected_remaining_qty, (complex, np.complexfloating))
        and expected_remaining_qty == 0
    ):
        _normalize_managed_target_sell_price(expected_target_sell_price)
        return None
    expected_remaining_qty, expected_target_sell_price = _normalize_required_managed_sell_intent(
        expected_remaining_qty,
        expected_target_sell_price,
    )
    renewal_snapshot_predicate = ""
    renewal_snapshot_params: tuple[object, ...] = ()
    if expected_renewal_snapshot is not None:
        renewal_snapshot_predicate = """
          AND state_revision = ?
          AND buy_status IS ?
          AND filled_qty IS ?
          AND filled_avg_price IS ?
          AND sell_status IS ?
          AND sell_alpaca_order_id IS ?
          AND sell_filled_qty IS ?
          AND sell_renewal_count = ?
          AND sell_renewal_requested_at IS ?
          AND target_sell_price IS ?
          AND COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)) IS ?
        """
        renewal_snapshot_params = (
            expected_renewal_snapshot.state_revision,
            expected_renewal_snapshot.buy_status,
            expected_renewal_snapshot.filled_qty,
            expected_renewal_snapshot.filled_avg_price,
            expected_renewal_snapshot.sell_status,
            expected_renewal_snapshot.sell_alpaca_order_id,
            expected_renewal_snapshot.sell_filled_qty,
            expected_renewal_snapshot.sell_renewal_count,
            expected_renewal_snapshot.requested_at,
            expected_renewal_snapshot.target_sell_price,
            expected_renewal_snapshot.remaining_qty,
        )
    owns_transaction = not conn.in_transaction
    claim = _execute_returning_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET state_revision = state_revision + 1,
            sell_status = 'pending_cancel',
            sell_renewal_requested_at = ?,
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND sell_client_order_id = ?
          AND sell_alpaca_order_id = ?
          AND target_sell_price IS NOT NULL
          AND ABS(target_sell_price - ?) <= 0.00000001
          AND closed_at IS NULL
          AND {_managed_position_quantity_is_positive_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
          AND {_managed_quantity_matches_sql("COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0))", "?")}
          AND LOWER(sell_status) IN
              ('accepted', 'accepted_for_bidding', 'new', 'partially_filled', 'pending_new',
               'pending_cancel')
          AND (
                sell_renewal_requested_at IS NULL
                OR julianday(sell_renewal_requested_at) IS NULL
                OR julianday(sell_renewal_requested_at) <= julianday(?)
              )
          {renewal_snapshot_predicate}
        RETURNING state_revision, sell_filled_qty, sell_renewal_count,
                  sell_renewal_requested_at
        """,
        (
            requested_at,
            notes,
            position_id,
            sell_client_order_id,
            sell_alpaca_order_id,
            expected_target_sell_price,
            expected_remaining_qty,
            expected_remaining_qty,
            reclaim_before,
            *renewal_snapshot_params,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca sell-renewal claim",
        decode=lambda returned: AlpacaManagedSellRenewalClaim(
            state_revision=int(returned[0]),  # type: ignore[index]
            sell_filled_qty=None if returned[1] is None else float(returned[1]),  # type: ignore[index]
            sell_renewal_count=int(returned[2]),  # type: ignore[index]
            requested_at=str(returned[3]),  # type: ignore[index]
        ),
    )
    _commit_owned_transaction(conn)
    return claim


def alpaca_managed_sell_renewal_snapshot_if_current(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_sell_client_order_id: str | None,
    expected_sell_alpaca_order_id: str | None,
    expected_sell_renewal_requested_at: str | None | object = _UNSET,
) -> AlpacaManagedSellRenewalSnapshot | None:
    """Load a continuation snapshot only while the caller's lease is exact."""
    renewal_predicate = ""
    renewal_params: tuple[object, ...] = ()
    if expected_sell_renewal_requested_at is not _UNSET:
        renewal_predicate = "AND sell_renewal_requested_at IS ?"
        renewal_params = (expected_sell_renewal_requested_at,)
    row = conn.execute(
        f"""
        SELECT state_revision, buy_status, filled_qty, filled_avg_price,
               target_sell_price,
               COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)),
               sell_status, sell_alpaca_order_id, sell_filled_qty,
               sell_renewal_count, sell_renewal_requested_at
        FROM alpaca_managed_positions
        WHERE id = ?
          AND closed_at IS NULL
          AND sell_client_order_id IS ?
          AND sell_alpaca_order_id IS ?
          {renewal_predicate}
        """,
        (
            position_id,
            expected_sell_client_order_id,
            expected_sell_alpaca_order_id,
            *renewal_params,
        ),
    ).fetchone()
    if row is None:
        return None
    return AlpacaManagedSellRenewalSnapshot(
        state_revision=int(row[0]),
        buy_status=None if row[1] is None else str(row[1]),
        filled_qty=None if row[2] is None else float(row[2]),
        filled_avg_price=None if row[3] is None else float(row[3]),
        target_sell_price=None if row[4] is None else float(row[4]),
        remaining_qty=None if row[5] is None else float(row[5]),
        sell_status=None if row[6] is None else str(row[6]),
        sell_alpaca_order_id=None if row[7] is None else str(row[7]),
        sell_filled_qty=None if row[8] is None else float(row[8]),
        sell_renewal_count=int(row[9]),
        requested_at=None if row[10] is None else str(row[10]),
    )


def _claim_alpaca_managed_sell_replacement(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    prior_sell_client_order_id: str,
    prior_sell_alpaca_order_id: str,
    prior_renewal_count: int,
    replacement_sell_client_order_id: str,
    requested_remaining_qty: float,
    expected_target_sell_price: float,
    notes: str,
    expected_renewal_snapshot: AlpacaManagedSellRenewalSnapshot | None = None,
    submission_claimed_at: str | None = None,
) -> AlpacaManagedSellReplacementSubmissionClaim | None:
    """Claim a replacement, optionally retaining its final-submission lease."""
    requested_remaining_qty, expected_target_sell_price = _normalize_required_managed_sell_intent(
        requested_remaining_qty,
        expected_target_sell_price,
    )
    renewal_snapshot_predicate = ""
    renewal_snapshot_params: tuple[object, ...] = ()
    if expected_renewal_snapshot is not None:
        renewal_snapshot_predicate = """
          AND state_revision = ?
          AND buy_status IS ?
          AND filled_qty IS ?
          AND filled_avg_price IS ?
          AND sell_status IS ?
          AND sell_alpaca_order_id IS ?
          AND sell_filled_qty IS ?
          AND sell_renewal_count = ?
          AND sell_renewal_requested_at IS ?
          AND target_sell_price IS ?
          AND COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)) IS ?
        """
        renewal_snapshot_params = (
            expected_renewal_snapshot.state_revision,
            expected_renewal_snapshot.buy_status,
            expected_renewal_snapshot.filled_qty,
            expected_renewal_snapshot.filled_avg_price,
            expected_renewal_snapshot.sell_status,
            expected_renewal_snapshot.sell_alpaca_order_id,
            expected_renewal_snapshot.sell_filled_qty,
            expected_renewal_snapshot.sell_renewal_count,
            expected_renewal_snapshot.requested_at,
            expected_renewal_snapshot.target_sell_price,
            expected_renewal_snapshot.remaining_qty,
        )
    with _managed_accounting_composite_savepoint(
        conn,
        "claim_alpaca_managed_sell_replacement",
    ):
        cursor = conn.execute(
            f"""
            UPDATE alpaca_managed_positions
            SET state_revision = state_revision + 1,
                sell_client_order_id = ?,
                sell_alpaca_order_id = NULL,
                sell_submitted_at = NULL,
                sell_status = 'submission_pending',
                sell_expires_at = NULL,
                sell_observation_broker_updated_at = NULL,
                sell_observation_filled_qty = NULL,
                sell_order_qty = COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)),
                sell_order_limit_price = target_sell_price,
                sell_renewal_count = sell_renewal_count + 1,
                remaining_qty = COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)),
                closed_sell_shortfall_reopen_pending = 0,
                sell_renewal_requested_at = NULL,
                sell_submission_retry_claimed_at = ?,
                notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND sell_client_order_id = ?
              AND sell_alpaca_order_id = ?
              AND sell_renewal_count = ?
              AND closed_at IS NULL
              AND {_managed_position_quantity_is_positive_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
              AND {_managed_quantity_matches_sql("COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0))", "?")}
              AND target_sell_price IS NOT NULL
              AND ABS(target_sell_price - ?) <= 0.00000001
              AND LOWER(sell_status) IN
                  ('canceled', 'expired', 'filled')
              {renewal_snapshot_predicate}
            RETURNING remaining_qty, sell_order_limit_price
            """,
            (
                replacement_sell_client_order_id,
                submission_claimed_at,
                notes,
                position_id,
                prior_sell_client_order_id,
                prior_sell_alpaca_order_id,
                prior_renewal_count,
                requested_remaining_qty,
                requested_remaining_qty,
                expected_target_sell_price,
                *renewal_snapshot_params,
            ),
        )
        returned_row = cursor.fetchone()
        remaining_qty = None if returned_row is None else _decode_required_managed_sell_intent(returned_row)[0]
        claim: AlpacaManagedSellReplacementSubmissionClaim | None = None
        if remaining_qty is not None:
            record_alpaca_managed_sell_generation(
                conn,
                position_id,
                prior_sell_alpaca_order_id,
                commit=False,
            )
            state_row = conn.execute(
                """
                SELECT state_revision
                FROM alpaca_managed_positions
                WHERE id = ?
                  AND sell_client_order_id = ?
                  AND sell_alpaca_order_id IS NULL
                  AND LOWER(sell_status) = 'submission_pending'
                  AND sell_submission_retry_claimed_at IS ?
                """,
                (position_id, replacement_sell_client_order_id, submission_claimed_at),
            ).fetchone()
            if state_row is None:
                raise RuntimeError("Managed sell replacement claim lost its local submission state.")
            claim = AlpacaManagedSellReplacementSubmissionClaim(
                remaining_qty=remaining_qty,
                state_revision=int(state_row[0]),
                claimed_at=submission_claimed_at or "",
            )
    _commit_owned_transaction(conn)
    return claim


def claim_alpaca_managed_sell_replacement(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    prior_sell_client_order_id: str,
    prior_sell_alpaca_order_id: str,
    prior_renewal_count: int,
    replacement_sell_client_order_id: str,
    requested_remaining_qty: float,
    expected_target_sell_price: float,
    notes: str,
    expected_renewal_snapshot: AlpacaManagedSellRenewalSnapshot | None = None,
) -> float | None:
    """Claim a replacement and return the authoritative remaining quantity."""
    claim = _claim_alpaca_managed_sell_replacement(
        conn,
        position_id,
        prior_sell_client_order_id=prior_sell_client_order_id,
        prior_sell_alpaca_order_id=prior_sell_alpaca_order_id,
        prior_renewal_count=prior_renewal_count,
        replacement_sell_client_order_id=replacement_sell_client_order_id,
        requested_remaining_qty=requested_remaining_qty,
        expected_target_sell_price=expected_target_sell_price,
        notes=notes,
        expected_renewal_snapshot=expected_renewal_snapshot,
    )
    return None if claim is None else claim.remaining_qty


def claim_alpaca_managed_sell_replacement_submission(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    prior_sell_client_order_id: str,
    prior_sell_alpaca_order_id: str,
    prior_renewal_count: int,
    replacement_sell_client_order_id: str,
    requested_remaining_qty: float,
    expected_target_sell_price: float,
    notes: str,
    expected_renewal_snapshot: AlpacaManagedSellRenewalSnapshot,
    claimed_at: str,
) -> AlpacaManagedSellReplacementSubmissionClaim | None:
    """Atomically claim a renewed replacement and its final submission lease."""
    return _claim_alpaca_managed_sell_replacement(
        conn,
        position_id,
        prior_sell_client_order_id=prior_sell_client_order_id,
        prior_sell_alpaca_order_id=prior_sell_alpaca_order_id,
        prior_renewal_count=prior_renewal_count,
        replacement_sell_client_order_id=replacement_sell_client_order_id,
        requested_remaining_qty=requested_remaining_qty,
        expected_target_sell_price=expected_target_sell_price,
        notes=notes,
        expected_renewal_snapshot=expected_renewal_snapshot,
        submission_claimed_at=claimed_at,
    )


def claim_alpaca_managed_sell_submission_retry(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    expected_target_sell_price: float,
    claimed_at: str,
    reclaim_before: str,
    notes: str,
) -> tuple[float, float] | None:
    """Claim a confirmed-missing sell and return its current quantity and target."""
    claim = claim_alpaca_managed_sell_submission_retry_with_revision(
        conn,
        position_id,
        sell_client_order_id=sell_client_order_id,
        expected_target_sell_price=expected_target_sell_price,
        claimed_at=claimed_at,
        reclaim_before=reclaim_before,
        notes=notes,
    )
    return None if claim is None else (claim.remaining_qty, claim.target_sell_price)


def claim_alpaca_managed_sell_submission_fence(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    claimed_at: str,
    expected_remaining_qty: float,
    expected_target_sell_price: float,
    notes: str,
) -> int | None:
    """Lease and fence an ordinary persisted sell intent before broker checks.

    The broker request cannot share SQLite's transaction.  Retaining this
    token through the POST lets the response writer distinguish an unchanged
    submission generation from a concurrently corrected position and route a
    stale accepted order through cancellation/accounting quarantine.
    """
    expected_remaining_qty, expected_target_sell_price = _normalize_required_managed_sell_intent(
        expected_remaining_qty,
        expected_target_sell_price,
    )
    owns_transaction = not conn.in_transaction
    row = _execute_returning_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET state_revision = state_revision + 1,
            sell_submission_retry_claimed_at = ?,
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND closed_at IS NULL
          AND sell_client_order_id = ?
          AND sell_alpaca_order_id IS NULL
          AND LOWER(sell_status) = 'submission_pending'
          AND sell_submission_retry_claimed_at IS NULL
          AND {_managed_position_quantity_is_positive_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
          AND {_managed_quantity_matches_sql("COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0))", "?")}
          AND sell_order_qty IS NOT NULL
          AND {_managed_quantity_matches_sql("sell_order_qty", "?")}
          AND target_sell_price IS NOT NULL
          AND ABS(target_sell_price - ?) <= 0.00000001
          AND sell_order_limit_price IS NOT NULL
          AND ABS(sell_order_limit_price - ?) <= 0.00000001
        RETURNING state_revision
        """,
        (
            claimed_at,
            notes,
            position_id,
            sell_client_order_id,
            expected_remaining_qty,
            expected_remaining_qty,
            expected_remaining_qty,
            expected_remaining_qty,
            expected_target_sell_price,
            expected_target_sell_price,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca sell-submission fence claim",
        decode=lambda returned: int(returned[0]),  # type: ignore[index]
    )
    _commit_owned_transaction(conn)
    return row


def claim_alpaca_managed_sell_submission_retry_with_revision(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    expected_target_sell_price: float,
    claimed_at: str,
    reclaim_before: str,
    notes: str,
) -> AlpacaManagedSellSubmissionClaim | None:
    """Claim a confirmed-missing sell and rebase it to current protection."""
    expected_target_sell_price = canonical_alpaca_limit_price(
        expected_target_sell_price,
        field_name="An expected managed sell target price",
    )
    owns_transaction = not conn.in_transaction
    claim = _execute_returning_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET state_revision = state_revision + 1,
            sell_status = 'submission_retrying',
            sell_order_qty = COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)),
            sell_order_limit_price = target_sell_price,
            sell_submission_retry_claimed_at = ?,
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND sell_client_order_id = ?
          AND sell_alpaca_order_id IS NULL
          AND closed_at IS NULL
          AND LOWER(sell_status) IN
              ('submission_pending', 'submission_unknown', 'submission_not_found', 'submission_retrying')
          AND (
                sell_submission_retry_claimed_at IS NULL
                OR julianday(sell_submission_retry_claimed_at) IS NULL
                OR julianday(sell_submission_retry_claimed_at) <= julianday(?)
              )
          AND {_managed_position_quantity_is_positive_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
          AND target_sell_price IS NOT NULL
          AND target_sell_price > 0
          AND ABS(target_sell_price - ?) <= 0.00000001
        RETURNING sell_order_qty, sell_order_limit_price, state_revision, buy_status
        """,
        (
            claimed_at,
            notes,
            position_id,
            sell_client_order_id,
            reclaim_before,
            expected_target_sell_price,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca sell-submission retry claim",
        decode=lambda returned: _decode_managed_sell_submission_claim(
            returned,
            claimed_at=claimed_at,
        ),
    )
    _commit_owned_transaction(conn)
    return claim


def confirm_alpaca_managed_sell_submission(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    claimed_at: str,
    expected_sell_status: str,
    expected_state_revision: int,
    expected_remaining_qty: float,
    expected_target_sell_price: float,
) -> int | None:
    """Fence any leased sell immediately before POST if protection is unchanged."""
    expected_remaining_qty, expected_target_sell_price = _normalize_required_managed_sell_intent(
        expected_remaining_qty,
        expected_target_sell_price,
    )
    owns_transaction = not conn.in_transaction
    row = _execute_returning_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET state_revision = state_revision + 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND state_revision = ?
          AND closed_at IS NULL
          AND sell_client_order_id = ?
          AND sell_alpaca_order_id IS NULL
          AND LOWER(sell_status) = LOWER(?)
          AND sell_submission_retry_claimed_at IS ?
          AND {_managed_position_quantity_is_positive_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
          AND {_managed_quantity_matches_sql("COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0))", "?")}
          AND sell_order_qty IS NOT NULL
          AND {_managed_quantity_matches_sql("sell_order_qty", "?")}
          AND target_sell_price IS NOT NULL
          AND ABS(target_sell_price - ?) <= 0.00000001
          AND sell_order_limit_price IS NOT NULL
          AND ABS(sell_order_limit_price - ?) <= 0.00000001
        RETURNING state_revision
        """,
        (
            position_id,
            int(expected_state_revision),
            sell_client_order_id,
            expected_sell_status,
            claimed_at,
            expected_remaining_qty,
            expected_remaining_qty,
            expected_remaining_qty,
            expected_remaining_qty,
            expected_target_sell_price,
            expected_target_sell_price,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca sell-submission confirmation",
        decode=lambda returned: int(returned[0]),  # type: ignore[index]
    )
    _commit_owned_transaction(conn)
    return row


def confirm_alpaca_managed_sell_submission_retry(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    claimed_at: str,
    expected_state_revision: int,
    expected_remaining_qty: float,
    expected_target_sell_price: float,
) -> int | None:
    """Fence a retry immediately before POST if its claimed protection is unchanged."""
    return confirm_alpaca_managed_sell_submission(
        conn,
        position_id,
        sell_client_order_id=sell_client_order_id,
        claimed_at=claimed_at,
        expected_sell_status="submission_retrying",
        expected_state_revision=expected_state_revision,
        expected_remaining_qty=expected_remaining_qty,
        expected_target_sell_price=expected_target_sell_price,
    )


def release_alpaca_managed_sell_submission_fence(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    claimed_at: str,
    expected_sell_status: str,
    expected_state_revision: int | None = None,
    allow_current_intent_rebase: bool = False,
    sell_status: str,
    notes: str,
) -> bool:
    """Release a leased submission generation after a proven pre-POST abort.

    A neutral ``submission_not_found`` release may rebase from the current
    parent economics when the exact claim token is still owned.  State-based
    diagnoses such as ``fractional_qty`` must retain their revision fence so a
    stale decision cannot overwrite a newer parent quantity.
    """
    sell_status = _managed_lifecycle_status(
        sell_status,
        field_name="Released managed Alpaca sell status",
    )
    if allow_current_intent_rebase and sell_status.lower() != "submission_not_found":
        raise ValueError("Only a neutral managed sell release may rebase a newer parent intent.")
    release_state_revision = None if allow_current_intent_rebase else expected_state_revision
    owns_transaction = not conn.in_transaction
    intent = _execute_returning_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET sell_status = ?,
            sell_order_qty = COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)),
            sell_order_limit_price = target_sell_price,
            sell_submission_retry_claimed_at = NULL,
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND sell_client_order_id = ?
          AND sell_alpaca_order_id IS NULL
          AND LOWER(sell_status) = LOWER(?)
          AND sell_submission_retry_claimed_at IS ?
          AND (? IS NULL OR state_revision = ?)
          AND (
                ? = 0
                OR {_managed_position_quantity_is_positive_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
              )
        RETURNING sell_order_qty, sell_order_limit_price
        """,
        (
            sell_status,
            notes,
            position_id,
            sell_client_order_id,
            expected_sell_status,
            claimed_at,
            release_state_revision,
            release_state_revision,
            int(allow_current_intent_rebase),
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca sell-submission fence release",
        decode=_decode_required_managed_sell_intent,
    )
    released_balanced_intent = False
    if intent is None and allow_current_intent_rebase:
        # A historical-lineage fill can balance the parent while a no-POST
        # diagnosis still owns this exact lease.  Clear that neutral lease but
        # retain its last positive frozen intent as audit history; zero is not
        # a valid broker order quantity and must not be promoted into it.
        cursor = _execute_owned_operation_step(
            conn,
            f"""
            UPDATE alpaca_managed_positions
            SET sell_status = ?,
                sell_submission_retry_claimed_at = NULL,
                notes = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND sell_client_order_id = ?
              AND sell_alpaca_order_id IS NULL
              AND LOWER(sell_status) = LOWER(?)
              AND sell_submission_retry_claimed_at IS ?
              AND {_managed_position_quantity_is_negligible_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
            """,
            (
                sell_status,
                notes,
                position_id,
                sell_client_order_id,
                expected_sell_status,
                claimed_at,
            ),
            owns_transaction=owns_transaction,
            operation="balanced managed Alpaca sell-submission fence release",
        )
        released_balanced_intent = cursor.rowcount == 1
    _commit_owned_transaction(conn)
    return intent is not None or released_balanced_intent


def release_alpaca_managed_sell_submission_claim_if_adopted(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    sell_alpaca_order_id: str,
    claimed_at: str,
    expected_state_revision: int,
) -> bool:
    """Release a stale claim only while its exact broker order remains current."""
    owns_transaction = not conn.in_transaction
    cursor = _execute_owned_operation_step(
        conn,
        """
        UPDATE alpaca_managed_positions
        SET sell_submission_retry_claimed_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND state_revision = ?
          AND sell_client_order_id = ?
          AND sell_alpaca_order_id = ?
          AND sell_submission_retry_claimed_at IS ?
        """,
        (
            position_id,
            int(expected_state_revision),
            sell_client_order_id,
            sell_alpaca_order_id,
            claimed_at,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca adopted-sell claim release",
    )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def attach_alpaca_managed_sell_order_if_current(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_sell_client_order_id: str | None,
    expected_sell_alpaca_order_id: str | None,
    expected_renewal_count: int,
    sell_renewal_count: int,
    sell_client_order_id: str,
    sell_alpaca_order_id: str | None,
    sell_submitted_at: str | None,
    sell_status: str,
    sell_expires_at: str | None,
    sell_order_qty: float | None = None,
    sell_order_limit_price: float | None = None,
    sell_renewal_requested_at: str | None = None,
    notes: str | None = None,
    expected_sell_submission_retry_claimed_at: str | None | object = _UNSET,
    expected_state_snapshot: AlpacaManagedSellRenewalSnapshot | None = None,
    allow_diagnostic_sell_intent_mismatch: bool = False,
) -> bool:
    """Attach a discovered open sell only while the caller's snapshot is current."""
    sell_status = _managed_lifecycle_status(
        sell_status,
        field_name="Attached managed Alpaca sell status",
    )
    if allow_diagnostic_sell_intent_mismatch and sell_status.lower() not in {
        *_ALPACA_MANAGED_SELL_DIAGNOSTIC_STATUSES,
        "pending_cancel",
    }:
        raise ValueError("A managed sell intent mismatch may only be attached as an explicit diagnostic quarantine.")
    sell_order_qty, sell_order_limit_price = _normalize_optional_managed_sell_intent(
        sell_order_qty,
        sell_order_limit_price,
    )
    normalized_sell_alpaca_order_id = (
        None
        if sell_alpaca_order_id is None
        else _canonical_alpaca_order_id(
            sell_alpaca_order_id,
            field_name="An attached managed Alpaca sell order ID",
        )
    )
    if allow_diagnostic_sell_intent_mismatch and normalized_sell_alpaca_order_id is None:
        raise ValueError("A diagnostic managed sell intent mismatch attachment requires a broker order ID.")
    retry_claim_predicate = ""
    retry_claim_params: tuple[object, ...] = ()
    if expected_sell_submission_retry_claimed_at is not _UNSET:
        retry_claim_predicate = "AND sell_submission_retry_claimed_at IS ?"
        retry_claim_params = (expected_sell_submission_retry_claimed_at,)
    snapshot_predicate = ""
    snapshot_params: tuple[object, ...] = ()
    if expected_state_snapshot is not None:
        snapshot_predicate = """
          AND state_revision = ?
          AND buy_status IS ?
          AND filled_qty IS ?
          AND filled_avg_price IS ?
          AND target_sell_price IS ?
          AND COALESCE(remaining_qty, filled_qty - COALESCE(sold_qty, 0)) IS ?
          AND sell_status IS ?
          AND sell_alpaca_order_id IS ?
          AND sell_filled_qty IS ?
          AND sell_renewal_count = ?
          AND sell_renewal_requested_at IS ?
        """
        snapshot_params = (
            expected_state_snapshot.state_revision,
            expected_state_snapshot.buy_status,
            expected_state_snapshot.filled_qty,
            expected_state_snapshot.filled_avg_price,
            expected_state_snapshot.target_sell_price,
            expected_state_snapshot.remaining_qty,
            expected_state_snapshot.sell_status,
            expected_state_snapshot.sell_alpaca_order_id,
            expected_state_snapshot.sell_filled_qty,
            expected_state_snapshot.sell_renewal_count,
            expected_state_snapshot.requested_at,
        )
    with _managed_accounting_composite_savepoint(
        conn,
        "attach_alpaca_managed_sell_order_if_current",
    ):
        prior_parent_intent = conn.execute(
            """
            SELECT sell_alpaca_order_id, sell_order_qty, sell_order_limit_price
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
        cursor = conn.execute(
            f"""
            UPDATE alpaca_managed_positions
            SET sell_client_order_id = ?,
                sell_alpaca_order_id = ?,
                sell_submitted_at = ?,
                sell_status = ?,
                sell_expires_at = ?,
                sell_observation_broker_updated_at = NULL,
                sell_observation_filled_qty = NULL,
                sell_order_qty = CASE
                    WHEN ? IS NOT NULL
                         AND (sell_alpaca_order_id IS NULL OR sell_alpaca_order_id = ?)
                    THEN COALESCE(sell_order_qty, ?)
                    ELSE COALESCE(?, sell_order_qty)
                END,
                sell_order_limit_price = CASE
                    WHEN ? IS NOT NULL
                         AND (sell_alpaca_order_id IS NULL OR sell_alpaca_order_id = ?)
                    THEN COALESCE(sell_order_limit_price, ?)
                    ELSE COALESCE(?, sell_order_limit_price)
                END,
                sell_renewal_count = ?,
                sell_renewal_requested_at = ?,
                sell_submission_retry_claimed_at = NULL,
                notes = COALESCE(?, notes),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
              AND sell_client_order_id IS ?
              AND sell_alpaca_order_id IS ?
              AND sell_renewal_count = ?
              AND ? >= sell_renewal_count
              {retry_claim_predicate}
              {snapshot_predicate}
              AND closed_at IS NULL
              AND {_managed_position_quantity_is_positive_sql(_MANAGED_REMAINING_QUANTITY_SQL)}
            """,
            (
                sell_client_order_id,
                normalized_sell_alpaca_order_id,
                sell_submitted_at,
                sell_status,
                sell_expires_at,
                normalized_sell_alpaca_order_id,
                normalized_sell_alpaca_order_id,
                sell_order_qty,
                sell_order_qty,
                normalized_sell_alpaca_order_id,
                normalized_sell_alpaca_order_id,
                sell_order_limit_price,
                sell_order_limit_price,
                sell_renewal_count,
                sell_renewal_requested_at,
                notes,
                position_id,
                expected_sell_client_order_id,
                expected_sell_alpaca_order_id,
                expected_renewal_count,
                sell_renewal_count,
                *retry_claim_params,
                *snapshot_params,
            ),
        )
        if cursor.rowcount == 1 and normalized_sell_alpaca_order_id is not None:
            if not allow_diagnostic_sell_intent_mismatch:
                _validate_managed_sell_generation_assignment(
                    conn,
                    position_id,
                    normalized_sell_alpaca_order_id,
                    submitted_qty=sell_order_qty,
                    submitted_limit_price=sell_order_limit_price,
                    parent_intent=prior_parent_intent,
                    validate_unattached_parent_intent=True,
                )
            persisted_sell_intent = conn.execute(
                """
                SELECT sell_order_qty, sell_order_limit_price
                FROM alpaca_managed_positions
                WHERE id = ?
                """,
                (position_id,),
            ).fetchone()
            if persisted_sell_intent is None:
                raise RuntimeError("Managed sell-order attachment lost its parent position.")
            persisted_sell_order_qty, persisted_sell_order_limit_price = _normalize_optional_managed_sell_intent(
                persisted_sell_intent[0],
                persisted_sell_intent[1],
            )
            record_alpaca_managed_sell_generation(
                conn,
                position_id,
                normalized_sell_alpaca_order_id,
                submitted_qty=persisted_sell_order_qty,
                submitted_limit_price=persisted_sell_order_limit_price,
                commit=False,
            )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def _mark_alpaca_managed_sell_filled(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_status: str,
    sell_filled_qty: float,
    sell_filled_avg_price: float,
    sell_filled_at: str | None,
    sell_broker_updated_at: str | None = None,
    sell_alpaca_order_id: str | None = None,
    sell_submitted_at: str | None = None,
    sell_expires_at: str | None = None,
    notes: str | None = None,
    expected_sell_client_order_id: str | None = None,
    expected_sell_status: str | None | object = _UNSET,
    expected_sell_alpaca_order_id: str | None | object = _UNSET,
    expected_sell_filled_qty: float | None | object = _UNSET,
    expected_sell_renewal_count: int | object = _UNSET,
    expected_sell_renewal_requested_at: str | None | object = _UNSET,
    expected_state_revision: int | object = _UNSET,
    observed_alpaca_asset_id: str | None = None,
    preserve_pending_cancel: bool = False,
) -> tuple[float, bool]:
    """Record an order's cumulative fills and return the remaining buy quantity."""
    sell_status = _managed_lifecycle_status(
        sell_status,
        field_name="Filled managed Alpaca sell status",
    )
    if type(preserve_pending_cancel) is not bool:
        raise ValueError("A managed sell fill cancellation-fence flag must be boolean.")
    if preserve_pending_cancel and sell_status != "pending_cancel":
        raise ValueError("Preserving a managed sell cancellation fence requires pending_cancel status.")
    observed_alpaca_asset_id = _normalized_optional_alpaca_asset_id(
        observed_alpaca_asset_id,
        field_name="Attributed managed Alpaca sell-fill asset ID",
    )
    expected_sell_values = (
        expected_sell_status,
        expected_sell_alpaca_order_id,
        expected_sell_filled_qty,
        expected_sell_renewal_count,
    )
    has_expected_sell_snapshot = all(value is not _UNSET for value in expected_sell_values)
    if any(value is not _UNSET for value in expected_sell_values) and not has_expected_sell_snapshot:
        raise ValueError("A managed sell fill CAS requires the complete expected sell state.")
    if expected_sell_alpaca_order_id not in (_UNSET, None):
        _canonical_alpaca_order_id(
            expected_sell_alpaca_order_id,
            field_name="An expected Alpaca sell fill order ID",
        )
    order_key = (
        f"legacy-{position_id}"
        if sell_alpaca_order_id is None
        else _canonical_alpaca_order_id(
            sell_alpaca_order_id,
            field_name="An Alpaca sell fill order ID",
        )
    )
    observed_qty = float(sell_filled_qty)
    observed_avg_price = float(sell_filled_avg_price)
    if not math.isfinite(observed_qty) or observed_qty < 0:
        raise ValueError("An Alpaca sell fill quantity must be finite and non-negative.")
    if not math.isfinite(observed_avg_price) or (observed_qty > 0 and observed_avg_price <= 0):
        raise ValueError("An Alpaca sell fill average price must be finite and positive.")
    observed_value = observed_qty * observed_avg_price
    if not math.isfinite(observed_value) or observed_value < 0 or (observed_qty > 0 and observed_value <= 0):
        raise ValueError("An Alpaca sell fill value must be finite and non-negative.")
    observed_broker_updated_at = _normalize_alpaca_broker_timestamp(sell_broker_updated_at)
    savepoint = "mark_alpaca_managed_sell_filled"
    owns_transaction = not conn.in_transaction
    savepoint_active = False
    try:
        if owns_transaction:
            # Serialize the read-before-write ledger decision at transaction
            # start.  Two deferred transactions can both acquire read snapshots
            # and then deadlock while upgrading; SQLite reports that upgrade as
            # SQLITE_BUSY without honoring the busy timeout.  BEGIN IMMEDIATE
            # lets the second owned caller wait before it observes prior fills.
            conn.execute("BEGIN IMMEDIATE")
        conn.execute(f"SAVEPOINT {savepoint}")
        savepoint_active = True
        _validate_managed_sell_generation_assignment(
            conn,
            position_id,
            order_key,
            submitted_qty=None,
            submitted_limit_price=None,
        )
        prior_fill = conn.execute(
            """
            SELECT filled_qty, filled_value
            FROM alpaca_managed_sell_fills
            WHERE managed_position_id = ? AND alpaca_order_id = ?
            """,
            (position_id, order_key),
        ).fetchone()
        prior_qty = 0.0 if prior_fill is None else float(prior_fill[0])
        prior_value = 0.0 if prior_fill is None else float(prior_fill[1])
        fill_quantity_scale = max(abs(prior_qty), abs(observed_qty))
        prior_unit_price = prior_value / prior_qty if prior_qty > 0.0 else None
        fill_mark_price = max(
            observed_avg_price,
            0.0 if prior_unit_price is None else prior_unit_price,
        )
        fill_notional_scale = max(
            abs(prior_value),
            abs(observed_value),
            fill_quantity_scale * fill_mark_price,
        )
        fill_quantity_tolerance = min(
            managed_quantity_tolerance(fill_quantity_scale),
            (
                managed_residual_notional_tolerance(fill_notional_scale) / fill_mark_price
                if math.isfinite(fill_notional_scale) and math.isfinite(fill_mark_price) and fill_mark_price > 0.0
                else 0.0
            ),
        )
        fill_value_tolerance = managed_value_reconciliation_tolerance(max(abs(prior_value), abs(observed_value)))
        # The conditional conflict update is the regression check.  Its write
        # lock remains held through the totals and managed-position update, so
        # two connections cannot both validate against an older fill and then
        # commit observations out of order. Alpaca may correct an average fill
        # price or cumulative quantity. Once a ledger row has a broker update
        # timestamp, conflicting accounting is authoritative only at a strictly
        # newer revision; otherwise a delayed pre-correction response could
        # overwrite newer accounting. Legacy rows with no revision retain their
        # historical monotonic-quantity behavior until a timestamp is seeded.
        fill_cursor = conn.execute(
            f"""
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value, broker_updated_at)
            SELECT ?, ?, ?, ?, ?
            WHERE EXISTS (
                SELECT 1
                FROM alpaca_managed_positions AS managed
                WHERE managed.id = ?
                  AND (
                        managed.sell_observation_broker_updated_at IS NULL
                        OR (
                            ? IS NOT NULL
                            AND managed.sell_alpaca_order_id IS NOT ?
                        )
                        OR (
                            ? IS NOT NULL
                            AND ? > managed.sell_observation_broker_updated_at
                        )
                        OR (
                            ? IS NOT NULL
                            AND ? = managed.sell_observation_broker_updated_at
                            AND managed.sell_observation_filled_qty IS NOT NULL
                            AND ABS(managed.sell_observation_filled_qty - {observed_qty!r})
                                <= {fill_quantity_tolerance!r}
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM alpaca_managed_sell_fills AS persisted
                            WHERE persisted.managed_position_id = managed.id
                              AND persisted.alpaca_order_id = ?
                              AND (
                                    (
                                        ABS(persisted.filled_qty - {observed_qty!r})
                                            <= {fill_quantity_tolerance!r}
                                        AND ABS(persisted.filled_value - {observed_value!r})
                                            <= {fill_value_tolerance!r}
                                    )
                                    OR (
                                        persisted.filled_qty = 0
                                        AND persisted.filled_value = 0
                                        AND persisted.broker_updated_at IS NULL
                                        AND managed.sell_observation_filled_qty IS NULL
                                    )
                                  )
                        )
                      )
            )
            ON CONFLICT(managed_position_id, alpaca_order_id) DO UPDATE SET
                filled_qty = excluded.filled_qty,
                filled_value = excluded.filled_value,
                broker_updated_at = CASE
                    WHEN excluded.broker_updated_at IS NULL
                        THEN alpaca_managed_sell_fills.broker_updated_at
                    WHEN alpaca_managed_sell_fills.broker_updated_at IS NULL
                        THEN excluded.broker_updated_at
                    WHEN excluded.broker_updated_at > alpaca_managed_sell_fills.broker_updated_at
                        THEN excluded.broker_updated_at
                    ELSE alpaca_managed_sell_fills.broker_updated_at
                END
            WHERE (
                    excluded.filled_qty > alpaca_managed_sell_fills.filled_qty
                        + {fill_quantity_tolerance!r}
                    AND (
                          alpaca_managed_sell_fills.broker_updated_at IS NULL
                          OR (
                               excluded.broker_updated_at IS NOT NULL
                               AND excluded.broker_updated_at
                                   > alpaca_managed_sell_fills.broker_updated_at
                             )
                        )
                  )
               OR (
                    ABS(excluded.filled_qty - alpaca_managed_sell_fills.filled_qty)
                        <= {fill_quantity_tolerance!r}
                    AND ABS(excluded.filled_value - alpaca_managed_sell_fills.filled_value)
                        > {fill_value_tolerance!r}
                    AND excluded.broker_updated_at IS NOT NULL
                    AND alpaca_managed_sell_fills.broker_updated_at IS NOT NULL
                    AND excluded.broker_updated_at > alpaca_managed_sell_fills.broker_updated_at
                  )
               OR (
                    ABS(excluded.filled_qty - alpaca_managed_sell_fills.filled_qty)
                        <= {fill_quantity_tolerance!r}
                    AND ABS(excluded.filled_value - alpaca_managed_sell_fills.filled_value)
                        <= {fill_value_tolerance!r}
                    AND excluded.broker_updated_at IS NOT NULL
                    AND alpaca_managed_sell_fills.broker_updated_at IS NULL
                  )
            """,
            (
                position_id,
                order_key,
                observed_qty,
                observed_value,
                observed_broker_updated_at,
                position_id,
                sell_alpaca_order_id,
                sell_alpaca_order_id,
                observed_broker_updated_at,
                observed_broker_updated_at,
                observed_broker_updated_at,
                observed_broker_updated_at,
                order_key,
            ),
        )
        fill_changed = fill_cursor.rowcount == 1
        accepted_newer_revision = False
        is_idempotent_replay = False
        persisted_broker_updated_at: str | None = None
        if not fill_changed:
            persisted_fill = conn.execute(
                """
                SELECT filled_qty, filled_value, broker_updated_at
                FROM alpaca_managed_sell_fills
                WHERE managed_position_id = ? AND alpaca_order_id = ?
                """,
                (position_id, order_key),
            ).fetchone()
            is_idempotent_replay = bool(
                persisted_fill is not None
                and abs(float(persisted_fill[0]) - observed_qty) <= fill_quantity_tolerance
                and abs(float(persisted_fill[1]) - observed_value) <= fill_value_tolerance
            )
            persisted_qty = None if persisted_fill is None else float(persisted_fill[0])
            persisted_broker_updated_at = (
                None if persisted_fill is None or persisted_fill[2] is None else str(persisted_fill[2])
            )
            if persisted_qty is not None and observed_qty < persisted_qty - fill_quantity_tolerance:
                raise SellFillQuantityRegressionError(
                    "Alpaca sell filled quantity moved backwards for a managed order."
                )
            if is_idempotent_replay and observed_broker_updated_at is not None:
                # An unchanged later response establishes a safe high-water
                # mark for any subsequent same-quantity price correction,
                # including ledger rows created before this column existed.
                revision_cursor = conn.execute(
                    """
                    UPDATE alpaca_managed_sell_fills
                    SET broker_updated_at = ?
                    WHERE managed_position_id = ? AND alpaca_order_id = ?
                      AND (broker_updated_at IS NULL OR broker_updated_at < ?)
                    """,
                    (
                        observed_broker_updated_at,
                        position_id,
                        order_key,
                        observed_broker_updated_at,
                    ),
                )
                accepted_newer_revision = revision_cursor.rowcount == 1
                if accepted_newer_revision:
                    persisted_broker_updated_at = observed_broker_updated_at

        observation_mutated = fill_changed or accepted_newer_revision
        idempotent_revision_is_current = bool(
            is_idempotent_replay
            and (
                (observed_broker_updated_at is None and persisted_broker_updated_at is None)
                or observed_broker_updated_at == persisted_broker_updated_at
            )
        )
        observation_is_current = observation_mutated or idempotent_revision_is_current

        row = conn.execute(
            """
            SELECT filled_qty, filled_avg_price, target_sell_price,
                   buy_order_limit_price, sell_order_limit_price
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Managed Alpaca position {position_id} does not exist.")

        buy_qty = None if row[0] is None else float(row[0])
        buy_avg_price = None if row[1] is None else float(row[1])
        target_sell_price = None if row[2] is None else float(row[2])
        buy_limit_price = None if row[3] is None else float(row[3])
        sell_limit_price = None if row[4] is None else float(row[4])
        if (
            buy_qty is None
            or buy_avg_price is None
            or not math.isfinite(buy_qty)
            or not math.isfinite(buy_avg_price)
            or buy_qty <= 0
            or buy_avg_price <= 0
        ):
            raise ValueError(f"Managed Alpaca position {position_id} is missing a valid filled buy quantity or price.")

        ledger_rows = conn.execute(
            """
            SELECT alpaca_order_id, filled_qty, filled_value
            FROM alpaca_managed_sell_fills
            WHERE managed_position_id = ?
            """,
            (position_id,),
        ).fetchall()
        for ledger_order_id, ledger_qty, ledger_value in ledger_rows:
            _canonical_alpaca_order_id(
                ledger_order_id,
                field_name="A persisted Alpaca sell fill order ID",
            )
            ledger_qty_value = float(ledger_qty)
            ledger_value_value = float(ledger_value)
            if (
                not math.isfinite(ledger_qty_value)
                or not math.isfinite(ledger_value_value)
                or ledger_qty_value < 0
                or ledger_value_value < 0
                or ((ledger_qty_value == 0) != (ledger_value_value == 0))
            ):
                raise ValueError("Persisted Alpaca sell fill economics are invalid.")
            if ledger_qty_value > 0:
                unit_price = ledger_value_value / ledger_qty_value
                if not math.isfinite(unit_price) or unit_price <= 0:
                    raise ValueError("Persisted Alpaca sell fill unit price is invalid.")
        totals = conn.execute(
            """
            SELECT COALESCE(SUM(filled_qty), 0), COALESCE(SUM(filled_value), 0)
            FROM alpaca_managed_sell_fills
            WHERE managed_position_id = ?
            """,
            (position_id,),
        ).fetchone()
        sold_qty = float(totals[0])
        sold_value = float(totals[1])
        if not math.isfinite(sold_qty) or not math.isfinite(sold_value) or sold_qty < 0 or sold_value < 0:
            raise ValueError("Cumulative Alpaca sell accounting must remain finite and non-negative.")
        # Keep an overfill visible to the reconciler instead of silently treating it
        # as a completed managed position.  A negative remaining quantity requires
        # manual review; automatically closing it would conceal a possible short.
        remaining_qty = buy_qty - sold_qty
        cumulative_sell_avg_price = sold_value / sold_qty if sold_qty > 0 else None
        quantity_scale = max(abs(remaining_qty), buy_qty, sold_qty)
        residual_is_negligible = _managed_accounting_residual_is_negligible(
            remaining_qty,
            quantity_scale=quantity_scale,
            mark_prices=(
                buy_avg_price,
                target_sell_price,
                buy_limit_price,
                sell_limit_price,
                cumulative_sell_avg_price,
            ),
            value_scale=sold_value,
        )
        accounting_overfill = remaining_qty < 0.0 and not residual_is_negligible
        matched_qty = min(sold_qty, buy_qty)
        realized_pl: float | None = None
        realized_pl_pct: float | None = None
        if not accounting_overfill and _managed_accounting_quantity_is_positive(
            matched_qty,
            quantity_scale=quantity_scale,
            mark_prices=(buy_avg_price, target_sell_price, cumulative_sell_avg_price),
            value_scale=sold_value,
        ):
            realized_pl, realized_pl_pct = _managed_realized_pl_values(
                sold_value=sold_value,
                matched_qty=matched_qty,
                buy_price=buy_avg_price,
            )
        derived_accounting_values = (
            remaining_qty,
            matched_qty,
            cumulative_sell_avg_price,
            realized_pl,
            realized_pl_pct,
        )
        if any(value is not None and not math.isfinite(value) for value in derived_accounting_values):
            raise ValueError("Derived Alpaca sell accounting must remain finite.")
        current = conn.execute(
            """
            SELECT sell_client_order_id, sell_status, closed_at, symbol, alpaca_asset_id,
                   sell_alpaca_order_id, sell_filled_qty, sell_renewal_count,
                   sell_renewal_requested_at, state_revision,
                   sell_observation_broker_updated_at, sell_observation_filled_qty
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
        current_asset_id = (
            None
            if current is None
            else _canonical_optional_alpaca_asset_id(
                current[4],
                field_name="Persisted managed Alpaca sell-fill asset ID",
            )
        )
        if (
            current is not None
            and current_asset_id is not None
            and observed_alpaca_asset_id is not None
            and current_asset_id != observed_alpaca_asset_id
        ):
            raise ValueError("Attributed Alpaca sell fill changed the managed position's stable asset identity.")
        effective_asset_id = (
            observed_alpaca_asset_id
            if observed_alpaca_asset_id is not None
            else None
            if current_asset_id is None
            else current_asset_id
        )
        effective_symbol = (
            ""
            if current is None
            else _canonical_managed_symbol(
                current[3],
                field_name="Persisted managed Alpaca sell-fill symbol",
            )
        )
        closed_reopen_is_identity_fenced = effective_asset_id is not None
        generation_is_current = bool(
            current is not None
            and current[2] is None
            and (expected_sell_client_order_id is None or current[0] == expected_sell_client_order_id)
            and (
                not has_expected_sell_snapshot
                or (
                    current[1] == expected_sell_status
                    and current[5] == expected_sell_alpaca_order_id
                    and current[6] == expected_sell_filled_qty
                    and current[7] == expected_sell_renewal_count
                )
            )
            and (expected_sell_renewal_requested_at is _UNSET or current[8] == expected_sell_renewal_requested_at)
            and (expected_state_revision is _UNSET or current[9] == int(expected_state_revision))
        )
        current_sell_status = str(current[1] or "").lower()
        observed_sell_status = str(sell_status).lower()
        cancellation_fence_is_current = bool(
            preserve_pending_cancel and current_sell_status == "pending_cancel" and current[2] is None
        )
        observed_status_can_advance = bool(
            current_sell_status not in _ALPACA_MANAGED_SELL_STICKY_STATUSES
            or observed_sell_status == current_sell_status
            or observed_sell_status == "filled"
            or observed_sell_status in _ALPACA_MANAGED_SELL_DIAGNOSTIC_STATUSES
        )
        persisted_status_broker_updated_at = None if current[10] is None else str(current[10])
        observed_order_is_new = bool(sell_alpaca_order_id is not None and current[5] != sell_alpaca_order_id)
        status_revision_is_current = bool(
            persisted_status_broker_updated_at is None
            or observed_order_is_new
            or (
                observed_broker_updated_at is not None
                and observed_broker_updated_at > persisted_status_broker_updated_at
            )
            or (
                observed_broker_updated_at == persisted_status_broker_updated_at
                and (
                    observed_sell_status == current_sell_status
                    or (
                        current_sell_status == "pending_cancel"
                        and observed_sell_status in _ALPACA_MANAGED_SELL_CLOSEABLE_STATUSES
                    )
                )
            )
        )
        current_observation_metadata_can_apply = bool(
            generation_is_current and observation_mutated and status_revision_is_current
        )
        status_observation_can_apply = bool(current_observation_metadata_can_apply and observed_status_can_advance)
        late_fill_invalidated_close = bool(
            fill_changed and current is not None and current[2] is not None and not residual_is_negligible
        )
        late_fill_identity_is_unfenced = late_fill_invalidated_close and not closed_reopen_is_identity_fenced
        late_fill_active_owner_conflict = bool(
            late_fill_invalidated_close
            and closed_reopen_is_identity_fenced
            and conn.execute(
                """
                SELECT 1
                FROM alpaca_managed_positions AS other
                WHERE other.id != ?
                  AND other.closed_at IS NULL
                  AND (
                        UPPER(other.symbol) = UPPER(?)
                        OR (? IS NOT NULL AND other.alpaca_asset_id = ?)
                      )
                LIMIT 1
                """,
                (position_id, effective_symbol, effective_asset_id, effective_asset_id),
            ).fetchone()
            is not None
        )
        newly_observed_overfill = fill_changed and accounting_overfill

        if observation_mutated:
            conn.execute(
                """
            UPDATE alpaca_managed_positions
            SET sell_status = ?,
                alpaca_asset_id = COALESCE(alpaca_asset_id, ?),
                sell_alpaca_order_id = COALESCE(?, sell_alpaca_order_id),
                sell_submitted_at = COALESCE(?, sell_submitted_at),
                sell_expires_at = COALESCE(?, sell_expires_at),
                sell_observation_broker_updated_at = CASE
                    WHEN ? THEN ? ELSE sell_observation_broker_updated_at END,
                sell_observation_filled_qty = CASE
                    WHEN ? THEN ? ELSE sell_observation_filled_qty END,
                sell_filled_qty = ?,
                sell_filled_avg_price = ?,
                sell_filled_at = CASE
                    WHEN ? AND ? IS NOT NULL THEN ? ELSE sell_filled_at END,
                sold_qty = ?,
                sold_value = ?,
                remaining_qty = ?,
                realized_pl = ?,
                realized_pl_pct = ?,
                notes = COALESCE(?, notes),
                closed_at = CASE WHEN ? THEN NULL ELSE closed_at END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
                (
                    (
                        "pending_cancel"
                        if cancellation_fence_is_current
                        else "position_quantity_mismatch"
                        if late_fill_active_owner_conflict or late_fill_identity_is_unfenced
                        else "late_fill_after_close"
                        if late_fill_invalidated_close
                        else "quantity_mismatch"
                        if newly_observed_overfill
                        else sell_status
                        if status_observation_can_apply
                        else current[1]
                    ),
                    observed_alpaca_asset_id,
                    sell_alpaca_order_id if status_observation_can_apply else None,
                    sell_submitted_at if status_observation_can_apply else None,
                    sell_expires_at if status_observation_can_apply else None,
                    1 if status_observation_can_apply else 0,
                    observed_broker_updated_at,
                    1 if status_observation_can_apply else 0,
                    observed_qty,
                    sold_qty,
                    cumulative_sell_avg_price,
                    1 if current_observation_metadata_can_apply else 0,
                    sell_filled_at,
                    sell_filled_at,
                    sold_qty,
                    sold_value,
                    remaining_qty,
                    realized_pl,
                    realized_pl_pct,
                    (
                        None
                        if cancellation_fence_is_current
                        else "a newly discovered sell fill could not safely reopen its closed position because the "
                        "attributed lineage has no single stable asset identity"
                        if late_fill_identity_is_unfenced
                        else "a newly discovered sell fill belongs to a closed position whose symbol/asset has a newer "
                        "active managed owner; accounting was corrected while the old row remained closed"
                        if late_fill_active_owner_conflict
                        else "a newly discovered sell fill invalidated a previously closed managed position"
                        if late_fill_invalidated_close
                        else "cumulative managed sell fills exceed the managed buy quantity"
                        if newly_observed_overfill
                        else notes
                        if status_observation_can_apply
                        else None
                    ),
                    (
                        1
                        if late_fill_invalidated_close
                        and not late_fill_active_owner_conflict
                        and not late_fill_identity_is_unfenced
                        else 0
                    ),
                    position_id,
                ),
            )
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        savepoint_active = False
    except BaseException as exc:
        if savepoint_active:
            _rollback_and_release_savepoint(conn, savepoint, exc)
        if owns_transaction:
            _rollback_owned_operation_if_active(
                conn,
                exc,
                operation="managed Alpaca sell-fill accounting",
            )
        raise
    if owns_transaction:
        _commit_owned_transaction(conn)
    return remaining_qty, generation_is_current and observation_is_current


def mark_alpaca_managed_sell_filled(
    conn: sqlite3.Connection,
    position_id: int,
    **kwargs: object,
) -> float:
    remaining_qty, _ = _mark_alpaca_managed_sell_filled(conn, position_id, **kwargs)
    return remaining_qty


def mark_alpaca_managed_sell_filled_if_current(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_sell_client_order_id: str,
    **kwargs: object,
) -> tuple[float, bool]:
    return _mark_alpaca_managed_sell_filled(
        conn,
        position_id,
        expected_sell_client_order_id=expected_sell_client_order_id,
        **kwargs,
    )


def close_alpaca_managed_position(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    closed_at: str | None,
    notes: str | None = None,
) -> None:
    normalized_closed_at = _normalize_alpaca_closure_timestamp(closed_at)
    owns_transaction = not conn.in_transaction
    _execute_owned_operation_step(
        conn,
        """
        UPDATE alpaca_managed_positions
        SET closed_at = COALESCE(?, CURRENT_TIMESTAMP),
            closed_sell_shortfall_reopen_pending = 0,
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (normalized_closed_at, notes, position_id),
        owns_transaction=owns_transaction,
        operation="managed Alpaca position closure",
    )
    _commit_owned_transaction(conn)


def close_alpaca_managed_position_if_current_and_complete(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    expected_sell_client_order_id: str,
    expected_sell_alpaca_order_id: str,
    closed_at: str | None,
    expected_sell_status: str = "filled",
    expected_sell_filled_qty: float | None | object = _UNSET,
    expected_sell_renewal_count: int | object = _UNSET,
    expected_sell_renewal_requested_at: str | None | object = _UNSET,
    expected_state_revision: int | object = _UNSET,
    notes: str | None = None,
) -> bool:
    """Close only an exact active generation with confirmed inactive exposure.

    The broker fill and the close are persisted in separate transactions.  A
    complete state predicate therefore prevents a stale closer from hiding a
    concurrently renewed or diagnostically quarantined managed position. An
    in-flight sell retry also retains ownership until its broker call resolves.
    The default remains a broker-filled leaf; reconciliation may pass another
    terminal status only after independently auditing every attributed sell
    lineage and confirming that no leaf can still execute.
    """
    normalized_status = str(expected_sell_status).lower()
    if normalized_status not in _ALPACA_MANAGED_SELL_CLOSEABLE_STATUSES:
        raise ValueError("Managed Alpaca position closure requires an inactive sell status.")
    normalized_closed_at = _normalize_alpaca_closure_timestamp(closed_at)
    snapshot_predicate = ""
    snapshot_params: tuple[object, ...] = ()
    if expected_sell_filled_qty is not _UNSET:
        snapshot_predicate += "AND sell_filled_qty IS ?\n"
        snapshot_params += (expected_sell_filled_qty,)
    if expected_sell_renewal_count is not _UNSET:
        snapshot_predicate += "AND sell_renewal_count = ?\n"
        snapshot_params += (expected_sell_renewal_count,)
    if expected_sell_renewal_requested_at is not _UNSET:
        snapshot_predicate += "AND sell_renewal_requested_at IS ?\n"
        snapshot_params += (expected_sell_renewal_requested_at,)
    if expected_state_revision is not _UNSET:
        snapshot_predicate += "AND state_revision = ?\n"
        snapshot_params += (int(expected_state_revision),)
    owns_transaction = not conn.in_transaction
    cursor = _execute_owned_operation_step(
        conn,
        f"""
        UPDATE alpaca_managed_positions
        SET closed_at = COALESCE(?, CURRENT_TIMESTAMP),
            closed_sell_shortfall_reopen_pending = 0,
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND closed_at IS NULL
          AND sell_client_order_id = ?
          AND sell_alpaca_order_id = ?
          AND LOWER(sell_status) = ?
          AND {_managed_position_quantity_is_negligible_sql("COALESCE(remaining_qty, 1)")}
          AND sell_submission_retry_claimed_at IS NULL
          {snapshot_predicate}
        """,
        (
            normalized_closed_at,
            notes,
            position_id,
            expected_sell_client_order_id,
            expected_sell_alpaca_order_id,
            normalized_status,
            *snapshot_params,
        ),
        owns_transaction=owns_transaction,
        operation="managed Alpaca fenced position closure",
    )
    _commit_owned_transaction(conn)
    return cursor.rowcount == 1


def save_equity_records(conn: sqlite3.Connection, records: list[dict]) -> None:
    if not records:
        return
    rows = []
    for record in records:
        risk_free_return = None if pd.isna(record["risk_free_return"]) else float(record["risk_free_return"])
        row = (
            str(record["asset_symbol"]),
            str(record["signal_symbol"]),
            _canonical_integrity_float(record["buy_rsi"]),
            _canonical_integrity_float(record["profit_target_multiple"]),
            str(record["date"]),
            _canonical_integrity_float(record["equity"]),
            _canonical_integrity_float(record["daily_return"]),
            risk_free_return,
            int(record["in_position"]),
            str(record["action_executed"]),
            str(record["pending_action"]),
            int(record["trades_executed"]),
        )
        rows.append(
            (
                *row,
                _strategy_equity_integrity_digest(
                    asset_symbol=row[0],
                    signal_symbol=row[1],
                    buy_rsi=row[2],
                    profit_target_multiple=row[3],
                    date=row[4],
                    equity=row[5],
                    daily_return=row[6],
                    risk_free_return=row[7],
                    in_position=row[8],
                    action_executed=row[9],
                    pending_action=row[10],
                    trades_executed=row[11],
                ),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO strategy_equity
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, date, equity,
         daily_return, risk_free_return, in_position, action_executed,
         pending_action, trades_executed, integrity_digest)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def clear_equity_records(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> None:
    conn.execute(
        """
        DELETE FROM strategy_equity
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND buy_rsi = ?
          AND profit_target_multiple = ?
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    )


def prune_non_best_equity_records(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> None:
    conn.execute(
        """
        DELETE FROM strategy_equity
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND NOT (buy_rsi = ? AND profit_target_multiple = ?)
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    )


def _sample_variance(count: int, m2: float) -> float:
    if count <= 1:
        return np.nan
    return max(m2, 0.0) / (count - 1)


def _merge_centered_moments(
    left_count: int,
    left_mean: float,
    left_m2: float,
    right_count: int,
    right_mean: float,
    right_m2: float,
) -> tuple[int, float, float]:
    """Merge two variance accumulators using the parallel Welford formula."""
    if left_count == 0:
        return right_count, right_mean, right_m2
    if right_count == 0:
        return left_count, left_mean, left_m2
    count = left_count + right_count
    delta = right_mean - left_mean
    mean = left_mean + delta * right_count / count
    m2 = left_m2 + right_m2 + delta * delta * left_count * right_count / count
    return count, mean, m2


def _update_centered_moment(count: int, mean: float, m2: float, value: float) -> tuple[int, float, float]:
    # Match the sequential Welford update used by the optimized kernel.  The
    # algebraically equivalent one-item parallel merge rounds differently and
    # can flip SQL best-strategy ordering by one ULP after reconstructing the
    # retained equity curve.
    new_count = count + 1
    delta = value - mean
    new_mean = mean + delta / new_count
    new_m2 = m2 + delta * (value - new_mean)
    return new_count, new_mean, new_m2


def _rollup_metrics(rollup: SummaryRollup) -> dict[str, float | None]:
    if rollup.first_equity is None or rollup.last_equity is None or rollup.return_count <= 0:
        return {
            "total_return": None,
            "cagr": None,
            "annualized_vol": None,
            "sharpe": None,
            "kelly_fraction": None,
            "max_drawdown": rollup.max_drawdown,
            "hit_rate": None,
        }

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        if rollup.first_equity > 0:
            equity_ratio = np.float64(rollup.last_equity) / np.float64(rollup.first_equity)
            total_return = equity_ratio - 1.0
        else:
            equity_ratio = np.float64(np.nan)
            total_return = np.float64(np.nan)
        cagr = (
            np.float_power(equity_ratio, np.float64(252 / rollup.return_count)) - 1.0
            if rollup.last_equity > 0
            else np.float64(np.nan)
        )
        return_variance = _sample_variance(rollup.return_count, rollup.return_m2)
        annualized_vol = np.sqrt(return_variance) * np.sqrt(252) if pd.notna(return_variance) else np.nan
        kelly_fraction = (
            np.float64(rollup.return_mean) / np.float64(return_variance)
            if pd.notna(return_variance) and return_variance > 0
            else np.nan
        )
        hit_rate = rollup.positive_return_count / rollup.return_count

        excess_variance = _sample_variance(rollup.excess_return_count, rollup.excess_return_m2)
        if pd.notna(excess_variance) and excess_variance > 0:
            sharpe = np.sqrt(252) * np.float64(rollup.excess_return_mean) / np.sqrt(excess_variance)
        else:
            sharpe = np.nan

    total_return = _require_representable_metric("total return", total_return)
    cagr = _require_representable_metric("CAGR", cagr)
    _require_representable_metric(
        "return variance",
        return_variance,
        allow_nan=rollup.return_count < 2,
    )
    annualized_vol = _require_representable_metric(
        "annualized volatility",
        annualized_vol,
        allow_nan=rollup.return_count < 2,
    )
    kelly_fraction = _require_representable_metric(
        "Kelly fraction",
        kelly_fraction,
        allow_nan=return_variance <= 0.0 or np.isnan(return_variance),
    )
    sharpe = _require_representable_metric(
        "Sharpe ratio",
        sharpe,
        allow_nan=excess_variance <= 0.0 or np.isnan(excess_variance),
    )
    hit_rate = _require_representable_metric("hit rate", hit_rate)
    max_drawdown = rollup.max_drawdown
    if max_drawdown is not None:
        max_drawdown = _require_representable_metric("maximum drawdown", max_drawdown)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "kelly_fraction": kelly_fraction,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
    }


def _update_summary_rollup(rollup: SummaryRollup, records: list[dict]) -> SummaryRollup:
    for record in records:
        equity = float(record["equity"])
        if rollup.first_equity is None:
            rollup.first_equity = equity
            rollup.last_equity = equity
            rollup.running_max_equity = equity
            rollup.max_drawdown = 0.0
            continue

        daily_return = float(record["daily_return"])
        rollup.return_count, rollup.return_mean, rollup.return_m2 = _update_centered_moment(
            rollup.return_count,
            rollup.return_mean,
            rollup.return_m2,
            daily_return,
        )
        rollup.return_sum += daily_return
        rollup.return_sum_squares += daily_return * daily_return
        if daily_return > 0:
            rollup.positive_return_count += 1

        risk_free_return = record.get("risk_free_return")
        if not pd.isna(risk_free_return):
            excess_return = daily_return - float(risk_free_return)
            (
                rollup.excess_return_count,
                rollup.excess_return_mean,
                rollup.excess_return_m2,
            ) = _update_centered_moment(
                rollup.excess_return_count,
                rollup.excess_return_mean,
                rollup.excess_return_m2,
                excess_return,
            )
            rollup.excess_return_sum += excess_return
            rollup.excess_return_sum_squares += excess_return * excess_return

        rollup.last_equity = equity
        rollup.running_max_equity = max(float(rollup.running_max_equity or equity), equity)
        drawdown = equity / rollup.running_max_equity - 1.0 if rollup.running_max_equity > 0 else np.nan
        if pd.notna(drawdown):
            rollup.max_drawdown = min(float(rollup.max_drawdown or 0.0), float(drawdown))
    return rollup


def _load_strategy_summary_rollup(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> SummaryRollup | None:
    row = conn.execute(
        f"""
        SELECT {_STRATEGY_SUMMARY_INTEGRITY_COLUMN_SQL}, integrity_digest
        FROM strategy_summary
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND buy_rsi = ?
          AND profit_target_multiple = ?
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    ).fetchone()
    if row is None or not _strategy_summary_integrity_row_is_valid(row):
        return None
    return _summary_rollup_from_row(_strategy_summary_validation_row(row)[6:21])


def _summary_rollup_from_row(row: tuple | None) -> SummaryRollup | None:
    if row is None or row[0] is None:
        return None
    return_count = int(row[3] or 0)
    return_sum = float(row[4] or 0.0)
    return_sum_squares = float(row[5] or 0.0)
    excess_return_count = int(row[6] or 0)
    excess_return_sum = float(row[7] or 0.0)
    excess_return_sum_squares = float(row[8] or 0.0)
    legacy_return_mean, legacy_return_m2 = _legacy_centered_moment(
        return_count,
        return_sum,
        return_sum_squares,
    )
    legacy_excess_mean, legacy_excess_m2 = _legacy_centered_moment(
        excess_return_count,
        excess_return_sum,
        excess_return_sum_squares,
    )
    return SummaryRollup(
        first_equity=float(row[0]),
        last_equity=float(row[1]),
        running_max_equity=float(row[2]),
        return_count=return_count,
        return_sum=return_sum,
        return_sum_squares=return_sum_squares,
        return_mean=legacy_return_mean if row[11] is None else float(row[11]),
        return_m2=legacy_return_m2 if row[12] is None else float(row[12]),
        excess_return_count=excess_return_count,
        excess_return_sum=excess_return_sum,
        excess_return_sum_squares=excess_return_sum_squares,
        excess_return_mean=legacy_excess_mean if row[13] is None else float(row[13]),
        excess_return_m2=legacy_excess_m2 if row[14] is None else float(row[14]),
        positive_return_count=int(row[9] or 0),
        max_drawdown=None if row[10] is None else float(row[10]),
    )


def _legacy_centered_moment(count: int, total: float, total_squares: float) -> tuple[float, float]:
    """Read old raw moments without allowing them to bypass schema invalidation."""
    if count <= 0:
        return 0.0, 0.0
    mean = total / count
    m2 = float(np.longdouble(total_squares) - np.longdouble(total) * np.longdouble(total) / np.longdouble(count))
    return mean, max(m2, 0.0)


def _load_strategy_summary_rollups_for_asset(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> dict[tuple[float, float], SummaryRollup]:
    rows = conn.execute(
        f"""
        SELECT {_STRATEGY_SUMMARY_INTEGRITY_COLUMN_SQL}, integrity_digest
        FROM strategy_summary
        WHERE asset_symbol = ?
          AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchall()
    if any(not _strategy_summary_integrity_row_is_valid(row) for row in rows):
        return {}
    identities = [(row[2], row[3]) for row in rows]
    if len(set(identities)) != len(identities):
        return {}
    rollups: dict[tuple[float, float], SummaryRollup] = {}
    for row in rows:
        validation_row = _strategy_summary_validation_row(row)
        rollup = _summary_rollup_from_row(validation_row[6:21])
        if rollup is not None:
            rollups[(float(row[2]), float(row[3]))] = rollup
    return rollups


def save_strategy_summary(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
    rollup: SummaryRollup,
) -> None:
    conn.execute(
        _STRATEGY_SUMMARY_UPSERT_SQL,
        _strategy_summary_row(
            asset_symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
            state,
            rollup,
        ),
    )


_STRATEGY_SUMMARY_UPSERT_SQL = """
INSERT OR REPLACE INTO strategy_summary
(asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, start_date,
 end_date, trading_days, trades_executed, total_return, cagr,
 annualized_vol, sharpe, kelly_fraction, max_drawdown, hit_rate,
 first_equity, last_equity, running_max_equity, return_count,
 return_sum, return_sum_squares, excess_return_count, excess_return_sum,
 excess_return_sum_squares, positive_return_count, return_mean, return_m2,
 excess_return_mean, excess_return_m2, integrity_digest)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _strategy_summary_row(
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
    rollup: SummaryRollup,
) -> tuple:
    metrics = _rollup_metrics(rollup)
    row = (
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
        state["start_date"],
        state["last_date"],
        rollup.trading_days,
        int(state["trades_executed"]),
        _nullable_float(metrics["total_return"]),
        _nullable_float(metrics["cagr"]),
        _nullable_float(metrics["annualized_vol"]),
        _nullable_float(metrics["sharpe"]),
        _nullable_float(metrics["kelly_fraction"]),
        _nullable_float(metrics["max_drawdown"]),
        _nullable_float(metrics["hit_rate"]),
        rollup.first_equity,
        rollup.last_equity,
        rollup.running_max_equity,
        rollup.return_count,
        rollup.return_sum,
        rollup.return_sum_squares,
        rollup.excess_return_count,
        rollup.excess_return_sum,
        rollup.excess_return_sum_squares,
        rollup.positive_return_count,
        rollup.return_mean,
        rollup.return_m2,
        rollup.excess_return_mean,
        rollup.excess_return_m2,
    )
    return (*row, _strategy_summary_integrity_digest(row))


def save_strategy_summaries(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if rows:
        conn.executemany(_STRATEGY_SUMMARY_UPSERT_SQL, rows)


def _nullable_float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


def clear_asset_state(conn: sqlite3.Connection, asset_symbol: str, signal_symbol: str) -> None:
    params = (asset_symbol, signal_symbol)
    conn.execute("DELETE FROM strategy_state WHERE asset_symbol = ? AND signal_symbol = ?", params)
    conn.execute("DELETE FROM strategy_equity WHERE asset_symbol = ? AND signal_symbol = ?", params)
    conn.execute("DELETE FROM strategy_summary WHERE asset_symbol = ? AND signal_symbol = ?", params)
    conn.execute("DELETE FROM strategy_config WHERE asset_symbol = ? AND signal_symbol = ?", params)


def clear_signal_state(conn: sqlite3.Connection, signal_symbol: str) -> None:
    """Invalidate every strategy that depends on a corrected RSI symbol."""
    dependent_pairs = conn.execute(
        "SELECT DISTINCT asset_symbol, signal_symbol FROM strategy_state WHERE signal_symbol = ?",
        (signal_symbol,),
    ).fetchall()
    for asset_symbol, dependent_signal_symbol in dependent_pairs:
        asset_symbol = str(asset_symbol)
        dependent_signal_symbol = str(dependent_signal_symbol)
        if _strategy_pair_uses_private_signal_history(
            conn,
            asset_symbol,
            dependent_signal_symbol,
        ):
            continue
        clear_asset_state(conn, asset_symbol, dependent_signal_symbol)


def _clear_market_symbol_dependent_state(conn: sqlite3.Connection, symbol: str) -> None:
    """Invalidate every strategy that consumes a corrected market symbol."""
    dependent_pairs = conn.execute(
        """
        SELECT DISTINCT asset_symbol, signal_symbol
        FROM strategy_state
        WHERE asset_symbol = ? OR signal_symbol = ?
        """,
        (symbol, symbol),
    ).fetchall()
    for asset_symbol, signal_symbol in dependent_pairs:
        asset_symbol = str(asset_symbol)
        signal_symbol = str(signal_symbol)
        if asset_symbol != symbol and _strategy_pair_uses_private_signal_history(
            conn,
            asset_symbol,
            signal_symbol,
        ):
            continue
        clear_asset_state(conn, asset_symbol, signal_symbol)


def strategy_state_generation(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT generation FROM strategy_state_generation WHERE id = 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO strategy_state_generation (id, generation) VALUES (1, 0)")
        return 0
    return int(row[0])


def _bump_strategy_state_generation(conn: sqlite3.Connection) -> int:
    conn.execute("UPDATE strategy_state_generation SET generation = generation + 1 WHERE id = 1")
    return strategy_state_generation(conn)


def clear_all_strategy_state(conn: sqlite3.Connection) -> None:
    """Invalidate all simulations after a shared benchmark correction."""
    conn.execute("DELETE FROM strategy_state")
    conn.execute("DELETE FROM strategy_equity")
    conn.execute("DELETE FROM strategy_summary")
    conn.execute("DELETE FROM strategy_config")
    _bump_strategy_state_generation(conn)


def expected_state_count(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM strategy_state
            WHERE asset_symbol = ? AND signal_symbol = ?
            """,
            (asset_symbol, signal_symbol),
        ).fetchone()[0]
    )


def earliest_state_date(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> str | None:
    row = conn.execute(
        """
        SELECT MIN(last_date)
        FROM strategy_state
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def load_saved_market_data(conn: sqlite3.Connection, symbols: list[str]) -> pd.DataFrame:
    """Load strategy inputs on the first symbol's persisted trading calendar."""
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in symbols)
    raw = pd.read_sql_query(
        f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM market_data
        WHERE symbol IN ({placeholders})
        ORDER BY date, symbol
        """,
        conn,
        params=symbols,
        parse_dates=["date"],
    )
    if raw.empty:
        return pd.DataFrame()

    frames_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        group = raw[raw["symbol"].eq(symbol)].copy()
        if group.empty:
            if symbol == RISK_FREE_SYMBOL:
                continue
            return pd.DataFrame()
        group = group.set_index("date")[["open", "high", "low", "close", "volume"]]
        group.columns = [
            f"{symbol}_Open",
            f"{symbol}_High",
            f"{symbol}_Low",
            f"{symbol}_Close",
            f"{symbol}_Volume",
        ]
        frames_by_symbol[symbol] = group

    calendar_symbol = symbols[0]
    out = frames_by_symbol[calendar_symbol].sort_index()
    for symbol in symbols:
        if symbol == calendar_symbol or symbol not in frames_by_symbol:
            continue
        out = out.join(frames_by_symbol[symbol], how="left")

    risk_free = frames_by_symbol.get(RISK_FREE_SYMBOL)
    if risk_free is not None:
        # Match load_strategy_data: benchmark gaps must not change the asset
        # and signal trading calendar during a saved-state rebuild.
        out[risk_free.columns] = risk_free.reindex(
            out.index,
            method="ffill",
        )
    return out


# ``@`` is excluded by every accepted workflow/user market-symbol grammar.
# The hex-encoded JSON identity is reversible and injective, so private keys
# cannot collide with either a real symbol or another strategy pair.
_STRATEGY_SIGNAL_CACHE_PREFIX = "@strategy-signal-v1/"


def _strategy_signal_cache_symbol(asset_symbol: str, signal_symbol: str) -> str:
    """Return the private persisted signal namespace for one strategy pair."""
    if asset_symbol == signal_symbol:
        return signal_symbol
    identity = json.dumps(
        [asset_symbol, signal_symbol],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{_STRATEGY_SIGNAL_CACHE_PREFIX}{identity.hex()}"


def _persisted_strategy_signal_symbol(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> str:
    """Resolve a pair-scoped workflow snapshot, falling back to legacy state."""
    scoped_symbol = _strategy_signal_cache_symbol(asset_symbol, signal_symbol)
    if scoped_symbol == signal_symbol:
        return signal_symbol
    scoped_row = conn.execute(
        "SELECT 1 FROM market_data WHERE symbol = ? LIMIT 1",
        (scoped_symbol,),
    ).fetchone()
    return scoped_symbol if scoped_row is not None else signal_symbol


def _strategy_pair_uses_private_signal_history(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> bool:
    return _persisted_strategy_signal_symbol(conn, asset_symbol, signal_symbol) != signal_symbol


def _history_in_storage_namespace(
    history: pd.DataFrame,
    source_symbol: str,
    storage_symbol: str,
) -> pd.DataFrame:
    """Rename one semantic history for an internal persisted namespace."""
    if source_symbol == storage_symbol:
        return history
    renamed = history.rename(
        columns={f"{source_symbol}_{field}": f"{storage_symbol}_{field}" for field in _MARKET_DATA_FIELDS}
    )
    providers = history.attrs.get(_MARKET_DATA_PROVIDERS_ATTR)
    if isinstance(providers, Mapping) and source_symbol in providers:
        renamed.attrs[_MARKET_DATA_PROVIDERS_ATTR] = {
            **providers,
            storage_symbol: providers[source_symbol],
        }
    return renamed


def _load_saved_strategy_market_data(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    signal_storage_symbol: str,
) -> pd.DataFrame:
    """Load one pair using its exact persisted signal snapshot."""
    data = load_saved_market_data(
        conn,
        list(dict.fromkeys([asset_symbol, signal_storage_symbol, RISK_FREE_SYMBOL])),
    )
    if data.empty or signal_storage_symbol == signal_symbol:
        return data
    return data.rename(
        columns={f"{signal_storage_symbol}_{field}": f"{signal_symbol}_{field}" for field in _MARKET_DATA_FIELDS}
    )


def load_saved_close_series(conn: sqlite3.Connection, symbol: str) -> pd.Series:
    rows = pd.read_sql_query(
        """
        SELECT date, close
        FROM market_data
        WHERE symbol = ?
          AND close IS NOT NULL
        ORDER BY date
        """,
        conn,
        params=(symbol,),
        parse_dates=["date"],
    )
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.set_index("date")["close"].astype(float)


def _saved_market_dates(conn: sqlite3.Connection, symbol: str) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT date FROM market_data WHERE symbol = ?",
            (symbol,),
        ).fetchall()
    }


def _saved_strategy_signal_alignment(
    conn: sqlite3.Connection,
    *,
    asset_symbol: str,
    signal_symbol: str,
    signal_storage_symbol: str,
) -> str | None:
    """Return the signal observation backing this pair's persisted checkpoint."""
    checkpoint_row = conn.execute(
        """
        SELECT MAX(last_date)
        FROM strategy_state
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchone()
    checkpoint = checkpoint_row[0] if checkpoint_row else None
    if checkpoint is None:
        return None
    asset_dates = pd.read_sql_query(
        """
        SELECT date
        FROM market_data
        WHERE symbol = ? AND date <= ?
        ORDER BY date
        """,
        conn,
        params=(asset_symbol, str(checkpoint)),
        parse_dates=["date"],
    )
    signal_close = load_saved_close_series(conn, signal_storage_symbol)
    if asset_dates.empty or signal_close.empty:
        return None
    _aligned, observation_dates = align_signal_values_to_asset_sessions(
        signal_close,
        pd.DatetimeIndex(asset_dates["date"]),
    )
    observation_date = observation_dates.get(pd.Timestamp(checkpoint), pd.NaT)
    return None if pd.isna(observation_date) else _date_str(observation_date)


MAX_SIGNAL_OBSERVATION_LAG_DAYS = 7
MAX_SIGNAL_OBSERVATION_LEAD_DAYS = 3


def _signal_observation_can_lead_final_asset_session(
    asset_session: pd.Timestamp,
    observation_session: pd.Timestamp,
) -> bool:
    """Allow a short lead only for 24/7 weekend observations."""
    if observation_session <= asset_session:
        return True
    return bool(
        asset_session.weekday() == 4
        and observation_session.weekday() >= 5
        and observation_session <= asset_session + pd.Timedelta(days=MAX_SIGNAL_OBSERVATION_LEAD_DAYS)
    )


def _historically_added_market_symbols(
    conn: sqlite3.Connection,
    data: pd.DataFrame,
    symbols: list[str],
    *,
    exclude_strategy_pair: tuple[str, str] | None = None,
) -> set[str]:
    """Find incoming sessions that precede a dependent strategy checkpoint."""
    historical_symbols: set[str] = set()
    for symbol in symbols:
        columns = [f"{symbol}_{field}" for field in _MARKET_DATA_FIELDS]
        if any(column not in data.columns for column in columns):
            continue
        symbol_data = data.loc[~data.loc[:, columns].isna().all(axis=1)]
        incoming_dates = {_date_str(date) for date in symbol_data.index}
        added_dates = incoming_dates.difference(_saved_market_dates(conn, symbol))
        if not added_dates:
            continue
        exclusion_clause = (
            " AND NOT (asset_symbol = ? AND signal_symbol = ?)" if exclude_strategy_pair is not None else ""
        )
        exclusion_params = exclude_strategy_pair or ()
        if symbol == RISK_FREE_SYMBOL:
            checkpoint_row = conn.execute(
                f"SELECT MAX(last_date) FROM strategy_state WHERE last_date IS NOT NULL{exclusion_clause}",
                exclusion_params,
            ).fetchone()
            latest_checkpoint = checkpoint_row[0] if checkpoint_row else None
            if latest_checkpoint is not None and any(
                added_date <= str(latest_checkpoint) for added_date in added_dates
            ):
                historical_symbols.add(symbol)
            continue

        checkpoint_rows = conn.execute(
            f"""
            SELECT asset_symbol, signal_symbol, last_date
            FROM strategy_state
            WHERE last_date IS NOT NULL
              AND (asset_symbol = ? OR signal_symbol = ?)
              {exclusion_clause}
            """,
            (symbol, symbol, *exclusion_params),
        ).fetchall()
        dependent_checkpoints = [
            str(last_date)
            for asset_symbol, signal_symbol, last_date in checkpoint_rows
            if str(asset_symbol) == symbol
            or not _strategy_pair_uses_private_signal_history(
                conn,
                str(asset_symbol),
                str(signal_symbol),
            )
        ]
        latest_checkpoint = max(dependent_checkpoints, default=None)
        if latest_checkpoint is not None and any(added_date <= str(latest_checkpoint) for added_date in added_dates):
            historical_symbols.add(symbol)
    return historical_symbols


def signal_observation_is_fresh(asset_date: object, observation_date: object) -> bool:
    """Whether a signal observation is close enough to an asset session to act on."""
    try:
        asset_session = pd.Timestamp(asset_date)
        observation_session = pd.Timestamp(observation_date)
    except (TypeError, ValueError, OverflowError):
        return False
    if pd.isna(asset_session) or pd.isna(observation_session):
        return False
    asset_session = asset_session.tz_localize(None).normalize()
    observation_session = observation_session.tz_localize(None).normalize()
    earliest = asset_session - pd.Timedelta(days=MAX_SIGNAL_OBSERVATION_LAG_DAYS)
    return bool(
        observation_session >= earliest
        and _signal_observation_can_lead_final_asset_session(
            asset_session,
            observation_session,
        )
    )


def align_signal_values_to_asset_sessions(
    values: pd.Series,
    asset_dates: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    """Map settled signal observations to the ETF row preceding their next open.

    A row's RSI drives the pending action executed at the following asset open.
    For each non-final asset row, use the latest signal observation strictly
    before the next recorded asset session. This is identical to an exact-date
    join for ordinary exchange-traded proxies, while allowing a Sunday 24/7
    observation to drive Monday's open from Friday's pending state.

    The final row has no recorded next session. Admit only a bounded Saturday or
    Sunday lead, preserving 24/7 proxy observations without pulling a later
    exchange weekday backward into a stale or incomplete asset history.
    """
    asset_index = pd.DatetimeIndex(pd.to_datetime(asset_dates)).tz_localize(None)
    if len(asset_index) == 0:
        empty_values = pd.Series(index=asset_index, dtype=float)
        empty_dates = pd.Series(index=asset_index, dtype="datetime64[ns]")
        return empty_values, empty_dates

    signal = values.copy()
    signal.index = pd.DatetimeIndex(pd.to_datetime(signal.index)).tz_localize(None)
    signal = signal[~signal.index.duplicated(keep="last")].sort_index()
    aligned_values = np.full(len(asset_index), np.nan, dtype=np.float64)
    aligned_dates = np.full(len(asset_index), np.datetime64("NaT"), dtype="datetime64[ns]")
    if signal.empty:
        return (
            pd.Series(aligned_values, index=asset_index),
            pd.Series(aligned_dates, index=asset_index),
        )

    signal_dates = signal.index.to_numpy(dtype="datetime64[ns]")
    signal_values = signal.to_numpy(dtype=np.float64)
    cutoffs = np.empty(len(asset_index), dtype="datetime64[ns]")
    if len(asset_index) > 1:
        cutoffs[:-1] = asset_index[1:].to_numpy(dtype="datetime64[ns]")
    cutoffs[-1] = asset_index[-1].to_datetime64()

    positions = np.searchsorted(signal_dates, cutoffs, side="left") - 1
    positions[-1] = (
        np.searchsorted(
            signal_dates,
            asset_index[-1].to_datetime64(),
            side="right",
        )
        - 1
    )
    final_weekend_horizon = asset_index[-1] + pd.Timedelta(days=MAX_SIGNAL_OBSERVATION_LEAD_DAYS)
    final_lead_start = np.searchsorted(
        signal_dates,
        asset_index[-1].to_datetime64(),
        side="right",
    )
    final_lead_end = np.searchsorted(
        signal_dates,
        final_weekend_horizon.to_datetime64(),
        side="right",
    )
    if asset_index[-1].weekday() == 4 and final_lead_start < final_lead_end:
        lead_dates = pd.DatetimeIndex(signal_dates[final_lead_start:final_lead_end])
        weekend_offsets = np.flatnonzero(lead_dates.weekday >= 5)
        if len(weekend_offsets):
            positions[-1] = final_lead_start + int(weekend_offsets[-1])
    usable = positions >= 0
    usable_indices = np.flatnonzero(usable)
    if len(usable_indices):
        # Non-final rows store an action that executes at the following asset
        # open, so freshness must be measured at that execution session rather
        # than at the earlier row carrying the pending action.  The final cutoff
        # remains the final asset session itself, preserving its explicit
        # Friday/weekend lead handling above.
        action_session_dates = cutoffs[usable_indices]
        observation_session_dates = signal_dates[positions[usable_indices]]
        earliest_observations = action_session_dates - np.timedelta64(MAX_SIGNAL_OBSERVATION_LAG_DAYS, "D")
        fresh = observation_session_dates >= earliest_observations
        usable[usable_indices] = fresh
    aligned_values[usable] = signal_values[positions[usable]]
    aligned_dates[usable] = signal_dates[positions[usable]]
    return (
        pd.Series(aligned_values, index=asset_index),
        pd.Series(aligned_dates, index=asset_index),
    )


def _saved_signal_dependent_alignments(
    conn: sqlite3.Connection,
    signal_symbol: str,
    *,
    asset_symbol: str | None = None,
    exclude_strategy_pair: tuple[str, str] | None = None,
) -> dict[tuple[str, str], str | None]:
    """Snapshot each dependent's final processed signal observation."""
    if asset_symbol is not None and exclude_strategy_pair is not None:
        raise ValueError("Cannot include an asset and exclude a strategy in one dependent-alignment query.")
    if asset_symbol is not None:
        asset_filter = "AND asset_symbol = ?"
        params: tuple[str, ...] = (signal_symbol, asset_symbol)
    elif exclude_strategy_pair is not None:
        asset_filter = "AND NOT (asset_symbol = ? AND signal_symbol = ?)"
        params = (signal_symbol, *exclude_strategy_pair)
    else:
        asset_filter = ""
        params = (signal_symbol,)
    dependent_sessions = conn.execute(
        f"""
        SELECT asset_symbol, MAX(last_date)
        FROM strategy_state
        WHERE signal_symbol = ?
          AND last_date IS NOT NULL
          {asset_filter}
        GROUP BY asset_symbol
        """,
        params,
    ).fetchall()
    dependent_sessions = [
        row
        for row in dependent_sessions
        if not _strategy_pair_uses_private_signal_history(
            conn,
            str(row[0]),
            signal_symbol,
        )
    ]
    if not dependent_sessions:
        return {}

    signal_close = load_saved_close_series(conn, signal_symbol)
    if signal_close.empty:
        return {(str(dependent_asset), str(last_date)): None for dependent_asset, last_date in dependent_sessions}
    dependent_assets = list(dict.fromkeys(str(row[0]) for row in dependent_sessions))
    placeholders = ",".join("?" for _ in dependent_assets)
    saved_asset_dates: dict[str, list[str]] = {asset: [] for asset in dependent_assets}
    for dependent_asset, date in conn.execute(
        f"""
        SELECT symbol, date
        FROM market_data
        WHERE symbol IN ({placeholders})
        ORDER BY symbol, date
        """,
        dependent_assets,
    ).fetchall():
        saved_asset_dates[str(dependent_asset)].append(str(date))

    alignments: dict[tuple[str, str], str | None] = {}
    for dependent_asset, last_date in dependent_sessions:
        dependent_asset = str(dependent_asset)
        last_date = str(last_date)
        asset_dates = pd.DatetimeIndex(pd.to_datetime(saved_asset_dates[dependent_asset]))
        if len(asset_dates) == 0:
            alignments[(dependent_asset, last_date)] = None
            continue
        _aligned, observation_dates = align_signal_values_to_asset_sessions(signal_close, asset_dates)
        target = pd.Timestamp(last_date)
        observation_date = observation_dates.get(target, pd.NaT)
        alignments[(dependent_asset, last_date)] = None if pd.isna(observation_date) else _date_str(observation_date)
    return alignments


def _signal_additions_affect_processed_inputs(
    conn: sqlite3.Connection,
    signal_symbol: str,
    added_dates: set[str],
    *,
    exclude_strategy_pair: tuple[str, str] | None = None,
) -> bool:
    """Whether added signal bars can change a dependent's processed final RSI input."""
    if not added_dates:
        return False
    exclusion_clause = " AND NOT (asset_symbol = ? AND signal_symbol = ?)" if exclude_strategy_pair is not None else ""
    exclusion_params = exclude_strategy_pair or ()
    dependent_sessions = conn.execute(
        f"""
        SELECT asset_symbol, MAX(last_date)
        FROM strategy_state
        WHERE signal_symbol = ?
          AND last_date IS NOT NULL
          {exclusion_clause}
        GROUP BY asset_symbol
        """,
        (signal_symbol, *exclusion_params),
    ).fetchall()
    dependent_sessions = [
        row
        for row in dependent_sessions
        if not _strategy_pair_uses_private_signal_history(
            conn,
            str(row[0]),
            signal_symbol,
        )
    ]
    for dependent_asset, last_date in dependent_sessions:
        last_date = str(last_date)
        next_asset_row = conn.execute(
            """
            SELECT MIN(date)
            FROM market_data
            WHERE symbol = ? AND date > ?
            """,
            (str(dependent_asset), last_date),
        ).fetchone()
        next_asset_date = next_asset_row[0] if next_asset_row else None
        if next_asset_date is not None and any(added_date < str(next_asset_date) for added_date in added_dates):
            return True
        if next_asset_date is None:
            try:
                asset_session = pd.Timestamp(last_date)
            except (TypeError, ValueError, OverflowError):
                continue
            for added_date in added_dates:
                try:
                    observation_session = pd.Timestamp(added_date)
                except (TypeError, ValueError, OverflowError):
                    continue
                if _signal_observation_can_lead_final_asset_session(
                    asset_session,
                    observation_session,
                ):
                    return True
    return False


def load_aligned_rsi_for_asset_session(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    rsi_period: int,
    asset_date: object,
) -> tuple[str, float] | None:
    """Return canonical-close-derived RSI aligned to one persisted asset row."""
    asset_rows = pd.read_sql_query(
        "SELECT date FROM market_data WHERE symbol = ? ORDER BY date",
        conn,
        params=(asset_symbol,),
        parse_dates=["date"],
    )
    persisted_signal_symbol = _persisted_strategy_signal_symbol(
        conn,
        asset_symbol,
        signal_symbol,
    )
    canonical_rsi = _canonical_saved_rsi_series(conn, persisted_signal_symbol, rsi_period)
    if asset_rows.empty or canonical_rsi is None or canonical_rsi.empty:
        return None

    aligned, observation_dates = align_signal_values_to_asset_sessions(
        canonical_rsi,
        pd.Index(asset_rows["date"]),
    )
    target = pd.Timestamp(asset_date).tz_localize(None)
    if target not in aligned.index:
        return None
    value = aligned.loc[target]
    observation_date = observation_dates.loc[target]
    if pd.isna(value) or pd.isna(observation_date):
        return None
    return _date_str(observation_date), float(value)


def _action_code(action: object) -> int:
    if action == "none":
        return ACTION_NONE
    if action == "buy":
        return ACTION_BUY
    if action == "sell":
        return ACTION_SELL
    raise ValueError(f"Unknown persisted pending action: {action!r}.")


def _action_label(action_code: int) -> str:
    if action_code == ACTION_BUY:
        return "buy"
    if action_code == ACTION_SELL:
        return "sell"
    return "none"


def _trading_cost_rate(cfg: BacktestConfig) -> float:
    return (cfg.fee_bps + cfg.slippage_bps) / 10_000.0


def _risk_free_returns_from_data(data: pd.DataFrame) -> np.ndarray:
    risk_free_col = f"{RISK_FREE_SYMBOL}_Close"
    if risk_free_col not in data.columns:
        return np.full(len(data), np.nan, dtype=np.float64)
    annual_yield = data[risk_free_col].to_numpy(dtype=np.float64) / 100.0
    invalid = np.isinf(annual_yield) | ((~np.isnan(annual_yield)) & (annual_yield <= -1.0))
    if invalid.any():
        value = data[risk_free_col].iloc[int(np.flatnonzero(invalid)[0])]
        raise AssetMarketDataError(f"{RISK_FREE_SYMBOL} Close must be greater than -100 and finite (got {value!r}).")
    return (1.0 + annual_yield) ** (1 / 252) - 1.0


def _canonical_risk_free_returns_for_dates(
    conn: sqlite3.Connection,
    dates: pd.DatetimeIndex,
) -> np.ndarray | None:
    """Derive benchmark returns with saved-data join and forward-fill semantics."""
    rows = conn.execute(
        """
        SELECT date, close, TYPEOF(date), TYPEOF(close)
        FROM market_data
        WHERE symbol = ?
        ORDER BY date
        """,
        (RISK_FREE_SYMBOL,),
    ).fetchall()
    if not rows:
        return np.full(len(dates), np.nan, dtype=np.float64)
    date_values: list[pd.Timestamp] = []
    close_values: list[float] = []
    for date_value, close_value, date_type, close_type in rows:
        if date_type != "text" or close_type not in {"integer", "real"}:
            return None
        try:
            timestamp = pd.Timestamp(date_value)
            numeric_close = float(close_value)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            pd.isna(timestamp)
            or str(date_value) != timestamp.date().isoformat()
            or not math.isfinite(numeric_close)
            or numeric_close <= -100.0
        ):
            return None
        date_values.append(timestamp)
        close_values.append(numeric_close)
    benchmark_dates = pd.DatetimeIndex(date_values)
    if benchmark_dates.has_duplicates or not benchmark_dates.is_monotonic_increasing:
        return None
    aligned_close = pd.Series(close_values, index=benchmark_dates).reindex(
        dates,
        method="ffill",
    )
    try:
        return _risk_free_returns_from_data(
            pd.DataFrame(
                {f"{RISK_FREE_SYMBOL}_Close": aligned_close.to_numpy(dtype=np.float64)},
                index=dates,
            )
        )
    except (AssetMarketDataError, TypeError, ValueError, OverflowError):
        return None


def _strategy_states_exactly_match(persisted: dict, expected: dict) -> bool:
    """Bind compact account state to the same canonical optimized replay."""
    for field_name in (
        "start_date",
        "last_date",
        "cash",
        "shares",
        "in_position",
        "entry_date",
        "pending_action",
        "prev_equity",
        "trades_executed",
    ):
        if persisted.get(field_name) != expected.get(field_name):
            return False
    persisted_entry = persisted.get("entry_price")
    expected_entry = expected.get("entry_price")
    if pd.isna(persisted_entry) or pd.isna(expected_entry):
        return bool(pd.isna(persisted_entry) and pd.isna(expected_entry))
    return bool(float(persisted_entry) == float(expected_entry))


def _canonical_strategy_grid_replay(
    conn: sqlite3.Connection,
    *,
    asset_symbol: str,
    signal_symbol: str,
    config_pairs: list[tuple[float, float]],
    base_cfg: BacktestConfig,
    start_date: str,
    end_date: str,
    rsi_entry_rule: str,
) -> dict[tuple[float, float], tuple[dict, SummaryRollup]] | None:
    """Replay every compact config from independently persisted market inputs."""
    market_rows = conn.execute(
        """
        SELECT date, open, high, close,
               TYPEOF(date), TYPEOF(open), TYPEOF(high), TYPEOF(close)
        FROM market_data
        WHERE symbol = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        (asset_symbol, start_date, end_date),
    ).fetchall()
    if not market_rows or not config_pairs:
        return None
    try:
        dates = pd.DatetimeIndex(pd.to_datetime([row[0] for row in market_rows], format="%Y-%m-%d"))
        open_prices = np.asarray([row[1] for row in market_rows], dtype=np.float64)
        high_prices = np.asarray([row[2] for row in market_rows], dtype=np.float64)
        close_prices = np.asarray([row[3] for row in market_rows], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        any(
            row[4] != "text" or any(storage_type not in {"integer", "real"} for storage_type in row[5:])
            for row in market_rows
        )
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or dates[0].date().isoformat() != start_date
        or dates[-1].date().isoformat() != end_date
        or not np.isfinite(open_prices).all()
        or not np.isfinite(high_prices).all()
        or not np.isfinite(close_prices).all()
        or np.any(open_prices <= 0.0)
        or np.any(close_prices <= 0.0)
        or np.any(high_prices < np.maximum(open_prices, close_prices))
    ):
        return None
    persisted_signal_symbol = _persisted_strategy_signal_symbol(
        conn,
        asset_symbol,
        signal_symbol,
    )
    canonical_signal_rsi = _canonical_saved_rsi_series(
        conn,
        persisted_signal_symbol,
        base_cfg.rsi_period,
    )
    if canonical_signal_rsi is None:
        return None
    aligned_rsi, _observation_dates = align_signal_values_to_asset_sessions(
        canonical_signal_rsi,
        dates,
    )
    rsi_values = aligned_rsi.to_numpy(dtype=np.float64)
    if not np.isfinite(rsi_values).any():
        return None
    risk_free_returns = _canonical_risk_free_returns_for_dates(conn, dates)
    if risk_free_returns is None:
        return None

    config_count = len(config_pairs)
    buy_rsi_values = np.asarray([pair[0] for pair in config_pairs], dtype=np.float64)
    profit_target_values = np.asarray([pair[1] for pair in config_pairs], dtype=np.float64)
    initial_capital = float(base_cfg.initial_capital)
    try:
        results = run_grid_summary(
            open_prices,
            high_prices,
            close_prices,
            rsi_values,
            risk_free_returns,
            buy_rsi_values,
            profit_target_values,
            np.zeros(config_count, dtype=np.int64),
            np.full(config_count, initial_capital),
            np.zeros(config_count),
            np.zeros(config_count, dtype=np.bool_),
            np.full(config_count, np.nan),
            np.full(config_count, ACTION_NONE, dtype=np.int64),
            np.full(config_count, initial_capital),
            np.zeros(config_count, dtype=np.int64),
            np.full(config_count, np.nan),
            np.full(config_count, np.nan),
            np.full(config_count, np.nan),
            np.zeros(config_count, dtype=np.int64),
            np.zeros(config_count),
            np.zeros(config_count),
            np.zeros(config_count, dtype=np.int64),
            np.zeros(config_count),
            np.zeros(config_count),
            np.zeros(config_count, dtype=np.int64),
            np.full(config_count, np.nan),
            _trading_cost_rate(base_cfg),
            rsi_entry_rule_code(rsi_entry_rule),
        )
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.all(results[0]):
        return None
    date_strings = [timestamp.date().isoformat() for timestamp in dates]
    replayed: dict[tuple[float, float], tuple[dict, SummaryRollup]] = {}
    for config_idx, config_pair in enumerate(config_pairs):
        entry_row_index = int(results[23][config_idx])
        replayed[config_pair] = (
            {
                "start_date": date_strings[0],
                "last_date": date_strings[-1],
                "cash": float(results[1][config_idx]),
                "shares": float(results[2][config_idx]),
                "in_position": bool(results[3][config_idx]),
                "entry_price": float(results[4][config_idx]),
                "entry_date": (
                    date_strings[entry_row_index] if bool(results[3][config_idx]) and entry_row_index >= 0 else None
                ),
                "pending_action": _action_label(int(results[5][config_idx])),
                "prev_equity": float(results[6][config_idx]),
                "trades_executed": int(results[7][config_idx]),
            },
            _rollup_from_arrays(
                float(results[8][config_idx]),
                float(results[9][config_idx]),
                float(results[10][config_idx]),
                int(results[11][config_idx]),
                float(results[12][config_idx]),
                float(results[13][config_idx]),
                int(results[14][config_idx]),
                float(results[15][config_idx]),
                float(results[16][config_idx]),
                int(results[17][config_idx]),
                float(results[18][config_idx]),
                float(results[19][config_idx]),
                float(results[20][config_idx]),
                float(results[21][config_idx]),
                float(results[22][config_idx]),
            ),
        )
    return replayed


def _market_arrays(
    data: pd.DataFrame,
    rsi: pd.Series,
    asset_symbol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aligned_rsi, _observation_dates = align_signal_values_to_asset_sessions(rsi, data.index)
    rsi_values = aligned_rsi.to_numpy(dtype=np.float64)
    if len(data) and not np.isfinite(rsi_values).any():
        raise AssetMarketDataError(
            f"No finite RSI observations for {asset_symbol} align with its finalized market sessions."
        )
    return (
        data[f"{asset_symbol}_Open"].to_numpy(dtype=np.float64),
        data[f"{asset_symbol}_High"].to_numpy(dtype=np.float64),
        data[f"{asset_symbol}_Close"].to_numpy(dtype=np.float64),
        rsi_values,
        _risk_free_returns_from_data(data),
    )


def _rollup_to_arrays(
    rollup: SummaryRollup,
) -> tuple[float, float, float, int, float, float, int, float, float, int, float, float, float, float, float]:
    return (
        np.nan if rollup.first_equity is None else float(rollup.first_equity),
        np.nan if rollup.last_equity is None else float(rollup.last_equity),
        np.nan if rollup.running_max_equity is None else float(rollup.running_max_equity),
        int(rollup.return_count),
        float(rollup.return_sum),
        float(rollup.return_sum_squares),
        int(rollup.excess_return_count),
        float(rollup.excess_return_sum),
        float(rollup.excess_return_sum_squares),
        int(rollup.positive_return_count),
        np.nan if rollup.max_drawdown is None else float(rollup.max_drawdown),
        float(rollup.return_mean),
        float(rollup.return_m2),
        float(rollup.excess_return_mean),
        float(rollup.excess_return_m2),
    )


def _rollup_from_arrays(
    first_equity: float,
    last_equity: float,
    running_max_equity: float,
    return_count: int,
    return_sum: float,
    return_sum_squares: float,
    excess_return_count: int,
    excess_return_sum: float,
    excess_return_sum_squares: float,
    positive_return_count: int,
    max_drawdown: float,
    return_mean: float,
    return_m2: float,
    excess_return_mean: float,
    excess_return_m2: float,
) -> SummaryRollup:
    return SummaryRollup(
        first_equity=None if np.isnan(first_equity) else float(first_equity),
        last_equity=None if np.isnan(last_equity) else float(last_equity),
        running_max_equity=None if np.isnan(running_max_equity) else float(running_max_equity),
        return_count=int(return_count),
        return_sum=float(return_sum),
        return_sum_squares=float(return_sum_squares),
        return_mean=float(return_mean),
        return_m2=float(return_m2),
        excess_return_count=int(excess_return_count),
        excess_return_sum=float(excess_return_sum),
        excess_return_sum_squares=float(excess_return_sum_squares),
        excess_return_mean=float(excess_return_mean),
        excess_return_m2=float(excess_return_m2),
        positive_return_count=int(positive_return_count),
        max_drawdown=None if np.isnan(max_drawdown) else float(max_drawdown),
    )


def _equity_records_from_arrays(
    *,
    date_strings: list[str],
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    equity_values: np.ndarray,
    daily_returns: np.ndarray,
    risk_free_returns: np.ndarray,
    in_position_values: np.ndarray,
    action_executed_values: np.ndarray,
    pending_action_values: np.ndarray,
    trades_executed_values: np.ndarray,
) -> list[dict]:
    return [
        {
            "asset_symbol": asset_symbol,
            "signal_symbol": signal_symbol,
            "buy_rsi": buy_rsi,
            "profit_target_multiple": profit_target_multiple,
            "date": date_str,
            "equity": float(equity_values[idx]),
            "daily_return": float(daily_returns[idx]),
            "risk_free_return": float(risk_free_returns[idx]) if not np.isnan(risk_free_returns[idx]) else np.nan,
            "in_position": int(in_position_values[idx]),
            "action_executed": _action_label(int(action_executed_values[idx])),
            "pending_action": _action_label(int(pending_action_values[idx])),
            "trades_executed": int(trades_executed_values[idx]),
        }
        for idx, date_str in enumerate(date_strings)
    ]


def load_best_strategy_summary(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    rsi_entry_rule: str = "lower",
) -> dict[str, object] | None:
    """Load the best row only when every candidate summary authenticates."""
    order_by = best_strategy_order_by_clause(rsi_entry_rule)
    rows = conn.execute(
        f"""
        SELECT {_STRATEGY_SUMMARY_INTEGRITY_COLUMN_SQL}, integrity_digest
        FROM strategy_summary
        WHERE asset_symbol = ? AND signal_symbol = ?
        ORDER BY {order_by}
        """,
        (asset_symbol, signal_symbol),
    ).fetchall()
    if not rows or any(
        not _strategy_summary_integrity_row_is_valid(row)
        or not _strategy_summary_validation_row_is_semantically_valid(_strategy_summary_validation_row(row))
        for row in rows
    ):
        return None
    identities = [(row[2], row[3]) for row in rows]
    if len(set(identities)) != len(identities):
        return None
    return dict(zip(_STRATEGY_SUMMARY_INTEGRITY_COLUMNS, rows[0][:-1], strict=True))


def _best_summary_config(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    rsi_entry_rule: str = "lower",
) -> tuple[float, float] | None:
    row = load_best_strategy_summary(
        conn,
        asset_symbol,
        signal_symbol,
        rsi_entry_rule,
    )
    if row is None:
        return None
    return float(row["buy_rsi"]), float(row["profit_target_multiple"])


def _complete_curve_values_match(first: object, second: object) -> bool:
    try:
        first_value = float(first)
        second_value = float(second)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(first_value) or not math.isfinite(second_value):
        return False
    tolerance = _float64_roundoff_tolerance(first_value, second_value)
    return bool(abs(np.longdouble(first_value) - np.longdouble(second_value)) <= tolerance)


def _complete_curve_optional_values_match(first: object, second: object) -> bool:
    first_missing = first is None or pd.isna(first)
    second_missing = second is None or pd.isna(second)
    if first_missing or second_missing:
        return bool(first_missing and second_missing)
    return _complete_curve_values_match(first, second)


def _canonical_optional_float_values_match(first: object, second: object) -> bool:
    """Compare persisted canonical-origin floats without an arithmetic tolerance."""
    first_missing = first is None or pd.isna(first)
    second_missing = second is None or pd.isna(second)
    if first_missing or second_missing:
        return bool(first_missing and second_missing)
    try:
        first_value = float(first)
        second_value = float(second)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(math.isfinite(first_value) and math.isfinite(second_value) and first_value == second_value)


def _complete_curve_rollups_match(first: SummaryRollup, second: SummaryRollup) -> bool:
    for field_name in (
        "return_count",
        "excess_return_count",
        "positive_return_count",
    ):
        if getattr(first, field_name) != getattr(second, field_name):
            return False
    for field_name in (
        "first_equity",
        "last_equity",
        "running_max_equity",
        "return_sum",
        "return_sum_squares",
        "return_mean",
        "return_m2",
        "excess_return_sum",
        "excess_return_sum_squares",
        "excess_return_mean",
        "excess_return_m2",
        "max_drawdown",
    ):
        if not _complete_curve_values_match(
            getattr(first, field_name),
            getattr(second, field_name),
        ):
            return False
    return True


def _strategy_equity_curve_rows(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> list[tuple]:
    return conn.execute(
        """
        SELECT date, equity, daily_return, risk_free_return, in_position,
               action_executed, pending_action, trades_executed, integrity_digest
        FROM strategy_equity
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND buy_rsi = ?
          AND profit_target_multiple = ?
        ORDER BY date
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    ).fetchall()


def _equity_curve_is_complete(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    *,
    equity_rows: list[tuple] | None = None,
    rsi_period: int | None = None,
    rsi_entry_rule: str = "lower",
    base_cfg: BacktestConfig | None = None,
    expected_buy_rsi_values: list[float] | None = None,
    expected_profit_target_values: list[float] | None = None,
    expected_strategy_fingerprint: str | None = None,
    allow_unbound_backtest_config: bool = False,
) -> bool:
    if base_cfg is None:
        if (
            not allow_unbound_backtest_config
            or expected_buy_rsi_values is not None
            or expected_profit_target_values is not None
            or expected_strategy_fingerprint is not None
        ):
            return False
    else:
        if (
            expected_buy_rsi_values is None
            or expected_profit_target_values is None
            or (rsi_period is not None and rsi_period != base_cfg.rsi_period)
        ):
            return False
        try:
            normalized_buy_rsi, normalized_profit_targets = _validated_strategy_grid_inputs(
                base_cfg,
                expected_buy_rsi_values,
                expected_profit_target_values,
            )
            expected_pairs = _strategy_config_pairs(
                list(normalized_buy_rsi),
                list(normalized_profit_targets),
            )
            derived_strategy_fingerprint = strategy_config_fingerprint(
                base_cfg,
                list(normalized_buy_rsi),
                list(normalized_profit_targets),
                rsi_entry_rule,
            )
        except (TypeError, ValueError, OverflowError):
            return False
        selected_pair = (float(buy_rsi), float(profit_target_multiple))
        if not expected_pairs or selected_pair not in expected_pairs:
            return False
        if expected_strategy_fingerprint is not None and (
            type(expected_strategy_fingerprint) is not str
            or expected_strategy_fingerprint != derived_strategy_fingerprint
        ):
            return False
        if not strategy_config_matches_fingerprint(
            conn,
            asset_symbol,
            signal_symbol,
            derived_strategy_fingerprint,
        ):
            return False
        if not _strategy_rows_match_config(
            conn,
            asset_symbol,
            signal_symbol,
            expected_pairs,
            base_cfg=base_cfg,
            rsi_entry_rule=rsi_entry_rule,
        ):
            return False
    persisted_signal_symbol = _persisted_strategy_signal_symbol(
        conn,
        asset_symbol,
        signal_symbol,
    )
    has_canonical_signal_history = (
        conn.execute(
            "SELECT 1 FROM market_data WHERE symbol = ? LIMIT 1",
            (persisted_signal_symbol,),
        ).fetchone()
        is not None
    )
    resolved_rsi_period = None
    rsi_entry_rule_value = None
    if has_canonical_signal_history:
        resolved_rsi_period = _resolve_strategy_rsi_period(
            conn,
            persisted_signal_symbol,
            rsi_period,
        )
        if resolved_rsi_period is None:
            return False
        try:
            rsi_entry_rule_value = rsi_entry_rule_code(rsi_entry_rule)
        except (TypeError, ValueError, OverflowError):
            return False
    summary_rows = conn.execute(
        f"""
        SELECT {_STRATEGY_SUMMARY_INTEGRITY_COLUMN_SQL}, integrity_digest
        FROM strategy_summary
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND buy_rsi = ?
          AND profit_target_multiple = ?
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    ).fetchall()
    if len(summary_rows) != 1:
        return False
    summary_row = summary_rows[0]
    summary_validation_row = _strategy_summary_validation_row(summary_row)
    if not _strategy_summary_integrity_row_is_valid(
        summary_row
    ) or not _strategy_summary_validation_row_is_semantically_valid(summary_validation_row):
        return False
    summary = summary_validation_row[2:]
    if any(value is None for value in summary[:19]):
        return False
    start_date, end_date = str(summary[0]), str(summary[1])
    trading_days, summary_trades = summary[2:4]
    summary_return_count = summary[7]
    summary_excess_count = summary[10]
    summary_positive_count = summary[13]
    if (
        not isinstance(trading_days, int)
        or not isinstance(summary_trades, int)
        or not isinstance(summary_return_count, int)
        or not isinstance(summary_excess_count, int)
        or not isinstance(summary_positive_count, int)
        or trading_days <= 0
        or summary_trades < 0
        or summary_return_count != trading_days - 1
        or not 0 <= summary_excess_count <= summary_return_count
        or not 0 <= summary_positive_count <= summary_return_count
    ):
        return False

    state = load_strategy_state(
        conn,
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
        rsi_period=resolved_rsi_period,
        rsi_entry_rule=rsi_entry_rule,
    )
    if state is None:
        return False
    state_values = (
        state["start_date"],
        state["last_date"],
        state["in_position"],
        state["pending_action"],
        state["prev_equity"],
        state["trades_executed"],
    )
    if any(value is None for value in state_values):
        return False
    state_start, state_last = str(state["start_date"]), str(state["last_date"])
    state_in_position = state["in_position"]
    state_pending = state["pending_action"]
    state_equity = state["prev_equity"]
    state_trades = state["trades_executed"]
    if (
        state_start != start_date
        or state_last != end_date
        or state_in_position not in (0, 1)
        or state_pending not in {"none", "buy"}
        or not isinstance(state_trades, int)
        or state_trades < 0
        or state_trades != summary_trades
    ):
        return False
    if base_cfg is not None:
        canonical_replay = _canonical_strategy_grid_replay(
            conn,
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            config_pairs=[(float(buy_rsi), float(profit_target_multiple))],
            base_cfg=base_cfg,
            start_date=start_date,
            end_date=end_date,
            rsi_entry_rule=rsi_entry_rule,
        )
        expected_replay = (
            None if canonical_replay is None else canonical_replay.get((float(buy_rsi), float(profit_target_multiple)))
        )
        summary_rollup = _summary_rollup_from_row(summary[4:19])
        if (
            expected_replay is None
            or summary_rollup is None
            or not _strategy_states_exactly_match(state, expected_replay[0])
            or summary_rollup != expected_replay[1]
        ):
            return False

    if equity_rows is None:
        equity_rows = _strategy_equity_curve_rows(
            conn,
            asset_symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
        )
    if any(
        not _strategy_equity_integrity_row_is_valid(
            row,
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            buy_rsi=buy_rsi,
            profit_target_multiple=profit_target_multiple,
        )
        for row in equity_rows
    ):
        return False
    equity_dates = [str(row[0]) for row in equity_rows]
    if (
        len(equity_dates) != trading_days
        or len(set(equity_dates)) != len(equity_dates)
        or not equity_dates
        or equity_dates[0] != start_date
        or equity_dates[-1] != end_date
    ):
        return False

    market_rows = conn.execute(
        """
            SELECT date, open, high, close,
                   TYPEOF(open), TYPEOF(high), TYPEOF(close)
            FROM market_data
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
        """,
        (asset_symbol, start_date, end_date),
    ).fetchall()
    market_dates = [str(row[0]) for row in market_rows]
    if equity_dates != market_dates:
        return False
    canonical_risk_free_returns = _canonical_risk_free_returns_for_dates(
        conn,
        pd.DatetimeIndex(pd.to_datetime(market_dates, format="%Y-%m-%d")),
    )
    if canonical_risk_free_returns is None:
        return False
    canonical_rsi_values: np.ndarray | None = None
    if has_canonical_signal_history:
        if resolved_rsi_period is None:
            return False
        canonical_signal_rsi = _canonical_saved_rsi_series(
            conn,
            persisted_signal_symbol,
            resolved_rsi_period,
        )
        if canonical_signal_rsi is None:
            return False
        aligned_rsi, _observation_dates = align_signal_values_to_asset_sessions(
            canonical_signal_rsi,
            pd.DatetimeIndex(pd.to_datetime(market_dates, format="%Y-%m-%d")),
        )
        canonical_rsi_values = aligned_rsi.to_numpy(dtype=np.float64)
        if not np.isfinite(canonical_rsi_values).any():
            return False

    canonical_curve_results: tuple | None = None
    if base_cfg is not None:
        if canonical_rsi_values is None:
            return False
        try:
            canonical_curve_results = run_single_equity_curve(
                np.asarray([row[1] for row in market_rows], dtype=np.float64),
                np.asarray([row[2] for row in market_rows], dtype=np.float64),
                np.asarray([row[3] for row in market_rows], dtype=np.float64),
                canonical_rsi_values,
                canonical_risk_free_returns,
                float(buy_rsi),
                float(profit_target_multiple),
                float(base_cfg.initial_capital),
                _trading_cost_rate(base_cfg),
                rsi_entry_rule_code(rsi_entry_rule),
            )
        except (TypeError, ValueError, OverflowError):
            return False
        if any(len(canonical_curve_results[index]) != len(equity_rows) for index in range(7)):
            return False

    records: list[dict] = []
    previous_equity: float | None = None
    previous_in_position = 0
    previous_pending = "none"
    previous_trades = 0
    position_start_equity: float | None = None
    position_entry_open: float | None = None
    position_entry_date: str | None = None
    position_max_high: float | None = None
    position_target_price: float | None = None
    position_entry_close: float | None = None
    position_shares: float | None = None
    position_cost_rate: float | None = None
    inferred_cost_rate: float | None = None
    # Entry and exit cost rates are derived by subtracting ratios near one.
    # Bound only that arithmetic's float64 uncertainty; a fixed decimal floor
    # can hide a real negative cost at large supported account values.
    cost_rate_roundoff = 32.0 * _float64_inward_ulp(1.0)
    for row_idx, (row, market_row) in enumerate(zip(equity_rows, market_rows, strict=True)):
        _, equity, daily_return, risk_free_return, in_position, action, pending, trades, _digest = row
        if any(storage_type not in {"integer", "real"} for storage_type in market_row[4:]):
            return False
        try:
            market_open = float(market_row[1])
            market_high = float(market_row[2])
            market_close = float(market_row[3])
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not math.isfinite(market_open)
            or not math.isfinite(market_high)
            or not math.isfinite(market_close)
            or market_open <= 0.0
            or market_close <= 0.0
            or any(
                market_high < observed_price
                and observed_price - market_high
                > 4.0
                * max(
                    _float64_inward_ulp(observed_price),
                    _float64_inward_ulp(market_high),
                )
                for observed_price in (market_open, market_close)
            )
        ):
            return False
        if (
            in_position not in (0, 1)
            or action not in {"none", "buy", "sell"}
            or pending not in {"none", "buy"}
            or (in_position == 1 and pending != "none")
            or not isinstance(trades, int)
            or trades < previous_trades
        ):
            return False
        try:
            equity_value = float(equity)
            daily_return_value = float(daily_return)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(equity_value) or equity_value <= 0.0 or not math.isfinite(daily_return_value):
            return False
        if canonical_curve_results is not None and (
            equity_value != float(canonical_curve_results[0][row_idx])
            or daily_return_value != float(canonical_curve_results[1][row_idx])
            or int(in_position) != int(canonical_curve_results[3][row_idx])
            or action != _action_label(int(canonical_curve_results[4][row_idx]))
            or pending != _action_label(int(canonical_curve_results[5][row_idx]))
            or trades != int(canonical_curve_results[6][row_idx])
        ):
            return False
        if risk_free_return is not None:
            try:
                risk_free_value = float(risk_free_return)
            except (TypeError, ValueError, OverflowError):
                return False
            if not math.isfinite(risk_free_value):
                return False
        else:
            risk_free_value = np.nan
        expected_risk_free = float(canonical_risk_free_returns[row_idx])
        if not _canonical_optional_float_values_match(
            risk_free_value,
            expected_risk_free,
        ):
            return False

        expected_return = 0.0 if previous_equity is None else equity_value / previous_equity - 1.0
        if (
            daily_return_value != expected_return
            if base_cfg is not None
            else not _complete_curve_values_match(daily_return_value, expected_return)
        ):
            return False

        trade_delta = trades - previous_trades
        held_sell_fill_price: float | None = None
        if previous_in_position == 1:
            if position_start_equity is None or position_entry_open is None or position_target_price is None:
                return False
            if market_open >= position_target_price:
                held_sell_fill_price = market_open
            elif market_high >= position_target_price:
                held_sell_fill_price = position_target_price
        if trade_delta == 0:
            if action != "none" or in_position != previous_in_position:
                return False
            if previous_pending == "buy" and previous_in_position == 0 and previous_equity is not None:
                # Every retained market row has a positive open, so a pending
                # all-cash buy cannot remain deferred on this session.
                return False
            if previous_in_position == 1 and held_sell_fill_price is not None:
                # The simulator's resting limit must execute on the first
                # target-touching session; a later high cannot justify a hold.
                return False
            if (
                previous_equity is not None
                and previous_in_position == 0
                and in_position == 0
                and (
                    equity_value != previous_equity
                    if base_cfg is not None
                    else not _complete_curve_values_match(equity_value, previous_equity)
                )
            ):
                return False
        elif trade_delta == 1:
            if action == "buy":
                if previous_in_position != 0 or previous_pending != "buy" or in_position != 1:
                    return False
                if previous_equity is None:
                    return False
                position_start_equity = previous_equity
                position_entry_open = market_open
                position_entry_date = str(row[0])
                position_max_high = market_high
                position_entry_close = market_close
                try:
                    position_target_price = _target_sell_price(
                        market_open,
                        profit_target_multiple,
                    )
                except (TypeError, ValueError, OverflowError):
                    return False
                if market_open >= position_target_price or market_high >= position_target_price:
                    # A buy whose target is reached on its entry session is
                    # represented as a two-trade, end-flat row.
                    return False
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    position_shares = equity_value / market_close
                    position_cost_rate = previous_equity / (position_shares * market_open) - 1.0
                if (
                    not math.isfinite(position_shares)
                    or position_shares <= 0.0
                    or not math.isfinite(position_cost_rate)
                    or position_cost_rate >= 1.0
                    or position_cost_rate < -cost_rate_roundoff
                ):
                    return False
                position_cost_rate = max(position_cost_rate, 0.0)
                if inferred_cost_rate is None:
                    inferred_cost_rate = position_cost_rate
                elif not math.isclose(
                    position_cost_rate,
                    inferred_cost_rate,
                    rel_tol=0.0,
                    abs_tol=cost_rate_roundoff,
                ):
                    return False
            elif action == "sell":
                if previous_in_position != 1 or in_position != 0:
                    return False
                if held_sell_fill_price is None:
                    return False
            else:
                return False
        elif trade_delta == 2:
            if action != "sell" or previous_in_position != 0 or previous_pending != "buy" or in_position != 0:
                return False
            if previous_equity is None:
                return False
            position_start_equity = previous_equity
            position_entry_open = market_open
            position_entry_date = str(row[0])
            position_max_high = market_high
            try:
                position_target_price = _target_sell_price(
                    market_open,
                    profit_target_multiple,
                )
            except (TypeError, ValueError, OverflowError):
                return False
            if market_open >= position_target_price:
                held_sell_fill_price = market_open
            elif market_high >= position_target_price:
                held_sell_fill_price = position_target_price
            else:
                return False
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                gross_ratio = equity_value * position_entry_open / (position_start_equity * held_sell_fill_price)
                round_trip_cost_rate = (1.0 - gross_ratio) / (1.0 + gross_ratio)
            if (
                not math.isfinite(gross_ratio)
                or gross_ratio <= 0.0
                or gross_ratio > 1.0 + cost_rate_roundoff
                or not math.isfinite(round_trip_cost_rate)
                or round_trip_cost_rate >= 1.0
                or round_trip_cost_rate < -cost_rate_roundoff
            ):
                return False
            round_trip_cost_rate = max(round_trip_cost_rate, 0.0)
            if inferred_cost_rate is None:
                inferred_cost_rate = round_trip_cost_rate
            elif not math.isclose(
                round_trip_cost_rate,
                inferred_cost_rate,
                rel_tol=0.0,
                abs_tol=cost_rate_roundoff,
            ):
                return False
        else:
            return False

        if canonical_rsi_values is not None:
            canonical_rsi = canonical_rsi_values[row_idx]
            expected_pending = "none"
            if np.isfinite(canonical_rsi) and in_position == 0:
                entry_signal = (
                    canonical_rsi >= buy_rsi if rsi_entry_rule_value == RSI_ENTRY_UPPER else canonical_rsi <= buy_rsi
                )
                if entry_signal:
                    expected_pending = "buy"
            if pending != expected_pending:
                return False

        if previous_in_position == 1:
            if position_max_high is None:
                return False
            position_max_high = max(position_max_high, market_high)
            if position_shares is None or position_cost_rate is None:
                return False
            expected_position_equity = (
                position_shares * market_close
                if in_position == 1
                else position_shares * held_sell_fill_price * (1.0 - position_cost_rate)
                if held_sell_fill_price is not None
                else math.nan
            )
            if not math.isfinite(expected_position_equity):
                return False
            if position_entry_close is None or position_entry_close <= 0.0:
                return False
            valuation_price = market_close if in_position == 1 else float(held_sell_fill_price)
            if equity_value != expected_position_equity:
                # Exact kernel-origin equality needs no error budget.  In
                # particular, scaling an entry-equity ULP by a finite but
                # extreme price ratio can overflow even though the observed
                # and reconstructed account values are identical.
                account_roundoff = (
                    32.0 * _float64_inward_ulp(position_start_equity) * max(1.0, valuation_price / position_entry_close)
                )
                account_roundoff = max(
                    account_roundoff,
                    32.0
                    * max(
                        _float64_inward_ulp(equity_value),
                        _float64_inward_ulp(expected_position_equity),
                    ),
                )
                account_roundoff = min(
                    np.longdouble(account_roundoff),
                    _float64_roundoff_tolerance(
                        equity_value,
                        expected_position_equity,
                    ),
                )
                if (
                    not np.isfinite(account_roundoff)
                    or abs(np.longdouble(equity_value) - np.longdouble(expected_position_equity)) > account_roundoff
                ):
                    return False
        if position_start_equity is not None:
            if position_entry_open is None or position_max_high is None or position_target_price is None:
                return False
            with np.errstate(over="ignore", invalid="ignore"):
                if in_position == 0:
                    if held_sell_fill_price is None:
                        return False
                    maximum_market_equity = position_start_equity * held_sell_fill_price / position_entry_open
                else:
                    maximum_market_equity = position_start_equity * position_max_high / position_entry_open
            if (
                not math.isfinite(maximum_market_equity)
                or equity_value > maximum_market_equity
                and not _complete_curve_values_match(
                    equity_value,
                    maximum_market_equity,
                )
            ):
                return False
            if in_position == 0:
                position_start_equity = None
                position_entry_open = None
                position_entry_date = None
                position_max_high = None
                position_target_price = None
                position_entry_close = None
                position_shares = None
                position_cost_rate = None
        records.append(
            {
                "equity": equity_value,
                "daily_return": daily_return_value,
                "risk_free_return": risk_free_value,
            }
        )
        previous_equity = equity_value
        previous_in_position = int(in_position)
        previous_pending = str(pending)
        previous_trades = trades

    if not _zero_trade_rollup_is_semantically_valid(
        summary_validation_row,
        equity_records=records,
    ):
        return False

    if (
        previous_in_position != state_in_position
        or previous_pending != state_pending
        or previous_trades != state_trades
        # These persisted values originate from the same kernel observations;
        # no arithmetic separates them. Exact equality prevents a fixed
        # absolute comparison floor from accepting unrelated tiny accounts.
        or previous_equity != float(state_equity)
        or float(equity_rows[0][1]) != float(summary[4])
        or previous_equity != float(summary[5])
    ):
        return False
    if state_in_position == 1:
        try:
            state_entry_price = float(state["entry_price"])
            state_shares = float(state["shares"])
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            position_entry_open is None
            or position_entry_date is None
            or position_entry_close is None
            or position_shares is None
            or (
                state_entry_price != position_entry_open
                if base_cfg is not None
                else not _complete_curve_values_match(state_entry_price, position_entry_open)
            )
            or state.get("entry_date") != position_entry_date
        ):
            return False
        if state_shares != position_shares:
            # Dividing the entry-equity ULP by a subnormal close can overflow
            # the tolerance itself.  Bypass that derived comparison when the
            # persisted and reconstructed float64 share values already match
            # exactly; a non-exact value still fails closed on overflow.
            share_roundoff = 32.0 * _float64_inward_ulp(position_start_equity) / position_entry_close
            share_roundoff = max(
                share_roundoff,
                32.0
                * max(
                    _float64_inward_ulp(state_shares),
                    _float64_inward_ulp(position_shares),
                ),
            )
            share_roundoff = min(
                np.longdouble(share_roundoff),
                _float64_roundoff_tolerance(
                    state_shares,
                    position_shares,
                ),
            )
            if (
                not np.isfinite(share_roundoff)
                or abs(np.longdouble(state_shares) - np.longdouble(position_shares)) > share_roundoff
            ):
                return False

    summary_rollup = _summary_rollup_from_row(summary[4:19])
    if summary_rollup is None:
        return False
    curve_rollup = _update_summary_rollup(SummaryRollup(), records)
    if (
        summary_rollup != curve_rollup
        if base_cfg is not None
        else not _complete_curve_rollups_match(summary_rollup, curve_rollup)
    ):
        return False
    curve_metrics = _rollup_metrics(curve_rollup)
    for stored_value, metric_name in zip(
        summary[19:],
        (
            "total_return",
            "cagr",
            "annualized_vol",
            "sharpe",
            "kelly_fraction",
            "hit_rate",
        ),
        strict=True,
    ):
        metrics_match = (
            _canonical_optional_float_values_match(stored_value, curve_metrics[metric_name])
            if base_cfg is not None
            else _complete_curve_optional_values_match(stored_value, curve_metrics[metric_name])
        )
        if not metrics_match:
            return False
    return True


def load_complete_strategy_equity_curve(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    *,
    rsi_period: int | None = None,
    rsi_entry_rule: str = "lower",
    base_cfg: BacktestConfig | None = None,
    expected_buy_rsi_values: list[float] | None = None,
    expected_profit_target_values: list[float] | None = None,
    expected_strategy_fingerprint: str | None = None,
    allow_unbound_backtest_config: bool = False,
) -> pd.DataFrame | None:
    """Return the retained curve only when its summary, state, and rows agree."""
    if not conn.in_transaction:
        with _consistent_storage_read_snapshot(conn):
            return load_complete_strategy_equity_curve(
                conn,
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                rsi_period=rsi_period,
                rsi_entry_rule=rsi_entry_rule,
                base_cfg=base_cfg,
                expected_buy_rsi_values=expected_buy_rsi_values,
                expected_profit_target_values=expected_profit_target_values,
                expected_strategy_fingerprint=expected_strategy_fingerprint,
                allow_unbound_backtest_config=allow_unbound_backtest_config,
            )
    equity_rows = _strategy_equity_curve_rows(
        conn,
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
    )
    if not _equity_curve_is_complete(
        conn,
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
        equity_rows=equity_rows,
        rsi_period=rsi_period,
        rsi_entry_rule=rsi_entry_rule,
        base_cfg=base_cfg,
        expected_buy_rsi_values=expected_buy_rsi_values,
        expected_profit_target_values=expected_profit_target_values,
        expected_strategy_fingerprint=expected_strategy_fingerprint,
        allow_unbound_backtest_config=allow_unbound_backtest_config,
    ):
        return None
    equity_df = pd.DataFrame(
        ((row[0], row[1]) for row in equity_rows),
        columns=["date", "equity"],
    )
    equity_df["date"] = pd.to_datetime(equity_df["date"], format="%Y-%m-%d", errors="raise")
    return equity_df


def load_complete_strategy_latest_action(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    *,
    rsi_period: int | None = None,
    rsi_entry_rule: str = "lower",
    base_cfg: BacktestConfig | None = None,
    expected_buy_rsi_values: list[float] | None = None,
    expected_profit_target_values: list[float] | None = None,
    expected_strategy_fingerprint: str | None = None,
    allow_unbound_backtest_config: bool = False,
) -> str | None:
    """Return the final action only after authenticating the complete curve."""
    if not conn.in_transaction:
        with _consistent_storage_read_snapshot(conn):
            return load_complete_strategy_latest_action(
                conn,
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                rsi_period=rsi_period,
                rsi_entry_rule=rsi_entry_rule,
                base_cfg=base_cfg,
                expected_buy_rsi_values=expected_buy_rsi_values,
                expected_profit_target_values=expected_profit_target_values,
                expected_strategy_fingerprint=expected_strategy_fingerprint,
                allow_unbound_backtest_config=allow_unbound_backtest_config,
            )
    equity_rows = _strategy_equity_curve_rows(
        conn,
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
    )
    if not _equity_curve_is_complete(
        conn,
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
        equity_rows=equity_rows,
        rsi_period=rsi_period,
        rsi_entry_rule=rsi_entry_rule,
        base_cfg=base_cfg,
        expected_buy_rsi_values=expected_buy_rsi_values,
        expected_profit_target_values=expected_profit_target_values,
        expected_strategy_fingerprint=expected_strategy_fingerprint,
        allow_unbound_backtest_config=allow_unbound_backtest_config,
    ):
        return None
    return str(equity_rows[-1][5])


def _replace_best_equity_curve(
    conn: sqlite3.Connection,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    rsi_entry_rule: str,
) -> None:
    persisted_signal_symbol = _persisted_strategy_signal_symbol(
        conn,
        asset_symbol,
        signal_symbol,
    )
    full_data = _load_saved_strategy_market_data(
        conn,
        asset_symbol,
        signal_symbol,
        persisted_signal_symbol,
    )
    if full_data.empty:
        return

    canonical_signal_close = load_saved_close_series(conn, persisted_signal_symbol)
    if canonical_signal_close.empty:
        return
    rsi = ensure_rsi_values(
        conn,
        persisted_signal_symbol,
        base_cfg.rsi_period,
        canonical_signal_close,
        rebuild=False,
    )
    open_prices, high_prices, close_prices, rsi_values, risk_free_returns = _market_arrays(
        full_data,
        rsi,
        asset_symbol,
    )
    (
        equity_values,
        daily_returns,
        risk_free_returns,
        in_position_values,
        action_executed_values,
        pending_action_values,
        trades_executed_values,
        cash,
        shares,
        in_position,
        entry_price,
        pending_action,
        prev_equity,
        trades_executed,
    ) = run_single_equity_curve(
        open_prices,
        high_prices,
        close_prices,
        rsi_values,
        risk_free_returns,
        buy_rsi,
        profit_target_multiple,
        base_cfg.initial_capital,
        _trading_cost_rate(base_cfg),
        rsi_entry_rule_code(rsi_entry_rule),
    )
    date_strings = [_date_str(date) for date in full_data.index]
    entry_rows = np.flatnonzero(action_executed_values == ACTION_BUY)
    state = {
        "start_date": date_strings[0] if date_strings else None,
        "last_date": date_strings[-1] if date_strings else None,
        "cash": float(cash),
        "shares": float(shares),
        "in_position": bool(in_position),
        "entry_price": float(entry_price),
        "entry_date": (date_strings[int(entry_rows[-1])] if bool(in_position) and len(entry_rows) else None),
        "pending_action": _action_label(int(pending_action)),
        "prev_equity": float(prev_equity),
        "trades_executed": int(trades_executed),
    }
    records = _equity_records_from_arrays(
        date_strings=date_strings,
        asset_symbol=asset_symbol,
        signal_symbol=signal_symbol,
        buy_rsi=buy_rsi,
        profit_target_multiple=profit_target_multiple,
        equity_values=equity_values,
        daily_returns=daily_returns,
        risk_free_returns=risk_free_returns,
        in_position_values=in_position_values,
        action_executed_values=action_executed_values,
        pending_action_values=pending_action_values,
        trades_executed_values=trades_executed_values,
    )
    clear_equity_records(conn, asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)
    save_equity_records(conn, records)

    rollup = _update_summary_rollup(SummaryRollup(), records)
    save_strategy_summary(
        conn,
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
        state,
        rollup,
    )


def _process_asset_grid(
    conn: sqlite3.Connection,
    data: pd.DataFrame,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rebuild: bool,
    signal_history: pd.DataFrame | None = None,
    strategy_fingerprint: str | None = None,
    authoritative_histories: dict[str, pd.DataFrame] | None = None,
    presynchronized_authoritative_symbols: set[str] | None = None,
    market_history_observation_run_id: str | None = None,
    commit: bool = True,
    grid_compute_observer: Callable[[float], None] | None = None,
    rsi_entry_rule: str = "lower",
    isolate_strategy_signal_history: bool | None = None,
) -> None:
    if isolate_strategy_signal_history is not None and type(isolate_strategy_signal_history) is not bool:
        raise ValueError("isolate_strategy_signal_history must be a boolean or None.")
    rsi_entry_rule_value = rsi_entry_rule_code(rsi_entry_rule)
    symbols = list(dict.fromkeys([asset_symbol, signal_symbol, RISK_FREE_SYMBOL]))
    private_signal_storage_symbol = _strategy_signal_cache_symbol(asset_symbol, signal_symbol)
    has_private_signal_history = bool(
        private_signal_storage_symbol != signal_symbol
        and conn.execute(
            "SELECT 1 FROM market_data WHERE symbol = ? LIMIT 1",
            (private_signal_storage_symbol,),
        ).fetchone()
    )
    scoped_signal_history = bool(
        private_signal_storage_symbol != signal_symbol
        and (
            isolate_strategy_signal_history is True
            or (isolate_strategy_signal_history is None and has_private_signal_history)
        )
    )
    retire_private_signal_history = bool(
        private_signal_storage_symbol != signal_symbol
        and isolate_strategy_signal_history is False
        and has_private_signal_history
    )
    signal_storage_symbol = private_signal_storage_symbol if scoped_signal_history else signal_symbol
    stored_signal_history = (
        _history_in_storage_namespace(signal_history, signal_symbol, signal_storage_symbol)
        if scoped_signal_history and signal_history is not None
        else None
    )
    discarded_strategy_pair = (asset_symbol, signal_symbol) if rebuild else None
    dependency_excluded_pair = (
        (asset_symbol, signal_symbol)
        if scoped_signal_history or retire_private_signal_history
        else discarded_strategy_pair
    )
    expected_state_generation = strategy_state_generation(conn)
    canonical_strategy_fingerprint = strategy_config_fingerprint(
        base_cfg,
        buy_rsi_values,
        profit_target_values,
        rsi_entry_rule,
    )
    if strategy_fingerprint is None:
        strategy_fingerprint = canonical_strategy_fingerprint
    elif type(strategy_fingerprint) is not str or strategy_fingerprint != canonical_strategy_fingerprint:
        raise ValueError("The supplied strategy fingerprint does not match the canonical configuration.")
    persisted_state_is_complete = False
    if not rebuild:
        expected_pairs = _strategy_config_pairs(buy_rsi_values, profit_target_values)
        persisted_state_is_complete = strategy_config_matches_fingerprint(
            conn,
            asset_symbol,
            signal_symbol,
            strategy_fingerprint,
        ) and _strategy_rows_match_config(
            conn,
            asset_symbol,
            signal_symbol,
            expected_pairs,
            base_cfg=base_cfg,
            rsi_entry_rule=rsi_entry_rule,
        )
    presynchronized_symbols = presynchronized_authoritative_symbols or set()
    dependent_alignments_before: dict[tuple[str, str], str | None] = {}
    dependent_alignment_asset_filter: str | None = None
    processed_alignment_revised = False
    signal_dates_before: set[str] | None = None
    signal_dependency_alignments_before: dict[
        str,
        dict[tuple[str, str], str | None],
    ] = {}
    signal_market_dates_before: dict[str, set[str]] = {}
    strategy_signal_alignment_before: str | None = None
    strategy_signal_dates_before: set[str] = set()
    strategy_signal_had_history = False
    strategy_signal_revised = retire_private_signal_history
    if authoritative_histories is None and (scoped_signal_history or retire_private_signal_history):
        raise ValueError(
            "An existing or requested pair-scoped signal history requires complete authoritative histories."
        )
    if authoritative_histories is not None:
        authoritative_symbols = set(authoritative_histories)
        unexpected_symbols = authoritative_symbols.difference(symbols)
        if unexpected_symbols:
            unexpected = ", ".join(sorted(unexpected_symbols))
            raise ValueError(f"Unexpected authoritative market symbols: {unexpected}.")
        unexpected_presynchronized = presynchronized_symbols.difference(authoritative_symbols)
        if unexpected_presynchronized:
            unexpected = ", ".join(sorted(unexpected_presynchronized))
            raise ValueError(f"Presynchronized market symbols lack authoritative histories: {unexpected}.")
        if scoped_signal_history and signal_symbol not in authoritative_symbols:
            raise ValueError("Pair-scoped signal history requires a distinct canonical authoritative signal history.")
        if (
            scoped_signal_history
            and signal_history is not None
            and authoritative_histories[signal_symbol] is signal_history
        ):
            raise ValueError("Pair-scoped signal history cannot also be the global canonical signal history.")
        fallback_symbols = [symbol for symbol in symbols if symbol not in authoritative_symbols]

        # Validate the complete batch before the first synchronization write so
        # a malformed later symbol cannot leave a partially updated state when
        # this function is used outside the workflow's surrounding transaction.
        for symbol, history in authoritative_histories.items():
            validate_market_data_frame(history, symbol, source="Authoritative history")
        if scoped_signal_history:
            if stored_signal_history is None:
                raise ValueError("Pair-scoped workflow processing requires a complete signal history.")
            if stored_signal_history.empty:
                raise AssetMarketDataError("Pair-scoped signal history must not be empty.")
            validate_market_data_frame(
                stored_signal_history,
                signal_storage_symbol,
                source="Pair-scoped signal history",
            )
        if signal_history is not None and signal_symbol in fallback_symbols:
            validate_market_data_frame(signal_history, signal_symbol, source="Signal history")
        if retire_private_signal_history:
            conn.execute("DELETE FROM market_data WHERE symbol = ?", (private_signal_storage_symbol,))
            conn.execute("DELETE FROM rsi_values WHERE signal_symbol = ?", (private_signal_storage_symbol,))
            conn.execute(
                "DELETE FROM market_history_removal_candidates WHERE symbol = ?",
                (private_signal_storage_symbol,),
            )
        if scoped_signal_history:
            strategy_signal_dates_before = _saved_market_dates(conn, signal_storage_symbol)
            strategy_signal_had_history = bool(strategy_signal_dates_before)
            strategy_signal_alignment_before = _saved_strategy_signal_alignment(
                conn,
                asset_symbol=asset_symbol,
                signal_symbol=signal_symbol,
                signal_storage_symbol=signal_storage_symbol,
            )
        signal_dependency_alignments_before = {
            symbol: _saved_signal_dependent_alignments(
                conn,
                symbol,
                exclude_strategy_pair=dependency_excluded_pair,
            )
            for symbol in symbols
            if symbol != RISK_FREE_SYMBOL and symbol not in presynchronized_symbols
        }
        signal_market_dates_before = {
            symbol: _saved_market_dates(conn, symbol) for symbol in signal_dependency_alignments_before
        }

        # Before mutating a shared signal, snapshot every persisted dependent.
        # If the signal was genuinely synchronized earlier in this transaction,
        # only this asset's calendar can still change its saved final alignment.
        dependent_alignment_asset_filter = (
            asset_symbol if signal_symbol in presynchronized_symbols and dependency_excluded_pair is None else None
        )
        dependent_alignments_before = _saved_signal_dependent_alignments(
            conn,
            signal_symbol,
            asset_symbol=dependent_alignment_asset_filter,
            exclude_strategy_pair=dependency_excluded_pair,
        )
        if signal_symbol not in presynchronized_symbols:
            signal_dates_before = _saved_market_dates(conn, signal_symbol)
        fallback_revisions = _revised_market_symbols(conn, data, fallback_symbols)
        fallback_revisions.update(
            _historically_added_market_symbols(
                conn,
                data,
                fallback_symbols,
                exclude_strategy_pair=dependency_excluded_pair,
            )
        )
        signal_history_revisions = (
            _revised_market_symbols(conn, signal_history, [signal_symbol])
            if signal_history is not None and signal_symbol in fallback_symbols
            else set()
        )
        if signal_history is not None and signal_symbol in fallback_symbols:
            signal_history_revisions.update(
                _historically_added_market_symbols(
                    conn,
                    signal_history,
                    [signal_symbol],
                    exclude_strategy_pair=dependency_excluded_pair,
                )
            )

        revised_symbols = set()
        for symbol, history in authoritative_histories.items():
            if symbol in presynchronized_symbols:
                continue
            # A date that is a normal tail append for this symbol's own market
            # table can still revise a strategy that uses the symbol in the
            # other dependency role and has already processed farther ahead.
            revised_symbols.update(
                _historically_added_market_symbols(
                    conn,
                    history,
                    [symbol],
                    exclude_strategy_pair=dependency_excluded_pair,
                )
            )
            if _synchronize_market_data_history(
                conn,
                history,
                symbol,
                observation_run_id=market_history_observation_run_id,
            ):
                revised_symbols.add(symbol)

        if scoped_signal_history:
            assert stored_signal_history is not None
            strategy_signal_revised = _synchronize_market_data_history(
                conn,
                stored_signal_history,
                signal_storage_symbol,
                observation_run_id=market_history_observation_run_id,
                confirm_boundary_removals=False,
            )
            strategy_signal_alignment_after = _saved_strategy_signal_alignment(
                conn,
                asset_symbol=asset_symbol,
                signal_symbol=signal_symbol,
                signal_storage_symbol=signal_storage_symbol,
            )
            strategy_checkpoint_row = conn.execute(
                """
                SELECT MAX(last_date)
                FROM strategy_state
                WHERE asset_symbol = ? AND signal_symbol = ?
                """,
                (asset_symbol, signal_symbol),
            ).fetchone()
            strategy_checkpoint = strategy_checkpoint_row[0] if strategy_checkpoint_row else None
            added_strategy_signal_dates = _saved_market_dates(conn, signal_storage_symbol).difference(
                strategy_signal_dates_before
            )
            strategy_signal_revised = bool(
                strategy_signal_revised
                or strategy_signal_alignment_after != strategy_signal_alignment_before
                or (
                    strategy_checkpoint is not None
                    and any(date <= str(strategy_checkpoint) for date in added_strategy_signal_dates)
                )
                or (
                    not strategy_signal_had_history
                    and conn.execute(
                        """
                        SELECT 1
                        FROM strategy_state
                        WHERE asset_symbol = ? AND signal_symbol = ?
                        LIMIT 1
                        """,
                        (asset_symbol, signal_symbol),
                    ).fetchone()
                    is not None
                )
            )

        # The complete histories above are canonical.  Only retain the legacy
        # merged-frame save for symbols without a canonical history.
        if fallback_symbols:
            save_market_data(conn, data, fallback_symbols)
        if signal_history is not None and signal_symbol in fallback_symbols:
            save_market_data(conn, signal_history, [signal_symbol])
        revised_symbols.update(fallback_revisions)
        # Simulations and persisted state must always use exactly the histories
        # synchronized above.  The independently downloaded merged frame may
        # cover different dates or contain a provider revision.
        data = (
            _load_saved_strategy_market_data(
                conn,
                asset_symbol,
                signal_symbol,
                signal_storage_symbol,
            )
            if scoped_signal_history
            else load_saved_market_data(conn, symbols)
        )
    else:
        if presynchronized_symbols:
            unexpected = ", ".join(sorted(presynchronized_symbols))
            raise ValueError(f"Presynchronized market symbols lack authoritative histories: {unexpected}.")
        # Legacy/direct callers do not have canonical per-symbol histories to
        # synchronize. Detect every exact price correction before replacing the
        # persisted frame so compact strategy state is invalidated whenever its
        # inputs change.
        signal_dependency_alignments_before = {
            symbol: _saved_signal_dependent_alignments(
                conn,
                symbol,
                exclude_strategy_pair=dependency_excluded_pair,
            )
            for symbol in symbols
            if symbol != RISK_FREE_SYMBOL
        }
        signal_market_dates_before = {
            symbol: _saved_market_dates(conn, symbol) for symbol in signal_dependency_alignments_before
        }
        # The merged frame and optional signal history can both append or
        # revise this shared signal, including while rebuilding a new asset.
        dependent_alignments_before = _saved_signal_dependent_alignments(
            conn,
            signal_symbol,
            exclude_strategy_pair=dependency_excluded_pair,
        )
        signal_dates_before = _saved_market_dates(conn, signal_symbol)
        # Rebuilding the current asset does not make shared signal or benchmark
        # overwrites harmless to other compact states. Detect existing-session
        # revisions before either frame is persisted in every mode.
        revised_symbols = _revised_market_symbols(conn, data, symbols)
        revised_symbols.update(
            _historically_added_market_symbols(
                conn,
                data,
                symbols,
                exclude_strategy_pair=dependency_excluded_pair,
            )
        )
        signal_history_revisions = (
            set() if signal_history is None else _revised_market_symbols(conn, signal_history, [signal_symbol])
        )
        if signal_history is not None:
            signal_history_revisions.update(
                _historically_added_market_symbols(
                    conn,
                    signal_history,
                    [signal_symbol],
                    exclude_strategy_pair=dependency_excluded_pair,
                )
            )
        save_market_data(conn, data, symbols)
        if signal_history is not None:
            save_market_data(conn, signal_history, [signal_symbol])

    signal_alignment_revised_symbols: set[str] = set()
    for updated_symbol, alignments_before in signal_dependency_alignments_before.items():
        alignments_after = _saved_signal_dependent_alignments(
            conn,
            updated_symbol,
            exclude_strategy_pair=dependency_excluded_pair,
        )
        if any(
            alignments_after.get(dependent) != prior_observation_date
            for dependent, prior_observation_date in alignments_before.items()
        ):
            signal_alignment_revised_symbols.add(updated_symbol)
    signal_horizon_revised_symbols = {
        updated_symbol
        for updated_symbol, dates_before in signal_market_dates_before.items()
        if _signal_additions_affect_processed_inputs(
            conn,
            updated_symbol,
            _saved_market_dates(conn, updated_symbol).difference(dates_before),
            exclude_strategy_pair=dependency_excluded_pair,
        )
    }

    dependent_alignments_after = _saved_signal_dependent_alignments(
        conn,
        signal_symbol,
        asset_symbol=dependent_alignment_asset_filter,
        exclude_strategy_pair=dependency_excluded_pair,
    )
    processed_alignment_revised = any(
        dependent_alignments_after.get(dependent) != prior_observation_date
        for dependent, prior_observation_date in dependent_alignments_before.items()
    )
    historical_signal_addition = False
    if signal_dates_before is not None and dependent_alignments_before:
        added_signal_dates = _saved_market_dates(conn, signal_symbol).difference(signal_dates_before)
        historical_signal_addition = any(
            signal_date <= last_date
            for signal_date in added_signal_dates
            for _dependent_asset, last_date in dependent_alignments_before
        )

    dependent_revised_symbols = revised_symbols.union(
        signal_history_revisions,
        signal_alignment_revised_symbols,
        signal_horizon_revised_symbols,
    )
    if historical_signal_addition or processed_alignment_revised:
        # A normal signal tail can still be historical relative to an asset
        # that has already processed beyond that observation. A changed final
        # alignment likewise means the compact pending action used the wrong
        # signal observation.
        dependent_revised_symbols.add(signal_symbol)
    # Keep global invalidation below, but do not replay a private pair merely
    # because the unused canonical representation of its signal changed.
    current_pair_revised_symbols = dependent_revised_symbols.difference(
        {signal_symbol} if scoped_signal_history else set()
    )
    signal_revised = signal_symbol in current_pair_revised_symbols or strategy_signal_revised
    corrected_existing_session = bool(current_pair_revised_symbols) or strategy_signal_revised
    if RISK_FREE_SYMBOL in dependent_revised_symbols:
        # ^IRX feeds every strategy's Sharpe rollup, so a historical correction
        # cannot be repaired safely by rebuilding only the current asset.
        clear_all_strategy_state(conn)
        expected_state_generation = strategy_state_generation(conn)
    else:
        for revised_symbol in sorted(dependent_revised_symbols):
            # A symbol may be traded by several signal strategies and may also
            # serve as another strategy's signal. Every such compact state
            # depends on the corrected candle or calendar.
            _clear_market_symbol_dependent_state(conn, revised_symbol)
        if strategy_signal_revised:
            # A private signal snapshot belongs only to this pair. Its provider
            # correction must replay that pair without invalidating another
            # asset that intentionally uses an incompatible snapshot.
            clear_asset_state(conn, asset_symbol, signal_symbol)
    if corrected_existing_session:
        # The compact resume state cannot replay a changed historical bar.  Load
        # the corrected full history we just persisted and recompute this asset.
        data = (
            _load_saved_strategy_market_data(
                conn,
                asset_symbol,
                signal_symbol,
                signal_storage_symbol,
            )
            if scoped_signal_history
            else load_saved_market_data(conn, symbols)
        )
        rebuild = True
    elif not rebuild and not persisted_state_is_complete:
        # Another asset sharing this signal may have invalidated the compact
        # state after this asset's preflight check.  Rebuild from saved history
        # rather than continuing from an incompatible tail state.
        data = (
            _load_saved_strategy_market_data(
                conn,
                asset_symbol,
                signal_symbol,
                signal_storage_symbol,
            )
            if scoped_signal_history
            else load_saved_market_data(conn, symbols)
        )
        rebuild = True

    if not rebuild and persisted_state_is_complete and not data.empty:
        # A prior run can persist fresh market rows and fail before advancing
        # the compact strategy state.  A later tail-only request must not jump
        # over those already-saved sessions merely because they are absent from
        # the caller's frame.  Widen the compute window only when such a gap is
        # present, preserving the ordinary one-row incremental fast path.
        asset_columns = [f"{asset_symbol}_{field}" for field in _MARKET_DATA_FIELDS]
        incoming_asset_data = data.loc[~data.loc[:, asset_columns].isna().all(axis=1)]
        if not incoming_asset_data.empty:
            incoming_asset_dates = {_date_str(date) for date in incoming_asset_data.index}
            incoming_asset_end = max(incoming_asset_dates)
            checkpoint_rows = conn.execute(
                """
                SELECT DISTINCT last_date
                FROM strategy_state
                WHERE asset_symbol = ? AND signal_symbol = ?
                """,
                (asset_symbol, signal_symbol),
            ).fetchall()
            if len(checkpoint_rows) == 1 and checkpoint_rows[0][0] is not None:
                checkpoint_date = str(checkpoint_rows[0][0])
                saved_unprocessed_dates = {
                    str(row[0])
                    for row in conn.execute(
                        """
                        SELECT date
                        FROM market_data
                        WHERE symbol = ?
                          AND date > ?
                          AND date <= ?
                        """,
                        (asset_symbol, checkpoint_date, incoming_asset_end),
                    ).fetchall()
                }
                if not saved_unprocessed_dates.issubset(incoming_asset_dates):
                    saved_data = (
                        _load_saved_strategy_market_data(
                            conn,
                            asset_symbol,
                            signal_symbol,
                            signal_storage_symbol,
                        )
                        if scoped_signal_history
                        else load_saved_market_data(conn, symbols)
                    )
                    data = saved_data.loc[[_date_str(date) <= incoming_asset_end for date in saved_data.index]]

    if data.empty:
        if commit:
            conn.commit()
        return
    if rebuild:
        clear_asset_state(conn, asset_symbol, signal_symbol)

    canonical_signal_close = load_saved_close_series(conn, signal_storage_symbol)
    if canonical_signal_close.empty:
        canonical_signal_close = data[f"{signal_symbol}_Close"].dropna().sort_index()
    rsi = ensure_rsi_values(
        conn,
        signal_storage_symbol,
        base_cfg.rsi_period,
        canonical_signal_close,
        rebuild=rebuild or signal_revised,
    )

    config_pairs = [
        (float(buy_rsi), float(profit_target_multiple))
        for buy_rsi in buy_rsi_values
        for profit_target_multiple in profit_target_values
    ]
    if not config_pairs:
        if commit:
            conn.commit()
        return

    open_prices, high_prices, close_prices, rsi_values, risk_free_returns = _market_arrays(
        data,
        rsi,
        asset_symbol,
    )
    date_values = pd.to_datetime(data.index).to_numpy(dtype="datetime64[ns]")
    date_strings = [_date_str(date) for date in data.index]
    config_count = len(config_pairs)

    buy_rsi_array = np.empty(config_count, dtype=np.float64)
    profit_target_array = np.empty(config_count, dtype=np.float64)
    start_indices = np.empty(config_count, dtype=np.int64)
    cash_values = np.empty(config_count, dtype=np.float64)
    share_values = np.empty(config_count, dtype=np.float64)
    in_position_values = np.empty(config_count, dtype=np.bool_)
    entry_price_values = np.empty(config_count, dtype=np.float64)
    pending_action_values = np.empty(config_count, dtype=np.int64)
    prev_equity_values = np.empty(config_count, dtype=np.float64)
    trades_executed_values = np.empty(config_count, dtype=np.int64)

    first_equity_values = np.empty(config_count, dtype=np.float64)
    last_equity_values = np.empty(config_count, dtype=np.float64)
    running_max_equity_values = np.empty(config_count, dtype=np.float64)
    return_count_values = np.empty(config_count, dtype=np.int64)
    return_sum_values = np.empty(config_count, dtype=np.float64)
    return_sum_squares_values = np.empty(config_count, dtype=np.float64)
    return_mean_values = np.empty(config_count, dtype=np.float64)
    return_m2_values = np.empty(config_count, dtype=np.float64)
    excess_return_count_values = np.empty(config_count, dtype=np.int64)
    excess_return_sum_values = np.empty(config_count, dtype=np.float64)
    excess_return_sum_squares_values = np.empty(config_count, dtype=np.float64)
    excess_return_mean_values = np.empty(config_count, dtype=np.float64)
    excess_return_m2_values = np.empty(config_count, dtype=np.float64)
    positive_return_count_values = np.empty(config_count, dtype=np.int64)
    max_drawdown_values = np.empty(config_count, dtype=np.float64)
    resume_close_values = np.full(config_count, np.nan, dtype=np.float64)
    history_prefix_observation_counts = np.zeros(config_count, dtype=np.int64)
    state_start_dates: list[str | None] = []
    state_entry_dates: list[str | None] = []
    saved_close_by_date: dict[str, float] = {}
    history_prefix_counts_by_window: dict[tuple[str, str], int] = {}
    states_by_config = {} if rebuild else _load_strategy_states_for_asset(conn, asset_symbol, signal_symbol)
    rollups_by_config = {} if rebuild else _load_strategy_summary_rollups_for_asset(conn, asset_symbol, signal_symbol)

    for config_idx, (buy_rsi, profit_target_multiple) in enumerate(config_pairs):
        config_key = (buy_rsi, profit_target_multiple)
        state = states_by_config.get(config_key)
        if state is None:
            state = initial_strategy_state(base_cfg)
            rollup = SummaryRollup()
        else:
            rollup = rollups_by_config.get(config_key)
            if rollup is None:
                raise ValueError(
                    "Persisted strategy history has no authenticated summary rollup for "
                    f"{asset_symbol}/{signal_symbol} at buy RSI {buy_rsi} and target "
                    f"{profit_target_multiple}; rebuild is required."
                )

        if state["last_date"] is None:
            start_idx = 0
        else:
            last_date_value = np.datetime64(pd.Timestamp(state["last_date"]).to_datetime64())
            start_idx = int(np.searchsorted(date_values, last_date_value, side="right"))

        (
            first_equity,
            last_equity,
            running_max_equity,
            return_count,
            return_sum,
            return_sum_squares,
            excess_return_count,
            excess_return_sum,
            excess_return_sum_squares,
            positive_return_count,
            max_drawdown,
            return_mean,
            return_m2,
            excess_return_mean,
            excess_return_m2,
        ) = _rollup_to_arrays(rollup)

        buy_rsi_array[config_idx] = buy_rsi
        profit_target_array[config_idx] = profit_target_multiple
        start_indices[config_idx] = start_idx
        cash_values[config_idx] = float(state["cash"])
        share_values[config_idx] = float(state["shares"])
        in_position_values[config_idx] = bool(state["in_position"])
        entry_price_values[config_idx] = np.nan if pd.isna(state["entry_price"]) else float(state["entry_price"])
        pending_action_values[config_idx] = _action_code(state["pending_action"])
        prev_equity_values[config_idx] = float(state["prev_equity"])
        trades_executed_values[config_idx] = int(state["trades_executed"])
        first_equity_values[config_idx] = first_equity
        last_equity_values[config_idx] = last_equity
        running_max_equity_values[config_idx] = running_max_equity
        return_count_values[config_idx] = return_count
        return_sum_values[config_idx] = return_sum
        return_sum_squares_values[config_idx] = return_sum_squares
        return_mean_values[config_idx] = return_mean
        return_m2_values[config_idx] = return_m2
        excess_return_count_values[config_idx] = excess_return_count
        excess_return_sum_values[config_idx] = excess_return_sum
        excess_return_sum_squares_values[config_idx] = excess_return_sum_squares
        excess_return_mean_values[config_idx] = excess_return_mean
        excess_return_m2_values[config_idx] = excess_return_m2
        positive_return_count_values[config_idx] = positive_return_count
        max_drawdown_values[config_idx] = max_drawdown
        state_start_dates.append(state["start_date"])
        state_entry_dates.append(state.get("entry_date"))

        state_start_date = state["start_date"]
        state_last_date = state["last_date"]
        if state_start_date is not None and state_last_date is not None:
            normalized_start_date = str(state_start_date)
            normalized_last_date = str(state_last_date)
            history_window = (normalized_start_date, normalized_last_date)
            if history_window not in history_prefix_counts_by_window:
                history_prefix_counts_by_window[history_window] = int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM market_data
                        WHERE symbol = ?
                          AND date >= ?
                          AND date < ?
                          AND date <= ?
                        """,
                        (
                            asset_symbol,
                            normalized_start_date,
                            date_strings[0],
                            normalized_last_date,
                        ),
                    ).fetchone()[0]
                )
            history_prefix_observation_counts[config_idx] = history_prefix_counts_by_window[history_window]

        if state["in_position"] and state["last_date"] is not None:
            state_last_date = str(state["last_date"])
            if state_last_date not in saved_close_by_date:
                close_row = conn.execute(
                    "SELECT close FROM market_data WHERE symbol = ? AND date = ?",
                    (asset_symbol, state_last_date),
                ).fetchone()
                if close_row is None or close_row[0] is None:
                    raise ValueError(
                        "An in-position resumable state requires its last closing price in saved "
                        f"market data for {asset_symbol} on {state_last_date}."
                    )
                saved_close = float(close_row[0])
                if not np.isfinite(saved_close) or saved_close <= 0.0:
                    raise ValueError(
                        "An in-position resumable state requires a positive finite last closing "
                        f"price for {asset_symbol} on {state_last_date}."
                    )
                saved_close_by_date[state_last_date] = saved_close
            resume_close_values[config_idx] = saved_close_by_date[state_last_date]

    grid_compute_started = time.perf_counter()
    try:
        (
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
        ) = run_grid_summary(
            open_prices,
            high_prices,
            close_prices,
            rsi_values,
            risk_free_returns,
            buy_rsi_array,
            profit_target_array,
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
            _trading_cost_rate(base_cfg),
            rsi_entry_rule_value,
            return_mean_values=return_mean_values,
            return_m2_values=return_m2_values,
            excess_return_mean_values=excess_return_mean_values,
            excess_return_m2_values=excess_return_m2_values,
            resume_close_values=resume_close_values,
            history_prefix_observation_counts=history_prefix_observation_counts,
        )
    finally:
        if grid_compute_observer is not None:
            grid_compute_observer(max(0.0, time.perf_counter() - grid_compute_started))

    strategy_state_rows: list[tuple] = []
    strategy_summary_rows: list[tuple] = []
    for config_idx, (buy_rsi, profit_target_multiple) in enumerate(config_pairs):
        if not updated[config_idx]:
            continue

        start_idx = int(start_indices[config_idx])
        entry_row_index = int(out_entry_row_index[config_idx])
        state = {
            "start_date": state_start_dates[config_idx] or date_strings[start_idx],
            "last_date": date_strings[-1],
            "cash": float(out_cash[config_idx]),
            "shares": float(out_shares[config_idx]),
            "in_position": bool(out_in_position[config_idx]),
            "entry_price": float(out_entry_price[config_idx]),
            "entry_date": (
                date_strings[entry_row_index]
                if bool(out_in_position[config_idx]) and entry_row_index >= 0
                else state_entry_dates[config_idx]
                if bool(out_in_position[config_idx])
                else None
            ),
            "pending_action": _action_label(int(out_pending_action[config_idx])),
            "prev_equity": float(out_prev_equity[config_idx]),
            "trades_executed": int(out_trades_executed[config_idx]),
        }
        rollup = _rollup_from_arrays(
            float(out_first_equity[config_idx]),
            float(out_last_equity[config_idx]),
            float(out_running_max_equity[config_idx]),
            int(out_return_count[config_idx]),
            float(out_return_sum[config_idx]),
            float(out_return_sum_squares[config_idx]),
            int(out_excess_return_count[config_idx]),
            float(out_excess_return_sum[config_idx]),
            float(out_excess_return_sum_squares[config_idx]),
            int(out_positive_return_count[config_idx]),
            float(out_max_drawdown[config_idx]),
            float(out_return_mean[config_idx]),
            float(out_return_m2[config_idx]),
            float(out_excess_return_mean[config_idx]),
            float(out_excess_return_m2[config_idx]),
        )
        strategy_state_rows.append(
            _strategy_state_row(
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                state,
            )
        )
        strategy_summary_rows.append(
            _strategy_summary_row(
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                state,
                rollup,
            )
        )

    save_strategy_states(conn, strategy_state_rows)
    save_strategy_summaries(conn, strategy_summary_rows)

    # Reconstructing a curve also recomputes its summary with sequential
    # arithmetic.  That can move a numerically tied strategy by one ULP in the
    # SQL ordering, so converge on a winner before deleting any candidate
    # curve.  Each grid point is rebuilt at most once; revisiting an incomplete
    # point means reconstruction made no durable progress and must fail closed.
    rebuilt_configs: set[tuple[float, float]] = set()
    for _ in range(len(set(config_pairs)) + 1):
        best_config = _best_summary_config(
            conn,
            asset_symbol,
            signal_symbol,
            rsi_entry_rule,
        )
        if best_config is None:
            break
        best_buy_rsi, best_profit_target_multiple = best_config
        if _equity_curve_is_complete(
            conn,
            asset_symbol,
            signal_symbol,
            best_buy_rsi,
            best_profit_target_multiple,
            rsi_period=base_cfg.rsi_period,
            rsi_entry_rule=rsi_entry_rule,
            allow_unbound_backtest_config=True,
        ):
            prune_non_best_equity_records(
                conn,
                asset_symbol,
                signal_symbol,
                best_buy_rsi,
                best_profit_target_multiple,
            )
            break
        if best_config in rebuilt_configs:
            raise RuntimeError("Best-strategy equity curve reconstruction did not converge.")
        rebuilt_configs.add(best_config)
        _replace_best_equity_curve(
            conn,
            base_cfg,
            asset_symbol,
            signal_symbol,
            best_buy_rsi,
            best_profit_target_multiple,
            rsi_entry_rule,
        )
    else:
        raise RuntimeError("Best-strategy equity curve reconstruction exceeded the grid size.")

    if strategy_state_generation(conn) != expected_state_generation:
        raise RuntimeError("Strategy state generation changed during an atomic asset update.")
    save_strategy_config(conn, asset_symbol, signal_symbol, strategy_fingerprint)
    if commit:
        conn.commit()


def process_asset_grid(
    conn: sqlite3.Connection,
    data: pd.DataFrame,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rebuild: bool,
    signal_history: pd.DataFrame | None = None,
    strategy_fingerprint: str | None = None,
    authoritative_histories: dict[str, pd.DataFrame] | None = None,
    presynchronized_authoritative_symbols: set[str] | None = None,
    market_history_observation_run_id: str | None = None,
    commit: bool = True,
    grid_compute_observer: Callable[[float], None] | None = None,
    rsi_entry_rule: str = "lower",
    isolate_strategy_signal_history: bool | None = None,
) -> None:
    """Atomically synchronize and process one strategy grid.

    Workflow callers pass ``commit=False`` while an outer immediate
    transaction owns commit and rollback. Direct callers retain the historical
    ``commit=True`` behavior, with a savepoint ensuring that a rejected asset
    cannot leave partial market, RSI, or compact strategy mutations behind.
    """
    rsi_entry_rule_code(rsi_entry_rule)
    normalized_buy_rsi, normalized_profit_targets = _validated_strategy_grid_inputs(
        base_cfg,
        buy_rsi_values,
        profit_target_values,
    )
    buy_rsi_values = list(normalized_buy_rsi)
    profit_target_values = list(normalized_profit_targets)
    canonical_strategy_fingerprint = strategy_config_fingerprint(
        base_cfg,
        buy_rsi_values,
        profit_target_values,
        rsi_entry_rule,
    )
    if strategy_fingerprint is not None and (
        type(strategy_fingerprint) is not str or strategy_fingerprint != canonical_strategy_fingerprint
    ):
        raise ValueError("The supplied strategy fingerprint does not match the canonical configuration.")
    strategy_fingerprint = canonical_strategy_fingerprint
    arguments = (
        conn,
        data,
        base_cfg,
        asset_symbol,
        signal_symbol,
        buy_rsi_values,
        profit_target_values,
        rebuild,
    )
    keyword_arguments = {
        "signal_history": signal_history,
        "strategy_fingerprint": strategy_fingerprint,
        "authoritative_histories": authoritative_histories,
        "presynchronized_authoritative_symbols": presynchronized_authoritative_symbols,
        "market_history_observation_run_id": market_history_observation_run_id,
        "grid_compute_observer": grid_compute_observer,
        "rsi_entry_rule": rsi_entry_rule,
        "isolate_strategy_signal_history": isolate_strategy_signal_history,
    }
    if not commit:
        _process_asset_grid(
            *arguments,
            commit=False,
            **keyword_arguments,
        )
        return

    # Reusing a static name is safe: SQLite savepoints nest, and ROLLBACK TO /
    # RELEASE target the most recent savepoint with that name. Keeping this
    # independent of the module-level clock also leaves timing instrumentation
    # free to patch ``time`` without affecting transaction control.
    savepoint_name = "process_asset_grid_atomic"
    owns_transaction = not conn.in_transaction
    savepoint_active = False
    try:
        if owns_transaction:
            conn.execute("BEGIN")
        conn.execute(f"SAVEPOINT {savepoint_name}")
        savepoint_active = True
        _process_asset_grid(
            *arguments,
            commit=False,
            **keyword_arguments,
        )
        conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
        savepoint_active = False
        _commit_owned_transaction(conn)
    except BaseException as exc:
        if savepoint_active:
            _rollback_and_release_savepoint(conn, savepoint_name, exc)
        if owns_transaction:
            _rollback_owned_operation_if_active(
                conn,
                exc,
                operation="atomic asset-grid processing",
            )
        raise
