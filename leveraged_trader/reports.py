from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .accounting import (
    MANAGED_QUANTITY_ABSOLUTE_TOLERANCE,
    MANAGED_QUANTITY_MAX_TOLERANCE,
    MANAGED_QUANTITY_RELATIVE_TOLERANCE,
    managed_quantity_tolerance,
    managed_residual_quantity_is_negligible,
    managed_value_reconciliation_tolerance,
)
from .config import BacktestConfig, validate_backtest_configuration
from .storage import (
    alpaca_order_id_is_canonical,
    load_aligned_rsi_for_asset_session,
    load_best_strategy_summary,
    load_complete_strategy_equity_curve,
    load_complete_strategy_latest_action,
    load_strategy_state,
    rsi_entry_rule_code,
    signal_observation_is_fresh,
    strategy_config_fingerprint,
)

REALIZED_PNL_COLUMNS = [
    "Closed Positions",
    "Complete Closed Positions",
    "Incomplete Closed Positions",
    "Total Buy Cost",
    "Total Sell Value",
    "Realized P/L",
    "Realized P/L %",
]
REALIZED_PNL_WORKFLOW_COLUMNS = ["Workflow", *REALIZED_PNL_COLUMNS]

_StrategyReportCacheKey = tuple[str, str, str]
_StrategyReportCacheValue = tuple[dict[str, object], pd.DataFrame, dict[str, object], str]
_StrategyReportCache = dict[_StrategyReportCacheKey, _StrategyReportCacheValue]


def _alpaca_broker_timestamp_is_canonical(value: object) -> bool:
    """Return whether a report input is one canonical persisted broker time."""
    if type(value) is not str:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError):
        return False
    if parsed.tzinfo is None:
        return False
    return parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z") == value


def _close_after_snapshot_rollback_failure(
    conn: sqlite3.Connection,
    failure: BaseException,
) -> None:
    """Invalidate a connection whose owned read snapshot could not be ended."""
    try:
        conn.close()
    except BaseException as close_failure:
        failure.add_note(f"Failed to close the SQLite connection after read-snapshot rollback failed: {close_failure}")


def _rollback_snapshot_after_failure(
    conn: sqlite3.Connection,
    failure: BaseException,
    *,
    context: str,
) -> None:
    """Preserve an operation failure while making a failed snapshot unusable."""
    try:
        conn.rollback()
    except BaseException as rollback_failure:
        failure.add_note(f"Failed to roll back {context}: {rollback_failure}")
        _close_after_snapshot_rollback_failure(conn, failure)


@contextmanager
def _consistent_read_snapshot(conn: sqlite3.Connection) -> Iterator[None]:
    """Own a deferred read snapshot only when the caller does not own one."""
    owns_snapshot = not conn.in_transaction
    if owns_snapshot:
        try:
            conn.execute("BEGIN")
        except BaseException as exc:
            _rollback_snapshot_after_failure(
                conn,
                exc,
                context="pending SQLite read-snapshot setup",
            )
            raise
    try:
        yield
    except BaseException as exc:
        if owns_snapshot:
            _rollback_snapshot_after_failure(
                conn,
                exc,
                context="the owned SQLite read snapshot",
            )
        raise
    else:
        if owns_snapshot:
            try:
                conn.rollback()
            except BaseException as rollback_failure:
                _close_after_snapshot_rollback_failure(conn, rollback_failure)
                raise


def _workflow_rsi_entry_rule(workflow_asset: object) -> str:
    explicit_rule = str(getattr(workflow_asset, "rsi_entry_rule", "")).strip().lower()
    if explicit_rule in {"lower", "upper"}:
        return explicit_rule
    workflow = str(getattr(workflow_asset, "workflow", "")).strip().lower()
    direction = str(getattr(workflow_asset, "direction", "")).strip().lower()
    return "upper" if workflow == "short" or direction == "inverse" else "lower"


def _validate_report_strategy_provenance(
    *,
    base_cfg: BacktestConfig | None,
    expected_buy_rsi_values: list[float] | None,
    expected_profit_target_values: list[float] | None,
    rsi_entry_rule: str | None,
    expected_strategy_fingerprint: str | None,
    allow_unbound_backtest_config: bool,
) -> None:
    """Require enough inputs to derive, rather than merely trust, provenance."""
    if base_cfg is None:
        if not allow_unbound_backtest_config:
            raise ValueError(
                "base_cfg is required to authenticate reports; pass "
                "allow_unbound_backtest_config=True only for structural legacy inspection."
            )
        if (
            expected_buy_rsi_values is not None
            or expected_profit_target_values is not None
            or expected_strategy_fingerprint is not None
        ):
            raise ValueError("Structural legacy report inspection cannot accept partial strategy provenance.")
        return
    if expected_buy_rsi_values is None or expected_profit_target_values is None:
        raise ValueError(
            "expected_buy_rsi_values and expected_profit_target_values are required "
            "to authenticate reports against their complete strategy grid."
        )
    if rsi_entry_rule is None:
        # A mixed-workflow summary derives each row's rule from workflow metadata,
        # so there is no single caller-supplied fingerprint to compare here.  Still
        # validate the complete configuration/grid contract up front; individual
        # curve loads below derive and verify the rule-specific fingerprint.
        strategy_config_fingerprint(
            base_cfg,
            expected_buy_rsi_values,
            expected_profit_target_values,
            "lower",
        )
        if expected_strategy_fingerprint is not None:
            raise ValueError(
                "rsi_entry_rule is required when expected_strategy_fingerprint is supplied for an authenticated report."
            )
        return
    derived_fingerprint = strategy_config_fingerprint(
        base_cfg,
        expected_buy_rsi_values,
        expected_profit_target_values,
        rsi_entry_rule,
    )
    if expected_strategy_fingerprint is not None and (
        type(expected_strategy_fingerprint) is not str or expected_strategy_fingerprint != derived_fingerprint
    ):
        raise ValueError(
            "expected_strategy_fingerprint does not match the supplied backtest "
            "configuration, strategy grid, and RSI entry rule."
        )


def summarize_saved_results(
    conn: sqlite3.Connection,
    workflow_assets: pd.DataFrame,
    *,
    rsi_period: int | None = None,
    rsi_entry_rule: str | None = None,
    base_cfg: BacktestConfig | None = None,
    expected_buy_rsi_values: list[float] | None = None,
    expected_profit_target_values: list[float] | None = None,
    expected_strategy_fingerprint: str | None = None,
    allow_unbound_backtest_config: bool = False,
    _strategy_report_cache: _StrategyReportCache | None = None,
    _workflow_results_preverified: bool = False,
    deadline_check: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if type(_workflow_results_preverified) is not bool:
        raise ValueError("_workflow_results_preverified must be a boolean.")
    _validate_report_strategy_provenance(
        base_cfg=base_cfg,
        expected_buy_rsi_values=expected_buy_rsi_values,
        expected_profit_target_values=expected_profit_target_values,
        rsi_entry_rule=rsi_entry_rule,
        expected_strategy_fingerprint=expected_strategy_fingerprint,
        allow_unbound_backtest_config=allow_unbound_backtest_config,
    )
    with _consistent_read_snapshot(conn):
        return _summarize_saved_results_snapshot(
            conn,
            workflow_assets,
            rsi_period=rsi_period,
            rsi_entry_rule=rsi_entry_rule,
            base_cfg=base_cfg,
            expected_buy_rsi_values=expected_buy_rsi_values,
            expected_profit_target_values=expected_profit_target_values,
            expected_strategy_fingerprint=expected_strategy_fingerprint,
            strategy_report_cache=_strategy_report_cache,
            workflow_results_preverified=_workflow_results_preverified,
            deadline_check=deadline_check,
        )


def _summarize_saved_results_snapshot(
    conn: sqlite3.Connection,
    workflow_assets: pd.DataFrame,
    *,
    rsi_period: int | None,
    rsi_entry_rule: str | None,
    base_cfg: BacktestConfig | None,
    expected_buy_rsi_values: list[float] | None,
    expected_profit_target_values: list[float] | None,
    expected_strategy_fingerprint: str | None,
    strategy_report_cache: _StrategyReportCache | None,
    workflow_results_preverified: bool,
    deadline_check: Callable[[], None] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    best_curves = []
    for workflow_asset in workflow_assets.itertuples(index=False):
        if deadline_check is not None:
            deadline_check()
        asset_symbol = workflow_asset.symbol
        signal_symbol = workflow_asset.rsi_symbol
        entry_rule = rsi_entry_rule if rsi_entry_rule is not None else _workflow_rsi_entry_rule(workflow_asset)
        if strategy_report_cache is None:
            best_row = load_best_strategy_summary(
                conn,
                asset_symbol,
                signal_symbol,
                entry_rule,
            )
        else:
            best_row = load_best_strategy_summary(
                conn,
                asset_symbol,
                signal_symbol,
                entry_rule,
                strategy_state_preverified=True,
            )
        if best_row is None:
            continue

        curve_kwargs: dict[str, object] = {}
        if strategy_report_cache is not None:
            curve_kwargs["strategy_state_preverified"] = True
        if workflow_results_preverified:
            equity_rows = conn.execute(
                """
                SELECT date, equity
                FROM strategy_equity
                WHERE asset_symbol = ?
                  AND signal_symbol = ?
                  AND buy_rsi = ?
                  AND profit_target_multiple = ?
                ORDER BY date
                """,
                (
                    asset_symbol,
                    signal_symbol,
                    float(best_row["buy_rsi"]),
                    float(best_row["profit_target_multiple"]),
                ),
            ).fetchall()
            equity_df = pd.DataFrame(equity_rows, columns=["date", "equity"])
            if not equity_df.empty:
                equity_df["date"] = pd.to_datetime(
                    equity_df["date"],
                    format="%Y-%m-%d",
                    errors="raise",
                )
        else:
            equity_df = load_complete_strategy_equity_curve(
                conn,
                asset_symbol,
                signal_symbol,
                float(best_row["buy_rsi"]),
                float(best_row["profit_target_multiple"]),
                rsi_period=rsi_period,
                rsi_entry_rule=entry_rule,
                base_cfg=base_cfg,
                expected_buy_rsi_values=expected_buy_rsi_values,
                expected_profit_target_values=expected_profit_target_values,
                expected_strategy_fingerprint=expected_strategy_fingerprint,
                allow_unbound_backtest_config=base_cfg is None,
                **curve_kwargs,
            )
        if equity_df is None or equity_df.empty:
            continue

        if strategy_report_cache is not None:
            buy_rsi = float(best_row["buy_rsi"])
            profit_target_multiple = float(best_row["profit_target_multiple"])
            state = load_strategy_state(
                conn,
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                rsi_period=rsi_period,
                rsi_entry_rule=entry_rule,
            )
            latest_action_row = conn.execute(
                """
                SELECT action_executed
                FROM strategy_equity
                WHERE asset_symbol = ?
                  AND signal_symbol = ?
                  AND buy_rsi = ?
                  AND profit_target_multiple = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
            ).fetchone()
            if state is None or latest_action_row is None or latest_action_row[0] not in {"none", "buy", "sell"}:
                continue
            strategy_report_cache[(str(asset_symbol), str(signal_symbol), entry_rule)] = (
                best_row,
                equity_df,
                state,
                str(latest_action_row[0]),
            )

        best_equity = equity_df.set_index("date")["equity"]
        summary_rows.append(
            {
                "Asset": asset_symbol,
                "RSI Symbol": signal_symbol,
                "Start Date": best_row["start_date"],
                "End Date": best_row["end_date"],
                "Trading Days": int(best_row["trading_days"]),
                "Buy RSI": float(best_row["buy_rsi"]),
                "Sell Return Multiple": float(best_row["profit_target_multiple"]),
                "Trades Executed": int(best_row["trades_executed"]),
                "Total Return": best_row["total_return"],
                "CAGR": best_row["cagr"],
                "Annualized Vol": best_row["annualized_vol"],
                "Sharpe": best_row["sharpe"],
                "Kelly Fraction": best_row["kelly_fraction"],
                "Max Drawdown": best_row["max_drawdown"],
                "Hit Rate": best_row["hit_rate"],
            }
        )
        best_curves.append(best_equity.rename(f"{asset_symbol}_RSI_Strategy"))

    optimization_summary = pd.DataFrame(summary_rows)
    if not optimization_summary.empty:
        optimization_summary = optimization_summary.sort_values(
            by=["Sharpe", "Total Return", "CAGR"],
            ascending=False,
            na_position="last",
        )

    curves = pd.concat(best_curves, axis=1, join="outer", sort=False).sort_index() if best_curves else pd.DataFrame()

    return optimization_summary, curves


def build_buy_signal_report(
    conn: sqlite3.Connection,
    optimization_summary: pd.DataFrame,
    rsi_period: int,
    *,
    rsi_entry_rule: str = "lower",
    base_cfg: BacktestConfig | None = None,
    expected_buy_rsi_values: list[float] | None = None,
    expected_profit_target_values: list[float] | None = None,
    expected_strategy_fingerprint: str | None = None,
    allow_unbound_backtest_config: bool = False,
    _strategy_report_cache: _StrategyReportCache | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    return build_pending_action_report(
        conn,
        optimization_summary,
        rsi_period,
        pending_action_filter="buy",
        require_multiple_trades=True,
        min_sharpe=1.0,
        rsi_entry_rule=rsi_entry_rule,
        base_cfg=base_cfg,
        expected_buy_rsi_values=expected_buy_rsi_values,
        expected_profit_target_values=expected_profit_target_values,
        expected_strategy_fingerprint=expected_strategy_fingerprint,
        allow_unbound_backtest_config=allow_unbound_backtest_config,
        _strategy_report_cache=_strategy_report_cache,
        deadline_check=deadline_check,
    )


def build_sell_signal_report(
    conn: sqlite3.Connection,
    optimization_summary: pd.DataFrame,
    rsi_period: int,
    *,
    rsi_entry_rule: str = "lower",
    base_cfg: BacktestConfig | None = None,
    expected_buy_rsi_values: list[float] | None = None,
    expected_profit_target_values: list[float] | None = None,
    expected_strategy_fingerprint: str | None = None,
    allow_unbound_backtest_config: bool = False,
    _strategy_report_cache: _StrategyReportCache | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    return build_pending_action_report(
        conn,
        optimization_summary,
        rsi_period,
        pending_action_filter="sell",
        require_multiple_trades=False,
        min_sharpe=None,
        rsi_entry_rule=rsi_entry_rule,
        base_cfg=base_cfg,
        expected_buy_rsi_values=expected_buy_rsi_values,
        expected_profit_target_values=expected_profit_target_values,
        expected_strategy_fingerprint=expected_strategy_fingerprint,
        allow_unbound_backtest_config=allow_unbound_backtest_config,
        _strategy_report_cache=_strategy_report_cache,
        deadline_check=deadline_check,
    )


def build_alpaca_realized_pnl_summary(
    conn: sqlite3.Connection,
    *,
    include_workflow: bool = False,
) -> pd.DataFrame:
    with _consistent_read_snapshot(conn):
        return _build_alpaca_realized_pnl_summary_snapshot(
            conn,
            include_workflow=include_workflow,
        )


def _report_quantity_residual_is_negligible(
    residual_quantity: object,
    *,
    quantity_values: tuple[object, ...],
    mark_prices: tuple[object, ...],
    value_values: tuple[object, ...],
) -> bool:
    """Apply the shared share-and-notional policy to loaded report values."""
    try:
        residual = float(residual_quantity)
    except (TypeError, ValueError, OverflowError):
        return False
    if residual == 0.0:
        return True
    normalized_quantities: list[float] = [abs(residual)]
    for value in quantity_values:
        if value is None or pd.isna(value):
            continue
        try:
            numeric = abs(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
        if not np.isfinite(numeric):
            return False
        normalized_quantities.append(numeric)
    normalized_prices: list[float] = []
    for value in mark_prices:
        if value is None or pd.isna(value):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return False
        if not np.isfinite(numeric):
            return False
        if numeric > 0.0:
            normalized_prices.append(numeric)
    mark_price = max(normalized_prices, default=0.0)
    quantity_scale = max(normalized_quantities)
    normalized_values: list[float] = [quantity_scale * mark_price]
    for value in value_values:
        if value is None or pd.isna(value):
            continue
        try:
            numeric = abs(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
        if not np.isfinite(numeric):
            return False
        normalized_values.append(numeric)
    value_scale = max(normalized_values)
    return managed_residual_quantity_is_negligible(
        residual,
        quantity_scale=quantity_scale,
        mark_price=mark_price,
        value_scale=value_scale,
    )


def _build_alpaca_realized_pnl_summary_snapshot(
    conn: sqlite3.Connection,
    *,
    include_workflow: bool,
) -> pd.DataFrame:
    orphan_fill = conn.execute(
        """
        SELECT fills.managed_position_id
        FROM alpaca_managed_sell_fills AS fills
        LEFT JOIN alpaca_managed_positions AS positions
          ON positions.id = fills.managed_position_id
        WHERE positions.id IS NULL
        LIMIT 1
        """
    ).fetchone()
    if orphan_fill is not None:
        raise ValueError(
            "Realized P/L cannot be summarized while a managed sell fill has no parent "
            f"position (managed_position_id={orphan_fill[0]!r})."
        )

    ambiguous_order = conn.execute(
        """
        SELECT alpaca_order_id
        FROM alpaca_managed_sell_fills
        GROUP BY alpaca_order_id
        HAVING COUNT(DISTINCT managed_position_id) > 1
        LIMIT 1
        """
    ).fetchone()
    if ambiguous_order is not None:
        raise ValueError(
            "Realized P/L cannot be summarized while one Alpaca sell order belongs to "
            f"multiple managed positions (alpaca_order_id={ambiguous_order[0]!r})."
        )

    invalid_fill_identity_positions: set[int] = set()
    invalid_fill_notional_positions: set[int] = set()
    for managed_position_id, order_id, fill_qty, fill_value, submitted_qty, submitted_limit in conn.execute(
        """
        SELECT managed_position_id, alpaca_order_id, filled_qty, filled_value,
               submitted_qty, submitted_limit_price
        FROM alpaca_managed_sell_fills
        """
    ).fetchall():
        position_id = int(managed_position_id)
        if not alpaca_order_id_is_canonical(order_id):
            invalid_fill_identity_positions.add(position_id)
        try:
            quantity = float(fill_qty)
            value = float(fill_value)
            submitted = None if submitted_qty is None else float(submitted_qty)
            unit_price = value / quantity if quantity > 0.0 else None
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            continue
        if (
            submitted is not None
            and quantity > submitted
            and not _report_quantity_residual_is_negligible(
                quantity - submitted,
                quantity_values=(quantity, submitted),
                mark_prices=(unit_price, submitted_limit),
                value_values=(value,),
            )
        ):
            invalid_fill_notional_positions.add(position_id)

    positions = pd.read_sql_query(
        """
        SELECT positions.id, positions.workflow, positions.buy_status,
               positions.buy_causality_quarantine, positions.buy_order_qty,
               positions.buy_order_limit_price, positions.filled_qty,
               positions.filled_avg_price, positions.sell_filled_qty,
               positions.sell_filled_avg_price, positions.sold_qty,
               positions.sold_value, positions.remaining_qty, positions.closed_at,
               positions.target_sell_price, positions.sell_order_limit_price,
               positions.last_corporate_action_id, positions.corporate_action_adjusted_at,
               positions.corporate_action_cash_in_lieu_qty,
               CASE
                   WHEN (positions.buy_order_qty IS NULL
                         OR (TYPEOF(positions.buy_order_qty) IN ('integer', 'real')
                             AND positions.buy_order_qty
                                 - positions.buy_order_qty IS NOT NULL))
                    AND (positions.buy_order_limit_price IS NULL
                         OR (TYPEOF(positions.buy_order_limit_price) IN ('integer', 'real')
                             AND positions.buy_order_limit_price
                                 - positions.buy_order_limit_price IS NOT NULL))
                    AND (positions.filled_qty IS NULL
                         OR (TYPEOF(positions.filled_qty) IN ('integer', 'real')
                             AND positions.filled_qty - positions.filled_qty IS NOT NULL))
                    AND (positions.filled_avg_price IS NULL
                         OR (TYPEOF(positions.filled_avg_price) IN ('integer', 'real')
                             AND positions.filled_avg_price
                                 - positions.filled_avg_price IS NOT NULL))
                    AND (positions.target_sell_price IS NULL
                         OR (TYPEOF(positions.target_sell_price) IN ('integer', 'real')
                             AND positions.target_sell_price
                                 - positions.target_sell_price IS NOT NULL
                             AND positions.target_sell_price > 0))
                    AND (positions.sell_order_limit_price IS NULL
                         OR (TYPEOF(positions.sell_order_limit_price) IN ('integer', 'real')
                             AND positions.sell_order_limit_price
                                 - positions.sell_order_limit_price IS NOT NULL
                             AND positions.sell_order_limit_price > 0))
                    AND (positions.sell_filled_qty IS NULL
                         OR (TYPEOF(positions.sell_filled_qty) IN ('integer', 'real')
                             AND positions.sell_filled_qty
                                 - positions.sell_filled_qty IS NOT NULL))
                    AND (positions.sell_filled_avg_price IS NULL
                         OR (TYPEOF(positions.sell_filled_avg_price) IN ('integer', 'real')
                             AND positions.sell_filled_avg_price
                                 - positions.sell_filled_avg_price IS NOT NULL))
                    AND TYPEOF(positions.sold_qty) IN ('integer', 'real')
                    AND positions.sold_qty - positions.sold_qty IS NOT NULL
                    AND TYPEOF(positions.sold_value) IN ('integer', 'real')
                    AND positions.sold_value - positions.sold_value IS NOT NULL
                    AND (positions.remaining_qty IS NULL
                         OR (TYPEOF(positions.remaining_qty) IN ('integer', 'real')
                             AND positions.remaining_qty
                                 - positions.remaining_qty IS NOT NULL))
                    AND (positions.corporate_action_cash_in_lieu_qty IS NULL
                         OR (TYPEOF(positions.corporate_action_cash_in_lieu_qty) IN ('integer', 'real')
                             AND positions.corporate_action_cash_in_lieu_qty
                                 - positions.corporate_action_cash_in_lieu_qty IS NOT NULL
                             AND positions.corporate_action_cash_in_lieu_qty > 0
                             AND positions.corporate_action_cash_in_lieu_qty < 1))
                   THEN 1 ELSE 0
               END AS parent_numeric_types_valid,
               COUNT(fills.managed_position_id) AS ledger_fill_count,
               SUM(CAST(fills.filled_qty AS REAL)) AS ledger_sold_qty,
               SUM(CAST(fills.filled_value AS REAL)) AS ledger_sold_value,
               SUM(
                   CASE
                       WHEN fills.managed_position_id IS NOT NULL
                        AND (
                              fills.filled_qty IS NULL
                              OR fills.filled_value IS NULL
                              OR TYPEOF(fills.filled_qty) NOT IN ('integer', 'real')
                              OR TYPEOF(fills.filled_value) NOT IN ('integer', 'real')
                              OR fills.filled_qty - fills.filled_qty IS NULL
                              OR fills.filled_value - fills.filled_value IS NULL
                              OR NOT (
                                  (fills.filled_qty = 0 AND fills.filled_value = 0)
                                  OR (fills.filled_qty > 0 AND fills.filled_value > 0)
                              )
                              OR (
                                  fills.filled_qty > 0
                                  AND (
                                      CAST(fills.filled_value AS REAL) / fills.filled_qty <= 0
                                      OR (
                                          CAST(fills.filled_value AS REAL) / fills.filled_qty
                                      ) - (
                                          CAST(fills.filled_value AS REAL) / fills.filled_qty
                                      ) IS NULL
                                  )
                              )
                              OR (
                                  fills.submitted_qty IS NOT NULL
                                  AND (
                                      TYPEOF(fills.submitted_qty) NOT IN ('integer', 'real')
                                      OR fills.submitted_qty - fills.submitted_qty IS NULL
                                      OR fills.submitted_qty <= 0
                                      OR fills.filled_qty > fills.submitted_qty + MIN(
                                          ?,
                                          ? + ? * MAX(
                                              ABS(fills.filled_qty),
                                              ABS(fills.submitted_qty)
                                          )
                                      )
                                  )
                              )
                              OR (
                                  fills.submitted_limit_price IS NOT NULL
                                  AND (
                                      TYPEOF(fills.submitted_limit_price) NOT IN ('integer', 'real')
                                      OR fills.submitted_limit_price
                                          - fills.submitted_limit_price IS NULL
                                      OR fills.submitted_limit_price <= 0
                                      OR (
                                          fills.filled_qty > 0
                                          AND CAST(fills.filled_value AS REAL) / fills.filled_qty
                                              < fills.submitted_limit_price
                                                - CASE
                                                    WHEN fills.submitted_limit_price < 1
                                                    THEN 0.00005
                                                    ELSE 0.005
                                                  END
                                      )
                                  )
                              )
                            )
                       THEN 1 ELSE 0
                   END
               ) AS invalid_ledger_fill_count
        FROM alpaca_managed_positions AS positions
        LEFT JOIN alpaca_managed_sell_fills AS fills
          ON fills.managed_position_id = positions.id
        WHERE positions.closed_at IS NOT NULL
          AND (
                positions.filled_qty IS NOT NULL
                OR LOWER(COALESCE(positions.buy_status, '')) IN (
                    'filled',
                    'partially_filled',
                    'fill_quantity_regression',
                    'incomplete_fill_metadata'
                )
                OR positions.filled_avg_price IS NOT NULL
                OR positions.filled_at IS NOT NULL
                OR positions.target_sell_price IS NOT NULL
                OR positions.sell_client_order_id IS NOT NULL
                OR positions.sell_alpaca_order_id IS NOT NULL
                OR positions.sell_status IS NOT NULL
                OR positions.sell_filled_qty IS NOT NULL
                OR positions.sell_filled_avg_price IS NOT NULL
                OR positions.sell_filled_at IS NOT NULL
                OR positions.sold_qty != 0
                OR positions.sold_value != 0
                OR positions.remaining_qty IS NOT NULL
                OR fills.managed_position_id IS NOT NULL
              )
        GROUP BY positions.id
        """,
        conn,
        params=(
            MANAGED_QUANTITY_MAX_TOLERANCE,
            MANAGED_QUANTITY_ABSOLUTE_TOLERANCE,
            MANAGED_QUANTITY_RELATIVE_TOLERANCE,
        ),
    )
    if positions.empty:
        row = _realized_pnl_summary_row(0, 0, 0.0, 0.0)
        if include_workflow:
            row = {"Workflow": "Total", **row}
        return pd.DataFrame(
            [row],
            columns=REALIZED_PNL_WORKFLOW_COLUMNS if include_workflow else REALIZED_PNL_COLUMNS,
        )

    numeric_columns = [
        "buy_order_qty",
        "buy_order_limit_price",
        "filled_qty",
        "filled_avg_price",
        "sell_filled_qty",
        "sell_filled_avg_price",
        "target_sell_price",
        "sell_order_limit_price",
        "sold_qty",
        "sold_value",
        "remaining_qty",
        "corporate_action_cash_in_lieu_qty",
        "parent_numeric_types_valid",
        "ledger_fill_count",
        "ledger_sold_qty",
        "ledger_sold_value",
        "invalid_ledger_fill_count",
    ]
    for column in numeric_columns:
        positions[column] = pd.to_numeric(positions[column], errors="coerce")
    positions["sell_fill_identities_valid"] = ~positions["id"].isin(
        invalid_fill_identity_positions | invalid_fill_notional_positions
    )

    aggregate_sell_pair = positions["sold_qty"].gt(0) & positions["sold_value"].gt(0)
    legacy_sell_pair = positions["sold_qty"].eq(0) & positions["sold_value"].eq(0)
    has_sell_fill_ledger = positions["ledger_fill_count"].gt(0)
    positions["effective_sold_qty"] = positions["sold_qty"].where(aggregate_sell_pair)
    positions["effective_sold_value"] = positions["sold_value"].where(aggregate_sell_pair)
    legacy_scalar_fallback = legacy_sell_pair & ~has_sell_fill_ledger
    positions.loc[legacy_scalar_fallback, "effective_sold_qty"] = positions.loc[
        legacy_scalar_fallback,
        "sell_filled_qty",
    ]
    with np.errstate(over="ignore", invalid="ignore"):
        positions.loc[legacy_scalar_fallback, "effective_sold_value"] = (
            positions.loc[legacy_scalar_fallback, "sell_filled_qty"]
            * positions.loc[legacy_scalar_fallback, "sell_filled_avg_price"]
        )
        ledger_quantity_difference = (positions["ledger_sold_qty"] - positions["sold_qty"]).abs()
        ledger_quantity_scale = positions[["ledger_sold_qty", "sold_qty"]].abs().max(axis=1)
        ledger_quantity_tolerance = ledger_quantity_scale.map(managed_quantity_tolerance)
        ledger_value_difference = (positions["ledger_sold_value"] - positions["sold_value"]).abs()
        ledger_value_scale = positions[["ledger_sold_value", "sold_value"]].abs().max(axis=1)
        ledger_value_tolerance = ledger_value_scale.map(
            managed_value_reconciliation_tolerance,
        )
        adjusted_buy_cost = positions["filled_qty"] * positions["filled_avg_price"]
        buy_intent_value = positions["buy_order_qty"] * positions["buy_order_limit_price"]
        adjusted_buy_cost_tolerance = pd.concat(
            [adjusted_buy_cost.abs(), buy_intent_value.abs()],
            axis=1,
        ).max(axis=1).map(managed_value_reconciliation_tolerance)
        buy_quantity_scale = positions[["buy_order_qty", "filled_qty"]].abs().max(axis=1)
        buy_quantity_tolerance = buy_quantity_scale.map(managed_quantity_tolerance)
        buy_price_tolerance = np.where(
            positions["buy_order_limit_price"].lt(1.0),
            0.00005,
            0.005,
        )
    ledger_quantity_reconciles_notional = positions.apply(
        lambda row: _report_quantity_residual_is_negligible(
            row["ledger_sold_qty"] - row["sold_qty"],
            quantity_values=(row["ledger_sold_qty"], row["sold_qty"], row["filled_qty"]),
            mark_prices=(
                row["filled_avg_price"],
                row["target_sell_price"],
                row["sell_order_limit_price"],
                (row["sold_value"] / row["sold_qty"] if row["sold_qty"] > 0 else None),
                (row["ledger_sold_value"] / row["ledger_sold_qty"] if row["ledger_sold_qty"] > 0 else None),
            ),
            value_values=(row["sold_value"], row["ledger_sold_value"]),
        ),
        axis=1,
    )
    buy_quantity_reconciles_notional = positions.apply(
        lambda row: bool(
            row["filled_qty"] <= row["buy_order_qty"]
            or _report_quantity_residual_is_negligible(
                row["filled_qty"] - row["buy_order_qty"],
                quantity_values=(row["filled_qty"], row["buy_order_qty"]),
                mark_prices=(row["filled_avg_price"], row["buy_order_limit_price"]),
                value_values=(
                    row["filled_qty"] * row["filled_avg_price"],
                    row["buy_order_qty"] * row["buy_order_limit_price"],
                ),
            )
        ),
        axis=1,
    )
    buy_quantity_matches_intent = positions.apply(
        lambda row: bool(
            pd.notna(row["filled_qty"])
            and pd.notna(row["buy_order_qty"])
            and _report_quantity_residual_is_negligible(
                row["filled_qty"] - row["buy_order_qty"],
                quantity_values=(row["filled_qty"], row["buy_order_qty"]),
                mark_prices=(row["filled_avg_price"], row["buy_order_limit_price"]),
                value_values=(
                    row["filled_qty"] * row["filled_avg_price"],
                    row["buy_order_qty"] * row["buy_order_limit_price"],
                ),
            )
        ),
        axis=1,
    )
    buy_reports_terminal_fill = positions["buy_status"].fillna("").astype(str).str.strip().str.lower().eq("filled")
    buy_has_causality_quarantine = positions["buy_causality_quarantine"].fillna("").astype(str).str.strip().ne("")
    ledger_values_are_finite = np.isfinite(positions[["ledger_sold_qty", "ledger_sold_value"]]).all(axis=1)
    positions["sell_fill_ledger_reconciles"] = ~has_sell_fill_ledger | (
        aggregate_sell_pair
        & positions["invalid_ledger_fill_count"].eq(0)
        & positions["sell_fill_identities_valid"]
        & ledger_values_are_finite
        & ledger_quantity_difference.le(ledger_quantity_tolerance)
        & ledger_quantity_reconciles_notional
        & ledger_value_difference.le(ledger_value_tolerance)
    )
    legacy_buy_intent = positions["buy_order_qty"].isna() & positions["buy_order_limit_price"].isna()
    complete_buy_intent = (
        positions["buy_order_qty"].notna()
        & positions["buy_order_limit_price"].notna()
        & np.isfinite(positions[["buy_order_qty", "buy_order_limit_price"]]).all(axis=1)
        & positions["buy_order_qty"].gt(0)
        & positions["buy_order_limit_price"].gt(0)
        & positions["filled_qty"].le(positions["buy_order_qty"] + buy_quantity_tolerance)
        & buy_quantity_reconciles_notional
        & (~buy_reports_terminal_fill | buy_quantity_matches_intent)
        & ~buy_has_causality_quarantine
        & positions["filled_avg_price"].le(positions["buy_order_limit_price"] + buy_price_tolerance)
    )
    corporate_action_identity_is_valid = (
        positions["last_corporate_action_id"].map(alpaca_order_id_is_canonical)
        & positions["corporate_action_adjusted_at"].map(_alpaca_broker_timestamp_is_canonical)
    )
    corporate_action_buy_intent = (
        corporate_action_identity_is_valid
        & positions["buy_order_qty"].notna()
        & positions["buy_order_limit_price"].notna()
        & np.isfinite(
            positions[
                [
                    "buy_order_qty",
                    "buy_order_limit_price",
                    "filled_qty",
                    "filled_avg_price",
                ]
            ]
        ).all(axis=1)
        & positions["buy_order_qty"].gt(0)
        & positions["buy_order_limit_price"].gt(0)
        & positions["filled_qty"].gt(0)
        & positions["filled_avg_price"].gt(0)
        & adjusted_buy_cost.le(buy_intent_value + adjusted_buy_cost_tolerance)
    )
    positions["buy_intent_reconciles"] = (
        legacy_buy_intent | complete_buy_intent | corporate_action_buy_intent
    ) & ~buy_has_causality_quarantine
    with np.errstate(over="ignore", invalid="ignore"):
        positions["effective_remaining_qty"] = positions["remaining_qty"].where(
            positions["remaining_qty"].notna(),
            positions["filled_qty"] - positions["effective_sold_qty"],
        )
    complete = positions.dropna(
        subset=[
            "filled_qty",
            "filled_avg_price",
            "effective_sold_qty",
            "effective_sold_value",
            "effective_remaining_qty",
        ],
    ).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        complete["effective_buy_cost"] = complete["filled_qty"] * complete["filled_avg_price"]
        complete_quantity_difference = (complete["effective_sold_qty"] - complete["filled_qty"]).abs()
        complete_quantity_scale = complete[["filled_qty", "effective_sold_qty"]].abs().max(axis=1)
        complete_quantity_tolerance = complete_quantity_scale.map(
            managed_quantity_tolerance,
        )
    complete_quantity_reconciles_notional = (
        complete.apply(
            lambda row: _report_quantity_residual_is_negligible(
                row["effective_sold_qty"] - row["filled_qty"],
                quantity_values=(row["effective_sold_qty"], row["filled_qty"]),
                mark_prices=(
                    row["filled_avg_price"],
                    row["target_sell_price"],
                    row["sell_order_limit_price"],
                    row["effective_sold_value"] / row["effective_sold_qty"],
                ),
                value_values=(row["effective_buy_cost"], row["effective_sold_value"]),
            ),
            axis=1,
        )
        if not complete.empty
        else pd.Series(dtype=bool, index=complete.index)
    )
    complete_remaining_is_negligible = (
        complete.apply(
            lambda row: _report_quantity_residual_is_negligible(
                row["effective_remaining_qty"],
                quantity_values=(
                    row["effective_remaining_qty"],
                    row["effective_sold_qty"],
                    row["filled_qty"],
                ),
                mark_prices=(
                    row["filled_avg_price"],
                    row["target_sell_price"],
                    row["sell_order_limit_price"],
                    row["effective_sold_value"] / row["effective_sold_qty"],
                ),
                value_values=(row["effective_buy_cost"], row["effective_sold_value"]),
            ),
            axis=1,
        )
        if not complete.empty
        else pd.Series(dtype=bool, index=complete.index)
    )
    complete_values_are_finite = np.isfinite(
        complete[
            [
                "filled_qty",
                "filled_avg_price",
                "effective_sold_qty",
                "effective_sold_value",
                "effective_remaining_qty",
                "effective_buy_cost",
            ]
        ]
    ).all(axis=1)
    complete_optional_prices_are_valid = (
        complete["target_sell_price"].isna()
        | (np.isfinite(complete["target_sell_price"]) & complete["target_sell_price"].gt(0))
    ) & (
        complete["sell_order_limit_price"].isna()
        | (np.isfinite(complete["sell_order_limit_price"]) & complete["sell_order_limit_price"].gt(0))
    )
    complete = complete[
        complete_values_are_finite
        & complete_optional_prices_are_valid
        & complete["filled_qty"].gt(0)
        & complete["filled_avg_price"].gt(0)
        & complete["effective_sold_qty"].gt(0)
        & complete["effective_sold_value"].gt(0)
        & complete["effective_buy_cost"].gt(0)
        & complete["parent_numeric_types_valid"].eq(1)
        & complete["buy_intent_reconciles"]
        & complete["sell_fill_ledger_reconciles"]
        & complete_quantity_difference.le(complete_quantity_tolerance)
        & complete_quantity_reconciles_notional
        & complete["effective_remaining_qty"].abs().le(complete_quantity_tolerance)
        & complete_remaining_is_negligible
        & complete["corporate_action_cash_in_lieu_qty"].isna()
    ]

    if include_workflow:
        positions["Workflow"] = positions["workflow"].fillna("Unknown").replace("", "Unknown")
        complete["Workflow"] = complete["workflow"].fillna("Unknown").replace("", "Unknown")
        rows = []
        for workflow, workflow_positions in positions.groupby("Workflow", sort=True, dropna=False):
            workflow_complete = complete[complete["Workflow"].eq(workflow)]
            rows.append(
                {
                    "Workflow": workflow,
                    **_finite_realized_pnl_summary_row(
                        closed_positions=len(workflow_positions),
                        complete=workflow_complete,
                    ),
                }
            )
        rows.append(
            {
                "Workflow": "Total",
                **_finite_realized_pnl_summary_row(
                    closed_positions=len(positions),
                    complete=complete,
                ),
            }
        )
        return pd.DataFrame(rows, columns=REALIZED_PNL_WORKFLOW_COLUMNS)

    return pd.DataFrame(
        [
            _finite_realized_pnl_summary_row(
                closed_positions=len(positions),
                complete=complete,
            )
        ],
        columns=REALIZED_PNL_COLUMNS,
    )


def _finite_realized_pnl_summary_row(
    *,
    closed_positions: int,
    complete: pd.DataFrame,
) -> dict[str, object]:
    """Summarize only when every derived total is representable as a finite float."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        total_buy_cost = float(complete["effective_buy_cost"].sum())
        total_sell_value = float(complete["effective_sold_value"].sum())
        realized_pl = np.float64(total_sell_value) - np.float64(total_buy_cost)
        realized_pl_pct = (
            np.divide(realized_pl, np.float64(total_buy_cost)) * np.float64(100.0)
            if total_buy_cost > 0
            else np.float64(0.0)
        )
    if not np.isfinite([total_buy_cost, total_sell_value, realized_pl, realized_pl_pct]).all():
        # Treat unrepresentable accounting as incomplete instead of publishing
        # infinities or NaNs as plausible monetary totals.
        return _realized_pnl_summary_row(closed_positions, 0, 0.0, 0.0)
    return _realized_pnl_summary_row(
        closed_positions,
        len(complete),
        total_buy_cost,
        total_sell_value,
    )


def _realized_pnl_summary_row(
    closed_positions: int,
    complete_closed_positions: int,
    total_buy_cost: float,
    total_sell_value: float,
) -> dict[str, object]:
    realized_pl = total_sell_value - total_buy_cost
    realized_pl_pct = (realized_pl / total_buy_cost * 100.0) if total_buy_cost > 0 else 0.0
    return {
        "Closed Positions": closed_positions,
        "Complete Closed Positions": complete_closed_positions,
        "Incomplete Closed Positions": closed_positions - complete_closed_positions,
        "Total Buy Cost": total_buy_cost,
        "Total Sell Value": total_sell_value,
        "Realized P/L": realized_pl,
        "Realized P/L %": realized_pl_pct,
    }


def build_pending_action_report(
    conn: sqlite3.Connection,
    optimization_summary: pd.DataFrame,
    rsi_period: int,
    pending_action_filter: str,
    require_multiple_trades: bool,
    min_sharpe: float | None,
    rsi_entry_rule: str = "lower",
    base_cfg: BacktestConfig | None = None,
    expected_buy_rsi_values: list[float] | None = None,
    expected_profit_target_values: list[float] | None = None,
    expected_strategy_fingerprint: str | None = None,
    allow_unbound_backtest_config: bool = False,
    _strategy_report_cache: _StrategyReportCache | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    if type(rsi_period) is not int or not 2 <= rsi_period <= 10_000:
        raise ValueError("rsi_period must be an integer between 2 and 10000.")
    if base_cfg is not None:
        validate_backtest_configuration(base_cfg)
        if rsi_period != base_cfg.rsi_period:
            raise ValueError("rsi_period must match base_cfg.rsi_period for an authenticated actionable report.")
    rsi_entry_rule_code(rsi_entry_rule)
    _validate_report_strategy_provenance(
        base_cfg=base_cfg,
        expected_buy_rsi_values=expected_buy_rsi_values,
        expected_profit_target_values=expected_profit_target_values,
        rsi_entry_rule=rsi_entry_rule,
        expected_strategy_fingerprint=expected_strategy_fingerprint,
        allow_unbound_backtest_config=allow_unbound_backtest_config,
    )
    with _consistent_read_snapshot(conn):
        return _build_pending_action_report_snapshot(
            conn,
            optimization_summary,
            rsi_period,
            pending_action_filter,
            require_multiple_trades,
            min_sharpe,
            rsi_entry_rule,
            base_cfg,
            expected_buy_rsi_values,
            expected_profit_target_values,
            expected_strategy_fingerprint,
            _strategy_report_cache,
            deadline_check,
        )


def _build_pending_action_report_snapshot(
    conn: sqlite3.Connection,
    optimization_summary: pd.DataFrame,
    rsi_period: int,
    pending_action_filter: str,
    require_multiple_trades: bool,
    min_sharpe: float | None,
    rsi_entry_rule: str,
    base_cfg: BacktestConfig | None,
    expected_buy_rsi_values: list[float] | None,
    expected_profit_target_values: list[float] | None,
    expected_strategy_fingerprint: str | None,
    strategy_report_cache: _StrategyReportCache | None,
    deadline_check: Callable[[], None] | None,
) -> pd.DataFrame:
    if pending_action_filter not in {"buy", "sell"}:
        raise ValueError(f"Unsupported pending action report: {pending_action_filter}")

    columns = [
        "Asset",
        "RSI Symbol",
        "Date",
        "RSI Observation Date",
        "Start Date",
        "Trading Days",
        "Latest RSI",
        "Buy RSI",
        "Sell Return Multiple",
        "Trades Executed",
        "Sharpe",
        "In Position",
        "Pending Action",
    ]
    if optimization_summary.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    processed_strategies: set[tuple[str, str, float, float]] = set()
    for _, summary_row in optimization_summary.iterrows():
        if deadline_check is not None:
            deadline_check()
        asset_symbol = str(summary_row["Asset"])
        signal_symbol = str(summary_row["RSI Symbol"])
        cached_report = (
            None
            if strategy_report_cache is None
            else strategy_report_cache.get((asset_symbol, signal_symbol, rsi_entry_rule))
        )
        if cached_report is None:
            persisted_summary = load_best_strategy_summary(
                conn,
                asset_symbol,
                signal_symbol,
                rsi_entry_rule,
            )
            if persisted_summary is None:
                continue
        else:
            persisted_summary = cached_report[0]
        try:
            requested_buy_rsi = float(summary_row["Buy RSI"])
            requested_profit_target = float(summary_row["Sell Return Multiple"])
            buy_rsi = float(persisted_summary["buy_rsi"])
            profit_target_multiple = float(persisted_summary["profit_target_multiple"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        # The DataFrame selects an asset and an exact strategy identity only.
        # Eligibility and displayed metrics must come from the authenticated
        # persisted best row, never from caller-controlled report input.
        if (
            not np.isfinite(requested_buy_rsi)
            or not np.isfinite(requested_profit_target)
            or requested_buy_rsi != buy_rsi
            or requested_profit_target != profit_target_multiple
        ):
            continue
        strategy_identity = (
            asset_symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
        )
        if strategy_identity in processed_strategies:
            continue
        processed_strategies.add(strategy_identity)

        # The summary and compact state each carry an authenticated digest, but
        # those independent digests do not prove that they describe the same
        # run. Authenticate their shared chronology, fills, endpoint, and the
        # retained curve before any summary-derived eligibility gate is used.
        complete_curve = (
            cached_report[1]
            if cached_report is not None
            else load_complete_strategy_equity_curve(
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
                allow_unbound_backtest_config=base_cfg is None,
            )
        )
        if complete_curve is None:
            continue
        trades_executed = int(persisted_summary["trades_executed"])
        if require_multiple_trades and trades_executed <= 1:
            continue
        try:
            sharpe = float(persisted_summary["sharpe"])
        except (TypeError, ValueError):
            sharpe = float("nan")
        if min_sharpe is not None and (pd.isna(sharpe) or sharpe < min_sharpe):
            continue

        state = (
            cached_report[2]
            if cached_report is not None
            else load_strategy_state(
                conn,
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                rsi_period=rsi_period,
                rsi_entry_rule=rsi_entry_rule,
            )
        )
        if state is None or state["last_date"] is None:
            continue

        pending_action_matches = state["pending_action"] == pending_action_filter
        latest_sell_executed = False
        if pending_action_filter == "sell" and not pending_action_matches:
            latest_action = (
                cached_report[3]
                if cached_report is not None
                else load_complete_strategy_latest_action(
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
                    allow_unbound_backtest_config=base_cfg is None,
                )
            )
            latest_sell_executed = latest_action == "sell"

        if not pending_action_matches and not latest_sell_executed:
            continue

        latest_rsi = load_aligned_rsi_for_asset_session(
            conn,
            asset_symbol,
            signal_symbol,
            rsi_period,
            state["last_date"],
        )
        event_only = latest_sell_executed and not pending_action_matches
        if not event_only and latest_rsi is None:
            continue
        if (
            not event_only
            and latest_rsi is not None
            and not signal_observation_is_fresh(state["last_date"], latest_rsi[0])
        ):
            continue

        in_position = bool(state["in_position"])
        pending_action = state["pending_action"]
        latest_rsi_date = pd.NA if latest_rsi is None else latest_rsi[0]
        latest_rsi_value = pd.NA if latest_rsi is None else float(latest_rsi[1])
        start_date = persisted_summary["start_date"]
        trading_days = int(persisted_summary["trading_days"])

        should_include = (pending_action_filter == "buy" and pending_action_matches and not in_position) or (
            pending_action_filter == "sell" and ((pending_action_matches and in_position) or latest_sell_executed)
        )
        if should_include:
            rows.append(
                {
                    "Asset": asset_symbol,
                    "RSI Symbol": signal_symbol,
                    # Alpaca validates this against the immediately preceding
                    # ETF trading session. The source observation can be a
                    # weekend date and is exposed separately for auditability.
                    "Date": state["last_date"],
                    "RSI Observation Date": latest_rsi_date,
                    "Start Date": start_date,
                    "Trading Days": trading_days,
                    "Latest RSI": latest_rsi_value,
                    "Buy RSI": buy_rsi,
                    "Sell Return Multiple": profit_target_multiple,
                    "Trades Executed": trades_executed,
                    "Sharpe": sharpe,
                    "In Position": in_position,
                    # A target exit executes intraday and leaves no resting sell
                    # action in simulator state. Classify that latest execution
                    # as a sell here so the sell report represents real events.
                    "Pending Action": pending_action_filter if latest_sell_executed else pending_action,
                }
            )

    return pd.DataFrame(rows, columns=columns)
