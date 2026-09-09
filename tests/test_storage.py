from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
import tempfile
import threading
import unittest
from collections.abc import Callable
from contextlib import closing
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pandas as pd

import leveraged_trader.reports as reports_module
import leveraged_trader.storage as storage_module
from leveraged_trader.backtest import performance_summary
from leveraged_trader.config import RISK_FREE_SYMBOL, BacktestConfig
from leveraged_trader.indicators import compute_rsi
from leveraged_trader.optimized_backtest import run_grid_summary, run_single_equity_curve
from leveraged_trader.storage import (
    AssetMarketDataError,
    SellFillQuantityRegressionError,
    SummaryRollup,
    _market_arrays,
    _merge_centered_moments,
    _rollup_metrics,
    _strategy_state_integrity_digest,
    _synchronize_market_data_history,
    _update_summary_rollup,
    adopt_alpaca_managed_buy_order_if_submission_not_found,
    adopt_alpaca_managed_position_asset_if_current,
    align_signal_values_to_asset_sessions,
    alpaca_managed_buy_fill_observation_authorizes_mutation,
    apply_alpaca_closed_position_broker_correction,
    attach_alpaca_managed_sell_order_if_current,
    claim_alpaca_managed_sell_replacement,
    close_alpaca_managed_buy_if_current_and_unfilled,
    close_alpaca_managed_position_if_current_and_complete,
    ensure_rsi_values,
    init_state_db,
    load_alpaca_managed_positions,
    load_recently_closed_alpaca_managed_positions,
    mark_alpaca_closed_correction_audited,
    mark_alpaca_managed_buy_filled,
    mark_alpaca_managed_sell_filled_if_current,
    process_asset_grid,
    record_alpaca_managed_sell_generation,
    record_alpaca_managed_sell_order,
    save_alpaca_managed_buy_order,
    save_equity_records,
    save_market_data,
    signal_observation_is_fresh,
    strategy_config_fingerprint,
    strategy_state_generation,
    strategy_state_matches_config,
    update_alpaca_managed_buy_status_if_current,
    update_alpaca_managed_sell_status_if_current,
)


def sample_strategy_data(periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    asset_close = np.linspace(100.0, 140.0, periods)
    signal_close = pd.Series(
        [100.0 - i for i in range(periods // 2)] + [80.0 + i for i in range(periods - periods // 2)],
        index=dates,
    )
    return pd.DataFrame(
        {
            "TQQQ_Open": asset_close,
            "TQQQ_High": asset_close + 1.0,
            "TQQQ_Low": asset_close - 1.0,
            "TQQQ_Close": asset_close,
            "TQQQ_Volume": 1_000_000,
            "QQQ_Open": signal_close,
            "QQQ_High": signal_close + 1.0,
            "QQQ_Low": signal_close - 1.0,
            "QQQ_Close": signal_close,
            "QQQ_Volume": 2_000_000,
            f"{RISK_FREE_SYMBOL}_Open": 5.0,
            f"{RISK_FREE_SYMBOL}_High": 5.0,
            f"{RISK_FREE_SYMBOL}_Low": 5.0,
            f"{RISK_FREE_SYMBOL}_Close": 5.0,
            f"{RISK_FREE_SYMBOL}_Volume": 0,
        },
        index=dates,
    )


def one_ulp_best_strategy_data() -> pd.DataFrame:
    """Return a tiny grid whose equivalent targets historically swapped by one ULP."""
    periods = 7
    rng = np.random.default_rng(4)
    dates = pd.date_range("2025-01-02", periods=periods, freq="B")
    asset_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, periods)))
    asset_open = asset_close * (1.0 + rng.normal(0.0, 0.002, periods))
    asset_high = np.maximum(asset_close, asset_open) * 1.005
    asset_low = np.minimum(asset_close, asset_open) * 0.995
    signal_close = np.linspace(200.0, 100.0, periods)
    risk_free_yield = rng.uniform(0.0, 8.0, periods)
    return pd.DataFrame(
        {
            "TQQQ_Open": asset_open,
            "TQQQ_High": asset_high,
            "TQQQ_Low": asset_low,
            "TQQQ_Close": asset_close,
            "TQQQ_Volume": 1,
            "QQQ_Open": signal_close,
            "QQQ_High": signal_close,
            "QQQ_Low": signal_close,
            "QQQ_Close": signal_close,
            "QQQ_Volume": 1,
            f"{RISK_FREE_SYMBOL}_Open": risk_free_yield,
            f"{RISK_FREE_SYMBOL}_High": risk_free_yield,
            f"{RISK_FREE_SYMBOL}_Low": risk_free_yield,
            f"{RISK_FREE_SYMBOL}_Close": risk_free_yield,
            f"{RISK_FREE_SYMBOL}_Volume": 0,
        },
        index=dates,
    )


def summarize_saved_results(
    conn: sqlite3.Connection,
    workflow_assets: pd.DataFrame,
    **kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exercise structural legacy fixtures only through an explicit opt-in."""
    kwargs.setdefault(
        "allow_unbound_backtest_config",
        kwargs.get("base_cfg") is None,
    )
    return reports_module.summarize_saved_results(
        conn,
        workflow_assets,
        **kwargs,
    )


def build_sell_signal_report(
    conn: sqlite3.Connection,
    optimization_summary: pd.DataFrame,
    rsi_period: int,
    **kwargs: object,
) -> pd.DataFrame:
    """Exercise structural legacy fixtures only through an explicit opt-in."""
    kwargs.setdefault(
        "allow_unbound_backtest_config",
        kwargs.get("base_cfg") is None,
    )
    return reports_module.build_sell_signal_report(
        conn,
        optimization_summary,
        rsi_period,
        **kwargs,
    )


class RollbackToFailureConnection(sqlite3.Connection):
    """Inject one savepoint rollback failure and observe unsafe releases."""

    failed_savepoint: str | None = None
    rollback_to_failed = False
    release_after_failed_rollback = False

    def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
        normalized_sql = " ".join(sql.split())
        if self.failed_savepoint is not None and normalized_sql == (f"ROLLBACK TO SAVEPOINT {self.failed_savepoint}"):
            self.rollback_to_failed = True
            raise sqlite3.OperationalError("forced rollback-to failure")
        if self.rollback_to_failed and normalized_sql == (f"RELEASE SAVEPOINT {self.failed_savepoint}"):
            self.release_after_failed_rollback = True
        return super().execute(sql, parameters)


class RollbackAndConnectionRollbackFailureConnection(RollbackToFailureConnection):
    """Inject failures in both savepoint and connection-level rollback."""

    connection_rollback_failed = False

    def rollback(self) -> None:
        if self.rollback_to_failed:
            self.connection_rollback_failed = True
            raise sqlite3.OperationalError("forced connection rollback failure")
        super().rollback()


class CommitFailureConnection(sqlite3.Connection):
    """Inject a final commit failure and optionally its cleanup rollback failure."""

    fail_next_commit = False
    fail_rollback_after_commit = False
    commit_failure_observed = False
    rollback_failure_observed = False

    def commit(self) -> None:
        if self.fail_next_commit:
            self.fail_next_commit = False
            self.commit_failure_observed = True
            raise sqlite3.OperationalError("forced commit failure")
        super().commit()

    def rollback(self) -> None:
        if self.commit_failure_observed and self.fail_rollback_after_commit:
            self.fail_rollback_after_commit = False
            self.rollback_failure_observed = True
            raise sqlite3.OperationalError("forced rollback after commit failure")
        super().rollback()


class ExecuteFailureConnection(sqlite3.Connection):
    """Inject an execute failure after acquiring an operation-owned write lock."""

    fail_sql_fragment: str | None = None
    fail_rollback = False
    rollback_failure_observed = False

    def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
        normalized_sql = " ".join(sql.split())
        if self.fail_sql_fragment is not None and self.fail_sql_fragment in normalized_sql:
            self.fail_sql_fragment = None
            if not self.in_transaction:
                super().execute("BEGIN IMMEDIATE")
            raise sqlite3.OperationalError("forced execute failure")
        return super().execute(sql, parameters)

    def rollback(self) -> None:
        if self.fail_rollback:
            self.fail_rollback = False
            self.rollback_failure_observed = True
            raise sqlite3.OperationalError("forced rollback after execute failure")
        super().rollback()


class SellGenerationFailureConnection(sqlite3.Connection):
    """Inject one failure between a managed parent update and ledger write."""

    sell_generation_failure: BaseException | None = None

    def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
        if self.sell_generation_failure is not None and "INSERT INTO alpaca_managed_sell_fills" in " ".join(
            sql.split()
        ):
            failure = self.sell_generation_failure
            self.sell_generation_failure = None
            raise failure
        return super().execute(sql, parameters)


class ManagedLateFailureConnection(sqlite3.Connection):
    """Inject a BaseException after an earlier managed-accounting write."""

    fail_execute_fragment: str | None = None
    fail_executemany_fragment: str | None = None

    def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
        normalized_sql = " ".join(sql.split())
        if self.fail_execute_fragment is not None and self.fail_execute_fragment in normalized_sql:
            self.fail_execute_fragment = None
            raise KeyboardInterrupt("forced late managed-accounting execute failure")
        return super().execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: object) -> sqlite3.Cursor:
        cursor = super().executemany(sql, seq_of_parameters)  # type: ignore[arg-type]
        normalized_sql = " ".join(sql.split())
        if self.fail_executemany_fragment is not None and self.fail_executemany_fragment in normalized_sql:
            self.fail_executemany_fragment = None
            raise KeyboardInterrupt("forced late managed-accounting executemany failure")
        return cursor


class _ReturningFetchFailureCursor:
    """Raise after SQLite has applied a mutation and produced RETURNING data."""

    def __init__(self, cursor: sqlite3.Cursor, connection: ReturningFetchFailureConnection) -> None:
        self._cursor = cursor
        self._connection = connection

    def fetchone(self) -> object:
        row = self._cursor.fetchone()
        if row is not None and self._connection.fail_returning_fetch:
            self._connection.fail_returning_fetch = False
            raise ValueError("forced RETURNING row decode failure")
        return row

    def __getattr__(self, name: str) -> object:
        return getattr(self._cursor, name)


class ReturningFetchFailureConnection(sqlite3.Connection):
    """Inject a fetch/decode failure after a RETURNING mutation ran."""

    fail_returning_fetch = False

    def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
        cursor = super().execute(sql, parameters)
        if " RETURNING " in f" {' '.join(sql.split())} ":
            return _ReturningFetchFailureCursor(cursor, self)  # type: ignore[return-value]
        return cursor


class _SelectedFetchFailureCursor:
    """Raise while decoding one explicitly selected non-RETURNING lookup."""

    def __init__(self, cursor: sqlite3.Cursor, connection: SelectedFetchFailureConnection) -> None:
        self._cursor = cursor
        self._connection = connection

    def fetchone(self) -> object:
        row = self._cursor.fetchone()
        if self._connection.fail_selected_fetch:
            self._connection.fail_selected_fetch = False
            raise ValueError("forced selected row decode failure")
        return row

    def __getattr__(self, name: str) -> object:
        return getattr(self._cursor, name)


class SelectedFetchFailureConnection(sqlite3.Connection):
    """Inject one fetch failure for a caller-selected SQL fragment."""

    fail_selected_fetch = False
    selected_sql_fragment: str | None = None

    def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
        cursor = super().execute(sql, parameters)
        normalized_sql = " ".join(sql.split())
        if (
            self.fail_selected_fetch
            and self.selected_sql_fragment is not None
            and self.selected_sql_fragment in normalized_sql
        ):
            return _SelectedFetchFailureCursor(cursor, self)  # type: ignore[return-value]
        return cursor


class ClosedCorrectionFinalCasMissConnection(sqlite3.Connection):
    """Turn the final closed-correction CAS into a deterministic no-op."""

    miss_next_closed_correction_final_cas = False

    def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
        normalized_sql = " ".join(sql.split())
        if (
            self.miss_next_closed_correction_final_cas
            and normalized_sql.startswith("UPDATE alpaca_managed_positions SET state_revision = state_revision + 1")
            and "alpaca_asset_id = COALESCE(alpaca_asset_id, ?)" in normalized_sql
        ):
            self.miss_next_closed_correction_final_cas = False
            return super().execute("UPDATE alpaca_managed_positions SET notes = notes WHERE 0")
        return super().execute(sql, parameters)


class _FetchedRowCursor:
    """Return a row fetched before a deterministic cross-connection interleave."""

    def __init__(self, row: object) -> None:
        self._row = row

    def fetchone(self) -> object:
        return self._row


class BuyIntentBackfillRaceConnection(sqlite3.Connection):
    """Backfill a legacy buy intent after fill accounting observed it as NULL."""

    peer: sqlite3.Connection | None = None
    backfill_buy_intent_on_fetch = False

    def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
        normalized_sql = " ".join(sql.split())
        if (
            self.backfill_buy_intent_on_fetch
            and normalized_sql
            == "SELECT buy_order_qty, buy_order_limit_price FROM alpaca_managed_positions WHERE id = ?"
        ):
            row = super().execute(sql, parameters).fetchone()
            self.backfill_buy_intent_on_fetch = False
            assert self.peer is not None
            self.peer.execute(
                "UPDATE alpaca_managed_positions SET buy_order_qty = 1, buy_order_limit_price = 100 WHERE id = ?",
                parameters,
            )
            self.peer.commit()
            return _FetchedRowCursor(row)  # type: ignore[return-value]
        return super().execute(sql, parameters)


class StorageOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        init_state_db(self.conn)
        self.cfg = BacktestConfig(rsi_period=3)

    def tearDown(self) -> None:
        self.conn.close()

    def process_grid(self, data: pd.DataFrame, *, rebuild: bool) -> None:
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0, 70.0],
            profit_target_values=[1.05, 1.50],
            rebuild=rebuild,
        )

    def persisted_strategy_fingerprint(self) -> str:
        row = self.conn.execute(
            "SELECT fingerprint FROM strategy_config WHERE asset_symbol = 'TQQQ' AND signal_symbol = 'QQQ'"
        ).fetchone()
        self.assertIsNotNone(row)
        return str(row[0])

    def test_direct_strategy_boundaries_reject_boolean_numeric_inputs_before_writes(self) -> None:
        data = sample_strategy_data(periods=4)
        invalid_cfg = BacktestConfig(rsi_period=3, fee_bps=True)
        for operation in (
            lambda: strategy_config_fingerprint(invalid_cfg, [30.0], [1.5]),
            lambda: strategy_config_fingerprint(
                BacktestConfig(rsi_period=3, slippage_bps=True),
                [30.0],
                [1.5],
            ),
            lambda: strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                invalid_cfg,
                [30.0],
                [1.5],
            ),
            lambda: process_asset_grid(
                self.conn,
                data,
                invalid_cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                strategy_fingerprint="caller-supplied",
            ),
            lambda: strategy_config_fingerprint(self.cfg, [True], [1.5]),
            lambda: strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [True],
                [1.5],
            ),
            lambda: process_asset_grid(
                self.conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                [True],
                [1.5],
                rebuild=True,
                strategy_fingerprint="caller-supplied",
            ),
        ):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()

        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0],
            0,
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_config").fetchone()[0],
            0,
        )

    def test_direct_strategy_boundaries_reject_textual_grids_before_writes(self) -> None:
        data = sample_strategy_data(periods=4)
        invalid_grids = (
            ("30", [1.5]),
            (b"30", [1.5]),
            (["30"], [1.5]),
            ([b"30"], [1.5]),
            ([30.0], "1.5"),
            ([30.0], b"1.5"),
            ([30.0], ["1.5"]),
            ([30.0], [b"1.5"]),
        )
        for buy_rsi_values, profit_target_values in invalid_grids:
            operations = (
                lambda buys=buy_rsi_values, targets=profit_target_values: strategy_config_fingerprint(
                    self.cfg,
                    buys,
                    targets,
                ),
                lambda buys=buy_rsi_values, targets=profit_target_values: process_asset_grid(
                    self.conn,
                    data,
                    self.cfg,
                    "TQQQ",
                    "QQQ",
                    buys,
                    targets,
                    rebuild=True,
                ),
            )
            for operation in operations:
                with (
                    self.subTest(
                        buy_rsi_values=buy_rsi_values,
                        profit_target_values=profit_target_values,
                        operation=operation,
                    ),
                    self.assertRaisesRegex(ValueError, "numeric scalars"),
                ):
                    operation()

        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0],
            0,
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_config").fetchone()[0],
            0,
        )

    def test_process_grid_rejects_noncanonical_explicit_fingerprints_before_writes(
        self,
    ) -> None:
        class FingerprintSubclass(str):
            pass

        data = sample_strategy_data(periods=4)
        canonical = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        invalid_fingerprints: tuple[object, ...] = (
            "",
            "caller-supplied",
            canonical.encode(),
            FingerprintSubclass(canonical),
        )

        for invalid_fingerprint in invalid_fingerprints:
            with (
                self.subTest(strategy_fingerprint=invalid_fingerprint),
                self.assertRaisesRegex(ValueError, "canonical configuration"),
            ):
                process_asset_grid(
                    self.conn,
                    data,
                    self.cfg,
                    "TQQQ",
                    "QQQ",
                    [30.0],
                    [1.5],
                    rebuild=True,
                    strategy_fingerprint=invalid_fingerprint,  # type: ignore[arg-type]
                )

        for table_name in (
            "market_data",
            "rsi_values",
            "strategy_config",
            "strategy_state",
            "strategy_summary",
            "strategy_equity",
        ):
            with self.subTest(table_name=table_name):
                self.assertEqual(
                    self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0],
                    0,
                )

    def test_empty_direct_grid_cannot_delete_existing_strategy_data(self) -> None:
        data = sample_strategy_data()
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
        )
        table_names = (
            "market_data",
            "rsi_values",
            "strategy_config",
            "strategy_state",
            "strategy_summary",
            "strategy_equity",
        )
        before = {
            table_name: self.conn.execute(f"SELECT * FROM {table_name} ORDER BY rowid").fetchall()
            for table_name in table_names
        }

        for buy_rsi_values, profit_target_values, rebuild in (
            ([], [1.5], False),
            ([30.0], [], False),
            ([], [1.5], True),
            ([30.0], [], True),
        ):
            with (
                self.subTest(
                    buy_rsi_values=buy_rsi_values,
                    profit_target_values=profit_target_values,
                    rebuild=rebuild,
                ),
                self.assertRaisesRegex(ValueError, "at least one value"),
            ):
                process_asset_grid(
                    self.conn,
                    data.iloc[-1:],
                    self.cfg,
                    "TQQQ",
                    "QQQ",
                    buy_rsi_values,
                    profit_target_values,
                    rebuild=rebuild,
                )

        after = {
            table_name: self.conn.execute(f"SELECT * FROM {table_name} ORDER BY rowid").fetchall()
            for table_name in table_names
        }
        self.assertEqual(after, before)

    def test_config_fingerprint_normalizes_supported_real_subclasses(self) -> None:
        canonical_cfg = BacktestConfig(
            initial_capital=100_000.0,
            rsi_period=3,
            buy_rsi=30.0,
            profit_target_multiple=10.0,
            fee_bps=1.0,
            slippage_bps=2.0,
        )
        subclass_cfg = BacktestConfig(
            initial_capital=Fraction(100_000, 1),
            rsi_period=3,
            buy_rsi=np.float32(30.0),
            profit_target_multiple=Fraction(10, 1),
            fee_bps=np.float32(1.0),
            slippage_bps=Fraction(2, 1),
        )
        canonical = strategy_config_fingerprint(canonical_cfg, [30.0], [1.5])
        normalized = strategy_config_fingerprint(
            subclass_cfg,
            [np.float32(30.0)],
            [Fraction(3, 2)],
        )

        self.assertEqual(normalized, canonical)
        process_asset_grid(
            self.conn,
            sample_strategy_data(),
            subclass_cfg,
            "TQQQ",
            "QQQ",
            [np.float32(30.0)],
            [Fraction(3, 2)],
            rebuild=True,
        )
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                canonical_cfg,
                [30.0],
                [1.5],
            )
        )

    def test_config_fingerprint_canonicalizes_signed_zero_everywhere(self) -> None:
        positive_zero_cfg = BacktestConfig(
            rsi_period=3,
            buy_rsi=0.0,
            fee_bps=0.0,
            slippage_bps=0.0,
        )
        negative_zero_cfg = BacktestConfig(
            rsi_period=3,
            buy_rsi=-0.0,
            fee_bps=-0.0,
            slippage_bps=-0.0,
        )

        self.assertEqual(
            strategy_config_fingerprint(
                positive_zero_cfg,
                [-0.0, 0.0, 30.0],
                [1.5],
            ),
            strategy_config_fingerprint(
                negative_zero_cfg,
                [0.0, -0.0, 30.0],
                [1.5],
            ),
        )

    def refresh_strategy_state_integrity(self) -> None:
        rows = self.conn.execute(
            """
            SELECT asset_symbol, signal_symbol, buy_rsi, profit_target_multiple,
                   start_date, last_date, cash, shares, in_position, entry_price,
                   entry_date, pending_action, prev_equity, trades_executed
            FROM strategy_state
            """
        ).fetchall()
        self.conn.executemany(
            """
            UPDATE strategy_state
            SET integrity_digest = ?
            WHERE asset_symbol = ?
              AND signal_symbol = ?
              AND buy_rsi = ?
              AND profit_target_multiple = ?
            """,
            [
                (
                    _strategy_state_integrity_digest(
                        asset_symbol=row[0],
                        signal_symbol=row[1],
                        buy_rsi=row[2],
                        profit_target_multiple=row[3],
                        start_date=row[4],
                        last_date=row[5],
                        cash=row[6],
                        shares=row[7],
                        in_position=row[8],
                        entry_price=row[9],
                        entry_date=row[10],
                        pending_action=row[11],
                        prev_equity=row[12],
                        trades_executed=row[13],
                    ),
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                )
                for row in rows
            ],
        )

    def refresh_strategy_summary_metrics(self) -> None:
        """Keep synthetic raw rollups internally consistent in preflight tests."""
        rows = self.conn.execute(
            """
            SELECT buy_rsi, profit_target_multiple, first_equity, last_equity,
                   running_max_equity, return_count, return_sum,
                   return_sum_squares, excess_return_count, excess_return_sum,
                   excess_return_sum_squares, positive_return_count, max_drawdown,
                   return_mean, return_m2, excess_return_mean, excess_return_m2
            FROM strategy_summary
            """
        ).fetchall()
        updates = []
        for row in rows:
            rollup = storage_module._summary_rollup_from_row(row[2:])
            self.assertIsNotNone(rollup)
            metrics = _rollup_metrics(rollup)
            updates.append(
                (
                    *(
                        None if pd.isna(metrics[name]) else float(metrics[name])
                        for name in (
                            "total_return",
                            "cagr",
                            "annualized_vol",
                            "sharpe",
                            "kelly_fraction",
                            "hit_rate",
                        )
                    ),
                    row[0],
                    row[1],
                )
            )
        self.conn.executemany(
            """
            UPDATE strategy_summary
            SET total_return = ?, cagr = ?, annualized_vol = ?, sharpe = ?,
                kelly_fraction = ?, hit_rate = ?
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            updates,
        )
        self.refresh_strategy_summary_integrity()

    def refresh_strategy_summary_integrity(self) -> None:
        rows = self.conn.execute(
            f"""
            SELECT {storage_module._STRATEGY_SUMMARY_INTEGRITY_COLUMN_SQL}
            FROM strategy_summary
            """
        ).fetchall()
        self.conn.executemany(
            """
            UPDATE strategy_summary
            SET integrity_digest = ?
            WHERE asset_symbol = ?
              AND signal_symbol = ?
              AND buy_rsi = ?
              AND profit_target_multiple = ?
            """,
            [
                (
                    storage_module._strategy_summary_integrity_digest(row),
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                )
                for row in rows
            ],
        )

    def rewrite_retained_risk_free_returns(
        self,
        transform: Callable[[int, object], float],
    ) -> tuple[float, float]:
        """Rewrite the retained curve and re-sign its coordinated summary."""
        buy_rsi, profit_target = self.conn.execute(
            "SELECT DISTINCT buy_rsi, profit_target_multiple FROM strategy_equity"
        ).fetchone()
        rows = self.conn.execute(
            """
            SELECT date, equity, daily_return, risk_free_return, in_position,
                   action_executed, pending_action, trades_executed
            FROM strategy_equity
            ORDER BY date
            """
        ).fetchall()
        records = []
        for row_index, row in enumerate(rows):
            risk_free_return = transform(row_index, row[3])
            records.append(
                {
                    "asset_symbol": "TQQQ",
                    "signal_symbol": "QQQ",
                    "buy_rsi": float(buy_rsi),
                    "profit_target_multiple": float(profit_target),
                    "date": str(row[0]),
                    "equity": float(row[1]),
                    "daily_return": float(row[2]),
                    "risk_free_return": float(risk_free_return),
                    "in_position": int(row[4]),
                    "action_executed": str(row[5]),
                    "pending_action": str(row[6]),
                    "trades_executed": int(row[7]),
                }
            )
        storage_module.save_equity_records(self.conn, records)
        state = storage_module.load_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            float(buy_rsi),
            float(profit_target),
            rsi_period=self.cfg.rsi_period,
        )
        self.assertIsNotNone(state)
        storage_module.save_strategy_summary(
            self.conn,
            "TQQQ",
            "QQQ",
            float(buy_rsi),
            float(profit_target),
            state,
            _update_summary_rollup(SummaryRollup(), records),
        )
        return float(buy_rsi), float(profit_target)

    def test_direct_grid_failure_rolls_back_partial_market_and_rsi_writes(self) -> None:
        data = sample_strategy_data(periods=2)

        with self.assertRaisesRegex(AssetMarketDataError, "No finite RSI observations"):
            process_asset_grid(
                self.conn,
                data,
                BacktestConfig(rsi_period=14),
                "TQQQ",
                "QQQ",
                [30.0],
                [2.0],
                rebuild=True,
            )

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM market_data").fetchone()[0], 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM rsi_values").fetchone()[0], 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM strategy_state").fetchone()[0], 0)

    def test_direct_grid_failure_preserves_an_existing_outer_transaction(self) -> None:
        self.conn.execute(
            """
            INSERT INTO market_data
            (symbol, date, open, high, low, close, volume)
            VALUES ('SENTINEL', '2026-01-01', 1.0, 1.0, 1.0, 1.0, 0.0)
            """
        )
        data = sample_strategy_data(periods=2)

        with self.assertRaisesRegex(AssetMarketDataError, "No finite RSI observations"):
            process_asset_grid(
                self.conn,
                data,
                BacktestConfig(rsi_period=14),
                "TQQQ",
                "QQQ",
                [30.0],
                [2.0],
                rebuild=True,
            )

        self.assertTrue(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute("SELECT symbol, date FROM market_data ORDER BY symbol, date").fetchall(),
            [("SENTINEL", "2026-01-01")],
        )
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM rsi_values").fetchone()[0], 0)
        self.conn.rollback()

    def test_explicit_rebuild_skips_persisted_state_deep_validation(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)

        with patch(
            "leveraged_trader.storage._strategy_rows_match_config",
            side_effect=AssertionError("explicit rebuild must not inspect discarded compact state"),
        ) as persisted_preflight:
            self.process_grid(data, rebuild=True)

        persisted_preflight.assert_not_called()
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_state").fetchone()[0],
            4,
        )

    def test_explicit_rebuild_does_not_parse_a_corrupt_discarded_checkpoint(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        self.conn.execute("UPDATE strategy_state SET last_date = 'not-a-date'")

        self.process_grid(data, rebuild=True)

        self.assertEqual(
            self.conn.execute("SELECT DISTINCT last_date FROM strategy_state").fetchall(),
            [(data.index[-1].date().isoformat(),)],
        )

    def test_negative_zero_rsi_key_authenticates_and_resumes_incrementally(self) -> None:
        data = sample_strategy_data()
        cfg = BacktestConfig(rsi_period=3)
        grid_kwargs = {
            "base_cfg": cfg,
            "asset_symbol": "TQQQ",
            "signal_symbol": "QQQ",
            "buy_rsi_values": [-0.0],
            "profit_target_values": [1.5],
        }
        process_asset_grid(
            self.conn,
            data,
            rebuild=True,
            **grid_kwargs,
        )

        self.assertEqual(
            self.conn.execute("SELECT buy_rsi, typeof(buy_rsi) FROM strategy_state").fetchone(),
            (0.0, "real"),
        )
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [-0.0],
                [1.5],
            )
        )

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as resumed_grid:
            process_asset_grid(
                self.conn,
                data.iloc[-1:],
                rebuild=False,
                **grid_kwargs,
            )
        self.assertEqual(resumed_grid.call_args.args[7].tolist(), [1])
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [-0.0],
                [1.5],
            )
        )

    def test_explicit_rebuild_still_invalidates_another_signal_for_the_same_asset(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        self.conn.execute(
            """
            INSERT INTO strategy_state
            (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple,
             start_date, last_date, cash, shares, in_position, entry_price,
             pending_action, prev_equity, trades_executed, integrity_digest)
            SELECT asset_symbol, 'SPY', buy_rsi, profit_target_multiple,
                   start_date, last_date, cash, shares, in_position, entry_price,
                   pending_action, prev_equity, trades_executed, integrity_digest
            FROM strategy_state
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )
        prepended = data.iloc[[0]].copy()
        prepended.index = pd.to_datetime(["2026-01-01"])

        self.process_grid(pd.concat([prepended, data]), rebuild=True)

        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_state WHERE signal_symbol = 'SPY'").fetchone()[0],
            0,
        )

    def test_corrupted_nonbest_pending_action_forces_rebuild_before_resume(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=12)
        asset_prices = np.arange(100.0, 112.0)
        data = pd.DataFrame(index=dates)
        for field, values in (
            ("Open", asset_prices),
            ("High", asset_prices * 1.01),
            ("Low", asset_prices * 0.99),
            ("Close", asset_prices),
            ("Volume", np.full(len(dates), 1_000.0)),
        ):
            data[f"AAA_{field}"] = values
        for field in ("Open", "High", "Low", "Close"):
            data[f"{RISK_FREE_SYMBOL}_{field}"] = 5.0
        data[f"{RISK_FREE_SYMBOL}_Volume"] = 0.0
        cfg = BacktestConfig(rsi_period=3, fee_bps=0.0, slippage_bps=0.0)
        grid_kwargs = {
            "base_cfg": cfg,
            "asset_symbol": "AAA",
            "signal_symbol": "AAA",
            "buy_rsi_values": [0.0, 100.0],
            "profit_target_values": [2.0],
        }

        process_asset_grid(
            self.conn,
            data.iloc[:10],
            rebuild=True,
            **grid_kwargs,
        )
        self.assertEqual(
            self.conn.execute(
                """
                SELECT cash, shares, in_position, pending_action, trades_executed
                FROM strategy_state
                WHERE buy_rsi = 0.0
                """
            ).fetchone(),
            (100_000.0, 0.0, 0, "none", 0),
        )
        self.conn.execute("UPDATE strategy_state SET pending_action = 'buy' WHERE buy_rsi = 0.0")
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "AAA",
                "AAA",
                cfg,
                [0.0, 100.0],
                [2.0],
            )
        )
        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as resumed_grid:
            process_asset_grid(
                self.conn,
                data,
                rebuild=False,
                **grid_kwargs,
            )
        self.assertEqual(resumed_grid.call_args.args[7].tolist(), [0, 0])

        fresh = sqlite3.connect(":memory:")
        try:
            init_state_db(fresh)
            process_asset_grid(
                fresh,
                data,
                rebuild=True,
                **grid_kwargs,
            )
            for table in ("strategy_state", "strategy_summary", "strategy_equity"):
                actual = self.conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3, 4, 5").fetchall()
                expected = fresh.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3, 4, 5").fetchall()
                self.assertEqual(actual, expected)
        finally:
            fresh.close()

    def test_blob_encoded_position_flag_forces_rebuild_before_resume(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data.iloc[:30], rebuild=True)
        self.conn.execute(
            """
            UPDATE strategy_state
            SET in_position = CAST('0' AS BLOB)
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )
        self.assertEqual(
            self.conn.execute(
                """
                SELECT typeof(in_position)
                FROM strategy_state
                WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
                """
            ).fetchone(),
            ("blob",),
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as resumed_grid:
            self.process_grid(data, rebuild=False)
        self.assertEqual(resumed_grid.call_args.args[7].tolist(), [0, 0, 0, 0])

    def test_update_rejects_unknown_persisted_pending_action(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        self.conn.execute(
            """
            UPDATE strategy_state
            SET pending_action = 'legacy-hold'
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )
        self.refresh_strategy_state_integrity()

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data, rebuild=False)
        self.assertTrue(all(index == 0 for index in grid_summary.call_args.args[7]))
        repaired_action = self.conn.execute(
            "SELECT pending_action FROM strategy_state WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05"
        ).fetchone()[0]
        self.assertNotEqual(repaired_action, "legacy-hold")

    def test_update_rejects_incoherent_persisted_position_state(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        self.conn.execute(
            """
            UPDATE strategy_state
            SET shares = 10.0,
                in_position = 0,
                entry_price = 100.0,
                pending_action = 'none',
                cash = prev_equity
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )
        self.refresh_strategy_state_integrity()

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data, rebuild=False)
        self.assertTrue(all(index == 0 for index in grid_summary.call_args.args[7]))
        repaired = self.conn.execute(
            "SELECT shares, in_position, entry_price FROM strategy_state "
            "WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05"
        ).fetchone()
        self.assertNotEqual(repaired, (10.0, 0, 100.0))

    def test_authenticated_held_state_rejects_underflowed_entry_notional(self) -> None:
        minimum_subnormal = float(np.nextafter(0.0, np.inf))
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [1e-200, 1e-200],
                    "TQQQ_High": [1e-200, 1e-200],
                    "TQQQ_Low": [1e-200, 1e-200],
                    "TQQQ_Close": [1e-200, 1e-200],
                    "TQQQ_Volume": [1.0, 1.0],
                },
                index=dates,
            ),
            ["TQQQ"],
        )
        state = {
            "start_date": "2026-01-01",
            "last_date": "2026-01-02",
            "cash": minimum_subnormal,
            "shares": 1e-200,
            "in_position": True,
            "entry_price": 1e-200,
            "entry_date": "2026-01-02",
            "pending_action": "none",
            "prev_equity": minimum_subnormal,
            "trades_executed": 1,
        }
        storage_module.save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            state,
        )

        persisted = self.conn.execute(
            """
            SELECT shares * entry_price, integrity_digest
            FROM strategy_state
            WHERE asset_symbol = 'TQQQ' AND signal_symbol = 'QQQ'
            """
        ).fetchone()
        self.assertEqual(persisted[0], 0.0)
        self.assertIsInstance(persisted[1], str)
        self.assertIsNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
            )
        )

    def test_authenticated_high_capital_held_state_rejects_material_mark_gap(self) -> None:
        capital = 1e15
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0, 100.0],
                    "TQQQ_High": [100.0, 100.0],
                    "TQQQ_Low": [100.0, 100.0],
                    "TQQQ_Close": [100.0, 100.0],
                    "TQQQ_Volume": [1.0, 1.0],
                },
                index=dates,
            ),
            ["TQQQ"],
        )
        storage_module.save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-02",
                "cash": 0.0,
                "shares": (capital - 500.0) / 100.0,
                "in_position": True,
                "entry_price": 100.0,
                "entry_date": "2026-01-02",
                "pending_action": "none",
                "prev_equity": capital,
                "trades_executed": 1,
            },
        )

        self.assertIsNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
            )
        )

    def test_authenticated_subnormal_held_mark_requires_relative_continuity(self) -> None:
        minimum_subnormal = float(np.nextafter(0.0, np.inf))
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [1.0, 1.0],
                    "TQQQ_High": [1.0, 1.0],
                    "TQQQ_Low": [1.0, 1.0],
                    "TQQQ_Close": [1.0, 1.0],
                    "TQQQ_Volume": [1.0, 1.0],
                },
                index=dates,
            ),
            ["TQQQ"],
        )
        state = {
            "start_date": "2026-01-01",
            "last_date": "2026-01-02",
            "cash": 0.0,
            "shares": minimum_subnormal,
            "in_position": True,
            "entry_price": 1.0,
            "entry_date": "2026-01-02",
            "pending_action": "none",
            "prev_equity": minimum_subnormal,
            "trades_executed": 1,
        }
        storage_module.save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            state,
        )
        self.assertIsNotNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
            )
        )

        storage_module.save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            {**state, "prev_equity": 33.0 * minimum_subnormal},
        )
        self.assertFalse(
            storage_module._complete_curve_values_match(
                minimum_subnormal,
                33.0 * minimum_subnormal,
            )
        )
        self.assertIsNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
            )
        )

    def test_authenticated_high_price_held_state_rejects_material_entry_gap(self) -> None:
        price = 1e15
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [price, price],
                    "TQQQ_High": [price, price],
                    "TQQQ_Low": [price, price],
                    "TQQQ_Close": [price, price],
                    "TQQQ_Volume": [1.0, 1.0],
                },
                index=dates,
            ),
            ["TQQQ"],
        )
        storage_module.save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-02",
                "cash": 0.0,
                "shares": 1.0,
                "in_position": True,
                "entry_price": price + 500.0,
                "entry_date": "2026-01-02",
                "pending_action": "none",
                "prev_equity": price,
                "trades_executed": 1,
            },
        )

        self.assertIsNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
            )
        )

    def test_update_rejects_held_state_with_wrong_last_close_equity(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        held_config = self.conn.execute(
            """
            SELECT buy_rsi, profit_target_multiple
            FROM strategy_state
            WHERE in_position = 1
            LIMIT 1
            """
        ).fetchone()
        self.assertIsNotNone(held_config)
        self.conn.execute(
            """
            UPDATE strategy_state
            SET shares = shares * 2.0
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            held_config,
        )
        self.refresh_strategy_state_integrity()
        corrupted_shares = self.conn.execute(
            "SELECT shares FROM strategy_state WHERE buy_rsi = ? AND profit_target_multiple = ?",
            held_config,
        ).fetchone()[0]

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data.iloc[-1:], rebuild=False)
        self.assertTrue(all(index == 0 for index in grid_summary.call_args.args[7]))
        stored_shares = self.conn.execute(
            "SELECT shares FROM strategy_state WHERE buy_rsi = ? AND profit_target_multiple = ?",
            held_config,
        ).fetchone()[0]
        self.assertNotEqual(stored_shares, corrupted_shares)

    def test_compact_held_states_persist_exact_kernel_entry_session(self) -> None:
        data = sample_strategy_data(periods=8)
        for field in ("Open", "High", "Low", "Close"):
            data[f"TQQQ_{field}"] = 100.0
        cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)

        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5, 2.0],
            rebuild=True,
        )

        retained_entry_date = self.conn.execute(
            """
            SELECT date
            FROM strategy_equity
            WHERE action_executed = 'buy'
            """
        ).fetchone()[0]
        held_states = self.conn.execute(
            """
            SELECT profit_target_multiple, entry_price, entry_date
            FROM strategy_state
            ORDER BY profit_target_multiple
            """
        ).fetchall()

        self.assertEqual(retained_entry_date, "2026-01-06")
        self.assertEqual(
            held_states,
            [
                (1.5, 100.0, retained_entry_date),
                (2.0, 100.0, retained_entry_date),
            ],
        )

        extended = sample_strategy_data(periods=9)
        for field in ("Open", "High", "Low", "Close"):
            extended[f"TQQQ_{field}"] = 100.0
        process_asset_grid(
            self.conn,
            extended,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5, 2.0],
            rebuild=False,
        )
        self.assertEqual(
            self.conn.execute(
                """
                SELECT profit_target_multiple, entry_date
                FROM strategy_state
                ORDER BY profit_target_multiple
                """
            ).fetchall(),
            [(1.5, retained_entry_date), (2.0, retained_entry_date)],
        )

    def test_compact_held_state_rejects_coordinated_wrong_entry_chronology(self) -> None:
        data = sample_strategy_data(periods=8)
        for field in ("Open", "High", "Low", "Close"):
            data[f"TQQQ_{field}"] = 100.0
        wrong_entry_date = pd.Timestamp("2026-01-05")
        for field in ("Open", "High", "Low", "Close"):
            data.loc[wrong_entry_date, f"TQQQ_{field}"] = 90.0
        cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5, 2.0],
            rebuild=True,
        )

        row = self.conn.execute(
            """
            SELECT asset_symbol, signal_symbol, buy_rsi, profit_target_multiple,
                   start_date, last_date, cash, shares, in_position, entry_price,
                   pending_action, prev_equity, trades_executed
            FROM strategy_state
            WHERE profit_target_multiple = 2.0
            """
        ).fetchone()
        forged_entry_date = wrong_entry_date.date().isoformat()
        forged_entry_price = 90.0
        forged_digest = _strategy_state_integrity_digest(
            asset_symbol=row[0],
            signal_symbol=row[1],
            buy_rsi=row[2],
            profit_target_multiple=row[3],
            start_date=row[4],
            last_date=row[5],
            cash=row[6],
            shares=row[7],
            in_position=row[8],
            entry_price=forged_entry_price,
            entry_date=forged_entry_date,
            pending_action=row[10],
            prev_equity=row[11],
            trades_executed=row[12],
        )
        self.conn.execute(
            """
            UPDATE strategy_state
            SET entry_price = ?, entry_date = ?, integrity_digest = ?
            WHERE profit_target_multiple = 2.0
            """,
            (forged_entry_price, forged_entry_date, forged_digest),
        )
        summary_row = self.conn.execute(
            f"""
            SELECT {storage_module._STRATEGY_SUMMARY_INTEGRITY_COLUMN_SQL}
            FROM strategy_summary
            WHERE profit_target_multiple = 2.0
            """
        ).fetchone()
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET integrity_digest = ?
            WHERE profit_target_multiple = 2.0
            """,
            (storage_module._strategy_summary_integrity_digest(summary_row),),
        )

        self.assertIsNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
            )
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [1.5, 2.0],
            )
        )

    def test_held_state_chronology_ignores_coordinated_rsi_cache_forgery(self) -> None:
        dates = pd.date_range("2026-01-02", periods=8, freq="B")
        asset_prices = np.full(len(dates), 100.0)
        asset_prices[1] = 90.0
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])

        for entry_rule, buy_rsi, signal_direction, forged_rsi in (
            ("lower", 30.0, -1.0, 0.0),
            ("upper", 70.0, 1.0, 100.0),
        ):
            with self.subTest(entry_rule=entry_rule):
                conn = sqlite3.connect(":memory:")
                try:
                    init_state_db(conn)
                    signal_prices = 100.0 + signal_direction * np.arange(len(dates))
                    data = pd.DataFrame(
                        {
                            "TQQQ_Open": asset_prices,
                            "TQQQ_High": asset_prices,
                            "TQQQ_Low": asset_prices,
                            "TQQQ_Close": asset_prices,
                            "TQQQ_Volume": 1_000_000.0,
                            "QQQ_Open": signal_prices,
                            "QQQ_High": signal_prices,
                            "QQQ_Low": signal_prices,
                            "QQQ_Close": signal_prices,
                            "QQQ_Volume": 2_000_000.0,
                            f"{RISK_FREE_SYMBOL}_Open": 5.0,
                            f"{RISK_FREE_SYMBOL}_High": 5.0,
                            f"{RISK_FREE_SYMBOL}_Low": 5.0,
                            f"{RISK_FREE_SYMBOL}_Close": 5.0,
                            f"{RISK_FREE_SYMBOL}_Volume": 0.0,
                        },
                        index=dates,
                    )
                    cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
                    process_asset_grid(
                        conn,
                        data,
                        cfg,
                        "TQQQ",
                        "QQQ",
                        [buy_rsi],
                        [2.0],
                        rebuild=True,
                        rsi_entry_rule=entry_rule,
                    )

                    state_row = conn.execute(
                        """
                        SELECT asset_symbol, signal_symbol, buy_rsi,
                               profit_target_multiple, start_date, last_date,
                               cash, shares, in_position, entry_price,
                               entry_date, pending_action, prev_equity,
                               trades_executed
                        FROM strategy_state
                        """
                    ).fetchone()
                    self.assertEqual(state_row[10], "2026-01-06")
                    forged_entry_date = "2026-01-05"
                    forged_entry_price = 90.0
                    forged_digest = _strategy_state_integrity_digest(
                        asset_symbol=state_row[0],
                        signal_symbol=state_row[1],
                        buy_rsi=state_row[2],
                        profit_target_multiple=state_row[3],
                        start_date=state_row[4],
                        last_date=state_row[5],
                        cash=state_row[6],
                        shares=state_row[7],
                        in_position=state_row[8],
                        entry_price=forged_entry_price,
                        entry_date=forged_entry_date,
                        pending_action=state_row[11],
                        prev_equity=state_row[12],
                        trades_executed=state_row[13],
                    )
                    conn.execute(
                        """
                        UPDATE strategy_state
                        SET entry_price = ?, entry_date = ?, integrity_digest = ?
                        """,
                        (forged_entry_price, forged_entry_date, forged_digest),
                    )
                    # The first signal close has no period-1 RSI. A coordinated
                    # cache edit can fabricate one, but canonical market closes
                    # must remain the chronology authority.
                    conn.execute(
                        """
                        UPDATE rsi_values
                        SET rsi = ?
                        WHERE signal_symbol = 'QQQ'
                          AND rsi_period = 1
                          AND date = '2026-01-02'
                        """,
                        (forged_rsi,),
                    )

                    self.assertIsNone(
                        storage_module.load_strategy_state(
                            conn,
                            "TQQQ",
                            "QQQ",
                            buy_rsi,
                            2.0,
                            rsi_period=1,
                            rsi_entry_rule=entry_rule,
                        )
                    )
                    self.assertFalse(
                        strategy_state_matches_config(
                            conn,
                            "TQQQ",
                            "QQQ",
                            cfg,
                            [buy_rsi],
                            [2.0],
                            rsi_entry_rule=entry_rule,
                        )
                    )
                    summary, curves = summarize_saved_results(
                        conn,
                        workflow_assets,
                        rsi_period=1,
                        rsi_entry_rule=entry_rule,
                    )
                    self.assertTrue(summary.empty)
                    self.assertTrue(curves.empty)
                finally:
                    conn.close()

    def test_held_state_rebuild_uses_current_rsi_period_when_legacy_cache_remains(self) -> None:
        data = sample_strategy_data()
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])

        for entry_rule, buy_rsi in (("lower", 30.0), ("upper", 70.0)):
            with self.subTest(entry_rule=entry_rule):
                conn = sqlite3.connect(":memory:")
                try:
                    init_state_db(conn)
                    for rsi_period in (3, 4):
                        process_asset_grid(
                            conn,
                            data,
                            BacktestConfig(
                                rsi_period=rsi_period,
                                fee_bps=0.0,
                                slippage_bps=0.0,
                            ),
                            "TQQQ",
                            "QQQ",
                            [buy_rsi],
                            [100.0],
                            rebuild=True,
                            rsi_entry_rule=entry_rule,
                        )

                    self.assertEqual(
                        conn.execute(
                            "SELECT DISTINCT rsi_period FROM rsi_values WHERE signal_symbol = 'QQQ' ORDER BY rsi_period"
                        ).fetchall(),
                        [(3,), (4,)],
                    )
                    # A library caller without authoritative configuration
                    # still fails closed when two cache periods are possible.
                    self.assertIsNone(
                        storage_module.load_complete_strategy_equity_curve(
                            conn,
                            "TQQQ",
                            "QQQ",
                            buy_rsi,
                            100.0,
                            rsi_entry_rule=entry_rule,
                            allow_unbound_backtest_config=True,
                        )
                    )

                    curve = storage_module.load_complete_strategy_equity_curve(
                        conn,
                        "TQQQ",
                        "QQQ",
                        buy_rsi,
                        100.0,
                        rsi_period=4,
                        rsi_entry_rule=entry_rule,
                        allow_unbound_backtest_config=True,
                    )
                    self.assertIsNotNone(curve)
                    assert curve is not None
                    self.assertEqual(len(curve), len(data))
                    summary, reported_curves = summarize_saved_results(
                        conn,
                        workflow_assets,
                        rsi_period=4,
                        rsi_entry_rule=entry_rule,
                    )
                    self.assertEqual(summary["Asset"].tolist(), ["TQQQ"])
                    self.assertEqual(len(reported_curves), len(data))
                finally:
                    conn.close()

    def test_flat_pending_action_chronology_rejects_coordinated_redigest(self) -> None:
        dates = pd.date_range("2026-01-02", periods=4, freq="B")
        asset_close = np.asarray([100.0, 100.0, 101.0, 110.0])
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])

        for entry_rule, buy_rsi, signal_close, forged_action in (
            ("lower", 30.0, [100.0, 90.0, 100.0, 110.0], "buy"),
            ("upper", 70.0, [100.0, 110.0, 100.0, 90.0], "buy"),
        ):
            with self.subTest(entry_rule=entry_rule), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                signal_close_array = np.asarray(signal_close, dtype=float)
                data = pd.DataFrame(
                    {
                        "TQQQ_Open": asset_close,
                        "TQQQ_High": asset_close * 1.2,
                        "TQQQ_Low": asset_close * 0.9,
                        "TQQQ_Close": asset_close,
                        "TQQQ_Volume": 1_000_000.0,
                        "QQQ_Open": signal_close_array,
                        "QQQ_High": signal_close_array,
                        "QQQ_Low": signal_close_array,
                        "QQQ_Close": signal_close_array,
                        "QQQ_Volume": 2_000_000.0,
                        f"{RISK_FREE_SYMBOL}_Open": 5.0,
                        f"{RISK_FREE_SYMBOL}_High": 5.0,
                        f"{RISK_FREE_SYMBOL}_Low": 5.0,
                        f"{RISK_FREE_SYMBOL}_Close": 5.0,
                        f"{RISK_FREE_SYMBOL}_Volume": 0.0,
                    },
                    index=dates,
                )
                cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
                process_asset_grid(
                    conn,
                    data,
                    cfg,
                    "TQQQ",
                    "QQQ",
                    [buy_rsi],
                    [1.05],
                    rebuild=True,
                    rsi_entry_rule=entry_rule,
                )
                row = conn.execute(
                    """
                    SELECT asset_symbol, signal_symbol, buy_rsi, profit_target_multiple,
                           start_date, last_date, cash, shares, in_position, entry_price,
                           entry_date, pending_action, prev_equity, trades_executed
                    FROM strategy_state
                    """
                ).fetchone()
                self.assertFalse(bool(row[8]))
                self.assertEqual(row[11], "none")
                forged_digest = _strategy_state_integrity_digest(
                    asset_symbol=row[0],
                    signal_symbol=row[1],
                    buy_rsi=row[2],
                    profit_target_multiple=row[3],
                    start_date=row[4],
                    last_date=row[5],
                    cash=row[6],
                    shares=row[7],
                    in_position=row[8],
                    entry_price=row[9],
                    entry_date=row[10],
                    pending_action=forged_action,
                    prev_equity=row[12],
                    trades_executed=row[13],
                )
                conn.execute(
                    "UPDATE strategy_state SET pending_action = ?, integrity_digest = ?",
                    (forged_action, forged_digest),
                )
                equity_row = conn.execute(
                    """
                    SELECT date, equity, daily_return, risk_free_return, in_position,
                           action_executed, trades_executed
                    FROM strategy_equity
                    ORDER BY date DESC
                    LIMIT 1
                    """
                ).fetchone()
                forged_equity_digest = storage_module._strategy_equity_integrity_digest(
                    asset_symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=buy_rsi,
                    profit_target_multiple=1.05,
                    date=equity_row[0],
                    equity=equity_row[1],
                    daily_return=equity_row[2],
                    risk_free_return=equity_row[3],
                    in_position=equity_row[4],
                    action_executed=equity_row[5],
                    pending_action=forged_action,
                    trades_executed=equity_row[6],
                )
                conn.execute(
                    """
                    UPDATE strategy_equity
                    SET pending_action = ?, integrity_digest = ?
                    WHERE date = ?
                    """,
                    (forged_action, forged_equity_digest, equity_row[0]),
                )

                self.assertIsNone(
                    storage_module.load_strategy_state(
                        conn,
                        "TQQQ",
                        "QQQ",
                        buy_rsi,
                        1.05,
                        rsi_period=1,
                        rsi_entry_rule=entry_rule,
                    )
                )
                self.assertFalse(
                    strategy_state_matches_config(
                        conn,
                        "TQQQ",
                        "QQQ",
                        cfg,
                        [buy_rsi],
                        [1.05],
                        rsi_entry_rule=entry_rule,
                    )
                )
                self.assertIsNone(
                    storage_module.load_complete_strategy_equity_curve(
                        conn,
                        "TQQQ",
                        "QQQ",
                        buy_rsi,
                        1.05,
                        rsi_period=1,
                        rsi_entry_rule=entry_rule,
                        allow_unbound_backtest_config=True,
                    )
                )
                summary, curves = summarize_saved_results(
                    conn,
                    workflow_assets,
                    rsi_period=1,
                    rsi_entry_rule=entry_rule,
                )
                self.assertTrue(summary.empty)
                self.assertTrue(curves.empty)

    def test_authenticated_tiny_held_state_rejects_incoherent_share_value(self) -> None:
        dates = pd.date_range("2026-01-01", periods=4, freq="B")
        asset_prices = np.full(4, 100.0)
        signal_prices = np.array([100.0, 99.0, 98.0, 97.0])
        data = pd.DataFrame(
            {
                "TQQQ_Open": asset_prices,
                "TQQQ_High": asset_prices,
                "TQQQ_Low": asset_prices,
                "TQQQ_Close": asset_prices,
                "TQQQ_Volume": 1,
                "QQQ_Open": signal_prices,
                "QQQ_High": signal_prices,
                "QQQ_Low": signal_prices,
                "QQQ_Close": signal_prices,
                "QQQ_Volume": 1,
                f"{RISK_FREE_SYMBOL}_Open": 0.0,
                f"{RISK_FREE_SYMBOL}_High": 0.0,
                f"{RISK_FREE_SYMBOL}_Low": 0.0,
                f"{RISK_FREE_SYMBOL}_Close": 0.0,
                f"{RISK_FREE_SYMBOL}_Volume": 0,
            },
            index=dates,
        )

        for initial_capital, forged_shares in (
            (5e-14, 8e-16),
            (1e-300, 1.6e-302),
        ):
            with self.subTest(initial_capital=initial_capital):
                cfg = BacktestConfig(
                    initial_capital=initial_capital,
                    rsi_period=1,
                    fee_bps=0.0,
                    slippage_bps=0.0,
                )
                process_asset_grid(
                    self.conn,
                    data,
                    cfg,
                    "TQQQ",
                    "QQQ",
                    [30.0],
                    [2.0],
                    rebuild=True,
                )
                self.assertIsNotNone(
                    storage_module.load_strategy_state(
                        self.conn,
                        "TQQQ",
                        "QQQ",
                        30.0,
                        2.0,
                        rsi_period=1,
                    )
                )
                self.assertTrue(
                    strategy_state_matches_config(
                        self.conn,
                        "TQQQ",
                        "QQQ",
                        cfg,
                        [30.0],
                        [2.0],
                    )
                )

                self.conn.execute(
                    "UPDATE strategy_state SET shares = ?",
                    (forged_shares,),
                )
                self.refresh_strategy_state_integrity()

                self.assertIsNone(
                    storage_module.load_strategy_state(
                        self.conn,
                        "TQQQ",
                        "QQQ",
                        30.0,
                        2.0,
                        rsi_period=1,
                    )
                )
                self.assertFalse(
                    strategy_state_matches_config(
                        self.conn,
                        "TQQQ",
                        "QQQ",
                        cfg,
                        [30.0],
                        [2.0],
                    )
                )

    def test_maximum_float_account_comparisons_fail_closed(self) -> None:
        maximum_float = np.finfo(np.float64).max
        self.assertTrue(storage_module._complete_curve_values_match(maximum_float, maximum_float))
        self.assertFalse(storage_module._complete_curve_values_match(maximum_float, 1.0))
        self.assertFalse(storage_module._complete_curve_values_match(maximum_float, -maximum_float))

        storage_module.save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0, 100.0],
                    "TQQQ_High": [100.0, 100.0],
                    "TQQQ_Low": [100.0, 100.0],
                    "TQQQ_Close": [100.0, 100.0],
                    "TQQQ_Volume": [1, 1],
                },
                index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
            ),
            ["TQQQ"],
        )
        coherent_state = {
            "start_date": "2026-01-01",
            "last_date": "2026-01-02",
            "cash": 0.0,
            "shares": 0.01,
            "in_position": True,
            "entry_price": 100.0,
            "entry_date": "2026-01-02",
            "pending_action": "none",
            "prev_equity": 1.0,
            "trades_executed": 1,
        }
        storage_module.save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            coherent_state,
        )
        self.assertIsNotNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
            )
        )

        storage_module.save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {**coherent_state, "prev_equity": maximum_float},
        )

        self.assertIsNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
            )
        )

    def test_authenticated_held_state_rejects_entry_from_earlier_session(self) -> None:
        storage_module.save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [90.0, 100.0],
                    "TQQQ_High": [90.0, 100.0],
                    "TQQQ_Low": [90.0, 100.0],
                    "TQQQ_Close": [90.0, 100.0],
                    "TQQQ_Volume": [1_000_000.0, 1_000_000.0],
                },
                index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
            ),
            ["TQQQ"],
        )
        storage_module.save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-02",
                "cash": 0.0,
                "shares": 1_000.0,
                "in_position": True,
                "entry_price": 90.0,
                "entry_date": "2026-01-01",
                "pending_action": "none",
                "prev_equity": 100_000.0,
                "trades_executed": 1,
            },
        )

        self.assertIsNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
            )
        )

    def test_update_rebuilds_incoherent_persisted_summary_rollup_before_coercion(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET return_sum_squares = -1.0
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data, rebuild=False)
        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0, 0, 0, 0])
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )

    def test_authenticated_positive_return_count_must_agree_with_return_sum(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        changed = self.conn.execute(
            """
            UPDATE strategy_summary
            SET positive_return_count = 0,
                hit_rate = 0.0
            WHERE return_count > 0 AND return_sum > 0.0
            """
        ).rowcount
        self.assertGreater(changed, 0)
        self.refresh_strategy_summary_integrity()

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )

    def test_impossible_persisted_return_domain_forces_full_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        return_count = self.conn.execute(
            """
            SELECT return_count
            FROM strategy_summary
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        ).fetchone()[0]
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET return_sum = ?,
                return_sum_squares = ?,
                positive_return_count = 0,
                return_mean = -2.0,
                return_m2 = 0.0
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """,
            (-2.0 * return_count, 4.0 * return_count),
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )
        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0, 0, 0, 0])
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )

    def test_one_return_rollup_must_match_equity_endpoints_and_positive_count(self) -> None:
        signal_dates = pd.date_range("2026-01-01", periods=6, freq="B")
        signal_close = pd.Series([105.0, 104.0, 103.0, 102.0, 101.0, 100.0], index=signal_dates)
        signal_history = pd.DataFrame(
            {
                "QQQ_Open": signal_close,
                "QQQ_High": signal_close,
                "QQQ_Low": signal_close,
                "QQQ_Close": signal_close,
                "QQQ_Volume": 1.0,
            },
            index=signal_dates,
        )
        asset_dates = signal_dates[-2:]
        data = pd.DataFrame(
            {
                "TQQQ_Open": [100.0, 100.0],
                "TQQQ_High": [100.0, 110.0],
                "TQQQ_Low": [100.0, 100.0],
                "TQQQ_Close": [100.0, 110.0],
                "TQQQ_Volume": 1.0,
                "QQQ_Open": signal_close.loc[asset_dates],
                "QQQ_High": signal_close.loc[asset_dates],
                "QQQ_Low": signal_close.loc[asset_dates],
                "QQQ_Close": signal_close.loc[asset_dates],
                "QQQ_Volume": 1.0,
                f"{RISK_FREE_SYMBOL}_Open": 5.0,
                f"{RISK_FREE_SYMBOL}_High": 5.0,
                f"{RISK_FREE_SYMBOL}_Low": 5.0,
                f"{RISK_FREE_SYMBOL}_Close": 5.0,
                f"{RISK_FREE_SYMBOL}_Volume": 0.0,
            },
            index=asset_dates,
        )
        cfg = BacktestConfig(rsi_period=3, fee_bps=0.0, slippage_bps=0.0)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0],
            profit_target_values=[1.5],
            rebuild=True,
            signal_history=signal_history,
        )
        original_rollup = self.conn.execute(
            """
            SELECT return_sum, return_sum_squares, positive_return_count,
                   return_mean, return_m2
            FROM strategy_summary
            """
        ).fetchone()
        self.assertEqual(original_rollup[2], 1)
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [1.5],
            )
        )

        self.conn.execute("UPDATE strategy_summary SET positive_return_count = 0")
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [1.5],
            )
        )
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET return_sum = ?, return_sum_squares = ?, positive_return_count = ?,
                return_mean = ?, return_m2 = ?
            """,
            original_rollup,
        )

        self.conn.execute(
            """
            UPDATE strategy_summary
            SET return_sum = 0.2, return_sum_squares = 0.04,
                positive_return_count = 1, return_mean = 0.2, return_m2 = 0.0
            """
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [1.5],
            )
        )

        self.conn.execute("UPDATE strategy_summary SET positive_return_count = 0")
        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            process_asset_grid(
                self.conn,
                data.iloc[-1:],
                cfg,
                "TQQQ",
                "QQQ",
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
                rebuild=False,
            )
        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0])
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [1.5],
            )
        )

    def test_multi_return_rollup_moments_must_match_equity_endpoints(self) -> None:
        data = sample_strategy_data(periods=3)
        cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [100.0],
            [10.0],
            rebuild=True,
        )
        self.assertEqual(
            self.conn.execute("SELECT return_count FROM strategy_summary").fetchone()[0],
            2,
        )
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET return_sum = 0.05,
                return_sum_squares = 0.0125,
                return_mean = 0.025,
                return_m2 = 0.01125,
                positive_return_count = 1
            """
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [100.0],
                [10.0],
            )
        )

    def test_constant_multi_return_rollup_path_must_be_consistent_in_preflight(self) -> None:
        data = sample_strategy_data(periods=4)
        cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [100.0],
            [10.0],
            rebuild=True,
        )
        self.conn.execute(
            """
            UPDATE strategy_state
            SET cash = 195.3125,
                shares = 0.0,
                in_position = 0,
                entry_price = NULL,
                entry_date = NULL,
                pending_action = 'none',
                prev_equity = 195.3125,
                trades_executed = 2
            """
        )
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET trades_executed = 2,
                first_equity = 100.0,
                last_equity = 195.3125,
                running_max_equity = 195.3125,
                return_count = 3,
                return_sum = 0.75,
                return_sum_squares = 0.1875,
                positive_return_count = 3,
                max_drawdown = 0.0,
                return_mean = 0.25,
                return_m2 = 0.0
            """
        )
        self.refresh_strategy_state_integrity()
        self.refresh_strategy_summary_metrics()

        def matches() -> bool:
            return strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [100.0],
                [10.0],
            )

        self.assertFalse(matches())

        mutations = (
            (
                "UPDATE strategy_state SET cash = 156.25, prev_equity = 156.25",
                "UPDATE strategy_summary SET last_equity = 156.25, running_max_equity = 156.25",
            ),
            (None, "UPDATE strategy_summary SET positive_return_count = 2"),
            (None, "UPDATE strategy_summary SET max_drawdown = -0.5"),
        )
        for state_mutation, summary_mutation in mutations:
            with self.subTest(summary_mutation=summary_mutation):
                if state_mutation is not None:
                    self.conn.execute(state_mutation)
                self.conn.execute(summary_mutation)
                self.assertFalse(matches())
                self.conn.execute("UPDATE strategy_state SET cash = 195.3125, prev_equity = 195.3125")
                self.conn.execute(
                    """
                    UPDATE strategy_summary
                    SET last_equity = 195.3125,
                        running_max_equity = 195.3125,
                        positive_return_count = 3,
                        max_drawdown = 0.0
                    """
                )

    def test_preflight_does_not_treat_tiny_nonzero_m2_as_constant_returns(self) -> None:
        return_count = 1_000
        dates = pd.date_range("2020-01-02", periods=return_count + 1, freq="B")
        data = pd.DataFrame(index=dates)
        for symbol, volume in (("TQQQ", 1_000_000.0), ("QQQ", 2_000_000.0)):
            data[f"{symbol}_Open"] = 100.0
            data[f"{symbol}_High"] = 101.0
            data[f"{symbol}_Low"] = 99.0
            data[f"{symbol}_Close"] = 100.0
            data[f"{symbol}_Volume"] = volume
        for field in ("Open", "High", "Low", "Close"):
            data[f"{RISK_FREE_SYMBOL}_{field}"] = 5.0
        data[f"{RISK_FREE_SYMBOL}_Volume"] = 0.0
        cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [100.0],
            [10.0],
            rebuild=True,
        )

        average_return = 0.01
        return_offset = np.sqrt(4.5e-12 / return_count)
        returns = np.where(
            np.arange(return_count) % 2 == 0,
            average_return - return_offset,
            average_return + return_offset,
        )
        return_sum = 0.0
        return_sum_squares = 0.0
        return_mean = 0.0
        return_m2 = 0.0
        equity = np.longdouble(100.0)
        for observation_count, daily_return in enumerate(returns, start=1):
            return_sum += float(daily_return)
            return_sum_squares += float(daily_return * daily_return)
            delta = float(daily_return) - return_mean
            return_mean += delta / observation_count
            return_m2 += delta * (float(daily_return) - return_mean)
            equity *= np.longdouble(1.0) + np.longdouble(daily_return)
        last_equity = float(equity)
        self.assertGreater(return_m2, 0.0)

        self.conn.execute(
            """
            UPDATE strategy_state
            SET cash = ?,
                shares = 0.0,
                in_position = 0,
                entry_price = NULL,
                entry_date = NULL,
                pending_action = 'none',
                prev_equity = ?,
                trades_executed = 2
            """,
            (last_equity, last_equity),
        )
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET trades_executed = 2,
                first_equity = 100.0,
                last_equity = ?,
                running_max_equity = ?,
                return_count = ?,
                return_sum = ?,
                return_sum_squares = ?,
                positive_return_count = ?,
                max_drawdown = 0.0,
                return_mean = ?,
                return_m2 = ?
            """,
            (
                last_equity,
                last_equity,
                return_count,
                return_sum,
                return_sum_squares,
                return_count,
                return_mean,
                return_m2,
            ),
        )
        self.refresh_strategy_state_integrity()
        self.refresh_strategy_summary_metrics()

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [100.0],
                [10.0],
            )
        )

    def test_two_return_multiset_path_must_be_consistent_in_preflight(self) -> None:
        data = sample_strategy_data(periods=3)
        cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [100.0],
            [10.0],
            rebuild=True,
        )
        self.conn.execute(
            """
            UPDATE strategy_state
            SET cash = 100.0,
                shares = 0.0,
                in_position = 0,
                entry_price = NULL,
                entry_date = NULL,
                pending_action = 'none',
                prev_equity = 100.0,
                trades_executed = 2
            """
        )
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET trades_executed = 2,
                first_equity = 100.0,
                last_equity = 100.0,
                running_max_equity = 100.0,
                return_count = 2,
                return_sum = 0.5,
                return_sum_squares = 1.25,
                positive_return_count = 1,
                max_drawdown = -0.5,
                return_mean = 0.25,
                return_m2 = 1.125
            """
        )
        self.refresh_strategy_state_integrity()
        self.refresh_strategy_summary_metrics()

        def matches() -> bool:
            return strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [100.0],
                [10.0],
            )

        self.assertFalse(matches())

        for column, value in (
            ("positive_return_count", 2),
            ("max_drawdown", -0.9),
            ("running_max_equity", 110.0),
        ):
            with self.subTest(column=column):
                self.conn.execute(f"UPDATE strategy_summary SET {column} = ?", (value,))
                self.assertFalse(matches())
                reset = {
                    "positive_return_count": 1,
                    "max_drawdown": -0.5,
                    "running_max_equity": 100.0,
                }[column]
                self.conn.execute(f"UPDATE strategy_summary SET {column} = ?", (reset,))

    def test_two_return_preflight_allows_roundoff_ambiguous_zero_root(self) -> None:
        data = sample_strategy_data(periods=3)
        cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [100.0],
            [10.0],
            rebuild=True,
        )
        self.conn.execute(
            """
            UPDATE strategy_state
            SET cash = 50.0,
                shares = 0.0,
                in_position = 0,
                entry_price = NULL,
                entry_date = NULL,
                pending_action = 'none',
                prev_equity = 50.0,
                trades_executed = 2
            """
        )
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET trades_executed = 2,
                first_equity = 100.0,
                last_equity = 50.0,
                running_max_equity = 100.0,
                return_count = 2,
                return_sum = -0.5,
                return_sum_squares = 0.25,
                positive_return_count = 0,
                max_drawdown = -0.5,
                return_mean = -0.25,
                return_m2 = 0.125
            """
        )
        self.refresh_strategy_state_integrity()
        self.refresh_strategy_summary_metrics()

        def matches() -> bool:
            return strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [100.0],
                [10.0],
            )

        self.assertFalse(matches())

        # A material +25% root is not sign-ambiguous, so keeping the persisted
        # positive count at zero must still fail preflight.
        self.conn.execute("UPDATE strategy_state SET cash = 62.5, prev_equity = 62.5")
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET last_equity = 62.5,
                return_sum = -0.25,
                return_sum_squares = 0.3125,
                return_mean = -0.125,
                return_m2 = 0.28125
            """
        )
        self.assertFalse(matches())

    def test_two_return_preflight_allows_ill_conditioned_endpoint_reconstruction(self) -> None:
        data = sample_strategy_data(periods=3)
        cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [100.0],
            [10.0],
            rebuild=True,
        )
        returns = (np.nextafter(-1.0, 0.0), -0.5)
        return_sum = 0.0
        return_sum_squares = 0.0
        return_mean = 0.0
        return_m2 = 0.0
        first_equity = 1e12
        last_equity = first_equity
        max_drawdown = 0.0
        for observation_count, daily_return in enumerate(returns, start=1):
            return_sum += daily_return
            return_sum_squares += daily_return * daily_return
            delta = daily_return - return_mean
            return_mean += delta / observation_count
            return_m2 += delta * (daily_return - return_mean)
            last_equity *= 1.0 + daily_return
            max_drawdown = min(
                max_drawdown,
                last_equity / first_equity - 1.0,
            )

        self.conn.execute(
            """
            UPDATE strategy_state
            SET cash = ?,
                shares = 0.0,
                in_position = 0,
                entry_price = NULL,
                entry_date = NULL,
                pending_action = 'none',
                prev_equity = ?,
                trades_executed = 2
            """,
            (last_equity, last_equity),
        )
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET trades_executed = 2,
                first_equity = ?,
                last_equity = ?,
                running_max_equity = ?,
                return_count = 2,
                return_sum = ?,
                return_sum_squares = ?,
                positive_return_count = 0,
                max_drawdown = ?,
                return_mean = ?,
                return_m2 = ?
            """,
            (
                first_equity,
                last_equity,
                first_equity,
                return_sum,
                return_sum_squares,
                max_drawdown,
                return_mean,
                return_m2,
            ),
        )
        self.refresh_strategy_state_integrity()
        self.refresh_strategy_summary_metrics()

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [100.0],
                [10.0],
            )
        )

        fake_last_equity = 1.0
        self.conn.execute(
            "UPDATE strategy_state SET cash = ?, prev_equity = ?",
            (fake_last_equity, fake_last_equity),
        )
        self.conn.execute(
            "UPDATE strategy_summary SET last_equity = ?, max_drawdown = ?",
            (
                fake_last_equity,
                fake_last_equity / first_equity - 1.0,
            ),
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [100.0],
                [10.0],
            )
        )
        self.conn.execute(
            "UPDATE strategy_state SET cash = ?, prev_equity = ?",
            (last_equity, last_equity),
        )
        self.conn.execute(
            "UPDATE strategy_summary SET last_equity = ?, max_drawdown = ?",
            (last_equity, max_drawdown),
        )

        # Endpoint cancellation does not make the reconstructed return signs
        # ambiguous: both roots remain materially negative.
        self.conn.execute("UPDATE strategy_summary SET positive_return_count = 2")
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [100.0],
                [10.0],
            )
        )

    def test_zero_return_moments_require_unchanged_persisted_equity_endpoints(self) -> None:
        data = sample_strategy_data(periods=4)
        cfg = BacktestConfig(rsi_period=1, fee_bps=0.0, slippage_bps=0.0)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [100.0],
            [10.0],
            rebuild=True,
        )
        first_equity, last_equity = self.conn.execute(
            "SELECT first_equity, last_equity FROM strategy_summary"
        ).fetchone()
        self.assertNotEqual(first_equity, last_equity)
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET return_sum = 0.0,
                return_sum_squares = 0.0,
                return_mean = 0.0,
                return_m2 = 0.0,
                positive_return_count = 0
            """
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [100.0],
                [10.0],
            )
        )

    def test_zero_trade_equity_change_fails_persisted_state_preflight(self) -> None:
        data = sample_strategy_data()
        signal_close = np.linspace(100.0, 140.0, len(data))
        data["QQQ_Open"] = signal_close
        data["QQQ_High"] = signal_close + 1.0
        data["QQQ_Low"] = signal_close - 1.0
        data["QQQ_Close"] = signal_close
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[0.0],
            profit_target_values=[1.5],
            rebuild=True,
        )
        return_count, trades_executed = self.conn.execute(
            "SELECT return_count, trades_executed FROM strategy_summary"
        ).fetchone()
        self.assertGreater(return_count, 0)
        self.assertEqual(trades_executed, 0)

        return_sum = -0.5
        return_sum_squares = 0.25
        return_mean = return_sum / return_count
        return_m2 = return_sum_squares - return_sum * return_sum / return_count
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET first_equity = 200000.0,
                last_equity = 100000.0,
                running_max_equity = 200000.0,
                return_sum = ?,
                return_sum_squares = ?,
                positive_return_count = 0,
                max_drawdown = -0.5,
                return_mean = ?,
                return_m2 = ?
            """,
            (return_sum, return_sum_squares, return_mean, return_m2),
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [0.0],
                [1.5],
            )
        )

        process_asset_grid(
            self.conn,
            data.iloc[-1:],
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[0.0],
            profit_target_values=[1.5],
            rebuild=False,
        )
        rebuilt = self.conn.execute(
            """
            SELECT first_equity, last_equity, running_max_equity,
                   return_sum, return_sum_squares, positive_return_count,
                   max_drawdown, return_mean, return_m2
            FROM strategy_summary
            """
        ).fetchone()
        np.testing.assert_allclose(
            rebuilt,
            [100000.0, 100000.0, 100000.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0],
        )

    def test_missing_centered_moments_force_full_saved_history_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET return_mean = NULL,
                return_m2 = NULL,
                excess_return_mean = NULL,
                excess_return_m2 = NULL
            """
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )

        self.process_grid(data.iloc[-1:], rebuild=False)

        rows = self.conn.execute(
            """
            SELECT trading_days, return_mean, return_m2,
                   excess_return_mean, excess_return_m2
            FROM strategy_summary
            """
        ).fetchall()
        self.assertEqual({row[0] for row in rows}, {len(data)})
        self.assertTrue(all(value is not None for row in rows for value in row[1:]))

    def test_centered_moments_preserve_tiny_variance_and_merge(self) -> None:
        returns = np.where(
            np.arange(1_000) % 2 == 0,
            0.001 + 1e-12,
            0.001 - 1e-12,
        )
        records = [{"equity": 100_000.0, "daily_return": 0.0, "risk_free_return": 0.0}]
        records.extend(
            {"equity": 100_000.0, "daily_return": float(value), "risk_free_return": 0.0} for value in returns
        )

        rollup = _update_summary_rollup(SummaryRollup(), records)
        metrics = _rollup_metrics(rollup)
        expected_variance = float(np.var(returns, ddof=1))
        self.assertGreater(float(metrics["annualized_vol"]), 0.0)
        np.testing.assert_allclose(
            metrics["annualized_vol"],
            np.sqrt(expected_variance * 252),
            rtol=1e-8,
        )

        left, right = np.array_split(returns, 2)
        merged_count, merged_mean, merged_m2 = _merge_centered_moments(
            len(left),
            float(left.mean()),
            float(np.square(left - left.mean()).sum()),
            len(right),
            float(right.mean()),
            float(np.square(right - right.mean()).sum()),
        )
        self.assertEqual(merged_count, len(returns))
        np.testing.assert_allclose(merged_mean, rollup.return_mean, rtol=0.0, atol=1e-18)
        np.testing.assert_allclose(merged_m2, rollup.return_m2, rtol=1e-8)

    def test_rollup_metrics_rejects_unrepresentable_cagr_consistently(self) -> None:
        rollup = SummaryRollup(
            first_equity=1.0,
            last_equity=100.0,
            running_max_equity=100.0,
            return_count=1,
            positive_return_count=1,
            max_drawdown=0.0,
        )

        with self.assertRaisesRegex(ValueError, "performance_summary produced a non-finite CAGR"):
            _rollup_metrics(rollup)

    def test_inconsistent_persisted_centered_m2_forces_full_history_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected = self.conn.execute(
            """
            SELECT return_m2, excess_return_m2, annualized_vol, sharpe
            FROM strategy_summary
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.5
            """
        ).fetchone()
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET return_m2 = 1000000000.0,
                excess_return_m2 = 1000000000.0
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.5
            """
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )

        self.process_grid(data.iloc[-1:], rebuild=False)

        rebuilt = self.conn.execute(
            """
            SELECT return_m2, excess_return_m2, annualized_vol, sharpe
            FROM strategy_summary
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.5
            """
        ).fetchone()
        np.testing.assert_allclose(rebuilt, expected, rtol=0.0, atol=0.0)

    def test_malformed_persisted_centered_moments_fail_closed(self) -> None:
        for malformed_m2 in (-1.0, float("inf"), float("nan")):
            with self.subTest(m2=malformed_m2):
                self.assertFalse(
                    storage_module._persisted_centered_moments_are_consistent(
                        2,
                        0.1,
                        0.01,
                        0.05,
                        malformed_m2,
                    )
                )

        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected = self.conn.execute(
            """
            SELECT return_m2, excess_return_m2
            FROM strategy_summary
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.5
            """
        ).fetchone()
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET return_m2 = -1.0
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.5
            """
        )
        self.refresh_strategy_summary_integrity()

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )
        self.process_grid(data.iloc[-1:], rebuild=False)
        self.assertEqual(
            self.conn.execute(
                """
                SELECT return_m2, excess_return_m2
                FROM strategy_summary
                WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.5
                """
            ).fetchone(),
            expected,
        )

    def test_best_summary_selection_rejects_redigested_negative_centered_moments(self) -> None:
        data = sample_strategy_data()
        for moment_column in ("return_m2", "excess_return_m2"):
            with self.subTest(moment_column=moment_column):
                self.process_grid(data, rebuild=True)
                best_config = storage_module._best_summary_config(self.conn, "TQQQ", "QQQ")
                self.assertIsNotNone(best_config)
                self.conn.execute(
                    f"""
                    UPDATE strategy_summary
                    SET {moment_column} = -1.0
                    WHERE buy_rsi = ? AND profit_target_multiple = ?
                    """,
                    best_config,
                )
                # Recompute every affected ranking field and authenticate the
                # coordinated mutation so this exercises semantic validation,
                # rather than merely detecting a stale integrity digest.
                self.refresh_strategy_summary_metrics()
                corrupted_row = self.conn.execute(
                    f"""
                    SELECT {storage_module._STRATEGY_SUMMARY_INTEGRITY_COLUMN_SQL},
                           integrity_digest
                    FROM strategy_summary
                    WHERE buy_rsi = ? AND profit_target_multiple = ?
                    """,
                    best_config,
                ).fetchone()
                self.assertTrue(storage_module._strategy_summary_integrity_row_is_valid(corrupted_row))
                self.assertEqual(
                    corrupted_row[storage_module._STRATEGY_SUMMARY_INTEGRITY_COLUMNS.index(moment_column)],
                    -1.0,
                )

                self.assertIsNone(storage_module.load_best_strategy_summary(self.conn, "TQQQ", "QQQ"))
                self.assertIsNone(storage_module._best_summary_config(self.conn, "TQQQ", "QQQ"))

    def test_corrupted_derived_summary_metric_forces_rebuild_before_selection(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        expected_best = storage_module._best_summary_config(self.conn, "TQQQ", "QQQ")
        self.assertIsNotNone(expected_best)
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET sharpe = -999.0
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            expected_best,
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )
        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0, 0, 0, 0])
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )
        self.assertEqual(
            storage_module._best_summary_config(self.conn, "TQQQ", "QQQ"),
            expected_best,
        )

    def test_coordinated_excess_moment_and_sharpe_mutation_forces_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        expected_best = storage_module._best_summary_config(self.conn, "TQQQ", "QQQ")
        self.assertIsNotNone(expected_best)
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET excess_return_sum = 0.0,
                excess_return_sum_squares = CAST(excess_return_count AS REAL),
                excess_return_mean = 0.0,
                excess_return_m2 = CAST(excess_return_count AS REAL),
                sharpe = 0.0
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            expected_best,
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )
        self.assertIsNone(storage_module._best_summary_config(self.conn, "TQQQ", "QQQ"))
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])
        reported, curves = summarize_saved_results(self.conn, workflow_assets)
        self.assertTrue(reported.empty)
        self.assertTrue(curves.empty)

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data.iloc[-1:], rebuild=False)
        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0, 0, 0, 0])
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )
        self.assertEqual(
            storage_module._best_summary_config(self.conn, "TQQQ", "QQQ"),
            expected_best,
        )

    def test_max_drawdown_mutation_forces_rebuild_before_selection(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        expected_best = storage_module._best_summary_config(self.conn, "TQQQ", "QQQ")
        self.assertIsNotNone(expected_best)
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET max_drawdown = -0.5
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            expected_best,
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )
        self.assertIsNone(storage_module._best_summary_config(self.conn, "TQQQ", "QQQ"))

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data.iloc[-1:], rebuild=False)
        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0, 0, 0, 0])
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )

    def test_derived_summary_metric_storage_class_confusion_forces_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        self.conn.execute("UPDATE strategy_summary SET sharpe = CAST(sharpe AS BLOB) WHERE sharpe IS NOT NULL")
        self.assertEqual(
            self.conn.execute(
                "SELECT DISTINCT typeof(sharpe) FROM strategy_summary WHERE sharpe IS NOT NULL"
            ).fetchall(),
            [("blob",)],
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )

    def test_tiny_derived_summary_metric_mutation_forces_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        config = self.conn.execute(
            """
            SELECT buy_rsi, profit_target_multiple
            FROM strategy_summary
            WHERE sharpe IS NOT NULL
            ORDER BY buy_rsi, profit_target_multiple
            LIMIT 1
            """
        ).fetchone()
        self.assertIsNotNone(config)
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET sharpe = sharpe + 0.00000000001
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            config,
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )

    def test_nonnumeric_configuration_key_storage_classes_fail_closed(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        match_args = (
            self.conn,
            "TQQQ",
            "QQQ",
            self.cfg,
            [30.0, 70.0],
            [1.05, 1.50],
        )

        self.conn.execute("DROP TRIGGER leveraged_trader_strategy_state_identity_update_guard")
        self.conn.execute(
            """
            UPDATE strategy_state
            SET buy_rsi = CAST('broken' AS BLOB)
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )
        self.assertFalse(strategy_state_matches_config(*match_args))
        self.conn.execute("DROP TRIGGER leveraged_trader_strategy_summary_identity_update_guard")
        self.conn.execute(
            """
            UPDATE strategy_state
            SET buy_rsi = 30.0
            WHERE typeof(buy_rsi) = 'blob'
            """
        )

        self.conn.execute(
            """
            UPDATE strategy_summary
            SET buy_rsi = CAST('broken' AS BLOB)
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )
        self.assertFalse(strategy_state_matches_config(*match_args))

    def test_missing_summary_rollup_forces_full_saved_history_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        self.conn.execute("DELETE FROM strategy_summary")
        self.conn.execute("DELETE FROM strategy_equity")

        self.process_grid(data.iloc[-1:], rebuild=False)

        summaries = self.conn.execute(
            "SELECT trading_days, end_date FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        self.assertEqual(len(summaries), 4)
        self.assertEqual({row[0] for row in summaries}, {len(data)})
        self.assertEqual({row[1] for row in summaries}, {data.index[-1].date().isoformat()})

    def test_stale_state_last_date_forces_full_saved_history_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_states = self.conn.execute(
            "SELECT * FROM strategy_state ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        self.conn.execute(
            """
            UPDATE strategy_state
            SET last_date = ?
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """,
            (data.index[len(data) // 2].date().isoformat(),),
        )

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0, 0, 0, 0])
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_state ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_states,
        )
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )

    def test_nonuniform_coordinated_state_and_summary_chronology_forces_full_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_states = self.conn.execute(
            "SELECT * FROM strategy_state ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        stale_idx = len(data) // 2
        stale_date = data.index[stale_idx].date().isoformat()
        config_params = (30.0, 1.05)
        self.conn.execute(
            """
            UPDATE strategy_state
            SET last_date = ?
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            (stale_date, *config_params),
        )
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET end_date = ?,
                trading_days = ?,
                return_count = ?
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            (stale_date, stale_idx + 1, stale_idx, *config_params),
        )

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0, 0, 0, 0])
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_state ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_states,
        )
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )

    def test_summary_count_must_match_persisted_asset_session_window(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET trading_days = trading_days + 1,
                return_count = return_count + 1
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0, 0, 0, 0])
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )

    def test_nullable_null_state_last_date_forces_full_saved_history_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_states = self.conn.execute(
            "SELECT * FROM strategy_state ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        # Simulate a pre-existing/custom SQLite schema that predates or omits the
        # current NOT NULL constraint. CREATE TABLE IF NOT EXISTS does not repair
        # constraints on an existing table.
        self.conn.executescript(
            """
            ALTER TABLE strategy_state RENAME TO strict_strategy_state;
            CREATE TABLE strategy_state (
                asset_symbol TEXT NOT NULL,
                signal_symbol TEXT NOT NULL,
                buy_rsi REAL NOT NULL,
                profit_target_multiple REAL NOT NULL,
                start_date TEXT,
                last_date TEXT,
                cash REAL NOT NULL,
                shares REAL NOT NULL,
                in_position INTEGER NOT NULL,
                entry_price REAL,
                pending_action TEXT NOT NULL,
                prev_equity REAL NOT NULL,
                trades_executed INTEGER NOT NULL,
                PRIMARY KEY (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)
            );
            INSERT INTO strategy_state
            (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple,
             start_date, last_date, cash, shares, in_position, entry_price,
             pending_action, prev_equity, trades_executed)
            SELECT asset_symbol, signal_symbol, buy_rsi, profit_target_multiple,
                   start_date, last_date, cash, shares, in_position, entry_price,
                   pending_action, prev_equity, trades_executed
            FROM strict_strategy_state;
            DROP TABLE strict_strategy_state;
            UPDATE strategy_state
            SET last_date = NULL
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05;
            """
        )
        with self.assertRaisesRegex(ValueError, "strategy_state cannot be rebuilt"):
            init_state_db(self.conn)

        self.conn.execute(
            "UPDATE strategy_state SET last_date = ? WHERE last_date IS NULL",
            (data.index[-1].date().isoformat(),),
        )
        init_state_db(self.conn)

        self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_state ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_states,
        )
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )

    def test_noncanonical_mixed_timezone_chronology_forces_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        noncanonical_last_date = f"{data.index[-1].date().isoformat()}T00:00:00+00:00"
        self.conn.execute(
            "UPDATE strategy_state SET last_date = ?",
            (noncanonical_last_date,),
        )
        self.conn.execute(
            "UPDATE strategy_summary SET end_date = ?",
            (noncanonical_last_date,),
        )

        self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(
            {row[0] for row in self.conn.execute("SELECT DISTINCT last_date FROM strategy_state")},
            {data.index[-1].date().isoformat()},
        )
        self.assertEqual(
            {row[0] for row in self.conn.execute("SELECT DISTINCT trading_days FROM strategy_summary")},
            {len(data)},
        )

    def test_state_summary_trade_count_mismatch_forces_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_trade_counts = self.conn.execute(
            "SELECT buy_rsi, profit_target_multiple, trades_executed "
            "FROM strategy_state ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        self.conn.execute(
            """
            UPDATE strategy_state
            SET trades_executed = trades_executed + 2
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )

        self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(
            self.conn.execute(
                "SELECT buy_rsi, profit_target_multiple, trades_executed "
                "FROM strategy_state ORDER BY buy_rsi, profit_target_multiple"
            ).fetchall(),
            expected_trade_counts,
        )

    def test_impossible_persisted_trade_count_forces_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_trade_counts = self.conn.execute(
            "SELECT buy_rsi, profit_target_multiple, trades_executed "
            "FROM strategy_state ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        return_count = int(
            self.conn.execute(
                "SELECT return_count FROM strategy_summary WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05"
            ).fetchone()[0]
        )
        impossible_trade_count = 2 * return_count + 1
        self.conn.execute(
            "UPDATE strategy_state SET trades_executed = ? WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05",
            (impossible_trade_count,),
        )
        self.conn.execute(
            "UPDATE strategy_summary SET trades_executed = ? WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05",
            (impossible_trade_count,),
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )
        self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(
            self.conn.execute(
                "SELECT buy_rsi, profit_target_multiple, trades_executed "
                "FROM strategy_state ORDER BY buy_rsi, profit_target_multiple"
            ).fetchall(),
            expected_trade_counts,
        )

    def test_fractional_persisted_chronology_counts_force_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_state_trades = self.conn.execute(
            """
            SELECT trades_executed
            FROM strategy_state
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        ).fetchone()[0]
        expected_summary = self.conn.execute(
            """
            SELECT trading_days, return_count, trades_executed
            FROM strategy_summary
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        ).fetchone()
        self.conn.execute(
            """
            UPDATE strategy_state
            SET trades_executed = trades_executed + 0.5
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET trading_days = trading_days + 0.5,
                return_count = return_count + 0.5,
                trades_executed = trades_executed + 0.5
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        )

        self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertEqual(
            self.conn.execute(
                """
                SELECT trades_executed
                FROM strategy_state
                WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
                """
            ).fetchone()[0],
            expected_state_trades,
        )
        self.assertEqual(
            self.conn.execute(
                """
                SELECT trading_days, return_count, trades_executed
                FROM strategy_summary
                WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
                """
            ).fetchone(),
            expected_summary,
        )

    def test_market_arrays_rejects_rsi_with_no_finite_aligned_observation(self) -> None:
        data = sample_strategy_data(periods=4)
        future_rsi = pd.Series(
            [25.0],
            index=pd.to_datetime(["2027-01-04"]),
        )

        with self.assertRaisesRegex(AssetMarketDataError, "No finite RSI observations"):
            _market_arrays(data, future_rsi, "TQQQ")

    def test_market_arrays_allows_partial_rsi_alignment_gaps(self) -> None:
        data = sample_strategy_data(periods=4)
        partial_rsi = pd.Series(
            [25.0],
            index=pd.DatetimeIndex([data.index[2]]),
        )

        _open, _high, _close, rsi_values, _risk_free = _market_arrays(data, partial_rsi, "TQQQ")

        self.assertTrue(np.isnan(rsi_values[:2]).all())
        np.testing.assert_allclose(rsi_values[2:], [25.0, 25.0])

    def test_signal_alignment_preserves_exchange_proxy_dates_and_uses_weekend_observations(self) -> None:
        asset_dates = pd.to_datetime(["2026-01-08", "2026-01-09"])
        ordinary = pd.Series([40.0, 35.0], index=asset_dates)
        aligned_ordinary, ordinary_sources = align_signal_values_to_asset_sessions(ordinary, asset_dates)

        weekend = pd.Series(
            [40.0, 35.0, 20.0, 9.0],
            index=pd.to_datetime(["2026-01-08", "2026-01-09", "2026-01-10", "2026-01-11"]),
        )
        aligned_weekend, weekend_sources = align_signal_values_to_asset_sessions(weekend, asset_dates)

        self.assertEqual(aligned_ordinary.tolist(), [40.0, 35.0])
        self.assertEqual(ordinary_sources.dt.date.tolist(), [date.date() for date in asset_dates])
        self.assertEqual(aligned_weekend.tolist(), [40.0, 9.0])
        self.assertEqual(weekend_sources.iloc[-1], pd.Timestamp("2026-01-11"))

    def test_signal_alignment_does_not_pull_future_weekday_or_weekend_across_business_days(self) -> None:
        asset_dates = pd.to_datetime(["2026-01-07"])
        signals = pd.Series(
            [40.0, 30.0, 20.0, 10.0],
            index=pd.to_datetime(["2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10"]),
        )

        aligned, sources = align_signal_values_to_asset_sessions(signals, asset_dates)

        self.assertEqual(aligned.tolist(), [40.0])
        self.assertEqual(sources.iloc[0], pd.Timestamp("2026-01-07"))
        self.assertFalse(signal_observation_is_fresh("2026-01-07", "2026-01-08"))
        self.assertFalse(signal_observation_is_fresh("2026-01-07", "2026-01-10"))

    def test_signal_alignment_rejects_observations_more_than_seven_days_old(self) -> None:
        asset_dates = pd.to_datetime(["2026-07-01"])
        stale_signal = pd.Series([20.0], index=pd.to_datetime(["2026-06-23"]))

        aligned, sources = align_signal_values_to_asset_sessions(stale_signal, asset_dates)

        self.assertTrue(pd.isna(aligned.iloc[0]))
        self.assertTrue(pd.isna(sources.iloc[0]))
        self.assertFalse(signal_observation_is_fresh("2026-07-01", "2026-06-23"))
        self.assertTrue(signal_observation_is_fresh("2026-07-01", "2026-06-24"))

    def test_signal_alignment_rejects_stale_observation_at_later_execution_session(self) -> None:
        asset_dates = pd.to_datetime(["2026-01-01", "2026-01-20"])
        stale_signal = pd.Series([20.0], index=pd.to_datetime(["2026-01-02"]))

        aligned, sources = align_signal_values_to_asset_sessions(stale_signal, asset_dates)
        result = run_single_equity_curve(
            open_prices=np.array([100.0, 100.0]),
            high_prices=np.array([100.0, 100.0]),
            close_prices=np.array([100.0, 100.0]),
            rsi_values=aligned.to_numpy(dtype=np.float64),
            risk_free_returns=np.zeros(2),
            buy_rsi=30.0,
            profit_target_multiple=2.0,
            initial_capital=100_000.0,
            trading_cost_rate=0.0,
        )

        self.assertTrue(aligned.isna().all())
        self.assertTrue(sources.isna().all())
        self.assertEqual(result[4].tolist(), [0, 0])
        self.assertEqual(result[6].tolist(), [0, 0])

    def test_signal_alignment_accepts_exact_execution_lag_boundary_and_sunday_close(self) -> None:
        long_gap_asset_dates = pd.to_datetime(["2026-01-01", "2026-01-20"])
        boundary_signal = pd.Series([20.0], index=pd.to_datetime(["2026-01-13"]))

        boundary_aligned, boundary_sources = align_signal_values_to_asset_sessions(
            boundary_signal,
            long_gap_asset_dates,
        )

        self.assertEqual(boundary_aligned.tolist(), [20.0, 20.0])
        self.assertEqual(boundary_sources.tolist(), [pd.Timestamp("2026-01-13")] * 2)
        self.assertTrue(signal_observation_is_fresh("2026-01-20", "2026-01-13"))

        weekend_asset_dates = pd.to_datetime(["2026-01-09", "2026-01-12"])
        sunday_signal = pd.Series([15.0], index=pd.to_datetime(["2026-01-11"]))

        weekend_aligned, weekend_sources = align_signal_values_to_asset_sessions(
            sunday_signal,
            weekend_asset_dates,
        )

        self.assertEqual(weekend_aligned.tolist(), [15.0, 15.0])
        self.assertEqual(weekend_sources.tolist(), [pd.Timestamp("2026-01-11")] * 2)

    def test_weekend_signal_tail_rebuilds_friday_pending_action_for_monday_open(self) -> None:
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        asset_close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
        signal_close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)

        def frame(symbol: str, close: pd.Series, volume: float) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    f"{symbol}_Open": close,
                    f"{symbol}_High": close + 1.0,
                    f"{symbol}_Low": close - 1.0,
                    f"{symbol}_Close": close,
                    f"{symbol}_Volume": volume,
                },
                index=close.index,
            )

        asset_history = frame("TQQQ", asset_close, 1_000_000.0)
        signal_history = frame("QQQ", signal_close, 2_000_000.0)
        risk_free_history = frame(RISK_FREE_SYMBOL, pd.Series(5.0, index=dates), 0.0)
        data = pd.concat([asset_history, signal_history, risk_free_history], axis=1)
        cfg = BacktestConfig(rsi_period=2)
        histories = {
            "TQQQ": asset_history,
            "QQQ": signal_history,
            RISK_FREE_SYMBOL: risk_free_history,
        }
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=True,
            authoritative_histories=histories,
        )
        self.assertEqual(
            self.conn.execute("SELECT pending_action FROM strategy_state").fetchone()[0],
            "none",
        )

        weekend_close = pd.Series(
            [50.0, 25.0],
            index=pd.to_datetime(["2026-01-10", "2026-01-11"]),
        )
        extended_signal = pd.concat([signal_history, frame("QQQ", weekend_close, 2_000_000.0)])
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=False,
            authoritative_histories={**histories, "QQQ": extended_signal},
        )

        last_date, pending_action = self.conn.execute("SELECT last_date, pending_action FROM strategy_state").fetchone()
        latest_rsi_date = self.conn.execute(
            "SELECT date FROM rsi_values WHERE signal_symbol = 'QQQ' ORDER BY date DESC LIMIT 1"
        ).fetchone()[0]
        self.assertEqual(last_date, "2026-01-09")
        self.assertEqual(latest_rsi_date, "2026-01-11")
        self.assertEqual(pending_action, "buy")
        equity_last_date, equity_pending_action = self.conn.execute(
            "SELECT date, pending_action FROM strategy_equity ORDER BY date DESC LIMIT 1"
        ).fetchone()
        summary_end_date, summary_trading_days = self.conn.execute(
            "SELECT end_date, trading_days FROM strategy_summary"
        ).fetchone()
        self.assertEqual((equity_last_date, equity_pending_action), (last_date, pending_action))
        self.assertEqual((summary_end_date, summary_trading_days), (last_date, len(dates)))

    def test_late_signal_tail_on_processed_asset_session_rebuilds_state(self) -> None:
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        asset_close = pd.Series(100.0, index=dates)
        signal_close = pd.Series([100.0, 101.0, 102.0, 103.0, 50.0], index=dates)

        def frame(symbol: str, close: pd.Series, volume: float) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    f"{symbol}_Open": close,
                    f"{symbol}_High": close + 1.0,
                    f"{symbol}_Low": close - 1.0,
                    f"{symbol}_Close": close,
                    f"{symbol}_Volume": volume,
                },
                index=close.index,
            )

        asset_history = frame("TQQQ", asset_close, 1_000_000.0)
        signal_history = frame("QQQ", signal_close, 2_000_000.0)
        risk_free_history = frame(RISK_FREE_SYMBOL, pd.Series(5.0, index=dates), 0.0)
        data = pd.concat([asset_history, signal_history, risk_free_history], axis=1)
        cfg = BacktestConfig(rsi_period=2)
        lagging_histories = {
            "TQQQ": asset_history,
            "QQQ": signal_history.iloc[:-1],
            RISK_FREE_SYMBOL: risk_free_history,
        }
        complete_histories = {
            "TQQQ": asset_history,
            "QQQ": signal_history,
            RISK_FREE_SYMBOL: risk_free_history,
        }

        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=True,
            authoritative_histories=lagging_histories,
        )
        self.assertEqual(
            self.conn.execute("SELECT pending_action FROM strategy_state").fetchone()[0],
            "none",
        )

        with patch(
            "leveraged_trader.storage.run_grid_summary",
            wraps=run_grid_summary,
        ) as grid_summary:
            process_asset_grid(
                self.conn,
                data,
                cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [2.0],
                rebuild=False,
                authoritative_histories=complete_histories,
            )

        last_date, pending_action = self.conn.execute("SELECT last_date, pending_action FROM strategy_state").fetchone()
        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0])
        self.assertEqual((last_date, pending_action), ("2026-01-09", "buy"))
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0],
            len(dates),
        )

    def test_signal_tail_inside_processed_range_rebuilds_with_final_alignment_unchanged(self) -> None:
        asset_dates = pd.date_range("2026-01-01", "2026-01-20", freq="B")
        initial_signal_dates = asset_dates[asset_dates <= pd.Timestamp("2026-01-05")]
        appended_signal_date = pd.Timestamp("2026-01-06")

        def frame(symbol: str, index: pd.DatetimeIndex, values: list[float]) -> pd.DataFrame:
            close = pd.Series(values, index=index, dtype=float)
            return pd.DataFrame(
                {
                    f"{symbol}_Open": close,
                    f"{symbol}_High": close + 1.0,
                    f"{symbol}_Low": close - 1.0,
                    f"{symbol}_Close": close,
                    f"{symbol}_Volume": 1_000_000.0,
                },
                index=index,
            )

        asset_history = frame("TQQQ", asset_dates, [100.0] * len(asset_dates))
        initial_signal = frame("QQQ", initial_signal_dates, [100.0, 101.0, 102.0])
        appended_signal = frame("QQQ", pd.DatetimeIndex([appended_signal_date]), [50.0])
        extended_signal = pd.concat([initial_signal, appended_signal])
        risk_free_history = frame(RISK_FREE_SYMBOL, asset_dates, [5.0] * len(asset_dates))
        cfg = BacktestConfig(rsi_period=2)

        process_asset_grid(
            self.conn,
            pd.concat([asset_history, initial_signal, risk_free_history], axis=1),
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories={
                "TQQQ": asset_history,
                "QQQ": initial_signal,
                RISK_FREE_SYMBOL: risk_free_history,
            },
        )

        expected_conn = sqlite3.connect(":memory:")
        init_state_db(expected_conn)
        try:
            with patch(
                "leveraged_trader.storage.run_grid_summary",
                wraps=run_grid_summary,
            ) as grid_summary:
                process_asset_grid(
                    self.conn,
                    pd.concat([asset_history, extended_signal, risk_free_history], axis=1),
                    cfg,
                    "TQQQ",
                    "QQQ",
                    [30.0],
                    [1.5],
                    rebuild=False,
                    authoritative_histories={
                        "TQQQ": asset_history,
                        "QQQ": extended_signal,
                        RISK_FREE_SYMBOL: risk_free_history,
                    },
                )
            process_asset_grid(
                expected_conn,
                pd.concat([asset_history, extended_signal, risk_free_history], axis=1),
                cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories={
                    "TQQQ": asset_history,
                    "QQQ": extended_signal,
                    RISK_FREE_SYMBOL: risk_free_history,
                },
            )

            self.assertEqual(grid_summary.call_args.args[7].tolist(), [0])
            for table in ("strategy_state", "strategy_equity", "strategy_summary"):
                with self.subTest(table=table):
                    actual = self.conn.execute(
                        f"SELECT * FROM {table} WHERE asset_symbol = 'TQQQ' ORDER BY rowid"
                    ).fetchall()
                    expected = expected_conn.execute(
                        f"SELECT * FROM {table} WHERE asset_symbol = 'TQQQ' ORDER BY rowid"
                    ).fetchall()
                    self.assertEqual(actual, expected)
        finally:
            expected_conn.close()

    def test_matching_asset_and_signal_tail_session_resumes_incrementally(self) -> None:
        initial = sample_strategy_data(periods=8)
        appended = initial.iloc[[-1]].copy()
        appended.index = pd.DatetimeIndex([initial.index[-1] + pd.offsets.BDay()])
        extended = pd.concat([initial, appended])

        process_asset_grid(
            self.conn,
            initial,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(initial),
        )

        with patch(
            "leveraged_trader.storage.run_grid_summary",
            wraps=run_grid_summary,
        ) as grid_summary:
            process_asset_grid(
                self.conn,
                extended,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=False,
                authoritative_histories=self.canonical_histories(extended),
            )

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [len(initial)])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0],
            len(extended),
        )

    def test_saved_unprocessed_market_tail_is_replayed_before_new_input(self) -> None:
        data = sample_strategy_data(periods=6)
        data.index = pd.date_range("2026-01-02", periods=len(data), freq="W-FRI")
        cfg = BacktestConfig(rsi_period=2)
        buy_rsi_values = [30.0, 70.0]
        profit_target_values = [1.5]

        process_asset_grid(
            self.conn,
            data.iloc[:3],
            cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values,
            profit_target_values,
            rebuild=True,
        )
        # Model a crash after market persistence but before compact strategy
        # state advances over these two sessions.
        storage_module.save_market_data(
            self.conn,
            data.iloc[3:5],
            ["TQQQ", "QQQ", RISK_FREE_SYMBOL],
        )

        with patch(
            "leveraged_trader.storage.run_grid_summary",
            wraps=run_grid_summary,
        ) as grid_summary:
            process_asset_grid(
                self.conn,
                data.iloc[5:],
                cfg,
                "TQQQ",
                "QQQ",
                buy_rsi_values,
                profit_target_values,
                rebuild=False,
            )

        expected_conn = sqlite3.connect(":memory:")
        init_state_db(expected_conn)
        try:
            process_asset_grid(
                expected_conn,
                data,
                cfg,
                "TQQQ",
                "QQQ",
                buy_rsi_values,
                profit_target_values,
                rebuild=True,
            )

            self.assertEqual(grid_summary.call_args.args[7].tolist(), [3, 3])
            self.assertTrue(
                strategy_state_matches_config(
                    self.conn,
                    "TQQQ",
                    "QQQ",
                    cfg,
                    buy_rsi_values,
                    profit_target_values,
                )
            )
            for table in ("strategy_state", "strategy_equity", "strategy_summary"):
                with self.subTest(table=table):
                    actual = self.conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                    expected = expected_conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                    self.assertEqual(actual, expected)
        finally:
            expected_conn.close()

    def test_new_asset_session_resumes_after_previously_saved_weekend_signal(self) -> None:
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        weekend_observation = pd.Timestamp("2026-01-11")
        next_session = pd.Timestamp("2026-01-12")

        def frame(symbol: str, index: pd.DatetimeIndex, start: float) -> pd.DataFrame:
            close = pd.Series(np.arange(len(index), dtype=float) + start, index=index)
            return pd.DataFrame(
                {
                    f"{symbol}_Open": close,
                    f"{symbol}_High": close + 1.0,
                    f"{symbol}_Low": close - 1.0,
                    f"{symbol}_Close": close,
                    f"{symbol}_Volume": 1_000_000.0,
                },
                index=index,
            )

        signal_dates = dates.append(pd.DatetimeIndex([weekend_observation]))
        signal_history = frame("QQQ", signal_dates, 100.0)
        risk_free_history = frame(RISK_FREE_SYMBOL, signal_dates, 5.0)
        tqqq_history = frame("TQQQ", dates, 100.0)
        upro_history = frame("UPRO", dates, 50.0)

        for asset_symbol, asset_history in (("TQQQ", tqqq_history), ("UPRO", upro_history)):
            process_asset_grid(
                self.conn,
                pd.concat([asset_history, signal_history, risk_free_history], axis=1),
                self.cfg,
                asset_symbol,
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories={
                    asset_symbol: asset_history,
                    "QQQ": signal_history,
                    RISK_FREE_SYMBOL: risk_free_history,
                },
            )

        extended_asset_dates = dates.append(pd.DatetimeIndex([next_session]))
        extended_tqqq = frame("TQQQ", extended_asset_dates, 100.0)
        with patch(
            "leveraged_trader.storage.run_grid_summary",
            wraps=run_grid_summary,
        ) as grid_summary:
            process_asset_grid(
                self.conn,
                pd.concat(
                    [extended_tqqq, signal_history, risk_free_history],
                    axis=1,
                    sort=False,
                ),
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=False,
                authoritative_histories={
                    "TQQQ": extended_tqqq,
                    "QQQ": signal_history,
                    RISK_FREE_SYMBOL: risk_free_history,
                },
                presynchronized_authoritative_symbols={"QQQ", RISK_FREE_SYMBOL},
            )

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [len(dates)])
        self.assertEqual(
            self.conn.execute(
                "SELECT asset_symbol FROM strategy_state WHERE signal_symbol = 'QQQ' ORDER BY asset_symbol"
            ).fetchall(),
            [("TQQQ",), ("UPRO",)],
        )

    def test_new_asset_processed_first_invalidates_existing_signal_dependent(self) -> None:
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        lead_date = pd.Timestamp("2026-01-11")

        def frame(symbol: str, index: pd.DatetimeIndex, values: list[float]) -> pd.DataFrame:
            close = pd.Series(values, index=index, dtype=float)
            return pd.DataFrame(
                {
                    f"{symbol}_Open": close,
                    f"{symbol}_High": close + 1.0,
                    f"{symbol}_Low": close - 1.0,
                    f"{symbol}_Close": close,
                    f"{symbol}_Volume": 1_000_000.0,
                },
                index=index,
            )

        tqqq_history = frame("TQQQ", dates, [100.0] * len(dates))
        upro_history = frame("UPRO", dates, [50.0] * len(dates))
        signal_history = frame("QQQ", dates, [100.0, 101.0, 102.0, 103.0, 104.0])
        risk_free_history = frame(RISK_FREE_SYMBOL, dates, [5.0] * len(dates))
        process_asset_grid(
            self.conn,
            pd.concat([tqqq_history, signal_history, risk_free_history], axis=1),
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories={
                "TQQQ": tqqq_history,
                "QQQ": signal_history,
                RISK_FREE_SYMBOL: risk_free_history,
            },
        )

        extended_dates = dates.append(pd.DatetimeIndex([lead_date]))
        extended_signal = frame("QQQ", extended_dates, [100.0, 101.0, 102.0, 103.0, 104.0, 50.0])
        extended_risk_free = frame(RISK_FREE_SYMBOL, extended_dates, [5.0] * len(extended_dates))
        process_asset_grid(
            self.conn,
            pd.concat([upro_history, extended_signal, extended_risk_free], axis=1),
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories={
                "UPRO": upro_history,
                "QQQ": extended_signal,
                RISK_FREE_SYMBOL: extended_risk_free,
            },
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT asset_symbol FROM strategy_state WHERE signal_symbol = 'QQQ' ORDER BY asset_symbol"
            ).fetchall(),
            [("UPRO",)],
        )

        expected_conn = sqlite3.connect(":memory:")
        init_state_db(expected_conn)
        try:
            process_asset_grid(
                self.conn,
                pd.concat([tqqq_history, extended_signal, extended_risk_free], axis=1),
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=False,
                authoritative_histories={
                    "TQQQ": tqqq_history,
                    "QQQ": extended_signal,
                    RISK_FREE_SYMBOL: extended_risk_free,
                },
                presynchronized_authoritative_symbols={"QQQ", RISK_FREE_SYMBOL},
            )
            process_asset_grid(
                expected_conn,
                pd.concat([tqqq_history, extended_signal, extended_risk_free], axis=1),
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories={
                    "TQQQ": tqqq_history,
                    "QQQ": extended_signal,
                    RISK_FREE_SYMBOL: extended_risk_free,
                },
            )

            for table in ("strategy_state", "strategy_equity", "strategy_summary"):
                with self.subTest(table=table):
                    actual = self.conn.execute(
                        f"SELECT * FROM {table} WHERE asset_symbol = 'TQQQ' ORDER BY rowid"
                    ).fetchall()
                    expected = expected_conn.execute(
                        f"SELECT * FROM {table} WHERE asset_symbol = 'TQQQ' ORDER BY rowid"
                    ).fetchall()
                    self.assertEqual(actual, expected)
        finally:
            expected_conn.close()

    def test_non_authoritative_signal_mutation_checks_all_saved_dependents(self) -> None:
        initial = sample_strategy_data(periods=6)
        upro_data = initial.rename(columns=lambda column: column.replace("TQQQ_", "UPRO_"))
        for asset_symbol, data in (("TQQQ", initial), ("UPRO", upro_data)):
            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                asset_symbol,
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories=self.canonical_histories(data, asset_symbol),
            )

        sqqq_data = initial.rename(columns=lambda column: column.replace("TQQQ_", "SQQQ_"))
        signal_history = self.canonical_histories(initial)["QQQ"]
        lead_signal = signal_history.iloc[[-1]].copy()
        lead_signal.index = pd.DatetimeIndex([initial.index[-1] + pd.Timedelta(days=1)])
        extended_signal = pd.concat([signal_history, lead_signal])
        process_asset_grid(
            self.conn,
            sqqq_data,
            self.cfg,
            "SQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            signal_history=extended_signal,
        )

        self.assertEqual(
            self.conn.execute(
                "SELECT asset_symbol FROM strategy_state WHERE signal_symbol = 'QQQ' ORDER BY asset_symbol"
            ).fetchall(),
            [("SQQQ",)],
        )

    def test_non_authoritative_rebuild_detects_shared_signal_value_revision(self) -> None:
        initial = sample_strategy_data(periods=8)
        upro_data = initial.rename(columns=lambda column: column.replace("TQQQ_", "UPRO_"))
        for asset_symbol, data in (("TQQQ", initial), ("UPRO", upro_data)):
            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                asset_symbol,
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories=self.canonical_histories(data, asset_symbol),
            )

        corrected = initial.rename(columns=lambda column: column.replace("TQQQ_", "SQQQ_"))
        corrected_date = corrected.index[3]
        corrected.loc[corrected_date, "QQQ_Close"] += 10.0
        corrected.loc[corrected_date, "QQQ_High"] += 10.0
        process_asset_grid(
            self.conn,
            corrected,
            self.cfg,
            "SQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
        )

        self.assertEqual(
            self.conn.execute(
                "SELECT asset_symbol FROM strategy_state WHERE signal_symbol = 'QQQ' ORDER BY asset_symbol"
            ).fetchall(),
            [("SQQQ",)],
        )

    def test_non_authoritative_matching_tail_append_remains_incremental(self) -> None:
        initial = sample_strategy_data(periods=8)
        appended = initial.iloc[[-1]].copy()
        appended.index = pd.DatetimeIndex([initial.index[-1] + pd.offsets.BDay()])
        extended = pd.concat([initial, appended])

        process_asset_grid(
            self.conn,
            initial,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
        )
        with patch(
            "leveraged_trader.storage.run_grid_summary",
            wraps=run_grid_summary,
        ) as grid_summary:
            process_asset_grid(
                self.conn,
                extended,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=False,
                signal_history=self.canonical_histories(extended)["QQQ"],
            )

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [len(initial)])

    def test_non_authoritative_late_signal_tail_rebuilds_processed_state(self) -> None:
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        asset_close = pd.Series(100.0, index=dates)
        signal_close = pd.Series([100.0, 101.0, 102.0, 103.0, 50.0], index=dates)

        def frame(symbol: str, close: pd.Series, volume: float) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    f"{symbol}_Open": close,
                    f"{symbol}_High": close + 1.0,
                    f"{symbol}_Low": close - 1.0,
                    f"{symbol}_Close": close,
                    f"{symbol}_Volume": volume,
                },
                index=close.index,
            )

        asset_history = frame("TQQQ", asset_close, 1_000_000.0)
        signal_history = frame("QQQ", signal_close, 2_000_000.0)
        risk_free_history = frame(RISK_FREE_SYMBOL, pd.Series(5.0, index=dates), 0.0)
        lagging_signal_history = signal_history.iloc[:-1]
        lagging_data = pd.concat(
            [asset_history, lagging_signal_history, risk_free_history],
            axis=1,
        )
        cfg = BacktestConfig(rsi_period=2)

        process_asset_grid(
            self.conn,
            lagging_data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=True,
            signal_history=lagging_signal_history,
        )
        self.assertEqual(
            self.conn.execute("SELECT pending_action FROM strategy_state").fetchone()[0],
            "none",
        )

        process_asset_grid(
            self.conn,
            lagging_data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=False,
            signal_history=signal_history,
        )

        self.assertEqual(
            self.conn.execute("SELECT last_date, pending_action FROM strategy_state").fetchone(),
            ("2026-01-09", "buy"),
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0],
            len(dates),
        )

    def test_non_authoritative_late_risk_free_bar_invalidates_global_state(self) -> None:
        data = sample_strategy_data(periods=8)
        data.loc[
            data.index[-1],
            [
                f"{RISK_FREE_SYMBOL}_Open",
                f"{RISK_FREE_SYMBOL}_High",
                f"{RISK_FREE_SYMBOL}_Low",
                f"{RISK_FREE_SYMBOL}_Close",
            ],
        ] = 50.0
        lagging = data.copy()
        risk_free_columns = [column for column in lagging if column.startswith(f"{RISK_FREE_SYMBOL}_")]
        lagging.loc[lagging.index[-1], risk_free_columns] = np.nan
        upro_lagging = lagging.rename(columns=lambda column: column.replace("TQQQ_", "UPRO_"))

        for asset_symbol, history in (("TQQQ", lagging), ("UPRO", upro_lagging)):
            process_asset_grid(
                self.conn,
                history,
                self.cfg,
                asset_symbol,
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
            )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(DISTINCT asset_symbol) FROM strategy_state").fetchone()[0],
            2,
        )

        with patch(
            "leveraged_trader.storage.run_grid_summary",
            wraps=run_grid_summary,
        ) as grid_summary:
            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=False,
            )

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0])
        self.assertEqual(
            self.conn.execute("SELECT DISTINCT asset_symbol FROM strategy_state").fetchall(),
            [("TQQQ",)],
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT close FROM market_data WHERE symbol = ? AND date = ?",
                (RISK_FREE_SYMBOL, data.index[-1].date().isoformat()),
            ).fetchone()[0],
            50.0,
        )

    def test_negative_risk_free_yield_is_valid_and_produces_finite_daily_returns(self) -> None:
        data = sample_strategy_data()
        for field in ("Open", "High", "Low", "Close"):
            data[f"{RISK_FREE_SYMBOL}_{field}"] = -0.105

        self.process_grid(data, rebuild=True)

        risk_free_returns = np.asarray(
            [
                row[0]
                for row in self.conn.execute("SELECT risk_free_return FROM strategy_equity ORDER BY date").fetchall()
            ],
            dtype=float,
        )
        self.assertTrue(np.isfinite(risk_free_returns).all())
        self.assertTrue((risk_free_returns < 0.0).all())

    def test_risk_free_yield_outside_compounding_domain_is_rejected(self) -> None:
        data = sample_strategy_data()
        for field in ("Open", "High", "Low", "Close"):
            data[f"{RISK_FREE_SYMBOL}_{field}"] = -100.0

        with self.assertRaisesRegex(ValueError, "greater than -100 and finite"):
            self.process_grid(data, rebuild=True)

    def test_authoritative_history_identical_sync_reads_once_without_writes(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]

        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        changes_before = self.conn.total_changes
        statements: list[str] = []
        self.conn.set_trace_callback(statements.append)
        try:
            revised = _synchronize_market_data_history(self.conn, history, "TQQQ")
        finally:
            self.conn.set_trace_callback(None)

        market_statements = [statement.upper() for statement in statements if "MARKET_DATA" in statement.upper()]
        self.assertFalse(revised)
        self.assertEqual(self.conn.total_changes, changes_before)
        self.assertEqual(
            sum(statement.lstrip().startswith("SELECT") for statement in market_statements),
            1,
        )
        self.assertFalse(
            any(statement.lstrip().startswith(("INSERT", "UPDATE", "DELETE")) for statement in market_statements)
        )

    def test_authoritative_history_tail_append_writes_only_new_session(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        self.assertFalse(_synchronize_market_data_history(self.conn, history.iloc[:-1], "TQQQ"))
        changes_before = self.conn.total_changes

        revised = _synchronize_market_data_history(self.conn, history, "TQQQ")

        self.assertFalse(revised)
        self.assertEqual(self.conn.total_changes - changes_before, 1)

    def test_late_risk_free_tail_bar_replays_forward_filled_strategy_state(self) -> None:
        data = sample_strategy_data()
        data.loc[
            data.index[-1],
            [
                f"{RISK_FREE_SYMBOL}_Open",
                f"{RISK_FREE_SYMBOL}_High",
                f"{RISK_FREE_SYMBOL}_Low",
                f"{RISK_FREE_SYMBOL}_Close",
            ],
        ] = 50.0
        lagging_histories = self.canonical_histories(data)
        lagging_histories[RISK_FREE_SYMBOL] = lagging_histories[RISK_FREE_SYMBOL].iloc[:-1]
        expected_conn = sqlite3.connect(":memory:")
        init_state_db(expected_conn)
        try:
            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories=lagging_histories,
            )
            stale_risk_free_return = self.conn.execute(
                "SELECT risk_free_return FROM strategy_equity ORDER BY date DESC LIMIT 1"
            ).fetchone()[0]

            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=False,
                authoritative_histories=self.canonical_histories(data),
            )
            process_asset_grid(
                expected_conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories=self.canonical_histories(data),
            )

            refreshed_risk_free_return = self.conn.execute(
                "SELECT risk_free_return FROM strategy_equity ORDER BY date DESC LIMIT 1"
            ).fetchone()[0]
            expected_risk_free_return = expected_conn.execute(
                "SELECT risk_free_return FROM strategy_equity ORDER BY date DESC LIMIT 1"
            ).fetchone()[0]
            self.assertNotEqual(stale_risk_free_return, refreshed_risk_free_return)
            self.assertAlmostEqual(refreshed_risk_free_return, expected_risk_free_return)

            comparison_columns = (
                "start_date, end_date, trading_days, trades_executed, total_return, cagr, "
                "annualized_vol, sharpe, kelly_fraction, max_drawdown, hit_rate, "
                "first_equity, last_equity, running_max_equity, return_count, return_sum, "
                "return_sum_squares, excess_return_count, excess_return_sum, "
                "excess_return_sum_squares, positive_return_count"
            )
            actual_summary = self.conn.execute(f"SELECT {comparison_columns} FROM strategy_summary").fetchone()
            expected_summary = expected_conn.execute(f"SELECT {comparison_columns} FROM strategy_summary").fetchone()
            self.assertEqual(actual_summary[:4], expected_summary[:4])
            for actual, expected in zip(actual_summary[4:], expected_summary[4:], strict=True):
                if actual is None or expected is None:
                    self.assertIs(actual, expected)
                else:
                    self.assertAlmostEqual(actual, expected)
        finally:
            expected_conn.close()

    def test_authoritative_history_correction_updates_only_changed_session(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        revision = history.copy()
        revision.loc[revision.index[10], "TQQQ_Close"] = 123.45
        revision.loc[revision.index[10], "TQQQ_High"] = 124.0
        changes_before = self.conn.total_changes

        revised = _synchronize_market_data_history(self.conn, revision, "TQQQ")

        self.assertTrue(revised)
        self.assertEqual(self.conn.total_changes - changes_before, 1)

    def test_authoritative_history_detects_exact_price_corrections_and_tolerates_volume_noise(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        revision = history.copy()
        corrected_date = revision.index[10]
        revision.loc[corrected_date, "TQQQ_Close"] = np.nextafter(
            revision.loc[corrected_date, "TQQQ_Close"],
            np.inf,
        )
        changes_before = self.conn.total_changes

        self.assertTrue(_synchronize_market_data_history(self.conn, revision, "TQQQ"))
        self.assertEqual(self.conn.total_changes - changes_before, 1)
        self.assertEqual(
            self.conn.execute(
                "SELECT close FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (corrected_date.date().isoformat(),),
            ).fetchone()[0],
            revision.loc[corrected_date, "TQQQ_Close"],
        )

        changes_before = self.conn.total_changes
        revision["TQQQ_Volume"] = revision["TQQQ_Volume"].astype(float)
        revision.loc[corrected_date, "TQQQ_Volume"] += 5e-7
        self.assertFalse(_synchronize_market_data_history(self.conn, revision, "TQQQ"))
        self.assertEqual(self.conn.total_changes, changes_before)

        revision.loc[corrected_date, "TQQQ_Volume"] += 1e-4
        self.assertTrue(_synchronize_market_data_history(self.conn, revision, "TQQQ"))
        self.assertEqual(self.conn.total_changes - changes_before, 1)

    def test_authoritative_yahoo_history_persists_adjacent_float32_corrections(self) -> None:
        captured_variants = {
            "TQQQ": (38.83776092529297, 38.837764739990234),
            "SQQQ": (143.35150146484375, 143.3514862060547),
            "AAPL": (242.0931396484375, 242.09312438964844),
        }
        date = pd.Timestamp("2026-01-02")
        for symbol, (first_close, second_close) in captured_variants.items():
            with self.subTest(symbol=symbol):
                history = pd.DataFrame(
                    {
                        f"{symbol}_Open": [first_close],
                        f"{symbol}_High": [max(first_close, second_close) + 1.0],
                        f"{symbol}_Low": [min(first_close, second_close) - 1.0],
                        f"{symbol}_Close": [first_close],
                        f"{symbol}_Volume": [1_000.0],
                    },
                    index=[date],
                )
                history.attrs["market_data_providers"] = {symbol: "yahoo_finance"}
                self.assertFalse(_synchronize_market_data_history(self.conn, history, symbol))
                changes_before = self.conn.total_changes

                observed_again = history.copy()
                observed_again.loc[date, f"{symbol}_Close"] = second_close
                self.assertTrue(_synchronize_market_data_history(self.conn, observed_again, symbol))
                self.assertEqual(self.conn.total_changes - changes_before, 1)
                self.assertEqual(
                    self.conn.execute(
                        "SELECT close FROM market_data WHERE symbol = ? AND date = ?",
                        (symbol, date.date().isoformat()),
                    ).fetchone()[0],
                    second_close,
                )

                material_revision = history.copy()
                material_revision.loc[date, f"{symbol}_Close"] = first_close + 0.01
                self.assertTrue(_synchronize_market_data_history(self.conn, material_revision, symbol))

    def test_yahoo_float32_correction_rebuilds_completed_strategy_state(self) -> None:
        data = sample_strategy_data()
        revised_date = data.index[10]
        first_close = 38.83776092529297
        second_close = 38.837764739990234
        data.loc[
            revised_date,
            ["TQQQ_Open", "TQQQ_High", "TQQQ_Low", "TQQQ_Close"],
        ] = [first_close, first_close + 1.0, first_close - 1.0, first_close]

        def yahoo_histories(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
            histories = self.canonical_histories(frame)
            for symbol, history in histories.items():
                history.attrs["market_data_providers"] = {symbol: "yahoo_finance"}
            return histories

        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=yahoo_histories(data),
        )
        observed_again = data.copy()
        observed_again.loc[revised_date, "TQQQ_Close"] = second_close
        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            process_asset_grid(
                self.conn,
                observed_again,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=False,
                authoritative_histories=yahoo_histories(observed_again),
            )

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0])
        self.assertEqual(
            self.conn.execute(
                "SELECT close FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (revised_date.date().isoformat(),),
            ).fetchone()[0],
            second_close,
        )

    def test_legacy_revision_detection_is_exact_for_every_provider(self) -> None:
        date = pd.Timestamp("2026-01-02")
        first_close = 38.83776092529297
        second_close = 38.837764739990234
        original = pd.DataFrame(
            {
                "TQQQ_Open": [first_close],
                "TQQQ_High": [first_close + 1.0],
                "TQQQ_Low": [first_close - 1.0],
                "TQQQ_Close": [first_close],
                "TQQQ_Volume": [1_000.0],
            },
            index=[date],
        )
        save_market_data(self.conn, original, ["TQQQ"])
        observed_again = original.copy()
        observed_again.loc[date, "TQQQ_Close"] = second_close
        observed_again.attrs["market_data_providers"] = {"TQQQ": "yahoo_finance"}

        self.assertEqual(
            storage_module._revised_market_symbols(self.conn, observed_again, ["TQQQ"]),
            {"TQQQ"},
        )

        observed_again.attrs["market_data_providers"] = {"TQQQ": "other"}
        self.assertEqual(
            storage_module._revised_market_symbols(self.conn, observed_again, ["TQQQ"]),
            {"TQQQ"},
        )

    def test_yahoo_high_correction_crossing_target_is_persisted_and_replayed(self) -> None:
        date = pd.Timestamp("2026-01-02")
        below_target = float(np.nextafter(np.float32(100.0), -np.inf))
        history = pd.DataFrame(
            {
                "TQQQ_Open": [90.0],
                "TQQQ_High": [below_target],
                "TQQQ_Low": [85.0],
                "TQQQ_Close": [95.0],
                "TQQQ_Volume": [1_000.0],
            },
            index=[date],
        )
        history.attrs["market_data_providers"] = {"TQQQ": "yahoo_finance"}
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))

        corrected = history.copy()
        corrected.loc[date, "TQQQ_High"] = 100.0
        self.assertTrue(_synchronize_market_data_history(self.conn, corrected, "TQQQ"))
        self.assertEqual(
            self.conn.execute(
                "SELECT high FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (date.date().isoformat(),),
            ).fetchone()[0],
            100.0,
        )

        def simulation(high: float) -> tuple:
            return run_single_equity_curve(
                np.array([70.0, 80.0, 90.0]),
                np.array([70.0, 80.0, high]),
                np.array([70.0, 80.0, 90.0]),
                np.array([20.0, 50.0, 50.0]),
                np.zeros(3),
                30.0,
                1.25,
                1_000.0,
                0.0,
            )

        self.assertEqual(simulation(below_target)[9], 1)
        corrected_result = simulation(100.0)
        self.assertEqual(corrected_result[9], 0)
        self.assertEqual(corrected_result[7], 1_250.0)

    def test_legacy_yahoo_float32_correction_rebuilds_and_replaces_candle(self) -> None:
        data = sample_strategy_data()
        revised_date = data.index[10]
        first_close = 38.83776092529297
        second_close = 38.837764739990234
        data.loc[
            revised_date,
            ["TQQQ_Open", "TQQQ_High", "TQQQ_Low", "TQQQ_Close"],
        ] = [first_close, first_close + 1.0, first_close - 1.0, first_close]
        data.attrs["market_data_providers"] = {"TQQQ": "yahoo_finance"}
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
        )
        observed_again = data.copy()
        observed_again.loc[revised_date, "TQQQ_Close"] = second_close
        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            process_asset_grid(
                self.conn,
                observed_again,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=False,
            )

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0])
        self.assertEqual(
            self.conn.execute(
                "SELECT close FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (revised_date.date().isoformat(),),
            ).fetchone()[0],
            second_close,
        )

        process_asset_grid(
            self.conn,
            observed_again,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT close FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (revised_date.date().isoformat(),),
            ).fetchone()[0],
            second_close,
        )

    def test_partial_authoritative_histories_replay_yahoo_fallback_correction(self) -> None:
        data = sample_strategy_data()
        revised_date = data.index[10]
        first_close = 38.83776092529297
        second_close = 38.837764739990234
        data.loc[
            revised_date,
            ["TQQQ_Open", "TQQQ_High", "TQQQ_Low", "TQQQ_Close"],
        ] = [first_close, first_close + 1.0, first_close - 1.0, first_close]
        data.attrs["market_data_providers"] = {"TQQQ": "yahoo_finance"}
        qqq_history = self.canonical_histories(data)["QQQ"]
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories={"QQQ": qqq_history},
        )
        observed_again = data.copy()
        observed_again.loc[revised_date, "TQQQ_Close"] = second_close
        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            process_asset_grid(
                self.conn,
                observed_again,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=False,
                authoritative_histories={"QQQ": qqq_history},
            )

        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0])
        self.assertEqual(
            self.conn.execute(
                "SELECT close FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (revised_date.date().isoformat(),),
            ).fetchone()[0],
            second_close,
        )

    def test_authoritative_history_removal_and_backfill_are_revisions(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        removed_date = history.index[10]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        changes_before = self.conn.total_changes

        revised_removal = _synchronize_market_data_history(
            self.conn,
            history.drop(index=removed_date),
            "TQQQ",
        )

        self.assertTrue(revised_removal)
        self.assertEqual(self.conn.total_changes - changes_before, 1)
        changes_before = self.conn.total_changes

        revised_backfill = _synchronize_market_data_history(self.conn, history, "TQQQ")

        self.assertTrue(revised_backfill)
        self.assertEqual(self.conn.total_changes - changes_before, 1)

    def test_authoritative_history_shortened_range_preserves_persisted_prefix_and_tail(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        self.assertFalse(_synchronize_market_data_history(self.conn, history.iloc[15:], "TQQQ"))
        self.assertFalse(_synchronize_market_data_history(self.conn, history.iloc[:-10], "TQQQ"))

        stored_dates = self.conn.execute("SELECT date FROM market_data WHERE symbol = 'TQQQ' ORDER BY date").fetchall()
        self.assertEqual(
            stored_dates,
            [(date.date().isoformat(),) for date in history.index],
        )

    def test_authoritative_history_null_values_fail_before_writing(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        history.loc[history.index[10], "TQQQ_Volume"] = np.nan
        changes_before = self.conn.total_changes

        with self.assertRaisesRegex(ValueError, "Volume must be non-negative and finite"):
            _synchronize_market_data_history(self.conn, history, "TQQQ")

        self.assertEqual(self.conn.total_changes, changes_before)

    def test_repeated_identical_boundary_removal_is_confirmed_and_replayed(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        first_date = history.index[0]
        shortened = history.iloc[1:]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))

        self.assertFalse(_synchronize_market_data_history(self.conn, shortened, "TQQQ"))
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (first_date.date().isoformat(),),
            ).fetchone()[0],
            1,
        )

        self.conn.commit()
        self.assertTrue(_synchronize_market_data_history(self.conn, shortened, "TQQQ"))
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (first_date.date().isoformat(),),
            ).fetchone()[0],
            0,
        )

    def test_boundary_removal_requires_observations_from_distinct_workflow_runs(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        first_date = history.index[0]
        shortened = history.iloc[1:]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))

        self.assertFalse(
            _synchronize_market_data_history(
                self.conn,
                shortened,
                "TQQQ",
                observation_run_id="workflow-run-1",
            )
        )
        self.conn.commit()
        self.assertFalse(
            _synchronize_market_data_history(
                self.conn,
                shortened,
                "TQQQ",
                observation_run_id="workflow-run-1",
            )
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT consecutive_observations FROM market_history_removal_candidates WHERE symbol = 'TQQQ'"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (first_date.date().isoformat(),),
            ).fetchone()[0],
            1,
        )

        self.conn.commit()
        self.assertTrue(
            _synchronize_market_data_history(
                self.conn,
                shortened,
                "TQQQ",
                observation_run_id="workflow-run-2",
            )
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                (first_date.date().isoformat(),),
            ).fetchone()[0],
            0,
        )

    def test_boundary_removal_candidate_is_cleared_when_full_history_recovers(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        self.assertFalse(_synchronize_market_data_history(self.conn, history.iloc[1:], "TQQQ"))

        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        self.assertFalse(_synchronize_market_data_history(self.conn, history.iloc[1:], "TQQQ"))

        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ'").fetchone()[0],
            len(history),
        )

    def test_authoritative_history_missing_columns_fails_before_writing(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"].drop(columns="TQQQ_Volume")
        changes_before = self.conn.total_changes

        with self.assertRaisesRegex(ValueError, "TQQQ_Volume"):
            _synchronize_market_data_history(self.conn, history, "TQQQ")

        self.assertEqual(self.conn.total_changes, changes_before)

    def test_authoritative_history_rejects_boolean_and_complex_values_before_writing(self) -> None:
        for label, invalid_value in (("boolean", True), ("complex", 100.0 + 1.0j)):
            with self.subTest(label=label):
                history = self.canonical_histories(sample_strategy_data())["TQQQ"]
                history["TQQQ_Open"] = history["TQQQ_Open"].astype(object)
                history.iloc[0, history.columns.get_loc("TQQQ_Open")] = invalid_value
                changes_before = self.conn.total_changes

                with self.assertRaisesRegex(AssetMarketDataError, "not boolean or complex"):
                    _synchronize_market_data_history(self.conn, history, "TQQQ")

                self.assertEqual(self.conn.total_changes, changes_before)
                self.assertEqual(
                    self.conn.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ'").fetchone()[0],
                    0,
                )

    def test_authoritative_history_rejects_temporal_values_before_writing(self) -> None:
        temporal_values = (
            pd.Series(pd.to_datetime(["2026-01-02"] * 40)),
            pd.Series(pd.to_timedelta(range(40), unit="D")),
            dt.date(2026, 1, 2),
            dt.datetime(2026, 1, 2, 12),
            dt.timedelta(days=1),
            pd.Timestamp("2026-01-02"),
            pd.Timedelta(days=1),
            np.datetime64("2026-01-02"),
            np.timedelta64(1, "D"),
        )
        for temporal_value in temporal_values:
            with self.subTest(value_type=type(temporal_value).__name__):
                history = self.canonical_histories(sample_strategy_data())["TQQQ"]
                if isinstance(temporal_value, pd.Series):
                    history["TQQQ_Open"] = temporal_value.to_numpy()
                else:
                    history["TQQQ_Open"] = history["TQQQ_Open"].astype(object)
                    history.iloc[0, history.columns.get_loc("TQQQ_Open")] = temporal_value
                changes_before = self.conn.total_changes

                with self.assertRaisesRegex(AssetMarketDataError, "not a date, datetime, or timedelta"):
                    _synchronize_market_data_history(self.conn, history, "TQQQ")

                self.assertEqual(self.conn.total_changes, changes_before)
                self.assertEqual(
                    self.conn.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ'").fetchone()[0],
                    0,
                )

    def test_market_data_rejects_numeric_object_session_labels_before_writing(self) -> None:
        invalid_labels = (0, 0.0, True, np.int64(0), np.float64(0.0), np.bool_(False))
        for label in invalid_labels:
            with self.subTest(label=label, label_type=type(label).__name__):
                history = self.canonical_histories(sample_strategy_data(periods=1))["TQQQ"]
                history.index = pd.Index([label], dtype=object)

                with self.assertRaisesRegex(AssetMarketDataError, "must use calendar dates"):
                    storage_module.validate_market_data_frame(history, "TQQQ")
                with self.assertRaisesRegex(AssetMarketDataError, "must use calendar dates"):
                    storage_module.save_market_data(self.conn, history, ["TQQQ"])

                self.assertEqual(
                    self.conn.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ'").fetchone()[0],
                    0,
                )

    def test_authoritative_history_accepts_nullable_and_numeric_string_columns(self) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        history = pd.DataFrame(
            {
                "TQQQ_Open": pd.Series(["100", "101"], index=index, dtype="string"),
                "TQQQ_High": pd.Series([102, 103], index=index, dtype="Int64"),
                "TQQQ_Low": pd.Series([99, 100], index=index, dtype="Int64"),
                "TQQQ_Close": pd.Series([101.5, 102.5], index=index, dtype="Float64"),
                "TQQQ_Volume": pd.Series(["1000", "1100"], index=index, dtype="string"),
            },
            index=index,
        )

        _synchronize_market_data_history(self.conn, history, "TQQQ")

        rows = self.conn.execute(
            "SELECT open, high, low, close, volume FROM market_data WHERE symbol = 'TQQQ' ORDER BY date"
        ).fetchall()
        self.assertEqual(rows, [(100.0, 102.0, 99.0, 101.5, 1000.0), (101.0, 103.0, 100.0, 102.5, 1100.0)])

    def test_presynchronized_authoritative_histories_skip_only_their_sync_calls(self) -> None:
        data = sample_strategy_data()
        histories = self.canonical_histories(data)
        _synchronize_market_data_history(self.conn, histories["QQQ"], "QQQ")
        _synchronize_market_data_history(self.conn, histories[RISK_FREE_SYMBOL], RISK_FREE_SYMBOL)

        with patch(
            "leveraged_trader.storage._synchronize_market_data_history",
            wraps=_synchronize_market_data_history,
        ) as synchronize:
            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories=histories,
                presynchronized_authoritative_symbols={"QQQ", RISK_FREE_SYMBOL},
            )

        self.assertEqual(
            [call.args[2] for call in synchronize.call_args_list],
            ["TQQQ"],
        )

    def test_presynchronized_symbol_must_have_an_authoritative_history(self) -> None:
        data = sample_strategy_data()

        with self.assertRaisesRegex(ValueError, "Presynchronized market symbols"):
            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories=self.canonical_histories(data),
                presynchronized_authoritative_symbols={"SPY"},
            )

    @staticmethod
    def canonical_histories(data: pd.DataFrame, asset_symbol: str = "TQQQ") -> dict[str, pd.DataFrame]:
        return {
            asset_symbol: data[[column for column in data if column.startswith(f"{asset_symbol}_")]].copy(),
            "QQQ": data[[column for column in data if column.startswith("QQQ_")]].copy(),
            RISK_FREE_SYMBOL: data[[column for column in data if column.startswith(f"{RISK_FREE_SYMBOL}_")]].copy(),
        }

    def test_process_asset_grid_stores_summaries_for_all_configs_and_only_best_equity(self) -> None:
        data = sample_strategy_data()

        self.process_grid(data, rebuild=True)

        summary_count = self.conn.execute("SELECT COUNT(*) FROM strategy_summary").fetchone()[0]
        state_count = self.conn.execute("SELECT COUNT(*) FROM strategy_state").fetchone()[0]
        equity_configs = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT buy_rsi, profit_target_multiple
                FROM strategy_equity
            )
            """
        ).fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertEqual(summary_count, 4)
        self.assertEqual(state_count, 4)
        self.assertEqual(equity_configs, 1)
        self.assertEqual(equity_rows, len(data))

    def test_missing_signal_session_preserves_asset_calendar_and_best_result_parity(self) -> None:
        data = sample_strategy_data(periods=20)
        histories = self.canonical_histories(data)
        missing_signal_date = data.index[8]
        histories["QQQ"] = histories["QQQ"].drop(index=missing_signal_date)
        merged = pd.concat(list(histories.values()), axis=1, join="outer")

        process_asset_grid(
            self.conn,
            merged,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0, 70.0],
            [1.05, 1.5],
            rebuild=True,
            authoritative_histories=histories,
        )

        best_buy_rsi, best_target = self.conn.execute(
            "SELECT DISTINCT buy_rsi, profit_target_multiple FROM strategy_equity"
        ).fetchone()
        equity = pd.read_sql_query(
            """
            SELECT date, equity, risk_free_return, in_position, pending_action, trades_executed
            FROM strategy_equity
            ORDER BY date
            """,
            self.conn,
            parse_dates=["date"],
        )
        state = self.conn.execute(
            """
            SELECT last_date, in_position, pending_action, prev_equity, trades_executed
            FROM strategy_state
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            (best_buy_rsi, best_target),
        ).fetchone()
        summary = self.conn.execute(
            """
            SELECT end_date, trading_days, total_return, cagr, sharpe
            FROM strategy_summary
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            (best_buy_rsi, best_target),
        ).fetchone()
        recomputed = performance_summary(
            equity.set_index("date")["equity"],
            equity.set_index("date")["risk_free_return"],
        )

        self.assertEqual(equity["date"].tolist(), data.index.tolist())
        self.assertIn(missing_signal_date, equity["date"].tolist())
        last_equity = equity.iloc[-1]
        self.assertEqual(state[0], data.index[-1].date().isoformat())
        self.assertEqual(bool(state[1]), bool(last_equity["in_position"]))
        self.assertEqual(state[2], last_equity["pending_action"])
        self.assertAlmostEqual(float(state[3]), float(last_equity["equity"]))
        self.assertEqual(int(state[4]), int(last_equity["trades_executed"]))
        self.assertEqual(summary[:2], (state[0], len(data)))
        self.assertAlmostEqual(float(summary[2]), recomputed["Total Return"])
        self.assertAlmostEqual(float(summary[3]), recomputed["CAGR"])
        self.assertAlmostEqual(float(summary[4]), recomputed["Sharpe"])

        reported_summary, reported_curves = summarize_saved_results(
            self.conn,
            pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}]),
        )
        self.assertEqual(len(reported_summary), 1)
        self.assertEqual(float(reported_summary.iloc[0]["Buy RSI"]), float(best_buy_rsi))
        self.assertEqual(float(reported_summary.iloc[0]["Sell Return Multiple"]), float(best_target))
        self.assertEqual(reported_curves.index.tolist(), data.index.tolist())
        np.testing.assert_allclose(
            reported_curves.iloc[:, 0].to_numpy(dtype=float),
            equity["equity"].to_numpy(dtype=float),
        )

    def test_authoritative_histories_are_always_the_strategy_computation_input(self) -> None:
        canonical = sample_strategy_data()
        divergent = canonical.copy()
        divergent.loc[:, ["TQQQ_Open", "TQQQ_High", "TQQQ_Low", "TQQQ_Close"]] *= 2.0
        expected_conn = sqlite3.connect(":memory:")
        init_state_db(expected_conn)
        try:
            process_asset_grid(
                self.conn,
                divergent,
                self.cfg,
                "TQQQ",
                "QQQ",
                [100.0],
                [10.0],
                rebuild=True,
                authoritative_histories=self.canonical_histories(canonical),
            )
            process_asset_grid(
                expected_conn,
                canonical,
                self.cfg,
                "TQQQ",
                "QQQ",
                [100.0],
                [10.0],
                rebuild=True,
                authoritative_histories=self.canonical_histories(canonical),
            )

            columns = (
                "start_date, last_date, cash, shares, in_position, entry_price, "
                "pending_action, prev_equity, trades_executed"
            )
            actual_state = self.conn.execute(f"SELECT {columns} FROM strategy_state").fetchone()
            expected_state = expected_conn.execute(f"SELECT {columns} FROM strategy_state").fetchone()
            self.assertEqual(actual_state, expected_state)
        finally:
            expected_conn.close()

    def test_process_asset_grid_reports_exact_grid_compute_duration(self) -> None:
        observed: list[float] = []
        with patch("leveraged_trader.storage.time") as mock_time:
            mock_time.perf_counter.side_effect = [10.0, 12.5]
            process_asset_grid(
                self.conn,
                sample_strategy_data(),
                self.cfg,
                "TQQQ",
                "QQQ",
                buy_rsi_values=[30.0],
                profit_target_values=[1.50],
                rebuild=True,
                grid_compute_observer=observed.append,
            )

        self.assertEqual(observed, [2.5])

    def test_best_summary_matches_stored_best_equity_curve(self) -> None:
        data = sample_strategy_data()

        self.process_grid(data, rebuild=True)

        best_summary = pd.read_sql_query(
            """
            SELECT *
            FROM strategy_summary
            ORDER BY sharpe DESC
            LIMIT 1
            """,
            self.conn,
        ).iloc[0]
        equity_df = pd.read_sql_query(
            """
            SELECT date, equity, risk_free_return
            FROM strategy_equity
            ORDER BY date
            """,
            self.conn,
            parse_dates=["date"],
        )
        curve_summary = performance_summary(
            equity_df.set_index("date")["equity"],
            equity_df.set_index("date")["risk_free_return"],
        )

        self.assertAlmostEqual(best_summary["sharpe"], curve_summary["Sharpe"])
        self.assertAlmostEqual(best_summary["total_return"], curve_summary["Total Return"])
        self.assertAlmostEqual(best_summary["cagr"], curve_summary["CAGR"])

    def test_one_ulp_tie_retains_the_sql_selected_best_equity_curve(self) -> None:
        data = one_ulp_best_strategy_data()
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0],
            profit_target_values=[5.0, 6.0],
            rebuild=True,
        )

        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])
        optimization_summary, best_curves = summarize_saved_results(
            self.conn,
            workflow_assets,
        )
        retained_config = self.conn.execute(
            """
            SELECT DISTINCT buy_rsi, profit_target_multiple
            FROM strategy_equity
            """
        ).fetchone()

        self.assertEqual(len(optimization_summary), 1)
        self.assertEqual(len(best_curves), len(data))
        self.assertEqual(
            retained_config,
            (
                optimization_summary.iloc[0]["Buy RSI"],
                optimization_summary.iloc[0]["Sell Return Multiple"],
            ),
        )

    def test_best_curve_reconstruction_converges_before_pruning_tied_candidate(self) -> None:
        data = one_ulp_best_strategy_data()
        real_replace = storage_module._replace_best_equity_curve
        replaced_targets: list[float] = []
        first_excess_moments: tuple[float, float, float, float] | None = None

        def replace_and_shift_winner(*args: object, **kwargs: object) -> None:
            nonlocal first_excess_moments
            real_replace(*args, **kwargs)
            conn = args[0]
            buy_rsi = float(args[4])
            target = float(args[5])
            assert isinstance(conn, sqlite3.Connection)
            replaced_targets.append(target)
            if len(replaced_targets) == 1:
                first_excess_moments = tuple(
                    float(value)
                    for value in conn.execute(
                        """
                        SELECT excess_return_sum, excess_return_sum_squares,
                               excess_return_mean, excess_return_m2
                        FROM strategy_summary
                        WHERE buy_rsi = ? AND profit_target_multiple = ?
                        """,
                        (buy_rsi, target),
                    ).fetchone()
                )
                conn.execute(
                    """
                    UPDATE strategy_summary
                    SET excess_return_sum = -CAST(excess_return_count AS REAL),
                        excess_return_sum_squares =
                            excess_return_m2 + CAST(excess_return_count AS REAL),
                        excess_return_mean = -1.0
                    WHERE buy_rsi = ? AND profit_target_multiple = ?
                    """,
                    (buy_rsi, target),
                )
                self.refresh_strategy_summary_metrics()
            elif len(replaced_targets) == 2:
                assert first_excess_moments is not None
                conn.execute(
                    """
                    UPDATE strategy_summary
                    SET excess_return_sum = ?,
                        excess_return_sum_squares = ?,
                        excess_return_mean = ?,
                        excess_return_m2 = ?
                    WHERE buy_rsi = ? AND profit_target_multiple = 5.0
                    """,
                    (*first_excess_moments, buy_rsi),
                )
                self.refresh_strategy_summary_metrics()

        with patch(
            "leveraged_trader.storage._replace_best_equity_curve",
            side_effect=replace_and_shift_winner,
        ):
            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                buy_rsi_values=[30.0],
                profit_target_values=[5.0, 6.0],
                rebuild=True,
            )

        retained_target = self.conn.execute("SELECT DISTINCT profit_target_multiple FROM strategy_equity").fetchone()[0]
        self.assertEqual(replaced_targets, [5.0, 6.0])
        self.assertEqual(retained_target, 5.0)

    def test_unchanged_update_rebuilds_semantically_corrupted_best_equity_curve(self) -> None:
        data = one_ulp_best_strategy_data()
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0],
            profit_target_values=[5.0, 6.0],
            rebuild=True,
        )
        final_date = data.index[-1].date().isoformat()
        columns = (
            "equity, daily_return, risk_free_return, in_position, action_executed, pending_action, trades_executed"
        )
        expected_final_row = self.conn.execute(
            f"SELECT {columns} FROM strategy_equity WHERE date = ?",
            (final_date,),
        ).fetchone()
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])
        summary, _ = summarize_saved_results(self.conn, workflow_assets)
        self.assertTrue(
            build_sell_signal_report(
                self.conn,
                summary,
                self.cfg.rsi_period,
                base_cfg=self.cfg,
                expected_buy_rsi_values=[30.0],
                expected_profit_target_values=[5.0, 6.0],
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            ).empty
        )

        self.conn.execute(
            """
            UPDATE strategy_equity
            SET equity = 1.0,
                daily_return = 99.0,
                in_position = 0,
                action_executed = 'sell',
                pending_action = 'none',
                trades_executed = 999
            WHERE date = ?
            """,
            (final_date,),
        )
        # The final action is not trusted independently of the authenticated
        # complete curve, even when the strategy state itself still verifies.
        self.assertTrue(
            build_sell_signal_report(
                self.conn,
                summary,
                self.cfg.rsi_period,
                base_cfg=self.cfg,
                expected_buy_rsi_values=[30.0],
                expected_profit_target_values=[5.0, 6.0],
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            ).empty
        )

        process_asset_grid(
            self.conn,
            data.iloc[-1:],
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0],
            profit_target_values=[5.0, 6.0],
            rebuild=False,
        )

        rebuilt_final_row = self.conn.execute(
            f"SELECT {columns} FROM strategy_equity WHERE date = ?",
            (final_date,),
        ).fetchone()
        rebuilt_summary, _ = summarize_saved_results(self.conn, workflow_assets)
        self.assertEqual(rebuilt_final_row, expected_final_row)
        self.assertTrue(
            build_sell_signal_report(
                self.conn,
                rebuilt_summary,
                self.cfg.rsi_period,
                base_cfg=self.cfg,
                expected_buy_rsi_values=[30.0],
                expected_profit_target_values=[5.0, 6.0],
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            ).empty
        )

    def test_process_asset_grid_updates_summaries_and_rewrites_single_best_equity_curve(self) -> None:
        data = sample_strategy_data()
        first_window = data.iloc[:25]
        update_window = data.iloc[24:]

        self.process_grid(first_window, rebuild=True)
        self.process_grid(update_window, rebuild=False)

        end_dates = self.conn.execute("SELECT DISTINCT end_date FROM strategy_summary").fetchall()
        equity_configs = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT buy_rsi, profit_target_multiple
                FROM strategy_equity
            )
            """
        ).fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertEqual(end_dates, [(data.index[-1].date().isoformat(),)])
        self.assertEqual(equity_configs, 1)
        self.assertEqual(equity_rows, len(data))

    def test_unchanged_update_rebuilds_truncated_best_equity_curve(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        first_date = data.index[0].date().isoformat()
        self.conn.execute("DELETE FROM strategy_equity WHERE date != ?", (first_date,))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0], 1)

        self.process_grid(data.iloc[-1:], rebuild=False)

        stored_dates = [
            row[0] for row in self.conn.execute("SELECT date FROM strategy_equity ORDER BY date").fetchall()
        ]
        self.assertEqual(stored_dates, [date.date().isoformat() for date in data.index])

    def test_missing_nonbest_summary_forces_safe_full_grid_rebuild(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data.iloc[:25], rebuild=True)
        best_config = self.conn.execute(
            "SELECT DISTINCT buy_rsi, profit_target_multiple FROM strategy_equity"
        ).fetchone()
        missing_config = self.conn.execute(
            """
            SELECT buy_rsi, profit_target_multiple
            FROM strategy_summary
            WHERE buy_rsi != ? OR profit_target_multiple != ?
            LIMIT 1
            """,
            best_config,
        ).fetchone()
        self.assertIsNotNone(missing_config)
        self.conn.execute(
            """
            DELETE FROM strategy_summary
            WHERE buy_rsi = ? AND profit_target_multiple = ?
            """,
            missing_config,
        )

        self.process_grid(data, rebuild=False)

        summaries = self.conn.execute(
            "SELECT buy_rsi, profit_target_multiple, start_date, end_date, trading_days FROM strategy_summary"
        ).fetchall()
        self.assertEqual(len(summaries), 4)
        self.assertTrue(all(row[2] == data.index[0].date().isoformat() for row in summaries))
        self.assertTrue(all(row[3] == data.index[-1].date().isoformat() for row in summaries))
        self.assertTrue(all(row[4] == len(data) for row in summaries))

    def test_missing_summary_rollup_forces_safe_full_history_rebuild(self) -> None:
        data = sample_strategy_data()
        first_window = data.iloc[:25]
        update_window = data.iloc[24:]

        process_asset_grid(
            self.conn,
            first_window,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0],
            profit_target_values=[1.50],
            rebuild=True,
        )
        self.conn.execute("UPDATE strategy_summary SET first_equity = NULL")

        process_asset_grid(
            self.conn,
            update_window,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0],
            profit_target_values=[1.50],
            rebuild=False,
        )

        trading_days = self.conn.execute("SELECT trading_days FROM strategy_summary").fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertEqual(trading_days, len(data))
        self.assertEqual(equity_rows, len(data))

    def test_corrected_persisted_session_rebuilds_rsi_and_strategy_state(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        previous_rsi = self.conn.execute(
            "SELECT rsi FROM rsi_values WHERE signal_symbol = 'QQQ' ORDER BY date DESC LIMIT 1"
        ).fetchone()[0]

        revision = data.iloc[-1:].copy()
        revision.loc[:, "QQQ_Close"] = 200.0
        revision.loc[:, "QQQ_High"] = 201.0
        self.process_grid(revision, rebuild=False)

        stored_rsi = self.conn.execute(
            "SELECT rsi FROM rsi_values WHERE signal_symbol = 'QQQ' ORDER BY date DESC LIMIT 1"
        ).fetchone()[0]
        expected_close = data["QQQ_Close"].copy()
        expected_close.iloc[-1] = 200.0
        expected_rsi = compute_rsi(expected_close, self.cfg.rsi_period).iloc[-1]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertNotEqual(stored_rsi, previous_rsi)
        self.assertAlmostEqual(stored_rsi, expected_rsi)
        self.assertEqual(equity_rows, len(data))

    def test_incremental_rsi_matches_neutral_flat_full_recompute(self) -> None:
        close = pd.Series([100.0] * 10, index=pd.date_range("2026-01-02", periods=10, freq="B"))

        ensure_rsi_values(self.conn, "QQQ", self.cfg.rsi_period, close.iloc[:5], rebuild=True)
        incremental = ensure_rsi_values(self.conn, "QQQ", self.cfg.rsi_period, close, rebuild=False)
        expected = compute_rsi(close, self.cfg.rsi_period)

        self.assertEqual(float(incremental.dropna().iloc[-1]), 50.0)
        pd.testing.assert_series_equal(incremental, expected, check_names=False)

    def test_materially_different_tiny_cached_close_path_forces_rsi_rebuild(self) -> None:
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        canonical_close = pd.Series(
            np.tile([1e-20, 2e-20], 5),
            index=dates,
        )
        forged_close = pd.Series(
            np.arange(1.0, 11.0) * 1e-13,
            index=dates,
        )
        ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            forged_close,
            rebuild=True,
        )

        rebuilt = ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            canonical_close,
            rebuild=False,
        )

        expected = compute_rsi(canonical_close, self.cfg.rsi_period)
        pd.testing.assert_series_equal(rebuilt, expected, check_names=False)
        stored_closes = pd.Series(
            [
                row[0]
                for row in self.conn.execute(
                    """
                    SELECT close
                    FROM rsi_values
                    WHERE signal_symbol = 'QQQ' AND rsi_period = ?
                    ORDER BY date
                    """,
                    (self.cfg.rsi_period,),
                ).fetchall()
            ],
            index=dates,
        )
        pd.testing.assert_series_equal(stored_closes, canonical_close)

    def test_tiny_corrupt_wilder_average_forces_full_rsi_rebuild(self) -> None:
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        close = pd.Series(np.tile([1e-20, 2e-20], 5), index=dates)
        prefix = close.iloc[:-1]
        ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            prefix,
            rebuild=True,
        )
        self.conn.execute(
            """
            UPDATE rsi_values
            SET avg_gain = avg_gain + 0.0000000000005
            WHERE signal_symbol = 'QQQ'
              AND rsi_period = ?
              AND date = ?
            """,
            (self.cfg.rsi_period, prefix.index[-1].date().isoformat()),
        )

        rebuilt = ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            close,
            rebuild=False,
        )

        expected = compute_rsi(close, self.cfg.rsi_period)
        pd.testing.assert_series_equal(rebuilt, expected, check_names=False)

    def test_subnormal_cached_wilder_drift_forces_rebuild_before_append(self) -> None:
        ulp = np.nextafter(0.0, np.inf)
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        close = pd.Series(
            np.asarray([1000, 1004, 1000, 1004, 1000, 1004, 1000, 1004, 1000, 1004]) * ulp,
            index=dates,
        )
        ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            close.iloc[:-1],
            rebuild=True,
        )
        corrupt_date = dates[-2].date().isoformat()
        canonical_gain = self.conn.execute(
            "SELECT avg_gain FROM rsi_values WHERE signal_symbol = 'QQQ' AND rsi_period = ? AND date = ?",
            (self.cfg.rsi_period, corrupt_date),
        ).fetchone()[0]
        self.assertEqual(canonical_gain, ulp)
        self.conn.execute(
            "UPDATE rsi_values SET avg_gain = ? WHERE signal_symbol = 'QQQ' AND rsi_period = ? AND date = ?",
            (5.0 * ulp, self.cfg.rsi_period, corrupt_date),
        )

        rebuilt = ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            close,
            rebuild=False,
        )

        expected = compute_rsi(close, self.cfg.rsi_period)
        pd.testing.assert_series_equal(rebuilt, expected, check_names=False)
        self.assertEqual(float(rebuilt.iloc[-1]), float(expected.iloc[-1]))

    def test_negative_persisted_rsi_average_forces_full_rebuild(self) -> None:
        close = pd.Series(
            np.linspace(100.0, 110.0, 10),
            index=pd.date_range("2026-01-02", periods=10, freq="B"),
        )
        ensure_rsi_values(self.conn, "QQQ", self.cfg.rsi_period, close, rebuild=True)
        self.conn.execute(
            """
            UPDATE rsi_values
            SET avg_gain = -1.0
            WHERE signal_symbol = 'QQQ'
              AND rsi_period = ?
              AND date = ?
            """,
            (self.cfg.rsi_period, close.index[-2].date().isoformat()),
        )

        rebuilt = ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            close,
            rebuild=False,
        )

        expected = compute_rsi(close, self.cfg.rsi_period)
        pd.testing.assert_series_equal(rebuilt, expected, check_names=False)
        avg_gain, avg_loss = self.conn.execute(
            """
            SELECT avg_gain, avg_loss
            FROM rsi_values
            WHERE signal_symbol = 'QQQ' AND rsi_period = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (self.cfg.rsi_period,),
        ).fetchone()
        self.assertGreaterEqual(avg_gain, 0.0)
        self.assertGreaterEqual(avg_loss, 0.0)

    def test_missing_interior_persisted_rsi_row_forces_full_rebuild(self) -> None:
        close = pd.Series(
            np.linspace(100.0, 110.0, 12),
            index=pd.date_range("2026-01-02", periods=12, freq="B"),
        )
        ensure_rsi_values(self.conn, "QQQ", self.cfg.rsi_period, close, rebuild=True)
        missing_date = close.index[6].date().isoformat()
        self.conn.execute(
            """
            DELETE FROM rsi_values
            WHERE signal_symbol = 'QQQ' AND rsi_period = ? AND date = ?
            """,
            (self.cfg.rsi_period, missing_date),
        )

        rebuilt = ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            close,
            rebuild=False,
        )

        expected = compute_rsi(close, self.cfg.rsi_period)
        pd.testing.assert_series_equal(rebuilt, expected, check_names=False)
        cached_count = self.conn.execute(
            """
            SELECT COUNT(*) FROM rsi_values
            WHERE signal_symbol = 'QQQ' AND rsi_period = ?
            """,
            (self.cfg.rsi_period,),
        ).fetchone()[0]
        self.assertEqual(cached_count, len(close))

    def test_invalid_interior_persisted_rsi_semantics_force_full_rebuild(self) -> None:
        close = pd.Series(
            np.linspace(100.0, 110.0, 12),
            index=pd.date_range("2026-01-02", periods=12, freq="B"),
        )
        ensure_rsi_values(self.conn, "QQQ", self.cfg.rsi_period, close, rebuild=True)
        corrupt_date = close.index[6].date().isoformat()
        self.conn.execute(
            """
            UPDATE rsi_values
            SET rsi = 12.345
            WHERE signal_symbol = 'QQQ' AND rsi_period = ? AND date = ?
            """,
            (self.cfg.rsi_period, corrupt_date),
        )

        rebuilt = ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            close,
            rebuild=False,
        )

        expected = compute_rsi(close, self.cfg.rsi_period)
        pd.testing.assert_series_equal(rebuilt, expected, check_names=False)
        stored_rsi = self.conn.execute(
            """
            SELECT rsi FROM rsi_values
            WHERE signal_symbol = 'QQQ' AND rsi_period = ? AND date = ?
            """,
            (self.cfg.rsi_period, corrupt_date),
        ).fetchone()[0]
        self.assertAlmostEqual(stored_rsi, expected.loc[pd.Timestamp(corrupt_date)])

    def test_rsi_prefix_read_preserves_valid_cached_suffix(self) -> None:
        close = pd.Series(
            np.linspace(100.0, 110.0, 12),
            index=pd.date_range("2026-01-02", periods=12, freq="B"),
        )
        ensure_rsi_values(self.conn, "QQQ", self.cfg.rsi_period, close, rebuild=True)

        prefix = ensure_rsi_values(
            self.conn,
            "QQQ",
            self.cfg.rsi_period,
            close.iloc[:6],
            rebuild=False,
        )

        expected = compute_rsi(close, self.cfg.rsi_period)
        pd.testing.assert_series_equal(prefix, expected.iloc[:6], check_names=False)
        cached_count = self.conn.execute(
            """
            SELECT COUNT(*) FROM rsi_values
            WHERE signal_symbol = 'QQQ' AND rsi_period = ?
            """,
            (self.cfg.rsi_period,),
        ).fetchone()[0]
        self.assertEqual(cached_count, len(close))

    def test_strategy_summary_integrity_columns_exist_on_fresh_and_existing_schema(self) -> None:
        expected_columns = {
            "return_mean",
            "return_m2",
            "excess_return_mean",
            "excess_return_m2",
            "integrity_digest",
        }
        fresh_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(strategy_summary)").fetchall()}
        self.assertTrue(expected_columns.issubset(fresh_columns))

        legacy_conn = sqlite3.connect(":memory:")
        try:
            init_state_db(legacy_conn)
            for column in expected_columns:
                legacy_conn.execute(f"ALTER TABLE strategy_summary DROP COLUMN {column}")
            init_state_db(legacy_conn)
            migrated_columns = {row[1] for row in legacy_conn.execute("PRAGMA table_info(strategy_summary)").fetchall()}
            self.assertTrue(expected_columns.issubset(migrated_columns))
        finally:
            legacy_conn.close()

    def test_init_state_db_rejects_partial_managed_tables_without_side_effects(self) -> None:
        schemas = {
            "strategy_state": (
                "asset_symbol TEXT NOT NULL, signal_symbol TEXT NOT NULL, "
                "buy_rsi REAL NOT NULL, profit_target_multiple REAL NOT NULL, "
                "PRIMARY KEY (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)"
            ),
            "strategy_summary": (
                "asset_symbol TEXT NOT NULL, signal_symbol TEXT NOT NULL, "
                "buy_rsi REAL NOT NULL, profit_target_multiple REAL NOT NULL, "
                "PRIMARY KEY (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)"
            ),
            "strategy_config": "asset_symbol TEXT NOT NULL, signal_symbol TEXT NOT NULL",
            "market_data": "symbol TEXT NOT NULL, date TEXT NOT NULL",
            "rsi_values": ("signal_symbol TEXT NOT NULL, rsi_period INTEGER NOT NULL, date TEXT NOT NULL"),
        }
        for table_name, columns_sql in schemas.items():
            with self.subTest(table=table_name), closing(sqlite3.connect(":memory:")) as conn:
                conn.execute(f"CREATE TABLE {table_name} ({columns_sql})")
                objects_before = conn.execute(
                    "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
                ).fetchall()

                with self.assertRaisesRegex(
                    ValueError,
                    rf"{table_name} is missing required managed columns",
                ):
                    init_state_db(conn)

                self.assertEqual(
                    conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                    objects_before,
                )

    def test_constraint_stripped_import_restores_nullability_and_identity_types(self) -> None:
        cases = (
            ("strategy_config", "asset_symbol", (None, "QQQ", "fingerprint")),
            (
                "strategy_state",
                "asset_symbol",
                (None, "QQQ", 30.0, 1.5, "2026-01-02", "2026-01-02", 100.0, 0.0, 0, None, "none", 100.0, 0, None, None),
            ),
            ("market_data", "symbol", (None, "2026-01-02", None, None, None, None, None)),
            ("rsi_values", "signal_symbol", (None, 14, "2026-01-02", 100.0, None, None, None)),
        )
        for table_name, identity_column, null_row in cases:
            with self.subTest(table=table_name), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                conn.execute(f"ALTER TABLE {table_name} RENAME TO imported_{table_name}")
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM imported_{table_name}")
                conn.execute(f"DROP TABLE imported_{table_name}")

                init_state_db(conn)

                columns = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")]
                placeholders = ", ".join("?" for _ in null_row)
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})",
                        null_row,
                    )
                self.assertEqual(
                    next(
                        int(row[3])
                        for row in conn.execute(f"PRAGMA table_info({table_name})")
                        if str(row[1]) == identity_column
                    ),
                    1,
                )

    def test_managed_identity_guards_reject_blob_insert_and_update(self) -> None:
        cases = (
            (
                "strategy_config",
                "asset_symbol",
                "INSERT INTO strategy_config VALUES ('TQQQ', 'QQQ', 'fingerprint')",
            ),
            (
                "market_data",
                "symbol",
                "INSERT INTO market_data (symbol, date) VALUES ('TQQQ', '2026-01-02')",
            ),
            (
                "rsi_values",
                "signal_symbol",
                "INSERT INTO rsi_values (signal_symbol, rsi_period, date, close) "
                "VALUES ('QQQ', 14, '2026-01-02', 100.0)",
            ),
        )
        for table_name, identity_column, seed_sql in cases:
            with self.subTest(table=table_name), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                columns = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")]
                values = [None] * len(columns)
                values[columns.index(identity_column)] = sqlite3.Binary(b"invalid")
                for required_column, value in {
                    "signal_symbol": "QQQ",
                    "rsi_period": 14,
                    "date": "2026-01-02",
                    "close": 100.0,
                    "fingerprint": "fingerprint",
                }.items():
                    if required_column in columns and values[columns.index(required_column)] is None:
                        values[columns.index(required_column)] = value
                placeholders = ", ".join("?" for _ in values)
                with self.assertRaisesRegex(sqlite3.IntegrityError, "invalid managed storage identity"):
                    conn.execute(
                        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})",
                        values,
                    )

                conn.execute(seed_sql)
                with self.assertRaisesRegex(sqlite3.IntegrityError, "invalid managed storage identity"):
                    conn.execute(f"UPDATE {table_name} SET {identity_column} = CAST('invalid' AS BLOB)")

    def test_init_rejects_forged_state_revision_trigger_without_side_effects(self) -> None:
        self.conn.execute("DROP TRIGGER alpaca_managed_positions_increment_state_revision")
        self.conn.execute(
            """
            CREATE TRIGGER alpaca_managed_positions_increment_state_revision
            AFTER UPDATE ON alpaca_managed_positions
            BEGIN
                SELECT 1;
            END
            """
        )
        objects_before = self.conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

        with self.assertRaisesRegex(ValueError, "required managed trigger contract"):
            init_state_db(self.conn)

        self.assertEqual(
            self.conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
            objects_before,
        )

    def test_init_rejects_unexpected_triggers_on_managed_tables_atomically(self) -> None:
        cases = (
            (
                "hostile_close_limit",
                "market_data",
                """
                CREATE TRIGGER hostile_close_limit
                BEFORE INSERT ON market_data
                WHEN NEW.close >= 100
                BEGIN
                    SELECT RAISE(ABORT, 'hostile close restriction');
                END
                """,
            ),
            (
                "hostile_strategy_config",
                "strategy_config",
                """
                CREATE TRIGGER hostile_strategy_config
                BEFORE UPDATE ON strategy_config
                BEGIN
                    SELECT RAISE(ABORT, 'hostile config restriction');
                END
                """,
            ),
        )
        for trigger_name, table_name, trigger_sql in cases:
            with self.subTest(trigger=trigger_name, table=table_name), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                conn.execute(trigger_sql)
                objects_before = conn.execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
                ).fetchall()

                with self.assertRaisesRegex(ValueError, "unexpected explicit trigger"):
                    init_state_db(conn)

                self.assertEqual(
                    conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                    objects_before,
                )

    def test_init_preserves_prefixed_triggers_owned_by_external_tables(self) -> None:
        with closing(sqlite3.connect(":memory:")) as conn:
            init_state_db(conn)
            conn.executescript(
                """
                CREATE TABLE external_source (value INTEGER NOT NULL);
                CREATE TABLE external_audit (
                    trigger_name TEXT NOT NULL,
                    value INTEGER NOT NULL
                );
                """
            )
            trigger_names = (
                "leveraged_trader_external_audit",
                "strategy_state_generation_external_audit",
                "alpaca_managed_positions_external_audit",
                "alpaca_managed_sell_fills_external_audit",
            )
            for trigger_name in trigger_names:
                conn.execute(
                    f"""
                    CREATE TRIGGER {trigger_name}
                    AFTER INSERT ON external_source
                    BEGIN
                        INSERT INTO external_audit (trigger_name, value)
                        VALUES ('{trigger_name}', NEW.value);
                    END
                    """
                )
            trigger_sql_before = conn.execute(
                """
                SELECT name, tbl_name, sql
                FROM sqlite_master
                WHERE type = 'trigger' AND tbl_name = 'external_source'
                ORDER BY name
                """
            ).fetchall()

            init_state_db(conn)

            self.assertEqual(
                conn.execute(
                    """
                    SELECT name, tbl_name, sql
                    FROM sqlite_master
                    WHERE type = 'trigger' AND tbl_name = 'external_source'
                    ORDER BY name
                    """
                ).fetchall(),
                trigger_sql_before,
            )
            conn.execute("INSERT INTO external_source VALUES (7)")
            self.assertEqual(
                conn.execute("SELECT trigger_name, value FROM external_audit ORDER BY trigger_name").fetchall(),
                [(trigger_name, 7) for trigger_name in sorted(trigger_names)],
            )

    def test_init_rejects_owned_trigger_name_collision_atomically(self) -> None:
        trigger_name = "alpaca_managed_positions_increment_state_revision"
        self.conn.execute(f"DROP TRIGGER {trigger_name}")
        self.conn.execute(
            f"""
            CREATE TRIGGER {trigger_name}
            BEFORE INSERT ON strategy_config
            BEGIN
                SELECT RAISE(ABORT, 'hostile name collision');
            END
            """
        )
        objects_before = self.conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

        with self.assertRaisesRegex(ValueError, "required managed trigger contract"):
            init_state_db(self.conn)

        self.assertEqual(
            self.conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
            objects_before,
        )

    def test_init_rejects_temporary_trigger_on_managed_table_atomically(self) -> None:
        self.conn.execute(
            """
            CREATE TEMP TRIGGER hostile_temporary_market_insert
            BEFORE INSERT ON main.market_data
            BEGIN
                SELECT RAISE(ABORT, 'hostile temporary restriction');
            END
            """
        )
        main_before = self.conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        temp_before = self.conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_temp_master ORDER BY type, name"
        ).fetchall()

        with self.assertRaisesRegex(ValueError, "Temporary schema object"):
            init_state_db(self.conn)

        self.assertEqual(
            self.conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
            main_before,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_temp_master ORDER BY type, name"
            ).fetchall(),
            temp_before,
        )

    def test_managed_position_revision_fence_rejects_id_changes_and_revision_jumps(self) -> None:
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status)
            VALUES ('TQQQ', 'QQQ', 30.0, 1.5, '2026-01-02',
                    'buy-revision-fence', 'accepted')
            """
        )

        with self.assertRaisesRegex(
            sqlite3.IntegrityError,
            "invalid managed-position revision transition",
        ):
            self.conn.execute("UPDATE alpaca_managed_positions SET id = id + 1 WHERE id = 1")
        with self.assertRaisesRegex(
            sqlite3.IntegrityError,
            "invalid managed-position revision transition",
        ):
            self.conn.execute("UPDATE alpaca_managed_positions SET state_revision = state_revision + 2 WHERE id = 1")

        self.conn.execute("UPDATE alpaca_managed_positions SET state_revision = state_revision + 1 WHERE id = 1")
        self.conn.execute("UPDATE alpaca_managed_positions SET buy_status = 'filled' WHERE id = 1")
        self.assertEqual(
            self.conn.execute("SELECT id, state_revision FROM alpaca_managed_positions").fetchone(),
            (1, 2),
        )

    def test_init_restores_managed_position_autoincrement_identity(self) -> None:
        canonical_sql = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'alpaca_managed_positions'"
        ).fetchone()[0]
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute(canonical_sql.replace(" PRIMARY KEY AUTOINCREMENT", " PRIMARY KEY"))
            conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
                 buy_client_order_id, buy_status)
                VALUES ('TQQQ', 'QQQ', 30.0, 1.5, '2026-01-02',
                        'buy-autoincrement-1', 'accepted')
                """
            )

            init_state_db(conn)

            table_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'alpaca_managed_positions'"
            ).fetchone()[0]
            self.assertIn("AUTOINCREMENT", table_sql.upper())
            conn.execute("DELETE FROM alpaca_managed_positions WHERE id = 1")
            conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
                 buy_client_order_id, buy_status)
                VALUES ('TQQQ', 'QQQ', 30.0, 1.5, '2026-01-03',
                        'buy-autoincrement-2', 'accepted')
                """
            )
            self.assertEqual(
                conn.execute("SELECT id FROM alpaca_managed_positions").fetchone(),
                (2,),
            )

        spoofed_variants = (
            canonical_sql.replace(
                " PRIMARY KEY AUTOINCREMENT",
                " PRIMARY KEY /* id INTEGER PRIMARY KEY AUTOINCREMENT */",
            ),
            canonical_sql.replace(
                " PRIMARY KEY AUTOINCREMENT,",
                " PRIMARY KEY, -- id INTEGER PRIMARY KEY AUTOINCREMENT\n",
            ),
            canonical_sql.replace(
                " PRIMARY KEY AUTOINCREMENT",
                " PRIMARY KEY CHECK ('id INTEGER PRIMARY KEY AUTOINCREMENT' != '')",
            ),
        )
        for create_sql in spoofed_variants:
            with self.subTest(create_sql=create_sql), closing(sqlite3.connect(":memory:")) as conn:
                conn.execute(create_sql)
                init_state_db(conn)
                installed_sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'alpaca_managed_positions'"
                ).fetchone()[0]
                self.assertIn("AUTOINCREMENT", installed_sql.upper())

    def test_managed_position_rebuild_preserves_deleted_identity_high_water(self) -> None:
        canonical_sql = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'alpaca_managed_positions'"
        ).fetchone()[0]
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute(canonical_sql.replace("buy_status TEXT NOT NULL", "buy_status TEXT"))
            conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES (1, 'TQQQ', 'QQQ', 30.0, 1.5, '2026-01-02',
                        'buy-high-water-live', 'accepted')
                """
            )
            conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES (7, 'UPRO', 'SPY', 30.0, 1.5, '2026-01-02',
                        'buy-high-water-deleted', 'accepted')
                """
            )
            conn.execute("DELETE FROM alpaca_managed_positions WHERE id = 7")
            self.assertEqual(
                conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'alpaca_managed_positions'").fetchone(),
                (7,),
            )

            init_state_db(conn)
            conn.execute("DELETE FROM alpaca_managed_positions WHERE id = 1")
            conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES ('TQQQ', 'QQQ', 30.0, 1.5, '2026-01-03',
                        'buy-after-high-water-rebuild', 'accepted')
                """
            )

            self.assertEqual(
                conn.execute("SELECT id FROM alpaca_managed_positions").fetchone(),
                (8,),
            )

    def test_managed_position_rebuild_rejects_invalid_sequence_atomically(self) -> None:
        canonical_sql = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'alpaca_managed_positions'"
        ).fetchone()[0]
        for label, sequence_sql in (
            ("text", "UPDATE sqlite_sequence SET seq = 'invalid'"),
            ("negative", "UPDATE sqlite_sequence SET seq = -1"),
            ("below live id", "UPDATE sqlite_sequence SET seq = 4"),
            (
                "duplicate",
                "INSERT INTO sqlite_sequence(name, seq) VALUES ('alpaca_managed_positions', 5)",
            ),
        ):
            with self.subTest(label=label), closing(sqlite3.connect(":memory:")) as conn:
                noncanonical_sql = canonical_sql.replace(
                    "buy_status TEXT NOT NULL",
                    "buy_status TEXT",
                )
                conn.execute(noncanonical_sql)
                conn.execute(
                    """
                    INSERT INTO alpaca_managed_positions
                    (id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                     buy_signal_date, buy_client_order_id, buy_status)
                    VALUES (5, 'TQQQ', 'QQQ', 30.0, 1.5, '2026-01-02',
                            'buy-invalid-sequence', 'accepted')
                    """
                )
                conn.execute(sequence_sql)
                schema_before = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'alpaca_managed_positions'"
                ).fetchone()[0]
                objects_before = conn.execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
                ).fetchall()
                rows_before = conn.execute("SELECT * FROM alpaca_managed_positions").fetchall()
                sequence_before = conn.execute(
                    "SELECT name, seq, TYPEOF(seq) FROM sqlite_sequence ORDER BY rowid"
                ).fetchall()

                with self.assertRaisesRegex(
                    ValueError,
                    "AUTOINCREMENT sequence",
                ):
                    init_state_db(conn)

                self.assertEqual(
                    conn.execute(
                        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'alpaca_managed_positions'"
                    ).fetchone()[0],
                    schema_before,
                )
                self.assertEqual(
                    conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                    objects_before,
                )
                self.assertEqual(
                    conn.execute("SELECT * FROM alpaca_managed_positions").fetchall(),
                    rows_before,
                )
                self.assertEqual(
                    conn.execute("SELECT name, seq, TYPEOF(seq) FROM sqlite_sequence ORDER BY rowid").fetchall(),
                    sequence_before,
                )

    def test_init_canonically_rebuilds_extra_owned_table_constraints(self) -> None:
        contracts = storage_module._canonical_storage_table_contracts()
        for table_name, (canonical_sql, _columns) in contracts.items():
            with self.subTest(table=table_name), closing(sqlite3.connect(":memory:")) as conn:
                prefix, suffix = canonical_sql.rsplit(")", 1)
                conn.execute(f"{prefix}, CHECK (0)){suffix}")

                init_state_db(conn)

                installed_sql = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table_name,),
                ).fetchone()[0]
                self.assertEqual(
                    storage_module._normalized_schema_sql(installed_sql),
                    storage_module._normalized_schema_sql(canonical_sql),
                )

    def test_canonical_rebuild_is_foreign_key_safe_and_preserves_sequence(self) -> None:
        contracts = storage_module._canonical_storage_table_contracts()
        positions_sql = contracts["alpaca_managed_positions"][0]
        fills_sql = contracts["alpaca_managed_sell_fills"][0]
        positions_prefix, positions_suffix = positions_sql.rsplit(")", 1)
        fills_prefix, fills_suffix = fills_sql.rsplit(")", 1)
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"{positions_prefix}, CHECK (buy_rsi > 0)){positions_suffix}")
            conn.execute(
                f"{fills_prefix}, FOREIGN KEY (managed_position_id) "
                "REFERENCES alpaca_managed_positions(id) ON DELETE CASCADE, "
                f"CHECK (filled_qty >= 0)){fills_suffix}"
            )
            conn.executemany(
                """
                INSERT INTO alpaca_managed_positions
                (id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES (?, ?, 'QQQ', 30.0, 1.5, '2026-01-02', ?, 'accepted')
                """,
                [
                    (1, "TQQQ", "buy-parent-live"),
                    (7, "UPRO", "buy-parent-deleted-high-water"),
                ],
            )
            conn.execute(
                """
                INSERT INTO alpaca_managed_sell_fills
                (managed_position_id, alpaca_order_id, filled_qty, filled_value)
                VALUES (1, 'sell-child-live', 1.0, 150.0)
                """
            )
            conn.execute("DELETE FROM alpaca_managed_positions WHERE id = 7")
            self.assertEqual(
                conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'alpaca_managed_positions'").fetchone(),
                (7,),
            )

            init_state_db(conn)

            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone(), (1,))
            self.assertEqual(
                conn.execute(
                    "SELECT managed_position_id, alpaca_order_id, filled_qty, filled_value "
                    "FROM alpaca_managed_sell_fills"
                ).fetchall(),
                [(1, "sell-child-live", 1.0, 150.0)],
            )
            self.assertEqual(
                conn.execute("PRAGMA foreign_key_list(alpaca_managed_sell_fills)").fetchall(),
                [],
            )
            self.assertEqual(
                conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'alpaca_managed_positions'").fetchone(),
                (7,),
            )
            conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES ('UPRO', 'SPY', 30.0, 1.5, '2026-01-03',
                        'buy-after-coordinated-rebuild', 'accepted')
                """
            )
            self.assertEqual(
                conn.execute(
                    "SELECT id FROM alpaca_managed_positions "
                    "WHERE buy_client_order_id = 'buy-after-coordinated-rebuild'"
                ).fetchone(),
                (8,),
            )

    def test_canonical_rebuild_orders_every_managed_foreign_key_child_first(self) -> None:
        contracts = storage_module._canonical_storage_table_contracts()
        positions_sql = contracts["alpaca_managed_positions"][0]
        aliases_sql = contracts["alpaca_symbol_aliases"][0]
        positions_prefix, positions_suffix = positions_sql.rsplit(")", 1)
        aliases_prefix, aliases_suffix = aliases_sql.rsplit(")", 1)
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"{positions_prefix}, CHECK (buy_rsi > 0)){positions_suffix}")
            conn.execute(
                f"{aliases_prefix}, FOREIGN KEY (alpaca_asset_id) "
                "REFERENCES alpaca_managed_positions(id) ON DELETE CASCADE, "
                f"CHECK (symbol != '')){aliases_suffix}"
            )
            conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES (1, 'TQQQ', 'QQQ', 30.0, 1.5, '2026-01-02',
                        'buy-alias-parent', 'accepted')
                """
            )
            conn.execute(
                """
                INSERT INTO alpaca_symbol_aliases
                (alpaca_asset_id, symbol)
                VALUES ('1', 'TQQQ')
                """
            )

            init_state_db(conn)

            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone(), (1,))
            self.assertEqual(
                conn.execute("SELECT alpaca_asset_id, symbol FROM alpaca_symbol_aliases").fetchall(),
                [("1", "TQQQ")],
            )
            self.assertEqual(
                conn.execute("PRAGMA foreign_key_list(alpaca_symbol_aliases)").fetchall(),
                [],
            )

    def test_canonical_rebuild_rejects_external_foreign_key_child_atomically(self) -> None:
        positions_sql = storage_module._canonical_storage_table_contracts()["alpaca_managed_positions"][0]
        positions_prefix, positions_suffix = positions_sql.rsplit(")", 1)
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"{positions_prefix}, CHECK (buy_rsi > 0)){positions_suffix}")
            conn.execute(
                """
                CREATE TABLE external_audit (
                    managed_id INTEGER NOT NULL,
                    note TEXT NOT NULL,
                    FOREIGN KEY (managed_id) REFERENCES alpaca_managed_positions(id)
                        ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES (1, 'TQQQ', 'QQQ', 30.0, 1.5, '2026-01-02',
                        'buy-external-parent', 'accepted')
                """
            )
            conn.execute("INSERT INTO external_audit VALUES (1, 'retain me')")
            objects_before = conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
            position_rows_before = conn.execute("SELECT * FROM alpaca_managed_positions").fetchall()
            audit_rows_before = conn.execute("SELECT * FROM external_audit").fetchall()

            with self.assertRaisesRegex(ValueError, "external table external_audit"):
                init_state_db(conn)

            self.assertEqual(
                conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                objects_before,
            )
            self.assertEqual(
                conn.execute("SELECT * FROM alpaca_managed_positions").fetchall(),
                position_rows_before,
            )
            self.assertEqual(
                conn.execute("SELECT * FROM external_audit").fetchall(),
                audit_rows_before,
            )

    def test_init_rejects_noncanonical_managed_table_casing_without_losing_sequence(self) -> None:
        positions_sql = storage_module._canonical_storage_table_contracts()["alpaca_managed_positions"][0]
        uppercase_sql = positions_sql.replace(
            "CREATE TABLE alpaca_managed_positions",
            "CREATE TABLE ALPACA_MANAGED_POSITIONS",
            1,
        )
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute(uppercase_sql)
            conn.executemany(
                """
                INSERT INTO ALPACA_MANAGED_POSITIONS
                (id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES (?, ?, 'QQQ', 30.0, 1.5, '2026-01-02', ?, 'accepted')
                """,
                [
                    (1, "TQQQ", "buy-live"),
                    (7, "UPRO", "buy-deleted-high-water"),
                ],
            )
            conn.execute("DELETE FROM ALPACA_MANAGED_POSITIONS WHERE id = 7")
            objects_before = conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
            sequence_before = conn.execute("SELECT name, seq FROM sqlite_sequence ORDER BY name").fetchall()

            with self.assertRaisesRegex(ValueError, "must use canonical casing"):
                init_state_db(conn)

            self.assertEqual(
                conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                objects_before,
            )
            self.assertEqual(
                conn.execute("SELECT name, seq FROM sqlite_sequence ORDER BY name").fetchall(),
                sequence_before,
            )
            conn.execute(
                """
                INSERT INTO ALPACA_MANAGED_POSITIONS
                (symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES ('UPRO', 'SPY', 30.0, 1.5, '2026-01-03',
                        'buy-after-rejection', 'accepted')
                """
            )
            self.assertEqual(
                conn.execute(
                    "SELECT id FROM ALPACA_MANAGED_POSITIONS WHERE buy_client_order_id = 'buy-after-rejection'"
                ).fetchone(),
                (8,),
            )

    def test_init_rejects_noncanonical_managed_index_and_trigger_casing_atomically(self) -> None:
        cases = (
            (
                "index",
                "alpaca_managed_positions_one_active_symbol",
                storage_module._ALPACA_ACTIVE_IDENTITY_INDEX_SQL["alpaca_managed_positions_one_active_symbol"],
            ),
            (
                "trigger",
                storage_module._ALPACA_STATE_REVISION_TRIGGER_NAME,
                storage_module._ALPACA_STATE_REVISION_TRIGGER_SQL,
            ),
        )
        for object_type, object_name, canonical_sql in cases:
            with self.subTest(object_type=object_type), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                conn.execute(f'DROP {object_type.upper()} "{object_name}"')
                conn.execute(canonical_sql.replace(object_name, object_name.upper(), 1))
                objects_before = conn.execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
                ).fetchall()

                with self.assertRaisesRegex(ValueError, "must use canonical casing"):
                    init_state_db(conn)

                self.assertEqual(
                    conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                    objects_before,
                )

    def test_noncanonical_constraint_does_not_block_supported_legacy_addition(self) -> None:
        canonical_sql = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'strategy_state'"
        ).fetchone()[0]
        legacy_sql = canonical_sql.replace(",\n        entry_date TEXT", "")
        prefix, suffix = legacy_sql.rsplit(")", 1)
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute(f"{prefix}, CHECK (cash >= 0)){suffix}")

            init_state_db(conn)

            installed_sql = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'strategy_state'"
            ).fetchone()[0]
            self.assertEqual(
                storage_module._normalized_schema_sql(installed_sql),
                storage_module._normalized_schema_sql(canonical_sql),
            )

    def test_init_rejects_existing_invalid_managed_identity_storage_values(self) -> None:
        cases = (
            ("state_revision", "-1"),
            ("state_revision", "CAST('broken' AS BLOB)"),
            ("signal_symbol", "CAST('QQQ' AS BLOB)"),
            ("buy_rsi", "CAST('30' AS BLOB)"),
            ("buy_signal_date", "CAST('2026-01-02' AS BLOB)"),
        )
        for column_name, invalid_sql in cases:
            with self.subTest(column=column_name), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                conn.execute(
                    """
                    INSERT INTO alpaca_managed_positions
                    (symbol, signal_symbol, buy_rsi, profit_target_multiple,
                     buy_signal_date, buy_client_order_id, buy_status)
                    VALUES ('TQQQ', 'QQQ', 30.0, 1.5, '2026-01-02',
                            'buy-invalid-existing', 'accepted')
                    """
                )
                conn.execute("DROP TRIGGER leveraged_trader_alpaca_managed_positions_identity_update_guard")
                conn.execute("DROP TRIGGER alpaca_managed_positions_validate_revision_update")
                conn.execute(f"UPDATE alpaca_managed_positions SET {column_name} = {invalid_sql}")
                with self.assertRaisesRegex(
                    ValueError,
                    "invalid managed identity storage value",
                ):
                    init_state_db(conn)

    def test_init_state_db_restores_missing_strategy_identity_constraints(self) -> None:
        identities = {
            "strategy_config": (
                ("asset_symbol", "signal_symbol"),
                ("TQQQ", "QQQ"),
                ("asset_symbol", "signal_symbol", "fingerprint"),
                ("TQQQ", "QQQ", "fingerprint"),
            ),
            "strategy_state": (
                ("asset_symbol", "signal_symbol", "buy_rsi", "profit_target_multiple"),
                ("TQQQ", "QQQ", 30.0, 1.5),
                (
                    "asset_symbol",
                    "signal_symbol",
                    "buy_rsi",
                    "profit_target_multiple",
                    "start_date",
                    "last_date",
                    "cash",
                    "shares",
                    "in_position",
                    "entry_price",
                    "pending_action",
                    "prev_equity",
                    "trades_executed",
                ),
                (
                    "TQQQ",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    "2026-01-02",
                    100_000.0,
                    0.0,
                    0,
                    None,
                    "none",
                    100_000.0,
                    0,
                ),
            ),
            "strategy_summary": (
                ("asset_symbol", "signal_symbol", "buy_rsi", "profit_target_multiple"),
                ("TQQQ", "QQQ", 30.0, 1.5),
                (
                    "asset_symbol",
                    "signal_symbol",
                    "buy_rsi",
                    "profit_target_multiple",
                    "trading_days",
                    "trades_executed",
                ),
                ("TQQQ", "QQQ", 30.0, 1.5, 0, 0),
            ),
            "strategy_equity": (
                ("asset_symbol", "signal_symbol", "buy_rsi", "profit_target_multiple", "date"),
                ("TQQQ", "QQQ", 30.0, 1.5, "2026-01-02"),
                (
                    "asset_symbol",
                    "signal_symbol",
                    "buy_rsi",
                    "profit_target_multiple",
                    "date",
                    "equity",
                    "daily_return",
                    "in_position",
                    "action_executed",
                    "pending_action",
                    "trades_executed",
                ),
                (
                    "TQQQ",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    100_000.0,
                    0.0,
                    0,
                    "none",
                    "none",
                    0,
                ),
            ),
        }
        for table_name, (columns, _values, insert_columns, insert_values) in identities.items():
            with self.subTest(table=table_name):
                conn = sqlite3.connect(":memory:")
                try:
                    init_state_db(conn)
                    conn.execute(f"ALTER TABLE {table_name} RENAME TO imported_{table_name}")
                    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM imported_{table_name}")
                    conn.execute(f"DROP TABLE imported_{table_name}")

                    init_state_db(conn)

                    self.assertTrue(
                        storage_module._table_has_unique_index_for_columns(
                            conn,
                            table_name,
                            columns,
                        )
                    )
                    column_sql = ", ".join(insert_columns)
                    placeholders = ", ".join("?" for _ in insert_values)
                    conn.execute(
                        f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})",
                        insert_values,
                    )
                    with self.assertRaises(sqlite3.IntegrityError):
                        conn.execute(
                            f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})",
                            insert_values,
                        )
                finally:
                    conn.close()

    def test_init_state_db_rejects_ambiguous_duplicate_strategy_identities(self) -> None:
        identities = {
            "strategy_config": (
                ("asset_symbol", "signal_symbol"),
                ("TQQQ", "QQQ"),
            ),
            "strategy_state": (
                ("asset_symbol", "signal_symbol", "buy_rsi", "profit_target_multiple"),
                ("TQQQ", "QQQ", 30.0, 1.5),
            ),
            "strategy_summary": (
                ("asset_symbol", "signal_symbol", "buy_rsi", "profit_target_multiple"),
                ("TQQQ", "QQQ", 30.0, 1.5),
            ),
            "strategy_equity": (
                ("asset_symbol", "signal_symbol", "buy_rsi", "profit_target_multiple", "date"),
                ("TQQQ", "QQQ", 30.0, 1.5, "2026-01-02"),
            ),
        }
        for table_name, (columns, values) in identities.items():
            with self.subTest(table=table_name):
                conn = sqlite3.connect(":memory:")
                try:
                    init_state_db(conn)
                    conn.execute(f"ALTER TABLE {table_name} RENAME TO imported_{table_name}")
                    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM imported_{table_name}")
                    conn.execute(f"DROP TABLE imported_{table_name}")
                    column_sql = ", ".join(columns)
                    placeholders = ", ".join("?" for _ in values)
                    conn.executemany(
                        f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})",
                        [values, values],
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        rf"{table_name} contains duplicate strategy identities",
                    ):
                        init_state_db(conn)
                finally:
                    conn.close()

    def test_init_state_db_strategy_preflight_failure_is_side_effect_free(self) -> None:
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE TABLE strategy_config (asset_symbol TEXT, signal_symbol TEXT, fingerprint TEXT)")
            conn.executemany(
                "INSERT INTO strategy_config VALUES ('TQQQ', 'QQQ', ?)",
                [("one",), ("two",)],
            )
            objects_before = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
            rows_before = conn.execute("SELECT * FROM strategy_config").fetchall()

            with self.assertRaisesRegex(ValueError, "duplicate strategy identities"):
                init_state_db(conn)

            self.assertEqual(
                conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                objects_before,
            )
            self.assertEqual(conn.execute("SELECT * FROM strategy_config").fetchall(), rows_before)

    def test_init_state_db_mid_migration_failure_rolls_back_schema_changes(self) -> None:
        with closing(sqlite3.connect(":memory:")) as conn:
            objects_before = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()

            with (
                patch.object(
                    storage_module,
                    "_ensure_alpaca_managed_sell_fill_columns",
                    side_effect=ValueError("forced late migration failure"),
                ),
                self.assertRaisesRegex(ValueError, "forced late migration failure"),
            ):
                init_state_db(conn)

            self.assertFalse(conn.in_transaction)
            self.assertEqual(
                conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                objects_before,
            )
            self.assertNotIn(
                "strategy_state",
                {row[0] for row in conn.execute("SELECT name FROM sqlite_master")},
            )

    def test_init_outermost_savepoint_commit_failure_is_not_durable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/init-outermost-commit-failure.sqlite"
            conn = sqlite3.connect(db_path, factory=CommitFailureConnection)
            try:
                conn.fail_next_commit = True

                with self.assertRaisesRegex(sqlite3.OperationalError, "forced commit failure"):
                    init_state_db(conn)

                self.assertTrue(conn.commit_failure_observed)
                self.assertFalse(conn.in_transaction)
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute("SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall(),
                    [],
                )

    def test_init_rollback_to_failure_never_releases_partial_migration(self) -> None:
        storage_module._canonical_storage_schema_object_contracts()
        with closing(sqlite3.connect(":memory:", factory=RollbackToFailureConnection)) as conn:
            conn.failed_savepoint = "init_state_db_atomic"
            original = RuntimeError("forced migration failure")
            with (
                patch.object(
                    storage_module,
                    "_ensure_strategy_state_columns",
                    side_effect=original,
                ),
                self.assertRaises(RuntimeError) as raised,
            ):
                init_state_db(conn)

            self.assertIs(raised.exception, original)
            self.assertTrue(conn.rollback_to_failed)
            self.assertFalse(conn.release_after_failed_rollback)
            self.assertFalse(conn.in_transaction)
            self.assertEqual(
                conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall(),
                [],
            )
            self.assertTrue(any("forced rollback-to failure" in note for note in original.__notes__))

    def test_shared_accounting_savepoint_cleanup_rolls_back_after_rollback_to_failure(self) -> None:
        with closing(sqlite3.connect(":memory:", factory=RollbackToFailureConnection)) as conn:
            conn.execute("CREATE TABLE accounting_probe (value INTEGER NOT NULL)")
            conn.commit()
            savepoint = "mark_alpaca_managed_sell_filled"
            conn.execute(f"SAVEPOINT {savepoint}")
            conn.execute("INSERT INTO accounting_probe VALUES (1)")
            conn.failed_savepoint = savepoint
            original = KeyboardInterrupt("forced accounting failure")

            storage_module._rollback_and_release_savepoint(conn, savepoint, original)

            self.assertTrue(conn.rollback_to_failed)
            self.assertFalse(conn.release_after_failed_rollback)
            self.assertFalse(conn.in_transaction)
            self.assertEqual(conn.execute("SELECT * FROM accounting_probe").fetchall(), [])
            self.assertTrue(any("forced rollback-to failure" in note for note in original.__notes__))

    def test_double_rollback_failure_closes_connection_before_later_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/rollback-failure.sqlite"
            conn = sqlite3.connect(
                db_path,
                factory=RollbackAndConnectionRollbackFailureConnection,
            )
            try:
                conn.execute("CREATE TABLE accounting_probe (value INTEGER NOT NULL)")
                conn.commit()
                savepoint = "mark_alpaca_managed_sell_filled"
                conn.execute(f"SAVEPOINT {savepoint}")
                conn.execute("INSERT INTO accounting_probe VALUES (1)")
                conn.failed_savepoint = savepoint
                original = RuntimeError("forced accounting failure")

                storage_module._rollback_and_release_savepoint(
                    conn,
                    savepoint,
                    original,
                )

                self.assertTrue(conn.rollback_to_failed)
                self.assertTrue(conn.connection_rollback_failed)
                self.assertFalse(conn.release_after_failed_rollback)
                with self.assertRaisesRegex(
                    sqlite3.ProgrammingError,
                    "closed database",
                ):
                    conn.commit()
                self.assertTrue(any("forced connection rollback failure" in note for note in original.__notes__))
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute("SELECT * FROM accounting_probe").fetchall(),
                    [],
                )

    def test_init_commit_failure_rolls_back_existing_outer_transaction(self) -> None:
        with closing(sqlite3.connect(":memory:", factory=CommitFailureConnection)) as conn:
            conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
            conn.commit()
            conn.execute("INSERT INTO transaction_probe VALUES (1)")
            conn.fail_next_commit = True

            with self.assertRaisesRegex(sqlite3.OperationalError, "forced commit failure"):
                init_state_db(conn)

            self.assertTrue(conn.commit_failure_observed)
            self.assertFalse(conn.in_transaction)
            self.assertEqual(conn.execute("SELECT * FROM transaction_probe").fetchall(), [])
            self.assertIsNone(conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'strategy_state'").fetchone())

    def test_init_commit_and_rollback_failure_closes_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/init-commit-failure.sqlite"
            conn = sqlite3.connect(db_path, factory=CommitFailureConnection)
            try:
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
                conn.commit()
                conn.execute("INSERT INTO transaction_probe VALUES (1)")
                conn.fail_next_commit = True
                conn.fail_rollback_after_commit = True

                with self.assertRaisesRegex(
                    sqlite3.OperationalError,
                    "forced commit failure",
                ) as raised:
                    init_state_db(conn)

                self.assertTrue(conn.commit_failure_observed)
                self.assertTrue(conn.rollback_failure_observed)
                self.assertTrue(
                    any("forced rollback after commit failure" in note for note in raised.exception.__notes__)
                )
                with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                    conn.commit()
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute("SELECT * FROM transaction_probe").fetchall(),
                    [],
                )
                self.assertIsNone(
                    observer.execute("SELECT 1 FROM sqlite_master WHERE name = 'strategy_state'").fetchone()
                )

    def test_process_grid_commit_and_rollback_failure_closes_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/grid-commit-failure.sqlite"
            conn = sqlite3.connect(db_path, factory=CommitFailureConnection)
            try:
                init_state_db(conn)
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
                conn.commit()
                conn.execute("INSERT INTO transaction_probe VALUES (1)")
                conn.fail_next_commit = True
                conn.fail_rollback_after_commit = True

                with self.assertRaisesRegex(
                    sqlite3.OperationalError,
                    "forced commit failure",
                ) as raised:
                    process_asset_grid(
                        conn,
                        sample_strategy_data(),
                        self.cfg,
                        "TQQQ",
                        "QQQ",
                        [30.0],
                        [1.5],
                        rebuild=True,
                    )

                self.assertTrue(conn.commit_failure_observed)
                self.assertTrue(conn.rollback_failure_observed)
                self.assertTrue(
                    any("forced rollback after commit failure" in note for note in raised.exception.__notes__)
                )
                with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                    conn.commit()
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute("SELECT * FROM transaction_probe").fetchall(),
                    [],
                )
                for table_name in (
                    "market_data",
                    "strategy_config",
                    "strategy_state",
                    "strategy_summary",
                    "strategy_equity",
                ):
                    with self.subTest(table_name=table_name):
                        self.assertEqual(
                            observer.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0],
                            0,
                        )

    def test_process_grid_outermost_savepoint_commit_failure_is_not_durable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/grid-outermost-commit-failure.sqlite"
            conn = sqlite3.connect(db_path, factory=CommitFailureConnection)
            try:
                init_state_db(conn)
                conn.fail_next_commit = True

                with self.assertRaisesRegex(sqlite3.OperationalError, "forced commit failure"):
                    process_asset_grid(
                        conn,
                        sample_strategy_data(),
                        self.cfg,
                        "TQQQ",
                        "QQQ",
                        [30.0],
                        [1.5],
                        rebuild=True,
                    )

                self.assertTrue(conn.commit_failure_observed)
                self.assertFalse(conn.in_transaction)
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                for table_name in (
                    "market_data",
                    "rsi_values",
                    "strategy_config",
                    "strategy_state",
                    "strategy_summary",
                    "strategy_equity",
                ):
                    with self.subTest(table_name=table_name):
                        self.assertEqual(
                            observer.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0],
                            0,
                        )

    def test_init_state_db_rejects_orphan_sell_fill_before_mutating_schema(self) -> None:
        with closing(sqlite3.connect(":memory:")) as conn:
            init_state_db(conn)
            conn.execute("DROP TRIGGER alpaca_managed_sell_fills_validate_parent_insert")
            conn.execute(
                """
                INSERT INTO alpaca_managed_sell_fills
                (managed_position_id, alpaca_order_id, filled_qty, filled_value)
                VALUES (999, 'sell-orphan', 1, 120)
                """
            )
            objects_before = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
            rows_before = conn.execute("SELECT * FROM alpaca_managed_sell_fills").fetchall()

            with self.assertRaisesRegex(ValueError, "orphan managed position identity"):
                init_state_db(conn)

            self.assertEqual(
                conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                objects_before,
            )
            self.assertEqual(
                conn.execute("SELECT * FROM alpaca_managed_sell_fills").fetchall(),
                rows_before,
            )

    def test_init_state_db_restores_all_core_upsert_identities(self) -> None:
        identities = {
            "market_data": (
                ("symbol", "date"),
                ("TQQQ", "2026-01-02"),
                ("symbol", "date"),
                ("TQQQ", "2026-01-02"),
            ),
            "rsi_values": (
                ("signal_symbol", "rsi_period", "date"),
                ("QQQ", 14, "2026-01-02"),
                ("signal_symbol", "rsi_period", "date", "close"),
                ("QQQ", 14, "2026-01-02", 100.0),
            ),
            "market_history_removal_candidates": (
                ("symbol",),
                ("TQQQ",),
                ("symbol", "missing_dates_fingerprint", "missing_date_count", "consecutive_observations"),
                ("TQQQ", "fingerprint", 1, 1),
            ),
            "alpaca_managed_sell_fills": (
                ("managed_position_id", "alpaca_order_id"),
                (1, "sell-1"),
                ("managed_position_id", "alpaca_order_id", "filled_qty", "filled_value"),
                (1, "sell-1", 1.0, 100.0),
            ),
            "alpaca_symbol_aliases": (
                ("alpaca_asset_id", "symbol"),
                ("asset-1", "TQQQ"),
                ("alpaca_asset_id", "symbol"),
                ("asset-1", "TQQQ"),
            ),
        }
        for table_name, (columns, _values, insert_columns, insert_values) in identities.items():
            with self.subTest(table=table_name), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                conn.execute(f"ALTER TABLE {table_name} RENAME TO imported_{table_name}")
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM imported_{table_name}")
                conn.execute(f"DROP TABLE imported_{table_name}")

                init_state_db(conn)

                self.assertTrue(storage_module._table_has_unique_index_for_columns(conn, table_name, columns))
                if table_name == "alpaca_managed_sell_fills":
                    conn.execute(
                        """
                        INSERT INTO alpaca_managed_positions
                        (id, symbol, signal_symbol, buy_rsi,
                         profit_target_multiple, buy_signal_date,
                         buy_client_order_id, buy_status)
                        VALUES (1, 'TQQQ', 'QQQ', 30, 1.5,
                                '2026-01-02', 'buy-parent-1', 'accepted')
                        """
                    )
                column_sql = ", ".join(insert_columns)
                placeholders = ", ".join("?" for _ in insert_values)
                conn.execute(
                    f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})",
                    insert_values,
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})",
                        insert_values,
                    )

    def test_init_state_db_restores_managed_client_order_identities(self) -> None:
        create_sql = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'alpaca_managed_positions'"
        ).fetchone()[0]
        self.conn.execute("ALTER TABLE alpaca_managed_positions RENAME TO imported_positions")
        self.conn.execute(
            create_sql.replace(
                "buy_client_order_id TEXT NOT NULL UNIQUE",
                "buy_client_order_id TEXT NOT NULL",
            ).replace(
                "sell_client_order_id TEXT UNIQUE",
                "sell_client_order_id TEXT",
            )
        )
        self.conn.execute("DROP TABLE imported_positions")

        init_state_db(self.conn)

        self.assertTrue(
            storage_module._table_has_unique_index_for_columns(
                self.conn, "alpaca_managed_positions", ("buy_client_order_id",)
            )
        )
        self.assertTrue(
            storage_module._table_has_unique_index_for_columns(
                self.conn, "alpaca_managed_positions", ("sell_client_order_id",)
            )
        )

        with closing(sqlite3.connect(":memory:")) as legacy_conn:
            init_state_db(legacy_conn)
            legacy_sql = legacy_conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'alpaca_managed_positions'"
            ).fetchone()[0]
            legacy_conn.execute("ALTER TABLE alpaca_managed_positions RENAME TO imported_positions")
            legacy_conn.execute(legacy_sql.replace("sell_client_order_id TEXT UNIQUE,", ""))
            legacy_conn.execute("DROP TABLE imported_positions")

            init_state_db(legacy_conn)

            self.assertIn(
                "sell_client_order_id",
                {row[1] for row in legacy_conn.execute("PRAGMA table_info(alpaca_managed_positions)")},
            )
            self.assertTrue(
                storage_module._table_has_unique_index_for_columns(
                    legacy_conn,
                    "alpaca_managed_positions",
                    ("sell_client_order_id",),
                )
            )

    def test_init_state_db_rejects_non_rowid_managed_position_primary_keys(self) -> None:
        canonical_sql = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'alpaca_managed_positions'"
        ).fetchone()[0]
        variants = (
            canonical_sql.replace("PRIMARY KEY AUTOINCREMENT", "PRIMARY KEY") + " WITHOUT ROWID",
            canonical_sql.replace("PRIMARY KEY AUTOINCREMENT", "PRIMARY KEY DESC"),
        )
        for create_sql in variants:
            with self.subTest(create_sql=create_sql), closing(sqlite3.connect(":memory:")) as conn:
                conn.execute(create_sql)
                with self.assertRaisesRegex(ValueError, "rowid-backed INTEGER PRIMARY KEY"):
                    init_state_db(conn)

    def test_init_state_db_rejects_nocase_identity_constraint(self) -> None:
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE TABLE market_data (symbol TEXT, date TEXT, UNIQUE(symbol COLLATE NOCASE, date))")
            with self.assertRaisesRegex(ValueError, "incompatible unique identity"):
                init_state_db(conn)

    def test_init_state_db_rejects_wrong_declared_core_identity_affinity(self) -> None:
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute(
                """
                CREATE TABLE alpaca_managed_sell_fills (
                    managed_position_id TEXT NOT NULL,
                    alpaca_order_id TEXT NOT NULL,
                    filled_qty REAL NOT NULL,
                    filled_value REAL NOT NULL
                )
                """
            )

            with self.assertRaisesRegex(ValueError, "incompatible declared affinity"):
                init_state_db(conn)

            self.assertEqual(
                conn.execute("PRAGMA table_info(alpaca_managed_sell_fills)").fetchall(),
                [
                    (0, "managed_position_id", "TEXT", 1, None, 0),
                    (1, "alpaca_order_id", "TEXT", 1, None, 0),
                    (2, "filled_qty", "REAL", 1, None, 0),
                    (3, "filled_value", "REAL", 1, None, 0),
                ],
            )

    def test_init_state_db_rejects_extra_strict_active_owner_indexes(self) -> None:
        cases = (
            (
                "alpaca_managed_positions_one_active_symbol",
                "CREATE UNIQUE INDEX extra_owner_identity ON alpaca_managed_positions(UPPER(symbol))",
            ),
            (
                "alpaca_managed_positions_one_active_asset",
                "CREATE UNIQUE INDEX extra_owner_identity ON alpaca_managed_positions(alpaca_asset_id)",
            ),
        )
        for required_index, extra_index_sql in cases:
            with self.subTest(index=required_index), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                conn.execute(f"DROP INDEX {required_index}")
                conn.execute(extra_index_sql)

                with self.assertRaisesRegex(ValueError, "extra unique index"):
                    init_state_db(conn)

    def test_init_state_db_rejects_invalid_managed_owner_identity_values(self) -> None:
        cases = (
            ("   ", None, "QQQ", "invalid symbol identity"),
            ("tqqq", None, "QQQ", "noncanonical symbol identity"),
            ("BAD SYMBOL", None, "QQQ", "noncanonical symbol identity"),
            ("ſ", None, "QQQ", "noncanonical symbol identity"),
            ("ß", None, "QQQ", "noncanonical symbol identity"),
            ("straße", None, "QQQ", "noncanonical symbol identity"),
            ("\u00a0TQQQ\u00a0", None, "QQQ", "invalid symbol identity"),
            ("TQQQ", sqlite3.Binary(b"asset-tqqq"), "QQQ", "invalid Alpaca asset identity"),
            ("TQQQ", "   ", "QQQ", "invalid Alpaca asset identity"),
            ("TQQQ", "bad asset", "QQQ", "noncanonical Alpaca asset identity"),
            ("TQQQ", "asset_bad", "QQQ", "noncanonical Alpaca asset identity"),
            ("TQQQ", "\u00a0asset-tqqq\u00a0", "QQQ", "invalid Alpaca asset identity"),
            ("TQQQ", None, "ſ", "noncanonical signal symbol identity"),
            ("TQQQ", None, "\u00a0QQQ\u00a0", "noncanonical signal symbol identity"),
        )
        for symbol, asset_id, signal_symbol, expected_message in cases:
            with (
                self.subTest(symbol=symbol, asset_id=asset_id, signal_symbol=signal_symbol),
                closing(sqlite3.connect(":memory:")) as conn,
            ):
                init_state_db(conn)
                conn.execute("DROP TRIGGER leveraged_trader_alpaca_managed_positions_identity_insert_guard")
                conn.execute(
                    """
                    INSERT INTO alpaca_managed_positions
                    (symbol, alpaca_asset_id, signal_symbol, buy_rsi,
                     profit_target_multiple, buy_signal_date, buy_client_order_id, buy_status)
                    VALUES (?, ?, ?, 30, 1.5, '2026-01-02', 'buy-owner-id', 'accepted')
                    """,
                    (symbol, asset_id, signal_symbol),
                )

                with self.assertRaisesRegex(ValueError, expected_message):
                    init_state_db(conn)

    def test_init_state_db_rejects_noncanonical_managed_alias_identity_atomically(
        self,
    ) -> None:
        for asset_id, symbol in (
            ("bad asset", " tqqq "),
            ("asset-tqqq", "ſ"),
            ("\u00a0asset-tqqq\u00a0", "TQQQ"),
        ):
            with self.subTest(asset_id=asset_id, symbol=symbol), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                conn.execute(
                    "INSERT INTO alpaca_symbol_aliases (alpaca_asset_id, symbol) VALUES (?, ?)",
                    (asset_id, symbol),
                )
                conn.commit()
                objects_before = conn.execute(
                    "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
                ).fetchall()

                with self.assertRaisesRegex(ValueError, "noncanonical managed identity"):
                    init_state_db(conn)

                self.assertEqual(
                    conn.execute("SELECT alpaca_asset_id, symbol FROM alpaca_symbol_aliases").fetchall(),
                    [(asset_id, symbol)],
                )
                self.assertEqual(
                    conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                    objects_before,
                )

    def test_init_state_db_rejects_conflicting_active_identity_index_names(self) -> None:
        for index_name in (
            "alpaca_managed_positions_one_active_symbol",
            "alpaca_managed_positions_one_active_asset",
        ):
            with self.subTest(index=index_name), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                conn.execute(f"DROP INDEX {index_name}")
                conn.execute(f"CREATE UNIQUE INDEX {index_name} ON alpaca_managed_positions(buy_signal_date)")
                objects_before = conn.execute(
                    "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
                ).fetchall()

                with self.assertRaisesRegex(ValueError, "conflicts with the required"):
                    init_state_db(conn)

                self.assertEqual(
                    conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                    objects_before,
                )

    def test_init_rejects_spaced_quoted_active_symbol_expression_atomically(self) -> None:
        index_name = "alpaca_managed_positions_one_active_symbol"
        self.conn.execute(f"DROP INDEX {index_name}")
        self.conn.execute(
            f"""
            CREATE UNIQUE INDEX {index_name}
            ON alpaca_managed_positions(UPPER("sym bol"))
            WHERE closed_at IS NULL
            """
        )
        objects_before = self.conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

        with self.assertRaisesRegex(ValueError, "conflicts with the required"):
            init_state_db(self.conn)

        self.assertEqual(
            self.conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
            objects_before,
        )

    def test_init_accepts_quoted_parenthesized_active_identity_indexes(self) -> None:
        variants = {
            "alpaca_managed_positions_one_active_symbol": """
                CREATE UNIQUE INDEX "alpaca_managed_positions_one_active_symbol"
                ON "alpaca_managed_positions" (UPPER("symbol"))
                WHERE (("closed_at" IS NULL))
            """,
            "alpaca_managed_positions_one_active_asset": """
                CREATE UNIQUE INDEX "alpaca_managed_positions_one_active_asset"
                ON "alpaca_managed_positions" (("alpaca_asset_id"))
                WHERE (("alpaca_asset_id" IS NOT NULL) AND ("closed_at" IS NULL))
            """,
        }
        for index_name, index_sql in variants.items():
            with self.subTest(index=index_name), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                conn.execute(f"DROP INDEX {index_name}")
                conn.execute(index_sql)

                init_state_db(conn)

                self.assertTrue(
                    storage_module._active_identity_index_is_semantically_valid(
                        conn,
                        index_name,
                    )
                )

    def test_init_state_db_rejects_ambiguous_core_schema_before_side_effects(self) -> None:
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute(
                "CREATE TABLE alpaca_managed_sell_fills "
                "(managed_position_id INTEGER, alpaca_order_id TEXT, filled_qty REAL, filled_value REAL)"
            )
            conn.executemany(
                "INSERT INTO alpaca_managed_sell_fills VALUES (1, 'sell-1', 1, 100)",
                [(), ()],
            )
            objects_before = conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()

            with self.assertRaisesRegex(ValueError, "duplicate Alpaca sell-order identities"):
                init_state_db(conn)

            self.assertEqual(
                conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                objects_before,
            )

        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE TABLE market_data (symbol TEXT, date TEXT)")
            conn.execute("INSERT INTO market_data VALUES (CAST('TQQQ' AS BLOB), '2026-01-02')")
            with self.assertRaisesRegex(ValueError, "invalid identity storage type"):
                init_state_db(conn)

        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute(
                "CREATE TABLE alpaca_managed_positions "
                "(id INTEGER, buy_client_order_id TEXT, sell_client_order_id TEXT)"
            )
            with self.assertRaisesRegex(ValueError, "rowid-backed INTEGER PRIMARY KEY"):
                init_state_db(conn)

    def test_init_state_db_repairs_valid_constraint_stripped_generation_singleton(self) -> None:
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE TABLE strategy_state_generation (id INTEGER, generation INTEGER)")
            conn.execute("INSERT INTO strategy_state_generation VALUES (1, 7)")

            init_state_db(conn)

            self.assertEqual(
                conn.execute("SELECT id, generation FROM strategy_state_generation").fetchall(),
                [(1, 7)],
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO strategy_state_generation VALUES (2, 8)")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("DELETE FROM strategy_state_generation")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("UPDATE strategy_state_generation SET generation = 2 WHERE id = 1")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("INSERT OR REPLACE INTO strategy_state_generation VALUES (1, 0)")
            conn.execute("UPDATE strategy_state_generation SET generation = generation + 1 WHERE id = 1")
            self.assertEqual(
                conn.execute("SELECT generation FROM strategy_state_generation").fetchone()[0],
                8,
            )

    def test_direct_strategy_reads_reject_duplicate_identity_rows(self) -> None:
        self.process_grid(sample_strategy_data(), rebuild=True)
        self.conn.execute("ALTER TABLE strategy_state RENAME TO imported_strategy_state")
        self.conn.execute("CREATE TABLE strategy_state AS SELECT * FROM imported_strategy_state")
        self.conn.execute("DROP TABLE imported_strategy_state")
        self.conn.execute(
            """
            INSERT INTO strategy_state
            SELECT * FROM strategy_state
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            LIMIT 1
            """
        )
        self.assertIsNone(
            storage_module.load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.05,
            )
        )
        duplicate_state_rowid = self.conn.execute(
            """
            SELECT MAX(rowid) FROM strategy_state
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            """
        ).fetchone()[0]
        self.conn.execute("DELETE FROM strategy_state WHERE rowid = ?", (duplicate_state_rowid,))

        self.conn.execute("ALTER TABLE strategy_summary RENAME TO imported_strategy_summary")
        self.conn.execute("CREATE TABLE strategy_summary AS SELECT * FROM imported_strategy_summary")
        self.conn.execute("DROP TABLE imported_strategy_summary")
        self.conn.execute(
            """
            INSERT INTO strategy_summary
            SELECT * FROM strategy_summary
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
            LIMIT 1
            """
        )
        self.assertIsNone(storage_module.load_best_strategy_summary(self.conn, "TQQQ", "QQQ"))

    def test_direct_strategy_config_match_rejects_duplicate_fingerprints(self) -> None:
        self.process_grid(sample_strategy_data(), rebuild=True)
        expected_fingerprint = self.conn.execute(
            """
            SELECT fingerprint FROM strategy_config
            WHERE asset_symbol = 'TQQQ' AND signal_symbol = 'QQQ'
            """
        ).fetchone()[0]
        self.conn.execute("ALTER TABLE strategy_config RENAME TO imported_strategy_config")
        self.conn.execute("CREATE TABLE strategy_config AS SELECT * FROM imported_strategy_config")
        self.conn.execute("DROP TABLE imported_strategy_config")
        self.conn.execute(
            """
            INSERT INTO strategy_config (asset_symbol, signal_symbol, fingerprint)
            VALUES ('TQQQ', 'QQQ', 'conflicting-fingerprint')
            """
        )

        self.assertFalse(
            storage_module.strategy_config_matches_fingerprint(
                self.conn,
                "TQQQ",
                "QQQ",
                expected_fingerprint,
            )
        )

    def test_init_rejects_hostile_expression_index_atomically(self) -> None:
        with closing(sqlite3.connect(":memory:")) as conn:
            init_state_db(conn)
            conn.execute("CREATE INDEX hostile_market_expression ON market_data(json_extract(symbol, '$'))")
            objects_before = conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()

            with self.assertRaisesRegex(ValueError, "unexpected explicit index"):
                init_state_db(conn)

            self.assertEqual(
                conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                objects_before,
            )

    def test_init_rejects_owned_index_name_collision_atomically(self) -> None:
        index_name = "alpaca_managed_positions_one_active_symbol"
        self.conn.execute(f"DROP INDEX {index_name}")
        self.conn.execute(f"CREATE INDEX {index_name} ON strategy_config(fingerprint)")
        objects_before = self.conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

        with self.assertRaisesRegex(ValueError, "conflicts with the required"):
            init_state_db(self.conn)

        self.assertEqual(
            self.conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
            objects_before,
        )

    def test_partial_owned_identity_index_is_rejected_atomically(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            init_state_db(conn)
            conn.execute("ALTER TABLE strategy_state RENAME TO imported_strategy_state")
            conn.execute("CREATE TABLE strategy_state AS SELECT * FROM imported_strategy_state")
            conn.execute("DROP TABLE imported_strategy_state")
            conn.execute(
                """
                CREATE UNIQUE INDEX leveraged_trader_strategy_state_identity_unique
                ON strategy_state(asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)
                WHERE buy_rsi >= 0.0
                """
            )
            objects_before = conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()

            with self.assertRaisesRegex(ValueError, "required managed index contract"):
                init_state_db(conn)

            self.assertEqual(
                conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
                objects_before,
            )
        finally:
            conn.close()

    def test_strategy_equity_integrity_migration_does_not_bless_existing_rows(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        self.assertTrue(
            all(
                row[0] is not None
                for row in self.conn.execute("SELECT integrity_digest FROM strategy_equity").fetchall()
            )
        )
        self.conn.execute("ALTER TABLE strategy_equity DROP COLUMN integrity_digest")

        init_state_db(self.conn)

        self.assertTrue(
            all(row[0] is None for row in self.conn.execute("SELECT integrity_digest FROM strategy_equity").fetchall())
        )
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])
        reported, curves = summarize_saved_results(self.conn, workflow_assets)
        self.assertTrue(reported.empty)
        self.assertTrue(curves.empty)

        self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertTrue(
            all(
                row[0] is not None
                for row in self.conn.execute("SELECT integrity_digest FROM strategy_equity").fetchall()
            )
        )

    def test_strategy_summary_integrity_migration_does_not_bless_existing_rows(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        expected_summaries = self.conn.execute(
            "SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple"
        ).fetchall()
        self.conn.execute("ALTER TABLE strategy_summary DROP COLUMN integrity_digest")

        init_state_db(self.conn)

        self.assertTrue(
            all(row[0] is None for row in self.conn.execute("SELECT integrity_digest FROM strategy_summary").fetchall())
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )

        with patch("leveraged_trader.storage.run_grid_summary", wraps=run_grid_summary) as grid_summary:
            self.process_grid(data.iloc[-1:], rebuild=False)
        self.assertEqual(grid_summary.call_args.args[7].tolist(), [0, 0, 0, 0])
        self.assertEqual(
            self.conn.execute("SELECT * FROM strategy_summary ORDER BY buy_rsi, profit_target_multiple").fetchall(),
            expected_summaries,
        )

    def test_full_asset_history_correction_rebuilds_resumed_state(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)

        revision = data.copy()
        revision.loc[revision.index[10], ["TQQQ_Open", "TQQQ_High", "TQQQ_Low", "TQQQ_Close"]] = [20, 21, 19, 20]
        self.process_grid(revision, rebuild=False)

        stored_close = self.conn.execute(
            "SELECT close FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
            (revision.index[10].date().isoformat(),),
        ).fetchone()[0]
        summary_end_date = self.conn.execute("SELECT DISTINCT end_date FROM strategy_summary").fetchone()[0]
        self.assertEqual(stored_close, 20.0)
        self.assertEqual(summary_end_date, revision.index[-1].date().isoformat())

    def test_one_ulp_target_boundary_correction_rebuilds_authoritative_and_fallback_state(self) -> None:
        dates = pd.date_range("2026-01-02", periods=5, freq="B")
        below_target = np.nextafter(1.01, 0.0)
        initial = pd.DataFrame(index=dates)
        initial["TQQQ_Open"] = 1.0
        initial["TQQQ_High"] = below_target
        initial["TQQQ_Low"] = 1.0
        initial["TQQQ_Close"] = 1.0
        initial["TQQQ_Volume"] = 1_000.0
        for field in ("Open", "High", "Low", "Close"):
            initial[f"{RISK_FREE_SYMBOL}_{field}"] = 0.0
        initial[f"{RISK_FREE_SYMBOL}_Volume"] = 0.0
        corrected = initial.copy()
        corrected.loc[dates[3], "TQQQ_High"] = 1.01
        cfg = BacktestConfig(rsi_period=2, fee_bps=0.0, slippage_bps=0.0)

        def histories(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
            return {
                "TQQQ": frame[[column for column in frame if column.startswith("TQQQ_")]],
                RISK_FREE_SYMBOL: frame[[column for column in frame if column.startswith(f"{RISK_FREE_SYMBOL}_")]],
            }

        for authority in ("authoritative", "fallback"):
            conn = sqlite3.connect(":memory:")
            init_state_db(conn)
            try:
                process_asset_grid(
                    conn,
                    initial,
                    cfg,
                    "TQQQ",
                    "TQQQ",
                    [100.0],
                    [1.01],
                    rebuild=True,
                    authoritative_histories=histories(initial),
                )
                self.assertEqual(
                    conn.execute("SELECT trades_executed, in_position FROM strategy_state").fetchone(),
                    (1, 1),
                )
                authoritative_histories = histories(corrected)
                if authority == "fallback":
                    authoritative_histories = {
                        RISK_FREE_SYMBOL: authoritative_histories[RISK_FREE_SYMBOL],
                    }

                with patch(
                    "leveraged_trader.storage.run_grid_summary",
                    wraps=run_grid_summary,
                ) as grid_summary:
                    process_asset_grid(
                        conn,
                        corrected,
                        cfg,
                        "TQQQ",
                        "TQQQ",
                        [100.0],
                        [1.01],
                        rebuild=False,
                        authoritative_histories=authoritative_histories,
                    )

                with self.subTest(authority=authority):
                    self.assertEqual(grid_summary.call_args.args[7].tolist(), [0])
                    self.assertEqual(
                        conn.execute("SELECT trades_executed, in_position FROM strategy_state").fetchone(),
                        (3, 1),
                    )
                    self.assertEqual(
                        conn.execute(
                            "SELECT high FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
                            (dates[3].date().isoformat(),),
                        ).fetchone()[0],
                        1.01,
                    )
            finally:
                conn.close()

    def test_partial_authority_detects_fallback_signal_revision(self) -> None:
        data = sample_strategy_data(periods=20)
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(data),
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ_", "UPRO_"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(upro_data, "UPRO"),
            strategy_fingerprint=fingerprint,
        )

        revision = data.copy()
        revision.loc[revision.index[5], ["QQQ_High", "QQQ_Close"]] += 10.0
        histories = self.canonical_histories(revision)
        process_asset_grid(
            self.conn,
            revision,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories={
                "TQQQ": histories["TQQQ"],
                RISK_FREE_SYMBOL: histories[RISK_FREE_SYMBOL],
            },
            strategy_fingerprint=fingerprint,
        )

        remaining = self.conn.execute(
            "SELECT asset_symbol, signal_symbol FROM strategy_state ORDER BY asset_symbol, signal_symbol"
        ).fetchall()
        self.assertEqual(remaining, [("TQQQ", "QQQ")])

    def test_partial_authority_detects_fallback_benchmark_revision(self) -> None:
        data = sample_strategy_data(periods=20)
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(data),
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ_", "UPRO_"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(upro_data, "UPRO"),
            strategy_fingerprint=fingerprint,
        )
        before_generation = strategy_state_generation(self.conn)

        revision = data.copy()
        revision.loc[
            revision.index[5],
            [
                f"{RISK_FREE_SYMBOL}_Open",
                f"{RISK_FREE_SYMBOL}_High",
                f"{RISK_FREE_SYMBOL}_Low",
                f"{RISK_FREE_SYMBOL}_Close",
            ],
        ] = 3.0
        histories = self.canonical_histories(revision)
        process_asset_grid(
            self.conn,
            revision,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories={
                "TQQQ": histories["TQQQ"],
                "QQQ": histories["QQQ"],
            },
            strategy_fingerprint=fingerprint,
        )

        upro_count = self.conn.execute("SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = 'UPRO'").fetchone()[0]
        self.assertEqual(upro_count, 0)
        self.assertEqual(strategy_state_generation(self.conn), before_generation + 1)

    def test_partial_authority_fallback_asset_revision_invalidates_every_dependency_role(self) -> None:
        data = sample_strategy_data(periods=20)
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(data),
            strategy_fingerprint=fingerprint,
        )

        spy_data = data.copy()
        for field in ("Open", "High", "Low", "Close", "Volume"):
            spy_data[f"SPY_{field}"] = data[f"QQQ_{field}"]
        spy_histories = {
            "TQQQ": spy_data[[column for column in spy_data if column.startswith("TQQQ_")]],
            "SPY": spy_data[[column for column in spy_data if column.startswith("SPY_")]],
            RISK_FREE_SYMBOL: spy_data[[column for column in spy_data if column.startswith(f"{RISK_FREE_SYMBOL}_")]],
        }
        process_asset_grid(
            self.conn,
            spy_data,
            self.cfg,
            "TQQQ",
            "SPY",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=spy_histories,
            strategy_fingerprint=fingerprint,
        )

        upro_data = pd.DataFrame(index=data.index)
        for field in ("Open", "High", "Low", "Close", "Volume"):
            upro_data[f"UPRO_{field}"] = data[f"TQQQ_{field}"]
            upro_data[f"TQQQ_{field}"] = data[f"TQQQ_{field}"]
            upro_data[f"{RISK_FREE_SYMBOL}_{field}"] = data[f"{RISK_FREE_SYMBOL}_{field}"]
        upro_histories = {
            symbol: upro_data[[column for column in upro_data if column.startswith(f"{symbol}_")]]
            for symbol in ("UPRO", "TQQQ", RISK_FREE_SYMBOL)
        }
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "TQQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=upro_histories,
            strategy_fingerprint=fingerprint,
        )

        revision = data.copy()
        revision.loc[
            revision.index[5],
            ["TQQQ_Open", "TQQQ_High", "TQQQ_Low", "TQQQ_Close"],
        ] = [80.0, 82.0, 79.0, 81.0]
        histories = self.canonical_histories(revision)
        process_asset_grid(
            self.conn,
            revision,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories={
                "QQQ": histories["QQQ"],
                RISK_FREE_SYMBOL: histories[RISK_FREE_SYMBOL],
            },
            strategy_fingerprint=fingerprint,
        )

        remaining = self.conn.execute(
            "SELECT asset_symbol, signal_symbol FROM strategy_state ORDER BY asset_symbol, signal_symbol"
        ).fetchall()
        self.assertEqual(remaining, [("TQQQ", "QQQ")])

    def test_authoritative_asset_tail_append_invalidates_signal_role_through_lead_horizon(self) -> None:
        initial = sample_strategy_data(periods=18)
        through_dependent_session = sample_strategy_data(periods=19)
        appended = sample_strategy_data(periods=20)
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            initial,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(initial),
            strategy_fingerprint=fingerprint,
        )

        upro_history = through_dependent_session[
            [column for column in through_dependent_session if column.startswith("TQQQ_")]
        ].rename(columns=lambda column: column.replace("TQQQ_", "UPRO_"))
        tqqq_signal_history = through_dependent_session[
            [column for column in through_dependent_session if column.startswith("TQQQ_")]
        ]
        dependent_risk_free = through_dependent_session[
            [column for column in through_dependent_session if column.startswith(f"{RISK_FREE_SYMBOL}_")]
        ]
        upro_data = pd.concat(
            [upro_history, tqqq_signal_history, dependent_risk_free],
            axis=1,
            join="outer",
        )
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "TQQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories={
                "UPRO": upro_history,
                "TQQQ": tqqq_signal_history,
                RISK_FREE_SYMBOL: dependent_risk_free,
            },
            strategy_fingerprint=fingerprint,
        )
        dependent_last_date = self.conn.execute(
            "SELECT last_date FROM strategy_state WHERE asset_symbol = 'UPRO'"
        ).fetchone()[0]
        self.assertEqual(
            dependent_last_date,
            through_dependent_session.index[-1].date().isoformat(),
        )

        appended_asset = appended[[column for column in appended if column.startswith("TQQQ_")]]
        lagging_signal = initial[[column for column in initial if column.startswith("QQQ_")]]
        appended_risk_free = appended[[column for column in appended if column.startswith(f"{RISK_FREE_SYMBOL}_")]]
        current_data = pd.concat(
            [appended_asset, lagging_signal, appended_risk_free],
            axis=1,
            join="outer",
        )
        process_asset_grid(
            self.conn,
            current_data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories={
                "TQQQ": appended_asset,
                "QQQ": lagging_signal,
                RISK_FREE_SYMBOL: appended_risk_free,
            },
            strategy_fingerprint=fingerprint,
        )

        remaining = self.conn.execute(
            "SELECT asset_symbol, signal_symbol FROM strategy_state ORDER BY asset_symbol, signal_symbol"
        ).fetchall()
        self.assertEqual(remaining, [("TQQQ", "QQQ")])

    def test_partial_authority_weekday_signal_backfill_preserves_prior_asset_state(self) -> None:
        complete = sample_strategy_data(periods=21)
        dependent_end = complete.index[-3]
        missing_signal_date = complete.index[-2]
        upro_history = complete.loc[:dependent_end][
            [column for column in complete if column.startswith("TQQQ_")]
        ].rename(columns=lambda column: column.replace("TQQQ_", "UPRO_"))
        tqqq_signal_history = complete[[column for column in complete if column.startswith("TQQQ_")]].drop(
            index=missing_signal_date
        )
        dependent_risk_free = complete.loc[:dependent_end][
            [column for column in complete if column.startswith(f"{RISK_FREE_SYMBOL}_")]
        ]
        upro_data = pd.concat(
            [upro_history, tqqq_signal_history, dependent_risk_free],
            axis=1,
            join="outer",
        )
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "TQQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories={
                "UPRO": upro_history,
                "TQQQ": tqqq_signal_history,
                RISK_FREE_SYMBOL: dependent_risk_free,
            },
            strategy_fingerprint=fingerprint,
        )
        alignments = storage_module._saved_signal_dependent_alignments(
            self.conn,
            "TQQQ",
        )
        self.assertEqual(
            alignments,
            {
                (
                    "UPRO",
                    dependent_end.date().isoformat(),
                ): dependent_end.date().isoformat()
            },
        )

        tqqq_complete = complete[[column for column in complete if column.startswith("TQQQ_")]]
        qqq_complete = complete[[column for column in complete if column.startswith("QQQ_")]]
        risk_free_complete = complete[[column for column in complete if column.startswith(f"{RISK_FREE_SYMBOL}_")]]
        current_data = pd.concat(
            [tqqq_complete, qqq_complete, risk_free_complete],
            axis=1,
        )
        process_asset_grid(
            self.conn,
            current_data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            # TQQQ is intentionally a fallback input here: authoritative
            # synchronization would independently identify Jan 29 as an
            # insertion behind the already-saved Jan 30 tail.
            authoritative_histories={
                "QQQ": qqq_complete,
                RISK_FREE_SYMBOL: risk_free_complete,
            },
            strategy_fingerprint=fingerprint,
        )

        remaining = self.conn.execute(
            "SELECT asset_symbol, signal_symbol FROM strategy_state ORDER BY asset_symbol, signal_symbol"
        ).fetchall()
        self.assertEqual(remaining, [("TQQQ", "QQQ"), ("UPRO", "TQQQ")])

    def test_partial_authority_detects_historical_fallback_asset_addition(self) -> None:
        data = sample_strategy_data(periods=20)
        missing_date = data.index[5]
        histories = self.canonical_histories(data)
        histories["TQQQ"] = histories["TQQQ"].drop(index=missing_date)
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=histories,
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0],
            len(data) - 1,
        )

        complete_histories = self.canonical_histories(data)
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories={
                "QQQ": complete_histories["QQQ"],
                RISK_FREE_SYMBOL: complete_histories[RISK_FREE_SYMBOL],
            },
        )

        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0],
            len(data),
        )

    def test_partial_authority_detects_historical_fallback_signal_addition(self) -> None:
        data = sample_strategy_data(periods=20)
        missing_date = data.index[5]
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        histories = self.canonical_histories(data)
        histories["QQQ"] = histories["QQQ"].drop(index=missing_date)
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=histories,
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ_", "UPRO_"))
        upro_histories = self.canonical_histories(upro_data, "UPRO")
        upro_histories["QQQ"] = upro_histories["QQQ"].drop(index=missing_date)
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=upro_histories,
            strategy_fingerprint=fingerprint,
        )

        complete_histories = self.canonical_histories(data)
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories={
                "TQQQ": complete_histories["TQQQ"],
                RISK_FREE_SYMBOL: complete_histories[RISK_FREE_SYMBOL],
            },
            strategy_fingerprint=fingerprint,
        )

        remaining = self.conn.execute(
            "SELECT asset_symbol, signal_symbol FROM strategy_state ORDER BY asset_symbol, signal_symbol"
        ).fetchall()
        persisted_signal_date = self.conn.execute(
            "SELECT COUNT(*) FROM market_data WHERE symbol = 'QQQ' AND date = ?",
            (missing_date.date().isoformat(),),
        ).fetchone()[0]
        self.assertEqual(remaining, [("TQQQ", "QQQ")])
        self.assertEqual(persisted_signal_date, 1)

    def test_partial_authority_detects_historical_fallback_benchmark_addition(self) -> None:
        data = sample_strategy_data(periods=20)
        missing_date = data.index[5]
        histories = self.canonical_histories(data)
        histories[RISK_FREE_SYMBOL] = histories[RISK_FREE_SYMBOL].drop(index=missing_date)
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=histories,
        )
        before_generation = strategy_state_generation(self.conn)

        complete_histories = self.canonical_histories(data)
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories={
                "TQQQ": complete_histories["TQQQ"],
                "QQQ": complete_histories["QQQ"],
            },
        )

        self.assertEqual(strategy_state_generation(self.conn), before_generation + 1)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_state").fetchone()[0],
            1,
        )

    def test_authoritative_asset_history_removes_deleted_session_and_rebuilds(self) -> None:
        data = sample_strategy_data()
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(data),
        )

        removed_date = data.index[10]
        reduced_data = data.drop(index=removed_date)
        process_asset_grid(
            self.conn,
            reduced_data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=self.canonical_histories(reduced_data),
        )

        persisted = self.conn.execute(
            "SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
            (removed_date.date().isoformat(),),
        ).fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]
        self.assertEqual(persisted, 0)
        self.assertEqual(equity_rows, len(reduced_data))

    def test_confirmed_earliest_asset_session_removal_rebuilds_strategy_state(self) -> None:
        data = sample_strategy_data()
        histories = self.canonical_histories(data)
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=histories,
        )
        shortened_data = data.iloc[1:]
        shortened_histories = self.canonical_histories(shortened_data)

        process_asset_grid(
            self.conn,
            shortened_data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=shortened_histories,
        )
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0], len(data))

        process_asset_grid(
            self.conn,
            shortened_data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=shortened_histories,
        )

        start_date = self.conn.execute("SELECT start_date FROM strategy_state").fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]
        self.assertEqual(start_date, shortened_data.index[0].date().isoformat())
        self.assertEqual(equity_rows, len(shortened_data))

    def test_authoritative_asset_history_backfill_rebuilds_from_the_missing_session(self) -> None:
        data = sample_strategy_data()
        backfilled_date = data.index[10]
        incomplete_data = data.drop(index=backfilled_date)
        process_asset_grid(
            self.conn,
            incomplete_data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(incomplete_data),
        )

        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=self.canonical_histories(data),
        )

        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]
        self.assertEqual(equity_rows, len(data))

    def test_authoritative_benchmark_history_removal_invalidates_other_assets(self) -> None:
        data = sample_strategy_data()
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(data),
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(upro_data, "UPRO"),
            strategy_fingerprint=fingerprint,
        )

        reduced_risk_free = data.drop(index=data.index[10])[
            [column for column in data if column.startswith(f"{RISK_FREE_SYMBOL}_")]
        ]
        histories = self.canonical_histories(data)
        histories[RISK_FREE_SYMBOL] = reduced_risk_free
        process_asset_grid(
            self.conn,
            data.drop(index=data.index[10]),
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=histories,
            strategy_fingerprint=fingerprint,
        )

        upro_state_count = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = 'UPRO'"
        ).fetchone()[0]
        self.assertEqual(upro_state_count, 0)

    def test_authoritative_signal_history_removal_invalidates_all_signal_dependents(self) -> None:
        data = sample_strategy_data()
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(data),
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(upro_data, "UPRO"),
            strategy_fingerprint=fingerprint,
        )

        removed_date = data.index[10]
        histories = self.canonical_histories(data)
        histories["QQQ"] = histories["QQQ"].drop(index=removed_date)
        process_asset_grid(
            self.conn,
            data.drop(index=removed_date),
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=histories,
            strategy_fingerprint=fingerprint,
        )

        upro_state_count = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = 'UPRO' AND signal_symbol = 'QQQ'"
        ).fetchone()[0]
        persisted_signal = self.conn.execute(
            "SELECT COUNT(*) FROM market_data WHERE symbol = 'QQQ' AND date = ?",
            (removed_date.date().isoformat(),),
        ).fetchone()[0]
        self.assertEqual(upro_state_count, 0)
        self.assertEqual(persisted_signal, 0)

    def test_saved_best_curve_uses_the_same_forward_filled_benchmark_calendar(self) -> None:
        data = sample_strategy_data()
        missing_benchmark_date = data.index[10]
        histories = self.canonical_histories(data)
        histories[RISK_FREE_SYMBOL] = histories[RISK_FREE_SYMBOL].drop(index=missing_benchmark_date)

        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=histories,
        )

        equity_dates = {row[0] for row in self.conn.execute("SELECT date FROM strategy_equity").fetchall()}
        summary_trading_days = self.conn.execute("SELECT trading_days FROM strategy_summary").fetchone()[0]

        self.assertIn(missing_benchmark_date.date().isoformat(), equity_dates)
        self.assertEqual(len(equity_dates), len(data))
        self.assertEqual(summary_trading_days, len(data))

    def test_benchmark_invalidation_advances_the_persisted_generation(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        before_generation = strategy_state_generation(self.conn)

        revision = data.copy()
        revision.loc[
            revision.index[10],
            [
                f"{RISK_FREE_SYMBOL}_Open",
                f"{RISK_FREE_SYMBOL}_High",
                f"{RISK_FREE_SYMBOL}_Low",
                f"{RISK_FREE_SYMBOL}_Close",
            ],
        ] = 3.0
        self.process_grid(revision, rebuild=False)

        self.assertEqual(strategy_state_generation(self.conn), before_generation + 1)

    def test_risk_free_history_correction_invalidates_other_assets(self) -> None:
        data = sample_strategy_data()
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )

        revision = data.copy()
        revision.loc[
            revision.index[10],
            [
                f"{RISK_FREE_SYMBOL}_Open",
                f"{RISK_FREE_SYMBOL}_High",
                f"{RISK_FREE_SYMBOL}_Low",
                f"{RISK_FREE_SYMBOL}_Close",
            ],
        ] = 3.0
        process_asset_grid(
            self.conn,
            revision,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            strategy_fingerprint=fingerprint,
        )

        upro_state_count = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = 'UPRO'"
        ).fetchone()[0]
        self.assertEqual(upro_state_count, 0)

    def test_signal_correction_invalidates_other_assets_using_that_signal(self) -> None:
        data = sample_strategy_data()
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0, 70.0], [1.05, 1.50])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0, 70.0],
            [1.05, 1.50],
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0, 70.0],
            [1.05, 1.50],
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )

        revision = data.iloc[-1:].copy()
        revision.loc[:, "QQQ_Close"] = 200.0
        revision.loc[:, "QQQ_High"] = 201.0
        process_asset_grid(
            self.conn,
            revision,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0, 70.0],
            [1.05, 1.50],
            rebuild=False,
            strategy_fingerprint=fingerprint,
        )

        upro_states = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = 'UPRO' AND signal_symbol = 'QQQ'"
        ).fetchone()[0]
        self.assertEqual(upro_states, 0)

        process_asset_grid(
            self.conn,
            upro_data.iloc[-1:],
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0, 70.0],
            [1.05, 1.50],
            rebuild=False,
            strategy_fingerprint=fingerprint,
        )
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "UPRO",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )

    def test_short_lived_asset_rebuild_preserves_canonical_signal_rsi_history(self) -> None:
        data = sample_strategy_data()
        signal_history = data[[column for column in data.columns if column.startswith("QQQ_")]].copy()
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            signal_history=signal_history,
        )
        short_data = data.iloc[-12:].rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            short_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            signal_history=signal_history,
        )

        rsi_count = self.conn.execute(
            "SELECT COUNT(*) FROM rsi_values WHERE signal_symbol = 'QQQ' AND rsi_period = ?",
            (self.cfg.rsi_period,),
        ).fetchone()[0]
        self.assertEqual(rsi_count, len(signal_history))

    def test_strategy_state_requires_exact_grid_and_backtest_fingerprint(self) -> None:
        data = sample_strategy_data()
        buy_rsi_values = [30.0, 70.0]
        profit_target_values = [1.05, 1.50]
        fingerprint = strategy_config_fingerprint(self.cfg, buy_rsi_values, profit_target_values)
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )

        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                buy_rsi_values,
                profit_target_values,
            )
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [25.0, 65.0],
                profit_target_values,
            )
        )

        changed_grid = [25.0, 65.0]
        process_asset_grid(
            self.conn,
            data.iloc[-1:],
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=changed_grid,
            profit_target_values=profit_target_values,
            rebuild=False,
        )
        actual_pairs = {
            (float(buy_rsi), float(profit_target_multiple))
            for buy_rsi, profit_target_multiple in self.conn.execute(
                "SELECT buy_rsi, profit_target_multiple FROM strategy_state"
            ).fetchall()
        }
        self.assertEqual(
            actual_pairs,
            {(buy_rsi, target) for buy_rsi in changed_grid for target in profit_target_values},
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                BacktestConfig(rsi_period=4),
                changed_grid,
                profit_target_values,
            )
        )

    def test_canonical_risk_free_provenance_rejects_coordinated_redigest_and_rebuilds(
        self,
    ) -> None:
        data = sample_strategy_data()
        buy_rsi_values = [30.0, 70.0]
        profit_target_values = [1.05, 1.5]
        self.process_grid(data, rebuild=True)
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                buy_rsi_values,
                profit_target_values,
            )
        )

        buy_rsi, target = self.rewrite_retained_risk_free_returns(
            lambda row_index, value: float(value) + (0.001 if row_index % 2 else -0.001)
        )
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])

        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=self.cfg.rsi_period,
                allow_unbound_backtest_config=True,
            )
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                buy_rsi_values,
                profit_target_values,
            )
        )
        self.assertTrue(
            summarize_saved_results(
                self.conn,
                workflow_assets,
                rsi_period=self.cfg.rsi_period,
                rsi_entry_rule="lower",
                base_cfg=self.cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            )[0].empty
        )

        self.process_grid(data.iloc[-1:], rebuild=False)

        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                buy_rsi_values,
                profit_target_values,
            )
        )
        self.assertEqual(
            len(
                summarize_saved_results(
                    self.conn,
                    workflow_assets,
                    rsi_period=self.cfg.rsi_period,
                    rsi_entry_rule="lower",
                    base_cfg=self.cfg,
                    expected_buy_rsi_values=buy_rsi_values,
                    expected_profit_target_values=profit_target_values,
                    expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
                )[0]
            ),
            1,
        )

    def test_nonpositive_canonical_signal_close_invalidates_resume_and_reports(self) -> None:
        data = sample_strategy_data()
        buy_rsi_values = [30.0, 70.0]
        profit_target_values = [1.05, 1.5]
        self.process_grid(data, rebuild=True)
        buy_rsi, target = self.conn.execute(
            "SELECT DISTINCT buy_rsi, profit_target_multiple FROM strategy_equity"
        ).fetchone()
        self.conn.execute("UPDATE market_data SET close = close - 200.0 WHERE symbol = 'QQQ'")

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                buy_rsi_values,
                profit_target_values,
            )
        )
        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=self.cfg.rsi_period,
                base_cfg=self.cfg,
                expected_buy_rsi_values=[30.0, 70.0],
                expected_profit_target_values=[1.05, 1.5],
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            )
        )
        self.assertTrue(
            summarize_saved_results(
                self.conn,
                pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}]),
                rsi_period=self.cfg.rsi_period,
                rsi_entry_rule="lower",
                base_cfg=self.cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            )[0].empty
        )

    def test_complete_curve_requires_exact_canonical_risk_free_value(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        buy_rsi, target = self.conn.execute(
            "SELECT DISTINCT buy_rsi, profit_target_multiple FROM strategy_equity"
        ).fetchone()
        first_row = self.conn.execute(
            """
            SELECT date, equity, daily_return, risk_free_return, in_position,
                   action_executed, pending_action, trades_executed
            FROM strategy_equity
            ORDER BY date
            LIMIT 1
            """
        ).fetchone()
        forged_risk_free = np.nextafter(float(first_row[3]), math.inf)
        storage_module.save_equity_records(
            self.conn,
            [
                {
                    "asset_symbol": "TQQQ",
                    "signal_symbol": "QQQ",
                    "buy_rsi": float(buy_rsi),
                    "profit_target_multiple": float(target),
                    "date": str(first_row[0]),
                    "equity": float(first_row[1]),
                    "daily_return": float(first_row[2]),
                    "risk_free_return": forged_risk_free,
                    "in_position": int(first_row[4]),
                    "action_executed": str(first_row[5]),
                    "pending_action": str(first_row[6]),
                    "trades_executed": int(first_row[7]),
                }
            ],
        )

        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=self.cfg.rsi_period,
                base_cfg=self.cfg,
                expected_buy_rsi_values=[30.0, 70.0],
                expected_profit_target_values=[1.05, 1.5],
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            )
        )

    def test_complete_curve_rejects_material_high_capital_held_valuation_gap(self) -> None:
        cfg = BacktestConfig(initial_capital=1e15, rsi_period=3)
        process_asset_grid(
            self.conn,
            sample_strategy_data(),
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=True,
        )
        self.assertIsNotNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
                rsi_period=cfg.rsi_period,
                base_cfg=cfg,
                expected_buy_rsi_values=[30.0],
                expected_profit_target_values=[2.0],
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            )
        )
        rows = self.conn.execute(
            """
            SELECT date, equity, daily_return, risk_free_return, in_position,
                   action_executed, pending_action, trades_executed
            FROM strategy_equity
            ORDER BY date
            """
        ).fetchall()
        changed_records = []
        for row_index, row in enumerate(rows):
            if row[4] != 1 or row[5] == "buy" or row_index == len(rows) - 1:
                continue
            changed_records.append(
                {
                    "asset_symbol": "TQQQ",
                    "signal_symbol": "QQQ",
                    "buy_rsi": 30.0,
                    "profit_target_multiple": 2.0,
                    "date": str(row[0]),
                    "equity": float(row[1]) + 500.0,
                    "daily_return": float(row[2]),
                    "risk_free_return": (None if row[3] is None else float(row[3])),
                    "in_position": int(row[4]),
                    "action_executed": str(row[5]),
                    "pending_action": str(row[6]),
                    "trades_executed": int(row[7]),
                }
            )
        self.assertTrue(changed_records)
        storage_module.save_equity_records(self.conn, changed_records)

        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
                rsi_period=cfg.rsi_period,
                base_cfg=cfg,
                expected_buy_rsi_values=[30.0],
                expected_profit_target_values=[2.0],
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            )
        )

    def test_complete_curve_accepts_exact_held_state_after_subnormal_entry_close(self) -> None:
        minimum_subnormal = float(np.nextafter(0.0, np.inf))
        dates = pd.date_range("2026-01-01", periods=4, freq="B")
        asset_open = np.asarray([1.0, 1.0, 1.0, minimum_subnormal])
        asset_close = np.asarray([1.0, 1.0, minimum_subnormal, minimum_subnormal])
        signal_close = np.asarray([100.0, 90.0, 90.0, 90.0])
        risk_free_yield = np.full(4, 5.0)
        data = pd.DataFrame(
            {
                "TQQQ_Open": asset_open,
                "TQQQ_High": np.maximum(asset_open, asset_close),
                "TQQQ_Low": np.minimum(asset_open, asset_close),
                "TQQQ_Close": asset_close,
                "TQQQ_Volume": 1,
                "QQQ_Open": signal_close,
                "QQQ_High": signal_close,
                "QQQ_Low": signal_close,
                "QQQ_Close": signal_close,
                "QQQ_Volume": 1,
                f"{RISK_FREE_SYMBOL}_Open": risk_free_yield,
                f"{RISK_FREE_SYMBOL}_High": risk_free_yield,
                f"{RISK_FREE_SYMBOL}_Low": risk_free_yield,
                f"{RISK_FREE_SYMBOL}_Close": risk_free_yield,
                f"{RISK_FREE_SYMBOL}_Volume": 0,
            },
            index=dates,
        )
        cfg = BacktestConfig(
            initial_capital=1.0,
            rsi_period=1,
            fee_bps=0.0,
            slippage_bps=0.0,
        )

        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=True,
        )

        curve = storage_module.load_complete_strategy_equity_curve(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            rsi_period=1,
            base_cfg=cfg,
            expected_buy_rsi_values=[30.0],
            expected_profit_target_values=[2.0],
            expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
        )
        self.assertIsNotNone(curve)
        np.testing.assert_array_equal(
            curve["equity"].to_numpy(),
            np.asarray([1.0, 1.0, minimum_subnormal, minimum_subnormal]),
        )
        state = storage_module.load_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            rsi_period=1,
        )
        self.assertIsNotNone(state)
        self.assertEqual(state["shares"], 1.0)
        self.assertEqual(state["prev_equity"], minimum_subnormal)

    def test_canonical_risk_free_alignment_keeps_non_asset_observation(self) -> None:
        asset_dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-06"])
        benchmark_dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
        benchmark_yields = np.asarray([1.0, 10.0])
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    f"{RISK_FREE_SYMBOL}_Open": benchmark_yields,
                    f"{RISK_FREE_SYMBOL}_High": benchmark_yields,
                    f"{RISK_FREE_SYMBOL}_Low": benchmark_yields,
                    f"{RISK_FREE_SYMBOL}_Close": benchmark_yields,
                    f"{RISK_FREE_SYMBOL}_Volume": 0.0,
                },
                index=benchmark_dates,
            ),
            [RISK_FREE_SYMBOL],
        )

        actual = storage_module._canonical_risk_free_returns_for_dates(
            self.conn,
            pd.DatetimeIndex(asset_dates),
        )
        expected = np.asarray(
            [
                np.nan,
                (1.0 + 0.01) ** (1 / 252) - 1.0,
                (1.0 + 0.10) ** (1 / 252) - 1.0,
            ]
        )

        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual, expected)

    def test_missing_risk_free_history_replays_with_no_excess_observations(self) -> None:
        data = sample_strategy_data()
        risk_free_columns = [column for column in data.columns if column.startswith(f"{RISK_FREE_SYMBOL}_")]
        data = data.astype({column: float for column in risk_free_columns})
        data.loc[:, risk_free_columns] = np.nan
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
        )

        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0],
                [1.5],
            )
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT excess_return_count, excess_return_sum, excess_return_sum_squares FROM strategy_summary"
            ).fetchone(),
            (0, 0.0, 0.0),
        )
        buy_rsi, target = self.conn.execute(
            "SELECT DISTINCT buy_rsi, profit_target_multiple FROM strategy_equity"
        ).fetchone()
        self.assertIsNotNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=self.cfg.rsi_period,
                base_cfg=self.cfg,
                expected_buy_rsi_values=[30.0],
                expected_profit_target_values=[1.5],
                expected_strategy_fingerprint=self.persisted_strategy_fingerprint(),
            )
        )

    def test_complete_curve_base_config_rejects_alternate_trading_cost_replay(self) -> None:
        data = sample_strategy_data()
        buy_rsi_values = [30.0]
        profit_target_values = [1.5]
        canonical_cfg = self.cfg
        alternate_cfg = BacktestConfig(
            rsi_period=canonical_cfg.rsi_period,
            initial_capital=canonical_cfg.initial_capital,
            fee_bps=500.0,
            slippage_bps=0.0,
        )
        process_asset_grid(
            self.conn,
            data,
            canonical_cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values,
            profit_target_values,
            rebuild=True,
        )
        canonical_fingerprint = self.conn.execute("SELECT fingerprint FROM strategy_config").fetchone()[0]
        with closing(sqlite3.connect(":memory:")) as alternate_conn:
            init_state_db(alternate_conn)
            process_asset_grid(
                alternate_conn,
                data,
                alternate_cfg,
                "TQQQ",
                "QQQ",
                buy_rsi_values,
                profit_target_values,
                rebuild=True,
            )
            for table_name in ("strategy_state", "strategy_summary", "strategy_equity"):
                columns = [str(row[1]) for row in alternate_conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
                rows = alternate_conn.execute(f"SELECT {', '.join(columns)} FROM {table_name}").fetchall()
                self.conn.execute(f"DELETE FROM {table_name}")
                placeholders = ", ".join("?" for _ in columns)
                self.conn.executemany(
                    f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})",
                    rows,
                )
        self.assertEqual(
            self.conn.execute("SELECT fingerprint FROM strategy_config").fetchone()[0],
            canonical_fingerprint,
        )
        buy_rsi, target = self.conn.execute(
            "SELECT DISTINCT buy_rsi, profit_target_multiple FROM strategy_equity"
        ).fetchone()

        self.assertIsNotNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=canonical_cfg.rsi_period,
                allow_unbound_backtest_config=True,
            )
        )
        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=canonical_cfg.rsi_period,
                base_cfg=canonical_cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=canonical_fingerprint,
            )
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                canonical_cfg,
                buy_rsi_values,
                profit_target_values,
            )
        )
        self.assertTrue(
            summarize_saved_results(
                self.conn,
                pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}]),
                rsi_period=canonical_cfg.rsi_period,
                rsi_entry_rule="lower",
                base_cfg=canonical_cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=canonical_fingerprint,
            )[0].empty
        )

    def test_compact_resume_binds_state_equity_to_its_summary_endpoint(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.5],
            )
        )
        self.conn.execute(
            """
            UPDATE strategy_state
            SET cash = cash * 2.0,
                prev_equity = prev_equity * 2.0
            WHERE buy_rsi = 30.0 AND profit_target_multiple = 1.05
              AND in_position = 0
            """
        )
        self.refresh_strategy_state_integrity()

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.5],
            )
        )

    def test_compact_resume_binds_summary_first_equity_to_initial_capital(self) -> None:
        data = sample_strategy_data(periods=4)
        cfg = BacktestConfig(rsi_period=3)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=True,
        )
        self.assertEqual(
            self.conn.execute("SELECT trades_executed FROM strategy_state").fetchone(),
            (0,),
        )
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [2.0],
            )
        )
        self.conn.execute("UPDATE strategy_state SET cash = 200000.0, prev_equity = 200000.0")
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET first_equity = 200000.0,
                last_equity = 200000.0,
                running_max_equity = 200000.0
            """
        )
        self.refresh_strategy_state_integrity()
        self.refresh_strategy_summary_integrity()

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [2.0],
            )
        )

    def test_compact_resume_rejects_tiny_capital_behind_matching_fingerprint(self) -> None:
        data = sample_strategy_data(periods=4)
        persisted_cfg = BacktestConfig(rsi_period=3, initial_capital=5e-10)
        requested_cfg = BacktestConfig(rsi_period=3, initial_capital=1e-300)
        process_asset_grid(
            self.conn,
            data,
            persisted_cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=True,
        )
        self.conn.execute(
            "UPDATE strategy_config SET fingerprint = ?",
            (strategy_config_fingerprint(requested_cfg, [30.0], [2.0]),),
        )

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                requested_cfg,
                [30.0],
                [2.0],
            )
        )

    def test_compact_resume_rejects_tiny_state_summary_endpoint_mismatch(self) -> None:
        data = sample_strategy_data(periods=4)
        cfg = BacktestConfig(rsi_period=3, initial_capital=1e-10)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=True,
        )
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [2.0],
            )
        )
        self.conn.execute("UPDATE strategy_state SET cash = 5e-10, prev_equity = 5e-10")
        self.refresh_strategy_state_integrity()

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [2.0],
            )
        )

    def test_compact_resume_rejects_tiny_coordinated_zero_trade_growth(self) -> None:
        data = sample_strategy_data(periods=4)
        cfg = BacktestConfig(rsi_period=3, initial_capital=5e-10)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [2.0],
            rebuild=True,
        )
        self.conn.execute("UPDATE strategy_state SET cash = 8e-10, prev_equity = 8e-10")
        self.conn.execute(
            """
            UPDATE strategy_summary
            SET last_equity = 8e-10,
                running_max_equity = 8e-10
            """
        )
        self.refresh_strategy_state_integrity()
        self.refresh_strategy_summary_integrity()

        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                cfg,
                [30.0],
                [2.0],
            )
        )

    def test_rsi_entry_rule_is_part_of_strategy_identity(self) -> None:
        buy_rsi_values = [70.0]
        profit_target_values = [1.50]

        lower_fingerprint = strategy_config_fingerprint(
            self.cfg,
            buy_rsi_values,
            profit_target_values,
            "lower",
        )
        upper_fingerprint = strategy_config_fingerprint(
            self.cfg,
            buy_rsi_values,
            profit_target_values,
            "upper",
        )

        self.assertNotEqual(lower_fingerprint, upper_fingerprint)

    def test_upper_rsi_entry_rule_processes_high_rsi_signals(self) -> None:
        periods = 20
        dates = pd.date_range("2026-01-02", periods=periods, freq="B")
        asset_close = np.linspace(100.0, 120.0, periods)
        signal_close = np.linspace(100.0, 140.0, periods)
        data = pd.DataFrame(
            {
                "SQQQ_Open": asset_close,
                "SQQQ_High": asset_close + 1.0,
                "SQQQ_Low": asset_close - 1.0,
                "SQQQ_Close": asset_close,
                "SQQQ_Volume": 1_000_000,
                "QQQ_Open": signal_close,
                "QQQ_High": signal_close + 1.0,
                "QQQ_Low": signal_close - 1.0,
                "QQQ_Close": signal_close,
                "QQQ_Volume": 2_000_000,
                f"{RISK_FREE_SYMBOL}_Open": 5.0,
                f"{RISK_FREE_SYMBOL}_High": 5.0,
                f"{RISK_FREE_SYMBOL}_Low": 5.0,
                f"{RISK_FREE_SYMBOL}_Close": 5.0,
                f"{RISK_FREE_SYMBOL}_Volume": 0,
            },
            index=dates,
        )

        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "SQQQ",
            "QQQ",
            [70.0],
            [5.0],
            rebuild=True,
            rsi_entry_rule="upper",
        )

        trades, equity_rows = self.conn.execute(
            """
            SELECT s.trades_executed, COUNT(e.date)
            FROM strategy_state s
            JOIN strategy_equity e
              ON e.asset_symbol = s.asset_symbol
             AND e.signal_symbol = s.signal_symbol
             AND e.buy_rsi = s.buy_rsi
             AND e.profit_target_multiple = s.profit_target_multiple
            WHERE s.asset_symbol = 'SQQQ'
            GROUP BY s.trades_executed
            """
        ).fetchone()
        self.assertGreater(trades, 0)
        self.assertEqual(equity_rows, periods)

    def test_authenticated_curve_and_reports_require_the_exact_full_grid_fingerprint(
        self,
    ) -> None:
        data = sample_strategy_data()
        buy_rsi_values = [30.0]
        profit_target_values = [1.5]
        canonical_cfg = BacktestConfig(
            rsi_period=3,
            fee_bps=10.0,
            slippage_bps=20.0,
            auto_adjust=True,
        )
        process_asset_grid(
            self.conn,
            data,
            canonical_cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values,
            profit_target_values,
            rebuild=True,
        )
        canonical_fingerprint = strategy_config_fingerprint(
            canonical_cfg,
            buy_rsi_values,
            profit_target_values,
        )
        buy_rsi, target = self.conn.execute(
            "SELECT DISTINCT buy_rsi, profit_target_multiple FROM strategy_equity"
        ).fetchone()
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])

        self.assertIsNotNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=canonical_cfg.rsi_period,
                base_cfg=canonical_cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=canonical_fingerprint,
            )
        )
        canonical_summary, canonical_curves = summarize_saved_results(
            self.conn,
            workflow_assets,
            rsi_period=canonical_cfg.rsi_period,
            rsi_entry_rule="lower",
            base_cfg=canonical_cfg,
            expected_buy_rsi_values=buy_rsi_values,
            expected_profit_target_values=profit_target_values,
        )
        self.assertFalse(canonical_summary.empty)
        self.assertFalse(canonical_curves.empty)
        self.assertIsNotNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=canonical_cfg.rsi_period,
                base_cfg=canonical_cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
            )
        )

        alternate_configs = (
            BacktestConfig(
                rsi_period=3,
                fee_bps=10.0,
                slippage_bps=20.0,
                auto_adjust=False,
            ),
            BacktestConfig(
                rsi_period=3,
                fee_bps=20.0,
                slippage_bps=10.0,
                auto_adjust=True,
            ),
        )
        for alternate_cfg in alternate_configs:
            with self.subTest(alternate_cfg=alternate_cfg):
                alternate_fingerprint = strategy_config_fingerprint(
                    alternate_cfg,
                    buy_rsi_values,
                    profit_target_values,
                )
                self.assertNotEqual(alternate_fingerprint, canonical_fingerprint)
                # A valid persisted token cannot authenticate unrelated caller
                # configuration: the loader first derives the identity from the
                # supplied config and complete grid.
                self.assertIsNone(
                    storage_module.load_complete_strategy_equity_curve(
                        self.conn,
                        "TQQQ",
                        "QQQ",
                        buy_rsi,
                        target,
                        rsi_period=alternate_cfg.rsi_period,
                        base_cfg=alternate_cfg,
                        expected_buy_rsi_values=buy_rsi_values,
                        expected_profit_target_values=profit_target_values,
                        expected_strategy_fingerprint=canonical_fingerprint,
                    )
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "does not match the supplied backtest configuration",
                ):
                    summarize_saved_results(
                        self.conn,
                        workflow_assets,
                        rsi_period=alternate_cfg.rsi_period,
                        rsi_entry_rule="lower",
                        base_cfg=alternate_cfg,
                        expected_buy_rsi_values=buy_rsi_values,
                        expected_profit_target_values=profit_target_values,
                        expected_strategy_fingerprint=canonical_fingerprint,
                    )
                summary, curves = summarize_saved_results(
                    self.conn,
                    workflow_assets,
                    rsi_period=alternate_cfg.rsi_period,
                    rsi_entry_rule="lower",
                    base_cfg=alternate_cfg,
                    expected_buy_rsi_values=buy_rsi_values,
                    expected_profit_target_values=profit_target_values,
                    expected_strategy_fingerprint=alternate_fingerprint,
                )
                self.assertTrue(summary.empty)
                self.assertTrue(curves.empty)

        wrong_buy_rsi_values = [30.0, 70.0]
        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=canonical_cfg.rsi_period,
                base_cfg=canonical_cfg,
                expected_buy_rsi_values=wrong_buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=canonical_fingerprint,
            )
        )
        with self.assertRaisesRegex(ValueError, "complete strategy grid"):
            summarize_saved_results(
                self.conn,
                workflow_assets,
                rsi_period=canonical_cfg.rsi_period,
                rsi_entry_rule="lower",
                base_cfg=canonical_cfg,
                expected_strategy_fingerprint=canonical_fingerprint,
            )
        with self.assertRaisesRegex(
            ValueError,
            "does not match the supplied backtest configuration",
        ):
            summarize_saved_results(
                self.conn,
                workflow_assets,
                rsi_period=canonical_cfg.rsi_period,
                rsi_entry_rule="lower",
                base_cfg=canonical_cfg,
                expected_buy_rsi_values=wrong_buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=canonical_fingerprint,
            )
        self.conn.execute("DELETE FROM strategy_config")
        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                buy_rsi,
                target,
                rsi_period=canonical_cfg.rsi_period,
                base_cfg=canonical_cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=canonical_fingerprint,
            )
        )

    def test_authenticated_curve_requires_exact_grid_rows_and_selected_membership(
        self,
    ) -> None:
        data = sample_strategy_data()
        cfg = BacktestConfig(rsi_period=3)
        buy_rsi_values = [30.0, 70.0]
        profit_target_values = [1.5]
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values,
            profit_target_values,
            rebuild=True,
        )
        full_fingerprint = strategy_config_fingerprint(
            cfg,
            buy_rsi_values,
            profit_target_values,
        )
        self.assertIsNotNone(
            storage_module.load_complete_strategy_latest_action(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                base_cfg=cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=full_fingerprint,
            )
        )
        self.conn.execute(
            "DELETE FROM strategy_state WHERE asset_symbol = 'TQQQ' AND signal_symbol = 'QQQ' AND buy_rsi = 70",
        )
        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                base_cfg=cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=full_fingerprint,
            )
        )
        summary, curves = summarize_saved_results(
            self.conn,
            pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}]),
            base_cfg=cfg,
            expected_buy_rsi_values=buy_rsi_values,
            expected_profit_target_values=profit_target_values,
            expected_strategy_fingerprint=full_fingerprint,
            rsi_entry_rule="lower",
        )
        self.assertTrue(summary.empty)
        self.assertTrue(curves.empty)

        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values,
            profit_target_values,
            rebuild=True,
        )
        subset_buy_rsi = [30.0]
        subset_fingerprint = strategy_config_fingerprint(
            cfg,
            subset_buy_rsi,
            profit_target_values,
        )
        storage_module.save_strategy_config(
            self.conn,
            "TQQQ",
            "QQQ",
            subset_fingerprint,
        )
        for selected_buy_rsi in (30.0, 70.0):
            with self.subTest(selected_buy_rsi=selected_buy_rsi):
                self.assertIsNone(
                    storage_module.load_complete_strategy_equity_curve(
                        self.conn,
                        "TQQQ",
                        "QQQ",
                        selected_buy_rsi,
                        1.5,
                        base_cfg=cfg,
                        expected_buy_rsi_values=subset_buy_rsi,
                        expected_profit_target_values=profit_target_values,
                        expected_strategy_fingerprint=subset_fingerprint,
                    )
                )

    def test_authenticated_curve_rejects_an_adjacent_canonical_daily_return(
        self,
    ) -> None:
        data = sample_strategy_data()
        buy_rsi_values = [30.0]
        profit_target_values = [1.5]
        cfg = BacktestConfig(rsi_period=3)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values,
            profit_target_values,
            rebuild=True,
        )
        fingerprint = strategy_config_fingerprint(
            cfg,
            buy_rsi_values,
            profit_target_values,
        )
        row = self.conn.execute(
            """
            SELECT asset_symbol, signal_symbol, buy_rsi, profit_target_multiple,
                   date, equity, daily_return, risk_free_return, in_position,
                   action_executed, pending_action, trades_executed
            FROM strategy_equity
            WHERE daily_return != 0
            ORDER BY date
            LIMIT 1
            """
        ).fetchone()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertIsNotNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                str(row[0]),
                str(row[1]),
                float(row[2]),
                float(row[3]),
                rsi_period=cfg.rsi_period,
                base_cfg=cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=fingerprint,
            )
        )
        forged_daily_return = float(np.nextafter(float(row[6]), np.inf))
        self.assertNotEqual(forged_daily_return, float(row[6]))
        save_equity_records(
            self.conn,
            [
                {
                    "asset_symbol": row[0],
                    "signal_symbol": row[1],
                    "buy_rsi": row[2],
                    "profit_target_multiple": row[3],
                    "date": row[4],
                    "equity": row[5],
                    "daily_return": forged_daily_return,
                    "risk_free_return": row[7],
                    "in_position": row[8],
                    "action_executed": row[9],
                    "pending_action": row[10],
                    "trades_executed": row[11],
                }
            ],
        )

        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                str(row[0]),
                str(row[1]),
                float(row[2]),
                float(row[3]),
                rsi_period=cfg.rsi_period,
                base_cfg=cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=fingerprint,
            )
        )

    def test_authenticated_curve_rejects_adjacent_held_equity_at_max_capital(
        self,
    ) -> None:
        data = sample_strategy_data()
        buy_rsi_values = [30.0]
        profit_target_values = [1.5]
        cfg = BacktestConfig(initial_capital=1e15, rsi_period=3)
        process_asset_grid(
            self.conn,
            data,
            cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values,
            profit_target_values,
            rebuild=True,
        )
        fingerprint = strategy_config_fingerprint(
            cfg,
            buy_rsi_values,
            profit_target_values,
        )
        rows = self.conn.execute(
            """
            SELECT date, equity, daily_return, risk_free_return, in_position,
                   action_executed, pending_action, trades_executed
            FROM strategy_equity
            ORDER BY date
            """
        ).fetchall()
        row_index = next(index for index, row in enumerate(rows[1:-1], start=1) if row[4] == 1 and row[5] == "none")
        changed = list(rows[row_index])
        changed[1] = float(np.nextafter(float(changed[1]), np.inf))
        changed[2] = float(changed[1]) / float(rows[row_index - 1][1]) - 1.0
        following = list(rows[row_index + 1])
        following[2] = float(following[1]) / float(changed[1]) - 1.0
        self.assertGreater(float(changed[1]) - float(rows[row_index][1]), 0.0)
        save_equity_records(
            self.conn,
            [
                {
                    "asset_symbol": "TQQQ",
                    "signal_symbol": "QQQ",
                    "buy_rsi": 30.0,
                    "profit_target_multiple": 1.5,
                    "date": row[0],
                    "equity": row[1],
                    "daily_return": row[2],
                    "risk_free_return": row[3],
                    "in_position": row[4],
                    "action_executed": row[5],
                    "pending_action": row[6],
                    "trades_executed": row[7],
                }
                for row in (changed, following)
            ],
        )

        self.assertIsNone(
            storage_module.load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                rsi_period=cfg.rsi_period,
                base_cfg=cfg,
                expected_buy_rsi_values=buy_rsi_values,
                expected_profit_target_values=profit_target_values,
                expected_strategy_fingerprint=fingerprint,
            )
        )


class AlpacaManagedStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        init_state_db(self.conn)

    def tearDown(self) -> None:
        self.conn.close()

    def save_position(
        self,
        *,
        symbol: str,
        client_order_id: str,
        alpaca_asset_id: str | None = "asset-tqqq",
        buy_order_qty: float | None = 2,
        buy_order_limit_price: float | None = 105,
    ) -> int:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol=symbol,
            alpaca_asset_id=alpaca_asset_id,
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id=client_order_id,
            buy_alpaca_order_id=f"buy-{client_order_id}",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="filled",
            buy_order_qty=buy_order_qty,
            buy_order_limit_price=buy_order_limit_price,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
        )
        return position_id

    def close_position(self, position_id: int) -> None:
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET closed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()

    def test_neutral_sell_submission_release_rebases_current_parent_intent(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-neutral-release-rebase",
            buy_alpaca_order_id="buy-neutral-release-rebase",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=4,
            buy_order_limit_price=105,
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
        )
        sell_client_order_id = "sell-neutral-release-rebase"
        claim_token = "2026-01-02T14:32:00Z"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=None,
            sell_submitted_at=None,
            sell_status="submission_pending",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        claimed_revision = storage_module.claim_alpaca_managed_sell_submission_fence(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            claimed_at=claim_token,
            expected_remaining_qty=2,
            expected_target_sell_price=150,
            notes="claim neutral release generation",
        )
        self.assertIsNotNone(claimed_revision)
        assert claimed_revision is not None

        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=3,
                filled_avg_price=101,
                filled_at="2026-01-02T14:33:00Z",
                target_sell_price=151.5,
            )
        )
        revised_parent = self.conn.execute(
            "SELECT state_revision, sell_order_qty, sell_order_limit_price FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        self.assertGreater(revised_parent[0], claimed_revision)
        self.assertEqual(revised_parent[1:], (2.0, 150.0))

        self.assertFalse(
            storage_module.release_alpaca_managed_sell_submission_fence(
                self.conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                claimed_at=claim_token,
                expected_sell_status="submission_pending",
                expected_state_revision=claimed_revision,
                sell_status="fractional_qty",
                notes="stale fractional decision",
            )
        )
        self.assertTrue(
            storage_module.release_alpaca_managed_sell_submission_fence(
                self.conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                claimed_at=claim_token,
                expected_sell_status="submission_pending",
                expected_state_revision=claimed_revision,
                allow_current_intent_rebase=True,
                sell_status="submission_not_found",
                notes="neutral release rebased to the corrected parent",
            )
        )
        released = self.conn.execute(
            "SELECT sell_status, sell_submission_retry_claimed_at, "
            "sell_order_qty, sell_order_limit_price "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(released, ("submission_not_found", None, 3.0, 151.5))

    def test_neutral_sell_submission_release_clears_balanced_historical_fill(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="buy-neutral-release-balanced",
        )
        sell_client_order_id = "sell-neutral-release-balanced"
        claim_token = "2026-01-02T14:32:00Z"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=None,
            sell_submitted_at=None,
            sell_status="submission_pending",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        claimed_revision = storage_module.claim_alpaca_managed_sell_submission_fence(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            claimed_at=claim_token,
            expected_remaining_qty=2,
            expected_target_sell_price=150,
            notes="claim generation balanced by a historical fill",
        )
        self.assertIsNotNone(claimed_revision)
        assert claimed_revision is not None

        remaining_qty, generation_is_current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id="__lineage-fill-only__:sell-historical-balanced",
            sell_status="filled",
            sell_filled_qty=2,
            sell_filled_avg_price=150,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id="sell-historical-balanced",
        )
        before_release = self.conn.execute(
            "SELECT state_revision, sell_status, sell_alpaca_order_id, "
            "sell_submission_retry_claimed_at, sell_order_qty, sell_order_limit_price, "
            "remaining_qty, sold_qty FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        self.assertEqual(remaining_qty, 0)
        self.assertFalse(generation_is_current)
        self.assertGreater(before_release[0], claimed_revision)
        self.assertEqual(
            before_release[1:],
            ("submission_pending", None, claim_token, 2.0, 150.0, 0.0, 2.0),
        )

        self.assertFalse(
            storage_module.release_alpaca_managed_sell_submission_fence(
                self.conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                claimed_at="2026-01-02T14:31:00Z",
                expected_sell_status="submission_pending",
                expected_state_revision=claimed_revision,
                allow_current_intent_rebase=True,
                sell_status="submission_not_found",
                notes="stale neutral release after historical fill",
            )
        )
        after_stale_release = self.conn.execute(
            "SELECT sell_status, sell_submission_retry_claimed_at, sell_order_qty, "
            "sell_order_limit_price, remaining_qty FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        self.assertEqual(
            after_stale_release,
            ("submission_pending", claim_token, 2.0, 150.0, 0.0),
        )

        self.assertTrue(
            storage_module.release_alpaca_managed_sell_submission_fence(
                self.conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                claimed_at=claim_token,
                expected_sell_status="submission_pending",
                expected_state_revision=claimed_revision,
                allow_current_intent_rebase=True,
                sell_status="submission_not_found",
                notes="neutral release after historical fill balanced the parent",
            )
        )
        released = self.conn.execute(
            "SELECT sell_status, sell_alpaca_order_id, sell_submission_retry_claimed_at, "
            "sell_order_qty, sell_order_limit_price, remaining_qty, sold_qty, notes "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value FROM alpaca_managed_sell_fills "
            "WHERE managed_position_id = ? AND alpaca_order_id = ?",
            (position_id, "sell-historical-balanced"),
        ).fetchone()

        self.assertEqual(
            released,
            (
                "submission_not_found",
                None,
                None,
                2.0,
                150.0,
                0.0,
                2.0,
                "neutral release after historical fill balanced the parent",
            ),
        )
        self.assertEqual(ledger, (2.0, 300.0))

    def test_neutral_sell_submission_release_cannot_clear_a_new_token_or_attached_order(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="buy-neutral-release-ownership",
        )
        sell_client_order_id = "sell-neutral-release-ownership"
        token_a = "2026-01-03T14:30:00Z"
        token_b = "2026-01-03T14:40:00Z"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=None,
            sell_submitted_at=None,
            sell_status="submission_not_found",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        claim_a = storage_module.claim_alpaca_managed_sell_submission_retry_with_revision(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            expected_target_sell_price=150,
            claimed_at=token_a,
            reclaim_before="2026-01-03T14:29:00Z",
            notes="claim token A",
        )
        self.assertIsNotNone(claim_a)
        assert claim_a is not None
        claim_b = storage_module.claim_alpaca_managed_sell_submission_retry_with_revision(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            expected_target_sell_price=150,
            claimed_at=token_b,
            reclaim_before="2026-01-03T14:35:00Z",
            notes="reclaim with token B",
        )
        self.assertIsNotNone(claim_b)
        assert claim_b is not None

        self.assertFalse(
            storage_module.release_alpaca_managed_sell_submission_fence(
                self.conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                claimed_at=token_a,
                expected_sell_status="submission_retrying",
                expected_state_revision=claim_a.state_revision,
                allow_current_intent_rebase=True,
                sell_status="submission_not_found",
                notes="stale token A release",
            )
        )
        after_stale_release = self.conn.execute(
            "SELECT sell_status, sell_alpaca_order_id, sell_submission_retry_claimed_at, state_revision "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        self.assertEqual(
            after_stale_release,
            ("submission_retrying", None, token_b, claim_b.state_revision),
        )

        self.conn.execute(
            "UPDATE alpaca_managed_positions SET sell_alpaca_order_id = ? WHERE id = ?",
            ("sell-attached-before-release", position_id),
        )
        self.conn.commit()
        self.assertFalse(
            storage_module.release_alpaca_managed_sell_submission_fence(
                self.conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                claimed_at=token_b,
                expected_sell_status="submission_retrying",
                expected_state_revision=claim_b.state_revision,
                allow_current_intent_rebase=True,
                sell_status="submission_not_found",
                notes="release after broker attachment",
            )
        )
        after_attachment = self.conn.execute(
            "SELECT sell_status, sell_alpaca_order_id, sell_submission_retry_claimed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        self.assertEqual(
            after_attachment,
            ("submission_retrying", "sell-attached-before-release", token_b),
        )

    def test_returning_mutators_rollback_decode_failures_without_losing_caller_work(
        self,
    ) -> None:
        operation_names = (
            "buy_intent",
            "buy_confirmation",
            "buy_fill",
            "asset_adoption",
            "initial_sell_intent",
            "initial_sell_submission",
            "sell_renewal",
            "sell_replacement",
            "sell_submission_fence",
            "sell_submission_retry",
            "sell_confirmation",
        )

        for operation_name in operation_names:
            for caller_owns_transaction in (False, True):
                with (
                    self.subTest(
                        operation=operation_name,
                        caller_owns_transaction=caller_owns_transaction,
                    ),
                    closing(sqlite3.connect(":memory:", factory=ReturningFetchFailureConnection)) as conn,
                ):
                    init_state_db(conn)
                    conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
                    conn.commit()

                    def seed_buy(
                        *,
                        filled: bool,
                        asset_id: str | None = "asset-tqqq",
                        operation_name: str = operation_name,
                    ) -> int:
                        position_id = save_alpaca_managed_buy_order(
                            conn,
                            symbol="TQQQ",
                            alpaca_asset_id=asset_id,
                            signal_symbol="QQQ",
                            buy_rsi=30,
                            profit_target_multiple=1.5,
                            buy_signal_date="2026-01-02",
                            buy_client_order_id=f"returning-{operation_name}",
                            buy_alpaca_order_id=f"buy-returning-{operation_name}",
                            buy_submitted_at="2026-01-02T14:30:00Z",
                            buy_status="accepted",
                            buy_order_qty=2,
                            buy_order_limit_price=105,
                        )
                        if filled:
                            mark_alpaca_managed_buy_filled(
                                conn,
                                position_id,
                                buy_status="filled",
                                filled_qty=2,
                                filled_avg_price=100,
                                filled_at="2026-01-02T14:31:00Z",
                                target_sell_price=150,
                            )
                        return position_id

                    if operation_name == "buy_intent":

                        def invoke() -> object:
                            return storage_module.claim_alpaca_managed_buy_intent(
                                conn,
                                symbol="TQQQ",
                                signal_symbol="QQQ",
                                buy_rsi=30,
                                profit_target_multiple=1.5,
                                buy_signal_date="2026-01-02",
                                buy_client_order_id="returning-buy-intent",
                                buy_order_qty=1,
                                buy_order_limit_price=100,
                            )

                    elif operation_name == "buy_confirmation":
                        claim = storage_module.claim_alpaca_managed_buy_intent(
                            conn,
                            symbol="TQQQ",
                            signal_symbol="QQQ",
                            buy_rsi=30,
                            profit_target_multiple=1.5,
                            buy_signal_date="2026-01-02",
                            buy_client_order_id="returning-buy-confirmation",
                            buy_order_qty=1,
                            buy_order_limit_price=100,
                        )

                        def invoke(claim: object = claim) -> object:
                            return storage_module.confirm_alpaca_managed_buy_submission(
                                conn,
                                int(claim.position_id),  # type: ignore[attr-defined]
                                expected_buy_submission_attempt_count=claim.attempt_count,  # type: ignore[attr-defined]
                                expected_state_revision=claim.state_revision,  # type: ignore[attr-defined]
                                expected_buy_order_qty=1,
                                expected_buy_order_limit_price=100,
                                claimed_at="2026-01-02T14:30:01Z",
                            )

                    elif operation_name == "buy_fill":
                        position_id = seed_buy(filled=False)

                        def invoke(position_id: int = position_id) -> object:
                            return mark_alpaca_managed_buy_filled(
                                conn,
                                position_id,
                                buy_status="filled",
                                filled_qty=2,
                                filled_avg_price=100,
                                filled_at="2026-01-02T14:31:00Z",
                                target_sell_price=150,
                            )

                    elif operation_name == "asset_adoption":
                        position_id = seed_buy(filled=True, asset_id=None)
                        expected_revision = int(
                            conn.execute(
                                "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                                (position_id,),
                            ).fetchone()[0]
                        )

                        def invoke(
                            position_id: int = position_id,
                            expected_revision: int = expected_revision,
                        ) -> object:
                            return adopt_alpaca_managed_position_asset_if_current(
                                conn,
                                position_id,
                                expected_state_revision=expected_revision,
                                alpaca_asset_id="asset-adopted",
                            )

                    else:
                        position_id = seed_buy(filled=True)
                        sell_client_order_id = f"sell-client-{operation_name}"
                        sell_order_id = f"sell-{operation_name}"

                        if operation_name == "initial_sell_intent":

                            def invoke(
                                position_id: int = position_id,
                                sell_client_order_id: str = sell_client_order_id,
                            ) -> object:
                                return storage_module.claim_alpaca_managed_initial_sell_intent(
                                    conn,
                                    position_id,
                                    sell_order_namespace="returning",
                                    sell_client_order_id=sell_client_order_id,
                                    expected_remaining_qty=2,
                                    expected_target_sell_price=150,
                                    notes="claim initial sell",
                                )

                        elif operation_name == "initial_sell_submission":
                            snapshot = storage_module.alpaca_managed_sell_renewal_snapshot_if_current(
                                conn,
                                position_id,
                                expected_sell_client_order_id=None,
                                expected_sell_alpaca_order_id=None,
                            )
                            self.assertIsNotNone(snapshot)

                            def invoke(
                                position_id: int = position_id,
                                snapshot: object = snapshot,
                                sell_client_order_id: str = sell_client_order_id,
                            ) -> object:
                                return storage_module.claim_alpaca_managed_initial_sell_submission(
                                    conn,
                                    position_id,
                                    sell_order_namespace="returning",
                                    sell_client_order_id=sell_client_order_id,
                                    expected_remaining_qty=2,
                                    expected_target_sell_price=150,
                                    claimed_at="2026-01-02T14:32:00Z",
                                    expected_state_snapshot=snapshot,  # type: ignore[arg-type]
                                    notes="claim initial sell submission",
                                )

                        elif operation_name == "sell_renewal":
                            record_alpaca_managed_sell_order(
                                conn,
                                position_id,
                                sell_client_order_id=sell_client_order_id,
                                sell_alpaca_order_id=sell_order_id,
                                sell_submitted_at="2026-01-02T14:32:00Z",
                                sell_status="accepted",
                                sell_order_qty=2,
                                sell_order_limit_price=150,
                            )

                            def invoke(
                                position_id: int = position_id,
                                sell_client_order_id: str = sell_client_order_id,
                                sell_order_id: str = sell_order_id,
                            ) -> object:
                                return storage_module.claim_alpaca_managed_sell_renewal_with_revision(
                                    conn,
                                    position_id,
                                    sell_client_order_id=sell_client_order_id,
                                    sell_alpaca_order_id=sell_order_id,
                                    expected_target_sell_price=150,
                                    expected_remaining_qty=2,
                                    requested_at="2026-01-03T14:30:00Z",
                                    reclaim_before="2026-01-03T14:29:00Z",
                                    notes="claim renewal",
                                )

                        elif operation_name == "sell_replacement":
                            record_alpaca_managed_sell_order(
                                conn,
                                position_id,
                                sell_client_order_id=sell_client_order_id,
                                sell_alpaca_order_id=sell_order_id,
                                sell_submitted_at="2026-01-02T14:32:00Z",
                                sell_status="canceled",
                                sell_order_qty=2,
                                sell_order_limit_price=150,
                            )

                            def invoke(
                                position_id: int = position_id,
                                sell_client_order_id: str = sell_client_order_id,
                                sell_order_id: str = sell_order_id,
                            ) -> object:
                                return claim_alpaca_managed_sell_replacement(
                                    conn,
                                    position_id,
                                    prior_sell_client_order_id=sell_client_order_id,
                                    prior_sell_alpaca_order_id=sell_order_id,
                                    prior_renewal_count=0,
                                    replacement_sell_client_order_id=f"{sell_client_order_id}-r1",
                                    requested_remaining_qty=2,
                                    expected_target_sell_price=150,
                                    notes="claim replacement",
                                )

                        elif operation_name in {
                            "sell_submission_fence",
                            "sell_submission_retry",
                            "sell_confirmation",
                        }:
                            initial_status = (
                                "submission_not_found"
                                if operation_name == "sell_submission_retry"
                                else "submission_pending"
                            )
                            record_alpaca_managed_sell_order(
                                conn,
                                position_id,
                                sell_client_order_id=sell_client_order_id,
                                sell_alpaca_order_id=None,
                                sell_submitted_at=None,
                                sell_status=initial_status,
                                sell_order_qty=2,
                                sell_order_limit_price=150,
                            )
                            if operation_name == "sell_submission_fence":

                                def invoke(
                                    position_id: int = position_id,
                                    sell_client_order_id: str = sell_client_order_id,
                                ) -> object:
                                    return storage_module.claim_alpaca_managed_sell_submission_fence(
                                        conn,
                                        position_id,
                                        sell_client_order_id=sell_client_order_id,
                                        claimed_at="2026-01-03T14:30:00Z",
                                        expected_remaining_qty=2,
                                        expected_target_sell_price=150,
                                        notes="claim sell fence",
                                    )

                            elif operation_name == "sell_submission_retry":

                                def invoke(
                                    position_id: int = position_id,
                                    sell_client_order_id: str = sell_client_order_id,
                                ) -> object:
                                    return storage_module.claim_alpaca_managed_sell_submission_retry_with_revision(
                                        conn,
                                        position_id,
                                        sell_client_order_id=sell_client_order_id,
                                        expected_target_sell_price=150,
                                        claimed_at="2026-01-03T14:30:00Z",
                                        reclaim_before="2026-01-03T14:29:00Z",
                                        notes="claim sell retry",
                                    )

                            else:
                                claimed_at = "2026-01-03T14:30:00Z"
                                state_revision = storage_module.claim_alpaca_managed_sell_submission_fence(
                                    conn,
                                    position_id,
                                    sell_client_order_id=sell_client_order_id,
                                    claimed_at=claimed_at,
                                    expected_remaining_qty=2,
                                    expected_target_sell_price=150,
                                    notes="seed sell fence",
                                )
                                self.assertIsNotNone(state_revision)

                                def invoke(
                                    position_id: int = position_id,
                                    state_revision: int = int(state_revision),
                                    sell_client_order_id: str = sell_client_order_id,
                                    claimed_at: str = claimed_at,
                                ) -> object:
                                    return storage_module.confirm_alpaca_managed_sell_submission(
                                        conn,
                                        position_id,
                                        sell_client_order_id=sell_client_order_id,
                                        claimed_at=claimed_at,
                                        expected_sell_status="submission_pending",
                                        expected_state_revision=state_revision,
                                        expected_remaining_qty=2,
                                        expected_target_sell_price=150,
                                    )

                    positions_before = conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall()
                    fills_before = conn.execute(
                        "SELECT * FROM alpaca_managed_sell_fills ORDER BY managed_position_id, alpaca_order_id"
                    ).fetchall()
                    aliases_before = conn.execute(
                        "SELECT * FROM alpaca_symbol_aliases ORDER BY alpaca_asset_id, symbol"
                    ).fetchall()
                    if caller_owns_transaction:
                        conn.execute("INSERT INTO transaction_probe VALUES (1)")

                    conn.fail_returning_fetch = True
                    with self.assertRaisesRegex(ValueError, "forced RETURNING row decode failure"):
                        invoke()

                    self.assertFalse(conn.fail_returning_fetch)
                    self.assertEqual(conn.in_transaction, caller_owns_transaction)
                    self.assertEqual(
                        conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall(),
                        positions_before,
                    )
                    self.assertEqual(
                        conn.execute(
                            "SELECT * FROM alpaca_managed_sell_fills ORDER BY managed_position_id, alpaca_order_id"
                        ).fetchall(),
                        fills_before,
                    )
                    self.assertEqual(
                        conn.execute("SELECT * FROM alpaca_symbol_aliases ORDER BY alpaca_asset_id, symbol").fetchall(),
                        aliases_before,
                    )
                    self.assertEqual(
                        conn.execute("SELECT value FROM transaction_probe").fetchall(),
                        [(1,)] if caller_owns_transaction else [],
                    )
                    if caller_owns_transaction:
                        conn.rollback()

    def test_buy_component_accounting_rejects_scaled_conflicts_and_overflow(
        self,
    ) -> None:
        timestamp = "2026-07-20T12:00:00Z"
        cases = (
            (
                "scaled",
                2.0,
                1_000_000_000_000_250.0,
                {
                    "buy-a": {
                        "broker_updated_at": timestamp,
                        "filled_qty": 1.0,
                        "filled_value": 1_000_000_000_000_000.0,
                    },
                    "buy-b": {
                        "broker_updated_at": timestamp,
                        "filled_qty": 1.0,
                        "filled_value": 1_000_000_000_000_000.0,
                    },
                },
                2_000_000_000_000_000.0,
            ),
            (
                "overflow",
                1e308,
                1e308,
                {
                    "buy-a": {
                        "broker_updated_at": timestamp,
                        "filled_qty": 5e307,
                        "filled_value": 1e308,
                    },
                    "buy-b": {
                        "broker_updated_at": timestamp,
                        "filled_qty": 5e307,
                        "filled_value": 1e308,
                    },
                },
                1e25,
            ),
        )
        for label, filled_qty, filled_avg_price, components, buy_limit in cases:
            with self.subTest(case=label):
                position_id = save_alpaca_managed_buy_order(
                    self.conn,
                    symbol=f"BAD-{label}",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"component-{label}",
                    buy_alpaca_order_id="buy-b",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    # Exercise arithmetic overflow through a legacy intent:
                    # a broker-submittable durable quantity cannot itself be
                    # as large as this synthetic accounting fixture.
                    buy_order_qty=None if label == "overflow" else filled_qty,
                    buy_order_limit_price=None if label == "overflow" else buy_limit,
                )

                with self.assertRaisesRegex(ValueError, "component accounting conflicts"):
                    mark_alpaca_managed_buy_filled(
                        self.conn,
                        position_id,
                        buy_status="filled",
                        filled_qty=filled_qty,
                        filled_avg_price=filled_avg_price,
                        filled_at="2026-01-02T14:31:00Z",
                        target_sell_price=150,
                        buy_fill_broker_updated_at=timestamp,
                        buy_fill_broker_oldest_updated_at=timestamp,
                        buy_fill_component_revisions=components,
                        buy_alpaca_order_id="buy-b",
                    )

                self.assertEqual(
                    self.conn.execute(
                        "SELECT filled_qty, filled_avg_price, buy_fill_component_revisions "
                        "FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    (None, None, None),
                )

    def test_unchanged_buy_component_revision_requires_exact_accounting(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="EXACT",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="component-exact",
            buy_alpaca_order_id="buy-b",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=2_000_000_000_000_000.0,
        )
        first_revision = "2026-07-20T12:00:00Z"
        second_revision = "2026-07-20T12:01:00Z"
        initial_components = {
            "buy-a": {
                "broker_updated_at": first_revision,
                "filled_qty": 1.0,
                "filled_value": 1_000_000_000_000_000.0,
            },
            "buy-b": {
                "broker_updated_at": first_revision,
                "filled_qty": 1.0,
                "filled_value": 1_000_000_000_000_000.0,
            },
        }
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=1_000_000_000_000_000.0,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=1_500_000_000_000_000.0,
                buy_fill_broker_updated_at=first_revision,
                buy_fill_broker_oldest_updated_at=first_revision,
                buy_fill_component_revisions=initial_components,
                buy_alpaca_order_id="buy-b",
            )
        )
        before = self.conn.execute(
            "SELECT state_revision, filled_avg_price, buy_fill_broker_updated_at, "
            "buy_fill_component_revisions FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        applied = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=1_000_000_000_000_250.0,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=1_500_000_000_000_000.0,
            buy_fill_broker_updated_at=second_revision,
            buy_fill_broker_oldest_updated_at=first_revision,
            buy_fill_component_revisions={
                "buy-a": {
                    "broker_updated_at": first_revision,
                    "filled_qty": 1.0,
                    "filled_value": 1_000_000_000_000_500.0,
                },
                "buy-b": {
                    "broker_updated_at": second_revision,
                    "filled_qty": 1.0,
                    "filled_value": 1_000_000_000_000_000.0,
                },
            },
            buy_alpaca_order_id="buy-b",
            expected_buy_status="filled",
            expected_buy_alpaca_order_id="buy-b",
            expected_filled_qty=2,
            expected_filled_avg_price=1_000_000_000_000_000.0,
            expected_target_sell_price=1_500_000_000_000_000.0,
            expected_sell_client_order_id=None,
        )
        after = self.conn.execute(
            "SELECT state_revision, filled_avg_price, buy_fill_broker_updated_at, "
            "buy_fill_component_revisions FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(applied)
        self.assertEqual(after, before)

    def test_buy_component_revision_ids_must_be_canonical_and_unique(self) -> None:
        timestamp = "2026-07-20T12:00:00Z"
        invalid_observations = (
            {1: timestamp},
            {1: timestamp, "1": timestamp},
            {" buy-a": timestamp},
            {"buy-a ": timestamp},
            {"buy\na": timestamp},
            {"buy/a": timestamp},
            {"A" * 129: timestamp},
            {" buy-a": timestamp, "buy-a ": timestamp},
        )
        for observed in invalid_observations:
            with (
                self.subTest(observed=observed),
                self.assertRaisesRegex(
                    ValueError,
                    "canonical broker order ID",
                ),
            ):
                alpaca_managed_buy_fill_observation_authorizes_mutation(
                    persisted_broker_updated_at=None,
                    persisted_component_revisions=None,
                    observed_broker_updated_at=timestamp,
                    observed_oldest_broker_updated_at=timestamp,
                    observed_component_revisions=observed,  # type: ignore[arg-type]
                )

        for persisted in (
            '{" buy-a":"2026-07-20T12:00:00Z"}',
            '{"buy-a":"2026-07-20T12:00:00Z","buy-a":"2026-07-20T12:01:00Z"}',
        ):
            with self.subTest(persisted=persisted), self.assertRaises(ValueError):
                alpaca_managed_buy_fill_observation_authorizes_mutation(
                    persisted_broker_updated_at=timestamp,
                    persisted_component_revisions=persisted,
                    observed_broker_updated_at="2026-07-20T12:02:00Z",
                    observed_oldest_broker_updated_at="2026-07-20T12:02:00Z",
                    observed_component_revisions={"buy-a": "2026-07-20T12:02:00Z"},
                )

        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="CANONICAL",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="component-canonical-id",
            buy_alpaca_order_id="buy-a",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=100,
        )
        with self.assertRaisesRegex(ValueError, "canonical broker order ID"):
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=1,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
                buy_fill_broker_updated_at=timestamp,
                buy_fill_broker_oldest_updated_at=timestamp,
                buy_fill_component_revisions={" buy-a": timestamp},
                buy_alpaca_order_id="buy-a",
            )
        self.assertEqual(
            self.conn.execute(
                "SELECT filled_qty, filled_avg_price, buy_fill_component_revisions "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            (None, None, None),
        )

    def test_persisted_buy_component_json_rejects_excessive_nesting_before_decoding(self) -> None:
        nesting_depth = 10_000
        encoded = "[" * nesting_depth + "0" + "]" * nesting_depth

        with (
            patch("leveraged_trader.storage.json.loads") as mock_loads,
            self.assertRaisesRegex(ValueError, "supported JSON nesting depth of 128"),
        ):
            storage_module._decode_alpaca_buy_component_revisions(encoded)

        mock_loads.assert_not_called()

    def test_persisted_buy_component_json_rejects_blob_values_before_decoding(self) -> None:
        deeply_nested_bytes = ("[" * 10_000 + "0" + "]" * 10_000).encode()
        for encoded in (b"{}", bytearray(b"{}"), deeply_nested_bytes):
            with (
                self.subTest(encoded_type=type(encoded).__name__, encoded_length=len(encoded)),
                patch("leveraged_trader.storage.json.loads") as mock_loads,
                self.assertRaisesRegex(ValueError, "invalid JSON"),
            ):
                storage_module._decode_alpaca_buy_component_revisions(encoded)  # type: ignore[arg-type]
            mock_loads.assert_not_called()

    def test_persisted_buy_component_json_nesting_guard_has_stable_boundary(self) -> None:
        supported = "[" * 128 + "0" + "]" * 128
        excessive = "[" * 129 + "0" + "]" * 129
        hidden_after_malformed_closers = "]" * 200 + excessive

        storage_module._reject_excessive_alpaca_buy_component_json_nesting(supported)
        for encoded in (excessive, hidden_after_malformed_closers):
            with (
                self.subTest(encoded_prefix=encoded[:10]),
                self.assertRaisesRegex(
                    ValueError,
                    "supported JSON nesting depth of 128",
                ),
            ):
                storage_module._reject_excessive_alpaca_buy_component_json_nesting(encoded)

    def test_persisted_buy_component_json_nesting_guard_ignores_escaped_string_content(self) -> None:
        string_value = ('[{"\\\\' * 1_000) + ('\\\\"}]' * 1_000)
        encoded = json.dumps({"text": string_value})

        storage_module._reject_excessive_alpaca_buy_component_json_nesting(encoded)

        self.assertEqual(json.loads(encoded), {"text": string_value})

    def test_managed_execute_failures_end_owned_transactions_but_preserve_callers(
        self,
    ) -> None:
        for operation_name in ("buy", "sell_generation"):
            with self.subTest(operation=operation_name), tempfile.TemporaryDirectory() as tmp_dir:
                db_path = f"{tmp_dir}/{operation_name}-execute-failure.sqlite"
                conn = sqlite3.connect(db_path, timeout=0.2, factory=ExecuteFailureConnection)
                try:
                    init_state_db(conn)
                    conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
                    conn.commit()
                    position_id: int | None = None
                    if operation_name == "sell_generation":
                        position_id = save_alpaca_managed_buy_order(
                            conn,
                            symbol="TQQQ",
                            signal_symbol="QQQ",
                            buy_rsi=30,
                            profit_target_multiple=1.5,
                            buy_signal_date="2026-01-02",
                            buy_client_order_id="rsi-buy-execute-failure-parent",
                            buy_alpaca_order_id="buy-execute-failure-parent",
                            buy_submitted_at="2026-01-02T14:30:00Z",
                            buy_status="accepted",
                            buy_order_qty=1,
                            buy_order_limit_price=100,
                        )

                    def invoke(
                        operation_name: str = operation_name,
                        conn: ExecuteFailureConnection = conn,
                        position_id: int | None = position_id,
                    ) -> object:
                        if operation_name == "buy":
                            return save_alpaca_managed_buy_order(
                                conn,
                                symbol="TQQQ",
                                signal_symbol="QQQ",
                                buy_rsi=30,
                                profit_target_multiple=1.5,
                                buy_signal_date="2026-01-02",
                                buy_client_order_id="rsi-buy-execute-failure",
                                buy_alpaca_order_id="buy-execute-failure",
                                buy_submitted_at="2026-01-02T14:30:00Z",
                                buy_status="accepted",
                                buy_order_qty=1,
                                buy_order_limit_price=100,
                            )
                        if position_id is None:
                            raise AssertionError("sell-generation fixture requires a managed position")
                        return record_alpaca_managed_sell_generation(
                            conn,
                            position_id,
                            "sell-execute-failure",
                        )

                    fail_fragment = (
                        "INSERT INTO alpaca_managed_positions"
                        if operation_name == "buy"
                        else "INSERT INTO alpaca_managed_sell_fills"
                    )
                    conn.fail_sql_fragment = fail_fragment
                    with self.assertRaisesRegex(sqlite3.OperationalError, "forced execute failure"):
                        invoke()

                    self.assertFalse(conn.in_transaction)
                    with sqlite3.connect(db_path, timeout=0.2) as observer:
                        observer.execute("INSERT INTO transaction_probe VALUES (1)")
                        if operation_name == "buy":
                            self.assertEqual(
                                observer.execute("SELECT COUNT(*) FROM alpaca_managed_positions").fetchone(),
                                (0,),
                            )
                        else:
                            self.assertEqual(
                                observer.execute("SELECT COUNT(*) FROM alpaca_managed_sell_fills").fetchone(),
                                (0,),
                            )

                    conn.execute("INSERT INTO transaction_probe VALUES (2)")
                    conn.fail_sql_fragment = fail_fragment
                    with self.assertRaisesRegex(sqlite3.OperationalError, "forced execute failure"):
                        invoke()

                    self.assertTrue(conn.in_transaction)
                    self.assertEqual(
                        conn.execute("SELECT value FROM transaction_probe ORDER BY value").fetchall(),
                        [(1,), (2,)],
                    )
                    conn.commit()
                    with sqlite3.connect(db_path) as observer:
                        self.assertEqual(
                            observer.execute("SELECT value FROM transaction_probe ORDER BY value").fetchall(),
                            [(1,), (2,)],
                        )
                finally:
                    conn.close()

    def test_missing_submission_adoption_collision_ends_owned_transaction_but_preserves_caller(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/buy-adoption-collision.sqlite"
            conn = sqlite3.connect(db_path, timeout=0.2)
            try:
                init_state_db(conn)
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
                conn.commit()
                closed_position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-adoption-collision-closed",
                    buy_alpaca_order_id="buy-adoption-collision-closed",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                )
                conn.execute(
                    """
                    UPDATE alpaca_managed_positions
                    SET buy_status = 'submission_not_found',
                        buy_alpaca_order_id = NULL,
                        closed_at = '2026-01-03T00:00:00Z'
                    WHERE id = ?
                    """,
                    (closed_position_id,),
                )
                conn.commit()
                save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-03",
                    buy_client_order_id="rsi-buy-adoption-collision-active",
                    buy_alpaca_order_id="buy-adoption-collision-active",
                    buy_submitted_at="2026-01-03T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=1,
                    buy_order_limit_price=101,
                )

                def adopt_closed_position() -> bool:
                    return adopt_alpaca_managed_buy_order_if_submission_not_found(
                        conn,
                        workflow=None,
                        symbol="TQQQ",
                        signal_symbol="QQQ",
                        buy_rsi=30,
                        profit_target_multiple=1.5,
                        buy_signal_date="2026-01-02",
                        buy_client_order_id="rsi-buy-adoption-collision-closed",
                        buy_alpaca_order_id="buy-adoption-collision-recovered",
                        buy_submitted_at="2026-01-02T14:30:00Z",
                        buy_status="accepted",
                        buy_order_qty=1,
                        buy_order_limit_price=100,
                    )

                with self.assertRaises(sqlite3.IntegrityError):
                    adopt_closed_position()

                self.assertFalse(conn.in_transaction)
                with sqlite3.connect(db_path, timeout=0.2) as observer:
                    observer.execute("INSERT INTO transaction_probe VALUES (1)")
                    observer.commit()
                    self.assertEqual(
                        observer.execute(
                            "SELECT buy_status, buy_alpaca_order_id, closed_at "
                            "FROM alpaca_managed_positions WHERE id = ?",
                            (closed_position_id,),
                        ).fetchone(),
                        ("submission_not_found", None, "2026-01-03T00:00:00Z"),
                    )

                conn.execute("INSERT INTO transaction_probe VALUES (2)")
                with self.assertRaises(sqlite3.IntegrityError):
                    adopt_closed_position()

                self.assertTrue(conn.in_transaction)
                self.assertEqual(
                    conn.execute("SELECT value FROM transaction_probe ORDER BY value").fetchall(),
                    [(1,), (2,)],
                )
                conn.commit()
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute("SELECT value FROM transaction_probe ORDER BY value").fetchall(),
                    [(1,), (2,)],
                )

    def test_missing_submission_adoption_execute_and_rollback_failure_closes_connection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/buy-adoption-execute-rollback-failure.sqlite"
            conn = sqlite3.connect(db_path, factory=ExecuteFailureConnection)
            try:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-adoption-rollback-failure",
                    buy_alpaca_order_id="buy-adoption-rollback-failure",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                )
                conn.execute(
                    """
                    UPDATE alpaca_managed_positions
                    SET buy_status = 'submission_not_found',
                        buy_alpaca_order_id = NULL,
                        closed_at = '2026-01-03T00:00:00Z'
                    WHERE id = ?
                    """,
                    (position_id,),
                )
                conn.commit()
                conn.fail_sql_fragment = "UPDATE alpaca_managed_positions SET buy_alpaca_order_id"
                conn.fail_rollback = True

                with self.assertRaisesRegex(
                    sqlite3.OperationalError,
                    "forced execute failure",
                ) as raised:
                    adopt_alpaca_managed_buy_order_if_submission_not_found(
                        conn,
                        workflow=None,
                        symbol="TQQQ",
                        signal_symbol="QQQ",
                        buy_rsi=30,
                        profit_target_multiple=1.5,
                        buy_signal_date="2026-01-02",
                        buy_client_order_id="rsi-buy-adoption-rollback-failure",
                        buy_alpaca_order_id="buy-adoption-recovered",
                        buy_submitted_at="2026-01-02T14:30:00Z",
                        buy_status="accepted",
                        buy_order_qty=1,
                        buy_order_limit_price=100,
                    )

                self.assertTrue(conn.rollback_failure_observed)
                self.assertTrue(
                    any("forced rollback after execute failure" in note for note in raised.exception.__notes__)
                )
                with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                    conn.commit()
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute(
                        "SELECT buy_status, buy_alpaca_order_id, closed_at FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    ("submission_not_found", None, "2026-01-03T00:00:00Z"),
                )

    def test_managed_multi_step_mutators_rollback_late_failures_and_preserve_callers(
        self,
    ) -> None:
        for operation_name in ("sell_generation", "asset_adoption", "symbol_migration"):
            for caller_owns_transaction in (False, True):
                with (
                    self.subTest(
                        operation=operation_name,
                        caller_owns_transaction=caller_owns_transaction,
                    ),
                    closing(
                        sqlite3.connect(
                            ":memory:",
                            factory=ManagedLateFailureConnection,
                        )
                    ) as conn,
                ):
                    init_state_db(conn)
                    conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
                    conn.commit()
                    position_id = save_alpaca_managed_buy_order(
                        conn,
                        symbol=f"MUT-{operation_name}",
                        alpaca_asset_id=None,
                        signal_symbol="QQQ",
                        buy_rsi=30,
                        profit_target_multiple=1.5,
                        buy_signal_date="2026-01-02",
                        buy_client_order_id=f"rsi-buy-mutator-{operation_name}",
                        buy_alpaca_order_id=f"buy-mutator-{operation_name}",
                        buy_submitted_at="2026-01-02T14:30:00Z",
                        buy_status="accepted",
                        buy_order_qty=1,
                        buy_order_limit_price=100,
                    )
                    expected_revision = int(
                        conn.execute(
                            "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                            (position_id,),
                        ).fetchone()[0]
                    )
                    parent_before = conn.execute(
                        "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone()
                    aliases_before = conn.execute(
                        "SELECT * FROM alpaca_symbol_aliases ORDER BY alpaca_asset_id, symbol"
                    ).fetchall()
                    if caller_owns_transaction:
                        conn.execute("INSERT INTO transaction_probe VALUES (1)")

                    if operation_name == "sell_generation":
                        conn.fail_execute_fragment = (
                            "UPDATE alpaca_managed_positions SET state_revision = state_revision + 1 WHERE id = ?"
                        )

                        def invoke(position_id: int = position_id) -> object:
                            return record_alpaca_managed_sell_generation(
                                conn,
                                position_id,
                                "sell-mutator-late-failure",
                            )

                    elif operation_name == "asset_adoption":
                        conn.fail_execute_fragment = "INSERT INTO alpaca_symbol_aliases"

                        def invoke(
                            position_id: int = position_id,
                            expected_revision: int = expected_revision,
                        ) -> object:
                            return adopt_alpaca_managed_position_asset_if_current(
                                conn,
                                position_id,
                                expected_state_revision=expected_revision,
                                alpaca_asset_id="asset-mutator-adoption",
                            )

                    else:
                        conn.fail_executemany_fragment = "INSERT INTO alpaca_symbol_aliases"

                        def invoke(position_id: int = position_id) -> object:
                            return storage_module.migrate_alpaca_managed_position_symbol(
                                conn,
                                position_id,
                                alpaca_asset_id="asset-mutator-migration",
                                current_symbol="MUT-MIGRATED",
                            )

                    with self.assertRaisesRegex(
                        KeyboardInterrupt,
                        "forced late managed-accounting",
                    ):
                        invoke()

                    self.assertEqual(conn.in_transaction, caller_owns_transaction)
                    self.assertEqual(
                        conn.execute(
                            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                            (position_id,),
                        ).fetchone(),
                        parent_before,
                    )
                    self.assertEqual(
                        conn.execute("SELECT * FROM alpaca_symbol_aliases ORDER BY alpaca_asset_id, symbol").fetchall(),
                        aliases_before,
                    )
                    self.assertEqual(
                        conn.execute(
                            "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                            (position_id,),
                        ).fetchone(),
                        (0,),
                    )
                    self.assertEqual(
                        conn.execute("SELECT value FROM transaction_probe").fetchall(),
                        [(1,)] if caller_owns_transaction else [],
                    )
                    if caller_owns_transaction:
                        conn.commit()

    def test_stale_sell_quarantine_collision_preserves_caller_work_before_fallback(
        self,
    ) -> None:
        closed_position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-stale-quarantine-closed",
            alpaca_asset_id="asset-stale-quarantine-closed",
        )
        retry_claimed_at = "2026-01-03T14:30:00Z"
        closed_at = "2026-01-03T15:00:00Z"
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET sell_client_order_id = 'sell-stale-quarantine-closed',
                sell_submission_retry_claimed_at = ?,
                closed_at = ?
            WHERE id = ?
            """,
            (retry_claimed_at, closed_at, closed_position_id),
        )
        self.conn.commit()
        self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-stale-quarantine-active",
            alpaca_asset_id="asset-stale-quarantine-active",
        )
        expected_revision = int(
            self.conn.execute(
                "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                (closed_position_id,),
            ).fetchone()[0]
        )
        self.conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
        self.conn.commit()
        self.conn.execute("INSERT INTO transaction_probe VALUES (1)")

        applied = storage_module.quarantine_alpaca_managed_sell_stale_submission(
            self.conn,
            closed_position_id,
            sell_client_order_id="sell-stale-quarantine-closed",
            expected_sell_submission_retry_claimed_at=retry_claimed_at,
            stale_alpaca_order_ids=[],
            notes="retained closed-position quarantine",
            observed_alpaca_asset_id="asset-stale-quarantine-closed",
            expected_state_revision=expected_revision,
        )

        self.assertFalse(applied)
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute("SELECT value FROM transaction_probe").fetchall(),
            [(1,)],
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT sell_status, closed_at, notes FROM alpaca_managed_positions WHERE id = ?",
                (closed_position_id,),
            ).fetchone(),
            (
                "quantity_mismatch",
                closed_at,
                "retained closed-position quarantine",
            ),
        )

    def test_managed_late_failure_and_rollback_failure_closes_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/managed-late-rollback-failure.sqlite"
            conn = sqlite3.connect(db_path, factory=ExecuteFailureConnection)
            try:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-late-rollback-failure",
                    buy_alpaca_order_id="buy-late-rollback-failure",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                )
                conn.fail_sql_fragment = (
                    "UPDATE alpaca_managed_positions SET state_revision = state_revision + 1 WHERE id = ?"
                )
                conn.fail_rollback = True

                with self.assertRaisesRegex(
                    sqlite3.OperationalError,
                    "forced execute failure",
                ) as raised:
                    record_alpaca_managed_sell_generation(
                        conn,
                        position_id,
                        "sell-late-rollback-failure",
                    )

                self.assertTrue(conn.rollback_failure_observed)
                self.assertTrue(
                    any("forced rollback after execute failure" in note for note in raised.exception.__notes__)
                )
                with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                    conn.commit()
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute(
                        "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                        (position_id,),
                    ).fetchone(),
                    (0,),
                )

    def test_asset_adoption_does_not_swallow_failed_savepoint_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/asset-adoption-savepoint-cleanup-failure.sqlite"
            conn = sqlite3.connect(
                db_path,
                factory=RollbackAndConnectionRollbackFailureConnection,
            )
            try:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    alpaca_asset_id=None,
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-adoption-cleanup-failure",
                    buy_alpaca_order_id="buy-adoption-cleanup-failure",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                )
                expected_revision = int(
                    conn.execute(
                        "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone()[0]
                )
                conn.execute(
                    """
                    CREATE TRIGGER reject_asset_alias
                    BEFORE INSERT ON alpaca_symbol_aliases
                    BEGIN
                        SELECT RAISE(ABORT, 'forced alias integrity failure');
                    END
                    """
                )
                conn.commit()
                conn.failed_savepoint = "adopt_alpaca_managed_position_asset_if_current"

                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "forced alias integrity failure",
                ) as raised:
                    adopt_alpaca_managed_position_asset_if_current(
                        conn,
                        position_id,
                        expected_state_revision=expected_revision,
                        alpaca_asset_id="asset-adoption-cleanup-failure",
                    )

                self.assertTrue(conn.rollback_to_failed)
                self.assertTrue(conn.connection_rollback_failed)
                self.assertTrue(
                    any("forced connection rollback failure" in note for note in raised.exception.__notes__)
                )
                with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                    conn.commit()
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute(
                        "SELECT alpaca_asset_id FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    (None,),
                )
                self.assertEqual(
                    observer.execute("SELECT COUNT(*) FROM alpaca_symbol_aliases").fetchone(),
                    (0,),
                )

    def test_managed_execute_and_rollback_failure_closes_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/execute-rollback-failure.sqlite"
            conn = sqlite3.connect(db_path, factory=ExecuteFailureConnection)
            try:
                init_state_db(conn)
                conn.fail_sql_fragment = "INSERT INTO alpaca_managed_positions"
                conn.fail_rollback = True

                with self.assertRaisesRegex(
                    sqlite3.OperationalError,
                    "forced execute failure",
                ) as raised:
                    save_alpaca_managed_buy_order(
                        conn,
                        symbol="TQQQ",
                        signal_symbol="QQQ",
                        buy_rsi=30,
                        profit_target_multiple=1.5,
                        buy_signal_date="2026-01-02",
                        buy_client_order_id="rsi-buy-execute-rollback-failure",
                        buy_alpaca_order_id="buy-execute-rollback-failure",
                        buy_submitted_at="2026-01-02T14:30:00Z",
                        buy_status="accepted",
                        buy_order_qty=1,
                        buy_order_limit_price=100,
                    )

                self.assertTrue(conn.rollback_failure_observed)
                self.assertTrue(
                    any("forced rollback after execute failure" in note for note in raised.exception.__notes__)
                )
                with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                    conn.commit()
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute("SELECT COUNT(*) FROM alpaca_managed_positions").fetchone(),
                    (0,),
                )

    def test_sell_fill_outermost_savepoint_commit_failure_is_not_durable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/sell-fill-outermost-commit-failure.sqlite"
            conn = sqlite3.connect(db_path, factory=CommitFailureConnection)
            try:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    alpaca_asset_id="asset-tqqq",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-sell-fill-commit-failure",
                    buy_alpaca_order_id="buy-sell-fill-commit-failure",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="filled",
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                )
                mark_alpaca_managed_buy_filled(
                    conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=2,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                before = conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()
                conn.fail_next_commit = True

                with self.assertRaisesRegex(sqlite3.OperationalError, "forced commit failure"):
                    storage_module.mark_alpaca_managed_sell_filled(
                        conn,
                        position_id,
                        sell_status="partially_filled",
                        sell_filled_qty=1,
                        sell_filled_avg_price=110,
                        sell_filled_at="2026-01-02T14:32:00Z",
                        sell_alpaca_order_id="sell-fill-commit-failure",
                    )

                self.assertFalse(conn.in_transaction)
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute(
                        "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    before,
                )
                self.assertEqual(
                    observer.execute(
                        "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                        (position_id,),
                    ).fetchone(),
                    (0,),
                )

    def test_closed_correction_outermost_savepoint_commit_failure_is_not_durable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = f"{tmp_dir}/correction-outermost-commit-failure.sqlite"
            conn = sqlite3.connect(db_path, factory=CommitFailureConnection)
            try:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    alpaca_asset_id="asset-tqqq",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-correction-commit-failure",
                    buy_alpaca_order_id="buy-correction-commit-failure",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="filled",
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                )
                mark_alpaca_managed_buy_filled(
                    conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=2,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                closed_at = "2026-01-02T14:35:00Z"
                conn.execute(
                    "UPDATE alpaca_managed_positions SET closed_at = ? WHERE id = ?",
                    (closed_at, position_id),
                )
                conn.commit()
                before = conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()
                expected_revision = int(
                    conn.execute(
                        "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone()[0]
                )
                conn.fail_next_commit = True

                with self.assertRaisesRegex(sqlite3.OperationalError, "forced commit failure"):
                    apply_alpaca_closed_position_broker_correction(
                        conn,
                        position_id,
                        expected_closed_at=closed_at,
                        expected_state_revision=expected_revision,
                        alpaca_asset_id="asset-tqqq",
                        buy_order_qty=2,
                        buy_order_limit_price=105,
                        buy_status="filled",
                        filled_qty=2,
                        filled_avg_price=100,
                        filled_at="2026-01-02T14:31:00Z",
                        target_sell_price=150,
                        sell_status="partially_filled",
                        sell_fills=[("sell-correction-commit-failure", 1, 110)],
                        sell_filled_at="2026-01-02T14:33:00Z",
                        notes="correction must roll back",
                    )

                self.assertFalse(conn.in_transaction)
            finally:
                conn.close()

            with sqlite3.connect(db_path) as observer:
                self.assertEqual(
                    observer.execute(
                        "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    before,
                )
                self.assertEqual(
                    observer.execute(
                        "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                        (position_id,),
                    ).fetchone(),
                    (0,),
                )

    def test_managed_commit_failure_rolls_back_close_before_later_commit(self) -> None:
        with closing(sqlite3.connect(":memory:", factory=CommitFailureConnection)) as conn:
            init_state_db(conn)
            position_id = save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-commit-failure",
                buy_alpaca_order_id="buy-commit-failure",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
                buy_order_qty=1,
                buy_order_limit_price=100,
            )
            conn.fail_next_commit = True

            with self.assertRaisesRegex(sqlite3.OperationalError, "forced commit failure"):
                storage_module.close_alpaca_managed_position(
                    conn,
                    position_id,
                    closed_at="2026-01-02T14:31:00Z",
                    notes="this close must roll back",
                )

            self.assertTrue(conn.commit_failure_observed)
            self.assertFalse(conn.in_transaction)
            self.assertIsNone(
                conn.execute(
                    "SELECT closed_at FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()[0]
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET notes = 'unrelated later work' WHERE id = ?",
                (position_id,),
            )
            conn.commit()
            self.assertEqual(
                conn.execute(
                    "SELECT closed_at, notes FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone(),
                (None, "unrelated later work"),
            )

    def test_managed_composite_commit_failure_rolls_back_parent_and_child(self) -> None:
        with closing(sqlite3.connect(":memory:", factory=CommitFailureConnection)) as conn:
            init_state_db(conn)
            position_id = save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-composite-commit-failure",
                buy_alpaca_order_id="buy-composite-commit-failure",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="filled",
                buy_order_qty=1,
                buy_order_limit_price=100,
            )
            mark_alpaca_managed_buy_filled(
                conn,
                position_id,
                buy_status="filled",
                filled_qty=1,
                filled_avg_price=99,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            before = conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone()
            conn.fail_next_commit = True

            with self.assertRaisesRegex(sqlite3.OperationalError, "forced commit failure"):
                record_alpaca_managed_sell_order(
                    conn,
                    position_id,
                    sell_client_order_id="rsi-exit-TQQQ-composite-commit-failure",
                    sell_alpaca_order_id="sell-composite-commit-failure",
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="accepted",
                    sell_order_qty=1,
                    sell_order_limit_price=150,
                )

            self.assertFalse(conn.in_transaction)
            self.assertEqual(
                conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone(),
                before,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                    (position_id,),
                ).fetchone(),
                (0,),
            )

    def test_managed_parent_ledger_composites_roll_back_inner_write_failures(
        self,
    ) -> None:
        def seed_position(
            conn: sqlite3.Connection,
            suffix: str,
        ) -> int:
            position_id = save_alpaca_managed_buy_order(
                conn,
                symbol=f"CMP-{suffix}",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id=f"rsi-buy-composite-{suffix}",
                buy_alpaca_order_id=f"buy-composite-{suffix}",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="filled",
                buy_order_qty=2,
                buy_order_limit_price=105,
            )
            mark_alpaca_managed_buy_filled(
                conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
            return position_id

        def prepare_record(
            conn: sqlite3.Connection,
            position_id: int,
            suffix: str,
        ) -> Callable[[], object]:
            return lambda: record_alpaca_managed_sell_order(
                conn,
                position_id,
                sell_client_order_id=f"sell-client-{suffix}",
                sell_alpaca_order_id=f"sell-order-{suffix}",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_order_qty=2,
                sell_order_limit_price=150,
            )

        def prepare_update(
            conn: sqlite3.Connection,
            position_id: int,
            suffix: str,
        ) -> Callable[[], object]:
            sell_client_order_id = f"sell-client-{suffix}"
            record_alpaca_managed_sell_order(
                conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                sell_alpaca_order_id=None,
                sell_submitted_at=None,
                sell_status="submission_pending",
                sell_order_qty=2,
                sell_order_limit_price=150,
            )
            return lambda: update_alpaca_managed_sell_status_if_current(
                conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                sell_status="accepted",
                sell_alpaca_order_id=f"sell-order-{suffix}",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_order_qty=2,
                sell_order_limit_price=150,
            )

        def prepare_generic_status(
            conn: sqlite3.Connection,
            position_id: int,
            suffix: str,
        ) -> Callable[[], object]:
            return lambda: storage_module.update_alpaca_managed_sell_status(
                conn,
                position_id,
                sell_status="accepted",
                sell_alpaca_order_id=f"sell-order-{suffix}",
                sell_submitted_at="2026-01-02T14:32:00Z",
            )

        def prepare_replacement(
            conn: sqlite3.Connection,
            position_id: int,
            suffix: str,
        ) -> Callable[[], object]:
            sell_client_order_id = f"sell-client-{suffix}"
            sell_alpaca_order_id = f"sell-order-{suffix}"
            record_alpaca_managed_sell_order(
                conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                sell_alpaca_order_id=sell_alpaca_order_id,
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_order_qty=2,
                sell_order_limit_price=150,
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET sell_status = 'canceled' WHERE id = ?",
                (position_id,),
            )
            conn.commit()
            return lambda: claim_alpaca_managed_sell_replacement(
                conn,
                position_id,
                prior_sell_client_order_id=sell_client_order_id,
                prior_sell_alpaca_order_id=sell_alpaca_order_id,
                prior_renewal_count=0,
                replacement_sell_client_order_id=f"replacement-{suffix}",
                requested_remaining_qty=2,
                expected_target_sell_price=150,
                notes="replacement claim",
            )

        def prepare_attach(
            conn: sqlite3.Connection,
            position_id: int,
            suffix: str,
        ) -> Callable[[], object]:
            sell_client_order_id = f"sell-client-{suffix}"
            record_alpaca_managed_sell_order(
                conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                sell_alpaca_order_id=None,
                sell_submitted_at=None,
                sell_status="submission_pending",
                sell_order_qty=2,
                sell_order_limit_price=150,
            )
            return lambda: attach_alpaca_managed_sell_order_if_current(
                conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                expected_sell_alpaca_order_id=None,
                expected_renewal_count=0,
                sell_renewal_count=0,
                sell_client_order_id=sell_client_order_id,
                sell_alpaca_order_id=f"sell-order-{suffix}",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_expires_at=None,
                sell_order_qty=2,
                sell_order_limit_price=150,
            )

        preparations = (
            ("record", prepare_record),
            ("update", prepare_update),
            ("generic_status", prepare_generic_status),
            ("replacement", prepare_replacement),
            ("attach", prepare_attach),
        )
        for operation_name, prepare in preparations:
            with (
                self.subTest(operation=operation_name),
                closing(
                    sqlite3.connect(
                        ":memory:",
                        factory=SellGenerationFailureConnection,
                    )
                ) as conn,
            ):
                init_state_db(conn)
                position_id = seed_position(conn, operation_name)
                operation = prepare(conn, position_id, operation_name)
                parent_before = conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()
                ledger_before = conn.execute(
                    "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ? ORDER BY alpaca_order_id",
                    (position_id,),
                ).fetchall()
                conn.sell_generation_failure = sqlite3.OperationalError("forced sell-generation write failure")

                with self.assertRaisesRegex(
                    sqlite3.OperationalError,
                    "forced sell-generation write failure",
                ):
                    operation()

                self.assertFalse(conn.in_transaction)
                self.assertEqual(
                    conn.execute(
                        "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    parent_before,
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT * FROM alpaca_managed_sell_fills "
                        "WHERE managed_position_id = ? ORDER BY alpaca_order_id",
                        (position_id,),
                    ).fetchall(),
                    ledger_before,
                )

    def test_managed_composite_base_exception_preserves_outer_transaction(
        self,
    ) -> None:
        with closing(sqlite3.connect(":memory:", factory=SellGenerationFailureConnection)) as conn:
            init_state_db(conn)
            position_id = save_alpaca_managed_buy_order(
                conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-composite-base-exception",
                buy_alpaca_order_id="buy-composite-base-exception",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
                buy_order_qty=1,
                buy_order_limit_price=100,
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET notes = 'outer work' WHERE id = ?",
                (position_id,),
            )
            expected_outer_state = conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone()
            conn.sell_generation_failure = KeyboardInterrupt("forced sell-generation interruption")

            with self.assertRaisesRegex(
                KeyboardInterrupt,
                "forced sell-generation interruption",
            ):
                record_alpaca_managed_sell_order(
                    conn,
                    position_id,
                    sell_client_order_id="sell-client-base-exception",
                    sell_alpaca_order_id="sell-order-base-exception",
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="accepted",
                    sell_order_qty=1,
                    sell_order_limit_price=120,
                )

            self.assertTrue(conn.in_transaction)
            self.assertEqual(
                conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone(),
                expected_outer_state,
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM alpaca_managed_sell_fills").fetchone(),
                (0,),
            )
            conn.commit()
            self.assertEqual(
                conn.execute(
                    "SELECT notes, sell_client_order_id FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone(),
                ("outer work", None),
            )

    def test_managed_composite_rejects_invalid_child_economics_before_parent_write(
        self,
    ) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-invalid-composite-economics",
        )
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "quantity must be positive"):
            record_alpaca_managed_sell_order(
                self.conn,
                position_id,
                sell_client_order_id="sell-invalid-composite-economics",
                sell_alpaca_order_id="sell-invalid-composite-economics",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_order_qty=0,
                sell_order_limit_price=150,
            )

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchone(),
            (0,),
        )

    def test_managed_buy_intent_inputs_are_complete_positive_and_numeric(self) -> None:
        invalid_intents = (
            (1, None),
            (None, 100),
            (0, 100),
            (-1, 100),
            (1.5, 100),
            (2**53 + 1, 100),
            (1e23, 100),
            (1, 0),
            (1, math.inf),
            (1, 100.001),
            (1, 0.99995),
            (1, 0.00009),
            ("1", 100),
            (True, 100),
        )

        for index, (buy_order_qty, buy_order_limit_price) in enumerate(invalid_intents):
            with (
                self.subTest(intent=(buy_order_qty, buy_order_limit_price)),
                self.assertRaises(ValueError),
            ):
                save_alpaca_managed_buy_order(
                    self.conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"rsi-buy-invalid-intent-{index}",
                    buy_alpaca_order_id=f"buy-invalid-intent-{index}",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=buy_order_qty,  # type: ignore[arg-type]
                    buy_order_limit_price=buy_order_limit_price,
                )

        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM alpaca_managed_positions").fetchone(),
            (0,),
        )

        with self.assertRaisesRegex(ValueError, "complete immutable order intent"):
            storage_module.claim_alpaca_managed_buy_intent(
                self.conn,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-missing-claim-intent",
                buy_order_qty=None,  # type: ignore[arg-type]
                buy_order_limit_price=None,  # type: ignore[arg-type]
            )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM alpaca_managed_positions").fetchone(),
            (0,),
        )

        for index, (buy_order_qty, buy_order_limit_price) in enumerate(
            ((1.0, 100), (1, 0.9999), (1, 0.0001), (2**53, 100))
        ):
            with self.subTest(valid_intent=(buy_order_qty, buy_order_limit_price)):
                position_id = save_alpaca_managed_buy_order(
                    self.conn,
                    symbol=f"VALID{index}",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"rsi-buy-valid-intent-{index}",
                    buy_alpaca_order_id=f"buy-valid-intent-{index}",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=buy_order_qty,
                    buy_order_limit_price=buy_order_limit_price,
                )
                self.assertEqual(
                    self.conn.execute(
                        "SELECT buy_order_qty, buy_order_limit_price FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    (float(buy_order_qty), float(buy_order_limit_price)),
                )

    def test_final_buy_fence_requires_the_exact_durable_intent(self) -> None:
        claim = storage_module.claim_alpaca_managed_buy_intent(
            self.conn,
            symbol="TQQQ",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-corrupt-final-fence",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET buy_order_limit_price = 105.001 WHERE id = ?",
            (claim.position_id,),
        )
        self.conn.commit()
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (claim.position_id,),
        ).fetchone()

        confirmed_revision = storage_module.confirm_alpaca_managed_buy_submission(
            self.conn,
            claim.position_id,
            expected_buy_submission_attempt_count=claim.attempt_count,
            expected_state_revision=claim.state_revision,
            expected_buy_order_qty=2,
            expected_buy_order_limit_price=105,
            claimed_at="2026-01-02T14:30:01Z",
        )

        self.assertIsNone(confirmed_revision)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (claim.position_id,),
            ).fetchone(),
            before,
        )
        with self.assertRaisesRegex(ValueError, "invalid durable broker intent economics"):
            load_alpaca_managed_positions(self.conn, active_only=True)

    def test_loaded_managed_positions_reject_non_wire_canonical_quantities(self) -> None:
        position_id = self.save_position(
            symbol="WIRE",
            client_order_id="buy-corrupt-wire-quantity",
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET sell_order_qty = 1e23, sell_order_limit_price = 150 WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()

        with self.assertRaisesRegex(ValueError, "invalid durable broker intent economics"):
            load_alpaca_managed_positions(self.conn, active_only=True)

    def test_buy_observation_requires_the_confirmed_revision_and_exact_intent(self) -> None:
        claim = storage_module.claim_alpaca_managed_buy_intent(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-post-fence-corruption",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        confirmed_revision = storage_module.confirm_alpaca_managed_buy_submission(
            self.conn,
            claim.position_id,
            expected_buy_submission_attempt_count=claim.attempt_count,
            expected_state_revision=claim.state_revision,
            expected_buy_order_qty=2,
            expected_buy_order_limit_price=105,
            claimed_at="2026-01-02T14:30:01Z",
        )
        assert confirmed_revision is not None
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_order_qty = 2.000000001, state_revision = state_revision + 1 WHERE id = ?",
            (claim.position_id,),
        )
        self.conn.commit()
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (claim.position_id,),
        ).fetchone()

        with self.assertRaisesRegex(RuntimeError, "lost its active-state persistence race"):
            save_alpaca_managed_buy_order(
                self.conn,
                symbol="TQQQ",
                alpaca_asset_id="asset-tqqq",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-post-fence-corruption",
                buy_alpaca_order_id="buy-post-fence-corruption",
                buy_submitted_at="2026-01-02T14:30:02Z",
                buy_status="accepted",
                buy_order_qty=2,
                buy_order_limit_price=105,
                require_applied=True,
                require_existing=True,
                expected_buy_submission_attempt_count=claim.attempt_count,
                expected_state_revision=confirmed_revision,
            )

        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (claim.position_id,),
            ).fetchone(),
            before,
        )

    def test_sell_intent_claim_rolls_back_a_post_init_corrupt_target(self) -> None:
        position_id = self.save_position(
            symbol="CORRUPTTARGET",
            client_order_id="buy-corrupt-target",
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET target_sell_price = 150.000000001 WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "Alpaca price tick"):
            storage_module.claim_alpaca_managed_initial_sell_intent(
                self.conn,
                position_id,
                sell_order_namespace="corrupt-target",
                sell_client_order_id="sell-corrupt-target",
                expected_remaining_qty=2,
                expected_target_sell_price=150,
                notes="must roll back invalid copied economics",
            )

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )

    def test_managed_sell_intent_limit_prices_must_be_canonical_before_writes(
        self,
    ) -> None:
        position_id = self.save_position(
            symbol="SELLTICK",
            client_order_id="buy-off-tick-sell-intent",
        )
        parent_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        operations = (
            lambda: record_alpaca_managed_sell_generation(
                self.conn,
                position_id,
                "sell-off-tick-generation",
                submitted_qty=2,
                submitted_limit_price=150.001,
            ),
            lambda: record_alpaca_managed_sell_order(
                self.conn,
                position_id,
                sell_client_order_id="sell-off-tick-parent",
                sell_alpaca_order_id="sell-off-tick-parent",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_order_qty=2,
                sell_order_limit_price=150.001,
            ),
        )

        for operation in operations:
            with (
                self.subTest(operation=operation),
                self.assertRaisesRegex(
                    ValueError,
                    "Alpaca price tick",
                ),
            ):
                operation()

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            parent_before,
        )

    def test_managed_sell_intent_quantities_must_be_wire_canonical_before_writes(
        self,
    ) -> None:
        position_id = self.save_position(
            symbol="SELLWIRE",
            client_order_id="buy-invalid-sell-wire-quantity",
        )
        parent_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        operations = (
            lambda: record_alpaca_managed_sell_generation(
                self.conn,
                position_id,
                "sell-invalid-wire-generation",
                submitted_qty=1e23,
                submitted_limit_price=150,
            ),
            lambda: record_alpaca_managed_sell_order(
                self.conn,
                position_id,
                sell_client_order_id="sell-invalid-wire-parent",
                sell_alpaca_order_id="sell-invalid-wire-parent",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_order_qty=1e23,
                sell_order_limit_price=150,
            ),
        )

        for operation in operations:
            with (
                self.subTest(operation=operation),
                self.assertRaisesRegex(
                    ValueError,
                    "quantity precision",
                ),
            ):
                operation()

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            parent_before,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchone(),
            (0,),
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchone(),
            (0,),
        )

    def test_managed_strategy_economics_are_validated_on_save_claim_and_retry(self) -> None:
        invalid_economics = (
            (math.nan, 1.5),
            (math.inf, 1.5),
            (-0.01, 1.5),
            (100.01, 1.5),
            (30, 1.0),
            (30, math.inf),
            (30, 100.01),
            (True, 1.5),
            ("30", 1.5),
        )
        retry_position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="RETRY",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-invalid-economics-retry",
            buy_alpaca_order_id=None,
            buy_submitted_at=None,
            buy_status="submission_not_found",
            buy_order_qty=1,
            buy_order_limit_price=100,
        )
        self.close_position(retry_position_id)
        retry_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (retry_position_id,),
        ).fetchone()

        for index, (buy_rsi, profit_target_multiple) in enumerate(invalid_economics):
            with (
                self.subTest(path="save", economics=(buy_rsi, profit_target_multiple)),
                self.assertRaises(ValueError),
            ):
                save_alpaca_managed_buy_order(
                    self.conn,
                    symbol=f"SAVE{index}",
                    signal_symbol="QQQ",
                    buy_rsi=buy_rsi,  # type: ignore[arg-type]
                    profit_target_multiple=profit_target_multiple,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"buy-invalid-economics-save-{index}",
                    buy_alpaca_order_id=None,
                    buy_submitted_at=None,
                    buy_status="accepted",
                )
            with (
                self.subTest(path="claim", economics=(buy_rsi, profit_target_multiple)),
                self.assertRaises(ValueError),
            ):
                storage_module.claim_alpaca_managed_buy_intent(
                    self.conn,
                    symbol=f"CLAIM{index}",
                    signal_symbol="QQQ",
                    buy_rsi=buy_rsi,  # type: ignore[arg-type]
                    profit_target_multiple=profit_target_multiple,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"buy-invalid-economics-claim-{index}",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                )
            with (
                self.subTest(path="retry", economics=(buy_rsi, profit_target_multiple)),
                self.assertRaises(ValueError),
            ):
                storage_module.claim_alpaca_managed_buy_intent(
                    self.conn,
                    symbol="RETRY",
                    signal_symbol="QQQ",
                    buy_rsi=buy_rsi,  # type: ignore[arg-type]
                    profit_target_multiple=profit_target_multiple,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="buy-invalid-economics-retry",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                    allow_retry_after_not_found=True,
                )

        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (retry_position_id,),
            ).fetchone(),
            retry_before,
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM alpaca_managed_positions").fetchone(),
            (1,),
        )

    def test_managed_target_sell_price_must_be_a_canonical_alpaca_tick(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TICK",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-off-tick-target",
            buy_alpaca_order_id="buy-off-tick-target",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=100,
        )
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        for off_tick_target in (150.0049, 0.99995, 0.00009):
            with (
                self.subTest(target=off_tick_target),
                self.assertRaisesRegex(ValueError, "Alpaca price tick"),
            ):
                mark_alpaca_managed_buy_filled(
                    self.conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=1,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=off_tick_target,
                )

        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )

        correction_id = self.save_position(
            symbol="TICKCORR",
            client_order_id="buy-off-tick-correction-target",
        )
        self.close_position(correction_id)
        correction_snapshot = load_alpaca_managed_positions(self.conn).query("id == @correction_id").iloc[0]
        correction_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (correction_id,),
        ).fetchone()
        with self.assertRaisesRegex(ValueError, "Alpaca price tick"):
            apply_alpaca_closed_position_broker_correction(
                self.conn,
                correction_id,
                expected_closed_at=str(correction_snapshot["closed_at"]),
                expected_state_revision=int(correction_snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150.0049,
                sell_status="filled",
                sell_fills=[],
                sell_filled_at=None,
                notes="off-tick correction must fail",
            )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (correction_id,),
            ).fetchone(),
            correction_before,
        )

    def test_init_state_db_rejects_invalid_managed_economics_atomically(self) -> None:
        cases = (
            ("buy_rsi", -0.01, "strategy economics"),
            ("buy_rsi", math.inf, "strategy economics"),
            ("profit_target_multiple", 1.0, "strategy economics"),
            ("profit_target_multiple", 100.01, "strategy economics"),
            ("target_sell_price", 150.0049, "target sell price"),
            ("buy_order_qty", 1.5, "buy intent"),
            ("buy_order_qty", 1e23, "buy intent"),
            ("buy_order_limit_price", 100.001, "buy intent"),
            ("sell_order_qty", 1e23, "sell intent"),
            ("sell_order_limit_price", 150.001, "sell intent"),
        )
        for column, invalid_value, message in cases:
            with self.subTest(column=column, value=invalid_value), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="buy-invalid-imported-economics",
                    buy_alpaca_order_id="buy-invalid-imported-economics",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                )
                mark_alpaca_managed_buy_filled(
                    conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=1,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN closed_correction_audited_at")
                conn.execute(
                    f"UPDATE alpaca_managed_positions SET {column} = ? WHERE id = ?",
                    (invalid_value, position_id),
                )
                conn.commit()
                row_before = conn.execute(
                    f"SELECT {column}, state_revision FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()

                with self.assertRaisesRegex(ValueError, message):
                    init_state_db(conn)

                self.assertFalse(conn.in_transaction)
                self.assertNotIn(
                    "closed_correction_audited_at",
                    {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)")},
                )
                self.assertEqual(
                    conn.execute(
                        f"SELECT {column}, state_revision FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    row_before,
                )

    def test_init_state_db_rejects_invalid_sell_ledger_economics_atomically(
        self,
    ) -> None:
        cases = (
            ("negative filled quantity", "filled_qty = -1", (), "sell-fill economics"),
            ("negative filled value", "filled_value = -1", (), "sell-fill economics"),
            ("inconsistent zero pair", "filled_qty = 1", (), "sell-fill economics"),
            ("nonfinite filled value", "filled_value = ?", (math.inf,), "sell-fill economics"),
            (
                "overflowing unit price",
                "filled_qty = ?, filled_value = ?",
                (5e-324, 1e308),
                "sell-fill economics",
            ),
            (
                "underflowing unit price",
                "filled_qty = ?, filled_value = ?",
                (1e308, 5e-324),
                "sell-fill economics",
            ),
            (
                "nonnumeric filled quantity",
                "filled_qty = ?",
                (sqlite3.Binary(b"1"),),
                "sell-fill economics",
            ),
            ("zero submitted quantity", "submitted_qty = 0", (), "sell-generation economics"),
            ("negative submitted quantity", "submitted_qty = -1", (), "sell-generation economics"),
            ("nonfinite submitted quantity", "submitted_qty = ?", (math.inf,), "sell-generation economics"),
            ("non-wire submitted quantity", "submitted_qty = 1e23", (), "sell-generation economics"),
            (
                "nonnumeric submitted quantity",
                "submitted_qty = ?",
                (sqlite3.Binary(b"1"),),
                "sell-generation economics",
            ),
            ("negative submitted price", "submitted_limit_price = -1", (), "sell-generation economics"),
            (
                "nonfinite submitted price",
                "submitted_limit_price = ?",
                (math.inf,),
                "sell-generation economics",
            ),
            (
                "off-tick submitted price",
                "submitted_limit_price = 150.001",
                (),
                "sell-generation economics",
            ),
            (
                "off-tick subdollar submitted price",
                "submitted_limit_price = 0.12345",
                (),
                "sell-generation economics",
            ),
            (
                "oversized submitted price",
                "submitted_limit_price = 1e26",
                (),
                "sell-generation economics",
            ),
            (
                "nonnumeric submitted price",
                "submitted_limit_price = ?",
                (sqlite3.Binary(b"150"),),
                "sell-generation economics",
            ),
        )
        for label, assignment, parameters, message in cases:
            with self.subTest(case=label), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="buy-invalid-imported-sell-ledger",
                    buy_alpaca_order_id="buy-invalid-imported-sell-ledger",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=2,
                    buy_order_limit_price=100,
                )
                record_alpaca_managed_sell_generation(
                    conn,
                    position_id,
                    "sell-invalid-imported-ledger",
                    submitted_qty=2,
                    submitted_limit_price=150,
                )
                conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN closed_correction_audited_at")
                conn.execute(
                    f"UPDATE alpaca_managed_sell_fills SET {assignment} WHERE managed_position_id = ?",
                    (*parameters, position_id),
                )
                conn.commit()
                row_before = conn.execute(
                    "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                    (position_id,),
                ).fetchone()

                with self.assertRaisesRegex(ValueError, message):
                    init_state_db(conn)

                self.assertFalse(conn.in_transaction)
                self.assertNotIn(
                    "closed_correction_audited_at",
                    {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)")},
                )
                self.assertEqual(
                    conn.execute(
                        "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                        (position_id,),
                    ).fetchone(),
                    row_before,
                )

        with closing(sqlite3.connect(":memory:")) as conn:
            init_state_db(conn)
            position_id = save_alpaca_managed_buy_order(
                conn,
                symbol="OVERFLOW",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="buy-overflowing-imported-sell-ledger",
                buy_alpaca_order_id="buy-overflowing-imported-sell-ledger",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
                buy_order_qty=2,
                buy_order_limit_price=100,
            )
            conn.executemany(
                """
                INSERT INTO alpaca_managed_sell_fills
                (managed_position_id, alpaca_order_id, filled_qty, filled_value)
                VALUES (?, ?, 1, ?)
                """,
                (
                    (position_id, "sell-overflow-1", 1e308),
                    (position_id, "sell-overflow-2", 1e308),
                ),
            )
            conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN closed_correction_audited_at")
            conn.commit()
            rows_before = conn.execute("SELECT * FROM alpaca_managed_sell_fills ORDER BY alpaca_order_id").fetchall()

            with self.assertRaisesRegex(ValueError, "overflowing cumulative sell economics"):
                init_state_db(conn)

            self.assertNotIn(
                "closed_correction_audited_at",
                {str(row[1]) for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)")},
            )
            self.assertEqual(
                conn.execute("SELECT * FROM alpaca_managed_sell_fills ORDER BY alpaca_order_id").fetchall(),
                rows_before,
            )

    def test_sell_ledger_readers_reject_post_init_corrupt_economics(self) -> None:
        position_id = self.save_position(
            symbol="CORRUPTLEDGER",
            client_order_id="buy-corrupt-ledger-reader",
        )
        storage_module.record_alpaca_managed_sell_generation(
            self.conn,
            position_id,
            "sell-corrupt-ledger-reader",
            submitted_qty=2,
            submitted_limit_price=150,
        )

        self.conn.execute(
            "UPDATE alpaca_managed_sell_fills SET filled_qty = -1, filled_value = -150 WHERE managed_position_id = ?",
            (position_id,),
        )
        self.conn.commit()
        corrupt_fill = self.conn.execute(
            "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
            (position_id,),
        ).fetchone()
        with self.assertRaisesRegex(ValueError, "sell-fill economics"):
            storage_module.alpaca_managed_sell_fill_observations(
                self.conn,
                position_id,
            )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchone(),
            corrupt_fill,
        )

        self.conn.execute(
            "UPDATE alpaca_managed_sell_fills "
            "SET filled_qty = 0, filled_value = 0, submitted_limit_price = 150.001 "
            "WHERE managed_position_id = ?",
            (position_id,),
        )
        self.conn.commit()
        corrupt_generation = self.conn.execute(
            "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
            (position_id,),
        ).fetchone()
        with self.assertRaisesRegex(ValueError, "Alpaca price tick"):
            storage_module.alpaca_managed_sell_generation_intents(
                self.conn,
                position_id,
            )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchone(),
            corrupt_generation,
        )

    def test_managed_buy_fill_quarantines_impossible_intent_economics(self) -> None:
        impossible_fills = (
            (2, 99, "quantity exceeds"),
            (0.5, 99, "filled status reports less"),
            (1, 100.006, "average price exceeds"),
        )
        for index, (filled_qty, filled_avg_price, issue) in enumerate(impossible_fills):
            position_id = save_alpaca_managed_buy_order(
                self.conn,
                symbol=f"BAD{index}",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id=f"rsi-buy-causal-fill-{index}",
                buy_alpaca_order_id=f"buy-causal-fill-{index}",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
                buy_order_qty=1,
                buy_order_limit_price=100,
            )
            with self.subTest(fill=(filled_qty, filled_avg_price)):
                self.assertTrue(
                    mark_alpaca_managed_buy_filled(
                        self.conn,
                        position_id,
                        buy_status="filled",
                        filled_qty=filled_qty,
                        filled_avg_price=filled_avg_price,
                        filled_at="2026-01-02T14:31:00Z",
                        target_sell_price=150,
                    )
                )
            recorded = self.conn.execute(
                "SELECT filled_qty, filled_avg_price, notes, buy_causality_quarantine "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone()
            self.assertEqual(recorded[:2], (filled_qty, filled_avg_price))
            self.assertIn(issue, recorded[2])
            self.assertIn("realized-P/L publication is quarantined", recorded[2])
            self.assertIn(issue, recorded[3])
            self.assertIn("realized-P/L publication is quarantined", recorded[3])
            init_state_db(self.conn)
            reloaded = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
            self.assertEqual(reloaded["filled_qty"], filled_qty)

        valid_position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="GOOD",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-causal-fill-valid",
            buy_alpaca_order_id="buy-causal-fill-valid",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=100,
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                valid_position_id,
                buy_status="partially_filled",
                filled_qty=0.4,
                filled_avg_price=100.005,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
        )
        self.assertIsNone(
            self.conn.execute(
                "SELECT notes FROM alpaca_managed_positions WHERE id = ?",
                (valid_position_id,),
            ).fetchone()[0]
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                valid_position_id,
                buy_status="filled",
                filled_qty=1,
                filled_avg_price=100,
                filled_at="2026-01-02T14:32:00Z",
                target_sell_price=150,
            )
        )

    def test_buy_intent_backfill_quarantines_an_existing_impossible_fill(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="BACKFILL",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-causality-intent-backfill",
            buy_alpaca_order_id="buy-causality-intent-backfill",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=99,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT buy_order_qty, buy_order_limit_price, buy_causality_quarantine "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            (None, None, None),
        )

        self.assertEqual(
            save_alpaca_managed_buy_order(
                self.conn,
                symbol="BACKFILL",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="buy-causality-intent-backfill",
                buy_alpaca_order_id="buy-causality-intent-backfill",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="filled",
                buy_order_qty=1,
                buy_order_limit_price=100,
            ),
            position_id,
        )

        persisted = self.conn.execute(
            "SELECT buy_order_qty, buy_order_limit_price, buy_status, buy_causality_quarantine "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        self.assertEqual(persisted[:3], (1, 100, "filled"))
        self.assertIn("quantity exceeds the immutable managed buy intent", persisted[3])
        self.assertTrue(load_alpaca_managed_positions(self.conn)["id"].eq(position_id).any())
        init_state_db(self.conn)

    def test_buy_status_mutators_preserve_quarantine_when_leaving_diagnostic_status(self) -> None:
        for mutator_name, existing_marker in (
            ("update_alpaca_managed_buy_status", None),
            ("update_alpaca_managed_buy_status_if_current", "stale causality marker"),
        ):
            with self.subTest(mutator=mutator_name), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="STATUS",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"buy-causality-{mutator_name}",
                    buy_alpaca_order_id=f"buy-causality-{mutator_name}",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                )
                conn.execute(
                    """
                    UPDATE alpaca_managed_positions
                    SET buy_status = 'identity_mismatch', filled_qty = 2,
                        filled_avg_price = 99, target_sell_price = 150,
                        remaining_qty = 2, buy_causality_quarantine = ?
                    WHERE id = ?
                    """,
                    (existing_marker, position_id),
                )
                conn.commit()
                self.assertTrue(load_alpaca_managed_positions(conn)["id"].eq(position_id).any())

                mutator = getattr(storage_module, mutator_name)
                applied = mutator(
                    conn,
                    position_id,
                    expected_buy_status="identity_mismatch",
                    expected_buy_alpaca_order_id=f"buy-causality-{mutator_name}",
                    expected_filled_qty=2,
                    expected_sell_client_order_id=None,
                    buy_status="canceled",
                )

                self.assertTrue(applied)
                persisted = conn.execute(
                    "SELECT buy_status, buy_causality_quarantine FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()
                self.assertEqual(persisted[0], "canceled")
                self.assertIn("quantity exceeds the immutable managed buy intent", persisted[1])
                self.assertTrue(load_alpaca_managed_positions(conn)["id"].eq(position_id).any())
                init_state_db(conn)

    def test_managed_lifecycle_status_storage_classes_are_validated_on_init_and_load(self) -> None:
        for column_name, nullable in (("buy_status", False), ("sell_status", True)):
            with self.subTest(column=column_name), closing(sqlite3.connect(":memory:")) as conn:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="STATUS",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"buy-blob-{column_name}",
                    buy_alpaca_order_id=f"buy-blob-{column_name}",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                )
                stored_status = conn.execute(f"SELECT {column_name} FROM alpaca_managed_positions").fetchone()[0]
                self.assertEqual(stored_status is None, nullable)
                conn.execute(
                    f"UPDATE alpaca_managed_positions SET {column_name} = ? WHERE id = ?",
                    (sqlite3.Binary(b"filled"), position_id),
                )
                conn.commit()

                with self.assertRaisesRegex(ValueError, "lifecycle status"):
                    load_alpaca_managed_positions(conn)
                with self.assertRaisesRegex(ValueError, "lifecycle status storage value"):
                    init_state_db(conn)
                self.assertEqual(
                    conn.execute(
                        f"SELECT TYPEOF({column_name}) FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    ("blob",),
                )

    def test_public_status_mutators_reject_values_that_sqlite_would_store_as_blobs(self) -> None:
        with self.assertRaisesRegex(ValueError, "string lifecycle status"):
            save_alpaca_managed_buy_order(
                self.conn,
                symbol="BLOB",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="buy-blob-status-input",
                buy_alpaca_order_id="buy-blob-status-input",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status=sqlite3.Binary(b"accepted"),  # type: ignore[arg-type]
                buy_order_qty=1,
                buy_order_limit_price=100,
            )

        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TEXT",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-text-status-input",
            buy_alpaca_order_id="buy-text-status-input",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=100,
        )
        with self.assertRaisesRegex(ValueError, "string lifecycle status"):
            update_alpaca_managed_buy_status_if_current(
                self.conn,
                position_id,
                expected_buy_status="accepted",
                expected_buy_alpaca_order_id="buy-text-status-input",
                expected_filled_qty=None,
                expected_sell_client_order_id=None,
                buy_status=sqlite3.Binary(b"filled"),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "string lifecycle status"):
            record_alpaca_managed_sell_order(
                self.conn,
                position_id,
                sell_client_order_id="sell-blob-status-input",
                sell_alpaca_order_id=None,
                sell_submitted_at=None,
                sell_status=sqlite3.Binary(b"accepted"),  # type: ignore[arg-type]
            )
        self.assertEqual(
            self.conn.execute(
                "SELECT buy_status, sell_status FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            ("accepted", None),
        )

    def test_imported_terminal_buy_underfill_requires_quarantine_marker(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-imported-unmarked-underfill",
            buy_alpaca_order_id="buy-imported-unmarked-underfill",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="filled",
            buy_order_qty=2,
            buy_order_limit_price=100,
        )
        # A terminal broker lifecycle observation can transiently precede its
        # fill economics and remains compatible with initialization and reads.
        init_state_db(self.conn)
        self.assertTrue(load_alpaca_managed_positions(self.conn)["id"].eq(position_id).any())
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET filled_qty = 1,
                filled_avg_price = 99,
                target_sell_price = 150,
                remaining_qty = 1,
                notes = NULL
            WHERE id = ?
            """,
            (position_id,),
        )
        self.conn.commit()
        corrupted = self.conn.execute(
            "SELECT filled_qty, filled_avg_price, notes, state_revision FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "unquarantined managed buy fill"):
            init_state_db(self.conn)

        self.assertEqual(
            self.conn.execute(
                "SELECT filled_qty, filled_avg_price, notes, state_revision FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            corrupted,
        )

    def test_loaded_terminal_buy_underfill_requires_quarantine_marker(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-loaded-unmarked-underfill",
            buy_alpaca_order_id="buy-loaded-unmarked-underfill",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=100,
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=1,
                filled_avg_price=99,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
        )
        supported = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
        self.assertIn("filled status reports less", supported["notes"])
        self.assertIn("realized-P/L publication is quarantined", supported["notes"])

        self.conn.execute(
            "UPDATE alpaca_managed_positions SET notes = NULL, buy_causality_quarantine = NULL WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()

        with self.assertRaisesRegex(ValueError, "invalid durable broker intent economics"):
            load_alpaca_managed_positions(self.conn)

        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_status = 'identity_mismatch', notes = 'structured buy quarantine' "
            "WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()
        quarantined = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
        self.assertEqual(quarantined["buy_status"], "identity_mismatch")
        init_state_db(self.conn)

    def test_structured_buy_causality_marker_survives_later_sell_notes(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="STICKY",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-sticky-causality-marker",
            buy_alpaca_order_id="buy-sticky-causality-marker",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=100,
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="canceled",
                filled_qty=2,
                filled_avg_price=99,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
        )
        marker = self.conn.execute(
            "SELECT buy_causality_quarantine FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()[0]
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="sell-sticky-causality-marker",
            sell_alpaca_order_id="sell-sticky-causality-marker",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
            notes="later sell-side diagnostic",
        )

        persisted = self.conn.execute(
            "SELECT notes, buy_causality_quarantine FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        self.assertEqual(persisted, ("later sell-side diagnostic", marker))
        self.assertIn("quantity exceeds the immutable managed buy intent", marker)
        self.assertTrue(load_alpaca_managed_positions(self.conn)["id"].eq(position_id).any())
        init_state_db(self.conn)

    def test_note_only_buy_causality_quarantine_is_promoted_during_migration(
        self,
    ) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="LEGACY",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-legacy-note-only-causality-marker",
            buy_alpaca_order_id="buy-legacy-note-only-causality-marker",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=100,
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="canceled",
                filled_qty=2,
                filled_avg_price=99,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
        )
        legacy_note = self.conn.execute(
            "SELECT notes FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()[0]
        self.conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN buy_causality_quarantine")
        self.conn.commit()

        init_state_db(self.conn)

        marker = self.conn.execute(
            "SELECT buy_causality_quarantine FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()[0]
        self.assertEqual(marker, legacy_note)
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="sell-legacy-note-only-causality-marker",
            sell_alpaca_order_id="sell-legacy-note-only-causality-marker",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
            notes="later sell-side diagnostic",
        )
        persisted = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
        self.assertEqual(persisted["notes"], "later sell-side diagnostic")
        self.assertEqual(persisted["buy_causality_quarantine"], marker)
        init_state_db(self.conn)

    def test_legacy_closed_sell_shortfall_note_is_promoted_during_migration(self) -> None:
        position_id = self.save_position(
            symbol="LEGACYSHORTFALL",
            client_order_id="buy-legacy-closed-sell-shortfall",
        )
        legacy_note = "authoritative Alpaca closed correction reopened shares with incomplete managed-sell coverage"
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET notes = ? WHERE id = ?",
            (legacy_note, position_id),
        )
        self.conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN closed_sell_shortfall_reopen_pending")
        self.conn.commit()

        init_state_db(self.conn)

        persisted = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
        self.assertEqual(persisted["closed_sell_shortfall_reopen_pending"], 1)
        self.assertEqual(persisted["notes"], legacy_note)
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET closed_sell_shortfall_reopen_pending = 0 WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()

        init_state_db(self.conn)

        self.assertEqual(
            load_alpaca_managed_positions(self.conn)
            .query("id == @position_id")
            .iloc[0]["closed_sell_shortfall_reopen_pending"],
            0,
        )

    def test_buy_fill_causality_is_fenced_to_the_intent_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = f"{temp_dir}/state.sqlite3"
            with (
                closing(sqlite3.connect(db_path, factory=BuyIntentBackfillRaceConnection)) as conn,
                closing(sqlite3.connect(db_path)) as peer,
            ):
                init_state_db(conn)
                conn.peer = peer
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="RACE",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="buy-causality-intent-race",
                    buy_alpaca_order_id="buy-causality-intent-race",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=None,
                    buy_order_limit_price=None,
                )
                conn.backfill_buy_intent_on_fetch = True

                applied = mark_alpaca_managed_buy_filled(
                    conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=2,
                    filled_avg_price=99,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )

                self.assertFalse(applied)
                self.assertEqual(
                    conn.execute(
                        "SELECT buy_order_qty, buy_order_limit_price, filled_qty, "
                        "buy_causality_quarantine FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    (1, 100, None, None),
                )
                self.assertTrue(load_alpaca_managed_positions(conn)["id"].eq(position_id).any())

                self.assertTrue(
                    mark_alpaca_managed_buy_filled(
                        conn,
                        position_id,
                        buy_status="filled",
                        filled_qty=2,
                        filled_avg_price=99,
                        filled_at="2026-01-02T14:31:00Z",
                        target_sell_price=150,
                    )
                )
                marker = conn.execute(
                    "SELECT buy_causality_quarantine FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()[0]
                self.assertIn("quantity exceeds the immutable managed buy intent", marker)

    def test_imported_buy_fill_economics_must_be_finite_and_internally_consistent(
        self,
    ) -> None:
        invalid_economics = (
            (-1, 100),
            (0, 100),
            (0.5, None),
            (0.5, -10),
            (None, 100),
            (math.inf, 100),
            (1, math.inf),
        )
        for index, (filled_qty, filled_avg_price) in enumerate(invalid_economics):
            with (
                self.subTest(filled_qty=filled_qty, filled_avg_price=filled_avg_price),
                closing(sqlite3.connect(":memory:")) as conn,
            ):
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol=f"INVALID{index}",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"buy-invalid-imported-fill-economics-{index}",
                    buy_alpaca_order_id=f"buy-invalid-imported-fill-economics-{index}",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                )
                conn.execute(
                    "UPDATE alpaca_managed_positions "
                    "SET buy_status = 'partially_filled', filled_qty = ?, filled_avg_price = ? "
                    "WHERE id = ?",
                    (filled_qty, filled_avg_price, position_id),
                )
                conn.commit()
                before = conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()

                with self.assertRaisesRegex(ValueError, "invalid durable broker intent economics"):
                    load_alpaca_managed_positions(conn)
                with self.assertRaisesRegex(ValueError, "unquarantined managed buy fill"):
                    init_state_db(conn)

                self.assertFalse(conn.in_transaction)
                self.assertEqual(
                    conn.execute(
                        "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    before,
                )

    def test_impossible_buy_fill_preserves_existing_notes_without_duplication(
        self,
    ) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-causal-note-preservation",
            buy_alpaca_order_id="buy-causal-note-preservation",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=100,
            notes="prior broker context",
        )
        fill_arguments = {
            "buy_status": "filled",
            "filled_qty": 2,
            "filled_avg_price": 99,
            "filled_at": "2026-01-02T14:31:00Z",
            "target_sell_price": 150,
        }

        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                **fill_arguments,
            )
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                **fill_arguments,
            )
        )
        notes = self.conn.execute(
            "SELECT notes FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()[0]

        self.assertIn("prior broker context", notes)
        self.assertEqual(
            notes.count("quantity exceeds the immutable managed buy intent"),
            1,
        )

    def test_legacy_null_buy_intent_can_still_record_a_causal_fill(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="OLD",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-OLD-legacy-null-intent",
            buy_alpaca_order_id="buy-legacy-null-intent",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="filled",
            buy_order_qty=None,
            buy_order_limit_price=None,
        )

        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
        )

    def test_closed_correction_quarantines_impossible_buy_fill(self) -> None:
        for index, (filled_qty, filled_avg_price, issue) in enumerate(
            (
                (3, 100, "quantity exceeds"),
                (1, 100, "filled status reports less"),
                (2, 105.006, "average price exceeds"),
            )
        ):
            position_id = self.save_position(
                symbol=f"BAD{index}",
                client_order_id=f"rsi-buy-impossible-closed-correction-{index}",
                alpaca_asset_id=f"asset-bad-{index}",
            )
            self.close_position(position_id)
            snapshot = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
            with self.subTest(fill=(filled_qty, filled_avg_price)):
                applied, _reopened, _conflict, _remaining = apply_alpaca_closed_position_broker_correction(
                    self.conn,
                    position_id,
                    expected_closed_at=str(snapshot["closed_at"]),
                    expected_state_revision=int(snapshot["state_revision"]),
                    alpaca_asset_id=f"asset-bad-{index}",
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                    buy_status="filled",
                    filled_qty=filled_qty,
                    filled_avg_price=filled_avg_price,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                    sell_status="filled",
                    sell_fills=[(f"sell-impossible-correction-{index}", 2, 300)],
                    sell_filled_at="2026-01-02T14:33:00Z",
                    notes="impossible correction",
                )
                self.assertTrue(applied)
            recorded = self.conn.execute(
                "SELECT filled_qty, filled_avg_price, notes FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone()
            self.assertEqual(recorded[:2], (filled_qty, filled_avg_price))
            self.assertIn(issue, recorded[2])
            self.assertIn("realized-P/L publication is quarantined", recorded[2])

    def test_closed_legacy_correction_retains_causality_and_existing_notes(
        self,
    ) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="LEG",
            alpaca_asset_id="asset-leg",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-legacy-causal-correction",
            buy_alpaca_order_id="buy-legacy-causal-correction",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="filled",
            buy_order_qty=None,
            buy_order_limit_price=None,
            notes="prior closed context",
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=1,
            filled_avg_price=99,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]

        applied, reopened, conflict, remaining = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-leg",
            buy_order_qty=1,
            buy_order_limit_price=100,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=99,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="filled",
            sell_fills=[("sell-legacy-causal-correction", 2, 300)],
            sell_filled_at="2026-01-02T14:33:00Z",
            notes="correction context",
        )
        row = self.conn.execute(
            "SELECT buy_order_qty, buy_order_limit_price, filled_qty, notes FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertTrue(applied)
        self.assertFalse(reopened)
        self.assertFalse(conflict)
        self.assertEqual(remaining, 0)
        self.assertEqual(row[:3], (1, 100, 2))
        self.assertIn("prior closed context", row[3])
        self.assertIn("correction context", row[3])
        self.assertEqual(
            row[3].count("quantity exceeds the immutable managed buy intent"),
            1,
        )

    def test_sell_fill_base_exception_rolls_back_ledger_and_position(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-base-exception-fill",
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="rsi-exit-TQQQ-base-exception-fill",
            sell_alpaca_order_id="sell-base-exception-fill",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        ledger_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
            (position_id,),
        ).fetchall()

        with (
            patch(
                "leveraged_trader.storage.managed_residual_quantity_is_negligible",
                side_effect=KeyboardInterrupt("interrupted accounting"),
            ),
            self.assertRaisesRegex(KeyboardInterrupt, "interrupted accounting"),
        ):
            mark_alpaca_managed_sell_filled_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id="rsi-exit-TQQQ-base-exception-fill",
                sell_status="partially_filled",
                sell_filled_qty=1,
                sell_filled_avg_price=110,
                sell_filled_at="2026-01-02T14:33:00Z",
                sell_broker_updated_at="2026-01-02T14:33:00Z",
                sell_alpaca_order_id="sell-base-exception-fill",
            )

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchall(),
            ledger_before,
        )

    def test_owned_sell_fill_transactions_serialize_before_reading_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/state.sqlite"
            with closing(sqlite3.connect(db_path)) as setup_conn:
                init_state_db(setup_conn)
                position_id = save_alpaca_managed_buy_order(
                    setup_conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-concurrent-storage",
                    buy_alpaca_order_id="buy-concurrent-storage",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="filled",
                )
                mark_alpaca_managed_buy_filled(
                    setup_conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=5,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                record_alpaca_managed_sell_order(
                    setup_conn,
                    position_id,
                    sell_client_order_id="rsi-exit-TQQQ-concurrent-storage",
                    sell_alpaca_order_id="sell-concurrent-storage",
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="partially_filled",
                )

            barrier = threading.Barrier(2)
            results: list[bool] = []
            errors: list[BaseException] = []

            def record_fill(observed_qty: float) -> None:
                with closing(sqlite3.connect(db_path, timeout=5.0)) as fill_conn:
                    fill_conn.execute("PRAGMA busy_timeout = 5000")
                    barrier.wait()
                    try:
                        _remaining, current = mark_alpaca_managed_sell_filled_if_current(
                            fill_conn,
                            position_id,
                            expected_sell_client_order_id="rsi-exit-TQQQ-concurrent-storage",
                            sell_status="partially_filled",
                            sell_filled_qty=observed_qty,
                            sell_filled_avg_price=150,
                            sell_filled_at="2026-01-02T15:00:00Z",
                            sell_alpaca_order_id="sell-concurrent-storage",
                        )
                    except SellFillQuantityRegressionError:
                        results.append(False)
                    except BaseException as exc:  # pragma: no cover - asserted below
                        errors.append(exc)
                    else:
                        results.append(current)

            threads = [threading.Thread(target=record_fill, args=(quantity,)) for quantity in (2.0, 3.0)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10.0)

            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertIn(True, results)
            with closing(sqlite3.connect(db_path)) as verify_conn:
                ledger = verify_conn.execute(
                    """
                    SELECT filled_qty, filled_value
                    FROM alpaca_managed_sell_fills
                    WHERE managed_position_id = ? AND alpaca_order_id = ?
                    """,
                    (position_id, "sell-concurrent-storage"),
                ).fetchone()
                parent = verify_conn.execute(
                    """
                    SELECT sold_qty, sold_value, remaining_qty
                    FROM alpaca_managed_positions
                    WHERE id = ?
                    """,
                    (position_id,),
                ).fetchone()

        self.assertEqual(ledger, (3.0, 450.0))
        self.assertEqual(parent, (3.0, 450.0, 2.0))

    def test_closed_correction_base_exception_rolls_back_revision_fence(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-base-exception-correction",
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with (
            patch(
                "leveraged_trader.storage.managed_residual_quantity_is_negligible",
                side_effect=SystemExit("interrupted correction"),
            ),
            self.assertRaisesRegex(SystemExit, "interrupted correction"),
        ):
            apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
                sell_status="filled",
                sell_fills=[("sell-base-exception-correction", 1, 110)],
                sell_filled_at="2026-01-02T14:33:00Z",
                notes="interrupted closed correction",
            )

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchone(),
            (0,),
        )

    def test_managed_buy_broker_observation_cannot_replace_frozen_strategy_identity(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            workflow="Long",
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-frozen-strategy",
            buy_alpaca_order_id=None,
            buy_submitted_at=None,
            buy_status="submission_pending",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )

        with self.assertRaisesRegex(RuntimeError, "persistence race"):
            save_alpaca_managed_buy_order(
                self.conn,
                workflow="Short",
                symbol="TQQQ",
                alpaca_asset_id="asset-tqqq",
                signal_symbol="SPY",
                buy_rsi=70,
                profit_target_multiple=2,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-frozen-strategy",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="filled",
                buy_order_qty=2,
                buy_order_limit_price=105,
                require_applied=True,
            )
        row = self.conn.execute(
            """
            SELECT workflow, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                   buy_status, buy_alpaca_order_id
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()

        self.assertEqual(
            row,
            ("Long", "TQQQ", "QQQ", 30, 1.5, "submission_pending", None),
        )

    def test_active_buy_observation_cannot_erase_pending_cancel_ownership(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            workflow="Long",
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-sticky-pending-cancel",
            buy_alpaca_order_id=None,
            buy_submitted_at=None,
            buy_status="submission_pending",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        quarantined = update_alpaca_managed_buy_status_if_current(
            self.conn,
            position_id,
            expected_buy_status="submission_pending",
            expected_buy_alpaca_order_id=None,
            expected_filled_qty=None,
            expected_sell_client_order_id=None,
            expected_buy_submission_attempt_count=1,
            buy_status="pending_cancel",
            buy_alpaca_order_id="buy-1",
            notes="cancellation ownership is durable",
        )
        before_replay = self.conn.execute(
            """
            SELECT state_revision, buy_status, buy_alpaca_order_id,
                   buy_observation_broker_updated_at, notes
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(RuntimeError, "persistence race"):
            save_alpaca_managed_buy_order(
                self.conn,
                workflow="Long",
                symbol="TQQQ",
                alpaca_asset_id="asset-tqqq",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="rsi-buy-TQQQ-sticky-pending-cancel",
                buy_alpaca_order_id="buy-1",
                buy_submitted_at="2026-01-02T14:30:00Z",
                buy_status="accepted",
                buy_order_qty=2,
                buy_order_limit_price=105,
                require_applied=True,
                expected_buy_submission_attempt_count=1,
                buy_broker_updated_at="2026-01-02T14:30:00Z",
            )
        after_replay = self.conn.execute(
            """
            SELECT state_revision, buy_status, buy_alpaca_order_id,
                   buy_observation_broker_updated_at, notes
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (position_id,),
        ).fetchone()

        self.assertTrue(quarantined)
        self.assertEqual(after_replay, before_replay)
        self.assertEqual(
            after_replay,
            (1, "pending_cancel", "buy-1", None, "cancellation ownership is durable"),
        )

    def test_missing_buy_adoption_cas_checks_frozen_strategy_identity(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            workflow="Long",
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-frozen-adoption",
            buy_alpaca_order_id=None,
            buy_submitted_at=None,
            buy_status="submission_not_found",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.close_position(position_id)

        conflicting_adoption = adopt_alpaca_managed_buy_order_if_submission_not_found(
            self.conn,
            workflow="Short",
            symbol="TQQQ",
            signal_symbol="SPY",
            buy_rsi=70,
            profit_target_multiple=2,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-frozen-adoption",
            buy_alpaca_order_id="buy-1",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
            alpaca_asset_id="asset-tqqq",
        )
        after_conflict = self.conn.execute(
            "SELECT buy_status, buy_alpaca_order_id, closed_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        matching_adoption = adopt_alpaca_managed_buy_order_if_submission_not_found(
            self.conn,
            workflow="Long",
            symbol="tqqq",
            signal_symbol="qqq",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-frozen-adoption",
            buy_alpaca_order_id="buy-1",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
            alpaca_asset_id="asset-tqqq",
        )
        after_match = self.conn.execute(
            "SELECT buy_status, buy_alpaca_order_id, closed_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(conflicting_adoption)
        self.assertEqual(after_conflict[:2], ("submission_not_found", None))
        self.assertIsNotNone(after_conflict[2])
        self.assertTrue(matching_adoption)
        self.assertEqual(after_match, ("accepted", "buy-1", None))

    def test_missing_buy_adoption_requires_exact_immutable_order_economics(self) -> None:
        for suffix, observed_qty, observed_price in (
            ("quantity", 2_000_000.01, 100),
            ("price", 2_000_000, 100.000000001),
        ):
            with self.subTest(suffix=suffix):
                client_order_id = f"rsi-buy-TQQQ-exact-adoption-{suffix}"
                position_id = save_alpaca_managed_buy_order(
                    self.conn,
                    workflow="Long",
                    symbol="TQQQ",
                    alpaca_asset_id=f"asset-tqqq-{suffix}",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=client_order_id,
                    buy_alpaca_order_id=None,
                    buy_submitted_at=None,
                    buy_status="submission_not_found",
                    buy_order_qty=2_000_000,
                    buy_order_limit_price=100,
                )
                self.close_position(position_id)

                with self.assertRaises(ValueError):
                    adopt_alpaca_managed_buy_order_if_submission_not_found(
                        self.conn,
                        workflow="Long",
                        symbol="TQQQ",
                        signal_symbol="QQQ",
                        buy_rsi=30,
                        profit_target_multiple=1.5,
                        buy_signal_date="2026-01-02",
                        buy_client_order_id=client_order_id,
                        buy_alpaca_order_id=f"buy-mismatched-{suffix}",
                        buy_submitted_at="2026-01-02T14:30:00Z",
                        buy_status="accepted",
                        buy_order_qty=observed_qty,
                        buy_order_limit_price=observed_price,
                        alpaca_asset_id=f"asset-tqqq-{suffix}",
                    )
                after_mismatch = self.conn.execute(
                    "SELECT buy_status, buy_alpaca_order_id, closed_at FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()
                matching = adopt_alpaca_managed_buy_order_if_submission_not_found(
                    self.conn,
                    workflow="Long",
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=client_order_id,
                    buy_alpaca_order_id=f"buy-matching-{suffix}",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=2_000_000,
                    buy_order_limit_price=100,
                    alpaca_asset_id=f"asset-tqqq-{suffix}",
                )

                self.assertEqual(after_mismatch[:2], ("submission_not_found", None))
                self.assertIsNotNone(after_mismatch[2])
                self.assertTrue(matching)
                self.close_position(position_id)

    def test_whole_share_residual_and_overfill_never_fall_within_quantity_tolerance(self) -> None:
        buy_qty = 100_000_000
        for suffix, sell_qty, expected_remaining in (
            ("shortfall", buy_qty - 1, 1),
            ("overfill", buy_qty + 1, -1),
            ("half-shortfall", buy_qty - 0.5, 0.5),
            ("half-overfill", buy_qty + 0.5, -0.5),
        ):
            with self.subTest(suffix=suffix):
                position_id = save_alpaca_managed_buy_order(
                    self.conn,
                    symbol=f"BIG-{suffix}",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"rsi-buy-BIG-{suffix}",
                    buy_alpaca_order_id=f"buy-{suffix}",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="filled",
                    buy_order_qty=buy_qty,
                    buy_order_limit_price=105,
                )
                mark_alpaca_managed_buy_filled(
                    self.conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=buy_qty,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                sell_client_order_id = f"rsi-exit-BIG-{suffix}"
                sell_alpaca_order_id = f"sell-{suffix}"
                record_alpaca_managed_sell_order(
                    self.conn,
                    position_id,
                    sell_client_order_id=sell_client_order_id,
                    sell_alpaca_order_id=sell_alpaca_order_id,
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="accepted",
                    sell_order_qty=sell_qty,
                    sell_order_limit_price=150,
                )
                mark_alpaca_managed_sell_filled_if_current(
                    self.conn,
                    position_id,
                    expected_sell_client_order_id=sell_client_order_id,
                    sell_status="filled",
                    sell_filled_qty=sell_qty,
                    sell_filled_avg_price=150,
                    sell_filled_at="2026-01-02T14:33:00Z",
                    sell_broker_updated_at="2026-01-02T14:33:00Z",
                    sell_alpaca_order_id=sell_alpaca_order_id,
                )

                closed = close_alpaca_managed_position_if_current_and_complete(
                    self.conn,
                    position_id,
                    expected_sell_client_order_id=sell_client_order_id,
                    expected_sell_alpaca_order_id=sell_alpaca_order_id,
                    expected_sell_status="filled",
                    closed_at="2026-01-02T14:33:00Z",
                )
                remaining_qty, closed_at, realized_pl = self.conn.execute(
                    "SELECT remaining_qty, closed_at, realized_pl FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()

                self.assertFalse(closed)
                self.assertEqual(remaining_qty, expected_remaining)
                self.assertIsNone(closed_at)
                if expected_remaining < 0:
                    self.assertIsNone(realized_pl)

    def test_active_position_asset_adoption_is_fenced_and_conflict_safe(self) -> None:
        legacy_id = self.save_position(
            symbol="LEG",
            client_order_id="rsi-buy-LEG-20260102",
            alpaca_asset_id=None,
        )
        self.save_position(
            symbol="OWN",
            client_order_id="rsi-buy-OWN-20260102",
            alpaca_asset_id="asset-owned",
        )
        revision = int(
            self.conn.execute(
                "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                (legacy_id,),
            ).fetchone()[0]
        )

        conflict = adopt_alpaca_managed_position_asset_if_current(
            self.conn,
            legacy_id,
            expected_state_revision=revision,
            alpaca_asset_id="asset-owned",
        )
        adopted_revision = adopt_alpaca_managed_position_asset_if_current(
            self.conn,
            legacy_id,
            expected_state_revision=revision,
            alpaca_asset_id=" asset-legacy ",
        )
        stale = adopt_alpaca_managed_position_asset_if_current(
            self.conn,
            legacy_id,
            expected_state_revision=revision,
            alpaca_asset_id="asset-other",
        )
        row = self.conn.execute(
            "SELECT alpaca_asset_id, state_revision FROM alpaca_managed_positions WHERE id = ?",
            (legacy_id,),
        ).fetchone()
        aliases = self.conn.execute(
            "SELECT symbol FROM alpaca_symbol_aliases WHERE alpaca_asset_id = 'asset-legacy'"
        ).fetchall()

        self.assertIsNone(conflict)
        self.assertEqual(adopted_revision, revision + 1)
        self.assertIsNone(stale)
        self.assertEqual(row, ("asset-legacy", revision + 1))
        self.assertEqual(aliases, [("LEG",)])

    def test_asset_adoption_propagates_deferred_commit_integrity_failure(self) -> None:
        legacy_id = self.save_position(
            symbol="LEG",
            client_order_id="rsi-buy-LEG-deferred-adoption",
            alpaca_asset_id=None,
        )
        revision = int(
            self.conn.execute(
                "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                (legacy_id,),
            ).fetchone()[0]
        )
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("CREATE TABLE deferred_asset_parent (alpaca_asset_id TEXT PRIMARY KEY)")
        self.conn.execute(
            """
            CREATE TABLE deferred_asset_guard (
                alpaca_asset_id TEXT NOT NULL,
                FOREIGN KEY (alpaca_asset_id)
                    REFERENCES deferred_asset_parent(alpaca_asset_id)
                    DEFERRABLE INITIALLY DEFERRED
            )
            """
        )
        self.conn.execute(
            """
            CREATE TRIGGER deferred_asset_adoption_guard
            AFTER UPDATE OF alpaca_asset_id ON alpaca_managed_positions
            WHEN NEW.alpaca_asset_id IS NOT NULL
            BEGIN
                INSERT INTO deferred_asset_guard (alpaca_asset_id)
                VALUES (NEW.alpaca_asset_id);
            END
            """
        )
        self.conn.commit()

        with self.assertRaises(sqlite3.IntegrityError):
            adopt_alpaca_managed_position_asset_if_current(
                self.conn,
                legacy_id,
                expected_state_revision=revision,
                alpaca_asset_id="asset-deferred-missing",
            )

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT alpaca_asset_id, state_revision FROM alpaca_managed_positions WHERE id = ?",
                (legacy_id,),
            ).fetchone(),
            (None, revision),
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM deferred_asset_guard").fetchone(),
            (0,),
        )

    def test_init_state_db_adds_buy_fill_broker_timestamp_to_legacy_positions(self) -> None:
        with closing(sqlite3.connect(":memory:")) as legacy_conn, legacy_conn:
            init_state_db(legacy_conn)
            legacy_conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN buy_fill_broker_updated_at")
            legacy_conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN buy_observation_broker_updated_at")
            legacy_conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN buy_fill_component_revisions")
            legacy_conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN buy_fill_pending_observation")
            legacy_conn.execute("ALTER TABLE alpaca_managed_positions DROP COLUMN buy_cancellation_alpaca_order_ids")

            init_state_db(legacy_conn)

            columns = {str(row[1]) for row in legacy_conn.execute("PRAGMA table_info(alpaca_managed_positions)")}

        self.assertIn("buy_fill_broker_updated_at", columns)
        self.assertIn("buy_observation_broker_updated_at", columns)
        self.assertIn("buy_fill_component_revisions", columns)
        self.assertIn("buy_fill_pending_observation", columns)
        self.assertIn("buy_cancellation_alpaca_order_ids", columns)

    def test_zero_fill_buy_status_rejects_older_terminal_broker_revision(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-zero-fill-ordering",
            buy_alpaca_order_id="buy-zero-fill-ordering",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.assertTrue(
            update_alpaca_managed_buy_status_if_current(
                self.conn,
                position_id,
                expected_buy_status="accepted",
                expected_buy_alpaca_order_id="buy-zero-fill-ordering",
                expected_filled_qty=None,
                expected_sell_client_order_id=None,
                buy_status="new",
                buy_alpaca_order_id="buy-zero-fill-ordering",
                buy_broker_updated_at="2026-07-20T12:02:00Z",
            )
        )
        before = self.conn.execute(
            "SELECT state_revision, buy_status, buy_observation_broker_updated_at, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        stale_closed = close_alpaca_managed_buy_if_current_and_unfilled(
            self.conn,
            position_id,
            expected_buy_status="new",
            expected_buy_alpaca_order_id="buy-zero-fill-ordering",
            buy_status="canceled",
            buy_alpaca_order_id="buy-zero-fill-ordering",
            closed_at="2026-07-20T12:01:00Z",
            buy_broker_updated_at="2026-07-20T12:01:00Z",
        )
        after = self.conn.execute(
            "SELECT state_revision, buy_status, buy_observation_broker_updated_at, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(stale_closed)
        self.assertEqual(after, before)
        self.assertEqual(after[1:], ("new", "2026-07-20T12:02:00.000000Z", None))

    def test_first_active_buy_fill_rejects_older_lifecycle_revision(self) -> None:
        order_id = "buy-active-first-fill-lifecycle-fence"
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-active-first-fill-lifecycle-fence",
            buy_alpaca_order_id=order_id,
            buy_submitted_at="2026-07-20T12:00:00Z",
            buy_status="new",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_broker_updated_at="2026-07-20T12:02:00Z",
        )
        before = self.conn.execute(
            "SELECT state_revision, buy_status, filled_qty, buy_observation_broker_updated_at, "
            "buy_fill_broker_updated_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        applied = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-07-20T12:01:00Z",
            target_sell_price=150,
            buy_fill_broker_updated_at="2026-07-20T12:01:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:01:00Z",
            buy_fill_component_revisions={order_id: "2026-07-20T12:01:00Z"},
            buy_alpaca_order_id=order_id,
            expected_buy_status="new",
            expected_buy_alpaca_order_id=order_id,
            expected_filled_qty=None,
            expected_filled_avg_price=None,
            expected_target_sell_price=None,
            expected_sell_client_order_id=None,
        )
        after = self.conn.execute(
            "SELECT state_revision, buy_status, filled_qty, buy_observation_broker_updated_at, "
            "buy_fill_broker_updated_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(applied)
        self.assertEqual(after, before)

    def test_buy_component_revision_cannot_authorize_unchanged_ancestor_fill_change(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-component-accounting",
            buy_alpaca_order_id="buy-root",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        root_revision = "2026-07-20T12:01:00Z"
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=1,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
                buy_fill_broker_updated_at=root_revision,
                buy_fill_broker_oldest_updated_at=root_revision,
                buy_fill_component_revisions={
                    "buy-root": {
                        "broker_updated_at": root_revision,
                        "filled_qty": 1,
                        "filled_value": 100,
                    }
                },
                buy_alpaca_order_id="buy-root",
            )
        )
        before = self.conn.execute(
            "SELECT state_revision, buy_status, filled_qty, filled_avg_price, "
            "buy_fill_broker_updated_at, buy_fill_component_revisions "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        applied = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=150,
            buy_fill_broker_updated_at="2026-07-20T12:02:00Z",
            buy_fill_broker_oldest_updated_at=root_revision,
            buy_fill_component_revisions={
                "buy-root": {
                    "broker_updated_at": root_revision,
                    "filled_qty": 2,
                    "filled_value": 200,
                },
                "buy-leaf": {
                    "broker_updated_at": "2026-07-20T12:02:00Z",
                    "filled_qty": 0,
                    "filled_value": 0,
                },
            },
            buy_alpaca_order_id="buy-leaf",
            expected_buy_status="partially_filled",
            expected_buy_alpaca_order_id="buy-root",
            expected_filled_qty=1,
            expected_filled_avg_price=100,
            expected_target_sell_price=150,
            expected_sell_client_order_id=None,
        )
        after = self.conn.execute(
            "SELECT state_revision, buy_status, filled_qty, filled_avg_price, "
            "buy_fill_broker_updated_at, buy_fill_component_revisions "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(applied)
        self.assertEqual(after, before)

    def test_closed_buy_component_revision_cannot_reopen_from_unchanged_ancestor(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-closed-component-accounting",
            buy_alpaca_order_id="buy-root",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        root_revision = "2026-07-20T12:01:00Z"
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            buy_fill_broker_updated_at=root_revision,
            buy_fill_broker_oldest_updated_at=root_revision,
            buy_fill_component_revisions={
                "buy-root": {
                    "broker_updated_at": root_revision,
                    "filled_qty": 1,
                    "filled_value": 100,
                }
            },
            buy_alpaca_order_id="buy-root",
        )
        self.close_position(position_id)
        before = load_alpaca_managed_positions(self.conn).iloc[0]

        result = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(before["closed_at"]),
            expected_state_revision=int(before["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=150,
            sell_status=None,
            sell_fills=[],
            sell_filled_at=None,
            notes="conflicting unchanged ancestor accounting",
            buy_fill_broker_updated_at="2026-07-20T12:02:00Z",
            buy_fill_broker_oldest_updated_at=root_revision,
            buy_fill_component_revisions={
                "buy-root": {
                    "broker_updated_at": root_revision,
                    "filled_qty": 2,
                    "filled_value": 200,
                },
                "buy-leaf": {
                    "broker_updated_at": "2026-07-20T12:02:00Z",
                    "filled_qty": 0,
                    "filled_value": 0,
                },
            },
            force_reopen_buy=True,
        )
        after = load_alpaca_managed_positions(self.conn).iloc[0]

        self.assertEqual(result, (False, False, False, 0.0))
        self.assertEqual(after["filled_qty"], 1)
        self.assertFalse(pd.isna(after["closed_at"]))

    def test_legacy_active_zero_buy_fill_does_not_bypass_lifecycle_revision(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-legacy-zero-lifecycle",
        )
        buy_order_id = "buy-rsi-buy-TQQQ-legacy-zero-lifecycle"
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET buy_status = 'canceled', filled_qty = 0, filled_avg_price = NULL,
                filled_at = NULL, target_sell_price = NULL, remaining_qty = 0,
                buy_fill_broker_updated_at = '2026-07-20T12:00:00.000000Z',
                buy_fill_component_revisions = NULL,
                buy_observation_broker_updated_at = '2026-07-20T12:02:00.000000Z'
            WHERE id = ?
            """,
            (position_id,),
        )
        self.conn.commit()

        def apply_fill(revision: str) -> bool:
            return bool(
                mark_alpaca_managed_buy_filled(
                    self.conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=1,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                    buy_fill_broker_updated_at=revision,
                    buy_fill_broker_oldest_updated_at=revision,
                    buy_fill_component_revisions={
                        buy_order_id: {
                            "broker_updated_at": revision,
                            "filled_qty": 1,
                            "filled_value": 100,
                        }
                    },
                    buy_alpaca_order_id=buy_order_id,
                    expected_buy_status="canceled",
                    expected_buy_alpaca_order_id=buy_order_id,
                    expected_filled_qty=0,
                    expected_filled_avg_price=None,
                    expected_target_sell_price=None,
                    expected_sell_client_order_id=None,
                )
            )

        applied = apply_fill("2026-07-20T12:01:00Z")
        equal_revision_applied = apply_fill("2026-07-20T12:02:00Z")
        row = self.conn.execute(
            "SELECT buy_status, filled_qty, buy_observation_broker_updated_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(applied)
        self.assertFalse(equal_revision_applied)
        self.assertEqual(row, ("canceled", 0, "2026-07-20T12:02:00.000000Z"))

    def test_managed_buy_fill_rejects_invalid_numeric_inputs(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-invalid-fill-inputs",
        )
        invalid_values = (
            {"filled_qty": 0},
            {"filled_qty": -1},
            {"filled_qty": float("nan")},
            {"filled_qty": float("inf")},
            {"filled_avg_price": 0},
            {"filled_avg_price": float("nan")},
            {"target_sell_price": 0},
            {"target_sell_price": float("inf")},
        )
        base = {
            "buy_status": "filled",
            "filled_qty": 2,
            "filled_avg_price": 100,
            "filled_at": "2026-01-02T14:31:00Z",
            "target_sell_price": 150,
        }

        for invalid in invalid_values:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                mark_alpaca_managed_buy_filled(
                    self.conn,
                    position_id,
                    **(base | invalid),
                )

    def test_exact_buy_fill_replay_does_not_advance_state_revision(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-exact-fill-replay",
        )
        order_id = "buy-rsi-buy-TQQQ-exact-fill-replay"
        revisions = {order_id: "2026-07-20T12:02:00Z"}
        common = {
            "buy_status": "filled",
            "filled_qty": 2,
            "filled_avg_price": 100,
            "filled_at": "2026-01-02T14:31:00Z",
            "target_sell_price": 150,
            "buy_fill_broker_updated_at": "2026-07-20T12:02:00Z",
            "buy_fill_broker_oldest_updated_at": "2026-07-20T12:02:00Z",
            "buy_fill_component_revisions": revisions,
            "buy_alpaca_order_id": order_id,
            "expected_buy_status": "filled",
            "expected_buy_alpaca_order_id": order_id,
            "expected_filled_qty": 2,
            "expected_filled_avg_price": 100,
            "expected_target_sell_price": 150,
            "expected_sell_client_order_id": None,
        }
        self.assertTrue(mark_alpaca_managed_buy_filled(self.conn, position_id, **common))
        before = self.conn.execute(
            "SELECT state_revision, buy_status, filled_qty, filled_avg_price, filled_at, "
            "target_sell_price, buy_fill_broker_updated_at, buy_fill_component_revisions "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        replay_result = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            **common,
            return_state_revision=True,
        )
        after = self.conn.execute(
            "SELECT state_revision, buy_status, filled_qty, filled_avg_price, filled_at, "
            "target_sell_price, buy_fill_broker_updated_at, buy_fill_component_revisions "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(replay_result, (True, before[0]))
        self.assertEqual(after, before)

    def test_buy_fill_replay_preserves_scale_aware_realized_accounting(self) -> None:
        sell_qty = float(np.nextafter(100.0, np.inf))
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-scale-aware-replay",
            buy_alpaca_order_id="buy-scale-aware-replay",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="filled",
            buy_order_qty=100,
            buy_order_limit_price=105,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=100,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
        )
        sell_client_order_id = "rsi-exit-TQQQ-scale-aware-replay"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-scale-aware-replay",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="filled",
            sell_order_qty=sell_qty,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=sell_qty,
            sell_filled_avg_price=150,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_broker_updated_at="2026-07-20T12:01:00Z",
            sell_alpaca_order_id="sell-scale-aware-replay",
        )
        before = self.conn.execute(
            "SELECT realized_pl, realized_pl_pct FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        common = {
            "buy_status": "filled",
            "filled_qty": 100,
            "filled_avg_price": 100,
            "filled_at": "2026-01-02T14:31:00Z",
            "target_sell_price": 150,
            "buy_fill_broker_updated_at": "2026-07-20T12:02:00Z",
            "buy_fill_broker_oldest_updated_at": "2026-07-20T12:02:00Z",
            "buy_fill_component_revisions": {
                "buy-scale-aware-replay": "2026-07-20T12:02:00Z",
            },
            "buy_alpaca_order_id": "buy-scale-aware-replay",
            "expected_buy_status": "filled",
            "expected_buy_alpaca_order_id": "buy-scale-aware-replay",
            "expected_filled_qty": 100,
            "expected_filled_avg_price": 100,
            "expected_target_sell_price": 150,
            "expected_sell_client_order_id": sell_client_order_id,
        }

        self.assertTrue(mark_alpaca_managed_buy_filled(self.conn, position_id, **common))
        after_mutation = self.conn.execute(
            "SELECT state_revision, realized_pl, realized_pl_pct FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        self.assertTrue(mark_alpaca_managed_buy_filled(self.conn, position_id, **common))
        after_replay = self.conn.execute(
            "SELECT state_revision, realized_pl, realized_pl_pct FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertIsNotNone(before[0])
        self.assertIsNotNone(before[1])
        self.assertAlmostEqual(after_mutation[1], before[0])
        self.assertAlmostEqual(after_mutation[2], before[1])
        self.assertEqual(after_replay, after_mutation)

    def test_buy_fill_advance_preserves_material_tiny_sell_accounting(self) -> None:
        sell_qty = 1e-9
        sell_price = 120.0
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-tiny-sell-buy-advance",
            buy_alpaca_order_id="buy-tiny-sell-buy-advance",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
        )
        sell_client_order_id = "rsi-exit-TQQQ-tiny-sell-buy-advance"
        sell_order_id = "sell-tiny-sell-buy-advance"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=sell_order_id,
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="partially_filled",
            sell_order_qty=1,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="partially_filled",
            sell_filled_qty=sell_qty,
            sell_filled_avg_price=sell_price,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id=sell_order_id,
        )

        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:34:00Z",
                target_sell_price=150,
            )
        )
        accounting = self.conn.execute(
            "SELECT sold_qty, sold_value, remaining_qty, realized_pl, realized_pl_pct "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertIsNotNone(accounting)
        assert accounting is not None
        self.assertEqual(accounting[0], sell_qty)
        self.assertEqual(accounting[1], sell_qty * sell_price)
        self.assertAlmostEqual(accounting[2], 2.0 - sell_qty, delta=1e-15)
        self.assertAlmostEqual(
            accounting[3],
            sell_qty * (sell_price - 100.0),
            delta=1e-20,
        )
        self.assertAlmostEqual(accounting[4], 20.0, delta=1e-12)
        init_state_db(self.conn)

    def test_invalid_buy_closure_timestamp_falls_back_to_auditable_time(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-invalid-close-time",
            buy_alpaca_order_id="buy-invalid-close-time",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )

        self.assertTrue(
            close_alpaca_managed_buy_if_current_and_unfilled(
                self.conn,
                position_id,
                expected_buy_status="accepted",
                expected_buy_alpaca_order_id="buy-invalid-close-time",
                buy_status="canceled",
                buy_alpaca_order_id="buy-invalid-close-time",
                closed_at="not-a-broker-time",
            )
        )
        stored = self.conn.execute(
            "SELECT closed_at, datetime(closed_at) FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        candidates = load_recently_closed_alpaca_managed_positions(self.conn)

        self.assertIsNotNone(stored[0])
        self.assertIsNotNone(stored[1])
        self.assertEqual(candidates["id"].tolist(), [position_id])

    def test_init_repairs_malformed_legacy_closure_timestamp_for_audit(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-malformed-legacy-close",
        )
        malformed_values = (
            "not-a-broker-time",
            "1",
            "0",
            "1234",
            "2451545",
            "now",
            1,
            sqlite3.Binary(b"2451545"),
        )
        for malformed_value in malformed_values:
            with self.subTest(value=malformed_value):
                self.conn.execute(
                    """
                    UPDATE alpaca_managed_positions
                    SET closed_at = ?,
                        closed_correction_audited_at = '2999-01-01T00:00:00Z'
                    WHERE id = ?
                    """,
                    (malformed_value, position_id),
                )
                self.conn.commit()
                persisted_malformed = self.conn.execute(
                    "SELECT closed_at, TYPEOF(closed_at) FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()

                init_state_db(self.conn)

                stored = self.conn.execute(
                    """
                    SELECT closed_at, datetime(closed_at), closed_correction_audited_at
                    FROM alpaca_managed_positions
                    WHERE id = ?
                    """,
                    (position_id,),
                ).fetchone()
                candidates = load_recently_closed_alpaca_managed_positions(self.conn)

                self.assertNotEqual(stored[0], persisted_malformed[0])
                self.assertIsNotNone(stored[1])
                self.assertIsNone(stored[2])
                self.assertEqual(candidates["id"].tolist(), [position_id])

        canonical_values = (
            "2026-07-20T12:00:00Z",
            "2026-07-20T12:00:00.123456Z",
            "2026-07-20 12:00:00",
        )
        for canonical_value in canonical_values:
            with self.subTest(canonical_value=canonical_value):
                self.conn.execute(
                    "UPDATE alpaca_managed_positions SET closed_at = ? WHERE id = ?",
                    (canonical_value, position_id),
                )
                self.conn.commit()

                init_state_db(self.conn)

                self.assertEqual(
                    self.conn.execute(
                        "SELECT closed_at FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    (canonical_value,),
                )

    def test_init_repairs_canonical_legacy_closure_that_predates_lifecycle(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-impossible-legacy-close",
        )
        impossible_closure_cases = (
            ("buy signal", "2000-01-01 00:00:00", None, None),
            ("buy submission", "2026-01-02T14:00:00Z", None, None),
            ("buy fill", "2026-01-02T14:30:30Z", None, None),
            (
                "sell submission",
                "2026-01-02T14:31:30Z",
                "2026-01-02T14:32:00Z",
                None,
            ),
            (
                "sell fill",
                "2026-01-02T14:32:30Z",
                "2026-01-02T14:32:00Z",
                "2026-01-02T14:33:00Z",
            ),
        )
        for boundary, impossible_closed_at, sell_submitted_at, sell_filled_at in impossible_closure_cases:
            with self.subTest(boundary=boundary, impossible_closed_at=impossible_closed_at):
                self.conn.execute(
                    """
                    UPDATE alpaca_managed_positions
                    SET closed_at = ?,
                        sell_submitted_at = ?,
                        sell_filled_at = ?,
                        closed_correction_audited_at = '2999-01-01T00:00:00Z'
                    WHERE id = ?
                    """,
                    (
                        impossible_closed_at,
                        sell_submitted_at,
                        sell_filled_at,
                        position_id,
                    ),
                )
                self.conn.commit()

                init_state_db(self.conn)

                stored = self.conn.execute(
                    """
                    SELECT closed_at, closed_correction_audited_at
                    FROM alpaca_managed_positions
                    WHERE id = ?
                    """,
                    (position_id,),
                ).fetchone()
                candidates = load_recently_closed_alpaca_managed_positions(self.conn)

                self.assertNotEqual(stored[0], impossible_closed_at)
                self.assertIsNone(stored[1])
                self.assertEqual(candidates["id"].tolist(), [position_id])

        legitimate_closed_at = "2026-01-02T14:34:00Z"
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET closed_at = ?,
                sell_submitted_at = '2026-01-02T14:32:00Z',
                sell_filled_at = '2026-01-02T14:33:00Z'
            WHERE id = ?
            """,
            (legitimate_closed_at, position_id),
        )
        self.conn.commit()

        init_state_db(self.conn)

        self.assertEqual(
            self.conn.execute(
                "SELECT closed_at FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            (legitimate_closed_at,),
        )

    def test_closed_zero_fill_buy_rejects_fill_older_than_lifecycle_revision(self) -> None:
        order_id = "buy-closed-zero-fill-lifecycle-fence"
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-closed-zero-fill-lifecycle-fence",
            buy_alpaca_order_id=order_id,
            buy_submitted_at="2026-07-20T12:00:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.assertTrue(
            close_alpaca_managed_buy_if_current_and_unfilled(
                self.conn,
                position_id,
                expected_buy_status="accepted",
                expected_buy_alpaca_order_id=order_id,
                buy_status="canceled",
                buy_alpaca_order_id=order_id,
                closed_at="2026-07-20T12:02:00Z",
                buy_broker_updated_at="2026-07-20T12:02:00Z",
            )
        )

        def apply_fill(broker_updated_at: str) -> tuple[bool, bool, bool, float]:
            snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
            return apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="partially_filled",
                filled_qty=1,
                filled_avg_price=100,
                filled_at="2026-07-20T12:01:00Z",
                target_sell_price=150,
                sell_status=None,
                sell_fills=[],
                sell_filled_at=None,
                notes="closed buy received a positive fill correction",
                buy_fill_broker_updated_at=broker_updated_at,
                buy_fill_broker_oldest_updated_at=broker_updated_at,
                buy_fill_component_revisions={order_id: broker_updated_at},
            )

        conflicting_equal = apply_fill("2026-07-20T12:02:00Z")
        after_equal = self.conn.execute(
            "SELECT buy_status, filled_qty, buy_observation_broker_updated_at, "
            "buy_fill_broker_updated_at, closed_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        stale = apply_fill("2026-07-20T12:01:00Z")
        after_stale = self.conn.execute(
            "SELECT buy_status, filled_qty, buy_observation_broker_updated_at, "
            "buy_fill_broker_updated_at, closed_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        current = apply_fill("2026-07-20T12:03:00Z")
        after_current = self.conn.execute(
            "SELECT buy_status, filled_qty, buy_fill_broker_updated_at, remaining_qty, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(conflicting_equal, (False, False, False, 0.0))
        self.assertEqual(stale, (False, False, False, 0.0))
        self.assertEqual(after_equal, after_stale)
        self.assertEqual(
            after_stale,
            (
                "canceled",
                None,
                "2026-07-20T12:02:00.000000Z",
                None,
                "2026-07-20T12:02:00Z",
            ),
        )
        self.assertEqual(current, (True, True, False, 1.0))
        self.assertEqual(
            after_current,
            ("partially_filled", 1, "2026-07-20T12:03:00.000000Z", 1, None),
        )

    def test_closed_zero_fill_leaf_allows_equal_revision_ancestor_fill_aggregate(self) -> None:
        root_order_id = "buy-closed-ancestor-fill-root"
        leaf_order_id = "buy-closed-ancestor-fill-leaf"
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-closed-ancestor-fill",
            buy_alpaca_order_id=leaf_order_id,
            buy_submitted_at="2026-07-20T12:00:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.assertTrue(
            close_alpaca_managed_buy_if_current_and_unfilled(
                self.conn,
                position_id,
                expected_buy_status="accepted",
                expected_buy_alpaca_order_id=leaf_order_id,
                buy_status="canceled",
                buy_alpaca_order_id=leaf_order_id,
                closed_at="2026-07-20T12:02:00Z",
                buy_broker_updated_at="2026-07-20T12:02:00Z",
            )
        )
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]

        result = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="canceled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-07-20T12:01:00Z",
            target_sell_price=150,
            sell_status=None,
            sell_fills=[],
            sell_filled_at=None,
            notes="ancestor fill shares the terminal leaf aggregate revision",
            buy_fill_broker_updated_at="2026-07-20T12:02:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:01:00Z",
            buy_fill_component_revisions={
                root_order_id: "2026-07-20T12:01:00Z",
                leaf_order_id: "2026-07-20T12:02:00Z",
            },
        )
        row = self.conn.execute(
            "SELECT buy_status, filled_qty, buy_fill_broker_updated_at, remaining_qty, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(result, (True, True, False, 1.0))
        self.assertEqual(row, ("canceled", 1, "2026-07-20T12:02:00.000000Z", 1, None))

    def test_boundary_offset_buy_closure_timestamps_fall_back_without_overflow(self) -> None:
        boundary_timestamps = (
            "0001-01-01T00:00:00+14:00",
            "9999-12-31T23:59:59-14:00",
        )
        for index, boundary_timestamp in enumerate(boundary_timestamps):
            with self.subTest(boundary_timestamp=boundary_timestamp):
                order_id = f"buy-boundary-close-{index}"
                position_id = save_alpaca_managed_buy_order(
                    self.conn,
                    symbol=f"TQ{index}",
                    alpaca_asset_id=f"asset-boundary-{index}",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"rsi-buy-boundary-close-{index}",
                    buy_alpaca_order_id=order_id,
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                )

                self.assertTrue(
                    close_alpaca_managed_buy_if_current_and_unfilled(
                        self.conn,
                        position_id,
                        expected_buy_status="accepted",
                        expected_buy_alpaca_order_id=order_id,
                        buy_status="canceled",
                        buy_alpaca_order_id=order_id,
                        closed_at=boundary_timestamp,
                    )
                )
                stored = self.conn.execute(
                    "SELECT closed_at, datetime(closed_at) FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()
                self.assertIsNotNone(stored[0])
                self.assertIsNotNone(stored[1])

    def test_invalid_sell_closure_timestamp_falls_back_to_auditable_time(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-invalid-sell-close-time",
        )
        sell_client_order_id = "rsi-exit-TQQQ-invalid-sell-close-time"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-invalid-close-time",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="filled",
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=2,
            sell_filled_avg_price=150,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id="sell-invalid-close-time",
        )

        self.assertTrue(
            close_alpaca_managed_position_if_current_and_complete(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                expected_sell_alpaca_order_id="sell-invalid-close-time",
                closed_at="not-a-broker-time",
            )
        )
        stored = self.conn.execute(
            "SELECT closed_at, datetime(closed_at) FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertIsNotNone(stored[0])
        self.assertIsNotNone(stored[1])

    def test_closed_correction_cannot_seed_components_behind_legacy_buy_scalar(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-closed-stale-legacy-map",
        )
        sell_client_order_id = "rsi-exit-TQQQ-closed-stale-legacy-map"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-closed-stale-legacy-map",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="filled",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=2,
            sell_filled_avg_price=150,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id="sell-closed-stale-legacy-map",
        )
        self.close_position(position_id)
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:10:00.000000Z', "
            "buy_fill_component_revisions = NULL WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
        order_id = str(snapshot["buy_alpaca_order_id"])
        common = {
            "alpaca_asset_id": "asset-tqqq",
            "buy_order_qty": 2,
            "buy_order_limit_price": 105,
            "buy_status": "filled",
            "filled_qty": 2,
            "filled_avg_price": 100,
            "filled_at": "2026-01-02T14:31:00Z",
            "target_sell_price": 150,
            "sell_status": "filled",
            "sell_fills": [("sell-closed-stale-legacy-map", 2, 300)],
            "sell_filled_at": "2026-01-02T14:33:00Z",
            "notes": "stale legacy component replay",
            "buy_fill_broker_updated_at": "2026-07-20T12:02:00Z",
            "buy_fill_broker_oldest_updated_at": "2026-07-20T12:02:00Z",
            "buy_fill_component_revisions": {order_id: "2026-07-20T12:02:00Z"},
        }

        replayed = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            **common,
        )
        after_replay = load_alpaca_managed_positions(self.conn).iloc[0]
        corrected = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(after_replay["closed_at"]),
            expected_state_revision=int(after_replay["state_revision"]),
            **{
                **common,
                "filled_avg_price": 101,
                "target_sell_price": 151.5,
                "buy_fill_broker_updated_at": "2026-07-20T12:03:00Z",
                "buy_fill_broker_oldest_updated_at": "2026-07-20T12:03:00Z",
                "buy_fill_component_revisions": {order_id: "2026-07-20T12:03:00Z"},
            },
        )
        after_correction = load_alpaca_managed_positions(self.conn).iloc[0]

        self.assertTrue(replayed[0])
        self.assertFalse(corrected[0])
        self.assertEqual(after_replay["buy_fill_broker_updated_at"], "2026-07-20T12:10:00.000000Z")
        self.assertTrue(pd.isna(after_replay["buy_fill_component_revisions"]))
        self.assertEqual(after_correction["filled_avg_price"], 100)
        self.assertEqual(after_correction["target_sell_price"], 150)
        self.assertEqual(after_correction["state_revision"], after_replay["state_revision"])

    def test_closed_correction_rejects_components_contaminated_behind_scalar(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-closed-contaminated-map",
        )
        order_id = "buy-rsi-buy-TQQQ-closed-contaminated-map"
        sell_client_order_id = "rsi-exit-TQQQ-closed-contaminated-map"
        sell_order_id = "sell-closed-contaminated-map"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=sell_order_id,
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="filled",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=2,
            sell_filled_avg_price=150,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id=sell_order_id,
        )
        self.close_position(position_id)
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:10:00.000000Z', "
            "buy_fill_component_revisions = ? WHERE id = ?",
            (f'{{"{order_id}":"2026-07-20T12:02:00.000000Z"}}', position_id),
        )
        self.conn.commit()
        before = load_alpaca_managed_positions(self.conn).iloc[0]

        def correction(*, price: float, broker_updated_at: str) -> tuple[bool, bool, bool, float]:
            snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
            return apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=price,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=price * 1.5,
                sell_status="filled",
                sell_fills=[(sell_order_id, 2, 300)],
                sell_filled_at="2026-01-02T14:33:00Z",
                notes="contaminated component-map correction",
                buy_fill_broker_updated_at=broker_updated_at,
                buy_fill_broker_oldest_updated_at=broker_updated_at,
                buy_fill_component_revisions={order_id: broker_updated_at},
            )

        stale = correction(price=99, broker_updated_at="2026-07-20T12:03:00Z")
        after_stale = load_alpaca_managed_positions(self.conn).iloc[0]
        repaired = correction(price=100, broker_updated_at="2026-07-20T12:10:00Z")
        corrected = correction(price=101, broker_updated_at="2026-07-20T12:11:00Z")
        after_correction = load_alpaca_managed_positions(self.conn).iloc[0]

        self.assertFalse(stale[0])
        self.assertEqual(after_stale["state_revision"], before["state_revision"])
        self.assertEqual(after_stale["filled_avg_price"], 100)
        self.assertTrue(repaired[0])
        self.assertTrue(corrected[0])
        self.assertEqual(after_correction["filled_avg_price"], 101)
        self.assertEqual(after_correction["target_sell_price"], 151.5)
        self.assertEqual(after_correction["buy_fill_broker_updated_at"], "2026-07-20T12:11:00.000000Z")
        self.assertIn('"2026-07-20T12:11:00.000000Z"', after_correction["buy_fill_component_revisions"])

    def test_closed_correction_keeps_scale_aware_fractional_residual_closed(self) -> None:
        sold_qty = float(np.nextafter(100.0, 0.0))
        remaining_qty = 100.0 - sold_qty
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-scale-aware-correction",
            buy_alpaca_order_id="buy-scale-aware-correction",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="filled",
            buy_order_qty=100,
            buy_order_limit_price=105,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=100,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]

        result = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=100,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=100,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="filled",
            sell_fills=[("sell-1", sold_qty, sold_qty * 150.0)],
            sell_filled_at="2026-01-02T14:33:00Z",
            notes="scale-aware closed correction",
        )
        row = self.conn.execute(
            "SELECT remaining_qty, closed_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(result[:3], (True, False, False))
        self.assertEqual(result[3], remaining_qty)
        self.assertEqual(row[0], remaining_qty)
        self.assertIsNotNone(row[1])

    def test_closed_correction_reopens_quantity_dust_with_material_notional(self) -> None:
        quantity = 1e-13
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="CORRECTDUST",
            alpaca_asset_id="asset-correct-dust",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-correct-value-dust",
            buy_alpaca_order_id="buy-correct-value-dust",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="filled",
            buy_order_qty=1,
            buy_order_limit_price=1e15,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2.0 * quantity,
            filled_avg_price=1e15,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=1.5e15,
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]

        result = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-correct-dust",
            buy_order_qty=1,
            buy_order_limit_price=1e15,
            buy_status="filled",
            filled_qty=2.0 * quantity,
            filled_avg_price=1e15,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=1.5e15,
            sell_status="partially_filled",
            sell_fills=[("sell-correct-value-dust", quantity, 150.0)],
            sell_filled_at="2026-01-02T14:33:00Z",
            notes="material marked value must reopen",
        )
        row = self.conn.execute(
            "SELECT remaining_qty, closed_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(result, (True, True, False, quantity))
        self.assertEqual(row, (quantity, None))

    def test_closed_correction_rejects_nonfinite_aggregate_accounting_atomically(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-overflowing-closed-correction",
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
        before = self.conn.execute(
            "SELECT state_revision, sell_status, sold_qty, sold_value, realized_pl, realized_pl_pct "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "cumulative Alpaca sell accounting must remain finite"):
            apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
                sell_status="filled",
                sell_fills=[("sell-1", 1, 1e308), ("sell-2", 1, 1e308)],
                sell_filled_at="2026-01-02T14:33:00Z",
                notes="overflowing closed correction",
            )
        after = self.conn.execute(
            "SELECT state_revision, sell_status, sold_qty, sold_value, realized_pl, realized_pl_pct "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        ledger = self.conn.execute(
            "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
            (position_id,),
        ).fetchone()[0]

        self.assertEqual(after, before)
        self.assertEqual(ledger, 0)

    def test_init_state_db_adds_sell_fill_metadata_to_legacy_ledger(self) -> None:
        with closing(sqlite3.connect(":memory:")) as legacy_conn, legacy_conn:
            init_state_db(legacy_conn)
            legacy_conn.execute("DROP TABLE alpaca_managed_sell_fills")
            legacy_conn.execute(
                """
                CREATE TABLE alpaca_managed_sell_fills (
                    managed_position_id INTEGER NOT NULL,
                    alpaca_order_id TEXT NOT NULL,
                    filled_qty REAL NOT NULL,
                    filled_value REAL NOT NULL,
                    PRIMARY KEY (managed_position_id, alpaca_order_id)
                )
                """
            )
            legacy_conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (id, symbol, signal_symbol, buy_rsi,
                 profit_target_multiple, buy_signal_date,
                 buy_client_order_id, buy_status)
                VALUES (1, 'TQQQ', 'QQQ', 30, 1.5,
                        '2026-01-02', 'buy-legacy-parent', 'accepted')
                """
            )
            legacy_conn.execute("INSERT INTO alpaca_managed_sell_fills VALUES (1, 'sell-legacy', 1, 150)")

            init_state_db(legacy_conn)

            columns = {str(row[1]) for row in legacy_conn.execute("PRAGMA table_info(alpaca_managed_sell_fills)")}
            ledger = legacy_conn.execute(
                "SELECT filled_qty, filled_value, broker_updated_at, "
                "submitted_qty, submitted_limit_price "
                "FROM alpaca_managed_sell_fills WHERE alpaca_order_id = 'sell-legacy'"
            ).fetchone()

        self.assertIn("broker_updated_at", columns)
        self.assertIn("submitted_qty", columns)
        self.assertIn("submitted_limit_price", columns)
        self.assertEqual(ledger, (1, 150, None, None, None))

    def test_init_migrates_provable_parent_only_sell_fill_and_prior_identity_guards(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-parent-only-legacy-fill",
        )
        sell_client_order_id = "rsi-exit-TQQQ-parent-only-legacy-fill"
        sell_order_id = "sell-parent-only-legacy-fill"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=sell_order_id,
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=2,
            sell_filled_avg_price=150,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id=sell_order_id,
        )
        self.conn.execute(
            "DELETE FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
            (position_id,),
        )
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET sell_order_qty = NULL,
                sell_order_limit_price = NULL,
                sold_qty = 0,
                sold_value = 0,
                remaining_qty = NULL,
                closed_at = sell_filled_at
            WHERE id = ?
            """,
            (position_id,),
        )
        for operation in ("insert", "update"):
            trigger_name = f"leveraged_trader_alpaca_managed_positions_identity_{operation}_guard"
            self.conn.execute(f"DROP TRIGGER {trigger_name}")
            prior_contracts = tuple(
                contract
                for contract in storage_module._STORAGE_IDENTITY_TYPE_CONTRACTS["alpaca_managed_positions"]
                if contract[0] != "id"
            )
            self.conn.execute(
                storage_module._storage_identity_type_guard_sql(
                    "alpaca_managed_positions",
                    prior_contracts,
                    operation,
                )
            )
        self.conn.commit()

        init_state_db(self.conn)

        self.assertEqual(
            self.conn.execute(
                """
                SELECT alpaca_order_id, filled_qty, filled_value,
                       broker_updated_at, submitted_qty, submitted_limit_price
                FROM alpaca_managed_sell_fills
                WHERE managed_position_id = ?
                """,
                (position_id,),
            ).fetchone(),
            (sell_order_id, 2, 300, None, None, None),
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT sold_qty, sold_value, remaining_qty FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            (2, 300, 0),
        )
        canonical_contracts = storage_module._canonical_storage_schema_object_contracts()
        for operation in ("insert", "update"):
            trigger_name = f"leveraged_trader_alpaca_managed_positions_identity_{operation}_guard"
            installed_sql = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                (trigger_name,),
            ).fetchone()[0]
            self.assertEqual(
                storage_module._normalized_schema_sql(installed_sql),
                canonical_contracts[trigger_name][2],
            )

    def test_init_rejects_incomplete_parent_only_sell_fill_as_legacy_accounting(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-incomplete-parent-only-fill",
        )
        sell_client_order_id = "rsi-exit-TQQQ-incomplete-parent-only-fill"
        sell_order_id = "sell-incomplete-parent-only-fill"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=sell_order_id,
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=1,
            sell_filled_avg_price=150,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id=sell_order_id,
        )
        self.conn.execute(
            "DELETE FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
            (position_id,),
        )
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET sell_order_qty = NULL,
                sell_order_limit_price = NULL,
                sold_qty = 0,
                sold_value = 0,
                remaining_qty = NULL,
                closed_at = sell_filled_at
            WHERE id = ?
            """,
            (position_id,),
        )
        self.conn.commit()
        parent_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "sell accounting that conflicts with its"):
            init_state_db(self.conn)

        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            parent_before,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchall(),
            [],
        )

    def test_init_rejects_parent_sell_accounting_that_conflicts_with_ledger(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-imported-sell-accounting-conflict",
        )
        sell_client_order_id = "rsi-exit-TQQQ-imported-sell-accounting-conflict"
        sell_order_id = "sell-imported-accounting-conflict"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=sell_order_id,
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="partially_filled",
            sell_filled_qty=1,
            sell_filled_avg_price=150,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_broker_updated_at="2026-07-20T12:01:00Z",
            sell_alpaca_order_id=sell_order_id,
        )

        cases = (
            ("sold_qty", 2, 1),
            ("sold_value", 300, 150),
            ("remaining_qty", 0, 1),
            ("sell_filled_qty", 2, 1),
            ("sell_filled_avg_price", 140, 150),
            ("realized_pl", 100, 50),
            ("realized_pl_pct", 100, 50),
            ("realized_pl", None, 50),
            ("realized_pl_pct", None, 50),
        )
        for column, conflicting_value, restored_value in cases:
            with self.subTest(column=column):
                self.conn.execute(
                    f"UPDATE alpaca_managed_positions SET {column} = ? WHERE id = ?",
                    (conflicting_value, position_id),
                )
                self.conn.commit()
                parent_before = self.conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()
                ledger_before = self.conn.execute(
                    "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                    (position_id,),
                ).fetchall()

                with self.assertRaisesRegex(ValueError, "sell accounting that conflicts with its"):
                    init_state_db(self.conn)

                self.assertFalse(self.conn.in_transaction)
                self.assertEqual(
                    self.conn.execute(
                        "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    parent_before,
                )
                self.assertEqual(
                    self.conn.execute(
                        "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                        (position_id,),
                    ).fetchall(),
                    ledger_before,
                )
                self.conn.execute(
                    f"UPDATE alpaca_managed_positions SET {column} = ? WHERE id = ?",
                    (restored_value, position_id),
                )
                self.conn.commit()

        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET sold_qty = ?, sold_value = ?, remaining_qty = ?,
                sell_filled_qty = ?, sell_filled_avg_price = ?,
                realized_pl = ?, realized_pl_pct = ?
            WHERE id = ?
            """,
            (
                np.nextafter(1.0, np.inf),
                np.nextafter(150.0, np.inf),
                np.nextafter(1.0, -np.inf),
                np.nextafter(1.0, np.inf),
                np.nextafter(150.0, np.inf),
                np.nextafter(50.0, np.inf),
                np.nextafter(50.0, np.inf),
                position_id,
            ),
        )
        self.conn.commit()

        init_state_db(self.conn)

    def test_sell_fill_parent_relation_is_enforced_when_foreign_keys_are_off(
        self,
    ) -> None:
        self.conn.execute("DROP TRIGGER alpaca_managed_positions_restrict_sell_fill_parent_id_update")
        init_state_db(self.conn)
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-parent-integrity",
        )
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
            VALUES (?, 'sell-parent-integrity', 1, 120)
            """,
            (position_id,),
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "no parent position"):
            self.conn.execute(
                """
                INSERT INTO alpaca_managed_sell_fills
                (managed_position_id, alpaca_order_id, filled_qty, filled_value)
                VALUES (999, 'sell-orphan', 1, 120)
                """
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "no parent position"):
            self.conn.execute(
                """
                UPDATE alpaca_managed_sell_fills
                SET managed_position_id = 999
                WHERE managed_position_id = ?
                """,
                (position_id,),
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "still owns sell fills"):
            self.conn.execute(
                "DELETE FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "still owns sell fills"):
            self.conn.execute(
                "UPDATE alpaca_managed_positions SET id = id + 1000 WHERE id = ?",
                (position_id,),
            )

        self.assertEqual(
            self.conn.execute(
                """
                SELECT managed_position_id
                FROM alpaca_managed_sell_fills
                WHERE alpaca_order_id = 'sell-parent-integrity'
                """
            ).fetchone(),
            (position_id,),
        )

    def test_sell_order_id_has_one_global_owner_and_legacy_duplicates_fail_preflight(
        self,
    ) -> None:
        first_position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-global-sell-owner",
            alpaca_asset_id="asset-global-owner-1",
        )
        second_position_id = self.save_position(
            symbol="UPRO",
            client_order_id="rsi-buy-UPRO-global-sell-owner",
            alpaca_asset_id="asset-global-owner-2",
        )
        order_id = "sell-global-owner"
        record_alpaca_managed_sell_generation(
            self.conn,
            first_position_id,
            order_id,
            submitted_qty=2,
            submitted_limit_price=150,
        )
        second_parent_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (second_position_id,),
        ).fetchone()
        ledger_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_sell_fills ORDER BY managed_position_id",
        ).fetchall()

        conflicting_writes = (
            lambda: record_alpaca_managed_sell_generation(
                self.conn,
                second_position_id,
                order_id,
                submitted_qty=2,
                submitted_limit_price=150,
            ),
            lambda: record_alpaca_managed_sell_order(
                self.conn,
                second_position_id,
                sell_client_order_id="rsi-exit-UPRO-global-sell-owner",
                sell_alpaca_order_id=order_id,
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_order_qty=2,
                sell_order_limit_price=150,
            ),
            lambda: attach_alpaca_managed_sell_order_if_current(
                self.conn,
                second_position_id,
                expected_sell_client_order_id=None,
                expected_sell_alpaca_order_id=None,
                expected_renewal_count=0,
                sell_renewal_count=0,
                sell_client_order_id="rsi-exit-UPRO-global-sell-owner",
                sell_alpaca_order_id=order_id,
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_expires_at=None,
                sell_order_qty=2,
                sell_order_limit_price=150,
            ),
            lambda: storage_module.update_alpaca_managed_sell_status(
                self.conn,
                second_position_id,
                sell_status="accepted",
                sell_alpaca_order_id=order_id,
                sell_submitted_at="2026-01-02T14:32:00Z",
            ),
        )
        for conflicting_write in conflicting_writes:
            with (
                self.subTest(conflicting_write=conflicting_write),
                self.assertRaisesRegex(ValueError, "multiple managed positions"),
            ):
                conflicting_write()
            self.assertFalse(self.conn.in_transaction)
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (second_position_id,),
                ).fetchone(),
                second_parent_before,
            )
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_sell_fills ORDER BY managed_position_id",
                ).fetchall(),
                ledger_before,
            )

        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                """
                INSERT INTO alpaca_managed_sell_fills
                (managed_position_id, alpaca_order_id, filled_qty, filled_value,
                 submitted_qty, submitted_limit_price)
                VALUES (?, ?, 0, 0, 2, 150)
                """,
                (second_position_id, order_id),
            )
        self.conn.rollback()

        global_index = "leveraged_trader_alpaca_managed_sell_fills_alpaca_order_identity_unique"
        self.conn.execute(f"DROP INDEX {global_index}")
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value,
             submitted_qty, submitted_limit_price)
            VALUES (?, ?, 0, 0, 2, 150)
            """,
            (second_position_id, order_id),
        )
        self.conn.commit()
        duplicate_rows = self.conn.execute(
            "SELECT * FROM alpaca_managed_sell_fills ORDER BY managed_position_id",
        ).fetchall()

        with self.assertRaisesRegex(ValueError, "duplicate Alpaca sell-order identities"):
            init_state_db(self.conn)

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_sell_fills ORDER BY managed_position_id",
            ).fetchall(),
            duplicate_rows,
        )
        self.assertIsNone(
            self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
                (global_index,),
            ).fetchone()
        )

    def test_generic_sell_status_update_owns_one_immutable_generation(self) -> None:
        first_position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-generic-sell-status-owner",
            alpaca_asset_id="asset-generic-sell-status-owner-1",
        )
        second_position_id = self.save_position(
            symbol="UPRO",
            client_order_id="rsi-buy-UPRO-generic-sell-status-owner",
            alpaca_asset_id="asset-generic-sell-status-owner-2",
        )
        self.assertEqual(
            storage_module.claim_alpaca_managed_initial_sell_intent(
                self.conn,
                first_position_id,
                sell_order_namespace=None,
                sell_client_order_id="rsi-exit-TQQQ-generic-sell-status-owner",
                expected_remaining_qty=2,
                expected_target_sell_price=150,
                notes="claim generic sell status owner",
            ),
            (2.0, 150.0),
        )

        storage_module.update_alpaca_managed_sell_status(
            self.conn,
            first_position_id,
            sell_status="accepted",
            sell_alpaca_order_id="sell-generic-status-owner",
            sell_submitted_at="2026-01-02T14:32:00Z",
        )
        first_after = self.conn.execute(
            """
            SELECT sell_alpaca_order_id, sell_order_qty, sell_order_limit_price
            FROM alpaca_managed_positions
            WHERE id = ?
            """,
            (first_position_id,),
        ).fetchone()
        ledger_after = self.conn.execute(
            """
            SELECT managed_position_id, alpaca_order_id, submitted_qty,
                   submitted_limit_price
            FROM alpaca_managed_sell_fills
            """
        ).fetchall()

        self.assertEqual(first_after, ("sell-generic-status-owner", 2, 150))
        self.assertEqual(
            ledger_after,
            [(first_position_id, "sell-generic-status-owner", 2, 150)],
        )

        second_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (second_position_id,),
        ).fetchone()
        with self.assertRaisesRegex(ValueError, "multiple managed positions"):
            storage_module.update_alpaca_managed_sell_status(
                self.conn,
                second_position_id,
                sell_status="accepted",
                sell_alpaca_order_id="sell-generic-status-owner",
            )
        with self.assertRaisesRegex(ValueError, "cannot replace its broker order identity"):
            storage_module.update_alpaca_managed_sell_status(
                self.conn,
                first_position_id,
                sell_status="accepted",
                sell_alpaca_order_id="sell-generic-status-replacement",
            )

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (second_position_id,),
            ).fetchone(),
            second_before,
        )
        self.assertEqual(
            self.conn.execute(
                """
                SELECT managed_position_id, alpaca_order_id, submitted_qty,
                       submitted_limit_price
                FROM alpaca_managed_sell_fills
                """
            ).fetchall(),
            ledger_after,
        )

    def test_init_rejects_legacy_parent_sell_order_ownership_conflicts_atomically(
        self,
    ) -> None:
        first_position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-legacy-parent-sell-owner",
            alpaca_asset_id="asset-legacy-parent-sell-owner-1",
        )
        second_position_id = self.save_position(
            symbol="UPRO",
            client_order_id="rsi-buy-UPRO-legacy-parent-sell-owner",
            alpaca_asset_id="asset-legacy-parent-sell-owner-2",
        )
        parent_index = "leveraged_trader_alpaca_managed_positions_sell_alpaca_order_identity_unique"
        order_id = "sell-legacy-parent-owner"
        self.conn.execute(f"DROP INDEX {parent_index}")
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET sell_alpaca_order_id = ? WHERE id IN (?, ?)",
            (order_id, first_position_id, second_position_id),
        )
        self.conn.commit()
        duplicate_rows = self.conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall()
        duplicate_objects = self.conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

        with self.assertRaisesRegex(ValueError, "duplicate Alpaca sell-order identities"):
            init_state_db(self.conn)

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall(),
            duplicate_rows,
        )
        self.assertEqual(
            self.conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
            duplicate_objects,
        )

        self.conn.execute(
            "UPDATE alpaca_managed_positions SET sell_alpaca_order_id = NULL WHERE id = ?",
            (second_position_id,),
        )
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value,
             submitted_qty, submitted_limit_price)
            VALUES (?, ?, 0, 0, 2, 150)
            """,
            (second_position_id, order_id),
        )
        self.conn.commit()
        conflicting_positions = self.conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall()
        conflicting_ledger = self.conn.execute(
            "SELECT * FROM alpaca_managed_sell_fills ORDER BY managed_position_id"
        ).fetchall()
        conflicting_objects = self.conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()

        with self.assertRaisesRegex(ValueError, "different managed positions"):
            init_state_db(self.conn)

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall(),
            conflicting_positions,
        )
        self.assertEqual(
            self.conn.execute("SELECT * FROM alpaca_managed_sell_fills ORDER BY managed_position_id").fetchall(),
            conflicting_ledger,
        )
        self.assertEqual(
            self.conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall(),
            conflicting_objects,
        )

    def test_first_sell_attachment_cannot_redefine_claimed_intent(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-first-sell-immutable",
        )
        sell_client_order_id = "rsi-exit-TQQQ-first-sell-immutable"
        self.assertEqual(
            storage_module.claim_alpaca_managed_initial_sell_intent(
                self.conn,
                position_id,
                sell_order_namespace=None,
                sell_client_order_id=sell_client_order_id,
                expected_remaining_qty=2,
                expected_target_sell_price=150,
                notes="claim immutable first sell",
            ),
            (2.0, 150.0),
        )
        snapshot = storage_module.alpaca_managed_sell_renewal_snapshot_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            expected_sell_alpaca_order_id=None,
        )
        self.assertIsNotNone(snapshot)
        parent_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        conflicting_attachments = (
            lambda: record_alpaca_managed_sell_order(
                self.conn,
                position_id,
                sell_client_order_id=sell_client_order_id,
                sell_alpaca_order_id="sell-first-immutable",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_order_qty=1,
                sell_order_limit_price=160,
            ),
            lambda: attach_alpaca_managed_sell_order_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                expected_sell_alpaca_order_id=None,
                expected_renewal_count=0,
                sell_renewal_count=0,
                sell_client_order_id=sell_client_order_id,
                sell_alpaca_order_id="sell-first-immutable",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_expires_at=None,
                sell_order_qty=1,
                sell_order_limit_price=160,
                expected_state_snapshot=snapshot,
            ),
            lambda: update_alpaca_managed_sell_status_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                sell_status="accepted",
                sell_alpaca_order_id="sell-first-immutable",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_order_qty=1,
                sell_order_limit_price=160,
            ),
        )
        for conflicting_attachment in conflicting_attachments:
            with (
                self.subTest(conflicting_attachment=conflicting_attachment),
                self.assertRaisesRegex(ValueError, "immutable generation economics"),
            ):
                conflicting_attachment()
            self.assertFalse(self.conn.in_transaction)
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone(),
                parent_before,
            )
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                    (position_id,),
                ).fetchall(),
                [],
            )

        with self.assertRaisesRegex(ValueError, "explicit diagnostic quarantine"):
            update_alpaca_managed_sell_status_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                sell_status="accepted",
                sell_alpaca_order_id="sell-first-immutable",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_order_qty=1,
                sell_order_limit_price=160,
                allow_diagnostic_sell_intent_mismatch=True,
            )
        with self.assertRaisesRegex(ValueError, "attached as an explicit diagnostic quarantine"):
            attach_alpaca_managed_sell_order_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                expected_sell_alpaca_order_id=None,
                expected_renewal_count=0,
                sell_renewal_count=0,
                sell_client_order_id=sell_client_order_id,
                sell_alpaca_order_id="sell-first-immutable",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_expires_at=None,
                sell_order_qty=1,
                sell_order_limit_price=160,
                expected_state_snapshot=snapshot,
                allow_diagnostic_sell_intent_mismatch=True,
            )
        with self.assertRaisesRegex(ValueError, "requires a broker order ID"):
            attach_alpaca_managed_sell_order_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                expected_sell_alpaca_order_id=None,
                expected_renewal_count=0,
                sell_renewal_count=0,
                sell_client_order_id=sell_client_order_id,
                sell_alpaca_order_id=None,
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="pending_cancel",
                sell_expires_at=None,
                sell_order_qty=1,
                sell_order_limit_price=120,
                expected_state_snapshot=snapshot,
                allow_diagnostic_sell_intent_mismatch=True,
            )
        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            parent_before,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchall(),
            [],
        )

        self.assertTrue(
            attach_alpaca_managed_sell_order_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                expected_sell_alpaca_order_id=None,
                expected_renewal_count=0,
                sell_renewal_count=0,
                sell_client_order_id=sell_client_order_id,
                sell_alpaca_order_id="sell-first-immutable",
                sell_submitted_at="2026-01-02T14:32:00Z",
                sell_status="accepted",
                sell_expires_at=None,
                sell_order_qty=None,
                sell_order_limit_price=None,
                expected_state_snapshot=snapshot,
            )
        )
        self.assertEqual(
            self.conn.execute(
                """
                SELECT sell_alpaca_order_id, sell_order_qty, sell_order_limit_price
                FROM alpaca_managed_positions
                WHERE id = ?
                """,
                (position_id,),
            ).fetchone(),
            ("sell-first-immutable", 2, 150),
        )
        self.assertEqual(
            self.conn.execute(
                """
                SELECT alpaca_order_id, submitted_qty, submitted_limit_price
                FROM alpaca_managed_sell_fills
                WHERE managed_position_id = ?
                """,
                (position_id,),
            ).fetchone(),
            ("sell-first-immutable", 2, 150),
        )

    def test_unattached_sell_order_replay_cannot_rewrite_claimed_generation(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-unattached-sell-generation",
        )
        sell_client_order_id = "rsi-exit-TQQQ-unattached-sell-generation"
        self.assertEqual(
            storage_module.claim_alpaca_managed_initial_sell_intent(
                self.conn,
                position_id,
                sell_order_namespace=None,
                sell_client_order_id=sell_client_order_id,
                expected_remaining_qty=2,
                expected_target_sell_price=150,
                notes="claim immutable unattached sell",
            ),
            (2.0, 150.0),
        )
        parent_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        conflicting_replays = (
            (
                "immutable generation economics",
                lambda: record_alpaca_managed_sell_order(
                    self.conn,
                    position_id,
                    sell_client_order_id=sell_client_order_id,
                    sell_alpaca_order_id=None,
                    sell_submitted_at=None,
                    sell_status="submission_pending",
                    sell_order_qty=1,
                    sell_order_limit_price=120,
                ),
            ),
            (
                "unattached generation's client order identity",
                lambda: record_alpaca_managed_sell_order(
                    self.conn,
                    position_id,
                    sell_client_order_id=f"{sell_client_order_id}-replacement",
                    sell_alpaca_order_id=None,
                    sell_submitted_at=None,
                    sell_status="submission_pending",
                    sell_order_qty=2,
                    sell_order_limit_price=150,
                ),
            ),
        )
        for message, replay in conflicting_replays:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                replay()
            self.assertFalse(self.conn.in_transaction)
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone(),
                parent_before,
            )
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                    (position_id,),
                ).fetchall(),
                [],
            )

        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=None,
            sell_submitted_at=None,
            sell_status="submission_pending",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        self.assertEqual(
            self.conn.execute(
                """
                SELECT sell_client_order_id, sell_alpaca_order_id,
                       sell_order_qty, sell_order_limit_price
                FROM alpaca_managed_positions
                WHERE id = ?
                """,
                (position_id,),
            ).fetchone(),
            (sell_client_order_id, None, 2, 150),
        )

    def test_managed_sell_generation_conflicting_replays_are_rejected_atomically(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-immutable-sell-generation",
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="rsi-exit-TQQQ-immutable-sell-generation",
            sell_alpaca_order_id="sell-immutable",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )

        conflicting_replays = (
            lambda: record_alpaca_managed_sell_generation(
                self.conn,
                position_id,
                "sell-immutable",
                submitted_qty=3,
                submitted_limit_price=99,
            ),
            lambda: record_alpaca_managed_sell_order(
                self.conn,
                position_id,
                sell_client_order_id="rsi-exit-TQQQ-immutable-sell-generation",
                sell_alpaca_order_id="sell-immutable",
                sell_submitted_at="2026-01-02T14:32:01Z",
                sell_status="accepted",
                sell_order_qty=1,
                sell_order_limit_price=160,
            ),
            lambda: attach_alpaca_managed_sell_order_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id="rsi-exit-TQQQ-immutable-sell-generation",
                expected_sell_alpaca_order_id="sell-immutable",
                expected_renewal_count=0,
                sell_renewal_count=0,
                sell_client_order_id="rsi-exit-TQQQ-immutable-sell-generation",
                sell_alpaca_order_id="sell-immutable",
                sell_submitted_at="2026-01-02T14:32:01Z",
                sell_status="accepted",
                sell_expires_at=None,
                sell_order_qty=1,
                sell_order_limit_price=160,
            ),
        )
        parent_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        ledger_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
            (position_id,),
        ).fetchall()

        for replay in conflicting_replays:
            with self.subTest(replay=replay), self.assertRaisesRegex(ValueError, "immutable generation economics"):
                replay()
            self.assertFalse(self.conn.in_transaction)
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone(),
                parent_before,
            )
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                    (position_id,),
                ).fetchall(),
                ledger_before,
            )

    def test_new_sell_generation_replaces_only_the_parent_generation_economics(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-distinct-sell-generations",
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="rsi-exit-TQQQ-distinct-sell-generations",
            sell_alpaca_order_id="sell-generation-zero",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="filled",
            sell_order_qty=1,
            sell_order_limit_price=150,
        )

        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="rsi-exit-TQQQ-distinct-sell-generations-r1",
            sell_alpaca_order_id="sell-generation-one",
            sell_submitted_at="2026-01-02T14:34:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=160,
            increment_renewal_count=True,
        )

        parent = self.conn.execute(
            "SELECT sell_alpaca_order_id, sell_order_qty, sell_order_limit_price "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        generations = self.conn.execute(
            "SELECT alpaca_order_id, submitted_qty, submitted_limit_price "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? ORDER BY alpaca_order_id",
            (position_id,),
        ).fetchall()

        self.assertEqual(parent, ("sell-generation-one", 2, 160))
        self.assertEqual(
            generations,
            [
                ("sell-generation-one", 2, 160),
                ("sell-generation-zero", 1, 150),
            ],
        )

    def test_init_rejects_current_parent_and_sell_ledger_intent_conflict(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-imported-sell-intent-conflict",
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="rsi-exit-TQQQ-imported-sell-intent-conflict",
            sell_alpaca_order_id="sell-imported-intent-conflict",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET sell_order_qty = 1, sell_order_limit_price = 160
            WHERE id = ?
            """,
            (position_id,),
        )
        self.conn.commit()
        parent_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        ledger_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "sell intent conflicts with its immutable"):
            init_state_db(self.conn)

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            parent_before,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchone(),
            ledger_before,
        )

    def test_sell_generation_migration_backfills_only_attached_current_order(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-sell-generation-backfill",
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="rsi-exit-TQQQ-sell-generation-backfill-r1",
            sell_alpaca_order_id="sell-current",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        record_alpaca_managed_sell_generation(self.conn, position_id, "sell-historical")
        self.conn.execute(
            "UPDATE alpaca_managed_sell_fills SET submitted_qty = NULL, "
            "submitted_limit_price = NULL WHERE managed_position_id = ?",
            (position_id,),
        )
        self.conn.commit()

        init_state_db(self.conn)
        stored = self.conn.execute(
            "SELECT alpaca_order_id, submitted_qty, submitted_limit_price "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? "
            "ORDER BY alpaca_order_id",
            (position_id,),
        ).fetchall()

        self.assertEqual(
            stored,
            [("sell-current", 2, 150), ("sell-historical", None, None)],
        )

    def test_managed_buy_fill_timestamp_rejects_delayed_equal_quantity_price(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-timestamped-price",
        )
        buy_order_id = "buy-rsi-buy-TQQQ-timestamped-price"

        def apply_price(
            average_price: float,
            broker_updated_at: str,
            expected_average_price: float,
        ) -> bool:
            return mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=average_price,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=average_price * 1.5,
                buy_fill_broker_updated_at=broker_updated_at,
                buy_alpaca_order_id=buy_order_id,
                expected_buy_status="filled",
                expected_buy_alpaca_order_id=buy_order_id,
                expected_filled_qty=2,
                expected_filled_avg_price=expected_average_price,
                expected_target_sell_price=expected_average_price * 1.5,
                expected_sell_client_order_id=None,
            )

        established = apply_price(100, "2026-07-20T12:00:00Z", 100)
        corrected = apply_price(101, "2026-07-20T12:02:00Z", 100)
        delayed = apply_price(99, "2026-07-20T12:01:00Z", 101)
        row = self.conn.execute(
            """
            SELECT filled_qty, filled_avg_price, target_sell_price,
                   buy_fill_broker_updated_at
            FROM alpaca_managed_positions WHERE id = ?
            """,
            (position_id,),
        ).fetchone()

        self.assertTrue(established)
        self.assertTrue(corrected)
        self.assertFalse(delayed)
        self.assertEqual(row, (2, 101, 151.5, "2026-07-20T12:02:00.000000Z"))

    def test_managed_buy_fill_timestamp_rejects_delayed_quantity_increase(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-delayed-quantity",
            buy_alpaca_order_id="buy-delayed-quantity",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        first = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=101,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=151.5,
            buy_fill_broker_updated_at="2026-07-20T12:02:00Z",
            buy_alpaca_order_id="buy-delayed-quantity",
        )
        delayed = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=150,
            buy_fill_broker_updated_at="2026-07-20T12:01:00Z",
            buy_alpaca_order_id="buy-delayed-quantity",
            expected_buy_status="partially_filled",
            expected_buy_alpaca_order_id="buy-delayed-quantity",
            expected_filled_qty=1,
            expected_filled_avg_price=101,
            expected_target_sell_price=151.5,
            expected_sell_client_order_id=None,
        )
        row = self.conn.execute(
            """
            SELECT buy_status, filled_qty, filled_avg_price, target_sell_price,
                   buy_fill_broker_updated_at
            FROM alpaca_managed_positions WHERE id = ?
            """,
            (position_id,),
        ).fetchone()

        self.assertTrue(first)
        self.assertFalse(delayed)
        self.assertEqual(
            row,
            (
                "partially_filled",
                1,
                101,
                151.5,
                "2026-07-20T12:02:00.000000Z",
            ),
        )

    def test_managed_buy_fill_rejects_mixed_replacement_chain_revisions(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-mixed-chain-revisions",
        )
        buy_order_id = "buy-rsi-buy-TQQQ-mixed-chain-revisions"

        def apply_price(
            average_price: float,
            *,
            oldest: str,
            latest: str,
            expected_average_price: float,
        ) -> bool:
            return mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=average_price,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=average_price * 1.5,
                buy_fill_broker_updated_at=latest,
                buy_fill_broker_oldest_updated_at=oldest,
                buy_alpaca_order_id=buy_order_id,
                expected_buy_status="filled",
                expected_buy_alpaca_order_id=buy_order_id,
                expected_filled_qty=2,
                expected_filled_avg_price=expected_average_price,
                expected_target_sell_price=expected_average_price * 1.5,
                expected_sell_client_order_id=None,
            )

        self.assertTrue(
            apply_price(
                100,
                oldest="2026-07-20T12:05:00Z",
                latest="2026-07-20T12:10:00Z",
                expected_average_price=100,
            )
        )
        self.assertFalse(
            apply_price(
                99,
                oldest="2026-07-20T12:04:00Z",
                latest="2026-07-20T12:11:00Z",
                expected_average_price=100,
            )
        )
        self.assertTrue(
            apply_price(
                101,
                oldest="2026-07-20T12:11:00Z",
                latest="2026-07-20T12:12:00Z",
                expected_average_price=100,
            )
        )
        row = self.conn.execute(
            "SELECT filled_avg_price, buy_fill_broker_updated_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(row, (101, "2026-07-20T12:12:00.000000Z"))

    def test_incompatible_persisted_buy_components_fall_back_to_scalar_freshness(self) -> None:
        persisted_components = '{"buy-1":"2026-07-20T12:02:00.000000Z"}'

        self.assertFalse(
            alpaca_managed_buy_fill_observation_authorizes_mutation(
                persisted_broker_updated_at="2026-07-20T12:10:00Z",
                persisted_component_revisions=persisted_components,
                observed_broker_updated_at="2026-07-20T12:03:00Z",
                observed_oldest_broker_updated_at="2026-07-20T12:03:00Z",
                observed_component_revisions={"buy-1": "2026-07-20T12:03:00Z"},
            )
        )
        self.assertTrue(
            alpaca_managed_buy_fill_observation_authorizes_mutation(
                persisted_broker_updated_at="2026-07-20T12:10:00Z",
                persisted_component_revisions=persisted_components,
                observed_broker_updated_at="2026-07-20T12:11:00Z",
                observed_oldest_broker_updated_at="2026-07-20T12:11:00Z",
                observed_component_revisions={"buy-1": "2026-07-20T12:11:00Z"},
            )
        )

    def test_incompatible_persisted_buy_components_cannot_bypass_scalar_watermark(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-contaminated-legacy-map",
        )
        order_id = "buy-rsi-buy-TQQQ-contaminated-legacy-map"
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:10:00.000000Z', "
            "buy_fill_component_revisions = ? WHERE id = ?",
            ('{"buy-rsi-buy-TQQQ-contaminated-legacy-map":"2026-07-20T12:02:00.000000Z"}', position_id),
        )
        self.conn.commit()
        before = self.conn.execute(
            "SELECT state_revision, filled_avg_price, target_sell_price, "
            "buy_fill_broker_updated_at, buy_fill_component_revisions "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        expected_snapshot = {
            "expected_buy_status": "filled",
            "expected_buy_alpaca_order_id": order_id,
            "expected_filled_qty": 2,
            "expected_filled_avg_price": 100,
            "expected_target_sell_price": 150,
            "expected_sell_client_order_id": None,
        }
        common = {
            "buy_status": "filled",
            "filled_qty": 2,
            "filled_at": "2026-01-02T14:31:00Z",
            "buy_alpaca_order_id": order_id,
            **expected_snapshot,
        }

        stale = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            **common,
            filled_avg_price=99,
            target_sell_price=148.5,
            buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:03:00Z",
            buy_fill_component_revisions={order_id: "2026-07-20T12:03:00Z"},
        )
        after_stale = self.conn.execute(
            "SELECT state_revision, filled_avg_price, target_sell_price, "
            "buy_fill_broker_updated_at, buy_fill_component_revisions "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        repaired = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            **common,
            filled_avg_price=100,
            target_sell_price=150,
            buy_fill_broker_updated_at="2026-07-20T12:10:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:10:00Z",
            buy_fill_component_revisions={order_id: "2026-07-20T12:10:00Z"},
        )
        corrected = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            **common,
            filled_avg_price=101,
            target_sell_price=151.5,
            buy_fill_broker_updated_at="2026-07-20T12:11:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:11:00Z",
            buy_fill_component_revisions={order_id: "2026-07-20T12:11:00Z"},
        )
        after_correction = self.conn.execute(
            "SELECT filled_avg_price, target_sell_price, buy_fill_broker_updated_at, "
            "buy_fill_component_revisions FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(stale)
        self.assertEqual(after_stale, before)
        self.assertTrue(repaired)
        self.assertTrue(corrected)
        self.assertEqual(after_correction[:3], (101, 151.5, "2026-07-20T12:11:00.000000Z"))
        self.assertIn('"2026-07-20T12:11:00.000000Z"', after_correction[3])

    def test_stale_component_snapshot_cannot_replace_legacy_buy_scalar_watermark(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-stale-legacy-map",
        )
        order_id = "buy-rsi-buy-TQQQ-stale-legacy-map"
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:10:00.000000Z', "
            "buy_fill_component_revisions = NULL WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()
        before = self.conn.execute(
            "SELECT state_revision, buy_status, filled_qty, filled_avg_price, "
            "buy_fill_broker_updated_at, buy_fill_component_revisions "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        expected_snapshot = {
            "expected_buy_status": "filled",
            "expected_buy_alpaca_order_id": order_id,
            "expected_filled_qty": 2,
            "expected_filled_avg_price": 100,
            "expected_target_sell_price": 150,
            "expected_sell_client_order_id": None,
        }

        replayed = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            buy_fill_broker_updated_at="2026-07-20T12:02:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
            buy_fill_component_revisions={order_id: "2026-07-20T12:02:00Z"},
            buy_alpaca_order_id=order_id,
            **expected_snapshot,
        )
        corrected = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=101,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=151.5,
            buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:03:00Z",
            buy_fill_component_revisions={order_id: "2026-07-20T12:03:00Z"},
            buy_alpaca_order_id=order_id,
            **expected_snapshot,
        )
        after = self.conn.execute(
            "SELECT state_revision, buy_status, filled_qty, filled_avg_price, "
            "buy_fill_broker_updated_at, buy_fill_component_revisions "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertTrue(replayed)
        self.assertFalse(corrected)
        self.assertEqual(after, before)

    def test_legacy_buy_revision_replay_seeds_components_then_accepts_successor_fill(
        self,
    ) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-legacy-lineage",
            buy_alpaca_order_id="buy-leaf",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            buy_alpaca_order_id="buy-leaf",
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:01:00.000000Z' "
            "WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()
        expected_snapshot = {
            "expected_buy_status": "partially_filled",
            "expected_buy_alpaca_order_id": "buy-leaf",
            "expected_filled_qty": 1,
            "expected_filled_avg_price": 100,
            "expected_target_sell_price": 150,
            "expected_sell_client_order_id": None,
        }

        seeded = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            buy_fill_broker_updated_at="2026-07-20T12:02:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:01:00Z",
            buy_fill_component_revisions={
                "buy-root": "2026-07-20T12:01:00Z",
                "buy-leaf": "2026-07-20T12:02:00Z",
            },
            buy_alpaca_order_id="buy-leaf",
            **expected_snapshot,
        )
        seeded_row = self.conn.execute(
            "SELECT buy_fill_broker_updated_at, buy_fill_component_revisions "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        mixed_stale = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=99,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=148.5,
            buy_fill_broker_updated_at="2026-07-20T12:04:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:00:00Z",
            buy_fill_component_revisions={
                "buy-root": "2026-07-20T12:00:00Z",
                "buy-leaf": "2026-07-20T12:04:00Z",
            },
            buy_alpaca_order_id="buy-leaf",
            **expected_snapshot,
        )
        filled = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=101,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=151.5,
            buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:01:00Z",
            buy_fill_component_revisions={
                "buy-root": "2026-07-20T12:01:00Z",
                "buy-leaf": "2026-07-20T12:03:00Z",
            },
            buy_alpaca_order_id="buy-leaf",
            **expected_snapshot,
        )
        filled_row = self.conn.execute(
            "SELECT buy_status, filled_qty, filled_avg_price, buy_fill_broker_updated_at, "
            "buy_fill_component_revisions FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertTrue(seeded)
        self.assertEqual(seeded_row[0], "2026-07-20T12:02:00.000000Z")
        self.assertIn('"buy-root":"2026-07-20T12:01:00.000000Z"', seeded_row[1])
        self.assertFalse(mixed_stale)
        self.assertTrue(filled)
        self.assertEqual(filled_row[:4], ("filled", 2, 101, "2026-07-20T12:03:00.000000Z"))
        self.assertIn('"buy-leaf":"2026-07-20T12:03:00.000000Z"', filled_row[4])
        with self.assertRaisesRegex(ValueError, "requires a broker timestamp"):
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=101,
                filled_at="2026-01-02T14:32:00Z",
                target_sell_price=151.5,
                buy_fill_broker_updated_at=None,
                buy_fill_component_revisions={"buy-leaf": None},
            )

    def test_legacy_buy_changed_successor_requires_matching_staged_replay(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-legacy-staged",
            buy_alpaca_order_id="buy-leaf",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            buy_alpaca_order_id="buy-leaf",
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:02:00.000000Z' "
            "WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()
        expected_snapshot = {
            "expected_buy_status": "partially_filled",
            "expected_buy_alpaca_order_id": "buy-leaf",
            "expected_filled_qty": 1,
            "expected_filled_avg_price": 100,
            "expected_target_sell_price": 150,
            "expected_sell_client_order_id": None,
        }
        revisions = {
            "buy-root": "2026-07-20T12:02:00Z",
            "buy-leaf": "2026-07-20T12:03:00Z",
        }

        staged = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=101,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=151.5,
            buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
            buy_fill_component_revisions=revisions,
            buy_alpaca_order_id="buy-leaf",
            **expected_snapshot,
        )
        staged_row = self.conn.execute(
            "SELECT filled_qty, buy_fill_broker_updated_at, "
            "buy_fill_component_revisions, buy_fill_pending_observation "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        conflicting = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=99,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=148.5,
            buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
            buy_fill_component_revisions=revisions,
            buy_alpaca_order_id="buy-leaf",
            **expected_snapshot,
        )
        applied = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=101,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=151.5,
            buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
            buy_fill_component_revisions=revisions,
            buy_alpaca_order_id="buy-leaf",
            **expected_snapshot,
        )
        applied_row = self.conn.execute(
            "SELECT buy_status, filled_qty, filled_avg_price, "
            "buy_fill_broker_updated_at, buy_fill_pending_observation "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(staged)
        self.assertEqual(staged_row[:2], (1, "2026-07-20T12:02:00.000000Z"))
        self.assertIsNotNone(staged_row[2])
        self.assertIsNotNone(staged_row[3])
        self.assertFalse(conflicting)
        self.assertTrue(applied)
        self.assertEqual(
            applied_row,
            ("filled", 2, 101, "2026-07-20T12:03:00.000000Z", None),
        )

    def test_buy_component_restage_preserves_caller_owned_transaction(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="RESTAGE",
            alpaca_asset_id="asset-restage",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-component-restage",
            buy_alpaca_order_id="buy-leaf",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            buy_alpaca_order_id="buy-leaf",
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:02:00.000000Z' "
            "WHERE id = ?",
            (position_id,),
        )
        self.conn.execute("CREATE TABLE restage_probe (value TEXT NOT NULL)")
        self.conn.commit()
        expected_snapshot = {
            "expected_buy_status": "partially_filled",
            "expected_buy_alpaca_order_id": "buy-leaf",
            "expected_filled_qty": 1,
            "expected_filled_avg_price": 100,
            "expected_target_sell_price": 150,
            "expected_sell_client_order_id": None,
        }
        self.assertFalse(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=101,
                filled_at="2026-01-02T14:32:00Z",
                target_sell_price=151.5,
                buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
                buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
                buy_fill_component_revisions={
                    "buy-root": "2026-07-20T12:02:00Z",
                    "buy-leaf": "2026-07-20T12:03:00Z",
                },
                buy_alpaca_order_id="buy-leaf",
                **expected_snapshot,
            )
        )
        staged_before = self.conn.execute(
            "SELECT buy_fill_component_revisions, buy_fill_pending_observation "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.conn.execute("BEGIN")
        self.conn.execute("INSERT INTO restage_probe VALUES ('caller work')")
        self.assertFalse(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=102,
                filled_at="2026-01-02T14:32:00Z",
                target_sell_price=153,
                buy_fill_broker_updated_at="2026-07-20T12:04:00Z",
                buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
                buy_fill_component_revisions={
                    "buy-root": "2026-07-20T12:02:00Z",
                    "buy-leaf": "2026-07-20T12:04:00Z",
                },
                buy_alpaca_order_id="buy-leaf",
                **expected_snapshot,
            )
        )
        self.assertTrue(self.conn.in_transaction)
        self.assertNotEqual(
            self.conn.execute(
                "SELECT buy_fill_component_revisions, buy_fill_pending_observation "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            staged_before,
        )
        self.conn.rollback()
        self.assertEqual(
            self.conn.execute(
                "SELECT buy_fill_component_revisions, buy_fill_pending_observation "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            staged_before,
        )
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM restage_probe").fetchone()[0], 0)

        self.close_position(position_id)
        closed_snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
        closed_before = self.conn.execute(
            "SELECT state_revision, buy_fill_component_revisions, "
            "buy_fill_pending_observation, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        self.conn.execute("BEGIN")
        self.conn.execute("INSERT INTO restage_probe VALUES ('closed caller work')")
        self.assertEqual(
            apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(closed_snapshot["closed_at"]),
                expected_state_revision=int(closed_snapshot["state_revision"]),
                alpaca_asset_id="asset-restage",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=102,
                filled_at="2026-01-02T14:32:00Z",
                target_sell_price=153,
                sell_status=None,
                sell_fills=[],
                sell_filled_at=None,
                notes="restage a newer closed observation",
                buy_fill_broker_updated_at="2026-07-20T12:04:00Z",
                buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
                buy_fill_component_revisions={
                    "buy-root": "2026-07-20T12:02:00Z",
                    "buy-leaf": "2026-07-20T12:04:00Z",
                },
            ),
            (False, False, False, 0.0),
        )
        self.assertTrue(self.conn.in_transaction)
        self.assertNotEqual(
            self.conn.execute(
                "SELECT state_revision, buy_fill_component_revisions, "
                "buy_fill_pending_observation, closed_at "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            closed_before,
        )
        self.conn.rollback()
        self.assertEqual(
            self.conn.execute(
                "SELECT state_revision, buy_fill_component_revisions, "
                "buy_fill_pending_observation, closed_at "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            closed_before,
        )
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM restage_probe").fetchone()[0], 0)

    def test_closed_legacy_component_seed_preserves_caller_owned_transaction(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-closed-legacy-seed-transaction",
            buy_alpaca_order_id="buy-leaf",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            buy_alpaca_order_id="buy-leaf",
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:02:00.000000Z' "
            "WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()
        self.close_position(position_id)
        self.conn.execute("CREATE TABLE closed_seed_probe (value TEXT NOT NULL)")
        self.conn.commit()
        closed_snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
        closed_before = self.conn.execute(
            "SELECT state_revision, buy_fill_component_revisions, "
            "buy_fill_pending_observation, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.conn.execute("BEGIN")
        self.conn.execute("INSERT INTO closed_seed_probe VALUES ('caller work')")
        self.assertEqual(
            apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(closed_snapshot["closed_at"]),
                expected_state_revision=int(closed_snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=101,
                filled_at="2026-01-02T14:32:00Z",
                target_sell_price=151.5,
                sell_status=None,
                sell_fills=[],
                sell_filled_at=None,
                notes="stage replacement lineage for a closed legacy scalar revision",
                buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
                buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
                buy_fill_component_revisions={
                    "buy-root": "2026-07-20T12:02:00Z",
                    "buy-leaf": "2026-07-20T12:03:00Z",
                },
            ),
            (False, False, False, 0.0),
        )
        self.assertTrue(self.conn.in_transaction)
        self.assertNotEqual(
            self.conn.execute(
                "SELECT state_revision, buy_fill_component_revisions, "
                "buy_fill_pending_observation, closed_at "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            closed_before,
        )

        self.conn.rollback()
        self.assertEqual(
            self.conn.execute(
                "SELECT state_revision, buy_fill_component_revisions, "
                "buy_fill_pending_observation, closed_at "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            closed_before,
        )
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM closed_seed_probe").fetchone()[0], 0)

    def test_closed_correction_success_preserves_caller_owned_transaction(self) -> None:
        position_id = self.save_position(
            symbol="CORRECTIONTXN",
            client_order_id="buy-closed-correction-success-transaction",
            alpaca_asset_id="asset-correction-transaction",
        )
        self.close_position(position_id)
        self.conn.execute("CREATE TABLE closed_correction_success_probe (value INTEGER NOT NULL)")
        self.conn.commit()
        snapshot = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
        before_parent = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.conn.execute("BEGIN")
        self.conn.execute("INSERT INTO closed_correction_success_probe VALUES (7)")
        result = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-correction-transaction",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="filled",
            sell_fills=[("sell-closed-correction-success-transaction", 2, 300)],
            sell_filled_at="2026-01-02T14:33:00Z",
            notes="successful correction inside caller transaction",
        )

        self.assertEqual(result, (True, False, False, 0.0))
        self.assertTrue(self.conn.in_transaction)
        self.assertNotEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before_parent,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT alpaca_order_id, filled_qty, filled_value "
                "FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchall(),
            [("sell-closed-correction-success-transaction", 2, 300)],
        )

        self.conn.rollback()
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before_parent,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchone(),
            (0,),
        )
        self.assertEqual(self.conn.execute("SELECT * FROM closed_correction_success_probe").fetchall(), [])

    def test_closed_correction_rejects_zero_buy_fill_with_average_price_without_writes(self) -> None:
        position_id = self.save_position(
            symbol="ZEROCORRECTION",
            client_order_id="buy-zero-closed-correction",
            alpaca_asset_id="asset-zero-correction",
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
        before_parent = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "zero managed Alpaca buy fill"):
            apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-zero-correction",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="canceled",
                filled_qty=0,
                filled_avg_price=100,
                filled_at=None,
                target_sell_price=None,
                sell_status=None,
                sell_fills=[],
                sell_filled_at=None,
                notes="invalid zero-fill correction",
            )

        self.assertFalse(self.conn.in_transaction)
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before_parent,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchone(),
            (0,),
        )
        self.assertEqual(
            load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]["filled_qty"],
            2,
        )

    def test_closed_correction_false_returns_preserve_caller_owned_transaction(self) -> None:
        for index, branch in enumerate(("fence_miss", "stale_observation", "final_cas_miss")):
            connection_factory = (
                ClosedCorrectionFinalCasMissConnection if branch == "final_cas_miss" else sqlite3.Connection
            )
            with (
                self.subTest(branch=branch),
                closing(sqlite3.connect(":memory:", factory=connection_factory)) as conn,
            ):
                init_state_db(conn)
                symbol = f"FALSE-{index}"
                asset_id = f"asset-false-{index}"
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol=symbol,
                    alpaca_asset_id=asset_id,
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=f"buy-closed-correction-{branch}",
                    buy_alpaca_order_id=f"buy-closed-correction-{branch}",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="filled",
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                )
                mark_alpaca_managed_buy_filled(
                    conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=2,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                if branch == "stale_observation":
                    conn.execute(
                        """
                        UPDATE alpaca_managed_positions
                        SET buy_status = 'canceled',
                            filled_qty = NULL,
                            filled_avg_price = NULL,
                            filled_at = NULL,
                            target_sell_price = NULL,
                            buy_fill_broker_updated_at = '2026-07-20T12:10:00.000000Z',
                            closed_at = '2026-01-03T00:00:00Z'
                        WHERE id = ?
                        """,
                        (position_id,),
                    )
                else:
                    conn.execute(
                        "UPDATE alpaca_managed_positions SET closed_at = '2026-01-03T00:00:00Z' WHERE id = ?",
                        (position_id,),
                    )
                conn.execute("CREATE TABLE closed_correction_probe (value INTEGER NOT NULL)")
                conn.commit()
                closed_before = conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()
                revision_and_close = conn.execute(
                    "SELECT state_revision, closed_at FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()
                expected_revision = int(revision_and_close[0])
                expected_closed_at = str(revision_and_close[1])
                if branch == "final_cas_miss":
                    conn.miss_next_closed_correction_final_cas = True  # type: ignore[attr-defined]

                conn.execute("BEGIN")
                conn.execute("INSERT INTO closed_correction_probe VALUES (7)")
                correction_kwargs: dict[str, object] = {}
                if branch == "stale_observation":
                    correction_kwargs.update(
                        buy_fill_broker_updated_at="2026-07-20T12:11:00Z",
                        buy_fill_broker_oldest_updated_at="2026-07-20T12:10:00Z",
                    )
                result = apply_alpaca_closed_position_broker_correction(
                    conn,
                    position_id,
                    expected_closed_at=expected_closed_at,
                    expected_state_revision=(expected_revision + 1 if branch == "fence_miss" else expected_revision),
                    alpaca_asset_id=asset_id,
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                    buy_status="filled",
                    filled_qty=2,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                    sell_status=None,
                    sell_fills=[],
                    sell_filled_at=None,
                    notes=f"exercise closed correction {branch}",
                    **correction_kwargs,
                )

                self.assertEqual(result, (False, False, False, 0.0))
                self.assertTrue(conn.in_transaction)
                self.assertEqual(conn.execute("SELECT value FROM closed_correction_probe").fetchall(), [(7,)])
                self.assertEqual(
                    conn.execute(
                        "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    closed_before,
                )
                conn.rollback()
                self.assertEqual(conn.execute("SELECT * FROM closed_correction_probe").fetchall(), [])
                self.assertEqual(
                    conn.execute(
                        "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    closed_before,
                )

    def test_closed_legacy_buy_changed_successor_requires_matching_staged_replay(
        self,
    ) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-closed-legacy-staged",
            buy_alpaca_order_id="buy-leaf",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            buy_alpaca_order_id="buy-leaf",
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:02:00.000000Z' "
            "WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()
        self.close_position(position_id)
        revisions = {
            "buy-root": "2026-07-20T12:02:00Z",
            "buy-leaf": "2026-07-20T12:03:00Z",
        }

        def correction(*, price: float) -> tuple[bool, bool, bool, float]:
            snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
            return apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=price,
                filled_at="2026-01-02T14:32:00Z",
                target_sell_price=price * 1.5,
                sell_status=None,
                sell_fills=[],
                sell_filled_at=None,
                notes="closed staged successor correction",
                buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
                buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
                buy_fill_component_revisions=revisions,
            )

        staged = correction(price=101)
        staged_row = self.conn.execute(
            "SELECT filled_qty, buy_fill_broker_updated_at, "
            "buy_fill_component_revisions, buy_fill_pending_observation "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        conflicting = correction(price=99)
        applied = correction(price=101)
        applied_row = self.conn.execute(
            "SELECT buy_status, filled_qty, filled_avg_price, "
            "buy_fill_broker_updated_at, buy_fill_pending_observation, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(staged, (False, False, False, 0.0))
        self.assertEqual(staged_row[:2], (1, "2026-07-20T12:02:00.000000Z"))
        self.assertIsNotNone(staged_row[2])
        self.assertIsNotNone(staged_row[3])
        self.assertEqual(conflicting, (False, False, False, 0.0))
        self.assertEqual(applied[:3], (True, True, False))
        self.assertEqual(
            applied_row,
            ("filled", 2, 101, "2026-07-20T12:03:00.000000Z", None, None),
        )

    def test_active_legacy_buy_stage_can_complete_after_position_closes(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-stage-before-close",
            buy_alpaca_order_id="buy-leaf",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=1,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            buy_alpaca_order_id="buy-leaf",
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:02:00.000000Z' "
            "WHERE id = ?",
            (position_id,),
        )
        self.conn.commit()
        revisions = {
            "buy-root": "2026-07-20T12:02:00Z",
            "buy-leaf": "2026-07-20T12:03:00Z",
        }
        staged = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=101,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=151.5,
            buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
            buy_fill_component_revisions=revisions,
            buy_alpaca_order_id="buy-leaf",
            expected_buy_status="partially_filled",
            expected_buy_alpaca_order_id="buy-leaf",
            expected_filled_qty=1,
            expected_filled_avg_price=100,
            expected_target_sell_price=150,
            expected_sell_client_order_id=None,
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]

        applied = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=101,
            filled_at="2026-01-02T14:32:00Z",
            target_sell_price=151.5,
            sell_status=None,
            sell_fills=[],
            sell_filled_at=None,
            notes="complete staged observation after close",
            buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:02:00Z",
            buy_fill_component_revisions=revisions,
        )
        row = self.conn.execute(
            "SELECT buy_status, filled_qty, filled_avg_price, "
            "buy_fill_pending_observation, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(staged)
        self.assertEqual(applied[:3], (True, True, False))
        self.assertEqual(row, ("filled", 2, 101, None, None))

    def test_closed_buy_correction_orders_each_replacement_component(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-closed-lineage",
            buy_alpaca_order_id="buy-leaf",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="partially_filled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=1,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
                buy_fill_broker_updated_at="2026-07-20T12:02:00Z",
                buy_fill_broker_oldest_updated_at="2026-07-20T12:01:00Z",
                buy_fill_component_revisions={
                    "buy-root": "2026-07-20T12:01:00Z",
                    "buy-leaf": "2026-07-20T12:02:00Z",
                },
                buy_alpaca_order_id="buy-leaf",
            )
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
        common = {
            "expected_closed_at": str(snapshot["closed_at"]),
            "expected_state_revision": int(snapshot["state_revision"]),
            "alpaca_asset_id": "asset-tqqq",
            "buy_order_qty": 2,
            "buy_order_limit_price": 105,
            "buy_status": "filled",
            "filled_qty": 2,
            "filled_avg_price": 101,
            "filled_at": "2026-01-02T14:32:00Z",
            "target_sell_price": 151.5,
            "sell_status": None,
            "sell_fills": [],
            "sell_filled_at": None,
            "notes": "closed successor fill correction",
        }

        mixed_stale = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            **common,
            buy_fill_broker_updated_at="2026-07-20T12:04:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:00:00Z",
            buy_fill_component_revisions={
                "buy-root": "2026-07-20T12:00:00Z",
                "buy-leaf": "2026-07-20T12:04:00Z",
            },
        )
        successor_fill = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            **common,
            buy_fill_broker_updated_at="2026-07-20T12:03:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:01:00Z",
            buy_fill_component_revisions={
                "buy-root": "2026-07-20T12:01:00Z",
                "buy-leaf": "2026-07-20T12:03:00Z",
            },
        )
        corrected = self.conn.execute(
            "SELECT buy_status, filled_qty, buy_fill_broker_updated_at, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(mixed_stale, (False, False, False, 0.0))
        self.assertEqual(successor_fill, (True, True, False, 2.0))
        self.assertEqual(
            corrected,
            ("filled", 2, "2026-07-20T12:03:00.000000Z", None),
        )

    def test_managed_buy_fill_does_not_bypass_revision_after_zero_bust(self) -> None:
        client_order_id = "rsi-buy-TQQQ-zero-bust-active"
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id=client_order_id,
        )
        buy_order_id = f"buy-{client_order_id}"
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET buy_status = 'canceled', filled_qty = NULL,
                filled_avg_price = NULL, filled_at = NULL,
                target_sell_price = NULL, remaining_qty = 0,
                buy_fill_broker_updated_at = '2026-07-20T12:10:00.000000Z',
                notes = 'newer zero-fill bust'
            WHERE id = ?
            """,
            (position_id,),
        )
        self.conn.commit()
        common = {
            "buy_status": "filled",
            "filled_qty": 2,
            "filled_avg_price": 100,
            "filled_at": "2026-01-02T14:31:00Z",
            "target_sell_price": 150,
            "buy_alpaca_order_id": buy_order_id,
            "expected_buy_status": "canceled",
            "expected_buy_alpaca_order_id": buy_order_id,
            "expected_filled_qty": None,
            "expected_filled_avg_price": None,
            "expected_target_sell_price": None,
            "expected_sell_client_order_id": None,
        }

        stale = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            **common,
            buy_fill_broker_updated_at="2026-07-20T12:11:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:10:00Z",
            notes="mixed stale lineage",
        )
        after_stale = self.conn.execute(
            "SELECT buy_status, filled_qty, buy_fill_broker_updated_at, notes "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        newer = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            **common,
            buy_fill_broker_updated_at="2026-07-20T12:12:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:11:00Z",
            notes="fully newer lineage",
        )

        self.assertFalse(stale)
        self.assertEqual(
            after_stale,
            (
                "canceled",
                None,
                "2026-07-20T12:10:00.000000Z",
                "newer zero-fill bust",
            ),
        )
        self.assertTrue(newer)

    def test_closed_buy_correction_does_not_bypass_revision_after_zero_bust(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-zero-bust-closed",
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET buy_fill_broker_updated_at = '2026-07-20T12:00:00.000000Z' "
            "WHERE id = ?",
            (position_id,),
        )
        self.close_position(position_id)

        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
        busted = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="canceled",
            filled_qty=0,
            filled_avg_price=None,
            filled_at=None,
            target_sell_price=None,
            sell_status=None,
            sell_fills=[],
            sell_filled_at=None,
            notes="newer zero-fill bust",
            buy_fill_broker_updated_at="2026-07-20T12:10:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:10:00Z",
        )
        after_bust = load_alpaca_managed_positions(self.conn).iloc[0]
        replayed = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(after_bust["closed_at"]),
            expected_state_revision=int(after_bust["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status=None,
            sell_fills=[],
            sell_filled_at=None,
            notes="mixed stale lineage",
            buy_fill_broker_updated_at="2026-07-20T12:11:00Z",
            buy_fill_broker_oldest_updated_at="2026-07-20T12:10:00Z",
        )
        row = self.conn.execute(
            "SELECT buy_status, filled_qty, buy_fill_broker_updated_at, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(busted[:2], (True, False))
        self.assertEqual(replayed, (False, False, False, 0.0))
        self.assertEqual(row[:3], ("canceled", None, "2026-07-20T12:10:00.000000Z"))
        self.assertIsNotNone(row[3])

    def test_managed_buy_stale_exact_replay_preserves_newer_notes(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-stale-note",
        )
        buy_order_id = "buy-rsi-buy-TQQQ-stale-note"
        common = {
            "buy_status": "filled",
            "filled_qty": 2,
            "filled_avg_price": 100,
            "filled_at": "2026-01-02T14:31:00Z",
            "target_sell_price": 150,
            "buy_alpaca_order_id": buy_order_id,
            "expected_buy_status": "filled",
            "expected_buy_alpaca_order_id": buy_order_id,
            "expected_filled_qty": 2,
            "expected_filled_avg_price": 100,
            "expected_target_sell_price": 150,
            "expected_sell_client_order_id": None,
        }
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                **common,
                buy_fill_broker_updated_at="2026-07-20T12:00:00Z",
            )
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                **common,
                buy_fill_broker_updated_at="2026-07-20T12:02:00Z",
                notes="newer diagnostic",
            )
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                **common,
                buy_fill_broker_updated_at="2026-07-20T12:01:00Z",
                notes="stale diagnostic",
            )
        )
        row = self.conn.execute(
            "SELECT buy_fill_broker_updated_at, notes FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(
            row,
            ("2026-07-20T12:02:00.000000Z", "newer diagnostic"),
        )

    def test_legacy_buy_timestamp_requires_matching_replay_before_price_change(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-legacy-buy-timestamp",
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=101,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=151.5,
                expected_buy_status="filled",
                expected_buy_alpaca_order_id=("buy-rsi-buy-TQQQ-legacy-buy-timestamp"),
                expected_filled_qty=2,
                expected_filled_avg_price=100,
                expected_target_sell_price=150,
                expected_sell_client_order_id=None,
            )
        )
        changed = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=99,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=148.5,
            buy_fill_broker_updated_at="2026-07-20T12:00:00Z",
            buy_alpaca_order_id="buy-rsi-buy-TQQQ-legacy-buy-timestamp",
            expected_buy_status="filled",
            expected_buy_alpaca_order_id=("buy-rsi-buy-TQQQ-legacy-buy-timestamp"),
            expected_filled_qty=2,
            expected_filled_avg_price=101,
            expected_target_sell_price=151.5,
            expected_sell_client_order_id=None,
        )
        row = self.conn.execute(
            """
            SELECT filled_qty, filled_avg_price, target_sell_price,
                   buy_fill_broker_updated_at
            FROM alpaca_managed_positions WHERE id = ?
            """,
            (position_id,),
        ).fetchone()

        self.assertFalse(changed)
        self.assertEqual(row, (2, 101, 151.5, None))

    def test_closed_correction_keeps_newer_sell_fill_when_delayed_audit_arrives(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-delayed-closed-sell",
        )
        self.close_position(position_id)

        def apply_sell(
            average_price: float,
            broker_updated_at: str,
            *,
            destructive_stale_path: bool = False,
        ) -> bool:
            snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
            applied, _reopened, _active_conflict, _remaining_qty = apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
                sell_status=("partially_filled" if destructive_stale_path else "filled"),
                sell_fills=[("sell-1", 2, 2 * average_price)],
                sell_filled_at=("2026-01-02T14:32:00Z" if destructive_stale_path else "2026-01-02T14:33:00Z"),
                notes="timestamped closed sell audit",
                sell_fill_broker_updated_at={"sell-1": broker_updated_at},
                current_sell_leaf_alpaca_order_id="sell-1",
                current_sell_leaf_status=("partially_filled" if destructive_stale_path else "filled"),
                sell_cancellation_alpaca_order_ids=(["sell-1"] if destructive_stale_path else []),
                force_reopen_sell=destructive_stale_path,
            )
            return applied

        self.assertTrue(apply_sell(155, "2026-07-20T14:35:00Z"))
        self.assertFalse(
            apply_sell(
                150,
                "2026-07-20T14:34:00Z",
                destructive_stale_path=True,
            )
        )
        ledger = self.conn.execute(
            """
            SELECT filled_qty, filled_value, broker_updated_at
            FROM alpaca_managed_sell_fills
            WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'
            """,
            (position_id,),
        ).fetchone()
        position = self.conn.execute(
            "SELECT sell_status, sell_filled_at, sold_qty, sold_value, realized_pl, closed_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(ledger, (2, 310, "2026-07-20T14:35:00.000000Z"))
        self.assertEqual(position[:5], ("filled", "2026-01-02T14:33:00Z", 2, 310, 110))
        self.assertIsNotNone(position[5])

    def test_closed_correction_keeps_newer_buy_price_when_delayed_audit_arrives(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-delayed-closed-buy",
        )
        self.close_position(position_id)

        def apply_buy(average_price: float, broker_updated_at: str) -> bool:
            snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
            applied, _reopened, _active_conflict, _remaining_qty = apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=average_price,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=average_price * 1.5,
                sell_status="filled",
                sell_fills=[("sell-1", 2, 300)],
                sell_filled_at="2026-01-02T14:33:00Z",
                notes="timestamped closed buy audit",
                buy_fill_broker_updated_at=broker_updated_at,
                sell_fill_broker_updated_at={"sell-1": "2026-07-20T12:00:00Z"},
                current_sell_leaf_alpaca_order_id="sell-1",
                current_sell_leaf_status="filled",
            )
            return applied

        # A matching replay safely establishes the high-water mark for a row
        # created before buy broker revisions were persisted.
        self.assertTrue(apply_buy(100, "2026-07-20T12:00:00Z"))
        self.assertTrue(apply_buy(101, "2026-07-20T12:02:00Z"))
        self.assertFalse(apply_buy(99, "2026-07-20T12:01:00Z"))
        row = self.conn.execute(
            """
            SELECT filled_qty, filled_avg_price, target_sell_price,
                   buy_fill_broker_updated_at, realized_pl
            FROM alpaca_managed_positions WHERE id = ?
            """,
            (position_id,),
        ).fetchone()

        self.assertEqual(row, (2, 101, 151.5, "2026-07-20T12:02:00.000000Z", 98))

    def test_closed_sell_parent_change_requires_its_current_leaf_revision(self) -> None:
        for newer_authority in ("buy", "historical_sell"):
            with self.subTest(newer_authority=newer_authority):
                client_order_id = f"rsi-buy-TQQQ-cross-authority-{newer_authority}"
                sell_client_order_id = f"rsi-exit-TQQQ-cross-authority-{newer_authority}"
                sell_order_id = f"sell-current-{newer_authority}"
                historical_order_id = f"sell-historical-{newer_authority}"
                position_id = self.save_position(
                    symbol="TQQQ",
                    client_order_id=client_order_id,
                )
                record_alpaca_managed_sell_order(
                    self.conn,
                    position_id,
                    sell_client_order_id=sell_client_order_id,
                    sell_alpaca_order_id=sell_order_id,
                    sell_submitted_at="2026-01-02T14:32:00Z",
                    sell_status="filled",
                    sell_order_qty=2,
                    sell_order_limit_price=150,
                )
                mark_alpaca_managed_sell_filled_if_current(
                    self.conn,
                    position_id,
                    expected_sell_client_order_id=sell_client_order_id,
                    sell_status="filled",
                    sell_filled_qty=2,
                    sell_filled_avg_price=155,
                    sell_filled_at="2026-01-02T14:33:00Z",
                    sell_broker_updated_at="2026-07-20T12:00:00Z",
                    sell_alpaca_order_id=sell_order_id,
                )
                if newer_authority == "historical_sell":
                    record_alpaca_managed_sell_generation(
                        self.conn,
                        position_id,
                        historical_order_id,
                    )
                    self.conn.execute(
                        "UPDATE alpaca_managed_sell_fills "
                        "SET broker_updated_at = '2026-07-20T12:00:00.000000Z' "
                        "WHERE managed_position_id = ? AND alpaca_order_id = ?",
                        (position_id, historical_order_id),
                    )
                    self.conn.commit()
                self.assertTrue(
                    mark_alpaca_managed_buy_filled(
                        self.conn,
                        position_id,
                        buy_status="filled",
                        filled_qty=2,
                        filled_avg_price=100,
                        filled_at="2026-01-02T14:31:00Z",
                        target_sell_price=150,
                        buy_fill_broker_updated_at="2026-07-20T12:00:00Z",
                        buy_alpaca_order_id=f"buy-{client_order_id}",
                        expected_buy_status="filled",
                        expected_buy_alpaca_order_id=f"buy-{client_order_id}",
                        expected_filled_qty=2,
                        expected_filled_avg_price=100,
                        expected_target_sell_price=150,
                        expected_sell_client_order_id=sell_client_order_id,
                    )
                )
                self.close_position(position_id)
                snapshot = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
                sell_fills = [(sell_order_id, 2, 310)]
                sell_timestamps = {sell_order_id: "2026-07-20T12:00:00Z"}
                if newer_authority == "historical_sell":
                    sell_fills.append((historical_order_id, 0, 0))
                    sell_timestamps[historical_order_id] = "2026-07-20T12:03:00Z"

                applied, reopened, _active_conflict, _remaining_qty = apply_alpaca_closed_position_broker_correction(
                    self.conn,
                    position_id,
                    expected_closed_at=str(snapshot["closed_at"]),
                    expected_state_revision=int(snapshot["state_revision"]),
                    alpaca_asset_id="asset-tqqq",
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                    buy_status="filled",
                    filled_qty=2,
                    filled_avg_price=(101 if newer_authority == "buy" else 100),
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=(151.5 if newer_authority == "buy" else 150),
                    sell_status="pending_cancel",
                    sell_fills=sell_fills,
                    sell_filled_at="2026-01-02T14:32:00Z",
                    notes="cross-side revision must not authorize sell cancellation",
                    buy_fill_broker_updated_at=(
                        "2026-07-20T12:02:00Z" if newer_authority == "buy" else "2026-07-20T12:00:00Z"
                    ),
                    sell_fill_broker_updated_at=sell_timestamps,
                    current_sell_leaf_alpaca_order_id=sell_order_id,
                    current_sell_leaf_status="partially_filled",
                    sell_cancellation_alpaca_order_ids=[sell_order_id],
                    force_reopen_sell=True,
                )
                row = self.conn.execute(
                    "SELECT sell_status, sell_filled_at, closed_at FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()

                self.assertFalse(applied)
                self.assertFalse(reopened)
                self.assertEqual(row[:2], ("filled", "2026-01-02T14:33:00Z"))
                self.assertIsNotNone(row[2])

    def test_closed_historical_equal_revision_cannot_authorize_cancellation(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-historical-cancel-fence",
        )
        historical_order_id = "sell-historical-cancel-fence"
        record_alpaca_managed_sell_generation(
            self.conn,
            position_id,
            historical_order_id,
        )
        self.conn.execute(
            "UPDATE alpaca_managed_sell_fills "
            "SET broker_updated_at = '2026-07-20T12:00:00.000000Z' "
            "WHERE managed_position_id = ? AND alpaca_order_id = ?",
            (position_id, historical_order_id),
        )
        self.conn.commit()
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]

        applied, reopened, _active_conflict, _remaining_qty = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="pending_cancel",
            sell_fills=[(historical_order_id, 0, 0)],
            sell_filled_at=None,
            notes="equal historical revision cannot own cancellation",
            sell_fill_broker_updated_at={historical_order_id: "2026-07-20T12:00:00Z"},
            current_sell_leaf_alpaca_order_id=None,
            current_sell_leaf_status=None,
            sell_cancellation_alpaca_order_ids=[historical_order_id],
            force_reopen_sell=True,
        )

        self.assertFalse(applied)
        self.assertFalse(reopened)

    def test_closed_current_sell_ancestor_revision_authorizes_parent_fill_time(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-current-ancestor-time",
        )
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value, broker_updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (position_id, "sell-root", 1, 150, "2026-07-20T12:01:00.000000Z"),
                (position_id, "sell-leaf", 1, 160, "2026-07-20T12:02:00.000000Z"),
            ],
        )
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET sell_alpaca_order_id = 'sell-leaf', sell_status = 'filled',
                sell_filled_qty = 2, sell_filled_avg_price = 155,
                sell_filled_at = '2026-01-02T14:33:00Z',
                sold_qty = 2, sold_value = 310, remaining_qty = 0,
                realized_pl = 110, realized_pl_pct = 55,
                closed_at = '2026-01-02T14:35:00Z'
            WHERE id = ?
            """,
            (position_id,),
        )
        self.conn.commit()

        def correct_root(*, revision: str, filled_at: str) -> bool:
            snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
            applied, _reopened, _conflict, _remaining = apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
                sell_status="filled",
                sell_fills=[("sell-root", 1, 150), ("sell-leaf", 1, 160)],
                sell_filled_at=filled_at,
                notes="current ancestor corrected parent fill time",
                sell_fill_broker_updated_at={
                    "sell-root": revision,
                    "sell-leaf": "2026-07-20T12:02:00Z",
                },
                current_sell_leaf_alpaca_order_id="sell-leaf",
                current_sell_leaf_status="filled",
                current_sell_lineage_alpaca_order_ids=[
                    "sell-root",
                    "sell-leaf",
                ],
            )
            return applied

        root_became_max = correct_root(
            revision="2026-07-20T12:03:00Z",
            filled_at="2026-01-02T14:35:00Z",
        )
        root_removed_as_max = correct_root(
            revision="2026-07-20T12:04:00Z",
            filled_at="2026-01-02T14:33:00Z",
        )
        row = self.conn.execute(
            "SELECT sell_filled_at, closed_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertTrue(root_became_max)
        self.assertTrue(root_removed_as_max)
        self.assertEqual(row[0], "2026-01-02T14:33:00Z")
        self.assertIsNotNone(row[1])

    def test_closed_pending_sell_cancellation_replay_survives_active_owner_conflict(
        self,
    ) -> None:
        position_id = self.save_position(
            symbol="OLD",
            client_order_id="rsi-buy-OLD-pending-sell-retry",
            alpaca_asset_id="asset-stable",
        )
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value, broker_updated_at)
            VALUES (?, ?, ?, ?, NULL)
            """,
            [
                (position_id, "sell-current", 2, 300),
                (position_id, "sell-historical", 0, 0),
            ],
        )
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET sell_alpaca_order_id = 'sell-current', sell_status = 'filled',
                sell_filled_qty = 2, sell_filled_avg_price = 150,
                sell_filled_at = '2026-01-02T14:33:00Z',
                sold_qty = 2, sold_value = 300, remaining_qty = 0,
                realized_pl = 100, realized_pl_pct = 50,
                closed_at = '2026-01-02T14:35:00Z'
            WHERE id = ?
            """,
            (position_id,),
        )
        self.conn.commit()
        self.save_position(
            symbol="NEW",
            client_order_id="rsi-buy-NEW-pending-sell-retry",
            alpaca_asset_id="asset-stable",
        )

        def apply_retry() -> tuple[bool, bool, bool, float]:
            snapshot = load_alpaca_managed_positions(self.conn).query("id == @position_id").iloc[0]
            return apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-stable",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
                sell_status="pending_cancel",
                sell_fills=[
                    ("sell-current", 2, 300),
                    ("sell-historical", 0, 0),
                ],
                sell_filled_at="2026-01-02T14:33:00Z",
                notes="retain exact sell cancellation ownership",
                sell_fill_broker_updated_at={
                    "sell-current": "2026-07-20T12:02:00Z",
                    "sell-historical": "2026-07-20T12:02:00Z",
                },
                current_sell_leaf_alpaca_order_id="sell-current",
                current_sell_leaf_status="new",
                current_sell_lineage_alpaca_order_ids=["sell-current"],
                sell_cancellation_alpaca_order_ids=[
                    "sell-current",
                    "sell-historical",
                ],
                force_reopen_sell=True,
            )

        first = apply_retry()
        second = apply_retry()
        row = self.conn.execute(
            "SELECT sell_status, closed_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(first, (True, False, True, 0.0))
        self.assertEqual(second, (True, False, True, 0.0))
        self.assertEqual(row[0], "pending_cancel")
        self.assertIsNotNone(row[1])

    def test_closed_correction_rejects_changed_legacy_null_timestamps(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-legacy-closed-timestamps",
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id="__historical-fill-only__:sell-legacy",
            sell_status="filled",
            sell_filled_qty=2,
            sell_filled_avg_price=155,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id="sell-legacy",
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]

        applied, reopened, active_conflict, remaining_qty = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=99,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=148.5,
            sell_status="partially_filled",
            sell_fills=[("sell-legacy", 1, 150)],
            sell_filled_at="2026-01-02T14:32:00Z",
            notes="unordered legacy correction",
            buy_fill_broker_updated_at="2026-07-20T12:01:00Z",
            sell_fill_broker_updated_at={"sell-legacy": "2026-07-20T12:01:00Z"},
            force_reopen=True,
        )
        row = self.conn.execute(
            """
            SELECT filled_avg_price, target_sell_price,
                   buy_fill_broker_updated_at, sell_status, sold_qty,
                   sold_value, remaining_qty, closed_at
            FROM alpaca_managed_positions WHERE id = ?
            """,
            (position_id,),
        ).fetchone()
        ledger = self.conn.execute(
            """
            SELECT filled_qty, filled_value, broker_updated_at
            FROM alpaca_managed_sell_fills
            WHERE managed_position_id = ? AND alpaca_order_id = 'sell-legacy'
            """,
            (position_id,),
        ).fetchone()

        self.assertFalse(applied)
        self.assertFalse(reopened)
        self.assertFalse(active_conflict)
        self.assertEqual(remaining_qty, 0)
        self.assertEqual(row[:7], (100, 150, None, None, 2, 310, 0))
        self.assertIsNotNone(row[7])
        self.assertEqual(ledger, (2, 310, None))

    def test_state_revision_invalidates_same_second_closed_correction_snapshot(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-20260102",
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
        snapshot_revision = int(snapshot["state_revision"])
        same_timestamp = str(snapshot["updated_at"])

        self.conn.execute(
            "UPDATE alpaca_managed_positions SET notes = 'first', updated_at = ? WHERE id = ?",
            (same_timestamp, position_id),
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET notes = 'second', updated_at = ? WHERE id = ?",
            (same_timestamp, position_id),
        )
        self.conn.commit()
        current_revision = int(
            self.conn.execute(
                "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone()[0]
        )

        applied, reopened, active_conflict, _remaining_qty = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=snapshot_revision,
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="filled",
            sell_fills=[("sell-1", 2, 300)],
            sell_filled_at="2026-01-02T14:33:00Z",
            notes="stale correction",
        )

        self.assertEqual(current_revision, snapshot_revision + 2)
        self.assertFalse(applied)
        self.assertFalse(reopened)
        self.assertFalse(active_conflict)
        self.assertEqual(
            self.conn.execute(
                "SELECT notes FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone()[0],
            "second",
        )

    def test_closed_audit_loader_rotates_a_bounded_batch_and_marker_cannot_touch_reopened_row(self) -> None:
        position_ids: list[int] = []
        for index in range(3):
            position_id = self.save_position(
                symbol=f"TQ{index}",
                client_order_id=f"rsi-buy-TQ{index}-20260102",
                alpaca_asset_id=f"asset-{index}",
            )
            self.close_position(position_id)
            position_ids.append(position_id)

        first_batch = load_recently_closed_alpaca_managed_positions(
            self.conn,
            audit_limit=2,
        )
        self.assertEqual(first_batch["id"].tolist(), position_ids[:2])
        for offset, position in enumerate(first_batch.to_dict("records"), start=1):
            self.assertTrue(
                mark_alpaca_closed_correction_audited(
                    self.conn,
                    int(position["id"]),
                    expected_state_revision=int(position["state_revision"]),
                    audited_at=f"2026-07-18T12:00:00.00000{offset}Z",
                )
            )

        rotated_batch = load_recently_closed_alpaca_managed_positions(
            self.conn,
            audit_limit=2,
        )
        self.assertEqual(rotated_batch["id"].tolist(), [position_ids[2], position_ids[0]])

        third_id = position_ids[2]
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET closed_at = NULL WHERE id = ?",
            (third_id,),
        )
        self.conn.commit()
        revision = int(
            self.conn.execute(
                "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                (third_id,),
            ).fetchone()[0]
        )
        self.assertFalse(
            mark_alpaca_closed_correction_audited(
                self.conn,
                third_id,
                expected_state_revision=revision,
            )
        )
        self.assertIsNone(
            self.conn.execute(
                "SELECT closed_correction_audited_at FROM alpaca_managed_positions WHERE id = ?",
                (third_id,),
            ).fetchone()[0]
        )

    def test_closed_audit_loader_never_bounds_or_marks_retained_sell_retries(self) -> None:
        position_ids: list[int] = []
        for index in range(3):
            position_id = self.save_position(
                symbol=f"RT{index}",
                client_order_id=f"rsi-buy-RT{index}-20260102",
                alpaca_asset_id=f"asset-retained-{index}",
            )
            self.close_position(position_id)
            position_ids.append(position_id)
        self.conn.executemany(
            "UPDATE alpaca_managed_positions "
            "SET sell_status = 'quantity_mismatch', sell_submission_retry_claimed_at = ? "
            "WHERE id = ?",
            [
                ("2026-07-20T12:00:00Z", position_ids[1]),
                ("2026-07-20T12:01:00Z", position_ids[2]),
            ],
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET closed_at = '2020-01-01T00:00:00Z' WHERE id = ?",
            (position_ids[2],),
        )
        self.conn.commit()

        candidates = load_recently_closed_alpaca_managed_positions(
            self.conn,
            audit_days=1,
            audit_limit=1,
        )

        self.assertEqual(candidates["id"].tolist(), [position_ids[1], position_ids[2], position_ids[0]])
        retained = candidates.iloc[0]
        self.assertFalse(
            mark_alpaca_closed_correction_audited(
                self.conn,
                int(retained["id"]),
                expected_state_revision=int(retained["state_revision"]),
            )
        )

    def test_closed_audit_loader_retains_stale_pending_sell_cancellation(self) -> None:
        pending_id = self.save_position(
            symbol="PC0",
            client_order_id="rsi-buy-PC0-20260102",
            alpaca_asset_id="asset-pending-cancel",
        )
        self.close_position(pending_id)
        recent_ids: list[int] = []
        for index in range(2):
            position_id = self.save_position(
                symbol=f"PC{index + 1}",
                client_order_id=f"rsi-buy-PC{index + 1}-20260102",
                alpaca_asset_id=f"asset-recent-{index}",
            )
            self.close_position(position_id)
            recent_ids.append(position_id)
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET sell_status = 'pending_cancel', closed_at = '2020-01-01T00:00:00Z' "
            "WHERE id = ?",
            (pending_id,),
        )
        self.conn.commit()

        candidates = load_recently_closed_alpaca_managed_positions(
            self.conn,
            audit_days=1,
            audit_limit=1,
        )

        self.assertEqual(candidates["id"].tolist(), [pending_id, recent_ids[0]])

    def test_late_partial_sell_fill_persists_when_new_active_owner_blocks_reopen(self) -> None:
        old_position_id = self.save_position(
            symbol="OLD",
            client_order_id="rsi-buy-OLD-old",
            alpaca_asset_id=None,
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions "
            "SET sell_client_order_id = 'rsi-exit-TQQQ-old', "
            "sell_status = 'quantity_mismatch', "
            "sell_submission_retry_claimed_at = '2026-07-20T12:00:00Z', "
            "closed_at = '2026-01-02T14:35:00Z' WHERE id = ?",
            (old_position_id,),
        )
        self.conn.commit()
        new_position_id = self.save_position(
            symbol="NEW",
            client_order_id="rsi-buy-NEW-new",
            alpaca_asset_id="asset-stable",
        )

        remaining_qty, generation_is_current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            old_position_id,
            expected_sell_client_order_id="__stale-fill-only__:sell-stale",
            sell_status="partially_filled",
            sell_filled_qty=0.5,
            sell_filled_avg_price=155,
            sell_filled_at="2026-07-20T12:01:00Z",
            sell_alpaca_order_id="sell-stale",
        )
        old_row = self.conn.execute(
            "SELECT alpaca_asset_id, sell_status, sold_qty, sold_value, remaining_qty, closed_at, "
            "sell_submission_retry_claimed_at FROM alpaca_managed_positions WHERE id = ?",
            (old_position_id,),
        ).fetchone()
        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value FROM alpaca_managed_sell_fills "
            "WHERE managed_position_id = ? AND alpaca_order_id = 'sell-stale'",
            (old_position_id,),
        ).fetchone()

        self.assertFalse(generation_is_current)
        self.assertEqual(remaining_qty, 1.5)
        self.assertEqual(
            old_row,
            (
                None,
                "position_quantity_mismatch",
                0.5,
                77.5,
                1.5,
                "2026-01-02T14:35:00Z",
                "2026-07-20T12:00:00Z",
            ),
        )
        self.assertEqual(ledger, (0.5, 77.5))
        self.assertIsNone(
            self.conn.execute(
                "SELECT closed_at FROM alpaca_managed_positions WHERE id = ?",
                (new_position_id,),
            ).fetchone()[0]
        )

    def test_delayed_equal_quantity_sell_fill_cannot_regress_accounting(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-delayed-sell-fill",
        )

        for average_price, broker_updated_at in (
            (151, "2026-07-20T12:02:00Z"),
            (150, "2026-07-20T12:01:00Z"),
        ):
            mark_alpaca_managed_sell_filled_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id="__historical-fill-only__:sell-1",
                sell_status="partially_filled",
                sell_filled_qty=1,
                sell_filled_avg_price=average_price,
                sell_filled_at="2026-07-20T12:00:00Z",
                sell_broker_updated_at=broker_updated_at,
                sell_alpaca_order_id="sell-1",
            )

        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value, broker_updated_at "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        position = self.conn.execute(
            "SELECT sold_qty, sold_value, sell_filled_avg_price, realized_pl, realized_pl_pct "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(ledger, (1, 151, "2026-07-20T12:02:00.000000Z"))
        self.assertEqual(position, (1, 151, 151, 51, 51))

    def test_zero_fill_sell_status_does_not_advance_positive_fill_revision(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-positive-to-zero-sell",
        )
        sell_client_order_id = "rsi-exit-TQQQ-positive-to-zero-sell"
        sell_order_id = "sell-positive-to-zero"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id=sell_order_id,
            sell_submitted_at="2026-07-20T12:00:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="partially_filled",
            sell_filled_qty=1,
            sell_filled_avg_price=110,
            sell_filled_at="2026-07-20T12:01:00Z",
            sell_broker_updated_at="2026-07-20T12:01:00Z",
            sell_alpaca_order_id=sell_order_id,
        )

        self.assertTrue(
            update_alpaca_managed_sell_status_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                sell_status="canceled",
                sell_alpaca_order_id=sell_order_id,
                observed_sell_filled_qty=0,
                sell_broker_updated_at="2026-07-20T12:02:00Z",
            )
        )
        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value, broker_updated_at "
            "FROM alpaca_managed_sell_fills "
            "WHERE managed_position_id = ? AND alpaca_order_id = ?",
            (position_id, sell_order_id),
        ).fetchone()

        self.assertEqual(ledger, (1, 110, "2026-07-20T12:01:00.000000Z"))

    def test_sell_status_observation_rejects_an_older_conflicting_broker_revision(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-status-revision",
        )
        sell_client_order_id = "rsi-exit-TQQQ-status-revision"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-1",
            sell_submitted_at="2026-07-20T11:59:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        self.assertTrue(
            update_alpaca_managed_sell_status_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                sell_status="canceled",
                sell_alpaca_order_id="sell-1",
                sell_broker_updated_at="2026-07-20T12:02:00Z",
            )
        )
        before = self.conn.execute(
            "SELECT state_revision, sell_status, sell_observation_broker_updated_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        exact_replay_is_current = update_alpaca_managed_sell_status_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="canceled",
            sell_alpaca_order_id="sell-1",
            sell_broker_updated_at="2026-07-20T12:02:00Z",
        )
        after_exact_replay = self.conn.execute(
            "SELECT state_revision, sell_status, sell_observation_broker_updated_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        equal_revision_conflict_applied = update_alpaca_managed_sell_status_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_alpaca_order_id="sell-1",
            sell_broker_updated_at="2026-07-20T12:02:00Z",
        )
        after_equal_revision_conflict = self.conn.execute(
            "SELECT state_revision, sell_status, sell_observation_broker_updated_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        older_same_status_is_current = update_alpaca_managed_sell_status_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="canceled",
            sell_alpaca_order_id="sell-1",
            sell_broker_updated_at="2026-07-20T12:01:00Z",
        )
        after_older_same_status = self.conn.execute(
            "SELECT state_revision, sell_status, sell_observation_broker_updated_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        stale_applied = update_alpaca_managed_sell_status_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_alpaca_order_id="sell-1",
            sell_broker_updated_at="2026-07-20T12:01:00Z",
        )
        after = self.conn.execute(
            "SELECT state_revision, sell_status, sell_observation_broker_updated_at "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertTrue(exact_replay_is_current)
        self.assertEqual(after_exact_replay, before)
        self.assertFalse(equal_revision_conflict_applied)
        self.assertEqual(after_equal_revision_conflict, before)
        self.assertFalse(older_same_status_is_current)
        self.assertEqual(after_older_same_status, before)
        self.assertFalse(stale_applied)
        self.assertEqual(after, before)
        self.assertEqual(after[1:], ("canceled", "2026-07-20T12:02:00.000000Z"))

    def test_rejected_sell_fill_replay_does_not_mutate_parent_or_report_current(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-rejected-sell-replay",
        )
        sell_client_order_id = "rsi-exit-TQQQ-rejected-sell-replay"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-1",
            sell_submitted_at="2026-07-20T11:59:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="canceled",
            sell_filled_qty=1,
            sell_filled_avg_price=151,
            sell_filled_at="2026-07-20T12:02:00Z",
            sell_broker_updated_at="2026-07-20T12:03:00Z",
            sell_alpaca_order_id="sell-1",
        )
        before = self.conn.execute(
            "SELECT state_revision, sell_status, sold_qty, sold_value, remaining_qty "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        remaining_qty, observation_is_current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=2,
            sell_filled_avg_price=150,
            sell_filled_at="2026-07-20T12:01:00Z",
            sell_broker_updated_at="2026-07-20T12:02:00Z",
            sell_alpaca_order_id="sell-1",
        )
        after = self.conn.execute(
            "SELECT state_revision, sell_status, sold_qty, sold_value, remaining_qty "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(observation_is_current)
        self.assertEqual(remaining_qty, 1)
        self.assertEqual(after, before)

    def test_sell_fill_rejects_nonfinite_derived_value_atomically(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-overflowing-sell",
        )
        sell_client_order_id = "rsi-exit-TQQQ-overflowing-sell"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-1",
            sell_submitted_at="2026-07-20T11:59:00Z",
            sell_status="accepted",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        before = self.conn.execute(
            "SELECT state_revision, sell_status, sold_qty, sold_value, remaining_qty "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "fill value must be finite"):
            mark_alpaca_managed_sell_filled_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                sell_status="filled",
                sell_filled_qty=2,
                sell_filled_avg_price=1e308,
                sell_filled_at="2026-07-20T12:01:00Z",
                sell_broker_updated_at="2026-07-20T12:02:00Z",
                sell_alpaca_order_id="sell-1",
            )

        after = self.conn.execute(
            "SELECT state_revision, sell_status, sold_qty, sold_value, remaining_qty "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value FROM alpaca_managed_sell_fills "
            "WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        self.assertEqual(after, before)
        self.assertEqual(ledger, (0, 0))

    def test_delayed_higher_quantity_sell_fill_cannot_overwrite_newer_correction(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-stale-higher-sell-fill",
        )
        common = {
            "expected_sell_client_order_id": "__historical-fill-only__:sell-1",
            "sell_alpaca_order_id": "sell-1",
        }
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_status="partially_filled",
            sell_filled_qty=1,
            sell_filled_avg_price=151,
            sell_filled_at="2026-07-20T12:02:00Z",
            sell_broker_updated_at="2026-07-20T12:03:00Z",
        )

        remaining_qty, generation_is_current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_status="filled",
            sell_filled_qty=2,
            sell_filled_avg_price=150,
            sell_filled_at="2026-07-20T12:01:00Z",
            sell_broker_updated_at="2026-07-20T12:02:00Z",
        )
        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value, broker_updated_at "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        position = self.conn.execute(
            "SELECT sell_status, sold_qty, sold_value, remaining_qty FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(generation_is_current)
        self.assertEqual(remaining_qty, 1)
        self.assertEqual(ledger, (1, 151, "2026-07-20T12:03:00.000000Z"))
        self.assertEqual(position, (None, 1, 151, 1))

    def test_unversioned_sell_fill_retains_legacy_monotonic_quantity_updates(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-unversioned-sell-fill",
        )
        common = {
            "expected_sell_client_order_id": "__historical-fill-only__:sell-1",
            "sell_status": "partially_filled",
            "sell_filled_avg_price": 150,
            "sell_filled_at": "2026-07-20T12:01:00Z",
            "sell_alpaca_order_id": "sell-1",
        }
        for filled_qty in (1, 2):
            mark_alpaca_managed_sell_filled_if_current(
                self.conn,
                position_id,
                **common,
                sell_filled_qty=filled_qty,
            )

        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value, broker_updated_at "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        position = self.conn.execute(
            "SELECT sold_qty, remaining_qty FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(ledger, (2, 300, None))
        self.assertEqual(position, (2, 0))

    def test_newer_equal_quantity_sell_fill_price_correction_is_applied(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-corrected-sell-fill",
        )

        for average_price, broker_updated_at in (
            (150, "2026-07-20T08:01:00-04:00"),
            (151, "2026-07-20T12:02:00Z"),
        ):
            mark_alpaca_managed_sell_filled_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id="__historical-fill-only__:sell-1",
                sell_status="partially_filled",
                sell_filled_qty=1,
                sell_filled_avg_price=average_price,
                sell_filled_at="2026-07-20T12:00:00Z",
                sell_broker_updated_at=broker_updated_at,
                sell_alpaca_order_id="sell-1",
            )

        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value, broker_updated_at "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        position = self.conn.execute(
            "SELECT sold_qty, sold_value, sell_filled_avg_price, realized_pl, realized_pl_pct "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(ledger, (1, 151, "2026-07-20T12:02:00.000000Z"))
        self.assertEqual(position, (1, 151, 151, 51, 51))

    def test_sell_lifecycle_revision_rejects_conflicting_same_revision_fill(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-sell-lifecycle-fill-fence",
        )
        sell_client_order_id = "rsi-exit-TQQQ-sell-lifecycle-fill-fence"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-1",
            sell_submitted_at="2026-07-20T12:00:00Z",
            sell_status="partially_filled",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="partially_filled",
            sell_filled_qty=1,
            sell_filled_avg_price=110,
            sell_filled_at="2026-07-20T12:01:00Z",
            sell_broker_updated_at="2026-07-20T12:01:00Z",
            sell_alpaca_order_id="sell-1",
        )
        self.assertTrue(
            update_alpaca_managed_sell_status_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                sell_status="canceled",
                sell_alpaca_order_id="sell-1",
                observed_sell_filled_qty=0,
                sell_broker_updated_at="2026-07-20T12:02:00Z",
            )
        )

        remaining_qty, observation_is_current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=1,
            sell_filled_avg_price=120,
            sell_filled_at="2026-07-20T12:02:00Z",
            sell_broker_updated_at="2026-07-20T12:02:00Z",
            sell_alpaca_order_id="sell-1",
        )
        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value, broker_updated_at "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        position = self.conn.execute(
            "SELECT sell_status, sell_observation_broker_updated_at, sold_qty, sold_value, remaining_qty "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(observation_is_current)
        self.assertEqual(remaining_qty, 1)
        self.assertEqual(ledger, (1, 110, "2026-07-20T12:01:00.000000Z"))
        self.assertEqual(position, ("canceled", "2026-07-20T12:02:00.000000Z", 1, 110, 1))

    def test_partial_sell_zero_lifecycle_rejects_conflicting_same_revision_fill(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-partial-zero-lifecycle-fill-fence",
        )
        sell_client_order_id = "rsi-exit-TQQQ-partial-zero-lifecycle-fill-fence"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-1",
            sell_submitted_at="2026-07-20T12:00:00Z",
            sell_status="partially_filled",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="partially_filled",
            sell_filled_qty=1,
            sell_filled_avg_price=110,
            sell_filled_at="2026-07-20T12:01:00Z",
            sell_broker_updated_at="2026-07-20T12:01:00Z",
            sell_alpaca_order_id="sell-1",
        )
        self.assertTrue(
            update_alpaca_managed_sell_status_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                sell_status="partially_filled",
                sell_alpaca_order_id="sell-1",
                observed_sell_filled_qty=0,
                sell_broker_updated_at="2026-07-20T12:02:00Z",
            )
        )

        remaining_qty, observation_is_current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="partially_filled",
            sell_filled_qty=1,
            sell_filled_avg_price=120,
            sell_filled_at="2026-07-20T12:02:00Z",
            sell_broker_updated_at="2026-07-20T12:02:00Z",
            sell_alpaca_order_id="sell-1",
        )
        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value, broker_updated_at "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        position = self.conn.execute(
            "SELECT sell_status, sell_observation_broker_updated_at, sell_observation_filled_qty, "
            "sold_qty, sold_value, remaining_qty FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertFalse(observation_is_current)
        self.assertEqual(remaining_qty, 1)
        self.assertEqual(ledger, (1, 110, "2026-07-20T12:01:00.000000Z"))
        self.assertEqual(
            position,
            ("partially_filled", "2026-07-20T12:02:00.000000Z", 0, 1, 110, 1),
        )

    def test_newer_exact_sell_fill_replay_advances_current_status_and_metadata(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-exact-sell-replay",
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="rsi-exit-TQQQ-exact-sell-replay",
            sell_alpaca_order_id="sell-1",
            sell_submitted_at="2026-07-20T11:59:00Z",
            sell_status="partially_filled",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        common = {
            "expected_sell_client_order_id": "rsi-exit-TQQQ-exact-sell-replay",
            "sell_filled_qty": 1,
            "sell_filled_avg_price": 150,
            "sell_filled_at": "2026-07-20T12:00:00Z",
            "sell_alpaca_order_id": "sell-1",
        }
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_status="partially_filled",
            sell_expires_at="2026-07-20T20:00:00Z",
            notes="initial partial fill",
            sell_broker_updated_at="2026-07-20T12:01:00Z",
        )

        remaining_qty, generation_is_current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_status="canceled",
            sell_expires_at="2026-07-21T20:00:00Z",
            notes="newer terminal replay",
            sell_broker_updated_at="2026-07-20T12:02:00Z",
        )
        # An older exact replay must not undo the accepted terminal observation.
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_status="filled",
            sell_expires_at="2026-07-19T20:00:00Z",
            notes="stale replay",
            sell_broker_updated_at="2026-07-20T12:01:30Z",
        )
        ledger = self.conn.execute(
            "SELECT filled_qty, filled_value, broker_updated_at "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        position = self.conn.execute(
            "SELECT sell_status, sell_expires_at, notes, sold_qty, remaining_qty "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertTrue(generation_is_current)
        self.assertEqual(remaining_qty, 1)
        self.assertEqual(ledger, (1, 150, "2026-07-20T12:02:00.000000Z"))
        self.assertEqual(
            position,
            (
                "canceled",
                "2026-07-21T20:00:00Z",
                "newer terminal replay",
                1,
                1,
            ),
        )

    def test_newer_exact_sell_revision_corrects_only_the_current_leaf_fill_time(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-fill-time-correction",
        )
        sell_client_order_id = "rsi-exit-TQQQ-fill-time-correction"
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-current",
            sell_submitted_at="2026-07-20T11:59:00Z",
            sell_status="partially_filled",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        common = {
            "expected_sell_client_order_id": sell_client_order_id,
            "sell_status": "partially_filled",
            "sell_filled_qty": 1,
            "sell_filled_avg_price": 150,
            "sell_alpaca_order_id": "sell-current",
        }
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_filled_at="2026-07-20T12:00:00Z",
            sell_broker_updated_at="2026-07-20T12:01:00Z",
        )

        remaining_qty, observation_is_current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_filled_at="2026-07-20T12:00:30Z",
            sell_broker_updated_at="2026-07-20T12:02:00Z",
        )
        # A historical lineage observation still contributes accounting, but
        # cannot replace the active leaf's execution chronology.
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id="__historical-fill-only__:sell-old",
            sell_status="replaced",
            sell_filled_qty=0.25,
            sell_filled_avg_price=149,
            sell_filled_at="2026-07-20T13:00:00Z",
            sell_broker_updated_at="2026-07-20T13:01:00Z",
            sell_alpaca_order_id="sell-old",
        )
        row = self.conn.execute(
            "SELECT sell_filled_at, sold_qty, remaining_qty FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertTrue(observation_is_current)
        self.assertEqual(remaining_qty, 1)
        self.assertEqual(row, ("2026-07-20T12:00:30Z", 1.25, 0.75))

    def test_legacy_untimestamped_sell_fill_requires_a_matching_replay_before_correction(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-legacy-sell-fill",
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="rsi-exit-TQQQ-legacy-sell-fill",
            sell_alpaca_order_id="sell-1",
            sell_submitted_at="2026-07-20T11:59:00Z",
            sell_status="new",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        common = {
            "expected_sell_client_order_id": "rsi-exit-TQQQ-legacy-sell-fill",
            "sell_filled_qty": 1,
            "sell_filled_at": "2026-07-20T12:00:00Z",
            "sell_alpaca_order_id": "sell-1",
        }
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_status="partially_filled",
            sell_filled_avg_price=151,
        )

        # A changed value cannot be ordered against a migrated NULL timestamp.
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_status="canceled",
            sell_filled_avg_price=150,
            sell_broker_updated_at="2026-07-20T12:02:00Z",
        )
        unknown_ordering = self.conn.execute(
            "SELECT f.filled_value, f.broker_updated_at, p.sell_status "
            "FROM alpaca_managed_sell_fills AS f "
            "JOIN alpaca_managed_positions AS p ON p.id = f.managed_position_id "
            "WHERE f.managed_position_id = ? AND f.alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()

        # A matching observation safely establishes the high-water mark; only
        # corrections after that point may change the value.
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_status="canceled",
            sell_filled_avg_price=151,
            sell_broker_updated_at="2026-07-20T12:02:00Z",
        )
        seeded = self.conn.execute(
            "SELECT f.filled_value, f.broker_updated_at, p.sell_status "
            "FROM alpaca_managed_sell_fills AS f "
            "JOIN alpaca_managed_positions AS p ON p.id = f.managed_position_id "
            "WHERE f.managed_position_id = ? AND f.alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            **common,
            sell_status="canceled",
            sell_filled_avg_price=152,
            sell_broker_updated_at="2026-07-20T12:03:00Z",
        )
        corrected = self.conn.execute(
            "SELECT filled_value, broker_updated_at FROM alpaca_managed_sell_fills "
            "WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()
        position = self.conn.execute(
            "SELECT sold_value, realized_pl FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        self.assertEqual(unknown_ordering, (151, None, "partially_filled"))
        self.assertEqual(seeded, (151, "2026-07-20T12:02:00.000000Z", "canceled"))
        self.assertEqual(corrected, (152, "2026-07-20T12:03:00.000000Z"))
        self.assertEqual(position, (152, 52))

    def test_legacy_untimestamped_sell_replay_can_atomically_own_cancellation(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-legacy-sell-cancel",
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="rsi-exit-TQQQ-1",
            sell_alpaca_order_id="sell-1",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="filled",
            sell_order_qty=2,
            sell_order_limit_price=150,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id="rsi-exit-TQQQ-1",
            sell_status="filled",
            sell_filled_qty=2,
            sell_filled_avg_price=150,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id="sell-1",
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]

        applied, reopened, active_conflict, remaining_qty = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="pending_cancel",
            sell_fills=[("sell-1", 2, 300)],
            sell_filled_at="2026-01-02T14:33:00Z",
            notes="legacy active sell cancellation fenced",
            sell_fill_broker_updated_at={"sell-1": "2026-07-20T12:02:00Z"},
            current_sell_leaf_alpaca_order_id="sell-1",
            current_sell_leaf_status="new",
            sell_cancellation_alpaca_order_ids=["sell-1"],
            force_reopen_sell=True,
        )
        row = self.conn.execute(
            "SELECT sell_status, closed_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        ledger_revision = self.conn.execute(
            "SELECT broker_updated_at FROM alpaca_managed_sell_fills "
            "WHERE managed_position_id = ? AND alpaca_order_id = 'sell-1'",
            (position_id,),
        ).fetchone()[0]

        self.assertTrue(applied)
        self.assertTrue(reopened)
        self.assertFalse(active_conflict)
        self.assertEqual(remaining_qty, 0)
        self.assertEqual(row, ("pending_cancel", None))
        self.assertEqual(ledger_revision, "2026-07-20T12:02:00.000000Z")

    def test_legacy_closed_identity_is_adopted_but_active_asset_owner_blocks_only_reopen(self) -> None:
        legacy_id = self.save_position(
            symbol="OLD",
            client_order_id="rsi-buy-OLD-20260102",
            alpaca_asset_id=None,
            buy_order_qty=None,
            buy_order_limit_price=None,
        )
        self.close_position(legacy_id)
        active_id = self.save_position(
            symbol="NEW",
            client_order_id="rsi-buy-NEW-20260102",
            alpaca_asset_id="asset-stable",
        )
        snapshot = load_alpaca_managed_positions(self.conn).query("id == @legacy_id").iloc[0]

        applied, reopened, active_conflict, remaining_qty = apply_alpaca_closed_position_broker_correction(
            self.conn,
            legacy_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-stable",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="position_quantity_mismatch",
            sell_fills=[],
            sell_filled_at=None,
            notes="unresolved broker correction",
        )
        corrected = load_alpaca_managed_positions(self.conn).query("id == @legacy_id").iloc[0]

        self.assertTrue(applied)
        self.assertFalse(reopened)
        self.assertTrue(active_conflict)
        self.assertEqual(remaining_qty, 2)
        self.assertEqual(corrected["alpaca_asset_id"], "asset-stable")
        self.assertEqual(corrected["buy_order_qty"], 2)
        self.assertEqual(corrected["buy_order_limit_price"], 105)
        self.assertFalse(pd.isna(corrected["closed_at"]))

        self.conn.execute(
            "UPDATE alpaca_managed_positions SET closed_at = '2000-01-01T00:00:00Z' WHERE id = ?",
            (active_id,),
        )
        self.conn.commit()
        retry = load_alpaca_managed_positions(self.conn).query("id == @legacy_id").iloc[0]
        applied, reopened, active_conflict, _remaining_qty = apply_alpaca_closed_position_broker_correction(
            self.conn,
            legacy_id,
            expected_closed_at=str(retry["closed_at"]),
            expected_state_revision=int(retry["state_revision"]),
            alpaca_asset_id="asset-stable",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="position_quantity_mismatch",
            sell_fills=[],
            sell_filled_at=None,
            notes="unresolved broker correction",
        )

        self.assertTrue(applied)
        self.assertTrue(reopened)
        self.assertFalse(active_conflict)

    def test_balanced_closed_price_correction_ignores_active_asset_owner(self) -> None:
        closed_id = self.save_position(
            symbol="OLD",
            client_order_id="rsi-buy-OLD-20260102",
            alpaca_asset_id="asset-stable",
        )
        self.close_position(closed_id)
        self.save_position(
            symbol="NEW",
            client_order_id="rsi-buy-NEW-20260102",
            alpaca_asset_id="asset-stable",
        )
        snapshot = load_alpaca_managed_positions(self.conn).query("id == @closed_id").iloc[0]

        applied, reopened, active_conflict, remaining_qty = apply_alpaca_closed_position_broker_correction(
            self.conn,
            closed_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-stable",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="filled",
            sell_fills=[("sell-1", 2, 310)],
            sell_filled_at="2026-01-02T14:33:00Z",
            notes="balanced price correction",
        )

        self.assertTrue(applied)
        self.assertFalse(reopened)
        self.assertFalse(active_conflict)
        self.assertEqual(remaining_qty, 0)
        corrected = load_alpaca_managed_positions(self.conn).query("id == @closed_id").iloc[0]
        self.assertFalse(pd.isna(corrected["closed_at"]))
        self.assertEqual(corrected["realized_pl"], 110)

    def test_closed_correction_retains_zero_fill_order_identity(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-20260102",
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]

        applied, reopened, active_conflict, remaining_qty = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="filled",
            filled_qty=2,
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
            sell_status="filled",
            sell_fills=[("sell-zero", 0, 0), ("sell-current", 2, 300)],
            sell_filled_at="2026-01-02T14:33:00Z",
            notes="historical sell fill corrected to zero",
        )
        ledger = self.conn.execute(
            "SELECT alpaca_order_id, filled_qty, filled_value "
            "FROM alpaca_managed_sell_fills WHERE managed_position_id = ? ORDER BY alpaca_order_id",
            (position_id,),
        ).fetchall()

        self.assertTrue(applied)
        self.assertFalse(reopened)
        self.assertFalse(active_conflict)
        self.assertEqual(remaining_qty, 0)
        self.assertEqual(ledger, [("sell-current", 2, 300), ("sell-zero", 0, 0)])

    def test_zero_busted_closed_position_remains_correction_audit_eligible(self) -> None:
        position_id = self.save_position(
            symbol="TQQQ",
            client_order_id="rsi-buy-TQQQ-20260102",
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]

        applied, reopened, active_conflict, remaining_qty = apply_alpaca_closed_position_broker_correction(
            self.conn,
            position_id,
            expected_closed_at=str(snapshot["closed_at"]),
            expected_state_revision=int(snapshot["state_revision"]),
            alpaca_asset_id="asset-tqqq",
            buy_order_qty=2,
            buy_order_limit_price=105,
            buy_status="canceled",
            filled_qty=0,
            filled_avg_price=None,
            filled_at=None,
            target_sell_price=None,
            sell_status="canceled",
            sell_fills=[("sell-zero", 0, 0)],
            sell_filled_at=None,
            notes="buy and sell fills were both corrected to zero",
        )
        never_filled_id = save_alpaca_managed_buy_order(
            self.conn,
            workflow="rsi",
            symbol="SQQQ",
            alpaca_asset_id="asset-sqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-SQQQ-20260102",
            buy_alpaca_order_id="buy-never-filled",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="canceled",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.close_position(never_filled_id)
        submission_not_found_id = save_alpaca_managed_buy_order(
            self.conn,
            workflow="rsi",
            symbol="UPRO",
            alpaca_asset_id="asset-upro",
            signal_symbol="SPY",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-UPRO-20260102",
            buy_alpaca_order_id=None,
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="submission_not_found",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.close_position(submission_not_found_id)

        candidates = load_recently_closed_alpaca_managed_positions(self.conn)

        self.assertTrue(applied)
        self.assertFalse(reopened)
        self.assertFalse(active_conflict)
        self.assertEqual(remaining_qty, 0)
        self.assertEqual(candidates["id"].tolist(), [never_filled_id, position_id])
        self.assertNotIn(submission_not_found_id, candidates["id"].tolist())

    def test_tiny_fractional_buy_receives_protection_and_sell_accounting(self) -> None:
        quantity = 1e-9
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TINY",
            alpaca_asset_id="asset-tiny",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-tiny-protection",
            buy_alpaca_order_id="buy-tiny-protection",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=105,
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=quantity,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
            )
        )
        sell_client_order_id = "sell-tiny-protection"
        self.assertEqual(
            storage_module.claim_alpaca_managed_initial_sell_intent(
                self.conn,
                position_id,
                sell_order_namespace="tiny",
                sell_client_order_id=sell_client_order_id,
                expected_remaining_qty=quantity,
                expected_target_sell_price=150,
                notes="protect tiny fill",
            ),
            (quantity, 150.0),
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-tiny-order",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=quantity,
            sell_order_limit_price=150,
        )

        remaining_qty, observation_is_current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=quantity,
            sell_filled_avg_price=120,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_broker_updated_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id="sell-tiny-order",
        )

        self.assertTrue(observation_is_current)
        self.assertEqual(remaining_qty, 0.0)
        self.assertEqual(
            self.conn.execute(
                "SELECT filled_qty, filled_value FROM alpaca_managed_sell_fills "
                "WHERE managed_position_id = ? AND alpaca_order_id = 'sell-tiny-order'",
                (position_id,),
            ).fetchone(),
            (quantity, quantity * 120),
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT sold_qty, sold_value, remaining_qty, sell_status FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            (quantity, quantity * 120, 0.0, "filled"),
        )

    def test_high_price_tiny_quantity_remains_protected_until_value_is_closed(self) -> None:
        quantity = 1e-13
        buy_price = 1e15
        sell_price = 1.5e15
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="VALUETINY",
            alpaca_asset_id="asset-value-tiny",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-value-tiny",
            buy_alpaca_order_id="buy-value-tiny",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=buy_price,
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="canceled",
                filled_qty=quantity,
                filled_avg_price=buy_price,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=sell_price,
            )
        )
        sell_client_order_id = "sell-value-tiny"
        self.assertEqual(
            storage_module.claim_alpaca_managed_initial_sell_intent(
                self.conn,
                position_id,
                sell_order_namespace="value-tiny",
                sell_client_order_id=sell_client_order_id,
                expected_remaining_qty=quantity,
                expected_target_sell_price=sell_price,
                notes="protect economically material tiny quantity",
            ),
            (quantity, sell_price),
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id=sell_client_order_id,
            sell_alpaca_order_id="sell-value-tiny-order",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=quantity,
            sell_order_limit_price=sell_price,
        )
        remaining, current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="accepted",
            sell_filled_qty=0.0,
            sell_filled_avg_price=sell_price,
            sell_filled_at=None,
            sell_broker_updated_at="2026-01-02T14:32:30Z",
            sell_alpaca_order_id="sell-value-tiny-order",
        )
        self.assertTrue(current)
        self.assertEqual(remaining, quantity)
        self.assertFalse(
            close_alpaca_managed_position_if_current_and_complete(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                expected_sell_alpaca_order_id="sell-value-tiny-order",
                closed_at="2026-01-02T14:32:31Z",
            )
        )

        remaining, current = mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id=sell_client_order_id,
            sell_status="filled",
            sell_filled_qty=quantity,
            sell_filled_avg_price=sell_price,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_broker_updated_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id="sell-value-tiny-order",
        )
        self.assertTrue(current)
        self.assertEqual(remaining, 0.0)
        self.assertTrue(
            close_alpaca_managed_position_if_current_and_complete(
                self.conn,
                position_id,
                expected_sell_client_order_id=sell_client_order_id,
                expected_sell_alpaca_order_id="sell-value-tiny-order",
                closed_at="2026-01-02T14:34:00Z",
            )
        )
        summary = reports_module.build_alpaca_realized_pnl_summary(self.conn)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Total Buy Cost"], 100.0)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 150.0)

    def test_same_revision_buy_replay_rejects_a_material_tiny_quantity_change(
        self,
    ) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="TINYREPLAY",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-tiny-replay",
            buy_alpaca_order_id="buy-tiny-replay",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=105,
        )
        broker_revision = "2026-01-02T14:31:00Z"
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=1e-9,
                filled_avg_price=100,
                filled_at=broker_revision,
                target_sell_price=150,
                buy_fill_broker_updated_at=broker_revision,
                buy_fill_broker_oldest_updated_at=broker_revision,
                buy_alpaca_order_id="buy-tiny-replay",
            )
        )
        before = self.conn.execute(
            "SELECT state_revision, filled_qty, buy_fill_broker_updated_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        replay_is_current = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=9e-9,
            filled_avg_price=100,
            filled_at=broker_revision,
            target_sell_price=150,
            buy_fill_broker_updated_at=broker_revision,
            buy_fill_broker_oldest_updated_at=broker_revision,
            buy_alpaca_order_id="buy-tiny-replay",
            expected_buy_status="partially_filled",
            expected_buy_alpaca_order_id="buy-tiny-replay",
            expected_filled_qty=1e-9,
            expected_filled_avg_price=100,
            expected_target_sell_price=150,
            expected_sell_client_order_id=None,
        )

        self.assertFalse(replay_is_current)
        self.assertEqual(
            self.conn.execute(
                "SELECT state_revision, filled_qty, buy_fill_broker_updated_at "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )

    def test_same_revision_buy_replay_rejects_material_scaled_quantity_change(
        self,
    ) -> None:
        buy_price = 1e15
        target_price = 1.5e15
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="SCALEDREPLAY",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-scaled-replay",
            buy_alpaca_order_id="buy-scaled-replay",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=buy_price,
        )
        broker_revision = "2026-01-02T14:31:00Z"
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=1,
                filled_avg_price=buy_price,
                filled_at=broker_revision,
                target_sell_price=target_price,
                buy_fill_broker_updated_at=broker_revision,
                buy_fill_broker_oldest_updated_at=broker_revision,
                buy_alpaca_order_id="buy-scaled-replay",
            )
        )
        before = self.conn.execute(
            "SELECT state_revision, filled_qty, buy_fill_broker_updated_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        replay_is_current = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=0.999999999,
            filled_avg_price=buy_price,
            filled_at=broker_revision,
            target_sell_price=target_price,
            buy_fill_broker_updated_at=broker_revision,
            buy_fill_broker_oldest_updated_at=broker_revision,
            buy_alpaca_order_id="buy-scaled-replay",
            expected_buy_status="partially_filled",
            expected_buy_alpaca_order_id="buy-scaled-replay",
            expected_filled_qty=1,
            expected_filled_avg_price=buy_price,
            expected_target_sell_price=target_price,
            expected_sell_client_order_id=None,
        )

        self.assertFalse(replay_is_current)
        self.assertEqual(
            self.conn.execute(
                "SELECT state_revision, filled_qty, buy_fill_broker_updated_at "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )

    def test_same_revision_component_buy_replay_rejects_material_scaled_quantity_change(
        self,
    ) -> None:
        buy_price = 1e15
        target_price = 1.5e15
        order_id = "buy-scaled-component-replay"
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="SCALEDCOMPONENT",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id=order_id,
            buy_alpaca_order_id=order_id,
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=buy_price,
        )
        broker_revision = "2026-01-02T14:31:00Z"
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=1,
                filled_avg_price=buy_price,
                filled_at=broker_revision,
                target_sell_price=target_price,
                buy_fill_broker_updated_at=broker_revision,
                buy_fill_broker_oldest_updated_at=broker_revision,
                buy_fill_component_revisions={
                    order_id: {
                        "broker_updated_at": broker_revision,
                        "filled_qty": 1,
                        "filled_value": buy_price,
                    }
                },
                buy_alpaca_order_id=order_id,
            )
        )
        before = self.conn.execute(
            "SELECT state_revision, filled_qty, buy_fill_broker_updated_at, "
            "buy_fill_component_revisions, buy_fill_pending_observation "
            "FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        corrected_qty = 0.999999999

        replay_is_current = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=corrected_qty,
            filled_avg_price=buy_price,
            filled_at=broker_revision,
            target_sell_price=target_price,
            buy_fill_broker_updated_at=broker_revision,
            buy_fill_broker_oldest_updated_at=broker_revision,
            buy_fill_component_revisions={
                order_id: {
                    "broker_updated_at": broker_revision,
                    "filled_qty": corrected_qty,
                    "filled_value": corrected_qty * buy_price,
                }
            },
            buy_alpaca_order_id=order_id,
            expected_buy_status="partially_filled",
            expected_buy_alpaca_order_id=order_id,
            expected_filled_qty=1,
            expected_filled_avg_price=buy_price,
            expected_target_sell_price=target_price,
            expected_sell_client_order_id=None,
        )

        self.assertFalse(replay_is_current)
        self.assertEqual(
            self.conn.execute(
                "SELECT state_revision, filled_qty, buy_fill_broker_updated_at, "
                "buy_fill_component_revisions, buy_fill_pending_observation "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )

    def test_same_revision_buy_replay_accepts_true_quantity_roundoff(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="ROUNDREPLAY",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-round-replay",
            buy_alpaca_order_id="buy-round-replay",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        broker_revision = "2026-01-02T14:31:00Z"
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=1,
                filled_avg_price=100,
                filled_at=broker_revision,
                target_sell_price=150,
                buy_fill_broker_updated_at=broker_revision,
                buy_fill_broker_oldest_updated_at=broker_revision,
                buy_alpaca_order_id="buy-round-replay",
            )
        )
        before = self.conn.execute(
            "SELECT state_revision, filled_qty, buy_fill_broker_updated_at FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        replay_result = mark_alpaca_managed_buy_filled(
            self.conn,
            position_id,
            buy_status="partially_filled",
            filled_qty=float(np.nextafter(1.0, 0.0)),
            filled_avg_price=100,
            filled_at=broker_revision,
            target_sell_price=150,
            buy_fill_broker_updated_at=broker_revision,
            buy_fill_broker_oldest_updated_at=broker_revision,
            buy_alpaca_order_id="buy-round-replay",
            expected_buy_status="partially_filled",
            expected_buy_alpaca_order_id="buy-round-replay",
            expected_filled_qty=1,
            expected_filled_avg_price=100,
            expected_target_sell_price=150,
            expected_sell_client_order_id=None,
            return_state_revision=True,
        )

        self.assertEqual(replay_result, (True, before[0]))
        self.assertEqual(
            self.conn.execute(
                "SELECT state_revision, filled_qty, buy_fill_broker_updated_at "
                "FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )

    def test_newer_buy_revision_accepts_share_bounded_scaled_correction(self) -> None:
        buy_price = 1e15
        target_price = 1.5e15
        first_revision = "2026-01-02T14:31:00Z"
        second_revision = "2026-01-02T14:32:00Z"
        corrected_qty = 0.999999999
        for with_components in (False, True):
            with self.subTest(with_components=with_components):
                suffix = "components" if with_components else "scalar"
                order_id = f"buy-scaled-newer-{suffix}"
                position_id = save_alpaca_managed_buy_order(
                    self.conn,
                    symbol=f"SCALEDNEWER{suffix.upper()}",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id=order_id,
                    buy_alpaca_order_id=order_id,
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                    buy_order_qty=2,
                    buy_order_limit_price=buy_price,
                )
                first_components = (
                    {
                        "buy_fill_component_revisions": {
                            order_id: {
                                "broker_updated_at": first_revision,
                                "filled_qty": 1,
                                "filled_value": buy_price,
                            }
                        }
                    }
                    if with_components
                    else {}
                )
                self.assertTrue(
                    mark_alpaca_managed_buy_filled(
                        self.conn,
                        position_id,
                        buy_status="partially_filled",
                        filled_qty=1,
                        filled_avg_price=buy_price,
                        filled_at=first_revision,
                        target_sell_price=target_price,
                        buy_fill_broker_updated_at=first_revision,
                        buy_fill_broker_oldest_updated_at=first_revision,
                        buy_alpaca_order_id=order_id,
                        **first_components,
                    )
                )
                corrected_components = (
                    {
                        "buy_fill_component_revisions": {
                            order_id: {
                                "broker_updated_at": second_revision,
                                "filled_qty": corrected_qty,
                                "filled_value": corrected_qty * buy_price,
                            }
                        }
                    }
                    if with_components
                    else {}
                )

                corrected = mark_alpaca_managed_buy_filled(
                    self.conn,
                    position_id,
                    buy_status="partially_filled",
                    filled_qty=corrected_qty,
                    filled_avg_price=buy_price,
                    filled_at=first_revision,
                    target_sell_price=target_price,
                    buy_fill_broker_updated_at=second_revision,
                    buy_fill_broker_oldest_updated_at=second_revision,
                    buy_alpaca_order_id=order_id,
                    expected_buy_status="partially_filled",
                    expected_buy_alpaca_order_id=order_id,
                    expected_filled_qty=1,
                    expected_filled_avg_price=buy_price,
                    expected_target_sell_price=target_price,
                    expected_sell_client_order_id=None,
                    **corrected_components,
                )
                row = self.conn.execute(
                    "SELECT filled_qty, buy_fill_broker_updated_at, "
                    "buy_fill_component_revisions "
                    "FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()

                self.assertTrue(corrected)
                self.assertEqual(row[0], corrected_qty)
                self.assertEqual(row[1], "2026-01-02T14:32:00.000000Z")
                if with_components:
                    self.assertIn('"filled_qty":0.999999999', row[2])
                else:
                    self.assertIsNone(row[2])

    def test_buy_correction_rejects_nonfinite_derived_realized_pl_atomically(
        self,
    ) -> None:
        quantity = 1e-6
        first_revision = "2026-01-02T14:31:00Z"
        second_revision = "2026-01-02T14:34:00Z"
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="PNLFINITE",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-finite-derived-pnl",
            buy_alpaca_order_id="buy-finite-derived-pnl",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=1,
            buy_order_limit_price=2,
        )
        self.assertTrue(
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=quantity,
                filled_avg_price=1,
                filled_at=first_revision,
                target_sell_price=1.5,
                buy_fill_broker_updated_at=first_revision,
                buy_fill_broker_oldest_updated_at=first_revision,
                buy_fill_component_revisions={
                    "buy-finite-derived-pnl": {
                        "broker_updated_at": first_revision,
                        "filled_qty": quantity,
                        "filled_value": quantity,
                    }
                },
                buy_alpaca_order_id="buy-finite-derived-pnl",
            )
        )
        record_alpaca_managed_sell_order(
            self.conn,
            position_id,
            sell_client_order_id="sell-finite-derived-pnl",
            sell_alpaca_order_id="sell-finite-derived-pnl",
            sell_submitted_at="2026-01-02T14:32:00Z",
            sell_status="accepted",
            sell_order_qty=quantity,
            sell_order_limit_price=1e6,
        )
        mark_alpaca_managed_sell_filled_if_current(
            self.conn,
            position_id,
            expected_sell_client_order_id="sell-finite-derived-pnl",
            sell_status="filled",
            sell_filled_qty=quantity,
            sell_filled_avg_price=1e6,
            sell_filled_at="2026-01-02T14:33:00Z",
            sell_broker_updated_at="2026-01-02T14:33:00Z",
            sell_alpaca_order_id="sell-finite-derived-pnl",
        )
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "realized P/L must remain finite"):
            mark_alpaca_managed_buy_filled(
                self.conn,
                position_id,
                buy_status="partially_filled",
                filled_qty=quantity,
                filled_avg_price=1e-302,
                filled_at=first_revision,
                target_sell_price=0.0001,
                buy_fill_broker_updated_at=second_revision,
                buy_fill_broker_oldest_updated_at=second_revision,
                buy_fill_component_revisions={
                    "buy-finite-derived-pnl": {
                        "broker_updated_at": second_revision,
                        "filled_qty": quantity,
                        "filled_value": 1e-308,
                    }
                },
                buy_alpaca_order_id="buy-finite-derived-pnl",
                expected_buy_status="partially_filled",
                expected_buy_alpaca_order_id="buy-finite-derived-pnl",
                expected_filled_qty=quantity,
                expected_filled_avg_price=1,
                expected_target_sell_price=1.5,
                expected_sell_client_order_id="sell-finite-derived-pnl",
            )

        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )

    def test_closed_correction_rejects_invalid_sell_economics_and_ids_atomically(
        self,
    ) -> None:
        position_id = self.save_position(
            symbol="CORRECTIONID",
            client_order_id="buy-correction-id",
        )
        self.close_position(position_id)
        snapshot = load_alpaca_managed_positions(self.conn).iloc[0]
        before_parent = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        invalid_fills = (
            [("sell-zero-value", 2, 0)],
            [(" sell-invalid", 2, 300)],
            [("sell-duplicate", 1, 150), ("sell-duplicate", 1, 150)],
        )
        for sell_fills in invalid_fills:
            with self.subTest(sell_fills=sell_fills), self.assertRaises(ValueError):
                apply_alpaca_closed_position_broker_correction(
                    self.conn,
                    position_id,
                    expected_closed_at=str(snapshot["closed_at"]),
                    expected_state_revision=int(snapshot["state_revision"]),
                    alpaca_asset_id="asset-tqqq",
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                    buy_status="filled",
                    filled_qty=2,
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                    sell_status="filled",
                    sell_fills=sell_fills,
                    sell_filled_at="2026-01-02T14:33:00Z",
                    notes="invalid correction",
                )
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone(),
                before_parent,
            )
            self.assertEqual(
                self.conn.execute(
                    "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                    (position_id,),
                ).fetchall(),
                [],
            )

        with self.assertRaisesRegex(ValueError, "Alpaca price tick"):
            apply_alpaca_closed_position_broker_correction(
                self.conn,
                position_id,
                expected_closed_at=str(snapshot["closed_at"]),
                expected_state_revision=int(snapshot["state_revision"]),
                alpaca_asset_id="asset-tqqq",
                buy_order_qty=2,
                buy_order_limit_price=105,
                buy_status="filled",
                filled_qty=2,
                filled_avg_price=100,
                filled_at="2026-01-02T14:31:00Z",
                target_sell_price=150,
                sell_status="filled",
                sell_fills=[],
                sell_order_intents={"sell-off-tick-correction": (2, 150.001)},
                sell_filled_at="2026-01-02T14:33:00Z",
                notes="off-tick correction must fail",
            )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before_parent,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchall(),
            [],
        )

    def test_public_sell_fill_rejects_noncanonical_order_id_without_writes(self) -> None:
        position_id = self.save_position(
            symbol="SELLID",
            client_order_id="buy-public-sell-id",
        )
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (position_id,),
        ).fetchone()

        with self.assertRaisesRegex(ValueError, "canonical broker order ID"):
            mark_alpaca_managed_sell_filled_if_current(
                self.conn,
                position_id,
                expected_sell_client_order_id="sell-public-id",
                sell_status="filled",
                sell_filled_qty=2,
                sell_filled_avg_price=120,
                sell_filled_at="2026-01-02T14:33:00Z",
                sell_broker_updated_at="2026-01-02T14:33:00Z",
                sell_alpaca_order_id=" sell-public-id",
            )

        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (position_id,),
            ).fetchone(),
            before,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_sell_fills WHERE managed_position_id = ?",
                (position_id,),
            ).fetchall(),
            [],
        )

    def test_managed_position_ids_and_symbol_migration_identities_are_canonical(
        self,
    ) -> None:
        position_id = self.save_position(
            symbol="IDENTITY",
            client_order_id="buy-positive-managed-id",
        )
        self.assertGreater(position_id, 0)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "UPDATE alpaca_managed_positions SET id = 0 WHERE id = ?",
                (position_id,),
            )
        self.conn.rollback()

        for asset_id, current_symbol in (
            ("asset_invalid", "IDENTITY"),
            ("asset-identity", "identity"),
            ("asset-identity", " IDENTITY"),
            ("\u00a0asset-identity\u00a0", "IDENTITY"),
            ("asset-identity", "ſ"),
            ("asset-identity", "ß"),
            ("asset-identity", "straße"),
            ("asset-identity", "\u00a0IDENTITY\u00a0"),
        ):
            with self.subTest(asset_id=asset_id, current_symbol=current_symbol), self.assertRaises(ValueError):
                storage_module.migrate_alpaca_managed_position_symbol(
                    self.conn,
                    position_id,
                    alpaca_asset_id=asset_id,
                    current_symbol=current_symbol,
                )
        for prior_symbol in (" invalid prior ", "identity", "ſ", "ß", "straße", "\u00a0IDENTITY\u00a0"):
            with self.subTest(prior_symbol=prior_symbol):
                self.conn.execute(
                    "UPDATE alpaca_managed_positions SET symbol = ? WHERE id = ?",
                    (prior_symbol, position_id),
                )
                self.conn.commit()
                with self.assertRaisesRegex(ValueError, "prior symbol"):
                    storage_module.migrate_alpaca_managed_position_symbol(
                        self.conn,
                        position_id,
                        alpaca_asset_id="asset-identity",
                        current_symbol="IDENTITY",
                    )
                self.assertEqual(
                    self.conn.execute("SELECT * FROM alpaca_symbol_aliases").fetchall(),
                    [],
                )

        positions_sql = storage_module._canonical_storage_table_contracts()["alpaca_managed_positions"][0].replace(
            " CHECK (id > 0)", ""
        )
        with closing(sqlite3.connect(":memory:")) as imported:
            imported.execute(positions_sql)
            imported.execute(
                """
                INSERT INTO alpaca_managed_positions
                (id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status)
                VALUES (0, 'TQQQ', 'QQQ', 30, 1.5, '2026-01-02',
                        'buy-invalid-managed-id', 'accepted')
                """
            )
            imported.commit()
            with self.assertRaisesRegex(ValueError, "positive managed identities"):
                init_state_db(imported)
            self.assertEqual(
                imported.execute("SELECT id, buy_client_order_id FROM alpaca_managed_positions").fetchall(),
                [(0, "buy-invalid-managed-id")],
            )

    def test_managed_owner_ingress_normalizes_and_rejects_invalid_identities_atomically(
        self,
    ) -> None:
        claim = storage_module.claim_alpaca_managed_buy_intent(
            self.conn,
            symbol=" tqqq ",
            alpaca_asset_id=" asset-tqqq ",
            signal_symbol=" qqq ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-canonical-owner",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.assertTrue(claim.claimed)
        self.assertEqual(
            self.conn.execute(
                "SELECT symbol, alpaca_asset_id, signal_symbol FROM alpaca_managed_positions WHERE id = ?",
                (claim.position_id,),
            ).fetchone(),
            ("TQQQ", "asset-tqqq", "QQQ"),
        )
        duplicate = storage_module.claim_alpaca_managed_buy_intent(
            self.conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-other",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-03",
            buy_client_order_id="buy-canonical-owner-duplicate",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.assertFalse(duplicate.claimed)
        self.assertTrue(duplicate.symbol_conflict)
        before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (claim.position_id,),
        ).fetchone()

        invalid_operations = (
            lambda: save_alpaca_managed_buy_order(
                self.conn,
                symbol="BAD SYMBOL",
                alpaca_asset_id="asset-valid",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="buy-invalid-owner-save",
                buy_alpaca_order_id=None,
                buy_submitted_at=None,
                buy_status="accepted",
            ),
            lambda: save_alpaca_managed_buy_order(
                self.conn,
                symbol="VALID",
                alpaca_asset_id="asset-valid",
                signal_symbol="BAD SIGNAL",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="buy-invalid-signal-save",
                buy_alpaca_order_id=None,
                buy_submitted_at=None,
                buy_status="accepted",
            ),
            lambda: storage_module.adopt_alpaca_managed_buy_order_if_submission_not_found(
                self.conn,
                workflow=None,
                symbol="TQQQ",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-02",
                buy_client_order_id="buy-canonical-owner",
                buy_alpaca_order_id="buy-invalid-recovery",
                buy_submitted_at=None,
                buy_status="accepted",
                buy_order_qty=2,
                buy_order_limit_price=105,
                alpaca_asset_id="bad asset",
            ),
            lambda: storage_module.adopt_alpaca_managed_position_asset_if_current(
                self.conn,
                int(claim.position_id),
                expected_state_revision=0,
                alpaca_asset_id="bad asset",
            ),
            lambda: storage_module.quarantine_alpaca_managed_sell_stale_submission(
                self.conn,
                int(claim.position_id),
                sell_client_order_id="sell-invalid-owner",
                expected_sell_submission_retry_claimed_at="2026-01-02T14:30:00Z",
                stale_alpaca_order_ids=[],
                notes="invalid identity must fail before quarantine",
                observed_alpaca_asset_id="bad asset",
            ),
            lambda: mark_alpaca_managed_sell_filled_if_current(
                self.conn,
                int(claim.position_id),
                expected_sell_client_order_id="sell-invalid-owner",
                sell_status="filled",
                sell_filled_qty=1,
                sell_filled_avg_price=120,
                sell_filled_at="2026-01-02T14:33:00Z",
                sell_alpaca_order_id="sell-invalid-owner",
                observed_alpaca_asset_id="bad asset",
            ),
        )
        for operation in invalid_operations:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (claim.position_id,),
            ).fetchone(),
            before,
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM alpaca_managed_sell_fills").fetchone()[0],
            0,
        )

    def test_managed_owner_ingress_rejects_unicode_identity_folding_atomically(self) -> None:
        class ManagedIdentitySubclass(str):
            pass

        for index, canonical_symbol in enumerate(("S", "SS", "STRASSE", "TQQQ")):
            claim = storage_module.claim_alpaca_managed_buy_intent(
                self.conn,
                symbol=canonical_symbol,
                alpaca_asset_id=f"asset-canonical-{index}",
                signal_symbol="QQQ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date=f"2026-01-{index + 2:02d}",
                buy_client_order_id=f"buy-canonical-unicode-collision-{index}",
                buy_order_qty=2,
                buy_order_limit_price=105,
            )
            self.assertTrue(claim.claimed)

        saved_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol=" save ",
            alpaca_asset_id=" asset-save ",
            signal_symbol=" qqq ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-10",
            buy_client_order_id="buy-safe-ascii-save-normalization",
            buy_alpaca_order_id=None,
            buy_submitted_at=None,
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        recovery_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="RECOVER",
            alpaca_asset_id=None,
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-11",
            buy_client_order_id="buy-safe-ascii-recovery-normalization",
            buy_alpaca_order_id=None,
            buy_submitted_at=None,
            buy_status="submission_not_found",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET closed_at = '2026-01-12T00:00:00Z' WHERE id = ?",
            (recovery_id,),
        )
        self.conn.commit()
        self.assertTrue(
            storage_module.adopt_alpaca_managed_buy_order_if_submission_not_found(
                self.conn,
                workflow=None,
                symbol=" recover ",
                signal_symbol=" qqq ",
                buy_rsi=30,
                profit_target_multiple=1.5,
                buy_signal_date="2026-01-11",
                buy_client_order_id="buy-safe-ascii-recovery-normalization",
                buy_alpaca_order_id="buy-safe-ascii-recovery-order",
                buy_submitted_at="2026-01-12T14:30:00Z",
                buy_status="accepted",
                buy_order_qty=2,
                buy_order_limit_price=105,
                alpaca_asset_id=" asset-recover ",
            )
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT symbol, alpaca_asset_id, signal_symbol FROM alpaca_managed_positions WHERE id = ?",
                (saved_id,),
            ).fetchone(),
            ("SAVE", "asset-save", "QQQ"),
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT symbol, alpaca_asset_id, signal_symbol FROM alpaca_managed_positions WHERE id = ?",
                (recovery_id,),
            ).fetchone(),
            ("RECOVER", "asset-recover", "QQQ"),
        )

        positions_before = self.conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall()
        aliases_before = self.conn.execute(
            "SELECT * FROM alpaca_symbol_aliases ORDER BY alpaca_asset_id, symbol"
        ).fetchall()
        invalid_symbols = (
            "ſ",
            "ß",
            "straße",
            "\u00a0tqqq\u00a0",
            "\ttqqq\t",
            "\ntqqq\n",
            ManagedIdentitySubclass("TQQQ"),
        )
        for index, invalid_symbol in enumerate(invalid_symbols):
            with self.subTest(field="symbol", value=invalid_symbol), self.assertRaises(ValueError):
                storage_module.claim_alpaca_managed_buy_intent(
                    self.conn,
                    symbol=invalid_symbol,
                    alpaca_asset_id=f"asset-invalid-symbol-{index}",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-02-01",
                    buy_client_order_id=f"buy-invalid-unicode-symbol-{index}",
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                )
            self.assertFalse(self.conn.in_transaction)

        for index, invalid_signal_symbol in enumerate(invalid_symbols):
            with self.subTest(field="signal_symbol", value=invalid_signal_symbol), self.assertRaises(ValueError):
                save_alpaca_managed_buy_order(
                    self.conn,
                    symbol=f"SIG-{index}",
                    alpaca_asset_id=f"asset-invalid-signal-{index}",
                    signal_symbol=invalid_signal_symbol,
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-02-02",
                    buy_client_order_id=f"buy-invalid-unicode-signal-{index}",
                    buy_alpaca_order_id=None,
                    buy_submitted_at=None,
                    buy_status="accepted",
                )
            self.assertFalse(self.conn.in_transaction)

        for index, invalid_asset_id in enumerate(
            (
                "\u00a0asset-tqqq\u00a0",
                "\tasset-tqqq\t",
                "\nasset-tqqq\n",
                "asset-ſ",
                ManagedIdentitySubclass("asset-tqqq"),
            )
        ):
            with self.subTest(field="alpaca_asset_id", value=invalid_asset_id), self.assertRaises(ValueError):
                storage_module.adopt_alpaca_managed_buy_order_if_submission_not_found(
                    self.conn,
                    workflow=None,
                    symbol="RECOVER",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-11",
                    buy_client_order_id="buy-safe-ascii-recovery-normalization",
                    buy_alpaca_order_id=f"buy-invalid-unicode-asset-{index}",
                    buy_submitted_at=None,
                    buy_status="accepted",
                    buy_order_qty=2,
                    buy_order_limit_price=105,
                    alpaca_asset_id=invalid_asset_id,
                )
            self.assertFalse(self.conn.in_transaction)

        self.assertEqual(
            self.conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall(),
            positions_before,
        )
        self.assertEqual(
            self.conn.execute("SELECT * FROM alpaca_symbol_aliases ORDER BY alpaca_asset_id, symbol").fetchall(),
            aliases_before,
        )

    def test_persisted_managed_owner_confusables_are_not_rehabilitated(self) -> None:
        position_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="S",
            alpaca_asset_id="asset-s",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="buy-persisted-confusable-owner",
            buy_alpaca_order_id="buy-persisted-confusable-order",
            buy_submitted_at="2026-01-02T14:30:00Z",
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )

        for column, corrupted_value in (
            ("symbol", "ſ"),
            ("signal_symbol", "ß"),
            ("alpaca_asset_id", "\u00a0asset-s\u00a0"),
        ):
            with self.subTest(column=column, value=corrupted_value):
                self.conn.execute(
                    f"UPDATE alpaca_managed_positions SET {column} = ? WHERE id = ?",
                    (corrupted_value, position_id),
                )
                self.conn.commit()
                corrupted_before = self.conn.execute(
                    "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()

                with self.assertRaises(ValueError):
                    storage_module.alpaca_managed_buy_order_observation_issue(
                        self.conn,
                        workflow=None,
                        symbol="s",
                        signal_symbol="qqq",
                        buy_rsi=30,
                        profit_target_multiple=1.5,
                        buy_signal_date="2026-01-02",
                        buy_client_order_id="buy-persisted-confusable-owner",
                        buy_alpaca_order_id="buy-persisted-confusable-order",
                        buy_order_qty=2,
                        buy_order_limit_price=105,
                        alpaca_asset_id="asset-s",
                    )
                with self.assertRaises(ValueError):
                    save_alpaca_managed_buy_order(
                        self.conn,
                        symbol="s",
                        alpaca_asset_id="asset-s",
                        signal_symbol="qqq",
                        buy_rsi=30,
                        profit_target_multiple=1.5,
                        buy_signal_date="2026-01-02",
                        buy_client_order_id="buy-persisted-confusable-owner",
                        buy_alpaca_order_id="buy-persisted-confusable-order",
                        buy_submitted_at="2026-01-02T14:30:00Z",
                        buy_status="accepted",
                        buy_order_qty=2,
                        buy_order_limit_price=105,
                    )
                if column != "alpaca_asset_id":
                    with self.assertRaises(ValueError):
                        storage_module.claim_alpaca_managed_buy_intent(
                            self.conn,
                            symbol="s",
                            alpaca_asset_id="asset-s",
                            signal_symbol="qqq",
                            buy_rsi=30,
                            profit_target_multiple=1.5,
                            buy_signal_date="2026-01-02",
                            buy_client_order_id="buy-persisted-confusable-owner",
                            buy_order_qty=2,
                            buy_order_limit_price=105,
                        )
                self.assertEqual(
                    self.conn.execute(
                        "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                        (position_id,),
                    ).fetchone(),
                    corrupted_before,
                )
                self.assertFalse(self.conn.in_transaction)
                canonical_value = {
                    "symbol": "S",
                    "signal_symbol": "QQQ",
                    "alpaca_asset_id": "asset-s",
                }[column]
                self.conn.execute(
                    f"UPDATE alpaca_managed_positions SET {column} = ? WHERE id = ?",
                    (canonical_value, position_id),
                )
                self.conn.commit()

        adopt_id = save_alpaca_managed_buy_order(
            self.conn,
            symbol="ADOPT",
            alpaca_asset_id=None,
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-03",
            buy_client_order_id="buy-persisted-confusable-adoption",
            buy_alpaca_order_id=None,
            buy_submitted_at=None,
            buy_status="accepted",
            buy_order_qty=2,
            buy_order_limit_price=105,
        )
        self.conn.execute(
            "UPDATE alpaca_managed_positions SET symbol = 'ſ' WHERE id = ?",
            (adopt_id,),
        )
        self.conn.commit()
        adoption_before = self.conn.execute(
            "SELECT * FROM alpaca_managed_positions WHERE id = ?",
            (adopt_id,),
        ).fetchone()
        aliases_before = self.conn.execute("SELECT * FROM alpaca_symbol_aliases").fetchall()

        with self.assertRaises(ValueError):
            storage_module.adopt_alpaca_managed_position_asset_if_current(
                self.conn,
                adopt_id,
                expected_state_revision=int(adoption_before[1]),
                alpaca_asset_id="asset-adopt",
            )
        self.assertEqual(
            self.conn.execute(
                "SELECT * FROM alpaca_managed_positions WHERE id = ?",
                (adopt_id,),
            ).fetchone(),
            adoption_before,
        )
        self.assertEqual(self.conn.execute("SELECT * FROM alpaca_symbol_aliases").fetchall(), aliases_before)
        with self.assertRaises(ValueError):
            storage_module.active_alpaca_managed_symbols(self.conn)
        with self.assertRaises(ValueError):
            storage_module.alpaca_managed_position_aliases(self.conn, adopt_id)

        self.conn.execute(
            "UPDATE alpaca_managed_positions SET symbol = 'ADOPT' WHERE id = ?",
            (adopt_id,),
        )
        self.conn.commit()
        restored_revision = int(
            self.conn.execute(
                "SELECT state_revision FROM alpaca_managed_positions WHERE id = ?",
                (adopt_id,),
            ).fetchone()[0]
        )
        self.assertIsNotNone(
            storage_module.adopt_alpaca_managed_position_asset_if_current(
                self.conn,
                adopt_id,
                expected_state_revision=restored_revision,
                alpaca_asset_id="asset-adopt",
            )
        )
        self.conn.execute("INSERT INTO alpaca_symbol_aliases (alpaca_asset_id, symbol) VALUES ('asset-adopt', 'ſ')")
        self.conn.commit()
        with self.assertRaises(ValueError):
            storage_module.active_alpaca_managed_symbols(self.conn)
        with self.assertRaises(ValueError):
            storage_module.alpaca_managed_position_aliases(self.conn, adopt_id)

    def test_lookup_fetch_failures_rollback_owned_writes_and_preserve_caller_work(
        self,
    ) -> None:
        operation_names = ("save_buy", "buy_status_replay", "sell_status_replay")
        for operation_name in operation_names:
            for caller_owns_transaction in (False, True):
                with (
                    self.subTest(
                        operation=operation_name,
                        caller_owns_transaction=caller_owns_transaction,
                    ),
                    closing(
                        sqlite3.connect(
                            ":memory:",
                            factory=SelectedFetchFailureConnection,
                        )
                    ) as conn,
                ):
                    init_state_db(conn)
                    conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
                    conn.commit()
                    position_id: int | None = None

                    if operation_name == "save_buy":
                        conn.selected_sql_fragment = (
                            "SELECT id, symbol, signal_symbol, alpaca_asset_id "
                            "FROM alpaca_managed_positions WHERE buy_client_order_id = ?"
                        )

                        def invoke() -> object:
                            return save_alpaca_managed_buy_order(
                                conn,
                                symbol="FETCHSAVE",
                                signal_symbol="QQQ",
                                buy_rsi=30,
                                profit_target_multiple=1.5,
                                buy_signal_date="2026-01-02",
                                buy_client_order_id="buy-fetch-save",
                                buy_alpaca_order_id="buy-fetch-save",
                                buy_submitted_at="2026-01-02T14:30:00Z",
                                buy_status="accepted",
                                buy_order_qty=1,
                                buy_order_limit_price=100,
                            )

                    else:
                        position_id = save_alpaca_managed_buy_order(
                            conn,
                            symbol="FETCHSTATUS",
                            signal_symbol="QQQ",
                            buy_rsi=30,
                            profit_target_multiple=1.5,
                            buy_signal_date="2026-01-02",
                            buy_client_order_id=f"buy-fetch-{operation_name}",
                            buy_alpaca_order_id=f"buy-fetch-{operation_name}",
                            buy_submitted_at="2026-01-02T14:30:00Z",
                            buy_status="accepted",
                            buy_order_qty=1,
                            buy_order_limit_price=100,
                        )
                        if operation_name == "buy_status_replay":
                            common = {
                                "expected_buy_status": "accepted",
                                "expected_buy_alpaca_order_id": "buy-fetch-buy_status_replay",
                                "expected_filled_qty": None,
                                "expected_sell_client_order_id": None,
                                "buy_status": "accepted",
                                "buy_alpaca_order_id": "buy-fetch-buy_status_replay",
                                "buy_broker_updated_at": "2026-01-02T14:31:00Z",
                            }
                            self.assertTrue(
                                update_alpaca_managed_buy_status_if_current(
                                    conn,
                                    position_id,
                                    **common,
                                )
                            )
                            conn.selected_sql_fragment = (
                                "SELECT 1 FROM alpaca_managed_positions WHERE id = ? "
                                "AND closed_at IS NULL AND buy_status = ?"
                            )

                            def invoke(
                                position_id: int = position_id,
                                common: dict[str, object] = common,
                            ) -> object:
                                return update_alpaca_managed_buy_status_if_current(
                                    conn,
                                    position_id,
                                    **common,
                                )

                        else:
                            mark_alpaca_managed_buy_filled(
                                conn,
                                position_id,
                                buy_status="filled",
                                filled_qty=1,
                                filled_avg_price=100,
                                filled_at="2026-01-02T14:31:00Z",
                                target_sell_price=150,
                            )
                            record_alpaca_managed_sell_order(
                                conn,
                                position_id,
                                sell_client_order_id="sell-fetch-status",
                                sell_alpaca_order_id="sell-fetch-status",
                                sell_submitted_at="2026-01-02T14:32:00Z",
                                sell_status="accepted",
                                sell_order_qty=1,
                                sell_order_limit_price=150,
                            )
                            common = {
                                "expected_sell_client_order_id": "sell-fetch-status",
                                "sell_status": "accepted",
                                "sell_alpaca_order_id": "sell-fetch-status",
                                "sell_broker_updated_at": "2026-01-02T14:33:00Z",
                            }
                            self.assertTrue(
                                update_alpaca_managed_sell_status_if_current(
                                    conn,
                                    position_id,
                                    **common,
                                )
                            )
                            conn.selected_sql_fragment = (
                                "SELECT 1 FROM alpaca_managed_positions WHERE id = ? "
                                "AND closed_at IS NULL AND sell_client_order_id IS ?"
                            )

                            def invoke(
                                position_id: int = position_id,
                                common: dict[str, object] = common,
                            ) -> object:
                                return update_alpaca_managed_sell_status_if_current(
                                    conn,
                                    position_id,
                                    **common,
                                )

                    positions_before = conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall()
                    if caller_owns_transaction:
                        conn.execute("INSERT INTO transaction_probe VALUES (1)")
                    conn.fail_selected_fetch = True

                    with self.assertRaisesRegex(
                        ValueError,
                        "forced selected row decode failure",
                    ):
                        invoke()

                    self.assertFalse(conn.fail_selected_fetch)
                    self.assertEqual(conn.in_transaction, caller_owns_transaction)
                    self.assertEqual(
                        conn.execute("SELECT * FROM alpaca_managed_positions ORDER BY id").fetchall(),
                        positions_before,
                    )
                    self.assertEqual(
                        conn.execute("SELECT * FROM transaction_probe").fetchall(),
                        [(1,)] if caller_owns_transaction else [],
                    )
                    if caller_owns_transaction:
                        conn.rollback()


if __name__ == "__main__":
    unittest.main()
