from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import leveraged_trader.reports as reports_module
from leveraged_trader.config import BacktestConfig
from leveraged_trader.reports import (
    build_alpaca_realized_pnl_summary,
)
from leveraged_trader.storage import (
    SummaryRollup,
    _best_summary_config,
    _update_summary_rollup,
    close_alpaca_managed_position_if_current_and_complete,
    ensure_rsi_values,
    init_state_db,
    load_best_strategy_summary,
    load_complete_strategy_equity_curve,
    load_strategy_state,
    save_equity_records,
    save_market_data,
    save_rsi_values,
    save_strategy_state,
    save_strategy_summary,
)


def build_buy_signal_report(
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
    return reports_module.build_buy_signal_report(
        conn,
        optimization_summary,
        rsi_period,
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


class SnapshotRollbackFailureConnection(sqlite3.Connection):
    """Inject one failure while ending an owned report read snapshot."""

    fail_next_rollback = False
    rollback_failure_observed = False

    def rollback(self) -> None:
        if self.fail_next_rollback:
            self.fail_next_rollback = False
            self.rollback_failure_observed = True
            raise sqlite3.OperationalError("forced snapshot rollback failure")
        super().rollback()


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        init_state_db(self.conn)
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "^IRX_Open": [0.0],
                    "^IRX_High": [0.0],
                    "^IRX_Low": [0.0],
                    "^IRX_Close": [0.0],
                    "^IRX_Volume": [0.0],
                },
                index=pd.to_datetime(["1900-01-01"]),
            ),
            ["^IRX"],
        )
        save_rsi_values(
            self.conn,
            "QQQ",
            14,
            pd.DataFrame(
                {
                    "close": [100.0],
                    "avg_gain": [1.0],
                    "avg_loss": [1.0],
                    "rsi": [25.0],
                },
                index=pd.to_datetime(["2026-01-02"]),
            ),
        )

    def tearDown(self) -> None:
        self.conn.close()

    def test_generic_pending_report_requires_bound_backtest_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "base_cfg is required"):
            reports_module.build_pending_action_report(
                self.conn,
                pd.DataFrame(),
                14,
                "buy",
                True,
                1.0,
            )

    def test_actionable_report_wrappers_require_bound_backtest_config(self) -> None:
        for builder in (
            reports_module.build_buy_signal_report,
            reports_module.build_sell_signal_report,
        ):
            with (
                self.subTest(builder=builder.__name__),
                self.assertRaisesRegex(
                    ValueError,
                    "base_cfg is required",
                ),
            ):
                builder(self.conn, pd.DataFrame(), 14)

    def test_generic_pending_report_requires_matching_bound_rsi_period(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "rsi_period must match base_cfg.rsi_period",
        ):
            reports_module.build_pending_action_report(
                self.conn,
                pd.DataFrame(),
                13,
                "buy",
                True,
                1.0,
                base_cfg=BacktestConfig(rsi_period=14),
            )

    def test_actionable_report_wrappers_require_matching_bound_rsi_period(self) -> None:
        for builder in (
            reports_module.build_buy_signal_report,
            reports_module.build_sell_signal_report,
        ):
            with (
                self.subTest(builder=builder.__name__),
                self.assertRaisesRegex(
                    ValueError,
                    "rsi_period must match base_cfg.rsi_period",
                ),
            ):
                builder(
                    self.conn,
                    pd.DataFrame(),
                    13,
                    base_cfg=BacktestConfig(rsi_period=14),
                )

    def test_empty_actionable_reports_reject_invalid_rsi_entry_rule(self) -> None:
        cfg = BacktestConfig(rsi_period=14)
        operations = (
            lambda rule: reports_module.build_pending_action_report(
                self.conn,
                pd.DataFrame(),
                14,
                "buy",
                True,
                1.0,
                rsi_entry_rule=rule,
                base_cfg=cfg,
            ),
            lambda rule: reports_module.build_buy_signal_report(
                self.conn,
                pd.DataFrame(),
                14,
                rsi_entry_rule=rule,
                base_cfg=cfg,
            ),
            lambda rule: reports_module.build_sell_signal_report(
                self.conn,
                pd.DataFrame(),
                14,
                rsi_entry_rule=rule,
                base_cfg=cfg,
            ),
        )
        for operation in operations:
            for invalid_rule in ("bogus", None, True, []):
                with (
                    self.subTest(operation=operation, invalid_rule=invalid_rule),
                    self.assertRaisesRegex(
                        ValueError,
                        "Unsupported RSI entry rule",
                    ),
                ):
                    operation(invalid_rule)

    def test_actionable_report_wrappers_validate_period_and_backtest_config_before_empty_return(
        self,
    ) -> None:
        invalid_cases = (
            (True, BacktestConfig(rsi_period=True), "rsi_period must be an integer"),
            (np.int64(14), BacktestConfig(rsi_period=14), "rsi_period must be an integer"),
            (14, BacktestConfig(rsi_period=np.int64(14)), "rsi_period must be an integer"),
            (
                14,
                BacktestConfig(initial_capital=float("nan")),
                "initial_capital must be a finite number",
            ),
        )
        for builder in (
            reports_module.build_buy_signal_report,
            reports_module.build_sell_signal_report,
        ):
            for rsi_period, base_cfg, expected_message in invalid_cases:
                with (
                    self.subTest(
                        builder=builder.__name__,
                        rsi_period=rsi_period,
                        base_rsi_period=base_cfg.rsi_period,
                    ),
                    self.assertRaisesRegex(ValueError, expected_message),
                ):
                    builder(
                        self.conn,
                        pd.DataFrame(),
                        rsi_period,
                        base_cfg=base_cfg,
                    )

    def test_generic_pending_report_validates_unbound_period_before_empty_return(self) -> None:
        with self.assertRaisesRegex(ValueError, "rsi_period must be an integer"):
            reports_module.build_pending_action_report(
                self.conn,
                pd.DataFrame(),
                True,
                "buy",
                True,
                1.0,
                allow_unbound_backtest_config=True,
            )

    def test_successful_report_closes_connection_when_snapshot_rollback_fails(
        self,
    ) -> None:
        conn = sqlite3.connect(":memory:", factory=SnapshotRollbackFailureConnection)
        try:
            conn.fail_next_rollback = True

            with self.assertRaisesRegex(
                sqlite3.OperationalError,
                "forced snapshot rollback failure",
            ):
                summarize_saved_results(
                    conn,
                    pd.DataFrame(columns=["symbol", "rsi_symbol"]),
                )

            self.assertTrue(conn.rollback_failure_observed)
            with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                conn.execute("SELECT 1")
        finally:
            conn.close()

    def test_failed_report_preserves_error_and_closes_on_snapshot_rollback_failure(
        self,
    ) -> None:
        conn = sqlite3.connect(":memory:", factory=SnapshotRollbackFailureConnection)
        original = RuntimeError("forced report failure")
        try:
            conn.fail_next_rollback = True
            with (
                patch.object(
                    reports_module,
                    "_summarize_saved_results_snapshot",
                    side_effect=original,
                ),
                self.assertRaises(RuntimeError) as raised,
            ):
                summarize_saved_results(
                    conn,
                    pd.DataFrame(columns=["symbol", "rsi_symbol"]),
                )

            self.assertIs(raised.exception, original)
            self.assertTrue(conn.rollback_failure_observed)
            self.assertTrue(any("forced snapshot rollback failure" in note for note in original.__notes__))
            with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                conn.execute("SELECT 1")
        finally:
            conn.close()

    def save_canonical_signal_history(self, *, rising: bool) -> None:
        dates = pd.date_range(end="2026-01-01", periods=15, freq="D")
        offsets = np.arange(len(dates), dtype=float)
        close = 100.0 + offsets if rising else 115.0 - offsets
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "QQQ_Open": close,
                    "QQQ_High": close,
                    "QQQ_Low": close,
                    "QQQ_Close": close,
                    "QQQ_Volume": 2_000_000.0,
                },
                index=dates,
            ),
            ["QQQ"],
        )
        ensure_rsi_values(
            self.conn,
            "QQQ",
            14,
            pd.Series(close, index=dates),
            rebuild=True,
        )

    def save_complete_executed_sell_curve(
        self,
        last_date: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        target_conn = self.conn if conn is None else conn
        dates = pd.to_datetime(["2026-01-01", last_date])
        if target_conn.execute("SELECT 1 FROM market_data WHERE symbol = '^IRX' LIMIT 1").fetchone() is None:
            save_market_data(
                target_conn,
                pd.DataFrame(
                    {
                        "^IRX_Open": [0.0],
                        "^IRX_High": [0.0],
                        "^IRX_Low": [0.0],
                        "^IRX_Close": [0.0],
                        "^IRX_Volume": [0.0],
                    },
                    index=pd.to_datetime(["1900-01-01"]),
                ),
                ["^IRX"],
            )
        daily_return = 120_000.0 / 100_000.0 - 1.0
        records = [
            {
                "asset_symbol": "TQQQ",
                "signal_symbol": "QQQ",
                "buy_rsi": 30.0,
                "profit_target_multiple": 2.0,
                "date": "2026-01-01",
                "equity": 100_000.0,
                "daily_return": 0.0,
                "risk_free_return": 0.0,
                "in_position": 0,
                "action_executed": "none",
                "pending_action": "buy",
                "trades_executed": 0,
            },
            {
                "asset_symbol": "TQQQ",
                "signal_symbol": "QQQ",
                "buy_rsi": 30.0,
                "profit_target_multiple": 2.0,
                "date": last_date,
                "equity": 120_000.0,
                "daily_return": daily_return,
                "risk_free_return": 0.0,
                "in_position": 0,
                "action_executed": "sell",
                "pending_action": "none",
                "trades_executed": 2,
            },
        ]
        state = {
            "start_date": "2026-01-01",
            "last_date": last_date,
            "cash": 120_000.0,
            "shares": 0.0,
            "in_position": False,
            "entry_price": float("nan"),
            "pending_action": "none",
            "prev_equity": 120_000.0,
            "trades_executed": 2,
        }
        save_market_data(
            target_conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0, 110.0],
                    "TQQQ_High": [101.0, 220.0],
                    "TQQQ_Low": [99.0, 109.0],
                    "TQQQ_Close": [100.0, 120.0],
                    "TQQQ_Volume": [1_000_000.0, 1_000_000.0],
                },
                index=dates,
            ),
            ["TQQQ"],
        )
        save_strategy_state(target_conn, "TQQQ", "QQQ", 30.0, 2.0, state)
        save_strategy_summary(
            target_conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            state,
            _update_summary_rollup(SummaryRollup(), records),
        )
        save_equity_records(target_conn, records)

    def replace_complete_executed_sell_curve(
        self,
        conn: sqlite3.Connection,
        last_date: str,
    ) -> None:
        conn.execute("DELETE FROM strategy_equity")
        conn.execute("DELETE FROM strategy_summary")
        conn.execute("DELETE FROM strategy_state")
        conn.execute("DELETE FROM market_data WHERE symbol = 'TQQQ'")
        self.save_complete_executed_sell_curve(last_date, conn=conn)

    def test_saved_results_use_one_snapshot_across_concurrent_generation(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "summary-race.sqlite"
            with sqlite3.connect(db_path) as setup_conn:
                init_state_db(setup_conn)
                setup_conn.execute("PRAGMA journal_mode = WAL")
                self.save_complete_executed_sell_curve(
                    "2026-01-02",
                    conn=setup_conn,
                )
            reader = sqlite3.connect(db_path, timeout=5.0)
            first_summary_read = threading.Event()
            writer_done = threading.Event()
            writer_failures: list[BaseException] = []

            def replace_generation() -> None:
                try:
                    if not first_summary_read.wait(5.0):
                        raise TimeoutError("Reader did not reach the summary barrier")
                    with sqlite3.connect(db_path, timeout=5.0) as writer:
                        writer.execute("BEGIN IMMEDIATE")
                        self.replace_complete_executed_sell_curve(
                            writer,
                            "2026-01-03",
                        )
                except BaseException as exc:
                    writer_failures.append(exc)
                finally:
                    writer_done.set()

            original_load = reports_module.load_best_strategy_summary

            def load_then_release_writer(*args: object, **kwargs: object) -> object:
                result = original_load(*args, **kwargs)
                first_summary_read.set()
                if not writer_done.wait(5.0):
                    raise TimeoutError("Concurrent writer did not commit")
                return result

            writer_thread = threading.Thread(target=replace_generation)
            writer_thread.start()
            try:
                with patch.object(
                    reports_module,
                    "load_best_strategy_summary",
                    side_effect=load_then_release_writer,
                ):
                    summary, curves = summarize_saved_results(
                        reader,
                        workflow_assets,
                    )
                self.assertFalse(reader.in_transaction)
            finally:
                writer_thread.join(timeout=5.0)
                reader.close()

            if writer_thread.is_alive():
                self.fail("Concurrent generation writer did not terminate")
            if writer_failures:
                raise writer_failures[0]
            self.assertEqual(summary["End Date"].tolist(), ["2026-01-02"])
            self.assertEqual(
                curves.index.tolist(),
                pd.to_datetime(["2026-01-01", "2026-01-02"]).tolist(),
            )

            with sqlite3.connect(db_path) as current_reader:
                current_summary, current_curves = summarize_saved_results(
                    current_reader,
                    workflow_assets,
                )
            self.assertEqual(current_summary["End Date"].tolist(), ["2026-01-03"])
            self.assertEqual(
                current_curves.index.tolist(),
                pd.to_datetime(["2026-01-01", "2026-01-03"]).tolist(),
            )

    def test_workflow_report_cache_reuses_one_authenticated_curve_exactly(self) -> None:
        self.save_complete_executed_sell_curve("2026-01-02")
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])

        strict_summary, strict_curves = summarize_saved_results(self.conn, workflow_assets)
        strict_buy = build_buy_signal_report(self.conn, strict_summary, 14)
        strict_sell = build_sell_signal_report(self.conn, strict_summary, 14)

        strategy_report_cache: dict[tuple[str, str, str], tuple] = {}
        with (
            patch.object(
                reports_module,
                "load_best_strategy_summary",
                wraps=reports_module.load_best_strategy_summary,
            ) as best_loader,
            patch.object(
                reports_module,
                "load_complete_strategy_equity_curve",
                wraps=reports_module.load_complete_strategy_equity_curve,
            ) as curve_loader,
        ):
            cached_summary, cached_curves = summarize_saved_results(
                self.conn,
                workflow_assets,
                _strategy_report_cache=strategy_report_cache,
            )
            cached_buy = build_buy_signal_report(
                self.conn,
                cached_summary,
                14,
                _strategy_report_cache=strategy_report_cache,
            )
            cached_sell = build_sell_signal_report(
                self.conn,
                cached_summary,
                14,
                _strategy_report_cache=strategy_report_cache,
            )

        self.assertEqual(best_loader.call_count, 1)
        self.assertEqual(curve_loader.call_count, 1)
        for strict, cached in (
            (strict_summary, cached_summary),
            (strict_curves, cached_curves),
            (strict_buy, cached_buy),
            (strict_sell, cached_sell),
        ):
            pd.testing.assert_frame_equal(strict, cached, check_exact=True)

    def test_preverified_workflow_report_does_not_reverify_the_curve(self) -> None:
        self.save_complete_executed_sell_curve("2026-01-02")
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])
        strict_summary, strict_curves = summarize_saved_results(self.conn, workflow_assets)

        with patch.object(
            reports_module,
            "load_complete_strategy_equity_curve",
            side_effect=AssertionError("curve verification repeated"),
        ):
            trusted_summary, trusted_curves = summarize_saved_results(
                self.conn,
                workflow_assets,
                _strategy_report_cache={},
                _workflow_results_preverified=True,
            )

        pd.testing.assert_frame_equal(strict_summary, trusted_summary, check_exact=True)
        pd.testing.assert_frame_equal(strict_curves, trusted_curves, check_exact=True)

    def test_saved_results_checks_deadline_between_assets(self) -> None:
        def expired() -> None:
            raise TimeoutError("report deadline expired")

        with self.assertRaisesRegex(TimeoutError, "report deadline expired"):
            summarize_saved_results(
                self.conn,
                pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}]),
                _strategy_report_cache={},
                deadline_check=expired,
            )

    def test_tied_best_strategy_uses_shared_workflow_specific_policy_and_retained_curve(self) -> None:
        def insert_tied_summaries(asset: str, signal: str, retained_buy_rsi: float) -> None:
            for buy_rsi in (30.0, 70.0):
                for profit_target in (2.0, 1.5):
                    save_strategy_summary(
                        self.conn,
                        asset,
                        signal,
                        buy_rsi,
                        profit_target,
                        {
                            "start_date": "2026-01-01",
                            "last_date": "2026-01-02",
                            "trades_executed": 2,
                        },
                        SummaryRollup(
                            first_equity=100_000.0,
                            last_equity=110_000.0,
                            running_max_equity=110_000.0,
                            return_count=1,
                            return_sum=0.1,
                            return_sum_squares=0.01,
                            return_mean=0.1,
                            return_m2=0.0,
                            excess_return_count=1,
                            excess_return_sum=0.1,
                            excess_return_sum_squares=0.01,
                            excess_return_mean=0.1,
                            excess_return_m2=0.0,
                            positive_return_count=1,
                            max_drawdown=0.0,
                        ),
                    )
            save_strategy_state(
                self.conn,
                asset,
                signal,
                retained_buy_rsi,
                1.5,
                {
                    "start_date": "2026-01-01",
                    "last_date": "2026-01-02",
                    "cash": 110_000.0,
                    "shares": 0.0,
                    "in_position": False,
                    "entry_price": float("nan"),
                    "pending_action": "none",
                    "prev_equity": 110_000.0,
                    "trades_executed": 2,
                },
            )
            dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
            save_market_data(
                self.conn,
                pd.DataFrame(
                    {
                        f"{asset}_Open": [100.0, 100.0],
                        f"{asset}_High": [101.0, 151.0],
                        f"{asset}_Low": [99.0, 100.0],
                        f"{asset}_Close": [100.0, 110.0],
                        f"{asset}_Volume": [1_000_000, 1_000_000],
                    },
                    index=dates,
                ),
                [asset],
            )
            save_equity_records(
                self.conn,
                [
                    {
                        "asset_symbol": asset,
                        "signal_symbol": signal,
                        "buy_rsi": retained_buy_rsi,
                        "profit_target_multiple": 1.5,
                        "date": date,
                        "equity": equity,
                        "daily_return": daily_return,
                        "risk_free_return": 0.0,
                        "in_position": 0,
                        "action_executed": "none" if row_index == 0 else "sell",
                        "pending_action": "buy" if row_index == 0 else "none",
                        "trades_executed": 0 if row_index == 0 else 2,
                    }
                    for row_index, (date, equity, daily_return) in enumerate(
                        (
                            ("2026-01-01", 100_000.0, 0.0),
                            ("2026-01-02", 110_000.0, 0.1),
                        )
                    )
                ],
            )

        insert_tied_summaries("LONGX", "LONG", 30.0)
        insert_tied_summaries("SHORTX", "SHORT", 70.0)

        self.assertEqual(_best_summary_config(self.conn, "LONGX", "LONG", "lower"), (30.0, 1.5))
        self.assertEqual(_best_summary_config(self.conn, "SHORTX", "SHORT", "upper"), (70.0, 1.5))

        summary, curves = summarize_saved_results(
            self.conn,
            pd.DataFrame(
                [
                    {"symbol": "LONGX", "rsi_symbol": "LONG", "workflow": "Long"},
                    {"symbol": "SHORTX", "rsi_symbol": "SHORT", "workflow": "Short"},
                ]
            ),
        )

        selected = summary.set_index("Asset")["Buy RSI"].to_dict()
        self.assertEqual(selected, {"LONGX": 30.0, "SHORTX": 70.0})
        self.assertEqual(summary["Sell Return Multiple"].tolist(), [1.5, 1.5])
        self.assertEqual(
            curves.columns.tolist(),
            ["LONGX_RSI_Strategy", "SHORTX_RSI_Strategy"],
        )

    def test_missing_summary_does_not_authenticate_corrupted_legacy_equity(self) -> None:
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-03",
                "cash": 110_000.0,
                "shares": 0.0,
                "in_position": False,
                "entry_price": float("nan"),
                "pending_action": "buy",
                "prev_equity": 110_000.0,
                "trades_executed": 2,
            },
        )
        self.conn.executemany(
            """
            INSERT INTO strategy_equity (
                asset_symbol, signal_symbol, buy_rsi, profit_target_multiple,
                date, equity, daily_return, risk_free_return, in_position,
                action_executed, pending_action, trades_executed
            ) VALUES ('TQQQ', 'QQQ', 30.0, 1.5, ?, ?, ?, 0.0, 0, 'none', 'buy', 2)
            """,
            [
                ("2026-01-01", 100_000.0, 0.0),
                # A missing legacy summary must not turn this corrupted
                # interior equity observation into an authenticated rollup.
                ("2026-01-02", 200_000.0, 1.0),
                ("2026-01-03", 110_000.0, -0.45),
            ],
        )

        summary, curves = summarize_saved_results(
            self.conn,
            pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}]),
        )

        self.assertTrue(summary.empty)
        self.assertTrue(curves.empty)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM strategy_summary").fetchone()[0],
            0,
        )

    def test_saved_results_rejects_interior_curve_corruption_with_intact_summary_digest(self) -> None:
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        equity_values = [100_000.0, 105_000.0, 110_000.0]
        daily_returns = [
            0.0,
            equity_values[1] / equity_values[0] - 1.0,
            equity_values[2] / equity_values[1] - 1.0,
        ]
        rollup = _update_summary_rollup(
            SummaryRollup(),
            [
                {
                    "equity": equity,
                    "daily_return": daily_return,
                    "risk_free_return": 0.0,
                }
                for equity, daily_return in zip(equity_values, daily_returns, strict=True)
            ],
        )
        state = {
            "start_date": "2026-01-01",
            "last_date": "2026-01-03",
            "cash": 110_000.0,
            "shares": 0.0,
            "in_position": False,
            "entry_price": float("nan"),
            "pending_action": "none",
            "prev_equity": 110_000.0,
            "trades_executed": 2,
        }
        save_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 1.5, state)
        save_strategy_summary(self.conn, "TQQQ", "QQQ", 30.0, 1.5, state, rollup)
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0, 100.0, 110.0],
                    "TQQQ_High": [100.0, 130.0, 151.0],
                    "TQQQ_Low": [100.0, 100.0, 110.0],
                    "TQQQ_Close": [100.0, 121.15384615384616, 110.0],
                    "TQQQ_Volume": [1_000_000, 1_000_000, 1_000_000],
                },
                index=dates,
            ),
            ["TQQQ"],
        )
        save_equity_records(
            self.conn,
            [
                {
                    "asset_symbol": "TQQQ",
                    "signal_symbol": "QQQ",
                    "buy_rsi": 30.0,
                    "profit_target_multiple": 1.5,
                    "date": date.date().isoformat(),
                    "equity": equity,
                    "daily_return": daily_return,
                    "risk_free_return": 0.0,
                    "in_position": int(row_index == 1),
                    "action_executed": ("none", "buy", "sell")[row_index],
                    "pending_action": "buy" if row_index == 0 else "none",
                    "trades_executed": row_index,
                }
                for row_index, (date, equity, daily_return) in enumerate(
                    zip(dates, equity_values, daily_returns, strict=True)
                )
            ],
        )
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])

        valid_summary, valid_curves = summarize_saved_results(self.conn, workflow_assets)

        self.assertEqual(len(valid_summary), 1)
        np.testing.assert_allclose(valid_curves.iloc[:, 0], equity_values)
        summary_digest = self.conn.execute("SELECT integrity_digest FROM strategy_summary").fetchone()[0]

        self.conn.execute("UPDATE strategy_equity SET equity = 200000.0 WHERE date = '2026-01-02'")
        invalid_summary, invalid_curves = summarize_saved_results(self.conn, workflow_assets)

        self.assertTrue(invalid_summary.empty)
        self.assertTrue(invalid_curves.empty)
        self.assertEqual(
            self.conn.execute("SELECT integrity_digest FROM strategy_summary").fetchone()[0],
            summary_digest,
        )

    def test_saved_results_rejects_reordered_curve_with_identical_aggregate_metrics(self) -> None:
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        equity_values = [100.0, 112.5, 140.625]
        daily_returns = [0.0, 0.125, 0.25]
        records = [
            {
                "asset_symbol": "TQQQ",
                "signal_symbol": "QQQ",
                "buy_rsi": 30.0,
                "profit_target_multiple": 1.5,
                "date": date.date().isoformat(),
                "equity": equity,
                "daily_return": daily_return,
                "risk_free_return": 0.0,
                "in_position": int(row_index == 1),
                "action_executed": ("none", "buy", "sell")[row_index],
                "pending_action": "buy" if row_index == 0 else "none",
                "trades_executed": row_index,
            }
            for row_index, (date, equity, daily_return) in enumerate(
                zip(dates, equity_values, daily_returns, strict=True)
            )
        ]
        rollup = _update_summary_rollup(SummaryRollup(), records)
        state = {
            "start_date": "2026-01-01",
            "last_date": "2026-01-03",
            "cash": equity_values[-1],
            "shares": 0.0,
            "in_position": False,
            "entry_price": float("nan"),
            "pending_action": "none",
            "prev_equity": equity_values[-1],
            "trades_executed": 2,
        }
        save_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 1.5, state)
        save_strategy_summary(self.conn, "TQQQ", "QQQ", 30.0, 1.5, state, rollup)
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0, 100.0, 112.5],
                    "TQQQ_High": [100.0, 120.0, 150.0],
                    "TQQQ_Low": [100.0, 100.0, 112.5],
                    "TQQQ_Close": [100.0, 116.12903225806451, 140.625],
                    "TQQQ_Volume": [1_000_000] * 3,
                },
                index=dates,
            ),
            ["TQQQ"],
        )
        save_equity_records(self.conn, records)
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}])
        self.assertEqual(len(summarize_saved_results(self.conn, workflow_assets)[0]), 1)

        # These two internally consistent observations have the same endpoints,
        # sums, squares, moments, and derived metrics in the opposite order.
        self.conn.execute("UPDATE strategy_equity SET equity = 125.0, daily_return = 0.25 WHERE date = '2026-01-02'")
        self.conn.execute("UPDATE strategy_equity SET daily_return = 0.125 WHERE date = '2026-01-03'")

        summary, curves = summarize_saved_results(self.conn, workflow_assets)

        self.assertTrue(summary.empty)
        self.assertTrue(curves.empty)

    def test_buy_report_requires_multiple_trades_and_min_sharpe(self) -> None:
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-02",
                "cash": 100000.0,
                "shares": 0.0,
                "in_position": False,
                "entry_price": float("nan"),
                "pending_action": "buy",
                "prev_equity": 100000.0,
                "trades_executed": 2,
            },
        )
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Start Date": "2026-01-01",
                    "Trading Days": 2,
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 1.1,
                }
            ]
        )

        report = build_buy_signal_report(self.conn, summary, 14)

        self.assertTrue(report.empty)

    def test_buy_report_cannot_forge_persisted_trade_and_sharpe_gates(self) -> None:
        dates = pd.date_range("2025-12-30", periods=4, freq="D")
        signal_close = np.asarray([100.0, 100.0, 100.0, 90.0])
        asset_close = np.full(len(dates), 100.0)
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": asset_close,
                    "TQQQ_High": asset_close,
                    "TQQQ_Low": asset_close,
                    "TQQQ_Close": asset_close,
                    "TQQQ_Volume": 1_000_000.0,
                    "QQQ_Open": signal_close,
                    "QQQ_High": signal_close,
                    "QQQ_Low": signal_close,
                    "QQQ_Close": signal_close,
                    "QQQ_Volume": 2_000_000.0,
                },
                index=dates,
            ),
            ["TQQQ", "QQQ"],
        )
        ensure_rsi_values(
            self.conn,
            "QQQ",
            3,
            pd.Series(signal_close, index=dates),
            rebuild=True,
        )
        state = {
            "start_date": dates[0].date().isoformat(),
            "last_date": dates[-1].date().isoformat(),
            "cash": 100_000.0,
            "shares": 0.0,
            "in_position": False,
            "entry_price": float("nan"),
            "pending_action": "buy",
            "prev_equity": 100_000.0,
            "trades_executed": 0,
        }
        save_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 2.0, state)
        save_strategy_summary(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            state,
            SummaryRollup(
                first_equity=100_000.0,
                last_equity=100_000.0,
                running_max_equity=100_000.0,
                return_count=3,
                return_sum=0.0,
                return_sum_squares=0.0,
                return_mean=0.0,
                return_m2=0.0,
                excess_return_count=3,
                excess_return_sum=0.0,
                excess_return_sum_squares=0.0,
                excess_return_mean=0.0,
                excess_return_m2=0.0,
                positive_return_count=0,
                max_drawdown=0.0,
            ),
        )
        forged_summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Start Date": "1900-01-01",
                    "Trading Days": 1_000_000,
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 1.5,
                }
            ]
        )

        report = build_buy_signal_report(self.conn, forged_summary, 3)

        self.assertTrue(report.empty)

    def test_buy_report_binds_authenticated_summary_to_authenticated_state(self) -> None:
        dates = pd.date_range("2025-12-30", periods=4, freq="D")
        signal_close = np.asarray([100.0, 100.0, 100.0, 90.0])
        asset_close = np.full(len(dates), 100.0)
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": asset_close,
                    "TQQQ_High": asset_close,
                    "TQQQ_Low": asset_close,
                    "TQQQ_Close": asset_close,
                    "TQQQ_Volume": 1_000_000.0,
                    "QQQ_Open": signal_close,
                    "QQQ_High": signal_close,
                    "QQQ_Low": signal_close,
                    "QQQ_Close": signal_close,
                    "QQQ_Volume": 2_000_000.0,
                },
                index=dates,
            ),
            ["TQQQ", "QQQ"],
        )
        ensure_rsi_values(
            self.conn,
            "QQQ",
            3,
            pd.Series(signal_close, index=dates),
            rebuild=True,
        )
        state = {
            "start_date": dates[0].date().isoformat(),
            "last_date": dates[-1].date().isoformat(),
            "cash": 100_000.0,
            "shares": 0.0,
            "in_position": False,
            "entry_price": float("nan"),
            "pending_action": "buy",
            "prev_equity": 100_000.0,
            "trades_executed": 0,
        }
        save_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 2.0, state)
        summary_state = {**state, "trades_executed": 2}
        save_strategy_summary(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            summary_state,
            SummaryRollup(
                first_equity=100_000.0,
                last_equity=133_089.0,
                running_max_equity=133_089.0,
                return_count=3,
                return_sum=0.3,
                return_sum_squares=0.0302,
                return_mean=0.1,
                return_m2=0.0002,
                excess_return_count=3,
                excess_return_sum=0.3,
                excess_return_sum_squares=0.0302,
                excess_return_mean=0.1,
                excess_return_m2=0.0002,
                positive_return_count=3,
                max_drawdown=0.0,
            ),
        )
        selector = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                }
            ]
        )

        self.assertIsNotNone(
            load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
                rsi_period=3,
            )
        )
        persisted_summary = load_best_strategy_summary(self.conn, "TQQQ", "QQQ")
        if persisted_summary is None:
            self.fail("Expected the independently authenticated persisted summary")
        self.assertEqual(persisted_summary["trades_executed"], 2)
        self.assertGreater(persisted_summary["sharpe"], 1.0)
        self.assertTrue(build_buy_signal_report(self.conn, selector, 3).empty)

    def test_actionable_state_without_canonical_signal_history_fails_closed(self) -> None:
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-02",
                "cash": 100_000.0,
                "shares": 0.0,
                "in_position": False,
                "entry_price": float("nan"),
                "pending_action": "buy",
                "prev_equity": 100_000.0,
                "trades_executed": 2,
            },
        )
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 1.5,
                }
            ]
        )

        self.assertIsNone(
            load_strategy_state(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
                rsi_period=14,
            )
        )
        self.assertTrue(build_buy_signal_report(self.conn, summary, 14).empty)

    def test_weekend_cached_rsi_without_canonical_history_fails_closed(self) -> None:
        asset_date = pd.Timestamp("2026-01-09")
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0],
                    "TQQQ_High": [101.0],
                    "TQQQ_Low": [99.0],
                    "TQQQ_Close": [100.0],
                    "TQQQ_Volume": [1_000_000.0],
                },
                index=[asset_date],
            ),
            ["TQQQ"],
        )
        save_rsi_values(
            self.conn,
            "QQQ",
            14,
            pd.DataFrame(
                {
                    "close": [100.0, 80.0, 60.0],
                    "avg_gain": [1.0, 0.5, 0.25],
                    "avg_loss": [1.0, 5.0, 10.0],
                    "rsi": [25.0, 15.0, 9.0],
                },
                index=pd.to_datetime(["2026-01-09", "2026-01-10", "2026-01-11"]),
            ),
        )
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-09",
                "cash": 100000.0,
                "shares": 0.0,
                "in_position": False,
                "entry_price": float("nan"),
                "pending_action": "buy",
                "prev_equity": 100000.0,
                "trades_executed": 2,
            },
        )
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Start Date": "2026-01-01",
                    "Trading Days": 5,
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 1.1,
                }
            ]
        )

        report = build_buy_signal_report(self.conn, summary, 14)

        self.assertTrue(report.empty)

    def test_sell_report_does_not_require_multiple_trades_or_min_sharpe(self) -> None:
        dates = pd.date_range(end="2026-01-02", periods=17, freq="D")
        signal_close = 117.0 - np.arange(len(dates), dtype=float)
        asset_close = np.full(len(dates), 100.0)
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": asset_close,
                    "TQQQ_High": asset_close,
                    "TQQQ_Low": asset_close,
                    "TQQQ_Close": asset_close,
                    "TQQQ_Volume": 1_000_000.0,
                    "QQQ_Open": signal_close,
                    "QQQ_High": signal_close,
                    "QQQ_Low": signal_close,
                    "QQQ_Close": signal_close,
                    "QQQ_Volume": 2_000_000.0,
                },
                index=dates,
            ),
            ["TQQQ", "QQQ"],
        )
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": dates[0].date().isoformat(),
                "last_date": "2026-01-02",
                "cash": 0.0,
                "shares": 1_000.0,
                "in_position": True,
                "entry_price": 100.0,
                "entry_date": "2026-01-01",
                "pending_action": "sell",
                "prev_equity": 100000.0,
                "trades_executed": 1,
            },
        )
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 1,
                    "Sharpe": float("nan"),
                }
            ]
        )

        report = build_sell_signal_report(self.conn, summary, 14)

        self.assertTrue(report.empty)

    def test_sell_report_authenticates_upper_rule_entry_chronology(self) -> None:
        dates = pd.date_range(end="2026-01-02", periods=17, freq="D")
        signal_close = 100.0 + np.arange(len(dates), dtype=float)
        asset_close = np.full(len(dates), 100.0)
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": asset_close,
                    "TQQQ_High": asset_close,
                    "TQQQ_Low": asset_close,
                    "TQQQ_Close": asset_close,
                    "TQQQ_Volume": 1_000_000.0,
                    "QQQ_Open": signal_close,
                    "QQQ_High": signal_close,
                    "QQQ_Low": signal_close,
                    "QQQ_Close": signal_close,
                    "QQQ_Volume": 2_000_000.0,
                },
                index=dates,
            ),
            ["TQQQ", "QQQ"],
        )
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            70.0,
            2.0,
            {
                "start_date": dates[0].date().isoformat(),
                "last_date": "2026-01-02",
                "cash": 0.0,
                "shares": 1_000.0,
                "in_position": True,
                "entry_price": 100.0,
                "entry_date": "2026-01-01",
                "pending_action": "sell",
                "prev_equity": 100_000.0,
                "trades_executed": 1,
            },
        )
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 70.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 1,
                    "Sharpe": float("nan"),
                }
            ]
        )

        self.assertTrue(build_sell_signal_report(self.conn, summary, 14).empty)
        report = build_sell_signal_report(
            self.conn,
            summary,
            14,
            rsi_entry_rule="upper",
        )
        self.assertTrue(report.empty)

    def test_sell_report_includes_latest_executed_target_exit(self) -> None:
        self.save_complete_executed_sell_curve("2026-01-02")
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 0.5,
                }
            ]
        )

        report = build_sell_signal_report(self.conn, summary, 14)

        self.assertEqual(report["Asset"].tolist(), ["TQQQ"])
        self.assertFalse(bool(report.loc[0, "In Position"]))
        self.assertEqual(report.loc[0, "Pending Action"], "sell")

    def test_pending_report_uses_one_snapshot_across_concurrent_generation(self) -> None:
        selector = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "report-race.sqlite"
            with sqlite3.connect(db_path) as setup_conn:
                init_state_db(setup_conn)
                setup_conn.execute("PRAGMA journal_mode = WAL")
                self.save_complete_executed_sell_curve(
                    "2026-01-02",
                    conn=setup_conn,
                )
            reader = sqlite3.connect(db_path, timeout=5.0)
            first_summary_read = threading.Event()
            writer_done = threading.Event()
            writer_failures: list[BaseException] = []

            def replace_generation() -> None:
                try:
                    if not first_summary_read.wait(5.0):
                        raise TimeoutError("Reader did not reach the summary barrier")
                    with sqlite3.connect(db_path, timeout=5.0) as writer:
                        writer.execute("BEGIN IMMEDIATE")
                        self.replace_complete_executed_sell_curve(
                            writer,
                            "2026-01-03",
                        )
                except BaseException as exc:
                    writer_failures.append(exc)
                finally:
                    writer_done.set()

            original_load = reports_module.load_best_strategy_summary

            def load_then_release_writer(*args: object, **kwargs: object) -> object:
                result = original_load(*args, **kwargs)
                first_summary_read.set()
                if not writer_done.wait(5.0):
                    raise TimeoutError("Concurrent writer did not commit")
                return result

            writer_thread = threading.Thread(target=replace_generation)
            writer_thread.start()
            try:
                with patch.object(
                    reports_module,
                    "load_best_strategy_summary",
                    side_effect=load_then_release_writer,
                ):
                    report = build_sell_signal_report(reader, selector, 14)
                self.assertFalse(reader.in_transaction)
            finally:
                writer_thread.join(timeout=5.0)
                reader.close()

            if writer_thread.is_alive():
                self.fail("Concurrent generation writer did not terminate")
            if writer_failures:
                raise writer_failures[0]
            self.assertEqual(report["Date"].tolist(), ["2026-01-02"])

            with sqlite3.connect(db_path) as current_reader:
                current = build_sell_signal_report(current_reader, selector, 14)
            self.assertEqual(current["Date"].tolist(), ["2026-01-03"])

    def test_pending_report_preserves_caller_transaction_and_cleans_owned_failure(
        self,
    ) -> None:
        self.save_complete_executed_sell_curve("2026-01-02")
        self.conn.commit()
        selector = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                }
            ]
        )
        self.conn.execute("BEGIN")
        build_sell_signal_report(self.conn, selector, 14)
        self.assertTrue(self.conn.in_transaction)
        self.conn.rollback()

        with (
            patch.object(
                reports_module,
                "_build_pending_action_report_snapshot",
                side_effect=KeyboardInterrupt,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            build_sell_signal_report(self.conn, selector, 14)
        self.assertFalse(self.conn.in_transaction)

    def test_signal_report_deduplicates_identical_strategy_selectors(self) -> None:
        self.save_complete_executed_sell_curve("2026-01-02")
        selector = {
            "Asset": "TQQQ",
            "RSI Symbol": "QQQ",
            "Buy RSI": 30.0,
            "Sell Return Multiple": 2.0,
        }

        report = build_sell_signal_report(
            self.conn,
            pd.DataFrame([selector, selector]),
            14,
        )

        self.assertEqual(report["Asset"].tolist(), ["TQQQ"])

    def test_sell_report_rejects_latest_action_when_complete_curve_does_not_authenticate(self) -> None:
        self.save_complete_executed_sell_curve("2026-01-02")
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 0.5,
                }
            ]
        )
        self.assertEqual(build_sell_signal_report(self.conn, summary, 14)["Asset"].tolist(), ["TQQQ"])

        self.conn.execute(
            """
            UPDATE strategy_equity
            SET equity = equity + 1.0
            WHERE asset_symbol = 'TQQQ' AND date = '2026-01-01'
            """
        )

        self.assertTrue(build_sell_signal_report(self.conn, summary, 14).empty)

    def test_executed_target_exit_does_not_require_fresh_rsi_audit_data(self) -> None:
        self.save_complete_executed_sell_curve("2026-01-15")
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 0.5,
                }
            ]
        )

        cases = [
            (None, pd.NA, pd.NA),
            (("2026-01-02", 25.0), "2026-01-02", 25.0),
        ]
        for aligned_rsi, expected_date, expected_rsi in cases:
            with (
                self.subTest(aligned_rsi=aligned_rsi),
                patch(
                    "leveraged_trader.reports.load_aligned_rsi_for_asset_session",
                    return_value=aligned_rsi,
                ),
            ):
                report = build_sell_signal_report(self.conn, summary, 14)

                self.assertEqual(report["Asset"].tolist(), ["TQQQ"])
                if pd.isna(expected_date):
                    self.assertTrue(pd.isna(report.loc[0, "RSI Observation Date"]))
                    self.assertTrue(pd.isna(report.loc[0, "Latest RSI"]))
                else:
                    self.assertEqual(report.loc[0, "RSI Observation Date"], expected_date)
                    self.assertEqual(report.loc[0, "Latest RSI"], expected_rsi)

    def test_signal_report_rejects_stale_live_rsi_observation(self) -> None:
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-15",
                "cash": 100000.0,
                "shares": 0.0,
                "in_position": False,
                "entry_price": float("nan"),
                "pending_action": "buy",
                "prev_equity": 100000.0,
                "trades_executed": 2,
            },
        )
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 1.5,
                }
            ]
        )

        with patch(
            "leveraged_trader.reports.load_aligned_rsi_for_asset_session",
            return_value=("2026-01-02", 25.0),
        ):
            report = build_buy_signal_report(self.conn, summary, 14)

        self.assertTrue(report.empty)

    def test_signal_report_rejects_tampered_strategy_state_digest_and_storage_type(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 1.5,
                }
            ]
        )
        base_state = {
            "start_date": "2026-01-01",
            "last_date": "2026-01-02",
            "cash": 100000.0,
            "shares": 0.0,
            "in_position": False,
            "entry_price": float("nan"),
            "pending_action": "none",
            "prev_equity": 100000.0,
            "trades_executed": 2,
        }
        cases = (
            (
                "stale digest",
                "UPDATE strategy_state SET pending_action = 'buy'",
                base_state,
            ),
            (
                "normalized blob",
                "UPDATE strategy_state SET in_position = CAST('0' AS BLOB)",
                {**base_state, "pending_action": "buy"},
            ),
        )
        for label, mutation, state in cases:
            with self.subTest(label=label):
                save_strategy_state(
                    self.conn,
                    "TQQQ",
                    "QQQ",
                    30.0,
                    2.0,
                    state,
                )
                self.conn.execute(mutation)

                loaded = load_strategy_state(
                    self.conn,
                    "TQQQ",
                    "QQQ",
                    30.0,
                    2.0,
                )
                report = build_buy_signal_report(self.conn, summary, 14)

                self.assertIsNone(loaded)
                self.assertTrue(report.empty)

    def test_signal_report_rejects_authenticated_but_incoherent_account_shape(self) -> None:
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": "2026-01-01",
                "last_date": "2026-01-02",
                "cash": 100_000.0,
                "shares": 10.0,
                "in_position": False,
                "entry_price": 100.0,
                "pending_action": "buy",
                "prev_equity": 100_000.0,
                "trades_executed": 2,
            },
        )
        optimization_summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 2,
                    "Sharpe": 1.5,
                }
            ]
        )

        self.assertIsNone(load_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 2.0))
        self.assertTrue(build_buy_signal_report(self.conn, optimization_summary, 14).empty)

    def test_complete_curve_rejects_authenticated_zero_trade_equity_growth(self) -> None:
        records = [
            {
                "asset_symbol": "TQQQ",
                "signal_symbol": "QQQ",
                "buy_rsi": 30.0,
                "profit_target_multiple": 1.5,
                "date": date,
                "equity": equity,
                "daily_return": daily_return,
                "risk_free_return": 0.0,
                "in_position": 0,
                "action_executed": "none",
                "pending_action": "none",
                "trades_executed": 0,
            }
            for date, equity, daily_return in (
                ("2026-01-01", 100_000.0, 0.0),
                ("2026-01-02", 110_000.0, 0.1),
            )
        ]
        state = {
            "start_date": "2026-01-01",
            "last_date": "2026-01-02",
            "cash": 110_000.0,
            "shares": 0.0,
            "in_position": False,
            "entry_price": float("nan"),
            "pending_action": "none",
            "prev_equity": 110_000.0,
            "trades_executed": 0,
        }
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0, 110.0],
                    "TQQQ_High": [100.0, 110.0],
                    "TQQQ_Low": [100.0, 110.0],
                    "TQQQ_Close": [100.0, 110.0],
                    "TQQQ_Volume": [1_000_000.0, 1_000_000.0],
                },
                index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
            ),
            ["TQQQ"],
        )
        save_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 1.5, state)
        save_strategy_summary(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            state,
            _update_summary_rollup(SummaryRollup(), records),
        )
        save_equity_records(self.conn, records)

        self.assertIsNone(
            load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                allow_unbound_backtest_config=True,
            )
        )
        summary, curves = summarize_saved_results(
            self.conn,
            pd.DataFrame([{"symbol": "TQQQ", "rsi_symbol": "QQQ"}]),
        )
        self.assertTrue(summary.empty)
        self.assertTrue(curves.empty)

    def test_unbound_complete_curve_rejects_large_dollar_negative_inferred_cost(self) -> None:
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        initial_equity = 1e15
        forged_held_equity = initial_equity + 500.0
        records = [
            {
                "asset_symbol": "TQQQ",
                "signal_symbol": "QQQ",
                "buy_rsi": 30.0,
                "profit_target_multiple": 1.5,
                "date": date.date().isoformat(),
                "equity": equity,
                "daily_return": daily_return,
                "risk_free_return": 0.0,
                "in_position": in_position,
                "action_executed": action,
                "pending_action": pending,
                "trades_executed": trades,
            }
            for date, equity, daily_return, in_position, action, pending, trades in (
                (dates[0], initial_equity, 0.0, 0, "none", "buy", 0),
                (
                    dates[1],
                    forged_held_equity,
                    forged_held_equity / initial_equity - 1.0,
                    1,
                    "buy",
                    "none",
                    1,
                ),
                (dates[2], forged_held_equity, 0.0, 1, "none", "none", 1),
            )
        ]
        state = {
            "start_date": dates[0].date().isoformat(),
            "last_date": dates[-1].date().isoformat(),
            "cash": 0.0,
            "shares": forged_held_equity,
            "in_position": True,
            "entry_price": 1.0,
            "entry_date": dates[1].date().isoformat(),
            "pending_action": "none",
            "prev_equity": forged_held_equity,
            "trades_executed": 1,
        }
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [1.0] * 3,
                    "TQQQ_High": [1.0] * 3,
                    "TQQQ_Low": [1.0] * 3,
                    "TQQQ_Close": [1.0] * 3,
                    "TQQQ_Volume": [1_000_000.0] * 3,
                    "^IRX_Open": [0.0] * 3,
                    "^IRX_High": [0.0] * 3,
                    "^IRX_Low": [0.0] * 3,
                    "^IRX_Close": [0.0] * 3,
                    "^IRX_Volume": [0.0] * 3,
                },
                index=dates,
            ),
            ["TQQQ", "^IRX"],
        )
        save_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 1.5, state)
        save_strategy_summary(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            state,
            _update_summary_rollup(SummaryRollup(), records),
        )
        save_equity_records(self.conn, records)

        self.assertIsNone(
            load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                allow_unbound_backtest_config=True,
            )
        )

    def test_complete_curve_exactly_binds_tiny_state_summary_endpoint(self) -> None:
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        records = [
            {
                "asset_symbol": "TQQQ",
                "signal_symbol": "QQQ",
                "buy_rsi": 30.0,
                "profit_target_multiple": 1.5,
                "date": date.date().isoformat(),
                "equity": 5e-14,
                "daily_return": 0.0,
                "risk_free_return": 0.0,
                "in_position": 0,
                "action_executed": "none",
                "pending_action": "none",
                "trades_executed": 0,
            }
            for date in dates
        ]
        summary_state = {
            "start_date": "2026-01-01",
            "last_date": "2026-01-02",
            "trades_executed": 0,
        }
        mismatched_state = {
            **summary_state,
            "cash": 8e-14,
            "shares": 0.0,
            "in_position": False,
            "entry_price": float("nan"),
            "pending_action": "none",
            "prev_equity": 8e-14,
        }
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0, 100.0],
                    "TQQQ_High": [100.0, 100.0],
                    "TQQQ_Low": [100.0, 100.0],
                    "TQQQ_Close": [100.0, 100.0],
                    "TQQQ_Volume": [1_000_000.0, 1_000_000.0],
                },
                index=dates,
            ),
            ["TQQQ"],
        )
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            mismatched_state,
        )
        save_strategy_summary(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            summary_state,
            _update_summary_rollup(SummaryRollup(), records),
        )
        save_equity_records(self.conn, records)

        self.assertIsNotNone(load_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 1.5))
        self.assertIsNotNone(load_best_strategy_summary(self.conn, "TQQQ", "QQQ"))
        self.assertIsNone(
            load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                allow_unbound_backtest_config=True,
            )
        )

    def test_complete_curve_rejects_impossible_tiny_interior_position_equity(self) -> None:
        dates = pd.date_range("2026-01-01", periods=4, freq="D")
        state = {
            "start_date": dates[0].date().isoformat(),
            "last_date": dates[-1].date().isoformat(),
            "cash": 0.0,
            "shares": 5e-16,
            "in_position": True,
            "entry_price": 100.0,
            "entry_date": dates[1].date().isoformat(),
            "pending_action": "none",
            "prev_equity": 5e-14,
            "trades_executed": 1,
        }
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0] * 4,
                    "TQQQ_High": [100.0] * 4,
                    "TQQQ_Low": [100.0] * 4,
                    "TQQQ_Close": [100.0] * 4,
                    "TQQQ_Volume": [1_000_000.0] * 4,
                },
                index=dates,
            ),
            ["TQQQ"],
        )

        def records_for(equities: list[float]) -> list[dict[str, object]]:
            daily_returns = [
                0.0,
                *(equities[index] / equities[index - 1] - 1.0 for index in range(1, len(equities))),
            ]
            return [
                {
                    "asset_symbol": "TQQQ",
                    "signal_symbol": "QQQ",
                    "buy_rsi": 30.0,
                    "profit_target_multiple": 2.0,
                    "date": date.date().isoformat(),
                    "equity": equity,
                    "daily_return": daily_return,
                    "risk_free_return": 0.0,
                    "in_position": int(row_index > 0),
                    "action_executed": "buy" if row_index == 1 else "none",
                    "pending_action": "buy" if row_index == 0 else "none",
                    "trades_executed": int(row_index > 0),
                }
                for row_index, (date, equity, daily_return) in enumerate(
                    zip(dates, equities, daily_returns, strict=True)
                )
            ]

        valid_records = records_for([5e-14] * 4)
        save_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 2.0, state)
        save_strategy_summary(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            state,
            _update_summary_rollup(SummaryRollup(), valid_records),
        )
        save_equity_records(self.conn, valid_records)
        self.assertIsNotNone(
            load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
                allow_unbound_backtest_config=True,
            )
        )

        impossible_records = records_for([5e-14, 5e-14, 8e-14, 5e-14])
        save_strategy_summary(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            state,
            _update_summary_rollup(SummaryRollup(), impossible_records),
        )
        save_equity_records(self.conn, impossible_records)

        self.assertIsNone(
            load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                2.0,
                allow_unbound_backtest_config=True,
            )
        )

    def test_complete_curve_rejects_profit_unreachable_from_market_highs(self) -> None:
        records = [
            {
                "asset_symbol": "TQQQ",
                "signal_symbol": "QQQ",
                "buy_rsi": 30.0,
                "profit_target_multiple": 1.5,
                "date": date,
                "equity": equity,
                "daily_return": daily_return,
                "risk_free_return": 0.0,
                "in_position": in_position,
                "action_executed": action,
                "pending_action": pending,
                "trades_executed": trades,
            }
            for date, equity, daily_return, in_position, action, pending, trades in (
                ("2026-01-01", 100_000.0, 0.0, 0, "none", "buy", 0),
                ("2026-01-02", 100_000.0, 0.0, 1, "buy", "none", 1),
                ("2026-01-03", 200_000.0, 1.0, 0, "sell", "none", 2),
            )
        ]
        state = {
            "start_date": "2026-01-01",
            "last_date": "2026-01-03",
            "cash": 200_000.0,
            "shares": 0.0,
            "in_position": False,
            "entry_price": float("nan"),
            "pending_action": "none",
            "prev_equity": 200_000.0,
            "trades_executed": 2,
        }
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": [100.0, 100.0, 100.0],
                    "TQQQ_High": [100.0, 100.0, 100.0],
                    "TQQQ_Low": [100.0, 100.0, 100.0],
                    "TQQQ_Close": [100.0, 100.0, 100.0],
                    "TQQQ_Volume": [1_000_000.0] * 3,
                },
                index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            ),
            ["TQQQ"],
        )
        save_strategy_state(self.conn, "TQQQ", "QQQ", 30.0, 1.5, state)
        save_strategy_summary(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            1.5,
            state,
            _update_summary_rollup(SummaryRollup(), records),
        )
        save_equity_records(self.conn, records)

        self.assertIsNone(
            load_complete_strategy_equity_curve(
                self.conn,
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                allow_unbound_backtest_config=True,
            )
        )

    def test_signal_reports_allow_missing_sharpe_metrics(self) -> None:
        dates = pd.date_range(end="2026-01-02", periods=17, freq="D")
        signal_close = 117.0 - np.arange(len(dates), dtype=float)
        asset_close = np.full(len(dates), 100.0)
        save_market_data(
            self.conn,
            pd.DataFrame(
                {
                    "TQQQ_Open": asset_close,
                    "TQQQ_High": asset_close,
                    "TQQQ_Low": asset_close,
                    "TQQQ_Close": asset_close,
                    "TQQQ_Volume": 1_000_000.0,
                    "QQQ_Open": signal_close,
                    "QQQ_High": signal_close,
                    "QQQ_Low": signal_close,
                    "QQQ_Close": signal_close,
                    "QQQ_Volume": 2_000_000.0,
                },
                index=dates,
            ),
            ["TQQQ", "QQQ"],
        )
        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": dates[0].date().isoformat(),
                "last_date": "2026-01-02",
                "cash": 100000.0,
                "shares": 0.0,
                "in_position": False,
                "entry_price": float("nan"),
                "pending_action": "buy",
                "prev_equity": 100000.0,
                "trades_executed": 0,
            },
        )
        summary = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 2.0,
                    "Trades Executed": 0,
                    "Sharpe": None,
                }
            ]
        )

        buy_report = build_buy_signal_report(self.conn, summary, 14)

        save_strategy_state(
            self.conn,
            "TQQQ",
            "QQQ",
            30.0,
            2.0,
            {
                "start_date": dates[0].date().isoformat(),
                "last_date": "2026-01-02",
                "cash": 0.0,
                "shares": 1_000.0,
                "in_position": True,
                "entry_price": 100.0,
                "entry_date": "2026-01-01",
                "pending_action": "sell",
                "prev_equity": 100000.0,
                "trades_executed": 1,
            },
        )
        sell_report = build_sell_signal_report(self.conn, summary, 14)

        self.assertTrue(buy_report.empty)
        self.assertTrue(sell_report.empty)

    def test_realized_pnl_summary_uses_complete_closed_managed_positions(self) -> None:
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sell_status, sell_filled_qty, sell_filled_avg_price, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "TQQQ",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    "rsi-buy-TQQQ-20260102",
                    "filled",
                    2.0,
                    100.0,
                    "filled",
                    2.0,
                    125.0,
                    "2026-01-03T15:00:00Z",
                ),
                (
                    "UPRO",
                    "SPY",
                    30.0,
                    1.5,
                    "2026-01-02",
                    "rsi-buy-UPRO-20260102",
                    "filled",
                    1.0,
                    50.0,
                    "filled",
                    None,
                    None,
                    "2026-01-03T15:00:00Z",
                ),
            ],
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 2)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Total Buy Cost"], 200.0)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 250.0)
        self.assertEqual(summary.loc[0, "Realized P/L"], 50.0)
        self.assertEqual(summary.loc[0, "Realized P/L %"], 25.0)

    def test_realized_pnl_summary_classifies_legacy_null_workflow_as_long_after_initialization(self) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (workflow, symbol, signal_symbol, buy_rsi, profit_target_multiple,
             buy_signal_date, buy_client_order_id, buy_status, filled_qty,
             filled_avg_price, sell_filled_qty, sell_filled_avg_price,
             realized_pl, realized_pl_pct, sold_qty, sold_value, remaining_qty,
             closed_at)
            VALUES (NULL, 'TQQQ', 'QQQ', 30, 1.5, '2026-01-02',
                    'rsi-buy-TQQQ-legacy-workflow-report', 'filled', 2, 100,
                    2, 125, 50, 25, 2, 250, 0,
                    '2026-01-03T15:00:00Z')
            """
        )
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
            VALUES (?, 'sell-legacy-workflow-report', 2, 250)
            """,
            (int(cursor.lastrowid),),
        )

        init_state_db(self.conn)
        summary = build_alpaca_realized_pnl_summary(self.conn, include_workflow=True)

        self.assertEqual(summary["Workflow"].tolist(), ["Long", "Total"])
        self.assertEqual(summary["Closed Positions"].tolist(), [1, 1])
        self.assertEqual(summary["Realized P/L"].tolist(), [50.0, 50.0])

    def test_realized_pnl_summary_adds_total_for_long_and_short_workflows(self) -> None:
        position_ids = []
        for workflow, symbol, quantity, buy_price, sell_price in (
            ("Long", "TQQQ", 1.0, 100.0, 120.0),
            ("Short", "SQQQ", 2.0, 100.0, 130.0),
        ):
            cursor = self.conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (workflow, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                 buy_signal_date, buy_client_order_id, buy_status, filled_qty,
                 filled_avg_price, sell_filled_qty, sell_filled_avg_price,
                 sold_qty, sold_value, remaining_qty, closed_at)
                VALUES (?, ?, 'QQQ', 30, 1.5, '2026-01-02', ?, 'filled', ?, ?,
                        ?, ?, ?, ?, 0, '2026-01-03T15:00:00Z')
                """,
                (
                    workflow,
                    symbol,
                    f"rsi-buy-{symbol}-total-row",
                    quantity,
                    buy_price,
                    quantity,
                    sell_price,
                    quantity,
                    quantity * sell_price,
                ),
            )
            position_ids.append((int(cursor.lastrowid), symbol, quantity, sell_price))
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
            VALUES (?, ?, ?, ?)
            """,
            [
                (position_id, f"sell-{symbol}-total-row", quantity, quantity * sell_price)
                for position_id, symbol, quantity, sell_price in position_ids
            ],
        )

        summary = build_alpaca_realized_pnl_summary(self.conn, include_workflow=True)

        self.assertEqual(summary["Workflow"].tolist(), ["Long", "Short", "Total"])
        total = summary.iloc[-1]
        self.assertEqual(total["Closed Positions"], 2)
        self.assertEqual(total["Complete Closed Positions"], 2)
        self.assertEqual(total["Incomplete Closed Positions"], 0)
        self.assertEqual(total["Total Buy Cost"], 300.0)
        self.assertEqual(total["Total Sell Value"], 380.0)
        self.assertEqual(total["Realized P/L"], 80.0)
        self.assertAlmostEqual(total["Realized P/L %"], 80.0 / 300.0 * 100.0)

    def test_realized_pnl_summary_does_not_hide_closed_fill_when_status_is_stale(self) -> None:
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sell_status, sell_filled_qty, sell_filled_avg_price, remaining_qty, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                "2026-01-02",
                "rsi-buy-TQQQ-20260102",
                "filled",
                1.0,
                100.0,
                "accepted",
                1.0,
                120.0,
                0.0,
                "2026-01-03T15:00:00Z",
            ),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Realized P/L"], 20.0)

    def test_realized_pnl_summary_quarantines_terminal_buy_underfill(self) -> None:
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, buy_order_qty, buy_order_limit_price,
             filled_qty, filled_avg_price, sold_qty, sold_value, remaining_qty,
             closed_at, notes)
            VALUES ('TQQQ', 'QQQ', 30, 1.5, '2026-01-02',
                    'rsi-buy-terminal-underfill', 'filled', 2, 105,
                    1, 100, 1, 120, 0, '2026-01-03T15:00:00Z',
                    'Alpaca buy filled status reports less than the immutable managed buy intent; protective ' ||
                    'accounting is retained but automated realized-P/L publication is quarantined')
            """
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

        self.conn.execute(
            "UPDATE alpaca_managed_positions SET buy_status = 'canceled', buy_causality_quarantine = notes"
        )
        marker_only = build_alpaca_realized_pnl_summary(self.conn)
        self.assertEqual(marker_only.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(marker_only.loc[0, "Incomplete Closed Positions"], 1)

        self.conn.execute("UPDATE alpaca_managed_positions SET buy_order_qty = NULL, buy_order_limit_price = NULL")
        legacy_marker_only = build_alpaca_realized_pnl_summary(self.conn)
        self.assertEqual(legacy_marker_only.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(legacy_marker_only.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(legacy_marker_only.loc[0, "Realized P/L"], 0.0)

    def test_realized_pnl_summary_rejects_inconsistent_closed_fill_accounting(self) -> None:
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                "2026-01-02",
                "rsi-buy-TQQQ-20260102",
                "filled",
                2.0,
                100.0,
                1.0,
                120.0,
                0.0,
                "2026-01-03T15:00:00Z",
            ),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Total Buy Cost"], 0.0)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 0.0)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

    def test_realized_pnl_summary_requires_child_sell_ledger_to_match_parent_aggregate(self) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                "2026-01-02",
                "rsi-buy-TQQQ-ledger-reconciliation",
                "filled",
                2.0,
                100.0,
                2.0,
                240.0,
                0.0,
                "2026-01-03T15:00:00Z",
            ),
        )
        position_id = int(cursor.lastrowid)
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
            VALUES (?, 'sell-order-1', 2.0, 240.0)
            """,
            (position_id,),
        )
        matching = build_alpaca_realized_pnl_summary(self.conn)
        self.assertEqual(matching.loc[0, "Complete Closed Positions"], 1)
        self.assertEqual(matching.loc[0, "Realized P/L"], 40.0)

        self.conn.execute(
            """
            UPDATE alpaca_managed_sell_fills
            SET filled_qty = 1.0, filled_value = 120.0
            WHERE managed_position_id = ?
            """,
            (position_id,),
        )

        inconsistent = build_alpaca_realized_pnl_summary(self.conn)
        self.assertEqual(inconsistent.loc[0, "Closed Positions"], 1)
        self.assertEqual(inconsistent.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(inconsistent.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(inconsistent.loc[0, "Total Buy Cost"], 0.0)
        self.assertEqual(inconsistent.loc[0, "Total Sell Value"], 0.0)

    def test_realized_pnl_summary_rejects_scale_relative_dollar_ledger_mismatch(
        self,
    ) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, buy_order_qty, buy_order_limit_price,
             filled_qty, filled_avg_price, sold_qty, sold_value, remaining_qty, closed_at)
            VALUES ('TQQQ', 'QQQ', 30, 1.5, '2026-01-02', 'buy-value-mismatch',
                    'filled', 1, 50000000, 1, 50000000, 1, 100000000.4, 0,
                    '2026-01-03T15:00:00Z')
            """
        )
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value,
             submitted_qty, submitted_limit_price)
            VALUES (?, 'sell-value-mismatch', 1, 100000000, 1, 100000000)
            """,
            (int(cursor.lastrowid),),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 0.0)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

        self.conn.execute(
            "UPDATE alpaca_managed_positions SET sold_value = ? WHERE id = ?",
            (
                float(np.nextafter(100_000_000.0, np.inf)),
                int(cursor.lastrowid),
            ),
        )
        roundoff_only = build_alpaca_realized_pnl_summary(self.conn)
        self.assertEqual(roundoff_only.loc[0, "Complete Closed Positions"], 1)
        self.assertEqual(roundoff_only.loc[0, "Incomplete Closed Positions"], 0)

    def test_realized_pnl_summary_exposes_closed_child_only_position_as_incomplete(
        self,
    ) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, closed_at)
            VALUES ('TQQQ', 'QQQ', 30, 1.5, '2026-01-02', 'buy-child-only',
                    'accepted', '2026-01-03T15:00:00Z')
            """
        )
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
            VALUES (?, 'sell-child-only', 1, 120)
            """,
            (int(cursor.lastrowid),),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)

    def test_realized_pnl_summary_rejects_orphan_sell_fill(self) -> None:
        self.conn.execute("DROP TRIGGER alpaca_managed_sell_fills_validate_parent_insert")
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
            VALUES (999, 'sell-orphan', 1, 120)
            """
        )

        with self.assertRaisesRegex(ValueError, "has no parent position"):
            build_alpaca_realized_pnl_summary(self.conn)

    def test_realized_pnl_summary_rejects_sell_order_owned_by_multiple_positions(
        self,
    ) -> None:
        position_ids: list[int] = []
        for index, symbol in enumerate(("TQQQ", "UPRO"), start=1):
            cursor = self.conn.execute(
                """
                INSERT INTO alpaca_managed_positions
                (symbol, alpaca_asset_id, signal_symbol, buy_rsi,
                 profit_target_multiple, buy_signal_date, buy_client_order_id,
                 buy_status, filled_qty, filled_avg_price, sold_qty, sold_value,
                 remaining_qty, closed_at)
                VALUES (?, ?, 'QQQ', 30, 1.5, '2026-01-02', ?, 'filled',
                        1, 100, 1, 150, 0, '2026-01-03T15:00:00Z')
                """,
                (symbol, f"asset-duplicate-report-{index}", f"buy-duplicate-report-{index}"),
            )
            position_ids.append(int(cursor.lastrowid))
        self.conn.execute("DROP INDEX leveraged_trader_alpaca_managed_sell_fills_alpaca_order_identity_unique")
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
            VALUES (?, 'sell-duplicate-report', 1, 150)
            """,
            ((position_ids[0],), (position_ids[1],)),
        )

        with self.assertRaisesRegex(ValueError, "belongs to multiple managed positions"):
            build_alpaca_realized_pnl_summary(self.conn)

    def test_realized_pnl_summary_uses_one_snapshot_across_orphan_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "realized-pnl-race.sqlite"
            with sqlite3.connect(db_path) as setup_conn:
                init_state_db(setup_conn)
                setup_conn.execute("PRAGMA journal_mode = WAL")
                setup_conn.execute("DROP TRIGGER alpaca_managed_sell_fills_validate_parent_insert")

            reader = sqlite3.connect(db_path, timeout=5.0)
            aggregation_started = threading.Event()
            writer_done = threading.Event()
            writer_failures: list[BaseException] = []
            snapshot_states: list[bool] = []

            def insert_orphan() -> None:
                try:
                    if not aggregation_started.wait(5.0):
                        raise TimeoutError("Reader did not reach the P/L aggregation barrier")
                    with sqlite3.connect(db_path, timeout=5.0) as writer:
                        writer.execute(
                            """
                            INSERT INTO alpaca_managed_sell_fills
                            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
                            VALUES (999, 'sell-concurrent-orphan', 1, 120)
                            """
                        )
                except BaseException as exc:
                    writer_failures.append(exc)
                finally:
                    writer_done.set()

            original_read_sql_query = pd.read_sql_query

            def read_after_writer(*args: object, **kwargs: object) -> pd.DataFrame:
                snapshot_states.append(reader.in_transaction)
                aggregation_started.set()
                if not writer_done.wait(5.0):
                    raise TimeoutError("Concurrent orphan writer did not commit")
                return original_read_sql_query(*args, **kwargs)

            writer_thread = threading.Thread(target=insert_orphan)
            writer_thread.start()
            try:
                with patch.object(
                    reports_module.pd,
                    "read_sql_query",
                    side_effect=read_after_writer,
                ):
                    summary = build_alpaca_realized_pnl_summary(reader)
                self.assertFalse(reader.in_transaction)
            finally:
                writer_thread.join(timeout=5.0)
                reader.close()

            if writer_thread.is_alive():
                self.fail("Concurrent orphan writer did not terminate")
            if writer_failures:
                raise writer_failures[0]
            self.assertEqual(snapshot_states, [True])
            self.assertEqual(summary.loc[0, "Closed Positions"], 0)

            with (
                sqlite3.connect(db_path) as current_reader,
                self.assertRaisesRegex(ValueError, "has no parent position"),
            ):
                build_alpaca_realized_pnl_summary(current_reader)

    def test_realized_pnl_summary_rejects_buy_fills_outside_immutable_intent(
        self,
    ) -> None:
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, buy_order_qty, buy_order_limit_price,
             filled_qty, filled_avg_price, sold_qty, sold_value, remaining_qty, closed_at)
            VALUES (?, 'QQQ', 30, 1.5, '2026-01-02', ?, 'filled', ?, ?, ?, ?, ?, ?, 0,
                    '2026-01-03T15:00:00Z')
            """,
            [
                (
                    "VALID",
                    "buy-valid-intent",
                    2.0,
                    100.0,
                    2.0,
                    100.0,
                    2.0,
                    240.0,
                ),
                (
                    "INVALID",
                    "buy-invalid-intent",
                    1.0,
                    100.0,
                    2.0,
                    1_000.0,
                    2.0,
                    300.0,
                ),
            ],
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 2)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Total Buy Cost"], 200.0)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 240.0)
        self.assertEqual(summary.loc[0, "Realized P/L"], 40.0)

    def test_realized_pnl_summary_casts_integer_fill_sums_before_aggregation(
        self,
    ) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES ('TQQQ', 'QQQ', 30, 1.5, '2026-01-02', 'buy-large-integer-fills',
                    'filled', 1e19, 1, 1e19, 1e19, 0, '2026-01-03T15:00:00Z')
            """
        )
        fill_value = 5_000_000_000_000_000_000
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
            VALUES (?, ?, ?, ?)
            """,
            [
                (int(cursor.lastrowid), "sell-large-1", fill_value, fill_value),
                (int(cursor.lastrowid), "sell-large-2", fill_value, fill_value),
            ],
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Total Buy Cost"], 1e19)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 1e19)

    def test_realized_pnl_summary_rejects_signed_child_fills_that_cancel_in_aggregate(self) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES ('TQQQ', 'QQQ', 30, 1.5, '2026-01-02', 'buy-signed-ledger',
                    'filled', 2, 100, 2, 240, 0, '2026-01-03T15:00:00Z')
            """
        )
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value)
            VALUES (?, ?, ?, ?)
            """,
            [
                (int(cursor.lastrowid), "sell-positive", 3.0, 360.0),
                (int(cursor.lastrowid), "sell-negative", -1.0, -120.0),
            ],
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

    def test_realized_pnl_summary_rejects_numeric_parent_values_stored_as_blobs(self) -> None:
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES ('TQQQ', 'QQQ', 30, 1.5, '2026-01-02', 'buy-blob-parent',
                    'filled', 2, 100, 2, 240, 0, '2026-01-03T15:00:00Z')
            """
        )
        self.conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET filled_qty = CAST(filled_qty AS BLOB),
                filled_avg_price = CAST(filled_avg_price AS BLOB),
                sold_qty = CAST(sold_qty AS BLOB),
                sold_value = CAST(sold_value AS BLOB),
                remaining_qty = CAST(remaining_qty AS BLOB)
            WHERE buy_client_order_id = 'buy-blob-parent'
            """
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

    def test_realized_pnl_summary_rejects_invalid_parent_sell_prices(self) -> None:
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             target_sell_price, sell_order_limit_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES ('TQQQ', 'QQQ', 30, 1.5, '2026-01-02', 'buy-invalid-parent-price',
                    'filled', 2, 100, 150, 150, 2, 240, 0, '2026-01-03T15:00:00Z')
            """
        )
        baseline = build_alpaca_realized_pnl_summary(self.conn)
        self.assertEqual(baseline.loc[0, "Complete Closed Positions"], 1)

        invalid_values = (
            ("target_sell_price", sqlite3.Binary(b"150")),
            ("target_sell_price", float("inf")),
            ("target_sell_price", -1.0),
            ("sell_order_limit_price", sqlite3.Binary(b"150")),
            ("sell_order_limit_price", float("inf")),
            ("sell_order_limit_price", -1.0),
        )
        for column, invalid_value in invalid_values:
            with self.subTest(column=column, invalid_value=invalid_value):
                self.conn.execute(
                    "UPDATE alpaca_managed_positions SET target_sell_price = 150, sell_order_limit_price = 150"
                )
                self.conn.execute(
                    f"UPDATE alpaca_managed_positions SET {column} = ?",
                    (invalid_value,),
                )

                summary = build_alpaca_realized_pnl_summary(self.conn)

                self.assertEqual(summary.loc[0, "Closed Positions"], 1)
                self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
                self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
                self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

    def test_realized_pnl_summary_rejects_sell_fill_below_immutable_limit(self) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES ('TQQQ', 'QQQ', 30, 1.5, '2026-01-02', 'buy-below-limit',
                    'filled', 1, 100, 1, 50, 0, '2026-01-03T15:00:00Z')
            """
        )
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value,
             submitted_qty, submitted_limit_price)
            VALUES (?, 'sell-below-limit', 1, 50, 1, 100)
            """,
            (int(cursor.lastrowid),),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

    def test_realized_pnl_summary_counts_corrupt_executed_rows_but_not_unfilled_intents(self) -> None:
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f"CORRUPT{index}",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    f"rsi-buy-CORRUPT{index}-20260102",
                    buy_status,
                    filled_qty,
                    100.0,
                    "2026-01-03T15:00:00Z",
                )
                for index, (filled_qty, buy_status) in enumerate(
                    ((None, "filled"), (0.0, "accepted"), (-1.0, "rejected")),
                    start=1,
                )
            ]
            + [
                (
                    "UNFILLED",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    "rsi-buy-UNFILLED-20260102",
                    "rejected",
                    None,
                    None,
                    "2026-01-03T15:00:00Z",
                )
            ],
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 3)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 3)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

    def test_realized_pnl_summary_rejects_mixed_aggregate_and_legacy_sell_metadata(self) -> None:
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sell_filled_qty, sell_filled_avg_price, sold_qty, sold_value,
             remaining_qty, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "MIXEDQTY",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    "rsi-buy-MIXEDQTY-20260102",
                    "filled",
                    2.0,
                    100.0,
                    1.0,
                    120.0,
                    2.0,
                    0.0,
                    0.0,
                    "2026-01-03T15:00:00Z",
                ),
                (
                    "MIXEDVALUE",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    "rsi-buy-MIXEDVALUE-20260102",
                    "filled",
                    2.0,
                    100.0,
                    2.0,
                    120.0,
                    0.0,
                    240.0,
                    0.0,
                    "2026-01-03T15:00:00Z",
                ),
            ],
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 2)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 2)
        self.assertEqual(summary.loc[0, "Total Buy Cost"], 0.0)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 0.0)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

    def test_realized_pnl_summary_rejects_nonfinite_closed_fill_value(self) -> None:
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                "2026-01-02",
                "rsi-buy-TQQQ-20260102",
                "filled",
                1.0,
                100.0,
                1.0,
                float("inf"),
                0.0,
                "2026-01-03T15:00:00Z",
            ),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)

    def test_realized_pnl_summary_rejects_overflowing_per_position_cost(self) -> None:
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                "2026-01-02",
                "rsi-buy-TQQQ-overflow",
                "filled",
                1e200,
                1e200,
                1e200,
                1e300,
                0.0,
                "2026-01-03T15:00:00Z",
            ),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertTrue(
            np.isfinite(
                summary[
                    [
                        "Total Buy Cost",
                        "Total Sell Value",
                        "Realized P/L",
                        "Realized P/L %",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        )

    def test_realized_pnl_summary_rejects_overflowing_aggregate(self) -> None:
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_positions
            (workflow, symbol, signal_symbol, buy_rsi, profit_target_multiple,
             buy_signal_date, buy_client_order_id, buy_status, filled_qty,
             filled_avg_price, sold_qty, sold_value, remaining_qty, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "Long",
                    f"EXTREME{index}",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    f"rsi-buy-EXTREME{index}-overflow",
                    "filled",
                    1.0,
                    9e307,
                    1.0,
                    9e307,
                    0.0,
                    "2026-01-03T15:00:00Z",
                )
                for index in range(2)
            ],
        )

        summary = build_alpaca_realized_pnl_summary(self.conn, include_workflow=True)

        self.assertEqual(summary.loc[0, "Workflow"], "Long")
        self.assertEqual(summary.loc[0, "Closed Positions"], 2)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 2)
        self.assertTrue(
            np.isfinite(
                summary[
                    [
                        "Total Buy Cost",
                        "Total Sell Value",
                        "Realized P/L",
                        "Realized P/L %",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        )

    def test_realized_pnl_summary_uses_scale_aware_fractional_quantity_tolerance(
        self,
    ) -> None:
        self.conn.executemany(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sold_qty, sold_value, remaining_qty, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "MISMATCHED-SOLD",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    "rsi-buy-MISMATCHED-SOLD",
                    "filled",
                    1e-9,
                    100.0,
                    9e-9,
                    9e-7,
                    0.0,
                    "2026-01-03T15:00:00Z",
                ),
                (
                    "MISMATCHED-REMAINING",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    "rsi-buy-MISMATCHED-REMAINING",
                    "filled",
                    1e-9,
                    100.0,
                    1e-9,
                    1.2e-7,
                    9e-9,
                    "2026-01-03T15:00:00Z",
                ),
                (
                    "MATCHED-FRACTIONAL",
                    "QQQ",
                    30.0,
                    1.5,
                    "2026-01-02",
                    "rsi-buy-MATCHED-FRACTIONAL",
                    "filled",
                    1e-9,
                    100.0,
                    1e-9,
                    1.2e-7,
                    0.0,
                    "2026-01-03T15:00:00Z",
                ),
            ],
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 3)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 2)
        self.assertAlmostEqual(summary.loc[0, "Total Buy Cost"], 1e-7)
        self.assertAlmostEqual(summary.loc[0, "Total Sell Value"], 1.2e-7)

    def test_realized_pnl_summary_includes_position_closed_with_shared_quantity_tolerance(
        self,
    ) -> None:
        sold_qty = float(np.nextafter(1.0, 0.0))
        remaining_qty = 1.0 - sold_qty
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, filled_qty, filled_avg_price,
             sell_client_order_id, sell_alpaca_order_id, sell_status,
             sell_filled_qty, sell_filled_avg_price, sold_qty, sold_value,
             remaining_qty)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "TQQQ",
                "QQQ",
                30.0,
                1.5,
                "2026-01-02",
                "rsi-buy-TQQQ-shared-tolerance",
                "filled",
                1.0,
                100.0,
                "rsi-exit-TQQQ-shared-tolerance",
                "sell-shared-tolerance",
                "filled",
                sold_qty,
                120.0 / sold_qty,
                sold_qty,
                120.0,
                remaining_qty,
            ),
        )

        closed = close_alpaca_managed_position_if_current_and_complete(
            self.conn,
            1,
            expected_sell_client_order_id="rsi-exit-TQQQ-shared-tolerance",
            expected_sell_alpaca_order_id="sell-shared-tolerance",
            closed_at="2026-01-03T15:00:00Z",
        )
        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertTrue(closed)
        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Total Buy Cost"], 100.0)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 120.0)
        self.assertEqual(summary.loc[0, "Realized P/L"], 20.0)

    def test_realized_pnl_rejects_quantity_dust_with_material_residual_notional(
        self,
    ) -> None:
        quantity = 1e-13
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, buy_order_qty, buy_order_limit_price,
             filled_qty, filled_avg_price, sold_qty, sold_value, remaining_qty,
             closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "VALUEDUST",
                "QQQ",
                30.0,
                1.5,
                "2026-01-02",
                "buy-value-dust-report",
                "filled",
                2.0 * quantity,
                1e15,
                2.0 * quantity,
                1e15,
                quantity,
                150.0,
                quantity,
                "2026-01-03T15:00:00Z",
            ),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Total Buy Cost"], 0.0)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 0.0)

    def test_realized_pnl_summary_rejects_nonfinite_child_unit_price(self) -> None:
        quantity = float(np.nextafter(0.0, np.inf))
        cursor = self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, buy_order_qty, buy_order_limit_price,
             filled_qty, filled_avg_price, sold_qty, sold_value, remaining_qty, closed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "SUBNORMAL",
                "QQQ",
                30.0,
                1.5,
                "2026-01-02",
                "buy-subnormal-unit-price",
                "filled",
                quantity,
                1e308,
                quantity,
                1e308,
                quantity,
                1.0,
                0.0,
                "2026-01-03T15:00:00Z",
            ),
        )
        self.assertTrue(np.isinf(1.0 / quantity))
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value,
             submitted_qty, submitted_limit_price)
            VALUES (?, 'sell-subnormal-unit-price', ?, 1, ?, 1)
            """,
            (int(cursor.lastrowid), quantity, quantity),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Total Buy Cost"], 0.0)
        self.assertEqual(summary.loc[0, "Total Sell Value"], 0.0)

    def test_realized_pnl_summary_rejects_noncanonical_child_order_identity(self) -> None:
        cursor = self.conn.execute(
            """
            INSERT INTO alpaca_managed_positions
            (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
             buy_client_order_id, buy_status, buy_order_qty, buy_order_limit_price,
             filled_qty, filled_avg_price, sold_qty, sold_value, remaining_qty, closed_at)
            VALUES ('BADID', 'QQQ', 30, 1.5, '2026-01-02',
                    'buy-noncanonical-child-id', 'filled', 1, 100,
                    1, 100, 1, 120, 0, '2026-01-03T15:00:00Z')
            """
        )
        self.conn.execute(
            """
            INSERT INTO alpaca_managed_sell_fills
            (managed_position_id, alpaca_order_id, filled_qty, filled_value,
             submitted_qty, submitted_limit_price)
            VALUES (?, ' sell-noncanonical', 1, 120, 1, 120)
            """,
            (int(cursor.lastrowid),),
        )

        summary = build_alpaca_realized_pnl_summary(self.conn)

        self.assertEqual(summary.loc[0, "Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Complete Closed Positions"], 0)
        self.assertEqual(summary.loc[0, "Incomplete Closed Positions"], 1)
        self.assertEqual(summary.loc[0, "Realized P/L"], 0.0)


if __name__ == "__main__":
    unittest.main()
