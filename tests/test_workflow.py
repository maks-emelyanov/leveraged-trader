from __future__ import annotations

import asyncio
import io
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from rich.console import Console

from leveraged_trader.benchmark import WorkflowPhaseTimings, WorkflowTimer
from leveraged_trader.config import AlpacaOrderConfig, BacktestConfig, UniverseConfig
from leveraged_trader.output import WorkflowReporter
from leveraged_trader.storage import (
    _synchronize_market_data_history,
    init_state_db,
    save_alpaca_managed_buy_order,
)
from leveraged_trader.workflow import (
    AssetRunJob,
    AssetRunPlan,
    AssetRunResult,
    PreparedAssetRun,
    WorkflowRunError,
    _build_reports_for_db,
    _prepare_asset_run,
    _prepare_workflow_asset,
    _process_asset_grid_for_db,
    _reconcile_alpaca_managed_positions_for_db,
    _run_asset_pipeline,
    _state_connection,
    _terminal_alpaca_display_results,
    _validate_database_path,
    _validate_optimization_grids,
    _validate_workflow_mode,
    _workflow_run_lock,
    _WorkflowStrategySession,
    _write_workflow_outputs,
    run_resumable_optimizations_async,
)


class WorkflowAsyncTests(unittest.TestCase):
    def test_workflow_database_path_rejects_nonpersistent_values(self) -> None:
        for db_path, message in [
            ("", "nonempty filesystem path"),
            (":memory:", "persistent filesystem path"),
        ]:
            with (
                self.subTest(db_path=db_path),
                self.assertRaisesRegex(ValueError, message),
            ):
                _validate_database_path(db_path)

    def test_nonpersistent_database_is_rejected_before_filesystem_changes(self) -> None:
        for db_path in ["", ":memory:"]:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(db_path=db_path):
                output_dir = Path(tmp) / "reports"
                with self.assertRaises(ValueError):
                    asyncio.run(
                        run_resumable_optimizations_async(
                            mode="update",
                            db_path=db_path,
                            base_cfg=BacktestConfig(),
                            universe_cfg=UniverseConfig(),
                            buy_rsi_values=[30.0],
                            profit_target_values=[1.5],
                            alpaca_cfg=AlpacaOrderConfig(),
                            output_dir=str(output_dir),
                        )
                    )

                self.assertFalse(output_dir.exists())

    def test_invalid_endpoint_and_grids_are_rejected_before_filesystem_changes(self) -> None:
        cases = [
            {
                "buy_rsi_values": [],
                "short_buy_rsi_values": [70.0],
                "profit_target_values": [1.5],
                "alpaca_cfg": AlpacaOrderConfig(),
            },
            {
                "buy_rsi_values": [30.0],
                "short_buy_rsi_values": [],
                "profit_target_values": [1.5],
                "alpaca_cfg": AlpacaOrderConfig(),
            },
            {
                "buy_rsi_values": [30.0],
                "short_buy_rsi_values": [70.0],
                "profit_target_values": [1.0],
                "alpaca_cfg": AlpacaOrderConfig(),
            },
            {
                "buy_rsi_values": [30.0],
                "short_buy_rsi_values": [70.0],
                "profit_target_values": [1.5],
                "alpaca_cfg": AlpacaOrderConfig(
                    enabled=True,
                    base_url="https://api.alpaca.markets",
                ),
            },
        ]
        for case in cases:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(case=case):
                db_path = Path(tmp) / "state.sqlite"
                output_dir = Path(tmp) / "reports"
                with self.assertRaises(ValueError):
                    asyncio.run(
                        run_resumable_optimizations_async(
                            mode="update",
                            db_path=str(db_path),
                            base_cfg=BacktestConfig(),
                            universe_cfg=UniverseConfig(),
                            buy_rsi_values=case["buy_rsi_values"],
                            short_buy_rsi_values=case["short_buy_rsi_values"],
                            profit_target_values=case["profit_target_values"],
                            alpaca_cfg=case["alpaca_cfg"],
                            output_dir=str(output_dir),
                        )
                    )

                self.assertFalse(Path(f"{db_path}.lock").exists())
                self.assertFalse(output_dir.exists())

    def test_workflow_mode_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode must be 'update' or 'rebuild'"):
            _validate_workflow_mode("refresh")

    def test_invalid_workflow_mode_is_rejected_before_lock_files_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.db"
            output_dir = Path(tmp) / "reports"
            with self.assertRaisesRegex(ValueError, "mode must be 'update' or 'rebuild'"):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="refresh",
                        db_path=str(db_path),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(output_dir),
                    )
                )

            self.assertFalse(Path(f"{db_path}.lock").exists())
            self.assertFalse(output_dir.exists())

    def test_workflow_uses_legacy_compatible_database_lock_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.db"
            output_dir = Path(tmp) / "reports"
            with _workflow_run_lock(str(db_path), str(output_dir)):
                self.assertTrue(Path(f"{db_path}.lock").exists())
                self.assertFalse(Path(f"{db_path}.leveraged-trader.lock").exists())

    def test_workflow_lock_canonicalizes_symlinked_database_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real_db_path = Path(tmp) / "real.sqlite"
            real_db_path.touch()
            alias_db_path = Path(tmp) / "alias.sqlite"
            alias_db_path.symlink_to(real_db_path)
            with (
                _workflow_run_lock(str(real_db_path), str(Path(tmp) / "outputs-one")),
                self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                _workflow_run_lock(str(alias_db_path), str(Path(tmp) / "outputs-two")),
            ):
                self.fail("symlinked database unexpectedly acquired a separate lock")

    def test_symlinked_database_acquires_legacy_and_canonical_lock_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real_db_path = Path(tmp) / "real.sqlite"
            real_db_path.touch()
            alias_db_path = Path(tmp) / "alias.sqlite"
            alias_db_path.symlink_to(real_db_path)
            with _workflow_run_lock(str(alias_db_path), str(Path(tmp) / "outputs")):
                self.assertTrue(Path(f"{alias_db_path}.lock").exists())
                self.assertTrue(Path(f"{real_db_path}.lock").exists())

    def test_workflow_rejects_hardlinked_database_without_creating_locks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_db_path = Path(tmp) / "first.sqlite"
            first_db_path.touch()
            second_db_path = Path(tmp) / "second.sqlite"
            os.link(first_db_path, second_db_path)
            output_dir = Path(tmp) / "outputs"

            with (
                self.assertRaisesRegex(WorkflowRunError, "multiple hard links"),
                _workflow_run_lock(str(second_db_path), str(output_dir)),
            ):
                self.fail("hardlinked database unexpectedly acquired workflow locks")

            self.assertFalse(Path(f"{first_db_path}.lock").exists())
            self.assertFalse(Path(f"{second_db_path}.lock").exists())
            self.assertFalse(output_dir.exists())

    def test_optimization_grid_rejects_out_of_range_values(self) -> None:
        for buy_values, target_values, message in [
            ([-1.0], [1.1], "between 0 and 100"),
            ([101.0], [1.1], "between 0 and 100"),
            ([30.0], [1.0], "greater than 1.0"),
            ([30.0], [0.5], "greater than 1.0"),
        ]:
            with (
                self.subTest(buy_values=buy_values, target_values=target_values),
                self.assertRaisesRegex(ValueError, message),
            ):
                _validate_optimization_grids(buy_values, target_values)

    def test_workflow_lock_rejects_overlapping_run_for_same_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            first_output = str(Path(tmp) / "outputs-one")
            second_output = str(Path(tmp) / "outputs-two")
            with (
                _workflow_run_lock(db_path, first_output),
                self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                _workflow_run_lock(db_path, second_output),
            ):
                self.fail("overlapping workflow unexpectedly acquired the lock")

    def test_workflow_lock_rejects_overlapping_run_for_same_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = str(Path(tmp) / "outputs")
            with (
                _workflow_run_lock(str(Path(tmp) / "state-one.sqlite"), output_dir),
                self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                _workflow_run_lock(str(Path(tmp) / "state-two.sqlite"), output_dir),
            ):
                self.fail("overlapping workflow unexpectedly acquired the output lock")

    def test_workflow_lock_cleans_up_after_non_contention_acquisition_error(self) -> None:
        first_lock_file = Mock()
        first_lock_file.fileno.return_value = 42
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        tmp = temporary_directory.name
        with (
            patch.object(Path, "open", side_effect=[first_lock_file, PermissionError("denied")]),
            patch("leveraged_trader.workflow.fcntl.flock") as mock_flock,
            self.assertRaisesRegex(PermissionError, "denied"),
            _workflow_run_lock(
                str(Path(tmp) / "state.sqlite"),
                str(Path(tmp) / "outputs"),
            ),
        ):
            self.fail("workflow unexpectedly acquired both locks")

        mock_flock.assert_called_once()
        first_lock_file.close.assert_called_once_with()

    @patch("leveraged_trader.workflow.reconcile_alpaca_managed_positions")
    @patch("leveraged_trader.workflow.migrate_alpaca_managed_position_symbols")
    def test_reconciliation_output_mentions_only_migrations_from_current_run(
        self,
        mock_migrate: Mock,
        mock_reconcile: Mock,
    ) -> None:
        columns = [
            "Position ID",
            "Workflow",
            "Asset",
            "Action",
            "Status",
            "Buy Client Order ID",
            "Sell Client Order ID",
            "Qty",
            "Limit Price",
            "Alpaca Order ID",
            "Message",
        ]
        mock_reconcile.return_value = pd.DataFrame(columns=columns)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)
                save_alpaca_managed_buy_order(
                    conn,
                    symbol="SATG",
                    signal_symbol="SATS",
                    buy_rsi=30,
                    profit_target_multiple=1.1,
                    buy_signal_date="2026-06-18",
                    buy_client_order_id="rsi-buy-SATG-20260618",
                    buy_alpaca_order_id="buy-1",
                    buy_submitted_at="2026-06-22T13:30:00Z",
                    buy_status="filled",
                )

            def migrate(conn: sqlite3.Connection, _cfg: AlpacaOrderConfig) -> dict[str, str]:
                conn.execute("UPDATE alpaca_managed_positions SET symbol = 'ECHX', alpaca_asset_id = 'asset-echo'")
                conn.commit()
                return {"SATG": "ECHX"}

            mock_migrate.side_effect = migrate
            result = _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())
            self.assertEqual(result["Status"].tolist(), ["symbol_migrated"])
            self.assertIn("SATG to ECHX", result.loc[0, "Message"])

            mock_migrate.side_effect = None
            mock_migrate.return_value = {}
            result = _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())
            self.assertTrue(result.empty)

    @staticmethod
    def _symbol_history(symbol: str, offset: float = 0.0) -> pd.DataFrame:
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        close = [100.0 + offset + index for index in range(len(dates))]
        return pd.DataFrame(
            {
                f"{symbol}_Open": close,
                f"{symbol}_High": [value + 1.0 for value in close],
                f"{symbol}_Low": [value - 1.0 for value in close],
                f"{symbol}_Close": close,
                f"{symbol}_Volume": [1_000_000.0] * len(dates),
            },
            index=dates,
        )

    def _process_session_asset(
        self,
        session: _WorkflowStrategySession,
        db_path: str,
        asset_symbol: str,
        signal_symbol: str,
        asset_history: pd.DataFrame,
        signal_history: pd.DataFrame,
        risk_free_history: pd.DataFrame,
    ) -> None:
        _process_asset_grid_for_db(
            db_path,
            pd.concat([asset_history, signal_history, risk_free_history], axis=1),
            asset_history,
            signal_history,
            risk_free_history,
            BacktestConfig(rsi_period=3),
            asset_symbol,
            signal_symbol,
            [30.0],
            [1.5],
            False,
            strategy_session=session,
        )

    def _run_workflow_with_grids(
        self,
        *,
        buy_rsi_values: list[float],
        profit_target_values: list[float],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(
                run_resumable_optimizations_async(
                    mode="update",
                    db_path=str(Path(tmp) / "unused.sqlite"),
                    base_cfg=BacktestConfig(),
                    universe_cfg=UniverseConfig(),
                    buy_rsi_values=buy_rsi_values,
                    profit_target_values=profit_target_values,
                    alpaca_cfg=AlpacaOrderConfig(),
                    output_dir=str(Path(tmp) / "outputs"),
                )
            )

    def test_empty_buy_rsi_grid_is_rejected_before_workflow_starts(self) -> None:
        with self.assertRaisesRegex(ValueError, "buy_rsi_values must not be empty"):
            self._run_workflow_with_grids(buy_rsi_values=[], profit_target_values=[1.5])

    def test_empty_profit_target_grid_is_rejected_before_workflow_starts(self) -> None:
        with self.assertRaisesRegex(ValueError, "profit_target_values must not be empty"):
            self._run_workflow_with_grids(buy_rsi_values=[30.0], profit_target_values=[])

    def test_non_finite_grid_value_is_rejected_before_workflow_starts(self) -> None:
        with self.assertRaisesRegex(ValueError, "profit_target_values must contain only finite numeric values"):
            self._run_workflow_with_grids(buy_rsi_values=[30.0], profit_target_values=[float("nan")])

    def test_immediate_state_transactions_serialize_independent_writers(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        second_acquired = threading.Event()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)

            def first_writer() -> None:
                with _state_connection(db_path, immediate=True) as conn:
                    conn.execute("UPDATE strategy_state_generation SET generation = generation + 1 WHERE id = 1")
                    first_started.set()
                    release_first.wait(timeout=2)

            def second_writer() -> None:
                first_started.wait(timeout=2)
                with _state_connection(db_path, immediate=True):
                    second_acquired.set()

            first = threading.Thread(target=first_writer)
            second = threading.Thread(target=second_writer)
            first.start()
            self.assertTrue(first_started.wait(timeout=1))
            second.start()
            self.assertFalse(second_acquired.wait(timeout=0.05))
            release_first.set()
            self.assertTrue(second_acquired.wait(timeout=2))
            first.join(timeout=2)
            second.join(timeout=2)

    def test_asset_transaction_separates_grid_compute_from_db_sync(self) -> None:
        phase_timings = WorkflowPhaseTimings()

        def fake_process_asset_grid(*_args: object, **kwargs: object) -> None:
            observer = kwargs["grid_compute_observer"]
            observer(2.0)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("leveraged_trader.workflow.process_asset_grid", side_effect=fake_process_asset_grid),
            patch("leveraged_trader.workflow.time") as mock_time,
        ):
            mock_time.perf_counter.side_effect = [10.0, 15.0]
            _process_asset_grid_for_db(
                str(Path(tmp) / "state.sqlite"),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                BacktestConfig(),
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                False,
                phase_timings,
            )

        snapshot = phase_timings.snapshot()
        self.assertEqual(snapshot.grid_compute_seconds, 2.0)
        self.assertEqual(snapshot.db_sync_seconds, 3.0)

    def test_strategy_session_synchronizes_shared_history_instances_once(self) -> None:
        signal_history = self._symbol_history("QQQ", 10.0)
        risk_free_history = self._symbol_history("^IRX", -95.0)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)
            session = _WorkflowStrategySession(db_path)
            try:
                with patch(
                    "leveraged_trader.storage._synchronize_market_data_history",
                    wraps=_synchronize_market_data_history,
                ) as synchronize:
                    self._process_session_asset(
                        session,
                        db_path,
                        "TQQQ",
                        "QQQ",
                        self._symbol_history("TQQQ"),
                        signal_history,
                        risk_free_history,
                    )
                    self._process_session_asset(
                        session,
                        db_path,
                        "UPRO",
                        "QQQ",
                        self._symbol_history("UPRO", 20.0),
                        signal_history,
                        risk_free_history,
                    )
            finally:
                session.close()

        self.assertEqual(
            [call.args[2] for call in synchronize.call_args_list],
            ["TQQQ", "QQQ", "^IRX", "UPRO"],
        )

    def test_strategy_session_resynchronizes_a_different_history_instance(self) -> None:
        signal_history = self._symbol_history("QQQ", 10.0)
        risk_free_history = self._symbol_history("^IRX", -95.0)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)
            session = _WorkflowStrategySession(db_path)
            try:
                self._process_session_asset(
                    session,
                    db_path,
                    "TQQQ",
                    "QQQ",
                    self._symbol_history("TQQQ"),
                    signal_history,
                    risk_free_history,
                )
                with patch(
                    "leveraged_trader.storage._synchronize_market_data_history",
                    wraps=_synchronize_market_data_history,
                ) as synchronize:
                    self._process_session_asset(
                        session,
                        db_path,
                        "UPRO",
                        "QQQ",
                        self._symbol_history("UPRO", 20.0),
                        signal_history.copy(),
                        risk_free_history,
                    )
            finally:
                session.close()

        self.assertEqual(
            [call.args[2] for call in synchronize.call_args_list],
            ["UPRO", "QQQ"],
        )

    def test_strategy_session_external_commit_invalidates_shared_history_cache(self) -> None:
        signal_history = self._symbol_history("QQQ", 10.0)
        risk_free_history = self._symbol_history("^IRX", -95.0)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)
            session = _WorkflowStrategySession(db_path)
            try:
                self._process_session_asset(
                    session,
                    db_path,
                    "TQQQ",
                    "QQQ",
                    self._symbol_history("TQQQ"),
                    signal_history,
                    risk_free_history,
                )
                with sqlite3.connect(db_path) as external:
                    external.execute("UPDATE strategy_state_generation SET generation = generation + 1 WHERE id = 1")
                with patch(
                    "leveraged_trader.storage._synchronize_market_data_history",
                    wraps=_synchronize_market_data_history,
                ) as synchronize:
                    self._process_session_asset(
                        session,
                        db_path,
                        "UPRO",
                        "QQQ",
                        self._symbol_history("UPRO", 20.0),
                        signal_history,
                        risk_free_history,
                    )
            finally:
                session.close()

        self.assertEqual(
            [call.args[2] for call in synchronize.call_args_list],
            ["UPRO", "QQQ", "^IRX"],
        )

    def test_strategy_session_does_not_cache_rolled_back_histories(self) -> None:
        asset_history = self._symbol_history("TQQQ")
        signal_history = self._symbol_history("QQQ", 10.0)
        risk_free_history = self._symbol_history("^IRX", -95.0)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)
            session = _WorkflowStrategySession(db_path)
            try:
                with (
                    patch(
                        "leveraged_trader.storage._synchronize_market_data_history",
                        wraps=_synchronize_market_data_history,
                    ) as synchronize,
                    patch(
                        "leveraged_trader.storage.run_grid_summary",
                        side_effect=RuntimeError("grid failed"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "grid failed"),
                ):
                    self._process_session_asset(
                        session,
                        db_path,
                        "TQQQ",
                        "QQQ",
                        asset_history,
                        signal_history,
                        risk_free_history,
                    )

                with patch(
                    "leveraged_trader.storage._synchronize_market_data_history",
                    wraps=_synchronize_market_data_history,
                ) as retry_synchronize:
                    self._process_session_asset(
                        session,
                        db_path,
                        "TQQQ",
                        "QQQ",
                        asset_history,
                        signal_history,
                        risk_free_history,
                    )
            finally:
                session.close()

        self.assertEqual(len(synchronize.call_args_list), 3)
        self.assertEqual(
            [call.args[2] for call in retry_synchronize.call_args_list],
            ["TQQQ", "QQQ", "^IRX"],
        )

    def test_cached_signal_correction_still_rebuilds_later_dependents(self) -> None:
        tqqq_history = self._symbol_history("TQQQ")
        upro_history = self._symbol_history("UPRO", 20.0)
        signal_history = self._symbol_history("QQQ", 10.0)
        risk_free_history = self._symbol_history("^IRX", -95.0)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)
            session = _WorkflowStrategySession(db_path)
            try:
                self._process_session_asset(
                    session,
                    db_path,
                    "TQQQ",
                    "QQQ",
                    tqqq_history,
                    signal_history,
                    risk_free_history,
                )
                self._process_session_asset(
                    session,
                    db_path,
                    "UPRO",
                    "QQQ",
                    upro_history,
                    signal_history,
                    risk_free_history,
                )
                corrected_signal = signal_history.copy()
                corrected_signal.loc[corrected_signal.index[10], "QQQ_Close"] += 5.0
                with patch(
                    "leveraged_trader.storage._synchronize_market_data_history",
                    wraps=_synchronize_market_data_history,
                ) as synchronize:
                    self._process_session_asset(
                        session,
                        db_path,
                        "TQQQ",
                        "QQQ",
                        tqqq_history.copy(),
                        corrected_signal,
                        risk_free_history,
                    )
                    self._process_session_asset(
                        session,
                        db_path,
                        "UPRO",
                        "QQQ",
                        upro_history.copy(),
                        corrected_signal,
                        risk_free_history,
                    )
            finally:
                session.close()

            with sqlite3.connect(db_path) as conn:
                states = conn.execute(
                    """
                    SELECT asset_symbol
                    FROM strategy_state
                    WHERE signal_symbol = 'QQQ'
                    ORDER BY asset_symbol
                    """
                ).fetchall()

        self.assertEqual([call.args[2] for call in synchronize.call_args_list].count("QQQ"), 1)
        self.assertEqual(states, [("TQQQ",), ("UPRO",)])

    def test_download_executor_is_isolated_and_state_processing_is_serialized(self) -> None:
        state_active = 0
        max_state_active = 0
        phase_timings = WorkflowPhaseTimings()
        download_threads: set[str] = set()
        state_threads: set[str] = set()

        def history(symbol: str) -> pd.DataFrame:
            index = pd.to_datetime(["2026-01-02"])
            return pd.DataFrame(
                {
                    f"{symbol}_Open": [100.0],
                    f"{symbol}_High": [101.0],
                    f"{symbol}_Low": [99.0],
                    f"{symbol}_Close": [100.0],
                    f"{symbol}_Volume": [1_000_000],
                },
                index=index,
            )

        def fake_prepare_asset_run(*args: object, **_kwargs: object) -> AssetRunPlan:
            download_threads.add(threading.current_thread().name)
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        def fake_load_strategy_data(**kwargs: object) -> pd.DataFrame:
            download_threads.add(threading.current_thread().name)
            return history(str(kwargs["asset_symbol"]))

        def fake_load_symbol_history(symbol: str, **_kwargs: object) -> pd.DataFrame:
            download_threads.add(threading.current_thread().name)
            return history(symbol)

        def fake_load_risk_free_history(**_kwargs: object) -> pd.DataFrame:
            download_threads.add(threading.current_thread().name)
            return history("^IRX")

        def fake_process_asset_grid(*_args: object, **_kwargs: object) -> None:
            nonlocal state_active, max_state_active
            state_threads.add(threading.current_thread().name)
            state_active += 1
            max_state_active = max(max_state_active, state_active)
            state_active -= 1

        async def run() -> list[AssetRunResult]:
            return await _run_asset_pipeline(
                jobs=[AssetRunJob(index, symbol, symbol) for index, symbol in enumerate(["AAA", "BBB"], start=1)],
                concurrency=2,
                db_path="state.sqlite",
                mode="update",
                base_cfg=BacktestConfig(),
                tradier_cfg=None,
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
                asset_progress=None,
                phase_timings=phase_timings,
            )

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=fake_prepare_asset_run),
            patch("leveraged_trader.workflow.load_strategy_data", side_effect=fake_load_strategy_data),
            patch("leveraged_trader.workflow.load_symbol_history", side_effect=fake_load_symbol_history),
            patch("leveraged_trader.workflow.load_signal_history", side_effect=fake_load_symbol_history) as mock_signal,
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                side_effect=fake_load_risk_free_history,
            ) as mock_risk_free,
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=fake_process_asset_grid),
        ):
            results = asyncio.run(run())

        self.assertEqual(max_state_active, 1)
        self.assertTrue(download_threads)
        self.assertTrue(all(name.startswith("workflow-download") for name in download_threads))
        self.assertTrue(state_threads)
        self.assertTrue(all(name.startswith("workflow-strategy") for name in state_threads))
        self.assertEqual(mock_signal.call_count, 2)
        self.assertEqual(mock_risk_free.call_count, 1)
        self.assertEqual([result.status for result in results], ["done", "done"])

    def test_startup_serializes_reconciliation_before_universe_writes(self) -> None:
        events: list[str] = []

        async def immediate_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        def reconcile(*_args: object) -> pd.DataFrame:
            events.append("reconcile")
            return pd.DataFrame(columns=["Action"])

        def load_universe(*_args: object) -> pd.DataFrame:
            self.assertEqual(events, ["reconcile"])
            events.append("universe")
            return pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])

        async def completed_asset_pipeline(**_kwargs: object) -> list[AssetRunResult]:
            return [
                AssetRunResult(
                    workflow_idx=1,
                    asset_symbol="TQQQ",
                    signal_symbol="QQQ",
                    action="Updating",
                    rows_processed=1,
                    status="done",
                    message="Processed 1 row",
                )
            ]

        with tempfile.TemporaryDirectory() as tmp:
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow.asyncio.to_thread", new=immediate_to_thread),
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch("leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db", side_effect=reconcile),
                patch("leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db", side_effect=load_universe),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=completed_asset_pipeline),
                patch(
                    "leveraged_trader.workflow._build_reports_for_db",
                    return_value=(
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                    ),
                ),
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    return_value=pd.DataFrame(columns=["Status"]),
                ),
                patch("leveraged_trader.workflow._load_alpaca_managed_positions_for_db", return_value=pd.DataFrame()),
                patch("leveraged_trader.workflow._write_workflow_outputs") as mock_write_outputs,
                patch("leveraged_trader.workflow.time") as mock_time,
            ):
                mock_time.perf_counter.side_effect = [float(value) for value in range(1, 9)]
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(Path(tmp) / "outputs"),
                        reporter=reporter,
                    )
                )

        self.assertEqual(events, ["reconcile", "universe"])
        phase_snapshot = mock_write_outputs.call_args.kwargs["phase_timings"].snapshot()
        self.assertEqual(phase_snapshot.report_generation_seconds, 2.0)
        self.assertEqual(phase_snapshot.alpaca_seconds, 2.0)

    def test_update_preflight_fetches_full_history_for_revision_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)
            with patch("leveraged_trader.workflow.strategy_state_matches_config", return_value=True):
                plan = _prepare_asset_run(
                    db_path,
                    "update",
                    BacktestConfig(),
                    "TQQQ",
                    "QQQ",
                    [30.0],
                    [1.5],
                )

        self.assertFalse(plan.rebuild)
        self.assertIsNone(plan.start)

    def test_report_build_excludes_assets_not_processed_in_current_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)
            workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])

            optimization_summary, _curves, buy_signals, _eligible, sell_signals, _pnl = _build_reports_for_db(
                db_path,
                workflow_assets,
                BacktestConfig(),
                processed_asset_pairs=set(),
            )

        self.assertTrue(optimization_summary.empty)
        self.assertTrue(buy_signals.empty)
        self.assertTrue(sell_signals.empty)

    def test_report_build_adds_workflow_and_side_prefixed_curve_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with sqlite3.connect(db_path) as conn:
                init_state_db(conn)
                conn.execute(
                    """
                    INSERT INTO strategy_summary
                    (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, start_date, end_date,
                     trading_days, trades_executed, total_return, cagr, annualized_vol, sharpe,
                     kelly_fraction, max_drawdown, hit_rate, first_equity, last_equity, running_max_equity,
                     return_count, return_sum, return_sum_squares, excess_return_count, excess_return_sum,
                     excess_return_sum_squares, positive_return_count)
                    VALUES
                    ('SQQQ', 'QQQ', 70.0, 1.5, '2026-01-02', '2026-01-05',
                     2, 2, 0.02, 0.10, 0.20, 1.20,
                     0.30, -0.01, 0.50, 100000.0, 102000.0, 102000.0,
                     2, 0.02, 0.0004, 2, 0.02, 0.0004, 1)
                    """
                )
                conn.executemany(
                    """
                    INSERT INTO strategy_equity
                    (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, date, equity,
                     daily_return, risk_free_return, in_position, action_executed, pending_action, trades_executed)
                    VALUES ('SQQQ', 'QQQ', 70.0, 1.5, ?, ?, ?, 0.0, ?, ?, ?, ?)
                    """,
                    [
                        ("2026-01-02", 100000.0, 0.0, 0, "none", "none", 0),
                        ("2026-01-05", 102000.0, 0.02, 0, "none", "buy", 2),
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO strategy_state
                    (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, start_date, last_date,
                     cash, shares, in_position, entry_price, pending_action, prev_equity, trades_executed)
                    VALUES
                    ('SQQQ', 'QQQ', 70.0, 1.5, '2026-01-02', '2026-01-05',
                     102000.0, 0.0, 0, NULL, 'buy', 102000.0, 2)
                    """
                )
                conn.execute(
                    """
                    INSERT INTO rsi_values
                    (signal_symbol, rsi_period, date, close, avg_gain, avg_loss, rsi)
                    VALUES ('QQQ', 14, '2026-01-05', 100.0, 1.0, 0.1, 75.0)
                    """
                )
                conn.commit()

            workflow_assets = pd.DataFrame([{"symbol": "SQQQ", "name": "S", "rsi_symbol": "QQQ"}])

            optimization_summary, curves, buy_signals, _eligible, _sell_signals, pnl = _build_reports_for_db(
                db_path,
                workflow_assets,
                BacktestConfig(),
                processed_asset_pairs={("SQQQ", "QQQ")},
                workflow_label="Short",
            )

        self.assertEqual(optimization_summary["Workflow"].tolist(), ["Short"])
        self.assertEqual(curves.columns.tolist(), ["Short_SQQQ_RSI_Strategy"])
        self.assertEqual(buy_signals["Workflow"].tolist(), ["Short"])
        self.assertNotIn("Workflow", pnl.columns)

    def test_workflow_reconciles_again_after_submitting_a_buy(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
        initial_reconciliation = pd.DataFrame(columns=["Action"])
        post_buy_reconciliation = pd.DataFrame([{"Action": "sell", "Status": "accepted"}])

        async def fake_asset_pipeline(**_kwargs: object) -> list[AssetRunResult]:
            return [
                AssetRunResult(
                    workflow_idx=1,
                    asset_symbol="TQQQ",
                    signal_symbol="QQQ",
                    action="Updating",
                    rows_processed=1,
                    status="done",
                    message="Processed 1 row",
                )
            ]

        async def immediate_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow.asyncio.to_thread", new=immediate_to_thread),
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    side_effect=[initial_reconciliation, post_buy_reconciliation],
                ) as mock_reconcile,
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                    return_value=workflow_assets,
                ),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=fake_asset_pipeline),
                patch(
                    "leveraged_trader.workflow._build_reports_for_db",
                    return_value=(
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                    ),
                ),
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    return_value=pd.DataFrame([{"Status": "submitted"}]),
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(),
                ),
                patch("leveraged_trader.workflow._write_workflow_outputs") as mock_write_outputs,
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=db_path,
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(enabled=True),
                        output_dir=str(Path(tmp) / "outputs"),
                        reporter=reporter,
                    )
                )

        self.assertEqual(mock_reconcile.call_count, 2)
        reconciliation_results = mock_write_outputs.call_args.kwargs["reconciliation_results"]
        self.assertEqual(reconciliation_results["Action"].tolist(), ["sell"])

    def test_workflow_runs_long_then_short_and_submits_combined_buy_signals(self) -> None:
        workflow_asset_groups = {
            "long": pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}]),
            "short": pd.DataFrame([{"symbol": "SQQQ", "name": "S", "rsi_symbol": "QQQ"}]),
        }
        pipeline_rules: list[str] = []
        submitted_buy_signals: list[pd.DataFrame] = []

        async def fake_asset_pipeline(**kwargs: object) -> list[AssetRunResult]:
            pipeline_rules.append(str(kwargs["rsi_entry_rule"]))
            jobs = kwargs["jobs"]
            return [
                AssetRunResult(
                    workflow_idx=job.workflow_idx,
                    asset_symbol=job.asset_symbol,
                    signal_symbol=job.signal_symbol,
                    action="Updating",
                    rows_processed=1,
                    status="done",
                    message="Processed 1 row",
                )
                for job in jobs
            ]

        def fake_build_reports(
            _db_path: str,
            _workflow_assets: pd.DataFrame,
            _base_cfg: BacktestConfig,
            _processed_asset_pairs: set[tuple[str, str]],
            workflow_label: str | None = None,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            asset = "TQQQ" if workflow_label == "Long" else "SQQQ"
            buy_signals = pd.DataFrame(
                [
                    {
                        "Workflow": workflow_label,
                        "Asset": asset,
                        "RSI Symbol": "QQQ",
                        "Date": "2026-01-02",
                        "Buy RSI": 30.0 if workflow_label == "Long" else 70.0,
                        "Sell Return Multiple": 1.5,
                    }
                ]
            )
            return (
                pd.DataFrame([{"Workflow": workflow_label, "Asset": asset}]),
                pd.DataFrame(),
                buy_signals,
                buy_signals.copy(),
                pd.DataFrame(),
                pd.DataFrame(),
            )

        def fake_submit(_db_path: str, buy_signals: pd.DataFrame, _alpaca_cfg: AlpacaOrderConfig) -> pd.DataFrame:
            submitted_buy_signals.append(buy_signals.copy())
            return pd.DataFrame(columns=["Status"])

        async def immediate_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow.asyncio.to_thread", new=immediate_to_thread),
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ),
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                    return_value=workflow_asset_groups,
                ),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=fake_asset_pipeline),
                patch("leveraged_trader.workflow._build_reports_for_db", side_effect=fake_build_reports),
                patch("leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db", side_effect=fake_submit),
                patch("leveraged_trader.workflow._load_alpaca_managed_positions_for_db", return_value=pd.DataFrame()),
                patch("leveraged_trader.workflow._write_workflow_outputs"),
                patch.object(reporter, "universe_assets", wraps=reporter.universe_assets) as mock_universe_assets,
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(Path(tmp) / "outputs"),
                        reporter=reporter,
                    )
                )

        self.assertEqual(pipeline_rules, ["lower", "upper"])
        mock_universe_assets.assert_called_once()
        rendered_universe = mock_universe_assets.call_args.args[0]
        self.assertEqual(rendered_universe["symbol"].tolist(), ["TQQQ", "SQQQ"])
        self.assertEqual(
            rendered_universe.attrs["universe_title"],
            "Executable Leveraged ETFs/ETNs From Merged Universe",
        )
        self.assertEqual(len(submitted_buy_signals), 1)
        self.assertEqual(submitted_buy_signals[0]["Workflow"].tolist(), ["Long", "Short"])
        self.assertEqual(submitted_buy_signals[0]["Asset"].tolist(), ["TQQQ", "SQQQ"])

    def test_workflow_fails_when_every_asset_is_skipped(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])

        async def skipped_asset_pipeline(**_kwargs: object) -> list[AssetRunResult]:
            return [
                AssetRunResult(
                    workflow_idx=1,
                    asset_symbol="TQQQ",
                    signal_symbol="QQQ",
                    action="Updating",
                    rows_processed=None,
                    status="skipped",
                    message="market data providers unavailable",
                )
            ]

        async def immediate_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow.asyncio.to_thread", new=immediate_to_thread),
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ),
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                    return_value=workflow_assets,
                ),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=skipped_asset_pipeline),
                patch("leveraged_trader.workflow._build_reports_for_db") as mock_build_reports,
                patch("leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db") as mock_submit_buys,
                self.assertRaisesRegex(WorkflowRunError, "No asset workflows completed successfully"),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(Path(tmp) / "outputs"),
                        reporter=reporter,
                    )
                )

        mock_build_reports.assert_not_called()
        mock_submit_buys.assert_not_called()

    def test_workflow_fails_when_no_assets_are_run(self) -> None:
        empty_assets = pd.DataFrame(columns=["symbol", "name", "rsi_symbol"])
        workflow_asset_groups = {
            "long": empty_assets,
            "short": empty_assets.copy(),
        }

        async def immediate_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow.asyncio.to_thread", new=immediate_to_thread),
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ),
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                    return_value=workflow_asset_groups,
                ),
                patch("leveraged_trader.workflow._run_asset_pipeline") as mock_run_pipeline,
                patch("leveraged_trader.workflow._build_reports_for_db") as mock_build_reports,
                patch("leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db") as mock_submit_buys,
                self.assertRaisesRegex(WorkflowRunError, "No executable assets were run"),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(Path(tmp) / "outputs"),
                        reporter=reporter,
                    )
                )

        mock_run_pipeline.assert_not_called()
        mock_build_reports.assert_not_called()
        mock_submit_buys.assert_not_called()

    def test_asset_pipeline_bounds_downloads_overlaps_db_and_sorts_results(self) -> None:
        jobs = [
            AssetRunJob(index, symbol, symbol) for index, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"], start=1)
        ]
        active_downloads = 0
        max_active_downloads = 0
        active_state = 0
        max_active_state = 0
        state_started = asyncio.Event()
        download_overlapped_state = False

        async def fake_prepare(**kwargs: object) -> PreparedAssetRun:
            nonlocal active_downloads, max_active_downloads, download_overlapped_state
            job = kwargs["job"]
            active_downloads += 1
            max_active_downloads = max(max_active_downloads, active_downloads)
            if state_started.is_set():
                download_overlapped_state = True
            await asyncio.sleep(0.005 if job.workflow_idx % 2 == 0 else 0.01)
            active_downloads -= 1
            data = pd.DataFrame({"Close": [100.0]})
            return PreparedAssetRun(
                job=job,
                plan=AssetRunPlan(
                    asset_symbol=job.asset_symbol,
                    signal_symbol=job.signal_symbol,
                    rebuild=False,
                    start=None,
                    action="Updating",
                    start_label="earliest overlapping history",
                ),
                data=data,
                asset_history=data,
                signal_history=data,
                risk_free_history=data,
            )

        async def fake_complete(
            outcome: PreparedAssetRun,
            **_kwargs: object,
        ) -> AssetRunResult:
            nonlocal active_state, max_active_state
            active_state += 1
            max_active_state = max(max_active_state, active_state)
            state_started.set()
            await asyncio.sleep(0.02)
            active_state -= 1
            return AssetRunResult(
                workflow_idx=outcome.job.workflow_idx,
                asset_symbol=outcome.job.asset_symbol,
                signal_symbol=outcome.job.signal_symbol,
                action="Updating",
                rows_processed=1,
                status="done",
                message="done",
            )

        with (
            patch("leveraged_trader.workflow._prepare_workflow_asset", new=fake_prepare),
            patch("leveraged_trader.workflow._complete_workflow_asset", new=fake_complete),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=jobs,
                    concurrency=2,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                )
            )

        self.assertEqual(max_active_downloads, 2)
        self.assertEqual(max_active_state, 1)
        self.assertTrue(download_overlapped_state)
        self.assertEqual([result.workflow_idx for result in results], [1, 2, 3, 4])

    def test_asset_pipeline_concurrency_one_is_fully_serial(self) -> None:
        jobs = [AssetRunJob(1, "AAA", "AAA"), AssetRunJob(2, "BBB", "BBB")]
        events: list[str] = []

        async def fake_prepare(**kwargs: object) -> AssetRunResult:
            job = kwargs["job"]
            events.append(f"prepare-{job.workflow_idx}")
            return AssetRunResult(
                workflow_idx=job.workflow_idx,
                asset_symbol=job.asset_symbol,
                signal_symbol=job.signal_symbol,
                action="Updating",
                rows_processed=0,
                status="skipped",
                message="prepared",
            )

        async def fake_complete(
            outcome: AssetRunResult,
            **_kwargs: object,
        ) -> AssetRunResult:
            events.append(f"complete-{outcome.workflow_idx}")
            return outcome

        with (
            patch("leveraged_trader.workflow._prepare_workflow_asset", new=fake_prepare),
            patch("leveraged_trader.workflow._complete_workflow_asset", new=fake_complete),
        ):
            asyncio.run(
                _run_asset_pipeline(
                    jobs=jobs,
                    concurrency=1,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                )
            )

        self.assertEqual(events, ["prepare-1", "complete-1", "prepare-2", "complete-2"])

    def test_asset_pipeline_closes_strategy_session_after_unexpected_failure(self) -> None:
        jobs = [AssetRunJob(1, "AAA", "AAA"), AssetRunJob(2, "BBB", "BBB")]
        close_threads: list[str] = []
        original_close = _WorkflowStrategySession.close

        async def fake_prepare(**kwargs: object) -> AssetRunResult:
            job = kwargs["job"]
            return AssetRunResult(
                workflow_idx=job.workflow_idx,
                asset_symbol=job.asset_symbol,
                signal_symbol=job.signal_symbol,
                action="Updating",
                rows_processed=0,
                status="skipped",
                message="prepared",
            )

        async def fail_complete(*_args: object, **_kwargs: object) -> AssetRunResult:
            raise RuntimeError("consumer failed")

        def record_close(session: _WorkflowStrategySession) -> None:
            close_threads.append(threading.current_thread().name)
            original_close(session)

        with (
            patch("leveraged_trader.workflow._prepare_workflow_asset", new=fake_prepare),
            patch("leveraged_trader.workflow._complete_workflow_asset", new=fail_complete),
            patch.object(_WorkflowStrategySession, "close", new=record_close),
            self.assertRaises(ExceptionGroup),
        ):
            asyncio.run(
                _run_asset_pipeline(
                    jobs=jobs,
                    concurrency=2,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                )
            )

        self.assertEqual(len(close_threads), 1)
        self.assertTrue(close_threads[0].startswith("workflow-strategy"))

    def test_asset_preparation_returns_skipped_result_for_empty_market_data(self) -> None:
        job = AssetRunJob(1, "TQQQ", "QQQ", workflow="Long")

        async def fake_to_thread(func: object, /, *args: object, **_kwargs: object) -> object:
            if getattr(func, "__name__", "") == "_prepare_asset_run":
                return AssetRunPlan(
                    asset_symbol="TQQQ",
                    signal_symbol="QQQ",
                    rebuild=False,
                    start=None,
                    action="Updating",
                    start_label="earliest overlapping history",
                )
            if getattr(func, "__name__", "") == "load_strategy_data":
                return pd.DataFrame()
            raise AssertionError(f"unexpected worker call: {getattr(func, '__name__', '')}")

        with patch("leveraged_trader.workflow.asyncio.to_thread", new=fake_to_thread):
            outcome = asyncio.run(
                _prepare_workflow_asset(
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    job=job,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    signal_locks={},
                    signal_histories={},
                    risk_free_history_lock=asyncio.Lock(),
                    risk_free_histories={},
                    phase_timings=WorkflowPhaseTimings(),
                )
            )

        self.assertIsInstance(outcome, AssetRunResult)
        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(outcome.rows_processed, 0)
        self.assertEqual(outcome.workflow, "Long")
        self.assertIn("No finalized daily market data", outcome.message)

    def test_asset_pipeline_drains_preparation_and_db_failures(self) -> None:
        jobs = [AssetRunJob(1, "AAA", "AAA"), AssetRunJob(2, "BBB", "BBB")]
        asset_progress = Mock()

        async def fake_prepare(**kwargs: object) -> PreparedAssetRun | AssetRunResult:
            job = kwargs["job"]
            if job.workflow_idx == 1:
                return AssetRunResult(
                    workflow_idx=1,
                    asset_symbol="AAA",
                    signal_symbol="AAA",
                    action="Updating",
                    rows_processed=None,
                    status="skipped",
                    message="provider failed",
                )
            data = pd.DataFrame({"Close": [100.0]})
            return PreparedAssetRun(
                job=job,
                plan=AssetRunPlan(
                    asset_symbol="BBB",
                    signal_symbol="BBB",
                    rebuild=False,
                    start=None,
                    action="Updating",
                    start_label="earliest overlapping history",
                ),
                data=data,
                asset_history=data,
                signal_history=data,
                risk_free_history=data,
            )

        with (
            patch("leveraged_trader.workflow._prepare_workflow_asset", new=fake_prepare),
            patch(
                "leveraged_trader.workflow._process_asset_grid_for_db",
                side_effect=RuntimeError("database failed"),
            ),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=jobs,
                    concurrency=2,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=asset_progress,
                    phase_timings=WorkflowPhaseTimings(),
                )
            )

        self.assertEqual([result.status for result in results], ["skipped", "skipped"])
        self.assertEqual([result.message for result in results], ["provider failed", "database failed"])
        self.assertEqual(asset_progress.finish_asset.call_count, 2)

    def test_terminal_alpaca_display_results_preserves_closed_position_id_gaps(self) -> None:
        managed_positions = pd.DataFrame(
            [
                {"id": 2, "symbol": "KLAG", "buy_client_order_id": "buy-KLAG"},
                {"id": 3, "symbol": "MPG", "buy_client_order_id": "buy-MPG", "closed_at": "2026-01-03"},
                {"id": 5, "symbol": "SATG", "buy_client_order_id": "buy-SATG"},
                {"id": 8, "symbol": "LULG", "buy_client_order_id": "buy-LULG", "closed_at": "2026-01-04"},
                {"id": 9, "symbol": "AXTU", "buy_client_order_id": "buy-AXTU"},
            ]
        )
        reconciliation_results = pd.DataFrame(
            [
                {"Position ID": 2, "Asset": "KLAG", "Action": "sell"},
                {"Position ID": 5, "Asset": "SATG", "Action": "sell"},
                {"Position ID": 9, "Asset": "AXTU", "Action": "sell"},
            ]
        )
        order_results = pd.DataFrame(
            [
                {"Asset": "AXTU", "Client Order ID": "buy-AXTU", "Status": "submitted"},
                {"Asset": "UNMG", "Client Order ID": "buy-UNMG", "Status": "submitted"},
            ]
        )

        display_orders, display_reconciliation = _terminal_alpaca_display_results(
            managed_positions=managed_positions,
            reconciliation_results=reconciliation_results,
            order_results=order_results,
        )

        self.assertEqual(display_reconciliation["Display ID"].tolist(), [1, 3, 5])
        self.assertEqual(display_orders.loc[0, "Display ID"], 5)
        self.assertTrue(pd.isna(display_orders.loc[1, "Display ID"]))

    def test_write_workflow_outputs_uses_terminal_display_ids_for_alpaca_tables(self) -> None:
        managed_positions = pd.DataFrame(
            [
                {"id": 2, "symbol": "KLAG", "buy_client_order_id": "buy-KLAG"},
                {"id": 3, "symbol": "MPG", "buy_client_order_id": "buy-MPG", "closed_at": "2026-01-03"},
                {"id": 5, "symbol": "AXTU", "buy_client_order_id": "buy-AXTU"},
            ]
        )
        reconciliation_results = pd.DataFrame(
            [
                {
                    "Position ID": 2,
                    "Asset": "KLAG",
                    "Action": "sell",
                    "Status": "new",
                    "Buy Client Order ID": "buy-KLAG",
                    "Sell Client Order ID": "sell-KLAG",
                    "Qty": 15,
                    "Limit Price": 77.69,
                    "Alpaca Order ID": "alpaca-sell-KLAG",
                    "Message": "managed sell already submitted",
                },
                {
                    "Position ID": 5,
                    "Asset": "AXTU",
                    "Action": "sell",
                    "Status": "new",
                    "Buy Client Order ID": "buy-AXTU",
                    "Sell Client Order ID": "sell-AXTU",
                    "Qty": 8,
                    "Limit Price": 9.57,
                    "Alpaca Order ID": "alpaca-sell-AXTU",
                    "Message": "managed sell already submitted",
                },
            ]
        )
        order_results = pd.DataFrame(
            [
                {
                    "Asset": "AXTU",
                    "Date": "2026-01-02",
                    "Client Order ID": "buy-AXTU",
                    "Notional": 100.0,
                    "Qty": 8,
                    "Limit Price": 9.57,
                    "Status": "submitted",
                    "Alpaca Order ID": "alpaca-buy-AXTU",
                    "Message": "submitted",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_buffer = io.StringIO()
            reporter = WorkflowReporter(
                console=Console(file=output_buffer, width=140, color_system=None, no_color=True)
            )
            _write_workflow_outputs(
                mode="update",
                db_path=str(Path(tmp) / "state.sqlite"),
                base_cfg=BacktestConfig(),
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
                alpaca_cfg=AlpacaOrderConfig(enabled=True, sell_enabled=True),
                output_dir=str(output_dir),
                workflow_concurrency=1,
                reporter=reporter,
                asset_run_results=[],
                optimization_summary=pd.DataFrame(),
                curves=pd.DataFrame(),
                buy_signals=pd.DataFrame(),
                eligible_buy_signals=pd.DataFrame(),
                sell_signals=pd.DataFrame(),
                realized_pnl_summary=pd.DataFrame(),
                managed_positions=managed_positions,
                reconciliation_results=reconciliation_results,
                sell_reconciliation_results=reconciliation_results,
                order_results=order_results,
                workflow_timer=WorkflowTimer.start(),
            )
            written_reconciliation = pd.read_csv(output_dir / "alpaca_reconciliation_results.csv")
            written_orders = pd.read_csv(output_dir / "alpaca_order_results.csv")
            output = output_buffer.getvalue()

        axtu_lines = [line for line in output.splitlines() if "AXTU" in line]
        self.assertEqual(len(axtu_lines), 2)
        for line in axtu_lines:
            self.assertRegex(line, r"^\s*3\s+AXTU\b")
        self.assertEqual(written_reconciliation["Position ID"].tolist(), [2, 5])
        self.assertNotIn("Display ID", written_reconciliation.columns)
        self.assertNotIn("Display ID", written_orders.columns)

    def test_write_workflow_outputs_omits_sell_signal_table_but_writes_csv(self) -> None:
        sell_signals = pd.DataFrame(
            [
                {
                    "Asset": "ZZSELL",
                    "RSI Symbol": "ZZRSI",
                    "Date": "2026-01-02",
                    "Pending Action": "sell",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_buffer = io.StringIO()
            reporter = WorkflowReporter(
                console=Console(file=output_buffer, width=100, color_system=None, no_color=True)
            )
            _write_workflow_outputs(
                mode="update",
                db_path=str(Path(tmp) / "state.sqlite"),
                base_cfg=BacktestConfig(),
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
                alpaca_cfg=AlpacaOrderConfig(),
                output_dir=str(output_dir),
                workflow_concurrency=1,
                reporter=reporter,
                asset_run_results=[],
                optimization_summary=pd.DataFrame(),
                curves=pd.DataFrame(),
                buy_signals=pd.DataFrame(),
                eligible_buy_signals=pd.DataFrame(),
                sell_signals=sell_signals,
                realized_pnl_summary=pd.DataFrame(),
                managed_positions=pd.DataFrame(),
                reconciliation_results=pd.DataFrame(columns=["Action"]),
                sell_reconciliation_results=pd.DataFrame(),
                order_results=pd.DataFrame(),
                workflow_timer=WorkflowTimer.start(),
            )
            output = output_buffer.getvalue()
            written_sell_signals = pd.read_csv(output_dir / "sell_signals.csv")

        self.assertNotIn("Sell Signals For Next Open", output)
        self.assertNotIn("ZZSELL", output)
        self.assertEqual(written_sell_signals["Asset"].tolist(), ["ZZSELL"])

    def test_write_workflow_outputs_prints_footer_without_benchmark_csv(self) -> None:
        asset_run_results = [
            AssetRunResult(
                workflow_idx=1,
                asset_symbol="TQQQ",
                signal_symbol="QQQ",
                action="Updating",
                rows_processed=12,
                status="done",
                message="Processed 12 rows",
            ),
            AssetRunResult(
                workflow_idx=2,
                asset_symbol="UPRO",
                signal_symbol="SPY",
                action="Updating",
                rows_processed=None,
                status="skipped",
                message="No finalized daily market data is available yet.",
            ),
        ]

        phase_timings = WorkflowPhaseTimings()
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("leveraged_trader.workflow.time") as mock_time,
        ):
            mock_time.perf_counter.side_effect = [10.0, 12.0]
            output_dir = Path(tmp) / "outputs"
            output_buffer = io.StringIO()
            reporter = WorkflowReporter(
                console=Console(file=output_buffer, width=100, color_system=None, no_color=True)
            )
            _write_workflow_outputs(
                mode="update",
                db_path=str(Path(tmp) / "state.sqlite"),
                base_cfg=BacktestConfig(),
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
                alpaca_cfg=AlpacaOrderConfig(),
                output_dir=str(output_dir),
                workflow_concurrency=3,
                reporter=reporter,
                asset_run_results=asset_run_results,
                optimization_summary=pd.DataFrame(),
                curves=pd.DataFrame(),
                buy_signals=pd.DataFrame(),
                eligible_buy_signals=pd.DataFrame(),
                sell_signals=pd.DataFrame(),
                realized_pnl_summary=pd.DataFrame(),
                managed_positions=pd.DataFrame(),
                reconciliation_results=pd.DataFrame(columns=["Action"]),
                sell_reconciliation_results=pd.DataFrame(),
                order_results=pd.DataFrame(),
                workflow_timer=WorkflowTimer.start(),
                phase_timings=phase_timings,
            )
            benchmark_csv_exists = (output_dir / "workflow_benchmark.csv").exists()
            output = output_buffer.getvalue()

        self.assertFalse(benchmark_csv_exists)
        self.assertEqual(phase_timings.snapshot().report_generation_seconds, 2.0)
        self.assertNotIn("Phase time:", output)
        self.assertIn("Workflow finished in", output)
        self.assertNotIn("Workflow Benchmark", output)
        self.assertEqual(output.rstrip().splitlines()[-1], "\u2500" * 100)

    def test_write_workflow_outputs_preserves_workflow_columns_and_side_prefixed_curves(self) -> None:
        optimization_summary = pd.DataFrame(
            [
                {
                    "Workflow": "Long",
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Start Date": "2026-01-02",
                    "End Date": "2026-01-05",
                    "Trading Days": 2,
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 1.5,
                    "Trades Executed": 2,
                    "Total Return": 0.05,
                    "CAGR": 0.10,
                    "Annualized Vol": 0.20,
                    "Sharpe": 1.20,
                    "Kelly Fraction": 0.30,
                    "Max Drawdown": -0.01,
                    "Hit Rate": 0.50,
                },
                {
                    "Workflow": "Short",
                    "Asset": "SQQQ",
                    "RSI Symbol": "QQQ",
                    "Start Date": "2026-01-02",
                    "End Date": "2026-01-05",
                    "Trading Days": 2,
                    "Buy RSI": 70.0,
                    "Sell Return Multiple": 1.5,
                    "Trades Executed": 2,
                    "Total Return": 0.04,
                    "CAGR": 0.08,
                    "Annualized Vol": 0.20,
                    "Sharpe": 1.10,
                    "Kelly Fraction": 0.25,
                    "Max Drawdown": -0.02,
                    "Hit Rate": 0.50,
                },
            ]
        )
        buy_signals = pd.DataFrame(
            [
                {
                    "Workflow": "Long",
                    "Asset": "TQQQ",
                    "RSI Symbol": "QQQ",
                    "Date": "2026-01-05",
                    "Start Date": "2026-01-02",
                    "Trading Days": 2,
                    "Latest RSI": 25.0,
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 1.5,
                    "Trades Executed": 2,
                    "Sharpe": 1.2,
                    "In Position": False,
                    "Pending Action": "buy",
                },
                {
                    "Workflow": "Short",
                    "Asset": "SQQQ",
                    "RSI Symbol": "QQQ",
                    "Date": "2026-01-05",
                    "Start Date": "2026-01-02",
                    "Trading Days": 2,
                    "Latest RSI": 75.0,
                    "Buy RSI": 70.0,
                    "Sell Return Multiple": 1.5,
                    "Trades Executed": 2,
                    "Sharpe": 1.1,
                    "In Position": False,
                    "Pending Action": "buy",
                },
            ]
        )
        sell_signals = pd.DataFrame(
            [
                {
                    "Workflow": "Short",
                    "Asset": "SQQQ",
                    "RSI Symbol": "QQQ",
                    "Date": "2026-01-05",
                    "Pending Action": "sell",
                }
            ]
        )
        order_results = pd.DataFrame(
            [
                {
                    "Workflow": "Short",
                    "Asset": "SQQQ",
                    "Date": "2026-01-05",
                    "Client Order ID": "buy-SQQQ",
                    "Notional": 100.0,
                    "Qty": 1,
                    "Limit Price": 100.0,
                    "Status": "submitted",
                    "Alpaca Order ID": "alpaca-buy-SQQQ",
                    "Message": "submitted",
                }
            ]
        )
        reconciliation_results = pd.DataFrame(
            [
                {
                    "Position ID": 1,
                    "Workflow": "Short",
                    "Asset": "SQQQ",
                    "Action": "sell",
                    "Status": "new",
                    "Qty": 1,
                    "Limit Price": 115.0,
                    "Message": "managed sell submitted",
                }
            ]
        )
        realized_pnl_summary = pd.DataFrame(
            [
                {
                    "Workflow": "Short",
                    "Closed Positions": 1,
                    "Complete Closed Positions": 1,
                    "Incomplete Closed Positions": 0,
                    "Total Buy Cost": 100.0,
                    "Total Sell Value": 115.0,
                    "Realized P/L": 15.0,
                    "Realized P/L %": 15.0,
                }
            ]
        )
        curves = pd.DataFrame(
            {
                "Long_TQQQ_RSI_Strategy": [100_000.0, 101_000.0],
                "Short_SQQQ_RSI_Strategy": [100_000.0, 102_000.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=160, color_system=None, no_color=True)
            )
            _write_workflow_outputs(
                mode="update",
                db_path=str(Path(tmp) / "state.sqlite"),
                base_cfg=BacktestConfig(),
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
                alpaca_cfg=AlpacaOrderConfig(enabled=True, sell_enabled=True),
                output_dir=str(output_dir),
                workflow_concurrency=1,
                reporter=reporter,
                asset_run_results=[],
                optimization_summary=optimization_summary,
                curves=curves,
                buy_signals=buy_signals,
                eligible_buy_signals=buy_signals.iloc[[1]].copy(),
                sell_signals=sell_signals,
                realized_pnl_summary=realized_pnl_summary,
                managed_positions=pd.DataFrame(),
                reconciliation_results=reconciliation_results,
                sell_reconciliation_results=reconciliation_results,
                order_results=order_results,
                workflow_timer=WorkflowTimer.start(),
            )

            written_curves = pd.read_csv(output_dir / "best_equity_curves.csv", index_col=0)
            written_summary = pd.read_csv(output_dir / "optimization_summary.csv")
            written_buy_signals = pd.read_csv(output_dir / "buy_signals.csv")
            written_eligible = pd.read_csv(output_dir / "eligible_buy_signals.csv")
            written_sell_signals = pd.read_csv(output_dir / "sell_signals.csv")
            written_orders = pd.read_csv(output_dir / "alpaca_order_results.csv")
            written_reconciliation = pd.read_csv(output_dir / "alpaca_reconciliation_results.csv")
            written_sell_orders = pd.read_csv(output_dir / "alpaca_sell_order_results.csv")
            written_realized_pnl = pd.read_csv(output_dir / "alpaca_realized_pnl.csv")

        self.assertEqual(
            written_curves.columns.tolist(),
            ["Long_TQQQ_RSI_Strategy", "Short_SQQQ_RSI_Strategy"],
        )
        for frame in [
            written_summary,
            written_buy_signals,
            written_eligible,
            written_sell_signals,
            written_orders,
            written_reconciliation,
            written_sell_orders,
            written_realized_pnl,
        ]:
            self.assertIn("Workflow", frame.columns)
        self.assertEqual(written_buy_signals["Workflow"].tolist(), ["Long", "Short"])
        self.assertEqual(written_eligible["Workflow"].tolist(), ["Short"])

    def test_write_workflow_outputs_keeps_full_csv_and_filters_terminal_summary_rows(self) -> None:
        optimization_summary = pd.DataFrame(
            [
                {
                    "Asset": f"A{index:03d}",
                    "RSI Symbol": f"R{index:03d}",
                    "Start Date": "2026-01-02",
                    "End Date": "2026-01-03",
                    "Trading Days": 2,
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 1.5,
                    "Trades Executed": 1 if index == 100 else 2,
                    "Total Return": 0.1,
                    "CAGR": 0.2,
                    "Annualized Vol": 0.3,
                    "Sharpe": 5.0 if index == 100 else (1.2 if index == 0 else 0.9),
                    "Kelly Fraction": 0.4,
                    "Max Drawdown": -0.1,
                    "Hit Rate": 0.5,
                }
                for index in range(101)
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_buffer = io.StringIO()
            reporter = WorkflowReporter(
                console=Console(file=output_buffer, width=120, color_system=None, no_color=True)
            )
            _write_workflow_outputs(
                mode="update",
                db_path=str(Path(tmp) / "state.sqlite"),
                base_cfg=BacktestConfig(),
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
                alpaca_cfg=AlpacaOrderConfig(),
                output_dir=str(output_dir),
                workflow_concurrency=1,
                reporter=reporter,
                asset_run_results=[],
                optimization_summary=optimization_summary,
                curves=pd.DataFrame(),
                buy_signals=pd.DataFrame(),
                eligible_buy_signals=pd.DataFrame(),
                sell_signals=pd.DataFrame(),
                realized_pnl_summary=pd.DataFrame(),
                managed_positions=pd.DataFrame(),
                reconciliation_results=pd.DataFrame(columns=["Action"]),
                sell_reconciliation_results=pd.DataFrame(),
                order_results=pd.DataFrame(),
                workflow_timer=WorkflowTimer.start(),
            )

            written_summary = pd.read_csv(output_dir / "optimization_summary.csv")

        self.assertEqual(len(written_summary), 101)
        output = output_buffer.getvalue()
        self.assertIn("A000", output)
        self.assertNotIn("A100", output)
        self.assertNotIn("A099", output)
        self.assertNotIn("Showing 100 of 101 rows", output)


if __name__ == "__main__":
    unittest.main()
