from __future__ import annotations

import asyncio
import io
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from rich.console import Console

from leveraged_trader.config import AlpacaOrderConfig, BacktestConfig, UniverseConfig
from leveraged_trader.output import WorkflowReporter
from leveraged_trader.storage import init_state_db
from leveraged_trader.workflow import (
    AssetRunPlan,
    AssetRunResult,
    WorkflowRunError,
    _build_reports_for_db,
    _prepare_asset_run,
    _process_workflow_asset,
    _state_connection,
    run_resumable_optimizations_async,
)


class WorkflowAsyncTests(unittest.TestCase):
    def _run_workflow_with_grids(
        self,
        *,
        buy_rsi_values: list[float],
        profit_target_values: list[float],
    ) -> None:
        asyncio.run(
            run_resumable_optimizations_async(
                mode="update",
                db_path="unused.sqlite",
                base_cfg=BacktestConfig(),
                universe_cfg=UniverseConfig(),
                buy_rsi_values=buy_rsi_values,
                profit_target_values=profit_target_values,
                alpaca_cfg=AlpacaOrderConfig(),
                output_dir="outputs",
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

    def test_state_processing_is_serialized_across_different_signal_symbols(self) -> None:
        state_active = 0
        max_state_active = 0

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

        async def fake_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
            nonlocal state_active, max_state_active
            name = getattr(func, "__name__", "")
            if name == "_prepare_asset_run":
                return AssetRunPlan(
                    asset_symbol=str(args[3]),
                    signal_symbol=str(args[4]),
                    rebuild=False,
                    start=None,
                    action="Updating",
                    start_label="earliest overlapping history",
                )
            if name == "load_strategy_data":
                return history(str(kwargs["asset_symbol"]))
            if name == "load_symbol_history":
                return history(str(args[0]))
            if name == "load_signal_history":
                return history(str(args[0]))
            if name == "load_risk_free_history":
                return history("^IRX")
            if name == "_process_asset_grid_for_db":
                state_active += 1
                max_state_active = max(max_state_active, state_active)
                await asyncio.sleep(0.01)
                state_active -= 1
                return None
            raise AssertionError(f"unexpected worker call: {name}")

        async def run() -> list[AssetRunResult]:
            signal_locks: dict[str, asyncio.Lock] = {}
            risk_free_history_lock = asyncio.Lock()
            risk_free_histories: dict[str, pd.DataFrame] = {}
            strategy_state_lock = asyncio.Lock()
            return await asyncio.gather(
                *[
                    _process_workflow_asset(
                        db_path="state.sqlite",
                        mode="update",
                        base_cfg=BacktestConfig(),
                        tradier_cfg=None,
                        workflow_idx=index,
                        total_workflows=2,
                        asset_symbol=symbol,
                        signal_symbol=symbol,
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        signal_locks=signal_locks,
                        signal_histories={},
                        risk_free_history_lock=risk_free_history_lock,
                        risk_free_histories=risk_free_histories,
                        strategy_state_lock=strategy_state_lock,
                    )
                    for index, symbol in enumerate(["AAA", "BBB"], start=1)
                ]
            )

        with patch("leveraged_trader.workflow.asyncio.to_thread", new=fake_to_thread):
            results = asyncio.run(run())

        self.assertEqual(max_state_active, 1)
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
            return pd.DataFrame(columns=["symbol", "name", "rsi_symbol"])

        with tempfile.TemporaryDirectory() as tmp:
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow.asyncio.to_thread", new=immediate_to_thread),
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch("leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db", side_effect=reconcile),
                patch("leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db", side_effect=load_universe),
                patch(
                    "leveraged_trader.workflow._build_reports_for_db",
                    return_value=(
                        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    ),
                ),
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    return_value=pd.DataFrame(columns=["Status"]),
                ),
                patch("leveraged_trader.workflow._load_alpaca_managed_positions_for_db", return_value=pd.DataFrame()),
                patch("leveraged_trader.workflow._write_workflow_outputs"),
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

        self.assertEqual(events, ["reconcile", "universe"])

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

    def test_workflow_reconciles_again_after_submitting_a_buy(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
        initial_reconciliation = pd.DataFrame(columns=["Action"])
        post_buy_reconciliation = pd.DataFrame([{"Action": "sell", "Status": "accepted"}])

        async def fake_process_asset(**kwargs: object) -> AssetRunResult:
            asset_progress = kwargs.get("asset_progress")
            if asset_progress is not None:
                asset_progress.finish_asset()
            return AssetRunResult(
                workflow_idx=int(kwargs["workflow_idx"]),
                asset_symbol="TQQQ",
                signal_symbol="QQQ",
                action="Updating",
                rows_processed=1,
                status="done",
                message="Processed 1 row",
            )

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
                patch("leveraged_trader.workflow._process_workflow_asset", new=fake_process_asset),
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

    def test_workflow_fails_when_every_asset_is_skipped(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])

        async def skipped_process_asset(**kwargs: object) -> AssetRunResult:
            asset_progress = kwargs.get("asset_progress")
            if asset_progress is not None:
                asset_progress.finish_asset()
            return AssetRunResult(
                workflow_idx=int(kwargs["workflow_idx"]),
                asset_symbol="TQQQ",
                signal_symbol="QQQ",
                action="Updating",
                rows_processed=None,
                status="skipped",
                message="market data providers unavailable",
            )

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
                patch("leveraged_trader.workflow._process_workflow_asset", new=skipped_process_asset),
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

    def test_workflow_concurrency_limits_asset_tasks(self) -> None:
        workflow_assets = pd.DataFrame(
            [
                {"symbol": "AAA", "name": "A", "rsi_symbol": "AAA"},
                {"symbol": "BBB", "name": "B", "rsi_symbol": "BBB"},
                {"symbol": "CCC", "name": "C", "rsi_symbol": "CCC"},
            ]
        )
        empty_orders = pd.DataFrame(columns=["Asset", "Action"])
        active_tasks = 0
        max_active_tasks = 0

        async def fake_process_asset(**kwargs: object) -> AssetRunResult:
            nonlocal active_tasks, max_active_tasks
            active_tasks += 1
            max_active_tasks = max(max_active_tasks, active_tasks)
            await asyncio.sleep(0.01)
            active_tasks -= 1
            asset_progress = kwargs.get("asset_progress")
            if asset_progress is not None:
                asset_progress.finish_asset()
            return AssetRunResult(
                workflow_idx=int(kwargs["workflow_idx"]),
                asset_symbol=str(kwargs["asset_symbol"]),
                signal_symbol=str(kwargs["signal_symbol"]),
                action="Updating",
                rows_processed=10,
                status="done",
                message="Processed 10 rows",
            )

        async def immediate_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            output_dir = str(Path(tmp) / "outputs")
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
                patch("leveraged_trader.workflow._process_workflow_asset", new=fake_process_asset),
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
                    return_value=empty_orders,
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
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=output_dir,
                        workflow_concurrency=2,
                        reporter=reporter,
                    )
                )

        self.assertEqual(max_active_tasks, 2)
        mock_write_outputs.assert_called_once()
        asset_run_results = mock_write_outputs.call_args.kwargs["asset_run_results"]
        self.assertEqual([result.workflow_idx for result in asset_run_results], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
