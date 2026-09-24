from __future__ import annotations

import asyncio
import csv
import ctypes
import hashlib
import io
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
from rich.console import Console

from leveraged_trader import _yfinance_deadline_worker as yfinance_deadline_worker
from leveraged_trader.alpaca import AlpacaBuyBatchError, AlpacaReconciliationError
from leveraged_trader.benchmark import WorkflowPhaseTimings, WorkflowTimer
from leveraged_trader.config import (
    AlpacaOrderConfig,
    BacktestConfig,
    TradierMarketDataConfig,
    UniverseConfig,
)
from leveraged_trader.market_data import (
    MARKET_DATA_PROVIDERS_ATTR,
    TRADIER_RECOVERED_SYMBOLS_ATTR,
    MarketDataDownloadError,
    load_market_data,
)
from leveraged_trader.output import WorkflowReporter
from leveraged_trader.reports import build_pending_action_report
from leveraged_trader.storage import (
    AssetMarketDataError,
    _synchronize_market_data_history,
    clear_asset_state,
    init_state_db,
    load_aligned_rsi_for_asset_session,
    load_alpaca_managed_positions,
    mark_alpaca_closed_correction_audited,
    mark_alpaca_managed_buy_filled,
    save_alpaca_managed_buy_order,
    strategy_config_fingerprint,
)
from leveraged_trader.workflow import (
    _RESEARCH_REPORT_FILENAMES,
    AssetRunJob,
    AssetRunPlan,
    AssetRunResult,
    PreparedAssetRun,
    WorkflowDeadlineExceeded,
    WorkflowRunError,
    WorkflowStateCleanupError,
    _alpaca_inactive_holdings_report,
    _alpaca_sell_order_results,
    _atomic_to_csv,
    _build_reports_for_db,
    _clear_stale_workflow_research_outputs,
    _complete_workflow_asset,
    _CompletedWorkflowStateInvalidError,
    _concat_report_frames,
    _initialize_state_db,
    _persist_workflow_research_outputs,
    _prefetch_workflow_histories,
    _prepare_asset_run,
    _prepare_asset_safe_tail_rows,
    _prepare_workflow_asset,
    _process_asset_grid_for_db,
    _reconcile_alpaca_managed_positions_for_db,
    _reconcile_alpaca_managed_positions_with_cancellation_retry,
    _run_asset_pipeline,
    _run_blocking,
    _scheduled_alpaca_reconciliation_is_due,
    _state_connection,
    _strategy_data_from_authoritative_histories,
    _terminal_alpaca_display_results,
    _unknown_alpaca_buy_submission_results,
    _validate_database_path,
    _validate_optimization_grids,
    _validate_output_directory_path,
    _validate_workflow_mode,
    _windows_local_app_data_path,
    _workflow_run_lock,
    _workflow_user_lock_base,
    _WorkflowMarketDataSession,
    _WorkflowStrategySession,
    _write_workflow_outputs,
    load_alpaca_snapshot,
    run_alpaca_reconciliation,
    run_resumable_optimizations_async,
    validate_alpaca_snapshot,
)


class WorkflowAsyncTests(unittest.TestCase):
    def test_prefetch_uses_deterministic_batches_of_32(self) -> None:
        session = _WorkflowMarketDataSession()
        symbols = [f"S{index:03d}" for index in range(65)]

        def batch_loader(batch: list[str], **_kwargs: object) -> tuple[dict, dict]:
            return {}, {symbol: "retry" for symbol in batch}

        with patch(
            "leveraged_trader.workflow.load_symbol_history_batch",
            side_effect=batch_loader,
        ) as batch_download:
            asyncio.run(
                _prefetch_workflow_histories(
                    list(reversed(symbols)),
                    base_cfg=BacktestConfig(),
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=session,
                )
            )

        self.assertEqual([len(call.args[0]) for call in batch_download.call_args_list], [32, 32, 1])
        self.assertEqual(batch_download.call_args_list[0].args[0], sorted(symbols)[:32])
        self.assertEqual(session.batch_count, 3)
        self.assertEqual(session.batch_retry_symbols, set(symbols))

    def test_expired_workflow_deadline_never_acquires_lock_or_submits_buys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with (
                patch("leveraged_trader.workflow._workflow_run_lock") as workflow_lock,
                patch("leveraged_trader.workflow.submit_alpaca_paper_buy_orders") as submit_buys,
                self.assertRaisesRegex(WorkflowDeadlineExceeded, "buy submission was not started"),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=db_path,
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(sqlite_db_path=db_path),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=tmp,
                        workflow_deadline_epoch=time.time() - 1.0,
                    )
                )

        workflow_lock.assert_not_called()
        submit_buys.assert_not_called()

    def test_asset_transaction_verifies_resumable_state_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            with (
                patch(
                    "leveraged_trader.workflow.strategy_state_matches_config",
                    return_value=True,
                ) as verify_state,
                patch(
                    "leveraged_trader.workflow.process_asset_grid",
                    return_value=False,
                ) as process_grid,
            ):
                rebuilt = _process_asset_grid_for_db(
                    db_path,
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
                )

        self.assertFalse(rebuilt)
        verify_state.assert_called_once()
        self.assertTrue(process_grid.call_args.kwargs["strategy_state_preverified"])

    def test_preparation_proof_avoids_duplicate_transaction_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            asset_history = self._symbol_history("TQQQ")
            signal_history = self._symbol_history("QQQ")
            risk_free_history = self._symbol_history("^IRX", -95.0)
            strategy_data = _strategy_data_from_authoritative_histories(
                asset_symbol="TQQQ",
                signal_symbol="QQQ",
                asset_history=asset_history,
                signal_history=signal_history,
                risk_free_history=risk_free_history,
            )
            _process_asset_grid_for_db(
                db_path,
                strategy_data,
                asset_history,
                signal_history,
                risk_free_history,
                BacktestConfig(),
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                True,
            )
            with (
                patch(
                    "leveraged_trader.workflow.strategy_state_matches_config",
                    return_value=True,
                ) as verify_state,
                patch(
                    "leveraged_trader.workflow.process_asset_grid",
                    return_value=False,
                ) as process_grid,
            ):
                plan = _prepare_asset_run(
                    db_path,
                    "update",
                    BacktestConfig(),
                    "TQQQ",
                    "QQQ",
                    [30.0],
                    [1.5],
                )
                safe_tail_rows = _prepare_asset_safe_tail_rows(
                    db_path,
                    plan,
                    asset_history,
                )
                rebuilt = _process_asset_grid_for_db(
                    db_path,
                    pd.DataFrame(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                    BacktestConfig(),
                    "TQQQ",
                    "QQQ",
                    [30.0],
                    [1.5],
                    plan.rebuild,
                    strategy_state_preverified=plan.strategy_state_preverified,
                    expected_strategy_state_generation=plan.strategy_state_generation,
                    prevalidated_asset_safe_tail_rows=safe_tail_rows,
                )

        self.assertFalse(rebuilt)
        self.assertEqual(safe_tail_rows, {})
        verify_state.assert_called_once()
        self.assertTrue(process_grid.call_args.kwargs["strategy_state_preverified"])
        self.assertEqual(process_grid.call_args.kwargs["_prevalidated_safe_tail_rows"], {})
        self.assertTrue(process_grid.call_args.kwargs["_authoritative_histories_prevalidated"])

    def test_columnwise_report_concat_sorts_the_outer_date_index(self) -> None:
        long_curve = pd.DataFrame(
            {"Long_TQQQ_RSI_Strategy": [101_000.0, 102_000.0]},
            index=pd.to_datetime(["2024-02-01", "2024-03-01"]),
        )
        short_curve = pd.DataFrame(
            {"Short_SQQQ_RSI_Strategy": [99_000.0, 98_000.0]},
            index=pd.to_datetime(["2024-01-01", "2024-04-01"]),
        )

        combined = _concat_report_frames([long_curve, short_curve], axis=1)

        self.assertEqual(
            combined.index.tolist(),
            pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]).tolist(),
        )
        self.assertTrue(combined.index.is_monotonic_increasing)

    def test_cli_module_imports_when_fcntl_is_unavailable(self) -> None:
        script = """
import builtins
import asyncio
import sys
import types

real_import = builtins.__import__

def without_fcntl(name, *args, **kwargs):
    if name == "fcntl":
        raise ImportError("simulated Windows")
    return real_import(name, *args, **kwargs)

sys.modules["msvcrt"] = types.SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda *_args: None)
builtins.__import__ = without_fcntl
import leveraged_trader.cli
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_alpaca_snapshot_validator_detects_csv_changed_after_commit(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            publication = workflow_module._begin_alpaca_snapshot_publication(
                output_dir,
                snapshot_kind="current-state",
            )
            for filename in publication.expected_filenames:
                workflow_module._publish_alpaca_snapshot_csv(
                    publication,
                    pd.DataFrame([{"Generation": "committed"}]),
                    filename=filename,
                    index=False,
                )
            workflow_module._commit_alpaca_snapshot_publication(publication)

            self.assertEqual(validate_alpaca_snapshot(output_dir), publication.generation)

            _atomic_to_csv(
                pd.DataFrame([{"Generation": "substituted"}]),
                output_dir / publication.expected_filenames[0],
                index=False,
            )
            with self.assertRaisesRegex(OSError, "does not match the committed manifest"):
                validate_alpaca_snapshot(output_dir)

    def test_failed_snapshot_republish_cannot_commit_a_stale_digest(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            publication = workflow_module._begin_alpaca_snapshot_publication(
                output_dir,
                snapshot_kind="current-state",
            )
            for filename in publication.expected_filenames:
                workflow_module._publish_alpaca_snapshot_csv(
                    publication,
                    pd.DataFrame([{"Generation": "first"}]),
                    filename=filename,
                    index=False,
                )

            filename = publication.expected_filenames[0]
            original_atomic_to_csv = workflow_module._atomic_to_csv

            def replace_then_fail(*args: object, **kwargs: object) -> str:
                original_atomic_to_csv(*args, **kwargs)
                raise OSError("simulated failure after CSV replacement")

            with (
                patch.object(workflow_module, "_atomic_to_csv", side_effect=replace_then_fail),
                self.assertRaisesRegex(OSError, "failure after CSV replacement"),
            ):
                workflow_module._publish_alpaca_snapshot_csv(
                    publication,
                    pd.DataFrame([{"Generation": "replacement"}]),
                    filename=filename,
                    index=False,
                )

            self.assertNotIn(filename, publication.digests)
            with self.assertRaisesRegex(OSError, "incomplete Alpaca snapshot"):
                workflow_module._commit_alpaca_snapshot_publication(publication)
            manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
            self.assertEqual(set(manifest["Status"]), {"publishing"})
            with self.assertRaisesRegex(OSError, "not committed"):
                validate_alpaca_snapshot(output_dir)

    def test_workflow_snapshot_skips_commit_after_any_publication_failure(self) -> None:
        from leveraged_trader import workflow as workflow_module

        publication = workflow_module._AlpacaSnapshotPublication(
            output_path=Path("unused"),
            generation="generation",
            snapshot_kind="workflow",
            expected_filenames=workflow_module._ALPACA_WORKFLOW_SNAPSHOT_FILENAMES,
            digests={},
        )
        with (
            patch.object(
                workflow_module,
                "_publish_alpaca_snapshot_csv",
                side_effect=OSError("simulated CSV publication failure"),
            ),
            patch.object(workflow_module, "_commit_alpaca_snapshot_publication") as mock_commit,
        ):
            failures = workflow_module._finish_alpaca_workflow_snapshot(
                publication,
                pd.DataFrame(),
                pd.DataFrame(),
                db_path="unused.sqlite",
            )

        self.assertGreaterEqual(len(failures), 1)
        mock_commit.assert_not_called()

    def test_reconciliation_publication_failure_suppresses_secret_bearing_cause(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "publicationCredential7zQ42mN"
        publication = workflow_module._AlpacaSnapshotPublication(
            output_path=Path("unused"),
            generation="generation",
            snapshot_kind="reconciliation",
            expected_filenames=workflow_module._ALPACA_RECONCILIATION_SNAPSHOT_FILENAMES,
            digests={},
        )
        raw_failure = OSError(f"publication reflected {credential}")
        with (
            patch.object(workflow_module, "_prepare_workflow_output_directory_for_publication"),
            patch.object(workflow_module, "_publish_alpaca_snapshot_csv", side_effect=raw_failure),
            patch.object(workflow_module, "_persist_current_alpaca_state"),
            self.assertRaises(AlpacaReconciliationError) as raised,
        ):
            workflow_module._persist_alpaca_reconciliation_snapshot(
                pd.DataFrame(),
                db_path="unused.sqlite",
                output_dir="unused",
                publication=publication,
                alpaca_cfg=AlpacaOrderConfig(
                    enabled=True,
                    api_key_id="paperkey",
                    api_secret_key=credential,
                ),
            )

        rendered_traceback = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn(credential, str(raised.exception))
        self.assertNotIn(credential, raised.exception.results.to_string())
        self.assertNotIn(credential, rendered_traceback)
        self.assertIsNone(raised.exception.__cause__)

    def test_reconciliation_publication_failure_retains_entry_credentials_after_rotation(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "Alpaca"
        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="paperkey",
            api_secret_key=credential,
        )
        publication = workflow_module._AlpacaSnapshotPublication(
            output_path=Path("unused"),
            generation="generation",
            snapshot_kind="reconciliation",
            expected_filenames=workflow_module._ALPACA_RECONCILIATION_SNAPSHOT_FILENAMES,
            digests={},
        )

        def rotate_then_fail(*_args: object, **_kwargs: object) -> None:
            cfg.api_secret_key = "rotated-credential"
            raise OSError(f"publication reflected {credential}")

        with (
            patch.object(workflow_module, "_prepare_workflow_output_directory_for_publication"),
            patch.object(workflow_module, "_publish_alpaca_snapshot_csv", side_effect=rotate_then_fail),
            patch.object(workflow_module, "_persist_current_alpaca_state"),
            self.assertRaises(AlpacaReconciliationError) as raised,
        ):
            workflow_module._persist_alpaca_reconciliation_snapshot(
                pd.DataFrame(),
                db_path="unused.sqlite",
                output_dir="unused",
                publication=publication,
                alpaca_cfg=cfg,
            )

        public_diagnostic = "\n".join(
            (
                str(raised.exception),
                "\n".join(raised.exception.results["Message"].dropna().astype(str)),
                "\n".join(raised.exception.__notes__),
            )
        )
        self.assertNotIn(credential, public_diagnostic)
        self.assertIn("[redacted credential]", public_diagnostic)

    def test_workflow_exception_sanitizer_removes_raw_implicit_context(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "Alpaca"
        try:
            raise ValueError(f"raw {credential} context")
        except ValueError as raw_failure:
            try:
                raise RuntimeError(f"outer {credential} failure")
            except RuntimeError as failure:
                self.assertIs(failure.__context__, raw_failure)
                safe_failure = workflow_module._sanitize_alpaca_workflow_exception(
                    failure,
                    alpaca_cfg=AlpacaOrderConfig(
                        enabled=True,
                        api_key_id="paperkey",
                        api_secret_key=credential,
                    ),
                )

        self.assertNotIn(credential, str(safe_failure))
        self.assertIsNone(safe_failure.__cause__)
        self.assertIsNone(safe_failure.__context__)
        self.assertTrue(safe_failure.__suppress_context__)

    def test_public_async_workflow_boundary_recursively_sanitizes_exception_groups(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "exception-group-secret-7zQ42"
        child = RuntimeError(f"child reflected {credential}")
        child.add_note(f"child note reflected {credential}")
        child.__context__ = ValueError(f"child context reflected {credential}")
        nested = ExceptionGroup(f"nested reflected {credential}", [child])
        original = ExceptionGroup(f"outer reflected {credential}", [nested])
        original.add_note(f"outer note reflected {credential}")

        async def raise_group(**_kwargs: object) -> None:
            raise original

        with (
            patch.object(
                workflow_module,
                "_run_resumable_optimizations_async_from_snapshot_impl",
                new=raise_group,
            ),
            self.assertRaises(ExceptionGroup) as raised,
        ):
            asyncio.run(
                run_resumable_optimizations_async(
                    mode="update",
                    db_path="unused.sqlite",
                    base_cfg=BacktestConfig(),
                    universe_cfg=UniverseConfig(),
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    alpaca_cfg=AlpacaOrderConfig(
                        enabled=True,
                        api_key_id="paper-key",
                        api_secret_key=credential,
                    ),
                    output_dir="unused",
                )
            )

        public_group = raised.exception
        public_nested = public_group.exceptions[0]
        self.assertIsNot(public_group, original)
        self.assertIsInstance(public_nested, ExceptionGroup)
        public_child = public_nested.exceptions[0]
        public_diagnostic = "\n".join(
            (
                str(public_group),
                "\n".join(public_group.__notes__),
                str(public_nested),
                str(public_child),
                "\n".join(public_child.__notes__),
                "".join(traceback.format_exception(public_group)),
            )
        )
        self.assertNotIn(credential, public_diagnostic)
        self.assertIn("[redacted credential]", public_diagnostic)
        self.assertIsNone(public_group.__cause__)
        self.assertIsNone(public_group.__context__)
        self.assertIsNone(public_child.__cause__)
        self.assertIsNone(public_child.__context__)

    def test_workflow_exception_group_sanitizer_preserves_grouped_cancellation(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "cancelled-group-secret-7zQ42"
        cancellation = asyncio.CancelledError(f"cancelled with {credential}")
        original = BaseExceptionGroup(f"group reflected {credential}", [cancellation])

        public_group = workflow_module._sanitize_alpaca_workflow_exception(
            original,
            alpaca_cfg=AlpacaOrderConfig(
                enabled=True,
                api_key_id="paper-key",
                api_secret_key=credential,
            ),
        )

        self.assertIs(type(public_group), BaseExceptionGroup)
        self.assertIs(public_group.exceptions[0], cancellation)
        self.assertIsInstance(public_group.exceptions[0], asyncio.CancelledError)
        self.assertNotIn(credential, str(public_group))
        self.assertNotIn(credential, str(public_group.exceptions[0]))
        self.assertIsNone(public_group.__cause__)
        self.assertIsNone(public_group.__context__)

    def test_workflow_exception_group_sanitizer_preserves_safe_custom_subclass_metadata(self) -> None:
        from leveraged_trader import workflow as workflow_module

        class TaggedGroup(ExceptionGroup):
            def __new__(
                cls,
                message: str,
                children: list[Exception] | tuple[Exception, ...],
                tag: str,
            ) -> TaggedGroup:
                group = super().__new__(cls, message, children)
                group.tag = tag
                return group

            def derive(self, children: tuple[Exception, ...]) -> TaggedGroup:
                return type(self)(self.message, children, self.tag)

        credential = "custom-group-secret-7zQ42"
        original = TaggedGroup(
            f"group reflected {credential}",
            [RuntimeError(f"child reflected {credential}")],
            f"tag reflected {credential}",
        )
        public_group = workflow_module._sanitize_alpaca_workflow_exception(
            original,
            alpaca_cfg=AlpacaOrderConfig(
                enabled=True,
                api_key_id="paper-key",
                api_secret_key=credential,
            ),
        )

        self.assertIs(type(public_group), TaggedGroup)
        self.assertNotIn(credential, public_group.message)
        self.assertNotIn(credential, public_group.tag)
        self.assertNotIn(credential, str(public_group.exceptions[0]))
        derived = public_group.derive(public_group.exceptions)
        self.assertIs(type(derived), TaggedGroup)
        self.assertEqual(derived.tag, public_group.tag)

    def test_workflow_exception_sanitizer_rebuilds_slot_backed_oserror(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "slot-backed-secret-7zQ42"
        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="paper-key",
            api_secret_key=credential,
        )

        def fail() -> None:
            raise FileNotFoundError(2, credential, f"/tmp/{credential}")

        with self.assertRaises(FileNotFoundError) as raised:
            workflow_module._run_alpaca_workflow_diagnostic_boundary(
                fail,
                alpaca_cfg=cfg,
            )

        public_failure = raised.exception
        rendered = "".join(traceback.format_exception(public_failure))
        self.assertIs(type(public_failure), FileNotFoundError)
        self.assertIsNone(public_failure.filename)
        self.assertNotIn(credential, str(public_failure))
        self.assertNotIn(credential, repr(public_failure))
        self.assertNotIn(credential, rendered)
        self.assertIsNone(public_failure.__cause__)
        self.assertIsNone(public_failure.__context__)

    def test_workflow_exception_sanitizer_normalizes_arbitrary_note_shapes(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "non-list-note-secret-7zQ42"
        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="paper-key",
            api_secret_key=credential,
        )
        note_shapes = (
            f"string note {credential}",
            (f"tuple note {credential}",),
            {f"set note {credential}"},
        )

        for raw_notes in note_shapes:
            failure = RuntimeError("safe failure")
            BaseException.__setattr__(failure, "__notes__", raw_notes)
            with self.subTest(note_type=type(raw_notes).__name__):
                public_failure = workflow_module._sanitize_alpaca_workflow_exception(
                    failure,
                    alpaca_cfg=cfg,
                )
                rendered = "".join(traceback.format_exception(public_failure))

                self.assertIs(type(public_failure.__notes__), list)
                self.assertNotIn(credential, "\n".join(public_failure.__notes__))
                self.assertNotIn(credential, rendered)

    def test_workflow_exception_sanitizer_handles_groups_deeper_than_recursion_limit(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "deep-group-secret-7zQ42"
        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="paper-key",
            api_secret_key=credential,
        )
        depth = 1_200
        original: BaseException = RuntimeError(f"leaf {credential}")
        for index in range(depth):
            original = ExceptionGroup(f"level {index} {credential}", [original])

        public_failure = workflow_module._sanitize_alpaca_workflow_exception(
            original,
            alpaca_cfg=cfg,
        )

        observed_depth = 0
        current = public_failure
        while isinstance(current, BaseExceptionGroup):
            self.assertNotIn(credential, current.message)
            self.assertIsNone(current.__cause__)
            self.assertIsNone(current.__context__)
            observed_depth += 1
            current = current.exceptions[0]
        self.assertEqual(observed_depth, depth)
        self.assertNotIn(credential, str(current))
        self.assertIsNone(current.__cause__)
        self.assertIsNone(current.__context__)

    def test_workflow_exception_sanitizer_preserves_system_exit_with_safe_code(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "system-exit-secret-7zQ42"
        original = SystemExit(credential)
        public_failure = workflow_module._sanitize_alpaca_workflow_exception(
            original,
            alpaca_cfg=AlpacaOrderConfig(
                enabled=True,
                api_key_id="paper-key",
                api_secret_key=credential,
            ),
        )

        self.assertIs(type(public_failure), SystemExit)
        self.assertIsNot(public_failure, original)
        self.assertNotIn(credential, str(public_failure))
        self.assertNotIn(credential, str(public_failure.code))
        self.assertIsNone(public_failure.__cause__)
        self.assertIsNone(public_failure.__context__)

    def test_workflow_exception_sanitizer_preserves_none_and_integer_system_exit_codes(self) -> None:
        from leveraged_trader import workflow as workflow_module

        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="paper-key",
            api_secret_key="system-exit-secret-7zQ42",
        )
        for code in (None, 0, 7):
            original = SystemExit(code)
            with self.subTest(code=code):
                public_failure = workflow_module._sanitize_alpaca_workflow_exception(
                    original,
                    alpaca_cfg=cfg,
                )

                self.assertIs(type(public_failure), SystemExit)
                self.assertIs(public_failure.code, code)
                self.assertIs(type(public_failure.code), type(code))

    def test_public_workflow_boundary_drops_secret_bearing_source_traceback(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "traceback-frame-secret-7zQ42"
        compiled_failure = compile(
            "raise RuntimeError('safe failure')",
            f"/tmp/{credential}/broker.py",
            "exec",
        )

        def fail() -> None:
            exec(compiled_failure, {})

        with self.assertRaises(RuntimeError) as raised:
            workflow_module._run_alpaca_workflow_diagnostic_boundary(
                fail,
                alpaca_cfg=AlpacaOrderConfig(
                    enabled=True,
                    api_key_id="paper-key",
                    api_secret_key=credential,
                ),
            )

        rendered = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn(credential, rendered)
        self.assertNotIn("broker.py", rendered)

    def test_workflow_exception_fallback_uses_a_credential_safe_dynamic_type_name(self) -> None:
        from leveraged_trader import workflow as workflow_module

        public_failure = workflow_module._sanitize_alpaca_workflow_exception(
            ValueError("safe failure"),
            alpaca_cfg=AlpacaOrderConfig(
                enabled=True,
                api_key_id="Error",
                api_secret_key="Exception",
            ),
        )
        rendered = "\n".join(
            (
                type(public_failure).__qualname__,
                repr(public_failure),
                "".join(traceback.format_exception(public_failure)),
            )
        )

        self.assertNotIn("Error", rendered)
        self.assertNotIn("Exception", rendered)

    def test_structured_workflow_exception_uses_safe_subclass_when_type_name_collides(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "Alpaca"
        results = pd.DataFrame([{"Status": "error", "Message": "safe failure"}])
        failures = (
            AlpacaBuyBatchError(
                "safe failure",
                results,
                broker_side_effects_possible=True,
            ),
            AlpacaReconciliationError("safe failure", results),
        )
        for original in failures:
            with self.subTest(exception_type=type(original).__name__):
                public_failure = workflow_module._sanitize_alpaca_workflow_exception(
                    original,
                    alpaca_cfg=AlpacaOrderConfig(
                        enabled=True,
                        api_key_id="paper-key",
                        api_secret_key=credential,
                    ),
                )
                rendered = "\n".join(
                    (
                        type(public_failure).__qualname__,
                        repr(public_failure),
                        "".join(traceback.format_exception(public_failure)),
                    )
                )

                self.assertIsInstance(public_failure, type(original))
                self.assertTrue(public_failure.results.equals(results))
                self.assertNotIn(credential, rendered)
                if isinstance(original, AlpacaBuyBatchError):
                    self.assertTrue(public_failure.broker_side_effects_possible)

    def test_workflow_exception_sanitizer_never_certifies_a_stateful_custom_renderer(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "stateful-render-secret-7zQ42"

        class StatefulError(Exception):
            def __init__(self) -> None:
                super().__init__("safe failure")
                self.render_calls = 0

            def __str__(self) -> str:
                self.render_calls += 1
                return "safe failure" if self.render_calls == 1 else credential

        original = StatefulError()
        public_failure = workflow_module._sanitize_alpaca_workflow_exception(
            original,
            alpaca_cfg=AlpacaOrderConfig(
                enabled=True,
                api_key_id="paper-key",
                api_secret_key=credential,
            ),
        )

        self.assertIsNot(public_failure, original)
        self.assertEqual(original.render_calls, 0)
        self.assertNotIn(credential, str(public_failure))
        self.assertNotIn(credential, str(public_failure))

    def test_workflow_scope_retains_exact_client_credentials_from_worker_context(self) -> None:
        from leveraged_trader import alpaca as alpaca_module
        from leveraged_trader import workflow as workflow_module

        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="entry-key",
            api_secret_key="entry-secret",
        )
        transmitted_credential = "intermediate-transmitted-credential"

        async def exercise() -> str:
            with workflow_module._alpaca_public_durable_diagnostic_scope(
                cfg,
                workflow_module._alpaca_config_sensitive_values(cfg),
            ):
                cfg.api_secret_key = transmitted_credential
                await workflow_module._run_blocking(None, alpaca_module.AlpacaClient, cfg)
                cfg.api_secret_key = "current-secret"
                return workflow_module._safe_alpaca_workflow_text(
                    f"worker used {transmitted_credential}",
                    alpaca_cfg=cfg,
                )

        diagnostic = asyncio.run(exercise())

        self.assertNotIn(transmitted_credential, diagnostic)
        self.assertIn("[redacted credential]", diagnostic)

    def test_workflow_broker_compositions_rescan_fixed_prefixes_and_notes(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "Alpaca"
        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="paperkey",
            api_secret_key=credential,
        )
        error = workflow_module._alpaca_reconciliation_followup_error(
            pd.DataFrame([{"Status": "error", "Message": "prior Alpaca diagnostic"}]),
            action="publish the Alpaca audit",
            failure=ValueError(""),
            alpaca_cfg=cfg,
        )
        workflow_module._add_safe_alpaca_workflow_note(
            error,
            "Also failed to publish the Alpaca audit",
            alpaca_cfg=cfg,
        )
        unknown_results = _unknown_alpaca_buy_submission_results(
            pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}]),
            ValueError(""),
            alpaca_cfg=cfg,
        )

        public_diagnostic = "\n".join(
            (
                str(error),
                "\n".join(error.results["Message"].dropna().astype(str)),
                "\n".join(error.__notes__),
                "\n".join(unknown_results["Message"].dropna().astype(str)),
            )
        )
        self.assertNotIn(credential, public_diagnostic)
        self.assertIn("[redacted credential]", public_diagnostic)

    def test_best_effort_alpaca_diagnostic_rescans_empty_exception_type_collision(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "ValueError"
        diagnostic = workflow_module._best_effort_alpaca_exception_diagnostic(
            ValueError(""),
            alpaca_cfg=AlpacaOrderConfig(
                enabled=True,
                api_key_id="paperkey",
                api_secret_key=credential,
            ),
        )

        self.assertNotIn(credential, diagnostic)
        self.assertIn("[redacted credential]", diagnostic)

    def test_loaded_alpaca_snapshot_remains_one_generation_after_new_publication(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            old_publication = workflow_module._begin_alpaca_snapshot_publication(
                output_dir,
                snapshot_kind="current-state",
            )
            for filename in old_publication.expected_filenames:
                workflow_module._publish_alpaca_snapshot_csv(
                    old_publication,
                    pd.DataFrame([{"Generation": "old", "Filename": filename}]),
                    filename=filename,
                    index=False,
                )
            workflow_module._commit_alpaca_snapshot_publication(old_publication)

            snapshot = load_alpaca_snapshot(output_dir)
            new_publication = workflow_module._begin_alpaca_snapshot_publication(
                output_dir,
                snapshot_kind="current-state",
            )
            workflow_module._publish_alpaca_snapshot_csv(
                new_publication,
                pd.DataFrame([{"Generation": "new"}]),
                filename=new_publication.expected_filenames[0],
                index=False,
            )

            self.assertEqual(snapshot.generation, old_publication.generation)
            self.assertEqual(snapshot.snapshot_kind, "current-state")
            self.assertEqual(snapshot.filenames, old_publication.expected_filenames)
            for filename in old_publication.expected_filenames:
                self.assertEqual(snapshot.read_csv(filename)["Generation"].tolist(), ["old"])
            with self.assertRaisesRegex(OSError, "not committed"):
                load_alpaca_snapshot(output_dir)

    def test_final_render_failure_preserves_immediate_post_buy_snapshot(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            publication = workflow_module._begin_alpaca_snapshot_publication(
                output_dir,
                snapshot_kind="workflow",
            )
            for filename in publication.expected_filenames:
                workflow_module._publish_alpaca_snapshot_csv(
                    publication,
                    pd.DataFrame([{"Audit": "committed", "Filename": filename}]),
                    filename=filename,
                    index=False,
                )
            workflow_module._commit_alpaca_snapshot_publication(publication)
            self.assertEqual(validate_alpaca_snapshot(output_dir), publication.generation)

            reporter = Mock(spec=WorkflowReporter)
            reporter.order_results.side_effect = RuntimeError("simulated final rendering failure")
            with (
                patch("leveraged_trader.workflow._begin_alpaca_snapshot_publication") as mock_begin,
                self.assertRaisesRegex(RuntimeError, "simulated final rendering failure"),
            ):
                _write_workflow_outputs(
                    mode="update",
                    db_path=str(Path(tmp) / "state.sqlite"),
                    base_cfg=BacktestConfig(),
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    alpaca_cfg=AlpacaOrderConfig(enabled=True),
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
                    managed_positions=pd.DataFrame(),
                    reconciliation_results=pd.DataFrame(),
                    sell_reconciliation_results=pd.DataFrame(),
                    order_results=pd.DataFrame(),
                    workflow_timer=WorkflowTimer.start(),
                    research_outputs_published=True,
                    broker_snapshot_committed=True,
                )

            mock_begin.assert_not_called()
            self.assertEqual(validate_alpaca_snapshot(output_dir), publication.generation)
            manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
            self.assertEqual(set(manifest["Status"]), {"committed"})

    def test_alpaca_snapshot_load_rejects_publication_started_during_read(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            old_publication = workflow_module._begin_alpaca_snapshot_publication(
                output_dir,
                snapshot_kind="workflow",
            )
            for filename in old_publication.expected_filenames:
                workflow_module._publish_alpaca_snapshot_csv(
                    old_publication,
                    pd.DataFrame([{"Generation": "old"}]),
                    filename=filename,
                    index=False,
                )
            workflow_module._commit_alpaca_snapshot_publication(old_publication)
            self.assertEqual(validate_alpaca_snapshot(output_dir), old_publication.generation)

            reader_reached_first_csv = threading.Event()
            resume_reader = threading.Event()
            real_open = workflow_module._open_verified_alpaca_snapshot_file
            reader_result: dict[str, object] = {}

            def pause_before_first_csv(
                path: Path,
                *,
                directory_descriptor: int | None,
            ) -> tuple[int, os.stat_result]:
                if path.name == old_publication.expected_filenames[0]:
                    reader_reached_first_csv.set()
                    if not resume_reader.wait(timeout=5):
                        raise TimeoutError("snapshot reader was not resumed")
                return real_open(path, directory_descriptor=directory_descriptor)

            def read_snapshot() -> None:
                try:
                    reader_result["snapshot"] = load_alpaca_snapshot(output_dir)
                except BaseException as exc:
                    reader_result["error"] = exc

            with patch(
                "leveraged_trader.workflow._open_verified_alpaca_snapshot_file",
                side_effect=pause_before_first_csv,
            ):
                reader = threading.Thread(target=read_snapshot)
                reader.start()
                self.assertTrue(reader_reached_first_csv.wait(timeout=5))
                new_publication = workflow_module._begin_alpaca_snapshot_publication(
                    output_dir,
                    snapshot_kind="workflow",
                )
                workflow_module._publish_alpaca_snapshot_csv(
                    new_publication,
                    pd.DataFrame([{"Generation": "new"}]),
                    filename=new_publication.expected_filenames[0],
                    index=False,
                )
                resume_reader.set()
                reader.join(timeout=5)

            self.assertFalse(reader.is_alive())
            self.assertNotIn("snapshot", reader_result)
            self.assertIsInstance(reader_result.get("error"), OSError)
            self.assertIn("does not match the committed manifest", str(reader_result["error"]))

    def test_alpaca_snapshot_manifest_rejects_abruptly_interrupted_mixed_generation(self) -> None:
        script = r"""
import os
import sys
from pathlib import Path

import pandas as pd
import leveraged_trader.workflow as workflow

output_path = Path(sys.argv[1])
output_path.mkdir(mode=0o700)

def publish(label):
    publication = workflow._begin_alpaca_snapshot_publication(
        output_path,
        snapshot_kind="workflow",
    )
    for filename in publication.expected_filenames:
        workflow._publish_alpaca_snapshot_csv(
            publication,
            pd.DataFrame([{"Generation": label}]),
            filename=filename,
            index=False,
        )
    workflow._commit_alpaca_snapshot_publication(publication)

publish("old")
workflow.validate_alpaca_snapshot(output_path)
real_atomic_to_csv = workflow._atomic_to_csv
published_broker_files = 0

def exit_after_second_broker_csv(frame, destination, *, index=True):
    global published_broker_files
    digest = real_atomic_to_csv(frame, destination, index=index)
    if destination.name in workflow._ALPACA_WORKFLOW_SNAPSHOT_FILENAMES:
        published_broker_files += 1
        if published_broker_files == 2:
            os._exit(73)
    return digest

workflow._atomic_to_csv = exit_after_second_broker_csv
publish("new")
"""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [sys.executable, "-c", script, str(output_dir)],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 73, result.stderr)
            observed_generations = {
                pd.read_csv(output_dir / filename)["Generation"].iloc[0]
                for filename in (
                    "alpaca_order_results.csv",
                    "alpaca_reconciliation_results.csv",
                    "managed_positions.csv",
                )
            }
            self.assertEqual(observed_generations, {"old", "new"})
            manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
            self.assertEqual(set(manifest["Status"]), {"publishing"})
            with self.assertRaisesRegex(OSError, "not committed"):
                validate_alpaca_snapshot(output_dir)

    def test_current_alpaca_state_snapshot_uses_one_sqlite_read_generation(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            output_dir = Path(tmp) / "outputs"
            _initialize_state_db(str(db_path))
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO alpaca_managed_positions
                    (workflow, symbol, signal_symbol, buy_rsi, profit_target_multiple,
                     buy_signal_date, buy_client_order_id, buy_alpaca_order_id,
                     buy_order_qty, buy_order_limit_price, buy_status, filled_qty,
                     filled_avg_price, filled_at, sold_qty, sold_value, remaining_qty)
                    VALUES ('Long', 'TQQQ', 'QQQ', 30, 1.5,
                            '2026-01-02', 'rsi-buy-TQQQ-20260102', 'buy-1',
                            10, 100, 'filled', 10, 100,
                            '2026-01-02T15:00:00Z', 0, 0, 10)
                    """
                )

            real_managed_loader = workflow_module._load_alpaca_managed_positions_for_db
            observed_snapshot_connections: list[sqlite3.Connection | None] = []

            def load_managed_then_close_position(
                requested_db_path: str,
                *,
                connection: sqlite3.Connection | None = None,
            ) -> pd.DataFrame:
                observed_snapshot_connections.append(connection)
                managed = real_managed_loader(
                    requested_db_path,
                    connection=connection,
                )
                with sqlite3.connect(db_path) as writer:
                    writer.execute(
                        """
                        UPDATE alpaca_managed_positions
                        SET sell_status = 'filled',
                            sell_filled_qty = 10,
                            sell_filled_avg_price = 110,
                            sell_filled_at = '2026-01-03T15:00:00Z',
                            sold_qty = 10,
                            sold_value = 1100,
                            remaining_qty = 0,
                            realized_pl = 100,
                            realized_pl_pct = 10,
                            closed_at = '2026-01-03T15:00:00Z'
                        """
                    )
                return managed

            with patch.object(
                workflow_module,
                "_load_alpaca_managed_positions_for_db",
                side_effect=load_managed_then_close_position,
            ):
                workflow_module._persist_current_alpaca_state(
                    db_path=str(db_path),
                    output_path=output_dir,
                )

            snapshot = load_alpaca_snapshot(output_dir)
            managed = snapshot.read_csv("managed_positions.csv")
            realized_pnl = snapshot.read_csv("alpaca_realized_pnl.csv")
            with sqlite3.connect(db_path) as conn:
                persisted_closed_at = conn.execute("SELECT closed_at FROM alpaca_managed_positions").fetchone()[0]

        self.assertEqual(len(observed_snapshot_connections), 1)
        self.assertIsNotNone(observed_snapshot_connections[0])
        self.assertTrue(managed["closed_at"].isna().all())
        self.assertEqual(realized_pnl["Workflow"].tolist(), ["Total"])
        self.assertEqual(realized_pnl["Closed Positions"].tolist(), [0])
        self.assertEqual(persisted_closed_at, "2026-01-03T15:00:00Z")

    def test_current_alpaca_state_snapshot_sanitizes_buy_quarantine_collision(self) -> None:
        from leveraged_trader import workflow as workflow_module

        credential = "Alpaca"
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            output_dir = Path(tmp) / "outputs"
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                    buy_alpaca_order_id="buy-1",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_order_qty=1,
                    buy_order_limit_price=100,
                    buy_status="filled",
                )
                mark_alpaca_managed_buy_filled(
                    conn,
                    position_id,
                    buy_status="filled",
                    filled_qty=2,
                    filled_avg_price=90,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=135,
                )
                raw_quarantine = conn.execute(
                    "SELECT buy_causality_quarantine FROM alpaca_managed_positions WHERE id = ?",
                    (position_id,),
                ).fetchone()[0]

            workflow_module._persist_current_alpaca_state(
                db_path=str(db_path),
                output_path=output_dir,
                alpaca_cfg=AlpacaOrderConfig(
                    sell_enabled=True,
                    api_key_id="paper-key",
                    api_secret_key=credential,
                ),
            )
            published = load_alpaca_snapshot(output_dir).read_csv("managed_positions.csv")

        self.assertIn(credential, raw_quarantine)
        self.assertNotIn(credential, published.loc[0, "buy_causality_quarantine"])
        self.assertIn("[redacted credential]", published.loc[0, "buy_causality_quarantine"])

    def test_reconciliation_only_refreshes_managed_outputs_without_strategy_work(self) -> None:
        reconciliation = pd.DataFrame([{"Action": "sell", "Status": "accepted", "Asset": "TQQQ"}])
        managed = pd.DataFrame([{"id": 1, "symbol": "TQQQ", "buy_client_order_id": "buy-1"}])
        pnl = pd.DataFrame([{"Workflow": "Long", "Closed Positions": 1}])
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir()
            for filename in _RESEARCH_REPORT_FILENAMES:
                (output_dir / filename).write_text("Generation\nprior-research\n", encoding="utf-8")
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=reconciliation,
                ) as mock_reconcile,
                patch(
                    "leveraged_trader.workflow._load_alpaca_managed_positions_for_db",
                    return_value=managed,
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_realized_pnl_for_db",
                    return_value=pnl,
                ),
            ):
                run_alpaca_reconciliation(
                    db_path=str(Path(tmp) / "state.sqlite"),
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="key",
                        api_secret_key="secret",
                        buy_limit_buffer_bps=float("nan"),
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                )

            self.assertEqual(mock_reconcile.call_count, 1)
            self.assertEqual(
                pd.read_csv(output_dir / "alpaca_reconciliation_results.csv")["Status"].tolist(),
                ["accepted"],
            )
            self.assertEqual(
                pd.read_csv(output_dir / "alpaca_realized_pnl.csv")["Workflow"].tolist(),
                ["Long"],
            )
            self.assertRegex(validate_alpaca_snapshot(output_dir), r"^[0-9a-f]{32}$")
            for filename in _RESEARCH_REPORT_FILENAMES:
                self.assertEqual(
                    (output_dir / filename).read_text(encoding="utf-8"),
                    "Generation\nprior-research\n",
                )

    def test_scheduled_reconciliation_idle_gate_preserves_reports_and_skips_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir()
            sentinel = output_dir / "alpaca_reconciliation_results.csv"
            sentinel.write_text("Status\nprior-success\n", encoding="utf-8")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                position_id = save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    alpaca_asset_id="asset-tqqq",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-idle",
                    buy_alpaca_order_id="buy-idle",
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
                    filled_avg_price=100,
                    filled_at="2026-01-02T14:31:00Z",
                    target_sell_price=150,
                )
                conn.execute(
                    "UPDATE alpaca_managed_positions SET closed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (position_id,),
                )
                position = load_alpaca_managed_positions(conn).iloc[0]
                self.assertTrue(
                    mark_alpaca_closed_correction_audited(
                        conn,
                        position_id,
                        expected_state_revision=int(position["state_revision"]),
                    )
                )

            with (
                patch("leveraged_trader.workflow._initialize_state_db") as initialize,
                patch("leveraged_trader.workflow._begin_alpaca_snapshot_publication") as begin_publication,
                patch("leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db") as reconcile,
                patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                run_alpaca_reconciliation(
                    db_path=str(db_path),
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="paper-key",
                        api_secret_key="paper-secret",
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                    closed_audit_min_interval_minutes=15,
                )

            initialize.assert_not_called()
            begin_publication.assert_not_called()
            reconcile.assert_not_called()
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "Status\nprior-success\n")

    def test_scheduled_reconciliation_due_gate_tracks_active_unresolved_and_audit_age(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        init_state_db(conn)
        position_id = save_alpaca_managed_buy_order(
            conn,
            symbol="TQQQ",
            alpaca_asset_id="asset-tqqq",
            signal_symbol="QQQ",
            buy_rsi=30,
            profit_target_multiple=1.5,
            buy_signal_date="2026-01-02",
            buy_client_order_id="rsi-buy-TQQQ-due-gate",
            buy_alpaca_order_id="buy-due-gate",
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
            filled_avg_price=100,
            filled_at="2026-01-02T14:31:00Z",
            target_sell_price=150,
        )

        @contextmanager
        def in_memory_probe(_db_path: str):
            yield conn

        with patch("leveraged_trader.workflow._read_only_state_connection", new=in_memory_probe):
            self.assertTrue(
                _scheduled_alpaca_reconciliation_is_due(
                    "unused.sqlite",
                    closed_audit_min_interval_minutes=15,
                )
            )
            conn.execute(
                "UPDATE alpaca_managed_positions SET closed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (position_id,),
            )
            conn.commit()
            self.assertTrue(
                _scheduled_alpaca_reconciliation_is_due(
                    "unused.sqlite",
                    closed_audit_min_interval_minutes=15,
                )
            )
            position = load_alpaca_managed_positions(conn).iloc[0]
            self.assertTrue(
                mark_alpaca_closed_correction_audited(
                    conn,
                    position_id,
                    expected_state_revision=int(position["state_revision"]),
                )
            )
            self.assertFalse(
                _scheduled_alpaca_reconciliation_is_due(
                    "unused.sqlite",
                    closed_audit_min_interval_minutes=15,
                )
            )
            conn.execute(
                "UPDATE alpaca_managed_positions "
                "SET closed_correction_audited_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-15 minutes') "
                "WHERE id = ?",
                (position_id,),
            )
            conn.commit()
            self.assertTrue(
                _scheduled_alpaca_reconciliation_is_due(
                    "unused.sqlite",
                    closed_audit_min_interval_minutes=15,
                )
            )
            conn.execute(
                "UPDATE alpaca_managed_positions "
                "SET buy_status = 'pending_cancel', "
                "closed_correction_audited_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
                (position_id,),
            )
            conn.commit()
            self.assertTrue(
                _scheduled_alpaca_reconciliation_is_due(
                    "unused.sqlite",
                    closed_audit_min_interval_minutes=15,
                )
            )

    def test_scheduled_reconciliation_due_gate_never_skips_unknown_state(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)

        @contextmanager
        def incomplete_probe(_db_path: str):
            yield conn

        with patch("leveraged_trader.workflow._read_only_state_connection", new=incomplete_probe):
            self.assertTrue(
                _scheduled_alpaca_reconciliation_is_due(
                    "unused.sqlite",
                    closed_audit_min_interval_minutes=15,
                )
            )
        with (
            patch(
                "leveraged_trader.workflow._read_only_state_connection",
                side_effect=OSError("database identity changed"),
            ),
            self.assertRaisesRegex(OSError, "identity changed"),
        ):
            _scheduled_alpaca_reconciliation_is_due(
                "unused.sqlite",
                closed_audit_min_interval_minutes=15,
            )

    def test_historical_sell_cancellation_is_rechecked_once_after_broker_confirmation_delay(self) -> None:
        pending = pd.DataFrame(
            [
                {
                    "Action": "sell",
                    "Status": "pending_cancel",
                    "Message": (
                        "final managed-sell submission blocked: historical managed sell exposure is still "
                        "executable; historical exposure cancellation requires confirmation: sell-old"
                    ),
                }
            ]
        )
        recovered = pd.DataFrame([{"Action": "sell", "Status": "renewed", "Message": "protected"}])
        with (
            patch(
                "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db_impl",
                side_effect=[
                    AlpacaReconciliationError("pending", pending, retryable_historical_cancellation=True),
                    recovered,
                ],
            ) as reconcile,
            patch("leveraged_trader.workflow.time.sleep") as sleep,
        ):
            result = _reconcile_alpaca_managed_positions_with_cancellation_retry(
                "state.sqlite", AlpacaOrderConfig(), 15
            )

        self.assertIs(result, recovered)
        self.assertEqual(reconcile.call_count, 2)
        self.assertEqual([call.args[2] for call in reconcile.call_args_list], [15, 15])
        sleep.assert_called_once_with(30)

    def test_historical_sell_cancellation_retry_retains_unresolved_failure(self) -> None:
        pending = pd.DataFrame(
            [
                {
                    "Action": "sell",
                    "Status": "pending_cancel",
                    "Message": (
                        "final managed-sell submission blocked: historical managed sell exposure is still "
                        "executable; historical exposure cancellation requires confirmation: sell-old"
                    ),
                }
            ]
        )
        first = AlpacaReconciliationError("pending", pending, retryable_historical_cancellation=True)
        unresolved = AlpacaReconciliationError("still pending", pending)
        with (
            patch(
                "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db_impl",
                side_effect=[first, unresolved],
            ) as reconcile,
            patch("leveraged_trader.workflow.time.sleep") as sleep,
            self.assertRaises(AlpacaReconciliationError) as raised,
        ):
            _reconcile_alpaca_managed_positions_with_cancellation_retry("state.sqlite", AlpacaOrderConfig())

        self.assertIs(raised.exception, unresolved)
        self.assertEqual(len(raised.exception.results), 2)
        self.assertTrue(raised.exception.results.loc[0, "Message"].startswith("Initial reconciliation"))
        self.assertEqual(raised.exception.results.loc[1, "Message"], pending.loc[0, "Message"])
        self.assertEqual(reconcile.call_count, 2)
        sleep.assert_called_once_with(30)

    def test_historical_cancellation_retry_retains_migration_and_completed_actions(self) -> None:
        pending = pd.DataFrame(
            [
                {"Action": "sell", "Status": "pending_cancel", "Message": "historical cancellation requested"},
                {"Action": "sell", "Status": "accepted_for_bidding", "Message": "protective sell accepted"},
                {"Action": "sell", "Status": "closed", "Message": "position closed"},
                {"Action": "buy", "Status": "canceled", "Message": "unfilled buy terminated"},
            ]
        )
        recovered = pd.DataFrame([{"Action": "sell", "Status": "renewed", "Message": "protected"}])
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            _initialize_state_db(db_path)
            with closing(sqlite3.connect(db_path)) as conn, conn:
                save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-cancel-retry",
                    buy_alpaca_order_id="buy-1",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                )
            with (
                patch(
                    "leveraged_trader.workflow.migrate_alpaca_managed_position_symbols",
                    side_effect=[{"OLD": "TQQQ"}, {}],
                ),
                patch(
                    "leveraged_trader.workflow.reconcile_alpaca_managed_positions",
                    side_effect=[
                        AlpacaReconciliationError("pending", pending, retryable_historical_cancellation=True),
                        recovered,
                    ],
                ) as reconcile,
                patch("leveraged_trader.workflow.time.sleep") as sleep,
            ):
                result = _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())

        self.assertEqual(
            result["Status"].tolist(),
            ["symbol_migrated", "accepted_for_bidding", "closed", "canceled", "renewed"],
        )
        self.assertTrue(result.iloc[:-1]["Message"].str.startswith("Initial reconciliation").all())
        self.assertEqual(result.iloc[-1]["Message"], "protected")
        self.assertEqual(reconcile.call_count, 2)
        sleep.assert_called_once_with(30)

    def test_historical_cancellation_retry_preserves_first_audit_after_unexpected_failure(self) -> None:
        pending = pd.DataFrame(
            [{"Action": "sell", "Status": "pending_cancel", "Message": "historical cancellation requested"}]
        )
        for during_delay in (False, True):
            with self.subTest(during_delay=during_delay):
                first = AlpacaReconciliationError("pending", pending, retryable_historical_cancellation=True)
                with (
                    patch(
                        "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db_impl",
                        side_effect=[first, OSError("cannot open database")],
                    ) as reconcile,
                    patch(
                        "leveraged_trader.workflow.time.sleep",
                        side_effect=KeyboardInterrupt("interrupted") if during_delay else None,
                    ),
                    self.assertRaises(AlpacaReconciliationError) as raised,
                ):
                    _reconcile_alpaca_managed_positions_for_db("state.sqlite", AlpacaOrderConfig())

                self.assertEqual(raised.exception.results["Status"].tolist(), ["pending_cancel", "error"])
                self.assertTrue(raised.exception.results.loc[0, "Message"].startswith("Initial reconciliation"))
                self.assertIn("recheck historical sell cancellation", raised.exception.results.loc[1, "Message"])
                self.assertEqual(reconcile.call_count, 1 if during_delay else 2)

    def test_historical_cancellation_retry_preserves_first_audit_after_typed_failure(self) -> None:
        initial = pd.DataFrame(
            [
                {"Action": "sell", "Status": "pending_cancel", "Message": "cancellation requested"},
                {"Action": "sell", "Status": "renewed", "Message": "other position protected"},
            ]
        )
        failed = AlpacaReconciliationError(
            "broker unavailable", pd.DataFrame([{"Action": "reconcile", "Status": "error", "Message": "503"}])
        )
        with (
            patch(
                "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db_impl",
                side_effect=[
                    AlpacaReconciliationError("pending", initial, retryable_historical_cancellation=True),
                    failed,
                ],
            ),
            patch("leveraged_trader.workflow.time.sleep"),
            self.assertRaises(AlpacaReconciliationError) as raised,
        ):
            _reconcile_alpaca_managed_positions_for_db("state.sqlite", AlpacaOrderConfig())

        self.assertIs(raised.exception, failed)
        self.assertEqual(failed.results["Status"].tolist(), ["pending_cancel", "renewed", "error"])
        self.assertTrue(failed.results.iloc[:2]["Message"].str.startswith("Initial reconciliation").all())

    def test_other_protective_reconciliation_errors_are_not_delayed(self) -> None:
        failed = AlpacaReconciliationError(
            "unprotected",
            pd.DataFrame([{"Action": "sell", "Status": "quantity_mismatch", "Message": "wrong quantity"}]),
        )
        with (
            patch(
                "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db_impl",
                side_effect=failed,
            ) as reconcile,
            patch("leveraged_trader.workflow.time.sleep") as sleep,
            self.assertRaises(AlpacaReconciliationError) as raised,
        ):
            _reconcile_alpaca_managed_positions_with_cancellation_retry("state.sqlite", AlpacaOrderConfig())

        self.assertIs(raised.exception, failed)
        reconcile.assert_called_once()
        sleep.assert_not_called()

    def test_historical_cancellation_does_not_delay_an_independent_error(self) -> None:
        for independent_status in ("error", "pending_cancel", "filled"):
            with self.subTest(independent_status=independent_status):
                failed = AlpacaReconciliationError(
                    "multiple failures",
                    pd.DataFrame(
                        [
                            {
                                "Action": "sell",
                                "Status": "pending_cancel",
                                "Message": (
                                    "final managed-sell submission blocked: historical managed sell exposure is "
                                    "still executable; historical exposure cancellation requires confirmation: "
                                    "sell-old"
                                ),
                            },
                            {
                                "Action": "reconcile",
                                "Status": independent_status,
                                "Message": "broker unavailable",
                            },
                        ]
                    ),
                )
                with (
                    patch(
                        "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db_impl",
                        side_effect=failed,
                    ) as reconcile,
                    patch("leveraged_trader.workflow.time.sleep") as sleep,
                    self.assertRaises(AlpacaReconciliationError) as raised,
                ):
                    _reconcile_alpaca_managed_positions_with_cancellation_retry("state.sqlite", AlpacaOrderConfig())

                self.assertIs(raised.exception, failed)
                reconcile.assert_called_once()
                sleep.assert_not_called()

    def test_reconciliation_only_persists_systemic_failure_rows_before_raising(self) -> None:
        reconciliation = pd.DataFrame(
            [
                {
                    "Position ID": 1,
                    "Workflow": "Long",
                    "Asset": "TQQQ",
                    "Action": "reconcile",
                    "Status": "error",
                    "Message": "401 unauthorized",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                save_alpaca_managed_buy_order(
                    conn,
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                    buy_alpaca_order_id="buy-1",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                )
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    side_effect=AlpacaReconciliationError("systemic failure", reconciliation),
                ),
                self.assertRaises(AlpacaReconciliationError),
            ):
                run_alpaca_reconciliation(
                    db_path=db_path,
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="key",
                        api_secret_key="secret",
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                )

            persisted = pd.read_csv(output_dir / "alpaca_reconciliation_results.csv")
            self.assertEqual(persisted["Status"].tolist(), ["error"])
            self.assertEqual(persisted["Asset"].tolist(), ["TQQQ"])
            self.assertEqual(pd.read_csv(output_dir / "managed_positions.csv")["symbol"].tolist(), ["TQQQ"])
            self.assertTrue((output_dir / "alpaca_realized_pnl.csv").is_file())

    def test_reconciliation_only_retains_success_rows_when_state_snapshot_fails(self) -> None:
        reconciliation = pd.DataFrame([{"Action": "sell", "Status": "accepted", "Asset": "TQQQ"}])
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            with (
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=reconciliation,
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_managed_positions_for_db",
                    side_effect=OSError("managed snapshot failed"),
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_realized_pnl_for_db",
                    return_value=pd.DataFrame(),
                ),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                run_alpaca_reconciliation(
                    db_path=str(Path(tmp) / "state.sqlite"),
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="key",
                        api_secret_key="secret",
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                )

            self.assertEqual(raised.exception.results["Status"].tolist(), ["accepted", "error"])
            persisted = pd.read_csv(output_dir / "alpaca_reconciliation_results.csv")
            self.assertEqual(persisted["Status"].tolist(), ["accepted", "error"])
            self.assertTrue((output_dir / "alpaca_sell_order_results.csv").is_file())
            self.assertTrue((output_dir / "alpaca_realized_pnl.csv").is_file())

    def test_reconciliation_only_continues_snapshot_after_audit_write_failure(self) -> None:
        reconciliation = pd.DataFrame([{"Action": "sell", "Status": "accepted", "Asset": "TQQQ"}])

        def fail_reconciliation_audit(
            frame: pd.DataFrame,
            destination: Path,
            *,
            index: bool = True,
        ) -> str:
            if destination.name == "alpaca_reconciliation_results.csv":
                raise OSError("audit write failed")
            return _atomic_to_csv(frame, destination, index=index)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            with (
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=reconciliation,
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame([{"symbol": "TQQQ"}]),
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_realized_pnl_for_db",
                    return_value=pd.DataFrame(),
                ),
                patch("leveraged_trader.workflow._atomic_to_csv", side_effect=fail_reconciliation_audit),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                run_alpaca_reconciliation(
                    db_path=str(Path(tmp) / "state.sqlite"),
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="key",
                        api_secret_key="secret",
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                )

            self.assertEqual(raised.exception.results["Status"].tolist(), ["accepted", "error"])
            self.assertTrue((output_dir / "alpaca_sell_order_results.csv").is_file())
            self.assertTrue((output_dir / "managed_positions.csv").is_file())
            self.assertTrue((output_dir / "alpaca_realized_pnl.csv").is_file())

    def test_reconciliation_only_normalizes_status_teardown_after_snapshot(self) -> None:
        reconciliation = pd.DataFrame([{"Action": "sell", "Status": "accepted", "Asset": "TQQQ"}])

        @contextmanager
        def failing_status(_reporter: WorkflowReporter, _message: str):
            yield
            raise OSError("status teardown failed")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            with (
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=reconciliation,
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(),
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_realized_pnl_for_db",
                    return_value=pd.DataFrame(),
                ),
                patch.object(WorkflowReporter, "status", new=failing_status),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                run_alpaca_reconciliation(
                    db_path=str(Path(tmp) / "state.sqlite"),
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="key",
                        api_secret_key="secret",
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                )

            self.assertEqual(raised.exception.results["Status"].tolist(), ["accepted", "error"])
            self.assertIn("status teardown failed", raised.exception.results.iloc[-1]["Message"])
            self.assertEqual(
                pd.read_csv(output_dir / "alpaca_reconciliation_results.csv")["Status"].tolist(),
                ["accepted"],
            )

    def test_reconciliation_only_reuses_preinvalidated_publication_and_preserves_typed_error(self) -> None:
        from leveraged_trader import workflow as workflow_module

        reconciliation_results = pd.DataFrame(
            [{"Action": "sell", "Status": "error", "Asset": "TQQQ", "Message": "broker failure"}]
        )
        reconciliation_error = AlpacaReconciliationError("reconciliation failed", reconciliation_results)
        structured_results = reconciliation_error.results

        @contextmanager
        def failing_status(_reporter: WorkflowReporter, _message: str):
            try:
                yield
            finally:
                raise OSError("status teardown failed")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            old_publication = workflow_module._begin_alpaca_snapshot_publication(
                output_dir,
                snapshot_kind="reconciliation",
            )
            for filename in old_publication.expected_filenames:
                workflow_module._publish_alpaca_snapshot_csv(
                    old_publication,
                    pd.DataFrame([{"Generation": "old"}]),
                    filename=filename,
                    index=False,
                )
            workflow_module._commit_alpaca_snapshot_publication(old_publication)

            publications: list[object] = []
            real_begin = workflow_module._begin_alpaca_snapshot_publication
            real_initialize = workflow_module._initialize_state_db

            def begin_publication(*args: object, **kwargs: object):
                publication = real_begin(*args, **kwargs)
                publications.append(publication)
                return publication

            def initialize_after_observing_invalidation(db_path: str) -> None:
                manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
                self.assertEqual(set(manifest["Status"]), {"publishing"})
                self.assertNotEqual(set(manifest["Generation"]), {old_publication.generation})
                real_initialize(db_path)

            def fail_after_observing_invalidation(*_args: object) -> pd.DataFrame:
                manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
                self.assertEqual(set(manifest["Status"]), {"publishing"})
                self.assertNotEqual(set(manifest["Generation"]), {old_publication.generation})
                raise reconciliation_error

            with (
                patch.object(workflow_module, "_begin_alpaca_snapshot_publication", side_effect=begin_publication),
                patch.object(
                    workflow_module,
                    "_initialize_state_db",
                    side_effect=initialize_after_observing_invalidation,
                ),
                patch.object(
                    workflow_module,
                    "_reconcile_alpaca_managed_positions_for_db",
                    side_effect=fail_after_observing_invalidation,
                ),
                patch.object(WorkflowReporter, "status", new=failing_status),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                run_alpaca_reconciliation(
                    db_path=str(Path(tmp) / "state.sqlite"),
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="key",
                        api_secret_key="secret",
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                )

            self.assertIs(raised.exception, reconciliation_error)
            self.assertIs(raised.exception.results, structured_results)
            self.assertEqual(len(publications), 1)
            self.assertEqual(validate_alpaca_snapshot(output_dir), publications[0].generation)
            self.assertIn(
                "status teardown failed",
                "\n".join(getattr(raised.exception, "__notes__", ())),
            )

    def test_reconciliation_only_termination_leaves_preinvalidated_manifest(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            old_publication = workflow_module._begin_alpaca_snapshot_publication(
                output_dir,
                snapshot_kind="reconciliation",
            )
            for filename in old_publication.expected_filenames:
                workflow_module._publish_alpaca_snapshot_csv(
                    old_publication,
                    pd.DataFrame([{"Generation": "old"}]),
                    filename=filename,
                    index=False,
                )
            workflow_module._commit_alpaca_snapshot_publication(old_publication)

            def terminate_after_observing_invalidation(*_args: object) -> pd.DataFrame:
                manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
                self.assertEqual(set(manifest["Status"]), {"publishing"})
                self.assertNotEqual(set(manifest["Generation"]), {old_publication.generation})
                raise KeyboardInterrupt("simulated termination")

            with (
                patch.object(
                    workflow_module,
                    "_reconcile_alpaca_managed_positions_for_db",
                    side_effect=terminate_after_observing_invalidation,
                ),
                self.assertRaisesRegex(KeyboardInterrupt, "simulated termination"),
            ):
                run_alpaca_reconciliation(
                    db_path=str(Path(tmp) / "state.sqlite"),
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="key",
                        api_secret_key="secret",
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                )

            manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
            self.assertEqual(set(manifest["Status"]), {"publishing"})
            with self.assertRaisesRegex(OSError, "not committed"):
                validate_alpaca_snapshot(output_dir)

    def test_reconciliation_wrapper_preserves_rows_when_transaction_finalization_fails(self) -> None:
        reconciliation = pd.DataFrame([{"Action": "sell", "Status": "accepted", "Asset": "TQQQ"}])

        @contextmanager
        def state_connection_with_failing_exit(*_args: object, **_kwargs: object):
            yield Mock()
            raise OSError("commit finalization failed")

        with (
            patch("leveraged_trader.workflow._state_connection", new=state_connection_with_failing_exit),
            patch("leveraged_trader.workflow.migrate_alpaca_managed_position_symbols", return_value={}),
            patch(
                "leveraged_trader.workflow.load_alpaca_managed_positions",
                return_value=pd.DataFrame(),
            ),
            patch(
                "leveraged_trader.workflow.reconcile_alpaca_managed_positions",
                return_value=reconciliation,
            ),
            self.assertRaises(AlpacaReconciliationError) as raised,
        ):
            _reconcile_alpaca_managed_positions_for_db(
                "unused.sqlite",
                AlpacaOrderConfig(
                    sell_enabled=True,
                    api_key_id="key",
                    api_secret_key="secret",
                ),
            )

        self.assertEqual(raised.exception.results["Status"].tolist(), ["accepted", "error"])
        self.assertIn("commit finalization failed", raised.exception.results.iloc[-1]["Message"])

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

    @patch("leveraged_trader.workflow.os.lstat")
    def test_direct_workflow_paths_reject_controls_before_filesystem_access(self, mock_lstat: Mock) -> None:
        for validator, value, message in (
            (_validate_database_path, "state\nforged.sqlite", "control or nonprintable"),
            (_validate_database_path, "state\x1b[31m.sqlite", "control or nonprintable"),
            (_validate_output_directory_path, "reports\rforged", "control or nonprintable"),
            (_validate_output_directory_path, "\t", "nonempty filesystem path"),
        ):
            with (
                self.subTest(validator=validator.__name__, value=repr(value)),
                self.assertRaisesRegex(ValueError, message),
            ):
                validator(value)

        mock_lstat.assert_not_called()

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

    def test_missing_database_with_trailing_separator_is_rejected_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "state.sqlite"
            supplied_database_path = f"{database_path}{os.sep}"
            output_dir = Path(tmp) / "reports"

            with (
                self.assertRaisesRegex(ValueError, "must not end with a path separator"),
                _workflow_run_lock(supplied_database_path, str(output_dir)),
            ):
                self.fail("workflow accepted a database path ending in a separator")

            self.assertFalse(database_path.exists())
            self.assertFalse(Path(f"{database_path}.lock").exists())
            self.assertFalse(output_dir.exists())

    def test_missing_output_component_before_parent_traversal_is_rejected_without_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database_path = root / "state.sqlite"
            output_dir = os.path.join(
                os.fspath(root),
                "missing",
                os.pardir,
                "reports",
            )

            with (
                self.assertRaisesRegex(FileNotFoundError, "cannot traverse '\\.\\.'"),
                _workflow_run_lock(str(database_path), output_dir),
            ):
                self.fail("workflow accepted parent traversal after a missing component")

            self.assertEqual(list(root.iterdir()), [])

    def test_missing_database_with_terminal_dot_component_is_rejected_without_artifacts(self) -> None:
        for terminal_component in (os.curdir, os.pardir):
            with self.subTest(terminal_component=terminal_component), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                database_path = root / "database" / "state.sqlite"
                supplied_database_path = f"{database_path}{os.sep}{terminal_component}"
                output_dir = root / "reports"

                with (
                    self.assertRaisesRegex(ValueError, "must not end with a '.' or '..' path component"),
                    _workflow_run_lock(supplied_database_path, str(output_dir)),
                ):
                    self.fail("workflow accepted a database path ending in a dot component")

                self.assertEqual(list(root.iterdir()), [])

    def test_windows_trailing_dot_or_space_aliases_are_rejected_before_lock_setup(self) -> None:
        invalid_database_paths = (
            r"state.sqlite.",
            "state.sqlite ",
            r"directory.\state.sqlite",
            r"directory \state.sqlite",
            r"C:\state\state.sqlite.",
            r"\\server\share\directory.\state.sqlite",
        )
        for database_path in invalid_database_paths:
            with (
                self.subTest(database_path=database_path),
                patch("leveraged_trader.workflow.os.name", "nt"),
                patch("leveraged_trader.workflow._prepare_workflow_user_lock_directory") as prepare_lock_directory,
                self.assertRaisesRegex(ValueError, "Windows path component ending in a space or period"),
                _workflow_run_lock(database_path, "reports"),
            ):
                self.fail("workflow accepted an aliasing Windows database path")
            prepare_lock_directory.assert_not_called()

        with (
            patch("leveraged_trader.workflow.os.name", "nt"),
            patch("leveraged_trader.workflow._prepare_workflow_user_lock_directory") as prepare_lock_directory,
            self.assertRaisesRegex(ValueError, "Windows path component ending in a space or period"),
            _workflow_run_lock("state.sqlite", r"reports.\daily"),
        ):
            self.fail("workflow accepted an aliasing Windows output path")
        prepare_lock_directory.assert_not_called()

        for valid_database_path in (
            r"C:\strategy.v2\state.sqlite",
            r"strategy\..\state.sqlite",
            r"\\server\share\strategy.v2\state.sqlite",
        ):
            with (
                self.subTest(valid_database_path=valid_database_path),
                patch("leveraged_trader.workflow.os.name", "nt"),
            ):
                _validate_database_path(valid_database_path)

    def test_existing_nonregular_database_is_rejected_before_stale_reports_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            db_path.mkdir()
            output_dir = Path(tmp) / "reports"
            output_dir.mkdir()
            stale_report = output_dir / "buy_signals.csv"
            stale_report.write_text("must remain unchanged\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must name a regular file"):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(db_path),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(output_dir),
                    )
                )

            self.assertEqual(stale_report.read_text(encoding="utf-8"), "must remain unchanged\n")
            self.assertFalse((output_dir / ".leveraged-trader.lock").exists())

    def test_database_runtime_rejection_precedes_stale_report_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            output_dir = Path(tmp) / "reports"
            output_dir.mkdir()
            stale_report = output_dir / "buy_signals.csv"
            stale_report.write_text("must remain unchanged\n", encoding="utf-8")

            with (
                patch(
                    "leveraged_trader.workflow._initialize_state_db",
                    side_effect=PermissionError("database is not an eligible private runtime file"),
                ),
                self.assertRaisesRegex(PermissionError, "not an eligible private runtime file"),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(db_path),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(output_dir),
                    )
                )

            self.assertEqual(stale_report.read_text(encoding="utf-8"), "must remain unchanged\n")

    def test_empty_output_directory_is_rejected_without_using_the_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            working_directory = Path(tmp)
            stale_report = working_directory / "buy_signals.csv"
            stale_report.write_text("must remain unchanged\n", encoding="utf-8")
            db_path = working_directory / "state.sqlite"
            previous_working_directory = Path.cwd()
            os.chdir(working_directory)
            try:
                with self.assertRaisesRegex(ValueError, "output_dir must be a nonempty filesystem path"):
                    asyncio.run(
                        run_resumable_optimizations_async(
                            mode="update",
                            db_path=str(db_path),
                            base_cfg=BacktestConfig(),
                            universe_cfg=UniverseConfig(),
                            buy_rsi_values=[30.0],
                            profit_target_values=[1.5],
                            alpaca_cfg=AlpacaOrderConfig(),
                            output_dir="",
                        )
                    )
            finally:
                os.chdir(previous_working_directory)

            self.assertEqual(stale_report.read_text(encoding="utf-8"), "must remain unchanged\n")
            self.assertFalse(db_path.exists())
            self.assertFalse(Path(f"{db_path}.lock").exists())

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

    def test_workflow_rejects_database_collisions_with_managed_output_paths(self) -> None:
        for filename in (
            "best_equity_curves.csv",
            "managed_positions.csv",
            ".leveraged-trader.lock",
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp) / "reports"
                db_path = output_dir / filename

                with (
                    self.assertRaisesRegex(ValueError, "must use distinct paths"),
                    _workflow_run_lock(str(db_path), str(output_dir)),
                ):
                    self.fail("workflow acquired colliding database and output paths")

                self.assertFalse(output_dir.exists())
                self.assertFalse(Path(f"{db_path}.lock").exists())

    def test_workflow_rejects_database_sidecar_output_namespaces_before_creating_files(self) -> None:
        from leveraged_trader import workflow as workflow_module

        for suffix in ("-wal", "-shm", "-journal"):
            for nested_output in (False, True):
                with (
                    self.subTest(suffix=suffix, nested_output=nested_output),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    db_path = Path(tmp) / "state.sqlite"
                    sidecar_path = Path(f"{db_path}{suffix}")
                    output_dir = sidecar_path / "reports" if nested_output else sidecar_path
                    with (
                        patch.object(workflow_module, "prepare_owned_runtime_directory") as prepare_directory,
                        patch.object(workflow_module, "_prepare_workflow_user_lock_directory") as prepare_lock_dir,
                        self.assertRaisesRegex(ValueError, "SQLite sidecars"),
                        _workflow_run_lock(str(db_path), str(output_dir)),
                    ):
                        self.fail("workflow acquired an output namespace reserved for a SQLite sidecar")

                    prepare_directory.assert_not_called()
                    prepare_lock_dir.assert_not_called()
                    self.assertFalse(output_dir.exists())
                    self.assertFalse(db_path.exists())
                    self.assertFalse(Path(f"{db_path}.lock").exists())

    def test_workflow_rejects_alpaca_account_lock_database_alias_before_creation(self) -> None:
        from leveraged_trader import workflow as workflow_module

        for case_variant in (False, True):
            with self.subTest(case_variant=case_variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                account_home = root / "account-home"
                account_home.mkdir(mode=0o700)
                lock_root = workflow_module._workflow_user_lock_root(account_home)
                account_lock = lock_root / workflow_module._ALPACA_ACCOUNT_LOCK_FILENAME
                db_path = account_lock.with_name(account_lock.name.upper()) if case_variant else account_lock
                output_dir = root / "reports"

                with (
                    patch.object(
                        workflow_module,
                        "_workflow_user_lock_base",
                        return_value=(account_home, None),
                    ),
                    patch.object(
                        workflow_module,
                        "_windows_directory_namespace_is_case_insensitive",
                        return_value=True if case_variant else None,
                    ),
                    self.assertRaisesRegex(ValueError, "must use distinct paths"),
                    _workflow_run_lock(
                        str(db_path),
                        str(output_dir),
                        serialize_alpaca_account=True,
                    ),
                ):
                    self.fail("workflow acquired its Alpaca account lock as the database")

                self.assertFalse(lock_root.exists())
                self.assertFalse(output_dir.exists())
                self.assertFalse(db_path.exists())

    def test_workflow_rejects_output_anchor_database_before_creating_directories(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            output_dir = root / "reports"
            lock_root = workflow_module._workflow_user_lock_root(account_home)
            output_digest = workflow_module._workflow_lock_path_digest(output_dir)
            db_path = lock_root / f".leveraged-trader-output-{output_digest}.lock"

            with (
                patch.object(
                    workflow_module,
                    "_workflow_user_lock_base",
                    return_value=(account_home, None),
                ),
                self.assertRaisesRegex(ValueError, "must use distinct paths"),
                _workflow_run_lock(str(db_path), str(output_dir)),
            ):
                self.fail("workflow acquired an output-anchor lock as the database")

            self.assertFalse(lock_root.exists())
            self.assertFalse(output_dir.exists())
            self.assertFalse(db_path.exists())

    def test_workflow_rejects_database_containing_prospective_lock_root_without_artifacts(self) -> None:
        from leveraged_trader import workflow as workflow_module

        for database_location in ("lock-root", "missing-ancestor"):
            with self.subTest(database_location=database_location), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                account_home = root / "account-home"
                account_home.mkdir(mode=0o700)
                lock_root = workflow_module._workflow_user_lock_root(account_home)
                db_path = lock_root if database_location == "lock-root" else account_home / ".local" / "state"
                output_dir = root / "reports"

                with (
                    patch.object(
                        workflow_module,
                        "_workflow_user_lock_base",
                        return_value=(account_home, None),
                    ),
                    self.assertRaisesRegex(ValueError, "must not contain the per-user workflow lock directory"),
                    _workflow_run_lock(str(db_path), str(output_dir)),
                ):
                    self.fail("workflow consumed the configured database path while creating its lock directory")

                self.assertFalse((account_home / ".local").exists())
                self.assertFalse(lock_root.exists())
                self.assertFalse(output_dir.exists())
                self.assertFalse(db_path.exists())

    def test_workflow_rejects_output_containing_prospective_lock_root_without_artifacts(self) -> None:
        from leveraged_trader import workflow as workflow_module

        for output_location, case_insensitive in (
            ("lock-root", False),
            ("missing-ancestor", False),
            ("casefolded-missing-ancestor", True),
        ):
            with self.subTest(output_location=output_location), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                account_home = root / "account-home"
                account_home.mkdir(mode=0o700)
                lock_root = workflow_module._workflow_user_lock_root(account_home)
                if output_location == "lock-root":
                    output_dir = lock_root
                elif output_location == "missing-ancestor":
                    output_dir = account_home / ".local" / "state"
                else:
                    output_dir = account_home / ".LOCAL" / "STATE"
                db_path = root / "state.sqlite"

                with (
                    patch.object(
                        workflow_module,
                        "_workflow_user_lock_base",
                        return_value=(account_home, None),
                    ),
                    patch.object(
                        workflow_module,
                        "_windows_directory_namespace_is_case_insensitive",
                        return_value=True if case_insensitive else None,
                    ),
                    self.assertRaisesRegex(ValueError, "output_dir must not contain"),
                    _workflow_run_lock(str(db_path), str(output_dir)),
                ):
                    self.fail("workflow placed its stable locks inside the output namespace")

                self.assertFalse((account_home / ".local").exists())
                self.assertFalse((account_home / ".LOCAL").exists())
                self.assertFalse(lock_root.exists())
                self.assertFalse(output_dir.exists())
                self.assertFalse(db_path.exists())
                self.assertFalse(Path(f"{db_path}.lock").exists())

    def test_workflow_rejects_alpaca_account_lock_as_output_namespace_before_creation(self) -> None:
        from leveraged_trader import workflow as workflow_module

        for case_variant in (False, True):
            with self.subTest(case_variant=case_variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                account_home = root / "account-home"
                account_home.mkdir(mode=0o700)
                lock_root = workflow_module._workflow_user_lock_root(account_home)
                account_lock = lock_root / workflow_module._ALPACA_ACCOUNT_LOCK_FILENAME
                output_dir = account_lock.with_name(account_lock.name.upper()) if case_variant else account_lock
                db_path = root / "state.sqlite"

                with (
                    patch.object(
                        workflow_module,
                        "_workflow_user_lock_base",
                        return_value=(account_home, None),
                    ),
                    patch.object(
                        workflow_module,
                        "_windows_directory_namespace_is_case_insensitive",
                        return_value=True if case_variant else None,
                    ),
                    self.assertRaisesRegex(ValueError, "account lock.*output directory.*distinct namespaces"),
                    _workflow_run_lock(
                        str(db_path),
                        str(output_dir),
                        serialize_alpaca_account=True,
                    ),
                ):
                    self.fail("workflow created its output directory at the Alpaca account lock")

                self.assertFalse(lock_root.exists())
                self.assertFalse(output_dir.exists())
                self.assertFalse(db_path.exists())

    def test_workflow_accepts_preexisting_alpaca_account_lock_for_unrelated_paths(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            lock_root = workflow_module._workflow_user_lock_root(account_home)
            lock_root.mkdir(mode=0o700, parents=True)
            account_lock = lock_root / workflow_module._ALPACA_ACCOUNT_LOCK_FILENAME
            account_lock.write_text("existing account lock\n", encoding="utf-8")
            account_lock.chmod(0o600)
            db_path = root / "state.sqlite"
            output_dir = root / "reports"

            with (
                patch.object(
                    workflow_module,
                    "_workflow_user_lock_base",
                    return_value=(account_home, None),
                ),
                _workflow_run_lock(
                    str(db_path),
                    str(output_dir),
                    serialize_alpaca_account=True,
                ) as locked_output,
            ):
                self.assertEqual(Path(locked_output), output_dir)
                self.assertEqual(Path(locked_output.database_path), db_path)

            self.assertEqual(account_lock.read_text(encoding="utf-8"), "existing account lock\n")

    @unittest.skipUnless(os.name == "posix", "POSIX symbolic-link handling is required")
    def test_workflow_reserves_sidecars_of_the_supplied_database_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_database = Path(tmp) / "target.sqlite"
            target_database.touch(mode=0o600)
            database_alias = Path(tmp) / "alias.sqlite"
            database_alias.symlink_to(target_database)
            output_dir = Path(f"{database_alias}-wal")

            with (
                self.assertRaisesRegex(ValueError, "SQLite sidecars"),
                _workflow_run_lock(str(database_alias), str(output_dir)),
            ):
                self.fail("workflow allowed output to occupy a supplied database alias sidecar")

            self.assertFalse(output_dir.exists())
            self.assertFalse(Path(f"{database_alias}.lock").exists())
            self.assertFalse(Path(f"{target_database}.lock").exists())

    def test_workflow_rejects_existing_inode_alias_of_database_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "state.sqlite"
            sidecar_path = Path(f"{db_path}-wal")
            sidecar_path.write_text("sidecar namespace\n", encoding="utf-8")
            output_dir = root / "reports"
            output_dir.mkdir(mode=0o700)
            managed_report = output_dir / "buy_signals.csv"
            os.link(sidecar_path, managed_report)

            with (
                self.assertRaisesRegex(ValueError, "must use distinct paths"),
                _workflow_run_lock(str(db_path), str(output_dir)),
            ):
                self.fail("workflow allowed an output file to alias a SQLite sidecar inode")

            self.assertEqual(managed_report.read_text(encoding="utf-8"), "sidecar namespace\n")
            self.assertFalse((output_dir / ".leveraged-trader.lock").exists())
            self.assertFalse(Path(f"{db_path}.lock").exists())

    def test_workflow_sidecar_paths_participate_in_case_variant_collision_checks(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe_path = root / "case-probe"
            probe_path.touch()
            with (
                patch.object(
                    workflow_module,
                    "_workflow_namespace_is_case_insensitive",
                    return_value=True,
                ),
                self.assertRaisesRegex(ValueError, "must use distinct paths"),
            ):
                workflow_module._validate_workflow_runtime_path_separation(
                    database_path=root / "state.sqlite",
                    output_path=root / "STATE.SQLITE-WAL",
                    database_lock_paths=set(),
                    case_sensitivity_probe_paths={probe_path},
                )

    def test_workflow_rejects_casefolded_sidecar_output_before_creating_runtime_artifacts(self) -> None:
        from leveraged_trader import workflow as workflow_module

        for suffix in ("-wal", "-shm", "-journal"):
            for nested_output in (False, True):
                for database_exists in (False, True):
                    with (
                        self.subTest(
                            suffix=suffix,
                            nested_output=nested_output,
                            database_exists=database_exists,
                        ),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        root = Path(tmp)
                        db_path = root / "state.sqlite"
                        if database_exists:
                            db_path.touch(mode=0o600)
                        casefolded_sidecar = root / f"{db_path.name}{suffix}".upper()
                        output_dir = casefolded_sidecar / "reports" if nested_output else casefolded_sidecar

                        with (
                            patch.object(
                                workflow_module,
                                "_workflow_namespace_is_case_insensitive",
                                return_value=True,
                            ),
                            patch.object(workflow_module, "prepare_owned_runtime_directory") as prepare_directory,
                            patch.object(
                                workflow_module,
                                "_prepare_workflow_user_lock_directory",
                            ) as prepare_lock_directory,
                            self.assertRaisesRegex(ValueError, "SQLite sidecars.*distinct paths"),
                            _workflow_run_lock(str(db_path), str(output_dir)),
                        ):
                            self.fail("workflow acquired a case-folded SQLite sidecar output namespace")

                        prepare_directory.assert_not_called()
                        prepare_lock_directory.assert_not_called()
                        self.assertFalse(output_dir.exists())
                        self.assertFalse(Path(f"{db_path}.lock").exists())

    def test_casefold_collision_uses_target_windows_directory_flag_not_its_parent(self) -> None:
        from leveraged_trader import workflow as workflow_module

        cases = (
            # A case-sensitive target directory may itself live in a default
            # case-insensitive parent namespace.
            (False, True, False),
            # An explicitly case-insensitive target can live below a
            # case-sensitive parent after its inherited flag is disabled.
            (True, False, True),
        )
        for target_is_case_insensitive, parent_is_case_insensitive, should_reject in cases:
            with (
                self.subTest(
                    target_is_case_insensitive=target_is_case_insensitive,
                    parent_is_case_insensitive=parent_is_case_insensitive,
                ),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                db_path = root / "state.sqlite"
                output_dir = root / "STATE.SQLITE-WAL"
                with (
                    patch.object(
                        workflow_module,
                        "_windows_directory_namespace_is_case_insensitive",
                        return_value=target_is_case_insensitive,
                    ) as windows_query,
                    patch.object(
                        workflow_module,
                        "_workflow_namespace_is_case_insensitive",
                        return_value=parent_is_case_insensitive,
                    ) as parent_probe,
                ):
                    if should_reject:
                        with self.assertRaisesRegex(ValueError, "SQLite sidecars.*distinct paths"):
                            workflow_module._validate_workflow_runtime_path_separation(
                                database_path=db_path,
                                output_path=output_dir,
                                database_lock_paths=set(),
                            )
                    else:
                        workflow_module._validate_workflow_runtime_path_separation(
                            database_path=db_path,
                            output_path=output_dir,
                            database_lock_paths=set(),
                        )

                windows_query.assert_called_with(root)
                parent_probe.assert_not_called()

    def test_casefold_collision_skips_letterless_direct_probe(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "123"
            db_path.touch(mode=0o600)
            output_dir = root / "123-WAL"
            observed_probes: list[Path] = []

            def case_insensitive_parent_probe(probe_path: Path) -> bool:
                observed_probes.append(probe_path)
                return probe_path == root

            with (
                patch.object(
                    workflow_module,
                    "_windows_directory_namespace_is_case_insensitive",
                    return_value=None,
                ),
                patch.object(
                    workflow_module,
                    "_workflow_namespace_is_case_insensitive",
                    side_effect=case_insensitive_parent_probe,
                ),
                self.assertRaisesRegex(ValueError, "SQLite sidecars.*distinct paths"),
            ):
                workflow_module._validate_workflow_runtime_path_separation(
                    database_path=db_path,
                    output_path=output_dir,
                    database_lock_paths=set(),
                )

            self.assertEqual(observed_probes, [root])

    def test_windows_casefold_collision_queries_nearest_existing_ancestor(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_namespace = root / "missing" / "namespace"
            db_path = missing_namespace / "state.sqlite"
            output_dir = missing_namespace / "STATE.SQLITE-WAL"

            with (
                patch.object(
                    workflow_module,
                    "_windows_directory_namespace_is_case_insensitive",
                    return_value=True,
                ) as windows_query,
                patch.object(
                    workflow_module,
                    "_workflow_namespace_is_case_insensitive",
                    side_effect=AssertionError("Windows must use the directory information query"),
                ),
                self.assertRaisesRegex(ValueError, "SQLite sidecars.*distinct paths"),
            ):
                workflow_module._validate_workflow_runtime_path_separation(
                    database_path=db_path,
                    output_path=output_dir,
                    database_lock_paths=set(),
                )

            windows_query.assert_called_with(root)

    def test_workflow_rejects_nonexistent_case_variant_report_database_before_creation(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reports"
            db_path = output_dir / "BEST_EQUITY_CURVES.CSV"

            with (
                patch.object(
                    workflow_module,
                    "_workflow_namespace_is_case_insensitive",
                    return_value=True,
                ),
                patch.object(workflow_module, "prepare_private_runtime_file") as prepare_database,
                self.assertRaisesRegex(ValueError, "must use distinct paths"),
                _workflow_run_lock(str(db_path), str(output_dir)),
            ):
                self.fail("workflow acquired case-variant database and output paths")

            prepare_database.assert_not_called()
            self.assertFalse(db_path.exists())

    def test_workflow_rejects_case_variant_output_lock_database_before_locking(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reports"
            db_path = output_dir / ".LEVERAGED-TRADER"

            def case_insensitive_namespace(probe_path: Path) -> bool:
                self.assertTrue(probe_path.exists())
                return True

            with (
                patch.object(
                    workflow_module,
                    "_workflow_namespace_is_case_insensitive",
                    side_effect=case_insensitive_namespace,
                ),
                patch.object(workflow_module, "_lock_file_nonblocking") as acquire_lock,
                patch.object(workflow_module, "prepare_private_runtime_file") as prepare_database,
                self.assertRaisesRegex(ValueError, "must use distinct paths"),
                _workflow_run_lock(str(db_path), str(output_dir)),
            ):
                self.fail("workflow acquired case-variant database and output lock paths")

            acquire_lock.assert_not_called()
            prepare_database.assert_not_called()
            self.assertFalse(db_path.exists())

    def test_case_sensitive_output_namespace_allows_distinct_case_variant_leaf(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reports"
            output_dir.mkdir()
            probe_path = output_dir / ".leveraged-trader.lock"
            probe_path.touch()

            with patch.object(
                workflow_module,
                "_workflow_namespace_is_case_insensitive",
                return_value=False,
            ):
                workflow_module._validate_workflow_runtime_path_separation(
                    database_path=output_dir / "BEST_EQUITY_CURVES.CSV",
                    output_path=output_dir,
                    database_lock_paths=set(),
                    case_sensitivity_probe_paths={probe_path},
                )

    def test_workflow_rejects_database_ancestor_of_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "runtime"
            output_dir = db_path / "reports"

            with (
                self.assertRaisesRegex(ValueError, "output directory or one of its ancestors"),
                _workflow_run_lock(str(db_path), str(output_dir)),
            ):
                self.fail("workflow acquired a database path containing its output directory")

            self.assertFalse(db_path.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX symbolic-link handling is required")
    def test_async_workflow_keeps_using_database_target_pinned_by_run_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first_database = Path(tmp) / "first.sqlite"
            second_database = Path(tmp) / "second.sqlite"
            first_database.touch(mode=0o600)
            second_database.touch(mode=0o600)
            database_alias = Path(tmp) / "state.sqlite"
            database_alias.symlink_to(first_database)
            observed: dict[str, object] = {}

            async def observe_locked_paths(**kwargs: object) -> None:
                database_alias.unlink()
                database_alias.symlink_to(second_database)
                observed.update(kwargs)

            with patch(
                "leveraged_trader.workflow._run_resumable_optimizations_unlocked",
                new=AsyncMock(side_effect=observe_locked_paths),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(database_alias),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(Path(tmp) / "reports"),
                    )
                )

            self.assertEqual(observed["db_path"], str(first_database.resolve()))
            self.assertEqual(
                observed["universe_cfg"].sqlite_db_path,
                str(first_database.resolve()),
            )

    @unittest.skipUnless(os.name == "posix", "POSIX inode identity is required")
    def test_moved_database_inode_remains_locked_and_transaction_is_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "state.sqlite"
            moved_database_path = Path(tmp) / "moved.sqlite"
            with closing(sqlite3.connect(database_path)) as conn, conn:
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")

            with (
                self.assertRaisesRegex(OSError, "disappeared"),
                _workflow_run_lock(
                    str(database_path),
                    str(Path(tmp) / "outputs-one"),
                ) as locked_output_dir,
                _state_connection(
                    locked_output_dir.database_path,
                    immediate=True,
                ) as conn,
            ):
                conn.execute("INSERT INTO transaction_probe VALUES (1)")
                database_path.rename(moved_database_path)
                with (
                    self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                    _workflow_run_lock(
                        str(moved_database_path),
                        str(Path(tmp) / "outputs-two"),
                    ),
                ):
                    self.fail("moved database inode acquired a second workflow lock")

            with closing(sqlite3.connect(moved_database_path)) as conn, conn:
                rows = conn.execute("SELECT value FROM transaction_probe").fetchall()
            self.assertEqual(rows, [])

    @unittest.skipUnless(os.name == "posix", "POSIX inode identity is required")
    def test_shared_executor_rejects_replacement_database_before_opening_it(self) -> None:
        async def attempt_connection(db_path: str) -> None:
            def write_replacement() -> None:
                with _state_connection(db_path, immediate=True) as conn:
                    conn.execute("INSERT INTO transaction_probe VALUES (1)")

            with ThreadPoolExecutor(max_workers=1) as executor:
                await _run_blocking(executor, write_replacement)

        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "state.sqlite"
            moved_database_path = Path(tmp) / "moved.sqlite"
            with closing(sqlite3.connect(database_path)) as conn, conn:
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")

            with (
                self.assertRaisesRegex(OSError, "changed identity"),
                _workflow_run_lock(
                    str(database_path),
                    str(Path(tmp) / "outputs"),
                ) as locked_output_dir,
            ):
                database_path.rename(moved_database_path)
                with closing(sqlite3.connect(database_path)) as conn, conn:
                    conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")
                asyncio.run(attempt_connection(locked_output_dir.database_path))

            with closing(sqlite3.connect(database_path)) as conn, conn:
                replacement_rows = conn.execute("SELECT value FROM transaction_probe").fetchall()
            with closing(sqlite3.connect(moved_database_path)) as conn, conn:
                pinned_rows = conn.execute("SELECT value FROM transaction_probe").fetchall()
            self.assertEqual(replacement_rows, [])
            self.assertEqual(pinned_rows, [])

    @unittest.skipUnless(os.name == "posix", "POSIX account locks are required")
    def test_database_parent_replacement_cannot_bypass_logical_path_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            database_parent = root / "database"
            database_parent.mkdir(mode=0o700)
            displaced_parent = root / "displaced-database"
            database_path = database_parent / "state.sqlite"

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                self.assertRaisesRegex(PermissionError, "changed while runtime files were in use"),
                _workflow_run_lock(
                    str(database_path),
                    str(root / "outputs-one"),
                ),
            ):
                database_parent.rename(displaced_parent)
                database_parent.mkdir(mode=0o700)
                with (
                    self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                    _workflow_run_lock(
                        str(database_path),
                        str(root / "outputs-two"),
                    ),
                ):
                    self.fail("replacement database parent bypassed the account-scoped path anchor")

    @unittest.skipUnless(os.name == "posix", "POSIX account locks are required")
    def test_output_parent_replacement_cannot_bypass_logical_path_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            output_parent = root / "runtime"
            output_parent.mkdir(mode=0o700)
            output_dir = output_parent / "reports"
            output_dir.mkdir(mode=0o700)
            displaced_parent = root / "displaced-runtime"

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                self.assertRaisesRegex(PermissionError, "changed while runtime files were in use"),
                _workflow_run_lock(
                    str(root / "state-one.sqlite"),
                    str(output_dir),
                ),
            ):
                output_parent.rename(displaced_parent)
                output_parent.mkdir(mode=0o700)
                output_dir.mkdir(mode=0o700)
                with (
                    self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                    _workflow_run_lock(
                        str(root / "state-two.sqlite"),
                        str(output_dir),
                    ),
                ):
                    self.fail("replacement output parent bypassed the account-scoped path anchor")

    @unittest.skipUnless(os.name == "posix", "POSIX account locks are required")
    def test_output_below_lock_root_remains_serialized_after_replacement(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            lock_root = workflow_module._workflow_user_lock_root(account_home)
            output_dir = lock_root / "outputs"
            displaced_output = lock_root / "displaced-outputs"

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                self.assertRaisesRegex(PermissionError, "changed while runtime files were in use"),
                _workflow_run_lock(
                    str(root / "state-one.sqlite"),
                    str(output_dir),
                ),
            ):
                output_dir.rename(displaced_output)
                output_dir.mkdir(mode=0o700)
                with (
                    self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                    _workflow_run_lock(
                        str(root / "state-two.sqlite"),
                        str(output_dir),
                    ),
                ):
                    self.fail("replacement output bypassed its stable logical-path anchor")

    @unittest.skipUnless(os.name == "posix", "POSIX file modes are required")
    def test_workflow_creates_private_output_and_lock_with_permissive_umask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_parent = Path(tmp) / "new-parent"
            output_dir = output_parent / "reports"
            old_umask = os.umask(0)
            try:
                with _workflow_run_lock(str(Path(tmp) / "state.sqlite"), str(output_dir)):
                    self.assertEqual(stat.S_IMODE(output_parent.stat().st_mode), 0o700)
                    self.assertEqual(stat.S_IMODE(output_dir.stat().st_mode), 0o700)
                    self.assertEqual(
                        stat.S_IMODE((output_dir / ".leveraged-trader.lock").stat().st_mode),
                        0o600,
                    )
            finally:
                os.umask(old_umask)

    @unittest.skipUnless(os.name == "posix", "POSIX file modes and flock are required")
    def test_workflow_rejects_readable_existing_lock_before_acquiring_it(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "workflow.lock"
            lock_path.touch(mode=0o644)
            lock_path.chmod(0o644)
            with lock_path.open("r", encoding="utf-8") as preopened_lock:
                assert workflow_module.fcntl is not None
                workflow_module.fcntl.flock(preopened_lock.fileno(), workflow_module.fcntl.LOCK_EX)
                with self.assertRaisesRegex(PermissionError, "must not be accessible by group or other"):
                    workflow_module._open_workflow_lock_file(lock_path)

            self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o644)

    @unittest.skipUnless(os.name == "posix", "POSIX file modes are required")
    def test_workflow_anchor_does_not_require_a_writable_output_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = root / "runtime"
            runtime_dir.mkdir(mode=0o700)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            output_parent = root / "read-only-parent"
            output_parent.mkdir(mode=0o700)
            output_dir = output_parent / "reports"
            output_dir.mkdir(mode=0o700)
            db_path = root / "database" / "state.sqlite"
            db_path.parent.mkdir(mode=0o700)
            output_parent.chmod(0o500)
            digest = hashlib.sha256(os.fsencode(output_dir.resolve())).hexdigest()[:32]
            anchor_name = f".leveraged-trader-output-{digest}.lock"
            try:
                with (
                    patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(runtime_dir)}),
                    patch(
                        "leveraged_trader.workflow.pwd.getpwuid",
                        return_value=Mock(pw_dir=str(account_home)),
                    ),
                    _workflow_run_lock(str(db_path), str(output_dir)),
                ):
                    self.assertTrue((output_dir / ".leveraged-trader.lock").is_file())
                    self.assertTrue(
                        (
                            account_home / ".local" / "state" / "leveraged-trader" / "workflow-locks" / anchor_name
                        ).is_file()
                    )
                    self.assertFalse((output_parent / anchor_name).exists())
            finally:
                output_parent.chmod(0o700)

    @unittest.skipUnless(os.name == "posix", "POSIX account locks are required")
    def test_account_anchors_cover_supplied_and_canonical_database_and_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            canonical_root = root / "canonical"
            canonical_root.mkdir(mode=0o700)
            canonical_parent = canonical_root / "runtime"
            canonical_parent.mkdir(mode=0o700)
            supplied_root = root / "supplied"
            supplied_root.symlink_to(canonical_root, target_is_directory=True)
            supplied_parent = supplied_root / "runtime"
            database_path = supplied_parent / "state.sqlite"
            output_dir = supplied_parent / "reports"
            output_dir.mkdir(mode=0o700)
            lock_directory = account_home / ".local" / "state" / "leveraged-trader" / "workflow-locks"

            def anchor_name(prefix: str, path: str) -> str:
                normalized = os.path.normcase(os.path.abspath(path))
                digest = hashlib.sha256(os.fsencode(normalized)).hexdigest()[:32]
                return f".leveraged-trader-{prefix}-{digest}.lock"

            expected_anchors = {
                anchor_name("database-path", str(database_path)),
                anchor_name("database-path", str((canonical_parent / "state.sqlite").resolve())),
                anchor_name("output", str(output_dir)),
                anchor_name("output", str((canonical_parent / "reports").resolve())),
            }
            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                _workflow_run_lock(str(database_path), str(output_dir)),
            ):
                self.assertEqual(
                    {path.name for path in lock_directory.iterdir()} & expected_anchors,
                    expected_anchors,
                )

    @unittest.skipUnless(os.name == "posix", "POSIX symbolic-link handling is required")
    def test_workflow_resolves_symlink_before_parent_component_for_runtime_paths(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            semantic_parent = root / "semantic-parent"
            symlink_target = semantic_parent / "inner"
            symlink_target.mkdir(parents=True, mode=0o700)
            redirect = root / "redirect"
            redirect.symlink_to(symlink_target, target_is_directory=True)
            supplied_database = os.path.join(
                os.fspath(redirect),
                os.pardir,
                "state.sqlite",
            )
            supplied_output = os.path.join(
                os.fspath(redirect),
                os.pardir,
                "reports",
            )

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                _workflow_run_lock(supplied_database, supplied_output) as locked_output,
            ):
                self.assertEqual(
                    locked_output.database_path,
                    os.fspath(semantic_parent / "state.sqlite"),
                )
                self.assertEqual(
                    os.fspath(locked_output),
                    os.fspath(semantic_parent / "reports"),
                )
                workflow_module._prepare_workflow_output_directory_for_publication(Path(supplied_output))
                _atomic_to_csv(
                    pd.DataFrame({"value": [1]}),
                    Path(supplied_output) / "probe.csv",
                    index=False,
                )
                stale_report = semantic_parent / "reports" / _RESEARCH_REPORT_FILENAMES[0]
                stale_report.write_text("stale", encoding="utf-8")
                _clear_stale_workflow_research_outputs(supplied_output)

                self.assertTrue((semantic_parent / "reports" / "probe.csv").is_file())
                self.assertFalse(stale_report.exists())

            self.assertFalse((root / "state.sqlite").exists())
            self.assertFalse((root / "reports").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX account locks are required")
    def test_workflow_rejects_lock_path_shared_across_acquisition_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            lock_directory = account_home / ".local" / "state" / "leveraged-trader" / "workflow-locks"
            lock_directory.mkdir(mode=0o700, parents=True)
            target_database = root / "target.sqlite"
            target_database.touch(mode=0o600)
            target_digest = hashlib.sha256(os.fsencode(os.path.normcase(os.path.abspath(target_database)))).hexdigest()[
                :32
            ]
            database_alias = lock_directory / (f".leveraged-trader-database-path-{target_digest}")
            database_alias.symlink_to(target_database)
            colliding_lock_path = Path(f"{database_alias}.lock")
            output_dir = root / "outputs"

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                self.assertRaisesRegex(ValueError, "distinct across acquisition tiers"),
                _workflow_run_lock(str(database_alias), str(output_dir)),
            ):
                self.fail("workflow acquired one lock path in two tiers")

            self.assertFalse(colliding_lock_path.exists())
            self.assertFalse(Path(f"{target_database}.lock").exists())
            self.assertFalse((output_dir / ".leveraged-trader.lock").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX account locks are required")
    def test_retargeted_database_rejects_dynamically_discovered_lock_tier_collision(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            lock_directory = account_home / ".local" / "state" / "leveraged-trader" / "workflow-locks"
            lock_directory.mkdir(mode=0o700, parents=True)
            first_database = root / "first.sqlite"
            first_database.touch(mode=0o600)
            target_database = root / "target.sqlite"
            target_database.touch(mode=0o600)
            target_digest = hashlib.sha256(os.fsencode(os.path.normcase(os.path.abspath(target_database)))).hexdigest()[
                :32
            ]
            database_alias = lock_directory / (f".leveraged-trader-database-path-{target_digest}")
            database_alias.symlink_to(first_database)
            real_prepare = workflow_module.prepare_private_runtime_file
            retargeted = False

            def retarget_before_database_is_pinned(path: str) -> object:
                nonlocal retargeted
                if Path(os.path.abspath(path)) == database_alias and not retargeted:
                    database_alias.unlink()
                    database_alias.symlink_to(target_database)
                    retargeted = True
                return real_prepare(path)

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                patch.object(
                    workflow_module,
                    "prepare_private_runtime_file",
                    side_effect=retarget_before_database_is_pinned,
                ),
            ):
                with (
                    self.assertRaisesRegex(ValueError, "distinct across acquisition tiers"),
                    _workflow_run_lock(
                        str(database_alias),
                        str(root / "outputs-one"),
                    ),
                ):
                    self.fail("workflow reacquired a dynamically shared lock path")

                self.assertTrue(retargeted)
                with _workflow_run_lock(
                    str(target_database),
                    str(root / "outputs-two"),
                ):
                    pass

    @unittest.skipUnless(os.name == "posix", "POSIX account locks are required")
    def test_retargeted_database_acquires_actual_target_account_anchor(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            first_database = root / "first.sqlite"
            first_database.touch(mode=0o600)
            target_parent = root / "target"
            target_parent.mkdir(mode=0o700)
            target_database = target_parent / "state.sqlite"
            target_database.touch(mode=0o600)
            database_alias = root / "database-alias.sqlite"
            database_alias.symlink_to(first_database)
            displaced_target_parent = root / "displaced-target"
            real_prepare = workflow_module.prepare_private_runtime_file
            retargeted = False
            second_workflow_was_blocked = False

            def retarget_before_database_is_pinned(path: str) -> object:
                nonlocal retargeted
                if Path(os.path.abspath(path)) == database_alias and not retargeted:
                    database_alias.unlink()
                    database_alias.symlink_to(target_database)
                    retargeted = True
                return real_prepare(path)

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                patch.object(
                    workflow_module,
                    "prepare_private_runtime_file",
                    side_effect=retarget_before_database_is_pinned,
                ),
                self.assertRaisesRegex(PermissionError, "changed while runtime files were in use"),
                _workflow_run_lock(
                    str(database_alias),
                    str(root / "outputs-one"),
                ) as locked_output_dir,
            ):
                self.assertTrue(retargeted)
                self.assertEqual(locked_output_dir.database_path, str(target_database))
                target_parent.rename(displaced_target_parent)
                target_parent.mkdir(mode=0o700)
                target_database.touch(mode=0o600)
                with (
                    self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                    _workflow_run_lock(
                        str(target_database),
                        str(root / "outputs-two"),
                    ),
                ):
                    self.fail("replacement database bypassed the actual-target account anchor")
                second_workflow_was_blocked = True

            self.assertTrue(second_workflow_was_blocked)

    @unittest.skipUnless(os.name == "posix", "POSIX account locks are required")
    def test_retargeted_and_direct_database_callers_share_lock_tier_order(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "z-account-home"
            account_home.mkdir(mode=0o700)
            first_database = root / "first.sqlite"
            first_database.touch(mode=0o600)
            target_parent = root / "a-target"
            target_parent.mkdir(mode=0o700)
            target_database = target_parent / "state.sqlite"
            target_database.touch(mode=0o600)
            database_alias = root / "database-alias.sqlite"
            database_alias.symlink_to(first_database)

            lock_directory = account_home / ".local" / "state" / "leveraged-trader" / "workflow-locks"
            target_digest = hashlib.sha256(os.fsencode(os.path.normcase(os.path.abspath(target_database)))).hexdigest()[
                :32
            ]
            target_anchor_lock = lock_directory / (f".leveraged-trader-database-path-{target_digest}.lock")
            target_local_lock = Path(f"{target_database}.lock")

            real_prepare = workflow_module.prepare_private_runtime_file
            real_open_lock = workflow_module._open_workflow_lock_file
            real_lock_nonblocking = workflow_module._lock_file_nonblocking
            opened_lock_paths: dict[int, Path] = {}
            direct_first_shared_lock = threading.Event()
            direct_anchor_held = threading.Event()
            direct_local_held = threading.Event()
            alias_anchor_held = threading.Event()
            release_workflow = threading.Event()
            acquired: list[str] = []
            failures: list[tuple[str, BaseException]] = []
            retargeted = False

            def retarget_alias_before_pin(path: str) -> object:
                nonlocal retargeted
                if Path(os.path.abspath(path)) == database_alias and not retargeted:
                    database_alias.unlink()
                    database_alias.symlink_to(target_database)
                    retargeted = True
                return real_prepare(path)

            def open_recorded_lock(lock_path: Path) -> object:
                lock_file = real_open_lock(lock_path)
                opened_lock_paths[id(lock_file)] = lock_path
                return lock_file

            def coordinated_lock(lock_file: object) -> None:
                lock_path = opened_lock_paths[id(lock_file)]
                role = threading.current_thread().name
                if (
                    role == "direct-database"
                    and lock_path == target_anchor_lock
                    and direct_local_held.is_set()
                    and alias_anchor_held.is_set()
                ):
                    raise BlockingIOError
                if (
                    role == "alias-database"
                    and lock_path == target_local_lock
                    and direct_local_held.is_set()
                    and alias_anchor_held.is_set()
                ):
                    raise BlockingIOError

                real_lock_nonblocking(lock_file)
                if role == "direct-database" and lock_path == target_anchor_lock:
                    direct_anchor_held.set()
                    direct_first_shared_lock.set()
                elif role == "direct-database" and lock_path == target_local_lock:
                    direct_local_held.set()
                    direct_first_shared_lock.set()
                    if not direct_anchor_held.is_set():
                        alias_anchor_held.wait(timeout=5)
                elif role == "alias-database" and lock_path == target_anchor_lock:
                    alias_anchor_held.set()

            def run_workflow(role: str, database_path: Path, output_name: str) -> None:
                try:
                    with _workflow_run_lock(
                        str(database_path),
                        str(root / output_name),
                    ):
                        acquired.append(role)
                        release_workflow.wait(timeout=5)
                except BaseException as exc:
                    failures.append((role, exc))

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                patch.object(
                    workflow_module,
                    "prepare_private_runtime_file",
                    side_effect=retarget_alias_before_pin,
                ),
                patch.object(
                    workflow_module,
                    "_open_workflow_lock_file",
                    side_effect=open_recorded_lock,
                ),
                patch.object(
                    workflow_module,
                    "_lock_file_nonblocking",
                    side_effect=coordinated_lock,
                ),
            ):
                direct_thread = threading.Thread(
                    target=run_workflow,
                    args=("direct", target_database, "outputs-direct"),
                    name="direct-database",
                )
                alias_thread = threading.Thread(
                    target=run_workflow,
                    args=("alias", database_alias, "outputs-alias"),
                    name="alias-database",
                )
                direct_thread.start()
                self.assertTrue(direct_first_shared_lock.wait(timeout=5))
                alias_thread.start()
                alias_thread.join(timeout=5)
                release_workflow.set()
                direct_thread.join(timeout=5)

                self.assertFalse(alias_thread.is_alive())
                self.assertFalse(direct_thread.is_alive())
                self.assertTrue(retargeted)
                self.assertEqual(acquired, ["direct"])
                self.assertEqual([role for role, _exc in failures], ["alias"])
                self.assertIsInstance(failures[0][1], WorkflowRunError)

                with _workflow_run_lock(
                    str(target_database),
                    str(root / "outputs-retry"),
                ):
                    pass

    @unittest.skipUnless(os.name == "posix", "POSIX account metadata is required")
    def test_workflow_user_lock_base_ignores_xdg_runtime_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            working_directory = Path(tmp) / "working"
            working_directory.mkdir(mode=0o700)
            absolute_runtime = Path(tmp) / "absolute-runtime"
            absolute_runtime.mkdir(mode=0o700)
            account_home = Path(tmp) / "account-home"
            account_home.mkdir(mode=0o700)
            previous_working_directory = Path.cwd()
            os.chdir(working_directory)
            try:
                for runtime_dir in (str(absolute_runtime), "relative-runtime"):
                    with (
                        self.subTest(runtime_dir=runtime_dir),
                        patch.dict(os.environ, {"XDG_RUNTIME_DIR": runtime_dir}),
                        patch(
                            "leveraged_trader.workflow.pwd.getpwuid",
                            return_value=Mock(pw_dir=str(account_home)),
                        ),
                    ):
                        lock_base, _guard = _workflow_user_lock_base()
                    self.assertEqual(lock_base, account_home.resolve())
            finally:
                os.chdir(previous_working_directory)

            self.assertNotEqual(account_home.resolve(), absolute_runtime.resolve())

    @unittest.skipUnless(os.name == "posix", "POSIX account metadata is required")
    def test_workflow_user_lock_base_normalizes_missing_or_invalid_account_home(self) -> None:
        expected_message = "Cannot find a private per-user directory for workflow locks"
        with patch.dict(os.environ, {}, clear=True):
            with (
                self.subTest(account_home="missing passwd record"),
                patch("leveraged_trader.workflow.pwd.getpwuid", side_effect=KeyError("missing")),
                self.assertRaisesRegex(PermissionError, expected_message),
            ):
                _workflow_user_lock_base()

            for account_home in ("", "relative-home"):
                with (
                    self.subTest(account_home=account_home),
                    patch(
                        "leveraged_trader.workflow.pwd.getpwuid",
                        return_value=Mock(pw_dir=account_home),
                    ),
                    self.assertRaisesRegex(PermissionError, expected_message),
                ):
                    _workflow_user_lock_base()

    def test_windows_workflow_user_lock_base_uses_known_folder_not_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stable_local_app_data = Path(tmp).resolve()
            observed_bases = []
            for environment in (
                {"USERPROFILE": r"C:\Users\alice", "HOME": r"C:\Users\alice"},
                {"USERPROFILE": r"D:\service-profile", "HOME": r"D:\service-profile"},
            ):
                with (
                    self.subTest(environment=environment),
                    patch.dict(os.environ, environment, clear=True),
                    patch("leveraged_trader.workflow.os.name", "nt"),
                    patch(
                        "leveraged_trader.workflow._windows_local_app_data_path",
                        return_value=stable_local_app_data,
                    ) as known_folder,
                    patch.object(Path, "home", side_effect=AssertionError("Path.home must not select lock roots")),
                ):
                    lock_base, guard = _workflow_user_lock_base()
                known_folder.assert_called_once_with()
                observed_bases.append(lock_base)
                self.assertIsNone(guard)

            self.assertEqual(observed_bases, [stable_local_app_data, stable_local_app_data])

    def test_windows_workflow_user_lock_base_fails_closed_without_known_folder(self) -> None:
        with (
            patch("leveraged_trader.workflow.os.name", "nt"),
            patch(
                "leveraged_trader.workflow._windows_local_app_data_path",
                side_effect=OSError("missing profile metadata"),
            ),
            self.assertRaisesRegex(PermissionError, "stable per-user directory"),
        ):
            _workflow_user_lock_base()

    def test_windows_directory_case_sensitivity_query_uses_target_directory_handle(self) -> None:
        from leveraged_trader import workflow as workflow_module

        directory_path = Path(r"C:\runtime")
        for flags, expected_case_insensitive in ((0, True), (1, False)):
            create_file = Mock(return_value=0x1234)
            get_file_information = Mock()

            def populate_case_sensitive_information(
                _handle: object,
                _information_class: object,
                information_pointer: object,
                _buffer_size: object,
                *,
                flags: int = flags,
            ) -> int:
                information_pointer._obj.flags = flags
                return 1

            get_file_information.side_effect = populate_case_sensitive_information
            close_handle = Mock(return_value=1)
            kernel32 = Mock(
                CreateFileW=create_file,
                GetFileInformationByHandleEx=get_file_information,
                CloseHandle=close_handle,
            )

            with (
                self.subTest(flags=flags),
                patch.object(workflow_module.os, "name", "nt"),
                patch.object(workflow_module.ctypes, "WinDLL", create=True, return_value=kernel32),
            ):
                actual = workflow_module._windows_directory_namespace_is_case_insensitive(directory_path)

            self.assertIs(actual, expected_case_insensitive)
            create_file.assert_called_once_with(
                os.fspath(directory_path),
                0x0080,
                0x0001 | 0x0002 | 0x0004,
                None,
                3,
                0x02000000,
                None,
            )
            self.assertEqual(get_file_information.call_args.args[0], 0x1234)
            self.assertEqual(get_file_information.call_args.args[1], 23)
            close_handle.assert_called_once_with(0x1234)

    def test_windows_known_folder_balances_successful_com_initialization(self) -> None:
        from leveraged_trader import workflow as workflow_module

        returned_path = r"C:\Users\alice\AppData\Local"
        expected_path = Mock(name="resolved_local_app_data")
        for initialization_result in (0, 1):
            get_known_folder_path = Mock(name="SHGetKnownFolderPath")

            def return_known_folder(
                _folder_id: object,
                _flags: object,
                _token: object,
                path_pointer: object,
            ) -> int:
                path_pointer._obj.value = 0x1234
                return 0

            get_known_folder_path.side_effect = return_known_folder
            shell32 = Mock(SHGetKnownFolderPath=get_known_folder_path)
            initialize_com = Mock(return_value=initialization_result)
            uninitialize_com = Mock()
            free_task_memory = Mock()
            ole32 = Mock(
                CoInitializeEx=initialize_com,
                CoUninitialize=uninitialize_com,
                CoTaskMemFree=free_task_memory,
            )
            path_candidate = Mock()
            path_candidate.is_absolute.return_value = True
            path_candidate.resolve.return_value = expected_path

            with (
                self.subTest(initialization_result=initialization_result),
                patch.object(workflow_module.os, "name", "nt"),
                patch.object(workflow_module.ctypes, "WinDLL", create=True, side_effect=(shell32, ole32)),
                patch.object(workflow_module.ctypes, "wstring_at", return_value=returned_path),
                patch.object(workflow_module, "Path", return_value=path_candidate) as path_constructor,
            ):
                actual_path = _windows_local_app_data_path()

            self.assertIs(actual_path, expected_path)
            initialize_com.assert_called_once_with(None, 0)
            get_known_folder_path.assert_called_once()
            free_task_memory.assert_called_once()
            uninitialize_com.assert_called_once_with()
            path_constructor.assert_called_once_with(returned_path)

    def test_windows_known_folder_uses_existing_incompatible_com_apartment(self) -> None:
        from leveraged_trader import workflow as workflow_module

        get_known_folder_path = Mock(name="SHGetKnownFolderPath")

        def return_known_folder(
            _folder_id: object,
            _flags: object,
            _token: object,
            path_pointer: object,
        ) -> int:
            path_pointer._obj.value = 0x1234
            return 0

        get_known_folder_path.side_effect = return_known_folder
        shell32 = Mock(SHGetKnownFolderPath=get_known_folder_path)
        changed_mode = ctypes.c_int32(0x80010106).value
        initialize_com = Mock(return_value=changed_mode)
        uninitialize_com = Mock()
        free_task_memory = Mock()
        ole32 = Mock(
            CoInitializeEx=initialize_com,
            CoUninitialize=uninitialize_com,
            CoTaskMemFree=free_task_memory,
        )
        expected_path = Mock(name="resolved_local_app_data")
        path_candidate = Mock()
        path_candidate.is_absolute.return_value = True
        path_candidate.resolve.return_value = expected_path

        with (
            patch.object(workflow_module.os, "name", "nt"),
            patch.object(workflow_module.ctypes, "WinDLL", create=True, side_effect=(shell32, ole32)),
            patch.object(workflow_module.ctypes, "wstring_at", return_value=r"C:\Users\alice\AppData\Local"),
            patch.object(workflow_module, "Path", return_value=path_candidate),
        ):
            actual_path = _windows_local_app_data_path()

        self.assertIs(actual_path, expected_path)
        get_known_folder_path.assert_called_once()
        free_task_memory.assert_called_once()
        uninitialize_com.assert_not_called()

    def test_windows_known_folder_fails_closed_when_com_initialization_fails(self) -> None:
        from leveraged_trader import workflow as workflow_module

        get_known_folder_path = Mock(name="SHGetKnownFolderPath")
        shell32 = Mock(SHGetKnownFolderPath=get_known_folder_path)
        initialize_com = Mock(return_value=ctypes.c_int32(0x8007000E).value)
        uninitialize_com = Mock()
        free_task_memory = Mock()
        ole32 = Mock(
            CoInitializeEx=initialize_com,
            CoUninitialize=uninitialize_com,
            CoTaskMemFree=free_task_memory,
        )

        with (
            patch.object(workflow_module.os, "name", "nt"),
            patch.object(workflow_module.ctypes, "WinDLL", create=True, side_effect=(shell32, ole32)),
            self.assertRaisesRegex(OSError, r"initialize COM \(HRESULT 0x8007000E\)"),
        ):
            _windows_local_app_data_path()

        get_known_folder_path.assert_not_called()
        free_task_memory.assert_not_called()
        uninitialize_com.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX sticky-directory semantics are required")
    def test_shared_output_parent_squatting_cannot_bypass_or_disable_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_dir = root / "runtime"
            runtime_dir.mkdir(mode=0o700)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            shared_parent = root / "shared"
            shared_parent.mkdir(mode=0o777)
            shared_parent.chmod(0o1777)
            output_dir = shared_parent / "reports"
            output_dir.mkdir(mode=0o700)
            displaced_output = shared_parent / "displaced-reports"
            digest = hashlib.sha256(os.fsencode(output_dir.resolve())).hexdigest()[:32]
            legacy_anchor = shared_parent / f".leveraged-trader-output-{digest}.lock"
            legacy_anchor.mkdir(mode=0o700)
            first_db = root / "database-one" / "state.sqlite"
            second_db = root / "database-two" / "state.sqlite"

            with (
                patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(runtime_dir)}),
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                self.assertRaisesRegex(PermissionError, "changed while runtime files were in use"),
                _workflow_run_lock(str(first_db), str(output_dir)),
            ):
                output_dir.rename(displaced_output)
                output_dir.mkdir(mode=0o700)
                with (
                    patch.dict(os.environ, {}, clear=True),
                    self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                    _workflow_run_lock(str(second_db), str(output_dir)),
                ):
                    self.fail("substituted shared output directory bypassed the per-user anchor")

            self.assertTrue(legacy_anchor.is_dir())

    @unittest.skipUnless(os.name == "posix", "POSIX file modes are required")
    def test_workflow_creates_private_custom_database_parent_under_common_umasks(self) -> None:
        for umask_value in (0, 0o002):
            with self.subTest(umask=oct(umask_value)), tempfile.TemporaryDirectory() as tmp:
                database_parent = Path(tmp) / "new-parent"
                db_path = database_parent / "state.sqlite"
                output_dir = Path(tmp) / "reports"
                old_umask = os.umask(umask_value)
                try:
                    with _workflow_run_lock(str(db_path), str(output_dir)):
                        _initialize_state_db(str(db_path))
                        self.assertEqual(stat.S_IMODE(database_parent.stat().st_mode), 0o700)
                        self.assertEqual(stat.S_IMODE(Path(f"{db_path}.lock").stat().st_mode), 0o600)
                        self.assertEqual(stat.S_IMODE(db_path.stat().st_mode), 0o600)
                finally:
                    os.umask(old_umask)

    @unittest.skipUnless(os.name == "posix", "POSIX file modes are required")
    def test_reconciliation_rejects_unsafe_output_before_database_or_broker_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reports"
            output_dir.mkdir(mode=0o777)
            output_dir.chmod(0o777)
            db_path = Path(tmp) / "state.sqlite"
            with (
                patch("leveraged_trader.workflow._initialize_state_db") as mock_initialize,
                patch("leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db") as mock_reconcile,
                self.assertRaisesRegex(PermissionError, "must not be writable by group or other"),
            ):
                run_alpaca_reconciliation(
                    db_path=str(db_path),
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="key",
                        api_secret_key="secret",
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                )

            mock_initialize.assert_not_called()
            mock_reconcile.assert_not_called()
            self.assertFalse(db_path.exists())
            self.assertFalse(Path(f"{db_path}.lock").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX symlink handling is required")
    def test_reconciliation_rejects_symlinked_output_before_database_or_broker_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_target = Path(tmp) / "actual-reports"
            output_target.mkdir(mode=0o700)
            output_dir = Path(tmp) / "reports"
            output_dir.symlink_to(output_target, target_is_directory=True)
            db_path = Path(tmp) / "state.sqlite"
            with (
                patch("leveraged_trader.workflow._initialize_state_db") as mock_initialize,
                patch("leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db") as mock_reconcile,
                self.assertRaisesRegex(PermissionError, "not a symbolic link"),
            ):
                run_alpaca_reconciliation(
                    db_path=str(db_path),
                    alpaca_cfg=AlpacaOrderConfig(
                        sell_enabled=True,
                        api_key_id="key",
                        api_secret_key="secret",
                    ),
                    output_dir=str(output_dir),
                    no_color=True,
                )

            mock_initialize.assert_not_called()
            mock_reconcile.assert_not_called()
            self.assertFalse(db_path.exists())
            self.assertFalse(Path(f"{db_path}.lock").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX symlink handling is required")
    def test_workflow_lock_rejects_symlink_without_touching_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "reports"
            output_dir.mkdir(mode=0o700)
            victim = Path(tmp) / "victim"
            victim.write_text("must remain unchanged\n", encoding="utf-8")
            victim.chmod(0o644)
            (output_dir / ".leveraged-trader.lock").symlink_to(victim)

            with (
                self.assertRaisesRegex(PermissionError, "must not be a symbolic link"),
                _workflow_run_lock(str(Path(tmp) / "state.sqlite"), str(output_dir)),
            ):
                self.fail("workflow unexpectedly acquired a substituted output lock")

            self.assertEqual(victim.read_text(encoding="utf-8"), "must remain unchanged\n")
            self.assertEqual(stat.S_IMODE(victim.stat().st_mode), 0o644)

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

    @unittest.skipUnless(os.name == "posix", "POSIX symbolic-link handling is required")
    def test_symlinked_legacy_database_lock_is_rejected_without_touching_target_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state.sqlite"
            output_dir = Path(tmp) / "outputs"
            victim = Path(tmp) / "victim"
            victim.write_text("must remain unchanged\n", encoding="utf-8")
            victim.chmod(0o644)
            Path(f"{db_path}.lock").symlink_to(victim)

            with (
                self.assertRaisesRegex(PermissionError, "must not be a symbolic link"),
                _workflow_run_lock(str(db_path), str(output_dir)),
            ):
                self.fail("workflow unexpectedly followed a substituted database lock")

            self.assertEqual(victim.read_text(encoding="utf-8"), "must remain unchanged\n")
            self.assertEqual(stat.S_IMODE(victim.stat().st_mode), 0o644)
            self.assertFalse(output_dir.exists())

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

    @unittest.skipUnless(os.name == "posix", "POSIX directory permissions are required")
    def test_unsafe_database_parent_is_rejected_before_output_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database_parent = root / "unsafe-database-parent"
            database_parent.mkdir(mode=0o700)
            database_parent.chmod(0o770)
            database_path = database_parent / "state.sqlite"
            output_dir = root / "outputs"

            with (
                self.assertRaisesRegex(PermissionError, "SQLite database parent directory"),
                _workflow_run_lock(str(database_path), str(output_dir)),
            ):
                self.fail("workflow accepted an unsafe SQLite database parent")

            self.assertFalse(database_path.exists())
            self.assertFalse(Path(f"{database_path}.lock").exists())
            self.assertFalse(output_dir.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX directory permissions are required")
    def test_unsafe_output_parent_is_rejected_before_database_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database_parent = root / "database"
            database_path = database_parent / "state.sqlite"
            unsafe_output_parent = root / "unsafe-output-parent"
            unsafe_output_parent.mkdir(mode=0o700)
            unsafe_output_parent.chmod(0o770)
            output_dir = unsafe_output_parent / "outputs"

            with (
                self.assertRaisesRegex(PermissionError, "Report output directory"),
                _workflow_run_lock(str(database_path), str(output_dir)),
            ):
                self.fail("workflow accepted an unsafe report-output parent")

            self.assertFalse(database_parent.exists())
            self.assertFalse(Path(f"{database_path}.lock").exists())
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

    def test_async_workflow_snapshots_mutable_inputs_before_returned_coroutine_runs(self) -> None:
        base_cfg = BacktestConfig(rsi_period=14)
        universe_cfg = UniverseConfig(request_timeout_seconds=30)
        buy_rsi_values = [30.0]
        profit_target_values = [1.5]
        alpaca_cfg = AlpacaOrderConfig()
        tradier_cfg = TradierMarketDataConfig(enabled=False, access_token=" token-a ")
        short_buy_rsi_values = [70.0]
        reporter = Mock(spec=WorkflowReporter)
        locked_output = Mock(database_path="locked.sqlite")
        account_lock_modes: list[bool] = []

        @contextmanager
        def fake_workflow_lock(
            _db_path: str,
            _output_dir: str,
            *,
            serialize_alpaca_account: bool = False,
        ):
            account_lock_modes.append(serialize_alpaca_account)
            yield locked_output

        with (
            patch("leveraged_trader.workflow._workflow_run_lock", new=fake_workflow_lock),
            patch(
                "leveraged_trader.workflow._run_resumable_optimizations_unlocked",
                new_callable=AsyncMock,
            ) as inner_worker,
        ):
            suspended_workflow = run_resumable_optimizations_async(
                mode="update",
                db_path="state.sqlite",
                base_cfg=base_cfg,
                universe_cfg=universe_cfg,
                buy_rsi_values=buy_rsi_values,
                profit_target_values=profit_target_values,
                alpaca_cfg=alpaca_cfg,
                output_dir="outputs",
                reporter=reporter,
                tradier_cfg=tradier_cfg,
                short_buy_rsi_values=short_buy_rsi_values,
            )

            base_cfg.rsi_period = 15
            universe_cfg.request_timeout_seconds = 31
            buy_rsi_values[:] = [40.0]
            profit_target_values[:] = [2.5]
            alpaca_cfg.enabled = True
            alpaca_cfg.sell_enabled = True
            alpaca_cfg.api_key_id = "mutated-key"
            alpaca_cfg.api_secret_key = "mutated-secret"
            tradier_cfg.enabled = True
            tradier_cfg.access_token = "token-b"
            short_buy_rsi_values[:] = [60.0]

            asyncio.run(suspended_workflow)

        self.assertEqual(account_lock_modes, [False])
        inner_worker.assert_awaited_once()
        inner_kwargs = inner_worker.await_args.kwargs
        self.assertIsNot(inner_kwargs["base_cfg"], base_cfg)
        self.assertEqual(inner_kwargs["base_cfg"].rsi_period, 14)
        self.assertIsNot(inner_kwargs["universe_cfg"], universe_cfg)
        self.assertEqual(inner_kwargs["universe_cfg"].request_timeout_seconds, 30)
        self.assertEqual(inner_kwargs["buy_rsi_values"], [30.0])
        self.assertIsNot(inner_kwargs["buy_rsi_values"], buy_rsi_values)
        self.assertEqual(inner_kwargs["profit_target_values"], [1.5])
        self.assertIsNot(inner_kwargs["profit_target_values"], profit_target_values)
        self.assertIsNot(inner_kwargs["alpaca_cfg"], alpaca_cfg)
        self.assertFalse(inner_kwargs["alpaca_cfg"].enabled)
        self.assertFalse(inner_kwargs["alpaca_cfg"].sell_enabled)
        self.assertIsNone(inner_kwargs["alpaca_cfg"].api_key_id)
        self.assertIsNot(inner_kwargs["tradier_cfg"], tradier_cfg)
        self.assertFalse(inner_kwargs["tradier_cfg"].enabled)
        self.assertEqual(inner_kwargs["tradier_cfg"].access_token, "token-a")
        self.assertEqual(inner_kwargs["short_buy_rsi_values"], [70.0])
        self.assertIsNot(inner_kwargs["short_buy_rsi_values"], short_buy_rsi_values)
        self.assertIs(inner_kwargs["reporter"], reporter)

    @unittest.skipUnless(os.name == "posix", "POSIX account locks are required")
    def test_alpaca_account_lock_serializes_distinct_database_and_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            first_db = root / "state-one.sqlite"
            first_output = root / "outputs-one"
            second_db = root / "state-two.sqlite"
            second_output = root / "outputs-two"

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                _workflow_run_lock(
                    str(first_db),
                    str(first_output),
                    serialize_alpaca_account=True,
                ),
            ):
                # A research-only run has no broker side effects and remains
                # independent when its database and output paths are distinct.
                with _workflow_run_lock(str(second_db), str(second_output)):
                    pass

                with (
                    self.assertRaisesRegex(WorkflowRunError, "Alpaca paper account"),
                    _workflow_run_lock(
                        str(second_db),
                        str(second_output),
                        serialize_alpaca_account=True,
                    ),
                ):
                    self.fail("distinct broker workflows concurrently acquired the Alpaca account lock")

    @unittest.skipUnless(os.name == "posix", "POSIX directory identity is required")
    def test_swapped_output_directory_cannot_bypass_lock_or_receive_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            displaced_output = Path(tmp) / "displaced-outputs"
            with (
                self.assertRaisesRegex(PermissionError, "changed while runtime files were in use"),
                _workflow_run_lock(
                    str(Path(tmp) / "state-one.sqlite"),
                    str(output_dir),
                ) as locked_output_dir,
            ):
                output_dir.rename(displaced_output)
                output_dir.mkdir(mode=0o700)

                with (
                    self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                    _workflow_run_lock(
                        str(Path(tmp) / "state-two.sqlite"),
                        str(output_dir),
                    ),
                ):
                    self.fail("substituted output directory bypassed the parent-anchored lock")

                with self.assertRaisesRegex(PermissionError, "changed while runtime files were in use"):
                    _atomic_to_csv(
                        pd.DataFrame([{"value": "must not be published"}]),
                        Path(locked_output_dir) / "report.csv",
                        index=False,
                    )

            self.assertFalse((output_dir / "report.csv").exists())
            self.assertFalse((displaced_output / "report.csv").exists())

    @unittest.skipUnless(os.name == "posix", "POSIX directory identity is required")
    def test_research_publication_does_not_recreate_rotated_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            account_home = root / "account-home"
            account_home.mkdir(mode=0o700)
            runtime_parent = root / "runtime"
            runtime_parent.mkdir(mode=0o700)
            output_dir = runtime_parent / "reports"
            displaced_runtime_parent = root / "displaced-runtime"

            with (
                patch(
                    "leveraged_trader.workflow.pwd.getpwuid",
                    return_value=Mock(pw_dir=str(account_home)),
                ),
                self.assertRaises(OSError),
                _workflow_run_lock(
                    str(root / "state.sqlite"),
                    str(output_dir),
                ) as locked_output_dir,
            ):
                runtime_parent.rename(displaced_runtime_parent)
                runtime_parent.mkdir(mode=0o700)
                _persist_workflow_research_outputs(
                    output_dir=locked_output_dir,
                    curves=pd.DataFrame(),
                    optimization_summary=pd.DataFrame(),
                    buy_signals=pd.DataFrame(),
                    eligible_buy_signals=pd.DataFrame(),
                    sell_signals=pd.DataFrame(),
                )

            self.assertTrue(runtime_parent.is_dir())
            self.assertFalse(output_dir.exists())
            self.assertEqual(
                list((displaced_runtime_parent / "reports").glob("*.csv")),
                [],
            )

    @unittest.skipUnless(os.name == "posix", "POSIX directory descriptors are required")
    def test_stale_report_cleanup_does_not_unlink_from_replacement_output_directory(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            displaced_output = Path(tmp) / "displaced-outputs"
            filename = _RESEARCH_REPORT_FILENAMES[0]
            (output_dir / filename).write_text("stale report\n", encoding="utf-8")
            real_open_directory = workflow_module.open_owned_runtime_directory

            @contextmanager
            def swap_after_directory_open(directory: object, *, label: str):
                with real_open_directory(directory, label=label) as descriptor:
                    output_dir.rename(displaced_output)
                    output_dir.mkdir(mode=0o700)
                    (output_dir / filename).write_text(
                        "replacement must remain\n",
                        encoding="utf-8",
                    )
                    yield descriptor

            with (
                patch.object(
                    workflow_module,
                    "open_owned_runtime_directory",
                    new=swap_after_directory_open,
                ),
                self.assertRaisesRegex(PermissionError, "changed while runtime files were in use"),
            ):
                _clear_stale_workflow_research_outputs(str(output_dir))

            self.assertFalse((displaced_output / filename).exists())
            self.assertEqual(
                (output_dir / filename).read_text(encoding="utf-8"),
                "replacement must remain\n",
            )

    @unittest.skipUnless(os.name == "posix", "POSIX directory descriptors are required")
    def test_fixed_research_cleanup_restores_entry_replaced_after_identity_check(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            filename = _RESEARCH_REPORT_FILENAMES[0]
            report_path = output_dir / filename
            report_path.write_text("stale report\n", encoding="utf-8")
            real_replace = workflow_module._replace_report_path
            substituted = False

            def substitute_before_quarantine(
                source: Path,
                destination: Path,
                *,
                directory_descriptor: int | None,
            ) -> None:
                nonlocal substituted
                if (
                    not substituted
                    and source.name == filename
                    and destination.name.startswith(".leveraged-trader-report-cleanup-")
                ):
                    assert directory_descriptor is not None
                    os.unlink(source.name, dir_fd=directory_descriptor)
                    replacement_descriptor = os.open(
                        source.name,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                        dir_fd=directory_descriptor,
                    )
                    try:
                        os.write(replacement_descriptor, b"replacement must remain\n")
                    finally:
                        os.close(replacement_descriptor)
                    substituted = True
                real_replace(
                    source,
                    destination,
                    directory_descriptor=directory_descriptor,
                )

            with (
                patch.object(
                    workflow_module,
                    "_replace_report_path",
                    side_effect=substitute_before_quarantine,
                ),
                self.assertRaisesRegex(OSError, "changed before unlink.*restored"),
            ):
                _clear_stale_workflow_research_outputs(str(output_dir))

            self.assertTrue(substituted)
            self.assertEqual(report_path.read_text(encoding="utf-8"), "replacement must remain\n")
            self.assertEqual(list(output_dir.glob(".leveraged-trader-report-cleanup-*.tmp")), [])

    @unittest.skipUnless(os.name == "posix", "POSIX directory descriptors are required")
    def test_report_cleanup_helpers_preserve_replacements_raced_before_quarantine(self) -> None:
        from leveraged_trader import workflow as workflow_module

        for helper_name in ("_unlink_matching_report_path", "_unlink_non_directory_entry"):
            with self.subTest(helper=helper_name), tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)
                report_path = output_dir / "report.csv"
                report_path.write_text("original\n", encoding="utf-8")
                expected = os.lstat(report_path)
                real_replace = workflow_module._replace_report_path
                substituted = False

                def substitute_before_quarantine(
                    source: Path,
                    destination: Path,
                    real_replace: object = real_replace,
                    *,
                    directory_descriptor: int | None,
                ) -> None:
                    nonlocal substituted
                    if not substituted and destination.name.startswith(".leveraged-trader-report-cleanup-"):
                        source.unlink()
                        source.write_text("replacement\n", encoding="utf-8")
                        substituted = True
                    assert callable(real_replace)
                    real_replace(
                        source,
                        destination,
                        directory_descriptor=directory_descriptor,
                    )

                helper = getattr(workflow_module, helper_name)
                arguments = (report_path, expected) if helper_name == "_unlink_matching_report_path" else (report_path,)
                with (
                    patch.object(
                        workflow_module,
                        "_replace_report_path",
                        side_effect=substitute_before_quarantine,
                    ),
                    patch.object(
                        workflow_module,
                        "_report_cleanup_anchor_can_remain_open_during_replace",
                        return_value=False,
                    ),
                    self.assertRaisesRegex(OSError, "changed before unlink.*restored"),
                ):
                    helper(*arguments, directory_descriptor=None)

                self.assertTrue(substituted)
                self.assertEqual(report_path.read_text(encoding="utf-8"), "replacement\n")
                self.assertEqual(list(output_dir.glob(".leveraged-trader-report-cleanup-*.tmp")), [])

    def test_report_cleanup_closes_nonsharing_anchor_before_quarantine_replace(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.csv"
            report_path.write_text("original\n", encoding="utf-8")
            expected = os.lstat(report_path)
            real_open_anchor = workflow_module._open_report_cleanup_anchor
            real_replace = workflow_module._replace_report_path
            anchor_descriptors: list[int] = []

            def capture_anchor(
                path: Path,
                expected: os.stat_result,
                *,
                directory_descriptor: int | None,
            ) -> tuple[int, os.stat_result]:
                anchor = real_open_anchor(
                    path,
                    expected,
                    directory_descriptor=directory_descriptor,
                )
                anchor_descriptors.append(anchor[0])
                return anchor

            def verify_anchor_closed(
                source: Path,
                destination: Path,
                *,
                directory_descriptor: int | None,
            ) -> None:
                self.assertEqual(len(anchor_descriptors), 1)
                with self.assertRaises(OSError):
                    os.fstat(anchor_descriptors[0])
                real_replace(
                    source,
                    destination,
                    directory_descriptor=directory_descriptor,
                )

            with (
                patch.object(
                    workflow_module,
                    "_open_report_cleanup_anchor",
                    side_effect=capture_anchor,
                ),
                patch.object(
                    workflow_module,
                    "_replace_report_path",
                    side_effect=verify_anchor_closed,
                ),
                patch.object(
                    workflow_module,
                    "_report_cleanup_anchor_can_remain_open_during_replace",
                    return_value=False,
                ),
            ):
                workflow_module._unlink_matching_report_path(
                    report_path,
                    expected,
                    directory_descriptor=None,
                )

            self.assertFalse(report_path.exists())
            self.assertEqual(list(Path(tmp).glob(".leveraged-trader-report-cleanup-*.tmp")), [])

    def test_report_cleanup_anchor_preserves_validation_failure_when_close_fails(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.csv"
            report_path.write_text("original\n", encoding="utf-8")
            expected = os.lstat(report_path)
            real_close = workflow_module.os.close

            def close_then_fail(file_descriptor: int) -> None:
                real_close(file_descriptor)
                raise OSError("anchor descriptor close failed")

            with (
                patch.object(workflow_module, "_same_file_identity", return_value=False),
                patch.object(workflow_module.os, "close", side_effect=close_then_fail),
                self.assertRaisesRegex(OSError, "changed before it could be pinned") as raised,
            ):
                workflow_module._open_report_cleanup_anchor(
                    report_path,
                    expected,
                    directory_descriptor=None,
                )

            self.assertEqual(len(raised.exception.__notes__), 1)
            self.assertIn("anchor descriptor close failed", raised.exception.__notes__[0])

    @unittest.skipUnless(os.name == "posix", "POSIX open-anchor rename behavior is required")
    def test_report_cleanup_preserves_rename_failure_when_anchor_close_fails(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.csv"
            report_path.write_text("original\n", encoding="utf-8")
            expected = os.lstat(report_path)
            real_open_anchor = workflow_module._open_report_cleanup_anchor
            real_close = workflow_module.os.close
            anchor_descriptors: list[int] = []

            def capture_anchor(
                path: Path,
                expected: os.stat_result,
                *,
                directory_descriptor: int | None,
            ) -> tuple[int, os.stat_result]:
                anchor = real_open_anchor(
                    path,
                    expected,
                    directory_descriptor=directory_descriptor,
                )
                anchor_descriptors.append(anchor[0])
                return anchor

            def close_then_fail(file_descriptor: int) -> None:
                real_close(file_descriptor)
                if file_descriptor in anchor_descriptors:
                    raise OSError("anchor descriptor close failed")

            with (
                patch.object(
                    workflow_module,
                    "_open_report_cleanup_anchor",
                    side_effect=capture_anchor,
                ),
                patch.object(
                    workflow_module,
                    "_replace_report_path",
                    side_effect=RuntimeError("quarantine rename failed"),
                ),
                patch.object(workflow_module.os, "close", side_effect=close_then_fail),
                self.assertRaisesRegex(RuntimeError, "quarantine rename failed") as raised,
            ):
                workflow_module._quarantine_and_unlink_report_path(
                    report_path,
                    expected,
                    directory_descriptor=None,
                )

            self.assertEqual(len(raised.exception.__notes__), 1)
            self.assertIn("anchor descriptor close failed", raised.exception.__notes__[0])
            self.assertEqual(report_path.read_text(encoding="utf-8"), "original\n")

    @unittest.skipUnless(os.name == "posix", "POSIX open-anchor rename behavior is required")
    def test_report_cleanup_preserves_verification_failure_when_anchor_close_fails(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.csv"
            report_path.write_text("original\n", encoding="utf-8")
            expected = os.lstat(report_path)
            real_open_anchor = workflow_module._open_report_cleanup_anchor
            real_report_path_status = workflow_module._report_path_status
            real_close = workflow_module.os.close
            anchor_descriptors: list[int] = []

            def capture_anchor(
                path: Path,
                expected: os.stat_result,
                *,
                directory_descriptor: int | None,
            ) -> tuple[int, os.stat_result]:
                anchor = real_open_anchor(
                    path,
                    expected,
                    directory_descriptor=directory_descriptor,
                )
                anchor_descriptors.append(anchor[0])
                return anchor

            def fail_quarantine_verification(
                path: Path,
                *,
                directory_descriptor: int | None,
            ) -> os.stat_result:
                observed = real_report_path_status(
                    path,
                    directory_descriptor=directory_descriptor,
                )
                if path.name.startswith(".leveraged-trader-report-cleanup-"):
                    raise RuntimeError("quarantine verification failed")
                return observed

            def close_then_fail(file_descriptor: int) -> None:
                real_close(file_descriptor)
                if file_descriptor in anchor_descriptors:
                    raise OSError("anchor descriptor close failed")

            with (
                patch.object(
                    workflow_module,
                    "_open_report_cleanup_anchor",
                    side_effect=capture_anchor,
                ),
                patch.object(
                    workflow_module,
                    "_report_path_status",
                    side_effect=fail_quarantine_verification,
                ),
                patch.object(workflow_module.os, "close", side_effect=close_then_fail),
                self.assertRaisesRegex(RuntimeError, "quarantine verification failed") as raised,
            ):
                workflow_module._quarantine_and_unlink_report_path(
                    report_path,
                    expected,
                    directory_descriptor=None,
                )

            self.assertEqual(len(raised.exception.__notes__), 1)
            self.assertIn("anchor descriptor close failed", raised.exception.__notes__[0])
            self.assertFalse(report_path.exists())
            self.assertEqual(len(list(Path(tmp).glob(".leveraged-trader-report-cleanup-*.tmp"))), 1)

    @unittest.skipUnless(os.name == "posix", "POSIX open-anchor rename behavior is required")
    def test_report_cleanup_raises_anchor_close_failure_after_success(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.csv"
            report_path.write_text("original\n", encoding="utf-8")
            expected = os.lstat(report_path)
            real_open_anchor = workflow_module._open_report_cleanup_anchor
            real_close = workflow_module.os.close
            anchor_descriptors: list[int] = []

            def capture_anchor(
                path: Path,
                expected: os.stat_result,
                *,
                directory_descriptor: int | None,
            ) -> tuple[int, os.stat_result]:
                anchor = real_open_anchor(
                    path,
                    expected,
                    directory_descriptor=directory_descriptor,
                )
                anchor_descriptors.append(anchor[0])
                return anchor

            def close_then_fail(file_descriptor: int) -> None:
                real_close(file_descriptor)
                if file_descriptor in anchor_descriptors:
                    raise OSError("anchor descriptor close failed")

            with (
                patch.object(
                    workflow_module,
                    "_open_report_cleanup_anchor",
                    side_effect=capture_anchor,
                ),
                patch.object(workflow_module.os, "close", side_effect=close_then_fail),
                self.assertRaisesRegex(OSError, "anchor descriptor close failed"),
            ):
                workflow_module._quarantine_and_unlink_report_path(
                    report_path,
                    expected,
                    directory_descriptor=None,
                )

            self.assertFalse(report_path.exists())

    def test_cancelled_blocking_worker_keeps_workflow_lock_until_worker_stops(self) -> None:
        async def exercise(db_path: str, first_output: str, second_output: str) -> None:
            started = threading.Event()
            release = threading.Event()

            def blocking_worker() -> None:
                started.set()
                release.wait(timeout=5)

            async def guarded_worker() -> None:
                with _workflow_run_lock(db_path, first_output):
                    await _run_blocking(None, blocking_worker)

            task = asyncio.create_task(guarded_worker())
            while not started.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            await asyncio.sleep(0.01)

            self.assertFalse(task.done())
            with (
                self.assertRaisesRegex(WorkflowRunError, "Another workflow"),
                _workflow_run_lock(db_path, second_output),
            ):
                self.fail("cancelled worker released its workflow lock before stopping")

            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with _workflow_run_lock(db_path, second_output):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(
                exercise(
                    str(Path(tmp) / "state.sqlite"),
                    str(Path(tmp) / "outputs-one"),
                    str(Path(tmp) / "outputs-two"),
                )
            )

    def test_cancelled_yahoo_worker_drains_after_killing_internal_stall(self) -> None:
        async def exercise() -> None:
            process_started = threading.Event()
            processes: list[subprocess.Popen[bytes]] = []
            real_popen = subprocess.Popen

            def capture_process(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
                process = real_popen(*args, **kwargs)
                processes.append(process)
                process_started.set()
                return process

            stalled_yfinance_command = (
                sys.executable,
                "-I",
                "-c",
                (
                    "import sys; import yfinance.multi as multi; "
                    "sys.stdin.buffer.read(); "
                    "multi._download_one_threaded=lambda *args,**kwargs: None; "
                    "multi._download_impl(multi._DownloadCtx(),tickers=['SPY'],threads=True,progress=False)"
                ),
            )

            with (
                patch(
                    "leveraged_trader.market_data._YFINANCE_REQUEST_TIMEOUT_SECONDS",
                    0.2,
                ),
                patch(
                    "leveraged_trader._http_deadline_worker.subprocess.Popen",
                    side_effect=capture_process,
                ),
                patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_COMMAND", stalled_yfinance_command),
                ThreadPoolExecutor(max_workers=1) as executor,
            ):
                task = asyncio.create_task(_run_blocking(executor, load_market_data, symbols=["TQQQ"]))
                while not process_started.is_set():
                    await asyncio.sleep(0.001)
                task.cancel()
                await asyncio.sleep(0.001)

                self.assertFalse(task.done())
                with self.assertRaises(asyncio.CancelledError):
                    async with asyncio.timeout(1):
                        await task

            self.assertEqual(len(processes), 1)
            self.assertIsNotNone(processes[0].returncode)

        asyncio.run(exercise())

    def test_blocking_executor_completions_do_not_require_external_heartbeats(self) -> None:
        async def exercise() -> None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                async with asyncio.timeout(2):
                    results = [await _run_blocking(executor, lambda value: value, value) for value in range(25)]
            self.assertEqual(results, list(range(25)))

        asyncio.run(exercise())

    @unittest.skipUnless(os.name == "posix", "POSIX fsync ordering is required")
    def test_atomic_csv_syncs_file_before_replace_and_directory_afterward(self) -> None:
        from leveraged_trader import workflow as workflow_module

        real_fsync = os.fsync
        real_publish = workflow_module._publish_verified_report
        events: list[str] = []

        def observe_fsync(file_descriptor: int) -> None:
            events.append("sync-directory" if stat.S_ISDIR(os.fstat(file_descriptor).st_mode) else "sync-file")
            real_fsync(file_descriptor)

        def observe_publish(*args: object, **kwargs: object) -> None:
            events.append("replace")
            real_publish(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            destination = output_dir / "report.csv"
            destination.write_text("value\nold\n", encoding="utf-8")

            with (
                patch.object(workflow_module.os, "fsync", side_effect=observe_fsync),
                patch.object(workflow_module, "_publish_verified_report", side_effect=observe_publish),
            ):
                _atomic_to_csv(pd.DataFrame([{"value": "new"}]), destination, index=False)

            self.assertEqual(events, ["sync-file", "replace", "sync-directory"])
            self.assertEqual(pd.read_csv(destination)["value"].tolist(), ["new"])

    def test_atomic_csv_neutralizes_formula_text_without_mutating_the_frame(self) -> None:
        dangerous_values = (
            "=2+2",
            "+cmd",
            "-2+3",
            "@SUM(A1:A2)",
            "\t=2+2",
            "\r=2+2",
            "\n=2+2",
            "  =2+2",
            "ordinary",
        )
        frame = pd.DataFrame(
            {
                "=diagnostic": dangerous_values,
                "number": [-1, *range(1, len(dangerous_values))],
                "date": pd.date_range("2024-01-01", periods=len(dangerous_values)),
            },
            index=pd.Index(["@first", *range(1, len(dangerous_values))], name="+row"),
        )
        original = frame.copy(deep=True)

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "report.csv"
            _atomic_to_csv(frame, destination, index=True)
            with destination.open(encoding="utf-8", newline="") as file:
                rows = list(csv.reader(file))

        self.assertEqual(rows[0], ["'+row", "'=diagnostic", "number", "date"])
        self.assertEqual(rows[1][0], "'@first")
        self.assertEqual(
            [row[1] for row in rows[1:]],
            ["'" + value if value != "ordinary" else value for value in dangerous_values],
        )
        self.assertEqual(rows[1][2], "-1")
        self.assertEqual(rows[1][3], "2024-01-01")
        pd.testing.assert_frame_equal(frame, original)

    @unittest.skipUnless(os.name == "posix", "POSIX directory fsync behavior is required")
    def test_atomic_csv_surfaces_directory_sync_failure_after_safe_replace(self) -> None:
        from leveraged_trader import workflow as workflow_module

        real_fsync = os.fsync

        def fail_directory_sync(file_descriptor: int) -> None:
            if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
                raise OSError("directory sync failed")
            real_fsync(file_descriptor)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            destination = output_dir / "report.csv"
            destination.write_text("value\nold\n", encoding="utf-8")

            with (
                patch.object(workflow_module.os, "fsync", side_effect=fail_directory_sync),
                self.assertRaisesRegex(OSError, "directory sync failed"),
            ):
                _atomic_to_csv(pd.DataFrame([{"value": "new"}]), destination, index=False)

            self.assertEqual(pd.read_csv(destination)["value"].tolist(), ["new"])
            self.assertEqual(
                [entry.name for entry in output_dir.iterdir() if entry.name != destination.name],
                [],
            )

    def test_windows_report_replace_is_write_through(self) -> None:
        from leveraged_trader import workflow as workflow_module

        move_file = Mock(return_value=1)
        kernel32 = Mock(MoveFileExW=move_file)
        source = Path("temporary.csv")
        destination = Path("report.csv")

        with (
            patch.object(workflow_module.os, "name", "nt"),
            patch.object(workflow_module.ctypes, "WinDLL", create=True, return_value=kernel32),
        ):
            workflow_module._replace_report_path(
                source,
                destination,
                directory_descriptor=None,
            )

        move_file.assert_called_once_with(
            os.fspath(source),
            os.fspath(destination),
            0x00000001 | 0x00000008,
        )

    def test_atomic_csv_never_reopens_substituted_temporary_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            destination = output_dir / "report.csv"
            victim = output_dir / "victim"
            victim.write_text("must remain unchanged\n", encoding="utf-8")
            original_to_csv = pd.DataFrame.to_csv

            def substitute_temporary_path(
                frame: pd.DataFrame,
                output: object,
                *args: object,
                **kwargs: object,
            ) -> object:
                temporary_path = next(output_dir.glob(".report.csv.*.tmp"))
                temporary_path.unlink()
                temporary_path.symlink_to(victim)
                return original_to_csv(frame, output, *args, **kwargs)

            with (
                patch.object(pd.DataFrame, "to_csv", new=substitute_temporary_path),
                self.assertRaisesRegex(OSError, "Temporary report path"),
            ):
                _atomic_to_csv(pd.DataFrame([{"value": "new"}]), destination, index=False)

            self.assertEqual(victim.read_text(encoding="utf-8"), "must remain unchanged\n")
            self.assertFalse(destination.exists())
            self.assertEqual(list(output_dir.glob(".report.csv.*.tmp")), [])

    def test_atomic_csv_preserves_primary_failure_and_attempts_every_cleanup(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            destination = output_dir / "report.csv"
            cleanup_paths: list[Path] = []

            def fail_cleanup(path: Path, **_kwargs: object) -> None:
                cleanup_paths.append(path)
                raise OSError(f"cleanup failed for {path.name}")

            with (
                patch.object(pd.DataFrame, "to_csv", side_effect=RuntimeError("serialization failed")),
                patch.object(workflow_module, "_unlink_non_directory_entry", side_effect=fail_cleanup),
                self.assertRaisesRegex(RuntimeError, "serialization failed") as raised,
            ):
                _atomic_to_csv(pd.DataFrame([{"value": "new"}]), destination, index=False)

            self.assertEqual(len(cleanup_paths), 3)
            self.assertTrue(cleanup_paths[0].name.endswith(".tmp"))
            self.assertEqual(cleanup_paths[1], Path(f"{cleanup_paths[0]}.anchor"))
            self.assertEqual(cleanup_paths[2], Path(f"{cleanup_paths[0]}.previous"))
            self.assertEqual(len(raised.exception.__notes__), 3)
            self.assertTrue(all("cleanup failed" in note for note in raised.exception.__notes__))

    def test_atomic_csv_preserves_serialization_failure_when_temporary_close_fails(self) -> None:
        from leveraged_trader import workflow as workflow_module

        real_fdopen = os.fdopen

        class CloseFailingTemporaryFile:
            def __init__(self, file_descriptor: int, *args: object, **kwargs: object) -> None:
                self._file = real_fdopen(file_descriptor, *args, **kwargs)

            def close(self) -> None:
                self._file.close()
                raise OSError("temporary descriptor close failed")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            destination = output_dir / "report.csv"

            with (
                patch.object(
                    workflow_module.os,
                    "fdopen",
                    side_effect=lambda file_descriptor, *args, **kwargs: CloseFailingTemporaryFile(
                        file_descriptor,
                        *args,
                        **kwargs,
                    ),
                ),
                patch.object(pd.DataFrame, "to_csv", side_effect=RuntimeError("serialization failed")),
                self.assertRaisesRegex(RuntimeError, "serialization failed") as raised,
            ):
                _atomic_to_csv(pd.DataFrame([{"value": "new"}]), destination, index=False)

            self.assertEqual(len(raised.exception.__notes__), 1)
            self.assertIn("temporary descriptor close failed", raised.exception.__notes__[0])
            self.assertFalse(destination.exists())
            self.assertEqual(list(output_dir.glob(".report.csv.*.tmp*")), [])

    def test_atomic_csv_raises_cleanup_failure_after_successful_publication(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            destination = output_dir / "report.csv"
            real_cleanup = workflow_module._unlink_non_directory_entry
            cleanup_calls = 0

            def fail_first_cleanup(path: Path, **kwargs: object) -> None:
                nonlocal cleanup_calls
                cleanup_calls += 1
                if cleanup_calls == 1:
                    raise OSError("cleanup failed after publication")
                real_cleanup(path, **kwargs)

            with (
                patch.object(workflow_module, "_unlink_non_directory_entry", side_effect=fail_first_cleanup),
                self.assertRaisesRegex(OSError, "cleanup failed after publication"),
            ):
                _atomic_to_csv(pd.DataFrame([{"value": "new"}]), destination, index=False)

            self.assertEqual(cleanup_calls, 3)
            self.assertTrue(destination.is_file())

    def test_workflow_lock_removes_only_exact_stale_managed_atomic_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs"
            output_dir.mkdir(mode=0o700)
            run_id = "0123456789abcdef0123456789abcdef"
            stale_names = (
                f".buy_signals.csv.{run_id}.tmp",
                f".alpaca_order_results.csv.{run_id}.tmp.anchor",
                f".managed_positions.csv.{run_id}.tmp.previous",
                f".leveraged-trader-report-cleanup-{run_id}.tmp",
            )
            for entry_name in stale_names:
                (output_dir / entry_name).write_text("abandoned\n", encoding="utf-8")

            preserved_names = (
                f".unmanaged.csv.{run_id}.tmp",
                f".buy_signals.csv.{run_id[:-1]}.tmp",
                f".buy_signals.csv.{run_id.upper()}.tmp",
                f".buy_signals.csv.{run_id}.tmp.backup",
                f".leveraged-trader-report-cleanup-{run_id[:-1]}.tmp",
                f".leveraged-trader-report-cleanup-{run_id.upper()}.tmp",
                f".leveraged-trader-report-cleanup-{run_id}.tmp.backup",
            )
            for entry_name in preserved_names:
                (output_dir / entry_name).write_text("must remain\n", encoding="utf-8")
            preserved_directory = output_dir / f".sell_signals.csv.{run_id}.tmp"
            preserved_directory.mkdir()

            with _workflow_run_lock(str(root / "state.sqlite"), str(output_dir)):
                for entry_name in stale_names:
                    self.assertFalse((output_dir / entry_name).exists())
                for entry_name in preserved_names:
                    self.assertEqual(
                        (output_dir / entry_name).read_text(encoding="utf-8"),
                        "must remain\n",
                    )
                self.assertTrue(preserved_directory.is_dir())

    def test_workflow_lock_preserves_database_named_like_stale_report_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs"
            output_dir.mkdir(mode=0o700)
            run_id = "0123456789abcdef0123456789abcdef"
            database_names = (
                f".buy_signals.csv.{run_id}.tmp",
                f".alpaca_order_results.csv.{run_id}.tmp.anchor",
                f".managed_positions.csv.{run_id}.tmp.previous",
                f".leveraged-trader-report-cleanup-{run_id}.tmp",
            )

            for database_name in database_names:
                with self.subTest(database_name=database_name):
                    database_path = output_dir / database_name
                    with closing(sqlite3.connect(database_path)) as conn, conn:
                        conn.execute("CREATE TABLE marker (value TEXT NOT NULL)")
                        conn.execute("INSERT INTO marker VALUES ('must remain')")

                    with (
                        self.assertRaisesRegex(ValueError, "reserved report-cleanup name"),
                        _workflow_run_lock(str(database_path), str(output_dir)),
                    ):
                        self.fail("workflow accepted a database path reserved for stale-report cleanup")

                    self.assertTrue(database_path.is_file())
                    with closing(sqlite3.connect(database_path)) as conn, conn:
                        self.assertEqual(conn.execute("SELECT value FROM marker").fetchone(), ("must remain",))

    @unittest.skipUnless(os.name == "posix", "POSIX directory descriptors are required")
    def test_atomic_csv_rotation_preserves_replacement_and_cleans_original_temporary(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs"
            output_dir.mkdir(mode=0o700)
            displaced_output = root / "displaced-outputs"
            destination = output_dir / "report.csv"
            real_open_temporary = workflow_module._open_private_report_temporary

            def rotate_after_temporary_creation(
                report_path: Path,
                *,
                directory_descriptor: int | None,
            ) -> tuple[int, Path]:
                result = real_open_temporary(
                    report_path,
                    directory_descriptor=directory_descriptor,
                )
                output_dir.rename(displaced_output)
                output_dir.mkdir(mode=0o700)
                destination.write_text("replacement must remain\n", encoding="utf-8")
                return result

            with (
                patch.object(
                    workflow_module,
                    "_open_private_report_temporary",
                    side_effect=rotate_after_temporary_creation,
                ),
                self.assertRaisesRegex(PermissionError, "changed while runtime files were in use"),
            ):
                _atomic_to_csv(
                    pd.DataFrame([{"value": "must not be published"}]),
                    destination,
                    index=False,
                )

            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "replacement must remain\n",
            )
            self.assertEqual(list(displaced_output.glob(".report.csv.*")), [])
            self.assertFalse((displaced_output / destination.name).exists())

    @unittest.skipUnless(os.name == "posix", "POSIX file modes are required")
    def test_atomic_csv_restores_owner_access_under_restrictive_umask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "report.csv"
            old_umask = os.umask(0o777)
            try:
                _atomic_to_csv(pd.DataFrame([{"value": "published"}]), destination, index=False)
            finally:
                os.umask(old_umask)

            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            self.assertEqual(destination.read_text(encoding="utf-8"), "value\npublished\n")

    def test_atomic_csv_detects_publish_substitution_and_restores_previous_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            destination = output_dir / "report.csv"
            destination.write_text("previous report\n", encoding="utf-8")
            victim = output_dir / "victim"
            victim.write_text("must remain unchanged\n", encoding="utf-8")
            original_replace = os.replace
            substituted = False

            def substitute_source_before_replace(
                source: object,
                target: object,
                *,
                src_dir_fd: int | None = None,
                dst_dir_fd: int | None = None,
            ) -> None:
                nonlocal substituted
                source_path = Path(source)
                target_path = Path(target)
                if not substituted and source_path.suffix == ".tmp" and target_path.name == destination.name:
                    if src_dir_fd is None:
                        source_path.unlink()
                        source_path.symlink_to(victim)
                    else:
                        os.unlink(source_path.name, dir_fd=src_dir_fd)
                        os.symlink(victim.name, source_path.name, dir_fd=src_dir_fd)
                    substituted = True
                if src_dir_fd is None:
                    original_replace(source, target)
                else:
                    original_replace(
                        source,
                        target,
                        src_dir_fd=src_dir_fd,
                        dst_dir_fd=dst_dir_fd,
                    )

            with (
                patch("leveraged_trader.workflow.os.replace", side_effect=substitute_source_before_replace),
                self.assertRaisesRegex(OSError, "substituted during publication"),
            ):
                _atomic_to_csv(pd.DataFrame([{"value": "new"}]), destination, index=False)

            self.assertTrue(substituted)
            self.assertFalse(destination.is_symlink())
            self.assertEqual(destination.read_text(encoding="utf-8"), "previous report\n")
            self.assertEqual(victim.read_text(encoding="utf-8"), "must remain unchanged\n")
            self.assertEqual(list(output_dir.glob(".report.csv.*")), [])

    def test_atomic_csv_detects_substitution_after_anchor_removal(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            destination = output_dir / "report.csv"
            destination.write_text("previous report\n", encoding="utf-8")
            victim = output_dir / "victim"
            victim.write_text("must remain unchanged\n", encoding="utf-8")
            original_unlink = workflow_module._unlink_matching_report_path
            substituted = False

            def unlink_anchor_then_substitute(
                path: Path,
                expected: os.stat_result,
                *,
                directory_descriptor: int | None = None,
            ) -> None:
                nonlocal substituted
                original_unlink(
                    path,
                    expected,
                    directory_descriptor=directory_descriptor,
                )
                if not substituted and path.name.endswith(".anchor"):
                    destination.unlink()
                    destination.symlink_to(victim)
                    substituted = True

            with (
                patch.object(
                    workflow_module,
                    "_unlink_matching_report_path",
                    side_effect=unlink_anchor_then_substitute,
                ),
                self.assertRaisesRegex(OSError, "changed after publication"),
            ):
                _atomic_to_csv(pd.DataFrame([{"value": "new"}]), destination, index=False)

            self.assertTrue(substituted)
            self.assertFalse(destination.is_symlink())
            self.assertEqual(destination.read_text(encoding="utf-8"), "previous report\n")
            self.assertEqual(victim.read_text(encoding="utf-8"), "must remain unchanged\n")
            self.assertEqual(list(output_dir.glob(".report.csv.*")), [])

    def test_workflow_lock_cleans_up_after_case_probe_open_error(self) -> None:
        first_lock_file = Mock()
        first_lock_file.fileno.return_value = 42
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        tmp = temporary_directory.name
        with (
            patch(
                "leveraged_trader.workflow._open_workflow_lock_file",
                side_effect=[first_lock_file, PermissionError("denied")],
            ),
            patch("leveraged_trader.workflow._revalidate_workflow_lock_file"),
            patch("leveraged_trader.workflow.fcntl.flock") as mock_flock,
            self.assertRaisesRegex(PermissionError, "denied"),
            _workflow_run_lock(
                str(Path(tmp) / "state.sqlite"),
                str(Path(tmp) / "outputs"),
            ),
        ):
            self.fail("workflow unexpectedly acquired both locks")

        mock_flock.assert_not_called()
        first_lock_file.close.assert_called_once_with()

    def test_workflow_lock_release_preserves_body_failure_and_closes_every_lock(self) -> None:
        lock_files = [
            Mock(name="first_lock_file"),
            Mock(name="second_lock_file"),
            Mock(name="third_lock_file"),
            Mock(name="fourth_lock_file"),
            Mock(name="database_identity_lock_file"),
        ]
        original = KeyboardInterrupt("original workflow failure")
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "leveraged_trader.workflow._open_workflow_lock_file",
                side_effect=lock_files,
            ),
            patch("leveraged_trader.workflow._lock_file_nonblocking"),
            patch("leveraged_trader.workflow._revalidate_workflow_lock_file"),
            patch(
                "leveraged_trader.workflow._unlock_file",
                side_effect=[OSError("simulated unlock failure"), None, None, None, None],
            ) as mock_unlock,
            self.assertRaises(KeyboardInterrupt) as raised,
            _workflow_run_lock(
                str(Path(tmp) / "state.sqlite"),
                str(Path(tmp) / "outputs"),
            ),
        ):
            raise original

        self.assertIs(raised.exception, original)
        self.assertIn(
            "Failed lock-file unlock while releasing workflow locks: simulated unlock failure",
            raised.exception.__notes__,
        )
        self.assertEqual(mock_unlock.call_count, 5)
        for lock_file in lock_files:
            lock_file.close.assert_called_once_with()

    def test_state_connection_rolls_back_base_exception_from_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")

            original = KeyboardInterrupt("simulated interrupt")
            with (
                self.assertRaises(KeyboardInterrupt) as raised,
                _state_connection(db_path, immediate=True) as conn,
            ):
                conn.execute("INSERT INTO transaction_probe VALUES (1)")
                raise original

            self.assertIs(raised.exception, original)
            with _state_connection(db_path, immediate=True) as conn:
                conn.execute("INSERT INTO transaction_probe VALUES (2)")

            with closing(sqlite3.connect(db_path)) as conn, conn:
                values = conn.execute("SELECT value FROM transaction_probe").fetchall()
            self.assertEqual(values, [(2,)])

    def test_state_connection_preserves_body_failure_when_rollback_fails(self) -> None:
        class RollbackFailConnection(sqlite3.Connection):
            fail_next_rollback = False

            def rollback(self) -> None:
                if self.fail_next_rollback:
                    self.fail_next_rollback = False
                    raise sqlite3.OperationalError("simulated rollback failure")
                super().rollback()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")

            real_connect = sqlite3.connect

            def connect_with_failing_rollback(*args: object, **kwargs: object) -> sqlite3.Connection:
                return real_connect(*args, **kwargs, factory=RollbackFailConnection)

            original = RuntimeError("original transaction-body failure")
            with (
                patch(
                    "leveraged_trader.workflow.sqlite3.connect",
                    side_effect=connect_with_failing_rollback,
                ),
                self.assertRaises(RuntimeError) as raised,
                _state_connection(db_path, immediate=True) as conn,
            ):
                self.assertIsInstance(conn, RollbackFailConnection)
                conn.fail_next_rollback = True
                conn.execute("INSERT INTO transaction_probe VALUES (1)")
                failing_conn = conn
                raise original

            self.assertIs(raised.exception, original)
            self.assertIn(
                "Failed rollback after SQLite transaction-body failure: simulated rollback failure",
                raised.exception.__notes__,
            )
            with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                failing_conn.execute("SELECT 1")

    def test_state_connection_revalidates_and_preserves_commit_failure(self) -> None:
        class CommitFailConnection(sqlite3.Connection):
            commit_failure: BaseException | None = None

            def commit(self) -> None:
                if self.commit_failure is not None:
                    raise self.commit_failure
                super().commit()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")

            real_connect = sqlite3.connect

            def connect_with_failing_commit(*args: object, **kwargs: object) -> sqlite3.Connection:
                return real_connect(*args, **kwargs, factory=CommitFailConnection)

            from leveraged_trader import workflow as workflow_module

            real_revalidate = workflow_module.revalidate_private_runtime_file
            revalidation_calls = 0

            def fail_cleanup_revalidation(guard: object) -> object:
                nonlocal revalidation_calls
                revalidation_calls += 1
                if revalidation_calls == 3:
                    raise OSError("simulated revalidation failure")
                return real_revalidate(guard)

            original = sqlite3.OperationalError("simulated commit failure")
            with (
                patch(
                    "leveraged_trader.workflow.sqlite3.connect",
                    side_effect=connect_with_failing_commit,
                ),
                patch(
                    "leveraged_trader.workflow.revalidate_private_runtime_file",
                    side_effect=fail_cleanup_revalidation,
                ),
                self.assertRaises(sqlite3.OperationalError) as raised,
                _state_connection(db_path, immediate=True) as conn,
            ):
                self.assertIsInstance(conn, CommitFailConnection)
                conn.commit_failure = original
                conn.execute("INSERT INTO transaction_probe VALUES (1)")
                failing_conn = conn

            self.assertIs(raised.exception, original)
            self.assertEqual(revalidation_calls, 3)
            self.assertIn(
                "Failed runtime-file revalidation after SQLite commit failure: simulated revalidation failure",
                raised.exception.__notes__,
            )
            with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                failing_conn.execute("SELECT 1")

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
            with closing(sqlite3.connect(db_path)) as conn, conn:
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

            def migrate(
                conn: sqlite3.Connection,
                _cfg: AlpacaOrderConfig,
                **_kwargs: object,
            ) -> dict[str, str]:
                conn.execute("UPDATE alpaca_managed_positions SET symbol = 'ECHX', alpaca_asset_id = 'asset-echo'")
                conn.commit()
                return {"SATG": "ECHX"}

            mock_migrate.side_effect = migrate
            result = _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())
            self.assertEqual(result["Status"].tolist(), ["symbol_migrated"])
            self.assertIn("SATG to ECHX", result.loc[0, "Message"])
            self.assertEqual(mock_migrate.call_args.kwargs, {"include_closed": False})
            self.assertEqual(
                mock_reconcile.call_args.kwargs,
                {
                    "migrate_closed_symbols": True,
                    "closed_audit_min_interval_minutes": None,
                },
            )

            mock_migrate.side_effect = None
            mock_migrate.return_value = {}
            result = _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())
            self.assertTrue(result.empty)

    @patch("leveraged_trader.workflow.reconcile_alpaca_managed_positions")
    @patch("leveraged_trader.workflow.migrate_alpaca_managed_position_symbols")
    def test_symbol_migration_failure_is_normalized_as_reconciliation_results(
        self,
        mock_migrate: Mock,
        mock_reconcile: Mock,
    ) -> None:
        mock_migrate.side_effect = RuntimeError("asset lookup unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                save_alpaca_managed_buy_order(
                    conn,
                    workflow="Long",
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

            with self.assertRaises(AlpacaReconciliationError) as raised:
                _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())

        result = raised.exception.results
        self.assertEqual(result["Status"].tolist(), ["error"])
        self.assertEqual(result["Action"].tolist(), ["symbol"])
        self.assertEqual(result["Workflow"].tolist(), ["Long"])
        self.assertIn("asset lookup unavailable", result.loc[0, "Message"])
        mock_reconcile.assert_not_called()

    @patch("leveraged_trader.workflow.reconcile_alpaca_managed_positions")
    @patch("leveraged_trader.workflow.migrate_alpaca_managed_position_symbols")
    def test_symbol_migration_failure_survives_failure_to_load_diagnostic_positions(
        self,
        mock_migrate: Mock,
        mock_reconcile: Mock,
    ) -> None:
        migration_failure = RuntimeError("asset lookup unavailable")
        mock_migrate.side_effect = migration_failure
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                save_alpaca_managed_buy_order(
                    conn,
                    workflow="Long",
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

            with (
                patch(
                    "leveraged_trader.workflow.load_alpaca_managed_positions",
                    side_effect=sqlite3.OperationalError("diagnostic position read failed"),
                ),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())

        result = raised.exception.results
        self.assertIs(raised.exception.__cause__, migration_failure)
        self.assertEqual(result["Status"].tolist(), ["error"])
        self.assertEqual(result["Action"].tolist(), ["symbol"])
        self.assertTrue(pd.isna(result.loc[0, "Position ID"]))
        self.assertIn("asset lookup unavailable", result.loc[0, "Message"])
        self.assertIn("diagnostic position read failed", result.loc[0, "Message"])
        self.assertIn(
            "diagnostic position read failed",
            "\n".join(getattr(raised.exception, "__notes__", ())),
        )
        mock_reconcile.assert_not_called()

    @patch("leveraged_trader.workflow.reconcile_alpaca_managed_positions")
    def test_post_commit_migration_revalidation_base_exception_is_auditable(
        self,
        mock_reconcile: Mock,
    ) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                save_alpaca_managed_buy_order(
                    conn,
                    workflow="Long",
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

            migration_returned = False
            post_migration_revalidations = 0
            revalidation_failure = KeyboardInterrupt("post-commit database identity check interrupted")
            real_revalidate = workflow_module.revalidate_active_sqlite_runtime_file

            def migrate(
                conn: sqlite3.Connection,
                _cfg: AlpacaOrderConfig,
                **_kwargs: object,
            ) -> dict[str, str]:
                nonlocal migration_returned
                conn.execute("UPDATE alpaca_managed_positions SET symbol = 'ECHX', alpaca_asset_id = 'asset-echo'")
                migration_returned = True
                return {"SATG": "ECHX"}

            def interrupt_after_commit(*args: object, **kwargs: object) -> object:
                nonlocal post_migration_revalidations
                if migration_returned:
                    post_migration_revalidations += 1
                    if post_migration_revalidations == 2:
                        raise revalidation_failure
                return real_revalidate(*args, **kwargs)

            with (
                patch.object(
                    workflow_module,
                    "migrate_alpaca_managed_position_symbols",
                    side_effect=migrate,
                ),
                patch.object(
                    workflow_module,
                    "revalidate_active_sqlite_runtime_file",
                    side_effect=interrupt_after_commit,
                ),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())

            with closing(sqlite3.connect(db_path)) as conn:
                persisted_identity = conn.execute(
                    "SELECT symbol, alpaca_asset_id FROM alpaca_managed_positions WHERE id = 1"
                ).fetchone()

        self.assertIs(raised.exception.__cause__, revalidation_failure)
        self.assertEqual(raised.exception.results["Status"].tolist(), ["error"])
        self.assertIn("committed, but post-commit validation failed", raised.exception.results.loc[0, "Message"])
        self.assertIn("post-commit database identity check interrupted", raised.exception.results.loc[0, "Message"])
        self.assertEqual(persisted_identity, ("ECHX", "asset-echo"))
        mock_reconcile.assert_not_called()

    @patch("leveraged_trader.workflow.reconcile_alpaca_managed_positions")
    def test_post_commit_migration_audit_loading_failure_is_auditable(
        self,
        mock_reconcile: Mock,
    ) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                save_alpaca_managed_buy_order(
                    conn,
                    workflow="Long",
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

            def migrate(
                conn: sqlite3.Connection,
                _cfg: AlpacaOrderConfig,
                **_kwargs: object,
            ) -> dict[str, str]:
                conn.execute("UPDATE alpaca_managed_positions SET symbol = 'ECHX', alpaca_asset_id = 'asset-echo'")
                return {"SATG": "ECHX"}

            loader_failure = sqlite3.OperationalError("committed migration audit read failed")
            with (
                patch.object(
                    workflow_module,
                    "migrate_alpaca_managed_position_symbols",
                    side_effect=migrate,
                ),
                patch.object(
                    workflow_module,
                    "load_alpaca_managed_positions",
                    side_effect=loader_failure,
                ),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())

            with closing(sqlite3.connect(db_path)) as conn:
                persisted_identity = conn.execute(
                    "SELECT symbol, alpaca_asset_id FROM alpaca_managed_positions WHERE id = 1"
                ).fetchone()

        self.assertIs(raised.exception.__cause__, loader_failure)
        self.assertEqual(raised.exception.results["Status"].tolist(), ["symbol_migrated", "error"])
        self.assertEqual(raised.exception.results["Action"].tolist(), ["symbol", "audit"])
        self.assertEqual(raised.exception.results.loc[0, "Asset"], "ECHX")
        self.assertIn("committed migration audit read failed", raised.exception.results.loc[1, "Message"])
        self.assertEqual(persisted_identity, ("ECHX", "asset-echo"))
        mock_reconcile.assert_not_called()

    @patch("leveraged_trader.workflow.migrate_alpaca_managed_position_symbols", return_value={})
    @patch("leveraged_trader.workflow.reconcile_alpaca_managed_positions")
    def test_reconciliation_base_exception_after_invocation_is_auditable(
        self,
        mock_reconcile: Mock,
        _mock_migrate: Mock,
    ) -> None:
        reconciliation_failure = KeyboardInterrupt("cancellation interrupted after request transmission")
        mock_reconcile.side_effect = reconciliation_failure

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)

            with self.assertRaises(AlpacaReconciliationError) as raised:
                _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())

        self.assertIs(raised.exception.__cause__, reconciliation_failure)
        self.assertEqual(raised.exception.results["Status"].tolist(), ["error"])
        self.assertEqual(raised.exception.results["Action"].tolist(), ["audit"])
        self.assertIn("broker work may have begun", raised.exception.results.loc[0, "Message"])
        self.assertIn("cancellation interrupted", raised.exception.results.loc[0, "Message"])

    def test_reconciliation_diagnostic_failure_cannot_mask_typed_audit(self) -> None:
        class BrokenDiagnostic(RuntimeError):
            def __str__(self) -> str:
                raise KeyboardInterrupt("diagnostic rendering interrupted")

        reconciliation_failure = BrokenDiagnostic()
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)

            with (
                patch(
                    "leveraged_trader.workflow.migrate_alpaca_managed_position_symbols",
                    return_value={},
                ),
                patch(
                    "leveraged_trader.workflow.reconcile_alpaca_managed_positions",
                    side_effect=reconciliation_failure,
                ),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())

        self.assertIs(raised.exception.__cause__, reconciliation_failure)
        self.assertEqual(raised.exception.results["Status"].tolist(), ["error"])
        self.assertIn("BrokenDiagnostic", raised.exception.results.loc[0, "Message"])

    @patch("leveraged_trader.workflow.reconcile_alpaca_managed_positions")
    @patch("leveraged_trader.workflow.migrate_alpaca_managed_position_symbols")
    def test_symbol_migration_is_committed_before_later_reconciliation_failure(
        self,
        mock_migrate: Mock,
        mock_reconcile: Mock,
    ) -> None:
        failure_results = pd.DataFrame(
            [
                {
                    "Position ID": 1,
                    "Workflow": "Long",
                    "Asset": "ECHX",
                    "Action": "sell",
                    "Status": "error",
                    "Buy Client Order ID": "rsi-buy-SATG-20260618",
                    "Sell Client Order ID": None,
                    "Qty": 2,
                    "Limit Price": 13.42,
                    "Alpaca Order ID": None,
                    "Message": "protective reconciliation failed",
                }
            ]
        )
        mock_reconcile.side_effect = AlpacaReconciliationError("sell failure", failure_results)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                save_alpaca_managed_buy_order(
                    conn,
                    workflow="Long",
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

            def migrate(
                conn: sqlite3.Connection,
                _cfg: AlpacaOrderConfig,
                **_kwargs: object,
            ) -> dict[str, str]:
                conn.execute("UPDATE alpaca_managed_positions SET symbol = 'ECHX', alpaca_asset_id = 'asset-echo'")
                return {"SATG": "ECHX"}

            mock_migrate.side_effect = migrate
            with self.assertRaises(AlpacaReconciliationError) as raised:
                _reconcile_alpaca_managed_positions_for_db(db_path, AlpacaOrderConfig())

            with closing(sqlite3.connect(db_path)) as conn:
                persisted_identity = conn.execute(
                    "SELECT symbol, alpaca_asset_id FROM alpaca_managed_positions WHERE id = 1"
                ).fetchone()

        result = raised.exception.results
        self.assertEqual(result["Status"].tolist(), ["symbol_migrated", "error"])
        self.assertIn("SATG to ECHX", result.loc[0, "Message"])
        self.assertEqual(result.loc[1, "Message"], "protective reconciliation failed")
        self.assertEqual(persisted_identity, ("ECHX", "asset-echo"))

    def test_committed_migration_combines_typed_failure_without_calling_str(self) -> None:
        class BrokenTypedReconciliation(AlpacaReconciliationError):
            str_calls = 0

            def __str__(self) -> str:
                type(self).str_calls += 1
                raise KeyboardInterrupt("typed reconciliation rendering interrupted")

        failure_results = pd.DataFrame(
            [{"Action": "sell", "Status": "error", "Message": "protective reconciliation failed"}]
        )
        reconciliation_failure = BrokenTypedReconciliation("unrenderable", failure_results)
        position = pd.DataFrame(
            [
                {
                    "id": 1,
                    "workflow": "Long",
                    "symbol": "ECHX",
                    "buy_client_order_id": "buy-1",
                    "sell_client_order_id": "sell-1",
                    "filled_qty": 2,
                    "target_sell_price": 13.42,
                    "sell_alpaca_order_id": "sell-order-1",
                }
            ]
        )

        @contextmanager
        def state_connection(*_args: object, **_kwargs: object):
            yield Mock()

        with (
            patch("leveraged_trader.workflow._state_connection", new=state_connection),
            patch("leveraged_trader.workflow.revalidate_active_sqlite_runtime_file"),
            patch(
                "leveraged_trader.workflow.migrate_alpaca_managed_position_symbols",
                return_value={"SATG": "ECHX"},
            ),
            patch("leveraged_trader.workflow.load_alpaca_managed_positions", return_value=position),
            patch(
                "leveraged_trader.workflow.reconcile_alpaca_managed_positions",
                side_effect=reconciliation_failure,
            ),
            self.assertRaises(AlpacaReconciliationError) as raised,
        ):
            _reconcile_alpaca_managed_positions_for_db("unused.sqlite", AlpacaOrderConfig())

        self.assertIs(raised.exception.__cause__, reconciliation_failure)
        self.assertEqual(str(raised.exception), "unrenderable")
        self.assertEqual(BrokenTypedReconciliation.str_calls, 0)
        self.assertEqual(raised.exception.results["Status"].tolist(), ["symbol_migrated", "error"])

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

    def test_invalid_grid_scalar_types_are_rejected_before_filesystem_changes(self) -> None:
        class OverflowingFloat:
            def __float__(self) -> float:
                raise OverflowError("outside float range")

        nested_temporal = np.empty((), dtype=object)
        nested_temporal[()] = np.array(np.datetime64(30, "ns"))
        cases = [
            ("Python boolean", [True], [70.0], [1.5]),
            ("NumPy boolean", [30.0], [np.bool_(True)], [1.5]),
            ("wrapped NumPy boolean", [np.array(True)], [70.0], [1.5]),
            ("Python complex", [30.0], [70.0], [complex(1.5, 1.0)]),
            ("NumPy complex", [np.complex128(30.0 + 2.0j)], [70.0], [1.5]),
            ("wrapped NumPy complex", [30.0], [np.array(70.0 + 2.0j)], [1.5]),
            ("Python date", [date(2026, 1, 2)], [70.0], [1.5]),
            ("Python datetime", [30.0], [datetime(2026, 1, 2, 12)], [1.5]),
            ("Python timedelta", [30.0], [70.0], [timedelta(days=2)]),
            ("NumPy datetime", [np.datetime64(30, "ns")], [70.0], [1.5]),
            ("NumPy timedelta", [30.0], [70.0], [np.timedelta64(2, "ns")]),
            ("wrapped NumPy datetime", [np.array(np.datetime64(30, "ns"))], [70.0], [1.5]),
            ("nested wrapped NumPy datetime", [nested_temporal], [70.0], [1.5]),
            ("overflowing conversion", [30.0], [70.0], [OverflowingFloat()]),
        ]

        for label, buy_values, short_buy_values, target_values in cases:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(label=label):
                db_path = Path(tmp) / "state.sqlite"
                output_dir = Path(tmp) / "reports"

                with self.assertRaisesRegex(ValueError, "must contain only finite numeric values"):
                    asyncio.run(
                        run_resumable_optimizations_async(
                            mode="update",
                            db_path=str(db_path),
                            base_cfg=BacktestConfig(),
                            universe_cfg=UniverseConfig(),
                            buy_rsi_values=buy_values,
                            short_buy_rsi_values=short_buy_values,
                            profit_target_values=target_values,
                            alpaca_cfg=AlpacaOrderConfig(),
                            output_dir=str(output_dir),
                        )
                    )

                self.assertFalse(db_path.exists())
                self.assertFalse(Path(f"{db_path}.lock").exists())
                self.assertFalse(output_dir.exists())

    def test_immediate_state_transactions_serialize_independent_writers(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        second_acquired = threading.Event()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
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

    def test_strategy_session_discards_connection_after_commit_failure(self) -> None:
        class CommitFailConnection(sqlite3.Connection):
            fail_next_commit = False

            def commit(self) -> None:
                if self.fail_next_commit:
                    self.fail_next_commit = False
                    raise sqlite3.OperationalError("simulated commit failure")
                super().commit()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute("CREATE TABLE commit_probe (value INTEGER NOT NULL)")

            real_connect = sqlite3.connect

            def connect_with_failing_commit(*args: object, **kwargs: object) -> sqlite3.Connection:
                return real_connect(*args, **kwargs, factory=CommitFailConnection)

            session = _WorkflowStrategySession(db_path)
            try:
                with patch(
                    "leveraged_trader.workflow.sqlite3.connect",
                    side_effect=connect_with_failing_commit,
                ):
                    failing_conn = session._connection()
                    self.assertIsInstance(failing_conn, CommitFailConnection)
                    failing_conn.fail_next_commit = True
                    cached_history = pd.DataFrame()
                    session.mark_synchronized({"QQQ": cached_history})

                    with (
                        self.assertRaisesRegex(
                            sqlite3.OperationalError,
                            "simulated commit failure",
                        ),
                        session.immediate_transaction() as conn,
                    ):
                        conn.execute("INSERT INTO commit_probe VALUES (1)")

                    self.assertIsNone(session._conn)
                    self.assertEqual(session.presynchronized_symbols({"QQQ": cached_history}), set())
                    with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                        failing_conn.execute("SELECT 1")

                    with session.immediate_transaction() as conn:
                        conn.execute("INSERT INTO commit_probe VALUES (2)")
            finally:
                session.close()

            with closing(sqlite3.connect(db_path)) as conn, conn:
                values = conn.execute("SELECT value FROM commit_probe").fetchall()

            self.assertEqual(values, [(2,)])

    def test_strategy_session_rolls_back_base_exception_from_transaction_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")

            session = _WorkflowStrategySession(db_path)
            try:
                with (
                    self.assertRaisesRegex(KeyboardInterrupt, "simulated interrupt"),
                    session.immediate_transaction() as conn,
                ):
                    conn.execute("INSERT INTO transaction_probe VALUES (1)")
                    raise KeyboardInterrupt("simulated interrupt")

                with session.immediate_transaction() as conn:
                    conn.execute("INSERT INTO transaction_probe VALUES (2)")
            finally:
                session.close()

            with closing(sqlite3.connect(db_path)) as conn, conn:
                values = conn.execute("SELECT value FROM transaction_probe").fetchall()

            self.assertEqual(values, [(2,)])

    def test_strategy_session_preserves_body_exception_when_rollback_fails(self) -> None:
        class RollbackFailConnection(sqlite3.Connection):
            fail_next_rollback = False

            def rollback(self) -> None:
                if self.fail_next_rollback:
                    self.fail_next_rollback = False
                    raise sqlite3.OperationalError("simulated rollback failure")
                super().rollback()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")

            real_connect = sqlite3.connect

            def connect_with_failing_rollback(*args: object, **kwargs: object) -> sqlite3.Connection:
                return real_connect(*args, **kwargs, factory=RollbackFailConnection)

            session = _WorkflowStrategySession(db_path)
            try:
                with patch(
                    "leveraged_trader.workflow.sqlite3.connect",
                    side_effect=connect_with_failing_rollback,
                ):
                    failing_conn = session._connection()
                    self.assertIsInstance(failing_conn, RollbackFailConnection)
                    failing_conn.fail_next_rollback = True
                    cached_history = pd.DataFrame()
                    session.mark_synchronized({"QQQ": cached_history})
                    original = KeyboardInterrupt("original transaction-body failure")

                    with (
                        self.assertRaises(KeyboardInterrupt) as raised,
                        session.immediate_transaction() as conn,
                    ):
                        conn.execute("INSERT INTO transaction_probe VALUES (1)")
                        raise original

                    self.assertIs(raised.exception, original)
                    self.assertIn(
                        "Failed rollback after SQLite transaction-body failure: simulated rollback failure",
                        raised.exception.__notes__,
                    )
                    self.assertIsNone(session._conn)
                    self.assertEqual(session.presynchronized_symbols({"QQQ": cached_history}), set())
                    with self.assertRaisesRegex(sqlite3.ProgrammingError, "closed database"):
                        failing_conn.execute("SELECT 1")

                    with session.immediate_transaction() as conn:
                        conn.execute("INSERT INTO transaction_probe VALUES (2)")
            finally:
                session.close()

            with closing(sqlite3.connect(db_path)) as conn, conn:
                values = conn.execute("SELECT value FROM transaction_probe").fetchall()

            self.assertEqual(values, [(2,)])

    def test_strategy_session_promotes_asset_error_when_rollback_fails(self) -> None:
        class RollbackFailConnection(sqlite3.Connection):
            fail_next_rollback = False

            def rollback(self) -> None:
                if self.fail_next_rollback:
                    self.fail_next_rollback = False
                    raise sqlite3.OperationalError("simulated rollback failure")
                super().rollback()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute("CREATE TABLE transaction_probe (value INTEGER NOT NULL)")

            real_connect = sqlite3.connect

            def connect_with_failing_rollback(*args: object, **kwargs: object) -> sqlite3.Connection:
                return real_connect(*args, **kwargs, factory=RollbackFailConnection)

            session = _WorkflowStrategySession(db_path)
            original = AssetMarketDataError("invalid market input")
            try:
                with patch(
                    "leveraged_trader.workflow.sqlite3.connect",
                    side_effect=connect_with_failing_rollback,
                ):
                    failing_conn = session._connection()
                    self.assertIsInstance(failing_conn, RollbackFailConnection)
                    failing_conn.fail_next_rollback = True
                    with (
                        self.assertRaises(WorkflowStateCleanupError) as raised,
                        session.immediate_transaction() as conn,
                    ):
                        conn.execute("INSERT INTO transaction_probe VALUES (1)")
                        raise original
            finally:
                session.close()

            self.assertIs(raised.exception.__cause__, original)
            self.assertIn(
                "Failed rollback after SQLite transaction-body failure: simulated rollback failure",
                original.__notes__,
            )

    def test_strategy_session_preserves_revalidation_failure_when_close_fails(self) -> None:
        session = _WorkflowStrategySession("unused.sqlite")
        conn = Mock(name="connection")
        data_version_result = Mock()
        data_version_result.fetchone.return_value = (1,)
        conn.execute.side_effect = [None, data_version_result]
        conn.close.side_effect = OSError("simulated close failure")
        session._conn = conn
        session._runtime_guard = Mock(name="runtime_guard")
        session._owner_thread_id = threading.get_ident()
        session._data_version = 1
        cached_history = pd.DataFrame()
        session.mark_synchronized({"QQQ": cached_history})
        original = PermissionError("simulated revalidation failure")

        with (
            patch(
                "leveraged_trader.workflow.revalidate_private_runtime_file",
                side_effect=original,
            ),
            self.assertRaises(PermissionError) as raised,
            session.immediate_transaction(),
        ):
            pass

        self.assertIs(raised.exception, original)
        self.assertIn(
            "Failed connection close after SQLite transaction-body failure: simulated close failure",
            raised.exception.__notes__,
        )
        self.assertIsNone(session._conn)
        self.assertIsNone(session._runtime_guard)
        self.assertIsNone(session._owner_thread_id)
        self.assertIsNone(session._data_version)
        self.assertEqual(session.presynchronized_symbols({"QQQ": cached_history}), set())

    def test_strategy_session_close_clears_state_when_connection_close_fails(self) -> None:
        session = _WorkflowStrategySession("unused.sqlite")
        conn = Mock(name="connection")
        conn.close.side_effect = OSError("simulated close failure")
        session._conn = conn
        session._runtime_guard = Mock(name="runtime_guard")
        session._owner_thread_id = threading.get_ident()
        session._data_version = 7
        cached_history = pd.DataFrame()
        session.mark_synchronized({"QQQ": cached_history})

        with self.assertRaisesRegex(OSError, "simulated close failure"):
            session.close()

        self.assertIsNone(session._conn)
        self.assertIsNone(session._runtime_guard)
        self.assertIsNone(session._owner_thread_id)
        self.assertIsNone(session._data_version)
        self.assertEqual(session.presynchronized_symbols({"QQQ": cached_history}), set())

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
            with closing(sqlite3.connect(db_path)) as conn, conn:
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

    def test_long_and_short_sessions_cannot_confirm_boundary_removal_in_one_run(self) -> None:
        signal_history = self._symbol_history("QQQ", 10.0)
        shortened_signal = signal_history.iloc[1:]
        risk_free_history = self._symbol_history("^IRX", -95.0)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)

            seed_session = _WorkflowStrategySession(
                db_path,
                history_observation_run_id="seed-run",
            )
            try:
                self._process_session_asset(
                    seed_session,
                    db_path,
                    "TQQQ",
                    "QQQ",
                    self._symbol_history("TQQQ"),
                    signal_history,
                    risk_free_history,
                )
            finally:
                seed_session.close()

            for asset_symbol, offset in [("TQQQ", 0.0), ("SQQQ", 20.0)]:
                side_session = _WorkflowStrategySession(
                    db_path,
                    history_observation_run_id="combined-run-1",
                )
                try:
                    self._process_session_asset(
                        side_session,
                        db_path,
                        asset_symbol,
                        "QQQ",
                        self._symbol_history(asset_symbol, offset),
                        shortened_signal.copy(),
                        risk_free_history.copy(),
                    )
                finally:
                    side_session.close()

            with closing(sqlite3.connect(db_path)) as conn, conn:
                stored_signal_count = conn.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'QQQ'").fetchone()[
                    0
                ]
                observations = conn.execute(
                    """
                    SELECT consecutive_observations
                    FROM market_history_removal_candidates
                    WHERE symbol = 'QQQ'
                    """
                ).fetchone()[0]
                states_after_combined_run = conn.execute(
                    "SELECT asset_symbol FROM strategy_state ORDER BY asset_symbol"
                ).fetchall()

            self.assertEqual(stored_signal_count, len(signal_history))
            self.assertEqual(observations, 1)
            self.assertEqual(states_after_combined_run, [("SQQQ",), ("TQQQ",)])

            next_run_session = _WorkflowStrategySession(
                db_path,
                history_observation_run_id="combined-run-2",
            )
            try:
                self._process_session_asset(
                    next_run_session,
                    db_path,
                    "SQQQ",
                    "QQQ",
                    self._symbol_history("SQQQ", 20.0),
                    shortened_signal.copy(),
                    risk_free_history.copy(),
                )
            finally:
                next_run_session.close()

            with closing(sqlite3.connect(db_path)) as conn, conn:
                confirmed_signal_count = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol = 'QQQ'"
                ).fetchone()[0]
                states_after_next_run = conn.execute(
                    "SELECT asset_symbol FROM strategy_state ORDER BY asset_symbol"
                ).fetchall()

            self.assertEqual(confirmed_signal_count, len(shortened_signal))
            self.assertEqual(states_after_next_run, [("SQQQ",)])

    def test_strategy_session_resynchronizes_a_different_history_instance(self) -> None:
        signal_history = self._symbol_history("QQQ", 10.0)
        risk_free_history = self._symbol_history("^IRX", -95.0)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
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
            with closing(sqlite3.connect(db_path)) as conn, conn:
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
                with closing(sqlite3.connect(db_path)) as external, external:
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
            with closing(sqlite3.connect(db_path)) as conn, conn:
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
            with closing(sqlite3.connect(db_path)) as conn, conn:
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
                corrected_signal.loc[
                    corrected_signal.index[10],
                    ["QQQ_High", "QQQ_Close"],
                ] += 5.0
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

            with closing(sqlite3.connect(db_path)) as conn, conn:
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
        validation_active = 0
        max_validation_active = 0
        validation_calls = 0
        validation_counter_lock = threading.Lock()
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
            nonlocal validation_active, max_validation_active, validation_calls
            download_threads.add(threading.current_thread().name)
            with validation_counter_lock:
                validation_calls += 1
                validation_active += 1
                max_validation_active = max(max_validation_active, validation_active)
            time.sleep(0.01)
            with validation_counter_lock:
                validation_active -= 1
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        def fake_load_symbol_history(symbol: str, **_kwargs: object) -> pd.DataFrame:
            download_threads.add(threading.current_thread().name)
            return history(symbol)

        def fake_load_risk_free_history(**_kwargs: object) -> pd.DataFrame:
            download_threads.add(threading.current_thread().name)
            return history("^IRX")

        def fake_process_asset_grid(*_args: object, **_kwargs: object) -> None:
            nonlocal state_active, max_state_active
            self.assertIs(_args[2], _args[3])
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
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                side_effect=fake_load_symbol_history,
            ) as mock_asset,
            patch("leveraged_trader.workflow.load_signal_history", side_effect=fake_load_symbol_history) as mock_signal,
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                side_effect=fake_load_risk_free_history,
            ) as mock_risk_free,
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=fake_process_asset_grid),
        ):
            results = asyncio.run(run())

        self.assertEqual(max_state_active, 1)
        self.assertEqual(max_validation_active, 1)
        self.assertEqual(validation_calls, 2)
        self.assertTrue(download_threads)
        self.assertTrue(all(name.startswith("workflow-download") for name in download_threads))
        self.assertTrue(state_threads)
        self.assertTrue(all(name.startswith("workflow-strategy") for name in state_threads))
        self.assertEqual(mock_asset.call_count, 2)
        self.assertEqual(mock_signal.call_count, 0)
        self.assertEqual(mock_risk_free.call_count, 1)
        self.assertEqual([result.status for result in results], ["done", "done"])

    def test_authoritative_histories_build_asset_calendar_frame_with_metadata(self) -> None:
        asset_dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        asset_history = self._symbol_history("TQQQ").loc[asset_dates].copy()
        signal_history = self._symbol_history("QQQ").loc[asset_dates[[0, 2]]].copy()
        risk_free_history = self._symbol_history("^IRX").iloc[[0]].copy()
        risk_free_history.index = pd.DatetimeIndex(["2026-01-01"])
        asset_history.attrs["market_data_providers"] = {"TQQQ": "tradier"}
        asset_history.attrs["tradier_recovered_symbols"] = ["TQQQ"]
        signal_history.attrs["market_data_providers"] = {"QQQ": "yahoo_finance"}
        risk_free_history.attrs["market_data_providers"] = {"^IRX": "yahoo_finance"}

        data = _strategy_data_from_authoritative_histories(
            asset_symbol="TQQQ",
            signal_symbol="QQQ",
            asset_history=asset_history,
            signal_history=signal_history,
            risk_free_history=risk_free_history,
        )

        self.assertEqual(data.index.tolist(), asset_dates.tolist())
        self.assertTrue(pd.isna(data.loc[asset_dates[1], "QQQ_Close"]))
        self.assertEqual(data["^IRX_Close"].tolist(), [100.0, 100.0, 100.0])
        self.assertEqual(
            data.attrs["market_data_providers"],
            {"TQQQ": "tradier", "QQQ": "yahoo_finance", "^IRX": "yahoo_finance"},
        )
        self.assertEqual(data.attrs["tradier_recovered_symbols"], ["TQQQ"])

    def test_startup_preserves_reconciliation_error_when_failure_rendering_fails(self) -> None:
        from leveraged_trader import workflow as workflow_module

        reconciliation_results = pd.DataFrame(
            [
                {
                    "Position ID": 1,
                    "Workflow": "Long",
                    "Asset": "TQQQ",
                    "Action": "sell",
                    "Status": "error",
                    "Message": "protective order failed",
                }
            ]
        )
        reconciliation_failure = AlpacaReconciliationError(
            "protective reconciliation failed",
            reconciliation_results,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch.object(workflow_module, "_initialize_state_db"),
                patch.object(
                    workflow_module,
                    "_reconcile_alpaca_managed_positions_for_db",
                    side_effect=reconciliation_failure,
                ),
                patch.object(workflow_module, "_persist_alpaca_reconciliation_snapshot"),
                patch.object(
                    reporter,
                    "reconciliation",
                    side_effect=BrokenPipeError("failure report pipe closed"),
                ),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                asyncio.run(
                    workflow_module._run_resumable_optimizations_unlocked(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(output_dir),
                        reporter=reporter,
                    )
                )

        self.assertIs(raised.exception, reconciliation_failure)
        self.assertIn(
            "Failed to publish the failed reconciliation audit: failure report pipe closed",
            "\n".join(getattr(raised.exception, "__notes__", ())),
        )

    def test_startup_preserves_structured_reconciliation_error_when_progress_teardown_fails(self) -> None:
        from leveraged_trader import workflow as workflow_module

        reconciliation_results = pd.DataFrame(
            [{"Action": "sell", "Status": "error", "Asset": "TQQQ", "Message": "broker failure"}]
        )
        reconciliation_error = AlpacaReconciliationError("reconciliation failed", reconciliation_results)
        structured_results = reconciliation_error.results

        @contextmanager
        def failing_progress(*_args: object, **_kwargs: object):
            try:
                yield Mock()
            finally:
                raise OSError("progress teardown failed")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch.object(
                    workflow_module,
                    "_reconcile_alpaca_managed_positions_for_db",
                    side_effect=reconciliation_error,
                ),
                patch.object(reporter, "step_progress", new=failing_progress),
                self.assertRaises(AlpacaReconciliationError) as raised,
            ):
                asyncio.run(
                    workflow_module._run_resumable_optimizations_unlocked(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(output_dir),
                        reporter=reporter,
                    )
                )

        self.assertIs(raised.exception, reconciliation_error)
        self.assertIs(raised.exception.results, structured_results)
        self.assertIn(
            "progress teardown failed",
            "\n".join(getattr(raised.exception, "__notes__", ())),
        )

    def test_startup_cancellation_leaves_preinvalidated_reconciliation_manifest(self) -> None:
        from leveraged_trader import workflow as workflow_module

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir(mode=0o700)
            old_publication = workflow_module._begin_alpaca_snapshot_publication(
                output_dir,
                snapshot_kind="reconciliation",
            )
            for filename in old_publication.expected_filenames:
                workflow_module._publish_alpaca_snapshot_csv(
                    old_publication,
                    pd.DataFrame([{"Generation": "old"}]),
                    filename=filename,
                    index=False,
                )
            workflow_module._commit_alpaca_snapshot_publication(old_publication)

            real_initialize = workflow_module._initialize_state_db

            def initialize_after_observing_invalidation(db_path: str) -> None:
                manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
                self.assertEqual(set(manifest["Status"]), {"publishing"})
                self.assertNotEqual(set(manifest["Generation"]), {old_publication.generation})
                real_initialize(db_path)

            def cancel_after_observing_invalidation(*_args: object) -> pd.DataFrame:
                manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
                self.assertEqual(set(manifest["Status"]), {"publishing"})
                self.assertNotEqual(set(manifest["Generation"]), {old_publication.generation})
                raise asyncio.CancelledError("simulated startup cancellation")

            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch.object(
                    workflow_module,
                    "_initialize_state_db",
                    side_effect=initialize_after_observing_invalidation,
                ),
                patch.object(
                    workflow_module,
                    "_reconcile_alpaca_managed_positions_for_db",
                    side_effect=cancel_after_observing_invalidation,
                ),
                self.assertRaisesRegex(asyncio.CancelledError, "simulated startup cancellation"),
            ):
                asyncio.run(
                    workflow_module._run_resumable_optimizations_unlocked(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(output_dir),
                        reporter=reporter,
                    )
                )

            manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
            self.assertEqual(set(manifest["Status"]), {"publishing"})
            with self.assertRaisesRegex(OSError, "not committed"):
                validate_alpaca_snapshot(output_dir)

    def test_startup_serializes_reconciliation_before_universe_writes(self) -> None:
        events: list[str] = []

        async def immediate_run_blocking(
            _executor: object,
            func: object,
            /,
            *args: object,
            **kwargs: object,
        ) -> object:
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
                patch("leveraged_trader.workflow._run_blocking", new=immediate_run_blocking),
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
                patch("leveraged_trader.workflow._load_alpaca_realized_pnl_for_db", return_value=pd.DataFrame()),
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
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            with (
                patch("leveraged_trader.workflow.strategy_state_matches_config", return_value=True),
                patch(
                    "leveraged_trader.workflow.load_best_strategy_summary",
                    return_value={"buy_rsi": 30.0, "profit_target_multiple": 1.5},
                ),
                patch(
                    "leveraged_trader.workflow.load_complete_strategy_equity_curve",
                    return_value=pd.DataFrame({"equity": [100_000.0]}),
                ),
            ):
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
        self.assertEqual(plan.preverified_best_config, (30.0, 1.5))

    def test_report_build_excludes_assets_not_processed_in_current_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
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

    def test_report_build_requires_both_grid_dimensions(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(
                ValueError,
                "must either both be provided",
            ),
        ):
            _build_reports_for_db(
                str(Path(tmp) / "state.sqlite"),
                workflow_assets,
                BacktestConfig(),
                processed_asset_pairs=set(),
                buy_rsi_values=[30.0],
            )

    def test_report_build_rejects_authenticated_extra_grid_pair_before_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            asset_history = self._symbol_history("TQQQ")
            signal_history = self._symbol_history("QQQ")
            risk_free_history = self._symbol_history("^IRX", -95.0)
            strategy_data = _strategy_data_from_authoritative_histories(
                asset_symbol="TQQQ",
                signal_symbol="QQQ",
                asset_history=asset_history,
                signal_history=signal_history,
                risk_free_history=risk_free_history,
            )
            base_cfg = BacktestConfig()
            _process_asset_grid_for_db(
                db_path,
                strategy_data,
                asset_history,
                signal_history,
                risk_free_history,
                base_cfg,
                "TQQQ",
                "QQQ",
                [20.0, 30.0],
                [2.0],
                True,
            )
            with closing(sqlite3.connect(db_path)) as conn, conn:
                conn.execute(
                    "UPDATE strategy_config SET fingerprint = ?",
                    (strategy_config_fingerprint(base_cfg, [30.0], [2.0]),),
                )
            workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])

            with (
                patch("leveraged_trader.workflow.summarize_saved_results") as summarize,
                self.assertRaisesRegex(
                    _CompletedWorkflowStateInvalidError,
                    "requested strategy grid",
                ) as raised,
            ):
                _build_reports_for_db(
                    db_path,
                    workflow_assets,
                    base_cfg,
                    processed_asset_pairs={("TQQQ", "QQQ")},
                    buy_rsi_values=[30.0],
                    profit_target_values=[2.0],
                )

            summarize.assert_not_called()
            self.assertEqual(raised.exception.pairs, {("TQQQ", "QQQ")})

    def test_report_build_reads_one_committed_sqlite_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                init_state_db(conn)
                conn.execute(
                    """
                    INSERT INTO market_data
                    (symbol, date, open, high, low, close, volume)
                    VALUES ('SNAPSHOT', '2026-01-02', 100, 100, 100, 100, 1)
                    """
                )
                conn.commit()

            observed_closes: list[float] = []

            def summarize_during_concurrent_commit(
                reader: sqlite3.Connection,
                _workflow_assets: pd.DataFrame,
                **_kwargs: object,
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                observed_closes.append(
                    float(reader.execute("SELECT close FROM market_data WHERE symbol = 'SNAPSHOT'").fetchone()[0])
                )
                with closing(sqlite3.connect(db_path)) as writer:
                    writer.execute("UPDATE market_data SET close = 200 WHERE symbol = 'SNAPSHOT'")
                    writer.commit()
                observed_closes.append(
                    float(reader.execute("SELECT close FROM market_data WHERE symbol = 'SNAPSHOT'").fetchone()[0])
                )
                return pd.DataFrame(), pd.DataFrame()

            workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
            with patch(
                "leveraged_trader.workflow.summarize_saved_results",
                side_effect=summarize_during_concurrent_commit,
            ):
                _build_reports_for_db(
                    db_path,
                    workflow_assets,
                    BacktestConfig(),
                    processed_asset_pairs=set(),
                )

            with closing(sqlite3.connect(db_path)) as conn:
                committed_close = float(
                    conn.execute("SELECT close FROM market_data WHERE symbol = 'SNAPSHOT'").fetchone()[0]
                )

        self.assertEqual(observed_closes, [100.0, 100.0])
        self.assertEqual(committed_close, 200.0)

    def test_report_build_rejects_completed_pair_whose_state_disappeared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])

            with (
                patch(
                    "leveraged_trader.workflow.strategy_state_matches_config",
                    return_value=True,
                ),
                self.assertRaisesRegex(
                    WorkflowRunError,
                    "Completed workflow state disappeared.*TQQQ/QQQ",
                ),
            ):
                _build_reports_for_db(
                    db_path,
                    workflow_assets,
                    BacktestConfig(),
                    processed_asset_pairs={("TQQQ", "QQQ")},
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                )

    def test_report_build_adds_workflow_and_side_prefixed_curve_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)

            dates = pd.date_range("2026-01-02", periods=20, freq="B")
            asset_close = np.full(len(dates), 100.0)
            asset_high = asset_close.copy()
            asset_high[15:] = 151.0
            asset_history = pd.DataFrame(
                {
                    "SQQQ_Open": asset_close,
                    "SQQQ_High": asset_high,
                    "SQQQ_Low": np.full(len(dates), 99.0),
                    "SQQQ_Close": asset_close,
                    "SQQQ_Volume": np.full(len(dates), 1_000_000.0),
                },
                index=dates,
            )
            signal_history = self._symbol_history("QQQ")
            risk_free_history = self._symbol_history("^IRX", -95.0)
            strategy_data = _strategy_data_from_authoritative_histories(
                asset_symbol="SQQQ",
                signal_symbol="QQQ",
                asset_history=asset_history,
                signal_history=signal_history,
                risk_free_history=risk_free_history,
            )
            base_cfg = BacktestConfig()
            _process_asset_grid_for_db(
                db_path,
                strategy_data,
                asset_history,
                signal_history,
                risk_free_history,
                base_cfg,
                "SQQQ",
                "QQQ",
                [70.0, 30.0],
                [1.5],
                True,
                rsi_entry_rule="upper",
            )

            workflow_assets = pd.DataFrame([{"symbol": "SQQQ", "name": "S", "rsi_symbol": "QQQ"}])

            optimization_summary, curves, buy_signals, _eligible, _sell_signals, pnl = _build_reports_for_db(
                db_path,
                workflow_assets,
                base_cfg,
                processed_asset_pairs={("SQQQ", "QQQ")},
                buy_rsi_values=[70.0, 30.0],
                profit_target_values=[1.5],
                workflow_label="Short",
            )

        self.assertEqual(optimization_summary["Workflow"].tolist(), ["Short"])
        self.assertEqual(optimization_summary["Buy RSI"].tolist(), [70.0])
        self.assertEqual(curves.columns.tolist(), ["Short_SQQQ_RSI_Strategy"])
        self.assertEqual(buy_signals["Workflow"].tolist(), ["Short"])
        self.assertIn("Workflow", pnl.columns)

    def test_workflow_reconciles_again_after_observing_a_buy_fill(self) -> None:
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

        async def immediate_run_blocking(
            _executor: object,
            func: object,
            /,
            *args: object,
            **kwargs: object,
        ) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            output_dir = Path(tmp) / "outputs"
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow._run_blocking", new=immediate_run_blocking),
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
                    return_value=pd.DataFrame([{"Status": "partially_filled"}]),
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(),
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_realized_pnl_for_db",
                    return_value=pd.DataFrame([{"Workflow": "Long", "Closed Positions": 1}]),
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
                        alpaca_cfg=AlpacaOrderConfig(
                            enabled=True,
                            api_key_id="key",
                            api_secret_key="secret",
                        ),
                        output_dir=str(output_dir),
                        reporter=reporter,
                    )
                )
            snapshot = load_alpaca_snapshot(output_dir)

        self.assertEqual(mock_reconcile.call_count, 2)
        self.assertEqual(snapshot.snapshot_kind, "workflow")
        self.assertEqual(snapshot.read_csv("alpaca_order_results.csv")["Status"].tolist(), ["partially_filled"])
        self.assertTrue(mock_write_outputs.call_args.kwargs["broker_snapshot_committed"])
        reconciliation_results = mock_write_outputs.call_args.kwargs["reconciliation_results"]
        self.assertEqual(reconciliation_results["Action"].tolist(), ["sell"])
        realized_pnl = mock_write_outputs.call_args.kwargs["realized_pnl_summary"]
        self.assertEqual(realized_pnl["Closed Positions"].tolist(), [1])

    def test_workflow_commits_empty_buy_snapshot_before_final_order_render_failure(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
        initial_reconciliation = pd.DataFrame(
            [
                {
                    "Position ID": 7,
                    "Workflow": "Long",
                    "Asset": "TQQQ",
                    "Action": "sell",
                    "Status": "accepted",
                    "Buy Client Order ID": "rsi-buy-TQQQ-20260102",
                    "Sell Client Order ID": "rsi-exit-TQQQ-7",
                    "Qty": 2,
                    "Limit Price": 150,
                    "Alpaca Order ID": "sell-protective",
                    "Message": "protective sell remains active",
                }
            ]
        )

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

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            output_dir = Path(tmp) / "outputs"
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=initial_reconciliation,
                ) as mock_reconcile,
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                    return_value=workflow_assets,
                ),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=fake_asset_pipeline),
                patch(
                    "leveraged_trader.workflow._build_reports_for_db",
                    return_value=(pd.DataFrame(),) * 6,
                ),
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    return_value=pd.DataFrame(),
                ),
                patch.object(
                    reporter,
                    "order_results",
                    side_effect=OSError("final order rendering failed"),
                ),
                self.assertRaisesRegex(OSError, "final order rendering failed"),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=db_path,
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(
                            enabled=True,
                            api_key_id="key",
                            api_secret_key="secret",
                        ),
                        output_dir=str(output_dir),
                        reporter=reporter,
                    )
                )

            snapshot = load_alpaca_snapshot(output_dir)
            protective_audit = snapshot.read_csv("alpaca_reconciliation_results.csv")
            manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")

        self.assertEqual(mock_reconcile.call_count, 1)
        self.assertEqual(snapshot.snapshot_kind, "workflow")
        self.assertEqual(protective_audit["Status"].tolist(), ["accepted"])
        self.assertEqual(protective_audit["Alpaca Order ID"].tolist(), ["sell-protective"])
        self.assertEqual(set(manifest["Status"]), {"committed"})

    def test_workflow_persists_submitted_buy_results_before_post_buy_reconciliation_failure(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
        initial_reconciliation = pd.DataFrame(columns=["Action"])
        failed_reconciliation = pd.DataFrame([{"Action": "sell", "Status": "error", "Message": "protection failed"}])
        order_results = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "Date": "2026-01-02",
                    "Status": "partially_filled",
                    "Alpaca Order ID": "buy-current",
                }
            ]
        )

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

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    side_effect=[
                        initial_reconciliation,
                        AlpacaReconciliationError("post-buy failed", failed_reconciliation),
                    ],
                ),
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
                    return_value=order_results,
                ),
                patch("leveraged_trader.workflow._persist_alpaca_reconciliation_failure"),
                self.assertRaises(AlpacaReconciliationError),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(
                            enabled=True,
                            api_key_id="key",
                            api_secret_key="secret",
                        ),
                        output_dir=str(output_dir),
                        reporter=reporter,
                    )
                )

            persisted = pd.read_csv(output_dir / "alpaca_order_results.csv")
            snapshot = load_alpaca_snapshot(output_dir)

        self.assertEqual(persisted["Alpaca Order ID"].tolist(), ["buy-current"])
        self.assertEqual(persisted["Status"].tolist(), ["partially_filled"])
        self.assertEqual(snapshot.snapshot_kind, "workflow")
        self.assertIn("alpaca_order_results.csv", snapshot.filenames)
        self.assertEqual(
            snapshot.read_csv("alpaca_order_results.csv")["Alpaca Order ID"].tolist(),
            ["buy-current"],
        )

    def test_workflow_persists_buy_results_before_post_submit_failures(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
        order_results = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "Date": "2026-01-02",
                    "Status": "submitted",
                    "Alpaca Order ID": "buy-current",
                }
            ]
        )

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

        def reports_for_side(
            _db_path: str,
            _workflow_assets: pd.DataFrame,
            _base_cfg: BacktestConfig,
            _processed_asset_pairs: set[tuple[str, str]],
            workflow_label: str | None = None,
            _rsi_entry_rule: str | None = None,
            _buy_rsi_values: list[float] | None = None,
            _profit_target_values: list[float] | None = None,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            if workflow_label != "Long":
                return (pd.DataFrame(),) * 6
            buy_signals = pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02"}])
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                buy_signals,
                buy_signals.copy(),
                pd.DataFrame(),
                pd.DataFrame(),
            )

        cases = (
            ("successful-submit-status", False, False, True, False, False, False, False, False),
            ("successful-submit-order-csv", False, False, False, False, True, False, False, False),
            ("batch-failure-status", True, False, True, False, False, False, False, False),
            ("batch-failure-state-snapshot", True, False, False, True, False, False, False, False),
            (
                "broker-visible-batch-and-reconciliation-failure",
                True,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
            ),
            ("ordinary-worker-failure", False, False, False, False, False, True, False, False),
            ("ordinary-worker-failure-status", False, False, True, False, False, True, False, False),
            (
                "ordinary-worker-diagnostic-failure",
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
            ),
            (
                "ordinary-worker-and-reconciliation-failure",
                False,
                False,
                True,
                False,
                False,
                True,
                True,
                False,
            ),
        )
        for (
            case,
            batch_failure,
            batch_broker_side_effects_possible,
            status_teardown_failure,
            state_snapshot_failure,
            order_csv_failure,
            worker_failure,
            reconciliation_failure,
            diagnostic_failure,
        ) in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp) / "outputs"
                reporter = WorkflowReporter(
                    console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
                )
                original_status = reporter.status

                @contextmanager
                def status_with_failing_teardown(
                    message: str,
                    original_status=original_status,
                    status_teardown_failure: bool = status_teardown_failure,
                ):
                    try:
                        with original_status(message):
                            yield
                    finally:
                        if status_teardown_failure and message == "Preparing Alpaca order results":
                            raise OSError("status teardown failed")

                def submit_orders(
                    *_args: object,
                    batch_failure: bool = batch_failure,
                    batch_broker_side_effects_possible: bool = batch_broker_side_effects_possible,
                    worker_failure: bool = worker_failure,
                    **_kwargs: object,
                ) -> pd.DataFrame:
                    if worker_failure:
                        raise sqlite3.OperationalError("submission worker finalization failed after POST")
                    if batch_failure:
                        failed_results = order_results.copy()
                        failed_results["Status"] = "error"
                        failed_results["Alpaca Order ID"] = None
                        raise AlpacaBuyBatchError(
                            "batch failed",
                            failed_results,
                            broker_side_effects_possible=batch_broker_side_effects_possible,
                        )
                    return order_results

                def persist_current_state(
                    *,
                    state_snapshot_failure: bool = state_snapshot_failure,
                    output_dir: Path = output_dir,
                    **kwargs: object,
                ) -> None:
                    if state_snapshot_failure and (output_dir / "alpaca_order_results.csv").is_file():
                        raise OSError("state snapshot failed")
                    publication = kwargs.get("publication")
                    for filename in (
                        "managed_positions.csv",
                        "alpaca_inactive_holdings.csv",
                        "alpaca_realized_pnl.csv",
                    ):
                        digest = _atomic_to_csv(pd.DataFrame(), output_dir / filename, index=False)
                        if publication is not None:
                            publication.digests[filename] = digest  # type: ignore[union-attr]

                def write_output(
                    frame: pd.DataFrame,
                    destination: Path,
                    *,
                    index: bool = True,
                    order_csv_failure: bool = order_csv_failure,
                ) -> str:
                    if order_csv_failure and destination.name == "alpaca_order_results.csv":
                        raise OSError("order CSV write failed")
                    return _atomic_to_csv(frame, destination, index=index)

                def unknown_results(
                    buy_signals: pd.DataFrame,
                    failure: BaseException,
                    *,
                    alpaca_cfg: AlpacaOrderConfig | None = None,
                    diagnostic_failure: bool = diagnostic_failure,
                ) -> pd.DataFrame:
                    if diagnostic_failure:
                        raise RuntimeError("diagnostic construction failed")
                    return _unknown_alpaca_buy_submission_results(
                        buy_signals,
                        failure,
                        alpaca_cfg=alpaca_cfg,
                    )

                initial_reconciliation = pd.DataFrame(columns=["Action"])
                credential = f"workflowCredential{case.replace('-', '')}7zQ42mN"
                failed_reconciliation = pd.DataFrame(
                    [{"Action": "sell", "Status": "error", "Message": "protection failed"}]
                )
                reconciliation_effects: list[pd.DataFrame | BaseException] = [
                    initial_reconciliation,
                    (
                        AlpacaReconciliationError(
                            f"protective reconciliation failed: {credential}",
                            failed_reconciliation,
                        )
                        if reconciliation_failure
                        else initial_reconciliation
                    ),
                ]
                if reconciliation_failure:
                    expected_exception = AlpacaReconciliationError
                elif batch_failure:
                    expected_exception = AlpacaBuyBatchError
                elif worker_failure:
                    expected_exception = sqlite3.OperationalError
                else:
                    expected_exception = OSError
                with (
                    patch(
                        "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                        side_effect=reconciliation_effects,
                    ) as reconcile,
                    patch(
                        "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                        return_value=workflow_assets,
                    ),
                    patch("leveraged_trader.workflow._run_asset_pipeline", new=completed_asset_pipeline),
                    patch("leveraged_trader.workflow._build_reports_for_db", side_effect=reports_for_side),
                    patch(
                        "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                        side_effect=submit_orders,
                    ),
                    patch(
                        "leveraged_trader.workflow._persist_current_alpaca_state",
                        side_effect=persist_current_state,
                    ),
                    patch(
                        "leveraged_trader.workflow._unknown_alpaca_buy_submission_results",
                        side_effect=unknown_results,
                    ),
                    patch("leveraged_trader.workflow._atomic_to_csv", side_effect=write_output),
                    patch.object(reporter, "status", new=status_with_failing_teardown),
                    self.assertRaises(expected_exception) as raised,
                ):
                    asyncio.run(
                        run_resumable_optimizations_async(
                            mode="update",
                            db_path=str(Path(tmp) / "state.sqlite"),
                            base_cfg=BacktestConfig(),
                            universe_cfg=UniverseConfig(),
                            buy_rsi_values=[30.0],
                            profit_target_values=[1.5],
                            alpaca_cfg=AlpacaOrderConfig(
                                enabled=True,
                                api_key_id="key",
                                api_secret_key=credential,
                            ),
                            output_dir=str(output_dir),
                            reporter=reporter,
                        )
                    )

                if order_csv_failure:
                    self.assertFalse((output_dir / "alpaca_order_results.csv").exists())
                else:
                    persisted = pd.read_csv(output_dir / "alpaca_order_results.csv")
                    if worker_failure:
                        self.assertEqual(persisted["Status"].tolist(), ["submission_unknown"])
                        if diagnostic_failure:
                            self.assertNotIn("Asset", persisted.columns)
                            self.assertIn("could not be constructed", persisted.loc[0, "Message"])
                        else:
                            self.assertEqual(persisted["Asset"].tolist(), ["TQQQ"])
                            self.assertIn("submission worker finalization failed", persisted.loc[0, "Message"])
                    elif batch_failure:
                        self.assertEqual(persisted["Status"].tolist(), ["error"])
                        self.assertTrue(pd.isna(persisted.loc[0, "Alpaca Order ID"]))
                    else:
                        self.assertEqual(persisted["Alpaca Order ID"].tolist(), ["buy-current"])
                        self.assertEqual(persisted["Status"].tolist(), ["submitted"])
                if not order_csv_failure and not state_snapshot_failure:
                    failed_snapshot = load_alpaca_snapshot(output_dir)
                    self.assertEqual(failed_snapshot.snapshot_kind, "workflow")
                    self.assertEqual(
                        failed_snapshot.read_csv("alpaca_order_results.csv")["Status"].tolist(),
                        persisted["Status"].tolist(),
                    )
                self.assertEqual(
                    reconcile.call_count,
                    2 if not batch_failure or batch_broker_side_effects_possible else 1,
                )
                if batch_failure and not state_snapshot_failure:
                    self.assertRegex(validate_alpaca_snapshot(output_dir), r"^[0-9a-f]{32}$")
                    manifest = pd.read_csv(output_dir / "alpaca_snapshot_manifest.csv")
                    self.assertEqual(set(manifest["Snapshot Kind"]), {"workflow"})
                    self.assertEqual(
                        tuple(manifest["Filename"]),
                        (
                            "alpaca_order_results.csv",
                            "alpaca_reconciliation_results.csv",
                            "alpaca_sell_order_results.csv",
                            "managed_positions.csv",
                            "alpaca_inactive_holdings.csv",
                            "alpaca_realized_pnl.csv",
                        ),
                    )
                if reconciliation_failure:
                    self.assertIn("protective reconciliation failed", str(raised.exception))
                    rendered_traceback = "".join(traceback.format_exception(raised.exception))
                    self.assertNotIn(credential, str(raised.exception))
                    self.assertNotIn(credential, raised.exception.results.to_string())
                    self.assertNotIn(credential, rendered_traceback)
                    self.assertIsNone(raised.exception.__cause__)
                    if worker_failure:
                        self.assertTrue(
                            any("submission worker finalization failed" in note for note in raised.exception.__notes__),
                            raised.exception.__notes__,
                        )
                    if status_teardown_failure:
                        self.assertTrue(
                            any("status teardown failed" in note for note in raised.exception.__notes__),
                            raised.exception.__notes__,
                        )
                elif batch_failure:
                    expected_note = "state snapshot failed" if state_snapshot_failure else "status teardown failed"
                    self.assertTrue(
                        any(expected_note in note for note in raised.exception.__notes__),
                        raised.exception.__notes__,
                    )
                elif worker_failure:
                    self.assertIn("submission worker finalization failed", str(raised.exception))
                    if diagnostic_failure:
                        self.assertTrue(
                            any("diagnostic construction failed" in note for note in raised.exception.__notes__),
                            raised.exception.__notes__,
                        )
                    if status_teardown_failure:
                        self.assertTrue(
                            any("status teardown failed" in note for note in raised.exception.__notes__),
                            raised.exception.__notes__,
                        )
                else:
                    expected_message = "order CSV write failed" if order_csv_failure else "status teardown failed"
                    self.assertIn(expected_message, str(raised.exception))

    def test_buy_batch_existing_broker_order_failure_reconciles_and_publishes_state(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
        optimization_summary = pd.DataFrame([{"Asset": "TQQQ", "Generation": "current"}])
        curves = pd.DataFrame(
            {"TQQQ_RSI_Strategy": [101_000.0]},
            index=pd.to_datetime(["2026-01-02"]),
        )
        buy_signals = pd.DataFrame([{"Asset": "TQQQ", "Date": "2026-01-02", "Generation": "current"}])
        eligible_buy_signals = buy_signals.copy()
        sell_signals = pd.DataFrame([{"Asset": "TQQQ", "Generation": "current"}])
        order_results = pd.DataFrame(
            [
                {
                    "Asset": "TQQQ",
                    "Date": "2026-01-02",
                    "Status": "error",
                    "Message": "verified broker buy could not be persisted locally",
                }
            ]
        )

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

        def fake_build_reports(
            _db_path: str,
            _workflow_assets: pd.DataFrame,
            _base_cfg: BacktestConfig,
            _processed_asset_pairs: set[tuple[str, str]],
            workflow_label: str | None = None,
            _rsi_entry_rule: str | None = None,
            _buy_rsi_values: list[float] | None = None,
            _profit_target_values: list[float] | None = None,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            if workflow_label == "Long":
                return (
                    optimization_summary,
                    curves,
                    buy_signals,
                    eligible_buy_signals,
                    sell_signals,
                    pd.DataFrame(),
                )
            return (pd.DataFrame(),) * 6

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir()
            for filename in _RESEARCH_REPORT_FILENAMES:
                (output_dir / filename).write_text("Generation\nstale\n", encoding="utf-8")
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ) as mock_reconcile,
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                    return_value=workflow_assets,
                ),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=completed_asset_pipeline),
                patch(
                    "leveraged_trader.workflow._build_reports_for_db",
                    side_effect=fake_build_reports,
                ),
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    side_effect=AlpacaBuyBatchError(
                        "batch failed",
                        order_results,
                        broker_side_effects_possible=True,
                    ),
                ),
                self.assertRaises(AlpacaBuyBatchError),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=db_path,
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(
                            enabled=True,
                            api_key_id="key",
                            api_secret_key="secret",
                        ),
                        output_dir=str(output_dir),
                        reporter=reporter,
                    )
                )

            self.assertEqual(pd.read_csv(output_dir / "alpaca_order_results.csv")["Status"].tolist(), ["error"])
            snapshot = load_alpaca_snapshot(output_dir)
            self.assertEqual(snapshot.snapshot_kind, "workflow")
            self.assertEqual(snapshot.read_csv("alpaca_order_results.csv")["Status"].tolist(), ["error"])
            self.assertEqual(mock_reconcile.call_count, 2)
            self.assertTrue((output_dir / "managed_positions.csv").is_file())
            self.assertTrue((output_dir / "alpaca_realized_pnl.csv").is_file())
            self.assertEqual(
                pd.read_csv(output_dir / "optimization_summary.csv")["Generation"].tolist(),
                ["current"],
            )
            self.assertEqual(
                pd.read_csv(output_dir / "buy_signals.csv")["Generation"].tolist(),
                ["current"],
            )
            for filename in _RESEARCH_REPORT_FILENAMES:
                self.assertTrue((output_dir / filename).is_file())

    def test_workflow_runs_long_then_short_and_submits_combined_buy_signals(self) -> None:
        workflow_asset_groups = {
            "long": pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}]),
            "short": pd.DataFrame([{"symbol": "SQQQ", "name": "S", "rsi_symbol": "QQQ"}]),
        }
        pipeline_rules: list[str] = []
        pipeline_observation_ids: list[str] = []
        pipeline_market_data_sessions: list[object] = []
        submitted_buy_signals: list[pd.DataFrame] = []

        async def fake_asset_pipeline(**kwargs: object) -> list[AssetRunResult]:
            pipeline_rules.append(str(kwargs["rsi_entry_rule"]))
            pipeline_observation_ids.append(str(kwargs["history_observation_run_id"]))
            pipeline_market_data_sessions.append(kwargs["market_data_session"])
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
            _rsi_entry_rule: str | None = None,
            _buy_rsi_values: list[float] | None = None,
            _profit_target_values: list[float] | None = None,
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

        async def immediate_run_blocking(
            _executor: object,
            func: object,
            /,
            *args: object,
            **kwargs: object,
        ) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow._run_blocking", new=immediate_run_blocking),
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
                patch("leveraged_trader.workflow._load_alpaca_realized_pnl_for_db", return_value=pd.DataFrame()),
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
        self.assertEqual(len(set(pipeline_observation_ids)), 1)
        self.assertIs(pipeline_market_data_sessions[0], pipeline_market_data_sessions[1])
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

    def test_workflow_rebuilds_disappeared_pair_then_rechecks_all_reports(self) -> None:
        pair = ("BLSG", "BLSH")
        workflow_assets = pd.DataFrame([{"symbol": pair[0], "name": pair[0], "rsi_symbol": pair[1]}])
        pipeline_calls: list[tuple[str, list[tuple[str, str]], object]] = []
        report_sides: list[str | None] = []

        async def fake_pipeline(**kwargs: object) -> list[AssetRunResult]:
            jobs = kwargs["jobs"]
            pipeline_calls.append(
                (
                    str(kwargs["mode"]),
                    [(job.asset_symbol, job.signal_symbol) for job in jobs],
                    kwargs["market_data_session"],
                )
            )
            return [
                AssetRunResult(
                    workflow_idx=job.workflow_idx,
                    asset_symbol=job.asset_symbol,
                    signal_symbol=job.signal_symbol,
                    action="Rebuilding" if kwargs["mode"] == "rebuild" else "Updating",
                    rows_processed=1,
                    status="done",
                    message="Processed 1 row",
                    workflow=job.workflow,
                )
                for job in jobs
            ]

        def fake_reports(
            _db_path: str,
            _workflow_assets: pd.DataFrame,
            _base_cfg: BacktestConfig,
            _processed_asset_pairs: set[tuple[str, str]],
            workflow_label: str | None = None,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            report_sides.append(workflow_label)
            if workflow_label == "Long" and report_sides.count("Long") == 1:
                raise _CompletedWorkflowStateInvalidError("Completed workflow state disappeared", {pair})
            return (pd.DataFrame(),) * 6

        with tempfile.TemporaryDirectory() as tmp:
            report_output = io.StringIO()
            reporter = WorkflowReporter(
                console=Console(file=report_output, width=100, color_system=None, no_color=True)
            )
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ),
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db", return_value=workflow_assets
                ),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=fake_pipeline),
                patch("leveraged_trader.workflow._build_reports_for_db", side_effect=fake_reports),
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    return_value=pd.DataFrame(columns=["Status"]),
                ) as submit,
                patch("leveraged_trader.workflow._load_alpaca_managed_positions_for_db", return_value=pd.DataFrame()),
                patch("leveraged_trader.workflow._load_alpaca_realized_pnl_for_db", return_value=pd.DataFrame()),
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

        self.assertEqual([call[:2] for call in pipeline_calls], [("update", [pair]), ("rebuild", [pair])])
        self.assertIs(pipeline_calls[0][2], pipeline_calls[1][2])
        self.assertEqual(report_sides, ["Long", "Long", "Short"])
        self.assertIn("Rebuilding invalidated strategy state: BLSG/BLSH", report_output.getvalue())
        submit.assert_called_once()

    def test_workflow_recovers_missing_and_mismatched_state_before_buy_worker(self) -> None:
        assets = pd.DataFrame(
            [
                {"symbol": "AAA", "name": "A", "rsi_symbol": "QQQ"},
                {"symbol": "BBB", "name": "B", "rsi_symbol": "QQQ"},
            ]
        )
        histories = {
            symbol: self._symbol_history(symbol, offset)
            for symbol, offset in (("AAA", 0.0), ("BBB", 20.0), ("QQQ", 10.0), ("^IRX", -95.0))
        }
        modes: list[str] = []
        submitted_with_both_states = False

        async def process_and_invalidate(**kwargs: object) -> list[AssetRunResult]:
            mode = str(kwargs["mode"])
            modes.append(mode)
            db_path = str(kwargs["db_path"])
            cfg = kwargs["base_cfg"]
            results = []
            for job in kwargs["jobs"]:
                asset_history = histories[job.asset_symbol]
                signal_history = histories[job.signal_symbol]
                risk_free_history = histories["^IRX"]
                data = _strategy_data_from_authoritative_histories(
                    asset_symbol=job.asset_symbol,
                    signal_symbol=job.signal_symbol,
                    asset_history=asset_history,
                    signal_history=signal_history,
                    risk_free_history=risk_free_history,
                )
                _process_asset_grid_for_db(
                    db_path,
                    data,
                    asset_history,
                    signal_history,
                    risk_free_history,
                    cfg,
                    job.asset_symbol,
                    job.signal_symbol,
                    kwargs["buy_rsi_values"],
                    kwargs["profit_target_values"],
                    True,
                    canonical_signal_history=signal_history,
                )
                results.append(
                    AssetRunResult(
                        job.workflow_idx,
                        job.asset_symbol,
                        job.signal_symbol,
                        "Rebuilding",
                        len(data),
                        "done",
                        "processed",
                        job.workflow,
                    )
                )
            if mode == "update":
                with closing(sqlite3.connect(db_path)) as conn, conn:
                    clear_asset_state(conn, "AAA", "QQQ")
                    conn.execute(
                        "UPDATE strategy_config SET fingerprint = 'stale' "
                        "WHERE asset_symbol = 'BBB' AND signal_symbol = 'QQQ'"
                    )
            return results

        def submit_after_authentication(
            db_path: str,
            _buy_signals: pd.DataFrame,
            _cfg: AlpacaOrderConfig,
        ) -> pd.DataFrame:
            nonlocal submitted_with_both_states
            with closing(sqlite3.connect(db_path)) as conn:
                rows = conn.execute(
                    "SELECT asset_symbol, COUNT(*) FROM strategy_state "
                    "WHERE asset_symbol IN ('AAA', 'BBB') GROUP BY asset_symbol ORDER BY asset_symbol"
                ).fetchall()
            submitted_with_both_states = rows == [("AAA", 1), ("BBB", 1)]
            return pd.DataFrame(columns=["Status"])

        with tempfile.TemporaryDirectory() as tmp:
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ),
                patch("leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db", return_value=assets),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=process_and_invalidate),
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    side_effect=submit_after_authentication,
                ) as submit,
                patch("leveraged_trader.workflow._load_alpaca_managed_positions_for_db", return_value=pd.DataFrame()),
                patch("leveraged_trader.workflow._load_alpaca_realized_pnl_for_db", return_value=pd.DataFrame()),
                patch("leveraged_trader.workflow._write_workflow_outputs"),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(rsi_period=3),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(Path(tmp) / "outputs"),
                        reporter=reporter,
                    )
                )
                summary = pd.read_csv(Path(tmp) / "outputs" / "optimization_summary.csv")

        self.assertEqual(modes, ["update", "rebuild"])
        self.assertEqual(summary["Asset"].tolist(), ["AAA", "BBB"])
        self.assertTrue(submitted_with_both_states)
        submit.assert_called_once()

    def test_workflow_recovery_reuses_cached_histories_without_confirming_boundary_removal(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
        full_signal_history = self._symbol_history("QQQ", 10.0)
        histories = {
            "TQQQ": self._symbol_history("TQQQ"),
            "QQQ": full_signal_history.iloc[1:],
            "^IRX": self._symbol_history("^IRX", -95.0),
        }
        observed_candidates: list[tuple[int, str]] = []

        def batch_loader(symbols: list[str], **_kwargs: object) -> tuple[dict, dict]:
            return {symbol: histories[symbol] for symbol in symbols}, {}

        def unexpected_individual_download(*_args: object, **_kwargs: object) -> None:
            self.fail("Recovery must reuse the histories fetched by the original batch.")

        def invalidate_before_first_report(*args: object, **kwargs: object) -> tuple[pd.DataFrame, ...]:
            if args[4] == "Long":
                with closing(sqlite3.connect(str(args[0]))) as conn, conn:
                    candidate = conn.execute(
                        "SELECT consecutive_observations, last_observed_run_id "
                        "FROM market_history_removal_candidates WHERE symbol = 'QQQ'"
                    ).fetchone()
                    self.assertIsNotNone(candidate)
                    observed_candidates.append(candidate)
                    self.assertEqual(
                        conn.execute(
                            "SELECT COUNT(*) FROM strategy_state "
                            "WHERE asset_symbol = 'TQQQ' AND signal_symbol = 'QQQ'"
                        ).fetchone()[0],
                        1,
                    )
                    if len(observed_candidates) == 1:
                        clear_asset_state(conn, "TQQQ", "QQQ")
            return _build_reports_for_db(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                _synchronize_market_data_history(conn, full_signal_history, "QQQ", observation_run_id="seed")
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ),
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db", return_value=workflow_assets
                ),
                patch("leveraged_trader.workflow.load_symbol_history_batch", side_effect=batch_loader) as download,
                patch("leveraged_trader.workflow.load_symbol_history", new=unexpected_individual_download),
                patch("leveraged_trader.workflow.load_signal_history", new=unexpected_individual_download),
                patch("leveraged_trader.workflow.load_risk_free_history", new=unexpected_individual_download),
                patch("leveraged_trader.workflow._build_reports_for_db", side_effect=invalidate_before_first_report),
                patch(
                    "leveraged_trader.storage._synchronize_market_data_history",
                    wraps=_synchronize_market_data_history,
                ) as synchronize,
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    return_value=pd.DataFrame(columns=["Status"]),
                ) as submit,
                patch("leveraged_trader.workflow._write_workflow_outputs"),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=db_path,
                        base_cfg=BacktestConfig(rsi_period=3),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(Path(tmp) / "outputs"),
                        workflow_concurrency=1,
                        reporter=reporter,
                    )
                )
            with closing(sqlite3.connect(db_path)) as conn:
                stored_signal_count = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol = 'QQQ'"
                ).fetchone()[0]
            summary = pd.read_csv(Path(tmp) / "outputs" / "optimization_summary.csv")

        signal_syncs = [call for call in synchronize.call_args_list if call.args[2] == "QQQ"]
        self.assertEqual(len(signal_syncs), 2)
        self.assertTrue(all(call.args[1] is histories["QQQ"] for call in signal_syncs))
        observation_ids = [call.kwargs["observation_run_id"] for call in signal_syncs]
        self.assertTrue(observation_ids[0])
        self.assertEqual(observation_ids[0], observation_ids[1])
        self.assertEqual(observed_candidates, [(1, observation_ids[0]), (1, observation_ids[0])])
        self.assertEqual(stored_signal_count, len(full_signal_history))
        self.assertEqual(summary["Asset"].tolist(), ["TQQQ"])
        download.assert_called_once()
        submit.assert_called_once()

    def test_workflow_aborts_before_submission_if_recovery_is_incomplete_or_fails(self) -> None:
        pair = ("TQQQ", "QQQ")
        workflow_assets = pd.DataFrame([{"symbol": pair[0], "name": "T", "rsi_symbol": pair[1]}])
        for recovery_outcome in ("skipped", "missing", "error"):
            with self.subTest(recovery_outcome=recovery_outcome), tempfile.TemporaryDirectory() as tmp:
                pipeline_modes: list[str] = []

                async def fake_pipeline(
                    _pipeline_modes: list[str] = pipeline_modes,
                    _recovery_outcome: str = recovery_outcome,
                    **kwargs: object,
                ) -> list[AssetRunResult]:
                    mode = str(kwargs["mode"])
                    _pipeline_modes.append(mode)
                    if mode == "rebuild":
                        if _recovery_outcome == "error":
                            raise RuntimeError("Recovery worker failed.")
                        if _recovery_outcome == "missing":
                            return []
                    return [
                        AssetRunResult(
                            job.workflow_idx,
                            job.asset_symbol,
                            job.signal_symbol,
                            "Rebuilding",
                            1,
                            "skipped" if mode == "rebuild" else "done",
                            "market data unavailable" if mode == "rebuild" else "processed",
                            job.workflow,
                        )
                        for job in kwargs["jobs"]
                    ]

                reporter = WorkflowReporter(
                    console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
                )
                expected_error = RuntimeError if recovery_outcome == "error" else WorkflowRunError
                expected_message = "Recovery worker failed" if recovery_outcome == "error" else "Could not recover all"
                with (
                    patch(
                        "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                        return_value=pd.DataFrame(columns=["Action"]),
                    ),
                    patch(
                        "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                        return_value=workflow_assets,
                    ),
                    patch("leveraged_trader.workflow._run_asset_pipeline", new=fake_pipeline),
                    patch(
                        "leveraged_trader.workflow._build_reports_for_db",
                        side_effect=_CompletedWorkflowStateInvalidError("Completed state disappeared", {pair}),
                    ),
                    patch("leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db") as submit,
                    self.assertRaisesRegex(expected_error, expected_message),
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

                self.assertEqual(pipeline_modes, ["update", "rebuild"])
                submit.assert_not_called()

    def test_workflow_aborts_if_rebuilt_pair_disappears_again(self) -> None:
        pair = ("BLSG", "BLSH")
        workflow_assets = pd.DataFrame([{"symbol": pair[0], "name": pair[0], "rsi_symbol": pair[1]}])
        pipeline_modes: list[str] = []

        async def fake_pipeline(**kwargs: object) -> list[AssetRunResult]:
            pipeline_modes.append(str(kwargs["mode"]))
            return [
                AssetRunResult(
                    job.workflow_idx, job.asset_symbol, job.signal_symbol, "Updating", 1, "done", "done", job.workflow
                )
                for job in kwargs["jobs"]
            ]

        with tempfile.TemporaryDirectory() as tmp:
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ),
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db", return_value=workflow_assets
                ),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=fake_pipeline),
                patch(
                    "leveraged_trader.workflow._build_reports_for_db",
                    side_effect=_CompletedWorkflowStateInvalidError("Completed workflow state disappeared", {pair}),
                ) as reports,
                patch("leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db") as submit,
                self.assertRaises(_CompletedWorkflowStateInvalidError),
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

        self.assertEqual(pipeline_modes, ["update", "rebuild"])
        self.assertEqual(reports.call_count, 2)
        submit.assert_not_called()

    def test_short_side_recovery_rechecks_long_report_before_submission(self) -> None:
        short_pair = ("SQQQ", "QQQ")
        workflow_asset_groups = {
            "long": pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}]),
            "short": pd.DataFrame([{"symbol": short_pair[0], "name": "S", "rsi_symbol": short_pair[1]}]),
        }
        pipeline_calls: list[tuple[str, str]] = []
        report_sides: list[str | None] = []

        async def fake_pipeline(**kwargs: object) -> list[AssetRunResult]:
            jobs = kwargs["jobs"]
            pipeline_calls.append((str(kwargs["mode"]), jobs[0].asset_symbol))
            return [
                AssetRunResult(
                    job.workflow_idx, job.asset_symbol, job.signal_symbol, "Updating", 1, "done", "done", job.workflow
                )
                for job in jobs
            ]

        def fake_reports(*args: object, **_kwargs: object) -> tuple[pd.DataFrame, ...]:
            workflow_label = args[4]
            report_sides.append(workflow_label)
            if workflow_label == "Short" and report_sides.count("Short") == 1:
                raise _CompletedWorkflowStateInvalidError("Completed short state disappeared", {short_pair})
            return (pd.DataFrame(),) * 6

        with tempfile.TemporaryDirectory() as tmp:
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ),
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                    return_value=workflow_asset_groups,
                ),
                patch("leveraged_trader.workflow._run_asset_pipeline", new=fake_pipeline),
                patch("leveraged_trader.workflow._build_reports_for_db", side_effect=fake_reports),
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    return_value=pd.DataFrame(columns=["Status"]),
                ) as submit,
                patch("leveraged_trader.workflow._load_alpaca_managed_positions_for_db", return_value=pd.DataFrame()),
                patch("leveraged_trader.workflow._load_alpaca_realized_pnl_for_db", return_value=pd.DataFrame()),
                patch("leveraged_trader.workflow._write_workflow_outputs"),
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=str(Path(tmp) / "state.sqlite"),
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        short_buy_rsi_values=[70.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(Path(tmp) / "outputs"),
                        reporter=reporter,
                    )
                )

        self.assertEqual(pipeline_calls, [("update", "TQQQ"), ("update", "SQQQ"), ("rebuild", "SQQQ")])
        self.assertEqual(report_sides, ["Long", "Short", "Long", "Short"])
        submit.assert_called_once()

    def test_workflow_fails_when_every_asset_is_skipped(self) -> None:
        workflow_assets = pd.DataFrame([{"symbol": "TQQQ", "name": "T", "rsi_symbol": "QQQ"}])
        reconciliation = pd.DataFrame(
            [
                {
                    "Position ID": 1,
                    "Asset": "TQQQ",
                    "Action": "sell",
                    "Status": "accepted",
                }
            ]
        )

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

        async def immediate_run_blocking(
            _executor: object,
            func: object,
            /,
            *args: object,
            **kwargs: object,
        ) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir()
            for filename in _RESEARCH_REPORT_FILENAMES:
                (output_dir / filename).write_text("Generation\nstale\n", encoding="utf-8")
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow._run_blocking", new=immediate_run_blocking),
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=reconciliation,
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
                        db_path=db_path,
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30.0],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=str(output_dir),
                        reporter=reporter,
                    )
                )

            self.assertEqual(
                pd.read_csv(output_dir / "alpaca_reconciliation_results.csv")["Status"].tolist(),
                ["accepted"],
            )
            self.assertTrue((output_dir / "managed_positions.csv").is_file())
            self.assertTrue((output_dir / "alpaca_realized_pnl.csv").is_file())
            for filename in _RESEARCH_REPORT_FILENAMES:
                self.assertFalse((output_dir / filename).exists())

        mock_build_reports.assert_not_called()
        mock_submit_buys.assert_not_called()

    def test_workflow_fails_when_no_assets_are_run(self) -> None:
        empty_assets = pd.DataFrame(columns=["symbol", "name", "rsi_symbol"])
        workflow_asset_groups = {
            "long": empty_assets,
            "short": empty_assets.copy(),
        }

        async def immediate_run_blocking(
            _executor: object,
            func: object,
            /,
            *args: object,
            **kwargs: object,
        ) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "outputs"
            output_dir.mkdir()
            for filename in _RESEARCH_REPORT_FILENAMES:
                (output_dir / filename).write_text("Generation\nstale\n", encoding="utf-8")
            reporter = WorkflowReporter(
                console=Console(file=io.StringIO(), width=100, color_system=None, no_color=True)
            )
            with (
                patch("leveraged_trader.workflow._run_blocking", new=immediate_run_blocking),
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
                        output_dir=str(output_dir),
                        reporter=reporter,
                    )
                )

            self.assertTrue((output_dir / "managed_positions.csv").is_file())
            self.assertTrue((output_dir / "alpaca_realized_pnl.csv").is_file())
            self.assertTrue(pd.read_csv(output_dir / "alpaca_reconciliation_results.csv").empty)
            for filename in _RESEARCH_REPORT_FILENAMES:
                self.assertFalse((output_dir / filename).exists())

        mock_run_pipeline.assert_not_called()
        mock_build_reports.assert_not_called()
        mock_submit_buys.assert_not_called()

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_asset_pipeline_skips_malformed_tradier_shape_locally(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "Open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1000,
                        }
                    }
                }
            ),
        )

        with patch(
            "leveraged_trader.workflow._prepare_asset_run",
            return_value=AssetRunPlan(
                asset_symbol="AAA",
                signal_symbol="AAA",
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            ),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "AAA", "AAA")],
                    concurrency=1,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(auto_adjust=False),
                    tradier_cfg=TradierMarketDataConfig(access_token="token"),
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                )
            )

        self.assertEqual([result.status for result in results], ["skipped"])
        self.assertIn("duplicate required columns", results[0].message)
        self.assertEqual(mock_get.call_count, 1)

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

    def test_asset_pipeline_preserves_primary_failure_when_session_close_fails(self) -> None:
        job = AssetRunJob(1, "AAA", "AAA")
        primary_failure = RuntimeError("consumer failed")

        async def fake_prepare(**_kwargs: object) -> AssetRunResult:
            return AssetRunResult(
                workflow_idx=1,
                asset_symbol="AAA",
                signal_symbol="AAA",
                action="Updating",
                rows_processed=0,
                status="skipped",
                message="prepared",
            )

        async def fail_complete(*_args: object, **_kwargs: object) -> AssetRunResult:
            raise primary_failure

        with (
            patch("leveraged_trader.workflow._prepare_workflow_asset", new=fake_prepare),
            patch("leveraged_trader.workflow._complete_workflow_asset", new=fail_complete),
            patch.object(
                _WorkflowStrategySession,
                "close",
                side_effect=OSError("close failed"),
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            asyncio.run(
                _run_asset_pipeline(
                    jobs=[job],
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

        self.assertIs(raised.exception, primary_failure)
        self.assertIn(
            "Failed strategy-session cleanup after the asset pipeline failed: close failed",
            raised.exception.__notes__,
        )

    def test_asset_pipeline_preserves_cancellation_when_session_close_fails(self) -> None:
        cancellation = asyncio.CancelledError("pipeline cancelled")

        async def cancel_prepare(**_kwargs: object) -> AssetRunResult:
            raise cancellation

        with (
            patch("leveraged_trader.workflow._prepare_workflow_asset", new=cancel_prepare),
            patch.object(
                _WorkflowStrategySession,
                "close",
                side_effect=OSError("close failed"),
            ),
            self.assertRaises(asyncio.CancelledError) as raised,
        ):
            asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "AAA", "AAA")],
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

        self.assertIs(raised.exception, cancellation)
        self.assertIn(
            "Failed strategy-session cleanup after the asset pipeline failed: close failed",
            raised.exception.__notes__,
        )

    def test_asset_pipeline_raises_session_close_failure_without_primary_failure(self) -> None:
        job = AssetRunJob(1, "AAA", "AAA")

        async def fake_prepare(**_kwargs: object) -> AssetRunResult:
            return AssetRunResult(
                workflow_idx=1,
                asset_symbol="AAA",
                signal_symbol="AAA",
                action="Updating",
                rows_processed=0,
                status="skipped",
                message="prepared",
            )

        with (
            patch("leveraged_trader.workflow._prepare_workflow_asset", new=fake_prepare),
            patch.object(
                _WorkflowStrategySession,
                "close",
                side_effect=OSError("close failed"),
            ),
            self.assertRaisesRegex(OSError, "close failed"),
        ):
            asyncio.run(
                _run_asset_pipeline(
                    jobs=[job],
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

    def test_asset_preparation_returns_skipped_result_for_empty_market_data(self) -> None:
        job = AssetRunJob(1, "TQQQ", "QQQ", workflow="Long")

        with (
            patch(
                "leveraged_trader.workflow._prepare_asset_run",
                return_value=AssetRunPlan(
                    asset_symbol="TQQQ",
                    signal_symbol="QQQ",
                    rebuild=False,
                    start=None,
                    action="Updating",
                    start_label="earliest overlapping history",
                ),
            ),
            patch("leveraged_trader.workflow.load_symbol_history", return_value=pd.DataFrame()),
        ):
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

    def test_asset_preparation_propagates_unexpected_download_runtime_failure(self) -> None:
        job = AssetRunJob(1, "TQQQ", "QQQ")
        with (
            patch(
                "leveraged_trader.workflow._prepare_asset_run",
                return_value=AssetRunPlan(
                    asset_symbol="TQQQ",
                    signal_symbol="QQQ",
                    rebuild=False,
                    start=None,
                    action="Updating",
                    start_label="earliest overlapping history",
                ),
            ),
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                side_effect=RuntimeError("executor invariant failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "executor invariant failed"),
        ):
            asyncio.run(
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
                )
            )

    def test_asset_pipeline_memoizes_shared_benchmark_download_failure(self) -> None:
        jobs = [AssetRunJob(1, "AAA", "QQQ"), AssetRunJob(2, "BBB", "QQQ")]

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                side_effect=lambda symbol, **_kwargs: self._symbol_history(symbol),
            ),
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                side_effect=MarketDataDownloadError({"^IRX": "timed out"}),
            ) as benchmark_download,
            patch("leveraged_trader.workflow.load_signal_history") as signal_download,
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

        self.assertEqual([result.status for result in results], ["skipped", "skipped"])
        self.assertEqual(benchmark_download.call_count, 1)
        signal_download.assert_not_called()

    def test_asset_pipeline_memoizes_shared_signal_download_failure(self) -> None:
        jobs = [AssetRunJob(1, "AAA", "QQQ"), AssetRunJob(2, "BBB", "QQQ")]

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                side_effect=lambda symbol, **_kwargs: self._symbol_history(symbol),
            ),
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ),
            patch(
                "leveraged_trader.workflow.load_signal_history",
                side_effect=MarketDataDownloadError({"QQQ": "timed out"}),
            ) as signal_download,
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

        self.assertEqual([result.status for result in results], ["skipped", "skipped"])
        self.assertEqual(signal_download.call_count, 2)
        self.assertIsNone(signal_download.call_args_list[0].kwargs.get("start"))
        self.assertEqual(signal_download.call_args_list[1].kwargs["start"], "2025-01-01")

    def test_unusable_full_signal_history_retries_a_pair_scoped_window(self) -> None:
        market_data_session = _WorkflowMarketDataSession()
        asset_history = self._symbol_history("RCAX")
        bounded_signal_history = self._symbol_history("RCAT", 10.0)
        processed: dict[str, object] = {}

        def plan(*_args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol="RCAX",
                signal_symbol="RCAT",
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        def load_signal(_symbol: str, *, start: str | None = None, **_kwargs: object) -> pd.DataFrame:
            if start is None:
                raise MarketDataDownloadError({"RCAT": "invalid historical OHLCV row"})
            return bounded_signal_history

        def process(*args: object, **kwargs: object) -> None:
            processed["signal_history"] = args[3]
            processed["canonical_signal_history"] = kwargs["canonical_signal_history"]
            processed["isolate_signal_history"] = kwargs["isolate_signal_history"]

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch("leveraged_trader.workflow.load_symbol_history", return_value=asset_history),
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ),
            patch("leveraged_trader.workflow.load_signal_history", side_effect=load_signal) as signal_download,
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=process),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "RCAX", "RCAT")],
                    concurrency=1,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=market_data_session,
                )
            )

        self.assertEqual([result.status for result in results], ["done"])
        self.assertEqual(signal_download.call_count, 2)
        self.assertEqual(signal_download.call_args_list[1].kwargs["start"], "2025-01-01")
        self.assertNotIn("RCAT", market_data_session.signal_histories)
        self.assertIs(
            market_data_session.calendar_signal_histories[("RCAX", "RCAT")],
            bounded_signal_history,
        )
        self.assertIs(processed["signal_history"], bounded_signal_history)
        self.assertIsNone(processed["canonical_signal_history"])
        self.assertIs(processed["isolate_signal_history"], True)

    def test_asset_pipelines_share_canonical_histories_across_workflow_sides(self) -> None:
        market_data_session = _WorkflowMarketDataSession()
        signal_history = self._symbol_history("QQQ", 10.0)
        corrected_signal_history = signal_history.copy()
        corrected_signal_history.iloc[-1, corrected_signal_history.columns.get_loc("QQQ_Close")] += 5.0
        processed_signal_histories: list[pd.DataFrame] = []

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        def process(*args: object, **_kwargs: object) -> None:
            processed_signal_histories.append(args[3])

        async def run_both_sides() -> None:
            for asset_symbol in ("TQQQ", "SQQQ"):
                results = await _run_asset_pipeline(
                    jobs=[AssetRunJob(1, asset_symbol, "QQQ")],
                    concurrency=1,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=market_data_session,
                )
                self.assertEqual([result.status for result in results], ["done"])

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                side_effect=lambda symbol, **_kwargs: self._symbol_history(symbol),
            ),
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ) as benchmark_download,
            patch(
                "leveraged_trader.workflow.load_signal_history",
                side_effect=[signal_history, corrected_signal_history],
            ) as signal_download,
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=process),
        ):
            asyncio.run(run_both_sides())

        self.assertEqual(signal_download.call_count, 1)
        self.assertEqual(benchmark_download.call_count, 1)
        self.assertEqual(len(processed_signal_histories), 2)
        self.assertIs(processed_signal_histories[0], processed_signal_histories[1])

    def test_cached_signal_calendar_recovery_is_shared_and_preserves_provenance(self) -> None:
        market_data_session = _WorkflowMarketDataSession()
        yahoo_signal_history = self._symbol_history("QQQ", 10.0)
        yahoo_signal_history.index = yahoo_signal_history.index + pd.DateOffset(months=3)
        recovered_signal_history = self._symbol_history("QQQ", 20.0)
        recovered_signal_history.attrs["provider_request_id"] = "tradier-request-1"
        processed: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        def process(*args: object, **_kwargs: object) -> None:
            processed.append((args[1], args[3]))

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                side_effect=lambda symbol, **_kwargs: self._symbol_history(symbol),
            ),
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ),
            patch(
                "leveraged_trader.workflow.load_signal_history",
                return_value=yahoo_signal_history,
            ) as signal_download,
            patch(
                "leveraged_trader.market_data._load_tradier_fallback_frames",
                return_value=({"QQQ": recovered_signal_history}, {}),
            ) as fallback,
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=process),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "AAA", "QQQ"), AssetRunJob(2, "BBB", "QQQ")],
                    concurrency=2,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(auto_adjust=False),
                    tradier_cfg=TradierMarketDataConfig(access_token="token"),
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=market_data_session,
                )
            )

        self.assertEqual([result.status for result in results], ["done", "done"])
        signal_download.assert_called_once()
        fallback.assert_called_once()
        self.assertEqual(len(processed), 2)
        self.assertIs(processed[0][1], processed[1][1])
        self.assertEqual(processed[0][1].attrs["provider_request_id"], "tradier-request-1")
        for data, signal_history in processed:
            self.assertEqual(data.attrs[MARKET_DATA_PROVIDERS_ATTR], {"QQQ": "tradier"})
            self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["QQQ"])
            self.assertEqual(signal_history.attrs[MARKET_DATA_PROVIDERS_ATTR], {"QQQ": "tradier"})

    def test_calendar_recovery_is_pair_scoped_and_preserves_canonical_signal(self) -> None:
        market_data_session = _WorkflowMarketDataSession()
        yahoo_signal_history = self._symbol_history("QQQ", 10.0)
        yahoo_signal_history.index = yahoo_signal_history.index + pd.DateOffset(months=3)
        yahoo_signal_history.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "yahoo_finance"}
        recovered_signal_history = self._symbol_history("QQQ", 20.0)
        processed: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        def load_asset(symbol: str, **_kwargs: object) -> pd.DataFrame:
            history = self._symbol_history(symbol)
            if symbol == "BBB":
                history.index = yahoo_signal_history.index.copy()
            return history

        def process(*args: object, **kwargs: object) -> None:
            processed[str(args[6])] = (
                args[1],
                args[3],
                kwargs["canonical_signal_history"],
            )

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch("leveraged_trader.workflow.load_symbol_history", side_effect=load_asset),
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ),
            patch(
                "leveraged_trader.workflow.load_signal_history",
                return_value=yahoo_signal_history,
            ) as signal_download,
            patch(
                "leveraged_trader.market_data._load_tradier_fallback_frames",
                return_value=({"QQQ": recovered_signal_history}, {}),
            ) as fallback,
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=process),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "AAA", "QQQ"), AssetRunJob(2, "BBB", "QQQ")],
                    concurrency=1,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(auto_adjust=False),
                    tradier_cfg=TradierMarketDataConfig(access_token="token"),
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=market_data_session,
                )
            )

        self.assertEqual([result.status for result in results], ["done", "done"])
        signal_download.assert_called_once()
        fallback.assert_called_once()
        self.assertIs(market_data_session.signal_histories["QQQ"], yahoo_signal_history)
        self.assertIs(
            market_data_session.calendar_signal_histories[("AAA", "QQQ")],
            processed["AAA"][1],
        )
        self.assertIs(market_data_session.calendar_signal_histories[("BBB", "QQQ")], yahoo_signal_history)
        self.assertIs(processed["BBB"][1], yahoo_signal_history)
        self.assertIs(processed["AAA"][2], yahoo_signal_history)
        self.assertIs(processed["BBB"][2], yahoo_signal_history)
        self.assertEqual(processed["AAA"][0].attrs[MARKET_DATA_PROVIDERS_ATTR], {"QQQ": "tradier"})
        self.assertEqual(processed["BBB"][0].attrs[MARKET_DATA_PROVIDERS_ATTR], {"QQQ": "yahoo_finance"})

    def test_disjoint_recovered_and_canonical_signals_remain_pair_scoped_in_state_and_reports(self) -> None:
        recovered_signal = self._symbol_history("QQQ", 20.0)
        recovered_signal.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "tradier"}
        recovered_signal.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = ["QQQ"]
        canonical_signal = self._symbol_history("QQQ", 10.0)
        canonical_signal.index = canonical_signal.index + pd.DateOffset(months=3)
        descending_close = np.arange(200.0, 180.0, -1.0)
        canonical_signal["QQQ_Open"] = descending_close
        canonical_signal["QQQ_High"] = descending_close + 1.0
        canonical_signal["QQQ_Low"] = descending_close - 1.0
        canonical_signal["QQQ_Close"] = descending_close
        canonical_signal.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "yahoo_finance"}

        aaa_history = self._symbol_history("AAA")
        bbb_history = self._symbol_history("BBB", 20.0)
        bbb_history.index = canonical_signal.index.copy()
        january_risk_free = self._symbol_history("^IRX", -95.0)
        april_risk_free = january_risk_free.copy()
        april_risk_free.index = canonical_signal.index.copy()
        risk_free_history = pd.concat([january_risk_free, april_risk_free]).sort_index()
        cfg = BacktestConfig(rsi_period=19)

        def strategy_data(
            asset_symbol: str,
            asset_history: pd.DataFrame,
            signal_history: pd.DataFrame,
        ) -> pd.DataFrame:
            return _strategy_data_from_authoritative_histories(
                asset_symbol=asset_symbol,
                signal_symbol="QQQ",
                asset_history=asset_history,
                signal_history=signal_history,
                risk_free_history=risk_free_history,
            )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            session = _WorkflowStrategySession(db_path, history_observation_run_id="mixed-provider-run")
            try:
                _process_asset_grid_for_db(
                    db_path,
                    strategy_data("AAA", aaa_history, recovered_signal),
                    aaa_history,
                    recovered_signal,
                    risk_free_history,
                    cfg,
                    "AAA",
                    "QQQ",
                    [50.0],
                    [1.5],
                    False,
                    strategy_session=session,
                    canonical_signal_history=canonical_signal,
                )
                _process_asset_grid_for_db(
                    db_path,
                    strategy_data("BBB", bbb_history, canonical_signal),
                    bbb_history,
                    canonical_signal,
                    risk_free_history,
                    cfg,
                    "BBB",
                    "QQQ",
                    [50.0],
                    [1.5],
                    False,
                    strategy_session=session,
                    canonical_signal_history=canonical_signal,
                )
            finally:
                session.close()

            with closing(sqlite3.connect(db_path)) as conn:
                global_dates = [
                    row[0]
                    for row in conn.execute(
                        "SELECT date FROM market_data WHERE symbol = 'QQQ' ORDER BY date"
                    ).fetchall()
                ]
                scoped_rows = conn.execute(
                    """
                    SELECT symbol, COUNT(*)
                    FROM market_data
                    WHERE symbol LIKE '@strategy-signal-v1/%'
                    GROUP BY symbol
                    """
                ).fetchall()
                aaa_rsi = load_aligned_rsi_for_asset_session(
                    conn,
                    "AAA",
                    "QQQ",
                    cfg.rsi_period,
                    aaa_history.index[-1],
                )
                bbb_rsi = load_aligned_rsi_for_asset_session(
                    conn,
                    "BBB",
                    "QQQ",
                    cfg.rsi_period,
                    bbb_history.index[-1],
                )
                bbb_integrity = conn.execute(
                    "SELECT integrity_digest FROM strategy_state WHERE asset_symbol = 'BBB'"
                ).fetchone()[0]

            self.assertEqual(global_dates, [date.date().isoformat() for date in canonical_signal.index])
            self.assertEqual(len(scoped_rows), 1)
            self.assertEqual(scoped_rows[0][1], len(recovered_signal))
            self.assertEqual(aaa_rsi, (aaa_history.index[-1].date().isoformat(), 100.0))
            self.assertEqual(bbb_rsi, (bbb_history.index[-1].date().isoformat(), 0.0))

            workflow_assets = pd.DataFrame(
                [
                    {"symbol": "AAA", "name": "AAA", "rsi_symbol": "QQQ"},
                    {"symbol": "BBB", "name": "BBB", "rsi_symbol": "QQQ"},
                ]
            )
            summary, _curves, _buy, _eligible, _sell, _pnl = _build_reports_for_db(
                db_path,
                workflow_assets,
                cfg,
                {("AAA", "QQQ"), ("BBB", "QQQ")},
                buy_rsi_values=[50.0],
                profit_target_values=[1.5],
            )
            with closing(sqlite3.connect(db_path)) as conn:
                pending_buys = build_pending_action_report(
                    conn,
                    summary,
                    cfg.rsi_period,
                    pending_action_filter="buy",
                    require_multiple_trades=False,
                    min_sharpe=None,
                    base_cfg=cfg,
                    expected_buy_rsi_values=[50.0],
                    expected_profit_target_values=[1.5],
                )
            self.assertEqual(pending_buys["Asset"].tolist(), ["BBB"])
            self.assertEqual(pending_buys["Latest RSI"].tolist(), [0.0])

            corrected_recovery = recovered_signal.copy()
            corrected_close = np.arange(160.0, 140.0, -1.0)
            corrected_recovery["QQQ_Open"] = corrected_close
            corrected_recovery["QQQ_High"] = corrected_close + 1.0
            corrected_recovery["QQQ_Low"] = corrected_close - 1.0
            corrected_recovery["QQQ_Close"] = corrected_close
            correction_session = _WorkflowStrategySession(
                db_path,
                history_observation_run_id="recovery-correction-run",
            )
            try:
                _process_asset_grid_for_db(
                    db_path,
                    strategy_data("AAA", aaa_history, corrected_recovery),
                    aaa_history,
                    corrected_recovery,
                    risk_free_history,
                    cfg,
                    "AAA",
                    "QQQ",
                    [50.0],
                    [1.5],
                    False,
                    strategy_session=correction_session,
                    canonical_signal_history=canonical_signal,
                )
            finally:
                correction_session.close()

            with closing(sqlite3.connect(db_path)) as conn:
                corrected_aaa_rsi = load_aligned_rsi_for_asset_session(
                    conn,
                    "AAA",
                    "QQQ",
                    cfg.rsi_period,
                    aaa_history.index[-1],
                )
                bbb_integrity_after_correction = conn.execute(
                    "SELECT integrity_digest FROM strategy_state WHERE asset_symbol = 'BBB'"
                ).fetchone()[0]
                global_count_after_correction = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol = 'QQQ'"
                ).fetchone()[0]
                scoped_count_after_correction = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol LIKE '@strategy-signal-v1/%'"
                ).fetchone()[0]
            self.assertEqual(corrected_aaa_rsi, (aaa_history.index[-1].date().isoformat(), 0.0))
            self.assertEqual(bbb_integrity_after_correction, bbb_integrity)
            self.assertEqual(global_count_after_correction, len(canonical_signal))
            self.assertEqual(scoped_count_after_correction, len(corrected_recovery))
            corrected_summary, *_rest = _build_reports_for_db(
                db_path,
                workflow_assets,
                cfg,
                {("AAA", "QQQ"), ("BBB", "QQQ")},
                buy_rsi_values=[50.0],
                profit_target_values=[1.5],
            )
            self.assertEqual(set(corrected_summary["Asset"]), {"AAA", "BBB"})

            # When the recovered pair later uses the canonical provider, the
            # workflow explicitly retires its private snapshot and rebuilds on
            # the global history instead of retaining a one-run 40-row union.
            extended_aaa = pd.concat(
                [aaa_history, aaa_history.set_axis(canonical_signal.index)],
            ).sort_index()
            transition_session = _WorkflowStrategySession(
                db_path,
                history_observation_run_id="canonical-transition-run",
            )
            try:
                _process_asset_grid_for_db(
                    db_path,
                    strategy_data("AAA", extended_aaa, canonical_signal),
                    extended_aaa,
                    canonical_signal,
                    risk_free_history,
                    cfg,
                    "AAA",
                    "QQQ",
                    [50.0],
                    [1.5],
                    False,
                    strategy_session=transition_session,
                    canonical_signal_history=canonical_signal,
                )
            finally:
                transition_session.close()

            with closing(sqlite3.connect(db_path)) as conn:
                remaining_scope_rows = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol LIKE '@strategy-signal-v1/%'"
                ).fetchone()[0]
                transitioned_rsi = load_aligned_rsi_for_asset_session(
                    conn,
                    "AAA",
                    "QQQ",
                    cfg.rsi_period,
                    canonical_signal.index[-1],
                )
            self.assertEqual(remaining_scope_rows, 0)
            self.assertEqual(transitioned_rsi, (canonical_signal.index[-1].date().isoformat(), 0.0))
            transitioned_summary, *_rest = _build_reports_for_db(
                db_path,
                workflow_assets,
                cfg,
                {("AAA", "QQQ"), ("BBB", "QQQ")},
                buy_rsi_values=[50.0],
                profit_target_values=[1.5],
            )
            self.assertEqual(set(transitioned_summary["Asset"]), {"AAA", "BBB"})

        with tempfile.TemporaryDirectory() as tmp:
            reverse_db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(reverse_db_path)) as conn, conn:
                init_state_db(conn)
            reverse_session = _WorkflowStrategySession(
                reverse_db_path,
                history_observation_run_id="reverse-provider-order-run",
            )
            try:
                for asset_symbol, asset_history, pair_signal in (
                    ("BBB", bbb_history, canonical_signal),
                    ("AAA", aaa_history, recovered_signal),
                ):
                    _process_asset_grid_for_db(
                        reverse_db_path,
                        strategy_data(asset_symbol, asset_history, pair_signal),
                        asset_history,
                        pair_signal,
                        risk_free_history,
                        cfg,
                        asset_symbol,
                        "QQQ",
                        [50.0],
                        [1.5],
                        False,
                        strategy_session=reverse_session,
                        canonical_signal_history=canonical_signal,
                    )
            finally:
                reverse_session.close()
            with closing(sqlite3.connect(reverse_db_path)) as conn:
                reverse_global_count = conn.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'QQQ'").fetchone()[
                    0
                ]
                reverse_scope_count = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol LIKE '@strategy-signal-v1/%'"
                ).fetchone()[0]
                reverse_aaa_rsi = load_aligned_rsi_for_asset_session(
                    conn,
                    "AAA",
                    "QQQ",
                    cfg.rsi_period,
                    aaa_history.index[-1],
                )
                reverse_bbb_rsi = load_aligned_rsi_for_asset_session(
                    conn,
                    "BBB",
                    "QQQ",
                    cfg.rsi_period,
                    bbb_history.index[-1],
                )
            self.assertEqual(reverse_global_count, len(canonical_signal))
            self.assertEqual(reverse_scope_count, len(recovered_signal))
            self.assertEqual(reverse_aaa_rsi, aaa_rsi)
            self.assertEqual(reverse_bbb_rsi, bbb_rsi)

    def test_global_signal_correction_preserves_private_pair_states(self) -> None:
        canonical_signal = self._symbol_history("QQQ", 10.0)
        canonical_signal.index = canonical_signal.index + pd.DateOffset(months=3)
        canonical_signal.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "yahoo_finance"}
        corrected_canonical = canonical_signal.copy()
        corrected_close = float(corrected_canonical.iloc[-1]["QQQ_Close"]) + 5.0
        corrected_canonical.iloc[-1, corrected_canonical.columns.get_loc("QQQ_Open")] = corrected_close
        corrected_canonical.iloc[-1, corrected_canonical.columns.get_loc("QQQ_High")] = corrected_close + 1.0
        corrected_canonical.iloc[-1, corrected_canonical.columns.get_loc("QQQ_Low")] = corrected_close - 1.0
        corrected_canonical.iloc[-1, corrected_canonical.columns.get_loc("QQQ_Close")] = corrected_close
        recovered_signal = self._symbol_history("QQQ", 20.0)
        recovered_signal.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "tradier"}
        recovered_signal.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = ["QQQ"]
        histories = {
            "AAA": self._symbol_history("AAA"),
            "CCC": self._symbol_history("CCC", 40.0),
            "BBB": self._symbol_history("BBB", 20.0).set_axis(canonical_signal.index),
        }
        january_risk_free = self._symbol_history("^IRX", -95.0)
        april_risk_free = january_risk_free.set_axis(canonical_signal.index)
        risk_free_history = pd.concat([january_risk_free, april_risk_free]).sort_index()
        cfg = BacktestConfig(rsi_period=3)

        def process(
            session: _WorkflowStrategySession,
            db_path: str,
            asset_symbol: str,
            pair_signal: pd.DataFrame,
            canonical: pd.DataFrame,
        ) -> None:
            asset_history = histories[asset_symbol]
            data = _strategy_data_from_authoritative_histories(
                asset_symbol=asset_symbol,
                signal_symbol="QQQ",
                asset_history=asset_history,
                signal_history=pair_signal,
                risk_free_history=risk_free_history,
            )
            _process_asset_grid_for_db(
                db_path,
                data,
                asset_history,
                pair_signal,
                risk_free_history,
                cfg,
                asset_symbol,
                "QQQ",
                [30.0],
                [1.5],
                False,
                strategy_session=session,
                canonical_signal_history=canonical,
            )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            session = _WorkflowStrategySession(db_path, history_observation_run_id="global-correction-run")
            try:
                process(session, db_path, "AAA", recovered_signal, canonical_signal)
                process(session, db_path, "CCC", recovered_signal.copy(), canonical_signal)
                process(session, db_path, "BBB", canonical_signal, canonical_signal)
                with closing(sqlite3.connect(db_path)) as reader:
                    private_integrity_before = dict(
                        reader.execute(
                            """
                            SELECT asset_symbol, integrity_digest
                            FROM strategy_state
                            WHERE asset_symbol IN ('AAA', 'CCC')
                            """
                        ).fetchall()
                    )
                process(session, db_path, "BBB", corrected_canonical, corrected_canonical)
            finally:
                session.close()

            with closing(sqlite3.connect(db_path)) as conn:
                private_integrity_after = dict(
                    conn.execute(
                        """
                        SELECT asset_symbol, integrity_digest
                        FROM strategy_state
                        WHERE asset_symbol IN ('AAA', 'CCC')
                        """
                    ).fetchall()
                )
                all_states = {
                    row[0] for row in conn.execute("SELECT DISTINCT asset_symbol FROM strategy_state").fetchall()
                }
                private_row_count = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol LIKE '@strategy-signal-v1/%'"
                ).fetchone()[0]
            self.assertEqual(private_integrity_after, private_integrity_before)
            self.assertEqual(all_states, {"AAA", "BBB", "CCC"})
            self.assertEqual(private_row_count, 2 * len(recovered_signal))

            workflow_assets = pd.DataFrame(
                [{"symbol": symbol, "name": symbol, "rsi_symbol": "QQQ"} for symbol in ("AAA", "BBB", "CCC")]
            )
            summary, *_rest = _build_reports_for_db(
                db_path,
                workflow_assets,
                cfg,
                {(symbol, "QQQ") for symbol in ("AAA", "BBB", "CCC")},
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
            )
            self.assertEqual(set(summary["Asset"]), {"AAA", "BBB", "CCC"})

    def test_unused_global_signal_correction_resumes_current_private_pair(self) -> None:
        from leveraged_trader import storage as storage_module

        canonical_signal = self._symbol_history("QQQ", 10.0)
        canonical_signal.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "yahoo_finance"}
        corrected_canonical = canonical_signal.copy()
        corrected_close = float(corrected_canonical.iloc[-1]["QQQ_Close"]) + 5.0
        corrected_canonical.iloc[-1, corrected_canonical.columns.get_loc("QQQ_Open")] = corrected_close
        corrected_canonical.iloc[-1, corrected_canonical.columns.get_loc("QQQ_High")] = corrected_close + 1.0
        corrected_canonical.iloc[-1, corrected_canonical.columns.get_loc("QQQ_Low")] = corrected_close - 1.0
        corrected_canonical.iloc[-1, corrected_canonical.columns.get_loc("QQQ_Close")] = corrected_close
        recovered_signal = self._symbol_history("QQQ", 20.0)
        recovered_signal.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "tradier"}
        recovered_signal.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = ["QQQ"]
        asset_history = self._symbol_history("AAA")
        risk_free_history = self._symbol_history("^IRX", -95.0)
        cfg = BacktestConfig(rsi_period=3)

        def process(db_path: str, canonical: pd.DataFrame, run_id: str) -> None:
            data = _strategy_data_from_authoritative_histories(
                asset_symbol="AAA",
                signal_symbol="QQQ",
                asset_history=asset_history,
                signal_history=recovered_signal,
                risk_free_history=risk_free_history,
            )
            session = _WorkflowStrategySession(db_path, history_observation_run_id=run_id)
            try:
                _process_asset_grid_for_db(
                    db_path,
                    data,
                    asset_history,
                    recovered_signal,
                    risk_free_history,
                    cfg,
                    "AAA",
                    "QQQ",
                    [30.0],
                    [1.5],
                    False,
                    strategy_session=session,
                    canonical_signal_history=canonical,
                )
            finally:
                session.close()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            process(db_path, canonical_signal, "initial-private-canonical-run")
            with closing(sqlite3.connect(db_path)) as conn:
                initial_state_digest = conn.execute(
                    "SELECT integrity_digest FROM strategy_state WHERE asset_symbol = 'AAA'"
                ).fetchone()[0]
                private_rsi_rows = conn.execute(
                    """
                    SELECT signal_symbol, rsi_period, date, close, avg_gain, avg_loss, rsi
                    FROM rsi_values
                    WHERE signal_symbol LIKE '@strategy-signal-v1/%'
                    ORDER BY date
                    """
                ).fetchall()

            start_indices: list[list[int]] = []
            real_run_grid_summary = storage_module.run_grid_summary

            def capture_start_indices(*args: object, **kwargs: object) -> tuple:
                start_indices.append(np.asarray(args[7], dtype=np.int64).tolist())
                return real_run_grid_summary(*args, **kwargs)

            with patch(
                "leveraged_trader.storage.run_grid_summary",
                side_effect=capture_start_indices,
            ):
                process(db_path, corrected_canonical, "corrected-unused-canonical-run")

            with closing(sqlite3.connect(db_path)) as conn:
                resumed_state_digest = conn.execute(
                    "SELECT integrity_digest FROM strategy_state WHERE asset_symbol = 'AAA'"
                ).fetchone()[0]
                resumed_private_rsi_rows = conn.execute(
                    """
                    SELECT signal_symbol, rsi_period, date, close, avg_gain, avg_loss, rsi
                    FROM rsi_values
                    WHERE signal_symbol LIKE '@strategy-signal-v1/%'
                    ORDER BY date
                    """
                ).fetchall()
                persisted_canonical_close = conn.execute(
                    "SELECT close FROM market_data WHERE symbol = 'QQQ' ORDER BY date DESC LIMIT 1"
                ).fetchone()[0]
            # Trusted authentication avoids both a duplicate canonical replay
            # and a no-work grid invocation when this pair's private inputs are
            # unchanged.
            self.assertEqual(start_indices, [])
            self.assertEqual(resumed_state_digest, initial_state_digest)
            self.assertEqual(resumed_private_rsi_rows, private_rsi_rows)
            self.assertEqual(persisted_canonical_close, corrected_close)

            summary, *_rest = _build_reports_for_db(
                db_path,
                pd.DataFrame([{"symbol": "AAA", "name": "AAA", "rsi_symbol": "QQQ"}]),
                cfg,
                {("AAA", "QQQ")},
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
            )
            self.assertEqual(summary["Asset"].tolist(), ["AAA"])

    def test_global_to_private_signal_transition_preserves_other_global_dependents(self) -> None:
        canonical_signal = self._symbol_history("QQQ", 10.0)
        canonical_signal.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "yahoo_finance"}
        recovered_signal = self._symbol_history("QQQ", 20.0)
        recovered_signal.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "tradier"}
        recovered_signal.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = ["QQQ"]
        risk_free_history = self._symbol_history("^IRX", -95.0)
        histories = {
            "AAA": self._symbol_history("AAA"),
            "BBB": self._symbol_history("BBB", 20.0),
        }
        cfg = BacktestConfig(rsi_period=3)

        def process(
            session: _WorkflowStrategySession,
            db_path: str,
            asset_symbol: str,
            pair_signal: pd.DataFrame,
        ) -> None:
            asset_history = histories[asset_symbol]
            data = _strategy_data_from_authoritative_histories(
                asset_symbol=asset_symbol,
                signal_symbol="QQQ",
                asset_history=asset_history,
                signal_history=pair_signal,
                risk_free_history=risk_free_history,
            )
            _process_asset_grid_for_db(
                db_path,
                data,
                asset_history,
                pair_signal,
                risk_free_history,
                cfg,
                asset_symbol,
                "QQQ",
                [30.0],
                [1.5],
                False,
                strategy_session=session,
                canonical_signal_history=canonical_signal,
            )

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            session = _WorkflowStrategySession(db_path, history_observation_run_id="global-to-private-run")
            try:
                process(session, db_path, "AAA", canonical_signal)
                process(session, db_path, "BBB", canonical_signal)
                with closing(sqlite3.connect(db_path)) as reader:
                    bbb_integrity_before = reader.execute(
                        "SELECT integrity_digest FROM strategy_state WHERE asset_symbol = 'BBB'"
                    ).fetchone()[0]
                process(session, db_path, "AAA", recovered_signal)
            finally:
                session.close()

            with closing(sqlite3.connect(db_path)) as conn:
                bbb_integrity_after = conn.execute(
                    "SELECT integrity_digest FROM strategy_state WHERE asset_symbol = 'BBB'"
                ).fetchone()[0]
                state_assets = {
                    row[0] for row in conn.execute("SELECT DISTINCT asset_symbol FROM strategy_state").fetchall()
                }
                global_signal_count = conn.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'QQQ'").fetchone()[
                    0
                ]
                private_signal_count = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol LIKE '@strategy-signal-v1/%'"
                ).fetchone()[0]
            self.assertEqual(bbb_integrity_after, bbb_integrity_before)
            self.assertEqual(state_assets, {"AAA", "BBB"})
            self.assertEqual(global_signal_count, len(canonical_signal))
            self.assertEqual(private_signal_count, len(recovered_signal))

            workflow_assets = pd.DataFrame(
                [{"symbol": symbol, "name": symbol, "rsi_symbol": "QQQ"} for symbol in ("AAA", "BBB")]
            )
            summary, *_rest = _build_reports_for_db(
                db_path,
                workflow_assets,
                cfg,
                {("AAA", "QQQ"), ("BBB", "QQQ")},
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
            )
            self.assertEqual(set(summary["Asset"]), {"AAA", "BBB"})

    def test_private_signal_tail_historical_to_asset_checkpoint_rebuilds_pair(self) -> None:
        def history(symbol: str, dates: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    f"{symbol}_Open": close,
                    f"{symbol}_High": close + 1.0,
                    f"{symbol}_Low": close - 1.0,
                    f"{symbol}_Close": close,
                    f"{symbol}_Volume": np.full(len(dates), 1_000_000.0),
                },
                index=dates,
            )

        asset_dates = pd.date_range("2026-01-02", periods=60, freq="B")
        initial_signal_dates = asset_dates[:20]
        extended_signal_dates = asset_dates[:30]
        asset_history = history("AAA", asset_dates, np.arange(100.0, 160.0))
        initial_recovery = history(
            "QQQ",
            initial_signal_dates,
            np.arange(100.0, 120.0),
        )
        extended_recovery = history(
            "QQQ",
            extended_signal_dates,
            np.concatenate([np.arange(100.0, 120.0), np.arange(118.0, 108.0, -1.0)]),
        )
        for recovered in (initial_recovery, extended_recovery):
            recovered.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "tradier"}
            recovered.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = ["QQQ"]
        canonical_signal = self._symbol_history("QQQ", 10.0)
        canonical_signal.index = canonical_signal.index + pd.DateOffset(months=4)
        canonical_signal.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "yahoo_finance"}
        asset_risk_free = history("^IRX", asset_dates, np.full(len(asset_dates), 5.0))
        canonical_risk_free = history(
            "^IRX",
            canonical_signal.index,
            np.full(len(canonical_signal), 5.0),
        )
        risk_free_history = pd.concat([asset_risk_free, canonical_risk_free]).sort_index()
        cfg = BacktestConfig(rsi_period=3)

        def process(db_path: str, pair_signal: pd.DataFrame, run_id: str) -> None:
            data = _strategy_data_from_authoritative_histories(
                asset_symbol="AAA",
                signal_symbol="QQQ",
                asset_history=asset_history,
                signal_history=pair_signal,
                risk_free_history=risk_free_history,
            )
            session = _WorkflowStrategySession(db_path, history_observation_run_id=run_id)
            try:
                _process_asset_grid_for_db(
                    db_path,
                    data,
                    asset_history,
                    pair_signal,
                    risk_free_history,
                    cfg,
                    "AAA",
                    "QQQ",
                    [30.0],
                    [1.5],
                    False,
                    strategy_session=session,
                    canonical_signal_history=canonical_signal,
                )
            finally:
                session.close()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
            process(db_path, initial_recovery, "initial-private-tail-run")
            with closing(sqlite3.connect(db_path)) as conn:
                initial_digest = conn.execute(
                    "SELECT integrity_digest FROM strategy_state WHERE asset_symbol = 'AAA'"
                ).fetchone()[0]

            process(db_path, extended_recovery, "extended-private-tail-run")
            with closing(sqlite3.connect(db_path)) as conn:
                extended_digest = conn.execute(
                    "SELECT integrity_digest FROM strategy_state WHERE asset_symbol = 'AAA'"
                ).fetchone()[0]
                scoped_count = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE symbol LIKE '@strategy-signal-v1/%'"
                ).fetchone()[0]
            self.assertNotEqual(extended_digest, initial_digest)
            self.assertEqual(scoped_count, len(extended_recovery))

            # Repeating the same complete snapshot must resume cleanly rather
            # than falling into a reconstruction/rollback loop.
            process(db_path, extended_recovery.copy(), "stable-private-tail-run")
            with closing(sqlite3.connect(db_path)) as conn:
                stable_digest = conn.execute(
                    "SELECT integrity_digest FROM strategy_state WHERE asset_symbol = 'AAA'"
                ).fetchone()[0]
            self.assertEqual(stable_digest, extended_digest)

            summary, *_rest = _build_reports_for_db(
                db_path,
                pd.DataFrame([{"symbol": "AAA", "name": "AAA", "rsi_symbol": "QQQ"}]),
                cfg,
                {("AAA", "QQQ")},
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
            )
            self.assertEqual(summary["Asset"].tolist(), ["AAA"])

    def test_disjoint_signal_calendar_is_rejected_without_tradier(self) -> None:
        market_data_session = _WorkflowMarketDataSession()
        yahoo_signal_history = self._symbol_history("QQQ", 10.0)
        yahoo_signal_history.index = yahoo_signal_history.index + pd.DateOffset(months=3)

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                return_value=self._symbol_history("AAA"),
            ),
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ),
            patch(
                "leveraged_trader.workflow.load_signal_history",
                return_value=yahoo_signal_history,
            ),
            patch("leveraged_trader.market_data._load_tradier_fallback_frames") as fallback,
            patch("leveraged_trader.workflow._process_asset_grid_for_db") as process,
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "AAA", "QQQ")],
                    concurrency=1,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(auto_adjust=False),
                    tradier_cfg=TradierMarketDataConfig(enabled=False),
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=market_data_session,
                )
            )

        self.assertEqual([result.status for result in results], ["skipped"])
        self.assertIn("No daily rows overlapped the retained AAA calendar", results[0].message)
        fallback.assert_not_called()
        process.assert_not_called()
        self.assertIs(market_data_session.signal_histories["QQQ"], yahoo_signal_history)
        self.assertNotIn(("AAA", "QQQ"), market_data_session.calendar_signal_histories)
        self.assertIn(("AAA", "QQQ"), market_data_session.calendar_signal_failures)

    def test_calendar_recovery_failure_does_not_poison_shared_signal_snapshot(self) -> None:
        market_data_session = _WorkflowMarketDataSession()
        yahoo_signal_history = self._symbol_history("QQQ", 10.0)
        yahoo_signal_history.index = yahoo_signal_history.index + pd.DateOffset(months=3)
        yahoo_signal_history.attrs[MARKET_DATA_PROVIDERS_ATTR] = {"QQQ": "yahoo_finance"}
        disjoint_tradier_history = self._symbol_history("QQQ", 20.0)
        disjoint_tradier_history.index = disjoint_tradier_history.index + pd.DateOffset(months=6)
        processed_assets: list[str] = []

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        def load_asset(symbol: str, **_kwargs: object) -> pd.DataFrame:
            history = self._symbol_history(symbol)
            if symbol == "BBB":
                history.index = yahoo_signal_history.index.copy()
            return history

        def process(*args: object, **_kwargs: object) -> None:
            processed_assets.append(str(args[6]))

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch("leveraged_trader.workflow.load_symbol_history", side_effect=load_asset),
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ),
            patch(
                "leveraged_trader.workflow.load_signal_history",
                return_value=yahoo_signal_history,
            ) as signal_download,
            patch(
                "leveraged_trader.market_data._load_tradier_fallback_frames",
                return_value=({"QQQ": disjoint_tradier_history}, {}),
            ) as fallback,
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=process),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "AAA", "QQQ"), AssetRunJob(2, "BBB", "QQQ")],
                    concurrency=1,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(auto_adjust=False),
                    tradier_cfg=TradierMarketDataConfig(access_token="token"),
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=market_data_session,
                )
            )

        self.assertEqual([result.status for result in results], ["skipped", "done"])
        self.assertIn("No daily rows overlapped the retained AAA calendar", results[0].message)
        self.assertEqual(processed_assets, ["BBB"])
        signal_download.assert_called_once()
        fallback.assert_called_once()
        self.assertIs(market_data_session.signal_histories["QQQ"], yahoo_signal_history)
        self.assertNotIn("QQQ", market_data_session.signal_failures)
        self.assertIn(("AAA", "QQQ"), market_data_session.calendar_signal_failures)

    def test_asset_pipeline_shares_one_symbol_snapshot_across_asset_and_signal_roles(self) -> None:
        market_data_session = _WorkflowMarketDataSession()
        canonical_x = self._symbol_history("X", 10.0)
        processed_histories: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        def load_asset(symbol: str, **_kwargs: object) -> pd.DataFrame:
            if symbol == "X":
                raise AssertionError("X was downloaded again after becoming the canonical signal snapshot")
            return self._symbol_history(symbol)

        def load_signal(symbol: str, **_kwargs: object) -> pd.DataFrame:
            return canonical_x if symbol == "X" else self._symbol_history(symbol, 20.0)

        def process(*args: object, **_kwargs: object) -> None:
            processed_histories.append((str(args[6]), args[2], args[3]))

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch("leveraged_trader.workflow.load_symbol_history", side_effect=load_asset) as asset_download,
            patch("leveraged_trader.workflow.load_signal_history", side_effect=load_signal) as signal_download,
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ),
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=process),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "A", "X"), AssetRunJob(2, "X", "Y")],
                    concurrency=1,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=market_data_session,
                )
            )

        self.assertEqual([result.status for result in results], ["done", "done"])
        self.assertEqual([call.args[0] for call in asset_download.call_args_list], ["A"])
        self.assertEqual([call.args[0] for call in signal_download.call_args_list], ["X", "Y"])
        self.assertEqual([asset for asset, _asset_history, _signal_history in processed_histories], ["A", "X"])
        self.assertIs(processed_histories[0][2], canonical_x)
        self.assertIs(processed_histories[1][1], canonical_x)

    def test_concurrent_asset_pipeline_downloads_cross_role_symbol_once(self) -> None:
        market_data_session = _WorkflowMarketDataSession()
        histories: dict[str, pd.DataFrame] = {}
        download_counts: dict[str, int] = {}
        download_counts_lock = threading.Lock()
        processed_histories: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        def load_once(symbol: str) -> pd.DataFrame:
            with download_counts_lock:
                download_counts[symbol] = download_counts.get(symbol, 0) + 1
                history = histories.setdefault(symbol, self._symbol_history(symbol, 10.0))
            if symbol == "X":
                # Let A/X reach its signal phase while X/Y still owns X's
                # canonical history lock. Without the cross-role lock this
                # reliably starts a second provider request for X.
                time.sleep(0.05)
            return history

        def process(*args: object, **_kwargs: object) -> None:
            processed_histories[str(args[6])] = (args[2], args[3])

        with (
            patch(
                "leveraged_trader.workflow._prepare_asset_run",
                side_effect=plan,
            ),
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                side_effect=lambda symbol, **_kwargs: load_once(symbol),
            ),
            patch(
                "leveraged_trader.workflow.load_signal_history",
                side_effect=lambda symbol, **_kwargs: load_once(symbol),
            ),
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ),
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=process),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "A", "X"), AssetRunJob(2, "X", "Y")],
                    concurrency=2,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=market_data_session,
                )
            )

        self.assertEqual([result.status for result in results], ["done", "done"])
        self.assertEqual(download_counts["X"], 1)
        self.assertIs(processed_histories["A"][1], processed_histories["X"][0])

    def test_self_rsi_asset_reuses_an_existing_canonical_signal_history(self) -> None:
        market_data_session = _WorkflowMarketDataSession()
        canonical_history = self._symbol_history("AAA", 10.0)
        market_data_session.signal_histories["AAA"] = canonical_history
        processed_histories: list[tuple[pd.DataFrame, pd.DataFrame]] = []

        def process(*args: object, **_kwargs: object) -> None:
            processed_histories.append((args[2], args[3]))

        with (
            patch(
                "leveraged_trader.workflow._prepare_asset_run",
                return_value=AssetRunPlan(
                    asset_symbol="AAA",
                    signal_symbol="AAA",
                    rebuild=False,
                    start=None,
                    action="Updating",
                    start_label="earliest overlapping history",
                ),
            ),
            patch("leveraged_trader.workflow.load_symbol_history") as asset_download,
            patch("leveraged_trader.workflow.load_signal_history") as signal_download,
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ),
            patch("leveraged_trader.workflow._process_asset_grid_for_db", side_effect=process),
        ):
            results = asyncio.run(
                _run_asset_pipeline(
                    jobs=[AssetRunJob(1, "AAA", "AAA")],
                    concurrency=1,
                    db_path="unused.sqlite",
                    mode="update",
                    base_cfg=BacktestConfig(),
                    tradier_cfg=None,
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    asset_progress=None,
                    phase_timings=WorkflowPhaseTimings(),
                    market_data_session=market_data_session,
                )
            )

        self.assertEqual([result.status for result in results], ["done"])
        asset_download.assert_not_called()
        signal_download.assert_not_called()
        self.assertEqual(len(processed_histories), 1)
        self.assertIs(processed_histories[0][0], canonical_history)
        self.assertIs(processed_histories[0][1], canonical_history)

    def test_known_benchmark_failure_skips_later_asset_downloads(self) -> None:
        jobs = [
            AssetRunJob(1, "AAA", "QQQ"),
            AssetRunJob(2, "BBB", "QQQ"),
            AssetRunJob(3, "CCC", "QQQ"),
        ]

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                side_effect=lambda symbol, **_kwargs: self._symbol_history(symbol),
            ) as asset_download,
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                side_effect=MarketDataDownloadError({"^IRX": "timed out"}),
            ) as benchmark_download,
            patch("leveraged_trader.workflow.load_signal_history") as signal_download,
        ):
            results = asyncio.run(
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

        self.assertEqual([result.status for result in results], ["skipped"] * 3)
        self.assertEqual(asset_download.call_count, 1)
        self.assertEqual(benchmark_download.call_count, 1)
        signal_download.assert_not_called()

    def test_known_full_signal_failure_reuses_one_bounded_retry_for_matching_calendars(self) -> None:
        jobs = [
            AssetRunJob(1, "AAA", "QQQ"),
            AssetRunJob(2, "BBB", "QQQ"),
            AssetRunJob(3, "CCC", "QQQ"),
        ]

        def plan(*args: object, **_kwargs: object) -> AssetRunPlan:
            return AssetRunPlan(
                asset_symbol=str(args[3]),
                signal_symbol=str(args[4]),
                rebuild=False,
                start=None,
                action="Updating",
                start_label="earliest overlapping history",
            )

        with (
            patch("leveraged_trader.workflow._prepare_asset_run", side_effect=plan),
            patch(
                "leveraged_trader.workflow.load_symbol_history",
                side_effect=lambda symbol, **_kwargs: self._symbol_history(symbol),
            ) as asset_download,
            patch(
                "leveraged_trader.workflow.load_risk_free_history",
                return_value=self._symbol_history("^IRX", -95.0),
            ) as benchmark_download,
            patch(
                "leveraged_trader.workflow.load_signal_history",
                side_effect=MarketDataDownloadError({"QQQ": "timed out"}),
            ) as signal_download,
        ):
            results = asyncio.run(
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

        self.assertEqual([result.status for result in results], ["skipped"] * 3)
        self.assertEqual(asset_download.call_count, 3)
        self.assertEqual(benchmark_download.call_count, 1)
        self.assertEqual(signal_download.call_count, 2)

    def test_asset_pipeline_drains_preparation_and_asset_data_failures(self) -> None:
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
                side_effect=AssetMarketDataError("asset data failed"),
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
        self.assertEqual([result.message for result in results], ["provider failed", "asset data failed"])
        self.assertEqual(asset_progress.finish_asset.call_count, 2)

    def test_new_self_signal_asset_reports_rsi_warmup_instead_of_failure(self) -> None:
        history = self._symbol_history("MNGU").iloc[:11]
        outcome = PreparedAssetRun(
            job=AssetRunJob(1, "MNGU", "MNGU", workflow="Long"),
            plan=AssetRunPlan(
                asset_symbol="MNGU",
                signal_symbol="MNGU",
                rebuild=True,
                start=None,
                action="Rebuilding",
                start_label="earliest overlapping history",
            ),
            data=history,
            asset_history=history,
            signal_history=history,
            risk_free_history=self._symbol_history("^IRX", -95.0),
            canonical_signal_history=history,
        )

        with patch(
            "leveraged_trader.workflow._process_asset_grid_for_db",
            side_effect=AssetMarketDataError(
                "No finite RSI observations for MNGU align with its finalized market sessions."
            ),
        ):
            result = asyncio.run(
                _complete_workflow_asset(
                    outcome,
                    db_path="unused.sqlite",
                    base_cfg=BacktestConfig(rsi_period=14),
                    buy_rsi_values=[30.0],
                    profit_target_values=[1.5],
                    rsi_entry_rule="lower",
                )
            )

        self.assertEqual(result.status, "warming_up")
        self.assertEqual(result.rows_processed, 11)
        self.assertEqual(
            result.message,
            "RSI warm-up: 11 of 15 settled MNGU observations are available; 4 more required.",
        )

    def test_asset_pipeline_propagates_database_failures(self) -> None:
        job = AssetRunJob(1, "AAA", "AAA")
        data = pd.DataFrame({"Close": [100.0]})
        prepared = PreparedAssetRun(
            job=job,
            plan=AssetRunPlan(
                asset_symbol="AAA",
                signal_symbol="AAA",
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

        async def fake_prepare(**_kwargs: object) -> PreparedAssetRun:
            return prepared

        with (
            patch("leveraged_trader.workflow._prepare_workflow_asset", new=fake_prepare),
            patch(
                "leveraged_trader.workflow._process_asset_grid_for_db",
                side_effect=sqlite3.DatabaseError("database failed"),
            ),
            self.assertRaisesRegex(sqlite3.DatabaseError, "database failed"),
        ):
            asyncio.run(
                _run_asset_pipeline(
                    jobs=[job],
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

    def test_asset_pipeline_propagates_programming_value_errors(self) -> None:
        job = AssetRunJob(1, "AAA", "AAA")
        data = pd.DataFrame({"Close": [100.0]})
        prepared = PreparedAssetRun(
            job=job,
            plan=AssetRunPlan(
                asset_symbol="AAA",
                signal_symbol="AAA",
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

        async def fake_prepare(**_kwargs: object) -> PreparedAssetRun:
            return prepared

        with (
            patch("leveraged_trader.workflow._prepare_workflow_asset", new=fake_prepare),
            patch(
                "leveraged_trader.workflow._process_asset_grid_for_db",
                side_effect=ValueError("unexpected invariant"),
            ),
            self.assertRaisesRegex(ValueError, "unexpected invariant"),
        ):
            asyncio.run(
                _run_asset_pipeline(
                    jobs=[job],
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

    def test_asset_preparation_propagates_database_failures_before_download(self) -> None:
        job = AssetRunJob(1, "AAA", "AAA")
        symbol_download = Mock()
        with (
            patch(
                "leveraged_trader.workflow._prepare_asset_run",
                side_effect=sqlite3.DatabaseError("database failed"),
            ),
            patch("leveraged_trader.workflow.load_symbol_history", symbol_download),
            self.assertRaisesRegex(sqlite3.DatabaseError, "database failed"),
        ):
            asyncio.run(
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

        symbol_download.assert_not_called()

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

    def test_inactive_holdings_are_accounted_separately_from_sell_orders(self) -> None:
        managed_positions = pd.DataFrame(
            [
                {
                    "id": 115,
                    "workflow": "Long",
                    "symbol": "LACG",
                    "alpaca_asset_id": "asset-lacg",
                    "sell_status": "broker_inactive",
                    "filled_qty": 49,
                    "filled_avg_price": 4.12,
                    "sold_qty": 0,
                    "remaining_qty": 49,
                    "target_sell_price": 6.18,
                    "closed_at": None,
                    "updated_at": "2026-09-14 14:12:42",
                },
                {
                    "id": 116,
                    "workflow": "Long",
                    "symbol": "DONE",
                    "sell_status": "broker_inactive",
                    "filled_qty": 1,
                    "filled_avg_price": 10,
                    "sold_qty": 0,
                    "remaining_qty": 1,
                    "target_sell_price": 15,
                    "closed_at": "2026-09-13 14:00:00",
                },
                {
                    "id": 117,
                    "workflow": "Long",
                    "symbol": "RECOVERED",
                    "sell_status": "accepted",
                    "filled_qty": 1,
                    "filled_avg_price": 10,
                    "sold_qty": 0,
                    "remaining_qty": 1,
                    "target_sell_price": 15,
                    "closed_at": None,
                },
            ]
        )
        reconciliation_results = pd.DataFrame(
            [
                {"Position ID": 115, "Asset": "LACG", "Action": "sell", "Status": "broker_inactive"},
                {"Position ID": 117, "Asset": "TQQQ", "Action": "sell", "Status": "new"},
                {"Position ID": 118, "Asset": "UPRO", "Action": "buy", "Status": "filled"},
            ]
        )

        inactive = _alpaca_inactive_holdings_report(managed_positions)
        sell_orders = _alpaca_sell_order_results(reconciliation_results)

        self.assertEqual(inactive["Asset"].tolist(), ["LACG"])
        self.assertEqual(inactive["Status"].tolist(), ["retained"])
        self.assertEqual(inactive["Qty"].tolist(), [49])
        self.assertAlmostEqual(inactive.loc[0, "Estimated Buy Cost"], 201.88)
        self.assertAlmostEqual(inactive.loc[0, "Target Value"], 302.82)
        self.assertIn("monitored for reactivation", inactive.loc[0, "Message"])
        self.assertEqual(sell_orders["Asset"].tolist(), ["TQQQ"])

    def test_failed_reconciliation_keeps_inactive_holdings_in_dedicated_terminal_lane(self) -> None:
        from leveraged_trader import workflow as workflow_module

        reconciliation_results = pd.DataFrame(
            [
                {"Position ID": 115, "Asset": "LACG", "Action": "sell", "Status": "broker_inactive"},
                {"Position ID": 116, "Asset": "TQQQ", "Action": "sell", "Status": "error"},
            ]
        )
        inactive_holdings = pd.DataFrame([{"Position ID": 115, "Asset": "LACG", "Status": "retained"}])
        failure = AlpacaReconciliationError("reconciliation failed", reconciliation_results)
        reporter = Mock(spec=WorkflowReporter)

        with (
            patch.object(workflow_module, "_persist_alpaca_reconciliation_snapshot"),
            patch.object(
                workflow_module,
                "_load_alpaca_inactive_holdings_for_db",
                return_value=inactive_holdings,
            ),
        ):
            workflow_module._persist_alpaca_reconciliation_failure(
                failure,
                db_path="state.sqlite",
                output_dir="outputs",
                publication=Mock(),
                reporter=reporter,
                alpaca_cfg=AlpacaOrderConfig(),
            )

        rendered_reconciliation = reporter.reconciliation.call_args.args[0]
        self.assertEqual(rendered_reconciliation["Asset"].tolist(), ["TQQQ"])
        reporter.inactive_holdings.assert_called_once()
        pd.testing.assert_frame_equal(
            reporter.inactive_holdings.call_args.args[0],
            inactive_holdings,
        )

    def test_write_workflow_outputs_uses_terminal_display_ids_for_alpaca_tables(self) -> None:
        managed_positions = pd.DataFrame(
            [
                {"id": 2, "symbol": "KLAG", "buy_client_order_id": "buy-KLAG"},
                {"id": 3, "symbol": "MPG", "buy_client_order_id": "buy-MPG", "closed_at": "2026-01-03"},
                {"id": 5, "symbol": "AXTU", "buy_client_order_id": "buy-AXTU"},
                {
                    "id": 7,
                    "workflow": "Long",
                    "symbol": "LACG",
                    "alpaca_asset_id": "asset-lacg",
                    "buy_client_order_id": "buy-LACG",
                    "sell_status": "broker_inactive",
                    "filled_qty": 49,
                    "filled_avg_price": 4.12,
                    "sold_qty": 0,
                    "remaining_qty": 49,
                    "target_sell_price": 6.18,
                    "updated_at": "2026-09-14 14:12:42",
                },
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
                {
                    "Position ID": 7,
                    "Workflow": "Long",
                    "Asset": "LACG",
                    "Action": "sell",
                    "Status": "broker_inactive",
                    "Buy Client Order ID": "buy-LACG",
                    "Sell Client Order ID": "sell-LACG",
                    "Qty": 49,
                    "Limit Price": 6.18,
                    "Alpaca Order ID": None,
                    "Message": "inactive holding retained at Alpaca",
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
            written_sell_orders = pd.read_csv(output_dir / "alpaca_sell_order_results.csv")
            written_inactive = pd.read_csv(output_dir / "alpaca_inactive_holdings.csv")
            written_orders = pd.read_csv(output_dir / "alpaca_order_results.csv")
            output = output_buffer.getvalue()

        axtu_lines = [line for line in output.splitlines() if "AXTU" in line]
        self.assertEqual(len(axtu_lines), 2)
        for line in axtu_lines:
            self.assertRegex(line, r"^\s*3\s+AXTU\b")
        self.assertEqual(written_reconciliation["Position ID"].tolist(), [2, 5, 7])
        self.assertEqual(written_sell_orders["Position ID"].tolist(), [2, 5])
        self.assertEqual(written_inactive["Asset"].tolist(), ["LACG"])
        self.assertEqual(written_inactive["Status"].tolist(), ["retained"])
        self.assertIn("Broker-Retained Inactive Alpaca Holdings", output)
        self.assertEqual(sum("LACG" in line for line in output.splitlines()), 1)
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
        self.assertNotIn("Timing (overlap-aware):", output)
        self.assertNotIn("Market-data batches:", output)
        self.assertIn("Workflow finished in", output)
        self.assertNotIn("Workflow Benchmark", output)
        self.assertEqual(output.rstrip().splitlines()[-1], "\u2500" * 100)

    def test_write_workflow_outputs_prints_detailed_timings_when_enabled(self) -> None:
        phase_timings = WorkflowPhaseTimings()
        phase_timings.add("download", 266.6)
        phase_timings.add("state_validation", 147.92)
        phase_timings.add("db_sync", 52.6)
        phase_timings.add("report_generation", 39.85)
        phase_timings.add("alpaca", 0.02)
        asset_run_results = [
            AssetRunResult(
                workflow_idx=1,
                asset_symbol="TQQQ",
                signal_symbol="QQQ",
                action="Updating",
                rows_processed=12,
                status="done",
                message="Processed 12 rows",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp, patch("leveraged_trader.workflow.time") as mock_time:
            mock_time.perf_counter.side_effect = [10.0, 10.0]
            output_buffer = io.StringIO()
            reporter = WorkflowReporter(
                console=Console(file=output_buffer, width=240, color_system=None, no_color=True)
            )
            _write_workflow_outputs(
                mode="update",
                db_path=str(Path(tmp) / "state.sqlite"),
                base_cfg=BacktestConfig(),
                buy_rsi_values=[30.0],
                profit_target_values=[1.5],
                alpaca_cfg=AlpacaOrderConfig(),
                output_dir=str(Path(tmp) / "outputs"),
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
                research_outputs_published=True,
                broker_snapshot_committed=True,
                market_data_batch_count=29,
                market_data_individual_retry_count=1,
                show_timings=True,
            )
            output = output_buffer.getvalue()

        self.assertIn(
            "Timing (overlap-aware): downloads 4m 26.60s, state verification 2m 27.92s, "
            "grid 0.00s, database 52.60s, reports 39.85s, Alpaca 0.02s.",
            output,
        )
        self.assertIn(
            "Market-data batches: 29; individual retries: 1; rebuilt assets: 0; updated assets: 1.",
            output,
        )

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

    @patch("leveraged_trader.workflow.reconcile_alpaca_managed_positions")
    @patch("leveraged_trader.workflow.migrate_alpaca_managed_position_symbols")
    def test_symbol_migration_failure_uses_cfg_aware_bounded_diagnostic(
        self,
        mock_migrate: Mock,
        mock_reconcile: Mock,
    ) -> None:
        secret = "workflow-opaque-secret-123456789"
        mock_migrate.side_effect = RuntimeError(f"asset lookup reflected credential={secret}\n" + "X" * 10_000)
        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="paper-key",
            api_secret_key=secret,
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)
                save_alpaca_managed_buy_order(
                    conn,
                    workflow="Long",
                    symbol="TQQQ",
                    signal_symbol="QQQ",
                    buy_rsi=30,
                    profit_target_multiple=1.5,
                    buy_signal_date="2026-01-02",
                    buy_client_order_id="rsi-buy-TQQQ-20260102",
                    buy_alpaca_order_id="buy-1",
                    buy_submitted_at="2026-01-02T14:30:00Z",
                    buy_status="accepted",
                )

            with self.assertRaises(AlpacaReconciliationError) as raised:
                _reconcile_alpaca_managed_positions_for_db(db_path, cfg)

        message = raised.exception.results.loc[0, "Message"]
        self.assertNotIn(secret, message)
        self.assertNotIn("\n", message)
        self.assertIn("[redacted credential]", message)
        self.assertLessEqual(len(message), 600)
        mock_reconcile.assert_not_called()

    @patch("leveraged_trader.workflow.reconcile_alpaca_managed_positions")
    @patch("leveraged_trader.workflow.migrate_alpaca_managed_position_symbols")
    def test_symbol_migration_empty_error_rescans_type_name_collision(
        self,
        mock_migrate: Mock,
        mock_reconcile: Mock,
    ) -> None:
        credential = "ValueError"
        mock_migrate.side_effect = ValueError("")
        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="paperkey",
            api_secret_key=credential,
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            with closing(sqlite3.connect(db_path)) as conn, conn:
                init_state_db(conn)

            with self.assertRaises(AlpacaReconciliationError) as raised:
                _reconcile_alpaca_managed_positions_for_db(db_path, cfg)

        public_diagnostic = str(raised.exception) + raised.exception.results.to_string()
        rendered_traceback = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn(credential, public_diagnostic)
        self.assertNotIn(credential, rendered_traceback)
        self.assertIn("[redacted credential]", public_diagnostic)
        self.assertIsNone(raised.exception.__cause__)
        mock_reconcile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
