from __future__ import annotations

import asyncio
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from .alpaca import reconcile_alpaca_managed_positions, submit_alpaca_paper_buy_orders
from .benchmark import WorkflowPhase, WorkflowPhaseTimings, WorkflowTimer
from .config import (
    RISK_FREE_SYMBOL,
    AlpacaOrderConfig,
    BacktestConfig,
    TradierMarketDataConfig,
    UniverseConfig,
)
from .market_data import (
    TRADIER_RECOVERED_SYMBOLS_ATTR,
    load_risk_free_history,
    load_signal_history,
    load_strategy_data,
    load_symbol_history,
)
from .output import AssetProgress, WorkflowReporter
from .reports import (
    build_alpaca_realized_pnl_summary,
    build_buy_signal_report,
    build_sell_signal_report,
    summarize_saved_results,
)
from .storage import (
    active_alpaca_managed_symbols,
    init_state_db,
    load_alpaca_managed_positions,
    process_asset_grid,
    save_workflow_assets,
    strategy_config_fingerprint,
    strategy_state_matches_config,
)
from .universe import determine_workflow_assets

DEFAULT_WORKFLOW_CONCURRENCY = 4
SQLITE_BUSY_TIMEOUT_MS = 60_000


def _validate_grid_values(name: str, values: list[float]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain only finite numeric values.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only finite numeric values.") from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"{name} must contain only finite numeric values.")


def _validate_optimization_grids(
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> None:
    _validate_grid_values("buy_rsi_values", buy_rsi_values)
    _validate_grid_values("profit_target_values", profit_target_values)


class WorkflowRunError(RuntimeError):
    """Raised when a workflow run cannot produce any usable strategy result."""


@dataclass(frozen=True)
class AssetRunPlan:
    asset_symbol: str
    signal_symbol: str
    rebuild: bool
    start: str | None
    action: str
    start_label: str


@dataclass(frozen=True)
class AssetRunResult:
    workflow_idx: int
    asset_symbol: str
    signal_symbol: str
    action: str
    rows_processed: int | None
    status: str
    message: str

    def as_output_row(self) -> dict[str, object]:
        row = asdict(self)
        return {
            "Workflow #": row["workflow_idx"],
            "Asset": row["asset_symbol"],
            "RSI Symbol": row["signal_symbol"],
            "Action": row["action"],
            "Rows": row["rows_processed"],
            "Status": row["status"],
            "Message": row["message"],
        }


@dataclass(frozen=True)
class AssetRunJob:
    workflow_idx: int
    asset_symbol: str
    signal_symbol: str


@dataclass(frozen=True)
class PreparedAssetRun:
    job: AssetRunJob
    plan: AssetRunPlan
    data: pd.DataFrame
    asset_history: pd.DataFrame
    signal_history: pd.DataFrame
    risk_free_history: pd.DataFrame


class _WorkflowStrategySession:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._owner_thread_id: int | None = None
        self._data_version: int | None = None
        self._synchronized_histories: dict[str, pd.DataFrame] = {}

    def _connection(self) -> sqlite3.Connection:
        thread_id = threading.get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = thread_id
        elif self._owner_thread_id != thread_id:
            raise RuntimeError("Workflow strategy session used from multiple threads.")
        if self._conn is None:
            self._conn = sqlite3.connect(
                self._db_path,
                timeout=SQLITE_BUSY_TIMEOUT_MS / 1000,
            )
            self._conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        return self._conn

    @contextmanager
    def immediate_transaction(self) -> sqlite3.Connection:
        conn = self._connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            data_version = int(conn.execute("PRAGMA data_version").fetchone()[0])
            if self._data_version is None:
                self._data_version = data_version
            elif data_version != self._data_version:
                self._synchronized_histories.clear()
                self._data_version = data_version
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def presynchronized_symbols(
        self,
        authoritative_histories: dict[str, pd.DataFrame],
    ) -> set[str]:
        return {
            symbol
            for symbol, history in authoritative_histories.items()
            if self._synchronized_histories.get(symbol) is history
        }

    def mark_synchronized(
        self,
        authoritative_histories: dict[str, pd.DataFrame],
    ) -> None:
        self._synchronized_histories.update(authoritative_histories)

    def close(self) -> None:
        if self._conn is not None:
            self._connection().close()
        self._conn = None
        self._owner_thread_id = None
        self._data_version = None
        self._synchronized_histories.clear()


@contextmanager
def _state_connection(db_path: str, *, immediate: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
    try:
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def _initialize_state_db(db_path: str) -> None:
    with _state_connection(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        init_state_db(conn)


async def _timed_run_blocking(
    phase_timings: WorkflowPhaseTimings | None,
    phase: WorkflowPhase,
    func: Callable[..., Any],
    /,
    *args: object,
    executor: Executor | None = None,
    **kwargs: object,
) -> Any:
    if phase_timings is None:
        return await _run_blocking(executor, func, *args, **kwargs)
    started = time.perf_counter()
    try:
        return await _run_blocking(executor, func, *args, **kwargs)
    finally:
        phase_timings.add(phase, time.perf_counter() - started)


async def _run_blocking(
    executor: Executor | None,
    func: Callable[..., Any],
    /,
    *args: object,
    **kwargs: object,
) -> Any:
    if executor is None:
        return await asyncio.to_thread(func, *args, **kwargs)
    call = partial(func, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(executor, call)


def _load_or_refresh_workflow_assets_for_db(
    db_path: str,
    universe_cfg: UniverseConfig,
) -> pd.DataFrame:
    workflow_assets = determine_workflow_assets(universe_cfg)
    with _state_connection(db_path) as conn:
        save_workflow_assets(conn, workflow_assets)
    return workflow_assets


def _reconcile_alpaca_managed_positions_for_db(
    db_path: str,
    alpaca_cfg: AlpacaOrderConfig,
) -> pd.DataFrame:
    with _state_connection(db_path) as conn:
        return reconcile_alpaca_managed_positions(conn, alpaca_cfg)


def _prepare_asset_run(
    db_path: str,
    mode: str,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> AssetRunPlan:
    with _state_connection(db_path) as conn:
        rebuild_asset = mode == "rebuild"
        if mode == "update" and not strategy_state_matches_config(
            conn,
            asset_symbol,
            signal_symbol,
            base_cfg,
            buy_rsi_values,
            profit_target_values,
        ):
            rebuild_asset = True

        # Fetch canonical history even in update mode.  Auto-adjusted asset and
        # benchmark series can revise old sessions; the state layer compares the
        # complete input before deciding whether a compact resume is still safe.
        start: str | None = None

    action = "Rebuilding" if rebuild_asset else "Updating"
    start_label = start if start is not None else "earliest overlapping history"
    return AssetRunPlan(
        asset_symbol=asset_symbol,
        signal_symbol=signal_symbol,
        rebuild=rebuild_asset,
        start=start,
        action=action,
        start_label=start_label,
    )


def _process_asset_grid_for_db(
    db_path: str,
    data: pd.DataFrame,
    asset_history: pd.DataFrame,
    signal_history: pd.DataFrame,
    risk_free_history: pd.DataFrame,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rebuild: bool,
    phase_timings: WorkflowPhaseTimings | None = None,
    strategy_session: _WorkflowStrategySession | None = None,
) -> None:
    # This transaction covers market-data synchronization, global benchmark
    # invalidation, and rebuilt state.  A second process waits here and then
    # observes the new generation/config instead of writing a stale resume.
    grid_compute_seconds = 0.0

    def observe_grid_compute(elapsed_seconds: float) -> None:
        nonlocal grid_compute_seconds
        grid_compute_seconds += elapsed_seconds
        if phase_timings is not None:
            phase_timings.add("grid_compute", elapsed_seconds)

    transaction_started = time.perf_counter()
    authoritative_histories = {
        asset_symbol: asset_history,
        signal_symbol: signal_history,
        RISK_FREE_SYMBOL: risk_free_history,
    }
    shared_histories = {
        signal_symbol: signal_history,
        RISK_FREE_SYMBOL: risk_free_history,
    }
    transaction = (
        strategy_session.immediate_transaction()
        if strategy_session is not None
        else _state_connection(db_path, immediate=True)
    )
    try:
        with transaction as conn:
            process_asset_grid(
                conn,
                data,
                base_cfg,
                asset_symbol,
                signal_symbol,
                buy_rsi_values,
                profit_target_values,
                rebuild=rebuild,
                signal_history=signal_history,
                authoritative_histories=authoritative_histories,
                presynchronized_authoritative_symbols=(
                    strategy_session.presynchronized_symbols(shared_histories)
                    if strategy_session is not None
                    else None
                ),
                commit=False,
                strategy_fingerprint=strategy_config_fingerprint(
                    base_cfg,
                    buy_rsi_values,
                    profit_target_values,
                ),
                grid_compute_observer=observe_grid_compute if phase_timings is not None else None,
            )
        if strategy_session is not None:
            strategy_session.mark_synchronized(shared_histories)
    finally:
        if phase_timings is not None:
            transaction_seconds = max(0.0, time.perf_counter() - transaction_started)
            phase_timings.add("db_sync", max(0.0, transaction_seconds - grid_compute_seconds))


def _build_reports_for_db(
    db_path: str,
    workflow_assets: pd.DataFrame,
    base_cfg: BacktestConfig,
    processed_asset_pairs: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report_assets = workflow_assets[
        workflow_assets.apply(
            lambda row: (str(row["symbol"]), str(row["rsi_symbol"])) in processed_asset_pairs,
            axis=1,
        )
    ].copy()
    with _state_connection(db_path) as conn:
        optimization_summary, curves = summarize_saved_results(conn, report_assets)
        buy_signals = build_buy_signal_report(
            conn,
            optimization_summary,
            rsi_period=base_cfg.rsi_period,
        )
        sell_signals = build_sell_signal_report(
            conn,
            optimization_summary,
            rsi_period=base_cfg.rsi_period,
        )
        active_managed_symbols = active_alpaca_managed_symbols(conn)
        if buy_signals.empty or not active_managed_symbols:
            eligible_buy_signals = buy_signals.copy()
        else:
            eligible_buy_signals = buy_signals[
                ~buy_signals["Asset"].astype(str).str.upper().isin(active_managed_symbols)
            ].copy()
        realized_pnl_summary = build_alpaca_realized_pnl_summary(conn)

    return optimization_summary, curves, buy_signals, eligible_buy_signals, sell_signals, realized_pnl_summary


def _submit_alpaca_paper_buy_orders_for_db(
    db_path: str,
    buy_signals: pd.DataFrame,
    alpaca_cfg: AlpacaOrderConfig,
) -> pd.DataFrame:
    with _state_connection(db_path) as conn:
        return submit_alpaca_paper_buy_orders(buy_signals, alpaca_cfg, conn=conn)


def _load_alpaca_managed_positions_for_db(db_path: str) -> pd.DataFrame:
    with _state_connection(db_path) as conn:
        return load_alpaca_managed_positions(conn)


def _processed_message(data: pd.DataFrame, start_label: str) -> str:
    message = f"Processed {len(data)} rows from {start_label}"
    recovered_symbols = data.attrs.get(TRADIER_RECOVERED_SYMBOLS_ATTR, [])
    if recovered_symbols:
        message += f"; Tradier fallback recovered {', '.join(sorted(recovered_symbols))}"
    return message


async def _prepare_workflow_asset(
    *,
    db_path: str,
    mode: str,
    base_cfg: BacktestConfig,
    tradier_cfg: TradierMarketDataConfig | None,
    job: AssetRunJob,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    signal_locks: dict[str, asyncio.Lock],
    signal_histories: dict[str, pd.DataFrame],
    risk_free_history_lock: asyncio.Lock,
    risk_free_histories: dict[str, pd.DataFrame],
    asset_progress: AssetProgress | None = None,
    phase_timings: WorkflowPhaseTimings | None = None,
    download_executor: Executor | None = None,
) -> PreparedAssetRun | AssetRunResult:
    action = "Rebuilding" if mode == "rebuild" else "Updating"
    try:
        plan = await _run_blocking(
            download_executor,
            _prepare_asset_run,
            db_path,
            mode,
            base_cfg,
            job.asset_symbol,
            job.signal_symbol,
            buy_rsi_values,
            profit_target_values,
        )
        action = plan.action
        if asset_progress is not None:
            asset_progress.start_asset(
                asset=job.asset_symbol,
                signal=job.signal_symbol,
                action=plan.action,
            )
        data = await _timed_run_blocking(
            phase_timings,
            "download",
            load_strategy_data,
            asset_symbol=job.asset_symbol,
            signal_symbol=job.signal_symbol,
            start=plan.start,
            end=None,
            auto_adjust=base_cfg.auto_adjust,
            tradier_cfg=tradier_cfg,
            executor=download_executor,
        )
        if data.empty:
            if asset_progress is not None:
                asset_progress.start_asset(
                    asset=job.asset_symbol,
                    signal=job.signal_symbol,
                    action="skipping",
                )
            return AssetRunResult(
                workflow_idx=job.workflow_idx,
                asset_symbol=job.asset_symbol,
                signal_symbol=job.signal_symbol,
                action=plan.action,
                rows_processed=0,
                status="skipped",
                message="No finalized daily market data is available yet.",
            )
        asset_history = await _timed_run_blocking(
            phase_timings,
            "download",
            load_symbol_history,
            job.asset_symbol,
            end=None,
            auto_adjust=base_cfg.auto_adjust,
            tradier_cfg=tradier_cfg,
            executor=download_executor,
        )
        if asset_history.empty:
            raise RuntimeError(
                f"No settled daily asset history is available for {job.asset_symbol}."
            )
        async with risk_free_history_lock:
            risk_free_history = risk_free_histories.get(RISK_FREE_SYMBOL)
            if risk_free_history is None:
                risk_free_history = await _timed_run_blocking(
                    phase_timings,
                    "download",
                    load_risk_free_history,
                    end=None,
                    auto_adjust=base_cfg.auto_adjust,
                    tradier_cfg=tradier_cfg,
                    executor=download_executor,
                )
                if risk_free_history.empty:
                    raise RuntimeError("No settled daily benchmark history is available for ^IRX.")
                risk_free_histories[RISK_FREE_SYMBOL] = risk_free_history
        signal_lock = signal_locks.setdefault(job.signal_symbol, asyncio.Lock())
        async with signal_lock:
            signal_history = signal_histories.get(job.signal_symbol)
            if signal_history is None:
                signal_history = await _timed_run_blocking(
                    phase_timings,
                    "download",
                    load_signal_history,
                    job.signal_symbol,
                    end=None,
                    auto_adjust=base_cfg.auto_adjust,
                    tradier_cfg=tradier_cfg,
                    executor=download_executor,
                )
                if signal_history.empty:
                    raise RuntimeError(
                        f"No settled daily signal history is available for {job.signal_symbol}."
                    )
                signal_histories[job.signal_symbol] = signal_history
        return PreparedAssetRun(
            job=job,
            plan=plan,
            data=data,
            asset_history=asset_history,
            signal_history=signal_history,
            risk_free_history=risk_free_history,
        )
    except Exception as exc:
        if asset_progress is not None:
            asset_progress.start_asset(
                asset=job.asset_symbol,
                signal=job.signal_symbol,
                action="skipping",
            )
        return AssetRunResult(
            workflow_idx=job.workflow_idx,
            asset_symbol=job.asset_symbol,
            signal_symbol=job.signal_symbol,
            action=action,
            rows_processed=None,
            status="skipped",
            message=str(exc),
        )


async def _complete_workflow_asset(
    outcome: PreparedAssetRun | AssetRunResult,
    *,
    db_path: str,
    base_cfg: BacktestConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    asset_progress: AssetProgress | None = None,
    phase_timings: WorkflowPhaseTimings | None = None,
    strategy_executor: Executor | None = None,
    strategy_session: _WorkflowStrategySession | None = None,
) -> AssetRunResult:
    try:
        if isinstance(outcome, AssetRunResult):
            return outcome

        try:
            await _run_blocking(
                strategy_executor,
                _process_asset_grid_for_db,
                db_path,
                outcome.data,
                outcome.asset_history,
                outcome.signal_history,
                outcome.risk_free_history,
                base_cfg,
                outcome.job.asset_symbol,
                outcome.job.signal_symbol,
                buy_rsi_values,
                profit_target_values,
                outcome.plan.rebuild,
                phase_timings,
                strategy_session,
            )
        except Exception as exc:
            if asset_progress is not None:
                asset_progress.start_asset(
                    asset=outcome.job.asset_symbol,
                    signal=outcome.job.signal_symbol,
                    action="skipping",
                )
            return AssetRunResult(
                workflow_idx=outcome.job.workflow_idx,
                asset_symbol=outcome.job.asset_symbol,
                signal_symbol=outcome.job.signal_symbol,
                action=outcome.plan.action,
                rows_processed=None,
                status="skipped",
                message=str(exc),
            )

        return AssetRunResult(
            workflow_idx=outcome.job.workflow_idx,
            asset_symbol=outcome.job.asset_symbol,
            signal_symbol=outcome.job.signal_symbol,
            action=outcome.plan.action,
            rows_processed=len(outcome.data),
            status="done",
            message=_processed_message(outcome.data, outcome.plan.start_label),
        )
    finally:
        if asset_progress is not None:
            asset_progress.finish_asset()


async def _run_asset_pipeline(
    *,
    jobs: list[AssetRunJob],
    concurrency: int,
    db_path: str,
    mode: str,
    base_cfg: BacktestConfig,
    tradier_cfg: TradierMarketDataConfig | None,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    asset_progress: AssetProgress | None,
    phase_timings: WorkflowPhaseTimings,
) -> list[AssetRunResult]:
    if not jobs:
        return []

    signal_locks: dict[str, asyncio.Lock] = {}
    signal_histories: dict[str, pd.DataFrame] = {}
    risk_free_history_lock = asyncio.Lock()
    risk_free_histories: dict[str, pd.DataFrame] = {}
    strategy_session = _WorkflowStrategySession(db_path)

    async def prepare(
        job: AssetRunJob,
        download_executor: Executor | None = None,
    ) -> PreparedAssetRun | AssetRunResult:
        return await _prepare_workflow_asset(
            db_path=db_path,
            mode=mode,
            base_cfg=base_cfg,
            tradier_cfg=tradier_cfg,
            job=job,
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            signal_locks=signal_locks,
            signal_histories=signal_histories,
            risk_free_history_lock=risk_free_history_lock,
            risk_free_histories=risk_free_histories,
            asset_progress=asset_progress,
            phase_timings=phase_timings,
            download_executor=download_executor,
        )

    async def complete(
        outcome: PreparedAssetRun | AssetRunResult,
        strategy_executor: Executor,
    ) -> AssetRunResult:
        return await _complete_workflow_asset(
            outcome,
            db_path=db_path,
            base_cfg=base_cfg,
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            asset_progress=asset_progress,
            phase_timings=phase_timings,
            strategy_executor=strategy_executor,
            strategy_session=strategy_session,
        )

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="workflow-strategy",
    ) as strategy_executor:
        try:
            if concurrency <= 1:
                results = [
                    await complete(await prepare(job), strategy_executor)
                    for job in jobs
                ]
                return sorted(results, key=lambda result: result.workflow_idx)

            worker_count = min(max(1, concurrency), len(jobs))
            job_queue: asyncio.Queue[AssetRunJob | None] = asyncio.Queue()
            prepared_queue: asyncio.Queue[PreparedAssetRun | AssetRunResult] = asyncio.Queue(maxsize=1)
            for job in jobs:
                job_queue.put_nowait(job)
            for _ in range(worker_count):
                job_queue.put_nowait(None)

            async def download_worker(download_executor: Executor) -> None:
                while True:
                    job = await job_queue.get()
                    if job is None:
                        return
                    await prepared_queue.put(await prepare(job, download_executor))

            async def strategy_consumer() -> list[AssetRunResult]:
                results: list[AssetRunResult] = []
                for _ in jobs:
                    outcome = await prepared_queue.get()
                    results.append(await complete(outcome, strategy_executor))
                return sorted(results, key=lambda result: result.workflow_idx)

            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="workflow-download",
            ) as download_executor:
                async with asyncio.TaskGroup() as task_group:
                    consumer_task = task_group.create_task(strategy_consumer())
                    for _ in range(worker_count):
                        task_group.create_task(download_worker(download_executor))

            return consumer_task.result()
        finally:
            await _run_blocking(strategy_executor, strategy_session.close)


async def run_resumable_optimizations_async(
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    universe_cfg: UniverseConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    workflow_concurrency: int = DEFAULT_WORKFLOW_CONCURRENCY,
    no_color: bool = False,
    reporter: WorkflowReporter | None = None,
    tradier_cfg: TradierMarketDataConfig | None = None,
) -> None:
    _validate_optimization_grids(buy_rsi_values, profit_target_values)
    concurrency = max(1, workflow_concurrency)
    universe_cfg = replace(universe_cfg, sqlite_db_path=db_path)
    reporter = reporter or WorkflowReporter(no_color=no_color)
    workflow_timer = WorkflowTimer.start()
    phase_timings = WorkflowPhaseTimings()
    reporter.run_header(
        started_at_utc=workflow_timer.started_at_utc,
        mode=mode,
        db_path=db_path,
        output_dir=output_dir,
        workflow_concurrency=concurrency,
    )

    with reporter.status("Initializing workflow state"):
        await asyncio.to_thread(_initialize_state_db, db_path)
    with reporter.status("Reconciling Alpaca positions and loading workflow assets"):
        # Both steps write to the same SQLite database.  Keep startup writes
        # serialized: universe discovery performs replace-table writes while
        # reconciliation updates managed positions.
        reconciliation_results = await _timed_run_blocking(
            phase_timings,
            "alpaca",
            _reconcile_alpaca_managed_positions_for_db,
            db_path,
            alpaca_cfg,
        )
        workflow_assets = await asyncio.to_thread(
            _load_or_refresh_workflow_assets_for_db,
            db_path,
            universe_cfg,
        )

    reporter.universe_assets(workflow_assets)

    total_workflows = len(workflow_assets)
    jobs = [
        AssetRunJob(
            workflow_idx=workflow_idx,
            asset_symbol=str(workflow_asset.symbol),
            signal_symbol=str(workflow_asset.rsi_symbol),
        )
        for workflow_idx, workflow_asset in enumerate(
            workflow_assets.itertuples(index=False),
            start=1,
        )
    ]
    with reporter.asset_progress(total_workflows) as asset_progress:
        asset_run_results = await _run_asset_pipeline(
            jobs=jobs,
            concurrency=concurrency,
            db_path=db_path,
            mode=mode,
            base_cfg=base_cfg,
            tradier_cfg=tradier_cfg,
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            asset_progress=asset_progress,
            phase_timings=phase_timings,
        )

    completed_runs = [result for result in asset_run_results if result.status == "done"]
    if asset_run_results and not completed_runs:
        details = "; ".join(
            f"{result.asset_symbol}: {result.message}"
            for result in asset_run_results
        )
        raise WorkflowRunError(f"No asset workflows completed successfully. {details}")

    with reporter.status("Building workflow reports"):
        processed_asset_pairs = {
            (result.asset_symbol, result.signal_symbol) for result in completed_runs
        }
        (
            optimization_summary,
            curves,
            buy_signals,
            eligible_buy_signals,
            sell_signals,
            realized_pnl_summary,
        ) = await _timed_run_blocking(
            phase_timings,
            "report_generation",
            _build_reports_for_db,
            db_path,
            workflow_assets,
            base_cfg,
            processed_asset_pairs,
        )
    with reporter.status("Preparing Alpaca order results"):
        order_results = await _timed_run_blocking(
            phase_timings,
            "alpaca",
            _submit_alpaca_paper_buy_orders_for_db,
            db_path,
            buy_signals,
            alpaca_cfg,
        )
    submitted_buy_results = (
        order_results[order_results["Status"].isin({"submitted", "existing"})]
        if "Status" in order_results
        else pd.DataFrame()
    )
    if alpaca_cfg.enabled and not submitted_buy_results.empty:
        with reporter.status("Reconciling newly submitted Alpaca buys"):
            post_buy_reconciliation = await _timed_run_blocking(
                phase_timings,
                "alpaca",
                _reconcile_alpaca_managed_positions_for_db,
                db_path,
                alpaca_cfg,
            )
        reconciliation_results = pd.concat(
            [reconciliation_results, post_buy_reconciliation],
            ignore_index=True,
        )
    with reporter.status("Loading managed Alpaca positions"):
        managed_positions = await asyncio.to_thread(_load_alpaca_managed_positions_for_db, db_path)
    sell_reconciliation_results = reconciliation_results[reconciliation_results["Action"].eq("sell")]

    _write_workflow_outputs(
        mode=mode,
        db_path=db_path,
        base_cfg=base_cfg,
        buy_rsi_values=buy_rsi_values,
        profit_target_values=profit_target_values,
        alpaca_cfg=alpaca_cfg,
        output_dir=output_dir,
        workflow_concurrency=concurrency,
        reporter=reporter,
        asset_run_results=asset_run_results,
        optimization_summary=optimization_summary,
        curves=curves,
        buy_signals=buy_signals,
        eligible_buy_signals=eligible_buy_signals,
        sell_signals=sell_signals,
        realized_pnl_summary=realized_pnl_summary,
        managed_positions=managed_positions,
        reconciliation_results=reconciliation_results,
        sell_reconciliation_results=sell_reconciliation_results,
        order_results=order_results,
        workflow_timer=workflow_timer,
        phase_timings=phase_timings,
    )


def run_resumable_optimizations(
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    universe_cfg: UniverseConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    workflow_concurrency: int = DEFAULT_WORKFLOW_CONCURRENCY,
    no_color: bool = False,
    reporter: WorkflowReporter | None = None,
    tradier_cfg: TradierMarketDataConfig | None = None,
) -> None:
    asyncio.run(
        run_resumable_optimizations_async(
            mode=mode,
            db_path=db_path,
            base_cfg=base_cfg,
            universe_cfg=universe_cfg,
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            alpaca_cfg=alpaca_cfg,
            output_dir=output_dir,
            workflow_concurrency=workflow_concurrency,
            no_color=no_color,
            reporter=reporter,
            tradier_cfg=tradier_cfg,
        )
    )


def _write_workflow_outputs(
    *,
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    workflow_concurrency: int,
    reporter: WorkflowReporter,
    asset_run_results: list[AssetRunResult],
    optimization_summary: pd.DataFrame,
    curves: pd.DataFrame,
    buy_signals: pd.DataFrame,
    eligible_buy_signals: pd.DataFrame,
    sell_signals: pd.DataFrame,
    realized_pnl_summary: pd.DataFrame,
    managed_positions: pd.DataFrame,
    reconciliation_results: pd.DataFrame,
    sell_reconciliation_results: pd.DataFrame,
    order_results: pd.DataFrame,
    workflow_timer: WorkflowTimer,
    phase_timings: WorkflowPhaseTimings | None = None,
) -> None:
    report_output_started = time.perf_counter()
    terminal_order_results, terminal_reconciliation_results = _terminal_alpaca_display_results(
        managed_positions=managed_positions,
        reconciliation_results=reconciliation_results,
        order_results=order_results,
    )
    reporter.settings(
        mode=mode,
        db_path=db_path,
        workflow_concurrency=workflow_concurrency,
        risk_free_symbol=RISK_FREE_SYMBOL,
        buy_rsi_values=buy_rsi_values,
        profit_target_values=profit_target_values,
    )
    reporter.asset_run_summary([result.as_output_row() for result in asset_run_results])
    reporter.optimization_summary(optimization_summary)
    reporter.signal_report(
        "Buy Signals For Next Open",
        buy_signals,
        empty_message="No optimized assets with more than one trade and Sharpe >= 1.0 have a pending buy signal.",
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    curves.to_csv(output_path / "best_equity_curves.csv")
    optimization_summary.to_csv(output_path / "optimization_summary.csv", index=False)
    buy_signals.to_csv(output_path / "buy_signals.csv", index=False)
    eligible_buy_signals.to_csv(output_path / "eligible_buy_signals.csv", index=False)
    sell_signals.to_csv(output_path / "sell_signals.csv", index=False)
    realized_pnl_summary.to_csv(output_path / "alpaca_realized_pnl.csv", index=False)
    managed_positions.to_csv(output_path / "managed_positions.csv", index=False)
    reconciliation_results.to_csv(output_path / "alpaca_reconciliation_results.csv", index=False)
    if alpaca_cfg.enabled:
        reporter.order_results(terminal_order_results)
    reporter.buy_signal_eligibility_summary(
        buy_signals=buy_signals,
        eligible_buy_signals=eligible_buy_signals,
        order_results=order_results,
    )
    order_results.to_csv(output_path / "alpaca_order_results.csv", index=False)

    if alpaca_cfg.sell_enabled:
        reporter.reconciliation(terminal_reconciliation_results)
    reporter.realized_pnl_summary(realized_pnl_summary)
    sell_reconciliation_results.to_csv(output_path / "alpaca_sell_order_results.csv", index=False)
    if phase_timings is not None:
        phase_timings.add("report_generation", time.perf_counter() - report_output_started)
    reporter.workflow_footer(
        workflow_timer.elapsed_seconds(),
        phase_timings.snapshot() if phase_timings is not None else None,
    )


def _terminal_alpaca_display_results(
    *,
    managed_positions: pd.DataFrame,
    reconciliation_results: pd.DataFrame,
    order_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_order_results = order_results.copy()
    display_reconciliation_results = reconciliation_results.copy()
    display_id_by_position, display_id_by_buy_client_order_id = _managed_position_display_id_maps(managed_positions)

    if display_id_by_buy_client_order_id and "Client Order ID" in display_order_results.columns:
        display_order_results["Display ID"] = (
            display_order_results["Client Order ID"].astype(str).map(display_id_by_buy_client_order_id)
        )
    if display_id_by_position and "Position ID" in display_reconciliation_results.columns:
        display_reconciliation_results["Display ID"] = (
            pd.to_numeric(display_reconciliation_results["Position ID"], errors="coerce").map(display_id_by_position)
        )
    return display_order_results, display_reconciliation_results


def _managed_position_display_id_maps(managed_positions: pd.DataFrame) -> tuple[dict[int, int], dict[str, int]]:
    if managed_positions.empty or not {"id", "buy_client_order_id"}.issubset(managed_positions.columns):
        return {}, {}

    managed = managed_positions[["id", "buy_client_order_id"]].copy()
    managed["id"] = pd.to_numeric(managed["id"], errors="coerce")
    managed = managed.dropna(subset=["id"])
    if managed.empty:
        return {}, {}

    managed["id"] = managed["id"].astype(int)
    managed = managed.sort_values("id", kind="stable")
    display_id_by_position = {
        position_id: display_id
        for display_id, position_id in enumerate(managed["id"].tolist(), start=1)
    }
    position_id_by_buy_client_order_id = {
        str(row["buy_client_order_id"]): int(row["id"])
        for row in managed.to_dict("records")
        if not pd.isna(row["buy_client_order_id"])
    }
    display_id_by_buy_client_order_id = {
        client_order_id: display_id_by_position[position_id]
        for client_order_id, position_id in position_id_by_buy_client_order_id.items()
    }
    return display_id_by_position, display_id_by_buy_client_order_id
