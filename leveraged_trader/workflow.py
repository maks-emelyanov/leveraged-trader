from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import pandas as pd

from .alpaca import reconcile_alpaca_managed_positions, submit_alpaca_paper_buy_orders
from .config import (
    AlpacaOrderConfig,
    BacktestConfig,
    RISK_FREE_SYMBOL,
    TradierMarketDataConfig,
    UniverseConfig,
)
from .market_data import TRADIER_RECOVERED_SYMBOLS_ATTR, load_strategy_data
from .output import AssetProgress, WorkflowReporter
from .reports import (
    build_alpaca_realized_pnl_summary,
    build_buy_signal_report,
    build_sell_signal_report,
    summarize_saved_results,
)
from .storage import (
    active_alpaca_managed_symbols,
    earliest_state_date,
    expected_state_count,
    init_state_db,
    load_alpaca_managed_positions,
    process_asset_grid,
    save_workflow_assets,
)
from .universe import determine_workflow_assets


DEFAULT_WORKFLOW_CONCURRENCY = 4
SQLITE_BUSY_TIMEOUT_MS = 60_000


@dataclass(frozen=True)
class AssetRunPlan:
    asset_symbol: str
    signal_symbol: str
    rebuild: bool
    start: Optional[str]
    action: str
    start_label: str


@dataclass(frozen=True)
class AssetRunResult:
    workflow_idx: int
    asset_symbol: str
    signal_symbol: str
    action: str
    rows_processed: Optional[int]
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


@contextmanager
def _state_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
    try:
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
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
    expected_combinations: int,
    asset_symbol: str,
    signal_symbol: str,
) -> AssetRunPlan:
    with _state_connection(db_path) as conn:
        rebuild_asset = mode == "rebuild"
        existing_count = expected_state_count(conn, asset_symbol, signal_symbol)
        if mode == "update" and existing_count != expected_combinations:
            rebuild_asset = True

        start: Optional[str] = None
        if not rebuild_asset:
            last_date = earliest_state_date(conn, asset_symbol, signal_symbol)
            if last_date is not None:
                start = last_date

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
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rebuild: bool,
) -> None:
    with _state_connection(db_path) as conn:
        process_asset_grid(
            conn,
            data,
            base_cfg,
            asset_symbol,
            signal_symbol,
            buy_rsi_values,
            profit_target_values,
            rebuild=rebuild,
        )


def _build_reports_for_db(
    db_path: str,
    workflow_assets: pd.DataFrame,
    base_cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with _state_connection(db_path) as conn:
        optimization_summary, curves = summarize_saved_results(conn, workflow_assets)
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


async def _process_workflow_asset(
    *,
    db_path: str,
    mode: str,
    base_cfg: BacktestConfig,
    tradier_cfg: Optional[TradierMarketDataConfig],
    expected_combinations: int,
    workflow_idx: int,
    total_workflows: int,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    signal_locks: dict[str, asyncio.Lock],
    asset_progress: Optional[AssetProgress] = None,
) -> AssetRunResult:
    action = "Rebuilding" if mode == "rebuild" else "Updating"
    try:
        plan = await asyncio.to_thread(
            _prepare_asset_run,
            db_path,
            mode,
            expected_combinations,
            asset_symbol,
            signal_symbol,
        )
        action = plan.action
        if asset_progress is not None:
            asset_progress.start_asset(asset=asset_symbol, signal=signal_symbol, action=plan.action)
        data = await asyncio.to_thread(
            load_strategy_data,
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            start=plan.start,
            end=None,
            auto_adjust=base_cfg.auto_adjust,
            tradier_cfg=tradier_cfg,
        )
        signal_lock = signal_locks.setdefault(signal_symbol, asyncio.Lock())
        async with signal_lock:
            await asyncio.to_thread(
                _process_asset_grid_for_db,
                db_path,
                data,
                base_cfg,
                asset_symbol,
                signal_symbol,
                buy_rsi_values,
                profit_target_values,
                plan.rebuild,
            )
        return AssetRunResult(
            workflow_idx=workflow_idx,
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            action=plan.action,
            rows_processed=len(data),
            status="done",
            message=_processed_message(data, plan.start_label),
        )
    except Exception as exc:
        if asset_progress is not None:
            asset_progress.start_asset(asset=asset_symbol, signal=signal_symbol, action="skipping")
        return AssetRunResult(
            workflow_idx=workflow_idx,
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            action=action,
            rows_processed=None,
            status="skipped",
            message=str(exc),
        )
    finally:
        if asset_progress is not None:
            asset_progress.finish_asset()


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
    reporter: Optional[WorkflowReporter] = None,
    tradier_cfg: Optional[TradierMarketDataConfig] = None,
) -> None:
    concurrency = max(1, workflow_concurrency)
    universe_cfg = replace(universe_cfg, sqlite_db_path=db_path)
    reporter = reporter or WorkflowReporter(no_color=no_color)

    with reporter.status("Initializing workflow state"):
        await asyncio.to_thread(_initialize_state_db, db_path)
    with reporter.status("Reconciling Alpaca positions and loading workflow assets"):
        reconciliation_task = asyncio.create_task(
            asyncio.to_thread(_reconcile_alpaca_managed_positions_for_db, db_path, alpaca_cfg)
        )
        workflow_assets_task = asyncio.create_task(
            asyncio.to_thread(_load_or_refresh_workflow_assets_for_db, db_path, universe_cfg)
        )
        reconciliation_results, workflow_assets = await asyncio.gather(reconciliation_task, workflow_assets_task)

    reporter.universe_assets(workflow_assets)

    expected_combinations = len(buy_rsi_values) * len(profit_target_values)
    total_workflows = len(workflow_assets)
    semaphore = asyncio.Semaphore(concurrency)
    signal_locks: dict[str, asyncio.Lock] = {}

    async def process_with_limit(
        workflow_idx: int,
        asset_symbol: str,
        signal_symbol: str,
        asset_progress: AssetProgress,
    ) -> AssetRunResult:
        async with semaphore:
            return await _process_workflow_asset(
                db_path=db_path,
                mode=mode,
                base_cfg=base_cfg,
                tradier_cfg=tradier_cfg,
                expected_combinations=expected_combinations,
                workflow_idx=workflow_idx,
                total_workflows=total_workflows,
                asset_symbol=asset_symbol,
                signal_symbol=signal_symbol,
                buy_rsi_values=buy_rsi_values,
                profit_target_values=profit_target_values,
                signal_locks=signal_locks,
                asset_progress=asset_progress,
            )

    with reporter.asset_progress(total_workflows) as asset_progress:
        asset_run_results = await asyncio.gather(
            *[
                process_with_limit(
                    workflow_idx,
                    str(workflow_asset.symbol),
                    str(workflow_asset.rsi_symbol),
                    asset_progress,
                )
                for workflow_idx, workflow_asset in enumerate(workflow_assets.itertuples(index=False), start=1)
            ]
        )

    with reporter.status("Building workflow reports"):
        (
            optimization_summary,
            curves,
            buy_signals,
            eligible_buy_signals,
            sell_signals,
            realized_pnl_summary,
        ) = await asyncio.to_thread(_build_reports_for_db, db_path, workflow_assets, base_cfg)
    with reporter.status("Preparing Alpaca order results"):
        order_results = await asyncio.to_thread(
            _submit_alpaca_paper_buy_orders_for_db,
            db_path,
            buy_signals,
            alpaca_cfg,
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
    reporter: Optional[WorkflowReporter] = None,
    tradier_cfg: Optional[TradierMarketDataConfig] = None,
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
) -> None:
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
    reporter.signal_report(
        "Sell Signals For Next Open",
        sell_signals,
        empty_message="No optimized assets have a pending sell signal.",
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
        reporter.order_results(order_results)
    order_results.to_csv(output_path / "alpaca_order_results.csv", index=False)

    if alpaca_cfg.sell_enabled:
        reporter.reconciliation(reconciliation_results)
    reporter.realized_pnl_summary(realized_pnl_summary)
    sell_reconciliation_results.to_csv(output_path / "alpaca_sell_order_results.csv", index=False)
