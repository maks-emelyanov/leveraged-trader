from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .alpaca import reconcile_alpaca_managed_positions, submit_alpaca_paper_buy_orders
from .config import AlpacaOrderConfig, BacktestConfig, RISK_FREE_SYMBOL, UniverseConfig
from .market_data import load_strategy_data
from .reports import build_buy_signal_report, build_sell_signal_report, summarize_saved_results
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


def load_or_refresh_workflow_assets(
    conn: sqlite3.Connection,
    universe_cfg: UniverseConfig,
) -> pd.DataFrame:
    workflow_assets = determine_workflow_assets(universe_cfg)
    save_workflow_assets(conn, workflow_assets)
    return workflow_assets


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    return optimization_summary, curves, buy_signals, eligible_buy_signals, sell_signals


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


async def _process_workflow_asset(
    *,
    db_path: str,
    mode: str,
    base_cfg: BacktestConfig,
    expected_combinations: int,
    workflow_idx: int,
    total_workflows: int,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    signal_locks: dict[str, asyncio.Lock],
) -> None:
    progress = f"[{workflow_idx}/{total_workflows}]"
    try:
        plan = await asyncio.to_thread(
            _prepare_asset_run,
            db_path,
            mode,
            expected_combinations,
            asset_symbol,
            signal_symbol,
        )
        print(
            f"{progress} {plan.action} {asset_symbol} using {signal_symbol} RSI from {plan.start_label}...",
            flush=True,
        )
        data = await asyncio.to_thread(
            load_strategy_data,
            asset_symbol=asset_symbol,
            signal_symbol=signal_symbol,
            start=plan.start,
            end=None,
            auto_adjust=base_cfg.auto_adjust,
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
        print(f"{progress} Finished {asset_symbol}: processed {len(data)} rows.", flush=True)
    except Exception as exc:
        print(f"{progress} Skipping {asset_symbol} using {signal_symbol} RSI: {exc}", flush=True)


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
) -> None:
    concurrency = max(1, workflow_concurrency)
    universe_cfg = replace(universe_cfg, sqlite_db_path=db_path)

    await asyncio.to_thread(_initialize_state_db, db_path)
    reconciliation_task = asyncio.create_task(
        asyncio.to_thread(_reconcile_alpaca_managed_positions_for_db, db_path, alpaca_cfg)
    )
    workflow_assets_task = asyncio.create_task(
        asyncio.to_thread(_load_or_refresh_workflow_assets_for_db, db_path, universe_cfg)
    )
    reconciliation_results, workflow_assets = await asyncio.gather(reconciliation_task, workflow_assets_task)

    expected_combinations = len(buy_rsi_values) * len(profit_target_values)
    total_workflows = len(workflow_assets)
    semaphore = asyncio.Semaphore(concurrency)
    signal_locks: dict[str, asyncio.Lock] = {}

    async def process_with_limit(
        workflow_idx: int,
        asset_symbol: str,
        signal_symbol: str,
    ) -> None:
        async with semaphore:
            await _process_workflow_asset(
                db_path=db_path,
                mode=mode,
                base_cfg=base_cfg,
                expected_combinations=expected_combinations,
                workflow_idx=workflow_idx,
                total_workflows=total_workflows,
                asset_symbol=asset_symbol,
                signal_symbol=signal_symbol,
                buy_rsi_values=buy_rsi_values,
                profit_target_values=profit_target_values,
                signal_locks=signal_locks,
            )

    await asyncio.gather(
        *[
            process_with_limit(
                workflow_idx,
                str(workflow_asset.symbol),
                str(workflow_asset.rsi_symbol),
            )
            for workflow_idx, workflow_asset in enumerate(workflow_assets.itertuples(index=False), start=1)
        ]
    )

    (
        optimization_summary,
        curves,
        buy_signals,
        eligible_buy_signals,
        sell_signals,
    ) = await asyncio.to_thread(_build_reports_for_db, db_path, workflow_assets, base_cfg)
    order_results = await asyncio.to_thread(_submit_alpaca_paper_buy_orders_for_db, db_path, buy_signals, alpaca_cfg)
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
        optimization_summary=optimization_summary,
        curves=curves,
        buy_signals=buy_signals,
        eligible_buy_signals=eligible_buy_signals,
        sell_signals=sell_signals,
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
    optimization_summary: pd.DataFrame,
    curves: pd.DataFrame,
    buy_signals: pd.DataFrame,
    eligible_buy_signals: pd.DataFrame,
    sell_signals: pd.DataFrame,
    managed_positions: pd.DataFrame,
    reconciliation_results: pd.DataFrame,
    sell_reconciliation_results: pd.DataFrame,
    order_results: pd.DataFrame,
) -> None:
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\nGrid search settings:")
    print(f"Mode: {mode}")
    print(f"SQLite state database: {db_path}")
    print(f"Workflow concurrency: {workflow_concurrency}")
    print("Start date: earliest overlapping available history for each leveraged asset and RSI symbol")
    print(f"Sharpe risk-free benchmark: {RISK_FREE_SYMBOL} 13-week U.S. Treasury bill")
    print(f"Buy RSI values: {buy_rsi_values[0]} to {buy_rsi_values[-1]} step 1")
    print(
        "Sell return multiples:"
        f" {profit_target_values[0]:.1f} to {profit_target_values[-1]:.1f} step 0.1"
    )
    print("\nBest Sharpe parameters by asset:")
    if optimization_summary.empty:
        print("No strategies produced more than one executed trade.")
    else:
        print(
            optimization_summary.drop(columns=["End Date", "Annualized Vol", "Hit Rate"]).set_index("Asset")
        )

    print("\nBuy signals for next open:")
    if buy_signals.empty:
        print(
            "No optimized assets with more than one trade and Sharpe >= 1.0 have a pending buy signal for next open."
        )
    else:
        print(buy_signals.set_index("Asset"))

    print("\nSell signals for next open:")
    if sell_signals.empty:
        print("No optimized assets have a pending sell signal for next open.")
    else:
        print(sell_signals.set_index("Asset"))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    curves.to_csv(output_path / "best_equity_curves.csv")
    optimization_summary.to_csv(output_path / "optimization_summary.csv", index=False)
    buy_signals.to_csv(output_path / "buy_signals.csv", index=False)
    eligible_buy_signals.to_csv(output_path / "eligible_buy_signals.csv", index=False)
    sell_signals.to_csv(output_path / "sell_signals.csv", index=False)
    managed_positions.to_csv(output_path / "managed_positions.csv", index=False)
    reconciliation_results.to_csv(output_path / "alpaca_reconciliation_results.csv", index=False)
    if alpaca_cfg.enabled:
        print("\nAlpaca paper order results:")
        if order_results.empty:
            print("No buy signals to submit.")
        else:
            print(order_results.set_index("Asset"))
    order_results.to_csv(output_path / "alpaca_order_results.csv", index=False)

    if alpaca_cfg.sell_enabled:
        print("\nAlpaca managed position reconciliation:")
        if reconciliation_results.empty:
            print("No managed Alpaca positions to reconcile.")
        else:
            print(reconciliation_results.set_index("Asset"))
    sell_reconciliation_results.to_csv(output_path / "alpaca_sell_order_results.csv", index=False)
