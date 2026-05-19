from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .alpaca import submit_alpaca_paper_buy_orders, submit_alpaca_paper_sell_orders
from .config import AlpacaOrderConfig, BacktestConfig, RISK_FREE_SYMBOL, UniverseConfig
from .market_data import load_strategy_data
from .reports import build_buy_signal_report, build_sell_signal_report, summarize_saved_results
from .storage import expected_state_count, earliest_state_date, init_state_db, process_asset_grid, save_workflow_assets
from .universe import determine_workflow_assets


def load_or_refresh_workflow_assets(
    conn: sqlite3.Connection,
    universe_cfg: UniverseConfig,
) -> pd.DataFrame:
    workflow_assets = determine_workflow_assets(universe_cfg)
    save_workflow_assets(conn, workflow_assets)
    return workflow_assets


def run_resumable_optimizations(
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    universe_cfg: UniverseConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        init_state_db(conn)
        workflow_assets = load_or_refresh_workflow_assets(conn, universe_cfg)
        expected_combinations = len(buy_rsi_values) * len(profit_target_values)
        total_workflows = len(workflow_assets)

        for workflow_idx, workflow_asset in enumerate(workflow_assets.itertuples(index=False), start=1):
            asset_symbol = workflow_asset.symbol
            signal_symbol = workflow_asset.rsi_symbol
            progress = f"[{workflow_idx}/{total_workflows}]"
            try:
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
                print(
                    f"{progress} {action} {asset_symbol} using {signal_symbol} RSI from {start_label}...",
                    flush=True,
                )
                data = load_strategy_data(
                    asset_symbol=asset_symbol,
                    signal_symbol=signal_symbol,
                    start=start,
                    end=None,
                    auto_adjust=base_cfg.auto_adjust,
                )
                process_asset_grid(
                    conn,
                    data,
                    base_cfg,
                    asset_symbol,
                    signal_symbol,
                    buy_rsi_values,
                    profit_target_values,
                    rebuild=rebuild_asset,
                )
                print(f"{progress} Finished {asset_symbol}: processed {len(data)} rows.", flush=True)
            except Exception as exc:
                print(f"{progress} Skipping {asset_symbol} using {signal_symbol} RSI: {exc}", flush=True)

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

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\nGrid search settings:")
    print(f"Mode: {mode}")
    print(f"SQLite state database: {db_path}")
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
    sell_signals.to_csv(output_path / "sell_signals.csv", index=False)

    order_results = submit_alpaca_paper_buy_orders(buy_signals, alpaca_cfg)
    if alpaca_cfg.enabled:
        print("\nAlpaca paper order results:")
        if order_results.empty:
            print("No buy signals to submit.")
        else:
            print(order_results.set_index("Asset"))
    order_results.to_csv(output_path / "alpaca_order_results.csv", index=False)

    sell_order_results = submit_alpaca_paper_sell_orders(sell_signals, alpaca_cfg)
    if alpaca_cfg.sell_enabled:
        print("\nAlpaca paper sell order results:")
        if sell_order_results.empty:
            print("No sell signals to submit.")
        else:
            print(sell_order_results.set_index("Asset"))
    sell_order_results.to_csv(output_path / "alpaca_sell_order_results.csv", index=False)
