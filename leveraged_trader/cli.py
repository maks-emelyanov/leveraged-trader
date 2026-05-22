from __future__ import annotations

import argparse
import os

import numpy as np

from .config import ALPACA_PAPER_BASE_URL, AlpacaOrderConfig, BacktestConfig, SQLITE_DB_PATH, UniverseConfig, load_dotenv
from .workflow import run_resumable_optimizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["update", "rebuild"],
        default="update",
        help="update resumes saved strategy state; rebuild recomputes all state from scratch.",
    )
    parser.add_argument(
        "--db",
        default=SQLITE_DB_PATH,
        help="SQLite file used for market data, RSI values, equity history, and strategy state.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated CSV reports and Alpaca order result files.",
    )
    parser.add_argument(
        "--alpaca-submit-buy-orders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Submit Alpaca paper market buy orders for current buy recommendations. "
            "Each order uses 10%% of account cash. Enabled by default; use "
            "--no-alpaca-submit-buy-orders to skip."
        ),
    )
    parser.add_argument(
        "--alpaca-submit-sell-orders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Submit Alpaca paper managed limit sell orders for filled managed buys. "
            "Each order sells the original filled buy quantity at the frozen target price. Enabled by default; use "
            "--no-alpaca-submit-sell-orders to skip."
        ),
    )
    parser.add_argument(
        "--alpaca-api-key-id",
        default=os.environ.get("ALPACA_API_KEY_ID"),
        help="Alpaca API key ID. Can also be set with ALPACA_API_KEY_ID.",
    )
    parser.add_argument(
        "--alpaca-api-secret-key",
        default=os.environ.get("ALPACA_API_SECRET_KEY"),
        help="Alpaca API secret key. Can also be set with ALPACA_API_SECRET_KEY.",
    )
    parser.add_argument(
        "--alpaca-base-url",
        default=os.environ.get("ALPACA_BASE_URL", ALPACA_PAPER_BASE_URL),
        help="Alpaca Trading API base URL. Defaults to the paper trading endpoint.",
    )
    parser.add_argument(
        "--alpaca-timeout-seconds",
        type=int,
        default=30,
        help="Timeout for Alpaca API requests.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    base_cfg = BacktestConfig(
        initial_capital=100_000,
        rsi_period=14,
        buy_rsi=30,
        profit_target_multiple=10.0,
        fee_bps=1.0,
        slippage_bps=2.0,
        auto_adjust=True,
    )

    mode = args.mode
    universe_cfg = UniverseConfig(top_n=None, sqlite_db_path=args.db)
    alpaca_cfg = AlpacaOrderConfig(
        enabled=args.alpaca_submit_buy_orders,
        sell_enabled=args.alpaca_submit_sell_orders,
        api_key_id=args.alpaca_api_key_id,
        api_secret_key=args.alpaca_api_secret_key,
        base_url=args.alpaca_base_url,
        timeout_seconds=args.alpaca_timeout_seconds,
    )
    buy_rsi_values = list(range(20, 51))
    profit_target_values = [round(x, 2) for x in np.arange(1.1, 5.05, 0.1)]

    run_resumable_optimizations(
        mode=mode,
        db_path=args.db,
        base_cfg=base_cfg,
        universe_cfg=universe_cfg,
        buy_rsi_values=buy_rsi_values,
        profit_target_values=profit_target_values,
        alpaca_cfg=alpaca_cfg,
        output_dir=args.output_dir,
    )
