from __future__ import annotations

import argparse
import os

import numpy as np

from .config import (
    ALPACA_PAPER_BASE_URL,
    SQLITE_DB_PATH,
    TRADIER_LIVE_BASE_URL,
    AlpacaOrderConfig,
    BacktestConfig,
    TradierMarketDataConfig,
    UniverseConfig,
    load_dotenv,
)
from .workflow import DEFAULT_WORKFLOW_CONCURRENCY, run_resumable_optimizations

REMOVED_ENV_VARS = {
    "ALPACA_BATCH_CASH_FRACTION": (
        "Alpaca buy batches now reserve min(number_of_eligible_buy_signals * 0.05, 0.50) "
        "of account cash; remove ALPACA_BATCH_CASH_FRACTION from the environment or .env."
    ),
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


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
            "Submit bounded Alpaca paper day limit buy orders for current buy recommendations. "
            "The batch uses a capped fraction of account cash to size whole-share quantities. Enabled by default; use "
            "--no-alpaca-submit-buy-orders to skip."
        ),
    )
    parser.add_argument(
        "--alpaca-submit-sell-orders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Submit and renew Alpaca paper managed GTC limit sell orders for filled managed buys. "
            "Each order sells the remaining whole-share managed quantity at the frozen target price. "
            "Enabled by default; use "
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
    parser.add_argument(
        "--alpaca-buy-limit-buffer-bps",
        type=float,
        default=_env_float("ALPACA_BUY_LIMIT_BUFFER_BPS", 500.0),
        help="Basis-point buffer above the price estimate for whole-share day buy limits (default: 500).",
    )
    parser.add_argument(
        "--alpaca-gtc-sell-renewal",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ALPACA_GTC_SELL_RENEWAL_ENABLED", True),
        help=(
            "Renew managed Alpaca GTC limit sells before Alpaca's aged-order expiration. "
            "Enabled by default; use --no-alpaca-gtc-sell-renewal to skip."
        ),
    )
    parser.add_argument(
        "--alpaca-gtc-sell-renewal-days-before-expiration",
        type=int,
        default=_env_int("ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION", 7),
        help="Renew managed Alpaca GTC limit sells this many days before expiration.",
    )
    parser.add_argument(
        "--tradier-fallback",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("TRADIER_FALLBACK_ENABLED", True),
        help=(
            "Use Tradier historical daily data as a fallback when Yahoo Finance skips symbols. "
            "Enabled by default when a Tradier token is configured; use --no-tradier-fallback to skip."
        ),
    )
    parser.add_argument(
        "--tradier-access-token",
        default=(
            os.environ.get("TRADIER_ACCESS_TOKEN")
            or os.environ.get("TRADIER_API_TOKEN")
            or os.environ.get("TRADIER_TOKEN")
        ),
        help=(
            "Tradier bearer token for market-data fallback. Can also be set with "
            "TRADIER_ACCESS_TOKEN, TRADIER_API_TOKEN, or TRADIER_TOKEN."
        ),
    )
    parser.add_argument(
        "--tradier-base-url",
        default=os.environ.get("TRADIER_BASE_URL", TRADIER_LIVE_BASE_URL),
        help="Tradier Brokerage API base URL for market data fallback.",
    )
    parser.add_argument(
        "--tradier-timeout-seconds",
        type=int,
        default=_env_int("TRADIER_TIMEOUT_SECONDS", 30),
        help="Timeout for Tradier market data requests.",
    )
    parser.add_argument(
        "--workflow-concurrency",
        type=int,
        default=DEFAULT_WORKFLOW_CONCURRENCY,
        help=(
            "Maximum number of assets to process concurrently during data loading and optimization. "
            "Use 1 for fully serial behavior."
        ),
    )
    parser.add_argument(
        "--require-workflow-source-success",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Abort before running strategies if an enabled universe discovery or active listing "
            "source fails. Disabled by default; source failures are otherwise recorded as a "
            "degraded universe."
        ),
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored terminal output.",
    )
    args = parser.parse_args()
    for name, message in REMOVED_ENV_VARS.items():
        if name in os.environ:
            parser.error(message)
    return args


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
    universe_cfg = UniverseConfig(
        top_n=None,
        sqlite_db_path=args.db,
        require_workflow_source_success=args.require_workflow_source_success,
    )
    alpaca_cfg = AlpacaOrderConfig(
        enabled=args.alpaca_submit_buy_orders,
        sell_enabled=args.alpaca_submit_sell_orders,
        api_key_id=args.alpaca_api_key_id,
        api_secret_key=args.alpaca_api_secret_key,
        base_url=args.alpaca_base_url,
        buy_limit_buffer_bps=args.alpaca_buy_limit_buffer_bps,
        timeout_seconds=args.alpaca_timeout_seconds,
        gtc_sell_renewal_enabled=args.alpaca_gtc_sell_renewal,
        gtc_sell_renewal_days_before_expiration=args.alpaca_gtc_sell_renewal_days_before_expiration,
    )
    tradier_cfg = TradierMarketDataConfig(
        enabled=args.tradier_fallback,
        access_token=args.tradier_access_token,
        base_url=args.tradier_base_url,
        timeout_seconds=args.tradier_timeout_seconds,
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
        workflow_concurrency=args.workflow_concurrency,
        no_color=args.no_color,
        tradier_cfg=tradier_cfg,
    )
