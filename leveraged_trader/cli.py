from __future__ import annotations

import argparse
import os
import stat
import sys

import numpy as np

from .config import (
    ALPACA_PAPER_BASE_URL,
    SQLITE_DB_PATH,
    TRADIER_LIVE_BASE_URL,
    TRADIER_PLACEHOLDER_TOKENS,
    AlpacaOrderConfig,
    BacktestConfig,
    TradierMarketDataConfig,
    UniverseConfig,
    load_dotenv,
    validate_alpaca_reconciliation_configuration,
    validate_runtime_configuration,
)
from .workflow import DEFAULT_WORKFLOW_CONCURRENCY, run_alpaca_reconciliation, run_resumable_optimizations

REMOVED_ENV_VARS = {
    "ALPACA_BATCH_CASH_FRACTION": (
        "Alpaca buy batches now reserve min(number_of_eligible_buy_signals * 0.05, 0.50) "
        "of account cash; remove ALPACA_BATCH_CASH_FRACTION from the environment or .env."
    ),
}
REMOVED_CREDENTIAL_CLI_FLAGS = {
    "--alpaca-api-key-id": "ALPACA_API_KEY_ID",
    "--alpaca-api-secret-key": "ALPACA_API_SECRET_KEY",
    "--tradier-access-token": "TRADIER_ACCESS_TOKEN",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: true, false, 1, 0, yes, no, on, off.")


def _bounded_int(value: str | int, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer between {minimum} and {maximum}.") from exc
    if isinstance(value, bool) or not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{name} must be an integer between {minimum} and {maximum}.")
    return parsed


def _bounded_float(value: str | float, *, name: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a finite number between {minimum:g} and {maximum:g}."
        ) from exc
    if not np.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{name} must be a finite number between {minimum:g} and {maximum:g}.")
    return parsed


def _timeout_seconds(value: str | int) -> int:
    return _bounded_int(value, name="timeout seconds", minimum=1, maximum=600)


def _renewal_days(value: str | int) -> int:
    return _bounded_int(value, name="renewal days", minimum=0, maximum=90)


def _workflow_concurrency(value: str | int) -> int:
    return _bounded_int(value, name="workflow concurrency", minimum=1, maximum=64)


def _scheduled_closed_audit_interval_minutes(value: str | int) -> int:
    return _bounded_int(value, name="scheduled closed-audit interval minutes", minimum=1, maximum=1_440)


def _workflow_deadline_epoch(value: str | int) -> int:
    return _bounded_int(
        value,
        name="workflow deadline epoch",
        minimum=1,
        maximum=9_223_372_036_854_775_807,
    )


def _buy_limit_buffer_bps(value: str | float) -> float:
    return _bounded_float(value, name="buy limit buffer bps", minimum=0.0, maximum=10_000.0)


def _printable_runtime_path(value: str, *, option: str) -> str:
    if not value or not value.strip():
        raise argparse.ArgumentTypeError(f"{option} must be a nonempty filesystem path.")
    if any(not character.isprintable() for character in value):
        raise argparse.ArgumentTypeError(f"{option} must not contain control characters.")
    return value


def _database_path(value: str) -> str:
    path = _printable_runtime_path(value, option="--db")
    if path == ":memory:":
        raise argparse.ArgumentTypeError("--db must be a persistent filesystem path; ':memory:' is not supported.")
    path_separators = tuple(separator for separator in (os.sep, os.altsep) if separator is not None)
    if path.endswith(path_separators) or os.path.basename(path) in {os.curdir, os.pardir}:
        raise argparse.ArgumentTypeError("--db must name a database file, not a directory.")
    try:
        path_status = os.lstat(path)
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise argparse.ArgumentTypeError("--db must be an accessible filesystem path.") from exc
    if stat.S_ISLNK(path_status.st_mode):
        try:
            path_status = os.stat(path)
        except OSError as exc:
            raise argparse.ArgumentTypeError("--db symbolic link must resolve to a regular file.") from exc
    if not stat.S_ISREG(path_status.st_mode):
        raise argparse.ArgumentTypeError("--db must name a regular database file when it already exists.")
    if path_status.st_nlink != 1:
        raise argparse.ArgumentTypeError("--db must not name a file with multiple hard links.")
    return path


def _output_directory_path(value: str) -> str:
    path = _printable_runtime_path(value, option="--output-dir")
    try:
        path_status = os.lstat(path)
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise argparse.ArgumentTypeError("--output-dir must be an accessible filesystem path.") from exc
    if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISDIR(path_status.st_mode):
        raise argparse.ArgumentTypeError("--output-dir must name a regular directory, not a file or symbolic link.")
    return path


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return _bounded_int(value, name=name, minimum=minimum, maximum=maximum)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return _bounded_float(value, name=name, minimum=minimum, maximum=maximum)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc


def _alpaca_config_from_args(args: argparse.Namespace) -> AlpacaOrderConfig:
    api_key_id = os.environ.get("ALPACA_API_KEY_ID")
    api_secret_key = os.environ.get("ALPACA_API_SECRET_KEY")
    return AlpacaOrderConfig(
        enabled=args.alpaca_submit_buy_orders,
        sell_enabled=args.alpaca_submit_sell_orders,
        # Normalize ordinary dotenv-style padding, but retain every other
        # whitespace/control character for the header-safety validator.
        api_key_id=api_key_id.strip(" ") if api_key_id is not None else None,
        api_secret_key=api_secret_key.strip(" ") if api_secret_key is not None else None,
        base_url=args.alpaca_base_url,
        buy_limit_buffer_bps=args.alpaca_buy_limit_buffer_bps,
        timeout_seconds=args.alpaca_timeout_seconds,
        gtc_sell_renewal_enabled=args.alpaca_gtc_sell_renewal,
        gtc_sell_renewal_days_before_expiration=args.alpaca_gtc_sell_renewal_days_before_expiration,
    )


def _tradier_access_token_from_env() -> str | None:
    for name in ("TRADIER_ACCESS_TOKEN", "TRADIER_API_TOKEN", "TRADIER_TOKEN"):
        value = os.environ.get(name)
        if value is None:
            continue
        normalized = value.strip(" ")
        # A copied .env.example commonly leaves the primary placeholder in
        # place while a user configures one of the documented fallback aliases.
        # Treat only known placeholders as absent; malformed real credentials
        # must retain precedence so validation can reject them explicitly.
        if normalized and normalized.casefold() not in TRADIER_PLACEHOLDER_TOKENS:
            return normalized
    return None


def _tradier_config_from_args(args: argparse.Namespace) -> TradierMarketDataConfig:
    return TradierMarketDataConfig(
        enabled=args.tradier_fallback,
        access_token=_tradier_access_token_from_env(),
        base_url=args.tradier_base_url,
        timeout_seconds=args.tradier_timeout_seconds,
    )


def _resolve_env_backed_defaults(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Coerce environment defaults only after argparse has handled help and CLI values."""
    try:
        if args.alpaca_gtc_sell_renewal is None:
            args.alpaca_gtc_sell_renewal = _env_bool("ALPACA_GTC_SELL_RENEWAL_ENABLED", True)
        if args.alpaca_gtc_sell_renewal_days_before_expiration is None:
            args.alpaca_gtc_sell_renewal_days_before_expiration = (
                _env_int(
                    "ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION",
                    AlpacaOrderConfig.gtc_sell_renewal_days_before_expiration,
                    minimum=0,
                    maximum=90,
                )
                if args.alpaca_gtc_sell_renewal
                else AlpacaOrderConfig.gtc_sell_renewal_days_before_expiration
            )

        if args.reconcile_only:
            if args.alpaca_buy_limit_buffer_bps is None:
                args.alpaca_buy_limit_buffer_bps = AlpacaOrderConfig.buy_limit_buffer_bps
            if args.tradier_fallback is None:
                args.tradier_fallback = TradierMarketDataConfig.enabled
            if args.tradier_timeout_seconds is None:
                args.tradier_timeout_seconds = TradierMarketDataConfig.timeout_seconds
            return

        if args.alpaca_buy_limit_buffer_bps is None:
            args.alpaca_buy_limit_buffer_bps = _env_float(
                "ALPACA_BUY_LIMIT_BUFFER_BPS",
                AlpacaOrderConfig.buy_limit_buffer_bps,
                minimum=0.0,
                maximum=10_000.0,
            )
        if args.tradier_fallback is None:
            args.tradier_fallback = _env_bool(
                "TRADIER_FALLBACK_ENABLED",
                TradierMarketDataConfig.enabled,
            )
        if args.tradier_timeout_seconds is None:
            args.tradier_timeout_seconds = (
                _env_int(
                    "TRADIER_TIMEOUT_SECONDS",
                    TradierMarketDataConfig.timeout_seconds,
                    minimum=1,
                    maximum=600,
                )
                if args.tradier_fallback
                else TradierMarketDataConfig.timeout_seconds
            )
    except ValueError as exc:
        parser.error(str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    arguments = sys.argv[1:]
    for argument in arguments:
        for option, environment_name in REMOVED_CREDENTIAL_CLI_FLAGS.items():
            if argument == option or argument.startswith(f"{option}="):
                parser.error(
                    f"{option} is no longer supported; set {environment_name} in the environment "
                    "or a private .env file."
                )
    parser.add_argument(
        "--mode",
        choices=["update", "rebuild"],
        default="update",
        help="update resumes saved strategy state; rebuild recomputes all state from scratch.",
    )
    parser.add_argument(
        "--reconcile-only",
        action="store_true",
        help="Only reconcile existing managed Alpaca positions; skip universe downloads and strategy work.",
    )
    parser.add_argument(
        "--db",
        type=_database_path,
        default=SQLITE_DB_PATH,
        help=(
            "Persistent SQLite file used for market data, RSI values, equity history, and strategy state; "
            "in-memory and hard-linked databases are not supported."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=_output_directory_path,
        default="outputs",
        help="Directory for generated CSV reports and Alpaca order result files.",
    )
    parser.add_argument(
        "--alpaca-submit-buy-orders",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Submit bounded Alpaca paper day limit buy orders for current buy recommendations. "
            "The batch uses a capped fraction of account cash to size whole-share quantities. Disabled by default."
        ),
    )
    parser.add_argument(
        "--alpaca-submit-sell-orders",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Submit and renew Alpaca paper managed GTC limit sell orders for filled managed buys. "
            "Each order sells the remaining whole-share managed quantity at the frozen target price. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--alpaca-base-url",
        default=os.environ.get("ALPACA_BASE_URL", ALPACA_PAPER_BASE_URL),
        help=("Alpaca Trading API base URL. Access is restricted to https://paper-api.alpaca.markets (default)."),
    )
    parser.add_argument(
        "--alpaca-timeout-seconds",
        type=_timeout_seconds,
        default=30,
        help="Timeout for Alpaca API requests.",
    )
    parser.add_argument(
        "--alpaca-buy-limit-buffer-bps",
        type=_buy_limit_buffer_bps,
        default=None,
        help="Basis-point buffer above the price estimate for whole-share day buy limits (default: 500).",
    )
    parser.add_argument(
        "--alpaca-gtc-sell-renewal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Renew managed Alpaca GTC limit sells before Alpaca's aged-order expiration. "
            "Enabled by default; use --no-alpaca-gtc-sell-renewal to skip."
        ),
    )
    parser.add_argument(
        "--alpaca-gtc-sell-renewal-days-before-expiration",
        type=_renewal_days,
        default=None,
        help="Renew managed Alpaca GTC limit sells this many days before expiration.",
    )
    parser.add_argument(
        "--tradier-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use Tradier historical daily data as a fallback when Yahoo Finance skips symbols. "
            "Tradier fallback requires --no-auto-adjust so provider price bases are not mixed. "
            "Use --no-tradier-fallback to skip."
        ),
    )
    parser.add_argument(
        "--auto-adjust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Yahoo Finance split/dividend-adjusted OHLC data (default: enabled). "
            "Disable adjustment to make raw-price Tradier fallback eligible."
        ),
    )
    parser.add_argument(
        "--tradier-base-url",
        default=os.environ.get("TRADIER_BASE_URL", TRADIER_LIVE_BASE_URL),
        help="Tradier Brokerage API base URL for market data fallback.",
    )
    parser.add_argument(
        "--tradier-timeout-seconds",
        type=_timeout_seconds,
        default=None,
        help="Timeout for Tradier market data requests.",
    )
    parser.add_argument(
        "--workflow-concurrency",
        type=_workflow_concurrency,
        default=DEFAULT_WORKFLOW_CONCURRENCY,
        help=(
            "Maximum number of concurrent asset download workers; SQLite strategy updates remain "
            "serialized. Use 1 for fully serial behavior."
        ),
    )
    parser.add_argument(
        "--strategy-state-verification",
        choices=["trusted", "canonical"],
        default="trusted",
        help=(
            "Verify authenticated local state and chronology before resuming (trusted, default), "
            "or additionally replay full canonical history for an explicit audit (canonical)."
        ),
    )
    parser.add_argument(
        "--scheduled-closed-audit-interval-minutes",
        type=_scheduled_closed_audit_interval_minutes,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--workflow-deadline-epoch",
        type=_workflow_deadline_epoch,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--require-workflow-source-success",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Abort before running strategies if an enabled universe discovery or active listing "
            "source fails. Enabled by default with Alpaca buy submission and disabled otherwise; "
            "source failures are otherwise recorded as a degraded universe."
        ),
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored terminal output.",
    )
    parser.add_argument(
        "--show-timings",
        action="store_true",
        help="Show overlap-aware workflow phase timings and market-data work counts.",
    )
    args = parser.parse_args()
    _resolve_env_backed_defaults(parser, args)
    if args.require_workflow_source_success is None:
        args.require_workflow_source_success = bool(args.alpaca_submit_buy_orders)
    if not args.reconcile_only:
        for name, message in REMOVED_ENV_VARS.items():
            if name in os.environ:
                parser.error(message)
    if args.reconcile_only and args.alpaca_submit_buy_orders:
        parser.error("--reconcile-only cannot be combined with --alpaca-submit-buy-orders.")
    if args.reconcile_only and not args.alpaca_submit_sell_orders:
        parser.error("--reconcile-only requires --alpaca-submit-sell-orders.")
    if args.scheduled_closed_audit_interval_minutes is not None and not args.reconcile_only:
        parser.error(
            "--scheduled-closed-audit-interval-minutes requires --reconcile-only."
        )
    try:
        alpaca_cfg = _alpaca_config_from_args(args)
        if args.reconcile_only:
            validate_alpaca_reconciliation_configuration(alpaca_cfg)
        else:
            validate_runtime_configuration(
                base_cfg=BacktestConfig(auto_adjust=args.auto_adjust),
                universe_cfg=UniverseConfig(
                    sqlite_db_path=args.db,
                    require_workflow_source_success=args.require_workflow_source_success,
                ),
                alpaca_cfg=alpaca_cfg,
                tradier_cfg=_tradier_config_from_args(args),
                workflow_concurrency=args.workflow_concurrency,
            )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        parse_args()
        return
    try:
        load_dotenv()
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
    args = parse_args()
    alpaca_cfg = _alpaca_config_from_args(args)
    if args.reconcile_only:
        run_alpaca_reconciliation(
            db_path=args.db,
            alpaca_cfg=alpaca_cfg,
            output_dir=args.output_dir,
            no_color=args.no_color,
            closed_audit_min_interval_minutes=args.scheduled_closed_audit_interval_minutes,
        )
        return

    base_cfg = BacktestConfig(
        initial_capital=100_000,
        rsi_period=14,
        buy_rsi=30,
        profit_target_multiple=10.0,
        fee_bps=1.0,
        slippage_bps=2.0,
        auto_adjust=args.auto_adjust,
    )

    mode = args.mode
    universe_cfg = UniverseConfig(
        top_n=None,
        sqlite_db_path=args.db,
        require_workflow_source_success=args.require_workflow_source_success,
    )
    tradier_cfg = _tradier_config_from_args(args)
    buy_rsi_values = list(range(20, 51))
    short_buy_rsi_values = list(range(50, 81))
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
        short_buy_rsi_values=short_buy_rsi_values,
        strategy_state_verification=args.strategy_state_verification,
        workflow_deadline_epoch=args.workflow_deadline_epoch,
        show_timings=args.show_timings,
    )
