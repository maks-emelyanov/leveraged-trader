from __future__ import annotations

import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from urllib.parse import urlsplit

from .runtime_files import open_private_text_file

RISK_FREE_SYMBOL = "^IRX"
ETF_DEFS_URL = "https://www.nasdaqtrader.com/trader.aspx?id=etf_definitions"
SQLITE_DB_PATH = "strategy_state.sqlite"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
TRADIER_LIVE_BASE_URL = "https://api.tradier.com/v1"
TRADIER_API_HOSTS = frozenset({"api.tradier.com", "sandbox.tradier.com"})
TRADIER_PLACEHOLDER_TOKENS = frozenset(
    {
        "",
        "replace_with_your_tradier_access_token",
        "your_tradier_access_token",
        "your_tradier_api_token",
        "replace_me",
    }
)
DOTENV_PATH = ".env"


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    rsi_period: int = 14
    buy_rsi: float = 30.0
    profit_target_multiple: float = 10.0
    fee_bps: float = 1.0  # commission-like cost per trade notional
    slippage_bps: float = 2.0  # slippage per trade notional
    auto_adjust: bool = True


@dataclass
class UniverseConfig:
    request_timeout_seconds: int = 30
    top_n: int | None = None
    sqlite_db_path: str = SQLITE_DB_PATH
    require_workflow_source_success: bool = False


@dataclass
class AlpacaOrderConfig:
    enabled: bool = False
    sell_enabled: bool = False
    api_key_id: str | None = None
    api_secret_key: str | None = None
    base_url: str = ALPACA_PAPER_BASE_URL
    buy_limit_buffer_bps: float = 500.0
    timeout_seconds: int = 30
    gtc_sell_renewal_enabled: bool = True
    gtc_sell_renewal_days_before_expiration: int = 7


@dataclass
class TradierMarketDataConfig:
    enabled: bool = True
    access_token: str | None = None
    base_url: str = TRADIER_LIVE_BASE_URL
    timeout_seconds: int = 30


def load_dotenv(path: str = DOTENV_PATH) -> None:
    try:
        with open_private_text_file(path, label="Environment file") as env_file:
            _load_dotenv_values(env_file)
    except FileNotFoundError:
        return


def _load_dotenv_values(env_file: Iterable[str]) -> None:
    pending_values: dict[str, str] = {}
    for raw_line in env_file:
        if "\0" in raw_line:
            raise ValueError("Environment file must not contain NUL bytes.")
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')):
            opening_quote = value[0]
            if len(value) < 2 or value[-1] != opening_quote:
                raise ValueError(f"Environment value for {key or '<empty key>'!r} has an unmatched opening quote.")
            # Remove exactly one syntactic pair. Additional quotes inside that
            # pair are part of the value and must not be silently discarded.
            value = value[1:-1]
        if key and key not in os.environ and key not in pending_values:
            pending_values[key] = value

    applied_values: list[tuple[str, str]] = []
    try:
        for key, value in pending_values.items():
            # Preserve exported environment values even if another thread set
            # one while this file was being parsed.
            if key not in os.environ:
                os.environ[key] = value
                applied_values.append((key, value))
    except BaseException:
        # Parsed dotenv keys were absent before this call, so remove only values
        # applied by this batch if an unexpected platform-level assignment
        # failure prevents an all-or-nothing update. Preserve a value that a
        # concurrent writer replaced after this batch assigned it.
        for key, value in reversed(applied_values):
            if os.environ.get(key) == value:
                os.environ.pop(key, None)
        raise


def is_official_tradier_api_base_url(base_url: str) -> bool:
    """Return whether a URL names a supported canonical Tradier API root."""
    normalized = base_url.strip()
    if (
        not normalized
        or "?" in normalized
        or "#" in normalized
        or any(character.isspace() or not character.isprintable() for character in normalized)
    ):
        return False
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
    except ValueError:
        return False

    hostname = (parsed.hostname or "").lower()
    canonical_authorities = {hostname, f"{hostname}:443"}
    return (
        parsed.scheme.lower() == "https"
        and hostname in TRADIER_API_HOSTS
        and parsed.netloc.lower() in canonical_authorities
        and port in {None, 443}
        and parsed.path in {"", "/", "/v1", "/v1/"}
        and parsed.username is None
        and parsed.password is None
    )


def validate_alpaca_paper_endpoint(
    cfg: AlpacaOrderConfig,
    *,
    unconditional: bool = False,
) -> None:
    """Refuse non-paper endpoints when orders are enabled or validation is unconditional."""
    base_url = cfg.base_url
    if type(base_url) is not str:
        raise ValueError("Alpaca base_url must be a string.")
    if not unconditional and not (cfg.enabled or cfg.sell_enabled):
        return
    normalized_base_url = base_url[:-1] if base_url.endswith("/") else base_url
    if normalized_base_url != ALPACA_PAPER_BASE_URL:
        raise ValueError(
            "Alpaca order submission is restricted to https://paper-api.alpaca.markets; "
            "live and other custom trading endpoints are not supported."
        )


def _require_bounded_number(
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number between {minimum:g} and {maximum:g}.")
    try:
        source_within_bounds = minimum <= value <= maximum
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number between {minimum:g} and {maximum:g}.") from exc
    if not source_within_bounds:
        raise ValueError(f"{name} must be a finite number between {minimum:g} and {maximum:g}.")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number between {minimum:g} and {maximum:g}.") from exc
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(f"{name} must be a finite number between {minimum:g} and {maximum:g}.")
    return numeric


def _require_bounded_integer(
    name: str,
    value: object,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
    return value


def _require_boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _validate_backtest_configuration_fields(
    base_cfg: BacktestConfig,
    *,
    minimum_initial_capital: float,
    require_strictly_positive_initial_capital: bool,
    minimum_rsi_period: int,
) -> None:
    if not isinstance(base_cfg, BacktestConfig):
        raise ValueError("base_cfg must be a BacktestConfig instance.")
    initial_capital = _require_bounded_number(
        "initial_capital",
        base_cfg.initial_capital,
        minimum=minimum_initial_capital,
        maximum=1e15,
    )
    if require_strictly_positive_initial_capital and initial_capital <= 0.0:
        raise ValueError("initial_capital must be a finite number greater than 0 and at most 1e15.")
    _require_bounded_integer(
        "rsi_period",
        base_cfg.rsi_period,
        minimum=minimum_rsi_period,
        maximum=10_000,
    )
    _require_bounded_number("buy_rsi", base_cfg.buy_rsi, minimum=0.0, maximum=100.0)
    profit_target_multiple = _require_bounded_number(
        "profit_target_multiple",
        base_cfg.profit_target_multiple,
        minimum=1.0,
        maximum=100.0,
    )
    if profit_target_multiple <= 1.0:
        raise ValueError("profit_target_multiple must be a finite number greater than 1.0 and at most 100.0.")
    fee_bps = _require_bounded_number("fee_bps", base_cfg.fee_bps, minimum=0.0, maximum=10_000.0)
    slippage_bps = _require_bounded_number(
        "slippage_bps",
        base_cfg.slippage_bps,
        minimum=0.0,
        maximum=10_000.0,
    )
    if fee_bps + slippage_bps >= 10_000.0:
        raise ValueError("fee_bps and slippage_bps must total less than 10000.")
    _require_boolean("auto_adjust", base_cfg.auto_adjust)


def validate_backtest_configuration(base_cfg: BacktestConfig) -> None:
    """Validate every backtest setting without requiring unrelated runtime config."""
    _validate_backtest_configuration_fields(
        base_cfg,
        minimum_initial_capital=0.01,
        require_strictly_positive_initial_capital=False,
        minimum_rsi_period=2,
    )


def validate_strategy_simulation_configuration(base_cfg: BacktestConfig) -> None:
    """Validate direct simulation inputs, including tiny positive test scales."""
    _validate_backtest_configuration_fields(
        base_cfg,
        minimum_initial_capital=0.0,
        require_strictly_positive_initial_capital=True,
        minimum_rsi_period=1,
    )


def validate_runtime_configuration(
    *,
    base_cfg: BacktestConfig,
    universe_cfg: UniverseConfig,
    alpaca_cfg: AlpacaOrderConfig,
    tradier_cfg: TradierMarketDataConfig | None,
    workflow_concurrency: int,
) -> None:
    """Validate every externally configurable runtime value before side effects."""
    validate_backtest_configuration(base_cfg)

    _require_bounded_integer(
        "universe request_timeout_seconds",
        universe_cfg.request_timeout_seconds,
        minimum=1,
        maximum=600,
    )
    if universe_cfg.top_n is not None:
        _require_bounded_integer("universe top_n", universe_cfg.top_n, minimum=1, maximum=100_000)
    _require_boolean(
        "universe require_workflow_source_success",
        universe_cfg.require_workflow_source_success,
    )

    _require_bounded_integer("workflow_concurrency", workflow_concurrency, minimum=1, maximum=64)
    _validate_alpaca_order_configuration(alpaca_cfg, validate_buy_settings=True)

    if tradier_cfg is not None:
        _require_boolean("Tradier enabled", tradier_cfg.enabled)
        _require_bounded_integer("Tradier timeout_seconds", tradier_cfg.timeout_seconds, minimum=1, maximum=600)
        if tradier_cfg.access_token is not None and type(tradier_cfg.access_token) is not str:
            raise ValueError("Tradier access_token must be a string or null.")
        if type(tradier_cfg.access_token) is str:
            access_token = tradier_cfg.access_token.strip(" ")
            if access_token and any(character.isspace() or not character.isprintable() for character in access_token):
                raise ValueError("Tradier access_token must not contain whitespace or control characters.")
            try:
                access_token.encode("latin-1")
            except UnicodeEncodeError as exc:
                raise ValueError("Tradier access_token must contain only HTTP-header-compatible characters.") from exc
            tradier_cfg.access_token = access_token or None
        if type(tradier_cfg.base_url) is not str:
            raise ValueError("Tradier base_url must be a string.")
        if (
            tradier_cfg.enabled
            and tradier_cfg.access_token
            and not is_official_tradier_api_base_url(tradier_cfg.base_url)
        ):
            raise ValueError(
                "Tradier bearer credentials are restricted to the official "
                "https://api.tradier.com or https://sandbox.tradier.com API root, "
                "optionally followed by /v1."
            )


def validate_alpaca_reconciliation_configuration(alpaca_cfg: AlpacaOrderConfig) -> None:
    """Validate only settings used while reconciling protective sell orders."""
    _validate_alpaca_order_configuration(alpaca_cfg, validate_buy_settings=False)


def validate_alpaca_buy_configuration(alpaca_cfg: AlpacaOrderConfig) -> None:
    """Validate every setting that can affect public Alpaca buy submission."""
    _validate_alpaca_order_configuration(alpaca_cfg, validate_buy_settings=True)


def validate_alpaca_paper_credentials(alpaca_cfg: AlpacaOrderConfig) -> tuple[str, str]:
    """Normalize surrounding spaces and reject header-unsafe credentials."""
    raw_api_key_id = alpaca_cfg.api_key_id if isinstance(alpaca_cfg.api_key_id, str) else ""
    raw_api_secret_key = alpaca_cfg.api_secret_key if isinstance(alpaca_cfg.api_secret_key, str) else ""
    if not raw_api_key_id.strip() or not raw_api_secret_key.strip():
        raise ValueError(
            "Enabled Alpaca paper trading requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY "
            "in the environment or a private .env file."
        )
    api_key_id = raw_api_key_id.strip(" ")
    api_secret_key = raw_api_secret_key.strip(" ")
    if any(
        character.isspace() or not character.isprintable()
        for credential in (api_key_id, api_secret_key)
        for character in credential
    ):
        raise ValueError("Alpaca paper credentials must not contain whitespace or control characters.")
    try:
        api_key_id.encode("latin-1")
        api_secret_key.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError("Alpaca paper credentials must contain only HTTP-header-compatible characters.") from exc
    alpaca_cfg.api_key_id = api_key_id
    alpaca_cfg.api_secret_key = api_secret_key
    placeholders = {
        "your_alpaca_paper_api_key_id",
        "your_alpaca_paper_api_secret_key",
    }
    if api_key_id.casefold() in placeholders or api_secret_key.casefold() in placeholders:
        raise ValueError("Replace the placeholder Alpaca credentials before enabling paper trading.")
    return api_key_id, api_secret_key


def _validate_alpaca_order_configuration(
    alpaca_cfg: AlpacaOrderConfig,
    *,
    validate_buy_settings: bool,
) -> None:
    _require_boolean("Alpaca enabled", alpaca_cfg.enabled)
    _require_boolean("Alpaca sell_enabled", alpaca_cfg.sell_enabled)
    _require_boolean("Alpaca gtc_sell_renewal_enabled", alpaca_cfg.gtc_sell_renewal_enabled)
    if validate_buy_settings:
        alpaca_cfg.buy_limit_buffer_bps = _require_bounded_number(
            "Alpaca buy_limit_buffer_bps",
            alpaca_cfg.buy_limit_buffer_bps,
            minimum=0.0,
            maximum=10_000.0,
        )
    _require_bounded_integer("Alpaca timeout_seconds", alpaca_cfg.timeout_seconds, minimum=1, maximum=600)
    if alpaca_cfg.gtc_sell_renewal_enabled:
        _require_bounded_integer(
            "Alpaca gtc_sell_renewal_days_before_expiration",
            alpaca_cfg.gtc_sell_renewal_days_before_expiration,
            minimum=0,
            maximum=90,
        )
    validate_alpaca_paper_endpoint(alpaca_cfg)
    if alpaca_cfg.enabled or alpaca_cfg.sell_enabled:
        validate_alpaca_paper_credentials(alpaca_cfg)
