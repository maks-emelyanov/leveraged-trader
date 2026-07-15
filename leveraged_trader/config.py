from __future__ import annotations

import os
from dataclasses import dataclass

RISK_FREE_SYMBOL = "^IRX"
ETF_DEFS_URL = "https://www.nasdaqtrader.com/trader.aspx?id=etf_definitions"
SQLITE_DB_PATH = "strategy_state.sqlite"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
TRADIER_LIVE_BASE_URL = "https://api.tradier.com/v1"
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
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def validate_alpaca_paper_endpoint(
    cfg: AlpacaOrderConfig,
    *,
    unconditional: bool = False,
) -> None:
    """Refuse non-paper endpoints when orders are enabled or validation is unconditional."""
    if not unconditional and not (cfg.enabled or cfg.sell_enabled):
        return
    normalized_base_url = cfg.base_url[:-1] if cfg.base_url.endswith("/") else cfg.base_url
    if normalized_base_url != ALPACA_PAPER_BASE_URL:
        raise ValueError(
            "Alpaca order submission is restricted to https://paper-api.alpaca.markets; "
            "live and other custom trading endpoints are not supported."
        )
