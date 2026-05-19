from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


RISK_FREE_SYMBOL = "^IRX"
ETF_DEFS_URL = "https://www.nasdaqtrader.com/trader.aspx?id=etf_definitions"
SQLITE_DB_PATH = "strategy_state.sqlite"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DOTENV_PATH = ".env"


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    rsi_period: int = 14
    buy_rsi: float = 30.0
    profit_target_multiple: float = 10.0
    fee_bps: float = 1.0        # commission-like cost per trade notional
    slippage_bps: float = 2.0   # slippage per trade notional
    auto_adjust: bool = True


@dataclass
class UniverseConfig:
    request_timeout_seconds: int = 30
    top_n: Optional[int] = None
    sqlite_db_path: str = SQLITE_DB_PATH


@dataclass
class AlpacaOrderConfig:
    enabled: bool = False
    sell_enabled: bool = False
    api_key_id: Optional[str] = None
    api_secret_key: Optional[str] = None
    base_url: str = ALPACA_PAPER_BASE_URL
    cash_fraction: float = 0.10
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
