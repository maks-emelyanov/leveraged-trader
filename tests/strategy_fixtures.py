"""Explicit configuration for historical tests of the old Sharpe-only selector."""

from dataclasses import dataclass

from leveraged_trader.config import BacktestConfig


@dataclass
class LegacyBacktestConfig(BacktestConfig):
    selection_method: str = "legacy_sharpe"
