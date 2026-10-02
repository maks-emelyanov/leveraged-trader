"""Issuer-confirmed closures still advertised in stale product inventories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class ProductClosure:
    name: str
    last_trading_date: date
    source_url: str


# Require both the symbol and complete product name. A reused ticker must not
# inherit an unrelated product's closure. Keep the issuer evidence with each
# entry rather than inferring closure from a failed price download or absence
# from an exchange directory (which can lag new listings).
PRODUCT_CLOSURES = {
    symbol: ProductClosure(
        f"ETRACS 2x Leveraged {factor} Factor TR ETN",
        date(2026, 8, 18),
        "https://etracs.ubs.com/news/show-article/id/724",
    )
    for symbol, factor in {
        "IWDL": "US Value",
        "IWFL": "US Growth",
        "IWML": "US Size",
        "MTUL": "MSCI US Momentum",
        "QULL": "MSCI US Quality",
        "SCDL": "US Dividend",
        "USML": "MSCI US Minimum Volatility",
    }.items()
}
PRODUCT_CLOSURES["PAAU"] = ProductClosure(
    "T-REX 2X Long PAAS Daily Target ETF",
    date(2026, 7, 10),
    "https://www.rexshares.com/t-rex-2x-long-paas-daily-target-etf-paau-to-liquidate/",
)


def _name_identity(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def split_closed_products(products: pd.DataFrame, *, as_of: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return current rows and an audit of closures effective before as_of."""
    reasons = []
    sources = []
    for row in products.itertuples(index=False):
        closure = PRODUCT_CLOSURES.get(str(row.symbol).strip().upper())
        if (
            closure is not None
            and as_of > closure.last_trading_date
            and _name_identity(row.name) == _name_identity(closure.name)
        ):
            sources.append(closure.source_url)
            reasons.append(f"issuer-confirmed closure; last trading date {closure.last_trading_date.isoformat()}")
        else:
            sources.append("")
            reasons.append("")
    closed = pd.Series([bool(reason) for reason in reasons], index=products.index, dtype=bool)
    inactive = products.loc[closed].copy()
    inactive["inactive_source"] = pd.Series(sources, index=products.index, dtype=str).loc[closed]
    inactive["inactive_reason"] = pd.Series(reasons, index=products.index, dtype=str).loc[closed]
    return products.loc[~closed].copy(), inactive
