from __future__ import annotations

import io
import re
from typing import Optional

import pandas as pd
import requests

from .config import ETF_DEFS_URL, UniverseConfig
from .storage import save_table_to_sqlite


TICKER_STOPWORDS = {
    "ETF",
    "ETFS",
    "ETN",
    "ETNS",
    "FUND",
    "TRUST",
    "SHARES",
    "SHARE",
    "DAILY",
    "TARGET",
    "LONG",
    "BULL",
    "BEAR",
    "SHORT",
    "ULTRA",
    "ULTRAPRO",
    "LEVERAGED",
    "DIREXION",
    "PROSHARES",
    "GRANITESHARES",
    "DEFIANCE",
    "TRADR",
    "REX",
}

RSI_SYMBOL_PATTERNS = [
    r"\b(?:2X|3X|1\.5X|200%|300%)\s+(?:DAILY\s+)?(?:TARGET\s+)?(?:LONG|BULL)\s+([A-Z][A-Z0-9.-]{0,5})\b",
    r"\b(?:LONG|BULL)\s+([A-Z][A-Z0-9.-]{0,5})\s+(?:DAILY\s+)?(?:ETF|ETN|SHARES?)\b",
    r"\b([A-Z][A-Z0-9.-]{0,5})\s+(?:DAILY\s+)?(?:LONG|BULL)\b",
    r"\b([A-Z][A-Z0-9.-]{0,5})\s+(?:2X|3X|1\.5X|200%|300%)\b",
    r"\b(?:ULTRAPRO|ULTRA)\s+([A-Z][A-Z0-9.-]{0,5})\b",
]


def _normalize_symbol_candidate(symbol: str) -> Optional[str]:
    candidate = symbol.strip(" .,-").upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,5}", candidate):
        return None
    if candidate in TICKER_STOPWORDS:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?X", candidate):
        return None
    return candidate


def infer_rsi_symbol(asset_symbol: str, fund_name: str) -> str:
    """
    Infer the unleveraged signal ticker from a leveraged ETF name.

    Most single-stock leveraged ETF names include the underlying ticker near
    phrases like "2x Long", "Bull", or "UltraPro". If no reliable ticker is
    present, fall back to the leveraged ETF itself.
    """
    asset_symbol = asset_symbol.upper()
    normalized_name = re.sub(r"\s+", " ", fund_name.upper()).strip()

    for pattern in RSI_SYMBOL_PATTERNS:
        match = re.search(pattern, normalized_name)
        if not match:
            continue
        candidate = _normalize_symbol_candidate(match.group(1))
        if candidate is not None and candidate != asset_symbol:
            return candidate

    return asset_symbol


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def _infer_column(columns: list[str], pattern: str) -> Optional[str]:
    regex = re.compile(pattern, re.I)
    for col in columns:
        if regex.search(col):
            return col
    return None


def load_current_etf_universe(timeout: int = 30) -> pd.DataFrame:
    """
    Load the current Nasdaq Trader ETF definitions table.
    Free public source for the ETF universe.
    """
    resp = requests.get(ETF_DEFS_URL, timeout=timeout)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    if not tables:
        raise RuntimeError("No tables found on Nasdaq Trader ETF definitions page.")

    df = _clean_columns(max(tables, key=len))
    cols = list(df.columns)

    symbol_col = _infer_column(cols, r"\bsymbol\b")
    fund_type_col = _infer_column(cols, r"\bfund type\b|^type$")
    if symbol_col is None or fund_type_col is None:
        raise RuntimeError(
            f"Could not infer required columns from ETF definitions page. Columns found: {cols}"
        )

    try:
        sym_idx = cols.index(symbol_col)
        type_idx = cols.index(fund_type_col)
        candidate_name_cols = cols[sym_idx + 1:type_idx]
        if candidate_name_cols:
            name_col = candidate_name_cols[0]
        else:
            object_cols = [
                c for c in cols if df[c].dtype == object and c not in {symbol_col, fund_type_col}
            ]
            name_col = max(object_cols, key=lambda c: df[c].astype(str).str.len().mean())
    except Exception as exc:
        raise RuntimeError(f"Could not infer fund name column. Columns found: {cols}") from exc

    out = df[[symbol_col, name_col, fund_type_col]].copy()
    out.columns = ["symbol", "name", "fund_type"]
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["name"] = out["name"].astype(str).str.strip()
    out["fund_type"] = out["fund_type"].astype(str).str.strip()

    out = out[out["fund_type"].str.startswith("ETF", na=False)].drop_duplicates(subset=["symbol"])
    out = out[out["symbol"].str.fullmatch(r"[A-Z\.]+", na=False)]
    return out.reset_index(drop=True)


LONG_LEVERAGED_PATTERNS = [
    r"\b2x\b",
    r"\b3x\b",
    r"\b1\.5x\b",
    r"\b200%\b",
    r"\b300%\b",
    r"\bultra\b",
    r"\bultrapro\b",
    r"\bbull\b",
    r"\blong\b",
    r"\bdaily target 2x long\b",
    r"\bdaily target 3x long\b",
]

INVERSE_PATTERNS = [
    r"\bbear\b",
    r"\bshort\b",
    r"\binverse\b",
    r"\bultrashort\b",
    r"\b-1x\b",
    r"\b-2x\b",
    r"\b-3x\b",
]


def is_long_leveraged_name(name: str) -> bool:
    n = name.lower()
    if any(re.search(p, n) for p in INVERSE_PATTERNS):
        return False
    if any(re.search(p, n) for p in LONG_LEVERAGED_PATTERNS):
        return True
    if "direxion daily" in n and "bull" in n:
        return True
    if "graniteshares 2x long" in n:
        return True
    if "leverage shares 2x long" in n:
        return True
    if "defiance daily target 2x long" in n:
        return True
    if "tradr 2x long" in n:
        return True
    return False


def select_universes(etf_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns current long leveraged single-stock ETFs and all current long leveraged ETFs.
    """
    single_stock = etf_df[
        etf_df["fund_type"].str.contains(r"ETF \(Single Stock\)", regex=True, na=False)
    ].copy()
    single_stock_long = single_stock[single_stock["name"].apply(is_long_leveraged_name)].copy()
    all_long_leveraged = etf_df[etf_df["name"].apply(is_long_leveraged_name)].copy()

    return (
        single_stock_long.sort_values("symbol").reset_index(drop=True),
        all_long_leveraged.sort_values("symbol").reset_index(drop=True),
    )


def determine_workflow_assets(cfg: UniverseConfig) -> pd.DataFrame:
    etf_df = load_current_etf_universe(timeout=cfg.request_timeout_seconds)
    single_stock_long, all_long_leveraged = select_universes(etf_df)
    nasdaq_universe = build_nasdaq_universe_table(etf_df)
    save_table_to_sqlite(nasdaq_universe, cfg.sqlite_db_path, "nasdaq_etf_universe")

    print(f"Current ETFs in Nasdaq table: {len(etf_df)}")
    print(f"Current long single-stock leveraged ETFs found: {len(single_stock_long)}")
    print(f"Current long leveraged ETFs found: {len(all_long_leveraged)}")
    print(f"Saved Nasdaq ETF universe to {cfg.sqlite_db_path}: table=nasdaq_etf_universe")

    if all_long_leveraged.empty:
        raise RuntimeError("Nasdaq ETF universe returned no current long leveraged ETFs.")

    workflow_assets = (
        all_long_leveraged.copy()
        if cfg.top_n is None
        else all_long_leveraged.head(cfg.top_n).copy()
    )

    workflow_assets["rsi_symbol"] = workflow_assets.apply(
        lambda row: infer_rsi_symbol(row["symbol"], row["name"]),
        axis=1,
    )

    if cfg.top_n is None:
        print("\nAll long leveraged ETFs from Nasdaq universe")
    else:
        print(f"\nFirst {cfg.top_n} long leveraged ETFs from Nasdaq universe")
    print(workflow_assets.to_string(index=False))

    return workflow_assets[["symbol", "name", "rsi_symbol"]].reset_index(drop=True)


def determine_workflow_symbols(cfg: UniverseConfig) -> list[str]:
    return determine_workflow_assets(cfg)["symbol"].tolist()


def build_nasdaq_universe_table(etf_df: pd.DataFrame) -> pd.DataFrame:
    out = etf_df.copy()
    out["is_long_leveraged"] = out["name"].apply(is_long_leveraged_name)
    out["is_single_stock"] = out["fund_type"].str.contains(
        r"ETF \(Single Stock\)",
        regex=True,
        na=False,
    )
    out["is_single_stock_long_leveraged"] = out["is_single_stock"] & out["is_long_leveraged"]
    out["rsi_symbol"] = out.apply(
        lambda row: infer_rsi_symbol(row["symbol"], row["name"]) if row["is_long_leveraged"] else row["symbol"],
        axis=1,
    )
    return out.sort_values("symbol").reset_index(drop=True)
