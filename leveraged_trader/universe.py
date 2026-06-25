from __future__ import annotations

import io
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from html import unescape

import pandas as pd
import requests

from .config import ETF_DEFS_URL, UniverseConfig
from .storage import save_table_to_sqlite

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_MUTUAL_FUND_TICKERS_URL = "https://www.sec.gov/files/company_tickers_mf.json"
REQUEST_HEADERS = {
    "User-Agent": "leveraged-trader/0.1 universe audit contact@example.com",
    "Accept": "text/html,text/csv,application/json;q=0.8,*/*;q=0.5",
}


@dataclass(frozen=True)
class UniverseSource:
    name: str
    url: str
    source_type: str
    parser: str = "html"
    enabled: bool = True
    notes: str = ""


class ActiveListedSymbols(set[str]):
    """Active symbols plus the completeness of their exchange-source snapshot."""

    def __init__(self, symbols: Iterable[str], source_status: list[dict[str, object]]) -> None:
        super().__init__(symbols)
        self.source_status = source_status

    @property
    def is_complete(self) -> bool:
        return bool(self.source_status) and all(
            status.get("status") == "loaded" for status in self.source_status
        )


WORKFLOW_SOURCE_STATUS_COLUMNS = [
    "source",
    "source_type",
    "url",
    "status",
    "parsed_row_count",
    "row_count",
    "error",
]


def _workflow_source_status_row(
    *,
    source: str,
    source_type: str,
    url: str,
    status: str,
    parsed_row_count: int = 0,
    row_count: int = 0,
    error: str = "",
) -> dict[str, object]:
    return {
        "source": source,
        "source_type": source_type,
        "url": url,
        "status": status,
        "parsed_row_count": parsed_row_count,
        "row_count": row_count,
        "error": error,
    }


def _workflow_source_status(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        df.attrs.get("workflow_source_status", []),
        columns=WORKFLOW_SOURCE_STATUS_COLUMNS,
    )


def _workflow_source_parse_status(parsed_rows: pd.DataFrame, matched_rows: pd.DataFrame) -> tuple[str, str]:
    """Classify a fetched source without mistaking zero matches for parser failure."""
    if parsed_rows.empty:
        return "parse_error", "No product rows could be parsed from a successful source response."
    if matched_rows.empty:
        return "loaded_zero_matches", ""
    return "loaded", ""


WORKFLOW_ISSUER_SOURCES = [
    ("ProShares", "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs"),
    ("Direxion", "https://www.direxion.com/all-etfs"),
    ("Leverage Shares", "https://leverageshares.com/en-us/etps/"),
    ("Leverage Shares", "https://leverageshares.com/en-us/etps/leverage-shares-2x-long/"),
    ("GraniteShares", "https://www.graniteshares.com/etfs/"),
    ("Defiance", "https://www.defianceetfs.com/etfs/"),
    ("AdvisorShares", "https://advisorshares.com/etfs/"),
    ("AXS Investments", "https://www.axsinvestments.com/our-funds/"),
    ("Kurv", "https://www.kurvinvest.com/etfs"),
    ("Innovator", "https://www.innovatoretfs.com/etf/finder/"),
    ("Innovator", "https://www.innovatoretfs.com/define/etfs/"),
    ("Tuttle Capital", "https://www.tuttlecap.com/etfs"),
    ("Tradr", "https://www.tradretfs.com/"),
    ("REX Shares", "https://www.rexshares.com/learn-more-about-the-full-t-rex-leveraged-etf-lineup/"),
    ("KraneShares", "https://kraneshares.com/leveraged-etfs/"),
    ("Volatility Shares", "https://www.volatilityshares.com/"),
    ("21Shares", "https://www.21shares.com/en-us/etfs"),
    ("YieldMax", "https://www.yieldmaxetfs.com/our-etfs"),
    ("Tidal", "https://www.tidalfinancialgroup.com/etfs/"),
    ("Roundhill", "https://www.roundhillinvestments.com/etf/"),
    ("Themes", "https://www.themesetfs.com/etfs/"),
    ("Simplify", "https://www.simplify.us/etfs"),
]
ISSUER_UNIVERSE_SOURCES = WORKFLOW_ISSUER_SOURCES

WORKFLOW_ETN_SOURCES = [
    UniverseSource(
        "MicroSectors",
        "https://microsectors.com/",
        source_type="etn_issuer",
        parser="microsectors_html",
        notes="Workflow ETN source; products carry issuer credit risk and are not ETFs.",
    ),
    UniverseSource(
        "UBS ETRACS",
        "https://etracs.ubs.com/product/list/index/strategy/leverage",
        source_type="etn_issuer",
        parser="etracs_leverage_table",
        notes="Workflow ETN source; products carry issuer credit risk and are not ETFs.",
    ),
]

AUDIT_UNIVERSE_SOURCES = [
    UniverseSource(
        "NYSE exchange-traded products directory",
        "https://www.nyse.com/listings_directory/etf",
        source_type="exchange_directory",
        notes="Audit-only exchange directory; does not override issuer or Nasdaq rows.",
    ),
    UniverseSource(
        "Nasdaq funds/ETFs directory",
        "https://www.nasdaq.com/market-activity/funds-and-etfs",
        source_type="exchange_directory",
        parser="registered_only",
        enabled=False,
        notes="Registered as an audit backstop; page is dynamic and slow to fetch reliably.",
    ),
    UniverseSource(
        "Cboe listed products",
        "https://www.cboe.com/us/equities/market_statistics/listed_symbols/csv/",
        source_type="exchange_directory",
        parser="cboe_symbol_csv",
        notes="Symbol-only audit cross-check; product names are not available from this endpoint.",
    ),
    UniverseSource(
        "ETFdb leveraged ETF directory",
        "https://etfdb.com/etfs/leveraged/",
        source_type="third_party_audit",
        parser="registered_only",
        enabled=False,
        notes="Registered as an audit backstop; commonly Cloudflare-blocked from unattended fetches.",
    ),
    UniverseSource(
        "VettaFi ETF database",
        "https://www.vettafi.com/etf-database/",
        source_type="third_party_audit",
        parser="registered_only",
        enabled=False,
        notes="Registered as an audit backstop; not authoritative for mappings.",
    ),
    UniverseSource(
        "ETF.com ETF finder",
        "https://www.etf.com/etfanalytics/etf-finder",
        source_type="third_party_audit",
        parser="registered_only",
        enabled=False,
        notes="Registered as an audit backstop; commonly Cloudflare-blocked from unattended fetches.",
    ),
    UniverseSource(
        "SEC EDGAR company ticker registry",
        SEC_COMPANY_TICKERS_URL,
        source_type="filing_audit",
        parser="sec_company_tickers",
        notes=(
            "Live SEC audit seed from the official company ticker registry; issuer or exchange rows remain "
            "authoritative for workflow mappings."
        ),
    ),
    UniverseSource(
        "SEC EDGAR exchange ticker registry",
        SEC_COMPANY_TICKERS_EXCHANGE_URL,
        source_type="filing_audit",
        parser="sec_exchange_tickers",
        notes=(
            "Live SEC audit seed with ticker, registrant name, and exchange fields; used only to flag "
            "possible missing leveraged products."
        ),
    ),
    UniverseSource(
        "SEC EDGAR mutual fund ticker registry",
        SEC_MUTUAL_FUND_TICKERS_URL,
        source_type="filing_audit",
        parser="sec_mutual_fund_tickers",
        notes=(
            "Live SEC audit seed with CIK, series, class, and symbol fields. It is symbol-only, so it "
            "supports coverage checks but does not classify leverage by itself."
        ),
    ),
    UniverseSource(
        "SEC EDGAR full-text search",
        "https://www.sec.gov/edgar/search/",
        source_type="filing_audit",
        parser="registered_only",
        enabled=False,
        notes="Registered for future prospectus text audits when a stable public full-text API is available.",
    ),
]


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
    "NASDAQ",
    "CLOUD",
    "LED",
}

EXCLUDED_UNIVERSE_SYMBOLS = {
    "NASDAQ",
    # False positives where "long" describes duration/horizon rather than leverage.
    "BGGG",
    "TMNL",
    "VCLT",
    "VGLT",
}
YAHOO_SYMBOL_ALIASES = {
    "BRKB": "BRK-B",
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
}

RSI_SYMBOL_PATTERNS = [
    r"\b(?:2X|3X|1\.5X|200%|300%)\s+(?:DAILY\s+)?(?:TARGET\s+)?(?:LONG|BULL)\s+([A-Z][A-Z0-9.-]{0,5})\b",
    r"\b(?:LONG|BULL)\s+([A-Z][A-Z0-9.-]{0,5})\s+(?:DAILY\s+)?(?:ETF|ETN|SHARES?)\b",
    r"\b([A-Z][A-Z0-9.-]{0,5})\s+(?:DAILY\s+)?(?:LONG|BULL)\b",
    r"\b([A-Z][A-Z0-9.-]{0,5})\s+(?:2X|3X|1\.5X|200%|300%)\b",
    r"\b(?:ULTRAPRO|ULTRA)\s+([A-Z][A-Z0-9.-]{0,5})\b",
]


LEVERAGE_NAME_PATTERNS = [
    r"\b[+-]?\d+(?:\.\d+)?\s*x\b",
    r"\b[23]00%\b",
    r"\bultrapro\b",
    r"\bultra\b",
    r"\bbull\s+[23]x\b",
    r"\b(?:daily\s+)?target\s+[23]x\b",
    r"\bleveraged\b",
]

LONG_DIRECTION_PATTERNS = [
    r"\bbull\b",
    r"\blong\b",
    r"\blong\s+exposure\b",
    r"\bultra\b",
    r"\bleveraged\s+long\b",
    r"\bleveraged\s+exposure\b",
    r"\b[23]x\s+leveraged\b",
    r"\+[23]x\b",
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

LEVERAGE_FALSE_POSITIVE_TERMS = [
    "ULTRA SHORT TERM",
    "ULTRA-SHORT TERM",
    "ULTRASHORT TERM",
    "ULTRA SHORT INCOME",
    "ULTRA-SHORT INCOME",
    "ULTRA BUFFER",
    "ULTRA-BUFFER",
    "SHORT DURATION",
    "LONG TERM",
    "LONG-TERM",
    "LONG MUNICIPAL",
]

MAX_RECOGNIZED_LEVERAGE = 5.0

SEC_ENTITY_AUDIT_PARSERS = {
    "sec_company_tickers",
    "sec_exchange_tickers",
}

SEC_AUDIT_PRODUCT_CONTEXT_PATTERNS = [
    r"\bETFS?\b",
    r"\bETNS?\b",
    r"\bEXCHANGE[-\s]+TRADED\b",
    r"\bFUNDS?\b",
    r"\bPROSHARES\b",
    r"\bDIREXION\b",
    r"\bGRANITESHARES\b",
    r"\bDEFIANCE\b",
    r"\bTRADR\b",
    r"\bT-?REX\b",
    r"\bMICROSECTORS\b",
    r"\bETRACS\b",
    r"\bLEVERAGE\s+SHARES\b",
    r"\bYIELDMAX\b",
    r"\bAXS\b",
    r"\bKURV\b",
    r"\bTUTTLE\b",
    r"\bREX\s+SHARES\b",
]


def normalize_yahoo_symbol(symbol: object) -> str | None:
    if symbol is None or pd.isna(symbol):
        return None
    candidate = str(symbol).strip(" .,-").upper()
    candidate = YAHOO_SYMBOL_ALIASES.get(candidate, candidate)
    if candidate in {"", "NAN", "NONE", "NULL"}:
        return None
    if candidate in EXCLUDED_UNIVERSE_SYMBOLS:
        return None
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]*", candidate):
        return None
    return candidate


def _normalize_symbol_candidate(symbol: str, known_symbols: set[str] | None = None) -> str | None:
    candidate = normalize_yahoo_symbol(symbol)
    if candidate is None:
        return None
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,5}", candidate):
        return None
    if candidate in TICKER_STOPWORDS:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?X", candidate):
        return None
    if known_symbols is not None and candidate not in known_symbols:
        return None
    return candidate


def infer_rsi_symbol(asset_symbol: str, fund_name: str, known_symbols: set[str] | None = None) -> str:
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
        candidate = _normalize_symbol_candidate(match.group(1), known_symbols=known_symbols)
        if candidate is not None and candidate != asset_symbol:
            return candidate

    return asset_symbol


def infer_leverage_and_direction(name: str) -> tuple[float | None, str | None]:
    normalized_name = str(name).upper()
    leverage: float | None = None

    match = re.search(r"(?<![A-Z0-9])([+-]?\d+(?:\.\d+)?)\s*X(?![A-Z0-9])", normalized_name)
    if match:
        leverage = abs(float(match.group(1)))
        if leverage > MAX_RECOGNIZED_LEVERAGE:
            leverage = None
    elif re.search(r"\b200%\b", normalized_name):
        leverage = 2.0
    elif re.search(r"\b300%\b", normalized_name) or re.search(r"\bULTRAPRO\b", normalized_name):
        leverage = 3.0
    elif re.search(r"\b(?:ULTRA|ULTRASHORT)\b", normalized_name):
        leverage = 2.0

    direction: str | None
    if any(re.search(pattern, normalized_name, re.I) for pattern in INVERSE_PATTERNS):
        direction = "inverse"
    elif (
        any(re.search(pattern, normalized_name, re.I) for pattern in LONG_DIRECTION_PATTERNS)
        or leverage is not None
        and leverage > 1.0
    ):
        direction = "long"
    else:
        direction = None

    return leverage, direction


def leveraged_name_filter(name: str) -> bool:
    normalized_name = f" {str(name).upper()} "
    if any(term in normalized_name for term in LEVERAGE_FALSE_POSITIVE_TERMS):
        return False
    leverage, _direction = infer_leverage_and_direction(normalized_name)
    if leverage is not None:
        return leverage > 1.0
    return any(re.search(pattern, normalized_name, re.I) for pattern in LEVERAGE_NAME_PATTERNS)


def _read_nasdaq_symbol_file(url: str, timeout: int) -> pd.DataFrame:
    resp = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    lines = [
        line
        for line in resp.text.splitlines()
        if line and not line.startswith("File Creation Time")
    ]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")


def load_active_listed_symbols(timeout: int = 30) -> set[str]:
    symbols: set[str] = set()
    source_status: list[dict[str, object]] = []
    for source_name, url, symbol_col in [
        ("nasdaq_listed", NASDAQ_LISTED_URL, "Symbol"),
        ("other_listed", OTHER_LISTED_URL, "ACT Symbol"),
    ]:
        try:
            listed = _read_nasdaq_symbol_file(url, timeout)
            if symbol_col not in listed.columns:
                raise ValueError(f"missing expected {symbol_col!r} column")
            source_symbols = {
                symbol
                for raw_symbol in listed[symbol_col].dropna()
                if (symbol := normalize_yahoo_symbol(raw_symbol)) is not None
            }
        except Exception as exc:
            source_status.append(
                {
                    "source": source_name,
                    "url": url,
                    "symbol_column": symbol_col,
                    "status": "error",
                    "symbol_count": 0,
                    "error": str(exc),
                }
            )
            continue
        symbols.update(source_symbols)
        source_status.append(
            {
                "source": source_name,
                "url": url,
                "symbol_column": symbol_col,
                "status": "loaded",
                "symbol_count": len(source_symbols),
                "error": "",
            }
        )
    return ActiveListedSymbols(symbols, source_status)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df


def _infer_column(columns: list[str], pattern: str) -> str | None:
    regex = re.compile(pattern, re.I)
    for col in columns:
        if regex.search(col):
            return col
    return None


def _read_html_tables(html: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(io.StringIO(html), flavor=["lxml"])
    except ValueError:
        return []


def load_current_etf_universe(timeout: int = 30) -> pd.DataFrame:
    """
    Load the current Nasdaq Trader ETF definitions table.
    Free public source for the ETF universe.
    """
    resp = requests.get(ETF_DEFS_URL, timeout=timeout, headers=REQUEST_HEADERS)
    resp.raise_for_status()

    tables = _read_html_tables(resp.text)
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
    out["symbol"] = out["symbol"].astype(str).map(normalize_yahoo_symbol)
    out["name"] = out["name"].astype(str).str.strip()
    out["fund_type"] = out["fund_type"].astype(str).str.strip()

    out = out[out["symbol"].notna()]
    out = out[out["fund_type"].str.startswith("ETF", na=False)].drop_duplicates(subset=["symbol"])
    out = out[out["symbol"].str.fullmatch(r"[A-Z][A-Z0-9.-]*", na=False)]
    return out.reset_index(drop=True)


def is_long_leveraged_name(name: str) -> bool:
    n = str(name).upper()
    if not leveraged_name_filter(n):
        return False
    _leverage, direction = infer_leverage_and_direction(n)
    return direction != "inverse"


def _first_matching_column(columns: Iterable[object], patterns: list[str]) -> object | None:
    normalized = [(column, str(column).strip()) for column in columns]
    for pattern in patterns:
        regex = re.compile(pattern, re.I)
        for raw_column, column in normalized:
            if regex.search(column):
                return raw_column
    return None


def _html_text(value: object) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def _fund_rows_to_universe(
    rows: Iterable[dict[str, object]],
    source_name: str,
    *,
    source_label: str,
    require_leveraged: bool,
    product_type: str = "ETF",
) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])

    out = out[["symbol", "name"]].copy()
    out["symbol"] = out["symbol"].map(normalize_yahoo_symbol)
    out["name"] = out["name"].map(_html_text)
    out = out[out["symbol"].notna()]
    out = out[out["name"].ne("")]
    if require_leveraged:
        out = out[out["name"].apply(leveraged_name_filter)]
    out["fund_type"] = f"{product_type} ({source_name})"
    out["source"] = source_label
    return out.drop_duplicates("symbol").reset_index(drop=True)


def _fund_table_to_universe(
    table: pd.DataFrame,
    source_name: str,
    *,
    source_label: str,
    require_leveraged: bool,
) -> pd.DataFrame:
    symbol_col = _first_matching_column(
        table.columns,
        [
            r"^ticker$",
            r"\bticker\b",
            r"^symbol$",
            r"\bsymbol\b",
            r"fund\s+ticker",
        ],
    )
    name_col = _first_matching_column(
        table.columns,
        [
            r"fund\s+name",
            r"^name$",
            r"\bname\b",
            r"\bfund\b",
            r"^etf$",
            r"^etn$",
            r"product\s+name",
        ],
    )
    if symbol_col is None or name_col is None:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])

    rows = (
        {"symbol": row[symbol_col], "name": row[name_col]}
        for _idx, row in table[[symbol_col, name_col]].iterrows()
    )
    return _fund_rows_to_universe(
        rows,
        source_name,
        source_label=source_label,
        require_leveraged=require_leveraged,
    )


def _issuer_table_to_universe(table: pd.DataFrame, issuer: str) -> pd.DataFrame:
    return _fund_table_to_universe(
        table,
        issuer,
        source_label=f"{issuer} issuer table",
        require_leveraged=True,
    )


def _html_cards_to_universe(
    html: str,
    source_name: str,
    *,
    source_label: str,
    require_leveraged: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    webflow_grid_pattern = re.compile(
        r'class="tag is-ticker[^"]*">(?P<symbol>[^<]+)</div>\s*</div>\s*'
        r'<div class="grid_table_cell">\s*'
        r'<div[^>]*class="[^"]*u-weight-medium[^"]*"[^>]*>(?P<name>[^<]+)</div>',
        re.I | re.S,
    )
    webflow_sort_pattern = re.compile(
        r'<div fs-cmssort-field="IDENTIFIER" class="text-weight-xbold">(?P<symbol>[^<]+)</div>'
        r'.{0,900}?<div role="cell" class="table3_column">\s*'
        r'<div fs-cmssort-field="IDENTIFIER">(?P<name>[^<]+)</div>',
        re.I | re.S,
    )
    nav_dropdown_pattern = re.compile(
        r'href="/etf/(?P<slug>[a-z0-9.-]+)"[^>]*class="nav_dropdown_link[^"]*"[^>]*>'
        r'.{0,900}?<div class="u-display-inline">(?P<symbol>[^<]+)</div>'
        r'.{0,900}?<div class="nav_dropdown_link_caption">(?P<name>[^<]+)</div>',
        re.I | re.S,
    )

    for pattern in [webflow_grid_pattern, webflow_sort_pattern, nav_dropdown_pattern]:
        for match in pattern.finditer(html):
            rows.append({"symbol": match.group("symbol"), "name": match.group("name")})

    return _fund_rows_to_universe(
        rows,
        source_name,
        source_label=source_label,
        require_leveraged=require_leveraged,
    )


def _microsectors_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    item_pattern = re.compile(
        r'<div class="item">\s*<a[^>]*>\s*<div class="suite-name">(?P<suite>.*?)</div>'
        r'(?P<body>.*?)(?=<div class="item">|</div></div></div></div></div>|$)',
        re.I | re.S,
    )
    product_pattern = re.compile(
        r'<div class="product-symbol">(?P<symbol>[^<]+)</div>\s*'
        r'<div class="product-description">(?P<description>[^<]+)</div>',
        re.I | re.S,
    )
    for item in item_pattern.finditer(html):
        suite = _html_text(item.group("suite"))
        for product in product_pattern.finditer(item.group("body")):
            description = _html_text(product.group("description"))
            rows.append(
                {
                    "symbol": product.group("symbol"),
                    "name": f"MicroSectors {suite} {description} ETN",
                }
            )

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} ETN issuer table",
        require_leveraged=require_leveraged,
        product_type="ETN",
    )


def _normalize_hidden_url_symbol(symbol: str) -> str:
    if len(symbol) % 2 == 0:
        midpoint = len(symbol) // 2
        if symbol[:midpoint] == symbol[midpoint:]:
            return symbol[:midpoint]
    return symbol


def _etracs_leverage_table_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for table in _read_html_tables(html):
        table = _clean_columns(table)
        symbol_col = _first_matching_column(table.columns, [r"ticker\s+symbol", r"^ticker$", r"^symbol$"])
        name_col = _first_matching_column(table.columns, [r"^name$", r"\bname\b"])
        leverage_col = _first_matching_column(table.columns, [r"^leverage$"])
        if symbol_col is None or name_col is None:
            continue

        for _idx, row in table.iterrows():
            symbol_text = str(row[symbol_col])
            symbol_match = re.search(r"/ussymbol/([A-Z][A-Z0-9.-]*)", symbol_text)
            if symbol_match is None:
                symbol_match = re.search(r"\b([A-Z][A-Z0-9.-]{1,5})\b\s*$", symbol_text)
            if symbol_match is None:
                continue
            symbol = _normalize_hidden_url_symbol(symbol_match.group(1))

            leverage_text = str(row[leverage_col]) if leverage_col is not None else ""
            if (
                require_leveraged
                and leverage_col is not None
                and not re.search(r"\d+(?:\.\d+)?\s*x", leverage_text, re.I)
            ):
                continue

            rows.append({"symbol": symbol, "name": row[name_col]})

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} ETN issuer table",
        require_leveraged=require_leveraged,
        product_type="ETN",
    )


def _html_source_to_universe(
    html: str,
    source_name: str,
    *,
    source_label: str,
    require_leveraged: bool,
) -> pd.DataFrame:
    rows = []
    for table in _read_html_tables(html):
        source_rows = _fund_table_to_universe(
            _clean_columns(table),
            source_name,
            source_label=source_label,
            require_leveraged=require_leveraged,
        )
        if not source_rows.empty:
            rows.append(source_rows)

    card_rows = _html_cards_to_universe(
        html,
        source_name,
        source_label=source_label,
        require_leveraged=require_leveraged,
    )
    if not card_rows.empty:
        rows.append(card_rows)

    if not rows:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
    return pd.concat(rows, ignore_index=True).drop_duplicates("symbol").reset_index(drop=True)


def load_issuer_etf_universe(timeout: int = 30) -> pd.DataFrame:
    rows = []
    status_rows = []
    for issuer, url in ISSUER_UNIVERSE_SOURCES:
        try:
            resp = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
            resp.raise_for_status()
        except Exception as exc:
            status_rows.append(
                _workflow_source_status_row(
                    source=issuer,
                    source_type="issuer_etf",
                    url=url,
                    status="source_error",
                    error=f"{type(exc).__name__}: {exc}"[:250],
                )
            )
            continue
        parsed_rows = _html_source_to_universe(
            resp.text,
            issuer,
            source_label=f"{issuer} issuer table",
            require_leveraged=False,
        )
        issuer_rows = _html_source_to_universe(
            resp.text,
            issuer,
            source_label=f"{issuer} issuer table",
            require_leveraged=True,
        )
        status, error = _workflow_source_parse_status(parsed_rows, issuer_rows)
        if not issuer_rows.empty:
            rows.append(issuer_rows)
        status_rows.append(
            _workflow_source_status_row(
                source=issuer,
                source_type="issuer_etf",
                url=url,
                status=status,
                parsed_row_count=len(parsed_rows),
                row_count=len(issuer_rows),
                error=error,
            )
        )

    if not rows:
        out = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
    else:
        out = pd.concat(rows, ignore_index=True).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    out.attrs["workflow_source_status"] = status_rows
    return out


def load_etn_universe(timeout: int = 30) -> pd.DataFrame:
    rows = []
    status_rows = []
    for source in WORKFLOW_ETN_SOURCES:
        try:
            resp = requests.get(source.url, timeout=timeout, headers=REQUEST_HEADERS)
            resp.raise_for_status()
            if source.parser == "microsectors_html":
                parsed_rows = _microsectors_html_to_universe(
                    resp.text,
                    source,
                    require_leveraged=False,
                )
                source_rows = _microsectors_html_to_universe(resp.text, source)
            elif source.parser == "etracs_leverage_table":
                parsed_rows = _etracs_leverage_table_to_universe(
                    resp.text,
                    source,
                    require_leveraged=False,
                )
                source_rows = _etracs_leverage_table_to_universe(resp.text, source)
            else:
                parsed_rows = _html_source_to_universe(
                    resp.text,
                    source.name,
                    source_label=f"{source.name} ETN issuer table",
                    require_leveraged=False,
                )
                source_rows = _html_source_to_universe(
                    resp.text,
                    source.name,
                    source_label=f"{source.name} ETN issuer table",
                    require_leveraged=True,
                )
                source_rows["fund_type"] = source_rows["fund_type"].str.replace(
                    r"^ETF",
                    "ETN",
                    regex=True,
                )
        except Exception as exc:
            status_rows.append(
                _workflow_source_status_row(
                    source=source.name,
                    source_type="issuer_etn",
                    url=source.url,
                    status="source_error",
                    error=f"{type(exc).__name__}: {exc}"[:250],
                )
            )
            continue
        status, error = _workflow_source_parse_status(parsed_rows, source_rows)
        if not source_rows.empty:
            rows.append(source_rows)
        status_rows.append(
            _workflow_source_status_row(
                source=source.name,
                source_type="issuer_etn",
                url=source.url,
                status=status,
                parsed_row_count=len(parsed_rows),
                row_count=len(source_rows),
                error=error,
            )
        )

    if not rows:
        out = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
    else:
        out = pd.concat(rows, ignore_index=True).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    out.attrs["workflow_source_status"] = status_rows
    return out


def _cboe_symbol_csv_to_universe(csv_text: str, source: UniverseSource) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])

    symbol_col = _first_matching_column(df.columns, [r"^symbol$", r"^ticker$", r"^name$"])
    if symbol_col is None:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])

    rows = ({"symbol": symbol, "name": symbol} for symbol in df[symbol_col])
    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} audit source",
        require_leveraged=False,
    )


def _sec_company_tickers_to_universe(json_text: str, source: UniverseSource) -> pd.DataFrame:
    try:
        raw = pd.read_json(io.StringIO(json_text), orient="index")
    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
    if not {"ticker", "title"}.issubset(raw.columns):
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])

    rows = (
        {"symbol": row["ticker"], "name": row["title"]}
        for _idx, row in raw[["ticker", "title"]].iterrows()
    )
    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} audit source",
        require_leveraged=False,
        product_type="SEC",
    )


def _sec_fields_data_json_to_rows(
    json_text: str,
    *,
    symbol_field: str,
    name_field: str | None,
) -> list[dict[str, object]]:
    try:
        payload = json.loads(json_text)
    except Exception:
        return []

    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        return []
    if symbol_field not in fields:
        return []

    symbol_idx = fields.index(symbol_field)
    name_idx = fields.index(name_field) if name_field in fields else None

    rows: list[dict[str, object]] = []
    for raw_row in data:
        if not isinstance(raw_row, list) or symbol_idx >= len(raw_row):
            continue
        symbol = raw_row[symbol_idx]
        name = raw_row[name_idx] if name_idx is not None and name_idx < len(raw_row) else symbol
        rows.append({"symbol": symbol, "name": name})
    return rows


def _sec_exchange_tickers_to_universe(json_text: str, source: UniverseSource) -> pd.DataFrame:
    rows = _sec_fields_data_json_to_rows(
        json_text,
        symbol_field="ticker",
        name_field="name",
    )
    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} audit source",
        require_leveraged=False,
        product_type="SEC",
    )


def _sec_mutual_fund_tickers_to_universe(json_text: str, source: UniverseSource) -> pd.DataFrame:
    rows = _sec_fields_data_json_to_rows(
        json_text,
        symbol_field="symbol",
        name_field=None,
    )
    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} audit source",
        require_leveraged=False,
        product_type="SEC MF",
    )


def _audit_source_status_row(
    source: UniverseSource,
    *,
    status: str,
    row_count: int = 0,
    error: str = "",
) -> dict[str, object]:
    return {
        "source": source.name,
        "source_type": source.source_type,
        "url": source.url,
        "parser": source.parser,
        "enabled": source.enabled,
        "status": status,
        "row_count": row_count,
        "error": error,
        "notes": source.notes,
    }


def _audit_source_columns() -> list[str]:
    return [
        "symbol",
        "name",
        "fund_type",
        "source",
        "audit_source_type",
        "source_url",
        "is_leveraged_candidate",
        "is_long_leveraged_candidate",
        "leverage",
        "direction",
    ]


def _sec_audit_row_has_product_context(name: object) -> bool:
    normalized_name = re.sub(r"\s+", " ", str(name).upper()).strip()
    return any(re.search(pattern, normalized_name, re.I) for pattern in SEC_AUDIT_PRODUCT_CONTEXT_PATTERNS)


def _audit_row_leverage_metadata(name: object, source: UniverseSource) -> pd.Series:
    if source.parser in SEC_ENTITY_AUDIT_PARSERS and not _sec_audit_row_has_product_context(name):
        return pd.Series(
            {
                "is_leveraged_candidate": False,
                "is_long_leveraged_candidate": False,
                "leverage": None,
                "direction": None,
            }
        )

    leverage, direction = infer_leverage_and_direction(str(name))
    return pd.Series(
        {
            "is_leveraged_candidate": leveraged_name_filter(str(name)),
            "is_long_leveraged_candidate": is_long_leveraged_name(str(name)),
            "leverage": leverage,
            "direction": direction,
        }
    )


def _with_audit_metadata(rows: pd.DataFrame, source: UniverseSource) -> pd.DataFrame:
    out = rows.copy()
    out["audit_source_type"] = source.source_type
    out["source_url"] = source.url
    leverage_metadata = out["name"].apply(lambda name: _audit_row_leverage_metadata(name, source))
    out = pd.concat([out, leverage_metadata], axis=1)
    return out[_audit_source_columns()].reset_index(drop=True)


def load_audit_universe_sources(timeout: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    status_rows = []
    for source in AUDIT_UNIVERSE_SOURCES:
        if not source.enabled or source.parser == "registered_only":
            status_rows.append(_audit_source_status_row(source, status="registered_only"))
            continue

        try:
            resp = requests.get(source.url, timeout=timeout, headers=REQUEST_HEADERS)
            resp.raise_for_status()
            if source.parser == "cboe_symbol_csv":
                source_rows = _cboe_symbol_csv_to_universe(resp.text, source)
            elif source.parser == "sec_company_tickers":
                source_rows = _sec_company_tickers_to_universe(resp.text, source)
            elif source.parser == "sec_exchange_tickers":
                source_rows = _sec_exchange_tickers_to_universe(resp.text, source)
            elif source.parser == "sec_mutual_fund_tickers":
                source_rows = _sec_mutual_fund_tickers_to_universe(resp.text, source)
            else:
                source_rows = _html_source_to_universe(
                    resp.text,
                    source.name,
                    source_label=f"{source.name} audit source",
                    require_leveraged=False,
                )
        except Exception as exc:
            status_rows.append(
                _audit_source_status_row(
                    source,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}"[:250],
                )
            )
            continue

        if not source_rows.empty:
            source_rows = _with_audit_metadata(source_rows, source)
            rows.append(source_rows)
        status_rows.append(
            _audit_source_status_row(
                source,
                status="loaded" if not source_rows.empty else "loaded_no_rows",
                row_count=len(source_rows),
            )
        )

    if rows:
        audit_rows = (
            pd.concat(rows, ignore_index=True)
            .drop_duplicates(["symbol", "source"])
            .sort_values(["source", "symbol"])
            .reset_index(drop=True)
        )
    else:
        audit_rows = pd.DataFrame(columns=_audit_source_columns())

    status = pd.DataFrame(status_rows)
    return audit_rows, status


def build_universe_audit_report(
    audit_rows: pd.DataFrame,
    merged_universe: pd.DataFrame,
    workflow_assets: pd.DataFrame,
) -> pd.DataFrame:
    columns = _audit_source_columns() + [
        "in_merged_universe",
        "in_workflow_universe",
        "audit_reason",
    ]
    if audit_rows.empty:
        return pd.DataFrame(columns=columns)

    merged_symbols = set(merged_universe["symbol"].dropna().astype(str))
    workflow_symbols = set(workflow_assets["symbol"].dropna().astype(str))

    out = audit_rows.copy()
    out["in_merged_universe"] = out["symbol"].isin(merged_symbols)
    out["in_workflow_universe"] = out["symbol"].isin(workflow_symbols)
    out = out[out["is_leveraged_candidate"] & ~out["in_merged_universe"]].copy()
    out["audit_reason"] = "leveraged-looking audit source row missing from merged source universe"
    return out[columns].sort_values(["source", "symbol"]).reset_index(drop=True)


def _merge_universe_sources(nasdaq_df: pd.DataFrame, issuer_df: pd.DataFrame) -> pd.DataFrame:
    nasdaq = nasdaq_df.copy()
    nasdaq["source"] = "Nasdaq ETF definitions"
    combined = pd.concat([nasdaq, issuer_df], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    # Nasdaq is the current ETF authority for duplicate symbols.  Issuer rows
    # still contribute issuer-only ETFs/ETNs, but cannot overwrite current
    # official metadata for an ETF that Nasdaq lists.
    combined["source_rank"] = combined["source"].ne("Nasdaq ETF definitions").astype(int)
    combined = combined.sort_values(["symbol", "source_rank"], ascending=[True, True])
    combined = combined.drop_duplicates("symbol", keep="first")
    return combined.drop(columns=["source_rank"]).reset_index(drop=True)


def _workflow_row_metadata(row: pd.Series, known_symbols: set[str] | None) -> pd.Series:
    leverage, direction = infer_leverage_and_direction(row["name"])
    rsi_symbol = infer_rsi_symbol(row["symbol"], row["name"], known_symbols=known_symbols)
    if rsi_symbol == row["symbol"]:
        confidence = "fallback_to_self"
        mapping_source = "asset_symbol"
    elif known_symbols is None or rsi_symbol in known_symbols:
        confidence = "inferred"
        mapping_source = "name_inference"
    else:
        confidence = "needs_review"
        mapping_source = "name_inference"

    return pd.Series(
        {
            "rsi_symbol": rsi_symbol,
            "leverage": leverage,
            "direction": direction,
            "underlying_symbol": rsi_symbol,
            "underlying_name": rsi_symbol,
            "mapping_source": mapping_source,
            "confidence": confidence,
        }
    )


def select_universes(etf_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns current long leveraged single-stock ETFs and all current long leveraged ETFs.
    """
    eligible = etf_df[~etf_df["symbol"].isin(EXCLUDED_UNIVERSE_SYMBOLS)].copy()
    single_stock = eligible[
        eligible["fund_type"].str.contains(r"ETF \(Single Stock\)", regex=True, na=False)
    ].copy()
    single_stock_long = single_stock.loc[
        single_stock["name"].map(is_long_leveraged_name).astype(bool)
    ].copy()
    all_long_leveraged = eligible.loc[
        eligible["name"].map(is_long_leveraged_name).astype(bool)
    ].copy()

    return (
        single_stock_long.sort_values("symbol").reset_index(drop=True),
        all_long_leveraged.sort_values("symbol").reset_index(drop=True),
    )


def determine_workflow_assets(cfg: UniverseConfig) -> pd.DataFrame:
    if cfg.top_n is not None and (
        isinstance(cfg.top_n, bool) or not isinstance(cfg.top_n, int) or cfg.top_n <= 0
    ):
        raise ValueError(f"Universe top_n must be a positive integer or None; got {cfg.top_n!r}.")

    nasdaq_df = load_current_etf_universe(timeout=cfg.request_timeout_seconds)
    issuer_df = load_issuer_etf_universe(timeout=cfg.request_timeout_seconds)
    etn_df = load_etn_universe(timeout=cfg.request_timeout_seconds)
    workflow_source_status = pd.concat(
        [_workflow_source_status(issuer_df), _workflow_source_status(etn_df)],
        ignore_index=True,
    )
    workflow_source_failures = workflow_source_status[
        ~workflow_source_status["status"].isin({"loaded", "loaded_zero_matches"})
    ].copy()
    discovered_df = pd.concat([issuer_df, etn_df], ignore_index=True, sort=False)
    active_symbols = load_active_listed_symbols(timeout=cfg.request_timeout_seconds)
    active_listing_complete = bool(getattr(active_symbols, "is_complete", True))
    active_listing_status = pd.DataFrame(
        getattr(active_symbols, "source_status", []),
        columns=["source", "url", "symbol_column", "status", "symbol_count", "error"],
    )
    inactive_discovered = pd.DataFrame(columns=[*discovered_df.columns, "inactive_reason"])
    if active_symbols and active_listing_complete:
        active_symbols = {str(symbol).upper() for symbol in active_symbols}
        is_active = discovered_df["symbol"].astype(str).str.upper().isin(active_symbols)
        inactive_discovered = discovered_df.loc[~is_active].copy()
        inactive_discovered["inactive_reason"] = "not present in active Nasdaq symbol files"
        discovered_df = discovered_df.loc[is_active].copy()
    etf_df = _merge_universe_sources(nasdaq_df, discovered_df)
    known_symbols: set[str] | None
    if active_symbols and active_listing_complete:
        known_symbols = set(etf_df["symbol"].dropna().astype(str).str.upper())
        known_symbols.update(active_symbols)
    else:
        known_symbols = None

    single_stock_long, all_long_leveraged = select_universes(etf_df)
    nasdaq_universe = build_nasdaq_universe_table(etf_df)
    save_table_to_sqlite(nasdaq_universe, cfg.sqlite_db_path, "nasdaq_etf_universe")
    save_table_to_sqlite(
        inactive_discovered,
        cfg.sqlite_db_path,
        "universe_inactive_discovered_products",
    )
    save_table_to_sqlite(
        active_listing_status,
        cfg.sqlite_db_path,
        "universe_active_listing_source_status",
    )
    save_table_to_sqlite(
        workflow_source_status,
        cfg.sqlite_db_path,
        "universe_workflow_source_status",
    )

    if cfg.require_workflow_source_success and not workflow_source_failures.empty:
        failed_sources = ", ".join(workflow_source_failures["source"].astype(str).tolist())
        raise RuntimeError(f"Workflow universe sources were unusable: {failed_sources}.")

    if all_long_leveraged.empty:
        raise RuntimeError("Nasdaq ETF universe returned no current long leveraged ETFs.")

    workflow_assets = (
        all_long_leveraged.copy()
        if cfg.top_n is None
        else all_long_leveraged.head(cfg.top_n).copy()
    )
    audit_rows, audit_status = load_audit_universe_sources(timeout=cfg.request_timeout_seconds)
    audit_report = build_universe_audit_report(audit_rows, etf_df, all_long_leveraged)
    save_table_to_sqlite(audit_rows, cfg.sqlite_db_path, "universe_audit_rows")
    save_table_to_sqlite(audit_report, cfg.sqlite_db_path, "universe_audit_missing_candidates")
    save_table_to_sqlite(audit_status, cfg.sqlite_db_path, "universe_audit_source_status")

    metadata = workflow_assets.apply(lambda row: _workflow_row_metadata(row, known_symbols), axis=1)
    workflow_assets = pd.concat([workflow_assets.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)

    if cfg.top_n is None:
        universe_title = "All Long Leveraged ETFs/ETNs From Merged Universe"
    else:
        universe_title = f"First {cfg.top_n} Long Leveraged ETFs/ETNs From Merged Universe"

    out = workflow_assets[["symbol", "name", "rsi_symbol"]].reset_index(drop=True)
    for column in [
        "leverage",
        "direction",
        "underlying_symbol",
        "underlying_name",
        "fund_type",
        "source",
        "mapping_source",
        "confidence",
    ]:
        if column in workflow_assets.columns:
            out[column] = workflow_assets[column].to_numpy()
    out.attrs["universe_title"] = universe_title
    out.attrs["universe_counts"] = {
        "Current ETFs in Nasdaq table": len(nasdaq_df),
        "Current issuer-discovered leveraged ETFs found": len(issuer_df),
        "Current issuer-discovered leveraged ETNs found": len(etn_df),
        "Inactive issuer-discovered ETFs/ETNs skipped": len(inactive_discovered),
        "Active listing sources loaded": int(
            (active_listing_status["status"] == "loaded").sum()
        ) if not active_listing_status.empty else 0,
        "Active listing snapshot complete": active_listing_complete,
        "Workflow universe sources failed": len(workflow_source_failures),
        "Merged current ETFs/ETNs": len(etf_df),
        "Current long single-stock leveraged ETFs found": len(single_stock_long),
        "Current long leveraged ETFs/ETNs found": len(all_long_leveraged),
        "Audit sources registered": len(AUDIT_UNIVERSE_SOURCES),
        "Audit rows parsed": len(audit_rows),
        "Audit leveraged candidates missing from merged universe": len(audit_report),
    }
    out.attrs["universe_degraded"] = not workflow_source_failures.empty
    out.attrs["universe_db_path"] = cfg.sqlite_db_path
    return out


def build_nasdaq_universe_table(etf_df: pd.DataFrame) -> pd.DataFrame:
    out = etf_df.copy()
    out["is_long_leveraged"] = out["name"].apply(is_long_leveraged_name)
    out["is_single_stock"] = out["fund_type"].str.contains(
        r"ETF \(Single Stock\)",
        regex=True,
        na=False,
    )
    out["is_single_stock_long_leveraged"] = out["is_single_stock"] & out["is_long_leveraged"]
    out["rsi_symbol"] = out.apply(_infer_nasdaq_table_rsi_symbol, axis=1)
    return out.sort_values("symbol").reset_index(drop=True)


def _infer_nasdaq_table_rsi_symbol(row: pd.Series) -> str:
    if row["is_long_leveraged"]:
        return infer_rsi_symbol(row["symbol"], row["name"])
    return row["symbol"]
