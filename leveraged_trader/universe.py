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


@dataclass(frozen=True)
class RsiSymbolMapping:
    rsi_symbol: str
    underlying_name: str
    mapping_source: str
    confidence: str
    mapping_reason: str


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
NON_FAILURE_WORKFLOW_SOURCE_STATUSES = {"loaded", "loaded_zero_matches", "registered_only"}
HEALTHY_WORKFLOW_SOURCE_STATUSES = NON_FAILURE_WORKFLOW_SOURCE_STATUSES


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
    UniverseSource("ProShares", "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs", "issuer_etf"),
    UniverseSource(
        "Direxion",
        "https://www.direxion.com/all-etfs",
        "issuer_etf",
        enabled=False,
        notes="Issuer page blocks unattended fetches; Nasdaq ETF definitions remain authoritative.",
    ),
    UniverseSource("Leverage Shares", "https://leverageshares.com/us/", "issuer_etf", parser="js_ticker_name"),
    UniverseSource("GraniteShares", "https://graniteshares.com/etfs/", "issuer_etf", parser="graniteshares_html"),
    UniverseSource(
        "Defiance",
        "https://www.defianceetfs.com/wp-json/defiance/v1/etfs-explore",
        "issuer_etf",
        parser="defiance_json",
    ),
    UniverseSource("AdvisorShares", "https://advisorshares.com/etfs/", "issuer_etf"),
    UniverseSource("AXS Investments", "https://www.axsinvestments.com/our-funds/", "issuer_etf"),
    UniverseSource("Kurv", "https://www.kurvinvest.com/etfs", "issuer_etf"),
    UniverseSource(
        "Innovator",
        "https://www.innovatoretfs.com/etf/finder/",
        "issuer_etf",
        enabled=False,
        notes="Dynamic product finder does not expose stable static ticker/name rows.",
    ),
    UniverseSource("Innovator", "https://www.innovatoretfs.com/define/etfs/", "issuer_etf"),
    UniverseSource("Tuttle Capital", "https://www.tuttlecap.com/etfs", "issuer_etf"),
    UniverseSource("Tradr", "https://www.tradretfs.com/", "issuer_etf"),
    UniverseSource(
        "REX Shares",
        "https://www.rexshares.com/t-rex-leveraged-etfs/",
        "issuer_etf",
        parser="rex_menu_html",
    ),
    UniverseSource(
        "KraneShares",
        "https://kraneshares.com/levered-etf-suite/",
        "issuer_etf",
        enabled=False,
        notes="Current issuer page does not expose stable static ticker/name rows.",
    ),
    UniverseSource(
        "Volatility Shares",
        "https://www.volatilityshares.com/",
        "issuer_etf",
        parser="volatilityshares_html",
    ),
    UniverseSource(
        "21Shares",
        "https://www.21shares.com/en-us",
        "issuer_etf",
        enabled=False,
        notes="No stable U.S. leveraged ETF product list is exposed for workflow discovery.",
    ),
    UniverseSource("YieldMax", "https://www.yieldmaxetfs.com/our-etfs", "issuer_etf"),
    UniverseSource(
        "Tidal",
        "https://www.tidalfinancialgroup.com/",
        "issuer_etf",
        enabled=False,
        notes="Tidal is a platform/provider page, not a stable issuer ETF product list.",
    ),
    UniverseSource("Roundhill", "https://www.roundhillinvestments.com/etf/", "issuer_etf"),
    UniverseSource("Themes", "https://themesetfs.com/etfs", "issuer_etf", parser="js_ticker_name"),
    UniverseSource("Simplify", "https://www.simplify.us/etfs", "issuer_etf"),
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
    "T-REX",
    "TREX",
    "NASDAQ",
    "MSCI",
    "PAY",
    "B",
    "HIGH",
    "REAL",
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
    # False positives where "ultra-short" describes bond duration rather than
    # inverse market exposure.
    "AMUN",
    "RBIL",
    "SGVA",
    "UYLD",
    "VGUS",
    "ZMUN",
    # SLTY shorts a frequently changing basket of 15-30 securities, so there
    # is no stable single underlying symbol from which to derive its RSI.
    "SLTY",
}
YAHOO_SYMBOL_ALIASES = {
    "BRKB": "BRK-B",
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
}

RSI_SYMBOL_OVERRIDES = {
    "AIQD": ("AIQ", "Global X Artificial Intelligence & Technology ETF proxy"),
    "AAPX": ("AAPL", "Apple Inc."),
    "ETNG": ("ETN", "Eaton Corp. plc"),
    "GOOX": ("GOOG", "Alphabet Inc."),
    "MSFX": ("MSFT", "Microsoft Corp."),
    "NVDQ": ("NVDA", "NVIDIA Corp."),
    "NVDX": ("NVDA", "NVIDIA Corp."),
    "TSLZ": ("TSLA", "Tesla Inc."),
    "TSLT": ("TSLA", "Tesla Inc."),
    "BULG": ("BULL", "Webull Corp."),
    "BULX": ("BULL", "Webull Corp."),
    "BERZ": ("FNGS", "MicroSectors FANG+ ETN proxy"),
    "BIS": ("IBB", "iShares Biotechnology ETF proxy"),
    "BNKD": ("KBWB", "Invesco KBW Bank ETF proxy"),
    "BZQ": ("EWZ", "iShares MSCI Brazil ETF proxy"),
    "DUG": ("XLE", "Energy Select Sector SPDR Fund proxy"),
    "EEV": ("EEM", "iShares MSCI Emerging Markets ETF proxy"),
    "EFU": ("EFA", "iShares MSCI EAFE ETF proxy"),
    "EPV": ("VGK", "Vanguard FTSE Europe ETF proxy"),
    "EUO": ("FXE", "Invesco CurrencyShares Euro Trust proxy"),
    "EWV": ("EWJ", "iShares MSCI Japan ETF proxy"),
    "FLYD": ("PEJ", "Invesco Leisure and Entertainment ETF proxy"),
    "FNGD": ("FNGS", "MicroSectors FANG+ ETN proxy"),
    "FXP": ("FXI", "iShares China Large-Cap ETF proxy"),
    "KOLD": ("UNG", "United States Natural Gas Fund proxy"),
    "MZZ": ("MDY", "SPDR S&P MidCap 400 ETF Trust proxy"),
    "NRGD": ("XLE", "Energy Select Sector SPDR Fund proxy"),
    "OILD": ("XOP", "SPDR S&P Oil & Gas Exploration & Production ETF proxy"),
    "QID": ("QQQ", "Invesco QQQ Trust proxy"),
    "QQDN": ("QQQ", "Invesco QQQ Trust proxy"),
    "REW": ("XLK", "Technology Select Sector SPDR Fund proxy"),
    "RXD": ("XLV", "Health Care Select Sector SPDR Fund proxy"),
    "SCC": ("XLY", "Consumer Discretionary Select Sector SPDR Fund proxy"),
    "SCO": ("USO", "United States Oil Fund proxy"),
    "SDD": ("IJR", "iShares Core S&P Small-Cap ETF proxy"),
    "SDP": ("XLU", "Utilities Select Sector SPDR Fund proxy"),
    "SIJ": ("XLI", "Industrial Select Sector SPDR Fund proxy"),
    "SKF": ("XLF", "Financial Select Sector SPDR Fund proxy"),
    "SKRE": ("KRE", "SPDR S&P Regional Banking ETF proxy"),
    "SMDD": ("MDY", "SPDR S&P MidCap 400 ETF Trust proxy"),
    "SMN": ("XLB", "Materials Select Sector SPDR Fund proxy"),
    "SRS": ("IYR", "iShares U.S. Real Estate ETF proxy"),
    "SSG": ("SOXX", "iShares Semiconductor ETF proxy"),
    "SZK": ("XLP", "Consumer Staples Select Sector SPDR Fund proxy"),
    "WTID": ("XLE", "Energy Select Sector SPDR Fund proxy"),
    "YCS": ("FXY", "Invesco CurrencyShares Japanese Yen Trust proxy"),
    "MST": ("MSTR", "MicroStrategy Inc."),
    "MSOX": ("MSOS", "AdvisorShares Pure US Cannabis ETF proxy"),
    "SATG": ("SATS", "EchoStar Corp."),
    "MQQQ": ("QQQ", "Invesco QQQ Trust proxy"),
    "QQQP": ("QQQ", "Invesco QQQ Trust proxy"),
    "WLDU": ("VT", "Vanguard Total World Stock ETF proxy"),
    "AIQU": ("AIQ", "Global X Artificial Intelligence & Technology ETF proxy"),
    "BDCX": ("BIZD", "VanEck BDC Income ETF proxy"),
    "BIB": ("IBB", "iShares Biotechnology ETF proxy"),
    "BNKU": ("KBWB", "Invesco KBW Bank ETF proxy"),
    "BULZ": ("FNGS", "MicroSectors FANG+ ETN proxy"),
    "CEFD": ("CEFS", "Saba Closed-End Funds ETF proxy"),
    "DIG": ("XLE", "Energy Select Sector SPDR Fund proxy"),
    "DRNL": ("DRNZ", "REX Drone ETF proxy"),
    "EET": ("EEM", "iShares MSCI Emerging Markets ETF proxy"),
    "EFO": ("EFA", "iShares MSCI EAFE ETF proxy"),
    "EZJ": ("EWJ", "iShares MSCI Japan ETF proxy"),
    "FDRX": ("FDRS", "Founder-Led ETF proxy"),
    "FLYU": ("PEJ", "Invesco Leisure and Entertainment ETF proxy"),
    "FNGO": ("FNGS", "MicroSectors FANG+ ETN proxy"),
    "FNGU": ("FNGS", "MicroSectors FANG+ ETN proxy"),
    "HDLB": ("SPHD", "Invesco S&P 500 High Dividend Low Volatility ETF proxy"),
    "IWDL": ("IWD", "iShares Russell 1000 Value ETF proxy"),
    "IWFL": ("IWF", "iShares Russell 1000 Growth ETF proxy"),
    "IWML": ("SIZE", "iShares MSCI USA Size Factor ETF proxy"),
    "LTL": ("XLC", "Communication Services Select Sector SPDR Fund proxy"),
    "MAGX": ("MAGS", "Roundhill Magnificent Seven ETF proxy"),
    "MLPR": ("AMLP", "Alerian MLP ETF proxy"),
    "MTUL": ("MTUM", "iShares MSCI USA Momentum Factor ETF proxy"),
    "MVRL": ("REM", "iShares Mortgage Real Estate ETF proxy"),
    "MVV": ("MDY", "SPDR S&P MidCap 400 ETF Trust proxy"),
    "NRGU": ("XLE", "Energy Select Sector SPDR Fund proxy"),
    "OILU": ("XOP", "SPDR S&P Oil & Gas Exploration & Production ETF proxy"),
    "PFFL": ("PFF", "iShares Preferred and Income Securities ETF proxy"),
    "QPUX": ("QTUM", "Defiance Quantum ETF proxy"),
    "QULL": ("QUAL", "iShares MSCI USA Quality Factor ETF proxy"),
    "ROM": ("XLK", "Technology Select Sector SPDR Fund proxy"),
    "RXL": ("XLV", "Health Care Select Sector SPDR Fund proxy"),
    "SAA": ("IJR", "iShares Core S&P Small-Cap ETF proxy"),
    "SCDL": ("SCHD", "Schwab US Dividend Equity ETF proxy"),
    "SKYU": ("SKYY", "First Trust Cloud Computing ETF proxy"),
    "SMHB": ("DES", "WisdomTree U.S. SmallCap Dividend Fund proxy"),
    "SPCL": ("UFO", "Procure Space ETF proxy"),
    "TARK": ("ARKK", "ARK Innovation ETF proxy"),
    "UCC": ("XLY", "Consumer Discretionary Select Sector SPDR Fund proxy"),
    "UCYB": ("CIBR", "First Trust Nasdaq Cybersecurity ETF proxy"),
    "UBR": ("EWZ", "iShares MSCI Brazil ETF proxy"),
    "UGE": ("XLP", "Consumer Staples Select Sector SPDR Fund proxy"),
    "UJB": ("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF proxy"),
    "UMDD": ("MDY", "SPDR S&P MidCap 400 ETF Trust proxy"),
    "UPV": ("VGK", "Vanguard FTSE Europe ETF proxy"),
    "UPW": ("XLU", "Utilities Select Sector SPDR Fund proxy"),
    "URE": ("IYR", "iShares U.S. Real Estate ETF proxy"),
    "USD": ("SOXX", "iShares Semiconductor ETF proxy"),
    "USML": ("USMV", "iShares MSCI USA Min Vol Factor ETF proxy"),
    "UVIX": ("VIXY", "ProShares VIX Short-Term Futures ETF proxy"),
    "UXI": ("XLI", "Industrial Select Sector SPDR Fund proxy"),
    "UXRP": ("XRP-USD", "XRP spot price proxy"),
    "UYG": ("XLF", "Financial Select Sector SPDR Fund proxy"),
    "UYM": ("XLB", "Materials Select Sector SPDR Fund proxy"),
    "XPP": ("FXI", "iShares China Large-Cap ETF proxy"),
    "XRPT": ("XRP-USD", "XRP spot price proxy"),
    "BOIL": ("UNG", "United States Natural Gas Fund proxy"),
    "COPZ": ("COPX", "Global X Copper Miners ETF proxy"),
    "UCO": ("USO", "United States Oil Fund proxy"),
    "UCOP": ("CPER", "United States Copper Index Fund proxy"),
    "ULE": ("FXE", "Invesco CurrencyShares Euro Trust proxy"),
    "UPAL": ("PALL", "abrdn Physical Palladium Shares ETF proxy"),
    "UPLT": ("PPLT", "abrdn Physical Platinum Shares ETF proxy"),
    "WTIU": ("XLE", "Energy Select Sector SPDR Fund proxy"),
    "YCL": ("FXY", "Invesco CurrencyShares Japanese Yen Trust proxy"),
    "BITX": ("BTC-USD", "Bitcoin spot price proxy"),
    "BITU": ("BTC-USD", "Bitcoin spot price proxy"),
    "BTCL": ("BTC-USD", "Bitcoin spot price proxy"),
    "AVAZ": ("AVAX-USD", "Avalanche spot price proxy"),
    "CHNU": ("LINK-USD", "Chainlink spot price proxy"),
    "CRDX": ("ADA-USD", "Cardano spot price proxy"),
    "ETHU": ("ETH-USD", "Ether spot price proxy"),
    "ETHT": ("ETH-USD", "Ether spot price proxy"),
    "ETU": ("ETH-USD", "Ether spot price proxy"),
    "SLON": ("SOL-USD", "Solana spot price proxy"),
    "SOLT": ("SOL-USD", "Solana spot price proxy"),
    "STLU": ("XLM-USD", "Stellar spot price proxy"),
    "SUIL": ("SUI20947-USD", "Sui spot price proxy"),
    "TXXD": ("DOGE-USD", "Dogecoin spot price proxy"),
    "TXXH": ("HYPE32196-USD", "Hyperliquid spot price proxy"),
}

RSI_SELF_FALLBACK_SYMBOL_OVERRIDES = {
    "BEGS": ("BEGS", "Rareview 2X Bull Cryptocurrency & Precious Metals ETF self-RSI fallback"),
}

RSI_SYMBOLS_REQUIRING_REVIEW: dict[str, str] = {
    # SK is SK Telecom's U.S. ticker, not an investable SK hynix underlying.
    # Until a stable market-data proxy is selected, using it would drive both
    # long and inverse SK hynix products from an unrelated company's RSI.
    symbol: "SK hynix has no validated U.S. underlying ticker or RSI proxy; SK is SK Telecom"
    for symbol in ("SKHU", "SKHX", "SKUU", "SKDD")
}

RSI_NAME_PATTERNS_REQUIRING_REVIEW = [
    (
        r"\bSK\s+HYNIX\b",
        "SK hynix has no validated U.S. underlying ticker or RSI proxy; SK is SK Telecom",
    ),
]

RSI_NAME_PROXY_PATTERNS = [
    (r"\bSPACE\s*X\b|\bSPACEX\b", "SPCX", "Space Exploration Technologies Corp. Class A"),
    (r"\bS\s*&\s*P\s*500\s+EQUAL\s+WEIGHT\b", "RSP", "Invesco S&P 500 Equal Weight ETF proxy"),
    (r"\bS\s*&\s*P\s*500\b|\bS&P500\b", "SPY", "SPDR S&P 500 ETF Trust proxy"),
    (r"\bNASDAQ[-\s]*100\b", "QQQ", "Invesco QQQ Trust proxy"),
    (r"\bDOW\s*30\b", "DIA", "SPDR Dow Jones Industrial Average ETF Trust proxy"),
    (r"\bRUSSELL\s*2000\b", "IWM", "iShares Russell 2000 ETF proxy"),
    (r"\b20\+\s*YEAR\s+TREASURY\b", "TLT", "iShares 20+ Year Treasury Bond ETF proxy"),
    (r"\b7-10\s*YEAR\s+TREASURY\b", "IEF", "iShares 7-10 Year Treasury Bond ETF proxy"),
    (r"\bGOLD\s+MINERS\b", "GDX", "VanEck Gold Miners ETF proxy"),
    (r"\bGOLD\b(?!\s+MINERS)", "GLD", "SPDR Gold Shares proxy"),
    (r"\bSILVER\b", "SLV", "iShares Silver Trust proxy"),
    (r"\bBITCOIN\b", "BTC-USD", "Bitcoin spot price proxy"),
    (r"\bDOGECOIN\b", "DOGE-USD", "Dogecoin spot price proxy"),
    (r"\bAVALANCHE\b", "AVAX-USD", "Avalanche spot price proxy"),
    (r"\bCHAINLINK\b", "LINK-USD", "Chainlink spot price proxy"),
    (r"\bCARDANO\b", "ADA-USD", "Cardano spot price proxy"),
    (r"\bETHER(?:EUM)?\b", "ETH-USD", "Ether spot price proxy"),
    (r"\bSOLANA\b", "SOL-USD", "Solana spot price proxy"),
    (r"\bXRP\b", "XRP-USD", "XRP spot price proxy"),
    (r"\bSUI\b", "SUI20947-USD", "Sui spot price proxy"),
    (r"\bSTELLAR\b", "XLM-USD", "Stellar spot price proxy"),
    (r"\bHYPE\b|\bHYPERLIQUID\b", "HYPE32196-USD", "Hyperliquid spot price proxy"),
]

NUMERIC_X_LEVERAGE_PATTERN = r"(?<![A-Z0-9])([+-]?\d+(?:\.\d+)?)\s*X(?![A-Z0-9])"
NUMERIC_PERCENT_LEVERAGE_PATTERN = r"(?<![A-Z0-9])([+-]?\d+(?:\.\d+)?)\s*%(?![A-Z0-9])"
RECOGNIZED_X_LEVERAGE_TOKEN = r"(?:1\.(?:0*[1-9]\d*)|[2-4](?:\.\d+)?|5(?:\.0+)?)\s*X"
RECOGNIZED_PERCENT_LEVERAGE_TOKEN = (
    r"(?:100\.(?:0*[1-9]\d*)|1(?:0[1-9]|[1-9]\d)(?:\.\d+)?|[2-4]\d{2}(?:\.\d+)?|500(?:\.0+)?)\s*%"
)
RECOGNIZED_X_LEVERAGE_PATTERN = rf"(?<![A-Z0-9]){RECOGNIZED_X_LEVERAGE_TOKEN}(?![A-Z0-9])"
RECOGNIZED_PERCENT_LEVERAGE_PATTERN = rf"(?<![A-Z0-9]){RECOGNIZED_PERCENT_LEVERAGE_TOKEN}(?![A-Z0-9])"
LEVERAGE_TOKEN_PATTERN = rf"(?:{RECOGNIZED_X_LEVERAGE_TOKEN}|{RECOGNIZED_PERCENT_LEVERAGE_TOKEN})"

RSI_SYMBOL_PATTERNS = [
    rf"\b{LEVERAGE_TOKEN_PATTERN}\s+(?:DAILY\s+)?(?:TARGET\s+)?(?:LONG|BULL)\s+([A-Z][A-Z0-9.-]{{0,5}})\b",
    rf"\b{LEVERAGE_TOKEN_PATTERN}\s+(?:DAILY\s+)?(?:TARGET\s+)?(?:SHORT|BEAR|INVERSE)\s+([A-Z][A-Z0-9.-]{{0,5}})\b",
    r"\b(?:LONG|BULL)\s+([A-Z][A-Z0-9.-]{0,5})\s+(?:DAILY\s+)?(?:ETF|ETN|SHARES?)\b",
    r"\b(?:SHORT|BEAR|INVERSE)\s+([A-Z][A-Z0-9.-]{0,5})\s+(?:DAILY\s+)?(?:ETF|ETN|SHARES?)\b",
    r"\b([A-Z][A-Z0-9.-]{0,5})\s+(?:DAILY\s+)?(?:LONG|BULL)\b",
    r"\b([A-Z][A-Z0-9.-]{0,5})\s+(?:DAILY\s+)?(?:SHORT|BEAR|INVERSE)\b",
    rf"\b([A-Z][A-Z0-9.-]{{0,5}})\s+{LEVERAGE_TOKEN_PATTERN}(?![A-Z0-9])",
    r"\b(?:ULTRAPRO|ULTRA)\s+(?:SHORT|BEAR|INVERSE)\s+([A-Z][A-Z0-9.-]{0,5})\b",
    r"\b(?:ULTRAPRO|ULTRA)\s+([A-Z][A-Z0-9.-]{0,5})\b",
]

SAFE_GENERIC_RSI_SYMBOLS = {
    "BRK-A",
    "BRK-B",
    "DIA",
    "GDX",
    "GLD",
    "IEF",
    "IWM",
    "QQQ",
    "RSP",
    "SLV",
    "SPY",
    "TLT",
    "VT",
}


LEVERAGE_NAME_PATTERNS = [
    RECOGNIZED_X_LEVERAGE_PATTERN,
    RECOGNIZED_PERCENT_LEVERAGE_PATTERN,
    r"\bultrapro\b",
    r"\bultra\b",
    r"\bbull\s+[2-5](?:\.\d+)?\s*x\b",
    r"\b(?:daily\s+)?target\s+[2-5](?:\.\d+)?\s*x\b",
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
    r"\bshort\b(?![-\s]+term\b)",
    r"\binverse\b",
    r"\bultrashort\b",
    r"(?<![A-Z0-9])-\d+(?:\.\d+)?\s*x(?![A-Z0-9])",
    r"(?<![A-Z0-9])-\d+(?:\.\d+)?\s*%(?![A-Z0-9])",
]

LEVERAGE_FALSE_POSITIVE_TERMS = [
    "ULTRA SHORT TERM",
    "ULTRA SHORT-TERM",
    "ULTRA-SHORT TERM",
    "ULTRA-SHORT-TERM",
    "ULTRASHORT TERM",
    "ULTRASHORT-TERM",
    "ULTRA SHORT INCOME",
    "ULTRA-SHORT INCOME",
    "ULTRA OPTION INCOME",
    "ULTRA BUFFER",
    "ULTRA-BUFFER",
    "SHORT DURATION",
    "LONG TERM",
    "LONG-TERM",
    "LONG MUNICIPAL",
]

LEVERAGE_DIRECTION_NAME_OVERRIDES = [
    (r"\bMSOS\s+DAILY\s+LEVERAGED\s+ETF\b", 2.0, "long"),
    (r"\bDEFIANCE\s+LEVERAGED\s+LONG\s+\+\s+INCOME\s+MSTR\s+ETF\b", 1.75, "long"),
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


def _normalized_fund_name(fund_name: object) -> str:
    return re.sub(r"\s+", " ", str(fund_name).upper()).strip()


def _mapping_from_curated_symbol(asset_symbol: str) -> RsiSymbolMapping | None:
    override = RSI_SYMBOL_OVERRIDES.get(asset_symbol)
    if override is None:
        return None
    rsi_symbol, underlying_name = override
    return RsiSymbolMapping(
        rsi_symbol=rsi_symbol,
        underlying_name=underlying_name,
        mapping_source="symbol_override",
        confidence="curated",
        mapping_reason="matched exact leveraged product symbol override",
    )


def _mapping_from_self_fallback_symbol(asset_symbol: str) -> RsiSymbolMapping | None:
    override = RSI_SELF_FALLBACK_SYMBOL_OVERRIDES.get(asset_symbol)
    if override is None:
        return None
    rsi_symbol, underlying_name = override
    return RsiSymbolMapping(
        rsi_symbol=rsi_symbol,
        underlying_name=underlying_name,
        mapping_source="self_fallback_override",
        confidence="fallback_to_self",
        mapping_reason="matched exact leveraged product symbol self-RSI fallback override",
    )


def _mapping_from_curated_name(normalized_name: str) -> RsiSymbolMapping | None:
    for pattern, rsi_symbol, underlying_name in RSI_NAME_PROXY_PATTERNS:
        if re.search(pattern, normalized_name):
            return RsiSymbolMapping(
                rsi_symbol=rsi_symbol,
                underlying_name=underlying_name,
                mapping_source="name_proxy",
                confidence="curated",
                mapping_reason=f"matched curated name proxy pattern {pattern}",
            )
    return None


def _mapping_from_review_symbol(asset_symbol: str) -> RsiSymbolMapping | None:
    review_reason = RSI_SYMBOLS_REQUIRING_REVIEW.get(asset_symbol)
    if review_reason is None:
        return None
    return RsiSymbolMapping(
        rsi_symbol=asset_symbol,
        underlying_name=asset_symbol,
        mapping_source="unresolved_basket",
        confidence="needs_review",
        mapping_reason=review_reason,
    )


def _mapping_from_review_name(asset_symbol: str, normalized_name: str) -> RsiSymbolMapping | None:
    for pattern, review_reason in RSI_NAME_PATTERNS_REQUIRING_REVIEW:
        if re.search(pattern, normalized_name):
            return RsiSymbolMapping(
                rsi_symbol=asset_symbol,
                underlying_name=asset_symbol,
                mapping_source="unresolved_single_stock",
                confidence="needs_review",
                mapping_reason=review_reason,
            )
    return None


def _looks_like_single_stock_product(fund_name: object, fund_type: object | None = None) -> bool:
    normalized_name = _normalized_fund_name(fund_name)
    normalized_type = _normalized_fund_name(fund_type or "")
    if "SINGLE STOCK" in normalized_type:
        return True
    if not leveraged_name_filter(normalized_name):
        return False
    direction_pattern = r"(?:LONG|BULL|SHORT|BEAR|INVERSE)"
    single_stock_patterns = [
        rf"\bT-?REX\b.*\b{LEVERAGE_TOKEN_PATTERN}\s+{direction_pattern}\b.*\bDAILY\s+TARGET\s+ETF\b",
        rf"\b{LEVERAGE_TOKEN_PATTERN}\s+{direction_pattern}\s+.+?\s+DAILY\s+(?:TARGET\s+)?(?:ETF|ETN)\b",
        rf"\b[A-Z][A-Z0-9.-]{{0,5}}\s+{direction_pattern}\s+{LEVERAGE_TOKEN_PATTERN}\s+(?:DAILY\s+)?(?:ETF|ETN|SHARES?)\b",
        rf"\b.+?\s+{LEVERAGE_TOKEN_PATTERN}\s+(?:DAILY\s+)?(?:TARGET\s+)?{direction_pattern}\s+(?:DAILY\s+)?(?:ETF|ETN|SHARES?)\b",
    ]
    return any(re.search(pattern, normalized_name) for pattern in single_stock_patterns)


def infer_rsi_mapping(
    asset_symbol: str,
    fund_name: str,
    known_symbols: set[str] | None = None,
    fund_type: object | None = None,
) -> RsiSymbolMapping:
    """
    Infer the unleveraged signal ticker and record how confident the mapping is.

    Curated symbol/name proxies are applied before generic ticker extraction. Generic
    extraction is constrained to known symbols, or to a small safe default proxy set
    for direct helper calls that do not provide a symbol universe.
    """
    asset_symbol = asset_symbol.upper()
    normalized_name = _normalized_fund_name(fund_name)

    for curated_mapping in [
        _mapping_from_curated_symbol(asset_symbol),
        _mapping_from_self_fallback_symbol(asset_symbol),
        _mapping_from_curated_name(normalized_name),
        _mapping_from_review_symbol(asset_symbol),
        _mapping_from_review_name(asset_symbol, normalized_name),
    ]:
        if curated_mapping is not None:
            return curated_mapping

    generic_known_symbols = known_symbols if known_symbols is not None else SAFE_GENERIC_RSI_SYMBOLS
    for pattern in RSI_SYMBOL_PATTERNS:
        match = re.search(pattern, normalized_name)
        if not match:
            continue
        candidate = _normalize_symbol_candidate(match.group(1), known_symbols=generic_known_symbols)
        if candidate is not None and candidate != asset_symbol:
            return RsiSymbolMapping(
                rsi_symbol=candidate,
                underlying_name=candidate,
                mapping_source="name_inference",
                confidence="inferred",
                mapping_reason=f"matched ticker inference pattern {pattern}",
            )

    if _looks_like_single_stock_product(fund_name, fund_type):
        return RsiSymbolMapping(
            rsi_symbol=asset_symbol,
            underlying_name=asset_symbol,
            mapping_source="unresolved_single_stock",
            confidence="needs_review",
            mapping_reason="single-stock-style product did not expose a reliable underlying ticker",
        )

    return RsiSymbolMapping(
        rsi_symbol=asset_symbol,
        underlying_name=asset_symbol,
        mapping_source="asset_symbol",
        confidence="fallback_to_self",
        mapping_reason="no reliable underlying proxy was found",
    )


def infer_rsi_symbol(asset_symbol: str, fund_name: str, known_symbols: set[str] | None = None) -> str:
    """
    Infer the unleveraged signal ticker from a leveraged ETF name.

    Most single-stock leveraged ETF names include the underlying ticker near
    phrases like "2x Long", "Bull", or "UltraPro". If no reliable ticker is
    present, fall back to the leveraged ETF itself.
    """
    return infer_rsi_mapping(asset_symbol, fund_name, known_symbols=known_symbols).rsi_symbol


def _rsi_mapping_review_table(workflow_assets: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "workflow",
        "symbol",
        "name",
        "rsi_symbol",
        "fund_type",
        "source",
        "mapping_source",
        "confidence",
        "mapping_reason",
    ]
    if workflow_assets.empty:
        return pd.DataFrame(columns=columns)
    review = workflow_assets.loc[
        workflow_assets["confidence"].eq("needs_review"),
        [column for column in columns if column in workflow_assets.columns],
    ].copy()
    return review.reindex(columns=columns).sort_values("symbol").reset_index(drop=True)


def _numeric_leverage_matches(normalized_name: str) -> list[tuple[int, float]]:
    matches: list[tuple[int, float]] = []
    for match in re.finditer(NUMERIC_X_LEVERAGE_PATTERN, normalized_name):
        matches.append((match.start(), abs(float(match.group(1)))))
    for match in re.finditer(NUMERIC_PERCENT_LEVERAGE_PATTERN, normalized_name):
        matches.append((match.start(), abs(float(match.group(1))) / 100.0))
    return sorted(matches)


def _has_leverage_false_positive_term(name: str) -> bool:
    normalized_name = f" {str(name).upper()} "
    if not any(term in normalized_name for term in LEVERAGE_FALSE_POSITIVE_TERMS):
        return False
    explicit_leverage = _recognized_numeric_leverage(normalized_name)
    return explicit_leverage is None or explicit_leverage <= 1.0


def _curated_leverage_and_direction(normalized_name: str) -> tuple[float, str] | None:
    for pattern, leverage, direction in LEVERAGE_DIRECTION_NAME_OVERRIDES:
        if re.search(pattern, normalized_name, re.I):
            return leverage, direction
    return None


def _has_unrecognized_product_leverage(normalized_name: str) -> bool:
    product_context_pattern = re.compile(
        r"\s+(?:(?:DAILY\s+)?(?:TARGET\s+)?(?:LONG|BULL|SHORT|BEAR|INVERSE)\b|"
        r"(?:LEVERAGED|DAILY|TARGET)\b|[+-]?\d+(?:\.\d+)?\s*(?:X|%)(?![A-Z0-9]))",
        re.I,
    )
    for pattern in [NUMERIC_X_LEVERAGE_PATTERN, NUMERIC_PERCENT_LEVERAGE_PATTERN]:
        for match in re.finditer(pattern, normalized_name):
            raw_value = abs(float(match.group(1)))
            leverage = raw_value / 100.0 if pattern == NUMERIC_PERCENT_LEVERAGE_PATTERN else raw_value
            if leverage > MAX_RECOGNIZED_LEVERAGE and product_context_pattern.match(
                normalized_name[match.end():]
            ):
                return True
    return False


def _recognized_numeric_leverage(normalized_name: str) -> float | None:
    matches = _numeric_leverage_matches(normalized_name)
    for _position, leverage in matches:
        if 1.0 < leverage <= MAX_RECOGNIZED_LEVERAGE:
            return leverage
    for _position, leverage in matches:
        if leverage <= MAX_RECOGNIZED_LEVERAGE:
            return leverage
    return None


def infer_leverage_and_direction(name: str) -> tuple[float | None, str | None]:
    normalized_name = str(name).upper()
    if _has_leverage_false_positive_term(normalized_name):
        return None, None
    curated_leverage = _curated_leverage_and_direction(normalized_name)
    if curated_leverage is not None:
        return curated_leverage
    leverage = (
        None
        if _has_unrecognized_product_leverage(normalized_name)
        else _recognized_numeric_leverage(normalized_name)
    )
    if leverage is None:
        if re.search(r"\bULTRAPRO\b", normalized_name):
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


def _has_unrecognized_numeric_leverage(normalized_name: str) -> bool:
    if _has_unrecognized_product_leverage(normalized_name):
        return True
    numeric_leverages = [leverage for _position, leverage in _numeric_leverage_matches(normalized_name)]
    return (
        any(leverage > MAX_RECOGNIZED_LEVERAGE for leverage in numeric_leverages)
        and not any(1.0 < leverage <= MAX_RECOGNIZED_LEVERAGE for leverage in numeric_leverages)
    )


def leveraged_name_filter(name: str) -> bool:
    normalized_name = f" {str(name).upper()} "
    if _has_leverage_false_positive_term(normalized_name):
        return False
    if _has_unrecognized_numeric_leverage(normalized_name):
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
    leverage, direction = infer_leverage_and_direction(n)
    return leverage is not None and leverage > 1.0 and direction == "long"


def is_short_leveraged_name(name: str) -> bool:
    n = str(name).upper()
    if not leveraged_name_filter(n):
        return False
    leverage, direction = infer_leverage_and_direction(n)
    return leverage is not None and leverage > 1.0 and direction == "inverse"


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


def _decode_quoted_js_text(value: str) -> str:
    text = value.replace(r"\/", "/")
    text = text.replace(r"\'", "'").replace(r'\"', '"')
    text = text.replace(r"\u0026", "&").replace(r"\u00ae", "®").replace(r"\u2122", "™")
    return text


def _js_object_field(body: str, field_names: Iterable[str]) -> str | None:
    field_pattern = "|".join(re.escape(field_name) for field_name in field_names)
    match = re.search(
        rf"(?P<keyquote>['\"])?\b(?:{field_pattern})\b(?P=keyquote)?\s*:\s*"
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        body,
        re.I | re.S,
    )
    if match is None:
        return None
    return _decode_quoted_js_text(match.group("value"))


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


def _workflow_issuer_source(raw_source: object) -> UniverseSource:
    if isinstance(raw_source, UniverseSource):
        return raw_source
    try:
        source, url = raw_source  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported workflow issuer source: {raw_source!r}") from exc
    return UniverseSource(str(source), str(url), "issuer_etf")


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


def _defiance_json_to_universe(
    json_text: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    try:
        payload = json.loads(json_text)
    except ValueError:
        payload = []
    if not isinstance(payload, list):
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])

    rows = (
        {"symbol": item.get("ticker"), "name": item.get("name")}
        for item in payload
        if isinstance(item, dict)
    )
    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=require_leveraged,
    )


def _js_ticker_name_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    object_pattern = re.compile(r"\{(?P<body>[^{}]{0,2500}?['\"]?\bticker\b['\"]?[^{}]{0,2500}?)\}", re.I | re.S)

    for match in object_pattern.finditer(html):
        body = match.group("body")
        symbol = _js_object_field(body, ["ticker"])
        name = _js_object_field(body, ["fund", "name"])
        if symbol is not None and name is not None:
            rows.append({"symbol": symbol, "name": name})

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=require_leveraged,
    )


def _graniteshares_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    rows = []
    product_pattern = re.compile(
        r'etf-table-cell--ticker__symbol">\s*(?P<symbol>[A-Z][A-Z0-9.-]{0,7})\s*</span>'
        r'.{0,1800}?etf-table-cell--name-title[^>]*>\s*(?P<name>.*?)\s*</span>',
        re.I | re.S,
    )
    for match in product_pattern.finditer(html):
        rows.append({"symbol": match.group("symbol"), "name": match.group("name")})

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=require_leveraged,
    )


def _rex_menu_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    menu_pattern = re.compile(
        r">\s*(?P<symbol>[A-Z][A-Z0-9.-]{0,7})\s*\|\s*(?P<sign>[+-])(?P<multiple>\d+(?:\.\d+)?)X\s+"
        r"Daily\s+(?P<underlying>[^<]+?)\s*</a>",
        re.I | re.S,
    )
    for match in menu_pattern.finditer(html):
        multiple = match.group("multiple")
        underlying = _html_text(match.group("underlying"))
        direction = "Inverse" if match.group("sign") == "-" else "Long"
        rows.append(
            {
                "symbol": match.group("symbol"),
                "name": f"T-REX {multiple}X {direction} {underlying} Daily Target ETF",
            }
        )

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=require_leveraged,
    )


def _volatilityshares_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    rows = []
    product_pattern = re.compile(
        r"<h4>\s*(?P<symbol>[A-Z][A-Z0-9.-]{0,7})\s*</h4>\s*<p>\s*(?P<name>.*?)\s*</p>",
        re.I | re.S,
    )
    for match in product_pattern.finditer(html):
        name = _html_text(match.group("name"))
        full_name = name if name.upper().endswith(" ETF") else f"{name} ETF"
        rows.append({"symbol": match.group("symbol"), "name": full_name})

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} issuer table",
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

            name = row[name_col]
            leverage: float | None = None
            leverage_text = str(row[leverage_col]) if leverage_col is not None else ""
            if leverage_col is not None:
                leverage, direction = infer_leverage_and_direction(leverage_text)
                if require_leveraged and (leverage is None or leverage <= 1.0):
                    continue
                if leverage is not None and leverage > 1.0 and not leveraged_name_filter(str(name)):
                    direction_label = "Inverse " if direction == "inverse" else ""
                    name = f"{name} {leverage:g}X {direction_label}Leveraged"
            elif require_leveraged and not leveraged_name_filter(str(name)):
                continue

            rows.append({"symbol": symbol, "name": name})

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} ETN issuer table",
        require_leveraged=False,
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


def _workflow_issuer_source_to_universe(
    content: str,
    source: UniverseSource,
    *,
    require_leveraged: bool,
) -> pd.DataFrame:
    if source.parser == "defiance_json":
        return _defiance_json_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "js_ticker_name":
        return _js_ticker_name_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "graniteshares_html":
        return _graniteshares_html_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "rex_menu_html":
        return _rex_menu_html_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "volatilityshares_html":
        return _volatilityshares_html_to_universe(content, source, require_leveraged=require_leveraged)
    return _html_source_to_universe(
        content,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=require_leveraged,
    )


def load_issuer_etf_universe(timeout: int = 30) -> pd.DataFrame:
    rows = []
    status_rows = []
    for raw_source in ISSUER_UNIVERSE_SOURCES:
        source = _workflow_issuer_source(raw_source)
        if not source.enabled or source.parser == "registered_only":
            status_rows.append(
                _workflow_source_status_row(
                    source=source.name,
                    source_type=source.source_type,
                    url=source.url,
                    status="registered_only",
                    error=source.notes,
                )
            )
            continue
        try:
            resp = requests.get(source.url, timeout=timeout, headers=REQUEST_HEADERS)
            resp.raise_for_status()
        except Exception as exc:
            status_rows.append(
                _workflow_source_status_row(
                    source=source.name,
                    source_type=source.source_type,
                    url=source.url,
                    status="source_error",
                    error=f"{type(exc).__name__}: {exc}"[:250],
                )
            )
            continue
        parsed_rows = _workflow_issuer_source_to_universe(
            resp.text,
            source,
            require_leveraged=False,
        )
        issuer_rows = _workflow_issuer_source_to_universe(
            resp.text,
            source,
            require_leveraged=True,
        )
        status, error = _workflow_source_parse_status(parsed_rows, issuer_rows)
        if not issuer_rows.empty:
            rows.append(issuer_rows)
        status_rows.append(
            _workflow_source_status_row(
                source=source.name,
                source_type=source.source_type,
                url=source.url,
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
                    source_type=source.source_type,
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
                source_type=source.source_type,
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
        "is_short_leveraged_candidate",
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
                "is_short_leveraged_candidate": False,
                "leverage": None,
                "direction": None,
            }
        )

    leverage, direction = infer_leverage_and_direction(str(name))
    return pd.Series(
        {
            "is_leveraged_candidate": leveraged_name_filter(str(name)),
            "is_long_leveraged_candidate": is_long_leveraged_name(str(name)),
            "is_short_leveraged_candidate": is_short_leveraged_name(str(name)),
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

        if source_rows.empty:
            status_rows.append(
                _audit_source_status_row(
                    source,
                    status="error",
                    error="Parser returned no rows from an enabled audit source.",
                )
            )
            continue

        source_rows = _with_audit_metadata(source_rows, source)
        rows.append(source_rows)
        status_rows.append(_audit_source_status_row(source, status="loaded", row_count=len(source_rows)))

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
    if "is_short_leveraged_candidate" not in out.columns:
        out["is_short_leveraged_candidate"] = out["direction"].eq("inverse") & out[
            "is_leveraged_candidate"
        ].astype(bool)
    out["in_merged_universe"] = out["symbol"].isin(merged_symbols)
    out["in_workflow_universe"] = out["symbol"].isin(workflow_symbols)
    leveraged_candidate = (
        out["is_long_leveraged_candidate"].astype(bool)
        | out["is_short_leveraged_candidate"].astype(bool)
    )
    out = out[leveraged_candidate & ~out["in_merged_universe"]].copy()
    out["audit_reason"] = out["is_short_leveraged_candidate"].map(
        {
            True: "inverse leveraged-looking audit source row missing from merged source universe",
            False: "long leveraged-looking audit source row missing from merged source universe",
        }
    )
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
    mapping = infer_rsi_mapping(
        row["symbol"],
        row["name"],
        known_symbols=known_symbols,
        fund_type=row.get("fund_type"),
    )

    return pd.Series(
        {
            "rsi_symbol": mapping.rsi_symbol,
            "leverage": leverage,
            "direction": direction,
            "underlying_symbol": mapping.rsi_symbol,
            "underlying_name": mapping.underlying_name,
            "mapping_source": mapping.mapping_source,
            "confidence": mapping.confidence,
            "mapping_reason": mapping.mapping_reason,
        }
    )


def _known_rsi_symbols(etf_df: pd.DataFrame, active_symbols: set[str]) -> set[str]:
    known_symbols = set(etf_df["symbol"].dropna().astype(str).str.upper())
    known_symbols.update(str(symbol).upper() for symbol in active_symbols)
    return known_symbols


def _unusable_universe_source_message(
    workflow_source_failures: pd.DataFrame,
    active_listing_failures: pd.DataFrame,
) -> str:
    failure_groups = []
    if not workflow_source_failures.empty:
        failed_sources = ", ".join(workflow_source_failures["source"].astype(str).tolist())
        failure_groups.append(f"workflow universe sources were unusable: {failed_sources}")
    if not active_listing_failures.empty:
        failed_sources = ", ".join(active_listing_failures["source"].astype(str).tolist())
        failure_groups.append(f"active listing sources were unusable: {failed_sources}")
    return "; ".join(failure_groups)


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


def select_short_workflow_universe(etf_df: pd.DataFrame) -> pd.DataFrame:
    eligible = etf_df[~etf_df["symbol"].isin(EXCLUDED_UNIVERSE_SYMBOLS)].copy()
    all_short_leveraged = eligible.loc[
        eligible["name"].map(is_short_leveraged_name).astype(bool)
    ].copy()
    return all_short_leveraged.sort_values("symbol").reset_index(drop=True)


def determine_workflow_asset_groups(cfg: UniverseConfig) -> dict[str, pd.DataFrame]:
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
        ~workflow_source_status["status"].isin(NON_FAILURE_WORKFLOW_SOURCE_STATUSES)
    ].copy()
    discovered_df = pd.concat([issuer_df, etn_df], ignore_index=True, sort=False)
    active_symbols = load_active_listed_symbols(timeout=cfg.request_timeout_seconds)
    active_symbol_set = {str(symbol).upper() for symbol in active_symbols} if active_symbols else set()
    active_listing_complete = bool(getattr(active_symbols, "is_complete", True))
    active_listing_status = pd.DataFrame(
        getattr(active_symbols, "source_status", []),
        columns=["source", "url", "symbol_column", "status", "symbol_count", "error"],
    )
    active_listing_failures = active_listing_status[
        active_listing_status["status"].ne("loaded")
    ].copy()
    inactive_discovered = pd.DataFrame(columns=[*discovered_df.columns, "inactive_reason"])
    if active_symbol_set and active_listing_complete:
        is_active = discovered_df["symbol"].astype(str).str.upper().isin(active_symbol_set)
        inactive_discovered = discovered_df.loc[~is_active].copy()
        inactive_discovered["inactive_reason"] = "not present in active Nasdaq symbol files"
        discovered_df = discovered_df.loc[is_active].copy()
    etf_df = _merge_universe_sources(nasdaq_df, discovered_df)
    known_symbols = _known_rsi_symbols(etf_df, active_symbol_set)

    nasdaq_universe = build_nasdaq_universe_table(etf_df, known_symbols=known_symbols)
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

    if cfg.require_workflow_source_success and (
        not workflow_source_failures.empty or not active_listing_failures.empty
    ):
        raise RuntimeError(
            f"Workflow universe source checks failed: "
            f"{_unusable_universe_source_message(workflow_source_failures, active_listing_failures)}."
        )

    single_stock_long, all_long_leveraged = select_universes(etf_df)
    all_short_leveraged = select_short_workflow_universe(etf_df)
    if all_long_leveraged.empty and all_short_leveraged.empty:
        raise RuntimeError("Nasdaq ETF universe returned no current leveraged ETFs/ETNs.")

    audit_rows, audit_status = load_audit_universe_sources(timeout=cfg.request_timeout_seconds)
    all_workflow_leveraged = pd.concat(
        [all_long_leveraged, all_short_leveraged],
        ignore_index=True,
        sort=False,
    ).drop_duplicates("symbol")
    audit_report = build_universe_audit_report(audit_rows, etf_df, all_workflow_leveraged)
    audit_source_failures = (
        audit_status.loc[audit_status["status"].eq("error")].copy()
        if "status" in audit_status.columns
        else pd.DataFrame(columns=audit_status.columns)
    )
    save_table_to_sqlite(audit_rows, cfg.sqlite_db_path, "universe_audit_rows")
    save_table_to_sqlite(audit_report, cfg.sqlite_db_path, "universe_audit_missing_candidates")
    save_table_to_sqlite(audit_status, cfg.sqlite_db_path, "universe_audit_source_status")

    workflow_candidates_by_side = {
        "long": _workflow_candidates(all_long_leveraged, known_symbols, workflow_label="Long"),
        "short": _workflow_candidates(all_short_leveraged, known_symbols, workflow_label="Short"),
    }
    short_self_fallback = workflow_candidates_by_side["short"]["confidence"].eq("fallback_to_self")
    workflow_candidates_by_side["short"].loc[short_self_fallback, "confidence"] = "needs_review"
    workflow_candidates_by_side["short"].loc[short_self_fallback, "mapping_reason"] = (
        "inverse product requires an underlying RSI proxy; self-RSI would invert the upper-RSI entry rule"
    )
    all_workflow_candidates = pd.concat(
        workflow_candidates_by_side.values(),
        ignore_index=True,
        sort=False,
    )
    rsi_mapping_review = _rsi_mapping_review_table(all_workflow_candidates)
    executable_by_side = {
        side: candidates.loc[
            ~candidates["confidence"].eq("needs_review")
        ].copy()
        for side, candidates in workflow_candidates_by_side.items()
    }
    save_table_to_sqlite(
        rsi_mapping_review,
        cfg.sqlite_db_path,
        "universe_rsi_mapping_review",
    )

    if executable_by_side["long"].empty and executable_by_side["short"].empty:
        raise RuntimeError(
            "Workflow universe has no executable leveraged ETFs/ETNs after excluding RSI mappings needing review."
        )

    common_counts = {
        "Current ETFs in Nasdaq table": len(nasdaq_df),
        "Current issuer-discovered leveraged ETFs found": len(issuer_df),
        "Current issuer-discovered leveraged ETNs found": len(etn_df),
        "Inactive issuer-discovered ETFs/ETNs skipped": len(inactive_discovered),
        "Active listing sources loaded": int(
            (active_listing_status["status"] == "loaded").sum()
        ) if not active_listing_status.empty else 0,
        "Active listing snapshot complete": active_listing_complete,
        "Active listing sources failed": len(active_listing_failures),
        "Workflow universe sources failed": len(workflow_source_failures),
        "Merged current ETFs/ETNs": len(etf_df),
        "Current long single-stock leveraged ETFs found": len(single_stock_long),
        "Current long leveraged ETFs/ETNs found": len(all_long_leveraged),
        "Current short leveraged ETFs/ETNs found": len(all_short_leveraged),
        "RSI mappings needing review": len(rsi_mapping_review),
        "RSI mappings excluded pending review": len(rsi_mapping_review),
        "Audit sources registered": len(AUDIT_UNIVERSE_SOURCES),
        "Audit sources failed": len(audit_source_failures),
        "Audit rows parsed": len(audit_rows),
        "Audit leveraged candidates missing from merged universe": len(audit_report),
    }
    common_attrs = {
        "universe_degraded": (
            not workflow_source_failures.empty
            or not active_listing_failures.empty
            or not audit_source_failures.empty
        ),
        "workflow_source_failures": workflow_source_failures.to_dict("records"),
        "active_listing_source_failures": active_listing_failures.to_dict("records"),
        "audit_source_failures": audit_source_failures.to_dict("records"),
        "rsi_mapping_review": rsi_mapping_review.to_dict("records"),
        "universe_db_path": cfg.sqlite_db_path,
    }

    return {
        "long": _workflow_assets_output(
            executable_by_side["long"],
            cfg,
            workflow_label="Long",
            universe_title_base="Executable Long Leveraged ETFs/ETNs From Merged Universe",
            count_label="Executable long leveraged ETFs/ETNs selected",
            common_counts=common_counts,
            common_attrs=common_attrs,
        ),
        "short": _workflow_assets_output(
            executable_by_side["short"],
            cfg,
            workflow_label="Short",
            universe_title_base="Executable Short Leveraged ETFs/ETNs From Merged Universe",
            count_label="Executable short leveraged ETFs/ETNs selected",
            common_counts=common_counts,
            common_attrs=common_attrs,
        ),
    }


def determine_workflow_assets(cfg: UniverseConfig) -> pd.DataFrame:
    workflow_assets = determine_workflow_asset_groups(cfg)["long"]
    if workflow_assets.empty:
        raise RuntimeError(
            "Workflow universe has no executable long leveraged ETFs/ETNs after excluding RSI mappings needing review."
        )
    return workflow_assets


def _workflow_candidates(
    products: pd.DataFrame,
    known_symbols: set[str] | None,
    *,
    workflow_label: str,
) -> pd.DataFrame:
    if products.empty:
        return pd.DataFrame(
            columns=[
                *products.columns,
                "rsi_symbol",
                "leverage",
                "direction",
                "underlying_symbol",
                "underlying_name",
                "mapping_source",
                "confidence",
                "mapping_reason",
                "workflow",
            ]
        )
    candidate_metadata = products.apply(
        lambda row: _workflow_row_metadata(row, known_symbols),
        axis=1,
    )
    candidates = pd.concat(
        [products.reset_index(drop=True), candidate_metadata.reset_index(drop=True)],
        axis=1,
    )
    candidates["workflow"] = workflow_label
    return candidates


def _workflow_assets_output(
    workflow_assets: pd.DataFrame,
    cfg: UniverseConfig,
    *,
    workflow_label: str,
    universe_title_base: str,
    count_label: str,
    common_counts: dict[str, object],
    common_attrs: dict[str, object],
) -> pd.DataFrame:
    selected_assets = (
        workflow_assets
        if cfg.top_n is None
        else workflow_assets.head(cfg.top_n).copy()
    )

    universe_title = universe_title_base if cfg.top_n is None else f"First {cfg.top_n} {universe_title_base}"

    base_columns = ["symbol", "name", "rsi_symbol"]
    out = selected_assets.reindex(columns=base_columns).reset_index(drop=True)
    for column in [
        "workflow",
        "leverage",
        "direction",
        "underlying_symbol",
        "underlying_name",
        "fund_type",
        "source",
        "mapping_source",
        "confidence",
        "mapping_reason",
    ]:
        if column in selected_assets.columns:
            out[column] = selected_assets[column].to_numpy()
    if "workflow" not in out.columns:
        out["workflow"] = workflow_label
    counts = dict(common_counts)
    counts[count_label] = len(out)
    out.attrs["universe_title"] = universe_title
    out.attrs["universe_counts"] = counts
    out.attrs.update(common_attrs)
    return out


def build_nasdaq_universe_table(
    etf_df: pd.DataFrame,
    *,
    known_symbols: set[str] | None = None,
) -> pd.DataFrame:
    out = etf_df.copy()
    if known_symbols is None:
        known_symbols = _known_rsi_symbols(out, set())
    out["is_long_leveraged"] = out["name"].apply(is_long_leveraged_name)
    out["is_short_leveraged"] = out["name"].apply(is_short_leveraged_name)
    out["is_single_stock"] = out["fund_type"].str.contains(
        r"ETF \(Single Stock\)",
        regex=True,
        na=False,
    )
    out["is_single_stock_long_leveraged"] = out["is_single_stock"] & out["is_long_leveraged"]
    mapping_metadata = out.apply(
        lambda row: _nasdaq_table_rsi_mapping_metadata(row, known_symbols),
        axis=1,
    )
    out = pd.concat([out.reset_index(drop=True), mapping_metadata.reset_index(drop=True)], axis=1)
    return out.sort_values("symbol").reset_index(drop=True)


def _nasdaq_table_rsi_mapping_metadata(
    row: pd.Series,
    known_symbols: set[str] | None,
) -> pd.Series:
    if row["is_long_leveraged"] or row.get("is_short_leveraged", False):
        mapping = infer_rsi_mapping(
            row["symbol"],
            row["name"],
            known_symbols=known_symbols,
            fund_type=row.get("fund_type"),
        )
        return pd.Series(
            {
                "rsi_symbol": mapping.rsi_symbol,
                "underlying_symbol": mapping.rsi_symbol,
                "underlying_name": mapping.underlying_name,
                "mapping_source": mapping.mapping_source,
                "confidence": mapping.confidence,
                "mapping_reason": mapping.mapping_reason,
            }
        )

    symbol = row["symbol"]
    return pd.Series(
        {
            "rsi_symbol": symbol,
            "underlying_symbol": symbol,
            "underlying_name": symbol,
            "mapping_source": "asset_symbol",
            "confidence": "not_applicable",
            "mapping_reason": "not a workflow leveraged product",
        }
    )
