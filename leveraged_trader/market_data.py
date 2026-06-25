from __future__ import annotations

import logging
import re
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
import yfinance.shared as yf_shared

from .config import RISK_FREE_SYMBOL, TradierMarketDataConfig
from .storage import _date_str

_YFINANCE_DOWNLOAD_LOCK = threading.Lock()
_OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Volume"]
_TRADIER_PLACEHOLDER_TOKENS = {
    "",
    "your_tradier_access_token",
    "your_tradier_api_token",
    "replace_me",
}
TRADIER_RECOVERED_SYMBOLS_ATTR = "tradier_recovered_symbols"
_NEW_YORK = ZoneInfo("America/New_York")


class MarketDataDownloadError(RuntimeError):
    def __init__(self, symbol_reasons: Mapping[str, str], source: str = "Yahoo Finance") -> None:
        self.source = source
        self.symbol_reasons = {
            symbol: _clean_yfinance_error(reason)
            for symbol, reason in symbol_reasons.items()
        }
        super().__init__(self._format_message())

    @property
    def symbols(self) -> list[str]:
        return sorted(self.symbol_reasons)

    def _format_message(self) -> str:
        reason_groups: dict[str, list[str]] = {}
        for symbol, reason in self.symbol_reasons.items():
            reason_groups.setdefault(reason, []).append(symbol)

        details = []
        for reason, symbols in reason_groups.items():
            symbol_list = ", ".join(sorted(symbols))
            details.append(f"{symbol_list}: {reason}")

        return (
            f"{self.source} did not return usable daily data. "
            f"Impacted symbols: {'; '.join(details)}."
        )


@contextmanager
def _suppress_yfinance_logger():
    logger = logging.getLogger("yfinance")
    previous_disabled = logger.disabled
    previous_level = logger.level
    logger.disabled = True
    logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        logger.disabled = previous_disabled
        logger.setLevel(previous_level)


def _clean_yfinance_error(message: str) -> str:
    cleaned = str(message).strip()
    cleaned = cleaned.replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(r"^\$?[A-Z0-9.-]+:\s*", "", cleaned)
    cleaned = cleaned.strip(" .")
    for prefix in ("possibly delisted; ",):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned or "No data returned"


def _raise_download_error(symbols: list[str], reason: str, source: str = "Yahoo Finance") -> None:
    raise MarketDataDownloadError({symbol: reason for symbol in symbols}, source=source)


def _tradier_api_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _tradier_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("-", "/")


def _tradier_token_error(cfg: TradierMarketDataConfig) -> str | None:
    if not cfg.enabled:
        return "Tradier fallback is disabled"
    token = (cfg.access_token or "").strip()
    if token.lower() in _TRADIER_PLACEHOLDER_TOKENS:
        return "TRADIER_ACCESS_TOKEN is not configured"
    return None


def _response_error_message(resp: requests.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error") or payload.get("errors") or payload.get("fault")
        if isinstance(error, dict):
            description = error.get("description") or error.get("message") or error.get("detail")
            if description:
                return str(description)
        if isinstance(error, str):
            return error

    body = resp.text.strip()
    if body:
        return body[:300]
    return f"HTTP {resp.status_code}"


def _tradier_history_days(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    history = payload.get("history")
    if not isinstance(history, Mapping):
        return []

    raw_days = history.get("day")
    if isinstance(raw_days, list):
        return [day for day in raw_days if isinstance(day, Mapping)]
    if isinstance(raw_days, Mapping):
        return [raw_days]
    return []


def _load_tradier_symbol_frame(
    symbol: str,
    start: str | None,
    end: str | None,
    cfg: TradierMarketDataConfig,
) -> pd.DataFrame:
    token_error = _tradier_token_error(cfg)
    if token_error is not None:
        raise MarketDataDownloadError({symbol: token_error}, source="Tradier")
    access_token = str(cfg.access_token).strip()

    params = {
        "symbol": _tradier_symbol(symbol),
        "interval": "daily",
    }
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end

    resp = requests.get(
        f"{_tradier_api_base_url(cfg.base_url)}/markets/history",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        params=params,
        timeout=cfg.timeout_seconds,
    )
    if resp.status_code >= 400:
        raise MarketDataDownloadError({symbol: _response_error_message(resp)}, source="Tradier")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise MarketDataDownloadError({symbol: "Tradier returned invalid JSON"}, source="Tradier") from exc

    days = _tradier_history_days(payload)
    if not days:
        raise MarketDataDownloadError({symbol: "No historical daily data returned"}, source="Tradier")

    df = pd.DataFrame(days)
    rename_map = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    required = ["Date", *_OHLCV_FIELDS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise MarketDataDownloadError(
            {symbol: f"Tradier response was missing required columns: {', '.join(missing)}"},
            source="Tradier",
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    for column in _OHLCV_FIELDS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Date", *_OHLCV_FIELDS]).set_index("Date")[_OHLCV_FIELDS].sort_index()
    if df.empty:
        raise MarketDataDownloadError({symbol: "No usable historical daily rows returned"}, source="Tradier")

    df.columns = [f"{symbol}_{column}" for column in _OHLCV_FIELDS]
    return df


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Extract one symbol's OHLCV frame from a multi-symbol yfinance download,
    handling both ticker-first and ticker-second MultiIndex layouts.
    """
    if raw.empty:
        _raise_download_error([symbol], "No data returned")

    if not isinstance(raw.columns, pd.MultiIndex):
        df = raw.copy()
        df.columns = [str(c) for c in df.columns]
        return df

    lvl0 = list(raw.columns.get_level_values(0))
    lvl1 = list(raw.columns.get_level_values(1))

    if symbol in lvl0:
        df = raw[symbol].copy()
        df.columns = [str(c) for c in df.columns]
        return df

    if symbol in lvl1:
        df = raw.xs(symbol, axis=1, level=1).copy()
        df.columns = [str(c) for c in df.columns]
        return df

    unique_lvl0 = sorted(set(map(str, lvl0)))
    unique_lvl1 = sorted(set(map(str, lvl1)))
    raise MarketDataDownloadError(
        {
            symbol: (
                "Symbol was missing from the Yahoo Finance response "
                f"(available level0={unique_lvl0}, level1={unique_lvl1})"
            )
        }
    )


def _download_yfinance(
    symbols: list[str],
    start: str | None,
    end: str | None,
    auto_adjust: bool,
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    with _YFINANCE_DOWNLOAD_LOCK, _suppress_yfinance_logger():
        yf_shared._ERRORS = {}
        try:
            raw = yf.download(
                tickers=symbols,
                start=start,
                end=end,
                period="max" if start is None and end is None else None,
                interval="1d",
                auto_adjust=auto_adjust,
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as exc:
            return None, {symbol: str(exc) for symbol in symbols}
        return raw, dict(yf_shared._ERRORS)


def _yfinance_symbol_frames(
    raw: pd.DataFrame | None,
    symbols: list[str],
    download_errors: Mapping[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {
        symbol: str(reason)
        for symbol, reason in download_errors.items()
        if symbol in symbols
    }

    if raw is None or raw.empty:
        for symbol in symbols:
            errors.setdefault(symbol, "No data returned")
        return frames, errors

    for symbol in symbols:
        if symbol in errors:
            continue
        try:
            df = _extract_symbol_frame(raw, symbol)
        except MarketDataDownloadError as exc:
            errors.update(exc.symbol_reasons)
            continue

        missing_fields = [column for column in _OHLCV_FIELDS if column not in df.columns]
        if missing_fields:
            errors[symbol] = (
                "Yahoo Finance response was missing required OHLCV columns: "
                f"{', '.join(missing_fields)}"
            )
            continue

        df = df[_OHLCV_FIELDS].copy().dropna(subset=_OHLCV_FIELDS)
        if df.empty:
            errors[symbol] = "Yahoo Finance response had no complete OHLCV daily rows"
            continue
        df.columns = [f"{symbol}_{column}" for column in _OHLCV_FIELDS]
        frames[symbol] = df

    return frames, errors


def _load_tradier_fallback_frames(
    symbols: list[str],
    start: str | None,
    end: str | None,
    cfg: TradierMarketDataConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    token_error = _tradier_token_error(cfg)
    if token_error is not None:
        return frames, {symbol: token_error for symbol in symbols}

    for symbol in symbols:
        try:
            frames[symbol] = _load_tradier_symbol_frame(symbol, start, end, cfg)
        except MarketDataDownloadError as exc:
            errors.update(exc.symbol_reasons)
        except requests.RequestException as exc:
            errors[symbol] = str(exc)

    return frames, errors


def _combined_provider_reason(yahoo_reason: str | None, tradier_reason: str | None) -> str:
    parts = []
    if yahoo_reason:
        parts.append(f"Yahoo Finance: {_clean_yfinance_error(yahoo_reason)}")
    if tradier_reason:
        parts.append(f"Tradier fallback: {_clean_yfinance_error(tradier_reason)}")
    return "; ".join(parts) or "No data returned"


def _merged_symbol_frames(frames_by_symbol: Mapping[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
    frames = [frames_by_symbol[symbol] for symbol in symbols]
    return pd.concat(frames, axis=1, join="inner").dropna().sort_index()


def _raise_no_overlap_error(
    symbols: list[str],
    use_tradier_fallback: bool,
    tradier_errors: Mapping[str, str],
) -> None:
    if not use_tradier_fallback:
        _raise_download_error(symbols, "No overlapping daily market data", source="Market data providers")

    symbol_reasons = {}
    for symbol in symbols:
        tradier_reason = tradier_errors.get(symbol)
        reason = "No overlapping daily market data after Yahoo Finance primary download and Tradier fallback"
        if tradier_reason:
            reason = f"{reason}; Tradier fallback: {_clean_yfinance_error(tradier_reason)}"
        symbol_reasons[symbol] = reason

    raise MarketDataDownloadError(symbol_reasons, source="Market data providers")


def exclude_current_trading_session(
    data: pd.DataFrame,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Exclude today's US daily candle so live signals use settled prior data.

    A wall-clock cutoff cannot prove a third-party daily bar is final.  Keeping
    the current session out of the strategy makes the following premarket run
    the only live-submission window for the prior, settled session.
    """
    if data.empty:
        return data

    eastern_now = now.astimezone(_NEW_YORK) if now and now.tzinfo else now
    if eastern_now is None:
        eastern_now = datetime.now(_NEW_YORK)
    elif eastern_now.tzinfo is None:
        eastern_now = eastern_now.replace(tzinfo=_NEW_YORK)

    latest_session = pd.Timestamp(data.index.max()).date()
    if latest_session < eastern_now.date():
        return data

    finalized = data[pd.to_datetime(data.index).date < eastern_now.date()].copy()
    finalized.attrs.update(data.attrs)
    return finalized


def exclude_unfinalized_daily_bar(
    data: pd.DataFrame,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for settled-session filtering."""
    return exclude_current_trading_session(data, now=now)


def load_market_data(
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
    symbols: list[str] | None = None,
    tradier_cfg: TradierMarketDataConfig | None = None,
) -> pd.DataFrame:
    """
    Downloads daily data and returns a merged DataFrame with SYMBOL_Field
    columns. Yahoo Finance is the primary source; Tradier can recover symbols
    that Yahoo skips.
    """
    if symbols is None:
        raise ValueError("symbols must be provided")

    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise ValueError("symbols must not be empty")

    raw, download_errors = _download_yfinance(symbols, start, end, auto_adjust)
    frames_by_symbol, yahoo_errors = _yfinance_symbol_frames(raw, symbols, download_errors)

    missing_symbols = [symbol for symbol in symbols if symbol not in frames_by_symbol]
    tradier_errors: dict[str, str] = {}
    recovered_symbols: list[str] = []
    use_tradier_fallback = tradier_cfg is not None and tradier_cfg.enabled
    if missing_symbols and use_tradier_fallback:
        tradier_frames, tradier_errors = _load_tradier_fallback_frames(
            missing_symbols,
            start,
            end,
            tradier_cfg,
        )
        for symbol, frame in tradier_frames.items():
            frames_by_symbol[symbol] = frame
            recovered_symbols.append(symbol)

    unresolved_symbols = [symbol for symbol in symbols if symbol not in frames_by_symbol]
    if unresolved_symbols:
        if use_tradier_fallback:
            symbol_reasons = {
                symbol: _combined_provider_reason(
                    yahoo_errors.get(symbol),
                    tradier_errors.get(symbol),
                )
                for symbol in unresolved_symbols
            }
            source = "Yahoo Finance and Tradier"
        else:
            symbol_reasons = {
                symbol: yahoo_errors.get(symbol, "No data returned")
                for symbol in unresolved_symbols
            }
            source = "Yahoo Finance"
        raise MarketDataDownloadError(
            symbol_reasons,
            source=source,
        )

    out = _merged_symbol_frames(frames_by_symbol, symbols)
    if out.empty and use_tradier_fallback:
        tradier_frames, no_overlap_tradier_errors = _load_tradier_fallback_frames(
            symbols,
            start,
            end,
            tradier_cfg,
        )
        tradier_errors.update(no_overlap_tradier_errors)
        for symbol, frame in tradier_frames.items():
            frames_by_symbol[symbol] = frame
            if symbol not in recovered_symbols:
                recovered_symbols.append(symbol)
        if tradier_frames:
            out = _merged_symbol_frames(frames_by_symbol, symbols)

    if out.empty:
        _raise_no_overlap_error(symbols, use_tradier_fallback, tradier_errors)

    out.index = pd.to_datetime(out.index).tz_localize(None)
    if recovered_symbols:
        out.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = sorted(recovered_symbols)

    return out


def load_strategy_data(
    asset_symbol: str,
    signal_symbol: str,
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
    tradier_cfg: TradierMarketDataConfig | None = None,
) -> pd.DataFrame:
    core_symbols = [asset_symbol, signal_symbol]
    data = load_market_data(
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        symbols=core_symbols,
        tradier_cfg=tradier_cfg,
    )
    risk_free = load_market_data(
        start=_date_str(data.index.min()),
        end=end,
        auto_adjust=auto_adjust,
        symbols=[RISK_FREE_SYMBOL],
        tradier_cfg=tradier_cfg,
    )
    out = data.join(risk_free, how="left").ffill()
    recovered_symbols = sorted(
        {
            *data.attrs.get(TRADIER_RECOVERED_SYMBOLS_ATTR, []),
            *risk_free.attrs.get(TRADIER_RECOVERED_SYMBOLS_ATTR, []),
        }
    )
    if recovered_symbols:
        out.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = recovered_symbols
    return exclude_current_trading_session(out)


def load_signal_history(
    signal_symbol: str,
    *,
    end: str | None = None,
    auto_adjust: bool = True,
    tradier_cfg: TradierMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Load the canonical, settled daily history for one RSI signal symbol."""
    return load_symbol_history(
        signal_symbol,
        end=end,
        auto_adjust=auto_adjust,
        tradier_cfg=tradier_cfg,
    )


def load_symbol_history(
    symbol: str,
    *,
    end: str | None = None,
    auto_adjust: bool = True,
    tradier_cfg: TradierMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Load complete, settled daily history for one persisted market symbol."""
    data = load_market_data(
        start=None,
        end=end,
        auto_adjust=auto_adjust,
        symbols=[symbol],
        tradier_cfg=tradier_cfg,
    )
    return exclude_current_trading_session(data)


def load_risk_free_history(
    *,
    end: str | None = None,
    auto_adjust: bool = True,
    tradier_cfg: TradierMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Load the complete canonical benchmark history shared by all strategies."""
    return load_symbol_history(
        RISK_FREE_SYMBOL,
        end=end,
        auto_adjust=auto_adjust,
        tradier_cfg=tradier_cfg,
    )
