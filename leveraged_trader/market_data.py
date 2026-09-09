from __future__ import annotations

import codecs
import json
import logging
import math
import re
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import urllib3
import yfinance as yf
import yfinance.multi as yf_multi

from ._http_deadline_worker import response_from_http_worker_payload, run_http_request_with_deadline
from ._yfinance_deadline_worker import (
    exception_from_yfinance_worker_error,
    execute_yfinance_download,
    run_yfinance_download_with_deadline,
)
from .config import (
    RISK_FREE_SYMBOL,
    TRADIER_API_HOSTS,
    TRADIER_PLACEHOLDER_TOKENS,
    TradierMarketDataConfig,
    is_official_tradier_api_base_url,
)
from .output import DEFAULT_DIAGNOSTIC_MAX_CHARS, safe_diagnostic_text
from .storage import _date_str, validate_market_data_frame

_YFINANCE_DOWNLOAD_LOCK = threading.Lock()
_YFINANCE_REQUEST_TIMEOUT_SECONDS = 30
_OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Volume"]
_TRADIER_COMPLETE_HISTORY_START = "1900-01-01"
TRADIER_RECOVERED_SYMBOLS_ATTR = "tradier_recovered_symbols"
MARKET_DATA_PROVIDERS_ATTR = "market_data_providers"
_NEW_YORK = ZoneInfo("America/New_York")
_MAX_JSON_NESTING_DEPTH = 128
_MAX_JSON_STRUCTURAL_TOKENS = 1_000_000
TRADIER_RESPONSE_MAX_BYTES = 8 * 1024 * 1024
_TRADIER_RESPONSE_CHUNK_BYTES = 64 * 1024
_TRADIER_RESPONSE_READ_TIMEOUT_SECONDS = 5.0
_TRADIER_WORKER_RESULT_MAX_BYTES = TRADIER_RESPONSE_MAX_BYTES + 8 * 1024 * 1024
_TRUSTED_TRADIER_RESPONSE_ENCODINGS = frozenset(
    {
        "ascii",
        "cp1252",
        "iso8859-1",
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-be",
        "utf-16-le",
        "utf-32",
        "utf-32-be",
        "utf-32-le",
    }
)


@dataclass(frozen=True)
class _TradierConfigSnapshot:
    """One internally owned set of values for a credentialed request."""

    enabled: bool
    access_token: str | None
    base_url: str
    timeout_seconds: int


class _DuplicateJsonKeyError(ValueError):
    """Raised when an authoritative provider payload has ambiguous object keys."""


class _NonFiniteJsonNumberError(ValueError):
    """Raised when an authoritative provider payload contains a non-finite number."""


class _ExcessiveJsonNestingError(ValueError):
    """Raised before a provider payload can exceed the supported JSON depth."""


class _ExcessiveJsonStructureError(ValueError):
    """Raised before a compact provider payload can expand into too many objects."""


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for key, value in pairs:
        if key in decoded:
            raise _DuplicateJsonKeyError(f"duplicate object key {key!r}")
        decoded[key] = value
    return decoded


def _reject_nonstandard_json_constant(value: str) -> object:
    raise _NonFiniteJsonNumberError(value)


def _reject_excessive_json_text_nesting(value: str) -> None:
    """Bound JSON depth and structural size before allocating decoded containers."""
    depth = 0
    structural_tokens = 0
    in_string = False
    escaped = False
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            structural_tokens += 1
            if structural_tokens > _MAX_JSON_STRUCTURAL_TOKENS:
                raise _ExcessiveJsonStructureError(
                    f"JSON structure exceeds the supported limit of {_MAX_JSON_STRUCTURAL_TOKENS} tokens"
                )
            depth += 1
            if depth > _MAX_JSON_NESTING_DEPTH:
                raise _ExcessiveJsonNestingError(
                    f"JSON nesting exceeds the supported depth of {_MAX_JSON_NESTING_DEPTH}"
                )
        elif character in ",:":
            # Outside a string, every separator represents another collection
            # member or key/value edge. Counting it together with container
            # openings rejects compact payloads such as millions of ``{}``
            # values before ``json.loads`` expands them into Python objects.
            structural_tokens += 1
            if structural_tokens > _MAX_JSON_STRUCTURAL_TOKENS:
                raise _ExcessiveJsonStructureError(
                    f"JSON structure exceeds the supported limit of {_MAX_JSON_STRUCTURAL_TOKENS} tokens"
                )
        elif character in "]}":
            # Structural validity remains json.loads' responsibility. Keeping
            # this floor at zero prevents malformed leading closers from making
            # a later deeply nested value look shallower than it is.
            depth = max(0, depth - 1)


def _reject_nonfinite_json_numbers(value: object) -> None:
    """Reject non-finite numbers and excessive depth in decoded JSON values."""
    pending = [(value, 0)]
    while pending:
        item, parent_depth = pending.pop()
        if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
            raise _NonFiniteJsonNumberError(repr(item))
        if isinstance(item, Mapping):
            depth = parent_depth + 1
            if depth > _MAX_JSON_NESTING_DEPTH:
                raise _ExcessiveJsonNestingError(
                    f"JSON nesting exceeds the supported depth of {_MAX_JSON_NESTING_DEPTH}"
                )
            pending.extend((child, depth) for child in item.values())
        elif isinstance(item, (list, tuple)):
            depth = parent_depth + 1
            if depth > _MAX_JSON_NESTING_DEPTH:
                raise _ExcessiveJsonNestingError(
                    f"JSON nesting exceeds the supported depth of {_MAX_JSON_NESTING_DEPTH}"
                )
            pending.extend((child, depth) for child in item)


def _strict_response_json(resp: requests.Response) -> object:
    """Decode real HTTP bodies without JSON's otherwise silent last-key-wins rule.

    The response-object fallback keeps lightweight injected clients and tests
    compatible when they expose only ``json()`` and no raw response body.
    """
    raw_text = getattr(resp, "text", None)
    if isinstance(raw_text, str) and raw_text.strip():
        _reject_excessive_json_text_nesting(raw_text)
        decoded = json.loads(
            raw_text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    else:
        decoded = resp.json()
    _reject_nonfinite_json_numbers(decoded)
    return decoded


def _tradier_response_content_length(response: requests.Response) -> int | None:
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    raw_length = headers.get("Content-Length")
    if raw_length is None:
        return None
    if not isinstance(raw_length, str) or re.fullmatch(r"\d+", raw_length.strip()) is None:
        raise requests.exceptions.InvalidHeader("Tradier returned an invalid Content-Length header.")
    return int(raw_length.strip())


def _tradier_response_has_header(response: requests.Response, name: str) -> bool:
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return False
    normalized_name = name.casefold()
    return any(isinstance(header_name, str) and header_name.casefold() == normalized_name for header_name in headers)


def _tradier_raw_socket(response: requests.Response) -> object | None:
    """Return the socket governing one Requests/urllib3 streamed response."""
    raw_response = response.raw
    http_response = getattr(raw_response, "_fp", None)
    buffered_reader = getattr(http_response, "fp", None)
    socket_io = getattr(buffered_reader, "raw", None)
    socket = getattr(socket_io, "_sock", None)
    if callable(getattr(socket, "settimeout", None)):
        return socket
    connection_socket = getattr(getattr(raw_response, "_connection", None), "sock", None)
    if callable(getattr(connection_socket, "settimeout", None)):
        return connection_socket
    return None


def _tradier_socket_is_closed(socket: object) -> bool:
    """Return whether a response socket can no longer accept timeout updates."""
    # ``http.client`` closes its socket wrapper as soon as a Connection: close
    # response reaches EOF, while its buffered reader can still own and expose
    # the complete response body.  CPython keeps the descriptor alive through
    # that reader, so ``fileno()`` alone is insufficient: the socket's logical
    # closed flag is what makes ``settimeout`` raise EBADF.
    if getattr(socket, "_closed", False) is True:
        return True
    fileno = getattr(socket, "fileno", None)
    if not callable(fileno):
        return False
    try:
        descriptor = fileno()
    except (OSError, ValueError):
        return True
    return isinstance(descriptor, int) and descriptor < 0


def _tradier_response_chunks(response: requests.Response, *, deadline: float) -> Iterable[bytes]:
    raw_headers = getattr(response, "headers", None)
    headers = raw_headers if isinstance(raw_headers, Mapping) else {}
    content_encoding = headers.get("Content-Encoding")
    if content_encoding is not None and (
        not isinstance(content_encoding, str) or content_encoding.strip().casefold() not in {"", "identity"}
    ):
        raise requests.exceptions.ContentDecodingError(
            "Tradier returned compressed content despite an identity-only request."
        )
    if isinstance(response, requests.Response):
        materialized_content = getattr(response, "_content", None)
        if getattr(response, "_content_consumed", False) and isinstance(materialized_content, bytes):
            yield materialized_content
            return

        read_available = getattr(response.raw, "read1", None)
        if not callable(read_available):
            read_available = getattr(response.raw, "read", None)
        if not callable(read_available):
            raise TypeError("Tradier HTTP response does not support bounded incremental reads.")
        socket = _tradier_raw_socket(response)
        if socket is None:
            raise TypeError("Tradier HTTP response does not expose a bounded read timeout.")
        while True:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise requests.exceptions.Timeout("Tradier response exceeded its overall deadline.")
            if not _tradier_socket_is_closed(socket):
                try:
                    socket.settimeout(min(remaining_seconds, _TRADIER_RESPONSE_READ_TIMEOUT_SECONDS))
                except OSError:
                    # A Connection: close peer can race this timeout update
                    # after delivering the complete body.  Reading through the
                    # still-live buffered response is bounded and nonblocking
                    # once the socket wrapper is closed; preserve unrelated
                    # socket failures.
                    if not _tradier_socket_is_closed(socket):
                        raise
            chunk = read_available(_TRADIER_RESPONSE_CHUNK_BYTES)
            if not chunk:
                break
            yield chunk
        return

    # Preserve lightweight injected clients that expose either concrete text
    # or only ``json()``. Production Requests responses always stream bytes.
    adapter_text = getattr(response, "text", None)
    if not isinstance(adapter_text, str):
        return
    if len(adapter_text) > TRADIER_RESPONSE_MAX_BYTES:
        raise ValueError(f"Tradier response exceeded the {TRADIER_RESPONSE_MAX_BYTES}-byte limit.")
    encoded_text = adapter_text.encode("utf-8")
    if len(encoded_text) > TRADIER_RESPONSE_MAX_BYTES:
        raise ValueError(f"Tradier response exceeded the {TRADIER_RESPONSE_MAX_BYTES}-byte limit.")
    yield encoded_text


def _trusted_tradier_response_encoding(response: requests.Response, body: bytes) -> str:
    raw_encoding = getattr(response, "encoding", None)
    encoding = raw_encoding.strip() if isinstance(raw_encoding, str) and raw_encoding.strip() else "utf-8"
    try:
        canonical_encoding = codecs.lookup(encoding).name
    except LookupError as exc:
        raise requests.exceptions.InvalidHeader("Tradier declared an unsupported character encoding.") from exc
    if canonical_encoding not in _TRUSTED_TRADIER_RESPONSE_ENCODINGS:
        raise requests.exceptions.InvalidHeader("Tradier declared an unsupported character encoding.")
    if canonical_encoding == "utf-8" and body.startswith(codecs.BOM_UTF8):
        canonical_encoding = "utf-8-sig"
    try:
        body.decode(canonical_encoding, errors="strict")
    except UnicodeError as exc:
        raise requests.exceptions.ContentDecodingError(
            "Tradier response was not valid in its declared character encoding."
        ) from exc
    return canonical_encoding


def _materialize_tradier_response(response: requests.Response, *, deadline: float) -> None:
    content_length = _tradier_response_content_length(response)
    if content_length is not None and content_length > TRADIER_RESPONSE_MAX_BYTES:
        raise ValueError(f"Tradier response exceeded the {TRADIER_RESPONSE_MAX_BYTES}-byte limit.")

    body = bytearray()
    chunks = iter(_tradier_response_chunks(response, deadline=deadline))
    while True:
        if time.monotonic() >= deadline:
            raise requests.exceptions.Timeout("Tradier response exceeded its overall deadline.")
        try:
            chunk = next(chunks)
        except StopIteration:
            if time.monotonic() >= deadline:
                raise requests.exceptions.Timeout("Tradier response exceeded its overall deadline.") from None
            break
        if time.monotonic() >= deadline:
            raise requests.exceptions.Timeout("Tradier response exceeded its overall deadline.")
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError("Tradier returned a non-byte response chunk.")
        if len(body) + len(chunk) > TRADIER_RESPONSE_MAX_BYTES:
            raise ValueError(f"Tradier response exceeded the {TRADIER_RESPONSE_MAX_BYTES}-byte limit.")
        body.extend(chunk)

    if not isinstance(response, requests.Response) and not isinstance(getattr(response, "text", None), str):
        return
    materialized = bytes(body)
    encoding = _trusted_tradier_response_encoding(response, materialized)
    if isinstance(response, requests.Response):
        response._content = materialized
    else:
        response.text = materialized.decode(encoding, errors="strict")
    response.encoding = encoding


def _request_exception_from_process(name: str, message: str) -> BaseException:
    exception_type = {
        "ConnectTimeout": requests.exceptions.ConnectTimeout,
        "ConnectionError": requests.exceptions.ConnectionError,
        "ContentDecodingError": requests.exceptions.ContentDecodingError,
        "InvalidHeader": requests.exceptions.InvalidHeader,
        "InvalidURL": requests.exceptions.InvalidURL,
        "ReadTimeout": requests.exceptions.ReadTimeout,
        "SSLError": requests.exceptions.SSLError,
        "Timeout": requests.exceptions.Timeout,
        "TooManyRedirects": requests.exceptions.TooManyRedirects,
        "TypeError": TypeError,
        "ValueError": ValueError,
    }.get(name, requests.exceptions.RequestException)
    return exception_type(message)


def _get_tradier_response_with_deadline(
    url: str,
    *,
    deadline: float,
    request_kwargs: dict[str, object],
) -> requests.Response:
    """Acquire headers and body under one interruptible wall-clock deadline."""
    # Unit-test response doubles may not be pickleable. Restrict this
    # compatibility path to explicit unittest.mock objects so an ordinary
    # production wrapper cannot silently disable the hard deadline.
    if isinstance(requests.get, Mock):
        return requests.get(url, **request_kwargs)
    payload = run_http_request_with_deadline(
        url,
        request_kwargs,
        mode="tradier",
        deadline=deadline,
        max_body_bytes=TRADIER_RESPONSE_MAX_BYTES,
        result_max_bytes=_TRADIER_WORKER_RESULT_MAX_BYTES,
        timeout_message="Tradier response exceeded its overall deadline.",
        connection_error_message="Tradier request worker exited without a complete response.",
    )

    return response_from_http_worker_payload(
        payload,
        max_body_bytes=TRADIER_RESPONSE_MAX_BYTES,
        invalid_response_message="Tradier request worker returned an invalid response.",
        exception_from_process=_request_exception_from_process,
    )


class MarketDataDownloadError(RuntimeError):
    def __init__(
        self,
        symbol_reasons: Mapping[str, str],
        source: str = "Yahoo Finance",
        *,
        sensitive_values: Iterable[object] = (),
    ) -> None:
        self.source = source
        self.symbol_reasons = {
            symbol: _clean_yfinance_error(reason, sensitive_values=sensitive_values)
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
            symbol_list = ", ".join(
                sorted(safe_diagnostic_text(symbol, max_chars=128) or "[invalid symbol]" for symbol in symbols)
            )
            details.append(f"{symbol_list}: {reason}")

        source = safe_diagnostic_text(self.source, max_chars=128) or "Market data provider"
        return f"{source} did not return usable daily data. Impacted symbols: {'; '.join(details)}."


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


def _clean_yfinance_error(
    message: object,
    *,
    sensitive_values: Iterable[object] = (),
) -> str:
    cleaned = safe_diagnostic_text(
        message,
        max_chars=DEFAULT_DIAGNOSTIC_MAX_CHARS,
        sensitive_values=sensitive_values,
    )
    cleaned = re.sub(r"^\$?[A-Z0-9.-]+:\s*", "", cleaned)
    cleaned = cleaned.strip(" .")
    for prefix in ("possibly delisted; ",):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned or "No data returned"


def _raise_download_error(symbols: list[str], reason: str, source: str = "Yahoo Finance") -> None:
    raise MarketDataDownloadError({symbol: reason for symbol in symbols}, source=source)


def _tradier_api_base_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _tradier_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("-", "/")


def _is_safe_requested_symbol(value: object) -> bool:
    """Return whether a caller-provided provider identifier is terminal-safe.

    Provider identifiers vary beyond exchange tickers (for example Yahoo index,
    currency, and ISIN forms), so validation deliberately avoids inventing a
    narrower grammar. Whitespace and non-printing characters are never valid
    identity material and must not reach provider clients or diagnostics.
    """
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and all(character.isprintable() and not character.isspace() for character in value)
    )


def _tradier_configuration_shape_error(
    cfg: TradierMarketDataConfig | _TradierConfigSnapshot,
) -> str | None:
    if type(cfg.enabled) is not bool:
        return "Tradier enabled must be a boolean"
    if type(cfg.timeout_seconds) is not int or not 1 <= cfg.timeout_seconds <= 600:
        return "TRADIER_TIMEOUT_SECONDS must be an integer between 1 and 600"
    if cfg.access_token is not None and type(cfg.access_token) is not str:
        return "TRADIER_ACCESS_TOKEN must be a string or null"
    # Preserve non-space whitespace so it is rejected instead of silently
    # normalized out of a header-bound credential.
    normalized_token = (cfg.access_token or "").strip(" ")
    if normalized_token and any(character.isspace() or not character.isprintable() for character in normalized_token):
        return "TRADIER_ACCESS_TOKEN must not contain whitespace or control characters"
    try:
        normalized_token.encode("latin-1")
    except UnicodeEncodeError:
        return "TRADIER_ACCESS_TOKEN must contain only HTTP-header-compatible characters"
    if type(cfg.base_url) is not str:
        return "TRADIER_BASE_URL must be a string"
    return None


def _tradier_config_error(
    cfg: TradierMarketDataConfig | _TradierConfigSnapshot,
) -> str | None:
    shape_error = _tradier_configuration_shape_error(cfg)
    if shape_error is not None:
        return shape_error
    if not cfg.enabled:
        return "Tradier fallback is disabled"
    token = (cfg.access_token or "").strip()
    if token.lower() in TRADIER_PLACEHOLDER_TOKENS:
        return "TRADIER_ACCESS_TOKEN is not configured"
    if not is_official_tradier_api_base_url(cfg.base_url):
        hosts = ", ".join(sorted(TRADIER_API_HOSTS))
        return f"TRADIER_BASE_URL must be an HTTPS Tradier API endpoint on one of: {hosts}"
    return None


def _tradier_config_snapshot(cfg: TradierMarketDataConfig) -> _TradierConfigSnapshot:
    """Detach request inputs from caller-owned mutable configuration."""
    return _TradierConfigSnapshot(
        enabled=cfg.enabled,
        access_token=cfg.access_token,
        base_url=cfg.base_url,
        timeout_seconds=cfg.timeout_seconds,
    )


def _tradier_sensitive_values(cfg: TradierMarketDataConfig | None) -> tuple[str, ...]:
    if cfg is None or not isinstance(cfg.access_token, str) or not cfg.access_token.strip():
        return ()
    return (cfg.access_token.strip(),)


def _response_error_message(
    resp: requests.Response,
    *,
    sensitive_values: Iterable[object] = (),
) -> str:
    try:
        payload = _strict_response_json(resp)
    except _DuplicateJsonKeyError as exc:
        return safe_diagnostic_text(
            f"Tradier returned ambiguous error JSON: {exc}",
            max_chars=DEFAULT_DIAGNOSTIC_MAX_CHARS,
            sensitive_values=sensitive_values,
        )
    except _NonFiniteJsonNumberError:
        return "Tradier returned invalid error JSON containing a non-finite number"
    except (_ExcessiveJsonNestingError, RecursionError):
        return "Tradier returned invalid error JSON with excessive nesting"
    except _ExcessiveJsonStructureError:
        return "Tradier returned invalid error JSON with excessive structure"
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error") or payload.get("errors") or payload.get("fault")
        if isinstance(error, dict):
            description = error.get("description") or error.get("message") or error.get("detail")
            if description:
                return safe_diagnostic_text(
                    description,
                    max_chars=DEFAULT_DIAGNOSTIC_MAX_CHARS,
                    sensitive_values=sensitive_values,
                )
        if isinstance(error, str):
            return safe_diagnostic_text(
                error,
                max_chars=DEFAULT_DIAGNOSTIC_MAX_CHARS,
                sensitive_values=sensitive_values,
            )

    body = getattr(resp, "text", "")
    if body:
        return safe_diagnostic_text(
            body,
            max_chars=DEFAULT_DIAGNOSTIC_MAX_CHARS,
            sensitive_values=sensitive_values,
        )
    return f"HTTP {resp.status_code}"


def _tradier_history_days(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    history = payload.get("history")
    if not isinstance(history, Mapping):
        return []

    raw_days = history.get("day")
    if isinstance(raw_days, list):
        if any(not isinstance(day, Mapping) for day in raw_days):
            raise ValueError("Tradier returned a daily history list containing a non-object entry")
        return raw_days
    if isinstance(raw_days, Mapping):
        return [raw_days]
    return []


def _history_boundary(value: str | None, *, label: str) -> pd.Timestamp | None:
    if value is None:
        return None
    error = f"Market-data {label} must be a date in canonical YYYY-MM-DD format; got {value!r}."
    if not isinstance(value, str):
        raise ValueError(error)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(error) from exc
    # ``strptime`` accepts some non-zero-padded components. The round trip keeps
    # provider parameters and local clipping on one unambiguous daily contract.
    if parsed.isoformat() != value:
        raise ValueError(error)
    return pd.Timestamp(parsed)


def _tradier_daily_date(value: object, *, row_number: int) -> pd.Timestamp:
    error = f"Tradier response daily date must be a canonical YYYY-MM-DD string; row {row_number} contained {value!r}"
    if type(value) is not str:
        raise ValueError(error)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(error) from exc
    if parsed.isoformat() != value:
        raise ValueError(error)
    return pd.Timestamp(parsed)


def _clip_history_frame_to_requested_bounds(
    frame: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Enforce the shared provider contract: inclusive start, exclusive end."""
    start_boundary = _history_boundary(start, label="start")
    end_boundary = _history_boundary(end, label="end")
    if start_boundary is not None and end_boundary is not None and end_boundary <= start_boundary:
        raise ValueError("Market-data end must be strictly after start.")
    if frame.empty or (start_boundary is None and end_boundary is None):
        return frame

    session_index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    if session_index.tz is not None:
        session_index = session_index.tz_localize(None)
    in_range = pd.Series(True, index=frame.index)
    if start_boundary is not None:
        in_range &= session_index >= start_boundary
    if end_boundary is not None:
        in_range &= session_index < end_boundary
    clipped = frame.loc[in_range.to_numpy()].copy()
    clipped.index = session_index[in_range.to_numpy()]
    return clipped


def _load_tradier_symbol_frame(
    symbol: str,
    start: str | None,
    end: str | None,
    cfg: TradierMarketDataConfig,
    *,
    auto_adjust: bool,
) -> pd.DataFrame:
    # Validate and consume one private immutable copy.  In particular, never
    # re-read a caller-owned base URL after approving it for a bearer token.
    request_cfg = _tradier_config_snapshot(cfg)
    config_error = _tradier_config_error(request_cfg)
    if config_error is not None:
        raise MarketDataDownloadError({symbol: config_error}, source="Tradier")
    if auto_adjust:
        raise MarketDataDownloadError(
            {
                symbol: (
                    "Tradier historical prices cannot be used when auto_adjust=True because "
                    "their adjustment basis is not guaranteed to match Yahoo Finance"
                )
            },
            source="Tradier",
        )
    access_token = str(request_cfg.access_token).strip()
    sensitive_values = (access_token,)

    params = {
        "symbol": _tradier_symbol(symbol),
        "interval": "daily",
    }
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end

    deadline = time.monotonic() + request_cfg.timeout_seconds
    remaining_seconds = deadline - time.monotonic()
    if remaining_seconds <= 0:
        raise MarketDataDownloadError(
            {symbol: "Tradier response exceeded its overall deadline"},
            source="Tradier",
            sensitive_values=sensitive_values,
        )
    request_url = f"{_tradier_api_base_url(request_cfg.base_url.strip())}/markets/history"
    request_kwargs: dict[str, object] = {
        "headers": {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
        "params": params,
        "timeout": urllib3.util.Timeout(
            total=remaining_seconds,
            connect=remaining_seconds,
            read=min(remaining_seconds, _TRADIER_RESPONSE_READ_TIMEOUT_SECONDS),
        ),
        "allow_redirects": False,
        "stream": True,
    }
    resp: requests.Response | None = None
    primary_failure: BaseException | None = None
    try:
        try:
            resp = _get_tradier_response_with_deadline(
                request_url,
                deadline=deadline,
                request_kwargs=request_kwargs,
            )
            status_code = getattr(resp, "status_code", None)
            if type(status_code) is int and status_code == 200 and _tradier_response_has_header(resp, "Content-Range"):
                raise requests.exceptions.InvalidHeader(
                    "Tradier returned an unsolicited Content-Range header for a complete history snapshot."
                )
            if type(status_code) is int and (status_code == 200 or 400 <= status_code < 600):
                # A history snapshot is authoritative only at exact 200.
                # Preserve HTTP-error bodies for provider diagnostics, but do
                # not consume or retain partial-success or redirect bodies.
                _materialize_tradier_response(resp, deadline=deadline)
        except requests.Timeout as exc:
            raise MarketDataDownloadError(
                {symbol: "Tradier response exceeded its overall deadline"},
                source="Tradier",
                sensitive_values=sensitive_values,
            ) from exc
        except (requests.RequestException, urllib3.exceptions.HTTPError, OSError, TypeError, ValueError) as exc:
            detail = safe_diagnostic_text(
                exc,
                max_chars=DEFAULT_DIAGNOSTIC_MAX_CHARS,
                sensitive_values=sensitive_values,
            )
            raise MarketDataDownloadError(
                {symbol: f"Tradier returned an invalid HTTP response: {detail}"},
                source="Tradier",
                sensitive_values=sensitive_values,
            ) from exc
        assert resp is not None
        return _tradier_symbol_frame_from_response(
            resp,
            symbol=symbol,
            start=start,
            end=end,
            sensitive_values=sensitive_values,
        )
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        close_response = getattr(resp, "close", None) if resp is not None else None
        if callable(close_response):
            try:
                close_response()
            except BaseException as close_failure:
                if primary_failure is None:
                    raise
                close_detail = safe_diagnostic_text(
                    close_failure,
                    max_chars=DEFAULT_DIAGNOSTIC_MAX_CHARS,
                    sensitive_values=sensitive_values,
                )
                primary_failure.add_note(f"Failed to close the Tradier HTTP response: {close_detail}")


def _tradier_symbol_frame_from_response(
    resp: requests.Response,
    *,
    symbol: str,
    start: str | None,
    end: str | None,
    sensitive_values: tuple[str, ...],
) -> pd.DataFrame:
    """Interpret one materialized response while its owner retains cleanup responsibility."""
    status_code = getattr(resp, "status_code", None)
    if type(status_code) is not int or status_code != 200:
        if type(status_code) is int and 400 <= status_code < 600:
            detail = _response_error_message(
                resp,
                sensitive_values=sensitive_values,
            )
        else:
            detail = f"Tradier returned unexpected HTTP status {status_code}"
        raise MarketDataDownloadError(
            {symbol: detail},
            source="Tradier",
            sensitive_values=sensitive_values,
        )

    try:
        payload = _strict_response_json(resp)
    except _DuplicateJsonKeyError as exc:
        raise MarketDataDownloadError(
            {symbol: f"Tradier returned ambiguous JSON: {exc}"},
            source="Tradier",
            sensitive_values=sensitive_values,
        ) from exc
    except _ExcessiveJsonNestingError as exc:
        raise MarketDataDownloadError(
            {symbol: "Tradier returned invalid JSON with excessive nesting"},
            source="Tradier",
            sensitive_values=sensitive_values,
        ) from exc
    except _ExcessiveJsonStructureError as exc:
        raise MarketDataDownloadError(
            {symbol: "Tradier returned invalid JSON with excessive structure"},
            source="Tradier",
            sensitive_values=sensitive_values,
        ) from exc
    except (RecursionError, ValueError) as exc:
        raise MarketDataDownloadError(
            {symbol: "Tradier returned invalid JSON"},
            source="Tradier",
            sensitive_values=sensitive_values,
        ) from exc
    if not isinstance(payload, Mapping):
        raise MarketDataDownloadError(
            {symbol: "Tradier returned JSON with an invalid top-level structure"},
            source="Tradier",
            sensitive_values=sensitive_values,
        )

    try:
        days = _tradier_history_days(payload)
    except ValueError as exc:
        raise MarketDataDownloadError(
            {symbol: str(exc)},
            source="Tradier",
            sensitive_values=sensitive_values,
        ) from exc
    if not days:
        raise MarketDataDownloadError(
            {symbol: "No historical daily data returned"},
            source="Tradier",
            sensitive_values=sensitive_values,
        )

    try:
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
                sensitive_values=sensitive_values,
            )
        duplicate = [column for column in required if int((df.columns == column).sum()) > 1]
        if duplicate:
            raise MarketDataDownloadError(
                {
                    symbol: (
                        "Tradier response had duplicate required columns after field "
                        f"normalization: {', '.join(duplicate)}"
                    )
                },
                source="Tradier",
                sensitive_values=sensitive_values,
            )

        df["Date"] = [
            _tradier_daily_date(value, row_number=row_number) for row_number, value in enumerate(df["Date"].tolist())
        ]
        df = df.set_index("Date")[_OHLCV_FIELDS]
        df.columns = [f"{symbol}_{column}" for column in _OHLCV_FIELDS]
        df = _clip_history_frame_to_requested_bounds(df, start=start, end=end)
        if df.empty:
            raise MarketDataDownloadError(
                {symbol: "Tradier response had no daily rows inside the requested date range"},
                source="Tradier",
                sensitive_values=sensitive_values,
            )
        try:
            validate_market_data_frame(df, symbol, source="Tradier response")
        except ValueError as exc:
            raise MarketDataDownloadError(
                {symbol: str(exc)},
                source="Tradier",
                sensitive_values=sensitive_values,
            ) from exc
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    except MarketDataDownloadError:
        raise
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
        detail = _clean_yfinance_error(str(exc), sensitive_values=sensitive_values)
        raise MarketDataDownloadError(
            {symbol: f"Tradier returned malformed daily history: {detail}"},
            source="Tradier",
            sensitive_values=sensitive_values,
        ) from exc
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

    def extracted_frame(level: int, label: object) -> pd.DataFrame:
        df = raw.xs(label, axis=1, level=level).copy()
        df.columns = [str(c) for c in df.columns]
        return df

    levels = (lvl0, lvl1)
    exact_matches = [(level, symbol) for level, values in enumerate(levels) if symbol in values]
    if len(exact_matches) == 1:
        exact_match = exact_matches[0]
        exact_frame = extracted_frame(*exact_match)
        if all(field in exact_frame.columns for field in _OHLCV_FIELDS):
            return exact_frame

        # A requested ticker can itself use Yahoo's title-cased OHLCV names
        # (for example caller spelling ``Open`` for ticker ``OPEN``).  In that
        # case an exact field-level match is weaker evidence than the single
        # case-insensitive level whose extracted frame has the complete OHLCV
        # schema.  Keep the ordinary no-exact-match ambiguity rule below: this
        # exception is only for recovering from a demonstrably invalid exact
        # extraction.
        viable_fallbacks: list[pd.DataFrame] = []
        for level, values in enumerate(levels):
            for label in dict.fromkeys(value for value in values if str(value).casefold() == symbol.casefold()):
                if (level, label) == exact_match:
                    continue
                candidate = extracted_frame(level, label)
                if all(field in candidate.columns for field in _OHLCV_FIELDS):
                    viable_fallbacks.append(candidate)
        if len(viable_fallbacks) == 1:
            return viable_fallbacks[0]
        if len(viable_fallbacks) > 1:
            raise MarketDataDownloadError(
                {
                    symbol: (
                        "Symbol's exact Yahoo Finance column match was not an OHLCV frame, "
                        "and multiple case-insensitive ticker frames were viable"
                    )
                }
            )
        return exact_frame
    if len(exact_matches) > 1:
        raise MarketDataDownloadError({symbol: "Symbol appeared exactly in multiple Yahoo Finance column levels"})

    casefold_matches: list[tuple[int, object]] = []
    for level, values in enumerate(levels):
        labels = list(dict.fromkeys(value for value in values if str(value).casefold() == symbol.casefold()))
        casefold_matches.extend((level, label) for label in labels)
    if len(casefold_matches) == 1:
        return extracted_frame(*casefold_matches[0])
    if casefold_matches:
        raise MarketDataDownloadError(
            {
                symbol: (
                    "Symbol matched multiple Yahoo Finance column labels "
                    "case-insensitively; response layout was ambiguous"
                )
            }
        )

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
    deadline = time.monotonic() + _YFINANCE_REQUEST_TIMEOUT_SECONDS
    injected_download = isinstance(yf.download, Mock) or isinstance(getattr(yf_multi, "_download_impl", None), Mock)
    symbol_aliases: Mapping[str, str] = {}
    try:
        if injected_download:
            # Keep deterministic in-process test doubles observable. Ordinary
            # production callables never bypass the killable worker boundary.
            with _YFINANCE_DOWNLOAD_LOCK, _suppress_yfinance_logger():
                raw, download_errors, symbol_aliases = execute_yfinance_download(
                    symbols,
                    start,
                    end,
                    auto_adjust,
                    request_timeout_seconds=float(_YFINANCE_REQUEST_TIMEOUT_SECONDS),
                )
        else:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0 or not _YFINANCE_DOWNLOAD_LOCK.acquire(timeout=remaining_seconds):
                raise requests.exceptions.Timeout("Yahoo Finance response exceeded its overall deadline.")
            try:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise requests.exceptions.Timeout("Yahoo Finance response exceeded its overall deadline.")
                payload = run_yfinance_download_with_deadline(
                    symbols,
                    start,
                    end,
                    auto_adjust,
                    request_timeout_seconds=remaining_seconds,
                    deadline=deadline,
                )
            finally:
                _YFINANCE_DOWNLOAD_LOCK.release()

            worker_error = exception_from_yfinance_worker_error(payload)
            if worker_error is not None:
                raise worker_error
            if type(payload) is not tuple or not payload:
                raise requests.exceptions.ConnectionError("Yahoo Finance worker returned an invalid response.")
            if type(payload[0]) is not str or payload[0] != "yfinance_response" or len(payload) != 4:
                raise requests.exceptions.ConnectionError("Yahoo Finance worker returned an invalid response.")
            _, raw, download_errors, symbol_aliases = payload
            if raw is not None and not isinstance(raw, pd.DataFrame):
                raise requests.exceptions.ConnectionError("Yahoo Finance worker returned an invalid market-data frame.")
            if not isinstance(download_errors, dict) or not all(
                type(key) is str and type(value) is str for key, value in download_errors.items()
            ):
                raise requests.exceptions.ConnectionError("Yahoo Finance worker returned invalid error metadata.")
            if not isinstance(symbol_aliases, dict) or not all(
                type(key) is str and type(value) is str for key, value in symbol_aliases.items()
            ):
                raise requests.exceptions.ConnectionError(
                    "Yahoo Finance worker returned invalid symbol-alias metadata."
                )
    except Exception as exc:
        diagnostic = safe_diagnostic_text(exc, max_chars=DEFAULT_DIAGNOSTIC_MAX_CHARS)
        return None, {symbol: diagnostic for symbol in symbols}

    runtime_aliases_by_requested_identity = {
        str(requested_identity).strip().removeprefix("$").casefold(): str(provider_identity)
        .strip()
        .removeprefix("$")
        .casefold()
        for provider_identity, requested_identity in symbol_aliases.items()
    }
    requested_by_runtime_identity: dict[str, str] = {}
    for symbol in symbols:
        requested_identity = symbol.strip().removeprefix("$").casefold()
        runtime_identity = runtime_aliases_by_requested_identity.get(requested_identity, requested_identity)
        prior = requested_by_runtime_identity.get(runtime_identity)
        if prior is not None and prior != symbol:
            raise ValueError(
                "Requested market-data symbols collide after Yahoo Finance resolved identifier aliases: "
                f"{prior!r} and {symbol!r}."
            )
        requested_by_runtime_identity[runtime_identity] = symbol
    return raw, _yfinance_errors_for_requested_symbols(
        symbols,
        download_errors,
        symbol_aliases=symbol_aliases,
    )


def _yfinance_errors_for_requested_symbols(
    symbols: list[str],
    download_errors: Mapping[str, object],
    *,
    symbol_aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Map yfinance's normalized diagnostic keys back to caller spellings."""

    def symbol_key(value: object) -> str:
        return str(value).strip().removeprefix("$").casefold()

    requested_by_key: dict[str, list[str]] = {}
    for symbol in symbols:
        requested_by_key.setdefault(symbol_key(symbol), []).append(symbol)

    aliases_by_key = {symbol_key(source): symbol_key(target) for source, target in (symbol_aliases or {}).items()}
    mapped: dict[str, str] = {}
    unmatched: list[str] = []
    for error_symbol, reason in download_errors.items():
        error_key = symbol_key(error_symbol)
        candidate_keys = (error_key, aliases_by_key.get(error_key, error_key))
        matched = False
        for candidate_key in dict.fromkeys(candidate_keys):
            for requested_symbol in requested_by_key.get(candidate_key, ()):
                mapped[requested_symbol] = str(reason)
                matched = True
        if not matched:
            unmatched.append(str(reason))

    # An ISIN or another identifier may be rewritten to a ticker without an
    # alias being exposed by a future yfinance version.  With exactly one
    # request and one error, ownership is still unambiguous.
    if not mapped and len(symbols) == 1 and len(unmatched) == 1:
        mapped[symbols[0]] = unmatched[0]
    return mapped


def _yahoo_price_ordering_tolerance(first: float, second: float) -> float:
    """Bound representation jitter observed in Yahoo adjusted daily prices."""
    # ``numpy.spacing(float64_max)`` overflows to infinity because there is no
    # finite successor.  ``math.ulp`` uses the finite predecessor gap there,
    # preserving the narrow repair allowance without turning it unbounded.
    return 4.0 * max(math.ulp(first), math.ulp(second))


def _repair_yahoo_adjusted_ohlc_jitter(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Repair only a few-float64-ULP Yahoo High/Low ordering inversions.

    Yahoo's independently adjusted OHLC fields can acquire a few float64 ULPs
    of ordering error.  Keep the original float64 values
    and snap only a narrowly bounded ordering inversion to the constraining
    Open/Close/Low or Open/Close/High value.  Malformed semantic types are left
    untouched for the shared validator to reject with its normal diagnostic.
    """
    repaired = data.copy()
    for field in _OHLCV_FIELDS:
        column = f"{symbol}_{field}"
        source_values = repaired[column]
        if pd.api.types.is_datetime64_any_dtype(source_values.dtype) or pd.api.types.is_timedelta64_dtype(
            source_values.dtype
        ):
            return data
        if any(
            isinstance(
                value,
                (
                    bool,
                    complex,
                    date,
                    datetime,
                    timedelta,
                    np.bool_,
                    np.complexfloating,
                    np.datetime64,
                    np.timedelta64,
                ),
            )
            for value in source_values.array
        ):
            return data
        repaired[column] = pd.to_numeric(source_values, errors="coerce")

    open_column = f"{symbol}_Open"
    high_column = f"{symbol}_High"
    low_column = f"{symbol}_Low"
    close_column = f"{symbol}_Close"
    open_position = repaired.columns.get_loc(open_column)
    high_position = repaired.columns.get_loc(high_column)
    low_position = repaired.columns.get_loc(low_column)
    close_position = repaired.columns.get_loc(close_column)
    for row_position in range(len(repaired)):
        try:
            open_value = float(repaired.iat[row_position, open_position])
            high_value = float(repaired.iat[row_position, high_position])
            low_value = float(repaired.iat[row_position, low_position])
            close_value = float(repaired.iat[row_position, close_position])
        except (TypeError, ValueError, OverflowError):
            # Preserve missing or otherwise non-coercible fields for the shared
            # validator, which converts the provider failure into a normalized
            # per-symbol diagnostic and permits configured fallback handling.
            continue
        if not all(math.isfinite(value) for value in (open_value, high_value, low_value, close_value)):
            continue
        required_high = max(open_value, close_value, low_value)
        required_low = min(open_value, close_value, high_value)
        if high_value < required_high and required_high - high_value <= _yahoo_price_ordering_tolerance(
            high_value,
            required_high,
        ):
            repaired.iat[row_position, high_position] = required_high
        if low_value > required_low and low_value - required_low <= _yahoo_price_ordering_tolerance(
            low_value,
            required_low,
        ):
            repaired.iat[row_position, low_position] = required_low
    return repaired


def _yfinance_symbol_frames(
    raw: pd.DataFrame | None,
    symbols: list[str],
    download_errors: Mapping[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    errors = _yfinance_errors_for_requested_symbols(symbols, download_errors)

    if raw is None or raw.empty:
        for symbol in symbols:
            errors.setdefault(symbol, "No data returned")
        return frames, errors

    unresolved_symbols = [symbol for symbol in symbols if symbol not in errors]
    if not isinstance(raw.columns, pd.MultiIndex) and len(unresolved_symbols) > 1:
        reason = "Yahoo Finance returned an ambiguous flat-column response for a multi-symbol request"
        for symbol in unresolved_symbols:
            errors[symbol] = reason

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
            errors[symbol] = f"Yahoo Finance response was missing required OHLCV columns: {', '.join(missing_fields)}"
            continue
        duplicate_fields = [column for column in _OHLCV_FIELDS if int((df.columns == column).sum()) > 1]
        if duplicate_fields:
            errors[symbol] = (
                f"Yahoo Finance response had duplicate required OHLCV columns: {', '.join(duplicate_fields)}"
            )
            continue

        df = df[_OHLCV_FIELDS].copy()
        if df.empty:
            errors[symbol] = "Yahoo Finance response had no OHLCV daily rows"
            continue
        # A multi-symbol Yahoo download uses the union of every symbol's
        # calendar, so a symbol can have wholly empty cross-calendar rows.
        # A single-symbol history has no such justification for an empty row
        # and must fail validation. Numeric coercion also happens afterward so
        # a wholly nonnumeric candle cannot masquerade as an empty calendar row.
        if len(symbols) > 1:
            df = df.loc[~df.loc[:, _OHLCV_FIELDS].isna().all(axis=1)]
        if df.empty:
            errors[symbol] = "Yahoo Finance response had no OHLCV daily rows"
            continue
        df.columns = [f"{symbol}_{column}" for column in _OHLCV_FIELDS]
        df = _repair_yahoo_adjusted_ohlc_jitter(df, symbol)
        try:
            validate_market_data_frame(df, symbol, source="Yahoo Finance response")
        except ValueError as exc:
            errors[symbol] = str(exc)
            continue
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        frames[symbol] = df

    return frames, errors


def _load_tradier_fallback_frames(
    symbols: list[str],
    start: str | None,
    end: str | None,
    cfg: TradierMarketDataConfig,
    *,
    auto_adjust: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}

    config_error = _tradier_config_error(cfg)
    if config_error is not None:
        return frames, {symbol: config_error for symbol in symbols}
    if auto_adjust:
        reason = (
            "Tradier historical prices cannot be used when auto_adjust=True because "
            "their adjustment basis is not guaranteed to match Yahoo Finance"
        )
        return frames, {symbol: reason for symbol in symbols}

    # An omitted start means complete history to the public loader. Tradier only
    # promises lifetime coverage when callers provide a reasonable bound, so do
    # not rely on the endpoint's implementation-defined default window.
    tradier_start = start if start is not None else _TRADIER_COMPLETE_HISTORY_START
    for symbol in symbols:
        try:
            frames[symbol] = _load_tradier_symbol_frame(
                symbol,
                tradier_start,
                end,
                cfg,
                auto_adjust=auto_adjust,
            )
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


def _merged_symbol_frames(
    frames_by_symbol: Mapping[str, pd.DataFrame],
    symbols: list[str],
    *,
    calendar_symbol: str | None = None,
) -> pd.DataFrame:
    normalized_frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = frames_by_symbol[symbol].copy()
        # Providers may represent the same daily session with either a
        # timezone-aware midnight or a timezone-naive date. Pandas cannot join
        # those indexes directly, so normalize every provider frame before the
        # first merge rather than only normalizing the completed result.
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
        normalized_frames[symbol] = frame

    frames = [normalized_frames[symbol] for symbol in symbols]
    if calendar_symbol is not None:
        out = normalized_frames[calendar_symbol].copy()
        for symbol in symbols:
            if symbol != calendar_symbol:
                out = out.join(normalized_frames[symbol], how="left")
        column_order = [column for frame in frames for column in frame.columns]
        return out.loc[:, column_order].sort_index()
    return pd.concat(frames, axis=1, join="inner").dropna().sort_index()


def _symbols_absent_from_retained_calendar(
    data: pd.DataFrame,
    symbols: list[str],
) -> list[str]:
    """Return symbols whose required fields are all absent on retained rows."""
    absent: list[str] = []
    for symbol in symbols:
        columns = [f"{symbol}_{field}" for field in _OHLCV_FIELDS]
        if all(column in data.columns for column in columns) and data.loc[:, columns].isna().all(axis=0).all():
            absent.append(symbol)
    return absent


def _raise_no_overlap_error(
    symbols: list[str],
    use_tradier_fallback: bool,
    tradier_errors: Mapping[str, str],
    *,
    sensitive_values: Iterable[object] = (),
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

    raise MarketDataDownloadError(
        symbol_reasons,
        source="Market data providers",
        sensitive_values=sensitive_values,
    )


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


def signal_history_overlaps_calendar(
    *,
    calendar_symbol: str,
    calendar_history: pd.DataFrame,
    signal_symbol: str,
    signal_history: pd.DataFrame,
) -> bool:
    """Whether one signal snapshot contributes data to an asset calendar."""
    if signal_symbol == calendar_symbol:
        return True

    symbols = [calendar_symbol, signal_symbol]
    anchored = _merged_symbol_frames(
        {
            calendar_symbol: calendar_history,
            signal_symbol: signal_history,
        },
        symbols,
        calendar_symbol=calendar_symbol,
    )
    return signal_symbol not in _symbols_absent_from_retained_calendar(anchored, symbols)


def recover_signal_history_for_calendar(
    *,
    calendar_symbol: str,
    calendar_history: pd.DataFrame,
    signal_symbol: str,
    signal_history: pd.DataFrame,
    auto_adjust: bool,
    tradier_cfg: TradierMarketDataConfig | None,
) -> pd.DataFrame:
    """Validate or recover a cached signal snapshot for an asset calendar.

    Workflow caches each symbol independently, so a Yahoo response can be
    usable on its own while contributing no rows after the signal is aligned
    to an asset's authoritative sessions. Match ``load_market_data``'s
    calendar-anchored behavior without repeating either Yahoo download.
    """
    if signal_history_overlaps_calendar(
        calendar_symbol=calendar_symbol,
        calendar_history=calendar_history,
        signal_symbol=signal_symbol,
        signal_history=signal_history,
    ):
        return signal_history

    calendar_reason = f"No daily rows overlapped the retained {calendar_symbol} calendar"
    if tradier_cfg is None or not tradier_cfg.enabled:
        raise MarketDataDownloadError(
            {signal_symbol: calendar_reason},
            source="Market data providers",
        )

    tradier_frames, tradier_errors = _load_tradier_fallback_frames(
        [signal_symbol],
        None,
        None,
        tradier_cfg,
        auto_adjust=auto_adjust,
    )
    recovered = tradier_frames.get(signal_symbol)
    if recovered is not None:
        # Independently cached histories contain settled sessions only. Apply
        # that same contract before deciding whether the replacement overlaps.
        recovered = exclude_current_trading_session(recovered).copy()
        if signal_history_overlaps_calendar(
            calendar_symbol=calendar_symbol,
            calendar_history=calendar_history,
            signal_symbol=signal_symbol,
            signal_history=recovered,
        ):
            providers = dict(recovered.attrs.get(MARKET_DATA_PROVIDERS_ATTR, {}))
            providers[signal_symbol] = "tradier"
            recovered.attrs[MARKET_DATA_PROVIDERS_ATTR] = providers
            recovered.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = sorted(
                {
                    *recovered.attrs.get(TRADIER_RECOVERED_SYMBOLS_ATTR, []),
                    signal_symbol,
                }
            )
            return recovered
        tradier_errors[signal_symbol] = calendar_reason

    raise MarketDataDownloadError(
        {
            signal_symbol: _combined_provider_reason(
                calendar_reason,
                tradier_errors.get(signal_symbol),
            )
        },
        source="Yahoo Finance and Tradier",
        sensitive_values=_tradier_sensitive_values(tradier_cfg),
    )


def load_market_data(
    start: str | None = None,
    end: str | None = None,
    auto_adjust: bool = True,
    symbols: list[str] | None = None,
    tradier_cfg: TradierMarketDataConfig | None = None,
    calendar_symbol: str | None = None,
) -> pd.DataFrame:
    """
    Downloads daily data and returns a merged DataFrame with SYMBOL_Field
    columns. Yahoo Finance is the primary source; Tradier can recover symbols
    that Yahoo skips. By default, multi-symbol results use their common
    calendar. When calendar_symbol is provided, that symbol's sessions are
    retained and the other symbols are left-joined.
    """
    if type(auto_adjust) is not bool:
        raise ValueError("auto_adjust must be a boolean")
    if tradier_cfg is not None and type(tradier_cfg) is not TradierMarketDataConfig:
        raise ValueError("tradier_cfg must be a TradierMarketDataConfig instance or null")
    if symbols is None:
        raise ValueError("symbols must be provided")

    if isinstance(symbols, (str, bytes)):
        raise ValueError("symbols must be a sequence of trimmed, nonempty strings")
    try:
        requested_symbols = list(symbols)
    except TypeError as exc:
        raise ValueError("symbols must be a sequence of trimmed, nonempty strings") from exc
    invalid_symbols = [
        (index, symbol) for index, symbol in enumerate(requested_symbols) if not _is_safe_requested_symbol(symbol)
    ]
    if invalid_symbols:
        index, symbol = invalid_symbols[0]
        raise ValueError(
            "Each requested market-data symbol must be a trimmed, nonempty string that is "
            "printable and contains no whitespace or control characters; "
            f"symbols[{index}] was {symbol!r}."
        )

    # Preserve the historical harmless treatment of exact duplicate requests,
    # but reject distinct caller identities that a provider resolves to the
    # same security. Otherwise one response can be persisted twice under
    # different strategy-state symbols.
    symbols = list(dict.fromkeys(requested_symbols))
    if not symbols:
        raise ValueError("symbols must not be empty")
    yahoo_identities: dict[str, str] = {}
    for symbol in symbols:
        # yfinance canonicalizes its requested ticker set with ``str.upper()``.
        # Use that exact provider operation here rather than ``casefold()``;
        # they differ for a few Unicode spellings (for example ``i`` and the
        # dotless ``ı``), which would otherwise collapse only after preflight.
        identity = symbol.upper()
        prior = yahoo_identities.get(identity)
        if prior is not None and prior != symbol:
            raise ValueError(
                f"Requested market-data symbols collide under Yahoo Finance identity: {prior!r} and {symbol!r}."
            )
        yahoo_identities[identity] = symbol

    if tradier_cfg is not None:
        shape_error = _tradier_configuration_shape_error(tradier_cfg)
        if shape_error is not None:
            raise ValueError(shape_error)
        if tradier_cfg.enabled:
            tradier_identities: dict[str, str] = {}
            for symbol in symbols:
                identity = _tradier_symbol(symbol)
                prior = tradier_identities.get(identity)
                if prior is not None and prior != symbol:
                    raise ValueError(
                        f"Requested market-data symbols collide under Tradier identity: {prior!r} and {symbol!r}."
                    )
                tradier_identities[identity] = symbol
    sensitive_values = _tradier_sensitive_values(tradier_cfg)

    if calendar_symbol is not None and not _is_safe_requested_symbol(calendar_symbol):
        raise ValueError(
            "calendar_symbol must be a trimmed, nonempty printable string without whitespace "
            "or control characters, or null"
        )
    if calendar_symbol is not None and calendar_symbol not in symbols:
        raise ValueError("calendar_symbol must be one of the requested symbols")

    start_boundary = _history_boundary(start, label="start")
    end_boundary = _history_boundary(end, label="end")
    if start_boundary is not None and end_boundary is not None and end_boundary <= start_boundary:
        raise ValueError("Market-data end must be strictly after start.")

    raw, download_errors = _download_yfinance(symbols, start, end, auto_adjust)
    frames_by_symbol, yahoo_errors = _yfinance_symbol_frames(raw, symbols, download_errors)
    for symbol, frame in list(frames_by_symbol.items()):
        clipped = _clip_history_frame_to_requested_bounds(frame, start=start, end=end)
        if clipped.empty:
            del frames_by_symbol[symbol]
            yahoo_errors[symbol] = "Yahoo Finance response had no daily rows inside the requested date range"
        else:
            frames_by_symbol[symbol] = clipped

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
            auto_adjust=auto_adjust,
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
            symbol_reasons = {symbol: yahoo_errors.get(symbol, "No data returned") for symbol in unresolved_symbols}
            source = "Yahoo Finance"
        raise MarketDataDownloadError(
            symbol_reasons,
            source=source,
            sensitive_values=sensitive_values,
        )

    out = _merged_symbol_frames(
        frames_by_symbol,
        symbols,
        calendar_symbol=calendar_symbol,
    )
    absent_from_calendar = _symbols_absent_from_retained_calendar(out, symbols) if calendar_symbol is not None else []
    if absent_from_calendar and use_tradier_fallback:
        # A symbol already loaded from Tradier has had its recovery attempt; if
        # it is still absent from the anchor calendar, retrying the same
        # provider cannot make that frame a usable recovery.  For Yahoo-backed
        # symbols, defer provider attribution until the replacement actually
        # contributes OHLCV data to at least one retained session.
        calendar_fallback_symbols = [symbol for symbol in absent_from_calendar if symbol not in recovered_symbols]
        tradier_frames: dict[str, pd.DataFrame] = {}
        calendar_tradier_errors: dict[str, str] = {}
        if calendar_fallback_symbols:
            tradier_frames, calendar_tradier_errors = _load_tradier_fallback_frames(
                calendar_fallback_symbols,
                start,
                end,
                tradier_cfg,
                auto_adjust=auto_adjust,
            )
        tradier_errors.update(calendar_tradier_errors)
        for symbol, frame in tradier_frames.items():
            frames_by_symbol[symbol] = frame
        if tradier_frames:
            out = _merged_symbol_frames(
                frames_by_symbol,
                symbols,
                calendar_symbol=calendar_symbol,
            )
        absent_from_calendar = _symbols_absent_from_retained_calendar(out, symbols)
        for symbol in tradier_frames:
            if symbol in absent_from_calendar:
                tradier_errors[symbol] = f"No daily rows overlapped the retained {calendar_symbol} calendar"
            elif symbol not in recovered_symbols:
                recovered_symbols.append(symbol)

    if absent_from_calendar:
        calendar_reason = f"No daily rows overlapped the retained {calendar_symbol} calendar"
        symbol_reasons = {}
        for symbol in absent_from_calendar:
            if use_tradier_fallback:
                tradier_reason = tradier_errors.get(symbol)
                if symbol in frames_by_symbol and symbol in recovered_symbols:
                    tradier_reason = calendar_reason
                symbol_reasons[symbol] = _combined_provider_reason(
                    yahoo_errors.get(symbol, calendar_reason),
                    tradier_reason,
                )
            else:
                symbol_reasons[symbol] = calendar_reason
        raise MarketDataDownloadError(
            symbol_reasons,
            source="Yahoo Finance and Tradier" if use_tradier_fallback else "Market data providers",
            sensitive_values=sensitive_values,
        )

    if out.empty and use_tradier_fallback:
        tradier_frames, no_overlap_tradier_errors = _load_tradier_fallback_frames(
            symbols,
            start,
            end,
            tradier_cfg,
            auto_adjust=auto_adjust,
        )
        tradier_errors.update(no_overlap_tradier_errors)
        for symbol, frame in tradier_frames.items():
            frames_by_symbol[symbol] = frame
            if symbol not in recovered_symbols:
                recovered_symbols.append(symbol)
        if tradier_frames:
            out = _merged_symbol_frames(
                frames_by_symbol,
                symbols,
                calendar_symbol=calendar_symbol,
            )

    if out.empty:
        _raise_no_overlap_error(
            symbols,
            use_tradier_fallback,
            tradier_errors,
            sensitive_values=sensitive_values,
        )

    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.attrs[MARKET_DATA_PROVIDERS_ATTR] = {
        symbol: "tradier" if symbol in recovered_symbols else "yahoo_finance" for symbol in symbols
    }
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
        calendar_symbol=asset_symbol,
    )
    risk_free = load_market_data(
        start=_date_str(data.index.min()),
        end=end,
        auto_adjust=auto_adjust,
        symbols=[RISK_FREE_SYMBOL],
        tradier_cfg=tradier_cfg,
    )
    out = data.join(risk_free, how="left")
    out[risk_free.columns] = out[risk_free.columns].ffill()
    recovered_symbols = sorted(
        {
            *data.attrs.get(TRADIER_RECOVERED_SYMBOLS_ATTR, []),
            *risk_free.attrs.get(TRADIER_RECOVERED_SYMBOLS_ATTR, []),
        }
    )
    out.attrs[MARKET_DATA_PROVIDERS_ATTR] = {
        **data.attrs.get(MARKET_DATA_PROVIDERS_ATTR, {}),
        **risk_free.attrs.get(MARKET_DATA_PROVIDERS_ATTR, {}),
    }
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
