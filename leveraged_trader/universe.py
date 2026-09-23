from __future__ import annotations

import codecs
import csv
import io
import ipaddress
import json
import math
import os
import re
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import unescape
from unittest.mock import Mock
from urllib.parse import urljoin, urlsplit
from zoneinfo import ZoneInfo

import idna
import pandas as pd
import requests
from lxml import html as lxml_html

from ._http_deadline_worker import response_from_http_worker_payload, run_http_request_with_deadline
from .config import ETF_DEFS_URL, UniverseConfig
from .output import safe_diagnostic_text
from .storage import save_table_to_sqlite

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_MUTUAL_FUND_TICKERS_URL = "https://www.sec.gov/files/company_tickers_mf.json"
DEFAULT_USER_AGENT = "leveraged-trader/0.1 (+https://github.com/maks-emelyanov/leveraged-trader)"
SEC_USER_AGENT_CONFIGURATION_ERROR = (
    "SEC audit request skipped: set SEC_USER_AGENT to a truthful application/operator "
    "identity and monitored contact email address."
)
SEC_CONTACT_EMAIL_PATTERN = re.compile(
    r"(?:^|(?<=[\s<(\[{])|(?<=mailto:))"
    r"(?P<local>[\w.!#$%&'*+/=?^`{|}~-]+)@"
    r"(?P<domain>[\w.-]+\.(?:[A-Z]{2,63}|XN--[A-Z0-9](?:[A-Z0-9-]{0,57}[A-Z0-9])?))"
    r"(?=$|[\s>)\]},;:]|\.(?![\w.-]))",
    re.I,
)
SEC_PLACEHOLDER_IDENTITY_PATTERN = re.compile(
    r"\b(?:your|example|sample|placeholder)[\s_-]*"
    r"(?:name|company|organization|operator|application|app|project|email|contact)\b"
    r"|\b(?:name|company|organization|operator|application|app|project)"
    r"[\s_-]*(?:here|name|placeholder)\b",
    re.I,
)
SEC_NONPUBLIC_HOST_NAMES = frozenset(
    {
        "alt",
        "arpa",
        "corp",
        "example",
        "example.com",
        "example.net",
        "example.org",
        "home",
        "internal",
        "invalid",
        "local",
        "localhost",
        "mail",
        "onion",
        "test",
    }
)
SEC_IDENTITY_URL_PATTERN = re.compile(r"https?://[^\s<>()]+", re.I)
SEC_IDENTITY_PROTOCOL_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Z0-9+.-]*://|(?:https?|ftp):/{0,2})",
    re.I,
)
# Preserve malformed trailing separators and non-ASCII code points while
# identifying dotted, schemeless host candidates.  The ordinary ASCII identity
# tokenizer below is intentionally narrower and would otherwise turn
# ``project.xn--\u00e9`` into the apparently valid prefix ``project.xn``.
SEC_SCHEMELESS_HOST_PATTERN = re.compile(
    r"(?<![\w.-])[\w-]+(?:[.-]+[\w-]*)+(?![\w.-])",
)
SEC_USER_AGENT_MAX_CHARS = 2_048
SEC_GENERIC_IDENTITY_PARTS = frozenset(
    {
        "admin",
        "app",
        "application",
        "company",
        "contact",
        "com",
        "email",
        "example",
        "ftp",
        "http",
        "https",
        "mail",
        "mailto",
        "maintainer",
        "name",
        "net",
        "none",
        "operator",
        "ops",
        "org",
        "organization",
        "owner",
        "placeholder",
        "project",
        "sample",
        "support",
        "team",
        "tbd",
        "test",
        "to",
        "unknown",
        "user",
        "version",
        "www",
        "your",
    }
)
REQUEST_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,text/csv,application/json;q=0.8,*/*;q=0.5",
    "Accept-Encoding": "identity",
}
UNIVERSE_FETCH_MAX_WORKERS = 8
UNIVERSE_REDIRECT_LIMIT = 5
UNIVERSE_RESPONSE_MAX_BYTES = 8 * 1024 * 1024
_UNIVERSE_RESPONSE_CHUNK_BYTES = 64 * 1024
_UNIVERSE_RESPONSE_READ_TIMEOUT_SECONDS = 5.0
_UNIVERSE_WORKER_RESULT_MAX_BYTES = UNIVERSE_RESPONSE_MAX_BYTES + 8 * 1024 * 1024
_TRUSTED_UNIVERSE_RESPONSE_ENCODINGS = frozenset(
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
NASDAQ_ETF_SOURCE_NAME = "Nasdaq ETF definitions"
NASDAQ_ETF_SOURCE_TYPE = "primary_etf"
_NEW_YORK = ZoneInfo("America/New_York")
_NASDAQ_ACTIVE_LISTING_MAX_AGE = timedelta(days=7)
_NASDAQ_ACTIVE_LISTING_MAX_FUTURE_SKEW = timedelta(hours=6)
# The directory loaders independently require exact schemas, fresh footers, and
# production-sized files. A tiny cross-source tolerance therefore represents
# ordinary delisting lag; requiring both a percentage floor and an absolute cap
# keeps a broader symbol mismatch from becoming destructively authoritative.
_NASDAQ_ACTIVE_PRIMARY_COVERAGE_MIN = 0.995
_NASDAQ_ACTIVE_PRIMARY_MAX_MISSING = 5
_NASDAQ_ACTIVE_PRIMARY_COVERAGE_MIN_ROWS = 100
_UNIVERSE_DIAGNOSTIC_MAX_CHARS = 250
_MAX_JSON_NESTING_DEPTH = 128
_MAX_JSON_STRUCTURAL_TOKENS = 1_000_000
_MAX_JS_QUOTED_TEXT_CHARS = 1_000_000
_MIN_UNIVERSE_REQUEST_TIMEOUT_SECONDS = 1
_MAX_UNIVERSE_REQUEST_TIMEOUT_SECONDS = 600
# The production page currently contains roughly 1,250 distinct product rows,
# 1,700 rows when repeated liquidity-provider assignments are included, and
# 1,200 usable ETF symbols. These deliberately conservative floors leave room
# for ordinary source churn and harmless consolidation of provider assignments
# while preventing a partial HTML response from becoming authoritative.
_NASDAQ_ETF_MINIMUM_RAW_ROWS = 1_200
_NASDAQ_ETF_MINIMUM_USABLE_ROWS = 1_000
_NASDAQ_ETF_SCHEMA_CANDIDATE_MIN_ROWS = 100
_LEVERAGE_SHARES_MINIMUM_PRODUCTS = 100
_YIELDMAX_MINIMUM_PRODUCTS = 50
_CLASSIC_JAVASCRIPT_MIME_TYPES = frozenset(
    {
        "application/ecmascript",
        "application/javascript",
        "application/x-ecmascript",
        "application/x-javascript",
        "text/ecmascript",
        "text/javascript",
        "text/javascript1.0",
        "text/javascript1.1",
        "text/javascript1.2",
        "text/javascript1.3",
        "text/javascript1.4",
        "text/javascript1.5",
        "text/jscript",
        "text/livescript",
        "text/x-ecmascript",
        "text/x-javascript",
    }
)
_NASDAQ_ACTIVE_LISTING_SCHEMAS = {
    NASDAQ_LISTED_URL: (
        (
            "Symbol",
            "Security Name",
            "Market Category",
            "Test Issue",
            "Financial Status",
            "Round Lot Size",
            "ETF",
            "NextShares",
        ),
        7,
        5_400,
    ),
    OTHER_LISTED_URL: (
        (
            "ACT Symbol",
            "Security Name",
            "Exchange",
            "CQS Symbol",
            "ETF",
            "Round Lot Size",
            "Test Issue",
            "NASDAQ Symbol",
        ),
        6,
        7_000,
    ),
}


def _nasdaq_directory_now() -> datetime:
    return datetime.now(_NEW_YORK)


def _is_sec_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if "\\" in parsed.netloc or parsed.username is not None or parsed.password is not None:
        return False
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    return hostname == "sec.gov" or hostname.endswith(".sec.gov")


def _validated_prepared_universe_url(url: str, *, redirect_target: bool = False) -> str:
    """Return Requests' canonical destination after strict authority validation."""
    label = "Universe source redirect target" if redirect_target else "Universe source URL"
    if not isinstance(url, str):
        raise requests.exceptions.InvalidURL(f"{label} has an invalid or ambiguous authority.")
    try:
        parsed = urlsplit(url)
    except ValueError:
        raise requests.exceptions.InvalidURL(f"{label} has an invalid or ambiguous authority.") from None
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        if redirect_target:
            raise requests.exceptions.InvalidURL("Universe source redirect target must use HTTP or HTTPS.")
        raise requests.exceptions.InvalidURL(
            f"Universe source URL must use HTTP or HTTPS, not {scheme or 'an empty scheme'}."
        )
    if "\\" in parsed.netloc:
        raise requests.exceptions.InvalidURL(f"{label} has an invalid or ambiguous authority.")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise requests.exceptions.InvalidURL(f"{label} must not contain URL userinfo.")
    try:
        port = parsed.port
    except ValueError:
        raise requests.exceptions.InvalidURL(f"{label} has an invalid or ambiguous authority.") from None
    if not parsed.netloc or parsed.hostname is None or (port is not None and not 1 <= port <= 65_535):
        raise requests.exceptions.InvalidURL(f"{label} has an invalid or ambiguous authority.")

    try:
        prepared_url = requests.Request("GET", url).prepare().url
    except (requests.RequestException, UnicodeError, ValueError):
        raise requests.exceptions.InvalidURL(f"{label} has an invalid or ambiguous authority.") from None
    if not isinstance(prepared_url, str):
        raise requests.exceptions.InvalidURL(f"{label} has an invalid or ambiguous authority.")
    try:
        prepared = urlsplit(prepared_url)
    except ValueError:
        raise requests.exceptions.InvalidURL(f"{label} has an invalid or ambiguous authority.") from None
    if (
        prepared.scheme.casefold() not in {"http", "https"}
        or not prepared.netloc
        or prepared.hostname is None
        or "\\" in prepared.netloc
        or prepared.username is not None
        or prepared.password is not None
        or "@" in prepared.netloc
    ):
        raise requests.exceptions.InvalidURL(f"{label} has an invalid or ambiguous authority.")
    return prepared_url


def _normalized_universe_origin(url: str) -> tuple[str, str, int]:
    """Return a canonical scheme, host, and effective port for a validated URL."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    port = parsed.port or (443 if scheme == "https" else 80)
    return scheme, hostname, port


def _sec_dns_label_is_valid(label: str) -> bool:
    """Validate one public-host label, including canonical IDNA A-labels."""
    if not 1 <= len(label) <= 63 or re.fullmatch(r"[A-Z0-9](?:[A-Z0-9-]*[A-Z0-9])?", label, re.I) is None:
        return False
    ascii_label = label.casefold()
    if not ascii_label.startswith("xn--"):
        return True
    try:
        decoded = idna.decode(ascii_label, strict=True)
        round_tripped = idna.encode(decoded, strict=True).decode("ascii").casefold()
    except UnicodeError:
        return False
    return round_tripped == ascii_label


def _sec_contact_email_is_valid(match: re.Match[str]) -> bool:
    """Return whether one detected contact has a usable dot-atom/DNS shape."""
    local = match.group("local")
    if len(local) > 64 or local.startswith(".") or local.endswith(".") or ".." in local:
        return False

    domain = match.group("domain")
    if len(domain) > 253:
        return False
    return all(_sec_dns_label_is_valid(label) for label in domain.split("."))


def _sec_identity_part_is_generic(part: str) -> bool:
    """Recognize generic labels with ordinary plural and version suffixes."""
    stem = re.sub(r"(?:v?\d+(?:[a-z]+\d*)*)\Z", "", part.casefold())
    if not stem:
        return True
    candidates = {stem}
    if stem.endswith("ies") and len(stem) > 3:
        candidates.add(f"{stem[:-3]}y")
    if stem.endswith("es") and len(stem) > 2:
        candidates.add(stem[:-2])
    if stem.endswith("s") and len(stem) > 1:
        candidates.add(stem[:-1])
    return not SEC_GENERIC_IDENTITY_PARTS.isdisjoint(candidates)


def _sec_host_is_nonpublic(value: str) -> bool:
    """Reject placeholder and special-use DNS namespaces as public contacts."""
    hostname = value.casefold().rstrip(".")
    return any(hostname == reserved or hostname.endswith(f".{reserved}") for reserved in SEC_NONPUBLIC_HOST_NAMES)


def _sec_host_is_numeric_address_candidate(value: str) -> bool:
    """Recognize numeric DNS spellings that could be mistaken for IPv4."""
    components = value.split(".")
    return bool(components) and all(
        re.fullmatch(r"(?:0X[0-9A-F]+|[0-9]+)", component, re.I | re.ASCII) is not None for component in components
    )


def _sec_identity_token_is_meaningful(token: str) -> bool:
    """Reject generic contact labels and version syntax as the only identity."""
    if sum(character.isalnum() for character in token) < 3 or not any(character.isalpha() for character in token):
        return False
    parts = re.findall(r"[A-Z0-9]+", token, re.I)
    return any(
        any(character.isalpha() for character in part) and not _sec_identity_part_is_generic(part) for part in parts
    )


def _sec_identity_url_is_meaningful(value: str) -> bool:
    """Accept a bounded public project URL, never protocol syntax or a placeholder host."""
    if len(value) > SEC_USER_AGENT_MAX_CHARS or "\\" in value:
        return False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and not 1 <= port <= 65_535)
        or len(hostname) > 253
    ):
        return False
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        if _sec_host_is_numeric_address_candidate(hostname):
            return False
    else:
        return False
    labels = hostname.split(".")
    if len(labels) < 2 or not all(_sec_dns_label_is_valid(label) for label in labels):
        return False
    return not _sec_host_is_nonpublic(hostname)


def _sec_identity_token_is_reserved_host(token: str) -> bool:
    """Reject schemeless invalid or placeholder hosts without parsing prose as URLs."""
    hostname = token.casefold().rstrip(".")
    if hostname == "localhost":
        return True
    if "." not in hostname:
        return False
    return _sec_host_is_nonpublic(hostname) or not all(_sec_dns_label_is_valid(label) for label in hostname.split("."))


def _configured_sec_user_agent() -> str | None:
    """Return an explicit, non-placeholder SEC identity with a contact email."""
    # Normalize ordinary syntax-level padding, but do not erase tabs/newlines:
    # they must be rejected as header controls below.
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip(" ")
    # This value is copied verbatim into an HTTP header.  Requests may reject
    # some control characters itself, but that is too late for a configuration
    # boundary and other clients can serialize them differently.  Ordinary
    # spaces remain necessary between the application, operator, and contact.
    if len(user_agent) > SEC_USER_AGENT_MAX_CHARS or any(not character.isprintable() for character in user_agent):
        return None
    try:
        user_agent.encode("latin-1")
    except UnicodeEncodeError:
        return None
    email_matches = list(SEC_CONTACT_EMAIL_PATTERN.finditer(user_agent))
    matched_at_signs = {match.start("local") + len(match.group("local")) for match in email_matches}
    if (
        not user_agent
        or not email_matches
        or any(index not in matched_at_signs for index, character in enumerate(user_agent) if character == "@")
        or any(not _sec_contact_email_is_valid(match) for match in email_matches)
    ):
        return None

    for match in email_matches:
        domain = match.group("domain").casefold().rstrip(".")
        if _sec_host_is_nonpublic(domain):
            return None

    # An email address alone is contact information, not the application or
    # operator identity requested by the SEC.  Require some alphabetic identity
    # text outside every address, while allowing either a person/project name or
    # an identifying project URL.
    identity = SEC_CONTACT_EMAIL_PATTERN.sub(" ", user_agent)
    identity_urls = SEC_IDENTITY_URL_PATTERN.findall(identity)
    if any(not _sec_identity_url_is_meaningful(url) for url in identity_urls):
        return None
    identity = SEC_IDENTITY_URL_PATTERN.sub(" ", identity)
    if SEC_PLACEHOLDER_IDENTITY_PATTERN.search(identity):
        return None
    # A protocol fragment is not an application identity and indicates a
    # malformed project URL rather than ordinary name punctuation.
    if SEC_IDENTITY_PROTOCOL_PATTERN.search(identity):
        return None
    schemeless_hosts = SEC_SCHEMELESS_HOST_PATTERN.findall(identity)
    if any(_sec_identity_token_is_reserved_host(token) for token in schemeless_hosts):
        return None
    identity_tokens = re.findall(r"[A-Z0-9]+(?:[._-]+[A-Z0-9]+)*", identity, re.I)
    if any(_sec_identity_token_is_reserved_host(token) for token in identity_tokens):
        return None
    if not identity_urls and not any(_sec_identity_token_is_meaningful(token) for token in identity_tokens):
        return None
    return user_agent


def _request_headers(url: str) -> dict[str, str]:
    """Use operator contact details only for requests whose destination is SEC-owned."""
    user_agent = DEFAULT_USER_AGENT
    if _is_sec_url(url):
        configured_user_agent = _configured_sec_user_agent()
        if configured_user_agent is None:
            raise ValueError(SEC_USER_AGENT_CONFIGURATION_ERROR)
        user_agent = configured_user_agent
    return {**REQUEST_HEADERS, "User-Agent": user_agent}


def _universe_response_content_length(response: requests.Response) -> int | None:
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    raw_length = headers.get("Content-Length")
    if raw_length is None:
        return None
    if not isinstance(raw_length, str) or re.fullmatch(r"\d+", raw_length.strip()) is None:
        raise requests.exceptions.InvalidHeader("Universe source returned an invalid Content-Length header.")
    return int(raw_length.strip())


def _universe_response_has_header(response: requests.Response, name: str) -> bool:
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return False
    normalized_name = name.casefold()
    return any(isinstance(header_name, str) and header_name.casefold() == normalized_name for header_name in headers)


def _universe_response_chunks(response: requests.Response) -> Iterable[bytes]:
    headers = response.headers if isinstance(response.headers, Mapping) else {}
    content_encoding = headers.get("Content-Encoding")
    if content_encoding is not None and (
        not isinstance(content_encoding, str) or content_encoding.strip().casefold() not in {"", "identity"}
    ):
        raise requests.exceptions.ContentDecodingError(
            "Universe source returned compressed content despite an identity-only request."
        )
    if isinstance(response, requests.Response):
        materialized_content = getattr(response, "_content", None)
        if getattr(response, "_content_consumed", False) and isinstance(materialized_content, bytes):
            yield materialized_content
            return
        raw_response = response.raw
        read_available = getattr(raw_response, "read1", None)
        if not callable(read_available):
            raise TypeError("Universe HTTP response does not support bounded incremental reads.")
        while chunk := read_available(
            _UNIVERSE_RESPONSE_CHUNK_BYTES,
            decode_content=False,
        ):
            yield chunk
        return

    # Lightweight response adapters must expose already-materialized concrete
    # text. The production requests client always takes the streaming branch.
    adapter_text = getattr(response, "text", None)
    if not isinstance(adapter_text, str):
        raise TypeError("Universe HTTP client returned an invalid response object.")
    if len(adapter_text) > UNIVERSE_RESPONSE_MAX_BYTES:
        raise ValueError(f"Universe source response exceeded the {UNIVERSE_RESPONSE_MAX_BYTES}-byte limit.")
    encoded_text = adapter_text.encode("utf-8")
    if len(encoded_text) > UNIVERSE_RESPONSE_MAX_BYTES:
        raise ValueError(f"Universe source response exceeded the {UNIVERSE_RESPONSE_MAX_BYTES}-byte limit.")
    yield encoded_text


def _trusted_universe_response_encoding(response: requests.Response, body: bytes) -> str:
    raw_encoding = getattr(response, "encoding", None)
    encoding = raw_encoding.strip() if isinstance(raw_encoding, str) and raw_encoding.strip() else "utf-8"
    try:
        canonical_encoding = codecs.lookup(encoding).name
    except LookupError as exc:
        raise requests.exceptions.InvalidHeader("Universe source declared an unsupported character encoding.") from exc
    if canonical_encoding not in _TRUSTED_UNIVERSE_RESPONSE_ENCODINGS:
        raise requests.exceptions.InvalidHeader("Universe source declared an unsupported character encoding.")
    if canonical_encoding == "utf-8" and body.startswith(codecs.BOM_UTF8):
        canonical_encoding = "utf-8-sig"
    try:
        body.decode(canonical_encoding, errors="strict")
    except UnicodeError as exc:
        raise requests.exceptions.ContentDecodingError(
            "Universe source response was not valid in its declared character encoding."
        ) from exc
    return canonical_encoding


def _materialize_universe_response(
    response: requests.Response,
    *,
    deadline: float,
) -> tuple[bytes, str]:
    content_length = _universe_response_content_length(response)
    if content_length is not None and content_length > UNIVERSE_RESPONSE_MAX_BYTES:
        raise ValueError(f"Universe source response exceeded the {UNIVERSE_RESPONSE_MAX_BYTES}-byte limit.")

    body = bytearray()
    chunks = iter(_universe_response_chunks(response))
    while True:
        if time.monotonic() >= deadline:
            raise requests.exceptions.Timeout("Universe source exceeded its response deadline.")
        try:
            chunk = next(chunks)
        except StopIteration:
            if time.monotonic() >= deadline:
                raise requests.exceptions.Timeout("Universe source exceeded its response deadline.") from None
            break
        if time.monotonic() >= deadline:
            raise requests.exceptions.Timeout("Universe source exceeded its response deadline.")
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError("Universe source returned a non-byte response chunk.")
        if len(body) + len(chunk) > UNIVERSE_RESPONSE_MAX_BYTES:
            raise ValueError(f"Universe source response exceeded the {UNIVERSE_RESPONSE_MAX_BYTES}-byte limit.")
        body.extend(chunk)

    materialized = bytes(body)
    return materialized, _trusted_universe_response_encoding(response, materialized)


def _universe_request_exception_from_process(name: str, message: str) -> BaseException:
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


def _get_universe_response_with_deadline(
    url: str,
    *,
    deadline: float,
    request_kwargs: dict[str, object],
) -> requests.Response:
    """Acquire one hop's headers and body under the shared absolute deadline."""
    # Unit-test response doubles may not be pickleable. Restrict this
    # compatibility path to explicit unittest.mock objects so an ordinary
    # production wrapper cannot silently disable the hard deadline.
    if isinstance(requests.get, Mock):
        return requests.get(url, **request_kwargs)
    payload = run_http_request_with_deadline(
        url,
        request_kwargs,
        mode="universe",
        deadline=deadline,
        max_body_bytes=UNIVERSE_RESPONSE_MAX_BYTES,
        result_max_bytes=_UNIVERSE_WORKER_RESULT_MAX_BYTES,
        timeout_message="Universe source exceeded its response deadline.",
        connection_error_message="Universe request worker exited without a complete response.",
    )

    return response_from_http_worker_payload(
        payload,
        max_body_bytes=UNIVERSE_RESPONSE_MAX_BYTES,
        invalid_response_message="Universe request worker returned an invalid response.",
        exception_from_process=_universe_request_exception_from_process,
    )


def _get_universe_response(url: str, timeout: int) -> requests.Response:
    """Return one bounded, decoded, closed response after safe redirects."""
    current_url = _validated_prepared_universe_url(url)
    deadline = time.monotonic() + timeout
    for redirect_count in range(UNIVERSE_REDIRECT_LIMIT + 1):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise requests.exceptions.Timeout("Universe source exceeded its response deadline.")
        # Requests exposes connect/read-idle timeouts, not an interruptible hard
        # wall-clock deadline. A short read timeout bounds ordinary socket stalls;
        # raw.read1 returns available body data without waiting to fill our chunk,
        # which lets the monotonic checks above run between incremental reads.
        request_kwargs: dict[str, object] = {
            "timeout": (
                remaining_seconds,
                min(remaining_seconds, _UNIVERSE_RESPONSE_READ_TIMEOUT_SECONDS),
            ),
            "headers": _request_headers(current_url),
            "allow_redirects": False,
            "stream": True,
        }
        response = _get_universe_response_with_deadline(
            current_url,
            deadline=deadline,
            request_kwargs=request_kwargs,
        )
        primary_failure: BaseException | None = None
        try:
            status_code = response.status_code
            if isinstance(status_code, int) and status_code in {301, 302, 303, 307, 308}:
                if redirect_count == UNIVERSE_REDIRECT_LIMIT:
                    raise requests.exceptions.TooManyRedirects(
                        f"Universe source exceeded {UNIVERSE_REDIRECT_LIMIT} redirects."
                    )
                location = response.headers.get("Location")
                if not isinstance(location, str) or not location.strip():
                    raise requests.exceptions.InvalidHeader("Universe source redirect omitted the Location header.")
                redirected_url = _validated_prepared_universe_url(
                    urljoin(current_url, location),
                    redirect_target=True,
                )
                current_scheme = urlsplit(current_url).scheme.lower()
                redirected_scheme = urlsplit(redirected_url).scheme.lower()
                if current_scheme == "https" and redirected_scheme != "https":
                    raise requests.exceptions.SSLError("Universe source refused an HTTPS-to-HTTP redirect downgrade.")
                if _normalized_universe_origin(redirected_url) != _normalized_universe_origin(current_url):
                    raise requests.exceptions.InvalidURL("Universe source refused a cross-origin redirect.")
                current_url = redirected_url
                continue

            if (
                type(status_code) is int
                and status_code == 200
                and _universe_response_has_header(response, "Content-Range")
            ):
                raise requests.exceptions.InvalidHeader(
                    "Universe source returned an unsolicited Content-Range header for a complete snapshot."
                )
            if type(status_code) is not int or status_code != 200:
                # ``Response.raise_for_status`` deliberately treats every 3xx
                # and 2xx response as successful. Redirect handling above is
                # the only supported non-200 path; a complete universe
                # snapshot must be an exact 200 before its body can become
                # authoritative input.
                if isinstance(status_code, int) and status_code >= 400:
                    response.raise_for_status()
                error = requests.exceptions.HTTPError(f"Universe source returned unexpected HTTP status {status_code}.")
                error.response = response
                raise error

            body, encoding = _materialize_universe_response(response, deadline=deadline)
            if isinstance(response, requests.Response):
                response._content = body
            else:
                response.text = body.decode(encoding, errors="strict")
            response.encoding = encoding
            return response
        except BaseException as exc:
            primary_failure = exc
            raise
        finally:
            try:
                response.close()
            except BaseException as close_failure:
                if primary_failure is None:
                    raise
                close_detail = safe_diagnostic_text(
                    close_failure,
                    max_chars=_UNIVERSE_DIAGNOSTIC_MAX_CHARS,
                )
                primary_failure.add_note(f"Failed to close the universe HTTP response: {close_detail}")
    raise AssertionError("unreachable")


def _validated_universe_request_timeout(timeout: object) -> int:
    """Return a timeout that is safe to use as a public loader deadline."""
    if (
        type(timeout) is not int
        or not _MIN_UNIVERSE_REQUEST_TIMEOUT_SECONDS <= timeout <= _MAX_UNIVERSE_REQUEST_TIMEOUT_SECONDS
    ):
        raise ValueError(
            "Universe timeout must be an integer between "
            f"{_MIN_UNIVERSE_REQUEST_TIMEOUT_SECONDS} and "
            f"{_MAX_UNIVERSE_REQUEST_TIMEOUT_SECONDS} seconds."
        )
    return timeout


@dataclass(frozen=True)
class UniverseSource:
    name: str
    url: str
    source_type: str
    parser: str = "html"
    enabled: bool = True
    notes: str = ""


@dataclass(frozen=True)
class _SourceFetchResult:
    text: str | None = None
    error: str = ""


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
        return bool(self.source_status) and all(status.get("status") == "loaded" for status in self.source_status)


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
        "error": safe_diagnostic_text(error, max_chars=_UNIVERSE_DIAGNOSTIC_MAX_CHARS),
    }


def _workflow_source_status(df: pd.DataFrame) -> pd.DataFrame:
    status = pd.DataFrame(
        df.attrs.get("workflow_source_status", []),
        columns=WORKFLOW_SOURCE_STATUS_COLUMNS,
    )
    if "error" in status.columns:
        status["error"] = status["error"].map(
            lambda value: (
                safe_diagnostic_text(value, max_chars=_UNIVERSE_DIAGNOSTIC_MAX_CHARS) if isinstance(value, str) else ""
            )
        )
    return status


def _universe_exception_diagnostic(exc: BaseException) -> str:
    """Describe a source failure without invoking arbitrary rendering hooks."""
    try:
        raw_type_name = type.__getattribute__(type(exc), "__name__")
    except BaseException:
        raw_type_name = "Exception"
    type_name = safe_diagnostic_text(
        raw_type_name,
        max_chars=min(64, _UNIVERSE_DIAGNOSTIC_MAX_CHARS),
    )
    type_name = type_name or "Exception"
    prefix = f"{type_name}: "
    detail = safe_diagnostic_text(
        exc,
        max_chars=max(0, _UNIVERSE_DIAGNOSTIC_MAX_CHARS - len(prefix)),
    )
    return f"{prefix}{detail}"[:_UNIVERSE_DIAGNOSTIC_MAX_CHARS]


def _workflow_source_parse_status(parsed_rows: pd.DataFrame, matched_rows: pd.DataFrame) -> tuple[str, str]:
    """Classify a fetched source without mistaking zero matches for parser failure."""
    if parsed_rows.empty:
        return "parse_error", "No product rows could be parsed from a successful source response."
    if matched_rows.empty:
        return "loaded_zero_matches", ""
    return "loaded", ""


def _fetch_source_text(source: UniverseSource, timeout: int) -> _SourceFetchResult:
    try:
        response = _get_universe_response(source.url, timeout)
        response.raise_for_status()
    except Exception as exc:
        return _SourceFetchResult(error=_universe_exception_diagnostic(exc))
    return _SourceFetchResult(text=response.text)


def _fetch_enabled_sources(
    sources: list[UniverseSource],
    timeout: int,
) -> list[_SourceFetchResult | None]:
    """Fetch enabled independent sources concurrently, preserving source order."""
    results: list[_SourceFetchResult | None] = [None] * len(sources)
    fetches = [
        (index, source)
        for index, source in enumerate(sources)
        if source.enabled
        and source.parser != "registered_only"
        and (not _is_sec_url(source.url) or _configured_sec_user_agent() is not None)
    ]
    if not fetches:
        return results

    worker_count = min(UNIVERSE_FETCH_MAX_WORKERS, len(fetches))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="universe-fetch") as executor:
        futures = [executor.submit(_fetch_source_text, source, timeout) for _index, source in fetches]
        for (index, _source), future in zip(fetches, futures, strict=True):
            results[index] = future.result()
    return results


WORKFLOW_ISSUER_SOURCES = [
    UniverseSource("ProShares", "https://www.proshares.com/our-etfs/find-leveraged-and-inverse-etfs", "issuer_etf"),
    UniverseSource(
        "Direxion",
        "https://www.direxion.com/all-etfs",
        "issuer_etf",
        enabled=False,
        notes="Issuer page blocks unattended fetches; Nasdaq ETF definitions remain authoritative.",
    ),
    UniverseSource(
        "Leverage Shares",
        "https://leverageshares.com/us/all-etfs/",
        "issuer_etf",
        parser="leverage_shares_html",
    ),
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
    UniverseSource(
        "Innovator",
        "https://www.innovatoretfs.com/define/etfs/",
        "issuer_etf",
        parser="innovator_html",
    ),
    UniverseSource("Tuttle Capital", "https://www.tuttlecap.com/etfs", "issuer_etf"),
    UniverseSource("Tradr", "https://www.tradretfs.com/", "issuer_etf", parser="tradr_html"),
    UniverseSource(
        "REX Shares",
        "https://www.cboe.com/us/equities/listings/listed_products/issuer_detail/TRXE/",
        "issuer_etf",
        parser="cboe_issuer_html",
        notes=("Official Cboe issuer listing is used because the REX website blocks unattended requests."),
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
    UniverseSource(
        "YieldMax",
        "https://yieldmaxetfs.com/",
        "issuer_etf",
        parser="yieldmax_html",
    ),
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
        parser="registered_only",
        enabled=False,
        notes=(
            "Registered as an audit backstop; listings are populated client-side and the public page "
            "does not expose product rows to the static HTML parser."
        ),
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

AUDIT_INVENTORY_ONLY_PARSERS = frozenset(
    {
        "cboe_symbol_csv",
        "sec_mutual_fund_tickers",
    }
)


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

# Nasdaq's symbol-directory files use their own source notation.  In
# particular, otherlisted.txt represents preferred shares with ``$`` and
# units/warrants or classes with a dot suffix.  A source symbol has at most one
# such suffix: dot suffixes are non-empty, while Nasdaq documents a bare
# trailing ``$`` for some preferred-share symbols.  Those are valid source
# rows even when the downstream Yahoo adapter does not support the raw spelling.
NASDAQ_DIRECTORY_SYMBOL_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:\.[A-Z0-9]+|\$[A-Z0-9]*)?")
EXCHANGE_PRODUCT_SYMBOL_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:[.-][A-Z0-9]+)?")
SEC_ENTITY_TICKER_PATTERN = re.compile(r"[A-Z][A-Z0-9]*(?:[.-][A-Z0-9]+)*")
SEC_UNAVAILABLE_ENTITY_TICKERS = frozenset({"NONE."})
SEC_UNAVAILABLE_MUTUAL_FUND_TICKERS = frozenset({"", "N/A", "NONE."})

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
    "BRZD": ("EWZ", "iShares MSCI Brazil ETF reference asset"),
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
    "HYGD": ("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF reference asset"),
    "JPND": ("EWJ", "iShares MSCI Japan ETF reference asset"),
    "KOLD": ("UNG", "United States Natural Gas Fund proxy"),
    "LQDD": ("LQD", "iShares iBoxx $ Investment Grade Corporate Bond ETF reference asset"),
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
    "SMHD": ("SMH", "VanEck Semiconductor ETF reference asset"),
    "SMDD": ("MDY", "SPDR S&P MidCap 400 ETF Trust proxy"),
    "SMN": ("XLB", "Materials Select Sector SPDR Fund proxy"),
    "SRS": ("IYR", "iShares U.S. Real Estate ETF proxy"),
    "SSG": ("SOXX", "iShares Semiconductor ETF proxy"),
    "SZK": ("XLP", "Consumer Staples Select Sector SPDR Fund proxy"),
    "TPEI": ("EWT", "iShares MSCI Taiwan ETF reference asset"),
    "WTID": ("XLE", "Energy Select Sector SPDR Fund proxy"),
    "XLCD": ("XLC", "Communication Services Select Sector SPDR Fund reference asset"),
    "XLPD": ("XLP", "Consumer Staples Select Sector SPDR Fund reference asset"),
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
    "BRZL": ("EWZ", "iShares MSCI Brazil ETF reference asset"),
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
    "HYGU": ("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF reference asset"),
    "IWDL": ("IWD", "iShares Russell 1000 Value ETF proxy"),
    "IWFL": ("IWF", "iShares Russell 1000 Growth ETF proxy"),
    "IWML": ("SIZE", "iShares MSCI USA Size Factor ETF proxy"),
    "JPNU": ("EWJ", "iShares MSCI Japan ETF reference asset"),
    "LTL": ("XLC", "Communication Services Select Sector SPDR Fund proxy"),
    "LQDU": ("LQD", "iShares iBoxx $ Investment Grade Corporate Bond ETF reference asset"),
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
    "RAML": ("DRAM", "Roundhill Memory ETF reference asset"),
    "ROM": ("XLK", "Technology Select Sector SPDR Fund proxy"),
    "RXL": ("XLV", "Health Care Select Sector SPDR Fund proxy"),
    "SAA": ("IJR", "iShares Core S&P Small-Cap ETF proxy"),
    "SCDL": ("SCHD", "Schwab US Dividend Equity ETF proxy"),
    "SKYU": ("SKYY", "First Trust Cloud Computing ETF proxy"),
    "SMHB": ("DES", "WisdomTree U.S. SmallCap Dividend Fund proxy"),
    "SMHU": ("SMH", "VanEck Semiconductor ETF reference asset"),
    "SPCL": ("UFO", "Procure Space ETF proxy"),
    "TARK": ("ARKK", "ARK Innovation ETF proxy"),
    "TAWN": ("EWT", "iShares MSCI Taiwan ETF reference asset"),
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
    "XLCU": ("XLC", "Communication Services Select Sector SPDR Fund reference asset"),
    "XLPU": ("XLP", "Consumer Staples Select Sector SPDR Fund reference asset"),
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
    "SKDD": ("SKHY", "SK hynix Inc. American depositary shares"),
    "SKHU": ("SKHY", "SK hynix Inc. American depositary shares"),
    "SKHX": ("SKHY", "SK hynix Inc. American depositary shares"),
    "SKUU": ("SKHY", "SK hynix Inc. American depositary shares"),
    "TXXD": ("DOGE-USD", "Dogecoin spot price proxy"),
    "TXXH": ("HYPE32196-USD", "Hyperliquid spot price proxy"),
}

# Exact ticker overrides are only safe while the listed product still represents
# the curated exposure. Exchange tickers can be recycled, so bind every proxy to
# a stable fingerprint expected in the current product name as well as its ticker.
# Patterns are keyed by the RSI proxy so products sharing an exposure also share
# the same identity rule.
RSI_PROXY_IDENTITY_PATTERNS = {
    "AIQ": r"\b(?:AIQ|ARTIFICIAL\s+INTELLIGENCE)\b",
    "AAPL": r"\b(?:AAPL|APPLE)\b",
    "ETN": r"\b(?:EATON|LONG\s+ETN)\b",
    "GOOG": r"\b(?:GOOGL?|ALPHABET)\b",
    "MSFT": r"\b(?:MSFT|MICROSOFT)\b",
    "NVDA": r"\b(?:NVDA|NVIDIA)\b",
    "TSLA": r"\b(?:TSLA|TESLA)\b",
    "BULL": r"\b(?:BULL|WEBULL)\b",
    "FNGS": r"\bFANG\+?\b",
    "IBB": r"\bBIOTECH(?:NOLOGY)?\b",
    "KBWB": r"\b(?:BIG\s+)?BANKS?\b",
    "EWZ": r"\bBRAZIL\b",
    "XLE": r"\b(?:ENERGY|BIG\s+OIL)\b",
    "EEM": r"\bEMERGING\s+MARKETS?\b",
    "EFA": r"\bEAFE\b",
    "VGK": r"\bEUROPE\b",
    "FXE": r"\bEURO\b",
    "EWJ": r"\bJAPAN\b",
    "PEJ": r"\b(?:TRAVEL|LEISURE|ENTERTAINMENT)\b",
    "FXI": r"\bCHINA\b",
    "HYG": r"\bHIGH\s+YIELD\b",
    "UNG": r"\bNATURAL\s+GAS\b",
    "LQD": r"\bINVESTMENT\s+GRADE(?:\s+CORPORATE\s+BONDS?)?\b",
    "MDY": r"\bMID[ -]?CAP\s*400\b",
    "XOP": r"\bOIL\s*(?:&|AND)?\s*GAS\s+EXPLORATION\b",
    "QQQ": r"\b(?:QQQ|INNOVATION\s+100)\b",
    "XLK": r"\bTECHNOLOGY\b",
    "XLV": r"\bHEALTH\s+CARE\b",
    "XLY": r"\bCONSUMER\s+DISCRETIONARY\b",
    "USO": r"\bCRUDE\s+OIL\b",
    "IJR": r"\bSMALL[ -]?CAP\s*600\b",
    "XLU": r"\bUTILITIES\b",
    "XLI": r"\bINDUSTRIALS?\b",
    "XLF": r"\bFINANCIALS?\b",
    "KRE": r"\bREGIONAL\s+BANKS?\b",
    "SMH": r"\bSEMICONDUCTORS?\b",
    "XLB": r"\b(?:BASIC\s+)?MATERIALS\b",
    "IYR": r"\bREAL\s+ESTATE\b",
    "SOXX": r"\bSEMICONDUCTORS?\b",
    "XLP": r"\bCONSUMER\s+STAPLES\b",
    "EWT": r"\bTAIWAN\b",
    "FXY": r"\bYEN\b",
    "MSTR": r"\b(?:MSTR|MICROSTRATEGY)\b",
    "MSOS": r"\b(?:MSOS|CANNABIS)\b",
    "SATS": r"\b(?:SATS|ECHOSTAR)\b",
    "VT": r"\bWORLD\b",
    "BIZD": r"\bBDC\b",
    "CEFS": r"\bCLOSED[ -]END\b",
    "DRNZ": r"\b(?:DRONE|AERIAL\s+AUTOMATION)\b",
    "FDRS": r"\bFOUNDER[ -]LED\b",
    "SPHD": r"\bHIGH\s+DIVIDEND\s+LOW\s+VOLATILITY\b",
    "IWD": r"\bVALUE\s+FACTOR\b",
    "IWF": r"\bGROWTH\s+FACTOR\b",
    "SIZE": r"\bSIZE\s+FACTOR\b",
    "XLC": r"\bCOMMUNICATION(?:S|\s+SERVICES)\b",
    "MAGS": r"\bMAGNIFICENT\s+SEVEN\b",
    "AMLP": r"\bMLP\b",
    "MTUM": r"\bMOMENTUM\s+FACTOR\b",
    "REM": r"\bMORTGAGE\s+REIT\b",
    "PFF": r"\bPREFERRED(?:\s+STOCK)?\b",
    "QTUM": r"\bQUANTUM\b",
    "QUAL": r"\bQUALITY\s+FACTOR\b",
    "DRAM": r"\bMEMORY\b",
    "SCHD": r"\bDIVIDEND\s+FACTOR\b",
    "SKYY": r"\bCLOUD\s+COMPUTING\b",
    "DES": r"\bSMALL\s+CAP\s+HIGH\s+DIVIDEND\b",
    "UFO": r"\bSPACE\b(?!\s*X\b)",
    "ARKK": r"\bINNOVATION\b",
    "CIBR": r"\bCYBERSECURITY\b",
    "USMV": r"\bMINIMUM\s+VOLATILITY\b",
    "VIXY": r"\bVIX\b",
    "XRP-USD": r"\bXRP\b",
    "COPX": r"\bCOPPER\s+MINERS?\b",
    "CPER": r"\bCOPPER\b(?!\s+MINERS?\b)",
    "PALL": r"\bPALLADIUM\b",
    "PPLT": r"\bPLATINUM\b",
    "BTC-USD": r"\bBITCOIN\b",
    "AVAX-USD": r"\bAVALANCHE\b",
    "LINK-USD": r"\bCHAINLINK\b",
    "ADA-USD": r"\bCARDANO\b",
    "ETH-USD": r"\bETHER(?:EUM)?\b",
    "SOL-USD": r"\bSOLANA\b",
    "XLM-USD": r"\bSTELLAR\b",
    "SUI20947-USD": r"\bSUI\b",
    "SKHY": r"\b(?:SK\s+HYNIX|SKHY)\b",
    "DOGE-USD": r"\bDOGECOIN\b",
    "HYPE32196-USD": r"\b(?:HYPE|HYPERLIQUID)\b",
}

RSI_SELF_FALLBACK_SYMBOL_OVERRIDES = {
    "BEGS": ("BEGS", "Rareview 2X Bull Cryptocurrency & Precious Metals ETF self-RSI fallback"),
}

RSI_SYMBOLS_REQUIRING_REVIEW: dict[str, str] = {}

RSI_NAME_PATTERNS_REQUIRING_REVIEW: list[tuple[str, str]] = []

RSI_NAME_PROXY_PATTERNS = [
    (r"\bSK\s+HYNIX\b", "SKHY", "SK hynix Inc. American depositary shares"),
    (r"\bSPACE\s*X\b|\bSPACEX\b", "SPCX", "Space Exploration Technologies Corp. Class A"),
    (r"\bS\s*&\s*P\s*500\s+EQUAL\s+WEIGHT\b", "RSP", "Invesco S&P 500 Equal Weight ETF proxy"),
    (
        r"\bS\s*&\s*P\s*500\b(?!\s+EQUAL\s+WEIGHT)|\bS&P500\b(?!\s+EQUAL\s+WEIGHT)",
        "SPY",
        "SPDR S&P 500 ETF Trust proxy",
    ),
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
EXPLICIT_POSITIVE_LEVERAGE_PATTERN = r"(?<![A-Z0-9])[+\uFF0B]\s*\d+(?:\.\d+)?\s*(?:X|%)(?![A-Z0-9])"
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
    r"\bultrapro\b",
    r"\bultra\b",
    r"\bbull\s+[2-5](?:\.\d+)?\s*x\b",
    r"\b(?:daily\s+)?target\s+[2-5](?:\.\d+)?\s*x\b",
    r"\bleveraged\b",
]

# Unlike an ``X`` multiple, a percentage in a product name commonly describes
# an outcome cap or participation rate rather than daily leverage.  Only treat
# percentage tokens as leverage when the surrounding product name explicitly
# describes a leveraged direction/reset objective.
PERCENT_LEVERAGE_CONTEXT_PATTERNS = [
    r"\b(?:LONG|BULL|SHORT|BEAR|INVERSE|LEVERAGED)\b",
]
# ``DAILY`` alone is not enough context: income products can advertise a daily
# distribution or yield percentage that has nothing to do with exposure.  A
# direction/leverage word above remains conclusive; otherwise require the
# percentage to participate in a nearby daily target/exposure/reset objective.
PERCENT_DAILY_OBJECTIVE_PATTERN = r"(?:TARGET(?:ED)?(?:\s+EXPOSURE)?|EXPOSURE|RESET)"
PERCENT_DEFINED_OUTCOME_PATTERNS = [
    r"\bPARTICIPATION\b",
    r"\bBUFFER(?:ED)?\b",
    r"\bDEFINED[- ]OUTCOME\b",
    r"\bUPSIDE\s+CAP\b",
    r"\bCAP(?:PED)?\s+(?:GAIN|RETURN|OUTCOME)\b",
]

LONG_DIRECTION_PATTERNS = [
    r"\bbull\b",
    r"\blong\b",
    r"\blong\s+exposure\b",
    r"\bultra\b",
    r"\bleveraged\s+long\b",
    r"\bleveraged\s+exposure\b",
    r"\b[23]x\s+leveraged\b",
    EXPLICIT_POSITIVE_LEVERAGE_PATTERN,
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

LEVERAGE_FALSE_POSITIVE_PATTERNS = [
    # In fixed-income names, spaced or hyphenated ``ultra short`` normally
    # describes duration rather than inverse exposure.  A joined ``UltraShort``
    # immediately followed by the asset class has the same generic meaning;
    # branded leveraged products such as ``UltraShort 20+ Year Treasury`` keep
    # their classification because the tenor separates the words.
    r"\bULTRA(?:\s+|-)SHORT(?:[-\s]+TERM)?[-\s]+(?:BONDS?|TREASUR(?:Y|IES)|FIXED[-\s]+INCOME)\b",
    r"\bULTRASHORT(?:[-\s]+TERM)?[-\s]+(?:BONDS?|TREASUR(?:Y|IES)|FIXED[-\s]+INCOME)\b",
]

LEVERAGE_DIRECTION_NAME_OVERRIDES = [
    (r"\bMSOS\s+DAILY\s+LEVERAGED\s+ETF\b", 2.0, "long"),
    (r"\bDEFIANCE\s+LEVERAGED\s+LONG\s+(?:\+|PLUS)\s+INCOME\s+MSTR\s+ETF\b", 1.75, "long"),
    (r"\b100%\s+TSLA\s*(?:\+|\bAND\b|&)\s*100%\s+SPCX\s+(?:DAILY\s+)?ETF\b", 2.0, "long"),
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


def _nasdaq_directory_symbol_identity(symbol: object) -> str | None:
    """Return a stable source identity without applying product exclusions."""
    if not isinstance(symbol, str):
        return None
    candidate = symbol.strip()
    if (
        not candidate
        or candidate != symbol
        or len(candidate) > 14
        or NASDAQ_DIRECTORY_SYMBOL_PATTERN.fullmatch(candidate) is None
    ):
        return None
    # These aliases are equivalent source identities for duplicate/conflict
    # checks.  Other documented Nasdaq suffix notation stays intact here.
    return YAHOO_SYMBOL_ALIASES.get(candidate, candidate)


def _exchange_product_symbol_identity(symbol: object) -> str | None:
    """Validate a canonical exchange-traded product ticker."""
    if not isinstance(symbol, str):
        return None
    candidate = symbol.strip()
    if (
        not candidate
        or candidate != symbol
        or len(candidate) > 14
        or EXCHANGE_PRODUCT_SYMBOL_PATTERN.fullmatch(candidate) is None
    ):
        return None
    return YAHOO_SYMBOL_ALIASES.get(candidate, candidate)


def _sec_entity_ticker_identity(symbol: object) -> str | None:
    """Validate an SEC company/exchange ticker independently of exclusions."""
    if not isinstance(symbol, str):
        return None
    candidate = symbol.strip()
    if not candidate or len(candidate) > 14 or SEC_ENTITY_TICKER_PATTERN.fullmatch(candidate) is None:
        return None
    return YAHOO_SYMBOL_ALIASES.get(candidate, candidate)


def _sec_mutual_fund_ticker_identity(symbol: str) -> str | None:
    """Normalize the SEC mutual-fund feed's documented legacy spellings."""
    if symbol != symbol.strip():
        return None
    candidate = symbol.upper()
    if candidate in SEC_UNAVAILABLE_MUTUAL_FUND_TICKERS:
        return ""
    if candidate.startswith("(") and candidate.endswith(")"):
        candidate = candidate[1:-1]
    return _sec_entity_ticker_identity(candidate)


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


def _mapping_from_curated_symbol(asset_symbol: str, normalized_name: str) -> RsiSymbolMapping | None:
    override = RSI_SYMBOL_OVERRIDES.get(asset_symbol)
    if override is None:
        return None
    rsi_symbol, underlying_name = override
    identity_pattern = RSI_PROXY_IDENTITY_PATTERNS.get(rsi_symbol)
    if identity_pattern is None or re.search(identity_pattern, normalized_name) is None:
        expected_identity = identity_pattern or f"a configured identity pattern for {rsi_symbol}"
        return RsiSymbolMapping(
            rsi_symbol=asset_symbol,
            underlying_name=asset_symbol,
            mapping_source="symbol_override_identity_mismatch",
            confidence="needs_review",
            mapping_reason=(
                f"exact symbol override for {rsi_symbol} did not match expected product identity {expected_identity}"
            ),
        )
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


def _mapping_from_curated_name(asset_symbol: str, normalized_name: str) -> RsiSymbolMapping | None:
    matches: dict[str, tuple[str, list[str]]] = {}
    for pattern, rsi_symbol, underlying_name in RSI_NAME_PROXY_PATTERNS:
        if re.search(pattern, normalized_name):
            matched_underlying_name, matched_patterns = matches.setdefault(rsi_symbol, (underlying_name, []))
            matched_patterns.append(pattern)
            matches[rsi_symbol] = matched_underlying_name, matched_patterns
    if not matches:
        return None
    if len(matches) > 1:
        matched_symbols = ", ".join(sorted(matches))
        return RsiSymbolMapping(
            rsi_symbol=asset_symbol,
            underlying_name=asset_symbol,
            mapping_source="ambiguous_name_proxy",
            confidence="needs_review",
            mapping_reason=f"product name matched multiple curated RSI proxies: {matched_symbols}",
        )
    rsi_symbol, (underlying_name, patterns) = next(iter(matches.items()))
    return RsiSymbolMapping(
        rsi_symbol=rsi_symbol,
        underlying_name=underlying_name,
        mapping_source="name_proxy",
        confidence="curated",
        mapping_reason=f"matched curated name proxy pattern {patterns[0]}",
    )


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

    curated_symbol_mapping = _mapping_from_curated_symbol(asset_symbol, normalized_name)
    self_fallback_mapping = _mapping_from_self_fallback_symbol(asset_symbol)
    curated_name_mapping = _mapping_from_curated_name(asset_symbol, normalized_name)
    if curated_symbol_mapping is not None:
        if curated_symbol_mapping.confidence == "needs_review":
            return curated_symbol_mapping
        if curated_name_mapping is not None and (
            curated_name_mapping.confidence == "needs_review"
            or curated_name_mapping.rsi_symbol != curated_symbol_mapping.rsi_symbol
        ):
            return RsiSymbolMapping(
                rsi_symbol=asset_symbol,
                underlying_name=asset_symbol,
                mapping_source="symbol_override_identity_mismatch",
                confidence="needs_review",
                mapping_reason="exact symbol override conflicted with curated product-name proxy metadata",
            )
        return curated_symbol_mapping

    for curated_mapping in [
        self_fallback_mapping,
        curated_name_mapping,
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


def _percent_leverage_has_context(normalized_name: str, match: re.Match[str]) -> bool:
    if any(re.search(pattern, normalized_name, re.I) for pattern in PERCENT_DEFINED_OUTCOME_PATTERNS):
        return False
    raw_multiple = match.group(1)
    if raw_multiple.startswith(("+", "-")):
        return True
    if any(re.search(pattern, normalized_name, re.I) for pattern in PERCENT_LEVERAGE_CONTEXT_PATTERNS):
        return True

    prefix = normalized_name[: match.start()].rstrip()
    suffix = normalized_name[match.end() :].lstrip()
    daily_objective = PERCENT_DAILY_OBJECTIVE_PATTERN
    return bool(
        re.search(rf"\bDAILY\s+{daily_objective}\s*$", prefix, re.I)
        or re.match(rf"^DAILY\s+{daily_objective}\b", suffix, re.I)
        or (re.search(r"\bDAILY\s*$", prefix, re.I) and re.match(rf"^{daily_objective}\b", suffix, re.I))
    )


def _numeric_leverage_matches(normalized_name: str) -> list[tuple[int, float]]:
    matches: list[tuple[int, float]] = []
    for match in re.finditer(NUMERIC_X_LEVERAGE_PATTERN, normalized_name):
        matches.append((match.start(), abs(float(match.group(1)))))
    for match in re.finditer(NUMERIC_PERCENT_LEVERAGE_PATTERN, normalized_name):
        if _percent_leverage_has_context(normalized_name, match):
            matches.append((match.start(), abs(float(match.group(1))) / 100.0))
    return sorted(matches)


def _has_leverage_false_positive_term(name: str) -> bool:
    normalized_name = f" {str(name).upper()} "
    if not (
        any(term in normalized_name for term in LEVERAGE_FALSE_POSITIVE_TERMS)
        or any(re.search(pattern, normalized_name) for pattern in LEVERAGE_FALSE_POSITIVE_PATTERNS)
    ):
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
            if pattern == NUMERIC_PERCENT_LEVERAGE_PATTERN and not _percent_leverage_has_context(
                normalized_name,
                match,
            ):
                continue
            raw_value = abs(float(match.group(1)))
            leverage = raw_value / 100.0 if pattern == NUMERIC_PERCENT_LEVERAGE_PATTERN else raw_value
            if leverage > MAX_RECOGNIZED_LEVERAGE and product_context_pattern.match(normalized_name[match.end() :]):
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
        None if _has_unrecognized_product_leverage(normalized_name) else _recognized_numeric_leverage(normalized_name)
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
    return any(leverage > MAX_RECOGNIZED_LEVERAGE for leverage in numeric_leverages) and not any(
        1.0 < leverage <= MAX_RECOGNIZED_LEVERAGE for leverage in numeric_leverages
    )


def _classify_leveraged_name(name: object) -> tuple[bool, float | None, str | None]:
    """Classify a product name with a single leverage/direction inference."""
    normalized_name = f" {str(name).upper()} "
    leverage, direction = infer_leverage_and_direction(normalized_name)
    if _has_leverage_false_positive_term(normalized_name) or _has_unrecognized_numeric_leverage(normalized_name):
        return False, leverage, direction
    if leverage is not None:
        return leverage > 1.0, leverage, direction
    is_candidate = any(re.search(pattern, normalized_name, re.I) for pattern in LEVERAGE_NAME_PATTERNS)
    return is_candidate, leverage, direction


def leveraged_name_filter(name: str) -> bool:
    is_candidate, _leverage, _direction = _classify_leveraged_name(name)
    return is_candidate


def _read_nasdaq_symbol_file(url: str, timeout: int) -> pd.DataFrame:
    schema = _NASDAQ_ACTIVE_LISTING_SCHEMAS.get(url)
    if schema is None:
        raise ValueError(f"Unsupported Nasdaq active-listing source URL: {url!r}.")
    expected_header, footer_pipe_count, minimum_data_rows = schema
    resp = _get_universe_response(url, timeout)
    resp.raise_for_status()
    material_lines = [line for line in resp.text.splitlines() if line]
    if not material_lines:
        raise ValueError("Nasdaq symbol file was empty.")

    try:
        csv_rows = list(csv.reader(material_lines, delimiter="|", strict=True))
    except csv.Error as exc:
        raise ValueError("Nasdaq symbol file was not valid pipe-delimited data.") from exc
    if tuple(csv_rows[0]) != expected_header:
        raise ValueError(
            "Nasdaq symbol file header did not match the complete expected schema; "
            f"expected {list(expected_header)!r}, got {csv_rows[0]!r}."
        )

    footer_indexes = [index for index, line in enumerate(material_lines) if line.startswith("File Creation Time")]
    if footer_indexes != [len(material_lines) - 1]:
        raise ValueError(
            "Nasdaq symbol file must end with exactly one File Creation Time footer and no trailing content."
        )
    footer = material_lines[-1]
    footer_match = re.fullmatch(
        rf"File Creation Time: (?P<timestamp>\d{{10}}:\d{{2}})\|{{{footer_pipe_count}}}",
        footer,
    )
    if footer_match is None:
        raise ValueError("Nasdaq symbol file contained an invalid File Creation Time footer.")
    try:
        parsed_footer_time = datetime.strptime(footer_match.group("timestamp"), "%m%d%Y%H:%M")
    except ValueError as exc:
        raise ValueError("Nasdaq symbol file contained an invalid File Creation Time timestamp.") from exc
    if parsed_footer_time.strftime("%m%d%Y%H:%M") != footer_match.group("timestamp"):
        raise ValueError("Nasdaq symbol file contained a noncanonical File Creation Time timestamp.")
    now = _nasdaq_directory_now().astimezone(_NEW_YORK).replace(tzinfo=None)
    snapshot_age = now - parsed_footer_time
    if snapshot_age > _NASDAQ_ACTIVE_LISTING_MAX_AGE:
        raise ValueError(
            f"Nasdaq active-listing snapshot was stale; its File Creation Time was {snapshot_age.days} days old."
        )
    if snapshot_age < -_NASDAQ_ACTIVE_LISTING_MAX_FUTURE_SKEW:
        raise ValueError("Nasdaq active-listing snapshot File Creation Time was implausibly far in the future.")

    data_rows = csv_rows[1:-1]
    malformed_row = next(
        (row_number for row_number, row in enumerate(data_rows, start=1) if len(row) != len(expected_header)),
        None,
    )
    if malformed_row is not None:
        raise ValueError(f"Nasdaq symbol file row {malformed_row} did not match its declared header width.")
    if len(data_rows) < minimum_data_rows:
        raise ValueError(
            "Nasdaq active-listing snapshot was implausibly small and may be truncated; "
            f"found {len(data_rows)} rows, expected at least {minimum_data_rows}."
        )

    # A real listed ticker can be text such as ``NA``.  Disabling pandas'
    # default NA vocabulary preserves source values so validation can
    # distinguish that ticker from an actually empty cell.
    listed = pd.read_csv(
        io.StringIO("\n".join(material_lines[:-1])),
        sep="|",
        dtype=str,
        keep_default_na=False,
    )
    listed.attrs["minimum_usable_symbols"] = minimum_data_rows
    listed.attrs["snapshot_created_at"] = parsed_footer_time.isoformat(timespec="minutes")
    return listed


def _required_active_listing_column(listed: pd.DataFrame, expected_label: str) -> object:
    normalized_expected = re.sub(r"[^a-z0-9]+", " ", expected_label.casefold()).strip()
    matches = [
        column
        for column in listed.columns
        if re.sub(r"[^a-z0-9]+", " ", str(column).casefold()).strip() == normalized_expected
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {expected_label!r} column; found {len(matches)} in {list(listed.columns)!r}"
        )
    return matches[0]


def load_active_listed_symbols(timeout: int = 30) -> set[str]:
    timeout = _validated_universe_request_timeout(timeout)
    symbols: set[str] = set()
    source_status: list[dict[str, object]] = []
    for source_name, url, symbol_col in [
        ("nasdaq_listed", NASDAQ_LISTED_URL, "Symbol"),
        ("other_listed", OTHER_LISTED_URL, "ACT Symbol"),
    ]:
        try:
            listed = _read_nasdaq_symbol_file(url, timeout)
            actual_symbol_col = _required_active_listing_column(listed, symbol_col)
            test_issue_col = _required_active_listing_column(listed, "Test Issue")
            raw_test_issue_values = listed[test_issue_col].tolist()
            invalid_test_issue_values = sorted(
                {repr(value) for value in raw_test_issue_values if type(value) is not str or value not in {"N", "Y"}}
            )
            if invalid_test_issue_values:
                raise ValueError(
                    "'Test Issue' column must contain only 'Y' or 'N' as literal values; "
                    f"found {invalid_test_issue_values!r}"
                )
            test_issue_values = pd.Series(raw_test_issue_values, index=listed.index, dtype="string")

            source_identities: list[str] = []
            invalid_symbol_rows: list[int] = []
            for row_number, raw_symbol in enumerate(listed[actual_symbol_col]):
                identity = _nasdaq_directory_symbol_identity(raw_symbol)
                if identity is None:
                    invalid_symbol_rows.append(row_number)
                else:
                    source_identities.append(identity)
            if invalid_symbol_rows:
                preview = ", ".join(str(row_number) for row_number in invalid_symbol_rows[:10])
                raise ValueError(f"symbol column contained missing or invalid values at rows: {preview}")

            validation = pd.DataFrame(
                {
                    "source_identity": source_identities,
                    "test_issue": test_issue_values.tolist(),
                }
            )
            duplicate_rows = validation.loc[validation["source_identity"].duplicated(keep=False)]
            if not duplicate_rows.empty:
                contradictory = duplicate_rows.groupby("source_identity", sort=False)["test_issue"].nunique().gt(1)
                contradictory_symbols = sorted(contradictory.index[contradictory].astype(str))
                if contradictory_symbols:
                    raise ValueError(
                        "duplicate normalized symbols contained contradictory 'Test Issue' values: "
                        + ", ".join(contradictory_symbols)
                    )
                duplicate_symbols = sorted(duplicate_rows["source_identity"].unique())
                raise ValueError(
                    "symbol column contained duplicate normalized symbols: " + ", ".join(duplicate_symbols)
                )

            listed = listed.loc[test_issue_values.eq("N")]
            source_symbols = {
                symbol
                for raw_symbol in listed[actual_symbol_col].dropna()
                if (symbol := normalize_yahoo_symbol(raw_symbol)) is not None
            }
            if not source_symbols:
                raise ValueError("source contained no usable listed symbols")
            minimum_usable_symbols = listed.attrs.get("minimum_usable_symbols")
            if type(minimum_usable_symbols) is int and len(source_symbols) < minimum_usable_symbols:
                raise ValueError(
                    "source contained implausibly few usable listed symbols and may be truncated; "
                    f"found {len(source_symbols)}, expected at least {minimum_usable_symbols}."
                )
        except Exception as exc:
            source_status.append(
                {
                    "source": source_name,
                    "url": url,
                    "symbol_column": symbol_col,
                    "status": "error",
                    "symbol_count": 0,
                    "error": safe_diagnostic_text(
                        exc,
                        max_chars=_UNIVERSE_DIAGNOSTIC_MAX_CHARS,
                    ),
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


def _normalized_column_label(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(column).casefold()).strip()


_CBOE_ISSUER_COLUMN_LABELS = {
    "symbol": frozenset({"symbol", "ticker", "ticker symbol"}),
    "product name": frozenset({"product", "product name", "fund", "fund name"}),
    "product type": frozenset({"product type", "fund type", "type"}),
}
_CBOE_ISSUER_PRODUCT_TYPES = frozenset({"ETF", "ETN"})


def _cboe_issuer_column_label(column: object) -> str:
    """Recover semantic labels that ``read_html`` may hide with ``.N`` suffixes."""
    text = str(column).strip()
    label = _normalized_column_label(text)
    mangled = re.fullmatch(r"(?P<label>.+)\.(?:[1-9]\d*)", text)
    if mangled is not None:
        base_label = _normalized_column_label(mangled.group("label"))
        if any(base_label in labels for labels in _CBOE_ISSUER_COLUMN_LABELS.values()):
            return base_label
    return label


def _cboe_issuer_table_schema(
    table: pd.DataFrame,
    *,
    table_number: int,
) -> tuple[str, str, str] | None:
    """Select one unambiguous Cboe product schema while ignoring page furniture."""
    columns = list(table.columns)
    matches = {
        role: [column for column in columns if _cboe_issuer_column_label(column) in labels]
        for role, labels in _CBOE_ISSUER_COLUMN_LABELS.items()
    }
    # A navigation, quote, or other incidental table may happen to expose one
    # generic label such as Symbol or Type. Two product-schema roles identify a
    # candidate strongly enough that a missing third role must fail closed.
    if sum(bool(role_matches) for role_matches in matches.values()) < 2:
        return None

    ambiguous = {role: role_matches for role, role_matches in matches.items() if len(role_matches) > 1}
    if ambiguous:
        details = "; ".join(f"{role}: {', '.join(map(str, role_matches))}" for role, role_matches in ambiguous.items())
        raise ValueError(f"Cboe issuer listing table {table_number} contained ambiguous required columns ({details}).")

    missing = [role for role, role_matches in matches.items() if not role_matches]
    if missing:
        raise ValueError(f"Cboe issuer listing table {table_number} omitted its required {', '.join(missing)} column.")

    return (
        str(matches["symbol"][0]),
        str(matches["product name"][0]),
        str(matches["product type"][0]),
    )


_NASDAQ_ETF_SYMBOL_COLUMN_LABELS = frozenset({"symbol", "fund symbol", "ticker", "ticker symbol"})
_NASDAQ_ETF_NAME_COLUMN_LABELS = frozenset(
    {
        "fund",
        "fund name",
        "security name",
        "etf name",
        "exchange traded fund name",
    }
)
_NASDAQ_ETF_TYPE_COLUMN_LABELS = frozenset({"fund type", "type"})
_NASDAQ_ETF_STRONG_NAME_COLUMN_LABELS = _NASDAQ_ETF_NAME_COLUMN_LABELS - {"security name"}
_NASDAQ_ETF_STRONG_TYPE_COLUMN_LABELS = frozenset({"fund type"})


def _nasdaq_etf_column_label(column: object) -> str:
    """Recover semantic headers hidden by pandas' duplicate ``.N`` suffix."""
    text = str(column).strip()
    label = _normalized_column_label(text)
    mangled = re.fullmatch(r"(?P<label>.+)\.(?:[1-9]\d*)", text)
    if mangled is None:
        return label
    base_label = _normalized_column_label(mangled.group("label"))
    accepted_labels = _NASDAQ_ETF_SYMBOL_COLUMN_LABELS | _NASDAQ_ETF_NAME_COLUMN_LABELS | _NASDAQ_ETF_TYPE_COLUMN_LABELS
    return base_label if base_label in accepted_labels else label


def _nasdaq_etf_table_schema(df: pd.DataFrame) -> tuple[str, str, str] | None:
    """Return an unambiguous explicit symbol/name/type schema, if present."""
    columns = list(df.columns)
    matches = {
        "symbol": [
            column for column in columns if _nasdaq_etf_column_label(column) in _NASDAQ_ETF_SYMBOL_COLUMN_LABELS
        ],
        "fund/security name": [
            column for column in columns if _nasdaq_etf_column_label(column) in _NASDAQ_ETF_NAME_COLUMN_LABELS
        ],
        "fund type": [
            column for column in columns if _nasdaq_etf_column_label(column) in _NASDAQ_ETF_TYPE_COLUMN_LABELS
        ],
    }
    # Generic page furniture commonly contains one semantic label, or a small
    # Security Name/Type pair. Two roles are authoritative evidence when a
    # symbol is present, either header is fund-specific, or the table has
    # inventory-like volume. This catches a truncated Fund Name/Fund Type table
    # without letting small navigation/reference tables block the real source.
    matched_role_count = sum(bool(role_matches) for role_matches in matches.values())
    if matched_role_count < 2:
        return None
    matched_name_labels = {_nasdaq_etf_column_label(column) for column in matches["fund/security name"]}
    matched_type_labels = {_nasdaq_etf_column_label(column) for column in matches["fund type"]}
    has_fund_specific_header = bool(
        matched_name_labels & _NASDAQ_ETF_STRONG_NAME_COLUMN_LABELS
        or matched_type_labels & _NASDAQ_ETF_STRONG_TYPE_COLUMN_LABELS
    )
    if not matches["symbol"] and not has_fund_specific_header and len(df) < _NASDAQ_ETF_SCHEMA_CANDIDATE_MIN_ROWS:
        return None
    ambiguous = {role: role_matches for role, role_matches in matches.items() if len(role_matches) > 1}
    if ambiguous:
        details = "; ".join(f"{role}: {', '.join(map(str, role_matches))}" for role, role_matches in ambiguous.items())
        raise RuntimeError(f"Nasdaq ETF definitions table contained ambiguous semantic columns ({details}).")
    missing = [role for role, role_matches in matches.items() if not role_matches]
    if missing:
        raise RuntimeError(
            "Nasdaq ETF definitions table matched an incomplete explicit symbol, "
            "fund/security name, and fund type schema; omitted required "
            f"{', '.join(missing)} column."
        )
    return (
        str(matches["symbol"][0]),
        str(matches["fund/security name"][0]),
        str(matches["fund type"][0]),
    )


def _read_html_tables(html: str) -> list[pd.DataFrame]:
    try:
        # ``NA`` is a real exchange ticker. Preserve literal source cells and
        # let each authoritative parser decide whether an empty value is valid.
        return pd.read_html(io.StringIO(html), flavor=["lxml"], keep_default_na=False)
    except ValueError:
        return []


def _read_html_tables_with_body_links(html: str) -> list[pd.DataFrame]:
    """Read source tables while retaining the product links used for identity checks."""
    try:
        return pd.read_html(
            io.StringIO(html),
            flavor=["lxml"],
            keep_default_na=False,
            extract_links="body",
        )
    except ValueError:
        return []


def _required_nasdaq_etf_text(value: object, *, field: str, row_number: int) -> str:
    if type(value) is not str:
        raise RuntimeError(
            f"Nasdaq Trader ETF definitions page contained a non-text {field} value at row {row_number}."
        )
    try:
        text = _html_text(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Nasdaq Trader ETF definitions page contained unsafe control characters in {field} at row {row_number}."
        ) from exc
    if not text:
        raise RuntimeError(f"Nasdaq Trader ETF definitions page contained an empty {field} value at row {row_number}.")
    return text


def load_current_etf_universe(timeout: int = 30) -> pd.DataFrame:
    """
    Load the current Nasdaq Trader ETF definitions table.
    Free public source for the ETF universe.
    """
    timeout = _validated_universe_request_timeout(timeout)
    resp = _get_universe_response(ETF_DEFS_URL, timeout)
    resp.raise_for_status()

    tables = _read_html_tables(resp.text)
    if not tables:
        raise RuntimeError("No tables found on Nasdaq Trader ETF definitions page.")

    cleaned_tables = [_clean_columns(table) for table in tables]
    table_candidates = [
        (table, schema) for table in cleaned_tables if (schema := _nasdaq_etf_table_schema(table)) is not None
    ]
    table_columns = [list(table.columns) for table in cleaned_tables]
    if not table_candidates:
        raise RuntimeError(
            "Could not identify an ETF definitions table with one explicit symbol, "
            f"fund/security name, and fund type column. Tables found: {table_columns}"
        )
    if len(table_candidates) != 1:
        candidate_columns = [list(table.columns) for table, _schema in table_candidates]
        raise RuntimeError(
            "ETF definitions page contained multiple tables with the required schema; "
            f"refusing ambiguous selection. Candidate columns: {candidate_columns}"
        )

    df, (symbol_col, name_col, fund_type_col) = table_candidates[0]

    source_rows = df[[symbol_col, name_col, fund_type_col]].copy()
    source_rows.columns = ["symbol", "name", "fund_type"]
    validated_rows: list[dict[str, object]] = []
    for row_number, row in enumerate(source_rows.itertuples(index=False), start=0):
        raw_symbol = row.symbol
        if _nasdaq_directory_symbol_identity(raw_symbol) is None:
            raise RuntimeError(
                "Nasdaq Trader ETF definitions page contained a missing, invalid, "
                f"or noncanonical symbol value at row {row_number}."
            )
        validated_rows.append(
            {
                "_source_symbol": raw_symbol,
                "symbol": normalize_yahoo_symbol(raw_symbol),
                "name": _required_nasdaq_etf_text(
                    row.name,
                    field="fund name",
                    row_number=row_number,
                ),
                "fund_type": _required_nasdaq_etf_text(
                    row.fund_type,
                    field="fund type",
                    row_number=row_number,
                ),
            }
        )
    if not validated_rows:
        raise RuntimeError("Nasdaq Trader ETF definitions page contained no usable ETF rows.")
    out = pd.DataFrame(validated_rows)

    normalized_source_rows = out.loc[out["symbol"].notna()]
    normalization_variants = normalized_source_rows.groupby("symbol", sort=False)["_source_symbol"].nunique()
    colliding_symbols = sorted(normalization_variants.index[normalization_variants.gt(1)].astype(str))
    if colliding_symbols:
        preview = ", ".join(colliding_symbols[:10])
        if len(colliding_symbols) > 10:
            preview = f"{preview}, and {len(colliding_symbols) - 10} more"
        raise RuntimeError(
            "Nasdaq Trader ETF definitions page contained source symbol spellings "
            f"with a normalization collision: {preview}."
        )

    # Check source identities before eligibility filtering.  Otherwise a
    # conflicting duplicate with a missing name or a non-ETF type could be
    # discarded first and let the remaining row silently define the symbol.
    duplicate_rows = out.loc[out["symbol"].notna() & out["symbol"].duplicated(keep=False)]
    if not duplicate_rows.empty:
        metadata_variants = duplicate_rows.groupby("symbol", sort=False)[["name", "fund_type"]].nunique(dropna=False)
        conflicting_symbols = sorted(metadata_variants.index[metadata_variants.max(axis=1).gt(1)].astype(str))
        if conflicting_symbols:
            preview = ", ".join(conflicting_symbols[:10])
            if len(conflicting_symbols) > 10:
                preview = f"{preview}, and {len(conflicting_symbols) - 10} more"
            raise RuntimeError(
                "Nasdaq Trader ETF definitions page contained conflicting metadata for "
                f"normalized duplicate symbols: {preview}."
            )

    if len(df) < _NASDAQ_ETF_MINIMUM_RAW_ROWS:
        raise RuntimeError(
            "Nasdaq Trader ETF definitions page was implausibly small and may be truncated; "
            f"found {len(df)} raw rows, expected at least {_NASDAQ_ETF_MINIMUM_RAW_ROWS}."
        )

    # Exact normalized duplicates are the only safe duplicates; collapse them
    # before applying the table's ETF eligibility rules.
    out = out.drop_duplicates(subset=["_source_symbol", "symbol", "name", "fund_type"])
    out = out[out["symbol"].notna()]
    out = out[out["name"].ne("")]
    out = out[out["fund_type"].str.startswith("ETF", na=False)]
    out = out[out["symbol"].str.fullmatch(r"[A-Z][A-Z0-9.-]*", na=False)].copy()

    out = out.drop_duplicates(subset=["symbol"]).drop(columns="_source_symbol").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Nasdaq Trader ETF definitions page contained no usable ETF rows.")
    if len(out) < _NASDAQ_ETF_MINIMUM_USABLE_ROWS:
        raise RuntimeError(
            "Nasdaq Trader ETF definitions page contained implausibly few usable ETF symbols and may be truncated; "
            f"found {len(out)}, expected at least {_NASDAQ_ETF_MINIMUM_USABLE_ROWS}."
        )
    out.attrs["workflow_source_status"] = [
        _workflow_source_status_row(
            source=NASDAQ_ETF_SOURCE_NAME,
            source_type=NASDAQ_ETF_SOURCE_TYPE,
            url=ETF_DEFS_URL,
            status="loaded",
            parsed_row_count=len(df),
            row_count=len(out),
        )
    ]
    return out


def is_long_leveraged_name(name: str) -> bool:
    is_candidate, leverage, direction = _classify_leveraged_name(name)
    return is_candidate and leverage is not None and leverage > 1.0 and direction == "long"


def is_short_leveraged_name(name: str) -> bool:
    is_candidate, leverage, direction = _classify_leveraged_name(name)
    return is_candidate and leverage is not None and leverage > 1.0 and direction == "inverse"


def _first_matching_column(columns: Iterable[object], patterns: list[str]) -> object | None:
    normalized = [(column, str(column).strip()) for column in columns]
    for pattern in patterns:
        regex = re.compile(pattern, re.I)
        for raw_column, column in normalized:
            if regex.search(column):
                return raw_column
    return None


def _html_text(value: object) -> str:
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", str(value))
    decoded = unescape(without_tags)
    if any(not character.isprintable() and not character.isspace() for character in decoded):
        raise ValueError("Authoritative source text contained unsafe control characters.")
    return re.sub(r"\s+", " ", decoded).strip()


def _decode_quoted_js_text(value: str) -> str:
    """Decode one JavaScript string body without evaluating source text."""
    if len(value) > _MAX_JS_QUOTED_TEXT_CHARS:
        raise ValueError("Embedded JavaScript string exceeded the safe parsing limit.")

    simple_escapes = {
        "'": "'",
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
    }
    hexadecimal = frozenset("0123456789abcdefABCDEF")
    decoded: list[str] = []
    offset = 0

    def fixed_unicode_code_unit(start: int) -> tuple[int, int]:
        end = start + 4
        digits = value[start:end]
        if len(digits) != 4 or any(character not in hexadecimal for character in digits):
            raise ValueError("Embedded JavaScript string contained an invalid Unicode escape.")
        return int(digits, 16), end

    while offset < len(value):
        character = value[offset]
        if character in "\r\n\u2028\u2029":
            raise ValueError("Embedded JavaScript string contained an unescaped line terminator.")
        if character != "\\":
            if 0xD800 <= ord(character) <= 0xDFFF:
                raise ValueError("Embedded JavaScript string contained an unpaired Unicode surrogate.")
            decoded.append(character)
            offset += 1
            continue

        if offset + 1 >= len(value):
            raise ValueError("Embedded JavaScript string ended with an incomplete escape.")
        escape = value[offset + 1]
        offset += 2

        # JavaScript line continuations contribute no character to the value.
        if escape == "\r":
            if offset < len(value) and value[offset] == "\n":
                offset += 1
            continue
        if escape in "\n\u2028\u2029":
            continue
        if escape in simple_escapes:
            decoded.append(simple_escapes[escape])
            continue
        if escape == "0":
            if offset < len(value) and value[offset].isdigit():
                raise ValueError("Embedded JavaScript string contained an unsupported legacy numeric escape.")
            decoded.append("\0")
            continue
        if escape == "x":
            end = offset + 2
            digits = value[offset:end]
            if len(digits) != 2 or any(character not in hexadecimal for character in digits):
                raise ValueError("Embedded JavaScript string contained an invalid hexadecimal escape.")
            decoded.append(chr(int(digits, 16)))
            offset = end
            continue
        if escape == "u":
            if offset < len(value) and value[offset] == "{":
                digit_start = offset + 1
                digit_end = digit_start
                while digit_end < len(value) and digit_end - digit_start < 6 and value[digit_end] in hexadecimal:
                    digit_end += 1
                if digit_end == digit_start or digit_end >= len(value) or value[digit_end] != "}":
                    raise ValueError("Embedded JavaScript string contained an invalid Unicode code-point escape.")
                code_point = int(value[digit_start:digit_end], 16)
                if code_point > 0x10FFFF or 0xD800 <= code_point <= 0xDFFF:
                    raise ValueError("Embedded JavaScript string contained an invalid Unicode code point.")
                decoded.append(chr(code_point))
                offset = digit_end + 1
                continue

            code_unit, offset = fixed_unicode_code_unit(offset)
            if 0xD800 <= code_unit <= 0xDBFF:
                if value[offset : offset + 2] != "\\u":
                    raise ValueError("Embedded JavaScript string contained an unpaired Unicode surrogate.")
                low_surrogate, low_end = fixed_unicode_code_unit(offset + 2)
                if not 0xDC00 <= low_surrogate <= 0xDFFF:
                    raise ValueError("Embedded JavaScript string contained an unpaired Unicode surrogate.")
                code_point = 0x10000 + ((code_unit - 0xD800) << 10) + (low_surrogate - 0xDC00)
                decoded.append(chr(code_point))
                offset = low_end
                continue
            if 0xDC00 <= code_unit <= 0xDFFF:
                raise ValueError("Embedded JavaScript string contained an unpaired Unicode surrogate.")
            decoded.append(chr(code_unit))
            continue
        if escape.isdigit():
            raise ValueError("Embedded JavaScript string contained an unsupported legacy numeric escape.")

        # ECMAScript NonEscapeCharacter sequences such as ``\-`` evaluate to
        # the escaped character itself. They are decoded explicitly so no
        # source spelling can leak into product-name classification.
        decoded.append(escape)

    return "".join(decoded)


def _scan_js_code(text: str) -> tuple[list[bool], bool]:
    """Mark JavaScript code positions and report whether lexical constructs close."""
    positions = [False] * len(text)
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    regex_literal = False
    regex_character_class = False
    previous_code_character = ""
    penultimate_code_character = ""
    antepenultimate_code_character = ""
    previous_code_was_regex_literal = False
    previous_code_word = ""
    current_code_word = ""
    regex_prefix_keywords = {
        "case",
        "default",
        "delete",
        "do",
        "else",
        "in",
        "instanceof",
        "new",
        "return",
        "throw",
        "typeof",
        "void",
    }
    offset = 0
    while offset < len(text):
        character = text[offset]
        following = text[offset + 1] if offset + 1 < len(text) else ""
        if line_comment:
            if character in "\r\n\u2028\u2029":
                line_comment = False
            offset += 1
            continue
        if block_comment:
            if character == "*" and following == "/":
                block_comment = False
                offset += 2
            else:
                offset += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            offset += 1
            continue
        if regex_literal:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "[":
                regex_character_class = True
            elif character == "]":
                regex_character_class = False
            elif character == "/" and not regex_character_class:
                regex_literal = False
                previous_code_character = "/"
                previous_code_was_regex_literal = True
                current_code_word = ""
                previous_code_word = ""
            offset += 1
            continue

        positions[offset] = True
        if character == "/" and following == "/":
            positions[offset] = False
            line_comment = True
            offset += 2
            continue
        if character == "/" and following == "*":
            positions[offset] = False
            block_comment = True
            offset += 2
            continue
        preceding_code_word = current_code_word.casefold() or previous_code_word
        postfix_update_before_slash = (
            previous_code_character in {"+", "-"}
            and penultimate_code_character == previous_code_character
            and (antepenultimate_code_character.isalnum() or antepenultimate_code_character in "_$)]}'\"`/")
        )
        if (
            character == "/"
            and not postfix_update_before_slash
            and (
                previous_code_character
                in {
                    "",
                    "=",
                    ">",
                    "<",
                    "(",
                    "[",
                    "{",
                    ",",
                    ":",
                    ";",
                    "!",
                    "&",
                    "|",
                    "?",
                    "+",
                    "-",
                    "*",
                    "/",
                    "%",
                    "^",
                    "~",
                }
                and not (previous_code_character == "/" and previous_code_was_regex_literal)
                or preceding_code_word in regex_prefix_keywords
            )
        ):
            positions[offset] = False
            regex_literal = True
            regex_character_class = False
            current_code_word = ""
            previous_code_word = ""
            offset += 1
            continue
        if character in {'"', "'", "`"}:
            quote = character
        if character.isalnum() or character in "_$":
            current_code_word += character
        else:
            if current_code_word:
                previous_code_word = current_code_word.casefold()
                current_code_word = ""
        if not character.isspace():
            previous_code_word = ""
            antepenultimate_code_character = penultimate_code_character
            penultimate_code_character = previous_code_character
            previous_code_character = character
            previous_code_was_regex_literal = False
        offset += 1
    # A line comment is valid through end-of-input.  The other states require a
    # closing delimiter; treating their contents as non-code must not make an
    # otherwise static prefix look like a complete declarative script.
    is_complete = quote is None and not block_comment and not regex_literal
    return positions, is_complete


def _js_code_positions(text: str) -> list[bool]:
    """Mark JavaScript source positions outside strings, comments, and regexes."""
    positions, _is_complete = _scan_js_code(text)
    return positions


def _js_unquoted_positions(text: str) -> list[bool]:
    code_positions = _js_code_positions(text)
    positions = [False] * len(text)
    brace_depth = 0
    bracket_depth = 0
    parenthesis_depth = 0
    for offset, character in enumerate(text):
        if not code_positions[offset]:
            continue
        positions[offset] = brace_depth == 0 and bracket_depth == 0 and parenthesis_depth == 0
        if character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth = max(0, brace_depth - 1)
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif character == "(":
            parenthesis_depth += 1
        elif character == ")":
            parenthesis_depth = max(0, parenthesis_depth - 1)
    return positions


def _js_has_non_comment_slash(text: str) -> bool:
    """Detect regex or division syntax without relying on ambiguous JS slash lexing."""
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    offset = 0
    while offset < len(text):
        character = text[offset]
        following = text[offset + 1] if offset + 1 < len(text) else ""
        if line_comment:
            if character in "\r\n\u2028\u2029":
                line_comment = False
            offset += 1
            continue
        if block_comment:
            if character == "*" and following == "/":
                block_comment = False
                offset += 2
            else:
                offset += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            offset += 1
            continue
        if character in {'"', "'", "`"}:
            quote = character
            offset += 1
            continue
        if character == "/" and following == "/":
            line_comment = True
            offset += 2
            continue
        if character == "/" and following == "*":
            block_comment = True
            offset += 2
            continue
        if character == "/":
            return True
        offset += 1
    return False


def _js_object_bodies(text: str) -> list[str]:
    """Return balanced, uncommented JavaScript object bodies within safe bounds."""
    max_nesting = 64
    max_objects = 10_000
    max_materialized_characters = 8_000_000
    positioned_bodies: list[tuple[int, str]] = []
    materialized_characters = 0
    object_starts: list[int] = []
    code_positions = _js_code_positions(text)
    for offset, character in enumerate(text):
        if not code_positions[offset]:
            continue
        if character == "{":
            if len(object_starts) >= max_nesting:
                raise ValueError("Embedded JavaScript object nesting exceeded the safe parsing limit.")
            object_starts.append(offset)
        elif character == "}" and object_starts:
            start = object_starts.pop()
            body_length = offset - start - 1
            materialized_characters += body_length
            if len(positioned_bodies) >= max_objects or materialized_characters > max_materialized_characters:
                raise ValueError("Embedded JavaScript objects exceeded the safe parsing limit.")
            positioned_bodies.append((start, text[start + 1 : offset]))

    ticker_key_pattern = re.compile(r"(?<![A-Za-z0-9_$])(?:['\"]ticker['\"]|ticker)(?![A-Za-z0-9_$])\s*:")
    for start in object_starts:
        for match in ticker_key_pattern.finditer(text, start + 1):
            if code_positions[match.start()]:
                raise ValueError("Embedded JavaScript contained an unterminated ticker object.")
    return [body for _start, body in sorted(positioned_bodies)]


def _js_object_fields(body: str, field_names: Iterable[str]) -> list[tuple[str, str | None]]:
    field_pattern = "|".join(re.escape(field_name) for field_name in field_names)
    key_regex = re.compile(
        rf"(?<![A-Za-z0-9_$])(?:"
        rf"(?P<keyquote>['\"])(?P<quoted_key>{field_pattern})(?P=keyquote)"
        rf"|(?P<bare_key>{field_pattern})"
        rf")(?![A-Za-z0-9_$])\s*:",
        re.S,
    )
    string_value_regex = re.compile(
        r"\s*(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )
    unquoted_positions = _js_unquoted_positions(body)
    fields: list[tuple[str, str | None]] = []
    for match in key_regex.finditer(body):
        if not unquoted_positions[match.start()]:
            continue
        value_match = string_value_regex.match(body, match.end())
        value: str | None = None
        if value_match is not None:
            following_code = next(
                (
                    offset
                    for offset in range(value_match.end(), len(body))
                    if unquoted_positions[offset] and not body[offset].isspace()
                ),
                None,
            )
            if following_code is None or body[following_code] == ",":
                value = _decode_quoted_js_text(value_match.group("value"))
        fields.append((str(match.group("quoted_key") or match.group("bare_key")), value))
    return fields


def _js_object_property_syntax_risk(body: str) -> tuple[set[str], bool]:
    """Identify alternate object syntax that can change parsed product fields.

    The Themes feed is JavaScript rather than JSON, but product records are
    consumed as static data.  A regex for ordinary ``key: value`` fields must
    not silently disagree with JavaScript about an escaped/computed key, a
    shorthand or method property, a spread, or an object-literal prototype.
    Return the relevant semantic keys that used alternate syntax and whether
    the object contains syntax which can dynamically add or inherit a key.
    """
    relevant_keys = {"ticker", "fund", "name", "externalLink"}
    alternate_keys: set[str] = set()
    dynamic_property_syntax = False
    positions = _js_unquoted_positions(body)
    comma_offsets = [offset for offset, character in enumerate(body) if character == "," and positions[offset]]
    segment_starts = [0, *(offset + 1 for offset in comma_offsets)]
    segment_ends = [*comma_offsets, len(body)]
    quoted_key_pattern = re.compile(
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )
    identifier_escape = r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]{1,6}\})"
    identifier_pattern = re.compile(rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*")

    def next_code_offset(start: int, end: int) -> int | None:
        return next(
            (offset for offset in range(start, end) if positions[offset] and not body[offset].isspace()),
            None,
        )

    for segment_start, segment_end in zip(segment_starts, segment_ends, strict=True):
        key_start = next_code_offset(segment_start, segment_end)
        if key_start is None:
            continue
        segment = body[key_start:segment_end]
        if segment.startswith("..."):
            dynamic_property_syntax = True
            continue
        if segment.startswith("["):
            dynamic_property_syntax = True
            computed_literal = re.match(
                r"\[\s*(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)\s*\]",
                segment,
                re.S,
            )
            if computed_literal is not None:
                decoded_key = _decode_quoted_js_text(computed_literal.group("value"))
                if decoded_key in relevant_keys:
                    alternate_keys.add(decoded_key)
            continue

        quoted_key = quoted_key_pattern.match(body, key_start, segment_end)
        if quoted_key is not None:
            raw_key = quoted_key.group("value")
            decoded_key = _decode_quoted_js_text(raw_key)
            following = next_code_offset(quoted_key.end(), segment_end)
            if decoded_key == "__proto__" and following is not None and body[following] == ":":
                dynamic_property_syntax = True
                continue
            separator = body[quoted_key.end() : following] if following is not None else ""
            if decoded_key in relevant_keys and (
                "\\" in raw_key or following is None or body[following] != ":" or separator.strip()
            ):
                alternate_keys.add(decoded_key)
            continue

        identifier = identifier_pattern.match(body, key_start, segment_end)
        if identifier is None:
            if body[key_start] == "*":
                method_name_start = next_code_offset(key_start + 1, segment_end)
                if method_name_start is not None:
                    if body[method_name_start] in "['\"":
                        dynamic_property_syntax = True
                    else:
                        method_name = identifier_pattern.match(body, method_name_start, segment_end)
                        if method_name is not None:
                            decoded_key = _decode_quoted_js_text(method_name.group())
                            if decoded_key in relevant_keys:
                                alternate_keys.add(decoded_key)
            continue

        raw_key = identifier.group()
        decoded_key = _decode_quoted_js_text(raw_key)
        following = next_code_offset(identifier.end(), segment_end)
        if decoded_key == "__proto__" and following is not None and body[following] == ":":
            dynamic_property_syntax = True
            continue
        separator = body[identifier.end() : following] if following is not None else ""
        if decoded_key in relevant_keys and (
            "\\" in raw_key or following is None or body[following] != ":" or separator.strip()
        ):
            alternate_keys.add(decoded_key)
            continue
        if decoded_key in {"get", "set", "async"} and following is not None and body[following] != ":":
            if body[following] == "*":
                following = next_code_offset(following + 1, segment_end)
            if following is not None:
                if body[following] in "['\"":
                    dynamic_property_syntax = True
                else:
                    method_name = identifier_pattern.match(body, following, segment_end)
                    if method_name is not None:
                        semantic_method_name = _decode_quoted_js_text(method_name.group())
                        if semantic_method_name in relevant_keys:
                            alternate_keys.add(semantic_method_name)

    return alternate_keys, dynamic_property_syntax


def _js_relevant_member_accesses(text: str) -> set[str]:
    """Return product fields accessed outside the supported static-record grammar."""
    relevant_keys = {"ticker", "fund", "name", "externalLink"}
    accesses: set[str] = set()
    positions = _js_code_positions(text)
    identifier_escape = r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]{1,6}\})"
    identifier_pattern = re.compile(rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*")
    quoted_key_pattern = re.compile(
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    for offset, character in enumerate(text):
        if not positions[offset]:
            continue
        if character == ".":
            key_start = next_code_offset(offset + 1)
            if key_start is None:
                continue
            key_match = identifier_pattern.match(text, key_start)
            if key_match is None:
                continue
            decoded_key = _decode_quoted_js_text(key_match.group())
            if decoded_key in relevant_keys:
                accesses.add(decoded_key)
            continue
        if character != "[":
            continue
        key_start = next_code_offset(offset + 1)
        if key_start is None:
            continue
        parenthesis_depth = 0
        while text[key_start] == "(":
            parenthesis_depth += 1
            key_start = next_code_offset(key_start + 1)
            if key_start is None:
                break
        if key_start is None:
            continue
        key_match = quoted_key_pattern.match(text, key_start)
        if key_match is None:
            continue
        bracket = next_code_offset(key_match.end())
        while parenthesis_depth and bracket is not None and text[bracket] == ")":
            parenthesis_depth -= 1
            bracket = next_code_offset(bracket + 1)
        if parenthesis_depth:
            continue
        if bracket is None or text[bracket] != "]":
            continue
        decoded_key = _decode_quoted_js_text(key_match.group("value"))
        if decoded_key in relevant_keys:
            accesses.add(decoded_key)

    return accesses


def _js_has_unresolved_member_access(
    text: str,
    *,
    global_roots_only: bool = False,
    global_root_names: set[str] | None = None,
) -> bool:
    """Return whether code contains a computed member key we cannot statically resolve."""
    positions = _js_code_positions(text)
    global_reference_offsets: set[int] = set()
    if global_roots_only:
        global_roots = global_root_names or {"globalThis", "self", "this", "window"}
        global_references = _js_semantic_identifier_reference_map(
            text,
            global_roots,
        )
        global_reference_offsets = set().union(*global_references.values())
    quoted_key_pattern = re.compile(
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )
    expression_prefix_keywords = {
        "await",
        "case",
        "delete",
        "in",
        "instanceof",
        "new",
        "return",
        "throw",
        "typeof",
        "void",
        "yield",
    }

    def previous_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, -1, -1) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def access_is_in_scope(offset: int) -> bool:
        if not global_roots_only:
            return True
        statement_start = max(
            (candidate for candidate in range(offset - 1, -1, -1) if positions[candidate] and text[candidate] in ";{}"),
            default=-1,
        )
        return any(statement_start < root_offset < offset for root_offset in global_reference_offsets)

    for offset, character in enumerate(text):
        if character != "[" or not positions[offset]:
            continue
        preceding = previous_code_offset(offset - 1)
        if preceding is None:
            continue
        preceding_character = text[preceding]
        is_member_access = (
            preceding_character in ").]'\"" or preceding_character.isalnum() or preceding_character in "_$"
        )
        if preceding_character.isalnum() or preceding_character in "_$":
            word_start = preceding
            while word_start > 0 and (text[word_start - 1].isalnum() or text[word_start - 1] in "_$"):
                word_start -= 1
            if text[word_start : preceding + 1] in expression_prefix_keywords:
                is_member_access = False
        if not is_member_access:
            continue
        if not access_is_in_scope(offset):
            continue
        key_start = next_code_offset(offset + 1)
        if key_start is None:
            return True
        key_match = quoted_key_pattern.match(text, key_start)
        if key_match is None:
            return True
        bracket = next_code_offset(key_match.end())
        if bracket is None or text[bracket] != "]":
            return True

    return False


def _js_semantic_identifier_reference_map(
    text: str,
    identifier_names: set[str],
) -> dict[str, set[int]]:
    """Return code offsets that refer to exact JavaScript identifiers/properties."""
    references = {identifier_name: set() for identifier_name in identifier_names}
    positions = _js_code_positions(text)
    identifier_escape = r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]{1,6}\})"
    identifier_pattern = re.compile(
        rf"(?<![A-Za-z0-9_$\\])(?P<identifier>"
        rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*"
        rf")(?![A-Za-z0-9_$])"
    )
    quoted_key_pattern = re.compile(
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    for match in identifier_pattern.finditer(text):
        if not positions[match.start()]:
            continue
        semantic_identifier = _decode_quoted_js_text(match.group("identifier"))
        if semantic_identifier in references:
            references[semantic_identifier].add(match.start())

    for offset, character in enumerate(text):
        if character != "[" or not positions[offset]:
            continue
        key_start = next_code_offset(offset + 1)
        if key_start is None:
            continue
        key_match = quoted_key_pattern.match(text, key_start)
        if key_match is None:
            continue
        bracket = next_code_offset(key_match.end())
        if bracket is None or text[bracket] != "]":
            continue
        semantic_identifier = _decode_quoted_js_text(key_match.group("value"))
        if semantic_identifier in references:
            references[semantic_identifier].add(offset)

    return references


def _js_semantic_identifier_references(text: str, identifier_name: str) -> set[int]:
    """Return code offsets that refer to one exact JavaScript identifier/property."""
    return _js_semantic_identifier_reference_map(text, {identifier_name})[identifier_name]


def _js_unsafe_dynamic_capabilities(text: str) -> set[str]:
    """Return mutation/evaluation capabilities forbidden in declarative feeds."""
    capability_names = {
        "Function",
        "Object",
        "Reflect",
        "constructor",
        "eval",
        "setInterval",
        "setTimeout",
    }
    references = _js_semantic_identifier_reference_map(text, capability_names)
    return {capability_name for capability_name, offsets in references.items() if offsets}


def _js_ticker_script_is_declarative(
    text: str,
    *,
    lexical_bindings: set[str] | None = None,
    global_properties: set[str] | None = None,
) -> bool:
    """Accept only static literal statements supported by the Themes parser."""
    positions, is_lexically_complete = _scan_js_code(text)
    if not is_lexically_complete:
        return False
    identifier_pattern = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
    reserved_binding_names = {
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "else",
        "enum",
        "export",
        "extends",
        "false",
        "finally",
        "for",
        "function",
        "if",
        "implements",
        "import",
        "in",
        "instanceof",
        "interface",
        "let",
        "new",
        "null",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "static",
        "super",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "typeof",
        "var",
        "void",
        "while",
        "with",
        "yield",
    }

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def identifier_at(start: int) -> tuple[str, int] | None:
        match = identifier_pattern.match(text, start)
        if match is None:
            return None
        return match.group(), match.end()

    def balanced_literal_end(offset: int) -> int | None:
        closing_delimiters = {"(": ")", "[": "]", "{": "}"}
        if offset >= len(text) or text[offset] not in "[{":
            return None
        stack = [closing_delimiters[text[offset]]]
        for candidate in range(offset + 1, len(text)):
            if not positions[candidate]:
                continue
            character = text[candidate]
            if character in closing_delimiters:
                stack.append(closing_delimiters[character])
            elif character in ")]}":
                if not stack or character != stack.pop():
                    return None
                if not stack:
                    return candidate + 1
        return None

    def static_scalar_object(start: int, end: int) -> bool:
        object_close = end - 1
        quoted_text_pattern = re.compile(
            r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
            re.S,
        )
        # A leading sign is a unary expression, not part of JavaScript's
        # NumericLiteral grammar.  This parser intentionally accepts data
        # literals only, with no operators even when their result is constant.
        number_pattern = re.compile(r"(?:0|[1-9]\d*)(?:\.\d+)?(?:[Ee][+-]?\d+)?")
        cursor = next_code_offset(start + 1)
        while cursor is not None and cursor < object_close:
            quoted_key = quoted_text_pattern.match(text, cursor, object_close)
            if quoted_key is not None:
                try:
                    _decode_quoted_js_text(quoted_key.group("value"))
                except ValueError:
                    return False
                key_end = quoted_key.end()
            else:
                bare_key = identifier_at(cursor)
                if bare_key is None:
                    return False
                key_end = bare_key[1]
            colon = next_code_offset(key_end)
            if colon is None or colon >= object_close or text[colon] != ":":
                return False
            value_start = next_code_offset(colon + 1)
            if value_start is None or value_start >= object_close:
                return False
            quoted_value = quoted_text_pattern.match(text, value_start, object_close)
            if quoted_value is not None:
                try:
                    _decode_quoted_js_text(quoted_value.group("value"))
                except ValueError:
                    return False
                value_end = quoted_value.end()
            else:
                scalar_value = re.match(r"(?:true|false|null)(?![A-Za-z0-9_$])", text[value_start:object_close])
                if scalar_value is not None:
                    value_end = value_start + scalar_value.end()
                else:
                    numeric_value = number_pattern.match(text, value_start, object_close)
                    if numeric_value is None:
                        return False
                    value_end = numeric_value.end()
            delimiter = next_code_offset(value_end)
            if delimiter == object_close:
                return True
            if delimiter is None or delimiter > object_close or text[delimiter] != ",":
                return False
            cursor = next_code_offset(delimiter + 1)
        return cursor == object_close

    def static_product_literal(start: int, end: int) -> bool:
        if text[start] == "{":
            body = text[start + 1 : end - 1]
            return bool(_js_object_fields(body, ["ticker"])) and static_scalar_object(start, end)

        array_close = end - 1
        cursor = next_code_offset(start + 1)
        while cursor is not None and cursor < array_close:
            if text[cursor] != "{":
                return False
            item_end = balanced_literal_end(cursor)
            if item_end is None or item_end > array_close:
                return False
            if not static_scalar_object(cursor, item_end):
                return False
            following = next_code_offset(item_end)
            if following == array_close:
                return True
            if following is None or following > array_close or text[following] != ",":
                return False
            cursor = next_code_offset(following + 1)
        return cursor == array_close

    def literal_end(start: int) -> int | None:
        offset = start
        wrapper_depth = 0
        while offset < len(text) and text[offset] == "(":
            wrapper_depth += 1
            following = next_code_offset(offset + 1)
            if following is None:
                return None
            offset = following
        unwrapped_end = balanced_literal_end(offset)
        if unwrapped_end is None or not static_product_literal(offset, unwrapped_end):
            return None
        literal_following = unwrapped_end
        for _wrapper in range(wrapper_depth):
            closing_wrapper = next_code_offset(literal_following)
            if closing_wrapper is None or text[closing_wrapper] != ")":
                return None
            literal_following = closing_wrapper + 1
        return literal_following

    declared_bindings: set[str] = set()
    assigned_global_properties: set[str] = set()

    def accept() -> bool:
        if lexical_bindings is not None:
            lexical_bindings.update(declared_bindings)
        if global_properties is not None:
            global_properties.update(assigned_global_properties)
        return True

    offset = next_code_offset(0)
    while offset is not None:
        if text[offset] == ";":
            offset = next_code_offset(offset + 1)
            continue

        value_start = offset
        first_identifier = identifier_at(offset)
        if first_identifier is not None and first_identifier[0] in {"const", "let", "var"}:
            declaration_kind = first_identifier[0]
            binding_start = next_code_offset(first_identifier[1])
            if binding_start is None:
                return False
            binding = identifier_at(binding_start)
            if binding is None or binding[0] in reserved_binding_names:
                return False
            binding_name = binding[0]
            if binding_name in declared_bindings or (
                declaration_kind == "var" and binding_name in assigned_global_properties
            ):
                return False
            declared_bindings.add(binding_name)
            # A top-level ``var`` declaration in a classic script initializes
            # a property on the same global object addressed by ``window`` and
            # its aliases. ``let`` and ``const`` remain lexical-only bindings.
            if declaration_kind == "var":
                assigned_global_properties.add(binding_name)
            equals = next_code_offset(binding[1])
            if equals is None or text[equals] != "=":
                return False
            value_start = next_code_offset(equals + 1)
            if value_start is None:
                return False
        elif first_identifier is not None and first_identifier[0] in {"globalThis", "self", "this", "window"}:
            separator = next_code_offset(first_identifier[1])
            if separator is None or text[separator] != ".":
                return False
            property_start = next_code_offset(separator + 1)
            if property_start is None:
                return False
            property_name = identifier_at(property_start)
            if property_name is None:
                return False
            if property_name[0] in assigned_global_properties:
                return False
            # ``window``, ``self``, ``globalThis``, and top-level ``this`` are
            # aliases for the same browser global in the classic scripts this
            # parser consumes.  A second assignment through another spelling
            # is still a rebind, not another static declaration.
            assigned_global_properties.add(property_name[0])
            equals = next_code_offset(property_name[1])
            if equals is None or text[equals] != "=":
                return False
            value_start = next_code_offset(equals + 1)
            if value_start is None:
                return False

        following = literal_end(value_start)
        if following is None:
            return False
        statement_end = next_code_offset(following)
        if statement_end is None:
            return accept()
        if text[statement_end] != ";":
            return False
        offset = next_code_offset(statement_end + 1)

    return accept()


def _js_cross_script_inventory_references(
    text: str,
    *,
    lexical_names: set[str],
    global_names: set[str],
) -> set[str]:
    """Find references to another script's static inventory bindings."""
    if not lexical_names and not global_names:
        return set()

    positions = _js_code_positions(text)
    identifier_escape = r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]{1,6}\})"
    identifier_pattern = re.compile(
        rf"(?<![A-Za-z0-9_$\\])(?P<identifier>"
        rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*"
        rf")(?![A-Za-z0-9_$])"
    )
    quoted_key_pattern = re.compile(
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )
    global_roots = {"globalThis", "self", "this", "window"}
    references: set[str] = set()

    def previous_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, -1, -1) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def identifier_before(end: int) -> tuple[str, int] | None:
        identifier_end = previous_code_offset(end)
        if identifier_end is None:
            return None
        match = next(
            (
                candidate
                for candidate in identifier_pattern.finditer(text, 0, identifier_end + 1)
                if candidate.end() == identifier_end + 1 and positions[candidate.start()]
            ),
            None,
        )
        if match is None:
            return None
        return _decode_quoted_js_text(match.group("identifier")), match.start()

    # Preserve aliases only when a bare global root is the complete right-hand
    # side of a simple assignment.  A bare root passed to analytics, logging, or
    # feature detection does not by itself touch every property on the browser
    # global, while an alias that is later used as ``target.rows`` still must be
    # recognized as access to a cross-script inventory.
    aliases_changed = True
    while aliases_changed:
        aliases_changed = False
        for root_match in identifier_pattern.finditer(text):
            if not positions[root_match.start()]:
                continue
            root_name = _decode_quoted_js_text(root_match.group("identifier"))
            if root_name not in global_roots:
                continue
            equals = previous_code_offset(root_match.start() - 1)
            if equals is None or text[equals] != "=":
                continue
            before_equals = previous_code_offset(equals - 1)
            after_equals = next_code_offset(equals + 1)
            following = next_code_offset(root_match.end())
            if (
                before_equals is None
                or after_equals != root_match.start()
                or following is not None
                and text[following] not in {",", ";", ")", "}"}
                or text[before_equals] in {"=", "!", "<", ">"}
                or following is not None
                and text[following] in {"=", ">"}
            ):
                continue
            alias = identifier_before(equals - 1)
            if alias is None:
                continue
            alias_name, alias_start = alias
            if alias_name in global_roots:
                continue
            alias_preceding = previous_code_offset(alias_start - 1)
            if alias_preceding is not None and text[alias_preceding] in {".", "]"}:
                continue
            global_roots.add(alias_name)
            aliases_changed = True

    for match in identifier_pattern.finditer(text):
        if not positions[match.start()]:
            continue
        name = _decode_quoted_js_text(match.group("identifier"))
        if name not in lexical_names and name not in global_names:
            continue
        preceding = previous_code_offset(match.start() - 1)
        if preceding is not None and text[preceding] == ".":
            root = identifier_before(preceding - 1)
            if name in (lexical_names | global_names) and root is not None and root[0] in global_roots:
                references.add(name)
            continue
        following = next_code_offset(match.end())
        # A bare object key or statement label does not read or replace a
        # binding with the same spelling.
        if following is not None and text[following] == ":":
            continue
        references.add(name)

    for root_match in identifier_pattern.finditer(text):
        if (
            not positions[root_match.start()]
            or _decode_quoted_js_text(root_match.group("identifier")) not in global_roots
        ):
            continue
        separator = next_code_offset(root_match.end())
        if separator is None or text[separator] not in ".[":
            continue
        if text[separator] != "[":
            continue
        key_start = next_code_offset(separator + 1)
        if key_start is None:
            continue
        key_match = quoted_key_pattern.match(text, key_start)
        if key_match is None:
            references.update(global_names)
            continue
        bracket_close = next_code_offset(key_match.end())
        if bracket_close is None or text[bracket_close] != "]":
            references.update(global_names)
            continue
        name = _decode_quoted_js_text(key_match.group("value"))
        if name in global_names:
            references.add(name)

    return references


def _js_dynamic_timer_inventory_reference(
    text: str,
    *,
    lexical_names: set[str],
    global_names: set[str],
) -> bool:
    """Detect timer-dispatched source that can reach a shared inventory."""
    timer_names = {"setInterval", "setTimeout"}
    timer_offsets = _js_semantic_identifier_reference_map(text, timer_names)
    positions = _js_code_positions(text)
    quoted_text_pattern = re.compile(
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    for timer_name, offsets in timer_offsets.items():
        for offset in offsets:
            call_start = next_code_offset(offset + len(timer_name))
            # Passing a timer capability around can hide a later string dispatch;
            # retain the conservative rejection for that indirect form.
            if call_start is None or text[call_start] != "(":
                return True
            argument_start = next_code_offset(call_start + 1)
            if argument_start is None or text[argument_start] == ")":
                continue
            if text[argument_start] == "`":
                return True
            if text[argument_start] not in {'"', "'"}:
                # Function and identifier callbacks remain visible to the
                # ordinary cross-script reference scan.
                continue
            literal = quoted_text_pattern.match(text, argument_start)
            if literal is None:
                return True
            try:
                dispatched_source = _decode_quoted_js_text(literal.group("value"))
            except ValueError:
                return True
            following = next_code_offset(literal.end())
            if following is None or text[following] not in {",", ")"}:
                # Concatenation or another computed first argument is executable
                # source whose eventual target cannot be proven unrelated.
                return True
            if _js_cross_script_inventory_references(
                dispatched_source,
                lexical_names=lexical_names,
                global_names=global_names,
            ):
                return True
    return False


def _js_has_object_prototype_mutation(text: str) -> bool:
    """Reject writes that can synthesize inherited product-record fields.

    This is deliberately a small semantic scan rather than a source regex.
    Prototype and intrinsic-method aliases are common in bundled JavaScript,
    and comma declarations or destructuring must not hide a mutation from the
    static product parser.  The scan only treats known mutators as dangerous,
    so harmless uses such as ``const O = Object; O.keys(item)`` remain valid.
    """
    positions = _js_code_positions(text)
    identifier_escape = r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]{1,6}\})"
    identifier_pattern = re.compile(rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*")
    quoted_text_pattern = re.compile(
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )

    # (kind, semantic value, source start, source end).  Comments, regexes,
    # whitespace, and string contents are excluded by ``positions``; quoted
    # property names are retained as a single decoded token.
    tokens: list[tuple[str, str, int, int]] = []
    offset = 0
    while offset < len(text):
        if not positions[offset] or text[offset].isspace():
            offset += 1
            continue
        if text[offset] in {'"', "'"}:
            quoted_match = quoted_text_pattern.match(text, offset)
            if quoted_match is not None:
                try:
                    value = _decode_quoted_js_text(quoted_match.group("value"))
                except ValueError:
                    value = quoted_match.group("value")
                tokens.append(("string", value, offset, quoted_match.end()))
                offset = quoted_match.end()
                continue
        identifier_match = identifier_pattern.match(text, offset)
        if identifier_match is not None:
            tokens.append(
                (
                    "identifier",
                    _decode_quoted_js_text(identifier_match.group()),
                    offset,
                    identifier_match.end(),
                )
            )
            offset = identifier_match.end()
            continue
        tokens.append(("punctuation", text[offset], offset, offset + 1))
        offset += 1

    if not tokens:
        return False

    opening_delimiters = {"(": ")", "[": "]", "{": "}"}
    delimiter_stack: list[tuple[str, int]] = []
    matching_delimiters: dict[int, int] = {}
    for token_index, (_kind, value, _start, _end) in enumerate(tokens):
        if value in opening_delimiters:
            delimiter_stack.append((value, token_index))
        elif value in opening_delimiters.values() and delimiter_stack:
            opening, opening_index = delimiter_stack[-1]
            if opening_delimiters[opening] == value:
                delimiter_stack.pop()
                matching_delimiters[opening_index] = token_index
                matching_delimiters[token_index] = opening_index

    object_mutators = {"assign", "defineProperties", "defineProperty", "setPrototypeOf"}
    reflect_mutators = {"defineProperty", "set", "setPrototypeOf"}
    legacy_mutators = {"__defineGetter__", "__defineSetter__"}
    object_aliases = {"Object"}
    reflect_aliases = {"Reflect"}
    prototype_aliases: set[str] = set()
    mutator_aliases: set[str] = set()
    prototype_getter_aliases: set[str] = set()

    def member_at(separator: int) -> tuple[str | None, int] | None:
        if separator >= len(tokens):
            return None
        if tokens[separator][1] == ".":
            member_index = separator + 1
            if member_index < len(tokens) and tokens[member_index][0] == "identifier":
                return tokens[member_index][1], member_index + 1
            return None
        if tokens[separator][1] != "[":
            return None
        closing = matching_delimiters.get(separator)
        if closing is None:
            return None
        # Resolve only a statically complete string expression.  Concatenated
        # literals are common in minified code and must not turn
        # ``Object['pro' + 'totype']`` into an unknown, therefore harmless,
        # member.  Any other computed expression remains unresolved.
        member_parts: list[str] = []
        cursor = separator + 1
        expect_string = True
        while cursor < closing:
            if expect_string:
                if tokens[cursor][0] != "string":
                    return None, closing + 1
                member_parts.append(tokens[cursor][1])
            elif tokens[cursor][1] != "+":
                return None, closing + 1
            expect_string = not expect_string
            cursor += 1
        if member_parts and not expect_string:
            return "".join(member_parts), closing + 1
        return None, closing + 1

    def enclosing_brace_scope(token_index: int) -> tuple[int, int]:
        containing = [
            (opening, closing)
            for opening, closing in matching_delimiters.items()
            if opening < token_index < closing and tokens[opening][1] == "{"
        ]
        if not containing:
            return 0, len(tokens)
        opening, closing = max(containing, key=lambda bounds: bounds[0])
        return opening + 1, closing

    def parameter_list_binds_object(opening: int, closing: int) -> bool:
        """Recognize Object as a direct parameter binding, not a default RHS."""
        segment_start = opening + 1
        cursor = segment_start
        while cursor <= closing:
            at_end = cursor == closing
            if not at_end and tokens[cursor][1] in opening_delimiters:
                nested_end = matching_delimiters.get(cursor)
                if nested_end is not None and nested_end < closing:
                    cursor = nested_end + 1
                    continue
            if at_end or tokens[cursor][1] == ",":
                binding = segment_start
                while binding < cursor and tokens[binding][1] == ".":
                    binding += 1
                if binding < cursor and tokens[binding][0] == "identifier" and tokens[binding][1] == "Object":
                    return True
                segment_start = cursor + 1
            cursor += 1
        return False

    # The intrinsic name may be shadowed inside a function/catch/arrow scope or
    # by a local declaration.  Treating every token spelled ``Object`` as the
    # built-in creates false prototype-mutation reports for ordinary callbacks.
    object_shadow_ranges: list[tuple[int, int]] = []
    for token_index, (_kind, value, _start, _end) in enumerate(tokens):
        if value == "function":
            parameter_open = token_index + 1
            while parameter_open < len(tokens) and tokens[parameter_open][1] not in {"(", "{", ";"}:
                parameter_open += 1
            if parameter_open >= len(tokens) or tokens[parameter_open][1] != "(":
                continue
            parameter_close = matching_delimiters.get(parameter_open)
            if parameter_close is None or not parameter_list_binds_object(parameter_open, parameter_close):
                continue
            body_open = parameter_close + 1
            if body_open < len(tokens) and tokens[body_open][1] == "{":
                body_close = matching_delimiters.get(body_open)
                if body_close is not None:
                    object_shadow_ranges.append((body_open + 1, body_close))
        elif value == "catch" and token_index + 1 < len(tokens) and tokens[token_index + 1][1] == "(":
            parameter_open = token_index + 1
            parameter_close = matching_delimiters.get(parameter_open)
            if parameter_close is None or not parameter_list_binds_object(parameter_open, parameter_close):
                continue
            body_open = parameter_close + 1
            if body_open < len(tokens) and tokens[body_open][1] == "{":
                body_close = matching_delimiters.get(body_open)
                if body_close is not None:
                    object_shadow_ranges.append((body_open + 1, body_close))
        elif value in {"const", "let", "var", "class"}:
            binding = token_index + 1
            if binding < len(tokens) and tokens[binding][0] == "identifier" and tokens[binding][1] == "Object":
                object_shadow_ranges.append(enclosing_brace_scope(token_index))

    def object_name_is_intrinsic_or_alias(name: str, token_index: int) -> bool:
        return name in object_aliases and not (
            name == "Object" and any(start <= token_index < end for start, end in object_shadow_ranges)
        )

    def strip_parentheses(start: int, end: int) -> tuple[int, int]:
        while start < end and tokens[start][1] == "(" and matching_delimiters.get(start) == end - 1:
            start += 1
            end -= 1
        return start, end

    def expression_end(start: int) -> int:
        stack: list[str] = []
        for token_index in range(start, len(tokens)):
            value = tokens[token_index][1]
            if value in opening_delimiters:
                stack.append(value)
                continue
            if value in opening_delimiters.values():
                if stack and opening_delimiters[stack[-1]] == value:
                    stack.pop()
                    continue
                return token_index
            if not stack and value in {",", ";"}:
                return token_index
        return len(tokens)

    for token_index in range(1, len(tokens) - 1):
        if tokens[token_index][1] != "=" or tokens[token_index + 1][1] != ">":
            continue
        parameter_end = token_index
        if tokens[parameter_end - 1][0] == "identifier":
            binds_object = tokens[parameter_end - 1][1] == "Object"
        elif tokens[parameter_end - 1][1] == ")":
            parameter_open = matching_delimiters.get(parameter_end - 1)
            binds_object = parameter_open is not None and parameter_list_binds_object(parameter_open, parameter_end - 1)
        else:
            binds_object = False
        if not binds_object:
            continue
        body_start = token_index + 2
        if body_start < len(tokens) and tokens[body_start][1] == "{":
            body_end = matching_delimiters.get(body_start)
            if body_end is not None:
                object_shadow_ranges.append((body_start + 1, body_end))
        else:
            object_shadow_ranges.append((body_start, expression_end(body_start)))

    def classified_expression(start: int, end: int) -> str | None:
        start, end = strip_parentheses(start, end)
        if start >= end:
            return None
        if end == start + 1 and tokens[start][0] == "identifier":
            name = tokens[start][1]
            if object_name_is_intrinsic_or_alias(name, start):
                return "object"
            if name in reflect_aliases:
                return "reflect"
            if name in prototype_aliases:
                return "prototype"
            if name in mutator_aliases:
                return "mutator"
            if name in prototype_getter_aliases:
                return "prototype_getter"

        if tokens[start][0] == "identifier":
            root = tokens[start][1]
            root_is_object = object_name_is_intrinsic_or_alias(root, start)
            root_is_reflect = root in reflect_aliases
            member = member_at(start + 1)
            if member is not None:
                member_name, member_end = member
                if (root_is_object or root_is_reflect) and member_name == "prototype" and member_end == end:
                    return "prototype"
                root_mutators = object_mutators if root_is_object else reflect_mutators
                if (root_is_object or root_is_reflect) and member_name in root_mutators:
                    if member_end == end:
                        return "mutator"
                    bind_member = member_at(member_end)
                    if bind_member is not None and bind_member[0] == "bind":
                        call_start = bind_member[1]
                        if (
                            call_start < end
                            and tokens[call_start][1] == "("
                            and matching_delimiters.get(call_start) == end - 1
                        ):
                            return "mutator"
                if (root_is_object or root_is_reflect) and member_name == "getPrototypeOf":
                    if (
                        member_end < end
                        and tokens[member_end][1] == "("
                        and matching_delimiters.get(member_end) == end - 1
                    ):
                        return "prototype"
                    if member_end == end:
                        return "prototype_getter"

            if (
                root in prototype_getter_aliases
                and start + 1 < end
                and tokens[start + 1][1] == "("
                and matching_delimiters.get(start + 1) == end - 1
            ):
                return "prototype"

        # A grouped intrinsic remains the receiver of the following member:
        # ``(Object).prototype`` and ``(Object).defineProperty(...)`` are
        # semantically identical to their ungrouped forms.
        if tokens[start][1] == "(":
            grouped_close = matching_delimiters.get(start)
            grouped_end = grouped_close + 1 if grouped_close is not None else None
            if grouped_end is not None and grouped_end < end:
                grouped_kind = classified_expression(start, grouped_end)
                member = member_at(grouped_end)
                if member is not None:
                    member_name, member_end = member
                    if grouped_kind in {"object", "reflect"} and member_name == "prototype" and member_end == end:
                        return "prototype"
                    grouped_mutators = object_mutators if grouped_kind == "object" else reflect_mutators
                    if grouped_kind in {"object", "reflect"} and member_name in grouped_mutators:
                        if member_end == end:
                            return "mutator"
                        bind_member = member_at(member_end)
                        if bind_member is not None and bind_member[0] == "bind":
                            call_start = bind_member[1]
                            if (
                                call_start < end
                                and tokens[call_start][1] == "("
                                and matching_delimiters.get(call_start) == end - 1
                            ):
                                return "mutator"
                    if grouped_kind in {"object", "reflect"} and member_name == "getPrototypeOf":
                        if (
                            member_end < end
                            and tokens[member_end][1] == "("
                            and matching_delimiters.get(member_end) == end - 1
                        ):
                            return "prototype"
                        if member_end == end:
                            return "prototype_getter"

        # ``({}).__proto__`` and equivalent static bracket notation expose a
        # prototype without naming Object.  Any such aliased prototype is a
        # dangerous mutation target, even when the base expression is wrapped.
        if end - start >= 2 and tokens[end - 2][1] == "." and tokens[end - 1][1] == "__proto__":
            return "prototype"
        if (
            end - start >= 3
            and tokens[end - 3][1] == "["
            and tokens[end - 2][0] == "string"
            and tokens[end - 2][1] == "__proto__"
            and tokens[end - 1][1] == "]"
        ):
            return "prototype"
        return None

    def simple_assignment_at(token_index: int) -> bool:
        if tokens[token_index][1] != "=":
            return False
        preceding = tokens[token_index - 1][1] if token_index else ""
        following = tokens[token_index + 1][1] if token_index + 1 < len(tokens) else ""
        return preceding not in {"=", "!", "<", ">"} and following not in {"=", ">"}

    def add_alias(alias: str, classification: str | None) -> bool:
        aliases = {
            "object": object_aliases,
            "reflect": reflect_aliases,
            "prototype": prototype_aliases,
            "mutator": mutator_aliases,
            "prototype_getter": prototype_getter_aliases,
        }.get(classification)
        if aliases is None or alias in aliases:
            return False
        aliases.add(alias)
        return True

    # Alias sources can themselves be aliases and can appear together in one
    # comma declaration, so propagate to a fixed point.  This remains bounded
    # by the number of identifiers in the input.
    aliases_changed = True
    while aliases_changed:
        aliases_changed = False
        for token_index in range(1, len(tokens) - 1):
            if not simple_assignment_at(token_index) or tokens[token_index - 1][0] != "identifier":
                continue
            lhs_index = token_index - 1
            if lhs_index and tokens[lhs_index - 1][1] in {".", "["}:
                continue
            rhs_start = token_index + 1
            rhs_end = expression_end(rhs_start)
            aliases_changed = (
                add_alias(
                    tokens[lhs_index][1],
                    classified_expression(rhs_start, rhs_end),
                )
                or aliases_changed
            )

        # Handle both declarations and assignment-pattern destructuring, with
        # optional local renaming: ``{defineProperty: define} = Object``.
        for token_index in range(1, len(tokens) - 1):
            if not simple_assignment_at(token_index) or tokens[token_index - 1][1] != "}":
                continue
            object_end = token_index - 1
            object_start = matching_delimiters.get(object_end)
            if object_start is None or tokens[object_start][1] != "{":
                continue
            source_kind = classified_expression(token_index + 1, expression_end(token_index + 1))
            if source_kind not in {"object", "reflect"}:
                continue
            entry_start = object_start + 1
            while entry_start < object_end:
                entry_end = entry_start
                nested_depth = 0
                while entry_end < object_end:
                    value = tokens[entry_end][1]
                    if value in opening_delimiters:
                        nested_depth += 1
                    elif value in opening_delimiters.values():
                        nested_depth = max(0, nested_depth - 1)
                    elif value == "," and nested_depth == 0:
                        break
                    entry_end += 1
                if entry_start < entry_end and tokens[entry_start][0] in {"identifier", "string"}:
                    imported_name = tokens[entry_start][1]
                    local_name = imported_name
                    if (
                        entry_start + 2 < entry_end
                        and tokens[entry_start + 1][1] == ":"
                        and tokens[entry_start + 2][0] == "identifier"
                    ):
                        local_name = tokens[entry_start + 2][1]
                    if imported_name == "prototype" and source_kind == "object":
                        aliases_changed = add_alias(local_name, "prototype") or aliases_changed
                    elif imported_name == "getPrototypeOf":
                        aliases_changed = add_alias(local_name, "prototype_getter") or aliases_changed
                    elif imported_name in (object_mutators if source_kind == "object" else reflect_mutators):
                        aliases_changed = add_alias(local_name, "mutator") or aliases_changed
                entry_start = entry_end + 1

    def mutation_operator_at(token_index: int) -> bool:
        if token_index >= len(tokens):
            return False
        first = tokens[token_index][1]
        second = tokens[token_index + 1][1] if token_index + 1 < len(tokens) else ""
        third = tokens[token_index + 2][1] if token_index + 2 < len(tokens) else ""
        if first == "=":
            return second not in {"=", ">"}
        if first in {"+", "-"} and second == first:
            return True
        if first in {"+", "-", "*", "/", "%", "&", "|", "^"} and second == "=":
            return True
        return first + second + third in {"**=", "&&=", "||=", "??="}

    prototype_ranges: set[tuple[int, int, bool]] = set()
    for token_index, (kind, name, _start, _end) in enumerate(tokens):
        preceding = tokens[token_index - 1][1] if token_index else ""
        is_standalone_identifier = kind == "identifier" and preceding != "."
        if is_standalone_identifier and name in prototype_aliases:
            prototype_ranges.add((token_index, token_index + 1, False))
        if is_standalone_identifier and object_name_is_intrinsic_or_alias(name, token_index):
            member = member_at(token_index + 1)
            if member is not None and member[0] == "prototype":
                prototype_ranges.add((token_index, member[1], True))
            elif member is not None and member[0] == "getPrototypeOf":
                call_start = member[1]
                if call_start < len(tokens) and tokens[call_start][1] == "(":
                    call_end = matching_delimiters.get(call_start)
                    if call_end is not None:
                        prototype_ranges.add((token_index, call_end + 1, False))
        if is_standalone_identifier and name in reflect_aliases:
            member = member_at(token_index + 1)
            if member is not None and member[0] == "getPrototypeOf":
                call_start = member[1]
                if call_start < len(tokens) and tokens[call_start][1] == "(":
                    call_end = matching_delimiters.get(call_start)
                    if call_end is not None:
                        prototype_ranges.add((token_index, call_end + 1, False))
        if is_standalone_identifier and name in prototype_getter_aliases:
            call_start = token_index + 1
            if call_start < len(tokens) and tokens[call_start][1] == "(":
                call_end = matching_delimiters.get(call_start)
                if call_end is not None:
                    prototype_ranges.add((token_index, call_end + 1, False))

        if tokens[token_index][1] == "(":
            grouped_close = matching_delimiters.get(token_index)
            grouped_end = grouped_close + 1 if grouped_close is not None else None
            if grouped_end is not None:
                grouped_kind = classified_expression(token_index, grouped_end)
                if grouped_kind == "prototype":
                    prototype_ranges.add((token_index, grouped_end, False))
                elif grouped_kind == "object":
                    grouped_member = member_at(grouped_end)
                    if grouped_member is not None and grouped_member[0] == "prototype":
                        prototype_ranges.add((token_index, grouped_member[1], True))

        member = member_at(token_index)
        if member is not None and member[0] == "__proto__":
            base_start = token_index - 1
            if base_start >= 0 and tokens[base_start][1] in opening_delimiters.values():
                base_start = matching_delimiters.get(base_start, base_start)
            if base_start >= 0:
                prototype_ranges.add((base_start, member[1], False))

    # Preserve prototype identity through harmless grouping parentheses so a
    # write through ``(p)[key]`` cannot evade the alias scan.
    ranges_changed = True
    while ranges_changed:
        ranges_changed = False
        for start, end, replaceable in tuple(prototype_ranges):
            if start and tokens[start - 1][1] == "(" and matching_delimiters.get(start - 1) == end:
                wrapped = (start - 1, end + 1, replaceable)
                if wrapped not in prototype_ranges:
                    prototype_ranges.add(wrapped)
                    ranges_changed = True

    for start, end, replaceable in prototype_ranges:
        if replaceable and mutation_operator_at(end):
            return True
        access = member_at(end)
        if access is None:
            continue
        property_name, property_end = access
        if mutation_operator_at(property_end):
            return True
        if property_name in legacy_mutators and property_end < len(tokens) and tokens[property_end][1] == "(":
            return True
        if start and tokens[start - 1][0] == "identifier" and tokens[start - 1][1] == "delete":
            return True

    def argument_ranges(call_start: int, call_end: int) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        argument_start = call_start + 1
        nested_depth = 0
        for token_index in range(argument_start, call_end):
            value = tokens[token_index][1]
            if value in opening_delimiters:
                nested_depth += 1
            elif value in opening_delimiters.values():
                nested_depth = max(0, nested_depth - 1)
            elif value == "," and nested_depth == 0:
                ranges.append((argument_start, token_index))
                argument_start = token_index + 1
        if argument_start < call_end:
            ranges.append((argument_start, call_end))
        return ranges

    mutator_callees: set[tuple[int, int]] = set()
    for token_index, (kind, name, _start, _end) in enumerate(tokens):
        preceding = tokens[token_index - 1][1] if token_index else ""
        if kind != "identifier" or preceding == ".":
            continue
        if name in mutator_aliases:
            mutator_callees.add((token_index, token_index + 1))
        name_is_object = object_name_is_intrinsic_or_alias(name, token_index)
        if not name_is_object and name not in reflect_aliases:
            continue
        member = member_at(token_index + 1)
        if member is None:
            continue
        valid_methods = object_mutators if name_is_object else reflect_mutators
        if member[0] in valid_methods:
            mutator_callees.add((token_index, member[1]))

    for token_index, (_kind, value, _start, _end) in enumerate(tokens):
        if value != "(":
            continue
        grouped_close = matching_delimiters.get(token_index)
        grouped_end = grouped_close + 1 if grouped_close is not None else None
        if grouped_end is None:
            continue
        grouped_kind = classified_expression(token_index, grouped_end)
        if grouped_kind not in {"object", "reflect"}:
            continue
        member = member_at(grouped_end)
        if member is None:
            continue
        valid_methods = object_mutators if grouped_kind == "object" else reflect_mutators
        if member[0] in valid_methods:
            mutator_callees.add((token_index, member[1]))

    callees_changed = True
    while callees_changed:
        callees_changed = False
        for start, end in tuple(mutator_callees):
            if start and tokens[start - 1][1] == "(" and matching_delimiters.get(start - 1) == end:
                wrapped = (start - 1, end + 1)
                if wrapped not in mutator_callees:
                    mutator_callees.add(wrapped)
                    callees_changed = True

    for _callee_start, callee_end in mutator_callees:
        target_argument_index = 0
        call_member = member_at(callee_end)
        if call_member is not None and call_member[0] in {"call", "apply"}:
            target_argument_index = 1
            indirect_kind = call_member[0]
            callee_end = call_member[1]
        else:
            indirect_kind = None
        if callee_end >= len(tokens) or tokens[callee_end][1] != "(":
            continue
        call_end = matching_delimiters.get(callee_end)
        if call_end is None:
            continue
        arguments = argument_ranges(callee_end, call_end)
        if target_argument_index >= len(arguments):
            continue
        if indirect_kind == "apply":
            # The target is the first element of apply's argument array.  If
            # that array is dynamic, conservatively reject any visible
            # prototype value rather than pretending to reproduce execution.
            argument_start, argument_end = arguments[target_argument_index]
            if any(
                classified_expression(candidate, argument_end) == "prototype"
                for candidate in range(argument_start, argument_end)
            ):
                return True
            continue
        argument_start, argument_end = arguments[target_argument_index]
        if classified_expression(argument_start, argument_end) == "prototype":
            return True

    return False


def _js_semantic_mutation_api_properties(
    text: str,
    *,
    object_aliases: set[str] | None = None,
    reflect_aliases: set[str] | None = None,
) -> set[tuple[str, str | None]]:
    """Return exact properties targeted through supported semantic mutation APIs.

    ``None`` represents a computed property argument. Static issuer records may
    not rely on a computed mutation because its effect cannot be reproduced by
    this parser without executing untrusted source JavaScript.
    """
    mutations: set[tuple[str, str | None]] = set()
    positions = _js_code_positions(text)
    identifier_escape = r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]{1,6}\})"
    identifier_pattern = re.compile(
        rf"(?<![A-Za-z0-9_$\\])(?P<identifier>"
        rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*"
        rf")(?![A-Za-z0-9_$])"
    )
    identifier_at_offset_pattern = re.compile(
        rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*"
    )
    quoted_key_pattern = re.compile(
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    object_roots = object_aliases or {"Object"}
    reflect_roots = reflect_aliases or {"Reflect"}
    for root_match in identifier_pattern.finditer(text):
        if not positions[root_match.start()]:
            continue
        root = _decode_quoted_js_text(root_match.group("identifier"))
        if root not in object_roots | reflect_roots:
            continue
        canonical_root = "Object" if root in object_roots else "Reflect"
        separator = next_code_offset(root_match.end())
        if separator is None:
            continue
        method: str | None = None
        member_end: int | None = None
        if text[separator] == ".":
            method_start = next_code_offset(separator + 1)
            if method_start is None:
                continue
            method_match = identifier_at_offset_pattern.match(text, method_start)
            if method_match is None:
                continue
            method = _decode_quoted_js_text(method_match.group())
            member_end = method_match.end()
        elif text[separator] == "[":
            method_start = next_code_offset(separator + 1)
            if method_start is None:
                continue
            method_match = quoted_key_pattern.match(text, method_start)
            if method_match is None:
                continue
            bracket = next_code_offset(method_match.end())
            if bracket is None or text[bracket] != "]":
                continue
            method = _decode_quoted_js_text(method_match.group("value"))
            member_end = bracket + 1
        if (canonical_root, method) not in {
            ("Object", "assign"),
            ("Object", "defineProperty"),
            ("Object", "defineProperties"),
            ("Object", "setPrototypeOf"),
            ("Reflect", "defineProperty"),
            ("Reflect", "set"),
            ("Reflect", "setPrototypeOf"),
        } or member_end is None:
            continue
        invocation_start = next_code_offset(member_end)
        called_indirectly = False
        if invocation_start is None:
            mutations.add((f"{root}.{method}", None))
            continue
        if text[invocation_start] in ".[":
            call_member_start = next_code_offset(invocation_start + 1)
            if call_member_start is None:
                mutations.add((f"{root}.{method}", None))
                continue
            call_member_end: int | None = None
            call_member: str | None = None
            if text[invocation_start] == ".":
                call_member_match = identifier_at_offset_pattern.match(text, call_member_start)
                if call_member_match is not None:
                    call_member = _decode_quoted_js_text(call_member_match.group())
                    call_member_end = call_member_match.end()
            else:
                call_member_match = quoted_key_pattern.match(text, call_member_start)
                if call_member_match is not None:
                    bracket = next_code_offset(call_member_match.end())
                    if bracket is not None and text[bracket] == "]":
                        call_member = _decode_quoted_js_text(call_member_match.group("value"))
                        call_member_end = bracket + 1
            if call_member != "call" or call_member_end is None:
                mutations.add((f"{root}.{method}", None))
                continue
            invocation_start = next_code_offset(call_member_end)
            called_indirectly = True
        call_start = invocation_start
        if call_start is None or text[call_start] != "(":
            mutations.add((f"{root}.{method}", None))
            continue

        parenthesis_depth = 1
        brace_depth = 0
        bracket_depth = 0
        argument_separators: list[int] = []
        call_end: int | None = None
        for offset in range(call_start + 1, len(text)):
            if not positions[offset]:
                continue
            character = text[offset]
            if character == "(":
                parenthesis_depth += 1
            elif character == ")":
                parenthesis_depth -= 1
                if parenthesis_depth == 0:
                    call_end = offset
                    break
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth = max(0, brace_depth - 1)
            elif character == "[":
                bracket_depth += 1
            elif character == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif character == "," and parenthesis_depth == 1 and brace_depth == 0 and bracket_depth == 0:
                argument_separators.append(offset)

        api_name = f"{canonical_root}.{method}"
        if (canonical_root, method) in {
            ("Object", "assign"),
            ("Object", "defineProperties"),
            ("Object", "setPrototypeOf"),
            ("Reflect", "setPrototypeOf"),
        }:
            mutations.add((api_name, None))
            continue
        if call_end is None or not argument_separators:
            mutations.add((api_name, None))
            continue
        argument_boundaries = [call_start, *argument_separators, call_end]
        property_index = 2 if called_indirectly else 1
        if len(argument_boundaries) <= property_index + 1:
            mutations.add((api_name, None))
            continue
        property_start = next_code_offset(argument_boundaries[property_index] + 1)
        property_end = argument_boundaries[property_index + 1]
        if property_start is None or property_start >= property_end:
            mutations.add((api_name, None))
            continue
        property_match = quoted_key_pattern.match(text, property_start, property_end)
        following_property = next_code_offset(property_match.end()) if property_match is not None else None
        if property_match is None or following_property != property_end:
            mutations.add((api_name, None))
            continue
        mutations.add((api_name, _decode_quoted_js_text(property_match.group("value"))))

    return mutations


def _inline_javascript_kind(
    script_type: str | None,
    *,
    script_language: str | None,
    has_src: bool,
    has_nomodule: bool,
) -> str | None:
    """Classify an inline script body using the browser's script-type boundary."""
    if has_src:
        return None
    if script_type is None:
        type_string = "text/javascript" if script_language in {None, ""} else f"text/{script_language}"
    elif script_type == "":
        type_string = "text/javascript"
    else:
        # HTML strips ASCII whitespace here, not arbitrary Unicode whitespace.
        type_string = script_type.strip("\t\n\f\r ")
    normalized_type = type_string.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))
    if normalized_type == "module":
        return "module"
    if normalized_type in _CLASSIC_JAVASCRIPT_MIME_TYPES and not has_nomodule:
        return "classic"
    return None


def _js_has_template_interpolation(text: str) -> bool:
    """Return whether executable source contains a real template interpolation."""
    positions = _js_code_positions(text)
    for offset, character in enumerate(text):
        if character != "`" or not positions[offset]:
            continue
        cursor = offset + 1
        while cursor < len(text):
            if text[cursor] == "\\":
                cursor += 2
                continue
            if text[cursor] == "`":
                break
            if text.startswith("${", cursor):
                return True
            cursor += 1
    return False


def _js_top_level_declared_identifiers(text: str, *, lexical_only: bool = False) -> set[str]:
    """Collect ordinary top-level bindings needed to respect module shadowing."""
    positions = _js_code_positions(text)
    top_level_positions = _js_unquoted_positions(text)
    identifier = r"[A-Za-z_$][A-Za-z0-9_$]*"
    declaration_pattern = re.compile(rf"\b(?P<kind>const|let|var|function|class)\s+(?P<name>{identifier})")
    bindings: set[str] = set()

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def matching_close(opening: int) -> int | None:
        delimiter = {"{": "}", "[": "]"}[text[opening]]
        depth = 0
        for offset in range(opening, len(text)):
            if not positions[offset]:
                continue
            if text[offset] == text[opening]:
                depth += 1
            elif text[offset] == delimiter:
                depth -= 1
                if depth == 0:
                    return offset
        return None

    for match in declaration_pattern.finditer(text):
        if (
            positions[match.start()]
            and top_level_positions[match.start()]
            and (not lexical_only or match.group("kind") in {"class", "const", "let"})
        ):
            bindings.add(match.group("name"))

    # The first regex covers the overwhelmingly common single declarator. Add
    # later names in a top-level comma-separated declaration without trying to
    # evaluate the initializer expressions between them.
    variable_declaration_pattern = re.compile(r"\b(?:const|let|var)\b")
    for match in variable_declaration_pattern.finditer(text):
        if not (positions[match.start()] and top_level_positions[match.start()]):
            continue
        if lexical_only and match.group() == "var":
            continue
        cursor = match.end()
        while cursor < len(text):
            delimiter = next(
                (
                    candidate
                    for candidate in range(cursor, len(text))
                    if top_level_positions[candidate] and text[candidate] in ",;"
                ),
                None,
            )
            if delimiter is None or text[delimiter] == ";":
                break
            name_start = next(
                (
                    candidate
                    for candidate in range(delimiter + 1, len(text))
                    if positions[candidate] and not text[candidate].isspace()
                ),
                None,
            )
            if name_start is None or not top_level_positions[name_start]:
                break
            name_match = re.match(identifier, text[name_start:])
            if name_match is None:
                break
            bindings.add(name_match.group())
            cursor = name_start + name_match.end()

        # Direct destructuring bindings are lexical names too.  Resolve the
        # ordinary shorthand/renamed forms needed for global shadowing without
        # interpreting initializer expressions.
        pattern_start = next_code_offset(match.end())
        if pattern_start is None or text[pattern_start] not in "{[":
            continue
        pattern_end = matching_close(pattern_start)
        if pattern_end is None:
            continue
        body = text[pattern_start + 1 : pattern_end]
        body_top_level = _js_unquoted_positions(body)
        boundaries = [0]
        boundaries.extend(
            offset + 1 for offset, character in enumerate(body) if character == "," and body_top_level[offset]
        )
        boundaries.append(len(body) + 1)
        for segment_start, next_start in zip(boundaries, boundaries[1:], strict=False):
            target = body[segment_start : next_start - 1].strip()
            if not target:
                continue
            top_level_equals = next(
                (
                    offset
                    for offset, character in enumerate(target)
                    if character == "=" and _js_unquoted_positions(target)[offset]
                ),
                None,
            )
            if top_level_equals is not None:
                target = target[:top_level_equals].strip()
            if text[pattern_start] == "{":
                target_positions = _js_unquoted_positions(target)
                colon = next(
                    (
                        offset
                        for offset, character in enumerate(target)
                        if character == ":" and target_positions[offset]
                    ),
                    None,
                )
                if colon is not None:
                    target = target[colon + 1 :].strip()
            target = target.removeprefix("...").strip()
            if re.fullmatch(identifier, target):
                bindings.add(target)
    return bindings


def _leverage_shares_global_product_references(
    text: str,
    *,
    global_aliases: set[str],
    local_bindings: set[str],
) -> set[int]:
    """Return references that can resolve to the browser-global inventory."""
    references = _js_semantic_identifier_references(text, "productsData")
    positions = _js_code_positions(text)
    identifier_escape = r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]{1,6}\})"
    identifier_pattern = re.compile(rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*")

    def previous_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, -1, -1) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def identifier_ending_at(end: int) -> str | None:
        matches = [match for match in identifier_pattern.finditer(text, 0, end) if match.end() == end]
        return _decode_quoted_js_text(matches[-1].group()) if matches else None

    global_references: set[int] = set()
    for reference in references:
        if text[reference] == "[":
            root_end = previous_code_offset(reference - 1)
            if root_end is not None and identifier_ending_at(root_end + 1) in global_aliases:
                global_references.add(reference)
            continue

        preceding = previous_code_offset(reference - 1)
        if preceding is not None and text[preceding] == ".":
            root_end = previous_code_offset(preceding - 1)
            if root_end is not None and text[root_end] == "?":
                root_end = previous_code_offset(root_end - 1)
            if root_end is not None and identifier_ending_at(root_end + 1) in global_aliases:
                global_references.add(reference)
            continue

        following = next_code_offset(reference + len("productsData"))
        if following is not None and text[following] == ":":
            # An object-literal property or label is not a reference to the
            # global inventory binding.
            continue
        if "productsData" not in local_bindings:
            global_references.add(reference)
    return global_references


def _leverage_shares_classic_script_aliases(
    text: str,
    *,
    global_aliases: set[str],
    object_aliases: set[str],
    reflect_aliases: set[str],
    mutator_aliases: dict[str, set[str]],
) -> tuple[set[str], set[str], set[str], dict[str, set[str]]]:
    """Resolve local aliases and carry top-level classic-script aliases forward."""
    positions = _js_code_positions(text)
    identifier_escape = r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]{1,6}\})"
    identifier = rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*"
    quoted_member = r"(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
    assignment_pattern = re.compile(
        rf"(?:^|[;,{{}}])\s*(?:(?:const|let|var)\s+)?"
        rf"(?:(?P<target>{identifier})\s*\.\s*)?(?P<alias>{identifier})\s*=\s*\(*\s*"
        rf"(?P<source>{identifier})"
        rf"(?P<members>(?:\s*(?:\.\s*{identifier}|\[\s*{quoted_member}\s*\]))*)"
        rf"\s*\)*(?=\s*(?:[,;}}]|$))",
        re.S,
    )
    member_source = rf"(?:\s*(?:\.\s*{identifier}|\[\s*{quoted_member}\s*\]))"
    sequence_assignment_pattern = re.compile(
        rf"(?:^|[;,{{}}])\s*(?:(?:const|let|var)\s+)?(?P<alias>{identifier})\s*=\s*"
        rf"\(\s*[^(),;]+,\s*(?P<source>{identifier})(?P<members>{member_source}+)\s*\)"
        rf"(?=\s*(?:[,;}}]|$))",
        re.S,
    )
    bound_assignment_pattern = re.compile(
        rf"(?:^|[;,{{}}])\s*(?:(?:const|let|var)\s+)?(?P<alias>{identifier})\s*=\s*(?P<group>\()?\s*"
        rf"(?P<source>{identifier})(?P<members>{member_source}+?)\s*\.\s*bind\s*\([^()]*\)"
        rf"\s*(?(group)\))(?=\s*(?:[,;}}]|$))",
        re.S,
    )
    member_pattern = re.compile(
        rf"\.\s*(?P<dot>{identifier})|\[\s*(?P<quoted>{quoted_member})\s*\]",
        re.S,
    )
    destructured_pattern = re.compile(
        rf"(?:^|[;,{{}}])\s*(?:(?:const|let|var)\s+)?"
        rf"(?P<object_open>\{{)\s*(?P<member>{identifier})\s*(?::\s*(?P<alias>{identifier}))?\s*\}}"
        rf"\s*=\s*(?P<source>{identifier})(?=\s*(?:[,;}}]|$))",
        re.S,
    )
    nesting_depth = 0
    top_level = [False] * len(text)
    for offset, character in enumerate(text):
        if not positions[offset]:
            continue
        top_level[offset] = nesting_depth == 0
        if character in "([{":
            nesting_depth += 1
        elif character in ")]}":
            nesting_depth = max(0, nesting_depth - 1)

    candidates: list[tuple[str, str, tuple[str, ...], bool, str | None]] = []

    def append_assignment_candidate(match: re.Match[str], *, bound: bool = False) -> None:
        alias_start = match.start("alias")
        if not positions[alias_start]:
            return
        members: list[str] = []
        for member_match in member_pattern.finditer(match.group("members")):
            raw_member = member_match.group("dot")
            if raw_member is None:
                quoted = member_match.group("quoted")
                raw_member = quoted[1:-1]
            members.append(_decode_quoted_js_text(raw_member))
        if bound:
            members.append("bind")
        candidates.append(
            (
                _decode_quoted_js_text(match.group("alias")),
                _decode_quoted_js_text(match.group("source")),
                tuple(members),
                top_level[alias_start],
                _decode_quoted_js_text(match.group("target"))
                if "target" in match.groupdict() and match.group("target")
                else None,
            )
        )

    for match in assignment_pattern.finditer(text):
        append_assignment_candidate(match)
    for match in sequence_assignment_pattern.finditer(text):
        append_assignment_candidate(match)
    for match in bound_assignment_pattern.finditer(text):
        append_assignment_candidate(match, bound=True)
    for match in destructured_pattern.finditer(text):
        member_start = match.start("member")
        if not positions[member_start]:
            continue
        raw_alias = match.group("alias") or match.group("member")
        candidates.append(
            (
                _decode_quoted_js_text(raw_alias),
                _decode_quoted_js_text(match.group("source")),
                (_decode_quoted_js_text(match.group("member")),),
                top_level[match.start("object_open")],
                None,
            )
        )

    script_global_aliases = set(global_aliases)
    script_object_aliases = set(object_aliases)
    script_reflect_aliases = set(reflect_aliases)
    script_mutator_aliases = {alias: set(kinds) for alias, kinds in mutator_aliases.items()}

    def destination_for(
        source: str,
        members: tuple[str, ...],
        *,
        globals_: set[str],
        objects: set[str],
        reflects: set[str],
        mutators: dict[str, set[str]],
    ) -> tuple[str, set[str] | None] | None:
        destination: tuple[str, set[str] | None] | None = None
        if source in globals_:
            destination = ("global", None)
        elif source in objects:
            destination = ("object", None)
        elif source in reflects:
            destination = ("reflect", None)
        elif source in mutators:
            destination = ("mutator", set(mutators[source]))
        for member in members:
            if destination is None:
                return None
            kind, mutation_kinds = destination
            if kind == "global" and member in {"globalThis", "self", "window"}:
                destination = ("global", None)
            elif kind == "global" and member == "Object":
                destination = ("object", None)
            elif kind == "global" and member == "Reflect":
                destination = ("reflect", None)
            elif kind == "reflect" and member == "apply":
                destination = ("mutator", {"reflectApply"})
            elif (
                kind == "object"
                and member in {"assign", "defineProperties", "defineProperty", "setPrototypeOf"}
                or kind == "reflect"
                and member in {"defineProperty", "deleteProperty", "set", "setPrototypeOf"}
            ):
                destination = ("mutator", {member})
            elif kind == "mutator" and member == "bind":
                destination = ("mutator", mutation_kinds)
            else:
                return None
        return destination

    def add_alias(
        alias: str,
        destination: tuple[str, set[str] | None] | None,
        *,
        globals_: set[str],
        objects: set[str],
        reflects: set[str],
        mutators: dict[str, set[str]],
    ) -> bool:
        if destination is None:
            return False
        destination_kind, mutation_kinds = destination
        if destination_kind == "mutator":
            assert mutation_kinds is not None
            prior_kinds = mutators.setdefault(alias, set())
            new_kinds = mutation_kinds - prior_kinds
            if not new_kinds:
                return False
            prior_kinds.update(new_kinds)
            return True
        destination_set = {"global": globals_, "object": objects, "reflect": reflects}[destination_kind]
        if alias in destination_set:
            return False
        destination_set.add(alias)
        return True

    changed = True
    while changed:
        changed = False
        for alias, source, members, _is_top_level, target in candidates:
            if target is not None and target not in script_global_aliases:
                continue
            destination = destination_for(
                source,
                members,
                globals_=script_global_aliases,
                objects=script_object_aliases,
                reflects=script_reflect_aliases,
                mutators=script_mutator_aliases,
            )
            if add_alias(
                alias,
                destination,
                globals_=script_global_aliases,
                objects=script_object_aliases,
                reflects=script_reflect_aliases,
                mutators=script_mutator_aliases,
            ):
                changed = True

    changed = True
    while changed:
        changed = False
        for alias, source, members, is_top_level, target in candidates:
            if not is_top_level:
                continue
            if target is not None and target not in global_aliases:
                continue
            destination = destination_for(
                source,
                members,
                globals_=global_aliases,
                objects=object_aliases,
                reflects=reflect_aliases,
                mutators=mutator_aliases,
            )
            if add_alias(
                alias,
                destination,
                globals_=global_aliases,
                objects=object_aliases,
                reflects=reflect_aliases,
                mutators=mutator_aliases,
            ):
                changed = True

    return (
        script_global_aliases,
        script_object_aliases,
        script_reflect_aliases,
        script_mutator_aliases,
    )


def _leverage_shares_has_dynamic_global_mutation(
    text: str,
    *,
    global_aliases: set[str],
    object_aliases: set[str],
    reflect_aliases: set[str],
    mutator_aliases: Mapping[str, set[str]],
) -> bool:
    """Detect mutation APIs that can replace a dynamic global property."""
    positions = _js_code_positions(text)
    identifier_escape = r"\\u(?:[0-9A-Fa-f]{4}|\{[0-9A-Fa-f]{1,6}\})"
    identifier_pattern = re.compile(
        rf"(?<![A-Za-z0-9_$\\])(?P<identifier>"
        rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*"
        rf")(?![A-Za-z0-9_$])"
    )
    identifier_at_offset_pattern = re.compile(
        rf"(?:[A-Za-z_$]|{identifier_escape})(?:[A-Za-z0-9_$]|{identifier_escape})*"
    )
    quoted_pattern = re.compile(
        r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)",
        re.S,
    )

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def matching_parenthesis(opening: int) -> int | None:
        depth = 0
        for offset in range(opening, len(text)):
            if not positions[offset]:
                continue
            if text[offset] == "(":
                depth += 1
            elif text[offset] == ")":
                depth -= 1
                if depth == 0:
                    return offset
        return None

    def skip_group_closes(offset: int | None) -> int | None:
        while offset is not None and text[offset] == ")":
            offset = next_code_offset(offset + 1)
        return offset

    def optional_chain_separator(offset: int | None) -> int | None:
        if offset is None or text[offset] != "?":
            return offset
        dot = next_code_offset(offset + 1)
        if dot is None or text[dot] != ".":
            return offset
        following = next_code_offset(dot + 1)
        return following if following is not None and text[following] in "[(" else dot

    def member_after(separator: int) -> tuple[str | None, int] | None:
        if separator >= len(text):
            return None
        member_start = next_code_offset(separator + 1)
        if member_start is None:
            return None
        if text[separator] == ".":
            match = identifier_at_offset_pattern.match(text, member_start)
            if match is None:
                return None
            return _decode_quoted_js_text(match.group()), match.end()
        if text[separator] != "[":
            return None
        match = quoted_pattern.match(text, member_start)
        if match is None:
            return None
        bracket = next_code_offset(match.end())
        if bracket is None or text[bracket] != "]":
            return None
        return _decode_quoted_js_text(match.group("value")), bracket + 1

    def call_arguments(call_start: int) -> list[tuple[int, int]] | None:
        parenthesis_depth = 1
        brace_depth = 0
        bracket_depth = 0
        argument_start = call_start + 1
        arguments: list[tuple[int, int]] = []
        for offset in range(call_start + 1, len(text)):
            if not positions[offset]:
                continue
            character = text[offset]
            if character == "(":
                parenthesis_depth += 1
            elif character == ")":
                parenthesis_depth -= 1
                if parenthesis_depth == 0:
                    arguments.append((argument_start, offset))
                    return arguments
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth = max(0, brace_depth - 1)
            elif character == "[":
                bracket_depth += 1
            elif character == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif character == "," and parenthesis_depth == 1 and brace_depth == 0 and bracket_depth == 0:
                arguments.append((argument_start, offset))
                argument_start = offset + 1
        return None

    def expression_is_global(start: int, end: int) -> bool:
        expression = text[start:end].strip()
        while expression.startswith("(") and expression.endswith(")"):
            expression = expression[1:-1].strip()
        root_match = identifier_at_offset_pattern.match(expression)
        if root_match is None:
            return False
        root = _decode_quoted_js_text(root_match.group())
        if root not in global_aliases:
            return False
        remainder = expression[root_match.end() :]
        if not remainder:
            return True
        member_match = re.fullmatch(
            rf"\s*(?:\.\s*(?P<dot>{identifier_escape}|[A-Za-z_$][A-Za-z0-9_$]*)|"
            rf"\[\s*(?P<quote>['\"])(?P<bracket>(?:\\.|(?!(?P=quote)).)*)(?P=quote)\s*\])\s*",
            remainder,
            re.S,
        )
        if member_match is None:
            return False
        raw_member = member_match.group("dot") or member_match.group("bracket")
        return _decode_quoted_js_text(raw_member) in {"globalThis", "self", "window"}

    def static_array_arguments(start: int, end: int) -> list[tuple[int, int]] | None:
        array_start = next_code_offset(start)
        array_end = next_code_offset(end - 1)
        if array_start is None or array_end is None or text[array_start] != "[" or text[array_end] != "]":
            return None
        parenthesis_depth = 0
        brace_depth = 0
        bracket_depth = 0
        argument_start = array_start + 1
        arguments: list[tuple[int, int]] = []
        for offset in range(argument_start, array_end):
            if not positions[offset]:
                continue
            character = text[offset]
            if character == "(":
                parenthesis_depth += 1
            elif character == ")":
                parenthesis_depth = max(0, parenthesis_depth - 1)
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth = max(0, brace_depth - 1)
            elif character == "[":
                bracket_depth += 1
            elif character == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif character == "," and parenthesis_depth == 0 and brace_depth == 0 and bracket_depth == 0:
                arguments.append((argument_start, offset))
                argument_start = offset + 1
        if text[argument_start:array_end].strip():
            arguments.append((argument_start, array_end))
        return arguments

    def static_property(start: int, end: int) -> str | None:
        expression = text[start:end].strip()
        match = quoted_pattern.fullmatch(expression)
        return _decode_quoted_js_text(match.group("value")) if match is not None else None

    def static_object_property_names(start: int, end: int) -> set[str] | None:
        """Return literal/shorthand object keys, or None for a dynamic source."""
        expression = text[start:end].strip()
        if len(expression) < 2 or expression[0] != "{" or expression[-1] != "}":
            return None
        body = expression[1:-1]
        top_level_positions = _js_unquoted_positions(body)
        boundaries = [0]
        boundaries.extend(
            offset + 1 for offset, character in enumerate(body) if character == "," and top_level_positions[offset]
        )
        boundaries.append(len(body) + 1)
        key_pattern = re.compile(
            r"\s*(?:(?P<quote>['\"])(?P<quoted>(?:\\.|(?!(?P=quote)).)*)(?P=quote)"
            r"|(?P<bare>[A-Za-z_$][A-Za-z0-9_$]*))\s*:"
        )
        shorthand_pattern = re.compile(r"\s*(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*")
        method_pattern = re.compile(r"\s*(?:(?:get|set|async)\s+)?(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
        names: set[str] = set()
        for start_offset, next_start in zip(boundaries, boundaries[1:], strict=False):
            segment = body[start_offset : next_start - 1]
            if not segment.strip():
                continue
            key_match = key_pattern.match(segment)
            if key_match is not None:
                raw_name = key_match.group("quoted")
                names.add(_decode_quoted_js_text(raw_name) if raw_name is not None else str(key_match.group("bare")))
                continue
            shorthand_match = shorthand_pattern.fullmatch(segment)
            if shorthand_match is not None:
                names.add(shorthand_match.group("name"))
                continue
            method_match = method_pattern.match(segment)
            if method_match is not None:
                names.add(method_match.group("name"))
                continue
            # Spreads, computed names, and other expressions can copy or define
            # productsData without spelling a static key at this call site.
            return None
        return names

    def mutation_methods_for_expression(start: int, end: int) -> set[str]:
        """Resolve a bounded static Object/Reflect/mutator expression."""
        expression = text[start:end].strip()
        while expression.startswith("(") and expression.endswith(")"):
            expression = expression[1:-1].strip()
        chain_pattern = re.compile(
            rf"(?P<root>{identifier_escape}|[A-Za-z_$][A-Za-z0-9_$]*)"
            rf"(?P<members>(?:\s*(?:\.\s*(?:{identifier_escape}|[A-Za-z_$][A-Za-z0-9_$]*)|"
            rf"\[\s*(?:'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\s*\]))*)",
            re.S,
        )
        chain = chain_pattern.fullmatch(expression)
        if chain is None:
            return set()
        root = _decode_quoted_js_text(chain.group("root"))
        if root in global_aliases:
            category = "global"
            methods: set[str] = set()
        elif root in object_aliases:
            category = "object"
            methods = set()
        elif root in reflect_aliases:
            category = "reflect"
            methods = set()
        elif root in mutator_aliases:
            category = "mutator"
            methods = set(mutator_aliases[root])
        else:
            return set()
        member_token = re.compile(
            rf"\.\s*(?P<dot>{identifier_escape}|[A-Za-z_$][A-Za-z0-9_$]*)|"
            r"\[\s*(?P<quoted>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\s*\]",
            re.S,
        )
        for match in member_token.finditer(chain.group("members")):
            raw_member = match.group("dot")
            if raw_member is None:
                quoted = match.group("quoted")
                raw_member = quoted[1:-1]
            member = _decode_quoted_js_text(raw_member)
            if category == "global" and member in {"globalThis", "self", "window"}:
                continue
            if category == "global" and member in {"Object", "Reflect"}:
                category = member.casefold()
                continue
            valid_methods = (
                {"assign", "defineProperties", "defineProperty", "setPrototypeOf"}
                if category == "object"
                else {"defineProperty", "deleteProperty", "set", "setPrototypeOf"}
                if category == "reflect"
                else set()
            )
            if member in valid_methods:
                category = "mutator"
                methods = {member}
                continue
            if category == "mutator" and member == "bind":
                continue
            return set()
        return methods if category == "mutator" else set()

    for root_match in identifier_pattern.finditer(text):
        if not positions[root_match.start()]:
            continue
        root = _decode_quoted_js_text(root_match.group("identifier"))
        invocation_start = optional_chain_separator(next_code_offset(root_match.end()))
        target_argument_index = 0
        possible_methods: set[str] = set()
        reflect_apply = False
        if root in mutator_aliases:
            if invocation_start is None:
                continue
            possible_methods.update(mutator_aliases[root])
        elif root in object_aliases | reflect_aliases:
            if invocation_start is None or text[invocation_start] not in ".[":
                continue
            member = member_after(invocation_start)
            if member is None:
                continue
            valid_methods = (
                {"assign", "defineProperties", "defineProperty", "setPrototypeOf"}
                if root in object_aliases
                else {"apply", "defineProperty", "deleteProperty", "set", "setPrototypeOf"}
            )
            method = member[0]
            if method not in valid_methods:
                continue
            if method == "apply":
                reflect_apply = True
            else:
                possible_methods.add(method)
            invocation_start = next_code_offset(member[1])
        else:
            continue

        invocation_start = optional_chain_separator(skip_group_closes(invocation_start))
        indirect_kind: str | None = None
        if invocation_start is not None and text[invocation_start] in ".[":
            call_member = member_after(invocation_start)
            if call_member is None or call_member[0] not in {"apply", "bind", "call"}:
                continue
            indirect_kind = call_member[0]
            target_argument_index = 1 if indirect_kind in {"apply", "call"} else 0
            invocation_start = next_code_offset(call_member[1])
        invocation_start = optional_chain_separator(invocation_start)
        if invocation_start is None or text[invocation_start] != "(":
            continue
        arguments = call_arguments(invocation_start)
        if arguments is not None and indirect_kind == "bind":
            bind_close = matching_parenthesis(invocation_start)
            invocation_start = optional_chain_separator(
                skip_group_closes(next_code_offset(bind_close + 1)) if bind_close is not None else None
            )
            if invocation_start is None or text[invocation_start] != "(":
                continue
            arguments = call_arguments(invocation_start)
            target_argument_index = 0
        if arguments is not None and indirect_kind == "apply":
            if len(arguments) <= 1:
                continue
            apply_arguments = static_array_arguments(*arguments[1])
            if apply_arguments is None:
                return True
            arguments = apply_arguments
            target_argument_index = 0
        if arguments is not None and (reflect_apply or "reflectApply" in possible_methods):
            if len(arguments) <= 2:
                continue
            possible_methods = mutation_methods_for_expression(*arguments[0])
            if not possible_methods:
                continue
            apply_arguments = static_array_arguments(*arguments[2])
            if apply_arguments is None:
                return True
            arguments = apply_arguments
            target_argument_index = 0
        property_argument_index = target_argument_index + 1
        if arguments is None or len(arguments) <= property_argument_index:
            continue
        if not expression_is_global(*arguments[target_argument_index]):
            continue
        if "setPrototypeOf" in possible_methods:
            return True
        for bulk_method in possible_methods & {"assign", "defineProperties"}:
            source_arguments = arguments[target_argument_index + 1 :]
            if bulk_method == "defineProperties":
                source_arguments = source_arguments[:1]
            for source_start, source_end in source_arguments:
                property_names = static_object_property_names(source_start, source_end)
                if property_names is None or "productsData" in property_names:
                    return True
        if possible_methods & {"defineProperty", "deleteProperty", "set"}:
            property_name = static_property(*arguments[property_argument_index])
            if property_name is None or property_name == "productsData":
                return True

    return False


def _leverage_shares_parameter_scope_analysis(
    text: str,
    *,
    global_aliases: set[str],
    object_aliases: set[str],
    reflect_aliases: set[str],
    mutator_aliases: Mapping[str, set[str]],
) -> tuple[str, set[str], set[str], set[str], dict[str, set[str]]]:
    """Respect bounded parameter scopes while following static capability flow."""
    positions = _js_code_positions(text)
    identifier = r"[A-Za-z_$][A-Za-z0-9_$]*"
    parameter_definition = re.compile(rf"(?P<name>{identifier})(?:\s*=\s*(?P<default>.*))?", re.S)
    definitions: list[tuple[str | None, list[tuple[str, str | None, bool]], int, int, int, list[list[str]]]] = []

    def matching_close(opening: int, opening_character: str, closing_character: str) -> int | None:
        depth = 0
        for offset in range(opening, len(text)):
            if not positions[offset]:
                continue
            if text[offset] == opening_character:
                depth += 1
            elif text[offset] == closing_character:
                depth -= 1
                if depth == 0:
                    return offset
        return None

    def next_code_offset(start: int) -> int | None:
        return next(
            (offset for offset in range(start, len(text)) if positions[offset] and not text[offset].isspace()),
            None,
        )

    def split_source_arguments(source: str) -> list[str]:
        source_positions = _js_code_positions(source)
        arguments: list[str] = []
        argument_start = 0
        nesting_depth = 0
        for offset, character in enumerate(source):
            if not source_positions[offset]:
                continue
            if character in "([{":
                nesting_depth += 1
            elif character in ")]}":
                nesting_depth = max(0, nesting_depth - 1)
            elif character == "," and nesting_depth == 0:
                arguments.append(source[argument_start:offset])
                argument_start = offset + 1
        if source[argument_start:].strip() or arguments:
            arguments.append(source[argument_start:])
        return arguments

    def call_arguments(call_open: int, call_close: int) -> list[str]:
        return split_source_arguments(text[call_open + 1 : call_close])

    def static_array_arguments(expression: str) -> list[str] | None:
        expression = expression.strip()
        while expression.startswith("(") and expression.endswith(")"):
            expression = expression[1:-1].strip()
        if len(expression) < 2 or expression[0] != "[" or expression[-1] != "]":
            return None
        return split_source_arguments(expression[1:-1])

    def parameters_from(source: str | None) -> list[tuple[str, str | None, bool]]:
        parameters: list[tuple[str, str | None, bool]] = []
        if source is None:
            return parameters
        for raw_parameter in split_source_arguments(source):
            match = parameter_definition.fullmatch(raw_parameter.strip())
            if match is None:
                # Destructuring/rest can carry a global through a container in
                # ways this bounded analysis cannot reproduce. Treat every
                # possible bound identifier as global whenever the function is
                # called, so member access in its body fails shut.
                parameters.extend(
                    (name, None, True)
                    for name in re.findall(identifier, raw_parameter)
                    if name not in {parameter for parameter, _default, _conservative in parameters}
                )
                continue
            parameters.append((match.group("name"), match.group("default"), False))
        return parameters

    function_pattern = re.compile(rf"\bfunction\s+(?P<name>{identifier})\s*\((?P<parameters>[^()]*)\)\s*\{{")
    for match in function_pattern.finditer(text):
        if not positions[match.start()]:
            continue
        body_start = match.end() - 1
        body_end = matching_close(body_start, "{", "}")
        if body_end is not None:
            parameters = parameters_from(match.group("parameters"))
            definitions.append((match.group("name"), parameters, body_start + 1, body_end, match.start(), []))

    function_expression_pattern = re.compile(
        rf"\b(?:const|let|var)\s+(?P<name>{identifier})\s*=\s*function\s*"
        rf"\((?P<parameters>[^()]*)\)\s*\{{"
    )
    for match in function_expression_pattern.finditer(text):
        if not positions[match.start()]:
            continue
        body_start = match.end() - 1
        body_end = matching_close(body_start, "{", "}")
        if body_end is not None:
            definitions.append(
                (
                    match.group("name"),
                    parameters_from(match.group("parameters")),
                    body_start + 1,
                    body_end,
                    match.start(),
                    [],
                )
            )

    arrow_pattern = re.compile(
        rf"\b(?:const|let|var)\s+(?P<name>{identifier})\s*=\s*"
        rf"(?:\((?P<parenthesized>[^()]*)\)|(?P<bare>{identifier}))\s*=>\s*"
    )
    for match in arrow_pattern.finditer(text):
        if not positions[match.start()]:
            continue
        body_start = next_code_offset(match.end())
        if body_start is None:
            continue
        if text[body_start] == "{":
            body_end = matching_close(body_start, "{", "}")
            if body_end is None:
                continue
            body_start += 1
        else:
            body_end = next(
                (offset for offset in range(body_start, len(text)) if positions[offset] and text[offset] == ";"),
                len(text),
            )
        raw_parameters = (
            match.group("parenthesized") if match.group("parenthesized") is not None else match.group("bare")
        )
        definitions.append(
            (match.group("name"), parameters_from(raw_parameters), body_start, body_end, match.start(), [])
        )

    assigned_arrow_pattern = re.compile(
        rf"(?:^|[;}}])\s*(?P<name>{identifier})\s*=\s*"
        rf"(?:\((?P<parenthesized>[^()]*)\)|(?P<bare>{identifier}))\s*=>\s*"
    )
    for match in assigned_arrow_pattern.finditer(text):
        if not positions[match.start("name")]:
            continue
        body_start = next_code_offset(match.end())
        if body_start is None:
            continue
        if text[body_start] == "{":
            body_end = matching_close(body_start, "{", "}")
            if body_end is None:
                continue
            body_start += 1
        else:
            body_end = next(
                (offset for offset in range(body_start, len(text)) if positions[offset] and text[offset] == ";"),
                len(text),
            )
        raw_parameters = (
            match.group("parenthesized") if match.group("parenthesized") is not None else match.group("bare")
        )
        definitions.append(
            (match.group("name"), parameters_from(raw_parameters), body_start, body_end, match.start("name"), [])
        )

    object_method_pattern = re.compile(
        rf"\b(?:const|let|var)\s+(?P<object>{identifier})\s*=\s*\{{\s*"
        rf"(?P<method>{identifier})\s*\((?P<parameters>[^()]*)\)\s*\{{"
    )
    for match in object_method_pattern.finditer(text):
        if not positions[match.start()]:
            continue
        body_start = match.end() - 1
        body_end = matching_close(body_start, "{", "}")
        if body_end is None:
            continue
        definitions.append(
            (
                f"{match.group('object')}.{match.group('method')}",
                parameters_from(match.group("parameters")),
                body_start + 1,
                body_end,
                match.start(),
                [],
            )
        )

    # Anonymous immediately-invoked functions and arrows never acquire a name
    # that the ordinary call scan can follow.  Recognize only their direct,
    # statically delimited invocation form.
    function_iife_pattern = re.compile(r"\(\s*function\s*\((?P<parameters>[^()]*)\)\s*\{")
    for match in function_iife_pattern.finditer(text):
        if not positions[match.start()]:
            continue
        body_open = match.end() - 1
        body_close = matching_close(body_open, "{", "}")
        wrapper_close = next_code_offset(body_close + 1) if body_close is not None else None
        call_open = (
            next_code_offset(wrapper_close + 1) if wrapper_close is not None and text[wrapper_close] == ")" else None
        )
        call_close = matching_close(call_open, "(", ")") if call_open is not None and text[call_open] == "(" else None
        if body_close is None or call_open is None or call_close is None:
            continue
        definitions.append(
            (
                None,
                parameters_from(match.group("parameters")),
                body_open + 1,
                body_close,
                match.start(),
                [call_arguments(call_open, call_close)],
            )
        )

    arrow_iife_pattern = re.compile(rf"\(\s*(?:\((?P<parenthesized>[^()]*)\)|(?P<bare>{identifier}))\s*=>\s*")
    for match in arrow_iife_pattern.finditer(text):
        if not positions[match.start()]:
            continue
        wrapper_close = matching_close(match.start(), "(", ")")
        call_open = next_code_offset(wrapper_close + 1) if wrapper_close is not None else None
        call_close = matching_close(call_open, "(", ")") if call_open is not None and text[call_open] == "(" else None
        if wrapper_close is None or call_open is None or call_close is None:
            continue
        body_start = next_code_offset(match.end())
        if body_start is None:
            continue
        if text[body_start] == "{":
            body_close = matching_close(body_start, "{", "}")
            if body_close is None or body_close > wrapper_close:
                continue
            body_start += 1
            body_end = body_close
        else:
            body_end = wrapper_close
        raw_parameters = (
            match.group("parenthesized") if match.group("parenthesized") is not None else match.group("bare")
        )
        definitions.append(
            (
                None,
                parameters_from(raw_parameters),
                body_start,
                body_end,
                match.start(),
                [call_arguments(call_open, call_close)],
            )
        )

    def expression_category(expression: str) -> tuple[str, set[str] | None] | None:
        expression = expression.strip()
        while expression.startswith("(") and expression.endswith(")"):
            expression = expression[1:-1].strip()
        root_match = re.match(identifier, expression)
        if root_match is None:
            return None
        root = root_match.group()
        remainder = expression[root_match.end() :]
        if not remainder:
            if root in global_aliases:
                return "global", None
            if root in object_aliases:
                return "object", None
            if root in reflect_aliases:
                return "reflect", None
            if root in mutator_aliases:
                return "mutator", set(mutator_aliases[root])
            return None
        member_match = re.fullmatch(
            r"\s*(?:\.\s*(?P<dot>[A-Za-z_$][A-Za-z0-9_$]*)|"
            r"\[\s*(['\"])(?P<bracket>(?:\\.|(?!\2).)*)\2\s*\])\s*",
            remainder,
            re.S,
        )
        if member_match is None:
            return None
        member = member_match.group("dot") or _decode_quoted_js_text(member_match.group("bracket"))
        if root in global_aliases:
            if member in {"globalThis", "self", "window"}:
                return "global", None
            if member == "Object":
                return "object", None
            if member == "Reflect":
                return "reflect", None
        if root in object_aliases and member in {
            "assign",
            "defineProperties",
            "defineProperty",
            "setPrototypeOf",
        }:
            return "mutator", {member}
        if root in reflect_aliases and member in {
            "apply",
            "defineProperty",
            "deleteProperty",
            "set",
            "setPrototypeOf",
        }:
            return "mutator", {"reflectApply" if member == "apply" else member}
        return None

    def static_string_value(expression: str | None) -> str | None:
        if expression is None:
            return None
        expression = expression.strip()
        match = re.fullmatch(r"(?P<quote>['\"])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)", expression, re.S)
        return _decode_quoted_js_text(match.group("value")) if match is not None else None

    masked = list(text)
    flowed_global_aliases: set[str] = set()
    flowed_object_aliases: set[str] = set()
    flowed_reflect_aliases: set[str] = set()
    flowed_mutator_aliases: dict[str, set[str]] = {}

    def record_parameter_flow(
        parameter: str,
        category: tuple[str, set[str] | None],
        flowed_parameters: set[str],
    ) -> None:
        flowed_parameters.add(parameter)
        category_name, mutation_kinds = category
        if category_name == "global":
            flowed_global_aliases.add(parameter)
        elif category_name == "object":
            flowed_object_aliases.add(parameter)
        elif category_name == "reflect":
            flowed_reflect_aliases.add(parameter)
        else:
            assert mutation_kinds is not None
            flowed_mutator_aliases.setdefault(parameter, set()).update(mutation_kinds)

    for function_name, parameters, body_start, body_end, definition_start, explicit_calls in definitions:
        flowed_parameters: set[str] = set()
        local_global_parameters: set[str] = set()
        constant_parameter_values: dict[str, set[str]] = {}
        nonconstant_parameters: set[str] = set()
        calls = list(explicit_calls)
        if function_name is not None:
            call_pattern = re.compile(
                rf"\b{re.escape(function_name)}\s*(?:(?:\.\s*(?P<dot_indirect>apply|bind|call)|"
                rf"\[\s*['\"](?P<bracket_indirect>apply|bind|call)['\"]\s*\])\s*)?\("
            )
            for call_match in call_pattern.finditer(text):
                if (
                    not positions[call_match.start()]
                    or definition_start <= call_match.start() < body_start
                    or body_start <= call_match.start() < body_end
                ):
                    continue
                call_open = call_match.end() - 1
                call_close = matching_close(call_open, "(", ")")
                if call_close is None:
                    continue
                arguments = call_arguments(call_open, call_close)
                indirect = call_match.group("dot_indirect") or call_match.group("bracket_indirect")
                if indirect == "call":
                    arguments = arguments[1:]
                elif indirect == "apply":
                    applied = static_array_arguments(arguments[1]) if len(arguments) > 1 else None
                    # Unknown apply arguments can hide any global capability;
                    # keep every relevant parameter live so later scans fail shut.
                    arguments = ["window"] * len(parameters) if applied is None else applied
                elif indirect == "bind":
                    invocation_open = next_code_offset(call_close + 1)
                    invocation_close = (
                        matching_close(invocation_open, "(", ")")
                        if invocation_open is not None and text[invocation_open] == "("
                        else None
                    )
                    if invocation_open is None or invocation_close is None:
                        continue
                    arguments = arguments[1:] + call_arguments(invocation_open, invocation_close)
                calls.append(arguments)

            for reflect_name in reflect_aliases:
                reflect_apply_pattern = re.compile(rf"\b{re.escape(reflect_name)}\s*\.\s*apply\s*\(")
                for apply_match in reflect_apply_pattern.finditer(text):
                    if not positions[apply_match.start()]:
                        continue
                    call_open = apply_match.end() - 1
                    call_close = matching_close(call_open, "(", ")")
                    if call_close is None:
                        continue
                    apply_call_arguments = call_arguments(call_open, call_close)
                    if not apply_call_arguments:
                        continue
                    applied_function = apply_call_arguments[0].strip()
                    while applied_function.startswith("(") and applied_function.endswith(")"):
                        applied_function = applied_function[1:-1].strip()
                    if applied_function != function_name:
                        continue
                    applied = static_array_arguments(apply_call_arguments[2]) if len(apply_call_arguments) > 2 else None
                    calls.append(["window"] * len(parameters) if applied is None else applied)

        for arguments in calls:
            for parameter_index, (parameter, default_expression, conservative) in enumerate(parameters):
                argument = arguments[parameter_index] if parameter_index < len(arguments) else default_expression
                constant_value = static_string_value(argument)
                if constant_value is None:
                    nonconstant_parameters.add(parameter)
                else:
                    constant_parameter_values.setdefault(parameter, set()).add(constant_value)
                if conservative:
                    record_parameter_flow(parameter, ("global", None), flowed_parameters)
                    local_global_parameters.add(parameter)
                    continue
                category = expression_category(argument) if argument is not None else None
                if category is None:
                    continue
                record_parameter_flow(parameter, category, flowed_parameters)
                if category[0] == "global":
                    local_global_parameters.add(parameter)

        # A called function can overwrite an initially harmless parameter with
        # a known global capability before using it.
        if calls:
            body = text[body_start:body_end]
            for parameter, _default_expression, _conservative in parameters:
                reassignment_pattern = re.compile(rf"\b{re.escape(parameter)}\s*=\s*(?P<value>[^;,\r\n}}]+)")
                for reassignment in reassignment_pattern.finditer(body):
                    category = expression_category(reassignment.group("value"))
                    if category is not None:
                        record_parameter_flow(parameter, category, flowed_parameters)
                        if category[0] == "global":
                            local_global_parameters.add(parameter)

        # Resolve computed global keys carried by another static parameter.
        # The live issuer uses this ordinary pattern for analytics APIs such as
        # ``c[l]`` with ``l='clarity'``; only an unknown/productsData key can
        # affect the inventory.
        body = text[body_start:body_end]
        safe_key_parameters = {
            parameter
            for parameter, values in constant_parameter_values.items()
            if (
                parameter not in nonconstant_parameters
                and values
                and "productsData" not in values
                and re.search(rf"\b{re.escape(parameter)}\s*(?:=|\+\+|--)", body) is None
            )
        }
        for global_parameter in local_global_parameters:
            for key_parameter in safe_key_parameters:
                safe_access_pattern = re.compile(
                    rf"\b{re.escape(global_parameter)}\s*\[\s*{re.escape(key_parameter)}\s*\]"
                )
                for access in safe_access_pattern.finditer(body):
                    if not positions[body_start + access.start()]:
                        continue
                    bracket_start = body.find("[", access.start(), access.end())
                    if bracket_start >= 0:
                        absolute_start = body_start + bracket_start
                        absolute_end = body_start + access.end()
                        masked[absolute_start:absolute_end] = " " * (absolute_end - absolute_start)

        parameter_names = {parameter for parameter, _default_expression, _conservative in parameters}
        masked_parameters = parameter_names - flowed_parameters
        references = _js_semantic_identifier_reference_map(text[body_start:body_end], masked_parameters)
        for parameter in masked_parameters:
            for reference in references[parameter]:
                absolute_reference = body_start + reference
                masked[absolute_reference : absolute_reference + len(parameter)] = " " * len(parameter)
    return (
        "".join(masked),
        flowed_global_aliases,
        flowed_object_aliases,
        flowed_reflect_aliases,
        flowed_mutator_aliases,
    )


class _DuplicateJsonObjectKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


class _NonFiniteJsonNumberError(ValueError):
    """Raised when an authoritative JSON body decodes to a non-finite number."""


class _ExcessiveJsonNestingError(ValueError):
    """Raised before an authoritative JSON body exceeds the supported depth."""


class _ExcessiveJsonStructureError(ValueError):
    """Raised before a compact JSON body can expand into too many objects."""


def _json_object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    item: dict[str, object] = {}
    for key, value in pairs:
        if key in item:
            raise _DuplicateJsonObjectKeyError(key)
        item[key] = value
    return item


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
    """Reject non-finite floats even when a valid numeric token overflows float."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise _NonFiniteJsonNumberError(repr(item))
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _strict_json_loads(
    json_text: str,
    *,
    source_description: str,
    invalid_json_message: str,
) -> object:
    """Decode JSON while rejecting ambiguous keys and non-finite numbers."""
    try:
        _reject_excessive_json_text_nesting(json_text)
        decoded = json.loads(
            json_text,
            object_pairs_hook=_json_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
        _reject_nonfinite_json_numbers(decoded)
        return decoded
    except _DuplicateJsonObjectKeyError as exc:
        raise ValueError(f"{source_description} contained a duplicate JSON object key: {exc.key!r}.") from exc
    except _NonFiniteJsonNumberError as exc:
        raise ValueError(f"{source_description} contained a non-finite JSON number: {exc}.") from exc
    except _ExcessiveJsonNestingError as exc:
        raise ValueError(
            f"{source_description} exceeded the supported JSON nesting depth of {_MAX_JSON_NESTING_DEPTH}."
        ) from exc
    except _ExcessiveJsonStructureError as exc:
        raise ValueError(
            f"{source_description} exceeded the supported JSON structural limit of "
            f"{_MAX_JSON_STRUCTURAL_TOKENS} tokens."
        ) from exc
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(invalid_json_message) from exc


def _fund_rows_to_universe(
    rows: Iterable[dict[str, object]],
    source_name: str,
    *,
    source_label: str,
    require_leveraged: bool,
    product_type: str = "ETF",
    allow_missing_symbols: bool = False,
) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])

    out = out[["symbol", "name"]].copy()
    normalized_symbols: list[str | None] = []
    source_spelling_by_identity: dict[str, str] = {}
    for row_number, raw_symbol in enumerate(out["symbol"].tolist()):
        if allow_missing_symbols and (raw_symbol is None or pd.isna(raw_symbol)):
            normalized_symbols.append(None)
            continue
        identity = _exchange_product_symbol_identity(raw_symbol)
        if identity is None:
            raise ValueError(
                f"{source_label} row {row_number} contained an invalid or noncanonical product ticker: {raw_symbol!r}."
            )
        assert isinstance(raw_symbol, str)
        prior_spelling = source_spelling_by_identity.get(identity)
        if prior_spelling is not None and prior_spelling != raw_symbol:
            raise ValueError(
                f"{source_label} contained source ticker spellings with a normalization collision for "
                f"{identity}: {prior_spelling!r} and {raw_symbol!r}."
            )
        source_spelling_by_identity[identity] = raw_symbol
        # Apply product exclusions only after proving that the authoritative
        # source identity itself is canonical. Malformed punctuation must not
        # be stripped into a different, valid exchange ticker.
        normalized_symbols.append(normalize_yahoo_symbol(identity))
    out["symbol"] = normalized_symbols
    out["name"] = out["name"].map(_html_text)
    out = out[out["symbol"].notna()]
    out = out[out["name"].ne("")]
    if require_leveraged:
        out = out[out["name"].apply(leveraged_name_filter)]
    out["fund_type"] = f"{product_type} ({source_name})"
    out["source"] = source_label
    dedupe_columns = "symbol" if require_leveraged else ["symbol", "name"]
    out = out.drop_duplicates(dedupe_columns)
    return out.reset_index(drop=True)


def _validated_structured_fund_rows(
    rows: Iterable[tuple[str, object, object]],
    *,
    source_description: str,
    reject_conflicting_duplicates: bool = True,
) -> list[dict[str, object]]:
    """Validate authoritative ticker/name records before eligibility filtering.

    Structured issuer feeds must not silently discard a malformed row or let a
    conflicting duplicate choose a symbol's metadata by input order. Exact
    duplicates after ticker and visible-name canonicalization are harmless and
    are collapsed here. Generic source parsing can retain distinct conflicts so
    the source-wide leverage/RSI comparison can exclude only those symbols.
    """
    validated_rows: list[dict[str, object]] = []
    canonical_by_symbol: dict[str, dict[str, object]] = {}
    for row_label, raw_symbol, raw_name in rows:
        symbol = _exchange_product_symbol_identity(raw_symbol)
        if symbol is None:
            raise ValueError(
                f"{source_description} {row_label} omitted or contained an invalid canonical ticker value."
            )
        if type(raw_name) is not str:
            raise ValueError(f"{source_description} {row_label} omitted its required product name.")
        name = _html_text(raw_name)
        if not name:
            raise ValueError(f"{source_description} {row_label} omitted its required product name.")

        canonical_row: dict[str, object] = {"symbol": symbol, "name": name}
        prior = canonical_by_symbol.get(symbol)
        if prior is not None:
            if prior != canonical_row and reject_conflicting_duplicates:
                raise ValueError(f"{source_description} contained conflicting records for normalized ticker {symbol}.")
            if prior == canonical_row:
                continue
        else:
            canonical_by_symbol[symbol] = canonical_row
        validated_rows.append(canonical_row)
    return validated_rows


def _workflow_issuer_source(raw_source: object) -> UniverseSource:
    if isinstance(raw_source, UniverseSource):
        return raw_source
    try:
        source, url = raw_source  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported workflow issuer source: {raw_source!r}") from exc
    return UniverseSource(str(source), str(url), "issuer_etf")


def _fund_table_column_label(column: object) -> str:
    """Return a semantic label, recovering pandas-mangled duplicate headers."""
    text = str(column).strip()
    mangled = re.fullmatch(r"(?P<label>.+)\.(?:[1-9]\d*)", text)
    if mangled is not None:
        base_label = _normalized_column_label(mangled.group("label"))
        words = set(base_label.split())
        if {"ticker", "symbol", "name"} & words or base_label in {"fund", "etf", "etn"}:
            return base_label
    return _normalized_column_label(text)


_FUND_TABLE_SYMBOL_LABELS = frozenset(
    {
        "symbol",
        "ticker",
        "ticker symbol",
        "trading symbol",
        "fund symbol",
        "fund ticker",
        "etf symbol",
        "etf ticker",
        "etn symbol",
        "etn ticker",
    }
)
_FUND_TABLE_NAME_LABELS = frozenset(
    {
        "name",
        "fund",
        "etf",
        "etn",
        "fund name",
        "product",
        "product name",
        "security name",
        "etf name",
        "etn name",
        "name of fund",
        "name of product",
        "name of security",
        "name of etf",
        "name of etn",
    }
)


def _fund_table_schema(table: pd.DataFrame, source_name: str) -> tuple[object, object] | None:
    """Select one unambiguous ticker/name schema while ignoring page furniture."""
    symbol_columns: list[object] = []
    name_columns: list[object] = []
    for column in table.columns:
        label = _fund_table_column_label(column)
        if label in _FUND_TABLE_SYMBOL_LABELS:
            symbol_columns.append(column)
        if label in _FUND_TABLE_NAME_LABELS:
            name_columns.append(column)

    # A quote, navigation, or layout table may expose only one generic role.
    # Requiring both keeps those incidental tables outside the authoritative
    # candidate set without guessing at columns by position.
    if not symbol_columns or not name_columns:
        return None

    if len(symbol_columns) != 1 or len(name_columns) != 1 or symbol_columns[0] == name_columns[0]:
        details = f"symbol={list(map(str, symbol_columns))}; name={list(map(str, name_columns))}"
        raise ValueError(f"{source_name} issuer table contained ambiguous semantic symbol/name columns ({details}).")
    return symbol_columns[0], name_columns[0]


def _fund_table_to_universe(
    table: pd.DataFrame,
    source_name: str,
    *,
    source_label: str,
    require_leveraged: bool,
) -> pd.DataFrame:
    schema = _fund_table_schema(table, source_name)
    if schema is None:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
    symbol_col, name_col = schema

    source_rows = (
        (f"row {row_number}", row[symbol_col], row[name_col])
        for row_number, (_idx, row) in enumerate(table[[symbol_col, name_col]].iterrows())
    )
    rows = _validated_structured_fund_rows(
        source_rows,
        source_description=f"{source_name} issuer table",
        reject_conflicting_duplicates=require_leveraged,
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
    payload = _strict_json_loads(
        json_text,
        source_description="Defiance issuer response",
        invalid_json_message="Defiance issuer response was not valid JSON.",
    )
    if not isinstance(payload, list):
        raise ValueError("Defiance issuer response must be a JSON array.")

    source_rows: list[tuple[str, object, object]] = []
    for row_number, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Defiance issuer row {row_number} was not a JSON object.")
        source_rows.append((f"row {row_number}", item.get("ticker"), item.get("name")))

    rows = _validated_structured_fund_rows(
        source_rows,
        source_description="Defiance issuer",
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
    source_rows: list[tuple[str, object, object]] = []
    try:
        document = lxml_html.fromstring(html)
    except (TypeError, ValueError):
        script_contents = [html]
    else:
        script_contents = [str(script.text or "") for script in document.xpath("//script")]
        if not script_contents:
            script_contents = [html]

    external_link_key_pattern = re.compile(
        r"(?<![A-Za-z0-9_$])(?:"
        r"(?P<keyquote>['\"])externalLink(?P=keyquote)|externalLink"
        r")(?![A-Za-z0-9_$])\s*:",
    )
    script_object_bodies = [(script_content, _js_object_bodies(script_content)) for script_content in script_contents]
    candidate_object_number = 0
    for _script_content, object_bodies in script_object_bodies:
        for body in object_bodies:
            body_has_ticker = bool(_js_object_fields(body, ["ticker"]))
            body_has_fund_or_name = bool(_js_object_fields(body, ["fund", "name"]))
            nested_bodies = _js_object_bodies(body)
            body_has_nested_ticker = any(_js_object_fields(nested_body, ["ticker"]) for nested_body in nested_bodies)
            body_references_ticker = bool(_js_semantic_identifier_references(body, "ticker"))
            if not (body_has_ticker or body_has_nested_ticker or body_references_ticker and body_has_fund_or_name):
                candidate_object_number += 1
                continue
            alternate_keys, dynamic_property_syntax = _js_object_property_syntax_risk(body)
            if alternate_keys or dynamic_property_syntax:
                raise ValueError(
                    f"{source.name} embedded-JavaScript object {candidate_object_number} used unsupported object "
                    "property syntax that could change its ticker, fund/name, or externalLink fields."
                )
            if body_has_ticker and body_has_nested_ticker:
                raise ValueError(
                    f"{source.name} embedded-JavaScript object {candidate_object_number} contained a nested ticker "
                    "record."
                )
            candidate_object_number += 1

    has_ticker_records = any(
        _js_object_fields(body, ["ticker"])
        for _script_content, object_bodies in script_object_bodies
        for body in object_bodies
    )
    has_non_declarative_ticker_script = False
    has_unsupported_dynamic_syntax = False
    unsafe_features: list[str] = []
    if has_ticker_records:
        member_accesses: set[str] = set()
        semantic_mutations: set[str] = set()
        unsafe_dynamic_capabilities: set[str] = set()
        has_unresolved_member_access = False
        has_template_literal = False
        has_code_slash = False
        has_legacy_html_comment = False
        # Classic scripts share their top-level environment.  Scan the document
        # as one program so an intrinsic alias established by one script cannot
        # hide a prototype mutation performed by a later script.  Semicolon and
        # newline separators keep adjacent script text lexically independent.
        has_object_prototype_mutation = _js_has_object_prototype_mutation("\n;\n".join(script_contents))
        ticker_script_targets: list[tuple[set[str], set[str]]] = []
        for script_content, object_bodies in script_object_bodies:
            script_has_ticker_records = any(_js_object_fields(body, ["ticker"]) for body in object_bodies)
            lexical_bindings: set[str] = set()
            global_properties: set[str] = set()
            script_member_accesses = (
                _js_relevant_member_accesses(script_content) if script_has_ticker_records else set()
            )
            script_semantic_mutations = {
                api_name if property_name is None else f"{api_name}({property_name})"
                for api_name, property_name in _js_semantic_mutation_api_properties(script_content)
                if property_name in {"ticker", "fund", "name", "externalLink"} or property_name is None
            }
            member_accesses.update(script_member_accesses)
            semantic_mutations.update(script_semantic_mutations)
            script_is_declarative = not script_has_ticker_records or _js_ticker_script_is_declarative(
                script_content,
                lexical_bindings=lexical_bindings,
                global_properties=global_properties,
            )
            has_non_declarative_ticker_script = has_non_declarative_ticker_script or not script_is_declarative
            ticker_script_targets.append((lexical_bindings, global_properties))
            if not (script_has_ticker_records or script_member_accesses or script_semantic_mutations):
                continue
            unsafe_dynamic_capabilities.update(_js_unsafe_dynamic_capabilities(script_content))
            has_unresolved_member_access = has_unresolved_member_access or _js_has_unresolved_member_access(
                script_content
            )
            script_positions = _js_code_positions(script_content)
            has_template_literal = has_template_literal or any(
                character == "`" and script_positions[offset] for offset, character in enumerate(script_content)
            )
            has_code_slash = has_code_slash or _js_has_non_comment_slash(script_content)
            has_legacy_html_comment = has_legacy_html_comment or any(
                script_positions[offset]
                and (script_content.startswith("<!--", offset) or script_content.startswith("-->", offset))
                for offset in range(len(script_content))
            )

        for script_index, (script_content, _object_bodies) in enumerate(script_object_bodies):
            other_lexical_names = set().union(
                *(
                    lexical_bindings
                    for target_index, (lexical_bindings, _global_properties) in enumerate(ticker_script_targets)
                    if target_index != script_index
                )
            )
            other_global_names = set().union(
                *(
                    global_properties
                    for target_index, (_lexical_bindings, global_properties) in enumerate(ticker_script_targets)
                    if target_index != script_index
                )
            )
            cross_script_references = _js_cross_script_inventory_references(
                script_content,
                lexical_names=other_lexical_names,
                global_names=other_global_names,
            )
            inventory_names = other_lexical_names | other_global_names
            dynamic_capabilities = _js_unsafe_dynamic_capabilities(script_content) & {
                "Function",
                "constructor",
                "eval",
                "setInterval",
                "setTimeout",
            }
            code_generation_capabilities = dynamic_capabilities & {"Function", "constructor", "eval"}
            dynamic_timer_inventory_reference = bool(dynamic_capabilities & {"setInterval", "setTimeout"}) and (
                _js_dynamic_timer_inventory_reference(
                    script_content,
                    lexical_names=other_lexical_names,
                    global_names=other_global_names,
                )
            )
            mutation_properties = _js_semantic_mutation_api_properties(script_content)
            mutates_global_inventory = bool(other_global_names) and any(
                property_name is None or property_name in other_global_names
                for _api_name, property_name in mutation_properties
            )
            if (
                cross_script_references
                or inventory_names
                and (code_generation_capabilities or dynamic_timer_inventory_reference)
                or other_global_names
                and _js_has_unresolved_member_access(script_content, global_roots_only=True)
                or mutates_global_inventory
            ):
                has_non_declarative_ticker_script = True
        has_unsupported_dynamic_syntax = bool(
            member_accesses
            or semantic_mutations
            or unsafe_dynamic_capabilities
            or has_unresolved_member_access
            or has_template_literal
            or has_code_slash
            or has_legacy_html_comment
            or has_object_prototype_mutation
            or has_non_declarative_ticker_script
        )
        if has_unsupported_dynamic_syntax:
            unsafe_features = sorted(member_accesses | semantic_mutations | unsafe_dynamic_capabilities)
            if has_unresolved_member_access:
                unsafe_features.append("unresolved computed member access")
            if has_template_literal:
                unsafe_features.append("template literals")
            if has_code_slash:
                unsafe_features.append("division or ambiguous regular-expression syntax")
            if has_legacy_html_comment:
                unsafe_features.append("legacy HTML comments")
            if has_object_prototype_mutation:
                unsafe_features.append("Object.prototype mutation")
            if has_non_declarative_ticker_script:
                unsafe_features.append("non-declarative ticker script")

    object_number = 0
    for _script_content, object_bodies in script_object_bodies:
        for body in object_bodies:
            ticker_fields = _js_object_fields(body, ["ticker"])
            if len(ticker_fields) > 1:
                raise ValueError(
                    f"{source.name} embedded-JavaScript object {object_number} contained duplicate or ambiguous "
                    "ticker fields."
                )

            unquoted_positions = _js_unquoted_positions(body)
            external_link_fields = [
                match for match in external_link_key_pattern.finditer(body) if unquoted_positions[match.start()]
            ]
            if len(external_link_fields) > 1:
                raise ValueError(
                    f"{source.name} embedded-JavaScript object {object_number} contained a duplicate or malformed "
                    "externalLink field."
                )
            if external_link_fields:
                if not ticker_fields:
                    contains_nested_ticker = any(
                        _js_object_fields(nested_body, ["ticker"]) for nested_body in _js_object_bodies(body)
                    )
                    if contains_nested_ticker:
                        raise ValueError(
                            f"{source.name} embedded-JavaScript object {object_number} contained an externalLink "
                            "field but omitted its required ticker field."
                        )
                    object_number += 1
                    continue
                field = external_link_fields[0]
                value_end = next(
                    (
                        offset
                        for offset in range(field.end(), len(body))
                        if body[offset] == "," and unquoted_positions[offset]
                    ),
                    len(body),
                )
                value_start = next(
                    (
                        offset
                        for offset in range(field.end(), value_end)
                        if unquoted_positions[offset] and not body[offset].isspace()
                    ),
                    None,
                )
                literal_match = (
                    re.match(r"(?:true|false)(?![A-Za-z0-9_$])", body[value_start:value_end])
                    if value_start is not None
                    else None
                )
                trailing_code = (
                    next(
                        (
                            offset
                            for offset in range(value_start + literal_match.end(), value_end)
                            if unquoted_positions[offset] and not body[offset].isspace()
                        ),
                        None,
                    )
                    if value_start is not None and literal_match is not None
                    else None
                )
                if literal_match is None or trailing_code is not None:
                    raise ValueError(
                        f"{source.name} embedded-JavaScript object {object_number} contained a duplicate or "
                        "malformed externalLink field."
                    )
                literal = literal_match.group()
                if literal == "true":
                    object_number += 1
                    continue

            if not ticker_fields:
                object_number += 1
                continue

            name_fields = _js_object_fields(body, ["fund", "name"])
            if len(name_fields) > 1:
                raise ValueError(
                    f"{source.name} embedded-JavaScript object {object_number} contained duplicate or ambiguous "
                    "fund/name fields."
                )
            if not name_fields:
                raise ValueError(
                    f"{source.name} embedded-JavaScript object {object_number} contained a ticker but omitted its "
                    "required fund/name field."
                )
            source_rows.append(
                (
                    f"embedded-JavaScript object {object_number}",
                    ticker_fields[0][1],
                    name_fields[0][1],
                )
            )
            object_number += 1

    if has_unsupported_dynamic_syntax:
        raise ValueError(
            f"{source.name} embedded JavaScript containing ticker records used unsupported dynamic "
            f"product-field syntax: {', '.join(unsafe_features)}."
        )

    rows = _validated_structured_fund_rows(
        source_rows,
        source_description=f"{source.name} issuer",
    )

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=require_leveraged,
    )


def _leverage_shares_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    """Parse the issuer's complete ``productsData`` JavaScript inventory."""
    try:
        document = lxml_html.fromstring(html)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source.name} issuer response was not valid HTML.") from exc

    assignment_pattern = re.compile(r"\bwindow\.productsData\s*=\s*")

    def assignment_is_direct_statement(script_text: str, assignment_start: int) -> bool:
        """Exclude conditionally controlled single-statement assignments."""
        positions = _js_code_positions(script_text)
        top_level = _js_unquoted_positions(script_text)
        statement_start = max(
            (
                offset + 1
                for offset in range(assignment_start)
                if positions[offset] and top_level[offset] and script_text[offset] == ";"
            ),
            default=0,
        )
        return not any(
            positions[offset] and not script_text[offset].isspace()
            for offset in range(statement_start, assignment_start)
        )

    assignment_locations: list[tuple[str, re.Match[str]]] = []
    product_inventory_references = 0
    product_inventory_mutations = False
    script_records: list[tuple[str, str]] = []
    for script in document.xpath("//script"):
        # HTML template contents are inert, and noscript contents are markup
        # fallback when scripting is disabled. Neither child script executes
        # in the scripting-enabled browser semantics modeled below.
        if any(
            isinstance(ancestor.tag, str) and ancestor.tag.casefold() in {"noscript", "template"}
            for ancestor in script.iterancestors()
        ):
            continue
        script_text = str(script.text or "")
        raw_script_type = script.get("type")
        raw_script_language = script.get("language")
        script_kind = _inline_javascript_kind(
            str(raw_script_type) if raw_script_type is not None else None,
            script_language=str(raw_script_language) if raw_script_language is not None else None,
            has_src=script.get("src") is not None,
            has_nomodule=script.get("nomodule") is not None,
        )
        if script_kind is None:
            continue
        script_records.append((script_text, script_kind))

    # Inline modules are deferred.  They can therefore observe global lexical
    # aliases established by classic scripts that occur later in document
    # order, while classic scripts themselves retain ordinary sequential
    # execution semantics.
    final_global_aliases = {"globalThis", "self", "this", "window"}
    final_object_aliases = {"Object"}
    final_reflect_aliases = {"Reflect"}
    final_mutator_aliases: dict[str, set[str]] = {}
    final_classic_lexical_bindings: set[str] = set()
    for script_text, script_kind in script_records:
        if script_kind != "classic":
            continue
        declared_bindings = _js_top_level_declared_identifiers(script_text, lexical_only=True)
        final_global_aliases.difference_update(declared_bindings)
        final_object_aliases.difference_update(declared_bindings)
        final_reflect_aliases.difference_update(declared_bindings)
        for binding in declared_bindings:
            final_mutator_aliases.pop(binding, None)
        _leverage_shares_classic_script_aliases(
            script_text,
            global_aliases=final_global_aliases,
            object_aliases=final_object_aliases,
            reflect_aliases=final_reflect_aliases,
            mutator_aliases=final_mutator_aliases,
        )
        final_classic_lexical_bindings.update(declared_bindings)

    global_aliases = {"globalThis", "self", "this", "window"}
    object_aliases = {"Object"}
    reflect_aliases = {"Reflect"}
    mutator_aliases: dict[str, set[str]] = {}
    classic_lexical_bindings: set[str] = set()
    for script_text, script_kind in script_records:
        if script_kind == "classic":
            declared_bindings = _js_top_level_declared_identifiers(script_text, lexical_only=True)
            local_bindings = classic_lexical_bindings | declared_bindings
            global_aliases.difference_update(declared_bindings)
            object_aliases.difference_update(declared_bindings)
            reflect_aliases.difference_update(declared_bindings)
            for binding in declared_bindings:
                mutator_aliases.pop(binding, None)
            # Establish top-level aliases before following calls in this same
            # script.  The first pass's script-local result is intentionally
            # discarded; masking parameter scopes below prevents aliases in
            # uncalled/local-shadowed function bodies from becoming global.
            _leverage_shares_classic_script_aliases(
                script_text,
                global_aliases=global_aliases,
                object_aliases=object_aliases,
                reflect_aliases=reflect_aliases,
                mutator_aliases=mutator_aliases,
            )
            (
                analysis_text,
                flowed_global_aliases,
                flowed_object_aliases,
                flowed_reflect_aliases,
                flowed_mutator_aliases,
            ) = _leverage_shares_parameter_scope_analysis(
                script_text,
                global_aliases=global_aliases,
                object_aliases=object_aliases,
                reflect_aliases=reflect_aliases,
                mutator_aliases=mutator_aliases,
            )
            (
                script_global_aliases,
                script_object_aliases,
                script_reflect_aliases,
                script_mutator_aliases,
            ) = _leverage_shares_classic_script_aliases(
                analysis_text,
                global_aliases=global_aliases,
                object_aliases=object_aliases,
                reflect_aliases=reflect_aliases,
                mutator_aliases=mutator_aliases,
            )
            script_global_aliases.update(flowed_global_aliases)
            script_object_aliases.update(flowed_object_aliases)
            script_reflect_aliases.update(flowed_reflect_aliases)
            for alias, mutation_kinds in flowed_mutator_aliases.items():
                script_mutator_aliases.setdefault(alias, set()).update(mutation_kinds)
            classic_lexical_bindings.update(declared_bindings)
        else:
            # Modules can consume browser-global/classic-script aliases, but
            # their own bindings are local to one module and must neither be
            # shadowed by inherited aliases nor leak into later script tags.
            declared_bindings = _js_top_level_declared_identifiers(script_text)
            local_bindings = final_classic_lexical_bindings | declared_bindings
            module_global_aliases = set(final_global_aliases) - declared_bindings
            module_object_aliases = set(final_object_aliases) - declared_bindings
            module_reflect_aliases = set(final_reflect_aliases) - declared_bindings
            module_mutator_aliases = {
                alias: set(kinds) for alias, kinds in final_mutator_aliases.items() if alias not in declared_bindings
            }
            _leverage_shares_classic_script_aliases(
                script_text,
                global_aliases=module_global_aliases,
                object_aliases=module_object_aliases,
                reflect_aliases=module_reflect_aliases,
                mutator_aliases=module_mutator_aliases,
            )
            (
                analysis_text,
                flowed_global_aliases,
                flowed_object_aliases,
                flowed_reflect_aliases,
                flowed_mutator_aliases,
            ) = _leverage_shares_parameter_scope_analysis(
                script_text,
                global_aliases=module_global_aliases,
                object_aliases=module_object_aliases,
                reflect_aliases=module_reflect_aliases,
                mutator_aliases=module_mutator_aliases,
            )
            (
                script_global_aliases,
                script_object_aliases,
                script_reflect_aliases,
                script_mutator_aliases,
            ) = _leverage_shares_classic_script_aliases(
                analysis_text,
                global_aliases=module_global_aliases,
                object_aliases=module_object_aliases,
                reflect_aliases=module_reflect_aliases,
                mutator_aliases=module_mutator_aliases,
            )
            script_global_aliases.update(flowed_global_aliases)
            script_object_aliases.update(flowed_object_aliases)
            script_reflect_aliases.update(flowed_reflect_aliases)
            for alias, mutation_kinds in flowed_mutator_aliases.items():
                script_mutator_aliases.setdefault(alias, set()).update(mutation_kinds)
        top_level_positions = _js_unquoted_positions(script_text)
        assignment_locations.extend(
            (script_text, match)
            for match in assignment_pattern.finditer(analysis_text)
            if (
                top_level_positions[match.start()]
                and "window" in script_global_aliases
                and assignment_is_direct_statement(script_text, match.start())
            )
        )
        script_product_references = _leverage_shares_global_product_references(
            analysis_text,
            global_aliases=script_global_aliases,
            local_bindings=local_bindings,
        )
        product_inventory_references += len(script_product_references)
        product_inventory_mutations = product_inventory_mutations or _js_has_template_interpolation(analysis_text)
        product_inventory_mutations = product_inventory_mutations or _leverage_shares_has_dynamic_global_mutation(
            analysis_text,
            global_aliases=script_global_aliases,
            object_aliases=script_object_aliases,
            reflect_aliases=script_reflect_aliases,
            mutator_aliases=script_mutator_aliases,
        )
        product_inventory_mutations = product_inventory_mutations or _js_has_object_prototype_mutation(analysis_text)
        product_inventory_mutations = product_inventory_mutations or _js_has_unresolved_member_access(
            analysis_text,
            global_roots_only=True,
            global_root_names=(
                script_global_aliases | script_object_aliases | script_reflect_aliases | set(script_mutator_aliases)
            ),
        )
        dynamic_capabilities = _js_unsafe_dynamic_capabilities(analysis_text)
        product_inventory_mutations = product_inventory_mutations or bool(dynamic_capabilities & {"Function", "eval"})
    if len(assignment_locations) != 1 or product_inventory_references != 1 or product_inventory_mutations:
        raise ValueError(f"{source.name} issuer response did not expose exactly one complete product inventory.")
    assignment_text, assignment_match = assignment_locations[0]
    array_start = assignment_match.end()
    if array_start >= len(assignment_text) or assignment_text[array_start] != "[":
        raise ValueError(f"{source.name} complete product inventory was not a JavaScript array.")
    code_positions = _js_code_positions(assignment_text)
    depth = 0
    array_end: int | None = None
    for offset in range(array_start, len(assignment_text)):
        if not code_positions[offset]:
            continue
        character = assignment_text[offset]
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                array_end = offset + 1
                break
    if array_end is None:
        raise ValueError(f"{source.name} complete product inventory was truncated.")
    assignment_terminator = re.match(r"\s*;", assignment_text[array_end:])
    if assignment_terminator is None:
        raise ValueError(f"{source.name} complete product inventory used an unsupported composed assignment.")
    array_text = assignment_text[array_start:array_end]
    array_code_positions = _js_code_positions(array_text)
    product_spans: list[tuple[int, int]] = []
    object_depth = 0
    object_start: int | None = None
    for offset, character in enumerate(array_text):
        if not array_code_positions[offset]:
            continue
        if character == "{":
            if object_depth == 0:
                object_start = offset
            object_depth += 1
        elif character == "}":
            if object_depth == 0:
                raise ValueError(f"{source.name} complete product inventory contained an unmatched object close.")
            object_depth -= 1
            if object_depth == 0:
                assert object_start is not None
                product_spans.append((object_start, offset + 1))
                object_start = None
    if object_depth != 0:
        raise ValueError(f"{source.name} complete product inventory contained a truncated product object.")

    cursor = 1
    for product_index, (start, end) in enumerate(product_spans):
        separator = array_text[cursor:start]
        separator_pattern = r"\s*" if product_index == 0 else r"\s*,\s*"
        if re.fullmatch(separator_pattern, separator) is None:
            raise ValueError(f"{source.name} complete product inventory contained a non-object product row.")
        cursor = end
    if re.fullmatch(r"\s*,?\s*", array_text[cursor:-1]) is None:
        raise ValueError(f"{source.name} complete product inventory contained a non-object product row.")
    if len(product_spans) < _LEVERAGE_SHARES_MINIMUM_PRODUCTS:
        raise ValueError(
            f"{source.name} complete product inventory exposed only {len(product_spans)} products; "
            f"at least {_LEVERAGE_SHARES_MINIMUM_PRODUCTS} are required."
        )

    def name_identity(value: str) -> tuple[tuple[bool, float | None, str | None], tuple[str, ...]]:
        normalized = _normalized_fund_name(value)
        normalized = re.sub(
            r"\b(?:LEVERAGE SHARES|LONG|SHORT|INVERSE|DAILY|TARGET|ETF|ETFS|FUND|HOT|NEW)\b",
            " ",
            normalized,
        )
        normalized = re.sub(r"(?<![A-Z0-9])[+-]?\d+(?:\.\d+)?\s*(?:X|%)(?![A-Z0-9])", " ", normalized)
        semantic_words = tuple(re.findall(r"[A-Z0-9]+", normalized))
        return _classify_leveraged_name(value), semantic_words

    def false_boolean_field(
        body: str,
        properties: Mapping[str, str],
        field_name: str,
        row_number: int,
    ) -> None:
        all_fields = _js_object_fields(body, [field_name])
        if len(all_fields) != 1 or properties.get(field_name) != "false":
            raise ValueError(f"{source.name} complete product row {row_number} must have {field_name}=false.")

    def validate_name_declarations(value: str, row_number: int) -> None:
        normalized = f" {value.upper()} "
        declared_leverages = {leverage for _position, leverage in _numeric_leverage_matches(normalized)}
        declared_directions: set[str] = set()
        if any(re.search(pattern, normalized, re.I) for pattern in LONG_DIRECTION_PATTERNS):
            declared_directions.add("long")
        if any(re.search(pattern, normalized, re.I) for pattern in INVERSE_PATTERNS):
            declared_directions.add("inverse")
        if len(declared_leverages) > 1 or len(declared_directions) > 1:
            raise ValueError(
                f"{source.name} complete product row {row_number} contained contradictory leverage or direction "
                "declarations."
            )

    source_url = urlsplit(source.url)
    source_site = (source_url.hostname or "").casefold().removeprefix("www.")
    source_rows: list[tuple[str, object, object]] = []
    path_by_ticker: dict[str, str] = {}
    ticker_by_path: dict[str, str] = {}
    reference_by_ticker: dict[str, str] = {}
    route_aliases = {
        "NVIDIA": "NVDA",
        "SK HYNIX": "SKHY",
        "TESLA": "TSLA",
        "WORLD STOCK": "WORLD",
    }

    def top_level_properties(body: str, row_number: int) -> dict[str, str]:
        positions = _js_unquoted_positions(body)
        boundaries = [0]
        boundaries.extend(offset + 1 for offset, character in enumerate(body) if character == "," and positions[offset])
        boundaries.append(len(body) + 1)
        key_pattern = re.compile(
            r"\s*(?:"
            r"(?P<quote>['\"])(?P<quoted>[A-Za-z_$][A-Za-z0-9_$]*)(?P=quote)"
            r"|(?P<bare>[A-Za-z_$][A-Za-z0-9_$]*))\s*:"
        )
        properties: dict[str, str] = {}
        for start, next_start in zip(boundaries, boundaries[1:], strict=False):
            segment = body[start : next_start - 1]
            match = key_pattern.match(segment)
            if match is None:
                raise ValueError(f"{source.name} complete product row {row_number} used unsupported object syntax.")
            key = str(match.group("quoted") or match.group("bare"))
            if key in properties:
                raise ValueError(f"{source.name} complete product row {row_number} contained a duplicate property.")
            value = segment[match.end() :].strip()
            if not value:
                raise ValueError(f"{source.name} complete product row {row_number} omitted a property value.")
            properties[key] = value
        return properties

    for row_number, (start, end) in enumerate(product_spans):
        body = array_text[start + 1 : end - 1]
        properties = top_level_properties(body, row_number)
        complete_string_literal = re.compile(r"(?P<quote>['\"])(?:\\.|(?!(?P=quote)).)*(?P=quote)", re.S)
        for required_string_field in (
            "ticker",
            "name",
            "fund",
            "product_url",
            "leverage_factor",
            "category",
        ):
            value = properties.get(required_string_field)
            if value is None or complete_string_literal.fullmatch(value) is None:
                raise ValueError(
                    f"{source.name} complete product row {row_number} omitted a literal {required_string_field} value."
                )
        raw_category2_literal = properties.get("category2")
        if raw_category2_literal is not None and complete_string_literal.fullmatch(raw_category2_literal) is None:
            raise ValueError(
                f"{source.name} complete product row {row_number} had a malformed literal category2 value."
            )
        ticker_fields = _js_object_fields(body, ["ticker"])
        name_fields = _js_object_fields(body, ["name", "fund"])
        url_fields = _js_object_fields(body, ["product_url"])
        factor_fields = _js_object_fields(body, ["leverage_factor"])
        category_fields = _js_object_fields(body, ["category"])
        category2_fields = _js_object_fields(body, ["category2"])
        if len(category2_fields) != (1 if raw_category2_literal is not None else 0):
            raise ValueError(
                f"{source.name} complete product row {row_number} omitted its unique literal category2 value."
            )
        if len(ticker_fields) != 1 or ticker_fields[0][1] is None:
            raise ValueError(f"{source.name} complete product row {row_number} omitted its unique ticker.")
        if (
            len(name_fields) != 2
            or {field_name for field_name, _value in name_fields} != {"name", "fund"}
            or any(value is None for _field_name, value in name_fields)
        ):
            raise ValueError(f"{source.name} complete product row {row_number} omitted its name/fund aliases.")
        canonical_names = {_html_text(value) for _field_name, value in name_fields}
        if len(canonical_names) != 1 or not next(iter(canonical_names), ""):
            raise ValueError(f"{source.name} complete product row {row_number} had conflicting name/fund aliases.")
        if len(url_fields) != 1 or url_fields[0][1] is None:
            raise ValueError(f"{source.name} complete product row {row_number} omitted its unique product URL.")
        if len(factor_fields) != 1 or factor_fields[0][1] is None:
            raise ValueError(f"{source.name} complete product row {row_number} omitted its leverage factor.")
        if len(category_fields) != 1 or category_fields[0][1] is None:
            raise ValueError(f"{source.name} complete product row {row_number} omitted its product category.")
        false_boolean_field(body, properties, "is_api_product", row_number)
        false_boolean_field(body, properties, "externalLink", row_number)

        raw_ticker = ticker_fields[0][1]
        ticker = _exchange_product_symbol_identity(raw_ticker)
        if ticker is None:
            raise ValueError(f"{source.name} complete product row {row_number} had an invalid canonical ticker.")
        name = next(iter(canonical_names))
        validate_name_declarations(name, row_number)
        try:
            factor = float(factor_fields[0][1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{source.name} complete product row {row_number} had an invalid leverage factor."
            ) from exc
        if not math.isfinite(factor) or factor == 0.0 or abs(factor) > MAX_RECOGNIZED_LEVERAGE:
            raise ValueError(f"{source.name} complete product row {row_number} had an invalid leverage factor.")
        expected_category = "Inverse" if factor < 0.0 else "Leveraged"
        if category_fields[0][1] != expected_category:
            raise ValueError(f"{source.name} complete product row {row_number} contradicted its product category.")
        expected_classification = (abs(factor) > 1.0, abs(factor), "inverse" if factor < 0.0 else "long")
        if _classify_leveraged_name(name) != expected_classification:
            raise ValueError(f"{source.name} complete product row {row_number} contradicted its leverage factor.")
        raw_url = url_fields[0][1]
        assert isinstance(raw_url, str)
        if raw_url != raw_url.strip() or any(not character.isprintable() for character in raw_url):
            raise ValueError(f"{source.name} complete product row {row_number} had an unsafe product URL.")
        product_url = urlsplit(raw_url)
        try:
            product_port = product_url.port
        except ValueError as exc:
            raise ValueError(f"{source.name} complete product row {row_number} had an invalid product URL.") from exc
        if (
            product_url.scheme.casefold() != "https"
            or (product_url.hostname or "").casefold().removeprefix("www.") != source_site
            or product_url.username is not None
            or product_url.password is not None
            or product_port not in {None, 443}
            or product_url.query
            or product_url.fragment
        ):
            raise ValueError(f"{source.name} complete product row {row_number} had an invalid product URL.")
        path = product_url.path.rstrip("/").casefold()
        daily_match = re.fullmatch(r"/us/etfs/leverage-shares-(?P<descriptor>[a-z0-9-]+)-daily-etf", path)
        legacy_match = re.fullmatch(
            r"/us/etfs/leverage-shares-(?P<multiple>\d+x)-(?P<underlying>[a-z0-9-]+)-"
            r"(?P<direction>long|short)-etf",
            path,
        )
        if daily_match is not None:
            raw_descriptor_parts = daily_match.group("descriptor").split("-")
            if (
                len(raw_descriptor_parts) == 4
                and raw_descriptor_parts[0].isdigit()
                and raw_descriptor_parts[2].isdigit()
            ):
                slug_name = (
                    f"{raw_descriptor_parts[0]}% {raw_descriptor_parts[1]} + "
                    f"{raw_descriptor_parts[2]}% {raw_descriptor_parts[3]} Daily ETF"
                )
            else:
                descriptor_parts = [f"{part}%" if part.isdigit() else part for part in raw_descriptor_parts]
                slug_name = f"{' '.join(descriptor_parts)} Daily ETF"
        elif legacy_match is not None:
            slug_name = (
                f"{legacy_match.group('multiple')} {legacy_match.group('direction')} "
                f"{legacy_match.group('underlying').replace('-', ' ')} Daily ETF"
            )
        else:
            raise ValueError(f"{source.name} complete product row {row_number} had an unknown product URL path.")
        for route_name, canonical_name in route_aliases.items():
            slug_name = re.sub(rf"\b{re.escape(route_name)}\b", canonical_name, slug_name, flags=re.I)
        display_classification, display_words = name_identity(name)
        route_classification, route_words = name_identity(slug_name)
        category2 = category2_fields[0][1] if raw_category2_literal is not None else None
        category2_identity = _exchange_product_symbol_identity(category2)
        ticker_named_underlying_route = (
            display_classification == route_classification
            and display_words == (ticker,)
            and category2_identity is not None
            and route_words == (category2_identity,)
        )
        if (display_classification, display_words) != (route_classification, route_words) and not (
            ticker_named_underlying_route
        ):
            raise ValueError(f"{source.name} complete product row {row_number} contradicted its product URL.")
        if ticker_named_underlying_route:
            reference_by_ticker[ticker] = category2_identity
        if ticker in path_by_ticker or path in ticker_by_path:
            raise ValueError(f"{source.name} complete product inventory contained a duplicate ticker or product URL.")
        path_by_ticker[ticker] = path
        ticker_by_path[path] = ticker
        source_rows.append((f"complete product row {row_number}", raw_ticker, name))

    rows = _validated_structured_fund_rows(
        source_rows,
        source_description=f"{source.name} complete product inventory",
    )
    out = _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} complete product inventory",
        require_leveraged=require_leveraged,
    )
    if reference_by_ticker:
        out["reference_security"] = out["symbol"].map(reference_by_ticker)
    return out


def _graniteshares_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    rows = []
    product_pattern = re.compile(
        r'etf-table-cell--ticker__symbol">\s*(?P<symbol>[A-Z][A-Z0-9.-]{0,7})\s*</span>'
        r".{0,1800}?etf-table-cell--name-title[^>]*>\s*(?P<name>.*?)\s*</span>",
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


def _cboe_issuer_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    """Parse the explicit product table on a Cboe issuer-listing page."""
    source_rows: list[tuple[str, object, object]] = []
    for table_number, raw_table in enumerate(_read_html_tables(html)):
        table = _clean_columns(raw_table)
        schema = _cboe_issuer_table_schema(table, table_number=table_number)
        if schema is None:
            continue
        symbol_col, name_col, product_type_col = schema
        for row_number, (_index, row) in enumerate(table.iterrows()):
            product_type = _html_text(row[product_type_col]).upper()
            if product_type not in _CBOE_ISSUER_PRODUCT_TYPES:
                displayed_type = repr(product_type) if product_type else "an empty value"
                raise ValueError(
                    f"Cboe issuer listing table {table_number} row {row_number} contained "
                    f"unrecognized product type {displayed_type}."
                )
            if product_type != "ETF":
                continue
            source_rows.append(
                (
                    f"table {table_number} row {row_number}",
                    row[symbol_col],
                    row[name_col],
                )
            )

    rows = _validated_structured_fund_rows(
        source_rows,
        source_description="Cboe issuer listing",
    )

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} Cboe listing",
        require_leveraged=require_leveraged,
    )


def _authoritative_x_multiple(
    value: object,
    *,
    source_description: str,
) -> tuple[float, str]:
    """Parse one complete issuer-supplied X multiple and retain its sign."""
    text = _html_text(value)
    match = re.fullmatch(r"(?P<sign>[+-]?)(?P<multiple>[1-9]\d*(?:\.\d+)?)\s*[Xx]", text)
    if match is None:
        raise ValueError(f"{source_description} must be a complete numeric X multiple; got {text!r}.")
    multiple = float(match.group("multiple"))
    if multiple > MAX_RECOGNIZED_LEVERAGE:
        raise ValueError(
            f"{source_description} contained unsupported leverage {text!r}; "
            f"the maximum recognized multiple is {MAX_RECOGNIZED_LEVERAGE:g}X."
        )
    return multiple, match.group("sign")


def _authoritative_name_leverage_and_direction(
    name: str,
    *,
    source_description: str,
) -> tuple[float | None, str | None]:
    """Reject internally contradictory name metadata before comparing a source field."""
    normalized_name = name.upper()
    numeric_leverages = {leverage for _position, leverage in _numeric_leverage_matches(normalized_name)}
    joined_leverage_matches = list(
        re.finditer(
            r"(?<![A-Z0-9])(?P<sign>[+\uFF0B-]?)\s*(?P<multiple>\d+(?:\.\d+)?)\s*X(?=LEVERAGED\b)",
            normalized_name,
        )
    )
    numeric_leverages.update(float(match.group("multiple")) for match in joined_leverage_matches)
    if len(numeric_leverages) > 1:
        raise ValueError(f"{source_description} product name contained conflicting leverage multiples.")

    explicit_long = bool(
        re.search(r"\bLONG\b(?![-\s]+(?:TERM|DURATION|MUNICIPAL)\b)|\bBULL\b", normalized_name)
        or re.search(EXPLICIT_POSITIVE_LEVERAGE_PATTERN, normalized_name)
        or any(match.group("sign") in {"+", "\uff0b"} for match in joined_leverage_matches)
    )
    explicit_inverse = bool(
        re.search(r"\b(?:BEAR|INVERSE)\b|\bSHORT\b(?![-\s]+(?:TERM|DURATION)\b)", normalized_name)
        or re.search(r"(?<![A-Z0-9])-\d+(?:\.\d+)?\s*(?:X|%)(?![A-Z0-9])", normalized_name)
        or any(match.group("sign") == "-" for match in joined_leverage_matches)
    )
    if explicit_long and explicit_inverse:
        raise ValueError(f"{source_description} product name contained conflicting direction tokens.")
    leverage, direction = infer_leverage_and_direction(name)
    if leverage is None and joined_leverage_matches:
        leverage = next(iter(numeric_leverages))
    if direction is None and joined_leverage_matches:
        direction = "inverse" if explicit_inverse else "long"
    return leverage, direction


_TRADR_COLUMN_LABELS = {
    "symbol": frozenset({"symbol", "ticker", "ticker symbol"}),
    "name": frozenset({"fund", "fund name", "product", "product name"}),
    "reference security": frozenset({"reference security"}),
    "target": frozenset({"target"}),
    "exposure": frozenset({"exposure"}),
    "reset period": frozenset({"reset period", "reset"}),
}

_TRADR_REFERENCE_SECURITY_ALIASES = {
    "INVESCO QQQ®": "QQQ",
}


def _tradr_column_label(column: object) -> str:
    """Recover semantic labels that pandas may hide behind duplicate suffixes."""
    text = str(column).strip()
    mangled = re.fullmatch(r"(?P<label>.+)\.(?:[1-9]\d*)", text)
    if mangled is not None:
        base_label = _normalized_column_label(mangled.group("label"))
        if any(base_label in labels for labels in _TRADR_COLUMN_LABELS.values()):
            return base_label
    return _normalized_column_label(text)


def _tradr_table_schema(
    columns: list[str],
    *,
    table_number: int,
) -> tuple[str, str, str, str, str, str] | None:
    matches = {
        field: [column for column in columns if _tradr_column_label(column) in accepted_labels]
        for field, accepted_labels in _TRADR_COLUMN_LABELS.items()
    }
    if not all(matches.values()):
        product_identity_present = bool(matches["symbol"] or matches["name"])
        current_metadata_present = bool(matches["target"] or matches["exposure"])
        if product_identity_present and current_metadata_present:
            missing = [field for field, field_matches in matches.items() if not field_matches]
            raise ValueError(
                f"Tradr issuer table {table_number} omitted current authoritative columns: {', '.join(missing)}."
            )
        return None
    ambiguous = [field for field, field_matches in matches.items() if len(field_matches) != 1]
    if ambiguous:
        raise ValueError(
            f"Tradr issuer table {table_number} contained ambiguous authoritative columns: {', '.join(ambiguous)}."
        )
    return (
        matches["symbol"][0],
        matches["name"][0],
        matches["reference security"][0],
        matches["target"][0],
        matches["exposure"][0],
        matches["reset period"][0],
    )


def _tradr_reference_security_identity(value: object, *, row_label: str) -> str | None:
    """Decode Tradr's authoritative reference without guessing arbitrary display text."""
    if type(value) is not str:
        raise ValueError(f"Tradr issuer table {row_label} Reference Security must be a string.")
    reference = _html_text(value)
    if reference != value.strip():
        raise ValueError(f"Tradr issuer table {row_label} Reference Security contained noncanonical whitespace.")
    if reference == "-":
        return None
    aliased = _TRADR_REFERENCE_SECURITY_ALIASES.get(reference.upper(), reference)
    identity = _exchange_product_symbol_identity(aliased)
    if identity is None:
        raise ValueError(
            f"Tradr issuer table {row_label} contained an unsupported Reference Security value: {value!r}."
        )
    return identity


def _product_name_rsi_mapping(asset_symbol: str, name: str) -> RsiSymbolMapping:
    """Infer a name's mapping after admitting every canonical ticker it exposes."""
    known_symbols = set(SAFE_GENERIC_RSI_SYMBOLS)
    normalized_name = _normalized_fund_name(name)
    for pattern in RSI_SYMBOL_PATTERNS:
        match = re.search(pattern, normalized_name)
        if match is None:
            continue
        candidate = _normalize_symbol_candidate(match.group(1))
        if candidate is not None:
            known_symbols.add(candidate)
    return infer_rsi_mapping(asset_symbol, name, known_symbols=known_symbols)


def _tradr_reference_contradicts_name(
    asset_symbol: str,
    name: str,
    reference_security: str,
) -> bool:
    mapping = _product_name_rsi_mapping(asset_symbol, name)
    # A self fallback expresses no independent underlying evidence. In that
    # case the issuer's explicit reference is the only authoritative mapping.
    if mapping.mapping_source in {
        "asset_symbol",
        "ambiguous_name_proxy",
        "symbol_override_identity_mismatch",
        "unresolved_basket",
        "unresolved_single_stock",
    }:
        return False
    return mapping.rsi_symbol != reference_security


def _tradr_name_reset_period(name: str, *, row_label: str) -> str:
    reset_periods = set(re.findall(r"\b(?:DAILY|MONTHLY|QUARTERLY)\b", name, re.I))
    normalized_periods = {period.casefold() for period in reset_periods}
    if len(normalized_periods) > 1:
        raise ValueError(f"Tradr issuer table {row_label} product name contained conflicting reset periods.")
    # Daily is Tradr's default objective and is omitted from a small number of
    # current official names (for example TARK). Monthly and quarterly products
    # state their non-default cadence explicitly.
    return next(iter(normalized_periods), "daily")


def _validate_tradr_table_row_widths(html: str) -> None:
    """Reject padded/truncated current rows before pandas erases their physical width."""
    try:
        document = lxml_html.fromstring(html)
    except (TypeError, ValueError) as exc:
        raise ValueError("Tradr issuer response was not parseable HTML.") from exc
    tables = [document] if str(document.tag).casefold() == "table" else document.xpath(".//table")
    for table_number, table in enumerate(tables):
        table_rows = table.xpath(".//tr")
        for header_row_number, table_row in enumerate(table_rows):
            header_cells = table_row.xpath("./th|./td")
            header_labels = [_normalized_column_label(" ".join(cell.itertext())) for cell in header_cells]
            current_semantics = {"target", "exposure"}
            product_semantics = {"ticker", "ticker symbol", "symbol", "fund name", "product name"}
            if not current_semantics.intersection(header_labels) or not product_semantics.intersection(header_labels):
                continue
            declared_width = len(header_cells)
            for row_number, data_row in enumerate(table_rows[header_row_number + 1 :]):
                data_cells = data_row.xpath("./td")
                if data_cells and len(data_cells) != declared_width:
                    raise ValueError(
                        f"Tradr issuer table {table_number} row {row_number} did not match its declared columns."
                    )
            break


def _tradr_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    """Parse Tradr's current product tables without accepting its stale legacy table."""
    _validate_tradr_table_row_widths(html)
    source_rows: list[tuple[str, object, object]] = []
    references_by_symbol: dict[str, set[str]] = {}
    for table_number, raw_table in enumerate(_read_html_tables(html)):
        table = _clean_columns(raw_table)
        columns = list(table.columns)
        schema = _tradr_table_schema(columns, table_number=table_number)
        if schema is None:
            continue
        symbol_col, name_col, reference_col, target_col, exposure_col, reset_col = schema

        for row_number, (_index, row) in enumerate(table.iterrows()):
            row_label = f"table {table_number} row {row_number}"
            raw_name = row[name_col]
            if type(raw_name) is not str or not (name := _html_text(raw_name)):
                raise ValueError(f"Tradr issuer table {row_label} omitted its required product name.")
            raw_symbol = row[symbol_col]
            symbol_identity = _exchange_product_symbol_identity(raw_symbol)
            if symbol_identity is None:
                raise ValueError(f"Tradr issuer table {row_label} contained an invalid or noncanonical product ticker.")
            reference_security = _tradr_reference_security_identity(
                row[reference_col],
                row_label=row_label,
            )
            if reference_security is not None:
                if _tradr_reference_contradicts_name(
                    symbol_identity,
                    name,
                    reference_security,
                ):
                    raise ValueError(
                        f"Tradr issuer table {row_label} Reference Security contradicted its product name."
                    )
                references = references_by_symbol.setdefault(symbol_identity, set())
                references.add(reference_security)
                if len(references) != 1:
                    raise ValueError(
                        f"Tradr issuer table contained conflicting Reference Security values for {symbol_identity}."
                    )

            target, target_sign = _authoritative_x_multiple(
                row[target_col],
                source_description=f"Tradr issuer table {row_label} Target",
            )
            exposure = _html_text(row[exposure_col]).casefold()
            if exposure not in {"long", "short"}:
                raise ValueError(f"Tradr issuer table {row_label} Exposure must be Long or Short; got {exposure!r}.")
            direction = "inverse" if exposure == "short" else "long"
            if (target_sign == "-" and direction != "inverse") or (target_sign == "+" and direction != "long"):
                raise ValueError(f"Tradr issuer table {row_label} Target sign contradicted its Exposure value.")

            name_leverage, name_direction = _authoritative_name_leverage_and_direction(
                name,
                source_description=f"Tradr issuer table {row_label}",
            )
            if name_leverage != target:
                raise ValueError(f"Tradr issuer table {row_label} Target leverage contradicted its product name.")
            if name_direction != direction:
                raise ValueError(f"Tradr issuer table {row_label} Exposure direction contradicted its product name.")

            reset_period = _html_text(row[reset_col]).casefold()
            if reset_period not in {"daily", "monthly", "quarterly"}:
                raise ValueError(
                    f"Tradr issuer table {row_label} Reset Period must be Daily, Monthly, or Quarterly; "
                    f"got {reset_period!r}."
                )
            if _tradr_name_reset_period(name, row_label=row_label) != reset_period:
                raise ValueError(f"Tradr issuer table {row_label} Reset Period contradicted its product name.")
            source_rows.append((row_label, raw_symbol, name))

    rows = _validated_structured_fund_rows(
        source_rows,
        source_description="Tradr issuer table",
    )

    out = _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=require_leveraged,
    )
    out["reference_security"] = out["symbol"].map(
        lambda symbol: next(iter(references_by_symbol.get(str(symbol), set())), pd.NA)
    )
    return out


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


def _etracs_ticker_token_identity(value: str) -> str:
    """Validate one complete ETRACS ticker spelling or product URL."""
    token = unescape(value).strip()
    parsed = urlsplit(token)
    path_match = re.fullmatch(
        r"(?:.*/)?ussymbol/(?P<symbol>[A-Z][A-Z0-9]*(?:[.-][A-Z0-9]+)?)/?",
        parsed.path,
    )
    if path_match is not None and not parsed.query and not parsed.fragment:
        candidate = path_match.group("symbol")
    elif not parsed.scheme and not parsed.netloc and "/" not in token:
        candidate = token
    else:
        raise ValueError(f"invalid ETRACS ticker identity {token!r}")

    if _exchange_product_symbol_identity(candidate) is None:
        raise ValueError(f"invalid ETRACS ticker identity {token!r}")
    return candidate


def _etracs_ticker_cell_identity(cell: object) -> str:
    """Require every material identity in a ticker cell to agree exactly.

    The live page can expose the ticker once in a product URL and again as
    visible text. Parsing the DOM keeps those independent identities separate;
    flattening the cell and halving a repeated string could instead turn a real
    six-character ticker into an unrelated three-character security.
    """
    hrefs = [str(value) for value in cell.xpath(".//*[@href]/@href")]
    text_tokens = [str(value).strip() for value in cell.xpath(".//text()") if str(value).strip()]
    identities = [_etracs_ticker_token_identity(value) for value in [*hrefs, *text_tokens]]
    if not identities:
        raise ValueError("ETRACS ticker cell omitted its authoritative identity")
    if len(set(identities)) != 1:
        raise ValueError(f"ETRACS ticker cell contained conflicting identities: {identities!r}")
    return identities[0]


def _etracs_leverage_table_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool = True,
) -> pd.DataFrame:
    source_rows: list[tuple[str, object, object]] = []
    try:
        document = lxml_html.fromstring(html)
    except (TypeError, ValueError) as exc:
        raise ValueError("ETRACS issuer response was not parseable HTML.") from exc
    tables = [document] if str(document.tag).casefold() == "table" else document.xpath(".//table")
    for table_number, table in enumerate(tables):
        table_rows = table.xpath(".//tr")
        schema: tuple[int, int, int, int, int] | None = None
        for header_row_number, table_row in enumerate(table_rows):
            cells = table_row.xpath("./th|./td")
            labels = [_normalized_column_label(" ".join(cell.itertext())) for cell in cells]

            def matching_indexes(patterns: list[str], *, cell_labels: list[str] = labels) -> list[int]:
                return [
                    index
                    for index, label in enumerate(cell_labels)
                    if any(re.search(pattern, label, re.I) for pattern in patterns)
                ]

            symbol_indexes = matching_indexes([r"ticker\s+symbol", r"^ticker$", r"^symbol$"])
            name_indexes = matching_indexes([r"^name$", r"\bname\b"])
            leverage_indexes = matching_indexes([r"^leverage$"])
            if symbol_indexes and name_indexes and not leverage_indexes:
                raise ValueError(f"ETRACS issuer table {table_number} omitted its authoritative Leverage column.")
            if symbol_indexes and name_indexes and leverage_indexes:
                if any(len(indexes) != 1 for indexes in (symbol_indexes, name_indexes, leverage_indexes)):
                    raise ValueError(f"ETRACS issuer table {table_number} contained ambiguous authoritative columns.")
                schema = (
                    symbol_indexes[0],
                    name_indexes[0],
                    leverage_indexes[0],
                    header_row_number,
                    len(cells),
                )
                break
        if schema is None:
            continue

        symbol_index, name_index, leverage_index, header_row_number, declared_width = schema
        required_index = max(symbol_index, name_index, leverage_index)
        for row_number, table_row in enumerate(table_rows[header_row_number + 1 :]):
            cells = table_row.xpath("./td")
            if not cells:
                continue
            if len(cells) <= required_index or len(cells) != declared_width:
                raise ValueError(
                    f"ETRACS issuer table {table_number} row {row_number} did not match its declared columns."
                )
            try:
                symbol = _etracs_ticker_cell_identity(cells[symbol_index])
            except ValueError as exc:
                raise ValueError(
                    f"ETRACS issuer table {table_number} row {row_number} contained an invalid ticker cell: {exc}"
                ) from exc

            row_label = f"table {table_number} row {row_number}"
            name = _html_text(" ".join(cells[name_index].itertext()))
            if not name:
                raise ValueError(f"ETRACS issuer table {row_label} omitted its required product name.")
            leverage_text = _html_text(" ".join(cells[leverage_index].itertext()))
            name_leverage, name_direction = _authoritative_name_leverage_and_direction(
                name,
                source_description=f"ETRACS issuer table {row_label}",
            )
            if leverage_text == "--":
                if name_leverage is not None or name_direction is not None:
                    raise ValueError(f"ETRACS issuer table {row_label} Leverage value contradicted its product name.")
                if require_leveraged:
                    continue
            else:
                leverage, leverage_sign = _authoritative_x_multiple(
                    leverage_text,
                    source_description=f"ETRACS issuer table {row_label} Leverage",
                )
                direction = "inverse" if leverage_sign == "-" else "long"
                if name_leverage is not None and name_leverage != leverage:
                    raise ValueError(f"ETRACS issuer table {row_label} Leverage value contradicted its product name.")
                if name_direction is not None and name_direction != direction:
                    raise ValueError(
                        f"ETRACS issuer table {row_label} Leverage direction contradicted its product name."
                    )
                if name_leverage is None or name_direction is None or not leveraged_name_filter(name):
                    direction_label = "Inverse " if direction == "inverse" else ""
                    name = f"{name} {leverage:g}X {direction_label}Leveraged"
                if require_leveraged and leverage <= 1.0:
                    continue

            source_rows.append((row_label, symbol, name))

    rows = _validated_structured_fund_rows(
        source_rows,
        source_description="ETRACS issuer table",
    )

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
    out = pd.concat(rows, ignore_index=True)
    dedupe_columns = "symbol" if require_leveraged else ["symbol", "name"]
    out = out.drop_duplicates(dedupe_columns)
    return out.reset_index(drop=True)


def _linked_body_cell(value: object) -> tuple[object, str | None]:
    if isinstance(value, tuple) and len(value) == 2:
        text, href = value
        return text, href if isinstance(href, str) and href else None
    return value, None


def _issuer_product_link_slug(
    href: str | None,
    source: UniverseSource,
    *,
    expected_path_prefix: tuple[str, ...],
    row_label: str,
) -> str:
    if href is None:
        raise ValueError(f"{source.name} issuer table {row_label} omitted its required product link.")
    linked_url = urlsplit(urljoin(source.url, href))
    source_url = urlsplit(source.url)
    try:
        linked_port = linked_url.port
        source_port = source_url.port
    except ValueError as exc:
        raise ValueError(f"{source.name} issuer table {row_label} contained an invalid product link.") from exc

    def site_identity(hostname: str | None) -> str:
        return (hostname or "").casefold().removeprefix("www.")

    if (
        linked_url.scheme.casefold() != "https"
        or site_identity(linked_url.hostname) != site_identity(source_url.hostname)
        or linked_url.username is not None
        or linked_url.password is not None
        or linked_port not in {None, 443}
        or source_port not in {None, 443}
        or linked_url.query
        or linked_url.fragment
    ):
        raise ValueError(f"{source.name} issuer table {row_label} contained an invalid product link.")

    path_parts = tuple(part for part in linked_url.path.split("/") if part)
    if len(path_parts) != len(expected_path_prefix) + 1 or path_parts[:-1] != expected_path_prefix:
        raise ValueError(f"{source.name} issuer table {row_label} contained an invalid product link path.")
    slug = path_parts[-1]
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,13}", slug) is None:
        raise ValueError(f"{source.name} issuer table {row_label} contained an invalid product-link ticker.")
    return slug


def _canonical_linked_issuer_ticker(
    raw_symbol: object,
    href: str | None,
    source: UniverseSource,
    *,
    row_label: str,
) -> object:
    if source.parser == "innovator_html":
        if not isinstance(raw_symbol, str) or re.fullmatch(r"[a-z][a-z0-9]*(?:[.-][a-z0-9]+)?", raw_symbol) is None:
            return raw_symbol
        canonical_symbol = raw_symbol.upper()
        if _exchange_product_symbol_identity(canonical_symbol) is None:
            return raw_symbol
        link_symbol = _issuer_product_link_slug(
            href,
            source,
            expected_path_prefix=(),
            row_label=row_label,
        )
        if link_symbol != raw_symbol:
            raise ValueError(f"Innovator issuer table {row_label} product link contradicted its displayed ticker.")
        return canonical_symbol

    if source.parser == "yieldmax_html" and isinstance(raw_symbol, str) and raw_symbol.endswith("*"):
        canonical_symbol = raw_symbol[:-1]
        if raw_symbol.count("*") != 1 or _exchange_product_symbol_identity(canonical_symbol) != canonical_symbol:
            return raw_symbol
        link_symbol = _issuer_product_link_slug(
            href,
            source,
            expected_path_prefix=("our-etfs",),
            row_label=row_label,
        )
        if link_symbol.casefold() != canonical_symbol.casefold():
            raise ValueError(f"YieldMax issuer table {row_label} product link contradicted its displayed ticker.")
        return canonical_symbol

    return raw_symbol


def _linked_issuer_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool,
) -> pd.DataFrame:
    rows = []
    for table_number, raw_table in enumerate(_read_html_tables_with_body_links(html)):
        table = _clean_columns(raw_table)
        schema = _fund_table_schema(table, source.name)
        if schema is None:
            continue
        symbol_col, name_col = schema
        source_rows = []
        for row_number, (_index, row) in enumerate(table[[symbol_col, name_col]].iterrows()):
            row_label = f"table {table_number} row {row_number}"
            raw_symbol, symbol_href = _linked_body_cell(row[symbol_col])
            raw_name, _name_href = _linked_body_cell(row[name_col])
            source_rows.append(
                (
                    row_label,
                    _canonical_linked_issuer_ticker(
                        raw_symbol,
                        symbol_href,
                        source,
                        row_label=row_label,
                    ),
                    raw_name,
                )
            )
        validated_rows = _validated_structured_fund_rows(
            source_rows,
            source_description=f"{source.name} issuer table",
            reject_conflicting_duplicates=require_leveraged,
        )
        source_frame = _fund_rows_to_universe(
            validated_rows,
            source.name,
            source_label=f"{source.name} issuer table",
            require_leveraged=require_leveraged,
        )
        if not source_frame.empty:
            rows.append(source_frame)

    card_rows = _html_cards_to_universe(
        html,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=require_leveraged,
    )
    if not card_rows.empty:
        rows.append(card_rows)

    if not rows:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
    out = pd.concat(rows, ignore_index=True)
    dedupe_columns = "symbol" if require_leveraged else ["symbol", "name"]
    return out.drop_duplicates(dedupe_columns).reset_index(drop=True)


def _yieldmax_html_to_universe(
    html: str,
    source: UniverseSource,
    *,
    require_leveraged: bool,
) -> pd.DataFrame:
    """Parse YieldMax's complete linked tables or homepage fund cards."""
    table_rows = _linked_issuer_html_to_universe(
        html,
        source,
        require_leveraged=False,
    )

    try:
        document = lxml_html.fromstring(html)
    except (lxml_html.ParserError, TypeError, ValueError):
        document = None

    cards = []
    source_rows: list[tuple[str, object, object]] = []
    if document is not None:
        class_token = './/*[contains(concat(" ", normalize-space(@class), " "), " {class_name} ")]'
        cards = document.xpath(
            'self::*[contains(concat(" ", normalize-space(@class), " "), " ym-fslider-card ")] | '
            + class_token.format(class_name="ym-fslider-card")
        )
        for card_number, card in enumerate(cards):
            row_label = f"homepage card {card_number}"
            ticker_nodes = card.xpath(class_token.format(class_name="ym-fslider-ticker"))
            name_nodes = card.xpath(class_token.format(class_name="ym-fslider-name"))
            if len(ticker_nodes) != 1 or len(name_nodes) != 1:
                raise ValueError(f"YieldMax issuer {row_label} must contain exactly one ticker and product name.")

            fund_page_hrefs = []
            for raw_href in card.xpath(".//a[@href]/@href"):
                if not isinstance(raw_href, str):
                    continue
                try:
                    path_parts = tuple(part for part in urlsplit(urljoin(source.url, raw_href)).path.split("/") if part)
                except ValueError:
                    continue
                if len(path_parts) == 2 and path_parts[0] == "our-etfs":
                    fund_page_hrefs.append(raw_href)
            if len(fund_page_hrefs) != 1:
                raise ValueError(f"YieldMax issuer {row_label} must contain exactly one product link.")

            raw_symbol = _html_text(" ".join(ticker_nodes[0].itertext()))
            product_href = fund_page_hrefs[0]
            canonical_symbol = _canonical_linked_issuer_ticker(
                raw_symbol,
                product_href,
                source,
                row_label=row_label,
            )
            linked_symbol = _issuer_product_link_slug(
                product_href,
                source,
                expected_path_prefix=("our-etfs",),
                row_label=row_label,
            )
            if not isinstance(canonical_symbol, str) or linked_symbol.casefold() != canonical_symbol.casefold():
                raise ValueError(f"YieldMax issuer {row_label} product link contradicted its displayed ticker.")
            source_rows.append(
                (
                    row_label,
                    canonical_symbol,
                    _html_text(" ".join(name_nodes[0].itertext())),
                )
            )

    validated_card_rows = _validated_structured_fund_rows(
        source_rows,
        source_description="YieldMax issuer homepage",
    )
    if cards and len(validated_card_rows) < _YIELDMAX_MINIMUM_PRODUCTS:
        raise ValueError(
            f"{source.name} complete product inventory exposed only {len(validated_card_rows)} homepage products; "
            f"at least {_YIELDMAX_MINIMUM_PRODUCTS} are required."
        )
    card_rows = _fund_rows_to_universe(
        validated_card_rows,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=False,
    )
    frames = [frame for frame in (table_rows, card_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])

    out = pd.concat(frames, ignore_index=True, sort=False)
    if require_leveraged:
        out = out.loc[out["name"].map(leveraged_name_filter).astype(bool)].copy()
    dedupe_columns = "symbol" if require_leveraged else ["symbol", "name"]
    return out.drop_duplicates(dedupe_columns).reset_index(drop=True)


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
    if source.parser == "leverage_shares_html":
        return _leverage_shares_html_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "graniteshares_html":
        return _graniteshares_html_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "rex_menu_html":
        return _rex_menu_html_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "cboe_issuer_html":
        return _cboe_issuer_html_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "tradr_html":
        return _tradr_html_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "volatilityshares_html":
        return _volatilityshares_html_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "yieldmax_html":
        return _yieldmax_html_to_universe(content, source, require_leveraged=require_leveraged)
    if source.parser == "innovator_html":
        return _linked_issuer_html_to_universe(content, source, require_leveraged=require_leveraged)
    return _html_source_to_universe(
        content,
        source.name,
        source_label=f"{source.name} issuer table",
        require_leveraged=require_leveraged,
    )


def _issuer_mapping_signatures(symbol: str, rows: pd.DataFrame) -> set[tuple[str, bool, bool]]:
    """Return mapping/executability outcomes possible across duplicate metadata."""
    known_symbols = set(SAFE_GENERIC_RSI_SYMBOLS)
    known_symbols.add(symbol)
    for name in rows["name"]:
        normalized_name = _normalized_fund_name(name)
        for pattern in RSI_SYMBOL_PATTERNS:
            match = re.search(pattern, normalized_name)
            if match is None:
                continue
            candidate = _normalize_symbol_candidate(match.group(1))
            if candidate is not None:
                known_symbols.add(candidate)

    mappings: list[RsiSymbolMapping] = []
    for _index, row in rows.iterrows():
        mappings.append(_product_row_rsi_mapping(row, known_symbols=known_symbols))

    # A source that supplies a validated Reference Security can add mapping
    # evidence which a primary Nasdaq row simply does not contain.  Treat an
    # unresolved self-mapping as missing information in that case, not as a
    # contradiction.  Concrete inferred/curated mappings still participate,
    # so two different references or a reference which disagrees with a
    # concrete primary mapping remain conflicts.
    has_validated_reference = any(mapping.mapping_source == "issuer_reference" for mapping in mappings)
    unresolved_mapping_sources = {
        "ambiguous_name_proxy",
        "asset_symbol",
        "symbol_override_identity_mismatch",
        "unresolved_basket",
        "unresolved_single_stock",
    }
    return {
        (
            mapping.rsi_symbol,
            mapping.confidence == "needs_review",
            mapping.mapping_source == "issuer_reference_conflict",
        )
        for mapping in mappings
        if not (has_validated_reference and mapping.mapping_source in unresolved_mapping_sources)
    }


def _workflow_product_structure(value: object) -> str | None:
    """Reduce source-decorated fund types to their legal product structure."""
    if not isinstance(value, str):
        return None
    match = re.match(r"^(ETF|ETN)\b", _normalized_fund_name(value))
    return match.group(1) if match is not None else None


def _missing_reference_security(value: object) -> bool:
    """Return whether a reference cell contains a scalar null sentinel."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if getattr(missing, "shape", ()) != ():
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _canonical_product_metadata_value(value: object) -> tuple[str, str, str]:
    """Return a stable comparison key for scalar product metadata."""
    if _missing_reference_security(value):
        return ("", "", "")
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    text = value if isinstance(value, str) else repr(value)
    return (value_type, text.casefold(), text)


def _canonical_product_row_key(
    row: pd.Series,
    *,
    validated_reference: str | None,
) -> tuple[object, ...]:
    """Choose equivalent product metadata without depending on feed order."""
    raw_reference = row.get("reference_security")
    supplies_reference = (
        validated_reference is not None and isinstance(raw_reference, str) and raw_reference == validated_reference
    )
    raw_source = row.get("source")
    internal_columns = {
        "_leverage_signature",
        "_product_structure_signature",
        "_workflow_source_status_index",
    }
    metadata = tuple(
        (str(column), _canonical_product_metadata_value(row[column]))
        for column in sorted(
            (column for column in row.index if str(column) not in internal_columns),
            key=str,
        )
    )
    return (
        0 if supplies_reference else 1,
        1 if _missing_reference_security(raw_source) else 0,
        _canonical_product_metadata_value(raw_source),
        metadata,
    )


def _leveraged_product_rows(parsed_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if parsed_rows.empty:
        return parsed_rows.copy().reset_index(drop=True), []

    # Resolve duplicate symbols before leverage filtering. Metadata that changes
    # legal product structure, leverage/direction, the inferred RSI underlying,
    # or whether a mapping needs review must not be selected by input order.
    # Exact and behaviorally equivalent duplicates select canonical provenance
    # deterministically, preferring a row which supplies their validated issuer
    # reference so source-decorated metadata stays aligned with the enrichment.
    # Source labels decorating ``fund_type`` do not create a conflict:
    # ``ETF (Issuer A)`` and ``ETF (Issuer B)`` are both ETFs.
    classified_rows = parsed_rows.copy()
    classified_rows["_leverage_signature"] = classified_rows["name"].map(_classify_leveraged_name)
    classified_rows["_product_structure_signature"] = (
        classified_rows["fund_type"].map(_workflow_product_structure)
        if "fund_type" in classified_rows.columns
        else None
    )
    conflicting_symbols: list[str] = []
    validated_references: dict[str, str] = {}
    for raw_symbol, symbol_rows in classified_rows.groupby("symbol", sort=False):
        leverage_signatures = set(symbol_rows["_leverage_signature"])
        leverage_eligibilities = {signature[0] for signature in leverage_signatures}
        if len(leverage_eligibilities) > 1:
            conflicting_symbols.append(str(raw_symbol))
            continue

        all_rows_leveraged = leverage_eligibilities == {True}
        product_structure_signatures = set(symbol_rows["_product_structure_signature"]) if all_rows_leveraged else set()
        mapping_signatures = _issuer_mapping_signatures(str(raw_symbol), symbol_rows) if all_rows_leveraged else set()
        present_references = (
            [reference for reference in symbol_rows["reference_security"] if not _missing_reference_security(reference)]
            if all_rows_leveraged and "reference_security" in symbol_rows.columns
            else []
        )
        references = {reference for reference in present_references if isinstance(reference, str) and reference}
        invalid_reference = any(
            not isinstance(reference, str) or not reference for reference in present_references
        ) or any(_exchange_product_symbol_identity(reference) != reference for reference in references)
        if all_rows_leveraged and (
            len(leverage_signatures) > 1
            or len(product_structure_signatures) > 1
            or len(mapping_signatures) > 1
            or any(reference_conflict for _symbol, _needs_review, reference_conflict in mapping_signatures)
            or len(references) > 1
            or invalid_reference
        ):
            conflicting_symbols.append(str(raw_symbol))
            continue
        if len(references) == 1:
            reference = next(iter(references))
            if _exchange_product_symbol_identity(reference) == reference:
                validated_references[str(raw_symbol)] = reference
    conflicting_symbols.sort()
    compatible_rows = classified_rows.loc[~classified_rows["symbol"].astype(str).isin(conflicting_symbols)]
    canonical_row_frames: list[pd.DataFrame] = []
    for raw_symbol, symbol_rows in compatible_rows.groupby("symbol", sort=False, dropna=False):
        validated_reference = validated_references.get(str(raw_symbol))
        canonical_offset = min(
            range(len(symbol_rows)),
            key=lambda offset: _canonical_product_row_key(
                symbol_rows.iloc[offset],
                validated_reference=validated_reference,
            ),
        )
        canonical_row_frames.append(symbol_rows.iloc[[canonical_offset]])
    canonical_rows = (
        pd.concat(canonical_row_frames, axis=0) if canonical_row_frames else compatible_rows.iloc[0:0].copy()
    ).drop(columns=["_leverage_signature", "_product_structure_signature"])
    if validated_references:
        canonical_rows["reference_security"] = (
            canonical_rows["symbol"]
            .astype(str)
            .map(validated_references)
            .combine_first(canonical_rows["reference_security"])
        )
    is_leveraged = canonical_rows["name"].map(leveraged_name_filter).astype(bool)
    return canonical_rows.loc[is_leveraged].copy().reset_index(drop=True), conflicting_symbols


def _classification_conflict_error(conflicting_symbols: list[str]) -> str:
    symbols = ", ".join(conflicting_symbols)
    return (
        "Conflicting leverage classifications, product structures, or RSI-mapping metadata were parsed for "
        "duplicate symbols "
        f"{symbols}; ambiguous symbols were excluded."
    )[:250]


_WORKFLOW_SYMBOL_SOURCES_ATTR = "workflow_symbol_sources"
_WORKFLOW_CANONICAL_ROWS_ATTR = "workflow_canonical_product_rows"
_WORKFLOW_SOURCE_STATUS_INDEX_COLUMN = "_workflow_source_status_index"


def _cross_source_classification_conflict_error(conflicting_symbols: list[str]) -> str:
    symbols = ", ".join(conflicting_symbols)
    return (
        "Conflicting leverage classifications, product structures, or RSI-mapping metadata were returned across "
        "workflow "
        f"sources for duplicate symbols {symbols}; ambiguous symbols were excluded."
    )[:250]


def _resolve_workflow_source_product_rows(
    tagged_source_rows: list[tuple[int, UniverseSource, pd.DataFrame]],
    status_rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Resolve duplicates across independent feeds without input-order choices."""
    if not tagged_source_rows:
        out = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        out.attrs[_WORKFLOW_SYMBOL_SOURCES_ATTR] = {}
        out.attrs[_WORKFLOW_CANONICAL_ROWS_ATTR] = []
        return out

    tagged_frames: list[pd.DataFrame] = []
    source_ref_by_status_index: dict[int, tuple[str, str]] = {}
    for status_index, source, source_rows in tagged_source_rows:
        tagged = source_rows.copy()
        tagged[_WORKFLOW_SOURCE_STATUS_INDEX_COLUMN] = status_index
        tagged_frames.append(tagged)
        source_ref_by_status_index[status_index] = (source.name, source.url)

    combined = pd.concat(tagged_frames, ignore_index=True, sort=False)
    source_indexes_by_symbol = {
        str(symbol): tuple(dict.fromkeys(int(value) for value in rows[_WORKFLOW_SOURCE_STATUS_INDEX_COLUMN]))
        for symbol, rows in combined.groupby("symbol", sort=False)
    }
    resolved, conflicting_symbols = _leveraged_product_rows(combined)
    if conflicting_symbols:
        conflicting_symbols_by_status_index: dict[int, list[str]] = {}
        for symbol in conflicting_symbols:
            for status_index in source_indexes_by_symbol.get(symbol, ()):
                conflicting_symbols_by_status_index.setdefault(status_index, []).append(symbol)
        for status_index, source_conflicting_symbols in sorted(conflicting_symbols_by_status_index.items()):
            conflict_error = _cross_source_classification_conflict_error(source_conflicting_symbols)
            status = status_rows[status_index]
            status["status"] = "parse_error"
            previous_error = str(status.get("error") or "").strip()
            status["error"] = safe_diagnostic_text(
                f"{conflict_error}; {previous_error}" if previous_error else conflict_error,
                max_chars=_UNIVERSE_DIAGNOSTIC_MAX_CHARS,
            )

    symbol_sources = {
        symbol: tuple(source_ref_by_status_index[index] for index in source_indexes)
        for symbol, source_indexes in source_indexes_by_symbol.items()
    }
    canonical_rows = combined.drop(
        columns=_WORKFLOW_SOURCE_STATUS_INDEX_COLUMN,
        errors="ignore",
    ).to_dict("records")
    resolved = resolved.drop(columns=_WORKFLOW_SOURCE_STATUS_INDEX_COLUMN, errors="ignore")
    resolved = resolved.sort_values("symbol").reset_index(drop=True)
    resolved.attrs[_WORKFLOW_SYMBOL_SOURCES_ATTR] = symbol_sources
    resolved.attrs[_WORKFLOW_CANONICAL_ROWS_ATTR] = canonical_rows
    return resolved


def _resolve_discovered_product_rows(
    nasdaq_rows: pd.DataFrame,
    issuer_rows: pd.DataFrame,
    etn_rows: pd.DataFrame,
    workflow_source_status: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Revalidate product identities across primary and discovered sources."""
    discovered_source_frames = (issuer_rows, etn_rows)
    frames: list[pd.DataFrame] = []
    for frame in discovered_source_frames:
        canonical_records = frame.attrs.get(_WORKFLOW_CANONICAL_ROWS_ATTR)
        canonical_frame = (
            pd.DataFrame.from_records(canonical_records) if isinstance(canonical_records, list) else frame.copy()
        )
        if not canonical_frame.empty:
            frames.append(canonical_frame)

    if frames:
        discovered_rows = pd.concat(frames, ignore_index=True, sort=False)
        resolved_discovered, _discovered_conflicting_symbols = _leveraged_product_rows(discovered_rows)
    else:
        discovered_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        resolved_discovered = discovered_rows.copy()

    identity_frames = [frame for frame in (nasdaq_rows.copy(), discovered_rows) if not frame.empty]
    if identity_frames:
        _resolved_workflow_rows, conflicting_symbols = _leveraged_product_rows(
            pd.concat(identity_frames, ignore_index=True, sort=False)
        )
    else:
        conflicting_symbols = []
    if conflicting_symbols:
        source_refs_by_symbol: dict[str, set[tuple[str, str]]] = {}
        for frame in discovered_source_frames:
            for symbol, refs in frame.attrs.get(_WORKFLOW_SYMBOL_SOURCES_ATTR, {}).items():
                source_refs_by_symbol.setdefault(str(symbol), set()).update(tuple(ref) for ref in refs)

        primary_symbols = set(nasdaq_rows["symbol"].dropna().astype(str)) if "symbol" in nasdaq_rows.columns else set()
        for symbol in conflicting_symbols:
            if symbol in primary_symbols:
                source_refs_by_symbol.setdefault(symbol, set()).add((NASDAQ_ETF_SOURCE_NAME, ETF_DEFS_URL))

        conflicting_symbols_by_source_ref: dict[tuple[str, str], list[str]] = {}
        for symbol in conflicting_symbols:
            for source_ref in source_refs_by_symbol.get(symbol, set()):
                conflicting_symbols_by_source_ref.setdefault(source_ref, []).append(symbol)

        status = workflow_source_status.copy()
        for (source_name, source_url), source_conflicting_symbols in sorted(conflicting_symbols_by_source_ref.items()):
            conflict_error = _cross_source_classification_conflict_error(source_conflicting_symbols)
            affected = status["source"].eq(source_name) & status["url"].eq(source_url)
            status.loc[affected, "status"] = "parse_error"
            for row_index in status.index[affected]:
                previous_error = str(status.at[row_index, "error"] or "").strip()
                if conflict_error not in previous_error:
                    status.at[row_index, "error"] = safe_diagnostic_text(
                        f"{conflict_error}; {previous_error}" if previous_error else conflict_error,
                        max_chars=_UNIVERSE_DIAGNOSTIC_MAX_CHARS,
                    )
        workflow_source_status = status

    conflicting_symbol_set = set(conflicting_symbols)
    resolved_nasdaq = nasdaq_rows.loc[~nasdaq_rows["symbol"].astype(str).isin(conflicting_symbol_set)].copy()
    resolved_discovered = resolved_discovered.loc[
        ~resolved_discovered["symbol"].astype(str).isin(conflicting_symbol_set)
    ].copy()
    return (
        resolved_nasdaq.sort_values("symbol").reset_index(drop=True),
        resolved_discovered.sort_values("symbol").reset_index(drop=True),
        workflow_source_status,
    )


def _workflow_etn_source_to_universe(content: str, source: UniverseSource) -> pd.DataFrame:
    if source.parser == "microsectors_html":
        return _microsectors_html_to_universe(content, source, require_leveraged=False)
    if source.parser == "etracs_leverage_table":
        return _etracs_leverage_table_to_universe(content, source, require_leveraged=False)

    parsed_rows = _html_source_to_universe(
        content,
        source.name,
        source_label=f"{source.name} ETN issuer table",
        require_leveraged=False,
    )
    parsed_rows["fund_type"] = parsed_rows["fund_type"].str.replace(
        r"^ETF",
        "ETN",
        regex=True,
    )
    return parsed_rows


def load_issuer_etf_universe(timeout: int = 30) -> pd.DataFrame:
    timeout = _validated_universe_request_timeout(timeout)
    rows: list[tuple[int, UniverseSource, pd.DataFrame]] = []
    status_rows = []
    sources = [_workflow_issuer_source(raw_source) for raw_source in ISSUER_UNIVERSE_SOURCES]
    fetch_results = _fetch_enabled_sources(sources, timeout)
    for source, fetch_result in zip(sources, fetch_results, strict=True):
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
        if fetch_result is None:
            raise RuntimeError(f"Missing fetch result for enabled universe source {source.name!r}.")
        if fetch_result.error:
            status_rows.append(
                _workflow_source_status_row(
                    source=source.name,
                    source_type=source.source_type,
                    url=source.url,
                    status="source_error",
                    error=fetch_result.error,
                )
            )
            continue
        try:
            content = fetch_result.text or ""
            if not content.strip():
                raise ValueError("successful response body was empty")
            parsed_rows = _workflow_issuer_source_to_universe(
                content,
                source,
                require_leveraged=False,
            )
            issuer_rows, conflicting_symbols = _leveraged_product_rows(parsed_rows)
        except Exception as exc:
            status_rows.append(
                _workflow_source_status_row(
                    source=source.name,
                    source_type=source.source_type,
                    url=source.url,
                    status="parse_error",
                    error=_universe_exception_diagnostic(exc),
                )
            )
            continue
        status, error = _workflow_source_parse_status(parsed_rows, issuer_rows)
        if conflicting_symbols:
            status = "parse_error"
            error = _classification_conflict_error(conflicting_symbols)
        status_index = len(status_rows)
        if not parsed_rows.empty:
            rows.append((status_index, source, parsed_rows))
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

    out = _resolve_workflow_source_product_rows(rows, status_rows)
    out.attrs["workflow_source_status"] = status_rows
    return out


def load_etn_universe(timeout: int = 30) -> pd.DataFrame:
    timeout = _validated_universe_request_timeout(timeout)
    rows: list[tuple[int, UniverseSource, pd.DataFrame]] = []
    status_rows = []
    sources = list(WORKFLOW_ETN_SOURCES)
    fetch_results = _fetch_enabled_sources(sources, timeout)
    for source, fetch_result in zip(sources, fetch_results, strict=True):
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
        if fetch_result is None:
            raise RuntimeError(f"Missing fetch result for enabled universe source {source.name!r}.")
        if fetch_result.error:
            status_rows.append(
                _workflow_source_status_row(
                    source=source.name,
                    source_type=source.source_type,
                    url=source.url,
                    status="source_error",
                    error=fetch_result.error,
                )
            )
            continue
        try:
            parsed_rows = _workflow_etn_source_to_universe(fetch_result.text or "", source)
            source_rows, conflicting_symbols = _leveraged_product_rows(parsed_rows)
        except Exception as exc:
            status_rows.append(
                _workflow_source_status_row(
                    source=source.name,
                    source_type=source.source_type,
                    url=source.url,
                    status="parse_error",
                    error=_universe_exception_diagnostic(exc),
                )
            )
            continue
        status, error = _workflow_source_parse_status(parsed_rows, source_rows)
        if conflicting_symbols:
            status = "parse_error"
            error = _classification_conflict_error(conflicting_symbols)
        status_index = len(status_rows)
        if not parsed_rows.empty:
            rows.append((status_index, source, parsed_rows))
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

    out = _resolve_workflow_source_product_rows(rows, status_rows)
    out.attrs["workflow_source_status"] = status_rows
    return out


def _cboe_symbol_csv_to_universe(csv_text: str, source: UniverseSource) -> pd.DataFrame:
    try:
        csv_rows = list(csv.reader(io.StringIO(csv_text), strict=True))
    except csv.Error as exc:
        raise ValueError("Cboe listed-products response was not valid CSV.") from exc
    if not csv_rows:
        raise ValueError("Cboe listed-products CSV omitted its header row.")

    header = csv_rows[0]
    normalized_header = [_normalized_column_label(column) for column in header]
    if any(not column for column in normalized_header):
        raise ValueError("Cboe listed-products CSV contained an empty header field.")
    duplicate_header_fields = sorted({column for column in normalized_header if normalized_header.count(column) > 1})
    if duplicate_header_fields:
        raise ValueError(
            "Cboe listed-products CSV contained duplicate header fields: " + ", ".join(duplicate_header_fields)
        )

    symbol_columns = [index for index, column in enumerate(normalized_header) if column in {"symbol", "ticker", "name"}]
    if len(symbol_columns) != 1:
        raise ValueError(
            "Cboe listed-products CSV must contain exactly one Symbol, Ticker, or Name column; "
            f"found {len(symbol_columns)}."
        )
    symbol_column = symbol_columns[0]

    rows: list[dict[str, object]] = []
    seen_identities: set[str] = set()
    for row_number, row in enumerate(csv_rows[1:], start=1):
        # Permit conventional blank lines, but reject any material record whose
        # field count or required symbol cell is incomplete.
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(
                f"Cboe listed-products CSV row {row_number} had {len(row)} fields; expected {len(header)}."
            )
        raw_symbol = row[symbol_column]
        identity = _exchange_product_symbol_identity(raw_symbol)
        if identity is None:
            raise ValueError(
                f"Cboe listed-products CSV row {row_number} contained an invalid or noncanonical symbol value."
            )
        if identity in seen_identities:
            raise ValueError(f"Cboe listed-products CSV contained a duplicate normalized symbol: {identity}.")
        seen_identities.add(identity)
        rows.append({"symbol": raw_symbol, "name": raw_symbol})

    return _fund_rows_to_universe(
        rows,
        source.name,
        source_label=f"{source.name} audit source",
        require_leveraged=False,
    )


def _deduplicate_sec_named_rows(
    rows: Iterable[dict[str, object]],
    *,
    source_description: str,
) -> list[dict[str, object]]:
    """Collapse exact SEC records and reject one identity with two names."""
    validated_rows: list[dict[str, object]] = []
    name_by_symbol: dict[str, str] = {}
    for row_number, row in enumerate(rows):
        symbol = row.get("symbol")
        raw_name = row.get("name")
        if type(symbol) is not str or type(raw_name) is not str:
            raise ValueError(f"{source_description} row {row_number} omitted its ticker or name.")
        name = _html_text(raw_name)
        if not name:
            raise ValueError(f"{source_description} row {row_number} omitted its required name value.")
        prior_name = name_by_symbol.get(symbol)
        if prior_name is not None:
            if prior_name != name:
                raise ValueError(f"{source_description} contained conflicting names for normalized ticker {symbol}.")
            continue
        name_by_symbol[symbol] = name
        validated_rows.append({"symbol": symbol, "name": name})
    return validated_rows


def _sec_company_tickers_to_universe(json_text: str, source: UniverseSource) -> pd.DataFrame:
    payload = _strict_json_loads(
        json_text,
        source_description="SEC company ticker registry response",
        invalid_json_message="SEC company ticker registry response was not valid JSON.",
    )
    if not isinstance(payload, dict):
        raise ValueError("SEC company ticker registry response must be a JSON object.")

    rows: list[dict[str, object]] = []
    for row_number, record in enumerate(payload.values()):
        if not isinstance(record, dict):
            raise ValueError(f"SEC company ticker registry row {row_number} was not a JSON object.")
        ticker = record.get("ticker")
        title = record.get("title")
        if type(ticker) is not str or not ticker.strip():
            raise ValueError(f"SEC company ticker registry row {row_number} omitted its required 'ticker' value.")
        if type(title) is not str or not title.strip():
            raise ValueError(f"SEC company ticker registry row {row_number} omitted its required 'title' value.")
        if ticker in SEC_UNAVAILABLE_ENTITY_TICKERS:
            continue
        normalized_ticker = _sec_entity_ticker_identity(ticker)
        if normalized_ticker is None or ticker != ticker.strip().upper():
            raise ValueError(
                f"SEC company ticker registry row {row_number} contained an invalid or noncanonical 'ticker' value."
            )
        rows.append({"symbol": normalized_ticker, "name": title})

    validated_rows = _deduplicate_sec_named_rows(
        rows,
        source_description="SEC company ticker registry",
    )
    return _fund_rows_to_universe(
        validated_rows,
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
    payload = _strict_json_loads(
        json_text,
        source_description="SEC ticker registry response",
        invalid_json_message="SEC ticker registry response was not valid JSON.",
    )

    if not isinstance(payload, dict):
        raise ValueError("SEC ticker registry response must be a JSON object.")
    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise ValueError("SEC ticker registry response requires list-valued 'fields' and 'data'.")

    def required_field_index(field: str) -> int:
        indexes = [index for index, value in enumerate(fields) if value == field]
        if len(indexes) != 1:
            raise ValueError(f"SEC ticker registry schema requires exactly one {field!r} field; found {len(indexes)}.")
        return indexes[0]

    symbol_idx = required_field_index(symbol_field)
    name_idx = required_field_index(name_field) if name_field is not None else None
    rows: list[dict[str, object]] = []
    for row_number, raw_row in enumerate(data):
        if not isinstance(raw_row, list) or len(raw_row) != len(fields):
            raise ValueError(f"SEC ticker registry row {row_number} did not match its declared field schema.")
        symbol = raw_row[symbol_idx]
        name = raw_row[name_idx] if name_idx is not None else symbol
        if name_idx is not None:
            if type(symbol) is not str or not symbol.strip():
                raise ValueError(f"SEC ticker registry row {row_number} omitted its required {symbol_field!r} value.")
            if type(name) is not str or not name.strip():
                raise ValueError(f"SEC ticker registry row {row_number} omitted its required {name_field!r} value.")
            if symbol in SEC_UNAVAILABLE_ENTITY_TICKERS:
                continue
            normalized_symbol = _sec_entity_ticker_identity(symbol)
            if normalized_symbol is None or symbol != symbol.strip().upper():
                raise ValueError(
                    f"SEC ticker registry row {row_number} contained an invalid or noncanonical {symbol_field!r} value."
                )
            symbol = normalized_symbol
        elif symbol is not None:
            if type(symbol) is not str:
                raise ValueError(f"SEC ticker registry row {row_number} contained a non-string {symbol_field!r} value.")
            normalized_symbol = _sec_mutual_fund_ticker_identity(symbol)
            if normalized_symbol is None:
                raise ValueError(
                    f"SEC ticker registry row {row_number} contained an invalid or noncanonical {symbol_field!r} value."
                )
            symbol = normalized_symbol or None
            name = normalized_symbol or None
        rows.append({"symbol": symbol, "name": name})
    return rows


def _sec_exchange_tickers_to_universe(json_text: str, source: UniverseSource) -> pd.DataFrame:
    rows = _sec_fields_data_json_to_rows(
        json_text,
        symbol_field="ticker",
        name_field="name",
    )
    validated_rows = _deduplicate_sec_named_rows(
        rows,
        source_description="SEC exchange ticker registry",
    )
    return _fund_rows_to_universe(
        validated_rows,
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
        allow_missing_symbols=True,
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
        "audit_capability": (
            "inventory_only"
            if source.parser in AUDIT_INVENTORY_ONLY_PARSERS
            else "registered_only"
            if source.parser == "registered_only"
            else "product_names"
        ),
        "enabled": source.enabled,
        "status": status,
        "row_count": row_count,
        "error": safe_diagnostic_text(error, max_chars=_UNIVERSE_DIAGNOSTIC_MAX_CHARS),
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
    if source.parser in AUDIT_INVENTORY_ONLY_PARSERS or (
        source.parser in SEC_ENTITY_AUDIT_PARSERS and not _sec_audit_row_has_product_context(name)
    ):
        return pd.Series(
            {
                "is_leveraged_candidate": False,
                "is_long_leveraged_candidate": False,
                "is_short_leveraged_candidate": False,
                "leverage": None,
                "direction": None,
            }
        )

    is_candidate, leverage, direction = _classify_leveraged_name(name)
    return pd.Series(
        {
            "is_leveraged_candidate": is_candidate,
            "is_long_leveraged_candidate": is_candidate
            and leverage is not None
            and leverage > 1.0
            and direction == "long",
            "is_short_leveraged_candidate": is_candidate
            and leverage is not None
            and leverage > 1.0
            and direction == "inverse",
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


def _deduplicate_audit_source_rows(rows: pd.DataFrame, source: UniverseSource) -> pd.DataFrame:
    """Collapse exact audit identities and reject conflicting ticker metadata."""
    if rows.empty:
        return rows

    metadata_columns = [column for column in ("name", "fund_type") if column in rows.columns]
    conflicting_symbols = []
    for raw_symbol, symbol_rows in rows.groupby("symbol", sort=False, dropna=False):
        if len(symbol_rows.loc[:, metadata_columns].drop_duplicates()) > 1:
            conflicting_symbols.append(str(raw_symbol))
    if conflicting_symbols:
        symbols = ", ".join(sorted(conflicting_symbols))
        raise ValueError(
            f"{source.name} audit source contained conflicting metadata for duplicate ticker(s): {symbols}."
        )
    return rows.drop_duplicates("symbol", keep="first").reset_index(drop=True)


def load_audit_universe_sources(timeout: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    timeout = _validated_universe_request_timeout(timeout)
    rows = []
    status_rows = []
    sources = list(AUDIT_UNIVERSE_SOURCES)
    fetch_results = _fetch_enabled_sources(sources, timeout)
    for source, fetch_result in zip(sources, fetch_results, strict=True):
        if not source.enabled or source.parser == "registered_only":
            status_rows.append(_audit_source_status_row(source, status="registered_only"))
            continue

        if _is_sec_url(source.url) and _configured_sec_user_agent() is None:
            status_rows.append(
                _audit_source_status_row(
                    source,
                    status="skipped_configuration",
                    error=SEC_USER_AGENT_CONFIGURATION_ERROR,
                )
            )
            continue

        if fetch_result is None:
            raise RuntimeError(f"Missing fetch result for enabled universe source {source.name!r}.")
        if fetch_result.error:
            status_rows.append(
                _audit_source_status_row(
                    source,
                    status="error",
                    error=fetch_result.error,
                )
            )
            continue

        try:
            content = fetch_result.text or ""
            if source.parser == "cboe_symbol_csv":
                source_rows = _cboe_symbol_csv_to_universe(content, source)
            elif source.parser == "sec_company_tickers":
                source_rows = _sec_company_tickers_to_universe(content, source)
            elif source.parser == "sec_exchange_tickers":
                source_rows = _sec_exchange_tickers_to_universe(content, source)
            elif source.parser == "sec_mutual_fund_tickers":
                source_rows = _sec_mutual_fund_tickers_to_universe(content, source)
            else:
                source_rows = _html_source_to_universe(
                    content,
                    source.name,
                    source_label=f"{source.name} audit source",
                    require_leveraged=False,
                )
            source_rows = _deduplicate_audit_source_rows(source_rows, source)
        except Exception as exc:
            status_rows.append(
                _audit_source_status_row(
                    source,
                    status="error",
                    error=_universe_exception_diagnostic(exc),
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
        loaded_status = "loaded_inventory_only" if source.parser in AUDIT_INVENTORY_ONLY_PARSERS else "loaded"
        status_rows.append(
            _audit_source_status_row(
                source,
                status=loaded_status,
                row_count=len(source_rows),
            )
        )

    if rows:
        audit_rows = (
            pd.concat(rows, ignore_index=True)
            .sort_values(
                ["source", "symbol", "is_leveraged_candidate"],
                ascending=[True, True, False],
                kind="stable",
            )
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
        out["is_short_leveraged_candidate"] = out["direction"].eq("inverse") & out["is_leveraged_candidate"].astype(
            bool
        )
    out["in_merged_universe"] = out["symbol"].isin(merged_symbols)
    out["in_workflow_universe"] = out["symbol"].isin(workflow_symbols)
    leveraged_candidate = out["is_long_leveraged_candidate"].astype(bool) | out["is_short_leveraged_candidate"].astype(
        bool
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
    # Production callers resolve and record cross-source conflicts before this
    # final merge. Keep this boundary fail-closed for injected/library frames as
    # well, so an unresolved discovered duplicate can never become first-row-wins.
    issuer = issuer_df.copy()
    if not issuer.empty and {"symbol", "name"}.issubset(issuer.columns):
        identity_frames = [frame for frame in (nasdaq.copy(), issuer.copy()) if not frame.empty]
        if identity_frames:
            _resolved_rows, conflicting_symbols = _leveraged_product_rows(
                pd.concat(identity_frames, ignore_index=True, sort=False)
            )
            if conflicting_symbols:
                conflicting_symbol_set = set(conflicting_symbols)
                nasdaq = nasdaq.loc[~nasdaq["symbol"].astype(str).isin(conflicting_symbol_set)].copy()
                issuer = issuer.loc[~issuer["symbol"].astype(str).isin(conflicting_symbol_set)].copy()
        issuer, _issuer_conflicting_symbols = _leveraged_product_rows(issuer)
    supplemental_references: dict[str, str] = {}
    if "reference_security" in issuer.columns:
        for raw_symbol, symbol_rows in issuer.groupby("symbol", sort=False):
            references = {str(value) for value in symbol_rows["reference_security"] if isinstance(value, str) and value}
            if len(references) > 1:
                raise ValueError(f"Issuer rows contained conflicting Reference Security values for {raw_symbol}.")
            if references:
                reference = next(iter(references))
                if _exchange_product_symbol_identity(reference) != reference:
                    raise ValueError(f"Issuer rows contained an invalid Reference Security value for {raw_symbol}.")
                supplemental_references[str(raw_symbol)] = reference
    combined = pd.concat([nasdaq, issuer], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    # Nasdaq is the current ETF authority for duplicate symbols.  Issuer rows
    # still contribute issuer-only ETFs/ETNs, but cannot overwrite current
    # official metadata for an ETF that Nasdaq lists.
    combined["source_rank"] = combined["source"].ne("Nasdaq ETF definitions").astype(int)
    combined = combined.sort_values(["symbol", "source_rank"], ascending=[True, True])
    combined = combined.drop_duplicates("symbol", keep="first")
    if supplemental_references:
        combined["reference_security"] = (
            combined["symbol"].map(supplemental_references).combine_first(combined.get("reference_security"))
        )
    return combined.drop(columns=["source_rank"]).reset_index(drop=True)


def _product_row_rsi_mapping(
    row: pd.Series,
    *,
    known_symbols: set[str] | None,
) -> RsiSymbolMapping:
    mapping = infer_rsi_mapping(
        str(row["symbol"]),
        str(row["name"]),
        known_symbols=known_symbols,
        fund_type=row.get("fund_type"),
    )
    raw_reference = row.get("reference_security")
    reference = raw_reference if isinstance(raw_reference, str) and raw_reference else None
    if reference is None:
        return mapping
    reference_identity = _exchange_product_symbol_identity(reference)
    asset_identity = _exchange_product_symbol_identity(row.get("symbol"))
    if reference_identity is not None and reference_identity == asset_identity:
        return RsiSymbolMapping(
            rsi_symbol=str(row["symbol"]),
            underlying_name=str(row["symbol"]),
            mapping_source="issuer_reference_conflict",
            confidence="needs_review",
            mapping_reason="issuer Reference Security pointed back to the leveraged product itself",
        )
    if reference_identity is None or (
        mapping.mapping_source
        not in {
            "ambiguous_name_proxy",
            "asset_symbol",
            "symbol_override_identity_mismatch",
            "unresolved_basket",
            "unresolved_single_stock",
        }
        and mapping.rsi_symbol != reference_identity
    ):
        return RsiSymbolMapping(
            rsi_symbol=str(row["symbol"]),
            underlying_name=str(row["symbol"]),
            mapping_source="issuer_reference_conflict",
            confidence="needs_review",
            mapping_reason="issuer Reference Security contradicted other RSI-mapping metadata",
        )
    return RsiSymbolMapping(
        rsi_symbol=reference_identity,
        underlying_name=reference_identity,
        mapping_source="issuer_reference",
        confidence="curated",
        mapping_reason="matched issuer-supplied Reference Security",
    )


def _workflow_row_metadata(row: pd.Series, known_symbols: set[str] | None) -> pd.Series:
    leverage, direction = infer_leverage_and_direction(row["name"])
    mapping = _product_row_rsi_mapping(row, known_symbols=known_symbols)

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
    single_stock = eligible[eligible["fund_type"].str.contains(r"ETF \(Single Stock\)", regex=True, na=False)].copy()
    single_stock_long = single_stock.loc[single_stock["name"].map(is_long_leveraged_name).astype(bool)].copy()
    all_long_leveraged = eligible.loc[eligible["name"].map(is_long_leveraged_name).astype(bool)].copy()

    return (
        single_stock_long.sort_values("symbol").reset_index(drop=True),
        all_long_leveraged.sort_values("symbol").reset_index(drop=True),
    )


def select_short_workflow_universe(etf_df: pd.DataFrame) -> pd.DataFrame:
    eligible = etf_df[~etf_df["symbol"].isin(EXCLUDED_UNIVERSE_SYMBOLS)].copy()
    all_short_leveraged = eligible.loc[eligible["name"].map(is_short_leveraged_name).astype(bool)].copy()
    return all_short_leveraged.sort_values("symbol").reset_index(drop=True)


def determine_workflow_asset_groups(cfg: UniverseConfig) -> dict[str, pd.DataFrame]:
    request_timeout_seconds = _validated_universe_request_timeout(cfg.request_timeout_seconds)
    if cfg.top_n is not None and (isinstance(cfg.top_n, bool) or not isinstance(cfg.top_n, int) or cfg.top_n <= 0):
        raise ValueError(f"Universe top_n must be a positive integer or None; got {cfg.top_n!r}.")

    try:
        nasdaq_df = load_current_etf_universe(timeout=request_timeout_seconds)
    except Exception as exc:
        status = "source_error" if isinstance(exc, requests.RequestException) else "parse_error"
        nasdaq_df = pd.DataFrame(columns=["symbol", "name", "fund_type"])
        nasdaq_df.attrs["workflow_source_status"] = [
            _workflow_source_status_row(
                source=NASDAQ_ETF_SOURCE_NAME,
                source_type=NASDAQ_ETF_SOURCE_TYPE,
                url=ETF_DEFS_URL,
                status=status,
                error=_universe_exception_diagnostic(exc),
            )
        ]
    nasdaq_source_status = _workflow_source_status(nasdaq_df)
    nasdaq_loaded_row_count = len(nasdaq_df)
    # Tests and library callers may supply a frame without loader metadata.
    # A populated explicit frame is healthy; an empty one still fails closed.
    if nasdaq_source_status.empty:
        status = "loaded" if not nasdaq_df.empty else "parse_error"
        error = "" if not nasdaq_df.empty else "Primary Nasdaq feed returned no usable ETF rows."
        nasdaq_source_status = pd.DataFrame(
            [
                _workflow_source_status_row(
                    source=NASDAQ_ETF_SOURCE_NAME,
                    source_type=NASDAQ_ETF_SOURCE_TYPE,
                    url=ETF_DEFS_URL,
                    status=status,
                    parsed_row_count=len(nasdaq_df),
                    row_count=len(nasdaq_df),
                    error=error,
                )
            ],
            columns=WORKFLOW_SOURCE_STATUS_COLUMNS,
        )
    issuer_df = load_issuer_etf_universe(timeout=request_timeout_seconds)
    etn_df = load_etn_universe(timeout=request_timeout_seconds)
    workflow_source_status = pd.concat(
        [nasdaq_source_status, _workflow_source_status(issuer_df), _workflow_source_status(etn_df)],
        ignore_index=True,
    )
    nasdaq_df, discovered_df, workflow_source_status = _resolve_discovered_product_rows(
        nasdaq_df,
        issuer_df,
        etn_df,
        workflow_source_status,
    )
    workflow_source_failures = workflow_source_status[
        ~workflow_source_status["status"].isin(NON_FAILURE_WORKFLOW_SOURCE_STATUSES)
    ].copy()
    active_symbols = load_active_listed_symbols(timeout=request_timeout_seconds)
    active_symbol_set = {str(symbol).upper() for symbol in active_symbols} if active_symbols else set()
    active_listing_complete = bool(getattr(active_symbols, "is_complete", True))
    active_listing_status = pd.DataFrame(
        getattr(active_symbols, "source_status", []),
        columns=["source", "url", "symbol_column", "status", "symbol_count", "error"],
    )
    if "error" in active_listing_status.columns:
        active_listing_status["error"] = active_listing_status["error"].map(
            lambda value: (
                safe_diagnostic_text(value, max_chars=_UNIVERSE_DIAGNOSTIC_MAX_CHARS) if isinstance(value, str) else ""
            )
        )
    active_listing_failures = active_listing_status[active_listing_status["status"].ne("loaded")].copy()
    primary_symbols = set(nasdaq_df["symbol"].dropna().astype(str).str.upper())
    if (
        active_symbol_set
        and active_listing_complete
        and len(primary_symbols) >= _NASDAQ_ACTIVE_PRIMARY_COVERAGE_MIN_ROWS
    ):
        covered_primary_symbols = primary_symbols.intersection(active_symbol_set)
        primary_coverage = len(covered_primary_symbols) / len(primary_symbols) if primary_symbols else 1.0
        missing_count = len(primary_symbols.difference(active_symbol_set))
        if primary_coverage < _NASDAQ_ACTIVE_PRIMARY_COVERAGE_MIN or missing_count > _NASDAQ_ACTIVE_PRIMARY_MAX_MISSING:
            active_listing_complete = False
            coverage_error = (
                "Active Nasdaq symbol snapshot did not cover enough of the current primary ETF inventory; "
                f"covered {primary_coverage:.2%}, missing {missing_count} symbols "
                f"(requires at least {_NASDAQ_ACTIVE_PRIMARY_COVERAGE_MIN:.2%} coverage "
                f"and at most {_NASDAQ_ACTIVE_PRIMARY_MAX_MISSING} missing symbols)."
            )
            active_listing_status = pd.concat(
                [
                    active_listing_status,
                    pd.DataFrame(
                        [
                            {
                                "source": "primary_nasdaq_coverage",
                                "url": ETF_DEFS_URL,
                                "symbol_column": "symbol",
                                "status": "error",
                                "symbol_count": len(covered_primary_symbols),
                                "error": coverage_error,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            active_listing_failures = active_listing_status[active_listing_status["status"].ne("loaded")].copy()
    inactive_primary = pd.DataFrame(columns=[*nasdaq_df.columns, "inactive_source", "inactive_reason"])
    inactive_discovered = pd.DataFrame(columns=[*discovered_df.columns, "inactive_source", "inactive_reason"])
    if active_symbol_set and active_listing_complete:
        primary_is_active = nasdaq_df["symbol"].astype(str).str.upper().isin(active_symbol_set)
        inactive_primary = nasdaq_df.loc[~primary_is_active].copy()
        inactive_primary["inactive_source"] = "primary_nasdaq"
        inactive_primary["inactive_reason"] = "not present in active Nasdaq symbol files"
        nasdaq_df = nasdaq_df.loc[primary_is_active].copy()
        # The active directories can corroborate discovered issuer products,
        # but their absence cannot prove an issuer-only symbol inactive. A
        # complete primary-ETF coverage check says nothing about rows that are
        # absent from that primary table, and filtering them would make any
        # partial directory snapshot destructively authoritative.
    inactive_products = pd.concat(
        [inactive_primary, inactive_discovered],
        ignore_index=True,
        sort=False,
    )
    etf_df = _merge_universe_sources(nasdaq_df, discovered_df)
    known_symbols = _known_rsi_symbols(etf_df, active_symbol_set)

    nasdaq_universe = build_nasdaq_universe_table(etf_df, known_symbols=known_symbols)
    save_table_to_sqlite(nasdaq_universe, cfg.sqlite_db_path, "nasdaq_etf_universe")
    save_table_to_sqlite(
        inactive_products,
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

    audit_rows, audit_status = load_audit_universe_sources(timeout=request_timeout_seconds)
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
        side: candidates.loc[~candidates["confidence"].eq("needs_review")].copy()
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
        "Current ETFs in Nasdaq table": nasdaq_loaded_row_count,
        "Current issuer-discovered leveraged ETFs found": len(issuer_df),
        "Current issuer-discovered leveraged ETNs found": len(etn_df),
        "Inactive primary Nasdaq ETFs skipped": len(inactive_primary),
        "Inactive issuer-discovered ETFs/ETNs skipped": len(inactive_discovered),
        "Active listing sources loaded": int((active_listing_status["status"] == "loaded").sum())
        if not active_listing_status.empty
        else 0,
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
        "Audit product-name rows parsed": int(audit_status.loc[audit_status["status"].eq("loaded"), "row_count"].sum())
        if {"status", "row_count"}.issubset(audit_status.columns)
        else 0,
        "Audit inventory-only rows parsed": int(
            audit_status.loc[
                audit_status["status"].eq("loaded_inventory_only"),
                "row_count",
            ].sum()
        )
        if {"status", "row_count"}.issubset(audit_status.columns)
        else 0,
        "Audit leveraged candidates missing from merged universe": len(audit_report),
    }
    common_attrs = {
        "universe_degraded": (
            not workflow_source_failures.empty or not active_listing_failures.empty or not audit_source_failures.empty
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
    selected_assets = workflow_assets if cfg.top_n is None else workflow_assets.head(cfg.top_n).copy()

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
    if out.empty:
        # DataFrame.apply(axis=1) returns an empty frame carrying the input
        # columns, rather than an empty Series, which would duplicate `symbol`
        # during the concat below. Preserve the normal output schema explicitly
        # so source-health tables can be saved before the workflow fails closed.
        for column in (
            "rsi_symbol",
            "underlying_symbol",
            "underlying_name",
            "mapping_source",
            "confidence",
            "mapping_reason",
        ):
            out[column] = pd.Series(index=out.index, dtype="object")
        return out.sort_values("symbol").reset_index(drop=True)
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
        mapping = _product_row_rsi_mapping(row, known_symbols=known_symbols)
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
