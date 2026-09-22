from __future__ import annotations

import io
import ipaddress
import math
import os
import pickle
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from contextvars import ContextVar
from urllib.parse import urlsplit

import requests
from urllib3.connection import HTTPConnection as _Urllib3HTTPConnection
from urllib3.connection import HTTPSConnection as _Urllib3HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool as _Urllib3HTTPConnectionPool
from urllib3.connectionpool import HTTPSConnectionPool as _Urllib3HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
from urllib3.util import connection as _urllib3_connection

_PROTOCOL_NAME = "leveraged-trader-http-deadline-v1"
HTTP_WORKER_REQUEST_MAX_BYTES = 1024 * 1024
_HTTP_WORKER_CHUNK_BYTES = 64 * 1024
_HTTP_WORKER_ERROR_MAX_CHARS = 4096
_HTTP_WORKER_REAP_TIMEOUT_SECONDS = 1.0
_DNS64_WELL_KNOWN_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")
_ALPACA_PAPER_API_AUTHORITY = "paper-api.alpaca.markets"
_ALPACA_BROKER_IDENTIFIER_PATH_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,127}")
_HTTP_WORKER_COMMAND = (
    sys.executable,
    "-I",
    "-m",
    "leveraged_trader._http_deadline_worker",
)


class _ExplicitHeaderAuth(requests.auth.AuthBase):
    """Suppress implicit netrc auth while retaining an explicit auth header."""

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        return request


_EXPLICIT_HEADER_AUTH = _ExplicitHeaderAuth()


class _RequestWriteDeadlineElapsed(RuntimeError):
    """A worker request was withheld before its first destination-bound byte."""


class AlpacaRequestWriteDeadlineElapsed(_RequestWriteDeadlineElapsed):
    """The Alpaca request was withheld before its first broker-bound byte."""


class HttpRequestDeadlineElapsed(_RequestWriteDeadlineElapsed):
    """The HTTP request was withheld because its overall deadline elapsed."""


def _validate_alpaca_request_target(
    method: str,
    url: str,
    *,
    allow_loopback_test_origin: bool = False,
) -> None:
    """Require one canonical paper-API route before credentialed transport."""
    if type(method) is not str or type(url) is not str or type(allow_loopback_test_origin) is not bool:
        raise requests.exceptions.InvalidURL("Alpaca request URL must be a canonical plain string.")
    # ``urlsplit`` deliberately strips leading C0 controls and spaces, and
    # removes ASCII tab/newline characters throughout a URL.  Requests instead
    # percent-encodes some of those characters when preparing the request, so
    # validating only the parsed path can authorize a different path from the
    # one placed on the wire.  Reject every raw whitespace/control character
    # before parsing to keep validation and transport byte-for-byte aligned.
    if any(character.isspace() or not character.isprintable() for character in url):
        raise requests.exceptions.InvalidURL("Alpaca request URL must not contain whitespace or control characters.")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError):
        raise requests.exceptions.InvalidURL("Alpaca request URL has an invalid or ambiguous authority.") from None

    if allow_loopback_test_origin:
        loopback_authority = f"127.0.0.1:{port}" if port is not None else ""
        if (
            parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1"
            and parsed.netloc == loopback_authority
            and parsed.username is None
            and parsed.password is None
            and port is not None
            and not parsed.query
            and not parsed.fragment
            and parsed.path.startswith("/")
        ):
            return

    canonical_origin = (
        parsed.scheme == "https"
        and parsed.netloc == _ALPACA_PAPER_API_AUTHORITY
        and parsed.hostname == _ALPACA_PAPER_API_AUTHORITY
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.query
        and not parsed.fragment
    )
    identifier = _ALPACA_BROKER_IDENTIFIER_PATH_PATTERN.pattern
    allowed_path = {
        "DELETE": re.fullmatch(rf"/v2/orders/{identifier}", parsed.path) is not None,
        "GET": (
            parsed.path
            in {
                "/v2/account",
                "/v2/calendar",
                "/v2/clock",
                "/v2/corporate_actions/announcements",
                "/v2/orders",
                "/v2/orders:by_client_order_id",
                "/v2/positions",
            }
            or re.fullmatch(rf"/v2/(?:assets|orders|positions)/{identifier}", parsed.path) is not None
        ),
        "POST": parsed.path == "/v2/orders",
    }.get(method, False)
    if not canonical_origin or not allowed_path:
        raise requests.exceptions.InvalidURL(
            "Alpaca request URL must use the canonical paper-trading origin and a supported API path."
        )


_ALPACA_REQUEST_WRITE_DEADLINE_MONOTONIC: ContextVar[float | None] = ContextVar(
    "http_worker_alpaca_request_write_deadline_monotonic",
    default=None,
)
_ALPACA_REQUEST_WRITE_DEADLINE_IS_OVERALL: ContextVar[bool] = ContextVar(
    "http_worker_alpaca_request_write_deadline_is_overall",
    default=False,
)


def _alpaca_request_write_deadline_error(*, stage: str) -> _RequestWriteDeadlineElapsed:
    if _ALPACA_REQUEST_WRITE_DEADLINE_IS_OVERALL.get():
        return HttpRequestDeadlineElapsed(f"Alpaca response exceeded its overall deadline {stage}")
    return AlpacaRequestWriteDeadlineElapsed(f"market submission window elapsed {stage}")


def _remaining_alpaca_request_write_seconds() -> float | None:
    deadline = _ALPACA_REQUEST_WRITE_DEADLINE_MONOTONIC.get()
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise _alpaca_request_write_deadline_error(stage="before the broker request could be written")
    return remaining


def _deadline_bounded_socket_timeout(configured_timeout: object, remaining: float) -> float:
    if configured_timeout is None:
        return remaining
    try:
        timeout = float(configured_timeout)
    except (TypeError, ValueError):
        return remaining
    if not math.isfinite(timeout) or timeout <= 0:
        return remaining
    return min(timeout, remaining)


def _resolve_socket_addresses_before_alpaca_deadline(
    host: str,
    port: int,
    family: socket.AddressFamily,
) -> list[tuple[object, ...]]:
    """Resolve without allowing a blocked resolver to authorize a late POST."""
    remaining = _remaining_alpaca_request_write_seconds()
    if remaining is None:
        return socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)

    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            result_queue.put(
                (True, socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)),
                block=False,
            )
        except BaseException as exc:
            with suppress(queue.Full):
                result_queue.put((False, exc), block=False)

    threading.Thread(
        target=resolve,
        name="alpaca-worker-order-dns-deadline",
        daemon=True,
    ).start()
    try:
        succeeded, result = result_queue.get(timeout=remaining)
    except queue.Empty as exc:
        raise _alpaca_request_write_deadline_error(stage="during Alpaca request DNS resolution") from exc
    _remaining_alpaca_request_write_seconds()
    if not succeeded:
        assert isinstance(result, BaseException)
        raise result
    assert isinstance(result, list)
    return result


def _connect_alpaca_socket_before_write_deadline(
    address: tuple[str, int],
    configured_timeout: object,
    *,
    source_address: tuple[str, int] | None,
    socket_options: Sequence[tuple[int, int, int]] | None,
) -> socket.socket:
    """Apply one absolute write deadline across DNS and candidate addresses."""
    host, port = address
    if host.startswith("["):
        host = host.strip("[]")
    host.encode("idna")
    family = _urllib3_connection.allowed_gai_family()
    addresses = _resolve_socket_addresses_before_alpaca_deadline(host, port, family)
    error: OSError | None = None
    for address_info in addresses:
        remaining = _remaining_alpaca_request_write_seconds()
        if remaining is None:
            raise RuntimeError("Alpaca deadline-aware connection requires an active write deadline")
        af, socktype, proto, _canonname, socket_address = address_info
        candidate: socket.socket | None = None
        try:
            candidate = socket.socket(af, socktype, proto)
            for option in socket_options or ():
                candidate.setsockopt(*option)
            candidate.settimeout(_deadline_bounded_socket_timeout(configured_timeout, remaining))
            if source_address:
                candidate.bind(source_address)
            candidate.connect(socket_address)
            remaining = _remaining_alpaca_request_write_seconds()
            assert remaining is not None
            candidate.settimeout(_deadline_bounded_socket_timeout(configured_timeout, remaining))
            return candidate
        except _RequestWriteDeadlineElapsed:
            if candidate is not None:
                candidate.close()
            raise
        except OSError as exc:
            error = exc
            if candidate is not None:
                candidate.close()
    if error is not None:
        raise error
    raise OSError("getaddrinfo returned an empty address list")


class _DeadlineAwareAlpacaConnectionMixin:
    """Prevent a worker's Alpaca request bytes from starting after cutoff."""

    _alpaca_request_write_started = False
    _alpaca_connection_setup_in_progress = False

    def _new_conn(self) -> socket.socket:
        if _ALPACA_REQUEST_WRITE_DEADLINE_MONOTONIC.get() is None:
            return super()._new_conn()  # type: ignore[misc]
        try:
            return _connect_alpaca_socket_before_write_deadline(
                (self._dns_host, self.port),  # type: ignore[attr-defined]
                self.timeout,  # type: ignore[attr-defined]
                source_address=self.source_address,  # type: ignore[attr-defined]
                socket_options=self.socket_options,  # type: ignore[attr-defined]
            )
        except _RequestWriteDeadlineElapsed:
            raise
        except socket.gaierror as exc:
            raise NewConnectionError(self, f"Failed to resolve {self.host}: {exc}") from exc  # type: ignore[attr-defined]
        except TimeoutError as exc:
            raise ConnectTimeoutError(
                self,
                f"Connection to {self.host} timed out. (connect timeout={self.timeout})",  # type: ignore[attr-defined]
            ) from exc
        except OSError as exc:
            raise NewConnectionError(self, f"Failed to establish a new connection: {exc}") from exc

    def connect(self) -> None:
        self._alpaca_connection_setup_in_progress = True
        try:
            super().connect()  # type: ignore[misc]
            _remaining_alpaca_request_write_seconds()
        except _RequestWriteDeadlineElapsed:
            if self.sock is not None:  # type: ignore[attr-defined]
                self.sock.close()  # type: ignore[attr-defined]
                self.sock = None  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            deadline = _ALPACA_REQUEST_WRITE_DEADLINE_MONOTONIC.get()
            if deadline is not None and time.monotonic() >= deadline:
                if self.sock is not None:  # type: ignore[attr-defined]
                    self.sock.close()  # type: ignore[attr-defined]
                    self.sock = None  # type: ignore[attr-defined]
                raise _alpaca_request_write_deadline_error(
                    stage="during Alpaca request connection or TLS setup"
                ) from exc
            raise
        finally:
            self._alpaca_connection_setup_in_progress = False

    def request(self, *args: object, **kwargs: object) -> None:
        self._alpaca_request_write_started = False
        super().request(*args, **kwargs)  # type: ignore[misc]

    def send(self, data: object) -> None:
        if _ALPACA_REQUEST_WRITE_DEADLINE_MONOTONIC.get() is None:
            super().send(data)  # type: ignore[misc]
            return
        try:
            remaining = _remaining_alpaca_request_write_seconds()
        except _RequestWriteDeadlineElapsed:
            if self._alpaca_request_write_started:
                raise TimeoutError("Alpaca request-write deadline elapsed after request transmission began") from None
            raise
        assert remaining is not None
        if self.sock is None:  # type: ignore[attr-defined]
            self.connect()
            remaining = _remaining_alpaca_request_write_seconds()
            assert remaining is not None
        self.sock.settimeout(  # type: ignore[attr-defined]
            _deadline_bounded_socket_timeout(self.timeout, remaining)  # type: ignore[attr-defined]
        )
        try:
            # Setting the socket timeout and rescheduling this process can consume
            # the last part of the absolute budget. Recheck at the byte boundary
            # rather than treating the earlier timeout calculation as authority
            # to begin a late broker mutation.
            _remaining_alpaca_request_write_seconds()
        except _RequestWriteDeadlineElapsed:
            request_write_started = self._alpaca_request_write_started
            sock = self.sock  # type: ignore[attr-defined]
            self.sock = None  # type: ignore[attr-defined]
            with suppress(OSError):
                sock.close()
            if request_write_started:
                raise TimeoutError("Alpaca request-write deadline elapsed after request transmission began") from None
            raise
        if not self._alpaca_connection_setup_in_progress:
            self._alpaca_request_write_started = True
        super().send(data)  # type: ignore[misc]


class _DeadlineAwareAlpacaHTTPConnection(_DeadlineAwareAlpacaConnectionMixin, _Urllib3HTTPConnection):
    pass


class _DeadlineAwareAlpacaHTTPSConnection(_DeadlineAwareAlpacaConnectionMixin, _Urllib3HTTPSConnection):
    pass


class _BoundedPickleBuffer(io.BytesIO):
    """Stop a pickle before it can build an over-limit protocol buffer."""

    def __init__(self, max_bytes: int) -> None:
        super().__init__()
        self._max_bytes = max_bytes

    def write(self, value: object) -> int:
        try:
            byte_count = memoryview(value).nbytes
        except TypeError:
            raise TypeError("HTTP request worker protocol received a non-buffer value.") from None
        if byte_count > self._max_bytes - self.tell():
            raise ValueError(f"HTTP request worker protocol exceeded its {self._max_bytes}-byte limit.")
        return super().write(value)  # type: ignore[arg-type]


def _is_public_network_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Accept public addresses, including safe RFC 6052 well-known translations."""
    if isinstance(address, ipaddress.IPv6Address) and address in _DNS64_WELL_KNOWN_PREFIX:
        # Python correctly classifies the well-known translation prefix as
        # reserved rather than globally routed IPv6.  On a DNS64/NAT64 network,
        # however, it represents the final 32-bit IPv4 destination. Permit only
        # translations of IPv4 addresses that independently pass this policy.
        embedded_ipv4 = ipaddress.IPv4Address(address.packed[-4:])
        return _is_public_network_address(embedded_ipv4)
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
        and not getattr(address, "is_site_local", False)
    )


def _normalized_resolver_hostname(value: object) -> str:
    """Return the DNS-comparison form used by the one-request pin."""
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return ""
    if not isinstance(value, str):
        return ""
    return value.casefold().rstrip(".")


def _resolver_port_matches(value: object, expected_port: int, scheme: str) -> bool:
    """Recognize the numeric and service-name ports urllib3 may resolve."""
    if type(value) is int:
        return value == expected_port
    if isinstance(value, str):
        stripped = value.strip().casefold()
        return stripped == str(expected_port) or (
            stripped == scheme and expected_port == (443 if scheme == "https" else 80)
        )
    return False


def _validated_public_address_infos(address_infos: object) -> tuple[tuple[object, ...], ...]:
    """Validate and freeze one DNS answer set before any socket connects."""
    if not isinstance(address_infos, (list, tuple)) or not address_infos:
        raise requests.exceptions.ConnectionError("Universe source DNS lookup returned no addresses.")

    validated: list[tuple[object, ...]] = []
    for address_info in address_infos:
        if not isinstance(address_info, tuple) or len(address_info) != 5:
            raise requests.exceptions.ConnectionError("Universe source DNS lookup returned an invalid address.")
        socket_address = address_info[4]
        if not isinstance(socket_address, tuple) or not socket_address or not isinstance(socket_address[0], str):
            raise requests.exceptions.ConnectionError("Universe source DNS lookup returned an invalid address.")
        address_text = socket_address[0].split("%", 1)[0]
        try:
            address = ipaddress.ip_address(address_text)
        except ValueError:
            raise requests.exceptions.ConnectionError(
                "Universe source DNS lookup returned an invalid address."
            ) from None
        if not _is_public_network_address(address):
            raise requests.exceptions.InvalidURL("Universe source resolved to a non-public network address.")
        validated.append(address_info)
    return tuple(validated)


def _universe_session_get(
    session: requests.Session,
    url: str,
    request_kwargs: dict[str, object],
) -> requests.Response:
    """Connect to one validated DNS snapshot while retaining URL-host TLS checks."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
    except ValueError:
        raise requests.exceptions.InvalidURL("Universe source URL has an invalid or ambiguous authority.") from None
    if hostname is None:
        raise requests.exceptions.InvalidURL("Universe source URL has an invalid or ambiguous authority.")

    expected_hostname = hostname.casefold().rstrip(".")
    scheme = parsed.scheme.casefold()
    # Environment proxy variables would move DNS resolution and the actual
    # connection outside this pin. Universe sources are public and do not need
    # a proxy, so fail closed by bypassing the environment for this session.
    if request_kwargs.get("proxies"):
        raise requests.exceptions.InvalidURL("Universe source requests must not use a proxy.")
    session.trust_env = False
    session.proxies.clear()
    original_getaddrinfo = socket.getaddrinfo
    pinned_address_infos = _validated_public_address_infos(
        original_getaddrinfo(hostname, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    )

    def pinned_getaddrinfo(
        resolver_host: object,
        resolver_port: object,
        *args: object,
        **kwargs: object,
    ) -> list[tuple[object, ...]]:
        if _normalized_resolver_hostname(resolver_host) != expected_hostname or not _resolver_port_matches(
            resolver_port, port, scheme
        ):
            raise requests.exceptions.InvalidURL("Universe source request attempted an unexpected network destination.")
        return list(pinned_address_infos)

    original_socket_getaddrinfo = socket.getaddrinfo
    socket.getaddrinfo = pinned_getaddrinfo
    try:
        # Keep the original URL here: urllib3 consequently sends its hostname
        # as Host and uses it for TLS SNI and certificate verification, while
        # the socket resolver can return only the validated frozen addresses.
        return session.get(url, **request_kwargs)
    finally:
        socket.getaddrinfo = original_socket_getaddrinfo


def _serialize_envelope(payload: object, *, max_bytes: int) -> bytes:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("HTTP request worker protocol byte limit must be a positive integer.")
    buffer = _BoundedPickleBuffer(max_bytes)
    pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump({"protocol": _PROTOCOL_NAME, "payload": payload})
    return buffer.getvalue()


def _deserialize_envelope(encoded: bytes, *, max_bytes: int) -> object:
    if len(encoded) > max_bytes:
        raise ValueError(f"HTTP request worker protocol exceeded its {max_bytes}-byte limit.")
    stream = io.BytesIO(encoded)
    envelope = pickle.Unpickler(stream).load()
    if stream.read(1):
        raise ValueError("HTTP request worker protocol contained trailing data.")
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"protocol", "payload"}
        or envelope.get("protocol") != _PROTOCOL_NAME
    ):
        raise ValueError("HTTP request worker returned an invalid protocol envelope.")
    return envelope["payload"]


def read_worker_request() -> object:
    """Read one bounded, versioned request from the parent process."""
    encoded = sys.stdin.buffer.read(HTTP_WORKER_REQUEST_MAX_BYTES + 1)
    return _deserialize_envelope(encoded, max_bytes=HTTP_WORKER_REQUEST_MAX_BYTES)


def write_worker_result(payload: object, *, max_bytes: int) -> None:
    """Write one bounded, versioned result to the parent process."""
    try:
        encoded = _serialize_envelope(payload, max_bytes=max_bytes)
    except BaseException as exc:
        message = str(exc)[:_HTTP_WORKER_ERROR_MAX_CHARS]
        encoded = _serialize_envelope(
            ("error", type(exc).__name__, message),
            max_bytes=HTTP_WORKER_REQUEST_MAX_BYTES,
        )
    try:
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
    except (BrokenPipeError, OSError):
        pass


def _kill_and_reap(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        with suppress(OSError):
            process.kill()
    try:
        process.wait(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        with suppress(OSError):
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
    for pipe in (process.stdin, process.stdout):
        if pipe is not None:
            with suppress(OSError, ValueError):
                pipe.close()


def run_deadline_subprocess(
    command: Sequence[str],
    request_payload: object,
    *,
    deadline: float,
    result_max_bytes: int,
    timeout_message: str,
    connection_error_message: str,
    environment_overrides: Mapping[str, str] | None = None,
) -> object:
    """Run a one-shot worker with stdin/stdout transfer under one deadline."""
    request_bytes = _serialize_envelope(
        request_payload,
        max_bytes=HTTP_WORKER_REQUEST_MAX_BYTES,
    )
    if time.monotonic() >= deadline:
        raise requests.exceptions.Timeout(timeout_message)

    if type(result_max_bytes) is not int or result_max_bytes < 1:
        raise ValueError("Worker result byte limit must be a positive integer.")
    worker_environment: dict[str, str] | None = None
    if environment_overrides is not None:
        if not isinstance(environment_overrides, Mapping) or any(
            type(key) is not str or not key or "=" in key or "\x00" in key or type(value) is not str or "\x00" in value
            for key, value in environment_overrides.items()
        ):
            raise ValueError("Worker environment overrides must contain valid string names and values.")
        worker_environment = dict(os.environ)
        worker_environment.update(environment_overrides)

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            env=worker_environment,
        )
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            _kill_and_reap(process)
            raise requests.exceptions.Timeout(timeout_message)
        assert process.stdin is not None
        assert process.stdout is not None
        result = bytearray()
        reader_done = threading.Event()
        reader_failure: list[BaseException] = []

        def write_request() -> None:
            try:
                # Thread construction and scheduling are part of the same
                # absolute budget. If setup consumed it, withhold the request
                # rather than handing stale authority to the child process.
                if time.monotonic() >= deadline:
                    return
                process.stdin.write(request_bytes)
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            finally:
                with suppress(OSError, ValueError):
                    process.stdin.close()

        def read_result() -> None:
            try:
                while len(result) <= result_max_bytes:
                    chunk = process.stdout.read(min(_HTTP_WORKER_CHUNK_BYTES, result_max_bytes + 1 - len(result)))
                    if not chunk:
                        break
                    result.extend(chunk)
            except BaseException as exc:
                reader_failure.append(exc)
            finally:
                reader_done.set()

        writer = threading.Thread(target=write_request, name="deadline-worker-stdin", daemon=True)
        reader = threading.Thread(target=read_result, name="deadline-worker-stdout", daemon=True)
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise requests.exceptions.Timeout(timeout_message)
        writer.start()
        reader.start()
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            _kill_and_reap(process)
            writer.join(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
            reader.join(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
            raise requests.exceptions.Timeout(timeout_message)
        if not reader_done.wait(timeout=remaining_seconds):
            _kill_and_reap(process)
            writer.join(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
            reader.join(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
            raise requests.exceptions.Timeout(timeout_message)
        if len(result) > result_max_bytes:
            _kill_and_reap(process)
            writer.join(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
            reader.join(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
            raise requests.exceptions.ConnectionError(connection_error_message)
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise requests.exceptions.Timeout(timeout_message)
        try:
            process.wait(timeout=remaining_seconds)
        except subprocess.TimeoutExpired as exc:
            raise requests.exceptions.Timeout(timeout_message) from exc
        writer.join(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
        reader.join(timeout=_HTTP_WORKER_REAP_TIMEOUT_SECONDS)
        if time.monotonic() >= deadline:
            raise requests.exceptions.Timeout(timeout_message)
        if reader_failure:
            raise requests.exceptions.ConnectionError(connection_error_message) from reader_failure[0]
        if process.returncode != 0:
            raise requests.exceptions.ConnectionError(connection_error_message)
        try:
            payload = _deserialize_envelope(bytes(result), max_bytes=result_max_bytes)
        except Exception as exc:
            raise requests.exceptions.ConnectionError(connection_error_message) from exc
        if time.monotonic() >= deadline:
            raise requests.exceptions.Timeout(timeout_message)
        return payload
    finally:
        if process is not None:
            _kill_and_reap(process)


def _response_content_length(response: requests.Response, *, source: str) -> int | None:
    headers = response.headers if isinstance(response.headers, Mapping) else {}
    raw_length = headers.get("Content-Length")
    if raw_length is None:
        return None
    if not isinstance(raw_length, str) or re.fullmatch(r"\d+", raw_length.strip()) is None:
        raise requests.exceptions.InvalidHeader(f"{source} returned an invalid Content-Length header.")
    return int(raw_length.strip())


def _response_has_header(response: requests.Response, name: str) -> bool:
    headers = response.headers if isinstance(response.headers, Mapping) else {}
    normalized_name = name.casefold()
    return any(isinstance(header_name, str) and header_name.casefold() == normalized_name for header_name in headers)


def _response_body(
    response: requests.Response,
    *,
    source: str,
    max_body_bytes: int,
) -> bytes:
    content_length = _response_content_length(response, source=source)
    if content_length is not None and content_length > max_body_bytes:
        raise ValueError(f"{source} response exceeded the {max_body_bytes}-byte limit.")

    headers = response.headers if isinstance(response.headers, Mapping) else {}
    content_encoding = headers.get("Content-Encoding")
    if content_encoding is not None and (
        not isinstance(content_encoding, str) or content_encoding.strip().casefold() not in {"", "identity"}
    ):
        raise requests.exceptions.ContentDecodingError(
            f"{source} returned compressed content despite an identity-only request."
        )

    read_available = getattr(response.raw, "read1", None)
    if not callable(read_available):
        read_available = getattr(response.raw, "read", None)
    if not callable(read_available):
        raise TypeError(f"{source} HTTP response does not support bounded incremental reads.")

    body = bytearray()
    while True:
        try:
            chunk = read_available(_HTTP_WORKER_CHUNK_BYTES, decode_content=False)
        except TypeError:
            chunk = read_available(_HTTP_WORKER_CHUNK_BYTES)
        if not chunk:
            break
        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError(f"{source} returned a non-byte response chunk.")
        if len(body) + len(chunk) > max_body_bytes:
            raise ValueError(f"{source} response exceeded the {max_body_bytes}-byte limit.")
        body.extend(chunk)
    return bytes(body)


def _http_worker_error_payload(
    failure: BaseException,
    *,
    cleanup_failures: Sequence[tuple[str, BaseException]] = (),
) -> tuple[object, ...]:
    """Serialize one primary worker failure without hiding cleanup failures."""
    message = str(failure)

    if cleanup_failures:
        # Reserve bounded space for every cleanup action.  Exception notes do
        # not cross the worker protocol, so include their equivalent in the
        # transmitted diagnostic while retaining the primary exception type.
        cleanup_detail = "; ".join(
            f"failed to {action}: {str(cleanup_failure)[:1024]}" for action, cleanup_failure in cleanup_failures
        )
        suffix = f"; cleanup also failed: {cleanup_detail}"
        if len(suffix) >= _HTTP_WORKER_ERROR_MAX_CHARS:
            message = suffix[-_HTTP_WORKER_ERROR_MAX_CHARS:]
        else:
            message = f"{message[: _HTTP_WORKER_ERROR_MAX_CHARS - len(suffix)]}{suffix}"

    return ("error", type(failure).__name__, message[:_HTTP_WORKER_ERROR_MAX_CHARS])


def _alpaca_session_request(
    session: requests.Session,
    method: str,
    url: str,
    request_kwargs: dict[str, object],
    *,
    request_write_deadline: float | None,
    request_deadline: float | None,
    allow_loopback_test_origin: bool,
) -> requests.Response:
    if "auth" in request_kwargs:
        raise TypeError("Alpaca worker requests must use explicit credential headers.")
    if request_kwargs.get("allow_redirects") is not False:
        raise TypeError("Alpaca worker requests must disable redirects.")
    if request_kwargs.get("stream") is not True:
        raise TypeError("Alpaca worker requests must stream bounded responses.")

    # Prevent ~/.netrc credentials from adding/replacing Authorization while
    # preserving the caller's explicit APCA credential headers. Redirects are
    # disabled above so those headers cannot be forwarded to another origin.
    session.auth = _EXPLICIT_HEADER_AUTH
    if allow_loopback_test_origin:
        # The test-only loopback seam must never hand explicit APCA headers to
        # an operator-configured HTTP proxy.
        session.trust_env = False
        session.proxies.clear()
    effective_write_deadline = request_deadline
    deadline_is_overall = request_deadline is not None
    if request_write_deadline is not None and (
        effective_write_deadline is None or request_write_deadline < effective_write_deadline
    ):
        effective_write_deadline = request_write_deadline
        deadline_is_overall = False
    if effective_write_deadline is None:
        return session.request(method, url, **request_kwargs)

    token = _ALPACA_REQUEST_WRITE_DEADLINE_MONOTONIC.set(effective_write_deadline)
    kind_token = _ALPACA_REQUEST_WRITE_DEADLINE_IS_OVERALL.set(deadline_is_overall)
    original_http_connection = _Urllib3HTTPConnectionPool.ConnectionCls
    original_https_connection = _Urllib3HTTPSConnectionPool.ConnectionCls
    try:
        _remaining_alpaca_request_write_seconds()
        _Urllib3HTTPConnectionPool.ConnectionCls = _DeadlineAwareAlpacaHTTPConnection
        _Urllib3HTTPSConnectionPool.ConnectionCls = _DeadlineAwareAlpacaHTTPSConnection
        return session.request(method, url, **request_kwargs)
    finally:
        _Urllib3HTTPConnectionPool.ConnectionCls = original_http_connection
        _Urllib3HTTPSConnectionPool.ConnectionCls = original_https_connection
        _ALPACA_REQUEST_WRITE_DEADLINE_IS_OVERALL.reset(kind_token)
        _ALPACA_REQUEST_WRITE_DEADLINE_MONOTONIC.reset(token)


def _http_request_payload(request: object) -> tuple[object, ...]:
    if not isinstance(request, dict):
        raise TypeError("HTTP request worker received an invalid request.")
    mode = request.get("mode")
    expected_keys = {
        "max_body_bytes",
        "mode",
        "request_kwargs",
        "url",
    }
    if "request_deadline" in request:
        expected_keys.add("request_deadline")
    if mode == "alpaca":
        expected_keys.update({"allow_loopback_test_origin", "method", "request_write_deadline"})
    if set(request) != expected_keys:
        raise TypeError("HTTP request worker received an invalid request.")
    url = request["url"]
    request_kwargs = request["request_kwargs"]
    max_body_bytes = request["max_body_bytes"]
    if (
        mode not in {"alpaca", "tradier", "universe"}
        or type(url) is not str
        or type(request_kwargs) is not dict
        or type(max_body_bytes) is not int
        or max_body_bytes <= 0
    ):
        raise TypeError("HTTP request worker received an invalid request.")
    method = request.get("method")
    if mode == "alpaca" and method not in {"DELETE", "GET", "POST"}:
        raise TypeError("HTTP request worker received an invalid request.")
    allow_loopback_test_origin = request.get("allow_loopback_test_origin")
    if mode == "alpaca" and type(allow_loopback_test_origin) is not bool:
        raise TypeError("HTTP request worker received an invalid request.")
    if mode == "alpaca":
        assert isinstance(method, str)
        assert isinstance(url, str)
        assert isinstance(allow_loopback_test_origin, bool)
        _validate_alpaca_request_target(
            method,
            url,
            allow_loopback_test_origin=allow_loopback_test_origin,
        )
        if allow_loopback_test_origin and "proxies" in request_kwargs:
            raise requests.exceptions.InvalidURL("Alpaca loopback test requests must not use a proxy.")
    request_write_deadline = request.get("request_write_deadline")
    if mode == "alpaca" and (
        request_write_deadline is not None
        and (type(request_write_deadline) not in {float, int} or not math.isfinite(request_write_deadline))
    ):
        raise TypeError("HTTP request worker received an invalid request.")

    request_deadline = request.get("request_deadline")
    if request_deadline is not None and (
        type(request_deadline) not in {float, int} or not math.isfinite(request_deadline)
    ):
        raise TypeError("HTTP request worker received an invalid request.")

    source = {"alpaca": "Alpaca", "tradier": "Tradier", "universe": "Universe source"}[mode]
    session: requests.Session | None = None
    response: requests.Response | None = None
    primary_failure: BaseException | None = None
    try:
        if mode != "alpaca" and request_deadline is not None and time.monotonic() >= request_deadline:
            raise HttpRequestDeadlineElapsed(
                f"{source} response exceeded its overall deadline before the HTTP request could begin"
            )
        session = requests.Session()
        if mode == "universe":
            response = _universe_session_get(session, url, request_kwargs)
        elif mode == "tradier":
            if "auth" in request_kwargs:
                raise TypeError("Tradier worker requests must use the explicit Authorization header.")
            if request_kwargs.get("allow_redirects") is not False:
                raise TypeError("Tradier worker requests must disable redirects.")
            # A truthy explicit AuthBase prevents Requests from replacing the
            # caller's Bearer header with credentials from ~/.netrc. Keep
            # trust_env enabled so an operator's proxy and CA-bundle settings
            # retain their normal Requests semantics.
            session.auth = _EXPLICIT_HEADER_AUTH
            response = session.get(url, **request_kwargs)
        else:
            assert isinstance(method, str)
            assert request_write_deadline is None or isinstance(request_write_deadline, (float, int))
            response = _alpaca_session_request(
                session,
                method,
                url,
                request_kwargs,
                request_write_deadline=(None if request_write_deadline is None else float(request_write_deadline)),
                request_deadline=(None if request_deadline is None else float(request_deadline)),
                allow_loopback_test_origin=allow_loopback_test_origin,
            )
        status_code = response.status_code
        if type(status_code) is int and status_code == 200 and _response_has_header(response, "Content-Range"):
            raise requests.exceptions.InvalidHeader(
                f"{source} returned an unsolicited Content-Range header for a complete snapshot."
            )
        should_materialize_body = type(status_code) is int and (
            mode == "alpaca" or status_code == 200 or (mode == "tradier" and 400 <= status_code < 600)
        )
        if should_materialize_body:
            body = _response_body(
                response,
                source=source,
                max_body_bytes=max_body_bytes,
            )
        else:
            # Redirect metadata and unexpected-success status are sufficient
            # for the parent. In particular, never transfer a partial 206 body
            # as a complete universe or historical-price snapshot. Tradier
            # 4xx/5xx bodies remain available for useful provider diagnostics.
            body = b""
        payload: tuple[object, ...] = (
            "response",
            status_code,
            dict(response.headers),
            response.encoding,
            body,
            response.url,
            response.reason,
        )
    except BaseException as exc:
        primary_failure = exc
        payload = _http_worker_error_payload(exc)
    finally:
        cleanup_failures: list[tuple[str, BaseException]] = []
        if response is not None:
            try:
                response.close()
            except BaseException as exc:
                cleanup_failures.append(("close the HTTP response", exc))
        if session is not None:
            try:
                session.close()
            except BaseException as exc:
                cleanup_failures.append(("close the HTTP session", exc))
        if cleanup_failures and primary_failure is not None:
            payload = _http_worker_error_payload(
                primary_failure,
                cleanup_failures=cleanup_failures,
            )
    return payload


def run_http_request_with_deadline(
    url: str,
    request_kwargs: dict[str, object],
    *,
    mode: str,
    deadline: float,
    max_body_bytes: int,
    result_max_bytes: int,
    timeout_message: str,
    connection_error_message: str,
    method: str | None = None,
    request_write_deadline: float | None = None,
    allow_loopback_test_origin: bool = False,
) -> object:
    """Run the lean HTTP worker without placing request data in argv or env."""
    request_payload = {
        "max_body_bytes": max_body_bytes,
        "mode": mode,
        "request_deadline": deadline,
        "request_kwargs": request_kwargs,
        "url": url,
    }
    if method is not None:
        request_payload["method"] = method
        request_payload["request_write_deadline"] = request_write_deadline
        request_payload["allow_loopback_test_origin"] = allow_loopback_test_origin
    return run_deadline_subprocess(
        _HTTP_WORKER_COMMAND,
        request_payload,
        deadline=deadline,
        result_max_bytes=result_max_bytes,
        timeout_message=timeout_message,
        connection_error_message=connection_error_message,
    )


def response_from_http_worker_payload(
    payload: object,
    *,
    max_body_bytes: int,
    invalid_response_message: str,
    exception_from_process: Callable[[str, str], BaseException],
) -> requests.Response:
    """Validate one worker result exactly before constructing a response."""

    def invalid_response() -> requests.exceptions.ConnectionError:
        return requests.exceptions.ConnectionError(invalid_response_message)

    if type(payload) is not tuple or not payload or type(payload[0]) is not str:
        raise invalid_response()
    if payload[0] == "error":
        if (
            len(payload) != 3
            or type(payload[1]) is not str
            or not payload[1]
            or len(payload[1]) > _HTTP_WORKER_ERROR_MAX_CHARS
            or type(payload[2]) is not str
            or len(payload[2]) > _HTTP_WORKER_ERROR_MAX_CHARS
        ):
            raise invalid_response()
        failure = exception_from_process(payload[1], payload[2])
        if not isinstance(failure, BaseException):
            raise invalid_response()
        raise failure
    if payload[0] != "response" or len(payload) != 7:
        raise invalid_response()

    _, status_code, headers, encoding, body, response_url, reason = payload
    if (
        type(status_code) is not int
        or not 100 <= status_code <= 599
        or type(headers) is not dict
        or any(type(name) is not str or type(value) is not str for name, value in headers.items())
        or (encoding is not None and type(encoding) is not str)
        or type(body) is not bytes
        or len(body) > max_body_bytes
        or type(response_url) is not str
        or not response_url
        or (reason is not None and type(reason) not in {str, bytes})
    ):
        raise invalid_response()

    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers)
    response.encoding = encoding
    response._content = body
    response._content_consumed = True
    response.url = response_url
    response.reason = reason
    return response


def main() -> None:
    request: object = None
    try:
        request = read_worker_request()
        payload = _http_request_payload(request)
    except BaseException as exc:
        payload = ("error", type(exc).__name__, str(exc)[:_HTTP_WORKER_ERROR_MAX_CHARS])

    result_limit = HTTP_WORKER_REQUEST_MAX_BYTES
    if isinstance(request, dict):
        max_body_bytes = request.get("max_body_bytes")
        if type(max_body_bytes) is int and max_body_bytes > 0:
            result_limit = max_body_bytes + 8 * 1024 * 1024
    write_worker_result(payload, max_bytes=result_limit)


if __name__ == "__main__":
    main()
