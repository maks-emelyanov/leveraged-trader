from __future__ import annotations

import json
import multiprocessing
import socket
import unittest
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Lock, Thread
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from leveraged_trader import _http_deadline_worker as http_deadline_worker
from leveraged_trader.config import UniverseConfig
from leveraged_trader.universe import (
    AUDIT_UNIVERSE_SOURCES,
    DEFAULT_USER_AGENT,
    EXCLUDED_UNIVERSE_SYMBOLS,
    ISSUER_UNIVERSE_SOURCES,
    NASDAQ_LISTED_URL,
    OTHER_LISTED_URL,
    WORKFLOW_ETN_SOURCES,
    ActiveListedSymbols,
    UniverseSource,
    _cboe_issuer_html_to_universe,
    _cboe_symbol_csv_to_universe,
    _decode_quoted_js_text,
    _defiance_json_to_universe,
    _etracs_leverage_table_to_universe,
    _fetch_enabled_sources,
    _fetch_source_text,
    _get_universe_response,
    _get_universe_response_with_deadline,
    _graniteshares_html_to_universe,
    _html_cards_to_universe,
    _issuer_table_to_universe,
    _js_ticker_name_to_universe,
    _leverage_shares_html_to_universe,
    _leveraged_product_rows,
    _merge_universe_sources,
    _microsectors_html_to_universe,
    _read_nasdaq_symbol_file,
    _resolve_discovered_product_rows,
    _resolve_workflow_source_product_rows,
    _rex_menu_html_to_universe,
    _sec_company_tickers_to_universe,
    _sec_exchange_tickers_to_universe,
    _sec_mutual_fund_tickers_to_universe,
    _strict_json_loads,
    _tradr_html_to_universe,
    _volatilityshares_html_to_universe,
    _with_audit_metadata,
    _workflow_candidates,
    _workflow_issuer_source_to_universe,
    build_nasdaq_universe_table,
    build_universe_audit_report,
    determine_workflow_asset_groups,
    determine_workflow_assets,
    infer_leverage_and_direction,
    infer_rsi_mapping,
    infer_rsi_symbol,
    is_long_leveraged_name,
    is_short_leveraged_name,
    leveraged_name_filter,
    load_active_listed_symbols,
    load_audit_universe_sources,
    load_current_etf_universe,
    load_etn_universe,
    load_issuer_etf_universe,
    select_short_workflow_universe,
    select_universes,
)


def _leverage_shares_products_page(
    *,
    count: int = 100,
    final_overrides: dict[str, object] | None = None,
) -> str:
    products = []
    for index in range(count):
        product: dict[str, object] = {
            "name": f"2x Long ASSET{index} Daily ETF",
            "fund": f"2x Long ASSET{index} Daily ETF",
            "ticker": f"X{index:03d}",
            "product_url": (f"https://leverageshares.com/us/etfs/leverage-shares-2x-long-asset{index}-daily-etf"),
            "leverage_factor": "2",
            "category": "Leveraged",
            "is_api_product": False,
            "externalLink": False,
        }
        if index == count - 1 and final_overrides:
            product.update(final_overrides)
        products.append(product)
    return f"<script>window.productsData = {json.dumps(products)};</script>"


def _streaming_response(
    *chunks: bytes,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    encoding: str | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://issuer.test/products"
    response.headers = headers or {}
    response.encoding = encoding
    response.raw = Mock()
    response.raw.read1.side_effect = [*chunks, b""]
    response.close = Mock()
    return response


def _universe_request_from_daemon(url: str, send_connection: object) -> None:
    try:
        response = _get_universe_response(url, 2)
        result: tuple[object, ...] = ("response", response.status_code, response.text)
    except BaseException as exc:
        result = ("error", type(exc).__name__, str(exc))
    try:
        send_connection.send(result)
    finally:
        send_connection.close()


def _address_info(address: str, *, port: int = 443) -> list[tuple[object, ...]]:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    socket_address = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
    return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", socket_address)]


class UniverseTests(unittest.TestCase):
    def test_default_user_agent_uses_public_project_identity(self) -> None:
        self.assertIn("github.com/maks-emelyanov/leveraged-trader", DEFAULT_USER_AGENT)

    def test_public_universe_loaders_reject_unbounded_timeouts_before_fetching(self) -> None:
        loaders = (
            load_active_listed_symbols,
            load_current_etf_universe,
            load_issuer_etf_universe,
            load_etn_universe,
            load_audit_universe_sources,
        )
        invalid_timeouts = (True, False, 0, 601, 1.0, 30.5, "30", None)

        with (
            patch(
                "leveraged_trader.universe._get_universe_response",
                side_effect=AssertionError("network fetch must not start"),
            ),
            patch(
                "leveraged_trader.universe._fetch_enabled_sources",
                side_effect=AssertionError("source fetches must not start"),
            ),
        ):
            for loader in loaders:
                for timeout in invalid_timeouts:
                    with (
                        self.subTest(loader=loader.__name__, timeout=timeout),
                        self.assertRaisesRegex(ValueError, "integer between 1 and 600"),
                    ):
                        loader(timeout=timeout)  # type: ignore[arg-type]

    def test_workflow_universe_rejects_unbounded_timeout_before_loading_sources(self) -> None:
        with patch("leveraged_trader.universe.load_current_etf_universe") as mock_load:
            for timeout in (True, 0, 601, 1.0, "30"):
                with (
                    self.subTest(timeout=timeout),
                    self.assertRaisesRegex(ValueError, "integer between 1 and 600"),
                ):
                    determine_workflow_asset_groups(
                        UniverseConfig(request_timeout_seconds=timeout)  # type: ignore[arg-type]
                    )

        mock_load.assert_not_called()

    @patch("leveraged_trader.universe._get_universe_response")
    def test_source_exception_diagnostic_is_bounded_printable_and_redacted(self, mock_get: Mock) -> None:
        secret = "TOP-SECRET-UNIVERSE-TOKEN"
        response = requests.Response()
        response.status_code = 500
        response.url = "https://issuer.test/products"
        response.reason = f"credential: {secret}\x1b[2J\x00END" + "X" * 1_000_000
        mock_get.return_value = response

        result = _fetch_source_text(
            UniverseSource("Issuer", response.url, "issuer_etf"),
            30,
        )

        self.assertIsNone(result.text)
        self.assertLessEqual(len(result.error), 250)
        self.assertTrue(all(character.isprintable() for character in result.error))
        self.assertNotIn(secret, result.error)
        self.assertNotIn("\x1b", result.error)
        self.assertNotIn("\x00", result.error)
        self.assertIn("redacted credential", result.error)

    @patch("leveraged_trader.universe._get_universe_response")
    def test_source_exception_diagnostic_does_not_invoke_custom_str(self, mock_get: Mock) -> None:
        rendering_calls: list[object] = []

        class BrokenSourceError(RuntimeError):
            def __str__(self) -> str:
                rendering_calls.append(self)
                raise LookupError("secondary formatting failure")

        mock_get.side_effect = BrokenSourceError("credential: TOP-SECRET-UNIVERSE-TOKEN")

        result = _fetch_source_text(
            UniverseSource("Issuer", "https://issuer.test/products", "issuer_etf"),
            30,
        )

        self.assertIsNone(result.text)
        self.assertEqual(rendering_calls, [])
        self.assertEqual(result.error, "BrokenSourceError: [redacted credential]")

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_response_accepts_exact_byte_limit_and_closes(self, mock_get: Mock) -> None:
        response = _streaming_response(b"abcd", b"efgh", headers={"Content-Length": "8"})
        mock_get.return_value = response

        with patch("leveraged_trader.universe.UNIVERSE_RESPONSE_MAX_BYTES", 8):
            result = _fetch_source_text(
                UniverseSource("Issuer", response.url, "issuer_etf"),
                30,
            )

        self.assertEqual(result.text, "abcdefgh")
        self.assertEqual(result.error, "")
        self.assertEqual(response.raw.read1.call_count, 3)
        response.close.assert_called_once()
        self.assertTrue(mock_get.call_args.kwargs["stream"])
        self.assertEqual(mock_get.call_args.kwargs["headers"]["Accept-Encoding"], "identity")

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_preserves_read_failure_when_response_close_also_fails(self, mock_get: Mock) -> None:
        response = _streaming_response()
        response.raw.read1.side_effect = requests.ReadTimeout("primary read failed")
        response.close.side_effect = OSError("response close failed")
        mock_get.return_value = response

        with self.assertRaises(requests.ReadTimeout) as raised:
            _get_universe_response(response.url, 30)

        self.assertIn("primary read failed", str(raised.exception))
        self.assertIn(
            "Failed to close the universe HTTP response: response close failed",
            getattr(raised.exception, "__notes__", []),
        )
        response.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_response_rejects_unhandled_3xx_before_reading_body(self, mock_get: Mock) -> None:
        response = _streaming_response(
            b"<table><tr><th>Symbol</th><th>Fund Name</th><th>Fund Type</th></tr>",
            b"<tr><td>TQQQ</td><td>ProShares UltraPro QQQ</td><td>ETF</td></tr></table>",
            status_code=300,
        )
        mock_get.return_value = response

        result = _fetch_source_text(
            UniverseSource("Issuer", response.url, "issuer_etf"),
            30,
        )

        self.assertIsNone(result.text)
        self.assertIn("unexpected HTTP status 300", result.error)
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    def test_universe_worker_does_not_materialize_unhandled_3xx_body(self) -> None:
        response = _streaming_response(
            b"<table><tr><th>Symbol</th><th>Fund Name</th><th>Fund Type</th></tr>",
            b"<tr><td>TQQQ</td><td>ProShares UltraPro QQQ</td><td>ETF</td></tr></table>",
            status_code=300,
        )
        session = Mock()

        with (
            patch.object(http_deadline_worker.requests, "Session", return_value=session),
            patch.object(http_deadline_worker, "_universe_session_get", return_value=response),
        ):
            payload = http_deadline_worker._http_request_payload(
                {
                    "max_body_bytes": 1_024,
                    "mode": "universe",
                    "request_kwargs": {"stream": True},
                    "url": response.url,
                }
            )

        self.assertEqual(payload[:2], ("response", 300))
        self.assertEqual(payload[4], b"")
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()
        session.close.assert_called_once()

    def test_universe_worker_response_schema_rejects_malformed_payloads(self) -> None:
        invalid_payloads = (
            ("response", "200", {}, None, b"data", "https://issuer.test/products", "OK"),
            ("response", 200, [], None, b"data", "https://issuer.test/products", "OK"),
            ("response", 200, {}, None, 10**12, "https://issuer.test/products", "OK"),
            ("response", 200, {}, None, b"12345", "https://issuer.test/products", "OK"),
            ("response", 200, {}, None, b"data", None, "OK"),
            ("error", "Timeout", object()),
        )
        for payload in invalid_payloads:
            with (
                self.subTest(payload=payload),
                patch(
                    "leveraged_trader.universe.run_http_request_with_deadline",
                    return_value=payload,
                ),
                patch("leveraged_trader.universe.UNIVERSE_RESPONSE_MAX_BYTES", 4),
                self.assertRaisesRegex(requests.exceptions.ConnectionError, "invalid response"),
            ):
                _get_universe_response_with_deadline(
                    "https://issuer.test/products",
                    deadline=1.0,
                    request_kwargs={},
                )

    def test_universe_worker_does_not_materialize_partial_content_body(self) -> None:
        body = b"complete-looking universe fragment"
        response = _streaming_response(
            body,
            status_code=206,
            headers={"Content-Range": f"bytes 0-{len(body) - 1}/{len(body) + 100}"},
        )
        session = Mock()

        with (
            patch.object(http_deadline_worker.requests, "Session", return_value=session),
            patch.object(http_deadline_worker, "_universe_session_get", return_value=response),
        ):
            payload = http_deadline_worker._http_request_payload(
                {
                    "max_body_bytes": 1_024,
                    "mode": "universe",
                    "request_kwargs": {"stream": True},
                    "url": response.url,
                }
            )

        self.assertEqual(payload[:2], ("response", 206))
        self.assertEqual(payload[2]["Content-Range"], f"bytes 0-{len(body) - 1}/{len(body) + 100}")
        self.assertEqual(payload[4], b"")
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()
        session.close.assert_called_once()

    def test_universe_worker_rejects_content_range_on_exact_200_without_reading(self) -> None:
        body = b"complete-looking universe body"
        response = _streaming_response(
            body,
            status_code=200,
            headers={"cOnTeNt-RaNgE": f"bytes 0-{len(body) - 1}/{len(body) + 1}"},
        )
        session = Mock()

        with (
            patch.object(http_deadline_worker.requests, "Session", return_value=session),
            patch.object(http_deadline_worker, "_universe_session_get", return_value=response),
        ):
            payload = http_deadline_worker._http_request_payload(
                {
                    "max_body_bytes": 1_024,
                    "mode": "universe",
                    "request_kwargs": {"stream": True},
                    "url": response.url,
                }
            )

        self.assertEqual(payload[:2], ("error", "InvalidHeader"))
        self.assertIn("unsolicited Content-Range", payload[2])
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()
        session.close.assert_called_once()

    def test_universe_worker_preserves_success_and_attempts_both_failing_cleanups(self) -> None:
        response = _streaming_response(b"complete universe body")
        response.close.side_effect = OSError("response close failed")
        session = Mock()
        session.close.side_effect = RuntimeError("session close failed")

        with (
            patch.object(http_deadline_worker.requests, "Session", return_value=session),
            patch.object(http_deadline_worker, "_universe_session_get", return_value=response),
        ):
            payload = http_deadline_worker._http_request_payload(
                {
                    "max_body_bytes": 1_024,
                    "mode": "universe",
                    "request_kwargs": {"stream": True},
                    "url": response.url,
                }
            )

        self.assertEqual(payload[:2], ("response", 200))
        self.assertEqual(payload[4], b"complete universe body")
        response.close.assert_called_once()
        session.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_response_rejects_oversized_content_length_without_reading(
        self,
        mock_get: Mock,
    ) -> None:
        response = _streaming_response(b"unread", headers={"Content-Length": "9"})
        mock_get.return_value = response

        with patch("leveraged_trader.universe.UNIVERSE_RESPONSE_MAX_BYTES", 8):
            result = _fetch_source_text(
                UniverseSource("Issuer", response.url, "issuer_etf"),
                30,
            )

        self.assertIsNone(result.text)
        self.assertIn("8-byte limit", result.error)
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_response_rejects_stream_that_crosses_byte_limit_and_closes(
        self,
        mock_get: Mock,
    ) -> None:
        response = _streaming_response(b"abcd", b"efghi")
        mock_get.return_value = response

        with patch("leveraged_trader.universe.UNIVERSE_RESPONSE_MAX_BYTES", 8):
            result = _fetch_source_text(
                UniverseSource("Issuer", response.url, "issuer_etf"),
                30,
            )

        self.assertIsNone(result.text)
        self.assertIn("8-byte limit", result.error)
        response.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_response_rejects_compression_before_decompression(self, mock_get: Mock) -> None:
        response = _streaming_response(
            b"compressed-body",
            headers={"Content-Encoding": "gzip"},
        )
        mock_get.return_value = response

        result = _fetch_source_text(
            UniverseSource("Issuer", response.url, "issuer_etf"),
            30,
        )

        self.assertIsNone(result.text)
        self.assertIn("compressed content", result.error)
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_response_rejects_untrusted_character_decoder_and_closes(self, mock_get: Mock) -> None:
        response = _streaming_response(rb"\x41", encoding="unicode_escape")
        mock_get.return_value = response

        result = _fetch_source_text(
            UniverseSource("Issuer", response.url, "issuer_etf"),
            30,
        )

        self.assertIsNone(result.text)
        self.assertIn("unsupported character encoding", result.error)
        response.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_concurrent_fetch_results_remain_individually_bounded_and_close_responses(
        self,
        mock_get: Mock,
    ) -> None:
        exact = _streaming_response(b"safe")
        oversized = _streaming_response(b"tool", b"arge")
        mock_get.side_effect = lambda url, **_kwargs: exact if url.endswith("/exact") else oversized
        sources = [
            UniverseSource("Exact", "https://issuer.test/exact", "issuer_etf"),
            UniverseSource("Oversized", "https://issuer.test/oversized", "issuer_etf"),
        ]

        with (
            patch("leveraged_trader.universe.UNIVERSE_FETCH_MAX_WORKERS", 2),
            patch("leveraged_trader.universe.UNIVERSE_RESPONSE_MAX_BYTES", 4),
        ):
            results = _fetch_enabled_sources(sources, 30)

        self.assertEqual(results[0].text, "safe")
        self.assertIsNone(results[1].text)
        self.assertIn("4-byte limit", results[1].error)
        exact.close.assert_called_once()
        oversized.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_streamed_response_checks_deadline_between_incremental_reads(self, mock_get: Mock) -> None:
        response = _streaming_response(b"first", b"second")
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.time.monotonic",
            side_effect=[100.0, 100.0, 100.2, 100.3, 100.9, 101.0],
        ):
            result = _fetch_source_text(
                UniverseSource("Issuer", response.url, "issuer_etf"),
                1,
            )

        self.assertIsNone(result.text)
        self.assertIn("response deadline", result.error)
        response.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_streamed_response_rejects_eof_observed_after_deadline(self, mock_get: Mock) -> None:
        response = _streaming_response(b"complete")
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.time.monotonic",
            side_effect=[100.0, 100.0, 100.1, 100.2, 100.3, 101.0],
        ):
            result = _fetch_source_text(
                UniverseSource("Issuer", response.url, "issuer_etf"),
                1,
            )

        self.assertIsNone(result.text)
        self.assertIn("response deadline", result.error)
        response.close.assert_called_once()

    def test_universe_production_worker_rejects_loopback_before_request(self) -> None:
        request_received = Event()

        class PrivateHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                request_received.set()
                self.send_response(200)
                self.end_headers()

            def log_message(self, _format: str, *args: object) -> None:
                pass

        wrapper_called = False

        def parent_wrapper(*_args: object, **_kwargs: object) -> requests.Response:
            nonlocal wrapper_called
            wrapper_called = True
            raise AssertionError("parent requests.get wrapper must not run")

        server = HTTPServer(("127.0.0.1", 0), PrivateHandler)
        worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        worker.start()
        try:
            with (
                patch("leveraged_trader.universe.requests.get", new=parent_wrapper),
                self.assertRaisesRegex(requests.exceptions.InvalidURL, "non-public network address"),
            ):
                _get_universe_response(f"http://127.0.0.1:{server.server_port}/metadata", 2)
        finally:
            server.shutdown()
            worker.join(timeout=3.0)
            server.server_close()

        self.assertFalse(worker.is_alive())
        self.assertFalse(wrapper_called)
        self.assertFalse(request_received.is_set())

    def test_universe_dns_pin_reuses_snapshot_and_blocks_rebinding_on_next_hop(self) -> None:
        public_answer = _address_info("93.184.216.34")
        private_answer = _address_info("127.0.0.1")

        class ResolvingSession:
            def __init__(self) -> None:
                self.trust_env = True
                self.proxies: dict[str, str] = {}
                self.connected = False
                self.resolutions: list[list[tuple[object, ...]]] = []
                self.requested_url = ""

            def get(self, url: str, **_kwargs: object) -> requests.Response:
                self.requested_url = url
                first = socket.getaddrinfo("issuer.example", 443, type=socket.SOCK_STREAM)
                second = socket.getaddrinfo("issuer.example", 443, type=socket.SOCK_STREAM)
                self.resolutions.extend((first, second))
                self.connected = True
                return requests.Response()

        first_session = ResolvingSession()
        rebound_session = ResolvingSession()
        with (
            patch.dict("os.environ", {"HTTPS_PROXY": "http://127.0.0.1:8080"}),
            patch.object(
                http_deadline_worker.socket,
                "getaddrinfo",
                side_effect=[public_answer, private_answer],
            ) as resolver,
        ):
            http_deadline_worker._universe_session_get(
                first_session,  # type: ignore[arg-type]
                "https://issuer.example/start",
                {"stream": True},
            )
            with self.assertRaisesRegex(requests.exceptions.InvalidURL, "non-public network address"):
                http_deadline_worker._universe_session_get(
                    rebound_session,  # type: ignore[arg-type]
                    "https://issuer.example/redirected",
                    {"stream": True},
                )

        self.assertEqual(resolver.call_count, 2)
        self.assertEqual(first_session.resolutions, [public_answer, public_answer])
        self.assertFalse(first_session.trust_env)
        self.assertEqual(first_session.requested_url, "https://issuer.example/start")
        self.assertTrue(first_session.connected)
        self.assertFalse(rebound_session.trust_env)
        self.assertFalse(rebound_session.connected)

    def test_universe_dns_pin_rejects_every_non_public_address_class(self) -> None:
        non_public_addresses = (
            "0.0.0.0",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.1.1",
            "224.0.0.1",
            "::1",
            "fe80::1",
            "fc00::1",
            "fec0::1",
            "ff02::1",
            "5f00::1",
            "::ffff:127.0.0.1",
        )
        for address in non_public_addresses:
            with (
                self.subTest(address=address),
                self.assertRaisesRegex(requests.exceptions.InvalidURL, "non-public network address"),
            ):
                http_deadline_worker._validated_public_address_infos(_address_info(address))

    def test_universe_dns_pin_rejects_local_use_nat64_prefix(self) -> None:
        # RFC 8215's /48 can be routed to local/private destinations and does
        # not expose one unambiguous, globally constrained IPv4 destination.
        with self.assertRaisesRegex(requests.exceptions.InvalidURL, "non-public network address"):
            http_deadline_worker._validated_public_address_infos(_address_info("64:ff9b:1::808:808"))

    def test_universe_dns_pin_allows_well_known_nat64_for_public_ipv4(self) -> None:
        address_infos = _address_info("64:ff9b::5db8:d822")

        validated = http_deadline_worker._validated_public_address_infos(address_infos)

        self.assertEqual(validated, tuple(address_infos))

    def test_universe_dns_pin_rejects_well_known_nat64_for_non_public_ipv4(self) -> None:
        translated_addresses = (
            "64:ff9b::",
            "64:ff9b::a00:1",
            "64:ff9b::6440:1",
            "64:ff9b::7f00:1",
            "64:ff9b::a9fe:1",
            "64:ff9b::c000:201",
            "64:ff9b::e000:1",
            "64:ff9b::f000:1",
        )
        for address in translated_addresses:
            with (
                self.subTest(address=address),
                self.assertRaisesRegex(requests.exceptions.InvalidURL, "non-public network address"),
            ):
                http_deadline_worker._validated_public_address_infos(_address_info(address))

    def test_universe_private_address_rejection_runs_inside_daemonic_process(self) -> None:
        context = multiprocessing.get_context("spawn")
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_universe_request_from_daemon,
            args=("http://127.0.0.1:9/issuer", send_connection),
        )
        process.daemon = True
        try:
            process.start()
            send_connection.close()
            self.assertTrue(receive_connection.poll(5.0))
            result = receive_connection.recv()
            self.assertEqual(result[:2], ("error", "InvalidURL"))
            self.assertIn("non-public network address", result[2])
            process.join(timeout=2.0)
            self.assertFalse(process.is_alive())
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            receive_connection.close()
            send_connection.close()
            process.close()

    def test_cross_source_conflict_diagnostics_list_only_each_contributors_symbols(self) -> None:
        primary_rows = [
            {
                "symbol": f"P{row:03d}",
                "name": "Example 2X Long QQQ Daily ETF",
                "fund_type": "ETF",
            }
            for row in range(50)
        ]
        primary_rows.append(
            {
                "symbol": "ZZZ",
                "name": "Example 2X Long QQQ Daily ETF",
                "fund_type": "ETF",
            }
        )
        issuer_records = [
            {
                "symbol": "ZZZ",
                "name": "Example 2X Long QQQ Daily ETN",
                "fund_type": "ETN (Issuer A)",
                "source": "Issuer A table",
            }
        ]
        etn_records = [
            {
                "symbol": f"P{row:03d}",
                "name": "Example 2X Long QQQ Daily ETN",
                "fund_type": "ETN (ETN B)",
                "source": "ETN B table",
            }
            for row in range(50)
        ]
        issuer_rows = pd.DataFrame(issuer_records)
        issuer_rows.attrs["workflow_canonical_product_rows"] = issuer_records
        issuer_rows.attrs["workflow_symbol_sources"] = {"ZZZ": (("Issuer A", "https://a.test"),)}
        etn_rows = pd.DataFrame(etn_records)
        etn_rows.attrs["workflow_canonical_product_rows"] = etn_records
        etn_rows.attrs["workflow_symbol_sources"] = {
            record["symbol"]: (("ETN B", "https://b.test"),) for record in etn_records
        }
        status = pd.DataFrame(
            [
                {
                    "source": "Nasdaq ETF definitions",
                    "source_type": "primary_etf",
                    "url": "https://www.nasdaqtrader.com/trader.aspx?id=etf_definitions",
                    "status": "loaded",
                    "error": "",
                },
                {
                    "source": "Issuer A",
                    "source_type": "issuer_etf",
                    "url": "https://a.test",
                    "status": "loaded",
                    "error": "",
                },
                {
                    "source": "ETN B",
                    "source_type": "etn_issuer",
                    "url": "https://b.test",
                    "status": "loaded",
                    "error": "",
                },
            ]
        )

        _primary, _discovered, resolved_status = _resolve_discovered_product_rows(
            pd.DataFrame(primary_rows),
            issuer_rows,
            etn_rows,
            status,
        )

        issuer_error = resolved_status.loc[resolved_status["source"].eq("Issuer A"), "error"].iloc[0]
        etn_error = resolved_status.loc[resolved_status["source"].eq("ETN B"), "error"].iloc[0]
        self.assertIn("ZZZ", issuer_error)
        self.assertNotIn("P000", issuer_error)
        self.assertIn("P000", etn_error)
        self.assertNotIn("ZZZ", etn_error)

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_request_requires_explicit_contact_identity(self, mock_get: Mock) -> None:
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov/files/example.json",
            source_type="filing_audit",
        )

        with patch.dict("os.environ", {"SEC_USER_AGENT": ""}):
            result = _fetch_source_text(source, 30)

        self.assertIsNone(result.text)
        self.assertIn("SEC_USER_AGENT", result.error)
        mock_get.assert_not_called()

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_request_rejects_example_identity_placeholder(self, mock_get: Mock) -> None:
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov/files/example.json",
            source_type="filing_audit",
        )

        with patch.dict(
            "os.environ",
            {"SEC_USER_AGENT": "leveraged-trader/0.1 your-name operator@example.test"},
        ):
            result = _fetch_source_text(source, 30)

        self.assertIsNone(result.text)
        self.assertIn("SEC_USER_AGENT", result.error)
        mock_get.assert_not_called()

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_request_rejects_missing_or_placeholder_non_email_identity(self, mock_get: Mock) -> None:
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov/files/example.json",
            source_type="filing_audit",
        )
        invalid_user_agents = [
            "ops@real-domain.com",
            "x ops@real-domain.com",
            "x-- ops@real-domain.com",
            "mailto:ops@real-domain.com",
            "email: ops@real-domain.com",
            "contact ops@real-domain.com",
            "operator/1.0 ops@real-domain.com",
            "app-v1 ops@real-domain.com",
            "app2 ops@real-domain.com",
            "appv2 ops@real-domain.com",
            "operator2 ops@real-domain.com",
            "version2 ops@real-domain.com",
            "name: ops@real-domain.com",
            "company: ops@real-domain.com",
            "organization: ops@real-domain.com",
            "Project Maintainer ops@real-domain.com",
            "Support Team ops@real-domain.com",
            "Example Project ops@real-domain.com",
            "Your Project ops@real-domain.com",
            "ProjectName ops@real-domain.com",
            "Project Placeholder ops@real-domain.com",
            "Test ops@real-domain.com",
            "Unknown ops@real-domain.com",
            "TBD ops@real-domain.com",
            "None ops@real-domain.com",
            "operators ops@real-domain.com",
            "contacts ops@real-domain.com",
            "applications ops@real-domain.com",
            "companies ops@real-domain.com",
            "organizations ops@real-domain.com",
            "maintainers ops@real-domain.com",
            "operators2 ops@real-domain.com",
            "applications-v2beta3 ops@real-domain.com",
            "https:// ops@real-domain.com",
            "http:// ops@real-domain.com",
            "ftp:// ops@real-domain.com",
            "ssh:// ops@real-domain.com",
            "git:// ops@real-domain.com",
            "file:// ops@real-domain.com",
            "smtp:// ops@real-domain.com",
            "https:/ ops@real-domain.com",
            "http:/ ops@real-domain.com",
            "https: ops@real-domain.com",
            "https:provider ops@real-domain.com",
            "https ops@real-domain.com",
            "http ops@real-domain.com",
            "ftp ops@real-domain.com",
            "https://example.com ops@real-domain.com",
            "https://status.sub.example.org ops@real-domain.com",
            "https://project.invalid ops@real-domain.com",
            "https://localhost ops@real-domain.com",
            "https://project.local ops@real-domain.com",
            "https://router.home.arpa ops@real-domain.com",
            "https://project.alt ops@real-domain.com",
            "https://project.internal ops@real-domain.com",
            "https://project.onion ops@real-domain.com",
            "http://127.0.0.1 ops@real-domain.com",
            "http://127.1 ops@real-domain.com",
            "http://0177.0.0.1 ops@real-domain.com",
            "http://0x7f.0.0.1 ops@real-domain.com",
            "http://192.168.1.1 ops@real-domain.com",
            "https://256.256.256.256/project ops@real-domain.com",
            "https://999.999/project ops@real-domain.com",
            "https://09.0.0.1/project ops@real-domain.com",
            "https://127.0.0.999/project ops@real-domain.com",
            "example.com ops@real-domain.com",
            "www.example.com ops@real-domain.com",
            "project.local ops@real-domain.com",
            "router.home.arpa ops@real-domain.com",
            "reverse.in-addr.arpa ops@real-domain.com",
            "project.alt ops@real-domain.com",
            "project.internal ops@real-domain.com",
            "project.onion ops@real-domain.com",
            "project.corp ops@real-domain.com",
            "project.home ops@real-domain.com",
            "project.mail ops@real-domain.com",
            f"{'A' * 2_049} ops@real-domain.com",
            "Example Application ops@real-domain.com",
            "Leveraged Trader contact@example.com",
            "Leveraged Trader Maintainer ops@sub.example.com",
            "Leveraged Trader Maintainer ops@alerts.example.net",
            "Leveraged Trader Maintainer ops@project.local",
            "Leveraged Trader Maintainer ops@home.arpa",
            "Leveraged Trader Maintainer ops@reverse.in-addr.arpa",
            "Leveraged Trader Maintainer ops@project.alt",
            "Leveraged Trader Maintainer ops@project.internal",
            "Leveraged Trader Maintainer ops@project.onion",
            "Leveraged Trader Maintainer ops@project.corp",
            "Leveraged Trader Maintainer ops@project.home",
            "Leveraged Trader Maintainer ops@project.mail",
            "https://project.corp/northstar ops@real-domain.com",
            "https://project.home/northstar ops@real-domain.com",
            "https://project.mail/northstar ops@real-domain.com",
            "https://project.xn--a/northstar ops@real-domain.com",
            "project.xn--a ops@real-domain.com",
            "project.xn-- ops@real-domain.com",
            "project.xn--- ops@real-domain.com",
            "project.xn--p1ai- ops@real-domain.com",
            "project.xn--\u00e9 ops@real-domain.com",
            "project.xn--\u00f1 ops@real-domain.com",
        ]

        for user_agent in invalid_user_agents:
            with (
                self.subTest(user_agent=user_agent),
                patch.dict("os.environ", {"SEC_USER_AGENT": user_agent}),
            ):
                result = _fetch_source_text(source, 30)
                self.assertIsNone(result.text)
                self.assertIn("SEC_USER_AGENT", result.error)

        mock_get.assert_not_called()

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_request_rejects_control_characters_in_contact_header(self, mock_get: Mock) -> None:
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov/files/example.json",
            source_type="filing_audit",
        )

        for control in ("\n", "\r", "\t", "\x00", "\x1b", "\u202e"):
            user_agent = f"Leveraged Trader{control}Maintainer sec-ops@leveraged-trader.dev"
            with (
                self.subTest(control=repr(control)),
                patch("leveraged_trader.universe.os.environ.get", return_value=user_agent),
            ):
                result = _fetch_source_text(source, 30)

            self.assertIsNone(result.text)
            self.assertIn("SEC_USER_AGENT", result.error)

        mock_get.assert_not_called()

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_request_rejects_non_latin_1_contact_header(self, mock_get: Mock) -> None:
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov/files/example.json",
            source_type="filing_audit",
        )

        for non_latin_1_text in ("€", "秘密", "😀"):
            user_agent = f"Leveraged Trader {non_latin_1_text} Maintainer sec-ops@leveraged-trader.dev"
            with (
                self.subTest(non_latin_1_text=repr(non_latin_1_text)),
                patch("leveraged_trader.universe.os.environ.get", return_value=user_agent),
            ):
                result = _fetch_source_text(source, 30)

            self.assertIsNone(result.text)
            self.assertIn("SEC_USER_AGENT", result.error)

        mock_get.assert_not_called()

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_request_rejects_malformed_contact_email_syntax(self, mock_get: Mock) -> None:
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov/files/example.json",
            source_type="filing_audit",
        )
        malformed_contacts = (
            "Leveraged Trader Maintainer .ops@real-domain.com",
            "Leveraged Trader Maintainer ops.@real-domain.com",
            "Leveraged Trader Maintainer ops..team@real-domain.com",
            "Leveraged Trader Maintainer ops@bad_domain.com",
            "Leveraged Trader Maintainer ops@real..domain.com",
            "Leveraged Trader Maintainer ops@-real-domain.com",
            "Leveraged Trader Maintainer ops@real-domain-.com",
            "Leveraged Trader Maintainer ops@real-domain.com-",
            "Leveraged Trader Maintainer foo@bar@real-domain.com",
            "Leveraged Trader Maintainer foo()bar@real-domain.com",
            "Leveraged Trader Maintainer foo,bar@real-domain.com",
            "Leveraged Trader Maintainer foo:bar@real-domain.com",
            r"Leveraged Trader Maintainer foo\bar@real-domain.com",
            "Leveraged Trader Maintainer ops@real-domain.com@evil",
            "Leveraged Trader Maintainer ops@real-domain.com.bad_domain",
            "Leveraged Trader Maintainer ops@real-domain.com.foo_",
            "Leveraged Trader Maintainer ops@project.xn--",
            "Leveraged Trader Maintainer ops@project.xn---p1ai",
            "Leveraged Trader Maintainer ops@project.xn--p1ai-",
            f"Leveraged Trader Maintainer ops@project.xn--{'a' * 60}",
            "Leveraged Trader Maintainer ops@project.xn--a",
            "Leveraged Trader Maintainer ops@project.xn--0",
            "Leveraged Trader Maintainer ops@project.xn--abc",
        )

        for user_agent in malformed_contacts:
            with (
                self.subTest(user_agent=user_agent),
                patch.dict("os.environ", {"SEC_USER_AGENT": user_agent}),
            ):
                result = _fetch_source_text(source, 30)

            self.assertIsNone(result.text)
            self.assertIn("SEC_USER_AGENT", result.error)

        mock_get.assert_not_called()

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_request_accepts_valid_contact_email_syntax(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="ok")
        response.raise_for_status.return_value = None
        mock_get.return_value = response
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov/files/example.json",
            source_type="filing_audit",
        )
        valid_contacts = (
            "Leveraged Trader Maintainer sec-ops@leveraged-trader.dev",
            "Leveraged Trader Maintainer sec.ops+edgar@sub.leveraged-trader.dev",
            "Leveraged Trader Maintainer sec-ops@leveraged-trader.dev.",
            "EDGAR Monitor/1.0 mailto:ops@real-domain.com",
            "Northstar Project email: ops@real-domain.com",
            "Northstar2 ops@real-domain.com",
            "Northstar Project Operators ops@real-domain.com",
            "Example Capital Research ops@real-domain.com",
            "Sample Size Analytics ops@real-domain.com",
            "Test Driven Analytics ops@real-domain.com",
            "Unknown Horizons Research ops@real-domain.com",
            "None Such Analytics ops@real-domain.com",
            "Your Health Analytics ops@real-domain.com",
            "Local Analytics Maintainer ops@real-domain.com",
            "Alternative Research Maintainer ops@public-project.org",
            "Northstar Project <ops@real-domain.com>",
            "Leveraged Trader Maintainer (ops@real-domain.com)",
            "https://github.com/maks-emelyanov/leveraged-trader ops@real-domain.com",
            "https://0x7f.project.dev/leveraged-trader ops@real-domain.com",
            "https://cafe.babe.dev/leveraged-trader ops@real-domain.com",
            "Northstar Maintainer ops@project.xn--p1ai",
            "Northstar Maintainer ops@project.XN--P1AI",
            "Northstar Maintainer ops@project.xn--zca",
            "Northstar Maintainer ops@project.xn--fa-hia",
        )

        for user_agent in valid_contacts:
            with (
                self.subTest(user_agent=user_agent),
                patch.dict("os.environ", {"SEC_USER_AGENT": user_agent}),
            ):
                result = _fetch_source_text(source, 30)

            self.assertEqual(result.text, "ok")
            self.assertEqual(mock_get.call_args.kwargs["headers"]["User-Agent"], user_agent)

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_user_agent_is_preserved_across_normalized_same_origin_redirect(self, mock_get: Mock) -> None:
        redirect = Mock(status_code=302, headers={"Location": "https://WWW.SEC.GOV:443/files/data.json"})
        final = Mock(status_code=200, text="ok")
        final.raise_for_status.return_value = None
        mock_get.side_effect = [redirect, final]
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov./files/example.json",
            source_type="filing_audit",
        )

        sec_user_agent = "Leveraged Trader Maintainer sec-ops@leveraged-trader.dev"
        with patch.dict("os.environ", {"SEC_USER_AGENT": sec_user_agent}):
            result = _fetch_source_text(source, 30)

        self.assertEqual(result.text, "ok")
        self.assertEqual(
            [call.kwargs["headers"]["User-Agent"] for call in mock_get.call_args_list],
            [sec_user_agent] * 2,
        )
        self.assertTrue(all(call.kwargs["allow_redirects"] is False for call in mock_get.call_args_list))
        redirect.close.assert_called_once()
        final.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_cross_origin_redirect_is_refused_before_contact_header_can_escape(self, mock_get: Mock) -> None:
        redirect = Mock(status_code=302, headers={"Location": "https://provider.example/data"})
        mock_get.return_value = redirect
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov/files/example.json",
            source_type="filing_audit",
        )

        sec_user_agent = "Leveraged Trader Maintainer sec-ops@leveraged-trader.dev"
        with patch.dict("os.environ", {"SEC_USER_AGENT": sec_user_agent}):
            result = _fetch_source_text(source, 30)

        self.assertIsNone(result.text)
        self.assertIn("cross-origin redirect", result.error)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(mock_get.call_args.kwargs["headers"]["User-Agent"], sec_user_agent)
        redirect.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_canonical_trailing_dot_sec_host_receives_sec_user_agent(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="ok")
        response.raise_for_status.return_value = None
        mock_get.return_value = response
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov./files/example.json",
            source_type="filing_audit",
        )
        sec_user_agent = "Leveraged Trader Maintainer sec-ops@leveraged-trader.dev"

        with patch.dict("os.environ", {"SEC_USER_AGENT": sec_user_agent}):
            result = _fetch_source_text(source, 30)

        self.assertEqual(result.text, "ok")
        self.assertEqual(mock_get.call_args.kwargs["headers"]["User-Agent"], sec_user_agent)
        self.assertEqual(mock_get.call_args.args[0], source.url)

    @patch("leveraged_trader.universe.requests.get")
    def test_ambiguous_backslash_authority_is_rejected_before_sec_header_routing(self, mock_get: Mock) -> None:
        source = UniverseSource(
            "Ambiguous host",
            r"https://evil.example\@www.sec.gov/files/example.json",
            source_type="filing_audit",
        )
        sec_user_agent = "Leveraged Trader Maintainer sec-ops@leveraged-trader.dev"

        with patch.dict("os.environ", {"SEC_USER_AGENT": sec_user_agent}):
            result = _fetch_source_text(source, 30)

        self.assertIsNone(result.text)
        self.assertIn("invalid or ambiguous authority", result.error)
        self.assertNotIn(sec_user_agent, result.error)
        mock_get.assert_not_called()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_source_rejects_url_userinfo_before_request(self, mock_get: Mock) -> None:
        unsafe_urls = (
            "https://operator:TOPSECRET123@provider.example/path",
            "https://operator:TOP%53ECRET%40123@provider.example/path",
            "https://operator@provider.example/path",
        )

        for unsafe_url in unsafe_urls:
            with self.subTest(unsafe_url=unsafe_url):
                result = _fetch_source_text(
                    UniverseSource("Unsafe source", unsafe_url, source_type="issuer_etf"),
                    30,
                )

                self.assertIsNone(result.text)
                self.assertIn("must not contain URL userinfo", result.error)
                self.assertNotIn("TOPSECRET", result.error)

        mock_get.assert_not_called()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_redirect_rejects_url_userinfo_before_followup_request(self, mock_get: Mock) -> None:
        unsafe_locations = (
            "https://operator:TOPSECRET123@provider.example/path",
            "https://operator:TOP%53ECRET%40123@provider.example/path",
            "//operator@provider.example/path",
        )

        for unsafe_location in unsafe_locations:
            with self.subTest(unsafe_location=unsafe_location):
                redirect = Mock(status_code=302, headers={"Location": unsafe_location})
                mock_get.reset_mock()
                mock_get.return_value = redirect

                result = _fetch_source_text(
                    UniverseSource("Safe source", "https://issuer.example/start", source_type="issuer_etf"),
                    30,
                )

                self.assertIsNone(result.text)
                self.assertIn("redirect target must not contain URL userinfo", result.error)
                self.assertNotIn("TOPSECRET", result.error)
                self.assertEqual(mock_get.call_count, 1)
                redirect.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_source_rejects_https_redirect_downgrade(self, mock_get: Mock) -> None:
        redirect = Mock(status_code=302, headers={"Location": "http://provider.example/data"})
        mock_get.return_value = redirect
        source = UniverseSource(
            "Secure source",
            "https://provider.example/start",
            source_type="issuer_etf",
        )

        result = _fetch_source_text(source, 30)

        self.assertIsNone(result.text)
        self.assertIn("HTTPS-to-HTTP redirect downgrade", result.error)
        self.assertEqual(mock_get.call_count, 1)
        redirect.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_source_rejects_unsafe_redirect_scheme(self, mock_get: Mock) -> None:
        redirect = Mock(status_code=302, headers={"Location": "file:///tmp/universe.csv"})
        mock_get.return_value = redirect
        source = UniverseSource(
            "Secure source",
            "https://provider.example/start",
            source_type="issuer_etf",
        )

        result = _fetch_source_text(source, 30)

        self.assertIsNone(result.text)
        self.assertIn("redirect target must use HTTP or HTTPS", result.error)
        self.assertEqual(mock_get.call_count, 1)
        redirect.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_redirect_rejects_cross_origin_literal_addresses(self, mock_get: Mock) -> None:
        unsafe_locations = (
            "https://127.0.0.1/admin",
            "https://0.0.0.0/admin",
            "https://10.0.0.7/admin",
            "https://169.254.169.254/latest/meta-data/",
            "https://[::1]/admin",
            "https://[::ffff:127.0.0.1]/admin",
            "https://[fe80::1]/admin",
            "https://[fc00::1]/admin",
            "https://224.0.0.1/admin",
        )

        for unsafe_location in unsafe_locations:
            with self.subTest(unsafe_location=unsafe_location):
                redirect = Mock(status_code=302, headers={"Location": unsafe_location})
                mock_get.reset_mock()
                mock_get.return_value = redirect

                result = _fetch_source_text(
                    UniverseSource("Safe source", "https://issuer.example/start", source_type="issuer_etf"),
                    30,
                )

                self.assertIsNone(result.text)
                self.assertIn("cross-origin redirect", result.error)
                self.assertEqual(mock_get.call_count, 1)
                redirect.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_redirect_rejects_cross_origin_host_and_effective_port(self, mock_get: Mock) -> None:
        unsafe_locations = (
            "https://redirect.example/data",
            "https://issuer.example.evil/data",
            "https://issuer.example:444/data",
        )

        for unsafe_location in unsafe_locations:
            with self.subTest(unsafe_location=unsafe_location):
                redirect = Mock(status_code=302, headers={"Location": unsafe_location})
                mock_get.reset_mock()
                mock_get.return_value = redirect

                result = _fetch_source_text(
                    UniverseSource("Safe source", "https://issuer.example/start", source_type="issuer_etf"),
                    30,
                )

                self.assertIsNone(result.text)
                self.assertIn("cross-origin redirect", result.error)
                self.assertEqual(mock_get.call_count, 1)
                redirect.close.assert_called_once()

    @patch("leveraged_trader.universe.requests.get")
    def test_universe_redirect_rejects_http_to_https_origin_change(self, mock_get: Mock) -> None:
        redirect = Mock(status_code=302, headers={"Location": "https://issuer.example/data"})
        mock_get.return_value = redirect

        result = _fetch_source_text(
            UniverseSource("Safe source", "http://issuer.example/start", source_type="issuer_etf"),
            30,
        )

        self.assertIsNone(result.text)
        self.assertIn("cross-origin redirect", result.error)
        self.assertEqual(mock_get.call_count, 1)
        redirect.close.assert_called_once()

    @patch("leveraged_trader.universe.load_current_etf_universe")
    def test_invalid_workflow_universe_limit_fails_before_discovery(self, mock_load_current: Mock) -> None:
        for invalid_limit in [0, -1, True, 1.5]:
            with (
                self.subTest(top_n=invalid_limit),
                self.assertRaisesRegex(ValueError, "positive integer or None"),
            ):
                determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite", top_n=invalid_limit))

        mock_load_current.assert_not_called()

    def test_positive_workflow_universe_limit_preserves_workflow_metadata(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "UPRO", "name": "ProShares UltraPro S&P500", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"TQQQ", "UPRO", "QQQ", "SPY"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite", top_n=1))

        self.assertEqual(len(workflow_assets), 1)
        self.assertIn("rsi_symbol", workflow_assets.columns)

    def test_workflow_asset_groups_include_short_inverse_products(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "SQQQ", "name": "ProShares UltraPro Short QQQ", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"TQQQ", "SQQQ", "QQQ"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_asset_groups = determine_workflow_asset_groups(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_asset_groups["long"]["symbol"].tolist(), ["TQQQ"])
        self.assertEqual(workflow_asset_groups["short"]["symbol"].tolist(), ["SQQQ"])
        self.assertEqual(workflow_asset_groups["short"].loc[0, "rsi_symbol"], "QQQ")
        self.assertEqual(workflow_asset_groups["short"].loc[0, "direction"], "inverse")

    def test_workflow_asset_groups_allow_short_only_universe(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                {"symbol": "SQQQ", "name": "ProShares UltraPro Short QQQ", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"SQQQ", "QQQ"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_asset_groups = determine_workflow_asset_groups(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertTrue(workflow_asset_groups["long"].empty)
        self.assertEqual(workflow_asset_groups["short"]["symbol"].tolist(), ["SQQQ"])
        self.assertEqual(
            workflow_asset_groups["short"].attrs["universe_counts"]["Executable short leveraged ETFs/ETNs selected"],
            1,
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"SQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
            self.assertRaisesRegex(RuntimeError, "no executable long leveraged"),
        ):
            determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_status_marks_partial_download_non_authoritative(
        self,
        mock_read: Mock,
    ) -> None:
        mock_read.side_effect = [
            pd.DataFrame({"Symbol": ["TQQQ"], "Test Issue": ["N"]}),
            RuntimeError("otherlisted unavailable"),
        ]

        active_symbols = load_active_listed_symbols()

        self.assertEqual(active_symbols, {"TQQQ"})
        self.assertFalse(active_symbols.is_complete)
        self.assertEqual(active_symbols.source_status[1]["status"], "error")

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_header_only_active_listing_source_is_non_authoritative(
        self,
        mock_read: Mock,
    ) -> None:
        mock_read.side_effect = [
            pd.DataFrame(columns=["Symbol", "Test Issue"]),
            pd.DataFrame({"ACT Symbol": ["TQQQ"], "Test Issue": ["N"]}),
        ]

        active_symbols = load_active_listed_symbols()

        self.assertEqual(active_symbols, {"TQQQ"})
        self.assertFalse(active_symbols.is_complete)
        self.assertEqual(active_symbols.source_status[0]["status"], "error")
        self.assertIn("no usable listed symbols", active_symbols.source_status[0]["error"])

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_excludes_nasdaq_test_issues(self, mock_read: Mock) -> None:
        mock_read.side_effect = [
            pd.DataFrame(
                {
                    "Symbol": ["TQQQ", "ZTEST"],
                    "Test Issue": ["N", "Y"],
                }
            ),
            pd.DataFrame(
                {
                    "ACT Symbol": ["SPY", "ATEST"],
                    "Test Issue": ["N", "Y"],
                }
            ),
        ]

        active_symbols = load_active_listed_symbols()

        self.assertEqual(active_symbols, {"SPY", "TQQQ"})
        self.assertTrue(active_symbols.is_complete)

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_rejects_invalid_or_missing_symbol_rows(self, mock_read: Mock) -> None:
        for bad_symbol in ("???", ""):
            with self.subTest(bad_symbol=bad_symbol):
                mock_read.side_effect = [
                    pd.DataFrame({"Symbol": ["TQQQ", bad_symbol], "Test Issue": ["N", "N"]}),
                    pd.DataFrame({"ACT Symbol": ["SPY"], "Test Issue": ["N"]}),
                ]

                active_symbols = load_active_listed_symbols()

                self.assertEqual(active_symbols, {"SPY"})
                self.assertFalse(active_symbols.is_complete)
                self.assertEqual(active_symbols.source_status[0]["status"], "error")
                self.assertIn("missing or invalid", active_symbols.source_status[0]["error"])

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_accepts_documented_non_yahoo_otherlisted_symbols(self, mock_read: Mock) -> None:
        mock_read.side_effect = [
            pd.DataFrame({"Symbol": ["TQQQ"], "Test Issue": ["N"]}),
            pd.DataFrame(
                {
                    "ACT Symbol": ["SPY", "ABR$D", "AAC.U", "DCOM$", "NA"],
                    "Test Issue": ["N", "N", "N", "N", "N"],
                }
            ),
        ]

        active_symbols = load_active_listed_symbols()

        self.assertEqual(active_symbols, {"AAC.U", "NA", "SPY", "TQQQ"})
        self.assertTrue(active_symbols.is_complete)

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_rejects_empty_or_repeated_source_suffixes(self, mock_read: Mock) -> None:
        for bad_symbol in ("TQQQ..", "A$$", "A$.", "A.", "A..B", "A.$"):
            with self.subTest(bad_symbol=bad_symbol):
                mock_read.side_effect = [
                    pd.DataFrame({"Symbol": ["TQQQ", bad_symbol], "Test Issue": ["N", "N"]}),
                    pd.DataFrame({"ACT Symbol": ["SPY"], "Test Issue": ["N"]}),
                ]

                active_symbols = load_active_listed_symbols()

                self.assertEqual(active_symbols, {"SPY"})
                self.assertFalse(active_symbols.is_complete)
                self.assertEqual(active_symbols.source_status[0]["status"], "error")
                self.assertIn("missing or invalid", active_symbols.source_status[0]["error"])

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_rejects_duplicate_normalized_symbol_rows(self, mock_read: Mock) -> None:
        for test_issues, expected_error in [
            (["N", "Y"], "contradictory 'Test Issue'"),
            (["N", "N"], "duplicate normalized symbols"),
        ]:
            with self.subTest(test_issues=test_issues):
                mock_read.side_effect = [
                    pd.DataFrame({"Symbol": ["BRKB", "BRK.B"], "Test Issue": test_issues}),
                    pd.DataFrame({"ACT Symbol": ["SPY"], "Test Issue": ["N"]}),
                ]

                active_symbols = load_active_listed_symbols()

                self.assertEqual(active_symbols, {"SPY"})
                self.assertFalse(active_symbols.is_complete)
                self.assertIn(expected_error, active_symbols.source_status[0]["error"])

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_fails_closed_without_test_issue_schema(self, mock_read: Mock) -> None:
        mock_read.side_effect = [
            pd.DataFrame({"Symbol": ["TQQQ"]}),
            pd.DataFrame({"ACT Symbol": ["SPY"], "Test Issue": ["N"]}),
        ]

        active_symbols = load_active_listed_symbols()

        self.assertEqual(active_symbols, {"SPY"})
        self.assertFalse(active_symbols.is_complete)
        self.assertEqual(active_symbols.source_status[0]["status"], "error")
        self.assertIn("exactly one 'Test Issue'", active_symbols.source_status[0]["error"])

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_fails_closed_on_ambiguous_test_issue_schema(self, mock_read: Mock) -> None:
        ambiguous = pd.DataFrame(
            [["TQQQ", "N", "Y"]],
            columns=["Symbol", "Test Issue", "Test Issue"],
        )
        mock_read.side_effect = [
            ambiguous,
            pd.DataFrame({"ACT Symbol": ["SPY"], "Test Issue": ["N"]}),
        ]

        active_symbols = load_active_listed_symbols()

        self.assertEqual(active_symbols, {"SPY"})
        self.assertFalse(active_symbols.is_complete)
        self.assertEqual(active_symbols.source_status[0]["status"], "error")
        self.assertIn("found 2", active_symbols.source_status[0]["error"])

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_fails_closed_on_unknown_test_issue_value(self, mock_read: Mock) -> None:
        for invalid_value in ("unknown", "n", " y "):
            with self.subTest(invalid_value=invalid_value):
                mock_read.side_effect = [
                    pd.DataFrame({"Symbol": ["TQQQ"], "Test Issue": [invalid_value]}),
                    pd.DataFrame({"ACT Symbol": ["SPY"], "Test Issue": ["N"]}),
                ]

                active_symbols = load_active_listed_symbols()

                self.assertEqual(active_symbols, {"SPY"})
                self.assertFalse(active_symbols.is_complete)
                self.assertEqual(active_symbols.source_status[0]["status"], "error")
                self.assertIn("only 'Y' or 'N'", active_symbols.source_status[0]["error"])

    def test_nasdaq_symbol_file_rejects_duplicate_raw_headers_before_pandas_mangles_them(self) -> None:
        for header in (
            "Symbol|Symbol|Test Issue",
            "Symbol|Test Issue|Test Issue",
        ):
            response = Mock(status_code=200, text=f"{header}\nTQQQ|TQQQ|N\n")
            with (
                self.subTest(header=header),
                patch(
                    "leveraged_trader.universe._get_universe_response",
                    return_value=response,
                ),
                self.assertRaisesRegex(ValueError, "complete expected schema"),
            ):
                _read_nasdaq_symbol_file(NASDAQ_LISTED_URL, 30)

    def test_nasdaq_symbol_file_preserves_na_ticker_text(self) -> None:
        header = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
        rows = [f"SYM{row}|Security {row}|Q|N|N|100|N|N" for row in range(5_399)]
        rows.append("NA|Security NA|Q|N|N|100|N|N")
        response = Mock(status_code=200, text="\n".join([header, *rows, "File Creation Time: 0101202612:00|||||||"]))
        response.raise_for_status.return_value = None
        with (
            patch("leveraged_trader.universe._get_universe_response", return_value=response),
            patch(
                "leveraged_trader.universe._nasdaq_directory_now",
                return_value=datetime(2026, 1, 1, 12, 0, tzinfo=ZoneInfo("America/New_York")),
            ),
        ):
            listed = _read_nasdaq_symbol_file(NASDAQ_LISTED_URL, 30)

        self.assertEqual(listed.loc[5_399, "Symbol"], "NA")

    def test_nasdaq_symbol_file_rejects_stale_or_future_snapshot_timestamps(self) -> None:
        header = "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
        rows = [f"SYM{row}|Security {row}|Q|N|N|100|N|N" for row in range(4_000)]
        now = datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo("America/New_York"))
        for label, footer_timestamp, error in (
            ("stale", "0101202612:00", "stale"),
            ("future", "0116202612:01", "future"),
        ):
            response = Mock(
                status_code=200,
                text="\n".join([header, *rows, f"File Creation Time: {footer_timestamp}|||||||"]),
            )
            response.raise_for_status.return_value = None
            with (
                self.subTest(label=label),
                patch("leveraged_trader.universe._get_universe_response", return_value=response),
                patch("leveraged_trader.universe._nasdaq_directory_now", return_value=now),
                self.assertRaisesRegex(ValueError, error),
            ):
                _read_nasdaq_symbol_file(NASDAQ_LISTED_URL, 30)

    @patch("leveraged_trader.universe._read_nasdaq_symbol_file")
    def test_active_listing_rejects_real_source_metadata_with_too_few_usable_symbols(self, mock_read: Mock) -> None:
        nasdaq = pd.DataFrame({"Symbol": [f"N{row}" for row in range(1_000)], "Test Issue": ["N"] * 1_000})
        nasdaq.attrs["minimum_usable_symbols"] = 4_000
        other = pd.DataFrame({"ACT Symbol": [f"O{row}" for row in range(1_000)], "Test Issue": ["N"] * 1_000})
        other.attrs["minimum_usable_symbols"] = 5_000
        mock_read.side_effect = [nasdaq, other]

        active_symbols = load_active_listed_symbols()

        self.assertFalse(active_symbols.is_complete)
        self.assertEqual(active_symbols, set())
        self.assertTrue(all(status["status"] == "error" for status in active_symbols.source_status))
        self.assertTrue(all("implausibly few" in status["error"] for status in active_symbols.source_status))

    def test_nasdaq_symbol_files_require_complete_footer_schema_and_cardinality(self) -> None:
        valid_headers = {
            NASDAQ_LISTED_URL: (
                "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares"
            ),
            OTHER_LISTED_URL: (
                "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol"
            ),
        }
        for url, header in valid_headers.items():
            footer = (
                "File Creation Time: 0101202612:00|||||||"
                if url == NASDAQ_LISTED_URL
                else "File Creation Time: 0101202612:00||||||"
            )
            malformed_bodies = {
                "footerless": f"{header}\nTQQQ|Security|Q|N|N|100|N|N\n",
                "truncated": f"{header}\nTQQQ|Security|Q|N|N|100|N|N\n{footer}\n",
                "trailing": f"{header}\nTQQQ|Security|Q|N|N|100|N|N\n{footer}\nextra\n",
            }
            for label, body in malformed_bodies.items():
                response = Mock(status_code=200, text=body)
                response.raise_for_status.return_value = None
                with (
                    self.subTest(url=url, label=label),
                    patch("leveraged_trader.universe._get_universe_response", return_value=response),
                    self.assertRaises(ValueError),
                ):
                    _read_nasdaq_symbol_file(url, 30)

    def test_nasdaq_metadata_wins_duplicate_issuer_symbol(self) -> None:
        merged = _merge_universe_sources(
            pd.DataFrame(
                [
                    {"symbol": "TQQQ", "name": "Nasdaq Current Name", "fund_type": "ETF"},
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "symbol": "TQQQ",
                        "name": "Issuer Stale Name",
                        "fund_type": "ETF (Issuer)",
                        "source": "Issuer table",
                    },
                ]
            ),
        )

        self.assertEqual(merged.loc[0, "name"], "Nasdaq Current Name")
        self.assertEqual(merged.loc[0, "source"], "Nasdaq ETF definitions")

    def test_nasdaq_winner_retains_validated_issuer_reference_security(self) -> None:
        name = "Tradr 2X Long Innovation Daily ETF"
        merged = _merge_universe_sources(
            pd.DataFrame([{"symbol": "FOOX", "name": name, "fund_type": "ETF"}]),
            pd.DataFrame(
                [
                    {
                        "symbol": "FOOX",
                        "name": name,
                        "fund_type": "ETF (Issuer)",
                        "source": "Tradr",
                        "reference_security": "SPY",
                    }
                ]
            ),
        )

        mapped = build_nasdaq_universe_table(merged, known_symbols={"FOOX", "SPY"})

        self.assertEqual(merged.loc[0, "source"], "Nasdaq ETF definitions")
        self.assertEqual(mapped.loc[0, "reference_security"], "SPY")
        self.assertEqual(mapped.loc[0, "rsi_symbol"], "SPY")
        self.assertEqual(mapped.loc[0, "mapping_source"], "issuer_reference")

    def test_inverse_issuer_self_reference_is_quarantined_instead_of_promoted(self) -> None:
        products = pd.DataFrame(
            [
                {
                    "symbol": "XINV",
                    "name": "Tradr 2X Short Innovation Daily ETF",
                    "fund_type": "ETF (Tradr)",
                    "source": "Tradr issuer table",
                    "reference_security": "XINV",
                }
            ]
        )

        candidates = _workflow_candidates(products, {"XINV"}, workflow_label="Short")

        self.assertEqual(candidates.loc[0, "rsi_symbol"], "XINV")
        self.assertEqual(candidates.loc[0, "mapping_source"], "issuer_reference_conflict")
        self.assertEqual(candidates.loc[0, "confidence"], "needs_review")
        self.assertIn("leveraged product itself", candidates.loc[0, "mapping_reason"])

    def test_direct_merge_excludes_primary_etf_discovered_etn_conflict(self) -> None:
        merged = _merge_universe_sources(
            pd.DataFrame(
                [
                    {"symbol": "SAFE", "name": "QQQ 2X Daily ETF", "fund_type": "ETF"},
                    {"symbol": "XLEV", "name": "NVDA 2X Daily ETF", "fund_type": "ETF"},
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "symbol": "XLEV",
                        "name": "NVDA 2X Daily ETN",
                        "fund_type": "ETN (Single Stock)",
                        "source": "ETN issuer table",
                    }
                ]
            ),
        )

        self.assertEqual(merged["symbol"].tolist(), ["SAFE"])

    def test_infers_underlying_symbol_from_leveraged_name(self) -> None:
        self.assertEqual(infer_rsi_symbol("TQQQ", "ProShares UltraPro QQQ"), "QQQ")

    def test_falls_back_to_asset_symbol_when_no_underlying_is_found(self) -> None:
        self.assertEqual(infer_rsi_symbol("XYZ", "Plain Fund Name"), "XYZ")

    def test_nasdaq_is_not_used_as_inferred_rsi_symbol(self) -> None:
        self.assertEqual(infer_rsi_symbol("QQQU", "Defiance Daily Target 2X Long NASDAQ ETF"), "QQQU")

    def test_direct_generic_inference_uses_safe_default_allowlist(self) -> None:
        mapping = infer_rsi_mapping("ENGY", "Ultra Energy")

        self.assertEqual(mapping.rsi_symbol, "ENGY")
        self.assertEqual(mapping.confidence, "fallback_to_self")
        self.assertEqual(infer_rsi_symbol("TQQQ", "ProShares UltraPro QQQ"), "QQQ")

    def test_inferred_underlying_must_be_known_when_known_symbols_are_provided(self) -> None:
        self.assertEqual(
            infer_rsi_symbol("ENGY", "ProShares Ultra Energy", known_symbols={"ENGY", "QQQ"}),
            "ENGY",
        )
        self.assertEqual(
            infer_rsi_symbol("AAPU", "Direxion Daily AAPL Bull 2X ETF", known_symbols={"AAPU", "AAPL"}),
            "AAPL",
        )
        self.assertEqual(
            infer_rsi_symbol("SPYU", "SPY 4X Daily ETF", known_symbols={"SPYU", "SPY"}),
            "SPY",
        )
        self.assertEqual(
            infer_rsi_symbol("FIVE", "5X Long QQQ Daily ETF", known_symbols={"FIVE", "QQQ"}),
            "QQQ",
        )
        self.assertEqual(
            infer_rsi_symbol("FIVP", "500% Long QQQ Daily ETF", known_symbols={"FIVP", "QQQ"}),
            "QQQ",
        )

    def test_sk_hynix_products_use_us_listed_underlying_rsi(self) -> None:
        for asset_symbol, direction in [
            ("SKHU", "Long"),
            ("SKHX", "Long"),
            ("SKUU", "Long"),
            ("SKDD", "Short"),
        ]:
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(
                    asset_symbol,
                    f"Example 2X {direction} SK Hynix Daily ETF",
                    known_symbols={asset_symbol, "SK"},
                )

                self.assertEqual(mapping.rsi_symbol, "SKHY")
                self.assertEqual(mapping.confidence, "curated")
                self.assertEqual(mapping.mapping_source, "symbol_override")

    def test_new_sk_hynix_product_uses_us_listed_underlying_by_name(self) -> None:
        mapping = infer_rsi_mapping(
            "NEWSK",
            "Example 2X Long SK Hynix Daily ETF",
            known_symbols={"NEWSK", "SK"},
        )

        self.assertEqual(mapping.rsi_symbol, "SKHY")
        self.assertEqual(mapping.confidence, "curated")
        self.assertEqual(mapping.mapping_source, "name_proxy")

    def test_normalizes_brkb_to_yahoo_symbol(self) -> None:
        self.assertEqual(infer_rsi_symbol("BRKU", "2X Long BRKB Daily ETF"), "BRK-B")
        self.assertEqual(infer_rsi_symbol("BRKU", "2X Long BRK.B Daily ETF"), "BRK-B")

    def test_curated_company_name_underlying_mappings(self) -> None:
        cases = {
            "AAPX": ("T-REX 2X Long Apple Daily Target ETF", "AAPL"),
            "ETNG": ("2x Long ETN Daily ETF", "ETN"),
            "GOOX": ("T-REX 2X Long Alphabet Daily Target ETF", "GOOG"),
            "MSFX": ("T-REX 2X Long Microsoft Daily Target ETF", "MSFT"),
            "NVDQ": ("Tradr 2X Short NVDA Daily ETF", "NVDA"),
            "NVDX": ("T-REX 2X Long NVIDIA Daily Target ETF", "NVDA"),
            "TSLZ": ("T-Rex 2X Inverse Tesla Daily Target ETF", "TSLA"),
            "TSLT": ("T-REX 2X Long Tesla Daily Target ETF", "TSLA"),
            "BULG": ("Leverage Shares 2X Long BULL Daily ETF", "BULL"),
            "MST": ("Defiance Leveraged Long + Income MSTR ETF", "MSTR"),
            "MSOX": ("MSOS Daily Leveraged ETF", "MSOS"),
            "SATG": ("Leverage Shares 2X Long SATS Daily ETF", "SATS"),
            "MQQQ": ("Tradr 2X Long Innovation 100 Monthly ETF", "QQQ"),
            "QQQP": ("Tradr 2X Long Innovation 100 Quarterly ETF", "QQQ"),
            "WLDU": ("2x Long World Daily ETF", "VT"),
            "AIQU": ("MicroSectors Artificial Intelligence (AI) 3X Long Exposure ETN", "AIQ"),
            "BDCX": ("ETRACS Quarterly Pay 1.5x Leveraged MarketVector BDC Liquid Index ETN", "BIZD"),
            "BIB": ("ProShares Ultra Nasdaq Biotechnology", "IBB"),
            "BNKU": ("MicroSectors Big Banks 3X Leveraged Exposure ETN", "KBWB"),
            "BULZ": ("MicroSectors Fang & Innovation 3X Leveraged Exposure ETN", "FNGS"),
            "CEFD": ("ETRACS Monthly Pay 1.5X Leveraged Closed-End Fund Index ETN", "CEFS"),
            "DIG": ("Ultra Energy", "XLE"),
            "DRNL": ("Defiance 2X Daily Long Pure Drone and Aerial Automation ETF", "DRNZ"),
            "EET": ("Ultra MSCI Emerging Markets", "EEM"),
            "EFO": ("Ultra MSCI EAFE", "EFA"),
            "EZJ": ("Ultra MSCI Japan", "EWJ"),
            "FDRX": ("Founder-Led 2X Daily ETF", "FDRS"),
            "FLYU": ("MicroSectors Travel 3X Leveraged Exposure ETN", "PEJ"),
            "FNGO": ("MicroSectors Fang+ 2X Leveraged Exposure ETN", "FNGS"),
            "FNGU": ("MicroSectors Fang+ 3X Leveraged Exposure ETN", "FNGS"),
            "HDLB": (
                "ETRACS Monthly Pay 2xLeveraged US High Dividend Low Volatility ETN Series B",
                "SPHD",
            ),
            "IWDL": ("ETRACS 2x Leveraged US Value Factor TR ETN", "IWD"),
            "IWFL": ("ETRACS 2x Leveraged US Growth Factor TR ETN", "IWF"),
            "IWML": ("ETRACS 2x Leveraged US Size Factor TR ETN", "SIZE"),
            "LTL": ("Ultra Communication Services", "XLC"),
            "MAGX": ("Daily 2X Long Magnificent Seven ETF", "MAGS"),
            "MLPR": ("ETRACS Quarterly Pay 1.5x Leveraged Alerian MLP Index ETN", "AMLP"),
            "MTUL": ("ETRACS 2x Leveraged MSCI US Momentum Factor TR ETN", "MTUM"),
            "MVRL": ("ETRACS Monthly Pay 1.5x Leveraged Mortgage REIT ETN", "REM"),
            "MVV": ("Ultra MidCap400", "MDY"),
            "NRGU": ("MicroSectors Big Oil 3X Leveraged Exposure ETN", "XLE"),
            "OILU": ("MicroSectors Oil & Gas Exploration & Production 3X Leveraged Exposure ETN", "XOP"),
            "PFFL": ("ETRACS Monthly Pay 2xLeveraged Preferred Stock ETN", "PFF"),
            "QPUX": ("Defiance 2X Daily Long Pure Quantum ETF", "QTUM"),
            "QULL": ("ETRACS 2x Leveraged MSCI US Quality Factor TR ETN", "QUAL"),
            "ROM": ("Ultra Technology", "XLK"),
            "RXL": ("Ultra Health Care", "XLV"),
            "SAA": ("Ultra SmallCap600", "IJR"),
            "SCDL": ("ETRACS 2x Leveraged US Dividend Factor TR ETN", "SCHD"),
            "SKYU": ("ProShares Ultra Cloud Computing", "SKYY"),
            "SMHB": (
                "ETRACS Monthly Pay 2xLeveraged US Small Cap High Dividend ETN Series B",
                "DES",
            ),
            "SPCL": ("Defiance Daily 2X Space ETF", "UFO"),
            "TARK": ("Tradr 2X Long Innovation ETF", "ARKK"),
            "UCC": ("Ultra Consumer Discretionary", "XLY"),
            "UCYB": ("ProShares Ultra Nasdaq Cybersecurity", "CIBR"),
            "UBR": ("Ultra MSCI Brazil Capped", "EWZ"),
            "UGE": ("Ultra Consumer Staples", "XLP"),
            "UJB": ("Ultra High Yield", "HYG"),
            "UMDD": ("UltraPro MidCap 400", "MDY"),
            "UPV": ("Ultra FTSE Europe", "VGK"),
            "UPW": ("Ultra Utilities", "XLU"),
            "URE": ("Ultra Real Estate", "IYR"),
            "USD": ("Ultra Semiconductors", "SOXX"),
            "USML": ("ETRACS 2x Leveraged MSCI US Minimum Volatility Factor TR ETN", "USMV"),
            "UVIX": ("2x Long VIX Futures ETF", "VIXY"),
            "UXI": ("Ultra Industrials", "XLI"),
            "UXRP": ("Ultra XRP ETF", "XRP-USD"),
            "UYG": ("Ultra Financials", "XLF"),
            "UYM": ("Ultra Materials", "XLB"),
            "XPP": ("Ultra FTSE China 50", "FXI"),
            "XRPT": ("Volatility Shares Trust XRP 2X ETF", "XRP-USD"),
            "BOIL": ("Ultra Bloomberg Natural Gas", "UNG"),
            "COPZ": ("Defiance Daily Target 2X Long Copper Miners ETF", "COPX"),
            "UCO": ("Ultra Bloomberg Crude Oil", "USO"),
            "UCOP": ("Ultra Copper K-1 Free ETF", "CPER"),
            "ULE": ("Ultra Euro", "FXE"),
            "UPAL": ("Ultra Palladium K-1 Free ETF", "PALL"),
            "UPLT": ("Ultra Platinum K-1 Free ETF", "PPLT"),
            "WTIU": ("MicroSectors Energy 3X Leveraged Exposure ETN", "XLE"),
            "YCL": ("Ultra Yen", "FXY"),
            "AVAZ": ("2x Avalanche ETF", "AVAX-USD"),
            "CHNU": ("2x Chainlink ETF", "LINK-USD"),
            "CRDX": ("2x Cardano ETF", "ADA-USD"),
            "STLU": ("2x Stellar ETF", "XLM-USD"),
            "SUIL": ("2x Sui ETF", "SUI20947-USD"),
            "TXXD": ("21Shares 2x Long Dogecoin ETF", "DOGE-USD"),
            "TXXH": ("21Shares 2x Long HYPE ETF", "HYPE32196-USD"),
        }

        for asset_symbol, (name, expected) in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name)
                self.assertEqual(mapping.rsi_symbol, expected)
                self.assertEqual(mapping.confidence, "curated")
                self.assertEqual(mapping.mapping_source, "symbol_override")

    def test_curated_benchmark_and_asset_proxy_mappings_win_before_regex(self) -> None:
        cases = {
            "UPRO": ("ProShares UltraPro S&P500", "SPY"),
            "SSO": ("Ultra S&P500", "SPY"),
            "URSP": ("Ultra S&P 500 Equal Weight", "RSP"),
            "UGL": ("Ultra Gold", "GLD"),
            "BITX": ("2x Bitcoin ETF", "BTC-USD"),
            "DOGU": ("2x Dogecoin ETF", "DOGE-USD"),
            "AVAU": ("2x Avalanche ETF", "AVAX-USD"),
            "LNKU": ("2x Chainlink ETF", "LINK-USD"),
            "CARD": ("2x Cardano ETF", "ADA-USD"),
            "ETHU": ("2x Ether ETF", "ETH-USD"),
            "SOLT": ("2x Solana ETF", "SOL-USD"),
            "XRPU": ("2x XRP ETF", "XRP-USD"),
            "SUIX": ("2x Sui ETF", "SUI20947-USD"),
            "XLMU": ("2x Stellar ETF", "XLM-USD"),
            "HYPX": ("2x Hyperliquid ETF", "HYPE32196-USD"),
        }

        for asset_symbol, (name, expected) in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name)
                self.assertEqual(mapping.rsi_symbol, expected)
                self.assertEqual(mapping.confidence, "curated")

    def test_reviewed_inverse_products_use_unlevered_rsi_proxies(self) -> None:
        expected = {
            "AIQD": "AIQ",
            "BERZ": "FNGS",
            "BIS": "IBB",
            "BNKD": "KBWB",
            "BZQ": "EWZ",
            "DUG": "XLE",
            "EEV": "EEM",
            "EFU": "EFA",
            "EPV": "VGK",
            "EUO": "FXE",
            "EWV": "EWJ",
            "FLYD": "PEJ",
            "FNGD": "FNGS",
            "FXP": "FXI",
            "KOLD": "UNG",
            "MZZ": "MDY",
            "NRGD": "XLE",
            "OILD": "XOP",
            "QID": "QQQ",
            "QQDN": "QQQ",
            "REW": "XLK",
            "RXD": "XLV",
            "SCC": "XLY",
            "SCO": "USO",
            "SDD": "IJR",
            "SDP": "XLU",
            "SIJ": "XLI",
            "SKF": "XLF",
            "SKRE": "KRE",
            "SMDD": "MDY",
            "SMN": "XLB",
            "SRS": "IYR",
            "SSG": "SOXX",
            "SZK": "XLP",
            "WTID": "XLE",
            "YCS": "FXY",
        }

        for asset_symbol, rsi_symbol in expected.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, "Inverse leveraged product")
                self.assertEqual(mapping.rsi_symbol, rsi_symbol)
                self.assertEqual(mapping.confidence, "curated")

    def test_ultrashort_duration_and_unstable_basket_products_are_excluded(self) -> None:
        self.assertTrue({"AMUN", "RBIL", "SGVA", "SLTY", "UYLD", "VGUS", "ZMUN"} <= EXCLUDED_UNIVERSE_SYMBOLS)

    def test_common_words_that_are_tickers_do_not_win_generic_rsi_inference(self) -> None:
        cases = {
            "PAYT": "ETRACS Quarterly Pay 1.5x Leveraged Example Index ETN",
            "SERB": "ETRACS High Dividend ETN Series B 2X Leveraged",
            "MSCT": "Ultra MSCI Example Index",
            "HIGHX": "Ultra High Yield",
            "REALX": "Ultra Real Estate",
        }
        known_symbols = {"PAYT", "SERB", "MSCT", "HIGHX", "REALX", "PAY", "B", "MSCI", "HIGH", "REAL"}

        for asset_symbol, name in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name, known_symbols=known_symbols)
                self.assertEqual(mapping.rsi_symbol, asset_symbol)
                self.assertEqual(mapping.confidence, "fallback_to_self")

    def test_resolved_basket_products_use_explicit_rsi_mappings(self) -> None:
        cases = {
            "BEGS": (
                "Rareview 2X Bull Cryptocurrency & Precious Metals ETF",
                "BEGS",
                "fallback_to_self",
                "self_fallback_override",
            ),
            "DRNL": (
                "Defiance 2X Daily Long Pure Drone and Aerial Automation ETF",
                "DRNZ",
                "curated",
                "symbol_override",
            ),
            "FDRX": (
                "Founder-Led 2X Daily ETF",
                "FDRS",
                "curated",
                "symbol_override",
            ),
            "FLYU": (
                "MicroSectors Travel 3X Leveraged Exposure ETN",
                "PEJ",
                "curated",
                "symbol_override",
            ),
        }

        for asset_symbol, (name, expected, confidence, mapping_source) in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name, known_symbols={asset_symbol})
                self.assertEqual(mapping.rsi_symbol, expected)
                self.assertEqual(mapping.confidence, confidence)
                self.assertEqual(mapping.mapping_source, mapping_source)

    def test_unresolved_single_stock_style_mapping_needs_review(self) -> None:
        cases = [
            (
                "FOOU",
                "T-REX 2X Long ExampleCorp Daily Target ETF",
                "ETF (Tuttle Capital)",
            ),
            (
                "FOUR",
                "4X Long ExampleCorp Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "FRAC",
                "1.25X Long ExampleCorp Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "P150",
                "150% Long ExampleCorp Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "TXGU",
                "10x Genomics 2X Long Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "AAPU",
                "Direxion Daily AAPL Bull 2X ETF",
                "ETF (Direxion)",
            ),
            (
                "FIVE",
                "Direxion Daily NVDA Bull 5X Shares",
                "ETF (Direxion)",
            ),
            (
                "FOOD",
                "T-REX 2X Inverse ExampleCorp Daily Target ETF",
                "ETF (Example Issuer)",
            ),
            (
                "FOOS",
                "2X Short ExampleCorp Daily ETF",
                "ETF (Example Issuer)",
            ),
            (
                "FOOB",
                "Direxion Daily AAPL Bear 2X Shares",
                "ETF (Direxion)",
            ),
        ]

        for asset_symbol, name, fund_type in cases:
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(
                    asset_symbol,
                    name,
                    known_symbols={asset_symbol},
                    fund_type=fund_type,
                )

                self.assertEqual(mapping.rsi_symbol, asset_symbol)
                self.assertEqual(mapping.confidence, "needs_review")
                self.assertEqual(mapping.mapping_source, "unresolved_single_stock")

    def test_spacex_products_map_to_spcx_before_regex(self) -> None:
        cases = {
            "SPAL": ("GraniteShares 2x Long SpaceX Daily ETF", "ETF (GraniteShares)"),
            "SPAX": ("T-REX 2X Long SpaceX Daily Target ETF", "ETF (Tuttle Capital)"),
            "SPCF": ("Ultra SpaceX", "ETF (ProShares)"),
            "SPCM": ("Tradr 2X Long SpaceX Daily ETF", "ETF (Tradr)"),
        }

        for asset_symbol, (name, fund_type) in cases.items():
            with self.subTest(asset_symbol=asset_symbol):
                mapping = infer_rsi_mapping(asset_symbol, name, fund_type=fund_type)
                self.assertEqual(mapping.rsi_symbol, "SPCX")
                self.assertEqual(mapping.underlying_name, "Space Exploration Technologies Corp. Class A")
                self.assertEqual(mapping.confidence, "curated")
                self.assertEqual(mapping.mapping_source, "name_proxy")

    def test_saved_universe_table_maps_spacex_products_to_spcx(self) -> None:
        universe = build_nasdaq_universe_table(
            pd.DataFrame(
                [
                    {
                        "symbol": "SPAL",
                        "name": "GraniteShares 2x Long SpaceX Daily ETF",
                        "fund_type": "ETF (GraniteShares)",
                        "source": "GraniteShares issuer table",
                    },
                    {
                        "symbol": "TQQQ",
                        "name": "ProShares UltraPro QQQ",
                        "fund_type": "ETF",
                        "source": "Nasdaq ETF definitions",
                    },
                ]
            )
        )

        spal = universe.loc[universe["symbol"] == "SPAL"].iloc[0]
        self.assertEqual(spal["rsi_symbol"], "SPCX")
        self.assertEqual(spal["underlying_name"], "Space Exploration Technologies Corp. Class A")
        self.assertEqual(spal["confidence"], "curated")
        self.assertNotIn("SPACEX", universe["rsi_symbol"].tolist())

    def test_saved_universe_table_preserves_mapping_schema_when_empty(self) -> None:
        universe = build_nasdaq_universe_table(pd.DataFrame(columns=["symbol", "name", "fund_type", "source"]))

        self.assertTrue(universe.empty)
        self.assertTrue(universe.columns.is_unique)
        self.assertEqual(
            universe.columns.tolist(),
            [
                "symbol",
                "name",
                "fund_type",
                "source",
                "is_long_leveraged",
                "is_short_leveraged",
                "is_single_stock",
                "is_single_stock_long_leveraged",
                "rsi_symbol",
                "underlying_symbol",
                "underlying_name",
                "mapping_source",
                "confidence",
                "mapping_reason",
            ],
        )

    def test_saved_universe_table_validates_generic_rsi_mappings(self) -> None:
        universe = build_nasdaq_universe_table(
            pd.DataFrame(
                [
                    {"symbol": "ENGY", "name": "Ultra Energy", "fund_type": "ETF"},
                    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                    {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                ]
            )
        )

        dig = universe.loc[universe["symbol"] == "ENGY"].iloc[0]
        tqqq = universe.loc[universe["symbol"] == "TQQQ"].iloc[0]
        self.assertEqual(dig["rsi_symbol"], "ENGY")
        self.assertEqual(dig["confidence"], "fallback_to_self")
        self.assertEqual(tqqq["rsi_symbol"], "QQQ")
        self.assertNotIn("ENERGY", universe["rsi_symbol"].tolist())

    def test_long_duration_names_are_not_leveraged(self) -> None:
        false_positive_names = [
            "Vanguard Long-Term Corporate Bond ETF",
            "Baillie Gifford Long Term Global Growth ETF",
            "Innovator U.S. Equity Ultra Buffer ETF",
            "ProShares UltraShort Term Bond ETF",
            "ProShares Ultra Short-Term Bond ETF",
            "ProShares Ultra-Short-Term Bond ETF",
            "ProShares UltraShort-Term Bond ETF",
            "Franklin Ultra Short Bond ETF",
            "Acme Ultra Short Treasury ETF",
            "Example Ultra-Short Fixed-Income ETF",
            "Example Ultrashort Treasury ETF",
            "YieldMax Ultra Option Income Strategy ETF",
        ]
        for name in false_positive_names:
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (None, None))
                self.assertFalse(leveraged_name_filter(name))
                self.assertFalse(is_long_leveraged_name(name))

        self.assertFalse(is_long_leveraged_name("MicroSectors FANG+ 1X Long Exposure ETN"))
        self.assertTrue(is_long_leveraged_name("GraniteShares 2x Long NVDA Daily ETF"))

        branded_inverse = "ProShares UltraShort 20+ Year Treasury ETF"
        self.assertEqual(infer_leverage_and_direction(branded_inverse), (2.0, "inverse"))
        self.assertTrue(leveraged_name_filter(branded_inverse))
        self.assertTrue(is_short_leveraged_name(branded_inverse))

        explicit_inverse = "Example 2X Ultra Short Treasury ETF"
        self.assertEqual(infer_leverage_and_direction(explicit_inverse), (2.0, "inverse"))
        self.assertTrue(leveraged_name_filter(explicit_inverse))
        self.assertTrue(is_short_leveraged_name(explicit_inverse))

    def test_explicit_leveraged_duration_names_are_not_suppressed(self) -> None:
        cases = [
            "Example 2X Long Long-Term Treasury ETF",
            "Example 2X Long-Term Treasury Bull ETF",
            "Example 2X Long Municipal Bond ETF",
        ]

        for name in cases:
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (2.0, "long"))
                self.assertTrue(leveraged_name_filter(name))
                self.assertTrue(is_long_leveraged_name(name))

    def test_short_term_underlying_names_are_not_inverse(self) -> None:
        long_vix = "ProShares Ultra VIX Short-Term Futures ETF"

        self.assertEqual(infer_leverage_and_direction(long_vix), (2.0, "long"))
        self.assertTrue(is_long_leveraged_name(long_vix))

        self.assertEqual(
            infer_leverage_and_direction("ProShares Short VIX Short-Term Futures ETF"),
            (None, "inverse"),
        )
        self.assertFalse(is_long_leveraged_name("ProShares Short VIX Short-Term Futures ETF"))

    def test_signed_inverse_multiples_are_not_long_leveraged(self) -> None:
        cases = {
            "-2X Daily MSTR ETF": 2.0,
            "-3X Tesla Daily ETF": 3.0,
            "-4X Daily SPY ETF": 4.0,
            "MAX S&P 500 -5X Leveraged ETN": 5.0,
            "Example -1.5X Daily QQQ ETF": 1.5,
            "Example -150% Daily QQQ ETF": 1.5,
        }

        for name, leverage in cases.items():
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (leverage, "inverse"))
                self.assertFalse(is_long_leveraged_name(name))
                self.assertTrue(is_short_leveraged_name(name))

    def test_short_leveraged_workflow_selection_uses_inverse_products(self) -> None:
        products = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "SQQQ", "name": "ProShares UltraPro Short QQQ", "fund_type": "ETF"},
                {"symbol": "SVIX", "name": "ProShares Short VIX Short-Term Futures ETF", "fund_type": "ETF"},
            ]
        )

        short_universe = select_short_workflow_universe(products)

        self.assertEqual(short_universe["symbol"].tolist(), ["SQQQ"])
        self.assertTrue(is_short_leveraged_name("ProShares UltraPro Short QQQ"))
        self.assertFalse(is_short_leveraged_name("ProShares Short VIX Short-Term Futures ETF"))

    def test_percent_based_leverage_names_are_classified(self) -> None:
        self.assertEqual(
            infer_leverage_and_direction("ProShares 200% Long QQQ ETF"),
            (2.0, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 150% Long QQQ ETF"),
            (1.5, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 125% Long QQQ ETF"),
            (1.25, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 300% Long QQQ ETF"),
            (3.0, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 400% Long SPY ETF"),
            (4.0, "long"),
        )
        self.assertEqual(
            infer_leverage_and_direction("ProShares 500% Long QQQ ETF"),
            (5.0, "long"),
        )
        self.assertTrue(is_long_leveraged_name("ProShares 150% Long QQQ ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 125% Long QQQ ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 200% Long QQQ ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 300% Long QQQ ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 400% Long SPY ETF"))
        self.assertTrue(is_long_leveraged_name("ProShares 500% Long QQQ ETF"))
        self.assertFalse(is_long_leveraged_name("ProShares 100% Long QQQ ETF"))
        self.assertFalse(is_long_leveraged_name("ProShares 50% Leveraged QQQ ETF"))

        self.assertEqual(
            infer_leverage_and_direction("Example 500% Short QQQ ETF"),
            (5.0, "inverse"),
        )
        self.assertFalse(is_long_leveraged_name("Example 500% Short QQQ ETF"))

    def test_defined_outcome_percentages_are_not_treated_as_leverage(self) -> None:
        names = [
            "Innovator 200% Upside Participation Buffer ETF",
            "Defined Outcome 200% Participation ETF",
            "Example 150% Participation Cap ETF",
            "Innovator Ultra Buffer 200% Outcome ETF",
            "Example Daily 200% Upside Participation ETF",
        ]

        for name in names:
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (None, None))
                self.assertFalse(leveraged_name_filter(name))
                self.assertFalse(is_long_leveraged_name(name))

        daily_target = "Example 200% Daily Target QQQ ETF"
        self.assertEqual(infer_leverage_and_direction(daily_target), (2.0, "long"))
        self.assertTrue(is_long_leveraged_name(daily_target))

    def test_income_distribution_percentages_are_not_treated_as_leverage(self) -> None:
        names = [
            "Daily 200% Distribution Rate ETF",
            "Daily 150% Income Target ETF",
            "200% Daily Yield ETF",
        ]

        for name in names:
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (None, None))
                self.assertFalse(leveraged_name_filter(name))
                self.assertFalse(is_long_leveraged_name(name))

    def test_daily_percentage_exposure_objectives_are_classified(self) -> None:
        names = [
            "Example 200% Daily Target QQQ ETF",
            "Example Daily Target 200% QQQ ETF",
            "Example Daily 200% Target Exposure QQQ ETF",
            "Example 200% Daily Exposure QQQ ETF",
            "Example 200% Daily Reset QQQ ETF",
        ]

        for name in names:
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (2.0, "long"))
                self.assertTrue(leveraged_name_filter(name))
                self.assertTrue(is_long_leveraged_name(name))

    def test_numeric_multiples_above_supported_cap_are_not_leveraged(self) -> None:
        for name in [
            "10X Long QQQ ETF",
            "6X Long QQQ ETF",
            "5.5X Long QQQ ETF",
            "MAX S&P 500 10X Leveraged ETN",
            "MAX S&P 500 -10X Leveraged ETN",
            "S&P 500 6X Daily ETF",
            "600% Leveraged QQQ ETF",
            "1000% Leveraged QQQ ETF",
            "MAX S&P 500 1000% Leveraged ETN",
            "10X Long QQQ ETF 2X",
            "600% 2X Long QQQ ETF",
        ]:
            with self.subTest(name=name):
                leverage, _direction = infer_leverage_and_direction(name)
                self.assertIsNone(leverage)
                self.assertFalse(leveraged_name_filter(name))
                self.assertFalse(is_long_leveraged_name(name))

    def test_company_name_multiplier_does_not_hide_product_leverage(self) -> None:
        name = "10x Genomics 2X Long Daily ETF"

        self.assertEqual(infer_leverage_and_direction(name), (2.0, "long"))
        self.assertTrue(leveraged_name_filter(name))
        self.assertTrue(is_long_leveraged_name(name))

    def test_embedded_company_name_multipliers_are_not_leverage(self) -> None:
        for name in [
            "8X8 INC /DE/",
            "10x Genomics, Inc.",
            "V2X, Inc.",
            "Ultragenyx Pharmaceutical Inc.",
        ]:
            self.assertEqual(infer_leverage_and_direction(name), (None, None))
            self.assertFalse(leveraged_name_filter(name))
            self.assertFalse(is_long_leveraged_name(name))

    def test_bare_leveraged_names_need_known_leverage_to_be_long(self) -> None:
        self.assertEqual(infer_leverage_and_direction("Example Leveraged ETF"), (None, None))
        self.assertTrue(leveraged_name_filter("Example Leveraged ETF"))
        self.assertFalse(is_long_leveraged_name("Example Leveraged ETF"))

        self.assertEqual(infer_leverage_and_direction("Example Daily Leveraged Exposure ETF"), (None, "long"))
        self.assertTrue(leveraged_name_filter("Example Daily Leveraged Exposure ETF"))
        self.assertFalse(is_long_leveraged_name("Example Daily Leveraged Exposure ETF"))

    def test_msox_has_curated_leverage_and_rsi_underlying(self) -> None:
        name = "MSOS Daily Leveraged ETF"
        mapping = infer_rsi_mapping("MSOX", name, known_symbols={"MSOX", "MSOS"})

        self.assertEqual(infer_leverage_and_direction(name), (2.0, "long"))
        self.assertTrue(is_long_leveraged_name(name))
        self.assertEqual(mapping.rsi_symbol, "MSOS")
        self.assertEqual(mapping.confidence, "curated")
        self.assertEqual(mapping.mapping_source, "symbol_override")

    def test_mst_has_curated_leverage_and_rsi_underlying(self) -> None:
        for name in [
            "Defiance Leveraged Long + Income MSTR ETF",
            "Defiance Leveraged Long Plus Income MSTR ETF",
        ]:
            with self.subTest(name=name):
                mapping = infer_rsi_mapping("MST", name, known_symbols={"MST", "MSTR"})

                self.assertEqual(infer_leverage_and_direction(name), (1.75, "long"))
                self.assertTrue(is_long_leveraged_name(name))
                self.assertEqual(mapping.rsi_symbol, "MSTR")
                self.assertEqual(mapping.confidence, "curated")
                self.assertEqual(mapping.mapping_source, "symbol_override")

    def test_elol_conjunction_spellings_share_one_leverage_classification(self) -> None:
        names = (
            "100% TSLA + 100% SPCX Daily ETF",
            "Leverage Shares 100% TSLA AND 100% SPCX Daily ETF",
            "100% TSLA & 100% SPCX Daily ETF",
        )

        for name in names:
            with self.subTest(name=name):
                self.assertEqual(infer_leverage_and_direction(name), (2.0, "long"))
                self.assertTrue(is_long_leveraged_name(name))

    def test_extra_issuer_sources_are_registered(self) -> None:
        issuer_sources = {source.name: source for source in ISSUER_UNIVERSE_SOURCES}

        for issuer in [
            "AdvisorShares",
            "AXS Investments",
            "Kurv",
            "Innovator",
            "Tuttle Capital",
            "Leverage Shares",
            "YieldMax",
            "Tidal",
            "Roundhill",
            "Themes",
            "Simplify",
        ]:
            self.assertIn(issuer, issuer_sources)

        self.assertEqual(issuer_sources["Tradr"].parser, "tradr_html")
        self.assertEqual(issuer_sources["REX Shares"].parser, "cboe_issuer_html")
        self.assertEqual(issuer_sources["Innovator"].parser, "innovator_html")
        self.assertEqual(issuer_sources["Leverage Shares"].parser, "leverage_shares_html")
        self.assertIn("/all-etfs/", issuer_sources["Leverage Shares"].url)
        self.assertEqual(issuer_sources["YieldMax"].parser, "yieldmax_html")
        self.assertEqual(issuer_sources["YieldMax"].url, "https://yieldmaxetfs.com/our-etfs/")
        self.assertIn("cboe.com", issuer_sources["REX Shares"].url)

    def test_etn_sources_are_registered(self) -> None:
        source_names = {source.name for source in WORKFLOW_ETN_SOURCES}

        self.assertIn("MicroSectors", source_names)
        self.assertIn("UBS ETRACS", source_names)

    def test_etn_source_status_uses_registered_source_type(self) -> None:
        with (
            patch(
                "leveraged_trader.universe.WORKFLOW_ETN_SOURCES",
                [
                    UniverseSource(
                        "Test ETN",
                        "https://etn.test",
                        source_type="etn_issuer",
                    )
                ],
            ),
            patch("leveraged_trader.universe.requests.get", side_effect=RuntimeError("offline")),
        ):
            etn_universe = load_etn_universe()

        status = etn_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["source_type"], "etn_issuer")

    def test_audit_sources_are_registered_without_being_authoritative(self) -> None:
        source_names = {source.name for source in AUDIT_UNIVERSE_SOURCES}

        for source_name in [
            "NYSE exchange-traded products directory",
            "Nasdaq funds/ETFs directory",
            "Cboe listed products",
            "ETFdb leveraged ETF directory",
            "VettaFi ETF database",
            "ETF.com ETF finder",
            "SEC EDGAR company ticker registry",
            "SEC EDGAR exchange ticker registry",
            "SEC EDGAR mutual fund ticker registry",
            "SEC EDGAR full-text search",
        ]:
            self.assertIn(source_name, source_names)

        self.assertTrue(all(source.source_type != "issuer" for source in AUDIT_UNIVERSE_SOURCES))

    def test_dynamic_nyse_audit_directory_is_registered_only(self) -> None:
        nyse_source = next(
            source for source in AUDIT_UNIVERSE_SOURCES if source.name == "NYSE exchange-traded products directory"
        )

        self.assertFalse(nyse_source.enabled)
        self.assertEqual(nyse_source.parser, "registered_only")
        self.assertIn("client-side", nyse_source.notes)

    def test_symbol_only_cboe_audit_is_reported_as_inventory_only(self) -> None:
        source = UniverseSource(
            "Cboe listed products",
            "https://example.test/cboe.csv",
            source_type="exchange_directory",
            parser="cboe_symbol_csv",
        )
        fetch_result = Mock(text="Symbol\nTQQQ\nULTRA\n", error="")

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch(
                "leveraged_trader.universe._fetch_enabled_sources",
                return_value=[fetch_result],
            ),
        ):
            rows, status = load_audit_universe_sources()

        self.assertEqual(rows["symbol"].tolist(), ["TQQQ", "ULTRA"])
        self.assertFalse(rows["is_leveraged_candidate"].any())
        self.assertTrue(
            build_universe_audit_report(
                rows,
                pd.DataFrame({"symbol": []}),
                pd.DataFrame({"symbol": []}),
            ).empty
        )
        self.assertEqual(status.loc[0, "status"], "loaded_inventory_only")
        self.assertEqual(status.loc[0, "audit_capability"], "inventory_only")
        self.assertEqual(status.loc[0, "row_count"], 2)

    def test_cboe_symbol_parser_retains_inventory_without_claiming_product_names(self) -> None:
        source = UniverseSource(
            "Cboe listed products",
            "https://example.test/cboe.csv",
            source_type="exchange_directory",
            parser="cboe_symbol_csv",
        )

        out = _cboe_symbol_csv_to_universe("Symbol\nTQQQ\n", source)

        self.assertEqual(out[["symbol", "name"]].to_dict("records"), [{"symbol": "TQQQ", "name": "TQQQ"}])

    def test_cboe_symbol_parser_accepts_live_shaped_name_inventory(self) -> None:
        source = UniverseSource(
            "Cboe listed products",
            "https://example.test/cboe.csv",
            source_type="exchange_directory",
            parser="cboe_symbol_csv",
        )
        csv_text = (
            "Name,Volume,Ask Size,Ask Price,Bid Size,Bid Price,Last Price,Shares Matched,Shares Routed\n"
            "TQQQ,5317723,1198,11.01,4086,11.00,11.01,5206863,110860\n"
            "NA,10,1,2.00,1,1.99,2.00,8,2\n"
        )

        out = _cboe_symbol_csv_to_universe(csv_text, source)

        self.assertEqual(out["symbol"].tolist(), ["TQQQ", "NA"])

    def test_cboe_symbol_parser_rejects_malformed_schema_cells_and_duplicates(self) -> None:
        source = UniverseSource(
            "Cboe listed products",
            "https://example.test/cboe.csv",
            source_type="exchange_directory",
            parser="cboe_symbol_csv",
        )
        malformed_csv_values = {
            "missing symbol schema": "Volume,Price\n10,12\n",
            "ambiguous symbol schema": "Symbol,Ticker\nTQQQ,TQQQ\n",
            "duplicate header": "Symbol,Symbol\nTQQQ,TQQQ\n",
            "short row": "Symbol,Volume\nTQQQ\n",
            "invalid symbol": "Symbol\nTQQQ\nTQQQ..\n",
            "normalization collision": "Symbol\nBRKB\nBRK.B\n",
            "empty symbol cell": "Symbol,Volume\n,10\n",
            "malformed quoting": 'Symbol,Volume\n"TQQQ,10\n',
        }

        for label, csv_text in malformed_csv_values.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                _cboe_symbol_csv_to_universe(csv_text, source)

    def test_cboe_audit_discards_entire_partial_malformed_inventory(self) -> None:
        source = UniverseSource(
            "Cboe listed products",
            "https://example.test/cboe.csv",
            source_type="exchange_directory",
            parser="cboe_symbol_csv",
        )
        fetch_result = Mock(text="Symbol\nTQQQ\nTQQQ..\n", error="")

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch("leveraged_trader.universe._fetch_enabled_sources", return_value=[fetch_result]),
        ):
            rows, status = load_audit_universe_sources()

        self.assertTrue(rows.empty)
        self.assertEqual(status.loc[0, "status"], "error")
        self.assertEqual(status.loc[0, "row_count"], 0)
        self.assertIn("invalid or noncanonical symbol", status.loc[0, "error"])

    @patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES")
    @patch("leveraged_trader.universe.requests.get")
    def test_enabled_audit_source_with_no_parsed_rows_is_an_error(
        self,
        mock_get: Mock,
        mock_sources: Mock,
    ) -> None:
        mock_sources.__iter__.return_value = iter(
            [UniverseSource("Empty directory", "https://example.test", "exchange_directory")]
        )
        response = Mock(status_code=200, text="")
        response.text = "<html><body>No parseable product table</body></html>"
        mock_get.return_value = response

        rows, status = load_audit_universe_sources()

        self.assertTrue(rows.empty)
        self.assertEqual(status.loc[0, "status"], "error")
        self.assertIn("Parser returned no rows", status.loc[0, "error"])

    def test_generic_audit_source_rejects_conflicting_duplicate_ticker_metadata(self) -> None:
        source = UniverseSource(
            "Generic audit directory",
            "https://example.test/products",
            "third_party_audit",
        )
        fetch_result = Mock(
            text=(
                "<table><tr><th>Ticker</th><th>Fund Name</th></tr>"
                "<tr><td>DUP</td><td>Plain Income ETF</td></tr>"
                "<tr><td>DUP</td><td>Example 2X Long QQQ Daily ETF</td></tr>"
                "</table>"
            ),
            error="",
        )

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch("leveraged_trader.universe._fetch_enabled_sources", return_value=[fetch_result]),
        ):
            rows, status = load_audit_universe_sources()

        self.assertTrue(rows.empty)
        self.assertEqual(status.loc[0, "status"], "error")
        self.assertEqual(status.loc[0, "row_count"], 0)
        self.assertIn("conflicting metadata for duplicate ticker", status.loc[0, "error"])
        self.assertIn("DUP", status.loc[0, "error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_sec_audit_source_is_skipped_without_contact_identity(self, mock_get: Mock) -> None:
        source = UniverseSource(
            "SEC test",
            "https://www.sec.gov/files/example.json",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch.dict("os.environ", {"SEC_USER_AGENT": ""}),
        ):
            rows, status = load_audit_universe_sources()

        self.assertTrue(rows.empty)
        self.assertEqual(status.loc[0, "status"], "skipped_configuration")
        self.assertIn("SEC_USER_AGENT", status.loc[0, "error"])
        mock_get.assert_not_called()

    def test_microsectors_parser_builds_etn_rows(self) -> None:
        source = UniverseSource(
            "MicroSectors",
            "https://example.test",
            source_type="etn_issuer",
            parser="microsectors_html",
        )
        html = """
            <div class="item"><a href="/fang">
                <div class="suite-name">Fang+</div>
                <div class="products">
                    <div class="product">
                        <div class="product-symbol">FNGU</div>
                        <div class="product-description">3X Leveraged Exposure</div>
                    </div>
                    <div class="product">
                        <div class="product-symbol">FNGS</div>
                        <div class="product-description">1X Long Exposure</div>
                    </div>
                    <div class="product">
                        <div class="product-symbol">FNGD</div>
                        <div class="product-description">-3X Inverse Leveraged Exposure</div>
                    </div>
                </div>
            </a></div>
        """

        out = _microsectors_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["FNGU", "FNGD"])
        self.assertTrue(out["fund_type"].str.startswith("ETN").all())

    def test_microsectors_parser_rejects_malformed_and_colliding_raw_tickers(self) -> None:
        source = UniverseSource(
            "MicroSectors",
            "https://example.test",
            source_type="etn_issuer",
            parser="microsectors_html",
        )
        cases = {
            "malformed ticker": (
                """
                <div class="item"><a><div class="suite-name">Test</div><div>
                    <div class="product-symbol">BAD..</div>
                    <div class="product-description">3X Leveraged Exposure</div>
                </div></a></div>
                """,
                "invalid or noncanonical product ticker",
            ),
            "normalization collision": (
                """
                <div class="item"><a><div class="suite-name">Test</div><div>
                    <div class="product-symbol">BRKB</div>
                    <div class="product-description">3X Leveraged Exposure</div>
                    <div class="product-symbol">BRK.B</div>
                    <div class="product-description">3X Leveraged Exposure</div>
                </div></a></div>
                """,
                "normalization collision.*BRK-B",
            ),
        }

        for label, (html, message) in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                _microsectors_html_to_universe(html, source, require_leveraged=False)

    def test_etracs_parser_extracts_ticker_from_hidden_url_cell(self) -> None:
        source = UniverseSource(
            "UBS ETRACS",
            "https://example.test",
            source_type="etn_issuer",
            parser="etracs_leverage_table",
        )
        html = """
            <table>
                <tr><th>Ticker symbol</th><th>Name</th><th>Leverage</th></tr>
                <tr>
                    <td><div>/product/detail/index/ussymbol/BDCX</div><span>BDCX</span></td>
                    <td>ETRACS Quarterly Pay 1.5x Leveraged MarketVector BDC Liquid Index ETN</td>
                    <td>1.50x</td>
                </tr>
                <tr>
                    <td><div>/product/detail/index/ussymbol/BDCY</div><span>BDCY</span></td>
                    <td>ETRACS Quarterly Pay MarketVector BDC Liquid Index ETN</td>
                    <td>1.50x</td>
                </tr>
                <tr>
                    <td><div>/product/detail/index/ussymbol/BDCZ</div><span>BDCZ</span></td>
                    <td>ETRACS Quarterly Pay MarketVector BDC Liquid Index ETN</td>
                    <td>-1.50x</td>
                </tr>
                <tr>
                    <td><div>/product/detail/index/ussymbol/HDLB</div><span>HDLB</span></td>
                    <td>ETRACS Monthly Pay 2xLeveraged US High Dividend Low Volatility ETN Series B</td>
                    <td>2.00x</td>
                </tr>
                <tr>
                    <td><div>/product/detail/index/ussymbol/PLAIN</div><span>PLAIN</span></td>
                    <td>ETRACS Plain Index ETN</td>
                    <td>--</td>
                </tr>
            </table>
        """

        out = _etracs_leverage_table_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["BDCX", "BDCY", "BDCZ", "HDLB"])
        self.assertEqual(
            out["fund_type"].tolist(),
            ["ETN (UBS ETRACS)"] * 4,
        )
        self.assertTrue(out.loc[out["symbol"].eq("BDCY"), "name"].item().endswith("1.5X Leveraged"))
        self.assertTrue(out.loc[out["symbol"].eq("BDCZ"), "name"].item().endswith("1.5X Inverse Leveraged"))
        self.assertFalse(is_long_leveraged_name(out.loc[out["symbol"].eq("BDCZ"), "name"].item()))
        self.assertTrue(
            out.loc[out["symbol"].eq("HDLB"), "name"].item().endswith("2X Leveraged"),
        )

    def test_etracs_parser_validates_complete_ticker_cell_identities(self) -> None:
        source = UniverseSource(
            "UBS ETRACS",
            "https://example.test",
            source_type="etn_issuer",
            parser="etracs_leverage_table",
        )
        malformed_cells = (
            "BAD.. GOOD",
            '<a href="/product/detail/index/ussymbol/BAD">GOOD</a>',
            '<a href="/product/detail/index/ussymbol/BAD..">BAD</a>',
        )
        for ticker_cell in malformed_cells:
            html = f"""
                <table>
                    <tr><th>Ticker symbol</th><th>Name</th><th>Leverage</th></tr>
                    <tr><td>{ticker_cell}</td><td>Example 2X Leveraged ETN</td><td>2.0x</td></tr>
                </table>
            """
            with (
                self.subTest(ticker_cell=ticker_cell),
                self.assertRaisesRegex(
                    ValueError,
                    "invalid ticker cell",
                ),
            ):
                _etracs_leverage_table_to_universe(html, source, require_leveraged=False)

        # A complete six-character identity must never be halved merely because
        # its two halves happen to be equal.
        six_character_html = """
            <table>
                <tr><th>Ticker symbol</th><th>Name</th><th>Leverage</th></tr>
                <tr>
                    <td>/product/detail/index/ussymbol/BADBAD</td>
                    <td>Example 2X Leveraged ETN</td>
                    <td>2.0x</td>
                </tr>
            </table>
        """
        out = _etracs_leverage_table_to_universe(
            six_character_html,
            source,
            require_leveraged=False,
        )
        self.assertEqual(out["symbol"].tolist(), ["BADBAD"])

    def test_etracs_parser_rejects_malformed_or_contradictory_leverage_cells(self) -> None:
        source = UniverseSource(
            "UBS ETRACS",
            "https://example.test",
            source_type="etn_issuer",
            parser="etracs_leverage_table",
        )
        cases = {
            "multiple mismatch": ("Example 3X Leveraged ETN", "2.00x"),
            "ordinary mismatch": ("Example 2X Leveraged ETN", "1.00x"),
            "positive inverse": ("Example 2X Inverse ETN", "2.00x"),
            "negative long": ("Example 2X Long ETN", "-2.00x"),
            "conflicting name direction": ("Example 2X Long Inverse ETN", "-2.00x"),
            "conflicting name multiples": ("Example 2X 3X Long ETN", "2.00x"),
            "joined multiple mismatch": ("Example 3xLeveraged ETN", "2.00x"),
            "sentinel mismatch": ("Example 2X Leveraged ETN", "--"),
            "malformed value": ("Example 2X Leveraged ETN", "2.00x trailing"),
        }
        for label, (name, leverage) in cases.items():
            html = f"""
                <table>
                    <tr><th>Ticker symbol</th><th>Name</th><th>Leverage</th></tr>
                    <tr><td>/ussymbol/BAD</td><td>{name}</td><td>{leverage}</td></tr>
                </table>
            """
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "ETRACS"):
                _etracs_leverage_table_to_universe(html, source, require_leveraged=False)

    def test_etracs_parser_rejects_ambiguous_headers_and_row_widths(self) -> None:
        source = UniverseSource(
            "UBS ETRACS",
            "https://example.test",
            source_type="etn_issuer",
            parser="etracs_leverage_table",
        )
        malformed_tables = {
            "duplicate leverage": """
                <tr><th>Ticker</th><th>Name</th><th>Leverage</th><th>Leverage</th></tr>
                <tr><td>BAD</td><td>Example 2X Leveraged ETN</td><td>2.00x</td><td>3.00x</td></tr>
            """,
            "duplicate ticker semantics": """
                <tr><th>Ticker</th><th>Ticker symbol</th><th>Name</th><th>Leverage</th></tr>
                <tr><td>BAD</td><td>OTHER</td><td>Example 2X Leveraged ETN</td><td>2.00x</td></tr>
            """,
            "extra row cell": """
                <tr><th>Ticker</th><th>Name</th><th>Leverage</th></tr>
                <tr><td>BAD</td><td>Example 2X Leveraged ETN</td><td>2.00x</td><td>-3.00x</td></tr>
            """,
            "missing leverage": """
                <tr><th>Ticker symbol</th><th>Name</th><th>Indicative value</th></tr>
                <tr><td>BAD</td><td>Example 2X Leveraged ETN</td><td>$20.00</td></tr>
            """,
        }
        for label, table_body in malformed_tables.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                _etracs_leverage_table_to_universe(
                    f"<table>{table_body}</table>",
                    source,
                    require_leveraged=False,
                )

    def test_etracs_parser_rejects_a_blank_name_before_eligibility_filtering(self) -> None:
        source = UniverseSource(
            "UBS ETRACS",
            "https://example.test",
            source_type="etn_issuer",
            parser="etracs_leverage_table",
        )
        html = """
            <table>
                <tr><th>Ticker symbol</th><th>Name</th><th>Leverage</th></tr>
                <tr><td>/ussymbol/BAD</td><td> </td><td>1.00x</td></tr>
            </table>
        """

        with self.assertRaisesRegex(ValueError, "omitted its required product name"):
            _etracs_leverage_table_to_universe(html, source, require_leveraged=False)

    def test_tradr_parser_accepts_live_schema_and_compatible_duplicates(self) -> None:
        source = UniverseSource(
            "Tradr",
            "https://example.test",
            source_type="issuer_etf",
            parser="tradr_html",
        )
        html = """
            <table>
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th>
                    <th>Target</th><th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>LONG</td><td>Tradr 2X Long TEST Daily ETF</td><td>TEST</td>
                    <td>2X</td><td>Long</td><td>Daily</td>
                </tr>
                <tr>
                    <td>SHORT</td><td>Tradr 2X Short TEST Daily ETF</td><td>TEST</td>
                    <td>2X</td><td>Short</td><td>Daily</td>
                </tr>
                <tr>
                    <td>NEG</td><td>Tradr 1.5X Short TEST Daily ETF</td><td>TEST</td>
                    <td>-1.5X</td><td>Short</td><td>Daily</td>
                </tr>
                <tr>
                    <td>MONTH</td><td>Tradr 2X Long TEST Monthly ETF</td><td>TEST</td>
                    <td>2x</td><td>Long</td><td>Monthly</td>
                </tr>
                <tr>
                    <td>TARK</td><td>Tradr 2X Long Innovation ETF</td><td>ARKK</td>
                    <td>2X</td><td>Long</td><td>Daily</td>
                </tr>
            </table>
            <table>
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th>
                    <th>Target</th><th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>LONG</td><td>Tradr 2X Long TEST Daily ETF</td><td>TEST</td>
                    <td>2x</td><td>LONG</td><td>DAILY</td>
                </tr>
            </table>
        """

        out = _tradr_html_to_universe(html, source, require_leveraged=False)

        self.assertEqual(out["symbol"].tolist(), ["LONG", "SHORT", "NEG", "MONTH", "TARK"])
        self.assertEqual(out["reference_security"].tolist(), ["TEST", "TEST", "TEST", "TEST", "ARKK"])
        mapped = build_nasdaq_universe_table(out, known_symbols={"ARKK", "TEST"})
        self.assertEqual(mapped["rsi_symbol"].tolist(), ["TEST", "TEST", "TEST", "TEST", "ARKK"])
        self.assertEqual(set(mapped["mapping_source"]), {"issuer_reference"})

    def test_tradr_parser_rejects_reference_security_that_contradicts_the_product_name(self) -> None:
        source = UniverseSource(
            "Tradr",
            "https://example.test",
            source_type="issuer_etf",
            parser="tradr_html",
        )
        html = """
            <table>
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th>
                    <th>Target</th><th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>WRONG</td><td>Tradr 2X Long NVDA Daily ETF</td><td>TSLA</td>
                    <td>2X</td><td>Long</td><td>Daily</td>
                </tr>
            </table>
        """

        with self.assertRaisesRegex(ValueError, "Reference Security contradicted"):
            _tradr_html_to_universe(html, source, require_leveraged=False)

    def test_tradr_parser_rejects_malformed_or_contradictory_authoritative_fields(self) -> None:
        source = UniverseSource(
            "Tradr",
            "https://example.test",
            source_type="issuer_etf",
            parser="tradr_html",
        )
        cases = {
            "blank name": (" ", "2X", "Long", "Daily"),
            "malformed target": ("Tradr 2X Long TEST Daily ETF", "2X trailing", "Long", "Daily"),
            "multiple mismatch": ("Tradr 2X Long TEST Daily ETF", "1X", "Long", "Daily"),
            "negative long": ("Tradr 2X Long TEST Daily ETF", "-2X", "Long", "Daily"),
            "positive short": ("Tradr 2X Short TEST Daily ETF", "+2X", "Short", "Daily"),
            "malformed exposure": ("Tradr 2X Long TEST Daily ETF", "2X", "Bull", "Daily"),
            "direction mismatch": ("Tradr 2X Long TEST Daily ETF", "2X", "Short", "Daily"),
            "conflicting name direction": ("Tradr 2X Long Inverse TEST Daily ETF", "-2X", "Short", "Daily"),
            "conflicting name multiples": ("Tradr 2X 3X Long TEST Daily ETF", "2X", "Long", "Daily"),
            "malformed reset": ("Tradr 2X Long TEST Daily ETF", "2X", "Long", "Weekly"),
            "reset mismatch": ("Tradr 2X Long TEST Monthly ETF", "2X", "Long", "Daily"),
        }
        for label, (name, target, exposure, reset_period) in cases.items():
            html = f"""
                <table>
                    <tr>
                        <th>Ticker</th><th>Fund Name</th><th>Reference Security</th><th>Target</th>
                        <th>Exposure</th><th>Reset Period</th>
                    </tr>
                    <tr>
                        <td>BAD</td><td>{name}</td><td>TEST</td><td>{target}</td>
                        <td>{exposure}</td><td>{reset_period}</td>
                    </tr>
                </table>
            """
            with self.subTest(label=label), self.assertRaises(ValueError):
                _tradr_html_to_universe(html, source, require_leveraged=False)

    def test_tradr_parser_rejects_an_ambiguous_candidate_table(self) -> None:
        source = UniverseSource(
            "Tradr",
            "https://example.test",
            source_type="issuer_etf",
            parser="tradr_html",
        )
        html = """
            <table>
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th><th>Target</th>
                    <th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>GOOD</td><td>Tradr 2X Long TEST Daily ETF</td><td>TEST</td>
                    <td>2X</td><td>Long</td><td>Daily</td>
                </tr>
            </table>
            <table>
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th><th>Target</th><th>Target</th>
                    <th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>BAD</td><td>Tradr 2X Long TEST Daily ETF</td><td>TEST</td>
                    <td>2X</td><td>3X</td><td>Long</td><td>Daily</td>
                </tr>
            </table>
        """

        with self.assertRaisesRegex(ValueError, "ambiguous authoritative columns"):
            _tradr_html_to_universe(html, source, require_leveraged=False)

    def test_tradr_parser_rejects_a_partial_current_candidate_but_ignores_legacy_tables(self) -> None:
        source = UniverseSource(
            "Tradr",
            "https://example.test",
            source_type="issuer_etf",
            parser="tradr_html",
        )
        current_table = """
            <table>
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th><th>Target</th>
                    <th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>GOOD</td><td>Tradr 2X Long TEST Daily ETF</td><td>TEST</td>
                    <td>2X</td><td>Long</td><td>Daily</td>
                </tr>
            </table>
        """
        legacy_table = """
            <table>
                <tr><th>Ticker</th><th>Fund Name</th><th>Reset Period</th></tr>
                <tr><td>STALE</td><td>Tradr 2X Long TEST Monthly ETF</td><td>Monthly</td></tr>
            </table>
        """
        partial_current_table = """
            <table>
                <tr><th>Ticker</th><th>Fund Name</th><th>Reference Security</th><th>Target</th><th>Exposure</th></tr>
                <tr><td>BAD</td><td>Tradr 2X Long TEST Daily ETF</td><td>TEST</td><td>2X</td><td>Long</td></tr>
            </table>
        """

        out = _tradr_html_to_universe(current_table + legacy_table, source, require_leveraged=False)
        self.assertEqual(out["symbol"].tolist(), ["GOOD"])
        with self.assertRaisesRegex(ValueError, "omitted current authoritative columns.*reset period"):
            _tradr_html_to_universe(
                current_table + partial_current_table,
                source,
                require_leveraged=False,
            )

    def test_tradr_parser_rejects_rows_that_do_not_match_the_declared_width(self) -> None:
        source = UniverseSource(
            "Tradr",
            "https://example.test",
            source_type="issuer_etf",
            parser="tradr_html",
        )
        malformed_tables = {
            "extra cell": """
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th><th>Target</th>
                    <th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>BAD</td><td>Tradr 2X Long TEST Daily ETF</td><td>TEST</td>
                    <td>2X</td><td>Long</td><td>Daily</td><td>-5X</td>
                </tr>
            """,
            "missing cell": """
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th><th>Target</th>
                    <th>Exposure</th><th>Reset Period</th><th>Options</th>
                </tr>
                <tr>
                    <td>BAD</td><td>Tradr 2X Long TEST Daily ETF</td><td>TEST</td>
                    <td>2X</td><td>Long</td><td>Daily</td>
                </tr>
            """,
        }
        for label, table_body in malformed_tables.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "declared columns"):
                _tradr_html_to_universe(
                    f"<table>{table_body}</table>",
                    source,
                    require_leveraged=False,
                )

    def test_sec_company_tickers_parser_builds_audit_rows(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        json_text = """
            {
                "0": {"cik_str": 927971, "ticker": "FNGU", "title": "BANK OF MONTREAL /CAN/"},
                "1": {"cik_str": 1114446, "ticker": "BDCX", "title": "UBS AG"}
            }
        """

        out = _sec_company_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["FNGU", "BDCX"])
        self.assertEqual(out["source"].tolist(), ["SEC EDGAR company ticker registry audit source"] * 2)

    def test_sec_named_audit_parsers_dedupe_exact_normalized_identities_and_reject_conflicts(self) -> None:
        company_source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        exchange_source = UniverseSource(
            "SEC EDGAR exchange ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_exchange_tickers",
        )
        cases = (
            (
                _sec_company_tickers_to_universe,
                company_source,
                (
                    '{"0":{"ticker":"BRKB","title":"Berkshire Holdings"},'
                    '"1":{"ticker":"BRK.B","title":"Berkshire Holdings"}}'
                ),
                (
                    '{"0":{"ticker":"BRKB","title":"Berkshire 2X Long ETF"},'
                    '"1":{"ticker":"BRK.B","title":"Berkshire 2X Short ETF"}}'
                ),
            ),
            (
                _sec_exchange_tickers_to_universe,
                exchange_source,
                ('{"fields":["name","ticker"],"data":[["Berkshire Holdings","BRKB"],["Berkshire Holdings","BRK.B"]]}'),
                (
                    '{"fields":["name","ticker"],"data":['
                    '["Berkshire 2X Long ETF","BRKB"],["Berkshire 2X Short ETF","BRK.B"]]}'
                ),
            ),
        )

        for parser, source, exact_duplicate, conflict in cases:
            with self.subTest(parser=source.parser, outcome="dedupe"):
                out = parser(exact_duplicate, source)
                self.assertEqual(
                    out[["symbol", "name"]].to_dict("records"), [{"symbol": "BRK-B", "name": "Berkshire Holdings"}]
                )
            with (
                self.subTest(parser=source.parser, outcome="conflict"),
                self.assertRaisesRegex(ValueError, "conflicting names for normalized ticker BRK-B"),
            ):
                parser(conflict, source)

    def test_sec_conflicting_normalized_identity_marks_entire_audit_source_degraded(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://registry.example.test/company-tickers.json",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        fetch_result = Mock(
            text=(
                '{"0":{"ticker":"BRKB","title":"Berkshire 2X Long ETF"},'
                '"1":{"ticker":"BRK.B","title":"Berkshire 2X Short ETF"}}'
            ),
            error="",
        )

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch("leveraged_trader.universe._fetch_enabled_sources", return_value=[fetch_result]),
        ):
            rows, status = load_audit_universe_sources()

        self.assertTrue(rows.empty)
        self.assertEqual(status.loc[0, "status"], "error")
        self.assertEqual(status.loc[0, "row_count"], 0)
        self.assertIn("conflicting names for normalized ticker BRK-B", status.loc[0, "error"])

    def test_sec_company_tickers_parser_rejects_malformed_json_schema_and_records(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        malformed_payloads = (
            "not JSON",
            "[]",
            '{"0": ["TQQQ", "ProShares UltraPro QQQ"]}',
            '{"0": {"ticker": "TQQQ"}}',
            '{"0": {"ticker": ["TQQQ"], "title": "ProShares UltraPro QQQ"}}',
        )

        for json_text in malformed_payloads:
            with self.subTest(json_text=json_text), self.assertRaises(ValueError):
                _sec_company_tickers_to_universe(json_text, source)

    def test_sec_audit_json_parsers_reject_duplicate_keys_at_every_object_depth(self) -> None:
        source = UniverseSource(
            "SEC test registry",
            "https://example.test",
            source_type="filing_audit",
        )
        malformed_payloads = [
            (
                "company top-level key",
                _sec_company_tickers_to_universe,
                '{"0":{"ticker":"AAA","title":"A"},"0":{"ticker":"BBB","title":"B"}}',
            ),
            (
                "company nested key",
                _sec_company_tickers_to_universe,
                '{"0":{"ticker":"AAA","title":"A","metadata":{"kind":1,"kind":2}}}',
            ),
            (
                "exchange container key",
                _sec_exchange_tickers_to_universe,
                '{"fields":["name","ticker"],"data":[["A","AAA"]],"data":[["B","BBB"]]}',
            ),
            (
                "mutual-fund nested key",
                _sec_mutual_fund_tickers_to_universe,
                ('{"metadata":{"version":1,"version":2},"fields":["symbol"],"data":[["AAA"]]}'),
            ),
        ]

        for label, parser, json_text in malformed_payloads:
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, "duplicate JSON object key"),
            ):
                parser(json_text, source)

    def test_sec_company_tickers_partial_schema_failure_marks_audit_source_error(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://registry.example.test/company-tickers.json",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        fetch_result = Mock(
            text=('{"0": {"ticker": "GOOD", "title": "Good Corp"}, "1": {"ticker": "BAD"}}'),
            error="",
        )

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch("leveraged_trader.universe._fetch_enabled_sources", return_value=[fetch_result]),
        ):
            rows, status = load_audit_universe_sources()

        self.assertTrue(rows.empty)
        self.assertEqual(status.loc[0, "status"], "error")
        self.assertEqual(status.loc[0, "row_count"], 0)
        self.assertIn("required 'title'", status.loc[0, "error"])

    def test_sec_company_tickers_invalid_partial_ticker_marks_audit_source_error(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://registry.example.test/company-tickers.json",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        fetch_result = Mock(
            text=('{"0": {"ticker": "GOOD", "title": "Good Corp"}, "1": {"ticker": "???", "title": "Bad Corp"}}'),
            error="",
        )

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch("leveraged_trader.universe._fetch_enabled_sources", return_value=[fetch_result]),
        ):
            rows, status = load_audit_universe_sources()

        self.assertTrue(rows.empty)
        self.assertEqual(status.loc[0, "status"], "error")
        self.assertEqual(status.loc[0, "row_count"], 0)
        self.assertIn("invalid or noncanonical", status.loc[0, "error"])

    def test_sec_company_tickers_accepts_valid_aliases_and_excluded_symbols(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        json_text = (
            '{"0": {"ticker": "BRK-B", "title": "Berkshire Hathaway"}, '
            '"1": {"ticker": "VCLT", "title": "Vanguard Long-Term Corporate Bond ETF"}}'
        )

        out = _sec_company_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["BRK-B"])

    def test_sec_company_tickers_skips_official_unavailable_ticker_sentinel(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        json_text = (
            '{"0": {"ticker": "GOOD", "title": "Good Corp"}, "1": {"ticker": "NONE.", "title": "EBR Systems, Inc."}}'
        )

        out = _sec_company_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["GOOD"])

    def test_sec_exchange_tickers_parser_builds_audit_rows(self) -> None:
        source = UniverseSource(
            "SEC EDGAR exchange ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_exchange_tickers",
        )
        json_text = """
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [
                    [123, "ProShares UltraPro QQQ", "TQQQ", "Nasdaq"],
                    [456, "Plain Company", "PLAIN", "NYSE"]
                ]
            }
        """

        out = _sec_exchange_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["TQQQ", "PLAIN"])
        self.assertEqual(out["name"].tolist(), ["ProShares UltraPro QQQ", "Plain Company"])
        self.assertEqual(out["source"].tolist(), ["SEC EDGAR exchange ticker registry audit source"] * 2)

    def test_sec_exchange_tickers_parser_requires_unambiguous_name_field(self) -> None:
        source = UniverseSource(
            "SEC EDGAR exchange ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_exchange_tickers",
        )
        missing_name = '{"fields": ["cik", "ticker", "exchange"], "data": [[1, "TQQQ", "Nasdaq"]]}'
        duplicate_name = '{"fields": ["cik", "name", "name", "ticker"], "data": [[1, "First", "Second", "TQQQ"]]}'
        duplicate_ticker = (
            '{"fields": ["cik", "name", "ticker", "ticker"], "data": [[1, "ProShares UltraPro QQQ", "TQQQ", "TQQQ"]]}'
        )

        for json_text, field in [
            (missing_name, "name"),
            (duplicate_name, "name"),
            (duplicate_ticker, "ticker"),
        ]:
            with (
                self.subTest(json_text=json_text, field=field),
                self.assertRaisesRegex(ValueError, f"exactly one '{field}' field"),
            ):
                _sec_exchange_tickers_to_universe(json_text, source)

    def test_sec_exchange_tickers_parser_requires_rows_to_match_the_declared_schema(self) -> None:
        source = UniverseSource(
            "SEC EDGAR exchange ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_exchange_tickers",
        )
        malformed_rows = (
            '{"fields": ["cik", "name", "ticker", "exchange"], "data": [[1, "ProShares UltraPro QQQ", "TQQQ"]]}',
            '{"fields": ["cik", "name", "ticker"], "data": [[1, "ProShares UltraPro QQQ", "TQQQ", "Nasdaq"]]}',
        )

        for json_text in malformed_rows:
            with (
                self.subTest(json_text=json_text),
                self.assertRaisesRegex(ValueError, "did not match its declared field schema"),
            ):
                _sec_exchange_tickers_to_universe(json_text, source)

    def test_sec_exchange_tickers_parser_rejects_container_field_values(self) -> None:
        source = UniverseSource(
            "SEC EDGAR exchange ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_exchange_tickers",
        )
        malformed_values = (
            '{"fields": ["name", "ticker"], "data": [[{"forged": "Fund"}, "TQQQ"]]}',
            '{"fields": ["name", "ticker"], "data": [["Fund", ["TQQQ"]]]}',
        )

        for json_text in malformed_values:
            with (
                self.subTest(json_text=json_text),
                self.assertRaisesRegex(ValueError, "omitted its required"),
            ):
                _sec_exchange_tickers_to_universe(json_text, source)

    def test_sec_exchange_schema_failure_is_reported_as_audit_source_error(self) -> None:
        source = UniverseSource(
            "SEC EDGAR exchange ticker registry",
            "https://registry.example.test/tickers.json",
            source_type="filing_audit",
            parser="sec_exchange_tickers",
        )
        fetch_result = Mock(
            text='{"fields": ["cik", "ticker", "exchange"], "data": [[1, "TQQQ", "Nasdaq"]]}',
            error="",
        )

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch("leveraged_trader.universe._fetch_enabled_sources", return_value=[fetch_result]),
        ):
            rows, status = load_audit_universe_sources()

        self.assertTrue(rows.empty)
        self.assertEqual(status.loc[0, "status"], "error")
        self.assertEqual(status.loc[0, "audit_capability"], "product_names")
        self.assertIn("exactly one 'name' field", status.loc[0, "error"])

    def test_sec_exchange_invalid_partial_ticker_marks_audit_source_error(self) -> None:
        source = UniverseSource(
            "SEC EDGAR exchange ticker registry",
            "https://registry.example.test/tickers.json",
            source_type="filing_audit",
            parser="sec_exchange_tickers",
        )
        fetch_result = Mock(
            text=('{"fields": ["name", "ticker"], "data": [["Good Corp", "GOOD"], ["Bad Corp", "???"]]}'),
            error="",
        )

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch("leveraged_trader.universe._fetch_enabled_sources", return_value=[fetch_result]),
        ):
            rows, status = load_audit_universe_sources()

        self.assertTrue(rows.empty)
        self.assertEqual(status.loc[0, "status"], "error")
        self.assertEqual(status.loc[0, "row_count"], 0)
        self.assertIn("invalid or noncanonical", status.loc[0, "error"])

    def test_sec_exchange_tickers_skips_official_unavailable_ticker_sentinel(self) -> None:
        source = UniverseSource(
            "SEC EDGAR exchange ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_exchange_tickers",
        )
        json_text = '{"fields": ["name", "ticker"], "data": [["Good Corp", "GOOD"], ["EBR Systems, Inc.", "NONE."]]}'

        out = _sec_exchange_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["GOOD"])

    def test_sec_mutual_fund_tickers_parser_builds_symbol_only_audit_rows(self) -> None:
        source = UniverseSource(
            "SEC EDGAR mutual fund ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_mutual_fund_tickers",
        )
        json_text = """
            {
                "fields": ["cik", "seriesId", "classId", "symbol"],
                "data": [
                    [123, "S0001", "C0001", "TQQQ"],
                    [456, "S0002", "C0002", null],
                    [789, "S0003", "C0003", "PLAIN"]
                ]
            }
        """

        out = _sec_mutual_fund_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["TQQQ", "PLAIN"])
        self.assertEqual(out["name"].tolist(), ["TQQQ", "PLAIN"])
        self.assertEqual(out["fund_type"].tolist(), ["SEC MF (SEC EDGAR mutual fund ticker registry)"] * 2)

    def test_sec_mutual_fund_tickers_parser_rejects_invalid_nonnull_symbol(self) -> None:
        source = UniverseSource(
            "SEC EDGAR mutual fund ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_mutual_fund_tickers",
        )
        json_text = (
            '{"fields": ["cik", "seriesId", "classId", "symbol"], '
            '"data": [[123, "S0001", "C0001", "TQQQ"], [456, "S0002", "C0002", "???"]]}'
        )

        with self.assertRaisesRegex(ValueError, "invalid or noncanonical 'symbol'"):
            _sec_mutual_fund_tickers_to_universe(json_text, source)

    def test_sec_mutual_fund_tickers_normalizes_official_legacy_spellings(self) -> None:
        source = UniverseSource(
            "SEC EDGAR mutual fund ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_mutual_fund_tickers",
        )
        json_text = (
            '{"fields": ["cik", "seriesId", "classId", "symbol"], "data": ['
            '[1, "S1", "C1", "elfnx"], [2, "S2", "C2", "(NWAKX)"], '
            '[3, "S3", "C3", "n/a"], [4, "S4", "C4", ""]]}'
        )

        out = _sec_mutual_fund_tickers_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["ELFNX", "NWAKX"])

    def test_sec_mutual_fund_source_remains_inventory_only_without_name_field(self) -> None:
        source = UniverseSource(
            "SEC EDGAR mutual fund ticker registry",
            "https://registry.example.test/mutual-funds.json",
            source_type="filing_audit",
            parser="sec_mutual_fund_tickers",
        )
        fetch_result = Mock(
            text=('{"fields": ["cik", "seriesId", "classId", "symbol"], "data": [[123, "S0001", "C0001", "TQQQ"]]}'),
            error="",
        )

        with (
            patch("leveraged_trader.universe.AUDIT_UNIVERSE_SOURCES", [source]),
            patch("leveraged_trader.universe._fetch_enabled_sources", return_value=[fetch_result]),
        ):
            rows, status = load_audit_universe_sources()

        self.assertEqual(rows[["symbol", "name"]].to_dict("records"), [{"symbol": "TQQQ", "name": "TQQQ"}])
        self.assertFalse(rows["is_leveraged_candidate"].any())
        self.assertEqual(status.loc[0, "status"], "loaded_inventory_only")
        self.assertEqual(status.loc[0, "audit_capability"], "inventory_only")

    def test_sec_entity_audit_metadata_requires_product_context(self) -> None:
        source = UniverseSource(
            "SEC EDGAR company ticker registry",
            "https://example.test",
            source_type="filing_audit",
            parser="sec_company_tickers",
        )
        rows = pd.DataFrame(
            [
                {
                    "symbol": "EGHT",
                    "name": "8X8 INC /DE/",
                    "fund_type": "SEC (SEC EDGAR company ticker registry)",
                    "source": "SEC EDGAR company ticker registry audit source",
                },
                {
                    "symbol": "UCTT",
                    "name": "Ultra Clean Holdings, Inc.",
                    "fund_type": "SEC (SEC EDGAR company ticker registry)",
                    "source": "SEC EDGAR company ticker registry audit source",
                },
                {
                    "symbol": "UPRO",
                    "name": "ProShares UltraPro S&P500",
                    "fund_type": "SEC (SEC EDGAR company ticker registry)",
                    "source": "SEC EDGAR company ticker registry audit source",
                },
                {
                    "symbol": "XXXX",
                    "name": "MAX S&P 500 4X Leveraged Exchange Traded Note",
                    "fund_type": "SEC (SEC EDGAR company ticker registry)",
                    "source": "SEC EDGAR company ticker registry audit source",
                },
            ]
        )

        out = _with_audit_metadata(rows, source)

        self.assertEqual(out.loc[out["symbol"] == "EGHT", "is_leveraged_candidate"].item(), False)
        self.assertEqual(out.loc[out["symbol"] == "UCTT", "is_leveraged_candidate"].item(), False)
        self.assertEqual(out.loc[out["symbol"] == "UPRO", "is_leveraged_candidate"].item(), True)
        self.assertEqual(out.loc[out["symbol"] == "XXXX", "leverage"].item(), 4.0)

    def test_audit_metadata_infers_leverage_once_per_row(self) -> None:
        source = UniverseSource("Audit", "https://audit.test", source_type="third_party_audit")
        rows = pd.DataFrame(
            [
                {
                    "symbol": "LONG",
                    "name": "Example 2X Long Daily ETF",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                },
                {
                    "symbol": "SHORT",
                    "name": "Example 3X Inverse Daily ETF",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                },
                {
                    "symbol": "PLAIN",
                    "name": "Example Income ETF",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                },
            ]
        )

        with patch(
            "leveraged_trader.universe.infer_leverage_and_direction",
            wraps=infer_leverage_and_direction,
        ) as mock_infer:
            out = _with_audit_metadata(rows, source)

        self.assertEqual(mock_infer.call_count, len(rows))
        self.assertEqual(out["is_long_leveraged_candidate"].tolist(), [True, False, False])
        self.assertEqual(out["is_short_leveraged_candidate"].tolist(), [False, True, False])

    def test_webflow_card_parser_finds_static_fund_rows(self) -> None:
        html = """
            <div class="tag is-ticker on-dark-bg">MSTU</div></div>
            <div class="grid_table_cell">
                <div aria-hidden="true" class="u-weight-medium u-text-balance">
                    T-REX 2X Long MSTR Daily Target ETF
                </div>
            </div>
            <div fs-cmssort-field="IDENTIFIER" class="text-weight-xbold">TSLZ</div>
            <div role="cell" class="table3_column">
                <div fs-cmssort-field="IDENTIFIER">T-REX 2X Inverse Tesla Daily Target ETF</div>
            </div>
            <a href="/etf/aapy" class="nav_dropdown_link w-inline-block">
                <div class="u-display-inline">AAPY</div>
                <div class="nav_dropdown_link_caption">Kurv Yield Premium Strategy Apple ETF</div>
            </a>
        """

        out = _html_cards_to_universe(
            html,
            "Example",
            source_label="Example issuer table",
            require_leveraged=True,
        )

        self.assertEqual(out["symbol"].tolist(), ["MSTU", "TSLZ"])

    def test_defiance_json_parser_builds_fund_rows(self) -> None:
        source = UniverseSource(
            "Defiance",
            "https://example.test",
            source_type="issuer_etf",
            parser="defiance_json",
        )
        json_text = """
            [
                {"ticker": "AMA", "name": "Defiance Daily Target 2X Long AMAT ETF"},
                {"ticker": "AIX", "name": "Defiance US 100 Tech AI Moat ETF"},
                {"ticker": "SPCQ", "name": "Defiance Daily Target 2X Short SPCX ETF"}
            ]
        """

        out = _defiance_json_to_universe(json_text, source)

        self.assertEqual(out["symbol"].tolist(), ["AMA", "SPCQ"])

    def test_innovator_parser_normalizes_link_verified_lowercase_tickers(self) -> None:
        source = UniverseSource(
            "Innovator",
            "https://www.innovatoretfs.com/define/etfs/",
            source_type="issuer_etf",
            parser="innovator_html",
        )
        html = """
            <table>
                <tr><th>Ticker</th><th>Name</th></tr>
                <tr>
                    <td><a href="/ajul">ajul</a></td>
                    <td><a href="/ajul">Equity Defined Protection ETF</a></td>
                </tr>
                <tr>
                    <td><a href="/ddtg">ddtg</a></td>
                    <td><a href="/ddtg">Equity Dual Directional 10 Buffer ETF</a></td>
                </tr>
            </table>
        """

        out = _workflow_issuer_source_to_universe(html, source, require_leveraged=False)

        self.assertEqual(out["symbol"].tolist(), ["AJUL", "DDTG"])

    def test_innovator_parser_rejects_unverified_lowercase_ticker_cleanup(self) -> None:
        source = UniverseSource(
            "Innovator",
            "https://www.innovatoretfs.com/define/etfs/",
            source_type="issuer_etf",
            parser="innovator_html",
        )
        invalid_rows = (
            '<td><a href="/aapr">ajul</a></td>',
            "<td>ajul</td>",
            '<td><a href="https://attacker.test/ajul">ajul</a></td>',
        )
        for ticker_cell in invalid_rows:
            html = f"""
                <table>
                    <tr><th>Ticker</th><th>Name</th></tr>
                    <tr>{ticker_cell}<td>Equity Defined Protection ETF</td></tr>
                </table>
            """
            with self.subTest(ticker_cell=ticker_cell), self.assertRaisesRegex(ValueError, "product link"):
                _workflow_issuer_source_to_universe(html, source, require_leveraged=False)

    def test_yieldmax_parser_removes_only_link_verified_trailing_footnote(self) -> None:
        source = UniverseSource(
            "YieldMax",
            "https://yieldmaxetfs.com/our-etfs/",
            source_type="issuer_etf",
            parser="yieldmax_html",
        )
        html = """
            <table>
                <tr><th>Ticker</th><th>Name</th></tr>
                <tr>
                    <td><a href="https://yieldmaxetfs.com/our-etfs/ybit/">YBIT*</a></td>
                    <td>YieldMax Bitcoin Option Income Strategy ETF</td>
                </tr>
                <tr>
                    <td><a href="https://yieldmaxetfs.com/our-etfs/nvdy/">NVDY</a></td>
                    <td>YieldMax NVDA Option Income Strategy ETF</td>
                </tr>
            </table>
        """

        out = _workflow_issuer_source_to_universe(html, source, require_leveraged=False)

        self.assertEqual(out["symbol"].tolist(), ["YBIT", "NVDY"])

    def test_yieldmax_parser_rejects_unverified_or_broad_footnote_cleanup(self) -> None:
        source = UniverseSource(
            "YieldMax",
            "https://yieldmaxetfs.com/our-etfs/",
            source_type="issuer_etf",
            parser="yieldmax_html",
        )
        invalid_ticker_cells = (
            '<a href="https://yieldmaxetfs.com/our-etfs/ibit/">YBIT*</a>',
            '<a href="https://attacker.test/our-etfs/ybit/">YBIT*</a>',
            '<a href="https://yieldmaxetfs.com/our-etfs/ybit/">YBIT**</a>',
        )
        for ticker_cell in invalid_ticker_cells:
            html = f"""
                <table>
                    <tr><th>Ticker</th><th>Name</th></tr>
                    <tr><td>{ticker_cell}</td><td>YieldMax Bitcoin Option Income Strategy ETF</td></tr>
                </table>
            """
            with self.subTest(ticker_cell=ticker_cell), self.assertRaises(ValueError):
                _workflow_issuer_source_to_universe(html, source, require_leveraged=False)

    def test_defiance_json_parser_rejects_any_malformed_structured_record(self) -> None:
        source = UniverseSource(
            "Defiance",
            "https://example.test",
            source_type="issuer_etf",
            parser="defiance_json",
        )
        malformed_payloads = {
            "invalid JSON": "{",
            "wrong top level": '{"ticker":"TQQQ"}',
            "non-object record": '[{"ticker":"TQQQ","name":"2X QQQ ETF"}, "bad"]',
            "missing ticker": '[{"name":"2X QQQ ETF"}]',
            "invalid ticker": '[{"ticker":"TQQQ..","name":"2X QQQ ETF"}]',
            "noncanonical ticker": '[{"ticker":" tqqq ","name":"2X QQQ ETF"}]',
            "missing name": '[{"ticker":"TQQQ"}]',
            "non-text name": '[{"ticker":"TQQQ","name":{"text":"2X QQQ ETF"}}]',
            "empty name": '[{"ticker":"TQQQ","name":"   "}]',
            "duplicate ticker key": ('[{"ticker":"TQQQ","ticker":"SQQQ","name":"2X QQQ ETF"}]'),
            "duplicate name key": ('[{"ticker":"TQQQ","name":"2X QQQ ETF","name":"3X QQQ ETF"}]'),
            "nested NaN": '[{"ticker":"TQQQ","name":"2X QQQ ETF","metadata":{"value":NaN}}]',
            "nested positive Infinity": ('[{"ticker":"TQQQ","name":"2X QQQ ETF","metadata":{"value":Infinity}}]'),
            "nested negative Infinity": ('[{"ticker":"TQQQ","name":"2X QQQ ETF","metadata":{"value":-Infinity}}]'),
            "overflowed numeric literal": ('[{"ticker":"TQQQ","name":"2X QQQ ETF","metadata":{"value":1e10000}}]'),
        }

        for label, json_text in malformed_payloads.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                _defiance_json_to_universe(json_text, source)

    def test_strict_json_decoder_rejects_excessive_nesting_before_decoding(self) -> None:
        nesting_depth = 10_000
        json_text = "[" * nesting_depth + "0" + "]" * nesting_depth

        with (
            patch("leveraged_trader.universe.json.loads") as mock_loads,
            self.assertRaisesRegex(ValueError, "supported JSON nesting depth of 128"),
        ):
            _strict_json_loads(
                json_text,
                source_description="Test source",
                invalid_json_message="Test source returned invalid JSON.",
            )

        mock_loads.assert_not_called()

    def test_strict_json_decoder_rejects_excessive_structure_before_decoding(self) -> None:
        with (
            patch("leveraged_trader.universe._MAX_JSON_STRUCTURAL_TOKENS", 4),
            patch("leveraged_trader.universe.json.loads") as mock_loads,
            self.assertRaisesRegex(ValueError, "supported JSON structural limit of 4 tokens"),
        ):
            _strict_json_loads(
                "[{}, {}, {}]",
                source_description="Test source",
                invalid_json_message="Test source returned invalid JSON.",
            )

        mock_loads.assert_not_called()

        quoted_structure = json.dumps({"value": "{},:[]" * 100})
        with patch("leveraged_trader.universe._MAX_JSON_STRUCTURAL_TOKENS", 2):
            self.assertEqual(
                _strict_json_loads(
                    quoted_structure,
                    source_description="Test source",
                    invalid_json_message="Test source returned invalid JSON.",
                ),
                {"value": "{},:[]" * 100},
            )

    def test_strict_json_decoder_accepts_supported_nesting_limit(self) -> None:
        nesting_depth = 128
        json_text = "[" * nesting_depth + "0" + "]" * nesting_depth

        decoded = _strict_json_loads(
            json_text,
            source_description="Test source",
            invalid_json_message="Test source returned invalid JSON.",
        )

        for _depth in range(nesting_depth):
            self.assertIsInstance(decoded, list)
            decoded = decoded[0]
        self.assertEqual(decoded, 0)

    def test_strict_json_nesting_guard_ignores_escaped_string_content(self) -> None:
        string_value = ('[{"\\\\' * 1_000) + ('\\\\"}]' * 1_000)
        json_text = json.dumps({"text": string_value})

        decoded = _strict_json_loads(
            json_text,
            source_description="Test source",
            invalid_json_message="Test source returned invalid JSON.",
        )

        self.assertEqual(decoded, {"text": string_value})

    def test_defiance_json_parser_rejects_conflicting_normalized_ticker_records(self) -> None:
        source = UniverseSource(
            "Defiance",
            "https://example.test",
            source_type="issuer_etf",
            parser="defiance_json",
        )
        conflicting_json = """
            [
                {"ticker": "BRKB", "name": "Defiance 2X Long Berkshire Daily ETF"},
                {"ticker": "BRK.B", "name": "Defiance 2X Short Berkshire Daily ETF"}
            ]
        """

        with self.assertRaisesRegex(ValueError, "conflicting records for normalized ticker BRK-B"):
            _defiance_json_to_universe(conflicting_json, source)

        canonical_duplicate_json = """
            [
                {"ticker": "BRKB", "name": "Defiance 2X Long Berkshire Daily ETF"},
                {"ticker": "BRK.B", "name": "Defiance 2X Long Berkshire Daily ETF"}
            ]
        """
        out = _defiance_json_to_universe(canonical_duplicate_json, source)

        self.assertEqual(
            out[["symbol", "name"]].to_dict("records"),
            [
                {
                    "symbol": "BRK-B",
                    "name": "Defiance 2X Long Berkshire Daily ETF",
                }
            ],
        )

    def test_quoted_javascript_decoder_handles_complete_escape_forms_and_surrogates(self) -> None:
        encoded = r"A\/B\\C\'D\"E\x26\u0026\u{1F680}\uD83D\uDE80"

        self.assertEqual(_decode_quoted_js_text(encoded), "A/B\\C'D\"E&&🚀🚀")
        self.assertEqual(_decode_quoted_js_text("left\\\nright"), "leftright")
        self.assertEqual(_decode_quoted_js_text("left\\\r\nright"), "leftright")

    def test_quoted_javascript_decoder_rejects_invalid_unicode_and_legacy_escapes(self) -> None:
        invalid_values = (
            "trailing\\",
            r"\x0",
            r"\xGG",
            r"\u123",
            r"\uZZZZ",
            r"\u{}",
            r"\u{1234567}",
            r"\u{110000}",
            r"\u{D800}",
            r"\uD800",
            r"\uDC00",
            r"\uD800\u0041",
            r"\1",
            r"\07",
            "raw\nline",
            "\ud800",
        )

        for value in invalid_values:
            with self.subTest(value=ascii(value)), self.assertRaises(ValueError):
                _decode_quoted_js_text(value)

    def test_script_ticker_name_parser_decodes_whitespace_before_mapping_and_rejects_controls(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        html = r"<script>{ticker:'ANUU',name:'2x Long Anthropic\u00a0Daily ETF'}</script>"

        out = _js_ticker_name_to_universe(html, source)
        candidates = _workflow_candidates(out, {"ANUU"}, workflow_label="Long")

        self.assertEqual(out.loc[0, "name"], "2x Long Anthropic Daily ETF")
        self.assertEqual(candidates.loc[0, "mapping_source"], "unresolved_single_stock")
        self.assertEqual(candidates.loc[0, "confidence"], "needs_review")

        unsafe_html = r"<script>{ticker:'BAD',name:'2x Long Bad\u0000 Daily ETF'}</script>"
        with self.assertRaisesRegex(ValueError, "unsafe control"):
            _js_ticker_name_to_universe(unsafe_html, source)

    def test_script_ticker_name_parser_handles_name_before_and_after_ticker(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://example.test",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        html = """
            <script>
            window.featuredEtfData = [
                {"name":"Leverage Shares 2x Long NVDA Daily ETF","ticker":"NVDG"},
                { ticker: 'AALG', fund: " 2x Long AAL Daily ETF" },
                { ticker: 'QUOT', fund: "2x Long Bob\\"s Daily ETF" },
                { ticker: 'APOS', fund: '2x Long Bob\\'s Daily ETF' },
                { ticker: 'PLAIN', fund: "Plain Equity ETF" }
            ];
            </script>
        """

        out = _js_ticker_name_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["NVDG", "AALG", "QUOT", "APOS"])
        self.assertIn("2x Long NVDA", out.loc[out["symbol"] == "NVDG", "name"].item())
        self.assertEqual(out.loc[out["symbol"] == "QUOT", "name"].item(), '2x Long Bob"s Daily ETF')
        self.assertEqual(out.loc[out["symbol"] == "APOS", "name"].item(), "2x Long Bob's Daily ETF")

    def test_script_ticker_name_parser_requires_complete_string_literal_values(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        composed_fields = (
            "{ticker:'FAKE' && 'REAL',fund:'2x Long QQQ Daily ETF'}",
            "{ticker:'REAL',fund:'2x Long QQQ Daily ETF' && 'Plain Equity ETF'}",
        )

        for object_literal in composed_fields:
            with self.subTest(object_literal=object_literal), self.assertRaises(ValueError):
                _js_ticker_name_to_universe(f"<script>{object_literal}</script>", source)

    def test_script_ticker_name_parser_preserves_javascript_property_case(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )

        uppercase_fields = _js_ticker_name_to_universe(
            "<script>{TICKER:'FAKE',FUND:'2x Long QQQ Daily ETF'}</script>",
            source,
        )
        self.assertTrue(uppercase_fields.empty)

        with self.assertRaisesRegex(ValueError, "omitted its required fund/name field"):
            _js_ticker_name_to_universe(
                "<script>{ticker:'REAL',FUND:'2x Long QQQ Daily ETF'}</script>",
                source,
            )

    def test_script_ticker_name_parser_ignores_unrelated_dynamic_script(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        html = """
            <script>{ticker:'REAL',fund:'2x Long QQQ Daily ETF'}</script>
            <script>
                const first = values[0];
                const keys = Object.keys(item);
                const value = Reflect.get(item, 'unrelated');
                console.log(user.name);
                const spread = {...defaults};
                const shorthand = {name};
                const tickerShorthand = {ticker};
                const external = {externalLink: true};
            </script>
        """

        out = _js_ticker_name_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["REAL"])

        global_inventory_with_unrelated_window_access = """
            <script>window.rows=[{ticker:'REAL',fund:'2x Long QQQ Daily ETF'}];</script>
            <script>const locationName = window.location;</script>
        """
        out = _js_ticker_name_to_universe(global_inventory_with_unrelated_window_access, source)
        self.assertEqual(out["symbol"].tolist(), ["REAL"])

        live_shaped_harmless_analytics = """
            <script>window.rows=[{ticker:'REAL',fund:'2x Long QQQ Daily ETF'}];</script>
            <script>
                (function (analyticsWindow, analyticsDocument, tagName, layerName) {
                    analyticsWindow[layerName] = analyticsWindow[layerName] || [];
                    analyticsWindow[layerName].push({event: 'page_view'});
                })(window, document, 'script', 'dataLayer');
                setTimeout(flushAnalytics, 100);
                const availableRoots = [globalThis, self, this];
            </script>
        """
        out = _js_ticker_name_to_universe(live_shaped_harmless_analytics, source)
        self.assertEqual(out["symbol"].tolist(), ["REAL"])

        static_external_link_comment = """
            <script>
                window.rows = [
                    {ticker:'REAL',fund:'2x Long QQQ Daily ETF',externalLink:false/* retained */}
                ];
            </script>
        """
        out = _js_ticker_name_to_universe(static_external_link_comment, source)
        self.assertEqual(out["symbol"].tolist(), ["REAL"])

    def test_script_ticker_name_parser_requires_declarative_ticker_script(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        record = "{ticker:'REAL',fund:'2x Long QQQ Daily ETF'}"
        mutating_scripts = (
            f"const rows=[{record}];rows.pop();",
            f"const rows=[{record}];rows.length=0;",
            f"const rows=[{record}];rows.splice(0,1);",
            f"let rows=[{record}];rows=[];",
            f"let row={record};row=null;",
            f"const rows=[foo(),{record}];",
            f"const rows=[...(false?[]:[{record}])];",
            f"const rows=[false && {record}];",
            "const row={ticker:'REAL',fund:'2x Long QQQ Daily ETF',helper:foo()};",
            "const row={ticker:'REAL',fund:'2x Long QQQ Daily ETF',helper:+1};",
            "const row={ticker:'REAL',fund:'2x Long QQQ Daily ETF',helper:-1};",
            "const row={ticker:'REAL',fund:'2x Long QQQ Daily ETF',helper:1+1};",
            "const row={ticker:'REAL',fund:'2x Long QQQ Daily ETF',helper:!false};",
            "const row={ticker:'REAL',fund:'2x Long QQQ Daily ETF',helper:'raw\nline'};",
            r"const row={ticker:'REAL',fund:'2x Long QQQ Daily ETF',helper:'\xGG'};",
            r"const row={ticker:'REAL',fund:'2x Long QQQ Daily ETF',helper:'\u{110000}'};",
            f"const null={record};",
            f"const false={record};",
            f"const this={record};",
            f"const for={record};",
            f"const class={record};",
            f"const await={record};",
            f"let rows=[{record}];// hidden\u2028rows.pop();",
            f"let rows=[{record}];// hidden\u2029rows.pop();",
            f"let rows=[{record}];// hidden\u2028rows.length=0;",
            f"let rows=[{record}];// hidden\u2029rows.length=0;",
            f"let rows=[{record}];// hidden\u2028rows=[];",
            f"let rows=[{record}];// hidden\u2029rows=[];",
            f"var rows=[{record}];var rows=[];",
            f"window.products=[{record}];globalThis.products=[];",
            f"var rows=[{record}];window.rows=[];",
            f"window.rows=[{record}];var rows=[];",
        )

        for script in mutating_scripts:
            with (
                self.subTest(script=script),
                self.assertRaisesRegex(ValueError, "non-declarative ticker script"),
            ):
                _js_ticker_name_to_universe(f"<script>{script}</script>", source)

        declarative_scripts = (
            record,
            f"const row={record};",
            f"window.products=[{record}];",
            f"/* static issuer data */ const row={record}; // retained\n",
        )
        for script in declarative_scripts:
            with self.subTest(script=script):
                out = _js_ticker_name_to_universe(f"<script>{script}</script>", source)
                self.assertEqual(out["symbol"].tolist(), ["REAL"])

    def test_script_ticker_name_parser_requires_lexically_complete_ticker_script(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        record = "const rows=[{ticker:'REAL',fund:'2x Long QQQ Daily ETF'}];"

        for trailing_source in ("/*", "'unterminated", '"unterminated', "/unterminated"):
            with (
                self.subTest(trailing_source=trailing_source),
                self.assertRaisesRegex(ValueError, "unsupported dynamic product-field syntax"),
            ):
                _js_ticker_name_to_universe(f"<script>{record}{trailing_source}</script>", source)

        out = _js_ticker_name_to_universe(f"<script>{record}// valid through EOF</script>", source)
        self.assertEqual(out["symbol"].tolist(), ["REAL"])

    def test_script_ticker_name_parser_rejects_cross_script_inventory_rebindings(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        record = "{ticker:'REAL',fund:'2x Long QQQ Daily ETF'}"
        scripts = (
            f"<script>var rows=[{record}];</script><script>window.rows=[];</script>",
            f"<script>window.rows=[];</script><script>var rows=[{record}];</script>",
            f"<script>window.products=[{record}];</script><script>self.products=[];</script>",
            f"<script>self.products=[];</script><script>window.products=[{record}];</script>",
            f"<script>let rows=[{record}];</script><script>rows=[];</script>",
            f"<script>const rows=[{record}];</script><script>rows;</script>",
            f'<script>window.rows=[{record}];</script><script>eval("window.rows=[]");</script>',
            f'<script>window.rows=[{record}];</script><script>Function("window.rows=[]")();</script>',
            f'<script>window.rows=[{record}];</script><script>setTimeout("window.rows=[]",0);</script>',
            f"<script>window.rows=[{record}];</script><script>setTimeout(()=>{{window.rows=[];}},0);</script>",
            f"<script>window.rows=[{record}];</script><script>window['ro'+'ws']=[];</script>",
            f"<script>window.rows=[{record}];</script><script>const key='rows';window[key]=[];</script>",
            f"<script>window.rows=[{record}];</script><script>const target=window;target.rows=[];</script>",
            f"<script>window.rows=[{record}];</script>"
            "<script>Object.defineProperty(window,'rows',{value:[]});</script>",
            f"<script>window.rows=[{record}];</script><script>Reflect.set(window,'rows',[]);</script>",
            f"<script>window.rows=[{record}];</script><script>Object.assign(window,{{rows:[]}});</script>",
        )

        for html in scripts:
            with (
                self.subTest(html=html),
                self.assertRaisesRegex(ValueError, "non-declarative ticker script"),
            ):
                _js_ticker_name_to_universe(html, source)

    def test_script_ticker_name_parser_rejects_cross_script_prototype_field_synthesis(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        record = "<script>{ticker:'REAL',fund:'2x Long QQQ Daily ETF'}</script>"
        mutation_scripts = (
            "const key='externalLink';Object.defineProperty(Object.prototype,key,{value:true});",
            "const key='externalLink';Reflect.set(Object.prototype,key,true);",
            "Object.setPrototypeOf(Object.prototype,{externalLink:true});",
            "Reflect.setPrototypeOf(Object.prototype,{externalLink:true});",
            "const key='externalLink';Object.prototype[key]=true;",
            "Object.prototype.externalLink=true;",
            "const key='externalLink';Object['prototype'][key]=true;",
            "(Object).prototype.externalLink=true;",
            "Object['pro'+'totype'].externalLink=true;",
            "(Object)['pro'+'totype'].externalLink=true;",
            "Object.prototype.__defineGetter__('externalLink',()=>true);",
            "const O=Object;const key='externalLink';O.defineProperty(Object.prototype,key,{value:true});",
            "const O=Object,p=O.prototype,k='externalLink';O.defineProperty(p,k,{value:true});",
            "const O=Object,p=O.prototype,k='externalLink';p[k]=true;",
            "const R=Reflect,p=Object.prototype,k='externalLink';R.set(p,k,true);",
            "const {defineProperty}=Object;defineProperty(Object.prototype,k,{value:true});",
            "const p=Object.getPrototypeOf({});p[k]=true;",
            "const p=({}).__proto__;p[k]=true;",
        )

        for mutation_script in mutation_scripts:
            with (
                self.subTest(mutation_script=mutation_script),
                self.assertRaisesRegex(ValueError, "unsupported dynamic product-field syntax"),
            ):
                _js_ticker_name_to_universe(f"{record}<script>{mutation_script}</script>", source)

        cross_script_alias_mutation = (
            f"{record}<script>const O=Object;</script><script>O.prototype.externalLink=true;</script>"
        )
        with self.assertRaisesRegex(ValueError, "Object.prototype mutation"):
            _js_ticker_name_to_universe(cross_script_alias_mutation, source)

        harmless_read = f"{record}<script>const inherited = Object.prototype.externalLink;</script>"
        out = _js_ticker_name_to_universe(harmless_read, source)
        self.assertEqual(out["symbol"].tolist(), ["REAL"])

        harmless_object_alias = f"{record}<script>const O=Object;O.keys(item);</script>"
        out = _js_ticker_name_to_universe(harmless_object_alias, source)
        self.assertEqual(out["symbol"].tolist(), ["REAL"])

        harmless_shadowed_objects = (
            "function inspect(Object){Object.prototype.externalLink=true;}",
            "{const Object={prototype:{}};Object.prototype.externalLink=true;}",
            "const inspect=(Object)=>{Object.prototype.externalLink=true;};",
        )
        for harmless_script in harmless_shadowed_objects:
            with self.subTest(harmless_script=harmless_script):
                out = _js_ticker_name_to_universe(f"{record}<script>{harmless_script}</script>", source)
                self.assertEqual(out["symbol"].tolist(), ["REAL"])

    def test_script_ticker_name_parser_ignores_external_products_and_validates_the_marker(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        live_shaped_html = """
            <script>
            window.products = [
                {ticker: 'INTERNAL', fund: '2x Long QQQ Daily ETF'},
                {ticker: 'LOCAL', fund: '2x Long SPY Daily ETF', externalLink: false},
                {
                    ticker: 'ELOL',
                    externalLink: true,
                    fund: '100% TSLA & 100% SPCX Daily ETF',
                    product_url: 'https://leverageshares.com/us/etfs/leverage-shares-100-tsla-100-spcx-daily-etf'
                }
            ];
            </script>
        """

        out = _js_ticker_name_to_universe(live_shaped_html, source)

        self.assertEqual(out["symbol"].tolist(), ["INTERNAL", "LOCAL"])
        malformed_markers = (
            "externalLink: true, externalLink: false",
            "externalLink: 'true'",
            "externalLink: true && false",
            "externalLink: externalLink",
        )
        for marker in malformed_markers:
            html = f"<script>{{ticker:'BAD',name:'2x Long QQQ Daily ETF',{marker}}}</script>"
            with (
                self.subTest(marker=marker),
                self.assertRaisesRegex(ValueError, "duplicate or malformed externalLink"),
            ):
                _js_ticker_name_to_universe(html, source)

    def test_script_ticker_name_parser_requires_exact_lowercase_external_link_literals(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )

        for literal in ("TRUE", "FALSE", "True", "False"):
            html = f"<script>{{ticker:'BAD',name:'2x Long QQQ Daily ETF',externalLink:{literal}}}</script>"
            with (
                self.subTest(literal=literal),
                self.assertRaisesRegex(ValueError, "duplicate or malformed externalLink"),
            ):
                _js_ticker_name_to_universe(html, source)

    def test_script_ticker_name_parser_rejects_semantic_external_link_key_bypasses(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        alternate_objects = (
            r"{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF','external\u004cink':true}",
            r"{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF',external\u004cink:true}",
            "{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF',['externalLink']:true}",
            "{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF',...{externalLink:true}}",
            "{externalLink:true,...{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF'}}",
            "{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF',externalLink/*comment*/:true}",
            "{ticker/*comment*/:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF'}",
            "{ticker:'ELOL',fund/*comment*/:'100% TSLA & 100% SPCX Daily ETF'}",
            "{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF','externalLink'/*comment*/:true}",
            "{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF',get externalLink(){return true}}",
            "{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF',get ['externalLink'](){return true}}",
            "{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF',__proto__:{externalLink:true}}",
            "{ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF','__proto__':{externalLink:true}}",
            "{['ticker']:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF'}",
        )

        for object_literal in alternate_objects:
            with (
                self.subTest(object_literal=object_literal),
                self.assertRaisesRegex(ValueError, "unsupported object property syntax"),
            ):
                _js_ticker_name_to_universe(f"<script>{object_literal}</script>", source)

    def test_script_ticker_name_parser_rejects_external_wrapper_around_ticker_record(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        html = """
            <script>
            {
                externalLink: true,
                product: {ticker: 'ELOL', fund: '100% TSLA & 100% SPCX Daily ETF'}
            }
            </script>
        """

        with self.assertRaisesRegex(ValueError, "externalLink field.*required ticker"):
            _js_ticker_name_to_universe(html, source)

        ticker_bearing_wrapper = """
            <script>
            {
                ticker: 'OUT',
                fund: 'External wrapper',
                externalLink: true,
                product: {ticker: 'ELOL', fund: '100% TSLA & 100% SPCX Daily ETF'}
            }
            </script>
        """
        with self.assertRaisesRegex(ValueError, "nested ticker"):
            _js_ticker_name_to_universe(ticker_bearing_wrapper, source)

        non_external_wrapper = ticker_bearing_wrapper.replace("externalLink: true", "externalLink: false")
        with self.assertRaisesRegex(ValueError, "nested ticker"):
            _js_ticker_name_to_universe(non_external_wrapper, source)

    def test_script_ticker_name_parser_rejects_regex_literal_containing_close_brace(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        regex_expressions = (
            "() => /}/",
            "+/}/",
            "-/}/",
            "~/}/",
            "1 + /}/",
            "1 - /}/",
            "1 * /}/",
            "1 % /}/",
            "1 ^ /}/",
            "1 < /}/",
        )
        for expression in regex_expressions:
            html = f"""
                <script>
                {{
                    ticker: 'ELOL',
                    fund: '100% TSLA & 100% SPCX Daily ETF',
                    helper: {expression},
                    externalLink: true
                }}
                </script>
            """

            with (
                self.subTest(expression=expression),
                self.assertRaisesRegex(ValueError, "unsupported dynamic product-field syntax"),
            ):
                _js_ticker_name_to_universe(html, source)

    def test_script_ticker_name_parser_rejects_relevant_member_accesses_and_mutations(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        static_record = "const row={ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF'};"
        dynamic_statements = (
            "row.externalLink=true;",
            r"row.external\u004cink=true;",
            "row['externalLink']=true;",
            "row[('externalLink')]=true;",
            "row[(/*comment*/'externalLink'/*comment*/)]=true;",
            r'row["external\u004cink"]=true;',
            "row.ticker='EVIL';",
            "row['fund']='2x Long QQQ Daily ETF';",
            "row.name += ' forged';",
            "row.externalLink++;",
            "Object.defineProperty(row,'externalLink',{value:true});",
            "Object.defineProperty(row,'ticker',{value:'EVIL'});",
            "Reflect.set(row,'fund','Plain Equity ETF');",
            "Object.defineProperty.call(Object,row,'fund',{value:'Plain Equity ETF'});",
            "Reflect.set.call(Reflect,row,'fund','Plain Equity ETF');",
            "Object.defineProperty['call'](Object,row,'ticker',{value:'EVIL'});",
            "Reflect['set']['call'](Reflect,row,'fund','Plain Equity ETF');",
            "Object.assign(row,{fund:'Plain Equity ETF'});",
            "const O=Object;O.defineProperty(row,'fund',{value:'Plain Equity ETF'});",
            "const R=Reflect;R.set(row,'fund','Plain Equity ETF');",
            "const a=Object;a.assign(row,{fund:'Plain Equity ETF'});",
            "eval(\"row.fund='Plain Equity ETF'\");",
            "Function(\"row.fund='Plain Equity ETF'\")();",
            "globalThis['eval'](\"row.fund='Plain Equity ETF'\");",
            "const F=globalThis['Function'];F(\"row.fund='Plain Equity ETF'\")();",
            "row['fu'+'nd']='Plain Equity ETF';",
            "const field='fund';row[field]='Plain Equity ETF';",
            "`${row.externalLink=true}`;",
        )

        for statement in dynamic_statements:
            with (
                self.subTest(statement=statement),
                self.assertRaisesRegex(ValueError, "unsupported dynamic product-field syntax"),
            ):
                _js_ticker_name_to_universe(f"<script>{static_record}{statement}</script>", source)

        with self.assertRaisesRegex(ValueError, "unsupported dynamic product-field syntax"):
            _js_ticker_name_to_universe(
                f"<script>{static_record}</script><script>window.row.externalLink=true;</script>",
                source,
            )

    def test_script_ticker_name_parser_rejects_ambiguous_executable_syntax(self) -> None:
        source = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        unsafe_scripts = (
            """
                const x=1;
                const row={
                    ticker:'ELOL',
                    fund:'100% TSLA & 100% SPCX Daily ETF',
                    helper:x++ / /}/,
                    externalLink:true
                };
            """,
            """
                let x=1,y=1;
                const row={ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF'};
                x++ / (row.externalLink=true) / y;
            """,
            """
                let of=1;
                const row={
                    ticker:'ELOL',
                    fund:'100% TSLA & 100% SPCX Daily ETF',
                    helper:of / /}+/,
                    externalLink:true
                };
            """,
            """
                const row={ticker:'ELOL',fund:'100% TSLA & 100% SPCX Daily ETF',helper:0,
                <!-- }
                externalLink:true};
            """,
        )

        for script in unsafe_scripts:
            with (
                self.subTest(script=script),
                self.assertRaisesRegex(ValueError, "unsupported dynamic product-field syntax"),
            ):
                _js_ticker_name_to_universe(f"<script>{script}</script>", source)

    def test_external_product_does_not_poison_strict_cross_source_resolution(self) -> None:
        themes = UniverseSource(
            "Themes",
            "https://themesetfs.com/etfs",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        issuer = UniverseSource("Real issuer", "https://issuer.example/products", "issuer_etf")
        themes_rows = _js_ticker_name_to_universe(
            """
            <script>
            window.products = [
                {ticker:'LOCAL', fund:'2x Long QQQ Daily ETF'},
                {ticker:'XLEV', fund:'2x Long QQQ Daily ETF', externalLink:true}
            ];
            </script>
            """,
            themes,
            require_leveraged=False,
        )
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "XLEV",
                    "name": "2x Long SPY Daily ETF",
                    "fund_type": "ETF (Real issuer)",
                    "source": "Real issuer table",
                }
            ]
        )
        statuses = [
            {"source": themes.name, "url": themes.url, "status": "loaded", "error": ""},
            {"source": issuer.name, "url": issuer.url, "status": "loaded", "error": ""},
        ]

        resolved = _resolve_workflow_source_product_rows(
            [(0, themes, themes_rows), (1, issuer, issuer_rows)],
            statuses,
        )

        self.assertEqual(resolved["symbol"].tolist(), ["LOCAL", "XLEV"])
        self.assertEqual(resolved.loc[resolved["symbol"].eq("XLEV"), "name"].item(), "2x Long SPY Daily ETF")
        self.assertEqual([status["status"] for status in statuses], ["loaded", "loaded"])

        resolved.attrs["workflow_source_status"] = statuses
        nasdaq_rows = pd.DataFrame([{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}])
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []
        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=resolved),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"LOCAL", "QQQ", "SPY", "TQQQ", "XLEV"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            groups = determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        self.assertEqual(groups["long"]["symbol"].tolist(), ["LOCAL", "TQQQ", "XLEV"])
        self.assertEqual(groups["long"].attrs["workflow_source_failures"], [])

    def test_script_ticker_name_parser_rejects_duplicate_or_ambiguous_fields(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://example.test",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        malformed_objects = {
            "duplicate ticker": ("{ticker:'WRONG',ticker:'RIGHT',name:'Example 2X Long QQQ Daily ETF'}"),
            "duplicate ticker with non-string decoy": (
                "{ticker:'WRONG',ticker:42,name:'Example 2X Long QQQ Daily ETF'}"
            ),
            "duplicate name": (
                "{ticker:'AAA',name:'Example 2X Long QQQ Daily ETF',name:'Forged 2X Long SPY Daily ETF'}"
            ),
            "ambiguous name aliases": (
                "{ticker:'AAA',name:'Example 2X Long QQQ Daily ETF',fund:'Forged 2X Long SPY Daily ETF'}"
            ),
            "ambiguous name alias with non-string decoy": (
                "{ticker:'AAA',name:'Example 2X Long QQQ Daily ETF',fund:['Forged']}"
            ),
        }

        for label, html in malformed_objects.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "duplicate or ambiguous"):
                _js_ticker_name_to_universe(html, source)

    def test_script_ticker_name_parser_rejects_partial_ticker_record(self) -> None:
        source = UniverseSource(
            "Example",
            "https://example.test",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )

        malformed_records = (
            "<script>{ticker:'MISS'}</script>",
            "<script>{ticker:'MISS', metadata:{active:true}}</script>",
            "<script>{metadata:{active:true}, ticker:'MISS'}</script>",
            "<script>{ticker:'MISS', padding:'" + "x" * 2_600 + "'}</script>",
            "<script>{ticker:'MISS'</script>",
            "<script>{ticker:'GOOD',name:'2x Long Good Daily ETF'};{ticker:'MISS'</script>",
        )
        for html in malformed_records:
            with self.subTest(length=len(html)), self.assertRaisesRegex(ValueError, "ticker"):
                _js_ticker_name_to_universe(html, source)

    def test_leverage_shares_parser_uses_complete_products_inventory(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        html = _leverage_shares_products_page(
            final_overrides={
                "name": "100% TSLA + 100% SPCX Daily ETF",
                "fund": "100% TSLA + 100% SPCX Daily ETF",
                "ticker": "ELOL",
                "product_url": ("https://leverageshares.com/us/etfs/leverage-shares-100-tsla-100-spcx-daily-etf"),
            }
        )

        complete = _leverage_shares_html_to_universe(html, source, require_leveraged=False)
        leveraged = _leverage_shares_html_to_universe(html, source)

        self.assertEqual(len(complete), 100)
        self.assertEqual(len(leveraged), 100)
        self.assertIn("ELOL", set(leveraged["symbol"]))

    def test_leverage_shares_parser_ignores_unrelated_dynamic_script(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        html = (
            _leverage_shares_products_page()
            + """
            <script>
                const first = values[0];
                const keys = Object.keys(item);
                const value = Reflect.get(item, 'unrelated');
            </script>
            <script>
                (function(c, l) {
                    c[l] = c[l] || function() {};
                })(window, 'clarity');
            </script>
        """
        )

        out = _leverage_shares_html_to_universe(html, source)

        self.assertEqual(len(out), 100)

    def test_leverage_shares_parser_respects_script_execution_types(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        inert_scripts = (
            '<script type="text/x-template">window.productsData = [];</script>',
            '<script type="application/json">window.productsData = [];</script>',
            '<script src="https://example.test/ignored.js">window.productsData = [];</script>',
            "<template><script>window.productsData = [];</script></template>",
            "<noscript><script>window.productsData = [];</script></noscript>",
        )

        for inert_script in inert_scripts:
            with self.subTest(inert_script=inert_script):
                out = _leverage_shares_html_to_universe(valid_html + inert_script, source)
                self.assertEqual(len(out), 100)

        with self.assertRaisesRegex(ValueError, "complete product inventory"):
            _leverage_shares_html_to_universe(
                _leverage_shares_products_page().replace("<script>", '<script type="text/x-template">', 1),
                source,
            )
        for inert_container in ("template", "noscript"):
            with (
                self.subTest(inert_container=inert_container),
                self.assertRaisesRegex(ValueError, "complete product inventory"),
            ):
                _leverage_shares_html_to_universe(
                    f"<{inert_container}>{valid_html}</{inert_container}>",
                    source,
                )

        with self.assertRaisesRegex(ValueError, "complete product inventory"):
            _leverage_shares_html_to_universe(valid_html.replace("<script>", "<script>if(false) ", 1), source)

    def test_leverage_shares_parser_follows_html_script_type_and_nomodule_rules(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        executable_mutations = (
            '<script type="\t MoDuLe \r\n">window.productsData=[];</script>',
            '<script type=" TEXT/JAVASCRIPT ">window.productsData=[];</script>',
            '<script language="JavaScript">window.productsData=[];</script>',
            '<script type="module" nomodule>window.productsData=[];</script>',
        )
        for mutation in executable_mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "complete product inventory"):
                _leverage_shares_html_to_universe(valid_html + mutation, source)

        inert_mutations = (
            '<script type="text/javascript; charset=utf-8">window.productsData=[];</script>',
            '<script type="\u00a0text/javascript\u00a0">window.productsData=[];</script>',
            '<script type="text/java\u017fcript">window.productsData=[];</script>',
            '<script type="   ">window.productsData=[];</script>',
            '<script language="json">window.productsData=[];</script>',
            "<script nomodule>window.productsData=[];</script>",
        )
        for mutation in inert_mutations:
            with self.subTest(mutation=mutation):
                out = _leverage_shares_html_to_universe(valid_html + mutation, source)
                self.assertEqual(len(out), 100)

    def test_leverage_shares_parser_respects_module_local_scope(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        harmless_modules = (
            "<script type='module'>const productsData=[];consume(productsData);</script>",
            "<script type='module'>const window={};window.productsData=[];</script>",
            ("<script type='module'>const w=window;</script><script>w['products'+'Data']=[];</script>"),
        )
        for harmless_module in harmless_modules:
            with self.subTest(harmless_module=harmless_module):
                out = _leverage_shares_html_to_universe(valid_html + harmless_module, source)
                self.assertEqual(len(out), 100)

        module_alias_mutations = (
            "<script type='module'>const w=window;w['products'+'Data']=[];</script>",
            (
                "<script type='module'>const w=window;const O=Object;"
                "O.defineProperty(w,'productsData',{value:[]});</script>"
            ),
        )
        for mutation in module_alias_mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "complete product inventory"):
                _leverage_shares_html_to_universe(valid_html + mutation, source)

        shadowed_inventory = _leverage_shares_products_page().replace(
            "<script>",
            "<script type='module'>const window={};",
            1,
        )
        with self.assertRaisesRegex(ValueError, "complete product inventory"):
            _leverage_shares_html_to_universe(shadowed_inventory, source)

    def test_leverage_shares_parser_respects_classic_lexical_shadowing(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        harmless_scripts = (
            "<script>const productsData=[];consume(productsData);</script>",
            "<script>const window={};window.productsData=[];</script>",
        )
        for harmless_script in harmless_scripts:
            with self.subTest(harmless_script=harmless_script):
                out = _leverage_shares_html_to_universe(valid_html + harmless_script, source)
                self.assertEqual(len(out), 100)

        shadowed_inventory = _leverage_shares_products_page().replace(
            "<script>",
            "<script>const window={};",
            1,
        )
        with self.assertRaisesRegex(ValueError, "complete product inventory"):
            _leverage_shares_html_to_universe(shadowed_inventory, source)

        destructured_shadowed_inventory = _leverage_shares_products_page().replace(
            "<script>",
            "<script>const {window}={window:{}};",
            1,
        )
        with self.assertRaisesRegex(ValueError, "complete product inventory"):
            _leverage_shares_html_to_universe(destructured_shadowed_inventory, source)

    def test_leverage_shares_parser_rejects_template_interpolation_mutations(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        template_mutations = (
            "<script>`${window.productsData=[]}`;</script>",
            "<script>tag`${window['products'+'Data'].pop()}`;</script>",
        )
        for mutation in template_mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "complete product inventory"):
                _leverage_shares_html_to_universe(valid_html + mutation, source)

        out = _leverage_shares_html_to_universe(
            valid_html + r"<script>const label=`\${productsData}`;</script>",
            source,
        )
        self.assertEqual(len(out), 100)

    def test_leverage_shares_parser_rejects_bulk_global_inventory_mutations(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        bulk_mutations = (
            "<script>Object.assign(window,{'productsData':[]});</script>",
            "<script>Object.defineProperties(window,{'productsData':{value:[]}});</script>",
            "<script>const O=Object;O.assign(window,{['products'+'Data']:[]});</script>",
        )
        for mutation in bulk_mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "complete product inventory"):
                _leverage_shares_html_to_universe(valid_html + mutation, source)

        bulk_alias_mutations = (
            (
                "<script>const a=Object.assign;</script>",
                "<script>a(window,{'productsData':[]});</script>",
            ),
            (
                "<script>const d=Object.defineProperties;</script>",
                "<script>d(window,{'productsData':{value:[]}});</script>",
            ),
        )
        for alias_declaration, mutation in bulk_alias_mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "complete product inventory"):
                _leverage_shares_html_to_universe(alias_declaration + valid_html + mutation, source)

        harmless_mutations = (
            "<script>Object.assign(window,{'unrelated':[]});</script>",
            "<script>Object.assign(localObject,{'productsData':[]});</script>",
        )
        for mutation in harmless_mutations:
            with self.subTest(mutation=mutation):
                out = _leverage_shares_html_to_universe(valid_html + mutation, source)
                self.assertEqual(len(out), 100)

    def test_leverage_shares_parser_accepts_ticker_named_fund_with_corroborated_underlying_route(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        sk_hynix_product = {
            "name": "2x Short SKHQ Daily ETF",
            "fund": "2x Short SKHQ Daily ETF",
            "ticker": "SKHQ",
            "product_url": "https://leverageshares.com/us/etfs/leverage-shares-2x-short-sk-hynix-daily-etf",
            "leverage_factor": "-2",
            "category": "Inverse",
        }

        with self.assertRaisesRegex(ValueError, "contradicted its product URL"):
            _leverage_shares_html_to_universe(
                _leverage_shares_products_page(final_overrides=sk_hynix_product),
                source,
            )

        sk_hynix_product["category2"] = "SKHY"
        out = _leverage_shares_html_to_universe(
            _leverage_shares_products_page(final_overrides=sk_hynix_product),
            source,
        )

        skhq = out.loc[out["symbol"].eq("SKHQ")].iloc[0]
        self.assertEqual(skhq["name"], "2x Short SKHQ Daily ETF")
        self.assertEqual(skhq["reference_security"], "SKHY")
        mapped = _workflow_candidates(out.loc[out["symbol"].eq("SKHQ")], {"SKHQ", "SKHY"}, workflow_label="Short")
        self.assertEqual(mapped.loc[0, "rsi_symbol"], "SKHY")
        self.assertEqual(mapped.loc[0, "mapping_source"], "issuer_reference")
        self.assertEqual(mapped.loc[0, "confidence"], "curated")

        sk_hynix_product["category2"] = "NVDA"
        with self.assertRaisesRegex(ValueError, "contradicted its product URL"):
            _leverage_shares_html_to_universe(
                _leverage_shares_products_page(final_overrides=sk_hynix_product),
                source,
            )

    def test_leverage_shares_parser_fails_closed_on_incomplete_product_record(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        html = _leverage_shares_products_page(final_overrides={"name": None, "fund": None})

        with self.assertRaisesRegex(ValueError, "literal name"):
            _leverage_shares_html_to_universe(html, source)

    def test_leverage_shares_parser_requires_complete_false_boolean_literals(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        malformed_fields = (
            ("is_api_product", "false || true"),
            ("externalLink", "false || true"),
            ("is_api_product", "False"),
            ("externalLink", "FALSE"),
        )

        for field_name, expression in malformed_fields:
            html = valid_html.replace(f'"{field_name}": false', f'"{field_name}": {expression}', 1)
            with (
                self.subTest(field_name=field_name, expression=expression),
                self.assertRaisesRegex(ValueError, rf"{field_name}=false"),
            ):
                _leverage_shares_html_to_universe(html, source)

    def test_leverage_shares_parser_requires_complete_unique_category2_literal(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        sk_hynix_product = {
            "name": "2x Short SKHQ Daily ETF",
            "fund": "2x Short SKHQ Daily ETF",
            "ticker": "SKHQ",
            "product_url": "https://leverageshares.com/us/etfs/leverage-shares-2x-short-sk-hynix-daily-etf",
            "leverage_factor": "-2",
            "category": "Inverse",
            "category2": "SKHY",
        }
        valid_html = _leverage_shares_products_page(final_overrides=sk_hynix_product)
        malformed_category2_values = (
            '"SKHY" && "NVDA"',
            '"SKHY", "category2": "NVDA"',
        )

        for expression in malformed_category2_values:
            html = valid_html.replace('"category2": "SKHY"', f'"category2": {expression}', 1)
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                _leverage_shares_html_to_universe(html, source)

    def test_leverage_shares_parser_rejects_ambiguous_name_declarations(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        ambiguous_products = (
            {
                "name": "2x 3x Long ASSET99 Daily ETF",
                "fund": "2x 3x Long ASSET99 Daily ETF",
            },
            {
                "name": "2x Long Inverse ASSET99 Daily ETF",
                "fund": "2x Long Inverse ASSET99 Daily ETF",
                "product_url": ("https://leverageshares.com/us/etfs/leverage-shares-2x-short-asset99-daily-etf"),
                "leverage_factor": "-2",
                "category": "Inverse",
            },
            {
                "name": "+2 x Short ASSET99 Daily ETF",
                "fund": "+2 x Short ASSET99 Daily ETF",
                "product_url": ("https://leverageshares.com/us/etfs/leverage-shares-2x-short-asset99-daily-etf"),
                "leverage_factor": "-2",
                "category": "Inverse",
            },
            {
                "name": "+200% Short ASSET99 Daily ETF",
                "fund": "+200% Short ASSET99 Daily ETF",
                "product_url": ("https://leverageshares.com/us/etfs/leverage-shares-2x-short-asset99-daily-etf"),
                "leverage_factor": "-2",
                "category": "Inverse",
            },
            {
                "name": "+4x Short ASSET99 Daily ETF",
                "fund": "+4x Short ASSET99 Daily ETF",
                "product_url": ("https://leverageshares.com/us/etfs/leverage-shares-2x-short-asset99-daily-etf"),
                "leverage_factor": "-2",
                "category": "Inverse",
            },
            {
                "name": "+ 2x Short ASSET99 Daily ETF",
                "fund": "+ 2x Short ASSET99 Daily ETF",
                "product_url": ("https://leverageshares.com/us/etfs/leverage-shares-2x-short-asset99-daily-etf"),
                "leverage_factor": "-2",
                "category": "Inverse",
            },
            {
                "name": "+\u00a02x Short ASSET99 Daily ETF",
                "fund": "+\u00a02x Short ASSET99 Daily ETF",
                "product_url": ("https://leverageshares.com/us/etfs/leverage-shares-2x-short-asset99-daily-etf"),
                "leverage_factor": "-2",
                "category": "Inverse",
            },
            {
                "name": "＋2x Short ASSET99 Daily ETF",
                "fund": "＋2x Short ASSET99 Daily ETF",
                "product_url": ("https://leverageshares.com/us/etfs/leverage-shares-2x-short-asset99-daily-etf"),
                "leverage_factor": "-2",
                "category": "Inverse",
            },
        )

        for product in ambiguous_products:
            html = _leverage_shares_products_page(final_overrides=product)
            with (
                self.subTest(name=product["name"]),
                self.assertRaisesRegex(ValueError, "contradictory leverage or direction declarations"),
            ):
                _leverage_shares_html_to_universe(html, source)

    def test_leverage_shares_parser_rejects_conflicting_or_duplicated_inventory(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        malformed_overrides = {
            "conflicting aliases": {"fund": "2x Short ASSET99 Daily ETF"},
            "URL contradiction": {
                "product_url": ("https://leverageshares.com/us/etfs/leverage-shares-2x-short-asset99-daily-etf")
            },
            "duplicate ticker": {"ticker": "X000"},
            "factor contradiction": {"leverage_factor": "3"},
            "category contradiction": {"category": "Inverse"},
            "API product": {"is_api_product": True},
            "external product": {"externalLink": True},
            "unsafe URL": {
                "product_url": (" https://leverageshares.com/us/etfs/leverage-shares-2x-long-asset99-daily-etf")
            },
        }

        for label, overrides in malformed_overrides.items():
            html = _leverage_shares_products_page(final_overrides=overrides)
            with self.subTest(label=label), self.assertRaises(ValueError):
                _leverage_shares_html_to_universe(html, source)

        with self.assertRaisesRegex(ValueError, "at least 100"):
            _leverage_shares_html_to_universe(_leverage_shares_products_page(count=99), source)

        valid_html = _leverage_shares_products_page()
        malformed_arrays = (
            valid_html.replace("[{", "[,,{", 1),
            valid_html.replace("}, {", "},, {", 1),
            valid_html.replace("}];", "},,,];", 1),
            valid_html.replace(
                '"is_api_product": false',
                '"is_api_product": false, "is_api_product": "true"',
                1,
            ),
            valid_html.replace("];", "].concat([]);", 1),
            valid_html.replace(
                '"externalLink": false',
                '"externalLink": false, ...{"ticker": "EVIL"}',
                1,
            ),
            valid_html.replace(
                '"ticker": "X000"',
                '"ticker": "X000", ["ticker"]: "EVIL"',
                1,
            ),
            valid_html.replace(
                '"ticker": "X000"',
                '"ticker": "X000", get ticker(){ return "EVIL"; }',
                1,
            ),
            valid_html.replace(
                "];</script>",
                "]; window.productsData.push({ticker: 'EVIL'});</script>",
                1,
            ),
            valid_html.replace(
                "];</script>",
                "]; window.productsData[0].ticker = 'EVIL';</script>",
                1,
            ),
            valid_html + "<script>window.productsData.push({ticker: 'EVIL'});</script>",
            valid_html + "<script>productsData.push({ticker: 'EVIL'});</script>",
            valid_html + "<script>window['productsData'].push({ticker: 'EVIL'});</script>",
            valid_html + "<script>globalThis.productsData.push({ticker: 'EVIL'});</script>",
            valid_html + "<script>self.productsData[0].ticker = 'EVIL';</script>",
            valid_html + "<script>setTimeout(() => { window.productsData.push({ticker: 'EVIL'}); }, 0);</script>",
            valid_html + "<script>(() => { productsData[0].ticker = 'EVIL'; })();</script>",
            valid_html + "<script>function mutate(){ globalThis.productsData.pop(); } mutate();</script>",
            valid_html + r"<script>window['products\u0044ata'] = [];</script>",
            valid_html + r"<script>window.produc\u0074sData = [];</script>",
            valid_html + "<script>Object.defineProperty(window,'productsData',{value:[]});</script>",
            valid_html + "<script>window['products'+'Data'] = [];</script>",
            valid_html + "<script>const p='products'+'Data';window[p] = [];</script>",
            valid_html + "<script>Object.defineProperty.call(Object,window,'productsData',{value:[]});</script>",
            valid_html + "<script>Reflect.set.call(Reflect,window,'productsData',[]);</script>",
            valid_html + "<script>Object.defineProperty['call'](Object,window,'productsData',{value:[]});</script>",
            valid_html + "<script>Reflect['set']['call'](Reflect,window,'productsData',[]);</script>",
            valid_html + "<script>const O=Object;O.defineProperty(window,'productsData',{value:[]});</script>",
            valid_html + "<script>const R=Reflect;R.set(window,'productsData',[]);</script>",
            valid_html + "<script>const a=Object;a.assign(window,{productsData:[]});</script>",
            valid_html + '<script>eval("window.productsData=[]");</script>',
            valid_html + '<script>Function("window.productsData=[]")();</script>',
            valid_html + '<script>globalThis["eval"]("window.productsData=[]");</script>',
            valid_html + '<script>const F=globalThis["Function"];F("window.productsData=[]")();</script>',
            valid_html.replace("window.productsData", "WINDOW.PRODUCTSDATA", 1),
        )
        for html in malformed_arrays:
            with self.subTest(malformed_array=html[:120]), self.assertRaises(ValueError):
                _leverage_shares_html_to_universe(html, source)

    def test_leverage_shares_parser_accepts_unrelated_timer_and_global_references(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        harmless_scripts = (
            "<script>setTimeout(noop, 0);</script>",
            "<script>setTimeout(() => noop(window), 0);</script>",
            "<script>noop(window);</script>",
        )

        for harmless_script in harmless_scripts:
            with self.subTest(script=harmless_script):
                result = _leverage_shares_html_to_universe(valid_html + harmless_script, source)
                self.assertEqual(len(result), 100)

    def test_leverage_shares_parser_rejects_cross_script_global_and_mutator_aliases(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        aliased_mutations = (
            ("<script>const w=window;</script>" + valid_html + "<script>w['products'+'Data']=[];</script>"),
            (
                "<script>const w=window;</script><script>const O=Object;</script>"
                + valid_html
                + "<script>O.defineProperty(w,'productsData',{value:[]});</script>"
            ),
            (
                "<script>const d=Object.defineProperty;</script>"
                + valid_html
                + "<script>d(window,'products'+'Data',{value:[]});</script>"
            ),
            (valid_html + "<script>setTimeout(()=>{const w=window;w['products'+'Data']=[]},0);</script>"),
        )

        for html in aliased_mutations:
            with self.subTest(html=html[-120:]), self.assertRaisesRegex(ValueError, "complete product inventory"):
                _leverage_shares_html_to_universe(html, source)

    def test_leverage_shares_parser_rejects_extended_global_and_mutator_aliases(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        mutations = (
            "<script>const w=(window);w['products'+'Data']=[];</script>",
            "<script>const w=((window));w['products'+'Data']=[];</script>",
            "<script>const w=globalThis.window;w['products'+'Data']=[];</script>",
            "<script>const {window:w}=globalThis;w['products'+'Data']=[];</script>",
            "<script>const {defineProperty:d}=Object;d(window,'productsData',{value:[]});</script>",
            "<script>if(true){} const w=window;w['products'+'Data']=[];</script>",
            "<script>const d=(Object.defineProperty);d(window,'productsData',{value:[]});</script>",
            "<script>const d=((Reflect.deleteProperty));d(window,'productsData');</script>",
            "<script>Object['define'+'Property'](window,'productsData',{value:[]});</script>",
            "<script>Object.defineProperty(globalThis.window,'productsData',{value:[]});</script>",
            "<script>Reflect.deleteProperty(window,'productsData');</script>",
            "<script>Reflect['delete'+'Property'](window,'productsData');</script>",
            "<script>const d=Reflect.deleteProperty;</script><script>d(window,'productsData');</script>",
            "<script>const {deleteProperty:d}=Reflect;</script><script>d(window,'productsData');</script>",
            "<script>Object.defineProperty.apply(Object,[window,'productsData',{value:[]}]);</script>",
            "<script>const d=Object.defineProperty;d.apply(Object,[window,'productsData',{value:[]}]);</script>",
            "<script>(Reflect.deleteProperty)(window,'productsData');</script>",
            "<script>(0,Reflect.deleteProperty)(window,'productsData');</script>",
            "<script>(Reflect.deleteProperty).call(Reflect,window,'productsData');</script>",
            "<script>Reflect.deleteProperty.bind(Reflect)(window,'productsData');</script>",
            "<script>Reflect?.deleteProperty(window,'productsData');</script>",
            "<script>Reflect.apply(Reflect.deleteProperty,Reflect,[window,'productsData']);</script>",
            "<script>const O=globalThis.Object;O.defineProperty(window,'productsData',{value:[]});</script>",
            "<script>const R=window.Reflect;R.deleteProperty(window,'productsData');</script>",
            "<script>const d=window.Reflect.deleteProperty;d(window,'productsData');</script>",
            "<script>window.q=window;q['products'+'Data']=[];</script>",
            "<script>window.d=Reflect.deleteProperty;d(window,'productsData');</script>",
            "<script>window.__proto__.productsData=[];</script>",
            "<script>Object.defineProperty(window.__proto__,'productsData',{value:[]});</script>",
            (
                "<script>const d=(0,Reflect.deleteProperty);</script>"
                + valid_html
                + "<script>d(window,'productsData');</script>"
            ),
            (
                "<script>const d=Reflect.deleteProperty.bind(Reflect);</script>"
                + valid_html
                + "<script>d(window,'productsData');</script>"
            ),
            (
                "<script>const a=Reflect.apply;</script>"
                + valid_html
                + "<script>a(Reflect.deleteProperty,Reflect,[window,'productsData']);</script>"
            ),
            "<script type='module'>w['products'+'Data']=[];</script>" + valid_html + "<script>const w=window;</script>",
            "<script type='module'>d(window,'productsData',{value:[]});</script>"
            + valid_html
            + "<script>const d=Object.defineProperty;</script>",
        )
        for mutation in mutations:
            html = mutation if valid_html in mutation else valid_html + mutation
            with (
                self.subTest(mutation=mutation[-160:]),
                self.assertRaisesRegex(ValueError, "complete product inventory"),
            ):
                _leverage_shares_html_to_universe(html, source)

    def test_leverage_shares_parser_tracks_function_parameter_global_flow(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        valid_html = _leverage_shares_products_page()
        mutations = (
            "<script>function f(w){w['products'+'Data']=[]} f(window);</script>",
            "<script>const f=w=>w['products'+'Data']=[];f(window);</script>",
            "<script>function f(x,w){w['products'+'Data']=[]} f(0,window);</script>",
            "<script>const f=(x,w)=>w['products'+'Data']=[];f(0,globalThis.window);</script>",
            "<script>function f(d){d(window,'productsData',{value:[]})} f(Object.defineProperty);</script>",
            "<script>const f=d=>d(window,'productsData');f(Reflect.deleteProperty);</script>",
            "<script>function f(o){o.defineProperty(window,'productsData',{value:[]})} f(Object);</script>",
            "<script>const f=r=>r.deleteProperty(window,'productsData');f(Reflect);</script>",
            "<script>function f(w){w['products'+'Data']=[]} f.call(null,window);</script>",
            "<script>const f=w=>w['products'+'Data']=[];f.apply(null,[window]);</script>",
            "<script>function f(w){w['products'+'Data']=[]} f['apply'](null,[window]);</script>",
            "<script>function f(w){w['products'+'Data']=[]} f.bind(null,window)();</script>",
            "<script>function f(w){w['products'+'Data']=[]} Reflect.apply(f,null,[window]);</script>",
            "<script>function f(w=window){w['products'+'Data']=[]} f();</script>",
            "<script>const f=(w=globalThis.window)=>w['products'+'Data']=[];f();</script>",
            "<script>function f(w){w=window;w['products'+'Data']=[]} f({});</script>",
            "<script>(function(w){w['products'+'Data']=[]})(window);</script>",
            "<script>(w=>w['products'+'Data']=[])(window);</script>",
            "<script>function f(...a){a[0]['products'+'Data']=[]} f(window);</script>",
            "<script>function f({w}){w['products'+'Data']=[]} f({w:window});</script>",
            "<script>const f=({w})=>w['products'+'Data']=[];f({w:window});</script>",
            "<script>const o={f(w){w['products'+'Data']=[]}};o.f(window);</script>",
            "<script>const f=function(w){w['products'+'Data']=[]};f(window);</script>",
            "<script>let f;f=w=>w['products'+'Data']=[];f(window);</script>",
            "<script>(function(c,l){c[l]=[]})(window,'productsData');</script>",
            "<script>function f(c,l){l='productsData';c[l]=[]} f(window,'clarity');</script>",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "complete product inventory"):
                _leverage_shares_html_to_universe(valid_html + mutation, source)

        harmless_local_shadows = (
            "<script>function f(window){window.productsData=[]} f({});</script>",
            "<script>function f(window){window['products'+'Data']=[]}</script>",
            "<script>const f=(window)=>window.productsData=[];f({});</script>",
            "<script>function f(w){w['products'+'Data']=[]} const window={};f(window);</script>",
            "<script>const f=w=>w['products'+'Data']=[];const window={};f(window);</script>",
        )
        for harmless_script in harmless_local_shadows:
            with self.subTest(script=harmless_script):
                out = _leverage_shares_html_to_universe(valid_html + harmless_script, source)
                self.assertEqual(len(out), 100)

    def test_script_ticker_name_parser_ignores_commented_objects(self) -> None:
        source = UniverseSource(
            "Example",
            "https://example.test",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        html = """
            <script>
                /* {ticker:'FAKE',name:'2x Long Fake Daily ETF'} */
                // {ticker:'ALSO',name:'2x Long Also Daily ETF'}
                {ticker:'REAL',name:'2x Long Real Daily ETF'}
            </script>
        """

        out = _js_ticker_name_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["REAL"])

        fabricated_sources = (
            "<script>({name:'2x Long QQQ Daily ETF', /* ticker:'FAKE' */ active:true})</script>",
            "<script>({ticker:'REAL', /* name:'2x Long Fake Daily ETF' */ active:true})</script>",
            "<script>({ticker:'REAL', // name:'2x Long Fake Daily ETF'\n active:true})</script>",
            r"<script>const re = /\{ticker:'FAKE',name:'2x Long QQQ Daily ETF'\}/;</script>",
            r"<script>function f(){ return /\{ticker:'FAKE',name:'2x Long QQQ Daily ETF'\}/; }</script>",
            r"<script>switch(x){ case /\{ticker:'FAKE',name:'2x Long QQQ Daily ETF'\}/: break; }</script>",
        )
        for fabricated_html in fabricated_sources:
            with self.subTest(fabricated_html=fabricated_html):
                if "ticker:'REAL'" in fabricated_html:
                    with self.assertRaisesRegex(ValueError, "ticker but omitted"):
                        _js_ticker_name_to_universe(fabricated_html, source)
                else:
                    fabricated = _js_ticker_name_to_universe(fabricated_html, source)
                    self.assertTrue(fabricated.empty)

    def test_script_ticker_name_parser_rejects_excessive_object_nesting(self) -> None:
        source = UniverseSource(
            "Example",
            "https://example.test",
            source_type="issuer_etf",
            parser="js_ticker_name",
        )
        html = "<script>" + "{" * 65 + "}" * 65 + "</script>"

        with self.assertRaisesRegex(ValueError, "safe parsing limit"):
            _js_ticker_name_to_universe(html, source)

    def test_leverage_shares_parser_ignores_commented_or_non_script_products_assignments(self) -> None:
        source = UniverseSource(
            "Leverage Shares",
            "https://leverageshares.com/us/all-etfs/",
            source_type="issuer_etf",
            parser="leverage_shares_html",
        )
        fake_assignment = "window.productsData = []"
        pages = (
            f"<script>/* {fake_assignment} */</script>",
            f"<script>// {fake_assignment}</script>",
            f"<pre>{fake_assignment}</pre>",
        )

        for html in pages:
            with self.subTest(html=html), self.assertRaisesRegex(ValueError, "complete product inventory"):
                _leverage_shares_html_to_universe(html, source)

    def test_graniteshares_parser_extracts_static_table_cells(self) -> None:
        source = UniverseSource(
            "GraniteShares",
            "https://example.test",
            source_type="issuer_etf",
            parser="graniteshares_html",
        )
        html = """
            <span class="etf-table-cell--ticker__symbol">AAPB</span>
            <span class="etf-table-cell--name-title Body1">
                GraniteShares 2x Long AAPL Daily ETF
            </span>
            <span class="etf-table-cell--ticker__symbol">PLAIN</span>
            <span class="etf-table-cell--name-title Body1">Plain Equity ETF</span>
        """

        out = _graniteshares_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["AAPB"])

    def test_rex_menu_parser_builds_generated_names(self) -> None:
        source = UniverseSource(
            "REX Shares",
            "https://example.test",
            source_type="issuer_etf",
            parser="rex_menu_html",
        )
        html = """
            <a href="https://www.rexshares.com/mstu/">MSTU | +2X Daily MSTR</a>
            <a href="https://www.rexshares.com/mstz/">MSTZ | -2X Daily MSTR</a>
        """

        out = _rex_menu_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["MSTU", "MSTZ"])
        self.assertIn("Inverse", out.loc[out["symbol"] == "MSTZ", "name"].item())

    def test_cboe_issuer_parser_builds_etf_rows(self) -> None:
        source = UniverseSource(
            "REX Shares",
            "https://www.cboe.com/example",
            source_type="issuer_etf",
            parser="cboe_issuer_html",
        )
        html = """
            <table>
                <tr><th>Product</th><th>Symbol</th><th>Product Type</th><th>List Date</th></tr>
                <tr><td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td><td>ETF</td><td>2024-01-01</td></tr>
                <tr><td>T-REX 2X Inverse MSTR Daily Target ETF</td><td>MSTZ</td><td>ETF</td><td>2024-01-01</td></tr>
                <tr><td>Example Note</td><td>NOTE</td><td>ETN</td><td>2024-01-01</td></tr>
            </table>
        """

        out = _cboe_issuer_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["MSTU", "MSTZ"])
        self.assertEqual(out["source"].tolist(), ["REX Shares Cboe listing"] * 2)

    def test_cboe_issuer_parser_ignores_unrelated_html_tables(self) -> None:
        source = UniverseSource(
            "REX Shares",
            "https://www.cboe.com/example",
            source_type="issuer_etf",
            parser="cboe_issuer_html",
        )
        html = """
            <table>
                <tr><th>Symbol</th><th>Last Price</th><th>Change</th></tr>
                <tr><td>VIX</td><td>18.2</td><td>-0.1</td></tr>
            </table>
            <table>
                <tr><th>Product</th><th>Symbol</th><th>Product Type</th></tr>
                <tr><td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td><td>ETF</td></tr>
            </table>
        """

        out = _cboe_issuer_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["MSTU"])

    def test_cboe_issuer_parser_rejects_ambiguous_or_incomplete_product_schema(self) -> None:
        source = UniverseSource(
            "REX Shares",
            "https://www.cboe.com/example",
            source_type="issuer_etf",
            parser="cboe_issuer_html",
        )
        malformed_tables = {
            "pandas-mangled duplicate symbol": (
                "<th>Product</th><th>Symbol</th><th>Symbol</th><th>Product Type</th>",
                "<td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td><td>MSTZ</td><td>ETF</td>",
                "ambiguous required columns",
            ),
            "semantic duplicate symbol": (
                "<th>Product</th><th>Symbol</th><th>Ticker</th><th>Product Type</th>",
                "<td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td><td>MSTZ</td><td>ETF</td>",
                "ambiguous required columns",
            ),
            "missing product type": (
                "<th>Product</th><th>Symbol</th>",
                "<td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td>",
                "omitted its required product type column",
            ),
        }

        for label, (header, row, expected_error) in malformed_tables.items():
            html = f"<table><tr>{header}</tr><tr>{row}</tr></table>"
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, expected_error),
            ):
                _cboe_issuer_html_to_universe(html, source)

    def test_cboe_issuer_parser_requires_recognized_type_on_every_product_row(self) -> None:
        source = UniverseSource(
            "REX Shares",
            "https://www.cboe.com/example",
            source_type="issuer_etf",
            parser="cboe_issuer_html",
        )
        for product_type in ("", "WARRANT"):
            html = f"""
                <table>
                    <tr><th>Product</th><th>Symbol</th><th>Product Type</th></tr>
                    <tr><td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td><td>ETF</td></tr>
                    <tr><td>T-REX 2X Long TSLA Daily Target ETF</td><td>TSLL</td><td>{product_type}</td></tr>
                </table>
            """
            with (
                self.subTest(product_type=product_type),
                self.assertRaisesRegex(ValueError, "unrecognized product type"),
            ):
                _cboe_issuer_html_to_universe(html, source)

    def test_cboe_issuer_parser_rejects_malformed_selected_etf_rows(self) -> None:
        source = UniverseSource(
            "REX Shares",
            "https://www.cboe.com/example",
            source_type="issuer_etf",
            parser="cboe_issuer_html",
        )
        malformed_rows = {
            "invalid ticker": (
                "T-REX 2X Long MSTR Daily Target ETF",
                "MSTU..",
            ),
            "missing name": ("", "MSTU"),
        }

        for label, (name, symbol) in malformed_rows.items():
            html = f"""
                <table>
                    <tr><th>Product</th><th>Symbol</th><th>Product Type</th></tr>
                    <tr><td>{name}</td><td>{symbol}</td><td>ETF</td></tr>
                </table>
            """
            with self.subTest(label=label), self.assertRaises(ValueError):
                _cboe_issuer_html_to_universe(html, source)

    def test_cboe_issuer_parser_rejects_conflicting_duplicate_ticker_records(self) -> None:
        source = UniverseSource(
            "REX Shares",
            "https://www.cboe.com/example",
            source_type="issuer_etf",
            parser="cboe_issuer_html",
        )
        conflicting_html = """
            <table>
                <tr><th>Product</th><th>Symbol</th><th>Product Type</th></tr>
                <tr><td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td><td>ETF</td></tr>
                <tr><td>T-REX 2X Inverse MSTR Daily Target ETF</td><td>MSTU</td><td>ETF</td></tr>
            </table>
        """

        with self.assertRaisesRegex(ValueError, "conflicting records for normalized ticker MSTU"):
            _cboe_issuer_html_to_universe(conflicting_html, source)

        identical_html = """
            <table>
                <tr><th>Product</th><th>Symbol</th><th>Product Type</th></tr>
                <tr><td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td><td>ETF</td></tr>
                <tr><td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td><td>ETF</td></tr>
            </table>
        """
        out = _cboe_issuer_html_to_universe(identical_html, source)

        self.assertEqual(out["symbol"].tolist(), ["MSTU"])

    def test_volatilityshares_parser_extracts_list_items(self) -> None:
        source = UniverseSource(
            "Volatility Shares",
            "https://example.test",
            source_type="issuer_etf",
            parser="volatilityshares_html",
        )
        html = """
            <li><a href="/bitx"><h4>BITX</h4><p>2x Bitcoin ETF</p></a></li>
            <li><a href="/plain"><h4>PLAIN</h4><p>Plain Bitcoin ETF</p></a></li>
        """

        out = _volatilityshares_html_to_universe(html, source)

        self.assertEqual(out["symbol"].tolist(), ["BITX"])

    def test_issuer_table_parser_rejects_entire_candidate_with_a_malformed_row(self) -> None:
        malformed_rows = [
            {"Ticker": float("nan"), "Fund Name": "Missing Symbol 2X Long ETF"},
            {"Ticker": 123.0, "Fund Name": "Numeric Symbol 2X Long ETF"},
            {"Ticker": " BAD ", "Fund Name": "Whitespace Symbol 2X Long ETF"},
            {"Ticker": "BAD SYMBOL", "Fund Name": "Invalid Symbol 2X Long ETF"},
            {"Ticker": "BAD", "Fund Name": ["Container Name 2X Long ETF"]},
        ]

        for malformed_row in malformed_rows:
            table = pd.DataFrame(
                [
                    {"Ticker": "GOOD", "Fund Name": "Example 2X Long GOOD Daily ETF"},
                    malformed_row,
                ]
            )
            with self.subTest(malformed_row=malformed_row), self.assertRaises(ValueError):
                _issuer_table_to_universe(table, "Example")

    def test_issuer_table_parser_rejects_control_characters_in_fund_names(self) -> None:
        for control in ("\x00", "\x1b[2J"):
            table = pd.DataFrame(
                [
                    {
                        "Ticker": "CTRL",
                        "Fund Name": f"Example 2X Long QQQ{control} Daily ETF",
                    }
                ]
            )

            with self.subTest(control=repr(control)), self.assertRaisesRegex(ValueError, "control characters"):
                _issuer_table_to_universe(table, "Example")

    def test_issuer_table_parser_requires_unambiguous_semantic_columns(self) -> None:
        ambiguous_tables = {
            "symbol aliases": pd.DataFrame(
                [{"Ticker": "WRONG", "Symbol": "RIGHT", "Fund Name": "Example 2X Long QQQ Daily ETF"}]
            ),
            "name aliases": pd.DataFrame(
                [
                    {
                        "Ticker": "AAA",
                        "Fund Name": "Example 2X Long QQQ Daily ETF",
                        "Product Name": "Forged 2X Long SPY Daily ETF",
                    }
                ]
            ),
            "pandas-mangled duplicate": pd.DataFrame(
                [["WRONG", "RIGHT", "Example 2X Long QQQ Daily ETF"]],
                columns=["Ticker", "Ticker.1", "Fund Name"],
            ),
        }

        for label, table in ambiguous_tables.items():
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, "ambiguous semantic symbol/name columns"),
            ):
                _issuer_table_to_universe(table, "Example")

    def test_issuer_table_parser_ignores_auxiliary_name_columns(self) -> None:
        table = pd.DataFrame(
            [
                {
                    "Ticker": "AAA",
                    "Fund Name": "Example 2X Long QQQ Daily ETF",
                    "Index Name": "Nasdaq-100 Index",
                    "Issuer Name": "Example Funds",
                    "Underlying Name": "Invesco QQQ Trust",
                }
            ]
        )

        out = _issuer_table_to_universe(table, "Example")

        self.assertEqual(
            out[["symbol", "name"]].to_dict("records"),
            [{"symbol": "AAA", "name": "Example 2X Long QQQ Daily ETF"}],
        )

    def test_audit_report_includes_leveraged_candidates_missing_from_merged_universe(self) -> None:
        audit_rows = pd.DataFrame(
            [
                {
                    "symbol": "MISSING",
                    "name": "Example 2X Long MISSING Daily ETF",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                    "audit_source_type": "third_party_audit",
                    "source_url": "https://example.test",
                    "is_leveraged_candidate": True,
                    "is_long_leveraged_candidate": True,
                    "leverage": 2.0,
                    "direction": "long",
                },
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                    "audit_source_type": "third_party_audit",
                    "source_url": "https://example.test",
                    "is_leveraged_candidate": True,
                    "is_long_leveraged_candidate": True,
                    "leverage": 3.0,
                    "direction": "long",
                },
                {
                    "symbol": "SHORTY",
                    "name": "Example 2X Short Missing ETF",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                    "audit_source_type": "third_party_audit",
                    "source_url": "https://example.test",
                    "is_leveraged_candidate": True,
                    "is_long_leveraged_candidate": False,
                    "leverage": 2.0,
                    "direction": "inverse",
                },
                {
                    "symbol": "PLAIN",
                    "name": "Plain Equity ETF",
                    "fund_type": "ETF (Audit)",
                    "source": "Audit source",
                    "audit_source_type": "third_party_audit",
                    "source_url": "https://example.test",
                    "is_leveraged_candidate": False,
                    "is_long_leveraged_candidate": False,
                    "leverage": None,
                    "direction": None,
                },
            ]
        )
        merged = pd.DataFrame([{"symbol": "TQQQ"}])
        workflow = pd.DataFrame([{"symbol": "TQQQ"}])

        report = build_universe_audit_report(audit_rows, merged, workflow)

        self.assertEqual(report["symbol"].tolist(), ["MISSING", "SHORTY"])
        short_row = report.loc[report["symbol"].eq("SHORTY")].iloc[0]
        self.assertIn("inverse leveraged-looking", short_row["audit_reason"])

    def test_inverse_self_rsi_fallback_is_excluded_for_review(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "MYST", "name": "Example 2X Inverse Mystery Index ETF", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ", "MYST"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            groups = determine_workflow_asset_groups(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertTrue(groups["short"].empty)
        review = groups["long"].attrs["rsi_mapping_review"]
        self.assertEqual([row["symbol"] for row in review], ["MYST"])
        self.assertEqual(review[0]["confidence"], "needs_review")
        self.assertIn("requires an underlying RSI proxy", review[0]["mapping_reason"])

    @patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 1)
    @patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 1)
    @patch("leveraged_trader.universe.requests.get")
    def test_load_current_etf_universe_filters_exclusions_and_normalizes_brkb(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.text = """
            <table>
                <tr><th>Symbol</th><th>Fund Name</th><th>Fund Type</th></tr>
                <tr><td>NASDAQ</td><td>Bad Nasdaq Row</td><td>ETF</td></tr>
                <tr><td>BGGG</td><td>Baillie Gifford Long Term Global Growth ETF</td><td>ETF</td></tr>
                <tr><td>BRKB</td><td>Berkshire Test Row</td><td>ETF</td></tr>
                <tr><td>TQQQ</td><td>ProShares UltraPro QQQ</td><td>ETF</td></tr>
            </table>
        """
        mock_get.return_value = response

        with patch.dict("os.environ", {"SEC_USER_AGENT": "leveraged-trader test operator@example.test"}):
            universe = load_current_etf_universe()

        self.assertNotIn("NASDAQ", universe["symbol"].tolist())
        self.assertNotIn("BGGG", universe["symbol"].tolist())
        self.assertIn("BRK-B", universe["symbol"].tolist())
        self.assertIn("TQQQ", universe["symbol"].tolist())
        self.assertEqual(
            mock_get.call_args.kwargs["headers"]["User-Agent"],
            DEFAULT_USER_AGENT,
        )
        self.assertEqual(universe.attrs["workflow_source_status"][0]["status"], "loaded")

    def test_load_current_etf_universe_rejects_header_only_table(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        header_only = pd.DataFrame(columns=["Symbol", "Fund Name", "Fund Type"])

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[header_only]),
            self.assertRaisesRegex(RuntimeError, "no usable ETF rows"),
        ):
            load_current_etf_universe()

    def test_load_current_etf_universe_rejects_implausibly_small_raw_snapshot(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        definitions = pd.DataFrame(
            [
                {"Symbol": "TQQQ", "Fund Name": "ProShares UltraPro QQQ", "Fund Type": "ETF"},
                {"Symbol": "SQQQ", "Fund Name": "ProShares UltraPro Short QQQ", "Fund Type": "ETF"},
            ]
        )

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[definitions]),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 3),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 1),
            self.assertRaisesRegex(RuntimeError, "implausibly small.*2 raw rows.*at least 3"),
        ):
            load_current_etf_universe()

    def test_load_current_etf_universe_rejects_implausibly_small_usable_snapshot(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        definitions = pd.DataFrame(
            [
                {"Symbol": "TQQQ", "Fund Name": "ProShares UltraPro QQQ", "Fund Type": "ETF"},
                {"Symbol": "SQQQ", "Fund Name": "ProShares UltraPro Short QQQ", "Fund Type": "ETF"},
                {"Symbol": "NOTE", "Fund Name": "Example Note", "Fund Type": "ETN"},
            ]
        )

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[definitions]),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 3),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 3),
            self.assertRaisesRegex(RuntimeError, "few usable ETF symbols.*found 2.*at least 3"),
        ):
            load_current_etf_universe()

    def test_load_current_etf_universe_accepts_complete_inventory_after_liquidity_row_consolidation(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        definitions = pd.DataFrame(
            [
                {
                    "Symbol": f"P{row:04d}",
                    "Fund Name": f"Product {row} ETF",
                    "Fund Type": "ETF",
                    "DLP Firm": "One consolidated liquidity-provider assignment",
                }
                for row in range(1_200)
            ]
        )

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[definitions]),
        ):
            universe = load_current_etf_universe()

        self.assertEqual(len(universe), 1_200)
        status = universe.attrs["workflow_source_status"][0]
        self.assertEqual((status["parsed_row_count"], status["row_count"]), (1_200, 1_200))

    def test_load_current_etf_universe_rejects_partial_content_with_complete_inventory_body(self) -> None:
        definitions = pd.DataFrame(
            [
                {
                    "Symbol": f"P{row:04d}",
                    "Fund Name": f"Product {row} ETF",
                    "Fund Type": "ETF",
                }
                for row in range(1_200)
            ]
        )
        body = definitions.to_html(index=False).encode()
        response = _streaming_response(
            body,
            status_code=206,
            headers={"Content-Range": f"bytes 0-{len(body) - 1}/{len(body) + 1}"},
        )

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            self.assertRaisesRegex(requests.exceptions.HTTPError, "unexpected HTTP status 206"),
        ):
            load_current_etf_universe()

        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    def test_load_current_etf_universe_rejects_content_range_on_exact_200(self) -> None:
        definitions = pd.DataFrame(
            [
                {
                    "Symbol": f"P{row:04d}",
                    "Fund Name": f"Product {row} ETF",
                    "Fund Type": "ETF",
                }
                for row in range(1_200)
            ]
        )
        body = definitions.to_html(index=False).encode()
        response = _streaming_response(
            body,
            status_code=200,
            headers={"cOnTeNt-RaNgE": f"bytes 0-{len(body) - 1}/{len(body) + 1}"},
        )

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            self.assertRaisesRegex(requests.exceptions.InvalidHeader, "unsolicited Content-Range"),
        ):
            load_current_etf_universe()

        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    def test_load_current_etf_universe_rejects_rows_with_missing_names(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        missing_names = pd.DataFrame(
            [
                {"Symbol": "AAA", "Fund Name": pd.NA, "Fund Type": "ETF"},
                {"Symbol": "BBB", "Fund Name": None, "Fund Type": "ETF"},
            ]
        )

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[missing_names]),
            self.assertRaisesRegex(RuntimeError, "fund name"),
        ):
            load_current_etf_universe()

    def test_load_current_etf_universe_rejects_normalization_collisions(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        definitions = pd.DataFrame(
            [
                {"Symbol": "BRKB", "Fund Name": "Berkshire Benchmark ETF", "Fund Type": "ETF"},
                {"Symbol": "BRK.B", "Fund Name": "Daily QQQ 2X Leveraged ETF", "Fund Type": "ETF"},
            ]
        )

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[definitions]),
            self.assertRaisesRegex(RuntimeError, "normalization collision.*BRK-B"),
        ):
            load_current_etf_universe()

    def test_load_current_etf_universe_rejects_duplicate_conflict_before_eligibility_filter(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        conflict_variants = [
            (
                {"Symbol": "DUP", "Fund Name": "Plain operating company", "Fund Type": "Equity"},
                "conflicting metadata.*DUP",
            ),
            (
                {"Symbol": "DUP", "Fund Name": pd.NA, "Fund Type": "ETF"},
                "fund name",
            ),
        ]

        for conflicting_row, expected_error in conflict_variants:
            with (
                self.subTest(conflicting_row=conflicting_row),
                patch("leveraged_trader.universe.requests.get", return_value=response),
                patch(
                    "leveraged_trader.universe._read_html_tables",
                    return_value=[
                        pd.DataFrame(
                            [
                                {"Symbol": "DUP", "Fund Name": "Daily QQQ 2X ETF", "Fund Type": "ETF"},
                                conflicting_row,
                            ]
                        )
                    ],
                ),
                self.assertRaisesRegex(RuntimeError, expected_error),
            ):
                load_current_etf_universe()

    def test_load_current_etf_universe_rejects_equivalent_normalized_duplicate_symbols(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        definitions = pd.DataFrame(
            [
                {"Symbol": "BRKB", "Fund Name": "Berkshire Benchmark ETF", "Fund Type": "ETF"},
                {"Symbol": "BRK.B", "Fund Name": " Berkshire  Benchmark ETF ", "Fund Type": "ETF"},
            ]
        )

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[definitions]),
            self.assertRaisesRegex(RuntimeError, "normalization collision.*BRK-B"),
        ):
            load_current_etf_universe()

    def test_load_current_etf_universe_validates_every_raw_row_before_filtering(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        malformed_rows = {
            "symbol": {
                "Symbol": "TQQQ..",
                "Fund Name": "Operating company",
                "Fund Type": "Equity",
            },
            "name": {
                "Symbol": "PLAIN",
                "Fund Name": None,
                "Fund Type": "Equity",
            },
            "fund type": {
                "Symbol": "PLAIN",
                "Fund Name": "Operating company",
                "Fund Type": ["Equity"],
            },
        }
        valid_row = {
            "Symbol": "TQQQ",
            "Fund Name": "ProShares UltraPro QQQ",
            "Fund Type": "ETF",
        }

        for field, malformed_row in malformed_rows.items():
            with (
                self.subTest(field=field),
                patch("leveraged_trader.universe.requests.get", return_value=response),
                patch(
                    "leveraged_trader.universe._read_html_tables",
                    return_value=[pd.DataFrame([valid_row, malformed_row])],
                ),
                self.assertRaisesRegex(RuntimeError, field),
            ):
                load_current_etf_universe()

    def test_load_current_etf_universe_uses_explicit_name_column_and_schema_table(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        larger_unrelated_table = pd.DataFrame(
            {
                "Instrument Code": [f"ROW{index}" for index in range(20)],
                "Currency": ["USD"] * 20,
                "Category": ["reference"] * 20,
            }
        )
        definitions = pd.DataFrame(
            [
                {
                    "Symbol": "TQQQ",
                    "Currency": "USD",
                    "Security Name": "ProShares UltraPro QQQ",
                    "Fund Type": "ETF",
                }
            ]
        )

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch(
                "leveraged_trader.universe._read_html_tables",
                return_value=[larger_unrelated_table, definitions],
            ),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 1),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 1),
        ):
            universe = load_current_etf_universe()

        self.assertEqual(
            universe[["symbol", "name"]].to_dict("records"), [{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ"}]
        )

    def test_primary_nasdaq_parser_rejects_partial_schema_before_selecting_another_table(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        incomplete_primary = pd.DataFrame([{"Symbol": "TQQQ", "Fund Name": "ProShares UltraPro QQQ"}])
        complete_decoy = pd.DataFrame([{"Ticker": "FAKE", "Security Name": "Decoy ETF", "Type": "ETF"}])

        with (
            patch("leveraged_trader.universe._get_universe_response", return_value=response),
            patch(
                "leveraged_trader.universe._read_html_tables",
                return_value=[incomplete_primary, complete_decoy],
            ),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 1),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 1),
            self.assertRaisesRegex(RuntimeError, "incomplete explicit symbol.*omitted required fund type"),
        ):
            load_current_etf_universe()

    def test_primary_nasdaq_parser_ignores_name_and_type_page_furniture(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        page_furniture = pd.DataFrame([{"Security Name": "Navigation link", "Type": "menu item"}])
        definitions = pd.DataFrame([{"Symbol": "TQQQ", "Fund Name": "ProShares UltraPro QQQ", "Fund Type": "ETF"}])

        with (
            patch("leveraged_trader.universe._get_universe_response", return_value=response),
            patch(
                "leveraged_trader.universe._read_html_tables",
                return_value=[page_furniture, definitions],
            ),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 1),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 1),
        ):
            universe = load_current_etf_universe()

        self.assertEqual(universe["symbol"].tolist(), ["TQQQ"])

    def test_primary_nasdaq_parser_rejects_fund_name_and_type_table_without_symbol(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        incomplete_inventory = pd.DataFrame(
            {
                "Fund Name": [f"Example Fund {index}" for index in range(150)],
                "Fund Type": ["ETF"] * 150,
            }
        )
        complete_decoy = pd.DataFrame([{"Symbol": "FAKE", "Security Name": "Decoy ETF", "Type": "ETF"}])

        with (
            patch("leveraged_trader.universe._get_universe_response", return_value=response),
            patch(
                "leveraged_trader.universe._read_html_tables",
                return_value=[incomplete_inventory, complete_decoy],
            ),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 1),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 1),
            self.assertRaisesRegex(RuntimeError, "incomplete explicit symbol.*omitted required symbol"),
        ):
            load_current_etf_universe()

    def test_primary_nasdaq_parser_rejects_inventory_sized_generic_name_type_table(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        incomplete_inventory = pd.DataFrame(
            {
                "Security Name": [f"Example Security {index}" for index in range(100)],
                "Type": ["ETF"] * 100,
            }
        )
        complete_decoy = pd.DataFrame([{"Symbol": "FAKE", "Security Name": "Decoy ETF", "Type": "ETF"}])

        with (
            patch("leveraged_trader.universe._get_universe_response", return_value=response),
            patch(
                "leveraged_trader.universe._read_html_tables",
                return_value=[incomplete_inventory, complete_decoy],
            ),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 1),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 1),
            self.assertRaisesRegex(RuntimeError, "incomplete explicit symbol.*omitted required symbol"),
        ):
            load_current_etf_universe()

    def test_primary_nasdaq_parser_ignores_one_role_inventory_sized_table(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        page_furniture = pd.DataFrame({"Fund Name": [f"Reference label {index}" for index in range(150)]})
        definitions = pd.DataFrame([{"Symbol": "TQQQ", "Fund Name": "ProShares UltraPro QQQ", "Fund Type": "ETF"}])

        with (
            patch("leveraged_trader.universe._get_universe_response", return_value=response),
            patch(
                "leveraged_trader.universe._read_html_tables",
                return_value=[page_furniture, definitions],
            ),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 1),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 1),
        ):
            universe = load_current_etf_universe()

        self.assertEqual(universe["symbol"].tolist(), ["TQQQ"])

    def test_load_current_etf_universe_accepts_current_nasdaq_fund_header(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        definitions = pd.DataFrame([{"Symbol": "TQQQ", "Fund": "ProShares UltraPro QQQ", "Type": "ETF"}])

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[definitions]),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_RAW_ROWS", 1),
            patch("leveraged_trader.universe._NASDAQ_ETF_MINIMUM_USABLE_ROWS", 1),
        ):
            universe = load_current_etf_universe()

        self.assertEqual(
            universe[["symbol", "name", "fund_type"]].to_dict("records"),
            [{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}],
        )

    def test_primary_nasdaq_parser_rejects_pandas_mangled_duplicate_semantic_headers(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        definitions = pd.DataFrame(
            [["TQQQ", "SQQQ", "ProShares UltraPro QQQ", "ETF"]],
            columns=["Symbol", "Symbol.1", "Fund Name", "Fund Type"],
        )

        with (
            patch("leveraged_trader.universe._get_universe_response", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[definitions]),
            self.assertRaisesRegex(RuntimeError, "ambiguous semantic columns.*Symbol.1"),
        ):
            load_current_etf_universe()

    def test_primary_nasdaq_mangled_identity_header_marks_source_degraded(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        definitions = pd.DataFrame(
            [["TQQQ", "SQQQ", "ProShares UltraPro QQQ", "ETF"]],
            columns=["Symbol", "Symbol.1", "Fund Name", "Fund Type"],
        )
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF (Issuer)",
                    "source": "Issuer table",
                }
            ]
        )
        issuer_rows.attrs["workflow_source_status"] = []
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe._get_universe_response", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[definitions]),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            groups = determine_workflow_asset_groups(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertTrue(groups["long"].attrs["universe_degraded"])
        saved_status = next(
            call.args[0] for call in mock_save_table.call_args_list if call.args[2] == "universe_workflow_source_status"
        )
        primary = saved_status.loc[saved_status["source"].eq("Nasdaq ETF definitions")].iloc[0]
        self.assertEqual(primary["status"], "parse_error")
        self.assertIn("ambiguous semantic columns", primary["error"])

    def test_load_current_etf_universe_rejects_ambiguous_schema_tables(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        first = pd.DataFrame([{"Symbol": "AAA", "Fund Name": "First ETF", "Fund Type": "ETF"}])
        second = pd.DataFrame([{"Ticker": "BBB", "Security Name": "Second ETF", "Type": "ETF"}])

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch("leveraged_trader.universe._read_html_tables", return_value=[first, second]),
            self.assertRaisesRegex(RuntimeError, "multiple tables.*ambiguous"),
        ):
            load_current_etf_universe()

    def test_load_current_etf_universe_rejects_positional_name_guess(self) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        definitions_without_name = pd.DataFrame([{"Symbol": "TQQQ", "Currency": "USD", "Fund Type": "ETF"}])

        with (
            patch("leveraged_trader.universe.requests.get", return_value=response),
            patch(
                "leveraged_trader.universe._read_html_tables",
                return_value=[definitions_without_name],
            ),
            self.assertRaisesRegex(RuntimeError, "explicit symbol.*fund/security name"),
        ):
            load_current_etf_universe()

    def test_primary_nasdaq_failure_is_persisted_and_marks_universe_degraded(self) -> None:
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF (Issuer)",
                    "source": "Issuer table",
                }
            ]
        )
        issuer_rows.attrs["workflow_source_status"] = []
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", side_effect=RuntimeError("header only")),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            groups = determine_workflow_asset_groups(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertTrue(groups["long"].attrs["universe_degraded"])
        failures = groups["long"].attrs["workflow_source_failures"]
        self.assertEqual(failures[0]["source"], "Nasdaq ETF definitions")
        saved_status = next(
            call.args[0] for call in mock_save_table.call_args_list if call.args[2] == "universe_workflow_source_status"
        )
        primary = saved_status.loc[saved_status["source"].eq("Nasdaq ETF definitions")].iloc[0]
        self.assertEqual(primary["status"], "parse_error")
        self.assertIn("header only", primary["error"])

    def test_strict_source_mode_rejects_primary_nasdaq_failure_after_persisting_health(self) -> None:
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF (Issuer)",
                    "source": "Issuer table",
                }
            ]
        )
        issuer_rows.attrs["workflow_source_status"] = []
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", side_effect=RuntimeError("offline")),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "Nasdaq ETF definitions"),
        ):
            determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        self.assertIn(
            "universe_workflow_source_status",
            [call.args[2] for call in mock_save_table.call_args_list],
        )

    def test_strict_source_mode_accepts_live_shaped_equivalent_nonleveraged_and_elol_rows(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {
                    "symbol": "AHD",
                    "name": "GraniteShares Autocallable HOOD ETF",
                    "fund_type": "ETF (Single Stock)",
                },
                {
                    "symbol": "ELOL",
                    "name": "Leverage Shares 100% TSLA AND 100% SPCX Daily ETF",
                    "fund_type": "ETF",
                },
            ]
        )
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "AHD",
                    "name": "GraniteShares Autocallable HOOD ETF",
                    "fund_type": "ETF (GraniteShares)",
                    "source": "GraniteShares issuer table",
                },
                {
                    "symbol": "ELOL",
                    "name": "100% TSLA + 100% SPCX Daily ETF",
                    "fund_type": "ETF (Leverage Shares)",
                    "source": "Leverage Shares complete product inventory",
                },
            ]
        )
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"AHD", "ELOL", "HOOD", "QQQ", "SPCX", "TQQQ", "TSLA"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            groups = determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        self.assertEqual(groups["long"]["symbol"].tolist(), ["ELOL", "TQQQ"])
        self.assertEqual(groups["long"].attrs["workflow_source_failures"], [])

    @patch("builtins.print")
    @patch(
        "leveraged_trader.universe.load_active_listed_symbols",
        return_value={"TQQQ", "BRKU", "BRK-B", "QQQ", "EXTRA", "BDCX"},
    )
    @patch("leveraged_trader.universe.load_audit_universe_sources")
    @patch("leveraged_trader.universe.load_etn_universe")
    @patch("leveraged_trader.universe.load_issuer_etf_universe")
    @patch("leveraged_trader.universe.save_table_to_sqlite")
    @patch("leveraged_trader.universe.load_current_etf_universe")
    def test_determine_workflow_assets_returns_universe_metadata_without_printing(
        self,
        mock_load_universe: Mock,
        mock_save_table: Mock,
        mock_load_issuer_universe: Mock,
        mock_load_etn_universe: Mock,
        mock_load_audit_sources: Mock,
        _mock_load_active_symbols: Mock,
        mock_print: Mock,
    ) -> None:
        mock_load_universe.return_value = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "BRKU",
                    "name": "2X Long BRKB Daily ETF",
                    "fund_type": "ETF (Single Stock)",
                },
                {
                    "symbol": "BGGG",
                    "name": "Baillie Gifford Long Term Global Growth ETF",
                    "fund_type": "ETF",
                },
            ]
        )
        mock_load_issuer_universe.return_value = pd.DataFrame(
            [
                {
                    "symbol": "EXTRA",
                    "name": "Example 2X Leveraged Broad Market ETF",
                    "fund_type": "ETF (Example Issuer)",
                    "source": "Example issuer table",
                },
            ]
        )
        mock_load_etn_universe.return_value = pd.DataFrame(
            [
                {
                    "symbol": "BDCX",
                    "name": "ETRACS Quarterly Pay 1.5x Leveraged MarketVector BDC Liquid Index ETN",
                    "fund_type": "ETN (UBS ETRACS)",
                    "source": "UBS ETRACS ETN issuer table",
                },
            ]
        )
        mock_load_audit_sources.return_value = (
            pd.DataFrame(
                [
                    {
                        "symbol": "MISSING",
                        "name": "Example 2X Long MISSING Daily ETF",
                        "fund_type": "ETF (Audit)",
                        "source": "Audit source",
                        "audit_source_type": "third_party_audit",
                        "source_url": "https://example.test",
                        "is_leveraged_candidate": True,
                        "is_long_leveraged_candidate": True,
                        "leverage": 2.0,
                        "direction": "long",
                    },
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "source": "Audit source",
                        "source_type": "third_party_audit",
                        "url": "https://example.test",
                        "parser": "html",
                        "enabled": True,
                        "status": "loaded",
                        "row_count": 1,
                        "error": "",
                        "notes": "",
                    }
                ]
            ),
        )

        workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(
            workflow_assets.attrs["universe_title"],
            "Executable Long Leveraged ETFs/ETNs From Merged Universe",
        )
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Current ETFs in Nasdaq table"], 3)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Current issuer-discovered leveraged ETFs found"], 1)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Current issuer-discovered leveraged ETNs found"], 1)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Inactive primary Nasdaq ETFs skipped"], 1)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Merged current ETFs/ETNs"], 4)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Current long leveraged ETFs/ETNs found"], 4)
        inactive_rows = next(
            call.args[0]
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_inactive_discovered_products"
        )
        self.assertEqual(inactive_rows["symbol"].tolist(), ["BGGG"])
        self.assertEqual(inactive_rows["inactive_source"].tolist(), ["primary_nasdaq"])
        self.assertEqual(
            inactive_rows["inactive_reason"].tolist(),
            ["not present in active Nasdaq symbol files"],
        )
        self.assertEqual(
            workflow_assets.attrs["universe_counts"]["Audit product-name rows parsed"],
            1,
        )
        self.assertEqual(
            workflow_assets.attrs["universe_counts"]["Audit inventory-only rows parsed"],
            0,
        )
        self.assertEqual(
            workflow_assets.attrs["universe_counts"]["Audit leveraged candidates missing from merged universe"],
            1,
        )
        self.assertEqual(workflow_assets.attrs["universe_db_path"], "state.sqlite")
        self.assertEqual(workflow_assets["symbol"].tolist(), ["BDCX", "BRKU", "EXTRA", "TQQQ"])
        self.assertEqual(workflow_assets["rsi_symbol"].tolist(), ["BIZD", "BRK-B", "EXTRA", "QQQ"])
        self.assertIn("confidence", workflow_assets.columns)
        saved_table_names = [call.args[2] for call in mock_save_table.call_args_list]
        self.assertEqual(
            saved_table_names,
            [
                "nasdaq_etf_universe",
                "universe_inactive_discovered_products",
                "universe_active_listing_source_status",
                "universe_workflow_source_status",
                "universe_audit_rows",
                "universe_audit_missing_candidates",
                "universe_audit_source_status",
                "universe_rsi_mapping_review",
            ],
        )
        mock_print.assert_not_called()

    def test_unresolved_single_stock_mapping_is_saved_for_review(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "FOOU",
                    "name": "T-REX 2X Long ExampleCorp Daily Target ETF",
                    "fund_type": "ETF (Single Stock)",
                },
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"FOOU", "TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["TQQQ"])
        self.assertEqual(workflow_assets["rsi_symbol"].tolist(), ["QQQ"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings needing review"], 1)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings excluded pending review"], 1)
        self.assertEqual(len(workflow_assets.attrs["rsi_mapping_review"]), 1)

        review_call = next(
            call for call in mock_save_table.call_args_list if call.args[2] == "universe_rsi_mapping_review"
        )
        review_df = review_call.args[0]
        self.assertEqual(review_df["symbol"].tolist(), ["FOOU"])
        self.assertEqual(review_df["confidence"].tolist(), ["needs_review"])

    def test_unresolved_short_single_stock_mapping_is_saved_for_review(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "SQQQ",
                    "name": "ProShares UltraPro Short QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "FOOS",
                    "name": "2X Short ExampleCorp Daily ETF",
                    "fund_type": "ETF (Example Issuer)",
                },
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"FOOS", "QQQ", "SQQQ", "TQQQ"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_asset_groups = determine_workflow_asset_groups(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_asset_groups["long"]["symbol"].tolist(), ["TQQQ"])
        self.assertEqual(workflow_asset_groups["short"]["symbol"].tolist(), ["SQQQ"])
        self.assertEqual(workflow_asset_groups["short"]["rsi_symbol"].tolist(), ["QQQ"])
        self.assertEqual(
            workflow_asset_groups["short"].attrs["universe_counts"]["RSI mappings needing review"],
            1,
        )
        self.assertEqual(
            workflow_asset_groups["short"].attrs["universe_counts"]["RSI mappings excluded pending review"],
            1,
        )

        review_call = next(
            call for call in mock_save_table.call_args_list if call.args[2] == "universe_rsi_mapping_review"
        )
        review_df = review_call.args[0]
        self.assertEqual(review_df["workflow"].tolist(), ["Short"])
        self.assertEqual(review_df["symbol"].tolist(), ["FOOS"])
        self.assertEqual(review_df["confidence"].tolist(), ["needs_review"])
        self.assertEqual(review_df["mapping_source"].tolist(), ["unresolved_single_stock"])

    def test_explicit_self_fallback_basket_mapping_remains_executable(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "fund_type": "ETF",
                },
                {
                    "symbol": "BEGS",
                    "name": "Rareview 2X Bull Cryptocurrency & Precious Metals ETF",
                    "fund_type": "ETF",
                },
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"BEGS", "TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["BEGS", "TQQQ"])
        self.assertEqual(workflow_assets["rsi_symbol"].tolist(), ["BEGS", "QQQ"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings needing review"], 0)
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings excluded pending review"], 0)
        self.assertEqual(workflow_assets.attrs["rsi_mapping_review"], [])

        begs = workflow_assets.loc[workflow_assets["symbol"] == "BEGS"].iloc[0]
        self.assertEqual(begs["confidence"], "fallback_to_self")
        self.assertEqual(begs["mapping_source"], "self_fallback_override")

        review_call = next(
            call for call in mock_save_table.call_args_list if call.args[2] == "universe_rsi_mapping_review"
        )
        review_df = review_call.args[0]
        self.assertTrue(review_df.empty)

    def test_spacex_underlying_maps_to_spcx_even_when_active_listing_is_partial(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {
                    "symbol": "SPAL",
                    "name": "GraniteShares 2x Long SpaceX Daily ETF",
                    "fund_type": "ETF (GraniteShares)",
                },
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []
        partial_listing = ActiveListedSymbols(
            {"TQQQ", "SPAL"},
            [
                {"source": "nasdaq_listed", "status": "loaded"},
                {"source": "other_listed", "status": "error"},
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=partial_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["SPAL", "TQQQ"])
        self.assertEqual(workflow_assets["rsi_symbol"].tolist(), ["SPCX", "QQQ"])
        self.assertNotIn("SPACEX", workflow_assets["rsi_symbol"].tolist())
        self.assertEqual(workflow_assets.attrs["universe_counts"]["RSI mappings needing review"], 0)
        self.assertEqual(workflow_assets.attrs["rsi_mapping_review"], [])

        nasdaq_call = next(call for call in mock_save_table.call_args_list if call.args[2] == "nasdaq_etf_universe")
        nasdaq_universe = nasdaq_call.args[0]
        self.assertNotIn("SPACEX", nasdaq_universe["rsi_symbol"].tolist())
        self.assertEqual(
            nasdaq_universe.loc[nasdaq_universe["symbol"] == "SPAL", "rsi_symbol"].item(),
            "SPCX",
        )
        self.assertEqual(
            nasdaq_universe.loc[nasdaq_universe["symbol"] == "SPAL", "confidence"].item(),
            "curated",
        )

    def test_partial_active_listing_still_rejects_unknown_inferred_rsi_symbols(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "ENGY", "name": "Ultra Energy", "fund_type": "ETF"},
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []
        partial_listing = ActiveListedSymbols(
            {"TQQQ"},
            [
                {"source": "nasdaq_listed", "status": "loaded"},
                {"source": "other_listed", "status": "error", "error": "offline"},
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=partial_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        dig = workflow_assets.loc[workflow_assets["symbol"] == "ENGY"].iloc[0]
        tqqq = workflow_assets.loc[workflow_assets["symbol"] == "TQQQ"].iloc[0]
        self.assertEqual(dig["rsi_symbol"], "ENGY")
        self.assertEqual(dig["confidence"], "fallback_to_self")
        self.assertEqual(tqqq["rsi_symbol"], "QQQ")
        self.assertNotIn("ENERGY", workflow_assets["rsi_symbol"].tolist())
        self.assertTrue(workflow_assets.attrs["universe_degraded"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Active listing sources failed"], 1)

        nasdaq_call = next(call for call in mock_save_table.call_args_list if call.args[2] == "nasdaq_etf_universe")
        nasdaq_universe = nasdaq_call.args[0]
        self.assertNotIn("ENERGY", nasdaq_universe["rsi_symbol"].tolist())

    def test_active_listing_absence_does_not_destructively_filter_issuer_only_product(self) -> None:
        nasdaq_rows = pd.DataFrame([{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}])
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "STALE",
                    "name": "Example 2X Long QQQ Daily ETF",
                    "fund_type": "ETF (Example)",
                    "source": "Example issuer table",
                }
            ]
        )
        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch(
                "leveraged_trader.universe.load_etn_universe",
                return_value=pd.DataFrame(columns=issuer_rows.columns),
            ),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["STALE", "TQQQ"])
        inactive_rows = next(
            call.args[0]
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_inactive_discovered_products"
        )
        self.assertTrue(inactive_rows.empty)
        self.assertEqual(
            workflow_assets.attrs["universe_counts"]["Inactive issuer-discovered ETFs/ETNs skipped"],
            0,
        )

    def test_complete_active_listing_filters_only_primary_products(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "ZTEST", "name": "Test UltraPro QQQ 3X ETF", "fund_type": "ETF"},
            ]
        )
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "STALE",
                    "name": "Example 2X Long QQQ Daily ETF",
                    "fund_type": "ETF (Example)",
                    "source": "Example issuer table",
                }
            ]
        )
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []
        active_listing = ActiveListedSymbols(
            {"TQQQ", "QQQ"},
            [
                {
                    "source": "nasdaq_listed",
                    "url": "https://example.test/nasdaq",
                    "symbol_column": "Symbol",
                    "status": "loaded",
                    "symbol_count": 2,
                    "error": "",
                },
                {
                    "source": "other_listed",
                    "url": "https://example.test/other",
                    "symbol_column": "ACT Symbol",
                    "status": "loaded",
                    "symbol_count": 2,
                    "error": "",
                },
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=active_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["STALE", "TQQQ"])
        inactive_rows = next(
            call.args[0]
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_inactive_discovered_products"
        )
        self.assertEqual(inactive_rows["symbol"].tolist(), ["ZTEST"])
        self.assertEqual(
            inactive_rows["inactive_source"].tolist(),
            ["primary_nasdaq"],
        )
        self.assertEqual(
            inactive_rows["inactive_reason"].tolist(),
            ["not present in active Nasdaq symbol files"],
        )
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Inactive primary Nasdaq ETFs skipped"], 1)
        self.assertEqual(
            workflow_assets.attrs["universe_counts"]["Inactive issuer-discovered ETFs/ETNs skipped"],
            0,
        )

    def test_active_listing_with_low_primary_coverage_cannot_destructively_filter(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                *[{"symbol": f"P{row}", "name": f"Plain ETF {row}", "fund_type": "ETF"} for row in range(99)],
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []
        active_listing = ActiveListedSymbols(
            {"QQQ", *(f"P{row}" for row in range(99))},
            [
                {"source": "nasdaq_listed", "status": "loaded", "symbol_count": 5_500},
                {"source": "other_listed", "status": "loaded", "symbol_count": 7_100},
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=active_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["TQQQ"])
        self.assertFalse(workflow_assets.attrs["universe_counts"]["Active listing snapshot complete"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Inactive primary Nasdaq ETFs skipped"], 0)
        saved_active_status = next(
            call.args[0]
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_active_listing_source_status"
        )
        coverage_status = saved_active_status.loc[saved_active_status["source"].eq("primary_nasdaq_coverage")].iloc[0]
        self.assertEqual(coverage_status["status"], "error")
        self.assertIn("99.00%", coverage_status["error"])

    def test_active_listing_at_primary_drift_boundaries_can_filter_inactive_products(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                *[
                    {
                        "symbol": f"ZTEST{row}",
                        "name": f"Test UltraPro QQQ 3X ETF {row}",
                        "fund_type": "ETF",
                    }
                    for row in range(5)
                ],
                *[{"symbol": f"P{row}", "name": f"Plain ETF {row}", "fund_type": "ETF"} for row in range(994)],
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []
        active_listing = ActiveListedSymbols(
            {"QQQ", "TQQQ", *(f"P{row}" for row in range(994))},
            [
                {"source": "nasdaq_listed", "status": "loaded", "symbol_count": 5_500},
                {"source": "other_listed", "status": "loaded", "symbol_count": 7_100},
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=active_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["TQQQ"])
        self.assertTrue(workflow_assets.attrs["universe_counts"]["Active listing snapshot complete"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Inactive primary Nasdaq ETFs skipped"], 5)
        inactive_rows = next(
            call.args[0]
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_inactive_discovered_products"
        )
        self.assertEqual(inactive_rows["symbol"].tolist(), [f"ZTEST{row}" for row in range(5)])

    def test_active_listing_with_too_many_missing_primary_symbols_cannot_filter(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                *[{"symbol": f"STALE{row}", "name": f"Plain stale ETF {row}", "fund_type": "ETF"} for row in range(5)],
                *[{"symbol": f"P{row}", "name": f"Plain ETF {row}", "fund_type": "ETF"} for row in range(1_194)],
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []
        active_listing = ActiveListedSymbols(
            {"QQQ", *(f"P{row}" for row in range(1_194))},
            [
                {"source": "nasdaq_listed", "status": "loaded", "symbol_count": 5_500},
                {"source": "other_listed", "status": "loaded", "symbol_count": 7_100},
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=active_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["TQQQ"])
        self.assertFalse(workflow_assets.attrs["universe_counts"]["Active listing snapshot complete"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Inactive primary Nasdaq ETFs skipped"], 0)
        saved_active_status = next(
            call.args[0]
            for call in mock_save_table.call_args_list
            if call.args[2] == "universe_active_listing_source_status"
        )
        coverage_status = saved_active_status.loc[saved_active_status["source"].eq("primary_nasdaq_coverage")].iloc[0]
        self.assertEqual(coverage_status["status"], "error")
        self.assertIn("missing 6 symbols", coverage_status["error"])

    def test_partial_active_listing_snapshot_does_not_filter_issuer_products(self) -> None:
        nasdaq_rows = pd.DataFrame([{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}])
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "ISSUER",
                    "name": "Example 2X Leveraged Broad Market ETF",
                    "fund_type": "ETF (Example)",
                    "source": "Example issuer table",
                }
            ]
        )
        partial_listing = ActiveListedSymbols(
            {"TQQQ"},
            [
                {"source": "nasdaq_listed", "status": "loaded"},
                {"source": "other_listed", "status": "error"},
            ],
        )
        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch(
                "leveraged_trader.universe.load_etn_universe",
                return_value=pd.DataFrame(columns=issuer_rows.columns),
            ),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=partial_listing),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertEqual(workflow_assets["symbol"].tolist(), ["ISSUER", "TQQQ"])
        self.assertFalse(workflow_assets.attrs["universe_counts"]["Active listing snapshot complete"])

    def test_issuer_source_failure_is_recorded_on_the_returned_universe(self) -> None:
        with (
            patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", [("Test Issuer", "https://issuer.test")]),
            patch("leveraged_trader.universe.requests.get", side_effect=RuntimeError("offline")),
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["status"], "source_error")
        self.assertIn("offline", str(status[0]["error"]))

    @patch("leveraged_trader.universe.requests.get")
    def test_cboe_malformed_candidate_table_is_recorded_as_parse_error(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr><th>Product</th><th>Symbol</th><th>Product Type</th></tr>
                <tr><td>T-REX 2X Long MSTR Daily Target ETF</td><td>MSTU</td><td>ETF</td></tr>
            </table>
            <table>
                <tr><th>Product</th><th>Symbol</th><th>Symbol</th><th>Product Type</th></tr>
                <tr><td>T-REX 2X Long TSLA Daily Target ETF</td><td>TSLL</td><td>TSLZ</td><td>ETF</td></tr>
            </table>
        """
        mock_get.return_value = response
        source = UniverseSource(
            "REX Shares",
            "https://www.cboe.com/example",
            source_type="issuer_etf",
            parser="cboe_issuer_html",
        )

        with patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", [source]):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "parse_error")
        self.assertEqual((status["parsed_row_count"], status["row_count"]), (0, 0))
        self.assertIn("ambiguous required columns", status["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_sources_fetch_concurrently_with_deterministic_results(self, mock_get: Mock) -> None:
        sources = [
            UniverseSource("Third", "https://issuer.test/third", "issuer_etf"),
            UniverseSource("First", "https://issuer.test/first", "issuer_etf"),
            UniverseSource("Second", "https://issuer.test/second", "issuer_etf"),
            UniverseSource("Fourth", "https://issuer.test/fourth", "issuer_etf"),
        ]
        symbols = {"Third": "CCC", "First": "AAA", "Second": "BBB", "Fourth": "DDD"}
        worker_pair_started = Event()
        counter_lock = Lock()
        active_fetches = 0
        max_active_fetches = 0

        def get_response(url: str, **_kwargs: object) -> Mock:
            nonlocal active_fetches, max_active_fetches
            with counter_lock:
                active_fetches += 1
                max_active_fetches = max(max_active_fetches, active_fetches)
                if active_fetches == 2:
                    worker_pair_started.set()
            if not worker_pair_started.wait(timeout=2):
                raise RuntimeError("source fetches did not overlap")
            with counter_lock:
                active_fetches -= 1
            response = Mock(status_code=200)
            response.text = url
            response.raise_for_status.return_value = None
            return response

        def parse_source(_content: str, source: UniverseSource, *, require_leveraged: bool) -> pd.DataFrame:
            self.assertFalse(require_leveraged)
            symbol = symbols[source.name]
            return pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "name": f"Example 2X Long {symbol} Daily ETF",
                        "fund_type": f"ETF ({source.name})",
                        "source": f"{source.name} issuer table",
                    }
                ]
            )

        mock_get.side_effect = get_response
        with (
            patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", sources),
            patch("leveraged_trader.universe.UNIVERSE_FETCH_MAX_WORKERS", 2),
            patch("leveraged_trader.universe._workflow_issuer_source_to_universe", side_effect=parse_source),
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertEqual(max_active_fetches, 2)
        self.assertEqual(issuer_universe["symbol"].tolist(), ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(
            [row["source"] for row in issuer_universe.attrs["workflow_source_status"]],
            [source.name for source in sources],
        )

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_response_is_parsed_once_before_filtering(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = "issuer response"
        mock_get.return_value = response
        parsed_rows = pd.DataFrame(
            [
                {
                    "symbol": "SAFE",
                    "name": "Example Income ETF",
                    "fund_type": "ETF (Example)",
                    "source": "Example issuer table",
                },
                {
                    "symbol": "FAST",
                    "name": "Example 2X Long FAST Daily ETF",
                    "fund_type": "ETF (Example)",
                    "source": "Example issuer table",
                },
            ]
        )
        source = UniverseSource("Example", "https://issuer.test", "issuer_etf")

        with (
            patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", [source]),
            patch(
                "leveraged_trader.universe._workflow_issuer_source_to_universe",
                return_value=parsed_rows,
            ) as mock_parse,
        ):
            issuer_universe = load_issuer_etf_universe()

        mock_parse.assert_called_once_with("issuer response", source, require_leveraged=False)
        self.assertEqual(issuer_universe["symbol"].tolist(), ["FAST"])
        status = issuer_universe.attrs["workflow_source_status"][0]
        self.assertEqual((status["parsed_row_count"], status["row_count"]), (2, 1))

    @patch("leveraged_trader.universe.requests.get")
    def test_etn_response_is_parsed_once_before_filtering(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = "ETN response"
        mock_get.return_value = response
        parsed_rows = pd.DataFrame(
            [
                {
                    "symbol": "PLAIN",
                    "name": "Example Index ETN",
                    "fund_type": "ETN (Example)",
                    "source": "Example ETN issuer table",
                },
                {
                    "symbol": "FAST",
                    "name": "Example 2X Long FAST ETN",
                    "fund_type": "ETN (Example)",
                    "source": "Example ETN issuer table",
                },
            ]
        )
        source = UniverseSource(
            "Example",
            "https://etn.test",
            "etn_issuer",
            parser="microsectors_html",
        )

        with (
            patch("leveraged_trader.universe.WORKFLOW_ETN_SOURCES", [source]),
            patch("leveraged_trader.universe._microsectors_html_to_universe", return_value=parsed_rows) as mock_parse,
        ):
            etn_universe = load_etn_universe()

        mock_parse.assert_called_once_with("ETN response", source, require_leveraged=False)
        self.assertEqual(etn_universe["symbol"].tolist(), ["FAST"])

    @patch("leveraged_trader.universe.requests.get")
    def test_etn_parser_exception_is_recorded_as_parse_error(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = "ETN response"
        mock_get.return_value = response
        source = UniverseSource(
            "Example",
            "https://etn.test",
            "etn_issuer",
            parser="microsectors_html",
        )

        with (
            patch("leveraged_trader.universe.WORKFLOW_ETN_SOURCES", [source]),
            patch(
                "leveraged_trader.universe._workflow_etn_source_to_universe",
                side_effect=RuntimeError("parser exploded"),
            ),
        ):
            etn_universe = load_etn_universe()

        self.assertTrue(etn_universe.empty)
        status = etn_universe.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "parse_error")
        self.assertIn("parser exploded", status["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_etracs_leverage_contradiction_fails_the_source_closed(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr><th>Ticker symbol</th><th>Name</th><th>Leverage</th></tr>
                <tr>
                    <td>/ussymbol/GOOD</td><td>Example 2X Leveraged ETN</td><td>2.00x</td>
                </tr>
                <tr>
                    <td>/ussymbol/BAD</td><td>Example 3X Leveraged ETN</td><td>2.00x</td>
                </tr>
            </table>
        """
        mock_get.return_value = response
        source = UniverseSource(
            "UBS ETRACS",
            "https://etracs.test",
            "etn_issuer",
            parser="etracs_leverage_table",
        )

        with patch("leveraged_trader.universe.WORKFLOW_ETN_SOURCES", [source]):
            etn_universe = load_etn_universe()

        self.assertTrue(etn_universe.empty)
        status = etn_universe.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "parse_error")
        self.assertEqual((status["parsed_row_count"], status["row_count"]), (0, 0))
        self.assertIn("Leverage value contradicted", status["error"])

    def test_cross_source_issuer_conflicts_exclude_symbol_and_mark_every_source(self) -> None:
        sources = [
            UniverseSource("Issuer A", "https://a.test", "issuer_etf"),
            UniverseSource("Issuer B", "https://b.test", "issuer_etf"),
        ]
        parsed_by_source = {
            "Issuer A": pd.DataFrame(
                [
                    {
                        "symbol": "XLEV",
                        "name": "NVDA 2X Daily ETF",
                        "fund_type": "ETF (Single Stock)",
                        "source": "Issuer A issuer table",
                    }
                ]
            ),
            "Issuer B": pd.DataFrame(
                [
                    {
                        "symbol": "XLEV",
                        "name": "TSLA 2X Daily ETF",
                        "fund_type": "ETF (Single Stock)",
                        "source": "Issuer B issuer table",
                    }
                ]
            ),
        }
        fetch_results = [Mock(text="response", error="") for _source in sources]

        with (
            patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", sources),
            patch("leveraged_trader.universe._fetch_enabled_sources", return_value=fetch_results),
            patch(
                "leveraged_trader.universe._workflow_issuer_source_to_universe",
                side_effect=lambda _content, source, **_kwargs: parsed_by_source[source.name],
            ),
        ):
            out = load_issuer_etf_universe()

        self.assertTrue(out.empty)
        statuses = out.attrs["workflow_source_status"]
        self.assertEqual([status["status"] for status in statuses], ["parse_error", "parse_error"])
        self.assertTrue(all("XLEV" in status["error"] for status in statuses))

    def test_source_local_conflict_cannot_reenter_from_later_issuer_sources(self) -> None:
        sources = [
            UniverseSource("Issuer A", "https://a.test", "issuer_etf"),
            UniverseSource("Issuer B", "https://b.test", "issuer_etf"),
            UniverseSource("Issuer C", "https://c.test", "issuer_etf"),
        ]
        parsed_by_source = {
            "Issuer A": pd.DataFrame(
                [
                    {"symbol": "XLEV", "name": "NVDA 2X Daily ETF", "fund_type": "ETF", "source": "A"},
                    {"symbol": "XLEV", "name": "TSLA 2X Daily ETF", "fund_type": "ETF", "source": "A"},
                ]
            ),
            "Issuer B": pd.DataFrame(
                [{"symbol": "XLEV", "name": "NVDA 2X Daily ETF", "fund_type": "ETF", "source": "B"}]
            ),
            "Issuer C": pd.DataFrame(
                [{"symbol": "XLEV", "name": "NVDA 2X Daily ETF", "fund_type": "ETF", "source": "C"}]
            ),
        }
        with (
            patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", sources),
            patch(
                "leveraged_trader.universe._fetch_enabled_sources",
                return_value=[Mock(text="response", error="") for _source in sources],
            ),
            patch(
                "leveraged_trader.universe._workflow_issuer_source_to_universe",
                side_effect=lambda _content, source, **_kwargs: parsed_by_source[source.name],
            ),
        ):
            out = load_issuer_etf_universe()

        self.assertTrue(out.empty)
        statuses = out.attrs["workflow_source_status"]
        self.assertEqual([status["status"] for status in statuses], ["parse_error"] * 3)
        self.assertTrue(all("XLEV" in status["error"] for status in statuses))

    def test_leveraged_and_ordinary_cross_source_rows_conflict_before_filtering(self) -> None:
        cases = (
            (
                load_issuer_etf_universe,
                "ISSUER_UNIVERSE_SOURCES",
                "_workflow_issuer_source_to_universe",
                "issuer_etf",
                "ETF",
            ),
            (
                load_etn_universe,
                "WORKFLOW_ETN_SOURCES",
                "_workflow_etn_source_to_universe",
                "etn_issuer",
                "ETN",
            ),
        )
        for loader, sources_attribute, parser_name, source_type, product_type in cases:
            sources = [
                UniverseSource("Source A", "https://a.test", source_type),
                UniverseSource("Source B", "https://b.test", source_type),
            ]
            parsed_by_source = {
                "Source A": pd.DataFrame(
                    [
                        {
                            "symbol": "XLEV",
                            "name": f"NVDA 2X Daily {product_type}",
                            "fund_type": product_type,
                            "source": "A",
                        }
                    ]
                ),
                "Source B": pd.DataFrame(
                    [
                        {
                            "symbol": "XLEV",
                            "name": f"Ordinary Income {product_type}",
                            "fund_type": product_type,
                            "source": "B",
                        }
                    ]
                ),
            }
            with (
                self.subTest(loader=loader.__name__),
                patch(f"leveraged_trader.universe.{sources_attribute}", sources),
                patch(
                    "leveraged_trader.universe._fetch_enabled_sources",
                    return_value=[Mock(text="response", error="") for _source in sources],
                ),
                patch(
                    f"leveraged_trader.universe.{parser_name}",
                    side_effect=lambda _content, source, rows=parsed_by_source, **_kwargs: rows[source.name],
                ),
            ):
                out = loader()

            self.assertTrue(out.empty)
            statuses = out.attrs["workflow_source_status"]
            self.assertEqual([status["status"] for status in statuses], ["parse_error", "parse_error"])
            self.assertTrue(all("XLEV" in status["error"] for status in statuses))

    def test_equivalent_cross_source_issuer_duplicates_ignore_fund_type_source_label(self) -> None:
        sources = [
            UniverseSource("Issuer A", "https://a.test", "issuer_etf"),
            UniverseSource("Issuer B", "https://b.test", "issuer_etf"),
        ]
        parsed_by_source = {
            source.name: pd.DataFrame(
                [
                    {
                        "symbol": "XLEV",
                        "name": "NVDA 2X Daily ETF",
                        "fund_type": f"ETF ({source.name})",
                        "source": f"{source.name} issuer table",
                    }
                ]
            )
            for source in sources
        }
        with (
            patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", sources),
            patch(
                "leveraged_trader.universe._fetch_enabled_sources",
                return_value=[Mock(text="response", error=""), Mock(text="response", error="")],
            ),
            patch(
                "leveraged_trader.universe._workflow_issuer_source_to_universe",
                side_effect=lambda _content, source, **_kwargs: parsed_by_source[source.name],
            ),
        ):
            out = load_issuer_etf_universe()

        self.assertEqual(out["symbol"].tolist(), ["XLEV"])
        self.assertEqual(
            [status["status"] for status in out.attrs["workflow_source_status"]],
            ["loaded", "loaded"],
        )

    def test_equivalent_nonleveraged_duplicates_ignore_unused_mapping_metadata(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "symbol": "AHD",
                    "name": "GraniteShares Autocallable HOOD ETF",
                    "fund_type": "ETF (Single Stock)",
                    "source": "Nasdaq ETF definitions",
                },
                {
                    "symbol": "AHD",
                    "name": "GraniteShares Autocallable HOOD ETF",
                    "fund_type": "ETF (GraniteShares)",
                    "source": "GraniteShares issuer table",
                },
            ]
        )

        resolved, conflicting_symbols = _leveraged_product_rows(rows)

        self.assertTrue(resolved.empty)
        self.assertEqual(conflicting_symbols, [])

    def test_source_local_etf_etn_structure_conflict_excludes_symbol(self) -> None:
        source = UniverseSource("Issuer A", "https://a.test", "issuer_etf")
        parsed_rows = pd.DataFrame(
            [
                {
                    "symbol": "XLEV",
                    "name": "NVDA 2X Daily ETF",
                    "fund_type": "ETF (Issuer A)",
                    "source": "Issuer A issuer table",
                },
                {
                    "symbol": "XLEV",
                    "name": "NVDA 2X Daily ETN",
                    "fund_type": "ETN (Issuer A)",
                    "source": "Issuer A issuer table",
                },
            ]
        )
        with (
            patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", [source]),
            patch(
                "leveraged_trader.universe._fetch_enabled_sources",
                return_value=[Mock(text="response", error="")],
            ),
            patch(
                "leveraged_trader.universe._workflow_issuer_source_to_universe",
                return_value=parsed_rows,
            ),
        ):
            out = load_issuer_etf_universe()

        self.assertTrue(out.empty)
        status = out.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "parse_error")
        self.assertIn("product structures", status["error"])
        self.assertIn("XLEV", status["error"])

    def test_cross_source_etn_conflicts_exclude_symbol_and_mark_every_source(self) -> None:
        sources = [
            UniverseSource("ETN A", "https://a.test", "etn_issuer"),
            UniverseSource("ETN B", "https://b.test", "etn_issuer"),
        ]
        parsed_by_source = {
            "ETN A": pd.DataFrame(
                [{"symbol": "XLEV", "name": "NVDA 2X Daily ETN", "fund_type": "ETN", "source": "ETN A"}]
            ),
            "ETN B": pd.DataFrame(
                [{"symbol": "XLEV", "name": "TSLA 2X Daily ETN", "fund_type": "ETN", "source": "ETN B"}]
            ),
        }
        with (
            patch("leveraged_trader.universe.WORKFLOW_ETN_SOURCES", sources),
            patch(
                "leveraged_trader.universe._fetch_enabled_sources",
                return_value=[Mock(text="response", error=""), Mock(text="response", error="")],
            ),
            patch(
                "leveraged_trader.universe._workflow_etn_source_to_universe",
                side_effect=lambda _content, source: parsed_by_source[source.name],
            ),
        ):
            out = load_etn_universe()

        self.assertTrue(out.empty)
        statuses = out.attrs["workflow_source_status"]
        self.assertEqual([status["status"] for status in statuses], ["parse_error", "parse_error"])
        self.assertTrue(all("XLEV" in status["error"] for status in statuses))

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_source_with_unparseable_success_response_is_a_parse_error(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = "<html><body>temporary maintenance page</body></html>"
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [("Test Issuer", "https://issuer.test")],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertTrue(status)
        self.assertTrue(all(row["status"] == "parse_error" for row in status))
        self.assertTrue(all("No product rows" in str(row["error"]) for row in status))

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_source_with_only_missing_fund_names_is_a_parse_error(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr><th>Ticker</th><th>Fund Name</th></tr>
                <tr><td>AAA</td><td></td></tr>
                <tr><td>BBB</td><td> </td></tr>
            </table>
        """
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [("Test Issuer", "https://issuer.test")],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["status"], "parse_error")
        self.assertIn("required product name", status[0]["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_source_with_empty_success_response_is_a_parse_error(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = ""
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [("Test Issuer", "https://issuer.test")],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["status"], "parse_error")
        self.assertIn("response body was empty", status[0]["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_parser_exception_is_recorded_without_aborting(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = "<html>broken parser input</html>"
        mock_get.return_value = response

        with (
            patch(
                "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
                [("Test Issuer", "https://issuer.test")],
            ),
            patch(
                "leveraged_trader.universe._workflow_issuer_source_to_universe",
                side_effect=RuntimeError("parser exploded"),
            ),
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["status"], "parse_error")
        self.assertIn("parser exploded", status[0]["error"])

    def test_registered_only_workflow_source_is_healthy(self) -> None:
        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [
                UniverseSource(
                    "Blocked Issuer",
                    "https://issuer.test",
                    "issuer_etf",
                    enabled=False,
                    notes="blocked",
                )
            ],
        ):
            issuer_universe = load_issuer_etf_universe()

        status = issuer_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["status"], "registered_only")
        self.assertIn("blocked", status[0]["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_source_with_valid_zero_leveraged_matches_is_healthy(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr><th>Ticker</th><th>Fund Name</th></tr>
                <tr><td>SAFE</td><td>Acme Income ETF</td></tr>
            </table>
        """
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [("Test Issuer", "https://issuer.test")],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"]
        self.assertEqual(status[0]["status"], "loaded_zero_matches")
        self.assertEqual(status[0]["parsed_row_count"], 1)
        self.assertEqual(status[0]["row_count"], 0)

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_parser_marks_conflicting_product_variants_as_parse_error(
        self,
        mock_get: Mock,
    ) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr><th>Ticker</th><th>Fund Name</th></tr>
                <tr><td>ZZZ</td><td>Acme Income ETF</td></tr>
                <tr><td>ZZZ</td><td>Acme ZZZ 2X Daily Leveraged ETF</td></tr>
            </table>
        """
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [("Test Issuer", "https://issuer.test")],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "parse_error")
        self.assertEqual((status["parsed_row_count"], status["row_count"]), (2, 0))
        self.assertIn("ZZZ", status["error"])
        self.assertIn("Conflicting leverage classifications", status["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_parser_retains_safe_rows_while_marking_partial_conflicts(
        self,
        mock_get: Mock,
    ) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr><th>Ticker</th><th>Fund Name</th></tr>
                <tr><td>SAFE</td><td>Acme SAFE 2X Daily Leveraged ETF</td></tr>
                <tr><td>ZZZ</td><td>Acme Income ETF</td></tr>
                <tr><td>ZZZ</td><td>Acme ZZZ 2X Daily Leveraged ETF</td></tr>
            </table>
        """
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [("Test Issuer", "https://issuer.test")],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertEqual(issuer_universe["symbol"].tolist(), ["SAFE"])
        status = issuer_universe.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "parse_error")
        self.assertEqual((status["parsed_row_count"], status["row_count"]), (3, 1))
        self.assertIn("ZZZ", status["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_issuer_parser_excludes_duplicate_symbol_with_conflicting_rsi_underlyings(
        self,
        mock_get: Mock,
    ) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr><th>Ticker</th><th>Fund Name</th></tr>
                <tr><td>SAFE</td><td>Acme SAFE 2X Daily Leveraged ETF</td></tr>
                <tr><td>DUP</td><td>Example 2X Long NVDA Daily ETF</td></tr>
            </table>
            <table>
                <tr><th>Symbol</th><th>Product Name</th></tr>
                <tr><td>DUP</td><td>Example 2X Long TSLA Daily ETF</td></tr>
            </table>
        """
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [("Test Issuer", "https://issuer.test")],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertEqual(issuer_universe["symbol"].tolist(), ["SAFE"])
        status = issuer_universe.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "parse_error")
        self.assertEqual((status["parsed_row_count"], status["row_count"]), (3, 1))
        self.assertIn("RSI-mapping metadata", status["error"])
        self.assertIn("DUP", status["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_tradr_stale_leveraged_duplicate_cannot_override_current_one_x_row(
        self,
        mock_get: Mock,
    ) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th><th>Target</th>
                    <th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>SARK</td><td>Tradr 1X Short Innovation Daily ETF</td>
                    <td>ARKK</td><td>-1X</td><td>Short</td><td>Daily</td>
                </tr>
                <tr>
                    <td>TARK</td><td>Tradr 2X Long Innovation Daily ETF</td><td>ARKK</td>
                    <td>2X</td><td>Long</td><td>Daily</td>
                </tr>
            </table>
            <table>
                <tr><th>Ticker</th><th>Fund Name</th><th>Reset Period</th></tr>
                <tr><td>SARK</td><td>Tradr 2X Short Innovation Daily ETF</td><td>Daily</td></tr>
                <tr><td>SPYM</td><td>Tradr 2X Long SPY Monthly ETF</td><td>Monthly</td></tr>
            </table>
        """
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [
                UniverseSource(
                    "Tradr",
                    "https://www.tradretfs.com/",
                    "issuer_etf",
                    parser="tradr_html",
                )
            ],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertEqual(issuer_universe["symbol"].tolist(), ["TARK"])
        status = issuer_universe.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "loaded")
        self.assertEqual((status["parsed_row_count"], status["row_count"]), (2, 1))

    @patch("leveraged_trader.universe.requests.get")
    def test_tradr_malformed_partial_feed_fails_the_source_closed(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th><th>Target</th>
                    <th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>GOOD</td><td>Tradr 2X Long TEST Daily ETF</td>
                    <td>TEST</td><td>2X</td><td>Long</td><td>Daily</td>
                </tr>
                <tr>
                    <td>BAD</td><td></td><td>TEST</td><td>2X</td><td>Long</td><td>Daily</td>
                </tr>
            </table>
        """
        mock_get.return_value = response

        with patch(
            "leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES",
            [
                UniverseSource(
                    "Tradr",
                    "https://www.tradretfs.com/",
                    "issuer_etf",
                    parser="tradr_html",
                )
            ],
        ):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "parse_error")
        self.assertEqual((status["parsed_row_count"], status["row_count"]), (0, 0))
        self.assertIn("required product name", status["error"])

    @patch("leveraged_trader.universe.requests.get")
    def test_tradr_reference_security_contradiction_fails_the_source_closed(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, text="")
        response.raise_for_status.return_value = None
        response.text = """
            <table>
                <tr>
                    <th>Ticker</th><th>Fund Name</th><th>Reference Security</th>
                    <th>Target</th><th>Exposure</th><th>Reset Period</th>
                </tr>
                <tr>
                    <td>WRONG</td><td>Tradr 2X Long NVDA Daily ETF</td><td>TSLA</td>
                    <td>2X</td><td>Long</td><td>Daily</td>
                </tr>
            </table>
        """
        mock_get.return_value = response
        source = UniverseSource(
            "Tradr",
            "https://www.tradretfs.com/",
            "issuer_etf",
            parser="tradr_html",
        )

        with patch("leveraged_trader.universe.ISSUER_UNIVERSE_SOURCES", [source]):
            issuer_universe = load_issuer_etf_universe()

        self.assertTrue(issuer_universe.empty)
        status = issuer_universe.attrs["workflow_source_status"][0]
        self.assertEqual(status["status"], "parse_error")
        self.assertIn("Reference Security contradicted", status["error"])

    def test_strict_workflow_source_mode_aborts_after_recording_source_health(self) -> None:
        nasdaq_rows = pd.DataFrame([{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}])
        issuer_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Test Issuer",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "error",
                "row_count": 0,
                "error": "offline",
            }
        ]
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []
        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "Test Issuer"),
        ):
            determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        self.assertIn(
            "universe_workflow_source_status",
            [call.args[2] for call in mock_save_table.call_args_list],
        )

    def test_all_empty_workflow_sources_persist_health_before_failing_closed(self) -> None:
        def empty_source(source: str, source_type: str) -> pd.DataFrame:
            rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
            rows.attrs["workflow_source_status"] = [
                {
                    "source": source,
                    "source_type": source_type,
                    "url": f"https://{source.lower()}.test",
                    "status": "source_error",
                    "parsed_row_count": 0,
                    "row_count": 0,
                    "error": "offline",
                }
            ]
            return rows

        nasdaq_rows = empty_source("Nasdaq", "nasdaq_etf")
        issuer_rows = empty_source("Issuer", "issuer_etf")
        etn_rows = empty_source("ETN", "workflow_etn")
        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=set()),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "Workflow universe source checks failed"),
        ):
            determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        saved_tables = {call.args[2]: call.args[0] for call in mock_save_table.call_args_list}
        self.assertIn("nasdaq_etf_universe", saved_tables)
        self.assertTrue(saved_tables["nasdaq_etf_universe"].empty)
        self.assertTrue(saved_tables["nasdaq_etf_universe"].columns.is_unique)
        self.assertIn("universe_workflow_source_status", saved_tables)
        self.assertEqual(
            saved_tables["universe_workflow_source_status"]["source"].tolist(),
            ["Nasdaq", "Issuer", "ETN"],
        )

    def test_strict_workflow_source_mode_aborts_on_active_listing_failure(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "fund_type": "ETF"},
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
            ]
        )
        empty_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        empty_rows.attrs["workflow_source_status"] = []
        partial_listing = ActiveListedSymbols(
            {"TQQQ"},
            [
                {"source": "nasdaq_listed", "status": "loaded"},
                {"source": "other_listed", "status": "error", "error": "offline"},
            ],
        )

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=empty_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value=partial_listing),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "active listing sources were unusable: other_listed"),
        ):
            determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        self.assertIn(
            "universe_active_listing_source_status",
            [call.args[2] for call in mock_save_table.call_args_list],
        )

    def test_primary_etf_discovered_etn_conflict_excludes_symbol_and_fails_strict_mode(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "XLEV", "name": "NVDA 2X Daily ETF", "fund_type": "ETF"},
            ]
        )
        issuer_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        issuer_rows.attrs["workflow_source_status"] = []
        etn_rows = pd.DataFrame(
            [
                {
                    "symbol": "XLEV",
                    "name": "NVDA 2X Daily ETN",
                    "fund_type": "ETN (Single Stock)",
                    "source": "ETN B issuer table",
                }
            ]
        )
        etn_rows.attrs["workflow_source_status"] = [
            {
                "source": "ETN B",
                "source_type": "etn_issuer",
                "url": "https://etn.test",
                "status": "loaded",
                "parsed_row_count": 1,
                "row_count": 1,
                "error": "",
            }
        ]
        etn_rows.attrs["workflow_symbol_sources"] = {"XLEV": (("ETN B", "https://etn.test"),)}

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ", "XLEV"}),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "Nasdaq ETF definitions, ETN B"),
        ):
            determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        saved_tables = {call.args[2]: call.args[0] for call in mock_save_table.call_args_list}
        self.assertNotIn("XLEV", saved_tables["nasdaq_etf_universe"]["symbol"].tolist())
        conflicting = saved_tables["universe_workflow_source_status"].loc[
            lambda rows: rows["source"].isin(["Nasdaq ETF definitions", "ETN B"])
        ]
        self.assertEqual(conflicting["status"].tolist(), ["parse_error", "parse_error"])
        self.assertTrue(conflicting["error"].str.contains("XLEV").all())

    def test_equivalent_primary_and_discovered_etf_duplicates_preserve_nasdaq_precedence(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "XLEV", "name": "NVDA 2X Daily ETF", "fund_type": "ETF"},
            ]
        )
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "XLEV",
                    "name": "NVDA 2X Long Daily ETF",
                    "fund_type": "ETF (Single Stock)",
                    "source": "Issuer A issuer table",
                }
            ]
        )
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Issuer A",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "loaded",
                "parsed_row_count": 1,
                "row_count": 1,
                "error": "",
            }
        ]
        issuer_rows.attrs["workflow_symbol_sources"] = {"XLEV": (("Issuer A", "https://issuer.test"),)}
        etn_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"TQQQ", "QQQ", "XLEV", "NVDA"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            groups = determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        saved_tables = {call.args[2]: call.args[0] for call in mock_save_table.call_args_list}
        xlev = saved_tables["nasdaq_etf_universe"].loc[lambda rows: rows["symbol"].eq("XLEV")].iloc[0]
        self.assertEqual(xlev["name"], "NVDA 2X Daily ETF")
        self.assertEqual(xlev["fund_type"], "ETF")
        self.assertEqual(xlev["source"], "Nasdaq ETF definitions")
        self.assertIn("XLEV", groups["long"]["symbol"].tolist())
        self.assertEqual(
            saved_tables["universe_workflow_source_status"]["status"].tolist(),
            ["loaded", "loaded"],
        )

    def test_primary_row_accepts_validated_discovered_reference_enrichment(self) -> None:
        name = "Tradr 2X Long Innovation Daily ETF"
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "FOOX", "name": name, "fund_type": "ETF"},
            ]
        )
        issuer_record = {
            "symbol": "FOOX",
            "name": name,
            "fund_type": "ETF (Issuer)",
            "source": "Issuer A issuer table",
            "reference_security": "SPY",
        }
        issuer_rows = pd.DataFrame([issuer_record])
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Issuer A",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "loaded",
                "parsed_row_count": 1,
                "row_count": 1,
                "error": "",
            }
        ]
        issuer_rows.attrs["workflow_symbol_sources"] = {"FOOX": (("Issuer A", "https://issuer.test"),)}
        issuer_rows.attrs["workflow_canonical_product_rows"] = [issuer_record]
        etn_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"TQQQ", "QQQ", "FOOX", "SPY"},
            ),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
        ):
            groups = determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        saved_tables = {call.args[2]: call.args[0] for call in mock_save_table.call_args_list}
        foox = saved_tables["nasdaq_etf_universe"].loc[lambda rows: rows["symbol"].eq("FOOX")].iloc[0]
        self.assertEqual(foox["source"], "Nasdaq ETF definitions")
        self.assertEqual(foox["reference_security"], "SPY")
        self.assertEqual(foox["rsi_symbol"], "SPY")
        self.assertEqual(foox["mapping_source"], "issuer_reference")
        self.assertIn("FOOX", groups["long"]["symbol"].tolist())
        self.assertEqual(
            saved_tables["universe_workflow_source_status"]["status"].tolist(),
            ["loaded", "loaded"],
        )

    def test_compatible_duplicate_issuer_rows_coalesce_reference_in_either_order(self) -> None:
        name = "Tradr 2X Long Innovation Daily ETF"
        sources = (
            UniverseSource("Issuer A", "https://issuer-a.test", "issuer_etf"),
            UniverseSource("Issuer B", "https://issuer-b.test", "issuer_etf"),
        )
        unreferenced_record = {
            "symbol": "FOOX",
            "name": name,
            "fund_type": "ETF (Issuer A)",
            "source": "Issuer A issuer table",
        }
        referenced_record = {
            "symbol": "FOOX",
            "name": name,
            "fund_type": "ETF (Issuer B)",
            "source": "Issuer B issuer table",
            "reference_security": "SPY",
        }

        source_records = ((sources[0], unreferenced_record), (sources[1], referenced_record))
        resolved_rows_by_order: list[dict[str, object]] = []
        persisted_rows_by_order: list[dict[str, object]] = []
        for reference_first in (False, True):
            ordered_source_records = tuple(reversed(source_records)) if reference_first else source_records
            ordered_records = tuple(record for _source, record in ordered_source_records)
            source_frames = [pd.DataFrame([record]) for record in ordered_records]
            status_rows = [
                {
                    "source": source.name,
                    "url": source.url,
                    "status": "loaded",
                    "error": "",
                }
                for source, _record in ordered_source_records
            ]
            issuer_rows = _resolve_workflow_source_product_rows(
                [
                    (index, source, source_frame)
                    for index, ((source, _record), source_frame) in enumerate(
                        zip(ordered_source_records, source_frames, strict=True)
                    )
                ],
                status_rows,
            )
            nasdaq_rows = pd.DataFrame([{"symbol": "FOOX", "name": name, "fund_type": "ETF"}])
            etn_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
            resolved_nasdaq, resolved_discovered, resolved_status = _resolve_discovered_product_rows(
                nasdaq_rows,
                issuer_rows,
                etn_rows,
                pd.DataFrame(status_rows),
            )
            merged = _merge_universe_sources(resolved_nasdaq, resolved_discovered)
            mapped = build_nasdaq_universe_table(merged, known_symbols={"FOOX", "SPY"})
            _single_stock, long_products = select_universes(merged)
            candidates = _workflow_candidates(long_products, {"FOOX", "SPY"}, workflow_label="Long")
            executable = candidates.loc[~candidates["confidence"].eq("needs_review")]

            resolved_row = resolved_discovered.loc[
                0,
                ["symbol", "name", "fund_type", "source", "reference_security"],
            ].to_dict()
            persisted_row = mapped.loc[
                0,
                [
                    "symbol",
                    "name",
                    "fund_type",
                    "source",
                    "reference_security",
                    "rsi_symbol",
                    "mapping_source",
                    "confidence",
                ],
            ].to_dict()
            resolved_rows_by_order.append(resolved_row)
            persisted_rows_by_order.append(persisted_row)

            with self.subTest(reference_first=reference_first):
                self.assertEqual(resolved_status["status"].tolist(), ["loaded", "loaded"])
                self.assertEqual(issuer_rows.loc[0, "source"], "Issuer B issuer table")
                self.assertEqual(issuer_rows.loc[0, "fund_type"], "ETF (Issuer B)")
                self.assertEqual(resolved_discovered.loc[0, "reference_security"], "SPY")
                self.assertEqual(resolved_discovered.loc[0, "source"], "Issuer B issuer table")
                self.assertEqual(resolved_discovered.loc[0, "fund_type"], "ETF (Issuer B)")
                self.assertEqual(
                    set(issuer_rows.attrs["workflow_symbol_sources"]["FOOX"]),
                    {(source.name, source.url) for source in sources},
                )
                self.assertEqual(
                    {record["source"] for record in issuer_rows.attrs["workflow_canonical_product_rows"]},
                    {"Issuer A issuer table", "Issuer B issuer table"},
                )
                self.assertEqual(merged.loc[0, "reference_security"], "SPY")
                self.assertEqual(merged.loc[0, "name"], name)
                self.assertEqual(merged.loc[0, "fund_type"], "ETF")
                self.assertEqual(merged.loc[0, "source"], "Nasdaq ETF definitions")
                self.assertEqual(mapped.loc[0, "rsi_symbol"], "SPY")
                self.assertEqual(mapped.loc[0, "mapping_source"], "issuer_reference")
                self.assertEqual(mapped.loc[0, "confidence"], "curated")
                self.assertEqual(executable["symbol"].tolist(), ["FOOX"])
                self.assertEqual(executable.loc[0, "rsi_symbol"], "SPY")
                self.assertEqual(source_frames[0].to_dict("records"), [ordered_records[0]])
                self.assertEqual(source_frames[1].to_dict("records"), [ordered_records[1]])

        self.assertEqual(resolved_rows_by_order[0], resolved_rows_by_order[1])
        self.assertEqual(persisted_rows_by_order[0], persisted_rows_by_order[1])

    def test_duplicate_issuer_rows_with_distinct_references_remain_conflicting(self) -> None:
        name = "Tradr 2X Long Innovation Daily ETF"
        sources = (
            UniverseSource("Issuer A", "https://issuer-a.test", "issuer_etf"),
            UniverseSource("Issuer B", "https://issuer-b.test", "issuer_etf"),
        )
        for references in (("SPY", "QQQ"), ("BRK.B", "BRK-B")):
            status_rows = [
                {
                    "source": source.name,
                    "url": source.url,
                    "status": "loaded",
                    "error": "",
                }
                for source in sources
            ]
            resolved = _resolve_workflow_source_product_rows(
                [
                    (
                        index,
                        source,
                        pd.DataFrame(
                            [
                                {
                                    "symbol": "FOOX",
                                    "name": name,
                                    "fund_type": f"ETF ({source.name})",
                                    "source": f"{source.name} issuer table",
                                    "reference_security": reference,
                                }
                            ]
                        ),
                    )
                    for index, (source, reference) in enumerate(zip(sources, references, strict=True))
                ],
                status_rows,
            )

            with self.subTest(references=references):
                self.assertTrue(resolved.empty)
                self.assertEqual([status["status"] for status in status_rows], ["parse_error", "parse_error"])
                self.assertTrue(all("FOOX" in str(status["error"]) for status in status_rows))

    def test_invalid_duplicate_reference_fails_closed_in_either_order(self) -> None:
        name = "Tradr 2X Long Innovation Daily ETF"
        sources = (
            UniverseSource("Issuer A", "https://issuer-a.test", "issuer_etf"),
            UniverseSource("Issuer B", "https://issuer-b.test", "issuer_etf"),
        )
        missing_record = {
            "symbol": "FOOX",
            "name": name,
            "fund_type": "ETF (Issuer A)",
            "source": "Issuer A issuer table",
        }
        for invalid_reference in ("spy", "BRK.B", 123):
            invalid_record = {
                "symbol": "FOOX",
                "name": name,
                "fund_type": "ETF (Issuer B)",
                "source": "Issuer B issuer table",
                "reference_security": invalid_reference,
            }
            source_records = ((sources[0], missing_record), (sources[1], invalid_record))
            for invalid_first in (False, True):
                ordered_source_records = tuple(reversed(source_records)) if invalid_first else source_records
                status_rows = [
                    {
                        "source": source.name,
                        "url": source.url,
                        "status": "loaded",
                        "error": "",
                    }
                    for source, _record in ordered_source_records
                ]
                resolved = _resolve_workflow_source_product_rows(
                    [
                        (index, source, pd.DataFrame([record]))
                        for index, (source, record) in enumerate(ordered_source_records)
                    ],
                    status_rows,
                )

                with self.subTest(invalid_reference=invalid_reference, invalid_first=invalid_first):
                    self.assertTrue(resolved.empty)
                    self.assertEqual([status["status"] for status in status_rows], ["parse_error", "parse_error"])
                    self.assertTrue(all("FOOX" in str(status["error"]) for status in status_rows))

    def test_primary_mapping_disagreement_with_discovered_reference_remains_a_conflict(self) -> None:
        nasdaq_rows = pd.DataFrame(
            [
                {"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"},
                {"symbol": "FOOX", "name": "Tradr 2X Long NVDA Daily ETF", "fund_type": "ETF"},
            ]
        )
        issuer_record = {
            "symbol": "FOOX",
            "name": "Tradr 2X Long Innovation Daily ETF",
            "fund_type": "ETF (Issuer)",
            "source": "Issuer A issuer table",
            "reference_security": "SPY",
        }
        issuer_rows = pd.DataFrame([issuer_record])
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Issuer A",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "loaded",
                "parsed_row_count": 1,
                "row_count": 1,
                "error": "",
            }
        ]
        issuer_rows.attrs["workflow_symbol_sources"] = {"FOOX": (("Issuer A", "https://issuer.test"),)}
        issuer_rows.attrs["workflow_canonical_product_rows"] = [issuer_record]
        etn_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch(
                "leveraged_trader.universe.load_active_listed_symbols",
                return_value={"TQQQ", "QQQ", "FOOX", "NVDA", "SPY"},
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "Nasdaq ETF definitions, Issuer A"),
        ):
            determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        saved_tables = {call.args[2]: call.args[0] for call in mock_save_table.call_args_list}
        self.assertNotIn("FOOX", saved_tables["nasdaq_etf_universe"]["symbol"].tolist())
        conflicting = saved_tables["universe_workflow_source_status"].loc[
            lambda rows: rows["source"].isin(["Nasdaq ETF definitions", "Issuer A"])
        ]
        self.assertEqual(conflicting["status"].tolist(), ["parse_error", "parse_error"])
        self.assertTrue(conflicting["error"].str.contains("FOOX").all())

    def test_issuer_etn_conflict_is_revalidated_and_fails_strict_source_mode(self) -> None:
        nasdaq_rows = pd.DataFrame([{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}])
        issuer_rows = pd.DataFrame(
            [
                {
                    "symbol": "XLEV",
                    "name": "NVDA 2X Daily ETF",
                    "fund_type": "ETF (Single Stock)",
                    "source": "Issuer A issuer table",
                }
            ]
        )
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Issuer A",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "loaded",
                "parsed_row_count": 1,
                "row_count": 1,
                "error": "",
            }
        ]
        issuer_rows.attrs["workflow_symbol_sources"] = {"XLEV": (("Issuer A", "https://issuer.test"),)}
        etn_rows = pd.DataFrame(
            [
                {
                    "symbol": "XLEV",
                    "name": "NVDA 2X Daily ETN",
                    "fund_type": "ETN (Single Stock)",
                    "source": "ETN B issuer table",
                }
            ]
        )
        etn_rows.attrs["workflow_source_status"] = [
            {
                "source": "ETN B",
                "source_type": "etn_issuer",
                "url": "https://etn.test",
                "status": "loaded",
                "parsed_row_count": 1,
                "row_count": 1,
                "error": "",
            }
        ]
        etn_rows.attrs["workflow_symbol_sources"] = {"XLEV": (("ETN B", "https://etn.test"),)}

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ", "XLEV"}),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "Issuer A, ETN B"),
        ):
            determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        saved_status = next(
            call.args[0] for call in mock_save_table.call_args_list if call.args[2] == "universe_workflow_source_status"
        )
        conflicting = saved_status.loc[saved_status["source"].isin(["Issuer A", "ETN B"])]
        self.assertEqual(conflicting["status"].tolist(), ["parse_error", "parse_error"])
        self.assertTrue(conflicting["error"].str.contains("XLEV").all())

    def test_issuer_etn_conflict_marks_all_three_intermediate_contributors(self) -> None:
        nasdaq_rows = pd.DataFrame([{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}])
        issuer_record = {
            "symbol": "XLEV",
            "name": "NVDA 2X Daily ETF",
            "fund_type": "ETF (Single Stock)",
            "source": "Issuer table",
        }
        issuer_rows = pd.DataFrame([issuer_record])
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": source,
                "source_type": "issuer_etf",
                "url": url,
                "status": "loaded",
                "parsed_row_count": 1,
                "row_count": 1,
                "error": "",
            }
            for source, url in (
                ("Issuer A", "https://a.test"),
                ("Issuer B", "https://b.test"),
            )
        ]
        issuer_rows.attrs["workflow_symbol_sources"] = {
            "XLEV": (("Issuer A", "https://a.test"), ("Issuer B", "https://b.test"))
        }
        issuer_rows.attrs["workflow_canonical_product_rows"] = [issuer_record, issuer_record]

        etn_record = {
            "symbol": "XLEV",
            "name": "TSLA 2X Daily ETN",
            "fund_type": "ETN (Single Stock)",
            "source": "ETN table",
        }
        etn_rows = pd.DataFrame([etn_record])
        etn_rows.attrs["workflow_source_status"] = [
            {
                "source": "ETN C",
                "source_type": "etn_issuer",
                "url": "https://c.test",
                "status": "loaded",
                "parsed_row_count": 1,
                "row_count": 1,
                "error": "",
            }
        ]
        etn_rows.attrs["workflow_symbol_sources"] = {"XLEV": (("ETN C", "https://c.test"),)}
        etn_rows.attrs["workflow_canonical_product_rows"] = [etn_record]

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ", "XLEV"}),
            patch("leveraged_trader.universe.save_table_to_sqlite") as mock_save_table,
            self.assertRaisesRegex(RuntimeError, "Issuer A, Issuer B, ETN C"),
        ):
            determine_workflow_asset_groups(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        saved_status = next(
            call.args[0] for call in mock_save_table.call_args_list if call.args[2] == "universe_workflow_source_status"
        )
        conflicting = saved_status.loc[saved_status["source"].isin(["Issuer A", "Issuer B", "ETN C"])]
        self.assertEqual(conflicting["status"].tolist(), ["parse_error"] * 3)
        self.assertTrue(conflicting["error"].str.contains("XLEV").all())

    def test_parse_error_workflow_source_marks_the_universe_degraded(self) -> None:
        nasdaq_rows = pd.DataFrame([{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}])
        issuer_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Test Issuer",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "parse_error",
                "row_count": 0,
                "error": "No product rows could be parsed from a successful source response.",
            }
        ]
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_assets = determine_workflow_assets(UniverseConfig(sqlite_db_path="state.sqlite"))

        self.assertTrue(workflow_assets.attrs["universe_degraded"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Workflow universe sources failed"], 1)

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
            self.assertRaisesRegex(RuntimeError, "Test Issuer"),
        ):
            determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

    def test_zero_match_workflow_source_is_healthy_in_strict_mode(self) -> None:
        nasdaq_rows = pd.DataFrame([{"symbol": "TQQQ", "name": "ProShares UltraPro QQQ", "fund_type": "ETF"}])
        issuer_rows = pd.DataFrame(columns=["symbol", "name", "fund_type", "source"])
        issuer_rows.attrs["workflow_source_status"] = [
            {
                "source": "Test Issuer",
                "source_type": "issuer_etf",
                "url": "https://issuer.test",
                "status": "loaded_zero_matches",
                "parsed_row_count": 1,
                "row_count": 0,
                "error": "",
            }
        ]
        etn_rows = pd.DataFrame(columns=issuer_rows.columns)
        etn_rows.attrs["workflow_source_status"] = []

        with (
            patch("leveraged_trader.universe.load_current_etf_universe", return_value=nasdaq_rows),
            patch("leveraged_trader.universe.load_issuer_etf_universe", return_value=issuer_rows),
            patch("leveraged_trader.universe.load_etn_universe", return_value=etn_rows),
            patch("leveraged_trader.universe.load_active_listed_symbols", return_value={"TQQQ", "QQQ"}),
            patch(
                "leveraged_trader.universe.load_audit_universe_sources",
                return_value=(pd.DataFrame(), pd.DataFrame()),
            ),
            patch("leveraged_trader.universe.save_table_to_sqlite"),
        ):
            workflow_assets = determine_workflow_assets(
                UniverseConfig(sqlite_db_path="state.sqlite", require_workflow_source_success=True)
            )

        self.assertFalse(workflow_assets.attrs["universe_degraded"])
        self.assertEqual(workflow_assets.attrs["universe_counts"]["Workflow universe sources failed"], 0)


if __name__ == "__main__":
    unittest.main()
