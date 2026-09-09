from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import urllib3
import yfinance.shared as yf_shared

from leveraged_trader import _http_deadline_worker as http_deadline_worker
from leveraged_trader import _yfinance_deadline_worker as yfinance_deadline_worker
from leveraged_trader._http_deadline_worker import run_deadline_subprocess
from leveraged_trader.config import TradierMarketDataConfig
from leveraged_trader.market_data import (
    MARKET_DATA_PROVIDERS_ATTR,
    TRADIER_RECOVERED_SYMBOLS_ATTR,
    MarketDataDownloadError,
    _download_yfinance,
    _get_tradier_response_with_deadline,
    _load_tradier_symbol_frame,
    _materialize_tradier_response,
    _strict_response_json,
    exclude_unfinalized_daily_bar,
    load_market_data,
    load_strategy_data,
    recover_signal_history_for_calendar,
)


def _tradier_streaming_response(
    *chunks: bytes,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    encoding: str | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = "https://api.tradier.com/v1/markets/history"
    response.headers = headers or {}
    response.encoding = encoding
    response.raw = Mock()
    response.raw.read1.side_effect = [*chunks, b""]
    response.close = Mock()
    return response


def _tradier_request_from_daemon(url: str, send_connection: object) -> None:
    try:
        deadline = time.monotonic() + 2.0
        response = _get_tradier_response_with_deadline(
            url,
            deadline=deadline,
            request_kwargs={
                "headers": {"Accept-Encoding": "identity"},
                "timeout": urllib3.util.Timeout(total=2.0, connect=2.0, read=2.0),
                "allow_redirects": False,
                "stream": True,
            },
        )
        _materialize_tradier_response(response, deadline=deadline)
        result: tuple[object, ...] = ("response", response.status_code, response.text)
    except BaseException as exc:
        result = ("error", type(exc).__name__, str(exc))
    try:
        send_connection.send(result)
    finally:
        send_connection.close()


class MarketDataTests(unittest.TestCase):
    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_requested_symbol_identities_are_validated_before_provider_requests(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        class MutableString(str):
            pass

        cases = (
            ({"symbols": ["SPY"], "auto_adjust": "false"}, "auto_adjust must be a boolean"),
            ({"symbols": ["SPY"], "auto_adjust": np.bool_(True)}, "auto_adjust must be a boolean"),
            ({"symbols": ["SPY"], "tradier_cfg": True}, "TradierMarketDataConfig"),
            (
                {
                    "symbols": ["SPY"],
                    "tradier_cfg": TradierMarketDataConfig(access_token=MutableString("secret")),
                },
                "ACCESS_TOKEN must be a string",
            ),
            (
                {
                    "symbols": ["SPY"],
                    "tradier_cfg": TradierMarketDataConfig(
                        access_token="secret",
                        base_url=MutableString("https://api.tradier.com/v1"),
                    ),
                },
                "BASE_URL must be a string",
            ),
            ({"symbols": ["SPY", "spy"]}, "Yahoo Finance identity"),
            ({"symbols": ["i", "ı"]}, "Yahoo Finance identity"),
            ({"symbols": [" SPY"]}, "trimmed, nonempty string"),
            ({"symbols": ["SPY", None]}, "trimmed, nonempty string"),
            ({"symbols": ["TQ\nQQ"]}, "control characters"),
            ({"symbols": ["TQ\x1b[2JQQ"]}, "control characters"),
            ({"symbols": ["TQ\x00QQ"]}, "control characters"),
            ({"symbols": ["SPY"], "calendar_symbol": " spy "}, "calendar_symbol"),
            ({"symbols": ["SPY"], "calendar_symbol": "SP\x1b[2JY"}, "calendar_symbol"),
            (
                {
                    "symbols": ["BRK-B", "BRK/B"],
                    "auto_adjust": False,
                    "tradier_cfg": TradierMarketDataConfig(access_token="token"),
                },
                "Tradier identity",
            ),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                load_market_data(**kwargs)

        mock_download.assert_not_called()
        mock_get.assert_not_called()

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_non_latin_1_tradier_token_is_rejected_before_any_provider_request(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        for access_token in ("opaque-€-token", "秘密", "😀"):
            with (
                self.subTest(access_token=repr(access_token)),
                self.assertRaisesRegex(ValueError, "HTTP-header-compatible"),
            ):
                load_market_data(
                    symbols=["TQQQ"],
                    auto_adjust=False,
                    tradier_cfg=TradierMarketDataConfig(access_token=access_token),
                )

        mock_download.assert_not_called()
        mock_get.assert_not_called()

    def test_download_error_neutralizes_untrusted_symbol_and_source_labels(self) -> None:
        error = MarketDataDownloadError(
            {"TQ\n\x1b[2J\x00QQ": "failure"},
            source="Provider\x1b[2J\x00",
        )

        diagnostic = str(error)
        self.assertTrue(all(character.isprintable() for character in diagnostic))
        self.assertNotIn("\n", diagnostic)
        self.assertNotIn("\x1b", diagnostic)
        self.assertNotIn("\x00", diagnostic)
        self.assertIn("TQ [2J QQ", diagnostic)

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf_multi._download_impl")
    def test_runtime_yahoo_isin_alias_cannot_duplicate_one_security_through_tradier(
        self,
        mock_download_impl: Mock,
        mock_get: Mock,
    ) -> None:
        isin = "US78462F1030"

        def resolve_isin(download_context: object, **_kwargs: object) -> pd.DataFrame:
            download_context.isins["SPY"] = isin
            return pd.DataFrame()

        mock_download_impl.side_effect = resolve_isin

        with self.assertRaisesRegex(ValueError, "collide after Yahoo Finance resolved identifier aliases"):
            load_market_data(
                symbols=[isin, "SPY"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        mock_get.assert_not_called()

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_exception_diagnostic_is_bounded_printable_and_redacted(self, mock_download: Mock) -> None:
        secret = "local-secret"
        mock_download.side_effect = RuntimeError(f"provider rejected {secret}\x1b[2J\x00END" + "X" * 1_000_000)

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["TQQQ"])

        diagnostic = raised.exception.symbol_reasons["TQQQ"]
        self.assertLessEqual(len(diagnostic), 512)
        self.assertTrue(all(character.isprintable() for character in diagnostic))
        self.assertNotIn(secret, diagnostic)
        self.assertNotIn("\x1b", diagnostic)
        self.assertNotIn("\x00", diagnostic)
        self.assertIn("redacted credential", diagnostic)

    @patch("leveraged_trader.market_data.yf.download")
    def test_exact_duplicate_symbol_and_calendar_identity_are_harmless(self, mock_download: Mock) -> None:
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [10.0],
                "High": [11.0],
                "Low": [9.0],
                "Close": [10.5],
                "Volume": [1_000.0],
            },
            index=pd.to_datetime(["2026-01-02"]),
        )

        out = load_market_data(symbols=["AAA", "AAA"], calendar_symbol="AAA")

        self.assertEqual(out.columns.tolist(), [f"AAA_{field}" for field in ["Open", "High", "Low", "Close", "Volume"]])
        self.assertEqual(mock_download.call_args.kwargs["tickers"], ["AAA"])

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_repairs_only_float64_ulp_scale_adjusted_ohlc_inversion(
        self,
        mock_download: Mock,
    ) -> None:
        open_price = 4_098_148.626555175
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [open_price],
                "High": [np.nextafter(open_price, -np.inf)],
                "Low": [open_price - 1.0],
                "Close": [open_price],
                "Volume": [1_000.0],
            },
            index=pd.to_datetime(["2026-01-02"]),
        )

        repaired = load_market_data(symbols=["AAA"])

        self.assertEqual(repaired.loc[pd.Timestamp("2026-01-02"), "AAA_High"], open_price)

        mock_download.return_value = mock_download.return_value.copy()
        mock_download.return_value.loc[:, "High"] = open_price - 1e-6
        with self.assertRaisesRegex(MarketDataDownloadError, "High must be at least"):
            load_market_data(symbols=["AAA"])

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_rejects_extreme_ohlc_inversion_instead_of_using_infinite_tolerance(
        self,
        mock_download: Mock,
    ) -> None:
        max_price = np.finfo(np.float64).max
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [max_price],
                "High": [1.0],
                "Low": [1.0],
                "Close": [max_price],
                "Volume": [1_000.0],
            },
            index=pd.to_datetime(["2026-01-02"]),
        )

        with self.assertRaisesRegex(MarketDataDownloadError, "High must be at least"):
            load_market_data(symbols=["AAA"])

    @patch("leveraged_trader.market_data.yf_multi._download_impl")
    def test_yfinance_end_only_download_retains_complete_history_period(
        self,
        mock_download_impl: Mock,
    ) -> None:
        mock_download_impl.return_value = pd.DataFrame()

        _download_yfinance(["SPY"], None, "2026-01-01", True)

        download_kwargs = mock_download_impl.call_args.kwargs
        self.assertEqual(download_kwargs["period"], "max")
        self.assertIsNone(download_kwargs["start"])
        self.assertEqual(download_kwargs["end"], "2026-01-01")

    @patch("leveraged_trader.market_data.yf_multi._download_impl")
    def test_yfinance_explicit_start_uses_date_range_instead_of_period(
        self,
        mock_download_impl: Mock,
    ) -> None:
        mock_download_impl.return_value = pd.DataFrame()

        _download_yfinance(["SPY"], "2025-01-01", "2026-01-01", True)

        download_kwargs = mock_download_impl.call_args.kwargs
        self.assertIsNone(download_kwargs["period"])
        self.assertEqual(download_kwargs["start"], "2025-01-01")
        self.assertEqual(download_kwargs["end"], "2026-01-01")

    @patch("leveraged_trader.market_data.run_yfinance_download_with_deadline")
    def test_yfinance_production_path_uses_absolute_deadline_worker(self, mock_worker: Mock) -> None:
        raw = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-01-02"]),
        )
        mock_worker.return_value = ("yfinance_response", raw, {"MISSING": "not found"}, {})
        started = time.monotonic()

        returned_raw, errors = _download_yfinance(["SPY", "MISSING"], None, "2026-01-03", True)

        self.assertIs(returned_raw, raw)
        self.assertEqual(errors, {"MISSING": "not found"})
        mock_worker.assert_called_once()
        self.assertEqual(
            mock_worker.call_args.args,
            (["SPY", "MISSING"], None, "2026-01-03", True),
        )
        deadline = mock_worker.call_args.kwargs["deadline"]
        request_timeout = mock_worker.call_args.kwargs["request_timeout_seconds"]
        self.assertGreater(deadline, started)
        self.assertLessEqual(deadline - started, 30.1)
        self.assertGreater(request_timeout, 0)
        self.assertLessEqual(request_timeout, deadline - started)

    def test_yfinance_internal_completion_stall_hits_deadline_and_reaps_worker(self) -> None:
        processes: list[subprocess.Popen[bytes]] = []
        real_popen = subprocess.Popen

        def capture_process(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        stalled_yfinance_command = (
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; import yfinance.multi as multi; "
                "sys.stdin.buffer.read(); "
                "multi._download_one_threaded=lambda *args,**kwargs: None; "
                "multi._download_impl(multi._DownloadCtx(),tickers=['SPY'],threads=True,progress=False)"
            ),
        )
        started = time.monotonic()
        with (
            patch("leveraged_trader.market_data._YFINANCE_REQUEST_TIMEOUT_SECONDS", 0.2),
            patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_COMMAND", stalled_yfinance_command),
            patch(
                "leveraged_trader._http_deadline_worker.subprocess.Popen",
                side_effect=capture_process,
            ),
        ):
            raw, errors = _download_yfinance(["SPY"], None, None, True)

        self.assertLess(time.monotonic() - started, 0.75)
        self.assertIsNone(raw)
        self.assertIn("overall deadline", errors["SPY"])
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(
            processes[0].returncode,
            f"Yahoo Finance deadline worker {processes[0].pid} was not reaped",
        )

    def test_yfinance_deadline_worker_round_trips_dataframe_payload(self) -> None:
        processes: list[subprocess.Popen[bytes]] = []
        real_popen = subprocess.Popen

        def capture_process(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        static_yfinance_command = (
            sys.executable,
            "-I",
            "-c",
            (
                "import pandas as pd; "
                "from leveraged_trader._http_deadline_worker import read_worker_request,write_worker_result; "
                "read_worker_request(); "
                "frame=pd.DataFrame({'Close':[101.0]},index=pd.to_datetime(['2026-01-02'])); "
                "write_worker_result(('yfinance_response',frame,{},{}),max_bytes=134217728)"
            ),
        )
        with (
            patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_COMMAND", static_yfinance_command),
            patch("leveraged_trader._http_deadline_worker.subprocess.Popen", side_effect=capture_process),
        ):
            raw, errors = _download_yfinance(["SPY"], None, None, True)

        self.assertEqual(errors, {})
        assert raw is not None
        self.assertEqual(raw.loc[pd.Timestamp("2026-01-02"), "Close"], 101.0)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        assert processes[0].stdin is not None
        assert processes[0].stdout is not None
        self.assertTrue(processes[0].stdin.closed)
        self.assertTrue(processes[0].stdout.closed)

    @patch.object(yfinance_deadline_worker.yf_multi, "_download_impl")
    def test_yfinance_worker_payload_retains_frame_errors_and_isin_aliases(
        self,
        mock_download_impl: Mock,
    ) -> None:
        raw = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-01-02"]),
        )

        def download(context: object, **kwargs: object) -> pd.DataFrame:
            self.assertEqual(kwargs["period"], "max")
            self.assertEqual(kwargs["timeout"], 17.0)
            context.errors["AAPL"] = "$AAPL: no timezone found"
            context.isins["AAPL"] = "US0378331005"
            return raw

        mock_download_impl.side_effect = download

        payload = yfinance_deadline_worker._yfinance_download_payload(
            {
                "auto_adjust": True,
                "end": "2026-01-03",
                "mode": "yfinance",
                "request_timeout_seconds": 17.0,
                "start": None,
                "symbols": ["US0378331005"],
            }
        )

        self.assertEqual(payload[0], "yfinance_response")
        self.assertIs(payload[1], raw)
        self.assertEqual(payload[2], {"AAPL": "$AAPL: no timezone found"})
        self.assertEqual(payload[3], {"AAPL": "US0378331005"})

    def test_yfinance_workers_reject_frames_over_memory_limit_before_writing_result(self) -> None:
        frame = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2026-01-02"]),
        )
        cases = (
            (
                {
                    "auto_adjust": True,
                    "end": None,
                    "mode": "yfinance",
                    "request_timeout_seconds": 17.0,
                    "start": None,
                    "symbols": ["SPY"],
                },
                "execute_yfinance_download",
                (frame, {}, {}),
                "market-data frame",
            ),
            (
                {
                    "mode": "yfinance_ticker_history",
                    "request_timeout_seconds": 17.0,
                    "symbol": "SPY",
                },
                "execute_yfinance_ticker_history",
                frame,
                "recent-history frame",
            ),
        )
        for request, execute_name, execute_result, frame_label in cases:
            with (
                self.subTest(mode=request["mode"]),
                patch.object(yfinance_deadline_worker, "read_worker_request", return_value=request),
                patch.object(yfinance_deadline_worker, execute_name, return_value=execute_result),
                patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_FRAME_MAX_MEMORY_BYTES", 1),
                patch.object(yfinance_deadline_worker, "write_worker_result") as write_result,
            ):
                yfinance_deadline_worker.main()

            payload = write_result.call_args.args[0]
            self.assertEqual(payload[:2], ("error", "ValueError"))
            self.assertIn(frame_label, payload[2])
            self.assertIn("memory limit", payload[2])

    def test_yfinance_worker_applies_address_space_limit_before_reading_request(self) -> None:
        fake_resource = Mock(RLIMIT_AS=9, RLIM_INFINITY=-1)
        fake_resource.getrlimit.return_value = (-1, -1)
        with (
            patch.object(yfinance_deadline_worker.sys, "platform", "linux"),
            patch.object(yfinance_deadline_worker, "resource", fake_resource),
            patch.object(yfinance_deadline_worker, "_linux_virtual_memory_bytes", return_value=100_000),
        ):
            yfinance_deadline_worker._apply_yfinance_worker_memory_limit()

        expected_limit = 100_000 + yfinance_deadline_worker._YFINANCE_WORKER_ADDRESS_SPACE_HEADROOM_BYTES
        fake_resource.setrlimit.assert_called_once_with(9, (expected_limit, -1))

        with (
            patch.object(
                yfinance_deadline_worker,
                "_apply_yfinance_worker_memory_limit",
                side_effect=RuntimeError("limit unavailable"),
            ),
            patch.object(yfinance_deadline_worker, "read_worker_request") as read_request,
            patch.object(yfinance_deadline_worker, "write_worker_result") as write_result,
        ):
            yfinance_deadline_worker.main(enforce_memory_limit=True)

        read_request.assert_not_called()
        self.assertEqual(write_result.call_args.args[0], ("error", "RuntimeError", "limit unavailable"))

    def test_yfinance_worker_reads_darwin_virtual_size_and_applies_address_space_limit(self) -> None:
        proc_pidinfo = Mock()

        def fill_task_info(
            _pid: int,
            flavor: int,
            _arg: int,
            task_info_pointer: object,
            task_info_size: int,
        ) -> int:
            self.assertEqual(flavor, yfinance_deadline_worker._DARWIN_PROC_PIDTASKINFO)
            task_info = ctypes.cast(
                task_info_pointer,
                ctypes.POINTER(yfinance_deadline_worker._DarwinProcTaskInfo),
            ).contents
            task_info.pti_virtual_size = 250_000
            return task_info_size

        proc_pidinfo.side_effect = fill_task_info
        libproc = Mock(proc_pidinfo=proc_pidinfo)
        fake_resource = Mock(RLIMIT_AS=9, RLIM_INFINITY=-1)
        fake_resource.getrlimit.return_value = (-1, -1)
        with (
            patch.object(yfinance_deadline_worker.sys, "platform", "darwin"),
            patch.object(yfinance_deadline_worker.ctypes, "CDLL", return_value=libproc),
            patch.object(yfinance_deadline_worker, "resource", fake_resource),
        ):
            yfinance_deadline_worker._apply_yfinance_worker_memory_limit()

        expected_limit = 250_000 + yfinance_deadline_worker._YFINANCE_WORKER_ADDRESS_SPACE_HEADROOM_BYTES
        fake_resource.setrlimit.assert_called_once_with(9, (expected_limit, -1))

        proc_pidinfo.return_value = 0
        proc_pidinfo.side_effect = None
        with (
            patch.object(yfinance_deadline_worker.ctypes, "CDLL", return_value=libproc),
            self.assertRaisesRegex(RuntimeError, "determine its current memory footprint"),
        ):
            yfinance_deadline_worker._darwin_virtual_memory_bytes()

    def test_yfinance_worker_installs_and_retains_windows_process_memory_job(self) -> None:
        kernel32 = Mock()
        kernel32.GetCurrentProcess.return_value = 123
        kernel32.CreateJobObjectW.return_value = 456
        kernel32.SetInformationJobObject.return_value = 1
        kernel32.AssignProcessToJobObject.return_value = 1
        psapi = Mock()

        def fill_memory_counters(_process: object, counters_pointer: object, _size: int) -> int:
            counters = ctypes.cast(
                counters_pointer,
                ctypes.POINTER(yfinance_deadline_worker._WindowsProcessMemoryCountersEx),
            ).contents
            counters.PrivateUsage = 300_000
            return 1

        psapi.GetProcessMemoryInfo.side_effect = fill_memory_counters

        def load_windows_dll(name: str, *, use_last_error: bool) -> Mock:
            self.assertTrue(use_last_error)
            return {"kernel32": kernel32, "psapi": psapi}[name]

        with (
            patch.object(yfinance_deadline_worker.sys, "platform", "win32"),
            patch.object(yfinance_deadline_worker.ctypes, "WinDLL", side_effect=load_windows_dll, create=True),
            patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_WINDOWS_JOB_HANDLE", None),
        ):
            yfinance_deadline_worker._apply_yfinance_worker_memory_limit()
            self.assertEqual(yfinance_deadline_worker._YFINANCE_WORKER_WINDOWS_JOB_HANDLE, 456)
            yfinance_deadline_worker._apply_yfinance_worker_memory_limit()

        set_job_args = kernel32.SetInformationJobObject.call_args.args
        self.assertEqual(set_job_args[0], 456)
        self.assertEqual(
            set_job_args[1],
            yfinance_deadline_worker._WINDOWS_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        )
        limits = ctypes.cast(
            set_job_args[2],
            ctypes.POINTER(yfinance_deadline_worker._WindowsJobObjectExtendedLimitInformation),
        ).contents
        self.assertEqual(
            limits.BasicLimitInformation.LimitFlags,
            yfinance_deadline_worker._WINDOWS_JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        )
        self.assertEqual(
            limits.ProcessMemoryLimit,
            300_000 + yfinance_deadline_worker._YFINANCE_WORKER_ADDRESS_SPACE_HEADROOM_BYTES,
        )
        kernel32.AssignProcessToJobObject.assert_called_once_with(456, 123)
        kernel32.CreateJobObjectW.assert_called_once_with(None, None)
        kernel32.CloseHandle.assert_not_called()

    def test_yfinance_worker_closes_failed_windows_job_and_fails_closed_elsewhere(self) -> None:
        kernel32 = Mock()
        kernel32.GetCurrentProcess.return_value = 123
        kernel32.CreateJobObjectW.return_value = 456
        kernel32.SetInformationJobObject.return_value = 1
        kernel32.AssignProcessToJobObject.return_value = 0
        psapi = Mock()

        def fill_memory_counters(_process: object, counters_pointer: object, _size: int) -> int:
            counters = ctypes.cast(
                counters_pointer,
                ctypes.POINTER(yfinance_deadline_worker._WindowsProcessMemoryCountersEx),
            ).contents
            counters.PrivateUsage = 300_000
            return 1

        psapi.GetProcessMemoryInfo.side_effect = fill_memory_counters

        def load_windows_dll(name: str, *, use_last_error: bool) -> Mock:
            self.assertTrue(use_last_error)
            return {"kernel32": kernel32, "psapi": psapi}[name]

        with (
            patch.object(yfinance_deadline_worker.sys, "platform", "win32"),
            patch.object(yfinance_deadline_worker.ctypes, "WinDLL", side_effect=load_windows_dll, create=True),
            patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_WINDOWS_JOB_HANDLE", None),
            self.assertRaisesRegex(RuntimeError, "enter its memory-limit Job Object"),
        ):
            yfinance_deadline_worker._apply_yfinance_worker_memory_limit()

        kernel32.CloseHandle.assert_called_once_with(456)

        with (
            patch.object(yfinance_deadline_worker.sys, "platform", "freebsd14"),
            self.assertRaisesRegex(RuntimeError, "cannot enforce its memory limit"),
        ):
            yfinance_deadline_worker._apply_yfinance_worker_memory_limit()

    def test_yfinance_worker_rejects_unsafe_shape_and_nested_object_content(self) -> None:
        frame = pd.DataFrame({"Close": [100.0]})
        with (
            patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_FRAME_MAX_CELLS", 0),
            self.assertRaisesRegex(ValueError, "safe transport shape limit"),
        ):
            yfinance_deadline_worker._validate_yfinance_frame_for_transport(
                frame,
                label="market-data frame",
            )

        nested = pd.DataFrame({"Close": [["provider", "object", "graph"]]})
        with self.assertRaisesRegex(ValueError, "non-scalar object value"):
            yfinance_deadline_worker._validate_yfinance_frame_for_transport(
                nested,
                label="market-data frame",
            )

        hidden_tuple_bytes = pd.DataFrame({"Close": [(b"x" * 2_048,)]})
        with (
            patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_FRAME_MAX_MEMORY_BYTES", 1_024),
            self.assertRaisesRegex(ValueError, "non-scalar object value"),
        ):
            yfinance_deadline_worker._validate_yfinance_frame_for_transport(
                hidden_tuple_bytes,
                label="market-data frame",
            )

        hidden_categorical_bytes = pd.DataFrame({"Close": pd.Categorical([(b"x" * 2_048,)])})
        with (
            patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_FRAME_MAX_MEMORY_BYTES", 1_024),
            self.assertRaisesRegex(ValueError, "non-scalar categorical value"),
        ):
            yfinance_deadline_worker._validate_yfinance_frame_for_transport(
                hidden_categorical_bytes,
                label="market-data frame",
            )

    def test_yfinance_worker_bounds_axes_and_rejects_dataframe_subclasses(self) -> None:
        tuple_label = pd.Index([(b"x" * 2_048, "Close")], tupleize_cols=False)
        with self.assertRaisesRegex(ValueError, "non-scalar column label"):
            yfinance_deadline_worker._validate_yfinance_frame_for_transport(
                pd.DataFrame([[100.0]], columns=tuple_label),
                label="market-data frame",
            )

        named_frame = pd.DataFrame({"Close": [100.0]})
        named_frame.index.name = "x" * 2_048
        with (
            patch.object(yfinance_deadline_worker, "_YFINANCE_WORKER_FRAME_MAX_MEMORY_BYTES", 1_024),
            self.assertRaisesRegex(ValueError, "memory limit"),
        ):
            yfinance_deadline_worker._validate_yfinance_frame_for_transport(
                named_frame,
                label="market-data frame",
            )

        class FrameSubclass(pd.DataFrame):
            pass

        with self.assertRaisesRegex(TypeError, "exact DataFrame"):
            yfinance_deadline_worker._validate_yfinance_frame_for_transport(
                FrameSubclass({"Close": [100.0]}),
                label="market-data frame",
            )

    def test_yfinance_worker_preflight_accepts_expected_pandas_layouts(self) -> None:
        expected_frames = (
            pd.DataFrame(
                [[100.0, 101.0]],
                index=pd.DatetimeIndex(["2026-01-02"], tz="America/New_York"),
                columns=pd.MultiIndex.from_tuples(
                    [("SPY", "Open"), ("SPY", "Close")],
                    names=["Ticker", "Price"],
                ),
            ),
            pd.DataFrame(
                {
                    "Close": pd.array([100.0, pd.NA], dtype="Float64"),
                    "Volume": pd.array([1_000, pd.NA], dtype="Int64"),
                }
            ),
            pd.DataFrame({"Close": pd.Categorical([100.0, 101.0])}),
        )
        for frame in expected_frames:
            with self.subTest(dtypes=tuple(map(str, frame.dtypes))):
                yfinance_deadline_worker._validate_yfinance_frame_for_transport(
                    frame,
                    label="market-data frame",
                )

    def test_yfinance_serialization_lock_wait_is_inside_overall_deadline(self) -> None:
        from leveraged_trader import market_data as market_data_module

        self.assertTrue(market_data_module._YFINANCE_DOWNLOAD_LOCK.acquire(blocking=False))
        started = time.monotonic()
        try:
            with (
                patch("leveraged_trader.market_data._YFINANCE_REQUEST_TIMEOUT_SECONDS", 0.05),
                patch("leveraged_trader.market_data.run_yfinance_download_with_deadline") as mock_worker,
            ):
                raw, errors = _download_yfinance(["SPY"], None, None, True)
        finally:
            market_data_module._YFINANCE_DOWNLOAD_LOCK.release()

        self.assertLess(time.monotonic() - started, 0.25)
        self.assertIsNone(raw)
        self.assertIn("overall deadline", errors["SPY"])
        mock_worker.assert_not_called()

    def test_yfinance_worker_response_schema_fails_closed(self) -> None:
        invalid_payloads = (
            (),
            ("error",),
            ("error", "ValueError", 1),
            ("error", "ValueError", "x" * 4097),
            ("unexpected", pd.DataFrame(), {}, {}),
            ("yfinance_response", "not-a-frame", {}, {}),
            ("yfinance_response", pd.DataFrame(), {1: "error"}, {}),
            ("yfinance_response", pd.DataFrame(), {}, {"SPY": 1}),
        )
        for payload in invalid_payloads:
            with (
                self.subTest(payload=payload),
                patch(
                    "leveraged_trader.market_data.run_yfinance_download_with_deadline",
                    return_value=payload,
                ),
            ):
                raw, errors = _download_yfinance(["SPY"], None, None, True)

            self.assertIsNone(raw)
            self.assertIn("invalid", errors["SPY"].lower())

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_daily_history_bounds_reject_noncanonical_values_before_provider_requests(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        invalid_values: list[object] = [
            "2026-01-05T00:00:00-05:00",
            "2026-01-05 00:00:00",
            "2026-1-5",
            " 2026-01-05 ",
            "2026-02-30",
            "",
            "NaT",
            pd.NaT,
        ]
        for label in ("start", "end"):
            for value in invalid_values:
                with (
                    self.subTest(label=label, value=value),
                    self.assertRaisesRegex(
                        ValueError,
                        rf"Market-data {label} must be a date in canonical YYYY-MM-DD format",
                    ),
                ):
                    load_market_data(
                        symbols=["AAA"],
                        auto_adjust=False,
                        tradier_cfg=TradierMarketDataConfig(access_token="token"),
                        **{label: value},
                    )

        mock_download.assert_not_called()
        mock_get.assert_not_called()

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_rows_are_clipped_to_the_exclusive_end_boundary(self, mock_download: Mock) -> None:
        mock_download.return_value = pd.DataFrame(
            [[10.0, 11.0, 9.0, 10.5, 1000.0]],
            index=pd.to_datetime(["2026-01-05"]),
            columns=["Open", "High", "Low", "Close", "Volume"],
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["AAA"], end="2026-01-05")

        self.assertIn("inside the requested date range", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_fallback_enforces_complete_start_and_exclusive_end(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": [
                            {
                                "date": "2026-01-02",
                                "open": 10,
                                "high": 11,
                                "low": 9,
                                "close": 10.5,
                                "volume": 1000,
                            },
                            {
                                "date": "2026-01-05",
                                "open": 11,
                                "high": 12,
                                "low": 10,
                                "close": 11.5,
                                "volume": 1100,
                            },
                        ]
                    }
                }
            ),
        )

        data = load_market_data(
            symbols=["AAA"],
            end="2026-01-05",
            auto_adjust=False,
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(data.index.tolist(), [pd.Timestamp("2026-01-02")])
        self.assertEqual(mock_get.call_args.kwargs["params"]["start"], "1900-01-01")
        self.assertEqual(mock_get.call_args.kwargs["params"]["end"], "2026-01-05")

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_exclusive_end_only_response_is_rejected(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-05",
                            "open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1000,
                        }
                    }
                }
            ),
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                end="2026-01-05",
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        self.assertIn("inside the requested date range", str(raised.exception))

    @patch("leveraged_trader.market_data.yf_multi._download_impl")
    def test_yfinance_download_reads_isolated_per_call_errors(self, mock_download_impl: Mock) -> None:
        calls = 0

        def fake_download_impl(context: object, **_: object) -> pd.DataFrame:
            nonlocal calls
            calls += 1
            if calls == 1:
                context.errors["MISSING"] = "$MISSING: possibly delisted; no timezone found"
                yf_shared._ERRORS = {"STALE": "stale shared error"}
            return pd.DataFrame()

        mock_download_impl.side_effect = fake_download_impl

        _raw, first_errors = _download_yfinance(["MISSING"], None, None, True)
        _raw, second_errors = _download_yfinance(["VALID"], None, None, True)

        self.assertEqual(
            first_errors,
            {"MISSING": "$MISSING: possibly delisted; no timezone found"},
        )
        self.assertEqual(second_errors, {})

    @patch("leveraged_trader.market_data.yf_multi._download_impl")
    def test_yfinance_context_errors_preserve_lowercase_requested_symbol(
        self,
        mock_download_impl: Mock,
    ) -> None:
        def fake_download_impl(context: object, **_: object) -> pd.DataFrame:
            context.errors["MISSING"] = "$MISSING: possibly delisted; no timezone found"
            return pd.DataFrame()

        mock_download_impl.side_effect = fake_download_impl

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["missing"])

        self.assertEqual(raised.exception.symbols, ["missing"])
        self.assertIn("missing: no timezone found", str(raised.exception))
        self.assertNotIn("No data returned", str(raised.exception))

    @patch("leveraged_trader.market_data.yf.download")
    def test_lowercase_requested_symbol_matches_uppercase_yahoo_multiindex(
        self,
        mock_download: Mock,
    ) -> None:
        fields = ["Open", "High", "Low", "Close", "Volume"]
        cases = {
            "ticker first": pd.MultiIndex.from_product([["SPY"], fields]),
            "ticker second": pd.MultiIndex.from_product([fields, ["SPY"]]),
        }

        for label, columns in cases.items():
            with self.subTest(label=label):
                mock_download.return_value = pd.DataFrame(
                    [[10.0, 11.0, 9.0, 10.5, 1000.0]],
                    index=pd.to_datetime(["2026-01-02"]),
                    columns=columns,
                )

                data = load_market_data(symbols=["spy"])

                self.assertEqual(
                    data.columns.tolist(),
                    ["spy_Open", "spy_High", "spy_Low", "spy_Close", "spy_Volume"],
                )
                self.assertEqual(data.loc[pd.Timestamp("2026-01-02"), "spy_Close"], 10.5)
                self.assertEqual(data.attrs[MARKET_DATA_PROVIDERS_ATTR], {"spy": "yahoo_finance"})

    @patch("leveraged_trader.market_data.yf.download")
    def test_ohlcv_named_symbol_prefers_exact_ticker_level_across_multiindex_layouts(
        self,
        mock_download: Mock,
    ) -> None:
        fields = ["Open", "High", "Low", "Close", "Volume"]
        values = [[10.0, 11.0, 9.0, 10.5, 1000.0]]

        for symbol in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"):
            layouts = {
                "ticker first": pd.MultiIndex.from_product([[symbol], fields]),
                "ticker second": pd.MultiIndex.from_product([fields, [symbol]]),
            }
            for layout, columns in layouts.items():
                with self.subTest(symbol=symbol, layout=layout):
                    mock_download.return_value = pd.DataFrame(
                        values,
                        index=pd.to_datetime(["2026-01-02"]),
                        columns=columns,
                    )

                    data = load_market_data(symbols=[symbol])

                    self.assertEqual(
                        data.columns.tolist(),
                        [f"{symbol}_{field}" for field in fields],
                    )
                    self.assertEqual(data.loc[pd.Timestamp("2026-01-02"), f"{symbol}_Close"], 10.5)

    @patch("leveraged_trader.market_data.yf.download")
    def test_title_case_ohlcv_named_symbol_prefers_casefold_ticker_frame(
        self,
        mock_download: Mock,
    ) -> None:
        fields = ["Open", "High", "Low", "Close", "Volume"]
        values = [[10.0, 11.0, 9.0, 10.5, 1000.0]]

        for requested_symbol in fields:
            returned_symbol = requested_symbol.upper()
            layouts = {
                "ticker first": pd.MultiIndex.from_product([[returned_symbol], fields]),
                "ticker second": pd.MultiIndex.from_product([fields, [returned_symbol]]),
            }
            for layout, columns in layouts.items():
                with self.subTest(symbol=requested_symbol, layout=layout):
                    mock_download.return_value = pd.DataFrame(
                        values,
                        index=pd.to_datetime(["2026-01-02"]),
                        columns=columns,
                    )

                    data = load_market_data(symbols=[requested_symbol])

                    self.assertEqual(
                        data.columns.tolist(),
                        [f"{requested_symbol}_{field}" for field in fields],
                    )
                    self.assertEqual(
                        data.loc[pd.Timestamp("2026-01-02"), f"{requested_symbol}_Close"],
                        10.5,
                    )

    @patch("leveraged_trader.market_data.yf.download")
    def test_casefold_symbol_match_is_rejected_when_both_multiindex_levels_match(
        self,
        mock_download: Mock,
    ) -> None:
        fields = ["Open", "High", "Low", "Close", "Volume"]
        mock_download.return_value = pd.DataFrame(
            [[10.0, 11.0, 9.0, 10.5, 1000.0]],
            index=pd.to_datetime(["2026-01-02"]),
            columns=pd.MultiIndex.from_product([fields, ["OPEN"]]),
        )

        with self.assertRaisesRegex(MarketDataDownloadError, "matched multiple.*case-insensitively"):
            load_market_data(symbols=["open"])

    @patch("leveraged_trader.market_data.yf_multi._download_impl")
    def test_yfinance_context_errors_follow_exposed_isin_alias(
        self,
        mock_download_impl: Mock,
    ) -> None:
        def fake_download_impl(context: object, **_: object) -> pd.DataFrame:
            context.isins["AAPL"] = "US0378331005"
            context.errors["AAPL"] = "$AAPL: possibly delisted; no timezone found"
            return pd.DataFrame()

        mock_download_impl.side_effect = fake_download_impl

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["US0378331005"])

        self.assertEqual(raised.exception.symbols, ["US0378331005"])
        self.assertIn("US0378331005: no timezone found", str(raised.exception))

    def test_mixed_provider_timezones_are_normalized_before_merge(self) -> None:
        yahoo_frame = pd.DataFrame(
            {"AAA_Close": [10.5, 11.5]},
            index=pd.DatetimeIndex(["2026-01-02", "2026-01-05"], tz="America/New_York"),
        )
        tradier_frame = pd.DataFrame(
            {"BBB_Close": [20.5, 21.5]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )

        for calendar_symbol in (None, "AAA"):
            with (
                self.subTest(calendar_symbol=calendar_symbol),
                patch(
                    "leveraged_trader.market_data._download_yfinance",
                    return_value=(pd.DataFrame(), {}),
                ),
                patch(
                    "leveraged_trader.market_data._yfinance_symbol_frames",
                    return_value=({"AAA": yahoo_frame}, {"BBB": "missing"}),
                ),
                patch(
                    "leveraged_trader.market_data._load_tradier_fallback_frames",
                    return_value=({"BBB": tradier_frame}, {}),
                ),
            ):
                data = load_market_data(
                    symbols=["AAA", "BBB"],
                    calendar_symbol=calendar_symbol,
                    tradier_cfg=TradierMarketDataConfig(access_token="token"),
                )

            self.assertEqual(data.index.tolist(), pd.to_datetime(["2026-01-02", "2026-01-05"]).tolist())
            self.assertIsNone(data.index.tz)
            self.assertEqual(data[["AAA_Close", "BBB_Close"]].values.tolist(), [[10.5, 20.5], [11.5, 21.5]])
            self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["BBB"])

    def test_calendar_anchored_merge_rejects_wholly_unpopulated_symbol(self) -> None:
        asset_frame = pd.DataFrame(
            {
                "AAA_Open": [10.0, 11.0],
                "AAA_High": [11.0, 12.0],
                "AAA_Low": [9.0, 10.0],
                "AAA_Close": [10.5, 11.5],
                "AAA_Volume": [1_000.0, 1_100.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )
        disjoint_signal_frame = pd.DataFrame(
            {
                "BBB_Open": [20.0],
                "BBB_High": [21.0],
                "BBB_Low": [19.0],
                "BBB_Close": [20.5],
                "BBB_Volume": [2_000.0],
            },
            index=pd.to_datetime(["2026-01-06"]),
        )

        with (
            patch(
                "leveraged_trader.market_data._download_yfinance",
                return_value=(pd.DataFrame(), {}),
            ),
            patch(
                "leveraged_trader.market_data._yfinance_symbol_frames",
                return_value=({"AAA": asset_frame, "BBB": disjoint_signal_frame}, {}),
            ),
            self.assertRaises(MarketDataDownloadError) as raised,
        ):
            load_market_data(
                symbols=["AAA", "BBB"],
                calendar_symbol="AAA",
            )

        self.assertEqual(raised.exception.symbols, ["BBB"])
        self.assertIn("No daily rows overlapped the retained AAA calendar", str(raised.exception))

    def test_calendar_anchored_merge_recovers_wholly_unpopulated_symbol_with_tradier(self) -> None:
        asset_frame = pd.DataFrame(
            {
                "AAA_Open": [10.0, 11.0],
                "AAA_High": [11.0, 12.0],
                "AAA_Low": [9.0, 10.0],
                "AAA_Close": [10.5, 11.5],
                "AAA_Volume": [1_000.0, 1_100.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )
        disjoint_signal_frame = pd.DataFrame(
            {
                "BBB_Open": [20.0],
                "BBB_High": [21.0],
                "BBB_Low": [19.0],
                "BBB_Close": [20.5],
                "BBB_Volume": [2_000.0],
            },
            index=pd.to_datetime(["2026-01-06"]),
        )
        recovered_signal_frame = pd.DataFrame(
            {
                "BBB_Open": [30.0],
                "BBB_High": [31.0],
                "BBB_Low": [29.0],
                "BBB_Close": [30.5],
                "BBB_Volume": [3_000.0],
            },
            index=pd.to_datetime(["2026-01-05"]),
        )
        cfg = TradierMarketDataConfig(access_token="token")

        with (
            patch(
                "leveraged_trader.market_data._download_yfinance",
                return_value=(pd.DataFrame(), {}),
            ),
            patch(
                "leveraged_trader.market_data._yfinance_symbol_frames",
                return_value=({"AAA": asset_frame, "BBB": disjoint_signal_frame}, {}),
            ),
            patch(
                "leveraged_trader.market_data._load_tradier_fallback_frames",
                return_value=({"BBB": recovered_signal_frame}, {}),
            ) as fallback,
        ):
            data = load_market_data(
                symbols=["AAA", "BBB"],
                auto_adjust=False,
                calendar_symbol="AAA",
                tradier_cfg=cfg,
            )

        self.assertEqual(fallback.call_args.args[0], ["BBB"])
        self.assertTrue(pd.isna(data.loc[pd.Timestamp("2026-01-02"), "BBB_Close"]))
        self.assertEqual(data.loc[pd.Timestamp("2026-01-05"), "BBB_Close"], 30.5)
        self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["BBB"])
        self.assertEqual(
            data.attrs[MARKET_DATA_PROVIDERS_ATTR],
            {"AAA": "yahoo_finance", "BBB": "tradier"},
        )

    def test_calendar_anchored_merge_rejects_disjoint_tradier_replacement(self) -> None:
        asset_frame = pd.DataFrame(
            {
                "AAA_Open": [10.0, 11.0],
                "AAA_High": [11.0, 12.0],
                "AAA_Low": [9.0, 10.0],
                "AAA_Close": [10.5, 11.5],
                "AAA_Volume": [1_000.0, 1_100.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )
        yahoo_signal_frame = pd.DataFrame(
            {
                "BBB_Open": [20.0],
                "BBB_High": [21.0],
                "BBB_Low": [19.0],
                "BBB_Close": [20.5],
                "BBB_Volume": [2_000.0],
            },
            index=pd.to_datetime(["2026-01-06"]),
        )
        disjoint_tradier_frame = pd.DataFrame(
            {
                "BBB_Open": [30.0],
                "BBB_High": [31.0],
                "BBB_Low": [29.0],
                "BBB_Close": [30.5],
                "BBB_Volume": [3_000.0],
            },
            index=pd.to_datetime(["2026-01-07"]),
        )

        with (
            patch(
                "leveraged_trader.market_data._download_yfinance",
                return_value=(pd.DataFrame(), {}),
            ),
            patch(
                "leveraged_trader.market_data._yfinance_symbol_frames",
                return_value=({"AAA": asset_frame, "BBB": yahoo_signal_frame}, {}),
            ),
            patch(
                "leveraged_trader.market_data._load_tradier_fallback_frames",
                return_value=({"BBB": disjoint_tradier_frame}, {}),
            ) as fallback,
            self.assertRaises(MarketDataDownloadError) as raised,
        ):
            load_market_data(
                symbols=["AAA", "BBB"],
                auto_adjust=False,
                calendar_symbol="AAA",
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.args[0], ["BBB"])
        self.assertEqual(raised.exception.symbols, ["BBB"])
        self.assertIn("No daily rows overlapped the retained AAA calendar", str(raised.exception))

    def test_cached_signal_history_recovers_against_authoritative_calendar(self) -> None:
        asset_frame = pd.DataFrame(
            {
                "AAA_Open": [10.0],
                "AAA_High": [11.0],
                "AAA_Low": [9.0],
                "AAA_Close": [10.5],
                "AAA_Volume": [1_000.0],
            },
            index=pd.to_datetime(["2026-01-05"]),
        )
        yahoo_signal_frame = pd.DataFrame(
            {
                "BBB_Open": [20.0],
                "BBB_High": [21.0],
                "BBB_Low": [19.0],
                "BBB_Close": [20.5],
                "BBB_Volume": [2_000.0],
            },
            index=pd.to_datetime(["2026-01-06"]),
        )
        tradier_signal_frame = pd.DataFrame(
            {
                "BBB_Open": [30.0],
                "BBB_High": [31.0],
                "BBB_Low": [29.0],
                "BBB_Close": [30.5],
                "BBB_Volume": [3_000.0],
            },
            index=pd.to_datetime(["2026-01-05"]),
        )
        tradier_signal_frame.attrs["provider_request_id"] = "request-1"
        cfg = TradierMarketDataConfig(access_token="token")

        with patch(
            "leveraged_trader.market_data._load_tradier_fallback_frames",
            return_value=({"BBB": tradier_signal_frame}, {}),
        ) as fallback:
            recovered = recover_signal_history_for_calendar(
                calendar_symbol="AAA",
                calendar_history=asset_frame,
                signal_symbol="BBB",
                signal_history=yahoo_signal_frame,
                auto_adjust=False,
                tradier_cfg=cfg,
            )

        fallback.assert_called_once_with(
            ["BBB"],
            None,
            None,
            cfg,
            auto_adjust=False,
        )
        self.assertEqual(recovered.loc[pd.Timestamp("2026-01-05"), "BBB_Close"], 30.5)
        self.assertEqual(recovered.attrs["provider_request_id"], "request-1")
        self.assertEqual(recovered.attrs[MARKET_DATA_PROVIDERS_ATTR], {"BBB": "tradier"})
        self.assertEqual(recovered.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["BBB"])

    def test_cached_signal_history_rejects_disjoint_tradier_replacement(self) -> None:
        def history(symbol: str, session: str) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    f"{symbol}_Open": [10.0],
                    f"{symbol}_High": [11.0],
                    f"{symbol}_Low": [9.0],
                    f"{symbol}_Close": [10.5],
                    f"{symbol}_Volume": [1_000.0],
                },
                index=pd.to_datetime([session]),
            )

        cfg = TradierMarketDataConfig(access_token="token")
        with (
            patch(
                "leveraged_trader.market_data._load_tradier_fallback_frames",
                return_value=({"BBB": history("BBB", "2026-01-07")}, {}),
            ) as fallback,
            self.assertRaises(MarketDataDownloadError) as raised,
        ):
            recover_signal_history_for_calendar(
                calendar_symbol="AAA",
                calendar_history=history("AAA", "2026-01-05"),
                signal_symbol="BBB",
                signal_history=history("BBB", "2026-01-06"),
                auto_adjust=False,
                tradier_cfg=cfg,
            )

        fallback.assert_called_once()
        self.assertEqual(raised.exception.symbols, ["BBB"])
        self.assertIn("Yahoo Finance: No daily rows overlapped the retained AAA calendar", str(raised.exception))
        self.assertIn("Tradier fallback: No daily rows overlapped the retained AAA calendar", str(raised.exception))

    def test_calendar_anchored_merge_preserves_legitimate_partial_symbol_history(self) -> None:
        asset_frame = pd.DataFrame(
            {
                "AAA_Open": [10.0, 11.0],
                "AAA_High": [11.0, 12.0],
                "AAA_Low": [9.0, 10.0],
                "AAA_Close": [10.5, 11.5],
                "AAA_Volume": [1_000.0, 1_100.0],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )
        partial_signal_frame = pd.DataFrame(
            {
                "BBB_Open": [20.0],
                "BBB_High": [21.0],
                "BBB_Low": [19.0],
                "BBB_Close": [20.5],
                "BBB_Volume": [2_000.0],
            },
            index=pd.to_datetime(["2026-01-05"]),
        )

        with (
            patch(
                "leveraged_trader.market_data._download_yfinance",
                return_value=(pd.DataFrame(), {}),
            ),
            patch(
                "leveraged_trader.market_data._yfinance_symbol_frames",
                return_value=({"AAA": asset_frame, "BBB": partial_signal_frame}, {}),
            ),
            patch("leveraged_trader.market_data._load_tradier_fallback_frames") as fallback,
        ):
            data = load_market_data(
                symbols=["AAA", "BBB"],
                auto_adjust=False,
                calendar_symbol="AAA",
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        fallback.assert_not_called()
        self.assertTrue(pd.isna(data.loc[pd.Timestamp("2026-01-02"), "BBB_Close"]))
        self.assertEqual(data.loc[pd.Timestamp("2026-01-05"), "BBB_Close"], 20.5)

    @patch("leveraged_trader.market_data.yf.download")
    def test_multi_symbol_flat_yahoo_response_is_rejected_as_ambiguous(
        self,
        mock_download: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [10.0],
                "High": [11.0],
                "Low": [9.0],
                "Close": [10.5],
                "Volume": [1000.0],
            },
            index=pd.to_datetime(["2026-01-02"]),
        )

        with self.assertRaisesRegex(MarketDataDownloadError, "ambiguous flat-column response"):
            load_market_data(symbols=["AAA", "BBB"])

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_daily_index_must_be_unique_date_only_and_sorted(
        self,
        mock_download: Mock,
    ) -> None:
        cases = [
            (pd.to_datetime(["2026-01-02", "2026-01-02"]), "duplicate date"),
            (pd.to_datetime(["2026-01-02 12:00"]), "date-only"),
            (pd.to_datetime(["2026-01-03", "2026-01-02"]), "sorted in increasing order"),
        ]
        for index, error_pattern in cases:
            with self.subTest(error_pattern=error_pattern):
                mock_download.return_value = pd.DataFrame(
                    {
                        "Open": [10.0] * len(index),
                        "High": [11.0] * len(index),
                        "Low": [9.0] * len(index),
                        "Close": [10.5] * len(index),
                        "Volume": [1000.0] * len(index),
                    },
                    index=index,
                )

                with self.assertRaisesRegex(MarketDataDownloadError, error_pattern):
                    load_market_data(symbols=["AAA"])

    @patch("leveraged_trader.market_data.yf.download")
    def test_strategy_data_uses_asset_calendar_when_signal_session_is_missing(
        self,
        mock_download: Mock,
    ) -> None:
        dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
        fields = ["Open", "High", "Low", "Close", "Volume"]

        def fake_download(**kwargs: object) -> pd.DataFrame:
            symbols = list(kwargs["tickers"])
            if symbols == ["AAA", "BBB"]:
                return pd.DataFrame(
                    [
                        [10.0, 11.0, 9.0, 10.5, 1000.0, 20.0, 21.0, 19.0, 20.5, 2000.0],
                        [10.5, 12.0, 10.0, 11.5, 1100.0, None, None, None, None, None],
                    ],
                    index=dates,
                    columns=pd.MultiIndex.from_product([["AAA", "BBB"], fields]),
                )
            self.assertEqual(symbols, ["^IRX"])
            return pd.DataFrame(
                [
                    [5.0, 5.0, 5.0, 5.0, 0.0],
                    [5.0, 5.0, 5.0, 5.0, 0.0],
                ],
                index=dates,
                columns=pd.MultiIndex.from_product([["^IRX"], fields]),
            )

        mock_download.side_effect = fake_download

        data = load_strategy_data("AAA", "BBB")

        self.assertEqual(data.index.tolist(), dates.tolist())
        self.assertEqual(data["AAA_Close"].tolist(), [10.5, 11.5])
        self.assertEqual(float(data.loc[dates[0], "BBB_Close"]), 20.5)
        self.assertTrue(pd.isna(data.loc[dates[1], "BBB_Close"]))
        self.assertEqual(data["^IRX_Close"].tolist(), [5.0, 5.0])

    def test_current_session_daily_bar_is_excluded_before_finalization(self) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        data = pd.DataFrame({"AAA_Close": [100.0, 110.0]}, index=index)
        data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = ["AAA"]

        finalized = exclude_unfinalized_daily_bar(
            data,
            now=datetime(2026, 1, 5, 15, 30, tzinfo=ZoneInfo("America/New_York")),
        )

        self.assertEqual(finalized.index.tolist(), [pd.Timestamp("2026-01-02")])
        self.assertEqual(finalized.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["AAA"])

    def test_current_session_daily_bar_is_excluded_after_close_too(self) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        data = pd.DataFrame({"AAA_Close": [100.0, 110.0]}, index=index)

        finalized = exclude_unfinalized_daily_bar(
            data,
            now=datetime(2026, 1, 5, 16, 15, tzinfo=ZoneInfo("America/New_York")),
        )

        self.assertEqual(finalized.index.tolist(), [pd.Timestamp("2026-01-02")])

    def test_yfinance_errors_become_single_human_readable_download_error(self) -> None:
        def fake_download(**_: object) -> pd.DataFrame:
            yf_shared._ERRORS = {
                "PRICE": "$PRICE: possibly delisted; no timezone found",
                "CLOUD": "$CLOUD: possibly delisted; no timezone found",
            }
            return pd.DataFrame()

        with (
            patch("leveraged_trader.market_data.yf.download", side_effect=fake_download),
            self.assertRaises(MarketDataDownloadError) as raised,
        ):
            load_market_data(symbols=["PRICE", "CLOUD"])

        message = str(raised.exception)
        self.assertEqual(raised.exception.symbols, ["CLOUD", "PRICE"])
        self.assertIn("Yahoo Finance did not return usable daily data", message)
        self.assertIn("CLOUD, PRICE: no timezone found", message)
        self.assertNotIn("$PRICE", message)
        self.assertNotIn("Failed download", message)

    @patch("leveraged_trader.market_data.yf.download")
    def test_missing_downloaded_symbol_identifies_impacted_symbol(self, mock_download: Mock) -> None:
        index = pd.to_datetime(["2026-01-02"])
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
        mock_download.return_value = pd.DataFrame(
            [[1.0, 2.0, 0.5, 1.5, 1000]],
            index=index,
            columns=columns,
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["AAA", "MISSING"])

        message = str(raised.exception)
        self.assertEqual(raised.exception.symbols, ["MISSING"])
        self.assertIn("MISSING", message)
        self.assertIn("missing from the Yahoo Finance response", message)

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_invalid_ohlcv_identifies_provider_symbol_and_date(self, mock_download: Mock) -> None:
        index = pd.to_datetime(["2026-01-02"])
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
        mock_download.return_value = pd.DataFrame(
            [[10.0, 11.0, 9.0, -1.0, 1000.0]],
            index=index,
            columns=columns,
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["AAA"])

        self.assertEqual(raised.exception.source, "Yahoo Finance")
        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("2026-01-02", str(raised.exception))
        self.assertIn("Close must be positive and finite", str(raised.exception))

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_boolean_and_complex_ohlcv_are_rejected_before_coercion(self, mock_download: Mock) -> None:
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
        cases = {
            "boolean": [True, True, True, True, False],
            "complex": [10 + 1j, 11 + 1j, 9 + 1j, 10.5 + 1j, 1000 + 1j],
        }

        for label, row in cases.items():
            with self.subTest(label=label):
                mock_download.return_value = pd.DataFrame(
                    [row],
                    index=pd.to_datetime(["2026-01-02"]),
                    columns=columns,
                )

                with self.assertRaises(MarketDataDownloadError) as raised:
                    load_market_data(symbols=["AAA"])

                self.assertEqual(raised.exception.symbols, ["AAA"])
                self.assertIn("2026-01-02", str(raised.exception))
                self.assertIn("not boolean or complex", str(raised.exception))

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_temporal_ohlcv_is_rejected_instead_of_becoming_epoch_numeric(self, mock_download: Mock) -> None:
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
        cases = {
            "datetime64": pd.Series(pd.to_datetime(["2026-01-02"])),
            "timedelta64": pd.Series(pd.to_timedelta([1], unit="D")),
        }

        for label, temporal_values in cases.items():
            with self.subTest(label=label):
                frame = pd.DataFrame(
                    [[10.0, 11.0, 9.0, 10.5, 1000.0]],
                    index=pd.to_datetime(["2026-01-02"]),
                    columns=columns,
                )
                frame[("AAA", "Open")] = temporal_values.to_numpy()
                mock_download.return_value = frame

                with self.assertRaises(MarketDataDownloadError) as raised:
                    load_market_data(symbols=["AAA"])

                self.assertEqual(raised.exception.symbols, ["AAA"])
                self.assertIn("not a date, datetime, or timedelta", str(raised.exception))

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_accepts_negative_risk_free_yield_candles(self, mock_download: Mock) -> None:
        index = pd.to_datetime(["2020-03-18"])
        columns = pd.MultiIndex.from_product([["^IRX"], ["Open", "High", "Low", "Close", "Volume"]])
        mock_download.return_value = pd.DataFrame(
            [[-0.10, -0.05, -0.235, -0.105, 0.0]],
            index=index,
            columns=columns,
        )

        data = load_market_data(symbols=["^IRX"])

        self.assertEqual(data.index.tolist(), index.tolist())
        self.assertEqual(float(data.iloc[0]["^IRX_Low"]), -0.235)

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_partial_ohlcv_row_is_rejected_instead_of_silently_dropped(
        self,
        mock_download: Mock,
    ) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
        mock_download.return_value = pd.DataFrame(
            [
                [10.0, 11.0, 9.0, 10.5, 1000.0],
                [10.5, 12.0, 10.0, None, 1100.0],
            ],
            index=index,
            columns=columns,
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["AAA"])

        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("2026-01-05", str(raised.exception))
        self.assertIn("Close must be positive and finite", str(raised.exception))

    @patch("leveraged_trader.market_data.yf.download")
    def test_yahoo_nonnumeric_candle_does_not_masquerade_as_empty_calendar_row(
        self,
        mock_download: Mock,
    ) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
        mock_download.return_value = pd.DataFrame(
            [
                [10.0, 11.0, 9.0, 10.5, 1000.0],
                ["bad", "bad", "bad", "bad", "bad"],
            ],
            index=index,
            columns=columns,
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["AAA"])

        self.assertIn("2026-01-05", str(raised.exception))
        self.assertIn("Open must be positive and finite", str(raised.exception))

    @patch("leveraged_trader.market_data.yf.download")
    def test_single_symbol_yahoo_empty_candle_is_not_treated_as_cross_calendar_gap(
        self,
        mock_download: Mock,
    ) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])
        mock_download.return_value = pd.DataFrame(
            [
                [10.0, 11.0, 9.0, 10.5, 1000.0],
                [None, None, None, None, None],
            ],
            index=index,
            columns=columns,
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["AAA"])

        self.assertIn("2026-01-05", str(raised.exception))
        self.assertIn("Open must be positive and finite", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_incomplete_yahoo_ohlcv_frame_uses_tradier_fallback(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        index = pd.to_datetime(["2026-01-02"])
        mock_download.return_value = pd.DataFrame(
            [[10.5]],
            index=index,
            columns=pd.MultiIndex.from_tuples([("AAA", "Close")]),
        )
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1000,
                        }
                    }
                }
            ),
        )

        data = load_market_data(
            symbols=["AAA"],
            auto_adjust=False,
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(data.columns.tolist(), ["AAA_Open", "AAA_High", "AAA_Low", "AAA_Close", "AAA_Volume"])
        self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["AAA"])
        self.assertEqual(data.attrs[MARKET_DATA_PROVIDERS_ATTR], {"AAA": "tradier"})

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_nullable_yahoo_missing_price_is_normalized_and_allows_tradier_fallback(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        index = pd.to_datetime(["2026-01-02"])
        frame = pd.DataFrame(
            {
                "Open": pd.array([10.0], dtype="Float64"),
                "High": pd.array([11.0], dtype="Float64"),
                "Low": pd.array([9.0], dtype="Float64"),
                "Close": pd.array([pd.NA], dtype="Float64"),
                "Volume": pd.array([1_000.0], dtype="Float64"),
            },
            index=index,
        )
        mock_download.return_value = frame

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(symbols=["AAA"])
        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("Close must be positive and finite", str(raised.exception))

        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1_000,
                        }
                    }
                }
            ),
        )
        recovered = load_market_data(
            symbols=["AAA"],
            auto_adjust=False,
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(recovered["AAA_Close"].tolist(), [10.5])
        self.assertEqual(recovered.attrs[MARKET_DATA_PROVIDERS_ATTR], {"AAA": "tradier"})

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_duplicate_yahoo_ohlcv_column_uses_tradier_fallback(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1000,
                        }
                    }
                }
            ),
        )
        fields = ["Open", "High", "Low", "Close", "Close", "Volume"]
        column_cases = {
            "flat": fields,
            "ticker-first MultiIndex": pd.MultiIndex.from_tuples([("AAA", field) for field in fields]),
        }

        for case, columns in column_cases.items():
            with self.subTest(case=case):
                mock_download.return_value = pd.DataFrame(
                    [[10.0, 11.0, 9.0, 10.5, 99.0, 1000.0]],
                    index=pd.to_datetime(["2026-01-02"]),
                    columns=columns,
                )
                mock_get.reset_mock()

                data = load_market_data(
                    symbols=["AAA"],
                    auto_adjust=False,
                    tradier_cfg=TradierMarketDataConfig(access_token="token"),
                )

                self.assertEqual(data["AAA_Close"].tolist(), [10.5])
                self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["AAA"])
                self.assertEqual(data.attrs[MARKET_DATA_PROVIDERS_ATTR], {"AAA": "tradier"})
                mock_get.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_invalid_ohlcv_is_rejected_with_symbol_context(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "high": 9,
                            "low": 8,
                            "close": 10,
                            "volume": 1000,
                        }
                    }
                }
            ),
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("Tradier fallback", str(raised.exception))
        self.assertIn("High must be at least", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_post_rename_duplicate_required_field_is_download_error(
        self,
        mock_get: Mock,
    ) -> None:
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "Open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1000,
                        }
                    }
                }
            ),
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertEqual(raised.exception.source, "Tradier")
        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("duplicate required columns", str(raised.exception))
        self.assertIn("Open", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_stream_accepts_exact_byte_limit_and_closes(self, mock_get: Mock) -> None:
        body = json.dumps(
            {
                "history": {
                    "day": {
                        "date": "2026-01-02",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                    }
                }
            }
        ).encode()
        split = len(body) // 2
        response = _tradier_streaming_response(
            body[:split],
            body[split:],
            headers={"Content-Length": str(len(body))},
        )
        mock_get.return_value = response

        with patch("leveraged_trader.market_data.TRADIER_RESPONSE_MAX_BYTES", len(body)):
            result = _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertEqual(result["AAA_Close"].tolist(), [10.5])
        self.assertEqual(response.raw.read1.call_count, 3)
        response.close.assert_called_once()
        self.assertTrue(mock_get.call_args.kwargs["stream"])
        self.assertFalse(mock_get.call_args.kwargs["allow_redirects"])
        self.assertEqual(mock_get.call_args.kwargs["headers"]["Accept-Encoding"], "identity")
        request_timeout = mock_get.call_args.kwargs["timeout"]
        self.assertIsInstance(request_timeout, urllib3.util.Timeout)
        self.assertGreater(request_timeout.total, 0)
        self.assertLessEqual(request_timeout._read, 5.0)
        socket = response.raw._fp.fp.raw._sock
        self.assertEqual(socket.settimeout.call_count, 3)
        self.assertTrue(all(0 < call.args[0] <= 5.0 for call in socket.settimeout.call_args_list))

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_preserves_read_failure_when_response_close_also_fails(self, mock_get: Mock) -> None:
        response = _tradier_streaming_response()
        response.raw.read1.side_effect = requests.ReadTimeout("primary read failed")
        response.close.side_effect = OSError("response close failed")
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertIn("overall deadline", str(raised.exception))
        self.assertIn(
            "Failed to close the Tradier HTTP response: response close failed",
            getattr(raised.exception, "__notes__", []),
        )
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_preserves_http_error_when_response_close_also_fails(self, mock_get: Mock) -> None:
        access_token = "tradier-cleanup-secret-123456789"
        response = _tradier_streaming_response(
            b'{"error":{"message":"rate limited"}}',
            status_code=429,
        )
        response.close.side_effect = OSError(f"credential={access_token}\n" + "X" * 10_000)
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token=access_token),
                auto_adjust=False,
            )

        self.assertIn("rate limited", str(raised.exception))
        notes = getattr(raised.exception, "__notes__", [])
        self.assertEqual(len(notes), 1)
        self.assertIn("Failed to close the Tradier HTTP response", notes[0])
        self.assertIn("redacted", notes[0])
        self.assertNotIn(access_token, notes[0])
        self.assertNotIn("\n", notes[0])
        self.assertLessEqual(len(notes[0]), 600)
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_preserves_invalid_json_when_response_close_also_fails(self, mock_get: Mock) -> None:
        response = _tradier_streaming_response(b'{"history":')
        response.close.side_effect = OSError("response close failed")
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertIn("invalid JSON", str(raised.exception))
        self.assertIn(
            "Failed to close the Tradier HTTP response: response close failed",
            getattr(raised.exception, "__notes__", []),
        )
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_propagates_response_close_failure_after_valid_response(self, mock_get: Mock) -> None:
        body = json.dumps(
            {
                "history": {
                    "day": {
                        "date": "2026-01-02",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                    }
                }
            }
        ).encode()
        response = _tradier_streaming_response(body)
        response.close.side_effect = OSError("response close failed")
        mock_get.return_value = response

        with self.assertRaisesRegex(OSError, "response close failed"):
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        response.close.assert_called_once()

    def test_tradier_updates_socket_timeout_from_remaining_budget_before_each_read(self) -> None:
        response = _tradier_streaming_response(b"a")

        with patch(
            "leveraged_trader.market_data.time.monotonic",
            side_effect=[0.0, 6.0, 6.0, 7.0, 8.0, 8.0],
        ):
            _materialize_tradier_response(response, deadline=10.0)

        socket = response.raw._fp.fp.raw._sock
        self.assertEqual(
            [call.args[0] for call in socket.settimeout.call_args_list],
            [4.0, 2.0],
        )

    def test_tradier_accepts_buffered_body_after_connection_close(self) -> None:
        body = json.dumps({"history": {"day": []}}).encode()

        class ClosingHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *args: object) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), ClosingHandler)
        worker = threading.Thread(target=server.handle_request, daemon=True)
        worker.start()
        response: requests.Response | None = None
        try:
            response = requests.get(
                f"http://127.0.0.1:{server.server_port}/history",
                headers={"Accept-Encoding": "identity"},
                timeout=5.0,
                stream=True,
            )
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())

            _materialize_tradier_response(response, deadline=time.monotonic() + 5.0)

            self.assertEqual(response.content, body)
        finally:
            if response is not None:
                response.close()
            server.server_close()

    def test_tradier_wall_clock_deadline_interrupts_slow_headers_and_connection_close_body(self) -> None:
        scenarios = ("headers", "connection-close-body")
        for scenario in scenarios:
            with self.subTest(scenario=scenario):

                class SlowHandler(BaseHTTPRequestHandler):
                    protocol_version = "HTTP/1.0"
                    test_scenario = ""

                    def do_GET(self) -> None:
                        try:
                            if self.test_scenario == "headers":
                                wire_response = b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n{}"
                                for byte in wire_response:
                                    self.connection.sendall(bytes((byte,)))
                                    time.sleep(0.08)
                                return
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            for byte in b'{"history":{"day":[]}}':
                                self.wfile.write(bytes((byte,)))
                                self.wfile.flush()
                                time.sleep(0.15)
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            pass

                    def log_message(self, _format: str, *args: object) -> None:
                        pass

                SlowHandler.test_scenario = scenario
                server = HTTPServer(("127.0.0.1", 0), SlowHandler)
                worker = threading.Thread(target=server.handle_request, daemon=True)
                processes: list[subprocess.Popen[bytes]] = []
                real_popen = subprocess.Popen

                def capture_process(
                    *args: object,
                    _real_popen: type[subprocess.Popen[bytes]] = real_popen,
                    _processes: list[subprocess.Popen[bytes]] = processes,
                    **kwargs: object,
                ) -> subprocess.Popen[bytes]:
                    process = _real_popen(*args, **kwargs)
                    _processes.append(process)
                    return process

                worker.start()
                started = time.monotonic()
                try:
                    with (
                        patch(
                            "leveraged_trader._http_deadline_worker.subprocess.Popen",
                            side_effect=capture_process,
                        ),
                        self.assertRaisesRegex(requests.Timeout, "overall deadline"),
                    ):
                        _get_tradier_response_with_deadline(
                            f"http://127.0.0.1:{server.server_port}/history",
                            deadline=time.monotonic() + 1.0,
                            request_kwargs={
                                "headers": {"Accept-Encoding": "identity"},
                                "timeout": urllib3.util.Timeout(total=10.0, connect=10.0, read=5.0),
                                "allow_redirects": False,
                                "stream": True,
                            },
                        )
                    self.assertLess(time.monotonic() - started, 1.5)
                    self.assertEqual(len(processes), 1)
                    self.assertGreater(processes[0].pid, 0)
                    # Inspect returncode directly: poll() would reap a zombie and
                    # could hide a production cleanup regression.
                    self.assertIsNotNone(
                        processes[0].returncode,
                        f"Tradier deadline worker {processes[0].pid} was not reaped",
                    )
                finally:
                    for process in processes:
                        if process.returncode is None:
                            process.kill()
                        process.wait(timeout=3.0)
                    worker.join(timeout=3.0)
                    server.server_close()
                self.assertFalse(worker.is_alive())

    def test_tradier_worker_response_schema_rejects_malformed_payloads(self) -> None:
        valid_response = (
            "response",
            200,
            {"Content-Type": "application/json"},
            "utf-8",
            b"{}",
            "https://api.tradier.com/v1/markets/history",
            "OK",
        )
        invalid_payloads = (
            [],
            (),
            (1, 200, {}, None, b"", "https://api.tradier.com", None),
            ("error", 1, "failure"),
            ("error", "", "failure"),
            ("error", "Timeout", 1),
            ("error", "Timeout", "x" * 4097),
            ("response", 200, {}, None, b"", "https://api.tradier.com"),
            (*valid_response, "trailing"),
            ("response", True, {}, None, b"", "https://api.tradier.com", None),
            ("response", "200", {}, None, b"", "https://api.tradier.com", None),
            ("response", 99, {}, None, b"", "https://api.tradier.com", None),
            ("response", 600, {}, None, b"", "https://api.tradier.com", None),
            ("response", 200, [], None, b"", "https://api.tradier.com", None),
            ("response", 200, {1: "value"}, None, b"", "https://api.tradier.com", None),
            ("response", 200, {"Header": 1}, None, b"", "https://api.tradier.com", None),
            ("response", 200, {}, b"utf-8", b"", "https://api.tradier.com", None),
            ("response", 200, {}, None, bytearray(), "https://api.tradier.com", None),
            ("response", 200, {}, None, 10**12, "https://api.tradier.com", None),
            ("response", 200, {}, None, b"12345", "https://api.tradier.com", None),
            ("response", 200, {}, None, b"", "", None),
            ("response", 200, {}, None, b"", 1, None),
            ("response", 200, {}, None, b"", "https://api.tradier.com", 1),
        )

        for payload in invalid_payloads:
            with (
                self.subTest(payload=payload),
                patch(
                    "leveraged_trader.market_data.run_http_request_with_deadline",
                    return_value=payload,
                ),
                patch("leveraged_trader.market_data.TRADIER_RESPONSE_MAX_BYTES", 4),
                self.assertRaisesRegex(requests.exceptions.ConnectionError, "invalid response"),
            ):
                _get_tradier_response_with_deadline(
                    "https://api.tradier.com/v1/markets/history",
                    deadline=time.monotonic() + 1,
                    request_kwargs={},
                )

    def test_tradier_worker_response_schema_preserves_valid_exact_types(self) -> None:
        payload = (
            "response",
            200,
            {"Content-Type": "application/octet-stream"},
            None,
            b"\x00\xff",
            "https://api.tradier.com/v1/markets/history",
            b"OK",
        )
        with patch(
            "leveraged_trader.market_data.run_http_request_with_deadline",
            return_value=payload,
        ):
            response = _get_tradier_response_with_deadline(
                "https://api.tradier.com/v1/markets/history",
                deadline=time.monotonic() + 1,
                request_kwargs={},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers, {"Content-Type": "application/octet-stream"})
        self.assertIsNone(response.encoding)
        self.assertEqual(response.content, b"\x00\xff")
        self.assertEqual(response.url, "https://api.tradier.com/v1/markets/history")
        self.assertEqual(response.reason, b"OK")

    def test_worker_serializer_rejects_over_limit_without_pickle_dumps(self) -> None:
        with (
            patch.object(
                http_deadline_worker.pickle,
                "dumps",
                side_effect=AssertionError("pickle.dumps must not materialize the worker envelope"),
            ) as legacy_dumps,
            self.assertRaisesRegex(ValueError, "64-byte limit"),
        ):
            http_deadline_worker._serialize_envelope(b"x" * 1_024, max_bytes=64)

        legacy_dumps.assert_not_called()

    def test_worker_serializer_accepts_protocol_five_pickle_buffers(self) -> None:
        payload = pd.DataFrame({"close": np.array([1.0, 2.0])})

        encoded = http_deadline_worker._serialize_envelope(payload, max_bytes=16_384)

        decoded = http_deadline_worker._deserialize_envelope(encoded, max_bytes=16_384)
        pd.testing.assert_frame_equal(decoded, payload)

    def test_deadline_worker_interrupts_partial_result_output_and_reaps_process(self) -> None:
        processes: list[subprocess.Popen[bytes]] = []
        real_popen = subprocess.Popen

        def capture_process(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        started = time.monotonic()
        with (
            patch("leveraged_trader._http_deadline_worker.subprocess.Popen", side_effect=capture_process),
            self.assertRaisesRegex(requests.Timeout, "partial-result deadline"),
        ):
            run_deadline_subprocess(
                (
                    sys.executable,
                    "-I",
                    "-c",
                    "import sys,time; sys.stdout.buffer.write(b'partial'); sys.stdout.buffer.flush(); time.sleep(5)",
                ),
                {"probe": True},
                deadline=time.monotonic() + 0.2,
                result_max_bytes=1024,
                timeout_message="partial-result deadline",
                connection_error_message="partial-result worker failed",
            )

        self.assertLess(time.monotonic() - started, 0.75)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)

    def test_deadline_worker_bounds_flooding_result_and_reaps_process(self) -> None:
        processes: list[subprocess.Popen[bytes]] = []
        real_popen = subprocess.Popen

        def capture_process(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        started = time.monotonic()
        with (
            patch("leveraged_trader._http_deadline_worker.subprocess.Popen", side_effect=capture_process),
            self.assertRaisesRegex(requests.ConnectionError, "flooding worker failed"),
        ):
            run_deadline_subprocess(
                (
                    sys.executable,
                    "-I",
                    "-c",
                    "import sys; block=b'x'*65536;\n"
                    "while True: sys.stdout.buffer.write(block); sys.stdout.buffer.flush()",
                ),
                {"probe": True},
                deadline=time.monotonic() + 5,
                result_max_bytes=1024,
                timeout_message="flooding worker deadline",
                connection_error_message="flooding worker failed",
            )

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)

    def test_tradier_production_worker_success_is_fast_and_ignores_parent_wrapper(self) -> None:
        body = b'{"history":{"day":[]}}'

        class SuccessHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Worker-Probe", "present")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *args: object) -> None:
                pass

        wrapper_called = False

        def parent_wrapper(*_args: object, **_kwargs: object) -> requests.Response:
            nonlocal wrapper_called
            wrapper_called = True
            raise AssertionError("parent requests.get wrapper must not run")

        server = HTTPServer(("127.0.0.1", 0), SuccessHandler)
        worker = threading.Thread(target=server.handle_request, daemon=True)
        worker.start()
        started = time.monotonic()
        deadline = started + 1.0
        response: requests.Response | None = None
        try:
            with patch("leveraged_trader.market_data.requests.get", new=parent_wrapper):
                response = _get_tradier_response_with_deadline(
                    f"http://127.0.0.1:{server.server_port}/history",
                    deadline=deadline,
                    request_kwargs={
                        "headers": {"Accept-Encoding": "identity"},
                        "timeout": urllib3.util.Timeout(total=1.0, connect=1.0, read=1.0),
                        "allow_redirects": False,
                        "stream": True,
                    },
                )
                _materialize_tradier_response(response, deadline=deadline)
        finally:
            worker.join(timeout=2.0)
            server.server_close()

        self.assertFalse(worker.is_alive())
        self.assertFalse(wrapper_called)
        self.assertLess(time.monotonic() - started, 1.0)
        assert response is not None
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Worker-Probe"], "present")
        self.assertEqual(response.content, body)
        self.assertEqual(response.encoding, "utf-8")

    def test_tradier_worker_does_not_replace_bearer_header_from_netrc(self) -> None:
        received_authorization: list[str | None] = []

        class AuthorizationHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                received_authorization.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, _format: str, *args: object) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), AuthorizationHandler)
        server_worker = threading.Thread(target=server.handle_request, daemon=True)
        response: requests.Response | None = None
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                netrc_path = Path(temporary_directory) / "netrc"
                netrc_path.write_text(
                    "machine 127.0.0.1 login ambient-user password ambient-password\n",
                    encoding="utf-8",
                )
                netrc_path.chmod(0o600)
                server_worker.start()
                deadline = time.monotonic() + 3.0
                with patch.dict(
                    os.environ,
                    {"NETRC": str(netrc_path), "NO_PROXY": "127.0.0.1,localhost"},
                ):
                    response = _get_tradier_response_with_deadline(
                        f"http://127.0.0.1:{server.server_port}/history",
                        deadline=deadline,
                        request_kwargs={
                            "headers": {
                                "Authorization": "Bearer explicit-token",
                                "Accept-Encoding": "identity",
                            },
                            "timeout": urllib3.util.Timeout(total=2.0, connect=2.0, read=2.0),
                            "allow_redirects": False,
                            "stream": True,
                        },
                    )
        finally:
            if response is not None:
                response.close()
            server_worker.join(timeout=3.0)
            server.server_close()

        self.assertFalse(server_worker.is_alive())
        self.assertEqual(received_authorization, ["Bearer explicit-token"])

    def test_tradier_worker_rejects_redirects_before_request(self) -> None:
        session = Mock()
        with patch.object(http_deadline_worker.requests, "Session", return_value=session):
            payload = http_deadline_worker._http_request_payload(
                {
                    "max_body_bytes": 1_024,
                    "mode": "tradier",
                    "request_kwargs": {"allow_redirects": True, "stream": True},
                    "url": "https://api.tradier.com/v1/markets/history",
                }
            )

        self.assertEqual(payload[:2], ("error", "TypeError"))
        self.assertIn("must disable redirects", payload[2])
        session.get.assert_not_called()
        session.close.assert_called_once()

    def test_tradier_production_worker_runs_inside_daemonic_process(self) -> None:
        body = b'{"history":{"day":[]}}'

        class SuccessHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *args: object) -> None:
                pass

        server = HTTPServer(("127.0.0.1", 0), SuccessHandler)
        server_worker = threading.Thread(target=server.handle_request, daemon=True)
        context = multiprocessing.get_context("spawn")
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=_tradier_request_from_daemon,
            args=(f"http://127.0.0.1:{server.server_port}/history", send_connection),
        )
        process.daemon = True
        server_worker.start()
        try:
            process.start()
            send_connection.close()
            self.assertTrue(receive_connection.poll(5.0))
            self.assertEqual(receive_connection.recv(), ("response", 200, body.decode()))
            process.join(timeout=2.0)
            self.assertFalse(process.is_alive())
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            receive_connection.close()
            send_connection.close()
            process.close()
            server_worker.join(timeout=2.0)
            server.server_close()

        self.assertFalse(server_worker.is_alive())

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_fails_closed_when_stream_has_no_bounded_socket_timeout(
        self,
        mock_get: Mock,
    ) -> None:
        response = _tradier_streaming_response(b"unread")
        response.raw._fp = None
        response.raw._connection = None
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertIn("does not expose a bounded read timeout", str(raised.exception))
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_uses_bounded_raw_read_fallback(self, mock_get: Mock) -> None:
        body = json.dumps(
            {
                "history": {
                    "day": {
                        "date": "2026-01-02",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                    }
                }
            }
        ).encode()
        response = _tradier_streaming_response()
        response.raw.read1 = None
        response.raw.read.side_effect = [body, b""]
        mock_get.return_value = response

        result = _load_tradier_symbol_frame(
            "AAA",
            None,
            None,
            TradierMarketDataConfig(access_token="token"),
            auto_adjust=False,
        )

        self.assertEqual(result["AAA_Close"].tolist(), [10.5])
        self.assertEqual(response.raw.read.call_count, 2)
        self.assertTrue(all(call.args == (64 * 1024,) and not call.kwargs for call in response.raw.read.call_args_list))
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_supports_simple_positional_read1_signature(self, mock_get: Mock) -> None:
        body = json.dumps(
            {
                "history": {
                    "day": {
                        "date": "2026-01-02",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                    }
                }
            }
        ).encode()
        response = _tradier_streaming_response()
        chunks = iter((body, b""))
        requested_sizes: list[int] = []

        def read1(amount: int) -> bytes:
            requested_sizes.append(amount)
            return next(chunks)

        response.raw.read1 = read1
        mock_get.return_value = response

        result = _load_tradier_symbol_frame(
            "AAA",
            None,
            None,
            TradierMarketDataConfig(access_token="token"),
            auto_adjust=False,
        )

        self.assertEqual(result["AAA_Close"].tolist(), [10.5])
        self.assertEqual(requested_sizes, [64 * 1024, 64 * 1024])
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_accepts_bounded_already_materialized_response(self, mock_get: Mock) -> None:
        body = json.dumps(
            {
                "history": {
                    "day": {
                        "date": "2026-01-02",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                    }
                }
            }
        ).encode()
        response = _tradier_streaming_response()
        response._content = body
        response._content_consumed = True
        mock_get.return_value = response

        result = _load_tradier_symbol_frame(
            "AAA",
            None,
            None,
            TradierMarketDataConfig(access_token="token"),
            auto_adjust=False,
        )

        self.assertEqual(result["AAA_Close"].tolist(), [10.5])
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_oversized_http_error_content_length_without_reading(
        self,
        mock_get: Mock,
    ) -> None:
        response = _tradier_streaming_response(
            b"unread error body",
            status_code=500,
            headers={"Content-Length": "9"},
        )
        mock_get.return_value = response

        with (
            patch("leveraged_trader.market_data.TRADIER_RESPONSE_MAX_BYTES", 8),
            self.assertRaises(MarketDataDownloadError) as raised,
        ):
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertIn("8-byte limit", str(raised.exception))
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_redirect_with_valid_looking_history_body(self, mock_get: Mock) -> None:
        body = json.dumps(
            {
                "history": {
                    "day": {
                        "date": "2026-01-02",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                    }
                }
            }
        ).encode()
        response = _tradier_streaming_response(
            body,
            status_code=302,
            headers={"Location": "https://api.tradier.com/v1/elsewhere"},
        )
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertIn("unexpected HTTP status 302", str(raised.exception))
        self.assertFalse(mock_get.call_args.kwargs["allow_redirects"])
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_partial_content_without_materializing_valid_history(
        self,
        mock_get: Mock,
    ) -> None:
        body = json.dumps(
            {
                "history": {
                    "day": {
                        "date": "2026-01-02",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                    }
                }
            }
        ).encode()
        response = _tradier_streaming_response(
            body,
            status_code=206,
            headers={"Content-Range": f"bytes 0-{len(body) - 1}/{len(body) + 100}"},
        )
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertIn("unexpected HTTP status 206", str(raised.exception))
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_content_range_on_exact_200_without_materializing_history(
        self,
        mock_get: Mock,
    ) -> None:
        body = json.dumps(
            {
                "history": {
                    "day": {
                        "date": "2026-01-02",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                    }
                }
            }
        ).encode()
        response = _tradier_streaming_response(
            body,
            status_code=200,
            headers={"cOnTeNt-RaNgE": f"bytes 0-{len(body) - 1}/{len(body) + 100}"},
        )
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertIn("unsolicited Content-Range", str(raised.exception))
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    def test_tradier_worker_rejects_content_range_on_exact_200_without_reading(self) -> None:
        body = b'{"history":{"day":[]}}'
        response = _tradier_streaming_response(
            body,
            status_code=200,
            headers={"cOnTeNt-RaNgE": f"bytes 0-{len(body) - 1}/{len(body) + 100}"},
        )
        session = Mock()
        session.get.return_value = response

        with patch.object(http_deadline_worker.requests, "Session", return_value=session):
            payload = http_deadline_worker._http_request_payload(
                {
                    "max_body_bytes": 1_024,
                    "mode": "tradier",
                    "request_kwargs": {"allow_redirects": False, "stream": True},
                    "url": response.url,
                }
            )

        self.assertEqual(payload[:2], ("error", "InvalidHeader"))
        self.assertIn("unsolicited Content-Range", payload[2])
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()
        session.close.assert_called_once()

    def test_tradier_worker_preserves_read_failure_and_reports_all_cleanup_failures(self) -> None:
        response = _tradier_streaming_response()
        response.raw.read1.side_effect = requests.ReadTimeout("primary read failed")
        response.close.side_effect = OSError("response close failed")
        session = Mock()
        session.get.return_value = response
        session.close.side_effect = RuntimeError("session close failed")

        with patch.object(http_deadline_worker.requests, "Session", return_value=session):
            payload = http_deadline_worker._http_request_payload(
                {
                    "max_body_bytes": 1_024,
                    "mode": "tradier",
                    "request_kwargs": {"allow_redirects": False, "stream": True},
                    "url": response.url,
                }
            )

        self.assertEqual(payload[:2], ("error", "ReadTimeout"))
        self.assertIn("primary read failed", payload[2])
        self.assertIn("failed to close the HTTP response: response close failed", payload[2])
        self.assertIn("failed to close the HTTP session: session close failed", payload[2])
        response.close.assert_called_once()
        session.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_preserves_http_error_diagnostic_with_content_range(self, mock_get: Mock) -> None:
        body = b'{"error":{"message":"rate limited"}}'
        response = _tradier_streaming_response(
            status_code=429,
            headers={"cOnTeNt-RaNgE": f"bytes 0-{len(body) - 1}/{len(body)}"},
        )
        response._content = body
        response._content_consumed = True
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertIn("rate limited", str(raised.exception))
        response.raw.read1.assert_not_called()
        response.close.assert_called_once()

    def test_tradier_worker_suppresses_partial_body_and_preserves_http_error_body(self) -> None:
        history_body = json.dumps(
            {
                "history": {
                    "day": {
                        "date": "2026-01-02",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                    }
                }
            }
        ).encode()
        error_body = b'{"error":{"message":"rate limited"}}'
        cases = (
            (
                206,
                history_body,
                {"Content-Range": f"bytes 0-{len(history_body) - 1}/{len(history_body) + 100}"},
                b"",
                0,
            ),
            (
                429,
                error_body,
                {"cOnTeNt-RaNgE": f"bytes 0-{len(error_body) - 1}/{len(error_body)}"},
                error_body,
                2,
            ),
        )

        for status_code, response_body, headers, expected_body, expected_reads in cases:
            with self.subTest(status_code=status_code):
                response = _tradier_streaming_response(
                    response_body,
                    status_code=status_code,
                    headers=headers,
                )
                session = Mock()
                session.get.return_value = response

                with patch.object(http_deadline_worker.requests, "Session", return_value=session):
                    payload = http_deadline_worker._http_request_payload(
                        {
                            "max_body_bytes": 1_024,
                            "mode": "tradier",
                            "request_kwargs": {"allow_redirects": False, "stream": True},
                            "url": response.url,
                        }
                    )

                self.assertEqual(payload[:2], ("response", status_code))
                self.assertEqual(payload[4], expected_body)
                self.assertEqual(response.raw.read1.call_count, expected_reads)
                response.close.assert_called_once()
                session.close.assert_called_once()

    @patch("leveraged_trader.market_data._strict_response_json")
    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_stream_over_byte_limit_before_json_parser(
        self,
        mock_get: Mock,
        mock_strict_json: Mock,
    ) -> None:
        response = _tradier_streaming_response(b"abcd", b"efghi")
        mock_get.return_value = response

        with (
            patch("leveraged_trader.market_data.TRADIER_RESPONSE_MAX_BYTES", 8),
            self.assertRaises(MarketDataDownloadError) as raised,
        ):
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertIn("8-byte limit", str(raised.exception))
        mock_strict_json.assert_not_called()
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_compressed_or_invalid_response_metadata(self, mock_get: Mock) -> None:
        cases = (
            (
                "compressed",
                _tradier_streaming_response(b"compressed", headers={"Content-Encoding": "gzip"}),
                "compressed content",
            ),
            (
                "invalid length",
                _tradier_streaming_response(b"body", headers={"Content-Length": "not-a-number"}),
                "invalid Content-Length",
            ),
        )
        for scenario, response, expected in cases:
            mock_get.return_value = response
            with self.subTest(scenario=scenario), self.assertRaises(MarketDataDownloadError) as raised:
                _load_tradier_symbol_frame(
                    "AAA",
                    None,
                    None,
                    TradierMarketDataConfig(access_token="token"),
                    auto_adjust=False,
                )

            self.assertIn(expected, str(raised.exception))
            response.raw.read1.assert_not_called()
            response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_normalizes_urllib3_raw_read_failures_and_closes(self, mock_get: Mock) -> None:
        access_token = "Pineapple987Quartz"
        failures = (
            urllib3.exceptions.ReadTimeoutError(
                None,
                "https://api.tradier.com/v1/markets/history",
                f"provider echoed {access_token}",
            ),
            urllib3.exceptions.ProtocolError(f"connection failed with {access_token}"),
        )
        for failure in failures:
            response = _tradier_streaming_response()
            response.raw.read1.side_effect = failure
            mock_get.return_value = response

            with self.subTest(failure=type(failure).__name__), self.assertRaises(MarketDataDownloadError) as raised:
                _load_tradier_symbol_frame(
                    "AAA",
                    None,
                    None,
                    TradierMarketDataConfig(access_token=access_token),
                    auto_adjust=False,
                )

            diagnostic = str(raised.exception)
            self.assertNotIn(access_token, diagnostic)
            self.assertIn("redacted", diagnostic)
            response.close.assert_called_once()

    @patch("leveraged_trader.market_data.time.monotonic")
    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_response_that_reaches_eof_after_deadline(
        self,
        mock_get: Mock,
        mock_monotonic: Mock,
    ) -> None:
        response = _tradier_streaming_response()
        mock_get.return_value = response
        mock_monotonic.side_effect = [0.0, 0.0, 0.0, 31.0]

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token", timeout_seconds=30),
                auto_adjust=False,
            )

        self.assertIn("overall deadline", str(raised.exception))
        response.close.assert_called_once()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_duplicate_json_object_keys_before_schema_validation(
        self,
        mock_get: Mock,
    ) -> None:
        response = Mock(
            status_code=200,
            text=(
                '{"history":{"day":{"date":"2026-01-02","date":"2026-01-03",'
                '"open":10,"high":11,"low":9,"close":10.5,"volume":1000}}}'
            ),
        )
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertEqual(raised.exception.source, "Tradier")
        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("ambiguous JSON", str(raised.exception))
        self.assertIn("duplicate object key 'date'", str(raised.exception))
        response.json.assert_not_called()

    def test_tradier_rejects_excessive_json_structure_before_decoding(self) -> None:
        response = Mock(text="[{}, {}, {}]")
        with (
            patch("leveraged_trader.market_data._MAX_JSON_STRUCTURAL_TOKENS", 4),
            patch("leveraged_trader.market_data.json.loads") as mock_loads,
            self.assertRaisesRegex(ValueError, "JSON structure exceeds the supported limit"),
        ):
            _strict_response_json(response)

        mock_loads.assert_not_called()

        quoted_structure = json.dumps({"value": "{},:[]" * 100})
        with patch("leveraged_trader.market_data._MAX_JSON_STRUCTURAL_TOKENS", 2):
            self.assertEqual(_strict_response_json(Mock(text=quoted_structure)), {"value": "{},:[]" * 100})

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_direct_helper_redacts_access_token_from_http_200_payload_errors(
        self,
        mock_get: Mock,
    ) -> None:
        access_token = "Pineapple987Quartz"
        valid_day = {
            "date": "2026-01-02",
            "open": 10,
            "high": 11,
            "low": 9,
            "close": 10.5,
            "volume": 1000,
        }
        cases = (
            (
                "duplicate JSON key",
                Mock(
                    status_code=200,
                    text=f'{{"{access_token}":1,"{access_token}":2}}',
                ),
            ),
            (
                "invalid daily date",
                Mock(
                    status_code=200,
                    text="",
                    json=Mock(return_value={"history": {"day": {**valid_day, "date": access_token}}}),
                ),
            ),
            (
                "invalid OHLCV",
                Mock(
                    status_code=200,
                    text="",
                    json=Mock(return_value={"history": {"day": {**valid_day, "close": access_token}}}),
                ),
            ),
        )

        for scenario, response in cases:
            mock_get.return_value = response
            with self.subTest(scenario=scenario), self.assertRaises(MarketDataDownloadError) as raised:
                _load_tradier_symbol_frame(
                    "AAA",
                    None,
                    None,
                    TradierMarketDataConfig(access_token=access_token),
                    auto_adjust=False,
                )

            diagnostic = raised.exception.symbol_reasons["AAA"]
            self.assertNotIn(access_token, diagnostic)
            self.assertNotIn(access_token, str(raised.exception))
            self.assertIn("redacted", diagnostic)

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_nonfinite_json_numbers_at_any_depth(self, mock_get: Mock) -> None:
        invalid_values = ("NaN", "Infinity", "-Infinity", "1e10000")
        for value in invalid_values:
            response = Mock(
                status_code=200,
                text=(
                    '{"history":{"day":{"date":"2026-01-02","open":10,"high":11,'
                    '"low":9,"close":10.5,"volume":1000},"ignored":{"value":'
                    f"{value}" + "}}}"
                ),
            )
            mock_get.return_value = response

            with self.subTest(value=value), self.assertRaises(MarketDataDownloadError) as raised:
                _load_tradier_symbol_frame(
                    "AAA",
                    None,
                    None,
                    TradierMarketDataConfig(access_token="token"),
                    auto_adjust=False,
                )

            self.assertEqual(raised.exception.source, "Tradier")
            self.assertIn("invalid JSON", str(raised.exception))
            response.json.assert_not_called()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_normalizes_excessively_nested_raw_json_as_download_error(self, mock_get: Mock) -> None:
        nesting_depth = 10_000
        response = Mock(
            status_code=200,
            text="[" * nesting_depth + "0" + "]" * nesting_depth,
        )
        mock_get.return_value = response

        with (
            patch("leveraged_trader.market_data.json.loads") as mock_json_loads,
            self.assertRaises(MarketDataDownloadError) as raised,
        ):
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertEqual(raised.exception.source, "Tradier")
        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("invalid JSON", str(raised.exception))
        self.assertIn("excessive nesting", str(raised.exception))
        mock_json_loads.assert_not_called()
        response.json.assert_not_called()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_excessively_nested_json_object_fallback(self, mock_get: Mock) -> None:
        nested_payload: object = 0
        for _ in range(10_000):
            nested_payload = [nested_payload]
        response = Mock(
            status_code=200,
            text="",
            json=Mock(return_value=nested_payload),
        )
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            _load_tradier_symbol_frame(
                "AAA",
                None,
                None,
                TradierMarketDataConfig(access_token="token"),
                auto_adjust=False,
            )

        self.assertEqual(raised.exception.source, "Tradier")
        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("invalid JSON with excessive nesting", str(raised.exception))
        response.json.assert_called_once_with()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_json_nesting_guard_ignores_escaped_string_content(self, mock_get: Mock) -> None:
        response = Mock(
            status_code=200,
            text=json.dumps(
                {
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1000,
                        }
                    },
                    "ignored": '"' + "[{" * 10_000,
                }
            ),
        )
        mock_get.return_value = response

        result = _load_tradier_symbol_frame(
            "AAA",
            None,
            None,
            TradierMarketDataConfig(access_token="token"),
            auto_adjust=False,
        )

        self.assertEqual(result["AAA_Close"].tolist(), [10.5])
        response.json.assert_not_called()

    @patch("leveraged_trader.market_data.yf.download")
    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_http_error_json_is_strict_and_recursion_safe(
        self,
        mock_get: Mock,
        mock_download: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        cases = (
            (
                '{"error":{"message":"first"},"error":{"message":"second"}}',
                "ambiguous error JSON",
            ),
            ('{"error":{"message":"failed"},"metadata":NaN}', "non-finite number"),
            ("[" * 10_000 + "0" + "]" * 10_000, "excessive nesting"),
        )
        for body, expected_message in cases:
            response = Mock(status_code=500, text=body)
            mock_get.return_value = response

            with self.subTest(expected_message=expected_message), self.assertRaises(MarketDataDownloadError) as raised:
                load_market_data(
                    symbols=["AAA"],
                    auto_adjust=False,
                    tradier_cfg=TradierMarketDataConfig(access_token="token"),
                )

            diagnostic = raised.exception.symbol_reasons["AAA"]
            self.assertIn(expected_message, diagnostic)
            self.assertNotIn("first", diagnostic)
            self.assertNotIn("second", diagnostic)
            response.json.assert_not_called()

    @patch("leveraged_trader.market_data.yf.download")
    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_http_error_diagnostic_is_bounded_printable_and_redacted(
        self,
        mock_get: Mock,
        mock_download: Mock,
    ) -> None:
        secret = "TOP-SECRET-TRADIER-TOKEN"
        padding = "X" * 1_000_000
        mock_download.return_value = pd.DataFrame()
        response = Mock(
            status_code=401,
            text=json.dumps({"error": {"message": f"credential: {secret}\x00\x1b[2J {padding}"}}),
        )
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token=secret),
            )

        diagnostic = raised.exception.symbol_reasons["AAA"]
        self.assertLessEqual(len(diagnostic), 512)
        self.assertTrue(all(character.isprintable() for character in diagnostic))
        self.assertNotIn(secret, diagnostic)
        self.assertNotIn("\x00", diagnostic)
        self.assertNotIn("\x1b", diagnostic)
        self.assertNotIn(padding[:1_000], diagnostic)
        self.assertIn("redacted credential", diagnostic)
        response.json.assert_not_called()

    @patch("leveraged_trader.market_data.yf.download")
    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_http_error_redacts_non_http_uri_userinfo(
        self,
        mock_get: Mock,
        mock_download: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        response = Mock(
            status_code=502,
            text=json.dumps(
                {
                    "error": {
                        "message": "proxy socks5://operator:TOPSECRET123@proxy.example failed",
                    }
                }
            ),
        )
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token="unrelated-token"),
            )

        diagnostic = raised.exception.symbol_reasons["AAA"]
        self.assertIn("proxy socks5://operator:[redacted]@proxy.example failed", diagnostic)
        self.assertNotIn("TOPSECRET123", diagnostic)
        response.json.assert_not_called()

    @patch("leveraged_trader.market_data.yf.download")
    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_http_error_does_not_disclose_an_oversized_access_token_prefix(
        self,
        mock_get: Mock,
        mock_download: Mock,
    ) -> None:
        access_token = "Q" * 3_000
        mock_download.return_value = pd.DataFrame()
        response = Mock(
            status_code=401,
            text=json.dumps({"error": {"message": f"provider echoed {access_token}"}}),
        )
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token=access_token),
            )

        diagnostic = raised.exception.symbol_reasons["AAA"]
        self.assertEqual(diagnostic, "[redacted credential]")
        self.assertNotIn(access_token[:512], diagnostic)
        response.json.assert_not_called()

    @patch("leveraged_trader.market_data.yf.download")
    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_http_error_redacts_access_token_crossing_diagnostic_scan_boundary(
        self,
        mock_get: Mock,
        mock_download: Mock,
    ) -> None:
        access_token = "opaquecredential1234"
        leaked_prefix = access_token[:18]
        mock_download.return_value = pd.DataFrame()
        response = Mock(
            status_code=403,
            text=json.dumps(
                {
                    "error": {
                        "message": "\x00" * (512 * 4 - len(leaked_prefix)) + access_token,
                    }
                }
            ),
        )
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token=access_token),
            )

        diagnostic = str(raised.exception)
        self.assertNotIn(access_token, diagnostic)
        self.assertNotIn(leaked_prefix, diagnostic)
        self.assertIn("redacted", diagnostic)
        self.assertTrue(all(character.isprintable() for character in diagnostic))
        response.json.assert_not_called()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_json_only_compatibility_client_rejects_nonfinite_values(self, mock_get: Mock) -> None:
        for value in (float("nan"), np.float32("nan"), np.float32("inf"), np.float32("-inf")):
            mock_get.return_value = Mock(
                status_code=200,
                text="",
                json=Mock(
                    return_value={
                        "history": {
                            "day": {
                                "date": "2026-01-02",
                                "open": 10,
                                "high": 11,
                                "low": 9,
                                "close": 10.5,
                                "volume": 1000,
                            },
                            "ignored": {"value": [value]},
                        }
                    }
                ),
            )

            with self.subTest(value=value), self.assertRaises(MarketDataDownloadError) as raised:
                _load_tradier_symbol_frame(
                    "AAA",
                    None,
                    None,
                    TradierMarketDataConfig(access_token="token"),
                    auto_adjust=False,
                )

            self.assertIn("invalid JSON", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_json_only_response_client_uses_compatibility_fallback(
        self,
        mock_get: Mock,
    ) -> None:
        class JsonOnlyResponse:
            status_code = 200

            @staticmethod
            def json() -> object:
                return {
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1000,
                        }
                    }
                }

        mock_get.return_value = JsonOnlyResponse()

        out = _load_tradier_symbol_frame(
            "AAA",
            None,
            None,
            TradierMarketDataConfig(
                access_token="token",
                base_url="https://api.tradier.com",
            ),
            auto_adjust=False,
        )

        self.assertEqual(out.index.tolist(), [pd.Timestamp("2026-01-02")])
        self.assertEqual(out["AAA_Close"].tolist(), [10.5])
        self.assertEqual(mock_get.call_args.args[0], "https://api.tradier.com/v1/markets/history")

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_request_uses_one_validated_immutable_config_snapshot(
        self,
        mock_get: Mock,
    ) -> None:
        original_token = "original-token"
        cfg = TradierMarketDataConfig(
            access_token=original_token,
            base_url="https://api.tradier.com/v1",
            timeout_seconds=30,
        )
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 10,
                            "high": 11,
                            "low": 9,
                            "close": 10.5,
                            "volume": 1000,
                        }
                    }
                }
            ),
        )

        def mutate_caller_config(base_url: str) -> bool:
            cfg.access_token = "replacement-token"
            cfg.base_url = "https://attacker.example/v1"
            cfg.timeout_seconds = 600
            return base_url == "https://api.tradier.com/v1"

        with patch(
            "leveraged_trader.market_data.is_official_tradier_api_base_url",
            side_effect=mutate_caller_config,
        ):
            _load_tradier_symbol_frame("AAA", None, None, cfg, auto_adjust=False)

        self.assertEqual(mock_get.call_args.args[0], "https://api.tradier.com/v1/markets/history")
        self.assertEqual(mock_get.call_args.kwargs["headers"]["Authorization"], f"Bearer {original_token}")
        request_timeout = mock_get.call_args.kwargs["timeout"].total
        self.assertGreater(request_timeout, 29)
        self.assertLessEqual(request_timeout, 30)

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_malformed_configuration_types_before_request(
        self,
        mock_get: Mock,
    ) -> None:
        cases = (
            (TradierMarketDataConfig(enabled="false"), "enabled must be a boolean"),
            (TradierMarketDataConfig(access_token=True), "ACCESS_TOKEN must be a string"),
            (
                TradierMarketDataConfig(access_token="token", base_url=123),
                "BASE_URL must be a string",
            ),
            (TradierMarketDataConfig(timeout_seconds=True), "TIMEOUT_SECONDS must be an integer"),
            (TradierMarketDataConfig(timeout_seconds=0), "TIMEOUT_SECONDS must be an integer"),
            (TradierMarketDataConfig(timeout_seconds=601), "TIMEOUT_SECONDS must be an integer"),
        )
        for cfg, expected_message in cases:
            with self.subTest(expected_message=expected_message), self.assertRaises(MarketDataDownloadError) as raised:
                _load_tradier_symbol_frame("AAA", None, None, cfg, auto_adjust=False)

            self.assertIn(expected_message, str(raised.exception))

        mock_get.assert_not_called()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_rejects_unsafe_bearer_header_values_before_request(self, mock_get: Mock) -> None:
        for access_token in (
            "opaque token",
            "\nopaque",
            "opaque\n",
            "opaque\nheader",
            "opaque\tvalue",
            "opaque\x00value",
            "opaque\x1bvalue",
        ):
            with (
                self.subTest(access_token=repr(access_token)),
                self.assertRaises(MarketDataDownloadError) as raised,
            ):
                _load_tradier_symbol_frame(
                    "AAA",
                    None,
                    None,
                    TradierMarketDataConfig(access_token=access_token),
                    auto_adjust=False,
                )

            self.assertIn("must not contain whitespace or control characters", str(raised.exception))

        mock_get.assert_not_called()

    @patch("leveraged_trader.market_data.requests.get")
    def test_tradier_requires_each_daily_date_to_be_a_canonical_date_string(
        self,
        mock_get: Mock,
    ) -> None:
        invalid_dates: tuple[object, ...] = (
            "2026-1-05",
            "2026-01-05T00:00:00Z",
            " 2026-01-05",
            "2026-02-30",
            20260105,
            None,
            True,
        )

        for invalid_date in invalid_dates:
            with self.subTest(invalid_date=invalid_date):
                mock_get.return_value = Mock(
                    status_code=200,
                    text="",
                    json=Mock(
                        return_value={
                            "history": {
                                "day": [
                                    {
                                        "date": "2026-01-02",
                                        "open": 10,
                                        "high": 11,
                                        "low": 9,
                                        "close": 10.5,
                                        "volume": 1000,
                                    },
                                    {
                                        "date": invalid_date,
                                        "open": 10.5,
                                        "high": 12,
                                        "low": 10,
                                        "close": 11.5,
                                        "volume": 1100,
                                    },
                                ]
                            }
                        }
                    ),
                )

                with self.assertRaises(MarketDataDownloadError) as raised:
                    _load_tradier_symbol_frame(
                        "AAA",
                        None,
                        None,
                        TradierMarketDataConfig(access_token="token"),
                        auto_adjust=False,
                    )

                self.assertEqual(raised.exception.source, "Tradier")
                self.assertEqual(raised.exception.symbols, ["AAA"])
                self.assertIn("canonical YYYY-MM-DD string", str(raised.exception))
                self.assertIn("row 1", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_non_object_json_is_normalized_to_download_error(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        response = Mock(status_code=200)
        response.json.return_value = []
        mock_get.return_value = response

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("invalid top-level structure", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_mixed_daily_history_structure_is_rejected_in_full(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": [
                            {
                                "date": "2026-01-02",
                                "open": 10,
                                "high": 11,
                                "low": 9,
                                "close": 10.5,
                                "volume": 1000,
                            },
                            "malformed-day",
                            {
                                "date": "2026-01-06",
                                "open": 11,
                                "high": 12,
                                "low": 10,
                                "close": 11.5,
                                "volume": 1100,
                            },
                        ]
                    }
                }
            ),
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("daily history list containing a non-object entry", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_boolean_ohlcv_is_rejected_before_coercion(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": True,
                            "high": True,
                            "low": True,
                            "close": True,
                            "volume": False,
                        }
                    }
                }
            ),
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("not boolean or complex", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_temporal_ohlcv_is_rejected_instead_of_becoming_epoch_numeric(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        for label, temporal_value in (
            ("timestamp", pd.Timestamp("2026-01-02")),
            ("timedelta", pd.Timedelta(days=1)),
        ):
            with self.subTest(label=label):
                mock_get.return_value = Mock(
                    status_code=200,
                    text="",
                    json=Mock(
                        return_value={
                            "history": {
                                "day": {
                                    "date": "2026-01-02",
                                    "open": temporal_value,
                                    "high": 11,
                                    "low": 9,
                                    "close": 10.5,
                                    "volume": 1000,
                                }
                            }
                        }
                    ),
                )

                with self.assertRaises(MarketDataDownloadError) as raised:
                    load_market_data(
                        symbols=["AAA"],
                        auto_adjust=False,
                        tradier_cfg=TradierMarketDataConfig(access_token="token"),
                    )

                self.assertEqual(raised.exception.symbols, ["AAA"])
                self.assertIn("not a date, datetime, or timedelta", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_partial_ohlcv_row_is_rejected_instead_of_silently_dropped(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": [
                            {
                                "date": "2026-01-02",
                                "open": 10,
                                "high": 11,
                                "low": 9,
                                "close": 10.5,
                                "volume": 1000,
                            },
                            {
                                "date": "2026-01-05",
                                "open": 10.5,
                                "high": 12,
                                "low": 10,
                                "close": None,
                                "volume": 1100,
                            },
                        ]
                    }
                }
            ),
        )

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["AAA"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        self.assertEqual(raised.exception.symbols, ["AAA"])
        self.assertIn("2026-01-05", str(raised.exception))
        self.assertIn("Close must be positive and finite", str(raised.exception))

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_recovers_yfinance_failed_symbol(self, mock_download: Mock, mock_get: Mock) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        columns = pd.MultiIndex.from_product([["AAA"], ["Open", "High", "Low", "Close", "Volume"]])

        def fake_download(**_: object) -> pd.DataFrame:
            yf_shared._ERRORS = {"MISSING": "$MISSING: possibly delisted; no timezone found"}
            return pd.DataFrame(
                [
                    [10.0, 11.0, 9.0, 10.5, 1000],
                    [10.5, 12.0, 10.0, 11.5, 1100],
                ],
                index=index,
                columns=columns,
            )

        mock_download.side_effect = fake_download
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": [
                            {"date": "2026-01-02", "open": 20, "high": 21, "low": 19, "close": 20.5, "volume": 2000},
                            {"date": "2026-01-05", "open": 21, "high": 22, "low": 20, "close": 21.5, "volume": 2100},
                        ]
                    }
                }
            ),
        )

        data = load_market_data(
            symbols=["AAA", "MISSING"],
            auto_adjust=False,
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(
            list(data.columns),
            [
                "AAA_Open",
                "AAA_High",
                "AAA_Low",
                "AAA_Close",
                "AAA_Volume",
                "MISSING_Open",
                "MISSING_High",
                "MISSING_Low",
                "MISSING_Close",
                "MISSING_Volume",
            ],
        )
        self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["MISSING"])
        self.assertEqual(
            data.attrs[MARKET_DATA_PROVIDERS_ATTR],
            {"AAA": "yahoo_finance", "MISSING": "tradier"},
        )
        self.assertEqual(mock_get.call_args.kwargs["params"]["symbol"], "MISSING")
        self.assertEqual(mock_get.call_args.kwargs["headers"]["Authorization"], "Bearer token")

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_recovers_yfinance_non_overlapping_pair(self, mock_download: Mock, mock_get: Mock) -> None:
        index = pd.to_datetime(["2026-01-02", "2026-01-05"])
        fields = ["Open", "High", "Low", "Close", "Volume"]
        columns = pd.MultiIndex.from_product([["AAA", "BBB"], fields])
        mock_download.return_value = pd.DataFrame(
            [
                [10.0, 11.0, 9.0, 10.5, 1000, None, None, None, None, None],
                [None, None, None, None, None, 20.0, 21.0, 19.0, 20.5, 2000],
            ],
            index=index,
            columns=columns,
        )

        def fake_tradier_get(_url: str, **kwargs: object) -> Mock:
            params = kwargs["params"]
            self.assertIsInstance(params, dict)
            symbol = params["symbol"]
            open_price = 30 if symbol == "AAA" else 40
            return Mock(
                status_code=200,
                text="",
                json=Mock(
                    return_value={
                        "history": {
                            "day": {
                                "date": "2026-01-05",
                                "open": open_price,
                                "high": open_price + 1,
                                "low": open_price - 1,
                                "close": open_price + 0.5,
                                "volume": 3000,
                            }
                        }
                    }
                ),
            )

        mock_get.side_effect = fake_tradier_get

        data = load_market_data(
            symbols=["AAA", "BBB"],
            auto_adjust=False,
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(list(data.index), [pd.Timestamp("2026-01-05")])
        self.assertEqual(data.loc[pd.Timestamp("2026-01-05"), "AAA_Open"], 30)
        self.assertEqual(data.loc[pd.Timestamp("2026-01-05"), "BBB_Open"], 40)
        self.assertEqual(data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR], ["AAA", "BBB"])

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_symbol_uses_slash_for_class_shares(self, mock_download: Mock, mock_get: Mock) -> None:
        def fake_download(**_: object) -> pd.DataFrame:
            yf_shared._ERRORS = {"BRK-B": "$BRK-B: possibly delisted; no timezone found"}
            return pd.DataFrame()

        mock_download.side_effect = fake_download
        mock_get.return_value = Mock(
            status_code=200,
            text="",
            json=Mock(
                return_value={
                    "history": {
                        "day": {
                            "date": "2026-01-02",
                            "open": 20,
                            "high": 21,
                            "low": 19,
                            "close": 20.5,
                            "volume": 2000,
                        }
                    }
                }
            ),
        )

        load_market_data(
            symbols=["BRK-B"],
            auto_adjust=False,
            tradier_cfg=TradierMarketDataConfig(access_token="token"),
        )

        self.assertEqual(mock_get.call_args.kwargs["params"]["symbol"], "BRK/B")

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_partial_failure_reports_only_unresolved_symbols(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        def fake_download(**_: object) -> pd.DataFrame:
            yf_shared._ERRORS = {
                "PRICE": "$PRICE: possibly delisted; no timezone found",
                "CLOUD": "$CLOUD: possibly delisted; no timezone found",
            }
            return pd.DataFrame()

        def fake_tradier_get(_url: str, **kwargs: object) -> Mock:
            params = kwargs["params"]
            self.assertIsInstance(params, dict)
            symbol = params["symbol"]
            if symbol == "CLOUD":
                return Mock(
                    status_code=200,
                    text="",
                    json=Mock(
                        return_value={
                            "history": {
                                "day": {
                                    "date": "2026-01-02",
                                    "open": 20,
                                    "high": 21,
                                    "low": 19,
                                    "close": 20.5,
                                    "volume": 2000,
                                }
                            }
                        }
                    ),
                )
            return Mock(status_code=200, text="", json=Mock(return_value={"history": {"day": []}}))

        mock_download.side_effect = fake_download
        mock_get.side_effect = fake_tradier_get

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["PRICE", "CLOUD"],
                auto_adjust=False,
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        message = str(raised.exception)
        self.assertEqual(raised.exception.symbols, ["PRICE"])
        self.assertIn("Yahoo Finance and Tradier did not return usable daily data", message)
        self.assertIn("PRICE: Yahoo Finance: no timezone found", message)
        self.assertIn("Tradier fallback: No historical daily data returned", message)
        self.assertNotIn("$PRICE", message)
        self.assertNotIn("CLOUD:", message)

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_adjusted_yahoo_history_never_mixes_with_tradier_history(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        def fake_download(**_: object) -> pd.DataFrame:
            yf_shared._ERRORS = {"MISSING": "$MISSING: no data returned"}
            return pd.DataFrame()

        mock_download.side_effect = fake_download

        with self.assertRaises(MarketDataDownloadError) as raised:
            load_market_data(
                symbols=["MISSING"],
                auto_adjust=True,
                tradier_cfg=TradierMarketDataConfig(access_token="token"),
            )

        self.assertIn("cannot be used when auto_adjust=True", str(raised.exception))
        mock_get.assert_not_called()

    @patch("leveraged_trader.market_data.requests.get")
    @patch("leveraged_trader.market_data.yf.download")
    def test_tradier_bearer_token_is_not_sent_to_custom_or_insecure_endpoint(
        self,
        mock_download: Mock,
        mock_get: Mock,
    ) -> None:
        mock_download.return_value = pd.DataFrame()

        for base_url in [
            "http://api.tradier.com/v1",
            "https://attacker.test/v1",
            "https://@api.tradier.com/v1",
            "https://api.tradier.com:/v1",
            "https://api.tradier.com/v1?",
            "https://api.tradier.com/v1#",
        ]:
            with self.subTest(base_url=base_url), self.assertRaises(MarketDataDownloadError) as raised:
                load_market_data(
                    symbols=["MISSING"],
                    auto_adjust=False,
                    tradier_cfg=TradierMarketDataConfig(access_token="secret", base_url=base_url),
                )
            self.assertIn("HTTPS Tradier API endpoint", str(raised.exception))

        mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
