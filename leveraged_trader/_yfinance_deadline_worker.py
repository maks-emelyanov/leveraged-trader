from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import yfinance.multi as yf_multi
import yfinance.shared as yf_shared

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None  # type: ignore[assignment]

from ._http_deadline_worker import read_worker_request, run_deadline_subprocess, write_worker_result

_YFINANCE_WORKER_COMMAND = (
    sys.executable,
    "-I",
    "-m",
    "leveraged_trader._yfinance_deadline_worker",
)
YFINANCE_WORKER_RESULT_MAX_BYTES = 128 * 1024 * 1024
_YFINANCE_WORKER_TEXT_MAX_CHARS = 4096
_YFINANCE_WORKER_FRAME_MAX_MEMORY_BYTES = 64 * 1024 * 1024
_YFINANCE_WORKER_FRAME_MAX_ROWS = 100_000
_YFINANCE_WORKER_FRAME_MAX_COLUMNS = 4_096
_YFINANCE_WORKER_FRAME_MAX_CELLS = 8 * 1024 * 1024
_YFINANCE_WORKER_METADATA_MAX_ENTRIES = 10_000
_YFINANCE_WORKER_METADATA_MAX_BYTES = 4 * 1024 * 1024
_YFINANCE_WORKER_ADDRESS_SPACE_HEADROOM_BYTES = 768 * 1024 * 1024
_YFINANCE_WORKER_THREAD_STACK_BYTES = 1024 * 1024
_DARWIN_PROC_PIDTASKINFO = 4
_WINDOWS_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_WINDOWS_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_YFINANCE_WORKER_WINDOWS_JOB_HANDLE: object | None = None


class _DarwinProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


class _WindowsIoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _WindowsJobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _WindowsJobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _WindowsJobObjectBasicLimitInformation),
        ("IoInfo", _WindowsIoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def exception_from_yfinance_worker_error(payload: object) -> BaseException | None:
    """Decode an exact, bounded worker error tuple or reject malformed lookalikes."""
    if type(payload) is not tuple or not payload or type(payload[0]) is not str or payload[0] != "error":
        return None
    if (
        len(payload) != 3
        or type(payload[1]) is not str
        or type(payload[2]) is not str
        or not payload[1]
        or len(payload[1]) > _YFINANCE_WORKER_TEXT_MAX_CHARS
        or len(payload[2]) > _YFINANCE_WORKER_TEXT_MAX_CHARS
    ):
        raise requests.exceptions.ConnectionError("Yahoo Finance worker returned an invalid response.")
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
    }.get(payload[1], requests.exceptions.RequestException)
    return exception_type(payload[2])


def _bounded_text(value: object) -> str:
    try:
        rendered = str(value)
    except BaseException:
        rendered = f"<{type(value).__name__} could not be rendered>"
    return rendered[:_YFINANCE_WORKER_TEXT_MAX_CHARS]


def _linux_virtual_memory_bytes() -> int:
    """Return this process's current virtual size without allocating a helper process."""
    try:
        with open("/proc/self/statm", encoding="ascii") as status_file:
            page_count_text = status_file.read(64).split(maxsplit=1)[0]
        page_count = int(page_count_text)
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError("Yahoo Finance worker could not determine its current memory footprint.") from exc
    if page_count < 1 or page_size < 1:
        raise RuntimeError("Yahoo Finance worker reported an invalid current memory footprint.")
    return page_count * page_size


def _darwin_virtual_memory_bytes() -> int:
    """Read Darwin's current virtual size from proc_taskinfo."""
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        task_info = _DarwinProcTaskInfo()
        task_info_size = ctypes.sizeof(task_info)
        returned_size = proc_pidinfo(
            os.getpid(),
            _DARWIN_PROC_PIDTASKINFO,
            0,
            ctypes.byref(task_info),
            task_info_size,
        )
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError("Yahoo Finance worker could not determine its current memory footprint.") from exc
    if returned_size != task_info_size or task_info.pti_virtual_size < 1:
        raise RuntimeError("Yahoo Finance worker could not determine its current memory footprint.")
    return int(task_info.pti_virtual_size)


def _apply_posix_address_space_limit(current_virtual_bytes: int) -> None:
    if resource is None or not hasattr(resource, "RLIMIT_AS"):
        raise RuntimeError("Yahoo Finance worker cannot enforce its memory limit on this host.")
    if current_virtual_bytes < 1:
        raise RuntimeError("Yahoo Finance worker reported an invalid current memory footprint.")

    desired_soft_limit = current_virtual_bytes + _YFINANCE_WORKER_ADDRESS_SPACE_HEADROOM_BYTES
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    infinity = resource.RLIM_INFINITY

    # Respect a pre-existing tighter process/cgroup policy. Otherwise lower
    # only the soft limit so the worker can never allocate unbounded memory,
    # while leaving the inherited hard limit unchanged.
    if soft_limit != infinity and soft_limit <= desired_soft_limit:
        return
    if hard_limit != infinity:
        desired_soft_limit = min(desired_soft_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (desired_soft_limit, hard_limit))


def _windows_last_error_code() -> int:
    get_last_error = getattr(ctypes, "get_last_error", None)
    if not callable(get_last_error):
        return 0
    return int(get_last_error())


def _windows_limit_failure(action: str) -> RuntimeError:
    error_code = _windows_last_error_code()
    suffix = f" (Windows error {error_code})" if error_code else ""
    return RuntimeError(f"Yahoo Finance worker could not {action}{suffix}.")


def _windows_process_private_memory_bytes(kernel32: object, psapi: object) -> int:
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    get_process_memory_info.restype = ctypes.c_int

    process = get_current_process()
    counters = _WindowsProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    if not process or not get_process_memory_info(process, ctypes.byref(counters), counters.cb):
        raise _windows_limit_failure("determine its current memory footprint")
    if counters.PrivateUsage < 1:
        raise RuntimeError("Yahoo Finance worker reported an invalid current memory footprint.")
    return int(counters.PrivateUsage)


def _apply_windows_process_memory_limit() -> None:
    """Retain a Job Object that caps this worker's committed process memory."""
    global _YFINANCE_WORKER_WINDOWS_JOB_HANDLE

    if _YFINANCE_WORKER_WINDOWS_JOB_HANDLE is not None:
        return
    win_dll = getattr(ctypes, "WinDLL", None)
    if not callable(win_dll):
        raise RuntimeError("Yahoo Finance worker cannot enforce its memory limit on this host.")
    try:
        kernel32 = win_dll("kernel32", use_last_error=True)
        psapi = win_dll("psapi", use_last_error=True)
        current_private_bytes = _windows_process_private_memory_bytes(kernel32, psapi)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError("Yahoo Finance worker cannot enforce its memory limit on this host.") from exc

    maximum_size = ctypes.c_size_t(-1).value
    if current_private_bytes > maximum_size - _YFINANCE_WORKER_ADDRESS_SPACE_HEADROOM_BYTES:
        raise RuntimeError("Yahoo Finance worker memory limit exceeds this host's addressable range.")
    process_memory_limit = current_private_bytes + _YFINANCE_WORKER_ADDRESS_SPACE_HEADROOM_BYTES

    create_job = kernel32.CreateJobObjectW
    create_job.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    create_job.restype = ctypes.c_void_p
    set_job_information = kernel32.SetInformationJobObject
    set_job_information.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    set_job_information.restype = ctypes.c_int
    assign_process = kernel32.AssignProcessToJobObject
    assign_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    assign_process.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int

    job_handle = create_job(None, None)
    if not job_handle:
        raise _windows_limit_failure("create its memory-limit Job Object")
    try:
        limits = _WindowsJobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _WINDOWS_JOB_OBJECT_LIMIT_PROCESS_MEMORY
        limits.ProcessMemoryLimit = process_memory_limit
        if not set_job_information(
            job_handle,
            _WINDOWS_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise _windows_limit_failure("configure its memory-limit Job Object")
        process = kernel32.GetCurrentProcess()
        if not process or not assign_process(job_handle, process):
            raise _windows_limit_failure("enter its memory-limit Job Object")
    except BaseException:
        close_handle(job_handle)
        raise
    _YFINANCE_WORKER_WINDOWS_JOB_HANDLE = job_handle


def _apply_yfinance_worker_memory_limit() -> None:
    """Bound provider allocations after trusted imports and before any network call."""
    if sys.platform.startswith("linux"):
        _apply_posix_address_space_limit(_linux_virtual_memory_bytes())
        return
    if sys.platform == "darwin":
        _apply_posix_address_space_limit(_darwin_virtual_memory_bytes())
        return
    if sys.platform == "win32":
        _apply_windows_process_memory_limit()
        return
    raise RuntimeError("Yahoo Finance worker cannot enforce its memory limit on this host.")


def _string_mapping(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Yahoo Finance returned invalid {label} metadata.")
    if len(value) > _YFINANCE_WORKER_METADATA_MAX_ENTRIES:
        raise ValueError(f"Yahoo Finance {label} metadata exceeded its entry limit.")
    normalized: dict[str, str] = {}
    encoded_size = 0
    for key, item in value.items():
        normalized_key = _bounded_text(key)
        normalized_item = _bounded_text(item)
        encoded_size += len(normalized_key.encode("utf-8")) + len(normalized_item.encode("utf-8"))
        if encoded_size > _YFINANCE_WORKER_METADATA_MAX_BYTES:
            raise ValueError(f"Yahoo Finance {label} metadata exceeded its byte limit.")
        normalized[normalized_key] = normalized_item
    return normalized


_TRANSPORT_SCALAR_TYPES = {
    str,
    bytes,
    bool,
    int,
    float,
    complex,
    date,
    datetime,
    time,
    timedelta,
    Decimal,
    pd.Timestamp,
    pd.Timedelta,
    pd.Period,
    pd.Interval,
}


def _transport_scalar(value: object) -> bool:
    """Return whether one value has no nested provider-controlled graph."""
    if value is None or value is pd.NA or value is pd.NaT or type(value) in _TRANSPORT_SCALAR_TYPES:
        return True
    return isinstance(value, np.generic) and not value.dtype.hasobject


def _transport_axis_name_bytes(axis: pd.Index) -> int:
    """Validate names omitted by Index.memory_usage and return their size."""
    names = axis.names if isinstance(axis, pd.MultiIndex) else (axis.name,)
    total = 0
    for name in names:
        if not _transport_scalar(name):
            raise ValueError("Yahoo Finance frame contained a non-scalar axis name.")
        total += sys.getsizeof(name)
    return total


def _transport_axis_values_are_safe(axis: pd.Index) -> bool:
    """Allow tuples only when pandas represents them as real MultiIndex rows."""
    if isinstance(axis, pd.MultiIndex):
        return all(
            type(value) is tuple
            and len(value) == axis.nlevels
            and all(_transport_scalar(level_value) for level_value in value)
            for value in axis
        )
    return all(_transport_scalar(value) for value in axis)


def _validate_yfinance_frame_for_transport(frame: pd.DataFrame, *, label: str) -> None:
    """Reject a provider frame before pickle can duplicate excessive data."""
    if type(frame) is not pd.DataFrame:
        raise TypeError(f"Yahoo Finance {label} must be an exact DataFrame.")
    row_count, column_count = frame.shape
    if (
        row_count > _YFINANCE_WORKER_FRAME_MAX_ROWS
        or column_count > _YFINANCE_WORKER_FRAME_MAX_COLUMNS
        or row_count * column_count > _YFINANCE_WORKER_FRAME_MAX_CELLS
    ):
        raise ValueError(f"Yahoo Finance {label} exceeded its safe transport shape limit.")

    # DataFrame.attrs is not used by either consumer and can contain arbitrary
    # provider-controlled object graphs that memory_usage() does not count.
    frame.attrs.clear()

    if not _transport_axis_values_are_safe(frame.index):
        raise ValueError(f"Yahoo Finance {label} contained a non-scalar index value.")
    if not _transport_axis_values_are_safe(frame.columns):
        raise ValueError(f"Yahoo Finance {label} contained a non-scalar column label.")
    for position, dtype in enumerate(frame.dtypes):
        fixed_width_dtype = isinstance(dtype, np.dtype) and not dtype.hasobject
        if isinstance(dtype, pd.CategoricalDtype):
            if not _transport_axis_values_are_safe(dtype.categories):
                raise ValueError(f"Yahoo Finance {label} contained a non-scalar categorical value.")
        elif not fixed_width_dtype and any(not _transport_scalar(value) for value in frame.iloc[:, position].array):
            raise ValueError(f"Yahoo Finance {label} contained a non-scalar object value.")

    try:
        memory_parts = frame.memory_usage(index=True, deep=True)
        memory_bytes = sum(int(value) for value in memory_parts.array)
        memory_bytes += int(frame.columns.memory_usage(deep=True))
        memory_bytes += _transport_axis_name_bytes(frame.index)
        memory_bytes += _transport_axis_name_bytes(frame.columns)
        memory_bytes += sum(
            _transport_axis_name_bytes(dtype.categories)
            for dtype in frame.dtypes
            if isinstance(dtype, pd.CategoricalDtype)
        )
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise ValueError(f"Yahoo Finance {label} memory usage could not be bounded.") from exc
    if memory_bytes < 0 or memory_bytes > _YFINANCE_WORKER_FRAME_MAX_MEMORY_BYTES:
        raise ValueError(
            f"Yahoo Finance {label} exceeded its {_YFINANCE_WORKER_FRAME_MAX_MEMORY_BYTES}-byte memory limit."
        )


def execute_yfinance_download(
    symbols: list[str],
    start: str | None,
    end: str | None,
    auto_adjust: bool,
    *,
    request_timeout_seconds: float,
) -> tuple[pd.DataFrame | None, dict[str, str], dict[str, str]]:
    """Execute one yfinance call and retain its per-call errors and aliases."""
    # yfinance creates one Python thread per ticker even though its semaphore
    # limits the number concurrently doing network work.  The platform default
    # stack (8 MiB on this WSL host) plus glibc arenas can exhaust the worker's
    # deliberately bounded address space before a 32-symbol batch starts.  A
    # 1 MiB stack is ample for these shallow Python/HTTP call stacks and keeps
    # the complete deterministic batch inside the existing memory boundary.
    if len(symbols) > 1:
        threading.stack_size(_YFINANCE_WORKER_THREAD_STACK_BYTES)
    download_kwargs = {
        "tickers": symbols,
        "start": start,
        "end": end,
        # Supplying only ``end`` with ``period=None`` makes yfinance derive
        # a one-month start boundary. Canonical symbol-history callers use an
        # omitted start to mean complete history, so retain the maximum period.
        "period": "max" if start is None else None,
        "interval": "1d",
        "auto_adjust": auto_adjust,
        "group_by": "ticker",
        "progress": False,
        "threads": True,
        "timeout": request_timeout_seconds,
    }
    yf_shared._ERRORS = {}
    symbol_aliases: Mapping[object, object] = {}
    if yf.download is yf_multi.download and hasattr(yf_multi, "_DownloadCtx") and hasattr(yf_multi, "_download_impl"):
        # yfinance 1.5 keeps failures in a per-download context and no longer
        # updates shared._ERRORS. Construct that context so diagnostics remain
        # available after the call. The identity check retains compatibility
        # with older yfinance versions and injected public download functions.
        download_context = yf_multi._DownloadCtx()
        raw = yf_multi._download_impl(download_context, **download_kwargs)
        download_errors: Mapping[object, object] = download_context.errors
        symbol_aliases = download_context.isins
    else:
        raw = yf.download(**download_kwargs)
        download_errors = yf_shared._ERRORS

    if raw is not None and not isinstance(raw, pd.DataFrame):
        raise TypeError("Yahoo Finance returned an invalid market-data frame.")
    return (
        raw,
        _string_mapping(download_errors, label="error"),
        _string_mapping(symbol_aliases, label="symbol-alias"),
    )


def execute_yfinance_ticker_history(
    symbol: str,
    *,
    request_timeout_seconds: float,
) -> pd.DataFrame:
    """Fetch the small recent-history frame used for Alpaca buy pricing."""
    history = yf.Ticker(symbol).history(
        period="5d",
        interval="1d",
        auto_adjust=True,
        timeout=request_timeout_seconds,
        raise_errors=True,
    )
    if not isinstance(history, pd.DataFrame):
        raise TypeError("Yahoo Finance returned an invalid recent-history frame.")
    return history


def _valid_worker_symbol(value: object) -> bool:
    return (
        type(value) is str
        and bool(value)
        and value == value.strip()
        and all(character.isprintable() and not character.isspace() for character in value)
    )


def _yfinance_download_payload(request: object) -> tuple[object, ...]:
    if not isinstance(request, dict) or set(request) != {
        "auto_adjust",
        "end",
        "mode",
        "request_timeout_seconds",
        "start",
        "symbols",
    }:
        raise TypeError("Yahoo Finance worker received an invalid request.")
    symbols = request["symbols"]
    start = request["start"]
    end = request["end"]
    auto_adjust = request["auto_adjust"]
    request_timeout_seconds = request["request_timeout_seconds"]
    if (
        request["mode"] != "yfinance"
        or type(symbols) is not list
        or not symbols
        or any(not _valid_worker_symbol(symbol) for symbol in symbols)
        or (start is not None and type(start) is not str)
        or (end is not None and type(end) is not str)
        or type(auto_adjust) is not bool
        or type(request_timeout_seconds) not in {int, float}
        or not math.isfinite(request_timeout_seconds)
        or not 0 < request_timeout_seconds <= 600
    ):
        raise TypeError("Yahoo Finance worker received an invalid request.")

    raw, download_errors, symbol_aliases = execute_yfinance_download(
        symbols,
        start,
        end,
        auto_adjust,
        request_timeout_seconds=float(request_timeout_seconds),
    )
    if raw is not None:
        _validate_yfinance_frame_for_transport(raw, label="market-data frame")
    return ("yfinance_response", raw, download_errors, symbol_aliases)


def _yfinance_ticker_history_payload(request: object) -> tuple[object, ...]:
    if not isinstance(request, dict) or set(request) != {
        "mode",
        "request_timeout_seconds",
        "symbol",
    }:
        raise TypeError("Yahoo Finance worker received an invalid request.")
    symbol = request["symbol"]
    request_timeout_seconds = request["request_timeout_seconds"]
    if (
        request["mode"] != "yfinance_ticker_history"
        or not _valid_worker_symbol(symbol)
        or type(request_timeout_seconds) not in {int, float}
        or not math.isfinite(request_timeout_seconds)
        or not 0 < request_timeout_seconds <= 600
    ):
        raise TypeError("Yahoo Finance worker received an invalid request.")
    history = execute_yfinance_ticker_history(
        symbol,
        request_timeout_seconds=float(request_timeout_seconds),
    )
    _validate_yfinance_frame_for_transport(history, label="recent-history frame")
    return ("yfinance_ticker_history_response", history)


def run_yfinance_download_with_deadline(
    symbols: Sequence[str],
    start: str | None,
    end: str | None,
    auto_adjust: bool,
    *,
    request_timeout_seconds: float,
    deadline: float,
) -> object:
    """Run yfinance in a killable process under one absolute deadline."""
    return run_deadline_subprocess(
        _YFINANCE_WORKER_COMMAND,
        {
            "auto_adjust": auto_adjust,
            "end": end,
            "mode": "yfinance",
            "request_timeout_seconds": request_timeout_seconds,
            "start": start,
            "symbols": list(symbols),
        },
        deadline=deadline,
        result_max_bytes=YFINANCE_WORKER_RESULT_MAX_BYTES,
        timeout_message="Yahoo Finance response exceeded its overall deadline.",
        connection_error_message="Yahoo Finance worker exited without a complete response.",
        # glibc otherwise reserves a separate large malloc arena for each of
        # yfinance's per-ticker threads.  The fixed arena count keeps a
        # 32-symbol batch inside the worker's address-space limit without
        # increasing WSL's memory allocation.  Non-glibc platforms ignore the
        # variable, so it is safe to pass to the isolated worker everywhere.
        environment_overrides={"MALLOC_ARENA_MAX": "2"},
    )


def run_yfinance_ticker_history_with_deadline(
    symbol: str,
    *,
    request_timeout_seconds: float,
    deadline: float,
) -> object:
    """Fetch recent ticker history in a killable process under one deadline."""
    return run_deadline_subprocess(
        _YFINANCE_WORKER_COMMAND,
        {
            "mode": "yfinance_ticker_history",
            "request_timeout_seconds": request_timeout_seconds,
            "symbol": symbol,
        },
        deadline=deadline,
        result_max_bytes=YFINANCE_WORKER_RESULT_MAX_BYTES,
        timeout_message="Yahoo Finance response exceeded its overall deadline.",
        connection_error_message="Yahoo Finance worker exited without a complete response.",
    )


def main(*, enforce_memory_limit: bool = False) -> None:
    try:
        if enforce_memory_limit:
            _apply_yfinance_worker_memory_limit()
        request = read_worker_request()
        if isinstance(request, dict) and request.get("mode") == "yfinance_ticker_history":
            payload = _yfinance_ticker_history_payload(request)
        else:
            payload = _yfinance_download_payload(request)
    except BaseException as exc:
        payload = ("error", type(exc).__name__, _bounded_text(exc))
    write_worker_result(payload, max_bytes=YFINANCE_WORKER_RESULT_MAX_BYTES)


if __name__ == "__main__":
    main(enforce_memory_limit=True)
