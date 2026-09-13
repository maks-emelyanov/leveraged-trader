from __future__ import annotations

import asyncio
import ctypes
import hashlib
import io
import math
import os
import sqlite3
import stat
import threading
import time
import uuid
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager, suppress
from contextvars import ContextVar, copy_context
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from functools import partial
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, TextIO
from unittest.mock import Mock

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on native Windows
    fcntl = None
    import msvcrt

import numpy as np
import pandas as pd

if os.name == "posix":
    import pwd

from .alpaca import (
    BUY_RESULTS_REQUIRING_RECONCILIATION,
    AlpacaBuyBatchError,
    AlpacaReconciliationError,
    _alpaca_config_sensitive_values,
    _alpaca_exception_diagnostic,
    _alpaca_public_durable_diagnostic_scope,
    _alpaca_public_exception,
    _alpaca_public_exception_cause,
    migrate_alpaca_managed_position_symbols,
    reconcile_alpaca_managed_positions,
    submit_alpaca_paper_buy_orders,
)
from .benchmark import WorkflowPhase, WorkflowPhaseTimings, WorkflowTimer
from .config import (
    RISK_FREE_SYMBOL,
    AlpacaOrderConfig,
    BacktestConfig,
    TradierMarketDataConfig,
    UniverseConfig,
    validate_alpaca_paper_endpoint,
    validate_alpaca_reconciliation_configuration,
    validate_runtime_configuration,
)
from .market_data import (
    MARKET_DATA_PROVIDERS_ATTR,
    TRADIER_RECOVERED_SYMBOLS_ATTR,
    MarketDataDownloadError,
    load_risk_free_history,
    load_signal_history,
    load_symbol_history,
    load_symbol_history_batch,
    recover_signal_history_for_calendar,
    signal_history_overlaps_calendar,
)
from .output import AssetProgress, WorkflowReporter
from .reports import (
    build_alpaca_realized_pnl_summary,
    build_buy_signal_report,
    build_sell_signal_report,
    summarize_saved_results,
)
from .runtime_files import (
    PrivateRuntimeDirectory,
    SqliteRuntimeGuard,
    activate_sqlite_runtime_file,
    open_owned_runtime_directory,
    preflight_owned_runtime_directory,
    prepare_owned_runtime_directory,
    prepare_private_runtime_file,
    require_owned_runtime_directory,
    revalidate_active_sqlite_runtime_file,
    revalidate_owned_runtime_directory,
    revalidate_private_runtime_file,
)
from .storage import (
    AssetMarketDataError,
    _DeferredCommitSqliteConnection,
    _safe_market_history_tail_rows,
    active_alpaca_managed_symbols,
    init_state_db,
    load_alpaca_managed_positions,
    load_best_strategy_summary,
    load_complete_strategy_equity_curve,
    process_asset_grid,
    save_workflow_assets,
    strategy_config_fingerprint,
    strategy_config_matches_fingerprint,
    strategy_state_generation,
    strategy_state_matches_config,
)
from .universe import determine_workflow_asset_groups

DEFAULT_WORKFLOW_CONCURRENCY = 4
MARKET_DATA_BATCH_SIZE = 32
SQLITE_BUSY_TIMEOUT_MS = 60_000
_MONOTONIC_CLOCK = time.monotonic
_WALL_CLOCK = time.time
LONG_WORKFLOW_LABEL = "Long"
SHORT_WORKFLOW_LABEL = "Short"
DEFAULT_SHORT_BUY_RSI_VALUES = list(range(50, 81))
_ACTIVE_OUTPUT_DIRECTORY: ContextVar[PrivateRuntimeDirectory | None] = ContextVar(
    "active_workflow_output_directory",
    default=None,
)
_CSV_FORMULA_PREFIXES = frozenset("=+-@\t\r\n")
_CSV_FORMULA_LEADING_WHITESPACE = " \t\r\n"
_RESEARCH_REPORT_FILENAMES = (
    "best_equity_curves.csv",
    "optimization_summary.csv",
    "buy_signals.csv",
    "eligible_buy_signals.csv",
    "sell_signals.csv",
)
_ALPACA_RECONCILIATION_SNAPSHOT_FILENAMES = (
    "alpaca_reconciliation_results.csv",
    "alpaca_sell_order_results.csv",
    "managed_positions.csv",
    "alpaca_realized_pnl.csv",
)
_ALPACA_WORKFLOW_SNAPSHOT_FILENAMES = (
    "alpaca_order_results.csv",
    *_ALPACA_RECONCILIATION_SNAPSHOT_FILENAMES,
)
_ALPACA_CURRENT_STATE_SNAPSHOT_FILENAMES = (
    "managed_positions.csv",
    "alpaca_realized_pnl.csv",
)
_ALPACA_SNAPSHOT_MANIFEST_FILENAME = "alpaca_snapshot_manifest.csv"
_ALPACA_ACCOUNT_LOCK_FILENAME = ".leveraged-trader-alpaca-paper-account.lock"
_SQLITE_RUNTIME_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


class _WindowsGuid(ctypes.Structure):
    _fields_ = (
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    )


class _WindowsFileCaseSensitiveInformation(ctypes.Structure):
    _fields_ = (("flags", ctypes.c_uint32),)


# FOLDERID_LocalAppData. Unlike Path.home()/expanduser(), the Known Folder API
# resolves the current Windows user's profile from OS account metadata instead
# of trusting USERPROFILE/HOME supplied by the calling environment.
_WINDOWS_LOCAL_APP_DATA_FOLDER_ID = _WindowsGuid(
    0xF1B32785,
    0x6FBA,
    0x4FCF,
    (ctypes.c_ubyte * 8)(0x9D, 0x55, 0x7B, 0x8E, 0x7F, 0x15, 0x70, 0x91),
)
_WINDOWS_COINIT_MULTITHREADED = 0x0
_WINDOWS_S_OK = 0x00000000
_WINDOWS_S_FALSE = 0x00000001
_WINDOWS_RPC_E_CHANGED_MODE = 0x80010106
_WINDOWS_FILE_READ_ATTRIBUTES = 0x0080
_WINDOWS_FILE_SHARE_READ = 0x0001
_WINDOWS_FILE_SHARE_WRITE = 0x0002
_WINDOWS_FILE_SHARE_DELETE = 0x0004
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_WINDOWS_FILE_CASE_SENSITIVE_INFO = 23
_WINDOWS_FILE_CS_FLAG_CASE_SENSITIVE_DIR = 0x00000001
_WINDOWS_MOVEFILE_REPLACE_EXISTING = 0x00000001
_WINDOWS_MOVEFILE_WRITE_THROUGH = 0x00000008
_ALPACA_SNAPSHOT_SCHEMA_VERSION = "1"
_REPORT_CLEANUP_PREFIX = ".leveraged-trader-report-cleanup-"
_ALPACA_SNAPSHOT_MANIFEST_COLUMNS = (
    "Schema Version",
    "Generation",
    "Status",
    "Snapshot Kind",
    "Filename",
    "SHA256",
)
_ALPACA_SNAPSHOT_FILENAMES_BY_KIND = {
    "reconciliation": _ALPACA_RECONCILIATION_SNAPSHOT_FILENAMES,
    "workflow": _ALPACA_WORKFLOW_SNAPSHOT_FILENAMES,
    "current-state": _ALPACA_CURRENT_STATE_SNAPSHOT_FILENAMES,
}
_MANAGED_OUTPUT_FILENAMES = (
    _RESEARCH_REPORT_FILENAMES + _ALPACA_WORKFLOW_SNAPSHOT_FILENAMES + (_ALPACA_SNAPSHOT_MANIFEST_FILENAME,)
)


@dataclass
class _AlpacaSnapshotPublication:
    output_path: Path
    generation: str
    snapshot_kind: str
    expected_filenames: tuple[str, ...]
    digests: dict[str, str]


@dataclass(frozen=True)
class AlpacaSnapshot:
    """One verified Alpaca CSV generation copied into immutable memory."""

    generation: str
    snapshot_kind: str
    files: Mapping[str, bytes]

    @property
    def filenames(self) -> tuple[str, ...]:
        """Return the manifest-ordered filenames covered by this generation."""
        return tuple(self.files)

    def read_csv(self, filename: str, **kwargs: Any) -> pd.DataFrame:
        """Parse one covered CSV without reopening its replaceable canonical path."""
        try:
            contents = self.files[filename]
        except KeyError as exc:
            raise KeyError(f"Alpaca snapshot does not cover {filename!r}.") from exc
        return pd.read_csv(io.BytesIO(contents), **kwargs)


class _LockedWorkflowOutputPath(str):
    """Canonical output path carrying the database identity pinned by the lock."""

    database_path: str

    def __new__(cls, output_path: str, *, database_path: str) -> _LockedWorkflowOutputPath:
        value = super().__new__(cls, output_path)
        value.database_path = database_path
        return value


def _unwrap_zero_dimensional_scalar(value: object) -> object:
    while isinstance(value, np.ndarray) and value.ndim == 0:
        # Preserve datetime64/timedelta64 semantics; ``item()`` can turn
        # nanosecond values into otherwise-valid Python integers.
        value = value[()]
    return value


def _validate_grid_values(name: str, values: list[float]) -> list[float]:
    if not values:
        raise ValueError(f"{name} must not be empty.")
    numeric_values = []
    for value in values:
        value = _unwrap_zero_dimensional_scalar(value)
        if isinstance(
            value,
            (
                bool,
                np.bool_,
                complex,
                np.complexfloating,
                date,
                timedelta,
                np.datetime64,
                np.timedelta64,
                np.ndarray,
            ),
        ):
            raise ValueError(f"{name} must contain only finite numeric values.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain only finite numeric values.") from exc
        if not math.isfinite(numeric_value):
            raise ValueError(f"{name} must contain only finite numeric values.")
        numeric_values.append(numeric_value)
    return numeric_values


def _validate_optimization_grids(
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> None:
    numeric_buy_rsi_values = _validate_grid_values("buy_rsi_values", buy_rsi_values)
    numeric_profit_target_values = _validate_grid_values(
        "profit_target_values",
        profit_target_values,
    )
    if any(not 0.0 <= value <= 100.0 for value in numeric_buy_rsi_values):
        raise ValueError("buy_rsi_values must be between 0 and 100 inclusive.")
    if any(not 1.0 < value <= 100.0 for value in numeric_profit_target_values):
        raise ValueError("profit_target_values must be greater than 1.0 and at most 100.0.")


def _validate_workflow_mode(mode: str) -> None:
    if mode not in {"update", "rebuild"}:
        raise ValueError(f"mode must be 'update' or 'rebuild'; got {mode!r}.")


def _validate_windows_path_component_aliases(path: str, *, label: str) -> None:
    """Reject Win32 names that alias after trailing-dot/space normalization."""
    if os.name != "nt":
        return

    windows_path = PureWindowsPath(path)
    for component in windows_path.parts:
        if component in {windows_path.anchor, windows_path.drive, windows_path.root, ".", ".."}:
            continue
        if component.endswith((".", " ")):
            raise ValueError(
                f"{label} must not contain a Windows path component ending in a space or period: {component!r}."
            )


def _validate_database_path(db_path: str) -> None:
    if not isinstance(db_path, str) or not db_path.strip():
        raise ValueError("db_path must be a nonempty filesystem path.")
    if any(not character.isprintable() for character in db_path):
        raise ValueError("db_path must not contain control or nonprintable characters.")
    if db_path == ":memory:":
        raise ValueError("db_path must be a persistent filesystem path; ':memory:' is not supported.")
    _validate_windows_path_component_aliases(db_path, label="db_path")

    path_separators = tuple(separator for separator in (os.sep, os.altsep) if separator is not None)
    if db_path.endswith(path_separators):
        raise ValueError("db_path must name a database file and must not end with a path separator.")
    if os.path.basename(db_path) in {os.curdir, os.pardir}:
        raise ValueError("db_path must name a database file and must not end with a '.' or '..' path component.")

    try:
        supplied_status = os.lstat(db_path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"db_path must be an accessible filesystem path: {db_path}.") from exc

    if stat.S_ISLNK(supplied_status.st_mode):
        try:
            supplied_status = os.stat(db_path)
        except OSError as exc:
            raise ValueError(f"db_path symbolic link must resolve to a regular file: {db_path}.") from exc
    if not stat.S_ISREG(supplied_status.st_mode):
        raise ValueError(f"db_path must name a regular file when it already exists: {db_path}.")


def _validate_output_directory_path(output_dir: str) -> None:
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a nonempty filesystem path.")
    if any(not character.isprintable() for character in output_dir):
        raise ValueError("output_dir must not contain control or nonprintable characters.")
    _validate_windows_path_component_aliases(output_dir, label="output_dir")


def _absolute_workflow_path_preserving_traversal(path: str | os.PathLike[str]) -> str:
    """Make a path absolute without resolving ``..`` ahead of symlinks."""
    path_value = os.fspath(path)
    return path_value if os.path.isabs(path_value) else os.path.join(os.getcwd(), path_value)


def _normalized_workflow_path(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path), strict=False))


def _case_preserving_workflow_path(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.fspath(path), strict=False)


def _case_preserving_workflow_alias_path(path: str | os.PathLike[str]) -> str:
    """Resolve a supplied path's parent while retaining its final alias name."""
    absolute_path = _absolute_workflow_path_preserving_traversal(path)
    parent_path, filename = os.path.split(absolute_path)
    return os.path.join(os.path.realpath(parent_path, strict=False), filename)


def _normalized_workflow_alias_path(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(_absolute_workflow_path_preserving_traversal(path))


def _case_variant_workflow_path(path: Path) -> Path | None:
    """Return an ASCII case variant suitable for probing one directory namespace."""
    for index, character in enumerate(path.name):
        if "a" <= character <= "z":
            variant = f"{path.name[:index]}{character.upper()}{path.name[index + 1 :]}"
            return path.with_name(variant)
        if "A" <= character <= "Z":
            variant = f"{path.name[:index]}{character.lower()}{path.name[index + 1 :]}"
            return path.with_name(variant)
    return None


def _workflow_namespace_is_case_insensitive(probe_path: Path) -> bool:
    """Detect case folding from an existing, single-link workflow artifact."""
    case_variant = _case_variant_workflow_path(probe_path)
    if case_variant is None:
        return False
    try:
        original = os.lstat(probe_path)
        variant = os.lstat(case_variant)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return os.path.samestat(original, variant)


def _windows_directory_namespace_is_case_insensitive(directory_path: Path) -> bool | None:
    """Query the case-sensitivity flag governing names inside one Windows directory."""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except (AttributeError, OSError) as exc:  # pragma: no cover - native Windows failure path
        raise OSError("Cannot load the Windows file-information API.") from exc

    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    get_file_information = kernel32.GetFileInformationByHandleEx
    get_file_information.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    get_file_information.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int

    directory_handle = create_file(
        os.fspath(directory_path),
        _WINDOWS_FILE_READ_ATTRIBUTES,
        _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE | _WINDOWS_FILE_SHARE_DELETE,
        None,
        _WINDOWS_OPEN_EXISTING,
        _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if directory_handle in (None, invalid_handle):
        error_code = ctypes.get_last_error()
        raise OSError(error_code, f"Windows could not open directory {directory_path} for case-sensitivity inspection.")

    operation_failure: BaseException | None = None
    try:
        case_sensitive_information = _WindowsFileCaseSensitiveInformation()
        if not get_file_information(
            directory_handle,
            _WINDOWS_FILE_CASE_SENSITIVE_INFO,
            ctypes.byref(case_sensitive_information),
            ctypes.sizeof(case_sensitive_information),
        ):
            error_code = ctypes.get_last_error()
            raise OSError(
                error_code,
                f"Windows could not query case sensitivity for directory {directory_path}.",
            )
        return not bool(case_sensitive_information.flags & _WINDOWS_FILE_CS_FLAG_CASE_SENSITIVE_DIR)
    except BaseException as exc:
        operation_failure = exc
        raise
    finally:
        if not close_handle(directory_handle):
            error_code = ctypes.get_last_error()
            close_failure = OSError(
                error_code,
                f"Windows could not close the case-sensitivity inspection handle for {directory_path}.",
            )
            if operation_failure is not None:
                operation_failure.add_note(str(close_failure))
            else:
                raise close_failure


def _casefolded_workflow_collision_parent(
    first_path: Path,
    second_path: Path,
    *,
    first_may_be_ancestor: bool,
) -> Path | None:
    """Return the namespace containing a case-only path collision."""
    canonical_first = Path(_case_preserving_workflow_path(first_path))
    canonical_second = Path(_case_preserving_workflow_path(second_path))
    first_parts = canonical_first.parts
    second_parts = canonical_second.parts
    if first_may_be_ancestor:
        if len(first_parts) > len(second_parts):
            return None
    elif len(first_parts) != len(second_parts):
        return None
    if any(first.casefold() != second.casefold() for first, second in zip(first_parts, second_parts, strict=False)):
        return None

    differing_index = next(
        (
            index
            for index, (first, second) in enumerate(zip(first_parts, second_parts, strict=False))
            if first != second
        ),
        None,
    )
    if differing_index is None:
        return None
    if differing_index == 0:
        first_anchor = canonical_first.anchor
        second_anchor = canonical_second.anchor
        if first_anchor and first_anchor.casefold() == second_anchor.casefold():
            return Path(first_anchor)
        return None
    return Path(*first_parts[:differing_index])


def _workflow_directory_namespace_is_case_insensitive(
    directory_path: Path,
    *,
    probe_paths: set[Path],
) -> bool:
    """Detect case folding for an existing namespace without creating artifacts."""
    existing_directory = Path(_case_preserving_workflow_path(directory_path))
    while True:
        existing_status = _existing_workflow_path_status(existing_directory)
        if existing_status is not None:
            if not stat.S_ISDIR(existing_status.st_mode):
                return False
            break
        parent = existing_directory.parent
        if parent == existing_directory:
            return False
        existing_directory = parent

    canonical_directory = _case_preserving_workflow_path(existing_directory)
    windows_case_insensitive = _windows_directory_namespace_is_case_insensitive(existing_directory)
    if windows_case_insensitive is not None:
        return windows_case_insensitive
    for probe_path in sorted(probe_paths, key=str):
        if _existing_workflow_path_status(probe_path) is None:
            continue
        if _case_preserving_workflow_path(probe_path.parent) != canonical_directory:
            continue
        if _case_variant_workflow_path(probe_path) is None:
            continue
        return _workflow_namespace_is_case_insensitive(probe_path)

    # If the namespace has no relevant existing child, its own directory entry
    # is the only non-mutating probe available. This accurately identifies the
    # volume-wide behavior used by ordinary case-insensitive filesystems and is
    # conservative for a missing descendant that will inherit its parent.
    return _workflow_namespace_is_case_insensitive(existing_directory)


def _existing_workflow_path_status(path: Path) -> os.stat_result | None:
    try:
        return os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _validate_workflow_runtime_path_separation(
    *,
    database_path: Path,
    output_path: Path,
    database_lock_paths: set[Path],
    database_alias_paths: set[Path] | None = None,
    output_anchor_lock_paths: set[Path] | None = None,
    account_lock_paths: set[Path] | None = None,
    runtime_lock_root_paths: set[Path] | None = None,
    database_identity_lock_path: Path | None = None,
    case_sensitivity_probe_paths: set[Path] | None = None,
) -> None:
    """Reject configurations where independently managed runtime artifacts overlap."""
    database_paths = {database_path, *(database_alias_paths or ())}
    database_runtime_paths = {
        runtime_path
        for candidate in database_paths
        for runtime_path in (
            candidate,
            *(Path(f"{candidate}{suffix}") for suffix in _SQLITE_RUNTIME_SIDECAR_SUFFIXES),
        )
    }
    output_artifacts = {output_path / filename for filename in _MANAGED_OUTPUT_FILENAMES}
    output_artifacts.add(output_path)
    output_artifacts.add(output_path / ".leveraged-trader.lock")
    if output_anchor_lock_paths is not None:
        output_artifacts.update(output_anchor_lock_paths)

    database_artifacts = {*database_runtime_paths, *database_lock_paths}
    if database_identity_lock_path is not None:
        normalized_identity_lock = _normalized_workflow_path(database_identity_lock_path)
        if any(_normalized_workflow_path(path) == normalized_identity_lock for path in database_artifacts):
            raise ValueError(
                "SQLite database and database lock artifacts must use distinct paths; "
                f"{database_identity_lock_path} conflicts with the database path or a path lock."
            )
        database_artifacts.add(database_identity_lock_path)

    case_probe_paths = {
        *database_artifacts,
        *output_artifacts,
        *(case_sensitivity_probe_paths or ()),
    }
    case_sensitivity_cache: dict[str, bool] = {}

    def namespace_is_case_insensitive(namespace_parent: Path) -> bool:
        canonical_parent = _case_preserving_workflow_path(namespace_parent)
        if canonical_parent not in case_sensitivity_cache:
            case_sensitivity_cache[canonical_parent] = _workflow_directory_namespace_is_case_insensitive(
                namespace_parent,
                probe_paths=case_probe_paths,
            )
        return case_sensitivity_cache[canonical_parent]

    for database_runtime_path in database_runtime_paths:
        canonical_database_runtime_path = _case_preserving_workflow_path(database_runtime_path)
        for runtime_lock_root in runtime_lock_root_paths or ():
            canonical_lock_root = _case_preserving_workflow_path(runtime_lock_root)
            try:
                common_path = os.path.commonpath((canonical_database_runtime_path, canonical_lock_root))
            except ValueError:  # Different Windows drives are necessarily distinct.
                common_path = ""
            if canonical_database_runtime_path == canonical_lock_root or common_path == canonical_database_runtime_path:
                raise ValueError(
                    "db_path and its SQLite sidecars must not contain the per-user workflow lock directory; "
                    f"{database_runtime_path} conflicts with {runtime_lock_root}."
                )
            casefolded_parent = _casefolded_workflow_collision_parent(
                database_runtime_path,
                runtime_lock_root,
                first_may_be_ancestor=True,
            )
            if casefolded_parent is not None and namespace_is_case_insensitive(casefolded_parent):
                raise ValueError(
                    "db_path and its SQLite sidecars must not contain the per-user workflow lock directory; "
                    f"{database_runtime_path} conflicts with {runtime_lock_root}."
                )

    canonical_output = _case_preserving_workflow_path(output_path)
    for runtime_lock_root in runtime_lock_root_paths or ():
        canonical_lock_root = _case_preserving_workflow_path(runtime_lock_root)
        try:
            common_path = os.path.commonpath((canonical_output, canonical_lock_root))
        except ValueError:  # Different Windows drives are necessarily distinct.
            common_path = ""
        if canonical_output == canonical_lock_root or common_path == canonical_output:
            raise ValueError(
                "output_dir must not contain the per-user workflow lock directory; "
                f"{output_path} conflicts with {runtime_lock_root}."
            )
        casefolded_parent = _casefolded_workflow_collision_parent(
            output_path,
            runtime_lock_root,
            first_may_be_ancestor=True,
        )
        if casefolded_parent is not None and namespace_is_case_insensitive(casefolded_parent):
            raise ValueError(
                "output_dir must not contain the per-user workflow lock directory; "
                f"{output_path} conflicts with {runtime_lock_root}."
            )

    for account_lock_path in account_lock_paths or ():
        canonical_account_lock = _case_preserving_workflow_path(account_lock_path)
        try:
            common_path = os.path.commonpath((canonical_account_lock, canonical_output))
        except ValueError:  # Different Windows drives are necessarily distinct.
            common_path = ""
        if common_path in {canonical_account_lock, canonical_output}:
            raise ValueError(
                "The Alpaca account lock and workflow output directory must use distinct namespaces; "
                f"{account_lock_path} conflicts with {output_path}."
            )
        for first_path, second_path in (
            (account_lock_path, output_path),
            (output_path, account_lock_path),
        ):
            casefolded_parent = _casefolded_workflow_collision_parent(
                first_path,
                second_path,
                first_may_be_ancestor=True,
            )
            if casefolded_parent is not None and namespace_is_case_insensitive(casefolded_parent):
                raise ValueError(
                    "The Alpaca account lock and workflow output directory must use distinct namespaces; "
                    f"{account_lock_path} conflicts with {output_path}."
                )

        for database_artifact in database_artifacts:
            canonical_database_artifact = _case_preserving_workflow_path(database_artifact)
            if canonical_account_lock == canonical_database_artifact:
                raise ValueError(
                    "The Alpaca account lock and SQLite database artifacts must use distinct paths; "
                    f"{account_lock_path} conflicts with {database_artifact}."
                )
            casefolded_parent = _casefolded_workflow_collision_parent(
                account_lock_path,
                database_artifact,
                first_may_be_ancestor=False,
            )
            if casefolded_parent is not None and namespace_is_case_insensitive(casefolded_parent):
                raise ValueError(
                    "The Alpaca account lock and SQLite database artifacts must use distinct paths; "
                    f"{account_lock_path} conflicts with {database_artifact}."
                )

        account_lock_status = _existing_workflow_path_status(account_lock_path)
        if account_lock_status is not None:
            for database_artifact in database_artifacts:
                database_status = _existing_workflow_path_status(database_artifact)
                if database_status is not None and os.path.samestat(account_lock_status, database_status):
                    raise ValueError(
                        "The Alpaca account lock and SQLite database artifacts must use distinct paths; "
                        f"{account_lock_path} aliases {database_artifact}."
                    )
            for output_candidate in (output_path, *output_path.parents, *output_artifacts):
                output_status = _existing_workflow_path_status(output_candidate)
                if output_status is not None and os.path.samestat(account_lock_status, output_status):
                    raise ValueError(
                        "The Alpaca account lock and workflow output directory must use distinct namespaces; "
                        f"{account_lock_path} aliases {output_candidate}."
                    )

    for database_runtime_path in database_runtime_paths:
        canonical_database_runtime_path = _case_preserving_workflow_path(database_runtime_path)
        try:
            common_path = os.path.commonpath((canonical_database_runtime_path, canonical_output))
        except ValueError:  # Different Windows drives are necessarily distinct.
            common_path = ""
        if canonical_database_runtime_path == canonical_output or common_path == canonical_database_runtime_path:
            raise ValueError(
                "db_path and its SQLite sidecars must use distinct paths; they must not be "
                "the report output directory or one of its ancestors."
            )
        casefolded_parent = _casefolded_workflow_collision_parent(
            database_runtime_path,
            output_path,
            first_may_be_ancestor=True,
        )
        if casefolded_parent is not None and namespace_is_case_insensitive(casefolded_parent):
            raise ValueError(
                "db_path and its SQLite sidecars must use distinct paths; they must not be "
                "the report output directory or one of its ancestors."
            )

    database_status = _existing_workflow_path_status(database_path)
    if database_status is not None:
        for output_ancestor in (output_path, *output_path.parents):
            output_ancestor_status = _existing_workflow_path_status(output_ancestor)
            if output_ancestor_status is not None and os.path.samestat(database_status, output_ancestor_status):
                raise ValueError("db_path must not be the report output directory or one of its ancestors.")

    for database_artifact in database_artifacts:
        for output_artifact in output_artifacts:
            casefolded_parent = _casefolded_workflow_collision_parent(
                database_artifact,
                output_artifact,
                first_may_be_ancestor=False,
            )
            if casefolded_parent is not None and namespace_is_case_insensitive(casefolded_parent):
                raise ValueError(
                    "SQLite database and workflow output artifacts must use distinct paths; "
                    f"{database_artifact} conflicts with {output_artifact}."
                )

    case_insensitive_parent_identities: set[tuple[int, int]] = set()
    for probe_path in case_sensitivity_probe_paths or ():
        if _workflow_namespace_is_case_insensitive(probe_path):
            parent_status = os.stat(probe_path.parent)
            case_insensitive_parent_identities.add((parent_status.st_dev, parent_status.st_ino))

    canonical_output_artifacts = {_case_preserving_workflow_path(path): path for path in output_artifacts}
    output_artifact_statuses = {
        output_artifact: _existing_workflow_path_status(output_artifact) for output_artifact in output_artifacts
    }
    parent_statuses: dict[Path, os.stat_result | None] = {}

    def parent_status(path: Path) -> os.stat_result | None:
        if path.parent not in parent_statuses:
            parent_statuses[path.parent] = _existing_workflow_path_status(path.parent)
        return parent_statuses[path.parent]

    for database_artifact in database_artifacts:
        database_parent_status = parent_status(database_artifact)
        database_parent_is_output = _case_preserving_workflow_path(database_artifact.parent) == canonical_output or (
            database_parent_status is not None
            and (output_status := _existing_workflow_path_status(output_path)) is not None
            and os.path.samestat(database_parent_status, output_status)
        )
        if database_parent_is_output and _is_managed_stale_report_artifact(database_artifact.name):
            raise ValueError(
                "SQLite database and lock artifacts must not use a reserved report-cleanup name "
                f"inside output_dir: {database_artifact}."
            )
        conflicting_output = canonical_output_artifacts.get(_case_preserving_workflow_path(database_artifact))
        database_artifact_status = _existing_workflow_path_status(database_artifact)
        if conflicting_output is None:
            for output_artifact, output_artifact_status in output_artifact_statuses.items():
                if (
                    database_artifact_status is not None
                    and output_artifact_status is not None
                    and os.path.samestat(database_artifact_status, output_artifact_status)
                ):
                    conflicting_output = output_artifact
                    break

                output_parent_status = parent_status(output_artifact)
                if (
                    database_parent_status is not None
                    and output_parent_status is not None
                    and os.path.samestat(database_parent_status, output_parent_status)
                    and (database_parent_status.st_dev, database_parent_status.st_ino)
                    in case_insensitive_parent_identities
                    and database_artifact.name.casefold() == output_artifact.name.casefold()
                ):
                    conflicting_output = output_artifact
                    break
        if conflicting_output is not None:
            raise ValueError(
                "SQLite database and workflow output artifacts must use distinct paths; "
                f"{database_artifact} conflicts with {conflicting_output}."
            )


def _lock_file_nonblocking(lock_file: Any) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return

    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write("\0")
        lock_file.flush()
    lock_file.seek(0)
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:  # pragma: no cover - exercised on native Windows
        raise BlockingIOError from exc


def _unlock_file(lock_file: Any) -> None:
    if fcntl is not None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return

    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # pragma: no cover - native Windows


def _require_safe_workflow_lock_status(lock_path: Path, file_status: os.stat_result) -> None:
    if not stat.S_ISREG(file_status.st_mode):
        raise PermissionError(f"Workflow lock {lock_path} must be a regular file.")
    if os.name == "posix":
        if file_status.st_uid != os.geteuid():
            raise PermissionError(f"Workflow lock {lock_path} must be owned by the current user.")
        if stat.S_IMODE(file_status.st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(
                f"Workflow lock {lock_path} must not be accessible by group or other users; "
                "remove it or change its mode to 0600 before running the workflow."
            )
    if file_status.st_nlink != 1:
        raise PermissionError(f"Workflow lock {lock_path} must not have multiple hard links.")


def _open_workflow_lock_file(lock_path: Path) -> Any:
    """Open one private lock inode without following a substituted path."""
    try:
        path_before_open = os.lstat(lock_path)
    except FileNotFoundError:
        path_before_open = None
    else:
        if stat.S_ISLNK(path_before_open.st_mode):
            raise PermissionError(f"Workflow lock {lock_path} must not be a symbolic link.")
        _require_safe_workflow_lock_status(lock_path, path_before_open)

    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    file_descriptor = os.open(lock_path, flags, 0o600)
    try:
        opened_file = os.fstat(file_descriptor)
        _require_safe_workflow_lock_status(lock_path, opened_file)
        if path_before_open is not None and (path_before_open.st_dev, path_before_open.st_ino) != (
            opened_file.st_dev,
            opened_file.st_ino,
        ):
            raise PermissionError(f"Workflow lock {lock_path} changed while it was being opened.")
        path_after_open = os.lstat(lock_path)
        _require_safe_workflow_lock_status(lock_path, path_after_open)
        if (path_after_open.st_dev, path_after_open.st_ino) != (opened_file.st_dev, opened_file.st_ino):
            raise PermissionError(f"Workflow lock {lock_path} changed while it was being opened.")
        if os.name == "posix":
            os.fchmod(file_descriptor, 0o600)
        return os.fdopen(file_descriptor, "a+", encoding="utf-8")
    except BaseException:
        os.close(file_descriptor)
        raise


def _preflight_existing_workflow_lock_file(lock_path: Path) -> None:
    """Reject an unsafe existing lock leaf before creating other runtime paths."""
    try:
        existing_lock = os.lstat(lock_path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(existing_lock.st_mode):
        raise PermissionError(f"Workflow lock {lock_path} must not be a symbolic link.")
    _require_safe_workflow_lock_status(lock_path, existing_lock)


def _revalidate_workflow_lock_file(lock_path: Path, lock_file: Any) -> None:
    opened_file = os.fstat(lock_file.fileno())
    _require_safe_workflow_lock_status(lock_path, opened_file)
    current_path = os.lstat(lock_path)
    _require_safe_workflow_lock_status(lock_path, current_path)
    if (current_path.st_dev, current_path.st_ino) != (opened_file.st_dev, opened_file.st_ino):
        raise PermissionError(f"Workflow lock {lock_path} changed while it was being acquired.")


def _release_workflow_lock_files(lock_files: list[Any]) -> list[tuple[str, BaseException]]:
    cleanup_failures: list[tuple[str, BaseException]] = []
    for lock_file in reversed(lock_files):
        try:
            _unlock_file(lock_file)
        except BaseException as exc:
            cleanup_failures.append(("lock-file unlock", exc))
        try:
            lock_file.close()
        except BaseException as exc:
            cleanup_failures.append(("lock-file close", exc))
    return cleanup_failures


def _open_workflow_database_anchor(guard: SqliteRuntimeGuard | None) -> int | None:
    """Hold the pinned database inode for the lifetime of the workflow lock."""
    if guard is None:
        return None

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(guard.database_path, flags)
    try:
        _revalidate_workflow_database_anchor(guard, descriptor)
        revalidate_private_runtime_file(guard)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _revalidate_workflow_database_anchor(
    guard: SqliteRuntimeGuard | None,
    descriptor: int | None,
) -> None:
    if guard is None or descriptor is None:
        return
    observed = os.fstat(descriptor)
    expected = guard.database_identity
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_dev != expected.device
        or observed.st_ino != expected.inode
        or observed.st_uid != os.geteuid()
        or observed.st_nlink != 1
    ):
        raise OSError(f"SQLite database {guard.database_path} changed identity or link count while it was open.")


def _owned_writable_directory(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> PrivateRuntimeDirectory | None:
    """Return a verified lock parent only when its owner can create entries."""
    try:
        directory = require_owned_runtime_directory(path, label=label)
    except (OSError, PermissionError):
        return None
    if os.name != "posix":
        return directory if os.access(path, os.W_OK) else None
    assert directory is not None
    directory_status = os.stat(directory.path, follow_symlinks=False)
    if not stat.S_IMODE(directory_status.st_mode) & stat.S_IWUSR:
        return None
    return directory


def _windows_local_app_data_path() -> Path:
    """Resolve LocalAppData through the Windows Known Folder API."""
    if os.name != "nt":
        raise OSError("The Windows Known Folder API is unavailable on this platform.")

    try:
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        ole32 = ctypes.WinDLL("ole32", use_last_error=True)
    except (AttributeError, OSError) as exc:  # pragma: no cover - native Windows failure path
        raise OSError("Cannot load the Windows Known Folder API.") from exc

    get_known_folder_path = shell32.SHGetKnownFolderPath
    get_known_folder_path.argtypes = (
        ctypes.POINTER(_WindowsGuid),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    get_known_folder_path.restype = ctypes.c_int32
    free_task_memory = ole32.CoTaskMemFree
    free_task_memory.argtypes = (ctypes.c_void_p,)
    free_task_memory.restype = None
    initialize_com = ole32.CoInitializeEx
    initialize_com.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    initialize_com.restype = ctypes.c_int32
    uninitialize_com = ole32.CoUninitialize
    uninitialize_com.argtypes = ()
    uninitialize_com.restype = None

    initialization_result = initialize_com(None, _WINDOWS_COINIT_MULTITHREADED)
    initialization_code = ctypes.c_uint32(initialization_result).value
    must_uninitialize = initialization_code in {_WINDOWS_S_OK, _WINDOWS_S_FALSE}
    if not must_uninitialize and initialization_code != _WINDOWS_RPC_E_CHANGED_MODE:
        raise OSError(f"Windows could not initialize COM (HRESULT 0x{initialization_code:08X}).")

    try:
        path_pointer = ctypes.c_void_p()
        result = get_known_folder_path(
            ctypes.byref(_WINDOWS_LOCAL_APP_DATA_FOLDER_ID),
            0,
            None,
            ctypes.byref(path_pointer),
        )
        if result != 0:
            if path_pointer.value is not None:
                free_task_memory(path_pointer)
            result_code = ctypes.c_uint32(result).value
            raise OSError(f"Windows could not resolve LocalAppData (HRESULT 0x{result_code:08X}).")
        if path_pointer.value is None:
            raise OSError("Windows returned an empty LocalAppData path.")

        try:
            path_value = ctypes.wstring_at(path_pointer.value)
        finally:
            free_task_memory(path_pointer)
        local_app_data = Path(path_value)
        if not path_value or not local_app_data.is_absolute():
            raise OSError("Windows returned an invalid LocalAppData path.")
        return local_app_data.resolve()
    finally:
        if must_uninitialize:
            uninitialize_com()


def _workflow_user_lock_base() -> tuple[Path, PrivateRuntimeDirectory | None]:
    """Select an account-bound base independent of caller-controlled environment."""
    if os.name == "posix":
        try:
            home_path = pwd.getpwuid(os.geteuid()).pw_dir
        except KeyError as exc:
            raise PermissionError("Cannot find a private per-user directory for workflow locks.") from exc
        if not home_path or not os.path.isabs(home_path):
            raise PermissionError("Cannot find a private per-user directory for workflow locks.")
    else:  # pragma: no cover - exercised on native Windows
        try:
            return _windows_local_app_data_path(), None
        except OSError as exc:
            raise PermissionError("Cannot find a stable per-user directory for workflow locks.") from exc

    try:
        home_base = require_owned_runtime_directory(
            home_path,
            label="Per-user workflow state directory",
        )
    except (OSError, PermissionError) as exc:
        raise PermissionError("Cannot find a private per-user directory for workflow locks.") from exc
    assert home_base is not None
    return Path(home_base.path), home_base


def _workflow_user_lock_root(user_base_path: Path) -> Path:
    return user_base_path / ".local" / "state" / "leveraged-trader" / "workflow-locks"


def _prepare_workflow_user_lock_directory(
    user_base_path: Path | None = None,
) -> tuple[Path, PrivateRuntimeDirectory | None]:
    if user_base_path is None:
        user_base_path, _user_base = _workflow_user_lock_base()
    lock_root = _workflow_user_lock_root(user_base_path)
    lock_directory = prepare_owned_runtime_directory(
        lock_root,
        label="Per-user workflow lock directory",
    )
    return (
        Path(lock_directory.path if lock_directory is not None else lock_root).resolve(),
        lock_directory,
    )


def _prepare_workflow_output_anchor_directory(
    canonical_output_path: Path,
    *,
    user_base_path: Path | None = None,
) -> tuple[Path, PrivateRuntimeDirectory | None]:
    """Keep the logical-path anchor outside the relocatable output namespace."""
    del canonical_output_path
    return _prepare_workflow_user_lock_directory(user_base_path)


def _workflow_lock_path_digest(path: str | os.PathLike[str]) -> str:
    normalized_path = _normalized_workflow_alias_path(path)
    return hashlib.sha256(os.fsencode(normalized_path)).hexdigest()[:32]


def _validate_workflow_lock_tier_separation(
    *,
    account_anchor_lock_paths: set[Path],
    path_local_lock_paths: set[Path],
    identity_lock_paths: set[Path],
) -> None:
    """Ensure one lock inode is never acquired twice by the same workflow."""
    tiers = (
        ("account-anchor", account_anchor_lock_paths),
        ("path-local", path_local_lock_paths),
        ("database-identity", identity_lock_paths),
    )
    for tier_index, (first_name, first_paths) in enumerate(tiers):
        for second_name, second_paths in tiers[tier_index + 1 :]:
            overlapping_paths = first_paths & second_paths
            if overlapping_paths:
                overlap = sorted(overlapping_paths, key=str)[0]
                raise ValueError(
                    "Workflow lock paths must be distinct across acquisition tiers; "
                    f"{overlap} belongs to both the {first_name} and {second_name} tiers."
                )


@contextmanager
def _workflow_run_lock(
    db_path: str,
    output_dir: str,
    *,
    serialize_alpaca_account: bool = False,
):
    _validate_database_path(db_path)
    _validate_output_directory_path(output_dir)
    canonical_db_path = Path(_case_preserving_workflow_path(db_path))
    canonical_output_candidate = Path(_case_preserving_workflow_path(output_dir))
    supplied_database_alias_path = Path(_case_preserving_workflow_alias_path(db_path))
    database_alias_paths = {
        Path(_absolute_workflow_path_preserving_traversal(db_path)),
        supplied_database_alias_path,
        canonical_db_path,
    }
    supplied_database_lock_path = Path(f"{supplied_database_alias_path}.lock")
    local_database_lock_paths = {
        supplied_database_lock_path,
        Path(f"{canonical_db_path}.lock"),
    }
    output_anchor_base_path, _output_anchor_base = _workflow_user_lock_base()
    prospective_output_anchor_parent = _workflow_user_lock_root(output_anchor_base_path)
    logical_database_paths = {
        _normalized_workflow_alias_path(db_path),
        os.path.normcase(os.fspath(canonical_db_path)),
    }
    logical_database_lock_paths = {
        prospective_output_anchor_parent / f".leveraged-trader-database-path-{_workflow_lock_path_digest(path)}.lock"
        for path in logical_database_paths
    }
    prospective_logical_output_paths = {
        _normalized_workflow_alias_path(output_dir),
        os.path.normcase(os.fspath(canonical_output_candidate)),
    }
    prospective_output_anchor_lock_paths = {
        prospective_output_anchor_parent / f".leveraged-trader-output-{_workflow_lock_path_digest(path)}.lock"
        for path in prospective_logical_output_paths
    }
    prospective_alpaca_account_lock_paths = (
        {prospective_output_anchor_parent / _ALPACA_ACCOUNT_LOCK_FILENAME} if serialize_alpaca_account else set()
    )
    _validate_workflow_runtime_path_separation(
        database_path=canonical_db_path,
        output_path=canonical_output_candidate,
        database_lock_paths=local_database_lock_paths | logical_database_lock_paths,
        database_alias_paths=database_alias_paths,
        output_anchor_lock_paths=prospective_output_anchor_lock_paths,
        account_lock_paths=prospective_alpaca_account_lock_paths,
        runtime_lock_root_paths={prospective_output_anchor_parent},
    )
    try:
        db_stat = canonical_db_path.stat()
    except FileNotFoundError:
        db_stat = None
    if db_stat is not None and stat.S_ISREG(db_stat.st_mode) and db_stat.st_nlink > 1:
        raise WorkflowRunError(
            f"Database {db_path} has multiple hard links; use one canonical path for the SQLite database."
        )
    for database_lock_path in local_database_lock_paths:
        _preflight_existing_workflow_lock_file(database_lock_path)
    for derived_lock_path in (
        logical_database_lock_paths | prospective_output_anchor_lock_paths | prospective_alpaca_account_lock_paths
    ):
        _preflight_existing_workflow_lock_file(derived_lock_path)
    for database_parent in {lock_path.parent for lock_path in local_database_lock_paths}:
        preflight_owned_runtime_directory(
            database_parent,
            label="SQLite database parent directory",
        )
    preflight_owned_runtime_directory(
        output_dir,
        label="Report output directory",
    )
    output_anchor_parent, output_anchor_directory = _prepare_workflow_output_anchor_directory(
        canonical_output_candidate,
        user_base_path=output_anchor_base_path,
    )
    logical_database_lock_paths = {
        output_anchor_parent / f".leveraged-trader-database-path-{_workflow_lock_path_digest(path)}.lock"
        for path in logical_database_paths
    }
    for database_lock_path in logical_database_lock_paths:
        _preflight_existing_workflow_lock_file(database_lock_path)
    database_lock_paths = local_database_lock_paths | logical_database_lock_paths
    # Validate (and, when necessary, privately create) every database-lock
    # parent before creating the report directory.  A rejected database
    # namespace must not leave an otherwise-unused output tree behind.
    database_lock_parent_guards = {
        lock_path: (
            output_anchor_directory
            if lock_path in logical_database_lock_paths
            else prepare_owned_runtime_directory(
                lock_path.parent,
                label="SQLite database parent directory",
            )
        )
        for lock_path in database_lock_paths
    }
    output_directory = prepare_owned_runtime_directory(
        output_dir,
        label="Report output directory",
    )
    canonical_output_path = Path(output_directory.path if output_directory is not None else output_dir).resolve()
    output_lock_path = canonical_output_path / ".leveraged-trader.lock"
    logical_output_paths = {
        _normalized_workflow_alias_path(output_dir),
        os.path.normcase(os.fspath(canonical_output_path)),
    }
    output_anchor_lock_paths = {
        output_anchor_parent / f".leveraged-trader-output-{_workflow_lock_path_digest(path)}.lock"
        for path in logical_output_paths
    }
    alpaca_account_lock_paths = (
        {output_anchor_parent / _ALPACA_ACCOUNT_LOCK_FILENAME} if serialize_alpaca_account else set()
    )
    for alpaca_account_lock_path in alpaca_account_lock_paths:
        _preflight_existing_workflow_lock_file(alpaca_account_lock_path)
    _validate_workflow_runtime_path_separation(
        database_path=canonical_db_path,
        output_path=canonical_output_path,
        database_lock_paths=database_lock_paths,
        database_alias_paths=database_alias_paths,
        output_anchor_lock_paths=output_anchor_lock_paths,
        account_lock_paths=alpaca_account_lock_paths,
        runtime_lock_root_paths={output_anchor_parent},
    )
    account_anchor_lock_paths = logical_database_lock_paths | output_anchor_lock_paths | alpaca_account_lock_paths
    path_local_lock_paths = local_database_lock_paths | {output_lock_path}
    identity_lock_paths: set[Path] = set()
    lock_validators: dict[Path, Callable[[], None]] = {}
    lock_files: list[Any] = []
    preopened_lock_files: dict[Path, Any] = {}
    database_guard: SqliteRuntimeGuard | None = None
    database_anchor_descriptor: int | None = None
    pinned_database_path = canonical_db_path
    database_revalidation_lock = threading.RLock()

    def revalidate_database() -> SqliteRuntimeGuard | None:
        nonlocal database_guard
        with database_revalidation_lock:
            _revalidate_workflow_database_anchor(database_guard, database_anchor_descriptor)
            # A workflow opens and closes several legitimate SQLite connections, so
            # WAL/journal inodes can be recreated during the run. Validate their
            # current safety without treating that normal lifecycle as DB replacement.
            guard_for_revalidation = (
                replace(
                    database_guard,
                    sidecars=tuple((path, None) for path, _identity in database_guard.sidecars),
                )
                if database_guard is not None
                else None
            )
            database_guard = revalidate_private_runtime_file(guard_for_revalidation)
            _revalidate_workflow_database_anchor(database_guard, database_anchor_descriptor)
            return database_guard

    def release_runtime_handles() -> list[tuple[str, BaseException]]:
        nonlocal database_anchor_descriptor
        cleanup_failures = _release_workflow_lock_files(lock_files)
        if database_anchor_descriptor is not None:
            try:
                os.close(database_anchor_descriptor)
            except BaseException as exc:
                cleanup_failures.append(("database-inode anchor close", exc))
            finally:
                database_anchor_descriptor = None
        return cleanup_failures

    def acquire_lock(lock_path: Path, *, before_and_after: Callable[[], None]) -> None:
        before_and_after()
        lock_file = preopened_lock_files.pop(lock_path, None)
        if lock_file is None:
            lock_file = _open_workflow_lock_file(lock_path)
        lock_files.append(lock_file)
        _lock_file_nonblocking(lock_file)
        _revalidate_workflow_lock_file(lock_path, lock_file)
        before_and_after()

    def acquire_lock_tiers() -> None:
        """Acquire stable account anchors before local paths and inode identities."""
        _validate_workflow_lock_tier_separation(
            account_anchor_lock_paths=account_anchor_lock_paths,
            path_local_lock_paths=path_local_lock_paths,
            identity_lock_paths=identity_lock_paths,
        )
        for lock_tier in (
            account_anchor_lock_paths,
            path_local_lock_paths,
            identity_lock_paths,
        ):
            for lock_path in sorted(lock_tier, key=str):
                acquire_lock(
                    lock_path,
                    before_and_after=lock_validators[lock_path],
                )

    def validate_output_anchor_lock() -> None:
        revalidate_owned_runtime_directory(
            output_anchor_directory,
            label="Per-user workflow lock directory",
        )
        revalidate_owned_runtime_directory(
            output_directory,
            label="Report output directory",
        )

    for lock_path in account_anchor_lock_paths:
        lock_validators[lock_path] = (
            validate_output_anchor_lock
            if lock_path in output_anchor_lock_paths
            else partial(
                revalidate_owned_runtime_directory,
                output_anchor_directory,
                label="Per-user workflow lock directory",
            )
        )
    for lock_path in path_local_lock_paths:
        lock_validators[lock_path] = (
            partial(
                revalidate_owned_runtime_directory,
                output_directory,
                label="Report output directory",
            )
            if lock_path == output_lock_path
            else partial(
                revalidate_owned_runtime_directory,
                database_lock_parent_guards[lock_path],
                label="SQLite database parent directory",
            )
        )

    try:
        _validate_workflow_lock_tier_separation(
            account_anchor_lock_paths=account_anchor_lock_paths,
            path_local_lock_paths=path_local_lock_paths,
            identity_lock_paths=identity_lock_paths,
        )
        case_sensitivity_probe_paths = {
            output_lock_path,
            *output_anchor_lock_paths,
            *alpaca_account_lock_paths,
        }
        for probe_path in sorted(case_sensitivity_probe_paths, key=str):
            probe_validator = lock_validators[probe_path]
            probe_validator()
            probe_file = _open_workflow_lock_file(probe_path)
            preopened_lock_files[probe_path] = probe_file
            _revalidate_workflow_lock_file(probe_path, probe_file)
            probe_validator()

        _validate_workflow_runtime_path_separation(
            database_path=canonical_db_path,
            output_path=canonical_output_path,
            database_lock_paths=database_lock_paths,
            database_alias_paths=database_alias_paths,
            output_anchor_lock_paths=output_anchor_lock_paths,
            account_lock_paths=alpaca_account_lock_paths,
            runtime_lock_root_paths={output_anchor_parent},
            case_sensitivity_probe_paths=case_sensitivity_probe_paths,
        )
        acquire_lock_tiers()

        revalidate_owned_runtime_directory(output_directory, label="Report output directory")
        database_guard = prepare_private_runtime_file(db_path)
        if database_guard is not None:
            pinned_database_path = Path(database_guard.database_path)
        database_anchor_descriptor = _open_workflow_database_anchor(database_guard)
        revalidate_database()

        pinned_database_logical_path = _normalized_workflow_path(pinned_database_path)
        pinned_database_anchor_lock_path = output_anchor_parent / (
            f".leveraged-trader-database-path-{_workflow_lock_path_digest(pinned_database_logical_path)}.lock"
        )
        pinned_database_anchor_was_known = pinned_database_anchor_lock_path in account_anchor_lock_paths
        if not pinned_database_anchor_was_known:
            _preflight_existing_workflow_lock_file(pinned_database_anchor_lock_path)
        logical_database_lock_paths.add(pinned_database_anchor_lock_path)
        account_anchor_lock_paths.add(pinned_database_anchor_lock_path)
        database_lock_paths.add(pinned_database_anchor_lock_path)
        pinned_database_lock_path = Path(f"{pinned_database_path}.lock")
        pinned_database_lock_was_known = pinned_database_lock_path in path_local_lock_paths
        if not pinned_database_lock_was_known:
            _preflight_existing_workflow_lock_file(pinned_database_lock_path)
        database_lock_paths.add(pinned_database_lock_path)
        path_local_lock_paths.add(pinned_database_lock_path)
        identity_lock_path: Path | None = None
        identity_lock_directory: PrivateRuntimeDirectory | None = None
        if database_guard is not None:
            identity_lock_parent, identity_lock_directory = _prepare_workflow_user_lock_directory(
                output_anchor_base_path
            )
            identity = database_guard.database_identity
            identity_lock_path = identity_lock_parent / (
                f".leveraged-trader-database-{identity.device:x}-{identity.inode:x}.lock"
            )

        _validate_workflow_runtime_path_separation(
            database_path=pinned_database_path,
            output_path=canonical_output_path,
            database_lock_paths=database_lock_paths,
            database_alias_paths=database_alias_paths,
            output_anchor_lock_paths=output_anchor_lock_paths,
            account_lock_paths=alpaca_account_lock_paths,
            runtime_lock_root_paths={output_anchor_parent},
            database_identity_lock_path=identity_lock_path,
            case_sensitivity_probe_paths=case_sensitivity_probe_paths,
        )

        if not pinned_database_anchor_was_known:

            def validate_pinned_database_anchor_lock() -> None:
                revalidate_database()
                revalidate_owned_runtime_directory(
                    output_anchor_directory,
                    label="Per-user workflow lock directory",
                )

            lock_validators[pinned_database_anchor_lock_path] = validate_pinned_database_anchor_lock

        if not pinned_database_lock_was_known:
            pinned_lock_parent = prepare_owned_runtime_directory(
                pinned_database_lock_path.parent,
                label="SQLite database parent directory",
            )

            def validate_pinned_database_lock() -> None:
                revalidate_database()
                revalidate_owned_runtime_directory(
                    pinned_lock_parent,
                    label="SQLite database parent directory",
                )

            lock_validators[pinned_database_lock_path] = validate_pinned_database_lock

        if identity_lock_path is not None:
            _preflight_existing_workflow_lock_file(identity_lock_path)
            identity_lock_paths.add(identity_lock_path)

            def validate_database_identity_lock() -> None:
                revalidate_database()
                revalidate_owned_runtime_directory(
                    identity_lock_directory,
                    label="Per-user workflow lock directory",
                )

            lock_validators[identity_lock_path] = validate_database_identity_lock

        _validate_workflow_lock_tier_separation(
            account_anchor_lock_paths=account_anchor_lock_paths,
            path_local_lock_paths=path_local_lock_paths,
            identity_lock_paths=identity_lock_paths,
        )

        if not pinned_database_anchor_was_known:
            # The actual target was discovered only after the initial account
            # tier had been acquired. Drop the pre-pin set and reacquire every
            # known lock in the shared tier order while retaining the database
            # inode descriptor; no workflow work has started at this point.
            revalidate_database()
            reanchor_cleanup_failures = _release_workflow_lock_files(lock_files)
            if reanchor_cleanup_failures:
                reanchor_failure = OSError("Failed to release workflow locks while anchoring the actual SQLite target.")
                for action, cleanup_failure in reanchor_cleanup_failures:
                    reanchor_failure.add_note(f"Failed {action}: {cleanup_failure}")
                raise reanchor_failure from reanchor_cleanup_failures[0][1]
            lock_files.clear()
            acquire_lock_tiers()
        else:
            if not pinned_database_lock_was_known:
                acquire_lock(
                    pinned_database_lock_path,
                    before_and_after=lock_validators[pinned_database_lock_path],
                )
            for lock_path in sorted(identity_lock_paths, key=str):
                acquire_lock(
                    lock_path,
                    before_and_after=lock_validators[lock_path],
                )

        revalidate_database()
        revalidate_owned_runtime_directory(output_directory, label="Report output directory")
        _clear_stale_atomic_report_artifacts(
            canonical_output_path,
            directory_guard=output_directory,
        )
        revalidate_owned_runtime_directory(output_directory, label="Report output directory")
    except BaseException as exc:
        for lock_file in reversed(lock_files):
            try:
                lock_file.close()
            except BaseException as cleanup_failure:
                exc.add_note(f"Failed lock-file close after workflow-lock acquisition failure: {cleanup_failure}")
        for probe_file in reversed(tuple(preopened_lock_files.values())):
            try:
                probe_file.close()
            except BaseException as cleanup_failure:
                exc.add_note(f"Failed probe-file close after workflow-lock acquisition failure: {cleanup_failure}")
        preopened_lock_files.clear()
        if database_anchor_descriptor is not None:
            try:
                os.close(database_anchor_descriptor)
            except BaseException as cleanup_failure:
                exc.add_note(
                    f"Failed database-inode anchor close after workflow-lock acquisition failure: {cleanup_failure}"
                )
            finally:
                database_anchor_descriptor = None
        if isinstance(exc, BlockingIOError):
            protected_resources = (
                "the configured Alpaca paper account, database " if serialize_alpaca_account else "database "
            )
            raise WorkflowRunError(
                f"Another workflow is already using {protected_resources}{db_path} or output directory {output_dir}."
            ) from exc
        raise
    with activate_sqlite_runtime_file(database_guard, revalidate=revalidate_database):
        output_context_token = _ACTIVE_OUTPUT_DIRECTORY.set(output_directory)
        try:
            yield _LockedWorkflowOutputPath(
                str(canonical_output_path),
                database_path=str(pinned_database_path),
            )
        except BaseException as exc:
            for description, revalidation in (
                ("database revalidation", revalidate_database),
                (
                    "output-directory revalidation",
                    partial(
                        revalidate_owned_runtime_directory,
                        output_directory,
                        label="Report output directory",
                    ),
                ),
            ):
                try:
                    revalidation()
                except BaseException as cleanup_failure:
                    exc.add_note(f"Failed {description} while releasing workflow locks: {cleanup_failure}")
            for action, cleanup_failure in release_runtime_handles():
                exc.add_note(f"Failed {action} while releasing workflow locks: {cleanup_failure}")
            raise
        else:
            revalidation_failure: BaseException | None = None
            for description, revalidation in (
                ("database revalidation", revalidate_database),
                (
                    "output-directory revalidation",
                    partial(
                        revalidate_owned_runtime_directory,
                        output_directory,
                        label="Report output directory",
                    ),
                ),
            ):
                try:
                    revalidation()
                except BaseException as exc:
                    if revalidation_failure is None:
                        revalidation_failure = exc
                    else:
                        revalidation_failure.add_note(f"Failed {description} while releasing workflow locks: {exc}")
            if revalidation_failure is not None:
                for action, cleanup_failure in release_runtime_handles():
                    revalidation_failure.add_note(f"Failed {action} while releasing workflow locks: {cleanup_failure}")
                raise revalidation_failure
            cleanup_failures = release_runtime_handles()
            if cleanup_failures:
                failed_action, failure = cleanup_failures[0]
                failure.add_note(f"Workflow-lock cleanup action failed: {failed_action}.")
                for action, cleanup_failure in cleanup_failures[1:]:
                    failure.add_note(f"Failed {action} while releasing workflow locks: {cleanup_failure}")
                raise failure
        finally:
            _ACTIVE_OUTPUT_DIRECTORY.reset(output_context_token)


class WorkflowRunError(RuntimeError):
    """Raised when a workflow run cannot produce any usable strategy result."""


class WorkflowDeadlineExceeded(WorkflowRunError):
    """Raised before broker submission when the analytics cutoff expires."""


@dataclass(frozen=True)
class _WorkflowDeadline:
    absolute_epoch: float
    monotonic_deadline: float

    @classmethod
    def from_epoch(cls, value: int | float) -> _WorkflowDeadline:
        if isinstance(value, bool):
            raise ValueError("workflow_deadline_epoch must be a finite positive Unix timestamp.")
        try:
            absolute_epoch = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("workflow_deadline_epoch must be a finite positive Unix timestamp.") from exc
        if not math.isfinite(absolute_epoch) or absolute_epoch <= 0.0:
            raise ValueError("workflow_deadline_epoch must be a finite positive Unix timestamp.")
        return cls(
            absolute_epoch=absolute_epoch,
            monotonic_deadline=_MONOTONIC_CLOCK() + (absolute_epoch - _WALL_CLOCK()),
        )

    def remaining_seconds(self) -> float:
        return self.monotonic_deadline - _MONOTONIC_CLOCK()

    def check(self, stage: str) -> None:
        if self.remaining_seconds() <= 0.0:
            raise WorkflowDeadlineExceeded(
                "The 9:20 AM analytics deadline expired "
                f"{stage}; active asset work was rolled back and Alpaca buy submission was not started."
            )


def _check_workflow_deadline(deadline: _WorkflowDeadline | None, stage: str) -> None:
    if deadline is not None:
        deadline.check(stage)


@dataclass(frozen=True)
class AssetRunPlan:
    asset_symbol: str
    signal_symbol: str
    rebuild: bool
    start: str | None
    action: str
    start_label: str
    strategy_state_preverified: bool = False
    strategy_state_generation: int | None = None
    preverified_best_config: tuple[float, float] | None = None


@dataclass(frozen=True)
class AssetRunResult:
    workflow_idx: int
    asset_symbol: str
    signal_symbol: str
    action: str
    rows_processed: int | None
    status: str
    message: str
    workflow: str | None = None

    def as_output_row(self, *, workflow: str | None = None) -> dict[str, object]:
        row = asdict(self)
        return {
            "Workflow #": row["workflow_idx"],
            "Workflow": row["workflow"] or workflow,
            "Asset": row["asset_symbol"],
            "RSI Symbol": row["signal_symbol"],
            "Action": row["action"],
            "Rows": row["rows_processed"],
            "Status": row["status"],
            "Message": row["message"],
        }


@dataclass(frozen=True)
class AssetRunJob:
    workflow_idx: int
    asset_symbol: str
    signal_symbol: str
    workflow: str | None = None


@dataclass(frozen=True)
class PreparedAssetRun:
    job: AssetRunJob
    plan: AssetRunPlan
    data: pd.DataFrame
    asset_history: pd.DataFrame
    signal_history: pd.DataFrame
    risk_free_history: pd.DataFrame
    canonical_signal_history: pd.DataFrame | None = None
    prevalidated_asset_safe_tail_rows: dict[str, tuple[object, ...]] | None = None


@dataclass(frozen=True)
class WorkflowSideOutput:
    label: str
    universe_assets: pd.DataFrame
    buy_rsi_values: list[float]
    rsi_entry_rule: str
    asset_run_results: list[AssetRunResult]
    optimization_summary: pd.DataFrame
    curves: pd.DataFrame
    buy_signals: pd.DataFrame
    eligible_buy_signals: pd.DataFrame
    sell_signals: pd.DataFrame


class WorkflowStateCleanupError(RuntimeError):
    """Raised when an asset-local rejection cannot be rolled back safely."""


class _WorkflowMarketDataSession:
    """Run-scoped authoritative histories shared across symbols, roles, and sides."""

    def __init__(self) -> None:
        self.history_locks: dict[str, asyncio.Lock] = {}
        self.histories: dict[str, pd.DataFrame] = {}
        self.failures: dict[str, str] = {}
        # Retain the role-specific attribute names used by the pipeline while
        # making every role resolve through one canonical per-symbol snapshot.
        self.signal_locks = self.history_locks
        self.signal_histories = self.histories
        self.signal_failures = self.failures
        self.calendar_signal_histories: dict[tuple[str, str], pd.DataFrame] = {}
        self.calendar_signal_failures: dict[tuple[str, str], str] = {}
        self.risk_free_history_lock = self.history_locks.setdefault(RISK_FREE_SYMBOL, asyncio.Lock())
        self.risk_free_histories = self.histories
        self.risk_free_failures = self.failures
        self.batch_count = 0
        self.batch_attempted_symbols: set[str] = set()
        self.batch_retry_symbols: set[str] = set()
        self.individual_retry_count = 0


class _WorkflowStrategySession:
    def __init__(self, db_path: str, *, history_observation_run_id: str | None = None) -> None:
        self._db_path = db_path
        self.history_observation_run_id = history_observation_run_id or uuid.uuid4().hex
        self._conn: sqlite3.Connection | None = None
        self._runtime_guard: SqliteRuntimeGuard | None = None
        self._owner_thread_id: int | None = None
        self._data_version: int | None = None
        self._synchronized_histories: dict[str, pd.DataFrame] = {}

    def _connection(self) -> sqlite3.Connection:
        thread_id = threading.get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = thread_id
        elif self._owner_thread_id != thread_id:
            raise RuntimeError("Workflow strategy session used from multiple threads.")
        if self._conn is None:
            expected_guard = revalidate_active_sqlite_runtime_file(self._db_path)
            runtime_guard = (
                prepare_private_runtime_file(self._db_path, expected_guard=expected_guard)
                if expected_guard is not None
                else prepare_private_runtime_file(self._db_path)
            )
            revalidate_active_sqlite_runtime_file(self._db_path, runtime_guard)
            runtime_db_path = runtime_guard.database_path if runtime_guard is not None else self._db_path
            conn = sqlite3.connect(runtime_db_path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
            try:
                conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
                self._runtime_guard = revalidate_private_runtime_file(runtime_guard)
                revalidate_active_sqlite_runtime_file(self._db_path, self._runtime_guard)
            except BaseException as exc:
                try:
                    conn.close()
                except BaseException as cleanup_failure:
                    exc.add_note(f"Failed connection close after SQLite setup failure: {cleanup_failure}")
                raise
            self._conn = conn
        return self._conn

    def _discard_connection(self, conn: sqlite3.Connection) -> None:
        try:
            conn.close()
        finally:
            if self._conn is conn:
                self._conn = None
                self._runtime_guard = None
                self._owner_thread_id = None
                self._data_version = None
                self._synchronized_histories.clear()

    def _recover_failed_commit(
        self,
        conn: sqlite3.Connection,
        failure: BaseException,
    ) -> None:
        cleanup_failures: list[tuple[str, BaseException]] = []
        try:
            if conn.in_transaction:
                conn.rollback()
        except BaseException as exc:
            cleanup_failures.append(("rollback", exc))
        try:
            revalidate_private_runtime_file(self._runtime_guard)
        except BaseException as exc:
            cleanup_failures.append(("runtime-file revalidation", exc))
        try:
            self._discard_connection(conn)
        except BaseException as exc:
            cleanup_failures.append(("connection close", exc))

        for action, cleanup_failure in cleanup_failures:
            failure.add_note(f"Failed {action} after SQLite commit failure: {cleanup_failure}")

    def _recover_failed_transaction_body(
        self,
        conn: sqlite3.Connection,
        failure: BaseException,
    ) -> None:
        """Rollback a failed body, discarding the session if cleanup is unsafe."""
        cleanup_failures: list[tuple[str, BaseException]] = []
        try:
            if conn.in_transaction:
                conn.rollback()
        except BaseException as exc:
            cleanup_failures.append(("rollback", exc))
        try:
            self._runtime_guard = revalidate_private_runtime_file(self._runtime_guard)
        except BaseException as exc:
            cleanup_failures.append(("runtime-file revalidation", exc))

        if cleanup_failures:
            try:
                self._discard_connection(conn)
            except BaseException as exc:
                cleanup_failures.append(("connection close", exc))
        for action, cleanup_failure in cleanup_failures:
            failure.add_note(f"Failed {action} after SQLite transaction-body failure: {cleanup_failure}")
        if cleanup_failures and isinstance(failure, AssetMarketDataError):
            raise WorkflowStateCleanupError(
                "SQLite cleanup failed after rejecting invalid asset market data; the workflow cannot safely continue."
            ) from failure

    @contextmanager
    def immediate_transaction(self) -> sqlite3.Connection:
        conn = self._connection()
        try:
            revalidate_active_sqlite_runtime_file(self._db_path, self._runtime_guard)
            conn.execute("BEGIN IMMEDIATE")
            data_version = int(conn.execute("PRAGMA data_version").fetchone()[0])
            if self._data_version is None:
                self._data_version = data_version
            elif data_version != self._data_version:
                self._synchronized_histories.clear()
                self._data_version = data_version
            yield conn
        except BaseException as exc:
            self._recover_failed_transaction_body(conn, exc)
            raise
        try:
            self._runtime_guard = revalidate_private_runtime_file(self._runtime_guard)
            revalidate_active_sqlite_runtime_file(self._db_path, self._runtime_guard)
        except BaseException as exc:
            self._recover_failed_transaction_body(conn, exc)
            raise
        try:
            conn.commit()
        except BaseException as exc:
            # A failed commit can leave SQLite's transaction state ambiguous.
            # Never reuse that connection: roll back when possible, revalidate
            # the runtime path, then close and forget all session-local caches.
            self._recover_failed_commit(conn, exc)
            raise
        try:
            self._runtime_guard = revalidate_private_runtime_file(self._runtime_guard)
            revalidate_active_sqlite_runtime_file(self._db_path, self._runtime_guard)
        except BaseException as exc:
            try:
                self._discard_connection(conn)
            except BaseException as cleanup_failure:
                exc.add_note(
                    f"Failed connection close after SQLite runtime-file revalidation failure: {cleanup_failure}"
                )
            raise

    def presynchronized_symbols(
        self,
        authoritative_histories: dict[str, pd.DataFrame],
    ) -> set[str]:
        return {
            symbol
            for symbol, history in authoritative_histories.items()
            if self._synchronized_histories.get(symbol) is history
        }

    def mark_synchronized(
        self,
        authoritative_histories: dict[str, pd.DataFrame],
    ) -> None:
        self._synchronized_histories.update(authoritative_histories)

    def close(self) -> None:
        conn = self._conn
        if conn is not None:
            self._discard_connection(conn)
            return
        self._runtime_guard = None
        self._owner_thread_id = None
        self._data_version = None
        self._synchronized_histories.clear()


def _recover_failed_state_connection(
    conn: sqlite3.Connection,
    runtime_guard: SqliteRuntimeGuard | None,
    failure: BaseException,
    *,
    failure_context: str,
) -> None:
    cleanup_failures: list[tuple[str, BaseException]] = []
    try:
        conn.rollback()
    except BaseException as exc:
        cleanup_failures.append(("rollback", exc))
    try:
        revalidate_private_runtime_file(runtime_guard)
    except BaseException as exc:
        cleanup_failures.append(("runtime-file revalidation", exc))
    try:
        conn.close()
    except BaseException as exc:
        cleanup_failures.append(("connection close", exc))

    for action, cleanup_failure in cleanup_failures:
        failure.add_note(f"Failed {action} after {failure_context}: {cleanup_failure}")
    if cleanup_failures and isinstance(failure, AssetMarketDataError):
        raise WorkflowStateCleanupError(
            "SQLite cleanup failed after rejecting invalid asset market data; the workflow cannot safely continue."
        ) from failure


@contextmanager
def _state_connection(
    db_path: str,
    *,
    immediate: bool = False,
    defer_body_commits: bool = False,
) -> sqlite3.Connection:
    expected_guard = revalidate_active_sqlite_runtime_file(db_path)
    runtime_guard = (
        prepare_private_runtime_file(db_path, expected_guard=expected_guard)
        if expected_guard is not None
        else prepare_private_runtime_file(db_path)
    )
    revalidate_active_sqlite_runtime_file(db_path, runtime_guard)
    runtime_db_path = runtime_guard.database_path if runtime_guard is not None else db_path
    connect_kwargs: dict[str, object] = {"timeout": SQLITE_BUSY_TIMEOUT_MS / 1000}
    if defer_body_commits:
        connect_kwargs["factory"] = _DeferredCommitSqliteConnection
    conn = sqlite3.connect(runtime_db_path, **connect_kwargs)
    try:
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        runtime_guard = revalidate_private_runtime_file(runtime_guard)
        revalidate_active_sqlite_runtime_file(db_path, runtime_guard)
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
    except BaseException as exc:
        _recover_failed_state_connection(
            conn,
            runtime_guard,
            exc,
            failure_context="SQLite transaction-body failure",
        )
        raise
    try:
        runtime_guard = revalidate_private_runtime_file(runtime_guard)
        revalidate_active_sqlite_runtime_file(db_path, runtime_guard)
    except BaseException as exc:
        _recover_failed_state_connection(
            conn,
            runtime_guard,
            exc,
            failure_context="pre-commit SQLite runtime-file revalidation failure",
        )
        raise
    try:
        if defer_body_commits:
            # Bypass the opt-in no-op override exactly once, after both the
            # transaction body and runtime-file identity have been validated.
            sqlite3.Connection.commit(conn)
        else:
            conn.commit()
    except BaseException as exc:
        _recover_failed_state_connection(
            conn,
            runtime_guard,
            exc,
            failure_context="SQLite commit failure",
        )
        raise
    try:
        runtime_guard = revalidate_private_runtime_file(runtime_guard)
        revalidate_active_sqlite_runtime_file(db_path, runtime_guard)
    except BaseException as exc:
        try:
            conn.close()
        except BaseException as cleanup_failure:
            exc.add_note(f"Failed connection close after SQLite runtime-file revalidation failure: {cleanup_failure}")
        raise
    else:
        conn.close()


def _initialize_state_db(db_path: str) -> None:
    with _state_connection(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        init_state_db(conn, commit=False)


async def _timed_run_blocking(
    phase_timings: WorkflowPhaseTimings | None,
    phase: WorkflowPhase,
    func: Callable[..., Any],
    /,
    *args: object,
    executor: Executor | None = None,
    **kwargs: object,
) -> Any:
    if phase_timings is None:
        return await _run_blocking(executor, func, *args, **kwargs)
    started = time.perf_counter()
    phase_timings.begin(phase, started)
    try:
        return await _run_blocking(executor, func, *args, **kwargs)
    finally:
        phase_timings.end(phase, time.perf_counter())


async def _run_blocking(
    executor: Executor | None,
    func: Callable[..., Any],
    /,
    *args: object,
    **kwargs: object,
) -> Any:
    owned_executor: ThreadPoolExecutor | None = None
    context = copy_context()
    call = partial(context.run, func, *args, **kwargs)
    if executor is None:
        # Avoid the event loop's default executor: asyncio.run() shuts that
        # executor down after the main coroutine returns, and that shutdown's
        # completion notification is outside the timer-polled wait below.
        owned_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="workflow-blocking",
        )
        executor = owned_executor
    future = asyncio.get_running_loop().run_in_executor(executor, call)

    async def wait_for_worker() -> Any:
        # Keep a timer registered while waiting. Some selector event loops can
        # miss an executor completion wakeup, but the timer still gives the
        # loop a bounded opportunity to process the already-queued callback.
        while not future.done():
            await asyncio.wait((future,), timeout=0.05)
        return future.result()

    try:
        try:
            return await wait_for_worker()
        except asyncio.CancelledError as cancelled:
            # Cancelling an asyncio Future cannot stop a running worker thread.
            # Do not let callers unwind process/file locks while that worker can
            # still mutate SQLite, broker state, or report files. Repeated
            # cancellation requests remain pending until the worker reaches its
            # boundary. Every external provider called through this helper must
            # therefore enforce its own bounded request timeout.
            while not future.done():
                try:
                    await wait_for_worker()
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if future.done() and not future.cancelled():
                with suppress(BaseException):
                    future.result()
            raise cancelled
    finally:
        if owned_executor is not None:
            owned_executor.shutdown(wait=True)


def _load_or_refresh_workflow_assets_for_db(
    db_path: str,
    universe_cfg: UniverseConfig,
) -> dict[str, pd.DataFrame]:
    workflow_asset_groups = determine_workflow_asset_groups(universe_cfg)
    # pandas' sqlite fallback commits table creation and insertion internally.
    # Hold an explicit transaction on a connection that defers those helper
    # commits so `_state_connection` can validate the database identity first.
    with _state_connection(db_path, immediate=True, defer_body_commits=True) as conn:
        save_workflow_assets(conn, _combined_workflow_assets(workflow_asset_groups))
    return workflow_asset_groups


def _combined_workflow_assets(workflow_asset_groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in workflow_asset_groups.values()]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    source = next((frame for frame in frames if frame.attrs), None)
    if source is not None:
        out.attrs.update(source.attrs)
    counts: dict[str, object] = {}
    for frame in frames:
        counts.update(frame.attrs.get("universe_counts", {}))
    out.attrs["universe_counts"] = counts
    out.attrs["universe_title"] = "Executable Leveraged ETFs/ETNs From Merged Universe"
    return out


def _with_workflow_column(df: pd.DataFrame, workflow_label: str) -> pd.DataFrame:
    out = df.copy()
    if "Workflow" in out.columns:
        out["Workflow"] = out["Workflow"].where(out["Workflow"].notna(), workflow_label)
    else:
        out.insert(0, "Workflow", workflow_label)
    return out


def _persist_current_alpaca_state(
    *,
    db_path: str,
    output_path: Path,
    publication: _AlpacaSnapshotPublication | None = None,
    alpaca_cfg: AlpacaOrderConfig | None = None,
) -> None:
    owns_publication = publication is None
    if publication is None:
        _prepare_workflow_output_directory_for_publication(output_path)
        publication = _begin_alpaca_snapshot_publication(
            output_path,
            snapshot_kind="current-state",
        )
    failures: list[tuple[str, BaseException]] = []
    loaded_frames: list[tuple[str, pd.DataFrame, str]] = []
    snapshot_completed = False
    try:
        with _state_connection(db_path) as conn:
            # Both reports describe one broker-state generation. A plain SELECT
            # does not start a transaction in sqlite3's legacy transaction mode,
            # so establish the read snapshot explicitly before either loader.
            conn.execute("BEGIN")
            for label, loader, filename in (
                (
                    "managed positions",
                    _load_alpaca_managed_positions_for_db,
                    "managed_positions.csv",
                ),
                (
                    "realized P/L",
                    _load_alpaca_realized_pnl_for_db,
                    "alpaca_realized_pnl.csv",
                ),
            ):
                try:
                    frame = loader(db_path, connection=conn)
                except BaseException as exc:
                    failures.append((label, exc))
                else:
                    loaded_frames.append(
                        (
                            label,
                            _safe_alpaca_workflow_result_messages(
                                frame,
                                alpaca_cfg=alpaca_cfg,
                            ),
                            filename,
                        )
                    )
        snapshot_completed = True
    except BaseException as exc:
        failures.append(("SQLite read snapshot", exc))

    if snapshot_completed:
        for label, frame, filename in loaded_frames:
            try:
                _publish_alpaca_snapshot_csv(
                    publication,
                    frame,
                    filename=filename,
                    index=False,
                )
            except BaseException as exc:
                failures.append((label, exc))
    if not failures and owns_publication:
        try:
            _commit_alpaca_snapshot_publication(publication)
        except BaseException as exc:
            failures.append(("snapshot manifest", exc))
    if failures:
        first_label, first_failure = failures[0]
        _add_safe_alpaca_workflow_note(
            first_failure,
            f"Failed to publish current Alpaca {first_label} state.",
            alpaca_cfg=alpaca_cfg,
        )
        for label, failure in failures[1:]:
            detail = _best_effort_alpaca_exception_diagnostic(failure, alpaca_cfg=alpaca_cfg)
            _add_safe_alpaca_workflow_note(
                first_failure,
                f"Also failed to publish current Alpaca {label} state: {detail}",
                alpaca_cfg=alpaca_cfg,
            )
        raise _sanitize_alpaca_workflow_exception(first_failure, alpaca_cfg=alpaca_cfg)


def _prepare_workflow_output_directory_for_publication(output_path: Path) -> None:
    """Prepare standalone output, or revalidate the directory pinned by a workflow."""
    active_output_directory = _ACTIVE_OUTPUT_DIRECTORY.get()
    absolute_output_path = _case_preserving_workflow_path(output_path)
    if active_output_directory is not None:
        if absolute_output_path != active_output_directory.path:
            raise PermissionError(
                f"Report output directory {absolute_output_path} is outside the active workflow output directory."
            )
        revalidate_owned_runtime_directory(
            active_output_directory,
            label="Report output directory",
        )
        return
    prepare_owned_runtime_directory(
        output_path,
        label="Report output directory",
    )


def _alpaca_snapshot_manifest_frame(
    publication: _AlpacaSnapshotPublication,
    *,
    status: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Schema Version": _ALPACA_SNAPSHOT_SCHEMA_VERSION,
                "Generation": publication.generation,
                "Status": status,
                "Snapshot Kind": publication.snapshot_kind,
                "Filename": filename,
                "SHA256": publication.digests.get(filename, ""),
            }
            for filename in publication.expected_filenames
        ]
    )


def _begin_alpaca_snapshot_publication(
    output_path: Path,
    *,
    snapshot_kind: str,
) -> _AlpacaSnapshotPublication:
    """Invalidate the prior manifest before replacing any covered broker CSV."""
    try:
        expected_filenames = _ALPACA_SNAPSHOT_FILENAMES_BY_KIND[snapshot_kind]
    except KeyError as exc:
        raise ValueError(f"Unknown Alpaca snapshot kind: {snapshot_kind}") from exc
    _prepare_workflow_output_directory_for_publication(output_path)
    publication = _AlpacaSnapshotPublication(
        output_path=output_path,
        generation=uuid.uuid4().hex,
        snapshot_kind=snapshot_kind,
        expected_filenames=expected_filenames,
        digests={},
    )
    _atomic_to_csv(
        _alpaca_snapshot_manifest_frame(publication, status="publishing"),
        output_path / _ALPACA_SNAPSHOT_MANIFEST_FILENAME,
        index=False,
    )
    return publication


def _publish_alpaca_snapshot_csv(
    publication: _AlpacaSnapshotPublication,
    frame: pd.DataFrame,
    *,
    filename: str,
    index: bool,
) -> None:
    if filename not in publication.expected_filenames:
        raise ValueError(f"Alpaca snapshot kind {publication.snapshot_kind!r} does not cover {filename!r}.")
    # A replacement can become visible before a later durability or directory
    # revalidation failure is reported. Drop the prior digest first so that a
    # failed republish can never leave this generation eligible for commit with
    # a digest describing the superseded contents.
    publication.digests.pop(filename, None)
    digest = _atomic_to_csv(
        frame,
        publication.output_path / filename,
        index=index,
    )
    if not isinstance(digest, str) or len(digest) != 64:
        raise OSError(f"Alpaca snapshot publication did not return a SHA-256 digest for {filename}.")
    publication.digests[filename] = digest


def _commit_alpaca_snapshot_publication(publication: _AlpacaSnapshotPublication) -> None:
    missing = [filename for filename in publication.expected_filenames if filename not in publication.digests]
    if missing:
        raise OSError("Cannot commit an incomplete Alpaca snapshot; missing " + ", ".join(missing))
    _atomic_to_csv(
        _alpaca_snapshot_manifest_frame(publication, status="committed"),
        publication.output_path / _ALPACA_SNAPSHOT_MANIFEST_FILENAME,
        index=False,
    )


def _best_effort_alpaca_exception_diagnostic(
    failure: BaseException,
    *,
    alpaca_cfg: AlpacaOrderConfig | None,
) -> str:
    """Return a bounded diagnostic without allowing exception rendering to mask broker state."""
    try:
        diagnostic = _alpaca_exception_diagnostic(failure, cfg=alpaca_cfg)
    except BaseException:
        diagnostic = ""
    if diagnostic:
        return diagnostic
    try:
        # Exception class names are output too: sanitize them against the same
        # credential set instead of appending an unchecked fallback.
        return _alpaca_exception_diagnostic(RuntimeError(type(failure).__name__), cfg=alpaca_cfg)
    except BaseException:
        return ""


def _safe_alpaca_workflow_text(
    value: str | BaseException,
    *,
    alpaca_cfg: AlpacaOrderConfig | None,
) -> str:
    """Sanitize a fully composed workflow broker diagnostic."""
    try:
        diagnostic = value if isinstance(value, BaseException) else RuntimeError(value)
        return _alpaca_exception_diagnostic(diagnostic, cfg=alpaca_cfg)
    except BaseException:
        return ""


def _safe_alpaca_workflow_result_messages(
    results: pd.DataFrame,
    *,
    alpaca_cfg: AlpacaOrderConfig | None,
) -> pd.DataFrame:
    """Return broker results whose final human-readable messages are credential-safe."""
    safe_results = results.copy()
    for column in ("Message", "message", "Notes", "notes", "buy_causality_quarantine"):
        if column in safe_results:
            safe_results[column] = safe_results[column].map(
                lambda value: (
                    _safe_alpaca_workflow_text(value, alpaca_cfg=alpaca_cfg) if isinstance(value, str) else value
                )
            )
    return safe_results


def _sanitize_alpaca_workflow_exception(
    failure: BaseException,
    *,
    alpaca_cfg: AlpacaOrderConfig | None,
) -> BaseException:
    """Sanitize an exception immediately before it crosses a workflow boundary."""
    return _alpaca_public_exception(
        failure,
        cfg=alpaca_cfg,
        diagnostic_columns=("Message", "message", "Notes", "notes", "buy_causality_quarantine"),
        preserve_typed_result_identity=True,
    )


def _run_alpaca_workflow_diagnostic_boundary(
    operation: Callable[[], Any],
    *,
    alpaca_cfg: AlpacaOrderConfig,
) -> Any:
    """Run one synchronous workflow boundary with a complete credential fence."""
    boundary_sensitive_values = _alpaca_config_sensitive_values(alpaca_cfg)
    public_failure: BaseException | None = None
    suppress_chain = False
    with _alpaca_public_durable_diagnostic_scope(alpaca_cfg, boundary_sensitive_values):
        try:
            return operation()
        except BaseException as exc:
            suppress_chain = _alpaca_public_exception_cause(exc, cfg=alpaca_cfg) is None
            public_failure = _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg) if suppress_chain else exc

    assert public_failure is not None
    if suppress_chain:
        raise public_failure from None
    raise public_failure


def _safe_alpaca_reconciliation_error(
    message: str,
    results: pd.DataFrame,
    *,
    alpaca_cfg: AlpacaOrderConfig | None,
) -> AlpacaReconciliationError:
    """Build a typed reconciliation failure after final composition redaction."""
    return AlpacaReconciliationError(
        _safe_alpaca_workflow_text(message, alpaca_cfg=alpaca_cfg),
        _safe_alpaca_workflow_result_messages(results, alpaca_cfg=alpaca_cfg),
    )


def _add_safe_alpaca_workflow_note(
    failure: BaseException,
    note: str,
    *,
    alpaca_cfg: AlpacaOrderConfig | None,
) -> None:
    """Attach a note only after scanning its complete fixed and variable text."""
    safe_note = _safe_alpaca_workflow_text(note, alpaca_cfg=alpaca_cfg)
    if safe_note:
        failure.add_note(safe_note)


def _alpaca_reconciliation_followup_error(
    reconciliation_results: pd.DataFrame,
    *,
    action: str,
    failure: BaseException,
    alpaca_cfg: AlpacaOrderConfig | None = None,
) -> AlpacaReconciliationError:
    failure_message = _best_effort_alpaca_exception_diagnostic(failure, alpaca_cfg=alpaca_cfg)
    diagnostic = pd.DataFrame(
        [
            {
                "Position ID": None,
                "Workflow": None,
                "Asset": None,
                "Action": "audit",
                "Status": "error",
                "Buy Client Order ID": None,
                "Sell Client Order ID": None,
                "Qty": None,
                "Limit Price": None,
                "Alpaca Order ID": None,
                "Message": f"Failed to {action} after broker reconciliation completed: {failure_message}",
            }
        ]
    )
    return _safe_alpaca_reconciliation_error(
        f"Alpaca reconciliation completed, but failed to {action}: {failure_message}",
        pd.concat([reconciliation_results, diagnostic], ignore_index=True),
        alpaca_cfg=alpaca_cfg,
    )


def _alpaca_reconciliation_failure_results(
    reconciliation_results: pd.DataFrame,
    *,
    action: str,
    failure: BaseException,
    alpaca_cfg: AlpacaOrderConfig,
) -> pd.DataFrame:
    """Append a CSV-safe diagnostic when reconciliation failed outside its typed error."""
    failure_message = _best_effort_alpaca_exception_diagnostic(failure, alpaca_cfg=alpaca_cfg)
    diagnostic = pd.DataFrame(
        [
            {
                "Position ID": None,
                "Workflow": None,
                "Asset": None,
                "Action": "audit",
                "Status": "error",
                "Buy Client Order ID": None,
                "Sell Client Order ID": None,
                "Qty": None,
                "Limit Price": None,
                "Alpaca Order ID": None,
                "Message": f"Failed to {action}: {failure_message}",
            }
        ]
    )
    return _safe_alpaca_workflow_result_messages(
        pd.concat([reconciliation_results, diagnostic], ignore_index=True),
        alpaca_cfg=alpaca_cfg,
    )


def _persist_alpaca_reconciliation_snapshot(
    reconciliation_results: pd.DataFrame,
    *,
    db_path: str,
    output_dir: str,
    publication: _AlpacaSnapshotPublication,
    alpaca_cfg: AlpacaOrderConfig | None = None,
) -> None:
    if alpaca_cfg is None:
        return _persist_alpaca_reconciliation_snapshot_impl(
            reconciliation_results,
            db_path=db_path,
            output_dir=output_dir,
            publication=publication,
            alpaca_cfg=None,
        )
    return _run_alpaca_workflow_diagnostic_boundary(
        lambda: _persist_alpaca_reconciliation_snapshot_impl(
            reconciliation_results,
            db_path=db_path,
            output_dir=output_dir,
            publication=publication,
            alpaca_cfg=alpaca_cfg,
        ),
        alpaca_cfg=alpaca_cfg,
    )


def _persist_alpaca_reconciliation_snapshot_impl(
    reconciliation_results: pd.DataFrame,
    *,
    db_path: str,
    output_dir: str,
    publication: _AlpacaSnapshotPublication,
    alpaca_cfg: AlpacaOrderConfig | None = None,
) -> None:
    """Publish broker mutations before unrelated workflow work can fail."""
    reconciliation_results = _safe_alpaca_workflow_result_messages(
        reconciliation_results,
        alpaca_cfg=alpaca_cfg,
    )
    output_path = Path(output_dir)
    sell_results = (
        reconciliation_results[reconciliation_results["Action"].eq("sell")]
        if "Action" in reconciliation_results
        else pd.DataFrame()
    )
    try:
        _prepare_workflow_output_directory_for_publication(output_path)
    except BaseException as exc:
        raise _alpaca_reconciliation_followup_error(
            reconciliation_results,
            action="prepare the Alpaca audit output directory",
            failure=exc,
            alpaca_cfg=alpaca_cfg,
        ) from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)
    if publication.output_path != output_path or publication.snapshot_kind != "reconciliation":
        exc = ValueError("Alpaca reconciliation snapshot publication does not match its output directory and kind.")
        raise _alpaca_reconciliation_followup_error(
            reconciliation_results,
            action="validate the active Alpaca snapshot publication",
            failure=exc,
            alpaca_cfg=alpaca_cfg,
        ) from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)

    failures: list[tuple[str, BaseException]] = []
    for action, publish in (
        (
            "publish the reconciliation audit",
            lambda: _publish_alpaca_snapshot_csv(
                publication,
                reconciliation_results,
                filename="alpaca_reconciliation_results.csv",
                index=False,
            ),
        ),
        (
            "publish the sell-order audit",
            lambda: _publish_alpaca_snapshot_csv(
                publication,
                sell_results,
                filename="alpaca_sell_order_results.csv",
                index=False,
            ),
        ),
        (
            "publish current managed-position and realized-P/L state",
            lambda: _persist_current_alpaca_state(
                db_path=db_path,
                output_path=output_path,
                publication=publication,
                alpaca_cfg=alpaca_cfg,
            ),
        ),
    ):
        try:
            publish()
        except BaseException as exc:
            failures.append((action, exc))
    if not failures:
        try:
            _commit_alpaca_snapshot_publication(publication)
        except BaseException as exc:
            failures.append(("commit the Alpaca snapshot manifest", exc))
    if failures:
        first_action, first_failure = failures[0]
        publication_error = _alpaca_reconciliation_followup_error(
            reconciliation_results,
            action=first_action,
            failure=first_failure,
            alpaca_cfg=alpaca_cfg,
        )
        for action, failure in failures[1:]:
            detail = _best_effort_alpaca_exception_diagnostic(failure, alpaca_cfg=alpaca_cfg)
            _add_safe_alpaca_workflow_note(
                publication_error,
                f"Also failed to {action}: {detail}",
                alpaca_cfg=alpaca_cfg,
            )
        try:
            _publish_alpaca_snapshot_csv(
                publication,
                publication_error.results,
                filename="alpaca_reconciliation_results.csv",
                index=False,
            )
        except BaseException as audit_failure:
            detail = _best_effort_alpaca_exception_diagnostic(audit_failure, alpaca_cfg=alpaca_cfg)
            _add_safe_alpaca_workflow_note(
                publication_error,
                f"Failed to publish the augmented reconciliation failure audit: {detail}",
                alpaca_cfg=alpaca_cfg,
            )
        raise publication_error from _alpaca_public_exception_cause(first_failure, cfg=alpaca_cfg)


def _finish_alpaca_workflow_snapshot(
    publication: _AlpacaSnapshotPublication,
    order_results: pd.DataFrame,
    reconciliation_results: pd.DataFrame,
    *,
    db_path: str,
    alpaca_cfg: AlpacaOrderConfig | None = None,
) -> list[tuple[str, BaseException]]:
    """Complete one workflow generation with its buy audit and current state."""
    order_results = _safe_alpaca_workflow_result_messages(
        order_results,
        alpaca_cfg=alpaca_cfg,
    )
    reconciliation_results = _safe_alpaca_workflow_result_messages(
        reconciliation_results,
        alpaca_cfg=alpaca_cfg,
    )
    sell_results = (
        reconciliation_results[reconciliation_results["Action"].eq("sell")]
        if "Action" in reconciliation_results
        else pd.DataFrame()
    )
    failures: list[tuple[str, BaseException]] = []
    for action, publish in (
        (
            "publish the Alpaca buy audit",
            lambda: _publish_alpaca_snapshot_csv(
                publication,
                order_results,
                filename="alpaca_order_results.csv",
                index=False,
            ),
        ),
        (
            "publish the Alpaca reconciliation audit",
            lambda: _publish_alpaca_snapshot_csv(
                publication,
                reconciliation_results,
                filename="alpaca_reconciliation_results.csv",
                index=False,
            ),
        ),
        (
            "publish the Alpaca sell-order audit",
            lambda: _publish_alpaca_snapshot_csv(
                publication,
                sell_results,
                filename="alpaca_sell_order_results.csv",
                index=False,
            ),
        ),
        (
            "publish current Alpaca state",
            lambda: _persist_current_alpaca_state(
                db_path=db_path,
                output_path=publication.output_path,
                publication=publication,
                alpaca_cfg=alpaca_cfg,
            ),
        ),
    ):
        try:
            publish()
        except BaseException as exc:
            failures.append((action, exc))
    if not failures:
        try:
            _commit_alpaca_snapshot_publication(publication)
        except BaseException as exc:
            failures.append(("commit the Alpaca workflow snapshot", exc))
    return failures


def _persist_alpaca_reconciliation_failure(
    exc: AlpacaReconciliationError,
    *,
    db_path: str,
    output_dir: str,
    publication: _AlpacaSnapshotPublication,
    reporter: WorkflowReporter,
    alpaca_cfg: AlpacaOrderConfig,
) -> None:
    _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg)
    _persist_alpaca_reconciliation_snapshot(
        exc.results,
        db_path=db_path,
        output_dir=output_dir,
        publication=publication,
        alpaca_cfg=alpaca_cfg,
    )
    reporter.reconciliation(exc.results)


def _reconcile_alpaca_managed_positions_for_db(
    db_path: str,
    alpaca_cfg: AlpacaOrderConfig,
) -> pd.DataFrame:
    return _run_alpaca_workflow_diagnostic_boundary(
        lambda: _reconcile_alpaca_managed_positions_for_db_impl(db_path, alpaca_cfg),
        alpaca_cfg=alpaca_cfg,
    )


def _reconcile_alpaca_managed_positions_for_db_impl(
    db_path: str,
    alpaca_cfg: AlpacaOrderConfig,
) -> pd.DataFrame:
    broker_results: pd.DataFrame | None = None
    completed_results: pd.DataFrame | None = None
    try:
        with _state_connection(db_path) as conn:
            migrations: dict[str, str] = {}
            migration_commit_completed = False
            try:
                revalidate_active_sqlite_runtime_file(db_path)
                migrations = migrate_alpaca_managed_position_symbols(
                    conn,
                    alpaca_cfg,
                    include_closed=False,
                )
                if migrations:
                    # Migration success is included in a durable audit even
                    # when later protective reconciliation fails. Make that
                    # boundary explicit: the migration helper deliberately
                    # uses savepoints/commit=False, so allowing a later typed
                    # failure to escape this context would otherwise roll the
                    # reported rename back.
                    revalidate_active_sqlite_runtime_file(db_path)
                    conn.commit()
                    migration_commit_completed = True
                    revalidate_active_sqlite_runtime_file(db_path)
            except BaseException as exc:
                diagnostic_notes: list[str] = []
                migration_detail = _best_effort_alpaca_exception_diagnostic(exc, alpaca_cfg=alpaca_cfg)

                failure_context = (
                    "managed Alpaca symbol migration committed, but post-commit validation failed before reconciliation"
                    if migration_commit_completed
                    else "managed Alpaca symbol migration failed before reconciliation"
                )
                generic_error_row = {
                    "Position ID": None,
                    "Workflow": None,
                    "Asset": None,
                    "Action": "symbol",
                    "Status": "error",
                    "Buy Client Order ID": None,
                    "Sell Client Order ID": None,
                    "Qty": None,
                    "Limit Price": None,
                    "Alpaca Order ID": None,
                    "Message": f"{failure_context}: {migration_detail}",
                }
                try:
                    active_positions = load_alpaca_managed_positions(conn, active_only=True)
                    migration_error_rows = [
                        {
                            **generic_error_row,
                            "Position ID": int(position["id"]),
                            "Workflow": position.get("workflow"),
                            "Asset": position.get("symbol"),
                            "Buy Client Order ID": position.get("buy_client_order_id"),
                            "Sell Client Order ID": position.get("sell_client_order_id"),
                            "Qty": position.get("filled_qty"),
                            "Limit Price": position.get("target_sell_price"),
                            "Alpaca Order ID": position.get("sell_alpaca_order_id"),
                        }
                        for position in active_positions.to_dict("records")
                    ]
                except BaseException as diagnostic_exc:
                    diagnostic_detail = _best_effort_alpaca_exception_diagnostic(
                        diagnostic_exc,
                        alpaca_cfg=alpaca_cfg,
                    )
                    generic_error_row["Message"] += (
                        f"; active managed positions could not be loaded for the failure audit: {diagnostic_detail}"
                    )
                    diagnostic_notes.append(
                        "Failed to load active managed positions while constructing the managed "
                        f"Alpaca migration failure audit: {diagnostic_detail}."
                    )
                    migration_error_rows = []

                migration_errors = pd.DataFrame(migration_error_rows or [generic_error_row])
                migration_error = _safe_alpaca_reconciliation_error(
                    "managed Alpaca symbol migration failed before protective reconciliation",
                    migration_errors,
                    alpaca_cfg=alpaca_cfg,
                )
                for note in diagnostic_notes:
                    _add_safe_alpaca_workflow_note(
                        migration_error,
                        note,
                        alpaca_cfg=alpaca_cfg,
                    )
                raise migration_error from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)
            migration_rows = []
            if migrations:
                try:
                    active_positions = load_alpaca_managed_positions(conn, active_only=True)
                    for prior_symbol, current_symbol in migrations.items():
                        matches = active_positions[
                            active_positions["symbol"].astype(str).str.upper().eq(current_symbol.upper())
                        ]
                        if matches.empty:
                            continue
                        position = matches.iloc[0]
                        migration_rows.append(
                            {
                                "Position ID": int(position["id"]),
                                "Workflow": position.get("workflow"),
                                "Asset": current_symbol,
                                "Action": "symbol",
                                "Status": "symbol_migrated",
                                "Buy Client Order ID": position.get("buy_client_order_id"),
                                "Sell Client Order ID": position.get("sell_client_order_id"),
                                "Qty": position.get("filled_qty"),
                                "Limit Price": position.get("target_sell_price"),
                                "Alpaca Order ID": position.get("sell_alpaca_order_id"),
                                "Message": (
                                    f"managed Alpaca symbol migrated from {prior_symbol} to {current_symbol} "
                                    "using stable asset identity"
                                ),
                            }
                        )
                except BaseException as exc:
                    committed_migrations = pd.DataFrame(
                        [
                            {
                                "Position ID": None,
                                "Workflow": None,
                                "Asset": current_symbol,
                                "Action": "symbol",
                                "Status": "symbol_migrated",
                                "Buy Client Order ID": None,
                                "Sell Client Order ID": None,
                                "Qty": None,
                                "Limit Price": None,
                                "Alpaca Order ID": None,
                                "Message": (
                                    f"managed Alpaca symbol migrated from {prior_symbol} to {current_symbol} "
                                    "using stable asset identity; detailed position audit could not be loaded"
                                ),
                            }
                            for prior_symbol, current_symbol in migrations.items()
                        ]
                    )
                    failed_results = _alpaca_reconciliation_failure_results(
                        committed_migrations,
                        action="construct the committed managed Alpaca symbol-migration audit",
                        failure=exc,
                        alpaca_cfg=alpaca_cfg,
                    )
                    raise _safe_alpaca_reconciliation_error(
                        "managed Alpaca symbol migration committed, but its audit could not be constructed",
                        failed_results,
                        alpaca_cfg=alpaca_cfg,
                    ) from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)
            migration_results = pd.DataFrame(migration_rows)
            reconciliation_invoked = False
            try:
                revalidate_active_sqlite_runtime_file(db_path)
                reconciliation_invoked = True
                reconciliation = reconcile_alpaca_managed_positions(
                    conn,
                    alpaca_cfg,
                    migrate_closed_symbols=True,
                )
            except AlpacaReconciliationError as exc:
                if migration_results.empty:
                    _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg)
                    raise
                combined_results = pd.concat([migration_results, exc.results], ignore_index=True)
                failure_message = _best_effort_alpaca_exception_diagnostic(
                    exc,
                    alpaca_cfg=alpaca_cfg,
                )
                raise _safe_alpaca_reconciliation_error(
                    failure_message,
                    combined_results,
                    alpaca_cfg=alpaca_cfg,
                ) from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)
            except BaseException as exc:
                action = (
                    "complete managed Alpaca reconciliation after broker work may have begun"
                    if reconciliation_invoked
                    else "start managed Alpaca reconciliation before broker work"
                )
                failed_results = _alpaca_reconciliation_failure_results(
                    migration_results,
                    action=action,
                    failure=exc,
                    alpaca_cfg=alpaca_cfg,
                )
                raise _safe_alpaca_reconciliation_error(
                    "managed Alpaca reconciliation ended before its broker outcome was available",
                    failed_results,
                    alpaca_cfg=alpaca_cfg,
                ) from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)
            broker_results = reconciliation
            completed_results = (
                reconciliation
                if migration_results.empty
                else pd.concat([migration_results, reconciliation], ignore_index=True)
            )
    except BaseException as exc:
        if broker_results is None:
            raise
        finalization_detail = _best_effort_alpaca_exception_diagnostic(exc, alpaca_cfg=alpaca_cfg)
        failure_row = pd.DataFrame(
            [
                {
                    "Position ID": None,
                    "Workflow": None,
                    "Asset": None,
                    "Action": "reconcile",
                    "Status": "error",
                    "Buy Client Order ID": None,
                    "Sell Client Order ID": None,
                    "Qty": None,
                    "Limit Price": None,
                    "Alpaca Order ID": None,
                    "Message": (
                        "managed Alpaca reconciliation completed broker work, but SQLite "
                        f"transaction finalization failed: {finalization_detail}"
                    ),
                }
            ]
        )
        auditable_results = pd.concat(
            [completed_results if completed_results is not None else broker_results, failure_row],
            ignore_index=True,
        )
        raise _safe_alpaca_reconciliation_error(
            "managed Alpaca reconciliation state finalization failed after broker work",
            auditable_results,
            alpaca_cfg=alpaca_cfg,
        ) from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)
    assert completed_results is not None
    return _safe_alpaca_workflow_result_messages(
        completed_results,
        alpaca_cfg=alpaca_cfg,
    )


def _prepare_asset_run(
    db_path: str,
    mode: str,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rsi_entry_rule: str = "lower",
    strategy_state_verification: str = "trusted",
) -> AssetRunPlan:
    # Verify once on the parallel preparation workers. The transaction later
    # fences this result with the captured global generation; a stale result
    # rebuilds safely instead of running a second verification pass.
    with _state_connection(db_path) as conn:
        rebuild_asset = mode == "rebuild"
        state_preverified = False
        verified_generation: int | None = None
        preverified_best_config: tuple[float, float] | None = None
        if not rebuild_asset:
            conn.execute("BEGIN")
            verified_generation = strategy_state_generation(conn)
            state_preverified = strategy_state_matches_config(
                conn,
                asset_symbol,
                signal_symbol,
                base_cfg,
                buy_rsi_values,
                profit_target_values,
                rsi_entry_rule,
                strategy_state_verification=strategy_state_verification,
            )
            if state_preverified:
                best_summary = load_best_strategy_summary(
                    conn,
                    asset_symbol,
                    signal_symbol,
                    rsi_entry_rule,
                    strategy_state_preverified=True,
                )
                if best_summary is not None:
                    candidate_best_config = (
                        float(best_summary["buy_rsi"]),
                        float(best_summary["profit_target_multiple"]),
                    )
                    retained_curve = load_complete_strategy_equity_curve(
                        conn,
                        asset_symbol,
                        signal_symbol,
                        *candidate_best_config,
                        rsi_period=base_cfg.rsi_period,
                        rsi_entry_rule=rsi_entry_rule,
                        allow_unbound_backtest_config=True,
                        strategy_state_preverified=True,
                    )
                    if retained_curve is not None:
                        preverified_best_config = candidate_best_config
                    else:
                        state_preverified = False
                else:
                    state_preverified = False
            rebuild_asset = not state_preverified
        # Fetch canonical history even in update mode.  Auto-adjusted asset and
        # benchmark series can revise old sessions; the state layer compares the
        # complete input before deciding whether a compact resume is still safe.
        start: str | None = None

    action = "Rebuilding" if rebuild_asset else "Updating"
    start_label = start if start is not None else "earliest overlapping history"
    return AssetRunPlan(
        asset_symbol=asset_symbol,
        signal_symbol=signal_symbol,
        rebuild=rebuild_asset,
        start=start,
        action=action,
        start_label=start_label,
        strategy_state_preverified=state_preverified,
        strategy_state_generation=verified_generation,
        preverified_best_config=preverified_best_config,
    )


def _prepare_asset_safe_tail_rows(
    db_path: str,
    plan: AssetRunPlan,
    asset_history: pd.DataFrame,
) -> dict[str, tuple[object, ...]] | None:
    """Classify one authoritative asset history on a parallel read snapshot."""
    if not plan.strategy_state_preverified or plan.strategy_state_generation is None:
        return None
    with _state_connection(db_path) as conn:
        conn.execute("BEGIN")
        if strategy_state_generation(conn) != plan.strategy_state_generation:
            return None
        return _safe_market_history_tail_rows(
            conn,
            asset_history,
            plan.asset_symbol,
        )


def _process_asset_grid_for_db(
    db_path: str,
    data: pd.DataFrame,
    asset_history: pd.DataFrame,
    signal_history: pd.DataFrame,
    risk_free_history: pd.DataFrame,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rebuild: bool,
    phase_timings: WorkflowPhaseTimings | None = None,
    strategy_session: _WorkflowStrategySession | None = None,
    rsi_entry_rule: str = "lower",
    canonical_signal_history: pd.DataFrame | None = None,
    strategy_state_verification: str = "trusted",
    workflow_deadline: _WorkflowDeadline | None = None,
    strategy_state_preverified: bool = False,
    expected_strategy_state_generation: int | None = None,
    prevalidated_asset_safe_tail_rows: dict[str, tuple[object, ...]] | None = None,
    prevalidated_best_config: tuple[float, float] | None = None,
) -> bool:
    # This transaction covers market-data synchronization, global benchmark
    # invalidation, and rebuilt state.  A second process waits here and then
    # observes the new generation/config instead of writing a stale resume.
    grid_compute_seconds = 0.0

    def observe_grid_compute(elapsed_seconds: float) -> None:
        nonlocal grid_compute_seconds
        grid_compute_seconds += elapsed_seconds
        if phase_timings is not None:
            phase_timings.add("grid_compute", elapsed_seconds)

    transaction_started = time.perf_counter()
    _check_workflow_deadline(workflow_deadline, "before an asset transaction")
    canonical_signal_history = canonical_signal_history if canonical_signal_history is not None else signal_history
    # Only the run-scoped provider snapshot may occupy the global symbol cache.
    # A calendar recovery is passed separately below and storage namespaces it
    # to this strategy pair, so another pair cannot inherit its RSI history.
    authoritative_histories = {
        asset_symbol: asset_history,
        signal_symbol: canonical_signal_history,
        RISK_FREE_SYMBOL: risk_free_history,
    }
    shared_histories = {
        signal_symbol: canonical_signal_history,
        RISK_FREE_SYMBOL: risk_free_history,
    }
    transaction = (
        strategy_session.immediate_transaction()
        if strategy_session is not None
        else _state_connection(db_path, immediate=True)
    )
    try:
        with transaction as conn:
            actual_rebuild = rebuild
            if not actual_rebuild:
                if strategy_state_preverified:
                    if type(expected_strategy_state_generation) is not int:
                        raise ValueError("A preverified strategy state requires its captured integer generation.")
                    actual_rebuild = strategy_state_generation(conn) != expected_strategy_state_generation
                    if not actual_rebuild:
                        expected_grid_count = len(
                            {
                                (float(buy_rsi), float(profit_target))
                                for buy_rsi in buy_rsi_values
                                for profit_target in profit_target_values
                            }
                        )
                        continuity = conn.execute(
                            """
                            SELECT
                                (SELECT COUNT(*) FROM strategy_state
                                 WHERE asset_symbol = ? AND signal_symbol = ?),
                                (SELECT COUNT(*) FROM strategy_summary
                                 WHERE asset_symbol = ? AND signal_symbol = ?)
                            """,
                            (asset_symbol, signal_symbol, asset_symbol, signal_symbol),
                        ).fetchone()
                        actual_rebuild = bool(
                            continuity is None
                            or int(continuity[0]) != expected_grid_count
                            or int(continuity[1]) != expected_grid_count
                        )
                else:
                    validation_started = _MONOTONIC_CLOCK()
                    try:
                        try:
                            actual_rebuild = not strategy_state_matches_config(
                                conn,
                                asset_symbol,
                                signal_symbol,
                                base_cfg,
                                buy_rsi_values,
                                profit_target_values,
                                rsi_entry_rule,
                                strategy_state_verification=strategy_state_verification,
                            )
                        except sqlite3.OperationalError as exc:
                            if "no such table" not in str(exc).lower():
                                raise
                            actual_rebuild = True
                    finally:
                        if phase_timings is not None:
                            phase_timings.add(
                                "state_validation",
                                max(0.0, _MONOTONIC_CLOCK() - validation_started),
                            )
            process_result = process_asset_grid(
                conn,
                data,
                base_cfg,
                asset_symbol,
                signal_symbol,
                buy_rsi_values,
                profit_target_values,
                rebuild=actual_rebuild,
                signal_history=signal_history,
                authoritative_histories=authoritative_histories,
                presynchronized_authoritative_symbols=(
                    strategy_session.presynchronized_symbols(shared_histories) if strategy_session is not None else None
                ),
                market_history_observation_run_id=(
                    strategy_session.history_observation_run_id if strategy_session is not None else None
                ),
                commit=False,
                strategy_fingerprint=strategy_config_fingerprint(
                    base_cfg,
                    buy_rsi_values,
                    profit_target_values,
                    rsi_entry_rule,
                ),
                grid_compute_observer=observe_grid_compute if phase_timings is not None else None,
                rsi_entry_rule=rsi_entry_rule,
                isolate_strategy_signal_history=signal_history is not canonical_signal_history,
                strategy_state_verification=strategy_state_verification,
                strategy_state_preverified=not actual_rebuild,
                _prevalidated_safe_tail_rows=(None if actual_rebuild else prevalidated_asset_safe_tail_rows),
                _prevalidated_best_config=(None if actual_rebuild else prevalidated_best_config),
                _authoritative_histories_prevalidated=True,
                deadline_check=(
                    None
                    if workflow_deadline is None
                    else partial(
                        _check_workflow_deadline,
                        workflow_deadline,
                        "during an asset transaction",
                    )
                ),
            )
            if type(process_result) is bool:
                actual_rebuild = process_result
        if strategy_session is not None:
            strategy_session.mark_synchronized(shared_histories)
        return actual_rebuild
    finally:
        if phase_timings is not None:
            transaction_seconds = max(0.0, time.perf_counter() - transaction_started)
            phase_timings.add("db_sync", max(0.0, transaction_seconds - grid_compute_seconds))


def _build_reports_for_db(
    db_path: str,
    workflow_assets: pd.DataFrame,
    base_cfg: BacktestConfig,
    processed_asset_pairs: set[tuple[str, str]],
    workflow_label: str | None = None,
    rsi_entry_rule: str | None = None,
    buy_rsi_values: list[float] | None = None,
    profit_target_values: list[float] | None = None,
    deadline_check: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if (buy_rsi_values is None) != (profit_target_values is None):
        raise ValueError(
            "buy_rsi_values and profit_target_values must either both be provided "
            "for report authentication or both be omitted."
        )
    if processed_asset_pairs and buy_rsi_values is None:
        raise ValueError(
            "buy_rsi_values and profit_target_values are required to authenticate reports for completed strategy state."
        )
    report_rsi_entry_rule = rsi_entry_rule
    if report_rsi_entry_rule is None and workflow_label is not None:
        report_rsi_entry_rule = "upper" if workflow_label.strip().lower() == "short" else "lower"
    report_assets = workflow_assets[
        workflow_assets.apply(
            lambda row: (str(row["symbol"]), str(row["rsi_symbol"])) in processed_asset_pairs,
            axis=1,
        )
    ].copy()
    authenticated_report_cfg = base_cfg if buy_rsi_values is not None and profit_target_values is not None else None
    structural_legacy_report = authenticated_report_cfg is None
    with _state_connection(db_path) as conn:
        # All report inputs must describe one committed strategy generation.
        # A deferred read transaction establishes its snapshot on the first
        # SELECT without taking a writer reservation in WAL mode.
        conn.execute("BEGIN")
        if buy_rsi_values is not None and profit_target_values is not None:
            expected_fingerprint = strategy_config_fingerprint(
                base_cfg,
                buy_rsi_values,
                profit_target_values,
                report_rsi_entry_rule or "lower",
            )
            expected_grid_count = len(
                {
                    (float(buy_rsi), float(profit_target))
                    for buy_rsi in buy_rsi_values
                    for profit_target in profit_target_values
                }
            )
            unauthenticated_pairs: set[tuple[str, str]] = set()
            disappeared_pairs: set[tuple[str, str]] = set()
            for asset_symbol, signal_symbol in processed_asset_pairs:
                if deadline_check is not None:
                    deadline_check()
                state_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = ? AND signal_symbol = ?",
                        (asset_symbol, signal_symbol),
                    ).fetchone()[0]
                )
                summary_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM strategy_summary WHERE asset_symbol = ? AND signal_symbol = ?",
                        (asset_symbol, signal_symbol),
                    ).fetchone()[0]
                )
                if state_count == 0 and summary_count == 0:
                    disappeared_pairs.add((asset_symbol, signal_symbol))
                    continue
                if (
                    state_count != expected_grid_count
                    or summary_count != expected_grid_count
                    or not strategy_config_matches_fingerprint(
                        conn,
                        asset_symbol,
                        signal_symbol,
                        expected_fingerprint,
                    )
                ):
                    unauthenticated_pairs.add((asset_symbol, signal_symbol))
            if disappeared_pairs:
                missing = ", ".join(f"{asset}/{signal}" for asset, signal in sorted(disappeared_pairs))
                raise WorkflowRunError(
                    f"Completed workflow state disappeared before reporting for: {missing}. "
                    "Broker submission was aborted."
                )
            if unauthenticated_pairs:
                missing = ", ".join(f"{asset}/{signal}" for asset, signal in sorted(unauthenticated_pairs))
                raise WorkflowRunError(
                    "Completed workflow state no longer matches the requested strategy "
                    f"grid for: {missing}. Broker submission was aborted."
                )
        strategy_report_cache = {}
        optimization_summary, curves = summarize_saved_results(
            conn,
            report_assets,
            rsi_period=base_cfg.rsi_period,
            rsi_entry_rule=report_rsi_entry_rule,
            base_cfg=authenticated_report_cfg,
            expected_buy_rsi_values=buy_rsi_values,
            expected_profit_target_values=profit_target_values,
            allow_unbound_backtest_config=structural_legacy_report,
            _strategy_report_cache=strategy_report_cache,
            _workflow_results_preverified=True,
            deadline_check=deadline_check,
        )
        reported_asset_pairs = (
            set(optimization_summary[["Asset", "RSI Symbol"]].astype(str).itertuples(index=False, name=None))
            if {"Asset", "RSI Symbol"}.issubset(optimization_summary.columns)
            else set()
        )
        missing_asset_pairs = processed_asset_pairs - reported_asset_pairs
        if missing_asset_pairs:
            missing = ", ".join(f"{asset}/{signal}" for asset, signal in sorted(missing_asset_pairs))
            raise WorkflowRunError(
                "Completed workflow state disappeared before report generation for: "
                f"{missing}. Broker submission was aborted."
            )
        buy_signals = build_buy_signal_report(
            conn,
            optimization_summary,
            rsi_period=base_cfg.rsi_period,
            rsi_entry_rule=report_rsi_entry_rule or "lower",
            base_cfg=authenticated_report_cfg,
            expected_buy_rsi_values=buy_rsi_values,
            expected_profit_target_values=profit_target_values,
            allow_unbound_backtest_config=structural_legacy_report,
            _strategy_report_cache=strategy_report_cache,
            deadline_check=deadline_check,
        )
        sell_signals = build_sell_signal_report(
            conn,
            optimization_summary,
            rsi_period=base_cfg.rsi_period,
            rsi_entry_rule=report_rsi_entry_rule or "lower",
            base_cfg=authenticated_report_cfg,
            expected_buy_rsi_values=buy_rsi_values,
            expected_profit_target_values=profit_target_values,
            allow_unbound_backtest_config=structural_legacy_report,
            _strategy_report_cache=strategy_report_cache,
            deadline_check=deadline_check,
        )
        active_managed_symbols = active_alpaca_managed_symbols(conn)
        if buy_signals.empty or not active_managed_symbols:
            eligible_buy_signals = buy_signals.copy()
        else:
            eligible_buy_signals = buy_signals[
                ~buy_signals["Asset"].astype(str).str.upper().isin(active_managed_symbols)
            ].copy()
        realized_pnl_summary = build_alpaca_realized_pnl_summary(
            conn,
            include_workflow=True,
        )

    if workflow_label is not None:
        optimization_summary = _with_workflow_column(optimization_summary, workflow_label)
        buy_signals = _with_workflow_column(buy_signals, workflow_label)
        eligible_buy_signals = _with_workflow_column(eligible_buy_signals, workflow_label)
        sell_signals = _with_workflow_column(sell_signals, workflow_label)
        if not curves.empty:
            curves = curves.rename(columns=lambda column: f"{workflow_label}_{column}")

    return optimization_summary, curves, buy_signals, eligible_buy_signals, sell_signals, realized_pnl_summary


def _submit_alpaca_paper_buy_orders_for_db(
    db_path: str,
    buy_signals: pd.DataFrame,
    alpaca_cfg: AlpacaOrderConfig,
) -> pd.DataFrame:
    with _state_connection(db_path) as conn:
        revalidate_active_sqlite_runtime_file(db_path)
        try:
            results = submit_alpaca_paper_buy_orders(buy_signals, alpaca_cfg, conn=conn)
        except AlpacaBuyBatchError as exc:
            _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg)
            raise
        return _safe_alpaca_workflow_result_messages(
            results,
            alpaca_cfg=alpaca_cfg,
        )


def _unknown_alpaca_buy_submission_results(
    buy_signals: pd.DataFrame,
    failure: BaseException,
    alpaca_cfg: AlpacaOrderConfig | None = None,
) -> pd.DataFrame:
    """Build a current audit when worker finalization obscures broker results."""
    columns = [
        "Workflow",
        "Asset",
        "Date",
        "Client Order ID",
        "Notional",
        "Qty",
        "Limit Price",
        "Status",
        "Alpaca Order ID",
        "Message",
    ]
    failure_message = _best_effort_alpaca_exception_diagnostic(failure, alpaca_cfg=alpaca_cfg)
    source_rows = buy_signals.to_dict("records") or [{}]
    rows = [
        {
            "Workflow": signal.get("Workflow"),
            "Asset": signal.get("Asset"),
            "Date": signal.get("Date"),
            "Client Order ID": None,
            "Notional": None,
            "Qty": None,
            "Limit Price": None,
            "Status": "submission_unknown",
            "Alpaca Order ID": None,
            "Message": (
                "Alpaca buy submission worker failed after broker side effects became possible; "
                f"protective reconciliation was required: {failure_message}"
            ),
        }
        for signal in source_rows
    ]
    return _safe_alpaca_workflow_result_messages(
        pd.DataFrame(rows, columns=columns),
        alpaca_cfg=alpaca_cfg,
    )


def _load_alpaca_managed_positions_for_db(
    db_path: str,
    *,
    connection: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    if connection is not None:
        return load_alpaca_managed_positions(connection)
    with _state_connection(db_path) as conn:
        return load_alpaca_managed_positions(conn)


def _load_alpaca_realized_pnl_for_db(
    db_path: str,
    *,
    connection: sqlite3.Connection | None = None,
) -> pd.DataFrame:
    if connection is not None:
        return build_alpaca_realized_pnl_summary(connection, include_workflow=True)
    with _state_connection(db_path) as conn:
        return build_alpaca_realized_pnl_summary(conn, include_workflow=True)


def _processed_message(data: pd.DataFrame, start_label: str) -> str:
    message = f"Processed {len(data)} rows from {start_label}"
    recovered_symbols = data.attrs.get(TRADIER_RECOVERED_SYMBOLS_ATTR, [])
    if recovered_symbols:
        message += f"; Tradier fallback recovered {', '.join(sorted(recovered_symbols))}"
    return message


def _strategy_data_from_authoritative_histories(
    *,
    asset_symbol: str,
    signal_symbol: str,
    asset_history: pd.DataFrame,
    signal_history: pd.DataFrame,
    risk_free_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build the strategy frame on the settled authoritative asset calendar."""
    data = asset_history.copy()
    if signal_symbol != asset_symbol:
        data = data.join(signal_history, how="left")
    if RISK_FREE_SYMBOL not in {asset_symbol, signal_symbol}:
        aligned_risk_free = risk_free_history.reindex(data.index, method="ffill")
        data = data.join(aligned_risk_free, how="left")

    histories = (asset_history, signal_history, risk_free_history)
    providers: dict[str, str] = {}
    recovered_symbols: set[str] = set()
    for history in histories:
        providers.update(history.attrs.get(MARKET_DATA_PROVIDERS_ATTR, {}))
        recovered_symbols.update(history.attrs.get(TRADIER_RECOVERED_SYMBOLS_ATTR, []))
    if providers:
        data.attrs[MARKET_DATA_PROVIDERS_ATTR] = providers
    if recovered_symbols:
        data.attrs[TRADIER_RECOVERED_SYMBOLS_ATTR] = sorted(recovered_symbols)
    return data


def _skipped_asset_result(
    *,
    job: AssetRunJob,
    plan: AssetRunPlan,
    message: str,
    asset_progress: AssetProgress | None,
    rows_processed: int | None = None,
) -> AssetRunResult:
    if asset_progress is not None:
        asset_progress.start_asset(
            asset=job.asset_symbol,
            signal=job.signal_symbol,
            action="skipping",
        )
    return AssetRunResult(
        workflow_idx=job.workflow_idx,
        asset_symbol=job.asset_symbol,
        signal_symbol=job.signal_symbol,
        action=plan.action,
        rows_processed=rows_processed,
        status="skipped",
        message=message,
        workflow=job.workflow,
    )


async def _prepare_workflow_asset(
    *,
    db_path: str,
    mode: str,
    base_cfg: BacktestConfig,
    tradier_cfg: TradierMarketDataConfig | None,
    job: AssetRunJob,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rsi_entry_rule: str = "lower",
    signal_locks: dict[str, asyncio.Lock],
    signal_histories: dict[str, pd.DataFrame],
    risk_free_history_lock: asyncio.Lock,
    risk_free_histories: dict[str, pd.DataFrame],
    signal_failures: dict[str, str] | None = None,
    calendar_signal_histories: dict[tuple[str, str], pd.DataFrame] | None = None,
    calendar_signal_failures: dict[tuple[str, str], str] | None = None,
    risk_free_failures: dict[str, str] | None = None,
    asset_progress: AssetProgress | None = None,
    phase_timings: WorkflowPhaseTimings | None = None,
    download_executor: Executor | None = None,
    market_data_session: _WorkflowMarketDataSession | None = None,
    strategy_state_verification: str = "trusted",
    workflow_deadline: _WorkflowDeadline | None = None,
    state_validation_lock: asyncio.Lock | None = None,
) -> PreparedAssetRun | AssetRunResult:
    # Database and runtime-file validation belongs outside the asset-local
    # download failure boundary.  A corrupt/inaccessible state database must
    # abort the workflow instead of being reported as one skipped symbol while
    # other symbols continue toward paper-order submission.
    _check_workflow_deadline(workflow_deadline, "before market-data downloads")

    async def prepare_plan() -> AssetRunPlan:
        return await _timed_run_blocking(
            phase_timings,
            "state_validation",
            _prepare_asset_run,
            db_path,
            mode,
            base_cfg,
            job.asset_symbol,
            job.signal_symbol,
            buy_rsi_values,
            profit_target_values,
            rsi_entry_rule,
            strategy_state_verification,
            executor=download_executor,
        )

    # Concurrent read connections make the digest-heavy trusted proof slower
    # on WSL by competing for the same SQLite pages and Python execution time.
    # Downloads remain concurrent, but state proof and safe-tail comparison use
    # one lane while the transaction consumer advances the preceding asset.
    if state_validation_lock is None:
        plan = await prepare_plan()
    else:
        async with state_validation_lock:
            plan = await prepare_plan()
    _check_workflow_deadline(workflow_deadline, "before market-data downloads")
    if signal_failures is None:
        signal_failures = {}
    if calendar_signal_histories is None:
        calendar_signal_histories = {}
    if calendar_signal_failures is None:
        calendar_signal_failures = {}
    if risk_free_failures is None:
        risk_free_failures = {}
    if asset_progress is not None:
        asset_progress.start_asset(
            asset=job.asset_symbol,
            signal=job.signal_symbol,
            action=plan.action,
        )

    # Once a shared prerequisite has failed, do not put every remaining asset
    # through another serialized provider request before discovering the same
    # outcome. Recheck after the asset request as well because another worker
    # can discover a failure while this worker is already downloading.
    async with risk_free_history_lock:
        risk_free_failure = risk_free_failures.get(RISK_FREE_SYMBOL)
    if risk_free_failure is not None:
        return _skipped_asset_result(
            job=job,
            plan=plan,
            message=risk_free_failure,
            asset_progress=asset_progress,
        )

    calendar_signal_key = (job.asset_symbol, job.signal_symbol)
    signal_lock = signal_locks.setdefault(job.signal_symbol, asyncio.Lock())
    async with signal_lock:
        signal_failure = signal_failures.get(job.signal_symbol)
        calendar_signal_failure = calendar_signal_failures.get(calendar_signal_key)
    if signal_failure is not None:
        return _skipped_asset_result(
            job=job,
            plan=plan,
            message=signal_failure,
            asset_progress=asset_progress,
        )
    if calendar_signal_failure is not None:
        return _skipped_asset_result(
            job=job,
            plan=plan,
            message=calendar_signal_failure,
            asset_progress=asset_progress,
        )

    # The same ticker can be an asset in one strategy and an RSI signal in
    # another.  Download it at most once for the whole workflow so a later
    # provider correction cannot replace the snapshot under completed work.
    asset_lock = signal_locks.setdefault(job.asset_symbol, asyncio.Lock())
    async with asset_lock:
        asset_failure = signal_failures.get(job.asset_symbol)
        if asset_failure is not None:
            return _skipped_asset_result(
                job=job,
                plan=plan,
                message=asset_failure,
                asset_progress=asset_progress,
            )
        asset_history = signal_histories.get(job.asset_symbol)
        if asset_history is None:
            if market_data_session is not None:
                market_data_session.individual_retry_count += 1
            try:
                asset_history = await _timed_run_blocking(
                    phase_timings,
                    "download",
                    load_symbol_history,
                    job.asset_symbol,
                    end=None,
                    auto_adjust=base_cfg.auto_adjust,
                    tradier_cfg=tradier_cfg,
                    deadline_monotonic=(None if workflow_deadline is None else workflow_deadline.monotonic_deadline),
                    executor=download_executor,
                )
            except MarketDataDownloadError as exc:
                _check_workflow_deadline(workflow_deadline, "during market-data downloads")
                asset_failure = str(exc)
                signal_failures[job.asset_symbol] = asset_failure
                return _skipped_asset_result(
                    job=job,
                    plan=plan,
                    message=asset_failure,
                    asset_progress=asset_progress,
                )
            if asset_history.empty:
                asset_failure = "No finalized daily market data is available yet."
                signal_failures[job.asset_symbol] = asset_failure
                return _skipped_asset_result(
                    job=job,
                    plan=plan,
                    message=asset_failure,
                    asset_progress=asset_progress,
                    rows_processed=0,
                )
            signal_histories[job.asset_symbol] = asset_history

    async with risk_free_history_lock:
        risk_free_failure = risk_free_failures.get(RISK_FREE_SYMBOL)
        if risk_free_failure is not None:
            return _skipped_asset_result(
                job=job,
                plan=plan,
                message=risk_free_failure,
                asset_progress=asset_progress,
            )
        risk_free_history = risk_free_histories.get(RISK_FREE_SYMBOL)
        if risk_free_history is None:
            if market_data_session is not None:
                market_data_session.individual_retry_count += 1
            try:
                risk_free_history = await _timed_run_blocking(
                    phase_timings,
                    "download",
                    load_risk_free_history,
                    end=None,
                    auto_adjust=base_cfg.auto_adjust,
                    tradier_cfg=tradier_cfg,
                    deadline_monotonic=(None if workflow_deadline is None else workflow_deadline.monotonic_deadline),
                    executor=download_executor,
                )
            except MarketDataDownloadError as exc:
                _check_workflow_deadline(workflow_deadline, "during market-data downloads")
                risk_free_failure = str(exc)
                risk_free_failures[RISK_FREE_SYMBOL] = risk_free_failure
                return _skipped_asset_result(
                    job=job,
                    plan=plan,
                    message=risk_free_failure,
                    asset_progress=asset_progress,
                )
            if risk_free_history.empty:
                risk_free_failure = "No settled daily benchmark history is available for ^IRX."
                risk_free_failures[RISK_FREE_SYMBOL] = risk_free_failure
                return _skipped_asset_result(
                    job=job,
                    plan=plan,
                    message=risk_free_failure,
                    asset_progress=asset_progress,
                )
            risk_free_histories[RISK_FREE_SYMBOL] = risk_free_history

    async with signal_lock:
        signal_failure = signal_failures.get(job.signal_symbol)
        if signal_failure is not None:
            return _skipped_asset_result(
                job=job,
                plan=plan,
                message=signal_failure,
                asset_progress=asset_progress,
            )
        calendar_signal_failure = calendar_signal_failures.get(calendar_signal_key)
        if calendar_signal_failure is not None:
            return _skipped_asset_result(
                job=job,
                plan=plan,
                message=calendar_signal_failure,
                asset_progress=asset_progress,
            )
        canonical_signal_history = signal_histories.get(job.signal_symbol)
        if canonical_signal_history is None:
            if market_data_session is not None:
                market_data_session.individual_retry_count += 1
            try:
                canonical_signal_history = await _timed_run_blocking(
                    phase_timings,
                    "download",
                    load_signal_history,
                    job.signal_symbol,
                    end=None,
                    auto_adjust=base_cfg.auto_adjust,
                    tradier_cfg=tradier_cfg,
                    deadline_monotonic=(None if workflow_deadline is None else workflow_deadline.monotonic_deadline),
                    executor=download_executor,
                )
            except MarketDataDownloadError as exc:
                _check_workflow_deadline(workflow_deadline, "during market-data downloads")
                signal_failure = str(exc)
                signal_failures[job.signal_symbol] = signal_failure
                return _skipped_asset_result(
                    job=job,
                    plan=plan,
                    message=signal_failure,
                    asset_progress=asset_progress,
                )
            if canonical_signal_history.empty:
                signal_failure = f"No settled daily signal history is available for {job.signal_symbol}."
                signal_failures[job.signal_symbol] = signal_failure
                return _skipped_asset_result(
                    job=job,
                    plan=plan,
                    message=signal_failure,
                    asset_progress=asset_progress,
                )
            signal_histories[job.signal_symbol] = canonical_signal_history

        signal_history = calendar_signal_histories.get(calendar_signal_key)
        if signal_history is None:
            if job.signal_symbol == job.asset_symbol or signal_history_overlaps_calendar(
                calendar_symbol=job.asset_symbol,
                calendar_history=asset_history,
                signal_symbol=job.signal_symbol,
                signal_history=canonical_signal_history,
            ):
                signal_history = canonical_signal_history
            else:
                for (_candidate_asset, candidate_signal), candidate_history in calendar_signal_histories.items():
                    recovered_symbols = candidate_history.attrs.get(TRADIER_RECOVERED_SYMBOLS_ATTR, [])
                    if candidate_signal != job.signal_symbol or job.signal_symbol not in recovered_symbols:
                        continue
                    if signal_history_overlaps_calendar(
                        calendar_symbol=job.asset_symbol,
                        calendar_history=asset_history,
                        signal_symbol=job.signal_symbol,
                        signal_history=candidate_history,
                    ):
                        signal_history = candidate_history
                        break
            if signal_history is None:
                try:
                    signal_history = await _timed_run_blocking(
                        phase_timings,
                        "download",
                        recover_signal_history_for_calendar,
                        calendar_symbol=job.asset_symbol,
                        calendar_history=asset_history,
                        signal_symbol=job.signal_symbol,
                        signal_history=canonical_signal_history,
                        auto_adjust=base_cfg.auto_adjust,
                        tradier_cfg=tradier_cfg,
                        executor=download_executor,
                    )
                except MarketDataDownloadError as exc:
                    _check_workflow_deadline(workflow_deadline, "during market-data downloads")
                    calendar_signal_failure = str(exc)
                    calendar_signal_failures[calendar_signal_key] = calendar_signal_failure
                    return _skipped_asset_result(
                        job=job,
                        plan=plan,
                        message=calendar_signal_failure,
                        asset_progress=asset_progress,
                    )
            calendar_signal_histories[calendar_signal_key] = signal_history

    data = _strategy_data_from_authoritative_histories(
        asset_symbol=job.asset_symbol,
        signal_symbol=job.signal_symbol,
        asset_history=asset_history,
        signal_history=signal_history,
        risk_free_history=risk_free_history,
    )
    prevalidated_asset_safe_tail_rows = None
    if signal_history is canonical_signal_history:

        async def prepare_safe_tail() -> dict[str, tuple[object, ...]] | None:
            return await _timed_run_blocking(
                phase_timings,
                "state_validation",
                _prepare_asset_safe_tail_rows,
                db_path,
                plan,
                asset_history,
                executor=download_executor,
            )

        if state_validation_lock is None:
            prevalidated_asset_safe_tail_rows = await prepare_safe_tail()
        else:
            async with state_validation_lock:
                prevalidated_asset_safe_tail_rows = await prepare_safe_tail()
    return PreparedAssetRun(
        job=job,
        plan=plan,
        data=data,
        asset_history=asset_history,
        signal_history=signal_history,
        risk_free_history=risk_free_history,
        canonical_signal_history=canonical_signal_history,
        prevalidated_asset_safe_tail_rows=prevalidated_asset_safe_tail_rows,
    )


async def _complete_workflow_asset(
    outcome: PreparedAssetRun | AssetRunResult,
    *,
    db_path: str,
    base_cfg: BacktestConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rsi_entry_rule: str,
    asset_progress: AssetProgress | None = None,
    phase_timings: WorkflowPhaseTimings | None = None,
    strategy_executor: Executor | None = None,
    strategy_session: _WorkflowStrategySession | None = None,
    strategy_state_verification: str = "trusted",
    workflow_deadline: _WorkflowDeadline | None = None,
) -> AssetRunResult:
    try:
        if isinstance(outcome, AssetRunResult):
            return outcome

        try:
            _check_workflow_deadline(workflow_deadline, "between asset transactions")
            actual_rebuild = await _run_blocking(
                strategy_executor,
                _process_asset_grid_for_db,
                db_path,
                outcome.data,
                outcome.asset_history,
                outcome.signal_history,
                outcome.risk_free_history,
                base_cfg,
                outcome.job.asset_symbol,
                outcome.job.signal_symbol,
                buy_rsi_values,
                profit_target_values,
                outcome.plan.rebuild,
                phase_timings,
                strategy_session,
                rsi_entry_rule,
                canonical_signal_history=outcome.canonical_signal_history,
                strategy_state_verification=strategy_state_verification,
                workflow_deadline=workflow_deadline,
                strategy_state_preverified=outcome.plan.strategy_state_preverified,
                expected_strategy_state_generation=outcome.plan.strategy_state_generation,
                prevalidated_asset_safe_tail_rows=outcome.prevalidated_asset_safe_tail_rows,
                prevalidated_best_config=outcome.plan.preverified_best_config,
            )
        # The transaction layer has normalized malformed market input to this
        # dedicated asset-local validation failure.  Every other exception is a
        # storage, runtime-integrity, concurrency, or programming failure and
        # must abort the workflow before reports or broker work can proceed.
        except AssetMarketDataError as exc:
            return _skipped_asset_result(
                job=outcome.job,
                plan=outcome.plan,
                message=str(exc),
                asset_progress=asset_progress,
            )

        return AssetRunResult(
            workflow_idx=outcome.job.workflow_idx,
            asset_symbol=outcome.job.asset_symbol,
            signal_symbol=outcome.job.signal_symbol,
            action="Rebuilding" if actual_rebuild else "Updating",
            rows_processed=len(outcome.data),
            status="done",
            message=_processed_message(outcome.data, outcome.plan.start_label),
            workflow=outcome.job.workflow,
        )
    finally:
        if asset_progress is not None:
            asset_progress.finish_asset()


async def _prefetch_workflow_histories(
    symbols: list[str],
    *,
    base_cfg: BacktestConfig,
    phase_timings: WorkflowPhaseTimings,
    market_data_session: _WorkflowMarketDataSession,
    workflow_deadline: _WorkflowDeadline | None = None,
) -> None:
    """Populate the run-scoped cache with validated Yahoo batches of 32."""
    unique_symbols = sorted(set(symbols).difference(market_data_session.batch_attempted_symbols))
    for offset in range(0, len(unique_symbols), MARKET_DATA_BATCH_SIZE):
        _check_workflow_deadline(workflow_deadline, "before market-data downloads")
        batch = unique_symbols[offset : offset + MARKET_DATA_BATCH_SIZE]
        histories, errors = await _timed_run_blocking(
            phase_timings,
            "download",
            load_symbol_history_batch,
            batch,
            end=None,
            auto_adjust=base_cfg.auto_adjust,
            deadline_monotonic=(None if workflow_deadline is None else workflow_deadline.monotonic_deadline),
        )
        _check_workflow_deadline(workflow_deadline, "during market-data downloads")
        market_data_session.batch_count += 1
        market_data_session.batch_attempted_symbols.update(batch)
        market_data_session.histories.update(histories)
        market_data_session.batch_retry_symbols.update(errors)


async def _run_asset_pipeline(
    *,
    jobs: list[AssetRunJob],
    concurrency: int,
    db_path: str,
    mode: str,
    base_cfg: BacktestConfig,
    tradier_cfg: TradierMarketDataConfig | None,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    asset_progress: AssetProgress | None,
    phase_timings: WorkflowPhaseTimings,
    rsi_entry_rule: str = "lower",
    history_observation_run_id: str | None = None,
    market_data_session: _WorkflowMarketDataSession | None = None,
    strategy_state_verification: str = "trusted",
    workflow_deadline: _WorkflowDeadline | None = None,
) -> list[AssetRunResult]:
    if not jobs:
        return []

    shared_market_data_session = market_data_session is not None
    market_data_session = market_data_session or _WorkflowMarketDataSession()
    injected_individual_download = any(
        isinstance(loader, Mock) for loader in (load_symbol_history, load_signal_history, load_risk_free_history)
    )
    if shared_market_data_session and not injected_individual_download:
        await _prefetch_workflow_histories(
            [
                RISK_FREE_SYMBOL,
                *(job.asset_symbol for job in jobs),
                *(job.signal_symbol for job in jobs),
            ],
            base_cfg=base_cfg,
            phase_timings=phase_timings,
            market_data_session=market_data_session,
            workflow_deadline=workflow_deadline,
        )
    signal_locks = market_data_session.signal_locks
    signal_histories = market_data_session.signal_histories
    signal_failures = market_data_session.signal_failures
    calendar_signal_histories = market_data_session.calendar_signal_histories
    calendar_signal_failures = market_data_session.calendar_signal_failures
    risk_free_history_lock = market_data_session.risk_free_history_lock
    risk_free_histories = market_data_session.risk_free_histories
    risk_free_failures = market_data_session.risk_free_failures
    state_validation_lock = asyncio.Lock()
    strategy_session = _WorkflowStrategySession(
        db_path,
        history_observation_run_id=history_observation_run_id,
    )

    async def prepare(
        job: AssetRunJob,
        download_executor: Executor | None = None,
    ) -> PreparedAssetRun | AssetRunResult:
        return await _prepare_workflow_asset(
            db_path=db_path,
            mode=mode,
            base_cfg=base_cfg,
            tradier_cfg=tradier_cfg,
            job=job,
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            rsi_entry_rule=rsi_entry_rule,
            signal_locks=signal_locks,
            signal_histories=signal_histories,
            risk_free_history_lock=risk_free_history_lock,
            risk_free_histories=risk_free_histories,
            signal_failures=signal_failures,
            calendar_signal_histories=calendar_signal_histories,
            calendar_signal_failures=calendar_signal_failures,
            risk_free_failures=risk_free_failures,
            asset_progress=asset_progress,
            phase_timings=phase_timings,
            download_executor=download_executor,
            market_data_session=market_data_session,
            strategy_state_verification=strategy_state_verification,
            workflow_deadline=workflow_deadline,
            state_validation_lock=state_validation_lock,
        )

    async def complete(
        outcome: PreparedAssetRun | AssetRunResult,
        strategy_executor: Executor,
    ) -> AssetRunResult:
        return await _complete_workflow_asset(
            outcome,
            db_path=db_path,
            base_cfg=base_cfg,
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            rsi_entry_rule=rsi_entry_rule,
            asset_progress=asset_progress,
            phase_timings=phase_timings,
            strategy_executor=strategy_executor,
            strategy_session=strategy_session,
            strategy_state_verification=strategy_state_verification,
            workflow_deadline=workflow_deadline,
        )

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="workflow-strategy",
    ) as strategy_executor:
        primary_failure: BaseException | None = None
        try:
            if concurrency <= 1:
                results = [await complete(await prepare(job), strategy_executor) for job in jobs]
                return sorted(results, key=lambda result: result.workflow_idx)

            worker_count = min(max(1, concurrency), len(jobs))
            job_queue: asyncio.Queue[AssetRunJob | None] = asyncio.Queue()
            prepared_queue: asyncio.Queue[PreparedAssetRun | AssetRunResult] = asyncio.Queue(maxsize=1)
            for job in jobs:
                job_queue.put_nowait(job)
            for _ in range(worker_count):
                job_queue.put_nowait(None)

            async def download_worker(download_executor: Executor) -> None:
                while True:
                    job = await job_queue.get()
                    if job is None:
                        return
                    await prepared_queue.put(await prepare(job, download_executor))

            async def strategy_consumer() -> list[AssetRunResult]:
                results: list[AssetRunResult] = []
                for _ in jobs:
                    outcome = await prepared_queue.get()
                    results.append(await complete(outcome, strategy_executor))
                return sorted(results, key=lambda result: result.workflow_idx)

            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="workflow-download",
            ) as download_executor:
                async with asyncio.TaskGroup() as task_group:
                    consumer_task = task_group.create_task(strategy_consumer())
                    for _ in range(worker_count):
                        task_group.create_task(download_worker(download_executor))

            return consumer_task.result()
        except BaseException as exc:
            primary_failure = exc
            raise
        finally:
            try:
                await _run_blocking(strategy_executor, strategy_session.close)
            except BaseException as cleanup_failure:
                if primary_failure is None:
                    raise
                primary_failure.add_note(
                    f"Failed strategy-session cleanup after the asset pipeline failed: {cleanup_failure}"
                )


def _empty_workflow_assets(workflow_label: str) -> pd.DataFrame:
    out = pd.DataFrame(columns=["symbol", "name", "rsi_symbol", "workflow"])
    out.attrs["universe_title"] = f"Executable {workflow_label} Leveraged ETFs/ETNs From Merged Universe"
    out.attrs["universe_counts"] = {}
    return out


def _normalize_workflow_asset_groups(
    workflow_assets: dict[str, pd.DataFrame] | pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if isinstance(workflow_assets, pd.DataFrame):
        return {
            "long": workflow_assets,
            "short": _empty_workflow_assets(SHORT_WORKFLOW_LABEL),
        }
    return {
        "long": workflow_assets.get("long", _empty_workflow_assets(LONG_WORKFLOW_LABEL)),
        "short": workflow_assets.get("short", _empty_workflow_assets(SHORT_WORKFLOW_LABEL)),
    }


def _workflow_jobs(
    workflow_assets: pd.DataFrame,
    *,
    workflow_label: str | None = None,
) -> list[AssetRunJob]:
    return [
        AssetRunJob(
            workflow_idx=workflow_idx,
            asset_symbol=str(workflow_asset.symbol),
            signal_symbol=str(workflow_asset.rsi_symbol),
            workflow=workflow_label,
        )
        for workflow_idx, workflow_asset in enumerate(
            workflow_assets.itertuples(index=False),
            start=1,
        )
    ]


def _completed_asset_pairs(asset_run_results: list[AssetRunResult]) -> set[tuple[str, str]]:
    return {(result.asset_symbol, result.signal_symbol) for result in asset_run_results if result.status == "done"}


def _concat_report_frames(frames: list[pd.DataFrame], *, axis: int = 0) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    if axis == 1:
        non_empty = [frame for frame in frames if not frame.empty]
        return pd.concat(non_empty, axis=1, join="outer", sort=False).sort_index() if non_empty else pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


class _DigestingTextWriter:
    """Hash the exact UTF-8 characters accepted by a report's text descriptor."""

    def __init__(self, file: TextIO) -> None:
        self._file = file
        self._digest = hashlib.sha256()

    def write(self, value: str) -> int:
        written = self._file.write(value)
        self._digest.update(value[:written].encode("utf-8"))
        return written

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _formula_safe_csv_text(value: object) -> object:
    """Keep externally controlled text from becoming a spreadsheet formula."""
    if not isinstance(value, str) or not value:
        return value
    stripped = str.lstrip(value, _CSV_FORMULA_LEADING_WHITESPACE)
    if value[0] in _CSV_FORMULA_PREFIXES or (stripped and stripped[0] in "=+-@"):
        return "'" + value
    return value


def _formula_safe_csv_axis(axis: pd.Index) -> pd.Index:
    """Neutralize string labels and names that pandas serializes as CSV cells."""
    safe_names = [_formula_safe_csv_text(name) for name in axis.names]
    names_changed = any(safe is not original for safe, original in zip(safe_names, axis.names, strict=True))
    if isinstance(axis, pd.MultiIndex):
        safe_values: list[tuple[object, ...]] = []
        values_changed = False
        for value in axis:
            safe_value = tuple(_formula_safe_csv_text(part) for part in value)
            safe_values.append(safe_value)
            values_changed = values_changed or any(
                safe is not original for safe, original in zip(safe_value, value, strict=True)
            )
        if not values_changed:
            return axis.set_names(safe_names) if names_changed else axis
        return pd.MultiIndex.from_tuples(safe_values, names=safe_names)

    safe_values = [_formula_safe_csv_text(value) for value in axis]
    values_changed = any(safe is not original for safe, original in zip(safe_values, axis, strict=True))
    if not values_changed:
        return axis.rename(safe_names[0]) if names_changed else axis
    return pd.Index(safe_values, name=safe_names[0], tupleize_cols=False)


def _formula_safe_csv_frame(frame: pd.DataFrame, *, index: bool) -> pd.DataFrame:
    """Return a serialization view with formula-capable text cells escaped."""
    safe_frame = frame.copy(deep=False)
    for position in range(frame.shape[1]):
        column = frame.iloc[:, position]
        if any(_formula_safe_csv_text(value) is not value for value in column.array):
            safe_frame.isetitem(position, column.map(_formula_safe_csv_text))
    safe_frame.columns = _formula_safe_csv_axis(frame.columns)
    if index:
        safe_frame.index = _formula_safe_csv_axis(frame.index)
    return safe_frame


def _atomic_to_csv(frame: pd.DataFrame, destination: Path, *, index: bool = True) -> str:
    """Write a CSV through one descriptor and verify the inode published by rename."""
    active_output_directory = _ACTIVE_OUTPUT_DIRECTORY.get()
    destination_parent = _case_preserving_workflow_path(destination.parent)
    if active_output_directory is not None and destination_parent == active_output_directory.path:
        revalidate_owned_runtime_directory(active_output_directory, label="Report output directory")
        directory_guard = active_output_directory
    else:
        directory_guard = require_owned_runtime_directory(
            destination.parent,
            label="Report output directory",
        )
    with open_owned_runtime_directory(
        directory_guard,
        label="Report output directory",
    ) as directory_descriptor:
        fd = -1
        temporary_path: Path | None = None
        anchor_path: Path | None = None
        previous_path: Path | None = None
        opened_file: os.stat_result | None = None
        previous_file: os.stat_result | None = None
        temporary_file: TextIO | None = None
        primary_failure: BaseException | None = None
        try:
            fd, temporary_path = _open_private_report_temporary(
                destination,
                directory_descriptor=directory_descriptor,
            )
            anchor_path = Path(f"{temporary_path}.anchor")
            previous_path = Path(f"{temporary_path}.previous")
            if os.name == "posix":
                os.fchmod(fd, 0o600)
            temporary_file = os.fdopen(fd, "w", encoding="utf-8", newline="")
            fd = -1
            digesting_file = _DigestingTextWriter(temporary_file)
            _formula_safe_csv_frame(frame, index=index).to_csv(digesting_file, index=index)
            temporary_file.flush()
            # A durable namespace update is useful only after the inode's
            # contents have reached stable storage.  Python maps this to
            # fsync(2) on POSIX and _commit() on Windows.
            os.fsync(temporary_file.fileno())
            published_digest = digesting_file.hexdigest()
            opened_file = os.fstat(temporary_file.fileno())
            _require_same_private_temporary_file(
                temporary_path,
                opened_file,
                expected_links=1,
                directory_descriptor=directory_descriptor,
            )
            _link_report_path(
                temporary_path,
                anchor_path,
                directory_descriptor=directory_descriptor,
            )
            _require_same_private_temporary_file(
                temporary_path,
                opened_file,
                expected_links=2,
                directory_descriptor=directory_descriptor,
            )
            _require_same_private_temporary_file(
                anchor_path,
                opened_file,
                expected_links=2,
                directory_descriptor=directory_descriptor,
            )
            previous_file = _anchor_previous_report(
                destination,
                previous_path,
                directory_descriptor=directory_descriptor,
            )
            revalidate_owned_runtime_directory(
                directory_guard,
                label="Report output directory",
            )
            if os.name == "posix":
                # Keep the descriptor open so the post-rename comparison is
                # against the file that actually received the CSV bytes.
                _publish_verified_report(
                    temporary_path,
                    anchor_path,
                    destination,
                    previous_path,
                    opened_file,
                    previous_file,
                    directory_descriptor=directory_descriptor,
                )

            if os.name != "posix":
                # Windows does not reliably permit replacing an open CRT file.
                temporary_file.close()
                temporary_file = None
                assert opened_file is not None
                _require_same_private_temporary_file(
                    temporary_path,
                    opened_file,
                    expected_links=2,
                    directory_descriptor=directory_descriptor,
                )
                _require_same_private_temporary_file(
                    anchor_path,
                    opened_file,
                    expected_links=2,
                    directory_descriptor=directory_descriptor,
                )
                _publish_verified_report(
                    temporary_path,
                    anchor_path,
                    destination,
                    previous_path,
                    opened_file,
                    previous_file,
                    directory_descriptor=directory_descriptor,
                )
            revalidate_owned_runtime_directory(
                directory_guard,
                label="Report output directory",
            )
            if os.name == "posix":
                if directory_descriptor is None:
                    raise OSError("Report output directory does not expose a descriptor for durable publication.")
                # Persist the destination replacement and removal of the
                # publication anchors before a caller can commit a manifest
                # that depends on this CSV generation.
                os.fsync(directory_descriptor)
            # On Windows, _replace_report_path uses MoveFileExW with
            # MOVEFILE_WRITE_THROUGH for the corresponding durable namespace
            # boundary; Windows does not expose directory fsync through os.
        except BaseException as exc:
            primary_failure = exc
            raise
        finally:
            cleanup_failures: list[tuple[str, BaseException]] = []
            if temporary_file is not None:
                try:
                    temporary_file.close()
                except BaseException as exc:
                    cleanup_failures.append(("close the report temporary descriptor", exc))
            if fd >= 0:
                try:
                    os.close(fd)
                except BaseException as exc:
                    cleanup_failures.append(("close the report temporary descriptor", exc))
            for cleanup_path in (temporary_path, anchor_path, previous_path):
                if cleanup_path is not None:
                    try:
                        _unlink_non_directory_entry(
                            cleanup_path,
                            directory_descriptor=directory_descriptor,
                        )
                    except BaseException as exc:
                        cleanup_failures.append((f"clean up report publication path {cleanup_path}", exc))
            if cleanup_failures:
                if primary_failure is not None:
                    for action, cleanup_failure in cleanup_failures:
                        primary_failure.add_note(f"Failed to {action}: {cleanup_failure}")
                else:
                    first_action, first_cleanup_failure = cleanup_failures[0]
                    first_cleanup_failure.add_note(f"Failed to {first_action}.")
                    for action, cleanup_failure in cleanup_failures[1:]:
                        first_cleanup_failure.add_note(f"Also failed to {action}: {cleanup_failure}")
                    raise first_cleanup_failure
    revalidate_owned_runtime_directory(directory_guard, label="Report output directory")
    return published_digest


def _open_private_report_temporary(
    destination: Path,
    *,
    directory_descriptor: int | None,
) -> tuple[int, Path]:
    """Create a report temporary in the verified output-directory namespace."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    temporary_name = f".{destination.name}.{uuid.uuid4().hex}.tmp"
    if directory_descriptor is None:
        temporary_path = destination.parent / temporary_name
        fd = os.open(temporary_path, flags, 0o600)
        return fd, temporary_path
    fd = os.open(
        temporary_name,
        flags,
        0o600,
        dir_fd=directory_descriptor,
    )
    return fd, destination.parent / temporary_name


def _same_file_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _report_path_status(
    path: Path,
    *,
    directory_descriptor: int | None,
) -> os.stat_result:
    if directory_descriptor is None:
        return os.lstat(path)
    return os.stat(
        path.name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _same_snapshot_file_version(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )


def _same_report_cleanup_file_version(first: os.stat_result, second: os.stat_result) -> bool:
    """Compare stable file metadata across a rename, which may update ctime."""
    return (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
    ) == (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
    )


def _open_verified_alpaca_snapshot_file(
    path: Path,
    *,
    directory_descriptor: int | None,
) -> tuple[int, os.stat_result]:
    observed = _report_path_status(path, directory_descriptor=directory_descriptor)
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        raise OSError(f"Alpaca snapshot file {path} must be one regular file with one link.")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if directory_descriptor is None:
        file_descriptor = os.open(path, flags)
    else:
        file_descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
    try:
        opened = os.fstat(file_descriptor)
        current = _report_path_status(path, directory_descriptor=directory_descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not _same_snapshot_file_version(observed, opened)
            or not _same_snapshot_file_version(opened, current)
        ):
            raise OSError(f"Alpaca snapshot file {path} changed while it was being opened.")
        if os.name == "posix":
            if opened.st_uid != os.geteuid():
                raise PermissionError(f"Alpaca snapshot file {path} must be owned by the current user.")
            if stat.S_IMODE(opened.st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
                raise PermissionError(f"Alpaca snapshot file {path} must not be accessible by group or other users.")
        return file_descriptor, opened
    except BaseException:
        os.close(file_descriptor)
        raise


def _validate_alpaca_snapshot_manifest(
    manifest: pd.DataFrame,
) -> tuple[str, str, tuple[str, ...], dict[str, str]]:
    if tuple(manifest.columns) != _ALPACA_SNAPSHOT_MANIFEST_COLUMNS or manifest.empty:
        raise OSError("Alpaca snapshot manifest has an invalid schema.")
    if set(manifest["Schema Version"]) != {_ALPACA_SNAPSHOT_SCHEMA_VERSION}:
        raise OSError("Alpaca snapshot manifest uses an unsupported schema version.")
    generations = set(manifest["Generation"])
    if len(generations) != 1:
        raise OSError("Alpaca snapshot manifest contains inconsistent generation IDs.")
    generation = generations.pop()
    if len(generation) != 32 or any(character not in "0123456789abcdef" for character in generation):
        raise OSError("Alpaca snapshot manifest contains an invalid generation ID.")
    if set(manifest["Status"]) != {"committed"}:
        raise OSError("Alpaca snapshot manifest is not committed; covered CSVs may span generations.")
    snapshot_kinds = set(manifest["Snapshot Kind"])
    if len(snapshot_kinds) != 1:
        raise OSError("Alpaca snapshot manifest contains inconsistent snapshot kinds.")
    snapshot_kind = snapshot_kinds.pop()
    try:
        expected_filenames = _ALPACA_SNAPSHOT_FILENAMES_BY_KIND[snapshot_kind]
    except KeyError as exc:
        raise OSError(f"Alpaca snapshot manifest has unknown kind {snapshot_kind!r}.") from exc
    if tuple(manifest["Filename"]) != expected_filenames:
        raise OSError("Alpaca snapshot manifest does not cover the required broker CSV set.")

    expected_digests = dict(zip(manifest["Filename"], manifest["SHA256"], strict=True))
    for filename, expected_digest in expected_digests.items():
        if len(expected_digest) != 64 or any(character not in "0123456789abcdef" for character in expected_digest):
            raise OSError(f"Alpaca snapshot manifest has an invalid digest for {filename}.")
    return generation, snapshot_kind, expected_filenames, expected_digests


def load_alpaca_snapshot(output_dir: str | os.PathLike[str]) -> AlpacaSnapshot:
    """Copy and authenticate one committed broker CSV generation.

    All returned bytes are read from verified descriptors and authenticated
    against the manifest that was opened with them. A later atomic publication
    can replace canonical paths without changing this in-memory generation.
    """
    output_path = Path(output_dir)
    directory_guard = require_owned_runtime_directory(
        output_path,
        label="Report output directory",
    )
    manifest_path = output_path / _ALPACA_SNAPSHOT_MANIFEST_FILENAME
    with open_owned_runtime_directory(
        directory_guard,
        label="Report output directory",
    ) as directory_descriptor:
        manifest_descriptor, manifest_file = _open_verified_alpaca_snapshot_file(
            manifest_path,
            directory_descriptor=directory_descriptor,
        )
        with os.fdopen(manifest_descriptor, "rb") as manifest_stream:
            manifest_contents = manifest_stream.read()
            if not _same_snapshot_file_version(manifest_file, os.fstat(manifest_stream.fileno())):
                raise OSError("Alpaca snapshot manifest changed while it was being read.")
        manifest = pd.read_csv(
            io.BytesIO(manifest_contents),
            dtype=str,
            keep_default_na=False,
        )
        generation, snapshot_kind, expected_filenames, expected_digests = _validate_alpaca_snapshot_manifest(manifest)

        snapshot_files: dict[str, bytes] = {}
        for filename in expected_filenames:
            report_path = output_path / filename
            report_descriptor, report_file = _open_verified_alpaca_snapshot_file(
                report_path,
                directory_descriptor=directory_descriptor,
            )
            with os.fdopen(report_descriptor, "rb") as report_stream:
                contents = report_stream.read()
                if not _same_snapshot_file_version(report_file, os.fstat(report_stream.fileno())):
                    raise OSError(f"Alpaca snapshot file {report_path} changed while it was read.")
            if hashlib.sha256(contents).hexdigest() != expected_digests[filename]:
                raise OSError(f"Alpaca snapshot file {report_path} does not match the committed manifest.")
            snapshot_files[filename] = contents

    revalidate_owned_runtime_directory(directory_guard, label="Report output directory")
    return AlpacaSnapshot(
        generation=generation,
        snapshot_kind=snapshot_kind,
        files=MappingProxyType(snapshot_files),
    )


def validate_alpaca_snapshot(output_dir: str | os.PathLike[str]) -> str:
    """Validate current broker CSV paths and return their generation ID.

    This compatibility check does not pin files for later callers. Use
    :func:`load_alpaca_snapshot` when consuming the CSV contents.
    """
    output_path = Path(output_dir)
    directory_guard = require_owned_runtime_directory(
        output_path,
        label="Report output directory",
    )
    manifest_path = output_path / _ALPACA_SNAPSHOT_MANIFEST_FILENAME
    with open_owned_runtime_directory(
        directory_guard,
        label="Report output directory",
    ) as directory_descriptor:
        manifest_descriptor, manifest_file = _open_verified_alpaca_snapshot_file(
            manifest_path,
            directory_descriptor=directory_descriptor,
        )
        with os.fdopen(manifest_descriptor, "r", encoding="utf-8", newline="") as manifest_stream:
            manifest = pd.read_csv(
                manifest_stream,
                dtype=str,
                keep_default_na=False,
            )
            if not _same_snapshot_file_version(manifest_file, os.fstat(manifest_stream.fileno())):
                raise OSError("Alpaca snapshot manifest changed while it was being read.")

        generation, _, expected_filenames, expected_digests = _validate_alpaca_snapshot_manifest(manifest)
        for filename in expected_filenames:
            expected_digest = expected_digests[filename]
            report_path = output_path / filename
            report_descriptor, report_file = _open_verified_alpaca_snapshot_file(
                report_path,
                directory_descriptor=directory_descriptor,
            )
            with os.fdopen(report_descriptor, "rb") as report_stream:
                digest = hashlib.file_digest(report_stream, "sha256").hexdigest()
                if not _same_snapshot_file_version(report_file, os.fstat(report_stream.fileno())):
                    raise OSError(f"Alpaca snapshot file {report_path} changed while it was read.")
            current_report = _report_path_status(
                report_path,
                directory_descriptor=directory_descriptor,
            )
            if not _same_snapshot_file_version(report_file, current_report):
                raise OSError(f"Alpaca snapshot file {report_path} changed during validation.")
            if digest != expected_digest:
                raise OSError(f"Alpaca snapshot file {report_path} does not match the committed manifest.")

        current_manifest = _report_path_status(
            manifest_path,
            directory_descriptor=directory_descriptor,
        )
        if not _same_snapshot_file_version(manifest_file, current_manifest):
            raise OSError("Alpaca snapshot manifest changed during validation.")
    revalidate_owned_runtime_directory(directory_guard, label="Report output directory")
    return generation


def _link_report_path(
    source: Path,
    destination: Path,
    *,
    directory_descriptor: int | None,
) -> None:
    if directory_descriptor is None:
        os.link(source, destination, follow_symlinks=False)
        return
    os.link(
        source.name,
        destination.name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _replace_report_path(
    source: Path,
    destination: Path,
    *,
    directory_descriptor: int | None,
) -> None:
    if os.name == "nt":
        if directory_descriptor is not None:
            raise OSError("Windows report publication unexpectedly received a directory descriptor.")
        _windows_replace_report_path(source, destination)
        return
    if directory_descriptor is None:
        os.replace(source, destination)
        return
    os.replace(
        source.name,
        destination.name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )


def _windows_replace_report_path(source: Path, destination: Path) -> None:
    """Atomically replace and durably publish one report path on Windows."""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    except (AttributeError, OSError) as exc:  # pragma: no cover - native Windows failure path
        raise OSError("Cannot load the Windows durable file-move API.") from exc

    move_file = kernel32.MoveFileExW
    move_file.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    )
    move_file.restype = ctypes.c_int
    flags = _WINDOWS_MOVEFILE_REPLACE_EXISTING | _WINDOWS_MOVEFILE_WRITE_THROUGH
    if not move_file(os.fspath(source), os.fspath(destination), flags):
        error_code = ctypes.get_last_error()
        raise OSError(error_code, f"Windows could not durably replace report path {destination}.")


def _require_same_private_temporary_file(
    path: Path,
    opened_file: os.stat_result,
    *,
    expected_links: int,
    directory_descriptor: int | None = None,
) -> None:
    path_file = _report_path_status(
        path,
        directory_descriptor=directory_descriptor,
    )
    if not stat.S_ISREG(path_file.st_mode) or path_file.st_nlink != expected_links:
        raise OSError(f"Temporary report path {path} must remain one regular file with {expected_links} link(s).")
    if not _same_file_identity(path_file, opened_file):
        raise OSError(f"Temporary report path {path} changed before publication.")


def _anchor_previous_report(
    destination: Path,
    previous_path: Path,
    *,
    directory_descriptor: int | None = None,
) -> os.stat_result | None:
    try:
        destination_file = _report_path_status(
            destination,
            directory_descriptor=directory_descriptor,
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(destination_file.st_mode) or destination_file.st_nlink != 1:
        return None

    _link_report_path(
        destination,
        previous_path,
        directory_descriptor=directory_descriptor,
    )
    anchored_file = _report_path_status(
        previous_path,
        directory_descriptor=directory_descriptor,
    )
    current_destination = _report_path_status(
        destination,
        directory_descriptor=directory_descriptor,
    )
    if (
        not stat.S_ISREG(anchored_file.st_mode)
        or anchored_file.st_nlink != 2
        or not _same_file_identity(destination_file, anchored_file)
        or not _same_file_identity(current_destination, anchored_file)
    ):
        _unlink_non_directory_entry(
            previous_path,
            directory_descriptor=directory_descriptor,
        )
        raise OSError(f"Existing report path {destination} changed while it was being preserved.")
    return anchored_file


def _publish_verified_report(
    temporary_path: Path,
    anchor_path: Path,
    destination: Path,
    previous_path: Path,
    opened_file: os.stat_result,
    previous_file: os.stat_result | None,
    *,
    directory_descriptor: int | None = None,
) -> None:
    _require_same_private_temporary_file(
        temporary_path,
        opened_file,
        expected_links=2,
        directory_descriptor=directory_descriptor,
    )
    _require_same_private_temporary_file(
        anchor_path,
        opened_file,
        expected_links=2,
        directory_descriptor=directory_descriptor,
    )
    _replace_report_path(
        temporary_path,
        destination,
        directory_descriptor=directory_descriptor,
    )
    try:
        _require_same_private_temporary_file(
            destination,
            opened_file,
            expected_links=2,
            directory_descriptor=directory_descriptor,
        )
    except OSError as exc:
        restored = _restore_report_after_substitution(
            destination,
            anchor_path,
            previous_path,
            opened_file,
            previous_file,
            directory_descriptor=directory_descriptor,
        )
        if not restored:
            _unlink_non_directory_entry(
                destination,
                directory_descriptor=directory_descriptor,
            )
        raise OSError(
            f"Report path {destination} was substituted during publication; "
            f"{'the previous or intended report was restored' if restored else 'the substituted path was removed'}."
        ) from exc

    _unlink_matching_report_path(
        anchor_path,
        opened_file,
        directory_descriptor=directory_descriptor,
    )
    try:
        _require_same_private_temporary_file(
            destination,
            opened_file,
            expected_links=1,
            directory_descriptor=directory_descriptor,
        )
    except OSError as exc:
        restored = False
        if previous_file is not None:
            restored = _restore_report_after_substitution(
                destination,
                anchor_path,
                previous_path,
                opened_file,
                previous_file,
                directory_descriptor=directory_descriptor,
            )
        if not restored:
            _unlink_non_directory_entry(
                destination,
                directory_descriptor=directory_descriptor,
            )
        raise OSError(
            f"Report path {destination} changed after publication; "
            f"{'the previous report was restored' if restored else 'the substituted path was removed'}."
        ) from exc
    if previous_file is not None:
        _unlink_matching_report_path(
            previous_path,
            previous_file,
            directory_descriptor=directory_descriptor,
        )


def _restore_report_after_substitution(
    destination: Path,
    anchor_path: Path,
    previous_path: Path,
    opened_file: os.stat_result,
    previous_file: os.stat_result | None,
    *,
    directory_descriptor: int | None = None,
) -> bool:
    restore_path = anchor_path
    restore_file = opened_file
    if previous_file is not None:
        restore_path = previous_path
        restore_file = previous_file

    try:
        restore_status = _report_path_status(
            restore_path,
            directory_descriptor=directory_descriptor,
        )
        if not stat.S_ISREG(restore_status.st_mode) or not _same_file_identity(restore_status, restore_file):
            return False
        _replace_report_path(
            restore_path,
            destination,
            directory_descriptor=directory_descriptor,
        )
        published = _report_path_status(
            destination,
            directory_descriptor=directory_descriptor,
        )
        return (
            stat.S_ISREG(published.st_mode) and published.st_nlink == 1 and _same_file_identity(published, restore_file)
        )
    except OSError:
        return False


def _unlink_matching_report_path(
    path: Path,
    expected: os.stat_result,
    *,
    directory_descriptor: int | None = None,
) -> None:
    try:
        observed = _report_path_status(
            path,
            directory_descriptor=directory_descriptor,
        )
    except FileNotFoundError:
        return
    if _same_file_identity(observed, expected) and not stat.S_ISDIR(observed.st_mode):
        _quarantine_and_unlink_report_path(
            path,
            expected,
            directory_descriptor=directory_descriptor,
        )


def _unlink_report_path(
    path: Path,
    *,
    directory_descriptor: int | None,
) -> None:
    if directory_descriptor is None:
        path.unlink()
        return
    os.unlink(path.name, dir_fd=directory_descriptor)


def _unlink_non_directory_entry(
    path: Path,
    *,
    directory_descriptor: int | None = None,
) -> None:
    try:
        observed = _report_path_status(
            path,
            directory_descriptor=directory_descriptor,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(observed.st_mode):
        _quarantine_and_unlink_report_path(
            path,
            observed,
            directory_descriptor=directory_descriptor,
        )


def _restore_quarantined_report_path(
    quarantine_path: Path,
    original_path: Path,
    expected: os.stat_result,
    *,
    directory_descriptor: int | None,
) -> bool:
    """Restore a raced cleanup entry without overwriting a newer occupant."""
    try:
        _link_report_path(
            quarantine_path,
            original_path,
            directory_descriptor=directory_descriptor,
        )
        restored = _report_path_status(
            original_path,
            directory_descriptor=directory_descriptor,
        )
        if not _same_file_identity(restored, expected):
            return False
        _unlink_report_path(
            quarantine_path,
            directory_descriptor=directory_descriptor,
        )
    except OSError:
        return False
    return True


def _open_report_cleanup_anchor(
    path: Path,
    expected: os.stat_result,
    *,
    directory_descriptor: int | None,
) -> tuple[int, os.stat_result]:
    """Pin the cleanup target so an unlinked inode cannot be recycled mid-cleanup."""
    if hasattr(os, "O_PATH"):
        flags = os.O_PATH
    else:
        if not stat.S_ISREG(expected.st_mode):
            raise OSError(f"Report cleanup path {path} cannot be safely pinned on this platform.")
        flags = os.O_RDONLY
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    if directory_descriptor is None:
        descriptor = os.open(path, flags)
    else:
        descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
    try:
        anchored = os.fstat(descriptor)
        if stat.S_ISDIR(anchored.st_mode) or not _same_file_identity(anchored, expected):
            raise OSError(f"Report cleanup path {path} changed before it could be pinned.")
        return descriptor, anchored
    except BaseException as operation_failure:
        try:
            os.close(descriptor)
        except BaseException as close_failure:
            operation_failure.add_note(f"Failed to close the report cleanup anchor descriptor: {close_failure}")
        raise


def _report_cleanup_anchor_can_remain_open_during_replace() -> bool:
    """Whether an ordinary descriptor permits renaming its entry on this platform."""
    return os.name == "posix"


def _quarantine_and_unlink_report_path(
    path: Path,
    expected: os.stat_result,
    *,
    directory_descriptor: int | None,
) -> None:
    """Move a cleanup target aside and verify its inode before deleting it."""
    opened_anchor_descriptor, anchored = _open_report_cleanup_anchor(
        path,
        expected,
        directory_descriptor=directory_descriptor,
    )
    anchor_descriptor: int | None = opened_anchor_descriptor
    anchor_was_released = False
    operation_failure: BaseException | None = None
    try:
        if not _report_cleanup_anchor_can_remain_open_during_replace():
            # Native Windows does not give ordinary CRT descriptors delete sharing,
            # so retaining this descriptor would make the quarantine rename fail.
            descriptor_to_close = anchor_descriptor
            anchor_descriptor = None
            anchor_was_released = True
            assert descriptor_to_close is not None
            os.close(descriptor_to_close)
            current = _report_path_status(
                path,
                directory_descriptor=directory_descriptor,
            )
            if stat.S_ISDIR(current.st_mode) or not _same_snapshot_file_version(current, anchored):
                raise OSError(f"Report cleanup path {path} changed after its anchor was released.")

        while True:
            quarantine_path = path.parent / f"{_REPORT_CLEANUP_PREFIX}{uuid.uuid4().hex}.tmp"
            try:
                _report_path_status(
                    quarantine_path,
                    directory_descriptor=directory_descriptor,
                )
            except FileNotFoundError:
                break

        _replace_report_path(
            path,
            quarantine_path,
            directory_descriptor=directory_descriptor,
        )
        quarantined = _report_path_status(
            quarantine_path,
            directory_descriptor=directory_descriptor,
        )
        quarantine_matches_anchor = (
            _same_report_cleanup_file_version(quarantined, anchored)
            if anchor_was_released
            else _same_file_identity(quarantined, anchored)
        )
        if not stat.S_ISDIR(quarantined.st_mode) and quarantine_matches_anchor:
            _unlink_report_path(
                quarantine_path,
                directory_descriptor=directory_descriptor,
            )
            return

        restored = _restore_quarantined_report_path(
            quarantine_path,
            path,
            quarantined,
            directory_descriptor=directory_descriptor,
        )
        raise OSError(
            f"Report cleanup path {path} changed before unlink; "
            f"the replacement was {'restored' if restored else f'preserved as {quarantine_path.name}'}."
        )
    except BaseException as exc:
        operation_failure = exc
        raise
    finally:
        if anchor_descriptor is not None:
            try:
                os.close(anchor_descriptor)
            except BaseException as close_failure:
                if operation_failure is not None:
                    operation_failure.add_note(f"Failed to close the report cleanup anchor descriptor: {close_failure}")
                else:
                    raise


def _is_managed_atomic_report_artifact(entry_name: str) -> bool:
    """Recognize only temporary leaves emitted by ``_atomic_to_csv``."""
    artifact_suffixes = (".tmp", ".tmp.anchor", ".tmp.previous")
    for report_name in _MANAGED_OUTPUT_FILENAMES:
        prefix = f".{report_name}."
        if not entry_name.startswith(prefix):
            continue
        remainder = entry_name[len(prefix) :]
        for suffix in artifact_suffixes:
            if not remainder.endswith(suffix):
                continue
            run_id = remainder[: -len(suffix)]
            return len(run_id) == 32 and all(character in "0123456789abcdef" for character in run_id)
    return False


def _is_managed_report_cleanup_artifact(entry_name: str) -> bool:
    """Recognize only quarantine leaves emitted by report cleanup."""
    if not entry_name.startswith(_REPORT_CLEANUP_PREFIX) or not entry_name.endswith(".tmp"):
        return False
    run_id = entry_name[len(_REPORT_CLEANUP_PREFIX) : -len(".tmp")]
    return len(run_id) == 32 and all(character in "0123456789abcdef" for character in run_id)


def _is_managed_stale_report_artifact(entry_name: str) -> bool:
    return _is_managed_atomic_report_artifact(entry_name) or _is_managed_report_cleanup_artifact(entry_name)


def _clear_stale_atomic_report_artifacts(
    output_path: Path,
    *,
    directory_guard: PrivateRuntimeDirectory | None,
) -> None:
    """Remove abandoned managed publication and cleanup artifacts under the run lock."""
    revalidate_owned_runtime_directory(directory_guard, label="Report output directory")
    with open_owned_runtime_directory(
        directory_guard,
        label="Report output directory",
    ) as directory_descriptor:
        entries = os.listdir(directory_descriptor if directory_descriptor is not None else output_path)
        for entry_name in entries:
            if not _is_managed_stale_report_artifact(entry_name):
                continue
            artifact_path = output_path / entry_name
            try:
                observed = _report_path_status(
                    artifact_path,
                    directory_descriptor=directory_descriptor,
                )
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(observed.st_mode):
                continue
            _unlink_matching_report_path(
                artifact_path,
                observed,
                directory_descriptor=directory_descriptor,
            )
    revalidate_owned_runtime_directory(directory_guard, label="Report output directory")


def _clear_stale_workflow_research_outputs(output_dir: str) -> None:
    """Remove only prior strategy reports before starting a new optimization run."""
    output_path = Path(output_dir)
    active_output_directory = _ACTIVE_OUTPUT_DIRECTORY.get()
    if active_output_directory is None or _case_preserving_workflow_path(output_path) != active_output_directory.path:
        directory_guard = require_owned_runtime_directory(
            output_path,
            label="Report output directory",
        )
    else:
        directory_guard = active_output_directory

    with open_owned_runtime_directory(
        directory_guard,
        label="Report output directory",
    ) as directory_descriptor:
        for filename in _RESEARCH_REPORT_FILENAMES:
            report_path = output_path / filename
            try:
                if directory_descriptor is None:
                    observed = os.lstat(report_path)
                else:
                    observed = os.stat(
                        filename,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(observed.st_mode):
                raise IsADirectoryError(f"Research report path {report_path} must not be a directory.")
            _unlink_matching_report_path(
                report_path,
                observed,
                directory_descriptor=directory_descriptor,
            )
    revalidate_owned_runtime_directory(directory_guard, label="Report output directory")


def _persist_workflow_research_outputs(
    *,
    output_dir: str,
    curves: pd.DataFrame,
    optimization_summary: pd.DataFrame,
    buy_signals: pd.DataFrame,
    eligible_buy_signals: pd.DataFrame,
    sell_signals: pd.DataFrame,
) -> None:
    """Publish the strategy-derived CSVs for one current workflow generation."""
    output_path = Path(output_dir)
    _prepare_workflow_output_directory_for_publication(output_path)
    _atomic_to_csv(curves, output_path / "best_equity_curves.csv")
    _atomic_to_csv(optimization_summary, output_path / "optimization_summary.csv", index=False)
    _atomic_to_csv(buy_signals, output_path / "buy_signals.csv", index=False)
    _atomic_to_csv(eligible_buy_signals, output_path / "eligible_buy_signals.csv", index=False)
    _atomic_to_csv(sell_signals, output_path / "sell_signals.csv", index=False)


async def _run_resumable_optimizations_unlocked(
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    universe_cfg: UniverseConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    workflow_concurrency: int = DEFAULT_WORKFLOW_CONCURRENCY,
    no_color: bool = False,
    reporter: WorkflowReporter | None = None,
    tradier_cfg: TradierMarketDataConfig | None = None,
    short_buy_rsi_values: list[float] | None = None,
    strategy_state_verification: str = "trusted",
    workflow_deadline: _WorkflowDeadline | None = None,
    show_timings: bool = False,
) -> None:
    validate_alpaca_paper_endpoint(alpaca_cfg)
    _validate_optimization_grids(buy_rsi_values, profit_target_values)
    short_buy_rsi_values = list(DEFAULT_SHORT_BUY_RSI_VALUES) if short_buy_rsi_values is None else short_buy_rsi_values
    _validate_optimization_grids(short_buy_rsi_values, profit_target_values)
    concurrency = workflow_concurrency
    universe_cfg = replace(universe_cfg, sqlite_db_path=db_path)
    reporter = reporter or WorkflowReporter(no_color=no_color)
    workflow_timer = WorkflowTimer.start()
    phase_timings = WorkflowPhaseTimings()
    history_observation_run_id = uuid.uuid4().hex
    reporter.run_header(
        started_at_utc=workflow_timer.started_at_utc,
        mode=mode,
        db_path=db_path,
        output_dir=output_dir,
        workflow_concurrency=concurrency,
    )
    _check_workflow_deadline(workflow_deadline, "before workflow initialization")

    # Invalidate any previously committed broker snapshot before initialization:
    # schema/data migrations can change state represented by those CSVs.
    reconciliation_publication = _begin_alpaca_snapshot_publication(
        Path(output_dir),
        snapshot_kind="reconciliation",
    )
    with reporter.status("Initializing workflow state"):
        await _run_blocking(None, _initialize_state_db, db_path)
    _clear_stale_workflow_research_outputs(output_dir)
    startup_reconciliation_error: AlpacaReconciliationError | None = None
    try:
        with reporter.step_progress(
            "Reconciling Alpaca positions and loading workflow assets",
            total=2,
        ) as startup_progress:
            # Both steps write to the same SQLite database.  Keep startup writes
            # serialized: universe discovery performs replace-table writes while
            # reconciliation updates managed positions.
            startup_progress.start_step("Reconciling Alpaca positions")
            try:
                try:
                    reconciliation_results = await _timed_run_blocking(
                        phase_timings,
                        "alpaca",
                        _reconcile_alpaca_managed_positions_for_db,
                        db_path,
                        alpaca_cfg,
                    )
                    reconciliation_results = _safe_alpaca_workflow_result_messages(
                        reconciliation_results,
                        alpaca_cfg=alpaca_cfg,
                    )
                except AlpacaReconciliationError as exc:
                    _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg)
                    try:
                        _persist_alpaca_reconciliation_failure(
                            exc,
                            db_path=db_path,
                            output_dir=output_dir,
                            publication=reconciliation_publication,
                            reporter=reporter,
                            alpaca_cfg=alpaca_cfg,
                        )
                    except BaseException as persistence_exc:
                        detail = _best_effort_alpaca_exception_diagnostic(
                            persistence_exc,
                            alpaca_cfg=alpaca_cfg,
                        )
                        _add_safe_alpaca_workflow_note(
                            exc,
                            f"Failed to publish the failed reconciliation audit: {detail}",
                            alpaca_cfg=alpaca_cfg,
                        )
                    raise
                _persist_alpaca_reconciliation_snapshot(
                    reconciliation_results,
                    db_path=db_path,
                    output_dir=output_dir,
                    publication=reconciliation_publication,
                    alpaca_cfg=alpaca_cfg,
                )
            except AlpacaReconciliationError as exc:
                # A progress context's teardown runs while this exception is in
                # flight and could otherwise replace its structured results.
                startup_reconciliation_error = exc
                raise
            startup_progress.finish_step()
            startup_progress.start_step("Loading workflow assets")
            workflow_asset_groups = await _run_blocking(
                None,
                _load_or_refresh_workflow_assets_for_db,
                db_path,
                universe_cfg,
            )
            startup_progress.finish_step()
    except BaseException as exc:
        if startup_reconciliation_error is None or exc is startup_reconciliation_error:
            raise
        detail = _best_effort_alpaca_exception_diagnostic(exc, alpaca_cfg=alpaca_cfg)
        _add_safe_alpaca_workflow_note(
            startup_reconciliation_error,
            f"Failed to finish the startup progress display while propagating reconciliation failure: {detail}",
            alpaca_cfg=alpaca_cfg,
        )
    if startup_reconciliation_error is not None:
        # Preserve the typed failure even if a nonstandard progress context
        # manager suppresses the exception raised by the reconciliation body.
        raise _sanitize_alpaca_workflow_exception(
            startup_reconciliation_error,
            alpaca_cfg=alpaca_cfg,
        )

    workflow_asset_groups = _normalize_workflow_asset_groups(workflow_asset_groups)
    workflow_specs = [
        {
            "key": "long",
            "label": LONG_WORKFLOW_LABEL,
            "rsi_entry_rule": "lower",
            "buy_rsi_values": buy_rsi_values,
            "assets": workflow_asset_groups["long"],
        },
        {
            "key": "short",
            "label": SHORT_WORKFLOW_LABEL,
            "rsi_entry_rule": "upper",
            "buy_rsi_values": short_buy_rsi_values,
            "assets": workflow_asset_groups["short"],
        },
    ]

    reporter.universe_assets(_combined_workflow_assets(workflow_asset_groups))

    asset_run_results_by_side: dict[str, list[AssetRunResult]] = {}
    all_asset_run_results: list[AssetRunResult] = []
    market_data_session = _WorkflowMarketDataSession()
    for workflow_spec in workflow_specs:
        workflow_key = str(workflow_spec["key"])
        workflow_label = str(workflow_spec["label"])
        workflow_assets = workflow_spec["assets"]
        assert isinstance(workflow_assets, pd.DataFrame)
        jobs = _workflow_jobs(workflow_assets, workflow_label=workflow_label)
        if jobs:
            with reporter.asset_progress(len(jobs), workflow_label=workflow_label) as asset_progress:
                asset_run_results = await _run_asset_pipeline(
                    jobs=jobs,
                    concurrency=concurrency,
                    db_path=db_path,
                    mode=mode,
                    base_cfg=base_cfg,
                    tradier_cfg=tradier_cfg,
                    buy_rsi_values=list(workflow_spec["buy_rsi_values"]),
                    profit_target_values=profit_target_values,
                    asset_progress=asset_progress,
                    phase_timings=phase_timings,
                    rsi_entry_rule=str(workflow_spec["rsi_entry_rule"]),
                    history_observation_run_id=history_observation_run_id,
                    market_data_session=market_data_session,
                    strategy_state_verification=strategy_state_verification,
                    workflow_deadline=workflow_deadline,
                )
        else:
            asset_run_results = []
        asset_run_results_by_side[workflow_key] = asset_run_results
        all_asset_run_results.extend(asset_run_results)

    completed_runs = [result for result in all_asset_run_results if result.status == "done"]
    if not completed_runs:
        details = (
            "; ".join(f"{result.asset_symbol}: {result.message}" for result in all_asset_run_results)
            if all_asset_run_results
            else "No executable assets were run."
        )
        raise WorkflowRunError(f"No asset workflows completed successfully. {details}")

    _check_workflow_deadline(workflow_deadline, "before reporting")
    with reporter.status("Building workflow reports"):
        side_outputs: list[WorkflowSideOutput] = []
        realized_pnl_summary = pd.DataFrame()
        report_deadline_kwargs: dict[str, object] = {}
        if workflow_deadline is not None:
            report_deadline_kwargs["deadline_check"] = partial(
                _check_workflow_deadline,
                workflow_deadline,
                "during report generation",
            )
        for workflow_spec in workflow_specs:
            workflow_key = str(workflow_spec["key"])
            workflow_label = str(workflow_spec["label"])
            workflow_assets = workflow_spec["assets"]
            assert isinstance(workflow_assets, pd.DataFrame)
            side_asset_run_results = asset_run_results_by_side.get(workflow_key, [])
            (
                side_optimization_summary,
                side_curves,
                side_buy_signals,
                side_eligible_buy_signals,
                side_sell_signals,
                realized_pnl_summary,
            ) = await _timed_run_blocking(
                phase_timings,
                "report_generation",
                _build_reports_for_db,
                db_path,
                workflow_assets,
                base_cfg,
                _completed_asset_pairs(side_asset_run_results),
                workflow_label,
                str(workflow_spec["rsi_entry_rule"]),
                list(workflow_spec["buy_rsi_values"]),
                profit_target_values,
                **report_deadline_kwargs,
            )
            side_outputs.append(
                WorkflowSideOutput(
                    label=workflow_label,
                    universe_assets=workflow_assets,
                    buy_rsi_values=list(workflow_spec["buy_rsi_values"]),
                    rsi_entry_rule=str(workflow_spec["rsi_entry_rule"]),
                    asset_run_results=side_asset_run_results,
                    optimization_summary=side_optimization_summary,
                    curves=side_curves,
                    buy_signals=side_buy_signals,
                    eligible_buy_signals=side_eligible_buy_signals,
                    sell_signals=side_sell_signals,
                )
            )

        optimization_summary = _concat_report_frames([side.optimization_summary for side in side_outputs])
        curves = _concat_report_frames([side.curves for side in side_outputs], axis=1)
        buy_signals = _concat_report_frames([side.buy_signals for side in side_outputs])
        eligible_buy_signals = _concat_report_frames([side.eligible_buy_signals for side in side_outputs])
        sell_signals = _concat_report_frames([side.sell_signals for side in side_outputs])
    _check_workflow_deadline(workflow_deadline, "after reporting")
    _persist_workflow_research_outputs(
        output_dir=output_dir,
        curves=curves,
        optimization_summary=optimization_summary,
        buy_signals=buy_signals,
        eligible_buy_signals=eligible_buy_signals,
        sell_signals=sell_signals,
    )
    # Broker submission is the irreversible side effect in this phase. Attempt
    # to publish its exact result immediately, but defer any output/status
    # failure until every possible broker side effect has been reconciled.
    _check_workflow_deadline(workflow_deadline, "before Alpaca buy submission")
    broker_publication = _begin_alpaca_snapshot_publication(
        Path(output_dir),
        snapshot_kind="workflow",
    )
    buy_batch_error: AlpacaBuyBatchError | None = None
    unknown_submission_failure: BaseException | None = None
    protective_reconciliation_required = False
    submission_worker_invoked = False
    submission_outcome_available = False
    broker_snapshot_committed = False
    post_submit_failures: list[tuple[str, BaseException]] = []
    try:
        with reporter.status("Preparing Alpaca order results"):
            try:
                submission_worker_invoked = True
                order_results = await _timed_run_blocking(
                    phase_timings,
                    "alpaca",
                    _submit_alpaca_paper_buy_orders_for_db,
                    db_path,
                    buy_signals,
                    alpaca_cfg,
                )
            except AlpacaBuyBatchError as exc:
                _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg)
                buy_batch_error = exc
                order_results = exc.results
            except BaseException as exc:
                unknown_submission_failure = exc
                protective_reconciliation_required = True
                post_submit_failures.append(("complete the Alpaca buy submission worker", exc))
                safe_failure = RuntimeError(_best_effort_alpaca_exception_diagnostic(exc, alpaca_cfg=alpaca_cfg))
                order_results = _unknown_alpaca_buy_submission_results(
                    buy_signals,
                    safe_failure,
                    alpaca_cfg=alpaca_cfg,
                )
            order_results = _safe_alpaca_workflow_result_messages(
                order_results,
                alpaca_cfg=alpaca_cfg,
            )
            submission_outcome_available = True
            try:
                _publish_alpaca_snapshot_csv(
                    broker_publication,
                    order_results,
                    filename="alpaca_order_results.csv",
                    index=False,
                )
            except BaseException as exc:
                post_submit_failures.append(("publish Alpaca buy results", exc))
    except BaseException as exc:
        if not submission_worker_invoked:
            raise
        protective_reconciliation_required = True
        action = (
            "finish the Alpaca submission status display"
            if submission_outcome_available
            else "construct the unknown Alpaca submission diagnostic"
        )
        post_submit_failures.append((action, exc))
        if not submission_outcome_available:
            if unknown_submission_failure is None:
                unknown_submission_failure = exc
                post_submit_failures.insert(0, ("complete the Alpaca buy submission worker", exc))
            try:
                safe_unknown_submission_failure = RuntimeError(
                    _best_effort_alpaca_exception_diagnostic(
                        unknown_submission_failure,
                        alpaca_cfg=alpaca_cfg,
                    )
                )
                order_results = _unknown_alpaca_buy_submission_results(
                    buy_signals,
                    safe_unknown_submission_failure,
                    alpaca_cfg=alpaca_cfg,
                )
            except BaseException as diagnostic_exc:
                post_submit_failures.append(
                    ("construct the fallback unknown Alpaca submission diagnostic", diagnostic_exc)
                )
                order_results = pd.DataFrame(
                    [
                        {
                            "Status": "submission_unknown",
                            "Message": (
                                "Alpaca buy submission outcome is unknown and its detailed diagnostic "
                                "could not be constructed; protective reconciliation was required."
                            ),
                        }
                    ]
                )
            order_results = _safe_alpaca_workflow_result_messages(
                order_results,
                alpaca_cfg=alpaca_cfg,
            )
            submission_outcome_available = True
            try:
                _publish_alpaca_snapshot_csv(
                    broker_publication,
                    order_results,
                    filename="alpaca_order_results.csv",
                    index=False,
                )
            except BaseException as publish_exc:
                post_submit_failures.append(("publish the fallback unknown Alpaca submission diagnostic", publish_exc))

    def attach_post_submit_failures(failure: BaseException) -> None:
        for action, deferred_failure in post_submit_failures:
            detail = _best_effort_alpaca_exception_diagnostic(deferred_failure, alpaca_cfg=alpaca_cfg)
            _add_safe_alpaca_workflow_note(
                failure,
                f"Failed to {action} after broker submission: {detail}",
                alpaca_cfg=alpaca_cfg,
            )

    def finish_workflow_snapshot(results: pd.DataFrame) -> None:
        nonlocal broker_snapshot_committed
        snapshot_failures = _finish_alpaca_workflow_snapshot(
            broker_publication,
            order_results,
            results,
            db_path=db_path,
            alpaca_cfg=alpaca_cfg,
        )
        post_submit_failures.extend(snapshot_failures)
        broker_snapshot_committed = not snapshot_failures

    def render_failed_reconciliation(results: pd.DataFrame) -> None:
        try:
            reporter.reconciliation(results)
        except BaseException as rendering_exc:
            post_submit_failures.append(("render the failed post-buy reconciliation", rendering_exc))

    def raise_post_submit_failure() -> None:
        if not post_submit_failures:
            return
        first_action, first_failure = post_submit_failures[0]
        _add_safe_alpaca_workflow_note(
            first_failure,
            f"Failure occurred while attempting to {first_action} after broker submission.",
            alpaca_cfg=alpaca_cfg,
        )
        for action, deferred_failure in post_submit_failures[1:]:
            detail = _best_effort_alpaca_exception_diagnostic(deferred_failure, alpaca_cfg=alpaca_cfg)
            _add_safe_alpaca_workflow_note(
                first_failure,
                f"Also failed to {action} after broker submission: {detail}",
                alpaca_cfg=alpaca_cfg,
            )
        raise _sanitize_alpaca_workflow_exception(first_failure, alpaca_cfg=alpaca_cfg)

    if buy_batch_error is not None:
        exc = buy_batch_error
        if alpaca_cfg.enabled and exc.broker_side_effects_possible:
            try:
                post_buy_reconciliation = await _timed_run_blocking(
                    phase_timings,
                    "alpaca",
                    _reconcile_alpaca_managed_positions_for_db,
                    db_path,
                    alpaca_cfg,
                )
            except AlpacaReconciliationError as reconciliation_exc:
                combined_results = pd.concat(
                    [reconciliation_results, reconciliation_exc.results],
                    ignore_index=True,
                )
                reconciliation_detail = _best_effort_alpaca_exception_diagnostic(
                    reconciliation_exc,
                    alpaca_cfg=alpaca_cfg,
                )
                buy_batch_detail = _best_effort_alpaca_exception_diagnostic(
                    exc,
                    alpaca_cfg=alpaca_cfg,
                )
                combined_exc = _safe_alpaca_reconciliation_error(
                    f"{reconciliation_detail}; Alpaca buy batch also failed: {buy_batch_detail}",
                    combined_results,
                    alpaca_cfg=alpaca_cfg,
                )
                finish_workflow_snapshot(combined_results)
                render_failed_reconciliation(combined_exc.results)
                attach_post_submit_failures(combined_exc)
                raise combined_exc from _alpaca_public_exception_cause(reconciliation_exc, cfg=alpaca_cfg)
            except BaseException as reconciliation_exc:
                failed_results = _alpaca_reconciliation_failure_results(
                    reconciliation_results,
                    action="complete post-buy reconciliation",
                    failure=reconciliation_exc,
                    alpaca_cfg=alpaca_cfg,
                )
                finish_workflow_snapshot(failed_results)
                render_failed_reconciliation(failed_results)
                buy_batch_detail = _best_effort_alpaca_exception_diagnostic(
                    exc,
                    alpaca_cfg=alpaca_cfg,
                )
                _add_safe_alpaca_workflow_note(
                    reconciliation_exc,
                    f"Alpaca buy batch also failed: {buy_batch_detail}",
                    alpaca_cfg=alpaca_cfg,
                )
                attach_post_submit_failures(reconciliation_exc)
                _sanitize_alpaca_workflow_exception(
                    reconciliation_exc,
                    alpaca_cfg=alpaca_cfg,
                )
                raise
            reconciliation_results = pd.concat(
                [reconciliation_results, post_buy_reconciliation],
                ignore_index=True,
            )
            finish_workflow_snapshot(reconciliation_results)
        else:
            finish_workflow_snapshot(reconciliation_results)
        _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg)
        try:
            reporter.order_results(exc.results)
        except BaseException as rendering_exc:
            post_submit_failures.append(("render Alpaca buy results", rendering_exc))
        attach_post_submit_failures(exc)
        raise _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg)

    submitted_buy_results = (
        order_results[order_results["Status"].isin(BUY_RESULTS_REQUIRING_RECONCILIATION)]
        if "Status" in order_results
        else pd.DataFrame()
    )
    if alpaca_cfg.enabled and (protective_reconciliation_required or not submitted_buy_results.empty):
        try:
            post_buy_reconciliation = await _timed_run_blocking(
                phase_timings,
                "alpaca",
                _reconcile_alpaca_managed_positions_for_db,
                db_path,
                alpaca_cfg,
            )
        except AlpacaReconciliationError as exc:
            combined_results = pd.concat(
                [reconciliation_results, exc.results],
                ignore_index=True,
            )
            failure_message = _best_effort_alpaca_exception_diagnostic(
                exc,
                alpaca_cfg=alpaca_cfg,
            )
            combined_exc = _safe_alpaca_reconciliation_error(
                failure_message,
                combined_results,
                alpaca_cfg=alpaca_cfg,
            )
            finish_workflow_snapshot(combined_results)
            render_failed_reconciliation(combined_exc.results)
            attach_post_submit_failures(combined_exc)
            raise combined_exc from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)
        except BaseException as reconciliation_exc:
            failed_results = _alpaca_reconciliation_failure_results(
                reconciliation_results,
                action="complete post-buy reconciliation",
                failure=reconciliation_exc,
                alpaca_cfg=alpaca_cfg,
            )
            finish_workflow_snapshot(failed_results)
            render_failed_reconciliation(failed_results)
            attach_post_submit_failures(reconciliation_exc)
            _sanitize_alpaca_workflow_exception(
                reconciliation_exc,
                alpaca_cfg=alpaca_cfg,
            )
            raise
        reconciliation_results = pd.concat(
            [reconciliation_results, post_buy_reconciliation],
            ignore_index=True,
        )
        finish_workflow_snapshot(reconciliation_results)
    else:
        # Even an empty or preflight-only buy result started a workflow
        # publication and invalidated the startup reconciliation manifest.
        # Commit that exact no-side-effect outcome before final rendering can
        # fail, just as we do after a broker-visible buy.
        finish_workflow_snapshot(reconciliation_results)
    raise_post_submit_failure()
    with reporter.status("Loading managed Alpaca positions"):
        managed_positions = await _run_blocking(None, _load_alpaca_managed_positions_for_db, db_path)
        realized_pnl_summary = await _run_blocking(None, _load_alpaca_realized_pnl_for_db, db_path)
        managed_positions = _safe_alpaca_workflow_result_messages(
            managed_positions,
            alpaca_cfg=alpaca_cfg,
        )
        realized_pnl_summary = _safe_alpaca_workflow_result_messages(
            realized_pnl_summary,
            alpaca_cfg=alpaca_cfg,
        )
    sell_reconciliation_results = reconciliation_results[reconciliation_results["Action"].eq("sell")]

    _write_workflow_outputs(
        mode=mode,
        db_path=db_path,
        base_cfg=base_cfg,
        buy_rsi_values=buy_rsi_values,
        profit_target_values=profit_target_values,
        alpaca_cfg=alpaca_cfg,
        output_dir=output_dir,
        workflow_concurrency=concurrency,
        reporter=reporter,
        asset_run_results=all_asset_run_results,
        optimization_summary=optimization_summary,
        curves=curves,
        buy_signals=buy_signals,
        eligible_buy_signals=eligible_buy_signals,
        sell_signals=sell_signals,
        realized_pnl_summary=realized_pnl_summary,
        managed_positions=managed_positions,
        reconciliation_results=reconciliation_results,
        sell_reconciliation_results=sell_reconciliation_results,
        order_results=order_results,
        workflow_timer=workflow_timer,
        phase_timings=phase_timings,
        workflow_side_outputs=side_outputs,
        short_buy_rsi_values=short_buy_rsi_values,
        research_outputs_published=True,
        broker_snapshot_committed=broker_snapshot_committed,
        market_data_batch_count=market_data_session.batch_count,
        market_data_individual_retry_count=market_data_session.individual_retry_count,
        show_timings=show_timings,
    )


def run_resumable_optimizations_async(
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    universe_cfg: UniverseConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    workflow_concurrency: int = DEFAULT_WORKFLOW_CONCURRENCY,
    no_color: bool = False,
    reporter: WorkflowReporter | None = None,
    tradier_cfg: TradierMarketDataConfig | None = None,
    short_buy_rsi_values: list[float] | None = None,
    strategy_state_verification: str = "trusted",
    workflow_deadline_epoch: int | float | None = None,
    show_timings: bool = False,
) -> Coroutine[Any, Any, None]:
    """Snapshot caller-owned value inputs and return the workflow coroutine."""
    base_cfg_snapshot = replace(base_cfg)
    universe_cfg_snapshot = replace(universe_cfg)
    buy_rsi_values_snapshot = list(buy_rsi_values)
    profit_target_values_snapshot = list(profit_target_values)
    alpaca_cfg_snapshot = replace(alpaca_cfg)
    tradier_cfg_snapshot = replace(tradier_cfg) if tradier_cfg is not None else None
    short_buy_rsi_values_snapshot = list(
        DEFAULT_SHORT_BUY_RSI_VALUES if short_buy_rsi_values is None else short_buy_rsi_values
    )
    return _run_resumable_optimizations_async_from_snapshot(
        mode=mode,
        db_path=db_path,
        base_cfg=base_cfg_snapshot,
        universe_cfg=universe_cfg_snapshot,
        buy_rsi_values=buy_rsi_values_snapshot,
        profit_target_values=profit_target_values_snapshot,
        alpaca_cfg=alpaca_cfg_snapshot,
        output_dir=output_dir,
        workflow_concurrency=workflow_concurrency,
        no_color=no_color,
        reporter=reporter,
        tradier_cfg=tradier_cfg_snapshot,
        short_buy_rsi_values=short_buy_rsi_values_snapshot,
        strategy_state_verification=strategy_state_verification,
        workflow_deadline_epoch=workflow_deadline_epoch,
        show_timings=show_timings,
    )


async def _run_resumable_optimizations_async_from_snapshot(
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    universe_cfg: UniverseConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    workflow_concurrency: int,
    no_color: bool,
    reporter: WorkflowReporter | None,
    tradier_cfg: TradierMarketDataConfig | None,
    short_buy_rsi_values: list[float],
    strategy_state_verification: str,
    workflow_deadline_epoch: int | float | None,
    show_timings: bool,
) -> None:
    boundary_sensitive_values = _alpaca_config_sensitive_values(alpaca_cfg)
    public_failure: BaseException | None = None
    suppress_chain = False
    with _alpaca_public_durable_diagnostic_scope(alpaca_cfg, boundary_sensitive_values):
        try:
            return await _run_resumable_optimizations_async_from_snapshot_impl(
                mode=mode,
                db_path=db_path,
                base_cfg=base_cfg,
                universe_cfg=universe_cfg,
                buy_rsi_values=buy_rsi_values,
                profit_target_values=profit_target_values,
                alpaca_cfg=alpaca_cfg,
                output_dir=output_dir,
                workflow_concurrency=workflow_concurrency,
                no_color=no_color,
                reporter=reporter,
                tradier_cfg=tradier_cfg,
                short_buy_rsi_values=short_buy_rsi_values,
                strategy_state_verification=strategy_state_verification,
                workflow_deadline_epoch=workflow_deadline_epoch,
                show_timings=show_timings,
            )
        except BaseException as exc:
            suppress_chain = _alpaca_public_exception_cause(exc, cfg=alpaca_cfg) is None
            public_failure = _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg) if suppress_chain else exc

    assert public_failure is not None
    if suppress_chain:
        raise public_failure from None
    raise public_failure


async def _run_resumable_optimizations_async_from_snapshot_impl(
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    universe_cfg: UniverseConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    workflow_concurrency: int,
    no_color: bool,
    reporter: WorkflowReporter | None,
    tradier_cfg: TradierMarketDataConfig | None,
    short_buy_rsi_values: list[float],
    strategy_state_verification: str,
    workflow_deadline_epoch: int | float | None,
    show_timings: bool,
) -> None:
    _validate_workflow_mode(mode)
    _validate_database_path(db_path)
    _validate_output_directory_path(output_dir)
    validate_alpaca_paper_endpoint(alpaca_cfg)
    _validate_optimization_grids(buy_rsi_values, profit_target_values)
    _validate_optimization_grids(short_buy_rsi_values, profit_target_values)
    if strategy_state_verification not in {"trusted", "canonical"}:
        raise ValueError("strategy_state_verification must be 'trusted' or 'canonical'.")
    workflow_deadline = (
        None if workflow_deadline_epoch is None else _WorkflowDeadline.from_epoch(workflow_deadline_epoch)
    )
    _check_workflow_deadline(workflow_deadline, "before acquiring the workflow lock")
    validate_runtime_configuration(
        base_cfg=base_cfg,
        universe_cfg=replace(universe_cfg, sqlite_db_path=db_path),
        alpaca_cfg=alpaca_cfg,
        tradier_cfg=tradier_cfg,
        workflow_concurrency=workflow_concurrency,
    )
    with _workflow_run_lock(
        db_path,
        output_dir,
        serialize_alpaca_account=alpaca_cfg.enabled or alpaca_cfg.sell_enabled,
    ) as locked_output_dir:
        locked_db_path = locked_output_dir.database_path
        _check_workflow_deadline(workflow_deadline, "after acquiring the workflow lock")
        await _run_resumable_optimizations_unlocked(
            mode=mode,
            db_path=locked_db_path,
            base_cfg=base_cfg,
            universe_cfg=replace(universe_cfg, sqlite_db_path=locked_db_path),
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            alpaca_cfg=alpaca_cfg,
            output_dir=locked_output_dir,
            workflow_concurrency=workflow_concurrency,
            no_color=no_color,
            reporter=reporter,
            tradier_cfg=tradier_cfg,
            short_buy_rsi_values=short_buy_rsi_values,
            strategy_state_verification=strategy_state_verification,
            workflow_deadline=workflow_deadline,
            show_timings=show_timings,
        )


def run_alpaca_reconciliation(
    *,
    db_path: str,
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    no_color: bool = False,
) -> None:
    _run_alpaca_workflow_diagnostic_boundary(
        lambda: _run_alpaca_reconciliation_impl(
            db_path=db_path,
            alpaca_cfg=alpaca_cfg,
            output_dir=output_dir,
            no_color=no_color,
        ),
        alpaca_cfg=alpaca_cfg,
    )


def _run_alpaca_reconciliation_impl(
    *,
    db_path: str,
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    no_color: bool = False,
) -> None:
    """Quickly reconcile managed positions without rerunning market-data research."""
    _validate_database_path(db_path)
    _validate_output_directory_path(output_dir)
    validate_alpaca_reconciliation_configuration(alpaca_cfg)
    if not alpaca_cfg.sell_enabled:
        raise ValueError("Reconciliation-only mode requires --alpaca-submit-sell-orders.")

    reporter = WorkflowReporter(no_color=no_color)
    workflow_timer = WorkflowTimer.start()
    with _workflow_run_lock(
        db_path,
        output_dir,
        serialize_alpaca_account=True,
    ) as locked_output_dir:
        locked_db_path = locked_output_dir.database_path
        # Initialization performs managed-state migrations, so retire the old
        # report generation before it can change any snapshot-covered row.
        reconciliation_publication = _begin_alpaca_snapshot_publication(
            Path(locked_output_dir),
            snapshot_kind="reconciliation",
        )
        _initialize_state_db(locked_db_path)
        reconciliation_results: pd.DataFrame | None = None
        reconciliation_error: AlpacaReconciliationError | None = None
        try:
            with reporter.status("Reconciling managed Alpaca positions"):
                try:
                    try:
                        reconciliation_results = _reconcile_alpaca_managed_positions_for_db(
                            locked_db_path,
                            alpaca_cfg,
                        )
                        reconciliation_results = _safe_alpaca_workflow_result_messages(
                            reconciliation_results,
                            alpaca_cfg=alpaca_cfg,
                        )
                    except AlpacaReconciliationError as exc:
                        _sanitize_alpaca_workflow_exception(exc, alpaca_cfg=alpaca_cfg)
                        try:
                            _persist_alpaca_reconciliation_failure(
                                exc,
                                db_path=locked_db_path,
                                output_dir=locked_output_dir,
                                publication=reconciliation_publication,
                                reporter=reporter,
                                alpaca_cfg=alpaca_cfg,
                            )
                        except BaseException as persistence_exc:
                            detail = _best_effort_alpaca_exception_diagnostic(
                                persistence_exc,
                                alpaca_cfg=alpaca_cfg,
                            )
                            _add_safe_alpaca_workflow_note(
                                exc,
                                f"Failed to publish the failed reconciliation audit: {detail}",
                                alpaca_cfg=alpaca_cfg,
                            )
                        raise
                    _persist_alpaca_reconciliation_snapshot(
                        reconciliation_results,
                        db_path=locked_db_path,
                        output_dir=locked_output_dir,
                        publication=reconciliation_publication,
                        alpaca_cfg=alpaca_cfg,
                    )
                except AlpacaReconciliationError as exc:
                    # Preserve the typed failure across status teardown so its
                    # per-position reconciliation results remain available.
                    reconciliation_error = exc
                    raise
                realized_pnl_summary = _load_alpaca_realized_pnl_for_db(locked_db_path)
                realized_pnl_summary = _safe_alpaca_workflow_result_messages(
                    realized_pnl_summary,
                    alpaca_cfg=alpaca_cfg,
                )
        except BaseException as exc:
            if reconciliation_error is not None:
                if exc is reconciliation_error:
                    raise
                detail = _best_effort_alpaca_exception_diagnostic(
                    exc,
                    alpaca_cfg=alpaca_cfg,
                )
                _add_safe_alpaca_workflow_note(
                    reconciliation_error,
                    "Failed to finish reconciliation status handling while propagating reconciliation failure: "
                    f"{detail}",
                    alpaca_cfg=alpaca_cfg,
                )
            else:
                if reconciliation_results is None:
                    raise
                raise _alpaca_reconciliation_followup_error(
                    reconciliation_results,
                    action="finish reconciliation audit and status handling",
                    failure=exc,
                    alpaca_cfg=alpaca_cfg,
                ) from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)
        if reconciliation_error is not None:
            # Preserve the typed failure even if a nonstandard status context
            # manager suppresses the exception raised by the reconciliation body.
            raise _sanitize_alpaca_workflow_exception(
                reconciliation_error,
                alpaca_cfg=alpaca_cfg,
            )

        assert reconciliation_results is not None
        try:
            reporter.reconciliation(reconciliation_results)
            reporter.realized_pnl_summary(realized_pnl_summary)
            reporter.workflow_footer(workflow_timer.elapsed_seconds())
        except BaseException as exc:
            raise _alpaca_reconciliation_followup_error(
                reconciliation_results,
                action="render the reconciliation report",
                failure=exc,
                alpaca_cfg=alpaca_cfg,
            ) from _alpaca_public_exception_cause(exc, cfg=alpaca_cfg)


def run_resumable_optimizations(
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    universe_cfg: UniverseConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    workflow_concurrency: int = DEFAULT_WORKFLOW_CONCURRENCY,
    no_color: bool = False,
    reporter: WorkflowReporter | None = None,
    tradier_cfg: TradierMarketDataConfig | None = None,
    short_buy_rsi_values: list[float] | None = None,
    strategy_state_verification: str = "trusted",
    workflow_deadline_epoch: int | float | None = None,
    show_timings: bool = False,
) -> None:
    asyncio.run(
        run_resumable_optimizations_async(
            mode=mode,
            db_path=db_path,
            base_cfg=base_cfg,
            universe_cfg=universe_cfg,
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            alpaca_cfg=alpaca_cfg,
            output_dir=output_dir,
            workflow_concurrency=workflow_concurrency,
            no_color=no_color,
            reporter=reporter,
            tradier_cfg=tradier_cfg,
            short_buy_rsi_values=short_buy_rsi_values,
            strategy_state_verification=strategy_state_verification,
            workflow_deadline_epoch=workflow_deadline_epoch,
            show_timings=show_timings,
        )
    )


def _write_workflow_outputs(
    *,
    mode: str,
    db_path: str,
    base_cfg: BacktestConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    alpaca_cfg: AlpacaOrderConfig,
    output_dir: str,
    workflow_concurrency: int,
    reporter: WorkflowReporter,
    asset_run_results: list[AssetRunResult],
    optimization_summary: pd.DataFrame,
    curves: pd.DataFrame,
    buy_signals: pd.DataFrame,
    eligible_buy_signals: pd.DataFrame,
    sell_signals: pd.DataFrame,
    realized_pnl_summary: pd.DataFrame,
    managed_positions: pd.DataFrame,
    reconciliation_results: pd.DataFrame,
    sell_reconciliation_results: pd.DataFrame,
    order_results: pd.DataFrame,
    workflow_timer: WorkflowTimer,
    phase_timings: WorkflowPhaseTimings | None = None,
    workflow_side_outputs: list[WorkflowSideOutput] | None = None,
    short_buy_rsi_values: list[float] | None = None,
    research_outputs_published: bool = False,
    broker_snapshot_committed: bool = False,
    market_data_batch_count: int = 0,
    market_data_individual_retry_count: int = 0,
    show_timings: bool = False,
) -> None:
    report_output_started = time.perf_counter()
    terminal_order_results, terminal_reconciliation_results = _terminal_alpaca_display_results(
        managed_positions=managed_positions,
        reconciliation_results=reconciliation_results,
        order_results=order_results,
    )
    reporter.settings(
        mode=mode,
        db_path=db_path,
        workflow_concurrency=workflow_concurrency,
        risk_free_symbol=RISK_FREE_SYMBOL,
        buy_rsi_values=buy_rsi_values,
        short_buy_rsi_values=short_buy_rsi_values,
        profit_target_values=profit_target_values,
    )
    if workflow_side_outputs:
        for side_output in workflow_side_outputs:
            reporter.asset_run_summary(
                [result.as_output_row(workflow=side_output.label) for result in side_output.asset_run_results],
                title=f"{side_output.label} Asset Run Summary",
            )
            reporter.optimization_summary(
                side_output.optimization_summary.drop(columns="Workflow", errors="ignore"),
                title=f"Best Sharpe Parameters By Asset — {side_output.label}",
            )
    else:
        reporter.asset_run_summary([result.as_output_row() for result in asset_run_results])
        reporter.optimization_summary(optimization_summary.drop(columns="Workflow", errors="ignore"))
    reporter.signal_report(
        "Buy Signals For Next Open",
        buy_signals,
        empty_message="No optimized assets with more than one trade and Sharpe >= 1.0 have a pending buy signal.",
    )

    output_path = Path(output_dir)
    _prepare_workflow_output_directory_for_publication(output_path)

    if not research_outputs_published:
        _persist_workflow_research_outputs(
            output_dir=output_dir,
            curves=curves,
            optimization_summary=optimization_summary,
            buy_signals=buy_signals,
            eligible_buy_signals=eligible_buy_signals,
            sell_signals=sell_signals,
        )
    broker_publication = (
        None
        if broker_snapshot_committed
        else _begin_alpaca_snapshot_publication(
            output_path,
            snapshot_kind="workflow",
        )
    )
    if broker_publication is not None:
        _publish_alpaca_snapshot_csv(
            broker_publication,
            realized_pnl_summary,
            filename="alpaca_realized_pnl.csv",
            index=False,
        )
        _publish_alpaca_snapshot_csv(
            broker_publication,
            managed_positions,
            filename="managed_positions.csv",
            index=False,
        )
        _publish_alpaca_snapshot_csv(
            broker_publication,
            reconciliation_results,
            filename="alpaca_reconciliation_results.csv",
            index=False,
        )
    if alpaca_cfg.enabled:
        reporter.order_results(terminal_order_results)
    reporter.buy_signal_eligibility_summary(
        buy_signals=buy_signals,
        eligible_buy_signals=eligible_buy_signals,
        order_results=order_results,
    )
    if broker_publication is not None:
        _publish_alpaca_snapshot_csv(
            broker_publication,
            order_results,
            filename="alpaca_order_results.csv",
            index=False,
        )

    if alpaca_cfg.sell_enabled:
        reporter.reconciliation(terminal_reconciliation_results)
    reporter.realized_pnl_summary(realized_pnl_summary)
    if broker_publication is not None:
        _publish_alpaca_snapshot_csv(
            broker_publication,
            sell_reconciliation_results,
            filename="alpaca_sell_order_results.csv",
            index=False,
        )
        _commit_alpaca_snapshot_publication(broker_publication)
    if phase_timings is not None:
        phase_timings.add("report_generation", time.perf_counter() - report_output_started)
    if phase_timings is not None and show_timings:
        phase_snapshot = phase_timings.snapshot()
        reporter.workflow_timings(
            download_seconds=phase_snapshot.download_seconds,
            state_validation_seconds=phase_snapshot.state_validation_seconds,
            grid_compute_seconds=phase_snapshot.grid_compute_seconds,
            db_sync_seconds=phase_snapshot.db_sync_seconds,
            report_generation_seconds=phase_snapshot.report_generation_seconds,
            alpaca_seconds=phase_snapshot.alpaca_seconds,
            batch_count=market_data_batch_count,
            individual_retry_count=market_data_individual_retry_count,
            rebuild_count=sum(
                result.status == "done" and result.action == "Rebuilding" for result in asset_run_results
            ),
            update_count=sum(result.status == "done" and result.action == "Updating" for result in asset_run_results),
        )
    reporter.workflow_footer(workflow_timer.elapsed_seconds())


def _terminal_alpaca_display_results(
    *,
    managed_positions: pd.DataFrame,
    reconciliation_results: pd.DataFrame,
    order_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_order_results = order_results.copy()
    display_reconciliation_results = reconciliation_results.copy()
    display_id_by_position, display_id_by_buy_client_order_id = _managed_position_display_id_maps(managed_positions)

    if display_id_by_buy_client_order_id and "Client Order ID" in display_order_results.columns:
        display_order_results["Display ID"] = (
            display_order_results["Client Order ID"].astype(str).map(display_id_by_buy_client_order_id)
        )
    if display_id_by_position and "Position ID" in display_reconciliation_results.columns:
        display_reconciliation_results["Display ID"] = pd.to_numeric(
            display_reconciliation_results["Position ID"], errors="coerce"
        ).map(display_id_by_position)
    return display_order_results, display_reconciliation_results


def _managed_position_display_id_maps(managed_positions: pd.DataFrame) -> tuple[dict[int, int], dict[str, int]]:
    if managed_positions.empty or not {"id", "buy_client_order_id"}.issubset(managed_positions.columns):
        return {}, {}

    managed = managed_positions[["id", "buy_client_order_id"]].copy()
    managed["id"] = pd.to_numeric(managed["id"], errors="coerce")
    managed = managed.dropna(subset=["id"])
    if managed.empty:
        return {}, {}

    managed["id"] = managed["id"].astype(int)
    managed = managed.sort_values("id", kind="stable")
    display_id_by_position = {
        position_id: display_id for display_id, position_id in enumerate(managed["id"].tolist(), start=1)
    }
    position_id_by_buy_client_order_id = {
        str(row["buy_client_order_id"]): int(row["id"])
        for row in managed.to_dict("records")
        if not pd.isna(row["buy_client_order_id"])
    }
    display_id_by_buy_client_order_id = {
        client_order_id: display_id_by_position[position_id]
        for client_order_id, position_id in position_id_by_buy_client_order_id.items()
    }
    return display_id_by_position, display_id_by_buy_client_order_id
