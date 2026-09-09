from __future__ import annotations

import ctypes
import errno
import os
import secrets
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TextIO

_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
_SQLITE_STAGING_PREFIX = ".leveraged-trader-sqlite-"
_SQLITE_STAGING_SUFFIX = ".tmp"
_SQLITE_STAGING_TOKEN_LENGTH = 32
_SQLITE_ORPHAN_STAGE_MINIMUM_AGE_SECONDS = 300
_ANONYMOUS_FILE_UNSUPPORTED_ERRORS = frozenset(
    {
        errno.EACCES,
        errno.EINVAL,
        errno.EISDIR,
        errno.ENOENT,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
        errno.EPERM,
    }
)


@dataclass(frozen=True)
class _RuntimeFileIdentity:
    device: int
    inode: int


class _RuntimeFileIdentityMismatch(OSError):
    """An opened runtime inode did not match the caller's snapshot."""


@dataclass(frozen=True)
class PrivateRuntimeDirectory:
    path: str
    identity: _RuntimeFileIdentity
    ancestry: tuple[tuple[str, _RuntimeFileIdentity], ...]


@dataclass(frozen=True)
class SqliteRuntimeGuard:
    """Identity snapshot for one SQLite database and its current sidecars."""

    database_path: str
    database_identity: _RuntimeFileIdentity
    parent: PrivateRuntimeDirectory
    sidecars: tuple[tuple[str, _RuntimeFileIdentity | None], ...]


@dataclass(frozen=True)
class _ActiveSqliteRuntimeFile:
    """Run-scoped database identity that every nested connection must retain."""

    database_path: str
    database_identity: _RuntimeFileIdentity
    guard: SqliteRuntimeGuard
    revalidate: Callable[[], SqliteRuntimeGuard | None]


_ACTIVE_SQLITE_RUNTIME_FILE: ContextVar[_ActiveSqliteRuntimeFile | None] = ContextVar(
    "active_sqlite_runtime_file",
    default=None,
)


def _normalized_runtime_file_path(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path), strict=False))


@contextmanager
def activate_sqlite_runtime_file(
    guard: SqliteRuntimeGuard | None,
    *,
    revalidate: Callable[[], SqliteRuntimeGuard | None],
) -> Iterator[None]:
    """Propagate one workflow-pinned SQLite identity into nested connections."""
    active = (
        _ActiveSqliteRuntimeFile(
            database_path=_normalized_runtime_file_path(guard.database_path),
            database_identity=guard.database_identity,
            guard=guard,
            revalidate=revalidate,
        )
        if guard is not None
        else None
    )
    token = _ACTIVE_SQLITE_RUNTIME_FILE.set(active)
    try:
        yield
    finally:
        _ACTIVE_SQLITE_RUNTIME_FILE.reset(token)


def revalidate_active_sqlite_runtime_file(
    path: str | os.PathLike[str] | None = None,
    guard: SqliteRuntimeGuard | None = None,
) -> SqliteRuntimeGuard | None:
    """Recheck and return the run-pinned snapshot for nested preparation."""
    active = _ACTIVE_SQLITE_RUNTIME_FILE.get()
    if active is None:
        return None
    if path is not None and _normalized_runtime_file_path(path) != active.database_path:
        raise OSError(f"SQLite database {path} is outside the database identity pinned by the active workflow.")
    refreshed_guard = active.revalidate() or active.guard
    if (
        _normalized_runtime_file_path(refreshed_guard.database_path) != active.database_path
        or refreshed_guard.database_identity != active.database_identity
    ):
        raise OSError("SQLite database changed identity after the active workflow pinned it.")
    if guard is not None and guard.database_identity != active.database_identity:
        raise OSError(f"SQLite database {path} changed identity after the active workflow pinned it.")
    return refreshed_guard


def require_private_file(path: str | os.PathLike[str], *, label: str) -> None:
    """Reject credential-bearing files that are accessible by group or other users."""
    if os.name != "posix":
        return

    file_descriptor = _open_existing_regular_file(path)
    try:
        _require_private_file_descriptor(file_descriptor, path=path, label=label)
    finally:
        os.close(file_descriptor)


@contextmanager
def open_private_text_file(
    path: str | os.PathLike[str],
    *,
    label: str,
    encoding: str = "utf-8",
) -> Iterator[TextIO]:
    """Open one verified private regular file and keep that descriptor for reading."""
    if os.name != "posix":
        with open(path, encoding=encoding) as file:
            yield file
        return

    try:
        file_descriptor = _open_existing_regular_file(path)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise PermissionError(f"{label} {path} is unsafe: {exc}") from exc

    try:
        _require_private_file_descriptor(file_descriptor, path=path, label=label)
        with os.fdopen(file_descriptor, encoding=encoding) as file:
            file_descriptor = -1
            yield file
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)


def _require_private_file_descriptor(
    file_descriptor: int,
    *,
    path: str | os.PathLike[str],
    label: str,
) -> None:
    mode = stat.S_IMODE(os.fstat(file_descriptor).st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"{label} {path} must not be accessible by group or other users; run 'chmod 600 {path}'.")


def _open_existing_regular_file(path: str | os.PathLike[str]) -> int:
    """Open an existing regular file without following a substituted symlink."""
    path_value = os.fspath(path)
    path_before_open = os.lstat(path_value)
    if stat.S_ISLNK(path_before_open.st_mode):
        raise OSError(f"Runtime path {path_value} must not be a symbolic link.")
    if not stat.S_ISREG(path_before_open.st_mode):
        raise OSError(f"Runtime path {path_value} must be a regular file.")
    if path_before_open.st_nlink == 0:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_value)
    if path_before_open.st_nlink > 1:
        raise OSError(f"Runtime path {path_value} must not have multiple hard links.")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    file_descriptor = os.open(path_value, flags)
    try:
        opened_file = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_file.st_mode):
            raise OSError(f"Runtime path {path_value} must be a regular file.")
        if opened_file.st_uid != os.geteuid():
            raise OSError(f"Runtime path {path_value} must be owned by the current user.")
        if opened_file.st_nlink == 0:
            # SQLite may unlink a WAL/SHM sidecar after the pathname snapshot
            # but before this descriptor is inspected. Treat that open-but-
            # unlinked inode exactly like the adjacent FileNotFoundError race.
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_value)
        if opened_file.st_nlink > 1:
            raise OSError(f"Runtime path {path_value} must not have multiple hard links.")
        if (path_before_open.st_dev, path_before_open.st_ino) != (opened_file.st_dev, opened_file.st_ino):
            raise OSError(f"Runtime path {path_value} changed while it was being secured.")
    except BaseException:
        os.close(file_descriptor)
        raise
    return file_descriptor


def _runtime_file_identity(file_status: os.stat_result) -> _RuntimeFileIdentity:
    return _RuntimeFileIdentity(file_status.st_dev, file_status.st_ino)


def _restrict_existing_runtime_file(
    path: str | os.PathLike[str],
    *,
    expected_identity: _RuntimeFileIdentity | None = None,
) -> _RuntimeFileIdentity:
    file_descriptor = _open_existing_regular_file(path)
    try:
        opened_identity = _runtime_file_identity(os.fstat(file_descriptor))
        if expected_identity is not None and opened_identity != expected_identity:
            raise _RuntimeFileIdentityMismatch(f"Runtime path {path} changed identity before it could be secured.")
        os.fchmod(file_descriptor, 0o600)
        return _runtime_file_identity(os.fstat(file_descriptor))
    finally:
        os.close(file_descriptor)


def _snapshot_and_restrict_sqlite_sidecar(
    path: str | os.PathLike[str],
) -> _RuntimeFileIdentity | None:
    """Harden the current sidecar inode without freezing its SQLite-managed lifecycle."""
    try:
        observed = os.lstat(path)
    except FileNotFoundError:
        return None

    try:
        return _restrict_existing_runtime_file(
            path,
            expected_identity=_runtime_file_identity(observed),
        )
    except FileNotFoundError:
        # WAL, SHM, and journal files can legitimately disappear when another
        # connection completes between the snapshot and descriptor open.
        return None
    except _RuntimeFileIdentityMismatch as exc:
        raise OSError(
            f"SQLite sidecar {path} changed while it was being prepared; the rejected path was left untouched."
        ) from exc


def _trusted_runtime_owner_uids() -> set[int]:
    """Return identities trusted to own immutable runtime-path ancestry."""
    return {0, os.geteuid(), os.lstat(os.path.sep).st_uid}


def _require_protected_runtime_directory_entry(
    *,
    child_path: str,
    child_status: os.stat_result,
    parent_path: str,
    parent_status: os.stat_result,
    label: str,
) -> None:
    _require_trusted_runtime_directory_parent(
        child_path=child_path,
        child_owner_uid=child_status.st_uid,
        parent_path=parent_path,
        parent_status=parent_status,
        label=label,
    )


def _require_trusted_runtime_directory_parent(
    *,
    child_path: str,
    child_owner_uid: int,
    parent_path: str,
    parent_status: os.stat_result,
    label: str,
) -> None:
    trusted_owners = _trusted_runtime_owner_uids()
    if parent_status.st_uid not in trusted_owners:
        # The owner can restore write permission even when the directory is
        # currently mode 0555, so its entries are not a stable trust anchor.
        raise PermissionError(
            f"{label} entry {child_path} is unsafe because parent {parent_path} is controlled by an untrusted owner."
        )

    parent_mode = stat.S_IMODE(parent_status.st_mode)
    if not parent_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return

    sticky_directory = bool(parent_status.st_mode & stat.S_ISVTX)
    sticky_owners_are_trusted = child_owner_uid in trusted_owners
    if not sticky_directory or not sticky_owners_are_trusted:
        raise PermissionError(
            f"{label} entry {child_path} is unsafe because parent {parent_path} permits directory-entry substitution."
        )


def _absolute_runtime_path_preserving_traversal(path: str | os.PathLike[str]) -> str:
    """Make a path absolute without collapsing components before symlink lookup."""
    path_value = os.fspath(path)
    return path_value if os.path.isabs(path_value) else os.path.join(os.getcwd(), path_value)


def _validate_runtime_directory_lexical_ancestry(
    path: str,
    *,
    label: str,
    allow_missing: bool = False,
) -> None:
    """Validate supplied entries in filesystem traversal order, including symlinks."""
    absolute_path = _absolute_runtime_path_preserving_traversal(path)
    path_components = absolute_path.split(os.path.sep)

    def reject_unresolved_parent_traversal(
        missing_path: str,
        remaining_components: list[str],
    ) -> None:
        if os.pardir in remaining_components:
            raise FileNotFoundError(
                errno.ENOENT,
                f"{label} cannot traverse '..' after missing component {missing_path}",
                missing_path,
            )

    child_path = os.path.sep
    for component_index, path_component in enumerate(path_components):
        if path_component in {"", os.curdir}:
            continue
        if path_component == os.pardir:
            try:
                current_status = os.stat(child_path, follow_symlinks=True)
            except FileNotFoundError:
                if os.path.lexists(child_path):
                    raise PermissionError(f"{label} ancestor {child_path} must resolve to a directory.") from None
                if allow_missing:
                    reject_unresolved_parent_traversal(
                        child_path,
                        path_components[component_index:],
                    )
                    return
                raise
            if not stat.S_ISDIR(current_status.st_mode):
                raise PermissionError(f"{label} ancestor {child_path} must resolve to a directory.")
            child_path = os.path.join(child_path, path_component)
            continue

        parent_path = child_path
        child_path = os.path.join(parent_path, path_component)
        try:
            parent_status = os.stat(parent_path, follow_symlinks=True)
        except FileNotFoundError:
            if os.path.lexists(parent_path):
                raise PermissionError(f"{label} ancestor {parent_path} must resolve to a directory.") from None
            if allow_missing:
                reject_unresolved_parent_traversal(
                    parent_path,
                    path_components[component_index:],
                )
                return
            raise
        if not stat.S_ISDIR(parent_status.st_mode):
            raise PermissionError(f"{label} ancestor {parent_path} must resolve to a directory.")
        try:
            child_status = os.lstat(child_path)
        except FileNotFoundError:
            if allow_missing:
                reject_unresolved_parent_traversal(
                    child_path,
                    path_components[component_index + 1 :],
                )
                return
            raise
        _require_protected_runtime_directory_entry(
            child_path=child_path,
            child_status=child_status,
            parent_path=parent_path,
            parent_status=parent_status,
            label=label,
        )


def _snapshot_runtime_directory_ancestry(
    path: str,
    *,
    label: str,
) -> tuple[tuple[str, _RuntimeFileIdentity], ...]:
    """Snapshot a canonical directory chain whose entries cannot be replaced by other users."""
    directory_statuses: list[tuple[str, os.stat_result]] = []
    candidate = path
    while True:
        candidate_status = os.lstat(candidate)
        if stat.S_ISLNK(candidate_status.st_mode) or not stat.S_ISDIR(candidate_status.st_mode):
            raise PermissionError(f"{label} ancestor {candidate} must be a regular directory.")
        directory_statuses.append((candidate, candidate_status))
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent

    for (child_path, child_status), (parent_path, parent_status) in zip(
        directory_statuses,
        directory_statuses[1:],
        strict=False,
    ):
        _require_protected_runtime_directory_entry(
            child_path=child_path,
            child_status=child_status,
            parent_path=parent_path,
            parent_status=parent_status,
            label=label,
        )

    return tuple(
        (directory_path, _runtime_file_identity(directory_status))
        for directory_path, directory_status in directory_statuses
    )


def require_owned_runtime_directory(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> PrivateRuntimeDirectory | None:
    """Require a stable, owned directory that other OS users cannot modify."""
    if os.name != "posix":
        return None

    path_value = _absolute_runtime_path_preserving_traversal(path)
    path_before_open = os.lstat(path_value)
    if stat.S_ISLNK(path_before_open.st_mode) or not stat.S_ISDIR(path_before_open.st_mode):
        raise PermissionError(f"{label} {path_value} must be a regular directory, not a symbolic link.")

    _validate_runtime_directory_lexical_ancestry(path_value, label=label)
    canonical_path = os.path.realpath(path_value, strict=True)

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    directory_descriptor = os.open(canonical_path, flags)
    try:
        opened_directory = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(opened_directory.st_mode):
            raise PermissionError(f"{label} {path_value} must be a directory.")
        if _runtime_file_identity(path_before_open) != _runtime_file_identity(opened_directory):
            raise PermissionError(f"{label} {path_value} changed while it was being verified.")
        if opened_directory.st_uid != os.geteuid():
            raise PermissionError(f"{label} {path_value} must be owned by the current user.")
        if stat.S_IMODE(opened_directory.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
            raise PermissionError(
                f"{label} {path_value} must not be writable by group or other users; run 'chmod go-w {path_value}'."
            )
        ancestry = _snapshot_runtime_directory_ancestry(canonical_path, label=label)
        if ancestry[0][1] != _runtime_file_identity(opened_directory):
            raise PermissionError(f"{label} {path_value} changed while its parent entry was being verified.")
        return PrivateRuntimeDirectory(
            path=canonical_path,
            identity=_runtime_file_identity(opened_directory),
            ancestry=ancestry,
        )
    finally:
        os.close(directory_descriptor)


def preflight_owned_runtime_directory(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> None:
    """Validate a directory or its deepest existing ancestor without creating it."""
    path_value = _absolute_runtime_path_preserving_traversal(path)
    if os.name == "posix":
        _validate_runtime_directory_lexical_ancestry(
            path_value,
            label=label,
            allow_missing=True,
        )
    canonical_path = os.path.realpath(path_value, strict=False)
    if os.name != "posix":
        candidate = canonical_path
        while not os.path.lexists(candidate):
            parent = os.path.dirname(candidate)
            if parent == candidate:
                raise FileNotFoundError(candidate)
            candidate = parent
        if not os.path.isdir(candidate):
            raise NotADirectoryError(f"{label} ancestor {candidate} must resolve to a directory.")
        return

    missing_directories: list[str] = []
    candidate = canonical_path
    while True:
        try:
            os.lstat(candidate)
            break
        except FileNotFoundError:
            missing_directories.append(candidate)
            parent = os.path.dirname(candidate)
            if parent == candidate:
                raise
            candidate = parent

    if not missing_directories:
        require_owned_runtime_directory(path_value, label=label)
        return

    _snapshot_runtime_directory_ancestry(candidate, label=label)
    first_missing_directory = missing_directories[-1]
    parent_path = os.path.dirname(first_missing_directory)
    parent_status = os.stat(parent_path, follow_symlinks=True)
    if not stat.S_ISDIR(parent_status.st_mode):
        raise PermissionError(f"{label} ancestor {parent_path} must resolve to a directory.")
    _require_trusted_runtime_directory_parent(
        child_path=first_missing_directory,
        child_owner_uid=os.geteuid(),
        parent_path=parent_path,
        parent_status=parent_status,
        label=label,
    )


def prepare_owned_runtime_directory(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> PrivateRuntimeDirectory | None:
    """Create missing path components privately, then verify the requested directory."""
    path_value = os.fspath(path)
    if os.name != "posix":
        os.makedirs(path_value, exist_ok=True)
        return None

    absolute_path = _absolute_runtime_path_preserving_traversal(path_value)
    _validate_runtime_directory_lexical_ancestry(
        absolute_path,
        label=label,
        allow_missing=True,
    )
    try:
        os.lstat(absolute_path)
    except FileNotFoundError:
        pass
    else:
        return require_owned_runtime_directory(absolute_path, label=label)

    canonical_path = os.path.realpath(absolute_path, strict=False)
    missing_directories: list[str] = []
    candidate = canonical_path
    while True:
        try:
            os.lstat(candidate)
            break
        except FileNotFoundError:
            missing_directories.append(candidate)
            parent = os.path.dirname(candidate)
            if parent == candidate:
                raise
            candidate = parent

    if missing_directories:
        _snapshot_runtime_directory_ancestry(candidate, label=label)

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    for directory_path in reversed(missing_directories):
        parent_path = os.path.dirname(directory_path)
        parent_status = os.stat(parent_path, follow_symlinks=True)
        if not stat.S_ISDIR(parent_status.st_mode):
            raise PermissionError(f"{label} ancestor {parent_path} must resolve to a directory.")
        _require_trusted_runtime_directory_parent(
            child_path=directory_path,
            child_owner_uid=os.geteuid(),
            parent_path=parent_path,
            parent_status=parent_status,
            label=label,
        )
        try:
            os.mkdir(directory_path, mode=0o700)
        except FileExistsError:
            # A concurrent creator is acceptable only if the final validation
            # proves that it supplied the same kind of safe directory.
            require_owned_runtime_directory(directory_path, label=label)
            continue

        directory_descriptor = os.open(directory_path, flags)
        try:
            created_directory = os.fstat(directory_descriptor)
            if not stat.S_ISDIR(created_directory.st_mode):
                raise PermissionError(f"{label} {directory_path} must be a directory.")
            if created_directory.st_uid != os.geteuid():
                raise PermissionError(f"{label} {directory_path} must be owned by the current user.")
            # mkdir's requested mode protects against a permissive umask;
            # fchmod also restores owner access under an unusually restrictive one.
            os.fchmod(directory_descriptor, 0o700)
        finally:
            os.close(directory_descriptor)
        require_owned_runtime_directory(directory_path, label=label)

    return require_owned_runtime_directory(absolute_path, label=label)


def revalidate_owned_runtime_directory(directory: PrivateRuntimeDirectory | None, *, label: str) -> None:
    if directory is None:
        return
    current = require_owned_runtime_directory(directory.path, label=label)
    if current is None or current.identity != directory.identity or current.ancestry != directory.ancestry:
        raise PermissionError(f"{label} {directory.path} changed while runtime files were in use.")


@contextmanager
def open_owned_runtime_directory(
    directory: PrivateRuntimeDirectory | None,
    *,
    label: str,
) -> Iterator[int | None]:
    """Keep operations bound to one verified directory inode.

    A pathname revalidation can detect a rename only after it happened.  Holding
    this descriptor also lets callers perform ``dir_fd`` operations against the
    directory that was verified, rather than a replacement subsequently
    installed at the same pathname.
    """
    if directory is None:
        yield None
        return

    revalidate_owned_runtime_directory(directory, label=label)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    directory_descriptor = os.open(directory.path, flags)
    operation_failure: BaseException | None = None
    try:
        opened_directory = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(opened_directory.st_mode):
            raise PermissionError(f"{label} {directory.path} must remain a directory.")
        if _runtime_file_identity(opened_directory) != directory.identity:
            raise PermissionError(f"{label} {directory.path} changed while it was being opened.")
        revalidate_owned_runtime_directory(directory, label=label)
        yield directory_descriptor
    except BaseException as exc:
        operation_failure = exc
        raise
    finally:
        try:
            os.close(directory_descriptor)
        except BaseException as close_failure:
            if operation_failure is not None:
                operation_failure.add_note(f"Failed to close the {label.lower()} descriptor: {close_failure}")
            else:
                raise


def prepare_private_runtime_file(
    path: str | os.PathLike[str],
    *,
    expected_guard: SqliteRuntimeGuard | None = None,
) -> SqliteRuntimeGuard | None:
    """Create and snapshot private SQLite state.

    The verified parent excludes pathname substitution by other OS users. Python's
    sqlite3 API cannot bind SQLite's internally opened sidecars to our descriptors,
    so other processes running under the same UID remain inside the trust boundary.
    """
    if os.name != "posix" or os.fspath(path) == ":memory:":
        return None

    path_value = _absolute_runtime_path_preserving_traversal(path)
    if expected_guard is not None and _normalized_runtime_file_path(path_value) != _normalized_runtime_file_path(
        expected_guard.database_path
    ):
        raise OSError(f"SQLite database {path_value} is outside the database identity pinned by the active workflow.")
    supplied_database_name = os.path.basename(path_value)
    if _is_sqlite_staging_name(supplied_database_name):
        raise ValueError(
            f"SQLite database filename is reserved for private publication stages: {supplied_database_name}"
        )
    lexical_parent = prepare_owned_runtime_directory(
        os.path.dirname(path_value) or os.curdir,
        label="SQLite database parent directory",
    )
    assert lexical_parent is not None
    database_name = supplied_database_name
    try:
        supplied_path = os.lstat(path_value)
    except FileNotFoundError:
        target_path = os.path.join(lexical_parent.path, database_name)
    else:
        target_path = (
            os.path.realpath(path_value, strict=True)
            if stat.S_ISLNK(supplied_path.st_mode)
            else os.path.join(lexical_parent.path, database_name)
        )

    target_database_name = os.path.basename(target_path)
    if _is_sqlite_staging_name(target_database_name):
        raise ValueError(
            f"SQLite database target filename is reserved for private publication stages: {target_database_name}"
        )

    if expected_guard is not None and os.path.normcase(os.path.abspath(target_path)) != os.path.normcase(
        os.path.abspath(expected_guard.database_path)
    ):
        raise OSError(
            f"SQLite database {path_value} changed target after the active workflow pinned it; "
            "the rejected path was left untouched."
        )

    parent = prepare_owned_runtime_directory(
        os.path.dirname(target_path) or os.curdir,
        label="SQLite database parent directory",
    )
    assert parent is not None
    if expected_guard is None:
        _recover_interrupted_named_runtime_file_publication(
            target_database_name,
            parent=parent,
        )

    try:
        database_identity = (
            _restrict_existing_runtime_file(
                target_path,
                expected_identity=expected_guard.database_identity,
            )
            if expected_guard is not None
            else _restrict_existing_runtime_file(target_path)
        )
    except FileNotFoundError as exc:
        if expected_guard is not None:
            raise OSError(
                f"SQLite database {target_path} disappeared after the active workflow pinned it; "
                "the rejected path was left untouched."
            ) from exc
        database_identity = None
    except _RuntimeFileIdentityMismatch as exc:
        raise OSError(
            f"SQLite database {target_path} changed identity after the active workflow pinned it; "
            "the rejected path was left untouched."
        ) from exc

    sidecars: list[tuple[str, _RuntimeFileIdentity | None]] = []
    for suffix in _SQLITE_SIDECAR_SUFFIXES:
        sidecar_path = f"{target_path}{suffix}"
        if expected_guard is not None:
            sidecar_identity = _snapshot_and_restrict_sqlite_sidecar(sidecar_path)
        else:
            try:
                sidecar_identity = _restrict_existing_runtime_file(sidecar_path)
            except FileNotFoundError:
                sidecar_identity = None
        sidecars.append((sidecar_path, sidecar_identity))

    revalidate_owned_runtime_directory(parent, label="SQLite database parent directory")
    if database_identity is None:
        try:
            database_identity = _publish_new_private_runtime_file(
                target_database_name,
                parent=parent,
            )
        except FileExistsError:
            # A cooperating concurrent initializer may have won publication.
            # Treat it exactly like a database that existed at entry.
            database_identity = _restrict_existing_runtime_file(target_path)

    return SqliteRuntimeGuard(
        database_path=target_path,
        database_identity=database_identity,
        parent=parent,
        sidecars=tuple(sidecars),
    )


def _new_sqlite_staging_name() -> str:
    return f"{_SQLITE_STAGING_PREFIX}{secrets.token_hex(16)}{_SQLITE_STAGING_SUFFIX}"


def _is_sqlite_staging_name(filename: str) -> bool:
    if not filename.startswith(_SQLITE_STAGING_PREFIX) or not filename.endswith(_SQLITE_STAGING_SUFFIX):
        return False
    token = filename[len(_SQLITE_STAGING_PREFIX) : -len(_SQLITE_STAGING_SUFFIX)]
    return len(token) == _SQLITE_STAGING_TOKEN_LENGTH and all(character in "0123456789abcdef" for character in token)


def _runtime_directory_entry_status(filename: str, *, directory_descriptor: int) -> os.stat_result:
    return os.stat(
        filename,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _open_runtime_recovery_entry(filename: str, *, directory_descriptor: int) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(filename, flags, dir_fd=directory_descriptor)


def _restore_quarantined_runtime_entry(
    quarantine_name: str,
    original_name: str,
    *,
    directory_descriptor: int,
) -> bool:
    """Restore a raced entry without overwriting a newer pathname occupant."""
    try:
        os.link(
            quarantine_name,
            original_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError:
        return False
    try:
        os.unlink(quarantine_name, dir_fd=directory_descriptor)
    except OSError:
        return False
    return True


def _retire_matching_sqlite_stage(
    staging_name: str,
    expected: os.stat_result,
    *,
    directory_descriptor: int,
) -> None:
    """Move a stage aside, verify its inode, and only then remove that inode."""
    while True:
        quarantine_name = _new_sqlite_staging_name()
        try:
            _runtime_directory_entry_status(
                quarantine_name,
                directory_descriptor=directory_descriptor,
            )
        except FileNotFoundError:
            break

    os.rename(
        staging_name,
        quarantine_name,
        src_dir_fd=directory_descriptor,
        dst_dir_fd=directory_descriptor,
    )
    quarantined = _runtime_directory_entry_status(
        quarantine_name,
        directory_descriptor=directory_descriptor,
    )
    if not stat.S_ISREG(quarantined.st_mode) or _runtime_file_identity(quarantined) != _runtime_file_identity(expected):
        restored = _restore_quarantined_runtime_entry(
            quarantine_name,
            staging_name,
            directory_descriptor=directory_descriptor,
        )
        raise OSError(
            "SQLite recovery stage changed before cleanup; "
            f"the replacement was {'restored' if restored else f'preserved as {quarantine_name}'}."
        )
    os.unlink(quarantine_name, dir_fd=directory_descriptor)


def _is_collectable_orphaned_sqlite_stage(
    observed: os.stat_result,
    *,
    oldest_allowed_timestamp: float,
) -> bool:
    """Recognize only an old, empty stage inode created by named publication."""
    return (
        stat.S_ISREG(observed.st_mode)
        and observed.st_uid == os.geteuid()
        and observed.st_nlink == 1
        and stat.S_IMODE(observed.st_mode) == 0o600
        and observed.st_size == 0
        and max(observed.st_mtime, observed.st_ctime) <= oldest_allowed_timestamp
    )


def _retire_orphaned_sqlite_stages(
    canonical_filename: str,
    *,
    directory_descriptor: int,
) -> None:
    """Collect hardened named stages left before their canonical link was made.

    The age threshold prevents one initializer from removing the unpublished
    stage of another initializer in the short create-and-link critical section.
    Opening and then quarantining by directory descriptor makes replacement
    races fail closed without unlinking the replacement inode.
    """
    oldest_allowed_timestamp = time.time() - _SQLITE_ORPHAN_STAGE_MINIMUM_AGE_SECONDS
    for staging_name in os.listdir(directory_descriptor):
        if staging_name == canonical_filename or not _is_sqlite_staging_name(staging_name):
            continue
        try:
            candidate = _runtime_directory_entry_status(
                staging_name,
                directory_descriptor=directory_descriptor,
            )
        except FileNotFoundError:
            continue
        if not _is_collectable_orphaned_sqlite_stage(
            candidate,
            oldest_allowed_timestamp=oldest_allowed_timestamp,
        ):
            continue

        try:
            staging_descriptor = _open_runtime_recovery_entry(
                staging_name,
                directory_descriptor=directory_descriptor,
            )
        except FileNotFoundError:
            continue
        try:
            opened = os.fstat(staging_descriptor)
            if _runtime_file_identity(opened) != _runtime_file_identity(
                candidate
            ) or not _is_collectable_orphaned_sqlite_stage(
                opened,
                oldest_allowed_timestamp=oldest_allowed_timestamp,
            ):
                continue
            _retire_matching_sqlite_stage(
                staging_name,
                opened,
                directory_descriptor=directory_descriptor,
            )
        finally:
            os.close(staging_descriptor)


def _recover_interrupted_named_runtime_file_publication(
    filename: str,
    *,
    parent: PrivateRuntimeDirectory,
) -> None:
    """Recover hardened stages left before or after canonical publication."""
    with open_owned_runtime_directory(
        parent,
        label="SQLite database parent directory",
    ) as directory_descriptor:
        assert directory_descriptor is not None
        _retire_orphaned_sqlite_stages(
            filename,
            directory_descriptor=directory_descriptor,
        )
        try:
            canonical = _runtime_directory_entry_status(
                filename,
                directory_descriptor=directory_descriptor,
            )
        except FileNotFoundError:
            return
        if (
            not stat.S_ISREG(canonical.st_mode)
            or canonical.st_uid != os.geteuid()
            or canonical.st_nlink != 2
            or stat.S_IMODE(canonical.st_mode) & (stat.S_IRWXG | stat.S_IRWXO)
        ):
            return

        matching_stages: list[tuple[str, os.stat_result]] = []
        for entry_name in os.listdir(directory_descriptor):
            if entry_name == filename or not _is_sqlite_staging_name(entry_name):
                continue
            try:
                candidate = _runtime_directory_entry_status(
                    entry_name,
                    directory_descriptor=directory_descriptor,
                )
            except FileNotFoundError:
                continue
            if _runtime_file_identity(candidate) == _runtime_file_identity(canonical):
                matching_stages.append((entry_name, candidate))
        if len(matching_stages) != 1:
            return
        staging_name, staging = matching_stages[0]
        if (
            not stat.S_ISREG(staging.st_mode)
            or staging.st_uid != os.geteuid()
            or staging.st_nlink != 2
            or stat.S_IMODE(staging.st_mode) & (stat.S_IRWXG | stat.S_IRWXO)
        ):
            return

        canonical_descriptor = _open_runtime_recovery_entry(
            filename,
            directory_descriptor=directory_descriptor,
        )
        staging_descriptor = -1
        operation_failure: BaseException | None = None
        try:
            staging_descriptor = _open_runtime_recovery_entry(
                staging_name,
                directory_descriptor=directory_descriptor,
            )
            opened_canonical = os.fstat(canonical_descriptor)
            opened_staging = os.fstat(staging_descriptor)
            expected_identity = _runtime_file_identity(canonical)
            if any(
                not stat.S_ISREG(observed.st_mode)
                or observed.st_uid != os.geteuid()
                or observed.st_nlink != 2
                or _runtime_file_identity(observed) != expected_identity
                for observed in (opened_canonical, opened_staging)
            ):
                return
            os.fchmod(canonical_descriptor, 0o600)
            _retire_matching_sqlite_stage(
                staging_name,
                opened_staging,
                directory_descriptor=directory_descriptor,
            )
            recovered = os.fstat(canonical_descriptor)
            published = _runtime_directory_entry_status(
                filename,
                directory_descriptor=directory_descriptor,
            )
            if (
                recovered.st_nlink != 1
                or published.st_nlink != 1
                or _runtime_file_identity(recovered) != expected_identity
                or _runtime_file_identity(published) != expected_identity
            ):
                raise OSError("SQLite database recovery did not restore one canonical private link.")
        except BaseException as exc:
            operation_failure = exc
            raise
        finally:
            cleanup_failures: list[tuple[str, BaseException]] = []
            if staging_descriptor >= 0:
                try:
                    os.close(staging_descriptor)
                except BaseException as exc:
                    cleanup_failures.append(("close the SQLite recovery staging descriptor", exc))
            try:
                os.close(canonical_descriptor)
            except BaseException as exc:
                cleanup_failures.append(("close the SQLite recovery canonical descriptor", exc))
            if cleanup_failures:
                if operation_failure is not None:
                    for action, cleanup_failure in cleanup_failures:
                        operation_failure.add_note(f"Failed to {action}: {cleanup_failure}")
                else:
                    first_action, first_cleanup_failure = cleanup_failures[0]
                    first_cleanup_failure.add_note(f"Failed to {first_action}.")
                    for action, cleanup_failure in cleanup_failures[1:]:
                        first_cleanup_failure.add_note(f"Also failed to {action}: {cleanup_failure}")
                    raise first_cleanup_failure
    revalidate_owned_runtime_directory(parent, label="SQLite database parent directory")


def _publish_new_private_runtime_file(
    filename: str,
    *,
    parent: PrivateRuntimeDirectory,
) -> _RuntimeFileIdentity:
    """Harden a new inode before atomically publishing its database pathname."""
    with open_owned_runtime_directory(
        parent,
        label="SQLite database parent directory",
    ) as directory_descriptor:
        assert directory_descriptor is not None
        anonymous_descriptor = _open_anonymous_runtime_file(directory_descriptor)
        if anonymous_descriptor is not None:
            try:
                identity = _harden_new_runtime_file(anonymous_descriptor, expected_links=0)
                try:
                    _link_anonymous_runtime_file(
                        anonymous_descriptor,
                        directory_descriptor=directory_descriptor,
                        filename=filename,
                    )
                except OSError as exc:
                    if exc.errno not in _ANONYMOUS_FILE_UNSUPPORTED_ERRORS:
                        raise
                else:
                    return identity
            finally:
                os.close(anonymous_descriptor)

        return _publish_named_staged_runtime_file(
            filename,
            directory_descriptor=directory_descriptor,
        )


def _open_anonymous_runtime_file(directory_descriptor: int) -> int | None:
    """Open an unpublished inode when the host filesystem supports ``O_TMPFILE``."""
    anonymous_flag = getattr(os, "O_TMPFILE", None)
    if anonymous_flag is None:
        return None

    flags = os.O_WRONLY | anonymous_flag
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        return os.open(os.curdir, flags, 0o600, dir_fd=directory_descriptor)
    except OSError as exc:
        if exc.errno not in _ANONYMOUS_FILE_UNSUPPORTED_ERRORS:
            raise
        return None


def _harden_new_runtime_file(
    file_descriptor: int,
    *,
    expected_links: int,
) -> _RuntimeFileIdentity:
    os.fchmod(file_descriptor, 0o600)
    observed = os.fstat(file_descriptor)
    if not stat.S_ISREG(observed.st_mode):
        raise OSError("New SQLite database must be a regular file.")
    if observed.st_uid != os.geteuid():
        raise OSError("New SQLite database must be owned by the current user.")
    if observed.st_nlink != expected_links:
        raise OSError("New SQLite database changed link count before publication.")
    return _runtime_file_identity(observed)


def _link_anonymous_runtime_file(
    file_descriptor: int,
    *,
    directory_descriptor: int,
    filename: str,
) -> None:
    """Publish an ``O_TMPFILE`` inode without introducing a cleanup pathname."""
    # ``AT_EMPTY_PATH`` is Linux-specific, as is ``O_TMPFILE``. Calling libc is
    # required because Python's os.link wrapper does not expose this flag.
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        linkat = libc.linkat
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS), filename) from exc
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    if (
        linkat(
            file_descriptor,
            b"",
            directory_descriptor,
            os.fsencode(filename),
            0x1000,  # AT_EMPTY_PATH
        )
        == 0
    ):
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), filename)


def _publish_named_staged_runtime_file(
    filename: str,
    *,
    directory_descriptor: int,
) -> _RuntimeFileIdentity:
    """Portable fallback that never uses the caller-visible path for rollback."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    while True:
        staging_name = _new_sqlite_staging_name()
        try:
            file_descriptor = os.open(
                staging_name,
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        break

    staging_present = True
    try:
        identity = _harden_new_runtime_file(file_descriptor, expected_links=1)
        os.link(
            staging_name,
            filename,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        os.unlink(staging_name, dir_fd=directory_descriptor)
        staging_present = False
        return identity
    finally:
        try:
            if staging_present:
                with suppress(FileNotFoundError):
                    os.unlink(staging_name, dir_fd=directory_descriptor)
        finally:
            os.close(file_descriptor)


def revalidate_private_runtime_file(guard: SqliteRuntimeGuard | None) -> SqliteRuntimeGuard | None:
    """Fail closed if SQLite state changed identity or gained unsafe links."""
    if guard is None:
        return None

    revalidate_owned_runtime_directory(guard.parent, label="SQLite database parent directory")
    try:
        database_identity = _restrict_existing_runtime_file(
            guard.database_path,
            expected_identity=guard.database_identity,
        )
    except FileNotFoundError as exc:
        raise OSError(f"SQLite database {guard.database_path} disappeared while it was open.") from exc
    except _RuntimeFileIdentityMismatch as exc:
        raise OSError(
            f"SQLite database {guard.database_path} changed identity while it was open; "
            "the rejected path was left untouched."
        ) from exc
    if database_identity != guard.database_identity:
        raise OSError(f"SQLite database {guard.database_path} changed identity while it was open.")

    refreshed_sidecars: list[tuple[str, _RuntimeFileIdentity | None]] = []
    for sidecar_path, expected_identity in guard.sidecars:
        try:
            observed = os.lstat(sidecar_path)
        except FileNotFoundError:
            refreshed_sidecars.append((sidecar_path, None))
            continue

        observed_identity = _runtime_file_identity(observed)
        unsafe = (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or observed.st_nlink != 1
            or (expected_identity is not None and observed_identity != expected_identity)
        )
        if unsafe:
            raise OSError(
                f"SQLite sidecar {sidecar_path} changed identity or link count while the database was open; "
                "the rejected path was left untouched."
            )

        try:
            refreshed_identity = _restrict_existing_runtime_file(
                sidecar_path,
                expected_identity=observed_identity,
            )
        except FileNotFoundError:
            refreshed_sidecars.append((sidecar_path, None))
            continue
        except OSError as exc:
            raise OSError(
                f"SQLite sidecar {sidecar_path} changed while it was being revalidated; "
                "the rejected path was left untouched."
            ) from exc
        if refreshed_identity != observed_identity:
            raise OSError(f"SQLite sidecar {sidecar_path} changed while it was being revalidated.")
        refreshed_sidecars.append((sidecar_path, refreshed_identity))

    revalidate_owned_runtime_directory(guard.parent, label="SQLite database parent directory")
    return SqliteRuntimeGuard(
        database_path=guard.database_path,
        database_identity=database_identity,
        parent=guard.parent,
        sidecars=tuple(refreshed_sidecars),
    )
