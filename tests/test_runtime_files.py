from __future__ import annotations

import errno
import os
import sqlite3
import stat
import tempfile
import unittest
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from leveraged_trader.runtime_files import (
    _require_protected_runtime_directory_entry,
    _trusted_runtime_owner_uids,
    activate_sqlite_runtime_file,
    open_owned_runtime_directory,
    open_private_text_file,
    prepare_owned_runtime_directory,
    prepare_private_runtime_file,
    require_owned_runtime_directory,
    require_private_file,
    revalidate_active_sqlite_runtime_file,
    revalidate_private_runtime_file,
)
from leveraged_trader.storage import init_state_db, save_table_to_sqlite, save_workflow_assets
from leveraged_trader.workflow import (
    _initialize_state_db,
    _load_or_refresh_workflow_assets_for_db,
    _state_connection,
    _WorkflowStrategySession,
)


@unittest.skipUnless(os.name == "posix", "POSIX file modes are required")
class RuntimeFilePermissionTests(unittest.TestCase):
    def test_owned_directory_preserves_body_failure_when_descriptor_close_fails(self) -> None:
        from leveraged_trader import runtime_files as runtime_files_module

        with tempfile.TemporaryDirectory() as tmp:
            runtime_directory = Path(tmp) / "runtime"
            runtime_directory.mkdir(mode=0o700)
            directory_guard = require_owned_runtime_directory(
                runtime_directory,
                label="Runtime directory",
            )
            real_close = os.close
            real_fstat = os.fstat
            fail_close = False

            def close_then_fail(file_descriptor: int) -> None:
                should_fail = fail_close and stat.S_ISDIR(real_fstat(file_descriptor).st_mode)
                real_close(file_descriptor)
                if should_fail:
                    raise OSError("directory descriptor close failed")

            with (
                patch.object(runtime_files_module.os, "close", side_effect=close_then_fail),
                self.assertRaisesRegex(RuntimeError, "operation failed") as raised,
                open_owned_runtime_directory(
                    directory_guard,
                    label="Runtime directory",
                ),
            ):
                fail_close = True
                raise RuntimeError("operation failed")

            self.assertEqual(len(raised.exception.__notes__), 1)
            self.assertIn("directory descriptor close failed", raised.exception.__notes__[0])

    def test_pathless_active_database_revalidation_invokes_pinned_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            guard = prepare_private_runtime_file(path)
            callbacks: list[str] = []

            with activate_sqlite_runtime_file(
                guard,
                revalidate=lambda: callbacks.append("revalidated"),
            ):
                revalidate_active_sqlite_runtime_file()

            self.assertEqual(callbacks, ["revalidated"])

    def test_filesystem_root_owner_is_a_trusted_system_identity(self) -> None:
        self.assertIn(os.lstat(os.path.sep).st_uid, _trusted_runtime_owner_uids())

    def test_directory_entry_rejects_untrusted_parent_owner_even_without_write_mode(self) -> None:
        trusted_owners = _trusted_runtime_owner_uids()
        untrusted_owner = max(trusted_owners) + 1
        parent_status = SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o555,
            st_uid=untrusted_owner,
        )
        child_status = SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o700,
            st_uid=os.geteuid(),
        )

        with self.assertRaisesRegex(PermissionError, "controlled by an untrusted owner"):
            _require_protected_runtime_directory_entry(
                child_path="/untrusted/runtime",
                child_status=child_status,
                parent_path="/untrusted",
                parent_status=parent_status,
                label="Runtime directory",
            )

    def test_owned_directory_accepts_sticky_shared_parent_but_rejects_nonsticky_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shared_parent = Path(tmp) / "shared"
            shared_parent.mkdir(mode=0o1777)
            shared_parent.chmod(0o1777)
            runtime_directory = shared_parent / "runtime"
            runtime_directory.mkdir(mode=0o700)

            guard = require_owned_runtime_directory(runtime_directory, label="Runtime directory")
            self.assertIsNotNone(guard)

            shared_parent.chmod(0o777)
            with self.assertRaisesRegex(PermissionError, "directory-entry substitution"):
                require_owned_runtime_directory(runtime_directory, label="Runtime directory")

    def test_owned_directory_rejects_intermediate_symlink_in_substitutable_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            safe_parent = Path(tmp) / "safe"
            runtime_directory = safe_parent / "runtime"
            runtime_directory.mkdir(parents=True, mode=0o700)
            shared_parent = Path(tmp) / "shared"
            shared_parent.mkdir(mode=0o777)
            shared_parent.chmod(0o777)
            (shared_parent / "redirect").symlink_to(safe_parent, target_is_directory=True)

            with self.assertRaisesRegex(PermissionError, "directory-entry substitution"):
                require_owned_runtime_directory(
                    shared_parent / "redirect" / "runtime",
                    label="Runtime directory",
                )

    def test_directory_preparation_rejects_unsafe_symlink_prefix_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            safe_parent = Path(tmp) / "safe"
            safe_parent.mkdir(mode=0o700)
            shared_parent = Path(tmp) / "shared"
            shared_parent.mkdir(mode=0o777)
            shared_parent.chmod(0o777)
            (shared_parent / "redirect").symlink_to(safe_parent, target_is_directory=True)

            with self.assertRaisesRegex(PermissionError, "directory-entry substitution"):
                prepare_owned_runtime_directory(
                    shared_parent / "redirect" / "new" / "runtime",
                    label="Runtime directory",
                )

            self.assertFalse((safe_parent / "new").exists())

    def test_directory_preparation_rejects_final_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.mkdir(mode=0o750)
            supplied = Path(tmp) / "runtime"
            supplied.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(PermissionError, "not a symbolic link"):
                prepare_owned_runtime_directory(
                    supplied,
                    label="Runtime directory",
                )

            self.assertTrue(supplied.is_symlink())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o750)

    def test_directory_preparation_rejects_dangling_intermediate_symlink_without_creating_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_target = Path(tmp) / "missing-target"
            redirect = Path(tmp) / "redirect"
            redirect.symlink_to(missing_target, target_is_directory=True)

            with self.assertRaisesRegex(PermissionError, "must resolve to a directory"):
                prepare_owned_runtime_directory(
                    redirect / "runtime",
                    label="Runtime directory",
                )

            self.assertTrue(redirect.is_symlink())
            self.assertFalse(missing_target.exists())

    def test_directory_preparation_rejects_parent_traversal_after_missing_component_without_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supplied = os.path.join(
                os.fspath(root),
                "missing",
                os.pardir,
                "reports",
            )

            with self.assertRaisesRegex(FileNotFoundError, "cannot traverse '\\.\\.'"):
                prepare_owned_runtime_directory(
                    supplied,
                    label="Report output directory",
                )

            self.assertEqual(list(root.iterdir()), [])

    def test_database_preparation_rejects_unsafe_lexical_parent_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            safe_parent = Path(tmp) / "safe"
            safe_parent.mkdir(mode=0o700)
            shared_parent = Path(tmp) / "shared"
            shared_parent.mkdir(mode=0o777)
            shared_parent.chmod(0o777)
            (shared_parent / "redirect").symlink_to(safe_parent, target_is_directory=True)

            with self.assertRaisesRegex(PermissionError, "directory-entry substitution"):
                prepare_private_runtime_file(shared_parent / "redirect" / "new" / "state.sqlite")

            self.assertFalse((safe_parent / "new").exists())

    def test_prepare_private_runtime_file_creates_owner_only_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"

            prepare_private_runtime_file(path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_prepare_private_runtime_file_creates_private_missing_parents(self) -> None:
        for umask_value in (0, 0o002):
            with self.subTest(umask=oct(umask_value)), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp) / "new-parent" / "nested"
                path = parent / "state.sqlite"
                old_umask = os.umask(umask_value)
                try:
                    prepare_private_runtime_file(path)
                finally:
                    os.umask(old_umask)

                self.assertEqual(stat.S_IMODE((Path(tmp) / "new-parent").stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_runtime_preparation_resolves_symlink_before_parent_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic_parent = root / "semantic-parent"
            symlink_target = semantic_parent / "inner"
            symlink_target.mkdir(parents=True, mode=0o700)
            redirect = root / "redirect"
            redirect.symlink_to(symlink_target, target_is_directory=True)
            supplied_database = os.path.join(
                os.fspath(redirect),
                os.pardir,
                "state.sqlite",
            )
            supplied_output = os.path.join(
                os.fspath(redirect),
                os.pardir,
                "reports",
            )

            output_guard = prepare_owned_runtime_directory(
                supplied_output,
                label="Report output directory",
            )
            database_guard = prepare_private_runtime_file(supplied_database)

            assert output_guard is not None
            assert database_guard is not None
            self.assertEqual(output_guard.path, os.fspath(semantic_parent / "reports"))
            self.assertEqual(
                database_guard.database_path,
                os.fspath(semantic_parent / "state.sqlite"),
            )
            self.assertTrue((semantic_parent / "reports").is_dir())
            self.assertTrue((semantic_parent / "state.sqlite").is_file())
            self.assertFalse((root / "reports").exists())
            self.assertFalse((root / "state.sqlite").exists())

            callbacks: list[str] = []
            with activate_sqlite_runtime_file(
                database_guard,
                revalidate=lambda: callbacks.append("revalidated"),
            ):
                revalidate_active_sqlite_runtime_file(supplied_database)
            self.assertEqual(callbacks, ["revalidated"])

    def test_directory_preparation_checks_unsafe_symlink_before_parent_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            semantic_parent = root / "semantic-parent"
            (semantic_parent / "inner").mkdir(parents=True, mode=0o700)
            shared_parent = root / "shared"
            shared_parent.mkdir(mode=0o777)
            shared_parent.chmod(0o777)
            redirect = shared_parent / "redirect"
            redirect.symlink_to(semantic_parent / "inner", target_is_directory=True)
            supplied_output = os.path.join(
                os.fspath(redirect),
                os.pardir,
                "reports",
            )

            with self.assertRaisesRegex(PermissionError, "directory-entry substitution"):
                prepare_owned_runtime_directory(
                    supplied_output,
                    label="Report output directory",
                )

            self.assertFalse((semantic_parent / "reports").exists())

    def test_new_sqlite_database_initializes_and_reopens_under_restrictive_umask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            old_umask = os.umask(0o777)
            try:
                _initialize_state_db(str(path))
                with closing(sqlite3.connect(path)) as conn, conn:
                    table_count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
            finally:
                os.umask(old_umask)

            self.assertGreater(table_count, 0)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_new_database_hardening_failures_leave_no_file_and_allow_retry(self) -> None:
        from leveraged_trader import runtime_files

        real_fchmod = runtime_files.os.fchmod
        real_fstat = runtime_files.os.fstat

        for operation in ("fchmod", "fstat"):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.sqlite"

                def failing_fchmod(file_descriptor: int, mode: int) -> None:
                    if stat.S_ISREG(real_fstat(file_descriptor).st_mode):
                        raise OSError("simulated database chmod failure")
                    real_fchmod(file_descriptor, mode)

                def failing_fstat(file_descriptor: int) -> os.stat_result:
                    observed = real_fstat(file_descriptor)
                    if stat.S_ISREG(observed.st_mode):
                        raise OSError("simulated database stat failure")
                    return observed

                failure = failing_fchmod if operation == "fchmod" else failing_fstat
                patch_target = f"leveraged_trader.runtime_files.os.{operation}"
                old_umask = os.umask(0o777)
                try:
                    with (
                        patch(patch_target, side_effect=failure),
                        self.assertRaisesRegex(OSError, "simulated database (chmod|stat) failure"),
                    ):
                        prepare_private_runtime_file(path)

                    self.assertFalse(path.exists())
                    self.assertEqual(list(Path(tmp).iterdir()), [])

                    prepare_private_runtime_file(path)
                finally:
                    os.umask(old_umask)

                self.assertTrue(path.is_file())
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_open_existing_regular_file_closes_descriptor_after_base_exception(self) -> None:
        from leveraged_trader import runtime_files

        class SimulatedBaseException(BaseException):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime-file"
            path.write_bytes(b"contents")
            closed_descriptors: list[int] = []
            inspected_descriptors: list[int] = []
            real_close = runtime_files.os.close

            def interrupting_fstat(file_descriptor: int) -> os.stat_result:
                inspected_descriptors.append(file_descriptor)
                raise SimulatedBaseException

            def recording_close(file_descriptor: int) -> None:
                closed_descriptors.append(file_descriptor)
                real_close(file_descriptor)

            with (
                patch("leveraged_trader.runtime_files.os.fstat", side_effect=interrupting_fstat),
                patch("leveraged_trader.runtime_files.os.close", side_effect=recording_close),
                self.assertRaises(SimulatedBaseException),
            ):
                runtime_files._open_existing_regular_file(path)

            self.assertEqual(closed_descriptors, inspected_descriptors)
            self.assertEqual(len(closed_descriptors), 1)

    def test_named_database_publication_recovers_interrupted_stage_unlink(self) -> None:
        from leveraged_trader import runtime_files

        real_unlink = runtime_files.os.unlink
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"

            def fail_staging_unlink(
                runtime_path: str | os.PathLike[str],
                *,
                dir_fd: int | None = None,
            ) -> None:
                if os.fspath(runtime_path).startswith(".leveraged-trader-sqlite-"):
                    raise OSError(errno.EIO, "simulated interruption before staging unlink")
                real_unlink(runtime_path, dir_fd=dir_fd)

            with (
                patch(
                    "leveraged_trader.runtime_files._open_anonymous_runtime_file",
                    return_value=None,
                ),
                patch(
                    "leveraged_trader.runtime_files.os.unlink",
                    side_effect=fail_staging_unlink,
                ),
                self.assertRaisesRegex(OSError, "simulated interruption"),
            ):
                prepare_private_runtime_file(path)

            staging_paths = list(Path(tmp).glob(".leveraged-trader-sqlite-*.tmp"))
            self.assertEqual(len(staging_paths), 1)
            self.assertEqual(path.stat().st_nlink, 2)
            self.assertTrue(path.samefile(staging_paths[0]))

            guard = prepare_private_runtime_file(path)

            self.assertIsNotNone(guard)
            self.assertEqual(path.stat().st_nlink, 1)
            self.assertEqual(list(Path(tmp).glob(".leveraged-trader-sqlite-*.tmp")), [])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_named_database_recovery_attempts_both_descriptor_closes(self) -> None:
        from leveraged_trader import runtime_files

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            path.write_bytes(b"database")
            path.chmod(0o600)
            staging_path = Path(tmp) / ".leveraged-trader-sqlite-0123456789abcdef0123456789abcdef.tmp"
            os.link(path, staging_path)
            real_open_recovery_entry = runtime_files._open_runtime_recovery_entry
            real_close = runtime_files.os.close
            recovery_descriptors: dict[str, int] = {}
            recovery_close_attempts: list[str] = []

            def record_recovery_descriptor(filename: str, *, directory_descriptor: int) -> int:
                descriptor = real_open_recovery_entry(
                    filename,
                    directory_descriptor=directory_descriptor,
                )
                recovery_descriptors[filename] = descriptor
                return descriptor

            def close_then_fail(file_descriptor: int) -> None:
                real_close(file_descriptor)
                if file_descriptor == recovery_descriptors.get(staging_path.name):
                    recovery_close_attempts.append(staging_path.name)
                    raise OSError("staging descriptor close failed")
                if file_descriptor == recovery_descriptors.get(path.name):
                    recovery_close_attempts.append(path.name)
                    raise OSError("canonical descriptor close failed")

            with (
                patch.object(
                    runtime_files,
                    "_open_runtime_recovery_entry",
                    side_effect=record_recovery_descriptor,
                ),
                patch.object(runtime_files.os, "close", side_effect=close_then_fail),
                self.assertRaisesRegex(OSError, "staging descriptor close failed") as raised,
            ):
                prepare_private_runtime_file(path)

            self.assertEqual(
                recovery_close_attempts,
                [staging_path.name, path.name],
            )
            self.assertEqual(len(raised.exception.__notes__), 2)
            self.assertIn("SQLite recovery staging descriptor", raised.exception.__notes__[0])
            self.assertIn("canonical descriptor close failed", raised.exception.__notes__[1])

    def test_named_database_recovery_preserves_operation_failure_when_both_closes_fail(self) -> None:
        from leveraged_trader import runtime_files

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            path.write_bytes(b"database")
            path.chmod(0o600)
            staging_path = Path(tmp) / ".leveraged-trader-sqlite-fedcba9876543210fedcba9876543210.tmp"
            os.link(path, staging_path)
            real_open_recovery_entry = runtime_files._open_runtime_recovery_entry
            real_close = runtime_files.os.close
            recovery_descriptors: dict[str, int] = {}
            recovery_close_attempts: list[str] = []

            def record_recovery_descriptor(filename: str, *, directory_descriptor: int) -> int:
                descriptor = real_open_recovery_entry(
                    filename,
                    directory_descriptor=directory_descriptor,
                )
                recovery_descriptors[filename] = descriptor
                return descriptor

            def close_then_fail(file_descriptor: int) -> None:
                real_close(file_descriptor)
                if file_descriptor == recovery_descriptors.get(staging_path.name):
                    recovery_close_attempts.append(staging_path.name)
                    raise OSError("staging descriptor close failed")
                if file_descriptor == recovery_descriptors.get(path.name):
                    recovery_close_attempts.append(path.name)
                    raise OSError("canonical descriptor close failed")

            with (
                patch.object(
                    runtime_files,
                    "_open_runtime_recovery_entry",
                    side_effect=record_recovery_descriptor,
                ),
                patch.object(
                    runtime_files,
                    "_retire_matching_sqlite_stage",
                    side_effect=RuntimeError("recovery operation failed"),
                ),
                patch.object(runtime_files.os, "close", side_effect=close_then_fail),
                self.assertRaisesRegex(RuntimeError, "recovery operation failed") as raised,
            ):
                prepare_private_runtime_file(path)

            self.assertEqual(
                recovery_close_attempts,
                [staging_path.name, path.name],
            )
            self.assertEqual(len(raised.exception.__notes__), 2)
            self.assertIn("staging descriptor close failed", raised.exception.__notes__[0])
            self.assertIn("canonical descriptor close failed", raised.exception.__notes__[1])

    def test_named_database_recovery_collects_old_private_orphan_stage(self) -> None:
        from leveraged_trader import runtime_files

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            staging_path = Path(tmp) / ".leveraged-trader-sqlite-0123456789abcdef0123456789abcdef.tmp"
            staging_path.touch(mode=0o600)
            staging_path.chmod(0o600)

            with patch(
                "leveraged_trader.runtime_files.time.time",
                return_value=(runtime_files.time.time() + runtime_files._SQLITE_ORPHAN_STAGE_MINIMUM_AGE_SECONDS + 1),
            ):
                guard = prepare_private_runtime_file(path)

            self.assertIsNotNone(guard)
            self.assertTrue(path.is_file())
            self.assertFalse(staging_path.exists())

    def test_named_database_recovery_preserves_unrelated_stage_shaped_entries(self) -> None:
        from leveraged_trader import runtime_files

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            prefix = ".leveraged-trader-sqlite-"
            suffix = ".tmp"
            nonempty = Path(tmp) / f"{prefix}11111111111111111111111111111111{suffix}"
            public = Path(tmp) / f"{prefix}22222222222222222222222222222222{suffix}"
            hardlinked = Path(tmp) / f"{prefix}33333333333333333333333333333333{suffix}"
            symlink = Path(tmp) / f"{prefix}44444444444444444444444444444444{suffix}"
            unrelated = Path(tmp) / "unrelated"
            nonempty.write_bytes(b"not an unpublished empty stage")
            nonempty.chmod(0o600)
            public.touch(mode=0o644)
            public.chmod(0o644)
            unrelated.touch(mode=0o600)
            os.link(unrelated, hardlinked)
            symlink.symlink_to(unrelated)

            with patch(
                "leveraged_trader.runtime_files.time.time",
                return_value=(runtime_files.time.time() + runtime_files._SQLITE_ORPHAN_STAGE_MINIMUM_AGE_SECONDS + 1),
            ):
                prepare_private_runtime_file(path)

            self.assertEqual(nonempty.read_bytes(), b"not an unpublished empty stage")
            self.assertEqual(stat.S_IMODE(public.stat().st_mode), 0o644)
            self.assertTrue(hardlinked.samefile(unrelated))
            self.assertTrue(symlink.is_symlink())

    def test_named_database_recovery_defers_fresh_orphan_until_later_run(self) -> None:
        from leveraged_trader import runtime_files

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            staging_path = Path(tmp) / ".leveraged-trader-sqlite-55555555555555555555555555555555.tmp"
            staging_path.touch(mode=0o600)
            staging_path.chmod(0o600)

            prepare_private_runtime_file(path)
            self.assertTrue(staging_path.exists())

            with patch(
                "leveraged_trader.runtime_files.time.time",
                return_value=(runtime_files.time.time() + runtime_files._SQLITE_ORPHAN_STAGE_MINIMUM_AGE_SECONDS + 1),
            ):
                prepare_private_runtime_file(path)

            self.assertFalse(staging_path.exists())

    def test_database_filename_cannot_claim_private_stage_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".leveraged-trader-sqlite-66666666666666666666666666666666.tmp"
            path.write_bytes(b"configured database")
            path.chmod(0o600)

            with self.assertRaisesRegex(ValueError, "reserved for private publication stages"):
                prepare_private_runtime_file(path)

            self.assertEqual(path.read_bytes(), b"configured database")

    def test_database_symlink_cannot_hide_private_stage_target_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reserved_path = Path(tmp) / ".leveraged-trader-sqlite-77777777777777777777777777777777.tmp"
            reserved_path.touch(mode=0o600)
            reserved_path.chmod(0o600)
            alias_path = Path(tmp) / "state.sqlite"
            alias_path.symlink_to(reserved_path.name)

            with self.assertRaisesRegex(ValueError, "target filename is reserved"):
                prepare_private_runtime_file(alias_path)

            self.assertTrue(reserved_path.is_file())
            self.assertTrue(alias_path.is_symlink())

    def test_named_database_recovery_leaves_ambiguous_hardlinks_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            path.write_bytes(b"database")
            path.chmod(0o600)
            staging_path = Path(tmp) / ".leveraged-trader-sqlite-0123456789abcdef0123456789abcdef.tmp"
            unrelated_path = Path(tmp) / "unrelated-hardlink"
            os.link(path, staging_path)
            os.link(path, unrelated_path)

            with self.assertRaisesRegex(OSError, "multiple hard links"):
                prepare_private_runtime_file(path)

            self.assertEqual(path.stat().st_nlink, 3)
            self.assertTrue(path.samefile(staging_path))
            self.assertTrue(path.samefile(unrelated_path))

    def test_anonymous_database_publication_falls_back_when_linking_is_blocked(self) -> None:
        for error_number in (errno.EPERM, errno.EACCES):
            with self.subTest(error_number=error_number), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.sqlite"

                def open_unlinked_fixture(directory_descriptor: int) -> int:
                    fixture_name = ".anonymous-runtime-file-fixture"
                    file_descriptor = os.open(
                        fixture_name,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                        dir_fd=directory_descriptor,
                    )
                    os.unlink(fixture_name, dir_fd=directory_descriptor)
                    return file_descriptor

                with (
                    patch(
                        "leveraged_trader.runtime_files._open_anonymous_runtime_file",
                        side_effect=open_unlinked_fixture,
                    ),
                    patch(
                        "leveraged_trader.runtime_files._link_anonymous_runtime_file",
                        side_effect=OSError(error_number, "simulated blocked AT_EMPTY_PATH"),
                    ) as blocked_link,
                ):
                    prepare_private_runtime_file(path)

                blocked_link.assert_called_once()
                self.assertTrue(path.is_file())
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertEqual([entry.name for entry in Path(tmp).iterdir()], ["state.sqlite"])

    def test_prepare_private_runtime_file_rejects_writable_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "shared"
            parent.mkdir(mode=0o770)
            parent.chmod(0o770)
            path = parent / "state.sqlite"

            with self.assertRaisesRegex(PermissionError, "must not be writable by group or other"):
                prepare_private_runtime_file(path)

            self.assertFalse(path.exists())
            self.assertEqual(stat.S_IMODE(parent.stat().st_mode), 0o770)

    def test_prepare_private_runtime_file_restricts_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            path.write_bytes(b"existing")
            path.chmod(0o644)

            prepare_private_runtime_file(path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(path.read_bytes(), b"existing")

    def test_prepare_private_runtime_file_restricts_existing_sqlite_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            sidecars = [Path(f"{path}{suffix}") for suffix in ("-wal", "-shm", "-journal")]
            path.write_bytes(b"database")
            path.chmod(0o644)
            for sidecar in sidecars:
                sidecar.write_bytes(sidecar.name.encode())
                sidecar.chmod(0o644)

            prepare_private_runtime_file(path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            for sidecar in sidecars:
                self.assertEqual(stat.S_IMODE(sidecar.stat().st_mode), 0o600)
                self.assertEqual(sidecar.read_bytes(), sidecar.name.encode())

    def test_prepare_private_runtime_file_rejects_sidecar_symlink_without_chmodding_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            path.write_bytes(b"database")
            target = Path(tmp) / "unrelated"
            target.write_bytes(b"unrelated")
            target.chmod(0o644)
            Path(f"{path}-wal").symlink_to(target)

            with self.assertRaisesRegex(OSError, "must not be a symbolic link"):
                prepare_private_runtime_file(path)

            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)
            self.assertEqual(target.read_bytes(), b"unrelated")

    def test_prepare_private_runtime_file_rolls_back_new_database_when_sidecar_is_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            target = Path(tmp) / "unrelated"
            target.write_bytes(b"must remain unchanged")
            Path(f"{path}-wal").symlink_to(target)

            with self.assertRaisesRegex(OSError, "must not be a symbolic link"):
                prepare_private_runtime_file(path)

            self.assertFalse(path.exists())
            self.assertEqual(target.read_bytes(), b"must remain unchanged")

    def test_prepare_private_runtime_file_failure_never_unlinks_concurrent_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            target = Path(tmp) / "unrelated"
            target.write_bytes(b"must remain unchanged")
            Path(f"{path}-wal").symlink_to(target)

            from leveraged_trader import runtime_files

            restrict_existing = runtime_files._restrict_existing_runtime_file
            replaced = False

            def replace_database_before_sidecar_validation(runtime_path: str) -> object:
                nonlocal replaced
                if runtime_path.endswith("-wal") and not replaced:
                    path.write_bytes(b"replacement database")
                    replaced = True
                return restrict_existing(runtime_path)

            with (
                patch(
                    "leveraged_trader.runtime_files._restrict_existing_runtime_file",
                    side_effect=replace_database_before_sidecar_validation,
                ),
                self.assertRaisesRegex(OSError, "must not be a symbolic link"),
            ):
                prepare_private_runtime_file(path)

            self.assertTrue(replaced)
            self.assertEqual(path.read_bytes(), b"replacement database")
            self.assertEqual(target.read_bytes(), b"must remain unchanged")

    def test_prepare_private_runtime_file_rejects_hardlinked_sqlite_sidecars(self) -> None:
        for suffix in ("-wal", "-shm", "-journal"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.sqlite"
                path.write_bytes(b"database")
                target = Path(tmp) / "unrelated"
                target.write_bytes(b"must remain unchanged")
                target.chmod(0o644)
                os.link(target, f"{path}{suffix}")

                with self.assertRaisesRegex(OSError, "multiple hard links"):
                    prepare_private_runtime_file(path)

                self.assertEqual(target.read_bytes(), b"must remain unchanged")
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_workflow_rejects_injected_sidecar_without_unlinking_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            victim = Path(tmp) / "unrelated"
            victim.write_bytes(b"must remain unchanged")
            original_prepare = prepare_private_runtime_file

            def prepare_then_inject_sidecar(runtime_path: str | os.PathLike[str]):
                guard = original_prepare(runtime_path)
                assert guard is not None
                os.link(victim, f"{guard.database_path}-shm")
                return guard

            with (
                patch(
                    "leveraged_trader.workflow.prepare_private_runtime_file",
                    side_effect=prepare_then_inject_sidecar,
                ),
                self.assertRaisesRegex(OSError, "SQLite sidecar.*changed identity or link count"),
            ):
                _initialize_state_db(str(path))

            self.assertEqual(victim.read_bytes(), b"must remain unchanged")
            self.assertEqual(victim.stat().st_nlink, 2)
            self.assertTrue(Path(f"{path}-shm").samefile(victim))

    def test_sidecar_revalidation_never_unlinks_a_concurrent_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            guard = prepare_private_runtime_file(path)
            assert guard is not None
            sidecar = Path(f"{path}-wal")
            sidecar.write_bytes(b"initial sidecar")

            from leveraged_trader import runtime_files

            restrict_existing = runtime_files._restrict_existing_runtime_file
            replaced = False

            def replace_during_sidecar_open(
                runtime_path: str | os.PathLike[str],
                **kwargs: object,
            ):
                nonlocal replaced
                if os.fspath(runtime_path).endswith("-wal") and not replaced:
                    sidecar.unlink()
                    sidecar.write_bytes(b"concurrent replacement")
                    replaced = True
                    raise OSError("simulated sidecar identity change")
                return restrict_existing(runtime_path, **kwargs)

            with (
                patch(
                    "leveraged_trader.runtime_files._restrict_existing_runtime_file",
                    side_effect=replace_during_sidecar_open,
                ),
                self.assertRaisesRegex(OSError, "rejected path was left untouched"),
            ):
                revalidate_private_runtime_file(guard)

            self.assertTrue(replaced)
            self.assertEqual(sidecar.read_bytes(), b"concurrent replacement")

    def test_database_revalidation_does_not_chmod_replacement_opened_after_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            guard = prepare_private_runtime_file(path)
            assert guard is not None

            from leveraged_trader import runtime_files

            open_existing = runtime_files._open_existing_regular_file
            replaced = False
            displaced = Path(tmp) / "displaced.sqlite"

            def replace_before_open(runtime_path: str | os.PathLike[str]) -> int:
                nonlocal replaced
                if os.fspath(runtime_path) == guard.database_path and not replaced:
                    path.rename(displaced)
                    path.write_bytes(b"replacement database")
                    path.chmod(0o644)
                    replaced = True
                return open_existing(runtime_path)

            with (
                patch(
                    "leveraged_trader.runtime_files._open_existing_regular_file",
                    side_effect=replace_before_open,
                ),
                self.assertRaisesRegex(OSError, "rejected path was left untouched"),
            ):
                revalidate_private_runtime_file(guard)

            self.assertTrue(replaced)
            self.assertEqual(path.read_bytes(), b"replacement database")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_nested_database_callers_reject_active_replacement_before_chmod(self) -> None:
        for caller in ("strategy-session", "state-connection", "save-table"):
            with self.subTest(caller=caller), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.sqlite"
                path.write_bytes(b"pinned database")
                path.chmod(0o600)
                guard = prepare_private_runtime_file(path)
                assert guard is not None
                displaced = Path(tmp) / "pinned.sqlite"
                real_prepare = prepare_private_runtime_file
                replacement_installed = False

                def replace_before_preparation(
                    runtime_path: str | os.PathLike[str],
                    *,
                    expected_guard=None,
                    pinned_guard=guard,
                    runtime_database=path,
                    displaced_database=displaced,
                    prepare_runtime_file=real_prepare,
                ):
                    nonlocal replacement_installed
                    self.assertIs(expected_guard, pinned_guard)
                    runtime_database.rename(displaced_database)
                    runtime_database.write_bytes(b"replacement database")
                    runtime_database.chmod(0o644)
                    replacement_installed = True
                    return prepare_runtime_file(runtime_path, expected_guard=expected_guard)

                patch_target = (
                    "leveraged_trader.storage.prepare_private_runtime_file"
                    if caller == "save-table"
                    else "leveraged_trader.workflow.prepare_private_runtime_file"
                )
                with (
                    activate_sqlite_runtime_file(
                        guard,
                        revalidate=lambda pinned_guard=guard: pinned_guard,
                    ),
                    patch(patch_target, side_effect=replace_before_preparation),
                    self.assertRaisesRegex(OSError, "changed identity.*left untouched"),
                ):
                    if caller == "strategy-session":
                        session = _WorkflowStrategySession(str(path))
                        try:
                            session._connection()
                        finally:
                            session.close()
                    elif caller == "state-connection":
                        with _state_connection(str(path)):
                            self.fail("state connection opened a replacement database")
                    else:
                        save_table_to_sqlite(
                            pd.DataFrame({"value": [1]}),
                            str(path),
                            "probe",
                        )

                self.assertTrue(replacement_installed)
                self.assertEqual(path.read_bytes(), b"replacement database")
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_active_preparation_accepts_normal_sqlite_sidecar_lifecycle(self) -> None:
        for lifecycle in ("appeared", "disappeared", "recreated"):
            with self.subTest(lifecycle=lifecycle), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.sqlite"
                path.write_bytes(b"database")
                sidecar = Path(f"{path}-wal")
                if lifecycle != "appeared":
                    sidecar.write_bytes(b"original sidecar")
                guard = prepare_private_runtime_file(path)
                assert guard is not None

                if lifecycle == "appeared":
                    sidecar.write_bytes(b"new sidecar")
                    sidecar.chmod(0o644)
                elif lifecycle == "disappeared":
                    sidecar.unlink()
                else:
                    sidecar.rename(Path(f"{sidecar}.displaced"))
                    sidecar.write_bytes(b"recreated sidecar")
                    sidecar.chmod(0o644)

                refreshed = prepare_private_runtime_file(path, expected_guard=guard)

                assert refreshed is not None
                refreshed_sidecars = dict(refreshed.sidecars)
                if lifecycle == "disappeared":
                    self.assertFalse(sidecar.exists())
                    self.assertIsNone(refreshed_sidecars[os.fspath(sidecar)])
                else:
                    self.assertEqual(stat.S_IMODE(sidecar.stat().st_mode), 0o600)
                    self.assertIsNotNone(refreshed_sidecars[os.fspath(sidecar)])

    def test_active_preparation_accepts_sidecar_unlinked_after_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            path.write_bytes(b"database")
            sidecar = Path(f"{path}-shm")
            sidecar.write_bytes(b"sidecar")
            guard = prepare_private_runtime_file(path)
            assert guard is not None

            real_open = os.open
            unlinked = False

            def open_then_unlink(runtime_path: str | os.PathLike[str], flags: int) -> int:
                nonlocal unlinked
                descriptor = real_open(runtime_path, flags)
                if os.fspath(runtime_path).endswith("-shm") and not unlinked:
                    sidecar.unlink()
                    unlinked = True
                return descriptor

            with patch("leveraged_trader.runtime_files.os.open", side_effect=open_then_unlink):
                refreshed = prepare_private_runtime_file(path, expected_guard=guard)

            assert refreshed is not None
            self.assertTrue(unlinked)
            self.assertIsNone(dict(refreshed.sidecars)[os.fspath(sidecar)])

    def test_active_sidecar_lstat_open_replacement_is_rejected_before_chmod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            path.write_bytes(b"database")
            sidecar = Path(f"{path}-wal")
            sidecar.write_bytes(b"original sidecar")
            guard = prepare_private_runtime_file(path)
            assert guard is not None

            from leveraged_trader import runtime_files

            open_existing = runtime_files._open_existing_regular_file
            displaced = Path(f"{sidecar}.displaced")
            replaced = False

            def replace_before_open(runtime_path: str | os.PathLike[str]) -> int:
                nonlocal replaced
                if os.fspath(runtime_path).endswith("-wal") and not replaced:
                    sidecar.rename(displaced)
                    sidecar.write_bytes(b"replacement sidecar")
                    sidecar.chmod(0o644)
                    replaced = True
                return open_existing(runtime_path)

            with (
                patch(
                    "leveraged_trader.runtime_files._open_existing_regular_file",
                    side_effect=replace_before_open,
                ),
                self.assertRaisesRegex(OSError, "changed while it was being prepared.*left untouched"),
            ):
                prepare_private_runtime_file(path, expected_guard=guard)

            self.assertTrue(replaced)
            self.assertEqual(sidecar.read_bytes(), b"replacement sidecar")
            self.assertEqual(stat.S_IMODE(sidecar.stat().st_mode), 0o644)

    def test_sidecar_revalidation_does_not_chmod_lstat_open_replacement(self) -> None:
        for present_at_snapshot in (False, True):
            with self.subTest(present_at_snapshot=present_at_snapshot), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.sqlite"
                sidecar = Path(f"{path}-wal")
                if present_at_snapshot:
                    path.write_bytes(b"database")
                    sidecar.write_bytes(b"snapshotted sidecar")
                guard = prepare_private_runtime_file(path)
                assert guard is not None
                if not present_at_snapshot:
                    sidecar.write_bytes(b"newly appeared sidecar")
                sidecar.chmod(0o600)

                from leveraged_trader import runtime_files

                open_existing = runtime_files._open_existing_regular_file
                replaced = False
                displaced = Path(f"{sidecar}.displaced")

                def replace_before_open(
                    runtime_path: str | os.PathLike[str],
                    sidecar_path: Path = sidecar,
                    displaced_path: Path = displaced,
                    open_runtime_file: Callable[[str | os.PathLike[str]], int] = open_existing,
                ) -> int:
                    nonlocal replaced
                    if os.fspath(runtime_path).endswith("-wal") and not replaced:
                        sidecar_path.rename(displaced_path)
                        sidecar_path.write_bytes(b"replacement sidecar")
                        sidecar_path.chmod(0o644)
                        replaced = True
                    return open_runtime_file(runtime_path)

                with (
                    patch(
                        "leveraged_trader.runtime_files._open_existing_regular_file",
                        side_effect=replace_before_open,
                    ),
                    self.assertRaisesRegex(OSError, "rejected path was left untouched"),
                ):
                    revalidate_private_runtime_file(guard)

                self.assertTrue(replaced)
                self.assertEqual(sidecar.read_bytes(), b"replacement sidecar")
                self.assertEqual(stat.S_IMODE(sidecar.stat().st_mode), 0o644)

    def test_prepare_private_runtime_file_hardens_resolved_database_and_sidecars_for_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "actual.sqlite"
            target.write_bytes(b"database")
            target.chmod(0o644)
            target_wal = Path(f"{target}-wal")
            target_wal.write_bytes(b"wal")
            target_wal.chmod(0o644)
            path = Path(tmp) / "state.sqlite"
            path.symlink_to(target)

            prepare_private_runtime_file(path)

            self.assertTrue(path.is_symlink())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(target_wal.stat().st_mode), 0o600)

    def test_prepare_private_runtime_file_does_not_chmod_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            path.mkdir(mode=0o755)
            mode_before = stat.S_IMODE(path.stat().st_mode)

            with self.assertRaisesRegex(OSError, "regular file"):
                prepare_private_runtime_file(path)

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), mode_before)

    def test_private_runtime_revalidation_reports_disappeared_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            guard = prepare_private_runtime_file(path)
            path.unlink()

            with self.assertRaisesRegex(OSError, "disappeared while it was open"):
                revalidate_private_runtime_file(guard)

    def test_workflow_initialization_defers_schema_commit_until_database_revalidation(self) -> None:
        for replacement in (False, True):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.sqlite"
                displaced = Path(tmp) / "displaced.sqlite"

                def initialize_then_displace(
                    conn: sqlite3.Connection,
                    *,
                    commit: bool = True,
                    runtime_path: Path = path,
                    displaced_path: Path = displaced,
                    install_replacement: bool = replacement,
                ) -> None:
                    self.assertFalse(commit)
                    init_state_db(conn, commit=commit)
                    self.assertTrue(conn.in_transaction)
                    runtime_path.rename(displaced_path)
                    if install_replacement:
                        runtime_path.touch(mode=0o600)

                with (
                    patch(
                        "leveraged_trader.workflow.init_state_db",
                        side_effect=initialize_then_displace,
                    ),
                    self.assertRaisesRegex(OSError, "disappeared while it was open|changed identity while it was open"),
                ):
                    _initialize_state_db(str(path))

                with closing(sqlite3.connect(displaced)) as conn:
                    self.assertEqual(
                        conn.execute("SELECT type, name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'").fetchall(),
                        [],
                    )
                if replacement:
                    self.assertTrue(path.exists())
                    with closing(sqlite3.connect(path)) as conn:
                        self.assertEqual(
                            conn.execute(
                                "SELECT type, name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
                            ).fetchall(),
                            [],
                        )

    def test_workflow_assets_defers_pandas_commit_until_database_revalidation(self) -> None:
        refreshed_assets = pd.DataFrame([{"symbol": "NEW"}])
        for replacement in (False, True):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "state.sqlite"
                displaced = Path(tmp) / "displaced.sqlite"
                with closing(sqlite3.connect(path)) as conn:
                    conn.execute("CREATE TABLE workflow_assets (symbol TEXT)")
                    conn.execute("INSERT INTO workflow_assets VALUES ('OLD')")
                    conn.commit()

                def save_then_displace(
                    conn: sqlite3.Connection,
                    workflow_assets: pd.DataFrame,
                    *,
                    runtime_path: Path = path,
                    displaced_path: Path = displaced,
                    install_replacement: bool = replacement,
                ) -> None:
                    save_workflow_assets(conn, workflow_assets)
                    self.assertTrue(conn.in_transaction)
                    runtime_path.rename(displaced_path)
                    if install_replacement:
                        runtime_path.touch(mode=0o600)

                with (
                    patch(
                        "leveraged_trader.workflow.determine_workflow_asset_groups",
                        return_value={"long": refreshed_assets},
                    ),
                    patch(
                        "leveraged_trader.workflow.save_workflow_assets",
                        side_effect=save_then_displace,
                    ),
                    self.assertRaisesRegex(OSError, "disappeared while it was open|changed identity while it was open"),
                ):
                    _load_or_refresh_workflow_assets_for_db(str(path), object())  # type: ignore[arg-type]

                with closing(sqlite3.connect(displaced)) as conn:
                    self.assertEqual(
                        conn.execute("SELECT symbol FROM workflow_assets").fetchall(),
                        [("OLD",)],
                    )
                if replacement:
                    self.assertTrue(path.exists())
                    with closing(sqlite3.connect(path)) as conn:
                        self.assertEqual(
                            conn.execute(
                                "SELECT type, name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
                            ).fetchall(),
                            [],
                        )

    def test_require_private_file_rejects_group_or_world_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("SECRET=value\n", encoding="utf-8")
            path.chmod(0o640)

            with self.assertRaisesRegex(PermissionError, "chmod 600"):
                require_private_file(path, label="Environment file")

    def test_require_private_file_accepts_owner_only_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("SECRET=value\n", encoding="utf-8")
            path.chmod(0o600)

            require_private_file(path, label="Environment file")

    def test_require_private_file_rejects_owner_only_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            os.mkfifo(path, mode=0o600)

            with self.assertRaisesRegex(OSError, "regular file"):
                require_private_file(path, label="Environment file")

    def test_open_private_text_file_keeps_the_verified_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("VALUE=verified\n", encoding="utf-8")
            path.chmod(0o600)

            with open_private_text_file(path, label="Environment file") as private_file:
                path.unlink()
                path.write_text("VALUE=substituted\n", encoding="utf-8")
                path.chmod(0o600)
                contents = private_file.read()

            self.assertEqual(contents, "VALUE=verified\n")

    def test_table_writer_creates_owner_only_sqlite_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.sqlite"

            save_table_to_sqlite(pd.DataFrame([{"symbol": "TQQQ"}]), str(path), "universe")

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_table_writer_preserves_pandas_sqlite_schema_and_values(self) -> None:
        frame = pd.DataFrame(
            {
                "name": pd.Series(["alpha", pd.NA], dtype="string"),
                "count": pd.Series([1, pd.NA], dtype="Int64"),
                "value": [1.25, float("nan")],
                "enabled": pd.Series([True, pd.NA], dtype="boolean"),
                "observed_at": pd.to_datetime(["2026-08-31T12:34:56", None]),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            expected_path = Path(tmp) / "expected.sqlite"
            actual_path = Path(tmp) / "actual.sqlite"
            with sqlite3.connect(expected_path) as conn:
                frame.to_sql("universe", conn, if_exists="replace", index=False)

            save_table_to_sqlite(frame, str(actual_path), "universe")

            with sqlite3.connect(expected_path) as expected, sqlite3.connect(actual_path) as actual:
                self.assertEqual(
                    actual.execute("PRAGMA table_info(universe)").fetchall(),
                    expected.execute("PRAGMA table_info(universe)").fetchall(),
                )
                self.assertEqual(
                    actual.execute("SELECT * FROM universe").fetchall(),
                    expected.execute("SELECT * FROM universe").fetchall(),
                )

    def test_table_writer_validates_before_its_only_commit_and_rolls_back(self) -> None:
        cases = (
            ("revalidate_private_runtime_file", 2),
            ("revalidate_active_sqlite_runtime_file", 4),
        )
        for validator_name, rejected_call in cases:
            with self.subTest(validator=validator_name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "universe.sqlite"
                with sqlite3.connect(path) as conn:
                    conn.execute("CREATE TABLE universe (symbol TEXT)")
                    conn.execute("INSERT INTO universe VALUES ('OLD')")

                validation_calls = 0

                def reject_before_commit(
                    *args: object,
                    _rejected_call: int = rejected_call,
                    _validator_name: str = validator_name,
                ) -> object | None:
                    nonlocal validation_calls
                    validation_calls += 1
                    if validation_calls == _rejected_call:
                        raise OSError("simulated pre-commit runtime-file validation failure")
                    return args[0] if _validator_name == "revalidate_private_runtime_file" else None

                with (
                    patch(f"leveraged_trader.storage.{validator_name}", side_effect=reject_before_commit),
                    self.assertRaisesRegex(OSError, "simulated pre-commit"),
                ):
                    save_table_to_sqlite(pd.DataFrame([{"symbol": "NEW"}]), str(path), "universe")

                with sqlite3.connect(path) as conn:
                    self.assertEqual(conn.execute("SELECT symbol FROM universe").fetchall(), [("OLD",)])

    def test_table_writer_rolls_back_failed_replacement_insert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.sqlite"
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE universe (symbol TEXT)")
                conn.execute("INSERT INTO universe VALUES ('OLD')")

            with self.assertRaisesRegex(pd.errors.DatabaseError, "Execution failed") as raised:
                save_table_to_sqlite(pd.DataFrame([{"symbol": object()}]), str(path), "universe")

            self.assertIsInstance(raised.exception.__cause__, sqlite3.ProgrammingError)
            with sqlite3.connect(path) as conn:
                self.assertEqual(conn.execute("SELECT symbol FROM universe").fetchall(), [("OLD",)])

    def test_table_writer_supports_empty_frames_and_quoted_table_names(self) -> None:
        frame = pd.DataFrame(
            {
                "symbol": pd.Series(dtype="string"),
                "score": pd.Series(dtype="float64"),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.sqlite"

            save_table_to_sqlite(frame, str(path), 'quoted "universe')

            with sqlite3.connect(path) as conn:
                self.assertEqual(
                    conn.execute('PRAGMA table_info("quoted ""universe")').fetchall(),
                    [
                        (0, "symbol", "TEXT", 0, None, 0),
                        (1, "score", "REAL", 0, None, 0),
                    ],
                )
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM "quoted ""universe"').fetchone(), (0,))

    def test_table_writer_closes_sqlite_connection(self) -> None:
        closed_connections: list[sqlite3.Connection] = []
        original_connect = sqlite3.connect

        def tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
            requested_factory = kwargs.pop("factory")

            class TrackingConnection(requested_factory):  # type: ignore[misc, valid-type]
                def close(self) -> None:
                    closed_connections.append(self)
                    super().close()

            return original_connect(*args, **kwargs, factory=TrackingConnection)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "universe.sqlite"
            with patch("leveraged_trader.storage.sqlite3.connect", side_effect=tracking_connect):
                save_table_to_sqlite(pd.DataFrame([{"symbol": "TQQQ"}]), str(path), "universe")

        self.assertEqual(len(closed_connections), 1)

    def test_workflow_initialization_restricts_existing_sqlite_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            path.touch(mode=0o644)
            path.chmod(0o644)

            _initialize_state_db(str(path))

            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
