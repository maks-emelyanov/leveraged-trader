from __future__ import annotations

import os
import pwd
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SCHEDULER = ROOT / "scripts" / "cron" / "run-scheduled"
SCHEDULER_CLEAN_ENVIRONMENT = ROOT / "scripts" / "cron" / "run-scheduled-clean-environment"
INSTALLER = ROOT / "scripts" / "cron" / "install-crontab"
RUNNER = ROOT / "scripts" / "cron" / "run-leveraged-trader"
INSTALLER_TEST_MODE_ENV = "LEVERAGED_TRADER_INSTALLER_TEST_ONLY_MODE"
INSTALLER_TEST_HOME_ENV = "LEVERAGED_TRADER_INSTALLER_TEST_ONLY_ACCOUNT_HOME"
SCHEDULED_RECONCILIATION_ARGUMENTS = (
    "--reconcile-only --alpaca-submit-sell-orders "
    "--scheduled-closed-audit-interval-minutes 15"
)
CRON_BOUNDARY_ENVIRONMENT_NAMES = (
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "DYLD_FALLBACK_FRAMEWORK_PATH",
    "LIBPATH",
    "SHLIB_PATH",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "FTP_PROXY",
    "ftp_proxy",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SSLKEYLOGFILE",
    "OPENSSL_CONF",
    "OPENSSL_CONF_INCLUDE",
    "OPENSSL_MODULES",
    "OPENSSL_ENGINES",
)


def test_readme_cron_update_instructions_cover_pinned_runtime_files() -> None:
    readme = README.read_text(encoding="utf-8")
    update_section = readme.split("### 7. Updating an existing installation", 1)[1].split("\n## ", 1)[0]

    assert "If you installed the managed cron schedule" in update_section
    assert "always reinstall" in update_section
    assert "./scripts/cron/install-crontab" in update_section
    assert "run-scheduled-clean-environment" in update_section
    assert "runtime-security" in update_section
    assert "pinned security-helper digest" in update_section


def _installer_test_account_home(tmp_path: Path) -> Path:
    return tmp_path / "installer-account-home"


def _installer_lock_root(account_home: Path) -> Path:
    return account_home / ".local" / "state" / "leveraged-trader" / "crontab-install"


def _installer_bootstrap_root(account_home: Path) -> Path:
    return account_home / ".local" / "state" / "leveraged-trader" / "cron-runtime"


def _legacy_installer_lock_root() -> Path:
    return Path("/tmp").resolve() / f".leveraged-trader-crontab-install-{os.geteuid()}"


@pytest.fixture(autouse=True)
def _isolate_installer_lock_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(INSTALLER_TEST_MODE_ENV, "1")
    monkeypatch.setenv(INSTALLER_TEST_HOME_ENV, str(_installer_test_account_home(tmp_path)))


def _process_start_identity(pid: int) -> str:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        fields = proc_stat.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        return f"proc:{fields[19]}"
    result = subprocess.run(
        ["ps", "-o", "lstart=", "-p", str(pid)],
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC0"},
        check=True,
        capture_output=True,
        text=True,
    )
    return f"ps:{result.stdout.strip()}"


def _terminate_and_reap(process: subprocess.Popen[str]) -> tuple[str | None, str | None]:
    """Terminate a live fixture process and close any captured pipe streams."""

    if process.poll() is None:
        process.kill()
    return process.communicate(timeout=2)


def _wait_for_path(
    path: Path,
    timeout: float = 10.0,
    *,
    process: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process is not None and process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"failed waiting for {path}: process exited with "
                f"{process.returncode}; stdout={stdout!r}; stderr={stderr!r}"
            )
        time.sleep(0.01)
    if process is not None:
        if process.poll() is None:
            process.kill()
        stdout, stderr = process.communicate(timeout=2)
        raise AssertionError(
            f"timed out waiting for {path}: process exited with "
            f"{process.returncode}; stdout={stdout!r}; stderr={stderr!r}"
        )
    raise AssertionError(f"timed out waiting for {path}")


def _run_scheduler(now: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_LOG_FILE", None)
    env.pop("LEVERAGED_TRADER_LOG_MAX_BYTES", None)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = now
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    return subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _cron_shell_command(schedule_line: str) -> str:
    """Apply Vixie/Cronie backslash-parity handling to a schedule command."""

    command = schedule_line.split(maxsplit=5)[5]
    parsed: list[str] = []
    preceding_backslashes = 0
    for character in command:
        if character == "%":
            if preceding_backslashes % 2 == 0:
                raise AssertionError("cron would split the command at an unescaped percent")
            parsed.pop()
            parsed.append(character)
            preceding_backslashes = 0
        else:
            parsed.append(character)
            if character == "\\":
                preceding_backslashes += 1
            else:
                preceding_backslashes = 0
    return "".join(parsed)


def _cron_assignment_value(value: str) -> str:
    """Apply cron's outer-quote handling to an environment value."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _install_test_schedule(
    tmp_path: Path,
    *,
    installer_bash: Path = Path("/bin/bash"),
) -> tuple[Path, Path, dict[str, str]]:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("", encoding="utf-8")
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CRONTAB_STATE"] = str(crontab_state)
    result = subprocess.run(
        [str(installer_bash), str(repo_dir / "scripts" / "cron" / "install-crontab")],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return repo_dir, crontab_state, env


def _run_installed_schedule(
    repo_dir: Path,
    crontab_state: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    installed_lines = crontab_state.read_text(encoding="utf-8").splitlines()
    schedule_line = next(line for line in installed_lines if line.startswith("* * * * * "))
    cron_env = env.copy()
    for line in installed_lines[: installed_lines.index(schedule_line)]:
        if "=" not in line or line.startswith("#"):
            continue
        name, value = line.split("=", 1)
        cron_env[name] = _cron_assignment_value(value)
    return subprocess.run(
        ["/bin/sh", "-c", _cron_shell_command(schedule_line)],
        cwd=repo_dir,
        env=cron_env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cron_parser_fixture_rejects_even_backslash_run_before_percent() -> None:
    with pytest.raises(AssertionError, match="unescaped percent"):
        _cron_shell_command(r"* * * * * printf '%s' 'slash\\%50'")


@pytest.mark.parametrize("weekday", ["1", "3", "5"])
def test_scheduled_full_run_explicitly_enables_buy_and_sell_submission(weekday: str) -> None:
    result = _run_scheduler(f"{weekday} 08:45")

    assert result.returncode == 0
    assert result.stdout.strip() == (
        "--require-workflow-source-success --alpaca-submit-buy-orders --alpaca-submit-sell-orders"
    )


def test_scheduler_runs_with_managed_minimal_path() -> None:
    env = {
        "PATH": "/usr/bin:/bin",
        "LEVERAGED_TRADER_SCHEDULE_NOW": "3 08:45",
        "LEVERAGED_TRADER_RUNNER": "/bin/echo",
    }

    result = subprocess.run(
        [str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == (
        "--require-workflow-source-success --alpaca-submit-buy-orders --alpaca-submit-sell-orders"
    )


def test_scheduler_uses_locked_project_tzdata_not_system_tzdir(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    schedule_python = repo_dir / ".venv" / "bin" / "python"
    schedule_python.parent.mkdir(parents=True)
    python_code = tmp_path / "schedule-python-code"
    uv_verified = tmp_path / "uv-verified"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/bin/bash
set -euo pipefail
[[ "$PWD" == "$EXPECTED_REPO_DIR" ]]
[[ "$#" -eq 8 ]]
[[ "$1" == "sync" && "$2" == "--project" && "$3" == "$EXPECTED_REPO_DIR" ]]
[[ "$4" == "--directory" && "$5" == "$EXPECTED_REPO_DIR" ]]
[[ "$6" == "--locked" && "$7" == "--check" && "$8" == "--quiet" ]]
[[ "$UV_PROJECT_ENVIRONMENT" == "$EXPECTED_PROJECT_ENVIRONMENT" ]]
for environment_name in $(compgen -e); do
    case "$environment_name" in
        UV_PROJECT_ENVIRONMENT) ;;
        UV_*) echo "unexpected inherited uv setting: $environment_name" >&2; exit 91 ;;
    esac
done
: > "$TEST_UV_VERIFIED"
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    schedule_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$#" -eq 4 && "$1" == "-I" && "$2" == "-S" && "$3" == "-c" ]]; then
    printf '1770215400 3 14:30'
    exit 0
fi
[[ "$#" -eq 4 && "$1" == "-I" && "$2" == "-c" && "$4" == "1770215400" ]]
[[ -e "$TEST_UV_VERIFIED" ]]
printf '%s' "$3" > "$SCHEDULE_PYTHON_CODE"
printf '3 09:30'
""",
        encoding="utf-8",
    )
    schedule_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env.pop("LEVERAGED_TRADER_LOG_FILE", None)
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["SCHEDULE_PYTHON_CODE"] = str(python_code)
    env["UV_BIN"] = str(fake_uv)
    env["TEST_UV_VERIFIED"] = str(uv_verified)
    env["EXPECTED_REPO_DIR"] = str(repo_dir)
    env["EXPECTED_PROJECT_ENVIRONMENT"] = str(repo_dir / ".venv")
    env["TZDIR"] = "/definitely/missing/system-zoneinfo"
    env["PYTHONTZPATH"] = "/untrusted/inherited-zoneinfo"
    env["UV_PROJECT"] = "/untrusted/project"
    env["UV_WORKING_DIR"] = "/untrusted/working-directory"
    env["UV_CONFIG_FILE"] = "/untrusted/uv.toml"
    env["UV_NO_DEV"] = "1"
    env["UV_NO_GROUP"] = "dev"
    env["UV_NO_DEFAULT_GROUPS"] = "1"
    env["UV_FROZEN"] = "1"
    env["UV_PYTHON"] = "/bin/false"
    env["UV_NO_EDITABLE"] = "1"
    env["UV_FUTURE_OVERRIDE"] = "unexpected"
    result = subprocess.run(
        ["bash", str(scheduler)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == SCHEDULED_RECONCILIATION_ARGUMENTS
    assert uv_verified.is_file()
    invoked_code = python_code.read_text(encoding="utf-8")
    assert "reset_tzpath(())" in invoked_code
    assert 'ZoneInfo("America/New_York")' in invoked_code
    assert "datetime.fromtimestamp(int(sys.argv[1]), eastern)" in invoked_code


def test_scheduler_rejects_stale_environment_before_exact_non_due_conversion(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    schedule_python = repo_dir / ".venv" / "bin" / "python"
    schedule_python.parent.mkdir(parents=True)
    site_conversion_called = tmp_path / "site-conversion-called"
    schedule_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$#" -eq 4 && "$1" == "-I" && "$2" == "-S" && "$3" == "-c" ]]; then
    printf '1770212640 3 13:44'
    exit 0
fi
: > "$SITE_CONVERSION_CALLED"
printf '3 08:44'
""",
        encoding="utf-8",
    )
    schedule_python.chmod(0o755)

    uv_called = tmp_path / "uv-called"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/bin/bash\n: > "$TEST_UV_CALLED"\nexit 23\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    runner_called = tmp_path / "runner-called"
    runner = tmp_path / "runner"
    runner.write_text('#!/bin/bash\n: > "$RUNNER_CALLED"\n', encoding="utf-8")
    runner.chmod(0o755)

    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env.pop("LEVERAGED_TRADER_LOG_FILE", None)
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["UV_BIN"] = str(fake_uv)
    env["TEST_UV_CALLED"] = str(uv_called)
    env["SITE_CONVERSION_CALLED"] = str(site_conversion_called)
    env["RUNNER_CALLED"] = str(runner_called)
    result = subprocess.run(
        ["/bin/bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 23
    assert uv_called.is_file()
    assert not site_conversion_called.exists()
    assert not runner_called.exists()
    assert "Locked project environment is not synchronized" in result.stderr


def test_scheduler_rejects_writable_pth_before_site_enabled_clock_conversion(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    package_dir = repo_dir / "leveraged_trader"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    project_environment = repo_dir / ".venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(project_environment)],
        check=True,
        capture_output=True,
        text=True,
    )
    schedule_python = project_environment / "bin" / "python"
    real_schedule_python = project_environment / "bin" / "python-real"
    schedule_python.rename(real_schedule_python)
    schedule_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$#" -eq 4 && "$1" == "-I" && "$2" == "-S" && "$3" == "-c" ]]; then
    printf '1770215400 3 14:30'
    exit 0
fi
script_dir="${BASH_SOURCE[0]%/*}"
exec "$script_dir/python-real" "$@"
""",
        encoding="utf-8",
    )
    schedule_python.chmod(0o755)
    site_packages_candidates = list(project_environment.glob("lib/python*/site-packages"))
    assert len(site_packages_candidates) == 1
    startup_marker = tmp_path / "pth-startup-executed"
    editable_artifact = site_packages_candidates[0] / "__editable__.leveraged_trader-0.1.0.pth"
    editable_artifact.write_text(
        "import os; open(os.environ['PTH_STARTUP_MARKER'], 'w').close()\n",
        encoding="utf-8",
    )
    editable_artifact.chmod(0o664)

    uv_marker = tmp_path / "uv-executed"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/bin/bash\n: > "$UV_EXECUTED"\n', encoding="utf-8")
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env.pop("LEVERAGED_TRADER_LOG_FILE", None)
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["UV_BIN"] = str(fake_uv)
    env["UV_EXECUTED"] = str(uv_marker)
    env["PTH_STARTUP_MARKER"] = str(startup_marker)
    result = subprocess.run(
        ["/bin/bash", str(repo_dir / "scripts" / "cron" / "run-scheduled")],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "project site-packages tree" in result.stderr
    assert "must not be group- or world-writable" in result.stderr
    assert not startup_marker.exists()
    assert not uv_marker.exists()


def test_scheduler_snapshots_due_minute_before_slow_environment_preflight(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    runtime_security = repo_dir / "scripts" / "cron" / "runtime-security"
    runtime_security_source = runtime_security.read_text(encoding="utf-8")
    import_validator_start = "runtime_security_validate_leveraged_trader_import_paths() {\n"
    assert import_validator_start in runtime_security_source
    runtime_security.write_text(
        runtime_security_source.replace(
            import_validator_start,
            import_validator_start
            + '    [[ -e "$CLOCK_SAMPLED" ]] || return 91\n'
            + '    : > "$IMPORT_SCAN_STARTED"\n'
            + "    /bin/sleep 0.2\n",
            1,
        ),
        encoding="utf-8",
    )
    schedule_python = repo_dir / ".venv" / "bin" / "python"
    schedule_python.parent.mkdir(parents=True)
    clock_sampled = tmp_path / "clock-sampled"
    import_scan_started = tmp_path / "import-scan-started"
    schedule_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$#" -eq 4 && "$1" == "-I" && "$2" == "-S" && "$3" == "-c" ]]; then
    : > "$CLOCK_SAMPLED"
    printf '1770215400 3 14:30'
    exit 0
fi
[[ "$#" -eq 4 && "$1" == "-I" && "$2" == "-c" && "$4" == "1770215400" ]]
[[ -e "$IMPORT_SCAN_STARTED" ]]
printf '3 08:45'
""",
        encoding="utf-8",
    )
    schedule_python.chmod(0o755)

    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/bin/bash\nset -euo pipefail\n[[ -e "$CLOCK_SAMPLED" ]] || exit 91\n/bin/sleep 0.2\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    runner_called = tmp_path / "runner-called"
    runner = tmp_path / "runner"
    runner.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$RUNNER_CALLED"\n',
        encoding="utf-8",
    )
    runner.chmod(0o755)

    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env.pop("LEVERAGED_TRADER_LOG_FILE", None)
    env["CLOCK_SAMPLED"] = str(clock_sampled)
    env["IMPORT_SCAN_STARTED"] = str(import_scan_started)
    env["RUNNER_CALLED"] = str(runner_called)
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["UV_BIN"] = str(fake_uv)
    result = subprocess.run(
        ["/bin/bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert clock_sampled.is_file()
    assert import_scan_started.is_file()
    assert runner_called.read_text(encoding="utf-8").splitlines() == [
        "--require-workflow-source-success",
        "--alpaca-submit-buy-orders",
        "--alpaca-submit-sell-orders",
        "--workflow-deadline-epoch",
        "1770217500",
    ]


def test_slow_non_due_preflight_does_not_hold_execution_lock_over_due_minute(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    runtime_security = repo_dir / "scripts" / "cron" / "runtime-security"
    runtime_security_source = runtime_security.read_text(encoding="utf-8")
    import_validator_start = "runtime_security_validate_leveraged_trader_import_paths() {\n"
    assert import_validator_start in runtime_security_source
    runtime_security.write_text(
        runtime_security_source.replace(
            import_validator_start,
            import_validator_start
            + '    : > "$PREFLIGHT_ENTERED"\n'
            + '    while [[ ! -e "$RELEASE_PREFLIGHT" ]]; do /bin/sleep 0.01; done\n',
            1,
        ),
        encoding="utf-8",
    )
    schedule_python = repo_dir / ".venv" / "bin" / "python"
    schedule_python.parent.mkdir(parents=True)
    schedule_python.write_text(
        """#!/bin/bash
if [[ "$2" == "-S" ]]; then
    printf '1770212640 3 13:44'
else
    printf '3 08:44'
fi
""",
        encoding="utf-8",
    )
    schedule_python.chmod(0o755)

    preflight_entered = tmp_path / "preflight-entered"
    release_preflight = tmp_path / "release-preflight"
    uv_called = tmp_path / "uv-called"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/bin/bash\nset -euo pipefail\n: > "$TEST_UV_CALLED"\nexit 0\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    runner_called = tmp_path / "runner-called"
    runner = tmp_path / "runner"
    runner.write_text('#!/bin/bash\n: > "$RUNNER_CALLED"\n', encoding="utf-8")
    runner.chmod(0o755)
    log_file = tmp_path / "cron.log"

    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["UV_BIN"] = str(fake_uv)
    env["PREFLIGHT_ENTERED"] = str(preflight_entered)
    env["RELEASE_PREFLIGHT"] = str(release_preflight)
    env["RUNNER_CALLED"] = str(runner_called)
    env["TEST_UV_CALLED"] = str(uv_called)
    early = subprocess.Popen(
        ["/bin/bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_path(preflight_entered)
        due_env = {**env, "LEVERAGED_TRADER_SCHEDULE_NOW": "3 08:45"}
        due = subprocess.run(
            ["/bin/bash", str(scheduler)],
            cwd=repo_dir,
            env=due_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert due.returncode == 0, due.stderr
        assert runner_called.is_file()

        release_preflight.touch()
        early_stdout, early_stderr = early.communicate(timeout=5)
        assert early.returncode == 0, (early_stdout, early_stderr)
        assert uv_called.is_file()
    finally:
        release_preflight.touch(exist_ok=True)
        _terminate_and_reap(early)


@pytest.mark.parametrize(
    ("tzdata_versions", "expected_status", "expect_import_scan"),
    [
        (("2025.3",), 0, False),
        (("2099.1",), 91, True),
        (("2025.3", "2099.1"), 91, True),
    ],
)
def test_clearly_non_due_utc_gate_requires_the_audited_tzdata_lock(
    tmp_path: Path,
    tzdata_versions: tuple[str, ...],
    expected_status: int,
    expect_import_scan: bool,
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    (repo_dir / "uv.lock").write_text(
        "".join(f'[[package]]\nname = "tzdata"\nversion = "{version}"\n' for version in tzdata_versions),
        encoding="utf-8",
    )
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    runtime_security = repo_dir / "scripts" / "cron" / "runtime-security"
    runtime_security_source = runtime_security.read_text(encoding="utf-8")
    import_validator_start = "runtime_security_validate_leveraged_trader_import_paths() {\n"
    assert import_validator_start in runtime_security_source
    runtime_security.write_text(
        runtime_security_source.replace(
            import_validator_start,
            import_validator_start + '    : > "$IMPORT_SCAN_CALLED"\n    return 91\n',
            1,
        ),
        encoding="utf-8",
    )

    clock_sampled = tmp_path / "clock-sampled"
    site_conversion_called = tmp_path / "site-conversion-called"
    schedule_python = repo_dir / ".venv" / "bin" / "python"
    schedule_python.parent.mkdir(parents=True)
    schedule_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$#" -eq 4 && "$1" == "-I" && "$2" == "-S" && "$3" == "-c" ]]; then
    : > "$CLOCK_SAMPLED"
    printf '1770172800 3 02:40'
    exit 0
fi
: > "$SITE_CONVERSION_CALLED"
exit 92
""",
        encoding="utf-8",
    )
    schedule_python.chmod(0o755)

    uv_called = tmp_path / "uv-called"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/bin/bash\n: > "$UV_CALLED"\nexit 93\n', encoding="utf-8")
    fake_uv.chmod(0o755)
    runner_called = tmp_path / "runner-called"
    runner = tmp_path / "runner"
    runner.write_text('#!/bin/bash\n: > "$RUNNER_CALLED"\nexit 94\n', encoding="utf-8")
    runner.chmod(0o755)
    import_scan_called = tmp_path / "import-scan-called"

    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env.pop("LEVERAGED_TRADER_LOG_FILE", None)
    env["CLOCK_SAMPLED"] = str(clock_sampled)
    env["SITE_CONVERSION_CALLED"] = str(site_conversion_called)
    env["IMPORT_SCAN_CALLED"] = str(import_scan_called)
    env["UV_CALLED"] = str(uv_called)
    env["RUNNER_CALLED"] = str(runner_called)
    env["UV_BIN"] = str(fake_uv)
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)

    result = subprocess.run(
        ["/bin/bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == expected_status, result.stderr
    assert clock_sampled.is_file()
    assert not site_conversion_called.exists()
    assert import_scan_called.exists() is expect_import_scan
    assert not uv_called.exists()
    assert not runner_called.exists()


def test_scheduler_samples_clock_before_reporting_failed_preflight(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    schedule_python = repo_dir / ".venv" / "bin" / "python"
    schedule_python.parent.mkdir(parents=True)
    python_called = tmp_path / "python-called"
    schedule_python.write_text(
        """#!/bin/bash
set -euo pipefail
: > "$PYTHON_CALLED"
if [[ "$2" == "-S" ]]; then
    printf '1770215400 3 14:30'
else
    [[ "$4" == "1770215400" ]]
    printf '3 09:30'
fi
""",
        encoding="utf-8",
    )
    schedule_python.chmod(0o755)

    uv_capture = tmp_path / "uv-capture"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$PWD" "$UV_PROJECT_ENVIRONMENT" "$@" > "$TEST_UV_CAPTURE"
exit 23
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    runner_called = tmp_path / "runner-called"
    runner = tmp_path / "runner"
    runner.write_text('#!/bin/bash\n: > "$RUNNER_CALLED"\n', encoding="utf-8")
    runner.chmod(0o755)
    log_file = tmp_path / "cron.log"

    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["RUNNER_CALLED"] = str(runner_called)
    env["PYTHON_CALLED"] = str(python_called)
    env["UV_BIN"] = str(fake_uv)
    env["TEST_UV_CAPTURE"] = str(uv_capture)
    result = subprocess.run(
        ["bash", str(scheduler)],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 23
    assert uv_capture.read_text(encoding="utf-8").splitlines() == [
        str(repo_dir),
        str(repo_dir / ".venv"),
        "sync",
        "--project",
        str(repo_dir),
        "--directory",
        str(repo_dir),
        "--locked",
        "--check",
        "--quiet",
    ]
    assert python_called.is_file()
    assert "no scheduled command was selected" in result.stderr
    assert "uv sync --locked" in result.stderr
    assert log_file.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "cron.log.lock").stat().st_mode & 0o777 == 0o600
    logged_failure = log_file.read_text(encoding="utf-8")
    assert "Scheduler preflight failed with status 23" in logged_failure
    assert "no scheduled command was selected" in logged_failure
    assert "uv sync --locked" in logged_failure
    assert not runner_called.exists()


def test_scheduler_fallback_lock_logs_preflight_failure_and_cleans_up(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    schedule_python = repo_dir / ".venv" / "bin" / "python"
    schedule_python.parent.mkdir(parents=True)
    schedule_python.write_text(
        """#!/bin/bash
if [[ "$2" == "-S" ]]; then
    printf '1770215400 3 14:30'
else
    printf '3 09:30'
fi
""",
        encoding="utf-8",
    )
    schedule_python.chmod(0o755)

    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/bash\necho 'simulated stale environment' >&2\nexit 23\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "mv", "ps", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    log_file = tmp_path / "cron.log"
    fallback_lock = tmp_path / "cron.log.lock.d"
    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["UV_BIN"] = str(fake_uv)
    result = subprocess.run(
        ["/bin/bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 23
    assert "simulated stale environment" in result.stderr
    logged_failure = log_file.read_text(encoding="utf-8")
    assert "Scheduler preflight failed with status 23" in logged_failure
    assert "simulated stale environment" in logged_failure
    assert not fallback_lock.exists()


def test_scheduler_fallback_owner_publication_uses_portable_paths(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    fallback_lock = tmp_path / "cron.log.lock.d"
    mv_arguments = tmp_path / "mv-arguments"
    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "ps", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    real_mv = shutil.which("mv")
    assert real_mv is not None
    fake_mv = bin_dir / "mv"
    fake_mv.write_text(
        """#!/bin/bash
set -euo pipefail
for argument in "$@"; do
    printf '%s\n' "$argument" >> "$MV_ARGUMENTS"
    case "$argument" in
        /dev/fd/*/*) exit 91 ;;
    esac
done
exec "$REAL_MV" "$@"
""",
        encoding="utf-8",
    )
    fake_mv.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["MV_ARGUMENTS"] = str(mv_arguments)
    env["REAL_MV"] = real_mv
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (fallback_lock / "owner").is_file()
    published_paths = mv_arguments.read_text(encoding="utf-8").splitlines()
    assert published_paths == ["./owner.pending", "./owner"]
    assert all(not path.startswith("/dev/fd/") for path in published_paths)


def test_scheduler_fallback_lock_descriptor_is_used_only_for_identity() -> None:
    scheduler_source = SCHEDULER.read_text(encoding="utf-8")

    assert "/dev/fd/5/" not in scheduler_source
    assert "$lock_descriptor_path/" not in scheduler_source
    assert "${lock_descriptor_path}/" not in scheduler_source


def test_scheduler_fallback_ps_identity_forces_c_locale_and_fixed_timezone(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    scheduler_source = scheduler.read_text(encoding="utf-8")
    proc_branch = 'if [[ -r "/proc/$pid/stat" ]]; then'
    assert scheduler_source.count(proc_branch) == 1
    scheduler.write_text(scheduler_source.replace(proc_branch, "if false; then", 1), encoding="utf-8")

    log_file = tmp_path / "cron.log"
    fallback_lock = tmp_path / "cron.log.lock.d"
    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "mv", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    real_ps = shutil.which("ps")
    assert real_ps is not None
    fake_ps = bin_dir / "ps"
    fake_ps.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${LC_ALL:-}" != C ]]; then
    echo "ps did not receive LC_ALL=C" >&2
    exit 93
fi
if [[ "${TZ:-}" != UTC0 ]]; then
    echo "ps did not receive TZ=UTC0" >&2
    exit 94
fi
exec "$REAL_PS" "$@"
""",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)

    env = os.environ.copy()
    env.pop("LANG", None)
    env.pop("LC_ALL", None)
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["REAL_PS"] = real_ps
    result = subprocess.run(
        ["/bin/bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    owner_line = (fallback_lock / "owner").read_text(encoding="utf-8").strip()
    assert owner_line.split(" ", 1)[1].startswith("ps:")


def test_scheduler_fallback_ps_identity_is_stable_across_inherited_timezones(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    scheduler_source = scheduler.read_text(encoding="utf-8")
    proc_branch = 'if [[ -r "/proc/$pid/stat" ]]; then'
    assert scheduler_source.count(proc_branch) == 1
    scheduler.write_text(scheduler_source.replace(proc_branch, "if false; then", 1), encoding="utf-8")

    log_file = tmp_path / "cron.log"
    fallback_lock = tmp_path / "cron.log.lock.d"
    record_file = tmp_path / "runs"
    first_started = tmp_path / "first-started"
    block_file = tmp_path / "block"
    block_file.touch()
    runner = tmp_path / "runner"
    runner.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$$" >> "$RECORD_FILE"
if [[ ! -e "$FIRST_STARTED" ]]; then
    : > "$FIRST_STARTED"
    while [[ -e "$BLOCK_FILE" ]]; do
        /bin/sleep 0.02
    done
fi
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "mv", "ps", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["RECORD_FILE"] = str(record_file)
    env["FIRST_STARTED"] = str(first_started)
    env["BLOCK_FILE"] = str(block_file)
    env["TZ"] = "UTC"

    first = subprocess.Popen(
        ["/bin/bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(100):
            if first_started.exists() and (fallback_lock / "owner").exists():
                break
            time.sleep(0.02)
        assert first_started.exists()
        assert (fallback_lock / "owner").exists()

        contender_env = env.copy()
        contender_env["TZ"] = "America/New_York"
        contender = subprocess.run(
            ["/bin/bash", str(scheduler)],
            cwd=repo_dir,
            env=contender_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )

        assert contender.returncode == 0, contender.stderr
        assert record_file.read_text(encoding="utf-8").splitlines() == [str(first.pid)]
    finally:
        block_file.unlink(missing_ok=True)
        _terminate_and_reap(first)


def test_scheduler_without_locked_project_python_fails_loudly(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    log_file = tmp_path / "cron.log"

    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["TZDIR"] = "/definitely/missing/system-zoneinfo"
    result = subprocess.run(
        ["bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "missing the Python executable needed for America/New_York scheduling" in result.stderr
    assert "Scheduler preflight failed with status 2" in log_file.read_text(encoding="utf-8")
    assert "missing the Python executable needed for America/New_York scheduling" in log_file.read_text(
        encoding="utf-8"
    )
    assert result.stdout == ""


def test_scheduler_logs_locked_timezone_preflight_failure(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    schedule_python = repo_dir / ".venv" / "bin" / "python"
    schedule_python.parent.mkdir(parents=True)
    schedule_python.write_text(
        """#!/bin/bash
if [[ "$2" == "-S" ]]; then
    printf '1770215400 3 14:30'
    exit 0
fi
echo 'simulated locked tzdata failure' >&2
exit 17
""",
        encoding="utf-8",
    )
    schedule_python.chmod(0o755)

    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    runner_called = tmp_path / "runner-called"
    runner = tmp_path / "runner"
    runner.write_text('#!/bin/bash\n: > "$RUNNER_CALLED"\n', encoding="utf-8")
    runner.chmod(0o755)
    log_file = tmp_path / "cron.log"

    env = os.environ.copy()
    env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["RUNNER_CALLED"] = str(runner_called)
    env["UV_BIN"] = str(fake_uv)
    result = subprocess.run(
        ["bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "simulated locked tzdata failure" in result.stderr
    assert "Unable to determine the current America/New_York schedule time" in result.stderr
    logged_failure = log_file.read_text(encoding="utf-8")
    assert "Scheduler preflight failed with status 2" in logged_failure
    assert "simulated locked tzdata failure" in logged_failure
    assert "Unable to determine the current America/New_York schedule time" in logged_failure
    assert not runner_called.exists()


@pytest.mark.parametrize("time", ["09:30", "09:31", "12:00", "15:59", "16:00"])
def test_scheduled_market_session_runs_reconciliation(time: str) -> None:
    result = _run_scheduler(f"3 {time}")

    assert result.returncode == 0
    assert result.stdout.strip() == SCHEDULED_RECONCILIATION_ARGUMENTS


@pytest.mark.parametrize("now", ["3 08:44", "3 08:46", "3 09:29", "3 16:01", "3 23:59"])
def test_scheduled_times_outside_run_windows_are_noops(now: str) -> None:
    result = _run_scheduler(now)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("now", ["6 08:45", "6 09:30", "7 16:00"])
def test_scheduled_weekends_are_noops(now: str) -> None:
    result = _run_scheduler(now)

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("now", ["", "0 09:30", "8 09:30", "3 9:30", "3 24:00", "Wednesday 09:30"])
def test_invalid_injected_schedule_time_fails(now: str) -> None:
    result = _run_scheduler(now)

    assert result.returncode == 2
    assert "Invalid scheduled" in result.stderr


def test_installer_uses_one_timezone_independent_idempotent_entry(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)

    crontab_state = tmp_path / "crontab"
    crontab_state.write_text(
        (
            "SHELL=/bin/csh\n"
            "SHELLOPTS=verbose\n"
            "BASHOPTS=failglob\n"
            "LD_PRELOAD=/trusted/libcron-preload.so\n"
            "https_proxy=http://cron-user:cron-password@trusted-proxy.invalid\n"
            "REQUESTS_CA_BUNDLE=/trusted/cron-ca.pem\n"
            "SSLKEYLOGFILE=/trusted/cron-tls.keys\n"
            "OPENSSL_CONF=/trusted/openssl.cnf\n"
            "OPENSSL_MODULES=/trusted/openssl-modules\n"
            "PATH=/unrelated/custom/bin\n"
            "MAILTO=ops@example.com\n"
        ),
        encoding="utf-8",
    )
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    for _ in range(2):
        subprocess.run(["bash", str(INSTALLER)], cwd=ROOT, env=env, check=True, capture_output=True, text=True)

    installed = crontab_state.read_text(encoding="utf-8")
    schedule_lines = [line for line in installed.splitlines() if "run-scheduled" in line]
    assert installed.count("# BEGIN leveraged-trader managed schedule") == 1
    assert installed.count("# END leveraged-trader managed schedule") == 1
    assert "MAILTO=ops@example.com" in installed
    assert "SHELL=/bin/csh" in installed
    assert "PATH=/unrelated/custom/bin" in installed
    assert "CRON_TZ=" not in installed
    assert len(schedule_lines) == 1
    assert schedule_lines[0].startswith("* * * * * ")
    managed_block = installed.split("# BEGIN leveraged-trader managed schedule", 1)[1].split(
        "# END leveraged-trader managed schedule",
        1,
    )[0]
    managed_lines = [line for line in managed_block.splitlines() if line]
    assert managed_lines[0] == "SHELL=/bin/sh"
    assert managed_lines[1:3] == ['SHELLOPTS=""', 'BASHOPTS=""']
    schedule_index = managed_lines.index(schedule_lines[0])
    assert managed_lines[3:schedule_index] == ['BASH_ENV=""', 'ENV=""'] + [
        f'{name}=""' for name in CRON_BOUNDARY_ENVIRONMENT_NAMES
    ]
    restored_lines = managed_lines[schedule_index + 1 : -3]
    assert "LD_PRELOAD=/trusted/libcron-preload.so" in restored_lines
    assert "https_proxy=http://cron-user:cron-password@trusted-proxy.invalid" in restored_lines
    assert "REQUESTS_CA_BUNDLE=/trusted/cron-ca.pem" in restored_lines
    assert "SSLKEYLOGFILE=/trusted/cron-tls.keys" in restored_lines
    assert "OPENSSL_CONF=/trusted/openssl.cnf" in restored_lines
    assert "OPENSSL_MODULES=/trusted/openssl-modules" in restored_lines
    assert managed_lines[-3:] == ["SHELL=/bin/csh", "SHELLOPTS=verbose", "BASHOPTS=failglob"]
    assert not any(line.startswith("PATH=") for line in managed_lines)
    assert "/usr/bin/env -i SHELL=/bin/sh PATH=/usr/bin:/bin" in installed
    assert "run-scheduled-clean-environment" in schedule_lines[0]
    assert "SHELL=/bin/sh PATH=/usr/bin:/bin" in installed


def test_installer_emits_portable_cron_line_from_long_checkout_path(tmp_path: Path) -> None:
    long_parent = tmp_path
    for index in range(4):
        long_parent /= f"checkout-segment-{index}-" + ("x" * 90)
    repo_dir = long_parent / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("", encoding="utf-8")
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    pending="${CRONTAB_STATE}.pending"
    cat > "$pending"
    while IFS= read -r line || [[ -n "$line" ]]; do
        (( ${#line} <= 1000 )) || exit 91
    done < "$pending"
    mv "$pending" "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    result = subprocess.run(
        ["/bin/bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    schedule_line = next(
        line for line in crontab_state.read_text(encoding="utf-8").splitlines() if line.startswith("* * * * * ")
    )
    assert len(schedule_line.encode("utf-8")) <= 1000
    assert schedule_line.startswith("* * * * * /usr/bin/env -i ")
    assert "run-scheduled-clean-environment" in schedule_line


def test_installed_bootstrap_executes_large_captured_scheduler_snapshot(tmp_path: Path) -> None:
    repo_dir, crontab_state, env = _install_test_schedule(tmp_path)
    marker = tmp_path / "captured-scheduler-executed"
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    padding = "# authenticated scheduler padding\n" * 5000
    scheduler_source = "#!/usr/bin/env bash\n" + padding + f"printf executed > {shlex.quote(str(marker))}\n"
    assert 131_072 < len(scheduler_source.encode("utf-8")) < 1_048_576
    scheduler.write_text(scheduler_source, encoding="utf-8")
    scheduler.chmod(0o755)

    result = _run_installed_schedule(repo_dir, crontab_state, env)

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8") == "executed"


def test_installed_command_rejects_tampered_private_bootstrap_before_execution(
    tmp_path: Path,
) -> None:
    repo_dir, crontab_state, env = _install_test_schedule(tmp_path)
    bootstrap_files = list(
        _installer_bootstrap_root(_installer_test_account_home(tmp_path)).glob("run-scheduled-clean-environment-*")
    )
    assert len(bootstrap_files) == 1
    marker = tmp_path / "tampered-bootstrap-executed"
    bootstrap = bootstrap_files[0]
    bootstrap.chmod(0o600)
    with bootstrap.open("a", encoding="utf-8") as installed_source:
        installed_source.write(f"\nprintf executed > {shlex.quote(str(marker))}\n")

    result = _run_installed_schedule(repo_dir, crontab_state, env)

    assert result.returncode == 2
    assert "bootstrap changed after crontab installation" in result.stderr
    assert not marker.exists()


def test_installed_command_never_executes_replaced_checkout_launcher(tmp_path: Path) -> None:
    repo_dir, crontab_state, env = _install_test_schedule(tmp_path)
    cron_dir = repo_dir / "scripts" / "cron"
    launcher_marker = tmp_path / "replacement-launcher-executed"
    scheduler_marker = tmp_path / "trusted-scheduler-executed"

    replacement = tmp_path / "replacement-clean-environment-launcher"
    replacement.write_text(
        f"#!/usr/bin/env bash\nprintf executed > {shlex.quote(str(launcher_marker))}\n",
        encoding="utf-8",
    )
    replacement.chmod(0o755)
    replacement.replace(cron_dir / "run-scheduled-clean-environment")

    scheduler = cron_dir / "run-scheduled"
    scheduler.write_text(
        f"#!/usr/bin/env bash\nprintf executed > {shlex.quote(str(scheduler_marker))}\n",
        encoding="utf-8",
    )
    scheduler.chmod(0o755)

    result = _run_installed_schedule(repo_dir, crontab_state, env)

    assert result.returncode == 0, result.stderr
    assert scheduler_marker.read_text(encoding="utf-8") == "executed"
    assert not launcher_marker.exists()


def test_installed_command_rejects_updated_runtime_security_until_reinstalled(tmp_path: Path) -> None:
    repo_dir, crontab_state, env = _install_test_schedule(tmp_path)
    cron_dir = repo_dir / "scripts" / "cron"
    scheduler_marker = tmp_path / "scheduler-executed"
    scheduler = cron_dir / "run-scheduled"
    scheduler.write_text(
        f"#!/usr/bin/env bash\nprintf executed > {shlex.quote(str(scheduler_marker))}\n",
        encoding="utf-8",
    )
    scheduler.chmod(0o755)
    runtime_security = cron_dir / "runtime-security"
    with runtime_security.open("a", encoding="utf-8") as helper:
        helper.write("\n# trusted update that requires reinstalling cron\n")

    result = _run_installed_schedule(repo_dir, crontab_state, env)

    assert result.returncode == 2
    assert "runtime security checks changed after crontab installation" in result.stderr
    assert not scheduler_marker.exists()

    reinstall = subprocess.run(
        ["/bin/bash", str(cron_dir / "install-crontab")],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert reinstall.returncode == 0, reinstall.stderr

    result = _run_installed_schedule(repo_dir, crontab_state, env)

    assert result.returncode == 0, result.stderr
    assert scheduler_marker.read_text(encoding="utf-8") == "executed"


@pytest.mark.parametrize(
    ("target_name", "unsafe_kind"),
    [
        ("run-scheduled-clean-environment", "writable-file"),
        ("run-scheduled-clean-environment", "writable-parent"),
        ("run-scheduled", "writable-file"),
        ("run-scheduled", "writable-parent"),
    ],
)
def test_installed_command_rejects_post_install_checkout_replacement(
    tmp_path: Path,
    target_name: str,
    unsafe_kind: str,
) -> None:
    repo_dir, crontab_state, env = _install_test_schedule(tmp_path)
    cron_dir = repo_dir / "scripts" / "cron"
    target = cron_dir / target_name
    marker = tmp_path / f"{target_name}-replacement-executed"
    replacement = tmp_path / f"malicious-{target_name}"
    replacement.write_text(
        f"#!/usr/bin/env bash\nprintf executed > {shlex.quote(str(marker))}\n",
        encoding="utf-8",
    )
    replacement.chmod(0o775 if unsafe_kind == "writable-file" else 0o755)
    replacement.replace(target)
    if unsafe_kind == "writable-parent":
        cron_dir.chmod(0o777)

    result = _run_installed_schedule(repo_dir, crontab_state, env)

    assert result.returncode == 2
    expected_message = (
        "must not be group- or world-writable"
        if unsafe_kind == "writable-file"
        else "replaceable through writable ancestor"
    )
    assert expected_message in result.stderr
    assert not marker.exists()


def test_installed_command_revalidates_custom_operational_bash_before_use(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    custom_bash = bin_dir / "custom-bash"
    shutil.copy2(Path("/bin/bash").resolve(), custom_bash)
    custom_bash.chmod(0o755)
    repo_dir, crontab_state, env = _install_test_schedule(
        tmp_path,
        installer_bash=custom_bash,
    )
    marker = tmp_path / "replacement-bash-executed"
    custom_bash.write_text(
        f"#!/bin/sh\nprintf executed > {shlex.quote(str(marker))}\n",
        encoding="utf-8",
    )
    custom_bash.chmod(0o775)

    result = _run_installed_schedule(repo_dir, crontab_state, env)

    assert result.returncode == 2
    assert "Bash executable must not be group- or world-writable" in result.stderr
    assert not marker.exists()


def test_installed_command_clears_openssl_configuration_before_python(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"

    openssl_config = tmp_path / "unsafe-openssl.cnf"
    openssl_config.write_text(
        "openssl_conf = openssl_init\n"
        "[openssl_init]\n"
        "ssl_conf = ssl_section\n"
        "[ssl_section]\n"
        "system_default = system_default_section\n"
        "[system_default_section]\n"
        "MaxProtocol = TLSv1.2\n",
        encoding="utf-8",
    )
    real_python = (ROOT / ".venv" / "bin" / "python").resolve()
    context_probe = "import ssl; print(int(ssl.create_default_context().maximum_version))"
    baseline = subprocess.run(
        [str(real_python), "-I", "-c", context_probe],
        check=True,
        capture_output=True,
        text=True,
    )
    configured = subprocess.run(
        [str(real_python), "-I", "-c", context_probe],
        env={**os.environ, "OPENSSL_CONF": str(openssl_config)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert configured.stdout != baseline.stdout

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text(f"OPENSSL_CONF={openssl_config}\n", encoding="utf-8")
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    subprocess.run(
        ["bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    probe_output = tmp_path / "scheduled-openssl-context"
    scheduled_probe = (
        'import os, ssl; print(os.environ.get("OPENSSL_CONF", "<unset>")); '
        "print(int(ssl.create_default_context().maximum_version))"
    )
    scheduler.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"exec {shlex.quote(str(real_python))} -I -c {shlex.quote(scheduled_probe)} "
        f"> {shlex.quote(str(probe_output))}\n",
        encoding="utf-8",
    )
    scheduler.chmod(0o755)

    installed_lines = crontab_state.read_text(encoding="utf-8").splitlines()
    schedule_line = next(line for line in installed_lines if line.startswith("* * * * * "))
    cron_command = _cron_shell_command(schedule_line)
    cron_env = env.copy()
    for assignment in installed_lines[: installed_lines.index(schedule_line)]:
        if "=" in assignment and not assignment.startswith("#"):
            name, value = assignment.split("=", 1)
            cron_env[name] = _cron_assignment_value(value)
    subprocess.run(["/bin/sh", "-c", cron_command], cwd=repo_dir, env=cron_env, check=True)

    environment_value, maximum_version = probe_output.read_text(encoding="utf-8").splitlines()
    assert environment_value == "<unset>"
    assert int(maximum_version) == int(baseline.stdout)


def test_installer_default_lock_namespace_uses_account_home_for_different_home_values(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    source = installer.read_text(encoding="utf-8")
    lock_call = 'secure_private_directory "$crontab_install_lock_root" "crontab installer lock" || exit $?'
    capture_and_exit = """printf '%s\\n' "$crontab_install_lock_root" > "$LOCK_PATH_CAPTURE"
exit 0"""
    assert source.count(lock_call) == 1
    installer.write_text(source.replace(lock_call, capture_and_exit, 1), encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    exit 0
fi
exit 2
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    selected_lock_roots = []
    for index in range(2):
        mutable_home = tmp_path / f"mutable-home-{index}"
        mutable_home.mkdir()
        capture = tmp_path / f"selected-lock-{index}"
        env = os.environ.copy()
        env.pop(INSTALLER_TEST_MODE_ENV, None)
        env.pop(INSTALLER_TEST_HOME_ENV, None)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["HOME"] = str(mutable_home)
        env["LOCK_PATH_CAPTURE"] = str(capture)
        result = subprocess.run(
            ["bash", str(installer)],
            cwd=repo_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        selected_lock_roots.append(Path(capture.read_text(encoding="utf-8").strip()))

    account_home = Path(pwd.getpwuid(os.geteuid()).pw_dir).resolve()
    expected_lock_root = _installer_lock_root(account_home)
    assert selected_lock_roots == [expected_lock_root, expected_lock_root]


def test_installer_records_absolute_external_uv_when_function_shadows_relative_path(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    tools_dir = repo_dir / "tools"
    tools_dir.mkdir()
    fake_uv_target = tools_dir / "uv-target"
    fake_uv_target.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv_target.chmod(0o755)
    fake_uv = tools_dir / "uv"
    fake_uv.symlink_to(fake_uv_target.name)

    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("", encoding="utf-8")
    fake_crontab = tools_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"tools:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'uv() { return 0; }; source "$1"',
            "installer-test",
            str(installer),
        ],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    installed = crontab_state.read_text(encoding="utf-8")
    schedule_line = next(line for line in installed.splitlines() if line.startswith("* * * * * "))
    assert f"'{fake_uv_target}'" in schedule_line
    assert "UV_BIN='uv'" not in schedule_line
    assert "UV_BIN='tools/uv'" not in schedule_line


@pytest.mark.parametrize(
    ("unsafe_kind", "expected_message"),
    [
        ("writable-file", "must not be group- or world-writable"),
        ("writable-parent", "replaceable through writable ancestor"),
    ],
)
def test_installer_rejects_replaceable_uv_executable(
    tmp_path: Path,
    unsafe_kind: str,
    expected_message: str,
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    bin_dir = tmp_path / "unsafe-bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o775 if unsafe_kind == "writable-file" else 0o755)
    if unsafe_kind == "writable-parent":
        bin_dir.chmod(0o777)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert expected_message in result.stderr
    assert not (repo_dir / "outputs").exists()


def test_installer_accepts_explicit_no_crontab_result(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)

    crontab_state = tmp_path / "crontab"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    echo "no crontab for test-user" >&2
    exit 1
elif [[ "${1:-}" == "-" ]]; then
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    subprocess.run(["bash", str(INSTALLER)], cwd=ROOT, env=env, check=True, capture_output=True, text=True)

    installed = crontab_state.read_text(encoding="utf-8")
    assert "# BEGIN leveraged-trader managed schedule" in installed
    assert "run-scheduled" in installed


def test_installer_read_error_does_not_write_crontab(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)

    write_marker = tmp_path / "write attempted"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    echo "crontab: spool temporarily unavailable" >&2
    exit 2
elif [[ "${1:-}" == "-" ]]; then
    : > "$CRONTAB_WRITE_MARKER"
    cat >/dev/null
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_WRITE_MARKER"] = str(write_marker)
    result = subprocess.run(
        ["bash", str(INSTALLER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "no changes were made" in result.stderr
    assert not write_marker.exists()


@pytest.mark.parametrize(
    "change_after_read",
    [1, 2],
    ids=["before-lock-recheck", "before-final-recheck"],
)
def test_installer_aborts_if_crontab_changes_during_preparation(
    tmp_path: Path,
    change_after_read: int,
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)

    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=initial@example.com\n", encoding="utf-8")
    read_count = tmp_path / "crontab-read-count"
    write_marker = tmp_path / "write-attempted"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    count=0
    if [[ -f "$CRONTAB_READ_COUNT" ]]; then
        count="$(<"$CRONTAB_READ_COUNT")"
    fi
    count=$((count + 1))
    printf '%s\n' "$count" > "$CRONTAB_READ_COUNT"
    cat "$CRONTAB_STATE"
    if (( count == CRONTAB_CHANGE_AFTER_READ )); then
        printf 'MAILTO=concurrent@example.com\n' > "$CRONTAB_STATE"
    fi
elif [[ "${1:-}" == "-" ]]; then
    : > "$CRONTAB_WRITE_MARKER"
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["CRONTAB_READ_COUNT"] = str(read_count)
    env["CRONTAB_CHANGE_AFTER_READ"] = str(change_after_read)
    env["CRONTAB_WRITE_MARKER"] = str(write_marker)
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 75
    assert "crontab changed while the installer was preparing" in result.stderr
    assert crontab_state.read_text(encoding="utf-8") == "MAILTO=concurrent@example.com\n"
    assert not write_marker.exists()
    assert not (_installer_lock_root(_installer_test_account_home(tmp_path)) / "active.lock").exists()


def test_installer_refuses_an_active_cooperating_installer_lock(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    lock_root = _installer_lock_root(_installer_test_account_home(tmp_path))
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_root.chmod(0o700)
    lock_dir = lock_root / "active.lock"
    if lock_dir.exists() or lock_dir.is_symlink():
        pytest.skip("another installer owns the process-global test lock")
    lock_dir.mkdir(mode=0o700)
    owner_file = lock_dir / "owner"
    owner_file.write_text(f"{os.getpid()} {_process_start_identity(os.getpid())}\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    write_marker = tmp_path / "write-attempted"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    : > "$CRONTAB_WRITE_MARKER"
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["CRONTAB_WRITE_MARKER"] = str(write_marker)
    try:
        result = subprocess.run(
            ["bash", str(installer)],
            cwd=repo_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        owner_file.unlink(missing_ok=True)
        lock_dir.rmdir()

    assert result.returncode == 75
    assert "installation is already in progress" in result.stderr
    assert crontab_state.read_text(encoding="utf-8") == "MAILTO=ops@example.com\n"
    assert not write_marker.exists()
    assert not lock_dir.exists()


def test_installer_recovers_a_stale_global_lock(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    lock_root = _installer_lock_root(_installer_test_account_home(tmp_path))
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_root.chmod(0o700)
    lock_dir = lock_root / "active.lock"
    if lock_dir.exists() or lock_dir.is_symlink():
        pytest.skip("another installer owns the process-global test lock")
    lock_dir.mkdir(mode=0o700)
    (lock_dir / "owner").write_text("99999999 proc:0\n", encoding="utf-8")
    (lock_dir / "owner.pending").write_text("interrupted publication\n", encoding="utf-8")
    recovery_marker = lock_dir / ".stale-recovery"
    recovery_marker.mkdir(mode=0o700)
    (recovery_marker / "owner.pending").write_text(
        "interrupted recovery publication\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "# BEGIN leveraged-trader managed schedule" in crontab_state.read_text(encoding="utf-8")
    # The lock-owning shell execs crontab, so no child can outlive the owner.
    # Successful writers intentionally leave dead-owner metadata for the next
    # invocation to recover.
    assert (lock_dir / "owner").is_file()


@pytest.mark.parametrize(
    "abandoned_state",
    [
        ("gate", "owner", "99999999 proc:0\n"),
        ("gate", "owner", f"{os.getpid()} proc:0\n"),
        ("gate", None, None),
        ("gate", "owner", "99999999"),
        ("gate", "owner.pending", "99999999"),
        ("claim", "owner", "99999999"),
        ("claim", "owner.pending", "99999999"),
    ],
    ids=[
        "dead-owner",
        "reused-pid",
        "expired-ownerless",
        "partial-owner",
        "partial-owner-pending",
        "partial-claim-owner",
        "partial-claim-owner-pending",
    ],
)
def test_installer_recovers_abandoned_acquisition_gate(
    tmp_path: Path,
    abandoned_state: tuple[str, str | None, str | None],
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    lock_root = _installer_lock_root(_installer_test_account_home(tmp_path))
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_root.chmod(0o700)
    gate_dir = lock_root / "acquisition.gate"
    artifact_kind, metadata_name, metadata = abandoned_state
    abandoned_dir = gate_dir if artifact_kind == "gate" else Path(f"{gate_dir}.recovery.crashed")
    abandoned_dir.mkdir(mode=0o700)
    if metadata_name is not None:
        (abandoned_dir / metadata_name).write_text(metadata or "", encoding="utf-8")
    old_timestamp = time.time() - 300
    os.utime(abandoned_dir, (old_timestamp, old_timestamp))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    write_marker = tmp_path / "write-attempted"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    : > "$CRONTAB_WRITE_MARKER"
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CRONTAB_STATE"] = str(crontab_state)
    env["CRONTAB_WRITE_MARKER"] = str(write_marker)
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not gate_dir.exists()
    assert not abandoned_dir.exists()
    assert "# BEGIN leveraged-trader managed schedule" in crontab_state.read_text(encoding="utf-8")
    assert write_marker.is_file()


def test_installer_stale_recovery_blocks_path_replacement_and_third_contender(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    lock_root = _installer_lock_root(_installer_test_account_home(tmp_path))
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_root.chmod(0o700)
    lock_dir = lock_root / "active.lock"
    lock_dir.mkdir(mode=0o700)
    (lock_dir / "owner").write_text("99999999 proc:0\n", encoding="utf-8")
    inspected_lock = lock_root / "inspected-stale.lock"
    swap_marker = tmp_path / "installer-lock-swapped"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    write_marker = tmp_path / "write-attempted"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    : > "$CRONTAB_WRITE_MARKER"
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    real_mkdir = shutil.which("mkdir")
    real_mv = shutil.which("mv")
    real_rm = shutil.which("rm")
    assert real_mkdir is not None and real_mv is not None and real_rm is not None
    live_owner = f"{os.getpid()} {_process_start_identity(os.getpid())}\n"
    fake_rm = bin_dir / "rm"
    fake_rm.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$PWD" == "$LOCK_DIR" && "$*" == *"./owner"* && ! -e "$SWAP_MARKER" ]]; then
    "$REAL_MV" "$LOCK_DIR" "$INSPECTED_LOCK"
    "$REAL_MKDIR" -m 700 "$LOCK_DIR"
    printf '%s' "$LIVE_OWNER" > "$LOCK_DIR/owner"
    : > "$SWAP_MARKER"
fi
exec "$REAL_RM" "$@"
""",
        encoding="utf-8",
    )
    fake_rm.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["CRONTAB_WRITE_MARKER"] = str(write_marker)
    env["LOCK_DIR"] = str(lock_dir)
    env["INSPECTED_LOCK"] = str(inspected_lock)
    env["SWAP_MARKER"] = str(swap_marker)
    env["LIVE_OWNER"] = live_owner
    env["REAL_MKDIR"] = real_mkdir
    env["REAL_MV"] = real_mv
    env["REAL_RM"] = real_rm
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    third_contender = subprocess.run(
        ["bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 75
    assert third_contender.returncode == 75
    assert "installation is already in progress" in result.stderr
    assert "installation is already in progress" in third_contender.stderr
    assert swap_marker.is_file()
    assert (lock_dir / "owner").read_text(encoding="utf-8") == live_owner
    assert not (lock_dir / ".stale-recovery").exists()
    assert not (inspected_lock / "owner").exists()
    assert crontab_state.read_text(encoding="utf-8") == "MAILTO=ops@example.com\n"
    assert not write_marker.exists()
    assert 'mv "$crontab_install_lock_dir"' not in installer.read_text(encoding="utf-8")


def test_installer_ignores_a_hostile_legacy_tmp_lock(tmp_path: Path) -> None:
    legacy_lock_root = _legacy_installer_lock_root()
    try:
        legacy_lock_root.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        pytest.skip(f"cannot prepare the legacy process-global lock root: {exc}")
    if legacy_lock_root.is_symlink() or legacy_lock_root.stat().st_uid != os.geteuid():
        pytest.skip("the legacy process-global lock root is not controlled by this test user")
    legacy_lock_root.chmod(0o700)
    legacy_lock_dir = legacy_lock_root / "active.lock"
    if legacy_lock_dir.exists() or legacy_lock_dir.is_symlink():
        pytest.skip("another test owns the legacy process-global lock")
    legacy_lock_dir.mkdir(mode=0o700)
    legacy_owner = legacy_lock_dir / "owner"
    legacy_owner_contents = f"{os.getpid()} {_process_start_identity(os.getpid())}\n"
    legacy_owner.write_text(legacy_owner_contents, encoding="utf-8")

    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)

    try:
        result = subprocess.run(
            ["bash", str(installer)],
            cwd=repo_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        legacy_owner.unlink(missing_ok=True)
        legacy_lock_dir.rmdir()

    assert result.returncode == 0, result.stderr
    assert "# BEGIN leveraged-trader managed schedule" in crontab_state.read_text(encoding="utf-8")
    assert (_installer_lock_root(_installer_test_account_home(tmp_path)) / "active.lock" / "owner").is_file()


@pytest.mark.parametrize(
    "existing",
    [
        "# BEGIN leveraged-trader managed schedule\n",
        "# END leveraged-trader managed schedule\n",
        "# END leveraged-trader managed schedule\n# BEGIN leveraged-trader managed schedule\n",
        (
            "# BEGIN leveraged-trader managed schedule\n"
            "# BEGIN leveraged-trader managed schedule\n"
            "# END leveraged-trader managed schedule\n"
            "# END leveraged-trader managed schedule\n"
        ),
        (
            "# BEGIN leveraged-trader managed schedule\n"
            "# END leveraged-trader managed schedule\n"
            "# BEGIN leveraged-trader managed schedule\n"
            "# END leveraged-trader managed schedule\n"
        ),
    ],
    ids=["orphaned-start", "orphaned-end", "reversed", "nested", "duplicate"],
)
def test_installer_rejects_malformed_managed_markers_without_writing(tmp_path: Path, existing: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)

    crontab_state = tmp_path / "crontab"
    crontab_state.write_text(existing, encoding="utf-8")
    write_marker = tmp_path / "write attempted"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    : > "$CRONTAB_WRITE_MARKER"
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["CRONTAB_WRITE_MARKER"] = str(write_marker)
    result = subprocess.run(
        ["bash", str(INSTALLER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Invalid existing leveraged-trader crontab markers" in result.stderr
    assert crontab_state.read_text(encoding="utf-8") == existing
    assert not write_marker.exists()


def test_installer_rejects_nonsticky_checkout_parent_before_creating_outputs_or_crontab(tmp_path: Path) -> None:
    shared_directory = tmp_path / "shared"
    shared_directory.mkdir(mode=0o777)
    shared_directory.chmod(0o777)
    repo_dir = shared_directory / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    write_marker = tmp_path / "crontab-write-attempted"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    : > "$CRONTAB_WRITE_MARKER"
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["CRONTAB_WRITE_MARKER"] = str(write_marker)
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "replaceable through writable ancestor" in result.stderr
    assert not (repo_dir / "outputs").exists()
    assert crontab_state.read_text(encoding="utf-8") == "MAILTO=ops@example.com\n"
    assert not write_marker.exists()


def test_installer_rejects_intermediate_symlink_invocation_before_outputs_or_crontab_write(
    tmp_path: Path,
) -> None:
    safe_directory = tmp_path / "safe"
    repo_dir = safe_directory / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    shared_directory = tmp_path / "shared"
    shared_directory.mkdir(mode=0o777)
    shared_directory.chmod(0o777)
    redirect = shared_directory / "redirect"
    redirect.symlink_to(safe_directory, target_is_directory=True)
    installer = redirect / "repo" / "scripts" / "cron" / "install-crontab"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    write_marker = tmp_path / "crontab-write-attempted"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    : > "$CRONTAB_WRITE_MARKER"
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["CRONTAB_WRITE_MARKER"] = str(write_marker)
    result = subprocess.run(
        ["bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "ancestor must not be a symbolic link" in result.stderr
    assert not (repo_dir / "outputs").exists()
    assert crontab_state.read_text(encoding="utf-8") == "MAILTO=ops@example.com\n"
    assert not write_marker.exists()


def test_installer_rejects_parent_component_in_raw_script_invocation(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer_with_parent_component = repo_dir / "scripts" / "cron" / ".." / "cron" / "install-crontab"

    result = subprocess.run(
        ["bash", str(installer_with_parent_component)],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must not contain '..' components" in result.stderr
    assert not (repo_dir / "outputs").exists()


def test_installer_rejects_control_character_in_script_path_before_outputs(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo\nunsafe"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"

    result = subprocess.run(
        ["bash", str(installer)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "script path must not contain ASCII control characters" in result.stderr
    assert not (repo_dir / "outputs").exists()


def test_scheduler_rejects_control_character_in_initial_working_directory(tmp_path: Path) -> None:
    working_directory = tmp_path / "cwd\nunsafe"
    working_directory.mkdir()
    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"

    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=working_directory,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "initial working directory must not contain ASCII control characters" in result.stderr


def test_installed_command_handles_equals_spaces_and_percent_signs_in_paths(tmp_path: Path) -> None:
    repo_dir = tmp_path / r"repo=equals 'quoted' ✓ with slash\%100 coverage"
    assert repo_dir.name.endswith(r"slash\%100 coverage")
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"

    bin_dir = tmp_path / r"tool 'bin' ✓ slash\%50 special"
    assert bin_dir.name.endswith(r"slash\%50 special")
    bin_dir.mkdir()
    custom_bash = bin_dir / "private Bash outside managed PATH"
    bash_as_sh = bin_dir / "sh"
    system_bash = shutil.which("bash")
    assert system_bash is not None
    custom_bash.symlink_to(system_bash)
    bash_as_sh.symlink_to(system_bash)
    uv_capture_path = tmp_path / "uv capture.txt"
    cli_capture_path = tmp_path / "cli capture.txt"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        f'printf \'%s\\n\' "$0" "$UV_PROJECT_ENVIRONMENT" "$@" > {shlex.quote(str(uv_capture_path))}\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    fake_python = repo_dir / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        f'#!/usr/bin/env bash\nprintf \'%s\\n\' "$0" "$@" > {shlex.quote(str(cli_capture_path))}\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    crontab_state = tmp_path / "installed crontab"
    crontab_state.write_text("", encoding="utf-8")
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["LC_ALL"] = "C"
    env["CRONTAB_STATE"] = str(crontab_state)
    env["TEST_UV_CAPTURE"] = str(uv_capture_path)
    env["CLI_CAPTURE"] = str(cli_capture_path)
    env["EXPECTED_BASH"] = str(Path(system_bash).resolve())
    subprocess.run(
        [str(custom_bash), str(installer)],
        cwd=repo_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    # Replace the copied scheduler with a deterministic due-command probe. The
    # installed command still has to clear its two test hooks, supply the safe
    # managed PATH, traverse percent/space-bearing paths, and reach the wrapper.
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    runner = repo_dir / "scripts" / "cron" / "run-leveraged-trader"
    scheduler.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        f'test "$BASH" = {shlex.quote(str(Path(system_bash).resolve()))}\n'
        'test "${LEVERAGED_TRADER_SCHEDULE_NOW+set}" != set\n'
        'test "${LEVERAGED_TRADER_RUNNER+set}" != set\n'
        'test "${BASH_ENV+set}" != set\n'
        'test "${ENV+set}" != set\n'
        'case ":$SHELLOPTS:" in *:noexec:*) exit 97 ;; esac\n'
        'case ":$BASHOPTS:" in *:failglob:*) exit 98 ;; esac\n'
        'test "${TZDIR+set}" != set\n'
        'test "${PYTHONTZPATH+set}" != set\n'
        'test "${UV_PROJECT+set}" != set\n'
        'test "${UV_WORKING_DIR+set}" != set\n'
        'test "${UV_CONFIG_FILE+set}" != set\n'
        'test "${UV_NO_DEV+set}" != set\n'
        'test "${UV_NO_GROUP+set}" != set\n'
        'test "${UV_NO_DEFAULT_GROUPS+set}" != set\n'
        'test "${UV_FROZEN+set}" != set\n'
        'test "${UV_PYTHON+set}" != set\n'
        'test "${UV_NO_EDITABLE+set}" != set\n'
        + "".join(
            f'test "${{{environment_name}+set}}" != set\n' for environment_name in CRON_BOUNDARY_ENVIRONMENT_NAMES
        )
        + 'test "$SHELL" = /bin/sh\n'
        + 'test "$PATH" = /usr/bin:/bin\n'
        + f"exec {shlex.quote(str(runner))} "
        "--require-workflow-source-success --alpaca-submit-buy-orders --alpaca-submit-sell-orders\n",
        encoding="utf-8",
    )
    scheduler.chmod(0o755)

    installed_lines = crontab_state.read_text(encoding="utf-8").splitlines()
    schedule_line = next(line for line in installed_lines if line.startswith("* * * * * "))
    assert "repo=equals" not in schedule_line
    bootstrap_files = list(
        _installer_bootstrap_root(_installer_test_account_home(tmp_path)).glob("run-scheduled-clean-environment-*")
    )
    assert len(bootstrap_files) == 1
    assert "repo=equals" in bootstrap_files[0].read_text(encoding="utf-8")
    assert not any(line.startswith("UV_BIN=") for line in installed_lines)
    assert "\\%" in schedule_line
    assert "%" not in schedule_line.replace("\\%", "")
    assert (repo_dir / "outputs").stat().st_mode & 0o777 == 0o700
    managed_block = (
        crontab_state.read_text(encoding="utf-8")
        .split(
            "# BEGIN leveraged-trader managed schedule",
            1,
        )[1]
        .split("# END leveraged-trader managed schedule", 1)[0]
    )
    managed_lines = [line for line in managed_block.splitlines() if line]
    assert "SHELL=/bin/sh\n" in managed_block
    assert 'SHELLOPTS=""\n' in managed_block
    assert 'BASHOPTS=""\n' in managed_block
    assert not any(line.startswith("PATH=") for line in managed_block.splitlines())
    schedule_index = managed_lines.index(schedule_line)
    assert managed_lines[3:schedule_index] == ['BASH_ENV=""', 'ENV=""'] + [
        f'{name}=""' for name in CRON_BOUNDARY_ENVIRONMENT_NAMES
    ]

    # Emulate cron's backslash-parity scan before handing the command to
    # /bin/sh. A full scheduled run must use the exact uv executable for a
    # read-only locked-environment check, then exec the isolated project Python.
    cron_command = _cron_shell_command(schedule_line)
    # These inherited test hooks must not be able to alter an installed
    # production command; the deterministic probe above verifies they are gone.
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "7 23:59"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/false"
    startup_marker = tmp_path / "inherited startup executed"
    startup_file = tmp_path / "inherited shell startup"
    startup_file.write_text(': > "$STARTUP_MARKER"\n', encoding="utf-8")
    env["STARTUP_MARKER"] = str(startup_marker)
    env["BASH_ENV"] = str(startup_file)
    env["ENV"] = str(startup_file)
    env["SHELLOPTS"] = "noexec"
    env["BASHOPTS"] = "failglob"
    env["TZDIR"] = "/definitely/missing/system-zoneinfo"
    env["PYTHONTZPATH"] = "/untrusted/inherited-zoneinfo"
    env["UV_PROJECT"] = "/untrusted/project"
    env["UV_WORKING_DIR"] = "/untrusted/working-directory"
    env["UV_CONFIG_FILE"] = "/untrusted/uv.toml"
    env["UV_NO_DEV"] = "1"
    env["UV_NO_GROUP"] = "dev"
    env["UV_NO_DEFAULT_GROUPS"] = "1"
    env["UV_FROZEN"] = "1"
    env["UV_PYTHON"] = "/bin/false"
    env["UV_NO_EDITABLE"] = "1"
    for environment_name in CRON_BOUNDARY_ENVIRONMENT_NAMES:
        env[environment_name] = "/untrusted/inherited-environment-value"
    env["PATH"] = "/usr/bin:/bin"
    # Apply the positional crontab assignments that precede the schedule line,
    # then emulate a platform where /bin/sh is Bash.  Without the empty
    # SHELLOPTS assignment, inherited `noexec` skips the whole command and exits
    # successfully before `/usr/bin/env -i` can run.
    cron_env = env.copy()
    for assignment in managed_lines[: managed_lines.index(schedule_line)]:
        name, value = assignment.split("=", 1)
        cron_env[name] = _cron_assignment_value(value)
    subprocess.run([str(bash_as_sh), "-c", cron_command], cwd=repo_dir, env=cron_env, check=True)
    assert not startup_marker.exists()

    assert uv_capture_path.read_text(encoding="utf-8").splitlines() == [
        str(fake_uv),
        str(repo_dir / ".venv"),
        "sync",
        "--project",
        str(repo_dir),
        "--directory",
        str(repo_dir),
        "--locked",
        "--check",
    ]
    assert cli_capture_path.read_text(encoding="utf-8").splitlines() == [
        str(fake_python),
        "-I",
        "-m",
        "leveraged_trader",
        "--require-workflow-source-success",
        "--alpaca-submit-buy-orders",
        "--alpaca-submit-sell-orders",
    ]


def test_cron_scripts_are_executable() -> None:
    for script in (INSTALLER, SCHEDULER, SCHEDULER_CLEAN_ENVIRONMENT, RUNNER):
        assert script.is_file()
        assert script.stat().st_mode & 0o100


def test_runner_reports_missing_locked_project_environment(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    fake_uv = tmp_path / "uv"
    uv_called = tmp_path / "uv-called"
    fake_uv.write_text(
        '#!/bin/bash\nprintf called > "$TEST_UV_CALLED"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env["UV_BIN"] = str(fake_uv)
    env["TEST_UV_CALLED"] = str(uv_called)
    result = subprocess.run(
        [str(repo_dir / "scripts" / "cron" / "run-leveraged-trader"), "--help"],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 127
    assert "Locked project environment is missing the Python executable" in result.stderr
    assert "uv sync --locked" in result.stderr
    assert not uv_called.exists()


def test_runner_revalidates_uv_executable_before_use(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o775)

    env = os.environ.copy()
    env["UV_BIN"] = str(fake_uv)
    result = subprocess.run(
        [str(repo_dir / "scripts" / "cron" / "run-leveraged-trader"), "--help"],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must not be group- or world-writable" in result.stderr


@pytest.mark.parametrize(
    "entrypoint_name",
    ["install-crontab", "run-scheduled", "run-leveraged-trader"],
)
@pytest.mark.parametrize(
    ("unsafe_kind", "expected_message"),
    [
        ("writable-helper", "must not be group- or world-writable"),
        ("writable-helper-parent", "replaceable through writable ancestor"),
        ("hardlinked-helper", "must not have multiple hard links"),
    ],
)
def test_cron_entrypoints_reject_untrusted_runtime_security_before_sourcing(
    tmp_path: Path,
    entrypoint_name: str,
    unsafe_kind: str,
    expected_message: str,
) -> None:
    repo_dir = tmp_path / "repo"
    cron_dir = repo_dir / "scripts" / "cron"
    shutil.copytree(ROOT / "scripts" / "cron", cron_dir)
    runtime_security = cron_dir / "runtime-security"
    executed_marker = tmp_path / "runtime-security-executed"
    with runtime_security.open("a", encoding="utf-8") as helper:
        helper.write('\n: > "$RUNTIME_SECURITY_EXECUTED"\n')
    if unsafe_kind == "writable-helper":
        runtime_security.chmod(0o664)
    elif unsafe_kind == "writable-helper-parent":
        cron_dir.chmod(0o777)
    else:
        os.link(runtime_security, tmp_path / "runtime-security-alias")

    env = os.environ.copy()
    env["RUNTIME_SECURITY_EXECUTED"] = str(executed_marker)
    result = subprocess.run(
        ["/bin/bash", str(cron_dir / entrypoint_name)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert expected_message in result.stderr
    assert not executed_marker.exists()


@pytest.mark.parametrize("entrypoint_name", ["run-scheduled", "run-leveraged-trader"])
@pytest.mark.parametrize(
    ("unsafe_kind", "expected_message"),
    [
        ("writable-python", "must not be group- or world-writable"),
        ("writable-python-parent", "replaceable through writable ancestor"),
        ("writable-intermediate-hop-parent", "replaceable through writable ancestor"),
    ],
)
def test_scheduled_entrypoints_reject_untrusted_project_python_before_uv_or_execution(
    tmp_path: Path,
    entrypoint_name: str,
    unsafe_kind: str,
    expected_message: str,
) -> None:
    repo_dir = tmp_path / "repo"
    cron_dir = repo_dir / "scripts" / "cron"
    shutil.copytree(ROOT / "scripts" / "cron", cron_dir)
    python_entry = repo_dir / ".venv" / "bin" / "python"
    python_entry.parent.mkdir(parents=True)
    python_marker = tmp_path / "python-executed"
    python_body = '#!/bin/bash\n: > "$PYTHON_EXECUTED"\n'

    if unsafe_kind == "writable-intermediate-hop-parent":
        safe_hop_dir = tmp_path / "safe-hop"
        safe_hop_dir.mkdir()
        first_hop = safe_hop_dir / "python-hop"
        unsafe_target_dir = tmp_path / "unsafe-target"
        unsafe_target_dir.mkdir()
        target = unsafe_target_dir / "python-real"
        target.write_text(python_body, encoding="utf-8")
        target.chmod(0o755)
        first_hop.symlink_to(target)
        python_entry.symlink_to(first_hop)
        unsafe_target_dir.chmod(0o777)
    else:
        python_entry.write_text(python_body, encoding="utf-8")
        python_entry.chmod(0o775 if unsafe_kind == "writable-python" else 0o755)
        if unsafe_kind == "writable-python-parent":
            python_entry.parent.chmod(0o777)

    uv_marker = tmp_path / "uv-executed"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/bin/bash\n: > "$UV_EXECUTED"\n', encoding="utf-8")
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env["UV_BIN"] = str(fake_uv)
    env["UV_EXECUTED"] = str(uv_marker)
    env["PYTHON_EXECUTED"] = str(python_marker)
    env.pop("LEVERAGED_TRADER_LOG_FILE", None)
    if entrypoint_name == "run-scheduled":
        env.pop("LEVERAGED_TRADER_SCHEDULE_NOW", None)
        env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"

    result = subprocess.run(
        ["/bin/bash", str(cron_dir / entrypoint_name)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert expected_message in result.stderr
    assert "project Python executable" in result.stderr
    assert not uv_marker.exists()
    assert not python_marker.exists()


def test_trusted_python_entry_preserves_virtual_environment_prefix(tmp_path: Path) -> None:
    virtual_environment = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(virtual_environment)],
        check=True,
        capture_output=True,
        text=True,
    )
    python_entry = virtual_environment / "bin" / "python"
    probe = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'builtin source "$1"; '
            'validated=$(runtime_security_validate_trusted_executable_entry "$2" '
            '"project Python executable"); '
            "printf '%s\\n' \"$validated\"; "
            "\"$validated\" -I -c 'import sys; print(sys.executable); print(sys.prefix)'",
            "venv-validation",
            str(ROOT / "scripts" / "cron" / "runtime-security"),
            str(python_entry),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr
    validated, executable, prefix = probe.stdout.splitlines()
    assert validated == str(python_entry)
    assert executable == str(python_entry)
    assert prefix == str(virtual_environment)


def test_executable_validation_rejects_newline_at_end_of_symlink_target(
    tmp_path: Path,
) -> None:
    executable_directory = tmp_path / "executables"
    executable_directory.mkdir()
    safe_target = executable_directory / "python"
    unsafe_target = executable_directory / "python\n"
    for target in (safe_target, unsafe_target):
        target.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        target.chmod(0o755)
    entry = executable_directory / "entry"
    entry.symlink_to(unsafe_target.name)

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'builtin source "$1"; runtime_security_physical_executable_path "$2" "test executable"',
            "executable-validation",
            str(ROOT / "scripts" / "cron" / "runtime-security"),
            str(entry),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "symbolic-link targets must not contain ASCII control characters" in result.stderr


def test_runner_revalidates_project_python_after_successful_uv_check(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    cron_dir = repo_dir / "scripts" / "cron"
    shutil.copytree(ROOT / "scripts" / "cron", cron_dir)
    python_entry = repo_dir / ".venv" / "bin" / "python"
    python_entry.parent.mkdir(parents=True)
    python_marker = tmp_path / "python-executed"
    python_entry.write_text('#!/bin/bash\n: > "$PYTHON_EXECUTED"\n', encoding="utf-8")
    python_entry.chmod(0o755)

    uv_marker = tmp_path / "uv-executed"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/bin/bash\n: > "$TEST_UV_EXECUTED"\nchmod 775 "$PROJECT_PYTHON"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env["UV_BIN"] = str(fake_uv)
    env["TEST_UV_EXECUTED"] = str(uv_marker)
    env["PROJECT_PYTHON"] = str(python_entry)
    env["PYTHON_EXECUTED"] = str(python_marker)
    result = subprocess.run(
        ["/bin/bash", str(cron_dir / "run-leveraged-trader")],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "project Python executable must not be group- or world-writable" in result.stderr
    assert uv_marker.is_file()
    assert not python_marker.exists()


def _validate_fixture_import_tree(
    *,
    repo_dir: Path,
    project_environment: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            'builtin source "$1"; runtime_security_validate_leveraged_trader_import_paths "$2" "$3" "$4"',
            "import-tree-validation",
            str(ROOT / "scripts" / "cron" / "runtime-security"),
            str(repo_dir),
            str(project_environment),
            sys.executable,
        ],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "relative_site_packages",
    [
        "lib/python3.12/dist-packages",
        "lib/python3/dist-packages",
        "lib64/python3.12/site-packages",
        "lib64/python3.12/dist-packages",
        "local/lib/python3.12/site-packages",
        "local/lib/python3.12/dist-packages",
        "local/lib64/python3.12/site-packages",
        "local/lib64/python3.12/dist-packages",
    ],
)
def test_import_validation_covers_all_virtual_environment_package_layouts(
    tmp_path: Path,
    relative_site_packages: str,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    site_packages = project_environment / relative_site_packages
    package_dir = site_packages / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    unsafe_dependency = site_packages / "unsafe_dependency.py"
    unsafe_dependency.write_text("VALUE = 1\n", encoding="utf-8")
    unsafe_dependency.chmod(0o664)
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert "project site-packages tree" in result.stderr
    assert "must not be group- or world-writable" in result.stderr


@pytest.mark.parametrize("relative_alias", ["lib64", "local/lib64"])
def test_import_validation_accepts_standard_lib64_aliases(
    tmp_path: Path,
    relative_alias: str,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    if relative_alias.startswith("local/"):
        lib_directory = project_environment / "local" / "lib"
    else:
        lib_directory = project_environment / "lib"
    site_packages = lib_directory / "python3.12" / "site-packages"
    package_dir = site_packages / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    alias_path = project_environment / relative_alias
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    alias_path.symlink_to("lib", target_is_directory=True)
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 0, result.stderr


def test_import_validation_rejects_newline_at_end_of_lib64_alias_target(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    (project_environment / "lib").mkdir(parents=True)
    (project_environment / "lib\n").mkdir()
    (project_environment / "lib64").symlink_to("lib\n", target_is_directory=True)

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert "package-layout aliases must not contain ASCII control characters" in result.stderr


def test_import_validation_rejects_replaceable_lib64_alias(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    (project_environment / "lib").mkdir(parents=True)
    (project_environment / "lib64").symlink_to("lib", target_is_directory=True)
    project_environment.chmod(0o777)
    try:
        result = _validate_fixture_import_tree(
            repo_dir=repo_dir,
            project_environment=project_environment,
        )
    finally:
        project_environment.chmod(0o755)

    assert result.returncode == 2
    assert "replaceable through writable ancestor" in result.stderr


@pytest.mark.parametrize("target_exists", [True, False])
def test_import_validation_checks_pth_path_targets(
    tmp_path: Path,
    target_exists: bool,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    site_packages = project_environment / "lib" / "python3.12" / "site-packages"
    package_dir = site_packages / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    external_import_root = tmp_path / "external-import-root"
    if target_exists:
        external_import_root.mkdir()
        unsafe_module = external_import_root / "unsafe_module.py"
        unsafe_module.write_text("VALUE = 1\n", encoding="utf-8")
        unsafe_module.chmod(0o664)
    (site_packages / "external-root.pth").write_text(
        f"{external_import_root}\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    if target_exists:
        assert ".pth import target" in result.stderr
        assert "must not be group- or world-writable" in result.stderr
    else:
        assert ".pth targets must exist before validation" in result.stderr


@pytest.mark.parametrize("trailing_whitespace", [" ", "\t", "\u00a0", "\u2003"])
def test_import_validation_checks_cpython_rstripped_pth_target(
    tmp_path: Path,
    trailing_whitespace: str,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    site_packages = project_environment / "lib" / "python3.12" / "site-packages"
    package_dir = site_packages / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    external_import_root = tmp_path / "external-import-root"
    external_import_root.mkdir()
    unsafe_module = external_import_root / "unsafe_module.py"
    unsafe_module.write_text("VALUE = 1\n", encoding="utf-8")
    unsafe_module.chmod(0o664)
    # Before validation, create a safe decoy at the literal spelling that the
    # shell used to inspect instead of CPython's rstripped path.
    (tmp_path / f"external-import-root{trailing_whitespace}").mkdir()
    (site_packages / "external-root.pth").write_text(
        f"{external_import_root}{trailing_whitespace}\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert ".pth import target" in result.stderr
    assert "must not be group- or world-writable" in result.stderr


def test_import_validation_ignores_cpython_whitespace_only_pth_lines(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    site_packages = project_environment / "lib" / "python3.12" / "site-packages"
    package_dir = site_packages / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    (site_packages / "whitespace-only.pth").write_text(
        " \t\u00a0\u2003\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("directive", "expected_message"),
    [
        (
            "import sys; sys.path.insert(0, '/tmp/unvalidated')",
            "unsupported executable directive",
        ),
        ("import unexpected_startup_hook", "unreviewed import hook"),
    ],
)
def test_import_validation_rejects_unsupported_pth_startup_code(
    tmp_path: Path,
    directive: str,
    expected_message: str,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    site_packages = project_environment / "lib" / "python3.12" / "site-packages"
    package_dir = site_packages / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    (site_packages / "startup-hook.pth").write_text(
        f"{directive}\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert expected_message in result.stderr


def test_import_validation_rejects_editable_finder_outside_project(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    package_dir = repo_dir / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    project_environment = repo_dir / ".venv"
    site_packages = project_environment / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    editable_module = "__editable___leveraged_trader_0_1_0_finder"
    (site_packages / "__editable__.leveraged_trader-0.1.0.pth").write_text(
        f"import {editable_module}; {editable_module}.install()\n",
        encoding="utf-8",
    )
    (site_packages / f"{editable_module}.py").write_text(
        "MAPPING: dict[str, str] = {'leveraged_trader': '/tmp/unvalidated'}\nNAMESPACES: dict[str, list[str]] = {}\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert "maps leveraged_trader outside the validated project package" in result.stderr


@pytest.mark.parametrize(
    "repo_name",
    [
        "repo'quoted",
        r"repo\\backslash",
        "repo'and\"double-quoted",
        "repo\u00a0space",
    ],
)
def test_import_validation_accepts_repr_encoded_editable_project_paths(
    tmp_path: Path,
    repo_name: str,
) -> None:
    repo_dir = tmp_path / repo_name
    package_dir = repo_dir / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    project_environment = repo_dir / ".venv"
    site_packages = project_environment / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    editable_module = "__editable___leveraged_trader_0_1_0_finder"
    (site_packages / "__editable__.leveraged_trader-0.1.0.pth").write_text(
        f"import {editable_module}; {editable_module}.install()\n",
        encoding="utf-8",
    )
    (site_packages / f"{editable_module}.py").write_text(
        f"MAPPING: dict[str, str] = {{'leveraged_trader': {str(package_dir)!r}}}\n"
        "NAMESPACES: dict[str, list[str]] = {}\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 0, result.stderr


def test_import_validation_rejects_system_site_packages(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    package_dir = project_environment / "lib" / "python3.12" / "site-packages" / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = true\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert "must not enable system site-packages" in result.stderr


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_import_validation_accepts_standard_pyvenv_line_endings(
    tmp_path: Path,
    line_ending: bytes,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    package_dir = project_environment / "lib" / "python3.12" / "site-packages" / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_bytes(
        line_ending.join(
            [
                b"home = /usr/bin",
                b"include-system-site-packages = false",
                b"",
            ]
        )
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("key_separator", ["\u00a0", "\u2007", "\u202f"])
def test_import_validation_rejects_unicode_system_site_override(
    tmp_path: Path,
    key_separator: str,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    package_dir = project_environment / "lib" / "python3.12" / "site-packages" / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        f"include-system-site-packages = false\ninclude-system-site-packages{key_separator}={key_separator}true\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert "must not enable system site-packages" in result.stderr


def test_import_validation_rejects_unicode_casefolded_system_site_override(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    package_dir = project_environment / "lib" / "python3.12" / "site-packages" / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\ninclude-system-site-pac\u212aages = true\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert "must not enable system site-packages" in result.stderr


def test_import_validation_rejects_lone_carriage_return_line_ending(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    package_dir = project_environment / "lib" / "python3.12" / "site-packages" / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_bytes(
        b"include-system-site-packages = false\nhome = /usr/bin\rinclude-system-site-packages = true\n"
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert "must not enable system site-packages" in result.stderr


def test_import_validation_accepts_key_literal_in_unrelated_value(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    package_dir = project_environment / "lib" / "python3.12" / "site-packages" / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        "home = /opt/include-system-site-packages/python\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 0, result.stderr


def test_import_validation_rejects_missing_system_site_packages_setting(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    package_dir = project_environment / "lib" / "python3.12" / "site-packages" / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        "home = /usr/bin\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert "must explicitly set include-system-site-packages to false" in result.stderr


def test_import_validation_rejects_bin_pyvenv_override(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_environment = repo_dir / ".venv"
    package_dir = project_environment / "lib" / "python3.12" / "site-packages" / "leveraged_trader"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (project_environment / "pyvenv.cfg").write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    bin_directory = project_environment / "bin"
    bin_directory.mkdir()
    (bin_directory / "pyvenv.cfg").write_text(
        "include-system-site-packages = true\n",
        encoding="utf-8",
    )

    result = _validate_fixture_import_tree(
        repo_dir=repo_dir,
        project_environment=project_environment,
    )

    assert result.returncode == 2
    assert "must not contain a bin/pyvenv.cfg override" in result.stderr


@pytest.mark.parametrize(
    "unsafe_target",
    [
        None,
        "module",
        "package-directory",
        "package-cache-directory",
        "package-cache",
        "package-extension",
        "source-symlink",
        "editable-artifact",
        "editable-finder",
        "editable-cache-directory",
        "editable-finder-cache",
        "shadow-module",
        "missing-editable-artifact",
        "unrelated-pth",
        "sitecustomize",
        "dependency-module",
        "nested-directory",
        "site-symlink",
        "pyvenv",
    ],
)
def test_runner_validates_complete_project_import_tree_before_exec(
    tmp_path: Path,
    unsafe_target: str | None,
) -> None:
    repo_dir = tmp_path / "repo"
    cron_dir = repo_dir / "scripts" / "cron"
    shutil.copytree(ROOT / "scripts" / "cron", cron_dir)

    package_dir = repo_dir / "leveraged_trader"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    project_module = package_dir / "__main__.py"
    project_module.write_text("raise SystemExit(0)\n", encoding="utf-8")
    package_cache_dir = package_dir / "__pycache__"
    package_cache_dir.mkdir()
    package_cache = package_cache_dir / "__main__.cpython-312.pyc"
    package_cache.write_bytes(b"validated cache fixture")
    package_extension = package_dir / "_native.so"
    package_extension.write_bytes(b"validated extension fixture")

    python_entry = repo_dir / ".venv" / "bin" / "python"
    python_entry.parent.mkdir(parents=True)
    python_capture = tmp_path / "python-capture"
    python_entry.write_text(
        "#!/bin/bash\n"
        'if [[ "${2:-}" == "-S" ]]; then\n'
        '    exec "$REAL_PYTHON" "$@"\n'
        "fi\n"
        'printf \'%s\\n\' "$@" > "$PYTHON_CAPTURE"\n',
        encoding="utf-8",
    )
    python_entry.chmod(0o755)

    site_packages = repo_dir / ".venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    pyvenv_config = repo_dir / ".venv" / "pyvenv.cfg"
    pyvenv_config.write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    editable_artifact = site_packages / "__editable__.leveraged_trader-0.1.0.pth"
    editable_artifact.write_text(
        "import __editable___leveraged_trader_0_1_0_finder\n",
        encoding="utf-8",
    )
    editable_finder = site_packages / "__editable___leveraged_trader_0_1_0_finder.py"
    editable_finder.write_text(
        f"MAPPING = {{'leveraged_trader': {str(package_dir)!r}}}\n",
        encoding="utf-8",
    )
    editable_cache_dir = site_packages / "__pycache__"
    editable_cache_dir.mkdir()
    editable_finder_cache = editable_cache_dir / "__editable___leveraged_trader_0_1_0_finder.cpython-312.pyc"
    editable_finder_cache.write_bytes(b"validated editable cache fixture")
    unrelated_pth = site_packages / "unrelated.pth"
    unrelated_pth.write_text("# validated unrelated path fixture\n", encoding="utf-8")
    sitecustomize = site_packages / "sitecustomize.py"
    sitecustomize.write_text("# validated site customization fixture\n", encoding="utf-8")
    dependency_dir = site_packages / "dependency_fixture"
    dependency_dir.mkdir()
    dependency_module = dependency_dir / "module.py"
    dependency_module.write_text("VALUE = 'validated'\n", encoding="utf-8")
    nested_dependency_dir = dependency_dir / "nested"
    nested_dependency_dir.mkdir()

    if unsafe_target == "module":
        project_module.chmod(0o664)
    elif unsafe_target == "package-directory":
        package_dir.chmod(0o775)
    elif unsafe_target == "package-cache-directory":
        package_cache_dir.chmod(0o775)
    elif unsafe_target == "package-cache":
        package_cache.chmod(0o664)
    elif unsafe_target == "package-extension":
        package_extension.chmod(0o664)
    elif unsafe_target == "source-symlink":
        (package_dir / "linked.py").symlink_to(project_module)
    elif unsafe_target == "editable-artifact":
        editable_artifact.chmod(0o664)
    elif unsafe_target == "editable-finder":
        editable_finder.chmod(0o664)
    elif unsafe_target == "editable-cache-directory":
        editable_cache_dir.chmod(0o775)
    elif unsafe_target == "editable-finder-cache":
        editable_finder_cache.chmod(0o664)
    elif unsafe_target == "shadow-module":
        shadow_module = site_packages / "leveraged_trader.py"
        shadow_module.write_text("raise SystemExit(0)\n", encoding="utf-8")
        shadow_module.chmod(0o664)
    elif unsafe_target == "missing-editable-artifact":
        editable_artifact.unlink()
    elif unsafe_target == "unrelated-pth":
        unrelated_pth.chmod(0o664)
    elif unsafe_target == "sitecustomize":
        sitecustomize.chmod(0o664)
    elif unsafe_target == "dependency-module":
        dependency_module.chmod(0o664)
    elif unsafe_target == "nested-directory":
        nested_dependency_dir.chmod(0o775)
    elif unsafe_target == "site-symlink":
        (site_packages / "dependency-link").symlink_to(dependency_dir, target_is_directory=True)
    elif unsafe_target == "pyvenv":
        pyvenv_config.chmod(0o664)

    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    env = os.environ.copy()
    env["UV_BIN"] = str(fake_uv)
    env["PYTHON_CAPTURE"] = str(python_capture)
    env["REAL_PYTHON"] = sys.executable
    result = subprocess.run(
        [str(cron_dir / "run-leveraged-trader"), "--help"],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    if unsafe_target is None:
        assert result.returncode == 0, result.stderr
        assert python_capture.read_text(encoding="utf-8").splitlines() == [
            "-I",
            "-m",
            "leveraged_trader",
            "--help",
        ]
    else:
        assert result.returncode == 2
        if unsafe_target == "source-symlink":
            assert "must not contain symbolic links" in result.stderr
        elif unsafe_target == "site-symlink":
            assert "trusted non-symlink regular files and directories" in result.stderr
        elif unsafe_target == "missing-editable-artifact":
            assert "Cannot locate a validated leveraged_trader install" in result.stderr
        else:
            assert "must not be group- or world-writable" in result.stderr
        assert not python_capture.exists()


def test_runner_rejects_stale_locked_environment_before_exec(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    python_marker = tmp_path / "python-called"
    fake_python = repo_dir / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        '#!/bin/bash\nprintf called > "$PYTHON_MARKER"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    uv_capture = tmp_path / "uv-args"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/bin/bash
set -euo pipefail
[[ "${UV_PROJECT+x}" != x && "${UV_WORKING_DIR+x}" != x && "${UV_CONFIG_FILE+x}" != x ]]
[[ "${UV_NO_DEV+x}" != x && "${UV_NO_GROUP+x}" != x && "${UV_NO_DEFAULT_GROUPS+x}" != x ]]
[[ "${UV_FROZEN+x}" != x && "${UV_PYTHON+x}" != x && "${UV_NO_EDITABLE+x}" != x ]]
printf '%s\n' "$UV_PROJECT_ENVIRONMENT" "$@" > "$TEST_UV_CAPTURE"
exit 42
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env["UV_BIN"] = str(fake_uv)
    env["TEST_UV_CAPTURE"] = str(uv_capture)
    env["PYTHON_MARKER"] = str(python_marker)
    env["UV_PROJECT"] = "/untrusted/project"
    env["UV_WORKING_DIR"] = "/untrusted/working-directory"
    env["UV_CONFIG_FILE"] = "/untrusted/uv.toml"
    env["UV_NO_DEV"] = "1"
    env["UV_NO_GROUP"] = "dev"
    env["UV_NO_DEFAULT_GROUPS"] = "1"
    env["UV_FROZEN"] = "1"
    env["UV_PYTHON"] = "/bin/false"
    env["UV_NO_EDITABLE"] = "1"
    result = subprocess.run(
        [str(repo_dir / "scripts" / "cron" / "run-leveraged-trader"), "--help"],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 42
    assert uv_capture.read_text(encoding="utf-8").splitlines() == [
        str(repo_dir / ".venv"),
        "sync",
        "--project",
        str(repo_dir),
        "--directory",
        str(repo_dir),
        "--locked",
        "--check",
    ]
    assert "Locked project environment is not synchronized" in result.stderr
    assert "uv sync --locked" in result.stderr
    assert not python_marker.exists()


def test_runner_ignores_inherited_pythonpath_sitecustomize(tmp_path: Path) -> None:
    injected_directory = tmp_path / "injected"
    injected_directory.mkdir()
    sitecustomize_marker = tmp_path / "sitecustomize-called"
    (injected_directory / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['SITECUSTOMIZE_MARKER']).write_text('called', encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env["UV_BIN"] = str(fake_uv)
    env["PYTHONPATH"] = str(injected_directory)
    env["SITECUSTOMIZE_MARKER"] = str(sitecustomize_marker)
    result = subprocess.run(
        [str(RUNNER), "--help"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert not sitecustomize_marker.exists()


def test_scheduler_invokes_production_runner_with_absolute_bash_outside_path(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"

    bin_without_bash = tmp_path / "bin-without-bash"
    bin_without_bash.mkdir()
    dirname = shutil.which("dirname")
    assert dirname is not None
    (bin_without_bash / "dirname").symlink_to(dirname)

    uv_capture = tmp_path / "uv-capture"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$TEST_UV_CAPTURE"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    python_capture = tmp_path / "python-capture"
    fake_python = repo_dir / ".venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$PYTHON_CAPTURE"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(bin_without_bash)
    env["UV_BIN"] = str(fake_uv)
    env["TEST_UV_CAPTURE"] = str(uv_capture)
    env["PYTHON_CAPTURE"] = str(python_capture)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env.pop("LEVERAGED_TRADER_LOG_FILE", None)
    env.pop("LEVERAGED_TRADER_RUNNER", None)
    env.pop("BASH_ENV", None)
    env.pop("ENV", None)
    result = subprocess.run(
        ["/bin/bash", str(scheduler)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert uv_capture.read_text(encoding="utf-8").splitlines() == [
        "sync",
        "--project",
        str(repo_dir),
        "--directory",
        str(repo_dir),
        "--locked",
        "--check",
    ]
    assert python_capture.read_text(encoding="utf-8").splitlines() == [
        "-I",
        "-m",
        "leveraged_trader",
        "--reconcile-only",
        "--alpaca-submit-sell-orders",
        "--scheduled-closed-audit-interval-minutes",
        "15",
    ]


def test_scheduler_rotates_log_at_size_limit_before_appending(tmp_path: Path) -> None:
    log_file = tmp_path / "cron log.txt"
    log_file.write_text("0123456789", encoding="utf-8")
    backup_file = tmp_path / "cron log.txt.1"
    backup_file.write_text("older backup", encoding="utf-8")

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_LOG_MAX_BYTES"] = "10"
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert backup_file.read_text(encoding="utf-8") == "0123456789"
    assert log_file.read_text(encoding="utf-8") == f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n"
    assert not (tmp_path / "cron log.txt.2").exists()
    assert not (tmp_path / "cron log.txt.rotate.lock").exists()
    assert (tmp_path / "cron log.txt.lock").stat().st_mode & 0o777 == 0o600
    assert log_file.stat().st_mode & 0o777 == 0o600
    assert backup_file.stat().st_mode & 0o777 == 0o600


def test_scheduler_appends_without_rotating_log_below_limit(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    log_file.write_text("existing\n", encoding="utf-8")

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 08:45"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_LOG_MAX_BYTES"] = "1000"
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert log_file.read_text(encoding="utf-8") == (
        "existing\n--require-workflow-source-success --alpaca-submit-buy-orders --alpaca-submit-sell-orders\n"
    )
    assert not (tmp_path / "cron.log.1").exists()


def test_scheduler_restricts_log_directory_to_owner_access(tmp_path: Path) -> None:
    log_directory = tmp_path / "shared-output"
    log_directory.mkdir(mode=0o777)
    log_directory.chmod(0o777)
    log_file = log_directory / "cron.log"

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert log_directory.stat().st_mode & 0o777 == 0o700
    assert log_file.stat().st_mode & 0o777 == 0o600


def test_scheduler_rejects_intermediate_symlink_before_creating_log_directory(tmp_path: Path) -> None:
    safe_directory = tmp_path / "safe"
    safe_directory.mkdir(mode=0o700)
    shared_directory = tmp_path / "shared"
    shared_directory.mkdir(mode=0o777)
    shared_directory.chmod(0o777)
    redirect = shared_directory / "redirect"
    redirect.symlink_to(safe_directory, target_is_directory=True)
    log_directory = redirect / "new" / "logs"

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_directory / "cron.log")
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "ancestor must not be a symbolic link" in result.stderr
    assert not (safe_directory / "new").exists()


def test_scheduler_rejects_symlink_before_dotdot_without_creating_either_namespace(tmp_path: Path) -> None:
    base_directory = tmp_path / "base"
    base_directory.mkdir(mode=0o700)
    actual_directory = tmp_path / "actual"
    actual_directory.mkdir(mode=0o700)
    (actual_directory / "inner").mkdir(mode=0o700)
    actual_logs = actual_directory / "logs"
    actual_logs.mkdir(mode=0o700)
    actual_directory.chmod(0o777)
    (base_directory / "jump").symlink_to(actual_directory / "inner", target_is_directory=True)
    lexical_log_file = base_directory / "jump" / ".." / "logs" / "cron.log"

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(lexical_log_file)
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must not contain '..' components" in result.stderr
    assert not (base_directory / "logs").exists()
    assert list(actual_logs.iterdir()) == []


def test_scheduler_rejects_newline_log_path_without_creating_truncated_prefix(tmp_path: Path) -> None:
    truncated_prefix = tmp_path / "clean"
    intended_prefix = tmp_path / "clean\nunsafe"
    raw_log_file = f"{intended_prefix}/logs/cron.log"

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = raw_log_file
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must not contain ASCII control characters" in result.stderr
    assert not truncated_prefix.exists()
    assert not intended_prefix.exists()


@pytest.mark.parametrize("suffix", ["/", "/."], ids=["trailing-slash", "final-dot"])
def test_scheduler_rejects_directory_spelling_as_log_path_without_creating_prefix(
    tmp_path: Path,
    suffix: str,
) -> None:
    logs_path = tmp_path / "logs"
    raw_log_file = f"{logs_path}{suffix}"

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = raw_log_file
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must name a file, not a directory spelling" in result.stderr
    assert not logs_path.exists()


def test_scheduler_rejects_nonsticky_writable_parent_before_creating_log_directory(tmp_path: Path) -> None:
    shared_directory = tmp_path / "shared"
    shared_directory.mkdir(mode=0o777)
    shared_directory.chmod(0o777)
    log_directory = shared_directory / "new" / "logs"

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_directory / "cron.log")
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "permits directory-entry substitution" in result.stderr
    assert not (shared_directory / "new").exists()


def test_scheduler_rejects_read_only_ancestor_with_mocked_untrusted_owner(tmp_path: Path) -> None:
    untrusted_directory = tmp_path / "untrusted"
    untrusted_directory.mkdir(mode=0o555)
    untrusted_directory.chmod(0o555)
    log_directory = untrusted_directory / "new" / "logs"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real_stat = shutil.which("stat")
    assert real_stat is not None
    fake_stat = bin_dir / "stat"
    fake_stat.write_text(
        """#!/bin/bash
set -euo pipefail
last="${!#}"
if [[ "$last" == "$UNTRUSTED_DIRECTORY" && "$*" == *"%u %a"* ]]; then
    printf '%s 555\n' "$UNTRUSTED_UID"
    exit 0
fi
exec "$REAL_STAT" "$@"
""",
        encoding="utf-8",
    )
    fake_stat.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["REAL_STAT"] = real_stat
    env["UNTRUSTED_DIRECTORY"] = str(untrusted_directory)
    env["UNTRUSTED_UID"] = str(max(os.geteuid(), 0) + 100_000)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_directory / "cron.log")
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "controlled by an untrusted owner" in result.stderr
    assert not (untrusted_directory / "new").exists()


def test_scheduler_accepts_trusted_child_beneath_fake_bsd_sticky_parent(tmp_path: Path) -> None:
    shared_directory = tmp_path / "shared"
    shared_directory.mkdir(mode=0o777)
    shared_directory.chmod(0o777)
    log_file = shared_directory / "logs" / "cron.log"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    real_stat = shutil.which("stat")
    assert real_stat is not None
    bsd_calls = tmp_path / "bsd-stat-calls"
    fake_stat = bin_dir / "stat"
    fake_stat.write_text(
        """#!/bin/bash
set -euo pipefail
case "${1:-}" in
    -c|-Lc)
        exit 1
        ;;
    -f)
        format="${2:-}"
        path="${3:-}"
        if [[ "$format" != "%u" ]]; then
            exit 2
        fi
        exec "$REAL_STAT" -c '%u' -- "$path"
        ;;
    -Lf)
        format="${2:-}"
        path="${3:-}"
        printf '%s %s\n' "$format" "$path" >> "$BSD_CALLS"
        case "$format" in
            "%u %p")
                owner_uid="$("$REAL_STAT" -L -c '%u' -- "$path")"
                if [[ "$path" == "$STICKY_PARENT" ]]; then
                    full_mode=41777
                else
                    permissions="$("$REAL_STAT" -L -c '%a' -- "$path")"
                    full_mode="4$permissions"
                fi
                printf '%s %s\n' "$owner_uid" "$full_mode"
                ;;
            "%l")
                exec "$REAL_STAT" -L -c '%h' -- "$path"
                ;;
            "%m")
                exec "$REAL_STAT" -L -c '%Y' -- "$path"
                ;;
            *)
                exit 2
                ;;
        esac
        ;;
    *)
        exit 2
        ;;
esac
""",
        encoding="utf-8",
    )
    fake_stat.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["REAL_STAT"] = real_stat
    env["BSD_CALLS"] = str(bsd_calls)
    env["STICKY_PARENT"] = str(shared_directory)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"%u %p {shared_directory}" in bsd_calls.read_text(encoding="utf-8")
    assert log_file.read_text(encoding="utf-8") == f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n"


@pytest.mark.parametrize(
    ("suffix", "description"),
    [("", "log"), (".1", "backup log"), (".lock", "lock")],
)
def test_scheduler_rejects_symlinked_runtime_files_without_touching_target(
    tmp_path: Path,
    suffix: str,
    description: str,
) -> None:
    log_file = tmp_path / "cron.log"
    if suffix:
        log_file.write_text("0123456789", encoding="utf-8")
    protected_target = tmp_path / "unrelated"
    protected_target.write_text("must remain unchanged", encoding="utf-8")
    protected_target.chmod(0o644)
    Path(f"{log_file}{suffix}").symlink_to(protected_target)

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_LOG_MAX_BYTES"] = "10"
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"Scheduler {description} path must not be a symbolic link" in result.stderr
    assert protected_target.read_text(encoding="utf-8") == "must remain unchanged"
    assert protected_target.stat().st_mode & 0o777 == 0o644


def test_scheduler_rejects_hardlinked_log_without_touching_target(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    protected_target = tmp_path / "unrelated"
    protected_target.write_text("must remain unchanged", encoding="utf-8")
    protected_target.chmod(0o644)
    os.link(protected_target, log_file)

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "must not have multiple hard links" in result.stderr
    assert protected_target.read_text(encoding="utf-8") == "must remain unchanged"
    assert protected_target.stat().st_mode & 0o777 == 0o644


def test_scheduler_detects_log_substitution_after_descriptor_open(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    protected_target = tmp_path / "unrelated"
    protected_target.write_text("must remain unchanged", encoding="utf-8")
    protected_target.chmod(0o644)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_chmod = bin_dir / "chmod"
    fake_chmod.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
/bin/chmod "$@"
last="${!#}"
if [[ "$last" == "/dev/fd/8" && ! -L "$ATTACK_LOG" ]]; then
    rm -f -- "$ATTACK_LOG"
    ln -s -- "$ATTACK_TARGET" "$ATTACK_LOG"
fi
""",
        encoding="utf-8",
    )
    fake_chmod.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["ATTACK_LOG"] = str(log_file)
    env["ATTACK_TARGET"] = str(protected_target)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "changed while it was being secured" in result.stderr
    assert protected_target.read_text(encoding="utf-8") == "must remain unchanged"
    assert protected_target.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize(
    ("suffix", "description"),
    [("", "log"), (".1", "backup log"), (".lock", "lock")],
)
def test_scheduler_rejects_nonregular_runtime_files(
    tmp_path: Path,
    suffix: str,
    description: str,
) -> None:
    log_file = tmp_path / "cron.log"
    if suffix:
        log_file.write_text("0123456789", encoding="utf-8")
    Path(f"{log_file}{suffix}").mkdir()

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_LOG_MAX_BYTES"] = "10"
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert f"Scheduler {description} path must be a regular file" in result.stderr
    assert Path(f"{log_file}{suffix}").is_dir()


def test_scheduler_rejects_symlinked_fallback_lock_directory(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    protected_directory = tmp_path / "unrelated directory"
    protected_directory.mkdir()
    marker = protected_directory / "marker"
    marker.write_text("must remain unchanged", encoding="utf-8")
    fallback_lock = tmp_path / "cron.log.lock.d"
    fallback_lock.symlink_to(protected_directory, target_is_directory=True)

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "stat"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "fallback lock path must be a regular directory" in result.stderr
    assert fallback_lock.is_symlink()
    assert marker.read_text(encoding="utf-8") == "must remain unchanged"


def test_scheduler_reports_flock_errors_instead_of_treating_them_as_contention(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_flock = bin_dir / "flock"
    fake_flock.write_text("#!/usr/bin/env bash\nexit 2\n", encoding="utf-8")
    fake_flock.chmod(0o755)
    log_file = tmp_path / "cron.log"

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Scheduler flock failed with status 2" in result.stderr
    assert not log_file.exists()


def test_scheduler_uses_fallback_when_flock_does_not_support_exit_code_option(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin-with-busybox-flock"
    bin_dir.mkdir()
    unexpected_call = tmp_path / "unexpected-flock-acquisition"
    fake_flock = bin_dir / "flock"
    fake_flock.write_text(
        """#!/bin/bash
if [[ "${1:-}" == "--help" ]]; then
    echo 'BusyBox v1.37.0'
    echo 'Usage: flock [-sxun] FD' >&2
    exit 1
fi
: > "$UNEXPECTED_FLOCK_ACQUISITION"
exit 99
""",
        encoding="utf-8",
    )
    fake_flock.chmod(0o755)
    for command in ["chmod", "dirname", "mkdir", "mv", "ps", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    log_file = tmp_path / "cron.log"
    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["UNEXPECTED_FLOCK_ACQUISITION"] = str(unexpected_call)
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not unexpected_call.exists()
    assert not Path(f"{log_file}.lock").exists()
    assert Path(f"{log_file}.lock.d").is_dir()
    assert log_file.read_text(encoding="utf-8") == f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n"


def test_stale_legacy_rotation_directory_cannot_disable_rotation(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    log_file.write_text("0123456789", encoding="utf-8")
    (tmp_path / "cron.log.rotate.lock").mkdir()

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_LOG_MAX_BYTES"] = "10"
    result = subprocess.run(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (tmp_path / "cron.log.1").read_text(encoding="utf-8") == "0123456789"
    assert log_file.read_text(encoding="utf-8") == f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n"


def test_scheduler_lock_skips_overlap_and_releases_automatically(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    record_file = tmp_path / "runs"
    block_file = tmp_path / "block"
    started_file = tmp_path / "started"
    block_file.touch()
    runner = tmp_path / "runner"
    runner.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf 'run\\n' >> "$RUN_RECORD"
if [[ -e "$BLOCK_FILE" ]]; then
    : > "$STARTED_FILE"
    while [[ -e "$BLOCK_FILE" ]]; do
        sleep 0.02
    done
fi
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    env = os.environ.copy()
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["RUN_RECORD"] = str(record_file)
    env["BLOCK_FILE"] = str(block_file)
    env["STARTED_FILE"] = str(started_file)
    first = subprocess.Popen(
        ["bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(100):
            if started_file.exists():
                break
            time.sleep(0.02)
        assert started_file.exists()

        overlapping = subprocess.run(
            ["bash", str(SCHEDULER)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        assert overlapping.returncode == 0
        assert record_file.read_text(encoding="utf-8").splitlines() == ["run"]

        block_file.unlink()
        first.communicate(timeout=2)
        assert first.returncode == 0
        after_release = subprocess.run(
            ["bash", str(SCHEDULER)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        assert after_release.returncode == 0
        assert record_file.read_text(encoding="utf-8").splitlines() == ["run", "run"]
    finally:
        block_file.unlink(missing_ok=True)
        _terminate_and_reap(first)


def test_scheduler_recovers_abandoned_pid_lock_without_flock(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    log_file.write_text("0123456789", encoding="utf-8")
    stale_lock = tmp_path / "cron.log.lock.d"
    stale_lock.mkdir()
    (stale_lock / "pid").write_text("99999999\n", encoding="utf-8")

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "mv", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_LOG_MAX_BYTES"] = "10"
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (stale_lock / "owner").is_file()
    assert not (stale_lock / "pid").exists()
    assert (tmp_path / "cron.log.1").read_text(encoding="utf-8") == "0123456789"
    assert log_file.read_text(encoding="utf-8") == f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n"


def test_scheduler_recovers_old_ownerless_fallback_lock_without_flock(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    stale_lock = tmp_path / "cron.log.lock.d"
    stale_lock.mkdir()
    (stale_lock / "owner.pending").write_text("interrupted publication\n", encoding="utf-8")
    recovery_marker = stale_lock / ".stale-recovery"
    recovery_marker.mkdir()
    (recovery_marker / "owner.pending").write_text(
        "interrupted recovery publication\n",
        encoding="utf-8",
    )
    old_timestamp = time.time() - 300
    os.utime(stale_lock, (old_timestamp, old_timestamp))

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "date", "dirname", "mkdir", "mv", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (stale_lock / "owner").is_file()
    assert log_file.read_text(encoding="utf-8") == f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n"


def test_scheduler_stale_recovery_blocks_path_replacement_and_third_contender(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    lock_dir = tmp_path / "cron.log.lock.d"
    lock_dir.mkdir()
    (lock_dir / "owner").write_text("99999999 proc:0\n", encoding="utf-8")
    inspected_lock = tmp_path / "inspected-stale.lock"
    swap_marker = tmp_path / "scheduler-lock-swapped"
    runner_called = tmp_path / "runner-called"
    runner = tmp_path / "runner"
    runner.write_text('#!/bin/bash\n: > "$RUNNER_CALLED"\n', encoding="utf-8")
    runner.chmod(0o755)

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "ps", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    real_mkdir = shutil.which("mkdir")
    real_mv = shutil.which("mv")
    real_rm = shutil.which("rm")
    assert real_mkdir is not None and real_mv is not None and real_rm is not None
    live_owner = f"{os.getpid()} {_process_start_identity(os.getpid())}\n"
    fake_rm = bin_dir / "rm"
    fake_rm.unlink()
    fake_rm.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$PWD" == "$LOCK_DIR" && "$*" == *"./owner"* && ! -e "$SWAP_MARKER" ]]; then
    "$REAL_MV" "$LOCK_DIR" "$INSPECTED_LOCK"
    "$REAL_MKDIR" "$LOCK_DIR"
    printf '%s' "$LIVE_OWNER" > "$LOCK_DIR/owner"
    : > "$SWAP_MARKER"
fi
exec "$REAL_RM" "$@"
""",
        encoding="utf-8",
    )
    fake_rm.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["RUNNER_CALLED"] = str(runner_called)
    env["LOCK_DIR"] = str(lock_dir)
    env["INSPECTED_LOCK"] = str(inspected_lock)
    env["SWAP_MARKER"] = str(swap_marker)
    env["LIVE_OWNER"] = live_owner
    env["REAL_MKDIR"] = real_mkdir
    env["REAL_MV"] = real_mv
    env["REAL_RM"] = real_rm
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    third_contender = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert third_contender.returncode == 0, third_contender.stderr
    assert swap_marker.is_file()
    assert (lock_dir / "owner").read_text(encoding="utf-8") == live_owner
    assert not (lock_dir / ".stale-recovery").exists()
    assert not (inspected_lock / "owner").exists()
    assert not runner_called.exists()
    assert not log_file.exists()
    assert 'mv "$lock_dir"' not in SCHEDULER.read_text(encoding="utf-8")


def test_scheduler_treats_recent_ownerless_fallback_lock_as_in_progress(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    owner_publication_in_progress = tmp_path / "cron.log.lock.d"
    owner_publication_in_progress.mkdir()

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "date", "dirname", "mkdir", "rm", "rmdir", "stat"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert owner_publication_in_progress.is_dir()
    assert not log_file.exists()


@pytest.mark.parametrize(
    "abandoned_state",
    [
        ("gate", "owner", "99999999 proc:0\n"),
        ("gate", "owner", f"{os.getpid()} proc:0\n"),
        ("gate", None, None),
        ("gate", "owner", "99999999"),
        ("gate", "owner.pending", "99999999"),
        ("claim", "owner", "99999999"),
        ("claim", "owner.pending", "99999999"),
    ],
    ids=[
        "dead-owner",
        "reused-pid",
        "expired-ownerless",
        "partial-owner",
        "partial-owner-pending",
        "partial-claim-owner",
        "partial-claim-owner-pending",
    ],
)
def test_scheduler_recovers_abandoned_acquisition_gate(
    tmp_path: Path,
    abandoned_state: tuple[str, str | None, str | None],
) -> None:
    log_file = tmp_path / "cron.log"
    gate_dir = tmp_path / "cron.log.lock.acquire.d"
    artifact_kind, metadata_name, metadata = abandoned_state
    abandoned_dir = gate_dir if artifact_kind == "gate" else Path(f"{gate_dir}.recovery.crashed")
    abandoned_dir.mkdir()
    if metadata_name is not None:
        (abandoned_dir / metadata_name).write_text(metadata or "", encoding="utf-8")
    old_timestamp = time.time() - 300
    os.utime(abandoned_dir, (old_timestamp, old_timestamp))

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "date", "dirname", "mkdir", "mv", "ps", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not gate_dir.exists()
    assert not abandoned_dir.exists()
    assert (tmp_path / "cron.log.lock.d" / "owner").is_file()
    assert log_file.read_text(encoding="utf-8") == (f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n")


def test_scheduler_recovers_reused_live_pid_when_process_start_identity_differs(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    stale_lock = tmp_path / "cron.log.lock.d"
    stale_lock.mkdir()
    (stale_lock / "owner").write_text(f"{os.getpid()} proc:0\n", encoding="utf-8")

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "mv", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (stale_lock / "owner").is_file()
    assert log_file.read_text(encoding="utf-8") == f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n"


def test_scheduler_fallback_owner_execs_runner_and_prevents_orphaned_overlap(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    fallback_lock = tmp_path / "cron.log.lock.d"
    runner_pid_file = tmp_path / "runner.pid"
    runner_fd_state = tmp_path / "runner-fd5"
    block_file = tmp_path / "block"
    block_file.touch()
    runner = tmp_path / "runner"
    runner.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$$" > "$RUNNER_PID_FILE"
if [[ -e /dev/fd/5 ]]; then
    printf 'open\n' > "$RUNNER_FD_STATE"
else
    printf 'closed\n' > "$RUNNER_FD_STATE"
fi
while [[ -e "$BLOCK_FILE" ]]; do
    /bin/sleep 0.02
done
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "mv", "ps", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["RUNNER_PID_FILE"] = str(runner_pid_file)
    env["RUNNER_FD_STATE"] = str(runner_fd_state)
    env["BLOCK_FILE"] = str(block_file)
    first = subprocess.Popen(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(100):
            if runner_pid_file.exists() and (fallback_lock / "owner").exists():
                break
            time.sleep(0.02)
        assert runner_pid_file.exists()
        assert (fallback_lock / "owner").exists()

        runner_pid = int(runner_pid_file.read_text(encoding="utf-8").strip())
        owner_pid = int((fallback_lock / "owner").read_text(encoding="utf-8").split(maxsplit=1)[0])
        assert runner_pid == first.pid
        assert owner_pid == first.pid
        assert runner_fd_state.read_text(encoding="utf-8") == "closed\n"

        first.kill()
        first.communicate(timeout=2)
        assert first.returncode != 0
        with pytest.raises(ProcessLookupError):
            os.kill(runner_pid, 0)

        retry_runner = tmp_path / "retry-runner"
        retry_runner.write_text(
            "#!/bin/bash\nprintf '%s\\n' \"$*\"\nexit 23\n",
            encoding="utf-8",
        )
        retry_runner.chmod(0o755)
        retry_env = env.copy()
        retry_env["LEVERAGED_TRADER_RUNNER"] = str(retry_runner)
        retry = subprocess.run(
            ["/bin/bash", str(SCHEDULER)],
            cwd=ROOT,
            env=retry_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        assert retry.returncode == 23, retry.stderr
        assert log_file.read_text(encoding="utf-8") == f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n"
    finally:
        block_file.unlink(missing_ok=True)
        _terminate_and_reap(first)


def test_scheduler_production_runner_avoids_uv_parent_orphan_window(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    scheduler = repo_dir / "scripts" / "cron" / "run-scheduled"
    log_file = tmp_path / "cron.log"
    fallback_lock = tmp_path / "cron.log.lock.d"

    uv_child_block = tmp_path / "uv-child-block"
    uv_child_block.touch()
    uv_child_finished = tmp_path / "uv-child-finished"
    uv_child = tmp_path / "uv-child"
    uv_child.write_text(
        """#!/bin/bash
set -euo pipefail
while [[ -e "$TEST_UV_CHILD_BLOCK" ]]; do
    /bin/sleep 0.02
done
: > "$TEST_UV_CHILD_FINISHED"
""",
        encoding="utf-8",
    )
    uv_child.chmod(0o755)

    uv_args = tmp_path / "uv-args"
    uv_topology = tmp_path / "uv-topology"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$@" > "$TEST_UV_ARGS"
if [[ "${1:-}" == "sync" \
    && "${2:-}" == "--project" \
    && "${3:-}" == "$EXPECTED_REPO_DIR" \
    && "${4:-}" == "--directory" \
    && "${5:-}" == "$EXPECTED_REPO_DIR" \
    && "${6:-}" == "--locked" \
    && "${7:-}" == "--check" \
    && "$#" == 7 ]]; then
    exit 0
fi
if [[ "${1:-}" == "run" ]]; then
    "$TEST_UV_RUN_CHILD" &
    child_pid=$!
    printf '%s %s\n' "$$" "$child_pid" > "$TEST_UV_TOPOLOGY"
    wait "$child_pid"
    exit $?
fi
exit 64
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    trader_pid_file = tmp_path / "trader.pid"
    trader_block = tmp_path / "trader-block"
    trader_block.touch()
    trader_python = repo_dir / ".venv" / "bin" / "python"
    trader_python.parent.mkdir(parents=True)
    trader_python.write_text(
        """#!/bin/bash
set -euo pipefail
[[ "$#" -ge 3 && "$1" == "-I" && "$2" == "-m" && "$3" == "leveraged_trader" ]]
shift 3
printf '%s\n' "$$" > "$TRADER_PID_FILE"
while [[ -e "$TRADER_BLOCK" ]]; do
    /bin/sleep 0.02
done
""",
        encoding="utf-8",
    )
    trader_python.chmod(0o755)

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["bash", "chmod", "dirname", "mkdir", "mv", "ps", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["UV_BIN"] = str(fake_uv)
    env["TEST_UV_ARGS"] = str(uv_args)
    env["TEST_UV_TOPOLOGY"] = str(uv_topology)
    env["TEST_UV_RUN_CHILD"] = str(uv_child)
    env["TEST_UV_CHILD_BLOCK"] = str(uv_child_block)
    env["TEST_UV_CHILD_FINISHED"] = str(uv_child_finished)
    env["TRADER_PID_FILE"] = str(trader_pid_file)
    env["TRADER_BLOCK"] = str(trader_block)
    env["EXPECTED_REPO_DIR"] = str(repo_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env.pop("LEVERAGED_TRADER_RUNNER", None)

    # Prove that this uv fixture models the unsafe `uv run` topology: killing
    # its parent leaves the long-lived command child alive.
    uv_parent = subprocess.Popen(
        [str(fake_uv), "run"],
        cwd=repo_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    first: subprocess.Popen[str] | None = None
    uv_child_pid: int | None = None
    try:
        _wait_for_path(uv_topology, process=uv_parent)
        uv_parent_pid, uv_child_pid = map(int, uv_topology.read_text(encoding="utf-8").split())
        assert uv_parent_pid == uv_parent.pid
        assert uv_child_pid != uv_parent.pid
        uv_parent.kill()
        assert uv_parent.wait(timeout=2) != 0
        os.kill(uv_child_pid, 0)
        uv_child_block.unlink()
        _wait_for_path(uv_child_finished)
        # The intentionally orphaned child inherits the parent's stderr pipe,
        # so collect it only after that child exits and releases the last writer.
        uv_parent.communicate(timeout=2)

        uv_args.unlink(missing_ok=True)
        uv_topology.unlink(missing_ok=True)
        first = subprocess.Popen(
            ["/bin/bash", str(scheduler)],
            cwd=repo_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _wait_for_path(trader_pid_file, process=first)
        _wait_for_path(fallback_lock / "owner", process=first)
        assert not uv_topology.exists()
        assert uv_args.read_text(encoding="utf-8").splitlines() == [
            "sync",
            "--project",
            str(repo_dir),
            "--directory",
            str(repo_dir),
            "--locked",
            "--check",
        ]

        trader_pid = int(trader_pid_file.read_text(encoding="utf-8").strip())
        owner_pid = int((fallback_lock / "owner").read_text(encoding="utf-8").split(maxsplit=1)[0])
        assert trader_pid == first.pid
        assert owner_pid == first.pid

        first.kill()
        first.communicate(timeout=2)
        assert first.returncode != 0
        with pytest.raises(ProcessLookupError):
            os.kill(trader_pid, 0)

        trader_python.write_text(
            "#!/bin/bash\n"
            '[[ "$#" -ge 3 && "$1" == -I && "$2" == -m && "$3" == leveraged_trader ]]\n'
            "shift 3\n"
            "printf '%s\\n' \"$*\"\n"
            "exit 23\n",
            encoding="utf-8",
        )
        trader_python.chmod(0o755)
        retry = subprocess.run(
            ["/bin/bash", str(scheduler)],
            cwd=repo_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        assert retry.returncode == 23, retry.stderr
        assert log_file.read_text(encoding="utf-8") == f"{SCHEDULED_RECONCILIATION_ARGUMENTS}\n"
    finally:
        uv_child_block.unlink(missing_ok=True)
        trader_block.unlink(missing_ok=True)
        _terminate_and_reap(uv_parent)
        if first is not None:
            _terminate_and_reap(first)
        if uv_child_pid is not None:
            try:
                os.kill(uv_child_pid, 0)
            except ProcessLookupError:
                pass
            else:
                os.kill(uv_child_pid, 9)


@pytest.mark.parametrize(
    "lock_suffix",
    ["lock.acquire.d", "lock.d"],
    ids=["acquisition-gate", "main-lock"],
)
def test_scheduler_retries_when_normal_lock_release_removes_contended_path(
    tmp_path: Path,
    lock_suffix: str,
) -> None:
    log_file = tmp_path / "cron.log"
    target_lock = tmp_path / f"cron.log.{lock_suffix}"
    release_entered = tmp_path / "release-entered"
    allow_release = tmp_path / "allow-release"
    contender_triggered = tmp_path / "contender-triggered"

    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mv", "ps", "rm", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    real_mkdir = shutil.which("mkdir")
    real_rmdir = shutil.which("rmdir")
    assert real_mkdir is not None and real_rmdir is not None
    fake_mkdir = bin_dir / "mkdir"
    fake_mkdir.write_text(
        """#!/bin/bash
set -euo pipefail
set +e
"$REAL_MKDIR" "$@"
status=$?
set -e
target="${!#}"
if (( status != 0 )) && [[ "$target" == "$TARGET_LOCK" \
    && -e "$RELEASE_ENTERED" && ! -e "$CONTENDER_TRIGGERED" ]]; then
    : > "$CONTENDER_TRIGGERED"
    : > "$ALLOW_RELEASE"
    while [[ -e "$TARGET_LOCK" || -L "$TARGET_LOCK" ]]; do
        /bin/sleep 0.01
    done
fi
exit "$status"
""",
        encoding="utf-8",
    )
    fake_mkdir.chmod(0o755)
    fake_rmdir = bin_dir / "rmdir"
    fake_rmdir.write_text(
        """#!/bin/bash
set -euo pipefail
target="${!#}"
if [[ "$target" == "$TARGET_LOCK" && ! -e "$RELEASE_ENTERED" ]]; then
    [[ ! -e "$TARGET_LOCK/owner" ]]
    : > "$RELEASE_ENTERED"
    while [[ ! -e "$ALLOW_RELEASE" ]]; do
        /bin/sleep 0.01
    done
fi
exec "$REAL_RMDIR" "$@"
""",
        encoding="utf-8",
    )
    fake_rmdir.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_LOG_MAX_BYTES"] = "invalid"
    env["TARGET_LOCK"] = str(target_lock)
    env["RELEASE_ENTERED"] = str(release_entered)
    env["ALLOW_RELEASE"] = str(allow_release)
    env["CONTENDER_TRIGGERED"] = str(contender_triggered)
    env["REAL_MKDIR"] = real_mkdir
    env["REAL_RMDIR"] = real_rmdir

    first = subprocess.Popen(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_path(release_entered, process=first)
        contender = subprocess.run(
            ["/bin/bash", str(SCHEDULER)],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_stdout, first_stderr = first.communicate(timeout=5)

        assert first.returncode == 2, (first_stdout, first_stderr)
        assert contender.returncode == 2, contender.stderr
        assert contender_triggered.is_file()
        assert not target_lock.exists()
    finally:
        allow_release.touch(exist_ok=True)
        _terminate_and_reap(first)


@pytest.mark.parametrize(
    "lock_name",
    ["acquisition.gate", "active.lock"],
    ids=["acquisition-gate", "main-lock"],
)
def test_installer_retries_when_normal_lock_release_removes_contended_path(
    tmp_path: Path,
    lock_name: str,
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    installer_source = installer.read_text(encoding="utf-8")
    acquisition_complete = """if (( crontab_install_status != 0 )); then
    exit "$crontab_install_status"
fi
"""
    assert installer_source.count(acquisition_complete) == 1
    installer.write_text(
        installer_source.replace(
            acquisition_complete,
            acquisition_complete
            + """if [[ "${TEST_EXIT_AFTER_LOCK_ACQUIRE:-}" == "1" ]]; then
    exit 0
fi
""",
            1,
        ),
        encoding="utf-8",
    )

    lock_root = _installer_lock_root(_installer_test_account_home(tmp_path))
    target_lock = lock_root / lock_name
    release_entered = tmp_path / "release-entered"
    allow_release = tmp_path / "allow-release"
    contender_triggered = tmp_path / "contender-triggered"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
[[ "${1:-}" == "-l" ]]
cat "$CRONTAB_STATE"
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    real_mkdir = shutil.which("mkdir")
    real_rmdir = shutil.which("rmdir")
    assert real_mkdir is not None and real_rmdir is not None
    fake_mkdir = bin_dir / "mkdir"
    fake_mkdir.write_text(
        """#!/bin/bash
set -euo pipefail
set +e
"$REAL_MKDIR" "$@"
status=$?
set -e
target="${!#}"
if (( status != 0 )) && [[ "$target" == "$TARGET_LOCK" \
    && -e "$RELEASE_ENTERED" && ! -e "$CONTENDER_TRIGGERED" ]]; then
    : > "$CONTENDER_TRIGGERED"
    : > "$ALLOW_RELEASE"
    while [[ -e "$TARGET_LOCK" || -L "$TARGET_LOCK" ]]; do
        /bin/sleep 0.01
    done
fi
exit "$status"
""",
        encoding="utf-8",
    )
    fake_mkdir.chmod(0o755)
    fake_rmdir = bin_dir / "rmdir"
    fake_rmdir.write_text(
        """#!/bin/bash
set -euo pipefail
target="${!#}"
if [[ "$target" == "$TARGET_LOCK" && ! -e "$RELEASE_ENTERED" ]]; then
    [[ ! -e "$TARGET_LOCK/owner" ]]
    : > "$RELEASE_ENTERED"
    while [[ ! -e "$ALLOW_RELEASE" ]]; do
        /bin/sleep 0.01
    done
fi
exec "$REAL_RMDIR" "$@"
""",
        encoding="utf-8",
    )
    fake_rmdir.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["TEST_EXIT_AFTER_LOCK_ACQUIRE"] = "1"
    env["TARGET_LOCK"] = str(target_lock)
    env["RELEASE_ENTERED"] = str(release_entered)
    env["ALLOW_RELEASE"] = str(allow_release)
    env["CONTENDER_TRIGGERED"] = str(contender_triggered)
    env["REAL_MKDIR"] = real_mkdir
    env["REAL_RMDIR"] = real_rmdir

    first = subprocess.Popen(
        ["/bin/bash", str(installer)],
        cwd=repo_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_path(release_entered, process=first)
        contender = subprocess.run(
            ["/bin/bash", str(installer)],
            cwd=repo_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_stdout, first_stderr = first.communicate(timeout=5)

        assert first.returncode == 0, (first_stdout, first_stderr)
        assert contender.returncode in {0, 75}, contender.stderr
        assert contender_triggered.is_file()
        assert contender.returncode != 2
    finally:
        allow_release.touch(exist_ok=True)
        _terminate_and_reap(first)


def test_installer_execs_crontab_as_lock_owner_and_retry_cannot_overlap_writer(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    lock_dir = _installer_lock_root(_installer_test_account_home(tmp_path)) / "active.lock"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)

    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    writer_pid_file = tmp_path / "first-writer.pid"
    writer_fd_state = tmp_path / "first-writer-fd5"
    first_started = tmp_path / "first-writer-started"
    first_block = tmp_path / "first-writer-block"
    overlap_marker = tmp_path / "overlapping-writer"
    first_block.touch()
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    if [[ ! -e "$FIRST_WRITER_STARTED" ]]; then
        printf '%s\n' "$$" > "$WRITER_PID_FILE"
        if [[ -e /dev/fd/5 ]]; then
            printf 'open\n' > "$WRITER_FD_STATE"
        else
            printf 'closed\n' > "$WRITER_FD_STATE"
        fi
        : > "$FIRST_WRITER_STARTED"
        while [[ -e "$FIRST_WRITER_BLOCK" ]]; do
            /bin/sleep 0.01
        done
    else
        first_writer="$(<"$WRITER_PID_FILE")"
        if kill -0 "$first_writer" 2>/dev/null; then
            : > "$OVERLAP_MARKER"
            exit 91
        fi
    fi
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["WRITER_PID_FILE"] = str(writer_pid_file)
    env["WRITER_FD_STATE"] = str(writer_fd_state)
    env["FIRST_WRITER_STARTED"] = str(first_started)
    env["FIRST_WRITER_BLOCK"] = str(first_block)
    env["OVERLAP_MARKER"] = str(overlap_marker)

    first = subprocess.Popen(
        ["/bin/bash", str(installer)],
        cwd=repo_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    writer_pid: int | None = None
    try:
        _wait_for_path(first_started, process=first)
        _wait_for_path(lock_dir / "owner", process=first)
        writer_pid = int(writer_pid_file.read_text(encoding="utf-8").strip())
        owner_pid = int((lock_dir / "owner").read_text(encoding="utf-8").split(maxsplit=1)[0])

        assert writer_pid == first.pid
        assert owner_pid == first.pid
        assert writer_fd_state.read_text(encoding="utf-8") == "closed\n"

        first.kill()
        first.wait(timeout=2)
        assert first.returncode != 0
        with pytest.raises(ProcessLookupError):
            os.kill(writer_pid, 0)

        retry = subprocess.run(
            ["/bin/bash", str(installer)],
            cwd=repo_dir,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert retry.returncode == 0, retry.stderr
        assert not overlap_marker.exists()
        assert "# BEGIN leveraged-trader managed schedule" in crontab_state.read_text(encoding="utf-8")
    finally:
        first_block.unlink(missing_ok=True)
        _terminate_and_reap(first)
        if writer_pid is not None:
            try:
                os.kill(writer_pid, 0)
            except ProcessLookupError:
                pass
            else:
                os.kill(writer_pid, 9)


def test_scheduler_terminal_cleanup_does_not_remove_replacement_lock_owner(tmp_path: Path) -> None:
    log_file = tmp_path / "cron.log"
    lock_dir = tmp_path / "cron.log.lock.d"
    retired_lock = tmp_path / "retired-scheduler-lock"
    swap_marker = tmp_path / "scheduler-release-swapped"
    replacement_owner = "replacement scheduler owner\n"
    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mkdir", "mv", "ps", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)

    real_mkdir = shutil.which("mkdir")
    real_mv = shutil.which("mv")
    real_rm = shutil.which("rm")
    assert real_mkdir is not None and real_mv is not None and real_rm is not None
    fake_rm = bin_dir / "rm"
    fake_rm.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$PWD" == "$LOCK_DIR" && "$*" == *"./owner"* && ! -e "$SWAP_MARKER" ]]; then
    "$REAL_MV" "$LOCK_DIR" "$RETIRED_LOCK"
    "$REAL_MKDIR" "$LOCK_DIR"
    printf '%s' "$REPLACEMENT_OWNER" > "$LOCK_DIR/owner"
    : > "$SWAP_MARKER"
fi
exec "$REAL_RM" "$@"
""",
        encoding="utf-8",
    )
    fake_rm.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = "/bin/echo"
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["LEVERAGED_TRADER_LOG_MAX_BYTES"] = "invalid"
    env["LOCK_DIR"] = str(lock_dir)
    env["RETIRED_LOCK"] = str(retired_lock)
    env["SWAP_MARKER"] = str(swap_marker)
    env["REPLACEMENT_OWNER"] = replacement_owner
    env["REAL_MKDIR"] = real_mkdir
    env["REAL_MV"] = real_mv
    env["REAL_RM"] = real_rm
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert swap_marker.is_file()
    assert (lock_dir / "owner").read_text(encoding="utf-8") == replacement_owner
    assert not (retired_lock / "owner").exists()
    assert "Cannot safely release scheduler fallback lock" in result.stderr


def test_installer_terminal_cleanup_does_not_remove_replacement_lock_owner(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    installer_source = installer.read_text(encoding="utf-8")
    acquisition_complete = """if (( crontab_install_status != 0 )); then
    exit "$crontab_install_status"
fi
"""
    assert installer_source.count(acquisition_complete) == 1
    installer.write_text(
        installer_source.replace(
            acquisition_complete,
            acquisition_complete + "exit 0\n",
            1,
        ),
        encoding="utf-8",
    )

    lock_dir = _installer_lock_root(_installer_test_account_home(tmp_path)) / "active.lock"
    retired_lock = lock_dir.parent / "retired-installer-lock"
    swap_marker = tmp_path / "installer-release-swapped"
    replacement_owner = "replacement installer owner\n"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
[[ "${1:-}" == "-l" ]]
cat "$CRONTAB_STATE"
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)
    real_mkdir = shutil.which("mkdir")
    real_mv = shutil.which("mv")
    real_rm = shutil.which("rm")
    assert real_mkdir is not None and real_mv is not None and real_rm is not None
    fake_rm = bin_dir / "rm"
    fake_rm.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$PWD" == "$LOCK_DIR" && "$*" == *"./owner"* && ! -e "$SWAP_MARKER" ]]; then
    "$REAL_MV" "$LOCK_DIR" "$RETIRED_LOCK"
    "$REAL_MKDIR" -m 700 "$LOCK_DIR"
    printf '%s' "$REPLACEMENT_OWNER" > "$LOCK_DIR/owner"
    : > "$SWAP_MARKER"
fi
exec "$REAL_RM" "$@"
""",
        encoding="utf-8",
    )
    fake_rm.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["LOCK_DIR"] = str(lock_dir)
    env["RETIRED_LOCK"] = str(retired_lock)
    env["SWAP_MARKER"] = str(swap_marker)
    env["REPLACEMENT_OWNER"] = replacement_owner
    env["REAL_MKDIR"] = real_mkdir
    env["REAL_MV"] = real_mv
    env["REAL_RM"] = real_rm
    result = subprocess.run(
        ["/bin/bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert swap_marker.is_file()
    assert (lock_dir / "owner").read_text(encoding="utf-8") == replacement_owner
    assert not (retired_lock / "owner").exists()
    assert "Cannot safely release crontab installer lock" in result.stderr


@pytest.mark.parametrize(
    "lock_suffix",
    ["lock.acquire.d", "lock.d"],
    ids=["acquisition-gate", "main-lock"],
)
def test_scheduler_persistent_absent_lock_mkdir_failure_is_fatal(
    tmp_path: Path,
    lock_suffix: str,
) -> None:
    log_file = tmp_path / "cron.log"
    target_lock = tmp_path / f"cron.log.{lock_suffix}"
    runner_called = tmp_path / "runner-called"
    runner = tmp_path / "runner"
    runner.write_text('#!/bin/bash\n: > "$RUNNER_CALLED"\n', encoding="utf-8")
    runner.chmod(0o755)
    bin_dir = tmp_path / "bin-without-flock"
    bin_dir.mkdir()
    for command in ["chmod", "dirname", "mv", "ps", "rm", "rmdir", "stat", "wc"]:
        executable = shutil.which(command)
        assert executable is not None
        (bin_dir / command).symlink_to(executable)
    real_mkdir = shutil.which("mkdir")
    assert real_mkdir is not None
    fake_mkdir = bin_dir / "mkdir"
    fake_mkdir.write_text(
        """#!/bin/bash
set -euo pipefail
target="${!#}"
if [[ "$target" == "$TARGET_LOCK" ]]; then
    exit 73
fi
exec "$REAL_MKDIR" "$@"
""",
        encoding="utf-8",
    )
    fake_mkdir.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["LEVERAGED_TRADER_SCHEDULE_NOW"] = "3 09:30"
    env["LEVERAGED_TRADER_RUNNER"] = str(runner)
    env["LEVERAGED_TRADER_LOG_FILE"] = str(log_file)
    env["RUNNER_CALLED"] = str(runner_called)
    env["TARGET_LOCK"] = str(target_lock)
    env["REAL_MKDIR"] = real_mkdir
    result = subprocess.run(
        ["/bin/bash", str(SCHEDULER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Cannot create scheduler fallback" in result.stderr
    assert not runner_called.exists()
    assert not target_lock.exists()


@pytest.mark.parametrize(
    "lock_name",
    ["acquisition.gate", "active.lock"],
    ids=["acquisition-gate", "main-lock"],
)
def test_installer_persistent_absent_lock_mkdir_failure_is_fatal(
    tmp_path: Path,
    lock_name: str,
) -> None:
    repo_dir = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts" / "cron", repo_dir / "scripts" / "cron")
    installer = repo_dir / "scripts" / "cron" / "install-crontab"
    target_lock = _installer_lock_root(_installer_test_account_home(tmp_path)) / lock_name
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_uv.chmod(0o755)
    crontab_state = tmp_path / "crontab"
    crontab_state.write_text("MAILTO=ops@example.com\n", encoding="utf-8")
    write_marker = tmp_path / "write-attempted"
    fake_crontab = bin_dir / "crontab"
    fake_crontab.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-l" ]]; then
    cat "$CRONTAB_STATE"
elif [[ "${1:-}" == "-" ]]; then
    : > "$WRITE_MARKER"
    cat > "$CRONTAB_STATE"
else
    exit 2
fi
""",
        encoding="utf-8",
    )
    fake_crontab.chmod(0o755)
    real_mkdir = shutil.which("mkdir")
    assert real_mkdir is not None
    fake_mkdir = bin_dir / "mkdir"
    fake_mkdir.write_text(
        """#!/bin/bash
set -euo pipefail
target="${!#}"
if [[ "$target" == "$TARGET_LOCK" ]]; then
    exit 73
fi
exec "$REAL_MKDIR" "$@"
""",
        encoding="utf-8",
    )
    fake_mkdir.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["HOME"] = str(tmp_path)
    env["CRONTAB_STATE"] = str(crontab_state)
    env["WRITE_MARKER"] = str(write_marker)
    env["TARGET_LOCK"] = str(target_lock)
    env["REAL_MKDIR"] = real_mkdir
    result = subprocess.run(
        ["/bin/bash", str(installer)],
        cwd=repo_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Cannot create the crontab installer" in result.stderr
    assert not write_marker.exists()
    assert not target_lock.exists()
