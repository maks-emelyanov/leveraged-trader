from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from leveraged_trader.alpaca import AlpacaClient, _alpaca_headers
from leveraged_trader.cli import main, parse_args
from leveraged_trader.config import (
    AlpacaOrderConfig,
    BacktestConfig,
    TradierMarketDataConfig,
    UniverseConfig,
    _load_dotenv_values,
    load_dotenv,
    validate_alpaca_paper_endpoint,
    validate_alpaca_reconciliation_configuration,
    validate_backtest_configuration,
    validate_runtime_configuration,
    validate_strategy_simulation_configuration,
)


class ConfigTests(unittest.TestCase):
    def test_load_dotenv_ignores_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            load_dotenv(str(Path(tmp) / ".env"))

    def test_order_submission_rejects_non_paper_alpaca_endpoint(self) -> None:
        cfg = AlpacaOrderConfig(
            enabled=True,
            base_url="https://api.alpaca.markets",
        )

        with self.assertRaisesRegex(ValueError, "restricted to https://paper-api.alpaca.markets"):
            validate_alpaca_paper_endpoint(cfg)

        with self.assertRaisesRegex(ValueError, "restricted to https://paper-api.alpaca.markets"):
            AlpacaClient(cfg)

    def test_non_paper_endpoint_is_allowed_when_order_submission_is_disabled(self) -> None:
        cfg = AlpacaOrderConfig(
            api_key_id="key",
            api_secret_key="secret",
            base_url="https://api.alpaca.markets",
        )

        validate_alpaca_paper_endpoint(cfg)
        with self.assertRaisesRegex(ValueError, "restricted to https://paper-api.alpaca.markets"):
            AlpacaClient(cfg)

    def test_alpaca_endpoint_validation_rejects_non_root_paper_urls(self) -> None:
        for base_url in [
            "https://paper-api.alpaca.markets:8443",
            "https://paper-api.alpaca.markets/custom",
            "https://user:secret@paper-api.alpaca.markets",
            "https://paper-api.alpaca.markets?mode=paper",
            "https://paper-api.alpaca.markets#paper",
        ]:
            with (
                self.subTest(base_url=base_url),
                self.assertRaisesRegex(ValueError, "restricted to https://paper-api.alpaca.markets"),
            ):
                AlpacaClient(
                    AlpacaOrderConfig(
                        api_key_id="key",
                        api_secret_key="secret",
                        base_url=base_url,
                    )
                )

    def test_alpaca_endpoint_validation_allows_one_trailing_slash(self) -> None:
        client = AlpacaClient(
            AlpacaOrderConfig(
                api_key_id="key",
                api_secret_key="secret",
                base_url="https://paper-api.alpaca.markets/",
            )
        )

        self.assertEqual(client.base_url, "https://paper-api.alpaca.markets")

    def test_alpaca_endpoint_validation_rejects_string_subclasses_before_canonicalization(self) -> None:
        class RedirectingString(str):
            def rstrip(self, _characters: str | None = None) -> str:
                return "https://api.alpaca.markets"

        cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="key",
            api_secret_key="secret",
            base_url=RedirectingString("https://paper-api.alpaca.markets/"),
        )

        for validate in (
            lambda: validate_alpaca_paper_endpoint(cfg),
            lambda: AlpacaClient(cfg),
        ):
            with self.subTest(validate=validate), self.assertRaisesRegex(ValueError, "base_url must be a string"):
                validate()

    def test_workflow_concurrency_defaults_to_four_and_accepts_override(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["leveraged-trader"]),
        ):
            default_args = parse_args()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["leveraged-trader", "--workflow-concurrency", "8"]),
        ):
            overridden_args = parse_args()

        self.assertEqual(default_args.workflow_concurrency, 4)
        self.assertEqual(overridden_args.workflow_concurrency, 8)

    def test_paper_order_submission_is_explicit_opt_in(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["leveraged-trader"]),
        ):
            args = parse_args()

        self.assertFalse(args.alpaca_submit_buy_orders)
        self.assertFalse(args.alpaca_submit_sell_orders)
        self.assertTrue(args.auto_adjust)

    def test_cli_rejects_nonpersistent_or_control_bearing_runtime_paths(self) -> None:
        cases = (
            ("--db", ":memory:", "persistent filesystem path"),
            ("--db", "state\nforged.sqlite", "control characters"),
            ("--db", "state\x1b[2J.sqlite", "control characters"),
            ("--db", "state\x00.sqlite", "control characters"),
            ("--output-dir", "reports\rforged", "control characters"),
            ("--output-dir", "reports\tforged", "control characters"),
        )

        for option, value, expected_message in cases:
            stderr = StringIO()
            with (
                self.subTest(option=option, value=repr(value)),
                patch.dict(os.environ, {}, clear=True),
                patch.object(sys, "argv", ["leveraged-trader", option, value]),
                patch("sys.stderr", stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                parse_args()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn(expected_message, stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_rejects_runtime_paths_with_the_wrong_existing_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "report-target"
            output_file.write_text("not a directory")
            cases = (
                ("--db", tmp, "regular database file"),
                ("--output-dir", str(output_file), "regular directory"),
            )

            for option, value, expected_message in cases:
                stderr = StringIO()
                with (
                    self.subTest(option=option),
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(sys, "argv", ["leveraged-trader", option, value]),
                    patch("sys.stderr", stderr),
                    self.assertRaises(SystemExit) as raised,
                ):
                    parse_args()

                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected_message, stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    @patch("leveraged_trader.cli.run_resumable_optimizations")
    @patch("leveraged_trader.cli.load_dotenv")
    def test_main_rejects_memory_database_before_entering_workflow(
        self,
        _mock_load_dotenv: object,
        mock_run: object,
    ) -> None:
        stderr = StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["leveraged-trader", "--db", ":memory:"]),
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("persistent filesystem path", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        mock_run.assert_not_called()

    def test_auto_adjust_can_be_disabled_for_tradier_compatibility(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["leveraged-trader", "--no-auto-adjust"]),
        ):
            args = parse_args()

        self.assertFalse(args.auto_adjust)

    @patch("leveraged_trader.cli.run_resumable_optimizations")
    @patch("leveraged_trader.cli.load_dotenv")
    def test_main_passes_unadjusted_mode_to_the_strategy_fingerprint_config(
        self,
        _mock_load_dotenv: object,
        mock_run: object,
    ) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["leveraged-trader", "--no-auto-adjust"]),
        ):
            main()

        self.assertFalse(mock_run.call_args.kwargs["base_cfg"].auto_adjust)

    def test_invalid_environment_values_are_rejected_instead_of_defaulted(self) -> None:
        invalid_values = {
            "ALPACA_GTC_SELL_RENEWAL_ENABLED": "flase",
            "ALPACA_BUY_LIMIT_BUFFER_BPS": "wide",
            "ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION": "soon",
            "TRADIER_TIMEOUT_SECONDS": "never",
        }
        for name, value in invalid_values.items():
            with (
                self.subTest(name=name),
                patch.dict(os.environ, {name: value}, clear=True),
                patch.object(sys, "argv", ["leveraged-trader"]),
                patch("sys.stderr", new_callable=StringIO),
                self.assertRaises(SystemExit) as raised,
            ):
                parse_args()
            self.assertEqual(raised.exception.code, 2)

    def test_entrypoint_help_ignores_malformed_environment_defaults(self) -> None:
        malformed_defaults = {
            "ALPACA_BUY_LIMIT_BUFFER_BPS": "not-a-number",
            "TRADIER_FALLBACK_ENABLED": "not-a-boolean",
        }
        env_backed_options = {
            "ALPACA_BATCH_CASH_FRACTION",
            "ALPACA_BUY_LIMIT_BUFFER_BPS",
            "ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION",
            "ALPACA_GTC_SELL_RENEWAL_ENABLED",
            "TRADIER_FALLBACK_ENABLED",
            "TRADIER_TIMEOUT_SECONDS",
        }
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            for name, value in malformed_defaults.items():
                for help_option in ("-h", "--help"):
                    env = os.environ.copy()
                    for env_name in env_backed_options:
                        env.pop(env_name, None)
                    env[name] = value
                    env["PYTHONPATH"] = os.pathsep.join(
                        path for path in (str(repo_root), env.get("PYTHONPATH", "")) if path
                    )

                    result = subprocess.run(
                        [sys.executable, "-m", "leveraged_trader", help_option],
                        cwd=tmp,
                        env=env,
                        capture_output=True,
                        text=True,
                    )

                    with self.subTest(name=name, help_option=help_option):
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("usage:", result.stdout)
                        self.assertIn("--alpaca-buy-limit-buffer-bps", result.stdout)
                        self.assertNotIn("must be", result.stderr)

    def test_entrypoint_help_does_not_read_local_dotenv(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_bytes(b"INVALID_UTF8=\xff\n")
            env_path.chmod(0o644)
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(path for path in (str(repo_root), env.get("PYTHONPATH", "")) if path)

            result = subprocess.run(
                [sys.executable, "-m", "leveraged_trader", "--help"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)
            self.assertEqual(result.stderr, "")

    def test_entrypoint_normalizes_local_dotenv_read_errors(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cases = [
            (b"SECRET=exposed\n", 0o644, "chmod 600"),
            (b"INVALID_UTF8=\xff\n", 0o600, "utf-8"),
            (b"VALID_BEFORE_BAD=must_not_be_applied\nBAD=embedded\0value\n", 0o600, "NUL bytes"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(path for path in (str(repo_root), env.get("PYTHONPATH", "")) if path)
            for contents, mode, expected_message in cases:
                with self.subTest(expected_message=expected_message):
                    env_path.write_bytes(contents)
                    env_path.chmod(mode)
                    result = subprocess.run(
                        [sys.executable, "-m", "leveraged_trader"],
                        cwd=tmp,
                        env=env,
                        capture_output=True,
                        text=True,
                    )

                    self.assertEqual(result.returncode, 1)
                    self.assertIn(expected_message, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_explicit_cli_values_override_invalid_environment_defaults(self) -> None:
        cases = [
            (
                {"ALPACA_BUY_LIMIT_BUFFER_BPS": "not-a-number"},
                ["--alpaca-buy-limit-buffer-bps", "250"],
                "alpaca_buy_limit_buffer_bps",
                250.0,
            ),
            (
                {"TRADIER_FALLBACK_ENABLED": "not-a-boolean"},
                ["--tradier-fallback"],
                "tradier_fallback",
                True,
            ),
            (
                {"TRADIER_FALLBACK_ENABLED": "not-a-boolean"},
                ["--no-tradier-fallback"],
                "tradier_fallback",
                False,
            ),
            (
                {"TRADIER_TIMEOUT_SECONDS": "not-an-integer"},
                ["--tradier-timeout-seconds", "45"],
                "tradier_timeout_seconds",
                45,
            ),
            (
                {"TRADIER_TIMEOUT_SECONDS": "not-an-integer"},
                ["--no-tradier-fallback"],
                "tradier_timeout_seconds",
                TradierMarketDataConfig.timeout_seconds,
            ),
        ]
        for environment, arguments, attribute, expected in cases:
            with (
                self.subTest(arguments=arguments),
                patch.dict(os.environ, environment, clear=True),
                patch.object(sys, "argv", ["leveraged-trader", *arguments]),
            ):
                args = parse_args()

            self.assertEqual(getattr(args, attribute), expected)

    def test_cli_rejects_unsafe_numeric_configuration(self) -> None:
        for arguments in [
            ["--alpaca-timeout-seconds", "0"],
            ["--alpaca-buy-limit-buffer-bps", "nan"],
            ["--alpaca-buy-limit-buffer-bps", "10001"],
            ["--alpaca-gtc-sell-renewal-days-before-expiration", "-1"],
            ["--tradier-timeout-seconds", "601"],
            ["--workflow-concurrency", "0"],
        ]:
            with (
                self.subTest(arguments=arguments),
                patch.dict(os.environ, {}, clear=True),
                patch.object(sys, "argv", ["leveraged-trader", *arguments]),
                patch("sys.stderr", new_callable=StringIO),
                self.assertRaises(SystemExit) as raised,
            ):
                parse_args()
            self.assertEqual(raised.exception.code, 2)

    def test_reconcile_only_rejects_buy_submission(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                sys,
                "argv",
                ["leveraged-trader", "--reconcile-only", "--alpaca-submit-buy-orders"],
            ),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_reconcile_only_requires_sell_submission(self) -> None:
        stderr = StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["leveraged-trader", "--reconcile-only"]),
            patch("sys.stderr", stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--reconcile-only requires --alpaca-submit-sell-orders", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_reconcile_only_does_not_accept_an_abbreviated_mode_flag(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["leveraged-trader", "--reconcile", "--alpaca-submit-sell-orders"]),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_reconcile_only_ignores_unrelated_environment_configuration(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "ALPACA_API_KEY_ID": "paper-key",
                    "ALPACA_API_SECRET_KEY": "paper-secret",
                    "ALPACA_BUY_LIMIT_BUFFER_BPS": "not-a-number",
                    "ALPACA_BATCH_CASH_FRACTION": "obsolete",
                    "TRADIER_FALLBACK_ENABLED": "not-a-boolean",
                    "TRADIER_TIMEOUT_SECONDS": "not-an-integer",
                },
                clear=True,
            ),
            patch.object(
                sys,
                "argv",
                ["leveraged-trader", "--reconcile-only", "--alpaca-submit-sell-orders"],
            ),
        ):
            args = parse_args()

        self.assertTrue(args.reconcile_only)
        self.assertEqual(args.alpaca_buy_limit_buffer_bps, AlpacaOrderConfig.buy_limit_buffer_bps)
        self.assertEqual(args.tradier_timeout_seconds, TradierMarketDataConfig.timeout_seconds)

    def test_disabled_sell_renewal_ignores_invalid_environment_day_count(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "ALPACA_API_KEY_ID": "paper-key",
                    "ALPACA_API_SECRET_KEY": "paper-secret",
                    "ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION": "not-a-day-count",
                },
                clear=True,
            ),
            patch.object(
                sys,
                "argv",
                [
                    "leveraged-trader",
                    "--reconcile-only",
                    "--alpaca-submit-sell-orders",
                    "--no-alpaca-gtc-sell-renewal",
                ],
            ),
        ):
            args = parse_args()

        self.assertTrue(args.reconcile_only)
        self.assertFalse(args.alpaca_gtc_sell_renewal)
        self.assertEqual(
            args.alpaca_gtc_sell_renewal_days_before_expiration,
            AlpacaOrderConfig.gtc_sell_renewal_days_before_expiration,
        )

    def test_explicit_sell_renewal_day_count_overrides_invalid_environment_value(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION": "not-a-day-count"},
                clear=True,
            ),
            patch.object(
                sys,
                "argv",
                [
                    "leveraged-trader",
                    "--alpaca-gtc-sell-renewal-days-before-expiration",
                    "11",
                ],
            ),
        ):
            args = parse_args()

        self.assertEqual(args.alpaca_gtc_sell_renewal_days_before_expiration, 11)

    def test_last_explicit_sell_renewal_toggle_controls_environment_day_validation(self) -> None:
        environment = {"ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION": "not-a-day-count"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                sys,
                "argv",
                [
                    "leveraged-trader",
                    "--alpaca-gtc-sell-renewal",
                    "--no-alpaca-gtc-sell-renewal",
                ],
            ),
        ):
            disabled_args = parse_args()

        self.assertFalse(disabled_args.alpaca_gtc_sell_renewal)

        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                sys,
                "argv",
                [
                    "leveraged-trader",
                    "--no-alpaca-gtc-sell-renewal",
                    "--alpaca-gtc-sell-renewal",
                ],
            ),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_reconciliation_validation_skips_buy_only_settings(self) -> None:
        validate_alpaca_reconciliation_configuration(
            AlpacaOrderConfig(
                sell_enabled=True,
                api_key_id="paper-key",
                api_secret_key="paper-secret",
                buy_limit_buffer_bps=float("nan"),
            )
        )

    def test_reconciliation_validation_skips_disabled_sell_renewal_day_count(self) -> None:
        validate_alpaca_reconciliation_configuration(
            AlpacaOrderConfig(
                gtc_sell_renewal_enabled=False,
                gtc_sell_renewal_days_before_expiration="not-a-day-count",  # type: ignore[arg-type]
            )
        )

    @patch("leveraged_trader.cli.run_alpaca_reconciliation")
    @patch("leveraged_trader.cli.load_dotenv")
    def test_reconcile_only_main_does_not_construct_unrelated_runtime_configuration(
        self,
        _mock_load_dotenv: object,
        mock_reconcile: object,
    ) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "ALPACA_API_KEY_ID": "paper-key",
                    "ALPACA_API_SECRET_KEY": "paper-secret",
                    "ALPACA_BUY_LIMIT_BUFFER_BPS": "invalid",
                    "TRADIER_TIMEOUT_SECONDS": "invalid",
                },
                clear=True,
            ),
            patch.object(
                sys,
                "argv",
                ["leveraged-trader", "--reconcile-only", "--alpaca-submit-sell-orders"],
            ),
        ):
            main()

        mock_reconcile.assert_called_once()

    def test_enabled_alpaca_submission_requires_environment_credentials(self) -> None:
        for submission_flag in ("--alpaca-submit-buy-orders", "--alpaca-submit-sell-orders"):
            stderr = StringIO()
            with (
                self.subTest(submission_flag=submission_flag),
                patch.dict(os.environ, {}, clear=True),
                patch.object(sys, "argv", ["leveraged-trader", submission_flag]),
                patch("sys.stderr", stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                parse_args()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_invalid_enabled_broker_configuration_is_reported_by_argparse(self) -> None:
        cases = [
            (
                {
                    "ALPACA_API_KEY_ID": "paper-key",
                    "ALPACA_API_SECRET_KEY": "paper-secret",
                },
                ["--alpaca-submit-sell-orders", "--alpaca-base-url", "https://api.alpaca.markets"],
                "restricted to https://paper-api.alpaca.markets",
            ),
            (
                {
                    "TRADIER_ACCESS_TOKEN": "tradier-secret",
                },
                ["--tradier-base-url", "https://example.com/v1"],
                "Tradier bearer credentials are restricted",
            ),
        ]
        for environment, arguments, expected_message in cases:
            stderr = StringIO()
            with (
                self.subTest(arguments=arguments),
                patch.dict(os.environ, environment, clear=True),
                patch.object(sys, "argv", ["leveraged-trader", *arguments]),
                patch("sys.stderr", stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                parse_args()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn(expected_message, stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_timezone_import_works_without_system_zoneinfo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_zoneinfo_path = Path(tmp) / "missing-zoneinfo"
            env = os.environ.copy()
            env["PYTHONTZPATH"] = str(missing_zoneinfo_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from zoneinfo import TZPATH; "
                        "assert len(TZPATH) == 1 and not __import__('pathlib').Path(TZPATH[0]).exists(), TZPATH; "
                        "import leveraged_trader.alpaca; "
                        "print(leveraged_trader.alpaca._NEW_YORK.key)"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "America/New_York")

    def test_secret_bearing_cli_flags_are_rejected(self) -> None:
        for option, value in [
            ("--alpaca-api-key-id", "SENSITIVE_KEY_ID_123"),
            ("--alpaca-api-secret-key", "SENSITIVE_ALPACA_SECRET_123"),
            ("--tradier-access-token", "SENSITIVE_TRADIER_TOKEN_123"),
        ]:
            stderr = StringIO()
            with (
                self.subTest(option=option),
                patch.dict(os.environ, {}, clear=True),
                patch.object(sys, "argv", ["leveraged-trader", option, value]),
                patch("sys.stderr", stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                parse_args()

            self.assertEqual(raised.exception.code, 2)
            self.assertNotIn(value, stderr.getvalue())

    @patch("leveraged_trader.cli.run_resumable_optimizations")
    @patch("leveraged_trader.cli.load_dotenv")
    def test_main_reads_and_normalizes_broker_credentials_from_environment(
        self,
        _mock_load_dotenv: object,
        mock_run: object,
    ) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "ALPACA_API_KEY_ID": "  paper-key  ",
                    "ALPACA_API_SECRET_KEY": "  paper-secret  ",
                    "TRADIER_ACCESS_TOKEN": "  tradier-secret  ",
                },
                clear=True,
            ),
            patch.object(sys, "argv", ["leveraged-trader"]),
        ):
            main()

        self.assertEqual(mock_run.call_args.kwargs["alpaca_cfg"].api_key_id, "paper-key")
        self.assertEqual(mock_run.call_args.kwargs["alpaca_cfg"].api_secret_key, "paper-secret")
        self.assertEqual(mock_run.call_args.kwargs["tradier_cfg"].access_token, "tradier-secret")

    @patch("leveraged_trader.cli.run_resumable_optimizations")
    @patch("leveraged_trader.cli.load_dotenv")
    def test_main_rejects_control_bearing_alpaca_environment_credentials_before_workflow(
        self,
        _mock_load_dotenv: object,
        mock_run: object,
    ) -> None:
        cases = (
            ("ALPACA_API_KEY_ID", "\tpaper-key"),
            ("ALPACA_API_KEY_ID", "paper-key\n"),
            ("ALPACA_API_SECRET_KEY", "\rpaper-secret"),
            ("ALPACA_API_SECRET_KEY", "paper-secret\x1b"),
        )
        for environment_name, unsafe_value in cases:
            environment = {
                "ALPACA_API_KEY_ID": "paper-key",
                "ALPACA_API_SECRET_KEY": "paper-secret",
            }
            environment[environment_name] = unsafe_value
            stderr = StringIO()
            with (
                self.subTest(environment_name=environment_name, unsafe_value=repr(unsafe_value)),
                patch.dict(os.environ, environment, clear=True),
                patch.object(sys, "argv", ["leveraged-trader", "--alpaca-submit-buy-orders"]),
                patch("sys.stderr", stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("whitespace or control characters", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

        mock_run.assert_not_called()

    @patch("leveraged_trader.cli.run_resumable_optimizations")
    @patch("leveraged_trader.cli.load_dotenv")
    def test_main_skips_blank_primary_tradier_token_before_alias_selection(
        self,
        _mock_load_dotenv: object,
        mock_run: object,
    ) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "TRADIER_ACCESS_TOKEN": "  ",
                    "TRADIER_API_TOKEN": "  fallback-token  ",
                },
                clear=True,
            ),
            patch.object(sys, "argv", ["leveraged-trader"]),
        ):
            main()

        self.assertEqual(mock_run.call_args.kwargs["tradier_cfg"].access_token, "fallback-token")

    @patch("leveraged_trader.cli.run_resumable_optimizations")
    @patch("leveraged_trader.cli.load_dotenv")
    def test_main_skips_example_primary_tradier_placeholder_before_alias_selection(
        self,
        _mock_load_dotenv: object,
        mock_run: object,
    ) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "TRADIER_ACCESS_TOKEN": "  Your_Tradier_Access_Token  ",
                    "TRADIER_API_TOKEN": "  fallback-token  ",
                },
                clear=True,
            ),
            patch.object(sys, "argv", ["leveraged-trader"]),
        ):
            main()

        self.assertEqual(mock_run.call_args.kwargs["tradier_cfg"].access_token, "fallback-token")

    def test_runtime_validation_fails_before_enabled_trading_without_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires ALPACA_API_KEY_ID"):
            validate_runtime_configuration(
                base_cfg=BacktestConfig(),
                universe_cfg=UniverseConfig(),
                alpaca_cfg=AlpacaOrderConfig(enabled=True),
                tradier_cfg=None,
                workflow_concurrency=4,
            )

    def test_runtime_validation_normalizes_enabled_alpaca_credentials(self) -> None:
        alpaca_cfg = AlpacaOrderConfig(
            enabled=True,
            api_key_id="  paper-key  ",
            api_secret_key="  paper-secret  ",
        )

        validate_runtime_configuration(
            base_cfg=BacktestConfig(),
            universe_cfg=UniverseConfig(),
            alpaca_cfg=alpaca_cfg,
            tradier_cfg=None,
            workflow_concurrency=4,
        )

        self.assertEqual(alpaca_cfg.api_key_id, "paper-key")
        self.assertEqual(alpaca_cfg.api_secret_key, "paper-secret")

    def test_runtime_validation_rejects_blank_or_case_varied_placeholder_alpaca_credentials(self) -> None:
        cases = [
            (" ", "paper-secret", "requires ALPACA_API_KEY_ID"),
            ("paper-key", "\t", "requires ALPACA_API_KEY_ID"),
            (" YOUR_ALPACA_PAPER_API_KEY_ID ", "paper-secret", "placeholder Alpaca credentials"),
            ("paper-key", " Your_Alpaca_Paper_Api_Secret_Key ", "placeholder Alpaca credentials"),
            ("paper-€-key", "paper-secret", "HTTP-header-compatible"),
            ("paper-key", "秘密", "HTTP-header-compatible"),
        ]
        for api_key_id, api_secret_key, expected_message in cases:
            with (
                self.subTest(api_key_id=api_key_id, api_secret_key=api_secret_key),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                validate_runtime_configuration(
                    base_cfg=BacktestConfig(),
                    universe_cfg=UniverseConfig(),
                    alpaca_cfg=AlpacaOrderConfig(
                        enabled=True,
                        api_key_id=api_key_id,
                        api_secret_key=api_secret_key,
                    ),
                    tradier_cfg=None,
                    workflow_concurrency=4,
                )

    def test_runtime_validation_rejects_numeric_strings(self) -> None:
        cases = [
            (
                BacktestConfig(initial_capital="100000"),  # type: ignore[arg-type]
                AlpacaOrderConfig(),
                "initial_capital",
            ),
            (
                BacktestConfig(),
                AlpacaOrderConfig(buy_limit_buffer_bps="500"),  # type: ignore[arg-type]
                "buy_limit_buffer_bps",
            ),
        ]
        for base_cfg, alpaca_cfg, expected_name in cases:
            with self.subTest(expected_name=expected_name), self.assertRaisesRegex(ValueError, expected_name):
                validate_runtime_configuration(
                    base_cfg=base_cfg,
                    universe_cfg=UniverseConfig(),
                    alpaca_cfg=alpaca_cfg,
                    tradier_cfg=None,
                    workflow_concurrency=4,
                )

    def test_runtime_validation_normalizes_real_float_overflow_to_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_capital must be a finite number"):
            validate_runtime_configuration(
                base_cfg=BacktestConfig(initial_capital=Fraction(10**10_000, 1)),
                universe_cfg=UniverseConfig(),
                alpaca_cfg=AlpacaOrderConfig(),
                tradier_cfg=None,
                workflow_concurrency=4,
            )

    def test_profit_target_validation_uses_strict_lower_and_inclusive_upper_bounds(self) -> None:
        def validate_runtime(base_cfg: BacktestConfig) -> None:
            validate_runtime_configuration(
                base_cfg=base_cfg,
                universe_cfg=UniverseConfig(),
                alpaca_cfg=AlpacaOrderConfig(),
                tradier_cfg=None,
                workflow_concurrency=4,
            )

        validators = (
            validate_backtest_configuration,
            validate_strategy_simulation_configuration,
            validate_runtime,
        )
        smallest_float_above_one = np.nextafter(1.0, np.inf)
        smallest_float_above_maximum = np.nextafter(100.0, np.inf)
        nonrepresentable_value_above_one = Fraction(2**53 + 1, 2**53)
        exact_value_above_maximum = Fraction(100 * 2**53 + 1, 2**53)

        for validator in validators:
            with self.subTest(validator=validator.__name__, profit_target_multiple=smallest_float_above_one):
                validator(BacktestConfig(profit_target_multiple=smallest_float_above_one))
            with self.subTest(validator=validator.__name__, profit_target_multiple=100.0):
                validator(BacktestConfig(profit_target_multiple=100.0))
            for invalid_value in (
                1.0,
                nonrepresentable_value_above_one,
                smallest_float_above_maximum,
                exact_value_above_maximum,
            ):
                with (
                    self.subTest(validator=validator.__name__, profit_target_multiple=invalid_value),
                    self.assertRaisesRegex(ValueError, "profit_target_multiple"),
                ):
                    validator(BacktestConfig(profit_target_multiple=invalid_value))

    def test_runtime_validation_rejects_combined_trading_costs_of_one_hundred_percent(self) -> None:
        for fee_bps, slippage_bps in [(10_000.0, 0.0), (4_000.0, 6_000.0), (9_999.0, 2.0)]:
            with (
                self.subTest(fee_bps=fee_bps, slippage_bps=slippage_bps),
                self.assertRaisesRegex(ValueError, "must total less than 10000"),
            ):
                validate_runtime_configuration(
                    base_cfg=BacktestConfig(fee_bps=fee_bps, slippage_bps=slippage_bps),
                    universe_cfg=UniverseConfig(),
                    alpaca_cfg=AlpacaOrderConfig(),
                    tradier_cfg=None,
                    workflow_concurrency=4,
                )

        validate_runtime_configuration(
            base_cfg=BacktestConfig(fee_bps=9_999.0, slippage_bps=0.999),
            universe_cfg=UniverseConfig(),
            alpaca_cfg=AlpacaOrderConfig(),
            tradier_cfg=None,
            workflow_concurrency=4,
        )

    def test_runtime_validation_restricts_tradier_bearer_token_hosts(self) -> None:
        invalid_base_urls = (
            "http://api.tradier.com/v1",
            "https://example.com/v1",
            "https://user:secret@api.tradier.com/v1",
            "https://@api.tradier.com/v1",
            "https://api.tradier.com:/v1",
            "https://api.tradier.com:0443/v1",
            "https://api.tradier.com//",
            "https://api.tradier.com/v1?",
            "https://api.tradier.com/v1#",
            "https://api.tradier.com/v1?mode=history",
            "https://api.tradier.com/v1#history",
        )
        for base_url in invalid_base_urls:
            with self.subTest(base_url=base_url), self.assertRaisesRegex(ValueError, "official"):
                validate_runtime_configuration(
                    base_cfg=BacktestConfig(),
                    universe_cfg=UniverseConfig(),
                    alpaca_cfg=AlpacaOrderConfig(),
                    tradier_cfg=TradierMarketDataConfig(
                        enabled=True,
                        access_token="secret",
                        base_url=base_url,
                    ),
                    workflow_concurrency=4,
                )

    def test_cli_accepts_supported_tradier_api_root_forms(self) -> None:
        supported_base_urls = (
            "https://api.tradier.com",
            "https://api.tradier.com/",
            "https://api.tradier.com/v1",
            "https://api.tradier.com/v1/",
            "https://sandbox.tradier.com",
            "https://sandbox.tradier.com:443/v1/",
        )
        for base_url in supported_base_urls:
            with (
                self.subTest(base_url=base_url),
                patch.dict(
                    os.environ,
                    {
                        "TRADIER_ACCESS_TOKEN": "secret",
                        "TRADIER_BASE_URL": base_url,
                    },
                    clear=True,
                ),
                patch.object(sys, "argv", ["leveraged-trader", "--no-auto-adjust"]),
            ):
                args = parse_args()

            self.assertEqual(args.tradier_base_url, base_url)

    def test_runtime_validation_accepts_case_insensitive_official_tradier_authority(self) -> None:
        validate_runtime_configuration(
            base_cfg=BacktestConfig(),
            universe_cfg=UniverseConfig(),
            alpaca_cfg=AlpacaOrderConfig(),
            tradier_cfg=TradierMarketDataConfig(
                enabled=True,
                access_token="secret",
                base_url="HTTPS://API.TRADIER.COM/v1",
            ),
            workflow_concurrency=4,
        )

    def test_runtime_validation_rejects_whitespace_and_controls_in_tradier_bearer_token(self) -> None:
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
                self.assertRaisesRegex(ValueError, "must not contain whitespace or control characters"),
            ):
                validate_runtime_configuration(
                    base_cfg=BacktestConfig(),
                    universe_cfg=UniverseConfig(),
                    alpaca_cfg=AlpacaOrderConfig(),
                    tradier_cfg=TradierMarketDataConfig(access_token=access_token),
                    workflow_concurrency=4,
                )

    def test_runtime_validation_normalizes_outer_spaces_in_tradier_bearer_token(self) -> None:
        tradier_cfg = TradierMarketDataConfig(access_token="  opaque-token  ")

        validate_runtime_configuration(
            base_cfg=BacktestConfig(),
            universe_cfg=UniverseConfig(),
            alpaca_cfg=AlpacaOrderConfig(),
            tradier_cfg=tradier_cfg,
            workflow_concurrency=4,
        )

        self.assertEqual(tradier_cfg.access_token, "opaque-token")

    def test_runtime_validation_rejects_non_latin_1_tradier_bearer_token(self) -> None:
        for access_token in ("opaque-€-token", "秘密", "😀"):
            with (
                self.subTest(access_token=repr(access_token)),
                self.assertRaisesRegex(ValueError, "HTTP-header-compatible"),
            ):
                validate_runtime_configuration(
                    base_cfg=BacktestConfig(),
                    universe_cfg=UniverseConfig(),
                    alpaca_cfg=AlpacaOrderConfig(),
                    tradier_cfg=TradierMarketDataConfig(access_token=access_token),
                    workflow_concurrency=4,
                )

    def test_runtime_validation_normalizes_latin_1_tradier_bearer_token(self) -> None:
        tradier_cfg = TradierMarketDataConfig(access_token="  opaque-é-token  ")

        validate_runtime_configuration(
            base_cfg=BacktestConfig(),
            universe_cfg=UniverseConfig(),
            alpaca_cfg=AlpacaOrderConfig(),
            tradier_cfg=tradier_cfg,
            workflow_concurrency=4,
        )

        self.assertEqual(tradier_cfg.access_token, "opaque-é-token")

    def test_runtime_validation_rejects_malformed_tradier_string_types(self) -> None:
        class MutableString(str):
            pass

        cases = (
            (TradierMarketDataConfig(access_token=True), "access_token must be a string or null"),
            (TradierMarketDataConfig(access_token=MutableString("secret")), "access_token must be a string or null"),
            (
                TradierMarketDataConfig(access_token="secret", base_url=123),
                "base_url must be a string",
            ),
            (
                TradierMarketDataConfig(access_token="secret", base_url=MutableString("https://api.tradier.com/v1")),
                "base_url must be a string",
            ),
        )
        for tradier_cfg, expected_message in cases:
            with (
                self.subTest(expected_message=expected_message),
                self.assertRaisesRegex(
                    ValueError,
                    expected_message,
                ),
            ):
                validate_runtime_configuration(
                    base_cfg=BacktestConfig(),
                    universe_cfg=UniverseConfig(),
                    alpaca_cfg=AlpacaOrderConfig(),
                    tradier_cfg=tradier_cfg,
                    workflow_concurrency=4,
                )

    def test_runtime_validation_rejects_non_string_alpaca_base_url(self) -> None:
        for enabled in (False, True):
            with self.subTest(enabled=enabled), self.assertRaisesRegex(ValueError, "base_url must be a string"):
                validate_runtime_configuration(
                    base_cfg=BacktestConfig(),
                    universe_cfg=UniverseConfig(),
                    alpaca_cfg=AlpacaOrderConfig(
                        enabled=enabled,
                        api_key_id="key" if enabled else None,
                        api_secret_key="secret" if enabled else None,
                        base_url=123,  # type: ignore[arg-type]
                    ),
                    tradier_cfg=None,
                    workflow_concurrency=4,
                )

    def test_runtime_validation_rejects_truthy_string_behavior_switches(self) -> None:
        cases = [
            (BacktestConfig(auto_adjust="false"), UniverseConfig(), AlpacaOrderConfig(), None),
            (
                BacktestConfig(),
                UniverseConfig(require_workflow_source_success="false"),
                AlpacaOrderConfig(),
                None,
            ),
            (BacktestConfig(), UniverseConfig(), AlpacaOrderConfig(enabled="false"), None),
            (BacktestConfig(), UniverseConfig(), AlpacaOrderConfig(sell_enabled="false"), None),
            (
                BacktestConfig(),
                UniverseConfig(),
                AlpacaOrderConfig(gtc_sell_renewal_enabled="false"),
                None,
            ),
            (BacktestConfig(), UniverseConfig(), AlpacaOrderConfig(), TradierMarketDataConfig(enabled="false")),
        ]
        for base_cfg, universe_cfg, alpaca_cfg, tradier_cfg in cases:
            with (
                self.subTest(
                    base_cfg=base_cfg,
                    universe_cfg=universe_cfg,
                    alpaca_cfg=alpaca_cfg,
                    tradier_cfg=tradier_cfg,
                ),
                self.assertRaisesRegex(ValueError, "must be a boolean"),
            ):
                validate_runtime_configuration(
                    base_cfg=base_cfg,
                    universe_cfg=universe_cfg,
                    alpaca_cfg=alpaca_cfg,
                    tradier_cfg=tradier_cfg,
                    workflow_concurrency=4,
                )

    def test_load_dotenv_does_not_override_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("EXISTING=value_from_file\nNEW_VALUE=ok\nNEW_VALUE=later_duplicate\n")
            env_path.chmod(0o600)

            with patch.dict(os.environ, {"EXISTING": "already_set"}, clear=False):
                load_dotenv(str(env_path))

                self.assertEqual(os.environ["EXISTING"], "already_set")
                self.assertEqual(os.environ["NEW_VALUE"], "ok")

    def test_load_dotenv_removes_one_balanced_outer_quote_pair(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _load_dotenv_values(
                [
                    "SINGLE='single value'\n",
                    'DOUBLE=" double value "\n',
                    "NESTED=\"'literal quotes'\"\n",
                    'REPEATED=""literal pair""\n',
                    "UNQUOTED_TRAILING=credential'\n",
                ]
            )

            self.assertEqual(os.environ["SINGLE"], "single value")
            self.assertEqual(os.environ["DOUBLE"], " double value ")
            self.assertEqual(os.environ["NESTED"], "'literal quotes'")
            self.assertEqual(os.environ["REPEATED"], '"literal pair"')
            self.assertEqual(os.environ["UNQUOTED_TRAILING"], "credential'")

    def test_load_dotenv_rejects_unclosed_or_mismatched_opening_quote_atomically(self) -> None:
        malformed_values = [
            "'unclosed",
            '"unclosed',
            "'mismatched\"",
            "\"mismatched'",
            "'",
            '"',
        ]
        for malformed_value in malformed_values:
            with (
                self.subTest(malformed_value=malformed_value),
                patch.dict(os.environ, {"EXISTING": "preserved"}, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "unmatched opening quote"):
                    _load_dotenv_values(
                        [
                            "VALID_BEFORE_BAD=must-not-be-applied\n",
                            f"MALFORMED={malformed_value}\n",
                        ]
                    )

                self.assertEqual(dict(os.environ), {"EXISTING": "preserved"})

    def test_load_dotenv_rejects_nul_atomically(self) -> None:
        malformed_lines = [
            b"BAD\0KEY=value\n",
            b"BAD=embedded\0value\n",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            for malformed_line in malformed_lines:
                with self.subTest(malformed_line=malformed_line):
                    env_path.write_bytes(b"VALID_BEFORE_BAD=must_not_be_applied\n" + malformed_line)
                    env_path.chmod(0o600)

                    with patch.dict(os.environ, {"EXISTING": "preserved"}, clear=True):
                        with self.assertRaisesRegex(ValueError, "NUL bytes"):
                            load_dotenv(str(env_path))

                        self.assertEqual(dict(os.environ), {"EXISTING": "preserved"})

    def test_load_dotenv_rollback_preserves_concurrent_replacement(self) -> None:
        class FailingEnvironment(dict[str, str]):
            def __setitem__(self, key: str, value: str) -> None:
                if key == "SECOND":
                    dict.__setitem__(self, "FIRST", "concurrent replacement")
                    raise OSError("simulated environment assignment failure")
                super().__setitem__(key, value)

        environment = FailingEnvironment()
        with (
            patch("leveraged_trader.config.os.environ", environment),
            self.assertRaisesRegex(OSError, "simulated environment assignment failure"),
        ):
            _load_dotenv_values(["FIRST=dotenv value\n", "SECOND=dotenv value\n"])

        self.assertEqual(environment, {"FIRST": "concurrent replacement"})

    @unittest.skipUnless(os.name == "posix", "POSIX file modes are required")
    def test_load_dotenv_rejects_group_or_world_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("SECRET=must_not_be_loaded\n", encoding="utf-8")
            env_path.chmod(0o644)

            with (
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(PermissionError, "chmod 600"),
            ):
                load_dotenv(str(env_path))

            self.assertNotIn("SECRET", os.environ)

    @unittest.skipUnless(os.name == "posix", "POSIX file types are required")
    def test_load_dotenv_rejects_owner_only_fifo_without_waiting_for_a_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            os.mkfifo(env_path, mode=0o600)

            with self.assertRaisesRegex(PermissionError, "regular file"):
                load_dotenv(str(env_path))

    @patch("leveraged_trader.cli.load_dotenv", side_effect=PermissionError(".env must be private"))
    def test_main_reports_insecure_dotenv_without_a_traceback(self, _mock_load_dotenv: object) -> None:
        with self.assertRaises(SystemExit) as raised:
            main()

        self.assertEqual(str(raised.exception), ".env must be private")

    @patch("leveraged_trader.cli.load_dotenv", side_effect=ValueError("Environment file contains malformed data"))
    def test_main_reports_malformed_dotenv_without_a_traceback(self, _mock_load_dotenv: object) -> None:
        with self.assertRaises(SystemExit) as raised:
            main()

        self.assertEqual(str(raised.exception), "Environment file contains malformed data")

    def test_placeholder_alpaca_credentials_are_rejected(self) -> None:
        cfg = AlpacaOrderConfig(
            api_key_id="your_alpaca_paper_api_key_id",
            api_secret_key="your_alpaca_paper_api_secret_key",
        )

        with self.assertRaisesRegex(ValueError, "placeholder Alpaca credentials"):
            _alpaca_headers(cfg)

    def test_removed_alpaca_batch_cash_fraction_env_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {"ALPACA_BATCH_CASH_FRACTION": "0.25"}, clear=False),
            patch.object(sys, "argv", ["leveraged-trader"]),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_removed_alpaca_batch_cash_fraction_cli_flag_is_rejected(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["leveraged-trader", "--alpaca-batch-cash-fraction", "0.25"]),
            patch("sys.stderr", new_callable=StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args()

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
