from __future__ import annotations

import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from leveraged_trader.alpaca import AlpacaClient, _alpaca_headers
from leveraged_trader.cli import parse_args
from leveraged_trader.config import AlpacaOrderConfig, load_dotenv, validate_alpaca_paper_endpoint


class ConfigTests(unittest.TestCase):
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

    def test_load_dotenv_does_not_override_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("EXISTING=value_from_file\nNEW_VALUE=ok\n")

            with patch.dict(os.environ, {"EXISTING": "already_set"}, clear=False):
                load_dotenv(str(env_path))

                self.assertEqual(os.environ["EXISTING"], "already_set")
                self.assertEqual(os.environ["NEW_VALUE"], "ok")

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
