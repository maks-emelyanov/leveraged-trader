from __future__ import annotations

import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from leveraged_trader.alpaca import _alpaca_headers
from leveraged_trader.cli import parse_args
from leveraged_trader.config import AlpacaOrderConfig, load_dotenv


class ConfigTests(unittest.TestCase):
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
