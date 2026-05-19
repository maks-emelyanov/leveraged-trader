from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from leveraged_trader.alpaca import _alpaca_headers
from leveraged_trader.config import AlpacaOrderConfig, load_dotenv


class ConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
