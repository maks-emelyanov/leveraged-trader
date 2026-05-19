from __future__ import annotations

import unittest

from leveraged_trader.universe import infer_rsi_symbol


class UniverseTests(unittest.TestCase):
    def test_infers_underlying_symbol_from_leveraged_name(self) -> None:
        self.assertEqual(infer_rsi_symbol("TQQQ", "ProShares UltraPro QQQ"), "QQQ")

    def test_falls_back_to_asset_symbol_when_no_underlying_is_found(self) -> None:
        self.assertEqual(infer_rsi_symbol("XYZ", "Plain Fund Name"), "XYZ")


if __name__ == "__main__":
    unittest.main()
