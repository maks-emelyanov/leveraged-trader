from __future__ import annotations

import unittest

import pandas as pd

from leveraged_trader.indicators import compute_rsi, compute_rsi_details


class IndicatorTests(unittest.TestCase):
    def test_rsi_reaches_100_for_all_gains_after_warmup(self) -> None:
        close = pd.Series(range(1, 20), dtype=float)
        rsi = compute_rsi(close, period=3)

        self.assertEqual(float(rsi.dropna().iloc[-1]), 100.0)

    def test_rsi_reaches_0_for_all_losses_after_warmup(self) -> None:
        close = pd.Series(range(20, 1, -1), dtype=float)
        rsi = compute_rsi(close, period=3)

        self.assertEqual(float(rsi.dropna().iloc[-1]), 0.0)

    def test_rsi_is_neutral_for_flat_prices_after_warmup(self) -> None:
        close = pd.Series([100.0] * 20)

        rsi = compute_rsi(close, period=3)
        details = compute_rsi_details(close, period=3)

        self.assertEqual(float(rsi.dropna().iloc[-1]), 50.0)
        self.assertEqual(float(details["rsi"].dropna().iloc[-1]), 50.0)


if __name__ == "__main__":
    unittest.main()
