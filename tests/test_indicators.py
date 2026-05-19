from __future__ import annotations

import unittest

import pandas as pd

from leveraged_trader.indicators import compute_rsi


class IndicatorTests(unittest.TestCase):
    def test_rsi_reaches_100_for_all_gains_after_warmup(self) -> None:
        close = pd.Series(range(1, 20), dtype=float)
        rsi = compute_rsi(close, period=3)

        self.assertEqual(float(rsi.dropna().iloc[-1]), 100.0)

    def test_rsi_reaches_0_for_all_losses_after_warmup(self) -> None:
        close = pd.Series(range(20, 1, -1), dtype=float)
        rsi = compute_rsi(close, period=3)

        self.assertEqual(float(rsi.dropna().iloc[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
