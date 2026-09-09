from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from leveraged_trader.backtest import performance_summary


class PerformanceSummaryTests(unittest.TestCase):
    def test_irregular_trading_calendar_does_not_add_synthetic_returns(self) -> None:
        equity = pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-20"]),
        )

        summary = performance_summary(equity)

        self.assertAlmostEqual(summary["Total Return"], 0.21)
        self.assertAlmostEqual(summary["CAGR"], (1.21 ** (252 / 2)) - 1.0)
        self.assertAlmostEqual(summary["Hit Rate"], 1.0)

    def test_missing_equity_is_rejected_instead_of_forward_filled(self) -> None:
        equity = pd.Series([100.0, np.nan, 110.0])

        with self.assertRaisesRegex(ValueError, "equity_curve must not contain missing values"):
            performance_summary(equity)

    def test_non_positive_equity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "equity_curve must contain only positive values"):
            performance_summary(pd.Series([100.0, 0.0]))

    def test_unrepresentable_returns_are_rejected_before_derived_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "equity_curve produces non-finite daily returns"):
            performance_summary(pd.Series([1e-308, 1e308]))

        with self.assertRaisesRegex(ValueError, "non-finite CAGR"):
            performance_summary(pd.Series([1.0, 100.0]))

    def test_unsorted_or_duplicate_observations_are_rejected(self) -> None:
        unsorted = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2026-01-03", "2026-01-02"]),
        )
        duplicate = pd.Series([100.0, 101.0], index=[1, 1])

        with self.assertRaisesRegex(ValueError, "index must be sorted"):
            performance_summary(unsorted)
        with self.assertRaisesRegex(ValueError, "index must not contain duplicate"):
            performance_summary(duplicate)

    def test_infinite_risk_free_return_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "risk_free_returns must not contain infinite"):
            performance_summary(
                pd.Series([100.0, 101.0]),
                pd.Series([0.0, np.inf]),
            )

    def test_risk_free_returns_must_remain_above_total_loss(self) -> None:
        for invalid_return in (-1.0, -1.0000001):
            with (
                self.subTest(invalid_return=invalid_return),
                self.assertRaisesRegex(ValueError, "risk_free_returns must contain values greater than -1.0"),
            ):
                performance_summary(
                    pd.Series([100.0, 100.0, 101.0]),
                    pd.Series([0.0, invalid_return, 0.0]),
                )

        boundary = np.nextafter(-1.0, 0.0)
        summary = performance_summary(
            pd.Series([100.0, 100.0, 101.0]),
            pd.Series([np.nan, boundary, 0.0]),
        )
        self.assertTrue(np.isfinite(summary["Sharpe"]))

    def test_boolean_and_complex_series_are_rejected_before_metric_calculation(self) -> None:
        invalid_series = {
            "boolean dtype": pd.Series([True, True, True]),
            "nullable boolean dtype": pd.Series([True, False, True], dtype="boolean"),
            "mixed object boolean": pd.Series([100.0, True, 121.0], dtype=object),
            "complex dtype": pd.Series([100 + 1j, 110 + 0j, 121 - 2j]),
            "mixed object complex": pd.Series([100.0, np.complex64(110 + 1j), 121.0], dtype=object),
        }

        for label, values in invalid_series.items():
            with (
                self.subTest(label=label, input="equity_curve"),
                self.assertRaisesRegex(ValueError, "equity_curve must contain numeric values"),
            ):
                performance_summary(values)
            with (
                self.subTest(label=label, input="risk_free_returns"),
                self.assertRaisesRegex(ValueError, "risk_free_returns must contain numeric values"),
            ):
                performance_summary(pd.Series([100.0, 110.0, 121.0]), values)

    def test_temporal_series_are_rejected_before_metric_calculation(self) -> None:
        invalid_series = {
            "datetime dtype": pd.Series(pd.date_range("2026-01-01", periods=3)),
            "timedelta dtype": pd.Series(pd.to_timedelta([1, 2, 3], unit="D")),
            "mixed Python date": pd.Series([100.0, date(2026, 1, 2), 121.0], dtype=object),
            "mixed Python datetime": pd.Series(
                [100.0, datetime(2026, 1, 2, 12), 121.0],
                dtype=object,
            ),
            "mixed Python timedelta": pd.Series(
                [100.0, timedelta(days=2), 121.0],
                dtype=object,
            ),
            "mixed NumPy datetime": pd.Series(
                [100.0, np.datetime64("2026-01-02"), 121.0],
                dtype=object,
            ),
            "mixed NumPy timedelta": pd.Series(
                [100.0, np.timedelta64(2, "D"), 121.0],
                dtype=object,
            ),
            "mixed pandas Timestamp": pd.Series(
                [100.0, pd.Timestamp("2026-01-02"), 121.0],
                dtype=object,
            ),
            "mixed pandas Timedelta": pd.Series(
                [100.0, pd.Timedelta(days=2), 121.0],
                dtype=object,
            ),
        }

        for label, values in invalid_series.items():
            with (
                self.subTest(label=label, input="equity_curve"),
                self.assertRaisesRegex(ValueError, "equity_curve must contain numeric values"),
            ):
                performance_summary(values)
            with (
                self.subTest(label=label, input="risk_free_returns"),
                self.assertRaisesRegex(ValueError, "risk_free_returns must contain numeric values"),
            ):
                performance_summary(pd.Series([100.0, 110.0, 121.0]), values)

    def test_ndarray_wrapped_values_are_rejected_before_metric_calculation(self) -> None:
        invalid_series = {
            "wrapped boolean": pd.Series([100.0, np.array(True), 121.0], dtype=object),
            "wrapped complex": pd.Series(
                [100.0, np.array(110.0 + 2.0j), 121.0],
                dtype=object,
            ),
            "wrapped datetime": pd.Series(
                [100.0, np.array(np.datetime64(110, "ns")), 121.0],
                dtype=object,
            ),
            "array container": pd.Series(
                [100.0, np.array([110.0]), 121.0],
                dtype=object,
            ),
        }

        for label, values in invalid_series.items():
            with (
                self.subTest(label=label, input="equity_curve"),
                self.assertRaisesRegex(ValueError, "equity_curve must contain numeric values"),
            ):
                performance_summary(values)
            with (
                self.subTest(label=label, input="risk_free_returns"),
                self.assertRaisesRegex(ValueError, "risk_free_returns must contain numeric values"),
            ):
                performance_summary(pd.Series([100.0, 110.0, 121.0]), values)

    def test_float_conversion_overflow_is_normalized_for_both_series(self) -> None:
        class OverflowingFloat:
            def __float__(self) -> float:
                raise OverflowError("outside float range")

        invalid = pd.Series([1.0, OverflowingFloat(), 3.0], dtype=object)

        with self.assertRaisesRegex(ValueError, "equity_curve must contain numeric values"):
            performance_summary(invalid)
        with self.assertRaisesRegex(ValueError, "risk_free_returns must contain numeric values"):
            performance_summary(pd.Series([100.0, 110.0, 121.0]), invalid)

    def test_risk_free_series_is_validated_even_when_equity_has_no_returns(self) -> None:
        with self.assertRaisesRegex(ValueError, "risk_free_returns must contain numeric values"):
            performance_summary(pd.Series([100.0]), pd.Series([True]))

    def test_nullable_numeric_series_and_complete_multiindex_remain_supported(self) -> None:
        index = pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1)])
        equity = pd.Series([100, 110, 121], index=index, dtype="Int64")
        risk_free = pd.Series([0.0, 0.0, 0.0], index=index, dtype="Float64")

        summary = performance_summary(equity, risk_free)

        self.assertAlmostEqual(summary["Total Return"], 0.21)
        self.assertAlmostEqual(summary["Hit Rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
