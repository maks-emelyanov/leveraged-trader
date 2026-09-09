from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

import leveraged_trader.indicators as indicators
from leveraged_trader.indicators import (
    compute_rsi,
    compute_rsi_details,
    rsi_value_from_average_gain_loss,
)


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

    def test_rsi_uses_wilder_simple_average_seed(self) -> None:
        close = pd.Series(
            [
                54.80,
                56.80,
                57.85,
                59.85,
                60.57,
                61.10,
                62.17,
                60.60,
                62.35,
                62.15,
                62.35,
                61.45,
                62.80,
                61.37,
                62.50,
            ]
        )

        details = compute_rsi_details(close, period=14)

        self.assertTrue(details["rsi"].iloc[:14].isna().all())
        self.assertAlmostEqual(details.loc[14, "avg_gain"], 0.8428571428571432)
        self.assertAlmostEqual(details.loc[14, "avg_loss"], 0.2928571428571430)
        self.assertAlmostEqual(details.loc[14, "rsi"], 74.21383647798743)

    def test_rsi_seed_average_does_not_overflow_for_finite_prices(self) -> None:
        close = pd.Series([1e-300, 1e308] * 7 + [1e-300])

        details = compute_rsi_details(close, period=14)

        self.assertTrue(np.isfinite(details.loc[14, "avg_gain"]))
        self.assertTrue(np.isfinite(details.loc[14, "avg_loss"]))
        np.testing.assert_allclose(details.loc[14, "avg_gain"], 5e307)
        np.testing.assert_allclose(details.loc[14, "avg_loss"], 5e307)
        self.assertEqual(details.loc[14, "rsi"], 50.0)

    def test_rsi_wilder_recurrence_matches_incremental_arithmetic_without_overflow(
        self,
    ) -> None:
        close = pd.Series(
            [1e-300, 1e308] + [1e308] * 13 + [1e-300, 1e308, 1e-300, 1e308],
        )

        details = compute_rsi_details(close, period=14)
        expected_gain = float(details.loc[14, "avg_gain"])
        expected_loss = float(details.loc[14, "avg_loss"])
        alpha = 1 / 14
        for position in range(15, len(close)):
            delta = float(close.iloc[position]) - float(close.iloc[position - 1])
            expected_gain = (1 - alpha) * expected_gain + alpha * max(delta, 0.0)
            expected_loss = (1 - alpha) * expected_loss + alpha * max(-delta, 0.0)
            self.assertEqual(details.loc[position, "avg_gain"], expected_gain)
            self.assertEqual(details.loc[position, "avg_loss"], expected_loss)

        self.assertTrue(np.isfinite(details.loc[16:, ["avg_gain", "avg_loss", "rsi"]]).all().all())

    def test_rsi_rejects_non_finite_computed_averages(self) -> None:
        maximum = np.finfo(float).max

        with self.assertRaisesRegex(ValueError, "average gain and loss must remain finite"):
            compute_rsi(pd.Series([-maximum, maximum]), period=1)

        for avg_gain, avg_loss in ((float("inf"), 0.0), (0.0, float("nan"))):
            with (
                self.subTest(avg_gain=avg_gain, avg_loss=avg_loss),
                self.assertRaisesRegex(ValueError, "average gain and loss must remain finite"),
            ):
                rsi_value_from_average_gain_loss(avg_gain, avg_loss)

    def test_rsi_rejects_non_finite_price_delta_before_warmup(self) -> None:
        close = pd.Series([-np.finfo(float).max, np.finfo(float).max])

        for calculator in (compute_rsi, compute_rsi_details):
            with (
                self.subTest(calculator=calculator.__name__),
                self.assertRaisesRegex(ValueError, "average gain and loss must remain finite"),
            ):
                calculator(close, period=14)

    def test_scalar_rsi_rejects_non_numeric_semantic_types_consistently(self) -> None:
        invalid_values = (
            True,
            np.bool_(False),
            "1.0",
            1 + 0j,
            date(2026, 1, 2),
            datetime(2026, 1, 2, 12),
            timedelta(days=1),
            np.datetime64("2026-01-02"),
            np.timedelta64(1, "D"),
            np.array(1.0),
        )

        for value in invalid_values:
            for avg_gain, avg_loss in ((value, 0.0), (0.0, value)):
                with (
                    self.subTest(value=repr(value), avg_gain=avg_gain, avg_loss=avg_loss),
                    self.assertRaisesRegex(ValueError, "average gain and loss must be numeric scalars"),
                ):
                    rsi_value_from_average_gain_loss(avg_gain, avg_loss)

    def test_rsi_rejects_negative_average_gain_or_loss(self) -> None:
        for avg_gain, avg_loss in ((-1.0, 0.0), (0.0, -1.0), (-1.0, -2.0), (-1.0, 1.0)):
            with (
                self.subTest(avg_gain=avg_gain, avg_loss=avg_loss),
                self.assertRaisesRegex(ValueError, "average gain and loss must be non-negative"),
            ):
                rsi_value_from_average_gain_loss(avg_gain, avg_loss)

    def test_rsi_rejects_non_positive_period(self) -> None:
        with self.assertRaisesRegex(ValueError, "period must be a positive integer"):
            compute_rsi(pd.Series([1.0, 2.0]), period=0)

    def test_rsi_rejects_numpy_temporal_period_scalars(self) -> None:
        for period in (np.timedelta64(3, "ns"), np.datetime64(3, "ns")):
            with (
                self.subTest(period=period),
                self.assertRaisesRegex(ValueError, "period must be a positive integer"),
            ):
                compute_rsi(pd.Series([1.0, 2.0]), period=period)

    def test_rsi_rejects_missing_duplicate_or_unsorted_index(self) -> None:
        cases = [
            (pd.Index([0.0, float("nan"), 2.0]), "missing labels"),
            (
                pd.MultiIndex.from_tuples([("A", 1), ("A", None), ("B", 1)]),
                "missing labels",
            ),
            (pd.Index([0, 0, 1]), "duplicate observations"),
            (pd.Index([0, 2, 1]), "sorted in increasing order"),
        ]
        for index, expected_message in cases:
            with (
                self.subTest(expected_message=expected_message),
                self.assertRaisesRegex(ValueError, expected_message),
            ):
                compute_rsi(pd.Series([1.0, 2.0, 3.0], index=index), period=2)

    def test_rsi_accepts_sorted_unique_complete_multiindex(self) -> None:
        index = pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1)])

        rsi = compute_rsi(pd.Series([1.0, 2.0, 3.0], index=index), period=2)

        self.assertTrue(rsi.index.equals(index))
        self.assertEqual(float(rsi.iloc[-1]), 100.0)

    def test_rsi_rejects_boolean_and_complex_price_values(self) -> None:
        cases = {
            "boolean dtype": pd.Series([True, False, True]),
            "nullable boolean dtype": pd.Series([True, False, True], dtype="boolean"),
            "mixed object boolean": pd.Series([1.0, True, 3.0], dtype=object),
            "complex dtype": pd.Series([1 + 0j, 2 + 1j, 3 + 0j]),
            "mixed object complex": pd.Series([1.0, np.complex64(2 + 1j), 3.0], dtype=object),
        }

        for label, close in cases.items():
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ValueError, "close must contain numeric values"),
            ):
                compute_rsi(close, period=2)

    def test_rsi_rejects_temporal_price_values(self) -> None:
        cases = {
            "datetime dtype": pd.Series(pd.date_range("2026-01-01", periods=3)),
            "timedelta dtype": pd.Series(pd.to_timedelta([1, 2, 3], unit="D")),
            "mixed Python date": pd.Series([1.0, date(2026, 1, 2), 3.0], dtype=object),
            "mixed Python datetime": pd.Series(
                [1.0, datetime(2026, 1, 2, 12), 3.0],
                dtype=object,
            ),
            "mixed Python timedelta": pd.Series(
                [1.0, timedelta(days=2), 3.0],
                dtype=object,
            ),
            "mixed NumPy datetime": pd.Series(
                [1.0, np.datetime64("2026-01-02"), 3.0],
                dtype=object,
            ),
            "mixed NumPy timedelta": pd.Series(
                [1.0, np.timedelta64(2, "D"), 3.0],
                dtype=object,
            ),
            "mixed pandas Timestamp": pd.Series(
                [1.0, pd.Timestamp("2026-01-02"), 3.0],
                dtype=object,
            ),
            "mixed pandas Timedelta": pd.Series(
                [1.0, pd.Timedelta(days=2), 3.0],
                dtype=object,
            ),
        }

        for label, close in cases.items():
            for implementation in (compute_rsi, compute_rsi_details):
                with (
                    self.subTest(label=label, implementation=implementation.__name__),
                    self.assertRaisesRegex(ValueError, "close must contain numeric values"),
                ):
                    implementation(close, period=2)

    def test_rsi_rejects_ndarray_wrapped_price_values(self) -> None:
        cases = {
            "wrapped boolean": pd.Series([1.0, np.array(True), 3.0], dtype=object),
            "wrapped complex": pd.Series(
                [1.0, np.array(2.0 + 1.0j), 3.0],
                dtype=object,
            ),
            "wrapped datetime": pd.Series(
                [1.0, np.array(np.datetime64(2, "ns")), 3.0],
                dtype=object,
            ),
            "array container": pd.Series(
                [1.0, np.array([2.0]), 3.0],
                dtype=object,
            ),
        }

        for label, close in cases.items():
            for implementation in (compute_rsi, compute_rsi_details):
                with (
                    self.subTest(label=label, implementation=implementation.__name__),
                    self.assertRaisesRegex(ValueError, "close must contain numeric values"),
                ):
                    implementation(close, period=2)

    def test_rsi_normalizes_float_conversion_overflow(self) -> None:
        class OverflowingFloat:
            def __float__(self) -> float:
                raise OverflowError("outside float range")

        close = pd.Series([1.0, OverflowingFloat(), 3.0], dtype=object)

        with self.assertRaisesRegex(ValueError, "close must contain numeric values"):
            compute_rsi(close, period=2)

    def test_rsi_accepts_nullable_numeric_extension_dtypes_and_preserves_index(self) -> None:
        index = pd.Index(["first", "second", "third"], name="session")

        for dtype in ("Int64", "Float64"):
            with self.subTest(dtype=dtype):
                close = pd.Series([1, 2, 3], index=index, dtype=dtype)

                rsi = compute_rsi(close, period=2)
                details = compute_rsi_details(close, period=2)

                self.assertTrue(rsi.index.equals(index))
                self.assertTrue(details.index.equals(index))
                self.assertEqual(float(rsi.iloc[-1]), 100.0)

    def test_rsi_wrappers_preserve_names_index_and_float64_outputs(self) -> None:
        index = pd.Index(["first", "second", "third", "z-last"], name="session")
        close = pd.Series([1, 2, 1, 3], index=index, name="price", dtype="Int64")

        rsi = compute_rsi(close, period=2)
        details = compute_rsi_details(close, period=2)

        self.assertEqual(rsi.name, "rsi")
        self.assertTrue(rsi.index.equals(index))
        self.assertEqual(rsi.dtype, np.dtype(np.float64))
        self.assertEqual(list(details.columns), ["close", "avg_gain", "avg_loss", "rsi"])
        self.assertTrue(details.index.equals(index))
        self.assertTrue(all(dtype == np.dtype(np.float64) for dtype in details.dtypes))
        pd.testing.assert_series_equal(rsi, details["rsi"])

    def test_rsi_core_receives_contiguous_array_and_series_api_skips_details_frame(self) -> None:
        noncontiguous = np.arange(40.0)[::2]
        close = pd.Series(noncontiguous, copy=False)
        self.assertFalse(close.to_numpy(copy=False).flags.c_contiguous)

        observed_arrays: list[np.ndarray] = []
        original_compute = indicators._compute_rsi_arrays

        def record_array(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            observed_arrays.append(values)
            return original_compute(values, period)

        with (
            patch("leveraged_trader.indicators._compute_rsi_arrays", side_effect=record_array),
            patch(
                "leveraged_trader.indicators._rsi_details_frame",
                side_effect=AssertionError("compute_rsi must not build the details frame"),
            ),
        ):
            result = compute_rsi(close, period=3)

        self.assertEqual(len(observed_arrays), 1)
        self.assertIsInstance(observed_arrays[0], np.ndarray)
        self.assertEqual(observed_arrays[0].dtype, np.dtype(np.float64))
        self.assertTrue(observed_arrays[0].flags.c_contiguous)
        self.assertEqual(len(result), len(close))


if __name__ == "__main__":
    unittest.main()
