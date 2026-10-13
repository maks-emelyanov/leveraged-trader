from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from leveraged_trader.grid_research import (
    DEFAULT_GRIDS,
    FROZEN_SELECTION_VERSION,
    GridSpec,
    _UnionMetrics,
    _utility,
    evaluate_asset,
    evaluate_period,
    summarize,
)
from leveraged_trader.research_selection import select_strategy


def fake_metrics(*args):
    size = len(args[5])
    return {
        "sharpe": np.full(size, 2.0),
        "total_return": np.full(size, 0.2),
        "completed_trades": np.full(size, 6),
        "mean_holding_sessions": np.full(size, 5.0),
        "max_holding_sessions": np.full(size, 10),
        "exposure_fraction": np.full(size, 0.5),
        "open_holding_sessions": np.zeros(size, dtype=int),
    }


def market_arrays():
    dates = np.arange("2010-01-01", "2021-01-01", dtype="datetime64[D]")
    dates = dates[np.is_busday(dates)]
    prices = np.arange(len(dates), dtype=float) + 100.0
    return dates, (prices, prices + 2, prices + 1, np.full(len(dates), 30.0), np.zeros(len(dates)))


class GridResearchTests(unittest.TestCase):
    grids = (
        GridSpec("current49", (20.0, 30.0), (60.0, 70.0), (1.05,)),
        GridSpec("alternative", (20.0, 30.0), (60.0, 70.0), (1.05, 1.1)),
    )

    def test_frozen_family_sizes(self):
        expected = [49, 1240, 217, 28, 56, 42, 63, 35]
        for direction in ("long", "inverse"):
            self.assertEqual([len(grid.pairs(direction)) for grid in DEFAULT_GRIDS], expected)

    def test_training_never_sees_future_and_shared_blocks_are_cached(self):
        dates, arrays = market_arrays()
        boundary = np.searchsorted(dates, np.datetime64("2017-01-01"))
        with patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=fake_metrics) as metric:
            rows = evaluate_asset("ASSET", "PROXY", "long", dates, arrays, grids=self.grids, validation_years=(2017,))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["active"] for row in rows))
        # Three shared development blocks, one shared holdout, one full prefix,
        # and the frozen winner's outer test at each of two execution costs.
        self.assertEqual(metric.call_count, 7)
        for call in metric.call_args_list[:5]:
            self.assertLess(np.max(call.args[0]), arrays[0][boundary])
        for call in metric.call_args_list[5:]:
            self.assertEqual(call.args[0][0], arrays[0][boundary])
        self.assertEqual(rows[0]["training_end"], "2016-12-30")
        self.assertEqual(rows[0]["test_start"], "2017-01-02")

    def test_validation_and_confirmation_preserve_v1_after_production_moves_to_v2(self):
        dates, arrays = market_arrays()
        expected_score = 2.0 - 0.5 * np.log1p(5.0 / 21.0) - 0.25 * np.log1p(10.0 / 63.0)
        production = select_strategy(
            *arrays, np.asarray([20.0]), np.asarray([1.05]), 100_000.0, 0.0003,
            metric_evaluator=fake_metrics,
        )
        self.assertEqual(production["version"], 2)
        self.assertGreater(production["selection_score"], expected_score)
        self.assertEqual(FROZEN_SELECTION_VERSION, 1)
        for evaluator, options in (
            (evaluate_asset, {"validation_years": (2017,)}),
            (evaluate_period, {"test_start": "2017-01-01", "test_end": "2019-12-31"}),
        ):
            with (
                self.subTest(evaluator=evaluator.__name__),
                patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=fake_metrics),
                patch("leveraged_trader.grid_research.select_strategy", wraps=select_strategy) as selector,
            ):
                rows = evaluator("ASSET", "PROXY", "long", dates, arrays, grids=self.grids, **options)
            self.assertEqual(selector.call_count, len(self.grids))
            for call in selector.call_args_list:
                self.assertEqual(call.kwargs["selection_version"], 1)
            for row in rows:
                self.assertEqual(row["training_selection_score"], expected_score)

    def test_rejected_training_remains_cash_in_denominator(self):
        dates, arrays = market_arrays()

        def failing(*args):
            result = fake_metrics(*args)
            result["completed_trades"][:] = 0
            return result

        with patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=failing):
            rows = evaluate_asset("ASSET", "PROXY", "long", dates, arrays, grids=self.grids, validation_years=(2017,))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["utility"] == 0 and not row["active"] for row in rows))
        result = summarize(rows)
        self.assertTrue(all(row["asset_years"] == 1 for row in result["grids"]))
        self.assertEqual(result["paired"][0]["discordant_groups_utility"], 0)

    def test_full_history_sharpe_gate_and_cost_stress_do_not_retune(self):
        dates, arrays = market_arrays()
        boundary = np.searchsorted(dates, np.datetime64("2017-01-01"))

        def evaluate(*args):
            result = fake_metrics(*args)
            if len(args[0]) == boundary:
                result["sharpe"][:] = 0.9
            return result

        with patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=evaluate):
            rows = evaluate_asset("ASSET", "PROXY", "long", dates, arrays, grids=self.grids, validation_years=(2017,))
        self.assertTrue(all(row["training_status"] == "passed" for row in rows))
        self.assertTrue(all(row["activation_status"] == "failed_full_history_gate" for row in rows))

        def stressed(*args):
            result = fake_metrics(*args)
            if args[8] == 0.001:
                result["sharpe"][:] = -1.0
            return result

        with patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=stressed) as metric:
            rows = evaluate_asset("ASSET", "PROXY", "long", dates, arrays, grids=self.grids, validation_years=(2017,))
        self.assertTrue(all(row["stress_utility"] < 0 < row["utility"] for row in rows))
        for index in (5, 6):
            np.testing.assert_array_equal(metric.call_args_list[-2].args[index], metric.call_args_list[-1].args[index])

    def test_continuous_confirmation_trains_once_and_includes_end_date(self):
        dates, arrays = market_arrays()
        with patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=fake_metrics) as metric:
            rows = evaluate_period(
                "ASSET", "PROXY", "inverse", dates, arrays, grids=self.grids,
                test_start="2017-01-01", test_end="2019-12-31",
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["test_end"], "2019-12-31")
        self.assertGreater(rows[0]["test_sessions"], 750)
        self.assertEqual(metric.call_count, 7)
        self.assertTrue(all(call.args[9] == 1 for call in metric.call_args_list))

    def test_union_cache_keeps_each_grid_own_neighbors(self):
        _, arrays = market_arrays()
        training = tuple(values[:1260] for values in arrays)
        union = np.asarray([(20.0, 1.1), (30.0, 1.1), (40.0, 1.1)])

        def evaluate(*args):
            result = fake_metrics(*args)
            result["sharpe"] = np.asarray([{20: 0.2, 30: 4.0, 40: 0.2}[int(buy)] for buy in args[5]])
            return result

        with patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=evaluate):
            evaluator = _UnionMetrics(training, union)
            cached = select_strategy(
                *training, np.asarray([30.0]), np.asarray([1.1]), 100_000.0, 0.0003,
                metric_evaluator=evaluator, selection_version=FROZEN_SELECTION_VERSION,
            )
        with patch("leveraged_trader.research_selection.grid_research_metrics", side_effect=evaluate):
            ordinary = select_strategy(
                *training, np.asarray([30.0]), np.asarray([1.1]), 100_000.0, 0.0003,
                selection_version=FROZEN_SELECTION_VERSION,
            )
            wide = select_strategy(
                *training, union[:, 0], union[:, 1], 100_000.0, 0.0003,
                selection_version=FROZEN_SELECTION_VERSION,
            )
        self.assertEqual(cached, ordinary)
        self.assertGreater(cached["selection_score"], wide["selection_score"])

    def test_zero_exposure_is_cash_but_invalid_exposed_metrics_fail(self):
        cash = {name: 0.0 for name in fake_metrics(np.zeros(1), None, None, None, None, [20])}
        cash["sharpe"] = None
        cash["mean_holding_sessions"] = None
        self.assertEqual(_utility(cash), 0.0)
        cash["exposure_fraction"] = 0.5
        with self.assertRaisesRegex(ValueError, "finite"):
            _utility(cash)

    def test_proxy_balancing_and_paired_bootstrap_preserve_cash_cells(self):
        dates, arrays = market_arrays()
        with patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=fake_metrics):
            original = evaluate_asset("A", "P1", "long", dates, arrays, grids=self.grids, validation_years=(2017,))
        rows = []
        for asset, signal in (("A", "P1"), ("B", "P1"), ("C", "P2")):
            for row in original:
                utility = 2.0 if signal == "P1" else -1.0
                if row["grid"] == "current49":
                    utility = 0.0
                rows.append({**row, "asset": asset, "signal": signal, "utility": utility, "stress_utility": utility})
        result = summarize(rows)
        alternative = next(row for row in result["grids"] if row["grid"] == "alternative")
        self.assertEqual(alternative["utility"], 0.5)  # Two proxies, not three equally weighted assets.
        comparison = result["paired"][0]
        self.assertEqual(comparison["delta_utility"], 0.5)
        self.assertEqual(comparison["discordant_groups_utility"], 2)
        self.assertEqual(comparison["improvement_share_utility"], 0.5)
        self.assertLess(comparison["familywise_lower_95_utility"], 0.0)
        self.assertEqual(summarize(rows), result)
        with self.assertRaisesRegex(ValueError, "exactly the same"):
            summarize(rows[:-1])
        with self.assertRaisesRegex(ValueError, "finite utility"):
            summarize([{**rows[0], "utility": float("nan")}, *rows[1:]])

    def test_bootstrap_uses_registered_seed_and_fixed_two_sided_family(self):
        dates, arrays = market_arrays()
        with patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=fake_metrics):
            original = evaluate_asset("A", "P1", "long", dates, arrays, grids=self.grids, validation_years=(2017,))
        deltas = np.asarray([-6.0, -2.0, -1.0, 0.1, 0.2, 1.0, 3.0, 8.0])
        rows = []
        for i, delta in enumerate(deltas):
            for row in original:
                utility = delta if row["grid"] == "alternative" else 0.0
                rows.append({
                    **row, "asset": f"A{i}", "signal": f"P{i}", "utility": utility, "stress_utility": utility,
                })
        comparison = summarize(rows)["paired"][0]
        rng = np.random.default_rng(20261012)
        indices = rng.integers(0, len(deltas), size=(2000, len(deltas)))
        expected = deltas[indices].mean(axis=1)
        # Even this two-grid confirmation subset retains the seven comparisons
        # registered for the original eight-grid discovery family.
        self.assertEqual(comparison["comparison_count"], 7)
        self.assertEqual(comparison["bootstrap_seed"], 20261012)
        for prefix, lower, upper in (("", 0.025, 0.975), ("familywise_", 0.05 / 14, 1 - 0.05 / 14)):
            for metric in ("utility", "stress_utility"):
                self.assertEqual(comparison[f"{prefix}lower_95_{metric}"], np.quantile(expected, lower))
                self.assertEqual(comparison[f"{prefix}upper_95_{metric}"], np.quantile(expected, upper))
        self.assertLess(comparison["familywise_lower_95_utility"], comparison["lower_95_utility"])
        with self.assertRaisesRegex(ValueError, "comparison_count"):
            summarize(rows, comparison_count=0)

    def test_matched_exposed_holding_deltas_balance_proxies_and_exclude_cash(self):
        dates, arrays = market_arrays()
        with patch("leveraged_trader.grid_research.grid_research_metrics", side_effect=fake_metrics):
            original = evaluate_asset("A", "P1", "long", dates, arrays, grids=self.grids, validation_years=(2017,))
        rows = []
        for asset, signal, mean, maximum in (("A", "P1", 7, 14), ("B", "P1", 11, 18), ("C", "P2", 3, 6)):
            for row in original:
                row = {**row, "asset": asset, "signal": signal}
                if row["grid"] == "alternative":
                    row.update(mean_holding_sessions=mean, max_holding_sessions=maximum)
                rows.append(row)
        for row in original:
            row = {**row, "asset": "D", "signal": "P3"}
            if row["grid"] == "alternative":
                row.update(exposure_fraction=0.0, mean_holding_sessions=0.0, max_holding_sessions=0)
            rows.append(row)
        comparison = summarize(rows)["paired"][0]
        self.assertEqual(comparison["matched_exposed_asset_years"], 3)
        self.assertEqual(comparison["matched_exposed_signal_groups"], 2)
        self.assertEqual(comparison["matched_exposed_delta_mean_holding_sessions"], 1.0)
        self.assertEqual(comparison["matched_exposed_delta_max_holding_sessions"], 1.0)
        unmatched = [{**row, "exposure_fraction": 0.0} for row in rows]
        comparison = summarize(unmatched)["paired"][0]
        self.assertEqual(comparison["matched_exposed_asset_years"], 0)
        self.assertIsNone(comparison["matched_exposed_delta_mean_holding_sessions"])
        self.assertIsNone(comparison["matched_exposed_delta_max_holding_sessions"])


if __name__ == "__main__":
    unittest.main()
