from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from leveraged_trader.grid_research import GridSpec
from leveraged_trader.grid_resolution import _DevelopmentUnionMetrics, evaluate_windows
from leveraged_trader.research_selection import select_strategy
from tests.test_grid_research import fake_metrics, market_arrays


class GridResolutionTests(unittest.TestCase):
    grids = (
        GridSpec("fine", (20.0, 30.0), (60.0, 70.0), (1.05, 1.1)),
        GridSpec("coarse", (20.0,), (60.0,), (1.05,)),
    )

    def evaluate(self, arrays=None, **kwargs):
        dates, ordinary = market_arrays()
        return evaluate_windows(
            "ASSET", "PROXY", "long", dates, ordinary if arrays is None else arrays,
            grids=self.grids, windows=((2017, "2017-01-01", "2018-01-01"),), **kwargs,
        )

    def test_v2_chronology_development_reuse_and_single_winner_holdouts(self):
        dates, arrays = market_arrays()
        boundary = np.searchsorted(dates, np.datetime64("2017-01-01"))
        with (
            patch("leveraged_trader.grid_resolution.grid_research_metrics", side_effect=fake_metrics) as metric,
            patch("leveraged_trader.grid_resolution.select_strategy", wraps=select_strategy) as selector,
        ):
            rows = self.evaluate()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["active"] for row in rows))
        self.assertEqual(metric.call_count, 8)
        self.assertEqual([len(call.args[5]) for call in metric.call_args_list[:5]], [4, 4, 4, 1, 1])
        for call in metric.call_args_list[:-2]:
            self.assertLess(np.max(call.args[0]), arrays[0][boundary])
        for call in metric.call_args_list[-2:]:
            self.assertEqual(call.args[0][0], arrays[0][boundary])
        self.assertTrue(all(call.kwargs["selection_version"] == 2 for call in selector.call_args_list))
        self.assertEqual(rows[0]["training_version"], 2)
        self.assertEqual(rows[0]["training_end"], "2016-12-30")
        self.assertEqual(rows[0]["test_end"], "2017-12-29")
        self.assertEqual(rows[0]["training_eligible_candidates"], 4)
        self.assertEqual(rows[0]["training_supported_candidates"], 4)
        self.assertEqual(rows[0]["training_union_neighbor_score"], rows[0]["training_selection_score"])
        self.assertGreater(rows[0]["activity_utility"], rows[0]["utility"])
        self.assertEqual(rows[0]["annual_completed_trades"], 252.0 * 6 / rows[0]["test_sessions"])

    def test_failed_holdout_stays_cash_and_cannot_rerank(self):
        dates, _ = market_arrays()
        training_count = np.searchsorted(dates, np.datetime64("2017-01-01"))
        holdout_count = training_count - training_count * 4 // 5

        def failed(*args):
            result = fake_metrics(*args)
            if len(args[0]) == holdout_count:
                self.assertEqual(len(args[5]), 1)
                result["sharpe"][:] = -1.0
                result["total_return"][:] = -0.2
            return result

        with patch("leveraged_trader.grid_resolution.grid_research_metrics", side_effect=failed) as metric:
            rows = self.evaluate()
        self.assertEqual(metric.call_count, 5)
        self.assertTrue(all(row["training_status"] == "failed_holdout" for row in rows))
        self.assertTrue(all(not row["active"] and row["utility"] == row["activity_utility"] == 0 for row in rows))
        self.assertTrue(all(row["buy_rsi"] == 20.0 for row in rows))

    def test_full_prefix_gate_and_stressed_execution_reuse_frozen_pairs(self):
        dates, _ = market_arrays()
        training_count = np.searchsorted(dates, np.datetime64("2017-01-01"))

        def failed_full(*args):
            result = fake_metrics(*args)
            if len(args[0]) == training_count:
                result["sharpe"][:] = 0.9
            return result

        with patch("leveraged_trader.grid_resolution.grid_research_metrics", side_effect=failed_full):
            rows = self.evaluate()
        self.assertTrue(all(row["training_status"] == "passed" for row in rows))
        self.assertTrue(all(row["activation_status"] == "failed_full_history_gate" for row in rows))

        def stressed(*args):
            result = fake_metrics(*args)
            if args[8] == 0.001:
                result["sharpe"][:] = -1.0
            return result

        with patch("leveraged_trader.grid_resolution.grid_research_metrics", side_effect=stressed) as metric:
            rows = self.evaluate()
        self.assertTrue(all(row["stress_utility"] < 0 < row["utility"] for row in rows))
        for index in (5, 6):
            np.testing.assert_array_equal(metric.call_args_list[-2].args[index], metric.call_args_list[-1].args[index])

    def test_grid_specific_neighbors_and_union_outlier_diagnostics(self):
        dates, arrays = market_arrays()
        grids = (
            GridSpec("fine", (20.0, 30.0, 40.0), (60.0,), (1.05,)),
            GridSpec("coarse", (30.0,), (60.0,), (1.05,)),
        )

        def peaked(*args):
            result = fake_metrics(*args)
            result["sharpe"] = np.where(args[5] == 30.0, 2.0, 0.01)
            return result

        with patch("leveraged_trader.grid_resolution.grid_research_metrics", side_effect=peaked):
            rows = evaluate_windows(
                "A", "P", "long", dates, arrays, grids=grids,
                windows=((2017, "2017-01-01", "2018-01-01"),),
            )
        self.assertEqual(rows[0]["training_status"], "insufficient_evidence")
        self.assertEqual(rows[0]["training_eligible_candidates"], 1)
        self.assertEqual(rows[0]["training_isolated_peak_rejections"], 1)
        self.assertEqual(rows[1]["training_status"], "passed")
        self.assertFalse(rows[1]["training_union_neighbor_supported"])
        self.assertEqual(rows[1]["training_union_neighbor_score"], 0.0)

    def test_cached_and_direct_v2_selections_match(self):
        _, arrays = market_arrays()
        training = tuple(values[:1260] for values in arrays)
        union = self.grids[0].pairs("long")
        with patch("leveraged_trader.grid_resolution.grid_research_metrics", side_effect=fake_metrics):
            evaluator = _DevelopmentUnionMetrics(training, union)
            for grid in self.grids:
                pairs = grid.pairs("long")
                cached = select_strategy(
                    *training, pairs[:, 0], pairs[:, 1], 100000.0, 0.0003, metric_evaluator=evaluator,
                )
                direct = select_strategy(
                    *training, pairs[:, 0], pairs[:, 1], 100000.0, 0.0003, metric_evaluator=fake_metrics,
                )
                self.assertEqual(cached, direct)

    def test_exclusive_boundaries_coverage_minimums_and_invalid_inputs(self):
        with patch("leveraged_trader.grid_resolution.grid_research_metrics", side_effect=fake_metrics):
            self.assertEqual(self.evaluate(min_training_sessions=10000), [])
            self.assertEqual(self.evaluate(min_test_sessions=300), [])
            with self.assertRaisesRegex(ValueError, "positive"):
                self.evaluate(min_test_sessions=0)
        dates, arrays = market_arrays()
        for windows in (
            ((2017, "2018-01-01", "2017-01-01"),),
            ((2017, "2017-01-01", "2018-01-01"), (2017, "2018-01-01", "2019-01-01")),
        ):
            with self.assertRaises(ValueError):
                evaluate_windows("A", "P", "long", dates, arrays, grids=self.grids, windows=windows)


if __name__ == "__main__":
    unittest.main()
