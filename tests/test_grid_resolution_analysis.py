from __future__ import annotations

import copy
import json
import unittest

from leveraged_trader.grid_resolution_analysis import summarize_resolution


def outcomes(count=12, delta_utility=-0.01, delta_return=-0.002, cash=False):
    rows = []
    for index in range(count):
        for grid, du, dr in (("r1_t1", 0, 0), ("r2_t2", delta_utility, delta_return)):
            rows.append({
                "asset": f"A{index}", "signal": f"P{index}", "direction": "long", "year": 2020,
                "grid": grid, "candidate_count": 4059 if grid == "r1_t1" else 1050,
                "active": not cash, "exposure_fraction": 0.0 if cash else 0.2,
                "completed_trades": 0 if cash else 5,
                "utility": 0.0 if cash else 1 + du,
                "stress_utility": 0.0 if cash else 0.9 + du,
                "total_return": 0.0 if cash else 0.1 + dr,
                "stress_total_return": 0.0 if cash else 0.09 + dr,
                "training_status": "insufficient_evidence" if cash else "passed",
                "mean_holding_sessions": 0.0 if cash else 5.0,
                "max_holding_sessions": 0.0 if cash else 10.0,
            })
    return rows


GRIDS = [{"name": "r1_t1", "candidate_count": 4059}, {"name": "r2_t2", "candidate_count": 1050}]


def summarize(rows, **kwargs):
    return summarize_resolution(rows, GRIDS, bootstrap_samples=1000, **kwargs)


class GridResolutionAnalysisTests(unittest.TestCase):
    def test_broad_small_loss_is_noninferior_and_json_safe(self):
        result = summarize(outcomes())
        self.assertEqual(result["recommendation"]["grid"], "r2_t2")
        self.assertFalse(result["boundary"]["identified"])
        self.assertEqual(result["comparisons"][0]["classification"], "noninferior")
        self.assertAlmostEqual(result["comparisons"][0]["metrics"]["utility"]["delta"], -0.01)
        json.dumps(result, allow_nan=False)

    def test_broad_practical_loss_is_degradation(self):
        result = summarize(outcomes(delta_utility=-0.1, delta_return=-0.02))
        self.assertEqual(result["comparisons"][0]["classification"], "degraded")
        self.assertEqual(result["boundary"]["degraded_grids"], ["r2_t2"])
        self.assertIsNone(result["recommendation"]["grid"])

    def test_either_primary_metric_can_establish_degradation(self):
        result = summarize(outcomes(delta_utility=0.1, delta_return=-0.02))
        self.assertEqual(result["comparisons"][0]["classification"], "degraded")
        self.assertEqual(result["comparisons"][0]["robust_degrading_fields"], ["total_return"])

    def test_cash_equality_never_establishes_efficiency_optimum(self):
        result = summarize(outcomes(cash=True))
        self.assertEqual(result["comparisons"][0]["classification"], "insufficient_evidence")
        self.assertIsNone(result["recommendation"]["grid"])
        self.assertFalse(result["boundary"]["identified"])
        self.assertEqual(result["grids"][0]["training_statuses"], {"insufficient_evidence": 12})

    def test_one_proxy_cannot_establish_degradation(self):
        result = summarize(outcomes(count=1, delta_utility=-100, delta_return=-0.5))
        self.assertEqual(result["comparisons"][0]["classification"], "insufficient_evidence")
        self.assertIsNone(result["comparisons"][0]["metrics"]["utility"]["leave_one_proxy_out_min"])
        self.assertFalse(result["boundary"]["identified"])

    def test_fewer_than_fifty_baseline_trades_blocks_claim(self):
        rows = outcomes()
        for row in rows:
            row["completed_trades"] = 1
        result = summarize(rows)
        self.assertFalse(result["comparisons"][0]["breadth_gates"]["baseline_completed_trades"])
        self.assertIsNone(result["recommendation"]["grid"])

    def test_stress_cost_deterioration_blocks_noninferiority(self):
        rows = outcomes()
        for row in rows:
            if row["grid"] == "r2_t2":
                row["stress_utility"] -= 1
        result = summarize(rows)
        comparison = result["comparisons"][0]
        self.assertEqual(comparison["base_classification"], "noninferior")
        self.assertEqual(comparison["stress_classification"], "degraded")
        self.assertIsNone(result["recommendation"]["grid"])

    def test_sparse_direction_blocks_global_recommendation(self):
        rows = outcomes()
        for row in copy.deepcopy(rows[:2]):
            row.update(asset="INV", signal="IP", direction="inverse")
            rows.append(row)
        result = summarize(rows)
        self.assertEqual(result["comparisons"][0]["classification"], "noninferior")
        self.assertIsNone(result["recommendation"]["grid"])
        inverse = next(row for row in result["comparisons"] if row["direction"] == "inverse")
        self.assertEqual(inverse["classification"], "insufficient_evidence")

    def test_proxy_balance_does_not_overweight_duplicate_assets(self):
        rows = outcomes()
        for row in rows[:2]:
            if row["grid"] == "r2_t2":
                row["utility"] += 1.0
        result = summarize(rows)
        original = result["comparisons"][0]["metrics"]["utility"]["delta"]
        duplicated = []
        for index in range(5):
            for row in rows[:2]:
                duplicated.append({**row, "asset": f"D{index}"})
        result = summarize(rows + duplicated)
        self.assertAlmostEqual(result["comparisons"][0]["metrics"]["utility"]["delta"], original)

    def test_trim_and_leave_one_out_keep_raw_primary_outlier(self):
        rows = outcomes()
        for row in rows[:2]:
            if row["grid"] == "r2_t2":
                row["utility"] += 120
        result = summarize(rows)
        utility = result["comparisons"][0]["metrics"]["utility"]
        self.assertAlmostEqual(utility["delta"], 9.99)
        self.assertAlmostEqual(utility["trimmed10_delta"], -0.01)
        self.assertAlmostEqual(utility["leave_one_proxy_out_min"], -0.01)

    def test_familywise_adjustment_includes_metrics_directions_and_frozen_count(self):
        result = summarize(outcomes(), comparison_count=50)
        settings = result["settings"]
        self.assertEqual(settings["familywise_contrasts"], 50 * 4 * 2)
        self.assertEqual(settings["tail_probability"], 0.05 / (2 * 50 * 4 * 2))
        self.assertTrue(settings["limited_tail_precision"])

    def test_unpaired_duplicate_nonfinite_and_count_mismatch_rejected(self):
        cases = [outcomes()[:-1], outcomes() + [outcomes()[0]]]
        nonfinite = outcomes()
        nonfinite[0]["utility"] = float("nan")
        cases.append(nonfinite)
        mismatch = outcomes()
        mismatch[0]["candidate_count"] = 1
        cases.append(mismatch)
        for rows in cases:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                summarize(rows)


if __name__ == "__main__":
    unittest.main()
