from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from leveraged_trader.optimized_backtest import RSI_ENTRY_UPPER
from leveraged_trader.research_selection import select_strategy


def metrics(
    sharpes: list[float],
    *,
    mean_holds: list[float] | None = None,
    max_holds: list[float] | None = None,
    trades: list[int] | None = None,
    returns: list[float] | None = None,
) -> dict[str, np.ndarray]:
    count = len(sharpes)
    return {
        "sharpe": np.asarray(sharpes, dtype=float),
        "completed_trades": np.asarray([6] * count if trades is None else trades),
        "mean_holding_sessions": np.asarray([5.0] * count if mean_holds is None else mean_holds),
        "max_holding_sessions": np.asarray([10.0] * count if max_holds is None else max_holds),
        "total_return": np.asarray([0.1] * count if returns is None else returns),
    }


class ResearchSelectionTests(unittest.TestCase):
    def select(
        self,
        *,
        row_count: int = 800,
        buys: tuple[float, ...] = (20.0, 30.0),
        targets: tuple[float, ...] = (1.1, 1.2),
        open_prices: np.ndarray | None = None,
        rsi_entry_rule: int = 0,
        selection_version: int | None = None,
    ) -> dict[str, object]:
        prices = np.arange(row_count, dtype=float) if open_prices is None else open_prices
        return select_strategy(
            prices,
            prices + 1.0,
            prices + 0.5,
            np.full(row_count, 25.0),
            np.zeros(row_count),
            np.asarray(buys),
            np.asarray(targets),
            100_000.0,
            0.0003,
            rsi_entry_rule,
            **({} if selection_version is None else {"selection_version": selection_version}),
        )

    def test_holdout_contains_only_frozen_winner_and_cannot_trigger_reranking(self) -> None:
        development = metrics([2.0, 1.5])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([-1.0], returns=[-0.2])],
        ) as evaluate:
            result = self.select()

        self.assertEqual(result["status"], "failed_holdout")
        self.assertEqual((result["buy_rsi"], result["profit_target_multiple"]), (20.0, 1.1))
        self.assertEqual(evaluate.call_count, 4)
        expected_blocks = [(0, 213), (213, 426), (426, 640), (640, 800)]
        for call, (start, stop) in zip(evaluate.call_args_list, expected_blocks, strict=True):
            np.testing.assert_array_equal(call.args[0], np.arange(start, stop, dtype=float))
        np.testing.assert_array_equal(evaluate.call_args_list[-1].args[5], [20.0])
        np.testing.assert_array_equal(evaluate.call_args_list[-1].args[6], [1.1])

    def test_changing_only_holdout_can_reject_but_cannot_change_selected_parameters(self) -> None:
        original = np.arange(800, dtype=float)
        changed = original.copy()
        changed[640:] = -999.0

        def evaluate(open_prices: np.ndarray, *args: object) -> dict[str, np.ndarray]:
            if len(open_prices) == 160:
                return metrics([-1.0], returns=[-0.3]) if open_prices[0] < 0 else metrics([1.0])
            return metrics([2.0, 1.5])

        with patch("leveraged_trader.research_selection.grid_research_metrics", side_effect=evaluate):
            passing = self.select(open_prices=original)
            rejected = self.select(open_prices=changed)

        self.assertEqual(passing["status"], "passed")
        self.assertEqual(rejected["status"], "failed_holdout")
        for field in (
            "buy_rsi", "profit_target_multiple", "selection_score", "development_sharpe_median",
            "development_sharpe_std", "development_completed_trades", "mean_holding_sessions",
            "development_min_annual_trades", "trade_activity_bonus",
        ):
            self.assertEqual(passing[field], rejected[field], field)

    def test_prefers_more_completed_trades_when_sharpe_and_holds_match(self) -> None:
        development = metrics([2.0, 2.0], trades=[3, 12])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([1.0])],
        ):
            result = self.select()

        self.assertEqual(result["buy_rsi"], 30.0)
        self.assertEqual(result["development_completed_trades"], 36)
        self.assertAlmostEqual(result["development_min_annual_trades"], 12 * 252.0 / 214)
        self.assertGreater(result["trade_activity_bonus"], 0.0)
        self.assertLess(result["trade_activity_bonus"], 0.5)

    def test_trade_activity_affects_score_beyond_an_exact_sharpe_tie(self) -> None:
        development = metrics([2.0, 1.95], trades=[3, 12])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([1.0])],
        ):
            result = self.select()

        self.assertEqual(result["buy_rsi"], 30.0)

    def test_bounded_activity_reward_preserves_a_material_sharpe_advantage(self) -> None:
        development = metrics([2.5, 2.0], trades=[3, 200])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([1.0])],
        ):
            result = self.select()

        self.assertEqual(result["buy_rsi"], 20.0)

    def test_activity_cannot_rescue_a_nonpositive_sharpe_and_holding_score(self) -> None:
        development = metrics([0.1, 0.1], trades=[200, 200])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development],
        ) as evaluate:
            result = self.select()

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["selection_score"])
        self.assertIsNone(result["development_min_annual_trades"])
        self.assertIsNone(result["trade_activity_bonus"])
        self.assertEqual(evaluate.call_count, 3)

    def test_busy_regimes_cannot_mask_a_low_trade_rate_in_another_block(self) -> None:
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[
                metrics([2.0, 2.0], trades=[50, 6]),
                metrics([2.0, 2.0], trades=[3, 6]),
                metrics([2.0, 2.0], trades=[50, 6]),
                metrics([1.0]),
            ],
        ):
            result = self.select()

        self.assertEqual(result["buy_rsi"], 30.0)
        self.assertEqual(result["development_block_completed_trades"], [6, 6, 6])
        self.assertEqual(result["development_completed_trades"], 18)

    def test_equal_trade_rates_receive_equal_rewards_across_history_lengths(self) -> None:
        results = []
        for row_count, completed_trades in ((900, 6), (1800, 12)):
            development = metrics([2.0], trades=[completed_trades])
            with patch(
                "leveraged_trader.research_selection.grid_research_metrics",
                side_effect=[development, development, development, metrics([1.0])],
            ):
                results.append(self.select(row_count=row_count, buys=(20.0,), targets=(1.1,)))

        self.assertEqual(results[1]["development_completed_trades"], 2 * results[0]["development_completed_trades"])
        for field in ("development_min_annual_trades", "trade_activity_bonus", "selection_score"):
            self.assertEqual(results[0][field], results[1][field], field)

    def test_more_trades_break_a_tie_created_by_neighbor_score_caps(self) -> None:
        development = metrics([2.0, 2.0], trades=[3, 12])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([1.0])],
        ):
            result = self.select(targets=(1.1, 1.1))

        self.assertEqual(result["buy_rsi"], 30.0)

    def test_holdout_trade_count_cannot_change_the_development_winner_or_reward(self) -> None:
        development = metrics([2.0, 2.0], trades=[3, 12])
        results = []
        for completed_trades in (1, 100):
            with patch(
                "leveraged_trader.research_selection.grid_research_metrics",
                side_effect=[development, development, development, metrics([1.0], trades=[completed_trades])],
            ):
                results.append(self.select())

        self.assertEqual(results[0]["status"], "failed_holdout")
        self.assertEqual(results[1]["status"], "passed")
        for field in (
            "buy_rsi", "profit_target_multiple", "selection_score", "development_completed_trades",
            "development_min_annual_trades", "trade_activity_bonus",
        ):
            self.assertEqual(results[0][field], results[1][field], field)

    def test_version_one_preserves_the_original_count_independent_tie_break(self) -> None:
        development = metrics([2.0, 2.0], trades=[3, 12])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([1.0])],
        ):
            result = self.select(selection_version=1)

        self.assertEqual(result["buy_rsi"], 20.0)
        self.assertEqual(result["version"], 1)

    def test_prefers_shorter_holds_when_financial_results_match(self) -> None:
        development = metrics([2.0, 2.0], mean_holds=[80.0, 5.0], max_holds=[120.0, 10.0])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([1.0])],
        ):
            result = self.select()

        self.assertEqual(result["buy_rsi"], 30.0)
        self.assertEqual(result["mean_holding_sessions"], 5.0)
        self.assertEqual(result["max_holding_sessions"], 10)

    def test_longest_hold_is_penalized_even_when_average_holds_match(self) -> None:
        development = metrics([2.0, 2.0], mean_holds=[10.0, 10.0], max_holds=[150.0, 15.0])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([1.0])],
        ):
            result = self.select()

        self.assertEqual(result["buy_rsi"], 30.0)

    def test_consistent_regimes_beat_one_exceptional_regime(self) -> None:
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[metrics([1.0, 1.5]), metrics([8.0, 1.5]), metrics([1.0, 1.5]), metrics([1.0])],
        ):
            result = self.select()

        self.assertEqual(result["buy_rsi"], 30.0)
        self.assertEqual(result["development_sharpe_std"], 0.0)

    def test_stable_parameter_region_beats_an_isolated_sharpe_peak(self) -> None:
        development = metrics([1.8, 1.8, 4.0, 0.2, 0.2])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([1.0])],
        ):
            result = self.select(buys=(10.0, 20.0, 30.0, 40.0, 50.0), targets=(1.1,) * 5)

        self.assertEqual(result["buy_rsi"], 10.0)
        self.assertEqual(result["candidate_count"], 5)
        self.assertEqual(result["development_sharpe_median"], 1.8)

    def test_invalid_neighbor_does_not_validate_an_isolated_candidate(self) -> None:
        development = metrics([3.0, 3.0], trades=[6, 0])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development],
        ) as evaluate:
            result = self.select(targets=(1.1, 1.1))

        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(evaluate.call_count, 3)

    def test_insufficient_history_never_evaluates_candidates(self) -> None:
        with patch("leveraged_trader.research_selection.grid_research_metrics") as evaluate:
            result = self.select(row_count=629)

        self.assertEqual(result["status"], "insufficient_history")
        self.assertIsNone(result["selection_score"])
        evaluate.assert_not_called()

    def test_one_sparse_or_losing_development_block_cannot_be_hidden_by_other_blocks(self) -> None:
        failures = (
            metrics([2.0, 2.0], trades=[2, 0]),
            metrics([2.0, 2.0], returns=[0.0, -0.1]),
            metrics([0.0, float("nan")]),
        )
        for failing_block in failures:
            with (
                self.subTest(failing_block=failing_block),
                patch(
                    "leveraged_trader.research_selection.grid_research_metrics",
                    side_effect=[metrics([3.0, 3.0]), failing_block, metrics([3.0, 3.0])],
                ) as evaluate,
            ):
                result = self.select()
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertEqual(evaluate.call_count, 3)

    def test_positive_holdout_requires_completed_trades(self) -> None:
        development = metrics([2.0, 1.5])
        with patch(
            "leveraged_trader.research_selection.grid_research_metrics",
            side_effect=[development, development, development, metrics([4.0], trades=[1])],
        ):
            result = self.select()

        self.assertEqual(result["status"], "failed_holdout")
        self.assertEqual(result["holdout_completed_trades"], 1)

    def test_fallback_is_selective_for_each_entry_direction(self) -> None:
        for direction, expected in ((0, 20.0), (RSI_ENTRY_UPPER, 30.0)):
            with self.subTest(direction=direction):
                self.assertEqual(self.select(row_count=20, rsi_entry_rule=direction)["buy_rsi"], expected)

    def test_duplicate_parameter_pairs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            self.select(buys=(20.0, 20.0), targets=(1.1, 1.1))


if __name__ == "__main__":
    unittest.main()
