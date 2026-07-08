from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from leveraged_trader.backtest import performance_summary
from leveraged_trader.config import RISK_FREE_SYMBOL, BacktestConfig
from leveraged_trader.indicators import compute_rsi
from leveraged_trader.storage import (
    _synchronize_market_data_history,
    ensure_rsi_values,
    init_state_db,
    process_asset_grid,
    strategy_config_fingerprint,
    strategy_state_generation,
    strategy_state_matches_config,
)


def sample_strategy_data(periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    asset_close = np.linspace(100.0, 140.0, periods)
    signal_close = pd.Series(
        [100.0 - i for i in range(periods // 2)]
        + [80.0 + i for i in range(periods - periods // 2)],
        index=dates,
    )
    return pd.DataFrame(
        {
            "TQQQ_Open": asset_close,
            "TQQQ_High": asset_close + 1.0,
            "TQQQ_Low": asset_close - 1.0,
            "TQQQ_Close": asset_close,
            "TQQQ_Volume": 1_000_000,
            "QQQ_Open": signal_close,
            "QQQ_High": signal_close + 1.0,
            "QQQ_Low": signal_close - 1.0,
            "QQQ_Close": signal_close,
            "QQQ_Volume": 2_000_000,
            f"{RISK_FREE_SYMBOL}_Open": 5.0,
            f"{RISK_FREE_SYMBOL}_High": 5.0,
            f"{RISK_FREE_SYMBOL}_Low": 5.0,
            f"{RISK_FREE_SYMBOL}_Close": 5.0,
            f"{RISK_FREE_SYMBOL}_Volume": 0,
        },
        index=dates,
    )


class StorageOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        init_state_db(self.conn)
        self.cfg = BacktestConfig(rsi_period=3)

    def tearDown(self) -> None:
        self.conn.close()

    def process_grid(self, data: pd.DataFrame, *, rebuild: bool) -> None:
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0, 70.0],
            profit_target_values=[1.05, 1.50],
            rebuild=rebuild,
        )

    def test_authoritative_history_identical_sync_reads_once_without_writes(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]

        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        changes_before = self.conn.total_changes
        statements: list[str] = []
        self.conn.set_trace_callback(statements.append)
        try:
            revised = _synchronize_market_data_history(self.conn, history, "TQQQ")
        finally:
            self.conn.set_trace_callback(None)

        market_statements = [
            statement.upper()
            for statement in statements
            if "MARKET_DATA" in statement.upper()
        ]
        self.assertFalse(revised)
        self.assertEqual(self.conn.total_changes, changes_before)
        self.assertEqual(
            sum(statement.lstrip().startswith("SELECT") for statement in market_statements),
            1,
        )
        self.assertFalse(
            any(
                statement.lstrip().startswith(("INSERT", "UPDATE", "DELETE"))
                for statement in market_statements
            )
        )

    def test_authoritative_history_tail_append_writes_only_new_session(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        self.assertFalse(_synchronize_market_data_history(self.conn, history.iloc[:-1], "TQQQ"))
        changes_before = self.conn.total_changes

        revised = _synchronize_market_data_history(self.conn, history, "TQQQ")

        self.assertFalse(revised)
        self.assertEqual(self.conn.total_changes - changes_before, 1)

    def test_authoritative_history_correction_updates_only_changed_session(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        revision = history.copy()
        revision.loc[revision.index[10], "TQQQ_Close"] = 123.45
        changes_before = self.conn.total_changes

        revised = _synchronize_market_data_history(self.conn, revision, "TQQQ")

        self.assertTrue(revised)
        self.assertEqual(self.conn.total_changes - changes_before, 1)

    def test_authoritative_history_preserves_float_comparison_tolerance(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        revision = history.copy()
        revision.loc[revision.index[10], "TQQQ_Close"] += 5e-11
        changes_before = self.conn.total_changes

        self.assertFalse(_synchronize_market_data_history(self.conn, revision, "TQQQ"))
        self.assertEqual(self.conn.total_changes, changes_before)

        revision.loc[revision.index[10], "TQQQ_Close"] += 1e-8
        self.assertTrue(_synchronize_market_data_history(self.conn, revision, "TQQQ"))
        self.assertEqual(self.conn.total_changes - changes_before, 1)

    def test_authoritative_history_removal_and_backfill_are_revisions(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        removed_date = history.index[10]
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        changes_before = self.conn.total_changes

        revised_removal = _synchronize_market_data_history(
            self.conn,
            history.drop(index=removed_date),
            "TQQQ",
        )

        self.assertTrue(revised_removal)
        self.assertEqual(self.conn.total_changes - changes_before, 1)
        changes_before = self.conn.total_changes

        revised_backfill = _synchronize_market_data_history(self.conn, history, "TQQQ")

        self.assertTrue(revised_backfill)
        self.assertEqual(self.conn.total_changes - changes_before, 1)

    def test_authoritative_history_null_values_do_not_create_false_corrections(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"]
        history.loc[history.index[10], "TQQQ_Volume"] = np.nan
        self.assertFalse(_synchronize_market_data_history(self.conn, history, "TQQQ"))
        changes_before = self.conn.total_changes

        revised = _synchronize_market_data_history(self.conn, history, "TQQQ")

        self.assertFalse(revised)
        self.assertEqual(self.conn.total_changes, changes_before)

    def test_authoritative_history_missing_columns_fails_before_writing(self) -> None:
        history = self.canonical_histories(sample_strategy_data())["TQQQ"].drop(
            columns="TQQQ_Volume"
        )
        changes_before = self.conn.total_changes

        with self.assertRaisesRegex(ValueError, "TQQQ_Volume"):
            _synchronize_market_data_history(self.conn, history, "TQQQ")

        self.assertEqual(self.conn.total_changes, changes_before)

    def test_presynchronized_authoritative_histories_skip_only_their_sync_calls(self) -> None:
        data = sample_strategy_data()
        histories = self.canonical_histories(data)
        _synchronize_market_data_history(self.conn, histories["QQQ"], "QQQ")
        _synchronize_market_data_history(self.conn, histories[RISK_FREE_SYMBOL], RISK_FREE_SYMBOL)

        with patch(
            "leveraged_trader.storage._synchronize_market_data_history",
            wraps=_synchronize_market_data_history,
        ) as synchronize:
            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories=histories,
                presynchronized_authoritative_symbols={"QQQ", RISK_FREE_SYMBOL},
            )

        self.assertEqual(
            [call.args[2] for call in synchronize.call_args_list],
            ["TQQQ"],
        )

    def test_presynchronized_symbol_must_have_an_authoritative_history(self) -> None:
        data = sample_strategy_data()

        with self.assertRaisesRegex(ValueError, "Presynchronized market symbols"):
            process_asset_grid(
                self.conn,
                data,
                self.cfg,
                "TQQQ",
                "QQQ",
                [30.0],
                [1.5],
                rebuild=True,
                authoritative_histories=self.canonical_histories(data),
                presynchronized_authoritative_symbols={"SPY"},
            )

    @staticmethod
    def canonical_histories(data: pd.DataFrame, asset_symbol: str = "TQQQ") -> dict[str, pd.DataFrame]:
        return {
            asset_symbol: data[[column for column in data if column.startswith(f"{asset_symbol}_")]].copy(),
            "QQQ": data[[column for column in data if column.startswith("QQQ_")]].copy(),
            RISK_FREE_SYMBOL: data[[column for column in data if column.startswith(f"{RISK_FREE_SYMBOL}_")]].copy(),
        }

    def test_process_asset_grid_stores_summaries_for_all_configs_and_only_best_equity(self) -> None:
        data = sample_strategy_data()

        self.process_grid(data, rebuild=True)

        summary_count = self.conn.execute("SELECT COUNT(*) FROM strategy_summary").fetchone()[0]
        state_count = self.conn.execute("SELECT COUNT(*) FROM strategy_state").fetchone()[0]
        equity_configs = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT buy_rsi, profit_target_multiple
                FROM strategy_equity
            )
            """
        ).fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertEqual(summary_count, 4)
        self.assertEqual(state_count, 4)
        self.assertEqual(equity_configs, 1)
        self.assertEqual(equity_rows, len(data))

    def test_process_asset_grid_reports_exact_grid_compute_duration(self) -> None:
        observed: list[float] = []
        with patch("leveraged_trader.storage.time") as mock_time:
            mock_time.perf_counter.side_effect = [10.0, 12.5]
            process_asset_grid(
                self.conn,
                sample_strategy_data(),
                self.cfg,
                "TQQQ",
                "QQQ",
                buy_rsi_values=[30.0],
                profit_target_values=[1.50],
                rebuild=True,
                grid_compute_observer=observed.append,
            )

        self.assertEqual(observed, [2.5])

    def test_best_summary_matches_stored_best_equity_curve(self) -> None:
        data = sample_strategy_data()

        self.process_grid(data, rebuild=True)

        best_summary = pd.read_sql_query(
            """
            SELECT *
            FROM strategy_summary
            ORDER BY sharpe DESC
            LIMIT 1
            """,
            self.conn,
        ).iloc[0]
        equity_df = pd.read_sql_query(
            """
            SELECT date, equity, risk_free_return
            FROM strategy_equity
            ORDER BY date
            """,
            self.conn,
            parse_dates=["date"],
        )
        curve_summary = performance_summary(
            equity_df.set_index("date")["equity"],
            equity_df.set_index("date")["risk_free_return"],
        )

        self.assertAlmostEqual(best_summary["sharpe"], curve_summary["Sharpe"])
        self.assertAlmostEqual(best_summary["total_return"], curve_summary["Total Return"])
        self.assertAlmostEqual(best_summary["cagr"], curve_summary["CAGR"])

    def test_process_asset_grid_updates_summaries_and_rewrites_single_best_equity_curve(self) -> None:
        data = sample_strategy_data()
        first_window = data.iloc[:25]
        update_window = data.iloc[24:]

        self.process_grid(first_window, rebuild=True)
        self.process_grid(update_window, rebuild=False)

        end_dates = self.conn.execute("SELECT DISTINCT end_date FROM strategy_summary").fetchall()
        equity_configs = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT buy_rsi, profit_target_multiple
                FROM strategy_equity
            )
            """
        ).fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertEqual(end_dates, [(data.index[-1].date().isoformat(),)])
        self.assertEqual(equity_configs, 1)
        self.assertEqual(equity_rows, len(data))

    def test_update_uses_legacy_equity_rollup_when_summary_rollup_is_missing(self) -> None:
        data = sample_strategy_data()
        first_window = data.iloc[:25]
        update_window = data.iloc[24:]

        process_asset_grid(
            self.conn,
            first_window,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0],
            profit_target_values=[1.50],
            rebuild=True,
        )
        self.conn.execute("UPDATE strategy_summary SET first_equity = NULL")

        process_asset_grid(
            self.conn,
            update_window,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=[30.0],
            profit_target_values=[1.50],
            rebuild=False,
        )

        trading_days = self.conn.execute(
            "SELECT trading_days FROM strategy_summary"
        ).fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertEqual(trading_days, len(data))
        self.assertEqual(equity_rows, len(data))

    def test_corrected_persisted_session_rebuilds_rsi_and_strategy_state(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        previous_rsi = self.conn.execute(
            "SELECT rsi FROM rsi_values WHERE signal_symbol = 'QQQ' ORDER BY date DESC LIMIT 1"
        ).fetchone()[0]

        revision = data.iloc[-1:].copy()
        revision.loc[:, "QQQ_Close"] = 200.0
        self.process_grid(revision, rebuild=False)

        stored_rsi = self.conn.execute(
            "SELECT rsi FROM rsi_values WHERE signal_symbol = 'QQQ' ORDER BY date DESC LIMIT 1"
        ).fetchone()[0]
        expected_close = data["QQQ_Close"].copy()
        expected_close.iloc[-1] = 200.0
        expected_rsi = compute_rsi(expected_close, self.cfg.rsi_period).iloc[-1]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]

        self.assertNotEqual(stored_rsi, previous_rsi)
        self.assertAlmostEqual(stored_rsi, expected_rsi)
        self.assertEqual(equity_rows, len(data))

    def test_incremental_rsi_matches_neutral_flat_full_recompute(self) -> None:
        close = pd.Series([100.0] * 10, index=pd.date_range("2026-01-02", periods=10, freq="B"))

        ensure_rsi_values(self.conn, "QQQ", self.cfg.rsi_period, close.iloc[:5], rebuild=True)
        incremental = ensure_rsi_values(self.conn, "QQQ", self.cfg.rsi_period, close, rebuild=False)
        expected = compute_rsi(close, self.cfg.rsi_period)

        self.assertEqual(float(incremental.dropna().iloc[-1]), 50.0)
        pd.testing.assert_series_equal(incremental, expected, check_names=False)

    def test_full_asset_history_correction_rebuilds_resumed_state(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)

        revision = data.copy()
        revision.loc[revision.index[10], ["TQQQ_Open", "TQQQ_High", "TQQQ_Low", "TQQQ_Close"]] = [20, 21, 19, 20]
        self.process_grid(revision, rebuild=False)

        stored_close = self.conn.execute(
            "SELECT close FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
            (revision.index[10].date().isoformat(),),
        ).fetchone()[0]
        summary_end_date = self.conn.execute(
            "SELECT DISTINCT end_date FROM strategy_summary"
        ).fetchone()[0]
        self.assertEqual(stored_close, 20.0)
        self.assertEqual(summary_end_date, revision.index[-1].date().isoformat())

    def test_authoritative_asset_history_removes_deleted_session_and_rebuilds(self) -> None:
        data = sample_strategy_data()
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(data),
        )

        removed_date = data.index[10]
        reduced_data = data.drop(index=removed_date)
        process_asset_grid(
            self.conn,
            reduced_data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=self.canonical_histories(reduced_data),
        )

        persisted = self.conn.execute(
            "SELECT COUNT(*) FROM market_data WHERE symbol = 'TQQQ' AND date = ?",
            (removed_date.date().isoformat(),),
        ).fetchone()[0]
        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]
        self.assertEqual(persisted, 0)
        self.assertEqual(equity_rows, len(reduced_data))

    def test_authoritative_asset_history_backfill_rebuilds_from_the_missing_session(self) -> None:
        data = sample_strategy_data()
        backfilled_date = data.index[10]
        incomplete_data = data.drop(index=backfilled_date)
        process_asset_grid(
            self.conn,
            incomplete_data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(incomplete_data),
        )

        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=self.canonical_histories(data),
        )

        equity_rows = self.conn.execute("SELECT COUNT(*) FROM strategy_equity").fetchone()[0]
        self.assertEqual(equity_rows, len(data))

    def test_authoritative_benchmark_history_removal_invalidates_other_assets(self) -> None:
        data = sample_strategy_data()
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(data),
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(upro_data, "UPRO"),
            strategy_fingerprint=fingerprint,
        )

        reduced_risk_free = data.drop(index=data.index[10])[
            [column for column in data if column.startswith(f"{RISK_FREE_SYMBOL}_")]
        ]
        histories = self.canonical_histories(data)
        histories[RISK_FREE_SYMBOL] = reduced_risk_free
        process_asset_grid(
            self.conn,
            data.drop(index=data.index[10]),
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=histories,
            strategy_fingerprint=fingerprint,
        )

        upro_state_count = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = 'UPRO'"
        ).fetchone()[0]
        self.assertEqual(upro_state_count, 0)

    def test_authoritative_signal_history_removal_invalidates_all_signal_dependents(self) -> None:
        data = sample_strategy_data()
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(data),
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=self.canonical_histories(upro_data, "UPRO"),
            strategy_fingerprint=fingerprint,
        )

        removed_date = data.index[10]
        histories = self.canonical_histories(data)
        histories["QQQ"] = histories["QQQ"].drop(index=removed_date)
        process_asset_grid(
            self.conn,
            data.drop(index=removed_date),
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            authoritative_histories=histories,
            strategy_fingerprint=fingerprint,
        )

        upro_state_count = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = 'UPRO' AND signal_symbol = 'QQQ'"
        ).fetchone()[0]
        persisted_signal = self.conn.execute(
            "SELECT COUNT(*) FROM market_data WHERE symbol = 'QQQ' AND date = ?",
            (removed_date.date().isoformat(),),
        ).fetchone()[0]
        self.assertEqual(upro_state_count, 0)
        self.assertEqual(persisted_signal, 0)

    def test_saved_best_curve_uses_the_same_forward_filled_benchmark_calendar(self) -> None:
        data = sample_strategy_data()
        missing_benchmark_date = data.index[10]
        histories = self.canonical_histories(data)
        histories[RISK_FREE_SYMBOL] = histories[RISK_FREE_SYMBOL].drop(index=missing_benchmark_date)

        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            authoritative_histories=histories,
        )

        equity_dates = {
            row[0]
            for row in self.conn.execute("SELECT date FROM strategy_equity").fetchall()
        }
        summary_trading_days = self.conn.execute(
            "SELECT trading_days FROM strategy_summary"
        ).fetchone()[0]

        self.assertIn(missing_benchmark_date.date().isoformat(), equity_dates)
        self.assertEqual(len(equity_dates), len(data))
        self.assertEqual(summary_trading_days, len(data))

    def test_benchmark_invalidation_advances_the_persisted_generation(self) -> None:
        data = sample_strategy_data()
        self.process_grid(data, rebuild=True)
        before_generation = strategy_state_generation(self.conn)

        revision = data.copy()
        revision.loc[revision.index[10], f"{RISK_FREE_SYMBOL}_Close"] = 3.0
        self.process_grid(revision, rebuild=False)

        self.assertEqual(strategy_state_generation(self.conn), before_generation + 1)

    def test_risk_free_history_correction_invalidates_other_assets(self) -> None:
        data = sample_strategy_data()
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0], [1.5])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )

        revision = data.copy()
        revision.loc[revision.index[10], f"{RISK_FREE_SYMBOL}_Close"] = 3.0
        process_asset_grid(
            self.conn,
            revision,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=False,
            strategy_fingerprint=fingerprint,
        )

        upro_state_count = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = 'UPRO'"
        ).fetchone()[0]
        self.assertEqual(upro_state_count, 0)

    def test_signal_correction_invalidates_other_assets_using_that_signal(self) -> None:
        data = sample_strategy_data()
        fingerprint = strategy_config_fingerprint(self.cfg, [30.0, 70.0], [1.05, 1.50])
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0, 70.0],
            [1.05, 1.50],
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )
        upro_data = data.rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            upro_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0, 70.0],
            [1.05, 1.50],
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )

        revision = data.iloc[-1:].copy()
        revision.loc[:, "QQQ_Close"] = 200.0
        process_asset_grid(
            self.conn,
            revision,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0, 70.0],
            [1.05, 1.50],
            rebuild=False,
            strategy_fingerprint=fingerprint,
        )

        upro_states = self.conn.execute(
            "SELECT COUNT(*) FROM strategy_state WHERE asset_symbol = 'UPRO' AND signal_symbol = 'QQQ'"
        ).fetchone()[0]
        self.assertEqual(upro_states, 0)

        process_asset_grid(
            self.conn,
            upro_data.iloc[-1:],
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0, 70.0],
            [1.05, 1.50],
            rebuild=False,
            strategy_fingerprint=fingerprint,
        )
        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "UPRO",
                "QQQ",
                self.cfg,
                [30.0, 70.0],
                [1.05, 1.50],
            )
        )

    def test_short_lived_asset_rebuild_preserves_canonical_signal_rsi_history(self) -> None:
        data = sample_strategy_data()
        signal_history = data[[column for column in data.columns if column.startswith("QQQ_")]].copy()
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            signal_history=signal_history,
        )
        short_data = data.iloc[-12:].rename(columns=lambda column: column.replace("TQQQ", "UPRO"))
        process_asset_grid(
            self.conn,
            short_data,
            self.cfg,
            "UPRO",
            "QQQ",
            [30.0],
            [1.5],
            rebuild=True,
            signal_history=signal_history,
        )

        rsi_count = self.conn.execute(
            "SELECT COUNT(*) FROM rsi_values WHERE signal_symbol = 'QQQ' AND rsi_period = ?",
            (self.cfg.rsi_period,),
        ).fetchone()[0]
        self.assertEqual(rsi_count, len(signal_history))

    def test_strategy_state_requires_exact_grid_and_backtest_fingerprint(self) -> None:
        data = sample_strategy_data()
        buy_rsi_values = [30.0, 70.0]
        profit_target_values = [1.05, 1.50]
        fingerprint = strategy_config_fingerprint(self.cfg, buy_rsi_values, profit_target_values)
        process_asset_grid(
            self.conn,
            data,
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=buy_rsi_values,
            profit_target_values=profit_target_values,
            rebuild=True,
            strategy_fingerprint=fingerprint,
        )

        self.assertTrue(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                buy_rsi_values,
                profit_target_values,
            )
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                self.cfg,
                [25.0, 65.0],
                profit_target_values,
            )
        )

        changed_grid = [25.0, 65.0]
        process_asset_grid(
            self.conn,
            data.iloc[-1:],
            self.cfg,
            "TQQQ",
            "QQQ",
            buy_rsi_values=changed_grid,
            profit_target_values=profit_target_values,
            rebuild=False,
        )
        actual_pairs = {
            (float(buy_rsi), float(profit_target_multiple))
            for buy_rsi, profit_target_multiple in self.conn.execute(
                "SELECT buy_rsi, profit_target_multiple FROM strategy_state"
            ).fetchall()
        }
        self.assertEqual(
            actual_pairs,
            {(buy_rsi, target) for buy_rsi in changed_grid for target in profit_target_values},
        )
        self.assertFalse(
            strategy_state_matches_config(
                self.conn,
                "TQQQ",
                "QQQ",
                BacktestConfig(rsi_period=4),
                changed_grid,
                profit_target_values,
            )
        )


if __name__ == "__main__":
    unittest.main()
