from __future__ import annotations

import io
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
from rich.console import Console

from leveraged_trader.benchmark import WorkflowBenchmark
from leveraged_trader.output import DEFAULT_NON_TERMINAL_WIDTH, WorkflowReporter


class OutputTests(unittest.TestCase):
    def test_default_reporter_uses_wide_width_for_non_terminal_output(self) -> None:
        output_buffer = io.StringIO()
        created_consoles: list[Console] = []

        def console_factory(**kwargs: object) -> Console:
            console = Console(file=output_buffer, record=True, color_system=None, **kwargs)
            created_consoles.append(console)
            return console

        with patch("leveraged_trader.output.Console", side_effect=console_factory):
            reporter = WorkflowReporter(no_color=True)

        self.assertEqual(len(created_consoles), 2)
        self.assertFalse(created_consoles[0].is_terminal)
        self.assertEqual(reporter.console.width, DEFAULT_NON_TERMINAL_WIDTH)

    def test_default_reporter_keeps_terminal_width_auto_sized(self) -> None:
        terminal_console = Console(file=io.StringIO(), force_terminal=True, color_system=None, no_color=True)

        with patch("leveraged_trader.output.Console", return_value=terminal_console) as mock_console:
            reporter = WorkflowReporter(no_color=True)

        mock_console.assert_called_once_with(no_color=True)
        self.assertIs(reporter.console, terminal_console)

    def test_injected_console_width_is_respected(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        self.assertIs(reporter.console, console)
        self.assertEqual(reporter.console.width, 120)

    def test_default_non_terminal_width_keeps_table_output_readable(self) -> None:
        output_buffer = io.StringIO()

        def console_factory(**kwargs: object) -> Console:
            return Console(file=output_buffer, record=True, color_system=None, **kwargs)

        with patch("leveraged_trader.output.Console", side_effect=console_factory):
            reporter = WorkflowReporter(no_color=True)

        reporter.reconciliation(
            pd.DataFrame(
                [
                    {
                        "Position ID": 1,
                        "Asset": "TQQQ",
                        "Action": "sell",
                        "Status": "submitted",
                        "Qty": 2,
                        "Limit Price": 150.0,
                        "Message": "submitted one-time GTC limit sell at frozen target price",
                    }
                ]
            )
        )
        output = reporter.console.export_text(styles=False)

        self.assertIn("Submitted GTC limit sell at frozen target price", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), DEFAULT_NON_TERMINAL_WIDTH)

    def test_reconciliation_table_wraps_message_without_truncating(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=80, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        reconciliation = pd.DataFrame(
            [
                {
                    "Position ID": 1,
                    "Asset": "TQQQ",
                    "Action": "sell",
                    "Status": "submitted",
                    "Buy Client Order ID": "rsi-buy-TQQQ-20260102",
                    "Sell Client Order ID": "rsi-exit-TQQQ-1",
                    "Qty": 2,
                    "Limit Price": 150.0,
                    "Alpaca Order ID": "alpaca-sell-order-1",
                    "Message": "submitted one-time GTC limit sell at frozen target price",
                }
            ]
        )

        reporter.reconciliation(reconciliation)
        output = console.export_text(styles=False)

        self.assertIn("Submitted GTC limit sell", output)
        self.assertIn("frozen target price", output)
        self.assertNotIn("ta...", output)
        self.assertNotIn("rsi-buy-TQQQ", output)
        self.assertNotIn("rsi-exit-TQQQ", output)
        self.assertNotIn("alpaca-sell-order-1", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 80)

    def test_asset_run_summary_sorts_by_workflow_index(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.asset_run_summary(
            [
                {
                    "Workflow #": 2,
                    "Asset": "BBB",
                    "RSI Symbol": "BBB",
                    "Action": "Updating",
                    "Rows": 20,
                    "Status": "done",
                    "Message": "Processed 20 rows",
                },
                {
                    "Workflow #": 1,
                    "Asset": "AAA",
                    "RSI Symbol": "AAA",
                    "Action": "Updating",
                    "Rows": 10,
                    "Status": "done",
                    "Message": "Processed 10 rows",
                },
            ]
        )
        output = console.export_text(styles=False)

        self.assertLess(output.index("AAA"), output.index("BBB"))

    def test_universe_assets_renders_as_rich_table(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame(
            [
                {
                    "symbol": "TQQQ",
                    "name": "ProShares UltraPro QQQ",
                    "rsi_symbol": "QQQ",
                }
            ]
        )
        universe.attrs["universe_title"] = "All Long Leveraged ETFs From Nasdaq Universe"
        universe.attrs["universe_counts"] = {
            "Current ETFs in Nasdaq table": 1,
            "Current long leveraged ETFs found": 1,
        }
        universe.attrs["universe_db_path"] = "state.sqlite"

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("All Long Leveraged ETFs From Nasdaq Universe", output)
        self.assertIn("Current ETFs in Nasdaq table", output)
        self.assertIn("Saved SQLite tables", output)
        self.assertIn("Asset", output)
        self.assertIn("Name", output)
        self.assertIn("RSI", output)
        self.assertIn("TQQQ", output)
        self.assertIn("ProShares UltraPro QQQ", output)
        self.assertIn("QQQ", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)

    def test_realized_pnl_summary_renders_percentage(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.realized_pnl_summary(
            pd.DataFrame(
                [
                    {
                        "Closed Positions": 2,
                        "Complete Closed Positions": 1,
                        "Incomplete Closed Positions": 1,
                        "Total Buy Cost": 200.0,
                        "Total Sell Value": 250.0,
                        "Realized P/L": 50.0,
                        "Realized P/L %": 25.0,
                    }
                ]
            )
        )
        output = console.export_text(styles=False)

        self.assertIn("Closed Managed Alpaca Realized P/L", output)
        self.assertIn("50.00", output)
        self.assertIn("25.00%", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)

    def test_settings_handles_empty_grid_values(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.settings(
            mode="update",
            db_path="state.sqlite",
            workflow_concurrency=1,
            risk_free_symbol="^IRX",
            buy_rsi_values=[],
            profit_target_values=[],
        )
        output = console.export_text(styles=False)

        self.assertIn("Buy RSI values", output)
        self.assertIn("Sell return multiples", output)
        self.assertIn("none", output)

    def test_benchmark_report_renders_runtime_and_memory_summary(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.benchmark_report(
            WorkflowBenchmark(
                status="completed",
                started_at_utc=datetime(2026, 1, 2, 14, 30, tzinfo=UTC),
                finished_at_utc=datetime(2026, 1, 2, 14, 31, 5, tzinfo=UTC),
                wall_seconds=65.25,
                cpu_seconds=32.5,
                cpu_utilization_percent=49.81,
                peak_rss_mb=123.45,
                current_rss_mb=120.25,
                asset_count=2,
                completed_asset_count=1,
                skipped_asset_count=1,
                rows_processed=10,
                workflow_concurrency=4,
            )
        )
        output = console.export_text(styles=False)

        self.assertIn("Workflow Benchmark", output)
        self.assertIn("1m 05.25s", output)
        self.assertIn("49.81%", output)
        self.assertIn("123.45 MB", output)
        self.assertIn("1 completed / 2 total; 1 skipped", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)


if __name__ == "__main__":
    unittest.main()
