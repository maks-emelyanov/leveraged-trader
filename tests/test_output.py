from __future__ import annotations

import io
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
from rich.console import Console

from leveraged_trader.output import DEFAULT_NON_TERMINAL_WIDTH, TableColumn, WorkflowReporter


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

    def test_run_header_renders_cron_friendly_run_context(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        started_at_utc = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)

        reporter.run_header(
            started_at_utc=started_at_utc,
            mode="update",
            db_path="state.sqlite",
            output_dir="outputs",
            workflow_concurrency=4,
        )
        output = console.export_text(styles=False)
        lines = [line for line in output.splitlines() if line]
        expected_started_local = started_at_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")

        self.assertIn(f"Workflow Run: {expected_started_local}", output)
        self.assertNotIn("Started local", output)
        self.assertNotIn("Started UTC", output)
        self.assertNotIn("2026-01-02T14:30:00+00:00", output)
        self.assertIn("state.sqlite", output)
        self.assertIn("outputs", output)
        self.assertIn("4", output)
        self.assertTrue(lines[0].startswith("\u256d\u2500 Workflow Run: "))
        self.assertIn("\u2502 Download workers  4", output)
        self.assertTrue(lines[-1].startswith("\u2570"))
        self.assertTrue(lines[-1].endswith("\u256f"))
        self.assertNotEqual(lines[-1], "\u2500" * 100)

    def test_dataframe_caps_terminal_rows_with_caption(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.dataframe(
            "Limited Table",
            pd.DataFrame({"Asset": ["AAA", "BBB", "CCC"]}),
            [TableColumn("Asset", no_wrap=True)],
            empty_message="empty",
            max_rows=2,
            truncated_detail="full data written to limited.csv",
        )
        output = console.export_text(styles=False)

        self.assertIn("AAA", output)
        self.assertIn("BBB", output)
        self.assertNotIn("CCC", output)
        self.assertIn("Showing 2 of 3 rows; full data written to limited.csv.", output)

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
                    "Position ID": 99,
                    "Display ID": 1,
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
        self.assertNotIn("99", output)
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
            "Merged current ETFs/ETNs": 1,
            "Current long leveraged ETFs/ETNs found": 1,
            "Executable long leveraged ETFs/ETNs selected": 1,
            "RSI mappings needing review": 0,
            "Audit rows parsed": 10,
        }
        universe.attrs["universe_db_path"] = "state.sqlite"

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("All Long Leveraged ETFs From Nasdaq Universe", output)
        self.assertIn("Current long leveraged ETFs/ETNs found", output)
        self.assertIn("Executable long leveraged ETFs/ETNs selected", output)
        self.assertIn("RSI mappings needing review", output)
        self.assertNotIn("Current ETFs in Nasdaq table", output)
        self.assertNotIn("Merged current ETFs/ETNs", output)
        self.assertNotIn("Audit rows parsed", output)
        self.assertNotIn("Saved SQLite tables", output)
        self.assertIn("Asset", output)
        self.assertIn("Name", output)
        self.assertIn("RSI", output)
        self.assertIn("TQQQ", output)
        self.assertIn("ProShares UltraPro QQQ", output)
        self.assertIn("QQQ", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)

    def test_universe_assets_renders_all_rows(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame(
            [
                {
                    "symbol": f"A{index:03d}",
                    "name": f"Leveraged ETF {index:03d}",
                    "rsi_symbol": f"R{index:03d}",
                }
                for index in range(76)
            ]
        )

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("A000", output)
        self.assertIn("A074", output)
        self.assertIn("A075", output)
        self.assertNotIn("Showing 75 of 76 rows", output)

    def test_universe_assets_renders_failed_source_details(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
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
        universe.attrs["universe_degraded"] = True
        universe.attrs["workflow_source_failures"] = [
            {
                "source": "Direxion",
                "source_type": "issuer_etf",
                "status": "source_error",
                "error": "HTTPError: 403 Client Error: Forbidden",
            }
        ]

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("Workflow universe is degraded", output)
        self.assertIn("Failed Workflow Universe Sources", output)
        self.assertIn("Direxion", output)
        self.assertIn("source_error", output)
        self.assertIn("HTTPError: 403", output)

    def test_universe_assets_renders_active_listing_failure_details(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
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
        universe.attrs["universe_degraded"] = True
        universe.attrs["active_listing_source_failures"] = [
            {
                "source": "other_listed",
                "status": "error",
                "error": "offline",
            }
        ]

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("Workflow universe is degraded", output)
        self.assertIn("Failed Active Listing Sources", output)
        self.assertIn("other_listed", output)
        self.assertIn("Offline", output)

    def test_universe_assets_renders_rsi_mapping_review_details(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        universe = pd.DataFrame(
            [
                {
                    "symbol": "FOOU",
                    "name": "T-REX 2X Long ExampleCorp Daily Target ETF",
                    "rsi_symbol": "FOOU",
                }
            ]
        )
        universe.attrs["rsi_mapping_review"] = [
            {
                "symbol": "FOOU",
                "name": "T-REX 2X Long ExampleCorp Daily Target ETF",
                "rsi_symbol": "FOOU",
                "mapping_reason": "single-stock-style product did not expose a reliable underlying ticker",
            }
        ]

        reporter.universe_assets(universe)
        output = console.export_text(styles=False)

        self.assertIn("Some RSI mappings need review", output)
        self.assertIn("RSI Mappings Needing Review", output)
        self.assertIn("FOOU", output)
        self.assertIn("Single-stock-style product", output)

    def test_optimization_summary_excludes_rows_below_trade_and_sharpe_thresholds(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=120, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.optimization_summary(
            pd.DataFrame(
                [
                    {
                        "Asset": "STXX",
                        "RSI Symbol": "STX",
                        "Start Date": "2026-01-02",
                        "Trading Days": 44,
                        "Buy RSI": 20.0,
                        "Sell Return Multiple": 1.1,
                        "Trades Executed": 0,
                        "Total Return": 0.0,
                        "CAGR": 0.0,
                        "Sharpe": -1751.799,
                        "Kelly Fraction": 0.0,
                        "Max Drawdown": 0.0,
                    }
                ]
            )
        )
        output = console.export_text(styles=False)

        self.assertIn("No strategies with at least 2 trades and Sharpe >= 1.0.", output)
        self.assertNotIn("STXX", output)
        self.assertNotIn("N/A", output)
        self.assertNotIn("-1751", output)

    def test_optimization_summary_shows_only_rows_with_enough_trades_and_sharpe(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=160, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)
        rows = [
            {
                "Asset": "GOOD",
                "RSI Symbol": "GD",
                "Start Date": "2026-01-02",
                "Trading Days": 2,
                "Buy RSI": 30.0,
                "Sell Return Multiple": 1.5,
                "Trades Executed": 2,
                "Total Return": 0.1,
                "CAGR": 0.2,
                "Sharpe": 1.0,
                "Kelly Fraction": 0.4,
                "Max Drawdown": -0.1,
            },
            {
                "Asset": "GREAT",
                "RSI Symbol": "GT",
                "Start Date": "2026-01-02",
                "Trading Days": 2,
                "Buy RSI": 30.0,
                "Sell Return Multiple": 1.5,
                "Trades Executed": 2,
                "Total Return": 0.2,
                "CAGR": 0.3,
                "Sharpe": 1.25,
                "Kelly Fraction": 0.5,
                "Max Drawdown": -0.1,
            },
            {
                "Asset": "LOW",
                "RSI Symbol": "LW",
                "Start Date": "2026-01-02",
                "Trading Days": 2,
                "Buy RSI": 30.0,
                "Sell Return Multiple": 1.5,
                "Trades Executed": 2,
                "Total Return": 0.1,
                "CAGR": 0.2,
                "Sharpe": 0.9999,
                "Kelly Fraction": 0.4,
                "Max Drawdown": -0.1,
            },
            {
                "Asset": "ONE",
                "RSI Symbol": "ON",
                "Start Date": "2026-01-02",
                "Trading Days": 2,
                "Buy RSI": 30.0,
                "Sell Return Multiple": 1.5,
                "Trades Executed": 1,
                "Total Return": 0.1,
                "CAGR": 0.2,
                "Sharpe": 5.0,
                "Kelly Fraction": 0.4,
                "Max Drawdown": -0.1,
            },
        ]
        rows.extend(
            [
                {
                    "Asset": f"A{index:03d}",
                    "RSI Symbol": f"R{index:03d}",
                    "Start Date": "2026-01-02",
                    "Trading Days": 2,
                    "Buy RSI": 30.0,
                    "Sell Return Multiple": 1.5,
                    "Trades Executed": 2,
                    "Total Return": 0.1,
                    "CAGR": 0.2,
                    "Sharpe": 0.8,
                    "Kelly Fraction": 0.4,
                    "Max Drawdown": -0.1,
                }
                for index in range(76)
            ]
        )

        reporter.optimization_summary(pd.DataFrame(rows))
        output = console.export_text(styles=False)

        self.assertIn("GOOD", output)
        self.assertIn("GREAT", output)
        self.assertNotIn("LOW", output)
        self.assertNotIn("ONE", output)
        self.assertNotIn("A075", output)
        self.assertNotIn("Showing", output)

    def test_signal_report_keeps_data_sufficiency_columns_when_wide(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=160, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.signal_report(
            "Buy Signals For Next Open",
            pd.DataFrame(
                [
                    {
                        "Asset": "ONX",
                        "RSI Symbol": "ON",
                        "Date": "2026-06-26",
                        "Start Date": "2026-05-28",
                        "Trading Days": 21,
                        "Latest RSI": 37.25,
                        "Buy RSI": 48.0,
                        "Sell Return Multiple": 1.2,
                        "Trades Executed": 2,
                        "Sharpe": 5.7836,
                        "In Position": False,
                        "Pending Action": "buy",
                    }
                ]
            ),
            empty_message="empty",
        )
        output = console.export_text(styles=False)

        self.assertIn("Start", output)
        self.assertIn("Latest RSI", output)
        self.assertIn("2026-05-28", output)
        self.assertIn("37.25", output)

    def test_signal_report_omits_some_columns_when_narrow(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.signal_report(
            "Buy Signals For Next Open",
            pd.DataFrame(
                [
                    {
                        "Asset": "ONX",
                        "RSI Symbol": "ON",
                        "Date": "2026-06-26",
                        "Start Date": "2026-05-28",
                        "Trading Days": 21,
                        "Latest RSI": 37.25,
                        "Buy RSI": 48.0,
                        "Sell Return Multiple": 1.2,
                        "Trades Executed": 2,
                        "Sharpe": 5.7836,
                        "In Position": False,
                        "Pending Action": "buy",
                    }
                ]
            ),
            empty_message="empty",
        )
        output = console.export_text(styles=False)

        self.assertNotIn("Start", output)
        self.assertNotIn("Latest RSI", output)
        self.assertNotIn("2026-05-28", output)
        self.assertNotIn("37.25", output)
        self.assertIn("Days", output)
        self.assertIn("21", output)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)

    def test_buy_signal_eligibility_summary_counts_managed_and_live_skips(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame({"Asset": ["AAA", "BBB", "CCC"]}),
            eligible_buy_signals=pd.DataFrame({"Asset": ["CCC"]}),
            order_results=pd.DataFrame(
                {
                    "Asset": ["AAA", "BBB", "CCC"],
                    "Status": ["managed", "held", "submitted"],
                }
            ),
        )
        output = console.export_text(styles=False)

        self.assertIn("Buy Signal Eligibility", output)
        self.assertIn("Eligible buy signals", output)
        self.assertIn("1 / 3", output)
        self.assertIn("Submitted/existing buys", output)
        self.assertIn("Skipped: active managed", output)
        self.assertIn("Skipped: Alpaca/live preflight", output)
        self.assertNotIn("Eligible after active managed filter", output)
        self.assertNotIn("Submitted or existing Alpaca buys", output)
        self.assertNotIn("Skipped by Alpaca/live preflight", output)

    def test_buy_signal_eligibility_summary_omits_zero_detail_rows(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.buy_signal_eligibility_summary(
            buy_signals=pd.DataFrame(),
            eligible_buy_signals=pd.DataFrame(),
            order_results=pd.DataFrame(),
        )
        output = console.export_text(styles=False)

        self.assertIn("Eligible buy signals", output)
        self.assertIn("0 / 0", output)
        self.assertNotIn("Submitted/existing buys", output)
        self.assertNotIn("Skipped: active managed", output)
        self.assertNotIn("Skipped: Alpaca/live preflight", output)

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

        self.assertNotIn("Mode", output)
        self.assertNotIn("SQLite database", output)
        self.assertNotIn("Workflow concurrency", output)
        self.assertIn("Buy RSI values", output)
        self.assertIn("Sell return multiples", output)
        self.assertIn("none", output)

    def test_workflow_footer_renders_elapsed_time_and_divider(self) -> None:
        console = Console(file=io.StringIO(), record=True, width=100, color_system=None, no_color=True)
        reporter = WorkflowReporter(console=console)

        reporter.workflow_footer(65.25)
        output = console.export_text(styles=False)
        lines = output.splitlines()

        self.assertIn("Workflow finished in 1m 05.25s.", output)
        self.assertNotIn("Workflow Benchmark", output)
        self.assertNotIn("CPU", output)
        self.assertNotIn("Peak RSS", output)
        self.assertEqual(lines[-1], "\u2500" * 100)
        for line in output.splitlines():
            self.assertLessEqual(len(line), 100)


if __name__ == "__main__":
    unittest.main()
