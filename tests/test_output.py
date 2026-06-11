from __future__ import annotations

import io
import unittest

import pandas as pd
from rich.console import Console

from leveraged_trader.output import WorkflowReporter


class OutputTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
