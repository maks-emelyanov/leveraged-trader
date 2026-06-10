from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from leveraged_trader.config import AlpacaOrderConfig, BacktestConfig, UniverseConfig
from leveraged_trader.workflow import run_resumable_optimizations_async


class WorkflowAsyncTests(unittest.TestCase):
    def test_workflow_concurrency_limits_asset_tasks(self) -> None:
        workflow_assets = pd.DataFrame(
            [
                {"symbol": "AAA", "name": "A", "rsi_symbol": "AAA"},
                {"symbol": "BBB", "name": "B", "rsi_symbol": "BBB"},
                {"symbol": "CCC", "name": "C", "rsi_symbol": "CCC"},
            ]
        )
        empty_orders = pd.DataFrame(columns=["Asset", "Action"])
        active_tasks = 0
        max_active_tasks = 0

        async def fake_process_asset(**_: object) -> None:
            nonlocal active_tasks, max_active_tasks
            active_tasks += 1
            max_active_tasks = max(max_active_tasks, active_tasks)
            await asyncio.sleep(0.01)
            active_tasks -= 1

        async def immediate_to_thread(func: object, /, *args: object, **kwargs: object) -> object:
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "state.sqlite")
            output_dir = str(Path(tmp) / "outputs")
            with (
                patch("leveraged_trader.workflow.asyncio.to_thread", new=immediate_to_thread),
                patch("leveraged_trader.workflow._initialize_state_db"),
                patch(
                    "leveraged_trader.workflow._reconcile_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(columns=["Action"]),
                ),
                patch(
                    "leveraged_trader.workflow._load_or_refresh_workflow_assets_for_db",
                    return_value=workflow_assets,
                ),
                patch("leveraged_trader.workflow._process_workflow_asset", new=fake_process_asset),
                patch(
                    "leveraged_trader.workflow._build_reports_for_db",
                    return_value=(
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                        pd.DataFrame(),
                    ),
                ),
                patch(
                    "leveraged_trader.workflow._submit_alpaca_paper_buy_orders_for_db",
                    return_value=empty_orders,
                ),
                patch(
                    "leveraged_trader.workflow._load_alpaca_managed_positions_for_db",
                    return_value=pd.DataFrame(),
                ),
                patch("leveraged_trader.workflow._write_workflow_outputs") as mock_write_outputs,
            ):
                asyncio.run(
                    run_resumable_optimizations_async(
                        mode="update",
                        db_path=db_path,
                        base_cfg=BacktestConfig(),
                        universe_cfg=UniverseConfig(),
                        buy_rsi_values=[30],
                        profit_target_values=[1.5],
                        alpaca_cfg=AlpacaOrderConfig(),
                        output_dir=output_dir,
                        workflow_concurrency=2,
                    )
                )

        self.assertEqual(max_active_tasks, 2)
        mock_write_outputs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
