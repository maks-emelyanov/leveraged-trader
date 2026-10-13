"""Reproduce the fixed, retrospective v1/v2 comparison from a read-only cache.

Run from the repository root with NUMBA_NUM_THREADS=2 and the project Python.
Use --db for the source cache and --output for a new artifact directory.
This compares already-observed histories; it is not independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from leveraged_trader.grid_research import BASE_COST, DEFAULT_GRIDS, INITIAL_CAPITAL
from leveraged_trader.optimized_backtest import RSI_ENTRY_LOWER, RSI_ENTRY_UPPER
from leveraged_trader.research_selection import (
    DEVELOPMENT_BLOCKS,
    TRADE_ACTIVITY_ANNUAL_SCALE,
    TRADE_ACTIVITY_WEIGHT,
    select_strategy,
)

ROOT = Path(__file__).resolve().parents[2]
CUTOFF = "2026-10-09"
MIN_TRAINING = 1260
METRICS = (
    "development_completed_trades", "observed_development_min_annual_trades",
    "development_sharpe_median", "development_sharpe_std", "mean_holding_sessions",
    "max_holding_sessions", "holdout_completed_trades", "holdout_annual_trades",
    "holdout_sharpe", "holdout_total_return", "holdout_mean_holding_sessions",
    "holdout_max_holding_sessions",
)


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def metric_summary(records: list[dict], version: str) -> dict:
    summary = {}
    for metric in METRICS:
        values = [row[version][metric] for row in records if row[version][metric] is not None]
        summary[metric] = {
            "count": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
        }
    return summary


def load_snapshot_inputs(module: object, source_db: Path) -> tuple[list, list, dict]:
    """Read a consistent SQLite backup without requiring the live cache to stop."""
    with tempfile.TemporaryDirectory(prefix="trade-activity-comparison-") as directory:
        snapshot = Path(directory) / "snapshot.sqlite"
        with (
            sqlite3.connect(source_db.resolve().as_uri() + "?mode=ro", uri=True) as source,
            sqlite3.connect(snapshot) as destination,
        ):
            source.backup(destination)
        before_digest = digest(snapshot)
        prepared, coverage = module.load_inputs(snapshot, CUTOFF, MIN_TRAINING)
        after_digest = digest(snapshot)
        if before_digest != after_digest:
            raise ValueError("Source snapshot changed during read-only input loading.")
        provenance = {
            "source_database": str(source_db.resolve()),
            "source_access": "SQLite mode=ro; consistent backup into a temporary research snapshot",
            "snapshot_access": "SQLite mode=ro",
            "snapshot_sha256_before": before_digest, "snapshot_sha256_after": after_digest,
            "snapshot_unchanged": True,
            "snapshot_retained": False,
            "latest_observation": max(
                (row["last_date"] for row in coverage if row["included"]), default=None,
            ),
        }
    return prepared, coverage, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "strategy_state.sqlite")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output
    outputs = ("coverage.csv", "selections.csv", "selections.json", "summary.json")
    if any((output / name).exists() for name in outputs):
        raise FileExistsError("Use a new output directory; comparison artifacts already exist.")
    if TRADE_ACTIVITY_WEIGHT != 0.5 or TRADE_ACTIVITY_ANNUAL_SCALE != 12.0:
        raise ValueError("This comparison fixes the activity weight at 0.5 and annual scale at 12.")
    output.mkdir(parents=True, exist_ok=True)
    script = ROOT / "scripts/research-grid.py"
    spec = importlib.util.spec_from_file_location("historical_grid_inputs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared, coverage, snapshot_provenance = load_snapshot_inputs(module, args.db)
    coverage_by_asset = {row["asset"]: row for row in coverage}
    grid = next(grid for grid in DEFAULT_GRIDS if grid.name == "current49")
    records = []
    for asset, signal, direction, dates, arrays in prepared:
        pairs = grid.pairs(direction)
        rule = RSI_ENTRY_LOWER if direction == "long" else RSI_ENTRY_UPPER
        record = {
            "asset": asset, "signal": signal, "direction": direction, "sessions": len(dates),
            "first_date": str(dates[0].astype("datetime64[D]")),
            "last_date": str(dates[-1].astype("datetime64[D]")),
            "input_sha256": coverage_by_asset[asset]["input_sha256"],
        }
        for version in (1, 2):
            selection = select_strategy(
                *arrays, pairs[:, 0], pairs[:, 1], INITIAL_CAPITAL, BASE_COST, rule,
                selection_version=version,
            )
            blocks = np.diff(np.linspace(0, selection["development_sessions"], DEVELOPMENT_BLOCKS + 1, dtype=int))
            trades = selection["development_block_completed_trades"]
            selection["observed_development_min_annual_trades"] = (
                float(np.min(252.0 * np.asarray(trades) / blocks)) if trades else None
            )
            selection["holdout_annual_trades"] = (
                252.0 * selection["holdout_completed_trades"] / selection["holdout_sessions"]
                if selection["selection_score"] is not None else None
            )
            record[f"v{version}"] = selection
        record["parameters_changed"] = any(
            record["v1"][field] != record["v2"][field]
            for field in ("buy_rsi", "profit_target_multiple")
        )
        records.append(record)
    matched_selected = [
        row for row in records if all(row[version]["selection_score"] is not None for version in ("v1", "v2"))
    ]
    matched_passed = [
        row for row in records if all(row[version]["status"] == "passed" for version in ("v1", "v2"))
    ]
    source_paths = (
        "scripts/research-grid.py", "leveraged_trader/research_selection.py",
        "leveraged_trader/research_metrics.py", "leveraged_trader/optimized_backtest.py",
        "leveraged_trader/grid_research.py", "leveraged_trader/indicators.py",
        "leveraged_trader/storage.py",
    )
    summary = {
        "comparison": "Fixed production-selection v1 versus v2 on identical cached histories",
        "limitations": [
            "Retrospective comparison of already-observed history, not independent validation.",
            "Current surviving asset universe and present signal mappings; histories differ across assets.",
            "Means weight assets equally and are descriptive, not an aggregated investable portfolio.",
            "Passed means selector acceptance only; production also requires full-history Sharpe >= 1.",
            "No weights, grid, or settings were tuned after inspecting these comparison results.",
        ],
        "settings": {
            "cutoff_inclusive": CUTOFF, "min_pre_2022_training_sessions": MIN_TRAINING,
            "initial_capital": INITIAL_CAPITAL, "trading_cost_per_side": BASE_COST,
            "rsi_period": 14, "grid": asdict(grid), "selection_versions": [1, 2],
            "trade_activity_weight": TRADE_ACTIVITY_WEIGHT,
            "trade_activity_annual_scale": TRADE_ACTIVITY_ANNUAL_SCALE,
            "annualization_sessions": 252,
            "activity_formula": "0.5 * min_block_annual_completed_trades / (min_block_annual_completed_trades + 12)",
            "development_fraction": 0.8, "development_blocks": DEVELOPMENT_BLOCKS,
            "holdout_fraction": 0.2,
        },
        "provenance": {
            **snapshot_provenance,
            "source_sha256": {path: digest(ROOT / path) for path in source_paths},
            "comparison_script_sha256": digest(Path(__file__)),
            "input_digests": "coverage.csv and each selections.json record",
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
        },
        "counts": {
            "included_assets": len(records), "excluded_assets": len(coverage) - len(records),
            "signal_proxies": len({row["signal"] for row in records}),
            "directions": dict(Counter(row["direction"] for row in records)),
            "parameters_changed": sum(row["parameters_changed"] for row in records),
            "matched_selected": len(matched_selected), "matched_passed": len(matched_passed),
            "status_transitions": dict(Counter(f'{row["v1"]["status"]} -> {row["v2"]["status"]}' for row in records)),
        },
        "versions": {
            version: {
                "statuses": dict(Counter(row[version]["status"] for row in records)),
                "selected_metrics": metric_summary(
                    [row for row in records if row[version]["selection_score"] is not None], version,
                ),
                "passed_metrics": metric_summary(
                    [row for row in records if row[version]["status"] == "passed"], version,
                ),
                "matched_selected_metrics": metric_summary(matched_selected, version),
                "matched_passed_metrics": metric_summary(matched_passed, version),
            } for version in ("v1", "v2")
        },
    }
    flat = []
    for record in records:
        row = {key: value for key, value in record.items() if key not in {"v1", "v2"}}
        for version in ("v1", "v2"):
            row.update({
                f"{version}_{key}": json.dumps(value) if isinstance(value, list) else value
                for key, value in record[version].items()
            })
        flat.append(row)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(flat).to_csv(output / "selections.csv", index=False)
    for name, content in (("selections.json", records), ("summary.json", summary)):
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
