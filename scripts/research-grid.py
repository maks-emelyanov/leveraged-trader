#!/usr/bin/env python3
"""Reproduce the fixed, read-only historical grid comparison.

Run validation, freeze, confirmation, then decide. Confirmation reads its frozen
leader and cannot rerank the grids. This script never writes a trading database.
The evaluator pins selection version 1 and the original outer utility so the
archived protocol remains reproducible after production selection changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from leveraged_trader.grid_research import (
    BASE_COST,
    DEFAULT_GRIDS,
    INITIAL_CAPITAL,
    STRESS_COST,
    evaluate_asset,
    evaluate_period,
    summarize,
)
from leveraged_trader.indicators import compute_rsi
from leveraged_trader.storage import _market_arrays, _persisted_strategy_signal_symbol


def input_digest(dates: np.ndarray, arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256(np.asarray(dates, dtype="datetime64[ns]").astype("<i8").tobytes())
    for values in arrays:
        digest.update(values.astype("<f8").tobytes())
    return digest.hexdigest()


def load_inputs(db: Path, cutoff: str, min_training: int, cohort: set[str] | None = None) -> tuple[list, list]:
    prepared, coverage = [], []
    # A single read snapshot keeps source revisions from mixing histories.
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("BEGIN")
        assets = pd.read_sql_query(
            "SELECT symbol, rsi_symbol, workflow, source FROM workflow_assets ORDER BY symbol", conn,
        )
        cached: dict[str, pd.DataFrame] = {}
        rsi_cache: dict[str, pd.Series] = {}

        def history(symbol: str) -> pd.DataFrame:
            if symbol not in cached:
                frame = pd.read_sql_query(
                    "SELECT date, open, high, low, close, volume FROM market_data "
                    "WHERE symbol = ? AND date <= ? ORDER BY date",
                    conn, params=(symbol, cutoff), parse_dates=["date"],
                ).set_index("date")
                cached[symbol] = frame
            return cached[symbol]

        benchmark = history("^IRX").rename(columns=lambda name: f"^IRX_{name.title()}")
        for asset in assets.itertuples(index=False):
            direction = "inverse" if asset.workflow.lower() == "short" else "long"
            item = dict(asset=asset.symbol, signal=asset.rsi_symbol, direction=direction, source=asset.source)
            if cohort is not None and asset.symbol not in cohort:
                continue
            frame = history(asset.symbol).rename(columns=lambda name, symbol=asset.symbol: f"{symbol}_{name.title()}")
            if len(frame.loc[:"2021-12-31"]) < min_training:
                coverage.append({**item, "included": False, "reason": "insufficient_pre_2022_price_history"})
                continue
            signal_storage = _persisted_strategy_signal_symbol(conn, asset.symbol, asset.rsi_symbol)
            try:
                if signal_storage not in rsi_cache:
                    rsi_cache[signal_storage] = compute_rsi(history(signal_storage)["close"], 14)
                rsi = rsi_cache[signal_storage]
                frame = frame.join(benchmark.reindex(frame.index, method="ffill"))
                arrays = _market_arrays(frame, rsi, asset.symbol)
                first_signal = int(np.flatnonzero(np.isfinite(arrays[3]))[0])
                frame = frame.iloc[first_signal:]
                arrays = tuple(np.ascontiguousarray(values[first_signal:]) for values in arrays)
                training_count = int(np.count_nonzero(frame.index < pd.Timestamp("2022-01-01")))
                if training_count < min_training:
                    coverage.append({**item, "included": False, "reason": "insufficient_pre_2022_aligned_history"})
                    continue
                item.update(
                    included=True, sessions=len(frame), first_date=str(frame.index[0].date()),
                    last_date=str(frame.index[-1].date()), input_sha256=input_digest(frame.index.to_numpy(), arrays),
                    signal_storage=signal_storage,
                )
                coverage.append(item)
                prepared.append((asset.symbol, asset.rsi_symbol, direction, frame.index.to_numpy(), arrays))
            except (ValueError, IndexError, KeyError) as exc:
                coverage.append({**item, "included": False, "reason": str(exc)})
    finally:
        conn.close()
    return prepared, coverage


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_confirmation_inputs(prepared: list, coverage: list, output: Path, cohort: set[str], cutoff: str) -> None:
    prior = pd.read_csv(output / "validation_coverage.csv").set_index("asset")
    current = {row["asset"]: row for row in coverage if row["included"]}
    if {row[0] for row in prepared} != cohort:
        raise ValueError("Confirmation must contain the entire frozen validation cohort.")
    for asset, signal, direction, dates, arrays in prepared:
        before = dates <= np.datetime64(cutoff)
        prefix_digest = input_digest(dates[before], tuple(values[before] for values in arrays))
        if (
            signal != prior.loc[asset, "signal"] or direction != prior.loc[asset, "direction"]
            or current[asset]["signal_storage"] != prior.loc[asset, "signal_storage"]
            or prefix_digest != prior.loc[asset, "input_sha256"]
        ):
            raise ValueError(f"Validation inputs or signal mapping changed for {asset}; cannot confirm this study.")


def freeze_choice(summary: dict, protocol_sha256: str) -> dict:
    counts = {grid.name: len(grid.pairs("long")) for grid in DEFAULT_GRIDS}
    ranked = sorted(
        summary["grids"],
        key=lambda row: (-row["utility"], counts[row["grid"]], row["grid"] != "current49", row["grid"]),
    )
    if not ranked:
        raise ValueError("No eligible validation outcomes to rank.")
    leader = ranked[0]["grid"]
    gates = {}
    comparison = None
    if leader != "current49":
        comparison = next(row for row in summary["paired"] if row["grid"] == leader and row["direction"] == "all")
        directions = [row for row in summary["paired"] if row["grid"] == leader and row["direction"] != "all"]
        gates = {
            "adjusted_paired_lower_bound_positive": comparison["familywise_lower_95_utility"] > 0,
            "broad_improvement": comparison["discordant_groups_utility"] >= 10
            and (comparison["improvement_share_utility"] or 0) >= 0.6,
            "no_direction_deterioration": bool(directions) and all(row["delta_utility"] >= 0 for row in directions),
            "stress_improvement_nonnegative": comparison["delta_stress_utility"] >= 0,
            "matched_holding_no_worse": comparison["matched_exposed_signal_groups"] > 0 and all(
                comparison[f"matched_exposed_delta_{metric}_holding_sessions"] <= 0 for metric in ("mean", "max")
            ),
        }
    return {
        "protocol_sha256": protocol_sha256, "leader": leader, "incumbent": "current49",
        "ranking": [row["grid"] for row in ranked], "validation_gates": gates,
        "validation_adoption_eligible": bool(gates) and all(gates.values()),
        "comparison": comparison,
        "confirmation_status": "pending; frozen before confirmation evaluation",
    }


def finalize_decision(frozen: dict, confirmation: dict) -> dict:
    if frozen["protocol_sha256"] != confirmation["protocol_sha256"]:
        raise ValueError("Confirmation belongs to a different protocol.")
    leader, incumbent = frozen["leader"], frozen["incumbent"]
    if {row["grid"] for row in confirmation["grids"]} != {leader, incumbent}:
        raise ValueError("Confirmation must contain only the frozen leader and incumbent.")
    comparison = next(
        (row for row in confirmation["paired"] if row["grid"] == leader and row["direction"] == "all"), None,
    )
    confirmation_passed = comparison is not None and all(
        comparison[field] >= 0 for field in ("delta_utility", "delta_stress_utility")
    )
    adopt = frozen["validation_adoption_eligible"] and confirmation_passed
    return {
        "protocol_sha256": frozen["protocol_sha256"], "leader": leader,
        "production_grid": leader if adopt else incumbent, "adopt_replacement": adopt,
        "validation_gates": frozen["validation_gates"],
        "confirmation_noninferiority_gate": confirmation_passed,
        "confirmation_activated_asset_periods": {
            row["grid"]: row["active_asset_years"] for row in confirmation["grids"]
        },
        "reason": (
            "All registered adoption gates passed; still a retrospective result."
            if adopt else (
                "No replacement met all registered adoption gates; retain incumbent without claiming it optimal."
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("strategy_state.sqlite"))
    parser.add_argument("--study-dir", type=Path, default=Path("research/grid-search-2026-10-12"))
    parser.add_argument("--phase", required=True, choices=("validation", "freeze", "confirmation", "decide"))
    args = parser.parse_args()
    output = args.study_dir
    protocol_bytes = (output / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    registered = {g["name"]: g for g in protocol["grids"]}
    actual = {g.name: json.loads(json.dumps(asdict(g))) for g in DEFAULT_GRIDS}
    if registered != actual:
        raise ValueError("Study grid definitions differ from the frozen protocol.")
    expected = dict(
        rsi_period=14, initial_capital=INITIAL_CAPITAL, base_cost_per_side_bps=BASE_COST * 10000,
        stress_cost_per_side_bps=STRESS_COST * 10000, min_test_sessions=126,
    )
    if any(not np.isclose(protocol[key], value) for key, value in expected.items()):
        raise ValueError("Protocol settings differ from the fixed research evaluator.")
    lower_quantile = 0.05 / (2 * (len(DEFAULT_GRIDS) - 1))
    if not (
        np.isclose(protocol["bootstrap"]["lower_quantile"], lower_quantile)
        and np.isclose(protocol["bootstrap"]["upper_quantile"], 1 - lower_quantile)
    ):
        raise ValueError("Bootstrap multiplicity adjustment differs from the protocol.")
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if args.phase == "freeze":
        destination = output / "frozen_choice.json"
        if destination.exists():
            raise FileExistsError("Grid-family choice is already frozen.")
        summary = json.loads((output / "validation_summary.json").read_text())
        if summary["protocol_sha256"] != protocol_sha256:
            raise ValueError("Validation summary belongs to a different protocol.")
        write_json(destination, freeze_choice(summary, protocol_sha256))
        print(f"Saved frozen choice to {destination}.", flush=True)
        return
    if args.phase == "decide":
        destination = output / "decision.json"
        if destination.exists():
            raise FileExistsError("Study decision is already recorded.")
        frozen = json.loads((output / "frozen_choice.json").read_text())
        confirmation = json.loads((output / "confirmation_summary.json").read_text())
        if frozen["protocol_sha256"] != protocol_sha256:
            raise ValueError("Frozen choice belongs to a different protocol.")
        write_json(destination, finalize_decision(frozen, confirmation))
        print(f"Saved study decision to {destination}.", flush=True)
        return
    destinations = [output / f"{args.phase}_{suffix}" for suffix in ("rows.csv", "summary.json", "coverage.csv")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Study results already exist; use a separate study directory for a rerun.")
    if args.phase == "validation":
        cutoff = f"{max(protocol['validation_years'])}-12-31"
        cohort = None
        grids = DEFAULT_GRIDS
    else:
        frozen = json.loads((output / "frozen_choice.json").read_text())
        if frozen["protocol_sha256"] != protocol_sha256:
            raise ValueError("Frozen grid choice belongs to a different protocol.")
        if frozen["leader"] not in registered:
            raise ValueError("Unknown frozen grid-family choice.")
        cutoff = protocol["confirmation_end"]
        cohort = set(pd.read_csv(output / "validation_rows.csv")["asset"])
        grids = tuple(g for g in DEFAULT_GRIDS if g.name in {"current49", frozen["leader"]})
    prepared, coverage = load_inputs(args.db, cutoff, protocol["min_training_sessions"], cohort)
    if args.phase == "confirmation":
        verify_confirmation_inputs(prepared, coverage, output, cohort, f"{max(protocol['validation_years'])}-12-31")
    print(f"Prepared {len(prepared)} assets through {cutoff}; source opened read-only.", flush=True)
    rows = []
    for i, values in enumerate(prepared, 1):
        if args.phase == "validation":
            result = evaluate_asset(
                *values, grids=grids, validation_years=tuple(protocol["validation_years"]),
                min_training_sessions=protocol["min_training_sessions"],
            )
        else:
            result = evaluate_period(
                *values, grids=grids, test_start=protocol["confirmation_start"], test_end=cutoff,
                min_training_sessions=protocol["min_training_sessions"],
            )
        rows.extend(result)
        if i % 10 == 0 or i == len(prepared):
            print(f"Completed {i}/{len(prepared)} assets; {len(rows)} grid-period records.", flush=True)
    frame = pd.DataFrame(rows)
    summary = summarize(
        rows, bootstrap_samples=protocol["bootstrap"]["resamples"],
        bootstrap_seed=protocol["bootstrap"]["seed"], comparison_count=len(DEFAULT_GRIDS) - 1,
    )
    summary["protocol_sha256"] = protocol_sha256
    frame.to_csv(destinations[0], index=False)
    write_json(destinations[1], summary)
    pd.DataFrame(coverage).to_csv(destinations[2], index=False)
    print(f"Saved {args.phase} results to {output}.", flush=True)


if __name__ == "__main__":
    main()
