#!/usr/bin/env python3
"""Run the registered fine-to-coarse grid study without touching trading state.

Phases: prepare, validation, freeze, confirmation, benchmark, summarize.
Each phase preserves prior artifacts; use a new --study-dir for an independent run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from leveraged_trader.grid_research import BASE_COST, DEFAULT_GRIDS, INITIAL_CAPITAL, GridSpec
from leveraged_trader.indicators import compute_rsi
from leveraged_trader.research_selection import select_strategy
from leveraged_trader.storage import _market_arrays, _persisted_strategy_signal_symbol

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2026-10-08"
LADDER = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (8, 8), (10, 10), (15, 15), (20, 20), (30, 30), (40, 49), (40, 98))


def axis(first: int, last: int, step: int, shifted: bool = False) -> tuple[int, ...]:
    offset = step // 2 if shifted else 0
    return tuple(sorted({first, last, *range(first + offset, last + 1, step)}))


def grid_definitions() -> tuple[list[GridSpec], list[dict]]:
    grids, metadata, names = [], [], set()

    def add(rsi_step: int, cents_step: int, family: str, shifted: bool = False) -> None:
        name = f"r{rsi_step}_t{cents_step}" + ("_shift" if shifted else "")
        if name in names:
            return
        grid = GridSpec(
            name,
            axis(10, 50, rsi_step, shifted),
            axis(50, 90, rsi_step, shifted),
            tuple(v / 100 for v in axis(102, 200, cents_step, shifted)),
        )
        grids.append(grid)
        metadata.append(
            dict(
                name=name,
                family=family,
                rsi_step=rsi_step,
                target_step=cents_step / 100,
                shifted=shifted,
                candidate_count=len(grid.pairs("long")),
            )
        )
        names.add(name)

    for rsi_step, cents_step in LADDER:
        add(rsi_step, cents_step, "joint")
    for rsi_step, _ in LADDER[1:]:
        add(rsi_step, 1, "rsi_only")
    for _, cents_step in LADDER[1:]:
        add(1, cents_step, "target_only")
    for rsi_step, cents_step in LADDER[1:]:
        add(rsi_step, cents_step, "phase_sensitivity", True)
    current = DEFAULT_GRIDS[0]
    grids.append(current)
    metadata.append(
        dict(name=current.name, family="incumbent", rsi_step=5, target_step=None, shifted=False, candidate_count=49)
    )
    return grids, metadata


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def input_digest(dates: np.ndarray, arrays: tuple[np.ndarray, ...]) -> str:
    value = hashlib.sha256(np.asarray(dates, dtype="datetime64[ns]").astype("<i8").tobytes())
    for array in arrays:
        value.update(array.astype("<f8").tobytes())
    return value.hexdigest()


def prepare(db: Path, output: Path) -> None:
    if (output / "protocol.json").exists():
        raise FileExistsError("Study already prepared; use a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    grids, metadata = grid_definitions()
    protocol = dict(
        study="Grid resolution, current robust selection version 2",
        cutoff=CUTOFF,
        validation_years=list(range(2013, 2023)),
        confirmation_years=list(range(2023, 2027)),
        min_training_sessions=630,
        min_test_sessions=126,
        selection_version=2,
        base_cost_per_side_bps=3,
        stress_cost_per_side_bps=10,
        initial_capital=100000,
        grids=[asdict(grid) for grid in grids],
        metadata=metadata,
        order=(
            "Evaluate exact fine baseline then progressively coarser joint grids, axis sweeps, phase shifts, incumbent."
        ),
        endpoints="Preserved at every spacing. Terminal joint grid has 2 RSI values x 2 targets.",
        outer_test=(
            "Annual flat-start frozen policy with expanding training; 2026 ends at cutoff. Not a daily "
            "reoptimization replay."
        ),
        cohort=(
            "Current cached products; eligibility separately requires prior 630 aligned sessions and 126 test sessions."
        ),
        primary_utility="Sharpe - .5*log1p(mean_hold/21) - .25*log1p(max_hold/63); unexposed cells zero.",
        activity="Selector v2 unchanged; outer annual completed trade rate and activity utility reported separately.",
        outliers=(
            "Reject unsupported parameter peaks using v2 development gates and neighbor cap. Keep every valid "
            "outcome in primary analysis; symmetric 10% proxy trimming and leave-one-proxy-out sensitivity only."
        ),
        bootstrap=dict(
            resamples=250000,
            seed=20261012,
            comparison_count=len(grids) - 1,
            clusters="signal proxy; keep assets/years paired",
            tails="two-sided Bonferroni .05/(2*comparison_count*4_metrics*3_direction_slices)",
        ),
        practical_margins=dict(utility=0.05, annual_period_return=0.01),
        evidence_requirements=dict(
            baseline_exposed_proxy_groups=10, discordant_proxy_groups=10, baseline_completed_trades=50
        ),
        decision=(
            "Only identify an optimum with supported noninferiority and a supported degradation boundary; "
            "otherwise report unresolved, exhaust the registered ladder, retain incumbent defaults."
        ),
        frozen_confirmation=(
            "Freeze an evidence-qualified candidate if any; otherwise smallest-grid exploratory leader by "
            "validation utility, plus baseline/incumbent and adjacent joint grids. Never promote exploratory "
            "ranking without evidence gates."
        ),
        runtime=(
            "Separate warmed uncached select_strategy timing, 5 deterministic representative assets, 3 repeats "
            "per grid, randomized order; no union cache timings used for speedup."
        ),
        prior_inspection=(
            "Historical data previously researched; retrospective confirmation, not pristine holdout. TQQQ "
            "pre-2022 prefix used only in a runtime pilot before registration."
        ),
        references=[
            "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html",
            "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf",
        ],
    )
    write_json(output / "protocol.json", protocol)
    # All source reads occur within one consistent read-only SQLite transaction.
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.execute("BEGIN")
        assets = pd.read_sql_query(
            "SELECT symbol, rsi_symbol, workflow, source FROM workflow_assets ORDER BY symbol", conn
        )
        cached, rsi_cache, coverage, prepared, packed = {}, {}, [], [], {}

        def history(symbol: str) -> pd.DataFrame:
            if symbol not in cached:
                cached[symbol] = pd.read_sql_query(
                    "SELECT date, open, high, low, close, volume FROM market_data "
                    "WHERE symbol=? AND date<=? ORDER BY date",
                    conn,
                    params=(symbol, CUTOFF),
                    parse_dates=["date"],
                ).set_index("date")
            return cached[symbol]

        benchmark = history("^IRX").rename(columns=lambda value: f"^IRX_{value.title()}")
        for asset in assets.itertuples(index=False):
            item = dict(
                asset=asset.symbol,
                signal=asset.rsi_symbol,
                direction="inverse" if asset.workflow.lower() == "short" else "long",
                source=asset.source,
            )
            frame = history(asset.symbol).rename(columns=lambda value, symbol=asset.symbol: f"{symbol}_{value.title()}")
            if len(frame) < 756:
                coverage.append({**item, "included": False, "reason": "fewer_than_756_price_sessions"})
                continue
            try:
                signal_storage = _persisted_strategy_signal_symbol(conn, asset.symbol, asset.rsi_symbol)
                if signal_storage not in rsi_cache:
                    rsi_cache[signal_storage] = compute_rsi(history(signal_storage)["close"], 14)
                frame = frame.join(benchmark.reindex(frame.index, method="ffill"))
                arrays = _market_arrays(frame, rsi_cache[signal_storage], asset.symbol)
                first_signal = int(np.flatnonzero(np.isfinite(arrays[3]))[0])
                dates = frame.index.to_numpy()[first_signal:]
                arrays = tuple(np.ascontiguousarray(array[first_signal:]) for array in arrays)
                if len(dates) < 756:
                    coverage.append({**item, "included": False, "reason": "fewer_than_756_aligned_sessions"})
                    continue
                item.update(
                    included=True,
                    sessions=len(dates),
                    first_date=str(dates[0].astype("datetime64[D]")),
                    last_date=str(dates[-1].astype("datetime64[D]")),
                    signal_storage=signal_storage,
                    input_sha256=input_digest(dates, arrays),
                    input_index=len(prepared),
                )
                prepared.append(item)
                packed[f"dates_{item['input_index']}"] = dates.astype("datetime64[ns]").astype(np.int64)
                packed[f"arrays_{item['input_index']}"] = np.stack(arrays)
                coverage.append(item)
            except (ValueError, IndexError, KeyError) as exc:
                coverage.append({**item, "included": False, "reason": str(exc)})
    np.savez_compressed(output / "inputs.npz", **packed)
    write_json(output / "inputs.json", prepared)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    sources = [
        "scripts/research-resolution.py",
        "leveraged_trader/grid_resolution.py",
        "leveraged_trader/grid_resolution_analysis.py",
        "leveraged_trader/research_selection.py",
        "leveraged_trader/research_metrics.py",
        "leveraged_trader/optimized_backtest.py",
        # Target rounding and return-domain checks are transitive simulator inputs.
        "leveraged_trader/pricing.py",
        "leveraged_trader/backtest.py",
        "leveraged_trader/indicators.py",
        "leveraged_trader/storage.py",
        "leveraged_trader/grid_research.py",
    ]
    write_json(
        output / "provenance.json",
        dict(
            source_database=str(db.resolve()),
            source_access="mode=ro; single transaction",
            protocol_sha256=digest(output / "protocol.json"),
            inputs_sha256=digest(output / "inputs.npz"),
            inputs_metadata_sha256=digest(output / "inputs.json"),
            source_sha256={name: digest(ROOT / name) for name in sources},
            numpy_version=np.__version__,
            pandas_version=pd.__version__,
            numba_threads=os.environ.get("NUMBA_NUM_THREADS"),
        ),
    )
    print(f"Prepared {len(prepared)} assets, {len(grids)} grids, immutable input archive.", flush=True)


def load(output: Path) -> tuple[dict, list, list, list]:
    protocol = json.loads((output / "protocol.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    if digest(output / "protocol.json") != provenance["protocol_sha256"]:
        raise ValueError("Registered protocol changed.")
    if digest(output / "inputs.npz") != provenance["inputs_sha256"]:
        raise ValueError("Frozen market inputs changed.")
    if digest(output / "inputs.json") != provenance["inputs_metadata_sha256"]:
        raise ValueError("Frozen input metadata changed.")
    for name, expected in provenance["source_sha256"].items():
        if not (ROOT / name).is_file() or digest(ROOT / name) != expected:
            raise ValueError(f"Registered source changed: {name}.")
    grids = [
        GridSpec(row["name"], tuple(row["long_rsi"]), tuple(row["short_rsi"]), tuple(row["targets"]))
        for row in protocol["grids"]
    ]
    prepared = []
    with np.load(output / "inputs.npz", allow_pickle=False) as archive:
        for row in json.loads((output / "inputs.json").read_text()):
            index = row["input_index"]
            dates = archive[f"dates_{index}"].astype("datetime64[ns]")
            arrays = tuple(np.ascontiguousarray(value) for value in archive[f"arrays_{index}"])
            if input_digest(dates, arrays) != row["input_sha256"]:
                raise ValueError(f"Input metadata changed for {row['asset']}.")
            prepared.append((row["asset"], row["signal"], row["direction"], dates, arrays))
    return protocol, grids, prepared, protocol["metadata"]


def run_phase(output: Path, phase: str) -> None:
    from leveraged_trader.grid_resolution import evaluate_windows

    protocol, grids, prepared, _ = load(output)
    destination = output / f"{phase}_rows.csv"
    if destination.exists():
        raise FileExistsError(destination)
    if phase == "confirmation":
        frozen_path = output / "frozen_choice.json"
        if digest(frozen_path) != (output / "frozen_choice.sha256").read_text().strip():
            raise ValueError("Frozen confirmation choice changed.")
        frozen = json.loads(frozen_path.read_text())
        if frozen["protocol_sha256"] != digest(output / "protocol.json"):
            raise ValueError("Frozen choice belongs to a different protocol.")
        if frozen["validation_rows_sha256"] != digest(output / "validation_rows.csv"):
            raise ValueError("Validation evidence changed after the choice was frozen.")
        grids = [grid for grid in grids if grid.name in frozen["confirmation_grids"]]
        if not grids or {grid.name for grid in grids} != set(frozen["confirmation_grids"]):
            raise ValueError("Confirmation requires exactly the registered frozen grids.")
    years = protocol[f"{phase}_years"]
    windows = [(year, f"{year}-01-01", min(f"{year + 1}-01-01", "2026-10-09")) for year in years]
    started, rows = time.perf_counter(), []
    for index, values in enumerate(prepared, 1):
        result = evaluate_windows(*values, grids=grids, windows=windows)
        rows.extend(result)
        if index % 5 == 0 or index == len(prepared):
            print(
                f"{phase}: {index}/{len(prepared)} assets; {len(rows)} rows; {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(destination, index=False)
    write_json(output / f"{phase}_run.json", dict(elapsed_seconds=time.perf_counter() - started, records=len(rows)))


def analyze(output: Path, phase: str) -> dict:
    from leveraged_trader.grid_resolution_analysis import summarize_resolution

    protocol, _, _, metadata = load(output)
    frame = pd.read_csv(output / f"{phase}_rows.csv")
    observed_grids = set(frame["grid"])
    metadata = [row for row in metadata if row["name"] in observed_grids]
    result = summarize_resolution(
        frame.to_dict("records"),
        metadata,
        baseline="r1_t1",
        comparison_count=protocol["bootstrap"]["comparison_count"],
        bootstrap_samples=protocol["bootstrap"]["resamples"],
    )
    write_json(output / f"{phase}_summary.json", result)
    return result


def freeze(output: Path) -> None:
    if (output / "frozen_choice.json").exists():
        raise FileExistsError("Choice already frozen.")
    if any((output / name).exists() for name in ("confirmation_rows.csv", "confirmation_run.json")):
        raise FileExistsError("Cannot freeze a choice after confirmation has run.")
    result = analyze(output, "validation")
    _, _, _, metadata = load(output)
    counts = {row["name"]: row["candidate_count"] for row in metadata}
    frame = pd.read_csv(output / "validation_rows.csv")
    utility = frame.groupby(["grid", "signal"])["utility"].mean().groupby("grid").mean()
    # Candidate status is filled from the evidence module; exploratory tie-break is predeclared.
    candidate = result.get("recommendation", {}).get("grid")
    exploratory = min(utility.index, key=lambda name: (-utility[name], counts[name], name))
    leader = candidate or exploratory
    joint = [row["name"] for row in metadata if row["family"] == "joint"]
    confirm = {"r1_t1", "current49", leader, "r40_t98"}
    if leader in joint:
        i = joint.index(leader)
        confirm.update(joint[max(0, i - 1) : i + 2])
    else:
        # Include the joint grids immediately bracketing the leader's candidate count.
        finer = [name for name in joint if counts[name] >= counts[leader]]
        coarser = [name for name in joint if counts[name] <= counts[leader]]
        if finer:
            confirm.add(min(finer, key=lambda name: counts[name]))
        if coarser:
            confirm.add(max(coarser, key=lambda name: counts[name]))
    for name in list(confirm):
        if name + "_shift" in counts:
            confirm.add(name + "_shift")
    write_json(
        output / "frozen_choice.json",
        dict(
            evidence_qualified_candidate=candidate,
            exploratory_leader=exploratory,
            confirmation_grids=sorted(confirm),
            protocol_sha256=digest(output / "protocol.json"),
            validation_rows_sha256=digest(output / "validation_rows.csv"),
            reason="Frozen before later-period results; exploratory leader alone cannot authorize promotion.",
        ),
    )
    (output / "frozen_choice.sha256").write_text(digest(output / "frozen_choice.json") + "\n")
    print(
        f"Frozen evidence candidate={candidate}; exploratory={exploratory}; confirmation={sorted(confirm)}", flush=True
    )


def benchmark(output: Path) -> None:
    destination = output / "benchmark_rows.csv"
    if destination.exists():
        raise FileExistsError(destination)
    _, grids, prepared, _ = load(output)
    eligible = []
    for asset, signal, direction, dates, arrays in prepared:
        stop = int(np.searchsorted(dates, np.datetime64("2023-01-01")))
        if stop >= 1260:
            eligible.append((asset, signal, direction, dates[:stop], tuple(v[:stop] for v in arrays)))
    eligible.sort(key=lambda item: (len(item[3]), item[0]))
    selected = [eligible[i] for i in np.linspace(0, len(eligible) - 1, 5, dtype=int)]
    rows, rng = [], np.random.default_rng(20261012)
    for asset, _, direction, _, arrays in selected:
        rule = 0 if direction == "long" else 1
        fine = grids[0].pairs(direction)
        select_strategy(*arrays, fine[:, 0], fine[:, 1], INITIAL_CAPITAL, BASE_COST, rule)
        for repeat in range(3):
            for index in rng.permutation(len(grids)):
                grid = grids[index]
                pairs = grid.pairs(direction)
                started = time.perf_counter()
                select_strategy(*arrays, pairs[:, 0], pairs[:, 1], INITIAL_CAPITAL, BASE_COST, rule)
                elapsed = time.perf_counter() - started
                rows.append(
                    dict(
                        asset=asset,
                        direction=direction,
                        sessions=len(arrays[0]),
                        grid=grid.name,
                        candidate_count=len(pairs),
                        repeat=repeat,
                        seconds=elapsed,
                    )
                )
        print(f"Benchmarked {asset}, {len(arrays[0])} sessions", flush=True)
    pd.DataFrame(rows).to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "strategy_state.sqlite")
    parser.add_argument("--study-dir", type=Path, default=ROOT / "research/grid-resolution-2026-10-12")
    parser.add_argument(
        "--phase", required=True, choices=("prepare", "validation", "freeze", "confirmation", "benchmark", "summarize")
    )
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.db, args.study_dir)
    elif args.phase in {"validation", "confirmation"}:
        run_phase(args.study_dir, args.phase)
    elif args.phase == "freeze":
        freeze(args.study_dir)
    elif args.phase == "benchmark":
        benchmark(args.study_dir)
    else:
        for phase in ("validation", "confirmation"):
            analyze(args.study_dir, phase)


if __name__ == "__main__":
    main()
