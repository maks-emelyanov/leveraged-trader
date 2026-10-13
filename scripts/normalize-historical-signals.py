#!/usr/bin/env python3
"""Trim historical trade tails descriptively; preserve original RSI/price units.

Inputs are exported simulated buys and completed round trips. This script never
accesses a database, fetches prices, or changes strategy selection or parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["asset_symbol", "signal_symbol", "date"]


def screen(frame: pd.DataFrame, fields: tuple[str, ...]) -> tuple[pd.DataFrame, list[dict]]:
    result = frame.copy()
    rules = []
    for field in fields:
        result[f"outlier_{field}"] = False
    for side, group in result.groupby("workflow"):
        for field in fields:
            if not np.isfinite(group[field]).all():
                raise ValueError(f"Nonfinite {field} observations require separate data-quality review.")
            q1, q3 = group[field].quantile([0.25, 0.75], interpolation="linear")
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (group[field] < low) | (group[field] > high)
            result.loc[group.index, f"outlier_{field}"] = mask
            rules.append({
                "workflow": side, "field": field, "q1": q1, "q3": q3,
                "lower_fence": low, "upper_fence": high, "flagged": int(mask.sum()),
            })
    result["excluded"] = result[[f"outlier_{field}" for field in fields]].any(axis=1)
    return result, rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buys", type=Path, required=True)
    parser.add_argument("--closed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError("Use an empty output directory to preserve earlier exports.")
    buys = pd.read_csv(args.buys, float_precision="round_trip")
    closed = pd.read_csv(args.closed, float_precision="round_trip")
    if any(frame.duplicated(KEYS).any() for frame in (buys, closed)):
        raise ValueError("Duplicate entry identities in the source exports.")
    buy_keys = pd.MultiIndex.from_frame(buys[KEYS])
    closed_keys = pd.MultiIndex.from_frame(closed[KEYS])
    if not closed_keys.isin(buy_keys).all():
        raise ValueError("A completed trade has no corresponding executed buy.")
    screened_buys, buy_rules = screen(buys, ("observed_signal_rsi",))
    screened_closed, closed_rules = screen(closed, ("observed_signal_rsi", "gross_price_multiple"))
    kept = screened_closed.loc[~screened_closed["excluded"]].copy()
    excluded = screened_closed.loc[screened_closed["excluded"]].copy()
    unresolved = buys.loc[~buy_keys.isin(closed_keys)].copy()
    summary = []
    for side, original in closed.groupby("workflow"):
        clean = kept.loc[kept["workflow"] == side]
        summary.append({
            "workflow": side, "original_completed_trades": len(original), "retained": len(clean),
            "excluded": len(original) - len(clean),
            "ranges": {
                field: {"min": clean[field].min(), "median": clean[field].median(), "max": clean[field].max()}
                for field in ("observed_signal_rsi", "buy_rsi", "profit_target_multiple", "gross_price_multiple")
            },
        })
    metadata = {
        "source_type": "cached simulated selected-strategy trade histories",
        "method": "Within each workflow, compute linear Q1/Q3 on original rows. Exclude an entire row if any "
        "screened field lies strictly outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. No recursive trimming or rescaling.",
        "completed_trade_fields": ["observed_signal_rsi", "gross_price_multiple"],
        "all_buy_fields": ["observed_signal_rsi"],
        "weighting": "Each trade has equal weight; funds/proxies are not equally weighted.",
        "limitations": [
            "Statistical tail observations may be valid large gains; exclusions do not establish data errors.",
            "Completed trades omit unresolved positions and cannot establish whole-strategy returns.",
            "This is descriptive cleaning of previously selected historical strategies, not parameter optimization.",
            "Profit targets are discrete strategy settings; they are not independently screened.",
        ],
        "source_sha256": {
            "buys": hashlib.sha256(args.buys.read_bytes()).hexdigest(),
            "completed": hashlib.sha256(args.closed.read_bytes()).hexdigest(),
        },
        "counts": {
            "buys": len(buys), "completed": len(closed), "retained_completed": len(kept),
            "excluded_completed": len(excluded), "unresolved_buys": len(unresolved),
            "rsi_flagged_buys": int(screened_buys["excluded"].sum()),
        },
        "buy_rules": buy_rules, "completed_trade_rules": closed_rules, "completed_trade_summary": summary,
    }
    (args.out / "source_buys.csv").write_bytes(args.buys.read_bytes())
    (args.out / "source_completed_trades.csv").write_bytes(args.closed.read_bytes())
    screened_buys.to_csv(args.out / "all_buys_with_rsi_flags.csv", index=False)
    kept.to_csv(args.out / "normalized_completed_trades.csv", index=False)
    excluded.to_csv(args.out / "excluded_completed_trades.csv", index=False)
    unresolved.to_csv(args.out / "unresolved_buys.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"counts": metadata["counts"], "summary": summary}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
