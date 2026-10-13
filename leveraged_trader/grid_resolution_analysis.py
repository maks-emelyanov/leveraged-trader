"""Paired, proxy-balanced evidence for a predeclared grid-resolution study.

No market outcomes are removed from the primary analysis. Symmetric trimming
and leave-one-proxy-out calculations check whether a conclusion depends on
outlying proxy outcomes. Cash policies remain in every paired denominator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

UTILITY_MARGIN = 0.05
RETURN_MARGIN = 0.01
MIN_EXPOSED_GROUPS = 10
MIN_DISCORDANT_GROUPS = 10
MIN_BASELINE_TRADES = 50
FIELDS = ("utility", "total_return", "stress_utility", "stress_total_return")
MARGINS = (UTILITY_MARGIN, RETURN_MARGIN, UTILITY_MARGIN, RETURN_MARGIN)


def _key(row: Mapping) -> tuple:
    return row["asset"], row["signal"], row["direction"], row["year"]


def _trimmed_mean(values: np.ndarray) -> float:
    count = int(len(values) * 0.1)
    ordered = np.sort(values)
    return float(np.mean(ordered[count:len(values) - count]))


def _summary(rows: list[dict], metadata: dict, direction: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["signal"]].append(row)
    result = {
        **metadata,
        "direction": direction,
        "asset_periods": len(rows),
        "signal_groups": len(groups),
        "accepted_asset_periods": sum(bool(row["active"]) for row in rows),
        "exposed_asset_periods": sum(row["exposure_fraction"] > 0 for row in rows),
        "exposed_signal_groups": sum(any(row["exposure_fraction"] > 0 for row in group) for group in groups.values()),
        "completed_trades": int(sum(row["completed_trades"] for row in rows)),
        "training_statuses": dict(Counter(row["training_status"] for row in rows)),
        "activation_statuses": dict(Counter(row.get("activation_status", row["training_status"]) for row in rows)),
    }
    for field in (*FIELDS, "exposure_fraction"):
        result[field] = float(np.mean([np.mean([row[field] for row in group]) for group in groups.values()]))
    for field in ("mean_holding_sessions", "max_holding_sessions", "open_holding_sessions", "sharpe"):
        observed = [
            [row.get(field) for row in group if row["exposure_fraction"] > 0 and row.get(field) is not None]
            for group in groups.values()
        ]
        means = [float(np.mean(values)) for values in observed if values]
        result[f"exposed_{field}"] = float(np.mean(means)) if means else None
    return result


def _classify(comparison: dict, *, stress: bool) -> str:
    if not comparison["breadth_sufficient"]:
        return "insufficient_evidence"
    fields = FIELDS[2:] if stress else FIELDS[:2]
    if any(comparison["metrics"][field]["upper"] < -comparison["metrics"][field]["margin"] for field in fields):
        return "degraded"
    if all(comparison["metrics"][field]["lower"] > -comparison["metrics"][field]["margin"] for field in fields):
        return "noninferior"
    return "inconclusive"


def _compare(
    rows: list[dict], reference: dict[tuple, dict], *, baseline: str, name: str,
    direction: str, samples: int, seed: int, tail: float,
) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["signal"]].append(row)
    deltas, group_details = [], []
    matched_holds = defaultdict(list)
    for signal in sorted(groups):
        group = groups[signal]
        paired = [(row, reference[_key(row)]) for row in group]
        delta = [float(np.mean([row[field] - base[field] for row, base in paired])) for field in FIELDS]
        deltas.append(delta)
        detail = {
            "signal": signal, "asset_periods": len(group),
            "baseline_exposed": any(base["exposure_fraction"] > 0 for _, base in paired),
            "candidate_exposed": any(row["exposure_fraction"] > 0 for row, _ in paired),
            "baseline_completed_trades": int(sum(base["completed_trades"] for _, base in paired)),
            "candidate_completed_trades": int(sum(row["completed_trades"] for row, _ in paired)),
            **{f"delta_{field}": value for field, value in zip(FIELDS, delta, strict=True)},
        }
        # Discordance is based on whole observed policies, not just an average
        # that might hide equal and opposite differences across years.
        detail["discordant"] = any(
            abs(row[field] - base[field]) > 1e-12 for row, base in paired for field in FIELDS[:2]
        )
        group_details.append(detail)
        matched = [
            (row, base) for row, base in paired if row["exposure_fraction"] > 0 and base["exposure_fraction"] > 0
        ]
        for field in ("mean_holding_sessions", "max_holding_sessions"):
            available = [(row.get(field), base.get(field)) for row, base in matched]
            available = [(value, base) for value, base in available if value is not None and base is not None]
            if available:
                matched_holds[field].append(float(np.mean([value - base for value, base in available])))
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    # Reinitialize the same seed for every contrast so identical proxy sets
    # receive identical draws; each draw keeps every asset/year in its proxy.
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws = values[indices].mean(axis=1)
    baseline_exposed = sum(row["baseline_exposed"] for row in group_details)
    candidate_exposed = sum(row["candidate_exposed"] for row in group_details)
    union_exposed = sum(row["baseline_exposed"] or row["candidate_exposed"] for row in group_details)
    discordant = sum(row["discordant"] for row in group_details)
    baseline_trades = sum(row["baseline_completed_trades"] for row in group_details)
    breadth = {
        "baseline_exposed_groups": baseline_exposed >= MIN_EXPOSED_GROUPS,
        "discordant_groups": discordant >= MIN_DISCORDANT_GROUPS,
        "baseline_completed_trades": baseline_trades >= MIN_BASELINE_TRADES,
    }
    result = {
        "grid": name, "baseline": baseline, "direction": direction,
        "signal_groups": len(values), "asset_periods": len(rows),
        "baseline_exposed_signal_groups": baseline_exposed,
        "candidate_exposed_signal_groups": candidate_exposed,
        "union_exposed_signal_groups": union_exposed,
        "discordant_signal_groups": discordant,
        "baseline_completed_trades": baseline_trades,
        "candidate_completed_trades": sum(row["candidate_completed_trades"] for row in group_details),
        "breadth_gates": breadth, "breadth_sufficient": all(breadth.values()),
        "metrics": {}, "proxy_deltas": group_details,
        "matched_exposed_holding_deltas": {
            field: float(np.mean(matched_holds[field])) if matched_holds[field] else None
            for field in ("mean_holding_sessions", "max_holding_sessions")
        },
    }
    for index, (field, margin) in enumerate(zip(FIELDS, MARGINS, strict=True)):
        vector = values[:, index]
        leave_one_out = (vector.sum() - vector) / (len(vector) - 1) if len(vector) > 1 else np.asarray([])
        result["metrics"][field] = {
            "delta": float(vector.mean()), "margin": margin,
            "lower": float(np.quantile(draws[:, index], tail)),
            "upper": float(np.quantile(draws[:, index], 1 - tail)),
            "unadjusted_lower_95": float(np.quantile(draws[:, index], 0.025)),
            "unadjusted_upper_95": float(np.quantile(draws[:, index], 0.975)),
            "trimmed10_delta": _trimmed_mean(vector),
            "leave_one_proxy_out_min": float(leave_one_out.min()) if len(leave_one_out) else None,
            "leave_one_proxy_out_max": float(leave_one_out.max()) if len(leave_one_out) else None,
            "improving_groups": int(np.count_nonzero(vector > 1e-12)),
            "worsening_groups": int(np.count_nonzero(vector < -1e-12)),
        }
    result["base_classification"] = _classify(result, stress=False)
    result["stress_classification"] = _classify(result, stress=True)
    result["robust_noninferiority"] = all(
        metric["trimmed10_delta"] > -metric["margin"]
        and metric["leave_one_proxy_out_min"] is not None
        and metric["leave_one_proxy_out_min"] > -metric["margin"]
        for metric in result["metrics"].values()
    )
    robust_degrading_fields = []
    for field in FIELDS[:2]:
        metrics = [result["metrics"][field], result["metrics"][f"stress_{field}"]]
        if all(
            metric["upper"] < -metric["margin"]
            and metric["trimmed10_delta"] < -metric["margin"]
            and metric["leave_one_proxy_out_max"] is not None
            and metric["leave_one_proxy_out_max"] < -metric["margin"]
            for metric in metrics
        ):
            robust_degrading_fields.append(field)
    result["robust_degrading_fields"] = robust_degrading_fields
    if not result["breadth_sufficient"]:
        classification = "insufficient_evidence"
    elif robust_degrading_fields:
        classification = "degraded"
    elif result["base_classification"] == result["stress_classification"] == "noninferior":
        classification = "noninferior" if result["robust_noninferiority"] else "outlier_sensitive"
    elif "degraded" in (result["base_classification"], result["stress_classification"]):
        classification = "outlier_sensitive_or_cost_sensitive_degradation"
    else:
        classification = "inconclusive"
    result["classification"] = classification
    return result


def summarize_resolution(
    rows: Sequence[dict], grids: Sequence[Mapping], *, baseline: str = "r1_t1",
    comparison_count: int | None = None, bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20261012,
) -> dict:
    """Summarize matched annual tests without selecting on later confirmation.

    ``grids`` supplies at least ``name`` and ``candidate_count`` per grid; other
    metadata is preserved. A larger frozen ``comparison_count`` keeps the same
    adjustment in a later subset. Familywise intervals cover every registered
    candidate contrast, all four outcomes, and all direction summaries.
    """
    if not rows or not grids:
        raise ValueError("Require nonempty rows and grid metadata.")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer.")
    metadata = {grid["name"]: dict(grid) for grid in grids}
    if len(metadata) != len(grids) or baseline not in metadata:
        raise ValueError("Require unique grid metadata including the baseline.")
    if any(type(grid.get("candidate_count")) is not int or grid["candidate_count"] < 1 for grid in metadata.values()):
        raise ValueError("Every grid requires a positive integer candidate_count.")
    observed = {row["grid"] for row in rows}
    if observed != set(metadata):
        raise ValueError("Observed grids must match metadata exactly.")
    actual_comparisons = len(metadata) - 1
    if comparison_count is None:
        comparison_count = actual_comparisons
    if type(comparison_count) is not int or comparison_count < max(1, actual_comparisons):
        raise ValueError("comparison_count must cover all non-baseline grids.")
    keyed = {}
    for row in rows:
        key = (row["grid"], *_key(row))
        if key in keyed:
            raise ValueError("Duplicate grid/asset/signal/direction/year outcome.")
        if any(not np.isfinite(row[field]) for field in (*FIELDS, "exposure_fraction", "completed_trades")):
            raise ValueError("All comparison outcomes must be finite.")
        if not 0 <= row["exposure_fraction"] <= 1 or row["completed_trades"] < 0:
            raise ValueError("Invalid exposure or completed trade count.")
        expected_count = metadata[row["grid"]]["candidate_count"]
        if row.get("candidate_count", expected_count) != expected_count:
            raise ValueError("Row candidate_count differs from grid metadata.")
        keyed[key] = row
    baseline_keys = {key[1:] for key in keyed if key[0] == baseline}
    if any({key[1:] for key in keyed if key[0] == name} != baseline_keys for name in metadata):
        raise ValueError("Every grid must include the same eligible asset periods, including cash.")
    direction_names = sorted({row["direction"] for row in rows})
    direction_count = len(direction_names) + 1
    familywise_contrasts = comparison_count * len(FIELDS) * direction_count
    tail = 0.05 / (2 * familywise_contrasts)
    summaries, directions, comparisons = [], [], []
    for direction in ("all", *direction_names):
        subset = [row for row in rows if direction == "all" or row["direction"] == direction]
        reference = {_key(row): row for row in subset if row["grid"] == baseline}
        for name, grid in metadata.items():
            selected = [row for row in subset if row["grid"] == name]
            (summaries if direction == "all" else directions).append(_summary(selected, grid, direction))
            if name != baseline:
                comparisons.append(_compare(
                    selected, reference, baseline=baseline, name=name, direction=direction,
                    samples=bootstrap_samples, seed=bootstrap_seed, tail=tail,
                ))
    eligible = []
    for name in metadata:
        if name == baseline:
            continue
        relevant = [row for row in comparisons if row["grid"] == name]
        if all(row["classification"] == "noninferior" for row in relevant):
            eligible.append(name)
    eligible.sort(key=lambda name: (metadata[name]["candidate_count"], name))
    recommendation = eligible[0] if eligible else None
    degraded = [row["grid"] for row in comparisons if row["direction"] == "all" and row["classification"] == "degraded"]
    degraded.sort(key=lambda name: (-metadata[name]["candidate_count"], name))
    return {
        "settings": {
            "baseline": baseline, "comparison_count": comparison_count,
            "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
            "bootstrap_cluster": "signal proxy, retaining all assets and years",
            "familywise_contrasts": familywise_contrasts, "tail_probability": tail,
            "expected_resamples_in_each_tail": bootstrap_samples * tail,
            "limited_tail_precision": bootstrap_samples * tail < 10,
            "weighting": "mean within proxy, then equal proxy weights; cash retained",
            "utility_margin": UTILITY_MARGIN, "return_margin": RETURN_MARGIN,
            "minimum_baseline_exposed_groups": MIN_EXPOSED_GROUPS,
            "minimum_discordant_groups": MIN_DISCORDANT_GROUPS,
            "minimum_baseline_completed_trades": MIN_BASELINE_TRADES,
            "exact_equal_sparse_outcomes": "insufficient evidence, not proof of equivalence",
            "outlier_policy": "retain raw outcomes; require symmetric trim10 and leave-one-proxy-out sensitivity",
        },
        "grids": summaries, "directions": directions, "comparisons": comparisons,
        "recommendation": {
            "grid": recommendation, "eligible_grids": eligible,
            "reason": (
                "Fewest candidates among grids meeting all base, stress, breadth, direction and sensitivity gates."
                if recommendation else "No grid meets all noninferiority, breadth, direction and sensitivity gates."
            ),
        },
        "boundary": {
            "identified": bool(degraded), "degraded_grids": degraded,
            "reason": (
                "Listed grids show supported practical degradation robust to costs and proxy exclusions."
                if degraded else "No adequately supported degradation boundary is identified in these outcomes."
            ),
        },
    }
