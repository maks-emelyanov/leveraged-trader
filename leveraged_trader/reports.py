from __future__ import annotations

import sqlite3

import pandas as pd

from .storage import load_strategy_state, refresh_strategy_summaries_for_asset

REALIZED_PNL_COLUMNS = [
    "Closed Positions",
    "Complete Closed Positions",
    "Incomplete Closed Positions",
    "Total Buy Cost",
    "Total Sell Value",
    "Realized P/L",
    "Realized P/L %",
]
REALIZED_PNL_WORKFLOW_COLUMNS = ["Workflow", *REALIZED_PNL_COLUMNS]


def summarize_saved_results(
    conn: sqlite3.Connection,
    workflow_assets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    best_curves = []
    for workflow_asset in workflow_assets.itertuples(index=False):
        asset_symbol = workflow_asset.symbol
        signal_symbol = workflow_asset.rsi_symbol
        summaries = pd.read_sql_query(
            """
            SELECT *
            FROM strategy_summary
            WHERE asset_symbol = ? AND signal_symbol = ?
            """,
            conn,
            params=(asset_symbol, signal_symbol),
        )
        if summaries.empty:
            refresh_strategy_summaries_for_asset(conn, asset_symbol, signal_symbol)
            summaries = pd.read_sql_query(
                """
                SELECT *
                FROM strategy_summary
                WHERE asset_symbol = ? AND signal_symbol = ?
                """,
                conn,
                params=(asset_symbol, signal_symbol),
            )

        if summaries.empty:
            continue

        best_row = summaries.sort_values(
            by=["sharpe", "total_return", "cagr"],
            ascending=False,
            na_position="last",
        ).iloc[0]
        equity_df = pd.read_sql_query(
            """
            SELECT date, equity
            FROM strategy_equity
            WHERE asset_symbol = ?
              AND signal_symbol = ?
              AND buy_rsi = ?
              AND profit_target_multiple = ?
            ORDER BY date
            """,
            conn,
            params=(
                asset_symbol,
                signal_symbol,
                float(best_row["buy_rsi"]),
                float(best_row["profit_target_multiple"]),
            ),
            parse_dates=["date"],
        )
        if equity_df.empty:
            continue

        best_equity = equity_df.set_index("date")["equity"]
        summary_rows.append(
            {
                "Asset": asset_symbol,
                "RSI Symbol": signal_symbol,
                "Start Date": best_row["start_date"],
                "End Date": best_row["end_date"],
                "Trading Days": int(best_row["trading_days"]),
                "Buy RSI": float(best_row["buy_rsi"]),
                "Sell Return Multiple": float(best_row["profit_target_multiple"]),
                "Trades Executed": int(best_row["trades_executed"]),
                "Total Return": best_row["total_return"],
                "CAGR": best_row["cagr"],
                "Annualized Vol": best_row["annualized_vol"],
                "Sharpe": best_row["sharpe"],
                "Kelly Fraction": best_row["kelly_fraction"],
                "Max Drawdown": best_row["max_drawdown"],
                "Hit Rate": best_row["hit_rate"],
            }
        )
        best_curves.append(best_equity.rename(f"{asset_symbol}_RSI_Strategy"))

    optimization_summary = pd.DataFrame(summary_rows)
    if not optimization_summary.empty:
        optimization_summary = optimization_summary.sort_values(
            by=["Sharpe", "Total Return", "CAGR"],
            ascending=False,
            na_position="last",
        )

    curves = pd.concat(best_curves, axis=1, join="outer", sort=False).sort_index() if best_curves else pd.DataFrame()

    return optimization_summary, curves


def build_buy_signal_report(
    conn: sqlite3.Connection,
    optimization_summary: pd.DataFrame,
    rsi_period: int,
) -> pd.DataFrame:
    return build_pending_action_report(
        conn,
        optimization_summary,
        rsi_period,
        pending_action_filter="buy",
        require_multiple_trades=True,
        min_sharpe=1.0,
    )


def build_sell_signal_report(
    conn: sqlite3.Connection,
    optimization_summary: pd.DataFrame,
    rsi_period: int,
) -> pd.DataFrame:
    return build_pending_action_report(
        conn,
        optimization_summary,
        rsi_period,
        pending_action_filter="sell",
        require_multiple_trades=False,
        min_sharpe=None,
    )


def build_alpaca_realized_pnl_summary(
    conn: sqlite3.Connection,
    *,
    include_workflow: bool = False,
) -> pd.DataFrame:
    positions = pd.read_sql_query(
        """
        SELECT workflow, filled_qty, filled_avg_price, sell_filled_qty,
               sell_filled_avg_price, sold_qty, sold_value, remaining_qty, closed_at
        FROM alpaca_managed_positions
        WHERE closed_at IS NOT NULL
          AND sell_status = 'filled'
        """,
        conn,
    )
    if positions.empty:
        row = _realized_pnl_summary_row(0, 0, 0.0, 0.0)
        if include_workflow:
            row = {"Workflow": "All", **row}
        return pd.DataFrame(
            [row],
            columns=REALIZED_PNL_WORKFLOW_COLUMNS if include_workflow else REALIZED_PNL_COLUMNS,
        )

    numeric_columns = [
        "filled_qty",
        "filled_avg_price",
        "sell_filled_qty",
        "sell_filled_avg_price",
        "sold_qty",
        "sold_value",
        "remaining_qty",
    ]
    for column in numeric_columns:
        positions[column] = pd.to_numeric(positions[column], errors="coerce")

    positions["effective_sold_qty"] = positions["sold_qty"].where(
        positions["sold_qty"].gt(0),
        positions["sell_filled_qty"],
    )
    positions["effective_sold_value"] = positions["sold_value"].where(
        positions["sold_value"].gt(0),
        positions["sell_filled_qty"] * positions["sell_filled_avg_price"],
    )
    positions["effective_remaining_qty"] = positions["remaining_qty"].where(
        positions["remaining_qty"].notna(),
        positions["filled_qty"] - positions["effective_sold_qty"],
    )
    complete = positions.dropna(
        subset=["filled_qty", "filled_avg_price", "effective_sold_qty", "effective_sold_value"],
    ).copy()
    complete = complete[
        complete["filled_qty"].gt(0)
        & complete["filled_avg_price"].gt(0)
        & complete["effective_sold_qty"].gt(0)
        & complete["effective_remaining_qty"].abs().le(1e-8)
    ]

    if include_workflow:
        positions["Workflow"] = positions["workflow"].fillna("Unknown").replace("", "Unknown")
        complete["Workflow"] = complete["workflow"].fillna("Unknown").replace("", "Unknown")
        rows = []
        for workflow, workflow_positions in positions.groupby("Workflow", sort=True, dropna=False):
            workflow_complete = complete[complete["Workflow"].eq(workflow)]
            total_buy_cost = float((workflow_complete["filled_qty"] * workflow_complete["filled_avg_price"]).sum())
            total_sell_value = float(workflow_complete["effective_sold_value"].sum())
            rows.append(
                {
                    "Workflow": workflow,
                    **_realized_pnl_summary_row(
                        closed_positions=len(workflow_positions),
                        complete_closed_positions=len(workflow_complete),
                        total_buy_cost=total_buy_cost,
                        total_sell_value=total_sell_value,
                    ),
                }
            )
        return pd.DataFrame(rows, columns=REALIZED_PNL_WORKFLOW_COLUMNS)

    total_buy_cost = float((complete["filled_qty"] * complete["filled_avg_price"]).sum())
    total_sell_value = float(complete["effective_sold_value"].sum())

    return pd.DataFrame(
        [
            _realized_pnl_summary_row(
                closed_positions=len(positions),
                complete_closed_positions=len(complete),
                total_buy_cost=total_buy_cost,
                total_sell_value=total_sell_value,
            )
        ],
        columns=REALIZED_PNL_COLUMNS,
    )


def _realized_pnl_summary_row(
    closed_positions: int,
    complete_closed_positions: int,
    total_buy_cost: float,
    total_sell_value: float,
) -> dict[str, object]:
    realized_pl = total_sell_value - total_buy_cost
    realized_pl_pct = (realized_pl / total_buy_cost * 100.0) if total_buy_cost > 0 else 0.0
    return {
        "Closed Positions": closed_positions,
        "Complete Closed Positions": complete_closed_positions,
        "Incomplete Closed Positions": closed_positions - complete_closed_positions,
        "Total Buy Cost": total_buy_cost,
        "Total Sell Value": total_sell_value,
        "Realized P/L": realized_pl,
        "Realized P/L %": realized_pl_pct,
    }


def build_pending_action_report(
    conn: sqlite3.Connection,
    optimization_summary: pd.DataFrame,
    rsi_period: int,
    pending_action_filter: str,
    require_multiple_trades: bool,
    min_sharpe: float | None,
) -> pd.DataFrame:
    if pending_action_filter not in {"buy", "sell"}:
        raise ValueError(f"Unsupported pending action report: {pending_action_filter}")

    columns = [
        "Asset",
        "RSI Symbol",
        "Date",
        "Start Date",
        "Trading Days",
        "Latest RSI",
        "Buy RSI",
        "Sell Return Multiple",
        "Trades Executed",
        "Sharpe",
        "In Position",
        "Pending Action",
    ]
    if optimization_summary.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, summary_row in optimization_summary.iterrows():
        asset_symbol = str(summary_row["Asset"])
        signal_symbol = str(summary_row["RSI Symbol"])
        buy_rsi = float(summary_row["Buy RSI"])
        profit_target_multiple = float(summary_row["Sell Return Multiple"])
        trades_executed = int(summary_row["Trades Executed"])
        if require_multiple_trades and trades_executed <= 1:
            continue
        try:
            sharpe = float(summary_row["Sharpe"])
        except (TypeError, ValueError):
            sharpe = float("nan")
        if min_sharpe is not None and (pd.isna(sharpe) or sharpe < min_sharpe):
            continue

        state = load_strategy_state(
            conn,
            asset_symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
        )
        if state is None or state["last_date"] is None:
            continue

        if state["pending_action"] != pending_action_filter:
            continue

        latest_rsi = conn.execute(
            """
            SELECT date, rsi
            FROM rsi_values
            WHERE signal_symbol = ?
              AND rsi_period = ?
              AND date = ?
              AND rsi IS NOT NULL
            LIMIT 1
            """,
            (signal_symbol, rsi_period, state["last_date"]),
        ).fetchone()
        if latest_rsi is None:
            continue

        in_position = bool(state["in_position"])
        pending_action = state["pending_action"]
        latest_rsi_value = float(latest_rsi[1])
        start_date = summary_row.get("Start Date", state["start_date"])
        trading_days = summary_row.get("Trading Days", pd.NA)

        should_include = (pending_action_filter == "buy" and not in_position) or (
            pending_action_filter == "sell" and in_position
        )
        if should_include:
            rows.append(
                {
                    "Asset": asset_symbol,
                    "RSI Symbol": signal_symbol,
                    "Date": latest_rsi[0],
                    "Start Date": start_date,
                    "Trading Days": trading_days,
                    "Latest RSI": latest_rsi_value,
                    "Buy RSI": buy_rsi,
                    "Sell Return Multiple": profit_target_multiple,
                    "Trades Executed": trades_executed,
                    "Sharpe": sharpe,
                    "In Position": in_position,
                    "Pending Action": pending_action,
                }
            )

    return pd.DataFrame(rows, columns=columns)
