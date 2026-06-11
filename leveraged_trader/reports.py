from __future__ import annotations

import sqlite3
from typing import Optional

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

    if best_curves:
        curves = pd.concat(best_curves, axis=1, join="outer", sort=False).sort_index()
    else:
        curves = pd.DataFrame()

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


def build_alpaca_realized_pnl_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    positions = pd.read_sql_query(
        """
        SELECT filled_qty, filled_avg_price, sell_filled_qty,
               sell_filled_avg_price, closed_at
        FROM alpaca_managed_positions
        WHERE closed_at IS NOT NULL
          AND sell_status = 'filled'
        """,
        conn,
    )
    if positions.empty:
        return pd.DataFrame(
            [
                {
                    "Closed Positions": 0,
                    "Complete Closed Positions": 0,
                    "Incomplete Closed Positions": 0,
                    "Total Buy Cost": 0.0,
                    "Total Sell Value": 0.0,
                    "Realized P/L": 0.0,
                    "Realized P/L %": 0.0,
                }
            ],
            columns=REALIZED_PNL_COLUMNS,
        )

    numeric_columns = [
        "filled_qty",
        "filled_avg_price",
        "sell_filled_qty",
        "sell_filled_avg_price",
    ]
    for column in numeric_columns:
        positions[column] = pd.to_numeric(positions[column], errors="coerce")

    complete = positions.dropna(subset=numeric_columns).copy()
    complete = complete[
        complete["filled_qty"].gt(0)
        & complete["filled_avg_price"].gt(0)
        & complete["sell_filled_qty"].gt(0)
        & complete["sell_filled_avg_price"].gt(0)
    ]

    total_buy_cost = float((complete["filled_qty"] * complete["filled_avg_price"]).sum())
    total_sell_value = float((complete["sell_filled_qty"] * complete["sell_filled_avg_price"]).sum())
    realized_pl = total_sell_value - total_buy_cost
    realized_pl_pct = (realized_pl / total_buy_cost * 100.0) if total_buy_cost > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "Closed Positions": len(positions),
                "Complete Closed Positions": len(complete),
                "Incomplete Closed Positions": len(positions) - len(complete),
                "Total Buy Cost": total_buy_cost,
                "Total Sell Value": total_sell_value,
                "Realized P/L": realized_pl,
                "Realized P/L %": realized_pl_pct,
            }
        ],
        columns=REALIZED_PNL_COLUMNS,
    )


def build_pending_action_report(
    conn: sqlite3.Connection,
    optimization_summary: pd.DataFrame,
    rsi_period: int,
    pending_action_filter: str,
    require_multiple_trades: bool,
    min_sharpe: Optional[float],
) -> pd.DataFrame:
    if pending_action_filter not in {"buy", "sell"}:
        raise ValueError(f"Unsupported pending action report: {pending_action_filter}")

    columns = [
        "Asset",
        "RSI Symbol",
        "Date",
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
        sharpe = float(summary_row["Sharpe"])
        if require_multiple_trades and trades_executed <= 1:
            continue
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

        should_include = (
            (pending_action_filter == "buy" and not in_position)
            or (pending_action_filter == "sell" and in_position)
        )
        if should_include:
            rows.append(
                {
                    "Asset": asset_symbol,
                    "RSI Symbol": signal_symbol,
                    "Date": latest_rsi[0],
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
