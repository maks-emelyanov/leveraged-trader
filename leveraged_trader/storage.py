from __future__ import annotations

import sqlite3
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from .backtest import initial_strategy_state, performance_summary, step_strategy_state
from .config import BacktestConfig, RISK_FREE_SYMBOL, UniverseConfig
from .indicators import compute_rsi_details


def save_table_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def save_workflow_assets(conn: sqlite3.Connection, workflow_assets: pd.DataFrame) -> None:
    workflow_assets.to_sql("workflow_assets", conn, if_exists="replace", index=False)


def init_state_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS strategy_state (
            asset_symbol TEXT NOT NULL,
            signal_symbol TEXT NOT NULL,
            buy_rsi REAL NOT NULL,
            profit_target_multiple REAL NOT NULL,
            start_date TEXT,
            last_date TEXT NOT NULL,
            cash REAL NOT NULL,
            shares REAL NOT NULL,
            in_position INTEGER NOT NULL,
            entry_price REAL,
            pending_action TEXT NOT NULL,
            prev_equity REAL NOT NULL,
            trades_executed INTEGER NOT NULL,
            PRIMARY KEY (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)
        );

        CREATE TABLE IF NOT EXISTS strategy_equity (
            asset_symbol TEXT NOT NULL,
            signal_symbol TEXT NOT NULL,
            buy_rsi REAL NOT NULL,
            profit_target_multiple REAL NOT NULL,
            date TEXT NOT NULL,
            equity REAL NOT NULL,
            daily_return REAL NOT NULL,
            risk_free_return REAL,
            in_position INTEGER NOT NULL,
            action_executed TEXT NOT NULL,
            pending_action TEXT NOT NULL,
            trades_executed INTEGER NOT NULL,
            PRIMARY KEY (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, date)
        );

        CREATE TABLE IF NOT EXISTS strategy_summary (
            asset_symbol TEXT NOT NULL,
            signal_symbol TEXT NOT NULL,
            buy_rsi REAL NOT NULL,
            profit_target_multiple REAL NOT NULL,
            start_date TEXT,
            end_date TEXT,
            trading_days INTEGER NOT NULL,
            trades_executed INTEGER NOT NULL,
            total_return REAL,
            cagr REAL,
            annualized_vol REAL,
            sharpe REAL,
            kelly_fraction REAL,
            max_drawdown REAL,
            hit_rate REAL,
            PRIMARY KEY (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)
        );

        CREATE TABLE IF NOT EXISTS rsi_values (
            signal_symbol TEXT NOT NULL,
            rsi_period INTEGER NOT NULL,
            date TEXT NOT NULL,
            close REAL NOT NULL,
            avg_gain REAL,
            avg_loss REAL,
            rsi REAL,
            PRIMARY KEY (signal_symbol, rsi_period, date)
        );

        CREATE TABLE IF NOT EXISTS market_data (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        );

        CREATE TABLE IF NOT EXISTS alpaca_managed_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            signal_symbol TEXT NOT NULL,
            buy_rsi REAL NOT NULL,
            profit_target_multiple REAL NOT NULL,
            buy_signal_date TEXT NOT NULL,
            buy_client_order_id TEXT NOT NULL UNIQUE,
            buy_alpaca_order_id TEXT,
            buy_submitted_at TEXT,
            buy_status TEXT NOT NULL,
            filled_qty REAL,
            filled_avg_price REAL,
            filled_at TEXT,
            target_sell_price REAL,
            sell_client_order_id TEXT UNIQUE,
            sell_alpaca_order_id TEXT,
            sell_submitted_at TEXT,
            sell_status TEXT,
            closed_at TEXT,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()


def _date_str(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def save_market_data(conn: sqlite3.Connection, data: pd.DataFrame, symbols: list[str]) -> None:
    rows = []
    for date, row in data.iterrows():
        date_str = _date_str(date)
        for symbol in symbols:
            rows.append(
                (
                    symbol,
                    date_str,
                    row.get(f"{symbol}_Open"),
                    row.get(f"{symbol}_High"),
                    row.get(f"{symbol}_Low"),
                    row.get(f"{symbol}_Close"),
                    row.get(f"{symbol}_Volume"),
                )
            )

    conn.executemany(
        """
        INSERT OR REPLACE INTO market_data
        (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def save_rsi_values(
    conn: sqlite3.Connection,
    signal_symbol: str,
    rsi_period: int,
    details: pd.DataFrame,
) -> None:
    rows = [
        (
            signal_symbol,
            rsi_period,
            _date_str(date),
            float(row["close"]),
            None if pd.isna(row["avg_gain"]) else float(row["avg_gain"]),
            None if pd.isna(row["avg_loss"]) else float(row["avg_loss"]),
            None if pd.isna(row["rsi"]) else float(row["rsi"]),
        )
        for date, row in details.iterrows()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO rsi_values
        (signal_symbol, rsi_period, date, close, avg_gain, avg_loss, rsi)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def load_rsi_series_for_dates(
    conn: sqlite3.Connection,
    signal_symbol: str,
    rsi_period: int,
    dates: pd.Index,
) -> pd.Series:
    if len(dates) == 0:
        return pd.Series(dtype=float)

    placeholders = ",".join("?" for _ in dates)
    params = [signal_symbol, rsi_period, *[_date_str(date) for date in dates]]
    df = pd.read_sql_query(
        f"""
        SELECT date, rsi
        FROM rsi_values
        WHERE signal_symbol = ?
          AND rsi_period = ?
          AND date IN ({placeholders})
        """,
        conn,
        params=params,
        parse_dates=["date"],
    )
    if df.empty:
        return pd.Series(index=dates, dtype=float)
    out = df.set_index("date")["rsi"].sort_index()
    out.index = pd.to_datetime(out.index)
    return out.reindex(pd.to_datetime(dates))


def ensure_rsi_values(
    conn: sqlite3.Connection,
    signal_symbol: str,
    rsi_period: int,
    close: pd.Series,
    rebuild: bool,
) -> pd.Series:
    close = close.dropna().sort_index()
    if close.empty:
        return pd.Series(dtype=float)

    if rebuild:
        conn.execute(
            "DELETE FROM rsi_values WHERE signal_symbol = ? AND rsi_period = ?",
            (signal_symbol, rsi_period),
        )
        details = compute_rsi_details(close, rsi_period)
        save_rsi_values(conn, signal_symbol, rsi_period, details)
        return details["rsi"]

    last = pd.read_sql_query(
        """
        SELECT date, close, avg_gain, avg_loss
        FROM rsi_values
        WHERE signal_symbol = ? AND rsi_period = ?
        ORDER BY date DESC
        LIMIT 1
        """,
        conn,
        params=(signal_symbol, rsi_period),
    )
    if last.empty:
        details = compute_rsi_details(close, rsi_period)
        save_rsi_values(conn, signal_symbol, rsi_period, details)
        return details["rsi"]

    last_date = pd.Timestamp(last.loc[0, "date"])
    new_close = close[close.index > last_date]
    if new_close.empty:
        return load_rsi_series_for_dates(conn, signal_symbol, rsi_period, close.index)

    prev_close = float(last.loc[0, "close"])
    avg_gain = last.loc[0, "avg_gain"]
    avg_loss = last.loc[0, "avg_loss"]
    if pd.isna(avg_gain) or pd.isna(avg_loss):
        details = compute_rsi_details(close, rsi_period)
        save_rsi_values(conn, signal_symbol, rsi_period, details)
        return details["rsi"]

    avg_gain = float(avg_gain)
    avg_loss = float(avg_loss)
    rows = []
    alpha = 1 / rsi_period
    for date, current_close in new_close.items():
        delta = float(current_close) - prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (1 - alpha) * avg_gain + alpha * gain
        avg_loss = (1 - alpha) * avg_loss + alpha * loss
        if avg_loss == 0:
            rsi = 100.0
        elif avg_gain == 0:
            rsi = 0.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rows.append(
            (
                signal_symbol,
                rsi_period,
                _date_str(date),
                float(current_close),
                avg_gain,
                avg_loss,
                rsi,
            )
        )
        prev_close = float(current_close)

    conn.executemany(
        """
        INSERT OR REPLACE INTO rsi_values
        (signal_symbol, rsi_period, date, close, avg_gain, avg_loss, rsi)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return load_rsi_series_for_dates(conn, signal_symbol, rsi_period, close.index)


def load_strategy_state(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> Optional[dict]:
    row = conn.execute(
        """
        SELECT start_date, last_date, cash, shares, in_position, entry_price,
               pending_action, prev_equity, trades_executed
        FROM strategy_state
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND buy_rsi = ?
          AND profit_target_multiple = ?
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    ).fetchone()
    if row is None:
        return None
    return {
        "start_date": row[0],
        "last_date": row[1],
        "cash": row[2],
        "shares": row[3],
        "in_position": bool(row[4]),
        "entry_price": row[5],
        "pending_action": row[6],
        "prev_equity": row[7],
        "trades_executed": row[8],
    }


def save_strategy_state(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO strategy_state
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, start_date,
         last_date, cash, shares, in_position, entry_price, pending_action,
         prev_equity, trades_executed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            asset_symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
            state["start_date"],
            state["last_date"],
            state["cash"],
            state["shares"],
            int(state["in_position"]),
            None if pd.isna(state["entry_price"]) else state["entry_price"],
            state["pending_action"],
            state["prev_equity"],
            state["trades_executed"],
        ),
    )


def save_alpaca_managed_buy_order(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    buy_signal_date: str,
    buy_client_order_id: str,
    buy_alpaca_order_id: Optional[str],
    buy_submitted_at: Optional[str],
    buy_status: str,
    notes: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO alpaca_managed_positions
        (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
         buy_client_order_id, buy_alpaca_order_id, buy_submitted_at, buy_status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(buy_client_order_id) DO UPDATE SET
            buy_alpaca_order_id = COALESCE(excluded.buy_alpaca_order_id, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(excluded.buy_submitted_at, buy_submitted_at),
            buy_status = excluded.buy_status,
            notes = COALESCE(excluded.notes, notes),
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
            buy_signal_date,
            buy_client_order_id,
            buy_alpaca_order_id,
            buy_submitted_at,
            buy_status,
            notes,
        ),
    )
    conn.commit()


def load_alpaca_managed_positions(conn: sqlite3.Connection, *, active_only: bool = False) -> pd.DataFrame:
    where = "WHERE closed_at IS NULL" if active_only else ""
    return pd.read_sql_query(
        f"""
        SELECT id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
               buy_signal_date, buy_client_order_id, buy_alpaca_order_id,
               buy_submitted_at, buy_status, filled_qty, filled_avg_price,
               filled_at, target_sell_price, sell_client_order_id,
               sell_alpaca_order_id, sell_submitted_at, sell_status,
               closed_at, notes, created_at, updated_at
        FROM alpaca_managed_positions
        {where}
        ORDER BY id
        """,
        conn,
    )


def active_alpaca_managed_symbols(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT UPPER(symbol)
        FROM alpaca_managed_positions
        WHERE closed_at IS NULL
        """
    ).fetchall()
    return {str(row[0]) for row in rows}


def update_alpaca_managed_buy_status(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    buy_status: str,
    buy_alpaca_order_id: Optional[str] = None,
    buy_submitted_at: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET buy_status = ?,
            buy_alpaca_order_id = COALESCE(?, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(?, buy_submitted_at),
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (buy_status, buy_alpaca_order_id, buy_submitted_at, notes, position_id),
    )
    conn.commit()


def mark_alpaca_managed_buy_filled(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    buy_status: str,
    filled_qty: float,
    filled_avg_price: float,
    filled_at: Optional[str],
    target_sell_price: float,
    buy_alpaca_order_id: Optional[str] = None,
    buy_submitted_at: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET buy_status = ?,
            buy_alpaca_order_id = COALESCE(?, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(?, buy_submitted_at),
            filled_qty = ?,
            filled_avg_price = ?,
            filled_at = COALESCE(?, filled_at),
            target_sell_price = ?,
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            buy_status,
            buy_alpaca_order_id,
            buy_submitted_at,
            filled_qty,
            filled_avg_price,
            filled_at,
            target_sell_price,
            notes,
            position_id,
        ),
    )
    conn.commit()


def record_alpaca_managed_sell_order(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_client_order_id: str,
    sell_alpaca_order_id: Optional[str],
    sell_submitted_at: Optional[str],
    sell_status: str,
    notes: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET sell_client_order_id = ?,
            sell_alpaca_order_id = ?,
            sell_submitted_at = ?,
            sell_status = ?,
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (sell_client_order_id, sell_alpaca_order_id, sell_submitted_at, sell_status, notes, position_id),
    )
    conn.commit()


def update_alpaca_managed_sell_status(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_status: str,
    sell_alpaca_order_id: Optional[str] = None,
    sell_submitted_at: Optional[str] = None,
    notes: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET sell_status = ?,
            sell_alpaca_order_id = COALESCE(?, sell_alpaca_order_id),
            sell_submitted_at = COALESCE(?, sell_submitted_at),
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (sell_status, sell_alpaca_order_id, sell_submitted_at, notes, position_id),
    )
    conn.commit()


def close_alpaca_managed_position(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    closed_at: Optional[str],
    notes: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET closed_at = COALESCE(?, CURRENT_TIMESTAMP),
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (closed_at, notes, position_id),
    )
    conn.commit()


def save_equity_records(conn: sqlite3.Connection, records: list[dict]) -> None:
    if not records:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO strategy_equity
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, date, equity,
         daily_return, risk_free_return, in_position, action_executed,
         pending_action, trades_executed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                r["asset_symbol"],
                r["signal_symbol"],
                r["buy_rsi"],
                r["profit_target_multiple"],
                r["date"],
                r["equity"],
                r["daily_return"],
                None if pd.isna(r["risk_free_return"]) else r["risk_free_return"],
                r["in_position"],
                r["action_executed"],
                r["pending_action"],
                r["trades_executed"],
            )
            for r in records
        ],
    )


def _nullable_float(value: object) -> Optional[float]:
    return None if pd.isna(value) else float(value)


def strategy_summary_count(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM strategy_summary
            WHERE asset_symbol = ? AND signal_symbol = ?
            """,
            (asset_symbol, signal_symbol),
        ).fetchone()[0]
    )


def refresh_strategy_summaries_for_asset(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> None:
    states = pd.read_sql_query(
        """
        SELECT *
        FROM strategy_state
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        conn,
        params=(asset_symbol, signal_symbol),
    )
    if states.empty:
        return

    equity_df = pd.read_sql_query(
        """
        SELECT buy_rsi, profit_target_multiple, date, equity, risk_free_return
        FROM strategy_equity
        WHERE asset_symbol = ? AND signal_symbol = ?
        ORDER BY buy_rsi, profit_target_multiple, date
        """,
        conn,
        params=(asset_symbol, signal_symbol),
        parse_dates=["date"],
    )
    if equity_df.empty:
        return

    state_by_config = {
        (float(row.buy_rsi), float(row.profit_target_multiple)): row
        for row in states.itertuples(index=False)
    }
    rows = []
    grouped = equity_df.groupby(["buy_rsi", "profit_target_multiple"], sort=False)
    for (buy_rsi, profit_target_multiple), group in grouped:
        state = state_by_config.get((float(buy_rsi), float(profit_target_multiple)))
        if state is None:
            continue

        group = group.sort_values("date")
        equity = group.set_index("date")["equity"]
        risk_free_returns = group.set_index("date")["risk_free_return"]
        summary = performance_summary(equity, risk_free_returns=risk_free_returns)
        if summary.empty:
            continue

        rows.append(
            (
                asset_symbol,
                signal_symbol,
                float(buy_rsi),
                float(profit_target_multiple),
                state.start_date,
                state.last_date,
                int(len(equity)),
                int(state.trades_executed),
                _nullable_float(summary.get("Total Return")),
                _nullable_float(summary.get("CAGR")),
                _nullable_float(summary.get("Annualized Vol")),
                _nullable_float(summary.get("Sharpe")),
                _nullable_float(summary.get("Kelly Fraction")),
                _nullable_float(summary.get("Max Drawdown")),
                _nullable_float(summary.get("Hit Rate")),
            )
        )

    if not rows:
        return

    conn.executemany(
        """
        INSERT OR REPLACE INTO strategy_summary
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, start_date,
         end_date, trading_days, trades_executed, total_return, cagr,
         annualized_vol, sharpe, kelly_fraction, max_drawdown, hit_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def clear_asset_state(conn: sqlite3.Connection, asset_symbol: str, signal_symbol: str) -> None:
    params = (asset_symbol, signal_symbol)
    conn.execute("DELETE FROM strategy_state WHERE asset_symbol = ? AND signal_symbol = ?", params)
    conn.execute("DELETE FROM strategy_equity WHERE asset_symbol = ? AND signal_symbol = ?", params)
    conn.execute("DELETE FROM strategy_summary WHERE asset_symbol = ? AND signal_symbol = ?", params)
    conn.commit()


def expected_state_count(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> int:
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM strategy_state
            WHERE asset_symbol = ? AND signal_symbol = ?
            """,
            (asset_symbol, signal_symbol),
        ).fetchone()[0]
    )


def earliest_state_date(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> Optional[str]:
    row = conn.execute(
        """
        SELECT MIN(last_date)
        FROM strategy_state
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def process_asset_grid(
    conn: sqlite3.Connection,
    data: pd.DataFrame,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rebuild: bool,
) -> None:
    if rebuild:
        clear_asset_state(conn, asset_symbol, signal_symbol)

    symbols = list(dict.fromkeys([asset_symbol, signal_symbol, RISK_FREE_SYMBOL]))
    save_market_data(conn, data, symbols)
    rsi = ensure_rsi_values(
        conn,
        signal_symbol,
        base_cfg.rsi_period,
        data[f"{signal_symbol}_Close"],
        rebuild=rebuild,
    )

    updated_any_config = False
    for buy_rsi in buy_rsi_values:
        for profit_target_multiple in profit_target_values:
            cfg = replace(
                base_cfg,
                buy_rsi=buy_rsi,
                profit_target_multiple=profit_target_multiple,
            )
            state = None if rebuild else load_strategy_state(
                conn,
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
            )
            if state is None:
                state = initial_strategy_state(base_cfg)

            work_data = data
            if state["last_date"] is not None:
                work_data = data[data.index > pd.Timestamp(state["last_date"])]
            if work_data.empty:
                continue

            records = []
            for date, row in work_data.iterrows():
                rsi_value = rsi.get(pd.Timestamp(date), np.nan)
                state, record = step_strategy_state(
                    state,
                    pd.Timestamp(date),
                    row,
                    rsi_value,
                    cfg,
                    asset_symbol,
                    signal_symbol,
                )
                records.append(record)

            save_strategy_state(
                conn,
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                state,
            )
            save_equity_records(conn, records)
            updated_any_config = True

    conn.commit()
    if updated_any_config or strategy_summary_count(conn, asset_symbol, signal_symbol) == 0:
        refresh_strategy_summaries_for_asset(conn, asset_symbol, signal_symbol)
