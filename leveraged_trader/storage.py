from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .backtest import initial_strategy_state
from .config import RISK_FREE_SYMBOL, BacktestConfig
from .indicators import compute_rsi_details, rsi_value_from_average_gain_loss
from .optimized_backtest import (
    ACTION_BUY,
    ACTION_NONE,
    ACTION_SELL,
    run_grid_summary,
    run_single_equity_curve,
)

SUMMARY_ROLLUP_COLUMNS = {
    "first_equity": "REAL",
    "last_equity": "REAL",
    "running_max_equity": "REAL",
    "return_count": "INTEGER",
    "return_sum": "REAL",
    "return_sum_squares": "REAL",
    "excess_return_count": "INTEGER",
    "excess_return_sum": "REAL",
    "excess_return_sum_squares": "REAL",
    "positive_return_count": "INTEGER",
}

ALPACA_MANAGED_POSITION_COLUMNS = {
    "buy_submission_claimed_at": "TEXT",
    "buy_submission_attempt_count": "INTEGER NOT NULL DEFAULT 1",
    "sell_expires_at": "TEXT",
    "sell_renewal_count": "INTEGER NOT NULL DEFAULT 0",
    "sell_renewal_requested_at": "TEXT",
    "sell_filled_qty": "REAL",
    "sell_filled_avg_price": "REAL",
    "sell_filled_at": "TEXT",
    "realized_pl": "REAL",
    "realized_pl_pct": "REAL",
    "sold_qty": "REAL NOT NULL DEFAULT 0",
    "sold_value": "REAL NOT NULL DEFAULT 0",
    "remaining_qty": "REAL",
}

STRATEGY_STATE_SCHEMA_VERSION = 2
_MARKET_DATA_FIELDS = ("Open", "High", "Low", "Close", "Volume")
SQLITE_BUSY_TIMEOUT_MS = 60_000


@dataclass
class SummaryRollup:
    first_equity: float | None = None
    last_equity: float | None = None
    running_max_equity: float | None = None
    return_count: int = 0
    return_sum: float = 0.0
    return_sum_squares: float = 0.0
    excess_return_count: int = 0
    excess_return_sum: float = 0.0
    excess_return_sum_squares: float = 0.0
    positive_return_count: int = 0
    max_drawdown: float | None = None

    @property
    def trading_days(self) -> int:
        return self.return_count + 1 if self.first_equity is not None else 0


def save_table_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    with sqlite3.connect(db_path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000) as conn:
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
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
            first_equity REAL,
            last_equity REAL,
            running_max_equity REAL,
            return_count INTEGER,
            return_sum REAL,
            return_sum_squares REAL,
            excess_return_count INTEGER,
            excess_return_sum REAL,
            excess_return_sum_squares REAL,
            positive_return_count INTEGER,
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

        CREATE TABLE IF NOT EXISTS strategy_config (
            asset_symbol TEXT NOT NULL,
            signal_symbol TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            PRIMARY KEY (asset_symbol, signal_symbol)
        );

        CREATE TABLE IF NOT EXISTS strategy_state_generation (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            generation INTEGER NOT NULL
        );

        INSERT OR IGNORE INTO strategy_state_generation (id, generation)
        VALUES (1, 0);

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
            buy_submission_claimed_at TEXT,
            buy_submission_attempt_count INTEGER NOT NULL DEFAULT 1,
            buy_status TEXT NOT NULL,
            filled_qty REAL,
            filled_avg_price REAL,
            filled_at TEXT,
            target_sell_price REAL,
            sell_client_order_id TEXT UNIQUE,
            sell_alpaca_order_id TEXT,
            sell_submitted_at TEXT,
            sell_status TEXT,
            sell_expires_at TEXT,
            sell_renewal_count INTEGER NOT NULL DEFAULT 0,
            sell_renewal_requested_at TEXT,
            sell_filled_qty REAL,
            sell_filled_avg_price REAL,
            sell_filled_at TEXT,
            realized_pl REAL,
            realized_pl_pct REAL,
            sold_qty REAL NOT NULL DEFAULT 0,
            sold_value REAL NOT NULL DEFAULT 0,
            remaining_qty REAL,
            closed_at TEXT,
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS alpaca_managed_sell_fills (
            managed_position_id INTEGER NOT NULL,
            alpaca_order_id TEXT NOT NULL,
            filled_qty REAL NOT NULL,
            filled_value REAL NOT NULL,
            PRIMARY KEY (managed_position_id, alpaca_order_id)
        );
        """
    )
    _ensure_strategy_summary_rollup_columns(conn)
    _ensure_alpaca_managed_position_columns(conn)
    conn.commit()


def _ensure_strategy_summary_rollup_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(strategy_summary)").fetchall()
    }
    for column_name, column_type in SUMMARY_ROLLUP_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE strategy_summary ADD COLUMN {column_name} {column_type}")


def _ensure_alpaca_managed_position_columns(conn: sqlite3.Connection) -> None:
    existing_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(alpaca_managed_positions)").fetchall()
    }
    for column_name, column_type in ALPACA_MANAGED_POSITION_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE alpaca_managed_positions ADD COLUMN {column_name} {column_type}")


def _date_str(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def strategy_config_fingerprint(
    base_cfg: BacktestConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> str:
    """Return a stable identity for every setting that changes a simulation."""
    payload = {
        "schema_version": STRATEGY_STATE_SCHEMA_VERSION,
        "backtest": asdict(base_cfg),
        "buy_rsi_values": sorted({float(value) for value in buy_rsi_values}),
        "profit_target_values": sorted({float(value) for value in profit_target_values}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _strategy_config_pairs(
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> set[tuple[float, float]]:
    return {
        (float(buy_rsi), float(profit_target_multiple))
        for buy_rsi in buy_rsi_values
        for profit_target_multiple in profit_target_values
    }


def strategy_state_matches_config(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    base_cfg: BacktestConfig,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
) -> bool:
    """Whether persisted state exactly matches the requested simulation setup."""
    expected_pairs = _strategy_config_pairs(buy_rsi_values, profit_target_values)
    if not expected_pairs:
        return False

    fingerprint = strategy_config_fingerprint(base_cfg, buy_rsi_values, profit_target_values)
    if not strategy_config_matches_fingerprint(conn, asset_symbol, signal_symbol, fingerprint):
        return False

    rows = conn.execute(
        """
        SELECT buy_rsi, profit_target_multiple
        FROM strategy_state
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchall()
    actual_pairs = {(float(row[0]), float(row[1])) for row in rows}
    return actual_pairs == expected_pairs


def strategy_config_matches_fingerprint(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    fingerprint: str,
) -> bool:
    config_row = conn.execute(
        """
        SELECT fingerprint
        FROM strategy_config
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchone()
    return config_row is not None and str(config_row[0]) == fingerprint


def save_strategy_config(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    fingerprint: str,
) -> None:
    conn.execute(
        """
        INSERT INTO strategy_config (asset_symbol, signal_symbol, fingerprint)
        VALUES (?, ?, ?)
        ON CONFLICT(asset_symbol, signal_symbol) DO UPDATE SET fingerprint = excluded.fingerprint
        """,
        (asset_symbol, signal_symbol, fingerprint),
    )


def _market_values_differ(existing: object, incoming: object) -> bool:
    if existing is None or pd.isna(existing):
        return incoming is not None and not pd.isna(incoming)
    if incoming is None or pd.isna(incoming):
        return True
    existing_value = float(existing)
    incoming_value = float(incoming)
    if existing_value == incoming_value:
        return False
    if not math.isfinite(existing_value) or not math.isfinite(incoming_value):
        return True
    return abs(existing_value - incoming_value) > 1e-12 + 1e-12 * abs(incoming_value)


def _revised_market_symbols(
    conn: sqlite3.Connection,
    data: pd.DataFrame,
    symbols: list[str],
) -> set[str]:
    """Detect corrections to an already persisted session before overwriting it."""
    if data.empty:
        return set()

    revised_symbols: set[str] = set()
    for date, row in data.iterrows():
        date_str = _date_str(date)
        for symbol in symbols:
            existing = conn.execute(
                """
                SELECT open, high, low, close, volume
                FROM market_data
                WHERE symbol = ? AND date = ?
                """,
                (symbol, date_str),
            ).fetchone()
            if existing is None:
                continue
            incoming = [row.get(f"{symbol}_{field}") for field in _MARKET_DATA_FIELDS]
            if any(_market_values_differ(old, new) for old, new in zip(existing, incoming, strict=True)):
                revised_symbols.add(symbol)
    return revised_symbols


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


def _market_history_values(
    data: pd.DataFrame,
    symbol: str,
) -> dict[str, tuple[object, ...]]:
    columns = [f"{symbol}_{field}" for field in _MARKET_DATA_FIELDS]
    values_by_date: dict[str, tuple[object, ...]] = {}
    for date, values in zip(
        data.index,
        data.loc[:, columns].itertuples(index=False, name=None),
        strict=True,
    ):
        values_by_date[_date_str(date)] = values
    return values_by_date


def _synchronize_market_data_history(
    conn: sqlite3.Connection,
    data: pd.DataFrame,
    symbol: str,
) -> bool:
    """Persist a complete single-symbol history and remove vanished sessions.

    This helper is deliberately used only with provider histories that are
    known to be complete.  Tail updates are not authoritative: treating those
    as complete would erase otherwise valid older bars.
    """
    if data.empty:
        return False

    expected_columns = {f"{symbol}_{field}" for field in _MARKET_DATA_FIELDS}
    missing_columns = expected_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Authoritative {symbol} history is missing required columns: {missing}.")

    existing_by_date = {
        str(row[0]): row[1:]
        for row in conn.execute(
            """
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE symbol = ?
            """,
            (symbol,),
        ).fetchall()
    }
    incoming_by_date = _market_history_values(data, symbol)
    existing_dates = set(existing_by_date)
    incoming_dates = set(incoming_by_date)
    changed_dates = {
        date
        for date in existing_dates.intersection(incoming_dates)
        if any(
            _market_values_differ(existing, incoming)
            for existing, incoming in zip(
                existing_by_date[date],
                incoming_by_date[date],
                strict=True,
            )
        )
    }
    new_dates = incoming_dates.difference(existing_dates)
    removed_dates = existing_dates.difference(incoming_dates)
    # A new tail date is a normal incremental update.  A newly discovered date
    # at or before the prior tail changes historical inputs and needs replay.
    prior_tail = max(existing_dates) if existing_dates else None
    historical_additions = {
        date for date in new_dates if prior_tail is not None and date <= prior_tail
    }
    if removed_dates:
        conn.executemany(
            "DELETE FROM market_data WHERE symbol = ? AND date = ?",
            [(symbol, date) for date in sorted(removed_dates)],
        )

    dates_to_write = new_dates.union(changed_dates)
    if dates_to_write:
        conn.executemany(
            """
            INSERT INTO market_data
            (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
            """,
            [
                (symbol, date, *values)
                for date, values in incoming_by_date.items()
                if date in dates_to_write
            ],
        )

    return bool(changed_dates or removed_dates or historical_additions)


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
        rsi = rsi_value_from_average_gain_loss(avg_gain, avg_loss)
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
    return load_rsi_series_for_dates(conn, signal_symbol, rsi_period, close.index)


def load_strategy_state(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> dict | None:
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
    return _strategy_state_from_row(row)


def _strategy_state_from_row(row: tuple) -> dict:
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


def _load_strategy_states_for_asset(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> dict[tuple[float, float], dict]:
    rows = conn.execute(
        """
        SELECT buy_rsi, profit_target_multiple, start_date, last_date, cash,
               shares, in_position, entry_price, pending_action, prev_equity,
               trades_executed
        FROM strategy_state
        WHERE asset_symbol = ?
          AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchall()
    return {
        (float(row[0]), float(row[1])): _strategy_state_from_row(row[2:])
        for row in rows
    }


_STRATEGY_STATE_UPSERT_SQL = """
INSERT OR REPLACE INTO strategy_state
(asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, start_date,
 last_date, cash, shares, in_position, entry_price, pending_action,
 prev_equity, trades_executed)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _strategy_state_row(
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
) -> tuple:
    return (
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
    )


def save_strategy_state(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
) -> None:
    conn.execute(
        _STRATEGY_STATE_UPSERT_SQL,
        _strategy_state_row(
            asset_symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
            state,
        ),
    )


def save_strategy_states(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if rows:
        conn.executemany(_STRATEGY_STATE_UPSERT_SQL, rows)


def save_alpaca_managed_buy_order(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    buy_signal_date: str,
    buy_client_order_id: str,
    buy_alpaca_order_id: str | None,
    buy_submitted_at: str | None,
    buy_status: str,
    notes: str | None = None,
) -> int:
    """Persist an observed broker state for a managed buy intent.

    New submissions must use ``claim_alpaca_managed_buy_intent`` first.  This
    helper intentionally remains an upsert because it is used after Alpaca has
    authoritatively identified an existing order by client order ID.
    """
    conn.execute(
        """
        INSERT INTO alpaca_managed_positions
        (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
         buy_client_order_id, buy_alpaca_order_id, buy_submitted_at,
         buy_submission_claimed_at, buy_status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
        ON CONFLICT(buy_client_order_id) DO UPDATE SET
            buy_alpaca_order_id = COALESCE(excluded.buy_alpaca_order_id, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(excluded.buy_submitted_at, buy_submitted_at),
            buy_status = excluded.buy_status,
            closed_at = CASE
                WHEN excluded.buy_alpaca_order_id IS NOT NULL
                     AND LOWER(excluded.buy_status) NOT IN
                         ('canceled', 'done_for_day', 'expired', 'rejected', 'stopped', 'suspended')
                    THEN NULL
                ELSE closed_at
            END,
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
    row = conn.execute(
        "SELECT id FROM alpaca_managed_positions WHERE buy_client_order_id = ?",
        (buy_client_order_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("Managed Alpaca buy position was not persisted.")
    return int(row[0])


def claim_alpaca_managed_buy_intent(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    buy_signal_date: str,
    buy_client_order_id: str,
    allow_retry_after_not_found: bool = False,
) -> tuple[int, bool]:
    """Atomically claim a managed-buy client order ID.

    Returns ``(position_id, True)`` only for the process that inserted the
    intent or reactivated a verified missing submission.  A retry is allowed
    only when the caller has already confirmed that Alpaca cannot find the
    deterministic client order ID.  Competing processes receive the existing
    ID without changing its broker state.
    """
    cursor = conn.execute(
        """
        INSERT INTO alpaca_managed_positions
        (symbol, signal_symbol, buy_rsi, profit_target_multiple, buy_signal_date,
         buy_client_order_id, buy_submission_claimed_at, buy_status, notes)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 'submission_pending', ?)
        ON CONFLICT(buy_client_order_id) DO NOTHING
        """,
        (
            symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
            buy_signal_date,
            buy_client_order_id,
            "managed buy submission intent persisted before broker request",
        ),
    )
    claimed = cursor.rowcount == 1
    if not claimed and allow_retry_after_not_found:
        retry_cursor = conn.execute(
            """
            UPDATE alpaca_managed_positions
            SET buy_status = 'submission_pending',
                buy_submission_claimed_at = CURRENT_TIMESTAMP,
                buy_submission_attempt_count = buy_submission_attempt_count + 1,
                closed_at = NULL,
                notes = 'managed buy submission retry claimed after Alpaca did not find the client order ID',
                updated_at = CURRENT_TIMESTAMP
            WHERE buy_client_order_id = ?
              AND buy_status = 'submission_not_found'
              AND buy_alpaca_order_id IS NULL
              AND filled_qty IS NULL
              AND sell_client_order_id IS NULL
              AND closed_at IS NOT NULL
            """,
            (buy_client_order_id,),
        )
        claimed = retry_cursor.rowcount == 1
    conn.commit()
    row = conn.execute(
        "SELECT id FROM alpaca_managed_positions WHERE buy_client_order_id = ?",
        (buy_client_order_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("Managed Alpaca buy position was not persisted.")
    return int(row[0]), claimed


def fail_alpaca_managed_buy_submission_if_pending(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    notes: str,
) -> bool:
    """Close only the still-unsubmitted intent owned by this submit attempt."""
    cursor = conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET buy_status = 'submission_failed',
            closed_at = CURRENT_TIMESTAMP,
            notes = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
          AND buy_status = 'submission_pending'
          AND buy_alpaca_order_id IS NULL
          AND closed_at IS NULL
        """,
        (notes, position_id),
    )
    conn.commit()
    return cursor.rowcount == 1


def load_alpaca_managed_positions(conn: sqlite3.Connection, *, active_only: bool = False) -> pd.DataFrame:
    where = "WHERE closed_at IS NULL" if active_only else ""
    return pd.read_sql_query(
        f"""
        SELECT id, symbol, signal_symbol, buy_rsi, profit_target_multiple,
               buy_signal_date, buy_client_order_id, buy_alpaca_order_id,
               buy_submitted_at, buy_submission_claimed_at, buy_submission_attempt_count,
               buy_status, filled_qty, filled_avg_price,
               filled_at, target_sell_price, sell_client_order_id,
               sell_alpaca_order_id, sell_submitted_at, sell_status,
               sell_expires_at, sell_renewal_count, sell_renewal_requested_at,
               sell_filled_qty, sell_filled_avg_price, sell_filled_at,
               realized_pl, realized_pl_pct, sold_qty, sold_value, remaining_qty,
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
    buy_alpaca_order_id: str | None = None,
    buy_submitted_at: str | None = None,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET buy_status = ?,
            buy_alpaca_order_id = COALESCE(?, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(?, buy_submitted_at),
            closed_at = CASE
                WHEN ? IS NOT NULL
                     AND LOWER(?) NOT IN ('canceled', 'done_for_day', 'expired', 'rejected', 'stopped', 'suspended')
                    THEN NULL
                ELSE closed_at
            END,
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            buy_status,
            buy_alpaca_order_id,
            buy_submitted_at,
            buy_alpaca_order_id,
            buy_status,
            notes,
            position_id,
        ),
    )
    conn.commit()


def mark_alpaca_managed_buy_filled(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    buy_status: str,
    filled_qty: float,
    filled_avg_price: float,
    filled_at: str | None,
    target_sell_price: float,
    buy_alpaca_order_id: str | None = None,
    buy_submitted_at: str | None = None,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET buy_status = ?,
            buy_alpaca_order_id = COALESCE(?, buy_alpaca_order_id),
            buy_submitted_at = COALESCE(?, buy_submitted_at),
            closed_at = CASE
                WHEN ? IS NOT NULL
                     AND LOWER(?) NOT IN ('canceled', 'done_for_day', 'expired', 'rejected', 'stopped', 'suspended')
                    THEN NULL
                ELSE closed_at
            END,
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
            buy_alpaca_order_id,
            buy_status,
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
    sell_alpaca_order_id: str | None,
    sell_submitted_at: str | None,
    sell_status: str,
    sell_expires_at: str | None = None,
    increment_renewal_count: bool = False,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET sell_client_order_id = ?,
            sell_alpaca_order_id = ?,
            sell_submitted_at = ?,
            sell_status = ?,
            sell_expires_at = ?,
            sell_renewal_count = sell_renewal_count + ?,
            sell_renewal_requested_at = NULL,
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            sell_client_order_id,
            sell_alpaca_order_id,
            sell_submitted_at,
            sell_status,
            sell_expires_at,
            1 if increment_renewal_count else 0,
            notes,
            position_id,
        ),
    )
    conn.commit()


def update_alpaca_managed_sell_status(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_status: str,
    sell_alpaca_order_id: str | None = None,
    sell_submitted_at: str | None = None,
    sell_expires_at: str | None = None,
    sell_renewal_requested_at: str | None = None,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET sell_status = ?,
            sell_alpaca_order_id = COALESCE(?, sell_alpaca_order_id),
            sell_submitted_at = COALESCE(?, sell_submitted_at),
            sell_expires_at = COALESCE(?, sell_expires_at),
            sell_renewal_requested_at = COALESCE(?, sell_renewal_requested_at),
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            sell_status,
            sell_alpaca_order_id,
            sell_submitted_at,
            sell_expires_at,
            sell_renewal_requested_at,
            notes,
            position_id,
        ),
    )
    conn.commit()


def mark_alpaca_managed_sell_filled(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    sell_status: str,
    sell_filled_qty: float,
    sell_filled_avg_price: float,
    sell_filled_at: str | None,
    sell_alpaca_order_id: str | None = None,
    sell_submitted_at: str | None = None,
    sell_expires_at: str | None = None,
    notes: str | None = None,
) -> float:
    """Record an order's cumulative fills and return the remaining buy quantity."""
    row = conn.execute(
        """
        SELECT filled_qty, filled_avg_price
        FROM alpaca_managed_positions
        WHERE id = ?
        """,
        (position_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Managed Alpaca position {position_id} does not exist.")

    buy_qty = None if row[0] is None else float(row[0])
    buy_avg_price = None if row[1] is None else float(row[1])
    if buy_qty is None or buy_avg_price is None or buy_qty <= 0 or buy_avg_price <= 0:
        raise ValueError(f"Managed Alpaca position {position_id} is missing a valid filled buy quantity or price.")

    order_key = sell_alpaca_order_id or f"legacy-{position_id}"
    observed_qty = float(sell_filled_qty)
    observed_value = observed_qty * float(sell_filled_avg_price)
    prior = conn.execute(
        """
        SELECT filled_qty, filled_value
        FROM alpaca_managed_sell_fills
        WHERE managed_position_id = ? AND alpaca_order_id = ?
        """,
        (position_id, order_key),
    ).fetchone()
    if prior is not None and observed_qty + 1e-8 < float(prior[0]):
        raise ValueError("Alpaca sell filled quantity moved backwards for a managed order.")

    conn.execute(
        """
        INSERT INTO alpaca_managed_sell_fills
        (managed_position_id, alpaca_order_id, filled_qty, filled_value)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(managed_position_id, alpaca_order_id) DO UPDATE SET
            filled_qty = excluded.filled_qty,
            filled_value = excluded.filled_value
        """,
        (position_id, order_key, observed_qty, observed_value),
    )
    totals = conn.execute(
        """
        SELECT COALESCE(SUM(filled_qty), 0), COALESCE(SUM(filled_value), 0)
        FROM alpaca_managed_sell_fills
        WHERE managed_position_id = ?
        """,
        (position_id,),
    ).fetchone()
    sold_qty = float(totals[0])
    sold_value = float(totals[1])
    # Keep an overfill visible to the reconciler instead of silently treating it
    # as a completed managed position.  A negative remaining quantity requires
    # manual review; automatically closing it would conceal a possible short.
    remaining_qty = buy_qty - sold_qty
    matched_qty = min(sold_qty, buy_qty)
    realized_pl = sold_value - matched_qty * buy_avg_price
    realized_pl_pct = realized_pl / (matched_qty * buy_avg_price) * 100.0 if matched_qty > 0 else None
    cumulative_sell_avg_price = sold_value / sold_qty if sold_qty > 0 else None

    conn.execute(
        """
        UPDATE alpaca_managed_positions
        SET sell_status = ?,
            sell_alpaca_order_id = COALESCE(?, sell_alpaca_order_id),
            sell_submitted_at = COALESCE(?, sell_submitted_at),
            sell_expires_at = COALESCE(?, sell_expires_at),
            sell_filled_qty = ?,
            sell_filled_avg_price = ?,
            sell_filled_at = COALESCE(?, sell_filled_at),
            sold_qty = ?,
            sold_value = ?,
            remaining_qty = ?,
            realized_pl = ?,
            realized_pl_pct = ?,
            notes = COALESCE(?, notes),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            sell_status,
            sell_alpaca_order_id,
            sell_submitted_at,
            sell_expires_at,
            sold_qty,
            cumulative_sell_avg_price,
            sell_filled_at,
            sold_qty,
            sold_value,
            remaining_qty,
            realized_pl,
            realized_pl_pct,
            notes,
            position_id,
        ),
    )
    conn.commit()
    return remaining_qty


def close_alpaca_managed_position(
    conn: sqlite3.Connection,
    position_id: int,
    *,
    closed_at: str | None,
    notes: str | None = None,
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


def clear_equity_records(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> None:
    conn.execute(
        """
        DELETE FROM strategy_equity
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND buy_rsi = ?
          AND profit_target_multiple = ?
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    )


def prune_non_best_equity_records(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> None:
    conn.execute(
        """
        DELETE FROM strategy_equity
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND NOT (buy_rsi = ? AND profit_target_multiple = ?)
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    )


def _sample_variance(count: int, total: float, total_squares: float) -> float:
    if count <= 1:
        return np.nan
    variance = (total_squares - (total * total / count)) / (count - 1)
    return max(variance, 0.0)


def _rollup_metrics(rollup: SummaryRollup) -> dict[str, float | None]:
    if rollup.first_equity is None or rollup.last_equity is None or rollup.return_count <= 0:
        return {
            "total_return": None,
            "cagr": None,
            "annualized_vol": None,
            "sharpe": None,
            "kelly_fraction": None,
            "max_drawdown": rollup.max_drawdown,
            "hit_rate": None,
        }

    total_return = rollup.last_equity / rollup.first_equity - 1.0 if rollup.first_equity > 0 else np.nan
    cagr = (
        (rollup.last_equity / rollup.first_equity) ** (252 / rollup.return_count) - 1.0
        if rollup.first_equity > 0 and rollup.last_equity > 0
        else np.nan
    )
    return_variance = _sample_variance(
        rollup.return_count,
        rollup.return_sum,
        rollup.return_sum_squares,
    )
    annualized_vol = np.sqrt(return_variance) * np.sqrt(252) if pd.notna(return_variance) else np.nan
    return_mean = rollup.return_sum / rollup.return_count
    kelly_fraction = return_mean / return_variance if pd.notna(return_variance) and return_variance > 0 else np.nan
    hit_rate = rollup.positive_return_count / rollup.return_count

    excess_variance = _sample_variance(
        rollup.excess_return_count,
        rollup.excess_return_sum,
        rollup.excess_return_sum_squares,
    )
    if pd.notna(excess_variance) and excess_variance > 0:
        excess_mean = rollup.excess_return_sum / rollup.excess_return_count
        sharpe = np.sqrt(252) * excess_mean / np.sqrt(excess_variance)
    else:
        sharpe = np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "kelly_fraction": kelly_fraction,
        "max_drawdown": rollup.max_drawdown,
        "hit_rate": hit_rate,
    }


def _update_summary_rollup(rollup: SummaryRollup, records: list[dict]) -> SummaryRollup:
    for record in records:
        equity = float(record["equity"])
        if rollup.first_equity is None:
            rollup.first_equity = equity
            rollup.last_equity = equity
            rollup.running_max_equity = equity
            rollup.max_drawdown = 0.0
            continue

        daily_return = float(record["daily_return"])
        rollup.return_count += 1
        rollup.return_sum += daily_return
        rollup.return_sum_squares += daily_return * daily_return
        if daily_return > 0:
            rollup.positive_return_count += 1

        risk_free_return = record.get("risk_free_return")
        if not pd.isna(risk_free_return):
            excess_return = daily_return - float(risk_free_return)
            rollup.excess_return_count += 1
            rollup.excess_return_sum += excess_return
            rollup.excess_return_sum_squares += excess_return * excess_return

        rollup.last_equity = equity
        rollup.running_max_equity = max(float(rollup.running_max_equity or equity), equity)
        drawdown = equity / rollup.running_max_equity - 1.0 if rollup.running_max_equity > 0 else np.nan
        if pd.notna(drawdown):
            rollup.max_drawdown = min(float(rollup.max_drawdown or 0.0), float(drawdown))
    return rollup


def _rollup_from_equity_frame(equity_df: pd.DataFrame) -> SummaryRollup:
    if equity_df.empty:
        return SummaryRollup()

    equity_df = equity_df.sort_values("date")
    equity = equity_df["equity"].astype(float)
    returns = equity.pct_change()
    rollup = SummaryRollup(
        first_equity=float(equity.iloc[0]),
        last_equity=float(equity.iloc[-1]),
        running_max_equity=float(equity.cummax().iloc[-1]),
        max_drawdown=float((equity / equity.cummax() - 1.0).min()),
    )
    valid_returns = returns.dropna()
    rollup.return_count = int(len(valid_returns))
    if rollup.return_count > 0:
        rollup.return_sum = float(valid_returns.sum())
        rollup.return_sum_squares = float((valid_returns * valid_returns).sum())
        rollup.positive_return_count = int((valid_returns > 0).sum())

    if "risk_free_return" in equity_df.columns:
        excess = (returns - equity_df["risk_free_return"]).dropna()
        rollup.excess_return_count = int(len(excess))
        if rollup.excess_return_count > 0:
            rollup.excess_return_sum = float(excess.sum())
            rollup.excess_return_sum_squares = float((excess * excess).sum())
    return rollup


def _load_strategy_summary_rollup(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> SummaryRollup | None:
    row = conn.execute(
        """
        SELECT first_equity, last_equity, running_max_equity, return_count,
               return_sum, return_sum_squares, excess_return_count,
               excess_return_sum, excess_return_sum_squares,
               positive_return_count, max_drawdown
        FROM strategy_summary
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND buy_rsi = ?
          AND profit_target_multiple = ?
        """,
        (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
    ).fetchone()
    return _summary_rollup_from_row(row)


def _summary_rollup_from_row(row: tuple | None) -> SummaryRollup | None:
    if row is None or row[0] is None:
        return None
    return SummaryRollup(
        first_equity=float(row[0]),
        last_equity=float(row[1]),
        running_max_equity=float(row[2]),
        return_count=int(row[3] or 0),
        return_sum=float(row[4] or 0.0),
        return_sum_squares=float(row[5] or 0.0),
        excess_return_count=int(row[6] or 0),
        excess_return_sum=float(row[7] or 0.0),
        excess_return_sum_squares=float(row[8] or 0.0),
        positive_return_count=int(row[9] or 0),
        max_drawdown=None if row[10] is None else float(row[10]),
    )


def _load_strategy_summary_rollups_for_asset(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> dict[tuple[float, float], SummaryRollup]:
    rows = conn.execute(
        """
        SELECT buy_rsi, profit_target_multiple, first_equity, last_equity,
               running_max_equity, return_count, return_sum,
               return_sum_squares, excess_return_count, excess_return_sum,
               excess_return_sum_squares, positive_return_count, max_drawdown
        FROM strategy_summary
        WHERE asset_symbol = ?
          AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchall()
    rollups: dict[tuple[float, float], SummaryRollup] = {}
    for row in rows:
        rollup = _summary_rollup_from_row(row[2:])
        if rollup is not None:
            rollups[(float(row[0]), float(row[1]))] = rollup
    return rollups


def _load_legacy_equity_rollup(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> SummaryRollup | None:
    equity_df = pd.read_sql_query(
        """
        SELECT date, equity, risk_free_return
        FROM strategy_equity
        WHERE asset_symbol = ?
          AND signal_symbol = ?
          AND buy_rsi = ?
          AND profit_target_multiple = ?
        ORDER BY date
        """,
        conn,
        params=(asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
        parse_dates=["date"],
    )
    if equity_df.empty:
        return None
    return _rollup_from_equity_frame(equity_df)


def save_strategy_summary(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
    rollup: SummaryRollup,
) -> None:
    conn.execute(
        _STRATEGY_SUMMARY_UPSERT_SQL,
        _strategy_summary_row(
            asset_symbol,
            signal_symbol,
            buy_rsi,
            profit_target_multiple,
            state,
            rollup,
        ),
    )


_STRATEGY_SUMMARY_UPSERT_SQL = """
INSERT OR REPLACE INTO strategy_summary
(asset_symbol, signal_symbol, buy_rsi, profit_target_multiple, start_date,
 end_date, trading_days, trades_executed, total_return, cagr,
 annualized_vol, sharpe, kelly_fraction, max_drawdown, hit_rate,
 first_equity, last_equity, running_max_equity, return_count,
 return_sum, return_sum_squares, excess_return_count, excess_return_sum,
 excess_return_sum_squares, positive_return_count)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _strategy_summary_row(
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    state: dict,
    rollup: SummaryRollup,
) -> tuple:
    metrics = _rollup_metrics(rollup)
    return (
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
        state["start_date"],
        state["last_date"],
        rollup.trading_days,
        int(state["trades_executed"]),
        _nullable_float(metrics["total_return"]),
        _nullable_float(metrics["cagr"]),
        _nullable_float(metrics["annualized_vol"]),
        _nullable_float(metrics["sharpe"]),
        _nullable_float(metrics["kelly_fraction"]),
        _nullable_float(metrics["max_drawdown"]),
        _nullable_float(metrics["hit_rate"]),
        rollup.first_equity,
        rollup.last_equity,
        rollup.running_max_equity,
        rollup.return_count,
        rollup.return_sum,
        rollup.return_sum_squares,
        rollup.excess_return_count,
        rollup.excess_return_sum,
        rollup.excess_return_sum_squares,
        rollup.positive_return_count,
    )


def save_strategy_summaries(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if rows:
        conn.executemany(_STRATEGY_SUMMARY_UPSERT_SQL, rows)


def _nullable_float(value: object) -> float | None:
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
    grouped = equity_df.groupby(["buy_rsi", "profit_target_multiple"], sort=False)
    for (buy_rsi, profit_target_multiple), group in grouped:
        state = state_by_config.get((float(buy_rsi), float(profit_target_multiple)))
        if state is None:
            continue

        group = group.sort_values("date")
        rollup = _rollup_from_equity_frame(group)
        save_strategy_summary(
            conn,
            asset_symbol,
            signal_symbol,
            float(buy_rsi),
            float(profit_target_multiple),
            {
                "start_date": state.start_date,
                "last_date": state.last_date,
                "trades_executed": int(state.trades_executed),
            },
            rollup,
        )
def clear_asset_state(conn: sqlite3.Connection, asset_symbol: str, signal_symbol: str) -> None:
    params = (asset_symbol, signal_symbol)
    conn.execute("DELETE FROM strategy_state WHERE asset_symbol = ? AND signal_symbol = ?", params)
    conn.execute("DELETE FROM strategy_equity WHERE asset_symbol = ? AND signal_symbol = ?", params)
    conn.execute("DELETE FROM strategy_summary WHERE asset_symbol = ? AND signal_symbol = ?", params)
    conn.execute("DELETE FROM strategy_config WHERE asset_symbol = ? AND signal_symbol = ?", params)


def clear_signal_state(conn: sqlite3.Connection, signal_symbol: str) -> None:
    """Invalidate every strategy that depends on a corrected RSI symbol."""
    conn.execute("DELETE FROM strategy_state WHERE signal_symbol = ?", (signal_symbol,))
    conn.execute("DELETE FROM strategy_equity WHERE signal_symbol = ?", (signal_symbol,))
    conn.execute("DELETE FROM strategy_summary WHERE signal_symbol = ?", (signal_symbol,))
    conn.execute("DELETE FROM strategy_config WHERE signal_symbol = ?", (signal_symbol,))


def strategy_state_generation(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT generation FROM strategy_state_generation WHERE id = 1"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO strategy_state_generation (id, generation) VALUES (1, 0)"
        )
        return 0
    return int(row[0])


def _bump_strategy_state_generation(conn: sqlite3.Connection) -> int:
    conn.execute(
        "UPDATE strategy_state_generation SET generation = generation + 1 WHERE id = 1"
    )
    return strategy_state_generation(conn)


def clear_all_strategy_state(conn: sqlite3.Connection) -> None:
    """Invalidate all simulations after a shared benchmark correction."""
    conn.execute("DELETE FROM strategy_state")
    conn.execute("DELETE FROM strategy_equity")
    conn.execute("DELETE FROM strategy_summary")
    conn.execute("DELETE FROM strategy_config")
    _bump_strategy_state_generation(conn)


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
) -> str | None:
    row = conn.execute(
        """
        SELECT MIN(last_date)
        FROM strategy_state
        WHERE asset_symbol = ? AND signal_symbol = ?
        """,
        (asset_symbol, signal_symbol),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def load_saved_market_data(conn: sqlite3.Connection, symbols: list[str]) -> pd.DataFrame:
    symbols = list(dict.fromkeys(symbols))
    placeholders = ",".join("?" for _ in symbols)
    raw = pd.read_sql_query(
        f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM market_data
        WHERE symbol IN ({placeholders})
        ORDER BY date, symbol
        """,
        conn,
        params=symbols,
        parse_dates=["date"],
    )
    if raw.empty:
        return pd.DataFrame()

    frames_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        group = raw[raw["symbol"].eq(symbol)].copy()
        if group.empty:
            return pd.DataFrame()
        group = group.set_index("date")[["open", "high", "low", "close", "volume"]]
        group.columns = [
            f"{symbol}_Open",
            f"{symbol}_High",
            f"{symbol}_Low",
            f"{symbol}_Close",
            f"{symbol}_Volume",
        ]
        frames_by_symbol[symbol] = group

    risk_free = frames_by_symbol.pop(RISK_FREE_SYMBOL, None)
    if frames_by_symbol:
        out = pd.concat(list(frames_by_symbol.values()), axis=1, join="inner").dropna().sort_index()
    elif risk_free is not None:
        return risk_free.sort_index()
    else:
        return pd.DataFrame()

    if risk_free is not None:
        # Match load_strategy_data: benchmark gaps must not change the asset
        # and signal trading calendar during a saved-state rebuild.
        out = out.join(risk_free, how="left")
        out[risk_free.columns] = out[risk_free.columns].ffill()
    return out


def load_saved_close_series(conn: sqlite3.Connection, symbol: str) -> pd.Series:
    rows = pd.read_sql_query(
        """
        SELECT date, close
        FROM market_data
        WHERE symbol = ?
          AND close IS NOT NULL
        ORDER BY date
        """,
        conn,
        params=(symbol,),
        parse_dates=["date"],
    )
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.set_index("date")["close"].astype(float)


def _action_code(action: object) -> int:
    if action == "buy":
        return ACTION_BUY
    if action == "sell":
        return ACTION_SELL
    return ACTION_NONE


def _action_label(action_code: int) -> str:
    if action_code == ACTION_BUY:
        return "buy"
    if action_code == ACTION_SELL:
        return "sell"
    return "none"


def _trading_cost_rate(cfg: BacktestConfig) -> float:
    return (cfg.fee_bps + cfg.slippage_bps) / 10_000.0


def _risk_free_returns_from_data(data: pd.DataFrame) -> np.ndarray:
    risk_free_col = f"{RISK_FREE_SYMBOL}_Close"
    if risk_free_col not in data.columns:
        return np.full(len(data), np.nan, dtype=np.float64)
    annual_yield = data[risk_free_col].to_numpy(dtype=np.float64) / 100.0
    return (1.0 + annual_yield) ** (1 / 252) - 1.0


def _market_arrays(
    data: pd.DataFrame,
    rsi: pd.Series,
    asset_symbol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        data[f"{asset_symbol}_Open"].to_numpy(dtype=np.float64),
        data[f"{asset_symbol}_Close"].to_numpy(dtype=np.float64),
        rsi.reindex(data.index).to_numpy(dtype=np.float64),
        _risk_free_returns_from_data(data),
    )


def _rollup_to_arrays(
    rollup: SummaryRollup,
) -> tuple[float, float, float, int, float, float, int, float, float, int, float]:
    return (
        np.nan if rollup.first_equity is None else float(rollup.first_equity),
        np.nan if rollup.last_equity is None else float(rollup.last_equity),
        np.nan if rollup.running_max_equity is None else float(rollup.running_max_equity),
        int(rollup.return_count),
        float(rollup.return_sum),
        float(rollup.return_sum_squares),
        int(rollup.excess_return_count),
        float(rollup.excess_return_sum),
        float(rollup.excess_return_sum_squares),
        int(rollup.positive_return_count),
        np.nan if rollup.max_drawdown is None else float(rollup.max_drawdown),
    )


def _rollup_from_arrays(
    first_equity: float,
    last_equity: float,
    running_max_equity: float,
    return_count: int,
    return_sum: float,
    return_sum_squares: float,
    excess_return_count: int,
    excess_return_sum: float,
    excess_return_sum_squares: float,
    positive_return_count: int,
    max_drawdown: float,
) -> SummaryRollup:
    return SummaryRollup(
        first_equity=None if np.isnan(first_equity) else float(first_equity),
        last_equity=None if np.isnan(last_equity) else float(last_equity),
        running_max_equity=None if np.isnan(running_max_equity) else float(running_max_equity),
        return_count=int(return_count),
        return_sum=float(return_sum),
        return_sum_squares=float(return_sum_squares),
        excess_return_count=int(excess_return_count),
        excess_return_sum=float(excess_return_sum),
        excess_return_sum_squares=float(excess_return_sum_squares),
        positive_return_count=int(positive_return_count),
        max_drawdown=None if np.isnan(max_drawdown) else float(max_drawdown),
    )


def _equity_records_from_arrays(
    *,
    date_strings: list[str],
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
    equity_values: np.ndarray,
    daily_returns: np.ndarray,
    risk_free_returns: np.ndarray,
    in_position_values: np.ndarray,
    action_executed_values: np.ndarray,
    pending_action_values: np.ndarray,
    trades_executed_values: np.ndarray,
) -> list[dict]:
    return [
        {
            "asset_symbol": asset_symbol,
            "signal_symbol": signal_symbol,
            "buy_rsi": buy_rsi,
            "profit_target_multiple": profit_target_multiple,
            "date": date_str,
            "equity": float(equity_values[idx]),
            "daily_return": float(daily_returns[idx]),
            "risk_free_return": float(risk_free_returns[idx]) if not np.isnan(risk_free_returns[idx]) else np.nan,
            "in_position": int(in_position_values[idx]),
            "action_executed": _action_label(int(action_executed_values[idx])),
            "pending_action": _action_label(int(pending_action_values[idx])),
            "trades_executed": int(trades_executed_values[idx]),
        }
        for idx, date_str in enumerate(date_strings)
    ]


def _best_summary_config(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
) -> tuple[float, float] | None:
    row = conn.execute(
        """
        SELECT buy_rsi, profit_target_multiple
        FROM strategy_summary
        WHERE asset_symbol = ? AND signal_symbol = ?
        ORDER BY sharpe IS NULL,
                 sharpe DESC,
                 total_return IS NULL,
                 total_return DESC,
                 cagr IS NULL,
                 cagr DESC
        LIMIT 1
        """,
        (asset_symbol, signal_symbol),
    ).fetchone()
    if row is None:
        return None
    return float(row[0]), float(row[1])


def _equity_curve_exists(
    conn: sqlite3.Connection,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> bool:
    return bool(
        conn.execute(
            """
            SELECT 1
            FROM strategy_equity
            WHERE asset_symbol = ?
              AND signal_symbol = ?
              AND buy_rsi = ?
              AND profit_target_multiple = ?
            LIMIT 1
            """,
            (asset_symbol, signal_symbol, buy_rsi, profit_target_multiple),
        ).fetchone()
    )


def _replace_best_equity_curve(
    conn: sqlite3.Connection,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi: float,
    profit_target_multiple: float,
) -> None:
    full_data = load_saved_market_data(
        conn,
        list(dict.fromkeys([asset_symbol, signal_symbol, RISK_FREE_SYMBOL])),
    )
    if full_data.empty:
        return

    rsi = ensure_rsi_values(
        conn,
        signal_symbol,
        base_cfg.rsi_period,
        full_data[f"{signal_symbol}_Close"],
        rebuild=False,
    )
    open_prices, close_prices, rsi_values, risk_free_returns = _market_arrays(
        full_data,
        rsi,
        asset_symbol,
    )
    (
        equity_values,
        daily_returns,
        risk_free_returns,
        in_position_values,
        action_executed_values,
        pending_action_values,
        trades_executed_values,
        cash,
        shares,
        in_position,
        entry_price,
        pending_action,
        prev_equity,
        trades_executed,
    ) = run_single_equity_curve(
        open_prices,
        close_prices,
        rsi_values,
        risk_free_returns,
        buy_rsi,
        profit_target_multiple,
        base_cfg.initial_capital,
        _trading_cost_rate(base_cfg),
    )
    date_strings = [_date_str(date) for date in full_data.index]
    state = {
        "start_date": date_strings[0] if date_strings else None,
        "last_date": date_strings[-1] if date_strings else None,
        "cash": float(cash),
        "shares": float(shares),
        "in_position": bool(in_position),
        "entry_price": float(entry_price),
        "pending_action": _action_label(int(pending_action)),
        "prev_equity": float(prev_equity),
        "trades_executed": int(trades_executed),
    }
    records = _equity_records_from_arrays(
        date_strings=date_strings,
        asset_symbol=asset_symbol,
        signal_symbol=signal_symbol,
        buy_rsi=buy_rsi,
        profit_target_multiple=profit_target_multiple,
        equity_values=equity_values,
        daily_returns=daily_returns,
        risk_free_returns=risk_free_returns,
        in_position_values=in_position_values,
        action_executed_values=action_executed_values,
        pending_action_values=pending_action_values,
        trades_executed_values=trades_executed_values,
    )
    clear_equity_records(conn, asset_symbol, signal_symbol, buy_rsi, profit_target_multiple)
    save_equity_records(conn, records)

    rollup = _update_summary_rollup(SummaryRollup(), records)
    save_strategy_summary(
        conn,
        asset_symbol,
        signal_symbol,
        buy_rsi,
        profit_target_multiple,
        state,
        rollup,
    )


def process_asset_grid(
    conn: sqlite3.Connection,
    data: pd.DataFrame,
    base_cfg: BacktestConfig,
    asset_symbol: str,
    signal_symbol: str,
    buy_rsi_values: list[float],
    profit_target_values: list[float],
    rebuild: bool,
    signal_history: pd.DataFrame | None = None,
    strategy_fingerprint: str | None = None,
    authoritative_histories: dict[str, pd.DataFrame] | None = None,
    presynchronized_authoritative_symbols: set[str] | None = None,
    commit: bool = True,
    grid_compute_observer: Callable[[float], None] | None = None,
) -> None:
    symbols = list(dict.fromkeys([asset_symbol, signal_symbol, RISK_FREE_SYMBOL]))
    expected_state_generation = strategy_state_generation(conn)
    strategy_fingerprint = strategy_fingerprint or strategy_config_fingerprint(
        base_cfg,
        buy_rsi_values,
        profit_target_values,
    )
    config_is_current = strategy_config_matches_fingerprint(
        conn,
        asset_symbol,
        signal_symbol,
        strategy_fingerprint,
    )
    presynchronized_symbols = presynchronized_authoritative_symbols or set()
    if authoritative_histories is not None:
        authoritative_symbols = set(authoritative_histories)
        unexpected_symbols = authoritative_symbols.difference(symbols)
        if unexpected_symbols:
            unexpected = ", ".join(sorted(unexpected_symbols))
            raise ValueError(f"Unexpected authoritative market symbols: {unexpected}.")
        unexpected_presynchronized = presynchronized_symbols.difference(authoritative_symbols)
        if unexpected_presynchronized:
            unexpected = ", ".join(sorted(unexpected_presynchronized))
            raise ValueError(f"Presynchronized market symbols lack authoritative histories: {unexpected}.")

        revised_symbols = set()
        for symbol, history in authoritative_histories.items():
            if symbol in presynchronized_symbols:
                continue
            if _synchronize_market_data_history(conn, history, symbol) and not rebuild:
                revised_symbols.add(symbol)

        # The complete histories above are canonical.  Only retain the legacy
        # merged-frame save for symbols without a canonical history.
        fallback_symbols = [symbol for symbol in symbols if symbol not in authoritative_symbols]
        if fallback_symbols:
            save_market_data(conn, data, fallback_symbols)
        signal_history_revisions = set()
    else:
        if presynchronized_symbols:
            unexpected = ", ".join(sorted(presynchronized_symbols))
            raise ValueError(f"Presynchronized market symbols lack authoritative histories: {unexpected}.")
        revised_symbols = set() if rebuild else _revised_market_symbols(conn, data, symbols)
        signal_history_revisions = (
            set()
            if rebuild or signal_history is None
            else _revised_market_symbols(conn, signal_history, [signal_symbol])
        )
        save_market_data(conn, data, symbols)
        if signal_history is not None:
            save_market_data(conn, signal_history, [signal_symbol])

    signal_revised = signal_symbol in revised_symbols or signal_symbol in signal_history_revisions
    corrected_existing_session = bool(revised_symbols or signal_history_revisions)
    if RISK_FREE_SYMBOL in revised_symbols:
        # ^IRX feeds every strategy's Sharpe rollup, so a historical correction
        # cannot be repaired safely by rebuilding only the current asset.
        clear_all_strategy_state(conn)
        expected_state_generation = strategy_state_generation(conn)
    if signal_revised:
        clear_signal_state(conn, signal_symbol)
    if corrected_existing_session:
        # The compact resume state cannot replay a changed historical bar.  Load
        # the corrected full history we just persisted and recompute this asset.
        data = load_saved_market_data(conn, symbols)
        rebuild = True
    elif not rebuild and not config_is_current:
        # Another asset sharing this signal may have invalidated the compact
        # state after this asset's preflight check.  Rebuild from saved history
        # rather than continuing from an incompatible tail state.
        data = load_saved_market_data(conn, symbols)
        rebuild = True

    if data.empty:
        if commit:
            conn.commit()
        return
    if rebuild:
        clear_asset_state(conn, asset_symbol, signal_symbol)

    canonical_signal_close = load_saved_close_series(conn, signal_symbol)
    if canonical_signal_close.empty:
        canonical_signal_close = data[f"{signal_symbol}_Close"].dropna().sort_index()
    rsi = ensure_rsi_values(
        conn,
        signal_symbol,
        base_cfg.rsi_period,
        canonical_signal_close,
        rebuild=rebuild or signal_revised,
    )

    config_pairs = [
        (float(buy_rsi), float(profit_target_multiple))
        for buy_rsi in buy_rsi_values
        for profit_target_multiple in profit_target_values
    ]
    if not config_pairs:
        if commit:
            conn.commit()
        return

    open_prices, close_prices, rsi_values, risk_free_returns = _market_arrays(
        data,
        rsi,
        asset_symbol,
    )
    date_values = pd.to_datetime(data.index).to_numpy(dtype="datetime64[ns]")
    date_strings = [_date_str(date) for date in data.index]
    config_count = len(config_pairs)

    buy_rsi_array = np.empty(config_count, dtype=np.float64)
    profit_target_array = np.empty(config_count, dtype=np.float64)
    start_indices = np.empty(config_count, dtype=np.int64)
    cash_values = np.empty(config_count, dtype=np.float64)
    share_values = np.empty(config_count, dtype=np.float64)
    in_position_values = np.empty(config_count, dtype=np.bool_)
    entry_price_values = np.empty(config_count, dtype=np.float64)
    pending_action_values = np.empty(config_count, dtype=np.int64)
    prev_equity_values = np.empty(config_count, dtype=np.float64)
    trades_executed_values = np.empty(config_count, dtype=np.int64)

    first_equity_values = np.empty(config_count, dtype=np.float64)
    last_equity_values = np.empty(config_count, dtype=np.float64)
    running_max_equity_values = np.empty(config_count, dtype=np.float64)
    return_count_values = np.empty(config_count, dtype=np.int64)
    return_sum_values = np.empty(config_count, dtype=np.float64)
    return_sum_squares_values = np.empty(config_count, dtype=np.float64)
    excess_return_count_values = np.empty(config_count, dtype=np.int64)
    excess_return_sum_values = np.empty(config_count, dtype=np.float64)
    excess_return_sum_squares_values = np.empty(config_count, dtype=np.float64)
    positive_return_count_values = np.empty(config_count, dtype=np.int64)
    max_drawdown_values = np.empty(config_count, dtype=np.float64)
    state_start_dates: list[str | None] = []
    states_by_config = (
        {}
        if rebuild
        else _load_strategy_states_for_asset(conn, asset_symbol, signal_symbol)
    )
    rollups_by_config = (
        {}
        if rebuild
        else _load_strategy_summary_rollups_for_asset(conn, asset_symbol, signal_symbol)
    )

    for config_idx, (buy_rsi, profit_target_multiple) in enumerate(config_pairs):
        config_key = (buy_rsi, profit_target_multiple)
        state = states_by_config.get(config_key)
        if state is None:
            state = initial_strategy_state(base_cfg)
            rollup = SummaryRollup()
        else:
            rollup = rollups_by_config.get(config_key)
            if rollup is None:
                rollup = (
                    _load_legacy_equity_rollup(
                        conn,
                        asset_symbol,
                        signal_symbol,
                        buy_rsi,
                        profit_target_multiple,
                    )
                    or SummaryRollup()
                )

        if state["last_date"] is None:
            start_idx = 0
        else:
            last_date_value = np.datetime64(pd.Timestamp(state["last_date"]).to_datetime64())
            start_idx = int(np.searchsorted(date_values, last_date_value, side="right"))

        (
            first_equity,
            last_equity,
            running_max_equity,
            return_count,
            return_sum,
            return_sum_squares,
            excess_return_count,
            excess_return_sum,
            excess_return_sum_squares,
            positive_return_count,
            max_drawdown,
        ) = _rollup_to_arrays(rollup)

        buy_rsi_array[config_idx] = buy_rsi
        profit_target_array[config_idx] = profit_target_multiple
        start_indices[config_idx] = start_idx
        cash_values[config_idx] = float(state["cash"])
        share_values[config_idx] = float(state["shares"])
        in_position_values[config_idx] = bool(state["in_position"])
        entry_price_values[config_idx] = np.nan if pd.isna(state["entry_price"]) else float(state["entry_price"])
        pending_action_values[config_idx] = _action_code(state["pending_action"])
        prev_equity_values[config_idx] = float(state["prev_equity"])
        trades_executed_values[config_idx] = int(state["trades_executed"])
        first_equity_values[config_idx] = first_equity
        last_equity_values[config_idx] = last_equity
        running_max_equity_values[config_idx] = running_max_equity
        return_count_values[config_idx] = return_count
        return_sum_values[config_idx] = return_sum
        return_sum_squares_values[config_idx] = return_sum_squares
        excess_return_count_values[config_idx] = excess_return_count
        excess_return_sum_values[config_idx] = excess_return_sum
        excess_return_sum_squares_values[config_idx] = excess_return_sum_squares
        positive_return_count_values[config_idx] = positive_return_count
        max_drawdown_values[config_idx] = max_drawdown
        state_start_dates.append(state["start_date"])

    grid_compute_started = time.perf_counter()
    try:
        (
            updated,
            out_cash,
            out_shares,
            out_in_position,
            out_entry_price,
            out_pending_action,
            out_prev_equity,
            out_trades_executed,
            out_first_equity,
            out_last_equity,
            out_running_max_equity,
            out_return_count,
            out_return_sum,
            out_return_sum_squares,
            out_excess_return_count,
            out_excess_return_sum,
            out_excess_return_sum_squares,
            out_positive_return_count,
            out_max_drawdown,
        ) = run_grid_summary(
            open_prices,
            close_prices,
            rsi_values,
            risk_free_returns,
            buy_rsi_array,
            profit_target_array,
            start_indices,
            cash_values,
            share_values,
            in_position_values,
            entry_price_values,
            pending_action_values,
            prev_equity_values,
            trades_executed_values,
            first_equity_values,
            last_equity_values,
            running_max_equity_values,
            return_count_values,
            return_sum_values,
            return_sum_squares_values,
            excess_return_count_values,
            excess_return_sum_values,
            excess_return_sum_squares_values,
            positive_return_count_values,
            max_drawdown_values,
            _trading_cost_rate(base_cfg),
        )
    finally:
        if grid_compute_observer is not None:
            grid_compute_observer(max(0.0, time.perf_counter() - grid_compute_started))

    updated_any_config = bool(updated.any())
    strategy_state_rows: list[tuple] = []
    strategy_summary_rows: list[tuple] = []
    for config_idx, (buy_rsi, profit_target_multiple) in enumerate(config_pairs):
        if not updated[config_idx]:
            continue

        start_idx = int(start_indices[config_idx])
        state = {
            "start_date": state_start_dates[config_idx] or date_strings[start_idx],
            "last_date": date_strings[-1],
            "cash": float(out_cash[config_idx]),
            "shares": float(out_shares[config_idx]),
            "in_position": bool(out_in_position[config_idx]),
            "entry_price": float(out_entry_price[config_idx]),
            "pending_action": _action_label(int(out_pending_action[config_idx])),
            "prev_equity": float(out_prev_equity[config_idx]),
            "trades_executed": int(out_trades_executed[config_idx]),
        }
        rollup = _rollup_from_arrays(
            float(out_first_equity[config_idx]),
            float(out_last_equity[config_idx]),
            float(out_running_max_equity[config_idx]),
            int(out_return_count[config_idx]),
            float(out_return_sum[config_idx]),
            float(out_return_sum_squares[config_idx]),
            int(out_excess_return_count[config_idx]),
            float(out_excess_return_sum[config_idx]),
            float(out_excess_return_sum_squares[config_idx]),
            int(out_positive_return_count[config_idx]),
            float(out_max_drawdown[config_idx]),
        )
        strategy_state_rows.append(
            _strategy_state_row(
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                state,
            )
        )
        strategy_summary_rows.append(
            _strategy_summary_row(
                asset_symbol,
                signal_symbol,
                buy_rsi,
                profit_target_multiple,
                state,
                rollup,
            )
        )

    save_strategy_states(conn, strategy_state_rows)
    save_strategy_summaries(conn, strategy_summary_rows)

    if strategy_summary_count(conn, asset_symbol, signal_symbol) == 0:
        refresh_strategy_summaries_for_asset(conn, asset_symbol, signal_symbol)

    best_config = _best_summary_config(conn, asset_symbol, signal_symbol)
    if best_config is not None:
        best_buy_rsi, best_profit_target_multiple = best_config
        if updated_any_config or not _equity_curve_exists(
            conn,
            asset_symbol,
            signal_symbol,
            best_buy_rsi,
            best_profit_target_multiple,
        ):
            _replace_best_equity_curve(
                conn,
                base_cfg,
                asset_symbol,
                signal_symbol,
                best_buy_rsi,
                best_profit_target_multiple,
            )
        prune_non_best_equity_records(
            conn,
            asset_symbol,
            signal_symbol,
            best_buy_rsi,
            best_profit_target_multiple,
        )

    if strategy_state_generation(conn) != expected_state_generation:
        raise RuntimeError("Strategy state generation changed during an atomic asset update.")
    save_strategy_config(conn, asset_symbol, signal_symbol, strategy_fingerprint)
    if commit:
        conn.commit()
