from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Column, Table
from rich.text import Text

from .benchmark import WorkflowBenchmark

DEFAULT_NON_TERMINAL_WIDTH = 160

STATUS_STYLES = {
    "accepted": "green",
    "batch_budget_exhausted": "yellow",
    "filled": "green",
    "submitted": "green",
    "partially_filled": "yellow",
    "submission_pending": "yellow",
    "submission_unknown": "red",
    "submission_not_found": "yellow",
    "done": "green",
    "duplicate_signal": "yellow",
    "existing": "cyan",
    "managed": "cyan",
    "managed_only": "cyan",
    "open_order": "cyan",
    "renewed": "green",
    "disabled": "yellow",
    "fractional_qty": "yellow",
    "incomplete_fill_metadata": "red",
    "insufficient_notional": "yellow",
    "not_held": "yellow",
    "pending_cancel": "yellow",
    "skipped": "yellow",
    "canceled": "red",
    "error": "red",
    "expired": "red",
    "rejected": "red",
    "stopped": "red",
    "suspended": "red",
    "quantity_mismatch": "red",
}


MESSAGE_ALIASES = {
    "buy filled; target sell price frozen from actual fill": "Buy filled; target sell price frozen from fill",
    "buy order is not filled yet": "Buy order is not filled yet",
    "buy order terminated without a filled position": "Buy order terminated without a filled position",
    "direct sell-signal submissions are disabled; managed reconciliation handles GTC limit sells": (
        "Direct sell-signal submissions are disabled; managed reconciliation handles exits"
    ),
    "filled quantity is fractional; no GTC limit sell submitted": (
        "Fractional filled quantity; no GTC limit sell submitted"
    ),
    "managed GTC sell is no longer active; no automatic resubmission": (
        "Managed GTC sell is inactive; no automatic resubmission"
    ),
    "managed GTC sell order already submitted": "Managed GTC sell order already submitted",
    "managed GTC sell cancellation is pending; replacement not submitted yet": (
        "Managed GTC sell cancellation pending; replacement not submitted yet"
    ),
    (
        "managed GTC sell expires soon; cancellation requested and replacement "
        "will be submitted after Alpaca confirms cancellation"
    ): "Managed GTC sell expires soon; cancellation requested",
    "managed limit sell submission is disabled": "Managed limit sell submission is disabled",
    "managed target sell filled; position closed": "Managed target sell filled; position closed",
    "open sell order already exists for symbol in Alpaca account": (
        "Open sell order already exists in Alpaca account"
    ),
    "prior managed GTC sell expired; submitted replacement at frozen target price": (
        "Expired managed GTC sell renewed at frozen target price"
    ),
    "renewed managed GTC limit sell after Alpaca confirmed cancellation": (
        "Renewed managed GTC sell after cancellation"
    ),
    "renewed managed GTC limit sell before Alpaca aged-order expiration": (
        "Renewed managed GTC sell before expiration"
    ),
    "submitted one-time GTC limit sell at frozen target price": (
        "Submitted GTC limit sell at frozen target price"
    ),
    "submitted managed GTC limit sell at frozen target price": "Submitted GTC limit sell at frozen target price",
    "symbol already has an active managed Alpaca position": "Symbol already has an active managed Alpaca position",
}


Formatter = Callable[[Any], str]


@dataclass(frozen=True)
class TableColumn:
    source: str
    header: str | None = None
    justify: str = "left"
    style: str | None = None
    min_width: int | None = None
    max_width: int | None = None
    ratio: int | None = None
    no_wrap: bool = False
    overflow: str = "fold"
    formatter: Formatter | None = None
    status: bool = False

    @property
    def title(self) -> str:
        return self.header or self.source


class AssetProgress:
    def __init__(self, progress: Progress, task_id: int) -> None:
        self._progress = progress
        self._task_id = task_id

    def start_asset(self, *, asset: str, signal: str, action: str) -> None:
        self._progress.update(
            self._task_id,
            status=f"{asset} using {signal} RSI ({action.lower()})",
        )

    def finish_asset(self) -> None:
        self._progress.update(self._task_id, advance=1)


class WorkflowReporter:
    def __init__(self, *, console: Console | None = None, no_color: bool = False) -> None:
        self.console = console or _default_console(no_color=no_color)

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        if self.console.is_terminal:
            with self.console.status(message):
                yield
        else:
            self.console.print(Text(message, style="dim"))
            yield

    @contextmanager
    def asset_progress(self, total: int) -> Iterator[AssetProgress]:
        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}", table_column=Column(ratio=1, overflow="fold")),
            console=self.console,
            expand=True,
            transient=True,
        )
        with progress:
            task_id = progress.add_task("Processing assets", total=total, status="")
            yield AssetProgress(progress, task_id)

    def section(self, title: str) -> None:
        self.console.print()
        self.console.print(Text(title, style="bold"))

    def settings(
        self,
        *,
        mode: str,
        db_path: str,
        workflow_concurrency: int,
        risk_free_symbol: str,
        buy_rsi_values: list[float],
        profit_target_values: list[float],
    ) -> None:
        self.section("Grid Search Settings")
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", no_wrap=True)
        table.add_column(ratio=1)
        table.add_row("Mode", mode)
        table.add_row("SQLite database", db_path)
        table.add_row("Workflow concurrency", str(workflow_concurrency))
        table.add_row("Start date", "Earliest overlapping history for each leveraged asset and RSI symbol")
        table.add_row("Sharpe benchmark", f"{risk_free_symbol} 13-week U.S. Treasury bill")
        table.add_row("Buy RSI values", _grid_range_label(buy_rsi_values, places=0, step_label="1"))
        table.add_row(
            "Sell return multiples",
            _grid_range_label(profit_target_values, places=1, step_label="0.1"),
        )
        self.console.print(table)

    def dataframe(
        self,
        title: str,
        df: pd.DataFrame,
        columns: list[TableColumn],
        *,
        empty_message: str,
        caption: str | None = None,
    ) -> None:
        self.section(title)
        if df.empty:
            self.console.print(Text(empty_message, style="dim"))
            return
        self.console.print(self._table(df, columns, caption=caption))

    def universe_assets(self, df: pd.DataFrame) -> None:
        title = str(df.attrs.get("universe_title", "Workflow ETF Universe"))
        counts = df.attrs.get("universe_counts", {})
        db_path = df.attrs.get("universe_db_path")
        universe_degraded = bool(df.attrs.get("universe_degraded", False))

        self.section(title)
        if counts or db_path:
            stats = Table.grid(padding=(0, 2))
            stats.add_column(style="bold", no_wrap=True)
            stats.add_column(ratio=1)
            for label, value in counts.items():
                stats.add_row(str(label), format_int(value))
            if db_path:
                stats.add_row(
                    "Saved SQLite tables",
                    f"{db_path}: nasdaq_etf_universe, universe_audit_*",
                )
            self.console.print(stats)
        if universe_degraded:
            self.console.print(
                Text(
                    "Workflow universe is degraded: one or more issuer/ETN sources failed. "
                    "See universe_workflow_source_status in SQLite.",
                    style="yellow",
                )
            )

        if df.empty:
            self.console.print(Text("No long leveraged ETFs found.", style="dim"))
            return

        self.console.print(
            self._table(
                df,
                [
                    TableColumn("symbol", "Asset", no_wrap=True),
                    TableColumn("name", "Name", ratio=1, min_width=30),
                    TableColumn("rsi_symbol", "RSI", no_wrap=True),
                ],
                caption=None,
            )
        )

    def asset_run_summary(self, rows: Iterable[Mapping[str, Any]]) -> None:
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Workflow #")
        self.dataframe(
            "Asset Run Summary",
            df,
            [
                TableColumn("Workflow #", "#", justify="right", no_wrap=True),
                TableColumn("Asset", no_wrap=True),
                TableColumn("RSI Symbol", "RSI", no_wrap=True),
                TableColumn("Action", no_wrap=True),
                TableColumn("Rows", justify="right", formatter=format_int, no_wrap=True),
                TableColumn("Status", status=True, no_wrap=True),
                TableColumn("Message", ratio=1, min_width=24, formatter=format_message),
            ],
            empty_message="No asset workflows were run.",
        )

    def optimization_summary(self, df: pd.DataFrame) -> None:
        display_df = df.drop(columns=["End Date", "Annualized Vol", "Hit Rate"], errors="ignore")
        self.dataframe(
            "Best Sharpe Parameters By Asset",
            display_df,
            [
                TableColumn("Asset", no_wrap=True),
                TableColumn("RSI Symbol", "RSI", no_wrap=True),
                TableColumn("Start Date", no_wrap=True),
                TableColumn("Trading Days", justify="right", formatter=format_int, no_wrap=True),
                TableColumn("Buy RSI", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn(
                    "Sell Return Multiple",
                    "Sell x",
                    justify="right",
                    formatter=format_decimal_2,
                    no_wrap=True,
                ),
                TableColumn("Trades Executed", "Trades", justify="right", formatter=format_int, no_wrap=True),
                TableColumn("Total Return", justify="right", formatter=format_decimal_4, no_wrap=True),
                TableColumn("CAGR", justify="right", formatter=format_decimal_4, no_wrap=True),
                TableColumn("Sharpe", justify="right", formatter=format_decimal_4, no_wrap=True),
                TableColumn("Kelly Fraction", "Kelly", justify="right", formatter=format_decimal_4, no_wrap=True),
                TableColumn("Max Drawdown", "Max DD", justify="right", formatter=format_decimal_4, no_wrap=True),
            ],
            empty_message="No strategies produced more than one executed trade.",
        )

    def signal_report(self, title: str, df: pd.DataFrame, *, empty_message: str) -> None:
        self.dataframe(
            title,
            df,
            [
                TableColumn("Asset", no_wrap=True),
                TableColumn("RSI Symbol", "RSI", no_wrap=True),
                TableColumn("Date", no_wrap=True),
                TableColumn("Latest RSI", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn("Buy RSI", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn(
                    "Sell Return Multiple",
                    "Sell x",
                    justify="right",
                    formatter=format_decimal_2,
                    no_wrap=True,
                ),
                TableColumn("Trades Executed", "Trades", justify="right", formatter=format_int, no_wrap=True),
                TableColumn("Sharpe", justify="right", formatter=format_decimal_4, no_wrap=True),
                TableColumn("In Position", "Held", formatter=format_bool, no_wrap=True),
                TableColumn("Pending Action", "Action", no_wrap=True),
            ],
            empty_message=empty_message,
        )

    def order_results(self, df: pd.DataFrame) -> None:
        self.dataframe(
            "Alpaca Paper Order Results",
            df,
            [
                TableColumn("Asset", no_wrap=True),
                TableColumn("Date", no_wrap=True),
                TableColumn("Notional", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn("Qty", justify="right", formatter=format_qty, no_wrap=True),
                TableColumn("Limit Price", "Limit", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn("Status", status=True, no_wrap=True),
                TableColumn("Message", ratio=1, min_width=24, formatter=format_message),
            ],
            empty_message="No buy signals to submit.",
            caption="Full client and Alpaca order IDs are written to alpaca_order_results.csv.",
        )

    def reconciliation(self, df: pd.DataFrame) -> None:
        self.dataframe(
            "Alpaca Managed Position Reconciliation",
            df,
            [
                TableColumn("Position ID", "ID", justify="right", formatter=format_int, no_wrap=True),
                TableColumn("Asset", no_wrap=True),
                TableColumn("Action", no_wrap=True),
                TableColumn("Status", status=True, no_wrap=True),
                TableColumn("Qty", justify="right", formatter=format_qty, no_wrap=True),
                TableColumn("Limit Price", "Limit", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn("Message", ratio=1, min_width=28, formatter=format_message),
            ],
            empty_message="No managed Alpaca positions to reconcile.",
            caption="Full client and Alpaca order IDs are written to alpaca_reconciliation_results.csv.",
        )

    def realized_pnl_summary(self, df: pd.DataFrame) -> None:
        self.dataframe(
            "Closed Managed Alpaca Realized P/L",
            df,
            [
                TableColumn("Closed Positions", "Closed", justify="right", formatter=format_int, no_wrap=True),
                TableColumn(
                    "Complete Closed Positions",
                    "Complete",
                    justify="right",
                    formatter=format_int,
                    no_wrap=True,
                ),
                TableColumn(
                    "Incomplete Closed Positions",
                    "Incomplete",
                    justify="right",
                    formatter=format_int,
                    no_wrap=True,
                ),
                TableColumn("Total Buy Cost", "Buy Cost", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn(
                    "Total Sell Value",
                    "Sell Value",
                    justify="right",
                    formatter=format_decimal_2,
                    no_wrap=True,
                ),
                TableColumn("Realized P/L", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn("Realized P/L %", justify="right", formatter=format_percent_2, no_wrap=True),
            ],
            empty_message="No closed managed Alpaca positions.",
            caption="Closed positions missing actual sell fill prices are excluded from realized P/L totals.",
        )

    def benchmark_report(self, benchmark: WorkflowBenchmark) -> None:
        self.section("Workflow Benchmark")
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", no_wrap=True)
        table.add_column(ratio=1)
        table.add_row("Status", benchmark.status)
        table.add_row("Started UTC", format_timestamp(benchmark.started_at_utc))
        table.add_row("Finished UTC", format_timestamp(benchmark.finished_at_utc))
        table.add_row("Wall time", format_duration(benchmark.wall_seconds))
        table.add_row("CPU time", format_duration(benchmark.cpu_seconds))
        table.add_row("CPU utilization", format_percent_2(benchmark.cpu_utilization_percent))
        table.add_row("Peak RSS", format_mb(benchmark.peak_rss_mb))
        table.add_row("Current RSS", format_mb(benchmark.current_rss_mb))
        table.add_row(
            "Assets",
            (
                f"{format_int(benchmark.completed_asset_count)} completed / "
                f"{format_int(benchmark.asset_count)} total; "
                f"{format_int(benchmark.skipped_asset_count)} skipped"
            ),
        )
        table.add_row("Rows processed", format_int(benchmark.rows_processed))
        table.add_row("Workflow concurrency", format_int(benchmark.workflow_concurrency))
        self.console.print(table)

    def _table(self, df: pd.DataFrame, columns: list[TableColumn], *, caption: str | None) -> Table:
        table = Table(
            box=box.SIMPLE_HEAVY,
            expand=True,
            row_styles=["", "dim"],
            caption=caption,
            caption_style="dim",
        )
        available_columns = [column for column in columns if column.source in df.columns]
        for column in available_columns:
            table.add_column(
                column.title,
                justify=column.justify,
                style=column.style,
                min_width=column.min_width,
                max_width=column.max_width,
                ratio=column.ratio,
                no_wrap=column.no_wrap,
                overflow=column.overflow,
            )

        for row in df.to_dict("records"):
            table.add_row(*[self._cell(row.get(column.source), column) for column in available_columns])
        return table

    def _cell(self, value: Any, column: TableColumn) -> Text:
        formatter = column.formatter or format_value
        text = formatter(value)
        if column.status:
            return Text(text, style=STATUS_STYLES.get(text.lower(), ""))
        return Text(text)


def format_value(value: Any) -> str:
    if _is_empty(value):
        return ""
    if isinstance(value, bool):
        return format_bool(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_decimal(value, 4)
    return str(value)


def _default_console(*, no_color: bool) -> Console:
    console = Console(no_color=no_color)
    if console.is_terminal:
        return console
    return Console(no_color=no_color, width=DEFAULT_NON_TERMINAL_WIDTH)


def _grid_range_label(values: list[float], *, places: int, step_label: str) -> str:
    if not values:
        return "none"
    return f"{float(values[0]):.{places}f} to {float(values[-1]):.{places}f} step {step_label}"


def format_message(value: Any) -> str:
    if _is_empty(value):
        return ""
    message = str(value).strip()
    if message.endswith("..."):
        message = message[:-3].rstrip()
    return MESSAGE_ALIASES.get(message, _sentence_case(message))


def format_bool(value: Any) -> str:
    if _is_empty(value):
        return ""
    return "yes" if bool(value) else "no"


def format_int(value: Any) -> str:
    if _is_empty(value):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def format_qty(value: Any) -> str:
    return format_decimal(value, 8, trim=True)


def format_decimal_2(value: Any) -> str:
    return format_decimal(value, 2)


def format_decimal_4(value: Any) -> str:
    return format_decimal(value, 4)


def format_percent_2(value: Any) -> str:
    text = format_decimal(value, 2)
    return f"{text}%" if text else text


def format_mb(value: Any) -> str:
    text = format_decimal(value, 2)
    return f"{text} MB" if text else "unavailable"


def format_duration(value: Any) -> str:
    if _is_empty(value):
        return ""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(seconds):
        return ""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:05.2f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes):02d}m {remaining_seconds:05.2f}s"


def format_timestamp(value: Any) -> str:
    if _is_empty(value):
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat(timespec="seconds")
        except TypeError:
            return value.isoformat()
    return str(value)


def format_decimal(value: Any, places: int, *, trim: bool = False) -> str:
    if _is_empty(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    text = f"{number:.{places}f}"
    if trim:
        text = text.rstrip("0").rstrip(".")
    return text


def _sentence_case(value: str) -> str:
    if not value:
        return value
    return value[0].upper() + value[1:]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
