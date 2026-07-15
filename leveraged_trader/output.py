from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Column, Table
from rich.text import Text

DEFAULT_NON_TERMINAL_WIDTH = 156

UNIVERSE_SUMMARY_COUNT_LABELS = (
    "Current long leveraged ETFs/ETNs found",
    "Current short leveraged ETFs/ETNs found",
    "Executable long leveraged ETFs/ETNs selected",
    "Executable short leveraged ETFs/ETNs selected",
    "RSI mappings needing review",
)
UNIVERSE_SUMMARY_NONZERO_COUNT_LABELS = (
    "RSI mappings excluded pending review",
    "Workflow universe sources failed",
    "Active listing sources failed",
    "Audit sources failed",
    "Audit leveraged candidates missing from merged universe",
)

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
    "symbol_migrated": "cyan",
    "disabled": "yellow",
    "fractional_qty": "yellow",
    "incomplete_fill_metadata": "red",
    "insufficient_notional": "yellow",
    "not_held": "yellow",
    "pending_cancel": "yellow",
    "parse_error": "red",
    "skipped": "yellow",
    "source_error": "red",
    "canceled": "red",
    "deferred": "yellow",
    "error": "red",
    "expired": "red",
    "held": "cyan",
    "inactive": "red",
    "not_tradable": "red",
    "open_sell_order": "cyan",
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
    "open sell order already exists for symbol in Alpaca account": ("Open sell order already exists in Alpaca account"),
    "prior managed GTC sell expired; submitted replacement at frozen target price": (
        "Expired managed GTC sell renewed at frozen target price"
    ),
    "renewed managed GTC limit sell after Alpaca confirmed cancellation": (
        "Renewed managed GTC sell after cancellation"
    ),
    "renewed managed GTC limit sell before Alpaca aged-order expiration": (
        "Renewed managed GTC sell before expiration"
    ),
    "submitted one-time GTC limit sell at frozen target price": ("Submitted GTC limit sell at frozen target price"),
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


class WorkflowStepProgress:
    def __init__(self, progress: Progress | None = None, task_id: int | None = None) -> None:
        self._progress = progress
        self._task_id = task_id

    def start_step(self, status: str) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, status=status)

    def finish_step(self) -> None:
        if self._progress is not None and self._task_id is not None:
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
    def step_progress(
        self,
        message: str,
        *,
        total: int,
    ) -> Iterator[WorkflowStepProgress]:
        if not self.console.is_terminal:
            self.console.print(Text(message, style="dim"))
            yield WorkflowStepProgress()
            return

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
            task_id = progress.add_task("Preparing workflow", total=total, status="")
            yield WorkflowStepProgress(progress, task_id)

    @contextmanager
    def asset_progress(self, total: int, *, workflow_label: str | None = None) -> Iterator[AssetProgress]:
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
            description = f"Processing {workflow_label} Assets" if workflow_label else "Processing Assets"
            task_id = progress.add_task(description, total=total, status="")
            yield AssetProgress(progress, task_id)

    def section(self, title: str) -> None:
        self.console.print()
        self.console.print(Text(title, style="bold"))

    def run_header(
        self,
        *,
        started_at_utc: datetime,
        mode: str,
        db_path: str,
        output_dir: str,
        workflow_concurrency: int,
    ) -> None:
        started_local = started_at_utc.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", no_wrap=True)
        table.add_column(ratio=1)
        table.add_row("Mode", mode)
        table.add_row("SQLite database", db_path)
        table.add_row("Output directory", output_dir)
        table.add_row("Download workers", str(workflow_concurrency))
        self.console.print()
        self.console.print(
            Panel(
                table,
                title=Text(f"Workflow Run: {started_local}", style="bold"),
                title_align="left",
                border_style="dim",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def settings(
        self,
        *,
        mode: str,
        db_path: str,
        workflow_concurrency: int,
        risk_free_symbol: str,
        buy_rsi_values: list[float],
        short_buy_rsi_values: list[float] | None = None,
        profit_target_values: list[float],
    ) -> None:
        self.section("Grid Search Settings")
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", no_wrap=True)
        table.add_column(ratio=1)
        table.add_row("Start date", "Earliest overlapping history for each leveraged asset and RSI symbol")
        table.add_row("Sharpe benchmark", f"{risk_free_symbol} 13-week U.S. Treasury bill")
        if short_buy_rsi_values is None:
            table.add_row("Buy RSI values", _grid_range_label(buy_rsi_values, places=0, step_label="1"))
        else:
            table.add_row("Long buy RSI values", _grid_range_label(buy_rsi_values, places=0, step_label="1"))
            table.add_row(
                "Short buy RSI values",
                _grid_range_label(short_buy_rsi_values, places=0, step_label="1"),
            )
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
        max_rows: int | None = None,
        truncated_detail: str | None = None,
    ) -> None:
        self.section(title)
        if df.empty:
            self.console.print(Text(empty_message, style="dim"))
            return
        display_df, table_caption = self._limited_dataframe(
            df,
            caption=caption,
            max_rows=max_rows,
            truncated_detail=truncated_detail,
        )
        self.console.print(self._table(display_df, columns, caption=table_caption))

    def universe_assets(self, df: pd.DataFrame) -> None:
        title = str(df.attrs.get("universe_title", "Workflow ETF Universe"))
        counts = df.attrs.get("universe_counts", {})
        universe_degraded = bool(df.attrs.get("universe_degraded", False))
        source_failures = pd.DataFrame(df.attrs.get("workflow_source_failures", []))
        active_listing_failures = pd.DataFrame(df.attrs.get("active_listing_source_failures", []))
        audit_source_failures = pd.DataFrame(df.attrs.get("audit_source_failures", []))
        mapping_review = pd.DataFrame(df.attrs.get("rsi_mapping_review", []))

        self.section(title)
        summary_counts = _universe_summary_counts(counts)
        if summary_counts:
            stats = Table.grid(padding=(0, 2))
            stats.add_column(style="bold", no_wrap=True)
            stats.add_column(ratio=1)
            for label, value in summary_counts:
                stats.add_row(label, format_int(value))
            self.console.print(stats)
        if universe_degraded:
            self.console.print(
                Text(
                    "Universe is degraded: one or more source checks failed. See the universe source-status "
                    "tables in SQLite.",
                    style="yellow",
                )
            )
            if not source_failures.empty:
                self.dataframe(
                    "Failed Workflow Universe Sources",
                    source_failures,
                    [
                        TableColumn("source", "Source", no_wrap=True),
                        TableColumn("source_type", "Type", no_wrap=True),
                        TableColumn("status", "Status", status=True, no_wrap=True),
                        TableColumn("error", "Error", ratio=1, min_width=28, formatter=format_compact_message),
                    ],
                    empty_message="No failed workflow universe sources.",
                    max_rows=12,
                    truncated_detail="full source health saved in SQLite",
                )
            if not active_listing_failures.empty:
                self.dataframe(
                    "Failed Active Listing Sources",
                    active_listing_failures,
                    [
                        TableColumn("source", "Source", no_wrap=True),
                        TableColumn("status", "Status", status=True, no_wrap=True),
                        TableColumn("error", "Error", ratio=1, min_width=28, formatter=format_compact_message),
                    ],
                    empty_message="No failed active listing sources.",
                    max_rows=12,
                    truncated_detail="full active listing health saved in SQLite",
                )
            if not audit_source_failures.empty:
                self.dataframe(
                    "Failed Audit Universe Sources",
                    audit_source_failures,
                    [
                        TableColumn("source", "Source", no_wrap=True),
                        TableColumn("source_type", "Type", no_wrap=True),
                        TableColumn("status", "Status", status=True, no_wrap=True),
                        TableColumn("error", "Error", ratio=1, min_width=28, formatter=format_compact_message),
                    ],
                    empty_message="No failed audit universe sources.",
                    max_rows=12,
                    truncated_detail="full audit source health saved in SQLite",
                )
        if not mapping_review.empty:
            self.console.print(
                Text(
                    "Some RSI mappings need review before relying on their signal proxy. "
                    "See universe_rsi_mapping_review in SQLite.",
                    style="yellow",
                )
            )
            self.dataframe(
                "RSI Mappings Needing Review",
                mapping_review,
                [
                    TableColumn("workflow", "Workflow", no_wrap=True),
                    TableColumn("symbol", "Asset", no_wrap=True),
                    TableColumn("name", "Name", ratio=1, min_width=26),
                    TableColumn("rsi_symbol", "RSI", no_wrap=True),
                    TableColumn("mapping_reason", "Reason", ratio=1, min_width=28, formatter=format_compact_message),
                ],
                empty_message="No RSI mappings need review.",
                max_rows=12,
                truncated_detail="full RSI mapping review saved in SQLite",
            )

        if df.empty:
            self.console.print(Text("No leveraged ETFs/ETNs found for this workflow.", style="dim"))
            return

        display_df = df.sort_values("symbol", kind="stable").reset_index(drop=True)
        self.console.print(
            self._table(
                display_df,
                [
                    TableColumn("workflow", "Workflow", no_wrap=True),
                    TableColumn("symbol", "Asset", no_wrap=True),
                    TableColumn("name", "Name", ratio=1, min_width=30),
                    TableColumn("rsi_symbol", "RSI", no_wrap=True),
                ],
                caption=None,
            )
        )

    def asset_run_summary(self, rows: Iterable[Mapping[str, Any]], *, title: str = "Asset Run Summary") -> None:
        df = pd.DataFrame(rows)
        if not df.empty and "Asset" in df.columns:
            df = df.sort_values("Asset", kind="stable").reset_index(drop=True)
        self.dataframe(
            title,
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

    def optimization_summary(self, df: pd.DataFrame, *, title: str = "Best Sharpe Parameters By Asset") -> None:
        display_df = df.drop(columns=["End Date", "Annualized Vol", "Hit Rate"], errors="ignore")
        if "Trades Executed" in display_df.columns:
            trades = pd.to_numeric(display_df["Trades Executed"], errors="coerce").fillna(0)
            no_trades = trades.le(0)
            for metric_column in ["Sharpe", "Kelly Fraction"]:
                if metric_column in display_df.columns:
                    display_df[metric_column] = display_df[metric_column].astype("object")
                    display_df.loc[no_trades, metric_column] = pd.NA
            display_df = display_df[trades.ge(2)]
        if "Sharpe" in display_df.columns:
            display_df = display_df[pd.to_numeric(display_df["Sharpe"], errors="coerce").ge(1.0)]
        self.dataframe(
            title,
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
                TableColumn("Sharpe", justify="right", formatter=format_optional_decimal_4, no_wrap=True),
                TableColumn(
                    "Kelly Fraction",
                    "Kelly",
                    justify="right",
                    formatter=format_optional_decimal_4,
                    no_wrap=True,
                ),
                TableColumn("Max Drawdown", "Max DD", justify="right", formatter=format_decimal_4, no_wrap=True),
            ],
            empty_message="No strategies with at least 2 trades and Sharpe >= 1.0.",
        )

    def signal_report(self, title: str, df: pd.DataFrame, *, empty_message: str) -> None:
        columns = [
            TableColumn("Workflow", no_wrap=True),
            TableColumn("Asset", no_wrap=True),
            TableColumn("RSI Symbol", "RSI", no_wrap=True),
            TableColumn("Date", no_wrap=True),
            TableColumn("Start Date", "Start", no_wrap=True),
            TableColumn("Trading Days", "Days", justify="right", formatter=format_int, no_wrap=True),
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
            TableColumn("Sharpe", justify="right", formatter=format_optional_decimal_4, no_wrap=True),
            TableColumn("In Position", "Held", formatter=format_bool, no_wrap=True),
            TableColumn("Pending Action", "Action", no_wrap=True),
        ]
        if self.console.width < 120:
            columns = [column for column in columns if column.source not in {"Start Date", "Latest RSI"}]
        self.dataframe(
            title,
            df,
            columns,
            empty_message=empty_message,
        )

    def buy_signal_eligibility_summary(
        self,
        *,
        buy_signals: pd.DataFrame,
        eligible_buy_signals: pd.DataFrame,
        order_results: pd.DataFrame,
    ) -> None:
        self.section("Buy Signal Eligibility")
        total_buy_signals = len(buy_signals)
        eligible_signals = len(eligible_buy_signals)
        active_managed_skips = max(total_buy_signals - eligible_signals, 0)
        submitted_or_existing = 0
        live_preflight_skips = 0
        if not order_results.empty and "Status" in order_results:
            statuses = order_results["Status"].astype(str).str.lower()
            submitted_or_existing = int(statuses.isin({"submitted", "existing"}).sum())
            live_preflight_skips = int((~statuses.isin({"submitted", "existing", "managed"})).sum())

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        table.add_row("Eligible buy signals", f"{format_int(eligible_signals)} / {format_int(total_buy_signals)}")
        if submitted_or_existing:
            table.add_row("Submitted/existing buys", format_int(submitted_or_existing))
        if active_managed_skips:
            table.add_row("Skipped: active managed", format_int(active_managed_skips))
        if live_preflight_skips:
            table.add_row("Skipped: Alpaca/live preflight", format_int(live_preflight_skips))
        self.console.print(table)

    def order_results(self, df: pd.DataFrame) -> None:
        columns = []
        if "Display ID" in df.columns:
            columns.append(TableColumn("Display ID", "ID", justify="right", formatter=format_int, no_wrap=True))
        columns.extend(
            [
                TableColumn("Workflow", no_wrap=True),
                TableColumn("Asset", no_wrap=True),
                TableColumn("Date", no_wrap=True),
                TableColumn("Notional", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn("Qty", justify="right", formatter=format_qty, no_wrap=True),
                TableColumn("Limit Price", "Limit", justify="right", formatter=format_decimal_2, no_wrap=True),
                TableColumn("Status", status=True, no_wrap=True),
                TableColumn("Message", ratio=1, min_width=24, formatter=format_message),
            ]
        )
        self.dataframe(
            "Alpaca Paper Order Results",
            df,
            columns,
            empty_message="No buy signals to submit.",
            caption="Full client and Alpaca order IDs are written to alpaca_order_results.csv.",
        )

    def reconciliation(self, df: pd.DataFrame) -> None:
        id_column = "Display ID" if "Display ID" in df.columns else "Position ID"
        self.dataframe(
            "Alpaca Managed Position Reconciliation",
            df,
            [
                TableColumn(id_column, "ID", justify="right", formatter=format_int, no_wrap=True),
                TableColumn("Workflow", no_wrap=True),
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

    def workflow_footer(
        self,
        elapsed_seconds: float,
    ) -> None:
        self.console.print()
        self.console.print(f"Workflow finished in {format_duration(elapsed_seconds)}.")
        self.console.rule(style="dim")

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

    def _limited_dataframe(
        self,
        df: pd.DataFrame,
        *,
        caption: str | None,
        max_rows: int | None,
        truncated_detail: str | None,
    ) -> tuple[pd.DataFrame, str | None]:
        if max_rows is None or len(df) <= max_rows:
            return df, caption

        shown_rows = max(max_rows, 0)
        display_df = df.head(shown_rows)
        limit_caption = f"Showing {format_int(shown_rows)} of {format_int(len(df))} rows"
        if truncated_detail:
            limit_caption = f"{limit_caption}; {truncated_detail}"
        limit_caption = f"{limit_caption}."
        return display_df, _combine_captions(caption, limit_caption)

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


def format_compact_message(value: Any) -> str:
    message = format_message(value)
    max_length = 140
    if len(message) <= max_length:
        return message
    return f"{message[: max_length - 3].rstrip()}..."


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


def format_optional_decimal_4(value: Any) -> str:
    if _is_empty(value):
        return "N/A"
    return format_decimal(value, 4)


def format_percent_2(value: Any) -> str:
    text = format_decimal(value, 2)
    return f"{text}%" if text else text


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


def _universe_summary_counts(counts: Mapping[object, object]) -> list[tuple[str, object]]:
    rows = [(label, counts[label]) for label in UNIVERSE_SUMMARY_COUNT_LABELS if label in counts]
    rows.extend(
        (label, counts[label])
        for label in UNIVERSE_SUMMARY_NONZERO_COUNT_LABELS
        if label in counts and _is_nonzero_count(counts[label])
    )
    return rows


def _is_nonzero_count(value: object) -> bool:
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return bool(value)


def _combine_captions(caption: str | None, extra: str) -> str:
    if not caption:
        return extra
    return f"{caption}\n{extra}"


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
