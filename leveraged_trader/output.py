from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Column, Table
from rich.text import Text

DEFAULT_NON_TERMINAL_WIDTH = 156
DEFAULT_DIAGNOSTIC_MAX_CHARS = 512

# Accept conventional separated vendor prefixes and camel/PascalCase segments.
# Requiring an uppercase boundary keeps an arbitrary all-lowercase word from
# being split merely because it happens to end in a credential-key word.
_DIAGNOSTIC_CREDENTIAL_VENDOR_PREFIX_EXPRESSION = r"(?:[A-Z0-9]++[._-]++)*(?:(?-i:[A-Za-z][A-Za-z0-9]*?(?=[A-Z])))?"
_DIAGNOSTIC_CREDENTIAL_BASE_KEY_EXPRESSION = (
    _DIAGNOSTIC_CREDENTIAL_VENDOR_PREFIX_EXPRESSION
    + r"(?:"
    + r"api[ ._-]*(?:key(?:[ ._-]*id)?|secret(?:[ ._-]*key)?|token)"
    + r"|(?:access|refresh|session|auth|id|client)[ ._-]*(?:token|secret)"
    + r"|secret[ ._-]*access[ ._-]*key"
    + r"|secret(?:[ ._-]*key)?"
    + r"|credential|token|authorization|password|passwd"
    + r")"
)
_DIAGNOSTIC_CREDENTIAL_KEY_SUFFIX_EXPRESSION = r"(?:[ ._-]*(?:header|value)){1,2}"
_DIAGNOSTIC_CREDENTIAL_KEY_EXPRESSION = (
    rf"(?:{_DIAGNOSTIC_CREDENTIAL_BASE_KEY_EXPRESSION})"
    rf"(?:{_DIAGNOSTIC_CREDENTIAL_KEY_SUFFIX_EXPRESSION})?"
)
_DIAGNOSTIC_CREDENTIAL_KEY_PATTERN = re.compile(
    rf"(?:{_DIAGNOSTIC_CREDENTIAL_KEY_EXPRESSION})\Z",
    re.I,
)
_DIAGNOSTIC_SECRET_ASSIGNMENT_PATTERN = re.compile(
    # Start only at the beginning of a connected dotted/dashed key. Allowing
    # every dot or dash to begin another match made a failed credential lookup
    # rescan the entire remaining suffix (quadratic for inputs such as A.A.A.).
    r"(?ix)(?<![A-Z0-9_.-])"
    r"(?P<credential_leading_separators>[._-]*+)"
    r"(?P<credential_key_quote>[\"'])?"
    rf"{_DIAGNOSTIC_CREDENTIAL_BASE_KEY_EXPRESSION}"
    rf"(?P<credential_key_suffix>{_DIAGNOSTIC_CREDENTIAL_KEY_SUFFIX_EXPRESSION})?"
    r"(?(credential_key_quote)(?P=credential_key_quote))"
    r"\s*(?::|=|(?(credential_key_suffix)(?!)|\bis\b))\s*"
    r"(?:[A-Z0-9!#$%&'*+.^_`|~-]+\s+)?"
    r'(?:"(?:\\.|[^"\\])*(?:"|$)|\'(?:\\.|[^\'\\])*(?:\'|$)|[^\s,;]+)'
)
_DIAGNOSTIC_AUTHORIZATION_ASSIGNMENT_START_PATTERN = re.compile(
    r"(?ix)(?<![A-Z0-9_.-])"
    r"(?P<credential_leading_separators>[._-]*+)"
    r"(?P<authorization_key_quote>[\"'])?"
    rf"{_DIAGNOSTIC_CREDENTIAL_VENDOR_PREFIX_EXPRESSION}authorization"
    rf"(?P<authorization_key_suffix>{_DIAGNOSTIC_CREDENTIAL_KEY_SUFFIX_EXPRESSION})?"
    r"(?(authorization_key_quote)(?P=authorization_key_quote))"
    r"\s*(?::|=|(?(authorization_key_suffix)(?!)|\bis\b))\s*"
)
_DIAGNOSTIC_BARE_AUTHORIZATION_SCHEME_PATTERN = re.compile(
    r"(?:api-?key|aws4-hmac-sha256|basic|bearer|digest|hoba|mutual|negotiate|oauth|"
    r"scram-sha-256|signature|token|vapid)\Z",
    re.I,
)
_DIAGNOSTIC_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_DIAGNOSTIC_CREDENTIALISH_TOKEN_PATTERN = re.compile(
    # As above, inspect a connected token once rather than restarting after
    # every punctuation character in a hostile provider-controlled value.
    r"(?i)(?<![A-Z0-9_.~-])"
    r"(?P<credential_token_leading_hyphens>-*+)"
    r"(?=[A-Z0-9.~-]*(?:secret|token|credential|api-key))"
    r"[A-Z0-9.]+(?:-[A-Z0-9.]+)+(?![A-Z0-9_])"
)
_DIAGNOSTIC_TECHNICAL_VERSION_SEGMENT_PATTERN = re.compile(
    r"(?:"
    r"v\d{1,4}(?:(?:alpha|beta|rc)\d{0,4})?"
    r"|(?:http|status)[1-5]\d{2}"
    r"|(?:java|node|php|python|py|ruby)\d{1,3}(?:\.\d{1,3}){0,2}"
    r"|rfc\d{1,5}"
    r"|(?:ssl|tls)\d(?:\.\d){0,2}"
    r")\Z",
    re.I,
)
# Userinfo is valid in any URI authority, not only HTTP(S).  Provider error
# bodies and proxy exceptions can contain schemes such as ``socks5`` or
# ``postgresql``; keep the RFC 3986 scheme grammar generic so those credentials
# receive the same structural redaction.
_DIAGNOSTIC_URL_SCHEME_PATTERN = re.compile(r"(?<![A-Z0-9+.-])[A-Z][A-Z0-9+.-]*://", re.I)
_DIAGNOSTIC_URL_USERINFO_OVERLAP_CHARS = 2_048
_DIAGNOSTIC_JSON_KEY_MAX_CHARS = 512
_DIAGNOSTIC_JSON_STRING_OVER_LIMIT = -1
_DIAGNOSTIC_NESTED_JSON_MAX_DEPTH = 4
_DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH = 3
_DIAGNOSTIC_COMPOSED_ENCODING_WORK_FACTOR = 32
_DIAGNOSTIC_AMBIGUOUS_PERCENT_SECRET_MAX_CHARS = 128
_DIAGNOSTIC_REDACTED_CREDENTIAL = "[redacted credential]"
_JSON_SIMPLE_ESCAPE_CHARACTERS = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_JSON_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")

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
    "batch_aborted": "yellow",
    "batch_budget_exhausted": "yellow",
    "broker_inactive": "yellow",
    "corrected": "green",
    "filled": "green",
    "submitted": "green",
    "partially_filled": "yellow",
    "submission_pending": "yellow",
    "submission_failed": "red",
    "submission_unknown": "red",
    "submission_not_found": "yellow",
    "unowned_existing": "red",
    "done": "green",
    "duplicate_signal": "yellow",
    "existing": "cyan",
    "managed": "cyan",
    "managed_only": "cyan",
    "open_order": "cyan",
    "renewed": "green",
    "symbol_migrated": "cyan",
    "disabled": "yellow",
    "fill_quantity_regression": "red",
    "fractional_qty": "yellow",
    "identity_mismatch": "red",
    "incomplete_fill_metadata": "red",
    "incomplete_order_metadata": "red",
    "insufficient_notional": "yellow",
    "managed_order_conflict": "red",
    "not_held": "yellow",
    "pending_cancel": "yellow",
    "position_quantity_mismatch": "red",
    "parse_error": "red",
    "skipped": "yellow",
    "source_error": "red",
    "canceled": "red",
    "deferred": "yellow",
    "done_for_day": "yellow",
    "error": "red",
    "expired": "red",
    "held": "cyan",
    "inactive": "red",
    "not_tradable": "red",
    "open_sell_order": "cyan",
    "rejected": "red",
    "stale_observation": "yellow",
    "stopped": "yellow",
    "superseded": "yellow",
    "suspended": "yellow",
    "quantity_mismatch": "red",
}


# Keep benign skips explicit: newly introduced or malformed statuses must fall
# through to the manual-review bucket instead of being reported as preflight
# skips. These groups cover every status currently produced by the buy-order
# pipeline, plus broker lifecycle states that must never be treated as retryable.
_BUY_SUCCESS_OR_RECONCILING_STATUSES = frozenset(
    {
        "submitted",
        "existing",
        "filled",
        "partially_filled",
    }
)
_BUY_AMBIGUOUS_STATUSES = frozenset(
    {
        "submission_pending",
        "submission_unknown",
        "pending_cancel",
        "superseded",
        "accepted",
        "accepted_for_bidding",
        "calculated",
        "new",
        "pending_new",
        "pending_replace",
        "replaced",
        "done_for_day",
        "stopped",
        "suspended",
    }
)
_BUY_REVIEW_STATUSES = frozenset(
    {
        "incomplete_fill_metadata",
        "incomplete_order_metadata",
        "fill_quantity_regression",
        "position_quantity_mismatch",
        "quantity_mismatch",
        "managed_order_conflict",
    }
)
_BUY_BUDGET_OR_SIZE_SKIP_STATUSES = frozenset(
    {
        "batch_budget_exhausted",
        "insufficient_notional",
        "fractional_qty",
    }
)
_BUY_PREFLIGHT_SKIP_STATUSES = frozenset(
    {
        "stale_observation",
        "duplicate_signal",
        "open_order",
        "open_sell_order",
        "held",
        "not_tradable",
        "inactive",
        "deferred",
    }
)
_BUY_FAILURE_STATUSES = frozenset(
    {
        "error",
        "identity_mismatch",
        "rejected",
        "canceled",
        "expired",
        "submission_failed",
        "submission_not_found",
        "unowned_existing",
    }
)
_BUY_BATCH_ABORT_STATUSES = frozenset({"batch_aborted"})
_KNOWN_BUY_RESULT_STATUSES = frozenset(
    {
        *_BUY_SUCCESS_OR_RECONCILING_STATUSES,
        *_BUY_AMBIGUOUS_STATUSES,
        *_BUY_REVIEW_STATUSES,
        *_BUY_BUDGET_OR_SIZE_SKIP_STATUSES,
        *_BUY_PREFLIGHT_SKIP_STATUSES,
        *_BUY_FAILURE_STATUSES,
        *_BUY_BATCH_ABORT_STATUSES,
        "managed",
    }
)


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
        # These should already have passed the CLI/runtime path boundary, but
        # library callers can supply them directly. Keep the run banner
        # single-line and terminal-safe if that upstream validation is bypassed.
        safe_db_path = "".join(character if character.isprintable() else " " for character in str(db_path))
        safe_output_dir = "".join(character if character.isprintable() else " " for character in str(output_dir))
        table.add_row("SQLite database", Text(safe_db_path))
        table.add_row("Output directory", Text(safe_output_dir))
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
            table.add_row("Buy RSI values", _grid_range_label(buy_rsi_values, places=0))
        else:
            table.add_row("Long buy RSI values", _grid_range_label(buy_rsi_values, places=0))
            table.add_row(
                "Short buy RSI values",
                _grid_range_label(short_buy_rsi_values, places=0),
            )
        table.add_row(
            "Sell return multiples",
            _grid_range_label(profit_target_values, places=1),
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
            TableColumn("RSI Observation Date", "RSI Date", no_wrap=True),
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
        concurrent_managed_skips = 0
        successful_or_reconciling = 0
        ambiguous_submissions = 0
        review_required = 0
        budget_or_size_skips = 0
        batch_safety_aborts = 0
        live_failures = 0
        live_preflight_skips = 0
        unclassified_outcomes = 0
        if not order_results.empty and "Status" in order_results:
            statuses = order_results["Status"].astype(str).str.lower()
            managed_results = order_results.loc[statuses.eq("managed")]
            identity_columns = [
                column
                for column in ("Workflow", "Asset", "Date")
                if column in managed_results.columns and column in eligible_buy_signals.columns
            ]
            if "Asset" in identity_columns and not managed_results.empty and not eligible_buy_signals.empty:
                eligible_identities = eligible_buy_signals[identity_columns].astype(str).drop_duplicates()
                concurrent_managed_skips = len(
                    managed_results[identity_columns]
                    .astype(str)
                    .merge(
                        eligible_identities,
                        how="inner",
                        on=identity_columns,
                    )
                )
            successful_or_reconciling = int(statuses.isin(_BUY_SUCCESS_OR_RECONCILING_STATUSES).sum())
            ambiguous_submissions = int(statuses.isin(_BUY_AMBIGUOUS_STATUSES).sum())
            review_required = int(statuses.isin(_BUY_REVIEW_STATUSES).sum())
            budget_or_size_skips = int(statuses.isin(_BUY_BUDGET_OR_SIZE_SKIP_STATUSES).sum())
            batch_safety_aborts = int(statuses.isin(_BUY_BATCH_ABORT_STATUSES).sum())
            live_preflight_skips = int(statuses.isin(_BUY_PREFLIGHT_SKIP_STATUSES).sum())
            live_failures = int(statuses.isin(_BUY_FAILURE_STATUSES).sum())
            unclassified_outcomes = int((~statuses.isin(_KNOWN_BUY_RESULT_STATUSES)).sum())

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        table.add_row("Eligible buy signals", f"{format_int(eligible_signals)} / {format_int(total_buy_signals)}")
        if successful_or_reconciling:
            table.add_row("Submitted/existing/filled buys", format_int(successful_or_reconciling))
        if active_managed_skips:
            table.add_row("Skipped: active managed", format_int(active_managed_skips))
        if concurrent_managed_skips:
            table.add_row("Skipped: concurrently managed", format_int(concurrent_managed_skips))
        if ambiguous_submissions:
            table.add_row("Pending/unknown broker outcome — do not retry", format_int(ambiguous_submissions))
        if review_required:
            table.add_row("Requires broker review/reconciliation", format_int(review_required))
        if budget_or_size_skips:
            table.add_row("Skipped: budget/size", format_int(budget_or_size_skips))
        if batch_safety_aborts:
            table.add_row("Skipped: batch safety abort", format_int(batch_safety_aborts))
        if live_preflight_skips:
            table.add_row("Skipped: Alpaca/live preflight", format_int(live_preflight_skips))
        if live_failures:
            table.add_row("Failed: Alpaca/live", format_int(live_failures))
        if unclassified_outcomes:
            table.add_row("Unclassified broker outcome — manual review", format_int(unclassified_outcomes))
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
                TableColumn("Workflow", no_wrap=True),
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

    def workflow_timings(
        self,
        *,
        download_seconds: float,
        state_validation_seconds: float,
        grid_compute_seconds: float,
        db_sync_seconds: float,
        report_generation_seconds: float,
        alpaca_seconds: float,
        batch_count: int,
        individual_retry_count: int,
        rebuild_count: int,
        update_count: int,
    ) -> None:
        self.console.print(
            "Timing (overlap-aware): "
            f"downloads {format_duration(download_seconds)}, "
            f"state verification {format_duration(state_validation_seconds)}, "
            f"grid {format_duration(grid_compute_seconds)}, "
            f"database {format_duration(db_sync_seconds)}, "
            f"reports {format_duration(report_generation_seconds)}, "
            f"Alpaca {format_duration(alpaca_seconds)}."
        )
        self.console.print(
            f"Market-data batches: {batch_count}; individual retries: {individual_retry_count}; "
            f"rebuilt assets: {rebuild_count}; updated assets: {update_count}."
        )

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
        # Table values can include provider-controlled labels. Rich ``Text``
        # avoids markup interpretation, but intentionally preserves terminal
        # control characters. Keep cells single-line and printable even if an
        # upstream validation boundary is accidentally bypassed.
        text = "".join(character if character.isprintable() else " " for character in str(formatter(value)))
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


def _grid_range_label(values: list[float], *, places: int) -> str:
    if not values:
        return "none"

    decimal_values = [Decimal(str(float(value))) for value in values]
    formatted_values = [_format_grid_decimal(value, minimum_places=places) for value in decimal_values]
    if len(values) == 1:
        return formatted_values[0]

    step = decimal_values[1] - decimal_values[0]
    is_regular = all(
        current - previous == step for previous, current in zip(decimal_values[:-1], decimal_values[1:], strict=True)
    )
    if is_regular:
        step_label = _format_grid_decimal(step, minimum_places=places)
        return f"{formatted_values[0]} to {formatted_values[-1]} step {step_label}"
    return ", ".join(formatted_values)


def _format_grid_decimal(value: Decimal, *, minimum_places: int) -> str:
    if value.is_zero():
        value = value.copy_abs()
    whole, dot, fractional = format(value, "f").partition(".")
    fractional = fractional.rstrip("0")
    if len(fractional) < minimum_places:
        fractional += "0" * (minimum_places - len(fractional))
    return f"{whole}.{fractional}" if fractional or minimum_places else whole


def _diagnostic_json_string_end(
    value: str,
    start: int,
    *,
    max_chars: int | None = None,
) -> int | None:
    """Return a JSON string's end, or a negative sentinel when it exceeds the bound."""
    if start >= len(value) or value[start] != '"':
        return None
    limit = len(value) if max_chars is None else min(len(value), start + max_chars)
    cursor = start + 1
    while cursor < limit:
        character = value[cursor]
        if character == '"':
            return cursor + 1
        if ord(character) < 0x20:
            return None
        if character != "\\":
            cursor += 1
            continue
        if cursor + 1 >= limit:
            return _DIAGNOSTIC_JSON_STRING_OVER_LIMIT if limit < len(value) else None
        escape_character = value[cursor + 1]
        if escape_character in _JSON_SIMPLE_ESCAPE_CHARACTERS:
            cursor += 2
            continue
        escape_end = cursor + 6
        if escape_character != "u":
            return None
        if escape_end > limit:
            return _DIAGNOSTIC_JSON_STRING_OVER_LIMIT if limit < len(value) else None
        if any(character not in _JSON_HEX_DIGITS for character in value[cursor + 2 : escape_end]):
            return None
        cursor = escape_end
    return _DIAGNOSTIC_JSON_STRING_OVER_LIMIT if limit < len(value) else None


def _decode_diagnostic_json_string_prefix(
    value: str,
    start: int,
) -> tuple[str, list[tuple[int, int]], int, bool] | None:
    """Decode one bounded JSON string prefix and retain its source spans."""
    if start >= len(value) or value[start] != '"':
        return None

    decoded: list[str] = []
    source_spans: list[tuple[int, int]] = []
    cursor = start + 1
    while cursor < len(value):
        character = value[cursor]
        if character == '"':
            return "".join(decoded), source_spans, cursor + 1, True
        if ord(character) < 0x20:
            return None
        if character != "\\":
            decoded.append(character)
            source_spans.append((cursor, cursor + 1))
            cursor += 1
            continue

        escape_start = cursor
        if cursor + 1 >= len(value):
            return "".join(decoded), source_spans, len(value), False
        escape_character = value[cursor + 1]
        if escape_character in _JSON_SIMPLE_ESCAPE_CHARACTERS:
            decoded.append(_JSON_SIMPLE_ESCAPE_CHARACTERS[escape_character])
            source_spans.append((escape_start, cursor + 2))
            cursor += 2
            continue
        if escape_character != "u":
            return None

        escape_end = cursor + 6
        if escape_end > len(value):
            available_digits = value[cursor + 2 :]
            if any(digit not in _JSON_HEX_DIGITS for digit in available_digits):
                return None
            return "".join(decoded), source_spans, len(value), False
        hex_digits = value[cursor + 2 : escape_end]
        if any(digit not in _JSON_HEX_DIGITS for digit in hex_digits):
            return None
        first_code_unit = int(hex_digits, 16)
        source_end = escape_end
        decoded_character = chr(first_code_unit)
        if (
            0xD800 <= first_code_unit <= 0xDBFF
            and escape_end + 6 <= len(value)
            and value[escape_end : escape_end + 2] == "\\u"
        ):
            second_hex_digits = value[escape_end + 2 : escape_end + 6]
            if all(digit in _JSON_HEX_DIGITS for digit in second_hex_digits):
                second_code_unit = int(second_hex_digits, 16)
                if 0xDC00 <= second_code_unit <= 0xDFFF:
                    decoded_character = chr(0x10000 + ((first_code_unit - 0xD800) << 10) + (second_code_unit - 0xDC00))
                    source_end = escape_end + 6
        decoded.append(decoded_character)
        source_spans.append((escape_start, source_end))
        cursor = source_end

    return "".join(decoded), source_spans, len(value), False


def _diagnostic_quoted_value_end(value: str, start: int) -> int:
    """Return a bounded diagnostic quote's end, tolerating non-JSON escapes."""
    quote = value[start]
    cursor = start + 1
    while cursor < len(value):
        character = value[cursor]
        if character == "\\" and cursor + 1 < len(value):
            cursor += 2
            continue
        cursor += 1
        if character == quote:
            return cursor
    return len(value)


def _diagnostic_json_value_end(value: str, start: int) -> int:
    """Find one JSON-like value boundary inside an already bounded diagnostic."""
    if start >= len(value):
        return start
    if value[start] in {'"', "'"}:
        return _diagnostic_quoted_value_end(value, start)
    if value[start] not in "[{":
        cursor = start
        while cursor < len(value) and value[cursor] not in ",;]}\r\n\t ":
            cursor += 1
        return cursor

    depth = 0
    quote: str | None = None
    escaped = False
    for cursor in range(start, len(value)):
        character = value[cursor]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
            if depth <= 0:
                return cursor + 1
    return len(value)


def _replace_diagnostic_spans(
    value: str,
    spans: Iterable[tuple[int, int]],
    *,
    marker: str,
) -> str:
    """Replace possibly overlapping spans without copying outside ``value``."""
    merged_spans: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged_spans and start <= merged_spans[-1][1]:
            prior_start, prior_end = merged_spans[-1]
            merged_spans[-1] = (prior_start, max(prior_end, end))
        else:
            merged_spans.append((start, end))
    if not merged_spans:
        return value
    pieces: list[str] = []
    cursor = 0
    for start, end in merged_spans:
        pieces.extend((value[cursor:start], marker))
        cursor = end
    pieces.append(value[cursor:])
    return "".join(pieces)


def _diagnostic_semantic_json_credential_assignment_spans(
    value: str,
    *,
    preserve_json_values: bool = False,
) -> list[tuple[int, int]]:
    """Locate bounded JSON fields after decoding valid escapes in their keys."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(value):
        key_start = value.find('"', cursor)
        if key_start < 0:
            break
        key_end = _diagnostic_json_string_end(
            value,
            key_start,
            max_chars=_DIAGNOSTIC_JSON_KEY_MAX_CHARS,
        )
        if key_end == _DIAGNOSTIC_JSON_STRING_OVER_LIMIT:
            previous_non_whitespace = key_start - 1
            while previous_non_whitespace >= 0 and value[previous_non_whitespace].isspace():
                previous_non_whitespace -= 1
            if previous_non_whitespace >= 0 and value[previous_non_whitespace] == ":":
                # This is a bounded JSON string value, not a candidate key.
                # Skip it as one token so quotes in multiply escaped content
                # cannot be reconsidered as keys by later iterations.
                value_end = _diagnostic_json_string_end(value, key_start)
                if value_end is not None:
                    cursor = value_end
                    continue
            # A provider controls this diagnostic and could hide a semantic
            # credential suffix beyond our fixed key-token scan budget. Avoid
            # both unbounded parsing and disclosure by dropping the remaining
            # ambiguous field/diagnostic text.
            spans.append((key_start, len(value)))
            break
        if key_end is None:
            cursor = key_start + 1
            continue
        cursor = key_end
        try:
            decoded_key = json.loads(value[key_start:key_end])
        except (TypeError, ValueError):
            continue
        if not isinstance(decoded_key, str) or _DIAGNOSTIC_CREDENTIAL_KEY_PATTERN.fullmatch(decoded_key) is None:
            continue

        value_start = key_end
        while value_start < len(value) and value[value_start].isspace():
            value_start += 1
        if value_start >= len(value) or value[value_start] != ":":
            continue
        value_start += 1
        while value_start < len(value) and value[value_start].isspace():
            value_start += 1
        if value_start >= len(value):
            spans.append((key_start, value_start))
            continue
        value_end = _diagnostic_json_value_end(value, value_start)
        spans.append((value_start if preserve_json_values else key_start, value_end))
        cursor = max(cursor, value_end)
    return spans


def _redact_semantic_json_credential_assignments(
    value: str,
    *,
    preserve_json_values: bool = False,
) -> str:
    """Redact bounded JSON fields after decoding valid escapes in their keys."""
    spans = _diagnostic_semantic_json_credential_assignment_spans(
        value,
        preserve_json_values=preserve_json_values,
    )
    marker = json.dumps(_DIAGNOSTIC_REDACTED_CREDENTIAL) if preserve_json_values else _DIAGNOSTIC_REDACTED_CREDENTIAL
    return _replace_diagnostic_spans(value, spans, marker=marker)


def _diagnostic_authorization_assignment_spans(value: str) -> list[tuple[int, int]]:
    """Locate entire bounded Authorization values, including auth parameters."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while match := _DIAGNOSTIC_AUTHORIZATION_ASSIGNMENT_START_PATTERN.search(value, cursor):
        value_start = match.end()
        if value_start < len(value) and value[value_start] in {'"', "'"}:
            value_end = _diagnostic_quoted_value_end(value, value_start)
        else:
            value_end = value_start
            # Digest and AWS parameter values may themselves contain commas and
            # semicolons (including quoted URIs and SignedHeaders lists). A raw
            # Authorization header has no trustworthy same-line prose boundary,
            # so retain only a following physical line rather than risk exposing
            # a later response/signature parameter.
            while value_end < len(value):
                if value[value_end] not in "\r\n":
                    value_end += 1
                    continue
                continuation_start = value_end + 1
                if value[value_end] == "\r" and continuation_start < len(value) and value[continuation_start] == "\n":
                    continuation_start += 1
                if continuation_start < len(value) and value[continuation_start] in " \t":
                    value_end = continuation_start + 1
                    continue
                break
        spans.append(
            (
                match.start() + len(match.group("credential_leading_separators")),
                value_end,
            )
        )
        cursor = max(value_end, match.end(), match.start() + 1)
    return spans


def _redact_authorization_assignments(value: str) -> str:
    """Redact an entire bounded Authorization value, including auth parameters."""
    return _replace_diagnostic_spans(
        value,
        _diagnostic_authorization_assignment_spans(value),
        marker="[redacted credential]",
    )


def _diagnostic_authorization_span_ends_with_bare_scheme(
    value: str,
    *,
    start: int,
    end: int,
) -> bool:
    """Recognize an auth line whose normalized continuation can supply its token."""
    match = _DIAGNOSTIC_AUTHORIZATION_ASSIGNMENT_START_PATTERN.search(value, start, end)
    if match is None:
        return False
    assignment_start = match.start() + len(match.group("credential_leading_separators"))
    if assignment_start != start:
        return False
    return _DIAGNOSTIC_BARE_AUTHORIZATION_SCHEME_PATTERN.fullmatch(value[match.end() : end].strip()) is not None


def _diagnostic_url_userinfo_spans(value: str, *, retained_chars: int) -> list[tuple[int, int]]:
    """Locate credentials in bounded URI authorities.

    The final ``@`` in an authority is its userinfo delimiter, including the
    permissive forms normalized by Requests. Scan a fixed overlap beyond the
    retained diagnostic so a long password cannot leak merely by crossing the
    truncation boundary. If that bounded authority itself is incomplete, hide
    it conservatively rather than emit a possible password prefix.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    while match := _DIAGNOSTIC_URL_SCHEME_PATTERN.search(value, cursor):
        authority_start = match.end()
        authority_end = authority_start
        # A pathless authority ends at prose whitespace or a JSON/string
        # boundary. Do not let it consume a later email address or URI. Preserve
        # the permissive ``user:pass word@host`` form emitted by some clients
        # when the first token after the space supplies the userinfo delimiter.
        while authority_end < len(value) and value[authority_end] not in '/?#\r\n"<>':
            authority_end += 1

        whitespace = next(
            (offset for offset in range(authority_start, authority_end) if value[offset].isspace()),
            None,
        )
        if whitespace is not None:
            compact_authority_end = whitespace
            compact_at_sign = value.rfind("@", authority_start, compact_authority_end)
            if compact_at_sign >= authority_start or value.find(":", authority_start, compact_authority_end) < 0:
                authority_end = compact_authority_end

        at_sign = value.rfind("@", authority_start, authority_end)
        if at_sign >= authority_start:
            password_separator = value.find(":", authority_start, at_sign)
            secret_start = password_separator + 1 if password_separator >= authority_start else authority_start
            if secret_start < at_sign:
                spans.append((secret_start, min(at_sign, retained_chars)))
            elif password_separator > authority_start:
                # An explicit but empty password still leaves the username as
                # reusable userinfo. Preserve the empty-password syntax while
                # hiding that credential rather than emitting it verbatim.
                spans.append((authority_start, min(password_separator, retained_chars)))
        elif authority_end == len(value) and len(value) > retained_chars and authority_start < retained_chars:
            # The authority ran through the bounded look-ahead. It may be an
            # ordinary very long host, but retaining it risks exposing userinfo
            # whose delimiter lies just beyond the scan. Preserve only the
            # already-visible scheme in this exceptional case.
            spans.append((authority_start, retained_chars))

        # Resume after this scheme token, not after its candidate authority: a
        # later URI may begin in prose before the first URI's structural scan
        # reaches a slash, and it must receive an independent userinfo scan.
        cursor = max(match.end(), match.start() + 1)

    return spans


def _diagnostic_json_url_userinfo_spans(
    value: str,
    *,
    retained_chars: int,
) -> list[tuple[int, int]]:
    """Locate URL userinfo after decoding bounded JSON string content."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(value):
        string_start = value.find('"', cursor)
        if string_start < 0:
            break
        parsed = _decode_diagnostic_json_string_prefix(value, string_start)
        if parsed is None:
            cursor = string_start + 1
            continue
        decoded, source_spans, string_end, complete = parsed
        decoded_retained_chars = sum(source_start < retained_chars for source_start, _ in source_spans)
        for decoded_start, decoded_end in _diagnostic_url_userinfo_spans(
            decoded,
            retained_chars=decoded_retained_chars,
        ):
            if decoded_end <= decoded_start or decoded_start >= len(source_spans):
                continue
            raw_start = source_spans[decoded_start][0]
            raw_end = source_spans[min(decoded_end, len(source_spans)) - 1][1]
            if raw_start < retained_chars:
                spans.append((raw_start, min(raw_end, retained_chars)))
        if not complete:
            break
        cursor = max(string_end, string_start + 1)
    return spans


def _diagnostic_sensitive_value_spans(
    value: str,
    *,
    retained_chars: int,
    secrets: set[str],
    json_sensitive_values: set[str],
) -> list[tuple[int, int]]:
    """Locate configured secrets in literal and valid JSON-escaped spellings."""
    spans: list[tuple[int, int]] = []
    for sensitive in secrets:
        start = value.find(sensitive)
        while 0 <= start < retained_chars:
            spans.append((start, min(start + len(sensitive), retained_chars)))
            start = value.find(sensitive, start + 1)

    if secrets:
        canonical_value, canonical_source_spans = _canonical_nfd_diagnostic_tokens(value)
        if canonical_value != value:
            spans.extend(
                _mapped_sensitive_value_spans(
                    canonical_value,
                    canonical_source_spans,
                    retained_chars=retained_chars,
                    sensitive_values={unicodedata.normalize("NFD", sensitive) for sensitive in secrets},
                )
            )

    if json_sensitive_values:
        json_needles = {_json_code_unit_spelling(sensitive) for sensitive in json_sensitive_values}
        decoded_needles = json_sensitive_values | json_needles
        canonical_decoded_needles = {unicodedata.normalize("NFD", sensitive) for sensitive in decoded_needles}
        for decoded, source_spans, exhausted, _used_percent_decoding in _composed_decoded_diagnostic_tokens(
            value,
            max_depth=_DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH,
        ):
            decoded_variants = [(decoded, source_spans)]
            normalized_decoded, normalized_source_spans = _normalized_composed_diagnostic_tokens(
                decoded,
                source_spans,
            )
            if normalized_decoded != decoded:
                decoded_variants.append((normalized_decoded, normalized_source_spans))
            for decoded_variant, variant_source_spans in decoded_variants:
                spans.extend(
                    _mapped_sensitive_value_spans(
                        decoded_variant,
                        variant_source_spans,
                        retained_chars=retained_chars,
                        sensitive_values=decoded_needles,
                    )
                )
                canonical_decoded, canonical_source_spans = _canonical_nfd_diagnostic_tokens(
                    decoded_variant,
                    source_spans=variant_source_spans,
                )
                if canonical_decoded != decoded_variant:
                    spans.extend(
                        _mapped_sensitive_value_spans(
                            canonical_decoded,
                            canonical_source_spans,
                            retained_chars=retained_chars,
                            sensitive_values=canonical_decoded_needles,
                        )
                    )
            if exhausted:
                # The bounded diagnostic still has an encoded layer. It can
                # contain the configured value beyond both the decode and
                # look-ahead budgets, so retaining any prefix would
                # make that credential recoverable after further decoding.
                spans.append((0, retained_chars))
        # A literal '+' and a raw '%' followed by hex digits are ambiguous with
        # their encoded meanings. Preserve the permissive mixed spellings the
        # former matcher accepted, but only for bounded secrets: longer
        # ambiguous values make the caller fail closed before reaching here.
        scan_start_limit = min(retained_chars, len(value))
        for sensitive in json_sensitive_values:
            if "+" not in sensitive and "%" not in sensitive:
                continue
            if not (("+" in sensitive and "+" in value) or ("%" in sensitive and "%" in value)):
                # The one-pass token stream already covers encoded '+'/'%' and
                # form-encoded spaces. A legacy fallback can add a match only
                # when one of those ambiguous characters is also literal in the
                # source, so avoid rescanning every shared-prefix offset when it
                # is absent.
                continue
            for start in range(scan_start_limit):
                if value[start] not in {"%", sensitive[0]} and not (sensitive[0] == " " and value[start] == "+"):
                    continue
                encoded_end = _ambiguous_url_percent_sensitive_match_end(value, start, sensitive)
                if encoded_end is not None:
                    spans.append((start, min(encoded_end, retained_chars)))
    return spans


def _diagnostic_composed_structural_credential_spans(
    value: str,
    *,
    retained_chars: int,
) -> list[tuple[int, int]]:
    """Locate credential structures exposed by bounded composed decoding."""
    spans, _normalized_json_exhausted = _diagnostic_composed_structural_credential_scan(
        value,
        retained_chars=retained_chars,
    )
    return spans


def _diagnostic_composed_structural_credential_scan(
    value: str,
    *,
    retained_chars: int,
) -> tuple[list[tuple[int, int]], bool]:
    """Return composed spans and whether a normalized JSON-only path exhausted."""
    spans: list[tuple[int, int]] = []
    normalized_json_exhausted = False
    for decoded, source_spans, exhausted, used_percent_decoding in _composed_decoded_diagnostic_tokens(
        value,
        max_depth=_DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH,
    ):
        decoded_variants = []
        if used_percent_decoding:
            decoded_variants.append((decoded, source_spans))
        normalized_decoded, normalized_source_spans = _normalized_composed_diagnostic_tokens(
            decoded,
            source_spans,
        )
        normalized_path = normalized_decoded != decoded
        if normalized_path:
            # Raw JSON paths retain their syntax through the dedicated nested
            # JSON scanner. Scan a JSON-only composed path here only when
            # decoding exposes controls/whitespace whose normalization can
            # create a credential structure that the raw scanner cannot see.
            decoded_variants.append((normalized_decoded, normalized_source_spans))
        for decoded_variant, variant_source_spans in decoded_variants:
            decoded_retained_chars = sum(source_start < retained_chars for source_start, _ in variant_source_spans)
            decoded_spans = _diagnostic_direct_structural_credential_spans(
                decoded_variant,
                retained_chars=decoded_retained_chars,
            )
            spans.extend(
                _mapped_diagnostic_spans(
                    variant_source_spans,
                    decoded_spans,
                    retained_chars=retained_chars,
                )
            )
        if exhausted and (used_percent_decoding or normalized_path):
            # Unknown structural credentials can be hidden just as configured
            # values can. Once the composed-decoding budget is exhausted, fail
            # closed instead of emitting a reversibly encoded prefix.
            spans.append((0, retained_chars))
            normalized_json_exhausted = normalized_json_exhausted or (normalized_path and not used_percent_decoding)
    return spans, normalized_json_exhausted


def _diagnostic_direct_structural_credential_spans(
    value: str,
    *,
    retained_chars: int,
) -> list[tuple[int, int]]:
    """Locate structural credentials without decoding the surrounding layer."""
    return [
        (start, end)
        for _kind, start, end in _diagnostic_typed_direct_structural_credential_spans(
            value,
            retained_chars=retained_chars,
        )
    ]


def _diagnostic_typed_direct_structural_credential_spans(
    value: str,
    *,
    retained_chars: int,
) -> list[tuple[str, int, int]]:
    """Locate direct structural credentials and retain their scanner kind."""
    spans = [
        ("url_userinfo", start, end)
        for start, end in _diagnostic_url_userinfo_spans(value, retained_chars=retained_chars)
    ]
    spans.extend(
        ("json_url_userinfo", start, end)
        for start, end in _diagnostic_json_url_userinfo_spans(value, retained_chars=retained_chars)
    )
    spans.extend(
        ("semantic_json", start, end) for start, end in _diagnostic_semantic_json_credential_assignment_spans(value)
    )
    spans.extend(("authorization", start, end) for start, end in _diagnostic_authorization_assignment_spans(value))
    for match in _DIAGNOSTIC_SECRET_ASSIGNMENT_PATTERN.finditer(value):
        spans.append(
            (
                "assignment",
                match.start() + len(match.group("credential_leading_separators")),
                match.end(),
            )
        )
    spans.extend(("bearer", *match.span()) for match in _DIAGNOSTIC_BEARER_PATTERN.finditer(value))
    for match in _DIAGNOSTIC_CREDENTIALISH_TOKEN_PATTERN.finditer(value):
        replacement = _redact_opaque_credentialish_token(match)
        if replacement == match.group(0):
            continue
        preserved_prefix, _marker, _suffix = replacement.partition(_DIAGNOSTIC_REDACTED_CREDENTIAL)
        spans.append(("opaque_token", match.start() + len(preserved_prefix), match.end()))
    return [
        (kind, start, min(end, retained_chars)) for kind, start, end in spans if start < retained_chars and end > start
    ]


def _diagnostic_is_serialized_json(value: str) -> bool:
    """Recognize a complete bounded JSON container or string."""
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in {'"', "[", "{"}:
        return False
    try:
        parsed = json.loads(stripped)
    except (RecursionError, TypeError, ValueError):
        return False
    return isinstance(parsed, (dict, list, str))


def _redact_nested_diagnostic_layer(
    value: str,
    *,
    secrets: set[str],
    json_sensitive_values: set[str],
) -> str:
    """Redact one decoded JSON-string layer without normalizing its syntax."""
    retained_chars = len(value)
    spans = _diagnostic_sensitive_value_spans(
        value,
        retained_chars=retained_chars,
        secrets=secrets,
        json_sensitive_values=json_sensitive_values,
    )
    spans.extend(_diagnostic_url_userinfo_spans(value, retained_chars=retained_chars))
    spans.extend(_diagnostic_json_url_userinfo_spans(value, retained_chars=retained_chars))
    composed_spans, normalized_json_exhausted = _diagnostic_composed_structural_credential_scan(
        value,
        retained_chars=retained_chars,
    )
    if normalized_json_exhausted:
        # Nested JSON redaction has historically used the explicit credential
        # marker when its bounded decode budget is exhausted. Preserve that
        # fail-closed contract for the newly scanned normalized JSON path too.
        return _DIAGNOSTIC_REDACTED_CREDENTIAL
    spans.extend(composed_spans)
    redacted = _replace_diagnostic_spans(value, spans, marker="[redacted]")
    return _redact_semantic_json_credential_assignments(
        redacted,
        preserve_json_values=True,
    )


def _redact_plain_diagnostic_layer(
    value: str,
    *,
    secrets: set[str],
    json_sensitive_values: set[str],
) -> str:
    """Redact one plain diagnostic layer without recursively decoding it."""
    redacted = _redact_nested_diagnostic_layer(
        value,
        secrets=secrets,
        json_sensitive_values=json_sensitive_values,
    )
    redacted = _redact_authorization_assignments(redacted)
    redacted = _DIAGNOSTIC_SECRET_ASSIGNMENT_PATTERN.sub(_redact_secret_assignment, redacted)
    redacted = _DIAGNOSTIC_BEARER_PATTERN.sub("Bearer [redacted]", redacted)
    return _DIAGNOSTIC_CREDENTIALISH_TOKEN_PATTERN.sub(_redact_opaque_credentialish_token, redacted)


def _diagnostic_contains_serialized_json_string(value: str) -> bool:
    """Detect unresolved JSON containers nested inside quoted values."""
    cursor = 0
    while cursor < len(value):
        string_start = value.find('"', cursor)
        if string_start < 0:
            return False
        parsed = _decode_diagnostic_json_string_prefix(value, string_start)
        if parsed is None:
            cursor = string_start + 1
            continue
        decoded, _source_spans, string_end, complete = parsed
        if not complete:
            return True
        if _diagnostic_is_serialized_json(decoded):
            return True
        cursor = max(string_end, string_start + 1)
    return False


def _decode_diagnostic_json_escape_unit(value: str, cursor: int) -> tuple[str, int] | None:
    """Decode one bounded JSON string-content unit starting at ``cursor``."""
    character = value[cursor]
    if character != "\\":
        return character, cursor + 1
    if cursor + 1 >= len(value):
        return None
    escape_character = value[cursor + 1]
    if escape_character in _JSON_SIMPLE_ESCAPE_CHARACTERS:
        return _JSON_SIMPLE_ESCAPE_CHARACTERS[escape_character], cursor + 2
    escape_end = cursor + 6
    if escape_character != "u" or escape_end > len(value):
        return None
    hex_digits = value[cursor + 2 : escape_end]
    if any(digit not in _JSON_HEX_DIGITS for digit in hex_digits):
        return None
    first_code_unit = int(hex_digits, 16)
    source_end = escape_end
    decoded_character = chr(first_code_unit)
    if (
        0xD800 <= first_code_unit <= 0xDBFF
        and escape_end + 6 <= len(value)
        and value[escape_end : escape_end + 2] == "\\u"
    ):
        second_hex_digits = value[escape_end + 2 : escape_end + 6]
        if all(digit in _JSON_HEX_DIGITS for digit in second_hex_digits):
            second_code_unit = int(second_hex_digits, 16)
            if 0xDC00 <= second_code_unit <= 0xDFFF:
                decoded_character = chr(0x10000 + ((first_code_unit - 0xD800) << 10) + (second_code_unit - 0xDC00))
                source_end = escape_end + 6
    return decoded_character, source_end


def _decode_diagnostic_escaped_json_string(
    value: str,
    start: int,
) -> tuple[str | None, int, bool] | None:
    """Decode a JSON string token whose delimiters are escaped one layer."""
    if start >= len(value) or value[start] != "\\":
        return None
    opening_unit = _decode_diagnostic_json_escape_unit(value, start)
    if opening_unit is None or opening_unit[0] != '"':
        return None

    decoded_token = ['"']
    string_escape = False
    cursor = opening_unit[1]
    while cursor < len(value):
        source_start = cursor
        decoded_unit = _decode_diagnostic_json_escape_unit(value, cursor)
        if decoded_unit is None:
            break
        character, cursor = decoded_unit
        if ord(character) < 0x20 or (character == '"' and value[source_start] != "\\"):
            break
        decoded_token.append(character)
        if string_escape:
            string_escape = False
            continue
        if character == "\\":
            string_escape = True
            continue
        if character != '"':
            continue
        try:
            decoded = json.loads("".join(decoded_token))
        except (RecursionError, TypeError, ValueError):
            return None
        if isinstance(decoded, str):
            return decoded, cursor, True
        return None

    partial_token = "".join(decoded_token)
    parsed_prefix = _decode_diagnostic_json_string_prefix(partial_token, 0)
    decoded = parsed_prefix[0] if parsed_prefix is not None else None
    if decoded is None:
        # Preserve the valid prefix even when the provider ended with an
        # invalid escape rather than a merely incomplete one.
        final_escape = partial_token.rfind("\\")
        if final_escape > 0:
            try:
                candidate = json.loads(partial_token[:final_escape] + '"')
            except (RecursionError, TypeError, ValueError):
                candidate = None
            if isinstance(candidate, str):
                decoded = candidate
    return decoded if isinstance(decoded, str) else None, len(value), False


def _diagnostic_may_contain_incomplete_url_userinfo(value: str) -> bool:
    """Recognize URL authority text whose missing suffix could hide userinfo."""
    if _diagnostic_url_userinfo_spans(value, retained_chars=len(value)):
        return True
    for match in _DIAGNOSTIC_URL_SCHEME_PATTERN.finditer(value):
        authority = value[match.end() :]
        authority = re.split(r"[/?#\r\n]", authority, maxsplit=1)[0]
        if ":" in authority or "@" in authority:
            return True
    return False


def _encode_diagnostic_escaped_json_string(value: str) -> str:
    """Encode a string value as a JSON token escaped through one outer layer."""
    token = json.dumps(value, ensure_ascii=True)
    return json.dumps(token, ensure_ascii=True)[1:-1]


def _redact_escaped_json_assignments(
    value: str,
    *,
    secrets: set[str],
    json_sensitive_values: set[str],
    max_chars: int,
    depth: int,
) -> str:
    """Redact escaped JSON-like assignments even when their container is malformed."""
    replacements: list[tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(value):
        key_start = value.find("\\", cursor)
        if key_start < 0:
            break
        parsed_key = _decode_diagnostic_escaped_json_string(value, key_start)
        if parsed_key is None:
            cursor = key_start + 1
            continue
        decoded_key, key_end, key_complete = parsed_key
        if not key_complete or decoded_key is None:
            cursor = key_start + 1
            continue

        colon_start = key_end
        while colon_start < len(value) and value[colon_start].isspace():
            colon_start += 1
        if colon_start >= len(value):
            break
        colon_unit = _decode_diagnostic_json_escape_unit(value, colon_start)
        if colon_unit is None or colon_unit[0] != ":":
            cursor = key_start + 1
            continue
        if replacements and key_start < replacements[-1][1]:
            return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]
        value_start = colon_unit[1]
        while value_start < len(value) and value[value_start].isspace():
            value_start += 1

        sensitive_key = _DIAGNOSTIC_CREDENTIAL_KEY_PATTERN.fullmatch(decoded_key) is not None
        if value_start >= len(value):
            if sensitive_key:
                return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]
            break

        parsed_value = _decode_diagnostic_escaped_json_string(value, value_start)
        if parsed_value is None:
            if not sensitive_key:
                cursor = value_start + 1
                continue
            value_end = _diagnostic_json_value_end(value, value_start)
            if value_end <= value_start:
                return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]
            replacements.append(
                (
                    value_start,
                    value_end,
                    _encode_diagnostic_escaped_json_string(_DIAGNOSTIC_REDACTED_CREDENTIAL),
                )
            )
            cursor = value_end
            continue

        decoded_value, value_end, value_complete = parsed_value
        if not value_complete:
            if sensitive_key or (
                decoded_value is not None and _diagnostic_may_contain_incomplete_url_userinfo(decoded_value)
            ):
                return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]
            cursor = value_start + 1
            continue
        if decoded_value is None:
            cursor = value_end
            continue

        if sensitive_key:
            redacted_value = _DIAGNOSTIC_REDACTED_CREDENTIAL
        elif depth >= _DIAGNOSTIC_NESTED_JSON_MAX_DEPTH:
            redacted_value = _redact_plain_diagnostic_layer(
                decoded_value,
                secrets=secrets,
                json_sensitive_values=json_sensitive_values,
            )
            wrapped = f'"{redacted_value}"'
            parsed_layer = _decode_diagnostic_json_string_prefix(wrapped, 0)
            has_another_encoded_layer = (
                parsed_layer is not None
                and parsed_layer[3]
                and parsed_layer[2] == len(wrapped)
                and parsed_layer[0] != redacted_value
            )
            if (
                _diagnostic_is_serialized_json(redacted_value)
                or _diagnostic_contains_serialized_json_string(redacted_value)
                or has_another_encoded_layer
            ):
                return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]
        else:
            redacted_value = _redact_nested_plain_json_string(
                decoded_value,
                secrets=secrets,
                json_sensitive_values=json_sensitive_values,
                max_chars=max_chars,
                depth=depth + 1,
            )
        if redacted_value != decoded_value:
            replacements.append(
                (
                    value_start,
                    value_end,
                    _encode_diagnostic_escaped_json_string(redacted_value),
                )
            )
        closing_quote_start = (
            value_end - 6 if value[value_end - 6 : value_end].casefold() == "\\u0022" else value_end - 2
        )
        cursor = max(closing_quote_start, key_start + 1)

    if not replacements:
        return value
    if any(
        start < previous_end
        for (_, previous_end, _), (start, _, _) in zip(replacements, replacements[1:], strict=False)
    ):
        return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]
    pieces: list[str] = []
    source_cursor = 0
    for start, end, replacement in replacements:
        pieces.extend((value[source_cursor:start], replacement))
        source_cursor = end
    pieces.append(value[source_cursor:])
    result = "".join(pieces)
    if len(result) <= max_chars:
        return result
    return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]


def _redact_nested_plain_json_string(
    value: str,
    *,
    secrets: set[str],
    json_sensitive_values: set[str],
    max_chars: int,
    depth: int,
) -> str:
    """Redact a decoded JSON string while leaving its parent syntax intact."""
    redacted = _redact_plain_diagnostic_layer(
        value,
        secrets=secrets,
        json_sensitive_values=json_sensitive_values,
    )
    redacted = _redact_escaped_json_assignments(
        redacted,
        secrets=secrets,
        json_sensitive_values=json_sensitive_values,
        max_chars=max_chars,
        depth=depth,
    )
    if depth >= _DIAGNOSTIC_NESTED_JSON_MAX_DEPTH:
        if _diagnostic_contains_serialized_json_string(redacted):
            return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]
        # Do not emit a still-escaped layer once the fixed recursion budget is
        # exhausted. It may hide a credential key that only becomes semantic
        # after another decode. The enclosing caller will JSON-encode this
        # marker, so its parent remains syntactically valid.
        wrapped = f'"{redacted}"'
        parsed = _decode_diagnostic_json_string_prefix(wrapped, 0)
        if parsed is not None:
            decoded, _source_spans, string_end, complete = parsed
            if complete and string_end == len(wrapped) and decoded != redacted:
                return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]
        return redacted

    # Provider diagnostics sometimes embed a serialized JSON fragment inside
    # prose, so the decoded parent value is not itself a complete JSON value.
    # Scan any ordinary JSON strings in that prose before considering another
    # whole-string escape layer.
    redacted = _redact_nested_json_string_content(
        redacted,
        secrets=secrets,
        json_sensitive_values=json_sensitive_values,
        max_chars=max_chars,
        depth=depth + 1,
    )

    # A provider may include the fragment with its quotes still JSON-escaped,
    # for example ``wrapper: {\"access\\u005ftoken\":\"secret\"}``. Treat
    # the bounded value as JSON string content, recurse only when that decoding
    # exposes a different layer, and re-encode only if redaction was necessary
    # so benign text retains its original spelling.
    wrapped = f'"{redacted}"'
    parsed = _decode_diagnostic_json_string_prefix(wrapped, 0)
    if parsed is None:
        return redacted
    decoded, _source_spans, string_end, complete = parsed
    if not complete or string_end != len(wrapped) or decoded == redacted:
        return redacted

    decoded_redacted = _redact_nested_plain_json_string(
        decoded,
        secrets=secrets,
        json_sensitive_values=json_sensitive_values,
        max_chars=max_chars,
        depth=depth + 1,
    )
    if decoded_redacted == decoded:
        return redacted
    encoded = json.dumps(decoded_redacted, ensure_ascii=True)[1:-1]
    if len(encoded) <= max_chars:
        return encoded
    return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]


def _redact_nested_json_string_content(
    value: str,
    *,
    secrets: set[str],
    json_sensitive_values: set[str],
    max_chars: int,
    depth: int = 0,
) -> str:
    """Redact serialized JSON strings with fixed-depth, bounded decoding work."""
    replacements: list[tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(value):
        string_start = value.find('"', cursor)
        if string_start < 0:
            break
        parsed = _decode_diagnostic_json_string_prefix(value, string_start)
        if parsed is None:
            cursor = string_start + 1
            continue
        decoded, _source_spans, string_end, complete = parsed
        if not complete:
            # Reaching the end of the full scan budget means this may be only
            # a prefix of a serialized diagnostic. Fail closed instead of
            # exposing a multiply escaped credential across that boundary.
            # A short, merely unmatched quote in ordinary prose is harmless.
            if len(value) >= max_chars:
                return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]
            break

        if not _diagnostic_is_serialized_json(decoded):
            redacted = _redact_nested_plain_json_string(
                decoded,
                secrets=secrets,
                json_sensitive_values=json_sensitive_values,
                max_chars=max_chars,
                depth=depth,
            )
            if redacted != decoded:
                encoded = json.dumps(redacted, ensure_ascii=True)[1:-1]
                replacements.append((string_start + 1, string_end - 1, encoded))
            cursor = max(string_end, string_start + 1)
            continue

        if depth >= _DIAGNOSTIC_NESTED_JSON_MAX_DEPTH:
            redacted = _DIAGNOSTIC_REDACTED_CREDENTIAL
        else:
            redacted = _redact_nested_diagnostic_layer(
                decoded,
                secrets=secrets,
                json_sensitive_values=json_sensitive_values,
            )
            redacted = _redact_nested_json_string_content(
                redacted,
                secrets=secrets,
                json_sensitive_values=json_sensitive_values,
                max_chars=max_chars,
                depth=depth + 1,
            )

        if redacted != decoded:
            encoded = json.dumps(redacted, ensure_ascii=True)[1:-1]
            replacements.append((string_start + 1, string_end - 1, encoded))
        cursor = max(string_end, string_start + 1)

    if not replacements:
        return value
    pieces: list[str] = []
    source_cursor = 0
    for start, end, replacement in replacements:
        pieces.extend((value[source_cursor:start], replacement))
        source_cursor = end
    pieces.append(value[source_cursor:])
    result = "".join(pieces)
    if len(result) <= max_chars:
        return result
    return _DIAGNOSTIC_REDACTED_CREDENTIAL[:max_chars]


def _redact_opaque_credentialish_token(match: re.Match[str]) -> str:
    """Redact key-labelled opaque tokens without treating hyphenated prose as secret."""
    leading_hyphens = match.group("credential_token_leading_hyphens")
    token = match.group(0)[len(leading_hyphens) :]
    escaped_quote_prefix = ""
    if match.start() > 0 and match.string[match.start() - 1] == "\\" and token[:5].casefold() == "u0022":
        escaped_quote_prefix = token[:5]
        token = token[5:]
    if re.search(r"(?:^|-)(?:api-key|secret|token|credential)\Z", token, re.I):
        return leading_hyphens + escaped_quote_prefix + "[redacted credential]"
    segments = token.split("-")
    technical_segments = {
        index
        for index, segment in enumerate(segments)
        if _DIAGNOSTIC_TECHNICAL_VERSION_SEGMENT_PATTERN.fullmatch(segment)
    }
    prose_segment_count = sum(
        index not in technical_segments and segment.casefold() not in {"api", "credential", "key", "secret", "token"}
        for index, segment in enumerate(segments)
    )
    for index, segment in enumerate(segments):
        if index in technical_segments and prose_segment_count >= 2:
            continue
        has_lower = any(character.islower() for character in segment)
        has_upper = any(character.isupper() for character in segment)
        has_digit = any(character.isdigit() for character in segment)
        opaque_segment = (
            len(segment) >= 16
            or "~" in segment
            or (len(segment) >= 6 and has_digit and (has_lower or has_upper))
            or (len(segment) >= 8 and has_digit)
            or (len(segment) >= 6 and has_lower and has_upper)
        )
        if opaque_segment:
            return leading_hyphens + escaped_quote_prefix + "[redacted credential]"
    return leading_hyphens + escaped_quote_prefix + token


def _redact_secret_assignment(match: re.Match[str]) -> str:
    """Preserve punctuation preceding a redacted credential assignment."""
    return match.group("credential_leading_separators") + _DIAGNOSTIC_REDACTED_CREDENTIAL


def _mapped_sensitive_value_spans(
    decoded: str,
    source_spans: list[tuple[int, int]],
    *,
    retained_chars: int,
    sensitive_values: set[str],
) -> list[tuple[int, int]]:
    """Map substring matches in a decoded token stream back to source spans."""
    spans: list[tuple[int, int]] = []
    for sensitive in sensitive_values:
        if not sensitive:
            continue
        start = decoded.find(sensitive)
        while start >= 0:
            source_start = source_spans[start][0]
            if source_start >= retained_chars:
                break
            source_end = source_spans[start + len(sensitive) - 1][1]
            spans.append((source_start, min(source_end, retained_chars)))
            start = decoded.find(sensitive, start + 1)
    return spans


def _canonical_nfd_diagnostic_tokens(
    value: str,
    *,
    source_spans: list[tuple[int, int]] | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Canonically decompose bounded text while retaining safe source spans.

    Every normalized character in one canonical combining sequence maps to the
    sequence's full source range. That coarser mapping remains monotonic when
    NFD reorders combining marks, so a match cannot leave behind a reordered
    source character at either edge of the redacted span.
    """
    if source_spans is None:
        source_spans = [(index, index + 1) for index in range(len(value))]

    normalized: list[str] = []
    normalized_source_spans: list[tuple[int, int]] = []
    segment: list[tuple[str, int, tuple[int, int]]] = []

    def flush_segment() -> None:
        if not segment:
            return
        ordered_segment = sorted(segment, key=lambda token: token[1])
        segment_source_start = min(token[2][0] for token in segment)
        segment_source_end = max(token[2][1] for token in segment)
        normalized.extend(token[0] for token in ordered_segment)
        normalized_source_spans.extend((segment_source_start, segment_source_end) for _token in ordered_segment)
        segment.clear()

    for index, character in enumerate(value):
        source_span = source_spans[index]
        for decomposed_character in unicodedata.normalize("NFD", character):
            combining_class = unicodedata.combining(decomposed_character)
            if combining_class == 0:
                flush_segment()
            segment.append((decomposed_character, combining_class, source_span))
    flush_segment()
    return "".join(normalized), normalized_source_spans


def _mapped_diagnostic_spans(
    source_spans: list[tuple[int, int]],
    decoded_spans: Iterable[tuple[int, int]],
    *,
    retained_chars: int,
) -> list[tuple[int, int]]:
    """Map decoded structural spans back to the bounded source diagnostic."""
    spans: list[tuple[int, int]] = []
    for decoded_start, decoded_end in decoded_spans:
        if decoded_end <= decoded_start or decoded_start >= len(source_spans):
            continue
        source_start = source_spans[decoded_start][0]
        if source_start >= retained_chars:
            continue
        bounded_decoded_end = min(decoded_end, len(source_spans))
        source_end = source_spans[bounded_decoded_end - 1][1]
        spans.append((source_start, min(source_end, retained_chars)))
    return spans


def _mapped_typed_diagnostic_spans(
    source_spans: list[tuple[int, int]],
    decoded_spans: Iterable[tuple[str, int, int]],
    *,
    retained_chars: int,
) -> list[tuple[str, int, int]]:
    """Map typed decoded structural spans back to their source positions."""
    spans: list[tuple[str, int, int]] = []
    for kind, decoded_start, decoded_end in decoded_spans:
        mapped = _mapped_diagnostic_spans(
            source_spans,
            [(decoded_start, decoded_end)],
            retained_chars=retained_chars,
        )
        spans.extend((kind, start, end) for start, end in mapped)
    return spans


def _normalized_diagnostic_tokens(value: str) -> tuple[str, list[tuple[int, int]]]:
    """Normalize diagnostic whitespace while retaining source spans."""
    normalized: list[str] = []
    source_spans: list[tuple[int, int]] = []
    cursor = 0
    pending_whitespace_start: int | None = None
    while cursor < len(value):
        character = value[cursor]
        printable = character if character.isprintable() else " "
        if printable.isspace():
            if normalized and pending_whitespace_start is None:
                pending_whitespace_start = cursor
            cursor += 1
            continue
        if pending_whitespace_start is not None:
            normalized.append(" ")
            source_spans.append((pending_whitespace_start, cursor))
            pending_whitespace_start = None
        normalized.append(printable)
        source_spans.append((cursor, cursor + 1))
        cursor += 1
    return "".join(normalized), source_spans


def _normalized_composed_diagnostic_tokens(
    value: str,
    source_spans: list[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]]]:
    """Normalize one decoded layer while retaining its raw source spans."""
    normalized, decoded_source_spans = _normalized_diagnostic_tokens(value)
    if normalized == value:
        return value, source_spans
    return normalized, _composed_diagnostic_source_spans(decoded_source_spans, source_spans)


def _json_code_unit_spelling(value: str) -> str:
    """Represent non-BMP characters as the code units used by JSON ``\\u`` pairs."""
    pieces: list[str] = []
    for character in value:
        code_point = ord(character)
        if code_point <= 0xFFFF:
            pieces.append(character)
            continue
        offset = code_point - 0x10000
        pieces.extend((chr(0xD800 + (offset >> 10)), chr(0xDC00 + (offset & 0x3FF))))
    return "".join(pieces)


def _json_decoded_diagnostic_tokens(value: str) -> tuple[str, list[tuple[int, int]]]:
    """Decode one JSON-content layer while retaining each token's source span."""
    decoded: list[str] = []
    source_spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(value):
        source_end = cursor + 1
        decoded_character = value[cursor]
        if value[cursor] == "\\" and cursor + 1 < len(value):
            escape_character = value[cursor + 1]
            simple_escape = _JSON_SIMPLE_ESCAPE_CHARACTERS.get(escape_character)
            if simple_escape is not None:
                decoded_character = simple_escape
                source_end = cursor + 2
            elif escape_character == "u" and cursor + 6 <= len(value):
                hex_digits = value[cursor + 2 : cursor + 6]
                if all(character in _JSON_HEX_DIGITS for character in hex_digits):
                    decoded_character = chr(int(hex_digits, 16))
                    source_end = cursor + 6
        decoded.append(decoded_character)
        source_spans.append((cursor, source_end))
        cursor = source_end
    return "".join(decoded), source_spans


def _json_escaped_sensitive_max_chars(sensitive: str) -> int:
    """Bound the longest valid JSON-content spelling of ``sensitive``."""
    return sum(12 if ord(character) > 0xFFFF else 6 for character in sensitive)


def _percent_encoded_utf8_token(value: str, start: int) -> tuple[str, int] | None:
    """Decode one canonical UTF-8 percent token without consuming invalid text."""
    if start + 3 > len(value) or value[start] != "%":
        return None
    first_hex = value[start + 1 : start + 3]
    if len(first_hex) != 2 or any(character not in _JSON_HEX_DIGITS for character in first_hex):
        return None
    first_byte = int(first_hex, 16)
    if first_byte <= 0x7F:
        byte_count = 1
    elif 0xC2 <= first_byte <= 0xDF:
        byte_count = 2
    elif 0xE0 <= first_byte <= 0xEF:
        byte_count = 3
    elif 0xF0 <= first_byte <= 0xF4:
        byte_count = 4
    else:
        return None

    encoded_bytes = bytearray()
    cursor = start
    for _ in range(byte_count):
        if cursor + 3 > len(value) or value[cursor] != "%":
            return None
        encoded_hex = value[cursor + 1 : cursor + 3]
        if len(encoded_hex) != 2 or any(character not in _JSON_HEX_DIGITS for character in encoded_hex):
            return None
        encoded_bytes.append(int(encoded_hex, 16))
        cursor += 3
    try:
        decoded = bytes(encoded_bytes).decode("utf-8", errors="surrogatepass")
    except UnicodeDecodeError:
        return None
    if len(decoded) != 1:
        return None
    return decoded, cursor


def _percent_decoded_diagnostic_tokens(value: str) -> tuple[str, list[tuple[int, int]]]:
    """Decode one URL-percent layer while retaining each token's source span."""
    decoded: list[str] = []
    source_spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(value):
        percent_token = _percent_encoded_utf8_token(value, cursor)
        if percent_token is not None:
            decoded_character, source_end = percent_token
        elif value[cursor] == "+":
            decoded_character = " "
            source_end = cursor + 1
        else:
            decoded_character = value[cursor]
            source_end = cursor + 1
        decoded.append(decoded_character)
        source_spans.append((cursor, source_end))
        cursor = source_end
    return "".join(decoded), source_spans


def _composed_diagnostic_source_spans(
    decoded_source_spans: list[tuple[int, int]],
    parent_source_spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map one decoded layer's spans back through its parent to raw text."""
    return [(parent_source_spans[start][0], parent_source_spans[end - 1][1]) for start, end in decoded_source_spans]


def _composed_decoded_diagnostic_tokens(
    value: str,
    *,
    max_depth: int,
) -> Iterator[tuple[str, list[tuple[int, int]], bool, bool]]:
    """Decode bounded JSON/percent compositions while retaining raw spans.

    The first flag tells callers that a path still has an encoded layer after
    either the depth or linear-work budget is exhausted. The second records
    whether the path has performed percent decoding. Credential callers use
    these flags to fail closed without preempting the dedicated raw-JSON logic.
    """
    current_layers = [(value, [(index, index + 1) for index in range(len(value))], False)]
    decoders = (_json_decoded_diagnostic_tokens, _percent_decoded_diagnostic_tokens)
    remaining_work = max(1, len(value)) * _DIAGNOSTIC_COMPOSED_ENCODING_WORK_FACTOR
    for depth in range(max_depth):
        next_layers: list[tuple[str, list[tuple[int, int]], bool]] = []
        for parent, parent_source_spans, used_percent_decoding in current_layers:
            for decoder in decoders:
                if len(parent) > remaining_work:
                    yield parent, parent_source_spans, True, used_percent_decoding
                    return
                remaining_work -= len(parent)
                decoded, decoded_source_spans = decoder(parent)
                if decoded == parent:
                    continue
                source_spans = _composed_diagnostic_source_spans(
                    decoded_source_spans,
                    parent_source_spans,
                )
                at_depth_limit = depth + 1 >= max_depth
                exhausted = at_depth_limit and _diagnostic_has_remaining_encoded_token(decoded)
                decoded_percent = used_percent_decoding or decoder is _percent_decoded_diagnostic_tokens
                yield decoded, source_spans, exhausted, decoded_percent
                if not at_depth_limit:
                    next_layers.append((decoded, source_spans, decoded_percent))
        current_layers = next_layers


def _diagnostic_has_remaining_encoded_token(value: str) -> bool:
    """Return whether another JSON/percent layer remains at the depth limit."""
    if _diagnostic_has_percent_decoding_token(value):
        return True
    cursor = 0
    while cursor < len(value):
        character = value[cursor]
        if character == "\\" and cursor + 1 < len(value):
            escape_character = value[cursor + 1]
            if escape_character in _JSON_SIMPLE_ESCAPE_CHARACTERS:
                return True
            if escape_character == "u" and cursor + 6 <= len(value):
                hex_digits = value[cursor + 2 : cursor + 6]
                if all(digit in _JSON_HEX_DIGITS for digit in hex_digits):
                    return True
        cursor += 1
    return False


def _diagnostic_has_percent_decoding_token(value: str) -> bool:
    """Return whether another URL-percent/form decoding pass can change text."""
    cursor = 0
    while cursor < len(value):
        if value[cursor] == "+":
            # A plus exposed only at the final decoding depth can still be a
            # form-encoded space. Treat it as an unresolved layer rather than
            # leak a credential whose spelling depends on that last decode.
            return True
        if value[cursor] == "%" and _percent_encoded_utf8_token(value, cursor) is not None:
            return True
        cursor += 1
    return False


def _composed_encoded_sensitive_max_chars(sensitive: str) -> int:
    """Bound a sensitive value through the composed-decoding scan budget."""
    one_layer_max = max(
        _json_escaped_sensitive_max_chars(sensitive),
        _url_percent_encoded_sensitive_max_chars(sensitive),
    )
    # Both one-layer spellings are ASCII. Every further JSON layer can spell
    # each source character as a six-character Unicode escape, while percent
    # encoding expands it by at most three. Scale once for every remaining
    # layer in the fixed-depth traversal so a fully escaped value cannot cross
    # the retained-prefix boundary beyond our look-ahead.
    remaining_layers = max(_DIAGNOSTIC_COMPOSED_ENCODING_MAX_DEPTH - 1, 0)
    return one_layer_max * (6**remaining_layers)


def _ambiguous_url_percent_sensitive_match_end(value: str, start: int, sensitive: str) -> int | None:
    """Match permissive mixed spellings containing a literal '+' or '%'."""
    cursor = start
    for expected_character in sensitive:
        try:
            encoded_bytes = expected_character.encode("utf-8", errors="surrogatepass")
        except (UnicodeEncodeError, ValueError):
            return None
        encoded_cursor = cursor
        for expected_byte in encoded_bytes:
            encoded_hex = value[encoded_cursor + 1 : encoded_cursor + 3]
            if (
                encoded_cursor + 3 > len(value)
                or value[encoded_cursor] != "%"
                or any(character not in _JSON_HEX_DIGITS for character in encoded_hex)
                or int(encoded_hex, 16) != expected_byte
            ):
                break
            encoded_cursor += 3
        else:
            cursor = encoded_cursor
            continue
        if cursor < len(value) and value[cursor] == expected_character:
            cursor += 1
            continue
        if expected_character == " " and cursor < len(value) and value[cursor] == "+":
            cursor += 1
            continue
        return None
    return cursor


def _url_percent_encoded_sensitive_max_chars(sensitive: str) -> int:
    """Bound a fully UTF-8 URL-percent-encoded spelling of ``sensitive``."""
    try:
        return sum(3 * len(character.encode("utf-8", errors="surrogatepass")) for character in sensitive)
    except (UnicodeEncodeError, ValueError):
        return len(sensitive)


def _diagnostic_type_only_fallback(value: object, *, max_chars: int) -> str:
    """Describe an unsupported object without invoking its rendering hooks."""
    try:
        type_name = type.__getattribute__(type(value), "__name__")
    except BaseException:
        type_name = "object"
    if not isinstance(type_name, str):
        type_name = "object"
    prefix = "<unprintable "
    suffix = ">"
    name_budget = max(0, max_chars - len(prefix) - len(suffix))
    bounded_name = str.__getitem__(type_name, slice(0, name_budget))
    return (prefix + bounded_name + suffix)[:max_chars]


def _bounded_trusted_scalar_render(value: object, *, max_chars: int) -> str | None:
    """Render only values whose exact/base implementation cannot call user code."""
    if isinstance(value, str):
        return str.__getitem__(value, slice(0, max_chars))
    if isinstance(value, bytes):
        prefix = bytes.__getitem__(value, slice(0, max_chars))
        return repr(prefix)[:max_chars]
    if isinstance(value, bytearray):
        prefix = bytearray.__getitem__(value, slice(0, max_chars))
        return repr(prefix)[:max_chars]

    value_type = type(value)
    if value is None:
        return "None"
    if value_type is bool:
        return "True" if value else "False"
    if value_type is int:
        # Decimal conversion is linear in the number of digits. Refuse values
        # that cannot fit within a small constant factor of the retained text.
        if value.bit_length() > max_chars * 4:
            return _diagnostic_type_only_fallback(value, max_chars=max_chars)
        return str(value)[:max_chars]
    if value_type is float:
        return str(value)[:max_chars]
    if value_type is complex:
        return str(value)[:max_chars]
    if value_type is memoryview:
        try:
            byte_view = value.cast("B")
            prefix = byte_view[:max_chars].tobytes()
        except (TypeError, ValueError):
            return _diagnostic_type_only_fallback(value, max_chars=max_chars)
        return f"memoryview({prefix!r})"[:max_chars]
    return None


def _bounded_exception_render(value: BaseException, *, max_chars: int) -> str:
    """Render bounded trusted exception arguments without calling ``__str__``."""
    try:
        args = BaseException.__dict__["args"].__get__(value, BaseException)
    except BaseException:
        return _diagnostic_type_only_fallback(value, max_chars=max_chars)
    if type(args) is not tuple or not args:
        return _diagnostic_type_only_fallback(value, max_chars=max_chars)

    pieces: list[str] = []
    rendered_chars = 0
    for argument in args:
        separator = ", " if pieces else ""
        remaining = max_chars - rendered_chars - len(separator)
        if remaining <= 0:
            break
        try:
            rendered_argument = _bounded_trusted_scalar_render(argument, max_chars=remaining)
        except BaseException:
            rendered_argument = None
        if rendered_argument is None:
            rendered_argument = _diagnostic_type_only_fallback(argument, max_chars=remaining)
        pieces.extend((separator, rendered_argument))
        rendered_chars += len(separator) + len(rendered_argument)
    return "".join(pieces)[:max_chars]


def _bounded_diagnostic_render(value: object, *, max_chars: int) -> str:
    """Render bounded trusted values without invoking arbitrary conversion."""
    try:
        rendered = _bounded_trusted_scalar_render(value, max_chars=max_chars)
    except BaseException:
        return _diagnostic_type_only_fallback(value, max_chars=max_chars)
    if rendered is not None:
        return rendered
    if isinstance(value, BaseException):
        return _bounded_exception_render(value, max_chars=max_chars)
    return _diagnostic_type_only_fallback(value, max_chars=max_chars)


def _collision_safe_diagnostic_marker(
    sensitive_values: Iterable[str],
    *,
    max_chars: int,
) -> str:
    """Return a bounded marker that contains none of the supplied secrets."""
    values = tuple(value for value in sensitive_values if value)
    for marker in (_DIAGNOSTIC_REDACTED_CREDENTIAL, "[redacted]", "█"):
        if len(marker) <= max_chars and not _diagnostic_contains_canonical_sensitive_value(marker, values):
            return marker
    return ""


def _diagnostic_contains_canonical_sensitive_value(value: str, sensitive_values: Iterable[str]) -> bool:
    """Return whether bounded text contains a canonically equivalent secret."""
    canonical_value = unicodedata.normalize("NFD", value)
    return any(
        unicodedata.normalize("NFD", sensitive) in canonical_value for sensitive in sensitive_values if sensitive
    )


def safe_diagnostic_text(
    value: object,
    *,
    max_chars: int = DEFAULT_DIAGNOSTIC_MAX_CHARS,
    sensitive_values: Iterable[object] = (),
) -> str:
    """Bound and neutralize externally controlled text before logs or state."""
    if max_chars < 1:
        return ""

    scan_chars = max_chars * 4
    secrets: set[str] = set()
    json_sensitive_values: set[str] = set()
    marker_sensitive_values: set[str] = set()
    force_generic_diagnostic = False
    oversized_sensitive_may_shrink = False
    for raw_sensitive in sensitive_values:
        if not isinstance(raw_sensitive, str):
            continue
        sensitive_length = str.__len__(raw_sensitive)
        if sensitive_length <= len(_DIAGNOSTIC_REDACTED_CREDENTIAL):
            marker_sensitive = str.__getitem__(raw_sensitive, slice(0, sensitive_length)).strip()
            if marker_sensitive:
                marker_sensitive_values.add(marker_sensitive)
        # A value longer than the bounded diagnostic cannot be retained and
        # matched in full. A generic diagnostic prevents either our
        # scan or final truncation from disclosing an arbitrary credential
        # prefix, without scaling work or memory with the credential length.
        # Continue through the iterable so a later short secret cannot appear
        # inside the marker selected for this early fail-closed result.
        if sensitive_length > max_chars:
            force_generic_diagnostic = True
            # Avoid stripping an arbitrarily large value merely to choose a
            # marker. If edge whitespace could reduce it to a short marker
            # substring, empty output is the only bounded fail-closed result.
            oversized_sensitive_may_shrink = (
                bool(
                    sensitive_length
                    and (str.__getitem__(raw_sensitive, 0).isspace() or str.__getitem__(raw_sensitive, -1).isspace())
                )
                or oversized_sensitive_may_shrink
            )
            continue
        bounded_sensitive = str.__getitem__(raw_sensitive, slice(0, max_chars))
        if sensitive := bounded_sensitive.strip():
            if len(sensitive) > _DIAGNOSTIC_AMBIGUOUS_PERCENT_SECRET_MAX_CHARS and any(
                character in sensitive for character in "+%"
            ):
                force_generic_diagnostic = True
                continue
            # Exceptions and container diagnostics commonly render values with
            # ``repr`` rather than copying them literally.  Search those bounded
            # escaped forms as well: otherwise a credential containing a
            # backslash or control character can evade the literal match while
            # remaining fully recoverable in a persisted diagnostic.
            # Normalize only the already-bounded value. Canonically equivalent
            # provider spellings (for example NFC credentials rendered as NFD)
            # then flow through the same bounded literal, escaped, composed,
            # overlap, and final-collision checks as the configured spelling.
            canonical_sensitive_values = {
                sensitive,
                unicodedata.normalize("NFC", sensitive),
                unicodedata.normalize("NFD", sensitive),
            }
            for canonical_sensitive in canonical_sensitive_values:
                secrets.add(canonical_sensitive)
                json_sensitive_values.add(canonical_sensitive)
                for rendered_sensitive in (repr(canonical_sensitive), ascii(canonical_sensitive)):
                    secrets.add(rendered_sensitive)
                    if (
                        len(rendered_sensitive) >= 2
                        and rendered_sensitive[0] in {"'", '"'}
                        and rendered_sensitive[-1] == rendered_sensitive[0]
                    ):
                        secrets.add(rendered_sensitive[1:-1])

    if force_generic_diagnostic:
        if oversized_sensitive_may_shrink:
            return ""
        return _collision_safe_diagnostic_marker(
            marker_sensitive_values,
            max_chars=max_chars,
        )

    # Do not scan or copy an arbitrarily large provider response merely to
    # construct a diagnostic. Read a bounded credential-sized overlap beyond
    # the retained prefix so a secret crossing the scan boundary is still seen
    # in full. Only characters from the original prefix are emitted; the
    # overlap exists solely for matching and therefore cannot create a new
    # truncation boundary that leaks another credential prefix.
    encoded_lengths = [len(secret) for secret in secrets]
    encoded_lengths.extend(_composed_encoded_sensitive_max_chars(sensitive) for sensitive in json_sensitive_values)
    overlap_chars = max(encoded_lengths, default=1) - 1
    render_chars = scan_chars + max(overlap_chars, _DIAGNOSTIC_URL_USERINFO_OVERLAP_CHARS)
    scanned = _bounded_diagnostic_render(value, max_chars=render_chars)
    serialized_json = _diagnostic_is_serialized_json(scanned)
    scanned = _redact_nested_json_string_content(
        scanned,
        secrets=secrets,
        json_sensitive_values=json_sensitive_values,
        max_chars=render_chars,
    )
    candidate = scanned[:scan_chars]
    redacted_spans = _diagnostic_sensitive_value_spans(
        scanned,
        retained_chars=scan_chars,
        secrets=secrets,
        json_sensitive_values=json_sensitive_values,
    )
    redacted_spans.extend(_diagnostic_url_userinfo_spans(scanned, retained_chars=scan_chars))
    redacted_spans.extend(_diagnostic_json_url_userinfo_spans(scanned, retained_chars=scan_chars))
    redacted_spans.extend(
        _diagnostic_composed_structural_credential_spans(
            scanned,
            retained_chars=scan_chars,
        )
    )
    normalized_scan, normalized_source_spans = _normalized_diagnostic_tokens(scanned)
    if normalized_scan != scanned:
        normalized_scan_chars = len(normalized_scan)
        normalized_sensitive_spans = _diagnostic_sensitive_value_spans(
            normalized_scan,
            retained_chars=normalized_scan_chars,
            secrets=secrets,
            json_sensitive_values=json_sensitive_values,
        )
        normalized_composed_spans = _diagnostic_composed_structural_credential_spans(
            normalized_scan,
            retained_chars=normalized_scan_chars,
        )
        redacted_spans.extend(
            _mapped_diagnostic_spans(
                normalized_source_spans,
                [*normalized_sensitive_spans, *normalized_composed_spans],
                retained_chars=scan_chars,
            )
        )

        candidate_direct_spans = _diagnostic_typed_direct_structural_credential_spans(
            candidate,
            retained_chars=len(candidate),
        )
        raw_lookahead_spans = _diagnostic_typed_direct_structural_credential_spans(
            scanned,
            retained_chars=scan_chars,
        )
        raw_handled_spans = [
            *candidate_direct_spans,
            *(span for span in raw_lookahead_spans if span[0] in {"url_userinfo", "json_url_userinfo"}),
        ]
        normalized_direct_spans = _diagnostic_typed_direct_structural_credential_spans(
            normalized_scan,
            retained_chars=normalized_scan_chars,
        )
        mapped_normalized_direct_spans = _mapped_typed_diagnostic_spans(
            normalized_source_spans,
            normalized_direct_spans,
            retained_chars=scan_chars,
        )
        redacted_spans.extend(
            (start, end)
            for kind, start, end in mapped_normalized_direct_spans
            if not any(
                raw_kind == kind
                and raw_start == start
                and raw_end > raw_start
                and not (
                    kind == "authorization"
                    and raw_end < end
                    and _diagnostic_authorization_span_ends_with_bare_scheme(
                        scanned,
                        start=raw_start,
                        end=raw_end,
                    )
                )
                for raw_kind, raw_start, raw_end in raw_handled_spans
            )
        )

    candidate = _replace_diagnostic_spans(candidate, redacted_spans, marker="[redacted]")
    candidate = _redact_semantic_json_credential_assignments(candidate)
    candidate = _redact_authorization_assignments(candidate)
    if not serialized_json and candidate != _DIAGNOSTIC_REDACTED_CREDENTIAL[:scan_chars]:
        candidate = _redact_nested_plain_json_string(
            candidate,
            secrets=secrets,
            json_sensitive_values=json_sensitive_values,
            max_chars=scan_chars,
            depth=0,
        )

    printable = "".join(character if character.isprintable() else " " for character in candidate)
    normalized = " ".join(printable.split())
    if normalized != candidate:
        normalized_serialized_json = _diagnostic_is_serialized_json(normalized)
        normalized_chars = len(normalized)
        normalized = _redact_nested_json_string_content(
            normalized,
            secrets=secrets,
            json_sensitive_values=json_sensitive_values,
            max_chars=normalized_chars,
        )
        normalized_chars = len(normalized)
        normalized_spans = _diagnostic_sensitive_value_spans(
            normalized,
            retained_chars=normalized_chars,
            secrets=secrets,
            json_sensitive_values=json_sensitive_values,
        )
        normalized_spans.extend(_diagnostic_url_userinfo_spans(normalized, retained_chars=normalized_chars))
        normalized_spans.extend(_diagnostic_json_url_userinfo_spans(normalized, retained_chars=normalized_chars))
        normalized_spans.extend(
            _diagnostic_composed_structural_credential_spans(
                normalized,
                retained_chars=normalized_chars,
            )
        )
        normalized = _replace_diagnostic_spans(normalized, normalized_spans, marker="[redacted]")
        normalized = _redact_semantic_json_credential_assignments(normalized)
        normalized = _redact_authorization_assignments(normalized)
        if not normalized_serialized_json and normalized != _DIAGNOSTIC_REDACTED_CREDENTIAL[:normalized_chars]:
            normalized = _redact_nested_plain_json_string(
                normalized,
                secrets=secrets,
                json_sensitive_values=json_sensitive_values,
                max_chars=normalized_chars,
                depth=0,
            )
    normalized = _DIAGNOSTIC_SECRET_ASSIGNMENT_PATTERN.sub(_redact_secret_assignment, normalized)
    normalized = _DIAGNOSTIC_BEARER_PATTERN.sub("Bearer [redacted]", normalized)
    normalized = _DIAGNOSTIC_CREDENTIALISH_TOKEN_PATTERN.sub(_redact_opaque_credentialish_token, normalized)
    if len(normalized) > max_chars:
        normalized = normalized[: max_chars - 1].rstrip() + "…"
    # Span replacement and truncation occur after the last structural scan and
    # may introduce a configured value through a marker or a new join boundary.
    if _diagnostic_contains_canonical_sensitive_value(normalized, json_sensitive_values):
        return _collision_safe_diagnostic_marker(
            json_sensitive_values,
            max_chars=max_chars,
        )
    return normalized


def format_message(value: Any) -> str:
    if _is_empty(value):
        return ""
    message = safe_diagnostic_text(value, max_chars=4_096)
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
    except OverflowError:
        return ""
    except (TypeError, ValueError):
        try:
            number = float(value)
        except OverflowError:
            return ""
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(number):
            return ""
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
    if not math.isfinite(seconds):
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
    except OverflowError:
        return ""
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
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
    if _is_empty(value):
        return False
    try:
        return int(value) != 0
    except OverflowError:
        return False
    except (TypeError, ValueError):
        try:
            number = float(value)
        except OverflowError:
            return False
        except (TypeError, ValueError):
            try:
                return bool(value)
            except (TypeError, ValueError):
                return False
        if not math.isfinite(number):
            return False
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
