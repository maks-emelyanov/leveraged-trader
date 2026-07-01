# Leveraged Trader

Leveraged Trader is a research and paper-trading workflow for RSI-based leveraged ETF/ETN strategies. It builds a current leveraged product universe, optimizes simple buy/sell rules against daily market data, writes recommendation reports, and queues guarded Alpaca paper-trading buy orders by default.

This project is intended for research and paper trading. It is not financial advice, and it should not be connected to live trading without additional review, testing, and risk controls.

## Features

- Discovers current long leveraged ETFs/ETNs from Nasdaq ETF definitions plus best-effort issuer and ETN pages; source health distinguishes fetch/parser failures from healthy zero-match pages, and issuer-only products are filtered only when both active-listing sources load successfully.
- Writes audit-only source checks for exchange directories, third-party ETF directories, and SEC EDGAR registry review.
- Infers an RSI signal symbol from each leveraged ETF name, with curated proxy mappings,
  explicit self-RSI fallbacks, and a review table for unresolved mappings.
- Downloads daily Yahoo Finance OHLCV data, with optional Tradier fallback for skipped symbols.
- Optimizes RSI buy thresholds and profit-target sell multiples.
- Uses a NumPy/Numba-backed optimization loop for the parameter grid.
- Downloads asset workflows concurrently with async orchestration while serializing shared SQLite strategy-state updates.
- Renders width-aware terminal progress and tables with semantic status coloring.
- Persists strategy state in SQLite for resumable updates, with transactional cross-process invalidation safety.
- Writes buy and sell recommendation reports.
- Submits guarded, budget-capped whole-share Alpaca paper buy limit orders by default.
- Atomically claims a durable Alpaca buy intent before submission, preventing concurrent workers from submitting or closing the same client order ID.
- Submits and renews managed Alpaca GTC limit sells from actual fill price times the original sell multiple, including partial buy fills.
- Guards Alpaca buys against already-held symbols, active managed positions, and open buy or sell orders.

## Quickstart

```bash
uv sync
cp .env.example .env
uv run leveraged-trader --help
```

Edit `.env` with Alpaca paper credentials before running the default Alpaca submission workflow.

```env
ALPACA_API_KEY_ID=your_alpaca_paper_api_key_id
ALPACA_API_SECRET_KEY=your_alpaca_paper_api_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_BUY_LIMIT_BUFFER_BPS=500
ALPACA_GTC_SELL_RENEWAL_ENABLED=true
ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION=7
TRADIER_ACCESS_TOKEN=your_tradier_access_token
```

The CLI also supports `--alpaca-timeout-seconds` for request timeout tuning (default: `30`),
`--alpaca-gtc-sell-renewal-days-before-expiration` for managed GTC sell renewal timing (default: `7`),
`--tradier-timeout-seconds` for market-data fallback timeout tuning (default: `30`),
`--workflow-concurrency` for asset-level concurrency tuning (default: `4`), and `--no-color`
for plain terminal output. The Tradier fallback flag is enabled by default, but fallback requests
are only usable when a non-placeholder Tradier token is configured.

Run an update:

```bash
uv run leveraged-trader --mode update
```

Rebuild all cached state:

```bash
uv run leveraged-trader --mode rebuild
```

Run the paper-trading workflow during the premarket window to submit prior-session signals for that day's open and reconcile managed GTC limit sells:

```bash
uv run leveraged-trader
```

Generate reports without submitting paper orders:

```bash
uv run leveraged-trader --no-alpaca-submit-buy-orders --no-alpaca-submit-sell-orders
```

Show all options:

```bash
uv run leveraged-trader --help
```

## Recommended Operating Schedule

This cadence describes the mechanics of the workflow; it is not investment advice. The workflow
intentionally excludes the current US daily bar, so live buy submissions are designed for the
premarket after the signal day has settled.

- Premarket before the regular open, for example 8:00-9:20 a.m. ET: run `uv run leveraged-trader`.
  This reconciles existing managed positions first, refreshes settled histories, submits eligible
  prior-session buy signals as regular-session day limit buys, then reconciles again if any managed
  buy was submitted or recovered.
- After Alpaca shows a managed buy filled, run
  `uv run leveraged-trader --no-alpaca-submit-buy-orders` to attach or renew managed GTC limit sells
  without placing new buy orders. This is useful shortly after the open and again later if a day
  limit buy fills later in the session.
- If a limit buy never fills, no sell is submitted. A later reconciliation records the terminal buy
  status and closes the managed intent without a position.
- Extra reconciliation runs are intended to be idempotent: deterministic client order IDs, active
  managed-position checks, and live open-order checks protect against duplicate managed buys or sells.

## CLI Reference

Common options:

- `--mode {update,rebuild}`: resume from SQLite state (`update`) or recompute from scratch (`rebuild`).
- `--db PATH`: SQLite state file path (default: `strategy_state.sqlite`).
- `--output-dir DIR`: output directory for generated CSV files (default: `outputs`).
- `--alpaca-submit-buy-orders / --no-alpaca-submit-buy-orders`: enable or skip buy order submission (default: enabled).
- `--alpaca-submit-sell-orders / --no-alpaca-submit-sell-orders`: enable or skip managed limit sell reconciliation/submission (default: enabled).
- `--alpaca-api-key-id VALUE`: override `ALPACA_API_KEY_ID`.
- `--alpaca-api-secret-key VALUE`: override `ALPACA_API_SECRET_KEY`.
- `--alpaca-base-url URL`: override `ALPACA_BASE_URL` (defaults to Alpaca paper endpoint).
- `--alpaca-timeout-seconds INT`: Alpaca request timeout in seconds (default: `30`).
- `--alpaca-buy-limit-buffer-bps FLOAT`: price buffer for whole-share day buy limits in basis points (default: `500`).
- `--alpaca-gtc-sell-renewal / --no-alpaca-gtc-sell-renewal`: renew managed Alpaca GTC sells before expiration.
- `--alpaca-gtc-sell-renewal-days-before-expiration INT`: renewal window for managed GTC sells (default: `7`).
- `--tradier-fallback / --no-tradier-fallback`: enable or skip Tradier fallback for Yahoo-skipped symbols.
- `--tradier-access-token VALUE`: override `TRADIER_ACCESS_TOKEN`, `TRADIER_API_TOKEN`, or `TRADIER_TOKEN`.
- `--tradier-base-url URL`: override `TRADIER_BASE_URL` (defaults to Tradier production `/v1`).
- `--tradier-timeout-seconds INT`: Tradier request timeout in seconds (default: `30`).
- `--workflow-concurrency INT`: maximum number of assets processed concurrently (default: `4`; use `1` for serial behavior).
- `--require-workflow-source-success / --no-require-workflow-source-success`: fail a universe run after recording source health if an issuer or ETN source fetch or parser failed; a successfully parsed zero-match page remains healthy (default: disabled).
- `--no-color`: disable colored terminal output.

## Outputs

CSV reports are written to `outputs/` by default:

- `best_equity_curves.csv`
- `optimization_summary.csv`
- `buy_signals.csv`
- `eligible_buy_signals.csv`
- `sell_signals.csv`
- `managed_positions.csv`
- `alpaca_realized_pnl.csv`
- `alpaca_reconciliation_results.csv`
- `alpaca_order_results.csv`
- `alpaca_sell_order_results.csv`

Use `--output-dir` to choose a different location:

```bash
uv run leveraged-trader --output-dir outputs/dev
```

The SQLite state database defaults to `strategy_state.sqlite`; use `--db` to override it. Universe
generation also persists `nasdaq_etf_universe`, `universe_audit_rows`,
`universe_audit_missing_candidates`, `universe_audit_source_status`,
`universe_workflow_source_status`, `universe_active_listing_source_status`, and
`universe_rsi_mapping_review` tables for source and RSI-mapping review. Long leveraged rows whose RSI
symbol cannot be mapped confidently are excluded from the executable workflow and saved to
`universe_rsi_mapping_review`; curated proxy mappings and explicit self-RSI fallbacks remain
executable and are annotated in `nasdaq_etf_universe`. A universe discovery or active listing source
failure, or a successful response that cannot be parsed, leaves the run marked as degraded in terminal
output. A successfully parsed source with zero leveraged matches remains healthy; use
`--require-workflow-source-success` when a partial universe caused by a fetch or parser failure is not
acceptable.

Terminal output is intentionally compact: concurrent asset work is shown as aggregate progress, then
the final asset summary is sorted by workflow index. The terminal Best Sharpe table shows only
strategies with at least two executed trades and Sharpe of 1.0 or greater; `optimization_summary.csv`
retains the full per-asset summary. CSV files retain full order IDs and detail, while terminal Alpaca
tables show chronological display IDs that preserve closed-position gaps, plus the most useful fields
with wrapped messages. A final workflow footer reports total elapsed time and ends with a divider for
appended logs. Redirected or cron-driven non-terminal output defaults to a 156-column layout so log
tables stay readable, while interactive terminal output uses the terminal's current width.

## Environment Variables

Supported environment variables:

- `ALPACA_API_KEY_ID`
- `ALPACA_API_SECRET_KEY`
- `ALPACA_BASE_URL`
- `ALPACA_BUY_LIMIT_BUFFER_BPS`
- `ALPACA_GTC_SELL_RENEWAL_ENABLED`
- `ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION`
- `TRADIER_ACCESS_TOKEN`
- `TRADIER_API_TOKEN` (fallback alias for `TRADIER_ACCESS_TOKEN`)
- `TRADIER_TOKEN` (fallback alias for `TRADIER_ACCESS_TOKEN`)
- `TRADIER_BASE_URL`
- `TRADIER_FALLBACK_ENABLED`
- `TRADIER_TIMEOUT_SECONDS`

`ALPACA_BATCH_CASH_FRACTION` is intentionally no longer supported. Buy sizing is dynamic, so remove that legacy key from `.env` or the CLI will fail fast instead of silently ignoring it.

If your environment has not installed project entry points yet, use the module entry point:

```bash
uv run python -m leveraged_trader --help
```

## Alpaca Safety

Alpaca submission is enabled by default. Use `--no-alpaca-submit-buy-orders` or `--no-alpaca-submit-sell-orders` to skip one or both submission workflows.

Buy orders:

- Reserve `min(number_of_eligible_buy_signals * 0.05, 0.50)` of current Alpaca paper account cash across all submitted buys.
- Split the dynamic batch budget equally across recommendations that pass all live preflight checks (including a quote) and size integer whole-share quantities.
- Use regular-session day limit orders with a configurable price buffer; a gap beyond the limit safely leaves the order unfilled.
- Serialize limit prices at Alpaca's valid tick: four decimals below `$1` and two at or above `$1`; buy caps round down and sell targets round up.
- Are skipped if the allocated batch budget is below one whole share at the protected limit price.
- Are skipped if the symbol is already held.
- Are skipped if the symbol already has an active managed position.
- Are skipped if an open buy or sell order already exists for the symbol.
- Are submitted only during Alpaca's premarket window, and only when the signal date is the immediately preceding trading session; intraday, after-close, and stale signals are deferred.
- Use `extended_hours=false` and `time_in_force=day`.
- Are persisted with the buy RSI and sell multiple that were selected when the buy was submitted.
- Atomically claim an intent before the request is sent; only that claimant may submit the deterministic client order ID. A timeout, duplicate-ID response, or other ambiguous broker result is recovered by client order ID when possible. A transient broker `404` remains blocked during a short visibility lease before the intent can be closed, preventing another workflow from orphaning an in-flight accepted order. Once that lease expires, only a closed `submission_not_found` record with no broker order ID may be atomically reclaimed for a retry attempt; rejected and other failed submissions remain final.

Managed sell orders:

- Are reconciled from persisted managed buy records, not from the latest optimized sell signal.
- Use the actual Alpaca filled average buy price times the original sell multiple.
- Create a protective GTC sell for confirmed whole shares even while the parent buy remains partially filled, and replace it if later buy fills change the covered quantity or average-fill target—even if the prior partial-fill sell already completed.
- Sell the remaining managed quantity with a GTC limit order; cumulative partial fills remain active until the full buy quantity is closed.
- Persist the deterministic sell client order ID before broker submission. Ambiguous sell POST outcomes are recovered by client order ID on later reconciliation, and matching open `rsi-exit-...` orders can be reattached to the managed row.
- Reconcile immediately after each submitted buy batch, so fills can receive their managed sell in the same workflow run.
- Renew active GTC sells before Alpaca's aged-order expiration, using the remaining managed quantity and frozen target price. The renewal-cancel intent is persisted before requesting cancellation, so a timeout can still be completed after Alpaca later reports the order canceled.
- Resubmit expired GTC sells when renewal is enabled and the managed position is still open.
- Are not resubmitted automatically after a sell order is rejected or manually canceled.
- Skip GTC sell submission for legacy fractional managed quantities and keep the managed position active for review.
- Keep the managed position active, blocking new buys, until cumulative managed sell fills close the full buy quantity.
- Block automatic renewal and require manual review if Alpaca reports a partial fill without a valid average fill price, or if cumulative sells exceed the managed buy quantity.
- Use `extended_hours=false` and `time_in_force=gtc`.
- Persist actual sell fill quantity and average price, then include closed managed positions in `alpaca_realized_pnl.csv`.

The raw `sell_signals.csv` report remains a strategy recommendation report. Direct Alpaca submissions from raw sell signals are disabled; live Alpaca exits for positions opened by this workflow are governed by `managed_positions.csv` and the reconciliation step.

Managed position lifecycle:

- Active rows have `closed_at` unset and block new buys for the same symbol.
- Filled managed sells set `sell_status="filled"`, store actual sell fill data, and populate `closed_at`; rows are retained as trade history.
- Closed rows missing actual sell fill data are counted as incomplete and excluded from realized P/L totals.
- Existing Alpaca positions that predate this table are not managed automatically unless imported into `alpaca_managed_positions`.
- Existing unmanaged Alpaca sell orders still protect against repeat buys while they remain open. Deterministic managed exit orders can be linked back to `managed_positions.csv`; unrelated sell orders are not linked automatically.

## Consistency and Concurrency

The risk-free benchmark is forward-filled onto the common asset/signal trading calendar both during
live processing and when SQLite data is rebuilt into an equity curve. This keeps reported strategy
days and benchmark returns consistent when the benchmark source has a missing session.

Strategy-state updates use a SQLite immediate transaction and a persisted generation counter. A
benchmark invalidation and the dependent asset/config updates therefore commit as one serialized
operation even if two workflow processes use the same database.

If every asset workflow fails, the command exits with an error before it writes reports or submits
new buys. Partial asset failures remain visible in the final asset summary while successful assets
continue through the workflow.

## Development

Useful checks:

```bash
uv run python -m compileall main.py leveraged_trader
uv run python -m unittest
uv run leveraged-trader --help
```

Install development tooling with `uv sync --group dev`, then run:

```bash
uv run ruff check .
uv run pytest
```

## Project Layout

See [docs/architecture.md](docs/architecture.md) for the module map and workflow boundaries.
