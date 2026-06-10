# Leveraged Trader

Leveraged Trader is a research and paper-trading workflow for RSI-based leveraged ETF strategies. It builds a current leveraged ETF universe, optimizes simple buy/sell rules against daily market data, writes recommendation reports, and queues guarded Alpaca paper-trading buy orders by default.

This project is intended for research and paper trading. It is not financial advice, and it should not be connected to live trading without additional review, testing, and risk controls.

## Features

- Discovers current long leveraged ETFs from Nasdaq ETF definitions.
- Infers an RSI signal symbol from each leveraged ETF name.
- Downloads daily Yahoo Finance OHLCV data.
- Optimizes RSI buy thresholds and profit-target sell multiples.
- Processes asset workflows concurrently with async orchestration.
- Persists strategy state in SQLite for resumable updates.
- Writes buy and sell recommendation reports.
- Submits guarded whole-share Alpaca paper buy market orders by default.
- Tracks submitted Alpaca buys as managed live positions in SQLite.
- Submits one-time managed Alpaca GTC limit sells from actual fill price times the original sell multiple.
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
```

The CLI also supports `--alpaca-timeout-seconds` for request timeout tuning (default: `30`) and
`--workflow-concurrency` for asset-level concurrency tuning (default: `4`).

Run an update:

```bash
uv run leveraged-trader --mode update
```

Rebuild all cached state:

```bash
uv run leveraged-trader --mode rebuild
```

Submit paper buys for the current run's recommendations and reconcile managed GTC limit sells:

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
- `--workflow-concurrency INT`: maximum number of assets processed concurrently (default: `4`; use `1` for serial behavior).

## Outputs

CSV reports are written to `outputs/` by default:

- `best_equity_curves.csv`
- `optimization_summary.csv`
- `buy_signals.csv`
- `eligible_buy_signals.csv`
- `sell_signals.csv`
- `managed_positions.csv`
- `alpaca_reconciliation_results.csv`
- `alpaca_order_results.csv`
- `alpaca_sell_order_results.csv`

Use `--output-dir` to choose a different location:

```bash
uv run leveraged-trader --output-dir outputs/dev
```

The SQLite state database defaults to `strategy_state.sqlite`; use `--db` to override it.

## Environment Variables

Supported environment variables:

- `ALPACA_API_KEY_ID`
- `ALPACA_API_SECRET_KEY`
- `ALPACA_BASE_URL`

If your environment has not installed project entry points yet, use the module entry point:

```bash
uv run python -m leveraged_trader --help
```

## Alpaca Safety

Alpaca submission is enabled by default. Use `--no-alpaca-submit-buy-orders` or `--no-alpaca-submit-sell-orders` to skip one or both submission workflows.

Buy orders:

- Use 10% of current Alpaca paper account cash to size an integer whole-share `qty`.
- Are skipped if the 10% cash allocation is below one whole share at the latest price estimate.
- Are skipped if the symbol is already held.
- Are skipped if the symbol already has an active managed position.
- Are skipped if an open buy or sell order already exists for the symbol.
- Use `extended_hours=false` and `time_in_force=day`.
- Are persisted with the buy RSI and sell multiple that were selected when the buy was submitted.

Managed sell orders:

- Are reconciled from persisted managed buy records, not from the latest optimized sell signal.
- Use the actual Alpaca filled average buy price times the original sell multiple.
- Sell the exact filled buy quantity with a one-time GTC limit order.
- Are not resubmitted automatically after a sell order is canceled, rejected, expired, or otherwise inactive.
- Skip GTC sell submission for legacy fractional managed quantities and keep the managed position active for review.
- Keep the managed position active, blocking new buys, until the managed sell is filled.
- Use `extended_hours=false` and `time_in_force=gtc`.

The raw `sell_signals.csv` report remains a strategy recommendation report. Direct Alpaca submissions from raw sell signals are disabled; live Alpaca exits for positions opened by this workflow are governed by `managed_positions.csv` and the reconciliation step.

Managed position lifecycle:

- Active rows have `closed_at` unset and block new buys for the same symbol.
- Filled managed sells set `sell_status="filled"` and populate `closed_at`; rows are retained as trade history.
- Existing Alpaca positions that predate this table are not managed automatically unless imported into `alpaca_managed_positions`.
- Existing unmanaged Alpaca sell orders still protect against repeat buys while they remain open, but they are not linked to `managed_positions.csv`.

## Development

Useful checks:

```bash
python3 -m compileall main.py leveraged_trader
python3 -m unittest discover
uv run leveraged-trader --help
```

Ruff and pytest settings are included in `pyproject.toml` for teams that install those tools:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

## Project Layout

See [docs/architecture.md](docs/architecture.md) for the module map and workflow boundaries.
