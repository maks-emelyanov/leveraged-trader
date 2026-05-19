# Leveraged Trader

Leveraged Trader is a research and paper-trading workflow for RSI-based leveraged ETF strategies. It builds a current leveraged ETF universe, optimizes simple buy/sell rules against daily market data, writes recommendation reports, and queues guarded Alpaca paper-trading market orders by default.

This project is intended for research and paper trading. It is not financial advice, and it should not be connected to live trading without additional review, testing, and risk controls.

## Features

- Discovers current long leveraged ETFs from Nasdaq ETF definitions.
- Infers an RSI signal symbol from each leveraged ETF name.
- Downloads daily Yahoo Finance OHLCV data.
- Optimizes RSI buy thresholds and profit-target sell multiples.
- Persists strategy state in SQLite for resumable updates.
- Writes buy and sell recommendation reports.
- Submits Alpaca paper buy and sell market orders by default.
- Guards Alpaca buys against already-held symbols and open buy orders.
- Guards Alpaca sells against missing positions and open sell orders.

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

The CLI also supports `--alpaca-timeout-seconds` for request timeout tuning (default: `30`).

Run an update:

```bash
uv run leveraged-trader --mode update
```

Rebuild all cached state:

```bash
uv run leveraged-trader --mode rebuild
```

Submit paper orders for the current run's recommendations:

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
- `--alpaca-submit-sell-orders / --no-alpaca-submit-sell-orders`: enable or skip sell order submission (default: enabled).
- `--alpaca-api-key-id VALUE`: override `ALPACA_API_KEY_ID`.
- `--alpaca-api-secret-key VALUE`: override `ALPACA_API_SECRET_KEY`.
- `--alpaca-base-url URL`: override `ALPACA_BASE_URL` (defaults to Alpaca paper endpoint).
- `--alpaca-timeout-seconds INT`: Alpaca request timeout in seconds (default: `30`).

## Outputs

CSV reports are written to `outputs/` by default:

- `best_equity_curves.csv`
- `optimization_summary.csv`
- `buy_signals.csv`
- `sell_signals.csv`
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

- Use 10% of current Alpaca paper account cash.
- Use notional orders for Alpaca-fractionable symbols and whole-share `qty` orders otherwise.
- Are skipped if the symbol is already held.
- Are skipped if an open buy order already exists for the symbol.
- Use `extended_hours=false` and `time_in_force=day`.

Sell orders:

- Sell the full currently held Alpaca paper position.
- Are skipped if the symbol is not held.
- Are skipped if an open sell order already exists for the symbol.
- Use `extended_hours=false` and `time_in_force=day`.

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
