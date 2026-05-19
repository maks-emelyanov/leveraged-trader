# Architecture

Leveraged Trader is organized as a small package with IO-heavy boundaries kept separate from strategy and reporting logic.

## Entry Points

- `main.py` is a compatibility wrapper.
- `leveraged_trader.cli:main` is the package entry point.
- The `leveraged-trader` console script is defined in `pyproject.toml`.

## Modules

- `leveraged_trader.config`: constants, dataclasses, and `.env` loading.
- `leveraged_trader.universe`: Nasdaq ETF universe loading and RSI symbol inference.
- `leveraged_trader.market_data`: Yahoo Finance daily OHLCV loading.
- `leveraged_trader.indicators`: indicator calculations such as RSI.
- `leveraged_trader.backtest`: strategy stepping, backtests, and performance summaries.
- `leveraged_trader.storage`: SQLite schema, persisted strategy state, market data, RSI values, and summaries.
- `leveraged_trader.reports`: best-strategy summaries and pending buy/sell recommendation reports.
- `leveraged_trader.alpaca`: Alpaca paper account, position, open-order, and order submission integration.
- `leveraged_trader.workflow`: orchestration for update/rebuild runs, report writing, and default Alpaca submission.
- `leveraged_trader.cli`: command-line argument parsing and top-level configuration.

## Data Flow

1. Load the current leveraged ETF universe from Nasdaq.
2. Infer each leveraged asset's RSI signal symbol.
3. Download daily Yahoo Finance data for the leveraged asset, signal symbol, and risk-free benchmark.
4. Update or rebuild SQLite state for each parameter combination.
5. Summarize the best strategy per asset.
6. Build buy and sell recommendation reports.
7. Write CSV outputs to `outputs/` or the configured output directory.
8. Submit guarded Alpaca paper orders for the current recommendations unless disabled by CLI flags.

`update` mode resumes from persisted SQLite strategy state, while `rebuild` mode recomputes strategy state from scratch.

## Alpaca Boundary

Alpaca submission is isolated in `alpaca.py`. The workflow passes in-memory recommendations to the submitters. CSV files are outputs, not order inputs.

The order guards intentionally check the live Alpaca paper account before submission:

- buys skip already-held symbols and symbols with open buy orders;
- sells skip missing positions and symbols with open sell orders;
- all orders use regular-session day market orders.

## Runtime Configuration

The top-level CLI supports overriding all Alpaca credentials and API settings by flag or environment variable. `load_dotenv` reads local `.env` values only when the corresponding key is not already set in the environment, so exported shell variables still take precedence.
