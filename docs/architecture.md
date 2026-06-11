# Architecture

Leveraged Trader is organized as a small package with IO-heavy boundaries kept separate from strategy and reporting logic.

## Entry Points

- `main.py` is a compatibility wrapper.
- `leveraged_trader.cli:main` is the package entry point.
- The `leveraged-trader` console script is defined in `pyproject.toml`.

## Modules

- `leveraged_trader.config`: constants, dataclasses, and `.env` loading.
- `leveraged_trader.universe`: multi-source leveraged ETF/ETN universe discovery, leverage/direction parsing, and RSI symbol inference.
- `leveraged_trader.market_data`: Yahoo Finance daily OHLCV loading with Tradier fallback for skipped symbols.
- `leveraged_trader.indicators`: indicator calculations such as RSI.
- `leveraged_trader.backtest`: shared strategy initial state and performance summary calculations.
- `leveraged_trader.storage`: SQLite schema, persisted strategy state, market data, RSI values, summaries, and Alpaca managed-position records.
- `leveraged_trader.reports`: best-strategy summaries and pending buy/sell recommendation reports.
- `leveraged_trader.alpaca`: Alpaca paper account, position, open-order, managed-position reconciliation, and order submission integration.
- `leveraged_trader.output`: terminal progress, section headings, status coloring, and width-aware table rendering.
- `leveraged_trader.workflow`: async orchestration for update/rebuild runs, report writing, and default Alpaca submission.
- `leveraged_trader.cli`: command-line argument parsing and top-level configuration.

## Data Flow

1. Load the current leveraged ETF/ETN universe from Nasdaq ETF definitions plus best-effort issuer ETF and ETN tables.
   Issuer discovery includes ProShares, Direxion, Leverage Shares, GraniteShares, Defiance, AdvisorShares, AXS Investments, Kurv, Innovator, Tuttle Capital, Tradr, REX Shares, KraneShares, Volatility Shares, 21Shares, YieldMax, Tidal, Roundhill, Themes, Simplify, MicroSectors, and UBS ETRACS.
2. Merge universe sources by symbol, infer leverage/direction, and infer each leveraged asset's RSI signal symbol.
   Inferred underlyings are validated against active listed symbols when Nasdaq symbol files are available; otherwise the workflow falls back to name inference.
3. Write audit-only universe source tables for exchange directories, third-party ETF directories, and SEC EDGAR registry review. These audit sources can flag missing leveraged-looking candidates, but they do not override Nasdaq or issuer rows.
4. Download daily Yahoo Finance data for the leveraged asset, signal symbol, and risk-free benchmark.
   If Yahoo skips an asset or signal symbol, retry that symbol with Tradier historical daily data when configured.
5. Update or rebuild SQLite state for each parameter combination.
6. Summarize the best strategy per asset.
7. Build buy and sell recommendation reports.
8. Write CSV outputs to `outputs/` or the configured output directory.
9. Reconcile Alpaca managed positions and submit or renew anchored GTC limit sells for whole-share filled buys unless disabled by CLI flags.
10. Calculate closed managed Alpaca realized P/L from actual buy and sell fill prices.
11. Submit guarded whole-share Alpaca paper buy orders for the current recommendations unless disabled by CLI flags.

`update` mode resumes from persisted SQLite strategy state, while `rebuild` mode recomputes strategy state from scratch.
Asset-level work is scheduled through `asyncio` with blocking Yahoo Finance, Tradier, Alpaca, and SQLite calls isolated in worker threads. Each asset task uses its own SQLite connection with a busy timeout; tasks that share the same RSI signal symbol serialize their SQLite update step to avoid racing shared RSI rows. `--workflow-concurrency` controls the maximum number of assets in flight.
Terminal output is routed through `WorkflowReporter`, which keeps live concurrent progress separate from final report tables. Asset tasks return structured run results; the final asset summary is sorted by original workflow index even when tasks complete out of order. Terminal tables intentionally show compact columns and wrap long messages, while CSV outputs retain the complete data.
The optimization grid stores resumable state and summary rollups for every parameter combination, but `strategy_equity` keeps daily equity rows only for each asset's current best parameter set. This keeps `best_equity_curves.csv` available without writing every grid point's full daily curve to SQLite.
The inner grid loop runs over NumPy arrays through `optimized_backtest.py`, using Numba when available and a compatible Python fallback otherwise. Python code prepares state arrays and persists the compact results; the compiled loop handles the repeated day-by-day strategy simulation.

## Alpaca Boundary

Alpaca submission is isolated in `alpaca.py`. The workflow passes in-memory recommendations to the submitters. CSV files are outputs, not order inputs.

The order guards intentionally check the live Alpaca paper account before submission:

- buys skip already-held symbols, active managed-position symbols, and symbols with open buy or sell orders;
- submitted buys are persisted with the strategy parameters selected at entry time;
- submitted buys use integer whole-share quantities sized from the configured cash allocation;
- filled managed buys submit a GTC limit sell at the actual average fill price multiplied by the original sell multiple;
- active managed GTC sells are canceled and renewed before Alpaca's aged-order expiration, using the original filled quantity and frozen target price;
- expired managed GTC sells are resubmitted when renewal is enabled and the managed position is still open;
- rejected, manually canceled, or otherwise inactive managed sells keep the managed position active so later optimizations cannot rebuy the symbol automatically;
- legacy fractional managed quantities do not submit GTC sells automatically and remain active for review;
- filled managed sells persist actual sell fill quantity and average price, calculate realized P/L, and mark the row closed by setting `closed_at` while retaining the row as history;
- closed managed rows missing actual sell fill data are counted as incomplete and excluded from realized P/L totals;
- unmanaged Alpaca positions or sell orders are not backfilled automatically, though open sell orders still block new buys for that symbol;
- buy orders use regular-session day orders, and managed limit sell orders use regular-session GTC orders with persisted expiration and renewal metadata.

Strategy sell signals remain report outputs, and direct Alpaca submissions from raw sell signals are disabled. Live exits for positions opened by the workflow are driven by managed-position reconciliation instead of the latest optimized parameter row.

## Runtime Configuration

The top-level CLI supports overriding Alpaca credentials, Tradier fallback settings, and API settings by flag or environment variable. `load_dotenv` reads local `.env` values only when the corresponding key is not already set in the environment, so exported shell variables still take precedence. Use `--no-color` for plain terminal output.
