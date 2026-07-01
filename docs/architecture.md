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

1. Initialize the SQLite schema and reconcile active Alpaca managed positions before refreshing market data. This startup reconciliation can attach or renew managed GTC sells for buys that filled after a previous run.
2. Load the current leveraged ETF/ETN universe from Nasdaq ETF definitions plus best-effort issuer ETF and ETN tables. Each issuer/ETN source's result is saved to `universe_workflow_source_status`; fetch failures and unparseable responses are surfaced as a degraded universe and can be made fatal with `--require-workflow-source-success`, while a successfully parsed source with zero leveraged matches remains healthy.
   Issuer discovery includes ProShares, Direxion, Leverage Shares, GraniteShares, Defiance, AdvisorShares, AXS Investments, Kurv, Innovator, Tuttle Capital, Tradr, REX Shares, KraneShares, Volatility Shares, 21Shares, YieldMax, Tidal, Roundhill, Themes, Simplify, MicroSectors, and UBS ETRACS.
3. Merge universe sources by symbol, infer leverage/direction, and infer each leveraged asset's RSI signal symbol. RSI mappings use curated symbol/name proxies, validated generic ticker inference, and explicit self-RSI fallbacks where no better single-symbol proxy is available. Rows that still need manual RSI mapping review are saved to `universe_rsi_mapping_review` and excluded from the executable workflow. An optional workflow `top_n` limit must be a positive integer; `None` selects every executable discovered asset.
   Inferred underlyings are validated against active listed symbols only when both Nasdaq symbol files load successfully; a partial listing snapshot is recorded for audit but cannot exclude issuer-discovered products.
4. Write audit-only universe source tables for exchange directories, third-party ETF directories, and SEC EDGAR registry review. These audit sources can flag missing leveraged-looking candidates, but they do not override Nasdaq or issuer rows.
5. Download canonical full daily Yahoo Finance histories for the leveraged asset, RSI signal symbol, and risk-free benchmark before each update. Historical corrections or removed sessions invalidate the affected asset state; a benchmark correction or removal invalidates all strategy rollups. The risk-free benchmark is forward-filled onto the asset/signal calendar in both live and SQLite rebuild paths. The current US session is excluded so signals use settled prior-session data.
   If Yahoo skips an asset or signal symbol, retry that symbol with Tradier historical daily data when configured.
6. Update or rebuild SQLite state for each parameter combination.
7. Summarize the best strategy per asset.
8. Build buy and sell recommendation reports, eligible-buy reports, and closed managed Alpaca realized P/L from actual buy and sell fill prices.
9. Submit guarded whole-share Alpaca paper buy orders for the current recommendations unless disabled by CLI flags.
10. If any managed buy was submitted or recovered, run a second Alpaca reconciliation so fast fills can receive their managed GTC limit sells in the same workflow run.
11. Load managed-position state and write CSV outputs to `outputs/` or the configured output directory.

`update` mode resumes from persisted SQLite strategy state, while `rebuild` mode recomputes strategy state from scratch.
Asset-level work is scheduled through `asyncio` with blocking Yahoo Finance, Tradier, Alpaca, and SQLite calls isolated in worker threads. Downloads remain concurrent, while one workflow-wide lock serializes SQLite strategy-state processing: a shared benchmark correction can invalidate every strategy, and same-signal tasks also share canonical RSI history. Each processing unit also uses a SQLite `BEGIN IMMEDIATE` transaction plus a persisted state generation check, so independent processes cannot commit stale dependent state after an invalidation. `--workflow-concurrency` controls the maximum number of assets in flight. Individual asset failures remain visible in the summary, but a run with no completed assets raises a workflow error before reports or buy submission.
Terminal output is routed through `WorkflowReporter`, which keeps live concurrent progress separate from final report tables. Asset tasks return structured run results; the final asset summary is sorted by original workflow index even when tasks complete out of order. Terminal tables intentionally show compact columns and wrap long messages, while CSV outputs retain the complete data. For example, the terminal Best Sharpe table is filtered to strategies with at least two trades and Sharpe of 1.0 or greater, and terminal Alpaca tables use dense display IDs while the CSV files keep raw database and broker identifiers.
A workflow timer is collected around the orchestration run and emitted at the output boundary as a simple elapsed-time footer followed by a divider for appended logs.
The optimization grid stores resumable state and summary rollups for every parameter combination, but `strategy_equity` keeps daily equity rows only for each asset's current best parameter set. This keeps `best_equity_curves.csv` available without writing every grid point's full daily curve to SQLite.
The inner grid loop runs over NumPy arrays through `optimized_backtest.py`, using Numba when available and a compatible Python fallback otherwise. Python code prepares state arrays and persists the compact results; the compiled loop handles the repeated day-by-day strategy simulation.

## Alpaca Boundary

Alpaca submission is isolated in `alpaca.py`. The workflow passes in-memory recommendations to the submitters. CSV files are outputs, not order inputs.

The order guards intentionally check the live Alpaca paper account before submission:

- buys skip already-held symbols, active managed-position symbols, and symbols with open buy or sell orders;
- buys submit only in Alpaca's premarket window when the signal date is the immediately preceding trading session; otherwise they are deferred;
- submitted buys are persisted with the strategy parameters selected at entry time;
- a managed-buy intent is atomically claimed before a broker request; only its claimant can submit the deterministic client order ID. Timeout, duplicate-ID, and server-error ambiguity is recovered by client order ID, while a transient `404` stays blocked for a bounded submission-visibility lease instead of being closed by another process. After expiry, only a closed `submission_not_found` intent without a broker order ID can be atomically reclaimed for a new attempt;
- submitted buys use a dynamic batch cash allocation of `min(number_of_eligible_buy_signals * 0.05, 0.50)` of current Alpaca cash, equally split only among recommendations that pass live preflight and quote checks, as whole-share day limit orders with a configurable protective price buffer. Limit prices use four decimals below `$1` and two at or above `$1`; buys round down to their cap and sells round up to their target;
- filled or partially filled managed buys submit a GTC limit sell at the actual average fill price multiplied by the original sell multiple; later parent-buy fills replace the active sell when its covered quantity or target changes, including after an earlier partial-fill sell completed;
- a second reconciliation pass runs after a buy batch so newly filled buys can receive their managed exit without waiting for the next workflow run;
- managed GTC sells persist the deterministic sell client order ID before broker submission, so ambiguous sell POST outcomes can be recovered by client order ID and matching open `rsi-exit-...` orders can be reattached to the managed row;
- active managed GTC sells are canceled and renewed before Alpaca's aged-order expiration, using the remaining managed quantity and frozen target price. The renewal-cancel intent is persisted before requesting cancellation, so a timeout can still complete after Alpaca later reports the order canceled;
- expired managed GTC sells are resubmitted when renewal is enabled and the managed position is still open;
- rejected, manually canceled, or otherwise inactive managed sells keep the managed position active so later optimizations cannot rebuy the symbol automatically;
- legacy fractional managed quantities do not submit GTC sells automatically and remain active for review;
- managed sell fills are accumulated by Alpaca order; realized P/L uses matched quantity, and a position closes only after cumulative fills exactly cover the full managed buy quantity; incomplete fill metadata and overfills fail closed for review;
- closed managed rows missing actual sell fill data are counted as incomplete and excluded from realized P/L totals;
- unmanaged Alpaca positions or unrelated sell orders are not backfilled automatically, though open sell orders still block new buys for that symbol;
- buy orders use regular-session day orders, and managed limit sell orders use regular-session GTC orders with persisted expiration and renewal metadata.

Strategy sell signals remain report outputs, and direct Alpaca submissions from raw sell signals are disabled. Live exits for positions opened by the workflow are driven by managed-position reconciliation instead of the latest optimized parameter row.

## Runtime Configuration

The top-level CLI supports overriding Alpaca credentials, Tradier fallback settings, and API settings by flag or environment variable. `load_dotenv` reads local `.env` values only when the corresponding key is not already set in the environment, so exported shell variables still take precedence. `ALPACA_BATCH_CASH_FRACTION` is rejected if present because buy sizing is now derived from the current eligible buy count. Use `--no-color` for plain terminal output.
