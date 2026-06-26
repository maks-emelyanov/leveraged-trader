# Changelog

## Unreleased

- Added SQLite-backed Alpaca managed-position tracking with fill reconciliation.
- Changed live Alpaca exits to managed limit sells anchored to actual fill price and the original sell multiple.
- Changed Alpaca buy submissions to whole-share quantity orders and managed exits to managed GTC limit sells.
- Added managed Alpaca GTC sell renewal before aged-order expiration.
- Added closed managed Alpaca realized P/L reporting from actual fill prices.
- Disabled direct Alpaca submissions from raw sell-signal reports.
- Added async workflow orchestration with configurable asset-level concurrency.
- Added Rich-powered workflow progress and wrapped terminal tables.
- Added Yahoo ticker normalization and concise yfinance skip messages for failed downloads.
- Added Tradier historical data fallback for symbols skipped by Yahoo Finance.
- Changed leveraged ETF universe generation to merge Nasdaq ETF definitions with best-effort issuer tables and validate inferred underlyings against active listed symbols.
- Added additional issuer universe sources, including Leverage Shares, YieldMax, Tidal, Roundhill, Themes, and Simplify.
- Changed optimization storage to persist summary metrics for every grid point but daily equity only for best curves.
- Added a NumPy/Numba-backed optimization loop for faster parameter-grid processing.
- Added active managed-position and open sell order guardrails for buy submissions.
- Added managed-position, eligible-buy, and reconciliation CSV outputs.
- Changed Alpaca buy batch sizing to reserve 5% of cash per eligible buy signal, capped at 50%.
- Removed the obsolete fixed Alpaca batch cash fraction option and now reject the legacy environment variable.
- Expanded README usage docs with a CLI reference and environment variable section.
- Documented Alpaca timeout configuration and option discovery via `--help`.
- Clarified `update` vs `rebuild` behavior and runtime config precedence in architecture docs.

## 0.1.0

- Split the original monolithic script into the `leveraged_trader` package.
- Added current leveraged ETF universe discovery.
- Added resumable SQLite-backed strategy state.
- Added buy and sell recommendation reports.
- Added guarded Alpaca paper buy and sell submission workflows.
- Added local `.env` loading with placeholder credential validation.
- Added output directory support for generated CSV reports.
