# Changelog

## Unreleased

- Added SQLite-backed Alpaca managed-position tracking with fill reconciliation.
- Changed live Alpaca exits to managed limit sells anchored to actual fill price and the original sell multiple.
- Added active managed-position and open sell order guardrails for buy submissions.
- Added managed-position, eligible-buy, and reconciliation CSV outputs.
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
