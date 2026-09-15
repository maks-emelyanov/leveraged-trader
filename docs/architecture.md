# Architecture

Leveraged Trader is organized as a small package with IO-heavy boundaries kept separate from strategy and reporting logic.

## Entry Points

- `main.py` is a compatibility wrapper.
- `leveraged_trader.__main__` implements `python -m leveraged_trader` by delegating to the CLI.
- `leveraged_trader.cli:main` is the package entry point.
- The `leveraged-trader` console script is defined in `pyproject.toml`.

## Modules

- `leveraged_trader.__init__`: package version metadata.
- `leveraged_trader.config`: constants, dataclasses, and `.env` loading.
- `leveraged_trader.universe`: multi-source leveraged ETF/ETN universe discovery, leverage/direction parsing, and RSI symbol inference.
- `leveraged_trader.market_data`: Yahoo Finance daily OHLCV loading with guarded raw-price Tradier fallback for skipped symbols.
- `leveraged_trader.indicators`: indicator calculations such as RSI.
- `leveraged_trader.backtest`: shared strategy initial state and performance summary calculations.
- `leveraged_trader.benchmark`: overlap-aware workflow phase and total elapsed-time measurement.
- `leveraged_trader.accounting`: shared quantity and notional tolerances for managed-position reconciliation and reporting.
- `leveraged_trader.pricing`: shared Decimal-based Alpaca target-price and tick-rounding semantics.
- `leveraged_trader.optimized_backtest`: validated NumPy/Numba grid and equity-curve simulation kernels.
- `leveraged_trader.storage`: SQLite schema, persisted strategy state, market data, RSI values, summaries, and Alpaca managed-position records.
- `leveraged_trader.runtime_files`: private runtime-directory, SQLite-file, sidecar, and publication identity checks.
- `leveraged_trader.reports`: best-strategy summaries, pending buy recommendations, and latest sell-event reports.
- `leveraged_trader.alpaca`: Alpaca paper account, position, account-order, managed-position reconciliation, and order submission integration.
- `leveraged_trader._http_deadline_worker`: isolated, killable HTTP transport for bounded Alpaca, Tradier, and universe-source requests.
- `leveraged_trader._yfinance_deadline_worker`: isolated, killable Yahoo Finance download and recent-price requests.
- `leveraged_trader.output`: terminal progress, section headings, status coloring, and width-aware table rendering.
- `leveraged_trader.workflow`: async orchestration for update/rebuild runs, report writing, optional Alpaca submission, and lightweight reconciliation-only runs.
- `leveraged_trader.cli`: command-line argument parsing and top-level configuration.

## Data Flow

Before the data flow begins, the public workflow validates mode, persistent database path, paper-only
Alpaca endpoint, and optimization grids without creating filesystem artifacts. It then acquires
nonblocking process locks for both the database and output directory. The output lock uses an internal
`.leveraged-trader.lock` plus deterministic account-scoped path anchors in a verified state directory
resolved from OS account metadata (the passwd home on POSIX and the LocalAppData Known Folder on
Windows). Separate anchors cover the supplied absolute and canonical database/output paths, so
replacing either parent or using a different process environment cannot split mutual
exclusion, and an unwritable or shared local parent cannot disable startup. Generated local lock names
are ignored by Git. The database lock preserves the legacy supplied-path name and also covers the
resolved target of a symlink. Hard-linked SQLite files
are rejected because path-derived SQLite journal/WAL files would not share identity safely. On POSIX,
SQLite connects through the canonical target only after verifying that its parent is owned by the
current user and is not group- or world-writable; database and sidecar identities are checked around
connections and transactions. The top-level lock pins that database inode through nested connections
and shared workers, and every Alpaca order submission or cancellation revalidates it immediately
before the HTTP mutation. Processes sharing the same OS user remain within this filesystem trust
boundary because Python's SQLite binding opens VFS sidecars by pathname.
Order-enabled workflows additionally acquire one fixed private per-user Alpaca paper-account lock,
so separate databases and output directories cannot race the same broker exposure or cash snapshot.

1. Initialize the SQLite schema and, when managed-sell submission is explicitly enabled, reconcile active Alpaca managed positions before refreshing market data. This startup reconciliation can attach or renew managed GTC sells for buys that filled after a previous run. Multi-position reconciliation reuses one strictly validated, paginated account-order snapshot; exact ID lookups remain the fallback for absent orders and replacement links, and an unusable historical snapshot falls back to the validated open-order list before per-order reads.
2. Load the current leveraged ETF/ETN universe from Nasdaq ETF definitions plus best-effort issuer ETF and ETN tables, then split executable products into long and inverse/short workflow groups. The primary Nasdaq feed and every issuer/ETN source are saved to `universe_workflow_source_status`; fetch failures and unparseable responses are surfaced as a degraded universe and can be made fatal with `--require-workflow-source-success`, which is enabled automatically for Alpaca buy submission unless explicitly disabled, while a successfully parsed issuer source with zero leveraged matches remains healthy. An empty or implausibly small Nasdaq ETF table is a parser failure, not an authoritative snapshot.
   Issuer discovery includes ProShares, Direxion, Leverage Shares, GraniteShares, Defiance, AdvisorShares, AXS Investments, Kurv, Innovator, Tuttle Capital, Tradr, REX Shares, KraneShares, Volatility Shares, 21Shares, YieldMax, Tidal, Roundhill, Themes, Simplify, MicroSectors, and UBS ETRACS. REX products come from the official Cboe issuer listing because the issuer site blocks unattended requests, while Tradr uses only its current explicit target/exposure tables so legacy product tables cannot override current leverage.
3. Merge universe sources by symbol, infer leverage/direction, and infer each leveraged asset's RSI signal symbol. RSI mappings use curated symbol/name proxies, validated generic ticker inference, and explicit self-RSI fallbacks where appropriate. Exact-ticker overrides must still match an expected exposure fingerprint in the current product name, protecting against stale metadata and ticker reuse; names matching multiple distinct curated proxies require review. Curated inverse mappings point to an unlevered ETF or spot-market proxy for the product's benchmark. Long-product self-RSI fallbacks remain executable; inverse products without an underlying RSI proxy require manual review because using the product's own RSI would invert the high-RSI entry rule. Review rows are saved to `universe_rsi_mapping_review` and excluded from the executable workflow. Leveraged-looking false positives, such as funds where "ultra-short" describes bond duration, are explicitly excluded, as are products with a changing basket and no stable single RSI proxy. An optional workflow `top_n` limit must be a positive integer; `None` selects every executable discovered asset.
   Usable symbols from each successfully loaded Nasdaq active-listing file may corroborate an inferred underlying, and rows marked `Test Issue=Y` are excluded from that positive evidence. Destructive filtering requires both files plus at least 99.5% coverage of a real-sized primary Nasdaq ETF inventory with no more than five primary symbols absent, and applies only to those absent primary rows, with every exclusion persisted alongside its source and reason. Issuer/ETN discoveries remain eligible when absent from the directories because primary coverage cannot prove an issuer-only inventory complete. A missing, duplicate, or unrecognized `Test Issue` field or a larger cross-source discrepancy makes the snapshot non-authoritative; it is recorded for audit but cannot exclude products.
4. Write audit-only universe source tables for exchange directories, third-party ETF directories, and SEC EDGAR registry review. Sources with product names can flag missing long or inverse leveraged-looking candidates, but they do not override Nasdaq or issuer rows. The SEC exchange registry must expose one unambiguous ticker field and one unambiguous product-name field before it can claim product-name coverage. Symbol-only feeds such as Cboe's listed-products CSV and the SEC mutual-fund ticker registry are explicitly recorded as `loaded_inventory_only`; their rows are counted separately and are not presented as leveraged-product coverage. Dynamic directories without a stable machine-readable response, including the NYSE listings page, remain registered backstops and are not fetched by the static HTML parser. SEC requests require an explicitly configured `SEC_USER_AGENT` containing a truthful identity and monitored contact email; without one, SEC sources are recorded as `skipped_configuration` without making the executable universe degraded. An enabled, configured audit source that fails to fetch, cannot be parsed, or produces no product rows marks the reported universe as degraded and is shown in terminal source-health details. Audit failures remain non-fatal and are intentionally outside `--require-workflow-source-success` because they cannot remove rows from the executable source universe.
5. Download canonical full daily Yahoo Finance histories for the leveraged asset, RSI signal symbol, and risk-free benchmark before each update. Each complete history is compared with its persisted symbol in one bulk read, while SQLite writes are limited to new, changed, or removed sessions. Historical corrections or removed sessions invalidate the affected asset state; a benchmark correction, removal, or newly supplied session inside an already processed horizon invalidates all strategy rollups. A newly published signal tail observation also invalidates every dependent strategy when it falls within an already processed asset horizon, including the bounded post-session window used by weekend signals, so the affected state is replayed instead of merely appending RSI data. The asset's settled sessions define the trading calendar; missing signal sessions do not remove asset sessions, complete-history RSI observations are aligned as-of to that calendar, and the risk-free benchmark is left-joined and forward-filled. An RSI observation may lag an asset session by at most seven calendar days. Only a final Friday may use a following Saturday or Sunday observation from a 24/7 proxy; future weekdays and weekend observations spanning an intervening business day fail closed, and reports enforce the same rule. Boundary removals require matching observations from two distinct top-level runs, so long and short processing cannot confirm one transient truncation twice. The current US session is excluded so signals use settled prior-session data.
   If Yahoo skips an asset or signal symbol, retry that symbol with Tradier historical daily data
   only when automatic adjustment is disabled. Adjusted Yahoo histories fail closed rather than
   mixing them with a provider whose adjustment basis is not guaranteed to match.
   The workflow obtains those complete histories in deterministic Yahoo batches of 32 unique
   symbols. Every extracted frame passes the same per-symbol validation; provider errors, missing or
   ambiguous frames, and timeouts are retried individually through the normal Yahoo/Tradier path.
   If strict validation of a signal's full history fails but its asset history is usable, the
   workflow retries the signal from at least one year before that asset's first settled session.
   This bounded recovery remains private to the exact asset/signal pair in SQLite and cannot become
   another strategy's canonical signal cache. Newly listed self-signaled products that do not yet
   have the RSI period plus one settled observations report `warming_up` and remain non-actionable
   until enough sessions accumulate.
   Tiny provider-to-provider variation in one coherent adjusted-price factor is retained from the
   already authenticated local history when volume is unchanged; an individual OHLC correction,
   material factor change, calendar change, or volume change still invalidates exact compact state.
6. Update or rebuild SQLite state for each parameter combination. Incremental resume requires the expected configuration fingerprint plus authenticated strategy-state and strategy-summary rows for every requested parameter pair. The summary digest covers identity, chronology, raw rollups, centered moments, and derived metrics; legacy, missing, or modified digests force a rebuild, and best-strategy/report readers reject the complete candidate set if any row fails authentication. Derived ranking metrics are also recomputed from the raw rollup before it can resume. Explicit rebuild skips validation of state it will discard. Since full daily equity is retained only for the best pair, any missing or incomplete non-best rollup forces a safe full asset-grid rebuild. A held resume must reconcile its cash and shares to the persisted final close, and summary variance uses persisted Welford mean/M2 accumulators instead of cancellation-prone raw-moment subtraction. RSI uses Wilder's recursive averages after simple-average seeding over the first period; invalid negative cached averages force recomputation. Long products use low-RSI entries (`RSI <= Buy RSI`), while inverse/short products use high-RSI entries (`RSI >= Buy RSI`). A held position models the broker's resting GTC sell limit: the target uses the same decimal multiplication and upward Alpaca tick rounding as live orders, a favorable opening gap fills at the opening price, and an intraday High touch fills at the target. On a buy session the daily-bar model assumes the target becomes active immediately after the opening fill; daily OHLC cannot encode the scheduler's short attachment delay or the intraday ordering around it.
   Resume defaults to trusted local-state verification: the protected database and schema,
   independently versioned strategy-semantics fingerprint, complete grid, authenticated row digests,
   chronology, and state generation are checked once before the asset transaction, whose captured
   generation fence makes the proof stale-safe. Trusted proofs use a single serialized SQLite read
   lane on WSL while provider retries remain concurrent, and reporting reuses the same run's proof.
   `--strategy-state-verification canonical` additionally performs a full-history audit replay.
   Rebuilds use a pristine-state kernel path and select the winner from validated in-memory summaries
   with the shared SQL order before writing the grid and one winning equity curve.
7. Summarize the best strategy per asset.
8. Build side-labeled buy recommendations, eligible-buy reports, and sell-event reports, plus closed managed Alpaca realized P/L from actual buy and sell fill prices. A best strategy whose target exit executed on its latest settled session appears in `sell_signals.csv` even though its simulator state is already flat with no pending sell action. This row records a simulated result and is not a live sell instruction.
9. Submit guarded whole-share Alpaca paper buy orders for the combined current recommendations only when `--alpaca-submit-buy-orders` is explicitly set.
10. If any managed buy was submitted or recovered, run a second Alpaca reconciliation so fast fills can receive their managed GTC limit sells in the same workflow run.
11. Load managed-position state and write CSV outputs to `outputs/` or the configured output directory. Each CSV is written through its original temporary-file descriptor, anchored to that inode, and atomically replaces its destination only after publication identity checks. On POSIX, create/link/replace/restore/unlink operations stay relative to the verified output-directory descriptor, so directory rotation cannot redirect publication or cleanup. A detected pathname substitution restores the prior or intended report, or removes the substituted entry and fails. Related Alpaca CSVs publish a `publishing` checksum manifest before the first replacement and atomically commit the generation plus each file's SHA-256 digest last, so an abrupt interruption leaves a snapshot that readers can reject instead of accepting a mixed generation.

`update` mode resumes from persisted SQLite strategy state, while `rebuild` mode recomputes strategy state from scratch.
`--reconcile-only` acquires the same database/output locks, initializes the schema, reconciles existing
managed positions, refreshes realized P/L and managed-position CSVs, and exits without universe,
market-data, RSI, or optimization work. Buy submission is invalid in this mode; managed sells still
require the explicit `--alpaca-submit-sell-orders` flag.
Asset-level work is scheduled through `asyncio` as a bounded producer/consumer pipeline with blocking Yahoo Finance and Tradier preparation calls isolated in a dedicated download worker pool. The long workflow runs first, followed by the inverse/short workflow; both use the same serialized SQLite strategy consumer. `--workflow-concurrency` controls the fixed pool of asset download workers, which feed a capacity-one prepared-data queue to that consumer; `1` retains fully serial behavior. SQLite strategy processing runs on its own single-thread executor and remains one-at-a-time because a shared benchmark correction can invalidate every strategy. The consumer retains one thread-confined SQLite connection during the asset pipeline but still opens and commits a separate `BEGIN IMMEDIATE` transaction for each asset. Identical cached signal and risk-free history frames are synchronized once per workflow; `PRAGMA data_version` invalidates that run-local cache after any external SQLite commit, and failed transactions never populate it. Yahoo Finance calls remain internally serialized because yfinance exposes process-global error, logger, and cache state. Each call runs in a one-shot isolated process under one absolute deadline covering worker startup, DNS, all internal requests and completion waits, result serialization, and transfer, followed by bounded process cleanup. Parsing, Tradier fallback, and downloads can still overlap the SQLite consumer. Signal and risk-free download caches are populated under narrow locks. Each processing unit also uses a persisted state generation check, so independent processes cannot commit stale dependent state after an invalidation. Alpaca reconciliation and submission run after both asset pipelines through their existing worker-thread boundaries; their Yahoo price lookup uses the same killable deadline boundary. Individual asset failures remain visible in the side-specific summaries, but a run with no completed assets raises a workflow error before reports or buy submission.
Terminal output is routed through `WorkflowReporter`, which keeps live concurrent progress separate from final report tables. The pipeline returns structured run results with workflow identity preserved; long and short ETF/ETN asset summaries are clearly separated and use their table titles instead of a redundant `Workflow` column, while combined buy-signal and Alpaca sections include side attribution where applicable. Signal tables retain the separate RSI observation date even in narrow layouts so weekend or 24/7 proxy observations are not mistaken for the asset's settled-session date. Terminal tables intentionally show compact columns and wrap long messages, while CSV outputs retain the complete data. For example, the terminal Best Sharpe table is filtered to strategies with at least two trades and Sharpe of 1.0 or greater, and terminal Alpaca tables use chronological display IDs that preserve closed-position gaps while the CSV files keep raw database and broker identifiers.
A workflow timer is collected around the orchestration run and emitted at the output boundary as a compact total wall-time footer followed by a divider for appended logs. Internal phase instrumentation remains overlap-aware; `--show-timings` prints the detailed phase durations and market-data work counts immediately before the footer, while normal terminal output omits them.
The optimization grid stores resumable state and summary rollups for every parameter combination, but `strategy_equity` keeps daily equity rows only for each asset's current best parameter set. This keeps `best_equity_curves.csv` available without writing every grid point's full daily curve to SQLite. Resume therefore verifies complete rollup coverage across the requested grid; it rebuilds rather than initializing a missing non-best rollup from an unreconstructable partial history. The retained best curve is also regenerated when its row count, endpoints, or dates no longer match the selected summary and persisted asset calendar. If safe-tail synchronization retains authenticated historical candles while ignoring tiny coherent adjustment-factor drift in an incoming frame, curve append or replacement reads the exact post-synchronization SQLite snapshot so the curve and compact rollups cannot consume different historical values.
The inner grid loop runs over NumPy arrays through `optimized_backtest.py`, using Numba when available and a compatible Python fallback otherwise. Python code prepares state arrays and persists the compact results; the compiled loop handles the repeated day-by-day strategy simulation.

## Alpaca Boundary

Alpaca submission is isolated in `alpaca.py` and is disabled by default. The workflow passes in-memory recommendations to the submitters only after the corresponding explicit CLI opt-in. CSV files are outputs, not order inputs.

The order guards intentionally check the live Alpaca paper account before submission:

- buys skip already-held symbols, active managed-position symbols, and symbols with open buy or sell orders;
- buys submit only after Alpaca's clock supplies a literal closed status and a `next_open` strictly after its timestamp, in the premarket window where the signal date is the immediately preceding trading session and the RSI observation is no more than seven calendar days old; only a final Friday may use a following Saturday or Sunday observation from a 24/7 proxy, while future weekdays and weekends spanning an intervening business day fail before broker access. The final fence anchors the later of the accepted broker timestamp and host time to monotonic elapsed time, reserves the configured request timeout plus five seconds before `next_open`, and enforces one absolute request-write cutoff through DNS, all candidate TCP connections, and TLS. Runtime-state validation is repeated immediately before POST; a local identity failure remains batch-fatal but is proven undispatched and releases its managed generation and budget for a safe retry;
- submitted buys are persisted with the strategy parameters selected at entry time;
- a managed-buy intent, including its immutable workflow, signal symbol, RSI threshold, profit multiple, signal date, original quantity, and limit price, is atomically claimed before a broker request; only its claimant can submit the deterministic client order ID. Timeout, duplicate-ID, and server-error ambiguity is recovered by client order ID, while a transient `404` stays blocked for a bounded submission-visibility lease instead of being closed by another process. After expiry, only a closed `submission_not_found` intent without a broker order ID and with the same strategy economics, quantity, and limit can be atomically reclaimed for a new attempt. Legacy intents missing that baseline fail closed for review;
- recovered, submitted, and reconciled buy payloads fail closed when client/order ID, symbol or asset ID, side, order type, time-in-force, quantity, price, or frozen strategy economics contradicts the persisted intent. `done_for_day`, `stopped`, and `suspended` buys remain active because they can still fill, and partial fills can still receive protective sells. Valid broker replacement chains are followed to their current successor;
- a buy batch raises a workflow error when account setup fails or every attempted preflight/submission fails because of broker or transport availability, while retaining completed per-signal diagnostics;
- submitted buys use a dynamic batch cash allocation of `min(number_of_eligible_buy_signals * 0.05, 0.50)` of current Alpaca cash, equally split only among recommendations that pass live preflight and quote checks, as whole-share day limit orders with a configurable protective price buffer. Limit prices use four decimals below `$1` and two at or above `$1`; buys round down to their cap and sells round up to their target;
- filled or partially filled managed buys submit a GTC limit sell at the actual average fill price multiplied by the original sell multiple; later parent-buy fills replace the active sell when its covered quantity or target changes, including after an earlier partial-fill sell completed;
- a second reconciliation pass runs after a buy batch so newly filled buys can receive their managed exit without waiting for the next workflow run;
- managed GTC sells persist a deterministic client order ID whose durable namespace derives from the account-wide buy client order ID rather than a database-local row ID. Initial submission atomically guards both remaining quantity and target price; ambiguous POST outcomes are recovered only after broker identity and intent-chronology validation, and matching open `rsi-exit-...` orders can be reattached to the managed row. If final pre-submit validation finds an executable historical sell, reconciliation persists a `pending_cancel` lease and cancellation timestamp before requesting broker cancellation. A later pass must prove the historical lineage inactive, the deterministic replacement absent, the account open-order snapshot conflict-free, and the live holding equal to the managed remainder before it may submit. Rows stranded by the former pre-fence metadata quarantine are eligible for this path only when the exact historical-cancellation diagnostic, missing attached order and leases, and a persisted historical sell generation all match; other metadata quarantines remain blocked for review;
- reconciliation fails nonzero, after persisting its result CSVs, whenever any current active managed position cannot confirm required protection or safe fill accounting. This includes broker authentication, availability, or transport failures, malformed or rejected protective-order responses, live-position quantity drift, and broker identity or order-metadata mismatches. The only exception is an exact-quantity paper-account holding whose deterministic sell is absent and whose stable Alpaca asset identity is confirmed inactive and non-tradable: it remains active under `broker_inactive`, blocks duplicate buys, is rechecked for automatic recovery, and is accounted for in the dedicated broker-retained inactive-holdings report without requiring removal from Alpaca. Its monitoring row remains in the complete reconciliation audit but is excluded from the sell-order outcome report;
- broker-replaced managed sells are followed through validated `replaced_by`/`replaces` lineage rather than leaving the managed row attached to an obsolete order;
- active Alpaca positions are keyed to Alpaca's stable asset ID before reconciliation. If Alpaca reports a new ticker for the same asset, broker order, client-order, and asset responses are identity-checked before the managed row changes. The complete migration batch is transactional and rejects duplicate active asset IDs or current tickers. All observed ticker aliases continue blocking duplicate buys, old-ticker open sells can be recovered by asset ID, existing order lineage is retained, and new buy or replacement-sell requests use the validated stable asset ID rather than a ticker snapshot. Alpaca may display an unreplaced protective order under its historical ticker; coverage is determined by asset ID and remaining quantity rather than symbol text alone. Unchanged persisted asset/ticker pairs skip historical-order lookups. A `symbol_migrated` reconciliation row is emitted only in the workflow run that applies the migration;
- active managed GTC sells are canceled and renewed before Alpaca's aged-order expiration, using the remaining managed quantity and frozen target price. The renewal-cancel intent is persisted before requesting cancellation, so a timeout can still complete after Alpaca later reports the order canceled;
- expired managed GTC sells are resubmitted when renewal is enabled and the managed position is still open;
- renewal and resubmission are reconciliation actions, not an independent scheduler, so recurring workflow runs must keep managed sell submission enabled;
- rejected, manually canceled, or otherwise inactive managed sells keep the managed position active so later optimizations cannot rebuy the symbol automatically;
- legacy fractional managed quantities do not submit GTC sells automatically and remain active for review;
- managed sell fills are accumulated by Alpaca order; realized P/L uses matched quantity, and a position closes only after cumulative fills exactly cover the full managed buy quantity. Fresh and replacement protective orders require the broker's live holding to equal the managed remaining quantity, so splits, corporate actions, or discretionary quantity changes fail closed instead of submitting incorrect coverage. Incomplete fill metadata, fill regressions, and overfills also fail closed for review. Managed rows that predate the inverse/Short workflow and its persisted workflow column are migrated to the only workflow that existed when they were opened, `Long`, so their realized P/L is not reported under an unknown side;
- closed managed rows missing actual sell fill data are counted as incomplete and excluded from realized P/L totals;
- unmanaged Alpaca positions or unrelated sell orders are not backfilled automatically, though open sell orders still block new buys for that symbol;
- buy orders use regular-session day orders, and managed limit sell orders use regular-session GTC orders with persisted expiration and renewal metadata.

Strategy sell rows remain report outputs and include latest-session target exits already executed by
the daily-bar simulation. They are not actionable live sell submissions, and direct Alpaca
submissions from raw sell rows are disabled. Live exits for positions opened by the workflow are
driven by managed-position reconciliation instead of the latest optimized parameter row.

## Runtime Configuration

The top-level CLI supports API behavior settings by flag or environment variable, but Alpaca and Tradier credentials are accepted only from environment variables or a private `.env` file so they do not appear in process listings or shell history. Alpaca buy and managed-sell submission default to off. Alpaca access is hard-restricted to `https://paper-api.alpaca.markets`; the validation allows one trailing slash but rejects live endpoints, custom paths, ports, queries, fragments, and embedded user information. When Tradier fallback and a bearer token are enabled, the base URL is similarly restricted to an official HTTPS `api.tradier.com` or `sandbox.tradier.com` API root, optionally followed by `/v1`. `load_dotenv` reads local `.env` values only when the corresponding key is not already set in the environment, so exported shell variables still take precedence. A configured `SEC_USER_AGENT` must contain both a meaningful non-placeholder operator/application identity and a monitored contact email; a bare email and reserved example/test domains are rejected. The SEC identity is host-scoped to SEC requests, while other universe hosts use the generic project identity. Universe redirects are followed only when the normalized scheme, host, and effective port remain unchanged; cross-origin redirects are rejected before a follow-up request. Each accepted hop bypasses proxies, validates that every resolved address is public, and pins that DNS snapshot to the socket connection without replacing the URL hostname used for HTTPS SNI and certificate verification. `ALPACA_BATCH_CASH_FRACTION` is rejected on full workflow runs because buy sizing is now derived from the current eligible buy count; reconciliation-only mode ignores it because that path cannot submit buys. Use `--no-color` for plain terminal output.

## Scheduled Operation

The cron installer adds a single host-timezone-independent once-per-minute entry through
an authenticated copy of `scripts/cron/run-scheduled-clean-environment`. The installer stores a
content-addressed bootstrap below the account's private
`~/.local/state/leveraged-trader/cron-runtime` directory. After `/usr/bin/env -i`, a fixed root-owned
system Bash captures that file, verifies its install-time SHA-256 digest, and evaluates only the
authenticated bytes. This keeps the complete entry below a conservative 1,000-byte cron line limit
while ensuring a checkout path containing `=` is a script argument rather than another `env`
assignment. The bootstrap pins the `runtime-security` digest and the checkout origin; its launcher
then revalidates the checkout files and external tools and executes a validated in-memory snapshot of
`scripts/cron/run-scheduled` through the validated absolute Bash and `uv` paths. Updating the clean
launcher or `runtime-security` requires reinstalling the cron entry. The project's isolated Python
first snapshots the Unix epoch and UTC minute without site initialization. A coarse UTC gate skips times that cannot be scheduled
under either Eastern standard or daylight time before traversing the virtual environment, but only
while the validated lockfile pins the audited `tzdata` release; an updated lock disables the shortcut
until its bounds are reviewed. Candidate minutes receive the complete import-tree validation; the same
Python then resets `ZoneInfo` search
paths and loads `America/New_York` from pinned `tzdata` for the exact host-timezone-independent
decision. Exact non-due minutes stop before the environment check. Due minutes perform a non-mutating
`uv sync --locked --check` explicitly scoped to this checkout and `.venv`, after clearing inherited uv
project, config-file, and dependency-group overrides; missing or stale environments fail with a
manual `uv sync --locked` instruction before the command can run. Sampling precedes the potentially
slow checks, so an 08:45 invocation remains due if preflight finishes during 08:46. At 08:45 ET on weekdays the
gate requires successful executable-universe sources and runs the full workflow with both Alpaca
submission flags. From 09:30 through 16:00 ET it runs `--reconcile-only` with managed-sell submission
every minute. Other times and weekends are no-ops. Tests can inject the otherwise dynamic
clock with `LEVERAGED_TRADER_SCHEDULE_NOW='<ISO weekday> <HH:MM>'` and replace the command path with
`LEVERAGED_TRADER_RUNNER`; the installed command explicitly clears both hooks plus inherited
`BASH_ENV`, `ENV`, `TZDIR`, and `PYTHONTZPATH` overrides before production execution. It selects
`/bin/sh` only for the managed cron command, restores the prior effective shell for subsequent jobs,
neutralizes inherited `SHELLOPTS` and `BASHOPTS` before that outer shell starts, restores their prior
effective values afterward, and applies a minimal environment to the command. Clock and environment
preflight occur before the execution/log lock, and non-due invocations exit without acquiring it, so
a slow 08:44 preflight cannot block the unique 08:45 invocation. A due invocation locks afterward;
the production runner repeats the environment check under that lock before side effects. Preflight
failures attempt a locked log append and always reach standard error. A compatible file-descriptor
`flock` protects each rotation and
complete append; systems without util-linux's conflict-exit-code capability use a
PID-and-process-start-identity directory lock that atomically
recovers abandoned or PID-reused owners. Lock contention skips a due invocation after preflight; lock
subsystem and unverifiable legacy-lock errors fail visibly. The scheduler rotates a cron log of at
least 10 MiB to one `.1` backup and creates scheduled runtime files with an owner-only umask. Path
ancestry is validated before either installer or scheduler creates entries: symbolic-link components, `..`
components, ASCII control characters, and untrusted-owned ancestors are rejected, and group- or
world-writable parents require sticky semantics with a trusted child owner. Directory-designating log
spellings ending in `/` or `/.` are rejected before normalization. Subsequent log and lock operations
use the exact validated absolute directory. Cron command paths use POSIX single-quote encoding so the
managed `/bin/sh` entry preserves spaces, quotes, percent signs, and non-ASCII bytes. The runner
repeats the non-mutating `uv sync --locked --check` under the execution lock, fails with a manual
`uv sync --locked` instruction if the environment changed after the clock check, then directly execs
`.venv/bin/python -I -m leveraged_trader`. Isolated Python ignores inherited `PYTHONPATH`, user-site
packages, and a `sitecustomize` supplied through either, while the direct exec keeps the published
fallback-lock PID alive for all workflow and broker side effects instead of leaving the trader beneath
a killable `uv run` parent. The managed crontab boundary also clears executable-loader settings,
proxy and CA overrides, TLS key logging, and OpenSSL configuration, module, and engine paths before
the outer shell starts; the command removes them from its environment and the block restores prior
assignments for later jobs.
The installer aborts without writing if the existing crontab cannot be read or if its managed markers
are orphaned, reversed, nested, or duplicated. A private per-user installer lock serializes cooperating
installers across checkouts. Its default location is below `~/.local/state/leveraged-trader`, avoiding
predictable first-claim entries in shared temporary directories. Repeated crontab snapshot checks
abort if a non-cooperating editor changes the table before replacement.

POSIX runtime hardening is also applied outside cron: SQLite uses a canonical path in an owned parent
that is not group- or world-writable. The database and any WAL, shared-memory, or journal sidecars are
restricted to mode `0600` and revalidated for type, ownership, link count, and identity around use;
new database inodes are hardened before atomic publication and only after existing sidecars pass
validation, avoiding rollback through a replaceable canonical pathname. Empty single-link named
stages orphaned before publication are retired after a grace period through descriptor-relative
inode checks. The staging filename namespace is reserved and cannot be selected directly or through
a database symlink, preventing orphan collection from mistaking a configured database for a stage.
Sidecar symlinks, hardlinks,
and detected substitutions fail closed and are reported without pathname cleanup, preventing a
check/unlink race from deleting a concurrent replacement. Report publication similarly
verifies the exact inode written through the temporary descriptor and confines cleanup to the verified
directory descriptor. Credential-bearing `.env` files
must be private regular files with one link and are read through the descriptor that was validated.
