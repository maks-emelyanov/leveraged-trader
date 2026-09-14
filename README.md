# Leveraged Trader

Leveraged Trader is a research and paper-trading workflow for RSI-based leveraged ETF/ETN strategies. It builds current long and inverse leveraged product universes, optimizes simple buy/sell rules against daily market data, writes recommendation reports, and can submit guarded Alpaca paper-trading orders when explicitly enabled.

This project is intended for research and paper trading. It is not financial advice. Alpaca clients
enforce the exact paper endpoint and reject live or custom trading endpoints.

Licensed under the MIT License. See [LICENSE](LICENSE).

## Features

- Discovers current long and inverse leveraged ETFs/ETNs from Nasdaq ETF definitions plus best-effort issuer and ETN pages; persisted source health rejects empty or implausibly small primary Nasdaq snapshots and distinguishes issuer fetch/parser failures from healthy zero-match pages. Complete active-listing snapshots exclude inactive primary Nasdaq rows and Nasdaq test issues, while issuer-only products remain eligible because absence from those directories does not prove that a newly discovered product is inactive.
- Writes audit-only source checks for exchange directories, third-party ETF directories, and SEC EDGAR registry review, including long and inverse leveraged-looking products missing from the merged universe; symbol-only inventories are labeled and counted separately from product-name coverage.
- Infers an RSI signal symbol from each leveraged ETF name, with curated proxy mappings,
  explicit long-product self-RSI fallbacks, and a review table for unresolved mappings. Inverse
  products without an underlying RSI proxy are held for review because applying the inverse entry
  rule to the product's own RSI would reverse the intended signal.
- Downloads daily Yahoo Finance OHLCV data, with optional raw-price Tradier fallback for skipped
  symbols when automatic price adjustment is disabled.
- Computes SMA-seeded Wilder RSI and optimizes low-RSI long entries, high-RSI inverse-product entries, and shared profit-target sell multiples.
- Models each profit target as a resting, tick-rounded GTC limit: favorable opening gaps fill at the open, and intraday High touches fill at the limit.
  On an entry session, the daily-bar model assumes the managed target becomes active immediately
  after the opening fill; OHLC data cannot represent the scheduler's seconds-long attachment delay.
- Uses a NumPy/Numba-backed optimization loop for the parameter grid.
- Uses bounded async download workers feeding serialized SQLite strategy-state updates.
- Renders width-aware terminal progress and tables with semantic status coloring.
- Persists strategy state in SQLite for resumable updates, with transactional cross-process invalidation safety.
- Prevents overlapping runs that share a database or output directory with process-level file locks, and serializes order-enabled runs for the local user's Alpaca paper account even when their paths differ.
- Writes buy recommendations and sell-event strategy reports; latest-session simulated target exits
  appear in `sell_signals.csv` for research and audit, not as live sell instructions.
- Publishes each CSV atomically so readers never observe a partially written file, and commits related
  Alpaca CSVs with a checksum manifest so readers can reject interrupted mixed generations.
- Submits guarded, budget-capped whole-share Alpaca paper buy limit orders only with an explicit opt-in flag.
- Atomically claims a durable Alpaca buy intent before submission, preventing concurrent workers from submitting or closing the same client order ID.
- Submits and renews managed Alpaca GTC limit sells from actual fill price times the original sell multiple, including partial buy fills.
- Guards Alpaca buys against already-held symbols, active managed positions, and open buy or sell orders.

## Strategy and Backtest Contract

The default CLI uses a fixed research configuration. These strategy settings are not currently CLI
options:

| Setting | Default CLI behavior |
| --- | --- |
| Starting capital | `$100,000` of simulated cash per parameter combination |
| RSI | 14-session SMA-seeded Wilder RSI |
| Long-product entry grid | RSI thresholds `20` through `50`, inclusive, in increments of `1`; enter when RSI is at or below the threshold |
| Inverse-product entry grid | RSI thresholds `50` through `80`, inclusive, in increments of `1`; enter when the underlying proxy RSI is at or above the threshold |
| Profit-target grid | Return multiples `1.10` through `5.00`, inclusive, in increments of `0.10` |
| Simulated trading costs | `1` basis point of fee plus `2` basis points of slippage on each buy and sell notional |
| Position sizing | Commit all available simulated cash, including fractional shares; this differs from whole-share Alpaca paper-order sizing |
| Risk-free series | Yahoo Finance `^IRX`, interpreted as an annual percentage yield and converted to a 252-session daily return |

An RSI entry signal on a settled asset session schedules a simulated purchase at the next asset
session's opening price. The position is marked at each close. Its tick-rounded target is treated as a
resting limit immediately after entry: a favorable opening gap fills at the open, otherwise an
intraday High at or above the limit fills at the limit. The current US session is excluded from the
research history. The backtest's fractional, all-cash execution model is deliberately different from
the live paper-order budget, whole-share sizing, quote checks, and fill behavior.

Metrics use the asset's settled-session calendar and 252-session annualization. The `^IRX` daily
risk-free return is `(1 + annual_yield / 100) ** (1 / 252) - 1` and is forward-filled onto that
calendar. Each asset's winning grid row is selected by non-null Sharpe, then total return, then CAGR.
Ties prefer the more selective RSI threshold (lower for the Long workflow and higher for the Short
workflow), followed by the smaller profit target.

## Setup

The automated schedule requires a Unix-like system with `bash` and per-user `crontab` support.
Run all setup commands from the repository root.

### 1. Get the project

Clone or download the repository, then enter its directory:

```bash
cd /path/to/project
```

The checkout can live anywhere. The cron setup derives its paths from the repository location.

### 2. Install uv

Install `uv` on Linux or macOS using its official standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell if requested by the installer, then verify that `uv` is available:

```bash
uv --version
```

Other supported installation methods are documented in the
[official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### 3. Install Python and project dependencies

Synchronize the locked environment:

```bash
uv sync --locked
```

`uv` installs a compatible Python version when necessary and creates the local `.venv` environment.
The virtual environment does not need to be activated when commands are run through `uv run`.

### 4. Configure API credentials

Copy the tracked example instead of renaming or editing it, then restrict access to the personal
configuration file:

```bash
cp .env.example .env
chmod 600 .env
```

Edit `.env` and replace the placeholder Alpaca values with credentials from an Alpaca **paper**
trading account:

```env
ALPACA_API_KEY_ID=your_alpaca_paper_api_key_id
ALPACA_API_SECRET_KEY=your_alpaca_paper_api_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_BUY_LIMIT_BUFFER_BPS=500
ALPACA_GTC_SELL_RENEWAL_ENABLED=true
ALPACA_GTC_SELL_RENEWAL_DAYS_BEFORE_EXPIRATION=7
TRADIER_ACCESS_TOKEN=your_tradier_access_token
SEC_USER_AGENT=leveraged-trader/0.1 your-name your-email@example.com
```

`ALPACA_BASE_URL` must remain exactly `https://paper-api.alpaca.markets` (one trailing slash is also
accepted). The application rejects live endpoints, alternate ports, paths, queries, and other custom
trading URLs whenever an Alpaca client is created.

`TRADIER_ACCESS_TOKEN` is optional and is used for market-data fallback. Tradier's historical price
basis is not guaranteed to match Yahoo's adjusted history, so fallback is eligible only on runs that
explicitly use `--no-auto-adjust`; adjusted runs fail closed instead of mixing providers. Remove the
token line or disable fallback with `--no-tradier-fallback` if no Tradier account is configured.
When fallback and a token are enabled, `TRADIER_BASE_URL` must use the official HTTPS API root at
`api.tradier.com` or `sandbox.tradier.com`, optionally followed by `/v1`; custom hosts are not
supported.
`.env` is ignored by Git; do not commit it or share its contents.
On POSIX systems the application refuses to read `.env` when group or other users have any access;
restore owner-only access with `chmod 600 .env` before running it.
Alpaca and Tradier credentials are accepted only through the environment or this private `.env`
file. They are intentionally not accepted as command-line values, which would expose them in shell
history and process listings.

`SEC_USER_AGENT` identifies audit-only SEC EDGAR requests. Set it to a truthful application/operator
identity and monitored contact email address. If it is omitted, lacks a contact email, or still
contains the example placeholder, SEC audit sources are skipped and their configuration status is
recorded without degrading the executable workflow universe. Non-SEC universe sources use a generic
project User-Agent and never receive the operator contact value. Redirects are followed only when
their normalized scheme, host, and effective port match the source URL; cross-origin redirects are
rejected. Each permitted hop bypasses proxies, rejects any DNS answer containing a non-public
address, and pins the approved answer set to the actual connection while retaining normal HTTPS
hostname verification.

### 5. Validate the installation safely

Confirm that the CLI loads, then perform a report-only run. Alpaca buy and managed-sell submission
are both disabled by default:

```bash
uv run leveraged-trader --help
uv run leveraged-trader
```

Review the generated reports and configuration before explicitly enabling paper-order submission:

```bash
uv run leveraged-trader --alpaca-submit-buy-orders --alpaca-submit-sell-orders
```

### 6. Install the automated schedule

The automated schedule is optional and requires `cron`. Check whether `crontab` is already
available:

```bash
command -v crontab
```

If it is missing, install and start cron using the commands for the system:

```bash
# Debian or Ubuntu
sudo apt update
sudo apt install cron
sudo systemctl enable --now cron

# Fedora or RHEL
sudo dnf install cronie
sudo systemctl enable --now crond
```

On macOS, `crontab` is normally preinstalled, although macOS may request permission for cron to
access files in protected locations. Native Windows Task Scheduler is not supported by these
scripts. Under WSL, cron must be installed and started, WSL must remain running when the jobs are
due, and service management availability depends on the WSL configuration.

Confirm that the applicable service is running:

```bash
# Debian or Ubuntu
systemctl status cron

# Fedora or RHEL
systemctl status crond
```

Install the project's managed entries into the current user's crontab:

```bash
./scripts/cron/install-crontab
```

The installer preserves unrelated entries and adds one once-per-minute cron entry. The entry is
kept below the conservative 1,000-byte cron line limit. It starts with `/usr/bin/env -i`; a fixed,
root-owned system Bash then reads a content-addressed bootstrap from the account's private
`~/.local/state/leveraged-trader/cron-runtime` directory, verifies its SHA-256 digest, and evaluates
the captured bytes. The bootstrap contains the install-time clean launcher and binds its checkout
origin, validated Bash and SHA-256 paths, and the `runtime-security` digest; the launcher also
validates the installed `uv` path before use. This fixed command boundary keeps checkout paths
containing `=` from being interpreted as `env` assignments.
Concurrent leveraged-trader installers for the same user are serialized across checkouts, and the installer
aborts if the crontab changes after its initial snapshot instead of overwriting the newer entries.
`run-scheduled` first uses the project's isolated Python without site initialization to snapshot the
Unix epoch and UTC minute. A coarse UTC gate skips weekends and times that cannot map to any scheduled
`America/New_York` minute under either Eastern standard or daylight time, without recursively scanning
the virtual environment or running `uv sync --check`. That shortcut is enabled only while the validated
lockfile pins the audited `tzdata` release; changing the locked release safely falls through to the exact
timezone conversion until its offset bounds are reviewed. Candidate minutes receive the complete import-tree
validation before the same project Python loads pinned `tzdata` and makes the exact
`America/New_York` decision. Exact non-due minutes stop there; due minutes then perform the non-mutating
`uv sync --locked --check` explicitly scoped to this project and its `.venv`, with inherited uv project,
config-file, and dependency-group overrides ignored. The scheduler therefore does not depend on the host
timezone, system zoneinfo files, or implementation-specific `CRON_TZ` behavior. A missing or stale locked
environment fails visibly before a due command can run. Sampling first means an 8:45 invocation remains
due even when validation finishes during 8:46. The gate runs:

- 8:45 a.m. Monday-Friday: require successful executable-universe sources, then run the full
  workflow with Alpaca paper buy and managed sell submission. The authenticated clock snapshot also
  supplies an absolute 9:20 a.m. Eastern analytics deadline; expiry rolls back the active asset and
  prevents buy submission.
- Every minute from 9:30 a.m. through 4:00 p.m. Monday-Friday: run the lightweight managed-position
  reconciliation path with sell submission enabled and buy submission disabled.

Candidate-minute clock and locked-environment preflight run before the nonblocking execution/log lock, and a
non-due invocation exits without acquiring that lock. Thus a slow 8:44 preflight cannot occupy the
unique 8:45 execution slot. A due invocation acquires the lock after preflight, and the production
runner repeats the locked-environment check under the lock before side effects. Preflight failures
attempt to acquire the same lock long enough to append to `outputs/cron.log` and are always emitted
on standard error. Lock contention skips the due invocation. The scheduler rotates
`outputs/cron.log` to `outputs/cron.log.1` before an append when the current log reaches 10 MiB. The
single backup is replaced on the next rotation, bounding retained cron logs while preserving the most
recent prior segment. A compatible file-descriptor `flock` protects rotation and the complete append;
systems without util-linux's conflict-exit-code capability use a PID-and-process-start-identity lock
with stale recovery. On that fallback path,
the scheduler ultimately execs `.venv/bin/python -I -m leveraged_trader` so the published lock-owner
PID remains the process performing workflow and broker side effects. Isolated mode ignores inherited
Python environment settings and the user site, so a `sitecustomize` supplied through either cannot
alter scheduled trading code. A due run repeats the
read-only locked-environment check while holding that execution lock, then directly execs the trader;
it never puts broker work beneath a killable `uv run` parent. Lock facility or metadata errors fail
visibly. The scheduler creates runtime files with owner-only permissions only after validating every
directory ancestor; symbolic-link, `..`, and ASCII-control components, untrusted owners, and
nonsticky writable parents
fail before creation, while trusted entries beneath sticky shared roots such as `/tmp` remain
supported. A configured log path must name a file rather than end in `/` or `/.`. The installer
serializes crontab replacement with a lock below the user's private
`~/.local/state/leveraged-trader` namespace; the installer never relies on a first-claimed `/tmp`
pathname. Managed entries temporarily select `/bin/sh` for cron's outer command and restore the prior effective shell
immediately afterward, so later jobs are unaffected. The command applies its minimal PATH locally,
clears `BASH_ENV`, `ENV`, the direct-invocation clock and runner test hooks, inherited `TZDIR` and
`PYTHONTZPATH`, inherited Bash startup-option state, uv routing/group overrides, executable-loader and
proxy/CA settings, TLS key logging, and OpenSSL configuration, module, and engine paths. The clean
launcher authenticates `runtime-security`, revalidates its checkout origin and external tools, then
opens, validates, and captures `run-scheduled` before invoking that snapshot with the installer's
validated absolute Bash path. Changes to the clean launcher or `runtime-security` therefore require
rerunning `install-crontab`; a trusted scheduler update is picked up on the next invocation. Prior
crontab assignments are restored after the managed job. POSIX shell quoting protects paths containing spaces, quotes, percent
signs, or non-ASCII characters.

Verify the installed entries and inspect the shared log:

```bash
crontab -l
tail -F outputs/cron.log
```

The installer creates the output directory but not `cron.log`; the log first appears when a due run
or due-run preflight writes to it. `tail -F` waits for that first creation and follows later log
rotation. If the local `tail` does not support `-F`, wait for the file to appear and then use
`tail -f outputs/cron.log`.

The installer is idempotent: run it again after every project update, moving the checkout or the `uv`
executable, or changing the managed schedule. Reinstallation publishes a current authenticated
`scripts/cron/run-scheduled-clean-environment` launcher snapshot and pins the current
`scripts/cron/runtime-security` digest; without it, a launcher update remains unapplied and a helper
update makes scheduled runs fail closed on a digest mismatch. The installer also records the detected
`uv` path in the managed crontab block, and the wrapper resolves the repository relative to its own
location. The recorded `uv` path is an absolute external executable, even when the interactive shell
has a same-named function or a relative PATH entry.

To operate without cron, skip this step and run the commands in the recommended operating schedule
manually.

#### Disable or remove the automated schedule

Removing the managed crontab block prevents future scheduled runs, including future automatic paper
buy and managed-sell requests. It does not stop a workflow process that is already running.

Open the current user's crontab:

```bash
crontab -e
```

Delete the complete block beginning with `# BEGIN leveraged-trader managed schedule` and ending with
`# END leveraged-trader managed schedule`, including both marker lines, then save and exit. Do not
remove unrelated entries. Verify that the managed block is gone:

```bash
crontab -l
```

This leaves the checkout, SQLite database, reports, and inactive authenticated bootstrap files in
place. Run `./scripts/cron/install-crontab` again to restore the schedule. If a workflow was already
running when the block was removed, let it finish or review that process separately; editing the
crontab affects only future launches.

### 7. Updating an existing installation

After pulling any project changes, resynchronize the locked environment:

```bash
uv sync --locked
```

If you installed the managed cron schedule, always reinstall its managed block after every project
update so its authenticated `scripts/cron/run-scheduled-clean-environment` launcher snapshot and
pinned security-helper digest from `scripts/cron/runtime-security` are current:

```bash
./scripts/cron/install-crontab
```

The CLI also supports `--alpaca-timeout-seconds` for request timeout tuning (default: `30`),
`--alpaca-gtc-sell-renewal-days-before-expiration` for managed GTC sell renewal timing (default: `7`),
`--tradier-timeout-seconds` for market-data fallback timeout tuning (default: `30`),
`--workflow-concurrency` for market-data worker tuning (default: `4`), and `--no-color`
for plain terminal output. The Tradier fallback flag is enabled by default, but fallback requests
require both a non-placeholder Tradier token and `--no-auto-adjust`.

Run an update:

```bash
uv run leveraged-trader --mode update
```

Rebuild all cached state:

```bash
uv run leveraged-trader --mode rebuild
```

Run the paper-trading workflow during the premarket window to submit prior-session signals for that
day's open and reconcile managed GTC limit sells:

```bash
uv run leveraged-trader --alpaca-submit-buy-orders --alpaca-submit-sell-orders
```

Generate reports without submitting paper orders:

```bash
uv run leveraged-trader
```

Show all options:

```bash
uv run leveraged-trader --help
```

## Recommended Operating Schedule

This cadence describes the mechanics of the workflow; it is not investment advice. The workflow
intentionally excludes the current US daily bar, so live buy submissions are designed for the
premarket after the signal day has settled.

- Premarket before the regular open, for example 8:00-9:20 a.m. ET: run
  `uv run leveraged-trader --alpaca-submit-buy-orders --alpaca-submit-sell-orders`.
  This reconciles existing managed positions first, refreshes settled histories, submits eligible
  prior-session buy signals as regular-session day limit buys, then reconciles again if any managed
  buy was submitted or recovered.
- During the regular session, run
  `uv run leveraged-trader --reconcile-only --alpaca-submit-sell-orders` to attach or renew managed
  GTC limit sells without downloading universes, refreshing market data, or placing new buy orders.
  The installed scheduler does this every minute from 9:30 a.m. through 4:00 p.m. ET so a newly
  filled buy receives protection promptly.
- If a limit buy never fills, no sell is submitted. A later reconciliation records the terminal buy
  status and closes the managed intent without a position.
- Extra reconciliation runs are intended to be idempotent: deterministic client order IDs, active
  managed-position checks, and live open-order checks protect against duplicate managed buys or sells.

The underlying `scripts/cron/run-leveraged-trader` wrapper can also be called directly. It accepts
normal CLI arguments, runs the non-mutating `uv sync --locked --check`, and then directly execs
`.venv/bin/python -I -m leveraged_trader`. Isolated mode ignores inherited Python environment
settings and the user site, including a `sitecustomize` supplied through either. Direct execution
preserves the scheduler's fallback-lock PID through the complete trading command. If the locked
environment is missing or stale, the wrapper fails with an instruction to run `uv sync --locked`;
scheduled operation never modifies the environment:

```bash
./scripts/cron/run-leveraged-trader --reconcile-only --alpaca-submit-sell-orders
```

## CLI Reference

Common options:

- `--mode {update,rebuild}`: resume from SQLite state (`update`) or recompute from scratch (`rebuild`).
- `--strategy-state-verification {trusted,canonical}`: authenticate protected local state, the
  strategy fingerprint, full grid, row digests, chronology, and generation before resume (`trusted`,
  the default), or additionally replay canonical history for an explicit audit (`canonical`). Trusted
  verification preserves already authenticated OHLC values across tiny coherent Yahoo adjustment-factor
  drift when volume and the trading calendar are unchanged; real candle corrections still rebuild.
- `--reconcile-only`: reconcile existing managed Alpaca positions without universe downloads or
  strategy work; requires `--alpaca-submit-sell-orders` and cannot be combined with buy submission.
- `--db PATH`: persistent SQLite state file path (default: `strategy_state.sqlite`); empty paths,
  `:memory:`, hard-linked database files, and database parents writable by another OS user are
  rejected.
- `--output-dir DIR`: output directory for generated CSV files (default: `outputs`); on POSIX it
  must be owned by the current user and not be group- or world-writable.
- `--alpaca-submit-buy-orders / --no-alpaca-submit-buy-orders`: enable or skip buy order submission
  (default: disabled).
- `--alpaca-submit-sell-orders / --no-alpaca-submit-sell-orders`: enable or skip managed limit sell
  reconciliation/submission (default: disabled).
- `--alpaca-base-url URL`: override `ALPACA_BASE_URL`; Alpaca access is restricted to the exact paper
  endpoint `https://paper-api.alpaca.markets`.
- `--alpaca-timeout-seconds INT`: Alpaca request timeout in seconds (default: `30`).
- `--alpaca-buy-limit-buffer-bps FLOAT`: price buffer for whole-share day buy limits in basis points (default: `500`).
- `--alpaca-gtc-sell-renewal / --no-alpaca-gtc-sell-renewal`: renew managed Alpaca GTC sells before expiration.
- `--alpaca-gtc-sell-renewal-days-before-expiration INT`: renewal window for managed GTC sells (default: `7`).
- `--tradier-fallback / --no-tradier-fallback`: enable or skip Tradier fallback for Yahoo-skipped symbols.
- `--auto-adjust / --no-auto-adjust`: use adjusted Yahoo OHLC data (default: enabled); Tradier
  fallback is eligible only with `--no-auto-adjust` so histories never mix adjustment bases.
- `--tradier-base-url URL`: override `TRADIER_BASE_URL` (defaults to Tradier production `/v1`). With
  fallback and a bearer token enabled, the URL must be an official HTTPS `api.tradier.com` or
  `sandbox.tradier.com` API root, optionally followed by `/v1`; custom hosts are rejected.
- `--tradier-timeout-seconds INT`: Tradier request timeout in seconds (default: `30`).
- `--workflow-concurrency INT`: maximum concurrent asset download workers; SQLite strategy updates
  remain serialized (default: `4`; use `1` for fully serial behavior).
- `--require-workflow-source-success / --no-require-workflow-source-success`: fail a universe run after recording source health if the primary Nasdaq feed, an issuer/ETN source, or an active-listing source failed; a successfully parsed issuer zero-match page remains healthy (default: enabled with Alpaca buy submission and disabled otherwise). Use the negative form explicitly to allow paper buys from a degraded universe.
- `--no-color`: disable colored terminal output.
- `--show-timings`: show overlap-aware workflow phase timings and market-data work counts.

## Outputs

CSV reports are written to `outputs/` by default:

- `best_equity_curves.csv`: the complete retained equity curve for each asset's winning grid row,
  with Long/Short-prefixed curve names.
- `optimization_summary.csv`: one winning strategy row per successfully processed asset, including
  its parameters and performance metrics. It is not a dump of every grid combination.
- `buy_signals.csv`: winning strategies that are flat with a buy pending for the next open, have more
  than one executed simulated trade, have Sharpe of at least `1.0`, and have a sufficiently fresh RSI
  observation.
- `eligible_buy_signals.csv`: `buy_signals.csv` after excluding symbols already represented by an
  active persisted managed position. “Eligible” is a local pre-broker classification: live holdings,
  open orders, quotes, cash, market-clock checks, and concurrent state can still make Alpaca skip a
  row.
- `sell_signals.csv`: winning-strategy sell-event rows, including targets executed on the latest
  settled session. These are research results, not live order instructions.
- `managed_positions.csv`: persisted managed Alpaca buy and protective-sell lifecycle state.
- `alpaca_inactive_holdings.csv`: open holdings that Alpaca reports as inactive and non-tradable,
  retained at the paper broker with quantity, estimated remaining buy cost, frozen target, and
  last-check accounting. These amounts are lifecycle accounting, not realized P/L or a current
  market valuation.
- `alpaca_realized_pnl.csv`: realized P/L summaries derived from complete managed-position fills.
- `alpaca_reconciliation_results.csv`: per-position results from the latest managed-position
  reconciliation.
- `alpaca_order_results.csv`: per-signal Alpaca paper-buy preflight and submission outcomes.
- `alpaca_sell_order_results.csv`: managed protective-sell submission, recovery, renewal, and
  cancellation outcomes.
- `alpaca_snapshot_manifest.csv`: the committed generation and SHA-256 digest for every broker CSV
  covered by the current snapshot kind.

The signal CSVs are reports and are never replayed as order queues. Order-enabled workflows use the
authenticated in-memory report generation and repeat all applicable broker checks immediately before
submission.

The Alpaca manifest is written with `Status=publishing` before any covered broker CSV changes and
atomically replaced with `Status=committed` only after every listed SHA-256 digest is available.
Programmatic readers should load broker reports through the race-safe snapshot API instead of
validating and then reopening the replaceable CSV paths:

```python
from leveraged_trader.workflow import load_alpaca_snapshot

output_dir = "outputs"
snapshot = load_alpaca_snapshot(output_dir)
managed_positions = snapshot.read_csv("managed_positions.csv")
```

The returned in-memory files all belong to `snapshot.generation`, even if another workflow publishes
after the load; `snapshot.snapshot_kind` and `snapshot.filenames` describe the manifest scope. A
missing, publishing, changed, or checksum-mismatched snapshot is rejected.
Snapshot schema version 2 adds `alpaca_inactive_holdings.csv` to every broker-state generation;
consumers should use `snapshot.filenames` rather than assuming the older fixed file set.
`validate_alpaca_snapshot(output_dir)` remains available as an integrity check of the current paths,
but its result must not be used to authorize later direct opens because publication can begin between
the validation and those opens.

`sell_signals.csv` includes a best strategy's target exit when that exit executed on its latest
settled asset session, even though the simulator is already flat and has no pending sell action.
It is a strategy result report, not an order queue; managed Alpaca exits are reconciled separately
from persisted managed positions.

Use `--output-dir` to choose a different location:

```bash
uv run leveraged-trader --output-dir outputs/dev
```

The SQLite state database defaults to `strategy_state.sqlite`; use `--db` to override it. Workflow
runs connect through the canonical database path, restrict the database and SQLite WAL,
shared-memory, and journal sidecars to owner read/write permissions on POSIX systems, and create an
adjacent `<database>.lock` file. A missing database is hardened on an unpublished or private staging
inode and linked into its canonical pathname only after the parent and existing sidecars pass
validation, so rejected preparation never has to unlink that caller-visible pathname. Old empty
single-link staging inodes left by an interruption before publication are collected after a grace
period using descriptor-relative identity checks. The canonical
parent must be owned by the current user and must not
be writable by group or other users. Sidecar symlinks and hardlinks are rejected, and database and
sidecar identities are revalidated around connections and transactions. Rejected sidecar pathnames
are left untouched because POSIX has no portable conditional-unlink operation that could remove a
checked inode without racing a concurrent replacement. A top-level run pins the
database inode across every nested connection and worker, and rechecks that identity immediately
before an Alpaca order submission or cancellation. Processes running under the same OS user remain
inside the SQLite filesystem trust boundary. Lock files for conventional
`.sqlite`, `.sqlite3`, and `.db` names are ignored by Git; a custom database suffix may require a
local ignore rule. The database must be a persistent file and must not have hard links. Symlink
aliases are supported and acquire both the supplied-path lock and a lock for the resolved database
target. Managed
Alpaca rows persist `alpaca_asset_id`, while `alpaca_symbol_aliases` retains every ticker observed for
that asset so historical symbols remain protected after a rename. Universe generation also persists
`nasdaq_etf_universe`, `universe_audit_rows`,
`universe_audit_missing_candidates`, `universe_audit_source_status`,
`universe_workflow_source_status`, `universe_active_listing_source_status`, and
`universe_rsi_mapping_review` tables for source and RSI-mapping review. When both Nasdaq active-listing
files load successfully and cover at least 99.5% of the real-sized primary ETF inventory with no more
than five primary symbols absent, that authoritative snapshot filters those absent primary Nasdaq rows.
Issuer/ETN discoveries are retained even when absent from the directories,
because primary coverage cannot establish the completeness of issuer-only products. Excluded primary rows
and their source/reason are retained in `universe_inactive_discovered_products`. A partial snapshot is
audit-only and cannot exclude products.
Symbol-only Cboe and SEC mutual-fund
feeds are recorded as inventory-only and are not counted as product-name leverage coverage. Leveraged workflow rows whose RSI
symbol cannot be mapped confidently are excluded from the executable workflow and saved to
`universe_rsi_mapping_review`; curated proxy mappings and long-product self-RSI fallbacks remain
executable and are annotated in `nasdaq_etf_universe`. Exact-ticker overrides must also match the
expected exposure in the current product name, and names matching multiple distinct curated proxies
are sent to review. Inverse-product self-RSI fallbacks are instead
marked for review and excluded because the high-RSI inverse entry rule requires an underlying proxy.
Curated inverse mappings use an unlevered ETF or spot-market proxy for the product's benchmark rather
than the inverse product's own RSI. Products are excluded when leveraged-looking wording is not actual
inverse exposure (for example, "ultra-short" bond-duration funds), or when a changing basket means no
stable single RSI proxy exists.
A workflow discovery, active listing, or enabled audit-source failure leaves the run marked as
degraded in terminal output. A workflow source that parses successfully but contains zero leveraged
matches remains healthy. Conflicting leverage classifications for duplicate symbols are excluded and
mark the source as a parser failure; unambiguous rows remain available in a non-strict degraded run.
An enabled audit directory that produces no parseable product rows is treated as a parser failure
because an empty exchange or registry directory is not a credible successful snapshot. Audit sources
remain coverage checks and do not contribute executable rows. Use
`--require-workflow-source-success` when a partial executable universe caused by a workflow discovery or
active-listing failure is not acceptable; Alpaca buy submission enables it automatically unless
`--no-require-workflow-source-success` is explicitly supplied. The option does not make audit-only
source failures fatal.

Terminal output is intentionally compact: concurrent asset work is shown as aggregate progress, with
long and short ETF/ETN workflow results reported separately before combined buy and Alpaca sections.
Each asset run summary is identified by its Long or Short table title and therefore omits a redundant
`Workflow` column. Signal tables always show the separate RSI observation date, including in narrow
layouts, so a weekend or 24/7 proxy observation is not mistaken for the asset session date.
The terminal Best Sharpe table shows only strategies with at least two executed trades and Sharpe of
1.0 or greater; `optimization_summary.csv` retains the full per-asset summary. CSV files retain full
order IDs and detail, include a `Workflow` column where side attribution applies, and
`best_equity_curves.csv` uses side-prefixed curve names. Terminal Alpaca
tables show chronological display IDs that preserve closed-position gaps, plus the most useful fields
with wrapped messages. A final workflow footer reports total elapsed time followed by a divider for
appended logs. Redirected or cron-driven non-terminal output defaults to a 156-column layout so log
tables stay readable, while interactive terminal output uses the terminal's current width.
Detailed phase timings and market-data work counts are hidden by default; pass `--show-timings` to
include them immediately before the final workflow footer.

To tune download concurrency locally, compare equivalent update runs against copies of the same
SQLite state with `--workflow-concurrency 1`, `2`, `4`, and `8`; disable Alpaca submissions during
the comparison and use the final total elapsed time as the result. Yahoo Finance requests remain
internally serialized for correctness. Each request runs in a one-shot isolated process under a
30-second absolute deadline that includes worker startup, DNS, yfinance's internal requests and
completion wait, and result transfer, followed by bounded process cleanup. Values above the default
`4` therefore mainly help when Tradier fallback or local parsing is significant and can otherwise
add memory use and provider pressure.

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
- `SEC_USER_AGENT` (truthful identity/contact for SEC EDGAR audit requests)

`ALPACA_BATCH_CASH_FRACTION` is intentionally no longer supported. Buy sizing is dynamic, so remove
that legacy key from `.env`. Full workflow runs fail fast when it is present instead of silently
ignoring it; `--reconcile-only` ignores it because that mode cannot submit buys or perform buy sizing.

If your environment has not installed project entry points yet, use the module entry point:

```bash
uv run python -m leveraged_trader --help
```

## Alpaca Safety

Alpaca buy and managed-sell submission are disabled by default. Enable them independently with
`--alpaca-submit-buy-orders` and `--alpaca-submit-sell-orders`; a deliberate full paper-trading run
uses both flags.

Buy orders:

- Reserve `min(number_of_eligible_buy_signals * 0.05, 0.50)` of current Alpaca paper account cash across all submitted buys.
- Split the dynamic batch budget equally across recommendations that pass all live preflight checks (including a quote) and size integer whole-share quantities.
- Use regular-session day limit orders with a configurable price buffer; a gap beyond the limit safely leaves the order unfilled.
- Serialize limit prices at Alpaca's valid tick: four decimals below `$1` and two at or above `$1`; buy caps round down and sell targets round up.
- Are skipped if the allocated batch budget is below one whole share at the protected limit price.
- Are skipped if the symbol is already held.
- Are skipped if the symbol already has an active managed position.
- Are skipped if an open buy or sell order already exists for the symbol.
- Are submitted only during Alpaca's premarket window, and only when the signal date is the immediately preceding trading session; the broker clock must report a literal closed status and a future `next_open`, while intraday, after-close, stale, and malformed-clock signals fail closed. The final fence anchors the later of the accepted broker timestamp and host time to a monotonic deadline, reserves the configured Alpaca request timeout plus five seconds before `next_open`, and applies one absolute request-write cutoff across DNS resolution, every candidate connection, and TLS setup. Runtime-state identity is revalidated immediately before the broker POST; a failure remains a systemic batch error but is recorded as proven undispatched so its managed claim and budget can be retried safely.
- Require an RSI observation no more than seven calendar days before the asset signal date. Only a
  final Friday may use a following Saturday or Sunday observation from a 24/7 proxy; future weekdays
  and weekend observations spanning an intervening business day fail before any broker call.
- Use `extended_hours=false` and `time_in_force=day`.
- Are persisted with the buy RSI and sell multiple that were selected when the buy was submitted.
- Atomically claim an intent, including its immutable workflow, signal symbol, RSI threshold, profit
  multiple, signal date, original quantity, and limit price, before the request is sent; only that
  claimant may submit the deterministic client order ID. A timeout, duplicate-ID response, or other
  ambiguous broker result is recovered by client order ID when possible. A transient broker `404`
  remains blocked during a short visibility lease before the intent can be closed, preventing another
  workflow from orphaning an in-flight accepted order. Once that lease expires, only a closed
  `submission_not_found` record with no broker order ID and the same frozen economics, quantity, and
  limit may be atomically reclaimed for a retry attempt; rejected and other failed submissions remain
  final. Legacy intents without that immutable baseline fail closed for review.
- Validate recovered, submitted, and reconciled broker payloads against the intended client/order ID,
  symbol or asset ID, side, type, time-in-force, quantity, and limit price. Contradictory identity
  fails closed, and valid Alpaca replacement chains are followed to their current successor.
- Submit each new buy against the stable Alpaca asset ID validated during preflight, so a ticker
  rename or reassignment between the final exposure checks and the broker request cannot redirect it.
- Keep `done_for_day`, `stopped`, and `suspended` buys managed because they can still fill; any
  confirmed partial fill remains eligible for a protective sell. A system-wide broker/preflight or
  transport failure makes the command fail after retaining its per-signal diagnostic results.

Managed sell orders:

- Are reconciled from persisted managed buy records, not from the latest optimized sell signal.
- Use the actual Alpaca filled average buy price times the original sell multiple.
- Create a protective GTC sell for confirmed whole shares even while the parent buy remains partially filled, and replace it if later buy fills change the covered quantity or average-fill target—even if the prior partial-fill sell already completed.
- Sell the remaining managed quantity with a GTC limit order; cumulative partial fills remain active until the full buy quantity is closed.
- Require the live Alpaca holding to exactly match the managed remaining quantity before submitting a fresh or replacement protective order, so splits, corporate actions, and discretionary quantity changes fail closed instead of creating incorrect coverage.
- Persist a deterministic, buy-specific sell client order ID before broker submission. Its durable
  namespace is derived from the account-wide buy client order ID rather than a database-local row
  number. Ambiguous sell POST outcomes are recovered only after identity and intent chronology
  validation, and matching open `rsi-exit-...` orders can be reattached to the managed row.
- Follow a broker-replaced sell through its validated `replaced_by`/`replaces` chain instead of
  treating the obsolete order ID as current.
- Reconcile immediately after each submitted buy batch, so fills can receive their managed sell in the same workflow run.
- Renew active GTC sells before Alpaca's aged-order expiration, using the remaining managed quantity and frozen target price. The renewal-cancel intent is persisted before requesting cancellation, so a timeout can still be completed after Alpaca later reports the order canceled.
- Track managed holdings by Alpaca asset ID across ticker changes. Before changing a managed ticker, the workflow validates attached broker/client-order identity and Alpaca's asset response, rejects asset or ticker collisions, and applies the complete migration batch transactionally. Existing broker order IDs remain attached to their original lineage, old-ticker open sells can be recovered by asset ID, every observed ticker alias blocks duplicate buys, and replacement exits are submitted by validated asset ID rather than a ticker snapshot. Alpaca can continue displaying an existing protective order under its pre-rename ticker even though the current position uses the new ticker; matching asset IDs establish that the order protects the renamed position. Unchanged positions avoid historical-order lookups. A `symbol_migrated` row appears in reconciliation output only on the run that applies the change.
- Resubmit expired GTC sells when renewal is enabled and the managed position is still open.
- Require recurring runs with managed sell submission enabled for renewal and resubmission to occur; persisted state alone does not schedule broker requests.
- Are not resubmitted automatically after a sell order is rejected or manually canceled.
- Move an exact-quantity Alpaca paper position into the broker-retained inactive-holdings lane when
  both its deterministic sell is absent and Alpaca identifies the held asset as inactive and
  non-tradable. The active managed row continues to block new buys and is rechecked on recurring
  runs; it is reported in `alpaca_inactive_holdings.csv` instead of as a sell-order outcome. If the
  asset becomes tradable again, the workflow automatically resumes managed-sell submission. Its
  monitoring event remains in the complete reconciliation audit, but it is excluded from
  `alpaca_sell_order_results.csv`. The holding does not need to be removed from Alpaca.
- Skip GTC sell submission for legacy fractional managed quantities and keep the managed position active for review.
- Keep the managed position active, blocking new buys, until cumulative managed sell fills close the full buy quantity.
- Block automatic renewal and require manual review if Alpaca reports a partial fill without a valid average fill price, if observed fills regress, or if cumulative sells exceed the managed buy quantity.
- Use `extended_hours=false` and `time_in_force=gtc`.
- Persist actual sell fill quantity and average price, then include closed managed positions in `alpaca_realized_pnl.csv`.
- Fail the command whenever any current active managed position cannot confirm required protection
  or safe fill accounting—including broker errors, malformed or rejected protective responses,
  live-quantity drift, and identity or order-metadata mismatches—after preserving reconciliation and
  sell diagnostics in CSV. The narrowly verified inactive/non-tradable paper-position quarantine above
  is the only exception because Alpaca cannot accept an executable order for it.

The raw `sell_signals.csv` report includes latest-session simulated target exits for strategy review.
Those rows describe exits the daily-bar simulation already executed; they are not actionable live
sell submissions. Direct Alpaca submissions from raw sell signals are disabled, and live Alpaca exits
for positions opened by this workflow are governed by `managed_positions.csv` and reconciliation.

Managed position lifecycle:

- Active rows have `closed_at` unset and block new buys for the current symbol and every ticker alias recorded for the same Alpaca asset ID.
- Filled managed sells set `sell_status="filled"`, store actual sell fill data, and populate `closed_at`; rows are retained as trade history.
- Closed rows missing actual sell fill data are counted as incomplete and excluded from realized P/L totals.
- Existing Alpaca positions that predate this table are not managed automatically unless imported into `alpaca_managed_positions`.
- Existing unmanaged Alpaca sell orders still protect against repeat buys while they remain open. Deterministic managed exit orders can be linked back to `managed_positions.csv`; unrelated sell orders are not linked automatically.
- To audit live coverage, compare each open Alpaca position with an active managed row and an open sell order by `alpaca_asset_id`, then verify that the order's unfilled quantity covers the held quantity. Symbol-only checks are insufficient across ticker changes.

## Consistency and Concurrency

The leveraged asset's settled sessions define the strategy trading calendar both during live
processing and when SQLite data is rebuilt into an equity curve. Missing signal sessions never
remove asset sessions; RSI is calculated from the signal's complete canonical history and aligned
as-of to asset sessions. An aligned RSI observation may lag its asset session by at most seven
calendar days. A final Friday may additionally use a settled Saturday or Sunday observation from a
24/7 proxy for the following open; later weekdays and weekend observations spanning an intervening
business day fail closed instead of driving stale asset history. Reports apply the same rule. The risk-free
benchmark is left-joined and forward-filled on that asset calendar, keeping reported strategy days
and benchmark returns consistent across source gaps.

Canonical market histories are compared with their persisted symbols in one bulk read, and SQLite
writes are limited to new, changed, or removed sessions. Cached signal and risk-free histories are
synchronized once per workflow while their downloaded frames remain unchanged. Boundary-removal
confirmation records at most one observation per top-level run, so the long and short workflows
cannot confirm the same transient truncation twice. A commit from another database connection
discards that run-local synchronization cache before the next asset is processed. If a newly
published signal tail observation falls inside an already processed asset horizon, every strategy
that depends on that signal is invalidated and replayed; appending the RSI row alone is not treated
as a safe incremental resume. When trusted verification retains authenticated OHLC values across
tiny coherent adjustment-factor drift, both compact strategy state and the retained winning equity
curve continue from the exact post-synchronization SQLite history rather than mixing in ignored
provider values.

An update resumes a parameter grid only when its configuration fingerprint, authenticated
strategy-state row, and authenticated complete non-null summary rollup exist for every requested
parameter combination. The summary digest covers its identity, chronology, raw moments, and derived
metrics; legacy or modified rows without a matching digest force a rebuild rather than being silently
backfilled. Because daily
equity rows are retained only for the best combination, a missing non-best rollup cannot be derived
safely and forces a full asset-grid rebuild. Held state is checked against the asset's persisted last
close, retained best-equity dates must exactly match the summary and asset calendar, and volatility
rollups use resumable centered moments to remain stable for nearly constant return series. Derived
ranking metrics are recomputed from those rollups before resume. Best-strategy selection and report
readers authenticate every candidate summary, while pending-action reads authenticate the compact
strategy-state digest.

Strategy-state updates use a thread-confined SQLite connection with a separate immediate transaction
and persisted generation check for every asset. A benchmark invalidation and the dependent
asset/config updates therefore commit as one serialized operation even if two workflow processes use
the same database. Failed transactions do not populate the shared-history synchronization cache.

Each top-level workflow also acquires nonblocking process locks for its database and output directory.
Runs sharing either resource fail fast instead of overlapping; this includes direct callers of the
public asynchronous API. Database locks retain the legacy `<database>.lock` name and additionally use
the resolved target for symlink aliases. Deterministic account-scoped anchors cover both the supplied
and canonical database and output paths, while the output directory retains its internal
`.leveraged-trader.lock`. The anchors live in a per-user state directory resolved from OS account
metadata (the passwd home on POSIX and the LocalAppData Known Folder on Windows), so renaming or
replacing a database/output parent, using a different process environment, or making the local parent
read-only cannot split mutual exclusion. An order-enabled workflow also holds a fixed
per-user Alpaca paper-account anchor, preventing separate strategy databases from racing the same
broker exposure and cash snapshot. Generated local lock names are ignored by Git.
CSVs are written to same-directory temporary files and atomically replaced, so a reader sees either
the previous complete file or the new complete file. On POSIX, creation, publication, restoration,
and cleanup remain relative to one verified output-directory descriptor; replacing the directory
pathname cannot redirect cleanup into the replacement. Related broker CSV batches first publish an
invalidating manifest and commit their generation and per-file SHA-256 digests last, making a process
or host interruption between individual atomic replacements detectable by readers.

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
uv run pytest -q
uv build
```

The checked-in GitHub Actions workflow runs those lint, test, and build checks against the lockfile
and audits the installed dependency set with `pip-audit` on every push and pull request.

## Project Layout

See [docs/architecture.md](docs/architecture.md) for the module map and workflow boundaries.
