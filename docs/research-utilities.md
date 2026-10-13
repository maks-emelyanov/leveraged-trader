# Research utilities

These repository tools describe cached simulations and saved evidence. Run the commands from a
checkout after the [README setup](../README.md#setup), using a fresh output directory for each run.
Saved study artifacts and the source trading database are not part of the installed package.

## Trade-activity comparison

[The saved comparison](../research/trade-activity-2026-10-12/) evaluates robust selection versions
1 and 2 on identical histories with the current 49-pair grid. Version 2 adds the bounded reward
for completed trades described in the README. The comparison fixes its weight at 0.5 and annual
scale at 12; neither value was fitted by this script.

The cohort requires 1,260 aligned sessions before 2022. It contains 76 funds, evenly split between
long and inverse products, with 33 signal proxies. The inclusive input cutoff is October 9, 2026;
the latest cached observation is October 8. The two versions select the same parameters and
produce the same acceptance outcome for every asset: 75 have insufficient development evidence,
and TQQQ fails its holdout. TQQQ selects RSI 30 and target 1.02 in both versions. Its minimum
development trade rate is about 1.128 round trips per year, giving a version 2 bonus of about
0.043. No asset passes either selector in this sample.

These are retrospective comparisons of already-observed histories, not independent validation
of the activity preference. The sample cannot establish improved realized activity or returns.
The production full-history Sharpe threshold and other buy guards still apply after selector
acceptance. Aggregate metrics in this study weight assets equally and do not represent an
investable portfolio.

```bash
NUMBA_NUM_THREADS=2 .venv/bin/python research/trade-activity-2026-10-12/reproduce.py \
  --db strategy_state.sqlite \
  --output /tmp/trade-activity-rerun
```

The runner opens the source database with SQLite `mode=ro`, makes a consistent temporary backup,
and verifies that the backup remains unchanged while reading it. It does not fetch market data
or access a broker. It writes:

- `coverage.csv`: included and excluded assets, signal mappings, and input hashes.
- `selections.json` and `selections.csv`: each asset's results under both versions.
- `summary.json`: settings, counts, descriptive metrics, and source/input provenance.

An existing result file prevents a rerun in that output directory. The temporary database is
deleted when input loading finishes. Reproducing the saved results requires the same cached
observations, mappings, and source versions; the database snapshot is not distributed. The runner
records source and input hashes for comparison, but does not enforce equality with the saved study.

## Historical signal cleaning

`scripts/normalize-historical-signals.py` applies one descriptive interquartile-range screen to
previously exported simulated buys and completed round trips. Despite the filename, it preserves
original RSI and price units. It does not optimize parameter values or alter strategy state.

The inputs are CSV files keyed by `asset_symbol`, `signal_symbol`, and entry `date`, with no
duplicate identities. Every completed trade must have a matching buy. Both files need `workflow`
and finite `observed_signal_rsi` values. Completed trades additionally need finite
`gross_price_multiple`, plus `buy_rsi` and `profit_target_multiple` for the descriptive ranges.
Other export columns are preserved. The saved source files provide complete schema examples.

Within each workflow, the script calculates linear-interpolated Q1 and Q3 from the original rows
and flags values strictly outside `Q1 - 1.5 * IQR` through `Q3 + 1.5 * IQR`. It screens RSI for all
buys, and separately screens RSI and gross exit/entry price multiples for completed trades. A
completed row is excluded if either field is flagged. Fences are calculated once: there is no
recursive trimming, rescaling, or separate screening of configured profit targets. All-buy RSI
flags and completed-trade RSI flags use their respective samples, so they can differ.

```bash
.venv/bin/python scripts/normalize-historical-signals.py \
  --buys research/historical-signal-cleaning-2026-10-12/source_buys.csv \
  --closed research/historical-signal-cleaning-2026-10-12/source_completed_trades.csv \
  --out /tmp/historical-signal-cleaning-rerun
```

The output directory must be empty. It receives copies of both source files, their SHA-256 hashes
in `summary.json`, and these derived exports:

- `all_buys_with_rsi_flags.csv`: every buy with its RSI flag.
- `normalized_completed_trades.csv`: retained completed trades, with screening flags.
- `excluded_completed_trades.csv`: removed completed trades and their flags.
- `unresolved_buys.csv`: buys with no matching completed trade in the supplied exports.
- `summary.json`: counts, workflow-specific fences, and retained descriptive ranges.

The [saved export](../research/historical-signal-cleaning-2026-10-12/) contains 1,852 buys and
1,696 completed trades. The screen retains 1,479 completed trades and excludes 217; 156 buys have
no completed match. No RSI values are flagged in the all-buy sample. Of the completed trades,
1,207 long and 272 inverse trades remain.

Tail observations can be valid large gains. Excluding them does not establish data errors, and
completed trades omit unresolved positions. Each trade has equal weight, so frequent traders
contribute more observations. The retained sample cannot establish whole-strategy performance,
an optimal RSI threshold, or an optimal profit target. These exports are separate from the
[grid-family](grid-search-evidence.md) and [grid-resolution](grid-resolution-evidence.md) studies;
those primary analyses retain every valid market outcome.
